use anyhow::{anyhow, bail, Context, Result};
use ash::{vk, Entry};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    fs,
    path::{Path, PathBuf},
};

const ELEMENTS: usize = 256;
const FILTER_ELEMENTS: usize = 9;
const BATCHNORM_PARAM_ELEMENTS: usize = 4;
const F32_SIZE: usize = std::mem::size_of::<f32>();

#[derive(Debug)]
struct Args {
    device: String,
    conv2d_spv: PathBuf,
    batchnorm_spv: PathBuf,
    relu_spv: PathBuf,
    softmax_spv: PathBuf,
    fixture_manifest: PathBuf,
    report: PathBuf,
    dump_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    epsilon: f32,
    inputs: FixtureInputs,
    edges: FixtureEdges,
}

#[derive(Debug, Deserialize)]
struct FixtureInputs {
    conv2d_input: NpyRef,
    conv2d_filter: NpyRef,
    batchnorm_params: NpyRef,
}

#[derive(Debug, Deserialize)]
struct FixtureEdges {
    conv2d_output: NpyRef,
    batchnorm_output: NpyRef,
    relu_output: NpyRef,
    softmax_output: NpyRef,
}

#[derive(Debug, Deserialize)]
struct NpyRef {
    path: PathBuf,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct EdgeReport {
    tensor_sha256: String,
    oracle_sha256: String,
    mismatch_count: usize,
    max_abs_error: f32,
    max_ulp_error: u32,
    dump_path: String,
}

#[derive(Debug)]
struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

#[derive(Debug)]
struct Pipeline {
    shader: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let manifest: FixtureManifest =
        serde_json::from_slice(&fs::read(&args.fixture_manifest).with_context(|| {
            format!(
                "reading fixture manifest {}",
                args.fixture_manifest.display()
            )
        })?)?;

    let conv2d_input = read_npy_f32(&manifest.inputs.conv2d_input.path)?;
    let conv2d_filter = read_npy_f32(&manifest.inputs.conv2d_filter.path)?;
    let batchnorm_params = read_npy_f32(&manifest.inputs.batchnorm_params.path)?;
    let oracle_conv2d = read_npy_f32(&manifest.edges.conv2d_output.path)?;
    let oracle_batchnorm = read_npy_f32(&manifest.edges.batchnorm_output.path)?;
    let oracle_relu = read_npy_f32(&manifest.edges.relu_output.path)?;
    let oracle_softmax = read_npy_f32(&manifest.edges.softmax_output.path)?;
    if conv2d_input.len() != ELEMENTS
        || conv2d_filter.len() != FILTER_ELEMENTS
        || batchnorm_params.len() != BATCHNORM_PARAM_ELEMENTS
        || oracle_conv2d.len() != ELEMENTS
        || oracle_batchnorm.len() != ELEMENTS
        || oracle_relu.len() != ELEMENTS
        || oracle_softmax.len() != ELEMENTS
    {
        bail!("unexpected fixture sizes for 256-element Conv2D -> BatchNorm -> ReLU -> Softmax harness");
    }

    let conv2d_spv = fs::read(&args.conv2d_spv)
        .with_context(|| format!("reading {}", args.conv2d_spv.display()))?;
    let batchnorm_spv = fs::read(&args.batchnorm_spv)
        .with_context(|| format!("reading {}", args.batchnorm_spv.display()))?;
    let relu_spv =
        fs::read(&args.relu_spv).with_context(|| format!("reading {}", args.relu_spv.display()))?;
    let softmax_spv = fs::read(&args.softmax_spv)
        .with_context(|| format!("reading {}", args.softmax_spv.display()))?;
    let conv2d_spv_sha256 = sha256_bytes(&conv2d_spv);
    let batchnorm_spv_sha256 = sha256_bytes(&batchnorm_spv);
    let relu_spv_sha256 = sha256_bytes(&relu_spv);
    let softmax_spv_sha256 = sha256_bytes(&softmax_spv);

    fs::create_dir_all(&args.dump_dir)
        .with_context(|| format!("creating {}", args.dump_dir.display()))?;

    let run = unsafe {
        run_vulkan(
            &args.device,
            &conv2d_spv,
            &batchnorm_spv,
            &relu_spv,
            &softmax_spv,
            &conv2d_input,
            &conv2d_filter,
            &batchnorm_params,
        )?
    };

    let conv2d_dump = args.dump_dir.join("edge_conv2d_output.raw.f32");
    let batchnorm_dump = args.dump_dir.join("edge_batchnorm_output.raw.f32");
    let relu_dump = args.dump_dir.join("edge_relu_output.raw.f32");
    let softmax_dump = args.dump_dir.join("edge_softmax_output.raw.f32");
    fs::write(&conv2d_dump, f32_slice_bytes(&run.conv2d_output))
        .with_context(|| format!("writing {}", conv2d_dump.display()))?;
    fs::write(&batchnorm_dump, f32_slice_bytes(&run.batchnorm_output))
        .with_context(|| format!("writing {}", batchnorm_dump.display()))?;
    fs::write(&relu_dump, f32_slice_bytes(&run.relu_output))
        .with_context(|| format!("writing {}", relu_dump.display()))?;
    fs::write(&softmax_dump, f32_slice_bytes(&run.softmax_output))
        .with_context(|| format!("writing {}", softmax_dump.display()))?;

    let conv2d_edge = compare_edge(
        &run.conv2d_output,
        &oracle_conv2d,
        manifest.epsilon,
        conv2d_dump,
    )?;
    let batchnorm_edge = compare_edge(
        &run.batchnorm_output,
        &oracle_batchnorm,
        manifest.epsilon,
        batchnorm_dump,
    )?;
    let relu_edge = compare_edge(&run.relu_output, &oracle_relu, manifest.epsilon, relu_dump)?;
    let softmax_edge = compare_edge(
        &run.softmax_output,
        &oracle_softmax,
        manifest.epsilon,
        softmax_dump,
    )?;
    let mismatch_count = conv2d_edge.mismatch_count
        + batchnorm_edge.mismatch_count
        + relu_edge.mismatch_count
        + softmax_edge.mismatch_count;
    let max_abs_error = conv2d_edge
        .max_abs_error
        .max(batchnorm_edge.max_abs_error)
        .max(relu_edge.max_abs_error)
        .max(softmax_edge.max_abs_error);
    let max_ulp_error = conv2d_edge
        .max_ulp_error
        .max(batchnorm_edge.max_ulp_error)
        .max(relu_edge.max_ulp_error)
        .max(softmax_edge.max_ulp_error);
    let status = if mismatch_count == 0 { "pass" } else { "fail" };

    let mut edges = BTreeMap::new();
    edges.insert("conv2d_output", conv2d_edge);
    edges.insert("batchnorm_output", batchnorm_edge);
    edges.insert("relu_output", relu_edge);
    edges.insert("softmax_output", softmax_edge);

    let report = json!({
        "status": status,
        "device": {
            "role": args.device,
            "name": run.device_name,
            "vendor_id": format!("0x{:04x}", run.vendor_id),
            "device_id": format!("0x{:04x}", run.device_id),
        },
        "graph": ["conv2d", "batchnorm", "relu", "softmax"],
        "graph_execution_kind": "multi-dispatch_chained_vulkan_storage_buffers",
        "single_dispatch_graph_runtime_claimed": false,
        "artificial_padding_used": false,
        "element_count": ELEMENTS,
        "epsilon": manifest.epsilon,
        "dispatch": {
            "conv2d": {"x": 256, "y": 1, "z": 1, "local_size": [1, 1, 1]},
            "batchnorm": {"x": 1, "y": 1, "z": 1, "local_size": [256, 1, 1]},
            "relu": {"x": 1, "y": 1, "z": 1, "local_size": [256, 1, 1]},
            "softmax": {"x": 1, "y": 1, "z": 1, "local_size": [256, 1, 1]},
            "barrier": "VK_ACCESS_SHADER_WRITE_BIT -> VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT"
        },
        "source_spv": {
            "conv2d": {"path": args.conv2d_spv, "sha256": conv2d_spv_sha256},
            "batchnorm": {"path": args.batchnorm_spv, "sha256": batchnorm_spv_sha256},
            "relu": {"path": args.relu_spv, "sha256": relu_spv_sha256},
            "softmax": {"path": args.softmax_spv, "sha256": softmax_spv_sha256}
        },
        "fixture_manifest": args.fixture_manifest,
        "fixture_hashes": {
            "conv2d_input": manifest.inputs.conv2d_input.sha256,
            "conv2d_filter": manifest.inputs.conv2d_filter.sha256,
            "batchnorm_params": manifest.inputs.batchnorm_params.sha256,
            "conv2d_output_oracle": manifest.edges.conv2d_output.sha256,
            "batchnorm_output_oracle": manifest.edges.batchnorm_output.sha256,
            "relu_output_oracle": manifest.edges.relu_output.sha256,
            "softmax_output_oracle": manifest.edges.softmax_output.sha256
        },
        "edges": edges,
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
        "max_ulp_error": max_ulp_error,
        "non_claims": {
            "claims_full_tensorflow_gpu_support": false,
            "claims_single_dispatch_graph_runtime": false,
            "uses_cublas_or_cudnn": false,
            "uses_vectoradd_fallback": false,
            "uses_fabricated_semantic_input": false,
            "uses_handwritten_spirv": false,
            "uses_pad_to_1024": false,
            "includes_softmax": true
        }
    });
    fs::write(&args.report, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing {}", args.report.display()))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

struct RunOutput {
    conv2d_output: Vec<f32>,
    batchnorm_output: Vec<f32>,
    relu_output: Vec<f32>,
    softmax_output: Vec<f32>,
    device_name: String,
    vendor_id: u32,
    device_id: u32,
}

unsafe fn run_vulkan(
    role: &str,
    conv2d_spv: &[u8],
    batchnorm_spv: &[u8],
    relu_spv: &[u8],
    softmax_spv: &[u8],
    conv2d_input: &[f32],
    conv2d_filter: &[f32],
    batchnorm_params: &[f32],
) -> Result<RunOutput> {
    let entry = Entry::load().context("loading Vulkan entry")?;
    let app_name = CString::new("cuda-spirv-chained-graph-runtime")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&app_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_1);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = entry
        .create_instance(&instance_info, None)
        .context("creating Vulkan instance")?;

    let physicals = instance
        .enumerate_physical_devices()
        .context("enumerating physical devices")?;
    let (physical, props, queue_family_index) =
        select_physical_device(&instance, &physicals, role)?;
    let priority = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priority)];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    let device = instance
        .create_device(physical, &device_info, None)
        .context("creating Vulkan logical device")?;
    let queue = device.get_device_queue(queue_family_index, 0);

    let descriptor_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    let descriptor_layout = device
        .create_descriptor_set_layout(&descriptor_layout_info, None)
        .context("creating descriptor set layout")?;
    let descriptor_layouts = [descriptor_layout];
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_layouts);
    let pipeline_layout = device
        .create_pipeline_layout(&pipeline_layout_info, None)
        .context("creating pipeline layout")?;

    let conv2d_pipeline = create_pipeline(&device, pipeline_layout, conv2d_spv)
        .context("creating Conv2D compute pipeline")?;
    let batchnorm_pipeline = create_pipeline(&device, pipeline_layout, batchnorm_spv)
        .context("creating BatchNorm compute pipeline")?;
    let relu_pipeline = create_pipeline(&device, pipeline_layout, relu_spv)
        .context("creating ReLU compute pipeline")?;
    let softmax_pipeline = create_pipeline(&device, pipeline_layout, softmax_spv)
        .context("creating Softmax compute pipeline")?;

    let memory_props = instance.get_physical_device_memory_properties(physical);
    let tensor_bytes = (ELEMENTS * F32_SIZE) as vk::DeviceSize;
    let filter_bytes = (FILTER_ELEMENTS * F32_SIZE) as vk::DeviceSize;
    let batchnorm_param_bytes = (BATCHNORM_PARAM_ELEMENTS * F32_SIZE) as vk::DeviceSize;
    let conv2d_input_buffer = create_storage_buffer(
        &device,
        &memory_props,
        tensor_bytes,
        Some(f32_slice_bytes(conv2d_input)),
    )?;
    let conv2d_filter_buffer = create_storage_buffer(
        &device,
        &memory_props,
        filter_bytes,
        Some(f32_slice_bytes(conv2d_filter)),
    )?;
    let conv2d_output_buffer = create_storage_buffer(&device, &memory_props, tensor_bytes, None)?;
    let batchnorm_param_buffer = create_storage_buffer(
        &device,
        &memory_props,
        batchnorm_param_bytes,
        Some(f32_slice_bytes(batchnorm_params)),
    )?;
    let batchnorm_output_buffer =
        create_storage_buffer(&device, &memory_props, tensor_bytes, None)?;
    let relu_aux_buffer = create_storage_buffer(
        &device,
        &memory_props,
        tensor_bytes,
        Some(&vec![0_u8; tensor_bytes as usize]),
    )?;
    let relu_output_buffer = create_storage_buffer(&device, &memory_props, tensor_bytes, None)?;
    let softmax_aux_buffer = create_storage_buffer(
        &device,
        &memory_props,
        tensor_bytes,
        Some(&vec![0_u8; tensor_bytes as usize]),
    )?;
    let softmax_output_buffer = create_storage_buffer(&device, &memory_props, tensor_bytes, None)?;

    let pool_size = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(12)];
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(4)
        .pool_sizes(&pool_size);
    let descriptor_pool = device
        .create_descriptor_pool(&descriptor_pool_info, None)
        .context("creating descriptor pool")?;
    let layouts = [
        descriptor_layout,
        descriptor_layout,
        descriptor_layout,
        descriptor_layout,
    ];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let sets = device
        .allocate_descriptor_sets(&allocate_info)
        .context("allocating descriptor sets")?;
    update_descriptor_set(
        &device,
        sets[0],
        [
            &conv2d_input_buffer,
            &conv2d_filter_buffer,
            &conv2d_output_buffer,
        ],
    );
    update_descriptor_set(
        &device,
        sets[1],
        [
            &conv2d_output_buffer,
            &batchnorm_param_buffer,
            &batchnorm_output_buffer,
        ],
    );
    update_descriptor_set(
        &device,
        sets[2],
        [
            &batchnorm_output_buffer,
            &relu_aux_buffer,
            &relu_output_buffer,
        ],
    );
    update_descriptor_set(
        &device,
        sets[3],
        [
            &relu_output_buffer,
            &softmax_aux_buffer,
            &softmax_output_buffer,
        ],
    );

    let command_pool_info =
        vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
    let command_pool = device
        .create_command_pool(&command_pool_info, None)
        .context("creating command pool")?;
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = device
        .allocate_command_buffers(&command_buffer_info)
        .context("allocating command buffer")?[0];
    let begin = vk::CommandBufferBeginInfo::default();
    device
        .begin_command_buffer(command_buffer, &begin)
        .context("beginning command buffer")?;
    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        conv2d_pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[sets[0]],
        &[],
    );
    device.cmd_dispatch(command_buffer, ELEMENTS as u32, 1, 1);
    let barrier = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &barrier,
        &[],
        &[],
    );
    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        batchnorm_pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[sets[1]],
        &[],
    );
    device.cmd_dispatch(command_buffer, 1, 1, 1);
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &barrier,
        &[],
        &[],
    );
    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        relu_pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[sets[2]],
        &[],
    );
    device.cmd_dispatch(command_buffer, 1, 1, 1);
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &barrier,
        &[],
        &[],
    );
    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        softmax_pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[sets[3]],
        &[],
    );
    device.cmd_dispatch(command_buffer, 1, 1, 1);
    device
        .end_command_buffer(command_buffer)
        .context("ending command buffer")?;

    let command_buffers = [command_buffer];
    let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
    device
        .queue_submit(queue, &submit, vk::Fence::null())
        .context("submitting command buffer")?;
    device
        .queue_wait_idle(queue)
        .context("waiting for queue idle")?;

    let conv2d_output = read_buffer_f32(&device, &conv2d_output_buffer, ELEMENTS)?;
    let batchnorm_output = read_buffer_f32(&device, &batchnorm_output_buffer, ELEMENTS)?;
    let relu_output = read_buffer_f32(&device, &relu_output_buffer, ELEMENTS)?;
    let softmax_output = read_buffer_f32(&device, &softmax_output_buffer, ELEMENTS)?;
    device.device_wait_idle().ok();

    device.destroy_command_pool(command_pool, None);
    device.destroy_descriptor_pool(descriptor_pool, None);
    destroy_buffer(&device, softmax_output_buffer);
    destroy_buffer(&device, softmax_aux_buffer);
    destroy_buffer(&device, relu_output_buffer);
    destroy_buffer(&device, relu_aux_buffer);
    destroy_buffer(&device, batchnorm_output_buffer);
    destroy_buffer(&device, batchnorm_param_buffer);
    destroy_buffer(&device, conv2d_output_buffer);
    destroy_buffer(&device, conv2d_filter_buffer);
    destroy_buffer(&device, conv2d_input_buffer);
    device.destroy_pipeline(softmax_pipeline.pipeline, None);
    device.destroy_shader_module(softmax_pipeline.shader, None);
    device.destroy_pipeline(relu_pipeline.pipeline, None);
    device.destroy_shader_module(relu_pipeline.shader, None);
    device.destroy_pipeline(batchnorm_pipeline.pipeline, None);
    device.destroy_shader_module(batchnorm_pipeline.shader, None);
    device.destroy_pipeline(conv2d_pipeline.pipeline, None);
    device.destroy_shader_module(conv2d_pipeline.shader, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
    device.destroy_descriptor_set_layout(descriptor_layout, None);
    device.destroy_device(None);
    instance.destroy_instance(None);

    Ok(RunOutput {
        conv2d_output,
        batchnorm_output,
        relu_output,
        softmax_output,
        device_name: CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned(),
        vendor_id: props.vendor_id,
        device_id: props.device_id,
    })
}

unsafe fn create_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    spv: &[u8],
) -> Result<Pipeline> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spv)).context("reading SPIR-V")?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);
    let shader = device
        .create_shader_module(&shader_info, None)
        .context("creating shader module")?;
    let entry = CString::new("main")?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry);
    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout)];
    let pipeline = device
        .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        .map_err(|(_, err)| anyhow!("creating compute pipeline: {err:?}"))?[0];
    Ok(Pipeline { shader, pipeline })
}

unsafe fn select_physical_device(
    instance: &ash::Instance,
    physicals: &[vk::PhysicalDevice],
    role: &str,
) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> {
    for physical in physicals {
        let props = instance.get_physical_device_properties(*physical);
        let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
        let matches_role = match role {
            "llvmpipe" => {
                name.to_ascii_lowercase().contains("llvmpipe") || props.vendor_id == 0x10005
            }
            "intel" => props.vendor_id == 0x8086,
            "nvidia" => props.vendor_id == 0x10de,
            other => bail!("unsupported --device {other}; expected intel, llvmpipe, or nvidia"),
        };
        if !matches_role {
            continue;
        }
        let queue_families = instance.get_physical_device_queue_family_properties(*physical);
        if let Some((idx, _)) = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        {
            return Ok((*physical, props, idx as u32));
        }
    }
    bail!("no matching Vulkan compute device found for role {role}");
}

unsafe fn create_storage_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    initial: Option<&[u8]>,
) -> Result<Buffer> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = device
        .create_buffer(&buffer_info, None)
        .context("creating storage buffer")?;
    let req = device.get_buffer_memory_requirements(buffer);
    let memory_type_index = find_memory_type(
        memory_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type_index);
    let memory = device
        .allocate_memory(&alloc, None)
        .context("allocating buffer memory")?;
    device
        .bind_buffer_memory(buffer, memory, 0)
        .context("binding buffer memory")?;
    if let Some(bytes) = initial {
        if bytes.len() as vk::DeviceSize > size {
            bail!("initial buffer payload is larger than target buffer");
        }
        let mapped = device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .context("mapping initial buffer memory")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        device.unmap_memory(memory);
    }
    Ok(Buffer {
        buffer,
        memory,
        size,
    })
}

unsafe fn read_buffer_f32(
    device: &ash::Device,
    buffer: &Buffer,
    elements: usize,
) -> Result<Vec<f32>> {
    let byte_len = elements * F32_SIZE;
    if byte_len as vk::DeviceSize > buffer.size {
        bail!("read size exceeds buffer size");
    }
    let mapped = device
        .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
        .context("mapping readback buffer memory")?;
    let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), byte_len);
    let values = bytes
        .chunks_exact(F32_SIZE)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk")))
        .collect();
    device.unmap_memory(buffer.memory);
    Ok(values)
}

unsafe fn destroy_buffer(device: &ash::Device, buffer: Buffer) {
    device.destroy_buffer(buffer.buffer, None);
    device.free_memory(buffer.memory, None);
}

unsafe fn update_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    buffers: [&Buffer; 3],
) {
    let infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[0].buffer)
            .offset(0)
            .range(buffers[0].size),
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[1].buffer)
            .offset(0)
            .range(buffers[1].size),
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[2].buffer)
            .offset(0)
            .range(buffers[2].size),
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[0])),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[1])),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[2])),
    ];
    device.update_descriptor_sets(&writes, &[]);
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for idx in 0..props.memory_type_count {
        let supported = (type_bits & (1 << idx)) != 0;
        let has_flags = props.memory_types[idx as usize]
            .property_flags
            .contains(flags);
        if supported && has_flags {
            return Ok(idx);
        }
    }
    bail!("no HOST_VISIBLE|HOST_COHERENT memory type for storage buffer");
}

fn compare_edge(
    actual: &[f32],
    expected: &[f32],
    epsilon: f32,
    dump_path: PathBuf,
) -> Result<EdgeReport> {
    if actual.len() != expected.len() {
        bail!(
            "edge length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        );
    }
    let mut mismatch_count = 0_usize;
    let mut max_abs_error = 0.0_f32;
    let mut max_ulp_error = 0_u32;
    for (actual, expected) in actual.iter().copied().zip(expected.iter().copied()) {
        let abs = (actual - expected).abs();
        let ulp = ulp_distance(actual, expected);
        if abs > epsilon {
            mismatch_count += 1;
        }
        max_abs_error = max_abs_error.max(abs);
        max_ulp_error = max_ulp_error.max(ulp);
    }
    Ok(EdgeReport {
        tensor_sha256: sha256_bytes(f32_slice_bytes(actual)),
        oracle_sha256: sha256_bytes(f32_slice_bytes(expected)),
        mismatch_count,
        max_abs_error,
        max_ulp_error,
        dump_path: dump_path.display().to_string(),
    })
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    fn ordered(v: f32) -> i32 {
        let bits = v.to_bits() as i32;
        if bits < 0 {
            i32::MIN - bits
        } else {
            bits
        }
    }
    ordered(a).abs_diff(ordered(b))
}

fn parse_args() -> Result<Args> {
    let mut values = std::env::args().skip(1);
    let mut map = BTreeMap::new();
    while let Some(flag) = values.next() {
        if !flag.starts_with("--") {
            bail!("unexpected positional argument {flag}");
        }
        let value = values
            .next()
            .ok_or_else(|| anyhow!("missing value for {flag}"))?;
        map.insert(flag, value);
    }
    let get = |map: &mut BTreeMap<String, String>, name: &str| -> Result<String> {
        map.remove(name)
            .ok_or_else(|| anyhow!("missing required CLI flag {name}"))
    };
    let mut map = map;
    let args = Args {
        device: get(&mut map, "--device")?,
        conv2d_spv: PathBuf::from(get(&mut map, "--conv2d-spv")?),
        batchnorm_spv: PathBuf::from(get(&mut map, "--batchnorm-spv")?),
        relu_spv: PathBuf::from(get(&mut map, "--relu-spv")?),
        softmax_spv: PathBuf::from(get(&mut map, "--softmax-spv")?),
        fixture_manifest: PathBuf::from(get(&mut map, "--fixture-manifest")?),
        report: PathBuf::from(get(&mut map, "--report")?),
        dump_dir: PathBuf::from(get(&mut map, "--dump-dir")?),
    };
    if !map.is_empty() {
        bail!("unknown CLI flags: {:?}", map.keys().collect::<Vec<_>>());
    }
    Ok(args)
}

fn read_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("{} is not a NumPy .npy file", path.display());
    }
    let major = bytes[6];
    let header_len_offset = 8;
    let (header_len, data_offset) = match major {
        1 => (
            u16::from_le_bytes(bytes[header_len_offset..header_len_offset + 2].try_into()?)
                as usize,
            10,
        ),
        2 | 3 => (
            u32::from_le_bytes(bytes[header_len_offset..header_len_offset + 4].try_into()?)
                as usize,
            12,
        ),
        _ => bail!("unsupported .npy version {} in {}", major, path.display()),
    };
    let header_end = data_offset + header_len;
    if header_end > bytes.len() {
        bail!("truncated .npy header in {}", path.display());
    }
    let header = std::str::from_utf8(&bytes[data_offset..header_end])
        .with_context(|| format!("decoding .npy header {}", path.display()))?;
    if !header.contains("'descr': '<f4'") && !header.contains("\"descr\": \"<f4\"") {
        bail!("{} is not little-endian float32: {header}", path.display());
    }
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        bail!(
            "{} uses Fortran order, unsupported by harness",
            path.display()
        );
    }
    let payload = &bytes[header_end..];
    if payload.len() % F32_SIZE != 0 {
        bail!("{} float32 payload is not 4-byte aligned", path.display());
    }
    Ok(payload
        .chunks_exact(F32_SIZE)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk")))
        .collect())
}

fn f32_slice_bytes(values: &[f32]) -> &[u8] {
    bytemuck::cast_slice(values)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}
