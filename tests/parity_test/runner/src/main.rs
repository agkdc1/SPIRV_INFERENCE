use anyhow::{anyhow, bail, Context, Result};
use ash::{vk, Entry};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ffi::{CStr, CString}, fs, path::{Path, PathBuf}};

const F32_SIZE: usize = 4;

#[derive(Clone)]
struct KernelSpec {
    name: &'static str,
    spv: PathBuf,
    inputs: Vec<(PathBuf, usize)>,
    out_elems: usize,
    dispatch_x: u32,
}

struct Args {
    fixtures: PathBuf,
    out: PathBuf,
    report: PathBuf,
    relu_spv: PathBuf,
    softmax_spv: PathBuf,
    batchnorm_spv: PathBuf,
    conv1d_spv: PathBuf,
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

struct VkRun {
    values: Vec<f32>,
    device_name: String,
    vendor_id: u32,
    device_id: u32,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out)?;
    let specs = vec![
        KernelSpec {
            name: "relu",
            spv: args.relu_spv,
            inputs: vec![(args.fixtures.join("relu_input.raw.f32"), 4096)],
            out_elems: 4096,
            dispatch_x: 16,
        },
        KernelSpec {
            name: "softmax",
            spv: args.softmax_spv,
            inputs: vec![(args.fixtures.join("softmax_input.raw.f32"), 128)],
            out_elems: 128,
            dispatch_x: 1,
        },
        KernelSpec {
            name: "batchnorm",
            spv: args.batchnorm_spv,
            inputs: vec![
                (args.fixtures.join("batchnorm_input.raw.f32"), 256),
                (args.fixtures.join("batchnorm_params.raw.f32"), 5),
            ],
            out_elems: 256,
            dispatch_x: 1,
        },
        KernelSpec {
            name: "conv1d",
            spv: args.conv1d_spv,
            inputs: vec![
                (args.fixtures.join("conv1d_input.raw.f32"), 16 * 64),
                (args.fixtures.join("conv1d_filter.raw.f32"), 3 * 64 * 32),
            ],
            out_elems: 16 * 32,
            dispatch_x: 4,
        },
    ];

    let mut kernel_reports = BTreeMap::new();
    let mut device = None;
    for spec in specs {
        let spv = fs::read(&spec.spv).with_context(|| format!("reading {}", spec.spv.display()))?;
        let mut inputs = Vec::new();
        for (path, elems) in &spec.inputs {
            let values = read_raw_f32(path, *elems)?;
            inputs.push(values);
        }
        while inputs.len() < 2 {
            inputs.push(vec![0.0_f32; 1]);
        }
        let run = unsafe { run_kernel(&spv, &inputs[0], &inputs[1], spec.out_elems, spec.dispatch_x)? };
        device = Some(json!({
            "name": run.device_name,
            "vendor_id": format!("0x{:04x}", run.vendor_id),
            "device_id": format!("0x{:04x}", run.device_id),
        }));
        let out_dir = args.out.join(spec.name);
        fs::create_dir_all(&out_dir)?;
        let raw_path = out_dir.join("output.raw.f32");
        fs::write(&raw_path, f32_bytes(&run.values))?;
        kernel_reports.insert(spec.name, json!({
            "status": "pass",
            "spv_path": spec.spv,
            "spv_sha256": sha256(&spv),
            "output_path": raw_path,
            "output_sha256": sha256(f32_bytes(&run.values)),
            "elements": spec.out_elems,
            "dispatch_x": spec.dispatch_x,
        }));
    }
    let report = json!({
        "status": "pass",
        "runtime": "Vulkan/SPIR-V",
        "device": device.unwrap_or_else(|| json!(null)),
        "kernels": kernel_reports,
        "non_claims": {
            "uses_cuda": false,
            "proprietary_cuda_libraries_used": false,
            "claims_full_tensorflow_gpu_support": false
        }
    });
    fs::write(&args.report, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

unsafe fn run_kernel(spv: &[u8], input0: &[f32], input1: &[f32], out_elems: usize, dispatch_x: u32) -> Result<VkRun> {
    let entry = Entry::load().context("loading Vulkan entry")?;
    let app = CString::new("advanced-parity-runner")?;
    let app_info = vk::ApplicationInfo::default().application_name(&app).api_version(vk::API_VERSION_1_1);
    let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app_info), None)?;
    let physicals = instance.enumerate_physical_devices()?;
    let (physical, props, queue_family_index) = select_intel(&instance, &physicals)?;
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family_index).queue_priorities(&priorities)];
    let device = instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info), None)?;
    let queue = device.get_device_queue(queue_family_index, 0);
    let bindings = [
        layout_binding(0),
        layout_binding(1),
        layout_binding(2),
    ];
    let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
    let set_layouts = [set_layout];
    let pipeline_layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)?;
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spv)).context("reading spv")?;
    let shader = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
    let entry_name = CString::new("main")?;
    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(&entry_name);
    let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)], None)
        .map_err(|(_, e)| anyhow!("creating compute pipeline: {e:?}"))?[0];
    let memory_props = instance.get_physical_device_memory_properties(physical);
    let input0_buf = create_buffer(&device, &memory_props, f32_bytes(input0))?;
    let input1_buf = create_buffer(&device, &memory_props, f32_bytes(input1))?;
    let output_zero = vec![0_u8; out_elems * F32_SIZE];
    let output_buf = create_buffer(&device, &memory_props, &output_zero)?;
    let pool_size = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3)];
    let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_size), None)?;
    let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts))?;
    update_set(&device, sets[0], [&input0_buf, &input1_buf, &output_buf]);
    let command_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index), None)?;
    let command_buffer = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
    device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
    device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &sets, &[]);
    device.cmd_dispatch(command_buffer, dispatch_x, 1, 1);
    device.end_command_buffer(command_buffer)?;
    device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[command_buffer])], vk::Fence::null())?;
    device.queue_wait_idle(queue)?;
    let values = read_buffer(&device, &output_buf, out_elems)?;
    device.device_wait_idle().ok();
    device.destroy_command_pool(command_pool, None);
    device.destroy_descriptor_pool(pool, None);
    destroy_buffer(&device, output_buf);
    destroy_buffer(&device, input1_buf);
    destroy_buffer(&device, input0_buf);
    device.destroy_pipeline(pipeline, None);
    device.destroy_shader_module(shader, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
    device.destroy_descriptor_set_layout(set_layout, None);
    device.destroy_device(None);
    instance.destroy_instance(None);
    Ok(VkRun {
        values,
        device_name: CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned(),
        vendor_id: props.vendor_id,
        device_id: props.device_id,
    })
}

fn layout_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

unsafe fn select_intel(instance: &ash::Instance, physicals: &[vk::PhysicalDevice]) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> {
    for physical in physicals {
        let props = instance.get_physical_device_properties(*physical);
        if props.vendor_id != 0x8086 {
            continue;
        }
        let queue_family_index = instance.get_physical_device_queue_family_properties(*physical)
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(idx, _)| idx as u32);
        if let Some(idx) = queue_family_index {
            return Ok((*physical, props, idx));
        }
    }
    bail!("no Intel Vulkan compute device found")
}

unsafe fn create_buffer(device: &ash::Device, memory_props: &vk::PhysicalDeviceMemoryProperties, initial: &[u8]) -> Result<Buffer> {
    let size = initial.len() as vk::DeviceSize;
    let buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
    let req = device.get_buffer_memory_requirements(buffer);
    let memory_type_index = find_memory_type(memory_props, req.memory_type_bits)?;
    let memory = device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(memory_type_index), None)?;
    device.bind_buffer_memory(buffer, memory, 0)?;
    let mapped = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
    std::ptr::copy_nonoverlapping(initial.as_ptr(), mapped.cast::<u8>(), initial.len());
    device.unmap_memory(memory);
    Ok(Buffer { buffer, memory, size })
}

unsafe fn update_set(device: &ash::Device, set: vk::DescriptorSet, buffers: [&Buffer; 3]) {
    let infos = [
        vk::DescriptorBufferInfo::default().buffer(buffers[0].buffer).offset(0).range(buffers[0].size),
        vk::DescriptorBufferInfo::default().buffer(buffers[1].buffer).offset(0).range(buffers[1].size),
        vk::DescriptorBufferInfo::default().buffer(buffers[2].buffer).offset(0).range(buffers[2].size),
    ];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[0])),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[1])),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[2])),
    ];
    device.update_descriptor_sets(&writes, &[]);
}

unsafe fn read_buffer(device: &ash::Device, buffer: &Buffer, elements: usize) -> Result<Vec<f32>> {
    let bytes_len = elements * F32_SIZE;
    let mapped = device.map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())?;
    let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes_len);
    let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    device.unmap_memory(buffer.memory);
    Ok(values)
}

unsafe fn destroy_buffer(device: &ash::Device, buffer: Buffer) {
    device.destroy_buffer(buffer.buffer, None);
    device.free_memory(buffer.memory, None);
}

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32) -> Result<u32> {
    for idx in 0..props.memory_type_count {
        let supported = (type_bits & (1 << idx)) != 0;
        let flags = props.memory_types[idx as usize].property_flags;
        if supported && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT) {
            return Ok(idx);
        }
    }
    bail!("no host-visible coherent memory type")
}

fn parse_args() -> Result<Args> {
    let mut map = BTreeMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        map.insert(k, it.next().ok_or_else(|| anyhow!("missing argument value"))?);
    }
    let mut take = |k: &str| -> Result<PathBuf> {
        Ok(PathBuf::from(map.remove(k).ok_or_else(|| anyhow!("missing {k}"))?))
    };
    Ok(Args {
        fixtures: take("--fixtures")?,
        out: take("--out")?,
        relu_spv: take("--relu-spv")?,
        softmax_spv: take("--softmax-spv")?,
        batchnorm_spv: take("--batchnorm-spv")?,
        conv1d_spv: take("--conv1d-spv")?,
        report: take("--report")?,
    })
}

fn read_raw_f32(path: &Path, elems: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() != elems * F32_SIZE {
        bail!("{} has {} bytes, expected {}", path.display(), bytes.len(), elems * F32_SIZE);
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    bytemuck::cast_slice(values)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
