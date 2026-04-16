mod fixture;

use anyhow::{anyhow, bail, Context, Result};
use ash::{vk, Entry};
use fixture::{f32_bytes, sha256};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const F32_SIZE: usize = 4;

#[derive(Debug)]
struct Args {
    device: String,
    fixtures: PathBuf,
    kernel_dir: PathBuf,
    report: PathBuf,
    dump_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    epsilon: f32,
    files: BTreeMap<String, FileRef>,
}

#[derive(Debug, Deserialize)]
struct FileRef {
    path: PathBuf,
    sha256: String,
    elements: usize,
}

#[derive(Debug)]
struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

#[derive(Debug)]
struct VkOutput {
    values: Vec<f32>,
    device_name: String,
    vendor_id: u32,
    device_id: u32,
    elapsed_ms: f64,
}

#[derive(Clone, Copy)]
struct KernelSpec<'a> {
    name: &'a str,
    spv_name: &'a str,
    input_elements: usize,
    aux_elements: usize,
    output_elements: usize,
    dispatch_x: u32,
}

fn main() -> Result<()> {
    let raw = std::env::args().collect::<Vec<_>>();
    if raw.len() == 3 && raw[1] == "--write-fixtures" {
        let manifest = fixture::write_fixture_dir(Path::new(&raw[2]))?;
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }

    let args = parse_args()?;
    if !args.fixtures.exists() {
        let fixture_dir = args
            .fixtures
            .parent()
            .context("--fixtures must include a manifest filename")?;
        fixture::write_fixture_dir(fixture_dir)?;
    }
    fs::create_dir_all(&args.dump_dir)
        .with_context(|| format!("creating {}", args.dump_dir.display()))?;
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }

    let manifest_dir = args
        .fixtures
        .parent()
        .context("--fixtures must include a manifest filename")?;
    let manifest: Manifest = serde_json::from_slice(
        &fs::read(&args.fixtures)
            .with_context(|| format!("reading {}", args.fixtures.display()))?,
    )?;
    let input = read_manifest_f32(&manifest, manifest_dir, "input", fixture::INPUT)?;
    let conv_weight =
        read_manifest_f32(&manifest, manifest_dir, "conv_weight", fixture::CONV_WEIGHT)?;
    let bn_param = read_manifest_f32(
        &manifest,
        manifest_dir,
        "batchnorm_param",
        fixture::BN_PARAM,
    )?;
    let fc_param = read_manifest_f32(
        &manifest,
        manifest_dir,
        "fc_param",
        fixture::FC_WEIGHT + fixture::FC_BIAS,
    )?;
    let conv_ref = read_manifest_f32(
        &manifest,
        manifest_dir,
        "conv_output_ref",
        fixture::CONV_OUT,
    )?;
    let bn_ref = read_manifest_f32(
        &manifest,
        manifest_dir,
        "batchnorm_output_ref",
        fixture::CONV_OUT,
    )?;
    let relu_ref = read_manifest_f32(
        &manifest,
        manifest_dir,
        "relu_output_ref",
        fixture::CONV_OUT,
    )?;
    let logits_ref = read_manifest_f32(&manifest, manifest_dir, "logits_ref", fixture::LOGITS)?;
    let softmax_ref = read_manifest_f32(
        &manifest,
        manifest_dir,
        "softmax_output_ref",
        fixture::LOGITS,
    )?;

    let specs = [
        KernelSpec {
            name: "conv1d",
            spv_name: "conv1d_infer.spv",
            input_elements: fixture::INPUT,
            aux_elements: fixture::CONV_WEIGHT,
            output_elements: fixture::CONV_OUT,
            dispatch_x: 1,
        },
        KernelSpec {
            name: "batchnorm",
            spv_name: "batchnorm_infer.spv",
            input_elements: fixture::CONV_OUT,
            aux_elements: fixture::BN_PARAM,
            output_elements: fixture::CONV_OUT,
            dispatch_x: 1,
        },
        KernelSpec {
            name: "relu",
            spv_name: "relu_infer.spv",
            input_elements: fixture::CONV_OUT,
            aux_elements: 1,
            output_elements: fixture::CONV_OUT,
            dispatch_x: 1,
        },
        KernelSpec {
            name: "fc",
            spv_name: "fc_infer.spv",
            input_elements: fixture::CONV_OUT,
            aux_elements: fixture::FC_WEIGHT + fixture::FC_BIAS,
            output_elements: fixture::LOGITS,
            dispatch_x: 1,
        },
        KernelSpec {
            name: "softmax",
            spv_name: "softmax10_infer.spv",
            input_elements: fixture::LOGITS,
            aux_elements: 1,
            output_elements: fixture::LOGITS,
            dispatch_x: 1,
        },
    ];

    let unused = vec![0.0_f32; 1];
    let conv_spv = fs::read(args.kernel_dir.join(specs[0].spv_name))?;
    let conv = unsafe { run_kernel(&args.device, &conv_spv, &input, &conv_weight, specs[0])? };
    let bn_spv = fs::read(args.kernel_dir.join(specs[1].spv_name))?;
    let bn = unsafe { run_kernel(&args.device, &bn_spv, &conv.values, &bn_param, specs[1])? };
    let relu_spv = fs::read(args.kernel_dir.join(specs[2].spv_name))?;
    let relu = unsafe { run_kernel(&args.device, &relu_spv, &bn.values, &unused, specs[2])? };
    let fc_spv = fs::read(args.kernel_dir.join(specs[3].spv_name))?;
    let logits = unsafe { run_kernel(&args.device, &fc_spv, &relu.values, &fc_param, specs[3])? };
    let softmax_spv = fs::read(args.kernel_dir.join(specs[4].spv_name))?;
    let softmax = unsafe {
        run_kernel(
            &args.device,
            &softmax_spv,
            &logits.values,
            &unused,
            specs[4],
        )?
    };

    let outputs = [
        ("conv1d", &conv.values, &conv_ref, "conv1d_output.raw.f32"),
        ("batchnorm", &bn.values, &bn_ref, "batchnorm_output.raw.f32"),
        ("relu", &relu.values, &relu_ref, "relu_output.raw.f32"),
        ("fc", &logits.values, &logits_ref, "logits_output.raw.f32"),
        (
            "softmax",
            &softmax.values,
            &softmax_ref,
            "softmax_output.raw.f32",
        ),
    ];
    let mut edge_reports = BTreeMap::new();
    let mut mismatch_count = 0_usize;
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    let mut max_ulp_error = 0_u32;
    for (name, actual, expected, filename) in outputs {
        let dump_path = args.dump_dir.join(filename);
        fs::write(&dump_path, f32_bytes(actual))
            .with_context(|| format!("writing {}", dump_path.display()))?;
        let edge = compare(actual, expected, manifest.epsilon);
        mismatch_count += edge.mismatch_count;
        max_abs_error = max_abs_error.max(edge.max_abs_error);
        max_rel_error = max_rel_error.max(edge.max_rel_error);
        max_ulp_error = max_ulp_error.max(edge.max_ulp_error);
        edge_reports.insert(
            name,
            json!({
                "status": if edge.mismatch_count == 0 { "pass" } else { "fail" },
                "tensor_sha256": sha256(f32_bytes(actual)),
                "oracle_sha256": sha256(f32_bytes(expected)),
                "dump_path": dump_path,
                "mismatch_count": edge.mismatch_count,
                "max_abs_error": edge.max_abs_error,
                "max_rel_error": edge.max_rel_error,
                "max_ulp_error": edge.max_ulp_error,
            }),
        );
    }

    let spv_hashes = specs
        .iter()
        .map(|s| {
            let path = args.kernel_dir.join(s.spv_name);
            let bytes = fs::read(&path).unwrap_or_default();
            (
                s.name,
                json!({"path": path, "sha256": sha256(&bytes), "bytes": bytes.len()}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let latency_ms = json!({
        "conv1d": conv.elapsed_ms,
        "batchnorm": bn.elapsed_ms,
        "relu": relu.elapsed_ms,
        "fc": logits.elapsed_ms,
        "softmax": softmax.elapsed_ms,
        "total": conv.elapsed_ms + bn.elapsed_ms + relu.elapsed_ms + logits.elapsed_ms + softmax.elapsed_ms,
    });
    let status = if mismatch_count == 0 { "pass" } else { "fail" };
    let report = json!({
        "status": status,
        "device": {
            "role": args.device,
            "name": softmax.device_name,
            "vendor_id": format!("0x{:04x}", softmax.vendor_id),
            "device_id": format!("0x{:04x}", softmax.device_id),
        },
        "graph": ["conv1d", "batchnorm", "relu", "fc", "softmax"],
        "graph_execution_kind": "multi-dispatch_vulkan_spirv_storage_buffers",
        "fixture_manifest": args.fixtures,
        "fixture_hashes": manifest.files.iter().map(|(k, v)| (k, v.sha256.clone())).collect::<BTreeMap<_, _>>(),
        "spirv_modules": spv_hashes,
        "edges": edge_reports,
        "latency_ms": latency_ms,
        "thresholds": {
            "max_abs_error": manifest.epsilon,
            "max_rel_error": 1.0e-4,
            "max_ulp_error": 64,
        },
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "max_ulp_error": max_ulp_error,
        "top1": top1(&softmax.values),
        "output_sha256": sha256(f32_bytes(&softmax.values)),
        "non_claims": {
            "uses_cuda": false,
            "uses_cublas": false,
            "uses_cudnn": false,
            "uses_tensorrt": false,
            "claims_tensorflow_runtime": false,
            "claims_full_tensorflow_gpu_support": false,
            "uses_handwritten_spirv": false
        }
    });
    fs::write(&args.report, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing {}", args.report.display()))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[derive(Debug)]
struct Compare {
    mismatch_count: usize,
    max_abs_error: f32,
    max_rel_error: f32,
    max_ulp_error: u32,
}

fn compare(actual: &[f32], expected: &[f32], max_abs: f32) -> Compare {
    let mut out = Compare {
        mismatch_count: 0,
        max_abs_error: 0.0,
        max_rel_error: 0.0,
        max_ulp_error: 0,
    };
    for (a, e) in actual.iter().zip(expected.iter()) {
        let abs = (*a - *e).abs();
        let rel = abs / e.abs().max(1.0e-30);
        let u = ulp(*a, *e);
        out.max_abs_error = out.max_abs_error.max(abs);
        out.max_rel_error = out.max_rel_error.max(rel);
        out.max_ulp_error = out.max_ulp_error.max(u);
        if abs > max_abs || rel > 1.0e-4 || u > 64 {
            out.mismatch_count += 1;
        }
    }
    out
}

fn ulp(a: f32, b: f32) -> u32 {
    let ia = ordered_i32(a);
    let ib = ordered_i32(b);
    ia.abs_diff(ib)
}

fn ordered_i32(v: f32) -> i32 {
    let bits = v.to_bits() as i32;
    if bits < 0 {
        i32::MIN - bits
    } else {
        bits
    }
}

fn top1(values: &[f32]) -> serde_json::Value {
    let (class, confidence) = values
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    json!({"class": class, "confidence": confidence})
}

unsafe fn run_kernel(
    device_selector: &str,
    spv: &[u8],
    input0: &[f32],
    input1: &[f32],
    spec: KernelSpec<'_>,
) -> Result<VkOutput> {
    if input0.len() != spec.input_elements || input1.len() != spec.aux_elements {
        bail!("{} input size mismatch", spec.name);
    }
    let entry = Entry::load().context("loading Vulkan entry")?;
    let app = CString::new("inference-poc")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app)
        .api_version(vk::API_VERSION_1_1);
    let instance = entry.create_instance(
        &vk::InstanceCreateInfo::default().application_info(&app_info),
        None,
    )?;
    let physicals = instance.enumerate_physical_devices()?;
    let (physical, props, queue_family_index) =
        select_device(&instance, &physicals, device_selector)?;
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)];
    let device = instance.create_device(
        physical,
        &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
        None,
    )?;
    let queue = device.get_device_queue(queue_family_index, 0);
    let bindings = [layout_binding(0), layout_binding(1), layout_binding(2)];
    let set_layout = device.create_descriptor_set_layout(
        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
        None,
    )?;
    let set_layouts = [set_layout];
    let pipeline_layout = device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
        None,
    )?;
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spv)).context("reading spv")?;
    let shader =
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
    let entry_name = CString::new("main")?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)],
            None,
        )
        .map_err(|(_, e)| anyhow!("creating compute pipeline: {e:?}"))?[0];
    let memory_props = instance.get_physical_device_memory_properties(physical);
    let input0_buf = create_buffer(&device, &memory_props, f32_bytes(input0))?;
    let input1_buf = create_buffer(&device, &memory_props, f32_bytes(input1))?;
    let output_zero = vec![0_u8; spec.output_elements * F32_SIZE];
    let output_buf = create_buffer(&device, &memory_props, &output_zero)?;
    let pool_size = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3)];
    let pool = device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_size),
        None,
    )?;
    let sets = device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts),
    )?;
    update_set(&device, sets[0], [&input0_buf, &input1_buf, &output_buf]);
    let command_pool = device.create_command_pool(
        &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index),
        None,
    )?;
    let command_buffer = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0];
    device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &sets,
        &[],
    );
    device.cmd_dispatch(command_buffer, spec.dispatch_x, 1, 1);
    device.end_command_buffer(command_buffer)?;
    let start = Instant::now();
    let command_buffers = [command_buffer];
    device.queue_submit(
        queue,
        &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
        vk::Fence::null(),
    )?;
    device.queue_wait_idle(queue)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let values = read_buffer(&device, &output_buf, spec.output_elements)?;
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
    Ok(VkOutput {
        values,
        device_name: CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned(),
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        elapsed_ms,
    })
}

fn layout_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

unsafe fn select_device(
    instance: &ash::Instance,
    physicals: &[vk::PhysicalDevice],
    selector: &str,
) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> {
    for physical in physicals {
        let props = instance.get_physical_device_properties(*physical);
        let vendor_match = match selector {
            "intel" => props.vendor_id == 0x8086,
            "nvidia" => props.vendor_id == 0x10de,
            "auto" => true,
            other => bail!("unknown --device {other}; expected intel|nvidia|auto"),
        };
        if !vendor_match {
            continue;
        }
        let queue_family_index = instance
            .get_physical_device_queue_family_properties(*physical)
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(idx, _)| idx as u32);
        if let Some(idx) = queue_family_index {
            return Ok((*physical, props, idx));
        }
    }
    bail!("no matching Vulkan compute device found for selector {selector}")
}

unsafe fn create_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    initial: &[u8],
) -> Result<Buffer> {
    let size = initial.len() as vk::DeviceSize;
    let buffer = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let req = device.get_buffer_memory_requirements(buffer);
    let memory_type_index = find_memory_type(memory_props, req.memory_type_bits)?;
    let memory = device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type_index),
        None,
    )?;
    device.bind_buffer_memory(buffer, memory, 0)?;
    let mapped = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
    std::ptr::copy_nonoverlapping(initial.as_ptr(), mapped.cast::<u8>(), initial.len());
    device.unmap_memory(memory);
    Ok(Buffer {
        buffer,
        memory,
        size,
    })
}

unsafe fn update_set(device: &ash::Device, set: vk::DescriptorSet, buffers: [&Buffer; 3]) {
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

unsafe fn read_buffer(device: &ash::Device, buffer: &Buffer, elements: usize) -> Result<Vec<f32>> {
    let bytes_len = elements * F32_SIZE;
    let mapped = device.map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())?;
    let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes_len);
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    device.unmap_memory(buffer.memory);
    Ok(values)
}

unsafe fn destroy_buffer(device: &ash::Device, buffer: Buffer) {
    device.destroy_buffer(buffer.buffer, None);
    device.free_memory(buffer.memory, None);
}

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32) -> Result<u32> {
    for i in 0..props.memory_type_count {
        let suitable = (type_bits & (1 << i)) != 0;
        let flags = props.memory_types[i as usize].property_flags;
        if suitable
            && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(i);
        }
    }
    bail!("no host-visible coherent Vulkan memory type")
}

fn read_manifest_f32(
    manifest: &Manifest,
    base: &Path,
    key: &str,
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let r = manifest
        .files
        .get(key)
        .with_context(|| format!("manifest missing {key}"))?;
    if r.elements != expected_elements {
        bail!(
            "{key}: expected {expected_elements} elements, manifest has {}",
            r.elements
        );
    }
    let path = base.join(&r.path);
    let data = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    if sha256(&data) != r.sha256 {
        bail!("{key}: sha256 mismatch for {}", path.display());
    }
    read_raw_f32_bytes(&data, expected_elements)
}

fn read_raw_f32_bytes(data: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if data.len() != expected_elements * F32_SIZE {
        bail!(
            "raw f32 byte length mismatch: got {}, expected {}",
            data.len(),
            expected_elements * F32_SIZE
        );
    }
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn parse_args() -> Result<Args> {
    let mut device = "auto".to_string();
    let mut fixtures = PathBuf::from("inference_poc/fixtures/manifest.json");
    let mut kernel_dir = PathBuf::from("inference_poc/kernels");
    let mut report = PathBuf::from("inference_poc/out/local_vulkan/report.json");
    let mut dump_dir = PathBuf::from("inference_poc/out/local_vulkan/tensors");
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => device = it.next().context("--device requires a value")?,
            "--fixtures" => {
                fixtures = PathBuf::from(it.next().context("--fixtures requires a value")?)
            }
            "--kernel-dir" => {
                kernel_dir = PathBuf::from(it.next().context("--kernel-dir requires a value")?)
            }
            "--report" => report = PathBuf::from(it.next().context("--report requires a value")?),
            "--dump-dir" => {
                dump_dir = PathBuf::from(it.next().context("--dump-dir requires a value")?)
            }
            "--help" | "-h" => {
                println!(
                    "usage: inference_poc [--device intel|nvidia|auto] [--fixtures manifest.json] [--kernel-dir dir] [--report path] [--dump-dir dir]\n       inference_poc --write-fixtures inference_poc/fixtures"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(Args {
        device,
        fixtures,
        kernel_dir,
        report,
        dump_dir,
    })
}
