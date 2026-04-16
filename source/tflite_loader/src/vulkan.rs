use crate::tensor::f32_bytes;
use anyhow::{anyhow, bail, Context, Result};
use ash::{vk, Entry};
use std::{
    ffi::{CStr, CString},
    time::Instant,
};

const F32_SIZE: usize = 4;

#[derive(Debug)]
struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KernelOutput {
    pub values: Vec<f32>,
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone)]
pub struct KernelRun<'a> {
    pub label: &'a str,
    pub spv: &'a [u8],
    pub input0: &'a [f32],
    pub input1: &'a [f32],
    pub output_elements: usize,
    pub dispatch_x: u32,
    pub push: &'a [i32],
}

pub fn groups(elements: usize, local_size: usize) -> u32 {
    elements.div_ceil(local_size) as u32
}

pub fn run_kernel(device_selector: &str, spec: &KernelRun<'_>) -> Result<KernelOutput> {
    unsafe { run_kernel_inner(device_selector, spec) }
}

unsafe fn run_kernel_inner(device_selector: &str, spec: &KernelRun<'_>) -> Result<KernelOutput> {
    let entry = Entry::load().context("loading Vulkan entry")?;
    let app = CString::new("tflite_loader")?;
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
    let mut ranges = Vec::new();
    if !spec.push.is_empty() {
        ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of_val(spec.push) as u32),
        );
    }
    let pipeline_layout = device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges),
        None,
    )?;
    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(spec.spv)).context("reading SPIR-V")?;
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
        .map_err(|(_, e)| anyhow!("creating compute pipeline for {}: {e:?}", spec.label))?[0];
    let memory_props = instance.get_physical_device_memory_properties(physical);
    let input0_buf = create_buffer(&device, &memory_props, f32_bytes(spec.input0))?;
    let input1_buf = create_buffer(&device, &memory_props, f32_bytes(spec.input1))?;
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
    if !spec.push.is_empty() {
        let push_bytes = std::slice::from_raw_parts(
            spec.push.as_ptr().cast::<u8>(),
            std::mem::size_of_val(spec.push),
        );
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_bytes,
        );
    }
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
    Ok(KernelOutput {
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
            "any" | "auto" => true,
            other => bail!("unknown --device {other}; expected intel|nvidia|any"),
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
    let size = initial.len().max(4) as vk::DeviceSize;
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
