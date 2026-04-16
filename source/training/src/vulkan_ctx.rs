use anyhow::{anyhow, bail, Context, Result};
use ash::{vk, Entry};
use std::ffi::{CStr, CString};

const F32: usize = 4;

pub struct GpuBuf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

struct GpuPipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

#[derive(Clone, Copy, Debug)]
pub struct Buf(pub usize);

#[derive(Clone, Copy, Debug)]
pub struct Pipe(pub usize);

pub struct VulkanTrainer {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    bufs: Vec<GpuBuf>,
    pipes: Vec<GpuPipe>,
}

impl VulkanTrainer {
    pub fn new(device_selector: &str) -> Result<Self> {
        unsafe { Self::init(device_selector) }
    }

    unsafe fn init(device_selector: &str) -> Result<Self> {
        let entry = Entry::load().context("loading Vulkan")?;
        let app = CString::new("spv-training")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app)
            .api_version(vk::API_VERSION_1_1);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )?;
        let physicals = instance.enumerate_physical_devices()?;
        let (physical, props, qfi) = select_device(&instance, &physicals, device_selector)?;
        let priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&priorities)];
        let device = instance.create_device(
            physical,
            &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
            None,
        )?;
        let queue = device.get_device_queue(qfi, 0);
        let bindings = [
            storage_binding(0),
            storage_binding(1),
            storage_binding(2),
        ];
        let desc_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let desc_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_size),
            None,
        )?;
        let layouts = [desc_layout];
        let desc_set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&layouts),
        )?[0];
        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(qfi)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let cmd_buf = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let mem_props = instance.get_physical_device_memory_properties(physical);
        let device_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        println!("[vulkan] device: {} (vendor 0x{:04x})", device_name, props.vendor_id);
        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            cmd_pool,
            cmd_buf,
            desc_layout,
            desc_pool,
            desc_set,
            mem_props,
            device_name,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            bufs: Vec::new(),
            pipes: Vec::new(),
        })
    }

    pub fn alloc(&mut self, elements: usize) -> Result<Buf> {
        let size = (elements * F32).max(4) as vk::DeviceSize;
        let buf = unsafe { alloc_buf(&self.device, &self.mem_props, size)? };
        let idx = self.bufs.len();
        self.bufs.push(buf);
        Ok(Buf(idx))
    }

    pub fn upload(&self, handle: Buf, data: &[f32]) -> Result<()> {
        let buf = &self.bufs[handle.0];
        let bytes = data.len() * F32;
        assert!(bytes as u64 <= buf.size, "upload overflow");
        unsafe {
            let ptr = self.device.map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), ptr.cast::<u8>(), bytes);
            self.device.unmap_memory(buf.memory);
        }
        Ok(())
    }

    pub fn download(&self, handle: Buf, elements: usize) -> Result<Vec<f32>> {
        let buf = &self.bufs[handle.0];
        let bytes = elements * F32;
        assert!(bytes as u64 <= buf.size);
        unsafe {
            let ptr = self.device.map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())?;
            let sl = std::slice::from_raw_parts(ptr.cast::<u8>(), bytes);
            let v = sl.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            self.device.unmap_memory(buf.memory);
            Ok(v)
        }
    }

    pub fn load_pipeline(&mut self, spv: &[u8], push_size: u32) -> Result<Pipe> {
        unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))
                .context("reading SPIR-V")?;
            let shader = self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&code),
                None,
            )?;
            let mut ranges = Vec::new();
            if push_size > 0 {
                ranges.push(
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .size(push_size),
                );
            }
            let layouts = [self.desc_layout];
            let layout = self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )?;
            let entry_name = CString::new("main")?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(&entry_name);
            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, e)| anyhow!("pipeline: {e:?}"))?[0];
            self.device.destroy_shader_module(shader, None);
            let idx = self.pipes.len();
            self.pipes.push(GpuPipe { pipeline, layout });
            Ok(Pipe(idx))
        }
    }

    pub fn dispatch(&self, pipe: Pipe, buffers: [Buf; 3], push: &[u8], groups_x: u32) -> Result<()> {
        unsafe {
            let p = &self.pipes[pipe.0];
            let b = [&self.bufs[buffers[0].0], &self.bufs[buffers[1].0], &self.bufs[buffers[2].0]];
            // update descriptor set
            let infos = [
                vk::DescriptorBufferInfo::default().buffer(b[0].buffer).offset(0).range(b[0].size),
                vk::DescriptorBufferInfo::default().buffer(b[1].buffer).offset(0).range(b[1].size),
                vk::DescriptorBufferInfo::default().buffer(b[2].buffer).offset(0).range(b[2].size),
            ];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.desc_set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[0])),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.desc_set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[1])),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.desc_set).dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[2])),
            ];
            self.device.update_descriptor_sets(&writes, &[]);
            // record
            self.device.reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(self.cmd_buf, &vk::CommandBufferBeginInfo::default())?;
            self.device.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.cmd_buf, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[self.desc_set], &[],
            );
            if !push.is_empty() {
                self.device.cmd_push_constants(
                    self.cmd_buf, p.layout, vk::ShaderStageFlags::COMPUTE, 0, push,
                );
            }
            self.device.cmd_dispatch(self.cmd_buf, groups_x, 1, 1);
            self.device.end_command_buffer(self.cmd_buf)?;
            let bufs = [self.cmd_buf];
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&bufs)],
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
        }
        Ok(())
    }
}

impl Drop for VulkanTrainer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            for p in self.pipes.drain(..) {
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_pipeline_layout(p.layout, None);
            }
            for b in self.bufs.drain(..) {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
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
        let ok = match selector {
            "nvidia" => props.vendor_id == 0x10de,
            "intel" => props.vendor_id == 0x8086,
            "any" => true,
            _ => bail!("unknown device selector: {selector}"),
        };
        if !ok { continue; }
        let qfi = instance
            .get_physical_device_queue_family_properties(*physical)
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32);
        if let Some(idx) = qfi {
            return Ok((*physical, props, idx));
        }
    }
    bail!("no Vulkan compute device for selector '{selector}'")
}

unsafe fn alloc_buf(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
) -> Result<GpuBuf> {
    let buffer = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let req = device.get_buffer_memory_requirements(buffer);
    let mti = find_memory_type(mem_props, req.memory_type_bits)?;
    let memory = device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mti),
        None,
    )?;
    device.bind_buffer_memory(buffer, memory, 0)?;
    // zero-init
    let ptr = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
    std::ptr::write_bytes(ptr.cast::<u8>(), 0, size as usize);
    device.unmap_memory(memory);
    Ok(GpuBuf { buffer, memory, size })
}

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32) -> Result<u32> {
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0 {
            let flags = props.memory_types[i as usize].property_flags;
            if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT) {
                return Ok(i);
            }
        }
    }
    bail!("no host-visible coherent memory type")
}
