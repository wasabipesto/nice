#![cfg(feature = "vulkan")]

//! Vulkan compute backend.
//!
//! The GPU-portable sibling of [`crate::client_process_gpu`]. Same algorithm,
//! same results; the difference is that kernels are generated as WGSL per base
//! (see [`codegen`]) and compiled to SPIR-V by `naga` at runtime, instead of
//! being compiled from CUDA C by NVRTC.
//!
//! Both `ash` and `cudarc` `dlopen` their driver at runtime, so a binary can
//! carry both backends and require neither at build time.
//!
//! # Why runtime shader generation rather than shipped SPIR-V
//!
//! The entire kernel design depends on every divisor being a compile-time
//! constant. Specialization constants would leave that to the driver's
//! post-specialization folding, which is exactly the thing that turned out not
//! to be reliable (see [`codegen`] on 64-bit division). Generating the source
//! keeps the guarantee in our hands, and matches what the CUDA path already
//! does with NVRTC.

pub mod codegen;

use anyhow::{Context as _, Result, bail};
use ash::vk;
use codegen::{KernelConfig, MISS_STRIDE, WORKGROUP_SIZE, detailed_wgsl};
use log::{debug, info};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Upper bound on workgroups per dispatch. Threads grid-stride past this, so
/// it trades launch width against per-thread loop overhead rather than
/// limiting the batch.
const MAX_WORKGROUPS: u32 = 4096;

/// Nanoseconds to wait for a dispatch before giving up.
const FENCE_TIMEOUT_NS: u64 = 120_000_000_000;

/// Push constant block; must match `struct Params` in the generated WGSL.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    cnt_lo: u32,
    cnt_hi: u32,
    miss_cap: u32,
    pad: u32,
}

#[allow(clippy::cast_possible_truncation)] // 32 bytes; the const assert below pins it
const PUSH_CONSTANT_SIZE: u32 = std::mem::size_of::<Params>() as u32;

// Vulkan only guarantees 128 bytes of push constant space.
const _: () = assert!(
    PUSH_CONSTANT_SIZE <= 128,
    "push constant block exceeds the guaranteed minimum"
);

/// A storage buffer with its memory, persistently mapped.
struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u32,
    len: usize,
}

impl Buf {
    fn as_slice(&self) -> &[u32] {
        // Safety: mapped HOST_COHERENT memory of exactly `len` u32s, and the
        // caller has waited on the fence for any dispatch writing it.
        unsafe { std::slice::from_raw_parts(self.ptr.cast_const(), self.len) }
    }

    fn zero(&self) {
        unsafe { std::ptr::write_bytes(self.ptr, 0, self.len) }
    }
}

/// A compiled detailed-mode pipeline for one base.
struct DetailedPipeline {
    cfg: KernelConfig,
    module: vk::ShaderModule,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// Command buffer + fence, serialized because queue submission needs external
/// synchronization.
struct Submitter {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

/// Vulkan device handle plus a cache of per-base compiled pipelines.
pub struct VulkanContext {
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    submitter: Mutex<Submitter>,
    pipelines: Mutex<HashMap<u32, Arc<DetailedPipeline>>>,
    /// Human-readable device name, for logging.
    pub device_name: String,
}

// Safety: every raw handle here is used only through `&self` methods that take
// the submitter mutex before touching the queue or command buffer; ash's Device
// and Instance are themselves Send + Sync.
unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

impl VulkanContext {
    /// Initialize Vulkan and verify that shader generation and pipeline
    /// creation work.
    ///
    /// `device_ordinal` indexes the compute-capable devices reported by the
    /// loader, in the order the loader reports them.
    ///
    /// # Errors
    /// Returns an error if the loader is missing, if no device supports
    /// compute with `shaderInt64`, or if a smoke-test shader fails to build.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let start = Instant::now();
        // Safety: loads libvulkan; no other thread is using the loader yet.
        let entry = unsafe { ash::Entry::load() }
            .context("loading the Vulkan loader (is libvulkan installed?)")?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"nice")
            .api_version(vk::make_api_version(0, 1, 2, 0));
        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
        }
        .context("creating the Vulkan instance")?;

        let selected = compute_devices(&instance).and_then(|candidates| {
            candidates.get(device_ordinal).cloned().with_context(|| {
                let names: Vec<&str> = candidates.iter().map(|(_, _, n)| n.as_str()).collect();
                format!(
                    "Vulkan device {device_ordinal} out of range; {} available: {names:?}",
                    candidates.len()
                )
            })
        });
        let (physical, queue_family, device_name) = match selected {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };

        let prios = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&prios)];
        let feats = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&qci)
                    .enabled_features(&feats),
                None,
            )
        }
        .context("creating the Vulkan logical device")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .context("creating the command pool")?;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .context("allocating the command buffer")?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .context("creating the fence")?;

        let ctx = Self {
            entry,
            instance,
            device,
            queue,
            mem_props,
            submitter: Mutex::new(Submitter { pool, cmd, fence }),
            pipelines: Mutex::new(HashMap::new()),
            device_name,
        };

        // Smoke test: build a real pipeline now so a broken driver or a naga
        // regression fails at startup with a clear error rather than mid-field.
        ctx.detailed_pipeline(10)
            .context("Vulkan smoke test failed (shader generation or pipeline creation)")?;
        info!(
            "Vulkan device {device_ordinal}: {} (init {:.2}s)",
            ctx.device_name,
            start.elapsed().as_secs_f64()
        );
        Ok(ctx)
    }

    /// Get or build the detailed-mode pipeline for a base.
    fn detailed_pipeline(&self, base: u32) -> Result<Arc<DetailedPipeline>> {
        if let Some(p) = self.pipelines.lock().unwrap().get(&base) {
            return Ok(p.clone());
        }
        let build = Instant::now();
        let cfg = KernelConfig::new(base)?;
        let src = detailed_wgsl(&cfg);
        let spirv = compile_wgsl(&src)
            .with_context(|| format!("compiling the detailed shader for base {base}"))?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl = unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_SIZE)];
        let layouts = [dsl];
        let layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        }?;
        let module = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
        }
        .context("driver rejected the generated SPIR-V")?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
        }
        .map_err(|(_, e)| e)
        .context("creating the compute pipeline")?[0];

        debug!(
            "Vulkan detailed pipeline for base {base}: {} SPIR-V words, chunk {} digits (div {}), built in {:.2}s",
            spirv.len(),
            cfg.chunk_digits,
            cfg.chunk_div,
            build.elapsed().as_secs_f64()
        );
        let p = Arc::new(DetailedPipeline {
            cfg,
            module,
            dsl,
            layout,
            pipeline,
        });
        self.pipelines.lock().unwrap().insert(base, p.clone());
        Ok(p)
    }

    /// Allocate a host-visible storage buffer of `len` u32s.
    fn alloc_buf(&self, len: usize) -> Result<Buf> {
        let size = (len * 4) as vk::DeviceSize;
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let need = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // Prefer device-local host-visible (iGPU, or ReBAR) so the shader's
        // atomics stay on the fast side of the bus; fall back to plain
        // host-visible. Traffic is tiny either way: the detailed kernel has no
        // input at all and near-misses are rare.
        let pick = |extra: vk::MemoryPropertyFlags| {
            (0..self.mem_props.memory_type_count).find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && self.mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(need | extra)
            })
        };
        let mem_type = pick(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| pick(vk::MemoryPropertyFlags::empty()))
            .context("no host-visible memory type for a storage buffer")?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }?;
        let ptr = unsafe {
            self.device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }?
        .cast::<u32>();
        let buf = Buf {
            buffer,
            memory,
            ptr,
            len,
        };
        buf.zero();
        Ok(buf)
    }

    fn free_buf(&self, buf: &Buf) {
        unsafe {
            self.device.unmap_memory(buf.memory);
            self.device.destroy_buffer(buf.buffer, None);
            self.device.free_memory(buf.memory, None);
        }
    }

    /// Record and submit one dispatch, waiting for it to complete.
    fn dispatch(
        &self,
        pipe: &DetailedPipeline,
        set: vk::DescriptorSet,
        params: Params,
        groups: u32,
    ) -> Result<()> {
        let s = self.submitter.lock().unwrap();
        unsafe {
            self.device
                .reset_command_buffer(s.cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                s.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.device
                .cmd_bind_pipeline(s.cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            self.device.cmd_bind_descriptor_sets(
                s.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            let bytes = std::slice::from_raw_parts(
                std::ptr::from_ref(&params).cast::<u8>(),
                PUSH_CONSTANT_SIZE as usize,
            );
            self.device.cmd_push_constants(
                s.cmd,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
            self.device.cmd_dispatch(s.cmd, groups, 1, 1);
            self.device.end_command_buffer(s.cmd)?;

            self.device.reset_fences(&[s.fence])?;
            let cmds = [s.cmd];
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                s.fence,
            )?;
            match self
                .device
                .wait_for_fences(&[s.fence], true, FENCE_TIMEOUT_NS)
            {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => bail!(
                    "Vulkan dispatch timed out after {}s",
                    FENCE_TIMEOUT_NS / 1_000_000_000
                ),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for p in self.pipelines.lock().unwrap().values() {
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_pipeline_layout(p.layout, None);
                self.device.destroy_descriptor_set_layout(p.dsl, None);
                self.device.destroy_shader_module(p.module, None);
            }
            let s = self.submitter.lock().unwrap();
            self.device.destroy_fence(s.fence, None);
            self.device.destroy_command_pool(s.pool, None);
            drop(s);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// One detailed-mode field in progress: the buffers persist across batches so
/// the histogram accumulates exactly as the CUDA path's does.
pub(crate) struct DetailedRun<'a> {
    ctx: &'a VulkanContext,
    pipe: Arc<DetailedPipeline>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    hist: Buf,
    miss_count: Buf,
    miss_data: Buf,
    miss_capacity: u32,
}

impl<'a> DetailedRun<'a> {
    pub(crate) fn new(ctx: &'a VulkanContext, base: u32, miss_capacity: usize) -> Result<Self> {
        let pipe = ctx.detailed_pipeline(base)?;
        let hist = ctx.alloc_buf(pipe.cfg.hist_bins() as usize)?;
        let miss_count = ctx.alloc_buf(1)?;
        let miss_data = ctx.alloc_buf(miss_capacity * MISS_STRIDE as usize)?;

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let pool = unsafe {
            ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&sizes)
                    .max_sets(1),
                None,
            )
        }?;
        let layouts = [pipe.dsl];
        let set = unsafe {
            ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }?[0];

        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [&hist, &miss_count, &miss_data]
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                #[allow(clippy::cast_possible_truncation)]
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            ctx,
            pipe,
            pool,
            set,
            hist,
            miss_count,
            miss_data,
            miss_capacity: u32::try_from(miss_capacity).unwrap_or(u32::MAX),
        })
    }

    /// Process `count` candidates starting at `start`.
    pub(crate) fn dispatch(&self, start: u128, count: u64) -> Result<()> {
        #[allow(clippy::cast_possible_truncation)]
        let params = Params {
            s0: start as u32,
            s1: (start >> 32) as u32,
            s2: (start >> 64) as u32,
            s3: (start >> 96) as u32,
            cnt_lo: count as u32,
            cnt_hi: (count >> 32) as u32,
            miss_cap: self.miss_capacity,
            pad: 0,
        };
        #[allow(clippy::cast_possible_truncation)]
        let groups = count
            .div_ceil(u64::from(WORKGROUP_SIZE))
            .min(u64::from(MAX_WORKGROUPS)) as u32;
        self.ctx
            .dispatch(&self.pipe, self.set, params, groups.max(1))
    }

    /// Add the device histogram into `acc` and reset it.
    ///
    /// The device bins are u32 and a field is far larger than u32 — a single
    /// base-40 field is ~1e12 candidates. Draining after every dispatch keeps
    /// each bin below the batch size (which is itself capped under 2^32) and
    /// lets the host accumulate at u128. This is why the Vulkan kernel needs
    /// no 64-bit atomics, unlike the CUDA one.
    pub(crate) fn drain_histogram(&self, acc: &mut [u128]) {
        for (a, &v) in acc.iter_mut().zip(self.hist.as_slice()) {
            *a += u128::from(v);
        }
        self.hist.zero();
    }

    /// Near-misses recorded so far, as `(n, num_uniques)`. Returns an error if
    /// the kernel tried to write more than the buffer holds.
    pub(crate) fn near_misses(&self) -> Result<Vec<(u128, u32)>> {
        let written = self.miss_count.as_slice()[0] as usize;
        if written > self.miss_capacity as usize {
            bail!(
                "near-miss buffer overflow: {written} > {}",
                self.miss_capacity
            );
        }
        let data = self.miss_data.as_slice();
        let stride = MISS_STRIDE as usize;
        Ok((0..written)
            .map(|i| {
                let o = i * stride;
                let lo = u128::from(data[o]) | (u128::from(data[o + 1]) << 32);
                let hi = u128::from(data[o + 2]) | (u128::from(data[o + 3]) << 32);
                ((hi << 64) | lo, data[o + 4])
            })
            .collect())
    }

    pub(crate) fn config(&self) -> &KernelConfig {
        &self.pipe.cfg
    }
}

impl Drop for DetailedRun<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
        }
        self.ctx.free_buf(&self.hist);
        self.ctx.free_buf(&self.miss_count);
        self.ctx.free_buf(&self.miss_data);
    }
}

/// Compute-capable devices with `shaderInt64`, as `(device, queue family, name)`.
///
/// Hardware sorts before software rasterizers: lavapipe is genuinely useful for
/// testing (it is the reason the parity tests can run without a GPU) but should
/// never be picked for throughput by default.
///
/// # Errors
/// Returns an error if enumeration fails or nothing qualifies.
fn compute_devices(instance: &ash::Instance) -> Result<Vec<(vk::PhysicalDevice, u32, String)>> {
    let mut candidates = Vec::new();
    for physical in
        unsafe { instance.enumerate_physical_devices() }.context("enumerating physical devices")?
    {
        let feats = unsafe { instance.get_physical_device_features(physical) };
        if feats.shader_int64 != vk::TRUE {
            continue;
        }
        let Some(qf) = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let software = props.device_type == vk::PhysicalDeviceType::CPU;
        candidates.push((physical, u32::try_from(qf).unwrap_or(0), name, software));
    }
    if candidates.is_empty() {
        bail!("no Vulkan device supports compute with shaderInt64");
    }
    candidates.sort_by_key(|&(_, _, _, software)| u8::from(software));
    Ok(candidates
        .into_iter()
        .map(|(p, q, n, _)| (p, q, n))
        .collect())
}

/// Compile generated WGSL to SPIR-V.
///
/// # Errors
/// Returns an error if the source fails to parse or validate, which for
/// generated source means a codegen bug.
pub fn compile_wgsl(src: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| anyhow::anyhow!("WGSL parse failed: {}", e.emit_to_string(src)))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        // IMMEDIATES is naga 30's name for the push-constant address space;
        // its SPIR-V backend maps it to StorageClass::PushConstant.
        naga::valid::Capabilities::SHADER_INT64 | naga::valid::Capabilities::IMMEDIATES,
    );
    let info = validator
        .validate(&module)
        .map_err(|e| anyhow::anyhow!("WGSL validation failed: {e:?}"))?;
    let mut opts = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    // The kernel indexes fixed-size private arrays whose bounds the generator
    // controls; the default Restrict policy would clamp every access in the
    // hot loop. This is what the CUDA path effectively gets.
    opts.bounds_check_policies = naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::Unchecked,
        buffer: naga::proc::BoundsCheckPolicy::Unchecked,
        image_load: naga::proc::BoundsCheckPolicy::Unchecked,
        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    naga::back::spv::write_vec(&module, &info, &opts, Some(&pipeline))
        .map_err(|e| anyhow::anyhow!("SPIR-V generation failed: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_config::{MAX_GPU_DIGIT_MASK_BASE, gpu_supports_base};

    /// Every supported base must generate WGSL that parses, validates and
    /// reaches SPIR-V. Needs no Vulkan device, so it runs everywhere.
    #[test]
    fn generated_shaders_compile_for_all_supported_bases() {
        let mut n = 0;
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            let cfg = KernelConfig::new(base).expect("config");
            let src = detailed_wgsl(&cfg);
            let spirv = compile_wgsl(&src)
                .unwrap_or_else(|e| panic!("base {base} failed to compile: {e}\n\n{src}"));
            assert!(spirv.len() > 100, "base {base}: suspiciously small SPIR-V");
            assert_eq!(spirv[0], 0x0723_0203, "base {base}: bad SPIR-V magic");
            n += 1;
        }
        assert!(n > 20, "only {n} bases compiled");
    }

    /// The generated `struct Params` must have exactly the fields the host
    /// pushes, in the same order — a mismatch would silently feed the kernel
    /// the wrong range.
    #[test]
    fn push_constant_block_matches_the_generated_struct() {
        let src = detailed_wgsl(&KernelConfig::new(40).unwrap());
        for field in ["s0", "s1", "s2", "s3", "cnt_lo", "cnt_hi", "miss_cap", "pad"] {
            assert!(src.contains(field), "generated Params lacks {field}");
        }
        assert_eq!(PUSH_CONSTANT_SIZE as usize, std::mem::size_of::<Params>());
    }
}
