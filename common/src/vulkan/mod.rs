#![cfg(feature = "vulkan")]

//! Vulkan compute backend.
//!
//! The GPU-portable sibling of [`crate::client_process_cuda`]. Same algorithm,
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

use crate::NiceNumberSimple;
use crate::gpu_niceonly::{GPU_LSD_K, RangeSink};
use crate::stride_filter::StrideTable;
use anyhow::{Context as _, Result, bail};
use ash::vk;
use codegen::{
    KernelConfig, MAX_LANES_PER_RANGE, MISS_STRIDE, NICE_STRIDE, NiceonlyConfig, WORKGROUP_SIZE,
    detailed_wgsl, lane_shift_for, niceonly_wgsl,
};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Upper bound on workgroups per dispatch. Threads grid-stride past this, so
/// it trades launch width against per-thread loop overhead rather than
/// limiting the batch.
const MAX_WORKGROUPS: u32 = 4096;

/// Nanoseconds to wait for a dispatch before giving up.
const FENCE_TIMEOUT_NS: u64 = 120_000_000_000;

/// Range descriptors per niceonly dispatch. The pipeline hands over batches of
/// `LAUNCH_BATCH_RANGES` plus whatever the last MSD chunk added, so this only
/// has to be comfortably larger than that; the device buffers are sized from
/// it and reused across every dispatch of a field.
const RANGES_PER_DISPATCH: usize = 1 << 17;

/// Capacity of the niceonly output buffer (in nice numbers) per field.
/// Genuinely nice numbers are astronomically rare; this is pure headroom.
const NICE_OUT_CAPACITY: usize = 1 << 16;

/// Push constant block for the detailed shader; must match its `struct Params`.
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

/// Push constant block for the niceonly shader; must match its `struct Params`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NiceonlyParams {
    fs0: u32,
    fs1: u32,
    fs2: u32,
    fs3: u32,
    fs_mod_m: u32,
    num_ranges: u32,
    nice_cap: u32,
    lane_shift: u32,
}

#[allow(clippy::cast_possible_truncation)] // 32 bytes; the const assert below pins it
const PUSH_CONSTANT_SIZE: u32 = std::mem::size_of::<Params>() as u32;
#[allow(clippy::cast_possible_truncation)]
const NICEONLY_PUSH_CONSTANT_SIZE: u32 = std::mem::size_of::<NiceonlyParams>() as u32;

// Vulkan only guarantees 128 bytes of push constant space.
const _: () = assert!(
    PUSH_CONSTANT_SIZE <= 128 && NICEONLY_PUSH_CONSTANT_SIZE <= 128,
    "push constant block exceeds the guaranteed minimum"
);

/// Lane tiling pinned by `NICE_VULKAN_LANES`, as a shift, or `None` to size it
/// per dispatch.
///
/// `NICE_VULKAN_LANES=32` reproduces the fixed one-warp-per-range tiling the
/// CUDA kernel uses and this backend used through Phase 2, which is how the
/// tiling A/B is taken — the same shape of knob as `NICE_GPU_MSD_FLOOR`. Values
/// that are not a power of two in `1..=MAX_LANES_PER_RANGE` are ignored with a
/// warning rather than silently rounded.
fn pinned_lane_shift() -> Option<u32> {
    let raw = std::env::var("NICE_VULKAN_LANES").ok()?;
    match raw.parse::<u32>() {
        Ok(n) if n.is_power_of_two() && n <= MAX_LANES_PER_RANGE => {
            info!("Vulkan niceonly lanes pinned at {n} via NICE_VULKAN_LANES");
            Some(n.trailing_zeros())
        }
        _ => {
            warn!("ignoring invalid NICE_VULKAN_LANES '{raw}'; sizing lanes per dispatch");
            None
        }
    }
}

/// Reinterpret a push constant block as bytes for `cmd_push_constants`.
///
/// Safety: `T` is a `#[repr(C)]` struct of `u32`s, so every byte is
/// initialized and there is no padding to leak.
unsafe fn push_bytes<T>(params: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(params).cast::<u8>(), size_of::<T>()) }
}

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

    fn as_mut_slice(&mut self) -> &mut [u32] {
        // Safety: as `as_slice`, and `&mut self` means no dispatch is reading
        // it — every dispatch is fence-waited before this call can be reached.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn zero(&self) {
        unsafe { std::ptr::write_bytes(self.ptr, 0, self.len) }
    }
}

/// A compiled compute pipeline and everything created alongside it.
struct Shader {
    module: vk::ShaderModule,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// A compiled detailed-mode pipeline for one base.
struct DetailedPipeline {
    cfg: KernelConfig,
    shader: Shader,
}

/// A compiled niceonly pipeline for one base, plus the base's residue table
/// living on the device.
///
/// The table is immutable and depends only on the base, so it is built once
/// with the pipeline and shared by every field — the same arrangement as the
/// CUDA path's `NiceonlyPlan`, which holds its residues in a `CudaSlice`.
struct NiceonlyPipeline {
    cfg: NiceonlyConfig,
    shader: Shader,
    residues: Buf,
}

// Safety: `Buf`'s raw pointer is what makes this !Send. The residue table is
// filled once, before the pipeline is published behind an `Arc`, and from then
// on the host never touches the mapping again — only the device reads it, and
// only `VulkanContext::drop` (which has exclusive access) unmaps it.
unsafe impl Send for NiceonlyPipeline {}
unsafe impl Sync for NiceonlyPipeline {}

/// Submissions the device may be working on before the host waits for one.
///
/// The host's job between dispatches is to fill the next batch's range
/// descriptors, so one slot in flight while another is being filled is all the
/// overlap there is to have; a third only adds latency to `sync`. Every slot
/// costs a command buffer, a fence, and — for niceonly — its own descriptor
/// buffers, which is why this is 2 and not 8.
const SUBMISSION_SLOTS: usize = 2;

/// One command buffer and its fence.
struct Slot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Submitted and not yet waited for: the fence needs a wait before this
    /// slot's command buffer or host-side buffers can be touched again.
    in_flight: bool,
}

/// The submission ring, serialized because queue submission needs external
/// synchronization.
///
/// Slots are handed out round-robin by [`VulkanContext::acquire_slot`], which
/// waits for the fence of the slot it is about to reuse — so the host blocks
/// only once the device is [`SUBMISSION_SLOTS`] dispatches behind, rather than
/// after every one.
struct Submitter {
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    next: usize,
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
    niceonly_pipelines: Mutex<HashMap<u32, Arc<NiceonlyPipeline>>>,
    /// Human-readable device name, for logging.
    pub device_name: String,
}

// Safety: every raw handle here is used only through `&self` methods that take
// the submitter mutex before touching the queue or command buffer; ash's Device
// and Instance are themselves Send + Sync.
unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

/// Create the Vulkan instance, opting into portability enumeration when the
/// loader offers it.
///
/// Portability drivers (`MoltenVK` on macOS) are hidden by loaders ≥ 1.3.207
/// unless the instance enables `VK_KHR_portability_enumeration`; a no-op on
/// conformant Linux/Windows stacks where the extension is absent.
fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app = vk::ApplicationInfo::default()
        .application_name(c"nice")
        .api_version(vk::make_api_version(0, 1, 2, 0));
    let portability = unsafe { entry.enumerate_instance_extension_properties(None) }
        .unwrap_or_default()
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_enumeration::NAME));
    let instance_exts = [ash::khr::portability_enumeration::NAME.as_ptr()];
    let mut instance_info = vk::InstanceCreateInfo::default().application_info(&app);
    if portability {
        instance_info = instance_info
            .enabled_extension_names(&instance_exts)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }
    unsafe { entry.create_instance(&instance_info, None) }.context("creating the Vulkan instance")
}

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
        let instance = create_instance(&entry)?;

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
        // The spec requires VK_KHR_portability_subset to be enabled whenever
        // the physical device advertises it (again: MoltenVK).
        let portability_subset =
            unsafe { instance.enumerate_device_extension_properties(physical) }
                .unwrap_or_default()
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_subset::NAME));
        let device_exts = [ash::khr::portability_subset::NAME.as_ptr()];
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_features(&feats);
        if portability_subset {
            device_info = device_info.enabled_extension_names(&device_exts);
        }
        let device = unsafe { instance.create_device(physical, &device_info, None) }
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
        #[allow(clippy::cast_possible_truncation)]
        let cmds = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(SUBMISSION_SLOTS as u32),
            )
        }
        .context("allocating the command buffers")?;
        let mut slots = Vec::with_capacity(SUBMISSION_SLOTS);
        for cmd in cmds {
            // Created unsignaled, matching `in_flight: false`.
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .context("creating the fence")?;
            slots.push(Slot {
                cmd,
                fence,
                in_flight: false,
            });
        }

        let ctx = Self {
            entry,
            instance,
            device,
            queue,
            mem_props,
            submitter: Mutex::new(Submitter {
                pool,
                slots,
                next: 0,
            }),
            pipelines: Mutex::new(HashMap::new()),
            niceonly_pipelines: Mutex::new(HashMap::new()),
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

    /// Compile generated WGSL and build the descriptor/pipeline objects for it.
    fn build_shader(&self, src: &str, num_bindings: u32, push_size: u32) -> Result<Shader> {
        let spirv = compile_wgsl(src)?;
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_bindings)
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
            .size(push_size)];
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
        debug!("Vulkan shader: {} SPIR-V words", spirv.len());
        Ok(Shader {
            module,
            dsl,
            layout,
            pipeline,
        })
    }

    fn destroy_shader(&self, shader: &Shader) {
        unsafe {
            self.device.destroy_pipeline(shader.pipeline, None);
            self.device.destroy_pipeline_layout(shader.layout, None);
            self.device.destroy_descriptor_set_layout(shader.dsl, None);
            self.device.destroy_shader_module(shader.module, None);
        }
    }

    /// Get or build the detailed-mode pipeline for a base.
    fn detailed_pipeline(&self, base: u32) -> Result<Arc<DetailedPipeline>> {
        if let Some(p) = self.pipelines.lock().unwrap().get(&base) {
            return Ok(p.clone());
        }
        let build = Instant::now();
        let cfg = KernelConfig::new(base)?;
        let shader = self
            .build_shader(&detailed_wgsl(&cfg), 3, PUSH_CONSTANT_SIZE)
            .with_context(|| format!("compiling the detailed shader for base {base}"))?;

        debug!(
            "Vulkan detailed pipeline for base {base}: chunk {} digits (div {}), built in {:.2}s",
            cfg.chunk_digits,
            cfg.chunk_div,
            build.elapsed().as_secs_f64()
        );
        let p = Arc::new(DetailedPipeline { cfg, shader });
        self.pipelines.lock().unwrap().insert(base, p.clone());
        Ok(p)
    }

    /// Get or build the niceonly pipeline and device residue table for a base.
    ///
    /// # Errors
    /// Returns an error for a residue-empty base — callers must short-circuit
    /// those before they get here (`StrideTable` panics when indexed with no
    /// valid residues), for bases the shader cannot configure, or on any
    /// Vulkan failure.
    fn niceonly_pipeline(&self, base: u32) -> Result<Arc<NiceonlyPipeline>> {
        if let Some(p) = self.niceonly_pipelines.lock().unwrap().get(&base) {
            return Ok(p.clone());
        }
        let build = Instant::now();
        let table = StrideTable::new(base, GPU_LSD_K);
        let cfg = NiceonlyConfig::new(KernelConfig::new(base)?, &table)?;
        let shader = self
            .build_shader(&niceonly_wgsl(&cfg), 5, NICEONLY_PUSH_CONSTANT_SIZE)
            .with_context(|| format!("compiling the niceonly shader for base {base}"))?;

        let mut residues = self
            .alloc_buf(table.valid_residues.len())
            .inspect_err(|_| {
                self.destroy_shader(&shader);
            })?;
        for (slot, &r) in residues
            .as_mut_slice()
            .iter_mut()
            .zip(&table.valid_residues)
        {
            *slot = r;
        }

        debug!(
            "Vulkan niceonly pipeline for base {base}: M={}, R={}, built in {:.2}s",
            cfg.stride_m,
            cfg.stride_r,
            build.elapsed().as_secs_f64()
        );
        let p = Arc::new(NiceonlyPipeline {
            cfg,
            shader,
            residues,
        });
        self.niceonly_pipelines
            .lock()
            .unwrap()
            .insert(base, p.clone());
        Ok(p)
    }

    /// Build an uncached niceonly pipeline that reports prefilter survivors
    /// instead of nice numbers (see [`codegen::niceonly_probe_wgsl`]).
    ///
    /// Uncached and not registered with the context, so the caller owns it and
    /// must pass it to [`Self::destroy_niceonly_pipeline`]; that keeps a
    /// test-only shader out of the map `Drop` walks.
    #[cfg(test)]
    fn niceonly_probe_pipeline(&self, base: u32) -> Result<Arc<NiceonlyPipeline>> {
        let table = StrideTable::new(base, GPU_LSD_K);
        let cfg = NiceonlyConfig::new(KernelConfig::new(base)?, &table)?;
        let shader = self.build_shader(
            &codegen::niceonly_probe_wgsl(&cfg),
            5,
            NICEONLY_PUSH_CONSTANT_SIZE,
        )?;
        let mut residues = self
            .alloc_buf(table.valid_residues.len())
            .inspect_err(|_| {
                self.destroy_shader(&shader);
            })?;
        for (slot, &r) in residues
            .as_mut_slice()
            .iter_mut()
            .zip(&table.valid_residues)
        {
            *slot = r;
        }
        Ok(Arc::new(NiceonlyPipeline {
            cfg,
            shader,
            residues,
        }))
    }

    #[cfg(test)]
    fn destroy_niceonly_pipeline(&self, pipe: &NiceonlyPipeline) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        self.destroy_shader(&pipe.shader);
        self.free_buf(&pipe.residues);
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

    /// Allocate a single descriptor set bound to `bufs` at bindings 0..n.
    fn make_descriptor_set(
        &self,
        dsl: vk::DescriptorSetLayout,
        bufs: &[&Buf],
    ) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(u32::try_from(bufs.len()).unwrap_or(u32::MAX))];
        let pool = unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&sizes)
                    .max_sets(1),
                None,
            )
        }?;
        let layouts = [dsl];
        let set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }?[0];

        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
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
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok((pool, set))
    }

    /// Reserve the next submission slot, waiting for its previous dispatch to
    /// finish if it has one.
    ///
    /// The slot is the caller's until it submits: whatever host-side buffers
    /// that slot owns are free to overwrite, because the device is done reading
    /// them. That is the contract that lets a caller double-buffer per slot.
    ///
    /// The returned guard holds the submitter lock across the caller's fill,
    /// which is what makes the reservation exclusive — otherwise a second
    /// thread could be handed the same slot while the first was still filling
    /// it, since nothing marks it taken until the submit. Holding it costs no
    /// device time: the lock blocks other *hosts*, and the device is meanwhile
    /// working through the slots already submitted.
    fn acquire_slot(&self) -> Result<SlotGuard<'_>> {
        let mut guard = self.submitter.lock().unwrap();
        let idx = guard.next;
        guard.next = (guard.next + 1) % guard.slots.len();
        if guard.slots[idx].in_flight {
            let fence = guard.slots[idx].fence;
            wait_fence(&self.device, fence)?;
            unsafe { self.device.reset_fences(&[fence]) }?;
            guard.slots[idx].in_flight = false;
        }
        Ok(SlotGuard {
            ctx: self,
            guard,
            idx,
        })
    }

    /// Record and submit one dispatch on a reserved slot, without waiting.
    fn submit_locked(
        &self,
        s: &mut Submitter,
        slot: usize,
        shader: &Shader,
        set: vk::DescriptorSet,
        push: &[u8],
        groups: u32,
    ) -> Result<()> {
        let (cmd, fence) = (s.slots[slot].cmd, s.slots[slot].fence);
        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, shader.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                shader.layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                shader.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
            self.device.cmd_dispatch(cmd, groups, 1, 1);
            self.device.end_command_buffer(cmd)?;

            let cmds = [cmd];
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                fence,
            )?;
        }
        s.slots[slot].in_flight = true;
        Ok(())
    }

    /// Wait for every outstanding submission.
    ///
    /// Callers must do this before reading any buffer a dispatch wrote — the
    /// device's results are not there until they do.
    fn wait_all(&self) -> Result<()> {
        let mut s = self.submitter.lock().unwrap();
        for i in 0..s.slots.len() {
            if s.slots[i].in_flight {
                let fence = s.slots[i].fence;
                wait_fence(&self.device, fence)?;
                unsafe { self.device.reset_fences(&[fence]) }?;
                s.slots[i].in_flight = false;
            }
        }
        Ok(())
    }

    /// Submit one dispatch and wait for it — the synchronous path, for callers
    /// that read the device's output straight afterwards.
    fn dispatch(
        &self,
        shader: &Shader,
        set: vk::DescriptorSet,
        push: &[u8],
        groups: u32,
    ) -> Result<()> {
        self.acquire_slot()?.submit(shader, set, push, groups)?;
        self.wait_all()
    }
}

/// An exclusive reservation of one submission slot.
///
/// Holds the submitter lock, so the slot — and the host-side buffers the caller
/// keeps for it — cannot be handed out again until this is submitted or
/// dropped.
struct SlotGuard<'a> {
    ctx: &'a VulkanContext,
    guard: std::sync::MutexGuard<'a, Submitter>,
    idx: usize,
}

impl SlotGuard<'_> {
    fn index(&self) -> usize {
        self.idx
    }

    /// Record and submit this slot's dispatch, releasing the reservation.
    fn submit(
        mut self,
        shader: &Shader,
        set: vk::DescriptorSet,
        push: &[u8],
        groups: u32,
    ) -> Result<()> {
        let (ctx, idx) = (self.ctx, self.idx);
        ctx.submit_locked(&mut self.guard, idx, shader, set, push, groups)
    }
}

/// Wait on one fence, turning the timeout into an error rather than a hang.
fn wait_fence(device: &ash::Device, fence: vk::Fence) -> Result<()> {
    match unsafe { device.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS) } {
        Ok(()) => Ok(()),
        Err(vk::Result::TIMEOUT) => bail!(
            "Vulkan dispatch timed out after {}s",
            FENCE_TIMEOUT_NS / 1_000_000_000
        ),
        Err(e) => Err(e.into()),
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for p in self.pipelines.lock().unwrap().values() {
                self.destroy_shader(&p.shader);
            }
            for p in self.niceonly_pipelines.lock().unwrap().values() {
                self.destroy_shader(&p.shader);
                self.free_buf(&p.residues);
            }
            let s = self.submitter.lock().unwrap();
            for slot in &s.slots {
                self.device.destroy_fence(slot.fence, None);
            }
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
        let (pool, set) =
            ctx.make_descriptor_set(pipe.shader.dsl, &[&hist, &miss_count, &miss_data])?;

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
        // Safety: `NiceonlyParams`/`Params` are `#[repr(C)]` u32 blocks.
        self.ctx.dispatch(
            &self.pipe.shader,
            self.set,
            unsafe { push_bytes(&params) },
            groups.max(1),
        )
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
        // Not `device_wait_idle`: see the note on `Drop for NiceonlyRun`. The
        // submitted work this has to outlive is its own dispatches, and those
        // are exactly the slots `wait_all` waits on.
        let _ = self.ctx.wait_all();
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
        }
        self.ctx.free_buf(&self.hist);
        self.ctx.free_buf(&self.miss_count);
        self.ctx.free_buf(&self.miss_data);
    }
}

/// The range descriptors for one in-flight dispatch, and the descriptor set
/// that points at them.
///
/// One per submission slot. The host fills slot `i`'s buffers while the device
/// is still reading slot `i-1`'s, which is the entire point of the ring: a
/// single shared pair of buffers would be safe only if `launch` waited, and
/// waiting is what Phase 3 is removing.
struct RangeSlot {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    offsets: Buf,
    lens: Buf,
}

/// One niceonly field in progress.
///
/// The output buffers persist across every dispatch of the field and are shared
/// by all slots — the kernel appends to them through an atomic counter, so
/// concurrent dispatches cannot collide. Only the *inputs* need one copy per
/// slot.
pub(crate) struct NiceonlyRun<'a> {
    ctx: &'a VulkanContext,
    pipe: Arc<NiceonlyPipeline>,
    slots: Vec<RangeSlot>,
    nice_out: Buf,
    nice_count: Buf,
    /// `field_start mod M`, so the shader only reduces the 64-bit offset.
    fs_mod_m: u32,
    field_start: u128,
    /// Lane tiling pinned by `NICE_VULKAN_LANES`, or `None` to size it per
    /// dispatch from the batch's ranges.
    lane_shift: Option<u32>,
}

impl<'a> NiceonlyRun<'a> {
    /// # Errors
    /// Returns an error for a residue-empty or unconfigurable base, or on any
    /// Vulkan failure.
    pub(crate) fn new(ctx: &'a VulkanContext, base: u32, field_start: u128) -> Result<Self> {
        Self::from_pipeline(ctx, ctx.niceonly_pipeline(base)?, field_start)
    }

    /// As [`Self::new`], with the pipeline supplied rather than looked up —
    /// the seam the probe build hangs off.
    fn from_pipeline(
        ctx: &'a VulkanContext,
        pipe: Arc<NiceonlyPipeline>,
        field_start: u128,
    ) -> Result<Self> {
        let nice_out = ctx.alloc_buf(NICE_OUT_CAPACITY * NICE_STRIDE as usize)?;
        let nice_count = ctx.alloc_buf(1)?;
        let mut slots = Vec::with_capacity(SUBMISSION_SLOTS);
        for _ in 0..SUBMISSION_SLOTS {
            let offsets = ctx.alloc_buf(RANGES_PER_DISPATCH * 2)?;
            let lens = ctx.alloc_buf(RANGES_PER_DISPATCH)?;
            let (pool, set) = ctx.make_descriptor_set(
                pipe.shader.dsl,
                &[&pipe.residues, &offsets, &lens, &nice_out, &nice_count],
            )?;
            slots.push(RangeSlot {
                pool,
                set,
                offsets,
                lens,
            });
        }
        #[allow(clippy::cast_possible_truncation)]
        let fs_mod_m = (field_start % u128::from(pipe.cfg.stride_m)) as u32;
        Ok(Self {
            ctx,
            pipe,
            slots,
            nice_out,
            nice_count,
            fs_mod_m,
            field_start,
            lane_shift: pinned_lane_shift(),
        })
    }

    /// Collect the nice numbers found across the whole field.
    ///
    /// Waits for the device first. The pipeline already calls [`Self::sync`],
    /// but a dispatch that is still in flight has simply not written its hits
    /// yet, and the failure mode is *silently missing solutions* — so the read
    /// path guarantees it rather than relying on the caller's ordering.
    ///
    /// # Errors
    /// Returns an error if the kernel tried to write more hits than the buffer
    /// holds — which, given how rare nice numbers are, means a kernel bug
    /// rather than a genuine flood.
    pub(crate) fn finish(&self) -> Result<Vec<NiceNumberSimple>> {
        self.ctx.wait_all()?;
        let written = self.nice_count.as_slice()[0] as usize;
        if written > NICE_OUT_CAPACITY {
            bail!(
                "niceonly output buffer overflow: {written} > {NICE_OUT_CAPACITY} \
                 (this strongly suggests a kernel bug)"
            );
        }
        let data = self.nice_out.as_slice();
        let stride = NICE_STRIDE as usize;
        let mut hits: Vec<NiceNumberSimple> = (0..written)
            .map(|i| {
                let o = i * stride;
                let lo = u128::from(data[o]) | (u128::from(data[o + 1]) << 32);
                let hi = u128::from(data[o + 2]) | (u128::from(data[o + 3]) << 32);
                NiceNumberSimple {
                    number: (hi << 64) | lo,
                    num_uniques: self.pipe.cfg.kernel.base,
                }
            })
            .collect();
        hits.sort_by_key(|n| n.number);
        Ok(hits)
    }

    pub(crate) fn config(&self) -> &NiceonlyConfig {
        &self.pipe.cfg
    }
}

impl RangeSink for NiceonlyRun<'_> {
    fn launch(&mut self, offsets: &[u64], lens: &[u32], masks: &[u64]) -> Result<()> {
        anyhow::ensure!(
            offsets.len() == lens.len() && offsets.len() == masks.len(),
            "range descriptor slices have mismatched lengths ({}/{}/{})",
            offsets.len(),
            lens.len(),
            masks.len()
        );
        let _ = masks; // certificates not yet applied on this backend
        // `masks` (cross-end certificates) are not yet applied on this
        // backend: WGSL has no u64, so the mask test needs a two-word port.
        // Ignoring them only means checking more candidates - never fewer.
        for (batch_offsets, batch_lens) in offsets
            .chunks(RANGES_PER_DISPATCH)
            .zip(lens.chunks(RANGES_PER_DISPATCH))
        {
            // Reserving the slot is what waits — and only once the device is
            // `SUBMISSION_SLOTS` dispatches behind. Until it returns, nothing
            // may touch this slot's buffers, because the device may still be
            // reading them. `ctx` is copied out of `self` first so the
            // reservation does not borrow the run that owns the buffers.
            let ctx = self.ctx;
            let reservation = ctx.acquire_slot()?;
            let slot_idx = reservation.index();
            let slot = &mut self.slots[slot_idx];

            #[allow(clippy::cast_possible_truncation)]
            {
                let dst = slot.offsets.as_mut_slice();
                for (i, &o) in batch_offsets.iter().enumerate() {
                    dst[2 * i] = o as u32;
                    dst[2 * i + 1] = (o >> 32) as u32;
                }
            }
            slot.lens.as_mut_slice()[..batch_lens.len()].copy_from_slice(batch_lens);

            // Tile the dispatch to this batch's ranges rather than to CUDA's
            // warp. Every lane on a range repeats that range's setup, so a
            // batch of short ranges wants few lanes and a batch of long ones
            // wants many; batches are homogeneous enough for the mean to be a
            // good summary, because the MSD recursion bounds range length by
            // the floor.
            let mean_len = batch_lens.iter().map(|&l| u64::from(l)).sum::<u64>()
                / batch_lens.len().max(1) as u64;
            let lane_shift = self.lane_shift.unwrap_or_else(|| {
                lane_shift_for(
                    batch_lens.len() as u64,
                    mean_len,
                    self.pipe.cfg.stride_m,
                    self.pipe.cfg.stride_r,
                )
            });

            #[allow(clippy::cast_possible_truncation)]
            let params = NiceonlyParams {
                fs0: self.field_start as u32,
                fs1: (self.field_start >> 32) as u32,
                fs2: (self.field_start >> 64) as u32,
                fs3: (self.field_start >> 96) as u32,
                fs_mod_m: self.fs_mod_m,
                num_ranges: u32::try_from(batch_offsets.len()).unwrap_or(u32::MAX),
                nice_cap: NICE_OUT_CAPACITY as u32,
                lane_shift,
            };
            let threads = (batch_offsets.len() as u64) << lane_shift;
            #[allow(clippy::cast_possible_truncation)]
            let groups = threads
                .div_ceil(u64::from(WORKGROUP_SIZE))
                .min(u64::from(MAX_WORKGROUPS)) as u32;
            // Safety: `NiceonlyParams` is a `#[repr(C)]` block of u32s.
            reservation.submit(
                &self.pipe.shader,
                self.slots[slot_idx].set,
                unsafe { push_bytes(&params) },
                groups.max(1),
            )?;
        }
        Ok(())
    }

    /// Wait for every dispatch of this field. The pipeline calls this inside
    /// its timed region, which is what keeps `device_secs` honest now that
    /// `launch` returns before the device is done.
    fn sync(&mut self) -> Result<()> {
        self.ctx.wait_all()
    }
}

impl Drop for NiceonlyRun<'_> {
    fn drop(&mut self) {
        // Wait on the submission slots, not on the whole device.
        //
        // `vkDeviceWaitIdle` is specified as a `vkQueueWaitIdle` on *every*
        // queue of the device, so it requires external synchronization against
        // all of them. Today that is free — one field is processed at a time and
        // only the pipeline's consumer thread submits — but nothing states that
        // invariant, and it is not the sort of thing a caller notices breaking:
        // racing a teardown wait against a live submission is undefined
        // behaviour, and `nice-count` measured its cost as whole dispatches
        // silently doing nothing rather than as a crash.
        //
        // `wait_all` needs no such invariant. It takes the submitter lock and
        // waits the fences of the slots actually in flight, which is a superset
        // of this run's own work and the only work that can still be reading the
        // descriptor sets and buffers freed below.
        let _ = self.ctx.wait_all();
        unsafe {
            for slot in &self.slots {
                self.ctx.device.destroy_descriptor_pool(slot.pool, None);
            }
        }
        for slot in &self.slots {
            self.ctx.free_buf(&slot.offsets);
            self.ctx.free_buf(&slot.lens);
        }
        self.ctx.free_buf(&self.nice_out);
        self.ctx.free_buf(&self.nice_count);
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

    /// Every supported base with a non-empty residue set must generate a
    /// niceonly shader that parses, validates and reaches SPIR-V.
    #[test]
    fn generated_niceonly_shaders_compile_for_all_supported_bases() {
        use crate::stride_filter::StrideTable;
        let mut n = 0;
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base)
                || crate::residue_filter::get_residue_filter_u128(&base).is_empty()
            {
                continue;
            }
            let table = StrideTable::new(base, crate::gpu_niceonly::GPU_LSD_K);
            let cfg = NiceonlyConfig::new(KernelConfig::new(base).expect("config"), &table)
                .expect("niceonly config");
            let src = niceonly_wgsl(&cfg);
            let spirv = compile_wgsl(&src)
                .unwrap_or_else(|e| panic!("base {base} failed to compile: {e}\n\n{src}"));
            assert!(spirv.len() > 100, "base {base}: suspiciously small SPIR-V");
            assert_eq!(spirv[0], 0x0723_0203, "base {base}: bad SPIR-V magic");
            n += 1;
        }
        assert!(n > 20, "only {n} bases compiled");
    }

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
        for field in [
            "s0", "s1", "s2", "s3", "cnt_lo", "cnt_hi", "miss_cap", "pad",
        ] {
            assert!(src.contains(field), "generated Params lacks {field}");
        }
        assert_eq!(PUSH_CONSTANT_SIZE as usize, std::mem::size_of::<Params>());
    }

    /// Same for the niceonly block. A field-order mismatch here would feed the
    /// kernel the wrong field start and silently search the wrong numbers.
    #[test]
    fn niceonly_push_constant_block_matches_the_generated_struct() {
        let table = StrideTable::new(40, GPU_LSD_K);
        let cfg = NiceonlyConfig::new(KernelConfig::new(40).unwrap(), &table).unwrap();
        let src = niceonly_wgsl(&cfg);
        for field in [
            "fs0",
            "fs1",
            "fs2",
            "fs3",
            "fs_mod_m",
            "num_ranges",
            "nice_cap",
            "lane_shift",
        ] {
            assert!(src.contains(field), "generated Params lacks {field}");
        }
        assert_eq!(
            NICEONLY_PUSH_CONSTANT_SIZE as usize,
            std::mem::size_of::<NiceonlyParams>()
        );
    }

    /// The prefilter's own verdicts, off the device, against the host mirror.
    ///
    /// The niceonly parity tests cannot see this filter: the only nice number
    /// any of them finds is 69, and base 10 has no prefilter, so a filter that
    /// rejected *every* candidate would pass all of them. That failure is not
    /// hypothetical — the CUDA kernel shipped it in v3.2.14. So the probe
    /// shader reports prefilter survivors directly and they must match the
    /// mirror candidate for candidate, both the passes and the rejections.
    ///
    /// Runs on lavapipe too; see `client_process_vulkan` for the invocation.
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn prefilter_survivors_match_the_host_mirror() {
        use crate::gpu_config::vulkan_prefilter_params;
        use crate::gpu_niceonly::RangeSink;

        let ctx = VulkanContext::new(0).expect("Vulkan init");
        // Base 40 is the live prefilter base; 30 and 34 exercise the same
        // codegen at other chunk/limb constants.
        for base in [30u32, 34, 40] {
            let pre = vulkan_prefilter_params(base).expect("base has a prefilter");
            let kernel = KernelConfig::new(base).unwrap();
            let table = StrideTable::new(base, GPU_LSD_K);
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let len: u32 = 5_000_000;

            let pipe = ctx.niceonly_probe_pipeline(base).expect("probe pipeline");
            assert!(
                pipe.cfg.prefilter.is_some(),
                "base {base}: probe built without the prefilter"
            );
            // Every lane width, over the identical range. The tiling is pure
            // index arithmetic — `gid.x >> shift` picks the range and
            // `gid.x & (lanes - 1)` the lane — so a width the host never
            // happens to choose is exactly where an off-by-one would hide.
            let mut per_width = Vec::new();
            for shift in 0..=MAX_LANES_PER_RANGE.ilog2() {
                let mut run =
                    NiceonlyRun::from_pipeline(&ctx, pipe.clone(), start).expect("probe run");
                run.lane_shift = Some(shift);
                run.launch(&[0], &[len], &[0]).expect("dispatch");
                per_width.push(
                    run.finish()
                        .expect("results")
                        .iter()
                        .map(|n| n.number)
                        .collect::<Vec<u128>>(),
                );
            }
            ctx.destroy_niceonly_pipeline(&pipe);

            // Every stride candidate in the range, filtered by the mirror.
            let end = start + u128::from(len);
            let (mut n, mut idx) = table.first_valid_at_or_after(start);
            let mut want = Vec::new();
            let mut candidates = 0u32;
            while n < end {
                candidates += 1;
                if codegen::mirror_prefilter(n, &kernel, &pre).0 {
                    want.push(n);
                }
                n += u128::from(table.gap_table[idx]);
                idx = (idx + 1) % table.gap_table.len();
            }

            // Each lane width against the CPU mirror, not merely against each
            // other: if the tiling drops or duplicates candidates at one
            // width, comparing widths only says they disagree, while this says
            // which one is wrong.
            for (shift, got) in per_width.iter().enumerate() {
                assert_eq!(
                    got,
                    &want,
                    "base {base}: {} lanes disagree with the CPU mirror \
                     ({} survivors vs {})",
                    1 << shift,
                    got.len(),
                    want.len()
                );
            }
            assert!(!want.is_empty(), "base {base}: the mirror passed nothing");
            assert!(
                want.len() < candidates as usize,
                "base {base}: the prefilter rejected nothing"
            );
            #[allow(clippy::cast_precision_loss)]
            {
                println!(
                    "base {base}: {} of {candidates} candidates survive ({:.2}%), device agrees",
                    want.len(),
                    100.0 * want.len() as f64 / f64::from(candidates)
                );
            }
        }
    }

    /// The niceonly shader indexes `nice_out` at `NICE_STRIDE * pos`, and the
    /// host decodes it at the same stride; both must fit the buffer the run
    /// allocates.
    #[test]
    fn nice_out_buffer_holds_its_capacity() {
        assert_eq!(NICE_STRIDE, 4, "a u128 hit is four u32 slots");
        assert!(u32::try_from(NICE_OUT_CAPACITY).is_ok());
    }
}
