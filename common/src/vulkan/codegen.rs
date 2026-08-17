//! Per-base WGSL generation for the Vulkan backend.
//!
//! This is the Vulkan analog of the CUDA path's NVRTC JIT: [`detailed_wgsl`]
//! emits a complete compute shader with every base-dependent value baked in as
//! a literal, so each division has a compile-time-constant divisor. The CUDA
//! source does the same thing with `-D` defines; here it is string generation,
//! because WGSL has no preprocessor.
//!
//! # Why the chunk split looks different from `nice_kernels.cu`
//!
//! The CUDA kernel splits a multi-limb value by `CHUNK_DIV` with a single
//! 64-bit division, relying on nvcc to strength-reduce it. RADV/ACO does that
//! for 32-bit constant divisors but **not** 64-bit ones — NIR's
//! `nir_opt_idiv_const` is width-limited, so `u64 / const` becomes a ~220
//! instruction restoring shift-subtract loop. So we keep `CHUNK_DIV < 2^16`
//! (see [`crate::gpu_config::chunk_constants_u16`]) and do the split as two
//! 32-bit constant divisions over 16-bit halves:
//!
//! ```text
//! cur = rem*2^32 + v,  with rem < d < 2^16
//! c1 = rem*2^16 + (v >> 16);      q1 = c1/d;  r1 = c1 % d      [c1 < 2^32]
//! c2 = r1*2^16  + (v & 0xffff);   q2 = c2/d;  rem' = c2 % d    [c2 < 2^32]
//! q  = q1*2^16 + q2
//! ```
//!
//! Exact, because `cur = d*(q1*2^16 + q2) + r2` and `q2 < 2^16`. Measured 2.85x
//! faster than the 64-bit form on base 40 despite needing ~1.7x more chunk
//! iterations.
//!
//! Digit semantics still match the CPU's `while n != 0 { d = n % b; n /= b; }`
//! exactly: interior chunks contribute all `CHUNK_DIGITS` digits including
//! zeros, the most significant chunk contributes digits only until it reaches
//! zero.

use crate::gpu_config::{
    VulkanPrefilterParams, chunk_constants_u16, n_limbs, vulkan_prefilter_params,
};
use crate::number_stats;
use crate::stride_filter::StrideTable;
use anyhow::{Context as _, Result, ensure};
use std::fmt::Write as _;

/// Threads per workgroup. The workgroup histogram is sized from this.
pub const WORKGROUP_SIZE: u32 = 256;

/// Copies of the histogram held in workgroup memory, to spread atomic
/// contention the way the CUDA kernel's per-warp histograms do. Worst case
/// (base 128) this is 4 * 129 * 4 = 2064 bytes against a 32 KB guaranteed
/// limit.
pub const HIST_COPIES: u32 = 4;

/// u32 slots per near-miss record: `n` as two u64 halves (4) plus its
/// unique-digit count. `n` is carried at full u128 width because bases above
/// ~68 have candidates past u64.
pub const MISS_STRIDE: u32 = 5;

/// u32 slots per niceonly hit: just `n` at u128 width. Unlike a near-miss it
/// carries no digit count — a hit is nice by definition, so `num_uniques` is
/// the base.
pub const NICE_STRIDE: u32 = 4;

// The niceonly tiling and offset-reduction constants are shared with the
// CubeCL backend and live in `gpu_niceonly`; re-exported here so existing
// paths keep working.
pub use crate::gpu_niceonly::{
    MAX_LANES_PER_RANGE, MAX_STRIDE_MODULUS, lane_shift_for, stride_chunk_bits,
};

/// Everything the generator needs for one base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConfig {
    pub base: u32,
    pub n_limbs: u32,
    pub chunk_digits: u32,
    pub chunk_div: u32,
    pub near_miss_cutoff: u32,
}

impl KernelConfig {
    /// Derive the configuration for a base.
    ///
    /// # Errors
    /// Returns an error for bases with no valid u128 search range.
    pub fn new(base: u32) -> Result<Self> {
        let n_limbs = n_limbs(base)
            .with_context(|| format!("base {base} has no valid u128 search range"))?;
        let (chunk_digits, chunk_div) = chunk_constants_u16(base);
        Ok(Self {
            base,
            n_limbs,
            chunk_digits,
            chunk_div,
            near_miss_cutoff: number_stats::get_near_miss_cutoff(base),
        })
    }

    /// Limbs in n², n³ respectively.
    #[must_use]
    pub fn sq_limbs(&self) -> u32 {
        2 * self.n_limbs
    }
    #[must_use]
    pub fn cu_limbs(&self) -> u32 {
        3 * self.n_limbs
    }
    /// Histogram bins: unique-digit counts 0..=base.
    #[must_use]
    pub fn hist_bins(&self) -> u32 {
        self.base + 1
    }
}

/// Fully-unrolled schoolbook multiply: `r[0..ra+rb] = a[0..ra] * b[0..rb]`,
/// with all operands as scalar locals named `{a}0..`, `{b}0..`, `{r}0..`.
///
/// Scalars rather than arrays so nothing can land in scratch memory; the limb
/// counts are known here, which is the whole point of generating per base.
fn emit_mul(s: &mut String, a: &str, ra: u32, b: &str, rb: u32, r: &str) {
    for i in 0..ra + rb {
        let _ = writeln!(s, "    var {r}{i}: u32 = 0u;");
    }
    for i in 0..ra {
        let _ = writeln!(s, "    var {r}c{i}: u64 = 0lu;");
        for j in 0..rb {
            let _ = writeln!(
                s,
                "    {{ let t: u64 = u64({a}{i}) * u64({b}{j}) + u64({r}{k}) + {r}c{i};\n\
                 \x20     {r}{k} = u32(t); {r}c{i} = t >> 32u; }}",
                k = i + j
            );
        }
        let _ = writeln!(s, "    {r}{k} = u32({r}c{i});", k = i + rb);
    }
}

/// Record digit `d` in the private digit mask.
///
/// With `track_dup`, also OR the bit's *previous* value into `dup`. That makes
/// duplicate detection pure bitwise and branch-free — the caller tests `dup`
/// once per chunk rather than once per digit, which is the form the CUDA kernel
/// measured at +13-14% whole-kernel over a per-digit early exit.
fn emit_digit_set(s: &mut String, cfg: &KernelConfig, indent: &str, track_dup: bool) {
    match (cfg.base <= 64, track_dup) {
        (true, false) => {
            let _ = writeln!(s, "{indent}m0 = m0 | (1lu << d);");
        }
        (true, true) => {
            let _ = writeln!(
                s,
                "{indent}{{ let bit: u64 = 1lu << d; dup = dup | (m0 & bit); m0 = m0 | bit; }}"
            );
        }
        (false, false) => {
            let _ = writeln!(
                s,
                "{indent}if (d < 64u) {{ m0 = m0 | (1lu << d); }} else {{ m1 = m1 | (1lu << (d - 64u)); }}"
            );
        }
        (false, true) => {
            let _ = writeln!(
                s,
                "{indent}if (d < 64u) {{ let bit: u64 = 1lu << d; dup = dup | (m0 & bit); m0 = m0 | bit; }}\n\
                 {indent}else {{ let bit: u64 = 1lu << (d - 64u); dup = dup | (m1 & bit); m1 = m1 | bit; }}"
            );
        }
    }
}

/// `fn top_limb(len) -> i32`: index of the highest nonzero limb of `sv`, or -1.
fn emit_top_limb(s: &mut String) {
    let _ = writeln!(
        s,
        "fn top_limb(len: i32) -> i32 {{\n\
         \x20   var t: i32 = len - 1;\n\
         \x20   loop {{\n\
         \x20       if (t < 0) {{ break; }}\n\
         \x20       if (sv[t] != 0u) {{ break; }}\n\
         \x20       t = t - 1;\n\
         \x20   }}\n\
         \x20   return t;\n\
         }}\n"
    );
}

/// The chunked radix scan over `sv`, destroying it.
///
/// With `stop_on_dup` the function is `fn {name}(top_in: i32) -> bool` and
/// returns false at the first chunk containing a repeat; otherwise it is
/// `fn {name}(top_in: i32)` and every digit is recorded.
fn emit_scan(s: &mut String, cfg: &KernelConfig, name: &str, stop_on_dup: bool) {
    let KernelConfig {
        base,
        chunk_digits,
        chunk_div,
        ..
    } = *cfg;
    let sig = if stop_on_dup {
        format!("fn {name}(top_in: i32) -> bool {{")
    } else {
        format!("fn {name}(top_in: i32) {{")
    };
    let _ = writeln!(
        s,
        "{sig}\n\
         \x20   var top: i32 = top_in;\n\
         \x20   loop {{\n\
         \x20       if (top < 0) {{ break; }}\n\
         \x20       // split16: rem = sv mod CHUNK_DIV, sv /= CHUNK_DIV\n\
         \x20       var rem: u32 = 0u;\n\
         \x20       var i: i32 = top;\n\
         \x20       loop {{\n\
         \x20           if (i < 0) {{ break; }}\n\
         \x20           let vi: u32 = sv[i];\n\
         \x20           let c1: u32 = (rem << 16u) | (vi >> 16u);\n\
         \x20           let q1: u32 = c1 / {chunk_div}u;\n\
         \x20           let c2: u32 = ((c1 % {chunk_div}u) << 16u) | (vi & 0xffffu);\n\
         \x20           let q2: u32 = c2 / {chunk_div}u;\n\
         \x20           rem = c2 % {chunk_div}u;\n\
         \x20           sv[i] = (q1 << 16u) | q2;\n\
         \x20           i = i - 1;\n\
         \x20       }}\n\
         \x20       loop {{\n\
         \x20           if (top < 0) {{ break; }}\n\
         \x20           if (sv[top] != 0u) {{ break; }}\n\
         \x20           top = top - 1;\n\
         \x20       }}\n\
         \x20       var chunk: u32 = rem;"
    );
    if stop_on_dup {
        s.push_str("        var dup: u64 = 0lu;\n");
    }
    // Interior chunk: exactly chunk_digits digits, zeros included.
    let _ = writeln!(
        s,
        "        if (top >= 0) {{\n\
         \x20           for (var k: u32 = 0u; k < {chunk_digits}u; k = k + 1u) {{\n\
         \x20               let d: u32 = chunk % {base}u;\n\
         \x20               chunk = chunk / {base}u;"
    );
    emit_digit_set(s, cfg, "                ", stop_on_dup);
    let _ = writeln!(
        s,
        "            }}\n\
         \x20       }} else {{\n\
         \x20           // Most significant chunk: digits until it reaches zero.\n\
         \x20           loop {{\n\
         \x20               if (chunk == 0u) {{ break; }}\n\
         \x20               let d: u32 = chunk % {base}u;\n\
         \x20               chunk = chunk / {base}u;"
    );
    emit_digit_set(s, cfg, "                ", stop_on_dup);
    s.push_str("            }\n        }\n");
    if stop_on_dup {
        s.push_str("        if (dup != 0lu) { return false; }\n    }\n    return true;\n}\n\n");
    } else {
        s.push_str("    }\n}\n\n");
    }
}

/// Unpack `n = (n_lo, n_hi)` into the scalar limbs `n0..`.
fn emit_unpack_n(s: &mut String, n_limbs: u32) {
    for i in 0..n_limbs {
        let word = if i < 2 { "n_lo" } else { "n_hi" };
        let shift = (i & 1) * 32;
        let _ = writeln!(s, "    let n{i}: u32 = u32({word} >> {shift}u);");
    }
}

/// Clear the private digit mask.
fn emit_mask_clear(s: &mut String, cfg: &KernelConfig) {
    s.push_str("    m0 = 0lu;\n");
    if cfg.base > 64 {
        s.push_str("    m1 = 0lu;\n");
    }
}

/// Expression counting the set bits of the digit mask.
fn popcount_expr(cfg: &KernelConfig) -> &'static str {
    if cfg.base <= 64 {
        "countOneBits(u32(m0)) + countOneBits(u32(m0 >> 32u))"
    } else {
        "countOneBits(u32(m0)) + countOneBits(u32(m0 >> 32u))\n         + countOneBits(u32(m1)) + countOneBits(u32(m1 >> 32u))"
    }
}

/// The complete detailed-mode compute shader for one base.
///
/// Long by line count because it *is* the kernel — splitting it would scatter
/// one shader across several functions and make the generated source harder to
/// follow than the CUDA original it mirrors.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn detailed_wgsl(cfg: &KernelConfig) -> String {
    let KernelConfig {
        base,
        n_limbs,
        chunk_digits,
        chunk_div,
        near_miss_cutoff,
    } = *cfg;
    let cu_limbs = cfg.cu_limbs();
    let sq_limbs = cfg.sq_limbs();
    let hist_bins = cfg.hist_bins();
    let wg = WORKGROUP_SIZE;
    let copies = HIST_COPIES;

    let mut s = String::with_capacity(8192);
    let _ = writeln!(
        s,
        "// Generated for base {base}: n_limbs={n_limbs}, chunk={chunk_digits} digits\n\
         // (div {chunk_div}), near-miss cutoff {near_miss_cutoff}. Do not edit; see\n\
         // nice_common::vulkan::codegen.\n\n\
         struct Params {{\n\
         \x20   s0: u32, s1: u32, s2: u32, s3: u32,   // range start, u128 as 4 limbs\n\
         \x20   cnt_lo: u32, cnt_hi: u32,             // candidates in this dispatch\n\
         \x20   miss_cap: u32,\n\
         \x20   pad: u32,\n\
         }}\n\
         var<immediate> pc: Params;\n\n\
         @group(0) @binding(0) var<storage, read_write> hist: array<atomic<u32>>;\n\
         @group(0) @binding(1) var<storage, read_write> miss_count: array<atomic<u32>>;\n\
         @group(0) @binding(2) var<storage, read_write> miss_data: array<u32>;\n\n\
         var<workgroup> hist_s: array<atomic<u32>, {}>;\n\
         var<private> sv: array<u32, {cu_limbs}>;\n\
         var<private> m0: u64;",
        copies * hist_bins
    );
    if base > 64 {
        let _ = writeln!(s, "var<private> m1: u64;");
    }
    s.push('\n');

    emit_top_limb(&mut s);
    // No early exit: detailed mode counts every digit.
    emit_scan(&mut s, cfg, "scan", false);

    // --- num_unique ---------------------------------------------------------
    s.push_str("fn num_unique(n_lo: u64, n_hi: u64) -> u32 {\n");
    emit_unpack_n(&mut s, n_limbs);
    emit_mul(&mut s, "n", n_limbs, "n", n_limbs, "sq");
    emit_mul(&mut s, "sq", sq_limbs, "n", n_limbs, "cu");
    emit_mask_clear(&mut s, cfg);
    for i in 0..sq_limbs {
        let _ = writeln!(s, "    sv[{i}] = sq{i};");
    }
    for i in sq_limbs..cu_limbs {
        let _ = writeln!(s, "    sv[{i}] = 0u;");
    }
    let _ = writeln!(s, "    scan(top_limb({sq_limbs}));");
    for i in 0..cu_limbs {
        let _ = writeln!(s, "    sv[{i}] = cu{i};");
    }
    let _ = writeln!(s, "    scan(top_limb({cu_limbs}));");
    let _ = writeln!(s, "    return {};\n}}\n", popcount_expr(cfg));

    // --- entry point --------------------------------------------------------
    let _ = writeln!(
        s,
        "@compute @workgroup_size({wg})\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n\
         \x20       @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20       @builtin(num_workgroups) nwg: vec3<u32>) {{\n\
         \x20   for (var i: u32 = lid.x; i < {}u; i = i + {wg}u) {{\n\
         \x20       atomicStore(&hist_s[i], 0u);\n\
         \x20   }}\n\
         \x20   workgroupBarrier();\n\n\
         \x20   let start_lo: u64 = (u64(pc.s1) << 32u) | u64(pc.s0);\n\
         \x20   let start_hi: u64 = (u64(pc.s3) << 32u) | u64(pc.s2);\n\
         \x20   let count: u64 = (u64(pc.cnt_hi) << 32u) | u64(pc.cnt_lo);\n\
         \x20   let copy: u32 = (lid.x >> 5u) % {copies}u;\n\
         \x20   let stride: u64 = u64(nwg.x) * {wg}lu;\n\n\
         \x20   var idx: u64 = u64(gid.x);\n\
         \x20   loop {{\n\
         \x20       if (idx >= count) {{ break; }}\n\
         \x20       let n_lo: u64 = start_lo + idx;\n\
         \x20       var n_hi: u64 = start_hi;\n\
         \x20       if (n_lo < start_lo) {{ n_hi = n_hi + 1lu; }}\n\
         \x20       let u: u32 = num_unique(n_lo, n_hi);\n\
         \x20       atomicAdd(&hist_s[copy * {hist_bins}u + u], 1u);\n\
         \x20       if (u > {near_miss_cutoff}u) {{\n\
         \x20           let pos: u32 = atomicAdd(&miss_count[0], 1u);\n\
         \x20           if (pos < pc.miss_cap) {{\n\
         \x20               let o: u32 = {MISS_STRIDE}u * pos;\n\
         \x20               miss_data[o] = u32(n_lo);\n\
         \x20               miss_data[o + 1u] = u32(n_lo >> 32u);\n\
         \x20               miss_data[o + 2u] = u32(n_hi);\n\
         \x20               miss_data[o + 3u] = u32(n_hi >> 32u);\n\
         \x20               miss_data[o + 4u] = u;\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20       idx = idx + stride;\n\
         \x20   }}\n\n\
         \x20   workgroupBarrier();\n\
         \x20   for (var b: u32 = lid.x; b < {hist_bins}u; b = b + {wg}u) {{\n\
         \x20       var acc: u32 = 0u;\n\
         \x20       for (var c: u32 = 0u; c < {copies}u; c = c + 1u) {{\n\
         \x20           acc = acc + atomicLoad(&hist_s[c * {hist_bins}u + b]);\n\
         \x20       }}\n\
         \x20       if (acc != 0u) {{ atomicAdd(&hist[b], acc); }}\n\
         \x20   }}\n\
         }}",
        copies * hist_bins
    );

    s
}

// ============================================================================
// Niceonly
// ============================================================================

/// Everything the niceonly generator needs beyond the shared per-base config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NiceonlyConfig {
    pub kernel: KernelConfig,
    /// Stride modulus `M = (b-1)·b^k`.
    pub stride_m: u32,
    /// Number of valid residues mod `M`.
    pub stride_r: u32,
    /// Low-digit prefilter, when the base both allows and wants one.
    pub prefilter: Option<VulkanPrefilterParams>,
}

impl NiceonlyConfig {
    /// Derive the niceonly configuration for a base from its stride table.
    ///
    /// # Errors
    /// Returns an error for a residue-empty base (nothing to search — the
    /// caller must short-circuit those before ever building a table), or if the
    /// modulus is too large for the shader's byte-wise reduction
    /// (see [`MAX_STRIDE_MODULUS`]).
    pub fn new(kernel: KernelConfig, table: &StrideTable) -> Result<Self> {
        let base = kernel.base;
        ensure!(
            !table.valid_residues.is_empty(),
            "no valid stride residues for base {base} (residue-empty base?)"
        );
        ensure!(
            table.modulus <= MAX_STRIDE_MODULUS,
            "stride modulus {} exceeds the shader's {MAX_STRIDE_MODULUS} bound for base {base}",
            table.modulus
        );
        #[allow(clippy::cast_possible_truncation)]
        Ok(Self {
            kernel,
            stride_m: table.modulus as u32,
            stride_r: u32::try_from(table.valid_residues.len())
                .with_context(|| format!("residue count overflows u32 for base {base}"))?,
            prefilter: prefilter_enabled().then(|| vulkan_prefilter_params(base)).flatten(),
        })
    }
}

/// Whether the generator may emit the low-digit prefilter at all.
///
/// `NICE_VULKAN_PREFILTER=0` compiles it out everywhere, which is how the
/// with/against measurement is taken — the same shape of knob as
/// `NICE_GPU_MSD_FLOOR`. Anything else (including unset) leaves the per-base
/// decision to [`vulkan_prefilter_params`].
fn prefilter_enabled() -> bool {
    prefilter_enabled_for(std::env::var("NICE_VULKAN_PREFILTER").ok().as_deref())
}

/// The knob's parsing, separated from reading the environment so it can be
/// tested without a process-wide mutation.
fn prefilter_enabled_for(value: Option<&str>) -> bool {
    !matches!(value, Some("0" | "off" | "false"))
}

/// Reduce a u64 range offset mod `M` on the host, the way the shader does.
///
/// Exposed so the mirror test can check the chunked Horner against `%`
/// directly; the shader emits exactly this, unrolled.
#[must_use]
pub fn offset_mod_m(offset: u64, stride_m: u32) -> u32 {
    let c = stride_chunk_bits(stride_m);
    let mask = (1u64 << c) - 1;
    let mut acc: u32 = 0;
    for i in (0..64 / c).rev() {
        #[allow(clippy::cast_possible_truncation)]
        let chunk = ((offset >> (i * c)) & mask) as u32;
        acc = ((acc << c) | chunk) % stride_m;
    }
    acc
}

/// One `split16` step over the scalar limbs `{v}0..{v}{n_limbs}`: divides that
/// value by `chunk_div` in place and leaves the remainder — one chunk of
/// `chunk_digits` base-`b` digits — in `rem`.
///
/// The same arithmetic [`emit_scan`] uses, minus the `top` tracking: the
/// prefilter peels a fixed number of chunks off a value it never has to scan to
/// the end of, so there is nothing to skip and the fixed trip count keeps the
/// whole filter branch-free.
fn emit_split_step(s: &mut String, v: &str, n_limbs: u32, chunk_div: u32, indent: &str) {
    let _ = writeln!(s, "{indent}var rem: u32 = 0u;");
    for i in (0..n_limbs).rev() {
        let _ = writeln!(
            s,
            "{indent}{{ let vi: u32 = {v}{i};\n\
             {indent}  let c1: u32 = (rem << 16u) | (vi >> 16u);\n\
             {indent}  let q1: u32 = c1 / {chunk_div}u;\n\
             {indent}  let c2: u32 = ((c1 % {chunk_div}u) << 16u) | (vi & 0xffffu);\n\
             {indent}  rem = c2 % {chunk_div}u;\n\
             {indent}  {v}{i} = (q1 << 16u) | (c2 / {chunk_div}u); }}"
        );
    }
}

/// `{out}[0..limbs] = {x} * {y}` truncated to `limbs` chunks of base `div`.
///
/// Schoolbook, dropping every product and carry that lands at or above chunk
/// `limbs` — which is exactly reduction mod `div^limbs`, since those only ever
/// feed positions further up. The accumulator is a `u32` and stays one: every
/// operand is below `div < 2^16`, so the step is at most `div² - 1 < 2^32`
/// (pinned by `prefilter_multiply_cannot_overflow_a_u32`). That is the whole
/// point of the chunk representation — `%` and `/` here name a 32-bit constant,
/// which RADV strength-reduces, where the CUDA prefilter's u64 modulus would
/// not.
fn emit_mulmod_chunks(s: &mut String, out: &str, x: &str, y: &str, limbs: u32, div: u32) {
    for i in 0..limbs {
        let _ = writeln!(s, "    var {out}{i}: u32 = 0u;");
    }
    for i in 0..limbs {
        let _ = writeln!(s, "    {{ var carry: u32 = 0u;");
        for j in 0..limbs - i {
            let _ = writeln!(
                s,
                "        {{ let t: u32 = {x}{i} * {y}{j} + {out}{k} + carry;\n\
                 \x20         {out}{k} = t % {div}u; carry = t / {div}u; }}",
                k = i + j
            );
        }
        s.push_str("    }\n");
    }
}

/// `fn prefilter(n_lo, n_hi) -> bool`: the low-digit modular prefilter.
///
/// Checks whether the lowest `digits` base-`b` digits of n² and of n³ are all
/// distinct, using `x^k mod b^p == (x mod b^p)^k mod b^p` so no multi-limb
/// value is ever formed. A repeat there means the candidate cannot be nice, and
/// the caller skips the full check.
///
/// Fixed-length and branch-free, which is the point: lanes only save work when
/// their *whole* group is killed, so a uniform-cost filter that kills ~98% of
/// candidates beats a cheaper divergent one. Survivors pay for these digits
/// twice, since `check_is_nice` recomputes them from scratch — amortized to
/// noise by the kill rate, and it keeps `check_is_nice` untouched.
///
/// Soundness rests on n² and n³ really having `digits` digits everywhere in the
/// base's range; `gpu_config::vulkan_prefilter_params` returns `None` when they
/// might not, and there is deliberately no fallback that could turn the filter
/// on behind the generator's back (the CUDA path shipped that bug once — see
/// `prefilter_has_no_ifndef_fallback`).
fn emit_prefilter(s: &mut String, cfg: &KernelConfig, pre: &VulkanPrefilterParams) {
    let KernelConfig { base, n_limbs, .. } = *cfg;
    let VulkanPrefilterParams {
        limbs,
        chunk_digits,
        chunk_div,
        ..
    } = *pre;

    s.push_str("fn prefilter(n_lo: u64, n_hi: u64) -> bool {\n");
    for i in 0..n_limbs {
        let word = if i < 2 { "n_lo" } else { "n_hi" };
        let shift = (i & 1) * 32;
        let _ = writeln!(s, "    var v{i}: u32 = u32({word} >> {shift}u);");
    }
    // n mod chunk_div^limbs, low chunk first — the same repeated split the
    // digit scan does, stopped after `limbs` chunks.
    for r in 0..limbs {
        let _ = writeln!(s, "    var a{r}: u32 = 0u;");
        s.push_str("    {\n");
        emit_split_step(s, "v", n_limbs, chunk_div, "        ");
        let _ = writeln!(s, "        a{r} = rem;\n    }}");
    }
    emit_mulmod_chunks(s, "sq", "a", "a", limbs, chunk_div);
    emit_mulmod_chunks(s, "cu", "sq", "a", limbs, chunk_div);

    emit_mask_clear(s, cfg);
    s.push_str("    var dup: u64 = 0lu;\n");
    // Each chunk holds exactly `chunk_digits` digits, leading zeros included —
    // which is what the digit-count guarantee buys, and why they are real
    // digits of the value rather than padding.
    for (name, count) in [("sq", limbs), ("cu", limbs)] {
        for i in 0..count {
            let _ = writeln!(
                s,
                "    {{ var c: u32 = {name}{i};\n\
                 \x20     for (var k: u32 = 0u; k < {chunk_digits}u; k = k + 1u) {{\n\
                 \x20         let d: u32 = c % {base}u;\n\
                 \x20         c = c / {base}u;"
            );
            emit_digit_set(s, cfg, "            ", true);
            s.push_str("        }\n    }\n");
        }
    }
    s.push_str("    return dup == 0lu;\n}\n\n");
}

/// Rust mirror of the generated `prefilter`, chunk for chunk: the same
/// repeated `split16` over n's u32 limbs, the same truncated schoolbook
/// multiply, the same digit peel. Returns the verdict plus the n², n³
/// chunks so a test can check the arithmetic and not just the answer.
///
/// Deliberately written in `u32` rather than a wider accumulator: in a
/// debug build that makes every step of the shader's "this cannot overflow
/// a u32" argument a checked assertion, and the tests run in debug.
#[cfg(test)]
pub(crate) fn mirror_prefilter(
    n: u128,
    cfg: &KernelConfig,
    pre: &VulkanPrefilterParams,
) -> (bool, Vec<u32>, Vec<u32>) {
    let div = pre.chunk_div;
    let split_step = |v: &mut Vec<u32>| -> u32 {
        let mut rem: u32 = 0;
        for i in (0..v.len()).rev() {
            let vi = v[i];
            let c1 = (rem << 16) | (vi >> 16);
            let q1 = c1 / div;
            let c2 = ((c1 % div) << 16) | (vi & 0xffff);
            rem = c2 % div;
            v[i] = (q1 << 16) | (c2 / div);
        }
        rem
    };
    let mulmod = |x: &[u32], y: &[u32]| -> Vec<u32> {
        let limbs = pre.limbs as usize;
        let mut out = vec![0u32; limbs];
        for i in 0..limbs {
            let mut carry: u32 = 0;
            for j in 0..limbs - i {
                let acc = x[i] * y[j] + out[i + j] + carry;
                out[i + j] = acc % div;
                carry = acc / div;
            }
        }
        out
    };

    #[allow(clippy::cast_possible_truncation)]
    let mut v: Vec<u32> = (0..cfg.n_limbs).map(|i| (n >> (32 * i)) as u32).collect();
    let nm: Vec<u32> = (0..pre.limbs).map(|_| split_step(&mut v)).collect();
    let sq = mulmod(&nm, &nm);
    let cu = mulmod(&sq, &nm);

    let mut seen = [false; 128];
    let mut dup = false;
    for chunk in sq.iter().chain(&cu) {
        let mut rest = *chunk;
        for _ in 0..pre.chunk_digits {
            let digit = (rest % cfg.base) as usize;
            rest /= cfg.base;
            dup |= seen[digit];
            seen[digit] = true;
        }
    }
    (!dup, sq, cu)
}

/// The complete niceonly compute shader for one base.
///
/// Mirrors `niceonly_ranges_kernel` in `nice_kernels.cu`: a host-chosen group
/// of threads per MSD-valid range (see [`lane_shift_for`]), candidates
/// reconstructed on-device from the residue table, so the host ships ~12 bytes
/// per range and no per-candidate data at all.
#[must_use]
pub fn niceonly_wgsl(cfg: &NiceonlyConfig) -> String {
    niceonly_wgsl_impl(cfg, false)
}

/// The same shader with the full check removed, so it reports every candidate
/// the *prefilter* passes.
///
/// The device tests otherwise cannot see this filter at all. Nice numbers are
/// astronomically rare, so a prefilter that wrongly rejected every candidate
/// would agree with the CPU on every base the parity test can afford to run —
/// and rejecting everything is not a hypothetical, it is the bug the CUDA path
/// shipped in v3.2.14. This turns the filter's own output into something a test
/// can compare against [`mirror_prefilter`].
#[cfg(test)]
#[must_use]
pub(crate) fn niceonly_probe_wgsl(cfg: &NiceonlyConfig) -> String {
    niceonly_wgsl_impl(cfg, true)
}

#[must_use]
#[allow(clippy::too_many_lines)]
fn niceonly_wgsl_impl(cfg: &NiceonlyConfig, probe: bool) -> String {
    let NiceonlyConfig {
        kernel,
        stride_m,
        stride_r,
        prefilter,
    } = *cfg;
    let KernelConfig {
        base,
        n_limbs,
        chunk_digits,
        chunk_div,
        ..
    } = kernel;
    let cu_limbs = kernel.cu_limbs();
    let sq_limbs = kernel.sq_limbs();
    let wg = WORKGROUP_SIZE;

    let mut s = String::with_capacity(8192);
    let _ = writeln!(
        s,
        "// Generated for base {base}: n_limbs={n_limbs}, chunk={chunk_digits} digits\n\
         // (div {chunk_div}), stride M={stride_m} with R={stride_r} residues. Do not\n\
         // edit; see nice_common::vulkan::codegen.\n\n\
         struct Params {{\n\
         \x20   fs0: u32, fs1: u32, fs2: u32, fs3: u32,  // field start, u128 as 4 limbs\n\
         \x20   fs_mod_m: u32,                           // field start mod M, host-computed\n\
         \x20   num_ranges: u32,\n\
         \x20   nice_cap: u32,\n\
         \x20   lane_shift: u32,                         // log2 of the lanes per range\n\
         }}\n\
         var<immediate> pc: Params;\n\n\
         @group(0) @binding(0) var<storage, read> residues: array<u32>;\n\
         @group(0) @binding(1) var<storage, read> range_offsets: array<u32>; // lo, hi pairs\n\
         @group(0) @binding(2) var<storage, read> range_lens: array<u32>;\n\
         @group(0) @binding(3) var<storage, read_write> nice_out: array<u32>;\n\
         @group(0) @binding(4) var<storage, read_write> nice_count: array<atomic<u32>>;\n\n\
         var<private> sv: array<u32, {cu_limbs}>;\n\
         var<private> m0: u64;"
    );
    if base > 64 {
        let _ = writeln!(s, "var<private> m1: u64;");
    }
    s.push('\n');

    emit_top_limb(&mut s);
    // Early exit on the first duplicate: almost no candidate survives.
    emit_scan(&mut s, &kernel, "scan_dup", true);

    // --- check_is_nice ------------------------------------------------------
    //
    // n² is scanned before n³ is ever multiplied out. Almost every candidate
    // repeats a digit inside n² alone, so the cube multiply — the widest
    // `emit_mul` here — is usually dead work. This ordering is worth 20-27%
    // whole-kernel in the CUDA original; the early `return` below is what
    // makes the generated cube code unreachable for the common case.
    s.push_str("fn check_is_nice(n_lo: u64, n_hi: u64) -> bool {\n");
    emit_unpack_n(&mut s, n_limbs);
    emit_mul(&mut s, "n", n_limbs, "n", n_limbs, "sq");
    emit_mask_clear(&mut s, &kernel);
    for i in 0..sq_limbs {
        let _ = writeln!(s, "    sv[{i}] = sq{i};");
    }
    let _ = writeln!(
        s,
        "    if (!scan_dup(top_limb({sq_limbs}))) {{ return false; }}"
    );
    emit_mul(&mut s, "sq", sq_limbs, "n", n_limbs, "cu");
    for i in 0..cu_limbs {
        let _ = writeln!(s, "    sv[{i}] = cu{i};");
    }
    let _ = writeln!(
        s,
        "    if (!scan_dup(top_limb({cu_limbs}))) {{ return false; }}"
    );
    let _ = writeln!(s, "    return ({}) == {base}u;\n}}\n", popcount_expr(&kernel));

    // --- prefilter / candidate_is_nice --------------------------------------
    if let Some(pre) = prefilter {
        emit_prefilter(&mut s, &kernel, &pre);
    }
    s.push_str("fn candidate_is_nice(n_lo: u64, n_hi: u64) -> bool {\n");
    if prefilter.is_some() {
        s.push_str("    if (!prefilter(n_lo, n_hi)) { return false; }\n");
    }
    if probe {
        s.push_str("    return true; // probe build: report prefilter survivors\n}\n\n");
    } else {
        s.push_str("    return check_is_nice(n_lo, n_hi);\n}\n\n");
    }

    // --- offset_mod_m -------------------------------------------------------
    //
    // `n mod M` the CUDA way would be two 64-bit divisions by a constant, and
    // that is precisely the construct ACO does not strength-reduce. The host
    // pushes `field_start mod M` instead, leaving only the 64-bit *offset* to
    // reduce here — one chunk at a time, every divisor a 32-bit literal. Exact
    // because `acc < M <= 2^(32-c)`, so `acc << c` never leaves a u32; `c`
    // comes from `stride_chunk_bits`, the same function the host mirror uses.
    let chunk_bits = stride_chunk_bits(stride_m);
    let chunk_mask = (1u32 << chunk_bits) - 1;
    let per_word = 32 / chunk_bits;
    s.push_str("fn offset_mod_m(lo: u32, hi: u32) -> u32 {\n    var acc: u32 = 0u;\n");
    for chunk in 0..64 / chunk_bits {
        let (word, idx) = if chunk < per_word {
            ("hi", chunk)
        } else {
            ("lo", chunk - per_word)
        };
        let shift = 32 - chunk_bits - idx * chunk_bits;
        let _ = writeln!(
            s,
            "    acc = ((acc << {chunk_bits}u) | (({word} >> {shift}u) & {chunk_mask:#x}u)) % {stride_m}u;"
        );
    }
    s.push_str("    return acc;\n}\n\n");

    // --- lower_bound_residue ------------------------------------------------
    let _ = writeln!(
        s,
        "fn lower_bound_residue(m: u32) -> u32 {{\n\
         \x20   var lo: u32 = 0u;\n\
         \x20   var hi: u32 = {stride_r}u;\n\
         \x20   loop {{\n\
         \x20       if (lo >= hi) {{ break; }}\n\
         \x20       let mid: u32 = (lo + hi) >> 1u;\n\
         \x20       if (residues[mid] < m) {{ lo = mid + 1u; }} else {{ hi = mid; }}\n\
         \x20   }}\n\
         \x20   return lo;\n\
         }}\n"
    );

    // --- entry point --------------------------------------------------------
    //
    // `lane`/`warp` here are pure index arithmetic over `global_invocation_id`
    // — the lanes never communicate — so this tiling is independent of the
    // device's subgroup width. That is why the port needs no subgroup ops and
    // no VK_EXT_subgroup_size_control.
    let _ = writeln!(
        s,
        "@compute @workgroup_size({wg})\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n\
         \x20       @builtin(num_workgroups) nwg: vec3<u32>) {{\n\
         \x20   let lanes: u32 = 1u << pc.lane_shift;\n\
         \x20   let lane: u32 = gid.x & (lanes - 1u);\n\
         \x20   let nwarps: u32 = (nwg.x * {wg}u) >> pc.lane_shift;\n\
         \x20   let fs_lo: u64 = (u64(pc.fs1) << 32u) | u64(pc.fs0);\n\
         \x20   let fs_hi: u64 = (u64(pc.fs3) << 32u) | u64(pc.fs2);\n\n\
         \x20   var r: u32 = gid.x >> pc.lane_shift;\n\
         \x20   loop {{\n\
         \x20       if (r >= pc.num_ranges) {{ break; }}\n\
         \x20       let off_lo: u32 = range_offsets[2u * r];\n\
         \x20       let off_hi: u32 = range_offsets[2u * r + 1u];\n\
         \x20       let offset: u64 = (u64(off_hi) << 32u) | u64(off_lo);\n\n\
         \x20       // range_start = field_start + offset, range_end = + len\n\
         \x20       let rs_lo: u64 = fs_lo + offset;\n\
         \x20       var rs_hi: u64 = fs_hi;\n\
         \x20       if (rs_lo < fs_lo) {{ rs_hi = rs_hi + 1lu; }}\n\
         \x20       let re_lo: u64 = rs_lo + u64(range_lens[r]);\n\
         \x20       var re_hi: u64 = rs_hi;\n\
         \x20       if (re_lo < rs_lo) {{ re_hi = re_hi + 1lu; }}\n\n\
         \x20       // B0 = range_start - (range_start mod M)\n\
         \x20       let m: u32 = (pc.fs_mod_m + offset_mod_m(off_lo, off_hi)) % {stride_m}u;\n\
         \x20       let b0_lo: u64 = rs_lo - u64(m);\n\
         \x20       var b0_hi: u64 = rs_hi;\n\
         \x20       if (rs_lo < u64(m)) {{ b0_hi = b0_hi - 1lu; }}\n\n\
         \x20       // The g-th valid candidate at or after the range start is\n\
         \x20       // B0 + (g / R) * M + residues[g % R].\n\
         \x20       var g: u32 = lower_bound_residue(m) + lane;\n\
         \x20       loop {{\n\
         \x20           let cycle: u32 = g / {stride_r}u;\n\
         \x20           let j: u32 = g - cycle * {stride_r}u;\n\
         \x20           let add: u64 = u64(cycle) * {stride_m}lu + u64(residues[j]);\n\
         \x20           let n_lo: u64 = b0_lo + add;\n\
         \x20           var n_hi: u64 = b0_hi;\n\
         \x20           if (n_lo < b0_lo) {{ n_hi = n_hi + 1lu; }}\n\
         \x20           if (n_hi > re_hi || (n_hi == re_hi && n_lo >= re_lo)) {{ break; }}\n\
         \x20           if (candidate_is_nice(n_lo, n_hi)) {{\n\
         \x20               let pos: u32 = atomicAdd(&nice_count[0], 1u);\n\
         \x20               if (pos < pc.nice_cap) {{\n\
         \x20                   let o: u32 = {NICE_STRIDE}u * pos;\n\
         \x20                   nice_out[o] = u32(n_lo);\n\
         \x20                   nice_out[o + 1u] = u32(n_lo >> 32u);\n\
         \x20                   nice_out[o + 2u] = u32(n_hi);\n\
         \x20                   nice_out[o + 3u] = u32(n_hi >> 32u);\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20           g = g + lanes;\n\
         \x20       }}\n\
         \x20       r = r + nwarps;\n\
         \x20   }}\n\
         }}"
    );

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_config::{MAX_GPU_DIGIT_MASK_BASE, gpu_supports_base};

    #[test]
    fn generates_for_every_supported_base() {
        let mut n = 0;
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            let cfg = KernelConfig::new(base).expect("supported base has a config");
            let src = detailed_wgsl(&cfg);
            assert!(src.contains("@compute"), "base {base}");
            // The digit mask must cover the base.
            if base > 64 {
                assert!(src.contains("m1"), "base {base} needs a second mask word");
            }
            n += 1;
        }
        assert!(n > 20, "only {n} bases generated");
    }

    /// The workgroup histogram must fit the smallest guaranteed
    /// `maxComputeSharedMemorySize` (32 KB in Vulkan 1.0's limits).
    #[test]
    fn workgroup_histogram_fits_the_guaranteed_limit() {
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            let cfg = KernelConfig::new(base).unwrap();
            let bytes = HIST_COPIES * cfg.hist_bins() * 4;
            assert!(bytes <= 32 * 1024, "base {base}: {bytes} bytes of LDS");
        }
    }

    /// Every candidate must be representable in the near-miss record.
    #[test]
    fn near_miss_record_holds_the_full_candidate() {
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            let cfg = KernelConfig::new(base).unwrap();
            assert!(
                cfg.n_limbs <= 4,
                "base {base}: {} limbs exceeds the u128 record",
                cfg.n_limbs
            );
        }
    }

    /// Every base a niceonly shader can be built for must have a stride modulus
    /// inside the chunked reduction's bound — the invariant that lets
    /// `acc << c` stay in a u32 and so keeps every divisor 32-bit.
    /// `NiceonlyConfig::new` refuses anything past it rather than generating a
    /// shader that silently computes the wrong residue.
    ///
    /// The margin is no longer generous. At the k=3 stride depth #88 introduced,
    /// `M = (b-1)·b³` reaches 266 338 304 at base 128 against the 4-bit chunk's
    /// 2^28 = 268 435 456 — under 1% of headroom. A k=4 table, or a supported
    /// base above 128, would need a 2-bit chunk; this test is what would catch it.
    #[test]
    fn stride_modulus_fits_the_chunked_horner_bound() {
        let mut n = 0;
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base)
                || crate::residue_filter::get_residue_filter_u128(&base).is_empty()
            {
                continue;
            }
            let table = StrideTable::new(base, crate::gpu_niceonly::GPU_LSD_K);
            assert!(
                table.modulus <= MAX_STRIDE_MODULUS,
                "base {base}: M={} exceeds {MAX_STRIDE_MODULUS}",
                table.modulus
            );
            NiceonlyConfig::new(KernelConfig::new(base).unwrap(), &table)
                .unwrap_or_else(|e| panic!("base {base}: {e}"));
            n += 1;
        }
        assert!(n > 20, "only {n} bases checked");
    }

    /// The byte-wise Horner reduction must equal `%`.
    ///
    /// This is the one piece of the niceonly kernel that is *not* a
    /// transliteration of the CUDA original — CUDA reduces `n` with two 64-bit
    /// divisions by a constant, which is exactly the construct RADV/ACO expands
    /// into a ~220-instruction loop. So it is the piece that needs a mirror.
    /// Both halves are checked: the offset reduction itself, and the
    /// `field_start mod M` split the shader relies on to avoid ever reducing a
    /// u128 on-device.
    #[test]
    fn offset_horner_matches_modulo() {
        for base in [10u32, 12, 25, 40, 45, 62, 94, 128] {
            if crate::residue_filter::get_residue_filter_u128(&base).is_empty() {
                continue;
            }
            let table = StrideTable::new(base, crate::gpu_niceonly::GPU_LSD_K);
            #[allow(clippy::cast_possible_truncation)]
            let m = table.modulus as u32;

            let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
            for i in 0..2000u64 {
                let offset = x;
                assert_eq!(
                    u64::from(offset_mod_m(offset, m)),
                    offset % u64::from(m),
                    "base {base}: offset {offset} mod {m}"
                );
                // And the two-part form the shader computes: it never reduces
                // a u128, only `field_start mod M` (host) plus the offset.
                let field_start = u128::from(x) << 37 | u128::from(i);
                #[allow(clippy::cast_possible_truncation)]
                let fs_mod = (field_start % u128::from(m)) as u32;
                let combined = (fs_mod + offset_mod_m(offset, m)) % m;
                assert_eq!(
                    u128::from(combined),
                    (field_start + u128::from(offset)) % u128::from(m),
                    "base {base}: split reduction at fs={field_start} off={offset}"
                );
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i);
            }
        }
    }

    /// Rust mirror of the niceonly kernel's candidate loop, with all
    /// lanes' indices merged (so `g` increments by 1).
    /// Takes the range the way the shader does — a field start plus a u64
    /// offset — because that split is exactly what is being checked.
    fn mirror_kernel_candidates(
        field_start: u128,
        offset: u64,
        len: u32,
        table: &StrideTable,
    ) -> Vec<u128> {
        #[allow(clippy::cast_possible_truncation)]
        let modulus = table.modulus as u32;
        let residues: Vec<u32> = table
            .valid_residues
            .iter()
            .map(|&r| u32::try_from(r).unwrap())
            .collect();
        let r_count = u32::try_from(residues.len()).unwrap();

        #[allow(clippy::cast_possible_truncation)]
        let fs_mod_m = (field_start % u128::from(modulus)) as u32;
        let m = (fs_mod_m + offset_mod_m(offset, modulus)) % modulus;

        let range_start = field_start + u128::from(offset);
        let range_end = range_start + u128::from(len);
        let b0 = range_start - u128::from(m);
        let idx0 = u32::try_from(residues.partition_point(|&r| r < m)).unwrap();

        let mut out = Vec::new();
        let mut g = idx0;
        loop {
            let cycle = g / r_count;
            let j = g - cycle * r_count;
            let add = u64::from(cycle) * u64::from(modulus) + u64::from(residues[j as usize]);
            let n = b0 + u128::from(add);
            if n >= range_end {
                break;
            }
            out.push(n);
            g += 1;
        }
        out
    }

    /// Candidates via the trusted CPU stride table iteration.
    fn cpu_candidates(start: u128, end: u128, table: &StrideTable) -> Vec<u128> {
        let mut out = Vec::new();
        let (mut n, mut idx) = table.first_valid_at_or_after(start);
        while n < end {
            out.push(n);
            n += u128::from(table.gap_table[idx]);
            idx = (idx + 1) % table.gap_table.len();
        }
        out
    }

    /// The on-device candidate reconstruction must visit exactly the stride
    /// table's candidates — no more (it would waste work and could report a
    /// number the CPU never checks) and no fewer (it would miss a solution).
    ///
    /// Probes the awkward starts specifically: the field start itself, a
    /// modulus wraparound, and a residue landing strictly past the last valid
    /// one, which is the case where the binary search returns `R` and the
    /// candidate has to come from the next cycle.
    #[test]
    fn kernel_candidate_enumeration_matches_stride_table() {
        for base in [10u32, 12, 25, 40, 45, 62, 94] {
            let Ok(Some(base_range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let table = StrideTable::new(base, crate::gpu_niceonly::GPU_LSD_K);
            if table.valid_residues.is_empty() {
                continue;
            }
            let modulus = table.modulus;
            let field_start = base_range.range_start;

            // An offset whose residue lands past the last valid residue.
            let past_last = {
                let target = u128::from(table.valid_residues.last().unwrap() + 1);
                let cycle_base = field_start - (field_start % modulus);
                let mut s = cycle_base + target.min(modulus - 1);
                if s < field_start {
                    s += modulus;
                }
                s - field_start
            };
            let offsets = [
                0u128,
                1,
                modulus - 1,
                modulus * 7 + modulus / 2,
                (base_range.range_end - field_start) / 2,
                past_last,
            ];
            for offset in offsets {
                let Ok(offset) = u64::try_from(offset) else {
                    continue;
                };
                for size in [1u128, 250, 1999, 3 * modulus + 17] {
                    let start = field_start + u128::from(offset);
                    let end = (start + size).min(base_range.range_end);
                    if start >= end {
                        continue;
                    }
                    let len = u32::try_from(end - start).unwrap();
                    assert_eq!(
                        mirror_kernel_candidates(field_start, offset, len, &table),
                        cpu_candidates(start, end, &table),
                        "candidate mismatch: base {base} range [{start}, {end})"
                    );
                }
            }
        }
    }

    // --- prefilter ----------------------------------------------------------

    /// Bases to exercise the prefilter on: every base that enables one.
    fn prefilter_bases() -> Vec<(u32, KernelConfig, VulkanPrefilterParams)> {
        (10..=MAX_GPU_DIGIT_MASK_BASE)
            .filter(|&b| gpu_supports_base(b))
            .filter_map(|b| {
                Some((b, KernelConfig::new(b).ok()?, vulkan_prefilter_params(b)?))
            })
            .collect()
    }

    /// The chunk arithmetic must reproduce `n² mod b^p` and `n³ mod b^p`
    /// exactly, chunk by chunk.
    ///
    /// This is the piece with no CUDA original to transliterate — the kernel
    /// computes those residues in a u64 and reduces with `%`, which is the one
    /// construct RADV/ACO expands — so, like the byte-wise Horner in the stride
    /// reduction, it needs a mirror rather than a reading.
    #[test]
    fn prefilter_chunks_match_direct_modular_powers() {
        let mut checked = 0;
        for (base, cfg, pre) in prefilter_bases() {
            let range = crate::base_range::get_base_range_u128(base).unwrap().unwrap();
            let span = range.range_end - range.range_start;
            let modulus = u128::from(base).pow(pre.digits);
            let d = u128::from(pre.chunk_div);

            let mut x: u128 = 0x0123_4567_89ab_cdef_0f1e_2d3c_4b5a_6978;
            for i in 0..500u128 {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i);
                let n = range.range_start + (x % span);
                let (_, sq, cu) = mirror_prefilter(n, &cfg, &pre);

                let nm = n % modulus;
                let sq_direct = nm * nm % modulus;
                let cu_direct = sq_direct * nm % modulus;
                for (chunks, direct, name) in
                    [(&sq, sq_direct, "n²"), (&cu, cu_direct, "n³")]
                {
                    let rebuilt = chunks
                        .iter()
                        .rev()
                        .fold(0u128, |acc, &c| acc * d + u128::from(c));
                    assert_eq!(rebuilt, direct, "b{base} n={n}: {name} mod b^p mismatch");
                }
                checked += 1;
            }
        }
        assert!(checked > 1000, "only {checked} samples");
    }

    /// Soundness and selectivity, the two properties that matter.
    ///
    /// Soundness is the one that can lose a solution: a rejected candidate must
    /// really not be nice, checked against the CPU's own `get_is_nice`. The
    /// failure mode this guards is a prefilter peeling more digits than the
    /// value has, so that phantom leading zeros collide and every candidate is
    /// rejected — the CUDA path shipped exactly that (v3.2.14).
    #[test]
    fn prefilter_is_sound_and_selective() {
        const SAMPLES: u32 = 2000;
        for (base, cfg, pre) in prefilter_bases() {
            let range = crate::base_range::get_base_range_u128(base).unwrap().unwrap();
            let span = range.range_end - range.range_start;
            let mut x: u128 = 0xdead_beef_cafe_f00d_0d15_ea5e_feed_face;
            let mut rejected = 0u32;
            for i in 0..SAMPLES {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(u128::from(i));
                let n = range.range_start + (x % span);
                if mirror_prefilter(n, &cfg, &pre).0 {
                    continue;
                }
                rejected += 1;
                assert!(
                    !crate::client_process::get_is_nice(n, base),
                    "prefilter rejected a nice number: b{base} n={n}"
                );
            }
            assert!(
                rejected * 2 > SAMPLES,
                "prefilter suspiciously weak at b{base}: {rejected}/{SAMPLES}"
            );
            #[allow(clippy::cast_precision_loss)]
            {
                println!(
                    "b{base}: {} digits/value, kill rate {:.1}%",
                    pre.digits,
                    100.0 * f64::from(rejected) / f64::from(SAMPLES)
                );
            }
        }
    }

    /// The shader must carry the prefilter exactly when the config says so, and
    /// route candidates through `candidate_is_nice` either way — a base where
    /// it is disabled must not so much as mention it.
    #[test]
    fn prefilter_is_emitted_only_where_configured() {
        let mut with = 0;
        let mut without = 0;
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base)
                || crate::residue_filter::get_residue_filter_u128(&base).is_empty()
            {
                continue;
            }
            let table = StrideTable::new(base, crate::gpu_niceonly::GPU_LSD_K);
            let cfg = NiceonlyConfig::new(KernelConfig::new(base).unwrap(), &table).unwrap();
            let src = niceonly_wgsl(&cfg);
            assert!(src.contains("candidate_is_nice(n_lo, n_hi)"), "base {base}");
            if cfg.prefilter.is_some() {
                assert!(src.contains("fn prefilter("), "base {base} lost its prefilter");
                with += 1;
            } else {
                assert!(!src.contains("prefilter"), "base {base} gained a prefilter");
                without += 1;
            }
        }
        assert!(with > 0, "no base emitted a prefilter");
        assert!(without > 0, "every base emitted a prefilter");
    }

    /// The lane tiling must track the range size, never leave the shader's
    /// representable band, and never hand a range more lanes than it has
    /// candidates — which is the whole failure it exists to fix.
    #[test]
    fn lane_tiling_follows_the_range_size() {
        let table = StrideTable::new(40, crate::gpu_niceonly::GPU_LSD_K);
        #[allow(clippy::cast_possible_truncation)]
        let (m, r) = (table.modulus as u32, table.valid_residues.len() as u32);

        // A full batch of floor-250 ranges: ~490 numbers each, i.e. ~39
        // candidates — the configuration where a fixed 32 lanes leaves 1.2
        // candidates apiece, and where one lane each measured fastest.
        let batch = crate::gpu_niceonly::LAUNCH_BATCH_RANGES as u64;
        assert_eq!(lane_shift_for(batch, 490, m, r), 0, "short ranges want 1 lane");
        // Long ranges keep the full warp.
        assert_eq!(
            lane_shift_for(batch, 1 << 20, m, r),
            MAX_LANES_PER_RANGE.ilog2()
        );
        // A range with no candidates at all still gets one lane, not zero.
        assert_eq!(lane_shift_for(batch, 0, m, r), 0);
        // A batch too small to fill the device buys threads with lanes instead,
        // even though the work rule alone would ask for one.
        assert!(
            lane_shift_for(64, 490, m, r) > lane_shift_for(batch, 490, m, r),
            "a tiny batch must widen the tiling"
        );

        let mut prev = 0;
        for len in (0..24).map(|k| 1u64 << k) {
            let shift = lane_shift_for(batch, len, m, r);
            assert!(shift >= prev, "tiling must not shrink as ranges grow");
            assert!(1 << shift <= MAX_LANES_PER_RANGE, "len {len}: too many lanes");
            let candidates = len * u64::from(r) / u64::from(m);
            assert!(
                (1u64 << shift) <= candidates.max(1),
                "len {len}: {} lanes for {candidates} candidates",
                1u64 << shift
            );
            prev = shift;
        }
    }

    /// The measurement knob, without touching the process environment.
    #[test]
    fn the_prefilter_knob_only_accepts_off_switches() {
        assert!(prefilter_enabled_for(None));
        assert!(prefilter_enabled_for(Some("1")));
        assert!(prefilter_enabled_for(Some("")));
        for off in ["0", "off", "false"] {
            assert!(!prefilter_enabled_for(Some(off)), "{off} should disable");
        }
    }

    /// A residue-empty base must be refused, not silently generate a shader
    /// whose binary search runs over an empty table.
    #[test]
    fn niceonly_config_refuses_residue_empty_bases() {
        let mut n = 0;
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base)
                || !crate::residue_filter::get_residue_filter_u128(&base).is_empty()
            {
                continue;
            }
            let table = StrideTable::new(base, crate::gpu_niceonly::GPU_LSD_K);
            assert!(
                NiceonlyConfig::new(KernelConfig::new(base).unwrap(), &table).is_err(),
                "base {base} should be refused"
            );
            n += 1;
        }
        assert!(n >= 10, "only {n} residue-empty bases exercised");
    }
}

#[cfg(test)]
mod dump {
    /// Print a generated shader, for eyeballing codegen changes:
    ///
    /// ```text
    /// DUMP_BASE=40 DUMP_MODE=niceonly cargo test -p nice_common --features vulkan \
    ///   --lib dump_shader -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "developer aid"]
    fn dump_shader() {
        let base: u32 = std::env::var("DUMP_BASE").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
        let cfg = super::KernelConfig::new(base).unwrap();
        if std::env::var("DUMP_MODE").as_deref() == Ok("niceonly") {
            let table = crate::stride_filter::StrideTable::new(base, 2);
            let nc = super::NiceonlyConfig::new(cfg, &table).unwrap();
            println!("{}", super::niceonly_wgsl(&nc));
        } else {
            println!("{}", super::detailed_wgsl(&cfg));
        }
    }
}
