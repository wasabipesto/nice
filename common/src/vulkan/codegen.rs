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

use crate::gpu_config::{chunk_constants_u16, n_limbs};
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

/// Threads cooperating on one MSD-valid range, matching the CUDA kernel's
/// one-warp-per-range tiling. Nothing here is a hardware property — the lanes
/// stride through the range's candidates by index and never communicate — so
/// this is a tuning constant, not a subgroup width. 64 is the obvious thing to
/// try on RDNA, where a wave is currently split across two ranges.
pub const LANES_PER_RANGE: u32 = 32;

/// Largest stride modulus the niceonly shader's residue reduction accepts.
///
/// `n mod M` cannot be computed as a 64-bit division by a constant — that is
/// the one construct RADV/ACO does not strength-reduce (see the module docs).
/// Instead the shader reduces the range's 64-bit *offset* byte by byte,
/// `acc = (acc << 8 | byte) % M`, with `M` a 32-bit compile-time constant. The
/// running remainder satisfies `acc < M`, so `acc << 8` stays inside a u32
/// exactly while `M <= 2^24`. Real moduli are far below it: `M = (b-1)·b^k`
/// with `k = 2`, so even base 128 gives 127·16384 ≈ 2^21.
pub const MAX_STRIDE_MODULUS: u128 = 1 << 24;

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
        })
    }
}

/// Reduce a u64 range offset mod `M` on the host, the way the shader does.
///
/// Exposed so the mirror test can check the byte-wise Horner against `%`
/// directly; the shader emits exactly this, unrolled.
#[must_use]
pub fn offset_mod_m(offset: u64, stride_m: u32) -> u32 {
    let mut acc: u32 = 0;
    for byte in offset.to_be_bytes() {
        acc = ((acc << 8) | u32::from(byte)) % stride_m;
    }
    acc
}

/// The complete niceonly compute shader for one base.
///
/// Mirrors `niceonly_ranges_kernel` in `nice_kernels.cu`: one group of
/// [`LANES_PER_RANGE`] threads per MSD-valid range, candidates reconstructed
/// on-device from the residue table, so the host ships ~12 bytes per range and
/// no per-candidate data at all.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn niceonly_wgsl(cfg: &NiceonlyConfig) -> String {
    let NiceonlyConfig {
        kernel,
        stride_m,
        stride_r,
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
    let lanes = LANES_PER_RANGE;
    let lane_shift = lanes.trailing_zeros();

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
         \x20   pad: u32,\n\
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

    // --- offset_mod_m -------------------------------------------------------
    //
    // `n mod M` the CUDA way would be two 64-bit divisions by a constant, and
    // that is precisely the construct ACO does not strength-reduce. The host
    // pushes `field_start mod M` instead, leaving only the 64-bit *offset* to
    // reduce here — byte by byte, every divisor a 32-bit literal. Exact
    // because `acc < M <= 2^24`, so `acc << 8` never leaves a u32.
    s.push_str("fn offset_mod_m(lo: u32, hi: u32) -> u32 {\n    var acc: u32 = 0u;\n");
    for byte in 0..8u32 {
        let (word, shift) = if byte < 4 {
            ("hi", 24 - byte * 8)
        } else {
            ("lo", 24 - (byte - 4) * 8)
        };
        let _ = writeln!(
            s,
            "    acc = ((acc << 8u) | (({word} >> {shift}u) & 0xffu)) % {stride_m}u;"
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
         \x20   let lane: u32 = gid.x & {}u;\n\
         \x20   let nwarps: u32 = (nwg.x * {wg}u) >> {lane_shift}u;\n\
         \x20   let fs_lo: u64 = (u64(pc.fs1) << 32u) | u64(pc.fs0);\n\
         \x20   let fs_hi: u64 = (u64(pc.fs3) << 32u) | u64(pc.fs2);\n\n\
         \x20   var r: u32 = gid.x >> {lane_shift}u;\n\
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
         \x20           if (check_is_nice(n_lo, n_hi)) {{\n\
         \x20               let pos: u32 = atomicAdd(&nice_count[0], 1u);\n\
         \x20               if (pos < pc.nice_cap) {{\n\
         \x20                   let o: u32 = {NICE_STRIDE}u * pos;\n\
         \x20                   nice_out[o] = u32(n_lo);\n\
         \x20                   nice_out[o + 1u] = u32(n_lo >> 32u);\n\
         \x20                   nice_out[o + 2u] = u32(n_hi);\n\
         \x20                   nice_out[o + 3u] = u32(n_hi >> 32u);\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20           g = g + {lanes}u;\n\
         \x20       }}\n\
         \x20       r = r + nwarps;\n\
         \x20   }}\n\
         }}",
        lanes - 1
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
    /// inside the byte-wise reduction's bound — the invariant that lets
    /// `acc << 8` stay in a u32 and so keeps every divisor 32-bit. Bases with
    /// `M > 2^24` would have to fall back to a narrower chunk, and
    /// `NiceonlyConfig::new` refuses them rather than generating a shader that
    /// silently computes the wrong residue.
    #[test]
    fn stride_modulus_fits_the_byte_horner_bound() {
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
    /// [`LANES_PER_RANGE`] lanes' indices merged (so `g` increments by 1).
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
            n += table.gap_table[idx];
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
                let target = table.valid_residues.last().unwrap() + 1;
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
