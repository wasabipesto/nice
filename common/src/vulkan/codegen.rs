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
use anyhow::{Context as _, Result};
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
fn emit_digit_set(s: &mut String, cfg: &KernelConfig, indent: &str) {
    if cfg.base <= 64 {
        let _ = writeln!(s, "{indent}m0 = m0 | (1lu << d);");
    } else {
        let _ = writeln!(
            s,
            "{indent}if (d < 64u) {{ m0 = m0 | (1lu << d); }} else {{ m1 = m1 | (1lu << (d - 64u)); }}"
        );
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

    // --- top_limb -----------------------------------------------------------
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

    // --- scan_digits (no early exit; detailed mode counts every digit) ------
    let _ = writeln!(
        s,
        "fn scan(top_in: i32) {{\n\
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
         \x20       var chunk: u32 = rem;\n\
         \x20       if (top >= 0) {{"
    );
    // Interior chunk: exactly chunk_digits digits, zeros included.
    let _ = writeln!(
        s,
        "            for (var k: u32 = 0u; k < {chunk_digits}u; k = k + 1u) {{\n\
         \x20               let d: u32 = chunk % {base}u;\n\
         \x20               chunk = chunk / {base}u;"
    );
    emit_digit_set(&mut s, cfg, "                ");
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
    emit_digit_set(&mut s, cfg, "                ");
    let _ = writeln!(
        s,
        "            }}\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n"
    );

    // --- num_unique ---------------------------------------------------------
    s.push_str("fn num_unique(n_lo: u64, n_hi: u64) -> u32 {\n");
    for i in 0..n_limbs {
        let word = if i < 2 { "n_lo" } else { "n_hi" };
        let shift = (i & 1) * 32;
        let _ = writeln!(s, "    let n{i}: u32 = u32({word} >> {shift}u);");
    }
    emit_mul(&mut s, "n", n_limbs, "n", n_limbs, "sq");
    emit_mul(&mut s, "sq", sq_limbs, "n", n_limbs, "cu");
    s.push_str("    m0 = 0lu;\n");
    if base > 64 {
        s.push_str("    m1 = 0lu;\n");
    }
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
    if base <= 64 {
        s.push_str("    return countOneBits(u32(m0)) + countOneBits(u32(m0 >> 32u));\n}\n\n");
    } else {
        s.push_str(
            "    return countOneBits(u32(m0)) + countOneBits(u32(m0 >> 32u))\n\
             \x20        + countOneBits(u32(m1)) + countOneBits(u32(m1 >> 32u));\n}\n\n",
        );
    }

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
}
