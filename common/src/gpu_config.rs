//! Backend-neutral GPU kernel configuration.
//!
//! Everything here derives per-base constants from the search range and the
//! stride/residue machinery, with no reference to any particular GPU API. It
//! exists because [`crate::client_process_gpu`] is `#![cfg(feature = "gpu")]` —
//! nothing in it is reachable from a Vulkan-only build — while the constants
//! themselves are the same whichever backend consumes them.
//!
//! The CUDA path bakes these into NVRTC `-D` defines; the Vulkan path bakes
//! them into generated WGSL. Both must agree, and both must agree with the CPU
//! implementation, so there is exactly one derivation and it lives here.

use crate::base_range;

/// Highest base the GPU digit mask can represent (two u64 words). The kernels'
/// u32-limb arithmetic is width-generic (buffers are sized from `N_LIMBS` at
/// JIT time), so unlike the CPU's `U256` path there is no 256-bit ceiling; in
/// practice the u128 candidate representation caps usable bases around 97 via
/// `get_base_range_u128`.
pub const MAX_GPU_DIGIT_MASK_BASE: u32 = 128;

/// Whether a GPU backend can process this base natively. Bases outside this
/// fall back to the CPU implementation. Unlike the CPU fast path (capped at
/// `MAX_BASE_FOR_FIXED_WIDTH_U256` = 68 by its 256-bit type), the GPU's
/// limb-generic arithmetic handles every base with a valid u128 range.
///
/// Bases below 10 are excluded: their search ranges are trivially small
/// (b5's is two numbers), and `get_base_range_u128` panics outright on
/// degenerate ones like b4 — not worth guarding for on the GPU path.
#[must_use]
pub fn gpu_supports_base(base: u32) -> bool {
    (10..=MAX_GPU_DIGIT_MASK_BASE).contains(&base)
        && matches!(base_range::get_base_range_u128(base), Ok(Some(_)))
}

/// u32 limbs needed to hold the largest candidate in this base's range.
///
/// Returns `None` for bases with no valid u128 range.
#[must_use]
pub fn n_limbs(base: u32) -> Option<u32> {
    let range = base_range::get_base_range_u128(base).ok()??;
    let n_max = range.range_end - 1;
    let n_bits = 128 - n_max.leading_zeros();
    Some(n_bits.div_ceil(32).max(1))
}

/// Largest `(e, base^e)` with `base^e < limit`.
///
/// Both GPU backends extract digits in chunks: divide the multi-limb value by
/// `base^e` to peel off `e` digits at a time, then peel single digits from the
/// chunk. `limit` is set by how wide an integer the backend can divide by a
/// constant *cheaply* — see [`chunk_constants`] and [`chunk_constants_u16`].
#[must_use]
pub fn chunk_constants_below(base: u32, limit: u64) -> (u32, u32) {
    let mut e = 0u32;
    let mut div = 1u64;
    while div * u64::from(base) < limit {
        div *= u64::from(base);
        e += 1;
    }
    #[allow(clippy::cast_possible_truncation)]
    (e, div as u32)
}

/// Chunk constants for the CUDA backend: `base^e < 2^31`, so the chunk fits a
/// u32 and the `u64 / base^e` split is a 64-bit division by a compile-time
/// constant — which nvcc strength-reduces to a multiply-high.
#[must_use]
pub fn chunk_constants(base: u32) -> (u32, u32) {
    chunk_constants_below(base, 1 << 31)
}

/// Chunk constants for the Vulkan backend: `base^e < 2^16`.
///
/// The tighter bound is not arbitrary. RADV/ACO strength-reduces division by a
/// 32-bit compile-time constant to `v_mul_hi_u32`, but **not** division by a
/// 64-bit one — NIR's `nir_opt_idiv_const` only handles widths up to 32, so a
/// `u64 / const` expands to a ~220-instruction restoring shift-subtract loop.
/// Keeping `base^e < 2^16` lets the chunk split be done as two 32-bit constant
/// divisions over 16-bit halves instead (see the `split16` codegen), which is
/// exact and entirely multiply-high.
///
/// Measured on base 40, RTX-free (AMD Radeon 860M / RADV GFX1152): 3.88e8
/// candidates/s with this form against 1.36e8/s with the 64-bit divide, i.e.
/// **2.85x**, despite `e` dropping from 5 to 3 and so needing ~1.7x more chunk
/// iterations. Shader code size halves (4424 B vs 10508 B) and the division
/// expansion disappears from the ISA entirely (0 `s_brev_b64` vs 30).
#[must_use]
pub fn chunk_constants_u16(base: u32) -> (u32, u32) {
    chunk_constants_below(base, 1 << 16)
}

/// Parameters for the CUDA niceonly kernel's low-digit modular prefilter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefilterParams {
    /// Digits checked per value (the lowest `digits` of n² and of n³).
    pub digits: u32,
    /// `base^digits`, at most 2^48.
    pub modulus: u64,
    /// `2^64 mod modulus`.
    pub pow64_mod: u64,
}

/// Highest base where the niceonly kernel's fused prefilter still pays for
/// itself. Above this the prefilter is compiled out even though it would be
/// sound: warp divergence means the whole warp runs the full check whenever
/// any lane survives, so the prefilter only pays while per-lane survival is
/// very low (at 4% survival ~74% of warp iterations already hold a
/// survivor). Measured (G1 2026-07-12 + crossover sweep,
/// scratchpad/2026-07-gpu-compaction/g1-verdict.md): fused wins at b40
/// (1.1% survival, 16-30% faster on 3090/4090), loses at every live base
/// from b42 up (4%+ survival, 4-27% slower). b41 has no live range.
pub const GPU_PREFILTER_MAX_BASE: u32 = 40;

/// Fewest base-`b` digits that both n² and n³ are guaranteed to have across
/// the base's whole search range, or `None` for a base with no range.
///
/// This is the soundness bound for any low-digit prefilter: it may only check
/// `p` digits per value if both powers really *have* `p` digits everywhere in
/// the range, or the digit loop would extract phantom leading zeros and could
/// falsely reject a nice number. A conservative log-based bound at the range
/// start, which is the range's minimum — both powers are increasing in n.
///
/// Shared by both backends' derivations so they cannot disagree about what is
/// sound, only about how many digits they choose to check.
#[must_use]
fn guaranteed_low_digits(base: u32) -> Option<u32> {
    let range = base_range::get_base_range_u128(base).ok()??;
    #[allow(clippy::cast_precision_loss)]
    let ln_n_min = (range.range_start as f64).ln();
    let ln_base = f64::from(base).ln();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let digit_lower_bound =
        |power: f64| ((power * ln_n_min / ln_base).floor() - 1.0).max(0.0) as u32;
    Some(digit_lower_bound(2.0).min(digit_lower_bound(3.0)))
}

/// Compute the prefilter parameters for a base, or None when the prefilter
/// must stay disabled.
///
/// The prefilter checks the lowest p digits of n² and n³ using
/// `x^k mod b^p == (x mod b^p)^k mod b^p`. Two conditions must hold:
/// - the modulus must fit the kernel's u64 modular arithmetic, and
/// - n² and n³ must each be guaranteed at least p digits across the base's
///   whole range (see [`guaranteed_low_digits`]).
#[must_use]
pub fn prefilter_params(base: u32) -> Option<PrefilterParams> {
    // Profitability gate (see GPU_PREFILTER_MAX_BASE). Every consumer —
    // define injection, the CPU diagnostics mirror, the G0/G1 harnesses —
    // takes the on/off decision from this single function, so the kernel
    // and its mirrors always agree.
    if base > GPU_PREFILTER_MAX_BASE {
        return None;
    }

    let mut digits = 0u32;
    let mut modulus = 1u64;
    while modulus <= (1u64 << 48) / u64::from(base) {
        modulus *= u64::from(base);
        digits += 1;
    }

    if digits < 4 || guaranteed_low_digits(base)? < digits {
        return None;
    }

    // Exact: the remainder is < modulus <= 2^48.
    #[allow(clippy::cast_possible_truncation)]
    let pow64_mod = ((1u128 << 64) % u128::from(modulus)) as u64;
    Some(PrefilterParams {
        digits,
        modulus,
        pow64_mod,
    })
}

/// Parameters for the Vulkan niceonly shader's low-digit prefilter.
///
/// Same idea as [`PrefilterParams`], different number representation. CUDA
/// holds `n^k mod b^p` in a u64 and reduces with `% b^p`; the Vulkan shader
/// holds it as [`limbs`](Self::limbs) digits-chunks of base `chunk_div`, so
/// that every divisor it ever names is a 32-bit constant. See
/// [`vulkan_prefilter_params`] for why that matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanPrefilterParams {
    /// Digits checked per value: `limbs * chunk_digits`.
    pub digits: u32,
    /// Chunks the value is held in.
    pub limbs: u32,
    /// Base-`b` digits per chunk.
    pub chunk_digits: u32,
    /// `base^chunk_digits`, below 2^16.
    pub chunk_div: u32,
}

/// Chunks the Vulkan prefilter carries, i.e. its modulus is `chunk_div^3`.
///
/// Three is the value that matches CUDA's reach without exceeding it: a chunk
/// is just under 2^16, so three of them is just under the 2^48 modulus the
/// CUDA prefilter uses, and `digits` comes out equal at base 40 (9 either
/// way). It also keeps the truncated multiply down to six digit-chunk
/// mul-adds. A fourth chunk would check more digits per candidate, at ten
/// mul-adds and a wider reduction — untested, and the measured question at
/// base 40 is whether the prefilter pays at all, not whether it should be
/// longer.
pub const VULKAN_PREFILTER_LIMBS: u32 = 3;

/// Compute the Vulkan prefilter parameters for a base, or None when it must
/// stay disabled.
///
/// # Why this is not just [`prefilter_params`] with a different consumer
///
/// The CUDA prefilter is built out of 64-bit divisions by compile-time
/// constants: `lo % PRE_MOD` in `reduce_pre`, and `sq % BASE` / `sq /= BASE`
/// on a u64 in the digit peel. That is precisely the one construct RADV/ACO
/// does not strength-reduce (NIR's `nir_opt_idiv_const` is width-limited to
/// 32 bits), and the reason the digit scan needed `split16` in the first
/// place. Ported literally, the prefilter would drag that ~220-instruction
/// expansion back in three more times — and `__umul64hi`, the difficulty the
/// port was expected to have, is the least of it.
///
/// So the Vulkan prefilter carries `n^k mod b^p` as [`VULKAN_PREFILTER_LIMBS`]
/// chunks of `chunk_digits` digits each, in base `chunk_div = base^chunk_digits
/// < 2^16` — the same chunk the digit scan already uses. A truncated schoolbook
/// multiply over those chunks needs only `u32` intermediates (see
/// [`VulkanPrefilterParams`] and the codegen), the digits fall straight out of
/// each chunk with 32-bit divisions, and nothing 64-bit-divided is ever named.
#[must_use]
pub fn vulkan_prefilter_params(base: u32) -> Option<VulkanPrefilterParams> {
    // Same profitability gate as CUDA's. It was measured for CUDA warps
    // (GPU_PREFILTER_MAX_BASE), and the argument — a prefilter only pays while
    // per-lane survival is low enough that whole warps skip the full check —
    // is about SIMT execution rather than about NVIDIA, so it carries over.
    // The Vulkan filter checks at most as many digits as the CUDA one, so it
    // cannot be *more* selective at any base and the bound stays valid.
    if base > GPU_PREFILTER_MAX_BASE {
        return None;
    }
    let (chunk_digits, chunk_div) = chunk_constants_u16(base);
    let digits = chunk_digits * VULKAN_PREFILTER_LIMBS;
    if digits < 4 || guaranteed_low_digits(base)? < digits {
        return None;
    }
    Some(VulkanPrefilterParams {
        digits,
        limbs: VULKAN_PREFILTER_LIMBS,
        chunk_digits,
        chunk_div,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_constants_are_maximal() {
        for base in 2..=MAX_GPU_DIGIT_MASK_BASE {
            let (e, div) = chunk_constants(base);
            assert_eq!(u64::from(div), u64::from(base).pow(e), "base {base}");
            assert!(u64::from(div) < (1 << 31), "base {base}");
            assert!(
                u64::from(div) * u64::from(base) >= (1 << 31),
                "base {base}: chunk not maximal"
            );
        }
    }

    /// The Vulkan split16 chunk split needs `rem < CHUNK_DIV < 2^16` so that
    /// `rem << 16` and `(c1 % CHUNK_DIV) << 16` both stay inside a u32.
    #[test]
    fn u16_chunk_constants_fit_the_split16_invariant() {
        for base in 2..=MAX_GPU_DIGIT_MASK_BASE {
            let (e, div) = chunk_constants_u16(base);
            assert_eq!(u64::from(div), u64::from(base).pow(e), "base {base}");
            assert!(u64::from(div) < (1 << 16), "base {base}: div {div} >= 2^16");
            assert!(
                u64::from(div) * u64::from(base) >= (1 << 16),
                "base {base}: chunk not maximal"
            );
            assert!(e >= 1, "base {base}: no digits per chunk");
            // rem < div < 2^16, so (rem << 16) | x fits a u32.
            assert!((u64::from(div) - 1) << 16 < (1 << 32), "base {base}");
        }
    }

    /// The prefilter's truncated multiply accumulates
    /// `a*b + acc + carry` with every operand below `chunk_div`, in a `u32`.
    /// The worst case is `(D-1)^2 + 2(D-1) = D^2 - 1`, so the whole scheme
    /// rests on `chunk_div^2 <= 2^32` — which `chunk_div < 2^16` gives with
    /// nothing to spare. If the chunk bound ever loosens, this fails first.
    #[test]
    fn prefilter_multiply_cannot_overflow_a_u32() {
        for base in 2..=MAX_GPU_DIGIT_MASK_BASE {
            let (_, div) = chunk_constants_u16(base);
            let d = u64::from(div);
            assert!(d * d - 1 < (1 << 32), "base {base}: div {div} overflows");
        }
    }

    /// The digit-count guarantee is the prefilter's soundness condition, and
    /// [`guaranteed_low_digits`] computes it in floating point. Check it
    /// exactly, in integers, for every base that enables either prefilter: n²
    /// and n³ at the range start must really have at least `digits` digits.
    #[test]
    fn enabled_prefilters_really_have_the_digits_they_peel() {
        let digit_count = |mut v: u128, base: u128| {
            let mut n = 0;
            while v != 0 {
                v /= base;
                n += 1;
            }
            n
        };
        let mut checked = 0;
        // Bases below 10 are not GPU bases at all and `get_base_range_u128`
        // panics on the degenerate ones, so the prefilter derivations are only
        // ever asked about supported bases.
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            let cuda = prefilter_params(base).map(|p| p.digits);
            let vulkan = vulkan_prefilter_params(base).map(|p| p.digits);
            let Some(digits) = cuda.into_iter().chain(vulkan).max() else {
                continue;
            };
            let range = base_range::get_base_range_u128(base).unwrap().unwrap();
            // The smallest candidate has the fewest digits in both powers.
            let n = range.range_start;
            let b = u128::from(base);
            assert!(
                digit_count(n * n, b) >= digits,
                "base {base}: n² of {n} has too few digits for {digits}"
            );
            assert!(
                digit_count(n * n * n, b) >= digits,
                "base {base}: n³ of {n} has too few digits for {digits}"
            );
            checked += 1;
        }
        assert!(checked >= 5, "only {checked} bases enable a prefilter");
    }

    /// The Vulkan prefilter must never reach further than the CUDA one where
    /// both are enabled: its chunks are capped at 2^16 and CUDA's modulus at
    /// 2^48, so three chunks can only be shorter. Equal is the good case —
    /// base 40, the one live base for either, gets 9 digits both ways.
    #[test]
    fn vulkan_prefilter_is_no_longer_than_cudas() {
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            if let (Some(cuda), Some(vulkan)) = (prefilter_params(base), vulkan_prefilter_params(base))
            {
                assert!(
                    vulkan.digits <= cuda.digits,
                    "base {base}: vulkan {} > cuda {}",
                    vulkan.digits,
                    cuda.digits
                );
            }
        }
        assert_eq!(vulkan_prefilter_params(40).map(|p| p.digits), Some(9));
        assert_eq!(prefilter_params(40).map(|p| p.digits), Some(9));
    }

    /// Same guard as the CUDA path's: small bases whose n² is shorter than the
    /// digits the filter would peel must be refused, and bases past the
    /// profitability threshold compiled out.
    #[test]
    fn vulkan_prefilter_guard_matches_the_cuda_gates() {
        assert!(vulkan_prefilter_params(10).is_none());
        assert!(vulkan_prefilter_params(40).is_some());
        for base in [42, 52, 62, 68] {
            assert!(
                vulkan_prefilter_params(base).is_none(),
                "expected prefilter disabled at b{base}"
            );
        }
    }

    #[test]
    fn n_limbs_matches_the_range_width() {
        // Base 40's band tops out below 2^64, so two limbs.
        assert_eq!(n_limbs(40), Some(2));
        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            let limbs = n_limbs(base).expect("supported base has a range");
            assert!((1..=4).contains(&limbs), "base {base}: {limbs} limbs");
        }
    }
}

