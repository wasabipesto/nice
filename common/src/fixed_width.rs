//! Stack-resident fixed-width unsigned integer arithmetic for the hot loop.
//!
//! `malachite::Natural` allocates on the heap and uses dynamic-length limb
//! storage. For the bases of interest (≤80), n² and n³ fit in at most 304
//! bits, which means a 4-limb u256 suffices. Skipping the heap and using
//! native u64×u64→u128 multiplies (CPU `MUL`/`UMULH` pair) brings each
//! pow-and-extract round into the 100s-of-cycles range instead of the
//! 1000s.
//!
//! ## Bit-width budget
//!
//! | base | n bits | n² bits | n³ bits | needed |
//! |------|--------|---------|---------|--------|
//! | 40   | 43     | 86      | 128     | u128   |
//! | 50   | 57     | 113     | 170     | u256   |
//! | 60   | 71     | 142     | 213     | u256   |
//! | 80   | 102    | 203     | 304     | u512   |
//! | 97   | 126    | 251     | 377     | u512   |
//!
//! Bases ≥ 69 fall back to malachite `Natural` in the caller (n³ exceeds
//! 256 bits at base 70).

#![allow(clippy::pedantic)]

/// 256-bit unsigned integer, little-endian limbs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct U256 {
    pub limbs: [u64; 4],
}

impl U256 {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { limbs: [0; 4] }
    }

    #[inline(always)]
    pub const fn from_u128(x: u128) -> Self {
        Self {
            limbs: [x as u64, (x >> 64) as u64, 0, 0],
        }
    }

    #[inline(always)]
    pub const fn is_zero(&self) -> bool {
        (self.limbs[0] | self.limbs[1] | self.limbs[2] | self.limbs[3]) == 0
    }

    /// Compute `a * b` as a u256.
    /// `a` and `b` are each up to 128 bits; result is up to 256 bits.
    #[inline]
    pub fn mul_u128_u128(a: u128, b: u128) -> Self {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        // Four 64×64 → 128 partial products
        let p_ll = (a_lo as u128) * (b_lo as u128);
        let p_lh = (a_lo as u128) * (b_hi as u128);
        let p_hl = (a_hi as u128) * (b_lo as u128);
        let p_hh = (a_hi as u128) * (b_hi as u128);

        // Position (in u64 limbs):
        //   p_ll → limb 0..1
        //   p_lh → limb 1..2
        //   p_hl → limb 1..2
        //   p_hh → limb 2..3
        let limb0 = p_ll as u64;

        let mid =
            (p_ll >> 64) + (p_lh & 0xFFFF_FFFF_FFFF_FFFF) + (p_hl & 0xFFFF_FFFF_FFFF_FFFF);
        let limb1 = mid as u64;

        let upper = (mid >> 64) + (p_lh >> 64) + (p_hl >> 64) + (p_hh & 0xFFFF_FFFF_FFFF_FFFF);
        let limb2 = upper as u64;

        let limb3 = ((upper >> 64) + (p_hh >> 64)) as u64;

        Self {
            limbs: [limb0, limb1, limb2, limb3],
        }
    }

    /// Compute `self * b` truncated to 256 bits.
    /// Caller guarantees the true product fits in 256 bits (true for n³ at
    /// base ≤ 80).
    #[inline]
    pub fn mul_u128_truncating(&self, b: u128) -> Self {
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        // Accumulate partial products into wide slots, then resolve carries.
        // We need the result mod 2^256, so anything above limb 3 is discarded.
        let s = &self.limbs;

        // limb 0: s[0]*b_lo (low)
        let t00 = (s[0] as u128) * (b_lo as u128);
        let limb0 = t00 as u64;
        let mut carry: u128 = t00 >> 64;

        // limb 1: s[0]*b_hi (low) + s[1]*b_lo (low) + carry
        let t01 = (s[0] as u128) * (b_hi as u128);
        let t10 = (s[1] as u128) * (b_lo as u128);
        let acc1 = (t01 & 0xFFFF_FFFF_FFFF_FFFF) + (t10 & 0xFFFF_FFFF_FFFF_FFFF) + carry;
        let limb1 = acc1 as u64;
        carry = (acc1 >> 64) + (t01 >> 64) + (t10 >> 64);

        // limb 2: s[1]*b_hi (low) + s[2]*b_lo (low) + carry
        let t11 = (s[1] as u128) * (b_hi as u128);
        let t20 = (s[2] as u128) * (b_lo as u128);
        let acc2 = (t11 & 0xFFFF_FFFF_FFFF_FFFF) + (t20 & 0xFFFF_FFFF_FFFF_FFFF) + carry;
        let limb2 = acc2 as u64;
        carry = (acc2 >> 64) + (t11 >> 64) + (t20 >> 64);

        // limb 3: s[2]*b_hi (low) + s[3]*b_lo (low) + carry
        // Anything overflowing here is discarded (truncating).
        let t21 = (s[2] as u128) * (b_hi as u128);
        let t30 = (s[3] as u128) * (b_lo as u128);
        let acc3 = (t21 & 0xFFFF_FFFF_FFFF_FFFF) + (t30 & 0xFFFF_FFFF_FFFF_FFFF) + carry;
        let limb3 = acc3 as u64;

        Self {
            limbs: [limb0, limb1, limb2, limb3],
        }
    }

    /// In-place division by a small u32 divisor, returns the remainder.
    /// Standard long-division by a single-limb divisor.
    ///
    /// Skips work on the leading-zero limbs once we've found the highest
    /// non-zero limb. For the bases of interest (50–60) n³ uses 3 limbs, so
    /// this saves ~25% of the per-digit division work after the first call.
    #[inline]
    pub fn div_assign_rem_u32(&mut self, divisor: u32) -> u32 {
        let d = divisor as u128;
        let mut rem: u128 = 0;

        // Find highest non-zero limb to skip work on the leading zeros.
        let top = if self.limbs[3] != 0 {
            3
        } else if self.limbs[2] != 0 {
            2
        } else if self.limbs[1] != 0 {
            1
        } else {
            0
        };

        for i in (0..=top).rev() {
            let cur = (rem << 64) | (self.limbs[i] as u128);
            let q = cur / d;
            rem = cur % d;
            self.limbs[i] = q as u64;
        }
        rem as u32
    }
}

/// Divide a u128 by a small u32 divisor.
/// Plain `u128 / u32` on x86-64 calls `__udivti3` which doesn't exploit the
/// small-divisor case efficiently. We do limb-by-limb: high u64 / d_u64
/// (single native DIV), then a 96-bit/64-bit step. The second step still goes
/// through `__udivti3` but the dividend now fits in 96 bits which is
/// substantially faster than dividing a full 128-bit value.
///
/// Inline asm with x86-64 `DIV r/m64` was tried — it actually *regressed*
/// b40 throughput by ~12%, likely due to register-pressure pessimization
/// preventing surrounding code from being optimally scheduled. LLVM's
/// generated sequence (with the dividend known to fit in 96 bits) wins.
#[inline(always)]
pub fn div_rem_u128_small(n: u128, d: u32) -> (u128, u32) {
    let d128 = d as u128;
    let n_hi = (n >> 64) as u64;
    let n_lo = n as u64;
    let d_u64 = d as u64;

    // Step 1: u64 / u64. Native single DIV.
    let q_hi = (n_hi / d_u64) as u128;
    let r_hi = (n_hi % d_u64) as u128;

    // Step 2: combined fits in 96 bits — LLVM emits a fast sequence here.
    let combined = (r_hi << 64) | (n_lo as u128);
    let q_lo = combined / d128;
    let r = (combined % d128) as u32;

    ((q_hi << 64) | q_lo, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn natural_to_4_limbs(n: &malachite::natural::Natural) -> [u64; 4] {
        // `to_limbs_asc` is inherent on Natural and returns Vec<Limb> (u64 on 64-bit).
        let limbs = n.to_limbs_asc();
        let mut out = [0u64; 4];
        for (i, l) in limbs.iter().take(4).enumerate() {
            out[i] = *l;
        }
        out
    }

    fn naive_mul128_to_256(a: u128, b: u128) -> [u64; 4] {
        use malachite::natural::Natural;
        let prod = Natural::from(a) * Natural::from(b);
        natural_to_4_limbs(&prod)
    }

    #[test]
    fn test_zero_and_from_u128() {
        let z = U256::zero();
        assert!(z.is_zero());
        let v = U256::from_u128(0);
        assert!(v.is_zero());
        let v = U256::from_u128(42);
        assert_eq!(v.limbs, [42, 0, 0, 0]);
        let v = U256::from_u128(u128::MAX);
        assert_eq!(v.limbs, [u64::MAX, u64::MAX, 0, 0]);
    }

    #[test]
    fn test_mul_u128_u128_small() {
        let r = U256::mul_u128_u128(7, 6);
        assert_eq!(r.limbs, [42, 0, 0, 0]);
    }

    #[test]
    fn test_mul_u128_u128_at_boundary() {
        // 2^64 * 2^64 = 2^128 → limbs [0, 0, 1, 0]
        let r = U256::mul_u128_u128(1u128 << 64, 1u128 << 64);
        assert_eq!(r.limbs, [0, 0, 1, 0]);
    }

    #[test]
    fn test_mul_u128_u128_random() {
        // Verify against malachite for a handful of values
        let cases: &[(u128, u128)] = &[
            (1u128 << 100, 3u128 << 70),
            (u128::MAX, u128::MAX),
            (123_456_789_012_345_678_901_234_567_890u128, 999_999_999_999u128),
            (u64::MAX as u128, u64::MAX as u128),
            (6_553_600_000_000u128, 6_553_600_000_000u128), // base-40 max squared
        ];
        for &(a, b) in cases {
            let got = U256::mul_u128_u128(a, b);
            let want = naive_mul128_to_256(a, b);
            assert_eq!(got.limbs, want, "mul_u128_u128({a}, {b}) mismatch");
        }
    }

    #[test]
    fn test_mul_u128_truncating_against_natural() {
        use malachite::base::num::arithmetic::traits::Pow;
        use malachite::natural::Natural;

        // Realistic n values for bases up to 68 (where n^3 fits u256).
        // Base 40 max, base 50 max, base 60 max, base 68 max-ish.
        for &n in &[
            6_553_599_999_999u128,                // base 40 max-ish
            26_507_984_537_059_635u128,           // base 50 start
            2_176_782_335_999_999_999_999u128,    // base 60 max-ish
            6_500_000_000_000_000_000_000_000u128, // base 68 max-ish
        ] {
            let n_sq_u256 = U256::mul_u128_u128(n, n);
            let n_cu_u256 = n_sq_u256.mul_u128_truncating(n);

            let n_cu_nat = (&Natural::from(n)).pow(3u64);
            let want = natural_to_4_limbs(&n_cu_nat);
            assert_eq!(n_cu_u256.limbs, want, "n^3 mismatch for n={n}");
        }
    }

    #[test]
    fn test_div_assign_rem_u32() {
        // Verify against malachite for some random-ish values across bases
        use malachite::base::num::arithmetic::traits::{DivAssignRem, Pow};
        use malachite::natural::Natural;

        for &n in &[
            6_553_599_999_999u128,
            26_507_984_537_059_635u128,
            2_176_782_335_999_999_999_999u128,
        ] {
            let n_sq_u256 = U256::mul_u128_u128(n, n);
            let n_cu_u256 = n_sq_u256.mul_u128_truncating(n);

            for &base in &[10u32, 40, 50, 60] {
                let mut x = n_cu_u256;
                let mut digits_u256 = Vec::new();
                while !x.is_zero() {
                    digits_u256.push(x.div_assign_rem_u32(base));
                }

                let mut x_nat = (&Natural::from(n)).pow(3u64);
                let base_nat = Natural::from(base);
                let mut digits_nat = Vec::new();
                while x_nat > 0 {
                    let r = u32::try_from(&x_nat.div_assign_rem(&base_nat)).unwrap();
                    digits_nat.push(r);
                }

                assert_eq!(digits_u256, digits_nat, "digit stream mismatch n={n} base={base}");
            }
        }
    }
}
