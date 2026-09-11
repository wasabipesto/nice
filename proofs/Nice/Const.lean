/-
Numeric constants the Rust relies on, certified.

Each theorem here is a fact about a specific number the code hard-codes:
a width cutoff, a buffer size, a modulus bound. The Rust comments used to
state these with "empirically" or with numbers that had drifted; the
theorems are the reference now, and the `Lean:` tags at the Rust sites
point back here.
-/
import Nice.Spec.Range

-- `Nat.reducePow` refuses exponents above this; the bounds here go to 2^512.
set_option exponentiation.threshold 1024

namespace Nice.Const

open Nice

/-- Per class of `b mod 5`, the exponent `c` with `(hi - 1)^6 < b^(6k + c)`. -/
def classExp (b : ℕ) : ℕ :=
  match b % 5 with
  | 0 => 0
  | 2 => 2
  | 3 => 3
  | _ => 4

/-- The largest candidate's sixth power is below a power of `b` fixed by the
class; the bridge from the closed-form endpoints to every width bound. -/
theorem hi_pow_six_lt {b lo hi : ℕ} (hb : 2 ≤ b) (h : baseRange b = some (lo, hi)) :
    (hi - 1) ^ 6 < b ^ (6 * (b / 5) + classExp b) := by
  have hb0 : 0 < b := by omega
  have hmod : b % 5 < 5 := Nat.mod_lt _ (by norm_num)
  unfold baseRange at h
  unfold classExp
  interval_cases hr : b % 5
  · -- class 0: hi = b^k
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    have hk : 0 < b ^ (b / 5) := Nat.pow_pos hb0
    calc (b ^ (b / 5) - 1) ^ 6 < (b ^ (b / 5)) ^ 6 :=
          Nat.pow_lt_pow_left (Nat.sub_lt hk (by norm_num)) (by norm_num)
      _ = b ^ (6 * (b / 5) + 0) := by rw [← pow_mul]; ring_nf
  · simp at h
  · -- class 2: hi = ceilRoot 3 (b^(3k+1))
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    rcases Nat.eq_zero_or_pos (ceilRoot 3 (b ^ (3 * (b / 5) + 1))) with h0 | hpos
    · rw [h0]
      have h6 : (0 - 1 : ℕ) ^ 6 = 0 := by norm_num
      rw [h6]
      exact Nat.pow_pos hb0
    have := pow_lt_of_lt_ceilRoot (e := 3) (by norm_num) (Nat.sub_lt hpos Nat.one_pos)
    calc (ceilRoot 3 (b ^ (3 * (b / 5) + 1)) - 1) ^ 6
        = ((ceilRoot 3 (b ^ (3 * (b / 5) + 1)) - 1) ^ 3) ^ 2 := by rw [← pow_mul]
      _ < (b ^ (3 * (b / 5) + 1)) ^ 2 := Nat.pow_lt_pow_left this (by norm_num)
      _ = b ^ (6 * (b / 5) + 2) := by rw [← pow_mul]; ring_nf
  · -- class 3: hi = ceilRoot 2 (b^(2k+1))
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    rcases Nat.eq_zero_or_pos (ceilRoot 2 (b ^ (2 * (b / 5) + 1))) with h0 | hpos
    · rw [h0]
      have h6 : (0 - 1 : ℕ) ^ 6 = 0 := by norm_num
      rw [h6]
      exact Nat.pow_pos hb0
    have := pow_lt_of_lt_ceilRoot (e := 2) (by norm_num) (Nat.sub_lt hpos Nat.one_pos)
    calc (ceilRoot 2 (b ^ (2 * (b / 5) + 1)) - 1) ^ 6
        = ((ceilRoot 2 (b ^ (2 * (b / 5) + 1)) - 1) ^ 2) ^ 3 := by rw [← pow_mul]
      _ < (b ^ (2 * (b / 5) + 1)) ^ 3 := Nat.pow_lt_pow_left this (by norm_num)
      _ = b ^ (6 * (b / 5) + 3) := by rw [← pow_mul]; ring_nf
  · -- class 4: hi = ceilRoot 3 (b^(3k+2))
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    rcases Nat.eq_zero_or_pos (ceilRoot 3 (b ^ (3 * (b / 5) + 2))) with h0 | hpos
    · rw [h0]
      have h6 : (0 - 1 : ℕ) ^ 6 = 0 := by norm_num
      rw [h6]
      exact Nat.pow_pos hb0
    have := pow_lt_of_lt_ceilRoot (e := 3) (by norm_num) (Nat.sub_lt hpos Nat.one_pos)
    calc (ceilRoot 3 (b ^ (3 * (b / 5) + 2)) - 1) ^ 6
        = ((ceilRoot 3 (b ^ (3 * (b / 5) + 2)) - 1) ^ 3) ^ 2 := by rw [← pow_mul]
      _ < (b ^ (3 * (b / 5) + 2)) ^ 2 := Nat.pow_lt_pow_left this (by norm_num)
      _ = b ^ (6 * (b / 5) + 4) := by rw [← pow_mul]; ring_nf

/-- Claim NUM-2: every cube the U256 path is given fits in 256 bits, for
every base through 69 (`MAX_BASE_FOR_FIXED_WIDTH_U256 = 68` has one base of
slack). -/
theorem u256_cutoff {b lo hi : ℕ} (hb : 2 ≤ b) (hb69 : b ≤ 69)
    (h : baseRange b = some (lo, hi)) : (hi - 1) ^ 3 < 2 ^ 256 := by
  have h6 := hi_pow_six_lt hb h
  have hbnd : b ^ (6 * (b / 5) + classExp b) ≤ 2 ^ 512 := by
    interval_cases b <;> norm_num [classExp]
  have : ((hi - 1) ^ 3) ^ 2 < (2 ^ 256) ^ 2 := by
    calc ((hi - 1) ^ 3) ^ 2 = (hi - 1) ^ 6 := by rw [← pow_mul]
      _ < b ^ (6 * (b / 5) + classExp b) := h6
      _ ≤ 2 ^ 512 := hbnd
      _ = (2 ^ 256) ^ 2 := by norm_num
  exact (Nat.pow_lt_pow_iff_left (by norm_num)).mp this

/-- Claim NUM-2: base 70 does not fit. -/
theorem u256_fails_at_70 {lo hi : ℕ} (h : baseRange 70 = some (lo, hi)) :
    2 ^ 256 ≤ (hi - 1) ^ 3 := by
  unfold baseRange at h
  simp only [Nat.reduceMod, Nat.reduceDiv, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨-, rfl⟩ := h
  norm_num

/-- Claim NUM-1: base 40's cubes fit in 128 bits (`MAX_BASE_FOR_FIXED_WIDTH_U128`). -/
theorem u128_cutoff_40 {lo hi : ℕ} (h : baseRange 40 = some (lo, hi)) :
    (hi - 1) ^ 3 < 2 ^ 128 := by
  unfold baseRange at h
  simp only [Nat.reduceMod, Nat.reduceDiv, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨-, rfl⟩ := h
  norm_num

/-- Claim NUM-1: the next base with a range, 42, does not fit in 128 bits. -/
theorem u128_fails_at_42 {lo hi : ℕ} (h : baseRange 42 = some (lo, hi)) :
    2 ^ 128 ≤ (hi - 1) ^ 3 := by
  unfold baseRange at h
  simp only [Nat.reduceMod, Nat.reduceDiv, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨-, hhi⟩ := h
  have hspec : 42 ^ (3 * 8 + 1) ≤ hi ^ 3 := by rw [← hhi]; exact le_pow_ceilRoot (by norm_num) _
  by_contra hc
  rw [Nat.not_le] at hc
  have h1 : (hi - 1) ^ 3 < (2 ^ 43) ^ 3 := by
    calc (hi - 1) ^ 3 < 2 ^ 128 := hc
      _ < (2 ^ 43) ^ 3 := by norm_num
  have h2 : hi - 1 < 2 ^ 43 := (Nat.pow_lt_pow_iff_left (by norm_num)).mp h1
  have h3 : hi ^ 3 ≤ (2 ^ 43) ^ 3 := Nat.pow_le_pow_left (by omega) 3
  have : (42 : ℕ) ^ (3 * 8 + 1) ≤ (2 ^ 43) ^ 3 := hspec.trans h3
  norm_num at this

/-- Claim NUM-3: cubes have at most 38 digits for every base up to 64
(`MAX_FW_DIGITS`). -/
theorem max_fw_digits {b n : ℕ} (hb : 2 ≤ b) (hb64 : b ≤ 64) (h : InBaseRange b n) :
    numDigits b (n ^ 3) ≤ 38 := by
  have := numDigits_cu_le_of_inBaseRange hb h
  omega

/-- Claim NUM-4: the stride modulus `(b-1)·b³` fits a `u32` for bases up to 256. -/
theorem stride_modulus_u32 {b : ℕ} (hb : b ≤ 256) : (b - 1) * b ^ 3 < 2 ^ 32 :=
  calc (b - 1) * b ^ 3 ≤ 255 * 256 ^ 3 := Nat.mul_le_mul (by omega) (Nat.pow_le_pow_left hb 3)
    _ < 2 ^ 32 := by norm_num

/-- Claim NUM-4: and below `2^28` (`MAX_STRIDE_MODULUS`) for bases up to 128. -/
theorem stride_modulus_gpu {b : ℕ} (hb : b ≤ 128) : (b - 1) * b ^ 3 < 2 ^ 28 :=
  calc (b - 1) * b ^ 3 ≤ 127 * 128 ^ 3 := Nat.mul_le_mul (by omega) (Nat.pow_le_pow_left hb 3)
    _ < 2 ^ 28 := by norm_num

/-- Claim NUM-5: a digit of a base up to 64 indexes a `u64` mask. -/
theorem mask_width {b n j : ℕ} (hb : 0 < b) (hb64 : b ≤ 64) : digit b n j < 64 :=
  lt_of_lt_of_le (digit_lt_base hb n j) hb64

/-- Claim NUM-8: the CUDA `mod_m` combination `hi·(2^64 mod M) + lo` stays
below `2^64` whenever `M` fits a `u32`. -/
theorem mod_m_bound {M : ℕ} (hM : M < 2 ^ 32) : M * M + M < 2 ^ 64 :=
  calc M * M + M = M * (M + 1) := by ring
    _ ≤ (2 ^ 32 - 1) * 2 ^ 32 := Nat.mul_le_mul (by omega) (by omega)
    _ < 2 ^ 64 := by norm_num

/-- Claim NUM-9: 64 batches of 5·10⁷ candidates cannot overflow a `u32`
histogram bin (`DRAIN_INTERVAL · CUBECL_BATCH_SIZE`). -/
theorem histogram_bins : 64 * 50_000_000 < 2 ^ 32 := by norm_num

end Nice.Const
