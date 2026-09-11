/-
Positional digits.

Everything the Rust does is `n / b^j % b`; `Nat.digits` is the list form
Mathlib reasons about. This file defines the positional form and proves the
bridge lemmas so either can be used. Positions count from the least
significant digit, matching malachite's `to_digits_asc` and every extraction
loop in `common/src`.
-/
import Mathlib.Data.Nat.Digits.Defs
import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Data.Nat.Log
import Mathlib.Data.List.GetD

namespace Nice

/-- Digit `j` of `n` in base `b`, least significant first. -/
def digit (b n j : ℕ) : ℕ := n / b ^ j % b

/-- Number of base-`b` digits of `n`; `numDigits b 0 = 0`, as in `Nat.digits`. -/
def numDigits (b n : ℕ) : ℕ := (Nat.digits b n).length

theorem digit_lt_base {b : ℕ} (hb : 0 < b) (n j : ℕ) : digit b n j < b :=
  Nat.mod_lt _ hb

/-- The positional digit is the list digit (`0` past the end). -/
theorem digit_eq_getD {b : ℕ} (hb : 2 ≤ b) (n j : ℕ) :
    digit b n j = (Nat.digits b n).getD j 0 := by
  unfold digit
  rw [Nat.getD_digits n j hb]

/-- Digits at or above the length are zero. -/
theorem digit_eq_zero_of_le {b : ℕ} (hb : 2 ≤ b) {n j : ℕ} (h : numDigits b n ≤ j) :
    digit b n j = 0 := by
  rw [digit_eq_getD hb]
  exact List.getD_eq_default _ _ h

/-- `j < numDigits b n ↔ b ^ j ≤ n` for `n ≠ 0`: the form the range lemmas use. -/
theorem lt_numDigits_iff {b : ℕ} (hb : 2 ≤ b) {n : ℕ} (hn : n ≠ 0) (j : ℕ) :
    j < numDigits b n ↔ b ^ j ≤ n := by
  unfold numDigits
  rw [Nat.length_digits b n hb hn, Nat.lt_succ_iff, Nat.le_log_iff_pow_le hb hn]

/-- `numDigits b n ≤ j ↔ n < b ^ j` for `n ≠ 0`. -/
theorem numDigits_le_iff {b : ℕ} (hb : 2 ≤ b) {n : ℕ} (hn : n ≠ 0) (j : ℕ) :
    numDigits b n ≤ j ↔ n < b ^ j := by
  rw [← Nat.not_lt, lt_numDigits_iff hb hn, Nat.not_le]

/-- Every number is below `b ^ numDigits`. -/
theorem lt_pow_numDigits {b : ℕ} (hb : 2 ≤ b) (n : ℕ) : n < b ^ numDigits b n :=
  Nat.lt_base_pow_length_digits hb

end Nice
