/-
Proposed filters that are unsound, each with its witness. A theorem here is
a result: it is what stops the idea being re-derived.
-/
import Nice.Model.Lsd

namespace Nice

/-- Claim REF-1: the cross MSD×LSD skip shipped in v3.2.12–v3.2.15 treated
`first / b^k = last / b^k` as "the low digits are constant over the range"
and used the first element's low digits for every `n`. Base 10, `k = 2`,
range `[68, 70)`: both share the quotient `0`, 68's low square digits
collide with the square's fixed leading digit, so the range was skipped,
but 69 is nice. Concretely: the quotient test passes, the low digits are
not constant, and the range holds a nice number. -/
theorem msd_lsd_skip_unsound :
    68 / 10 ^ 2 = 69 / 10 ^ 2 ∧
    lowDigits 10 2 (68 ^ 2) ≠ lowDigits 10 2 (69 ^ 2) ∧
    IsNice 10 69 := by
  decide

/-- The general fact behind it: no range of size above one has constant
`n mod b^k` (for `b ≥ 2`, `k ≥ 1`), so a per-range low-digit claim is never
justified. -/
theorem mod_pow_not_constant {b k n : ℕ} (hb : 2 ≤ b) (hk : 1 ≤ k) :
    n % b ^ k ≠ (n + 1) % b ^ k := by
  intro h
  have hpos : 1 < b ^ k := by
    calc 1 < b := by omega
      _ = b ^ 1 := (pow_one b).symm
      _ ≤ b ^ k := Nat.pow_le_pow_right (by omega) hk
  have := Nat.sub_mod_eq_zero_of_mod_eq h.symm
  rw [Nat.add_sub_cancel_left] at this
  rw [Nat.mod_eq_of_lt hpos] at this
  exact one_ne_zero this

end Nice
