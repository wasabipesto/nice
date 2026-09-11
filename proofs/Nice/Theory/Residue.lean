/-
Dead base classes. Rust: `residue_filter.rs` (the oracle test's comment).

For `b ≡ 3 (mod 4)` the residue target `b(b-1)/2 mod (b-1)` is the odd
number `(b-1)/2`, while `n² + n³ = n²(n+1)` is always even, and the modulus
`b - 1` is even; so no residue passes and no nice number exists (RES-2).
Bases 11, 15, 19, 23, 27, 43, 47, … are dead for this reason.
-/
import Nice.Model.Residue

namespace Nice

theorem even_sq_add_cube (n : ℕ) : Even (n ^ 2 + n ^ 3) := by
  have h : n ^ 2 + n ^ 3 = n * (n * (n + 1)) := by ring
  rw [h]
  exact (Nat.even_mul_succ_self n).mul_left n

/-- The residue target for `b ≡ 3 (mod 4)` is the odd number `(b-1)/2`. -/
theorem residueTarget_of_three_mod_four {b : ℕ} (hb : b % 4 = 3) :
    b * (b - 1) / 2 % (b - 1) = (b - 1) / 2 ∧ ((b - 1) / 2) % 2 = 1 := by
  obtain ⟨t, ht⟩ : ∃ t, b = 2 * t + 1 := ⟨b / 2, by omega⟩
  have ht1 : 1 ≤ t := by omega
  subst ht
  have h1 : (2 * t + 1) * (2 * t + 1 - 1) / 2 = t * (2 * t + 1) := by
    rw [Nat.add_sub_cancel]
    have : (2 * t + 1) * (2 * t) = t * (2 * t + 1) * 2 := by ring
    rw [this, Nat.mul_div_cancel _ (by norm_num)]
  rw [h1, Nat.add_sub_cancel]
  constructor
  · have : t * (2 * t + 1) = 2 * t * t + t := by ring
    rw [this, Nat.mul_add_mod, Nat.mod_eq_of_lt (by omega)]
    omega
  · omega

/-- Claim RES-2: every base `b ≡ 3 (mod 4)` has no nice numbers. -/
theorem no_nice_of_three_mod_four {b : ℕ} (hb : b % 4 = 3) : ∀ n, ¬ IsNice b n := by
  intro n h
  have hb2 : 2 ≤ b := by omega
  have hres := nice_residue hb2 h
  obtain ⟨htgt, hodd⟩ := residueTarget_of_three_mod_four hb
  -- reduce the congruence mod (b-1) to mod 2
  have h2 : 2 ∣ b - 1 := by omega
  have hmod2 : n ^ 2 + n ^ 3 ≡ b * (b - 1) / 2 [MOD 2] := Nat.ModEq.of_dvd h2 hres
  have hleft : (n ^ 2 + n ^ 3) % 2 = 0 := Nat.even_iff.mp (even_sq_add_cube n)
  have hright : b * (b - 1) / 2 % 2 = 1 := by
    -- b(b-1)/2 = (b-1) * ((b-1)/2 / ... ); compute via the target
    have hdiv := Nat.div_add_mod (b * (b - 1) / 2) (b - 1)
    rw [htgt] at hdiv
    -- b(b-1)/2 = (b-1) * q + (b-1)/2 with (b-1) even and (b-1)/2 odd
    have : (b - 1) * (b * (b - 1) / 2 / (b - 1)) % 2 = 0 := by
      apply Nat.even_iff.mp
      exact (Nat.even_iff.mpr (by omega : (b - 1) % 2 = 0)).mul_right _
    omega
  unfold Nat.ModEq at hmod2
  omega

/-- Base 11, the smallest dead base, as an instance. -/
example : ∀ n, ¬ IsNice 11 n := no_nice_of_three_mod_four (by norm_num)

end Nice
