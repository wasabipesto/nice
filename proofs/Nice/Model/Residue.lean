/-
The residue filter (mod b-1). Rust: `residue_filter::get_residue_filter`.

A nice number's output digits sum to `b(b-1)/2`, and a digit sum is
congruent to the number mod `b-1`, so `n² + n³ ≡ b(b-1)/2 (mod b-1)`.
The filter keeps the residues `r < b-1` with `r² + r³ ≡ b(b-1)/2`.
-/
import Nice.Spec.Nice
import Mathlib.Data.Nat.Digits.Lemmas

namespace Nice

/-- The target residue: `b(b-1)/2 mod (b-1)`. -/
def residueTarget (b : ℕ) : ℕ := b * (b - 1) / 2 % (b - 1)

/-- Exactly the Rust table: residues `r < b-1` with `(r² + r³) % (b-1) = target`. -/
def residueFilter (b : ℕ) : Finset ℕ :=
  (Finset.range (b - 1)).filter fun r => (r ^ 2 + r ^ 3) % (b - 1) = residueTarget b

/-- Digit sums are congruent to the number mod `b - 1` (`Nat.modEq_digits_sum`
specialised; trivial for `b = 2` where the modulus is `1`). -/
theorem modEq_digits_sum_sub_one {b : ℕ} (hb : 2 ≤ b) (m : ℕ) :
    m ≡ (Nat.digits b m).sum [MOD b - 1] := by
  rcases Nat.lt_or_ge 2 b with h | h
  · obtain ⟨k, rfl⟩ : ∃ k, b = k + 1 := ⟨b - 1, by omega⟩
    rw [Nat.add_sub_cancel]
    refine Nat.modEq_digits_sum k (k + 1) ?_ m
    rw [Nat.add_mod_left]
    exact Nat.mod_eq_of_lt (by omega)
  · have : b = 2 := by omega
    subst this
    simp [Nat.ModEq, Nat.mod_one]

/-- Claim RES-1: a nice number satisfies the residue congruence. -/
theorem nice_residue {b n : ℕ} (hb : 2 ≤ b) (h : IsNice b n) :
    n ^ 2 + n ^ 3 ≡ b * (b - 1) / 2 [MOD b - 1] := by
  have hs := nice_digit_sum h
  unfold outputDigits at hs
  rw [List.sum_append] at hs
  calc n ^ 2 + n ^ 3
      ≡ (Nat.digits b (n ^ 2)).sum + (Nat.digits b (n ^ 3)).sum [MOD b - 1] :=
        Nat.ModEq.add (modEq_digits_sum_sub_one hb _) (modEq_digits_sum_sub_one hb _)
    _ = b * (b - 1) / 2 := hs

/-- Claim RES-1: the Rust filter keeps every nice number's residue. -/
theorem mem_residueFilter_of_isNice {b n : ℕ} (hb : 2 ≤ b) (h : IsNice b n) :
    n % (b - 1) ∈ residueFilter b := by
  unfold residueFilter residueTarget
  rw [Finset.mem_filter, Finset.mem_range]
  refine ⟨Nat.mod_lt _ (by omega), ?_⟩
  have hres : (n ^ 2 + n ^ 3) % (b - 1) = b * (b - 1) / 2 % (b - 1) := nice_residue hb h
  rw [← hres, Nat.add_mod (n ^ 2), Nat.pow_mod n 2, Nat.pow_mod n 3, ← Nat.add_mod]

/-- Claim RES-3: an empty filter means no nice numbers at all. -/
theorem no_nice_of_residueFilter_empty {b : ℕ} (hb : 2 ≤ b) (hemp : residueFilter b = ∅) :
    ∀ n, ¬ IsNice b n := fun n h => by
  have := mem_residueFilter_of_isNice hb h
  rw [hemp] at this
  exact Finset.notMem_empty _ this

/-- Base 11's filter is empty, so base 11 has no nice numbers. -/
theorem residueFilter_eleven : residueFilter 11 = ∅ := by decide

theorem no_nice_eleven : ∀ n, ¬ IsNice 11 n :=
  no_nice_of_residueFilter_empty (by norm_num) residueFilter_eleven

/-- The base-10 table the Rust test pins: `[0, 3, 6, 8]`. -/
theorem residueFilter_ten : residueFilter 10 = {0, 3, 6, 8} := by decide

end Nice
