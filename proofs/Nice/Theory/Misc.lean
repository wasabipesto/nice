/-
Smaller structural facts: the complement digit sum is determined (THY-3),
window filters are sound (THY-5), and the interval-domain Hall relaxation
is strictly incomplete (THY-6).
-/
import Nice.Model.Cross

namespace Nice

/-- Claim THY-3: once some output digits are fixed, the rest sum to the
complement — a digit-sum window on the unassigned positions is vacuous. -/
theorem complement_sum {b n : ℕ} {l₁ l₂ : List ℕ} (h : IsNice b n)
    (hsplit : outputDigits b n = l₁ ++ l₂) : l₂.sum = b * (b - 1) / 2 - l₁.sum := by
  have := nice_digit_sum h
  rw [hsplit, List.sum_append] at this
  omega

/-- And the complement digit *set* is determined too. -/
theorem complement_set {b n : ℕ} {l₁ l₂ : List ℕ} (h : IsNice b n)
    (hsplit : outputDigits b n = l₁ ++ l₂) : l₂.toFinset = Finset.range b \ l₁.toFinset := by
  have hnd : (l₁ ++ l₂).Nodup := hsplit ▸ h.nodup_iff.mpr List.nodup_range
  have hall : (l₁ ++ l₂).toFinset = Finset.range b := by
    rw [← hsplit, ← List.toFinset_range]
    exact List.toFinset_eq_of_perm _ _ h
  ext d
  rw [List.toFinset_append] at hall
  rw [Finset.mem_sdiff, ← hall, Finset.mem_union]
  simp only [List.mem_toFinset]
  constructor
  · intro hd
    refine ⟨Or.inr hd, fun hd1 => ?_⟩
    exact (List.nodup_append.mp hnd).2.2 d hd1 d hd rfl
  · rintro ⟨h1 | h1, h2⟩
    · exact absurd h1 h2
    · exact h1

/-- Claim THY-5: a window at positions `p ≤ j < p + w` of `n^e` depends only
on `n mod b^(p+w)`, so a window filter on exactly those digits is sound
(the shelved middle-window filter). -/
theorem window_sound {b n e p w j : ℕ} (hj : j < p + w) :
    digit b (n ^ e) j = digit b ((n % b ^ (p + w)) ^ e) j :=
  digit_pow_mod_pow e hj

/-- Claim THY-6: the interval-domain Hall check is strictly incomplete: this
base-10 range has a system of distinct representatives (so the filter keeps
it) but no nice number. -/
theorem hall_relaxation_incomplete :
    analyzeRange 10 47 60 = true ∧ ∀ n ∈ Finset.Icc 47 60, ¬ IsNice 10 n := by
  decide

/-- The size of the search range in closed form (`0` for bases without one). -/
noncomputable def baseRangeSize (b : ℕ) : ℕ :=
  match baseRange b with
  | some (lo, hi) => hi - lo
  | none => 0

/-- Claim THY-9: the random-digit witness model, `λ_b = |range_b| · b! / b^b`,
the expected number of nice numbers in base `b` if output digits were
uniform and independent. A heuristic, recorded as a definition only; the
project's cost model is `λ_b` times an empirical tail correction. -/
noncomputable def witnessRate (b : ℕ) : ℚ :=
  (baseRangeSize b : ℚ) * (Nat.factorial b : ℚ) / ((b : ℚ) ^ b)

end Nice
