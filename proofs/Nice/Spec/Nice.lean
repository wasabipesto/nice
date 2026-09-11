/-
The definition of a nice number and its immediate consequences.

`IsNice b n`: the base-`b` digits of `n²` followed by those of `n³` are a
permutation of `0, …, b-1`. Rust: `client_process::get_is_nice`.
-/
import Nice.Spec.Digits
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Group.List.Basic

namespace Nice

/-- The concatenated output digits of `n`, least significant first within each power. -/
def outputDigits (b n : ℕ) : List ℕ := Nat.digits b (n ^ 2) ++ Nat.digits b (n ^ 3)

/-- `n` is nice in base `b` (claim DEF-1). -/
def IsNice (b n : ℕ) : Prop := (outputDigits b n).Perm (List.range b)

instance (b n : ℕ) : Decidable (IsNice b n) := by unfold IsNice; infer_instance

/-- Distinct output digits; detailed mode's `num_uniques` (claim DEF-2). -/
def numUniques (b n : ℕ) : ℕ := (outputDigits b n).toFinset.card

/-- The three-part definition the `origin/proofs` branch used. -/
def Pandigital (b : ℕ) (ds : List ℕ) : Prop :=
  ds.length = b ∧ ds.Nodup ∧ ∀ d, d < b ↔ d ∈ ds

theorem pandigital_iff_perm {b : ℕ} {ds : List ℕ} :
    Pandigital b ds ↔ ds.Perm (List.range b) := by
  constructor
  · rintro ⟨_, hnd, hmem⟩
    apply List.perm_of_nodup_nodup_toFinset_eq hnd List.nodup_range
    ext x
    simp only [List.mem_toFinset, List.mem_range]
    exact (hmem x).symm
  · intro h
    refine ⟨by simpa using h.length_eq, h.nodup_iff.mpr List.nodup_range, fun d => ?_⟩
    rw [h.mem_iff, List.mem_range]

/-- Claim DEF-1a. -/
theorem isNice_iff_pandigital {b n : ℕ} : IsNice b n ↔ Pandigital b (outputDigits b n) :=
  pandigital_iff_perm.symm

/-- Claim RNG-1: a nice number's powers have `b` digits between them. -/
theorem nice_digit_count {b n : ℕ} (h : IsNice b n) :
    numDigits b (n ^ 2) + numDigits b (n ^ 3) = b := by
  have := h.length_eq
  simpa [outputDigits, numDigits] using this

/-- Gauss, list form. -/
theorem sum_range_mul_two (n : ℕ) : (List.range n).sum * 2 = n * (n - 1) := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.sum_append, List.sum_singleton, add_mul, ih]
    cases m with
    | zero => simp
    | succ k => simp only [Nat.add_sub_cancel]; ring

theorem sum_range_eq (n : ℕ) : (List.range n).sum = n * (n - 1) / 2 := by
  rw [← sum_range_mul_two, Nat.mul_div_cancel _ (by norm_num)]

/-- Claim RES-1a: the output digits of a nice number sum to `b(b-1)/2`. -/
theorem nice_digit_sum {b n : ℕ} (h : IsNice b n) :
    (outputDigits b n).sum = b * (b - 1) / 2 := by
  rw [h.sum_eq, sum_range_eq]

end Nice
