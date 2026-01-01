/-
  Formal verification of properties of square-cube pandigital ("nice") numbers.

  A number n is "nice" in base b if the concatenation of digits of n² and n³
  (in base b) contains all digits 0..b-1 exactly once (i.e., is pandigital).

  This file provides:
  1. Core definitions (digits, pandigitality, niceness)
  2. Soundness proofs for filtering rules used in the computational search
  3. Search space bounds and interval constraints
  4. Modular arithmetic constraints (residue class filtering)

  The goal is to provide machine-checked correctness guarantees for the
  computational search infrastructure, not to perform the search itself.
-/

import Mathlib.Data.List.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Tactic

open Nat List Finset

namespace Nice

/-! ## Core Definitions -/

/-- A list of digits is pandigital in base b if it contains each digit 0..b-1 exactly once. -/
def Pandigital (b : ℕ) (ds : List ℕ) : Prop :=
  ds.length = b ∧
  ds.Nodup ∧
  ∀ d, d < b ↔ d ∈ ds

/-- A number n is "nice" in base b if the concatenation of the digits of n² and n³
    forms a pandigital sequence in base b. -/
def IsNice (n b : ℕ) : Prop :=
  b ≥ 2 ∧
  Pandigital b (digits b (n^2) ++ digits b (n^3))

/-! ## Basic Properties -/

/-- The sum of all digits in a list. -/
def digitSum (ds : List ℕ) : ℕ := ds.sum

/-- The digit sum of a number in base b. -/
def digitSumBase (b n : ℕ) : ℕ := digitSum (digits b n)

/-- The number of digits in the base-b representation of n. -/
def numDigits (b n : ℕ) : ℕ := (digits b n).length

/-- Count the number of unique digits in a list. -/
def numUniqueDigits (ds : List ℕ) : ℕ := ds.toFinset.card

/-- If a list is pandigital in base b, it has exactly b elements. -/
theorem pandigital_length {b : ℕ} {ds : List ℕ} (h : Pandigital b ds) :
    ds.length = b := h.1

/-- If a list is pandigital in base b, all digits are distinct. -/
theorem pandigital_nodup {b : ℕ} {ds : List ℕ} (h : Pandigital b ds) :
    ds.Nodup := h.2.1

/-- If a list is pandigital in base b, it contains all digits less than b. -/
theorem pandigital_complete {b : ℕ} {ds : List ℕ} (h : Pandigital b ds) :
    ∀ d, d < b → d ∈ ds := fun d hd => (h.2.2 d).mp hd

/-- A pandigital list is a permutation of [0, 1, ..., b-1]. -/
theorem pandigital_perm {b : ℕ} {ds : List ℕ} (h : Pandigital b ds) :
    ds ~ (List.range b) := by
  apply List.perm_of_nodup_nodup_toFinset_eq h.2.1 List.nodup_range
  ext x
  simp only [List.mem_toFinset, List.mem_range]
  exact (h.2.2 x).symm

/-- The digit sum of a pandigital sequence in base b is b*(b-1)/2.
    This follows from the fact that 0 + 1 + ... + (b-1) = b*(b-1)/2 (Gauss formula). -/
theorem pandigital_digit_sum {b : ℕ} {ds : List ℕ} (h : Pandigital b ds) :
    digitSum ds = b * (b - 1) / 2 := by
  sorry

/-! ## Step 2: Digit Count Necessity -/

/-- If n is nice in base b, then the total number of digits in n² and n³ equals b. -/
theorem nice_digit_count {n b : ℕ} (h : IsNice n b) :
    numDigits b (n^2) + numDigits b (n^3) = b := by
  unfold numDigits
  have hpan : Pandigital b (digits b (n^2) ++ digits b (n^3)) := h.2
  have hlen := pandigital_length hpan
  simp only [List.length_append] at hlen
  exact hlen

/-! ## Step 3: Interval Bounds -/

/-- If b^k ≤ n² < b^(k+1), then n² has exactly k+1 digits in base b (for n ≥ 1, b ≥ 2). -/
theorem digit_count_from_power_bounds {b k n : ℕ} (hb : b ≥ 2) (hn : n ≥ 1)
    (hlow : b ^ k ≤ n ^ 2) (hhigh : n ^ 2 < b ^ (k + 1)) :
    numDigits b (n^2) = k + 1 := by
  unfold numDigits
  have hn2_pos : n ^ 2 ≠ 0 := pow_ne_zero 2 (Nat.one_le_iff_ne_zero.mp hn)
  rw [Nat.digits_len b (n^2) hb hn2_pos]
  have hlog := Nat.log_eq_of_pow_le_of_lt_pow hlow hhigh
  omega

/-- If n is nice in base b, and n² has k+1 digits and n³ has ℓ+1 digits, then k+ℓ+2 = b. -/
theorem nice_power_digit_sum {n b k ℓ : ℕ} (h : IsNice n b)
    (h2 : numDigits b (n ^ 2) = k + 1)
    (h3 : numDigits b (n ^ 3) = ℓ + 1) :
    k + ℓ + 2 = b := by
  have hcount := nice_digit_count h
  rw [h2, h3] at hcount
  omega

/-- Given the digit count constraint k + ℓ + 2 = b, we can bound n. -/
theorem nice_interval_bounds {n b : ℕ} (h : IsNice n b) :
    ∃ k ℓ : ℕ, k + ℓ + 2 = b ∧
               b^k ≤ n^2 ∧ n^2 < b^(k+1) ∧
               b^ℓ ≤ n^3 ∧ n^3 < b^(ℓ+1) := by
  sorry

/-! ## Step 4: Modular Constraints -/

/-- The digit sum of n in base b is congruent to n modulo (b-1). -/
theorem digit_sum_mod {b n : ℕ} (hb : b ≥ 2) :
    digitSumBase b n ≡ n [MOD b - 1] := by
  sorry

/-- If n is nice in base b, then n² + n³ ≡ b*(b-1)/2 [MOD b-1]. -/
theorem nice_mod_constraint {n b : ℕ} (h : IsNice n b) :
    n^2 + n^3 ≡ b * (b - 1) / 2 [MOD b - 1] := by
  have hb : b ≥ 2 := h.1
  have hpan := h.2
  have hsum := pandigital_digit_sum hpan
  unfold digitSum at hsum
  rw [List.sum_append] at hsum
  have h2 : digitSumBase b (n^2) ≡ n^2 [MOD b - 1] := digit_sum_mod hb
  have h3 : digitSumBase b (n^3) ≡ n^3 [MOD b - 1] := digit_sum_mod hb
  unfold digitSumBase digitSum at h2 h3
  calc n^2 + n^3 ≡ (digits b (n^2)).sum + (digits b (n^3)).sum [MOD b - 1] :=
         Nat.ModEq.add h2.symm h3.symm
       _ = b * (b - 1) / 2 := hsum

/-- A number can only be nice if it satisfies the residue constraint. -/
theorem nice_residue_necessary {n b : ℕ} (h : IsNice n b) :
    (n^2 + n^3) % (b - 1) = (b * (b - 1) / 2) % (b - 1) := by
  exact nice_mod_constraint h

/-! ## Filter Soundness -/

/-- Given a base b, compute the valid residue classes for n modulo (b-1). -/
def residueFilter (b : ℕ) : Finset ℕ :=
  Finset.filter (fun r => (r^2 + r^3) % (b - 1) = (b * (b - 1) / 2) % (b - 1))
                 (Finset.range (b - 1))

/-- The residue filter is sound: if n is nice, then n % (b-1) is in the filter. -/
theorem residue_filter_sound {n b : ℕ} (hb : b ≥ 2) (h : IsNice n b) :
    n % (b - 1) ∈ residueFilter b := by
  unfold residueFilter
  rw [Finset.mem_filter]
  constructor
  · rw [Finset.mem_range]
    exact Nat.mod_lt n (by omega : b - 1 > 0)
  · have hres := nice_residue_necessary h
    have h1 : n^2 % (b - 1) = (n % (b - 1))^2 % (b - 1) := Nat.pow_mod n 2 (b - 1)
    have h2 : n^3 % (b - 1) = (n % (b - 1))^3 % (b - 1) := Nat.pow_mod n 3 (b - 1)
    calc ((n % (b - 1))^2 + (n % (b - 1))^3) % (b - 1)
        = ((n % (b - 1))^2 % (b - 1) + (n % (b - 1))^3 % (b - 1)) % (b - 1) := by
            rw [Nat.add_mod]
      _ = (n^2 % (b - 1) + n^3 % (b - 1)) % (b - 1) := by rw [← h1, ← h2]
      _ = (n^2 + n^3) % (b - 1) := by rw [← Nat.add_mod]
      _ = (b * (b - 1) / 2) % (b - 1) := hres

/-! ## Nonexistence Results -/

/-- If a base b ≡ 1 (mod 5), then no nice numbers exist in that base.
    This is derived from digit count constraints: for k + ℓ + 2 = b where
    k is the digit count minus 1 for n² and ℓ for n³, the constraints
    from n⁶ require (2b-7)/5 < k < (2b-2)/5, which has no integer solutions
    when b ≡ 1 (mod 5). -/
theorem no_nice_when_one_mod_five {b : ℕ} (hb : b ≡ 1 [MOD 5]) (hb2 : b ≥ 2) :
    ∀ n, ¬IsNice n b := by
  sorry

/-- Helper: characterize when the residue filter is empty. -/
theorem residue_filter_empty_iff {b : ℕ} (hb : b ≥ 2) :
    residueFilter b = ∅ ↔ ∀ r < b - 1, (r^2 + r^3) % (b - 1) ≠ (b * (b - 1) / 2) % (b - 1) := by
  sorry

/-- If the residue filter is empty, no nice numbers exist. -/
theorem no_nice_when_filter_empty {b : ℕ} (hb : b ≥ 2) (hemp : residueFilter b = ∅) :
    ∀ n, ¬IsNice n b := by
  intro n hnice
  have hmem := residue_filter_sound hb hnice
  rw [hemp] at hmem
  exact Finset.notMem_empty _ hmem

/-! ## Known Nice Numbers -/

/-- 69 is nice in base 10. This can be verified by computing:
    69² = 4761, 69³ = 328509, concatenated digits are [1,6,7,4,9,0,5,8,2,3]
    which is a permutation of [0,1,2,3,4,5,6,7,8,9]. -/
theorem nice_69 : IsNice 69 10 := by
  sorry

/-! ## Examples of Filter Application -/

/-- Base 11 has an empty residue filter. -/
theorem residue_filter_11_empty : residueFilter 11 = ∅ := by
  sorry

/-- Therefore, no nice numbers exist in base 11. -/
theorem no_nice_base_11 : ∀ n, ¬IsNice n 11 := by
  exact no_nice_when_filter_empty (by norm_num : (11 : ℕ) ≥ 2) residue_filter_11_empty

end Nice
