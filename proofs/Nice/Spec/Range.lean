/-
The search interval per base.

A nice number's powers have `b` digits between them (`nice_digit_count`),
so the search only has to cover `{n : numDigits(n²) + numDigits(n³) = b}`.
The Rust (`base_range::get_base_range_natural`) gives that set in closed
form per `b mod 5` using ceiling roots. This file has the predicate, the
digit-count case analysis every side condition rests on, and the closed
form.
-/
import Nice.Spec.Nice
import Mathlib.Tactic

namespace Nice

/-- The set the search covers: digit counts of the two powers sum to `b`. -/
def InBaseRange (b n : ℕ) : Prop := numDigits b (n ^ 2) + numDigits b (n ^ 3) = b

instance (b n : ℕ) : Decidable (InBaseRange b n) := by unfold InBaseRange; infer_instance

theorem inBaseRange_of_isNice {b n : ℕ} (h : IsNice b n) : InBaseRange b n :=
  nice_digit_count h

/-! ### Digit counts of powers

With `m + 1` the digit count of `n`, `n²` has `2m+1` or `2m+2` digits and
`n³` has `3m+1`, `3m+2` or `3m+3`; comparing sixth powers rules out the
combinations `(2m+2, 3m+1)` and `(2m+1, 3m+3)`. -/

theorem numDigits_pos_of_ne_zero {b n : ℕ} (hb : 2 ≤ b) (hn : n ≠ 0) : 0 < numDigits b n :=
  (lt_numDigits_iff hb hn 0).mpr (by simpa using Nat.one_le_iff_ne_zero.mpr hn)

/-- Claim RNG-5: digit counts of powers are monotone. -/
theorem numDigits_pow_mono {b n m : ℕ} (e : ℕ) (h : n ≤ m) :
    numDigits b (n ^ e) ≤ numDigits b (m ^ e) :=
  Nat.le_length_digits_le b _ _ (Nat.pow_le_pow_left h e)

/-- The four possible digit-count pairs. -/
theorem numDigits_sq_cu {b n : ℕ} (hb : 2 ≤ b) (hn : n ≠ 0) :
    ∃ m, (numDigits b (n ^ 2) = 2 * m + 1 ∧ numDigits b (n ^ 3) = 3 * m + 1) ∨
         (numDigits b (n ^ 2) = 2 * m + 1 ∧ numDigits b (n ^ 3) = 3 * m + 2) ∨
         (numDigits b (n ^ 2) = 2 * m + 2 ∧ numDigits b (n ^ 3) = 3 * m + 2) ∨
         (numDigits b (n ^ 2) = 2 * m + 2 ∧ numDigits b (n ^ 3) = 3 * m + 3) := by
  have hpos := numDigits_pos_of_ne_zero hb hn
  obtain ⟨m, hm⟩ : ∃ m, numDigits b n = m + 1 := ⟨numDigits b n - 1, by omega⟩
  have hlo : b ^ m ≤ n := (lt_numDigits_iff hb hn m).mp (by omega)
  have hhi : n < b ^ (m + 1) := (numDigits_le_iff hb hn (m + 1)).mp (by omega)
  have hn2 : n ^ 2 ≠ 0 := pow_ne_zero _ hn
  have hn3 : n ^ 3 ≠ 0 := pow_ne_zero _ hn
  -- b^(2m) ≤ n² < b^(2m+2), b^(3m) ≤ n³ < b^(3m+3)
  have h2lo : b ^ (2 * m) ≤ n ^ 2 := by
    calc b ^ (2 * m) = (b ^ m) ^ 2 := by rw [← pow_mul, mul_comm]
      _ ≤ n ^ 2 := Nat.pow_le_pow_left hlo 2
  have h2hi : n ^ 2 < b ^ (2 * m + 2) := by
    calc n ^ 2 < (b ^ (m + 1)) ^ 2 := Nat.pow_lt_pow_left hhi (by norm_num)
      _ = b ^ (2 * m + 2) := by rw [← pow_mul]; ring_nf
  have h3lo : b ^ (3 * m) ≤ n ^ 3 := by
    calc b ^ (3 * m) = (b ^ m) ^ 3 := by rw [← pow_mul, mul_comm]
      _ ≤ n ^ 3 := Nat.pow_le_pow_left hlo 3
  have h3hi : n ^ 3 < b ^ (3 * m + 3) := by
    calc n ^ 3 < (b ^ (m + 1)) ^ 3 := Nat.pow_lt_pow_left hhi (by norm_num)
      _ = b ^ (3 * m + 3) := by rw [← pow_mul]; ring_nf
  have d2lo : 2 * m < numDigits b (n ^ 2) := (lt_numDigits_iff hb hn2 _).mpr h2lo
  have d2hi : numDigits b (n ^ 2) ≤ 2 * m + 2 := (numDigits_le_iff hb hn2 _).mpr h2hi
  have d3lo : 3 * m < numDigits b (n ^ 3) := (lt_numDigits_iff hb hn3 _).mpr h3lo
  have d3hi : numDigits b (n ^ 3) ≤ 3 * m + 3 := (numDigits_le_iff hb hn3 _).mpr h3hi
  -- exclusion 1: d2 = 2m+2 and d3 = 3m+1 is impossible
  have ex1 : ¬ (2 * m + 1 < numDigits b (n ^ 2) ∧ numDigits b (n ^ 3) ≤ 3 * m + 1) := by
    rintro ⟨ha, hc⟩
    have a : b ^ (2 * m + 1) ≤ n ^ 2 := (lt_numDigits_iff hb hn2 _).mp ha
    have c : n ^ 3 < b ^ (3 * m + 1) := (numDigits_le_iff hb hn3 _).mp hc
    have a' : b ^ (6 * m + 3) ≤ n ^ 6 := by
      calc b ^ (6 * m + 3) = (b ^ (2 * m + 1)) ^ 3 := by rw [← pow_mul]; ring_nf
        _ ≤ (n ^ 2) ^ 3 := Nat.pow_le_pow_left a 3
        _ = n ^ 6 := by rw [← pow_mul]
    have c' : n ^ 6 < b ^ (6 * m + 2) := by
      calc n ^ 6 = (n ^ 3) ^ 2 := by rw [← pow_mul]
        _ < (b ^ (3 * m + 1)) ^ 2 := Nat.pow_lt_pow_left c (by norm_num)
        _ = b ^ (6 * m + 2) := by rw [← pow_mul]; ring_nf
    have : b ^ (6 * m + 2) ≤ b ^ (6 * m + 3) := Nat.pow_le_pow_right (by omega) (by omega)
    omega
  -- exclusion 2: d2 = 2m+1 and d3 = 3m+3 is impossible
  have ex2 : ¬ (numDigits b (n ^ 2) ≤ 2 * m + 1 ∧ 3 * m + 2 < numDigits b (n ^ 3)) := by
    rintro ⟨ha, hc⟩
    have a : n ^ 2 < b ^ (2 * m + 1) := (numDigits_le_iff hb hn2 _).mp ha
    have c : b ^ (3 * m + 2) ≤ n ^ 3 := (lt_numDigits_iff hb hn3 _).mp hc
    have a' : n ^ 6 < b ^ (6 * m + 3) := by
      calc n ^ 6 = (n ^ 2) ^ 3 := by rw [← pow_mul]
        _ < (b ^ (2 * m + 1)) ^ 3 := Nat.pow_lt_pow_left a (by norm_num)
        _ = b ^ (6 * m + 3) := by rw [← pow_mul]; ring_nf
    have c' : b ^ (6 * m + 4) ≤ n ^ 6 := by
      calc b ^ (6 * m + 4) = (b ^ (3 * m + 2)) ^ 2 := by rw [← pow_mul]; ring_nf
        _ ≤ (n ^ 3) ^ 2 := Nat.pow_le_pow_left c 2
        _ = n ^ 6 := by rw [← pow_mul]
    have : b ^ (6 * m + 3) ≤ b ^ (6 * m + 4) := Nat.pow_le_pow_right (by omega) (by omega)
    omega
  refine ⟨m, ?_⟩
  omega

/-- Claim RNG-4: bases `b ≡ 1 (mod 5)` have an empty search range. -/
theorem not_inBaseRange_of_one_mod_five {b n : ℕ} (hb : 2 ≤ b) (h5 : b % 5 = 1) :
    ¬ InBaseRange b n := by
  intro h
  unfold InBaseRange at h
  have hn : n ≠ 0 := by
    rintro rfl
    simp [numDigits] at h
    omega
  obtain ⟨m, hm⟩ := numDigits_sq_cu hb hn
  omega

/-- Claim RNG-3: inside the range, for `b ≥ 6`, both powers have at least
three digits (so LSD depth `k ≤ 3` is sound). -/
theorem three_le_numDigits_of_inBaseRange {b n : ℕ} (hb : 6 ≤ b) (h : InBaseRange b n) :
    3 ≤ numDigits b (n ^ 2) ∧ 3 ≤ numDigits b (n ^ 3) := by
  unfold InBaseRange at h
  have hn : n ≠ 0 := by
    rintro rfl
    simp [numDigits] at h
    omega
  obtain ⟨m, hm⟩ := numDigits_sq_cu (b := b) (by omega) hn
  omega

/-- The largest cube digit count for a base: `numDigits b (n³) ≤ (3b + 1)/5`
inside the range. Base 64 gives 38 (`MAX_FW_DIGITS`). -/
theorem numDigits_cu_le_of_inBaseRange {b n : ℕ} (hb : 2 ≤ b) (h : InBaseRange b n) :
    5 * numDigits b (n ^ 3) ≤ 3 * b + 1 := by
  unfold InBaseRange at h
  have hn : n ≠ 0 := by
    rintro rfl
    simp [numDigits] at h
    omega
  obtain ⟨m, hm⟩ := numDigits_sq_cu hb hn
  omega

/-! ### Closed form -/

/-- Least `r` with `x ≤ r ^ e` (malachite's `ceiling_root`); `0` when `e = 0`. -/
noncomputable def ceilRoot (e x : ℕ) : ℕ :=
  if he : e = 0 then 0 else Nat.find (⟨x, Nat.le_self_pow he x⟩ : ∃ r, x ≤ r ^ e)

theorem le_pow_ceilRoot {e : ℕ} (he : e ≠ 0) (x : ℕ) : x ≤ (ceilRoot e x) ^ e := by
  unfold ceilRoot
  rw [dif_neg he]
  exact Nat.find_spec (⟨x, Nat.le_self_pow he x⟩ : ∃ r, x ≤ r ^ e)

theorem ceilRoot_le {e x r : ℕ} (he : e ≠ 0) (h : x ≤ r ^ e) : ceilRoot e x ≤ r := by
  unfold ceilRoot
  rw [dif_neg he]
  exact Nat.find_min' _ h

theorem pow_lt_of_lt_ceilRoot {e x r : ℕ} (he : e ≠ 0) (h : r < ceilRoot e x) : r ^ e < x := by
  by_contra hc
  exact absurd (ceilRoot_le he (Nat.le_of_not_lt hc)) (Nat.not_le_of_lt h)

theorem le_ceilRoot_iff {e x r : ℕ} (he : e ≠ 0) : ceilRoot e x ≤ r ↔ x ≤ r ^ e :=
  ⟨fun h => (le_pow_ceilRoot he x).trans (Nat.pow_le_pow_left h e), ceilRoot_le he⟩

theorem lt_ceilRoot_iff {e x r : ℕ} (he : e ≠ 0) : r < ceilRoot e x ↔ r ^ e < x := by
  rw [← Nat.not_le, le_ceilRoot_iff he, Nat.not_le]

/-- The Rust closed form (`get_base_range_natural`), half-open; `none` for
`b ≡ 1 (mod 5)`. -/
noncomputable def baseRange (b : ℕ) : Option (ℕ × ℕ) :=
  match b % 5 with
  | 0 => some (ceilRoot 3 (b ^ (3 * (b / 5) - 1)), b ^ (b / 5))
  | 1 => none
  | 2 => some (b ^ (b / 5), ceilRoot 3 (b ^ (3 * (b / 5) + 1)))
  | 3 => some (ceilRoot 3 (b ^ (3 * (b / 5) + 1)), ceilRoot 2 (b ^ (2 * (b / 5) + 1)))
  | _ => some (ceilRoot 2 (b ^ (2 * (b / 5) + 1)), ceilRoot 3 (b ^ (3 * (b / 5) + 2)))

/-- Membership in the closed-form interval. -/
def MemBaseRange (b n : ℕ) : Prop :=
  ∃ lo hi, baseRange b = some (lo, hi) ∧ lo ≤ n ∧ n < hi

/-- Digit count as a two-sided power bound, for `n ≠ 0`. -/
theorem numDigits_eq_iff {b : ℕ} (hb : 2 ≤ b) {n : ℕ} (hn : n ≠ 0) (j : ℕ) :
    numDigits b n = j + 1 ↔ b ^ j ≤ n ∧ n < b ^ (j + 1) := by
  rw [← lt_numDigits_iff hb hn, ← numDigits_le_iff hb hn]
  omega

/-- Claim RNG-2 (one direction): the closed-form interval covers the range. -/
theorem memBaseRange_of_inBaseRange {b n : ℕ} (hb : 2 ≤ b) (h : InBaseRange b n) :
    MemBaseRange b n := by
  have h' := h
  unfold InBaseRange at h'
  have hn : n ≠ 0 := by
    rintro rfl
    simp [numDigits] at h'
    omega
  have hn2 : n ^ 2 ≠ 0 := pow_ne_zero _ hn
  have hn3 : n ^ 3 ≠ 0 := pow_ne_zero _ hn
  obtain ⟨m, hm⟩ := numDigits_sq_cu hb hn
  unfold MemBaseRange baseRange
  rcases hm with ⟨h2, h3⟩ | ⟨h2, h3⟩ | ⟨h2, h3⟩ | ⟨h2, h3⟩
  · -- (2m+1, 3m+1): b = 5m+2, class 2, k = m
    have hb5 : b % 5 = 2 := by omega
    have hk : b / 5 = m := by omega
    obtain ⟨hlo2, hhi2⟩ := (numDigits_eq_iff hb hn2 _).mp h2
    obtain ⟨hlo3, hhi3⟩ := (numDigits_eq_iff hb hn3 _).mp h3
    refine ⟨_, _, by rw [hb5, hk]; rfl, ?_, ?_⟩
    · -- b^m ≤ n from b^(2m) ≤ n²
      rw [← Nat.pow_le_pow_iff_left (n := 2) (by norm_num), ← pow_mul]
      simpa [mul_comm] using hlo2
    · rw [lt_ceilRoot_iff (by norm_num)]
      simpa using hhi3
  · -- (2m+1, 3m+2): b = 5m+3, class 3, k = m
    have hb5 : b % 5 = 3 := by omega
    have hk : b / 5 = m := by omega
    obtain ⟨hlo2, hhi2⟩ := (numDigits_eq_iff hb hn2 _).mp h2
    obtain ⟨hlo3, hhi3⟩ := (numDigits_eq_iff hb hn3 _).mp h3
    refine ⟨_, _, by rw [hb5, hk]; rfl, ?_, ?_⟩
    · rw [le_ceilRoot_iff (by norm_num)]; exact hlo3
    · rw [lt_ceilRoot_iff (by norm_num)]; exact hhi2
  · -- (2m+2, 3m+2): b = 5m+4, class 4, k = m
    have hb5 : b % 5 = 4 := by omega
    have hk : b / 5 = m := by omega
    obtain ⟨hlo2, hhi2⟩ := (numDigits_eq_iff hb hn2 _).mp h2
    obtain ⟨hlo3, hhi3⟩ := (numDigits_eq_iff hb hn3 _).mp h3
    refine ⟨_, _, by rw [hb5, hk]; rfl, ?_, ?_⟩
    · rw [le_ceilRoot_iff (by norm_num)]; exact hlo2
    · rw [lt_ceilRoot_iff (by norm_num)]; exact hhi3
  · -- (2m+2, 3m+3): b = 5m+5, class 0, k = m+1
    have hb5 : b % 5 = 0 := by omega
    have hk : b / 5 = m + 1 := by omega
    obtain ⟨hlo2, hhi2⟩ := (numDigits_eq_iff hb hn2 _).mp h2
    obtain ⟨hlo3, hhi3⟩ := (numDigits_eq_iff hb hn3 _).mp h3
    refine ⟨_, _, by rw [hb5, hk]; rfl, ?_, ?_⟩
    · rw [le_ceilRoot_iff (by norm_num)]
      have : 3 * (m + 1) - 1 = 3 * m + 2 := by omega
      rw [this]; exact hlo3
    · rw [← Nat.pow_lt_pow_iff_left (n := 2) (by norm_num), ← pow_mul]
      have : (m + 1) * 2 = 2 * m + 1 + 1 := by ring
      rw [this]; exact hhi2

end Nice
