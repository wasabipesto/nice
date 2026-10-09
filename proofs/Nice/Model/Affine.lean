/-
The affine middle-digit filter. Rust: `affine_filter::{survives, middle_masks_const}`,
run on cross-end survivors by `stride_filter::walk_two_phase`.

Write `n = s + b^k·t`. Then `n² = s² + b^k·(2st + b^k·t²)`, so
`⌊n²/b^k⌋ = ⌊s²/b^k⌋ + 2st + b^k·t²`, and modulo `b^m` with `m ≤ k` the last
term vanishes: the `m` digits of `n²` from position `k` up are the low digits
of `⌊s²/b^k⌋ + 2st`. The cube is the same with `⌊s³/b^k⌋ + 3s²t` (AFF-1).
The stride walk knows `s` (the residue's suffix) and reads `t mod b^k` off
`n mod b^(2k)`, so these `2k` digits cost a few word operations.

A nice number's digits at distinct output positions are distinct, so the
filter's rejections (a repeat among the middle digits, or one of them among
the residue's low digits or the range certificate) are sound as long as the
middle positions are real digits and the known digits sit elsewhere
(`affineSurvives_of_isNice`). With `m = 1` the same lemma is the overlap
join's bottom-list digit step, and with `m = mid` its prefilter.
-/
import Nice.Model.Msd

namespace NiceSearch

/-- Shifting by `b^k` moves digit `k + i` to position `i`. -/
theorem digit_div_pow (b x k i : ℕ) : digit b (x / b ^ k) i = digit b x (k + i) := by
  unfold digit
  rw [Nat.div_div_eq_div_mul, ← pow_add]

/-- AFF-1 for the square: `⌊n²/b^k⌋ ≡ ⌊s²/b^k⌋ + 2st (mod b^m)` for `n = s + b^k·t`, `m ≤ k`. -/
theorem sq_div_pow_mod {b : ℕ} (hb : 0 < b) (k m s t : ℕ) (hm : m ≤ k) :
    (s + b ^ k * t) ^ 2 / b ^ k % b ^ m = (s ^ 2 / b ^ k + 2 * s * t) % b ^ m := by
  have hB : 0 < b ^ k := Nat.pow_pos hb
  obtain ⟨c, hc⟩ := Nat.pow_dvd_pow b hm
  have hexp : (s + b ^ k * t) ^ 2 = s ^ 2 + b ^ k * (2 * s * t + b ^ m * (c * t ^ 2)) := by
    rw [hc]; ring
  rw [hexp, Nat.add_mul_div_left _ _ hB, ← Nat.add_assoc, Nat.add_mul_mod_self_left]

/-- AFF-1 for the cube: `⌊n³/b^k⌋ ≡ ⌊s³/b^k⌋ + 3s²t (mod b^m)` for `n = s + b^k·t`, `m ≤ k`. -/
theorem cu_div_pow_mod {b : ℕ} (hb : 0 < b) (k m s t : ℕ) (hm : m ≤ k) :
    (s + b ^ k * t) ^ 3 / b ^ k % b ^ m = (s ^ 3 / b ^ k + 3 * s ^ 2 * t) % b ^ m := by
  have hB : 0 < b ^ k := Nat.pow_pos hb
  obtain ⟨c, hc⟩ := Nat.pow_dvd_pow b hm
  have hexp : (s + b ^ k * t) ^ 3 =
      s ^ 3 + b ^ k * (3 * s ^ 2 * t + b ^ m * (c * (3 * s * t ^ 2 + b ^ k * t ^ 3))) := by
    rw [hc]; ring
  rw [hexp, Nat.add_mul_div_left _ _ hB, ← Nat.add_assoc, Nat.add_mul_mod_self_left]

/-- Claim AFF-1: for `n = s + b^k·t` and `i < m ≤ k`, digit `k + i` of `n²` is digit
`i` of `(⌊s²/b^k⌋ + 2st) mod b^m`, and digit `k + i` of `n³` is digit `i` of
`(⌊s³/b^k⌋ + 3s²t) mod b^m`. -/
theorem affine_mid_digit {b k m s t i : ℕ} (hb : 0 < b) (hm : m ≤ k) (hi : i < m) :
    digit b ((s + b ^ k * t) ^ 2) (k + i) = digit b ((s ^ 2 / b ^ k + 2 * s * t) % b ^ m) i ∧
      digit b ((s + b ^ k * t) ^ 3) (k + i) =
        digit b ((s ^ 3 / b ^ k + 3 * s ^ 2 * t) % b ^ m) i := by
  constructor
  · rw [← digit_div_pow, ← digit_mod_pow (n := (s + b ^ k * t) ^ 2 / b ^ k) hi,
      sq_div_pow_mod hb k m s t hm]
  · rw [← digit_div_pow, ← digit_mod_pow (n := (s + b ^ k * t) ^ 3 / b ^ k) hi,
      cu_div_pow_mod hb k m s t hm]

/-! ### The filter -/

/-- The filter's `2k` digits from `nmod = n mod b^(2k)`: `k` of `n²` then `k` of
`n³`, from position `k` up (`middle_masks_const`, with `s` the residue's suffix
and `t = nmod / b^k`). -/
def middleDigits (b k nmod : ℕ) : List ℕ :=
  let s := nmod % b ^ k
  let t := nmod / b ^ k
  lowDigits b k ((s ^ 2 / b ^ k + 2 * s * t) % b ^ k) ++
    lowDigits b k ((s ^ 3 / b ^ k + 3 * s ^ 2 * t) % b ^ k)

/-- `affine_filter::survives`: the middle digits are pairwise distinct and avoid
`known` (the residue's low digits and the range certificate). -/
def affineSurvives (b k nmod : ℕ) (known : Finset ℕ) : Bool :=
  decide (middleDigits b k nmod).Nodup && (middleDigits b k nmod).all fun d => decide (d ∉ known)

/-- The filter reads digits `k .. 2k-1` of both powers. -/
theorem middleDigits_eq {b k n : ℕ} (hb : 0 < b) :
    middleDigits b k (n % b ^ (2 * k)) =
      (List.range k).map (fun i => digit b (n ^ 2) (k + i)) ++
        (List.range k).map (fun i => digit b (n ^ 3) (k + i)) := by
  have h2k : b ^ (2 * k) = b ^ k * b ^ k := by rw [two_mul, pow_add]
  have hs : n % b ^ (2 * k) % b ^ k = n % b ^ k := Nat.mod_mod_of_dvd n ⟨b ^ k, h2k⟩
  have ht : n % b ^ (2 * k) / b ^ k = n / b ^ k % b ^ k := by
    rw [h2k, Nat.mod_mul_right_div_self]
  have hn : n % b ^ k + b ^ k * (n / b ^ k) = n := Nat.mod_add_div n (b ^ k)
  unfold middleDigits lowDigits
  simp only [hs, ht]
  congr 1 <;> apply List.map_congr_left <;> intro i hi <;> rw [List.mem_range] at hi
  · have h := (affine_mid_digit (s := n % b ^ k) (t := n / b ^ k) hb (le_refl k) hi).1
    rw [hn] at h
    rw [h]
    congr 1
    exact Nat.ModEq.add_left _ (Nat.ModEq.mul_left _ (Nat.mod_modEq _ _))
  · have h := (affine_mid_digit (s := n % b ^ k) (t := n / b ^ k) hb (le_refl k) hi).2
    rw [hn] at h
    rw [h]
    congr 1
    exact Nat.ModEq.add_left _ (Nat.ModEq.mul_left _ (Nat.mod_modEq _ _))

/-- A nice number's digits at distinct output positions differ. -/
theorem digit_ne_of_isNice {b n e j e' j' : ℕ} (hb : 2 ≤ b) (h : IsNice b n)
    (he : e = 2 ∨ e = 3) (he' : e' = 2 ∨ e' = 3)
    (hj : j < numDigits b (n ^ e)) (hj' : j' < numDigits b (n ^ e'))
    (hne : ¬ (e = e' ∧ j = j')) : digit b (n ^ e) j ≠ digit b (n ^ e') j' := by
  intro hd
  have hnd : (outputDigits b n).Nodup := h.nodup_iff.mpr List.nodup_range
  have hi1 := outIndex_lt he hj
  have hi2 := outIndex_lt he' hj'
  have h1 := outputDigits_getD hb he hj
  have h2 := outputDigits_getD hb he' hj'
  rw [List.getD_eq_getElem _ _ hi1] at h1
  rw [List.getD_eq_getElem _ _ hi2] at h2
  have hidx : outIndex b n e j = outIndex b n e' j' := by
    have : (outputDigits b n)[outIndex b n e j] = (outputDigits b n)[outIndex b n e' j'] := by
      rw [h1, h2, hd]
    exact (List.Nodup.getElem_inj_iff hnd).mp this
  exact hne (outIndex_inj he he' hj hj' hidx)

/-- Claim AFF-1 (the filter): a nice number survives when its middle positions are
real digits of both powers and every known digit occurs below position `k` or at
`2k` or above. -/
theorem affineSurvives_of_isNice {b k n : ℕ} {known : Finset ℕ} (hb : 2 ≤ b) (h : IsNice b n)
    (h2 : 2 * k ≤ numDigits b (n ^ 2)) (h3 : 2 * k ≤ numDigits b (n ^ 3))
    (hknown : ∀ x ∈ known, ∃ e j, (e = 2 ∨ e = 3) ∧ (j < k ∨ 2 * k ≤ j) ∧
      j < numDigits b (n ^ e) ∧ digit b (n ^ e) j = x) :
    affineSurvives b k (n % b ^ (2 * k)) known = true := by
  unfold affineSurvives
  rw [middleDigits_eq (by omega)]
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
  constructor
  · rw [List.nodup_append]
    refine ⟨List.Nodup.map_on ?_ List.nodup_range, List.Nodup.map_on ?_ List.nodup_range, ?_⟩
    · intro i hi i' hi' heq
      rw [List.mem_range] at hi hi'
      by_contra hne
      exact digit_ne_of_isNice hb h (Or.inl rfl) (Or.inl rfl) (by omega) (by omega) (by omega) heq
    · intro i hi i' hi' heq
      rw [List.mem_range] at hi hi'
      by_contra hne
      exact digit_ne_of_isNice hb h (Or.inr rfl) (Or.inr rfl) (by omega) (by omega) (by omega) heq
    · intro x hx y hy hxy
      rw [List.mem_map] at hx hy
      obtain ⟨i, hi, rfl⟩ := hx
      obtain ⟨i', hi', rfl⟩ := hy
      rw [List.mem_range] at hi hi'
      exact digit_ne_of_isNice hb h (Or.inl rfl) (Or.inr rfl) (by omega) (by omega) (by omega) hxy
  · intro d hd hdk
    obtain ⟨e, j, he, hjpos, hj, hx⟩ := hknown d hdk
    rw [List.mem_append, List.mem_map, List.mem_map] at hd
    rcases hd with ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩ <;> rw [List.mem_range] at hi
    · exact digit_ne_of_isNice hb h (Or.inl rfl) he (by omega) hj (by omega) hx.symm
    · exact digit_ne_of_isNice hb h (Or.inr rfl) he (by omega) hj (by omega) hx.symm

/-- At or above `b^(2k-1)`, both powers have at least `2k` digits (so the
middle positions are real; the walk's gate). -/
theorem two_mul_le_numDigits {b k n e : ℕ} (hb : 2 ≤ b) (hn : b ^ (2 * k - 1) ≤ n)
    (he : e = 2 ∨ e = 3) : 2 * k ≤ numDigits b (n ^ e) := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simp
  have hn1 : 1 ≤ n := le_trans (Nat.one_le_pow _ _ (by omega)) hn
  have hpow : b ^ (2 * k) ≤ n ^ e := by
    calc b ^ (2 * k) ≤ b ^ (2 * (2 * k - 1)) := Nat.pow_le_pow_right (by omega) (by omega)
      _ = (b ^ (2 * k - 1)) ^ 2 := by rw [← pow_mul, Nat.mul_comm (2 * k - 1) 2]
      _ ≤ n ^ 2 := Nat.pow_le_pow_left hn 2
      _ ≤ n ^ e := Nat.pow_le_pow_right hn1 (by omega)
  have := (lt_numDigits_iff hb (pow_ne_zero e (by omega)) (2 * k)).mpr hpow
  omega

end NiceSearch
