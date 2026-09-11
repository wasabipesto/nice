/-
The multi-digit LSD filter (mod b^k). Rust: `lsd_filter::get_valid_multi_lsd_bitmap`.

The low `k` digits of `n²` and `n³` depend only on `n mod b^k` (LSD-1).
For a suffix `s`, the Rust extracts exactly `k` digits of `s² mod b^k` and
`k` digits of `s³ mod b^k` (zero-padded) and keeps `s` only if all `2k`
are pairwise distinct. That is sound whenever both powers really have at
least `k` digits, which `three_le_numDigits_of_inBaseRange` gives for
`k ≤ 3` inside a legal range for `b ≥ 6`.
-/
import Nice.Spec.Range
import Mathlib.Data.List.Sublists

namespace Nice

/-! ### LSD-1: low digits are determined by the suffix -/

/-- A digit is a quotient of the remainder one position up. -/
theorem digit_eq_mod_pow_succ_div (b n j : ℕ) : digit b n j = n % b ^ (j + 1) / b ^ j := by
  unfold digit
  rcases Nat.eq_zero_or_pos b with rfl | hb
  · simp
  have hpos : 0 < b ^ j := Nat.pow_pos hb
  rw [Nat.mod_pow_succ, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt (Nat.mod_lt _ hpos)]
  simp

/-- Digits below position `k` agree on numbers congruent mod `b^k`. -/
theorem digit_eq_of_modEq {b n m k j : ℕ} (h : n ≡ m [MOD b ^ k]) (hj : j < k) :
    digit b n j = digit b m j := by
  rw [digit_eq_mod_pow_succ_div, digit_eq_mod_pow_succ_div]
  have hd : b ^ (j + 1) ∣ b ^ k := Nat.pow_dvd_pow b hj
  rw [← Nat.mod_mod_of_dvd n hd, ← Nat.mod_mod_of_dvd m hd, h]

/-- Claim LSD-1: a low digit of `n^e` is the same digit of `(n mod b^k)^e`. -/
theorem digit_pow_mod_pow {b n k j : ℕ} (e : ℕ) (hj : j < k) :
    digit b (n ^ e) j = digit b ((n % b ^ k) ^ e) j :=
  digit_eq_of_modEq ((Nat.mod_modEq n (b ^ k)).symm.pow e) hj

/-- Reducing mod `b^k` first does not change a digit below position `k`. -/
theorem digit_mod_pow {b n k j : ℕ} (hj : j < k) : digit b (n % b ^ k) j = digit b n j :=
  digit_eq_of_modEq (Nat.mod_modEq n (b ^ k)) hj

/-! ### The bitmap -/

/-- The `k` low digits of `x`, zero-padded, least significant first: exactly
the Rust's fixed-width extraction loop. -/
def lowDigits (b k x : ℕ) : List ℕ := (List.range k).map (digit b x)

/-- The `2k` digits the Rust checks for suffix `s`. -/
def suffixDigits (b k s : ℕ) : List ℕ :=
  lowDigits b k (s ^ 2 % b ^ k) ++ lowDigits b k (s ^ 3 % b ^ k)

/-- The Rust bitmap as a set: suffixes whose `2k` fixed digits are pairwise distinct. -/
def lsdBitmap (b k : ℕ) : Finset ℕ :=
  (Finset.range (b ^ k)).filter fun s => (suffixDigits b k s).Nodup

/-- The low digits of a number are the first `k` entries of its digit list
when it has at least `k` digits. -/
theorem lowDigits_eq_take {b : ℕ} (hb : 2 ≤ b) {k x : ℕ} (hk : k ≤ numDigits b x) :
    lowDigits b k x = (Nat.digits b x).take k := by
  unfold lowDigits numDigits at *
  apply List.ext_getElem
  · simp [List.length_take, hk]
  · intro i hi hi'
    simp only [List.getElem_map, List.getElem_range, List.getElem_take]
    rw [digit_eq_getD hb, List.getD_eq_getElem]

/-- The suffix digits of `n mod b^k` are the low digits of `n²` and `n³`. -/
theorem suffixDigits_mod {b n k : ℕ} :
    suffixDigits b k (n % b ^ k) = lowDigits b k (n ^ 2) ++ lowDigits b k (n ^ 3) := by
  unfold suffixDigits lowDigits
  congr 1 <;> apply List.map_congr_left <;> intro j hj <;> rw [List.mem_range] at hj
  · rw [digit_mod_pow hj, ← digit_pow_mod_pow 2 hj]
  · rw [digit_mod_pow hj, ← digit_pow_mod_pow 3 hj]

/-- Claim LSD-2: the bitmap keeps every nice number's suffix, provided both
powers have at least `k` digits. -/
theorem mem_lsdBitmap_of_isNice {b n k : ℕ} (hb : 2 ≤ b)
    (hk2 : k ≤ numDigits b (n ^ 2)) (hk3 : k ≤ numDigits b (n ^ 3)) (h : IsNice b n) :
    n % b ^ k ∈ lsdBitmap b k := by
  unfold lsdBitmap
  rw [Finset.mem_filter, Finset.mem_range]
  refine ⟨Nat.mod_lt _ (Nat.pow_pos (by omega)), ?_⟩
  rw [suffixDigits_mod, lowDigits_eq_take hb hk2, lowDigits_eq_take hb hk3]
  have hnd : (outputDigits b n).Nodup := h.nodup_iff.mpr List.nodup_range
  exact hnd.sublist (List.Sublist.append (List.take_sublist _ _) (List.take_sublist _ _))

/-- Claim LSD-2 in the form the search uses: inside a legal range for `b ≥ 6`,
`k ≤ 3` needs no further hypothesis. -/
theorem mem_lsdBitmap_of_isNice' {b n k : ℕ} (hb : 6 ≤ b) (hk : k ≤ 3) (h : IsNice b n) :
    n % b ^ k ∈ lsdBitmap b k := by
  have ⟨h2, h3⟩ := three_le_numDigits_of_inBaseRange hb (inBaseRange_of_isNice h)
  exact mem_lsdBitmap_of_isNice (by omega) (by omega) (by omega) h

/-- The base-10 single-digit table the Rust documents: `{2, 3, 4, 7, 8, 9}`. -/
theorem lsdBitmap_ten_one : lsdBitmap 10 1 = {2, 3, 4, 7, 8, 9} := by decide

/-- Suffix 12 is rejected at `k = 2` in base 10 (the two 4s inside `144`),
the example from the 2026-08 fix. -/
example : 12 ∉ lsdBitmap 10 2 := by decide

/-- 69's suffix survives at every depth. -/
example : 69 % 10 ^ 2 ∈ lsdBitmap 10 2 := by decide

end Nice
