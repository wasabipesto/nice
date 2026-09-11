/-
The seeded nice check. Rust: `client_process::get_is_nice_with_known_lsd`.

A stride residue fixes the low `k` digits of both powers; the seeded check
starts its duplicate indicator from those `2k` digits, divides each power
by `b^k` once, and scans the remaining digits. It reports "nice" when no
digit repeats. That equals the plain check because, inside the range,
"no repeats" is the same as "permutation of `0..b-1`" (the digit count is
`b`), and the seeded scan sees a rearrangement of the same digits.
-/
import Nice.Model.Stride

open scoped List

namespace Nice

/-- A list of exactly `b` digits below `b` permutes `0..b-1` iff it has no repeats. -/
theorem perm_range_iff_nodup {b : ℕ} {l : List ℕ} (hlen : l.length = b) (hlt : ∀ d ∈ l, d < b) :
    l.Perm (List.range b) ↔ l.Nodup := by
  constructor
  · intro h
    exact h.nodup_iff.mpr List.nodup_range
  · intro hnd
    apply List.perm_of_nodup_nodup_toFinset_eq hnd List.nodup_range
    rw [List.toFinset_range]
    apply Finset.eq_of_subset_of_card_le
    · intro x hx
      rw [List.mem_toFinset] at hx
      exact Finset.mem_range.mpr (hlt x hx)
    · rw [Finset.card_range, List.toFinset_card_of_nodup hnd, hlen]

/-- Inside the range, niceness is just "no repeated output digit". -/
theorem isNice_iff_nodup {b n : ℕ} (hb : 2 ≤ b) (hr : InBaseRange b n) :
    IsNice b n ↔ (outputDigits b n).Nodup := by
  unfold IsNice
  apply perm_range_iff_nodup
  · unfold InBaseRange numDigits at hr
    simpa [outputDigits] using hr
  · intro d hd
    unfold outputDigits at hd
    rw [List.mem_append] at hd
    rcases hd with hd | hd <;> exact Nat.digits_lt_base hb hd

/-- The digits of `x / b^k` are the digits of `x` from position `k` up. -/
theorem digits_div_pow {b : ℕ} (hb : 2 ≤ b) (x k : ℕ) :
    Nat.digits b (x / b ^ k) = (Nat.digits b x).drop k := by
  rw [Nat.self_div_pow_eq_ofDigits_drop k x hb]
  apply Nat.digits_ofDigits b hb
  · intro d hd
    exact Nat.digits_lt_base hb (List.mem_of_mem_drop hd)
  · intro h
    rw [List.getLast_drop]
    exact Nat.getLast_digit_ne_zero b (by
      intro hx
      apply h
      simp [hx])

/-- The list the seeded scan checks: the residue's `2k` low digits, then
each power's digits above position `k`. -/
def seededDigits (b k n : ℕ) : List ℕ :=
  suffixDigits b k (n % b ^ k) ++ Nat.digits b (n ^ 2 / b ^ k) ++ Nat.digits b (n ^ 3 / b ^ k)

/-- The seeded scan sees a rearrangement of the output digits. -/
theorem seededDigits_perm {b n k : ℕ} (hb : 2 ≤ b)
    (hk2 : k ≤ numDigits b (n ^ 2)) (hk3 : k ≤ numDigits b (n ^ 3)) :
    (seededDigits b k n).Perm (outputDigits b n) := by
  unfold seededDigits outputDigits
  rw [suffixDigits_mod, lowDigits_eq_take hb hk2, lowDigits_eq_take hb hk3,
    digits_div_pow hb, digits_div_pow hb]
  set A := Nat.digits b (n ^ 2)
  set C := Nat.digits b (n ^ 3)
  calc A.take k ++ C.take k ++ A.drop k ++ C.drop k
      = A.take k ++ (C.take k ++ A.drop k) ++ C.drop k := by simp only [List.append_assoc]
    _ ~ A.take k ++ (A.drop k ++ C.take k) ++ C.drop k :=
        (List.perm_append_comm.append_left (A.take k)).append_right (C.drop k)
    _ = (A.take k ++ A.drop k) ++ (C.take k ++ C.drop k) := by simp only [List.append_assoc]
    _ = A ++ C := by rw [List.take_append_drop, List.take_append_drop]

/-- Claim STR-3: inside the range, the seeded check equals the plain one. -/
theorem seeded_iff_isNice {b n k : ℕ} (hb : 2 ≤ b) (hr : InBaseRange b n)
    (hk2 : k ≤ numDigits b (n ^ 2)) (hk3 : k ≤ numDigits b (n ^ 3)) :
    (seededDigits b k n).Nodup ↔ IsNice b n := by
  rw [isNice_iff_nodup hb hr, (seededDigits_perm hb hk2 hk3).nodup_iff]

end Nice
