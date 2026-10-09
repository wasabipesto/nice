/-
THY-4: the `b² − 1` sieve adds nothing to casting out `b − 1`s, for words of
three or more digits.

Modulo `b² − 1`, `b² ≡ 1`, so a digit word is worth the total `E` of its
digits at even positions plus `b` times the total `O` at odd positions
(`ofDigits_modEq_altSums`). For a nice number the output digits are
`0 .. b−1` once each, so `n² + n³ ≡ E + b·(T − E)` with `T = b(b−1)/2` and `E`
the total at even positions (counted within each power). A filter on
`(n² + n³) mod (b² − 1)` can therefore only learn which totals `E` are
possible. With both words at least three digits long, any set of `c₀` digits
(`c₀` the number of even positions) can take the even positions with nonzero
leading digits; the sums of `c₀`-subsets of `0 .. b−1` fill an interval of
more than `b` values (`exists_subset_sum`, Janzert's `pick_sum`); and
`E ↦ E + b·(T − E)` steps through the whole class of `T` mod `b − 1` in `b + 1`
steps. So every residue that casting out `b − 1`s allows is attained
(`sieve_b2_complete`).

The length hypothesis is load-bearing. In base 4 with lengths `(2, 2)` the
digit 0 cannot lead either word, so it sits at an even position, and only
three of the five residues mod `b + 1 = 5` occur (Janzert's
`base_four_sieve_is_incomplete`). Every nice band with `b ≥ 6` meets it
(RNG-3: both powers have at least three digits).
-/
import Nice.Spec.Range
import Mathlib.Tactic

namespace NiceSearch

/-! ### Words modulo `b² − 1` -/

/-- The totals of a list's entries at even and at odd positions. -/
def altSums : List ℕ → ℕ × ℕ
  | [] => (0, 0)
  | d :: ds => (d + (altSums ds).2, (altSums ds).1)

/-- Modulo `b² − 1` a digit word (least significant first) is worth its even
total plus `b` times its odd total. -/
theorem ofDigits_modEq_altSums {b : ℕ} (hb : 1 ≤ b) (l : List ℕ) :
    Nat.ofDigits b l ≡ (altSums l).1 + b * (altSums l).2 [MOD b ^ 2 - 1] := by
  have hb2 : b ^ 2 ≡ 1 [MOD b ^ 2 - 1] :=
    ((Nat.modEq_iff_dvd' (Nat.one_le_pow _ _ hb)).mpr (dvd_refl _)).symm
  induction l with
  | nil => simp [altSums, Nat.ofDigits, Nat.ModEq]
  | cons d ds ih =>
    rw [Nat.ofDigits_cons]
    simp only [altSums]
    calc d + b * Nat.ofDigits b ds
        ≡ d + b * ((altSums ds).1 + b * (altSums ds).2) [MOD b ^ 2 - 1] :=
          (ih.mul_left b).add_left d
      _ = d + b * (altSums ds).1 + b ^ 2 * (altSums ds).2 := by ring
      _ ≡ d + b * (altSums ds).1 + 1 * (altSums ds).2 [MOD b ^ 2 - 1] :=
          (hb2.mul_right _).add_left _
      _ = d + (altSums ds).2 + b * (altSums ds).1 := by ring

/-! ### Weaving two lists -/

/-- `[e₀, o₀, e₁, o₁, …]`: the first list at the even positions. -/
def weave : List ℕ → List ℕ → List ℕ
  | [], os => os
  | e :: es, os => e :: weave os es
termination_by es os => es.length + os.length

theorem weave_length (E O : List ℕ) : (weave E O).length = E.length + O.length := by
  induction E, O using weave.induct with
  | case1 os => simp [weave]
  | case2 e es os ih =>
    rw [weave, List.length_cons, ih, List.length_cons]
    omega

theorem weave_perm (E O : List ℕ) : (weave E O).Perm (E ++ O) := by
  induction E, O using weave.induct with
  | case1 os => simp [weave]
  | case2 e es os ih =>
    rw [weave, List.cons_append]
    exact (ih.trans List.perm_append_comm).cons e

theorem altSums_weave (E O : List ℕ) (h1 : O.length ≤ E.length) (h2 : E.length ≤ O.length + 1) :
    altSums (weave E O) = (E.sum, O.sum) := by
  induction E, O using weave.induct with
  | case1 os =>
    have : os = [] := List.eq_nil_of_length_eq_zero (by simpa using h1)
    subst this
    simp [weave, altSums]
  | case2 e es os ih =>
    simp only [List.length_cons] at h1 h2
    rw [weave, altSums, ih (by omega) (by omega)]
    simp [List.sum_cons]

theorem weave_getLast? (E O : List ℕ) (h1 : O.length ≤ E.length) (h2 : E.length ≤ O.length + 1)
    (hE : E ≠ []) :
    (weave E O).getLast? = if O.length < E.length then E.getLast? else O.getLast? := by
  induction E, O using weave.induct with
  | case1 os => exact absurd rfl hE
  | case2 e es os ih =>
    simp only [List.length_cons] at h1 h2
    rcases es with _ | ⟨e', es'⟩
    · rcases os with _ | ⟨o, _ | ⟨o', os''⟩⟩
      · simp [weave]
      · simp [weave]
      · simp at h1
    · have hos : os ≠ [] := by
        rintro rfl
        simp at h2
      have hw : weave os (e' :: es') ≠ [] := by
        intro h0
        have := weave_length os (e' :: es')
        rw [h0] at this
        simp at this
      rw [weave]
      obtain ⟨x, xs, hx⟩ := List.exists_cons_of_ne_nil hw
      rw [hx, List.getLast?_cons_cons, ← hx, ih (by simpa using h2) (by simpa using h1) hos]
      simp only [List.length_cons, List.getLast?_cons_cons]
      split_ifs <;> first | rfl | omega

/-! ### Subset sums fill an interval -/

/-- `0 + 1 + ⋯ + (c − 1)`. -/
def tri (c : ℕ) : ℕ := ∑ i ∈ Finset.range c, i

/-- The sums of the `c`-element subsets of `{0, …, b − 1}` are every value from
`tri c` to `tri c + c·(b − c)` (Janzert's `pick_sum`). -/
theorem exists_subset_sum : ∀ (b c v : ℕ), c ≤ b → v ≤ c * (b - c) →
    ∃ S ⊆ Finset.range b, S.card = c ∧ ∑ i ∈ S, i = tri c + v
  | 0, c, v, hc, hv => by
    obtain rfl : c = 0 := by omega
    exact ⟨∅, Finset.empty_subset _, rfl, by simp [tri]; omega⟩
  | b + 1, c, v, hc, hv => by
    rcases Nat.eq_zero_or_pos c with rfl | hc0
    · exact ⟨∅, Finset.empty_subset _, rfl, by simp [tri]; omega⟩
    by_cases hfit : c ≤ b ∧ v ≤ c * (b - c)
    · obtain ⟨S, hS, hcard, hsum⟩ := exists_subset_sum b c v hfit.1 hfit.2
      exact ⟨S, hS.trans (Finset.range_subset_range.mpr (Nat.le_succ b)), hcard, hsum⟩
    · -- take `b`, and a `(c − 1)`-subset of `{0, …, b − 1}`
      have hlow : b + 1 - c ≤ v := by
        rcases Nat.lt_or_ge b c with hbc | hbc
        · omega
        · have : b - c ≤ c * (b - c) := Nat.le_mul_of_pos_left _ hc0
          omega
      have he : (c - 1) * (b - (c - 1)) = c * (b + 1 - c) - (b + 1 - c) := by
        rw [show b - (c - 1) = b + 1 - c by omega, Nat.sub_one_mul]
      have hsub : c * (b + 1 - c) ≥ b + 1 - c := Nat.le_mul_of_pos_left _ hc0
      obtain ⟨S, hS, hcard, hsum⟩ :=
        exists_subset_sum b (c - 1) (v - (b + 1 - c)) (by omega) (by rw [he]; omega)
      have hbS : b ∉ S := fun h => by
        have := hS h
        simp at this
      refine ⟨insert b S, ?_, ?_, ?_⟩
      · intro x hx
        rw [Finset.mem_insert] at hx
        rcases hx with rfl | hx
        · exact Finset.mem_range.mpr (Nat.lt_succ_self _)
        · exact Finset.range_subset_range.mpr (Nat.le_succ b) (hS hx)
      · rw [Finset.card_insert_of_notMem hbS, hcard]
        omega
      · rw [Finset.sum_insert hbS, hsum]
        have htri : tri c = tri (c - 1) + (c - 1) := by
          obtain ⟨c', rfl⟩ : ∃ c', c = c' + 1 := ⟨c - 1, by omega⟩
          simp [tri, Finset.sum_range_succ]
        omega

/-! ### Sorted digit lists -/

/-- In a strictly increasing list of naturals, only the first entry can be 0. -/
theorem sorted_getElem_ne_zero {l : List ℕ} (hl : l.Pairwise (· < ·)) {i : ℕ} (hi : 0 < i)
    (h : i < l.length) : l[i] ≠ 0 := by
  have := List.pairwise_iff_getElem.mp hl 0 i (by omega) h hi
  omega

theorem zero_notMem_drop {l : List ℕ} (hl : l.Pairwise (· < ·)) {n : ℕ} (hn : 0 < n) :
    0 ∉ l.drop n := by
  intro h
  obtain ⟨i, hi, hx⟩ := List.getElem_of_mem h
  rw [List.getElem_drop] at hx
  rw [List.length_drop] at hi
  exact sorted_getElem_ne_zero hl (by omega) (by omega) hx

theorem getLast?_take_ne_zero {l : List ℕ} (hl : l.Pairwise (· < ·)) {n : ℕ} (hn : 2 ≤ n)
    (hlen : n ≤ l.length) : (l.take n).getLast? ≠ some 0 := by
  rw [List.getLast?_take, if_neg (by omega), List.getElem?_eq_getElem (by omega)]
  intro h
  exact sorted_getElem_ne_zero hl (by omega) (by omega) (Option.some.inj h)

/-! ### THY-4 -/

/-- Claim THY-4: with both words at least three digits long, every residue
mod `b² − 1` that the digit-sum congruence allows is the residue of `x + y`
for digit words `x`, `y` (least significant first) of lengths `L₂`, `L₃`,
with nonzero leading digits, that together use every digit `0 .. b−1` once.
So the `b² − 1` sieve on `n² + n³` rejects nothing that casting out `b − 1`s
keeps. -/
theorem sieve_b2_complete {b L₂ L₃ z : ℕ} (hL₂ : 3 ≤ L₂) (hL₃ : 3 ≤ L₃) (hb : L₂ + L₃ = b)
    (hz : z ≡ b * (b - 1) / 2 [MOD b - 1]) :
    ∃ d₂ d₃ : List ℕ, d₂.length = L₂ ∧ d₃.length = L₃ ∧ (d₂ ++ d₃).Perm (List.range b) ∧
      d₂.getLast? ≠ some 0 ∧ d₃.getLast? ≠ some 0 ∧
      Nat.ofDigits b d₂ + Nat.ofDigits b d₃ ≡ z [MOD b ^ 2 - 1] := by
  -- even and odd positions: `c₀` and `c₁` of them
  obtain ⟨c₀, hc₀⟩ : ∃ c₀, c₀ = (L₂ + 1) / 2 + (L₃ + 1) / 2 := ⟨_, rfl⟩
  obtain ⟨c₁, hc₁⟩ : ∃ c₁, c₁ = L₂ / 2 + L₃ / 2 := ⟨_, rfl⟩
  have hcb : c₀ + c₁ = b := by omega
  have hc₀2 : 2 ≤ c₀ := by omega
  have hc₁2 : 2 ≤ c₁ := by omega
  have hcc : b ≤ c₀ * c₁ := by nlinarith
  obtain ⟨T, hT⟩ : ∃ T, T = b * (b - 1) / 2 := ⟨_, rfl⟩
  have hTtri : tri b = T := by rw [hT, tri, Finset.sum_range_id]
  -- the class to hit: `b·T − (b − 1)·tri c₀ − z` is a multiple of `b − 1`
  have hdvd : ((b : ℤ) - 1) ∣ (b : ℤ) * T - ((b : ℤ) - 1) * (tri c₀ : ℤ) - z := by
    have h1 : ((b : ℤ) - 1) ∣ (T : ℤ) - z := by
      have := Nat.modEq_iff_dvd.mp hz
      rw [← hT] at this
      push_cast [Nat.cast_sub (show 1 ≤ b by omega)] at this
      exact this
    have : (b : ℤ) * T - ((b : ℤ) - 1) * (tri c₀ : ℤ) - z =
        ((b : ℤ) - 1) * (T - tri c₀) + ((T : ℤ) - z) := by ring
    rw [this]
    exact dvd_add (dvd_mul_right _ _) h1
  obtain ⟨u, hu⟩ := hdvd
  -- `j ≡ u (mod b + 1)`
  have hq : (0 : ℤ) < (b : ℤ) + 1 := by omega
  have hj0 : 0 ≤ u % ((b : ℤ) + 1) := Int.emod_nonneg _ (by omega)
  have hjq : u % ((b : ℤ) + 1) < (b : ℤ) + 1 := Int.emod_lt_of_pos _ hq
  obtain ⟨j, hj⟩ : ∃ j : ℕ, (j : ℤ) = u % ((b : ℤ) + 1) := ⟨_, Int.toNat_of_nonneg hj0⟩
  have hjb : j ≤ b := by omega
  -- the even digits: a `c₀`-subset with total `tri c₀ + j`
  obtain ⟨S, hS, hScard, hSsum⟩ :=
    exists_subset_sum b c₀ j (by omega) (by rw [show b - c₀ = c₁ by omega]; omega)
  set C := Finset.range b \ S with hC
  have hCcard : C.card = c₁ := by
    rw [hC, Finset.card_sdiff_of_subset hS, Finset.card_range, hScard]
    omega
  have hsumSC : ∑ i ∈ S, i + ∑ i ∈ C, i = T := by
    rw [hC, add_comm, Finset.sum_sdiff hS, ← hTtri, tri]
  -- the words
  set Sl := S.sort (· ≤ ·) with hSl
  set Cl := C.sort (· ≤ ·) with hCl
  have hSl_len : Sl.length = c₀ := by rw [hSl, Finset.length_sort, hScard]
  have hCl_len : Cl.length = c₁ := by rw [hCl, Finset.length_sort, hCcard]
  have hSl_sorted : Sl.Pairwise (· < ·) := List.sortedLT_iff_pairwise.mp (Finset.sortedLT_sort S)
  have hCl_sorted : Cl.Pairwise (· < ·) := List.sortedLT_iff_pairwise.mp (Finset.sortedLT_sort C)
  have hSl_sum : Sl.sum = ∑ i ∈ S, i := by
    rw [hSl]
    conv_rhs => rw [← Finset.sort_toFinset S (· ≤ ·)]
    rw [List.sum_toFinset _ (Finset.sort_nodup _ _), List.map_id']
  have hCl_sum : Cl.sum = ∑ i ∈ C, i := by
    rw [hCl]
    conv_rhs => rw [← Finset.sort_toFinset C (· ≤ ·)]
    rw [List.sum_toFinset _ (Finset.sort_nodup _ _), List.map_id']
  obtain ⟨a, ha⟩ : ∃ a, a = (L₂ + 1) / 2 := ⟨_, rfl⟩
  obtain ⟨c, hc⟩ : ∃ c, c = L₂ / 2 := ⟨_, rfl⟩
  refine ⟨weave (Sl.take a) (Cl.take c), weave (Sl.drop a) (Cl.drop c), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [weave_length, List.length_take, List.length_take]
    omega
  · rw [weave_length, List.length_drop, List.length_drop]
    omega
  · -- the two words use `Sl ++ Cl`, which is `0 .. b−1` once each
    have hp1 : (weave (Sl.take a) (Cl.take c) ++ weave (Sl.drop a) (Cl.drop c)).Perm
        ((Sl.take a ++ Cl.take c) ++ (Sl.drop a ++ Cl.drop c)) :=
      (weave_perm _ _).append (weave_perm _ _)
    have hp2 : ((Sl.take a ++ Cl.take c) ++ (Sl.drop a ++ Cl.drop c)).Perm (Sl ++ Cl) := by
      rw [← Multiset.coe_eq_coe]
      simp only [← Multiset.coe_add]
      conv_rhs => rw [← List.take_append_drop a Sl, ← List.take_append_drop c Cl]
      simp only [← Multiset.coe_add]
      abel
    refine (hp1.trans hp2).trans ?_
    apply List.perm_of_nodup_nodup_toFinset_eq
    · rw [List.nodup_append]
      refine ⟨Finset.sort_nodup _ _, Finset.sort_nodup _ _, ?_⟩
      intro x hx y hy hxy
      subst hxy
      rw [hSl, Finset.mem_sort] at hx
      rw [hCl, Finset.mem_sort, hC, Finset.mem_sdiff] at hy
      exact hy.2 hx
    · exact List.nodup_range
    · ext x
      rw [List.toFinset_append, Finset.mem_union, List.mem_toFinset, List.mem_toFinset,
        List.mem_toFinset, List.mem_range, hSl, hCl, Finset.mem_sort, Finset.mem_sort, hC,
        Finset.mem_sdiff, Finset.mem_range]
      constructor
      · rintro (hx | ⟨hx, -⟩)
        · exact Finset.mem_range.mp (hS hx)
        · exact hx
      · intro hx
        by_cases hxS : x ∈ S
        · exact Or.inl hxS
        · exact Or.inr ⟨hx, hxS⟩
  · -- the first word leads with an entry past the front of `Sl` or of `Cl`
    rw [weave_getLast? (Sl.take a) (Cl.take c)
      (by rw [List.length_take, List.length_take]; omega)
      (by rw [List.length_take, List.length_take]; omega)
      (by rw [← List.length_pos_iff_ne_nil, List.length_take]; omega)]
    split_ifs with hlt
    · exact getLast?_take_ne_zero hSl_sorted (by omega) (by omega)
    · rw [List.length_take, List.length_take] at hlt
      exact getLast?_take_ne_zero hCl_sorted (by omega) (by omega)
  · -- the second word has no 0 at all: it is past the front of both lists
    intro h
    have hmem := (weave_perm _ _).subset (List.mem_of_getLast? h)
    rw [List.mem_append] at hmem
    rcases hmem with hmem | hmem
    · exact zero_notMem_drop hSl_sorted (by omega) hmem
    · exact zero_notMem_drop hCl_sorted (by omega) hmem
  · -- the value: even total `tri c₀ + j`, odd total `T − tri c₀ − j`
    have hw2 := altSums_weave (Sl.take a) (Cl.take c)
      (by rw [List.length_take, List.length_take]; omega)
      (by rw [List.length_take, List.length_take]; omega)
    have hw3 := altSums_weave (Sl.drop a) (Cl.drop c)
      (by rw [List.length_drop, List.length_drop]; omega)
      (by rw [List.length_drop, List.length_drop]; omega)
    have hv := (ofDigits_modEq_altSums (show 1 ≤ b by omega) (weave (Sl.take a) (Cl.take c))).add
      (ofDigits_modEq_altSums (show 1 ≤ b by omega) (weave (Sl.drop a) (Cl.drop c)))
    rw [hw2, hw3] at hv
    simp only at hv
    have hE : (Sl.take a).sum + (Sl.drop a).sum = tri c₀ + j := by
      rw [← List.sum_append, List.take_append_drop, hSl_sum, hSsum]
    have hO : (Cl.take c).sum + (Cl.drop c).sum = T - (tri c₀ + j) := by
      rw [← List.sum_append, List.take_append_drop, hCl_sum]
      omega
    refine hv.trans ?_
    -- `E + b·O ≡ z`, in the integers
    apply Nat.modEq_iff_dvd.mpr
    have hET : tri c₀ + j ≤ T := by omega
    have hcast : ((((Sl.take a).sum + b * (Cl.take c).sum + ((Sl.drop a).sum +
        b * (Cl.drop c).sum)) : ℕ) : ℤ) = (tri c₀ + j : ℤ) + b * ((T : ℤ) - (tri c₀ + j)) := by
      have : (Sl.take a).sum + b * (Cl.take c).sum + ((Sl.drop a).sum + b * (Cl.drop c).sum) =
          (tri c₀ + j) + b * (T - (tri c₀ + j)) := by
        rw [← hO, ← hE]
        ring
      rw [this]
      push_cast [Nat.cast_sub hET]
      ring
    rw [hcast]
    push_cast [Nat.cast_sub (show 1 ≤ b ^ 2 from Nat.one_le_pow _ _ (by omega))]
    -- `z − (E + b·(T − E)) = −(b − 1)·(u − j)` and `b + 1 ∣ u − j`
    have hqd : ((b : ℤ) + 1) ∣ u - j := by
      rw [hj, Int.emod_def]
      simp
    obtain ⟨w, hw⟩ := hqd
    refine ⟨-w, ?_⟩
    have hu' : (z : ℤ) = (b : ℤ) * T - ((b : ℤ) - 1) * tri c₀ - ((b : ℤ) - 1) * u := by
      linarith
    rw [hu']
    have : u = j + ((b : ℤ) + 1) * w := by linarith
    rw [this]
    ring

end NiceSearch
