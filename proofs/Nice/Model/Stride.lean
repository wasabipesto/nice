/-
The stride table. Rust: `stride_filter::StrideTable`.

The residue filter (mod b-1) and the LSD filter (mod b^k) combine into one
set of valid residues mod `M = (b-1)·b^k`; the table walks from one valid
candidate to the next. This file has the residue set (STR-1), the exact
per-residue low-digit set (STR-4), and the walk as a mathematical object:
`nextValid` is `first_valid_at_or_after`, `walk` is `iterate_range`, and
`walk_eq_filter` says the walk visits exactly the valid numbers of the
range in order (STR-2).
-/
import Nice.Model.Residue
import Nice.Model.Lsd
import Mathlib.Data.List.Sort

namespace Nice

/-- The combined modulus `M = (b-1)·b^k`. -/
def strideModulus (b k : ℕ) : ℕ := (b - 1) * b ^ k

theorem strideModulus_pos {b k : ℕ} (hb : 2 ≤ b) : 0 < strideModulus b k :=
  Nat.mul_pos (by omega) (Nat.pow_pos (by omega))

/-- The Rust `valid_residues`: residues mod `M` passing both filters. -/
def validResidues (b k : ℕ) : Finset ℕ :=
  (Finset.range (strideModulus b k)).filter fun r =>
    r % (b - 1) ∈ residueFilter b ∧ r % b ^ k ∈ lsdBitmap b k

/-- Claim STR-1: `n mod M` is a valid residue iff `n` passes both filters. -/
theorem mem_validResidues_iff {b k n : ℕ} (hb : 2 ≤ b) :
    n % strideModulus b k ∈ validResidues b k ↔
      n % (b - 1) ∈ residueFilter b ∧ n % b ^ k ∈ lsdBitmap b k := by
  unfold validResidues
  rw [Finset.mem_filter, Finset.mem_range]
  have h1 : n % strideModulus b k % (b - 1) = n % (b - 1) :=
    Nat.mod_mod_of_dvd n (dvd_mul_right _ _)
  have h2 : n % strideModulus b k % b ^ k = n % b ^ k :=
    Nat.mod_mod_of_dvd n (dvd_mul_left _ _)
  rw [h1, h2]
  simp [Nat.mod_lt _ (strideModulus_pos hb)]

/-- Every nice number's residue is valid (for the production depths). -/
theorem mem_validResidues_of_isNice {b k n : ℕ} (hb : 6 ≤ b) (hk : k ≤ 3) (h : IsNice b n) :
    n % strideModulus b k ∈ validResidues b k :=
  (mem_validResidues_iff (by omega)).mpr
    ⟨mem_residueFilter_of_isNice (by omega) h, mem_lsdBitmap_of_isNice' hb hk h⟩

/-- Claim STR-4: the per-residue `low_digit_masks` entry, as a set of digits. -/
def lowMask (b k r : ℕ) : Finset ℕ := (suffixDigits b k (r % b ^ k)).toFinset

/-- The mask of `n`'s residue is exactly the set of low digits of `n²` and `n³`. -/
theorem lowMask_eq {b k n : ℕ} :
    lowMask b k (n % strideModulus b k) = (lowDigits b k (n ^ 2) ++ lowDigits b k (n ^ 3)).toFinset := by
  unfold lowMask
  have hd : b ^ k ∣ strideModulus b k := dvd_mul_left _ _
  rw [Nat.mod_mod_of_dvd n hd, suffixDigits_mod]

/-! ### The walk -/

/-- A valid candidate: `n mod M ∈ validResidues`. -/
def IsValid (b k n : ℕ) : Prop := n % strideModulus b k ∈ validResidues b k

instance (b k n : ℕ) : Decidable (IsValid b k n) := by unfold IsValid; infer_instance

/-- The candidates of a half-open range, as the filter of the range: what
`iterate_range` must visit. -/
def strideCandidates (b k start stop : ℕ) : List ℕ :=
  (List.range' start (stop - start)).filter fun n => decide (IsValid b k n)

theorem mem_strideCandidates {b k start stop n : ℕ} :
    n ∈ strideCandidates b k start stop ↔ start ≤ n ∧ n < stop ∧ IsValid b k n := by
  unfold strideCandidates
  rw [List.mem_filter, List.mem_range'_1, decide_eq_true_iff]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3⟩; exact ⟨h1, by omega, h3⟩
  · rintro ⟨h1, h2, h3⟩; exact ⟨⟨h1, by omega⟩, h3⟩

/-- Every nice number of a range is a stride candidate. -/
theorem mem_strideCandidates_of_isNice {b k start stop n : ℕ} (hb : 6 ≤ b) (hk : k ≤ 3)
    (hs : start ≤ n) (he : n < stop) (h : IsNice b n) :
    n ∈ strideCandidates b k start stop :=
  mem_strideCandidates.mpr ⟨hs, he, mem_validResidues_of_isNice hb hk h⟩

/-- Some valid number exists at or after any `n` when the residue set is nonempty. -/
theorem exists_valid_ge {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) (n : ℕ) :
    ∃ d, IsValid b k (n + d) := by
  obtain ⟨r, hr⟩ := hV
  set M := strideModulus b k with hM
  have hMpos : 0 < M := strideModulus_pos hb
  have hrM : r < M := Finset.mem_range.mp (Finset.mem_filter.mp hr).1
  refine ⟨M + r - n % M, ?_⟩
  unfold IsValid
  have hdiv := Nat.div_add_mod n M
  have hlt := Nat.mod_lt n hMpos
  have : n + (M + r - n % M) = M * (n / M + 1) + r := by
    rw [Nat.mul_add, Nat.mul_one]; omega
  rw [← hM, this, Nat.mul_add_mod, Nat.mod_eq_of_lt hrM]
  exact hr

/-- `first_valid_at_or_after n`: the least valid number `≥ n`. -/
noncomputable def nextValid {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) (n : ℕ) : ℕ :=
  n + Nat.find (exists_valid_ge hb hV n)

theorem le_nextValid {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) (n : ℕ) :
    n ≤ nextValid hb hV n :=
  Nat.le_add_right _ _

theorem nextValid_valid {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) (n : ℕ) :
    IsValid b k (nextValid hb hV n) :=
  Nat.find_spec (exists_valid_ge hb hV n)

/-- Nothing between `n` and `nextValid n` is valid. -/
theorem not_valid_of_lt_nextValid {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty)
    {n m : ℕ} (hn : n ≤ m) (hm : m < nextValid hb hV n) : ¬ IsValid b k m := by
  unfold nextValid at hm
  have := Nat.find_min (exists_valid_ge hb hV n) (m := m - n) (by omega)
  rwa [Nat.add_sub_cancel' hn] at this

theorem nextValid_le_of_valid {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty)
    {n m : ℕ} (hn : n ≤ m) (hm : IsValid b k m) : nextValid hb hV n ≤ m := by
  by_contra hc
  exact not_valid_of_lt_nextValid hb hV hn (Nat.lt_of_not_le hc) hm

/-- `iterate_range` from `n`: emit the next valid number and continue after it. -/
noncomputable def walk {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) (stop n : ℕ) :
    List ℕ :=
  if h : nextValid hb hV n < stop then
    nextValid hb hV n :: walk hb hV stop (nextValid hb hV n + 1)
  else []
termination_by stop - n
decreasing_by
  have := le_nextValid hb hV n
  omega

theorem mem_walk {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) (stop n x : ℕ) :
    x ∈ walk hb hV stop n ↔ n ≤ x ∧ x < stop ∧ IsValid b k x := by
  induction n using walk.induct hb hV stop generalizing x with
  | case1 n h ih =>
    rw [walk, dif_pos h, List.mem_cons, ih]
    constructor
    · rintro (rfl | ⟨h1, h2, h3⟩)
      · exact ⟨le_nextValid hb hV n, h, nextValid_valid hb hV n⟩
      · exact ⟨by have := le_nextValid hb hV n; omega, h2, h3⟩
    · rintro ⟨h1, h2, h3⟩
      by_cases hx : x = nextValid hb hV n
      · exact Or.inl hx
      · refine Or.inr ⟨?_, h2, h3⟩
        have := nextValid_le_of_valid hb hV h1 h3
        omega
  | case2 n h =>
    rw [walk, dif_neg h]
    simp only [List.not_mem_nil, false_iff]
    rintro ⟨h1, h2, h3⟩
    have := nextValid_le_of_valid hb hV h1 h3
    omega

theorem walk_sorted {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) (stop n : ℕ) :
    (walk hb hV stop n).Pairwise (· < ·) := by
  induction n using walk.induct hb hV stop with
  | case1 n h ih =>
    rw [walk, dif_pos h, List.pairwise_cons]
    refine ⟨fun y hy => ?_, ih⟩
    have := (mem_walk hb hV stop _ y).mp hy
    omega
  | case2 n h =>
    rw [walk, dif_neg h]
    exact List.Pairwise.nil

theorem strideCandidates_sorted (b k start stop : ℕ) :
    (strideCandidates b k start stop).Pairwise (· < ·) :=
  (List.pairwise_lt_range' (s := start) (n := stop - start) 1).filter _

/-- Two strictly increasing lists with the same members are equal. -/
theorem eq_of_pairwise_lt_of_mem_iff :
    ∀ {l₁ l₂ : List ℕ}, l₁.Pairwise (· < ·) → l₂.Pairwise (· < ·) →
      (∀ x, x ∈ l₁ ↔ x ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => by have := (h b).mpr (by simp); simp at this
  | a :: _, [], _, _, h => by have := (h a).mp (by simp); simp at this
  | a :: l₁, b :: l₂, h₁, h₂, h => by
    rw [List.pairwise_cons] at h₁ h₂
    have hab : a = b := by
      have ha : a ∈ b :: l₂ := (h a).mp (by simp)
      have hb : b ∈ a :: l₁ := (h b).mpr (by simp)
      rw [List.mem_cons] at ha hb
      rcases ha with rfl | ha
      · rfl
      rcases hb with rfl | hb
      · rfl
      have := h₁.1 b hb
      have := h₂.1 a ha
      omega
    subst hab
    congr 1
    apply eq_of_pairwise_lt_of_mem_iff h₁.2 h₂.2
    intro x
    constructor
    · intro hx
      have := (h x).mp (List.mem_cons_of_mem _ hx)
      rw [List.mem_cons] at this
      rcases this with rfl | h'
      · exact absurd (h₁.1 x hx) (lt_irrefl _)
      · exact h'
    · intro hx
      have := (h x).mpr (List.mem_cons_of_mem _ hx)
      rw [List.mem_cons] at this
      rcases this with rfl | h'
      · exact absurd (h₂.1 x hx) (lt_irrefl _)
      · exact h'

/-- Claim STR-2: the walk visits exactly the candidates of `[start, stop)`, in order. -/
theorem walk_eq_filter {b k : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty)
    (start stop : ℕ) : walk hb hV stop start = strideCandidates b k start stop := by
  apply eq_of_pairwise_lt_of_mem_iff (walk_sorted hb hV stop start)
    (strideCandidates_sorted b k start stop)
  intro x
  rw [mem_walk, mem_strideCandidates]

end Nice
