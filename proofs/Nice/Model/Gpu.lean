/-
GPU candidate enumeration. Rust: `cuda/nice_kernels.cu`, `vulkan/codegen.rs`,
`cubecl_backend.rs` (the same formula in three kernels).

Instead of walking the gap table, a lane computes its `g`-th candidate
directly: with `V` the sorted valid residues, `M` the modulus and `B0` the
multiple of `M` at or below the range start,

    ordinal g = B0 + (g / |V|) · M + V[g mod |V|].

Claim GPU-0: this is strictly increasing in `g`, every value is a valid
number, and every valid number `≥ B0` is hit — so the lanes' ordinals,
starting from the first index whose value reaches the range start,
enumerate exactly the stride candidates (`walk_eq_filter`).
-/
import Nice.Model.Stride
import Mathlib.Data.Finset.Sort

namespace Nice

/-- The Rust `valid_residues` vector: the residue set, sorted. -/
def residueList (b k : ℕ) : List ℕ := (validResidues b k).sort (· ≤ ·)

theorem residueList_sorted (b k : ℕ) : (residueList b k).Pairwise (· < ·) :=
  ((Finset.pairwise_sort _ _).and (Finset.sort_nodup _ _)).imp fun ⟨hle, hne⟩ =>
    lt_of_le_of_ne hle hne

theorem mem_residueList {b k r : ℕ} : r ∈ residueList b k ↔ r ∈ validResidues b k :=
  Finset.mem_sort _

theorem length_residueList (b k : ℕ) : (residueList b k).length = (validResidues b k).card :=
  Finset.length_sort _

/-- The `g`-th candidate from block base `B0`. -/
def ordinal (b k B0 g : ℕ) : ℕ :=
  let V := residueList b k
  B0 + g / V.length * strideModulus b k + V.getD (g % V.length) 0

theorem residueList_getD_lt {b k : ℕ} {i : ℕ} (hi : i < (residueList b k).length) :
    (residueList b k).getD i 0 < strideModulus b k := by
  rw [List.getD_eq_getElem _ _ hi]
  have hmem : (residueList b k)[i] ∈ residueList b k := List.getElem_mem hi
  rw [mem_residueList] at hmem
  unfold validResidues at hmem
  dsimp only at hmem
  exact Finset.mem_range.mp (Finset.mem_filter.mp hmem).1

/-- Every ordinal is a valid number (when `B0` is a multiple of `M`). -/
theorem ordinal_valid {b k B0 : ℕ} (hV : (validResidues b k).Nonempty)
    (hB0 : B0 % strideModulus b k = 0) (g : ℕ) : IsValid b k (ordinal b k B0 g) := by
  unfold IsValid ordinal
  simp only
  have hlen : 0 < (residueList b k).length := by
    rw [length_residueList]; exact Finset.card_pos.mpr hV
  have hi := Nat.mod_lt g hlen
  have hlt := residueList_getD_lt hi
  have hmem : (residueList b k).getD (g % (residueList b k).length) 0 ∈ validResidues b k := by
    rw [List.getD_eq_getElem _ _ hi, ← mem_residueList]
    exact List.getElem_mem hi
  rw [Nat.add_mod, Nat.add_mod B0, hB0, Nat.mul_mod_left, Nat.zero_add, Nat.zero_mod, Nat.zero_add,
    Nat.mod_mod, Nat.mod_eq_of_lt hlt]
  exact hmem

/-- Ordinals are strictly increasing. -/
theorem ordinal_strictMono {b k B0 : ℕ} (hb : 2 ≤ b) (hV : (validResidues b k).Nonempty) :
    StrictMono (ordinal b k B0) := by
  have hlen : 0 < (residueList b k).length := by
    rw [length_residueList]; exact Finset.card_pos.mpr hV
  have hM : 0 < strideModulus b k := strideModulus_pos hb
  intro g g' hgg'
  unfold ordinal
  simp only
  rcases Nat.lt_or_ge (g / (residueList b k).length) (g' / (residueList b k).length) with hq | hq
  · -- different blocks: a whole modulus separates them
    have h1 := residueList_getD_lt (Nat.mod_lt g hlen)
    have h2 : (g / (residueList b k).length + 1) * strideModulus b k ≤
        g' / (residueList b k).length * strideModulus b k := Nat.mul_le_mul_right _ hq
    rw [Nat.add_mul, Nat.one_mul] at h2
    omega
  · -- same block: the index moves within the sorted list
    have hq' : g / (residueList b k).length = g' / (residueList b k).length := by
      have := Nat.div_le_div_right (c := (residueList b k).length) (Nat.le_of_lt hgg')
      omega
    have hr : g % (residueList b k).length < g' % (residueList b k).length := by
      have h1 := Nat.div_add_mod g (residueList b k).length
      have h2 := Nat.div_add_mod g' (residueList b k).length
      rw [hq'] at h1
      omega
    rw [hq']
    have hlt : (residueList b k).getD (g % (residueList b k).length) 0 <
        (residueList b k).getD (g' % (residueList b k).length) 0 := by
      rw [List.getD_eq_getElem _ _ (Nat.mod_lt g hlen), List.getD_eq_getElem _ _ (Nat.mod_lt g' hlen)]
      exact List.pairwise_iff_getElem.mp (residueList_sorted b k) _ _ _ _ hr
    omega

/-- Every valid number at or above `B0` is some ordinal. -/
theorem exists_ordinal_eq {b k B0 n : ℕ} (hb : 2 ≤ b)
    (hB0 : B0 % strideModulus b k = 0) (hn : B0 ≤ n) (hv : IsValid b k n) :
    ∃ g, ordinal b k B0 g = n := by
  have hM : 0 < strideModulus b k := strideModulus_pos hb
  have hmem : n % strideModulus b k ∈ residueList b k := mem_residueList.mpr hv
  have hlen : 0 < (residueList b k).length := List.length_pos_of_mem hmem
  have hidx : (residueList b k).idxOf (n % strideModulus b k) < (residueList b k).length :=
    List.idxOf_lt_length_of_mem hmem
  refine ⟨(n - B0) / strideModulus b k * (residueList b k).length +
    (residueList b k).idxOf (n % strideModulus b k), ?_⟩
  unfold ordinal
  simp only
  rw [Nat.mul_comm ((n - B0) / strideModulus b k), Nat.mul_add_div hlen, Nat.mul_add_mod,
    Nat.div_eq_of_lt hidx, Nat.mod_eq_of_lt hidx, Nat.add_zero,
    List.getD_eq_getElem _ _ hidx, List.getElem_idxOf hidx]
  have h1 := Nat.div_add_mod (n - B0) (strideModulus b k)
  rw [Nat.mul_comm] at h1
  have h2 : (n - B0) % strideModulus b k = n % strideModulus b k := by
    have hsplit : n = n - B0 + B0 := (Nat.sub_add_cancel hn).symm
    conv_rhs => rw [hsplit]
    rw [Nat.add_mod, hB0, Nat.add_zero, Nat.mod_mod]
  omega

end Nice
