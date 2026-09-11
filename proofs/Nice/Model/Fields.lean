/-
Field and chunk generation. Rust: `generate_fields::break_range_into_fields`,
`generate_chunks::group_fields_into_chunks`, `db_util::chunks::reassign_fields_to_chunks`.

Field `i` of a base range `[lo, hi)` with field size `s` is
`[lo + i·s, min(lo + (i+1)·s, hi))`; chunk `j` groups `per` consecutive
fields. Every number is in exactly one field, every field lies inside the
chunk that contains its start point — which is what the start-point-only
SQL match in `reassign_fields_to_chunks` relies on (FLD-1).
-/
import Mathlib.Tactic

namespace Nice

/-- The field containing `n`. -/
def fieldIndex (lo s n : ℕ) : ℕ := (n - lo) / s

/-- `n` lies in field `i` (ignoring the truncation at `hi`, which only shortens the last field). -/
def InField (lo s i n : ℕ) : Prop := lo + i * s ≤ n ∧ n < lo + (i + 1) * s

/-- Claim FLD-1 (fields): each `n ≥ lo` is in exactly one field. -/
theorem inField_iff {lo s i n : ℕ} (hs : 0 < s) (hn : lo ≤ n) :
    InField lo s i n ↔ i = fieldIndex lo s n := by
  unfold InField fieldIndex
  constructor
  · rintro ⟨h1, h2⟩
    symm
    apply Nat.div_eq_of_lt_le <;> omega
  · rintro rfl
    have h := Nat.div_add_mod (n - lo) s
    have hr := Nat.mod_lt (n - lo) hs
    rw [Nat.mul_comm] at h
    rw [Nat.add_mul, Nat.one_mul]
    constructor <;> omega

/-- Claim FLD-1 (chunks): field `i` lies inside chunk `i / per`, and the
chunk containing the field's start point is that chunk. -/
theorem field_subset_chunk {lo s per i n : ℕ} (hper : 0 < per) (h : InField lo s i n) :
    InField lo (per * s) (i / per) n := by
  unfold InField at *
  obtain ⟨h1, h2⟩ := h
  have hd := Nat.div_add_mod i per
  have hd' := hd
  rw [Nat.mul_comm] at hd'
  have hr := Nat.mod_lt i hper
  constructor
  · calc lo + i / per * (per * s) = lo + (per * (i / per)) * s := by ring
      _ ≤ lo + i * s := by
        apply Nat.add_le_add_left
        apply Nat.mul_le_mul_right
        omega
      _ ≤ n := h1
  · calc n < lo + (i + 1) * s := h2
      _ ≤ lo + (i / per + 1) * (per * s) := by
        apply Nat.add_le_add_left
        have : i + 1 ≤ (i / per + 1) * per := by
          rw [Nat.add_mul, Nat.one_mul]; omega
        calc (i + 1) * s ≤ (i / per + 1) * per * s := Nat.mul_le_mul_right _ this
          _ = (i / per + 1) * (per * s) := by ring

theorem chunk_of_field_start {lo s per i : ℕ} (hs : 0 < s) (hper : 0 < per) :
    fieldIndex lo (per * s) (lo + i * s) = i / per := by
  unfold fieldIndex
  rw [Nat.add_sub_cancel_left]
  have hd := Nat.div_add_mod i per
  have hd' := hd
  rw [Nat.mul_comm] at hd'
  have hr := Nat.mod_lt i hper
  apply Nat.div_eq_of_lt_le
  · calc i / per * (per * s) = (per * (i / per)) * s := by ring
      _ ≤ i * s := Nat.mul_le_mul_right _ (by omega)
  · have : i + 1 ≤ (i / per + 1) * per := by
      rw [Nat.add_mul, Nat.one_mul]; omega
    calc i * s < (i + 1) * s := Nat.mul_lt_mul_of_pos_right (by omega) hs
      _ ≤ (i / per + 1) * per * s := Nat.mul_le_mul_right _ this
      _ = (i / per + 1) * (per * s) := by ring

end Nice
