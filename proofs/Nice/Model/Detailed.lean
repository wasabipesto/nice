/-
Detailed mode. Rust: `client_process::get_num_unique_digits`,
`distribution_stats.rs`, `number_stats.rs`.

Detailed mode counts distinct output digits (`num_uniques`) for every
candidate. Inside the range the output has exactly `b` digits, so the count
is `b` exactly for nice numbers (DEF-2), and it is never zero (DEF-3).
-/
import Nice.Model.Seeded

namespace Nice

/-- Claim DEF-2: inside the range, `num_uniques = b` is niceness. -/
theorem numUniques_eq_iff_isNice {b n : ℕ} (hb : 2 ≤ b) (hr : InBaseRange b n) :
    numUniques b n = b ↔ IsNice b n := by
  rw [isNice_iff_nodup hb hr]
  unfold numUniques
  have hlen : (outputDigits b n).length = b := by
    unfold InBaseRange numDigits at hr
    simpa [outputDigits] using hr
  have key : (outputDigits b n).toFinset.card = (outputDigits b n).length ↔
      (outputDigits b n).Nodup :=
    ⟨fun h => Multiset.coe_nodup.mp (Multiset.toFinset_card_eq_card_iff_nodup.mp h),
      List.toFinset_card_of_nodup⟩
  rw [hlen] at key
  exact key

/-- Claim DEF-3: a positive number has at least one output digit, so the
histogram's bin 0 is empty. -/
theorem one_le_numUniques {b n : ℕ} (hn : n ≠ 0) : 1 ≤ numUniques b n := by
  unfold numUniques
  apply Finset.card_pos.mpr
  have : Nat.digits b (n ^ 2) ≠ [] :=
    Nat.digits_ne_nil_iff_ne_zero.mpr (pow_ne_zero _ hn)
  obtain ⟨d, hd⟩ := List.exists_mem_of_ne_nil _ this
  exact ⟨d, List.mem_toFinset.mpr (List.mem_append_left _ hd)⟩

/-! ### DET-1: the accumulators -/

/-- Histogram bins fold batch by batch: the count of a value over a
concatenation is the sum of the counts (`DistributionAccumulator`). -/
theorem histogram_fold {α : Type*} (p : α → Bool) (l₁ l₂ : List α) :
    (l₁ ++ l₂).countP p = l₁.countP p + l₂.countP p :=
  List.countP_append

/-- Top-N compaction never drops a number that belongs in the final top-N:
an element with fewer than `N` strictly larger keys in the whole list has
fewer than `N` in any batch (`NumbersAccumulator`). Ties are not broken
here; the Rust breaks them by value, which only matters for equal keys. -/
theorem topN_of_superset {α : Type*} {key : α → ℕ} {N : ℕ} {l₁ l : List α} (hsub : List.Sublist l₁ l)
    (x : α) (hx : (l.filter fun y => decide (key x < key y)).length < N) :
    (l₁.filter fun y => decide (key x < key y)).length < N :=
  lt_of_le_of_lt (hsub.filter _).length_le hx

/-- Claim DEF-4: the near-miss cutoff `⌊0.9·b⌋`; a near miss has strictly
more distinct digits than this (`NEAR_MISS_CUTOFF_PERCENT = 0.9`, strict
`>` in every implementation). The Rust computes it in `f32`; for every
base the search can represent the rounding never crosses an integer, so
this is the same number. -/
def nearMissCutoff (b : ℕ) : ℕ := 9 * b / 10

def IsNearMiss (b n : ℕ) : Prop := nearMissCutoff b < numUniques b n

/-- Every nice number is a near miss. -/
theorem isNearMiss_of_isNice {b n : ℕ} (hb : 2 ≤ b) (h : IsNice b n) : IsNearMiss b n := by
  unfold IsNearMiss
  rw [(numUniques_eq_iff_isNice hb (inBaseRange_of_isNice h)).mpr h]
  unfold nearMissCutoff
  omega

end Nice
