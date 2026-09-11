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

end Nice
