/-
The MSD interval-domain filter. Rust: `msd_prefix_filter::{collect_power_domains,
has_distinct_assignment, analyze_range}`.

Over a range `[lo, hi]`, the quotient `n^e / b^j` is squeezed between the
endpoints' quotients, so digit `j` of `n^e` lies in a cyclic interval of
residues (MSD-1). The interval width one position down follows the
recurrence `diff_{j} = b·diff_{j+1} + (yd_j − xd_j)` (MSD-2); once it
reaches `b − 1` every lower position is unconstrained. A nice number's
digits at distinct positions are distinct, so they form a system of
distinct representatives of the domains: if none exists, no nice number
is in the range (MSD-4).
-/
import Nice.Model.Lsd
import Mathlib.Tactic

namespace Nice

/-! ### Cyclic interval domains -/

/-- Residues `lo, lo+1, …, lo+width-1` mod `b`. -/
def cyclicInterval (b lo width : ℕ) : Finset ℕ :=
  (Finset.range width).image fun i => (lo + i) % b

/-- Width `≥ b` is every digit. -/
theorem cyclicInterval_eq_range {b lo width : ℕ} (hb : 0 < b) (h : b ≤ width) :
    cyclicInterval b lo width = Finset.range b := by
  ext d
  simp only [cyclicInterval, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨i, -, rfl⟩
    exact Nat.mod_lt _ hb
  · intro hd
    refine ⟨(d + b - lo % b) % b, lt_of_lt_of_le (Nat.mod_lt _ hb) h, ?_⟩
    have hlo := Nat.mod_lt lo hb
    have : lo % b + (d + b - lo % b) = d + b := by omega
    rw [Nat.add_mod_mod, Nat.add_mod, Nat.add_mod_mod, this, Nat.add_mod_right,
      Nat.mod_eq_of_lt hd]

/-- Claim MSD-1: over `[lo, hi]`, digit `j` of `n^e` lies in the cyclic
interval starting at the low endpoint's digit with the quotient width. -/
theorem digit_mem_cyclicInterval {b lo hi n e j : ℕ} (hlo : lo ≤ n) (hhi : n ≤ hi) :
    digit b (n ^ e) j ∈
      cyclicInterval b (digit b (lo ^ e) j) (hi ^ e / b ^ j - lo ^ e / b ^ j + 1) := by
  unfold cyclicInterval digit
  rw [Finset.mem_image]
  have hu : lo ^ e / b ^ j ≤ n ^ e / b ^ j := Nat.div_le_div_right (Nat.pow_le_pow_left hlo e)
  have hv : n ^ e / b ^ j ≤ hi ^ e / b ^ j := Nat.div_le_div_right (Nat.pow_le_pow_left hhi e)
  refine ⟨n ^ e / b ^ j - lo ^ e / b ^ j, Finset.mem_range.mpr (by omega), ?_⟩
  rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod, Nat.add_sub_cancel' hu]

/-! ### The width recurrence -/

/-- Peeling one digit: `x / b^j = b · (x / b^(j+1)) + digit_j`. -/
theorem div_pow_succ (b x j : ℕ) : x / b ^ j = b * (x / b ^ (j + 1)) + digit b x j := by
  unfold digit
  rw [pow_succ, ← Nat.div_div_eq_div_mul, Nat.mul_comm]
  exact (Nat.div_add_mod' (x / b ^ j) b).symm

/-- Claim MSD-2: the interval width one position down. -/
theorem width_recurrence (b lo hi e j : ℕ) :
    ((hi ^ e / b ^ j : ℕ) : ℤ) - (lo ^ e / b ^ j : ℕ) =
      b * (((hi ^ e / b ^ (j + 1) : ℕ) : ℤ) - (lo ^ e / b ^ (j + 1) : ℕ)) +
        ((digit b (hi ^ e) j : ℤ) - digit b (lo ^ e) j) := by
  have h1 : ((hi ^ e / b ^ j : ℕ) : ℤ) =
      b * ((hi ^ e / b ^ (j + 1) : ℕ) : ℤ) + (digit b (hi ^ e) j : ℤ) := by
    exact_mod_cast div_pow_succ b (hi ^ e) j
  have h2 : ((lo ^ e / b ^ j : ℕ) : ℤ) =
      b * ((lo ^ e / b ^ (j + 1) : ℕ) : ℤ) + (digit b (lo ^ e) j : ℤ) := by
    exact_mod_cast div_pow_succ b (lo ^ e) j
  linear_combination h1 - h2

/-- Once a width reaches `b - 1`, every lower width does too (so the
Rust may stop collecting domains). -/
theorem width_ge_of_succ {b lo hi e j : ℕ} (hb : 2 ≤ b) (hle : lo ≤ hi)
    (h : b - 1 ≤ hi ^ e / b ^ (j + 1) - lo ^ e / b ^ (j + 1)) :
    b - 1 ≤ hi ^ e / b ^ j - lo ^ e / b ^ j := by
  have h1 := div_pow_succ b (hi ^ e) j
  have h2 := div_pow_succ b (lo ^ e) j
  have hdl : digit b (lo ^ e) j < b := digit_lt_base (by omega) _ _
  have hmono : lo ^ e / b ^ (j + 1) ≤ hi ^ e / b ^ (j + 1) :=
    Nat.div_le_div_right (Nat.pow_le_pow_left hle e)
  -- b · A ≥ b · B + b · (b - 1), and b · (b - 1) ≥ 2 (b - 1)
  have hA : b * (lo ^ e / b ^ (j + 1) + (b - 1)) ≤ b * (hi ^ e / b ^ (j + 1)) :=
    Nat.mul_le_mul_left _ (by omega)
  rw [Nat.mul_add] at hA
  have hbb : 2 * (b - 1) ≤ b * (b - 1) := Nat.mul_le_mul_right _ hb
  omega

/-! ### Systems of distinct representatives -/

/-- One constrained output position: power `e ∈ {2, 3}`, digit position `j`,
allowed digits `dom`. -/
structure Constraint where
  e : ℕ
  j : ℕ
  dom : Finset ℕ

/-- The constraints are honest for the range: each is at a real position of
its power for every `n`, and contains the digit that actually occurs. -/
def Sound (b lo hi : ℕ) (cs : List Constraint) : Prop :=
  ∀ c ∈ cs, (c.e = 2 ∨ c.e = 3) ∧
    ∀ n, lo ≤ n → n ≤ hi → c.j < numDigits b (n ^ c.e) ∧ digit b (n ^ c.e) c.j ∈ c.dom

/-- A choice of one digit per constraint, all distinct. -/
def HasSDR (cs : List Constraint) : Prop :=
  ∃ f : ℕ → ℕ → ℕ, (∀ c ∈ cs, f c.e c.j ∈ c.dom) ∧ (cs.map fun c => f c.e c.j).Nodup

/-- Index of position `j` of power `e` inside the concatenated output digits. -/
def outIndex (b n e j : ℕ) : ℕ := if e = 2 then j else numDigits b (n ^ 2) + j

theorem outputDigits_getD {b n e j : ℕ} (hb : 2 ≤ b) (he : e = 2 ∨ e = 3)
    (hj : j < numDigits b (n ^ e)) :
    (outputDigits b n).getD (outIndex b n e j) 0 = digit b (n ^ e) j := by
  unfold outputDigits outIndex
  rcases he with rfl | rfl
  · rw [if_pos rfl, List.getD_append _ _ _ _ hj, digit_eq_getD hb]
  · rw [if_neg (by norm_num)]
    unfold numDigits
    rw [List.getD_append_right _ _ _ _ (Nat.le_add_right _ _), Nat.add_sub_cancel_left,
      digit_eq_getD hb]

theorem outIndex_lt {b n e j : ℕ} (he : e = 2 ∨ e = 3) (hj : j < numDigits b (n ^ e)) :
    outIndex b n e j < (outputDigits b n).length := by
  unfold outIndex outputDigits numDigits at *
  rw [List.length_append]
  rcases he with rfl | rfl
  · rw [if_pos rfl]; omega
  · rw [if_neg (by norm_num)]; omega

theorem outIndex_inj {b n e j e' j' : ℕ} (he : e = 2 ∨ e = 3) (he' : e' = 2 ∨ e' = 3)
    (hj : j < numDigits b (n ^ e)) (hj' : j' < numDigits b (n ^ e'))
    (h : outIndex b n e j = outIndex b n e' j') : e = e' ∧ j = j' := by
  unfold outIndex at h
  rcases he with rfl | rfl <;> rcases he' with rfl | rfl
  · rw [if_pos rfl, if_pos rfl] at h; exact ⟨rfl, h⟩
  · rw [if_pos rfl, if_neg (by norm_num)] at h; omega
  · rw [if_neg (by norm_num), if_pos rfl] at h; omega
  · rw [if_neg (by norm_num), if_neg (by norm_num)] at h; exact ⟨rfl, by omega⟩

/-- Claim MSD-4 (spec): a nice number in the range is itself an SDR. -/
theorem hasSDR_of_isNice {b lo hi n : ℕ} {cs : List Constraint} (hb : 2 ≤ b)
    (hs : Sound b lo hi cs) (hd : (cs.map fun c => (c.e, c.j)).Nodup)
    (hlo : lo ≤ n) (hhi : n ≤ hi) (h : IsNice b n) : HasSDR cs := by
  refine ⟨fun e j => digit b (n ^ e) j, fun c hc => ((hs c hc).2 n hlo hhi).2, ?_⟩
  have hnd : (outputDigits b n).Nodup := h.nodup_iff.mpr List.nodup_range
  have hcs : cs.Nodup := hd.of_map _
  refine List.Nodup.map_on ?_ hcs
  intro c hc c' hc' heq
  have ⟨he, hpos⟩ := hs c hc
  have ⟨he', hpos'⟩ := hs c' hc'
  have hj := (hpos n hlo hhi).1
  have hj' := (hpos' n hlo hhi).1
  -- the digits sit at indices outIndex; equal digits at a nodup list → equal indices
  have h1 := outputDigits_getD hb he hj
  have h2 := outputDigits_getD hb he' hj'
  have hi1 := outIndex_lt he hj
  have hi2 := outIndex_lt he' hj'
  rw [List.getD_eq_getElem _ _ hi1] at h1
  rw [List.getD_eq_getElem _ _ hi2] at h2
  have hidx : outIndex b n c.e c.j = outIndex b n c'.e c'.j := by
    have : (outputDigits b n)[outIndex b n c.e c.j] = (outputDigits b n)[outIndex b n c'.e c'.j] := by
      rw [h1, h2]; exact heq
    exact (List.Nodup.getElem_inj_iff hnd).mp this
  have ⟨hee, hjj⟩ := outIndex_inj he he' hj hj' hidx
  -- (e, j) determines the constraint since the pairs are nodup
  have := List.inj_on_of_nodup_map hd hc hc' (by rw [hee, hjj])
  exact this

/-- Claim MSD-4: no SDR means no nice number in the range. -/
theorem no_nice_of_not_hasSDR {b lo hi : ℕ} {cs : List Constraint} (hb : 2 ≤ b)
    (hs : Sound b lo hi cs) (hd : (cs.map fun c => (c.e, c.j)).Nodup) (hno : ¬ HasSDR cs) :
    ∀ n, lo ≤ n → n ≤ hi → ¬ IsNice b n :=
  fun _ hlo hhi h => hno (hasSDR_of_isNice hb hs hd hlo hhi h)

end Nice
