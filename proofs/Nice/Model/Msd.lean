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

/-- A choice of one digit per constraint, all distinct (a system of
distinct representatives). -/
def HasSDR (cs : List Constraint) : Prop :=
  ∃ ds : List ℕ, List.Forall₂ (fun c d => d ∈ c.dom) cs ds ∧ ds.Nodup

/-- Claim MSD-6: dropping constraints keeps a system sound (so running out
of domain slots, or skipping a power, only weakens the check). -/
theorem Sound.sublist {b lo hi : ℕ} {cs cs' : List Constraint} (h : List.Sublist cs' cs)
    (hs : Sound b lo hi cs) : Sound b lo hi cs' :=
  fun c hc => hs c (h.subset hc)

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
  refine ⟨cs.map fun c => digit b (n ^ c.e) c.j, ?_, ?_⟩
  · rw [List.forall₂_map_right_iff, List.forall₂_same]
    exact fun c hc => ((hs c hc).2 n hlo hhi).2
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

/-! ### The executable model -/

/-- Backtracking SDR search over a candidate digit list: pick an unused
digit for the first constraint and recurse. The Rust uses Kuhn's matching
for the same question. -/
def sdrAux (cands : List ℕ) : Finset ℕ → List Constraint → Bool
  | _, [] => true
  | used, c :: cs => cands.any fun d => decide (d ∈ c.dom ∧ d ∉ used) && sdrAux cands (insert d used) cs

theorem sdrAux_iff (cands : List ℕ) (used : Finset ℕ) :
    ∀ cs : List Constraint, (∀ c ∈ cs, ∀ d ∈ c.dom, d ∈ cands) →
      (sdrAux cands used cs = true ↔
        ∃ ds : List ℕ, List.Forall₂ (fun c d => d ∈ c.dom) cs ds ∧ ds.Nodup ∧ ∀ d ∈ ds, d ∉ used) := by
  intro cs
  induction cs generalizing used with
  | nil =>
    intro _
    simp only [sdrAux, true_iff]
    exact ⟨[], List.Forall₂.nil, List.nodup_nil, by simp⟩
  | cons c cs ih =>
    intro hsub
    have hsub' : ∀ c ∈ cs, ∀ d ∈ c.dom, d ∈ cands := fun c hc => hsub c (List.mem_cons_of_mem _ hc)
    simp only [sdrAux, List.any_eq_true, Bool.and_eq_true, decide_eq_true_iff]
    constructor
    · rintro ⟨d, -, ⟨hd, hdu⟩, hrec⟩
      obtain ⟨ds, hf, hnd, hus⟩ := (ih _ hsub').mp hrec
      refine ⟨d :: ds, List.Forall₂.cons hd hf, ?_, ?_⟩
      · rw [List.nodup_cons]
        exact ⟨fun hmem => hus d hmem (Finset.mem_insert_self _ _), hnd⟩
      · intro x hx
        rw [List.mem_cons] at hx
        rcases hx with rfl | hx
        · exact hdu
        · exact fun hxu => hus x hx (Finset.mem_insert_of_mem hxu)
    · rintro ⟨ds, hf, hnd, hus⟩
      cases hf with
      | cons hd hf' =>
        rename_i d ds'
        rw [List.nodup_cons] at hnd
        refine ⟨d, hsub c (by simp) d hd, ⟨hd, hus d (by simp)⟩, (ih _ hsub').mpr ⟨ds', hf', hnd.2, ?_⟩⟩
        intro x hx hxu
        rw [Finset.mem_insert] at hxu
        rcases hxu with rfl | hxu
        · exact hnd.1 hx
        · exact hus x (List.mem_cons_of_mem _ hx) hxu

/-- `has_distinct_assignment`, as brute force over the digits of base `b`. -/
def sdrExists (b : ℕ) (cs : List Constraint) : Bool := sdrAux (List.range b) ∅ cs

theorem sdrExists_iff {b : ℕ} (cs : List Constraint) (hsub : ∀ c ∈ cs, ∀ d ∈ c.dom, d < b) :
    sdrExists b cs = true ↔ HasSDR cs := by
  unfold sdrExists HasSDR
  rw [sdrAux_iff _ _ cs (fun c hc d hd => List.mem_range.mpr (hsub c hc d hd))]
  constructor
  · rintro ⟨ds, hf, hnd, -⟩; exact ⟨ds, hf, hnd⟩
  · rintro ⟨ds, hf, hnd⟩; exact ⟨ds, hf, hnd, by simp⟩

/-- The interval width at position `j` for power `e` over `[lo, hi]`. -/
def width (b lo hi e j : ℕ) : ℕ := hi ^ e / b ^ j - lo ^ e / b ^ j

/-- Position `j` is constrained when it and every position above it has width
below `b - 1`. The Rust walks from the top and stops at the first wide
position; by `width_ge_of_succ` every position below a wide one is wide, so
this is the same set. -/
def constrained (b lo hi e j : ℕ) : Bool :=
  (List.range (numDigits b (lo ^ e))).all fun j' => decide (j ≤ j' → width b lo hi e j' < b - 1)

/-- `collect_power_domains`: the constrained positions, only when both
endpoints have the same digit count. -/
def powerDomains (b lo hi e : ℕ) : List Constraint :=
  if numDigits b (lo ^ e) = numDigits b (hi ^ e) then
    (((List.range (numDigits b (lo ^ e))).reverse.filter fun j => constrained b lo hi e j).map fun j =>
      ⟨e, j, cyclicInterval b (digit b (lo ^ e) j) (width b lo hi e j + 1)⟩)
  else []

theorem powerDomains_e {b lo hi e : ℕ} {c : Constraint} (hc : c ∈ powerDomains b lo hi e) :
    c.e = e := by
  unfold powerDomains at hc
  split_ifs at hc
  · rw [List.mem_map] at hc
    obtain ⟨j, -, rfl⟩ := hc
    rfl
  · simp at hc

theorem powerDomains_j_lt {b lo hi e : ℕ} {c : Constraint} (hc : c ∈ powerDomains b lo hi e) :
    c.j < numDigits b (lo ^ e) := by
  unfold powerDomains at hc
  split_ifs at hc
  · rw [List.mem_map] at hc
    obtain ⟨j, hj, rfl⟩ := hc
    have := (List.mem_filter.mp hj).1
    simpa using this
  · simp at hc

theorem powerDomains_nodup (b lo hi e : ℕ) :
    ((powerDomains b lo hi e).map fun c => (c.e, c.j)).Nodup := by
  unfold powerDomains
  split_ifs
  · rw [List.map_map]
    refine List.Nodup.map_on ?_ ((List.nodup_reverse.mpr List.nodup_range).filter _)
    intro x _ y _ h
    simpa using h
  · exact List.nodup_nil

/-- Claim MSD-3 (with MSD-1): the collected domains are sound. -/
theorem powerDomains_sound {b lo hi e : ℕ} (he : e = 2 ∨ e = 3) :
    Sound b lo hi (powerDomains b lo hi e) := by
  intro c hc
  have hce := powerDomains_e hc
  refine ⟨hce ▸ he, fun n hlo hhi => ⟨?_, ?_⟩⟩
  · have hj := powerDomains_j_lt hc
    have hmono : numDigits b (lo ^ e) ≤ numDigits b (n ^ e) := numDigits_pow_mono e hlo
    rw [hce]; omega
  · unfold powerDomains at hc
    split_ifs at hc
    · rw [List.mem_map] at hc
      obtain ⟨j, -, rfl⟩ := hc
      exact digit_mem_cyclicInterval hlo hhi
    · simp at hc

/-- `analyze_range` without the certificate: both powers' domains, one SDR
question. `true` is `Live`, `false` is `Rejected`. -/
def rangeDomains (b lo hi : ℕ) : List Constraint :=
  powerDomains b lo hi 2 ++ powerDomains b lo hi 3

theorem rangeDomains_nodup (b lo hi : ℕ) :
    ((rangeDomains b lo hi).map fun c => (c.e, c.j)).Nodup := by
  unfold rangeDomains
  rw [List.map_append, List.nodup_append]
  refine ⟨powerDomains_nodup b lo hi 2, powerDomains_nodup b lo hi 3, ?_⟩
  intro x hx y hy hxy
  rw [List.mem_map] at hx hy
  obtain ⟨c, hc, rfl⟩ := hx
  obtain ⟨c', hc', rfl⟩ := hy
  rw [powerDomains_e hc, powerDomains_e hc'] at hxy
  simp at hxy

theorem rangeDomains_sound (b lo hi : ℕ) : Sound b lo hi (rangeDomains b lo hi) := by
  intro c hc
  unfold rangeDomains at hc
  rw [List.mem_append] at hc
  rcases hc with hc | hc
  · exact powerDomains_sound (Or.inl rfl) c hc
  · exact powerDomains_sound (Or.inr rfl) c hc

def analyzeRange (b lo hi : ℕ) : Bool := sdrExists b (rangeDomains b lo hi)

/-- Every domain digit is below the base. -/
theorem cyclicInterval_lt {b lo width d : ℕ} (hb : 0 < b) (hd : d ∈ cyclicInterval b lo width) :
    d < b := by
  unfold cyclicInterval at hd
  rw [Finset.mem_image] at hd
  obtain ⟨i, -, rfl⟩ := hd
  exact Nat.mod_lt _ hb

theorem powerDomains_dom_lt {b lo hi e : ℕ} (hb : 0 < b) {c : Constraint}
    (hc : c ∈ powerDomains b lo hi e) : ∀ d ∈ c.dom, d < b := by
  unfold powerDomains at hc
  split_ifs at hc
  · rw [List.mem_map] at hc
    obtain ⟨j, -, rfl⟩ := hc
    exact fun d hd => cyclicInterval_lt hb hd
  · simp at hc

theorem rangeDomains_dom_lt {b lo hi : ℕ} (hb : 0 < b) :
    ∀ c ∈ rangeDomains b lo hi, ∀ d ∈ c.dom, d < b := by
  intro c hc
  unfold rangeDomains at hc
  rw [List.mem_append] at hc
  rcases hc with hc | hc <;> exact powerDomains_dom_lt hb hc

/-- Claim MSD-4 (model): a rejected range holds no nice number. -/
theorem no_nice_of_analyzeRange {b lo hi : ℕ} (hb : 2 ≤ b)
    (h : analyzeRange b lo hi = false) : ∀ n, lo ≤ n → n ≤ hi → ¬ IsNice b n := by
  apply no_nice_of_not_hasSDR hb (rangeDomains_sound b lo hi) (rangeDomains_nodup b lo hi)
  rw [← sdrExists_iff _ (rangeDomains_dom_lt (by omega))]
  simp [analyzeRange] at h
  simp [h]

/-! ### Recursive subdivision -/

/-- `get_valid_ranges_recursive` with subdivision factor 2: depth fuel,
minimum size, half-open `[start, stop)`. -/
def validRanges (b minSize : ℕ) : ℕ → ℕ → ℕ → List (ℕ × ℕ)
  | 0, start, stop => [(start, stop)]
  | d + 1, start, stop =>
    if stop - start ≤ minSize then [(start, stop)]
    else if analyzeRange b start (stop - 1) = false then []
    else if stop - start < 2 * minSize then [(start, stop)]
    else
      validRanges b minSize d start (start + (stop - start) / 2) ++
        validRanges b minSize d (start + (stop - start) / 2) stop

/-- Claim MSD-7: every nice number of the range lies in some emitted leaf. -/
theorem validRanges_cover {b minSize : ℕ} (hb : 2 ≤ b) (d : ℕ) :
    ∀ start stop n, start ≤ n → n < stop → IsNice b n →
      ∃ r ∈ validRanges b minSize d start stop, r.1 ≤ n ∧ n < r.2 := by
  induction d with
  | zero =>
    intro start stop n h1 h2 _
    exact ⟨(start, stop), by simp [validRanges], h1, h2⟩
  | succ d ih =>
    intro start stop n h1 h2 hn
    unfold validRanges
    split_ifs with hsmall hrej hnw
    · exact ⟨(start, stop), by simp, h1, h2⟩
    · exact absurd hn (no_nice_of_analyzeRange hb hrej n h1 (by omega))
    · exact ⟨(start, stop), by simp, h1, h2⟩
    · rcases Nat.lt_or_ge n (start + (stop - start) / 2) with hmid | hmid
      · obtain ⟨r, hr, hr1, hr2⟩ := ih start _ n h1 hmid hn
        exact ⟨r, List.mem_append_left _ hr, hr1, hr2⟩
      · obtain ⟨r, hr, hr1, hr2⟩ := ih _ stop n hmid h2 hn
        exact ⟨r, List.mem_append_right _ hr, hr1, hr2⟩

/-- Leaves are sub-intervals of the input. -/
theorem validRanges_subset {b minSize : ℕ} (d : ℕ) :
    ∀ start stop r, r ∈ validRanges b minSize d start stop → start ≤ r.1 ∧ r.2 ≤ stop := by
  induction d with
  | zero =>
    intro start stop r hr
    simp [validRanges] at hr
    subst hr; exact ⟨le_refl _, le_refl _⟩
  | succ d ih =>
    intro start stop r hr
    unfold validRanges at hr
    split_ifs at hr
    · simp at hr; subst hr; exact ⟨le_refl _, le_refl _⟩
    · simp at hr
    · simp at hr; subst hr; exact ⟨le_refl _, le_refl _⟩
    · rw [List.mem_append] at hr
      rcases hr with hr | hr
      · have := ih _ _ r hr; omega
      · have := ih _ _ r hr; omega

/-! ### Membership in the collected domains -/

theorem powerDomains_eq {b lo hi e : ℕ} {c : Constraint} (hc : c ∈ powerDomains b lo hi e) :
    numDigits b (lo ^ e) = numDigits b (hi ^ e) := by
  unfold powerDomains at hc
  split_ifs at hc with h
  · exact h
  · simp at hc

theorem powerDomains_constrained {b lo hi e : ℕ} {c : Constraint}
    (hc : c ∈ powerDomains b lo hi e) : constrained b lo hi e c.j = true := by
  unfold powerDomains at hc
  split_ifs at hc
  · rw [List.mem_map] at hc
    obtain ⟨j, hj, rfl⟩ := hc
    exact (List.mem_filter.mp hj).2
  · simp at hc

theorem powerDomains_dom {b lo hi e : ℕ} {c : Constraint} (hc : c ∈ powerDomains b lo hi e) :
    c.dom = cyclicInterval b (digit b (lo ^ e) c.j) (width b lo hi e c.j + 1) := by
  unfold powerDomains at hc
  split_ifs at hc
  · rw [List.mem_map] at hc
    obtain ⟨j, -, rfl⟩ := hc
    rfl
  · simp at hc

theorem mem_powerDomains {b lo hi e j : ℕ} (heq : numDigits b (lo ^ e) = numDigits b (hi ^ e))
    (hj : j < numDigits b (lo ^ e)) (hc : constrained b lo hi e j = true) :
    (⟨e, j, cyclicInterval b (digit b (lo ^ e) j) (width b lo hi e j + 1)⟩ : Constraint) ∈
      powerDomains b lo hi e := by
  unfold powerDomains
  rw [if_pos heq, List.mem_map]
  exact ⟨j, List.mem_filter.mpr ⟨List.mem_reverse.mpr (List.mem_range.mpr hj), hc⟩, rfl⟩

/-! ### Sub-ranges (MSD-9) -/

theorem numDigits_sub {b lo hi lo' hi' e : ℕ} (heq : numDigits b (lo ^ e) = numDigits b (hi ^ e))
    (h1 : lo ≤ lo') (h2 : lo' ≤ hi') (h3 : hi' ≤ hi) :
    numDigits b (lo' ^ e) = numDigits b (lo ^ e) ∧ numDigits b (hi' ^ e) = numDigits b (lo ^ e) := by
  have a := numDigits_pow_mono (b := b) e h1
  have b' := numDigits_pow_mono (b := b) e h2
  have c := numDigits_pow_mono (b := b) e h3
  omega

theorem width_sub {b lo hi lo' hi' e j : ℕ} (h1 : lo ≤ lo') (h3 : hi' ≤ hi) :
    width b lo' hi' e j ≤ width b lo hi e j := by
  unfold width
  have a := Nat.div_le_div_right (c := b ^ j) (Nat.pow_le_pow_left h1 e)
  have c := Nat.div_le_div_right (c := b ^ j) (Nat.pow_le_pow_left h3 e)
  omega

theorem constrained_sub {b lo hi lo' hi' e j : ℕ} (heq : numDigits b (lo ^ e) = numDigits b (hi ^ e))
    (h1 : lo ≤ lo') (h2 : lo' ≤ hi') (h3 : hi' ≤ hi) (hc : constrained b lo hi e j = true) :
    constrained b lo' hi' e j = true := by
  unfold constrained at *
  rw [(numDigits_sub heq h1 h2 h3).1]
  rw [List.all_eq_true] at *
  intro j' hj'
  have := hc j' hj'
  rw [decide_eq_true_iff] at *
  intro hjj
  exact lt_of_le_of_lt (width_sub h1 h3) (this hjj)

theorem cyclicInterval_sub {b u v u' v' : ℕ} (h1 : u ≤ u') (h2 : u' ≤ v') (h3 : v' ≤ v) :
    cyclicInterval b (u' % b) (v' - u' + 1) ⊆ cyclicInterval b (u % b) (v - u + 1) := by
  intro d hd
  unfold cyclicInterval at *
  rw [Finset.mem_image] at *
  obtain ⟨i, hi, rfl⟩ := hd
  rw [Finset.mem_range] at hi
  refine ⟨u' - u + i, Finset.mem_range.mpr (by omega), ?_⟩
  have : u + (u' - u + i) = u' + i := by omega
  rw [Nat.mod_add_mod, Nat.mod_add_mod, this]

/-- Every constraint of a range has a counterpart on any sub-range at the same
position with a smaller domain. -/
theorem powerDomains_sub {b lo hi lo' hi' e : ℕ} (h1 : lo ≤ lo') (h2 : lo' ≤ hi') (h3 : hi' ≤ hi) :
    ∀ c ∈ powerDomains b lo hi e,
      ∃ c' ∈ powerDomains b lo' hi' e, c'.e = c.e ∧ c'.j = c.j ∧ c'.dom ⊆ c.dom := by
  intro c hc
  have heq := powerDomains_eq hc
  have hj := powerDomains_j_lt hc
  have hcon := powerDomains_constrained hc
  have hce := powerDomains_e hc
  have hdom := powerDomains_dom hc
  obtain ⟨hl, hh⟩ := numDigits_sub heq h1 h2 h3
  refine ⟨⟨e, c.j, cyclicInterval b (digit b (lo' ^ e) c.j) (width b lo' hi' e c.j + 1)⟩,
    mem_powerDomains (by omega) (by omega) (constrained_sub heq h1 h2 h3 hcon), hce.symm, rfl, ?_⟩
  show cyclicInterval b (digit b (lo' ^ e) c.j) (width b lo' hi' e c.j + 1) ⊆ c.dom
  rw [hdom]
  unfold digit width
  apply cyclicInterval_sub
  · exact Nat.div_le_div_right (Nat.pow_le_pow_left h1 e)
  · exact Nat.div_le_div_right (Nat.pow_le_pow_left h2 e)
  · exact Nat.div_le_div_right (Nat.pow_le_pow_left h3 e)

theorem rangeDomains_sub {b lo hi lo' hi' : ℕ} (h1 : lo ≤ lo') (h2 : lo' ≤ hi') (h3 : hi' ≤ hi) :
    ∀ c ∈ rangeDomains b lo hi,
      ∃ c' ∈ rangeDomains b lo' hi', c'.e = c.e ∧ c'.j = c.j ∧ c'.dom ⊆ c.dom := by
  intro c hc
  unfold rangeDomains at *
  rw [List.mem_append] at hc
  rcases hc with hc | hc
  · obtain ⟨c', hc', h⟩ := powerDomains_sub h1 h2 h3 c hc
    exact ⟨c', List.mem_append_left _ hc', h⟩
  · obtain ⟨c', hc', h⟩ := powerDomains_sub h1 h2 h3 c hc
    exact ⟨c', List.mem_append_right _ hc', h⟩

/-- The function form of an SDR, for lists with distinct `(e, j)` pairs. -/
theorem hasSDR_iff_fun {cs : List Constraint} (hd : (cs.map fun c => (c.e, c.j)).Nodup) :
    HasSDR cs ↔ ∃ f : ℕ → ℕ → ℕ, (∀ c ∈ cs, f c.e c.j ∈ c.dom) ∧
      (cs.map fun c => f c.e c.j).Nodup := by
  constructor
  · rintro ⟨ds, hf, hnd⟩
    have hlen := hf.length_eq
    let f : ℕ → ℕ → ℕ := fun e j => ds.getD ((cs.map fun c => (c.e, c.j)).idxOf (e, j)) 0
    have hget : ∀ i (hi : i < cs.length), f cs[i].e cs[i].j = ds[i]'(hlen ▸ hi) := by
      intro i hi
      show ds.getD ((cs.map fun c => (c.e, c.j)).idxOf (cs[i].e, cs[i].j)) 0 = _
      have hi' : i < (cs.map fun c => (c.e, c.j)).length := by simpa using hi
      have hidx : (cs.map fun c => (c.e, c.j)).idxOf (cs[i].e, cs[i].j) = i := by
        have := hd.idxOf_getElem i hi'
        simpa using this
      rw [hidx, List.getD_eq_getElem _ _ (hlen ▸ hi)]
    refine ⟨f, ?_, ?_⟩
    · intro c hc
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hc
      rw [hget i hi]
      have := (List.forall₂_iff_get.mp hf).2 i hi (hlen ▸ hi)
      simpa using this
    · have : (cs.map fun c => f c.e c.j) = ds := by
        apply List.ext_getElem (by simpa using hlen)
        intro i h1 h2
        simp only [List.getElem_map]
        exact hget i (by simpa using h1)
      rw [this]
      exact hnd
  · rintro ⟨f, hmem, hnd⟩
    refine ⟨cs.map fun c => f c.e c.j, ?_, hnd⟩
    rw [List.forall₂_map_right_iff, List.forall₂_same]
    exact hmem

/-- An SDR for the finer constraints gives one for the coarser. -/
theorem hasSDR_of_sub {cs cs' : List Constraint} (hd : (cs.map fun c => (c.e, c.j)).Nodup)
    (hd' : (cs'.map fun c => (c.e, c.j)).Nodup)
    (hsub : ∀ c ∈ cs, ∃ c' ∈ cs', c'.e = c.e ∧ c'.j = c.j ∧ c'.dom ⊆ c.dom)
    (h : HasSDR cs') : HasSDR cs := by
  rw [hasSDR_iff_fun hd'] at h
  rw [hasSDR_iff_fun hd]
  obtain ⟨f, hmem, hnd⟩ := h
  refine ⟨f, ?_, ?_⟩
  · intro c hc
    obtain ⟨c', hc', he, hj, hdom⟩ := hsub c hc
    rw [← he, ← hj]
    exact hdom (hmem c' hc')
  · apply List.Nodup.map_on _ (hd.of_map _)
    intro c₁ h₁ c₂ h₂ heq
    obtain ⟨c₁', hc₁', he₁, hj₁, -⟩ := hsub c₁ h₁
    obtain ⟨c₂', hc₂', he₂, hj₂, -⟩ := hsub c₂ h₂
    have hcc : c₁' = c₂' := by
      apply List.inj_on_of_nodup_map hnd hc₁' hc₂'
      rw [he₁, hj₁, he₂, hj₂]
      exact heq
    apply List.inj_on_of_nodup_map hd h₁ h₂
    rw [← he₁, ← hj₁, ← he₂, ← hj₂, hcc]

/-- Claim MSD-9: a rejected range rejects every sub-range. -/
theorem analyzeRange_mono {b lo hi lo' hi' : ℕ} (hb : 2 ≤ b) (h1 : lo ≤ lo') (h2 : lo' ≤ hi')
    (h3 : hi' ≤ hi) (h : analyzeRange b lo hi = false) : analyzeRange b lo' hi' = false := by
  by_contra hc
  have hc' : analyzeRange b lo' hi' = true := by simpa using hc
  have hsdr' : HasSDR (rangeDomains b lo' hi') :=
    (sdrExists_iff _ (rangeDomains_dom_lt (by omega))).mp hc'
  have hsdr : HasSDR (rangeDomains b lo hi) :=
    hasSDR_of_sub (rangeDomains_nodup _ _ _) (rangeDomains_nodup _ _ _)
      (rangeDomains_sub h1 h2 h3) hsdr'
  have := (sdrExists_iff _ (rangeDomains_dom_lt (by omega))).mpr hsdr
  unfold analyzeRange at h
  rw [h] at this
  exact Bool.false_ne_true this

end Nice
