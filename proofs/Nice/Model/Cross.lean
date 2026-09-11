/-
The cross-end certificate and the niceonly pipeline. Rust:
`msd_prefix_filter::{analyze_range, get_valid_ranges_recursive_masked}`,
`stride_filter::iterate_range_masked`, `client_process::process_range_niceonly`.

A singleton domain at an output position `j ≥ k` means every `n` in the
range has that exact digit there (the certificate, `fixed_mask`). A stride
residue whose exact low digits (positions `< k`) contain such a digit
would repeat it across two distinct positions, so none of its candidates
in the range is nice (CRS-1). A certificate proved for a range holds on
its sub-ranges, so leaves inherit their ancestors' digits (CRS-2). The
pipeline is: masked subdivision, then for each leaf the stride walk with
the one-AND test and the nice check (END-1).
-/
import Nice.Model.Msd
import Nice.Model.Seeded

namespace Nice

/-! ### The certificate -/

theorem cyclicInterval_one {b x : ℕ} (hx : x < b) : cyclicInterval b x 1 = {x} := by
  unfold cyclicInterval
  rw [Finset.range_one, Finset.image_singleton, Nat.add_zero, Nat.mod_eq_of_lt hx]

theorem powerDomains_dom {b lo hi e : ℕ} {c : Constraint} (hc : c ∈ powerDomains b lo hi e) :
    c.dom = cyclicInterval b (digit b (lo ^ e) c.j) (width b lo hi e c.j + 1) := by
  unfold powerDomains at hc
  split_ifs at hc
  · rw [List.mem_map] at hc
    obtain ⟨j, -, rfl⟩ := hc
    rfl
  · simp at hc

theorem rangeDomains_dom {b lo hi : ℕ} {c : Constraint} (hc : c ∈ rangeDomains b lo hi) :
    c.dom = cyclicInterval b (digit b (lo ^ c.e) c.j) (width b lo hi c.e c.j + 1) := by
  unfold rangeDomains at hc
  rw [List.mem_append] at hc
  rcases hc with hc | hc
  · rw [powerDomains_dom hc, powerDomains_e hc]
  · rw [powerDomains_dom hc, powerDomains_e hc]

/-- `fixed_mask`: digits at singleton constrained positions `≥ k`. -/
def fixedDigits (b lo hi k : ℕ) : Finset ℕ :=
  (((rangeDomains b lo hi).filter fun c => decide (k ≤ c.j ∧ width b lo hi c.e c.j = 0)).map
    fun c => digit b (lo ^ c.e) c.j).toFinset

/-- `x` is a digit of `n²` or `n³` at some position `≥ k`. -/
def HighDigit (b k n x : ℕ) : Prop :=
  ∃ e j, (e = 2 ∨ e = 3) ∧ k ≤ j ∧ j < numDigits b (n ^ e) ∧ digit b (n ^ e) j = x

/-- Every certificate digit is a high digit of every `n` in the range. -/
theorem highDigit_of_mem_fixedDigits {b lo hi k n x : ℕ} (hb : 2 ≤ b) (hlo : lo ≤ n) (hhi : n ≤ hi)
    (hx : x ∈ fixedDigits b lo hi k) : HighDigit b k n x := by
  unfold fixedDigits at hx
  rw [List.mem_toFinset, List.mem_map] at hx
  obtain ⟨c, hc, rfl⟩ := hx
  rw [List.mem_filter, decide_eq_true_iff] at hc
  obtain ⟨hc, hk, hw⟩ := hc
  have ⟨he, hpos⟩ := rangeDomains_sound b lo hi c hc
  have ⟨hj, hmem⟩ := hpos n hlo hhi
  rw [rangeDomains_dom hc, hw, cyclicInterval_one (digit_lt_base (by omega) _ _),
    Finset.mem_singleton] at hmem
  exact ⟨c.e, c.j, he, hk, hj, hmem⟩

/-! ### CRS-1 -/

/-- A digit cannot be both a low digit (position `< k`) and a high digit
(position `≥ k`) of a nice number. -/
theorem no_nice_of_low_high {b k n x : ℕ} (hb : 2 ≤ b)
    (hk2 : k ≤ numDigits b (n ^ 2)) (hk3 : k ≤ numDigits b (n ^ 3))
    (hlow : x ∈ lowMask b k (n % strideModulus b k)) (hhigh : HighDigit b k n x) :
    ¬ IsNice b n := by
  intro h
  rw [lowMask_eq, List.mem_toFinset, List.mem_append] at hlow
  obtain ⟨e, j, he, hkj, hj, hd⟩ := hhigh
  have hnd : (outputDigits b n).Nodup := h.nodup_iff.mpr List.nodup_range
  -- the low digit: some e' ∈ {2,3}, j' < k with digit = x
  have hlow' : ∃ e' j', (e' = 2 ∨ e' = 3) ∧ j' < k ∧ digit b (n ^ e') j' = x := by
    unfold lowDigits at hlow
    rcases hlow with hl | hl <;> rw [List.mem_map] at hl <;> obtain ⟨j', hj', rfl⟩ := hl <;>
      rw [List.mem_range] at hj'
    · exact ⟨2, j', Or.inl rfl, hj', rfl⟩
    · exact ⟨3, j', Or.inr rfl, hj', rfl⟩
  obtain ⟨e', j', he', hj'k, hd'⟩ := hlow'
  have hj' : j' < numDigits b (n ^ e') := by rcases he' with rfl | rfl <;> omega
  have hi1 := outIndex_lt he hj
  have hi2 := outIndex_lt he' hj'
  have h1 := outputDigits_getD hb he hj
  have h2 := outputDigits_getD hb he' hj'
  rw [List.getD_eq_getElem _ _ hi1] at h1
  rw [List.getD_eq_getElem _ _ hi2] at h2
  have hidx : outIndex b n e j = outIndex b n e' j' := by
    have : (outputDigits b n)[outIndex b n e j] = (outputDigits b n)[outIndex b n e' j'] := by
      rw [h1, h2, hd, hd']
    exact (List.Nodup.getElem_inj_iff hnd).mp this
  have := (outIndex_inj he he' hj hj' hidx).2
  omega

/-- Claim CRS-1: a residue whose exact low digits meet the certificate has
no nice candidate in the range. -/
theorem no_nice_of_cross {b lo hi k n : ℕ} (hb : 2 ≤ b) (hlo : lo ≤ n) (hhi : n ≤ hi)
    (hk2 : k ≤ numDigits b (n ^ 2)) (hk3 : k ≤ numDigits b (n ^ 3))
    (hcross : ¬ Disjoint (lowMask b k (n % strideModulus b k)) (fixedDigits b lo hi k)) :
    ¬ IsNice b n := by
  rw [Finset.not_disjoint_iff] at hcross
  obtain ⟨x, hlow, hfix⟩ := hcross
  exact no_nice_of_low_high hb hk2 hk3 hlow (highDigit_of_mem_fixedDigits hb hlo hhi hfix)

/-! ### The masked recursion (CRS-2) -/

/-- `get_valid_ranges_recursive_masked`: leaves carry the union of every
analyzed ancestor's certificate. -/
def validRangesMasked (b k minSize : ℕ) :
    ℕ → ℕ → ℕ → Finset ℕ → List (ℕ × ℕ × Finset ℕ)
  | 0, start, stop, inh => [(start, stop, inh)]
  | d + 1, start, stop, inh =>
    if stop - start ≤ minSize then [(start, stop, inh)]
    else if analyzeRange b start (stop - 1) = false then []
    else
      let mask := inh ∪ fixedDigits b start (stop - 1) k
      if stop - start < 2 * minSize then [(start, stop, mask)]
      else
        validRangesMasked b k minSize d start (start + (stop - start) / 2) mask ++
          validRangesMasked b k minSize d (start + (stop - start) / 2) stop mask

/-- Claim CRS-2 with MSD-7: every nice `n` of the input lies in a leaf whose
mask consists of high digits of `n`. -/
theorem validRangesMasked_cover {b k minSize : ℕ} (hb : 2 ≤ b) (d : ℕ) :
    ∀ start stop inh n, start ≤ n → n < stop → IsNice b n →
      (∀ x ∈ inh, HighDigit b k n x) →
      ∃ r ∈ validRangesMasked b k minSize d start stop inh,
        r.1 ≤ n ∧ n < r.2.1 ∧ ∀ x ∈ r.2.2, HighDigit b k n x := by
  induction d with
  | zero =>
    intro start stop inh n h1 h2 _ hinh
    exact ⟨(start, stop, inh), by simp [validRangesMasked], h1, h2, hinh⟩
  | succ d ih =>
    intro start stop inh n h1 h2 hn hinh
    unfold validRangesMasked
    split_ifs with hsmall hrej hnw
    · exact ⟨(start, stop, inh), by simp, h1, h2, hinh⟩
    · exact absurd hn (no_nice_of_analyzeRange hb hrej n h1 (by omega))
    · refine ⟨(start, stop, inh ∪ fixedDigits b start (stop - 1) k), by simp, h1, h2, ?_⟩
      intro x hx
      rw [Finset.mem_union] at hx
      rcases hx with hx | hx
      · exact hinh x hx
      · exact highDigit_of_mem_fixedDigits hb h1 (by omega) hx
    · have hmask : ∀ x ∈ inh ∪ fixedDigits b start (stop - 1) k, HighDigit b k n x := by
        intro x hx
        rw [Finset.mem_union] at hx
        rcases hx with hx | hx
        · exact hinh x hx
        · exact highDigit_of_mem_fixedDigits hb h1 (by omega) hx
      rcases Nat.lt_or_ge n (start + (stop - start) / 2) with hmid | hmid
      · obtain ⟨r, hr, hr1, hr2, hr3⟩ := ih start _ _ n h1 hmid hn hmask
        exact ⟨r, List.mem_append_left _ hr, hr1, hr2, hr3⟩
      · obtain ⟨r, hr, hr1, hr2, hr3⟩ := ih _ stop _ n hmid h2 hn hmask
        exact ⟨r, List.mem_append_right _ hr, hr1, hr2, hr3⟩

theorem validRangesMasked_subset {b k minSize : ℕ} (d : ℕ) :
    ∀ start stop inh r, r ∈ validRangesMasked b k minSize d start stop inh →
      start ≤ r.1 ∧ r.2.1 ≤ stop := by
  induction d with
  | zero =>
    intro start stop inh r hr
    simp [validRangesMasked] at hr
    subst hr; exact ⟨le_refl _, le_refl _⟩
  | succ d ih =>
    intro start stop inh r hr
    unfold validRangesMasked at hr
    split_ifs at hr
    · simp at hr; subst hr; exact ⟨le_refl _, le_refl _⟩
    · simp at hr
    · simp at hr; subst hr; exact ⟨le_refl _, le_refl _⟩
    · rw [List.mem_append] at hr
      rcases hr with hr | hr
      · have := ih _ _ _ r hr; omega
      · have := ih _ _ _ r hr; omega

/-! ### END-1: the niceonly pipeline -/

/-- `process_range_niceonly`: masked subdivision, then per leaf the stride
walk, the one-AND certificate test, and the nice check (which the seeded
check implements, `seeded_iff_isNice`). -/
def processRangeNiceonly (b k minSize depth start stop : ℕ) : List ℕ :=
  (validRangesMasked b k minSize depth start stop ∅).flatMap fun r =>
    (strideCandidates b k r.1 r.2.1).filter fun n =>
      decide (Disjoint (lowMask b k (n % strideModulus b k)) r.2.2) && decide (IsNice b n)

/-- Claim END-1 (completeness): every nice number of the range is reported. -/
theorem niceonly_complete {b k minSize depth start stop n : ℕ} (hb : 6 ≤ b) (hk : k ≤ 3)
    (hs : start ≤ n) (he : n < stop) (h : IsNice b n) :
    n ∈ processRangeNiceonly b k minSize depth start stop := by
  obtain ⟨r, hr, hr1, hr2, hmask⟩ :=
    validRangesMasked_cover (b := b) (k := k) (minSize := minSize) (by omega) depth
      start stop ∅ n hs he h (by simp)
  unfold processRangeNiceonly
  rw [List.mem_flatMap]
  refine ⟨r, hr, ?_⟩
  rw [List.mem_filter, Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff]
  refine ⟨mem_strideCandidates_of_isNice hb hk hr1 hr2 h, ?_, h⟩
  have ⟨hk2, hk3⟩ := three_le_numDigits_of_inBaseRange hb (inBaseRange_of_isNice h)
  rw [Finset.disjoint_left]
  intro x hlow hhigh
  exact no_nice_of_low_high (by omega) (by omega) (by omega) hlow (hmask x hhigh) h

/-- Claim END-1 (soundness): everything reported is a nice number of the range. -/
theorem niceonly_sound {b k minSize depth start stop n : ℕ}
    (h : n ∈ processRangeNiceonly b k minSize depth start stop) :
    start ≤ n ∧ n < stop ∧ IsNice b n := by
  unfold processRangeNiceonly at h
  rw [List.mem_flatMap] at h
  obtain ⟨r, hr, hn⟩ := h
  rw [List.mem_filter, Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff] at hn
  obtain ⟨hc, -, hnice⟩ := hn
  obtain ⟨h1, h2, -⟩ := mem_strideCandidates.mp hc
  obtain ⟨h3, h4⟩ := validRangesMasked_subset depth start stop ∅ r hr
  exact ⟨by omega, by omega, hnice⟩

end Nice
