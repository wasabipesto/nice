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

/-! ### MSD-8: the common-prefix path -/

/-- Claim MSD-8: the over-64 prefix check is the singleton-domain case. Two
distinct constrained positions whose domains are the same singleton (a
repeated digit inside one power's common prefix, or a digit shared by the
two prefixes) rule out every `n` in the range. -/
theorem no_nice_of_equal_singletons {b lo hi n e j e' j' : ℕ} (hb : 2 ≤ b)
    (hlo : lo ≤ n) (hhi : n ≤ hi) (he : e = 2 ∨ e = 3) (he' : e' = 2 ∨ e' = 3)
    (hne : e ≠ e' ∨ j ≠ j') (hj : j < numDigits b (lo ^ e)) (hj' : j' < numDigits b (lo ^ e'))
    (hw : width b lo hi e j = 0) (hw' : width b lo hi e' j' = 0)
    (hd : digit b (lo ^ e) j = digit b (lo ^ e') j') : ¬ IsNice b n := by
  intro h
  have hnd : (outputDigits b n).Nodup := h.nodup_iff.mpr List.nodup_range
  have hjn : j < numDigits b (n ^ e) := lt_of_lt_of_le hj (numDigits_pow_mono e hlo)
  have hjn' : j' < numDigits b (n ^ e') := lt_of_lt_of_le hj' (numDigits_pow_mono e' hlo)
  -- both digits of n equal the low endpoint's digit at their positions
  have hsing : ∀ {e j}, width b lo hi e j = 0 → digit b (n ^ e) j = digit b (lo ^ e) j := by
    intro e j hw
    have := digit_mem_cyclicInterval (b := b) (e := e) (j := j) hlo hhi
    unfold width at hw
    rw [hw, cyclicInterval_one (digit_lt_base (by omega) _ _), Finset.mem_singleton] at this
    exact this
  have hi1 := outIndex_lt he hjn
  have hi2 := outIndex_lt he' hjn'
  have h1 := outputDigits_getD hb he hjn
  have h2 := outputDigits_getD hb he' hjn'
  rw [List.getD_eq_getElem _ _ hi1] at h1
  rw [List.getD_eq_getElem _ _ hi2] at h2
  have hidx : outIndex b n e j = outIndex b n e' j' := by
    have : (outputDigits b n)[outIndex b n e j] = (outputDigits b n)[outIndex b n e' j'] := by
      rw [h1, h2, hsing hw, hsing hw', hd]
    exact (List.Nodup.getElem_inj_iff hnd).mp this
  obtain ⟨hee, hjj⟩ := outIndex_inj he he' hjn hjn' hidx
  rcases hne with hne | hne <;> exact hne (by assumption)

/-- Certificates grow on sub-ranges: a singleton position of a range is a
singleton with the same digit on every sub-range. -/
theorem fixedDigits_sub {b lo hi lo' hi' k : ℕ} (h1 : lo ≤ lo') (h2 : lo' ≤ hi') (h3 : hi' ≤ hi) :
    fixedDigits b lo hi k ⊆ fixedDigits b lo' hi' k := by
  intro d hd
  unfold fixedDigits at *
  rw [List.mem_toFinset, List.mem_map] at *
  obtain ⟨c, hc, rfl⟩ := hd
  rw [List.mem_filter, decide_eq_true_iff] at hc
  obtain ⟨hc, hk, hw⟩ := hc
  obtain ⟨c', hc', he, hj, -⟩ := rangeDomains_sub h1 h2 h3 c hc
  have hw' : width b lo' hi' c'.e c'.j = 0 := by
    rw [he, hj]
    have := width_sub (b := b) (e := c.e) (j := c.j) h1 h3
    omega
  refine ⟨c', List.mem_filter.mpr ⟨hc', decide_eq_true_iff.mpr ⟨hj ▸ hk, hw'⟩⟩, ?_⟩
  -- width zero: the quotient is constant on [lo, hi], so the digit agrees
  rw [he, hj]
  unfold width at hw
  unfold digit
  have a := Nat.div_le_div_right (c := b ^ c.j) (Nat.pow_le_pow_left h1 c.e)
  have b' := Nat.div_le_div_right (c := b ^ c.j) (Nat.pow_le_pow_left h2 c.e)
  have c'' := Nat.div_le_div_right (c := b ^ c.j) (Nat.pow_le_pow_left h3 c.e)
  have : lo' ^ c.e / b ^ c.j = lo ^ c.e / b ^ c.j := by omega
  rw [this]

end Nice
