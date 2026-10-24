/-
The overlap join. Rust: `overlap_join::{Base::cert, Base::cert_floor,
Base::top_layer, Base::bot_dfs, join_range, join_slices}`, and the prefilter
that `cpu_join` and `cubecl_join` run on the join's survivors
(`test_fields::mid_mirror` is its definition).

An `L`-digit field and parameters `(t, k, p)` with `t ≤ L`, `k < L < t + k`:
the top `t` digits of `n` are a prefix `P = n / b^f0` (`f0 = L − t`), the low
`k` digits a residue `R = n mod b^k`, and the two share the `o = t + k − L`
digits at positions `f0 .. k−1`. The join pairs every certified prefix with
every residue whose low output digits are distinct, on equal shared digits
(the lowest `p` of them pick a partition, the rest a key) and on the
digit-sum class mod `b − 1`, then ANDs the two digit masks and checks what
survives.

Soundness of every rejection is the client's own reasoning (DESIGN):
- JOIN-1: the digits common to `a^j` and `e^j` above the cap, from the top
  down to the first disagreement, are digits of `n^j` for every `n` in
  `[a, e]`; a repeat among them rules the interval out.
- JOIN-2: the bottom search's digit step is AFF-1 with one digit, and it
  keeps every nice number's residue with its exact low output digits.
- JOIN-3: a nice number's certified high digits and its low digits sit at
  distinct positions, so the AND is zero.
- JOIN-4: the slices concatenate to the field.
- JOIN-5: every `n` is exactly one (prefix, residue) pair, found in partition
  `P mod b^p` under root `n mod (b − 1)`.
- JOIN-6: the prefilter tests real low digits, and the certificate only
  where every certified position is above them.
END-2 composes them: the modelled join reports exactly the nice numbers of
the field.
-/
import Nice.Model.Cross

namespace NiceSearch

/-- Digit counts are monotone. -/
theorem numDigits_mono {b x y : ℕ} (h : x ≤ y) : numDigits b x ≤ numDigits b y :=
  Nat.le_length_digits_le b x y h

theorem HighDigit.mono {b c c' n x : ℕ} (h : c ≤ c') (hx : HighDigit b c' n x) :
    HighDigit b c n x := by
  obtain ⟨e, j, he, hj, hlt, hd⟩ := hx
  exact ⟨e, j, he, le_trans h hj, hlt, hd⟩

/-! ### JOIN-1: certificates -/

/-- Positions `i ≥ c` below the length of `y` from which `x` and `y` agree all
the way up: `x / b^i = y / b^i`. `Base::cert` scans positions from the top
down and stops at the first disagreement; these are the positions it passes. -/
def certPositions (b x y c : ℕ) : List ℕ :=
  (List.range (numDigits b y)).filter fun i => decide (c ≤ i ∧ x / b ^ i = y / b ^ i)

/-- The certified digits of `[a, e]` at cap `c`: the square's, then the cube's. -/
def certDigits (b a e c : ℕ) : List ℕ :=
  (certPositions b (a ^ 2) (e ^ 2) c).map (digit b (a ^ 2)) ++
    (certPositions b (a ^ 3) (e ^ 3) c).map (digit b (a ^ 3))

/-- `Base::cert`: the certificate (a digit set) of `[a, e]` at cap `c`, or
`none` when two certified digits coincide. -/
def cert (b a e c : ℕ) : Option (Finset ℕ) :=
  if (certDigits b a e c).Nodup then some (certDigits b a e c).toFinset else none

/-- `Base::cert_floor`: no certified digit of `[a, e]` sits below this position. -/
def certFloor (b a e c : ℕ) : ℕ :=
  min (max c (numDigits b (e ^ 2 - a ^ 2))) (max c (numDigits b (e ^ 3 - a ^ 3)))

/-- A certified position is constant over the interval: every `n` in it has the
endpoint's digit there, at a real position. -/
theorem certPositions_digit {b a e c n i j : ℕ} (hb : 2 ≤ b) (han : a ≤ n) (hne : n ≤ e)
    (hi : i ∈ certPositions b (a ^ j) (e ^ j) c) :
    c ≤ i ∧ i < numDigits b (n ^ j) ∧ digit b (n ^ j) i = digit b (a ^ j) i := by
  unfold certPositions at hi
  rw [List.mem_filter, List.mem_range, decide_eq_true_iff] at hi
  obtain ⟨hlen, hc, hq⟩ := hi
  have h1 : a ^ j / b ^ i ≤ n ^ j / b ^ i := Nat.div_le_div_right (Nat.pow_le_pow_left han j)
  have h2 : n ^ j / b ^ i ≤ e ^ j / b ^ i := Nat.div_le_div_right (Nat.pow_le_pow_left hne j)
  have hnq : n ^ j / b ^ i = a ^ j / b ^ i := by omega
  have hBi : 0 < b ^ i := Nat.pow_pos (by omega)
  refine ⟨hc, ?_, ?_⟩
  · have he0 : e ^ j ≠ 0 := by
      intro h0
      rw [h0] at hlen
      simp [numDigits] at hlen
    have hbi : b ^ i ≤ e ^ j := (lt_numDigits_iff hb he0 i).mp hlen
    have hpos : 1 ≤ e ^ j / b ^ i := (Nat.one_le_div_iff hBi).mpr hbi
    have hbn : b ^ i ≤ n ^ j := (Nat.one_le_div_iff hBi).mp (by omega)
    exact (lt_numDigits_iff hb (by omega) i).mpr hbn
  · unfold digit
    rw [hnq]

/-- Certified positions sit at or above the floor. -/
theorem certFloor_le {b a e c i j : ℕ} (hb : 2 ≤ b) (hj : j = 2 ∨ j = 3)
    (hi : i ∈ certPositions b (a ^ j) (e ^ j) c) : certFloor b a e c ≤ i := by
  unfold certPositions at hi
  rw [List.mem_filter, List.mem_range, decide_eq_true_iff] at hi
  obtain ⟨-, hc, hq⟩ := hi
  have hdiff : e ^ j - a ^ j < b ^ i := by
    have h1 := Nat.div_add_mod (e ^ j) (b ^ i)
    have h2 := Nat.div_add_mod (a ^ j) (b ^ i)
    have h3 : e ^ j % b ^ i < b ^ i := Nat.mod_lt _ (Nat.pow_pos (by omega))
    rw [← hq] at h1
    generalize e ^ j % b ^ i = r₁ at h1 h3
    generalize a ^ j % b ^ i = r₂ at h2
    generalize b ^ i * (a ^ j / b ^ i) = q at h1 h2
    omega
  have hnd : numDigits b (e ^ j - a ^ j) ≤ i := by
    rcases Nat.eq_zero_or_pos (e ^ j - a ^ j) with h0 | h0
    · rw [h0]; simp [numDigits]
    · exact (numDigits_le_iff hb (by omega) i).mpr hdiff
  unfold certFloor
  rcases hj with rfl | rfl <;> omega

/-- The cap bounds the floor. -/
theorem le_certFloor (b a e c : ℕ) : c ≤ certFloor b a e c := by
  unfold certFloor; omega

/-- A certificate's digits, read off at the low endpoint, are `n`'s digits. -/
theorem certDigits_eq {b a e c n : ℕ} (hb : 2 ≤ b) (han : a ≤ n) (hne : n ≤ e) :
    certDigits b a e c =
      (certPositions b (a ^ 2) (e ^ 2) c).map (digit b (n ^ 2)) ++
        (certPositions b (a ^ 3) (e ^ 3) c).map (digit b (n ^ 3)) := by
  unfold certDigits
  congr 1 <;> apply List.map_congr_left <;> intro i hi
  · exact (certPositions_digit hb han hne hi).2.2.symm
  · exact (certPositions_digit hb han hne hi).2.2.symm

/-- Claim JOIN-1: every digit of a certificate of `[a, e]` is, for every `n` in
`[a, e]`, a digit of `n²` or `n³` at a position at or above the floor (and so
above the cap, `le_certFloor`). -/
theorem highDigit_of_cert {b a e c n : ℕ} {m : Finset ℕ} (hb : 2 ≤ b) (han : a ≤ n)
    (hne : n ≤ e) (h : cert b a e c = some m) :
    ∀ x ∈ m, HighDigit b (certFloor b a e c) n x := by
  unfold cert at h
  by_cases hnd : (certDigits b a e c).Nodup
  · rw [if_pos hnd] at h
    obtain rfl := Option.some.inj h
    intro x hx
    rw [List.mem_toFinset, certDigits_eq hb han hne, List.mem_append, List.mem_map,
      List.mem_map] at hx
    rcases hx with ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩
    · obtain ⟨-, hlt, -⟩ := certPositions_digit hb han hne hi
      exact ⟨2, i, Or.inl rfl, certFloor_le hb (Or.inl rfl) hi, hlt, rfl⟩
    · obtain ⟨-, hlt, -⟩ := certPositions_digit hb han hne hi
      exact ⟨3, i, Or.inr rfl, certFloor_le hb (Or.inr rfl) hi, hlt, rfl⟩
  · rw [if_neg hnd] at h
    cases h

/-- Claim JOIN-1: a certificate with a repeated digit rules out the interval. -/
theorem not_isNice_of_cert_none {b a e c n : ℕ} (hb : 2 ≤ b) (han : a ≤ n) (hne : n ≤ e)
    (h : cert b a e c = none) : ¬ IsNice b n := by
  intro hn
  unfold cert at h
  by_cases hnd : (certDigits b a e c).Nodup
  · rw [if_pos hnd] at h
    cases h
  apply hnd
  have hP : ∀ x y : ℕ, (certPositions b x y c).Nodup := fun x y =>
    List.nodup_range.filter _
  rw [certDigits_eq hb han hne, List.nodup_append]
  refine ⟨List.Nodup.map_on ?_ (hP _ _), List.Nodup.map_on ?_ (hP _ _), ?_⟩
  · intro i hi i' hi' heq
    by_contra hne'
    exact digit_ne_of_isNice hb hn (Or.inl rfl) (Or.inl rfl)
      (certPositions_digit hb han hne hi).2.1 (certPositions_digit hb han hne hi').2.1
      (by omega) heq
  · intro i hi i' hi' heq
    by_contra hne'
    exact digit_ne_of_isNice hb hn (Or.inr rfl) (Or.inr rfl)
      (certPositions_digit hb han hne hi).2.1 (certPositions_digit hb han hne hi').2.1
      (by omega) heq
  · intro x hx y hy hxy
    rw [List.mem_map] at hx hy
    obtain ⟨i, hi, rfl⟩ := hx
    obtain ⟨i', hi', rfl⟩ := hy
    exact digit_ne_of_isNice hb hn (Or.inl rfl) (Or.inr rfl)
      (certPositions_digit hb han hne hi).2.1 (certPositions_digit hb han hne hi').2.1
      (by omega) hxy

/-- A nice number's certificate exists. -/
theorem exists_cert_of_isNice {b a e c n : ℕ} (hb : 2 ≤ b) (han : a ≤ n) (hne : n ≤ e)
    (h : IsNice b n) : ∃ m, cert b a e c = some m := by
  cases hc : cert b a e c with
  | none => exact absurd h (not_isNice_of_cert_none hb han hne hc)
  | some m => exact ⟨m, rfl⟩

/-! ### The top layer -/

/-- One level of `Base::top_layer`: the children of each kept prefix that meet
`[s, eIncl]`, each kept when the certificate of its interval (clipped to the
field) passes. -/
def topStep (b L s eIncl c j : ℕ) (level : List (ℕ × Finset ℕ)) : List (ℕ × Finset ℕ) :=
  level.flatMap fun par =>
    (List.range b).filterMap fun d =>
      if s / b ^ (L - j) ≤ par.1 * b + d ∧ par.1 * b + d ≤ eIncl / b ^ (L - j) then
        (cert b (max ((par.1 * b + d) * b ^ (L - j)) s)
          (min ((par.1 * b + d) * b ^ (L - j) + b ^ (L - j) - 1) eIncl) c).map
          fun m => (par.1 * b + d, m)
      else none

/-- `Base::top_layer`: the prefixes of the top `j` digits of the `L`-digit
numbers of `[s, eIncl]` whose certificates pass, breadth first. -/
def topLayer (b L s eIncl c : ℕ) : ℕ → List (ℕ × Finset ℕ)
  | 0 => [(0, ∅)]
  | j + 1 => topStep b L s eIncl c (j + 1) (topLayer b L s eIncl c j)

/-- A number lies in its prefix's interval: `⌊n/w⌋·w ≤ n ≤ ⌊n/w⌋·w + w − 1`. -/
theorem div_mul_le_and_le {n w : ℕ} (hw : 0 < w) : n / w * w ≤ n ∧ n ≤ n / w * w + w - 1 := by
  have h1 := Nat.div_add_mod' n w
  have h2 := Nat.mod_lt n hw
  generalize n % w = r at h1 h2
  generalize n / w * w = q at h1 ⊢
  omega

/-- The top layer keeps every nice number's prefix, at every depth. -/
theorem mem_topLayer_of_isNice {b L s eIncl c n : ℕ} (hb : 2 ≤ b) (hs : s ≤ n)
    (he : n ≤ eIncl) (hL : n < b ^ L) (h : IsNice b n) :
    ∀ j ≤ L, ∃ m, (n / b ^ (L - j), m) ∈ topLayer b L s eIncl c j := by
  intro j
  induction j with
  | zero =>
    intro _
    refine ⟨∅, ?_⟩
    simp [topLayer, Nat.div_eq_of_lt hL]
  | succ j ih =>
    intro hj
    obtain ⟨m, hm⟩ := ih (by omega)
    have hw : 0 < b ^ (L - (j + 1)) := Nat.pow_pos (by omega)
    have hpar : n / b ^ (L - j) = n / b ^ (L - (j + 1)) / b := by
      rw [Nat.div_div_eq_div_mul, ← pow_succ]
      congr 2
      omega
    have hp : n / b ^ (L - j) * b + n / b ^ (L - (j + 1)) % b = n / b ^ (L - (j + 1)) := by
      rw [hpar]; exact Nat.div_add_mod' _ _
    obtain ⟨hlo, hhi⟩ := div_mul_le_and_le (n := n) hw
    obtain ⟨m', hm'⟩ := exists_cert_of_isNice (c := c) hb
      (a := max (n / b ^ (L - (j + 1)) * b ^ (L - (j + 1))) s)
      (e := min (n / b ^ (L - (j + 1)) * b ^ (L - (j + 1)) + b ^ (L - (j + 1)) - 1) eIncl)
      (max_le hlo hs) (le_min hhi he) h
    refine ⟨m', ?_⟩
    rw [topLayer, topStep, List.mem_flatMap]
    refine ⟨(n / b ^ (L - j), m), hm, ?_⟩
    rw [List.mem_filterMap]
    refine ⟨n / b ^ (L - (j + 1)) % b, List.mem_range.mpr (Nat.mod_lt _ (by omega)), ?_⟩
    simp only [hp]
    rw [if_pos ⟨Nat.div_le_div_right hs, Nat.div_le_div_right he⟩, hm']
    rfl

/-! ### JOIN-2: the bottom side -/

/-- The output digits at position `j` of `(r + d·b^j)²` and of its cube, the way
`Base::bot_dfs` computes them: `d²` and `d³` at `j = 0`, otherwise
`(⌊r²/b^j⌋ + 2d·(r mod b)) mod b` and `(⌊r³/b^j⌋ + 3d·(r mod b)²) mod b`. -/
def botDigits (b r j d : ℕ) : ℕ × ℕ :=
  if j = 0 then (d * d % b, d * d % b * d % b)
  else ((r ^ 2 / b ^ j % b + d * (2 * (r % b) % b)) % b,
    (r ^ 3 / b ^ j % b + d * (3 * (r % b * (r % b) % b) % b)) % b)

/-- Claim JOIN-2 (the digit step): `bot_dfs`'s formula is the digit, by AFF-1
with one digit. -/
theorem botDigits_eq {b r j d : ℕ} (hb : 0 < b) (hr : j = 0 → r = 0) :
    botDigits b r j d = (digit b ((r + d * b ^ j) ^ 2) j, digit b ((r + d * b ^ j) ^ 3) j) := by
  unfold botDigits
  split_ifs with hj
  · subst hj
    rw [hr rfl]
    simp only [pow_zero, Nat.mul_one, Nat.zero_add, digit, Nat.div_one, Prod.mk.injEq]
    constructor
    · rw [sq]
    · rw [pow_succ, sq, Nat.mod_mul_mod]
  · have h1 : 1 ≤ j := Nat.one_le_iff_ne_zero.mpr hj
    have hA := affine_mid_digit (b := b) (k := j) (m := 1) (s := r) (t := d) (i := 0) hb h1
      Nat.one_pos
    rw [Nat.add_zero, Nat.mul_comm (b ^ j) d] at hA
    rw [hA.1, hA.2]
    simp only [pow_one, digit, pow_zero, Nat.div_one, Nat.mod_mod, Prod.mk.injEq]
    have hr : r % b ≡ r [MOD b] := Nat.mod_modEq r b
    constructor
    · have h2 : 2 * (r % b) % b ≡ 2 * r [MOD b] := (Nat.mod_modEq _ b).trans (hr.mul_left 2)
      have := (Nat.mod_modEq (r ^ 2 / b ^ j) b).add (h2.mul_left d)
      rw [show r ^ 2 / b ^ j + d * (2 * r) = r ^ 2 / b ^ j + 2 * r * d by ring] at this
      exact this
    · have h3 : 3 * (r % b * (r % b) % b) % b ≡ 3 * r ^ 2 [MOD b] := by
        have hrr : r % b * (r % b) % b ≡ r * r [MOD b] := (Nat.mod_modEq _ b).trans (hr.mul hr)
        rw [sq]
        exact (Nat.mod_modEq _ b).trans (hrr.mul_left 3)
      have := (Nat.mod_modEq (r ^ 3 / b ^ j) b).add (h3.mul_left d)
      rw [show r ^ 3 / b ^ j + d * (3 * r ^ 2) = r ^ 3 / b ^ j + 3 * r ^ 2 * d by ring] at this
      exact this

/-- `Base::bot_dfs`: extend the residue `r` (digits below `j` chosen, `mask`
their output digits) one digit at a time, `fuel` more positions, keeping a
digit when its two new output digits differ from each other and from `mask`.
Positions `f0 .. f0 + pp − 1` are forced to the digits of `v`. -/
def botDfs (b f0 pp v : ℕ) : ℕ → ℕ → ℕ → Finset ℕ → List (ℕ × Finset ℕ)
  | 0, _, r, mask => [(r, mask)]
  | fuel + 1, j, r, mask =>
    (if f0 ≤ j ∧ j < f0 + pp then [v / b ^ (j - f0) % b] else List.range b).flatMap fun d =>
      if (botDigits b r j d).1 = (botDigits b r j d).2 ∨ (botDigits b r j d).1 ∈ mask ∨
          (botDigits b r j d).2 ∈ mask then []
      else botDfs b f0 pp v fuel (j + 1) (r + d * b ^ j)
        (insert (botDigits b r j d).2 (insert (botDigits b r j d).1 mask))

/-- The output digits of `n²` and `n³` below position `j`. -/
def lowSet (b n j : ℕ) : Finset ℕ :=
  (Finset.range j).image (digit b (n ^ 2)) ∪ (Finset.range j).image (digit b (n ^ 3))

theorem lowSet_zero (b n : ℕ) : lowSet b n 0 = ∅ := by simp [lowSet]

theorem lowSet_succ (b n j : ℕ) :
    lowSet b n (j + 1) = insert (digit b (n ^ 3) j) (insert (digit b (n ^ 2) j) (lowSet b n j)) := by
  ext z
  simp only [lowSet, Finset.range_add_one, Finset.image_insert, Finset.mem_insert,
    Finset.mem_union]
  tauto

/-- A low digit of `n`: some digit of `n²` or `n³` below position `j`. -/
theorem mem_lowSet {b n j x : ℕ} :
    x ∈ lowSet b n j ↔ ∃ e i, (e = 2 ∨ e = 3) ∧ i < j ∧ digit b (n ^ e) i = x := by
  unfold lowSet
  rw [Finset.mem_union, Finset.mem_image, Finset.mem_image]
  constructor
  · rintro (⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩) <;> rw [Finset.mem_range] at hi
    · exact ⟨2, i, Or.inl rfl, hi, rfl⟩
    · exact ⟨3, i, Or.inr rfl, hi, rfl⟩
  · rintro ⟨e, i, he | he, hi, rfl⟩ <;> subst he
    · exact Or.inl ⟨i, Finset.mem_range.mpr hi, rfl⟩
    · exact Or.inr ⟨i, Finset.mem_range.mpr hi, rfl⟩

/-- Claim JOIN-2: the bottom search keeps every nice number's residue, with its
exact low output digits, as long as both powers have the digits it reads and
the forced positions carry `n`'s digits. -/
theorem mem_botDfs_of_isNice {b f0 pp v n : ℕ} (hb : 2 ≤ b) (h : IsNice b n)
    (hv : ∀ i, f0 ≤ i → i < f0 + pp → v / b ^ (i - f0) % b = digit b n i) :
    ∀ fuel j, j + fuel ≤ numDigits b (n ^ 2) → j + fuel ≤ numDigits b (n ^ 3) →
      (n % b ^ (j + fuel), lowSet b n (j + fuel)) ∈
        botDfs b f0 pp v fuel j (n % b ^ j) (lowSet b n j) := by
  intro fuel
  induction fuel with
  | zero => intro j _ _; simp [botDfs]
  | succ fuel ih =>
    intro j h2 h3
    have hres : n % b ^ j + digit b n j * b ^ j = n % b ^ (j + 1) := by
      rw [Nat.mod_pow_succ]; unfold digit; ring
    have hg := botDigits_eq (b := b) (r := n % b ^ j) (j := j) (d := digit b n j) (by omega)
      (by intro hj; subst hj; rw [Nat.pow_zero, Nat.mod_one])
    rw [hres, ← digit_pow_mod_pow 2 (by omega : j < j + 1),
      ← digit_pow_mod_pow 3 (by omega : j < j + 1)] at hg
    have hne : digit b (n ^ 2) j ≠ digit b (n ^ 3) j :=
      digit_ne_of_isNice hb h (Or.inl rfl) (Or.inr rfl) (by omega) (by omega) (by omega)
    have hnew : ∀ e, (e = 2 ∨ e = 3) → digit b (n ^ e) j ∉ lowSet b n j := by
      intro e he hmem
      obtain ⟨e', i, he', hi, heq⟩ := mem_lowSet.mp hmem
      exact digit_ne_of_isNice hb h he' he (by rcases he' with rfl | rfl <;> omega)
        (by rcases he with rfl | rfl <;> omega) (by omega) heq
    rw [botDfs, List.mem_flatMap]
    refine ⟨digit b n j, ?_, ?_⟩
    · split_ifs with hf
      · rw [List.mem_singleton]; exact (hv j hf.1 hf.2).symm
      · exact List.mem_range.mpr (digit_lt_base (by omega) _ _)
    · rw [hg]
      simp only
      rw [if_neg (by
        rintro (h' | h' | h')
        · exact hne h'
        · exact hnew 2 (Or.inl rfl) h'
        · exact hnew 3 (Or.inr rfl) h'), hres, ← lowSet_succ]
      have := ih (j + 1) (by omega) (by omega)
      rwa [show j + 1 + fuel = j + (fuel + 1) by omega] at this

/-- `n`'s residue and low digits, as the bottom list holds them: the residues
mod `b^f0` from the search without forced digits, each extended to depth `k`
with the partition's digits forced. -/
def bottomList (b f0 k pp v : ℕ) : List (ℕ × Finset ℕ) :=
  (botDfs b f0 0 0 f0 0 0 ∅).flatMap fun q => botDfs b f0 pp v (k - f0) f0 q.1 q.2

theorem mem_bottomList_of_isNice {b f0 k pp v n : ℕ} (hb : 2 ≤ b) (h : IsNice b n) (hf0 : f0 ≤ k)
    (h2 : k ≤ numDigits b (n ^ 2)) (h3 : k ≤ numDigits b (n ^ 3))
    (hv : ∀ i, f0 ≤ i → i < f0 + pp → v / b ^ (i - f0) % b = digit b n i) :
    (n % b ^ k, lowSet b n k) ∈ bottomList b f0 k pp v := by
  unfold bottomList
  rw [List.mem_flatMap]
  have hpre := mem_botDfs_of_isNice (pp := 0) (v := 0) (f0 := f0) hb h
    (fun _ h1 h2 => absurd h2 (by omega)) f0 0 (by omega) (by omega)
  simp only [Nat.zero_add, pow_zero, Nat.mod_one, lowSet_zero] at hpre
  refine ⟨(n % b ^ f0, lowSet b n f0), hpre, ?_⟩
  have := mem_botDfs_of_isNice (pp := pp) (v := v) (f0 := f0) hb h hv (k - f0) f0
    (by omega) (by omega)
  rwa [show f0 + (k - f0) = k by omega] at this

/-! ### JOIN-3: the AND -/

/-- Claim JOIN-3: a nice number's high digits (positions `≥ k`) and its low
digits (positions `< k`) are disjoint, so its pair passes the AND. -/
theorem disjoint_high_lowSet {b k n : ℕ} {m : Finset ℕ} (hb : 2 ≤ b) (h : IsNice b n)
    (hk2 : k ≤ numDigits b (n ^ 2)) (hk3 : k ≤ numDigits b (n ^ 3))
    (hm : ∀ x ∈ m, HighDigit b k n x) : Disjoint m (lowSet b n k) := by
  rw [Finset.disjoint_left]
  intro x hx hlow
  obtain ⟨e, j, he, hkj, hj, hd⟩ := hm x hx
  obtain ⟨e', i, he', hi, hd'⟩ := mem_lowSet.mp hlow
  exact digit_ne_of_isNice hb h he he' hj (by rcases he' with rfl | rfl <;> omega) (by omega)
    (hd.trans hd'.symm)

/-! ### JOIN-4: slices -/

/-- `join_slices`: `[s, e)` cut, in order, into slices that end at a multiple of
`block` plus `step` (`step = block · max_prefixes`), the last at `e`. `fuel`
bounds the number of slices. -/
def joinSlices (block step : ℕ) : ℕ → ℕ → ℕ → List (ℕ × ℕ)
  | 0, _, _ => []
  | fuel + 1, s, e =>
    if s < e then
      (s, min (s / block * block + step) e) ::
        joinSlices block step fuel (min (s / block * block + step) e) e
    else []

/-- Claim JOIN-4: the slices' numbers, concatenated, are the field's, in order
(so every number is in exactly one slice). -/
theorem joinSlices_flatMap {block step : ℕ} (hblock : 0 < block) (hstep : block ≤ step) :
    ∀ fuel s e, e - s ≤ fuel →
      (joinSlices block step fuel s e).flatMap (fun sl => List.range' sl.1 (sl.2 - sl.1)) =
        List.range' s (e - s) := by
  intro fuel
  induction fuel with
  | zero =>
    intro s e h
    simp [joinSlices, show e - s = 0 by omega]
  | succ fuel ih =>
    intro s e h
    rw [joinSlices]
    split_ifs with hse
    · obtain ⟨e', he'⟩ : ∃ e', e' = min (s / block * block + step) e := ⟨_, rfl⟩
      rw [← he']
      have hd := Nat.div_add_mod' s block
      have hm := Nat.mod_lt s hblock
      have hs' : s < e' := by rw [he']; exact lt_min (by omega) hse
      have hle : e' ≤ e := by rw [he']; exact min_le_right _ _
      rw [List.flatMap_cons, ih e' e (by omega)]
      have hsplit : List.range' s (e - s) =
          List.range' s (e' - s) ++ List.range' (s + (e' - s)) (e - e') := by
        rw [List.range'_append_1]
        congr 1
        omega
      rw [hsplit, show s + (e' - s) = e' by omega]
    · simp [show e - s = 0 by omega]

/-! ### JOIN-5: pairs, partitions and the probe -/

/-- `Base::roots`: the residues `r < b − 1` with `r² + r³ ≡ b(b−1)/2 (mod b − 1)`. -/
def joinRoots (b : ℕ) : List ℕ :=
  (List.range (b - 1)).filter fun r => (r ^ 2 + r ^ 3) % (b - 1) = residueTarget b

theorem mem_joinRoots_of_isNice {b n : ℕ} (hb : 2 ≤ b) (h : IsNice b n) :
    n % (b - 1) ∈ joinRoots b := by
  have := mem_residueFilter_of_isNice hb h
  unfold residueFilter at this
  rw [Finset.mem_filter, Finset.mem_range] at this
  unfold joinRoots
  rw [List.mem_filter, List.mem_range]
  exact ⟨this.1, decide_eq_true this.2⟩

/-- `join_range` on partition value `v`: the pairs that pass the AND, as the
numbers they make, in the Rust's order (tops, then roots, then the bucket's
bottoms). A bucket holds the bottoms whose key digits (positions
`f0 + p .. k − 1`) match the top's and whose low part's digit-sum class is the
root minus the top's. -/
def joinPartition (b L t k pp s e v : ℕ) : List ℕ :=
  let w := b ^ (L - t)
  let bl := bottomList b (L - t) k pp v
  let tops := (topLayer b L s (e - 1) k (t - pp)).filterMap fun q =>
    if s / w ≤ q.1 * b ^ pp + v ∧ q.1 * b ^ pp + v ≤ (e - 1) / w then
      (cert b (max ((q.1 * b ^ pp + v) * w) s)
        (min ((q.1 * b ^ pp + v) * w + w - 1) (e - 1)) k).map fun m => (q.1 * b ^ pp + v, m)
    else none
  tops.flatMap fun top =>
    (joinRoots b).flatMap fun root =>
      (bl.filter fun q => q.1 / b ^ (L - t + pp) = top.1 / b ^ pp % b ^ (t + k - L - pp) ∧
          q.1 % w % (b - 1) = (root + (b - 1) - top.1 % (b - 1)) % (b - 1)).filterMap fun q =>
        if s ≤ top.1 * w + q.1 % w ∧ top.1 * w + q.1 % w < e ∧ Disjoint top.2 q.2 then
          some (top.1 * w + q.1 % w)
        else none

/-- Every `n` is exactly one (prefix, low part) pair: `n = P·w + r` with `r < w`
forces `P = n / w` and `r = n mod w`. -/
theorem join_pair_unique {n w P r : ℕ} (hw : 0 < w) (hr : r < w) (h : n = P * w + r) :
    P = n / w ∧ r = n % w := by
  have := (Nat.div_mod_unique (a := n) (c := r) (d := P) hw).mpr ⟨by rw [h]; ring, hr⟩
  exact ⟨this.1.symm, this.2.symm⟩

/-- The low part's digit-sum class is the root minus the prefix's: with
`w ≡ 1 (mod m)`, `n = P·w + (n mod w)` gives `n mod w ≡ n − P`. -/
theorem class_eq {m n w P : ℕ} (hm : 0 < m) (hw : w ≡ 1 [MOD m]) (hn : P * w + n % w = n) :
    n % w % m = (n % m + m - P % m) % m := by
  have hPm : P % m < m := Nat.mod_lt _ hm
  have h1 : n ≡ P + n % w [MOD m] := by
    have := (hw.mul_left P).add_right (n % w)
    rw [Nat.mul_one, hn] at this
    exact this
  have h2 : P % m + (n % m + m - P % m) ≡ P % m + n % w [MOD m] := by
    rw [Nat.add_sub_cancel' (le_trans hPm.le (Nat.le_add_left m (n % m)))]
    calc n % m + m ≡ n % m [MOD m] := by simp [Nat.ModEq]
      _ ≡ n [MOD m] := Nat.mod_modEq n m
      _ ≡ P + n % w [MOD m] := h1
      _ ≡ P % m + n % w [MOD m] := (Nat.mod_modEq P m).symm.add_right _
  exact (Nat.ModEq.add_left_cancel' (P % m) h2).symm

/-- Digits `f0 .. k − 1` of `n`, read off the residue mod `b^k`. -/
theorem mod_pow_div_pow {n b f k : ℕ} (hf : f ≤ k) : n % b ^ k / b ^ f = n / b ^ f % b ^ (k - f) := by
  rw [show b ^ k = b ^ f * b ^ (k - f) by rw [← pow_add]; congr 1; omega,
    Nat.mod_mul_right_div_self]

/-- Claim JOIN-5: the join finds every nice number of `[s, e)` in partition
`(n / b^f0) mod b^p`: its prefix's ancestor is in the top layer, its prefix's
certificate passes, its residue is in the bottom list with its low digits,
the probe of root `n mod (b − 1)` reaches its bucket, and the AND passes. -/
theorem mem_joinPartition_of_isNice {b L t k pp s e n : ℕ} (hb : 2 ≤ b) (h : IsNice b n)
    (ht : t ≤ L) (hk : k < L) (hL : L < t + k) (hpp : pp ≤ t + k - L)
    (hs : s ≤ n) (he : n < e) (hlo : b ^ (L - 1) ≤ n) (hhi : n < b ^ L) :
    n ∈ joinPartition b L t k pp s e (n / b ^ (L - t) % b ^ pp) := by
  -- the shapes: f0 = L − t, w = b^f0, the prefix P, the partition value v
  have hw : 0 < b ^ (L - t) := Nat.pow_pos (by omega)
  have hbpp : 0 < b ^ pp := Nat.pow_pos (by omega)
  have hf0k : L - t + pp ≤ k := by omega
  -- digit counts: n has L digits, so both powers have more than k
  have hnL : L ≤ numDigits b n := by
    have := (lt_numDigits_iff hb (by have := Nat.one_le_pow (L - 1) b (by omega); omega)
      (L - 1)).mpr hlo
    omega
  have hn1 : 1 ≤ n := le_trans (Nat.one_le_pow _ _ (by omega)) hlo
  have hk2 : k ≤ numDigits b (n ^ 2) :=
    le_trans (by omega) (numDigits_mono (Nat.le_self_pow (by norm_num) n))
  have hk3 : k ≤ numDigits b (n ^ 3) :=
    le_trans (by omega) (numDigits_mono (Nat.le_self_pow (by norm_num) n))
  -- the top layer holds the prefix's ancestor at depth t − p
  obtain ⟨m0, hm0⟩ := mem_topLayer_of_isNice (c := k) hb hs (by omega : n ≤ e - 1) hhi h
    (t - pp) (by omega)
  have hanc : n / b ^ (L - (t - pp)) = n / b ^ (L - t) / b ^ pp := by
    rw [Nat.div_div_eq_div_mul, ← pow_add]
    congr 2
    omega
  rw [hanc] at hm0
  have hP : n / b ^ (L - t) / b ^ pp * b ^ pp + n / b ^ (L - t) % b ^ pp = n / b ^ (L - t) :=
    Nat.div_add_mod' _ _
  -- the prefix's interval holds n, so its certificate exists
  obtain ⟨hlo', hhi'⟩ := div_mul_le_and_le (n := n) hw
  obtain ⟨tm, htm⟩ := exists_cert_of_isNice (c := k) hb
    (a := max (n / b ^ (L - t) * b ^ (L - t)) s)
    (e := min (n / b ^ (L - t) * b ^ (L - t) + b ^ (L - t) - 1) (e - 1))
    (max_le hlo' hs) (le_min hhi' (by omega)) h
  have htmk : ∀ x ∈ tm, HighDigit b k n x := fun x hx =>
    HighDigit.mono (le_certFloor _ _ _ _) (highDigit_of_cert hb (max_le hlo' hs)
      (le_min hhi' (by omega)) htm x hx)
  -- the bottom list holds n's residue
  have hv : ∀ i, L - t ≤ i → i < L - t + pp →
      n / b ^ (L - t) % b ^ pp / b ^ (i - (L - t)) % b = digit b n i := by
    intro i hi1 hi2
    have := digit_mod_pow (b := b) (n := n / b ^ (L - t)) (k := pp) (j := i - (L - t)) (by omega)
    rw [digit_div_pow, show L - t + (i - (L - t)) = i by omega] at this
    exact this
  have hbot := mem_bottomList_of_isNice (k := k) (pp := pp) hb h (by omega) hk2 hk3 hv
  -- the bucket: key digits and digit-sum class
  have hkey : n % b ^ k / b ^ (L - t + pp) =
      n / b ^ (L - t) / b ^ pp % b ^ (t + k - L - pp) := by
    rw [mod_pow_div_pow hf0k, Nat.div_div_eq_div_mul, ← pow_add]
    congr 2
    omega
  have hlow : n % b ^ k % b ^ (L - t) = n % b ^ (L - t) :=
    Nat.mod_mod_of_dvd n (Nat.pow_dvd_pow b (by omega))
  have hwm : b ^ (L - t) ≡ 1 [MOD b - 1] := by
    have hb1 : b ≡ 1 [MOD b - 1] :=
      ((Nat.modEq_iff_dvd' (show 1 ≤ b by omega)).mpr (dvd_refl _)).symm
    simpa using hb1.pow (L - t)
  have hclass := class_eq (m := b - 1) (by omega) hwm (Nat.div_add_mod' n (b ^ (L - t)))
  have hroot := mem_joinRoots_of_isNice hb h
  have hdisj := disjoint_high_lowSet hb h hk2 hk3 htmk
  have hn' : n / b ^ (L - t) * b ^ (L - t) + n % b ^ (L - t) = n := Nat.div_add_mod' _ _
  -- assemble
  unfold joinPartition
  simp only []
  rw [List.mem_flatMap]
  refine ⟨(n / b ^ (L - t), tm), ?_, ?_⟩
  · rw [List.mem_filterMap]
    refine ⟨(n / b ^ (L - t) / b ^ pp, m0), hm0, ?_⟩
    simp only [hP]
    rw [if_pos ⟨Nat.div_le_div_right hs, Nat.div_le_div_right (by omega : n ≤ e - 1)⟩, htm]
    rfl
  · rw [List.mem_flatMap]
    refine ⟨n % (b - 1), hroot, ?_⟩
    rw [List.mem_filterMap]
    refine ⟨(n % b ^ k, lowSet b n k), ?_, ?_⟩
    · simp only [List.mem_filter, decide_eq_true_eq]
      exact ⟨hbot, hkey, by rw [hlow]; exact hclass⟩
    · simp only [hlow, hn']
      rw [if_pos ⟨hs, he, hdisj⟩]

/-- What the join emits lies in the field. -/
theorem mem_joinPartition {b L t k pp s e v n : ℕ} (h : n ∈ joinPartition b L t k pp s e v) :
    s ≤ n ∧ n < e := by
  unfold joinPartition at h
  simp only [] at h
  rw [List.mem_flatMap] at h
  obtain ⟨top, -, h⟩ := h
  rw [List.mem_flatMap] at h
  obtain ⟨root, -, h⟩ := h
  rw [List.mem_filterMap] at h
  obtain ⟨q, -, h⟩ := h
  split_ifs at h with hc
  cases h
  exact ⟨hc.1, hc.2.1⟩

/-! ### JOIN-6: the prefilter -/

/-- The difference of two squares (cubes) a fixed width apart grows with the base point. -/
theorem sq_diff_mono {a a' d : ℕ} (h : a ≤ a') : (a + d) ^ 2 - a ^ 2 ≤ (a' + d) ^ 2 - a' ^ 2 := by
  have e1 : (a + d) ^ 2 = a ^ 2 + (2 * a * d + d ^ 2) := by ring
  have e2 : (a' + d) ^ 2 = a' ^ 2 + (2 * a' * d + d ^ 2) := by ring
  rw [e1, e2, Nat.add_sub_cancel_left, Nat.add_sub_cancel_left]
  have : 2 * a * d ≤ 2 * a' * d := Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ h)
  omega

theorem cu_diff_mono {a a' d : ℕ} (h : a ≤ a') : (a + d) ^ 3 - a ^ 3 ≤ (a' + d) ^ 3 - a' ^ 3 := by
  have e1 : (a + d) ^ 3 = a ^ 3 + (3 * a ^ 2 * d + 3 * a * d ^ 2 + d ^ 3) := by ring
  have e2 : (a' + d) ^ 3 = a' ^ 3 + (3 * a' ^ 2 * d + 3 * a' * d ^ 2 + d ^ 3) := by ring
  rw [e1, e2, Nat.add_sub_cancel_left, Nat.add_sub_cancel_left]
  have h1 : 3 * a ^ 2 * d ≤ 3 * a' ^ 2 * d :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (Nat.pow_le_pow_left h 2))
  have h2 : 3 * a * d ^ 2 ≤ 3 * a' * d ^ 2 := Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ h)
  omega

/-- The floor of a whole block grows with the block ("monotone in `P`", which is
why `FieldSetup` can use the first whole block's floor for all of them). -/
theorem certFloor_block_mono {b w k f p : ℕ} (hw : 0 < w) (hfp : f ≤ p) :
    certFloor b (f * w) (f * w + w - 1) k ≤ certFloor b (p * w) (p * w + w - 1) k := by
  unfold certFloor
  rw [show f * w + w - 1 = f * w + (w - 1) by omega, show p * w + w - 1 = p * w + (w - 1) by omega]
  have hfw : f * w ≤ p * w := Nat.mul_le_mul_right _ hfp
  have h2 := numDigits_mono (b := b) (sq_diff_mono (d := w - 1) hfw)
  have h3 := numDigits_mono (b := b) (cu_diff_mono (d := w - 1) hfw)
  omega

/-- `FieldSetup::full_floor`: the floor of the field's first whole block of
width `w`, or `0` if no whole block fits. -/
def fullFloor (b s e w k : ℕ) : ℕ :=
  if (s + w - 1) / w * w + w - 1 < e then
    certFloor b ((s + w - 1) / w * w) ((s + w - 1) / w * w + w - 1) k
  else 0

/-- The floor the prefilter compares with `k2` for `n`'s block of width
`b^(L − t)`: the field's `full_floor` for a whole block, the block's own
`cert_floor` for one the field cuts. -/
def prefilterFloor (b L t k s e n : ℕ) : ℕ :=
  if max (n / b ^ (L - t) * b ^ (L - t)) s = n / b ^ (L - t) * b ^ (L - t) ∧
      min (n / b ^ (L - t) * b ^ (L - t) + b ^ (L - t) - 1) (e - 1) =
        n / b ^ (L - t) * b ^ (L - t) + b ^ (L - t) - 1 then
    fullFloor b s e (b ^ (L - t)) k
  else certFloor b (max (n / b ^ (L - t) * b ^ (L - t)) s)
    (min (n / b ^ (L - t) * b ^ (L - t) + b ^ (L - t) - 1) (e - 1)) k

/-- The prefilter (`test_fields::mid_mirror`; `CpuJoin::prefilter` and the GPU
stage compute it): the output digits of `n²` and `n³` at positions
`0 .. k2 − 1` are distinct, and none is in the certificate of `n`'s block when
the floor the Rust uses for that block is at least `k2`. -/
def prefilterAt (b L t k k2 s e n : ℕ) : Bool :=
  decide (suffixDigits b k2 (n % b ^ k2)).Nodup &&
    (!decide (k2 ≤ prefilterFloor b L t k s e n) ||
      (suffixDigits b k2 (n % b ^ k2)).all fun d =>
        decide (d ∉ (cert b (max (n / b ^ (L - t) * b ^ (L - t)) s)
          (min (n / b ^ (L - t) * b ^ (L - t) + b ^ (L - t) - 1) (e - 1)) k).getD ∅))

/-- The floor the prefilter uses never exceeds the block's own. -/
theorem prefilter_floor_le {b L t k s e n : ℕ} (hb : 1 ≤ b) :
    prefilterFloor b L t k s e n ≤ certFloor b (max (n / b ^ (L - t) * b ^ (L - t)) s)
      (min (n / b ^ (L - t) * b ^ (L - t) + b ^ (L - t) - 1) (e - 1)) k := by
  have hw : 0 < b ^ (L - t) := Nat.pow_pos (by omega)
  unfold prefilterFloor
  split_ifs with hwhole
  · obtain ⟨ha, hee⟩ := hwhole
    rw [ha, hee]
    unfold fullFloor
    split_ifs
    · apply certFloor_block_mono hw
      -- the first whole block is at or before `n`'s: `⌈s / w⌉ ≤ n / w`
      have hsw : s ≤ n / b ^ (L - t) * b ^ (L - t) := by
        rw [← ha]
        exact le_max_right _ _
      have : (s + b ^ (L - t) - 1) / b ^ (L - t) < n / b ^ (L - t) + 1 := by
        rw [Nat.div_lt_iff_lt_mul hw, add_mul, one_mul]
        omega
      omega
    · exact Nat.zero_le _
  · exact le_refl _

/-- Claim JOIN-6: the prefilter keeps a nice number of the field whose powers
have at least `k2` digits. -/
theorem prefilterAt_of_isNice {b L t k k2 s e n : ℕ} (hb : 2 ≤ b) (h : IsNice b n)
    (hs : s ≤ n) (he : n < e) (h2 : k2 ≤ numDigits b (n ^ 2)) (h3 : k2 ≤ numDigits b (n ^ 3)) :
    prefilterAt b L t k k2 s e n = true := by
  have hw : 0 < b ^ (L - t) := Nat.pow_pos (by omega)
  obtain ⟨hlo', hhi'⟩ := div_mul_le_and_le (n := n) hw
  have ha : max (n / b ^ (L - t) * b ^ (L - t)) s ≤ n := max_le hlo' hs
  have hee : n ≤ min (n / b ^ (L - t) * b ^ (L - t) + b ^ (L - t) - 1) (e - 1) :=
    le_min hhi' (by omega)
  obtain ⟨tm, htm⟩ := exists_cert_of_isNice (c := k) hb ha hee h
  have hfl := prefilter_floor_le (b := b) (L := L) (t := t) (k := k) (s := s) (e := e) (n := n)
    (by omega)
  -- the low digits are distinct
  have hsuf : suffixDigits b k2 (n % b ^ k2) = lowDigits b k2 (n ^ 2) ++ lowDigits b k2 (n ^ 3) :=
    suffixDigits_mod
  have hnd : (suffixDigits b k2 (n % b ^ k2)).Nodup := by
    have := mem_lsdBitmap_of_isNice hb h2 h3 h
    unfold lsdBitmap at this
    rw [Finset.mem_filter] at this
    exact this.2
  unfold prefilterAt
  rw [htm, Option.getD_some]
  simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, Bool.not_eq_true',
    decide_eq_false_iff_not, not_le, List.all_eq_true]
  refine ⟨hnd, ?_⟩
  by_cases hfloor : k2 ≤ prefilterFloor b L t k s e n
  · right
    intro d hd hdm
    -- `d` sits below `k2`; the certificate's digits sit at or above the floor
    obtain ⟨e', j', he', hj', hlt', hd'⟩ := highDigit_of_cert hb ha hee htm d hdm
    rw [hsuf, List.mem_append] at hd
    unfold lowDigits at hd
    rcases hd with hd | hd <;> rw [List.mem_map] at hd <;> obtain ⟨i, hi, rfl⟩ := hd <;>
      rw [List.mem_range] at hi
    · exact digit_ne_of_isNice hb h (Or.inl rfl) he' (by omega) hlt' (by omega) hd'.symm
    · exact digit_ne_of_isNice hb h (Or.inr rfl) he' (by omega) hlt' (by omega) hd'.symm
  · left
    omega

/-! ### END-2: the overlap join end to end -/

/-- The overlap join over a field cut into `slices`: in every slice, every
partition's join survivors, through the prefilter and the full check. -/
def joinField (b L t k pp k2 : ℕ) (slices : List (ℕ × ℕ)) : List ℕ :=
  slices.flatMap fun sl =>
    (List.range (b ^ pp)).flatMap fun v =>
      (joinPartition b L t k pp sl.1 sl.2 v).filter fun n =>
        prefilterAt b L t k k2 sl.1 sl.2 n && decide (IsNice b n)

/-- Claim END-2 (completeness): the modelled overlap join reports every nice
number of a field of `L`-digit numbers, for shapes `t ≤ L`, `k < L < t + k`,
`p ≤ t + k − L`, a prefilter depth both powers reach, and slices that
concatenate to the field (`joinSlices_flatMap`). -/
theorem joinField_complete {b L t k pp k2 s e n : ℕ} {slices : List (ℕ × ℕ)} (hb : 2 ≤ b)
    (ht : t ≤ L) (hk : k < L) (hL : L < t + k) (hpp : pp ≤ t + k - L)
    (hlo : b ^ (L - 1) ≤ s) (hhi : e ≤ b ^ L) (hk2 : k2 ≤ numDigits b (s ^ 2))
    (hsl : (slices.flatMap fun sl => List.range' sl.1 (sl.2 - sl.1)) = List.range' s (e - s))
    (hs : s ≤ n) (he : n < e) (h : IsNice b n) : n ∈ joinField b L t k pp k2 slices := by
  have hn : n ∈ List.range' s (e - s) := List.mem_range'_1.mpr ⟨hs, by omega⟩
  rw [← hsl, List.mem_flatMap] at hn
  obtain ⟨sl, hsl', hn⟩ := hn
  rw [List.mem_range'_1] at hn
  have hn2 : k2 ≤ numDigits b (n ^ 2) := le_trans hk2 (numDigits_pow_mono 2 hs)
  have hn1 : 1 ≤ n := le_trans (Nat.one_le_pow _ _ (by omega)) (le_trans hlo hs)
  have hn3 : k2 ≤ numDigits b (n ^ 3) :=
    le_trans hn2 (numDigits_mono (Nat.pow_le_pow_right hn1 (by norm_num)))
  unfold joinField
  rw [List.mem_flatMap]
  refine ⟨sl, hsl', ?_⟩
  rw [List.mem_flatMap]
  refine ⟨n / b ^ (L - t) % b ^ pp, List.mem_range.mpr (Nat.mod_lt _ (Nat.pow_pos (by omega))), ?_⟩
  simp only [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨mem_joinPartition_of_isNice hb h ht hk hL hpp hn.1 (by omega) (le_trans hlo hs)
    (by omega), prefilterAt_of_isNice hb h hn.1 (by omega) hn2 hn3, h⟩

/-- Claim END-2 (soundness): everything the modelled join reports is a nice
number of the field. -/
theorem joinField_sound {b L t k pp k2 s e n : ℕ} {slices : List (ℕ × ℕ)}
    (hsl : (slices.flatMap fun sl => List.range' sl.1 (sl.2 - sl.1)) = List.range' s (e - s))
    (h : n ∈ joinField b L t k pp k2 slices) : s ≤ n ∧ n < e ∧ IsNice b n := by
  unfold joinField at h
  rw [List.mem_flatMap] at h
  obtain ⟨sl, hsl', h⟩ := h
  rw [List.mem_flatMap] at h
  obtain ⟨v, -, h⟩ := h
  simp only [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hmem, -, hnice⟩ := h
  have hin := mem_joinPartition hmem
  have : n ∈ List.range' s (e - s) := by
    rw [← hsl, List.mem_flatMap]
    exact ⟨sl, hsl', List.mem_range'_1.mpr ⟨hin.1, by omega⟩⟩
  rw [List.mem_range'_1] at this
  exact ⟨this.1, by omega, hnice⟩

/-- END-2 with the client's slicing (`join_slices` at blocks of
`b^(L − t + p)`, at most `max_prefixes` prefixes a slice). -/
theorem joinField_slices_complete {b L t k pp k2 s e n maxPrefixes : ℕ} (hb : 2 ≤ b)
    (ht : t ≤ L) (hk : k < L) (hL : L < t + k) (hpp : pp ≤ t + k - L)
    (hlo : b ^ (L - 1) ≤ s) (hhi : e ≤ b ^ L) (hk2 : k2 ≤ numDigits b (s ^ 2))
    (hs : s ≤ n) (he : n < e) (h : IsNice b n) :
    n ∈ joinField b L t k pp k2
      (joinSlices (b ^ (L - t + pp)) (b ^ (L - t + pp) * max maxPrefixes 1) (e - s) s e) :=
  joinField_complete hb ht hk hL hpp hlo hhi hk2
    (joinSlices_flatMap (Nat.pow_pos (by omega))
      (Nat.le_mul_of_pos_right _ (by omega)) (e - s) s e (le_refl _)) hs he h

end NiceSearch
