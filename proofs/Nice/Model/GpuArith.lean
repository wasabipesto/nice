/-
GPU index arithmetic. Rust: `gpu_niceonly.rs` (block tiling, lane tiling,
chunked Horner), `vulkan/codegen.rs` (split16), `cuda/nice_kernels.cu`
(`mod_m`), `gpu_config.rs` (prefilter gate).

Each kernel replaces one wide operation by a few narrow ones; each lemma
here says the replacement is exact and stays inside its word size.
-/
import Nice.Model.Lsd
import Mathlib.Tactic

namespace Nice

/-! ### GPU-4: lane tiling -/

/-- With `L` lanes, lane `l < L` takes ordinals `g0 + l + t·L`; every ordinal
at or after `g0` belongs to exactly one `(lane, step)`. -/
theorem lane_partition {L g0 g : ℕ} (hL : 0 < L) (hg : g0 ≤ g) :
    ∃! p : ℕ × ℕ, p.1 < L ∧ g = g0 + p.1 + p.2 * L := by
  refine ⟨((g - g0) % L, (g - g0) / L), ⟨Nat.mod_lt _ hL, ?_⟩, ?_⟩
  · have := Nat.div_add_mod (g - g0) L
    rw [Nat.mul_comm] at this
    dsimp only
    omega
  · rintro ⟨l, t⟩ ⟨hl, hg'⟩
    dsimp only at hl hg'
    simp only [Prod.mk.injEq]
    have h1 : g - g0 = l + t * L := by omega
    constructor
    · rw [h1, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]
    · rw [h1, Nat.add_mul_div_right _ _ hL, Nat.div_eq_of_lt hl, Nat.zero_add]

/-! ### GPU-9: `mod_m` -/

/-- Reducing a two-word number: `(hi·2^64 + lo) mod M` from the two halves'
residues and `2^64 mod M`. (The `u64` overflow bound is `Const.mod_m_bound`.) -/
theorem mod_m_split (M lo hi : ℕ) :
    (hi * 2 ^ 64 + lo) % M = (hi % M * (2 ^ 64 % M) + lo % M) % M := by
  conv_lhs => rw [Nat.add_mod, Nat.mul_mod]
  rw [Nat.mod_add_mod]

/-! ### GPU-7: chunked Horner -/

/-- One Horner step stays inside `u32` while `M ≤ 2^(32-c)`. -/
theorem horner_step_lt {M c acc ch : ℕ} (hc : c ≤ 32) (hM : M ≤ 2 ^ (32 - c))
    (hacc : acc < M) (hch : ch < 2 ^ c) : acc * 2 ^ c + ch < 2 ^ 32 := by
  have h1 : (acc + 1) * 2 ^ c ≤ M * 2 ^ c := Nat.mul_le_mul_right _ hacc
  have h2 : M * 2 ^ c ≤ 2 ^ (32 - c) * 2 ^ c := Nat.mul_le_mul_right _ hM
  rw [← pow_add, Nat.sub_add_cancel hc] at h2
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Horner over chunks, most significant first, reducing mod `M` at each step. -/
def hornerMod (M B : ℕ) (chunks : List ℕ) : ℕ :=
  chunks.foldl (fun acc ch => (acc * B + ch) % M) 0

theorem hornerMod_foldl {M : ℕ} (hM : 0 < M) (B : ℕ) :
    ∀ (chunks : List ℕ) (acc : ℕ), acc < M →
      chunks.foldl (fun acc ch => (acc * B + ch) % M) acc =
        (acc * B ^ chunks.length + Nat.ofDigits B chunks.reverse) % M := by
  intro chunks
  induction chunks with
  | nil => intro acc hacc; simp [Nat.mod_eq_of_lt hacc]
  | cons ch rest ih =>
    intro acc _
    rw [List.foldl_cons, ih _ (Nat.mod_lt _ hM), List.reverse_cons, Nat.ofDigits_append,
      List.length_reverse,
      Nat.ofDigits_singleton, List.length_cons]
    -- ((acc·B + ch) % M) · B^len + v ≡ (acc·B + ch) · B^len + v
    have key : ((acc * B + ch) % M * B ^ rest.length + Nat.ofDigits B rest.reverse) % M =
        ((acc * B + ch) * B ^ rest.length + Nat.ofDigits B rest.reverse) % M :=
      ((Nat.mod_modEq _ _).mul_right _).add_right _
    rw [key]
    congr 1
    ring

/-- The base-`B` chunks of `x`, most significant first, `n` of them (zero-padded). -/
def chunksBE (B n x : ℕ) : List ℕ := (lowDigits B n x).reverse

theorem ofDigits_lowDigits (B x : ℕ) : ∀ n, Nat.ofDigits B (lowDigits B n x) = x % B ^ n := by
  intro n
  induction n with
  | zero => simp only [lowDigits, List.range_zero, List.map_nil, Nat.ofDigits_nil, pow_zero,
      Nat.mod_one]
  | succ n ih =>
    unfold lowDigits at ih ⊢
    rw [List.range_succ, List.map_append, List.map_singleton, Nat.ofDigits_append, ih,
      Nat.ofDigits_singleton, List.length_map, List.length_range, Nat.mod_pow_succ]
    rfl

/-- The chunked Horner reduction of `x` is `x mod M` whenever `x < B^n`. -/
theorem hornerMod_chunksBE {M B n x : ℕ} (hM : 0 < M) (hx : x < B ^ n) :
    hornerMod M B (chunksBE B n x) = x % M := by
  unfold hornerMod chunksBE
  rw [hornerMod_foldl hM _ _ _ hM, List.reverse_reverse, ofDigits_lowDigits, Nat.mod_eq_of_lt hx]
  simp

/-! ### GPU-5: `split16` -/

/-- Dividing the 48-bit value `rem·2^32 + v` by `d < 2^16` (with `rem < d`)
through two 32-bit-safe steps is exact: the quotient is `q1·2^16 + q2`, the
remainder is `c2 mod d`, and every intermediate fits a `u32`. -/
theorem split16_exact {d rem v c1 q1 c2 q2 : ℕ} (hd : d < 2 ^ 16) (hrem : rem < d)
    (hv : v < 2 ^ 32) (hc1 : c1 = rem * 2 ^ 16 + v / 2 ^ 16) (hq1 : q1 = c1 / d)
    (hc2 : c2 = c1 % d * 2 ^ 16 + v % 2 ^ 16) (hq2 : q2 = c2 / d) :
    (rem * 2 ^ 32 + v) / d = q1 * 2 ^ 16 + q2 ∧ (rem * 2 ^ 32 + v) % d = c2 % d ∧
      q2 < 2 ^ 16 ∧ c1 < 2 ^ 32 ∧ c2 < 2 ^ 32 := by
  have hdpos : 0 < d := by omega
  have e1 := Nat.div_add_mod c1 d
  have e2 := Nat.div_add_mod c2 d
  have e3 := Nat.div_add_mod v (2 ^ 16)
  have hr1 := Nat.mod_lt c1 hdpos
  have hr2 := Nat.mod_lt c2 hdpos
  have hvlo := Nat.mod_lt v (by norm_num : 0 < 2 ^ 16)
  have hvhi : v / 2 ^ 16 < 2 ^ 16 := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num)]; norm_num; exact hv
  -- the exact decomposition x = d·Q + r2
  have hx : rem * 2 ^ 32 + v = d * (q1 * 2 ^ 16 + q2) + c2 % d := by
    subst hq1 hq2
    zify at e1 e2 e3 hc1 hc2 ⊢
    linear_combination (-(2 ^ 16 : ℤ)) * e1 - e2 - e3 - (2 ^ 16 : ℤ) * hc1 - hc2
  have hc1lt : c1 < 2 ^ 32 := by
    have : rem * 2 ^ 16 ≤ (2 ^ 16 - 2) * 2 ^ 16 := Nat.mul_le_mul_right _ (by omega)
    omega
  have hc2lt : c2 < 2 ^ 32 := by
    have : c1 % d * 2 ^ 16 ≤ (2 ^ 16 - 2) * 2 ^ 16 := Nat.mul_le_mul_right _ (by omega)
    omega
  have hq2lt : q2 < 2 ^ 16 := by
    rw [hq2, Nat.div_lt_iff_lt_mul hdpos]
    have : c1 % d * 2 ^ 16 ≤ (d - 1) * 2 ^ 16 := Nat.mul_le_mul_right _ (by omega)
    have : (d - 1) * 2 ^ 16 + 2 ^ 16 = d * 2 ^ 16 := by
      rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hdpos]
    omega
  refine ⟨?_, ?_, hq2lt, hc1lt, hc2lt⟩
  · rw [hx, Nat.mul_add_div hdpos, Nat.div_eq_of_lt hr2, Nat.add_zero]
  · rw [hx, Nat.mul_add_mod, Nat.mod_mod]

/-! ### GPU-8: the prefilter gate -/

/-- The prefilter is the LSD bitmap at depth `p` (`mem_lsdBitmap_of_isNice`),
sound only where both powers really have `p` digits. Where neither does,
the zero-padded extraction produces two phantom zeros and rejects every
candidate — the v3.2.14 failure mode. -/
theorem prefilter_rejects_all_of_short {b p n : ℕ} (hb : 2 ≤ b) (hp : 0 < p)
    (h2 : numDigits b (n ^ 2) < p) (h3 : numDigits b (n ^ 3) < p) :
    n % b ^ p ∉ lsdBitmap b p := by
  unfold lsdBitmap
  rw [Finset.mem_filter, not_and]
  intro _
  rw [suffixDigits_mod, List.nodup_append]
  rintro ⟨-, -, hdisj⟩
  have hz2 : 0 ∈ lowDigits b p (n ^ 2) := by
    unfold lowDigits
    rw [List.mem_map]
    exact ⟨p - 1, List.mem_range.mpr (by omega), digit_eq_zero_of_le hb (by omega)⟩
  have hz3 : 0 ∈ lowDigits b p (n ^ 3) := by
    unfold lowDigits
    rw [List.mem_map]
    exact ⟨p - 1, List.mem_range.mpr (by omega), digit_eq_zero_of_le hb (by omega)⟩
  exact hdisj 0 hz2 0 hz3 rfl

/-! ### GPU-1: block tiling -/

/-- The descending powers of two covering a remainder below 64 chunks. -/
def powersOfTwoBlocks (r : ℕ) : List ℕ := [32, 16, 8, 4, 2, 1].filter fun p => r &&& p ≠ 0

theorem powersOfTwoBlocks_sum : ∀ r < 64, (powersOfTwoBlocks r).sum = r := by decide

/-- `BlockTiling::new`: block lengths in numbers, given the field size and
the chunk size `C`: blocks of 64 chunks, then descending powers of two,
then the partial chunk. -/
def blockLens (size C : ℕ) : List ℕ :=
  List.replicate (size / C / 64) (64 * C) ++ (powersOfTwoBlocks (size / C % 64)).map (· * C) ++
    (if size % C = 0 then [] else [size % C])

theorem blockLens_sum (C size : ℕ) : (blockLens size C).sum = size := by
  unfold blockLens
  rw [List.sum_append, List.sum_append, List.sum_replicate, List.sum_map_mul_right, List.map_id',
    powersOfTwoBlocks_sum _ (Nat.mod_lt _ (by norm_num))]
  have h1 := Nat.div_add_mod (size / C) 64
  have h2 := Nat.div_add_mod size C
  split_ifs with h0
  · simp only [List.sum_nil, Nat.add_zero, smul_eq_mul]
    zify at h0 h1 h2 ⊢
    linear_combination (C : ℤ) * h1 + h2 - h0
  · simp only [List.sum_singleton, smul_eq_mul]
    zify at h1 h2 ⊢
    linear_combination (C : ℤ) * h1 + h2

/-- Consecutive blocks with the given lengths, from `start`. -/
def tile (start : ℕ) : List ℕ → List (ℕ × ℕ)
  | [] => []
  | l :: ls => (start, start + l) :: tile (start + l) ls

/-- The tiling covers `[start, start + Σ lens)`. -/
theorem tile_cover (start : ℕ) : ∀ (lens : List ℕ) (n : ℕ), start ≤ n → n < start + lens.sum →
    ∃ blk ∈ tile start lens, blk.1 ≤ n ∧ n < blk.2 := by
  intro lens
  induction lens generalizing start with
  | nil => intro n h1 h2; simp at h2; omega
  | cons l ls ih =>
    intro n h1 h2
    rw [List.sum_cons] at h2
    rcases Nat.lt_or_ge n (start + l) with h | h
    · exact ⟨(start, start + l), by simp [tile], h1, h⟩
    · obtain ⟨blk, hblk, hb1, hb2⟩ := ih (start + l) n h (by omega)
      exact ⟨blk, by simp [tile, hblk], hb1, hb2⟩

/-- The blocks never overlap: they are laid out in increasing order. -/
theorem tile_disjoint (start : ℕ) : ∀ (lens : List ℕ),
    (tile start lens).Pairwise fun a b => a.2 ≤ b.1 := by
  intro lens
  induction lens generalizing start with
  | nil => exact List.Pairwise.nil
  | cons l ls ih =>
    simp only [tile, List.pairwise_cons]
    refine ⟨fun blk hblk => ?_, ih _⟩
    -- every later block starts at or after start + l
    have : ∀ s (ls : List ℕ) blk, blk ∈ tile s ls → s ≤ blk.1 := by
      intro s ls
      induction ls generalizing s with
      | nil => intro blk h; simp [tile] at h
      | cons m ms ihm =>
        intro blk h
        simp only [tile, List.mem_cons] at h
        rcases h with rfl | h
        · exact le_refl _
        · have := ihm _ blk h; omega
    exact this _ _ blk hblk

/-- Claim GPU-1: the block tiling of a field partitions it. -/
theorem blockTiling_cover (C start size n : ℕ) (h1 : start ≤ n) (h2 : n < start + size) :
    ∃ blk ∈ tile start (blockLens size C), blk.1 ≤ n ∧ n < blk.2 :=
  tile_cover start _ n h1 (by rw [blockLens_sum]; exact h2)

/-! ### NUM-6, NUM-7: per-base constants -/

/-- `chunk_constants`: the exponent of the largest power of `b` below `bound`. -/
def chunkExp (b bound : ℕ) : ℕ := Nat.log b (bound - 1)

/-- Claim NUM-6: `b^e < bound ≤ b^(e+1)`, i.e. the chosen power is maximal. -/
theorem chunkExp_spec {b bound : ℕ} (hb : 2 ≤ b) (hbound : 2 ≤ bound) :
    b ^ chunkExp b bound < bound ∧ bound ≤ b ^ (chunkExp b bound + 1) := by
  unfold chunkExp
  constructor
  · have := Nat.pow_log_le_self b (by omega : bound - 1 ≠ 0)
    omega
  · have := Nat.lt_pow_succ_log_self (by omega : 1 < b) (bound - 1)
    rw [Nat.succ_eq_add_one] at this
    omega

/-- The `split16` bound for the `u16` constants: `(div - 1) · 2^16 < 2^32`. -/
theorem split16_shift_bound {d : ℕ} (hd : d < 2 ^ 16) : (d - 1) * 2 ^ 16 < 2 ^ 32 := by
  have : (d - 1) * 2 ^ 16 ≤ (2 ^ 16 - 2) * 2 ^ 16 := Nat.mul_le_mul_right _ (by omega)
  omega

/-- `prefilter_params`: digits guaranteed for both powers over `[start, ∞)`,
computed exactly from the range start (the Rust subtracts one for safety). -/
def prefilterDepth (b start : ℕ) : ℕ :=
  min (numDigits b (start ^ 2)) (numDigits b (start ^ 3)) - 1

/-- Claim NUM-7 with GPU-8: at that depth the prefilter is sound for every
candidate at or after the range start. -/
theorem prefilter_sound {b start n : ℕ} (hb : 2 ≤ b) (hs : start ≤ n) (h : IsNice b n) :
    n % b ^ prefilterDepth b start ∈ lsdBitmap b (prefilterDepth b start) := by
  apply mem_lsdBitmap_of_isNice hb _ _ h
  · unfold prefilterDepth
    have := numDigits_pow_mono (b := b) 2 hs
    omega
  · unfold prefilterDepth
    have := numDigits_pow_mono (b := b) 3 hs
    omega

end Nice
