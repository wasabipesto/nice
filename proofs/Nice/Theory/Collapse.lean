/-
The carry-blind collapse lemma (THY-2).

A linear digit statistic `L(d) = Σ w_i d_i` that is invariant under every
carry move (`d_i ↦ d_i − b`, `d_{i+1} ↦ d_{i+1} + 1`, which preserves the
value `Σ d_i b^i`) must have `w_{i+1} ≡ b · w_i`, and then `L(d) ≡ w_0 · N`
(mod m). So every carry-invariant linear filter is a congruence on `N`
itself: digit sums give mod `b − 1`, alternating sums mod `b + 1`, and
nothing new. Filters that are only invariant at the bottom `k` positions
(the LSD filter) or on a window are outside the hypothesis.
-/
import Mathlib.Tactic
import Mathlib.Data.Nat.Digits.Defs

namespace Nice

/-- The weighted digit statistic `Σ w_i d_i`, digits least significant first. -/
def wsum (w : ℕ → ℤ) : List ℤ → ℤ
  | [] => 0
  | d :: ds => w 0 * d + wsum (fun i => w (i + 1)) ds

/-- The value `Σ d_i b^i` of a digit sequence (over `ℤ`, so carry moves are
just rewrites). -/
def value (b : ℤ) : List ℤ → ℤ
  | [] => 0
  | d :: ds => d + b * value b ds

/-- A carry move at position `i`: borrow `b` from digit `i` into digit `i+1`. -/
def carry (b : ℤ) : ℕ → List ℤ → List ℤ
  | _, [] => []
  | 0, d :: e :: ds => (d - b) :: (e + 1) :: ds
  | 0, [d] => [d]
  | i + 1, d :: ds => d :: carry b i ds

/-- Carry moves preserve the value. -/
theorem value_carry (b : ℤ) : ∀ (i : ℕ) (ds : List ℤ), value b (carry b i ds) = value b ds := by
  intro i
  induction i with
  | zero =>
    intro ds
    match ds with
    | [] => rfl
    | [d] => rfl
    | d :: e :: ds => simp only [carry, value]; ring
  | succ i ih =>
    intro ds
    match ds with
    | [] => rfl
    | d :: ds => simp only [carry, value, ih]

/-- Invariance under the carry move at `i` on the sequence `[0,…,0,b,0,…]`
versus `[0,…,0,0,1,…]` forces `w_{i+1} ≡ b · w_i (mod m)`. -/
theorem weight_rel_of_invariant {m b : ℤ} {w : ℕ → ℤ}
    (hinv : ∀ i ds, wsum w (carry b i ds) ≡ wsum w ds [ZMOD m]) (i : ℕ) :
    w (i + 1) ≡ b * w i [ZMOD m] := by
  -- the sequence with `b` at position `i` and `0` at position `i+1`
  have key : ∀ (w : ℕ → ℤ) (i : ℕ),
      (∀ i ds, wsum w (carry b i ds) ≡ wsum w ds [ZMOD m]) → w (i + 1) ≡ b * w i [ZMOD m] := by
    intro w i
    induction i generalizing w with
    | zero =>
      intro h
      have := h 0 [b, 0]
      simp only [carry, wsum, sub_self, zero_add, mul_zero, add_zero, mul_one] at this
      rw [zero_add, mul_comm]
      exact this
    | succ i ih =>
      intro h
      apply ih (fun j => w (j + 1))
      intro j ds
      have := h (j + 1) (0 :: ds)
      simpa only [carry, wsum, mul_zero, zero_add] using this
  exact key w i hinv

/-- Claim THY-2: with `w_{i+1} ≡ b w_i`, the statistic collapses to `w_0 · N`. -/
theorem collapse {m b : ℤ} :
    ∀ (w : ℕ → ℤ), (∀ i, w (i + 1) ≡ b * w i [ZMOD m]) →
      ∀ ds : List ℤ, wsum w ds ≡ w 0 * value b ds [ZMOD m] := by
  intro w hw ds
  induction ds generalizing w with
  | nil => simp [wsum, value]
  | cons d ds ih =>
    simp only [wsum, value]
    have hrest := ih (fun i => w (i + 1)) (fun i => hw (i + 1))
    -- w 1 ≡ b w 0, so w 1 * value ≡ b w 0 value
    have h1 : w 1 * value b ds ≡ b * w 0 * value b ds [ZMOD m] := (hw 0).mul_right _
    calc w 0 * d + wsum (fun i => w (i + 1)) ds
        ≡ w 0 * d + w 1 * value b ds [ZMOD m] := hrest.add_left _
      _ ≡ w 0 * d + b * w 0 * value b ds [ZMOD m] := h1.add_left _
      _ = w 0 * (d + b * value b ds) := by ring

/-- The two together: a carry-invariant linear statistic is a function of
the value alone. -/
theorem collapse_of_invariant {m b : ℤ} {w : ℕ → ℤ}
    (hinv : ∀ i ds, wsum w (carry b i ds) ≡ wsum w ds [ZMOD m]) (ds : List ℤ) :
    wsum w ds ≡ w 0 * value b ds [ZMOD m] :=
  collapse w (weight_rel_of_invariant hinv) ds

end Nice
