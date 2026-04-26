-- Exp/Tanh/Sigmoid bounds: tanh in (-1,1), sigmoid in (0,1), exp valid on [-88,88]
import Mathlib
noncomputable section
open Real
namespace RAC.HyperbolicCordic.ExpTanhBounds

/-- v4.5.0 has neither `neg_one_lt_tanh` nor `tanh_lt_one` directly. Derive
    them from the identities `cosh + sinh = exp(x)` and `cosh - sinh = exp(-x)`,
    both > 0, plus `cosh > 0`. -/
theorem tanh_bounded (x : ℝ) : -1 < tanh x ∧ tanh x < 1 := by
  have hc : 0 < Real.cosh x := Real.cosh_pos x
  rw [Real.tanh_eq_sinh_div_cosh]
  refine ⟨?_, ?_⟩
  · -- `-1 < sinh x / cosh x` ⟺ `-cosh x < sinh x` ⟺ `0 < cosh x + sinh x = exp x`
    rw [lt_div_iff hc]
    have h1 := Real.cosh_add_sinh x
    have h2 := Real.exp_pos x
    linarith
  · -- `sinh x / cosh x < 1` ⟺ `sinh x < cosh x` ⟺ `0 < cosh x - sinh x = exp(-x)`
    rw [div_lt_iff hc]
    have h1 := Real.cosh_sub_sinh x
    have h2 := Real.exp_pos (-x)
    linarith

theorem tanh_odd (x : ℝ) : tanh (-x) = -tanh x := Real.tanh_neg x

/-- Helper: `sinh` is strictly positive on positives. Mathlib v4.5.0
    doesn't ship `Real.sinh_pos`; derive from `sinh x = (exp x - exp(-x))/2`
    and strict monotonicity of `exp`. -/
private lemma sinh_pos {x : ℝ} (hx : 0 < x) : 0 < Real.sinh x := by
  rw [Real.sinh_eq]
  have h_exp_lt : Real.exp (-x) < Real.exp x :=
    Real.exp_lt_exp.mpr (by linarith)
  linarith

/-- `tanh` is strictly monotone via the identity
      tanh y - tanh x = sinh(y - x) / (cosh x * cosh y)
    plus `sinh_pos` and `cosh_pos`. -/
theorem tanh_mono : StrictMono tanh := by
  intro x y hxy
  have hcx : 0 < Real.cosh x := Real.cosh_pos x
  have hcy : 0 < Real.cosh y := Real.cosh_pos y
  rw [Real.tanh_eq_sinh_div_cosh, Real.tanh_eq_sinh_div_cosh]
  rw [div_lt_div_iff hcx hcy]
  -- Want: sinh x * cosh y < sinh y * cosh x.
  -- From sinh_sub:  sinh y * cosh x - cosh y * sinh x = sinh (y - x) > 0.
  have h_sub : Real.sinh (y - x) =
      Real.sinh y * Real.cosh x - Real.cosh y * Real.sinh x :=
    Real.sinh_sub y x
  have h_pos : 0 < Real.sinh (y - x) := sinh_pos (sub_pos.mpr hxy)
  -- linarith can't see `cosh y * sinh x = sinh x * cosh y`; nlinarith handles
  -- the multiplication commutativity along with the order argument.
  nlinarith [h_sub, h_pos]

def sigmoid (x : ℝ) : ℝ := (1 + tanh (x/2)) / 2

/-- `sigmoid x = 1 / (1 + exp(-x))` — classical identity.
    Setting e := exp(x/2), ē := exp(-(x/2)), we have e·ē = 1, and
    after expanding tanh and clearing fractions the goal reduces to
    `e + e·ē² = e + ē`, i.e., `e·ē² = ē`, which is `(e·ē)·ē = ē`. -/
theorem sigmoid_eq (x : ℝ) : sigmoid x = 1 / (1 + exp (-x)) := by
  unfold sigmoid
  have h_inv : Real.exp (x/2) * Real.exp (-(x/2)) = 1 := by
    rw [← Real.exp_add, show x/2 + -(x/2) = (0:ℝ) by ring, Real.exp_zero]
  have h_neg_x : Real.exp (-x) = Real.exp (-(x/2)) * Real.exp (-(x/2)) := by
    rw [← Real.exp_add]; congr 1; ring
  -- Express tanh(x/2) directly in exp form.
  have h_tanh : Real.tanh (x/2) =
      (Real.exp (x/2) - Real.exp (-(x/2))) / (Real.exp (x/2) + Real.exp (-(x/2))) := by
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    have h2 : (2 : ℝ) ≠ 0 := by norm_num
    field_simp
  rw [h_tanh, h_neg_x]
  have hex : (0:ℝ) < Real.exp (x/2) := Real.exp_pos _
  have hey : (0:ℝ) < Real.exp (-(x/2)) := Real.exp_pos _
  have hsum_ne : Real.exp (x/2) + Real.exp (-(x/2)) ≠ 0 := by positivity
  have hrhs_ne : (1:ℝ) + Real.exp (-(x/2)) * Real.exp (-(x/2)) ≠ 0 := by positivity
  -- field_simp introduces a `Real.exp (-x / 2)` form (from `(-x)/2`) that
  -- ring can't unify with `Real.exp (-(x/2))`; pre-normalise.
  have h_neg_form : Real.exp (-x / 2) = Real.exp (-(x / 2)) := by congr 1; ring
  field_simp
  rw [h_neg_form]
  -- Residual reduces to `(e·ē - 1)·ē = 0`; close via h_inv.
  linear_combination 2 * Real.exp (-(x/2)) * h_inv

theorem sigmoid_range (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  unfold sigmoid
  obtain ⟨hl, hu⟩ := tanh_bounded (x / 2)
  refine ⟨?_, ?_⟩
  · linarith
  · linarith

theorem sigmoid_half : sigmoid 0 = 1/2 := by
  unfold sigmoid
  simp [Real.tanh_zero]

end RAC.HyperbolicCordic.ExpTanhBounds
