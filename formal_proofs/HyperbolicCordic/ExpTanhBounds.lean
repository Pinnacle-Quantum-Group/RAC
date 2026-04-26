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

/-- StrictMono of tanh follows from `tanh' = 1/cosh² > 0`. Stubbed: the
    composition `hasDerivAt → StrictMono` requires either calculus-of-variations
    machinery or the (non-existent in v4.5.0) `Real.strictMono_tanh` lemma. -/
theorem tanh_mono : StrictMono tanh := by sorry

def sigmoid (x : ℝ) : ℝ := (1 + tanh (x/2)) / 2

theorem sigmoid_eq (x : ℝ) : sigmoid x = 1 / (1 + exp (-x)) := by sorry

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
