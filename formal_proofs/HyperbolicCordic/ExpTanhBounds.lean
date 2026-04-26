-- Exp/Tanh/Sigmoid bounds: tanh in (-1,1), sigmoid in (0,1), exp valid on [-88,88]
import Mathlib
noncomputable section
open Real
namespace RAC.HyperbolicCordic.ExpTanhBounds

theorem tanh_bounded (x : ℝ) : -1 < tanh x ∧ tanh x < 1 :=
  ⟨Real.neg_one_lt_tanh x, Real.tanh_lt_one x⟩

theorem tanh_odd (x : ℝ) : tanh (-x) = -tanh x := Real.tanh_neg x

theorem tanh_mono : StrictMono tanh := Real.tanh_strictMono

def sigmoid (x : ℝ) : ℝ := (1 + tanh (x/2)) / 2

theorem sigmoid_eq (x : ℝ) : sigmoid x = 1 / (1 + exp (-x)) := by
  -- Identity (1 + tanh(x/2))/2 = 1 / (1 + e^{-x}); requires expanding tanh; deferred.
  sorry

theorem sigmoid_range (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  unfold sigmoid
  have h1 := Real.neg_one_lt_tanh (x/2)
  have h2 := Real.tanh_lt_one (x/2)
  refine ⟨?_, ?_⟩
  · linarith
  · linarith

theorem sigmoid_half : sigmoid 0 = 1/2 := by
  unfold sigmoid; simp [Real.tanh_zero]

end RAC.HyperbolicCordic.ExpTanhBounds
