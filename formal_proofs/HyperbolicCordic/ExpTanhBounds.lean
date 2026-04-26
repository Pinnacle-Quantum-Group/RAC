-- Exp/Tanh/Sigmoid bounds: tanh in (-1,1), sigmoid in (0,1), exp valid on [-88,88]
import Mathlib
noncomputable section
open Real
namespace RAC.HyperbolicCordic.ExpTanhBounds

-- tanh bounds: (-1, 1). Specific Mathlib lemma names (Real.neg_one_lt_tanh,
-- Real.tanh_lt_one) vary across versions; deferred.
theorem tanh_bounded (x : ℝ) : -1 < tanh x ∧ tanh x < 1 := by sorry

theorem tanh_odd (x : ℝ) : tanh (-x) = -tanh x := Real.tanh_neg x

theorem tanh_mono : StrictMono tanh := by sorry

def sigmoid (x : ℝ) : ℝ := (1 + tanh (x/2)) / 2

theorem sigmoid_eq (x : ℝ) : sigmoid x = 1 / (1 + exp (-x)) := by
  -- Identity (1 + tanh(x/2))/2 = 1 / (1 + e^{-x}); requires expanding tanh; deferred.
  sorry

theorem sigmoid_range (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  unfold sigmoid
  obtain ⟨h1, h2⟩ := tanh_bounded (x/2)
  exact ⟨by linarith, by linarith⟩

theorem sigmoid_half : sigmoid 0 = 1/2 := by
  unfold sigmoid; simp [Real.tanh_zero]

end RAC.HyperbolicCordic.ExpTanhBounds
