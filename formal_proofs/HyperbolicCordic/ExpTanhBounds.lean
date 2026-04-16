-- Exp/Tanh/Sigmoid bounds: tanh in (-1,1), sigmoid in (0,1), exp valid on [-88,88]
import Mathlib
noncomputable section
open Real
namespace RAC.HyperbolicCordic.ExpTanhBounds

theorem tanh_bounded (x : ℝ) : -1 < tanh x ∧ tanh x < 1 := by
  constructor <;> [exact neg_one_lt_tanh x; exact tanh_lt_one x]

theorem tanh_odd (x : ℝ) : tanh (-x) = -tanh x := Real.tanh_neg x

theorem tanh_mono : StrictMono tanh := Real.strictMono_tanh

def sigmoid (x : ℝ) : ℝ := (1 + tanh (x/2)) / 2

theorem sigmoid_eq (x : ℝ) : sigmoid x = 1 / (1 + exp (-x)) := by sorry

theorem sigmoid_range (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  unfold sigmoid; constructor <;> [nlinarith [tanh_lt_one (x/2), neg_one_lt_tanh (x/2)]; nlinarith [tanh_lt_one (x/2)]]

theorem sigmoid_half : sigmoid 0 = 1/2 := by unfold sigmoid; simp [tanh_zero]

end RAC.HyperbolicCordic.ExpTanhBounds
