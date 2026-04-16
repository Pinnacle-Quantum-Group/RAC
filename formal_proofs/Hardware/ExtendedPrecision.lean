-- Variable-iteration CORDIC: 8(edge)/16(infer)/24(train), error monotone decreasing
import Mathlib
namespace RAC.Hardware.ExtendedPrecision

def cordic_error (n : Nat) : ℝ := (2 : ℝ)^(-(n : ℤ) + 1)

theorem error_monotone (m n : Nat) (h : m < n) : cordic_error n < cordic_error m := by sorry

theorem training_matches_fp32 : cordic_error 24 = (2 : ℝ)^(-(23:ℤ)) := by sorry

theorem precision_hierarchy :
    cordic_error 24 < cordic_error 16 ∧ cordic_error 16 < cordic_error 8 := by
  constructor <;> sorry

end RAC.Hardware.ExtendedPrecision
