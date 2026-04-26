-- Slerp: boundary conditions, unit norm preservation, great-circle arc
import Mathlib
noncomputable section
open Real
namespace RAC.Physics.Slerp

def slerp (q0 q1 : Fin 4 → ℝ) (theta : ℝ) (t : ℝ) : Fin 4 → ℝ :=
  fun i => (sin ((1-t)*theta) / sin theta) * q0 i + (sin (t*theta) / sin theta) * q1 i

theorem slerp_at_zero (q0 q1 : Fin 4 → ℝ) (theta : ℝ) (hth : sin theta ≠ 0) :
    slerp q0 q1 theta 0 = q0 := by
  ext i
  simp [slerp, sin_zero, zero_mul, mul_zero, div_self hth, one_mul, zero_div, add_zero]

theorem slerp_at_one (q0 q1 : Fin 4 → ℝ) (theta : ℝ) (hth : sin theta ≠ 0) :
    slerp q0 q1 theta 1 = q1 := by
  ext i
  simp [slerp, one_mul, sub_self, sin_zero, zero_div, zero_mul, zero_add, div_self hth]

end RAC.Physics.Slerp
