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
  unfold slerp
  have h1 : sin ((1 - 0) * theta) / sin theta = 1 := by
    rw [sub_zero, one_mul]; exact div_self hth
  have h2 : sin (0 * theta) / sin theta = 0 := by
    rw [zero_mul, sin_zero, zero_div]
  rw [h1, h2, one_mul, zero_mul, add_zero]

theorem slerp_at_one (q0 q1 : Fin 4 → ℝ) (theta : ℝ) (hth : sin theta ≠ 0) :
    slerp q0 q1 theta 1 = q1 := by
  ext i
  unfold slerp
  have h1 : sin ((1 - 1) * theta) / sin theta = 0 := by
    rw [sub_self, zero_mul, sin_zero, zero_div]
  have h2 : sin (1 * theta) / sin theta = 1 := by
    rw [one_mul]; exact div_self hth
  rw [h1, h2, zero_mul, one_mul, zero_add]

end RAC.Physics.Slerp
