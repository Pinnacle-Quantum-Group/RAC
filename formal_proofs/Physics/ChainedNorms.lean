-- Chained 3D norm: √(√(x²+y²)² + z²) = √(x²+y²+z²)
import Mathlib
noncomputable section
open Real
namespace RAC.Physics.ChainedNorms

theorem chained_norm3 (x y z : ℝ) (hxy : 0 ≤ x^2 + y^2) :
    sqrt (sqrt (x^2 + y^2) ^ 2 + z^2) = sqrt (x^2 + y^2 + z^2) := by
  rw [sq_sqrt hxy]

theorem chained_norm4 (x y z w : ℝ) (hxy : 0 ≤ x^2+y^2) (hxyz : 0 ≤ x^2+y^2+z^2) :
    sqrt (sqrt (sqrt (x^2+y^2)^2 + z^2)^2 + w^2) = sqrt (x^2+y^2+z^2+w^2) := by
  rw [sq_sqrt (by positivity), sq_sqrt hxy]

end RAC.Physics.ChainedNorms
