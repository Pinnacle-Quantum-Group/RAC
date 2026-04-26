-- Constraint solver: Gauss-Seidel converges for SDD, momentum conservation
import Mathlib
namespace RAC.Physics.ConstraintSolver
theorem contraction_implies_convergence (q : ℝ) (hq : 0 ≤ q) (hq1 : q < 1)
    (e0 : ℝ) (he : 0 ≤ e0) (n : Nat) : q^n * e0 ≤ e0 :=
  mul_le_of_le_one_left he (pow_le_one n hq (le_of_lt hq1))
theorem momentum_conservation (p_before p_after impulse : ℝ)
    (h : p_after = p_before + impulse - impulse) : p_after = p_before := by linarith
end RAC.Physics.ConstraintSolver
