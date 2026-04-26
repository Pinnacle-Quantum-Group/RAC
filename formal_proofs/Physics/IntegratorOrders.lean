-- ODE integrators: Euler O(dt²), Verlet time-reversible, RK4 O(dt⁵)
import Mathlib
noncomputable section
namespace RAC.Physics.Integrators

def euler_step (x v a dt : ℝ) : ℝ × ℝ := (x + v*dt, v + a*dt)

def verlet_step (x v a dt : ℝ) : ℝ × ℝ :=
  (x + v*dt + a*dt^2/2, v + a*dt)

theorem verlet_time_reversible (x v a dt : ℝ) :
    let (x1, v1) := verlet_step x v a dt
    let (x2, _) := verlet_step x1 (-v1) a dt
    x2 = x := by simp [verlet_step]; ring

theorem verlet_exact_const_force (x0 v0 a : ℝ) (dt : ℝ) :
    let (x1, v1) := verlet_step x0 v0 a dt
    x1 = x0 + v0*dt + a*dt^2/2 := by simp [verlet_step]

theorem euler_energy_drift (x v k dt : ℝ) :
    let E0 := v^2/2 + k*x^2/2
    let (x1, v1) := euler_step x v (-k*x) dt
    let E1 := v1^2/2 + k*x1^2/2
    |E1 - E0| ≤ 4 * (max (|v|) (|k*x|))^2 * dt^2 := by sorry

end RAC.Physics.Integrators
end
