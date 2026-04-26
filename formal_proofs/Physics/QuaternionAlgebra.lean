-- Quaternion algebra: Hamilton product, norm multiplicativity, non-commutativity
import Mathlib
namespace RAC.Physics.QuaternionAlgebra

@[ext]
structure Quat where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

def qmul (a b : Quat) : Quat where
  w := a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
  x := a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
  y := a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
  z := a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w

def normSq (q : Quat) : ℝ := q.w^2 + q.x^2 + q.y^2 + q.z^2
def conj (q : Quat) : Quat := ⟨q.w, -q.x, -q.y, -q.z⟩

theorem normSq_mul (a b : Quat) : normSq (qmul a b) = normSq a * normSq b := by
  unfold normSq qmul; ring

theorem mul_conj (q : Quat) : qmul q (conj q) = ⟨normSq q, 0, 0, 0⟩ := by
  unfold qmul conj normSq; ext <;> simp <;> ring

theorem noncommutative : qmul ⟨0,1,0,0⟩ ⟨0,0,1,0⟩ ≠ qmul ⟨0,0,1,0⟩ ⟨0,1,0,0⟩ := by
  unfold qmul
  -- simp + Quat.mk.injEq + norm-arithmetic closes; the prior `norm_num`
  -- ran on a no-goals state and failed.
  simp [Quat.mk.injEq]

end RAC.Physics.QuaternionAlgebra
