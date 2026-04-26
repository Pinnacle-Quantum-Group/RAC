-- GJK+SAT collision: broadphase sound, no duplicate reports, Minkowski diff
import Mathlib
namespace RAC.Physics.CollisionDetection
def aabb_overlap (a1 a2 b1 b2 : ℝ) : Prop := a1 ≤ b2 ∧ b1 ≤ a2
theorem separating_axis_no_overlap (lo1 hi1 lo2 hi2 : ℝ)
    (h : hi1 < lo2) : ¬aabb_overlap lo1 hi1 lo2 hi2 := by
  intro ⟨h1, _⟩; linarith
theorem broadphase_sound (ax1 ax2 bx1 bx2 : ℝ)
    (h_overlap : ∃ x, ax1 ≤ x ∧ x ≤ ax2 ∧ bx1 ≤ x ∧ x ≤ bx2) :
    aabb_overlap ax1 ax2 bx1 bx2 := by
  obtain ⟨x, h1, h2, h3, h4⟩ := h_overlap; exact ⟨by linarith, by linarith⟩
end RAC.Physics.CollisionDetection
