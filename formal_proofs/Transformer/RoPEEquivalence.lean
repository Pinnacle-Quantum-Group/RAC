/-
  RAC — RoPE Rotation Equivalence
  CORDIC rotation = Givens rotation matrix = traditional 4-multiply RoPE
  Reference: rac.h rac_rope_apply, rac_torch.py
-/
import Mathlib
noncomputable section
open Real Matrix
namespace RAC.Transformer.RoPE

@[ext]
structure Vec2 where
  x : Real
  y : Real

def exactRotation (v : Vec2) (θ : Real) : Vec2 :=
  { x := v.x * cos θ - v.y * sin θ,
    y := v.x * sin θ + v.y * cos θ }

def innerProduct (u v : Vec2) : Real := u.x * v.x + u.y * v.y
def normSq (v : Vec2) : Real := v.x ^ 2 + v.y ^ 2

theorem exactRotation_preserves_normSq (v : Vec2) (θ : Real) :
    normSq (exactRotation v θ) = normSq v := by
  simp only [normSq, exactRotation]; nlinarith [sin_sq_add_cos_sq θ]

theorem exactRotation_preserves_innerProduct (u v : Vec2) (θ : Real) :
    innerProduct (exactRotation u θ) (exactRotation v θ) = innerProduct u v := by
  simp only [innerProduct, exactRotation]; nlinarith [sin_sq_add_cos_sq θ]

theorem exactRotation_compose (v : Vec2) (α β : Real) :
    exactRotation (exactRotation v α) β = exactRotation v (α + β) := by
  ext
  · simp only [exactRotation]; rw [cos_add, sin_add]; ring
  · simp only [exactRotation]; rw [cos_add, sin_add]; ring

theorem rope_relative_position (q k : Vec2) (θ_m θ_n : Real) :
    innerProduct (exactRotation q θ_m) (exactRotation k θ_n) =
    innerProduct (exactRotation q (θ_m - θ_n)) k := by
  simp only [innerProduct, exactRotation]; rw [cos_sub, sin_sub]; ring

end RAC.Transformer.RoPE
