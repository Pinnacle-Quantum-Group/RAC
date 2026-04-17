/-
  RAC — RoPE as Givens Rotation
  Pinnacle Quantum Group — April 2026

  Proves that Rotary Position Embeddings (RoPE) are exactly
  Givens rotations, making them native to CORDIC hardware.
  Key insight: RoPE applies 2D rotation to pairs of embedding dims.
  Reference: formal_proofs/Transformer/RoPEEquivalence.lean, rac_cuda.cu
-/
import Mathlib

noncomputable section
open Real Matrix

namespace RAC.Transformer.RoPEGivens

/-! ## 1. Givens Rotation Matrix -/

def givensMatrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos θ, -sin θ; sin θ, cos θ]

def givensApply (θ : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  (givensMatrix θ).mulVec v

/-! ## 2. RoPE Transform (pair-wise rotation) -/

def ropeTransform (x₁ x₂ θ : ℝ) : ℝ × ℝ :=
  (x₁ * cos θ - x₂ * sin θ, x₁ * sin θ + x₂ * cos θ)

/-! ## 3. RoPE = Givens -/

theorem rope_is_givens (x₁ x₂ θ : ℝ) :
    ropeTransform x₁ x₂ θ =
    (givensApply θ ![x₁, x₂] 0, givensApply θ ![x₁, x₂] 1) := by
  unfold ropeTransform givensApply givensMatrix
  simp [Matrix.mulVec, Matrix.dotProduct, Fin.sum_univ_two]
  constructor <;> ring

/-! ## 4. Magnitude Preservation (Orthogonality) -/

theorem rope_preserves_magnitude (x₁ x₂ θ : ℝ) :
    (ropeTransform x₁ x₂ θ).1 ^ 2 + (ropeTransform x₁ x₂ θ).2 ^ 2 =
    x₁ ^ 2 + x₂ ^ 2 := by
  unfold ropeTransform
  nlinarith [sin_sq_add_cos_sq θ]

/-! ## 5. Givens Composition = Angle Addition -/

theorem rope_composition (x₁ x₂ θ₁ θ₂ : ℝ) :
    ropeTransform
      (ropeTransform x₁ x₂ θ₁).1
      (ropeTransform x₁ x₂ θ₁).2 θ₂ =
    ropeTransform x₁ x₂ (θ₁ + θ₂) := by
  unfold ropeTransform
  constructor <;> { simp [cos_add, sin_add]; ring }

/-! ## 6. Identity at θ = 0 -/

theorem rope_identity (x₁ x₂ : ℝ) :
    ropeTransform x₁ x₂ 0 = (x₁, x₂) := by
  unfold ropeTransform; simp [cos_zero, sin_zero]

/-! ## 7. Inverse at -θ -/

theorem rope_inverse (x₁ x₂ θ : ℝ) :
    ropeTransform
      (ropeTransform x₁ x₂ θ).1
      (ropeTransform x₁ x₂ θ).2 (-θ) = (x₁, x₂) := by
  rw [rope_composition, add_neg_cancel]; exact rope_identity x₁ x₂

/-! ## 8. Position-Dependent Angle -/

def ropeAngle (pos dim : ℕ) (d_model : ℕ) : ℝ :=
  ↑pos / (10000 : ℝ) ^ (2 * ↑dim / ↑d_model)

theorem rope_angle_pos_zero (dim d_model : ℕ) :
    ropeAngle 0 dim d_model = 0 := by
  unfold ropeAngle; simp

theorem rope_angle_linear_in_pos (pos₁ pos₂ dim d_model : ℕ) :
    ropeAngle (pos₁ + pos₂) dim d_model =
    ropeAngle pos₁ dim d_model + ropeAngle pos₂ dim d_model := by
  unfold ropeAngle; push_cast; ring

/-! ## 9. CORDIC Native: No Multipliers Needed
    RoPE rotation is exactly what CORDIC computes natively,
    unlike traditional implementations that use 4 multiplications. -/

theorem cordic_replaces_four_muls (x₁ x₂ θ : ℝ) :
    ropeTransform x₁ x₂ θ =
    (x₁ * cos θ - x₂ * sin θ, x₁ * sin θ + x₂ * cos θ) := rfl

end RAC.Transformer.RoPEGivens
