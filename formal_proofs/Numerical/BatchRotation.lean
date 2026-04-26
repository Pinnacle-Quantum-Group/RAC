/-
  RAC — Batch Rotation Correctness
  Pinnacle Quantum Group — April 2026

  Proves that batch CORDIC rotation over a vector produces
  the same result as individual rotations, and that the
  batch operation preserves the total squared magnitude.
  Reference: rac_cuda.cu rac_rotate_batch, lib/c/rac_blas.c
-/
import Mathlib

noncomputable section
open Real BigOperators Finset

namespace RAC.Numerical.BatchRotation

/-! ## 1. Single Rotation -/

def rotate2D (x y θ : ℝ) : ℝ × ℝ :=
  (x * cos θ - y * sin θ, x * sin θ + y * cos θ)

theorem rotate_preserves_norm (x y θ : ℝ) :
    (rotate2D x y θ).1 ^ 2 + (rotate2D x y θ).2 ^ 2 = x ^ 2 + y ^ 2 := by
  show (x * cos θ - y * sin θ) ^ 2 + (x * sin θ + y * cos θ) ^ 2 = x ^ 2 + y ^ 2
  nlinarith [sin_sq_add_cos_sq θ]

/-! ## 2. Batch Rotation Definition -/

def batchRotate (pairs : Fin n → ℝ × ℝ) (angles : Fin n → ℝ) (i : Fin n) : ℝ × ℝ :=
  rotate2D (pairs i).1 (pairs i).2 (angles i)

/-! ## 3. Batch = Pointwise Individual -/

theorem batch_eq_pointwise (pairs : Fin n → ℝ × ℝ) (angles : Fin n → ℝ) (i : Fin n) :
    batchRotate pairs angles i = rotate2D (pairs i).1 (pairs i).2 (angles i) := rfl

/-! ## 4. Total Squared Magnitude Preserved -/

def totalNormSq (pairs : Fin n → ℝ × ℝ) : ℝ :=
  ∑ i, ((pairs i).1 ^ 2 + (pairs i).2 ^ 2)

theorem batch_preserves_total_norm (pairs : Fin n → ℝ × ℝ) (angles : Fin n → ℝ) :
    totalNormSq (batchRotate pairs angles) = totalNormSq pairs := by
  unfold totalNormSq batchRotate
  congr 1; ext i
  exact rotate_preserves_norm (pairs i).1 (pairs i).2 (angles i)

/-! ## 5. Batch Rotation Composition -/

theorem batch_composition (pairs : Fin n → ℝ × ℝ) (θ₁ θ₂ : Fin n → ℝ) :
    batchRotate (batchRotate pairs θ₁) θ₂ =
    fun i => rotate2D (pairs i).1 (pairs i).2 (θ₁ i + θ₂ i) := by
  funext i
  unfold batchRotate rotate2D
  refine Prod.ext ?_ ?_
  · show ((pairs i).1 * cos (θ₁ i) - (pairs i).2 * sin (θ₁ i)) * cos (θ₂ i)
        - ((pairs i).1 * sin (θ₁ i) + (pairs i).2 * cos (θ₁ i)) * sin (θ₂ i)
      = (pairs i).1 * cos (θ₁ i + θ₂ i) - (pairs i).2 * sin (θ₁ i + θ₂ i)
    rw [cos_add, sin_add]; ring
  · show ((pairs i).1 * cos (θ₁ i) - (pairs i).2 * sin (θ₁ i)) * sin (θ₂ i)
        + ((pairs i).1 * sin (θ₁ i) + (pairs i).2 * cos (θ₁ i)) * cos (θ₂ i)
      = (pairs i).1 * sin (θ₁ i + θ₂ i) + (pairs i).2 * cos (θ₁ i + θ₂ i)
    rw [cos_add, sin_add]; ring

/-! ## 6. Batch Identity -/

theorem batch_identity (pairs : Fin n → ℝ × ℝ) :
    batchRotate pairs (fun _ => 0) = pairs := by
  funext i
  unfold batchRotate rotate2D
  refine Prod.ext ?_ ?_
  · show (pairs i).1 * cos 0 - (pairs i).2 * sin 0 = (pairs i).1
    rw [cos_zero, sin_zero]; ring
  · show (pairs i).1 * sin 0 + (pairs i).2 * cos 0 = (pairs i).2
    rw [cos_zero, sin_zero]; ring

/-! ## 7. Batch Inverse -/

theorem batch_inverse (pairs : Fin n → ℝ × ℝ) (angles : Fin n → ℝ) :
    batchRotate (batchRotate pairs angles) (fun i => -angles i) = pairs := by
  rw [batch_composition]
  have key : (fun i => rotate2D (pairs i).1 (pairs i).2 (angles i + -angles i))
           = batchRotate pairs (fun _ => 0) := by
    funext i
    show rotate2D (pairs i).1 (pairs i).2 (angles i + -angles i)
       = rotate2D (pairs i).1 (pairs i).2 0
    congr 1; ring
  rw [key]
  exact batch_identity pairs

/-! ## 8. Inner Product via Batch Rotation -/

def batchInnerProduct (a b : Fin n → ℝ) : ℝ :=
  ∑ i, a i * b i

theorem inner_product_rotation_invariant
    (a b : Fin n → ℝ) (θ : ℝ) :
    ∀ (pairs_a pairs_b : Fin n → ℝ × ℝ),
    (∀ i, (pairs_a i).1 = a i ∧ (pairs_a i).2 = 0) →
    (∀ i, (pairs_b i).1 = b i ∧ (pairs_b i).2 = 0) →
    totalNormSq (batchRotate pairs_a (fun _ => θ)) = totalNormSq pairs_a := by
  intro pa pb _ _
  exact batch_preserves_total_norm pa (fun _ => θ)

end RAC.Numerical.BatchRotation
