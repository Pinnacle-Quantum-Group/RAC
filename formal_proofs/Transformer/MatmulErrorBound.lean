/-
  RAC — Matmul Error Bound
  |C_ij - (AB)_ij| ≤ K * ε_cordic * max_mag²
  Reference: rac_cuda.cu rac_matmul
-/
import Mathlib
noncomputable section
open Finset BigOperators
namespace RAC.Transformer.MatmulError

def exactMatmul (A : Fin M → Fin K → ℝ) (B : Fin K → Fin N → ℝ)
    (m : Fin M) (n : Fin N) : ℝ := ∑ k, A m k * B k n

def approxMatmul (A : Fin M → Fin K → ℝ) (B : Fin K → Fin N → ℝ)
    (err : Fin K → ℝ) (m : Fin M) (n : Fin N) : ℝ :=
  ∑ k, (A m k * B k n + err k)

theorem matmul_error_bound (A : Fin M → Fin K → ℝ) (B : Fin K → Fin N → ℝ)
    (err : Fin K → ℝ) (eps : ℝ) (h_err : ∀ k, |err k| ≤ eps)
    (m : Fin M) (n : Fin N) :
    |approxMatmul A B err m n - exactMatmul A B m n| ≤ K * eps := by
  -- Standard BLAS triangle inequality:
  --   approxMatmul - exactMatmul = ∑ err k    (the bilinear part cancels)
  --   |∑ err k| ≤ ∑ |err k| ≤ ∑ eps = K · eps.
  unfold approxMatmul exactMatmul
  have h_diff : (∑ k, (A m k * B k n + err k)) - (∑ k, A m k * B k n) = ∑ k, err k := by
    rw [← Finset.sum_sub_distrib]
    congr 1
    funext k; ring
  rw [h_diff]
  calc |∑ k, err k|
      ≤ ∑ k, |err k| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _k : Fin K, eps := Finset.sum_le_sum (fun k _ => h_err k)
    _ = (K : ℝ) * eps := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end RAC.Transformer.MatmulError
