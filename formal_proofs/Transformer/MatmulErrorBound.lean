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
  sorry

end RAC.Transformer.MatmulError
