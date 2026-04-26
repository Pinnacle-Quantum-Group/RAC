/-
  RAC Formal Proofs — MAC (Multiply-Accumulate) Equivalence
  Pinnacle Quantum Group — April 2026

  Proves: project(a,0,angle_b)*|b| = a*b, inner product, matmul equivalence.
  Reference: rac_cuda.cu rac_project, rac_matmul, rac.h
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section
open Real

namespace RAC.Cordic.MacEquivalence

def project (vx vy θ : ℝ) : ℝ := vx * cos θ + vy * sin θ

theorem project_zero (vx vy : ℝ) : project vx vy 0 = vx := by
  simp [project, cos_zero, sin_zero]

theorem project_pi (vx vy : ℝ) : project vx vy π = -vx := by
  simp [project, cos_pi, sin_pi]

theorem mac_equivalence (a b : ℝ) :
    let angle_b := if b ≥ 0 then (0 : ℝ) else π
    project a 0 angle_b * |b| = a * b := by
  simp only
  by_cases h : b ≥ 0
  · simp [h, project_zero, abs_of_nonneg h]
  · push_neg at h
    have hnle : ¬ (b ≥ 0) := not_le.mpr h
    simp [hnle, project_pi, abs_of_neg h]
    ring

def racInner (a b : Fin n → ℝ) : ℝ :=
  ∑ i, project (a i) 0 (if b i ≥ 0 then 0 else π) * |b i|

def stdInner (a b : Fin n → ℝ) : ℝ := ∑ i, a i * b i

theorem inner_product_equivalence (a b : Fin n → ℝ) :
    racInner a b = stdInner a b := by
  unfold racInner stdInner; congr 1; ext i
  exact mac_equivalence (a i) (b i)

def racMatmul (A : Fin M → Fin K → ℝ) (B : Fin K → Fin N → ℝ)
    (m : Fin M) (n : Fin N) : ℝ :=
  ∑ k, project (A m k) 0 (if B k n ≥ 0 then 0 else π) * |B k n|

def stdMatmul (A : Fin M → Fin K → ℝ) (B : Fin K → Fin N → ℝ)
    (m : Fin M) (n : Fin N) : ℝ :=
  ∑ k, A m k * B k n

theorem matmul_equivalence (A : Fin M → Fin K → ℝ) (B : Fin K → Fin N → ℝ)
    (m : Fin M) (n : Fin N) :
    racMatmul A B m n = stdMatmul A B m n := by
  unfold racMatmul stdMatmul; congr 1; ext k
  exact mac_equivalence (A m k) (B k n)

theorem coherence_bounded (θ_a θ_b : ℝ) :
    -1 ≤ cos (θ_a - θ_b) ∧ cos (θ_a - θ_b) ≤ 1 :=
  ⟨neg_one_le_cos (θ_a - θ_b), cos_le_one (θ_a - θ_b)⟩

end RAC.Cordic.MacEquivalence
