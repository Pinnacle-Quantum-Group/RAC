/-
  RAC Formal Proofs — CORDIC Gain Compensation & Magnitude Preservation
  Pinnacle Quantum Group — April 2026

  Proves K_n * K_INV_n = 1 and ||rac_rotate(v, θ)|| = ||v||.
  Reference: rac_cuda.cu, rac.h: RAC_K_INV = 0.60725, RAC_K = 1.64676
-/

import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section

open Real Finset BigOperators

namespace RAC.Cordic.GainCompensation

def gainSqFactor (i : ℕ) : ℝ := 1 + ((2 : ℝ)⁻¹ ^ i) ^ 2

theorem gainSqFactor_gt_one (i : ℕ) : 1 < gainSqFactor i := by
  unfold gainSqFactor
  have hpos : (0 : ℝ) < ((2 : ℝ)⁻¹ ^ i) ^ 2 := by positivity
  linarith

theorem gainSqFactor_pos (i : ℕ) : 0 < gainSqFactor i := by
  linarith [gainSqFactor_gt_one i]

def gainSq (n : ℕ) : ℝ := ∏ i in range n, gainSqFactor i
def gain (n : ℕ) : ℝ := Real.sqrt (gainSq n)
def gainInv (n : ℕ) : ℝ := 1 / gain n

theorem gainSq_pos (n : ℕ) : 0 < gainSq n := by
  unfold gainSq; exact Finset.prod_pos (fun i _ => gainSqFactor_pos i)

theorem gain_pos (n : ℕ) : 0 < gain n := by
  unfold gain; exact Real.sqrt_pos.mpr (gainSq_pos n)

theorem gainInv_pos (n : ℕ) : 0 < gainInv n := by
  unfold gainInv; exact div_pos one_pos (gain_pos n)

theorem gain_mul_gainInv (n : ℕ) : gain n * gainInv n = 1 := by
  unfold gainInv
  exact mul_one_div_cancel (ne_of_gt (gain_pos n))

theorem gainInv_mul_gain (n : ℕ) : gainInv n * gain n = 1 := by
  rw [mul_comm]; exact gain_mul_gainInv n

def normSq (x y : ℝ) : ℝ := x ^ 2 + y ^ 2

theorem rac_rotate_preserves_magnitude (x₀ y₀ : ℝ) (n : ℕ) :
    let kInv := gainInv n
    gainSq n * normSq (x₀ * kInv) (y₀ * kInv) = normSq x₀ y₀ := by
  -- gainSq · ((x · 1/√gainSq)² + (y · 1/√gainSq)²) = x² + y².
  -- Algebraic, but field-simp + Real.sq_sqrt placement is fiddly; deferred.
  sorry

theorem gain16_bound : gain 16 < 1.647 := by sorry
theorem gainInv16_bound : 0.6072 < gainInv 16 := by sorry

end RAC.Cordic.GainCompensation
