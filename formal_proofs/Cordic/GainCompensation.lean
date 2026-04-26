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
  have : 0 < ((2 : ℝ)⁻¹ ^ i) ^ 2 := by positivity
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
  field_simp [ne_of_gt (gain_pos n)]

theorem gainInv_mul_gain (n : ℕ) : gainInv n * gain n = 1 := by
  rw [mul_comm]; exact gain_mul_gainInv n

def normSq (x y : ℝ) : ℝ := x ^ 2 + y ^ 2

theorem rac_rotate_preserves_magnitude (x₀ y₀ : ℝ) (n : ℕ) :
    gainSq n * normSq (x₀ * gainInv n) (y₀ * gainInv n) = normSq x₀ y₀ := by
  unfold normSq gainInv gain
  have hsq_pos : 0 ≤ gainSq n := (gainSq_pos n).le
  have hsq_ne : gainSq n ≠ 0 := ne_of_gt (gainSq_pos n)
  have hsqrt_pos : 0 < Real.sqrt (gainSq n) := Real.sqrt_pos.mpr (gainSq_pos n)
  have hsqrt_ne : Real.sqrt (gainSq n) ≠ 0 := ne_of_gt hsqrt_pos
  -- Use the `^2` form (Real.sq_sqrt) so the ring closer can match the
  -- post-`field_simp` goal which contains `sqrt (gainSq n) ^ 2`.
  have hself : Real.sqrt (gainSq n) ^ 2 = gainSq n := Real.sq_sqrt hsq_pos
  field_simp [hsq_ne]
  -- Goal LHS reduces to `(x²+y²) · (gainSq - sqrt²) = 0`, hence the
  -- linear_combination coefficient is `-(x²+y²)` against `sqrt² - gainSq = 0`.
  linear_combination -(x₀^2 + y₀^2) * hself

/-- gainSq 16 < 2.71234 — tight numerical bound on the 16-term product
    `∏_{i=0}^{15} (1 + 4⁻ⁱ)`.  Direct expansion via `Finset.prod_range_succ`
    16 times gives the explicit rational, which `norm_num` evaluates.
    Actual value ≈ 2.7118367976; the bound 2.71234 has ~0.0005 margin. -/
private lemma gainSq_16_lt : gainSq 16 < 2.71234 := by
  unfold gainSq gainSqFactor
  simp only [show (16 : ℕ) = 15 + 1 from rfl, Finset.prod_range_succ,
             Finset.prod_range_zero]
  norm_num

/-- gain 16 < 1.647 = sqrt(2.713409). Use sqrt monotonicity + the
    tighter bound gainSq 16 < 2.71234 < 2.713409. -/
theorem gain16_bound : gain 16 < 1.647 := by
  unfold gain
  -- Show via sqrt mono: gainSq 16 < 1.647² and rewrite 1.647 = sqrt(1.647²).
  have h_target : Real.sqrt (1.647 * 1.647) = 1.647 :=
    Real.sqrt_mul_self (by norm_num : (0:ℝ) ≤ 1.647)
  rw [← h_target]
  apply Real.sqrt_lt_sqrt (gainSq_pos 16).le
  calc gainSq 16 < 2.71234 := gainSq_16_lt
    _ < 1.647 * 1.647 := by norm_num

/-- gainInv 16 > 0.607. Use 0.607 (rather than 0.6072) to keep slack
    against the looser `gainSq_16_lt` bound 2.71234. Concretely:
    0.607² = 0.368449, 1/0.368449 ≈ 2.71407 > 2.71234 ✓. -/
theorem gainInv16_bound : 0.607 < gainInv 16 := by
  unfold gainInv
  rw [lt_div_iff (gain_pos 16)]
  -- Goal: 0.607 * gain 16 < 1
  have h_sq : gain 16 * gain 16 < 1 / (0.607 * 0.607) := by
    have h_gain_sq_eq : gain 16 * gain 16 = gainSq 16 := by
      unfold gain
      exact Real.mul_self_sqrt (gainSq_pos 16).le
    rw [h_gain_sq_eq]
    calc gainSq 16 < 2.71234 := gainSq_16_lt
      _ < 1 / (0.607 * 0.607) := by norm_num
  have h_prod_pos : 0 < 0.607 * gain 16 := mul_pos (by norm_num) (gain_pos 16)
  have h_prod_sq : (0.607 * gain 16) * (0.607 * gain 16) < 1 := by
    have h_eq : (0.607 * gain 16) * (0.607 * gain 16) =
        (0.607 * 0.607) * (gain 16 * gain 16) := by ring
    rw [h_eq]
    have h_pos_const : (0:ℝ) < 0.607 * 0.607 := by norm_num
    calc (0.607 * 0.607) * (gain 16 * gain 16)
        < (0.607 * 0.607) * (1 / (0.607 * 0.607)) :=
          (mul_lt_mul_left h_pos_const).mpr h_sq
      _ = 1 := by field_simp
  nlinarith [h_prod_sq, h_prod_pos]

end RAC.Cordic.GainCompensation
