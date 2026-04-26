/-
  RAC — Precision Knob: Iterations vs Error Relationship
  Pinnacle Quantum Group — April 2026

  Formalizes the tunable precision property of CORDIC:
  k iterations yield k bits of angular precision, with error ≤ atan(2^{-k}).
  Common configurations: 8, 16, 24 iterations.
  Reference: rac_cuda.cu RAC_ITERS, lib/c/rac_cordic.c
-/
import Mathlib
import RAC.Cordic.ArctanFacts

noncomputable section
open Real BigOperators  -- BigOperators needed for `∑` and `∏` syntax
open RAC.Cordic.ArctanFacts

namespace RAC.Cordic.PrecisionKnob

/-! ## 1. Error vs Iterations: One Bit Per Step -/

def maxError (k : ℕ) : ℝ := arctan ((2 : ℝ)⁻¹ ^ k)

/-- Closed via `RAC.Cordic.ArctanFacts` (round 8). -/
theorem error_positive (k : ℕ) : 0 < maxError k :=
  arctan_pos (inv_two_pow_pos k)

theorem error_decreasing : StrictAnti maxError :=
  arctan_strictMono.comp_strictAnti inv_two_pow_strictAnti

theorem error_bounded_by_power (k : ℕ) : maxError k ≤ (2 : ℝ)⁻¹ ^ k :=
  arctan_le_self_of_nonneg (inv_two_pow_pos k).le

/-! ## 2. Common Configurations -/

theorem error_8bit : maxError 8 ≤ (2 : ℝ)⁻¹ ^ 8 := error_bounded_by_power 8
theorem error_16bit : maxError 16 ≤ (2 : ℝ)⁻¹ ^ 16 := error_bounded_by_power 16
theorem error_24bit : maxError 24 ≤ (2 : ℝ)⁻¹ ^ 24 := error_bounded_by_power 24

/-! ## 3. Total Angular Coverage -/

def totalCoverage (k : ℕ) : ℝ := ∑ i in Finset.range k, arctan ((2 : ℝ)⁻¹ ^ i)

theorem coverage_monotone : Monotone totalCoverage := by
  -- Avoid reusing the bound name `i` (used inside the sum) for the outer index;
  -- rename to `m n` so unification doesn't shadow.
  intro m n hmn
  unfold totalCoverage
  -- For non-negative summand on `Finset.range n \ Finset.range m`, the larger
  -- range has a larger sum.
  apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hmn)
  intro i _ _
  exact (error_positive i).le

theorem coverage_first_is_pi_over_4 :
    arctan ((2 : ℝ)⁻¹ ^ 0) = π / 4 := by
  simp [arctan_one]

/-! ## 4. Precision-Performance Tradeoff -/

structure PrecisionConfig where
  iterations : ℕ
  h_min : 4 ≤ iterations
  h_max : iterations ≤ 32

def angularPrecision (pc : PrecisionConfig) : ℝ :=
  maxError pc.iterations

def bitsOfPrecision (pc : PrecisionConfig) : ℕ := pc.iterations

theorem more_iters_more_precise (pc₁ pc₂ : PrecisionConfig)
    (h : pc₁.iterations < pc₂.iterations) :
    angularPrecision pc₂ < angularPrecision pc₁ :=
  error_decreasing h

/-! ## 5. Convergence: Error → 0 as k → ∞ -/

theorem error_tends_to_zero :
    Filter.Tendsto maxError Filter.atTop (nhds 0) := by
  apply squeeze_zero (fun k => le_of_lt (error_positive k)) error_bounded_by_power
  exact tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)

/-! ## 6. Gain Factor Accumulation -/

def gainFactor (k : ℕ) : ℝ := ∏ i in Finset.range k, sqrt (1 + (2 : ℝ)⁻¹ ^ (2 * i))

theorem gain_factor_pos (k : ℕ) : 0 < gainFactor k := by
  unfold gainFactor
  apply Finset.prod_pos
  intro i _
  exact sqrt_pos.mpr (by positivity)

/-- gainFactor is monotone increasing — each new factor `sqrt(1 + 4⁻ⁱ) ≥ 1`. -/
lemma gainFactor_mono : Monotone gainFactor := by
  intro m n hmn
  unfold gainFactor
  apply Finset.prod_le_prod_of_subset_of_one_le' (Finset.range_mono hmn)
  intros i _ _
  rw [show (1 : ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
  apply Real.sqrt_le_sqrt
  have : (0 : ℝ) ≤ (2 : ℝ)⁻¹ ^ (2 * i) := by positivity
  linarith

/-! ## Closing `gain_approaches_constant`: monotone-bounded analysis

    Strategy:
    (1) `gainFactor n ^ 2 = ∏_{i<n}(1 + 4⁻ⁱ)` — squaring commutes with product.
    (2) For all n, this product ≤ `gainSq16 · exp(4⁻¹⁵/3)` via splitting
        at i=16 and bounding the tail by `1 + x ≤ exp x` + geometric series.
    (3) `gainSq16 < 2.71234` (16-term expansion) and `exp(4⁻¹⁵/3) < 1.000001`
        give `gainFactor n^2 < 2.713339 < 1.6469² < 1.65²`.
    (4) So `gainFactor n < 1.6469 < 1.65` for all n; sup is in [gainFactor 16, 1.6469].
        And `gainFactor 16 > 1.6` (lower 16-term expansion). -/

/-- gainFactor² as the underlying product (sqrt and ² cancel). -/
private lemma gainFactor_sq_eq (n : ℕ) :
    gainFactor n ^ 2 = ∏ i in Finset.range n, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) := by
  unfold gainFactor
  rw [← Finset.prod_pow]
  apply Finset.prod_congr rfl
  intros i _
  rw [Real.sq_sqrt]
  positivity

/-- Concrete: `gainFactor 16 ^ 2 < 2.71234` — direct 16-term expansion. -/
private lemma gainFactor_16_sq_lt : gainFactor 16 ^ 2 < 2.71234 := by
  rw [gainFactor_sq_eq]
  simp only [show (16 : ℕ) = 15 + 1 from rfl, Finset.prod_range_succ,
             Finset.prod_range_zero]
  norm_num

/-- Concrete: `gainFactor 16 ^ 2 > 2.56` — direct 16-term expansion. -/
private lemma gainFactor_16_sq_gt : 2.56 < gainFactor 16 ^ 2 := by
  rw [gainFactor_sq_eq]
  simp only [show (16 : ℕ) = 15 + 1 from rfl, Finset.prod_range_succ,
             Finset.prod_range_zero]
  norm_num

/-- Lower bound: `1.6 < gainFactor 16` — from `gainFactor 16 ^ 2 > 2.56 = 1.6²`. -/
private lemma gainFactor_16_gt : 1.6 < gainFactor 16 := by
  have h_pos : 0 < gainFactor 16 := gain_factor_pos 16
  have h_sq : (1.6 : ℝ) ^ 2 < gainFactor 16 ^ 2 := by
    have := gainFactor_16_sq_gt; nlinarith
  exact lt_of_pow_lt_pow_left 2 h_pos.le h_sq

/-- Upper bound on partial product: `∀ n, ∏_{i<n}(1 + 4⁻ⁱ) ≤ 2.71234 · exp(4⁻¹⁵/3)`.
    Splits the product at index 16; the head ≤ 2.71234 (via `gainFactor_16_sq_lt`,
    rewritten); the tail ≤ exp(∑ tail) ≤ exp(4⁻¹⁵/3) via `Real.add_one_le_exp`
    and a geometric series upper bound. -/
private lemma partialProd_le (n : ℕ) :
    ∏ i in Finset.range n, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) ≤
      2.71234 * Real.exp (4⁻¹^(15:ℕ) / 3) := by
  -- Two cases: n ≤ 16 (just bound by gainFactor_16_sq) and n > 16 (split).
  by_cases hn : n ≤ 16
  · -- n ≤ 16: ∏_{i<n} ≤ ∏_{i<16} (mono in number of factors ≥ 1) < 2.71234.
    have h_head : ∏ i in Finset.range n, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) ≤
        ∏ i in Finset.range 16, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) := by
      apply Finset.prod_le_prod_of_subset_of_one_le' (Finset.range_mono hn)
      intros i _ _
      have : (0 : ℝ) ≤ (2 : ℝ)⁻¹ ^ (2 * i) := by positivity
      linarith
    have h_head_lt : ∏ i in Finset.range 16, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) < 2.71234 := by
      have := gainFactor_16_sq_lt; rw [gainFactor_sq_eq] at this; exact this
    have h_exp_ge_one : (1 : ℝ) ≤ Real.exp (4⁻¹^(15:ℕ) / 3) :=
      Real.one_le_exp (by positivity)
    calc ∏ i in Finset.range n, (1 + (2 : ℝ)⁻¹ ^ (2 * i))
        ≤ ∏ i in Finset.range 16, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) := h_head
      _ ≤ 2.71234 := le_of_lt h_head_lt
      _ = 2.71234 * 1 := (mul_one _).symm
      _ ≤ 2.71234 * Real.exp (4⁻¹^(15:ℕ) / 3) := by
          apply mul_le_mul_of_nonneg_left h_exp_ge_one (by norm_num)
  · -- n > 16: split product at 16. Tail ≤ exp(∑ tail terms) ≤ exp(4⁻¹⁵/3).
    push_neg at hn
    -- Stub the n > 16 branch for now; the head bound + tail bound chain
    -- requires careful Finset.prod_Ico manipulation + exp_sum + tail bound.
    sorry

/-- The sup of `gainFactor` exists and is strictly less than 1.65. -/
private lemma gainFactor_lt_165 (n : ℕ) : gainFactor n < 1.65 := by
  -- gainFactor n ^ 2 ≤ 2.71234 · exp(4⁻¹⁵/3) < 1.65² = 2.7225.
  -- exp(4⁻¹⁵/3) ≤ 1 + 2 · 4⁻¹⁵/3 (via Real.abs_exp_sub_one_le with x = 4⁻¹⁵/3 ≤ 1).
  have h_pos : 0 < gainFactor n := gain_factor_pos n
  have h_sq_le : gainFactor n ^ 2 ≤
      2.71234 * Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := by
    rw [gainFactor_sq_eq]; exact partialProd_le n
  -- Bound exp(4⁻¹⁵/3) ≤ 1 + 2·4⁻¹⁵/3 via `Real.abs_exp_sub_one_le`.
  have h_x_small : |((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3)| ≤ 1 := by
    rw [abs_of_nonneg (by positivity)]
    have : (4 : ℝ)⁻¹ ^ (15 : ℕ) ≤ 1 := by
      apply pow_le_one
      · positivity
      · norm_num
    linarith
  have h_exp_close : Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) ≤
      1 + 2 * ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := by
    have h_abs := Real.abs_exp_sub_one_le h_x_small
    have h_pos_x : 0 ≤ (4 : ℝ)⁻¹ ^ (15 : ℕ) / 3 := by positivity
    have h_exp_ge : 1 ≤ Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := Real.one_le_exp h_pos_x
    rw [abs_of_nonneg h_pos_x] at h_abs
    rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ Real.exp _ - 1)] at h_abs
    linarith
  -- 4⁻¹⁵ ≤ 1/(4^15) = 1/1073741824 < 10⁻⁹. So 2 · 4⁻¹⁵ / 3 < 10⁻⁹.
  have h_x_tiny : ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) ≤ 1e-9 := by
    have : (4 : ℝ)⁻¹ ^ (15 : ℕ) = 1 / 4^15 := by
      rw [inv_pow, one_div]
    rw [this]; norm_num
  -- Combine:
  -- gainFactor n ^ 2 ≤ 2.71234 · (1 + 2·1e-9) < 2.71234 · 1.0000001 < 2.7225 = 1.65²
  have h_final_sq : gainFactor n ^ 2 < (1.65 : ℝ) ^ 2 := by
    have h_step1 : 2.71234 * Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) ≤
        2.71234 * (1 + 2 * 1e-9) := by
      apply mul_le_mul_of_nonneg_left _ (by norm_num : (0:ℝ) ≤ 2.71234)
      calc Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3)
          ≤ 1 + 2 * ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := h_exp_close
        _ ≤ 1 + 2 * 1e-9 := by linarith
    calc gainFactor n ^ 2 ≤ 2.71234 * Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := h_sq_le
      _ ≤ 2.71234 * (1 + 2 * 1e-9) := h_step1
      _ < (1.65 : ℝ) ^ 2 := by norm_num
  exact lt_of_pow_lt_pow_left 2 (by norm_num) h_final_sq

/-- Final assembly. -/
theorem gain_approaches_constant :
    ∃ K : ℝ, 1.6 < K ∧ K < 1.65 ∧
    Filter.Tendsto gainFactor Filter.atTop (nhds K) := by
  have h_mono : Monotone gainFactor := gainFactor_mono
  have h_bdd : BddAbove (Set.range gainFactor) := by
    refine ⟨1.65, ?_⟩
    rintro _ ⟨n, rfl⟩
    exact (gainFactor_lt_165 n).le
  refine ⟨⨆ n, gainFactor n, ?_, ?_, tendsto_atTop_ciSup h_mono h_bdd⟩
  · -- 1.6 < ⨆: chain through gainFactor 16 > 1.6.
    calc (1.6 : ℝ) < gainFactor 16 := gainFactor_16_gt
      _ ≤ ⨆ n, gainFactor n := le_ciSup h_bdd 16
  · -- ⨆ < 1.65: every gainFactor n < 1.65, so sup ≤ ?, but for STRICT
    -- we need a single explicit upper bound M with M < 1.65 and ∀ n, gainFactor n ≤ M.
    -- Use M = 1.6469 (slightly below 1.65); gainFactor n^2 ≤ 2.71234·exp(...) < 1.6469².
    sorry

end RAC.Cordic.PrecisionKnob
