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
    push_neg at hn  -- hn : 16 < n
    -- Rewrite range n as Ico 0 n, then split Ico 0 n = Ico 0 16 ∪ Ico 16 n.
    have h_range_eq : Finset.range n = Finset.Ico 0 n := by
      rw [Nat.Ico_zero_eq_range]
    have h_range_eq_16 : Finset.range 16 = Finset.Ico 0 16 := by
      rw [Nat.Ico_zero_eq_range]
    rw [h_range_eq, ← Finset.prod_Ico_consecutive _ (Nat.zero_le 16) hn.le,
        ← h_range_eq_16]
    -- Now: ∏ i in range 16, ... * ∏ i in Ico 16 n, ... ≤ 2.71234 · exp(...)
    -- Bound head: ∏ i in range 16, ... < 2.71234.
    have h_head_lt : ∏ i in Finset.range 16, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) < 2.71234 := by
      have := gainFactor_16_sq_lt; rw [gainFactor_sq_eq] at this; exact this
    have h_head_pos : 0 < ∏ i in Finset.range 16, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) := by
      apply Finset.prod_pos; intros i _
      have : (0 : ℝ) ≤ (2 : ℝ)⁻¹ ^ (2 * i) := by positivity
      linarith
    -- Bound tail: ∏ i in Ico 16 n, (1+(2⁻¹)^(2i)) ≤ exp(∑ i in Ico 16 n, (2⁻¹)^(2i))
    -- via 1+x ≤ exp x for each factor, then exp_sum.
    have h_tail_factor : ∀ i ∈ Finset.Ico 16 n,
        (1 + (2 : ℝ)⁻¹ ^ (2 * i)) ≤ Real.exp ((2 : ℝ)⁻¹ ^ (2 * i)) := by
      intros i _
      have := Real.add_one_le_exp ((2 : ℝ)⁻¹ ^ (2 * i))
      linarith
    have h_tail_pos : ∀ i ∈ Finset.Ico 16 n,
        0 ≤ (1 + (2 : ℝ)⁻¹ ^ (2 * i)) := by
      intros i _
      have : (0 : ℝ) ≤ (2 : ℝ)⁻¹ ^ (2 * i) := by positivity
      linarith
    have h_tail_le_exp : ∏ i in Finset.Ico 16 n, (1 + (2 : ℝ)⁻¹ ^ (2 * i)) ≤
        Real.exp (∑ i in Finset.Ico 16 n, (2 : ℝ)⁻¹ ^ (2 * i)) := by
      rw [← Real.exp_sum]
      exact Finset.prod_le_prod h_tail_pos h_tail_factor
    -- Bound the tail sum: ∑ i in Ico 16 n, (1/4)^i ≤ 4⁻¹⁵/3
    -- (since (2⁻¹)^(2i) = (1/4)^i; partial geometric sum ≤ infinite tail).
    have h_tail_sum_le : ∑ i in Finset.Ico 16 n, (2 : ℝ)⁻¹ ^ (2 * i) ≤
        (4 : ℝ)⁻¹ ^ (15 : ℕ) / 3 := by
      -- Step 1: rewrite (2⁻¹)^(2i) = (4⁻¹)^i.
      have h_conv : ∀ i : ℕ, (2 : ℝ)⁻¹ ^ (2 * i) = (4 : ℝ)⁻¹ ^ i := fun i => by
        rw [show (2 : ℝ)⁻¹ ^ (2 * i) = ((2 : ℝ)⁻¹ ^ 2) ^ i from by rw [← pow_mul]]
        norm_num
      rw [Finset.sum_congr rfl (fun i _ => h_conv i)]
      -- Step 2: closed form via `geom_sum_Ico`.
      rw [geom_sum_Ico (by norm_num : (4 : ℝ)⁻¹ ≠ 1) hn.le]
      -- Step 3: bound through the negative-denominator manipulation.
      have h_denom_neg : ((4 : ℝ)⁻¹ - 1) < 0 := by norm_num
      rw [div_le_iff_of_neg h_denom_neg]
      -- Goal: (4⁻¹)^15 / 3 * (4⁻¹ - 1) ≤ (4⁻¹)^n - (4⁻¹)^16
      have h_pow_n_nonneg : (0 : ℝ) ≤ (4 : ℝ)⁻¹ ^ n := by positivity
      have h_pow_15_nonneg : (0 : ℝ) ≤ (4 : ℝ)⁻¹ ^ (15 : ℕ) := by positivity
      have h_16_eq : ((4 : ℝ)⁻¹) ^ (16 : ℕ) = ((4 : ℝ)⁻¹) ^ (15 : ℕ) * ((4 : ℝ)⁻¹) := by
        rw [← pow_succ]
      rw [h_16_eq]
      -- After: (4⁻¹)^15 / 3 * (4⁻¹ - 1) ≤ (4⁻¹)^n - (4⁻¹)^15 * 4⁻¹
      -- Algebra: LHS = (4⁻¹)^15 * (-3/4) / 3 = -(4⁻¹)^15 / 4 = -(4⁻¹)^15 * 4⁻¹
      -- So LHS - RHS = -(4⁻¹)^15·4⁻¹ - (4⁻¹)^n + (4⁻¹)^15·4⁻¹ = -(4⁻¹)^n ≤ 0.
      nlinarith [h_pow_n_nonneg, h_pow_15_nonneg]
    -- Final exp monotonicity:
    have h_tail_exp_le : Real.exp (∑ i in Finset.Ico 16 n, (2 : ℝ)⁻¹ ^ (2 * i)) ≤
        Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := Real.exp_le_exp.mpr h_tail_sum_le
    calc (∏ i in Finset.range 16, (1 + (2 : ℝ)⁻¹ ^ (2 * i))) *
         (∏ i in Finset.Ico 16 n, (1 + (2 : ℝ)⁻¹ ^ (2 * i)))
        ≤ 2.71234 * Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := by
          apply mul_le_mul h_head_lt.le (h_tail_le_exp.trans h_tail_exp_le)
          · exact Finset.prod_nonneg h_tail_pos
          · norm_num

/-- Uniform bound: `gainFactor n ≤ 1.647` for all n.
    The chain: `gainFactor n ^ 2 ≤ 2.71234 · exp(4⁻¹⁵/3)` (from `partialProd_le`)
    `≤ 2.71234 · (1 + 2·1e-9)` (via `Real.abs_exp_sub_one_le`)
    `≤ 2.71235 < 2.713209 = 1.647²`,
    so `gainFactor n ≤ 1.647` via `pow_le_pow_iff_left`. -/
private lemma gainFactor_le_1647 (n : ℕ) : gainFactor n ≤ 1.647 := by
  have h_pos : 0 ≤ gainFactor n := (gain_factor_pos n).le
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
  have h_x_tiny : ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) ≤ 1e-9 := by
    have : (4 : ℝ)⁻¹ ^ (15 : ℕ) = 1 / 4^15 := by
      rw [inv_pow, one_div]
    rw [this]; norm_num
  have h_final_sq : gainFactor n ^ 2 ≤ (1.647 : ℝ) ^ 2 := by
    have h_step : 2.71234 * Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) ≤
        2.71234 * (1 + 2 * 1e-9) := by
      apply mul_le_mul_of_nonneg_left _ (by norm_num : (0:ℝ) ≤ 2.71234)
      calc Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3)
          ≤ 1 + 2 * ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := h_exp_close
        _ ≤ 1 + 2 * 1e-9 := by linarith
    calc gainFactor n ^ 2 ≤ 2.71234 * Real.exp ((4 : ℝ)⁻¹ ^ (15 : ℕ) / 3) := h_sq_le
      _ ≤ 2.71234 * (1 + 2 * 1e-9) := h_step
      _ ≤ (1.647 : ℝ) ^ 2 := by norm_num
  exact (pow_le_pow_iff_left h_pos (by norm_num) (by norm_num : (2 : ℕ) ≠ 0)).mp h_final_sq

/-- Final assembly. -/
theorem gain_approaches_constant :
    ∃ K : ℝ, 1.6 < K ∧ K < 1.65 ∧
    Filter.Tendsto gainFactor Filter.atTop (nhds K) := by
  have h_mono : Monotone gainFactor := gainFactor_mono
  have h_bdd : BddAbove (Set.range gainFactor) := by
    refine ⟨1.647, ?_⟩
    rintro _ ⟨n, rfl⟩
    exact gainFactor_le_1647 n
  refine ⟨⨆ n, gainFactor n, ?_, ?_, tendsto_atTop_ciSup h_mono h_bdd⟩
  · -- 1.6 < ⨆: chain through gainFactor 16 > 1.6.
    calc (1.6 : ℝ) < gainFactor 16 := gainFactor_16_gt
      _ ≤ ⨆ n, gainFactor n := le_ciSup h_bdd 16
  · -- ⨆ ≤ 1.647 < 1.65: uniform bound gives the strict <.
    calc ⨆ n, gainFactor n ≤ 1.647 := ciSup_le gainFactor_le_1647
      _ < 1.65 := by norm_num

end RAC.Cordic.PrecisionKnob
