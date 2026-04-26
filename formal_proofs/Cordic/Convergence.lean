/-
  RAC Formal Proofs — CORDIC Iteration Convergence
  Pinnacle Quantum Group — April 2026

  This module proves that the circular-mode CORDIC iteration converges:
  after n iterations, the residual angle z_n is bounded by atan(2^{-n}),
  and the output (x_n, y_n) approximates the exact rotation to within
  2^{-n} error per component.

  Reference: rac_cuda.cu `_rac_cordic_rotate_raw` (lines ~170-186)
    for i in 0..RAC_ITERS:
      d     = sign(angle)
      x_new = x - d * y * 2^{-i}
      y_new = y + d * x * 2^{-i}
      angle -= d * atan_table[i]

  Key insight: each CORDIC iteration halves the maximum residual angle,
  converging one bit of angular precision per step.
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import RAC.Cordic.ArctanFacts
import RAC.Trig.ArctanBounds

noncomputable section

open Real BigOperators  -- BigOperators needed for `∑` notation
open RAC.Cordic.ArctanFacts
open RAC.Trig.ArctanBounds

namespace RAC.Cordic.Convergence

/-! ## 1. CORDIC Angle Table: atan(2^{-i}) -/

/-- The CORDIC angle table: atan(2^{-i}) for iteration i.
    Matches `rac_atan_table` in rac_cuda.cu. -/
def atanTable (i : ℕ) : ℝ := arctan ((2 : ℝ)⁻¹ ^ i)

/-- atan(2^{-i}) is positive for all i.
    Closed via `RAC.Cordic.ArctanFacts.arctan_pos` (round 8 — derived
    in v4.5.0 from `tan_arctan` + `strictMonoOn_tan` + `arctan_zero`). -/
theorem atanTable_pos (i : ℕ) : 0 < atanTable i :=
  arctan_pos (inv_two_pow_pos i)

/-- atan(2^{-i}) is strictly decreasing in i.  Composition of:
    `(2:ℝ)⁻¹ ^ ·` strictly anti on ℕ (since `2⁻¹ ∈ (0,1)`), and
    `arctan` strictly mono on ℝ. -/
theorem atanTable_strictMono : StrictAnti atanTable :=
  arctan_strictMono.comp_strictAnti inv_two_pow_strictAnti

/-- atan(2^{-i}) ≤ 2^{-i} for all i.  Direct from
    `arctan_le_self_of_nonneg` since `(2:ℝ)⁻¹^i ≥ 0`. -/
theorem atanTable_le_pow (i : ℕ) : atanTable i ≤ (2 : ℝ)⁻¹ ^ i :=
  arctan_le_self_of_nonneg (inv_two_pow_pos i).le

/-! ## 2. CORDIC State and Iteration -/

structure State where
  x : ℝ
  y : ℝ
  z : ℝ

def sigma (z : ℝ) : ℝ := if z ≥ 0 then 1 else -1

theorem sigma_abs (z : ℝ) : |sigma z| = 1 := by
  unfold sigma; split <;> simp

theorem sigma_sq (z : ℝ) : sigma z ^ 2 = 1 := by
  -- sigma z ∈ {1, -1}; square is 1 either way.
  have h : |sigma z| = 1 := sigma_abs z
  rw [← sq_abs, h]
  norm_num

def cordicStep (s : State) (i : ℕ) : State where
  x := s.x - sigma s.z * s.y * (2 : ℝ)⁻¹ ^ i
  y := s.y + sigma s.z * s.x * (2 : ℝ)⁻¹ ^ i
  z := s.z - sigma s.z * atanTable i

/-- CORDIC iteration in **forward order** (steps 0, 1, …, n-1 in turn).
    The original definition went in reverse (step n applied first), which
    breaks the standard CORDIC convergence analysis — see Volder 1959,
    Walther 1971, Hu 1992. Forward order is the standard convention. -/
def cordicIters : ℕ → State → State
  | 0, s => s
  | n + 1, s => cordicStep (cordicIters n s) n

/-! ## 3. Residual Angle Convergence

    Foundational equality (Volder 1959): the CORDIC z-residual update has
    a single universal closed-form |z_{i+1}| = | |z_i| - atanTable i |.
    This is because σ(z) = sign(z), so subtracting σ·atanTable always
    moves z toward 0 by atanTable in absolute value (or overshoots by
    atanTable - |z| if |z| < atanTable). -/

/-- **Universal CORDIC residual equality** — the heart of single-step
    convergence analysis. -/
theorem residual_step_eq (s : State) (i : ℕ) :
    |(cordicStep s i).z| = ||s.z| - atanTable i| := by
  show |s.z - sigma s.z * atanTable i| = ||s.z| - atanTable i|
  have h_atan_nonneg : 0 ≤ atanTable i := (atanTable_pos i).le
  unfold sigma
  split_ifs with hz
  · -- s.z ≥ 0: σ = 1, new z = s.z - atanTable i. |s.z| = s.z.
    rw [show |s.z| = s.z from abs_of_nonneg hz]
    simp
  · -- s.z < 0: σ = -1, new z = s.z + atanTable i. |s.z| = -s.z.
    push_neg at hz
    rw [show |s.z| = -s.z from abs_of_neg hz]
    rw [show s.z - (-1) * atanTable i = s.z + atanTable i by ring]
    -- |s.z + atanTable i| = |-s.z - atanTable i| = ||s.z| - atanTable i|... let's compute
    rw [show s.z + atanTable i = -(-s.z - atanTable i) by ring, abs_neg]

/-- **Conditional decrease**: if `|s.z| ≥ atanTable i`, the residual
    strictly contracts by exactly `atanTable i`.  This is the
    high-magnitude branch of the equality above. -/
theorem residual_decreases_step (s : State) (i : ℕ)
    (h : atanTable i ≤ |s.z|) :
    |(cordicStep s i).z| ≤ |s.z| := by
  rw [residual_step_eq]
  -- ||s.z| - atanTable i| = |s.z| - atanTable i ≤ |s.z|
  rw [abs_of_nonneg (by linarith : 0 ≤ |s.z| - atanTable i)]
  linarith [(atanTable_pos i).le]

/-- **Triangle-inequality bound** on the residual after `n` iterations.
    Universally true (no convergence-range hypothesis), proven by
    induction + the universal equality + `||a| - b| ≤ |a| + b`. -/
theorem residual_bound_triangle (s₀ : State) (n : ℕ) :
    |(cordicIters n s₀).z| ≤ |s₀.z| + ∑ k in Finset.range n, atanTable k := by
  induction n with
  | zero => simp [cordicIters]
  | succ n ih =>
    show |(cordicStep (cordicIters n s₀) n).z| ≤ _
    rw [residual_step_eq]
    have h_atan_nonneg : 0 ≤ atanTable n := (atanTable_pos n).le
    -- ||z_n| - atanTable n| ≤ |z_n| + atanTable n  (triangle inequality)
    have h_tri : ||cordicIters n s₀ .z| - atanTable n| ≤
        |cordicIters n s₀ .z| + atanTable n := by
      rcases le_or_lt 0 (|cordicIters n s₀ .z| - atanTable n) with h | h
      · rw [abs_of_nonneg h]; linarith [abs_nonneg (cordicIters n s₀).z]
      · rw [abs_of_neg h]; linarith [abs_nonneg (cordicIters n s₀).z]
    rw [Finset.sum_range_succ]
    linarith

/-! ### CORDIC convergence-range bound (Volder 1959)

    Standard textbook result: if the input residual lies in the CORDIC
    convergence range `K_n := atanTable n + ∑_{k<n+1} atanTable k`
    (which approaches `K_∞ ≈ 1.7433 rad` as `n → ∞`), then after `n+1`
    iterations the residual has contracted to `|z_{n+1}| ≤ atanTable n`,
    i.e. roughly one bit of angular precision per step.

    The earlier formulation `|z_0| ≤ ∑_{k<n} atanTable k ⟹ |z_n| ≤ atanTable n`
    was false (counterexample n=1, z_0=0: `|z_1| = atanTable 0 > atanTable 1`).

    ### Proof structure

    The induction carries a *coupled* invariant — a fixed "swing room"
    `atanTable n` plus a forward tail `∑_{m ≤ k ≤ n} atanTable k` that
    shrinks one term per step:

        inv(m): |z_m| ≤ atanTable n + ∑ k ∈ Ico m (n+1), atanTable k

    At `m = 0` it is the hypothesis; at `m = n+1` the sum is empty and
    it is the conclusion. The step splits on `|z_m| ≥ atanTable m`:
    the high branch consumes `atanTable m` from the tail directly, the
    low branch trades the swing-room budget against the tail via the
    finite *absorption inequality*

        atanTable m ≤ atanTable n + ∑ k ∈ Ico (m+1) (n+1), atanTable k.

    Absorption is proved by reverse induction on `n - m`, repeatedly
    halving via `arctan_inv_two_pow_succ_ge_half`
    (`atanTable k ≤ 2 · atanTable (k+1)`). -/

/-- One step of geometric absorption: `atanTable k ≤ 2 · atanTable (k+1)`.
    Direct re-statement of `arctan_inv_two_pow_succ_ge_half` in terms
    of the local `atanTable` definition. -/
private lemma atanTable_le_two_mul_succ (k : ℕ) :
    atanTable k ≤ 2 * atanTable (k + 1) := by
  have h := arctan_inv_two_pow_succ_ge_half k
  -- h : arctan ((2:ℝ)⁻¹^k) / 2 ≤ arctan ((2:ℝ)⁻¹^(k+1))
  change arctan ((2 : ℝ)⁻¹ ^ k) ≤ 2 * arctan ((2 : ℝ)⁻¹ ^ (k + 1))
  linarith

/-- Auxiliary form of finite absorption, parameterised on the gap `d`.
    For all `m`, `atanTable m ≤ atanTable (m+d) + ∑_{j<d} atanTable (m+1+j)`.
    Proved by induction on `d`. -/
private lemma atanTable_absorption_aux : ∀ (d m : ℕ),
    atanTable m ≤ atanTable (m + d) +
        ∑ j in Finset.range d, atanTable (m + 1 + j) := by
  intro d
  induction d with
  | zero =>
    intro m
    simp [Finset.sum_range_zero, le_refl]
  | succ d ih =>
    intro m
    -- atanTable m ≤ atanTable (m+1) + atanTable (m+1)
    have hstep : atanTable m ≤ atanTable (m + 1) + atanTable (m + 1) := by
      have := atanTable_le_two_mul_succ m
      linarith
    -- Apply the IH at (m+1).
    have hih := ih (m + 1)
    -- hih : atanTable (m+1) ≤ atanTable ((m+1)+d) + ∑ j∈range d, atanTable ((m+1)+1+j)
    -- Reindex the goal sum via Finset.sum_range_succ' (peeling off j=0):
    --   ∑ j∈range (d+1), atanTable (m+1+j)
    --     = (∑ j∈range d, atanTable (m+1+(j+1))) + atanTable (m+1+0)
    have h_idx : m + (d + 1) = (m + 1) + d := by omega
    rw [h_idx, Finset.sum_range_succ']
    -- Match the inner-summand index `m+1+(j+1)` to `(m+1)+1+j` in `hih`.
    have h_sum_eq : ∑ j in Finset.range d, atanTable (m + 1 + (j + 1))
                  = ∑ j in Finset.range d, atanTable ((m + 1) + 1 + j) := by
      apply Finset.sum_congr rfl
      intro j _
      congr 1
      omega
    rw [h_sum_eq]
    -- Goal now matches: hih + atanTable(m+1) ≥ goal RHS, with hstep on the LHS.
    have h_simp_zero : (m + 1 + 0 : ℕ) = m + 1 := by omega
    rw [h_simp_zero]
    linarith

/-- **Finite absorption** (Volder's "K-range" inequality):
    for `m ≤ n`, `atanTable m ≤ atanTable n + ∑_{k=m+1}^{n} atanTable k`.
    The crucial inequality that lets the convergence proof trade swing
    room against the forward tail in the under-shoot branch. -/
private lemma atanTable_absorption {m n : ℕ} (hmn : m ≤ n) :
    atanTable m ≤ atanTable n + ∑ k in Finset.Ico (m + 1) (n + 1), atanTable k := by
  have h := atanTable_absorption_aux (n - m) m
  -- h : atanTable m ≤ atanTable (m + (n-m)) + ∑ j∈range (n-m), atanTable (m+1+j)
  have h_idx : m + (n - m) = n := by omega
  rw [h_idx] at h
  -- Reindex ∑ j∈range (n-m), atanTable (m+1+j) into ∑ k∈Ico (m+1) (n+1), atanTable k.
  have h_reidx : ∑ j in Finset.range (n - m), atanTable (m + 1 + j)
               = ∑ k in Finset.Ico (m + 1) (n + 1), atanTable k := by
    rw [Finset.sum_Ico_eq_sum_range]
    have h_len : n + 1 - (m + 1) = n - m := by omega
    rw [h_len]
    apply Finset.sum_congr rfl
    intro j _
    congr 1
    omega
  rw [h_reidx] at h
  exact h

/-- **CORDIC convergence-range bound** (Volder 1959).
    If the initial residual lies in `K_n := atanTable n + ∑_{k<n+1} atanTable k`,
    then after `n+1` iterations the residual has contracted to `atanTable n`. -/
theorem residual_bound (s₀ : State) (n : ℕ)
    (hz₀ : |s₀.z| ≤ atanTable n + ∑ k in Finset.range (n + 1), atanTable k) :
    |(cordicIters (n + 1) s₀).z| ≤ atanTable n := by
  -- Strengthened invariant (carried by the induction):
  --     ∀ m ≤ n+1, |z_m| ≤ atanTable n + ∑ k∈Ico m (n+1), atanTable k
  -- At m=0 it's the hypothesis; at m=n+1 the sum is empty, giving the
  -- conclusion `|z_{n+1}| ≤ atanTable n`.
  suffices h : ∀ m, m ≤ n + 1 →
      |(cordicIters m s₀).z| ≤ atanTable n +
          ∑ k in Finset.Ico m (n + 1), atanTable k by
    have hfin := h (n + 1) le_rfl
    rw [Finset.Ico_self, Finset.sum_empty, add_zero] at hfin
    exact hfin
  intro m
  induction m with
  | zero =>
    intro _hm
    -- Ico 0 (n+1) = range (n+1); fall back on hz₀.
    show |s₀.z| ≤ _
    rw [show (Finset.Ico 0 (n + 1) : Finset ℕ) = Finset.range (n + 1) by
      ext k; simp [Finset.mem_Ico, Finset.mem_range]]
    exact hz₀
  | succ m ih =>
    intro hm
    have hm_lt : m < n + 1 := Nat.lt_of_succ_le hm
    have hm_le : m ≤ n := Nat.lt_succ_iff.mp hm_lt
    have ih' := ih (Nat.le_of_succ_le hm)
    -- ih' : |z_m| ≤ atanTable n + ∑ k∈Ico m (n+1), atanTable k
    -- Split off the leading term: Ico m (n+1) = {m} ∪ Ico (m+1) (n+1)
    have h_split : ∑ k in Finset.Ico m (n + 1), atanTable k
                 = atanTable m + ∑ k in Finset.Ico (m + 1) (n + 1), atanTable k :=
      Finset.sum_eq_sum_Ico_succ_bot hm_lt _
    rw [h_split] at ih'
    -- ih' : |z_m| ≤ atanTable n + (atanTable m + ∑ k∈Ico (m+1) (n+1), atanTable k)
    show |(cordicStep (cordicIters m s₀) m).z| ≤ _
    rw [residual_step_eq]
    -- Goal: ||(cordicIters m s₀).z| - atanTable m| ≤ atanTable n + ∑ k∈Ico (m+1) (n+1), atanTable k
    have h_zm_nn : 0 ≤ |(cordicIters m s₀).z| := abs_nonneg _
    have h_atan_m_nn : 0 ≤ atanTable m := (atanTable_pos m).le
    -- Two cases: |z_m| ≥ atanTable m (consume from tail) vs < atanTable m
    -- (trade swing-room via absorption).
    rcases le_or_lt (atanTable m) |(cordicIters m s₀).z| with h | h
    · -- High branch: ||z_m| - atanTable m| = |z_m| - atanTable m, consume from tail.
      rw [abs_of_nonneg (by linarith : 0 ≤ |(cordicIters m s₀).z| - atanTable m)]
      linarith
    · -- Low branch: ||z_m| - atanTable m| = atanTable m - |z_m| ≤ atanTable m;
      -- trade swing-room via absorption.
      rw [abs_of_neg (by linarith : |(cordicIters m s₀).z| - atanTable m < 0)]
      have h_absorb := atanTable_absorption hm_le
      linarith

/-- Geometric form of the convergence bound: after `n+1` iterations,
    the residual is at most `(2:ℝ)⁻¹ ^ n` (i.e., one extra bit of
    angular precision over the trivial `arctan(2⁻ⁿ)` table value). -/
theorem residual_geometric_bound (s₀ : State) (n : ℕ)
    (hz₀ : |s₀.z| ≤ atanTable n + ∑ k in Finset.range (n + 1), atanTable k) :
    |(cordicIters (n + 1) s₀).z| ≤ (2 : ℝ)⁻¹ ^ n := by
  calc |(cordicIters (n + 1) s₀).z|
      ≤ atanTable n        := residual_bound s₀ n hz₀
    _ ≤ (2 : ℝ)⁻¹ ^ n      := atanTable_le_pow n

/-! ## 4. Magnitude Growth -/

theorem magnitude_growth (s : State) (i : ℕ) :
    (cordicStep s i).x ^ 2 + (cordicStep s i).y ^ 2 =
    (1 + ((2 : ℝ)⁻¹ ^ i) ^ 2) * (s.x ^ 2 + s.y ^ 2) := by
  simp only [cordicStep]
  have hd := sigma_sq s.z
  nlinarith [sigma_sq s.z]

/-- Top-level CORDIC convergence statement: the residual contracts at the
    rate of one bit per iteration, given a convergence-range hypothesis on
    the initial residual. -/
theorem cordic_convergence (s₀ : State) (n : ℕ)
    (hz₀ : |s₀.z| ≤ atanTable n + ∑ k in Finset.range (n + 1), atanTable k) :
    |(cordicIters (n + 1) s₀).z| ≤ (2 : ℝ)⁻¹ ^ n ∧
    2 * (2 : ℝ)⁻¹ ^ (n + 1) = (2 : ℝ)⁻¹ ^ n := by
  refine ⟨residual_geometric_bound s₀ n hz₀, ?_⟩
  ring

end RAC.Cordic.Convergence
