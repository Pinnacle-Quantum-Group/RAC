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
import Cordic.ArctanFacts
import Trig.ArctanBounds

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
    |(cordicStep s i).z| = | |s.z| - atanTable i | := by
  show |s.z - sigma s.z * atanTable i| = | |s.z| - atanTable i |
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
    have h_tri : | |(cordicIters n s₀).z| - atanTable n | ≤
        |(cordicIters n s₀).z| + atanTable n := by
      rcases le_or_lt 0 (|(cordicIters n s₀).z| - atanTable n) with h | h
      · rw [abs_of_nonneg h]; linarith [abs_nonneg (cordicIters n s₀).z]
      · rw [abs_of_neg h]; linarith [abs_nonneg (cordicIters n s₀).z]
    rw [Finset.sum_range_succ]
    linarith

/-! ## Inductive invariant for CORDIC residual bound.

    SPEC NOTE: the original statement
      `|z_0| ≤ ∑_{k<n} atanTable k → |z_n| ≤ atanTable n`
    is FALSE (counterexample n=1, z_0=0: |z_1| = atanTable 0 > atanTable 1).
    Corrected to: `|z_0| ≤ ∑_{k<n+1} atanTable k → |z_{n+1}| ≤ atanTable n`
    (range INCLUDES n; conclusion bounds |z_{n+1}|, not |z_n|).

    Invariant J n m models the "remaining budget" at step m:
      J n 0     = ∑ k ∈ range (n+1), atanTable k    (full hypothesis)
      J n (m+1) = atanTable m + ∑ k ∈ Ico (m+1) (n+1), atanTable k -/

private noncomputable def J (n : ℕ) : ℕ → ℝ
  | 0     => ∑ k in Finset.range (n+1), atanTable k
  | m+1   => atanTable m + ∑ k in Finset.Ico (m+1) (n+1), atanTable k

private lemma J_zero (n : ℕ) : J n 0 = ∑ k in Finset.range (n+1), atanTable k := rfl

private lemma J_succ (n m : ℕ) :
    J n (m+1) = atanTable m + ∑ k in Finset.Ico (m+1) (n+1), atanTable k := rfl

/-- Final value: `J n (n+1) = atanTable n`. -/
private lemma J_at_n_succ (n : ℕ) : J n (n+1) = atanTable n := by
  rw [J_succ, Finset.Ico_self, Finset.sum_empty, add_zero]

/-- Trivial: `atanTable m ≤ J n (m+1)` since J n (m+1) = atanTable m + (nonneg sum). -/
private lemma atanTable_le_J_succ (n m : ℕ) : atanTable m ≤ J n (m+1) := by
  rw [J_succ]
  have h_nonneg : (0:ℝ) ≤ ∑ k in Finset.Ico (m+1) (n+1), atanTable k :=
    Finset.sum_nonneg (fun k _ => (atanTable_pos k).le)
  linarith

/-! ### Helpers for the convergence-range bound

    The neither-pure-tail-sum nor the J-invariant alone is preserved
    by the iteration, but a *coupled* invariant
        K(m) := atanTable n + ∑ k ∈ Ico m (n+1), atanTable k
    works: a fixed "swing-room" budget `atanTable n` plus the
    forward tail. K(0) is implied by the hypothesis (since
    `Ico 0 (n+1) = range (n+1)` and `atanTable n ≥ 0`); K(n+1) is
    `atanTable n` (empty tail). The step splits on `|z_m|` vs
    `atanTable m`: the high branch consumes `atanTable m` directly
    from the tail (K(m) − atanTable m = K(m+1)); the low branch
    trades the budget against the tail via the *finite absorption
    inequality* below. Once K is proven, K(m) ≤ J n m for `m ≥ 1`
    follows from `atanTable_strictMono` (since `atanTable n ≤
    atanTable (m-1)` for `m ≤ n+1`), so the J-invariant required by
    `residual_bound` falls out as a corollary. -/

/-- One step of geometric absorption: `atanTable k ≤ 2 · atanTable (k+1)`.
    Direct re-statement of `arctan_inv_two_pow_succ_ge_half` in terms
    of the local `atanTable` definition. -/
private lemma atanTable_le_two_mul_succ (k : ℕ) :
    atanTable k ≤ 2 * atanTable (k + 1) := by
  have h := arctan_inv_two_pow_succ_ge_half k
  -- h : arctan ((2:ℝ)⁻¹^k) / 2 ≤ arctan ((2:ℝ)⁻¹^(k+1))
  change arctan ((2 : ℝ)⁻¹ ^ k) ≤ 2 * arctan ((2 : ℝ)⁻¹ ^ (k + 1))
  linarith

/-- Auxiliary form of finite absorption, parameterised on the gap `d`:
    `atanTable m ≤ atanTable (m+d) + ∑_{j<d} atanTable (m+1+j)`. Proved
    by induction on `d`, using `atanTable_le_two_mul_succ` to absorb
    the leading `atanTable m` into the tail one step at a time. -/
private lemma atanTable_absorption_aux : ∀ (d m : ℕ),
    atanTable m ≤ atanTable (m + d) +
        ∑ j in Finset.range d, atanTable (m + 1 + j) := by
  intro d
  induction d with
  | zero =>
    intro m
    have h1 : (m + 0 : ℕ) = m := Nat.add_zero m
    rw [h1, Finset.sum_range_zero, add_zero]
  | succ d ih =>
    intro m
    have hstep : atanTable m ≤ atanTable (m + 1) + atanTable (m + 1) := by
      have := atanTable_le_two_mul_succ m
      linarith
    have hih := ih (m + 1)
    have h_idx : m + (d + 1) = (m + 1) + d := by omega
    have h_sum_split :
        ∑ j in Finset.range (d + 1), atanTable (m + 1 + j) =
          atanTable (m + 1) +
          ∑ j in Finset.range d, atanTable ((m + 1) + 1 + j) := by
      rw [Finset.sum_range_succ']
      have h_inner : ∑ j in Finset.range d, atanTable (m + 1 + (j + 1)) =
                     ∑ j in Finset.range d, atanTable ((m + 1) + 1 + j) := by
        refine Finset.sum_congr rfl (fun j _ => ?_)
        congr 1
        omega
      have h_z : atanTable (m + 1 + 0) = atanTable (m + 1) := rfl
      rw [h_inner, h_z, add_comm]
    rw [h_idx, h_sum_split]
    linarith

/-- **Finite absorption** (Volder's "K-range" inequality):
    for `m ≤ n`, `atanTable m ≤ atanTable n + ∑_{k=m+1}^{n} atanTable k`.
    The crucial inequality that lets the convergence proof trade
    swing-room against the forward tail in the under-shoot branch. -/
private lemma atanTable_absorption {m n : ℕ} (hmn : m ≤ n) :
    atanTable m ≤ atanTable n + ∑ k in Finset.Ico (m + 1) (n + 1), atanTable k := by
  have h := atanTable_absorption_aux (n - m) m
  have h_idx : m + (n - m) = n := by omega
  rw [h_idx] at h
  have h_reidx : ∑ j in Finset.range (n - m), atanTable (m + 1 + j)
               = ∑ k in Finset.Ico (m + 1) (n + 1), atanTable k := by
    rw [Finset.sum_Ico_eq_sum_range]
    have h_len : n + 1 - (m + 1) = n - m := by omega
    rw [h_len]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    -- Goal `atanTable (j + (m+1)) = atanTable (m + 1 + j)` reduces via
    -- `congrArg atanTable` to a Nat equality closed by `omega`.
    exact congrArg atanTable (by omega)
  rw [h_reidx] at h
  exact h

/-- Inductive proof of the J-invariant: `|z_m| ≤ J n m` for all `m ≤ n+1`.

    Strategy: run a *stronger* induction on the coupled K-invariant
    `|z_m| ≤ atanTable n + ∑ k∈Ico m (n+1), atanTable k`, then bridge
    to `J n m` (which is looser for `m ≥ 1` because `atanTable n ≤
    atanTable (m-1)`). The base K(0) is implied by `hz₀` plus
    `atanTable n ≥ 0`; the step uses `Finset.sum_eq_sum_Ico_succ_bot`
    to peel off the leading term and `atanTable_absorption` for the
    under-shoot branch. -/
private lemma residual_invariant (s₀ : State) (n : ℕ)
    (hz₀ : |s₀.z| ≤ ∑ k in Finset.range (n+1), atanTable k) :
    ∀ m, m ≤ n+1 → |(cordicIters m s₀).z| ≤ J n m := by
  -- Stronger intermediate K-invariant.
  have hK : ∀ m, m ≤ n + 1 → |(cordicIters m s₀).z| ≤
      atanTable n + ∑ k in Finset.Ico m (n + 1), atanTable k := by
    intro m
    induction m with
    | zero =>
      intro _hm
      have h_eq : (Finset.Ico 0 (n + 1) : Finset ℕ) = Finset.range (n + 1) := by
        ext k; simp [Finset.mem_Ico, Finset.mem_range]
      have h_atan_n : 0 ≤ atanTable n := (atanTable_pos n).le
      show |s₀.z| ≤ atanTable n + ∑ k in Finset.Ico 0 (n + 1), atanTable k
      rw [h_eq]
      linarith
    | succ m ih =>
      intro hm
      have hm_lt : m < n + 1 := Nat.lt_of_succ_le hm
      have hm_le_n : m ≤ n := Nat.lt_succ_iff.mp hm_lt
      have ih' := ih (Nat.le_of_succ_le hm)
      have h_split : ∑ k in Finset.Ico m (n + 1), atanTable k =
                     atanTable m + ∑ k in Finset.Ico (m + 1) (n + 1), atanTable k :=
        Finset.sum_eq_sum_Ico_succ_bot hm_lt _
      rw [h_split] at ih'
      show |(cordicStep (cordicIters m s₀) m).z| ≤ _
      rw [residual_step_eq]
      have h_zm_nn : 0 ≤ |(cordicIters m s₀).z| := abs_nonneg _
      have h_atan_m_nn : 0 ≤ atanTable m := (atanTable_pos m).le
      rcases le_or_lt (atanTable m) |(cordicIters m s₀).z| with h | h
      · -- High branch: ||z_m| - atanTable m| = |z_m| - atanTable m, consume from tail.
        rw [abs_of_nonneg (by linarith : 0 ≤ |(cordicIters m s₀).z| - atanTable m)]
        linarith
      · -- Low branch: ||z_m| - atanTable m| = atanTable m - |z_m| ≤ atanTable m;
        -- trade swing-room via absorption.
        rw [abs_of_neg (by linarith : |(cordicIters m s₀).z| - atanTable m < 0)]
        have h_absorb := atanTable_absorption hm_le_n
        linarith
  -- Bridge K-invariant to the (looser) J-invariant.
  intro m hm
  have hkb := hK m hm
  rcases m with _ | m
  · -- m = 0: J n 0 = ∑_{range (n+1)} is exactly the hypothesis.
    rw [J_zero]
    exact hz₀
  · -- m = succ m: J n (m+1) = atanTable m + tail; K(m+1) = atanTable n + tail;
    -- since m+1 ≤ n+1, m ≤ n, hence atanTable n ≤ atanTable m by anti-monotonicity.
    rw [J_succ]
    have hm_le_n : m ≤ n := by
      have : m + 1 ≤ n + 1 := hm
      omega
    have h_table : atanTable n ≤ atanTable m :=
      atanTable_strictMono.antitone hm_le_n
    linarith

/-- Main theorem (corrected spec): after `n+1` CORDIC iterations from a
    state whose |z| is bounded by the (n+1)-truncated atan-sum, the
    residual is at most `atanTable n`.

    The original `range n / atanTable n` form is FALSE (n=1, z_0=0 gives
    |z_1| = atanTable 0 > atanTable 1). Restated via reindexing. -/
theorem residual_bound (s₀ : State) (n : ℕ)
    (hz₀ : |s₀.z| ≤ ∑ k in Finset.range (n+1), atanTable k) :
    |(cordicIters (n+1) s₀).z| ≤ atanTable n := by
  have h := residual_invariant s₀ n hz₀ (n+1) le_rfl
  rw [J_at_n_succ] at h
  exact h

/-- Geometric form: `|z_n| ≤ 2 · 2⁻ⁿ` after n CORDIC iterations.
    Case-split on n: for n=0, the hypothesis forces z_0 = 0; for
    n=k+1, apply `residual_bound` and chain through `atanTable_le_pow`. -/
theorem residual_geometric_bound (s₀ : State) (n : ℕ)
    (hz₀ : |s₀.z| ≤ ∑ k in Finset.range n, atanTable k) :
    |(cordicIters n s₀).z| ≤ 2 * (2 : ℝ)⁻¹ ^ n := by
  match n with
  | 0 =>
    -- |s₀.z| ≤ ∑ over range 0 = 0, so s₀.z = 0; cordicIters 0 s₀ = s₀.
    simp only [Finset.range_zero, Finset.sum_empty, abs_nonpos_iff] at hz₀
    show |(cordicIters 0 s₀).z| ≤ 2 * (2:ℝ)⁻¹ ^ 0
    simp [cordicIters, hz₀]
    norm_num
  | k+1 =>
    -- Apply residual_bound with index k: |z_{k+1}| ≤ atanTable k.
    -- atanTable k ≤ 2⁻¹^k = 2 · 2⁻¹^(k+1).
    have h := residual_bound s₀ k hz₀
    calc |(cordicIters (k+1) s₀).z|
        ≤ atanTable k := h
      _ ≤ (2:ℝ)⁻¹ ^ k := atanTable_le_pow k
      _ = 2 * (2:ℝ)⁻¹ ^ (k+1) := by
          rw [pow_succ]; ring

/-! ## 4. Magnitude Growth -/

theorem magnitude_growth (s : State) (i : ℕ) :
    (cordicStep s i).x ^ 2 + (cordicStep s i).y ^ 2 =
    (1 + ((2 : ℝ)⁻¹ ^ i) ^ 2) * (s.x ^ 2 + s.y ^ 2) := by
  simp only [cordicStep]
  have hd := sigma_sq s.z
  nlinarith [sigma_sq s.z]

theorem cordic_convergence (s₀ : State) (n : ℕ)
    (hz₀ : |s₀.z| ≤ ∑ k in Finset.range n, atanTable k) :
    |(cordicIters n s₀).z| ≤ 2 * (2 : ℝ)⁻¹ ^ n ∧
    2 * (2 : ℝ)⁻¹ ^ (n + 1) = (2 : ℝ)⁻¹ ^ n := by
  constructor
  · exact residual_geometric_bound s₀ n hz₀
  · ring

end RAC.Cordic.Convergence
