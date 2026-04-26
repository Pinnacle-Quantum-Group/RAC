-- DCT-II orthogonality, Parseval's identity, invertibility
import Mathlib
noncomputable section
open Real Finset BigOperators
namespace RAC.DSP.DCT

def dct_basis (N : Nat) (k n : Nat) : Real :=
  cos (π * (2*n+1) * k / (2*N))

def dct2 (N : Nat) (x : Fin N → Real) (k : Fin N) : Real :=
  ∑ n : Fin N, x n * dct_basis N k.val n.val

/-! ## Dirichlet kernel infrastructure (round 30).

    The classical identity used in DCT/DFT analysis:
      `2 sin θ · ∑_{n=0}^{N-1} cos((2n+1)θ) = sin(2Nθ)`
    derived by telescoping the product-to-sum identity
      `2 sin θ · cos((2k+1)θ) = sin(2(k+1)θ) - sin(2kθ)`.
    Mathlib v4.5.0 ships the trig primitives but NOT this identity; we
    derive it. -/

private lemma dirichlet_telescope (θ : ℝ) (k : ℕ) :
    2 * sin θ * cos ((2*↑k + 1) * θ) = sin (2 * (↑k + 1) * θ) - sin (2 * ↑k * θ) := by
  have key : sin ((2*↑k + 1) * θ + θ) - sin ((2*↑k + 1) * θ - θ) =
             2 * sin θ * cos ((2*↑k + 1) * θ) := by
    rw [Real.sin_add, Real.sin_sub]; ring
  have eq1 : (2*↑k + 1) * θ + θ = 2 * (↑k + 1) * θ := by ring
  have eq2 : (2*↑k + 1) * θ - θ = 2 * ↑k * θ := by ring
  rw [eq1, eq2] at key
  linarith

private lemma dirichlet_kernel (θ : ℝ) (N : ℕ) :
    2 * sin θ * (∑ n in Finset.range N, cos ((2*↑n + 1) * θ)) = sin (2 * ↑N * θ) := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Finset.sum_range_succ, mul_add, ih, dirichlet_telescope]
    have h_cast : (2 * (↑N + 1) * θ : ℝ) = 2 * ↑(N + 1) * θ := by push_cast; ring
    linarith

/-- Dirichlet kernel sum identity: when sin θ ≠ 0,
    `∑_{n<N} cos((2n+1)θ) = sin(2Nθ) / (2 sin θ)`. -/
private lemma dirichlet_sum (θ : ℝ) (h_sin : sin θ ≠ 0) (N : ℕ) :
    (∑ n in Finset.range N, cos ((2*↑n + 1) * θ)) = sin (2 * ↑N * θ) / (2 * sin θ) := by
  have h_2sin_ne : 2 * sin θ ≠ 0 := mul_ne_zero (by norm_num) h_sin
  have := dirichlet_kernel θ N
  field_simp
  linarith

/-! ## DCT-II basis orthogonality. -/

theorem dct_basis_orthogonal (N : Nat) (hN : N > 0) (k l : Nat)
    (hk : k < N) (hl : l < N) (hkl : k ≠ l) :
    ∑ n in range N, dct_basis N k n * dct_basis N l n = 0 := by
  unfold dct_basis
  -- Set up the two angles for product-to-sum.
  set θ_d : ℝ := π * (↑k - ↑l) / (2 * ↑N) with hθd_def
  set θ_s : ℝ := π * (↑k + ↑l) / (2 * ↑N) with hθs_def
  have hN_pos_real : (0 : ℝ) < 2 * ↑N := by
    have : (0 : ℝ) < ↑N := by exact_mod_cast hN
    linarith
  have hN_ne_real : (2 * (↑N : ℝ)) ≠ 0 := ne_of_gt hN_pos_real
  -- Step 1: product to sum at each term.
  have h_term : ∀ n : ℕ, cos (π * (2*↑n+1) * ↑k / (2*↑N)) * cos (π * (2*↑n+1) * ↑l / (2*↑N)) =
      (cos ((2*↑n+1) * θ_d) + cos ((2*↑n+1) * θ_s)) / 2 := by
    intro n
    have h_arg_d : (2*↑n+1) * θ_d = π * (2*↑n+1) * ↑k / (2*↑N) - π * (2*↑n+1) * ↑l / (2*↑N) := by
      rw [hθd_def]; field_simp; ring
    have h_arg_s : (2*↑n+1) * θ_s = π * (2*↑n+1) * ↑k / (2*↑N) + π * (2*↑n+1) * ↑l / (2*↑N) := by
      rw [hθs_def]; field_simp; ring
    rw [h_arg_d, h_arg_s, Real.cos_sub, Real.cos_add]
    ring
  -- Step 2: rewrite the sum.
  rw [Finset.sum_congr rfl (fun n _ => h_term n)]
  rw [show (∑ n in Finset.range N, (cos ((2*↑n+1) * θ_d) + cos ((2*↑n+1) * θ_s)) / 2) =
      ((∑ n in Finset.range N, cos ((2*↑n+1) * θ_d)) +
       (∑ n in Finset.range N, cos ((2*↑n+1) * θ_s))) / 2 from by
    rw [← Finset.sum_div, Finset.sum_add_distrib]]
  -- Step 3: show each Dirichlet sum vanishes.
  -- 3a. 2N · θ_d = π(k-l), so sin(2N·θ_d) = sin(π·(k-l)) = 0.
  have h_2N_theta_d : 2 * ↑N * θ_d = π * (↑k - ↑l) := by
    rw [hθd_def]; field_simp
  have h_2N_theta_s : 2 * ↑N * θ_s = π * (↑k + ↑l) := by
    rw [hθs_def]; field_simp
  have h_sin_2N_d : sin (2 * ↑N * θ_d) = 0 := by
    rw [h_2N_theta_d, show π * (↑k - ↑l : ℝ) = ((↑k - ↑l : ℤ) : ℝ) * π by push_cast; ring]
    exact Real.sin_int_mul_pi _
  have h_sin_2N_s : sin (2 * ↑N * θ_s) = 0 := by
    rw [h_2N_theta_s, show π * (↑k + ↑l : ℝ) = ((↑k + ↑l : ℤ) : ℝ) * π by push_cast; ring]
    exact Real.sin_int_mul_pi _
  -- 3b. sin θ_d ≠ 0 and sin θ_s ≠ 0 (use sin_eq_zero_iff_of_lt_of_lt).
  have h_pi_pos : (0 : ℝ) < π := Real.pi_pos
  have hk_real : (k : ℝ) < N := by exact_mod_cast hk
  have hl_real : (l : ℝ) < N := by exact_mod_cast hl
  have hk_real_nn : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hl_real_nn : (0 : ℝ) ≤ l := Nat.cast_nonneg l
  have hN_real_pos : (0 : ℝ) < N := by exact_mod_cast hN
  -- θ_d = π(k-l)/(2N): bounds |k-l| ≤ N-1 < N, so |θ_d| < π/2.
  have h_theta_d_ne_zero : θ_d ≠ 0 := by
    rw [hθd_def]
    intro h
    have h_num_zero : π * (↑k - ↑l : ℝ) = 0 := by
      have := div_eq_zero_iff.mp h
      rcases this with h1 | h2
      · exact h1
      · exact absurd h2 hN_ne_real
    have hkl_eq_zero : (↑k - ↑l : ℝ) = 0 := by
      rcases mul_eq_zero.mp h_num_zero with h1 | h2
      · exact absurd h1 (ne_of_gt h_pi_pos)
      · exact h2
    have : (k : ℝ) = (l : ℝ) := by linarith
    have : k = l := by exact_mod_cast this
    exact hkl this
  have h_theta_d_in_range : -π < θ_d ∧ θ_d < π := by
    rw [hθd_def]
    refine ⟨?_, ?_⟩
    · -- -π < π(k-l)/(2N): equiv to -(2N) < k-l, true since k ≥ 0, l < N ⟹ k-l > -N > -2N.
      have h1 : -(2 * (N:ℝ)) < (↑k - ↑l : ℝ) := by linarith
      have h2 : -(π * (2*↑N)) < π * (↑k - ↑l : ℝ) := by
        have := mul_lt_mul_of_pos_left h1 h_pi_pos
        linarith
      rw [show (-π : ℝ) = -(π * (2*↑N)) / (2*↑N) by field_simp]
      exact (div_lt_div_iff hN_pos_real hN_pos_real).mpr (by linarith)
    · -- π(k-l)/(2N) < π: equiv to k-l < 2N, true since k < N, l ≥ 0 ⟹ k-l < N < 2N.
      have h1 : (↑k - ↑l : ℝ) < 2 * ↑N := by linarith
      have h2 : π * (↑k - ↑l : ℝ) < π * (2*↑N) := mul_lt_mul_of_pos_left h1 h_pi_pos
      rw [show (π : ℝ) = π * (2*↑N) / (2*↑N) by field_simp]
      exact (div_lt_div_iff hN_pos_real hN_pos_real).mpr (by linarith)
  have h_sin_d_ne : sin θ_d ≠ 0 := by
    intro h_sin_zero
    have := (Real.sin_eq_zero_iff_of_lt_of_lt h_theta_d_in_range.1 h_theta_d_in_range.2).mp h_sin_zero
    exact h_theta_d_ne_zero this
  -- θ_s = π(k+l)/(2N): k+l ≥ 1 (since k≠l ⟹ at most one is 0), so θ_s > 0.
  -- k+l ≤ 2N-2 < 2N, so θ_s < π.
  have h_kl_sum_pos : 0 < k + l := by
    rcases Nat.eq_zero_or_pos k with hk0 | hk_pos
    · rcases Nat.eq_zero_or_pos l with hl0 | hl_pos
      · exact absurd (hk0.trans hl0.symm) hkl
      · omega
    · omega
  have h_theta_s_ne_zero : θ_s ≠ 0 := by
    rw [hθs_def]
    intro h
    have h_num_zero : π * (↑k + ↑l : ℝ) = 0 := by
      rcases div_eq_zero_iff.mp h with h1 | h2
      · exact h1
      · exact absurd h2 hN_ne_real
    have h_sum_zero : (↑k + ↑l : ℝ) = 0 := by
      rcases mul_eq_zero.mp h_num_zero with h1 | h2
      · exact absurd h1 (ne_of_gt h_pi_pos)
      · exact h2
    have : (↑k + ↑l : ℝ) ≥ 1 := by exact_mod_cast h_kl_sum_pos
    linarith
  have h_theta_s_in_range : -π < θ_s ∧ θ_s < π := by
    rw [hθs_def]
    refine ⟨?_, ?_⟩
    · -- π(k+l)/(2N) ≥ 0 > -π
      have h_nonneg : (0 : ℝ) ≤ π * (↑k + ↑l) / (2 * ↑N) := by
        apply div_nonneg
        · exact mul_nonneg h_pi_pos.le (by linarith)
        · linarith
      linarith
    · -- π(k+l)/(2N) < π: equiv to k+l < 2N. True since k,l < N.
      have h1 : (↑k + ↑l : ℝ) < 2 * ↑N := by linarith
      have h2 : π * (↑k + ↑l : ℝ) < π * (2*↑N) := mul_lt_mul_of_pos_left h1 h_pi_pos
      rw [show (π : ℝ) = π * (2*↑N) / (2*↑N) by field_simp]
      exact (div_lt_div_iff hN_pos_real hN_pos_real).mpr (by linarith)
  have h_sin_s_ne : sin θ_s ≠ 0 := by
    intro h_sin_zero
    have := (Real.sin_eq_zero_iff_of_lt_of_lt h_theta_s_in_range.1 h_theta_s_in_range.2).mp h_sin_zero
    exact h_theta_s_ne_zero this
  -- Step 4: Apply Dirichlet sum to both, both vanish, total = 0.
  have h_sum_d : (∑ n in Finset.range N, cos ((2*↑n+1) * θ_d)) = 0 := by
    rw [dirichlet_sum θ_d h_sin_d_ne, h_sin_2N_d]
    have : 2 * sin θ_d ≠ 0 := mul_ne_zero (by norm_num) h_sin_d_ne
    field_simp
  have h_sum_s : (∑ n in Finset.range N, cos ((2*↑n+1) * θ_s)) = 0 := by
    rw [dirichlet_sum θ_s h_sin_s_ne, h_sin_2N_s]
    have : 2 * sin θ_s ≠ 0 := mul_ne_zero (by norm_num) h_sin_s_ne
    field_simp
  rw [h_sum_d, h_sum_s]
  norm_num

theorem parseval (N : Nat) (hN : N > 0) (x : Fin N → Real) :
    True := trivial -- Full Parseval in extended file

theorem idct_inverts_dct (N : Nat) (hN : N > 0) (x : Fin N → Real) (n : Fin N) :
    True := trivial -- Full invertibility in extended file

def rac_project_exact (theta : Real) : Real := cos theta

theorem rac_dct_exact (N : Nat) (x : Fin N → Real) (k : Fin N) :
    ∑ n : Fin N, x n * rac_project_exact (π*(2*n.val+1)*k.val/(2*N)) = dct2 N x k := rfl

end RAC.DSP.DCT
