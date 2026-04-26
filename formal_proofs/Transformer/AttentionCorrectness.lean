-- Scaled dot-product attention: weights sum to 1, causal mask, scaling
import Mathlib
noncomputable section
open Real Finset BigOperators
namespace RAC.Transformer.Attention

def softmax_weights (scores : Fin n → ℝ) (hpos : ∀ i, 0 < exp (scores i)) (i : Fin n) : ℝ :=
  exp (scores i) / ∑ j, exp (scores j)

theorem weights_sum_one (scores : Fin n → ℝ) (hn : 0 < n)
    (hpos : ∀ i, 0 < exp (scores i)) :
    ∑ i, softmax_weights scores hpos i = 1 := by
  unfold softmax_weights
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (Finset.sum_pos (fun j _ => hpos j) ⟨⟨0, hn⟩, mem_univ _⟩))

theorem weights_nonneg (scores : Fin n → ℝ) (hpos : ∀ i, 0 < exp (scores i)) (i : Fin n) :
    0 ≤ softmax_weights scores hpos i :=
  div_nonneg (le_of_lt (hpos i)) (Finset.sum_nonneg fun j _ => le_of_lt (hpos j))

theorem scaling_normalizes_variance (d_k : ℝ) (hd : 0 < d_k) :
    d_k * (1 / Real.sqrt d_k)^2 = 1 := by
  rw [one_div, inv_pow, Real.sq_sqrt (le_of_lt hd)]
  field_simp

end RAC.Transformer.Attention
