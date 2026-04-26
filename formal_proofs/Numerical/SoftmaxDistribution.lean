/-
  RAC — Softmax Distribution Invariants
  Reference: rac_cuda.cu rac_softmax
-/
import Mathlib
noncomputable section
open Finset BigOperators
namespace RAC.Numerical.Softmax

def softmax (f : Fin n → ℝ) (hf : ∀ i, 0 < f i) (i : Fin n) : ℝ :=
  f i / ∑ j, f j

theorem softmax_pos {f : Fin n → ℝ} (hf : ∀ i, 0 < f i) (hn : 0 < n) (i : Fin n) :
    0 < softmax f hf i := by
  unfold softmax
  exact div_pos (hf i) (Finset.sum_pos (fun j _ => hf j) ⟨i, mem_univ _⟩)

theorem softmax_sum_eq_one {f : Fin n → ℝ} (hf : ∀ i, 0 < f i) (hn : 0 < n) :
    ∑ i, softmax f hf i = 1 := by
  unfold softmax
  rw [Finset.sum_div]
  exact div_self (ne_of_gt (Finset.sum_pos (fun j _ => hf j) ⟨⟨0, hn⟩, mem_univ _⟩))

theorem softmax_le_one {f : Fin n → ℝ} (hf : ∀ i, 0 < f i) (hn : 0 < n) (i : Fin n) :
    softmax f hf i ≤ 1 := by
  unfold softmax
  apply div_le_one_of_le
  · exact Finset.single_le_sum (f := f) (fun j _ => le_of_lt (hf j)) (Finset.mem_univ i)
  · exact le_of_lt (Finset.sum_pos (fun j _ => hf j) ⟨i, Finset.mem_univ _⟩)

end RAC.Numerical.Softmax
