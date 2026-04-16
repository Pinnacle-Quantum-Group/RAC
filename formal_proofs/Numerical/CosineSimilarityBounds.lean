/-
  RAC — Cosine Similarity Bounds: rac_coherence ∈ [-1, 1]
  Reference: rac_cuda.cu rac_coherence
-/
import Mathlib
noncomputable section
open Real
namespace RAC.Numerical.CosineSimilarity

def coherence (θ_a θ_b : ℝ) : ℝ := cos (θ_a - θ_b)

theorem coherence_bounded (θ_a θ_b : ℝ) :
    -1 ≤ coherence θ_a θ_b ∧ coherence θ_a θ_b ≤ 1 :=
  ⟨neg_one_le_cos _, cos_le_one _⟩

theorem coherence_aligned (θ : ℝ) : coherence θ θ = 1 := by
  simp [coherence, sub_self, cos_zero]

theorem coherence_opposed (θ : ℝ) : coherence θ (θ + π) = -1 := by
  simp [coherence]; ring_nf; simp [cos_pi]
  sorry

theorem coherence_symmetric (θ_a θ_b : ℝ) :
    coherence θ_a θ_b = coherence θ_b θ_a := by
  simp [coherence, cos_neg, neg_sub]
  exact cos_neg (θ_a - θ_b) ▸ by ring_nf
  sorry

end RAC.Numerical.CosineSimilarity
