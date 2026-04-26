/-
  RAC — LayerNorm/RMSNorm Post-Conditions
  Reference: rac.h rac_layernorm, rac_rmsnorm
-/
import Mathlib
noncomputable section
open Real Finset BigOperators
namespace RAC.Transformer.Norm

-- The `0 < d` hypothesis was originally a `variable (hd : 0 < d)` but Lean 4
-- auto-bound only includes it in declarations that mention `hd`, which split
-- `vecMean`'s elaborated arity from its call sites. Use only `{d : Nat}` as
-- a global; pass `hd` explicitly only where it's mathematically needed
-- (theorems, not the unused-by-body `def`s).
variable {d : Nat}

def vecMean (x : Fin d → Real) : Real := (∑ i, x i) / d
def vecMeanSq (x : Fin d → Real) : Real := vecMean (fun i => (x i) ^ 2)
def vecRMS (x : Fin d → Real) : Real := Real.sqrt (vecMeanSq x)

def layerNormOutput (x : Fin d → Real) (eps : Real) : Fin d → Real :=
  let μ := vecMean x
  let σ2 := vecMean (fun i => (x i - μ) ^ 2)
  fun i => (x i - μ) / Real.sqrt (σ2 + eps)

theorem layerNorm_mean_zero (hd : 0 < d) (x : Fin d → Real) (eps : Real) (heps : 0 < eps) :
    vecMean (layerNormOutput x eps) = 0 := by sorry

def rmsNormOutput (x : Fin d → Real) (eps : Real) : Fin d → Real :=
  let rms2 := vecMeanSq x
  fun i => x i / Real.sqrt (rms2 + eps)

end RAC.Transformer.Norm
