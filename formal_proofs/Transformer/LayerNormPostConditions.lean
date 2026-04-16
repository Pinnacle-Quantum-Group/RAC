/-
  RAC — LayerNorm/RMSNorm Post-Conditions
  Reference: rac.h rac_layernorm, rac_rmsnorm
-/
import Mathlib
noncomputable section
open Real Finset BigOperators
namespace RAC.Transformer.Norm

variable {d : Nat} (hd : 0 < d)

def vecMean (x : Fin d → Real) : Real := (∑ i, x i) / d
def vecMeanSq (x : Fin d → Real) : Real := vecMean hd (fun i => (x i) ^ 2)
def vecRMS (x : Fin d → Real) : Real := Real.sqrt (vecMeanSq hd x)

def layerNormOutput (x : Fin d → Real) (eps : Real) : Fin d → Real :=
  let μ := vecMean hd x
  let σ2 := vecMean hd (fun i => (x i - μ) ^ 2)
  fun i => (x i - μ) / Real.sqrt (σ2 + eps)

theorem layerNorm_mean_zero (x : Fin d → Real) (eps : Real) (heps : 0 < eps) :
    vecMean hd (layerNormOutput hd x eps) = 0 := by sorry

def rmsNormOutput (x : Fin d → Real) (eps : Real) : Fin d → Real :=
  let rms2 := vecMeanSq hd x
  fun i => x i / Real.sqrt (rms2 + eps)

end RAC.Transformer.Norm
