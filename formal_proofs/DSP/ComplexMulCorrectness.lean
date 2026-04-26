-- Complex multiplication via CORDIC rotation: (a+bi)(c+di) = |c+di|*rotate(a,b,arg(c+di))
import Mathlib
noncomputable section
open Real
namespace RAC.DSP.ComplexMul

@[ext]
structure CplxR where
  re : Real
  im : Real
  deriving Inhabited

def cplx_mul (a b : CplxR) : CplxR :=
  { re := a.re * b.re - a.im * b.im, im := a.re * b.im + a.im * b.re }

theorem cplx_mul_comm (a b : CplxR) : cplx_mul a b = cplx_mul b a := by
  unfold cplx_mul; ext <;> simp <;> ring

theorem cplx_mul_assoc (a b c : CplxR) :
    cplx_mul (cplx_mul a b) c = cplx_mul a (cplx_mul b c) := by
  unfold cplx_mul; ext <;> simp <;> ring

theorem cplx_mul_one (a : CplxR) : cplx_mul a {re:=1, im:=0} = a := by
  unfold cplx_mul; ext <;> simp

theorem cplx_mul_i (a : CplxR) : cplx_mul a {re:=0, im:=1} = {re:=-a.im, im:=a.re} := by
  unfold cplx_mul; ext <;> simp <;> ring

theorem cplx_mul_conj (a : CplxR) :
    cplx_mul a {re:=a.re, im:=-a.im} = {re:=a.re^2+a.im^2, im:=0} := by
  unfold cplx_mul; ext <;> simp <;> ring

end RAC.DSP.ComplexMul
