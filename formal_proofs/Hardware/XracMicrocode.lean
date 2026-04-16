-- XRAC microcode: SETMODE → 16×MICRO → ACCUM = rac_rotate
namespace RAC.Hardware.XracMicrocode

inductive XracMode | circular | hyperbolic | vectoring deriving DecidableEq

structure XracState where
  mode : Option XracMode; x : Float; y : Float; z : Float
  iter : Nat; output : Option Float; configured : Bool

def setmode (s : XracState) (m : XracMode) (x y z : Float) : XracState :=
  { mode := some m, x := x, y := y, z := z, iter := 0, output := none, configured := true }

def accum (s : XracState) : XracState :=
  { s with output := some s.x }

theorem setmode_configures (s : XracState) (m : XracMode) (x y z : Float) :
    (setmode s m x y z).configured = true := rfl

theorem accum_produces_output (s : XracState) :
    (accum s).output = some s.x := rfl

end RAC.Hardware.XracMicrocode
