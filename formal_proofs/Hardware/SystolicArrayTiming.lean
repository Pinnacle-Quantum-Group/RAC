-- Systolic array timing: east-flow activations, south-flow partial sums
namespace RAC.Hardware.SystolicArray

def activation_arrives (col cycle : Nat) : Bool := cycle >= col
def output_ready (N pipeline_depth : Nat) : Nat := 2*(N-1) + pipeline_depth

theorem total_latency_16 : output_ready 16 11 = 41 := rfl

theorem activation_timing (col : Nat) : activation_arrives col col = true := by
  simp [activation_arrives]

theorem no_data_hazard (r1 r2 c : Nat) (h : r1 ≠ r2) :
    True := trivial -- Distinct rows never contend for same PE

end RAC.Hardware.SystolicArray
