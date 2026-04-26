/-
  RAC — RTL FSM Correctness for CORDIC Core
  IDLE → ITER(×16) → DONE → IDLE
  Reference: rtl/rac_cordic_core.v
-/
namespace RAC.Transformer.RtlFsm

inductive FsmState where | idle | iter (cycle : Nat) | done
  deriving DecidableEq, Repr

def isBusy : FsmState → Bool | .iter _ => true | _ => false
def isDone : FsmState → Bool | .done => true | _ => false

def next (s : FsmState) (start : Bool) : FsmState :=
  match s with
  | .idle => if start then .iter 0 else .idle
  | .iter c => if c < 15 then .iter (c + 1) else .done
  | .done => .idle

def run (s : FsmState) (start : Bool) : Nat → FsmState
  | 0 => s
  | n + 1 => run (next s start) false n

theorem reaches_done_in_17 : run .idle true 17 = .done := by native_decide
theorem returns_to_idle_in_18 : run .idle true 18 = .idle := by native_decide

theorem done_one_cycle :
    isDone (run .idle true 16) = false ∧
    isDone (run .idle true 17) = true ∧
    isDone (run .idle true 18) = false := by
  exact ⟨by native_decide, by native_decide, by native_decide⟩

theorem busy_during_iter :
    isBusy (run .idle true 0) = false ∧
    isBusy (run .idle true 1) = true ∧
    isBusy (run .idle true 16) = true ∧
    isBusy (run .idle true 17) = false := by
  exact ⟨by native_decide, by native_decide, by native_decide, by native_decide⟩

theorem busy_done_exclusive (s : FsmState) :
    ¬(isBusy s = true ∧ isDone s = true) := by
  cases s <;> simp [isBusy, isDone]

theorem idle_stable : ∀ n, run .idle false n = .idle := by
  intro n; induction n with
  | zero => rfl
  | succ n ih => simp [run, next]; exact ih

theorem iter_progression :
    next (.iter 0) false = .iter 1 ∧
    next (.iter 7) false = .iter 8 ∧
    next (.iter 14) false = .iter 15 ∧
    next (.iter 15) false = .done := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

end RAC.Transformer.RtlFsm
