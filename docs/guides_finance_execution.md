# Finance execution

Execution is modeled under an explicit `PhysicalLaw`. The package provides finite mathematical reference routes; it does not connect to venues or claim achievable fills.

## Event and accounting semantics

`ExecutionOrder`, `ExecutionFill`, `ExecutionAction`, and `ExecutionEvent` define the event surface. `ExecutionConstraints`, `ExecutionInventory`, `ExecutionAccounting`, and `ExecutionState` preserve limits, inventory, cash, fees, and status. `apply_execution_event` is the state transition; `ExecutionLedger` records it, `replay_execution_ledger` checks it, and `execution_cash_inventory_pnl` reports the explicit cash/inventory decomposition. Late, duplicate, invalid, or capacity-exceeding events remain status-bearing failures.

## Impact and order flow

`AlmgrenChrissModel` with `solve_almgren_chriss_schedule` provides the stated linear-permanent/quadratic-temporary reference. `TransientPropagatorModel`, `transient_impact_path`, and `diagnose_transient_manipulation` retain causal kernel state and finite-grid manipulation evidence. `ObizhaevaWangModel` and `obizhaeva_wang_path` expose the block-book displacement assumptions.

`HawkesOrderFlowModel`, `hawkes_intensity_path`, and `diagnose_hawkes_stability` retain excitation and stability evidence. `QueueReactiveModel`, `queue_reactive_jump_process`, and `diagnose_queue_intensities` are finite queue references, not exchange simulators.

## Control, market making, and evaluation

`JumpExecutionDefinition` → `JumpExecutionPlan` → `prepare_jump_execution` → `evaluate_jump_execution` is the controlled-jump route. `HJBExecutionDefinition`/`HJBExecutionPlan` with `solve_hjb_execution_reference` and `ImpulseExecutionDefinition` with `solve_impulse_execution_reference` keep residual and refinement evidence distinct.

Market-making references use `AvellanedaStoikovPlan`, `GLFTPlan`, `solve_avellaneda_stoikov_reference`, `solve_glft_reference`, and `compare_glft_to_finite_grid`. Robust execution uses `RobustExecutionDefinition`/`RobustExecutionPlan` and `solve_robust_execution_reference`. Mean-field execution remains candidate-only through `MeanFieldExecutionDefinition`, `MeanFieldExecutionPlan`, and `solve_mean_field_execution_candidate`.

Learned signature policies follow `CausalSignaturePolicySpec` → `prepare_causal_signature_policy` → `evaluate_causal_signature_policy`; `evaluate_signature_causality` checks information flow. `ExecutionPolicyEvaluationPlan` and `evaluate_execution_policy_holdout` keep holdout results separate from training. Qualification uses an exact `execution_support` tuple. Replay, synthetic profitability, or latency measurements do not establish live readiness, market access, best execution, or compliance.
