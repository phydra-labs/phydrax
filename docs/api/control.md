# Control

`phydrax.control` provides finite-horizon control problem contracts, linear-system
analysis, and local, exact finite-catalog, or bounded stochastic solvers. The common
contract keeps physical time, case axes, validity, status, and numerical provenance
explicit. It does not clip inputs, repair infeasible iterates, change methods after
failure, or hide a fallback.

## Choosing a path

| Need | Public entry point | Scope and evidence |
|---|---|---|
| Roll out a supplied control | `ControlProblem.rollout` | Discrete or differential dynamics; returns a `ControlTrajectory` without evaluating cost or constraints. |
| Audit a supplied control | `ControlProblem.evaluate` | Adds left-rectangle sampled cost and sampled constraint residuals. Nonlinear feasibility is checked only at declared sample sites and is not certified between them. |
| Unconstrained affine-quadratic control | `finite_horizon_lqr`, `continuous_lqr`, `discrete_lqr` | Riccati solutions with residual, conditioning, stability, and convergence diagnostics. |
| Unconstrained affine-quadratic feedback game | `control.games.finite_horizon_lq_feedback_nash` | Simultaneous full-state feedback Nash policy with explicit player ownership, per-player values, curvature, rank, conditioning, stationarity, Bellman, and causal-failure evidence. |
| Deterministic nonlinear game | `control.games.evaluate_game_policy`, `control.games.nominal_nash_residual`, `control.games.solve_ilq_feedback_game` | Physical evaluation, local first-order evidence, and residual-globalized iLQ remain separate; success is local nominal stationarity, not exact nonlinear feedback Nash. |
| Constrained open-loop or local feedback game | `control.games.solve_open_loop_ve`, `control.games.solve_open_loop_gne`, `control.games.solve_open_loop_game_kkt`, `control.games.solve_feedback_quasi_nash_model` | Choose the equilibrium concept and multiplier ownership first; evidence ranges from convex open-loop VE/GNE to local KKT or a fixed-active-set feedback branch. |
| Stochastic control or game | `control.stochastic.evaluate_feedback_policy`, exact LQG entry points, SMP/HJB references, or `control.games.solve_stochastic_policy_game` | Information, noise timing, training/holdout samples, and evidence claims are explicit; there is no universal stochastic solver. |
| Mean-field or finite-state game | Frozen/fixed/common-noise/constrained MFG, finite-$N$, MFC, common-information, and master-equation entry points below | Every layer reports its own law, information, statistical, and provenance ceiling; no layer silently implies another. |
| Constrained affine discrete control | `compile_linear_quadratic_control`, `solve_linear_quadratic_control` | Canonical, uncondensed QP with exact decision and constraint layouts. |
| Receding-horizon affine control | `solve_receding_horizon_mpc` | Re-solves canonical QPs and records every subproblem result and exact state handoff. |
| One unconstrained nonlinear case | `solve_ilqr` | iLQR with a fixed requested regularization and explicit line-search or curvature failure. |
| One constrained nonlinear case | `solve_multiple_shooting` | Dense SQP with independent state nodes, exact defect accounting, and sampled path constraints. |
| Implicit DAE or all-at-once trajectory optimization | `solve_direct_collocation` | Backward-Euler or midpoint transcription with interval controls, exact sparse derivatives, physical defect audits, and an explicitly selected dense-native or sparse-Ipopt method. |
| An explicit finite control catalog | `search_control_candidates` | Exact minimum over the declared coefficient arrays; retains invalidity, index, signature, and winner-reconstruction evidence. |
| A bounded stochastic initializer | `search_control` | Differential evolution over a continuous coefficient box; returns the best candidate found, never a global-optimality claim. |

## Shared axes, callbacks, and provenance

A shared `phydrax.dynamics.TimeGrid` contains a strictly increasing physical axis with
`num_times = num_steps + 1`. For `case_shape`, `state_shape`, and `control_shape`, the
foundation uses:

- states: `case_shape + (num_times,) + state_shape`;
- controls: `case_shape + (num_steps,) + control_shape`;
- per-node validity: `case_shape + (num_times,)`;
- per-case status: `case_shape`.

Discrete transitions and differential vector fields have the context-last signature
`callback(time, state, control, args)`. Running costs and path constraints use the same
signature and return one scalar per case and sample. Terminal callbacks use
`callback(time, state, args)`. Physical axes are never silently flattened; only local
linearization matrices flatten state, control, and output payloads.

`ControlTrajectory` retains `problem_id`, `dynamics_id`, `control_id`, `backend_id`,
`method_id`, `discretization_id`, `approximation_id`, and the raw `backend_status`.
There is no implicit sample mask: invalid nodes remain represented in `valid`, and masks
from another domain are not inferred. `ControlResult.valid` means that rollout and sampled
cost evaluation are valid; use `ControlResult.successful` or `status` when sampled
feasibility must also hold.

The stable foundation status codes are:

| Code | Constant | Meaning |
|---:|---|---|
| 0 | `CONTROL_SUCCESS` | Dynamics, sampled cost, and sampled constraints succeeded. |
| 1 | `CONTROL_DYNAMICS_FAILED` | The rollout or backend failed, or produced an invalid node. |
| 2 | `CONTROL_COST_INVALID` | A sampled objective term was nonfinite or otherwise invalid. |
| 3 | `CONTROL_INFEASIBLE` | A declared sampled constraint exceeded its tolerance. |

### Time, problem, trajectory, and result

::: phydrax.dynamics.TimeGrid

::: phydrax.control.ControlProblem

::: phydrax.control.ControlTrajectory

::: phydrax.control.ControlResult

## Dynamics

`DiscreteControlDynamics` scans the declared intervals and calls the transition once per
case. `DifferentialControlDynamics` lowers a controlled vector field to the canonical
Diffrax-backed differential problem, saves only on the shared control grid, and exposes
the backend result. Differential solver choices and tolerances are forwarded explicitly;
a failed integration is reported rather than retried with another solver.

`DiscreteTransition` and `DifferentialControlVectorField` are the public callback type
aliases for the context-last signatures described above.

::: phydrax.control.DiscreteControlDynamics

::: phydrax.control.DifferentialControlDynamics

## Costs and constraints

`evaluate_sampled_cost` uses the left endpoint on every interval and multiplies the
running rate by the physical interval duration before adding the terminal term. Cost
validity is separate from constraint feasibility.

A path or terminal residual is feasible when it is less than or equal to the requested
tolerance. `evaluate_sampled_feasibility` records every residual, the maximum positive
violation, and `feasible`, but `SampledControlFeasibility.certified` is always `False`.
Grid-sampled nonlinear constraints do not certify the continuous interval between nodes.
Only the B-spline coefficient convex-hull check documented below certifies its specific
continuous control-bound statement.

::: phydrax.control.RunningCost

::: phydrax.control.TerminalCost

::: phydrax.control.SampledControlLoss

::: phydrax.control.evaluate_sampled_cost

::: phydrax.control.PathConstraint

::: phydrax.control.TerminalConstraint

::: phydrax.control.SampledControlFeasibility

::: phydrax.control.evaluate_sampled_feasibility

## Control parameterizations, certificates, and refinement

| Parameterization | Coefficient axis | Use |
|---|---|---|
| `PiecewiseConstantControlParameterization` | `num_steps` | Exact left-endpoint-held interval controls; canonical for discrete rollout, QP decoding, MPC, and local seeds. |
| `PiecewiseLinearControlParameterization` | `num_times` | Continuous nodal interpolation on a physical time grid. |
| `BSplineControlParameterization` | `grid.coefficient_count` | Smooth, differentiable fixed-grid control with fewer or structured coefficients. |

Every parameterization publishes `parameter_shape`, `control_shape`,
`parameterization_id`, and `approximation_id`. Coefficients must have shape
`case_shape + parameter_shape`; they are not broadcast, clipped, or repaired.

`BSplineControlParameterization.bound_certificate` uses the nonnegative partition of
unity to prove that the continuous B-spline control stays inside componentwise bounds
when all relevant coefficients do. It does not certify states, dynamics, arbitrary
nonlinear path constraints, or optimality. `refine` returns both transferred coefficients
and the canonical `BSplineGridTransfer`; inspect its resolved `method`,
`condition_estimate`, and `projection_error_bound`. Nested equal-degree refinement can be
exact, while an L2 transfer is an explicitly diagnosed approximation.

::: phydrax.control.AbstractControlParameterization

::: phydrax.control.PiecewiseConstantControlParameterization

::: phydrax.control.PiecewiseLinearControlParameterization

::: phydrax.control.BSplineControlParameterization

::: phydrax.control.BSplineControlBoundCertificate

::: phydrax.control.BSplineControlRefinement

## Local linearization

The linearization functions evaluate a discrete transition or differential vector field
at explicit operating points using JAX forward JVPs. `AffineControlLinearization` keeps
operating values in their physical shapes and exposes flattened `A`, `B`, `C`, and `D`
matrices, affine offsets, per-case `valid`, and `LinearizationProvenance`. An optional
output callback also has the context-last signature
`output(time, state, control, args)`.

::: phydrax.control.LinearizationProvenance

::: phydrax.control.AffineControlLinearization

::: phydrax.control.linearize_discrete_dynamics

::: phydrax.control.linearize_differential_dynamics

::: phydrax.control.linearize_control_dynamics

## Lyapunov equations and Gramians

Dense Lyapunov solvers handle the continuous equation `A X + X Aᴴ + Q = 0` and the
discrete equation `X - A X Aᴴ = Q`. The bare `*_lyapunov_solution` functions return only
the differentiable matrix; prefer the diagnosed `solve_*` functions when a decision
depends on stability, separation, residual, or convergence. Finite-horizon variants do
not require a stable system. Krylov variants accept an operator action and report GMRES
evidence.

No method falls back to another method and no regularization is inserted. Dense
Lyapunov and dense Gramian entry points enforce `max_dimension` (default 128). Choose a
Krylov Lyapunov solve or a finite Gramian action when the dense state matrix is
inappropriate.

`LinearMatrixEquationStatus` values are `CONVERGED`,
`RESIDUAL_TOLERANCE_NOT_MET`, `SINGULAR_EQUATION`, `NONFINITE`, `UNSTABLE_SYSTEM`, and
`MARGINAL_SYSTEM`. `linear_matrix_equation_status_message` supplies their stable text.
For dense Gramians, inspect both the matrix-equation diagnostics and the PSD, rank,
singularity, and Gramian-condition diagnostics. Matrix-free actions additionally report
quadrature or finite-sum work and approximation diagnostics; they are finite-horizon
operations, not dense matrix reconstructions.

### Lyapunov API

::: phydrax.control.LinearMatrixEquationStatus

::: phydrax.control.LinearMatrixEquationDiagnostics

::: phydrax.control.LyapunovResult

::: phydrax.control.linear_matrix_equation_status_message

::: phydrax.control.continuous_lyapunov_solution

::: phydrax.control.discrete_lyapunov_solution

::: phydrax.control.solve_continuous_lyapunov

::: phydrax.control.solve_discrete_lyapunov

::: phydrax.control.finite_continuous_lyapunov

::: phydrax.control.finite_discrete_lyapunov

::: phydrax.control.solve_continuous_lyapunov_krylov

::: phydrax.control.solve_discrete_lyapunov_krylov

### Gramian API

For continuous systems, `horizon=None` requests the infinite integral; for discrete
systems, `steps=None` requests the infinite sum. Those infinite-horizon results are
successful only for the appropriate stable system. Finite horizons remain valid for
stable, marginal, or unstable dynamics and retain the horizon in diagnostics.

::: phydrax.control.GramianDiagnostics

::: phydrax.control.GramianResult

::: phydrax.control.GramianActionDiagnostics

::: phydrax.control.GramianActionResult

::: phydrax.control.continuous_controllability_gramian

::: phydrax.control.continuous_observability_gramian

::: phydrax.control.discrete_controllability_gramian

::: phydrax.control.discrete_observability_gramian

::: phydrax.control.continuous_controllability_gramian_action

::: phydrax.control.continuous_observability_gramian_action

::: phydrax.control.discrete_controllability_gramian_action

::: phydrax.control.discrete_observability_gramian_action

## Riccati equations and LQR

The cost convention is `xᵀ Q x / 2 + uᵀ R u / 2 + xᵀ S u`; finite-horizon LQR also
supports affine dynamics and linear or constant cost terms. Finite-horizon arrays carry
an explicit stage axis and are not time-broadcast. `AffineFeedbackPolicy` implements
`u = K x + k` online and preserves its physical time grid and policy ID.

Algebraic Riccati diagnostics report residuals, equation conditioning, control
conditioning, stabilizability, detectability, stability, iteration count, and method.
`RiccatiStatus` values are `SUCCESS`, `UNSTABILIZABLE`, `UNDETECTABLE`,
`UNSTABLE_SOLUTION`, `NONFINITE`, and `NONCONVERGED`. Indefinite or singular cost data are
rejected; no hidden regularization is added. CARE and DARE derivatives use the diagnosed
implicit matrix equation at the returned stabilizing solution.
Infinite-horizon results keep a transform-stable policy PyTree even on failure. Its
coefficients are raw numerical evidence and may be nonfinite for invalid cases; apply
the policy only where `result.valid` is true.

::: phydrax.control.RiccatiStatus

::: phydrax.control.AlgebraicRiccatiDiagnostics

::: phydrax.control.AlgebraicRiccatiResult

::: phydrax.control.solve_continuous_are

::: phydrax.control.solve_discrete_are

::: phydrax.control.AffineFeedbackPolicy

::: phydrax.control.QuadraticValueFunction

::: phydrax.control.FiniteHorizonLQRDiagnostics

::: phydrax.control.FiniteHorizonLQRResult

::: phydrax.control.InfiniteHorizonLQRResult

::: phydrax.control.finite_horizon_lqr

::: phydrax.control.continuous_lqr

::: phydrax.control.discrete_lqr

## Linear-quadratic feedback games

`phydrax.control.games` solves finite-horizon discrete affine linear-quadratic
games with simultaneous full-state feedback. Every player minimizes an
individual quadratic-affine cost that may depend on the full joint control.
The returned joint `AffineFeedbackPolicy` uses `u = K x + k`; its contiguous
control rows are owned by the ordered `PlayerControlPartition`.

For `case_shape = C`, `P` players, `T` stages, state size `n`, and joint control
size `m`, dynamics arrays have shapes `C + (T, n, n)` and `C + (T, n, m)`.
Player stage costs carry shapes `C + (P, T, ...)`; terminal costs carry
`C + (P, ...)`. Stage, player, and case axes are explicit and never
broadcast. The problem has `T` controls and `T + 1` player-value nodes.
Stage costs are discrete sums; they are not duration-weighted
`ControlProblem` sampled costs.

At each stage, player-owned first-order rows form one generally nonsymmetric
coupled Nash system. The authoritative policy comes only from a Phydrax
`DenseLU` solve with a declared multiple-RHS layout. A separate
diagnostic-only SVD supplies numerical rank and the full condition number; it
is never used to solve or repair the system. No inverse, pseudoinverse,
symmetrization of the coupled system, diagonal jitter, clipping,
regularization, zero policy, or method fallback is permitted.

A successful result certifies positive continuation-augmented own-action
curvature, full coupled rank, finite output, and bounded independent
stationarity and Bellman residuals. Failure to satisfy those conditions means
this method did not certify a unique feedback Nash policy; it is not a proof
that no equilibrium exists. `first_failed_stage` is `-1` on success, `T` for
a terminal-cost failure, or the first direct causal failure encountered by
the reverse recursion. Earlier stages then report `DEPENDENCY_FAILED`.
Invalid policy and value arrays are retained only as numerical evidence and
must not be applied.

The scope is deterministic, unconstrained, discrete-time, finite-horizon,
simultaneous, full-state feedback with all players minimizing. It does not
represent open-loop Nash, zero-sum maximizers, constrained or generalized
Nash equilibria, nonlinear iLQ games, stochastic games, partial observations,
or mean-field games.

::: phydrax.control.games.PlayerControlPartition

::: phydrax.control.games.LQFeedbackNashStatus

::: phydrax.control.games.FiniteHorizonLQFeedbackNashDiagnostics

::: phydrax.control.games.FiniteHorizonLQFeedbackNashResult

::: phydrax.control.games.finite_horizon_lq_feedback_nash

## Frequency response

The transfer functions solve the dense resolvent directly and never substitute a
pseudoinverse. `frequency_response` maps angular frequency to `s = iω` for continuous
systems and `z = exp(iω sample_time)` for discrete systems. The response object retains
evaluation points, input-to-state and input-to-output values, resolvents, poles,
condition numbers, stability, singular flags, system type, sample time, and method ID.

Frequency status codes are `FREQUENCY_SUCCESS`, `FREQUENCY_SINGULAR`,
`FREQUENCY_UNSTABLE`, and `FREQUENCY_NONFINITE`; use `frequency_status_name` for stable
text. An unstable system is explicitly invalid even if a finite resolvent can be
computed. These are dense direct operations, not matrix-free frequency solvers.

::: phydrax.control.FrequencyResponseResult

::: phydrax.control.frequency_status_name

::: phydrax.control.continuous_transfer_function

::: phydrax.control.discrete_transfer_function

::: phydrax.control.frequency_response

::: phydrax.control.input_to_state_response

::: phydrax.control.input_to_output_response

## Canonical QP, SOCP, and MPC compilation

`LinearQuadraticControlProblem` uses
`x[t+1] = A[t] x[t] + B[t] u[t] + c[t]` with an explicit stage axis.
Decisions remain uncondensed: all state nodes followed by all interval controls.

State and control boxes compile to native `Bounds`, not synthetic polyhedral rows.
`LinearControlConstraintLayout` therefore covers initial, dynamics, stage, and
terminal rows; `LinearControlBoundLayout` separately retains state/control bound
coordinates and bound-dual provenance.

`LinearControlCompilationPolicy("dense")` retains dense arrays.
`LinearControlCompilationPolicy("sparse")` additionally emits shared-pattern
`SparseLinearMap` representations for the stage-block Hessian and true constraint
operators. Sparse coefficients may carry case batches while preserving one route
pattern. Representation choice is explicit; there is no size-based switch.

### Prepared QPs

`prepare_linear_quadratic_control` composes compilation with the canonical convex
program lifecycle. `refresh_linear_quadratic_control` changes numeric dynamics,
costs, right-hand sides, and bounds while preserving horizon, shapes, row topology,
bound roles, problem identity, policy, and sparse pattern. The one-shot
`solve_linear_quadratic_control` remains a convenience composition.

Inspect `qp_result.status`, `valid`, `certificate`, `provenance`, bound multipliers,
and KKT diagnostics rather than assuming a returned primal is usable.

### Receding-horizon warm starts

MPC caches one prepared template per `(prediction horizon, terminal topology)` and
refreshes numeric data between compatible windows. Exact affine state handoff remains
independent of predicted local nodes.

`MPCWarmStartPolicy` explicitly chooses terminal-control filling (`"hold"` or
`"zero"`) and the strict interior margin. Primal states/controls, dynamics/stage
duals, inequality multipliers, and bound multipliers are shifted by their declared
layouts. New rows are initialized explicitly. A method that does not declare
warm-start support is rejected before rollout.

Terminal policy remains explicit: `"global"` applies terminal terms only when the
window reaches the global final node, `"always"` applies them at every endpoint, and
`"none"` omits them.

### Affine SOCP constraints

`StageSecondOrderConstraint` represents
`||F_x x + F_u u + f||₂ ≤ g_x x + g_u u + g₀` at every stage.
`TerminalSecondOrderConstraint` provides the terminal analogue.
`compile_linear_conic_control` appends exact SOC blocks to the same uncondensed
decision layout; `solve_linear_conic_control` requires an explicit conic-capable
policy such as `ClarabelInteriorPoint`.

These are exact affine SOC contracts. They are not sampled nonlinear or generic
chance-constraint certificates.

### QP API

::: phydrax.control.LinearQuadraticControlProblem

::: phydrax.control.LinearControlCompilationPolicy

::: phydrax.control.LinearControlDecisionLayout

::: phydrax.control.LinearControlConstraintLayout

::: phydrax.control.LinearControlBoundLayout

::: phydrax.control.LinearControlQPCompilation

::: phydrax.control.PreparedLinearControlQP

::: phydrax.control.LinearControlQPSolution

::: phydrax.control.compile_linear_quadratic_control

::: phydrax.control.prepare_linear_quadratic_control

::: phydrax.control.refresh_linear_quadratic_control

::: phydrax.control.solve_prepared_linear_quadratic_control

::: phydrax.control.decode_linear_control_solution

::: phydrax.control.solve_linear_quadratic_control

### SOCP API

::: phydrax.control.StageSecondOrderConstraint

::: phydrax.control.TerminalSecondOrderConstraint

::: phydrax.control.LinearControlConicCompilation

::: phydrax.control.LinearControlConicSolution

::: phydrax.control.compile_linear_conic_control

::: phydrax.control.solve_linear_conic_control

### MPC API

::: phydrax.control.MPCWarmStartPolicy

::: phydrax.control.RecedingHorizonMPC

::: phydrax.control.RecedingHorizonMPCResult

::: phydrax.control.solve_receding_horizon_mpc

## Iterative LQR

`solve_ilqr` accepts exactly one unbatched, unconstrained `ControlProblem`. Discrete
dynamics use their declared transition. Differential dynamics require an explicit
`DifferentialControlFlow`; iLQR never selects or retries an integration method. The
initial controls have shape `(num_steps,) + control_shape`.

The `regularization` value is the exact fixed diagonal shift in every backward pass; it
is not increased adaptively. A non-positive-definite shifted block, failed initial
rollout, rejected line search, or iteration limit produces the corresponding
`ILQRStatus` (`BACKWARD_PASS_NOT_POSITIVE_DEFINITE`, `INITIAL_ROLLOUT_FAILED`,
`LINE_SEARCH_FAILED`, or `MAX_ITERATIONS`) rather than a repaired solution.
`ILQRResult` retains the foundation-compatible trajectory, sampled loss and feasibility,
a nominal affine-feedback policy, and complete convergence histories. A converged iLQR
trajectory is a local result, not a global optimum.

::: phydrax.control.ILQRStatus

::: phydrax.control.DifferentialControlFlow

::: phydrax.control.ILQRPolicy

::: phydrax.control.ILQRDiagnostics

::: phydrax.control.ILQRResult

::: phydrax.control.solve_ilqr

## Multiple shooting

`solve_multiple_shooting` is a single-case dense SQP method. State nodes and interval
controls are independent decisions; the initial boundary, segment continuity, and every
declared path or terminal residual are represented in each local QP. The result retains
exact defects, residuals, constraint provenance, KKT and rollout audits, the last QP
result, and one history row per attempted QP.

The statuses are `MULTIPLE_SHOOTING_SUCCESS`, `MULTIPLE_SHOOTING_MAX_ITERATIONS`,
`MULTIPLE_SHOOTING_QP_FAILED`, `MULTIPLE_SHOOTING_LINE_SEARCH_FAILED`,
`MULTIPLE_SHOOTING_INTEGRATION_FAILED`, `MULTIPLE_SHOOTING_ROLLOUT_FAILED`, and
`MULTIPLE_SHOOTING_NONFINITE`. The QP dense guard is explicit. `hessian_regularization`
and `qp_regularization` are separate requested values; there is no projection, elastic
variable, feasibility repair, fallback, or implicit regularization. Nonlinear path
constraints remain sample-node checks and do not certify feasibility between nodes.

`compile_structured_multiple_shooting` lowers the same objective, boundary,
continuity, path, and terminal equations to the canonical
`StructuredNonlinearProgram`. `solve_structured_multiple_shooting` executes an
explicit structured method and independently re-evaluates every shooting
residual. This route does not alter the dense SQP contract or silently select it.

::: phydrax.control.MultipleShootingDecisionLayout

::: phydrax.control.MultipleShootingLinearization

::: phydrax.control.MultipleShootingHistory

::: phydrax.control.MultipleShootingResult

::: phydrax.control.linearize_multiple_shooting

::: phydrax.control.solve_multiple_shooting

::: phydrax.control.StructuredMultipleShootingCompilation

::: phydrax.control.StructuredMultipleShootingResult

::: phydrax.control.compile_structured_multiple_shooting

::: phydrax.control.solve_structured_multiple_shooting

## Direct collocation

`TrajectoryOptimizationProblem` is the continuous boundary-value contract used by
direct collocation. It accepts an input-aware `ContinuousSystem` or
`DifferentialAlgebraicSystem`, an optional fixed initial state, running, terminal, and
whole-trajectory costs, bound-form path and trajectory constraints, shared optimized
parameter coordinates, fixed arguments, and explicit case axes. A continuous
`ControlProblem` can be passed directly for the fixed-duration, fixed-initial-state
case; discrete control problems and variable-duration conversion are rejected.

`DirectCollocationPlan` owns a `TemporalMesh(role="collocation")`, one verified
`ThetaMethod`, scaling, sparse-derivative compilation, and physical audit policy. The
supported methods are endpoint backward Euler and implicit midpoint. States are nodal
decisions and controls are interval decisions, so no unused endpoint-control coordinate
is introduced. A variable-duration plan uses one log-duration coordinate and maps the
static reference mesh affinely into physical time.

The compiler produces:

- a dense-compatible `MinimizationProblem` for an explicitly selected ordinary
  method; and
- a `StructuredNonlinearProgram`, reusable `StructuredNonlinearTemplate`, and
  default `PreparedStructuredNonlinearProgram` with exact sparse Jacobian and
  optional exact sparse Lagrangian-Hessian plans.

Any `AbstractStructuredNonlinearMethod` can consume the structured route.
`PrimalDualInteriorPoint(mode="sparse-augmented")` uses the native augmented
KKT path; `IpoptMinimize` uses low-level sparse Ipopt callbacks. No backend is
selected from problem size and no backend failure triggers fallback.

`refresh_direct_collocation` rebinds fixed-shape numeric arguments while
retaining transcription and derivative topology. `solve_pooled_direct_collocation`
solves independent initial decisions through an explicitly sized structured
task pool. It never partitions an internal `case_shape`, because those cases may
share optimized parameters.

`DirectCollocationResult` retains the physical decision, a `ControlTrajectory`,
stage times, states, rates, and controls, raw dynamics defects, every declared
constraint block, the generic optimization result, optional structured result
and portable warm start, sparse topology counts, and physical feasibility
diagnostics. `DirectCollocationOffGridAudit` retains sampled times, raw
residuals, and per-case/per-interval defect and path-violation arrays rather
than only global maxima. Collocation constraints are enforced at their declared
stages. The audit evaluates additional piecewise-linear states and held
controls, but `certified=False`: neither site set is a continuous-time path
certificate.

The direct statuses are `DIRECT_COLLOCATION_SUCCESS`,
`DIRECT_COLLOCATION_OPTIMIZER_FAILED`, `DIRECT_COLLOCATION_NONFINITE`,
`DIRECT_COLLOCATION_DEFECT_FAILED`, `DIRECT_COLLOCATION_CONSTRAINT_FAILED`, and
`DIRECT_COLLOCATION_RECONSTRUCTION_FAILED`. Backend success is never sufficient:
scaled KKT evidence, raw physical defects, and raw physical constraint bounds are
checked independently.

`refine_direct_collocation` bisects an explicit interval selection and transfers only
the physical primal: states use the declared piecewise-linear interpolation, controls
preserve the held representation at target stage times, and shared parameters and
duration transfer identically. Mesh-shaped bounds require an explicit
`DirectCollocationBoundProvider`. Topology-changing dual multipliers are never reused.
`solve_refined_direct_collocation` records every selection, transfer, solve, objective
change, common-grid state/control change, and sampled-defect reduction. Its convergence
status remains sampled evidence, not a continuous certificate.

`replay_direct_collocation` independently replays one unbatched controlled-DAE result
through the native DAE consistency and implicit-stage lifecycle. It constructs a
`HeldInputPolicy`, preserves optimized parameters and duration, and reports node,
terminal, and algebraic discrepancies in `DirectCollocationReplayEvidence`. Replay
never rewrites the collocation result status.

::: phydrax.control.TrajectoryOptimizationContext

::: phydrax.control.TrajectoryOptimizationView

::: phydrax.control.BoundedPathConstraint

::: phydrax.control.BoundedTrajectoryConstraint

::: phydrax.control.TrajectoryOptimizationProblem

::: phydrax.control.DirectCollocationScaling

::: phydrax.control.DirectCollocationDerivativePolicy

::: phydrax.control.DirectCollocationAuditPolicy

::: phydrax.control.DirectCollocationPlan

::: phydrax.control.DirectCollocationBounds

::: phydrax.control.DirectCollocationDecision

::: phydrax.control.DirectCollocationDecisionLayout

::: phydrax.control.DirectCollocationConstraintLayout

::: phydrax.control.DirectCollocationCompilation

::: phydrax.control.PreparedDirectCollocation

::: phydrax.control.DirectCollocationDiagnostics

::: phydrax.control.DirectCollocationOffGridAudit

::: phydrax.control.DirectCollocationRefinementPolicy

::: phydrax.control.DirectCollocationRefinementSelection

::: phydrax.control.DirectCollocationPrimalTransfer

::: phydrax.control.DirectCollocationRefinementLevel

::: phydrax.control.DirectCollocationRefinementStudy

::: phydrax.control.DirectCollocationReplayPolicy

::: phydrax.control.DirectCollocationReplayEvidence


::: phydrax.control.DirectCollocationResult

::: phydrax.control.compile_direct_collocation

::: phydrax.control.prepare_direct_collocation

::: phydrax.control.solve_prepared_direct_collocation

::: phydrax.control.solve_direct_collocation

::: phydrax.control.select_direct_collocation_intervals

::: phydrax.control.refine_direct_collocation

::: phydrax.control.solve_refined_direct_collocation

::: phydrax.control.replay_direct_collocation


## Exact finite control catalogs

`search_control_candidates` evaluates an explicit catalog of complete coefficient
arrays. The candidate point shape must be exactly
`case_shape + parameterization.parameter_shape`. Use one correlated `FiniteAxis` so
that each catalog row remains one complete coefficient array; the adapter deliberately
rejects a Cartesian product of individual coefficient entries.

```py
catalog = phx.optim.FiniteProductSpace(
    phx.optim.FiniteAxis(
        jnp.asarray(
            [
                [[0.0], [0.0]],
                [[0.5], [0.5]],
                [[1.0], [0.0]],
            ]
        )
    )
)
selected = phx.control.search_control_candidates(
    problem,
    parameterization,
    catalog,
    search=phx.optim.FiniteExhaustiveSearch(batch_size=2),
)
```

Each search evaluation performs the ordinary rollout, sampled cost, and sampled
feasibility contracts. The scalar catalog score is the sum of `sampled_loss.total`
across all cases. A candidate is selectable only when every case has a successful
rollout, every sampled constraint is feasible, and the total score is finite.

The search kernel retains only objective, validity, and index evidence. When a valid
winner exists, the adapter reconstructs that coefficient array and performs one
additional full evaluation to return its native `ControlResult` and
`ControlTrajectory`. Consequently, `objective_evaluations == candidate_count`,
`winner_evaluations` is one or zero, and `total_control_evaluations` is their sum.
When all candidates are invalid, no coefficients, evaluation, trajectory, or control
ID are fabricated; accessing `trajectory` or `controls` raises.

The guarantee is exact only over the declared catalog. Sampled feasibility remains a
sample-site statement, and a winning piecewise or spline parameterization does not
certify behavior between those sites.

::: phydrax.control.ControlCandidateSearchResult

---

::: phydrax.control.search_control_candidates

## Bounded stochastic initialization

`search_control` combines a `ControlProblem`, any public control parameterization, and a
`phydrax.optim.DifferentialEvolutionSearch`. `CoefficientBounds` is the public pair
`(lower_bounds, upper_bounds)`, with each array matching
`case_shape + parameter_shape`. Initial coefficients and bounds are validated as supplied
and are never clipped or repaired. Invalid rollout or sampled-infeasible candidates are
counted as invalid evaluations rather than hidden behind a penalty.

`ControlSearchResult` retains the root key, search configuration, design signature,
population, objective history, invalid count, termination reason, all problem/dynamics/
time/control/parameterization/approximation IDs, and a native best-found trajectory that
can seed a local method. `population_converged` describes finite population dispersion
only. A finite bounded differential-evolution run does not certify basin coverage or
global optimality.

::: phydrax.control.ControlSearchResult

::: phydrax.control.search_control

## Batched local solves, advanced transcription, and certificates

Homogeneous iLQR case axes use `plan_ilqr`, `prepare_ilqr`, and
`solve_prepared_ilqr`. The prepared kernel flattens case axes only internally,
uses fixed iteration and line-search capacities, and restores every physical case
axis in trajectories, policies, histories, and statuses. A failed case does not
deactivate its siblings. Unsupported host-only backends fail preparation rather
than falling back.

`RadauIIAMethod(s)` constructs the right-Radau tableau of order `2s-1`.
`radau_collocation_defects` evaluates explicit or DAE stage and endpoint defects.
Fixed `DirectCollocationPhase`/`DirectCollocationLink` graphs, event links,
complementarity audits, finite weighted stochastic scenarios, and declared
manifold retractions are bounded transcriptions: phase/event/scenario/chart
topology is immutable inside one epoch.

Continuous path certification is distinct from the existing off-grid audit.
`AffineBernsteinPathEnvelope` bounds affine residuals over the represented segment
by Bernstein convex hulls. `LipschitzPathEnvelope` combines declared total-time
derivative bounds with a covering radius. Unsupported sampled callbacks remain
non-certifying. `BoundedControlCertificatePlan` can certify only a finite
coefficient box with a valid convex or Lipschitz/interval relaxation and a
continuously feasible incumbent. Work-limit results and differential-evolution
searches remain non-certificates.

::: phydrax.control.plan_ilqr

---

::: phydrax.control.RadauCollocationDefects

---

::: phydrax.control.MultiphaseDirectCollocationProblem

---

::: phydrax.control.CertifiedPathConstraint

---

::: phydrax.control.certify_continuous_path_constraints

---

::: phydrax.control.BoundedControlCertificatePlan

---

::: phydrax.control.certify_bounded_control_optimum

## Deterministic nonlinear games

`phydrax.control.games` keeps a joint physical control vector, while
`PlayerControlPartition` records the rows owned by each ordered player. For a case
shape `C`, horizon `T`, `P` players, state size `n`, and joint-control size `m`,
deterministic evaluations use states `C + (T + 1, n)`, controls
`C + (T, m)`, and costs `C + (P, T)`. A player axis is never folded into a case
axis. All callbacks receive `DiscreteStepContext`; stage costs are discrete summands
and are not multiplied by interval duration.

| Capability | Primary public API | Decision class | Evidence ceiling and explicit non-goals |
|---|---|---|---|
| Physical policy evaluation | `DeterministicFeedbackGameProblem`, `evaluate_game_policy` | Simultaneous full-state joint feedback: every player acts from the same pre-transition state. | `GamePolicyEvaluation` reports the physical rollout, ordered costs, validity, and causal failure. It does not test unilateral optimality. |
| Nominal Nash residual | `ILQGameScaling`, `nominal_nash_residual` | The supplied policy and its one supplied nominal trajectory are fixed. | `NominalNashResidual` contains exact discrete adjoints, dynamics defects, and player-owned first-order rows in physical and dimensionless coordinates. `LOCAL_NOMINAL_NASH_STATIONARY` is a local first-order label, not feedback-Nash existence, uniqueness, or global convergence. |
| One local quadratic suggestion | `suggest_local_affine_game_policy` | A physical affine policy in deviations from the supplied nominal states and controls. | `LocalAffineGameSuggestion` uses first-order dynamics and exact cost Hessians once. It is neither an iterative solver nor a certificate for the nonlinear game. |
| Residual-globalized nonlinear iLQ | `plan_ilq_feedback_game`, `prepare_ilq_feedback_game`, `refresh_ilq_feedback_game`, `solve_prepared_ilq_feedback_game`, `solve_ilq_feedback_game` | Simultaneous full-state feedback within the represented local affine policy family. | `LocalNominalNashResult` accepts trials only through the original unregularized dimensionless nominal residual. Proximal regularization may form a direction but cannot change the merit, final residual, or claim. Success remains local nominal stationarity, not an exact nonlinear feedback-Nash or global result. |

The supporting public result and status types are
`GamePolicyEvaluation`, `GamePolicyEvaluationStatus`, `NominalNashResidual`,
`LocalAffineGamePolicy`, `LocalAffineGameSuggestion`,
`LocalAffineGameSuggestionStatus`, `ILQFeedbackGamePlan`,
`PreparedILQFeedbackGame`, `ILQFeedbackGameStatus`,
`ILQFeedbackGameTrialReason`, and `LocalNominalNashDiagnostics`.

## Constraint ownership and equilibrium concepts

`GameConstraintBlock` declares a `GameConstraintScope`, a `GameConstraintSite`,
participants, an optional owner, control dependencies, an equality flag, and one
fixed residual shape. Equalities use residual `= 0`; inequalities use residual
`<= 0`. The scopes are not interchangeable:

| Scope | Physical meaning | Multiplier ownership |
|---|---|---|
| `PLAYER_LOCAL` | Depends only on its one owning player's declared variables. | One private copy for the owner. |
| `PLAYER_OWNED_COUPLED` | Belongs to one player but may depend on participating opponents. | One private copy for the owner; this creates a player-specific feasible set. |
| `SHARED` | One physical residual is shared by declared participants. | A VE uses one common multiplier. A generic GNE uses one multiplier copy per participating player. |

`OpenLoopGameConstraints`, `GameConstraintLayout`, and `GameMultiplierLayout`
preserve physical-residual rows separately from multiplier copies.
`evaluate_game_feasibility` returns `GameFeasibilityEvidence` at the declared
`PATH`, `TERMINAL`, or `TRAJECTORY` sites. This is sampled feasibility only; it is
not a continuous-time safety certificate.

| Solution concept | Primary public API | Constraint and time semantics | Evidence ceiling |
|---|---|---|---|
| Convex open-loop variational equilibrium (VE) | `FiniteHorizonLQOpenLoopVEProblem`, `solve_open_loop_ve` | Finite-horizon affine dynamics, convex quadratic player costs, affine player-local and shared constraints after exact condensation, and one common multiplier for each shared row. | `OpenLoopVEResult` audits structure, phase-I feasibility, the VI, original-scale KKT residuals, an independent natural projection, and isolation. It is an open-loop VE, not a generic GNE or feedback Nash result. |
| Convex open-loop generalized Nash equilibrium (GNE) | `FiniteHorizonLQOpenLoopGNEProblem`, `solve_open_loop_gne` | The same physical shared row is evaluated once but has a player-specific multiplier copy; player-owned coupled rows remain private. No common-multiplier restriction is imposed. | `OpenLoopGNEResult` certifies original player-specific KKT evidence. A finite global GNE gap is available only when the optional convex unilateral best-response audit succeeds with complete numerical bounds; KKT success alone is not that global gap claim. |
| Nonlinear private open-loop KKT | `NonlinearOpenLoopGameProblem`, `solve_open_loop_game_kkt` | Discrete simultaneous dynamics with open-loop controls and only `PLAYER_LOCAL` or `PLAYER_OWNED_COUPLED` constraints. Physically shared rows are rejected. | `OpenLoopGameKKTResult` reports original unscaled feasibility, stationarity, dual, complementarity, constraint-qualification, and solver evidence. It is a local nominal Nash/GNE KKT candidate, not feedback equilibrium or global equilibrium. |
| Fixed-active-set feedback quasi-Nash model | `ConstrainedFeedbackGameProblem`, `FeedbackQuasiNashPlan`, `solve_feedback_quasi_nash_model` | A `LocalAffineGameSuggestion` plus explicitly supplied stagewise path-constraint residuals, state/control Jacobians, multiplier convention, and active mask. | `FeedbackQuasiNashResult` is one local piecewise-affine branch. It performs no active-set search and supplies no derivative through a switch. Even success is not an exact nonlinear feedback Nash result, an off-trajectory feasibility claim, or a global GNE certificate. |

Prepared repeated solves use `OpenLoopVEPlan`, `PreparedOpenLoopVE`,
`plan_open_loop_ve`, `prepare_open_loop_ve`, `refresh_open_loop_ve`, and
`solve_prepared_open_loop_ve`; `OpenLoopGNEPlan`, `PreparedOpenLoopGNE`,
`plan_open_loop_gne`, `prepare_open_loop_gne`, `refresh_open_loop_gne`, and
`solve_prepared_open_loop_gne`; or `OpenLoopGameKKTPlan`,
`PreparedOpenLoopGameKKT`, `plan_open_loop_game_kkt`,
`prepare_open_loop_game_kkt`, `refresh_open_loop_game_kkt`, and
`solve_prepared_open_loop_game_kkt`. Refresh preserves player, constraint,
multiplier, case, and horizon topology.

The constrained status inventory is `GameFeasibilityStatus`, `OpenLoopVEStatus`,
`OpenLoopGNEStatus`, `OpenLoopGameKKTStatus`, and
`FeedbackQuasiNashStatus`. Status success never widens the solution concept named
by its corresponding result.

## Stochastic control and games

The stochastic-control contracts distinguish a physical action from the noise
increment that follows it. For `N` paths and `T` steps,
`PreparedControlledNoise.increments` has `(N, T) + noise_shape`;
`ControlledPathBatch.states` has `(N, T + 1) + state_shape`; and actions have
`(N, T) + action_shape`. `independence_labels`, not raw path count, define the
independent units used by Monte Carlo summaries.

Information is also a first-class value. `FullStateInformation` exposes the supplied
state, `CentralizedObservationInformation` exposes only the supplied observation,
and `GaussianBelief` carries a checked mean/covariance pair. Finite common-information
policies use a separate prescription contract described below. None of these classes
infers information from array shape or gives a policy an undeclared latent state,
noise increment, or random key.

| Capability | Primary public API | Decision/information and stochastic class | Evidence ceiling |
|---|---|---|---|
| Fixed-policy rollout, risk, and comparison | `ControlledTransitionProblem`, `PreparedControlledNoise`, `rollout_feedback`, `evaluate_feedback_policy`, `compare_feedback_policies` | One full-state feedback action is chosen from `DiscreteStepContext` and current state before the current noise is exposed. Comparison requires verified common random numbers. | `FeedbackPolicyEvaluation` reports empirical risk and separately qualified `MonteCarloEvidence`; `PairedPolicyComparison` reports right-minus-left return. Neither claims policy optimality. |
| Additive-noise LQG control and games | `finite_horizon_lqg_state_feedback`, `finite_horizon_lqg_feedback_nash` | Exact discrete full-state certainty equivalence for exogenous zero-mean additive noise with explicit factor and covariance axes. The game has one common process and an explicit player cost axis. | Exact policy and quadratic expected-value trace corrections for the declared finite-horizon LQG class only. State/action-dependent noise and partial observation are outside these entry points. |
| Multiplicative-noise LQ control and games | `finite_horizon_multiplicative_lq_state_feedback`, `finite_horizon_multiplicative_lq_feedback_nash` | Noise enters as explicitly declared affine channels in state and joint action, with a full channel covariance at every stage. | Exact expected quadratic-affine recursion for the declared all-minimizer class, subject to covariance, curvature, rank, solve, stationarity, and Bellman gates. It is not additive LQG or belief feedback. |
| Centralized Gaussian-belief LQG | `CentralizedLQGProblem`, `finite_horizon_centralized_lqg` | One observation arrives before each action; `BeliefFeedbackPolicy` acts only on a `GaussianBelief`. Process and observation noise are zero mean, mutually independent across time, and uncorrelated with each other. | `CentralizedLQGResult` provides the deterministic Riccati policy, Kalman schedule, Joseph covariance updates, and exact trace evidence for this classical centralized model. Cross-correlated noise, action-dependent observations, decentralized information, and covariance repair are excluded. |
| Frozen-policy fitted Bellman evaluation | `FittedBellmanProblem`, `FittedBellmanPlan`, `fit_frozen_policy_bellman` | Fixed features and one unchanged policy are evaluated backward on disjoint training and holdout path batches. | `FittedBellmanResult` reports ranks, conditioning, original/ridge normal-equation residuals, and separate holdout Bellman residuals. It never improves or replaces the policy. `bridge_fitted_bellman_to_bsde` preserves physical actions separately from BSDE martingale integrands. |
| Single-agent stochastic maximum principle (SMP) | `StochasticMaximumPrincipleProblem`, `evaluate_stochastic_maximum_principle` | Supplied open-loop Euler paths, adjoint predictor, distinct martingale-integrand predictor, and caller-declared pre-increment information cells. | `StochasticMaximumPrincipleResult` is pathwise necessary-condition evidence. Conditional stationarity is an equal-independent-cluster empirical mean; sufficiency additionally requires explicitly checked convexity. There is no feedback, Markov-perfect, coverage, or population-optimality claim. |
| Multi-player stochastic SMP | `OpenLoopStochasticGameSMPProblem`, `evaluate_open_loop_stochastic_game_smp` | Every player has its own adjoint pair; only that player's owned rows of the joint-action Hamiltonian gradient are retained. | `OpenLoopStochasticGameSMPResult` is open-loop Nash SMP evidence on supplied paths. It does not construct a strategy or claim feedback Nash. |
| Bounded one-dimensional HJB reference | `BoundedUniformGrid1D`, `DiscreteHJBProblem`, `solve_discrete_hjb_reference`, `refine_discrete_hjb_reference` | One scalar state, a finite physical-action catalog, declared boundary/terminal tables, an explicit time grid, and an upwind/central finite-difference operator. | `DiscreteHJBResult` and `DiscreteHJBRefinementResult` gate residuals and one nested refinement only. They do not establish a continuum viscosity solution or a result outside the bounded grid. |
| Zero-sum HJBI reference | `DiscreteZeroSumHJBIProblem`, `solve_discrete_hjbi_reference`, `scalar_lq_hjbi_solution` | Separate minimizer/maximizer action catalogs. The declared `max_min` and `min_max` orders are both evaluated; neither is rewritten. | `DiscreteZeroSumHJBIResult` requires both discrete orders, refinement, and the Isaacs-gap gate. This is distinct from an HJB minimum and from all-minimizer coupled HJB. The scalar LQ helper is exact only for its stated analytic model and well-posed horizon. |
| Coupled all-minimizer HJB | `DiscreteCoupledHJBProblem`, `CoupledHJBPolicyIterationPlan`, `solve_coupled_hjb_reference` | One scalar state, simultaneous finite actions, and player-specific value equations. Jacobi or Gauss--Seidel update order, damping, starts, ties, and branches are explicit. | `DiscreteCoupledHJBResult` is local finite-grid feedback fixed-point evidence for the supplied starts. It does not establish uniqueness, a viscosity solution, or a global Nash equilibrium. |
| Frozen-sample policy-game SAA | `StochasticPolicyGameProblem`, `plan_stochastic_policy_game`, `prepare_stochastic_policy_game`, `solve_prepared_stochastic_policy_game` | A finite joint policy-parameter vector has player-owned rows. Raw complete-path player costs are differentiated on frozen training noise; disjoint holdout noise is evaluation-only. | `StochasticPolicyGameResult` certifies at most local unscaled SAA pseudo-gradient stationarity. Holdout cluster summaries are not a population bound, and the unconstrained root solve does not certify boundary KKT, feedback Nash, or population Nash. |

The exact LQ result/status inventory is
`FiniteHorizonLQGStateFeedbackResult`, `LQGStateFeedbackStatus`,
`FiniteHorizonLQGFeedbackNashResult`, `LQGFeedbackNashStatus`,
`FiniteHorizonMultiplicativeLQStateFeedbackResult`,
`FiniteHorizonMultiplicativeLQStateFeedbackDiagnostics`,
`MultiplicativeLQStateFeedbackStatus`,
`FiniteHorizonMultiplicativeLQFeedbackNashResult`,
`FiniteHorizonMultiplicativeLQFeedbackNashDiagnostics`, and
`MultiplicativeLQFeedbackNashStatus`.

The remaining stochastic evidence inventory is `ControlledPathBatch`,
`FeedbackPolicyEvaluationStatus`, `FittedBellmanPrepared`,
`FittedBellmanStatus`, `FittedBellmanBSDEBridge`,
`SMPCausalInformationEvidence`, `SMPPathClusterEvidence`,
`StochasticMaximumPrincipleStatus`, `GameSMPCausalInformationEvidence`,
`GameSMPPathClusterEvidence`, `OpenLoopStochasticGameSMPStatus`,
`DiscreteHJBEvidence`, `DiscreteHJBStatus`, `DiscreteHJBIEvidence`,
`DiscreteHJBIStatus`, `CoupledHJBBranchEvidence`,
`DiscreteCoupledHJBEvidence`, `DiscreteCoupledHJBStatus`,
`StochasticPolicyGamePlan`, `PreparedStochasticPolicyGame`, and
`StochasticPolicyGameStatus`.

### Empirical-risk and statistical boundaries

- `sample_role="training"` describes reused or selected data and therefore carries
  no coverage claim, even if a coverage method was requested.
- Holdout intervals retain their declared assumptions. Asymptotic-normal intervals
  are not finite-sample guarantees. Hoeffding evidence requires valid declared
  return bounds. A confidence interval for a fixed policy is not an optimality
  interval.
- Repeated paths with one `independence_label` form one cluster. Increasing the
  number of dependent paths does not increase the independent-cluster count.
- SAA stationarity concerns the frozen empirical objective. A fresh holdout
  evaluation diagnoses the accepted parameters but does not convert the empirical
  root into a population equilibrium.
- HJB/HJBI residual and refinement gates are deterministic finite-grid evidence,
  not Monte Carlo coverage. SMP conditional means are empirical necessary-condition
  evidence, not Bellman or HJB residuals.

## Mean-field, finite-population, and finite-state games

Mean-field APIs keep law support, weights, effective sample size, time support,
source-path identity, and flow identity explicit. `EmpiricalMeanField.particles`
has `sample_shape + (num_times,) + state_shape`; its weights and validity have
`sample_shape + (num_times,)`. Common-noise scenario and outer-iteration histories
add distinct leading axes rather than collapsing them into particle samples. A
frozen law, a fixed-point candidate, a common-noise conditional law, an $N$-player
continuation, a social planner, and a master equation are different mathematical
objects:

| Capability | Primary public API | Law/information semantics | Evidence ceiling |
|---|---|---|---|
| Frozen-law response | `FrozenLawBestResponseProblem`, `solve_frozen_law_best_response` | A control-adapted `MeanFieldBSDEProblem` is evaluated against exactly one supplied `EmpiricalMeanField`. | `FrozenLawBestResponseResult` reports BSDE, Hamiltonian, law-validity, and ESS evidence. It does not infer an induced law, establish best-response optimality, or claim an MFG. |
| Induced-law MFG fixed point | `MeanFieldGameFixedPointProblem`, `MeanFieldGameFixedPointPlan`, `solve_mean_field_game_fixed_point` | Each outer step evaluates the current frozen law, then requires a newly sourced independently induced law. Unit damping adopts that law directly; damping below one requires an identified `law_mixture` callback that constructs the convex measure mixture or an evidenced coupling. Particle coordinates from unrelated sources are never interpolated. | `MeanFieldGameFixedPointResult` certifies a valid response and law distance within tolerance only. It remains a fixed-capacity candidate, not an $N$-player, common-noise, MFC, or globally optimal result. |
| Finite-scenario common-noise MFG | `CommonNoiseMeanFieldProblem`, `CommonNoiseMeanFieldPlan`, `solve_common_noise_mean_field_fixed_point` | One conditional empirical law and one public history per common-noise atom; conditional laws are never replaced by their unconditional mixture. Damping below one uses an identified scenario-local law-mixture callback and never couples particles across common-noise atoms. ESS is computed after idiosyncratic weights are aggregated by declared independent cluster. | `CommonNoiseMeanFieldResult` requires every positive-probability scenario to pass in the same outer iteration. It is conditional-law candidate evidence, not an unconditional MFG or an equilibrium theorem. |
| Constrained MFG | `ConstrainedMeanFieldGameProblem`, `ConstrainedMeanFieldGamePlan`, `solve_constrained_mean_field_game` | `MeanFieldConstraintConcept` separates individual, aggregate-generic, and aggregate-variational constraints. Physical aggregate rows are evaluated once; multiplier copies follow the declared generic or common convention. Identified aggregate-constraint derivative evidence binds the induced law and exact multiplier vector, and its aggregate price contribution enters the original representative-agent stationarity residual. | `ConstrainedMeanFieldGameResult` adds sampled individual feasibility/stationarity and aggregate-law primal, dual, complementarity, and complete price-adjusted stationarity evidence to law consistency. It is a sampled KKT candidate, not continuous safety, global equilibrium, MFC, finite-$N$, or master-equation evidence. |
| Finite-$N$ continuation | `FinitePopulationGameProblem`, `FinitePopulationContinuationPlan`, `evaluate_finite_population_continuation` | A separately evaluated $N$-player profile and each feasible unilateral-deviation problem are anchored to one valid MFG fixed-point result. Dependence is clustered explicitly. | `FinitePopulationContinuationResult` emits epsilon-Nash evidence only when every numerical bound, simultaneous statistical bound, provenance check, and finite-law comparison is complete. An MFG fixed point alone never implies this result. |
| Mean-field control (MFC) planner | `MeanFieldControlProblem`, `MeanFieldExternality`, `evaluate_mean_field_control_planner` | One social planner acts on its law-generating paths. The measure externality is mandatory and is either identified analytic Lions-derivative data or identified finite-particle adjoint data with an explicit bias bound. | `MeanFieldControlResult` reports planner-stationarity, measure-adjoint, welfare, ESS, path-identity, and provenance evidence. Valid evidence does not mean residual convergence or an MFC optimum, and it is not an MFG equilibrium. |
| Finite-state common information | `FiniteStateCommonInformationGame`, `CommonInformationEquilibriumSelector`, `solve_common_information_game` | Pure prescriptions depend only on finite public state and a player's own private type. Beliefs, type transitions, public-observation transitions, Bayesian supports, and the deterministic branch selector are explicit. | `CommonInformationGameResult` is a finite pure-prescription common-information Markov-perfect candidate. A missing pure Bayes-consistent stage Nash profile is an error; no mixed-strategy or approximate fallback is substituted. |
| Finite-state master-equation reference | `FiniteStateMasterEquationProblem`, `FinitePopulationSimplexLattice`, `solve_finite_state_master_equation_reference` | Values are tabulated on a finite physical-state set and the exact empirical-law count simplex for a declared population size. The representative state and aggregate population transition are separate. | `FiniteStateMasterEquationResult` reports exact finite-lattice Bellman/action-minimum/simplex residuals and neighbor-transfer differences. Those differences are not Lions derivatives; there is no interpolation, continuous-law, common-noise, MFG, MFC, or global master-equation claim. |

Supporting evidence types include `MeanFieldIndividualConstraintEvidence`,
`FinitePopulationJointPolicyEvaluation`, `FinitePopulationBestResponseEvidence`,
`CommonInformationPolicy`, `CommonInformationStageEquilibria`,
`CommonInformationBayesEvidence`, `FiniteStateMasterEquationEvidence`, and their
corresponding result/status classes.

The corresponding status inventory is `FrozenLawBestResponseStatus`,
`MeanFieldGameFixedPointStatus`, `CommonNoiseMeanFieldStatus`,
`ConstrainedMeanFieldGameStatus`, `FinitePopulationContinuationStatus`,
`MeanFieldControlStatus`, and `FiniteStateMasterEquationStatus`.

## Failure, provenance, and composition rules

Every result must be interpreted through its own `valid` or `successful` value,
status, evidence object, and stated certificate/result label. Returned arrays from a
failed case are diagnostic data, not an applicable policy or equilibrium. Plan,
problem, time, partition, policy, feature, information, realization, coupling,
flow, callback, feasible-set, and selector identifiers record which numerical
object was actually evaluated.

No game or stochastic-control entry point silently clips controls, projects
covariances, changes an active set, repairs infeasible iterates, inserts a
pseudoinverse, reuses selected paths as holdout data, mixes conditional laws, or
falls back to a different solution concept. There is no universal combined solver:
compose only APIs whose decision class, information pattern, time model, stochastic
law, constraint ownership, and evidence claim match the problem being posed.
