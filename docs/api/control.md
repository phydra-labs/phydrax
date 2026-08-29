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

::: phydrax.control.MultipleShootingDecisionLayout

::: phydrax.control.MultipleShootingLinearization

::: phydrax.control.MultipleShootingHistory

::: phydrax.control.MultipleShootingResult

::: phydrax.control.linearize_multiple_shooting

::: phydrax.control.solve_multiple_shooting

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

The compiler produces both:

- a dense-compatible `MinimizationProblem` for an explicitly selected native method
  such as `FilterInteriorPoint`, retaining that method's dimension guard; and
- a `StructuredNonlinearProgram` with exact sparse Jacobian callbacks for
  `IpoptMinimize.solve_structured`.

No backend is selected from problem size and no sparse-to-dense fallback is used.
`DirectCollocationDerivativePolicy(hessian="limited-memory")` requests Ipopt's declared
limited-memory Hessian approximation. `"exact-sparse"` compiles and verifies the
Lagrangian Hessian and supplies its lower triangle.

`DirectCollocationResult` retains the physical decision, a `ControlTrajectory`, stage
times, states, rates, and controls, raw dynamics defects, every declared constraint
block, the normalized optimization result and KKT certificate, sparse topology counts,
and physical feasibility diagnostics. `DirectCollocationOffGridAudit` retains sampled
times, raw residuals, and per-case/per-interval defect and path-violation arrays rather
than only global maxima. Collocation constraints are enforced at their declared stages.
The audit evaluates additional piecewise-linear states and held controls, but
`certified=False`: neither site set is a continuous-time path certificate.

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
