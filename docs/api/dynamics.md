# Dynamical systems, identification, and chaos

`phydrax.dynamics` separates four contracts that are often coupled implicitly:

1. a local continuous or discrete **system law**;
2. a pathwise numerical **evolution** of that law on an explicit grid;
3. masked, labeled **trajectory data** used by estimators;
4. **analysis or identification results** with validity, status, approximation, and
   method provenance.

The separation is intentional. A `ContinuousSystem` does not select an ODE solver.
A `DiscreteSystem` does not claim that its transition is an exact flow map. An
`EvolutionTrajectory` is numerical output; a `TrajectoryData` is estimator input with
case axes, reset boundaries, weights, and optional controls or derivatives. Analysis
never infers a missing mask, changes an estimator after failure, or silently replaces a
geometric state by flattened Euclidean coordinates.

## Choosing a path

| Need | Public entry point | Result and boundary |
|---|---|---|
| Declare a flow or map | `ContinuousSystem`, `DiscreteSystem` | Local law with a `StateLayout`, optional `InputLayout`, and stable system ID. |
| Evolve a map | `DiscreteEvolution`, `evolve` | One transition per `IterationGrid` segment with node and backend status. |
| Evolve a flow | `solver.DiffraxEvolution`, `evolve` | Diffrax-backed path on a `TimeGrid`; solver and tolerance provenance remain explicit. |
| Normalize trajectory sources | `trajectory_data_from_*` | Preserves solver, control, stochastic-realization, and reset masks rather than treating padding as observations. |
| Linear reduced dynamics | `fit_dmd`, `fit_edmd` | Diagnosed rank, conditioning, residual, decoder, and optional controlled map. |
| Sparse governing equations | `fit_sindy` | Strong, discrete, integral, or weak formulation with a declared feature library and sparse regressor. |
| Structured sparse equations | `StructuredSequentialThresholdedLeastSquares` | Exact forbidden entries, shared groups, and linear coefficient equalities. |
| Rational or implicit dynamics | `fit_implicit_sindy` | Searches nonzero normalized implicit equations; never admits the zero equation as a fit. |
| Structured-grid PDE discovery | `fit_pde_find` | `StructuredPDEData`, named spatial derivatives, explicit library terms, masks, and coefficient evidence. |
| Sections and return maps | `find_section_crossings`, `section_return_map` | Fixed-capacity crossings with direction, bracket, refinement, overflow, and validity evidence. |
| Periodic orbit and Floquet stability | `solve_periodic_orbit`, `floquet_spectrum` | Map fixed points or flow multiple shooting; dense or matrix-free tangent paths are declared. |
| Continue a branch | `continue_branch` | Natural or pseudo-arclength continuation with adaptive retries, tangents, spectra, bifurcation flags, and branch IDs. |
| Lyapunov spectrum | `finite_time_lyapunov_spectrum` | Full or leading finite-time periodic-QR spectrum with exact resumable checkpoint. |
| Covariant directions | `covariant_directions` | CLVs or a full adjoint dual basis; stored-frame versus recomputation work is explicit. |
| Finite-amplitude instability | `finite_size_growth` | Rescaled finite-size growth over the state geometry's local retraction. |
| Recurrence statistics | `recurrence_quantification` | Recurrence and eligibility masks, Theiler window, line histograms, and RQA statistics. |
| Scalar chaos test | `zero_one_test` | Modified 0--1 frequency ensemble with RNG, fit lags, selected contiguous segment, and convergence spread. |
| Correlation dimension | `correlation_dimension` | Pair counts on declared radii, Theiler exclusion, fit mask, local slopes, and fit evidence. |
| Surrogate significance | `surrogate_significance` | Explicit null generator, alternative, RNG seed, plus-one p-value, and all surrogate statistics. |
| Aggregate uncertainty | `summarize_chaos_uncertainty` | Weighted samples over named initial-condition, parameter, noise, process, or numerical axes plus bootstrap evidence. |
| Connect a shadowing solver | `ShadowingSensitivityProblem`, `evaluate_shadowing_candidate` | Matrix-free tangent defects and response evidence for an externally supplied candidate; no hidden shadowing solve. |

## Layouts, systems, grids, and evolution

A `StateLayout` owns physical shape, component labels, state geometry, and a stable ID.
An `InputLayout` additionally labels each flattened input component as a control, forcing,
or parameter. System callbacks use `callback(coordinate, state, inputs, args)` when an
input layout is present and `callback(coordinate, state, args)` otherwise. Inputs are
supplied by an explicit `AbstractInputPolicy`; they are not captured from a global
schedule.

`continuous_model_system` binds an `AbstractArrayModel` into this system contract
without capturing trainable leaves in a closure. A flat model is state-only; a
structured `(state, input)` model requires a matching `InputLayout`. Input/output
sizes are checked at construction, and the resulting `ContinuousModelVectorField`
remains an explicit PyTree child for partitioning, optimization, and export.

`discrete_model_system` binds a pointwise `AbstractArrayModel` as a complete
next-state map. The binding is autonomous or consumes exactly one structured
`(state, input)` pair, retains the model as an explicit trainable PyTree child,
and declares the coordinate step that the map learned. `DiscreteEvolution`
rejects intervals outside that step contract before calling the model. Axis
models and variable-duration or stochastic transitions require different
contracts and are not inferred from array inheritance.

`TimeGrid` requires finite, strictly increasing physical times. `IterationGrid` requires
strictly increasing integer iteration labels. Both are `EvolutionGrid` contracts, but
physical-time normalization and iteration normalization remain distinct. An evolution
segment returns `EvolutionStep`; a tangent action returns `EvolutionTangentStep`.
`EvolutionTrajectory.valid` is a node mask, while `status` and `backend_status` retain
segment-level failure evidence.

::: phydrax.dynamics.StateLayout

::: phydrax.dynamics.InputLayout

::: phydrax.dynamics.ContinuousSystem

::: phydrax.dynamics.ContinuousModelVectorField


::: phydrax.dynamics.continuous_model_system

::: phydrax.dynamics.DiscreteSystem

::: phydrax.dynamics.DiscreteModelTransition

::: phydrax.dynamics.discrete_model_system

::: phydrax.dynamics.AbstractInputPolicy

::: phydrax.dynamics.CallableInputPolicy

::: phydrax.dynamics.TimeGrid

::: phydrax.dynamics.IterationGrid

::: phydrax.dynamics.AbstractEvolution

::: phydrax.dynamics.AbstractDifferentiableEvolution

::: phydrax.dynamics.DiscreteEvolution

::: phydrax.solver.DiffraxEvolution

::: phydrax.dynamics.EvolutionStep

::: phydrax.dynamics.EvolutionTangentStep

::: phydrax.dynamics.EvolutionTrajectory

::: phydrax.dynamics.EvolutionJacobianAction

::: phydrax.dynamics.evolve

## Trajectory data and source adapters

`TrajectoryData` has shape `case_shape + (capacity,) + state_shape`. `sample_valid`
selects observed nodes. `transition_valid` separately selects adjacent evolution pairs;
`reset_mask` prevents delays, derivatives, weak windows, and rollout scoring from crossing
trajectory boundaries. `weights` are sample evidence, not a padding mask. Inputs declare
whether they align with samples or transitions. Derivatives carry a separate validity
mask.

Source adapters preserve the source contract:

- `trajectory_data_from_evolution` retains canonical evolution node and segment status;
- `trajectory_data_from_fixed_step` requires an explicit retained-state projection,
  projection ID, and `StateLayout`; it preserves times and validity without compacting
  rejected samples and rejects final-only one-sample retention;
- `trajectory_data_from_differential_solution` retains deterministic, DAE, or ensemble
  solver sample axes, including DAE state rates and their componentwise validity,
  delay solutions, and structured differential solutions;
- `trajectory_data_from_control` attaches source-aligned controls and their layout;
- `trajectory_data_from_stochastic` retains case and realization axes.

::: phydrax.dynamics.TrajectoryData

::: phydrax.dynamics.TrajectoryTransitions

::: phydrax.dynamics.identification.trajectory_data_from_evolution
::: phydrax.dynamics.identification.trajectory_data_from_fixed_step


::: phydrax.dynamics.identification.trajectory_data_from_differential_solution

::: phydrax.dynamics.identification.trajectory_data_from_control

::: phydrax.dynamics.identification.trajectory_data_from_stochastic

::: phydrax.dynamics.identification.delay_embed

::: phydrax.dynamics.identification.finite_difference_derivative

::: phydrax.dynamics.identification.local_polynomial_derivative

::: phydrax.dynamics.identification.bspline_derivative

## Learned discrete maps

`fit_discrete_model` learns deterministic fixed-step Euclidean maps directly
from `TrajectoryData`. Windows are indexed lazily by parent case and start;
overlapping arrays are not materialized. Every active prefix must preserve
sample and transition validity, avoid resets, use the declared coordinate step,
and consume each sample- or transition-aligned input exactly once. Invalid
padding is sanitized before model evaluation, while a non-finite or
out-of-geometry model or reference state fails closed.

A `DiscreteModelRolloutPolicy` owns one static maximum horizon, a traced active
horizon, global BPTT truncation, and rematerialization. Objective coefficients
are normalized over active nodes before the window evidence
`sqrt(weight[start] * weight[start + horizon])` is applied. Supervised,
deterministic reference-branch, and residual objectives share one authored
recurrent step. Full, prefix, chunked, rematerialized, and resumed execution are
required to agree; no JAXPR transformation or inferred carry is involved.

`gradient_accumulation=K` evaluates `K` independently keyed window batches at
fixed model, optimizer, target, and rollout-schedule state. Each objective emits
an evidence-weighted numerator and support; numerator gradients and supports are
summed and normalized once before the Optax update. This is exactly equivalent
to the pooled evidence-weighted objective for unequal batches and final epoch
tails. A zero-support group is consumed without advancing optimizer, target,
validation, callback, history, or checkpoint state. `steps` counts accepted
optimizer updates, while `TrainingProgress.microstep` counts consumed batches.

The first training contract accepts real `float32` or `float64` pointwise
models and Euclidean state layouts. Variable steps, stochastic transitions,
non-Euclidean discrepancies, and low-precision parameters are rejected rather
than assigned implicit semantics.
Checkpointed fits require a stable `model_id`; array shapes and Python type
alone are not accepted as the identity of static activations, bindings, or
architecture hyperparameters.


::: phydrax.dynamics.identification.DiscreteModelRolloutPolicy

::: phydrax.dynamics.identification.SupervisedDiscreteModelObjective

::: phydrax.dynamics.identification.ReferenceBranchDiscreteModelObjective

::: phydrax.dynamics.identification.ResidualDiscreteModelObjective

::: phydrax.dynamics.identification.DiscreteModelValidationPolicy

::: phydrax.dynamics.identification.DiscreteModelFitHistory

::: phydrax.dynamics.identification.DiscreteModelFitResult

::: phydrax.dynamics.identification.fit_discrete_model

## Feature libraries, DMD, and EDMD

Feature libraries preserve `StateLayout`, optional `InputLayout`, ordered feature names,
and a stable library ID. Polynomial enumeration is a bounded weighted total-degree lower
set rather than an accidental Cartesian explosion. `CustomFeatureLibrary` is the extension
boundary. Concatenation, tensor products, symmetry averaging, and declared linear feature
transforms remain compositional.

DMD solves one masked weighted transition regression. Controlled DMD includes the input
block in that same solve. EDMD evolves features and fits a separate physical-state decoder;
its Koopman feature map and state decoder are not conflated. Conversion to an executable
`DiscreteSystem` requires an ambient Euclidean state layout because an arbitrary manifold
needs a declared identification method.

::: phydrax.dynamics.identification.AbstractFeatureLibrary

::: phydrax.dynamics.identification.PolynomialFeatureLibrary

::: phydrax.dynamics.identification.FourierFeatureLibrary

::: phydrax.dynamics.identification.CustomFeatureLibrary

::: phydrax.dynamics.identification.ConcatenatedFeatureLibrary

::: phydrax.dynamics.identification.TensorProductFeatureLibrary

::: phydrax.dynamics.identification.SymmetryAveragedFeatureLibrary

::: phydrax.dynamics.identification.LinearTransformedFeatureLibrary

::: phydrax.dynamics.identification.fit_dmd

::: phydrax.dynamics.identification.DMDResult

::: phydrax.dynamics.identification.DMDDiagnostics

::: phydrax.dynamics.identification.fit_edmd

::: phydrax.dynamics.identification.EDMDResult

::: phydrax.dynamics.identification.EDMDDiagnostics

## SINDy formulations and sparse regression

A `SINDyProblem` pairs data, a feature library, and exactly one formulation:

- strong form consumes attached pointwise derivatives;
- discrete form regresses one-step state transitions;
- integral form uses endpoint differences and left or trapezoidal quadrature;
- weak form integrates compact test windows and never crosses invalid or reset segments.

`SequentialThresholdedLeastSquares` and `SR3Regression` return coefficients in physical
feature coordinates. Feature scaling changes conditioning, not the reported equation.
Unbiased refits, thresholds, ridge values, iteration histories, rank, condition, sample
counts, and convergence remain present. Selection embargoes overlapping temporal windows;
ensemble inclusion probabilities are computed only over valid fitted members.

::: phydrax.dynamics.identification.SINDyProblem

::: phydrax.dynamics.identification.StrongSINDyFormulation

::: phydrax.dynamics.identification.DiscreteSINDyFormulation

::: phydrax.dynamics.identification.IntegralSINDyFormulation

::: phydrax.dynamics.identification.WeakSINDyFormulation

::: phydrax.dynamics.identification.build_sindy_design

::: phydrax.dynamics.identification.SINDyDesign

::: phydrax.dynamics.identification.AbstractSparseRegression

::: phydrax.dynamics.identification.SequentialThresholdedLeastSquares

::: phydrax.dynamics.identification.SR3Regression

::: phydrax.dynamics.identification.fit_sindy

::: phydrax.dynamics.identification.SINDyResult

::: phydrax.dynamics.identification.SINDySelectionPolicy

::: phydrax.dynamics.identification.select_sindy_model

::: phydrax.dynamics.identification.SINDySelectionResult

::: phydrax.dynamics.identification.fit_ensemble_sindy

::: phydrax.dynamics.identification.EnsembleSINDyResult

## Structured, implicit, and PDE identification

`CoefficientStructure` is an exact contract over the physical coefficient matrix:
forbidden entries remain zero, named groups share activity, and a
`LinearCoefficientConstraint` adds declared equalities. The structured regressor solves
its constrained active-set system directly; it does not threshold first and repair later.

Implicit SINDy augments state features with attached derivatives and searches each
candidate target-feature normalization. Every candidate, including rejected candidates,
keeps its score, residual, support, and regression result. PDE-FIND uses
`StructuredPDEData` with named coordinate axes and an explicit time axis. Spatial
finite-difference derivatives honor boundary policy; feature terms and target derivatives
remain named objects.

::: phydrax.dynamics.identification.CoefficientStructure

::: phydrax.dynamics.identification.LinearCoefficientConstraint

::: phydrax.dynamics.identification.StructuredSequentialThresholdedLeastSquares

::: phydrax.dynamics.identification.named_coefficient_constraint

::: phydrax.dynamics.identification.shared_feature_groups

::: phydrax.dynamics.identification.ImplicitFeatureLibrary

::: phydrax.dynamics.identification.PolynomialImplicitFeatureLibrary

::: phydrax.dynamics.identification.ImplicitSINDyProblem

::: phydrax.dynamics.identification.fit_implicit_sindy

::: phydrax.dynamics.identification.ImplicitSINDyResult

::: phydrax.dynamics.identification.StructuredPDEData

::: phydrax.dynamics.identification.FiniteDifferencePDEDerivative

::: phydrax.dynamics.identification.PolynomialPDELibrary

::: phydrax.dynamics.identification.PDEIdentificationProblem

::: phydrax.dynamics.identification.fit_pde_find

::: phydrax.dynamics.identification.PDEIdentificationResult

## Sections, periodic orbits, and Floquet analysis

Section crossing uses endpoint signs only on valid trajectory transitions. Direction,
tangency tolerance, refinement method, coordinate tolerance, maximum iterations, and
fixed output capacity are all declared. Overflow is reported; crossings are not silently
truncated. Interpolation refines within saved data. Evolution refinement re-integrates the
bracket with the supplied evolution.

Periodic-map problems solve a fixed point. Periodic-flow problems use multiple shooting
with continuity and a declared phase condition. `PeriodicOrbitResidual` exposes the same
equations as a `NonlinearSystemProblem`; `solve_periodic_orbit` delegates Newton,
globalization, linear solve evidence, and work accounting to `phydrax.nonlinear`.
Floquet analysis routes dense or matrix-free monodromy operators through the shared
general-eigen runtime. For a complete autonomous-flow spectrum, the neutral multiplier
is removed from stability classification only when its distance from one is within the
declared tolerance; otherwise the result is invalid with
`FLOQUET_NEUTRAL_MISSING`. Partial leading spectra do not fabricate a neutral mode.

::: phydrax.dynamics.analysis.AffineSection

::: phydrax.dynamics.analysis.CallableSection

::: phydrax.dynamics.analysis.find_section_crossings

::: phydrax.dynamics.analysis.SectionCrossings

::: phydrax.dynamics.analysis.section_return_map

::: phydrax.dynamics.analysis.SectionReturnMap

::: phydrax.dynamics.analysis.PeriodicOrbitProblem
::: phydrax.dynamics.analysis.PeriodicOrbitResidual


::: phydrax.dynamics.analysis.ComponentPhaseCondition

::: phydrax.dynamics.analysis.OrthogonalityPhaseCondition

::: phydrax.dynamics.analysis.solve_periodic_orbit

::: phydrax.dynamics.analysis.PeriodicOrbitResult

::: phydrax.dynamics.analysis.monodromy_action

::: phydrax.dynamics.analysis.floquet_spectrum

::: phydrax.dynamics.analysis.FloquetResult
::: phydrax.dynamics.analysis.RelativeEquilibriumProblem

::: phydrax.dynamics.analysis.RelativePeriodicOrbitProblem


## Continuation boundary

Generic residual-curve continuation and local bifurcation certification are owned by
`phydrax.continuation`. Dynamics analysis supplies periodic-orbit and Floquet operators
that may be adapted into those workflows; it does not maintain a second continuation
runtime. See [API → Continuation and bifurcation](continuation.md) and the
[advanced solver workflow](../cookbook/advanced_solvers.md).

## Lyapunov, covariant, finite-size, and recurrence diagnostics

`finite_time_lyapunov_spectrum` advances the supplied evolution and periodically
orthonormalizes tangent vectors. Burn-in and reporting cadence count declared grid
segments; normalization uses physical elapsed time on `TimeGrid` and iteration distance
on `IterationGrid`. A checkpoint contains the complete frame, accumulated log stretch,
state, schedule, and numerical provenance needed for exact continuation.

`covariant_directions` performs a Ginelli backward triangular solve. `memory_mode="store"`
keeps every forward QR frame and uses linear work. `memory_mode="recompute"` rebuilds
prefixes and trades quadratic tangent work for constant QR-history working storage.
Returned direction arrays still consume output storage. `backward_discard` marks terminal
boundary contamination, while `backward_convergence_drift` compares independent terminal
coefficient seeds. A small covariance residual alone is not treated as finite-window
convergence.

Finite-size growth is not the infinitesimal Lyapunov spectrum. It evolves perturbed paths,
measures local-retraction separation at the declared amplitude, and rescales on a declared
cadence. RQA returns both recurrence and eligibility masks. The Theiler window excludes
temporally close pairs before line statistics are computed.

`recurrence_seed_candidates` turns a bounded unbatched trajectory into ordered,
temporally separated shooting seeds; it preflights the quadratic pair-distance memory
and rejects requests beyond the structural pair capacity. Edge tracking evolves two
opposite-outcome states over one fixed horizon and bisects their initial-condition
bracket. Since its classifier is callable, `EdgeTrackingProblem` requires an explicit
`problem_id`. Relative-equilibrium and relative-orbit problems likewise require
explicit IDs for their vector-field, generator, group-action, and phase callables.
Validity, fixed iteration history, bracket width, and terminal status remain explicit.

::: phydrax.dynamics.analysis.finite_time_lyapunov_spectrum

::: phydrax.dynamics.analysis.LyapunovSpectrumCheckpoint

::: phydrax.dynamics.analysis.LyapunovSpectrumResult

::: phydrax.dynamics.analysis.kaplan_yorke_dimension

::: phydrax.dynamics.analysis.covariant_directions

::: phydrax.dynamics.analysis.CovariantDirectionResult

::: phydrax.dynamics.analysis.finite_size_growth

::: phydrax.dynamics.analysis.FiniteSizeGrowthResult

::: phydrax.dynamics.analysis.recurrence_quantification

::: phydrax.dynamics.analysis.RecurrenceQuantificationResult
::: phydrax.dynamics.analysis.recurrence_seed_candidates

::: phydrax.dynamics.analysis.RecurrenceSeedCandidates

::: phydrax.dynamics.analysis.EdgeTrackingProblem

::: phydrax.dynamics.analysis.track_basin_edge

::: phydrax.dynamics.analysis.EdgeTrackingResult


## Statistical chaos diagnostics and uncertainty

The modified 0--1 test chooses a seeded frequency ensemble, uses the longest contiguous
valid segment after burn-in, records every frequency statistic and displacement curve,
and reports the median and median absolute deviation. Correlation dimension uses
Grassberger--Procaccia pair counts over caller-supplied radii. Its Theiler window, fit
indices, fit mask, local slopes, pair count, and coefficient of determination are result
evidence; the returned slope is not an automatic proof of a scaling regime.

Surrogate significance supports shuffling, Fourier phase randomization, and amplitude-
adjusted Fourier transforms. It stores every surrogate statistic and uses a plus-one
finite-sample p-value. `summarize_chaos_uncertainty` aggregates already computed scalar
diagnostics over explicit uncertainty axes. It does not manufacture initial-condition,
parameter, process-noise, or numerical-tolerance samples; those cases must come from the
corresponding solver or ensemble workflow.

::: phydrax.dynamics.analysis.zero_one_test

::: phydrax.dynamics.analysis.ZeroOneTestResult

::: phydrax.dynamics.analysis.correlation_dimension

::: phydrax.dynamics.analysis.CorrelationDimensionResult

::: phydrax.dynamics.analysis.surrogate_significance

::: phydrax.dynamics.analysis.SurrogateSignificanceResult

::: phydrax.dynamics.analysis.summarize_chaos_uncertainty

::: phydrax.dynamics.analysis.ChaosUncertaintyResult

## Shadowing sensitivity boundary

Phydrax exposes a solver boundary rather than claiming a universal shadowing algorithm.
`ShadowingSensitivityProblem` declares an evolution, an integrated endpoint parameter
forcing, an observable, optional observable derivatives, and optional neutral flow
direction. `evaluate_shadowing_candidate` evaluates one externally supplied tangent path
and time-dilation path. It returns dynamic defects, boundary residual, neutral-direction
inner products, quadrature weights, directional observable response, masks, and status.
`least_squares_residual()` supplies the residual vector needed by least-squares shadowing
or NILSS implementations. No optimization, regularization, segment preconditioner, or
convergence claim is hidden behind this evaluation call.

::: phydrax.dynamics.analysis.ShadowingSensitivityProblem

::: phydrax.dynamics.analysis.evaluate_shadowing_candidate

::: phydrax.dynamics.analysis.ShadowingCandidateResult

## Interoperability and hard boundaries

- **Structured, DAE, and delay systems:** differential, DAE, delay, rough, memory,
  hybrid, and semidiscrete solver results enter through
  `trajectory_data_from_differential_solution`. Their solver meaning stays in the
  source ID and masks. DAE rates retain their own validity, and a delay history is not
  inferred from a flat state vector.
- **Controlled dynamics:** `trajectory_data_from_control` attaches controls with
  transition alignment. Controlled DMD and controlled SINDy then use the same
  `InputLayout`; autonomous fitting never silently drops supplied controls.
- **Stochastic paths:** `trajectory_data_from_stochastic` preserves realization axes.
  Fit each realization, pool only with a declared case policy, and aggregate diagnostic
  uncertainty with a `noise` or `process` source axis. A deterministic Lyapunov spectrum
  is not a stochastic Lyapunov exponent.
- **Geometric states:** evolution, tangent actions, finite-size separation, and section
  refinement use `StateLayout.geometry`. Ambient DMD/EDMD and polynomial SINDy may fit
  coordinates, but conversion to a physical system is rejected unless the state geometry
  is Euclidean or a structured identification method is supplied.
- **Operator and PDE workflows:** neural operators can generate or denoise trajectories
  before construction of `TrajectoryData`; they do not change estimator masks. PDE-FIND
  consumes `StructuredPDEData` with named time and spatial axes, while semidiscrete PDE
  evolution uses the ordinary system/evolution analysis path. These are complementary,
  not interchangeable, representations.
- **Native ML and learned path models:** `phydrax.ml.MLBatch` is a generic
  case/sample contract; it does not infer temporal adjacency or resets from
  `TrajectoryData`. `KoopmanTemporalOperator` is a learned field operator, while
  `LogSignatureControl` and `SignaturePDEKernel` encode rough controls or whole
  paths rather than pointwise Markov states. Crossing these boundaries requires
  an explicit sample/transition choice or history embedding that preserves masks
  and reset boundaries; it is not an automatic estimator substitution.
- **Dense guards:** recurrence matrices and correlation pair distances are quadratic in
  samples and enforce `max_samples`. Dense periodic, continuation, and monodromy paths
  enforce their dimension guards. Choose matrix-free tangent actions or smaller declared
  outputs rather than bypassing a guard accidentally.

## Reproducible substrate benchmarks

Run `python -m tools.dynamics_benchmarks` from the project root. With no extra
flags, the existing baseline remains unchanged: exact Lorenz strong-SINDy
recovery and high-dimensional leading Lyapunov/covariant analysis through JVP
tangent actions. Its JSON output includes warm-run timing, array working-set
bytes, sparse support precision/recall, coefficient and exponent errors,
covariance defects, tangent-evaluation counts, convergence/status evidence,
environment, and one aggregate `passed` flag. It never constructs a dense
Jacobian in the high-dimensional case. The checked reference run is stored at
`benchmarks/dynamics_substrate.json`.

Opt in to learned thermodynamic comparisons with
`--scenarios deterministic`, `--scenarios stochastic`, or
`--scenarios thermodynamic`. `--architectures`, `--seeds`, and `--quick`
select model families, deterministic repetitions, and smoke-scale training.
Per-model records include parameter counts; compilation, training, and inference
times; derivative or transition likelihood errors; rollout or stationary-moment
errors; covariance error; and energy diagnostics where defined. The benchmark
reports comparative evidence and never treats one quick random seed as a
superiority claim.
