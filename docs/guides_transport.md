# Optimal transport

Phydrax treats optimal transport as numerical measure geometry, not as a standalone
matching utility. Native transport consumes the same explicit measures used by
`phydrax.integration`, preserves physical mass and event structure, and returns
status, convergence diagnostics, and provenance suitable for differentiated SciML
workflows.

## 1. Start from the scientific event

An atom is one realization of the event being compared. Decide that event before
choosing a distance:

- a scalar observation is a one-dimensional event;
- a state vector is one vector event;
- a spatial field is one flattened whole-field event, not one independent event per
  mesh node;
- a trajectory is one complete space-time event when temporal dependence matters;
- a structured PyTree needs an explicit encoder when it does not already lower to one
  array leaf.

Flattening is a coordinate representation, not a license to discard geometry. For a
field on nonuniform nodes, include physical quadrature through the event encoding or
use the operator-aware metrics, which apply the square root of event weights before a
squared Euclidean cost. This makes squared distance equal the chosen discrete L2 norm.

## 2. Probability mass versus physical mass

`DiscreteMeasureTarget` and `WeightedSampleTarget` have two distinct contracts:

- normalized targets represent probability laws with physical mass one;
- unnormalized targets preserve a finite physical mass.

Balanced transport requires equal source and target physical mass. `discrete_problem`
checks that invariant and does not repair a mismatch by normalization. Normalize
explicitly only when the scientific question is about shape conditional on unit mass.
When unequal mass is itself scientific signal, use `unbalanced_problem` with explicit
source and target KL penalties; never catch a balanced mass error and retry silently.

Masks, zero weights, and support-validity arrays remove atoms without changing the
static array shape. Active non-finite coordinates and invalid weights are errors. A
zero-weight padded atom is inert; a positive-weight padded atom is not.

## 3. Choose and scale the ground geometry

The ground cost answers what moving one unit of mass means. Built-in choices are:

- `SquaredEuclideanCost` for already commensurate coordinates;
- `WeightedSquaredEuclideanCost` for explicit positive component scales;
- `PeriodicSquaredEuclideanCost` for shortest wrapped displacement;
- `PrecomputedCost` for externally computed finite costs.

Coordinate scaling and regularization cannot be tuned independently. Under a squared
cost, rescaling coordinates by `s` rescales costs and the comparable `epsilon` by
`s**2`. Use physical nondimensionalization or declared scales. Do not let a channel's
unit or mesh resolution dominate because its raw numbers happen to be larger.

A custom cost subclasses `AbstractGroundCost` and implements a JAX-transformable
`pairwise` calculation. Pointwise costs support both dense and blockwise execution.
A precomputed matrix supports only dense execution because arbitrary blocks cannot be
reconstructed from coordinates.

## 4. Choose the transport quantity

| Question | Native method | Important limitation |
| --- | --- | --- |
| Exact distance between scalar empirical laws | `wasserstein_distance_1d` | One-dimensional events only |
| Scalable comparison of vector-event laws | `sliced_wasserstein_distance` | Finite-projection estimator |
| Smooth multivariate discrepancy | `sinkhorn_divergence` | Entropically regularized and iterative |
| Finite regularized coupling or barycentric action | `Sinkhorn(problem)` | Balanced mass only |
| Aggregate several finite laws on declared support | `SinkhornBarycenter(problem)` | Common physical mass and strict measure weights |
| Locally optimize barycenter support | `FreeSupportBarycenter(inner_solver)` | Quadratic barycentric costs and explicit initialization |
| Unequal-mass spatial or intensity discrepancy | `unbalanced_sinkhorn_divergence` | Requires explicit source/target KL penalties |
| Differentiable sort/rank/quantile/top-k | soft-order functions | Relaxations, not hard order |

The regularized Sinkhorn objective contains self-interaction bias. Use
`sinkhorn_divergence` for a discrepancy expected to vanish on identical measures. Use
the raw `SinkhornResult` when the coupling, transport component, regularization
component, or barycentric action is the actual object of interest.

Sliced Wasserstein stores normalized projections. Supply explicit projections when
replay, common random numbers, or deterministic gradient comparison matters.

For unbalanced transport, PhydraX minimizes transport cost plus entropy KL relative
to the physical product measure and two independently weighted marginal KL terms.
The divergence performs cross, source-self, and target-self solves and adds the mass
correction required by that convention. Use
`spatial_unbalanced_sinkhorn_divergence` or
`SpatialUnbalancedSinkhornDivergenceTerm` only when total intensity, count, or spatial
mass is meaningful. Ordinary empirical predictive laws remain normalized.

## 5. Configure Sinkhorn deliberately

```python
import coordax as cx
import jax.numpy as jnp
import phydrax as phx

source = phx.integration.discrete(
    jnp.asarray([[0.0], [1.0], [2.0]]),
    cx.Field(jnp.asarray([1.0, 2.0, 1.0]), dims=("atom",)),
    axes="atom",
    normalized=True,
)
target = phx.integration.discrete(
    jnp.asarray([[0.5], [1.5]]),
    cx.Field(jnp.asarray([1.0, 1.0]), dims=("atom",)),
    axes="atom",
    normalized=True,
)
problem = phx.transport.discrete_problem(
    source,
    target,
    cost=phx.transport.SquaredEuclideanCost(),
)

solver = phx.transport.Sinkhorn(
    0.2,
    max_iterations=500,
    min_iterations=1,
    tolerance=1e-7,
    check_every=5,
    block_size=128,
    early_stop=False,
    store_history=False,
)
```

- `epsilon` controls smoothing in ground-cost units. Smaller values reduce bias but
  increase conditioning difficulty.
- `tolerance` applies to the normalized marginal residual. The result also reports the
  physical residual and primal-dual gap.
- `check_every` controls diagnostic frequency, not update frequency.
- `early_stop=False` gives fixed compiled work and records the first converged check.
  `early_stop=True` permits short-circuiting.
- `store_history=True` retains residuals for each check. Leave it off in large repeated
  training loops unless the trace is needed.
- `block_size=None` uses dense reductions. A positive block size evaluates compiled
  cost and plan blocks without retaining the complete pairwise matrix.

Dense and blockwise modes solve the same discrete problem. Blockwise mode bounds
working memory but increases loop overhead; choose it when the source-by-target matrix
is the bottleneck. `dense_plan()` always materializes that matrix, even after a
blockwise solve. Prefer matrix-free `apply_source_to_target`,
`apply_target_to_source`, and their barycentric variants.

## 6. Treat convergence as part of the value

Every solve returns a fixed-structure `SinkhornDiagnostics` record:

- status and iteration counts;
- first converged iteration;
- normalized and physical marginal residuals;
- dual residual and primal-dual gap;
- optional residual history.

`require_converged(result)` turns a failed status into a JAX-compatible error. Native
training terms, the distributional semigroup objective, and the particle transform do
this automatically. A finite scalar from a nonconverged iterate is not transport
evidence and must not be optimized or reported as though it were.

For Sinkhorn divergence, inspect the cross solve and both self solves independently.
One converged cross solve does not certify the debiased result.

## 7. Differentiate the numerical problem you ran

Native Sinkhorn uses ordinary JAX control flow. Reverse- and forward-mode derivatives
therefore differentiate the finite executed iteration map. They do not invoke an
implicit exact-solution derivative. The solver configuration is part of the gradient:

- changing `max_iterations`, `early_stop`, or `epsilon` can change the derivative;
- nonconvergence invalidates a gradient even when every array is finite;
- common projections or PRNG replay are required for controlled sliced-distance
  comparisons;
- a prepared reference reuses the fixed target self solve but still differentiates the
  source cross and source self terms.

Validate gradients with directional finite differences at the intended regularization
and iteration budget. Do not infer a high-order differentiability guarantee from a
successful first derivative.

## 8. Reuse a fixed reference safely

`prepare_sinkhorn_reference` lowers a fixed target and validates its target self solve
once. `sinkhorn_divergence_against` then evaluates only the source-to-target and source
self solves for each candidate. The prepared object retains its cost, solver, encoder
semantics, and mass tolerance. Reprepare it when any of those scientific contracts
change.

Prepared references are useful for model training, calibration against one empirical
dataset, and repeated spatial-density objectives. They are not a generic cache keyed
only by target array shape.

## 9. Scientific integrations

### Predictive laws and neural operators

`predictive_sinkhorn_divergence` compares raw empirical laws of complete vector events.
`operator_ensemble_sinkhorn_divergence` and
`operator_ensemble_sliced_wasserstein` preserve physical case axes, query masks,
quadrature, and channel geometry from `OperatorPredictiveField`. They reject
incompatible query geometry rather than comparing accidentally aligned arrays.

The `measure="quadrature"` policy weights the whole-field event by physical
quadrature. `measure="uniform"` is an explicit alternative, not a fallback.
Case reduction may be `"none"`, `"mean"`, or `"sum"`; stochastic sample axes are
never treated as physical cases.

### Functional training terms

- `SpatialSinkhornDivergenceTerm` compares a model-built finite physical measure with a
  prepared reference.
- `SpatialUnbalancedSinkhornDivergenceTerm` compares unequal-mass physical intensity
  measures and rejects nonconvergence or transported-mass collapse.
- `EmpiricalSinkhornDivergenceTerm` compares a model-generated empirical law with a
  prepared reference.
- `SlicedWassersteinTerm` provides a projection-based whole-event discrepancy.
- `SoftQuantileFunctional` penalizes regularized empirical quantiles.

Each is an evaluated scalar term and retains method diagnostics in `TermEvaluation`.
The model-facing provider or measure builder determines which functions are objective
variables; no implicit domain sampling is introduced.

### Distributional semigroup consistency

`SinkhornDistributionalSemigroupObjective` draws direct and independently composed
process-law samples using deterministic split or fold-in key semantics, constructs
complete operator-output events, and penalizes their Sinkhorn divergence. It is valid
only for probabilistic operator models declaring process uncertainty. It does not
replace pathwise cocycle consistency, because independent marginal draws do not share a
driving realization.

### Deterministic particle transform

`optimal_transport_ensemble_transform` maps normalized weighted particles to an
equal-weight barycentric ensemble. Dimensions before `particle_axis` are independent
physical cases; trailing dimensions form each particle event. The result exposes the
source and transformed means, their error, and the native coupling. This is a
deterministic ensemble transform, not categorical resampling and not an automatic
replacement for particle-filter genealogy contracts.

### Continuous density to optimized finite support

`SemidiscreteTransportProblem` binds a `DensityTarget`, its already-materialized
`IntegrationRealization`, and a finite target. `SemidiscreteSinkhorn` integrates the
soft c-transform and target marginal on that same realization at every dual update.
This prevents an integration rule or random sample batch from becoming an undisclosed
empirical source measure. Reusing the problem is exact replay of the numerical
experiment and gives common random numbers to support gradients.

Transport convergence and integration success are independent. Inspect
`result.diagnostics` for the target marginal solve and
`result.integration_diagnostics` for integration status, error estimate, evaluation
count, and provenance. A result remains marked as a fixed-realization approximation
even if the target marginal solve converges. Normalized density mass is one;
unnormalized density mass is estimated explicitly and must match the physical atom
weights.

`SemidiscreteQuantizer` turns target locations into a sensor, particle, or collocation
design objective and delegates updates to Optax. Supply a smooth
`support_transform` for domain constraints—for an interval, an affine sigmoid map is
typical. This composes the constraint with differentiation and never clips or repairs
an optimizer update. The outer optimization rejects nonconverged integration or
transport before consuming the objective.

### Finite-law barycentric aggregation

Use `fixed_support_barycenter_problem` when several finite predictive, posterior, or
ensemble laws have a meaningful Wasserstein aggregate on a scientifically declared
support. All inputs must have common physical mass; `measure_weights` express the
aggregation question and must be strictly positive and already sum to one.
`SinkhornBarycenter` pads unequal atom counts behind explicit masks, and dense and
blockwise reductions solve the same finite problem. The result keeps every
measure-to-barycenter coupling and per-measure objective rather than returning only an
average point cloud.

Use `FreeSupportBarycenter` only when support locations themselves are part of the
scientific optimization. The problem support is the explicit initialization. The
outer coupling-weighted coordinate update is valid only for squared or weighted
squared Euclidean costs, produces a local optimum, and retains every inner solve.
Support collapse is a terminal diagnostic, not an invitation to merge or jitter atoms
silently.

The UQ aggregation helpers return both a `DiscreteMeasureTarget` and native transport
results. `BarycenterObjectiveTerm` is appropriate only when the scalar weighted
aggregate of entropic transport objectives is the intended training quantity. It
rejects nonconverged solves.

## Approximate scalable balanced transport

`PositiveFeatureSinkhorn` is a distinct approximate backend for
`SquaredEuclideanCost`. It never changes the declared finite measure problem:
physical mass, masks, atom order, and event encoding remain those of
`DiscreteTransportProblem`. Instead it replaces
`exp(-cost / epsilon)` with replayable nonnegative Gaussian features and performs
matrix-free scaling and plan actions in feature rank.

```python
import jax.random as jr

features = phx.transport.GaussianPositiveFeatures(
    jr.key(0),
    512,
    num_probes=64,
    probe_tolerance=0.2,
)
solver = phx.transport.PositiveFeatureSinkhorn(
    0.2,
    features,
    max_iterations=500,
    tolerance=1e-7,
)
result = solver(problem, exact_ground_cost=True)
```

The key, requested rank, probe pairs, exact and approximate probe values, relative
errors, and zero-row counts live in `result.approximation`. The solve is not
converged when feature construction is non-finite, a positive-mass kernel row is
zero, or the declared probe tolerance fails. Increasing rank changes the numerical
problem; it is approximation evidence, not an invisible implementation setting.

`result.regularized_cost` is the entropic objective of the represented surrogate
kernel. When `exact_ground_cost=True`, `result.exact_transport_cost` additionally
evaluates the computed plan against the exact ground cost in compiled blocks. It does
not turn the surrogate solve into an exact solve. `dense_plan()` is the only API here
that explicitly forms the full plan; plan actions remain factorized.

Dense and blockwise exact Sinkhorn are two executions of the same kernel. Blockwise
execution is compiled bounded-working-memory evaluation, not host-callback,
out-of-core, or Python streaming. Positive features are approximate and record that
fact in `TransportProvenance.approximation`; the common balanced plan contract lets
divergence, UQ metrics, transport terms, the semigroup objective, and the particle
transform retain that provenance while rejecting nonconvergence.

## Exact finite-state dynamic transport

A Schrödinger bridge controls an entire Markov path law, not only a coupling at one
time. Use `phx.transport.dynamic.SchrodingerBridgeProblem` when all states are
enumerated, both endpoint measures use the same ordered support, and the reference
`AbstractTransitionKernel` provides normalized transition log probabilities on that
support. Supply the physical time grid and an explicit `StateSpaceStepContext`;
sampler-only transitions are intentionally rejected.

The log-IPF solver retains forward and backward potentials. Its Doob transform is an
ordinary `AbstractTransitionKernel`, so state-space inference can consume it without a
parallel stochastic API. The endpoint targets still carry physical mass, masks, case
axes, and provenance. Each case is solved independently. `result.marginal_probabilities`
is a probability path law, while `result.marginal_weights()` restores physical mass.

Use `BridgeInferenceAdapter` for a categorical initial-prior plus controlled-transition
view, `TerminalDistributionControlAdapter` for the exact path-KL control cost and
physical terminal residual, and `bridge_path_law_diagnostics` for empirical path-law
checks. All three reject a nonconverged result. Keyed path sampling folds in case,
member, and step identities; replay is exact and a larger one-dimensional sample count
preserves the existing prefix.

This exact family does not approximate diffusion or particle bridges and does not
learn a transport map. An unreachable positive endpoint receives
`TransportStatus.INFEASIBLE_SUPPORT`; it is never clipped, normalized away, or
repaired.

## 10. Capability boundary

The native subsystem intentionally omits automatic balanced-to-unbalanced fallback,
epsilon schedules, acceleration, Gromov--Wasserstein, fused GW, exact general
multidimensional assignment, Gaussian-mixture OT, and neural transport maps. Add one
of these only for a concrete PhydraX scientific contract; do not mirror another OT
package's class hierarchy.

## 11. Benchmark the relevant regime

Deterministic JSON harnesses separate core, scalable, soft-order, and scientific paths:

```bash
python -m tools.transport_benchmarks --smoke
python -m tools.positive_feature_transport_benchmarks --smoke
python -m tools.soft_transport_benchmarks --smoke
python -m tools.transport_scientific_benchmarks --smoke
python -m tools.semidiscrete_transport_benchmarks --smoke
python -m tools.schrodinger_bridge_benchmarks --smoke
```

The core harness reports compile-plus-first, steady solve, plan application, backward
execution, convergence, and result bytes for dense and blockwise modes. The soft
harness reports approximation, order, range, and gradient evidence. The scientific
harness exercises nonuniform spatial density, whole-field operator ensembles, and the
particle transform. The positive-feature harness reports rank, probe and small-problem
approximation errors, compile and steady runtimes, plan-action cost, explicit memory
fields, and gradient timing and finiteness.
The semidiscrete harness separately reports the fixed integration size and provenance,
dual residuals, integration and transport statuses, replay equality, support-gradient
cost, and result memory.
The Schrödinger bridge harness reports exact solve and keyed-sampling timings,
endpoint residual, path KL, empirical marginal residual, convergence, and reference
process provenance as JSON.
