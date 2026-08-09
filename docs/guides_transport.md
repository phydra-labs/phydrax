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
If unequal mass is meaningful, the current balanced solver is not the right model;
unbalanced transport is not provided by this release.

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
| Differentiable sort/rank/quantile/top-k | soft-order functions | Relaxations, not hard order |

The regularized Sinkhorn objective contains self-interaction bias. Use
`sinkhorn_divergence` for a discrepancy expected to vanish on identical measures. Use
the raw `SinkhornResult` when the coupling, transport component, regularization
component, or barycentric action is the actual object of interest.

Sliced Wasserstein stores normalized projections. Supply explicit projections when
replay, common random numbers, or deterministic gradient comparison matters.

## 5. Configure Sinkhorn deliberately

```python
import phydrax as phx

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
- `block_size=None` uses dense reductions. A positive block size streams cost and plan
  blocks for solving, statistics, and actions.

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

## 10. Capability boundary

The native subsystem intentionally omits unbalanced Sinkhorn, automatic
balanced-to-unbalanced fallback, low-rank transport, epsilon schedules, acceleration,
Gromov--Wasserstein, fused GW, barycenters, semi-discrete solvers, exact general
multidimensional assignment, Gaussian-mixture OT, and neural transport maps. Add one
of these only for a concrete Phydrax scientific contract; do not mirror another OT
package's class hierarchy.

## 11. Benchmark the relevant regime

Three deterministic JSON harnesses separate core, soft-order, and scientific paths:

```bash
python -m tools.transport_benchmarks --smoke
python -m tools.soft_transport_benchmarks --smoke
python -m tools.transport_scientific_benchmarks --smoke
```

The core harness reports compile-plus-first, steady solve, plan application, backward
execution, convergence, and result bytes for dense and blockwise modes. The soft
harness reports approximation, order, range, and gradient evidence. The scientific
harness exercises nonuniform spatial density, whole-field operator ensembles, and the
particle transform.
