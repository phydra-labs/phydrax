# Reduced-order modeling

PhydraX treats a reduced model as a composition of independently owned scientific
objects: a physical representation, a reduced law or residual, an existing solver,
an optional hyperreduction, an input-support gate, and evidence. It does not use one
universal ROM trainer.

## Distinct model families

Keep these contracts separate:

- intrusive Galerkin and Petrov--Galerkin project a declared operator or residual;
- LSPG minimizes a declared time-discrete residual;
- identified dynamics fit an executable reduced continuous or discrete system;
- coefficient surrogates map parameters directly to reduced coordinates;
- nonlinear charts decode latent coordinates into physical fields;
- sensor-history models estimate state from observations.

A basis alone is not an executable ROM. A sensor estimator is not an autonomous
flow. A POD energy fraction is not an error certificate.

## Physical basis artifacts

Fit array POD with `phydrax.ml.decomposition.POD`, then bind the result to its
physical state, support, measure, geometry, sources, and evidence with
`reduced_basis_from_subspace_model`. General vector-space bases may be supplied as
`LinearSubspace` values.

The adapter reads physical components derived from the model's current
authoritative weighted basis, preserving the declared state/support/measure/
geometry identities and affine offset. Intrusive ROM construction still refuses
incomplete positive physical support. Projector-mode fit differentiation does not
admit basis-sensitive reduced-model differentiation: request basis mode with its
isolation and pivot conditions for a direct fit-basis derivative. Learned basis
replacement is prediction-only, not a new certified spectral fit.

`ReducedBasisArtifact.role` distinguishes state, nonlinear-term, residual, and ROQ
bases. These roles are not interchangeable.

An affine state representation has the form:

```text
u(mu) = g(mu) + P a(mu)
```

The homogeneous basis `P` and the lift `g` remain separate. Boundary lifts are not
inserted as basis vectors.

## Trial and test reduction

`TrialTestReduction` stores two `ConstraintMap` values:

- the trial map prolongs reduced coordinates into the full state space;
- the test map pulls full residual covectors into the reduced test dual.

For a full operator `A`, the reduced term is:

```text
Ar = R A P
```

where `R` is the test dual pullback. PhydraX never substitutes a Euclidean transpose
based only on matching array shapes.

The current affine route requires equal trial and test reduced dimensions and one
fixed reference support. Same-sized but differently identified meshes are refused.

## Affine offline and online execution

`AffineLinearROMProblem` declares ordered full operator, right-hand-side, lift, and
observation terms. Preparation computes:

```text
Ar,q    = R Aq P
br,j    = R bj
cr,q,p  = R Aq gp
```

For coefficients `theta`, `eta`, and `gamma`, online assembly is:

```text
Ar = sum_q theta_q Ar,q
fr = sum_j eta_j br,j - sum_q,p theta_q gamma_p cr,q,p
```

`PreparedAffineLinearROM.evaluate` assembles only reduced arrays and delegates the
solve to `phydrax.linalg`. It has no truth callback. Full-state reconstruction is
optional and separately measurable.

`ArrayAffineCoefficientMap` is a portable bounded affine parameter map. Applications
with other exact coefficient laws implement `AbstractAffineCoefficientMap` and bind
input schema, units, term order, support, and artifact identity.

## Support and fidelity

Support is assessed before reduced assembly. An unsupported input returns an invalid
result without solving. The prepared model also binds state, support, measure,
geometry, reduction, coefficient-map, and numeric-revision identities.

`AffineLinearROMFidelityEvaluator` exposes a prepared model as one fidelity level. It
never performs or admits truth fallback. The fidelity validity bit is the conjunction
of input support and native solve success.

Truth evaluation is supplied independently to `audit_affine_linear_rom`, which
reports error in the declared full-space norm. Audit does not alter the reduced
execution.

## Certification

`prepare_residual_dual_norm` constructs full residual atoms for affine RHS, lift, and
trial contributions, then stores a factored dual-norm Gram representation. Online
residual norm evaluation uses reduced coefficients only.

`ArrayAffineStabilityBound` is valid only for the exact operator family, error space,
support, and evidence identity supplied at construction. For a valid coercive model,
PhydraX reports the absolute bound:

```text
state error in X <= residual dual norm in X' / stability lower bound
```

It does not silently normalize by a truth-state norm. It does not claim a QoI bound.
Hyperreduced models cannot reuse the affine certificate without an additional
rigorous hyperreduction-defect term.

## Identified reduced dynamics

Use one `CasePartitionManifest` before fitting scaling, POD, derivatives, feature
libraries, or regularization. `partition_trajectory_data` applies this membership to
canonical `TrajectoryData` without re-splitting.

`project_trajectory_data` maps states and derivatives through an orthonormal physical
basis. `OperatorInferenceFeatureLibrary` has exactly these blocks:

```text
constant, state, input, unique symmetric state-quadratic monomials
```

State-input and input-quadratic terms are absent. `DenseBlockRidgeRegression` applies
one regularization value per block through an augmented native least-squares solve;
it does not form normal equations.

A successful identification result becomes an existing `ContinuousSystem` or
`DiscreteSystem`. `IdentifiedReducedDynamics` composes that system with encoding and
reconstruction; integration remains solver-owned.

Report projection floor, equation residual, reduced rollout error, reconstructed
physical rollout error, and support separately.

## Nonlinear projection

`FullResidualGalerkin` is a mathematically exact but full-order-assisted reference. It
expands the reduced state, evaluates the full residual, and applies the test dual
pullback. It is not a reduced-only performance route.

`ReducedLSPGProblem` minimizes the full time-discrete residual through the existing
nonlinear least-squares runtime. Its residual is whitened in the declared physical
dual norm.

## Reduced regions of coupled problems

One region of a spatial coupled problem (`phydrax.solver.coupling`) can be replaced by
its `FullResidualGalerkin` model while every interface binding, law, parameter binding,
and observation of the plan stays unchanged:

```python
cpl = phx.solver.coupling
provider = cpl.ComponentResidualProvider(triangles, field="u")
pod = phx.ml.decomposition.PhysicalPODPlan(rank, centered=False).fit(
    provider.state_space, snapshots, source_artifact_ids=("training-solves",)
)
basis = phx.rom.ReducedBasisArtifact(
    pod.subspace,
    role="state",
    state_contract_id=provider.residual_id,
    support_id=provider.support_id,
    measure_id="euclidean-coordinates",
    geometry_id=provider.geometry_id,
    source_artifact_ids=("training-solves",),
)
galerkin = phx.rom.FullResidualGalerkin(
    phx.rom.trial_test_reduction_from_bases(basis), provider
)
reduced = cpl.ReducedComponent("triangles", galerkin)
```

`ComponentResidualProvider` publishes the steady residual of one single-field
component on its solve coordinates (the free degrees of freedom after the owner's
boundary elimination). Its `support_id` is the field's discrete space and its
`geometry_id` the component's owner identity, so a basis fitted on another component,
mesh, or field is refused by `FullResidualGalerkin`. The snapshots are the
component's block of full-order `solve_coupled_problem` solutions at training
parameters, `solution.state[i][0]` for the component's position `i` in the prepared
chart.

`ReducedComponent(name, galerkin)` keeps the full component's name. Its state block
holds the reduced coordinates, its rows are the Galerkin rows `V^T R`, and its field
chart is the owner's chart composed with `V`; traces, conormal reaction fluxes,
boundary impositions, pointwise reconstruction, and the field-space identity are the
full owner's, acting on the reconstructed field. A linear owner contributes the dense
projected operator `V^T A V`; a nonaffine owner is solved by Newton. Parameters keep
reaching the owner through its runtime arguments, so implicit parameter derivatives
of the ROM-coupled solve are the ordinary coupled derivatives.

The interface certificate of a ROM-coupled solution measures the ROM: a mortar's
weak continuity is solved exactly at every rank, and its flux balance compares the
full owner's reaction of the reconstructed field with the multiplier. That balance
is the interface residual of the reduced model; it decays with the rank and passes
acceptance only when the basis spans the region's parametric solution manifold.

The basis, provider, and owner topology are fixed prepared structure: their array
leaves resolve as `FIXED`, and a new basis is a new `ReducedComponent`, a new
`owner_id`, and a new prepared problem, never an online gradient. A trainable model
hidden inside the provider is refused because it would be frozen silently; bind
learned values through a `ParameterBinding` instead. A Petrov–Galerkin reduction and
an owner that publishes a kernel are refused. The model is full-order assisted
(every residual and operator action calls the original owner); see
`examples/coupled_rom_swap.py` and
[Numerical interoperability](guides_numerical_interoperability.md#reduced-order-components).

## Hyperreduction

The methods have different contracts:

- DEIM uses a nonlinear-term collateral basis and a provider that evaluates only
  selected nonlinear entries;
- GNAT uses a time-discrete residual basis and selected residual evaluations inside
  LSPG;
- ECSW selects element contributions and nonnegative empirical quadrature weights.

Slicing a fully assembled nonlinear vector is not hyperreduction. Sampled providers
bind support, geometry, and implementation identity. Conditioning and held-out term,
residual, and rollout defects remain separate evidence.

## Empirical interpolation and ROQ

`prepare_empirical_interpolation` operates on a role-explicit basis artifact and
supports real or complex bases. It records node order, interpolation conditioning,
and maximum source-basis reproduction error.

For gravitational-wave ROQ, fit a role=`roq` basis on the exact frequency support,
then prepare empirical interpolation. This path does not require a dynamical ROM.

## Persistence

ROM archives use the bounded pickle-free array archive and lifecycle `ModelManifest`.
Basis artifacts, prepared affine and evolution models, index-one descriptor
reductions, and residual dual-norm artifacts carry the canonical `phx.NumericRevision`
of their arrays, and the manifest's numeric revision is that revision ID. Restoring an
empirical-interpolation archive recomputes the canonical revision from the archived
arrays and rejects a mismatch.
Prepared affine-model restoration is template-bound so that vector-space,
coefficient-map, solver, unit, support, and build identities cannot be substituted.

Legacy ROM NPZ files are not automatically migrated because they lack the spaces,
measures, lifts, reduced law, and evidence needed for safe interpretation.

## Qualification

Use `tools/rom_qualification.py` for the affine thermal-block, polynomial
operator-inference, selected-evaluation, and moving-front projection scenarios. Use
`benchmarks/rom_affine.py` to separate offline
projection, reduced assembly/solve, full reconstruction, and matched full solve
costs.

## Production maturity and deployment

`rom_capability_catalog` is the canonical maturity inventory. Candidate,
experimental, and internal capabilities are not silently presented as released
production profiles. `ROMResourcePolicy`, `ROMAdmissionEvidence`, and
`ROMCostEstimate` make resource and support admission explicit before execution.

`ROMDeploymentBundle` is a portable dependency graph over separately archived,
registered artifacts. It binds capability profiles, build provenance, execution
requirements, resource policy, and qualification evidence. It contains no
executable Python callable or pickle payload.

## Physical POD and snapshot manifests

`SnapshotManifest` binds case partition, state/field layouts, support, measure,
geometry, topology, quadrature, truth revision, and chunk identities.
`PhysicalPODPlan` performs method-of-snapshots POD through an
`AbstractVectorSpace` pairing and reports rank, retained/tail energy,
orthogonality, and unmet-target status. Its eigenvalues of the snapshot Gram
matrix are accurate to about $(N + \sqrt{m})\,\varepsilon\,\lambda_1$ for $N$
snapshots of $m$ coordinates, so singular values below
$\sqrt{(N + \sqrt{m})\,\varepsilon}\,\sigma_1$ are roundoff and never enter the
basis; `retained_energy` (default `1.0`: every resolved direction) and an
absolute `minimum_singular_value` can only lower the rank further. Snapshots
whose trailing singular values matter below that floor need an SVD in
orthonormal coordinates.

Use exact array `POD` with the native `DenseSVD()` when the full orthonormal-coordinate
singular spectrum is affordable or trailing modes below the snapshot-Gram
resolution floor matter. For leading array modes of resident weighted/masked
snapshots, explicitly select `RandomizedSVD`, an approximation tolerance/resource
policy, and a typed fit key. This route is fixed-width QR range compression with
leading certification, not a new snapshot-Gram or streaming algorithm. Its actual
projection energy is original-action capture; a finite approximation or high
energy fraction alone is neither a leading-order certificate nor a ROM error
bound. General arbitrary pairings stay with admitted exact/snapshot routes;
randomized non-diagonal custom metrics are refused rather than silently densified.

## Transient, mixed, and descriptor systems

`AffineEvolutionROMProblem` projects mass, spatial, forcing, and lift terms.
Binding coefficient arrays produces a native `DifferentialAlgebraicSystem`; time
integration remains solver-owned. Lift-rate terms enter separately from spatial
lift terms.

`RectangularLinearROMProblem` uses the native least-squares runtime when the test
rank exceeds the trial rank. `ReducedInfSupEvidence` is reduced stability
evidence, not a full mixed-problem certificate. `IndexOneDescriptorReduction`
accepts only a regular impulse-free index-one structure and reconstructs its
algebraic variables through the declared Schur block.

## Greedy, SCM, and goal-oriented outputs

`EstimatorGreedyPlan` performs deterministic host-side enrichment using a supplied
truth snapshot and estimator. Only theorem-backed estimators support a certified
greedy claim. `SuccessiveConstraintArtifact` solves a native small linear program
over declared affine stability constraints. `PrimalDualOutputBound` keeps output
correction and absolute error bound separate from ordinary observation output.

## Geometry, atlases, and state charts

`ReferencePhysicalRepresentation` requires explicit physical-to-reference and
reference-to-physical `FieldTransfer` values and a qualified round trip.
`ReducedBasisAtlasArtifact` binds local supports and transition matrices and
refuses uncovered queries or inconsistent transition cycles.

`QuadraticStateChart` supplies exact decode, JVP, and VJP operations under a
single symmetric-monomial convention. `CoordinateConditionedStateChart` wraps a
fixed registered decoder queried on one declared reference support. Neither chart
implies global injectivity.

## Sensing and assimilation

`SensorConfiguration` binds coordinates, channels, units, frame, geometry,
cadence, and noise. `ObservationHistory` carries masks and resets.
`LinearSensorHistoryEstimator` returns a reduced-state Gaussian estimate only
after a complete warm-up window. `ReducedKalmanAssimilator` updates the reduced
state belief; it never mutates the reduced dynamics model.

## Spectral-submanifold models

`SpectralSubmanifoldModel` lives in `dynamics.identification`. It binds a local
polynomial chart and reduced flow to hyperbolic spectral evidence, an observation
contract, a partition, and a finite validity radius. Queries outside that radius
are refused. Invariance and conjugacy evidence remain separate from generic
trajectory regression.

## Structure-preserving and control reduction

`SymplecticReduction` verifies the reduced symplectic form before projecting a
Hamiltonian matrix. `PortHamiltonianReduction` verifies skew interconnection,
positive-semidefinite dissipation, and a positive energy metric before
constructing reduced matrices.

Stable standard LTI balancing and rational Krylov reduction are owned by
`phydrax.control`; ROM deployment may reference their artifacts without
duplicating their algorithms.

## Immutable enrichment and distributed execution

`ROMGeneration` and `EnrichmentTransaction` form an immutable parent/child
lifecycle. `ActiveLearningPlan` selects admissible candidates but does not run
truth or mutate a model. `DistributedBasisArtifact` stores one row partition and
uses a named collective for projection; no full-state all-gather is implied.
