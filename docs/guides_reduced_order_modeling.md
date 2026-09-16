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
