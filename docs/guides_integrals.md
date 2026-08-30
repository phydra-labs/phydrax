# Integrals and measures

Phydrax integration is organized around three explicit objects:

1. a **target** defines the measure and normalization convention;
2. a **plan** defines how that target is discretized or sampled;
3. an **estimate** returns the value, status, diagnostics, provenance, and only a
   method-valid error estimate.

The same execution path supports fixed and adaptive quadrature, fixed-design
Bayesian quadrature, Monte Carlo, randomized quasi-Monte Carlo, importance
sampling, sparse grids, mapped cells, and mixed product plans.

For named analytic primitives such as Dawson's integral, the Faddeeva function,
and Voigt profiles, use [`phydrax.special`](guides_special_functions.md). Those
functions own fixed approximation and differentiation contracts; they do not run
through integration targets or quadrature plans.

## Targets define the mathematical quantity

A plan never decides whether an integral is normalized. The target does.

| Constructor | Quantity |
| --- | --- |
| `over(component)` | $\int f\,d\mu$ under the component's physical/counting measure |
| `mean_over(component)` | $\int f\,d\mu / \int 1\,d\mu$ |
| `expectation(probability)` | $\mathbb E_p[f]$ for a `ProbabilityDomain` |
| `density(base, log_density)` | $\int f\exp(\ell)\,d\nu$ relative to the base target $\nu$ |
| `normalized_density(base, log_density)` | $\int f\exp(\ell)\,d\nu / \int \exp(\ell)\,d\nu$ |
| `mapped(rule, mapping, jacobian)` | reference-cell quadrature mapped to physical coordinates |
| `weighted(samples, log_weights)` | reduction of externally supplied raw log weights |
| `discrete(points, weights, axes=...)` | deterministic reduction under supplied nonnegative quadrature weights |

For component targets, the induced measure follows the selected component:

- geometry `Interior()` uses physical volume, area, or length;
- geometry `Boundary()` uses boundary arclength or surface measure;
- scalar-interval `Interior()` uses interval length;
- scalar-interval `Boundary()` uses endpoint counting measure;
- fixed slices use unit-mass Dirac semantics;
- dataset components use the domain's declared probability or count measure.

`component.where`, `component.where_all`, and `component.weight_all` remain part of
the target measure. They are applied at quadrature or sample points before reduction.
A normalized target divides by the correspondingly filtered and weighted mass.

`density(base, ...)` preserves the base target's measure. In particular,
`density(mean_over(component), ...)` integrates against the normalized component
measure; only `normalized_density(...)` additionally normalizes by the supplied
density mass.

## Basic fixed quadrature

```python
import phydrax as phx
import jax.numpy as jnp

space = phx.domain.ScalarInterval(-1.0, 2.0, label="x")


@space.Function("x")
def square(x):
    return x**2


target = phx.integration.over(space.component())
plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(24))
estimate = phx.integration.integrate(square, target, plan)

value = estimate.value
assert estimate.successful
```

Fixed plans support interval rules such as `GaussLegendreRule`,
`GaussKronrodRule`, `ClenshawCurtisRule`, and `TanhSinhRule`. A deterministic fixed
rule reports no statistical uncertainty. `error_estimate is None` is deliberate.

A `ProbabilityDomain` used through `over(probability.component())` is lowered through
its quantile map and retains unit probability mass. This is measure-equivalent to
`expectation(probability)`; it is not a Lebesgue integral over the support endpoints.

## Measure-matched Gaussian rules

`GaussHermiteRule` integrates directly against a normalized standard-normal
reference measure. It avoids sending a bounded interval rule through a normal
quantile map, which is inefficient for tail-sensitive moments:

```python
normal = phx.domain.ProbabilityDomain(
    phx.uq.Normal(0.0, 1.0),
    label="z",
)
moment = phx.integration.integrate(
    normal.Function("z")(lambda z: z**20),
    phx.integration.expectation(normal),
    phx.integration.FixedQuadraturePlan(phx.integration.GaussHermiteRule(11)),
)
```

The rule requires a distribution with a standard-normal reference transform.
Built-in `Normal` and `LogNormal` provide one. Combine different reference
measures with `ProductIntegrationPlan`; one fixed rule is never silently
reinterpreted for an incompatible factor.

### Coupled standard-normal cubature

`GaussianCubatureRule(dimension, degree)` is the multidimensional counterpart
for a product of standard-normal reference measures. Its positive formulas are
certified for every monomial of total degree through three or five; this is a
total-degree contract, not a tensor-product order. Automatic selection uses the
`2d`-point Stroud--Secrest degree-three rule, the three-point Hermite rule in one
dimension at degree five, the nine-point Stroud--Secrest rule in two dimensions,
and the positive Stroud--Secrest axis/diagonal rule above two dimensions.
`maximum_points` and `maximum_rule_bytes` reject an infeasible formula before it
is materialized.

Use one grouped `ProductIntegrationPlan` entry so every transformed coordinate
shares the rule's single cubature axis:

```python
gaussian_x = phx.domain.ProbabilityDomain(
    phx.uq.Normal(1.0, 2.0), label="x"
)
gaussian_y = phx.domain.ProbabilityDomain(
    phx.uq.Normal(-2.0, 0.5), label="y"
)
gaussian_domain = phx.domain.ProductDomain(gaussian_x, gaussian_y)
gaussian_function = gaussian_domain.Function("x", "y")(
    lambda x, y: ((x - 1.0) / 2.0) ** 2 * ((y + 2.0) / 0.5) ** 2
)
gaussian_plan = phx.integration.ProductIntegrationPlan(
    {
        ("x", "y"): phx.integration.FixedQuadraturePlan(
            phx.integration.GaussianCubatureRule(2, 5)
        )
    }
)
gaussian_moment = phx.integration.integrate(
    gaussian_function,
    phx.integration.over(gaussian_domain.component()),
    gaussian_plan,
)
```

Every grouped factor must declare a standard-normal reference transform.
Built-in `Normal` and `LogNormal` factors qualify. A rule dimension mismatch,
an incompatible factor, or a family/degree combination outside the built-in
positive catalog is rejected rather than lowered as independent
one-dimensional quadrature.

## Curated multidimensional cubature

`CubatureRule(reference, degree)` returns a prepared positive rule with points
inside its canonical reference domain. Supported references are `"triangle"`,
`"tetrahedron"`, `"circle"`, `"disk"`, `"sphere"`, and `"ball"`. Rules expose
their family, certified reference polynomial degree, node count, source identity,
storage, and content-derived `rule_id`.

Triangle rules through degree 30 and tetrahedron rules through degree 15 use
Xiao--Gimbutas data. Higher requested simplex degrees use an explicit positive
Duffy--Gauss fallback by default; set `allow_duffy_fallback=False` to require a
tabulated rule. Circle, disk, and ball rules use positive periodic/radial product
constructions. Sphere rules use only positive-weight Lebedev formulas.

```python
mapping = lambda reference: reference
jacobian = lambda reference: jnp.ones((reference.shape[0],))
integrand = lambda point: point[:, 0] + point[:, 1]
triangle_rule = phx.integration.CubatureRule("triangle", 10)
triangle_target = phx.integration.mapped(
    triangle_rule,
    mapping,
    jacobian,
)
estimate = phx.integration.integrate(
    integrand,
    triangle_target,
    phx.integration.CellQuadraturePlan(triangle_rule),
)
```

The certified degree belongs to the canonical reference measure. Affine maps
preserve that polynomial contract; nonlinear Jacobians generally do not.
Polynomial degree is not an error bound, so fixed cubature deliberately reports
`error_estimate=None`.

Analytic circles and spheres expose native interior and boundary cubature maps,
and watertight `MeshRegion` boundaries expose direct triangle charts. Pass the
matching `CubatureRule` through `FixedQuadraturePlan` to avoid bounding-box masks
or tensor chart bias. Rigid transformations, translations, and uniform scaling
preserve native cubature. Nonuniform scaling currently falls back only when the
caller chooses another plan; it is never silently assigned an incorrect surface
Jacobian.

## Materialize once, reduce many times

Separate materialization from reduction when multiple integrands share a target and
plan:

```python
realization = phx.integration.materialize(target, plan)

integral_square = phx.integration.reduce(square, realization)
integral_one = phx.integration.reduce(1.0, realization)
```

`IntegrationRealization` carries the typed target, plan, batch, and execution key.
Reusing it guarantees that stochastic integrands see the same sample design. This is
the preferred common-random-number pattern for comparisons and parameter sweeps.

`integrate(integrand, target, plan)` is exactly the one-shot materialize-and-reduce
form.

An integrand may itself be any nonempty PyTree of `DomainFunction`, callable, or
array leaves. Reduction preserves that container structure and every leaf dtype in
`estimate.value`; method diagnostics are returned with the same leaf structure.

## Calibrate a reusable finite realization

Calibration corrects a finite measure to externally known normalized feature
expectations without discarding source points:

```python
samples = jnp.linspace(0.0, 1.0, 41)
source = phx.integration.materialize(
    phx.integration.weighted(
        samples,
        jnp.zeros((41,)),
        normalized=False,
        target_mass=jnp.asarray(7.0),
    )
)
calibrated = phx.integration.calibrate(
    source,
    phx.weighting.ExactMoments(jnp.array([0.7])),
    features=samples,
)

physical_first_moment = phx.integration.reduce(lambda x: x, calibrated)
```

The calibration target is always a normalized expectation: here it requests mean
`0.7`. The transformed measure retains physical mass `7`, so reducing `x` returns
`4.9`. `QuadraticMoments` provides soft reconciliation when targets are uncertain or
outside the finite moment hull.

Calibration accepts one-axis `PointIntegrationBatch` and `WeightedSampleBatch`
realizations. A feature callable, array, named `coordax.Field`, or supported feature
PyTree is canonicalized to `(source_points, moment_count)`. Sample values, named or
positional sample axis, explicit mask, zero-prior support, ancestry, support validity,
execution key, and source provenance remain attached. Strata, antithetic pairs,
replicates, component unions, and multi-axis measures are rejected until their joint
weight constraints can be preserved.

Each calibration or compression appends a `MeasureTransformationRecord`.
`TransformedIntegrationDiagnostics` retains that ordered history beside the
downstream reduction diagnostics. Because reweighting invalidates the original
quadrature or Monte Carlo error model, transformed reductions report
`error_estimate=None` rather than a misleading inherited bound.

## Compress a reusable finite realization

For several expensive integrands over the same finite positive measure, insert
compression between materialization and reduction:

```python
samples = jnp.linspace(0.0, 2.0, 25)
log_weights = jnp.zeros((25,))
source = phx.integration.materialize(phx.integration.weighted(samples, log_weights))
compressed = phx.integration.compress(
    source,
    phx.coresets.MomentRecombination(),
    features=jnp.stack((samples, samples**2), axis=1),
)

estimate_a = phx.integration.reduce(lambda x: jnp.exp(x), compressed)
estimate_b = phx.integration.reduce(lambda x: jnp.cos(x), compressed)
```

`MomentRecombination` returns nonnegative weights and preserves the supplied feature
moments, including mass through its internal constant feature. The result retains at
most one more active point than the supplied feature count. `KernelHerding(k)` instead
selects `k` equally weighted points that greedily reduce a blockwise kernel MMD.

Compression accepts the same finite one-axis measure substrate as calibration. It
rejects signed weights and any stratum, antithetic pair, replicate, or
component-union structure that cannot yet be preserved exactly. Source ancestry,
target mass, named sample axes, masks, execution key, and provenance survive the
reduction. Its `MeasureTransformationRecord` contains
`MeasureCompressionDiagnostics`; `TransformedIntegrationDiagnostics` keeps the
complete ordered transformation chain separate from downstream integration
diagnostics.

Compression discrepancy is not reported as `IntegrationEstimate.error_estimate`:
feature-moment residual and MMD are construction diagnostics, not general error bounds
for an arbitrary later integrand. Amortize compression only when its construction cost
is repaid across expensive evaluations.

## Adaptive one-dimensional quadrature

Use `AdaptiveQuadraturePlan` for a single active scalar interval. All other product
factors must be fixed.

```python
plan = phx.integration.AdaptiveQuadraturePlan(
    phx.integration.GaussKronrodRule(21),
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
    max_intervals=256,
    breakpoints=(0.25,),
    collect_partition=True,
)
estimate = phx.integration.integrate(square, target, plan)
```

The global stopping test is

$$
e \leq \varepsilon_{\mathrm{abs}}
       + \varepsilon_{\mathrm{rel}}|I|.
$$

The diagnostics expose the active interval count, bounds, local integral estimates,
embedded-rule errors, and partition contributions when requested. The global error is
the sum of active local embedded-rule errors. `throw=True` turns a failed status into
a JIT-safe runtime error; set `throw=False` only when the caller checks
`estimate.successful`.

Known discontinuities or singular locations belong in `breakpoints`; they seed
separate initial intervals. Adaptive execution is differentiable through the selected
refinement path, but refinement decisions are discrete.

## Adaptive triangle quadrature

`AdaptiveTrianglePlan` performs bounded four-way refinement over affine triangle
charts exposed by a geometry cubature atlas. The default positive degree-5 and
degree-10 rules form a paired error indicator:

```python
mesh_vertices = jnp.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
)
mesh_faces = jnp.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
mesh_domain = phx.domain.GeometryDomain(
    phx.geometry.MeshRegion(mesh_vertices, mesh_faces).compile()
)
mesh_boundary = mesh_domain.component({"x": phx.domain.Boundary()})
surface_field = mesh_domain.Function("x")(lambda x: jnp.sum(x * x))
plan = phx.integration.AdaptiveTrianglePlan(
    absolute_tolerance=1e-8,
    relative_tolerance=1e-8,
    max_cells=256,
    max_evaluations=20_000,
    collect_partition=True,
)
estimate = phx.integration.integrate(
    surface_field,
    phx.integration.over(mesh_boundary),
    plan,
)
```

The global stopping test matches adaptive interval quadrature. Because the two
triangle rules are not nested, the result reports
`error_kind="paired-reference-rule"`, not an embedded or statistical error.
`MAXIMUM_CELLS_REACHED`, `MAXIMUM_EVALUATIONS_REACHED`, and
`NONFINITE_INTEGRAND` remain distinct terminal statuses. Refinement is
differentiable through the chosen partition but its selection decisions are
discrete; fixed cubature remains preferable for repeatedly optimized objectives.

## Fixed-design Bayesian quadrature

Use Bayesian quadrature when a fixed set of evaluations should be interpreted
through a Gaussian-process prior rather than an IID sampling model. The initial
contract is intentionally closed: one normalized scalar Gaussian
`ProbabilityTarget`, one `SquaredExponentialKernel` optionally wrapped in
`ScaleKernel`, and one fixed `PointSampling` design.

```python
normal = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
target = phx.integration.expectation(normal, target_id="standard-normal")
kernel_mean = phx.integration.GaussianKernelMean(
    target,
    phx.kernels.SquaredExponentialKernel(length_scale=0.75),
)
plan = phx.integration.BayesianQuadraturePlan(
    kernel_mean,
    phx.domain.PointSampling(32, design="hammersley"),
    observation_noise=0.0,
    solve_regularization=1e-10,
)
realization = phx.integration.materialize(target, plan)
mean = phx.integration.reduce(
    normal.Function("z")(lambda z: jnp.exp(0.2 * z)),
    realization,
)
```

Materialization evaluates the analytic Gaussian kernel mean and kernel double
mean in the requested evaluation dtype, prepares the normalized dense-LU
`phydrax.linalg` solve, and retains the solve result. Reduction casts
`DomainFunction` points before invoking the integrand and applies the resulting
weights to scalar, array, `coordax.Field`, or PyTree outputs. Reusing the
realization shares exactly the same points, kernel system, and weights.

`observation_noise` is part of the GP observation model.
`solve_regularization` is a separate numerical diagonal shift; neither is
silently inferred from the other. With both set to zero, a singular design
returns `LINEAR_SOLVE_FAILED`. Non-finite outputs and posterior variance below
the dtype-aware roundoff envelope also fail explicitly. A tiny negative
variance within that envelope represents numerical zero; no positive variance
floor is introduced.

The estimate value is the posterior integral mean. Its error estimate has
`error_kind="bayesian-posterior-standard-deviation"`. **The posterior standard
deviation is not a deterministic or frequentist error bound.** It quantifies
uncertainty only under the declared GP prior, kernel hyperparameters,
observation noise, and fixed design. It must not be used as a stopping
certificate without an external calibration argument.

The kernel mean is bound to the target's stable `target_id`, probability label,
and Gaussian location/scale content; every part must match before reduction.
Only the preflighted `DenseLU` route is accepted. Its operator, residual, and
linear-accumulation dtypes must match integration accumulation; a lower
factorization dtype remains explicit and contributes to the posterior-variance
roundoff envelope. Unsupported density targets, non-Gaussian measures, kernel
sums/products, active acquisition, WSABI, and unnormalized evidence are outside
this capability rather than approximated by a fallback.

## Monte Carlo and variance reduction

```python
import jax.random as jr

plan = phx.integration.MonteCarloPlan(4096)
estimate = phx.integration.integrate(
    square,
    target,
    plan,
    key=jr.key(0),
)
```

In normal code, import `jax.random as jr` and pass `key=jr.key(0)`. Randomized plans
require an explicit key; deterministic plans reject `key=` because they do not
consume one.

Available direct designs are:

- `IIDDesign()` (the default);
- `LatinHypercubeDesign()` for scalar and box-like component targets;
- `AntitheticDesign(involution=...)` for an explicit measure-preserving pairing.

Antithetic diagnostics reduce pair means, not individual draws. A standard error is
reported only with at least two independent IID pairs. A single pair and antithetic
Latin-hypercube designs still return an estimate but deliberately report no
uncertainty. Antithetic plans require an even sample count.

A control variate states both the control and its known expectation:

```python
@space.Function("x")
def x_control(x):
    return x


control = phx.integration.ControlVariateEstimator(
    (x_control,),
    (0.5,),
    pilot_samples=128,
)
plan = phx.integration.MonteCarloPlan(
    4096,
    control_variate=control,
)
```

When coefficients are omitted, ordinary IID plans fit them on the first
`pilot_samples` and estimate on the disjoint remainder. Pilot observations are never
reused in the reported estimate or uncertainty. Alternatively, supply coefficients
explicitly. `same_sample_asymptotic=True` is an opt-in asymptotic mode and is named as
such because its finite-sample error estimate does not account for fitted-coefficient
uncertainty.

## Stratified Monte Carlo

`StratifiedMonteCarloPlan` requires an explicit positive-measure partition, such as a
`GeometryMeasurePartition` over boundary segments or interior simplices.

```python
partition = phx.geometry.GeometryMeasurePartition(
    (
        ((-1.0,), (0.5,)),
        ((0.5,), (2.0,)),
    ),
    (1.5, 1.5),
    kind="segment",
)


@space.Function("x")
def stratified_square(x):
    return x[0] ** 2


design = phx.integration.StratifiedDesign(
    partition,
    allocation="proportional",  # or "equal" / "explicit"
)
plan = phx.integration.StratifiedMonteCarloPlan(2048, design)
estimate = phx.integration.integrate(stratified_square, target, plan, key=jr.key(1))
```

The estimator is

$$
\widehat I = \sum_h \mu_h\,\overline f_h,
$$

and its variance estimate is

$$
\widehat{\mathrm{Var}}(\widehat I)
= \sum_h \mu_h^2 s_h^2/n_h.
$$

Allocations are deterministic for a fixed key and guarantee at least two samples per
stratum. `allocation="explicit"` accepts relative allocation weights whose length must
match the partition.

## Quasi-Monte Carlo

```python
plan = phx.integration.QuasiMonteCarloPlan(
    1024,
    sequence="sobol",
    scrambled=True,
    num_replicates=8,
)
estimate = phx.integration.integrate(square, target, plan, key=jr.key(2))
```

Sobol sample counts must be powers of two by default. Set
`allow_arbitrary_count=True` only when that loss of balance is intentional.
Randomized-QMC uncertainty is computed across independently scrambled replicates.
Unscrambled QMC is deterministic, permits one replicate, and reports no uncertainty.
The estimate's evaluation count includes every replicate.

Direct component integration and `ProductIntegrationPlan` use the same canonical
reference designs and exact target-measure transports as domain sampling. An axis
group such as `("x", "t")` consumes one joint design over the sum of both factors'
reference dimensions. Independent one-dimensional Sobol sequences are never
substituted for that joint net.

Integration remains responsible for target masses, masks, importance or Jacobian
weights, replicate diagnostics, and uncertainty estimates. Sharing the reference
transport does not turn weighted integration realizations into ordinary
`PointBatch` objects.

## Squaring randomized integral estimates

An unbiased integral estimate does not remain unbiased after a trainable squared
mismatch. If `I_hat = I + epsilon` and `E[epsilon] = 0`, then

$$
\mathbb E[(a + \widehat I)^2]
= (a + I)^2 + \operatorname{Var}(\widehat I).
$$

Use `MomentPenalty` for deterministic per-step plans, fixed realizations, or
caller-supplied realizations. It rejects resampled stochastic plans. Use
`RandomizedMomentPenalty` when a moment is resampled during optimization:

```text
term = phx.terms.RandomizedMomentPenalty(
    condition,
    phx.integration.per_step(
        phx.integration.over(condition.on),
        phx.integration.MonteCarloPlan(128),
    ),
    num_realizations=2,
    loss_mode="u_statistic",
)
```

`"u_statistic"` and `"independent_product"` estimate the intended squared
mean without the variance term, but an individual value may be negative.
They therefore require `FunctionalSolver.solve(..., keep_best=False)` and an
independent fixed realization for validation. `"plug_in"` is nonnegative but
retains the variance bias by explicit request.

Freezing a realization produced by a randomized plan is different: conditional
on those fixed sites it defines one deterministic finite objective. Validate
that objective against independent or refined sites to detect overfitting to
the realization.

## Importance sampling and weighted samples

```python
probability = phx.domain.ProbabilityDomain(
    phx.uq.Normal(0.0, 1.0),
    label="z",
)
proposal = phx.uq.Normal(0.0, 2.0)


@probability.Function("z")
def field(z):
    return z**2


plan = phx.integration.ImportanceSamplingPlan(
    4096,
    proposal,
    self_normalized=True,
)
estimate = phx.integration.integrate(
    field,
    phx.integration.expectation(probability),
    plan,
    key=jr.key(3),
)
```

Importance materialization retains raw target-to-proposal log ratios. Log weights are
normalized with a log-sum-exp calculation only during reduction. Diagnostics include
the estimated normalizer, normalizer error, entropy, coefficient of variation,
maximum normalized weight, effective sample size, finite-weight counts, and full
log-weighted moment diagnostics.

Self-normalization changes the estimand: it is a ratio estimator, not an unbiased
substitute for an ordinary importance estimate. Strict support handling is the only
supported policy. Built-in distributions combine support metadata with deterministic
target-quantile checks and report `PROPOSAL_SUPPORT_FAILURE`. Custom distributions
must provide truthful `support`, `contains`, and `icdf` semantics; black-box support
cannot be inferred exactly from finite samples.

For external weighted samples, use `weighted(...)`; this API accepts raw log
weights, explicit sample axes, masks, target mass, support validity, and
producer-owned stratum, pair, replicate, and ancestry IDs. Set
`independent=True` only when the sampled units genuinely have independent
provenance. Otherwise Phydrax returns the estimate and weight diagnostics but
deliberately leaves `error_estimate=None`. Effective sample size diagnoses
weight degeneracy; it never certifies independence. Use `from_samples(...)`
separately to attach authoritative component-measure weights to an existing
structured point batch.

## External discrete and weighted measures

`discrete(...)` and `weighted(...)` are already-materialized targets. They take
no integration plan and consume no random key:

```python
import coordax as cx
import jax.numpy as jnp

nodes = cx.Field(jnp.asarray([0.0, 0.5, 1.0]), dims=("node",))
weights = cx.Field(jnp.asarray([0.25, 0.5, 0.25]), dims=("node",))
target = phx.integration.discrete(nodes, weights, axes="node")
estimate = phx.integration.integrate(lambda x: x**2, target)

samples = cx.Field(
    jnp.arange(2 * 4, dtype=float).reshape((2, 4)),
    dims=("case", "particle"),
)
log_weights = cx.Field(jnp.zeros((2, 4)), dims=("case", "particle"))
empirical = phx.integration.weighted(
    samples,
    log_weights,
    sample_axes="particle",
    independent=False,
)
means = phx.integration.integrate(lambda values: values, empirical)
```

Named weight fields make reduced and retained axes explicit. A target may
reduce several sample axes at once; every other weight axis is retained.
Masks are evaluated per retained slice. An empty slice reports
`NO_VALID_SAMPLES`, an included invalid log weight reports `INVALID_WEIGHTS`,
and an included nonfinite value reports `NONFINITE_INTEGRAND`. Masked or
zero-weight nonfinite values do not poison a valid reduction.

`normalized=True` computes a weighted mean. With `normalized=False`, supplied
`target_mass` scales the normalized mean to a known physical mass; without
`target_mass`, raw log weights define the ordinary sample-mean estimator.
Self-normalization, known-mass scaling, and raw-weight estimation are distinct
estimands.

## Trajectory, filtering, time, and spatial measures

Producer adapters expose existing stochastic and spatial objects without
resampling them:

```python
trajectory = phx.stochastic.StochasticTrajectory(
    jnp.asarray([0.0, 0.4, 1.0]),
    jnp.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4)),
    realization_axes=("path",),
    realization_shape=(2,),
    state_axes=("space",),
    realizations=(None,),
)
marginal = phx.stochastic.trajectory_measure(trajectory, mode="marginal")
path = phx.stochastic.trajectory_measure(trajectory, mode="path")
time = phx.stochastic.time_measure(trajectory, rule="trapezoid")

marginal_mean = phx.integration.integrate(lambda states: states, marginal)
time_integrals = phx.integration.integrate(path.samples, time)
path_expectation = phx.integration.integrate(time_integrals.value, path)

space_axis = phx.discretization.FourierAxisSpec(4).materialize(0.0, 1.0)
spatial_discretization = phx.discretization.TensorSpectralDiscretization.from_axes(
    (space_axis,)
)
space = phx.integration.spatial_measure(
    spatial_discretization,
    spatial_dims="space",
)
spatial_integrals = phx.integration.integrate(path.samples, space)
```

For a `ParticleFilterResult` named `filter_result`,
`phx.uq.particle_posterior_measure(filter_result)` returns a weighted target;
integrating a particle observable against it retains case and filtering-time
axes.

Marginal trajectory measures retain time and mask failed states independently.
Path measures exclude an entire path after any failed saved state. IID standard
errors are enabled only when trajectory realization metadata declares distinct
independent path units. Missing, antithetic, coupled, or shared-noise metadata
suppresses the standard error.

`time_measure(...)` uses each path's saved schedule and supports left-point and
trapezoid rules on strictly increasing irregular times. Validity masks must be
contiguous prefixes; fewer than two active nodes produce
`NO_VALID_SAMPLES`. `spatial_measure(...)` reuses deterministic physical
quadrature, including separable tensor-grid weights, and reports fixed
diagnostics. Particle posterior measures retain physical case and filtering
time axes, preserve ancestry, mask failed steps and particles, and always
suppress IID uncertainty because filtering particles are dependent.

Reductions compose without a fused product abstraction. For a stochastic
field, reduce space first, then sampled time, then complete paths. Each stage
retains every axis needed by the next stage and contains failures through its
mask and status.

## Sparse grids and product plans

A `SparseGridPlan` couples scalar axes through a weighted total-degree Smolyak
index set. For dimension `d`, level `L`, and positive anisotropy `a`, Phydrax uses

`{alpha in N_0^d : sum(a[j] * alpha[j]) <= L - 1}`.

The bounded default is a nested Clenshaw--Curtis sequence with one midpoint at
axis level zero and `2**level + 1` nodes afterward:

```python
plan = phx.integration.SparseGridPlan(
    3,
    5,
    anisotropy=(1.0, 1.5, 3.0),
)
```

Anisotropy accepts finite positive real values. Smaller values refine an axis
more aggressively. Node identity is structural, so reuse between nested levels
does not depend on rounded floating-point coordinates.

Standard-normal reference measures use normalized Gauss--Hermite rules:

```python
normal = phx.domain.ProbabilityDomain(
    phx.uq.Normal(2.0, 0.5),
    label="z",
)
plan = phx.integration.SparseGridPlan(
    1,
    5,
    axis_rules="gauss-hermite",
)
mean_square = phx.integration.integrate(
    normal.Function("z")(lambda z: z**2),
    phx.integration.over(normal.component()),
    plan,
)
```

For mixed products, pass one rule per coupled axis. `"clenshaw-curtis"` supports
bounded scalar factors and bounded uniform-reference probability factors.
`"gauss-hermite"` requires a `ProbabilityDomain` whose distribution declares a
standard-normal reference transform. Built-in `Normal` and `LogNormal`
distributions provide that transform.

The deterministic error indicator is the difference from the immediately
coarser level. It is reported as
`error_kind="sparse-grid-level-difference"` and is not a statistical standard
error. Diagnostics separately report current and previous node counts, the
number of nonzero tensor terms, and the resolved axis rules.

Sparse grids support interior scalar/probability factors and fixed slices;
boundary selectors require a boundary-capable fixed plan.

!!! warning "Node-count migration"
    Multivariate sparse grids now use the conventional one-point
    Clenshaw--Curtis base rule. Existing constructors remain valid, but node
    locations, evaluation counts, and output ordering differ from releases that
    started every axis with both endpoints. Output ordering was never a public
    contract.

Use `ProductIntegrationPlan` when factor groups need different methods:

```python
plan = phx.integration.ProductIntegrationPlan(
    {
        ("x", "y"): phx.integration.SparseGridPlan(2, 4),
        "t": phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    }
)
```

Axis groups must be disjoint and cover every non-fixed label exactly once. Fixed
labels need no plan. Deterministic product factors require interior selectors;
Monte Carlo and QMC factors also accept selectors with an exact reference
transport, including interval boundaries.

Separate stochastic groups form a tensor-product estimator. Phydrax reports no
single-realization standard error for that dependent Cartesian sample; group the
labels under one joint stochastic plan when a paired joint design is intended.
Replicated randomized-QMC tensor products report uncertainty across independent
full-product replicates. Mixed deterministic/stochastic products preserve each
stochastic axis until deterministic factors have been integrated out.

Product plans honor `target.axes`: only selected factor axes and their geometry
corrections are reduced. Control variates are supported by direct Monte Carlo and QMC
plans, but are rejected inside product factors rather than silently ignored.

## Mapped reference cells and CAD boundaries

`mapped(reference_rule, mapping, jacobian)` handles arbitrary supplied
reference-to-physical maps. `CellQuadraturePlan` selects the reference rule used for
the mapped realization.

Supplying `target_mass=` makes that mass authoritative: mapped weights are rescaled
to it, and diagnostics report the same physical mass.

CAD geometries expose chart atlases internally. Boundary lowering maps reference
nodes through every segment or face chart, multiplies by chart Jacobians, and honors
trim masks. Measure-zero chart seams are not sampled twice. For analytic geometry,
physical boundary weights and normals are derived from the geometry's measure
partition.

## Estimate contract and failures

Every `IntegrationEstimate` contains:

- `value`: a `coordax.Field`, or an integrand-matching PyTree of fields, preserving
  non-integrated output axes and dtypes;
- `status` and `successful`;
- `num_evaluations`;
- `error_estimate` and `error_kind`, when the method justifies them;
- typed method-specific `diagnostics`;
- `provenance` identifying the method, target kind, and realization.

Inside compiled code, branch with JAX control flow on the array-valued `status` or
`successful` fields. Convert a concrete status to a message only outside compiled
execution:

```python
ok = estimate.successful
message = phx.integration.status_message(estimate.status)
```

`IntegrationStatus` distinguishes non-finite integrands, invalid normalization mass,
invalid bounds or weights, exhausted adaptive budgets, proposal-support failures, and
unsampled strata. A deterministic rule never fabricates a variance estimate from its
node values.

## Integral functionals

`phx.terms.IntegralFunctional` accepts an integration target and an explicit
realization source. Use `fixed(realization)` to reuse one materialization,
`per_step(target, plan)` to materialize from each evaluation key, or
`caller(target)` to require a caller-supplied realization. Raw signed integral
functionals are added directly to `FunctionalSolver`; residual penalties remain
nonnegative.

The local, nonlocal, spatial, and time-convolution operators under
`phx.operators` remain field-transform operators. They are separate from the global
measure-aware integration API described here.
