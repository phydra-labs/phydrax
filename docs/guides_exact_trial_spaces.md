# Exact PDE trial spaces

Trefftz trial spaces parameterize fields that satisfy a homogeneous PDE before any
boundary fitting. They complement PINNs and exact boundary enforcement:

- a PINN learns an unconstrained field and penalizes its interior PDE residual;
- an exact trial field omits the interior residual and fits boundary conditions;
- generic hard boundary enforcement changes the field and need not preserve its PDE
  solution space, so Phydrax rejects that composition.

The stable substrate covers finite real trial spaces for the Euclidean Laplace,
polyharmonic, homogeneous Helmholtz, and flat constant-metric Dirac equations in
dimension two or greater.
Every public basis carries a construction certificate. A certificate proves equation
satisfaction for the represented finite span; it does not claim that a fixed finite
span is complete or that a boundary-value problem is unique.

## Harmonic polynomial basis

`HarmonicPolynomialBasis(n, p)` spans harmonic polynomials in `n` coordinates with
total degree at most `p`. For each homogeneous degree, Phydrax constructs the nullspace
of the polynomial Laplacian through deterministic lexicographic pivot/free-column
elimination in exact rational arithmetic. SVD is used only to report conditioning; it
does not orient the basis or enter its content fingerprint.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

basis = phx.equations.HarmonicPolynomialBasis(
    4,
    3,
    normalization=phx.equations.SimilarityNormalization(
        jnp.zeros((4,)),
        2.0,
    ),
)
model = phx.equations.LinearTrefftzField(
    basis,
    initial_scale=0.01,
    key=jr.key(0),
)
```

`SimilarityNormalization` permits translation and one positive scalar dilation only.
Anisotropic scaling does not generally preserve the Euclidean Laplacian and is not an
accepted input.

## Polyharmonic Almansi basis

For positive order `m`, `PolyharmonicAlmansiBasis` builds features of the form

```text
|x-c|^(2j) h_j(x),  j = 0, ..., m-1,
```

where each `h_j` belongs to a harmonic polynomial basis. Every represented field
satisfies the `m`-fold Laplace equation. Density of the infinite Almansi family requires
an appropriate star-shaped domain about the declared center; that completeness
assumption is recorded separately from algebraic PDE satisfaction.

```python
basis = phx.equations.PolyharmonicAlmansiBasis(
    5,
    2,
    (3, 2),
)
```

## Helmholtz plane waves

`HelmholtzPlaneWaveBasis` uses real sine/cosine features with a positive constant
wavenumber and fixed unit propagation directions. The constructor validates rather than
silently rescaling materially invalid directions. Antipodal directions define the same
real feature pair and are rejected as duplicates.

```python
directions = phx.equations.sample_unit_directions(
    16,
    8,
    key=jr.key(1),
)
basis = phx.equations.HelmholtzPlaneWaveBasis(
    8,
    3.0,
    directions,
)
```

The basis alone does not impose an exterior radiation condition or guarantee uniqueness
at an interior resonance.

## Holomorphic composition and coefficient linearity

Two-dimensional complex-potential wrappers derive PDE exactness from a separate
`HolomorphicMapCertificate`. Dense and low-rank complex-affine HMLP layers preserve
that map-level exactness. Concatenating independent holomorphic branches also
preserves it.

Coefficient linearity is stricter. A branch bundle is a finite linear subspace only
when every child is linear in its coefficients. Multiplying two trainable
holomorphic factors creates a finite parametric family that is nonlinear in the
combined parameters, even when each child is individually polynomial-linear.
Consequently, `HolomorphicProductPotential` is not eligible for the direct linear
trial-space solver when it has multiple trainable factors.

Generic real-coordinate separation and arbitrary post-activations cannot attach this
certificate.

## Monogenic Clifford polynomials

`MonogenicPolynomialBasis` constructs a deterministic exact-rational kernel of the
left Dirac map over full Clifford-valued polynomials. The associated
`LinearMonogenicField` has real trainable coefficients and multivector-valued basis
features.

```python
algebra = phx.metrix.clifford.CliffordAlgebraSpec((1, 1, 1))
basis = phx.equations.MonogenicPolynomialBasis(algebra, 2)
model = phx.equations.LinearMonogenicField(basis)
```

The certificate family is `"dirac"`. It records the algebra identity, signature,
normalization, degree, basis rank, and left-action convention. For a nondegenerate
constant diagonal metric, the squared Dirac operator is the corresponding flat signed
Laplacian, so every component is metric-harmonic. Degenerate metrics are rejected
because they do not admit the reciprocal frame required by this operator.

Generic products and nonlinearities do not preserve monogenicity and therefore drop
the exact trial certificate.

## Binding and boundary fitting

A `LinearTrefftzField` is an ordinary Phydrax array model. Bind it with `Domain.Model`,
declare normal conditions, and use normal integration sources and residual penalties.
There is no interior PDE training term.

```python
dimension = 4
domain = phx.domain.HyperRectangle(
    (-1.0,) * dimension,
    (1.0,) * dimension,
)
u = domain.Model("x")(
    phx.equations.LinearTrefftzField(
        phx.equations.HarmonicPolynomialBasis(dimension, 2)
    )
)
target = domain.Function("x")(lambda x: 1.0 + x[0] - 0.5 * x[1])
boundary = domain.component({"x": phx.domain.Boundary()})
condition = phx.conditions.Dirichlet("u", boundary, target=target)
source = phx.integration.per_step(
    phx.integration.mean_over(boundary),
    phx.domain.PointSampling(512),
)
term = phx.terms.ResidualPenalty(condition, source)
solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(term,),
    enforcement=None,
)
```

Use ordinary gradient or nonlinear least-squares optimization for resampled objectives.
For a fixed realization and affine residual, use `solve_linear_trial_space` to assemble
and solve the real coefficient least-squares problem directly:

```text
fixed = phx.integration.fixed(
    phx.integration.materialize(
        phx.integration.mean_over(boundary),
        phx.domain.PointSampling(512),
        key=jr.key(2),
    )
)
term = phx.terms.ResidualPenalty(condition, fixed)
solver = phx.solver.FunctionalSolver(functions={"u": u}, terms=(term,))
result = phx.solver.solve_linear_trial_space(solver, key=jr.key(3))
solver = result.solver
```

The direct solver accepts only directly bound `LinearTrefftzField` values, fixed
integration realizations, and quadratic `ResidualPenalty` terms. It audits affinity at
multiple coefficient vectors before invoking `phydrax.linalg` and reports rank,
condition, nullity, residual, and solver provenance through its nested linear result.

## Certificates, algebra, and audits

`Domain.Model` attaches the model's `TrialSpaceCertificate` to the bound field. Generic
`DomainFunction` algebra drops that reserved certificate conservatively: Phydrax does
not infer whether arbitrary postprocessing preserves a specific PDE. Promotion to a
larger compatible domain preserves it.

Inspect and audit explicitly:

```text
certificate = phx.equations.trial_space_certificate(solver["u"])
batch = domain.component().sample(
    phx.domain.PointSampling(128),
    key=jr.key(4),
)
audit = phx.equations.audit_trial_space(solver["u"], batch)
```

`audit_trial_space` evaluates the known differential operator on the supplied batch and
reports maximum and RMS residuals. It does not sample hidden points or alter the model.

Certificates restricted to `off-singular-support` additionally require matching target
admissibility evidence. The audit validates support identity and the exact target batch
before differentiating; unrestricted all-space certificates reject irrelevant
admissibility reports.

## Hard resource bounds

Polynomial ranks grow combinatorially with dimension and degree. Each constructor
performs count and byte preflight through `TrefftzResourceBudget`; exceeding a limit is
an error, never an implicit truncation.

## Unsupported combinations

The initial contract intentionally excludes:

- one-dimensional problems;
- variable-coefficient or nonlinear equations;
- arbitrary source terms without a certified particular solution;
- generic hard boundary enforcement;
- trainable basis geometry or propagation directions;
- native complex trainable coefficients;
- exterior Helmholtz radiation;
- singular boundary integral kernels;
- claims of completeness for one finite basis.

Use `python -m tools.trefftz_benchmarks` for deterministic nD harmonic boundary-fit
records containing certificate identity, rank, error, residual, timing, and environment
evidence.
