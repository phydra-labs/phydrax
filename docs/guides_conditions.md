# Conditions, integration, terms, and enforcement

Phydrax separates a scientific requirement from the numerical choices used to
optimize it:

1. **Conditions** state field semantics on a domain component.
2. **Integration sources** own the target measure, integration plan, and lifetime
   of the numerical realization.
3. **Terms** reduce a condition to one real scalar penalty or contribute a signed
   scalar objective.
4. **Enforcement** compiles selected conditions into exact field transforms.

`FunctionalSolver` receives one ordered `terms` collection and one optional
`enforcement` input. This keeps scientific meaning independent of sampling,
quadrature, weighting, or hard-versus-soft treatment.

## Conditions state semantics

The generic condition constructors are:

- `phx.conditions.Residual(fields, on, operator)` for a pointwise residual;
- `phx.conditions.Moment(fields, on, operator, target=...)` for an integrated
  equality;
- `phx.conditions.Observation(fields, on, target, operator=...)` for observed
  field data.

Boundary and initial semantics use `Dirichlet`, `Initial`, `Neumann`, `Robin`,
and `Absorbing`. Their `on` attribute is the support on which the requirement
holds; they do not choose points or quadrature.

```python
import phydrax as phx

space = phx.domain.Interval1d(-1.0, 1.0)
interior = space.component()
boundary = space.component({"x": phx.domain.Boundary()})

pde = phx.conditions.Residual(
    "u",
    interior,
    lambda u: phx.operators.laplacian(u, var="x") - 2.0,
    label="poisson",
)
boundary_value = phx.conditions.Dirichlet(
    "u", boundary, target=lambda x: x[0] ** 2, label="boundary"
)
mass = phx.conditions.Moment("u", interior, lambda u: u, target=2.0 / 3.0, label="mass")
```

Physical condition families are grouped by field of use:

- `phx.conditions.cfd`: no-penetration, slip-wall, symmetry, and zero-normal-
  gradient velocity conditions;
- `phx.conditions.thermal`: heat-flux and convection conditions;
- `phx.conditions.solids`: traction, displacement, foundation, and elastic
  symmetry conditions;
- `phx.conditions.electromagnetics`: PEC/PMC, impedance, surface-source, and
  interface conditions;
- `phx.conditions.stochastic`: Kolmogorov, Fokker--Planck, and probability-flux
  conditions;
- `phx.conditions.conservation`: flow, pressure, reaction, charge, magnetic-flux,
  and Poynting-flux moments.

Every residual condition uses `ResidualPenalty`; every integrated conservation
condition uses `MomentPenalty`. The physical catalog does not introduce a
second numerical term hierarchy.

## Integration sources own numerical realization

A term always receives an explicit source. First choose its target measure:

- `phx.integration.mean_over(condition.on)` is a normalized physical or
  counting-measure mean and is the usual residual-loss target;
- `phx.integration.over(condition.on)` is an unnormalized integral and is the
  usual moment target.

Then choose who owns materialization:

```python
import jax.random as jr

residual_target = phx.integration.mean_over(pde.on)
plan = phx.domain.PointSampling(256)

# Draw a new realization for each evaluation.
per_step_source = phx.integration.per_step(residual_target, plan)

# Reuse one explicitly materialized realization.
realization = phx.integration.materialize(residual_target, plan, key=jr.key(0))
fixed_source = phx.integration.fixed(realization)

# Require the evaluation caller to provide a compatible realization.
caller_source = phx.integration.caller(residual_target)

# Let the solver maintain and refresh an adaptive realization.
policy = phx.sampling.collocation.R3(refresh_every=50)
adaptive_source = phx.integration.adaptive(residual_target, plan, policy)
adaptive_penalty = phx.terms.ResidualPenalty(pde, adaptive_source)
```

`per_step(target, plan)` is appropriate for training-time resampling.
`fixed(realization)` makes repeatable diagnostics explicit. A term using
`caller(target)` is evaluated with `realization=...`, which is useful when a
batch is shared across several diagnostics. Adaptive collocation runs through a
normal `ResidualPenalty` whose source is `AdaptiveIntegration`; the source and
policy, not a separate residual type, own the changing point population.

Residual attention is another source-owned policy:

```python
attention = phx.sampling.collocation.ResidualAttentionCollocation(
    refresh_every=5,
    decay=0.99,
    minimum_ess_fraction=0.35,
)
attention_source = phx.integration.adaptive(residual_target, plan, attention)
```

It retains the original points and updates mass-preserving local multipliers from
detached residual scores. The resulting objective is a changing training measure,
not an unbiased estimate of the original mean. Keep a fixed unweighted
`evaluation_term` for model selection and reporting.

## Terms produce scalars

`ResidualPenalty` integrates the Hermitian squared Frobenius norm

$$
\ell_r = s\int \lVert r(z)\rVert_F^2\,d\mu(z),
$$

so real and complex residuals both produce a real nonnegative scalar. `scale`
is a global nonnegative multiplier. An optional nonnegative `density`
`DomainFunction` weights the pointwise score inside the reduction.

`MomentPenalty` integrates the condition's field first and then squares its
mismatch with the declared target. `ObservationPenalty` applies residual-penalty
semantics to an `Observation`.

```python
pde_penalty = phx.terms.ResidualPenalty(
    pde,
    per_step_source,
    scale=1.0,
)

boundary_penalty = phx.terms.ResidualPenalty(
    boundary_value,
    phx.integration.per_step(
        phx.integration.mean_over(boundary_value.on),
        phx.integration.MonteCarloPlan(64),
    ),
    scale=10.0,
)

mass_penalty = phx.terms.RandomizedMomentPenalty(
    mass,
    phx.integration.per_step(
        phx.integration.over(mass.on),
        phx.integration.MonteCarloPlan(256),
    ),
    num_realizations=2,
    loss_mode="u_statistic",
)
```

For observed data, construct a target `DomainFunction`, declare the observation,
and choose the source independently:

```python
target = space.Function("x")(lambda x: x[0] ** 2)
observation = phx.conditions.Observation("u", interior, target)
data_penalty = phx.terms.ObservationPenalty(
    observation,
    fixed_source,
    scale=5.0,
)
```

Signed objectives implement the same scalar-term contract and belong in the
same solver collection. Use a signed integral objective only when the
mathematics supplies a variational objective, such as total potential energy;
do not substitute one for a squared residual merely because both are scalar.
See [Integrals and measures](guides_integrals.md) for target and plan details.

## End-to-end soft conditions

```python
import jax.random as jr
import optax

model = phx.nn.models.MLP(
    in_size="scalar",
    out_size="scalar",
    width_size=32,
    depth=2,
    key=jr.key(1),
)
u = space.Model("x")(model)

soft_solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(pde_penalty, boundary_penalty),
    enforcement=None,
)
soft_solver = soft_solver.solve(
    num_iter=20,
    optim=optax.adam(1e-3),
    seed=0,
    log_every=0,
)
```

The conditions contain the PDE and boundary meaning; the sources determine the
measures and Monte Carlo realizations; the penalties determine scalarization and
relative scale.

## Exact enforcement

A condition can instead be compiled into an exact transform. Compile
`EnforcementSpec` values once with `phx.enforcement.compile(...)`, then pass the
resulting `EnforcementProgram` to the solver.

```python
hard_boundary = phx.enforcement.EnforcementSpec(
    boundary_value,
    options={"var": "x"},
)
functions = {"u": u}
program = phx.enforcement.compile(functions, (hard_boundary,))

hard_solver = phx.solver.FunctionalSolver(
    functions=functions,
    terms=(pde_penalty,),
    enforcement=program,
)

# Terms always see the transformed field.
u_exact_on_boundary = hard_solver.ansatz_functions()["u"]
```

The hard solver omits the boundary penalty because the compiled ansatz satisfies
that condition by construction. Multiple specifications are validated, ordered
by boundary/initial/interior stage, and compiled once. Initial-value conditions
use `phx.conditions.Initial`; Neumann and Robin conditions can be compiled in the
same way when their derivative requirements are supported.

For the enforcement compiler and its options, see
[Exact enforcement](api/solver/enforcement.md). For solver evaluation
and optimization, see [Functional solver](api/solver/functional_solver.md).

## Joint linear conditions without pivots

Typed finite conditions can couple several fields and are projected jointly:

```python
import jax.numpy as jnp

value = phx.conditions.ArrayCodomain.from_shape((2,), dtype=float)
fields = phx.conditions.ProductFieldSpec(
    (
        phx.conditions.FieldSpec("u", value),
        phx.conditions.FieldSpec("v", value),
    )
)
condition = phx.conditions.Condition(
    "coupled-sum",
    fields,
    phx.conditions.MatrixLinearFunctional(
        ("u", "v"),
        ((2,), (2,)),
        (jnp.eye(2), jnp.eye(2)),
    ),
    value,
    phx.conditions.Equality(jnp.array([2.0, 4.0])),
)
raw = {"u": jnp.array([5.0, -1.0]), "v": jnp.zeros(2)}
bound = phx.conditions.bind_condition(condition, raw)
prepared = phx.enforcement.prepare_affine_projector(
    (bound,),
    phx.enforcement.ConstraintLinearCorrectionProvider(),
    correction_fields=("u", "v"),
)
projected = prepared.apply(raw)
```

The prepared operator contains the full off-diagonal response. Cyclic relations
therefore require neither a pivot nor sequential projections. Use
`CoefficientElimination` when the fields expose a certified finite linear
representation, a geometry/kernel correction provider for function fields, or a
nonlinear/feasibility realization for non-affine relations.
