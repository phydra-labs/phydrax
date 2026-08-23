# Integral and nonlocal physics

Integral physics is assembled from ordinary Phydrax fields, operators, conditions,
integration sources, and terms. There is no separate integral-PINN model class.

This page covers three distinct numerical contracts:

1. a deterministic integral inside a pointwise residual;
2. a randomized estimate of an integrated moment;
3. an auxiliary field whose differential equation and trace uniquely define the
   represented integral.

## Deterministic integral form of an evolution equation

For

$$
\partial_t u = F[u], \qquad u(x,t_0)=u_0(x),
$$

the causal integral residual is

$$
r_I(x,t)=u(x,t)-u_0(x)-\int_{t_0}^{t}F[u](x,s)\,ds.
$$

The initial field is part of the identity. Omitting it silently changes every
problem with nonzero initial data.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

space_time = (
    phx.domain.Interval1d(0.0, 1.0)
    @ phx.domain.TimeInterval(0.0, 1.0)
)
interior = space_time.component()
initial_slice = space_time.component({"t": phx.domain.FixedStart()})

model = phx.nn.models.MLP(
    in_size=2,
    out_size="scalar",
    width_size=32,
    depth=3,
    key=jr.key(0),
)
u = space_time.Model("x", "t")(model)
u0 = space_time.Function("x")(lambda x: jnp.sin(jnp.pi * x[0]))
rule = phx.integration.GaussLegendreRule(48)


def integral_heat_residual(field):
    rhs = phx.operators.laplacian(field, var="x")
    history = phx.operators.time_convolution(
        lambda lag: jnp.ones_like(lag),
        rhs,
        rule=rule,
    )
    return field - u0 - history


physics = phx.conditions.Residual(
    "u",
    interior,
    integral_heat_residual,
    label="integral-heat",
)
physics_term = phx.terms.ResidualPenalty(
    physics,
    phx.integration.per_step(
        phx.integration.mean_over(physics.on),
        phx.integration.MonteCarloPlan(512),
    ),
)
initial = phx.conditions.Initial("u", initial_slice, target=u0)
enforcement = phx.enforcement.compile(
    {"u": u},
    [phx.enforcement.EnforcementSpec(initial)],
)
solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(physics_term,),
    enforcement=enforcement,
)
```

`time_convolution` is deterministic. Its fixed interval rule is mapped onto each
`[t_0,t]`, returns exact zero at `t=t_0`, and records the rule in field metadata.
Randomized inner integral estimates must retain independent realizations and use an
estimator-aware term; they are not hidden behind this field operator.

### Strong and integral residuals are not equivalent norms

When the initial condition is exact,

$$
r_I(t)=\int_{t_0}^{t}r_S(s)\,ds,
\qquad
r_S=\partial_tu-F[u].
$$

Integration smooths and can cancel oscillatory temporal defects. Validate an
integral-form model with an independent strong residual, short-window balances, or
both. A one-sided history residual constrains causal consistency; the global
coordinate network still does not propagate state sequentially.

## Squaring a randomized integral estimate

If an unbiased integral estimate is

$$
\widehat I_\theta=I_\theta+\epsilon_\theta,
\qquad \mathbb E\epsilon_\theta=0,
$$

then

$$
\mathbb E\left[(a_\theta+\widehat I_\theta)^2\right]
=(a_\theta+I_\theta)^2+\operatorname{Var}(\widehat I_\theta).
$$

The variance usually depends on the trainable field. A sampled integral followed by
ordinary MSE therefore optimizes a different objective.

Use `RandomizedMomentPenalty` for a resampled moment:

```python
mass = phx.conditions.Moment(
    "u",
    interior,
    lambda field: field,
    target=1.0,
    label="unit-mass",
)
source = phx.integration.per_step(
    phx.integration.over(mass.on),
    phx.integration.MonteCarloPlan(128),
)
mass_term = phx.terms.RandomizedMomentPenalty(
    mass,
    source,
    num_realizations=2,
    loss_mode="u_statistic",
)
```

The modes are:

| Mode | Expected objective | Individual value |
| --- | --- | --- |
| `u_statistic` | unbiased squared mean | may be negative |
| `independent_product` | unbiased squared mean | may be negative |
| `plug_in` | includes estimator variance | nonnegative |

Signed unbiased terms require:

```python
solver.solve(..., keep_best=False)
```

Selecting the most negative sampled update selects estimator noise. Use a fixed,
independent realization for validation and model comparison.

Ordinary `MomentPenalty` accepts deterministic per-step integration, fixed
realizations, and caller-supplied realizations. It rejects resampled stochastic
integration rather than silently selecting plug-in MSE.

## Exact auxiliary integral fields

Auxiliary outputs are physical only when complete differential and trace conditions
uniquely define them. For the exponential memory

$$
z(t)=\int_{t_0}^{t}e^{-\lambda(t-s)}u(s)\,ds,
$$

the exact local realization is

$$
\partial_tz=u-\lambda z,
\qquad z(t_0)=0.
$$

Represent `u` and `z` as ordinary named `DomainFunction`s, enforce the zero trace of
`z`, and use a residual for the differential relation. A deterministic
`time_convolution` evaluated on an independent fixed batch can audit `z` directly.

Do not introduce free auxiliary outputs `D` and `v` with only

$$
L D+\Psi[u]=f,
\qquad T D=\psi[u]+v.
$$

An unrestricted `v` can absorb the second relation, so zero transformed loss need not
imply that `D` equals the declared integral.

## Choose the appropriate path

| Goal | Phydrax path |
| --- | --- |
| Learn a global field from deterministic nonlocal physics | `DomainFunction` integral operator + `ResidualPenalty` |
| Enforce a randomized integrated moment | `RandomizedMomentPenalty` |
| Solve a supplied causal Volterra equation | `ConvolutionVolterraProblem` or `StochasticVolterraProblem` |
| Solve a supplied Caputo initial-value problem | `CaputoFractionalProblem` |
| Learn a fractional field solution | deterministic `caputo_time_fractional` inside a residual |
| Replace an integral by exact local memory | named auxiliary field + complete differential/trace conditions |

Always report the inner quadrature or estimator, its budget, the outer residual
measure, and an independent operator-consistency check.
