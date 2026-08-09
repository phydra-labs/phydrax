# Inverse spacetime geometry

This workflow infers one Lorentzian metric from metric-component and curvature
observations while preserving its signature by construction. The trainable object is
an `ADMParameterization`: unconstrained raw lapse and spatial-factor fields become

\[
\alpha=\operatorname{softplus}(\widetilde\alpha)+\alpha_{\min}>0,
\qquad
\gamma=L L^{\mathsf T}>0.
\]

The resulting spacetime metric therefore remains nondegenerate and Lorentzian for
every optimizer iterate. No eigenvalue clipping or post-update repair is involved.

## Complete executable workflow

The example has one trainable isotropic expansion coefficient. It fits the metric
matrix and scalar curvature at the same sensor points. A larger neural raw-factor
model can replace `_IsotropicRawFactor` without changing the observation or solver
code.

```python
import equinox as eqx
import jax.numpy as jnp
import optax

import phydrax as phx


class _ConstantRawLapse(eqx.Module):
    def __call__(self, coordinates):
        minimum = jnp.asarray(1e-6, dtype=coordinates.dtype)
        return jnp.log(jnp.expm1(1.0 - minimum))


class _ZeroShift(eqx.Module):
    def __call__(self, coordinates):
        return jnp.zeros((3,), dtype=coordinates.dtype)


class _IsotropicRawFactor(eqx.Module):
    expansion: jnp.ndarray
    baseline: float = eqx.field(static=True)

    def __init__(self, expansion):
        self.expansion = jnp.asarray(expansion)
        self.baseline = 0.4

    def __call__(self, coordinates):
        raw_diagonal = self.baseline + self.expansion * coordinates[0]
        return jnp.eye(3, dtype=coordinates.dtype) * raw_diagonal


def parameterization(expansion, chart):
    return phx.metrix.ADMParameterization(
        _ConstantRawLapse(),
        _ZeroShift(),
        _IsotropicRawFactor(expansion),
        chart=chart,
    )


domain = phx.domain.HyperRectangle([-1.0] * 4, [1.0] * 4, label="x")
component = domain.component()
chart = phx.metrix.CoordinateChart("inverse_adm", ("t", "x", "y", "z"))

candidate = parameterization(-0.1, chart).metric()
target = parameterization(0.35, chart).metric()
metric_field = phx.operators.as_lorentzian_metric_field(
    domain, candidate, var="x"
)
target_metric_field = phx.operators.as_lorentzian_metric_field(
    domain, target, var="x"
)
target_curvature = phx.operators.domain_scalar_curvature(
    domain, target, var="x"
)


def scalar_curvature_observable(candidate_field):
    metric = phx.operators.lorentzian_metric_from_field(
        candidate_field,
        chart=chart,
        var="x",
    )
    return phx.operators.domain_scalar_curvature(
        candidate_field.domain,
        metric,
        var="x",
    )


metric_data = phx.conditions.Observation(
    "metric",
    component,
    target_metric_field,
    label="metric-data",
)
curvature_data = phx.conditions.Observation(
    "metric",
    component,
    target_curvature,
    operator=scalar_curvature_observable,
    label="curvature-data",
)
points = jnp.array(
    [
        [0.0, -0.4, 0.1, 0.2],
        [0.3, 0.2, -0.3, 0.1],
        [0.6, 0.1, 0.2, -0.2],
        [0.9, -0.2, -0.1, 0.3],
    ]
)
batch = component.points(points)
source = phx.integration.fixed(
    phx.integration.from_samples(
        phx.integration.mean_over(component),
        batch,
    )
)
solver = phx.solver.FunctionalSolver(
    functions={"metric": metric_field},
    terms=(
        phx.terms.ObservationPenalty(metric_data, source),
        phx.terms.ObservationPenalty(curvature_data, source, scale=0.1),
    ),
)
initial_loss = solver.loss()
trained = solver.solve(
    num_iter=80,
    optim=optax.adam(0.04),
    keep_best=True,
    jit=True,
    log_every=0,
)

trained_metric = phx.operators.lorentzian_metric_from_field(
    trained["metric"],
    chart=chart,
    var="x",
)
decomposition = phx.metrix.decompose_adm_metric(trained_metric, points)
report = phx.metrix.validate_adm_decomposition(
    decomposition,
    reference_metric=trained_metric(points),
)

print("loss:", float(initial_loss), "->", float(trained.loss()))
print("ADM valid:", bool(report.valid))
print("minimum lapse:", float(report.minimum_lapse))
print("minimum spatial eigenvalue:", float(report.minimum_spatial_eigenvalue))
```

## One parameter tree, many geometric observables

Only `metric_field` belongs in `FunctionalSolver.functions`. The curvature observation
reconstructs a `LorentzianMetric` from the current field inside its operator. This
makes metric data, Ricci/scalar/Einstein observables, and future physics residuals
share one optimizer parameter tree. Registering separately constructed trainable
metric and curvature fields would instead create independent parameter copies.

`as_lorentzian_metric_field` and `lorentzian_metric_from_field` are deterministic
adapters. Stochastic layers or dropout do not define a differentiable metric unless
their randomness is fixed outside the metric calculus.

## Hypersurface diagnostics and constraints

For a time-first ADM chart, Phydrax supplies the future unit normal, mixed spatial
projector, extrinsic curvature, and Einstein constraints. Its extrinsic-curvature
convention is

\[
K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}
=\frac{D_i\beta_j+D_j\beta_i-\partial_t\gamma_{ij}}{2\alpha}.
\]

With Einstein coupling \(\kappa\), the returned residuals are

\[
\mathcal H={}^{(3)}R+K^2-K_{ij}K^{ij}-2\kappa\rho,
\qquad
\mathcal M^i=D_j(K^{ij}-\gamma^{ij}K)-\kappa S^i.
\]

```python
normal = phx.metrix.adm_normal_vector(trained_metric, points)
projector = phx.metrix.adm_spacetime_projector(trained_metric, points)
extrinsic = phx.metrix.adm_extrinsic_curvature(trained_metric, points)
constraints = phx.metrix.adm_constraint_residuals(
    trained_metric,
    points,
    energy_density=0.0,
    momentum_density=jnp.zeros((3,)),
)
```

The ADM lapse, shift, and positive spatial metric are convention-independent.
`adm_normal_covector` and the four-dimensional metric/projector carry the declared
`"mostly_plus"` or `"mostly_minus"` sign. Source signs in the constraint functions
follow the equations displayed above.
