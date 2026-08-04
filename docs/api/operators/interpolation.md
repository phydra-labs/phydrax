# Smolyak interpolation

Phydrax fits reusable sparse polynomial surrogates as ordinary
`DomainFunction` objects. The fitted nodal state is immutable and excluded from
solver parameter partitions; evaluation remains compatible with JAX transforms
and Phydrax's named point batches.

## Fit and evaluate

```python
import jax.numpy as jnp
import phydrax as phx

x = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
z = phx.domain.ProbabilityDomain(
    phx.uq.Normal(0.0, 1.0),
    label="z",
)
domain = x @ z

@domain.Function("x", "z")
def observable(x, z):
    return jnp.stack((x + z, x**2 + z**2))

plan = phx.operators.SmolyakInterpolationPlan(
    2,
    5,
    anisotropy=(1.0, 1.5),
    axis_rules="auto",
)
approximation = phx.operators.interpolate_smolyak(observable, plan)
value = approximation({"x": jnp.asarray(0.2), "z": jnp.asarray(-0.4)})
```

The plan's axis order is the source function's `deps` order, not necessarily the
containing domain's label order. Every dependency must be a scalar factor.
Factors that remain in the domain but are absent from `deps` are preserved and
are not sampled during fitting.

`axis_rules="auto"` selects:

- nested real Leja nodes for bounded scalar and uniform-reference probability
  factors;
- Gauss--Hermite nodes for standard-normal-reference probability factors.

Explicit choices are `"leja"`, `"clenshaw-curtis"`, and
`"gauss-hermite"`. Probability interpolation requires a producer-owned
bidirectional reference transform. Built-in `Uniform`, `Normal`, and
`LogNormal` distributions provide one; `EmpiricalDistribution` does not.

Fitting evaluates the source through one coupled `PointsBatch`, coalescing all
structurally identical nested nodes first. The resulting
`SmolyakInterpolant` groups tensor terms by exact node-count signature and
vectorizes terms within each group. Scalar, vector, matrix, tensor, real, and
complex array outputs are retained without flattening.

The fit is eager and snapshots the source values. Repeated evaluations do not
retain or call the source function. Use `equinox.filter_jit` for complete
`DomainFunction`/`PointsBatch` calls or ordinary `jax.jit` around an array-only
query wrapper. Automatic first and higher derivatives use the interpolating
polynomial, including at interpolation nodes.

## Integration

An interpolant has no separate integration convention. Integrate the returned
`DomainFunction` against an explicit Phydrax target and plan:

```python
estimate = phx.integration.integrate(
    approximation,
    phx.integration.over(domain.component()),
    phx.integration.SparseGridPlan(
        2,
        5,
        axis_rules=("clenshaw-curtis", "gauss-hermite"),
    ),
)
```

This keeps physical integrals, normalized expectations, density targets, and
integration diagnostics under the existing measure-aware integration
substrate.

The implementation independently adopts sparse-combination and grouped tensor
ideas described by [SmolyAX](https://github.com/JoWestermann/smolyax) and its
[JOSS paper](https://github.com/JoWestermann/smolyax/blob/main/paper/paper.md),
without adding SmolyAX as a dependency.

## API

::: phydrax.operators.SmolyakInterpolationPlan

---

::: phydrax.operators.SmolyakInterpolant

---

::: phydrax.operators.interpolate_smolyak
