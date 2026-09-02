# Neural tangent kernels

`phydrax.nn.neural_tangent` provides finite-width empirical neural tangent
kernels as ordinary `phydrax.linalg` operators. It does not approximate an
infinite-width analytic kernel. Operator preparation remains matrix-free;
diagnostics materialize only below the explicit
`NTKDiagnosticsPolicy.dense_max_dimension` resource bound.

For a parameterized map `f(theta)` with Jacobian `J`,
`prepare_empirical_ntk(...)` exposes:

- the reusable prepared linearization;
- the Jacobian action `J v` and pullback `J* w`;
- the output-space empirical NTK `K = J J*`;
- the Euclidean parameter Gram action `J* J`;
- rectangular cross-kernels between maps prepared at the same parameter point.

The standard kernel uses the Euclidean trainable-parameter pairing. Passing a
`ParameterGeometry` instead constructs the explicitly named parameter-metric
tangent kernel `J G^-1 J*`; this is parameter- and point-dependent and is not
silently presented as the standard NTK.

```python
import jax.numpy as jnp
import phydrax as phx

X = jnp.asarray([[1.0, 2.0], [-1.0, 0.5]])
theta = jnp.asarray([0.2, -0.3])

prepared = phx.nn.neural_tangent.prepare_empirical_ntk(
    lambda value: X @ value,
    theta,
)
action = prepared.kernel.mv(jnp.ones((2,)))
diagnostics = phx.nn.neural_tangent.analyze_ntk(
    prepared,
    policy=phx.nn.neural_tangent.NTKDiagnosticsPolicy(
        dense_max_dimension=64,
        eigenvalue_count=2,
    ),
)
```

## Functional residual NTK

`phydrax.solver.prepare_functional_ntk(...)` differentiates the exact
square-root-weighted residual roots used by `FunctionalSolver`. Consequently,
the resulting kernel includes:

- differential operators in the residual;
- exact enforcement transforms;
- integration weights and physical measure normalization;
- density factors and masks;
- real and imaginary residual coordinates;
- authored residual block metadata;
- an optional exact `ParameterSubspace`.

```python
prepared = phx.solver.prepare_functional_ntk(solver, key=key)
full_kernel = prepared.kernel
momentum_kernel = prepared.block(
    phx.terms.ResidualBlockRef(0, "momentum")
)
```

With an exact `PreparedFunctionalUpdate`, `view="physical"` differentiates the
unchanged authored roots while `view="surrogate"` differentiates the frozen
pseudo-transient, causal, or balanced roots used by that optimizer update.

A kernel is conditioned on one exact parameter point and one exact prepared
integration realization. Cross-step comparisons therefore require an
independent fixed monitor realization. Kernels prepared on newly sampled points
are individually valid, but their entries do not share a stable observation
coordinate system.

## Diagnostics

`NTKDiagnosticsPolicy` selects either bounded dense analysis or matrix-free
Hutchinson/Lanczos analysis. Diagnostics report:

- diagonal and diagonal standard error;
- trace and trace standard error;
- trace of `K^2`;
- leading eigenvalues;
- stable and effective rank;
- dense numerical rank and nullity;
- an active-range condition number only when a positive eigenvalue is actually
  resolved;
- finite/converged evidence.

Rank deficiency is expected when the residual coordinate dimension exceeds the
selected parameter dimension, when enforcement removes sensitivity, or when the
model has parameter symmetries. Phydrax reports this state rather than adding
hidden damping or clipping eigenvalues.

## Training integration

`FunctionalTermBalancePolicy(method="ntk_trace", ...)` periodically balances
selected residual terms or named blocks by their measure-weighted NTK traces.
The estimator is evaluated on one frozen optimizer surrogate, uses common
same-update parameters and source realizations, and is detached before the
optimizer update. A multiplier refresh retains its previous value when the
stochastic relative standard error exceeds the declared policy threshold.

NTK trace balancing is supported only for selected nonnegative residual
penalties. Applying it automatically to signed energies, likelihoods, posterior
terms, or model regularizers would change their scientific meaning and is
rejected.

## API

::: phydrax.nn.neural_tangent.PreparedEmpiricalNTK

---

::: phydrax.nn.neural_tangent.NTKDiagnosticsPolicy

---

::: phydrax.nn.neural_tangent.NTKDiagnostics

---

::: phydrax.nn.neural_tangent.prepare_empirical_ntk

---

::: phydrax.nn.neural_tangent.analyze_ntk

---

::: phydrax.solver.PreparedFunctionalNTK

---

::: phydrax.solver.prepare_functional_ntk
