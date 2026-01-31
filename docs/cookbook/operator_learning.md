# Operator learning (DatasetDomain × coordinates)

This recipe shows the “operator-learning” decomposition

$$
\Omega \;=\; \Omega_{\text{data}}\times\Omega_x\times\cdots,
$$

where \(\Omega_{\text{data}}\) indexes a dataset of inputs (forcing, coefficients, initial conditions, etc.) and
\(\Omega_x\) is the coordinate domain where you evaluate outputs.

In Phydrax, \(\Omega_{\text{data}}\) is represented by `DatasetDomain`, and operator models are wrapped via
`Domain.Model(...)` so they can be used like any other `DomainFunction`.

## Dataset factor

`DatasetDomain` stores an in-memory PyTree of arrays with a shared leading dataset axis, and samples by indexing.
See [API → Domain → Composition](../api/domain/composition.md).

## DeepONet skeleton on \(\Omega_{\text{data}}\times\Omega_x\) {: data-toc-label="DeepONet skeleton on Ω_data × Ω_x"}

Assume each dataset sample contains a vector of coefficients \(c\in\mathbb{R}^K\) that parameterizes an input.
For this runnable example, we choose a simple analytic “operator” that maps \(c\) to a 1D field
\(u(x)=\sum_{k=1}^K c_k \sin(k\pi x)\).

!!! example
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import phydrax as phx

    key = jr.key(0)

    # N dataset samples, each carrying K coefficients.
    N = 32
    K = 8
    coeffs = jr.normal(key, shape=(N, K))

    data_dom = phx.domain.DatasetDomain(coeffs, label="data", measure="probability")
    geom = phx.domain.Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    latent = 32
    branch = phx.nn.MLP(in_size=K, out_size=latent, width_size=64, depth=2, key=jr.key(1))
    trunk = phx.nn.MLP(in_size=1, out_size=latent, width_size=64, depth=2, key=jr.key(2))
    deeponet = phx.nn.DeepONet(branch=branch, trunk=trunk, coord_dim=1, latent_size=latent)

    # u_hat(data, x): predicted field on the x-axis for each dataset sample
    u_hat = domain.Model("data", "x", structured=True)(deeponet)

    # Supervised target u_true(data, x): analytic mapping from coefficients to a function of x
    @domain.Function("data", "x")
    def u_true(c, x):
        x_axis = x[0]
        ks = jnp.arange(1, K + 1, dtype=float)
        basis = jnp.sin(jnp.pi * ks[:, None] * x_axis[None, :])  # (K, nx)
        return basis.T @ c  # (nx,)

    # Supervised residual on Ω_data × Ω_x.
    def residual(u_f):
        return u_f - u_true

    # Build a grid-aligned supervised loss by sampling data densely and x as a coord-separable axis.
    nx = 32
    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=domain.component(),
        operator=residual,
        constraint_vars="u",
        num_points=8,  # number of dataset samples per step
        structure=phx.domain.ProductStructure((("data", "x"),)),
        coord_separable={"x": phx.domain.UniformAxisSpec(nx)},
        dense_structure=phx.domain.ProductStructure((("data",),)),
        reduction="mean",
    )

    solver = phx.solver.FunctionalSolver(functions={"u": u_hat}, constraints=[constraint])
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

!!! note
    This page focuses on the domain/model wiring. For structured-input conventions and operator architectures (DeepONet/FNO),
    see [API → NN → Architectures](../api/nn/architectures.md). For sampling semantics, see [Guides → Domains and sampling](../guides_domain.md).
