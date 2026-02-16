# All of Phydrax

This page provides a high-level map of the library, how the parts fit together,
and where to look for specific functionality.

## Unifying formalism: minimizing functionals over domains

Phydrax is designed to make a single idea modular:

> Define fields on labeled domains and minimize scalar **functionals** built from operators and
> measures over domain components.

A typical objective can be written as

$$
L[u] = \sum_i w_i\int_{\Omega_i}\rho_i(u(z))\,d\mu_i(z),
$$

where each $\Omega_i$ is a domain component (interior, boundary, initial slice, data subset, …),
$\mu_i$ is the induced measure, and $\rho_i$ is a pointwise residual penalty (often a squared norm).
The library is organized so each piece (domains, sampling, operators, constraints, solvers) cleanly
corresponds to a part of this expression.

## The compositional contract

At a practical level, most workflows look like:

1) choose a **domain** \(\Omega\) and a **component** \(\Omega_{\text{comp}}\subseteq\Omega\),  
2) define one or more **fields** \(u_\theta:\Omega\to\mathbb{R}^m\) as `DomainFunction`s,  
3) build **residual operators** \(r=\mathcal{N}(u_\theta,\dots)\) using `phydrax.operators`,  
4) turn residuals into **constraint terms** \(\ell_i\) via sampling + reduction (mean/integral),  
5) sum terms into a **functional** \(L=\sum_i \ell_i\) and optimize with `FunctionalSolver`.

Two design choices make this interoperable:

- **Labeled product domains**: every coordinate is a named factor (`"x"`, `"t"`, `"data"`, `"p"`, …).
- **Structured batches**: sampling preserves axis semantics (paired sampling and coord-separable grids).

## Key choice points (what makes workflows differ)

### Sampling: point batches vs coord-separable grids

Phydrax supports two complementary evaluation regimes:

- `PointsBatch` (paired sampling): typical PINN-style collocation constraints. See [Guides → Domains and sampling](guides_domain.md).
- `CoordSeparableBatch` (axis/grid sampling): spectral/basis operators and neural operators (FNO/DeepONet). See [Guides → Differential operators](guides_differential.md).

### Differentiation: AD / jets / FD / basis

Differential operators support multiple backends (`backend="ad"|"jet"|"fd"|"basis"`) and autodiff modes
(`mode="reverse"|"forward"`). For deeper math, see [Appendix → Differentiation modes](appendix/differentiation_modes.md).

### Constraints: soft penalties vs enforced by construction

Boundary/initial conditions can be handled in two ways:

- **Soft**: add boundary/initial constraint terms (e.g. `ContinuousDirichletBoundaryConstraint`) to \(L\).
- **Enforced**: build an ansatz \(\tilde u=\mathcal{H}(u)\) satisfying conditions exactly, then train on the remaining terms.

The enforced route is staged as boundary → initial → interior data. See:

- [API → Constraints → Enforced constraint ansätze](api/constraints/enforced.md)
- [API → Solver → Enforced constraint pipelines](api/solver/enforced_constraints.md)
- [Appendix → Physics-Constrained Interpolation](appendix/physics_constrained_interpolation.md)

### Models: fields vs operators

- **Field learning**: learn \(u_\theta(x,t,\dots)\) directly (MLPs, separable models, etc.).
- **Operator learning**: learn \(G_\theta\) mapping inputs to fields, using a dataset factor \(\Omega_{\text{data}}\) so
  the domain becomes \(\Omega_{\text{data}}\times\Omega_x\times\cdots\). See [API → Domain → Composition](api/domain/composition.md)
  and [API → NN → Architectures](api/nn/architectures.md).

## A first real PDE example: Poisson on a square

This example trains a neural field \(u_\theta(x,y)\) to satisfy

$$
\Delta u = 4 \quad \text{in }\Omega=[-1,1]^2,\qquad
u = g \quad \text{on }\partial\Omega,
$$

with the analytic choice \(g(x,y)=x^2+y^2\) (so the exact solution is \(u^\star(x,y)=x^2+y^2\)).

*The configurations are kept small for demonstration purposes.*

!!! example
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)  # [-1,1]^2, label "x"

    # Exact solution / boundary target g(x,y) = x^2 + y^2
    @geom.Function("x")
    def g(x):
        return x[0] ** 2 + x[1] ** 2

    # Trainable field u_theta(x)
    model = phx.nn.MLP(
        in_size=2,
        out_size="scalar",
        width_size=16,
        depth=2,
        key=jr.key(0),
    )
    u = geom.Model("x")(model)

    structure = phx.domain.ProductStructure((("x",),))

    # Interior PDE residual: Δu - 4 = 0
    pde = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: phx.operators.laplacian(f, var="x") - 4.0,
        num_points=64,
        structure=structure,
        reduction="mean",
    )

    # Soft Dirichlet boundary: u - g = 0 on ∂Ω
    boundary = geom.component({"x": phx.domain.Boundary()})
    bc = phx.constraints.ContinuousDirichletBoundaryConstraint(
        "u",
        boundary,
        target=g,
        num_points=32,
        structure=structure,
        weight=10.0,
        reduction="mean",
    )

    solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=[pde, bc])
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

### Enforced boundary conditions (replace penalties with an ansatz)

Instead of penalizing boundary violations, you can enforce \(u=g\) **by construction** and train only on the interior
PDE term. This is often numerically cleaner and makes the “functional over a domain” story composable: constraints are
just extra terms, while enforcement is a map \(u\mapsto \tilde u\).

!!! example
    ```python
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def g(x):
        return x[0] ** 2 + x[1] ** 2

    model = phx.nn.MLP(in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0))
    u = geom.Model("x")(model)

    structure = phx.domain.ProductStructure((("x",),))
    pde = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: phx.operators.laplacian(f, var="x") - 4.0,
        num_points=64,
        structure=structure,
    )

    boundary = geom.component({"x": phx.domain.Boundary()})
    term = phx.solver.SingleFieldEnforcedConstraint(
        "u",
        boundary,
        lambda f: phx.constraints.enforce_dirichlet(f, boundary, var="x", target=g),
    )

    solver = phx.solver.FunctionalSolver(
        functions={"u": u},
        constraints=[pde],
        constraint_terms=[term],
        boundary_weight_num_reference=128,
        boundary_weight_key=jr.key(1),
    )
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

### Adding data (anchors / sensors) is “just another term”

Phydrax treats data-fit the same way as PDE residuals: as a constraint term on a domain component or point set.
For scattered anchor data \(\{(x_i,y_i)\}\), use `DiscreteInteriorDataConstraint`:

```python
import jax.numpy as jnp
import phydrax as phx

# Continuing from the Poisson example above:
# - geom is the geometry domain
# - u is your trainable field

anchors = jnp.array([[0.0, 0.0], [0.5, -0.5], [-0.25, 0.75]])
values = jnp.sum(anchors**2, axis=1)  # pretend we observed u(x)=x^2+y^2

data = phx.constraints.DiscreteInteriorDataConstraint(
    "u",
    geom,
    points={"x": anchors},
    values=values,
    weight=1.0,
)
```

### Operator learning (dataset × coordinates)

To model operators \(G: f \mapsto u(\cdot)\), represent the domain as a product
\(\Omega=\Omega_{\text{data}}\times\Omega_x\times\cdots\) using `DatasetDomain`, and use a structured model like
DeepONet/FNO. See [API → Domain → Composition](api/domain/composition.md) and
[API → NN → Architectures](api/nn/architectures.md).

## Notation

We use $x$ for spatial variables, $t$ for time, and $u(x, t)$ for fields. A typical
objective aggregates constraints as $L = \sum_i w_i\,\ell_i$.

## By task: “what do I compose?”

Below are the common SciML regimes expressed in Phydrax’s primitives.

- **Forward PDE solve (PINN-style)**: interior residual + boundary/initial terms (soft or enforced).
  Start at [Getting started](index.md) and then [Guides → Constraints](guides_constraints.md).
- **Enforced BC/IC**: build ansätze with `enforce_dirichlet` / `enforce_initial` / etc., and stage them via solver pipelines.
  See [API → Solver → Enforced constraint pipelines](api/solver/enforced_constraints.md).
- **Data assimilation / hybrid physics–data**: add `DiscreteInteriorDataConstraint` / `DiscreteTimeDataConstraint` alongside PDE residuals.
  See [API → Constraints → Discrete](api/constraints/discrete.md).
- **Inverse problems (unknown coefficients/parameters)**: represent unknowns as additional fields or domain parameters, and couple them in residual operators.
  See [API → Domain → Functions](api/domain/functions.md) and [API → Constraints](api/constraints/index.md).
- **Operator learning (DeepONet/FNO)**: use `DatasetDomain` and structured models on \(\Omega_{\text{data}}\times\Omega_x\).
  See [API → Domain → Composition](api/domain/composition.md) and [API → NN → Architectures](api/nn/architectures.md).
- **Integral / conservation laws**: build terms from `integral`/`mean` and use integral constraints (equality targets, flux balances, etc.).
  See [Guides → Integrals and measures](guides_integrals.md).
- **ODEs and dynamical systems**: treat time as a scalar domain and enforce residuals \(\dot u - f(u,t)=0\) via ODE constraints (continuous or discrete).
  See [API → Constraints → Discrete](api/constraints/discrete.md) and [API → Constraints → Continuous](api/constraints/continuous.md).
- **Cookbook recipes**: end-to-end patterns for Poisson, heat, inverse+data, and operator learning.
  Start at [Cookbook → Overview](cookbook/index.md).

## Where to go next

- [Cookbook](cookbook/index.md)
- [Domains and sampling](guides_domain.md)
- [Differential operators](guides_differential.md)
- [Integrals and measures](guides_integrals.md)
- [Constraints and objectives](guides_constraints.md)
- [Solvers and training](guides_solver.md)
- [API reference](api/phydrax.md)
- `phydrax.domain` for geometry, time, and sampling.
- `phydrax.constraints` for loss terms and enforced constraints.
- `phydrax.operators` for PDE operators.
- `phydrax.nn` for models and wrappers.
- `phydrax.solver` for training and evaluation loops.
