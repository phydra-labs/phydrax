<h1 align='center'>Phydrax</h1>

## Getting started

Phydrax is a scientific machine learning toolkit for PDEs, constraints, and domain-aware models, built on [JAX](https://github.com/jax-ml/jax) + [Equinox](https://github.com/patrick-kidger/equinox).
It provides composable building blocks for geometry, operators, and training pipelines, with
an emphasis on software modularity and interoperability.

## Unifying view: minimize functionals over domains

Phydrax organizes PDE/physics learning around a single pattern:

1) choose a domain (and components like interior/boundary/slices),  
2) define fields as functions on that domain,  
3) build composable operators of domain functions,  
4) build scalar objectives (functionals) as integrals/means of residuals over components,  
5) minimize the resulting functional.

Conceptually, a typical objective has the form

$$
L[u] = \sum_i w_i\int_{\Omega_i}\rho_i(u(z))\,d\mu_i(z),
$$

where each term corresponds to a constraint, data fit, or integral target on a domain component.

## Core objects (mental model)

Most workflows are composing a few primitives:

- **Domain**: a labeled product space $\Omega=\Omega_x\times\Omega_t\times\cdots$.
- **Component**: a subset like interior/boundary/initial slice where a term lives.
- **DomainFunction**: a field $u :\Omega\to\mathbb{R}^m$ with explicit label dependencies.
- **Operators**: maps $u\mapsto r$ like $\nabla u$, $\Delta u$, $\partial_t u$, integrals, etc.
- **Constraints**: scalar loss terms built from residuals on components.
- **FunctionalSolver**: sums constraints into a differentiable scalar objective and runs optimization.

Optional (but central in many PDE problems):

- **Enforced constraints**: build an ansatz $\tilde u$ that satisfies boundary/initial conditions by construction,
  then train on the remaining terms.

## Core flow

If you are new to the library, the general recipe is:

1. Define a domain (space, time, or products of both).
2. Define functions on that domain.
3. Add constraints and operators to construct a loss $L$.
4. Train or evaluate with a solver.

## Example

This example trains a neural field $u_\theta(x,y)$ to satisfy

$$
\Delta u = 4 \quad \text{in }\Omega=[-1,1]^2,\qquad
u = g \quad \text{on }\partial\Omega,
$$

*The configurations are kept minimal for structural demonstration purposes. Convergence requires larger networks, more iterations, and hyperparameter tuning.*

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
    # For deeper repeated stacks, consider scan=True to reduce compile cost.
    scan=False,
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

## Installation

Requires Python 3.11+.

First, install your preferred JAX distribution.
Otherwise, Phydrax will default to the cpu version.

```bash
uv add phydrax
```

No special builds or containers. Batteries-included, ready to go.

## Documentation

Can be found [here](https://phydra-labs.github.io/phydrax).

## Why JAX?

Partial Differential Equations and their variants are most naturally expressed in the language of operators, which can be thought of as maps between function spaces. While functions map points to values (think `Array`s), operators map entire functions to new functions.

JAX’s functional programming model and higher-order transformations act precisely as operators on functions. This creates a clean correspondence between the abstract operator calculus of PDEs and their concrete, composable, high-performance numerical realizations.

Furthermore, the JAX SciML ecosystem contains many fantastic libraries and projects, and Phydrax aims to be fully-compatible with them to push the possibilities of SciML as far as they can go.

## License

Source-available under the Phydra Non-Production License (PNPL).  
Research/piloting encouraged. 
Production/commercial use requires a separate license.

For production licensing and all other commercial inquiries including consulting, contracting, and custom software: partner@phydra.ai, or DM us on [X](https://x.com/PhydraLabs) or [LinkedIn](https://www.linkedin.com/company/phydra-labs).
