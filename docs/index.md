# Getting started

Phydrax is a scientific machine learning toolkit for PDEs, conditions, and domain-aware models, built on [JAX](https://github.com/jax-ml/jax) + [Equinox](https://github.com/patrick-kidger/equinox).
It provides composable building blocks for geometry, operators, explicit integration, and training pipelines, with
an emphasis on explicit numerical measures and data sampling.

## Unifying view: minimize functionals over domains

Phydrax organizes PDE/physics learning around a single pattern:

1) choose a domain (and components like interior/boundary/slices),  
2) define fields as functions on that domain,  
3) build composable operators of domain functions,  
4) describe residuals, moments, and observations as conditions,
5) attach each condition to an explicit integration source and train on its penalty term.

Conceptually, the optimized functional has the form

$$
\mathcal J[u] = \sum_i \ell_i[u] + \sum_k r_k(\theta),
$$

where each \(\ell_i\) is a nonnegative residual, moment, or observation penalty
with an explicit source, and \(r_k\) is a model-level loss.

## Core objects (mental model)

Most workflows are composing a few primitives:

- **Domain**: a labeled product space \(\Omega=\Omega_x\times\Omega_t\times\cdots\).
- **Component**: a subset like interior/boundary/initial slice where a term lives.
- **Metrix**: explicit charts and maps, tensor and form laws, positive and signed
  metrics, affine connections and curvature, Lie/Poisson/horizontal structures,
  and array manifolds for constrained optimization and geometric integration.
- **DomainFunction**: a real- or complex-valued field
  \(u:\Omega\to\mathbb{R}^m\) or \(\mathbb{C}^m\) with explicit label dependencies.
- **Operators**: maps \(u\mapsto r\) such as differential, integral, mechanics,
  and quantum matrix operators.
- **Integration**: explicit targets define measures, plans define numerical
  realizations, and estimates carry method-valid diagnostics and provenance.
- **Optimal transport**: finite measures lower into balanced discrete problems with
  explicit mass, geometry, entropic regularization, convergence diagnostics, and
  matrix-free coupling actions.
- **Conditions**: typed residual, moment, and observation declarations on components.
- **Terms**: nonnegative penalties that turn conditions into trainable scalar terms.
- **Model losses**: optional parameter-space penalties attached directly to models.
- **FunctionalSolver**: sums terms and model losses into a differentiable scalar functional and runs Optax, structured KFAC, Evosax, or explicit product-manifold optimization.
- **Native ML**: immutable JAX recipes for preprocessing, linear/probabilistic
  supervision, decomposition, kernels, neighbors, covariance, clustering,
  manifolds, trees, ensembles, selection, metrics, artifacts, and audited
  fitted-model conversion. Fits return numerical and derivative contracts.

Optional (but central in many PDE problems):

- **Enforcement**: build an ansatz \(\tilde u\) that satisfies boundary/initial conditions by construction,
  then train it against the remaining terms.

## Core flow

If you are new to the library, start with:

1. Define a domain (space, time, or products of both).
2. Define functions on that domain.
3. Declare conditions, their explicit integration sources, and penalty terms to construct a functional $\mathcal J$.
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

geom = phx.domain.GeometryDomain(
    phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
)  # [-1,1]^2, label "x"


# Exact solution / boundary target g(x,y) = x^2 + y^2
@geom.Function("x")
def g(x):
    return x[0] ** 2 + x[1] ** 2


# Trainable field u_theta(x)
model = phx.nn.models.MLP(
    in_size=2,
    out_size="scalar",
    width_size=16,
    depth=2,
    # For deeper repeated stacks, consider scan=True to reduce compile cost.
    scan=False,
    key=jr.key(0),
)
u = geom.Model("x")(model)

layout = phx.domain.SampleLayout((("x",),))
interior = geom.component()

# Interior PDE residual: Δu - 4 = 0
pde_condition = phx.conditions.Residual(
    "u",
    interior,
    lambda u: phx.operators.laplacian(u, var="x") - 4.0,
)
pde_source = phx.integration.per_step(
    phx.integration.mean_over(pde_condition.on),
    phx.domain.PointSampling(64, layout=layout),
)
pde_term = phx.terms.ResidualPenalty(pde_condition, pde_source)

# Soft Dirichlet boundary: u - g = 0 on ∂Ω
boundary = geom.component({"x": phx.domain.Boundary()})
boundary_condition = phx.conditions.Residual("u", boundary, lambda u: u - g)
boundary_source = phx.integration.per_step(
    phx.integration.mean_over(boundary_condition.on),
    phx.domain.PointSampling(32, layout=layout),
)
boundary_term = phx.terms.ResidualPenalty(boundary_condition, boundary_source, scale=10.0)

solver = phx.solver.FunctionalSolver(functions={"u": u}, terms=[pde_term, boundary_term])
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

## Why JAX?

Partial Differential Equations and their variants are most naturally expressed in the language of operators, which can be thought of as maps between function spaces. While functions map points to values (think `Array`s), operators map entire functions to new functions.

JAX’s functional programming model and higher-order transformations act precisely as operators on functions. This creates a clean correspondence between the abstract operator calculus of PDEs and their concrete, composable, high-performance numerical realizations.

Furthermore, the JAX SciML ecosystem contains many fantastic libraries and projects, and Phydrax aims to be fully-compatible with them to push the possibilities of SciML as far as they can go.

## License

Source-available under the Phydra Non-Production License (PNPL).  
Research/piloting encouraged. 
Production/commercial use requires a separate license.

For production licensing and all other commercial inquiries including consulting, contracting, and custom software: partner@phydra.ai, or DM us on [X](https://x.com/PhydraLabs) or [LinkedIn](https://www.linkedin.com/company/phydra-labs).

<br>
Next: [All of Phydrax](all-of-phydrax.md)
