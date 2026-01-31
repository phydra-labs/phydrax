# Getting started

Phydrax is a scientific machine learning toolkit for PDEs, constraints, and domain-aware models, built on [JAX](https://github.com/jax-ml/jax) + [Equinox](https://github.com/patrick-kidger/equinox).
It provides composable building blocks for geometry, operators, and training pipelines, with
an emphasis on explicit control of objectives and data sampling.

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

- **Domain**: a labeled product space \(\Omega=\Omega_x\times\Omega_t\times\cdots\).
- **Component**: a subset like interior/boundary/initial slice where a term lives.
- **DomainFunction**: a field \(u:\Omega\to\mathbb{R}^m\) with explicit label dependencies.
- **Operators**: maps \(u\mapsto r\) like \(\nabla u\), \(\Delta u\), \(\partial_t u\), integrals, etc.
- **Constraints**: scalar loss terms built from residuals on components.
- **FunctionalSolver**: sums constraints into a differentiable scalar objective and runs optimization.

Optional (but central in many PDE problems):

- **Enforced constraints**: build an ansatz \(\tilde u\) that satisfies boundary/initial conditions by construction,
  then train on the remaining terms.

## Core flow

If you are new to the library, start with:

1. Define a domain (space, time, or products of both).
2. Define functions on that domain.
3. Add constraints and operators to construct a loss $L$.
4. Train or evaluate with a solver.

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

For production licensing and all other commercial inquiries including consulting, contracting, and custom software: partner@phydra.ai, or DM us on [X](https://x.com/PhydraLabs).

<br>
Next: [All of Phydrax](all-of-phydrax.md)
