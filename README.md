<h1 align='center'>Phydrax</h1>

## Getting started

Phydrax is a scientific machine learning toolkit for PDEs, constraints,
domain-aware models, stochastic inference, differential equations, and control,
built on [JAX](https://github.com/jax-ml/jax) +
[Equinox](https://github.com/patrick-kidger/equinox). Its components expose
physical axes, masks, validity/status, stable identifiers, and numerical
provenance rather than hiding those distinctions behind one solver interface.

## Unifying view: minimize functionals over domains

Phydrax organizes PDE/physics learning around a single pattern:

1) choose a domain (and components like interior/boundary/slices),  
2) define fields as functions on that domain,  
3) build composable operators of domain functions,  
4) build scalar objectives (functionals) as integrals/means of residuals over components,  
5) minimize the resulting functional.

Conceptually, the optimized functional has the form

$$
\mathcal J[u] = \sum_i \ell_i[u] + \sum_j \mathcal F_j[u] + \sum_k r_k(\theta),
$$

where $\ell_i$ are residual/data constraint penalties, $\mathcal F_j$ are raw scalar
functionals such as signed energies, and $r_k$ are model-level losses.

## Core objects (mental model)

Most workflows are composing a few primitives:

- **Domain**: a labeled product space $\Omega=\Omega_x\times\Omega_t\times\cdots$.
- **Geometry**: analytic, simplicial, B-Rep, CSG, and reconstructed sources compile
  to one JAX kernel/state contract with explicit capabilities, field certificates,
  boundary atlases, topology identities, and design parameters.
- **Component**: a subset like interior/boundary/initial slice where a term lives.
- **Metrix**: differentiable coordinate and Riemannian geometry—charts, tensor
  transformations, metrics, connections, curvature, embedded charts, and
  metric-aware stochastic operators.
- **DomainFunction**: a real- or complex-valued field $u:\Omega\to\mathbb{R}^m$
  or $\mathbb{C}^m$ with explicit label dependencies.
- **Operators**: maps $u\mapsto r$ such as differential, integral, mechanics,
  and quantum matrix operators.
- **Integration**: explicit targets define measures, plans define numerical
  realizations, and estimates carry method-valid diagnostics and provenance.
- **Interpolation**: reusable anisotropic Smolyak surrogates preserve labeled
  domains, array-valued outputs, and JAX differentiation.
- **Stochastic processes and inference**: reproducible processes and
  trajectories, state-space models, Gaussian factors and nonlinear moment
  transforms, continuous-discrete inference, finite-state and particle
  filtering/smoothing, structural components, BSDEs, and finite-rank spatial
  noise.
- **Differential-equation solvers**: deterministic and stochastic integration,
  differentiable driving paths and CDEs, Neural CDE training, probabilistic
  numerical ODEs, and flow/map Lyapunov spectra.
- **Control and optimization**: typed finite-horizon problems, parameterized
  controls, sampled costs/constraints, linearization, frequency response,
  Lyapunov/Riccati equations, Gramians, LQR/iLQR, compiled QPs, multiple shooting,
  bounded initialization search, and receding-horizon MPC.
- **Sequence mixing**: `DiagonalStateSpaceMixer` is one generic diagonal
  continuous-time layer for irregular masked schedules, not a family of branded
  mixer classes.
- **Constraints**: scalar loss terms built from residuals on components.
- **Objectives**: raw scalar terms, including signed integral energies for Ritz minimization.
- **Model losses**: optional parameter-space penalties attached directly to models.
- **FunctionalSolver**: sums constraints, raw objectives, and model losses into a differentiable scalar functional and runs optimization.

Optional (but central in many PDE problems):

- **Enforced constraints**: build an ansatz $\tilde u$ that satisfies boundary/initial conditions by construction,
  then train on the remaining terms.

## Core flow

If you are new to the library, the general recipe is:

1. Define a domain (space, time, or products of both).
2. Define functions on that domain.
3. Add constraints, raw objectives, and operators to construct a functional $\mathcal J$.
4. Train or evaluate with a solver.

Geometry construction lives in `phx.geometry`; `phx.domain.GeometryDomain` is the
thin labeled-domain adapter used by sampling, integration, and constraints. See the
[geometry substrate API](docs/api/geometry.md).

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

# Conditions state scientific semantics.
interior = geom.component()
pde = phx.conditions.Residual(
    "u",
    interior,
    lambda u: phx.operators.laplacian(u, var="x") - 4.0,
)

boundary = geom.component({"x": phx.domain.Boundary()})
bc = phx.conditions.Dirichlet("u", boundary, target=g)

# Integration sources own numerical realization; terms produce scalar penalties.
pde_penalty = phx.terms.ResidualPenalty(
    pde,
    phx.integration.per_step(
        phx.integration.mean_over(pde.on),
        phx.integration.MonteCarloPlan(64),
    ),
)
bc_penalty = phx.terms.ResidualPenalty(
    bc,
    phx.integration.per_step(
        phx.integration.mean_over(bc.on),
        phx.integration.MonteCarloPlan(32),
    ),
    scale=10.0,
)

solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(pde_penalty, bc_penalty),
    enforcement=None,
)
solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
```

## Installation

Requires Python 3.11+.

First, install your preferred JAX distribution.
Otherwise, Phydrax will default to the cpu version.

```bash
uv add phydrax
```

The core install requires no special build or container. Optional integrations
are installed explicitly when needed.

## Documentation

Can be found [here](https://phydra-labs.github.io/phydrax).

Mathematical guides include
[Lagrangian and Hamiltonian mechanics](docs/guides_mechanics.md) and
[quantum operators and dynamics](docs/guides_quantum.md).
[Metrix](docs/api/metrix/index.md) adds metric-aware differential geometry for
curvilinear PDEs, manifold PINNs, embedded geometry, and Riemannian stochastic
generators.

Neural-operator support includes canonical source/query batches, DeepONet,
N-dimensional FNO variants including the experimental-tier dealiased HOFNO, graph
and geometry-informed operators, and transformer operators. See the
[operator-learning cookbook](docs/cookbook/operator_learning.md) for execution
contracts and audited architecture comparisons, and the
[architecture API](docs/api/nn/architectures.md) for constructor details.

Stochastic support includes reproducible SDE/SPDE path ensembles, finite-rank
spatial noise, semilinear exponential integration, convergence/error-budget
diagnostics, strong/weak/mild physics contracts, static random fields, latent
coefficient processes, and process-consistent marginal or pathwise
neural-operator rollouts. See the
[uncertainty guide](docs/guides_uncertainty.md),
[neural-operator uncertainty API](docs/api/uq/operator.md), and
[differential solver API](docs/api/solver/differential.md).

State-space inference includes rank-aware `GaussianFactor` operations, declared
nonlinear Gaussian transforms, continuous-discrete filtering and smoothing,
covariance-form and square-root Kalman paths, exact finite-state completion,
particle and ensemble smoothers, Rao--Blackwellized filtering, and compiled
structural components. Results keep physical cases, schedules and masks,
ancestry, IDs, validity/status, and approximation/regularization/backend
provenance explicit. Nonlinear transforms remain approximations; dense paths
enforce dimension guards; invalid covariance inputs are not silently repaired.
Square-root Kalman filtering is sequential only, and particle ancestry is
nondifferentiable. See the
[filtering cookbook](docs/cookbook/filtering.md),
[state-space API](docs/api/stochastic/state_space.md), and
[inference API](docs/api/uq/inference.md).

Controlled-dynamics support includes explicit causal or offline differentiable
driving paths, Diffrax-backed CDE integration, Neural CDE training, probabilistic
numerical ODE solutions, and flow/map Lyapunov spectra. A probabilistic ODE
solution quantifies numerical integration uncertainty; it is not a posterior
over an unknown physical model. See the
[controlled-dynamics cookbook](docs/cookbook/controlled_dynamics.md) and
[differential solver API](docs/api/solver/differential.md).

Control support includes linear and differential dynamics, control
parameterizations, sampled costs and constraints, linearization, Lyapunov and
Riccati equations, Gramians, frequency response, LQR/iLQR, dense multiple
shooting, compiled linear-control QPs, and receding-horizon MPC. Sampled
nonlinear constraints are not continuous-time certificates; iLQR and multiple
shooting accept one physical case; bounded coefficient search is not globally
optimal; dense guards and solver status are explicit rather than hidden behind
fallback or repair. See the [control cookbook](docs/cookbook/control.md),
[control API](docs/api/control.md), and [QP API](docs/api/optim.md).

Sensitivity utilities add score/Fisher actions and empirical
controllability/observability directions. Stationary linear-Gaussian state,
output, and cross spectra reuse diagnosed control resolvents and reject unstable,
singular, non-Hermitian, or non-positive-semidefinite inputs instead of clipping
or repairing them.

QPax 0.1.4 is a core runtime dependency and is integrated through its implicit
backend. The Phydrax dense solver remains the default; select QPax explicitly with
`method="qpax-implicit"`. QP results preserve primal/dual residuals, regularization,
validity/status, and backend provenance, and neither backend is used as a hidden
fallback.

High-dimensional PDE support is structure-aware rather than a claim that one generic
PINN removes the curse of dimensionality. Semilinear parabolic equations can use
query-conditioned or trajectory-node Feynman--Kac labels, global Deep Picard,
localized Deep BSDE terminal shooting, or backward Deep Splitting slice regression.
Strong-form residuals can use raw Hutchinson probes or unbiased coordinate sampling
without dense Hessians; high-dimensional density dynamics can learn time-conditioned
scores from particles. The executable benchmark harness exercises those production
paths rather than renamed analytic baselines:

```bash
python tools/high_dimensional_pde_benchmarks.py \
  --suite methods --dimensions 10,100,1000 --include-training
```

The checked benchmark output for dimensions 10, 100, and 1000 is
[`benchmarks/high_dimensional_methods.json`](benchmarks/high_dimensional_methods.json).

Deep Picard, Deep BSDE, and Deep Splitting are emitted only for dimensions declared
by their method specifications. Every record reports error, estimator uncertainty
when applicable, compile and steady runtime, valid fraction, and an explicit
working-set estimate. See the [BSDE cookbook](docs/cookbook/bsde.md),
[stochastic-dynamics cookbook](docs/cookbook/stochastic_dynamics.md), and
[randomized differential API](docs/api/operators/differential.md#raw-probes-and-coordinate-sampling).

## Why JAX?

Partial Differential Equations and their variants are most naturally expressed in the language of operators, which can be thought of as maps between function spaces. While functions map points to values (think `Array`s), operators map entire functions to new functions.

JAX’s functional programming model and higher-order transformations act precisely as operators on functions. This creates a clean correspondence between the abstract operator calculus of PDEs and their concrete, composable, high-performance numerical realizations.

Furthermore, the JAX SciML ecosystem contains many fantastic libraries and projects, and Phydrax aims to be fully-compatible with them to push the possibilities of SciML as far as they can go.

## License

Source-available under the Phydra Non-Production License (PNPL).  
Research/piloting encouraged. 
Production/commercial use requires a separate license.

For production licensing and all other commercial inquiries including consulting, contracting, and custom software: partner@phydra.ai, or DM us on [X](https://x.com/PhydraLabs) or [LinkedIn](https://www.linkedin.com/company/phydra-labs).
