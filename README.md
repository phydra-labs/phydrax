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
- **Discretization and solver substrates**: tensor supports, local finite
  differences, modal transforms/spectra, cochains, finite elements, finite
  volumes, material-particle supports, conservative SPH, measure-aware
  particle-grid splatting, WENO fluxes, fixed-capacity AMR, field spaces,
  measures, transfers, fixed-temporal differentiable replay, transactional
  gravity/cooling/stochastic source processes, compatible constrained MHD,
  temporal/stochastic composition, and auditable plan/preparation identities.
- **Finite-molecule atomistic learning**: scale-identified atomic structures and
  padded case-isolated batches reuse material-particle identities and `GraphIR`;
  PaiNN scalar/vector interactions produce invariant molecular energies and
  conservative forces with fail-closed neighborhood capacity, typed diagnostics,
  local rMD17 parsing, and domain-specific energy/force training.
- **Computational topology**: compact active subcomplexes, exact field-qualified
  homology, rational Betti dimensions, validated filtrations, persistent homology,
  fixed-capacity diagrams, and independently verified topology–Hodge evidence over
  canonical discretization complexes.
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
- **Sampling**: reference-space designs and fixed-kernel persistent Markov chains
  preserve explicit keys, chain/draw axes, transition evidence, and correlated-measure
  semantics when lowered into integration.
- **Linear algebra**: paired array/PyTree/block spaces, composable dense,
  matrix-free, sparse, and block operators, explicit system/least-squares/
  minimum-norm contracts, reusable plans and factorizations, and portable
  status, diagnostics, and provenance.
- **Empirical parameter geometry**: centered or uncentered weighted feature Gram
  actions compose with the same paired spaces, nullspaces, prepared linear solves,
  and diagnostics as every other matrix-free operator.
- **Optimal transport**: integration measures lower into balanced finite transport
  problems with explicit mass, ground geometry, stabilized Sinkhorn diagnostics,
  matrix-free plan actions, exact/sliced Wasserstein distances, and soft order.
- **Learned probability transport**: endpoint flow matching, deterministic continuous
  flows, VP/VE denoising score matching, replayable reverse-time diffusion, and
  probability-flow densities reuse explicit laws and differential-solver evidence.
- **Combinatorial optimization**: native exact finite, cardinality, assignment,
  and DAG path oracles preserve logical decisions, linear objective features,
  deterministic ties, independent certificates, JIT batching, and explicit
  blackbox surrogate pullbacks.
- **Interpolation**: reusable anisotropic Smolyak surrogates preserve labeled
  domains, array-valued outputs, and JAX differentiation.
- **Positive-definite kernels**: one covariance-safe PyTree algebra serves Gaussian
  processes, coresets, inducing-point selection, learned input transforms, and
  finite-feature inference.
- **Stochastic processes and inference**: reproducible processes and
  trajectories, state-space models, Gaussian factors and nonlinear moment
  transforms, continuous-discrete inference, finite-state and particle
  filtering/smoothing, structural components, BSDEs, and finite-rank spatial
  noise.
- **Differential-equation solvers**: deterministic, stochastic, delay/memory,
  rough, jump/hybrid, semidiscrete, differentiable-control, and probabilistic
  numerical integration.
- **Learned field evolution**: fixed physical measures project PDE rates onto
  selected model tangents, Diffrax evolves the resulting parameter ODE, and
  backward Diffrax characteristics feed optional time-slice field projection.
- **Electromagnetics**: compatible cochain Maxwell and a reciprocal-lattice
  Fourier-modal substrate cover general time-domain topology and periodic layered
  frequency-domain scattering, respectively, with full-tensor finite layers,
  field-certificate-aware geometry rasterization, boundary cascades, current
  planes, Brillouin-zone sources, and diffraction orders.
- **Variational quantum dynamics**: stable complex log amplitudes, connected discrete
  operators, validated finite symmetry sectors, persistent-chain local energies,
  damped SR, frozen R-hat/ESS diagnostics, portable exact-resume checkpoints, and
  real/imaginary-time TDVP reuse the sampling, integration, parameter-subspace, and
  linear-runtime contracts.
- **Dynamical systems, identification, and chaos**: typed flows/maps and
  pathwise evolution, mask-safe trajectory data, DMD/EDMD, strong/discrete/
  integral/weak and implicit SINDy, PDE-FIND, periodic orbits, continuation,
  Floquet/Lyapunov/covariant analysis, recurrence and statistical chaos
  diagnostics, explicit uncertainty aggregation, and a shadowing solver boundary.
- **Control and optimization**: typed finite-horizon problems, parameterized
  controls, sampled costs/constraints, linearization, frequency response,
  Lyapunov/Riccati equations, Gramians, LQR/iLQR, compiled QPs, multiple shooting,
  bounded initialization search, and receding-horizon MPC.
- **Nonlinear optimization**: typed scalar, residual, bound, nonlinear-constrained,
  state/design, stochastic, and continuation problems; matrix-free second-order,
  primal--dual, and moving-asymptote methods; fixed-mesh SIMP compliance design;
  implicit solution derivatives; continuation stability events; explicit status,
  diagnostics, certificates, and provenance.
- **Sequence mixing**: `DiagonalStateSpaceMixer` is the input-independent
  continuous-time baseline; `SelectiveStateSpaceMixer` adds input-dependent
  step, injection, and readout maps while preserving exact irregular-time
  affine recurrence and serial/associative parity.
- **Deployment**: deterministic learned inference boundaries export to ONNX or
  checksummed, pickle-free StableHLO/IREE artifacts with explicit static ABI and
  native parity evidence.
- **Constraints**: scalar loss terms built from residuals on components.
- **Objectives**: raw scalar terms, including signed integral energies for Ritz minimization.
- **Model losses**: optional parameter-space penalties attached directly to models.
- **Parameter geometry**: `phx.nn.parameters` maps raw optimizer coordinates to
  constrained physical values, selects explicit model PyTree subspaces, and
  provides standard or rank-stabilized low-rank adaptation with base-bound
  artifacts and exact deployment merging.
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

Numerical supports and finite field spaces live in `phx.discretization`, including
bounded, periodic, half-line, and real-line global tensor bases; rational Chebyshev
transforms and spectral resolution evidence; material-particle supports and
conservative SPH; and exact-sampling round-sphere spaces with S2FFT transforms,
Laplace--Beltrami actions, area measures, and SFNO interoperability. See the
[discretization guide](docs/guides_discretization.md), the
[particle-method guide](docs/guides_particle_methods.md), the
[particle-grid splatting guide](docs/guides_particle_splatting.md), the
[SPH guide](docs/guides_sph.md), the
[WCSPH guide](docs/guides_wcsph.md), the
[advanced SPH guide](docs/guides_multiphase_incompressible_sph.md), the
[particle qualification guide](docs/guides_particle_qualification.md), the
[global spectral guide](docs/guides_spectral_methods.md), the
[Fourier-modal Maxwell guide](docs/guides_fourier_modal_maxwell.md), the
[solver-substrate guide](docs/guides_solver_substrates.md), and the
[API](docs/api/discretization/index.md).

Finite nonperiodic molecular learning lives in `phx.atomistic`, with the
equivariant energy model in `phx.nn.atomistic`. The implementation covers typed
atomic structures and batches, resource-guarded dense molecular graphs, PaiNN
energies, forces derived from one scalar energy, typed energy/force fitting, and
offline local rMD17 data. It does not claim periodic systems, stress, long-range
electrostatics, or molecular-dynamics stability. See the
[atomistic guide](docs/guides_atomistic.md), the
[atomistic cookbook](docs/cookbook/atomistic.md), and the
[atomistic API](docs/api/atomistic.md).

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

The base install includes ASDEX for compile-time global sparse-derivative
detection and optimized coloring. Compiled sparse Jacobian and Hessian plans
evaluate through native JAX and integrate directly with `phydrax.linalg`; ASDEX
is not imported by ordinary Phydrax or linear-algebra use.

## Documentation

Can be found [here](https://phydra-labs.github.io/phydrax).

Mathematical guides include
[Lagrangian and Hamiltonian mechanics](docs/guides_mechanics.md) and
[quantum operators and dynamics](docs/guides_quantum.md).
The [persistent Markov measure cookbook](docs/cookbook/variational_boltzmann.md)
demonstrates correlated empirical integration outside quantum mechanics, while the
[VMC cookbook](docs/cookbook/quantum_vmc.md) builds a two-spin connected Hamiltonian
without materializing it.
[Metrix](docs/api/metrix/index.md) adds metric-aware differential geometry for
curvilinear PDEs, manifold PINNs, embedded geometry, and Riemannian stochastic
generators.

Neural-operator support includes canonical source/query batches, DeepONet,
N-dimensional FNO variants including the experimental-tier dealiased HOFNO, graph
and geometry-informed operators, and transformer operators. See the
[operator-learning cookbook](docs/cookbook/operator_learning.md) for execution
contracts and audited architecture comparisons, and the
[architecture API](docs/api/nn/architectures.md) for constructor details.

Free-boundary support composes implicit level-set measures, discontinuity-aware
PINN features, interface physics conditions, causal/narrow-band collocation,
explicit-front/level-set/reference-map and probabilistic Stefan workflows,
reference-map neural-operator contracts, conservative solver adapters,
interface-aware UQ, and problem-specific benchmark evidence. The
topology-preserving reference-map path is kept distinct from level-set,
phase-fraction, phase-field, particle, and complementarity paths.

Stochastic support includes reproducible SDE/SPDE path ensembles, finite-rank
spatial noise, semilinear exponential integration, convergence/error-budget
diagnostics, strong/weak/mild physics contracts, static random fields, latent
coefficient processes, and process-consistent marginal or pathwise
neural-operator rollouts. See the
[uncertainty guide](docs/guides_uncertainty.md),
[neural-operator uncertainty API](docs/api/uq/operator.md), and
[differential solver API](docs/api/solver/differential.md).

Native optimal transport covers balanced finite measures, dense and blockwise
log-domain Sinkhorn, debiased divergence, exact one-dimensional and sliced
Wasserstein distances, differentiable ordering, whole-field predictive metrics,
functional terms, semigroup consistency, and deterministic particle transforms.
See the [transport guide](docs/guides_transport.md),
[cookbook](docs/cookbook/optimal_transport.md), and
[API](docs/api/transport/index.md).

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

Dynamical-systems support separates local flow/map laws, numerical evolution,
masked trajectory data, identification, and analysis. Canonical adapters preserve
solver, control, stochastic, delay/memory, rough, and semidiscrete masks and
provenance. Sparse identification includes DMD/EDMD, multiple SINDy formulations,
selection/ensembles, exact coefficient structure, implicit SINDy, and PDE-FIND.
Nonlinear analysis includes sections, periodic orbits, Floquet spectra,
continuation and bifurcation indicators, finite-time Lyapunov spectra, covariant
directions, finite-size growth, recurrence, modified 0--1 and correlation-
dimension diagnostics, surrogates, and explicit uncertainty axes. Dense paths
retain hard guards; bifurcation and chaos results retain finite-resolution and
convergence evidence rather than claiming automatic certificates. See the
[nonlinear-dynamics cookbook](docs/cookbook/nonlinear_dynamics.md) and
[dynamics API](docs/api/dynamics.md).

Time integration preserves explicit, additive IMEX, residual, second-order,
partitioned, stochastic, and geometric equation forms. Diffrax methods coexist with
native SSPRK, endpoint theta, BDF1--BDF5, matrix-free Rosenbrock-W,
generalized-alpha, multirate partitioned RK, Gauss--Legendre collocation, geometric,
and exponential methods under explicit capability and provenance contracts. See the
[time-integrator API](docs/api/solver/time_integrators.md).

Controlled-dynamics support includes explicit causal or offline differentiable
driving paths, Diffrax-backed CDE integration, Neural CDE training, and
probabilistic numerical ODE solutions. A probabilistic ODE solution quantifies
numerical integration uncertainty; it is not a posterior over an unknown
physical model. See the
[controlled-dynamics cookbook](docs/cookbook/controlled_dynamics.md) and
[differential solver API](docs/api/solver/differential.md).

Control support includes linear and differential dynamics, control
parameterizations, sampled costs and constraints, linearization, Lyapunov and
Riccati equations, Gramians, frequency response, LQR/iLQR, dense multiple
shooting, controlled-DAE direct collocation with exact sparse derivatives,
dense or structural-sparse prepared linear-control QPs, explicit receding-horizon
warm-start shifting, and affine stage/terminal SOCP constraints. Direct
collocation supports fixed or variable duration, shared optimized parameters,
bound-form trajectory constraints, explicit native-dense or sparse-Ipopt
selection, typed callback/work evidence, KKT recertification, per-interval defect
audits, nested h-refinement with primal transfer, and controlled-DAE causal
replay. Native and Ipopt qualification artifacts retain analytic, active-path,
shared-parameter, stiff, unstable, and nonholonomic cases. Sampled nonlinear
constraints and off-grid audits are not continuous-time certificates; replay
does not rewrite collocation status; iLQR and multiple shooting accept one
physical case; bounded coefficient search is not globally optimal; dense guards
and solver status are explicit rather than hidden behind fallback or repair. See
the [control cookbook](docs/cookbook/control.md), [control API](docs/api/control.md),
and [mathematical-programming API](docs/api/optim.md).

Sensitivity utilities add score/Fisher actions and empirical
controllability/observability directions. Stationary linear-Gaussian state,
output, and cross spectra reuse diagnosed control resolvents and reject unstable,
singular, non-Hermitian, or non-positive-semidefinite inputs instead of clipping
or repairing them.

General nonlinear optimization uses one typed problem/result model across
matrix-free Newton methods, nonlinear and composite residual-plus-signed-scalar
least squares, proximal objectives, bounds and general constraints, state/design
systems, and stochastic risks. Curvature methods reuse prepared symbolic
linear-solve templates and diagnose numeric refreshes. Converged scalar,
least-squares, and strictly complementary constrained solutions expose implicit
derivatives without unrolling optimizer iterations.

Nonlinear algebraic systems have a separate `phydrax.nonlinear` contract for
matrix-free Newton roots, fixed-point acceleration, nonlinear preconditioning,
full-approximation multigrid, complementarity, and implicit root derivatives.
Generic parameterized curves and local bifurcation workflows live in
`phydrax.continuation`, with exact coordinate targets, complete nonlinear correctors,
metric-aware pseudo-arclength traversal, public-to-real complex/algebra coordinates,
stability evidence, full-augmented event localization, branch switching, and
fold/Hopf/pitchfork certification. Failed solves, singular derivative systems,
capability boundaries, and ambiguous certificates remain explicit. See the
[optimization API](docs/api/optim.md), [nonlinear systems API](docs/api/nonlinear.md),
and [continuation API](docs/api/continuation.md).

Canonical LPs, QPs, and product-cone programs live in `phydrax.optim`. They expose
native bounds, typed solver/differentiation policies, reusable numeric refresh,
warm-start contracts, independently audited KKT residuals and
infeasibility/recession rays, and explicit status/provenance. The native dense
LP/QP method remains the default. QPax 0.1.4 implicit differentiation, MPAX 0.2.4
first-order LP/QP execution, and Clarabel 0.11.1 host conic execution are selected
explicitly; none is a hidden fallback.

Optional MPAX, Clarabel, PETSc KSP/SNES, SLEPc EPS, PyAMGCL, and NVIDIA AmgX
execution lives behind the explicit lazy `phydrax.backends` boundary. Provider
availability, assembled versus matrix-free support, numeric refresh, convergence
evidence, transfers, and resource release remain visible; native planning never
selects an external provider or silently falls back to one. See the
[advanced solver cookbook](docs/cookbook/advanced_solvers.md),
[external backend API](docs/api/backends.md), and
[continuation API](docs/api/continuation.md).

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
