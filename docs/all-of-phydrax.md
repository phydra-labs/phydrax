# All of Phydrax

This page provides a high-level map of the library, how the parts fit together,
and where to look for specific functionality.

## Unifying formalism: minimizing functionals over domains

Phydrax is designed to make a single idea modular:

> Define fields on labeled domains and minimize scalar **functionals** built from operators and
> measures over domain components.

A training functional may combine three distinct kinds of scalar term:

$$
\mathcal J[u] = \sum_i \ell_i[u] + \sum_j \mathcal F_j[u] + \sum_k r_k(\theta).
$$

Here $\ell_i$ are nonnegative residual/data constraint penalties, $\mathcal F_j$ are
raw scalar functionals such as signed energies, and $r_k$ are model-level losses.
Domain components and their induced measures provide the sampling and integration
semantics for both constraints and raw integral objectives.

## The compositional contract

At a practical level, most workflows look like:

1) choose a **domain** \(\Omega\) and a **component** \(\Omega_{\text{comp}}\subseteq\Omega\),  
2) define one or more **fields** \(u_\theta:\Omega\to\mathbb{R}^m\) as `DomainFunction`s,  
3) build **residual operators** \(r=\mathcal{N}(u_\theta,\dots)\) using `phydrax.operators`,  
4) turn residuals into **constraint terms** \(\ell_i\), or define signed
   **objective terms** \(\mathcal F_j\) such as energies,
5) sum constraints, raw objectives, and optional model losses into
   \(\mathcal J\) and optimize with `FunctionalSolver`.

Two design choices make this interoperable:

- **Labeled product domains**: every coordinate is a named factor (`"x"`, `"t"`, `"data"`, `"p"`, …).
- **Structured batches**: sampling preserves axis semantics (paired sampling and coord-separable grids).

## Key choice points (what makes workflows differ)

### Sampling: point batches vs coord-separable grids

Phydrax supports two complementary evaluation regimes:

- `PointsBatch` (paired sampling): typical PINN-style collocation constraints. See [Guides → Domains and sampling](guides_domain.md).
- `CoordSeparableBatch` (axis/grid sampling): spectral/basis operators and neural operators (FNO/DeepONet). See [Guides → Differential operators](guides_differential.md).

Sampling owns sites and measure metadata; interpolation owns deterministic
reconstruction of stored values at query sites. Low-level source-to-target
execution is shared through `phydrax.sparse`: arbitrary edge relations,
fixed-width case-local rows, robust masked routing, target reduction, and
weighted forward/transpose/adjoint actions. Interpolation stencils,
`QueryNeighborhood`, `GraphIR`, and cochain incidences retain their own
geometry, topology, support, and measure semantics above that substrate.
Fourier evaluation, sparse Smolyak approximation, sparse Gaussian processes,
and stochastic estimators remain specialized methods rather than sparse
storage types. See [API → Operators → Interpolation](api/operators/interpolation.md).

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

### Uncertainty: stochastic functions, processes, inputs, and observations

`phydrax.uq` keeps epistemic, uncertain-input, observation, stochastic-process,
and numerical axes explicit in named `PredictiveField` results. NUTS/HMC, Laplace
approximation, deep ensembles, and Gaussian-process discrepancy models produce
coherent epistemic draws; probability domains, static random fields, and joint
QMC propagate uncertain inputs. Global Wiener, Poisson-clock, composite, and
coefficient-process realizations provide replayable process paths.
Complete-field Gaussian or conditional-flow operators define transition
marginals; typed Wiener/jump operator adapters define pathwise or composite
process transitions without pretending that independent marginal draws share
a path. Process diagnostics, calibration reports, shift matrices, and
retention gates keep raw results, statistical uncertainty, and provenance
explicit. See
[Guides → Uncertainty quantification](guides_uncertainty.md).

### Geometry: Euclidean coordinates vs metric-aware calculus

`phydrax.metrix` supplies explicit charts, tensor transformation laws,
positive-definite metric fields, Levi-Civita operators, curvature, embedded
charts, and metric-aware stochastic generators. Use it when a PDE, PINN, or
operator is posed in curvilinear coordinates or on a parameterized manifold.
Bounds, seams, sampling, and admissibility remain domain concerns; metric volume
can be attached to a component with `with_riemannian_measure`. See
[API → Metrix](api/metrix/index.md).

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

For row-indexed trajectories with a shared time step but different sequence
lengths, use `TrajectoryDatasetDomain` and `RaggedTimeSeriesDataConstraint`. This
keeps each sampled time tied to the dataset row that owns it while still allowing
time residuals and other `DomainFunction` operators.

When a row has static covariates and observed ragged signals, keep those semantics
separate: put the static covariates in the `TrajectoryDatasetDomain` input row,
expose measured signals with `TrajectorySignal`, and supervise row-level targets
with `TrajectoryCaseDataConstraint`. Observed trajectory signals and domain arrays
are JAX-traceable fixed state, not solver parameters.

If trajectory data must be exact, use `enforce_ragged_time_series` to build a hard
ansatz and train only the remaining physics constraints. Linear interpolation covers
first-order time residuals; cubic-Hermite interpolation covers second-order time
residuals and optional selected output components.

## Notation

We use $x$ for spatial variables, $t$ for time, $q$ for configuration, $v$ for
velocity, and $p$ for canonical momentum. $\mathcal J$ denotes the full optimized
functional, $\mathcal F$ a raw scalar objective, $L(q,v,t)$ a Lagrangian density,
$\mathcal S$ an action, and $H(q,p,t)$ a Hamiltonian.

## By task: “what do I compose?”

Below are the common SciML regimes expressed in Phydrax’s primitives.

- **Forward PDE solve (PINN-style)**: interior residual + boundary/initial terms (soft or enforced).
  Start at [Getting started](index.md) and then [Guides → Constraints](guides_constraints.md).
- **Enforced BC/IC**: build ansätze with `enforce_dirichlet` / `enforce_initial` / etc., and stage them via solver pipelines.
  See [API → Solver → Enforced constraint pipelines](api/solver/enforced_constraints.md).
- **Data assimilation / hybrid physics-data**: add `DiscreteInteriorDataConstraint`, `DiscreteTimeDataConstraint`, `SupervisedDatasetConstraint`, `RaggedTimeSeriesDataConstraint`, or `TrajectoryCaseDataConstraint` alongside PDE residuals. Use `TrajectorySignal` for fixed measured forcings/covariates on ragged trajectory domains, and `eval_constraints` for held-out data diagnostics.
  See [API → Constraints → Discrete](api/constraints/discrete.md).
- **Inverse problems (unknown coefficients/parameters)**: represent unknowns as additional fields or domain parameters, and couple them in residual operators.
  See [API → Domain → Functions](api/domain/functions.md) and [API → Constraints](api/constraints/index.md).
- **Operator learning**: use `DatasetDomain` and structured models on \(\Omega_{\text{data}}\times\Omega_x\). The canonical `OperatorBatch` path supports independent source/query discretizations across DeepONet, graph, geometry-informed, transformer, and spectral families; validate architecture choices with the audited benchmark protocol.
  See [Operator-learning cookbook](cookbook/operator_learning.md) and [API → NN → Architectures](api/nn/architectures.md).
- **Stochastic neural operators**: declare state, duration, optional source-time,
  typed drivers, query, and output roles with `OperatorTransitionSpec`. Adapt a
  process-valued probabilistic operator with `OperatorMarginalTransition`, an
  additive Wiener operator with `OperatorPathwiseTransition`, a jump-conditioned
  operator with `OperatorJumpTransition`, or a mixed-driver operator with
  `OperatorProcessTransition`. Their rollouts produce canonical
  `StochasticTrajectory` and `PredictiveField` results with physical cases,
  process realizations, time, geometry, and provenance kept separate. Train or
  diagnose adjacent likelihood, direct-horizon likelihood, semigroup, cocycle,
  weak-generator, and nonlocal jump-generator contracts independently.
  See [API → UQ → Neural-operator uncertainty](api/uq/operator.md#process-consistent-operator-transitions).
- **Integral / conservation laws**: build terms from `integral`/`mean` and use integral constraints (equality targets, flux balances, etc.).
  See [Guides → Integrals and measures](guides_integrals.md).
- **ODEs, SDEs, Lévy/rough/memory equations, interacting particles, and
  semidiscrete SPDEs**: either learn a trajectory by enforcing
  \(\dot u-f(u,t)=0\), or integrate an explicit finite-dimensional problem.
  Brownian, Poisson, stable Lévy, and fractional Gaussian realizations own
  replayable global randomness rather than acting as local seed wrappers.
  Native solvers cover Itô/Stratonovich SDEs, finite-activity jump and hybrid
  systems, truncated or Gaussian-closed stable Lévy equations, step-two rough
  equations, stochastic Volterra and delay equations, and empirical
  McKean--Vlasov particles with idiosyncratic and common noise. Spatial systems
  combine a tensor or spectral discretization with finite-rank noise.
  Semilinear splits support exact compatible modal convolution, exponential
  Euler, and commutative exponential Milstein; general systems retain the
  Diffrax backend. Stochastic collocation provides a separate deterministic
  quadrature path for finite-dimensional random inputs.
  See [API → Solver → Differential equations](api/solver/differential.md).
- **Coupled estimation and rare events**: declare refinement axes and
  coarse/fine transfers in a `StochasticHierarchy`, run paired levels with one
  realization, and allocate multilevel Monte Carlo work from measured
  correction variance and cost. Estimator state, checkpoints, and result
  archives preserve hierarchy and sampler identities. Canonical path events
  drive stopping diagnostics and adaptive multilevel splitting; Girsanov and
  jump compensator changes expose explicit path weights. A Smolyak surrogate can
  enter the same hierarchy as a paired control level.
  See [API → Integration](api/integration.md) and
  [API → Stochastic processes](api/stochastic/index.md).
- **Martingale and stopping-time validation**: declare observables and generator
  actions with `MartingaleProblem`, then evaluate interval or stopped
  martingale increments, predictable brackets, quadratic variation, and
  finite-activity jump compensators. Statistical reports use realization
  independence clusters rather than treating coupled paths as independent.
  See [API → Stochastic → Martingales](api/stochastic/martingales.md).
- **Filtering and smoothing**: compose a state prior, transition kernel,
  observation model, and masked schedule in `StateSpaceProblem`. Exact
  finite-state and linear-Gaussian marginal likelihoods avoid latent-path
  sampling. Bootstrap, guided, and Rao--Blackwellized particle filters,
  conditional SMC, fixed-lag and fixed-interval smoothers, particle Gibbs,
  particle marginal Metropolis--Hastings, and the high-dimensional ensemble
  transform filter all preserve the same status, ancestry, and process
  provenance contracts. Identifiability reports separate observation rank from
  posterior contraction.
  See [Filtering cookbook](cookbook/filtering.md) and
  [API → UQ → Filtering](api/uq/filtering.md).
- **Backward stochastic equations and semilinear high-dimensional PDEs**:
  evaluate terminal, local, and global BSDE residuals with explicit or
  autodifferentiated controls; fit one time-conditioned field from trajectory-node
  or query-conditioned Feynman--Kac labels; or alternate frozen labels and global
  optimization with Deep Picard iteration. Label batches retain conditional Monte
  Carlo errors and path-dependence clusters. Masked regularized least-squares,
  finite-activity compensated jumps, reflected path-dependent obstacles, empirical
  mean-field Hamiltonian control, and structured matrix-free nonlinear Picard
  sources have distinct declared contracts and diagnostics.
  `tools/high_dimensional_pde_benchmarks.py --suite methods` exercises the public
  query-conditioned label path and, with `--include-training`, the global Deep Picard
  training path. Its common result schema separates value/control, global-field,
  terminal, and estimator errors instead of treating unlike targets as one metric.
  See [BSDE cookbook](cookbook/bsde.md) and
  [API → Stochastic → BSDE](api/stochastic/bsde.md).
- **Static random fields and stochastic coefficient processes**: synthesize
  replayable Gaussian fields from a `SpatialNoiseBasis`, attach an explicit
  input role, and use stable mode IDs for deliberate cross-resolution coupling.
  `LatentGaussianCoefficientProcess` supplies reusable pathwise realizations;
  `LatentFlowJAXCoefficientProcess` supplies learned marginal transition laws.
  See [Guides → Uncertainty quantification](guides_uncertainty.md).
- **Curvilinear or manifold PDE/PINN**: define a `CoordinateChart` and
  `RiemannianMetric`, then use `riemannian_grad`, `riemannian_div`,
  `covariant_hessian`, or the metric overload of `laplace_beltrami`. Attach
  `sqrt(det(g))` to component integration with `with_riemannian_measure`.
  See [API → Metrix](api/metrix/index.md).
- **Stochastic PINNs, randomized residuals, and density equations**: use
  `ContinuousKolmogorovConstraint` for stationary or backward equations and
  `ContinuousFokkerPlanckConstraint` for stationary or forward density equations.
  Exact factor-HVP contractions avoid dense Hessians. When exact coordinate sums are
  still too expensive, raw Hutchinson probes or unbiased coordinate sampling expose
  estimator uncertainty to signed U-statistic, independent-product, or biased
  plug-in residual objectives. PDE-IR compilation statically rejects nonlinear
  combinations that would bias randomized intermediates.

  For high-dimensional density evolution with simulable particles,
  `trajectory_state_time_samples` plus `ScoreMatchingObjective` learns
  \(\nabla_x\log p_t(x)\) without representing or normalizing \(p_t\). This produces
  a score field, not a reconstructed density. Probability-flux boundaries, strong,
  weak, and mild SPDE solution concepts remain separate explicit contracts.
  See [Stochastic-dynamics cookbook](cookbook/stochastic_dynamics.md),
  [API → Objectives](api/objectives.md), and
  [API → Operators → Differential](api/operators/differential.md).
- **Uncertainty quantification**: use NUTS/HMC or Laplace for explicit posterior problems, ensembles for neural-model epistemic variation, Gaussian processes for model discrepancy, joint QMC for uncertain inputs, likelihoods/proper scores for observations, and conformal calibration for coverage.
  See [Guides → Uncertainty quantification](guides_uncertainty.md) and [API → Uncertainty quantification](api/uq/index.md).
- **Lagrangian/Hamiltonian mechanics**: build Euler–Lagrange, canonical Hamiltonian,
  Poisson-bracket, or Hamilton–Jacobi operators on labeled state spaces.
  See [Guides → Lagrangian and Hamiltonian mechanics](guides_mechanics.md).
- **Quantum systems and dynamics**: construct composite states, local operators,
  reduced densities, information measures, matrix commutators, and closed- or
  open-system residuals. Complex residual penalties remain real and nonnegative.
  See [Guides → Quantum operators and dynamics](guides_quantum.md),
  [Cookbook → Composite systems and a Bell state](cookbook/quantum_composite.md), and
  [Cookbook → Open-system amplitude damping](cookbook/quantum_open_system.md).
- **Ritz/energy minimization**: use `IntegralFunctional` for the raw signed energy,
  with essential boundary conditions enforced in the ansatz.
  See [Cookbook → Mechanics and Deep Ritz](cookbook/mechanics.md).
- **Stochastic path expectation**: use Euclidean bridge kernels for imaginary-time
  propagation or Feynman–Kac diffusion paths for terminal PDE and reliability quantities.
  See [Euclidean path integrals and Feynman–Kac expectations](guides_path_integrals.md).
- **Cookbook recipes**: end-to-end patterns for Poisson, deterministic and
  stochastic heat/reaction--diffusion, stochastic PINNs, inverse+data, operator
  learning, mechanics, and quantum dynamics.
  Start at [Cookbook → Overview](cookbook/index.md).

## Where to go next

- [Cookbook](cookbook/index.md)
- [Domains and sampling](guides_domain.md)
- [Differential operators](guides_differential.md)
- [Metrix: differentiable geometry](api/metrix/index.md)
- [Integrals and measures](guides_integrals.md)
- [Euclidean path integrals and Feynman–Kac expectations](guides_path_integrals.md)
- [Lagrangian and Hamiltonian mechanics](guides_mechanics.md)
- [Quantum operators and dynamics](guides_quantum.md)
- [Constraints and objectives](guides_constraints.md)
- [Uncertainty quantification](guides_uncertainty.md)
- [Solvers and training](guides_solver.md)
- [API reference](api/phydrax.md)
- `phydrax.domain` for geometry, time, and sampling.
- `phydrax.sampling` for typed reference designs and capability inspection.
- `phydrax.sparse` for JAX-native relations, routing kernels, and sparse linear actions.
- `phydrax.metrix` for charts, tensors, metrics, curvature, and stochastic geometry.
- `phydrax.data_utils` for CSV loading, array scaling, and case-index splits.
- `phydrax.constraints` for loss terms and enforced constraints.
- `phydrax.objectives` for raw signed scalar objectives.
- `phydrax.operators` for PDE operators.
- `phydrax.nn` for models and wrappers.
- `phydrax.stochastic` for Wiener and Poisson paths, composite realizations,
  stochastic trajectories, random fields, coefficient processes, coupling, and
  path/marginal transition contracts.
- `phydrax.solver` for training and evaluation loops.
