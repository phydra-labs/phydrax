# All of Phydrax

This page provides a high-level map of the library, how the parts fit together,
and where to look for specific functionality.

## Unifying formalism: minimizing functionals over domains

Phydrax is designed to make a single idea modular:

> Define fields on labeled domains and minimize scalar **functionals** built from operators and
> measures over domain components.

A training functional combines nonnegative penalty terms and model-level losses:

$$
\mathcal J[u] = \sum_i \ell_i[u] + \sum_k r_k(\theta).
$$

Each \(\ell_i\) pairs a residual, moment, or observation condition with an
explicit numerical integration source. Domain components and their induced
measures define the semantics of those sources.

## The compositional contract

At a practical level, most workflows look like:

1) choose a **domain** \(\Omega\) and a **component** \(\Omega_{\text{comp}}\subseteq\Omega\),  
2) define one or more **fields** \(u_\theta:\Omega\to\mathbb{R}^m\) as `DomainFunction`s,  
3) build **residual operators** \(r=\mathcal{N}(u_\theta,\dots)\) using `phydrax.operators`,  
4) declare residual, moment, or observation **conditions**, then pair them with
   explicit integration sources and **penalty terms**,
5) sum terms and optional model losses into \(\mathcal J\) and optimize with
   `FunctionalSolver`.

Two design choices make this interoperable:

- **Labeled product domains**: every coordinate is a named factor (`"x"`, `"t"`, `"data"`, `"p"`, …).
- **Structured batches**: sampling preserves axis semantics (paired sampling and coord-separable grids).

## Key choice points (what makes workflows differ)

### Sampling: point batches vs axis-based grids

PhydraX exposes two typed plans and matching batch schemas:

- `PointSampling` → `PointBatch` for paired collocation and scattered data;
- `GridSampling` → `GridBatch` for basis/spectral operators and neural operators
  with explicit coordinate axes.

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

Finite-dimensional algebra above those storage kernels is shared through
`phydrax.linalg`: paired array/PyTree/block spaces, composable explicit and
matrix-free operators, exact/least-squares/minimum-norm problem contracts,
deterministic capability-based planning, reusable factorizations, and portable
status, diagnostics, and provenance. Standard and generalized nonsymmetric
eigenproblems use the same spaces, operators, prepared transforms, cost models,
and failure semantics; dense QZ and native restarted-Arnoldi/Krylov--Schur paths
are explicit backend choices. Provider-neutral sparse derivative plans compile
known structural patterns natively or use ASDEX once for global pattern
detection and optimized coloring. Repeated Jacobian and Hessian evaluation is
native JAX and produces ordinary sparse coordinate operators. See
[API → Linear algebra runtime](api/linalg.md) and
[API → Sparse derivatives](api/sparse_derivatives.md).

### Discretization: supports, field spaces, and formulations

`phydrax.discretization` binds labeled continuum semantics to finite topology,
geometry, DOF layouts, measures, prepared operators, transfers, and complete
approximation bundles. Tensor support is independent of finite-difference,
spectral, or collocation calculus. Global tensor spectral spaces separate mathematical
modes, physical grids, modal DOFs, dealiasing, constrained/Galerkin/tau formulations,
periodic Leray projection, channel Stokes constraints, and temporal integration.
Exact-sampling round-sphere spaces instead expose S2FFT mode layouts, physical area
measure, scalar Laplace--Beltrami actions, complete-degree noise bases, and the
prepared discretization consumed by SFNO.
Local stencil programs, structured compact line solves, periodic/bounded SBP
calculus, entropy-conservative SBP flux differencing, finite-volume MAC projection,
WENO fluxes, mapped grids, fixed-capacity AMR, and distributed halo plans compose
without treating quadrature sites, mesh entities, and field DOFs as interchangeable.
Material particles retain stable entity IDs and a physical mass measure while
current positions remain temporal state. Fixed-h conservative barotropic SPH
uses canonical unordered pairs, normalized compact kernels, energy-derived
pressure forces, dense or fail-closed cell-list execution, exact `GraphIR`
views, and the native separable-Hamiltonian solver path. First-order WCSPH adds
explicit summation/continuity density semantics, Morris physical viscosity,
energy/dissipation ledgers, and native SSPRK integration. Structured
particle-grid splatting adds measure-aware extensive deposition, intensive
reconstruction, adjoint gather, explicit boundary loss, multilinear and
degree-one through degree-three B-spline assignments, mixed entity layouts,
route moments, and fast/deterministic/compensated reductions. See
[Guide → Particle methods](guides_particle_methods.md),
[Guide → Particle-grid splatting](guides_particle_splatting.md),
[Guide → Smoothed particle hydrodynamics](guides_sph.md),
[Guide → Weakly compressible SPH](guides_wcsph.md),
[Guide → SPH boundaries](guides_sph_boundaries.md), and
[Guide → Multiphase and incompressible SPH](guides_multiphase_incompressible_sph.md).
PINNs participate through trial/residual records rather than a fabricated mesh. See
[Guide → Discretization](guides_discretization.md),
[Guide → Global spectral methods](guides_spectral_methods.md), and
[Guide → Solver substrates](guides_solver_substrates.md).

### Precision is an execution contract

Precision is attached to executable stages, not inferred from one global dtype.
Finite differences separate coefficient storage, field storage, accumulation,
certification, communication, checkpoint, and output placement. Integration
separates integrand evaluation, reduction accumulation, adaptive/statistical
decisions, and output. Neural operators retain master parameters while creating
transient compute views, preserve geometry arrays, and record contraction and FFT
behavior. Spatial noise, predictive summaries, particle filters, and linear
solvers expose their own stage vocabulary.

Every supported policy resolves to content-addressed precision evidence. Parent
computations can retain child evidence—for example, an SPDE combines its spatial
discretization and noise-basis envelopes—and persistence compatibility includes
the effective precision contract. Static resource estimates use separate dtype
item-size assumptions; they are not presented as evidence that execution occurred.
Unsupported dtype placements fail during planning or preparation rather than
silently widening, narrowing, or changing real/complex kind.

### Differentiation: AD / jets / FD / basis

Differential operators support multiple backends (`backend="ad"|"jet"|"fd"|"basis"`) and autodiff modes
(`mode="reverse"|"forward"`). For deeper math, see [Appendix → Differentiation modes](appendix/differentiation_modes.md).

### Conditions: soft penalties vs enforcement by construction

Boundary/initial conditions can be handled in two ways:

- **Soft**: declare a boundary/initial condition, give it an integration source,
  and add its penalty term to `terms`.
- **Enforced**: build an ansatz \(\tilde u=\mathcal{H}(u)\) satisfying conditions
  exactly, then train on the remaining terms.

The enforced route is staged as boundary → initial → interior data. See:

- [API reference](api/phydrax.md)
- [API → Solver](api/solver/index.md)
- [Appendix → Physics-Constrained Interpolation](appendix/physics_constrained_interpolation.md)

### Exact PDE trial spaces

Finite Trefftz fields satisfy the homogeneous Laplace, polyharmonic, or
constant-wavenumber Helmholtz equation by construction and fit only boundary
conditions. Harmonic bases use a canonical exact-rational polynomial nullspace;
Almansi and plane-wave bases extend the same certificate/audit contract. Generic
hard boundary enforcement is rejected because it need not preserve the exact PDE
space. See [Exact PDE trial spaces](guides_exact_trial_spaces.md).

### Models: fields vs operators

- **Field learning**: learn \(u_\theta(x,t,\dots)\) directly (MLPs, separable models, etc.).
- **Operator learning**: learn \(G_\theta\) mapping inputs to fields, using a dataset factor \(\Omega_{\text{data}}\) so
  the domain becomes \(\Omega_{\text{data}}\times\Omega_x\times\cdots\). See [API → Domain → Composition](api/domain/composition.md)
  and [API → NN → Architectures](api/nn/architectures.md).

Trainable coordinates and their physical geometry are separate. Reusable
positivity, interval, symmetry, packed skew symmetry, semidefinite/definite
factorization, stability, and orthogonality maps live in
`phydrax.nn.parameters`; raw arrays remain optimizer leaves and physical values
are constructed on demand. The same package owns explicit model-PyTree
selection through `ParameterSubspace`.

### Native ML: fitted array models, not mutable estimators

`phydrax.ml` covers preprocessing and composition; linear, generalized-linear,
robust, sparse, discriminant, Bayes, and calibrated supervision; decomposition;
kernel, neighbor, covariance, mixture, clustering, manifold, outlier, tree, and
ensemble methods; selection, metrics, inspection, artifacts, and audited
conversion.

An immutable recipe plus `MLBatch` produces a `FitResult` containing a
solver-frozen `AbstractArrayModel`, fit diagnostics, validity/status, the resolved
method, and a per-input `GradientContract`. Exact discrete algorithms and smooth
relaxations have separate types. Dense-only recipes reject sparse storage rather
than allocating silently. The resulting model uses the same `ModelBinding` as
neural models, so it can remain a fixed domain closure or be explicitly unwrapped
as a trainable warm start. See [Native machine learning](guides/ml.md), the
[scientific ML workflow](cookbook/native_ml.md), and the
[complete ML API](api/ml/index.md).

### Irregular sequences: invariant affine recurrence

`phydrax.nn.operator.architectures.DiagonalStateSpaceMixer` is the
input-independent diagonal continuous-time baseline.
`SelectiveStateSpaceMixer` adds input-dependent positive step scaling, injection,
and readout while preserving an affine latent recurrence. Both use exact
zero-order-hold or linear interval integration on physical schedules and share
serial/associative execution semantics. The selective model additionally accepts
declared packed-segment resets and reports extrapolation diagnostics. Capability
metadata records both implementations as research status.

### Uncertainty: stochastic functions, processes, inputs, and observations

`phydrax.uq` keeps epistemic, uncertain-input, observation, stochastic-process,
and numerical axes explicit in named `PredictiveField` results. NUTS/HMC, Laplace
approximation, deep ensembles, and Gaussian-process discrepancy models produce
coherent epistemic draws. Scalar exact/FITC and computation-aware inference,
correlated heterotopic outputs, and linear-functional value/PDE observations share
the covariance-safe `phydrax.kernels` PyTree algebra. Computation-aware scalar
factors use native dense or sparse linear actions, bounded kernel-row workspaces,
small projected Cholesky solves, and explicit resource/conditioning evidence while
retaining unresolved action directions in covariance. Exact scalar GP inference
automatically selects weight space for a lower-rank finite-feature kernel; learned
feature maps and kernel hyperparameters remain differentiable leaves. Matrix-free
JVP/VJP propagation
transports diagonal, dense, low-rank, or operator-valued covariance through
scientific maps; normalized
errors-in-variables likelihoods account jointly for uncertain predictors and
observations. Probability domains, static random fields, and joint QMC propagate
full uncertain-input distributions. Global Wiener, Poisson-clock, composite, and
coefficient-process realizations provide replayable process paths.

Regular Bernoulli, Poisson, exponential-rate, and Normal families expose typed natural
and mean coordinates, normalized laws, weighted sufficient-statistic projection, and
exact matrix-free Fisher pullbacks. Boundary maximum-likelihood estimates and invalid
support or weight inputs remain explicit statuses rather than clipped parameters.

State-space inference binds each physical case and schedule step to one canonical
`StateSpaceStepContext`. `SampledStateSpaceInput` and `BSplineStateSpaceInput`
provide case-indexed exogenous signals with explicit support, breakpoint masks,
and stable input provenance rather than untyped callback payloads.
Euler--Maruyama transition kernels reuse canonical `ContinuousSystem` and
`WienerTerm` contracts, retain singular covariance support exactly, and expose
masked irregular-trajectory quasi-likelihoods without claiming exact SDE
likelihoods. Isothermal Port-Hamiltonian dynamics add the full
state-dependent Itô fluctuation--dissipation correction and a normalized
stationary Fokker--Planck diagnostic.
Complete-field Gaussian or conditional-flow operators define transition
marginals; typed Wiener/jump operator adapters define pathwise or composite
process transitions without pretending that independent marginal draws share
a path. Process diagnostics, calibration reports, shift matrices, and
retention gates keep raw results, statistical uncertainty, and provenance
explicit. See
[Guides → Uncertainty quantification](guides_uncertainty.md).

Gaussian inference uses `GaussianFactor` rather than silently converting every
covariance to a dense matrix. Rank, factor method, regularization, validity, and
status remain explicit through conditioning, nonlinear moment transforms,
expectation-only quadrature, and continuous-discrete filtering and smoothing.
First-order, scaled-unscented, spherical-radial, Gauss--Hermite, and keyed Monte
Carlo expectations are declared approximations; they do not make nonlinear
inference exact. Dense-only paths enforce dimension guards, and covariance inputs
are never silently repaired.

The completed state-space surface also includes SING natural-gradient
variational smoothing for additive-noise latent SDEs; square-root sequential
Kalman filtering/smoothing; exact finite-state backward smoothing, Viterbi paths
and expected statistics; particle backward/full smoothing; ensemble smoothing;
Rao--Blackwellized filtering; and structural model compilation. SING retains its
Euler-discretized ELBO, expectation rule, Gaussian-chain execution method,
accepted steps, natural residuals, coherent path samples, and explicit
unsupported-model boundary. Physical cases, schedule masks, state/process
ancestry, stable IDs, validity/status, and input/method/backend provenance remain
present in results. Square-root Kalman execution does not support the parallel
method. Discrete particle ancestry and resampling choices are nondifferentiable.


Certified multidimensional cubature extends the same measure contract: private
polynomial data owns positive simplex, spherical, periodic, and radial reference
rules, while integration plans and geometry-owned cubature atlases map those
rules to physical targets. Reference exactness, physical Jacobians, content
identity, resource limits, and paired adaptive-triangle error evidence remain
distinct rather than being collapsed into a generic quadrature flag.

### Moment calibration and target-aware finite measures

`phydrax.weighting` computes the minimum-relative-entropy reweighting of a finite
prior subject to exact feature expectations, or reconciles uncertain and
unreachable expectations with a diagonal quadratic discrepancy. Dense,
`SparseLinearMap`, and compatible matrix-free moment actions share one moment-space
geometry path with explicit affine-rank, target-residual, regularity, optimizer, KL,
effective-sample-size, support, and provenance evidence. Exact success means a
finite regular relative-interior solution; boundary and inconsistent targets remain
typed failures.

`phydrax.integration.calibrate` applies that contract to a reusable realized
measure while preserving physical mass, axes, masks, ancestry, support validity,
and execution key. Calibration and coreset compression compose as an ordered
transformation chain, allowing correction before support reduction without
conflating feature preservation with a general integration-error bound. See
[API → Moment weighting](api/weighting.md) and
[Guides → Integrals and measures](guides_integrals.md#calibrate-a-reusable-finite-realization).

### Optimal transport: geometry between finite measures

`phydrax.transport` lowers integration-native finite targets and explicit realizations
while retaining physical mass, active support, event encoding, ground cost, and
provenance. Stabilized dense and blockwise balanced and unbalanced Sinkhorn solvers
return potentials, objective components, residuals, status, and matrix-free plan
actions. The unbalanced family declares independent source and target marginal KL
penalties and represents transported-mass collapse explicitly instead of silently
normalizing unequal mass. Debiased divergences, exact and sliced Wasserstein distances,
soft order operations, prepared references, spatial/intensity UQ metrics, scalar
terms, distributional semigroup losses, and deterministic particle transforms reuse
the native substrate. See [Guides → Optimal transport](guides_transport.md).

### Dynamical systems, identification, nonlinear analysis, and chaos

`phydrax.dynamics` separates local system laws, pathwise numerical evolution,
masked trajectory data, identification, and nonlinear analysis. `StateLayout`
retains physical shape, labels, and state geometry; `ContinuousSystem` and
`DiscreteSystem` retain optional typed inputs; `TimeGrid` and `IterationGrid`
keep physical-time and map-iteration normalization distinct. Solver, control,
stochastic, memory/delay, rough, and canonical evolution outputs enter the same
`TrajectoryData` contract through explicit adapters without losing masks, reset
boundaries, case/realization axes, or provenance.
`DifferentialAlgebraicSystem` adds a state-shaped implicit residual with declared
differential/algebraic component roles and independent state, rate, and residual
scales. Its prepared BDF solver supports fixed and adaptive accepted grids,
consistent segmented continuation, frozen-grid implicit replay derivatives,
checkpointed reverse passes, guarded numerical reuse, and local regularity evidence
without introducing a second identification representation.
Array models bind into `ContinuousSystem` as explicit trainable PyTree children.
Structured Port-Hamiltonian fields provide state-dependent energy,
interconnection, dissipation, control, and forcing components while preserving
exact skew and semidefinite geometry; solver-owned isothermal dynamics add the
matching thermal diffusion without creating a second dynamics hierarchy.

Identification includes mask-safe DMD/DMDc and EDMD; strong, discrete, integral,
and weak SINDy; polynomial, Fourier, tensor-product, transformed, symmetry, and
custom feature libraries; STLSQ, SR3, temporally embargoed selection, and
ensembles; exact coefficient groups and equalities; implicit SINDy; and
structured-grid PDE-FIND. Coefficients are returned in named physical feature
coordinates. Ambient map conversion rejects non-Euclidean states unless a
geometry-aware identification method is declared.

Nonlinear analysis includes section crossings and return maps, multiple-shooting
periodic orbits, dense or matrix-free monodromy/Floquet analysis, resumable
finite-time Lyapunov spectra, covariant or adjoint directions, finite-size growth,
RQA, the modified 0--1 test, correlation dimension, surrogate significance,
explicit uncertainty-source aggregation, and a matrix-free shadowing-candidate
boundary. Bifurcation flags and statistical diagnostics are finite-resolution
evidence, not automatic certificates.

See [Nonlinear-dynamics cookbook](cookbook/nonlinear_dynamics.md) and
[API → Dynamical systems, identification, and chaos](api/dynamics.md).

### Controlled dynamics, estimation, and optimization

Differentiable driving-path classes and `solve_diffrax_cde` cover controlled
differential equations; `NeuralCDEVectorField` and `train_neural_cde` provide the
corresponding learned vector-field workflow. Path interpolation is explicit, so
causal, offline, piecewise-linear, and B-spline approximations are not conflated.
`solve_probabilistic_ode` returns calibrated Gaussian numerical uncertainty with
declared factorization, update, status, validity, and method provenance; it is a
probabilistic numerical ODE solver, not posterior uncertainty about an unknown
physical model.

`phydrax.control` composes typed time grids, control parameterizations, dynamics,
costs, and sampled constraints into trajectories with stable control,
discretization, approximation, method, and backend IDs. It includes
linearization and frequency response, Lyapunov/Riccati equations, Gramians,
finite- and infinite-horizon LQR, iLQR, dense multiple shooting, implicit
direct collocation, dense or structural-sparse prepared linear-control QPs,
explicit MPC warm-start shifting, and affine stage/terminal SOCP constraints.
Direct collocation accepts explicit systems or controlled state-shaped DAEs,
shared parameter coordinates, fixed or variable duration, exact sparse
derivatives, and explicitly selected dense-native or sparse-Ipopt optimization.
Typed Ipopt evidence retains callback work, topology, status, and warm starts;
per-interval sampled defects drive nested h-refinement with primal-only transfer;
and controlled-DAE replay binds held controls into consistency and every implicit
stage. Fingerprinted native/Ipopt campaigns derive graduation claims. Sampled
nonlinear path constraints and off-grid audits are not continuous-time
certificates, and replay does not rewrite collocation success. iLQR and multiple
shooting solve one physical case per call. Coefficient search is bounded
initialization, not a globally optimal solver. Dense algorithms enforce dimension
guards; no failed solve is hidden by fallback, projection, covariance repair, or
undeclared regularization.

Canonical LPs, QPs, and zero/nonnegative/SOC/rotated-SOC/PSD/exponential/power
product-cone programs live in `phydrax.optim`. PSD uses scaled upper-column symmetric
coordinates; EXP and POW use safeguarded JAX-native projectors and Moreau duals. Native
bounds remain separate from user constraint axes; typed methods, reusable
plan/prepare/bind/refresh lifecycles, strict warm starts, independent KKT/ray audits,
status, provenance, and regular projection-KKT sensitivities share one contract. Native
dense, QPax 0.1.4, optional MPAX 0.2.4, and optional Clarabel 0.11.1 methods remain
explicit choices with no automatic fallback or universal differentiability claim.

General nonlinear optimization lives in `phydrax.optim`. Scalar, block-residual,
proximal-composite, constrained, state/design, stochastic, manifold, and factor
graph problems share typed termination, status, diagnostics, provenance, and
PyTree contracts. Native methods include matrix-free Newton--Krylov and
Steihaug--Toint, dense/subspace dogleg and dogbox, robust-loss GN/LM,
trust-reflective bounds, variable projection, POUNDERS, projected quasi-Newton,
augmented Lagrangian, filter/SOC SQP with BFGS/SR1/exact Hessians, and a filter
interior-point method with restoration and KKT inertia planning. BOBYQA, COBYQA,
deterministic multistart, and explicitly recertified SciPy/NLopt/Ipopt/Ceres
boundaries cover black-box and specialist routes. Residual graphs retain block
sparsity, Schur ordering, manifold retractions, and incremental factor versions.

Nonlinear algebraic systems live in `phydrax.nonlinear`. Certified scalar
bracketing, Newton/trust methods, chord and limited-memory Broyden, DF-SANE,
pseudo-transient continuation, vector Halley, Type-I/II Anderson, Steffensen,
and deterministic robust attempt graphs all terminate on the original physical
residual. Finite nonlinear updates carry traced nested budgets and fixed
component evidence; failed multiplicative components truly short-circuit.
Semismooth complementarity offers infeasible and projected feasible searches.
Scaling, model/direction/certificate precision, batched small kernels, explicit
sharding, and first/second-order solution maps remain declared rather than
inferred.

Generic parameterized residual curves and local bifurcation workflows live in
`phydrax.continuation`. Natural and pseudo-arclength continuation compose complete
nonlinear correctors, exact coordinate targets, explicit state/residual geometry,
canonical real-coordinate maps, adaptive curvature rejection, full-augmented event
localization, and dense or Krylov stability analyzers. Fold, Hopf, branch-point, and
pitchfork workflows separate numerical convergence from nullspace, transversality,
spectral, symmetry, and normal-form certificates; branch switching requires certified
geometry and a validated seed.

Lyapunov spectra for flows and maps, control-theoretic Gramian actions, implicit
Lyapunov/Riccati sensitivities, state-space score/Fisher actions, empirical
controllability/observability directions, and stationary linear-Gaussian spectra
share diagnosed validity and method provenance. Stationary spectra require a
stable nonsingular resolvent and positive-semidefinite supplied spectra; inputs
are rejected rather than clipped or repaired.

### Geometry: Euclidean coordinates vs metric-aware calculus

`phydrax.metrix` supplies charts and differentiable maps; tensors and compressed
differential forms; positive and signed metrics; affine connections and curvature;
finite real algebras; Lie groups; symplectic, Poisson, horizontal, complex, and G2
structures; metric-aware stochastic kernels; and immutable, measure-orthonormal
Laplacian spectra. Exact rational multiplication tables keep commutators, Jordan
products, associators, and left/right regular actions explicit. The canonical octonion
bridge derives the seven-dimensional cross product and G2 three-form from that same
table, while derivation preparation exposes numerical infinitesimal-symmetry evidence.
Graph and cochain constructors bind spectra to explicit topology, metric, boundary,
and entity provenance. Positive norms, Lorentzian wave operators, Poisson brackets,
sub-Laplacians, and nonassociative bracket trees remain distinct named operations
rather than overloads with hidden defaults. Bounds, seams, sampling, and admissibility
remain domain concerns. See [API → Metrix](api/metrix/index.md).

For trainable arrays on spheres, hyperbolic spaces, probability simplices, matrix
manifolds, SO(n), or SPD(n), `ParameterGeometry` binds exact PyTree leaf paths to
declared metrics. Weighted product metrics, Riemannian SGD and momentum, conjugate
gradient, and L-BFGS update those leaves through tangent conversion, retraction, and
transport while ordinary leaves remain Euclidean.
See [API → Optimization](api/optim.md#riemannian-optimization).

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
        key=jr.key(0),
    )
    u = geom.Model("x")(model)

    layout = phx.domain.SampleLayout((("x",),))
    interior = geom.component()

    # Interior PDE residual: Δu - 4 = 0
    pde_condition = phx.conditions.Residual(
        "u", interior, lambda u: phx.operators.laplacian(u, var="x") - 4.0
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

### Enforced boundary conditions (replace penalties with an ansatz)

Instead of penalizing boundary violations, you can enforce \(u=g\) **by construction** and train only on the interior
PDE term. This is often numerically cleaner: terms are separate from enforcement,
which maps \(u\mapsto\tilde u\).

!!! example
    ```python
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )


    @geom.Function("x")
    def g(x):
        return x[0] ** 2 + x[1] ** 2


    model = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0)
    )
    u = geom.Model("x")(model)
    functions = {"u": u}

    layout = phx.domain.SampleLayout((("x",),))
    interior = geom.component()
    pde_condition = phx.conditions.Residual(
        "u", interior, lambda u: phx.operators.laplacian(u, var="x") - 4.0
    )
    pde_source = phx.integration.per_step(
        phx.integration.mean_over(pde_condition.on),
        phx.domain.PointSampling(64, layout=layout),
    )
    pde_term = phx.terms.ResidualPenalty(pde_condition, pde_source)

    boundary = geom.component({"x": phx.domain.Boundary()})
    boundary_condition = phx.conditions.Dirichlet("u", boundary, target=g)
    program = phx.enforcement.compile(
        functions,
        [phx.enforcement.EnforcementSpec(boundary_condition)],
        options=phx.enforcement.EnforcementOptions(num_reference=128),
        key=jr.key(1),
    )

    solver = phx.solver.FunctionalSolver(
        functions=functions, terms=[pde_term], enforcement=program
    )
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

### Adding data (anchors / sensors) is “just another term”

Phydrax treats data fit the same way as PDE residuals: an observation condition
paired with an explicit finite integration source. For scattered anchor data
\(\{(x_i,y_i)\}\), construct the point batch directly:

```python
import jax.numpy as jnp
import phydrax as phx

# Continuing from the Poisson example above:
# - geom is the geometry domain
# - u is your trainable field

anchors = jnp.array([[0.0, 0.0], [0.5, -0.5], [-0.25, 0.75]])
interior = geom.component()
batch = interior.points({"x": anchors})
data_condition = phx.conditions.Observation("u", interior, g)
data_source = phx.integration.fixed(
    phx.integration.from_samples(phx.integration.mean_over(data_condition.on), batch)
)
data_term = phx.terms.ObservationPenalty(data_condition, data_source)
```

### Operator learning (dataset × coordinates)

To model operators \(G: f \mapsto u(\cdot)\), represent the domain as a product
\(\Omega=\Omega_{\text{data}}\times\Omega_x\times\cdots\) using `DatasetDomain`, and use a structured model like
DeepONet/FNO. See [API → Domain → Composition](api/domain/composition.md) and
[API → NN → Architectures](api/nn/architectures.md).

For row-indexed trajectories with a shared time step but different sequence
lengths, use `TrajectoryDatasetDomain` and `TrajectoryCaseDataTerm`. This keeps
each sampled time tied to the dataset row that owns it while still allowing time
residuals and other `DomainFunction` operators.

When a row has static covariates and observed ragged signals, keep those semantics
separate: put the static covariates in the `TrajectoryDatasetDomain` input row,
expose measured signals with `TrajectorySignal`, and supervise row-level targets
with `TrajectoryCaseDataTerm`. Observed trajectory signals and domain arrays are
JAX-traceable fixed state, not solver parameters.

If trajectory data must be exact, use the corresponding helper in
`phx.enforcement` to build a hard ansatz and train only the remaining physics
terms. Linear interpolation covers first-order time residuals; cubic-Hermite
interpolation covers second-order time residuals and optional selected output
components.

## Notation

We use $x$ for spatial variables, $t$ for time, $q$ for configuration, $v$ for
velocity, and $p$ for canonical momentum. $\mathcal J$ denotes the full optimized
functional, $L(q,v,t)$ a Lagrangian density, \(\mathcal S\) an action, and
$H(q,p,t)$ a Hamiltonian.

## By task: “what do I compose?”

Below are the common SciML regimes expressed in Phydrax’s primitives.

- **Forward PDE solve (PINN-style)**: interior residual + boundary/initial terms (soft or enforced).
  Start at [Getting started](index.md) and continue with the conditions-and-terms guide.
- **Integral and nonlocal field learning**: compose deterministic causal,
  spatial, or fractional operators inside ordinary residuals. Use
  `RandomizedMomentPenalty` when a squared moment is resampled rather than
  silently squaring one stochastic estimate. See the
  [integral-physics cookbook](cookbook/integral_physics.md).
- **Enforced BC/IC**: declare `EnforcementSpec` values with `phx.enforcement`,
  compile them into an `EnforcementProgram`, and pass that program to the solver.
  See [API reference](api/phydrax.md).
- **Data assimilation / hybrid physics-data**: pair continuous `Observation`
  conditions with finite sources and `ObservationPenalty`; use likelihood-backed
  binary, multiclass, multilabel, or ordinal classification terms for discrete
  observations. Soft-target and focal terms remain explicit non-posterior scores.
  Dense-site, ragged-trajectory, and graph classification preserve their native
  masks/measures; Dice/Jaccard/Tversky terms aggregate one support realization before
  forming an overlap ratio. Transform one raw logit `DomainFunction` to probabilities
  or a physical order parameter inside physics operators rather than duplicating the
  trainable model in `FunctionalSolver.functions`.
  See [API reference](api/phydrax.md).
- **Inverse problems (unknown coefficients/parameters)**: represent unknowns as additional fields or domain parameters, and couple them in residual operators.
  See [API → Domain → Functions](api/domain/functions.md) and [API reference](api/phydrax.md).
- **Operator learning**: use `DatasetDomain` and structured models on \(\Omega_{\text{data}}\times\Omega_x\). The canonical `OperatorBatch` path supports independent source/query discretizations across DeepONet, graph, geometry-informed, transformer, and spectral families; validate architecture choices with the audited benchmark protocol.
  See [Operator-learning cookbook](cookbook/operator_learning.md) and [API → NN → Architectures](api/nn/architectures.md).
- **Irregular-time sequence mixing**: use `DiagonalStateSpaceMixer` for an
  input-independent stable diagonal continuous-time baseline, or
  `SelectiveStateSpaceMixer` when input-dependent step, injection, and readout
  maps are justified. Both preserve exact zero-order-hold or linear interval
  integration and serial/associative parity; the selective route also exposes
  reset-aware packed segments and time-step extrapolation diagnostics.
  See [API → NN → Architectures](api/nn/architectures.md).
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
- **Integral / conservation laws**: declare a moment condition, choose
  `mean_over(component)` or `over(component)`, and attach its source to a
  `MomentPenalty` term.
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
- **Time integration and differential-algebraic equations**: preserve explicit,
  additive IMEX, implicit residual, second-order, partitioned, stochastic, and
  geometric equation forms. Capability-checked methods include Diffrax ERK/ESDIRK/ARK,
  native SSPRK, endpoint theta, BDF1--BDF5, matrix-free Rosenbrock-W,
  generalized-alpha, partitioned RK, Gauss--Legendre IRK, and geometric/exponential
  families. Native residual solves retain consistent initialization, implicit
  derivatives, replay, continuation, local regularity, and complete attempt evidence.
  See [API → Solver → Time integrators](api/solver/time_integrators.md) and
  [API → Solver → Differential-algebraic equations](api/solver/differential_algebraic.md).
- **System identification and equation discovery**: normalize canonical
  evolution, differential/delay/memory/rough, controlled, or stochastic output
  as `TrajectoryData`; preserve sample/transition masks and reset boundaries;
  then choose DMD/EDMD, a strong/discrete/integral/weak SINDy formulation,
  structured or implicit sparse regression, or structured-grid PDE-FIND.
  Results retain design rank, conditioning, residuals, convergence history,
  physical coefficient names, source IDs, and all rejected selection or
  ensemble evidence.
  See [Nonlinear-dynamics cookbook](cookbook/nonlinear_dynamics.md) and
  [API → Dynamical systems, identification, and chaos](api/dynamics.md).
- **Nonlinear dynamics and chaos**: evolve one declared flow or map; extract
  sections and return maps; solve periodic orbits and Floquet spectra; continue
  equilibrium or orbit residuals through folds; or evaluate Lyapunov spectra,
  covariant directions, finite-size growth, recurrence statistics, the 0--1
  test, correlation dimension, and surrogate significance. Every path records
  its grid, estimator assumptions, masks, fit/Theiler/burn-in windows, RNG, and
  numerical convergence evidence. Aggregate initial-condition, parameter,
  path-noise, and numerical cases only through explicitly named uncertainty
  axes.
  See [Nonlinear-dynamics cookbook](cookbook/nonlinear_dynamics.md) and
  [API → Dynamical systems, identification, and chaos](api/dynamics.md).
- **Controlled differential equations and Neural CDEs**: select an explicit
  differentiable driving path, integrate with `solve_diffrax_cde`, or train a
  `NeuralCDEVectorField` with `train_neural_cde`. Offline cubic interpolation is
  noncausal; causal backward-Hermite, piecewise-linear, fixed B-spline, and
  callable paths declare distinct interpolation and derivative contracts.
  See [Controlled-dynamics cookbook](cookbook/controlled_dynamics.md) and
  [API → Solver → Differential equations](api/solver/differential.md).
- **Probabilistic numerical ODEs**: use `solve_probabilistic_ode` when numerical
  integration uncertainty is part of the result contract. Gaussian factors,
  calibration, step status, masks, and method/factorization provenance stay
  explicit. This numerical uncertainty is not a physical-model posterior.
  See [API → Solver → Differential equations](api/solver/differential.md).
- **Coupled estimation and rare events**: declare refinement axes and
  coarse/fine transfers in a `StochasticCouplingPlan`, run paired levels with one
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
- **Optimal control, QPs, and MPC**: compose `ControlProblem` from a typed grid,
  dynamics, parameterization, costs, and sampled constraints; use LQR/iLQR,
  compiled linear-control QPs, bounded coefficient search, dense multiple
  shooting, direct collocation, or receding-horizon MPC according to the problem
  structure. `TrajectoryOptimizationProblem` adds controlled implicit DAEs,
  bound-form global constraints, shared optimized parameters, and variable
  duration for direct collocation. Per-interval audit evidence supports explicit
  nested refinement, controlled DAEs can be causally replayed through a held input
  policy, and structured Ipopt results retain typed work/KKT/warm-start evidence.
  Results retain case/control axes, validity and backend status, plus control,
  discretization, approximation, method, and backend IDs. Nonlinear sampled
  constraints and off-grid audits are not continuous-time certificates; replay
  is independent evidence; iLQR and multiple shooting are single-case; and
  bounded search is not globally optimal. Dense paths enforce guards and never
  hide failure behind repair or fallback.
  See [Control cookbook](cookbook/control.md), [API → Control](api/control.md),
  and [API → Optimization](api/optim.md).
- **Linear systems, sensitivities, and spectra**: linearize dynamics; solve
  Lyapunov and Riccati equations; compute LQR policies, Gramian actions, frequency
  responses, flow/map Lyapunov spectra, state-space score/Fisher actions, and
  stationary linear-Gaussian spectra. Each path reports its stability,
  singularity, validity/status, regularization, and method/backend provenance.
  Dense dimension guards and explicit stability/positive-semidefinite
  requirements apply; no hidden clipping or repair is performed.
  See [API → Control](api/control.md), [API → UQ → Global sensitivity](api/uq/sensitivity.md),
  and [API → Solver → Differential equations](api/solver/differential.md).
- **Filtering and smoothing**: compose a state prior, transition kernel,
  observation model, masked schedule, and optional typed exogenous signal in
  `StateSpaceProblem`. Every transition and observation receives one
  context-last `StateSpaceStepContext`; sampled and B-spline inputs preserve
  endpoint values, breakpoint masks, internal-time evaluation, support, and
  `input_id`.

  Linear-Gaussian paths include sequential or parallel covariance-form Kalman
  filtering, sequential square-root filtering, and matching RTS smoothing.
  Square-root execution does not support the parallel method. Exact finite-state
  inference includes backward smoothing, Viterbi paths, transition counts, and
  expected sufficient statistics. Particle, ensemble, Rao--Blackwellized, and
  conditional-SMC paths include fixed-lag/full/backward smoothers and posterior
  simulation or MCMC where declared. Particle ancestry and resampling remain
  discrete and nondifferentiable.

  `GaussianFactor`, conditional moments, declared nonlinear Gaussian transforms,
  and continuous-discrete Gaussian filtering/smoothing preserve rank,
  approximation, regularization, validity/status, physical cases, schedule
  masks, stable IDs, and solver/backend provenance. Dense guards apply.

  SING adds a Gaussian information-form variational smoother for
  Euler--Maruyama latent SDEs with full-rank additive diffusion. It supports
  differentiable non-Gaussian observations, irregular masked schedules, per-case
  natural-gradient backtracking, sequential or associative Gaussian-chain
  conversion, fixed-posterior model gradients, coherent path sampling, and
  portable result export. Its objective is an ELBO, not a relabeled marginal
  likelihood; multiplicative or singular diffusion has no silent fallback.
  Nonlinear moment propagation and sampled continuous-discrete observations are
  approximations, not exact inference, and no invalid covariance is silently
  repaired. Structural local-level, trend, seasonal, autoregressive, regression,
  deterministic-transition, and process-noise components compile into the same
  state-space contract.
  See [Filtering cookbook](cookbook/filtering.md),
  [API → Stochastic → State-space models](api/stochastic/state_space.md),
  [API → UQ → Filtering](api/uq/filtering.md), and
  [API → UQ → Inference and ensembles](api/uq/inference.md).
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
- **Continuous learned transport and flow matching**: sample independent or native
  balanced endpoint couplings, construct explicit endpoint interpolants, train a
  state-shaped conditional velocity with `FlowMatchingTerm`, and advance source-law
  samples through `ContinuousTransport` and an existing `DiffraxEvolution`.
  `ContinuousFlowLaw` adds exact finite-dimensional Euclidean density evaluation;
  keyed Hutchinson log-density estimates remain separate uncertainty-bearing
  diagnostics. Fixed-query field objectives can retain masks, physical quadrature,
  and channel geometry through `OperatorFlowMatchingMetric`.
  See [Guides → Optimal transport](guides_transport.md) and
  [API → Continuous learned transport](api/transport/continuous.md).
- **Stochastic PINNs, randomized residuals, and density equations**: use
  `phx.conditions.stochastic.Kolmogorov` for stationary or backward equations
  and `phx.conditions.stochastic.FokkerPlanck` for stationary or forward density
  equations, each paired with an explicit residual-penalty source. Exact
  factor-HVP contractions avoid dense Hessians. When exact coordinate sums are
  still too expensive, raw Hutchinson probes or unbiased coordinate sampling
  expose estimator uncertainty to signed U-statistic, independent-product, or
  biased plug-in residual estimators. PDE-IR compilation statically rejects
  nonlinear combinations that would bias randomized intermediates.

  For high-dimensional density evolution with simulable particles,
  `trajectory_state_time_samples` plus `ScoreMatchingTerm` learns
  \(\nabla_x\log p_t(x)\) without representing or normalizing \(p_t\). This produces
  a score field, not a reconstructed density. Probability-flux boundaries, strong,
  weak, and mild SPDE solution concepts remain separate explicit contracts.
  See [Stochastic-dynamics cookbook](cookbook/stochastic_dynamics.md),
  [API reference](api/phydrax.md), and
  [API → Operators → Differential](api/operators/differential.md).
- **Uncertainty quantification**: use NUTS/HMC or Laplace for explicit posterior
  problems, ensembles for neural-model epistemic variation, scalar or correlated
  Gaussian processes for model discrepancy, linear-functional GPs for operator
  observations, joint QMC for uncertain inputs, proper likelihoods/scores for
  observations, and conformal calibration for coverage. Use FITC only after dense
  scaling is measured.
  See [Guides → Uncertainty quantification](guides_uncertainty.md),
  [API → Positive-definite kernels](api/kernels.md), and
  [API → Uncertainty quantification](api/uq/index.md).
- **Lagrangian/Hamiltonian mechanics**: build Euler–Lagrange, canonical Hamiltonian,
  Poisson-bracket, or Hamilton–Jacobi operators on labeled state spaces.
  See [Guides → Lagrangian and Hamiltonian mechanics](guides_mechanics.md).
- **Quantum systems and dynamics**: construct composite states, local operators,
  reduced densities, information measures, matrix commutators, and closed- or
  open-system residuals. Complex residual penalties remain real and nonnegative.
  See [Guides → Quantum operators and dynamics](guides_quantum.md),
  [Cookbook → Composite systems and a Bell state](cookbook/quantum_composite.md), and
  [Cookbook → Open-system amplitude damping](cookbook/quantum_open_system.md).
- **Ritz/energy minimization**: use an explicit integral source with the
  appropriate term, with essential boundary conditions enforced in the ansatz.
  See [Cookbook → Mechanics and Deep Ritz](cookbook/mechanics.md).
- **Stochastic path expectation**: use Euclidean bridge kernels for imaginary-time
  propagation or Feynman–Kac diffusion paths for terminal PDE and reliability quantities.
  See [Euclidean path integrals and Feynman–Kac expectations](guides_path_integrals.md).
- **Cookbook recipes**: end-to-end patterns for field and operator learning,
  stochastic dynamics, filtering and smoothing, controlled differential
  equations, probabilistic inference, optimal control, QPs/MPC, mechanics, and
  quantum dynamics.
  Start at [Cookbook → Overview](cookbook/index.md).

## Where to go next

- [Cookbook](cookbook/index.md)
- [Advanced solver workflows](cookbook/advanced_solvers.md)
- [External solver backends](api/backends.md)
- [Continuation and bifurcation](api/continuation.md)
- [Domains and sampling](guides_domain.md)
- [Discretization](guides_discretization.md)
- [Solver substrates](guides_solver_substrates.md)
- [Differential operators](guides_differential.md)
- [Linear algebra runtime](api/linalg.md)
- [Metrix: differentiable geometry](api/metrix/index.md)
- [Positive-definite kernels](api/kernels.md)
- [Integrals and measures](guides_integrals.md)
- [Special functions and named integrals](guides_special_functions.md)
- [Euclidean path integrals and Feynman–Kac expectations](guides_path_integrals.md)
- [Lagrangian and Hamiltonian mechanics](guides_mechanics.md)
- [Quantum operators and dynamics](guides_quantum.md)
- Conditions and terms
- [Uncertainty quantification](guides_uncertainty.md)
- [State-space models and transition adapters](api/stochastic/state_space.md)
- [Dynamical systems, identification, and chaos](api/dynamics.md)
- [Nonlinear dynamics and chaos cookbook](cookbook/nonlinear_dynamics.md)
- [Controlled dynamics](cookbook/controlled_dynamics.md)
- [Control workflows](cookbook/control.md)
- [Control API](api/control.md)
- [Solvers and training](guides_solver.md)
- [API reference](api/phydrax.md)
- `phydrax.domain` for geometry, time, and sampling.
- `phydrax.sampling` for typed reference designs and capability inspection.
- `phydrax.sparse` for JAX-native relations, routing kernels, and sparse linear actions.
- `phydrax.linalg` for paired vector spaces, composable operators, linear
  problems, solve policies, reusable plans and factorizations, general eigensolvers,
  diagnostics, and backend provenance.
- `phydrax.backends` for explicit lazy PETSc, SLEPc, PyAMGCL, and NVIDIA AmgX
  lifecycle bridges with availability, transfer, convergence, and provenance evidence.
- `phydrax.metrix` for charts, tensors, metrics, curvature, and stochastic geometry.
- `phydrax.data_utils` for CSV loading, array scaling, and case-index splits.
- `phydrax.conditions` for residual, moment, observation, and physical conditions.
- `phydrax.terms` for penalty and specialized numerical/data terms.
- `phydrax.integration` for targets, sources, and reductions.
- `phydrax.weighting` for exact and quadratically reconciled moment calibration.
- `phydrax.special` for JAX-native named special functions and integral primitives.
- `phydrax.enforcement` for exact condition transforms.
- `phydrax.operators` for PDE operators.
- `phydrax.nn` for models, wrappers, and the generic diagonal state-space mixer.
- `phydrax.dynamics` for typed flow/map laws, pathwise evolution, trajectory
  data, DMD/EDMD, SINDy/PDE-FIND, periodic-orbit and chaos analysis, uncertainty
  aggregation, and the shadowing solver boundary.
- `phydrax.stochastic` for process paths, trajectories, typed state-space
  problems and inputs, transition kernels, exact signature and log-signature
  features, and structural model compilation.
- `phydrax.kernels` for covariance-safe stationary, algebraic, transformed,
  finite-feature, structured-input, signature-PDE, graph/Hodge spectral, compact,
  combinatorial, and fixed-noise noncompact kernels shared by GP and coreset methods.
- `phydrax.uq` for Gaussian factors and transforms, filtering/smoothing,
  state-space estimation, sensitivities, and stochastic spectra.
- `phydrax.optim` for typed scalar, least-squares, proximal-composite, constrained,
  state/design, and stochastic optimization, differentiable solution maps,
  canonical QPs, and the explicit QPax backend.
- `phydrax.nonlinear` for nonlinear algebraic systems, fixed points,
  preconditioning, multigrid, variational inequalities, and implicit roots.
- `phydrax.continuation` for generic parameterized residual curves, stability,
  event localization, branch switching, and fold/Hopf/pitchfork workflows.
- `phydrax.control` for finite-horizon control, linear systems, LQR/iLQR,
  multiple shooting, compiled QPs, and MPC.
- `phydrax.solver` for training, differential, delay/memory, rough, stochastic,
  controlled, probabilistic, and geometry-preserving equation solvers.
