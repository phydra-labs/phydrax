# Global spectral methods

Phydrax separates a spectral basis, its physical evaluation support, modal degrees of
freedom, nonlinear realization, and temporal method. Tensor spectral objects are global
products rather than spectral elements. Spherical spaces instead bind exact S2FFT
sampling theorems to a round-sphere support; neither path invents element topology.
Local high-order tensor elements, mapped geometry, CG/DG coupling, and DGSEM
live in the finite-element compiler; see [Spectral elements](guides_spectral_elements.md).

## Spaces and representations

A basis plan owns mathematical modes. Preparing a tensor plan binds those modes to
explicit axis domains, quadrature, transforms, and exact field-space identities:

```python
import jax.numpy as jnp
import phydrax as phx
import jax.random as jr

key = jr.key(0)

space = phx.discretization.TensorSpectralPlan(
    (phx.discretization.FourierBasisPlan(128),),
    axis_names=("x",),
    field_name="u",
).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
```

`space.modal_space` has representation `"modal_coefficient"` and is the primary
state space. `space.physical_space` is the point-value evaluation space. Modes,
quadrature points, and DOFs remain separate even when a square transform gives them
the same count.

Use explicit projection and reconstruction:

```python
x = space.axes[0].nodes
values = jnp.sin(2 * jnp.pi * x)
coefficients = space.project(values)
reconstructed = space.reconstruct(coefficients)
```

Fourier fields use full complex modal storage. Real reconstruction is explicit and
`imaginary_leakage` exposes the discarded imaginary roundoff. Sine and cosine plans
encode homogeneous Dirichlet and Neumann endpoint semantics respectively. Chebyshev
and Legendre plans use the internal polynomial preparation substrate and budgeted
dense linear transforms.

## Axis domains and unbounded intervals

`AxisDomain` distinguishes bounded, periodic, half-line, and real-line support.
Endpoint inclusion belongs to the node rule rather than being inferred from
nonperiodicity: Gauss nodes include neither endpoint, Radau includes one, and
Lobatto includes both. Point-primary tensor measures use each axis's declared
quadrature weights.

Rational Chebyshev bases compactify infinity without storing nonfinite points:

```python
domain = phx.discretization.AxisDomain.real_line()
boundary = phx.discretization.SpectralBoundaryConditionPlan.decay()
basis = phx.discretization.ConstrainedBasisPlan(
    phx.discretization.RationalChebyshevLineBasisPlan(48, 4.0),
    boundary,
)
line = phx.discretization.TensorSpectralPlan(
    (basis,),
    axis_names=("x",),
).prepare((domain,))
```

The full-line map is `y = scale*t/sqrt(1-t**2)` on endpoint-free first-Fejér
nodes. The half-line plan uses `y = endpoint + scale*(1+t)/(1-t)` or its
negative-direction counterpart. Physical quadrature weights contain the map
Jacobian. Value traces at compactified infinity are exact modal functionals;
unsupported derivative traces fail during preparation.

Mapped finite-mode derivatives are projected actions. `axis.derivative_exact`
is false and `axis.derivative_residual` records an independent overresolved
closure check. The scale is a preparation identity. Compare a declared scale
set with `SpectralModalDiagnosticsPlan`; Phydrax does not silently optimize it.

## Modal transfer and resolution evidence

`prepare_spectral_modal_transfer` owns every resolution change used by
dealiasing and eigen verification. Fourier transfers split and recombine even
Nyquist modes. Polynomial transfers preserve degree identity. Constrained
transfers pass through base-modal coordinates and verify target traces rather
than assuming that numerical nullspace columns are nested.

`SpectralModalDiagnosticsPlan` reports physical tail norms, relative tail
norms, head-tail overlap, raw coefficient envelopes, local slopes, and
rounding-floor masks. A tail report is resolution evidence, not a proof that a
function belongs to one asymptotic convergence class.

General eigensolves can be compared with homogeneous chordal matching:

```python
resolution_domain = phx.discretization.AxisDomain.periodic(0.0, 1.0)
coarse = phx.discretization.TensorSpectralPlan(
    (phx.discretization.FourierBasisPlan(8),)
).prepare((resolution_domain,))
fine = phx.discretization.TensorSpectralPlan(
    (phx.discretization.FourierBasisPlan(12),)
).prepare((resolution_domain,))
coarse_result = phx.linalg.eigen.general_eigensolve(
    phx.linalg.eigen.GeneralEigenproblem(
        phx.discretization.spectral_derivative_operator(coarse, 0).operator
    )
)
fine_result = phx.linalg.eigen.general_eigensolve(
    phx.linalg.eigen.GeneralEigenproblem(
        phx.discretization.spectral_derivative_operator(fine, 0).operator
    )
)
transfer = phx.discretization.prepare_spectral_modal_transfer(coarse, fine)
evidence = phx.discretization.compare_spectral_eigen_resolutions(
    coarse_result,
    fine_result,
    coarse,
    fine,
    transfer,
)
```

Matching is one-to-one. Repeated modes are judged through transferred physical
subspaces, not through eigenvector signs, phases, or ordinal positions.

## Exact-sampling spherical spaces

`SphericalSpectralPlan` prepares scalar or spin-weighted fields on a round two-sphere.
The stable differential contract is scalar spin zero; nonzero-spin complex plans expose
analysis and synthesis without claiming scalar Laplace--Beltrami, kernel, stochastic,
or SFNO semantics.

```python
sphere = phx.discretization.SphericalSpectralPlan(
    32,
    sampling="mw",
    field_name="u",
).prepare(radius=1.0)

values = jnp.cos(sphere.transform.theta)[:, None] * jnp.ones(
    (1, sphere.transform.phi.size)
)
sphere_coefficients = sphere.project(values)
reconstructed = sphere.reconstruct(sphere_coefficients)
laplacian_values = sphere.laplacian(values)
```

The physical point-value field is the primary state. S2FFT coefficient storage has
shape `(L, 2*L-1)` and contains invalid `|m| > ell` capacity; `SphericalModeLayout`
owns the valid mask, conjugacy, degree groups, and logical mode identity. Invalid
capacity is masked before arithmetic and is not advertised as a modal field space.
`layout_id`, `transform_id`, and `execution_id` respectively distinguish coefficient
meaning, exact sampling realization, and recursive versus precomputed execution.

The physical measure sums to `4*pi*radius**2`. Scalar Laplace--Beltrami uses the
negative-semidefinite multiplier `-ell*(ell+1)/radius**2`; `eigenpairs` reports the
nonnegative spectrum of `-laplacian` and accepts only ranks ending at a complete
`2*ell+1` degree block. Explicit eigenbases and dense Laplacians are resource-bounded.

`SphericalSamplePlan` adds fixed-capacity Cartesian samples, masks, positive weights,
rank/condition budgets, and dense SVD-backed evaluate/fit. Its `healpix` constructor
generates deterministic ring or standard nested centers and equal-area weights; this is
bounded evaluation/fitting, not a fast HEALPix transform. Inactive rows are sanitized
before geometry and arithmetic.

Spin ladders apply the exact eth/ethbar multipliers in coefficient space. Coordinate
derivatives are explicitly colatitude/longitude-chart valued and return a pole-validity
mask. `SphericalRotationPlan` uses recursive Wigner-D blocks in the active ZYZ
convention, while `SphericalClebschGordanPlan` prepares only triangle- and
order-admissible product couplings. The common `SpectralModalTransferPlan` owns
spherical and lattice-harmonic zero-fill/truncation; no second transfer hierarchy is
created.

Polynomial spherical nonlinearities use `L_eval = p*(L-1)+1` through the existing
padding lifecycle. Nonpolynomial expressions require an explicit approximate modal
filter. `modal_integral` evaluates the scalar spin-zero constant coefficient directly,
and the ordinary spectral PDE compiler evolves spherical coefficients with the modal
Laplace--Beltrami multiplier. Closed S2 rejects boundary conditions and retains the
global-frame gradient/divergence nonclaim.

## Operators

Modal endomorphisms are exposed through canonical `phydrax.linalg` operators:

```python
laplacian = phx.discretization.spectral_laplacian_operator(space)
modal_rate = laplacian(coefficients)
```

Fourier derivatives are diagonal. Tensor sums retain their separable modal action.
Polynomial derivatives use prepared fixed-capacity coefficient matrices from the
internal orthogonal-polynomial substrate. Physical conveniences such as
`space.partial_derivative`, `space.gradient`, and `space.laplacian` accept and return
point values; modal evolution uses `modal_derivative` and `modal_laplacian`.
Constrained polynomial trial spaces are not generally closed under strong modal
differentiation. Their physical derivative conveniences differentiate the reconstructed
polynomial, while modal operator constructors fail explicitly rather than inventing a
diagonal closure.

The periodic Hilbert transform is an exact Fourier multiplier with
`-1j*sign(k)`. The mean and the even-grid Nyquist mode are explicitly zero:

```python
hilbert = phx.discretization.spectral_hilbert_operator(space, 0)
hilbert_coefficients = hilbert(coefficients)
```

## Implicit modal fields

`ImplicitModalField` represents a complete tensor-spectral state with one shared
coefficient model. For a `d`-axis space, the model receives
`[k_1 / s_1, ..., k_d / s_d, t]` and returns one complex coefficient (or a declared
component shape). `mode_scales` owns the explicit input normalization; spatial
derivatives remain the responsibility of the prepared spectral discretization.

```python
raw = phx.nn.models.ComplexOutputModel(
    phx.nn.models.MLP(
        in_size=2,
        out_size=2,
        width_size=64,
        depth=3,
        key=key,
    )
)
modal = phx.nn.models.ImplicitModalField(
    raw,
    space,
    real_field=True,
)
u_hat = modal.as_domain_function(
    phx.domain.ScalarInterval(0.0, 1.0, label="t")
)
```

`real_field=True` applies the canonical `HermitianSpectralCoordinates` projection,
including real self-conjugate modes. `modal.physical_values(t)` reconstructs through
the prepared transform; it never identifies point values with modal coefficients.

Train the field against the existing coefficient-resident PDE compiler rather than
reimplementing spectral derivatives or nonlinear products:

```text
physics = phx.terms.CompiledModalResidualTerm(
    compiled,
    function_name="u_hat",
    times=times,
)
initial = phx.terms.ModalObservationTerm(
    jnp.asarray([0.0]),
    initial_coefficients[None, ...],
    function_name="u_hat",
)
solver = phx.solver.FunctionalSolver(
    functions={"u_hat": u_hat},
    terms=(physics, initial),
)
```

`ModalObservationTerm` accepts coefficient-wise masks and nonnegative weights, so
known, missing, and merely uncomputed coefficients remain distinct. A masked target
may be non-finite only where its mask is false.

Two optional coefficient priors compose without changing PDE semantics:

- `ExponentialSpectralEnvelope` applies positive per-axis decay rates. `sum` is the
  tensor-product decay law; `mean` is an explicit dimension-normalized heuristic.
- `SpectralBasisModulation` evaluates exact prepared one-dimensional basis functions
  at declared coarse nodes and passes the resulting real feature vector to a model.

These priors are parameterizations, not regularity estimates or missing-mode
guarantees. Full tensor materialization is still exponential in the number of axes.
For nonlinear PDEs, `CompiledModalResidualTerm` materializes the declared state and
uses the compiler's explicit dealiasing policy. `maximum_query_points` and
`maximum_feature_bytes` fail before hidden tensor or feature-table growth exceeds the
declared resource budget.

## Entropy-stable Fourier split forms and Hermitian periodic encoding

`SpectralSplitFormPlan` is a bounded theorem: it accepts only the built-in analytic
entropy-conservative Euler two-point flux, a real all-Fourier periodic space, no source,
and no viscosity. Preparation certifies the diagonal norm, skew `Q=H D`, constants,
pair count, and pair-chunk workspace. Execution evaluates each unordered pair once and
scatters equal-and-opposite contributions; unsupported fluxes or resource excess fail
before execution. The projected-flux route remains distinct and does not inherit this
certificate.

`HermitianSpectralCoordinates` is the independent-real archive/backend chart for a
real periodic spectral field; the live ETDRK, compiled-flow, callback, and public
solution state remains full complex. A `RuntimeCheckpointLeafBinding` may apply the
chart to one indexed PyTree leaf. The checkpoint records coordinate/evidence identity
and reconstructs the exact full-complex shape and dtype; unbound leaves stay native.
Non-Hermitian states are rejected rather than clipped. Nonlinear transforms still use
full-complex work arrays, so coordinate encoding is not a halved peak-work claim.

Constant-viscosity Fourier--Chebyshev--Fourier channel plans default to the
`ultraspherical_banded` route. It is internally pressure eliminated, with fixed-band
Helmholtz/biharmonic/pressure-recovery systems and fixed-rank tau corrections, while
the public solve still returns primitive velocity, pressure, and affine pressure
gradient. The zero horizontal mode handles tangential Helmholtz solves, pressure
recovery, and either prescribed pressure gradient or a two-component bulk-flux Schur
solve; nonzero modes use wall-normal velocity/vorticity elimination before primitive
field recovery.

`dense_reference` remains a caller-explicit oracle. Banded production claims apply
only when the preparation report names `ultraspherical_banded`.
`ChannelStokesPreparationReport` exposes route, bandwidth, horizontal batch,
correction/constraint rank, operator/factor/workspace/persistent/preparation bytes,
pivot margin, and the required unsharded wall-normal axis. Variable viscosity,
failed pivots or constraints, and distributed spectral/line execution remain outside
the channel contract.

## Distributed spectral execution

`SpectralMeshTopology` binds a real caller-visible JAX `Mesh`, its shape, axis names,
platform, and device IDs. Construction never simulates unavailable devices.
`DistributedSpectralExecutionPlan` then prepares one of three fixed schedules:

- `slab`: one-dimensional mesh and all-Fourier full-complex transforms;
- `pencil`: a two-dimensional mesh, spatial rank at least three, and divisible
  canonical and padded transform dimensions;
- `channel`: rank-three Fourier–Chebyshev–Fourier layout with the Chebyshev axis 1
  replicated and only the two horizontal axes partitioned.

Preparation fixes physical/modal and padded layouts, every all-to-all transpose,
coefficient and accumulation precision, normalization, local shapes, collective count,
checkpoint/stage memory, and a hard byte ceiling. `prepare()` verifies that the named
devices remain available. `place`, `execute_transform`, `pad_modal`, `unpad_modal`,
`modal_derivative`, and `diagnostics` preserve the prepared sharding and perform no
host gather; `rotational_nonlinear` is restricted to periodic three-dimensional vector
plans. Channel execution calls a supplied modal action through `execute_channel`; it
does not implement a distributed Chebyshev transform or turn `ChannelStokesPlan` into
a distributed line solve.

```python
topology = phx.discretization.SpectralMeshTopology.one_device()
distributed = phx.discretization.DistributedSpectralExecutionPlan.from_discretization(
    topology,
    prepared_spectral_space,
    schedule="slab",
    maximum_bytes=memory_limit,
).prepare()
modal = distributed.execute_transform(
    physical_state,
    direction="physical_to_modal",
)
```

The one-device topology is the local route. A caller-supplied mesh is the actual
multi-device route; the plan does not initialize a distributed job, invent a device,
or claim scaling. A multi-host JAX deployment remains an external process-launch and
platform-evidence responsibility even when its devices participate in the mesh.

## Nonlinear evaluation and dealiasing

Nonlinear pseudospectral compilation requires an explicit policy. Quadratic Fourier
products normally use 3/2 overresolution; cubic products require 2× overresolution.

```text
method = phx.discretization.PseudospectralMethodPlan(
    dealiasing=phx.discretization.PaddingDealiasingPlan(
        maximum_polynomial_degree=2,
    ),
)
compiled = phx.equations.compile_semidiscrete_pde(
    problem,
    space,
    method,
)
```

Padding composes mode-aware embedding, a larger square transform, physical
pointwise evaluation, and mode-aware restriction. Even Fourier Nyquist modes are
split and recombined explicitly. `ModalFilterPlan` is an approximate cutoff for
nonpolynomial expressions. `NoDealiasingPlan` is an explicit acceptance of aliases;
it never reports exact nonlinear projection.

The prepared method also owns direct nonlinear actions for modal solver callbacks:

```text
prepared_method = method.prepare(
    space,
    required_polynomial_degree=2,
    nonlinear=True,
)
quadratic_coefficients = prepared_method.nonlinear_action(
    coefficients,
    lambda physical: physical**2,
)
```

This is the representation-safe path for nonlinear callbacks passed to spectral SPDE
constructors. Their initial state, reaction result, and state-shaped noise amplitudes
are modal; project initial physical data and reconstruct physical observables.

## All-coordinate spectral PDE residuals

`compile_spectral_residual` treats every declared coordinate, including time, as
one axis of a tensor spectral trial space. It evaluates selected
`PDEEquation.residual` expressions through modal derivatives and the prepared
nonlinear realization, without differentiating a neural model with respect to
query coordinates:

```text
method = phx.discretization.PseudospectralMethodPlan(
    dealiasing=phx.discretization.PolynomialClosureDealiasingPlan(2),
)
residual = phx.equations.compile_spectral_residual(
    problem,
    space,
    method,
    scope="full",
)
state = residual.project_state(predicted_values)
loss = residual.residual_energy(state, parameter_values)
```

`PolynomialClosureDealiasingPlan` differs from ordinary padding. Padding
overresolves enough to protect the retained Galerkin projection from aliases.
Polynomial closure represents the complete finite product bandwidth before the
residual is measured. `scope="retained"` deliberately measures only the
trial-space projection; `scope="full"` measures the prepared closure-space
residual. `SpectralResidualCompilationReport` records the two modal shapes,
polynomial degree, exactness, condition policy, and resource size.

The residual norm uses the prepared physical quadrature rather than assuming
that raw coefficient storage is orthonormal. A nonpolynomial field expression
has no finite exact closure: select `ModalFilterPlan` and
`require_exact=False` to request an explicitly approximate objective.
Nonlinear sine and constrained-basis closure, masked grids, per-case geometry,
and fields on different coordinate subsets are rejected.

Boundary and initial conditions are never silently omitted. Problems carrying
conditions compile only after the caller selects
`condition_handling="external"` and supplies a hard physical
`OperatorOutputPipeline`, or after those conditions have already been encoded
by the chosen basis.

For Fourier spaces, `SpatialNoiseBasis.from_spectrum` first constructs real
weighted-orthonormal Laplacian modes and then projects them into full complex
storage. Its complex modal columns preserve conjugate symmetry under real Wiener
coefficients; independent one-sided Fourier modes are never substituted.

Diffrax solves keep this public modal state complex but execute it as one coupled
real system with backend shape `(2,) + modal_shape`. The adapter applies the same
real Wiener controls to both components and reconstructs complex states before every
spectral callback, event, dense query, and saved output. Explicit native-complex and
reject policies remain available through `DiffraxComplexStatePolicy`.

The compiled state is modal. Use `compiled.project_state` for initial data and
`compiled.reconstruct_state` for observables and output. Constant-coefficient scalar
linear parts lower to `DiagonalLinearOperator`; general linearizations remain explicit
matrix-free operators.

## Periodic conservation and entropy

`ConservationProblemIR(..., boundaries=None)` declares a fully periodic problem.
A spectral conservation method differentiates projected physical fluxes and therefore
preserves the zero Fourier mode up to roundoff:

```text
method = phx.discretization.SpectralConservationMethodPlan(
    phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.ModalFilterPlan(2 / 3),
    ),
    entropy_diagnostics=True,
)
compiled = phx.equations.compile_conservation_problem(
    problem,
    space,
    method,
    entropy_pair=entropy_pair,
)
```

Physical-state, residual, and source integrals are first cast through the prepared
spectral `reduction_dtype` and then accumulated with a twofold compensated sum. The
conservation defect reduces residual and negative source contributions together
rather than subtracting two rounded integrals. This improves roundoff evidence without
changing the modal residual or claiming exact arithmetic.

The equation-owned `ConvexEntropyPair` supplies entropy, entropy variables, flux,
and admissibility. Spectral diagnostics report total entropy and its semidiscrete
rate. They do not claim entropy stability; a proven entropy-stable split form is a
separate numerical contract.

## Incompressible periodic and channel flow

`compile_periodic_incompressible_flow` prepares a velocity-only rotational
Navier–Stokes system on a two- or three-axis Fourier tensor space. The prepared
`PeriodicLerayProjector` removes longitudinal modes, assigns the pressure zero-mode
gauge, and zeros self-conjugate Nyquist modes that are incompatible with odd
real-field derivatives:

```python
space = phx.discretization.TensorSpectralPlan(
    (
        phx.discretization.FourierBasisPlan(64),
        phx.discretization.FourierBasisPlan(64),
    ),
    axis_names=("x", "y"),
    field_name="velocity",
).prepare(
    (
        phx.discretization.AxisDomain.periodic(0.0, 1.0),
        phx.discretization.AxisDomain.periodic(0.0, 1.0),
    )
)
problem = phx.equations.IncompressibleFlowProblem(2, 1e-3)
method = phx.discretization.PseudospectralMethodPlan(
    dealiasing=phx.discretization.PaddingDealiasingPlan(2),
)
compiled = phx.equations.compile_periodic_incompressible_flow(
    problem, space, method
)
```

The public and live temporal state remains full complex.
`HermitianSpectralCoordinates` provides an independent real chart for Newton,
continuation, Lyapunov, periodic-orbit analysis, and selected checkpoint leaves
without changing callback arithmetic. Paired Fourier modes use norm-preserving
real/imaginary coordinates; zero and Nyquist fixed points remain real. Coordinate
construction rejects nonintegral component dimensions and preflights its explicit
real-coordinate capacity. `TensorSpectralSymmetry` applies normalized Fourier
translations, supported reflections, and an orthogonal component action directly in
modal space. Translation/reflection composition follows the declared
semidirect-product action, including reflected translations.

Wall-bounded channel flow uses a Fourier x Chebyshev x Fourier tensor plan and a
separate constrained Stokes preparation:

```text
problem = phx.equations.IncompressibleFlowProblem(3, 1e-3)
constraint = phx.discretization.ChannelMeanConstraint(
    "pressure_gradient", (1.0, 0.0)
)
stokes = phx.discretization.ChannelStokesPlan(
    channel_space,
    viscosity=problem.viscosity,
    mean_constraint=constraint,
)
channel = phx.equations.compile_channel_flow(problem, stokes, method)
solution = phx.solver.solve_channel_sbdf2(
    channel,
    initial_velocity_coefficients,
    uniform_time_grid,
)
```

`IncompressibleFlowProblem` is shared by the spectral and MAC compilers and owns
viscosity plus an optional compiler-space forcing with a required stable identity.
The periodic and channel compilers interpret that forcing in modal coordinates.
Channel compilation requires the Stokes-plan viscosity to match the problem exactly.
The default Stokes route is the pressure-eliminated banded preparation described
above. Selecting `dense_reference` is an explicit comparison choice and does not
inherit the banded route's production or qualification evidence.

The public channel solve remains primitive-variable: `ChannelFlowSolution` reports
accepted velocity, pressure, affine pressure gradient, divergence, wall, gauge, bulk,
kinetic-energy, and status histories. Prepared continuation uses the complete
`ChannelSBDF2State`: previous/current velocity, previous/current nonlinear rate,
current pressure, pressure gradient, and history count. Backward Euler initializes
the history; every later accepted step uses SBDF2. The prepared method requires its
step size exactly, forbids retry or output-alignment step reduction, and atomically
retains the complete old history on failure. `ChannelMeanConstraint("bulk_flux",
target)` solves two zero-horizontal-mode pressure-gradient multipliers; it is not a
MAC fixed-flux controller.

`ConstantPowerFourierForcingPlan` selects a Hermitian-closed admissible shell, Leray
projects the input, and uses the full-complex native inner product. `power_input` is
volume-mean power: active forcing is scaled by
`volume * power_input / ||u_forced||²`, and the result reports requested/actual mean
and total power. A nonfinite or non-Hermitian input, or forced energy below
`minimum_forced_energy`, produces zero forcing with `active=False` and
`successful=False`; it is never silently renormalized from a low-energy state.

`SolenoidalOUForcingPlan` stores independent real coordinates in an orthonormal
solenoidal Hermitian Fourier basis. `advance` uses exact OU transitions from the
accepted start time to the half and end times and returns all three forcing values.
The RMS parameter is stationary expected volume RMS, with no instantaneous
normalization. `PreparedOUForcedETDRKMethod` couples those values to ETDRK2 at the
start/end stages or ETDRK4 at the start/half/half/end stages, and commits the
coefficient continuation only with the fluid step. Exact OU transition and restart
do not make the ETDRK quadrature of a time-varying acceleration exact.

`PeriodicModalTurbulenceStatisticsPlan` consumes the live full-complex velocity with
unit weight per admissible stored mode--never Hermitian multiplicities. Shell
`integral` is the native domain integral and `density` is integral divided by bin
width. Energy, dissipation, nonlinear transfer, and forcing injection spectra retain
their conservative shell totals; scalar output also includes enstrophy, helicity,
Taylor/Kolmogorov/integral scales, divergence/reality defects, and declared
high-wavenumber energy/dissipation tail fractions with separate validity flags.
`StreamingMomentPlan` supplies accepted-step sample- or time-weighted windows,
second moments, extrema, histograms, and fixed-capacity completed-block uncertainty.

`SpectralChannelStatisticsPlan` forms homogeneous-plane means, raw and central
Reynolds moments, and separate lower/upper wall quantities. Wall shear is
`rho * nu * d<streamwise velocity>/dy` in increasing wall-normal coordinate at both
walls; friction velocity uses the magnitude of each signed shear, and each friction
Reynolds number and wall coordinate uses the channel half-height and that wall's own
friction velocity.

Periodic diagnostics separately report nonlinear energy rate, forcing power,
viscous energy rate, positive dissipation, total semidiscrete energy rate, and the
energy-balance defect. Exact quadratic dealiasing supports the stated rotational
nonlinear energy identity; it is not an entropy-stability claim.

`PeriodicSpectralProductionPlan` is the public production constructor for this
periodic route. It requires an already prepared ETDRK method carrying Hermitian
coordinates, a matching `PeriodicModalTurbulenceStatisticsPlan`, `problem_id`,
absolute `start_time`/`end_time`, nominal `step_size`, and
`checkpoint_interval`. Optional output times must follow the start and stay within
the horizon. If constant-power forcing is supplied,
`constant_power_wiring="compiled"` verifies that the prepared drift already binds
the same forcing identity. `"adapter"` explicitly adds the forcing to the supplied
drift, which must therefore be unforced; the adapter does not detect or remove an
already embedded forcing term.
Alternatively, mutually exclusive `ou_forcing` and `ou_realization` inputs prepare
the coupled OU/ETDRK method. Its accepted state contains full-complex velocity and
real OU coefficients; checkpoint encoding compresses only the velocity leaf.

`SpectralChannelProductionPlan` instead accepts a prepared channel SBDF2 method,
matching velocity and pressure Hermitian coordinates, channel statistics, and required
`problem_id`, absolute `start_time`/`end_time`, and `checkpoint_interval`. It derives
`step_size` from the method--there is no separate step-size constructor argument--and
requires the end time, every output target, and both statistics-window bounds on the
exact lattice rooted at `start_time`. It installs a zero-retry policy
because the method forbids reduced steps. Both route plans derive a default absolute
`maximum_steps` from the horizon, while an explicit larger capacity remains an
absolute cap rather than a relative extension.

Calling `plan.prepare(checkpoint_root, ...)` constructs the durable store and bounded
compiled runtime; `prepared.initialize(...)` creates the accepted
`ProductionRunState`. The underlying `ProductionRunPlan` carries the complete
accepted PyTree, method/controller/RNG state, moment and trigger states, and output
cursor across segments. Checkpoint IDs include stored content; generations are
immutable and monotonically committed, with restore gated by matching
case/runtime/encoding identities. Scheduled accepted-endpoint snapshots use
deterministic event IDs, publication is byte bounded, checkpoint commit drains
earlier output, and a writer failure fails the run.

`SpectralStateArtifact` remains a modal seed/one-step artifact. Exact production
continuation instead checkpoints the complete runtime state. Periodic velocity and
the four channel velocity/history leaves plus channel pressure use
independent-real Hermitian archive coordinates; live state remains full complex.

The legacy `tools/incompressible_spectral_benchmarks.py` command and
`benchmarks/incompressible_spectral.json` cover small periodic and channel
smoke/performance cases only. They are not production-route qualification and do not
establish a universal DNS claim.

`tools/incompressible_flow_qualification.py` separates candidate evidence by route.
Use `periodic-spectral` with
`--output benchmarks/incompressible_periodic_qualification.json`, or
`spectral-channel` with
`--output benchmarks/incompressible_channel_qualification.json`, for the canonical
route artifacts. A generated artifact binds
one exact support tuple plus input, reference, and configuration identities to raw
metrics, gates, status, failure/inconclusive reasons, and a content-derived artifact
ID. Its `release_ready` value remains false. No result is implied merely because the
tool or output path exists.

The tool's assembly command consumes passed, existing route artifacts and emits an
unsigned `CapabilityProfile` candidate containing their artifact IDs. The candidate
has `signed=false` and `profile.released=false`; it is neither a release decision nor
a global DNS badge. Runtime measurements in the legacy benchmark are not promoted
into a qualification timing gate.


## Bounded Galerkin spaces

Common homogeneous endpoint constraints are built into polynomial trial bases:

```python
boundary = phx.discretization.SpectralBoundaryConditionPlan.dirichlet()
basis = phx.discretization.ConstrainedBasisPlan(
    phx.discretization.LegendreBasisPlan(64),
    boundary,
)
space = phx.discretization.TensorSpectralPlan(
    (basis,),
    axis_names=("x",),
).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
galerkin = phx.discretization.SpectralGalerkinMethodPlan().prepare(space)
```

Constraint nullspaces, minimum-norm lifts, and Galerkin solves route through
`phydrax.linalg`. `BoundaryLiftPlan` prepares inhomogeneous endpoint data separately
from the homogeneous unknown. Galerkin mass and stiffness actions remain tensor
products; the dense Poisson path is an explicitly budgeted reference solve.

## Generalized tau systems

`GeneralizedTauPlan` augments an existing square linear operator with explicit lift
columns and constraint rows:

```text
[A  L] [u  ] = [f]
[C  0] [tau]   [g]
```

The supplied tau count must equal the number of constraints. Phydrax does not infer
or select tau terms automatically. The augmented block operator, SVD factorization,
minimum-norm behavior, and diagnostics are all owned by `phydrax.linalg`.

## Exponential time integration

`ETDRKMethod(2)` and `ETDRKMethod(4)` integrate a `SemilinearDrift` whose linear
operator is diagonal. The method uses stable phi-function series at zero and small
arguments. `matrix_phi3_action` extends the shared matrix-function substrate; ETDRK
does not carry a private matrix-function convention.
`ETDRKMethod.prepare(drift, coordinates=...)` binds the complete semilinear drift,
the unbatched diagonal operator, and optional Hermitian acceptance contract once.
`PreparedETDRKMethod` keeps full-complex live state and accepts any finite positive
step, so production end/output clamping is permitted. This differs from prepared
channel SBDF2, whose Stokes factors bind one exact step and therefore require every
production horizon and output target to align with that step.


```text
method = phx.solver.ETDRKMethod(4)
solution = phx.solver.solve_etdrk(
    method,
    compiled.semilinear_drift,
    initial_coefficients,
    save_times,
    discretization_bundle=compiled.discretization_bundle,
)
```

Resolution, basis family, dealiasing shape, and precision are preparation identities.
Changing any of them requires replanning and may trigger a new JAX compilation.
