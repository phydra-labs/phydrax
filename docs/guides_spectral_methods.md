# Global spectral methods

Phydrax separates a spectral basis, its physical evaluation support, modal degrees of
freedom, nonlinear realization, and temporal method. Tensor spectral objects are global
products rather than spectral elements. Spherical spaces instead bind exact S2FFT
sampling theorems to a round-sphere support; neither path invents element topology.
Local high-order tensor elements, mapped geometry, CG/DG coupling, and DGSEM
live in the finite-element compiler; see [Spectral elements](guides_spectral_elements.md).

## Spaces and representations

A basis plan owns mathematical modes. Preparing a tensor plan binds those modes to
physical bounds, quadrature, transforms, and exact field-space identities:

```python
import jax.numpy as jnp
import phydrax as phx
import jax.random as jr

key = jr.key(0)

space = phx.discretization.TensorSpectralPlan(
    (phx.discretization.FourierBasisPlan(128),),
    axis_names=("x",),
    field_name="u",
).prepare(jnp.asarray([[0.0], [1.0]]))
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
`2*ell+1` degree block. Explicit eigenbases and dense Laplacians are separately
resource-bounded. Coordinate partial derivatives, arbitrary masks, HEALPix sampling,
and nonlinear spherical dealiasing are outside this contract.

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
).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
problem = phx.equations.IncompressibleFlowProblem(2, 1e-3)
method = phx.discretization.PseudospectralMethodPlan(
    dealiasing=phx.discretization.PaddingDealiasingPlan(2),
)
compiled = phx.equations.compile_periodic_incompressible_flow(
    problem, space, method
)
```

The public state remains full complex. `HermitianSpectralCoordinates` provides an
independent real chart for Newton, continuation, Lyapunov, and periodic-orbit
analysis without changing DNS storage. Paired Fourier modes use norm-preserving
real/imaginary coordinates; zero and Nyquist fixed points remain real.
Coordinate construction rejects nonintegral component dimensions and preflights its
explicit real-coordinate capacity. `TensorSpectralSymmetry` applies normalized
Fourier translations, supported reflections, and an orthogonal component action
directly in modal space. Translation/reflection composition follows the declared
semidirect-product action, including reflected translations.

Wall-bounded channel flow uses a Fourier x Chebyshev x Fourier tensor plan and a
separate constrained Stokes preparation:

```text
problem = phx.equations.IncompressibleFlowProblem(3, viscosity=1e-3)
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

`IncompressibleFlowProblem` is shared by periodic and channel spectral compilers and
owns viscosity plus an optional modal forcing with a required stable identity.
`ChannelStokesPlan` is a budgeted dense primitive velocity–pressure reference solve;
channel compilation requires its viscosity to match the problem exactly. Persistent
state contains velocity only; pressure, affine pressure gradient, divergence, wall
traces, gauge, bulk velocity, kinetic energy, and status are returned as solve
evidence. `ChannelMeanConstraint("bulk_flux", target)` augments the zero horizontal
mode with pressure-gradient Lagrange multipliers. The fixed-step SBDF2 path uses
backward Euler initialization, rejects nonuniform time grids, validates initial wall
and divergence constraints, latches the first failed Stokes step, and retains the
last accepted state through the remainder of its fixed-shape scan.

Periodic diagnostics separately report nonlinear energy rate, forcing power,
viscous energy rate, positive dissipation, total semidiscrete energy rate, and the
energy-balance defect. Exact quadratic dealiasing supports the stated rotational
nonlinear energy identity; it is not an entropy-stability claim.

`BoundedEvolutionObservationPlan` applies to implementations of
`AbstractEvolution`. The specialized channel SBDF2 service currently returns its
declared saved grid and is not silently adapted into that one-step evolution
contract. Callable observables and modal forcing always require explicit stable IDs.
`SpectralStateArtifact` stores full-complex coefficients in the atomic
checksum-validated array archive. Reads revalidate shape, dtype, restart kind, and the
content-derived artifact fingerprint. Artifacts without a step size are seeds.
Setting `restartable=True` requires a positive fixed step size and is only an exact
checkpoint for a one-step method whose complete runtime state is represented by
`(state, time, step, step_size)`.

The periodic/channel qualification campaign is:

```bash
python tools/incompressible_spectral_benchmarks.py \
  --output benchmarks/incompressible_spectral.json
```


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
).prepare(jnp.asarray([[-1.0], [1.0]]))
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
