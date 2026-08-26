# Global spectral methods

Phydrax separates a spectral basis, its physical evaluation grid, modal degrees of
freedom, nonlinear realization, and temporal method. These objects are global tensor
products; they are not spectral elements and do not introduce element topology.

## Spaces and representations

A basis plan owns mathematical modes. Preparing a tensor plan binds those modes to
physical bounds, quadrature, transforms, and exact field-space identities:

```python
import jax.numpy as jnp
import phydrax as phx

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

## Nonlinear evaluation and dealiasing

Nonlinear pseudospectral compilation requires an explicit policy. Quadratic Fourier
products normally use 3/2 overresolution; cubic products require 2× overresolution.

```python
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

```python
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

For Fourier spaces, `SpatialNoiseBasis.from_spectrum` first constructs real
weighted-orthonormal Laplacian modes and then projects them into full complex
storage. Its complex modal columns preserve conjugate symmetry under real Wiener
coefficients; independent one-sided Fourier modes are never substituted.

!!! warning
    Diffrax currently labels complex-dtype integration as work in progress. The
    conjugate-symmetric spectral SPDE paths are covered by replay, reality, and
    analytic-moment tests, but Phydrax does not strengthen that upstream guarantee.
    Use an explicitly real-valued state formulation when that guarantee is required.

The compiled state is modal. Use `compiled.project_state` for initial data and
`compiled.reconstruct_state` for observables and output. Constant-coefficient scalar
linear parts lower to `DiagonalLinearOperator`; general linearizations remain explicit
matrix-free operators.

## Periodic conservation and entropy

`ConservationProblemIR(..., boundaries=None)` declares a fully periodic problem.
A spectral conservation method differentiates projected physical fluxes and therefore
preserves the zero Fourier mode up to roundoff:

```python
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

The equation-owned `ConvexEntropyPair` supplies entropy, entropy variables, flux,
and admissibility. Spectral diagnostics report total entropy and its semidiscrete
rate. They do not claim entropy stability; a proven entropy-stable split form is a
separate numerical contract.

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

```python
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
