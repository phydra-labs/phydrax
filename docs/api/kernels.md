# Positive-definite kernels

`phydrax.kernels` is the shared covariance-function layer for Gaussian-process
inference, coreset compression, inducing-point selection, and other kernel
algorithms. A kernel is an Equinox PyTree, so array-valued scales, amplitudes,
feature maps, and transform parameters remain visible to JAX transformations.

## Contract

Every kernel implements three explicit operations:

- `pairwise(left, right)` evaluates two coordinate vectors;
- `matrix(left, right)` evaluates a Gram or cross-Gram matrix;
- `diagonal(points)` evaluates only the Gram diagonal.

A one-dimensional design may be passed as a vector. General designs have shape
`(point, coordinate)`. Kernels also expose `max_derivative_order`,
`is_unit_diagonal`, and a diagnostic `kernel_id`. The ID records a stable method
identity; it is not a serialization or reconstruction format.

Kernel algebra is covariance-safe by construction. Addition and kernel-by-kernel
multiplication produce flattened `SumKernel` and `ProductKernel` trees. Scalar
multiplication requires a finite nonnegative covariance scale. Use
`AmplitudeKernel(kernel, amplitude)` when a parameter is a standard-deviation
amplitude: it evaluates `amplitude**2 * kernel`. Subtraction, negation, and
unconstrained scalar scaling are intentionally absent because they do not generally
preserve positive definiteness.

```python
import jax.numpy as jnp
import phydrax as phx

base = phx.kernels.Matern52Kernel(length_scale=jnp.asarray([0.2, 0.5]))
periodic_feature = phx.kernels.InputTransformedKernel(
    phx.kernels.SquaredExponentialKernel(length_scale=jnp.ones((2,))),
    lambda point: jnp.asarray(
        [jnp.sin(2.0 * jnp.pi * point[0]), jnp.cos(2.0 * jnp.pi * point[0])]
    ),
    transform_id="unit-periodic",
    max_derivative_order=None,
)
kernel = phx.kernels.AmplitudeKernel(base + periodic_feature, 0.3)
coordinate = jnp.linspace(0.0, 1.0, 32)
points = jnp.stack((coordinate, coordinate**2), axis=1)
gram = kernel.matrix(points, points)
```

## Smoothness and transforms

`SquaredExponentialKernel` and `InverseMultiquadricKernel` declare no finite
mean-square derivative limit. `Matern32Kernel` certifies first derivatives and
`Matern52Kernel` certifies second derivatives. Kernel sums, products, scales, and
amplitudes propagate the most restrictive child certificate. Functional GP
observations use this certificate to reject unsupported differential covariance
blocks before inference.

`InputTransformedKernel` pulls a positive-definite kernel back through one
pointwise deterministic transform. Its transform may itself be an Equinox module,
which gives a deep kernel whose feature parameters are differentiable PyTree leaves.
Declare the transform's actual derivative support; `None` means no finite limit and
`0` means value observations only. `AffineInputTransform.from_points` provides
explicit coordinate standardization without changing the kernel's public contract.

`AbstractFiniteFeatureKernel` is the capability contract for covariances with
whitened real features. `FiniteFeatureKernel` is its callable-map implementation:
if `feature_map(x)` is `phi(x)` and `feature_factor` is `F`, the evaluated feature
is `phi(x) @ F`. `from_precision_cholesky` constructs that factor from a declared
triangular precision Cholesky factor. `kernel_features` resolves the same
capability through supported kernel algebra. Scalar exact GP inference uses it
automatically when the declared feature rank is smaller than the observation
count; otherwise it retains the dense observation-space path.

## Base interfaces

::: phydrax.kernels.AbstractPositiveDefiniteKernel
    options:
        members:
            - pairwise
            - matrix
            - diagonal
            - max_derivative_order
            - is_unit_diagonal
            - kernel_id

---

::: phydrax.kernels.AbstractUnitDiagonalKernel

---

::: phydrax.kernels.AbstractStationaryKernel

## Stationary kernels

::: phydrax.kernels.SquaredExponentialKernel

---

::: phydrax.kernels.Matern32Kernel

---

::: phydrax.kernels.Matern52Kernel

---

::: phydrax.kernels.InverseMultiquadricKernel

## Algebra

::: phydrax.kernels.SumKernel

---

::: phydrax.kernels.ProductKernel

---

::: phydrax.kernels.ScaleKernel

---

::: phydrax.kernels.AmplitudeKernel
    options:
        members:
            - __init__
            - variance_scale
            - pairwise
            - matrix
            - diagonal

## Input and feature transforms

::: phydrax.kernels.AffineInputTransform
    options:
        members:
            - __init__
            - from_points
            - __call__

---

::: phydrax.kernels.InputTransformedKernel
    options:
        members:
            - __init__
            - pairwise
            - matrix
            - diagonal

---

::: phydrax.kernels.FiniteFeatureKernel
    options:
        members:
            - __init__
            - from_precision_cholesky
            - features
            - pairwise
            - matrix
            - diagonal


## Finite-feature capability

::: phydrax.kernels.AbstractFiniteFeatureKernel
    options:
        members:
            - features
            - feature_rank

---

::: phydrax.kernels.kernel_feature_rank

---

::: phydrax.kernels.kernel_features

## Laplacian spectral kernels

`SpectralFeatureKernel` combines an immutable, measure-orthonormal
`DiscreteLaplacianEigenbasis` with a nonnegative spectral multiplier. Normalized
evaluation divides by the declared probability-measure average marginal variance,
not by each point's diagonal. Heat and Matérn multipliers keep geometry,
normalization, and covariance law separate. A finite basis therefore gives exact
weight-space GP inference for that declared truncated covariance.

The geometric spectral construction follows the functional-calculus perspective
described in [*The GeometricKernels Package: Heat and Matérn Kernels for Geometric
Learning on Manifolds, Meshes, and Graphs*](https://www.jmlr.org/papers/v26/24-1185.html).
Phydrax owns its metric/cochain IR, eigensolver provenance, feature algebra, and GP
integration directly; there is no runtime `geometric-kernels` dependency.

::: phydrax.kernels.AbstractSpectralMultiplier

---

::: phydrax.kernels.HeatSpectralMultiplier

---

::: phydrax.kernels.MaternSpectralMultiplier

---

::: phydrax.kernels.SpectralFeatureKernel

## Compact and combinatorial spaces

Sphere kernels use analytic addition-theorem expansions. SO, SU, Stiefel, and
Grassmann kernels use finite nonnegative tensor-character expansions of normalized
ambient similarities; they are covariance-safe homogeneous-space kernels, not
geodesic-distance exponentials or claims of a complete irreducible spectrum.
Hamming and hypercube kernels use stable Krawtchouk recurrences and
multiplicity-aware spectral weights. All constructors validate membership and
preserve JAX differentiation through multiplier hyperparameters.
Tolerance-close sphere points are radially canonicalized before addition-theorem
evaluation. Matrix and frame validators retain their accepted ambient values, so
their `diagonal(...)` methods report the actual self-covariance; `normalize=True`
normalizes coefficient mass and gives unit diagonal on exact manifold points rather
than asserting it for tolerance-close inputs.

::: phydrax.kernels.SphereSpectralKernel

---

::: phydrax.kernels.SpecialOrthogonalCharacterKernel

---

::: phydrax.kernels.SpecialUnitaryCharacterKernel

---

::: phydrax.kernels.StiefelSpectralKernel

---

::: phydrax.kernels.GrassmannSpectralKernel

---

::: phydrax.kernels.HammingSpectralKernel

---

::: phydrax.kernels.HypercubeSpectralKernel

## Hodge and operator-valued covariances

`CochainHodgeSpectralKernel` composes harmonic, exact, and coexact scalar
covariances with independent nonnegative amplitudes. Projected tangent and
differential-form kernels instead return covariance blocks in an ambient
coordinate representation. Projectors act at both endpoints, so the resulting
blocks satisfy the declared tangent constraints and remain positive semidefinite.

::: phydrax.kernels.CochainHodgeSpectralKernel

---

::: phydrax.kernels.AbstractOperatorValuedKernel

---

::: phydrax.kernels.ProjectedTangentKernel

---

::: phydrax.kernels.ProjectedDifferentialFormKernel

---

::: phydrax.kernels.sphere_tangent_projector

---

::: phydrax.kernels.sphere_tangent_kernel

---

::: phydrax.kernels.sphere_differential_form_kernel

## Noncompact fixed-noise features

Hyperbolic and SPD kernels use fixed Helgason or horospherical plane waves and
explicit, immutable multivariate-Cauchy importance proposals. Target weights
combine Matérn functional calculus at the shifted symmetric-space Laplacian
eigenvalues with the relative Harish-Chandra Plancherel density. A
convention-dependent global spectral-measure constant is left to
`AmplitudeKernel`.

Sampling is never parameter-dependent inside traced evaluation: optimize the
smooth kernel hyperparameters against fixed noise, inspect effective sample size
and the unbiased Monte Carlo standard error, then resample only through an explicit
method call. The Cauchy importance estimator has finite variance only for Matérn
smoothness greater than `0.25`; diagnostics report an infinite standard error at or
below that boundary and for singleton proposals, and reject zero-total-weight
samples. Factory-generated proposal IDs content-hash the fixed arrays, geometry,
and proposal scale and support reproducible prefix-rank convergence studies.

::: phydrax.kernels.NoncompactFeatureProposal

---

::: phydrax.kernels.ImportanceFeatureDiagnostics

---

::: phydrax.kernels.hyperbolic_feature_proposal

---

::: phydrax.kernels.HyperbolicRandomFeatureKernel

---

::: phydrax.kernels.spd_feature_proposal

---

::: phydrax.kernels.SPDRandomFeatureKernel

## Benchmark

Run the graph-spectrum, finite-feature, dense-parity, storage, and conditioning
smoke workflow with:

```bash
python -m tools.spectral_kernel_benchmarks --smoke
```