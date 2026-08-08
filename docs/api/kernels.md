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

`FiniteFeatureKernel` represents a covariance through whitened real features. If
`feature_map(x)` is `phi(x)` and `feature_factor` is `F`, the evaluated feature is
`phi(x) @ F`. `from_precision_cholesky` constructs that factor from a declared
triangular precision Cholesky factor. Scalar exact GP inference recognizes this
finite-feature structure and factors in feature space when it is cheaper than a
dense observation-space factorization.

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
