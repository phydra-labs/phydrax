#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Free-space convolution on bounded uniform Cartesian grids (Hockney method).

The Hockney–Eastwood doubled-grid FFT evaluates the open-boundary discrete
convolution ``out_i = Σ_j g(x_i − x_j) w_j s_j`` exactly (to roundoff) on a
grid embedded in a zero-padded grid of twice the extent per axis. Integrated
Green functions (Qiang, Lidia, Ryne, Limborg-Deprey, Phys. Rev. ST Accel.
Beams 9, 044204, 2006) replace the point-sampled Green function by its exact
integral over the source cell, which keeps the discretization accurate when the
cell aspect ratio is large or the source is under-resolved.
"""

from __future__ import annotations

from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PreparedTensorGrid
from ...typing import parse


FreeSpaceKernel: TypeAlias = Literal[
    "coulomb-igf",
    "newton-igf",
    "newton-softened",
    "biot-savart",
    "tabulated",
]

_RELATIVE_SPACING_TOLERANCE = 1.0e-9


class FreeSpaceConvolutionResult(StrictModule):
    """Open-boundary convolution output restricted to the original grid.

    ``field`` has the grid shape for scalar kernels and a trailing component
    axis for ``"biot-savart"``. ``gradient`` appends one derivative axis
    (``gradient[..., l] = ∂_l field`` for scalar kernels,
    ``gradient[..., i, l] = ∂_l field_i`` for vector kernels) and is ``None``
    when the plan prepared no gradient kernels. ``imaginary_leakage`` is the
    largest imaginary residue of the inverse transforms.
    """

    field: Array
    gradient: Array | None
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    imaginary_leakage: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


def _igf_primitive(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    # ∂³F/∂x∂y∂z = 1/r. Corner coordinates are half-integer multiples of the
    # spacing, so no argument below vanishes and every logarithm is finite.
    r = np.sqrt(x * x + y * y + z * z)
    return (
        -0.5 * z * z * np.arctan(x * y / (z * r))
        - 0.5 * y * y * np.arctan(x * z / (y * r))
        - 0.5 * x * x * np.arctan(y * z / (x * r))
        + y * z * np.log(x + r)
        + x * z * np.log(y + r)
        + x * y * np.log(z + r)
    )


def _igf_gradient_primitive(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    # ∂³/∂x∂y∂z of this primitive is ∂(1/r)/∂x = −x/r³; symmetric in (y, z).
    r = np.sqrt(x * x + y * y + z * z)
    return -x * np.arctan(y * z / (x * r)) + z * np.log(y + r) + y * np.log(z + r)


def _mixed_difference(values: np.ndarray) -> np.ndarray:
    for axis in range(3):
        values = np.diff(values, axis=axis)
    return values


def _integrated_octant(
    offsets: tuple[np.ndarray, ...],
    spacing: np.ndarray,
    *,
    gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Cell averages of ``1/r`` (and ``∇(1/r)``) on the non-negative octant."""
    corners = tuple(
        np.concatenate((axis_offsets, axis_offsets[-1:] + spacing[axis]))
        - 0.5 * spacing[axis]
        for axis, axis_offsets in enumerate(offsets)
    )
    x, y, z = np.meshgrid(*corners, indexing="ij")
    volume = float(np.prod(spacing))
    value = _mixed_difference(_igf_primitive(x, y, z)) / volume
    if not gradient:
        return value, None
    derivative = np.stack(
        (
            _mixed_difference(_igf_gradient_primitive(x, y, z)),
            _mixed_difference(_igf_gradient_primitive(y, z, x)),
            _mixed_difference(_igf_gradient_primitive(z, x, y)),
        ),
        axis=-1,
    )
    return value, derivative / volume


def _softened_octant(
    displacement: np.ndarray,
    softening: float,
    *,
    gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    squared = np.sum(displacement**2, axis=-1) + softening**2
    value = -1.0 / np.sqrt(squared)
    derivative = displacement / squared[..., None] ** 1.5 if gradient else None
    return value, derivative


def _biot_savart_octant(
    displacement: np.ndarray,
    dimension: int,
    *,
    gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    squared = np.sum(displacement**2, axis=-1)
    safe = np.where(squared > 0.0, squared, 1.0)
    if dimension == 2:
        rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        rotated = displacement @ rotation.T
        value = rotated / (2.0 * np.pi * safe[..., None])
        derivative = (
            (
                rotation * safe[..., None, None]
                - 2.0 * rotated[..., :, None] * displacement[..., None, :]
            )
            / (2.0 * np.pi * safe[..., None, None] ** 2)
            if gradient
            else None
        )
    else:
        value = displacement / (4.0 * np.pi * safe[..., None] ** 1.5)
        derivative = (
            (
                np.eye(3) * safe[..., None, None]
                - 3.0 * displacement[..., :, None] * displacement[..., None, :]
            )
            / (4.0 * np.pi * safe[..., None, None] ** 2.5)
            if gradient
            else None
        )
    return value, derivative


def _mirror_octant(
    values: np.ndarray,
    parity: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Extend octant samples to the doubled grid using per-axis parities.

    ``values`` covers offsets ``0 … n`` per axis; ``parity[axis, ...]`` is
    ``+1`` for components even under reflection of that axis and ``−1`` for
    odd components. Negative offsets ``−k`` sit at padded index ``2n − k``;
    the Nyquist index ``n`` keeps the positive sample, which no on-grid pair
    of sites ever addresses.
    """
    result = values
    for axis, count in enumerate(shape):
        index = np.arange(2 * count)
        mirrored = np.where(index <= count, index, 2 * count - index)
        result = np.take(result, mirrored, axis=axis)
        negative_shape = [1] * result.ndim
        negative_shape[axis] = 2 * count
        negative = (index > count).reshape(negative_shape)
        result = result * np.where(negative, parity[axis], 1.0)
    return result


def _padded_table(table: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Place samples at offsets ``−(n−1)…(n−1)`` per axis onto the doubled grid.

    Offset ``m`` sits at padded index ``m mod 2n``; the Nyquist index ``n`` is
    zero because no on-grid pair of sites addresses it.
    """
    dimension = len(shape)
    padded = np.zeros(
        tuple(2 * count for count in shape) + table.shape[dimension:],
        dtype=np.float64,
    )
    index = tuple(
        np.mod(np.arange(-(count - 1), count), 2 * count).reshape(
            (1,) * axis + (2 * count - 1,) + (1,) * (dimension - axis - 1)
        )
        for axis, count in enumerate(shape)
    )
    padded[index] = table
    return padded


def _component_parity(kernel: FreeSpaceKernel, dimension: int) -> np.ndarray:
    """Reflection parity of every kernel component per axis: ``(D, *components)``."""
    match kernel:
        case "coulomb-igf" | "newton-igf" | "newton-softened":
            return np.ones((dimension,))
        case "biot-savart":
            if dimension == 2:
                # K = ẑ × r / (2π r²): K_x is odd in y, K_y is odd in x.
                return np.asarray([[1.0, -1.0], [-1.0, 1.0]])
            # K = r / (4π r³): K_i is odd in x_i only.
            return 1.0 - 2.0 * np.eye(3)
        case "tabulated":
            raise ValueError("Tabulated kernels carry no reflection parity.")
        case _:
            assert_never(kernel)


def _gradient_parity(parity: np.ndarray, dimension: int) -> np.ndarray:
    flip = 1.0 - 2.0 * np.eye(dimension)
    flip_shape = (dimension,) + (1,) * (parity.ndim - 1) + (dimension,)
    return parity[..., None] * flip.reshape(flip_shape)


def _symmetric_kernel(
    kernel: FreeSpaceKernel,
    shape: tuple[int, ...],
    spacing: np.ndarray,
    softening: float | None,
    *,
    gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Doubled-grid samples of a reflection-symmetric kernel and its gradient."""
    dimension = len(shape)
    offsets = tuple(
        np.arange(count + 1, dtype=np.float64) * spacing[axis]
        for axis, count in enumerate(shape)
    )
    displacement = np.stack(np.meshgrid(*offsets, indexing="ij"), axis=-1)
    origin = (0,) * dimension
    match kernel:
        case "coulomb-igf":
            value, derivative = _integrated_octant(offsets, spacing, gradient=gradient)
            value = value / (4.0 * np.pi)
            derivative = None if derivative is None else derivative / (4.0 * np.pi)
        case "newton-igf":
            value, derivative = _integrated_octant(offsets, spacing, gradient=gradient)
            value = -value
            derivative = None if derivative is None else -derivative
        case "newton-softened":
            if softening is None:
                raise AssertionError("newton-softened lost its softening.")
            value, derivative = _softened_octant(
                displacement, softening, gradient=gradient
            )
            value[origin] = 0.0
        case "biot-savart":
            value, derivative = _biot_savart_octant(
                displacement, dimension, gradient=gradient
            )
            value[origin] = 0.0
            if derivative is not None:
                derivative[origin] = 0.0
        case "tabulated":
            raise ValueError("Tabulated kernels are supplied, not sampled.")
        case _:
            assert_never(kernel)
    parity = _component_parity(kernel, dimension)
    return (
        _mirror_octant(value, parity, shape),
        None
        if derivative is None
        else _mirror_octant(derivative, _gradient_parity(parity, dimension), shape),
    )


def _grid_spacing(grid: PreparedTensorGrid, *, integrated: bool) -> np.ndarray:
    spacing = []
    for axis in grid.structured_axes:
        if axis.periodic:
            raise ValueError("Free-space convolution requires bounded, aperiodic axes.")
        if integrated and axis.primary_entity != "interval":
            raise ValueError(
                "Integrated Green functions require cell-centered (interval) axes."
            )
        coordinates = np.asarray(
            axis.interval_centers
            if axis.primary_entity == "interval"
            else axis.point_coordinates,
            dtype=np.float64,
        )
        if coordinates.size < 2:
            raise ValueError("Free-space convolution needs at least two sites per axis.")
        steps = np.diff(coordinates)
        mean = float(np.mean(steps))
        if mean <= 0.0 or np.max(np.abs(steps - mean)) > (
            _RELATIVE_SPACING_TOLERANCE * mean
        ):
            raise ValueError("Free-space convolution requires uniform axis spacing.")
        spacing.append(mean)
    return np.asarray(spacing, dtype=np.float64)


class FreeSpaceConvolutionPlan(StrictModule, NonTrainableState):
    """Hockney doubled-grid FFT convolution with one declared free-space kernel.

    ``convolve(source)`` returns ``Σ_j g(x_i − x_j) w_j source_j`` where ``w``
    is the grid quadrature measure, so ``source`` is a density. Kernels:

    - ``"coulomb-igf"``: cell-integrated ``1/(4π r)`` (three dimensions); the
      result solves ``−∇²φ = source``. Owners divide by ``ε₀``.
    - ``"newton-igf"``: cell-integrated ``−1/r`` (three dimensions); the result
      solves ``∇²Φ = 4π source``. Owners multiply by ``G``.
    - ``"newton-softened"``: point-sampled ``−1/√(r² + softening²)`` in one to
      three dimensions with the self-cell sample excluded (mesh point-mass
      convention).
    - ``"biot-savart"``: point-sampled ``ẑ × r/(2π r²)`` (two dimensions, scalar
      source) or ``r/(4π r³)`` (three dimensions, vector source; the result is
      ``source × K``), zero at the self cell.

    - ``"tabulated"``: caller-supplied ``kernel_table`` of shape
      ``(2n₁ − 1, …, 2n_d − 1, *components)`` holding ``g`` at every on-grid
      displacement ``(m₁h₁, …, m_d h_d)``, ``m_a = −(n_a − 1) … n_a − 1``, for
      kernels without reflection symmetry (for example retarded or
      cell-integrated anisotropic Green functions). A scalar source then yields
      one field per trailing component.

    ``gradient=True`` prepares the derivative kernels under the same sampling
    rule (cell-integrated ``∇(1/r)`` for the integrated kernels, point-sampled
    analytic derivatives otherwise) so fields are obtained without finite
    differences; it is refused for tabulated kernels. All transforms are
    float64/complex128.
    """

    kernel: FreeSpaceKernel = eqx.field(static=True)
    grid: PreparedTensorGrid
    dimension: int = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    spacing: Array
    softening: float | None = eqx.field(static=True)
    kernel_transform: Array
    gradient_kernel_transform: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel: FreeSpaceKernel,
        grid: PreparedTensorGrid,
        /,
        *,
        softening: float | None = None,
        gradient: bool = False,
        kernel_table: ArrayLike | None = None,
    ) -> None:
        kernel_ = parse(kernel, FreeSpaceKernel, "kernel")
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        if not isinstance(gradient, bool):
            raise TypeError("gradient must be a bool.")
        dimension = len(grid.shape)
        softening_: float | None = None
        match kernel_:
            case "coulomb-igf" | "newton-igf":
                integrated = True
                if dimension != 3:
                    raise ValueError(
                        "Integrated Green functions are defined in three dimensions."
                    )
            case "newton-softened":
                integrated = False
                if dimension not in (1, 2, 3):
                    raise ValueError(
                        "The softened Newton kernel supports one to three dimensions."
                    )
                if softening is None:
                    raise ValueError("newton-softened requires a positive softening.")
                softening_ = float(softening)
                if not np.isfinite(softening_) or softening_ <= 0.0:
                    raise ValueError("softening must be finite and positive.")
            case "biot-savart":
                integrated = False
                if dimension not in (2, 3):
                    raise ValueError(
                        "Biot–Savart kernels support two or three dimensions."
                    )
            case "tabulated":
                integrated = False
                if kernel_table is None:
                    raise ValueError("The tabulated kernel requires kernel_table.")
                if gradient:
                    raise ValueError("Tabulated kernels do not prepare gradients.")
            case _:
                assert_never(kernel_)
        if softening is not None and softening_ is None:
            raise ValueError("softening applies only to the newton-softened kernel.")
        if kernel_table is not None and kernel_ != "tabulated":
            raise ValueError("kernel_table applies only to the tabulated kernel.")
        spacing = _grid_spacing(grid, integrated=integrated)
        shape = tuple(grid.shape)
        spatial_axes = tuple(range(dimension))
        table_fingerprint: str | None = None
        if kernel_table is None:
            kernel_padded, gradient_padded = _symmetric_kernel(
                kernel_, shape, spacing, softening_, gradient=gradient
            )
        else:
            table = np.asarray(kernel_table, dtype=np.float64)
            expected = tuple(2 * count - 1 for count in shape)
            if table.ndim < dimension or table.shape[:dimension] != expected:
                raise ValueError(
                    f"kernel_table must begin with the offset shape {expected}."
                )
            if not np.all(np.isfinite(table)):
                raise ValueError("kernel_table must be finite.")
            table_fingerprint = array_tree_fingerprint(table)
            kernel_padded = _padded_table(table, shape)
            gradient_padded = None
        kernel_transform = np.fft.fftn(kernel_padded, axes=spatial_axes)
        gradient_transform = (
            None
            if gradient_padded is None
            else np.fft.fftn(gradient_padded, axes=spatial_axes)
        )
        self.kernel = kernel_
        self.grid = grid
        self.dimension = dimension
        self.shape = shape
        self.padded_shape = tuple(2 * count for count in shape)
        self.spacing = jnp.asarray(spacing)
        self.softening = softening_
        self.kernel_transform = jnp.asarray(kernel_transform)
        self.gradient_kernel_transform = (
            None if gradient_transform is None else jnp.asarray(gradient_transform)
        )
        identity: dict[str, object] = {
            "kind": "free-space-convolution",
            "kernel": kernel_,
            "grid": grid.prepared_id,
            "spacing": spacing.tolist(),
            "softening": softening_,
            "gradient": gradient,
        }
        if table_fingerprint is not None:
            identity["kernel_table"] = table_fingerprint
        self.plan_id = canonical_fingerprint(identity)

    @property
    def source_shape(self) -> tuple[int, ...]:
        """Expected ``source`` shape: the grid shape, plus ``(3,)`` for 3-D Biot–Savart."""
        if self.kernel == "biot-savart" and self.dimension == 3:
            return self.shape + (3,)
        return self.shape

    def _apply_kernel(self, source_hat: Array, kernel_hat: Array) -> Array:
        if self.source_shape != self.shape:
            if kernel_hat.ndim == source_hat.ndim:
                return jnp.cross(source_hat, kernel_hat, axis=-1)
            # Gradient kernels: out[..., i, l] = ε_ijk source_j ∂_l K_k.
            return jnp.cross(source_hat[..., :, None], kernel_hat, axis=-2)
        extra = kernel_hat.ndim - source_hat.ndim
        return source_hat.reshape(source_hat.shape + (1,) * extra) * kernel_hat

    def convolve(self, source: ArrayLike, /) -> FreeSpaceConvolutionResult:
        """Convolve one density with the prepared kernel on the doubled grid."""
        density = jnp.asarray(source, dtype=jnp.float64)
        expected = self.source_shape
        if density.shape != expected:
            raise ValueError(f"Free-space convolution source must have shape {expected}.")
        component_rank = density.ndim - self.dimension
        weights = self.grid.quadrature_weights.reshape(self.shape + (1,) * component_rank)
        padding = tuple((0, count) for count in self.shape) + ((0, 0),) * component_rank
        padded = jnp.pad(density * weights, padding)
        spatial_axes = tuple(range(self.dimension))
        source_hat = jnp.fft.fftn(padded, axes=spatial_axes)
        slices = tuple(slice(0, count) for count in self.shape)
        field_padded = jnp.fft.ifftn(
            self._apply_kernel(source_hat, self.kernel_transform), axes=spatial_axes
        )
        field = field_padded[slices].real
        leakage = jnp.max(jnp.abs(field_padded.imag))
        finite = jnp.all(jnp.isfinite(field))
        gradient = None
        if self.gradient_kernel_transform is not None:
            gradient_padded = jnp.fft.ifftn(
                self._apply_kernel(source_hat, self.gradient_kernel_transform),
                axes=spatial_axes,
            )
            gradient = gradient_padded[slices].real
            leakage = jnp.maximum(leakage, jnp.max(jnp.abs(gradient_padded.imag)))
            finite = finite & jnp.all(jnp.isfinite(gradient))
        return FreeSpaceConvolutionResult(
            field,
            gradient,
            self.padded_shape,
            leakage,
            finite,
            self.plan_id,
        )


__all__ = [
    "FreeSpaceConvolutionPlan",
    "FreeSpaceConvolutionResult",
    "FreeSpaceKernel",
]
