#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..discretization.finite_volume._mac_momentum import (
    PreparedMACMomentumOperators,
)
from ..linalg import FFTLinearTransform, RealTrigonometricTransform
from ..linalg._linear_transform import AbstractLinearTransform


_ESSENTIAL_NORMAL_KINDS = (
    "no-slip",
    "free-slip",
    "symmetry",
    "velocity-inflow",
    "normal-flux-inflow",
)


def uniform_axis_spacing(axis, /) -> float | None:
    """Return the certified uniform spacing of one structured axis, if any."""
    widths = np.asarray(axis.interval_widths, dtype=float)
    if widths.size == 0:
        return None
    spacing = float(widths[0])
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.allclose(widths, spacing, rtol=1e-10, atol=1e-12)
    ):
        return None
    return spacing


def _periodic_axis_transform(axis, dtype, /):
    spacing = uniform_axis_spacing(axis)
    if spacing is None or not axis.periodic:
        return None
    count = int(axis.interval_widths.size)
    transform: AbstractLinearTransform = FFTLinearTransform(
        count, dtype=np.result_type(dtype, np.complex64)
    )
    angles = 2.0 * np.pi * np.arange(count) / count
    spectrum = 4.0 * np.sin(0.5 * angles) ** 2 / spacing**2
    trace = 0.0 if count == 1 else 2.0 * count / spacing**2
    return transform, jnp.asarray(spectrum, dtype=dtype), trace


def pressure_cell_axis_transform(axis, dtype, /):
    """Diagonalize a uniform cell-pressure axis with periodic/Neumann closure."""
    periodic = _periodic_axis_transform(axis, dtype)
    if periodic is not None:
        return periodic
    spacing = uniform_axis_spacing(axis)
    if spacing is None or axis.periodic:
        return None
    count = int(axis.interval_widths.size)
    transform: AbstractLinearTransform = RealTrigonometricTransform(
        "dct", 2, count, dtype=dtype
    )
    angles = np.arange(count) * np.pi / count
    spectrum = 4.0 * np.sin(0.5 * angles) ** 2 / spacing**2
    trace = max(2.0 * count - 2.0, 0.0) / spacing**2
    return transform, jnp.asarray(spectrum, dtype=dtype), trace


def normal_velocity_is_essential(
    momentum: PreparedMACMomentumOperators, component: int, /
) -> bool:
    axis = momentum.operators.discretization.grid.structured_axes[component]
    return axis.periodic or all(
        momentum.boundaries.side_kind(component, side) in _ESSENTIAL_NORMAL_KINDS
        for side in ("lower", "upper")
    )


def velocity_face_axis_transform(
    momentum: PreparedMACMomentumOperators,
    component: int,
    derivative_axis: int,
    /,
):
    """Diagonalize one uniform velocity-face Laplacian axis."""
    axis = momentum.operators.discretization.grid.structured_axes[derivative_axis]
    dtype = momentum.operators.pressure_space.dtype
    periodic = _periodic_axis_transform(axis, dtype)
    if periodic is not None:
        return periodic
    spacing = uniform_axis_spacing(axis)
    if spacing is None or axis.periodic:
        return None
    if derivative_axis == component:
        if not normal_velocity_is_essential(momentum, component):
            return None
        count = int(axis.interval_widths.size) - 1
        if count < 1:
            return None
        transform: AbstractLinearTransform = RealTrigonometricTransform(
            "dst", 1, count, dtype=dtype
        )
        angles = (np.arange(count) + 1.0) * np.pi / (count + 1.0)
        trace = 2.0 * count / spacing**2
    else:
        count = int(axis.interval_widths.size)
        lower_d = momentum.boundaries.tangential_dirichlet(derivative_axis, "lower")
        upper_d = momentum.boundaries.tangential_dirichlet(derivative_axis, "upper")
        if lower_d and upper_d:
            transform = RealTrigonometricTransform("dst", 2, count, dtype=dtype)
            angles = (np.arange(count) + 1.0) * np.pi / count
            trace = (2.0 * count + 2.0) / spacing**2
        elif not lower_d and not upper_d:
            transform = RealTrigonometricTransform("dct", 2, count, dtype=dtype)
            angles = np.arange(count) * np.pi / count
            trace = max(2.0 * count - 2.0, 0.0) / spacing**2
        elif lower_d:
            transform = RealTrigonometricTransform("dst", 4, count, dtype=dtype)
            angles = (np.arange(count) + 0.5) * np.pi / count
            trace = 2.0 * count / spacing**2
        else:
            transform = RealTrigonometricTransform("dct", 4, count, dtype=dtype)
            angles = (np.arange(count) + 0.5) * np.pi / count
            trace = 2.0 * count / spacing**2
    spectrum = 4.0 * np.sin(0.5 * angles) ** 2 / spacing**2
    return transform, jnp.asarray(spectrum, dtype=dtype), trace


def modal_sum(spectra: Sequence[Array], /, *, dtype=None) -> Array:
    """Form a tensor-product Kronecker sum without materializing matrices."""
    spectra_ = tuple(spectra)
    if not spectra_:
        if dtype is None:
            raise ValueError("An empty modal sum requires an explicit dtype.")
        return jnp.asarray(0.0, dtype=dtype)
    shape = tuple(int(value.size) for value in spectra_)
    result = jnp.zeros(shape, dtype=jnp.result_type(*[value.dtype for value in spectra_]))
    for axis, spectrum in enumerate(spectra_):
        reshape = [1] * len(shape)
        reshape[axis] = int(spectrum.size)
        result = result + spectrum.reshape(tuple(reshape))
    return result


def pressure_cell_line_coefficients(axis, dtype, /):
    """Assemble the nonperiodic Neumann cell-pressure line operator."""
    if axis.periodic:
        raise ValueError("A pressure hybrid line must be explicitly nonperiodic.")
    widths = jnp.asarray(axis.interval_widths, dtype=dtype)
    centers = jnp.asarray(axis.interval_centers, dtype=dtype)
    count = int(widths.size)
    if count < 1:
        raise ValueError("A pressure hybrid line must contain at least one cell.")
    if count == 1:
        empty = jnp.zeros((0,), dtype=dtype)
        return empty, jnp.zeros((1,), dtype=dtype), empty
    distances = centers[1:] - centers[:-1]
    lower = -1.0 / (widths[1:] * distances)
    upper = -1.0 / (widths[:-1] * distances)
    diagonal = jnp.zeros((count,), dtype=dtype)
    diagonal = diagonal.at[1:].add(-lower)
    diagonal = diagonal.at[:-1].add(-upper)
    return lower, diagonal, upper


def velocity_face_line_coefficients(
    momentum: PreparedMACMomentumOperators,
    component: int,
    line_axis: int,
    /,
):
    """Assemble one velocity-face line, retaining its staggered coefficients."""
    axis = momentum.operators.discretization.grid.structured_axes[line_axis]
    dtype = momentum.operators.pressure_space.dtype
    widths = jnp.asarray(axis.interval_widths, dtype=dtype)
    centers = jnp.asarray(axis.interval_centers, dtype=dtype)
    if axis.periodic:
        period = jnp.asarray(axis.bounds[1] - axis.bounds[0], dtype=dtype)
        previous = jnp.roll(centers, 1).at[0].add(-period)
        distances = centers - previous
        if line_axis == component:
            dual = momentum.face_dual_widths[component]
            lower_full = -1.0 / (dual * jnp.roll(widths, 1))
            upper_full = -1.0 / (dual * widths)
        else:
            lower_full = -1.0 / (widths * distances)
            upper_full = -1.0 / (widths * jnp.roll(distances, -1))
        return (
            lower_full[1:],
            -(lower_full + upper_full),
            upper_full[:-1],
            (lower_full[0], upper_full[-1]),
        )
    if line_axis == component:
        dual = momentum.face_dual_widths[component][1:-1]
        diagonal = 1.0 / (dual * widths[:-1]) + 1.0 / (dual * widths[1:])
        if dual.size == 1:
            empty = jnp.zeros((0,), dtype=dtype)
            return empty, diagonal, empty, None
        return (
            -1.0 / (dual[1:] * widths[1:-1]),
            diagonal,
            -1.0 / (dual[:-1] * widths[1:-1]),
            None,
        )
    count = int(widths.size)
    distances = centers[1:] - centers[:-1]
    lower = -1.0 / (widths[1:] * distances)
    upper = -1.0 / (widths[:-1] * distances)
    diagonal = jnp.zeros((count,), dtype=dtype)
    if count > 1:
        diagonal = diagonal.at[1:].add(-lower)
        diagonal = diagonal.at[:-1].add(-upper)
    if momentum.boundaries.tangential_dirichlet(line_axis, "lower"):
        diagonal = diagonal.at[0].add(1.0 / (widths[0] * (centers[0] - axis.bounds[0])))
    if momentum.boundaries.tangential_dirichlet(line_axis, "upper"):
        diagonal = diagonal.at[-1].add(
            1.0 / (widths[-1] * (axis.bounds[1] - centers[-1]))
        )
    return lower, diagonal, upper, None


def certify_separable_action(
    represented: Array,
    exact: Array,
    tolerance: float,
    /,
) -> tuple[Array, bool]:
    """Certify an exact matrix-free separable action on a deterministic probe."""
    represented_ = jnp.asarray(represented)
    exact_ = jnp.asarray(exact)
    if represented_.shape != exact_.shape:
        raise ValueError("Separable and physical actions must have the same shape.")
    defect = jnp.linalg.norm((represented_ - exact_).reshape((-1,)))
    real_dtype = jnp.real(exact_).dtype
    epsilon = jnp.finfo(real_dtype).eps
    scale = jnp.maximum(1.0, jnp.linalg.norm(exact_.reshape((-1,))))
    threshold = jnp.maximum(100.0 * tolerance, 4096.0 * epsilon * scale)
    certified = bool(np.asarray(jnp.isfinite(defect) & (defect <= threshold)))
    return defect, certified


def diagonal_resource_counts(shape, dtype, maximum_bytes: int, description: str, /):
    """Preflight diagonal factors and peak transform workspace."""
    count = int(np.prod(shape))
    factor_bytes = count * np.dtype(dtype).itemsize
    workspace_bytes = 3 * factor_bytes
    total_bytes = factor_bytes + workspace_bytes
    if total_bytes > maximum_bytes:
        raise ValueError(f"{description} resources exceed the configured budget.")
    return count, factor_bytes, workspace_bytes, total_bytes


def iterative_workspace_bytes(shape, dtype, maximum_bytes: int, description: str, /):
    """Preflight the fixed six-vector workspace used by MAC Krylov solves."""
    workspace_bytes = 6 * int(np.prod(shape)) * np.dtype(dtype).itemsize
    if workspace_bytes > maximum_bytes:
        raise ValueError(f"{description} resources exceed the configured budget.")
    return workspace_bytes


__all__ = []
