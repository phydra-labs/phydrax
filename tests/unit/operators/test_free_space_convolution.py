#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Free-space Hockney convolution with integrated Green functions.

References are independent of the implementation: tensor Gauss–Legendre cell
quadrature of the Coulomb kernel, the depolarization integrals of a homogeneous
ellipsoid (scipy quadrature), the closed-form field of a spherical Gaussian
charge, and O(N²) direct sums for point-sampled kernels.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import erf

import phydrax as phx
from phydrax.discretization import (
    PreparedTensorGrid,
    TensorGridPlan,
    UniformAxisSpec,
    UniformCellAxisSpec,
)


def _cell_grid(
    shape: tuple[int, ...], lower: tuple[float, ...], upper: tuple[float, ...]
) -> PreparedTensorGrid:
    return TensorGridPlan(tuple(UniformCellAxisSpec(count) for count in shape)).prepare(
        np.asarray([lower, upper])
    )


def _cell_average(
    offset: np.ndarray,
    spacing: np.ndarray,
    integrand: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    order: int = 24,
) -> float:
    nodes, weights = leggauss(order)
    axes = [offset[axis] + 0.5 * spacing[axis] * nodes for axis in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    weight = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    return float(np.sum(weight * integrand(x, y, z)) / 8.0)


def _depolarization_factors(a: float, b: float, c: float) -> np.ndarray:
    def factor(axis_length: float) -> float:
        def integrand(s: float) -> float:
            return 1.0 / (
                (axis_length**2 + s) * np.sqrt((a**2 + s) * (b**2 + s) * (c**2 + s))
            )

        value, _ = quad(integrand, 0.0, np.inf, limit=200)
        return 0.5 * a * b * c * value

    return np.asarray([factor(a), factor(b), factor(c)])


def _self_cell_inverse_distance(spacing: np.ndarray, order: int = 48) -> float:
    """Cell average of ``1/r`` about the cell center through its face pyramids.

    In spherical coordinates the pyramid over a face at distance ``d`` gives
    ``∫ 1/r dV = (d/2) ∫_face dA/|x|``, a smooth face integral.
    """
    nodes, weights = leggauss(order)
    half = 0.5 * spacing
    total = 0.0
    for axis in range(3):
        first, second = [other for other in range(3) if other != axis]
        u = half[first] * nodes
        v = half[second] * nodes
        uu, vv = np.meshgrid(u, v, indexing="ij")
        face = np.sum(
            weights[:, None] * weights[None, :] / np.sqrt(half[axis] ** 2 + uu**2 + vv**2)
        ) * (half[first] * half[second])
        total += 2.0 * 0.5 * half[axis] * face
    return total / float(np.prod(spacing))


def test_single_cell_potential_and_field_match_cell_quadrature() -> None:
    """A unit density in one cell produces the exact cell-averaged Coulomb kernel."""
    shape = (4, 5, 3)
    grid = _cell_grid(shape, (-1.0, -1.0, -1.0), (1.0, 2.0, 0.5))
    plan = phx.operators.FreeSpaceConvolutionPlan("coulomb-igf", grid, gradient=True)
    spacing = np.asarray(plan.spacing)
    source_index = (1, 3, 2)
    source = jnp.zeros(shape).at[source_index].set(1.0 / float(np.prod(spacing)))
    result = plan.convolve(source)
    assert bool(result.finite)
    assert result.gradient is not None
    field = np.asarray(result.field)
    gradient = np.asarray(result.gradient)
    for target_index in [(0, 0, 0), (3, 1, 2), (1, 4, 0), (1, 3, 2)]:
        offset = (np.asarray(target_index) - np.asarray(source_index)) * spacing
        expected = (
            _self_cell_inverse_distance(spacing) / (4.0 * np.pi)
            if target_index == source_index
            else _cell_average(
                offset,
                spacing,
                lambda x, y, z: 1.0 / (4.0 * np.pi * np.sqrt(x * x + y * y + z * z)),
            )
        )
        np.testing.assert_allclose(field[target_index], expected, rtol=1e-9, atol=1e-13)
        for axis in range(3):
            expected_gradient = _cell_average(
                offset,
                spacing,
                lambda x, y, z, axis=axis: (
                    -(x, y, z)[axis] / (4.0 * np.pi * (x * x + y * y + z * z) ** 1.5)
                ),
            )
            np.testing.assert_allclose(
                gradient[target_index + (axis,)], expected_gradient, rtol=1e-8, atol=1e-12
            )


def test_uniform_ellipsoid_interior_field_is_linear_with_depolarization_factors() -> None:
    """Inside a homogeneous ellipsoid ∇φ = −ρ (A_x x, A_y y, A_z z) with Σ A_i = 1."""
    semi_axes = (1.0, 0.7, 0.5)
    shape = (36, 28, 20)
    lower = tuple(-1.25 * axis for axis in semi_axes)
    upper = tuple(1.25 * axis for axis in semi_axes)
    grid = _cell_grid(shape, lower, upper)
    plan = phx.operators.FreeSpaceConvolutionPlan("coulomb-igf", grid, gradient=True)
    spacing = np.asarray(plan.spacing)
    centers = np.asarray(grid.points).reshape(shape + (3,))
    # Cell-averaged indicator through 4³ subcell sampling.
    offsets = (np.arange(4) + 0.5) / 4.0 - 0.5
    fill = np.zeros(shape)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                point = centers + np.asarray([dx, dy, dz]) * spacing
                fill += np.sum((point / np.asarray(semi_axes)) ** 2, axis=-1) <= 1.0
    density = 3.0 * fill / offsets.size**3
    result = plan.convolve(density)
    assert result.gradient is not None
    gradient = np.asarray(result.gradient)
    factors = _depolarization_factors(*semi_axes)
    np.testing.assert_allclose(np.sum(factors), 1.0, rtol=1e-8)
    interior = np.sum((centers / np.asarray(semi_axes)) ** 2, axis=-1) <= 0.25
    for axis in range(3):
        coordinate = centers[..., axis][interior]
        slope = np.sum(coordinate * gradient[..., axis][interior]) / np.sum(coordinate**2)
        np.testing.assert_allclose(slope, -3.0 * factors[axis], rtol=2e-2)
        residual = gradient[..., axis][interior] - slope * coordinate
        assert np.max(np.abs(residual)) < 2e-2 * 3.0 * factors[axis] * semi_axes[axis]


def test_gaussian_charge_field_matches_closed_form_on_axis() -> None:
    """E_r of a spherical Gaussian is Q[erf(r/√2σ) − √(2/π)(r/σ)e^{−r²/2σ²}]/(4π r²)."""
    sigma = 1.0
    count = 33
    half = 4.0 * sigma
    grid = _cell_grid((count,) * 3, (-half,) * 3, (half,) * 3)
    plan = phx.operators.FreeSpaceConvolutionPlan("coulomb-igf", grid, gradient=True)
    centers = np.asarray(grid.points).reshape((count,) * 3 + (3,))
    radius_squared = np.sum(centers**2, axis=-1)
    total_charge = 2.5
    density = (
        total_charge
        / ((2.0 * np.pi) ** 1.5 * sigma**3)
        * np.exp(-0.5 * radius_squared / sigma**2)
    )
    result = plan.convolve(density)
    assert result.gradient is not None
    middle = count // 2
    x = centers[:, middle, middle, 0]
    off_center = np.arange(count) != middle
    r = np.abs(x[off_center])
    radial = (
        total_charge
        / (4.0 * np.pi * r**2)
        * (
            erf(r / (np.sqrt(2.0) * sigma))
            - np.sqrt(2.0 / np.pi) * (r / sigma) * np.exp(-0.5 * r**2 / sigma**2)
        )
    )
    expected_field = radial * np.sign(x[off_center])
    field = -np.asarray(result.gradient)[:, middle, middle, 0]
    # Point-sampled density on h = σ/4.1: second-order error ≈ 0.18 (h/σ)² ≈ 1.1 %.
    tolerance = 2e-2 * np.max(np.abs(expected_field))
    np.testing.assert_allclose(field[off_center], expected_field, atol=tolerance)
    np.testing.assert_allclose(field[middle], 0.0, atol=1e-12 * np.max(np.abs(field)))
    potential = np.asarray(result.field)[:, middle, middle]
    expected_potential = (
        total_charge / (4.0 * np.pi * r) * erf(r / (np.sqrt(2.0) * sigma))
    )
    np.testing.assert_allclose(potential[off_center], expected_potential, rtol=5e-3)
    np.testing.assert_allclose(
        potential[middle],
        total_charge / (4.0 * np.pi * sigma) * np.sqrt(2.0 / np.pi),
        rtol=5e-3,
    )


def _triaxial_gaussian_field(points: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Unit-charge Gaussian field (ε₀ = 1) by the one-dimensional ellipsoidal integral.

    ``E_i(x) = (4π^{3/2})⁻¹ ∫₀^∞ 2 x_i/(2σ_i² + t) exp(−Σ x_k²/(2σ_k² + t))
    / Π √(2σ_k² + t) dt``.
    """
    doubled = 2.0 * sigmas**2
    field = np.zeros(points.shape)
    for index, point in enumerate(points):
        for axis in range(3):

            def integrand(t: float, point: np.ndarray = point, axis: int = axis) -> float:
                shifted = doubled + t
                return float(
                    2.0
                    * point[axis]
                    / shifted[axis]
                    * np.exp(-np.sum(point**2 / shifted))
                    / np.sqrt(np.prod(shifted))
                )

            value, _ = quad(integrand, 0.0, np.inf, limit=200)
            field[index, axis] = value / (4.0 * np.pi**1.5)
    return field


def test_integrated_kernel_converges_faster_than_point_green_function() -> None:
    """On cells of aspect ratio 12 the IGF field is second order; the point kernel is not."""
    sigmas = np.asarray([1.0, 1.0, 12.0])
    half = 4.0 * sigmas
    counts = (8, 16, 32)
    errors: dict[str, list[float]] = {"newton-igf": [], "newton-softened": []}
    for count in counts:
        grid = _cell_grid((count,) * 3, tuple(-half), tuple(half))
        centers = np.asarray(grid.points).reshape((count,) * 3 + (3,))
        density = np.exp(-0.5 * np.sum((centers / sigmas) ** 2, axis=-1)) / (
            (2.0 * np.pi) ** 1.5 * np.prod(sigmas)
        )
        middle = count // 2
        # Transverse row and diagonal through the bunch center (Qiang et al. 2006).
        samples = [(i, middle, middle) for i in range(count)] + [
            (i, i, middle) for i in range(count)
        ]
        # E = −∇φ with φ = −field/(4π) for the Newton kernel −1/r.
        reference = (
            4.0
            * np.pi
            * _triaxial_gaussian_field(
                np.asarray([centers[index] for index in samples]), sigmas
            )
        )
        spacing = float(np.min(2.0 * half / count))
        plans = {
            "newton-igf": phx.operators.FreeSpaceConvolutionPlan(
                "newton-igf", grid, gradient=True
            ),
            "newton-softened": phx.operators.FreeSpaceConvolutionPlan(
                "newton-softened", grid, softening=1e-7 * spacing, gradient=True
            ),
        }
        for name, plan in plans.items():
            result = plan.convolve(density)
            assert bool(result.finite)
            assert result.gradient is not None
            gradient = np.asarray(result.gradient)
            sampled = np.asarray([gradient[index] for index in samples])
            errors[name].append(
                float(np.max(np.abs(sampled - reference)) / np.max(np.abs(reference)))
            )
    igf = np.asarray(errors["newton-igf"])
    point = np.asarray(errors["newton-softened"])
    igf_order = np.log2(igf[0] / igf[-1]) / (len(counts) - 1)
    point_order = np.log2(point[0] / point[-1]) / (len(counts) - 1)
    assert np.all(igf < point)
    assert igf_order > 1.8
    assert point_order < 1.3
    assert igf[-1] < 1.0e-2


@pytest.mark.parametrize(
    ("dimension", "count"), [(2, 6), (3, 4)], ids=["planar", "volumetric"]
)
def test_biot_savart_velocity_and_gradient_match_direct_sums(
    dimension: int, count: int
) -> None:
    grid = _cell_grid((count,) * dimension, (0.0,) * dimension, (1.0,) * dimension)
    plan = phx.operators.FreeSpaceConvolutionPlan("biot-savart", grid, gradient=True)
    rng = np.random.default_rng(3)
    source = rng.normal(size=plan.source_shape)
    result = plan.convolve(source)
    centers = np.asarray(grid.points).reshape((-1, dimension))
    measure = (1.0 / count) ** dimension
    flat_source = source.reshape((centers.shape[0],) + source.shape[dimension:])
    velocity = np.zeros((centers.shape[0], dimension))
    gradient = np.zeros((centers.shape[0], dimension, dimension))
    for index in range(centers.shape[0]):
        displacement = centers[index] - centers
        squared = np.sum(displacement**2, axis=-1)
        mask = squared > 0.0
        safe = np.where(mask, squared, 1.0)
        if dimension == 2:
            rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
            rotated = displacement @ rotation.T
            kernel = rotated / (2.0 * np.pi * safe[:, None])
            kernel_gradient = (
                rotation * safe[:, None, None]
                - 2.0 * rotated[:, :, None] * displacement[:, None, :]
            ) / (2.0 * np.pi * safe[:, None, None] ** 2)
            kernel[~mask] = 0.0
            kernel_gradient[~mask] = 0.0
            velocity[index] = np.sum(kernel * flat_source[:, None] * measure, axis=0)
            gradient[index] = np.sum(
                kernel_gradient * flat_source[:, None, None] * measure, axis=0
            )
        else:
            kernel = displacement / (4.0 * np.pi * safe[:, None] ** 1.5)
            kernel_gradient = (
                np.eye(3) * safe[:, None, None]
                - 3.0 * displacement[:, :, None] * displacement[:, None, :]
            ) / (4.0 * np.pi * safe[:, None, None] ** 2.5)
            kernel[~mask] = 0.0
            kernel_gradient[~mask] = 0.0
            velocity[index] = np.sum(np.cross(flat_source, kernel) * measure, axis=0)
            gradient[index] = np.sum(
                np.cross(flat_source[:, :, None], kernel_gradient, axis=1) * measure,
                axis=0,
            )
    assert result.gradient is not None
    np.testing.assert_allclose(
        np.asarray(result.field).reshape(velocity.shape), velocity, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result.gradient).reshape(gradient.shape), gradient, atol=1e-11
    )


def test_softened_newton_kernel_matches_direct_sum_and_excludes_self_cell() -> None:
    count = 8
    grid = _cell_grid((count,), (0.0,), (1.0,))
    plan = phx.operators.FreeSpaceConvolutionPlan("newton-softened", grid, softening=0.05)
    rng = np.random.default_rng(5)
    density = rng.normal(size=(count,))
    result = plan.convolve(density)
    x = np.asarray(grid.points).reshape(-1)
    expected = np.asarray(
        [
            np.sum(
                np.where(x[i] != x, -1.0 / np.sqrt((x[i] - x) ** 2 + 0.05**2), 0.0)
                * density
                / count
            )
            for i in range(count)
        ]
    )
    np.testing.assert_allclose(np.asarray(result.field), expected, atol=1e-13)
    assert result.gradient is None


def test_kernel_and_grid_admissibility() -> None:
    cells = _cell_grid((4, 4), (0.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError):
        phx.operators.FreeSpaceConvolutionPlan("coulomb-igf", cells)
    with pytest.raises(ValueError):
        phx.operators.FreeSpaceConvolutionPlan("newton-softened", cells)
    with pytest.raises(ValueError):
        phx.operators.FreeSpaceConvolutionPlan("biot-savart", cells, softening=0.1)
    periodic = TensorGridPlan(
        (UniformCellAxisSpec(4, periodic=True), UniformCellAxisSpec(4))
    ).prepare(np.asarray([[0.0, 0.0], [1.0, 1.0]]))
    with pytest.raises(ValueError):
        phx.operators.FreeSpaceConvolutionPlan("biot-savart", periodic)
    points = TensorGridPlan((UniformAxisSpec(5),) * 3).prepare(
        np.asarray([[0.0] * 3, [1.0] * 3])
    )
    with pytest.raises(ValueError):
        phx.operators.FreeSpaceConvolutionPlan("coulomb-igf", points)
    plan = phx.operators.FreeSpaceConvolutionPlan(
        "newton-softened", points, softening=0.1
    )
    with pytest.raises(ValueError):
        plan.convolve(np.zeros((5, 5)))
