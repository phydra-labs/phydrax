#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
Route = D.FiniteElementMeshMotionRoute
_POSITIVE = D.MotionValidityPolicy(
    minimum_absolute_jacobian=0.0,
    minimum_relative_jacobian=0.0,
    maximum_displacement_fraction=None,
)


def _square(count):
    axis = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    index = np.arange(points.shape[0]).reshape(count + 1, count + 1)
    a = index[:-1, :-1].ravel()
    b = index[1:, :-1].ravel()
    c = index[1:, 1:].ravel()
    d = index[:-1, 1:].ravel()
    triangles = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    boundary = np.flatnonzero(np.any((points == 0.0) | (points == 1.0), axis=1))
    return points, triangles, boundary


def _annulus(inner, outer, rings, sectors):
    """Graded polar triangulation: the smallest cells hug the inclusion."""

    radii = inner + (outer - inner) * np.linspace(0.0, 1.0, rings + 1) ** 2
    angles = 2.0 * np.pi * np.arange(sectors) / sectors
    points = np.stack(
        (
            np.outer(radii, np.cos(angles)).ravel(),
            np.outer(radii, np.sin(angles)).ravel(),
        ),
        axis=1,
    )
    index = np.arange(points.shape[0]).reshape(rings + 1, sectors)
    following = np.roll(index, -1, axis=1)
    a = index[:-1].ravel()
    b = following[:-1].ravel()
    c = following[1:].ravel()
    d = index[1:].ravel()
    triangles = np.concatenate((np.stack((a, d, c), 1), np.stack((a, c, b), 1)))
    return points, triangles, index[0], index[-1]


def _extension(points, triangles, boundary, route, **controls):
    return D.FiniteElementMotionExtension(
        points,
        (("triangle", triangles),),
        boundary,
        policy=D.FiniteElementMeshMotionPolicy(route=route, **controls),
    )


def _orientation_preserved(points, triangles, displacement):
    validity = D.MotionValidityPlan(points, (("triangle", triangles),), policy=_POSITIVE)
    return validity.evaluate(points + np.asarray(displacement))


def _certified(points, triangles, displacement):
    mesh = D.CellMesh.from_triangles(points + np.asarray(displacement), triangles)
    return D.certify_cell_geometry_validity(mesh).all_certified


def test_harmonic_route_keeps_a_convex_planar_image_valid():
    points, triangles, boundary = _square(6)
    edge = points[boundary]
    # Straight edges of the square map to a convex quadrilateral.
    image = np.stack(
        (edge[:, 0] + 0.3 * edge[:, 1], edge[:, 1] * (1.0 + 0.2 * edge[:, 0])), 1
    )
    extension = _extension(points, triangles, boundary, Route.HARMONIC)

    result = eqx.filter_jit(extension.extend)(image - edge)

    assert bool(result.successful)
    np.testing.assert_allclose(result.displacement[boundary], image - edge, atol=0.0)
    evidence = _orientation_preserved(points, triangles, result.displacement)
    assert bool(evidence.valid)
    assert _certified(points, triangles, result.displacement)


def test_stiffened_elasticity_carries_a_rotating_inclusion_without_inversion():
    points, triangles, inner, outer = _annulus(0.25, 1.0, 6, 24)
    boundary = np.sort(np.concatenate((inner, outer)))
    angle = np.deg2rad(40.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    target = points[boundary].copy()
    is_inner = np.isin(boundary, inner)
    target[is_inner] = points[boundary][is_inner] @ rotation.T

    minima = {}
    for chi in (0.0, 1.0):
        extension = _extension(
            points,
            triangles,
            boundary,
            Route.LINEAR_ELASTICITY,
            stiffening_exponent=chi,
        )
        result = extension.extend(target - points[boundary])
        assert bool(result.successful)
        evidence = _orientation_preserved(points, triangles, result.displacement)
        minima[chi] = float(evidence.minimum_relative_jacobian)
        if chi == 1.0:
            assert bool(evidence.valid)
            assert _certified(points, triangles, result.displacement)

    # Stiff small cells near the inclusion rotate nearly rigidly.
    assert minima[1.0] > minima[0.0]


def test_winslow_keeps_a_distorted_convex_image_valid():
    points, triangles, boundary = _square(8)
    edge = points[boundary]
    # Convex trapezoid image with boundary vertices crowded toward one corner.
    crowded = edge**2
    image = np.stack(
        (crowded[:, 0] * (1.0 - 0.3 * crowded[:, 1]), crowded[:, 1] * 1.2), axis=1
    )
    extension = _extension(points, triangles, boundary, Route.WINSLOW)

    result = eqx.filter_jit(extension.extend)(image - edge)

    assert bool(result.successful)
    assert bool(_orientation_preserved(points, triangles, result.displacement).valid)
    assert _certified(points, triangles, result.displacement)


class _GaussianMonitor(eqx.Module):
    amplitude: jax.Array

    def __call__(self, points):
        distance = jnp.sum((points - jnp.asarray((0.3, 0.3))) ** 2, axis=-1)
        return 1.0 + self.amplitude * jnp.exp(-distance / 0.02)


def test_mmpde_concentrates_vertices_toward_the_monitor_peak():
    points, triangles, boundary = _square(6)
    extension = _extension(points, triangles, boundary, Route.MMPDE)
    fixed = np.zeros_like(points[boundary])

    result = eqx.filter_jit(extension.extend)(
        fixed, monitor=_GaussianMonitor(jnp.asarray(10.0))
    )

    assert bool(result.successful)
    moved = points + np.asarray(result.displacement)
    interior = np.setdiff1d(np.arange(points.shape[0]), boundary)
    peak = np.asarray((0.3, 0.3))
    before = np.linalg.norm(points[interior] - peak, axis=1)
    after = np.linalg.norm(moved[interior] - peak, axis=1)
    assert np.mean(after) < np.mean(before) - 0.01
    assert bool(_orientation_preserved(points, triangles, result.displacement).valid)

    def spread(amplitude):
        displacement = extension.extend(
            fixed, monitor=_GaussianMonitor(amplitude)
        ).displacement
        return jnp.sum(displacement**2)

    gradient = jax.grad(spread)(jnp.asarray(10.0))
    step = 1.0e-3
    finite_difference = (
        spread(jnp.asarray(10.0 + step)) - spread(jnp.asarray(10.0 - step))
    ) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1.0e-4)


def test_winslow_boundary_derivative_matches_finite_differences():
    points, triangles, boundary = _square(4)
    edge = points[boundary]
    direction = np.stack((0.2 * edge[:, 1], 0.1 * np.sin(np.pi * edge[:, 0])), 1)
    extension = _extension(points, triangles, boundary, Route.WINSLOW)

    def interior_energy(scale):
        return jnp.sum(extension.extend(scale * direction).displacement ** 3)

    gradient = jax.grad(interior_energy)(jnp.asarray(1.0))
    step = 1.0e-4
    finite_difference = (
        interior_energy(jnp.asarray(1.0 + step))
        - interior_energy(jnp.asarray(1.0 - step))
    ) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1.0e-5)


def test_route_selection_is_explicit_and_complete():
    points, triangles, boundary = _square(2)
    with pytest.raises(ValueError, match="PRESCRIBED"):
        _extension(points, triangles, boundary, Route.PRESCRIBED)
    harmonic = _extension(points, triangles, boundary, Route.HARMONIC)
    with pytest.raises(ValueError, match="monitor"):
        harmonic.extend(np.zeros_like(points[boundary]), monitor=_GaussianMonitor(1.0))
    mmpde = _extension(points, triangles, boundary, Route.MMPDE)
    with pytest.raises(ValueError, match="monitor"):
        mmpde.extend(np.zeros_like(points[boundary]))
    prescribed = _extension(
        points, triangles, np.arange(points.shape[0]), Route.PRESCRIBED
    )
    shift = np.full(points.shape, 0.1)
    np.testing.assert_array_equal(prescribed.extend(shift).displacement, shift)


def test_motion_validity_rejects_inversion_and_small_relative_jacobians():
    points, triangles, _ = _square(2)
    validity = D.MotionValidityPlan(points, (("triangle", triangles),))
    center = 4
    squeezed = points.copy()
    squeezed[center] = (0.99, 0.5)
    folded = points.copy()
    folded[center] = (1.2, 0.5)

    assert bool(validity.evaluate(points).valid)
    small = validity.evaluate(squeezed)
    assert int(small.status) & int(D.MotionValidityStatus.JACOBIAN_TOO_SMALL)
    assert bool(small.orientation_preserved)
    inverted = validity.evaluate(folded)
    assert int(inverted.status) & int(D.MotionValidityStatus.ORIENTATION_CHANGED)
    assert float(inverted.minimum_relative_jacobian) < 0.0
