#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

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


def _square(count: Any) -> Any:
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


def _annulus(inner: Any, outer: Any, rings: Any, sectors: Any) -> Any:
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


def _extension(
    points: Any, triangles: Any, boundary: Any, route: Any, **controls: Any
) -> Any:
    return D.FiniteElementMotionExtension(
        points,
        (("triangle", triangles),),
        boundary,
        policy=D.FiniteElementMeshMotionPolicy(route=route, **controls),
    )


def _orientation_preserved(points: Any, triangles: Any, displacement: Any) -> Any:
    validity = D.MotionValidityPlan(points, (("triangle", triangles),), policy=_POSITIVE)
    return validity.evaluate(points + np.asarray(displacement))


def _certified(points: Any, triangles: Any, displacement: Any) -> Any:
    mesh = D.CellMesh.from_triangles(points + np.asarray(displacement), triangles)
    return D.certify_cell_geometry_validity(mesh).all_certified


def test_mesh_motion_routes_scenario_1() -> None:
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


class _GaussianMonitor(D.AbstractMeshMonitor):
    amplitude: jax.Array

    def __call__(self, points: Any) -> Any:
        distance = jnp.sum((points - jnp.asarray((0.3, 0.3))) ** 2, axis=-1)
        return 1.0 + self.amplitude * jnp.exp(-distance / 0.02)


def test_mmpde_concentrates_vertices_toward_the_monitor_peak() -> None:
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

    def spread(amplitude: Any) -> Any:
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


def test_repeated_mmpde_solves_reuse_one_compiled_program() -> None:
    """Eager solves compile once per extension; new monitor values reuse it."""

    points, triangles, boundary = _square(6)
    fixed = np.zeros_like(points[boundary])
    extension = _extension(points, triangles, boundary, Route.MMPDE)
    extension.extend(fixed, monitor=_GaussianMonitor(jnp.asarray(1.0)))
    compiles: list[str] = []

    def listener(event: str, duration_secs: float, **kwargs: str | int) -> None:
        del duration_secs, kwargs
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        results = [
            extension.extend(fixed, monitor=_GaussianMonitor(jnp.asarray(amplitude)))
            for amplitude in (2.0, 5.0, 10.0)
        ]
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    assert compiles == []
    traced = eqx.filter_jit(extension.extend)(
        fixed, monitor=_GaussianMonitor(jnp.asarray(10.0))
    )
    assert all(bool(result.successful) for result in results)
    np.testing.assert_allclose(
        results[-1].displacement, traced.displacement, rtol=0.0, atol=1.0e-12
    )
    assert not np.allclose(results[0].displacement, results[1].displacement)


def test_winslow_boundary_derivative_matches_finite_differences() -> None:
    points, triangles, boundary = _square(4)
    edge = points[boundary]
    direction = np.stack((0.2 * edge[:, 1], 0.1 * np.sin(np.pi * edge[:, 0])), 1)
    extension = _extension(points, triangles, boundary, Route.WINSLOW)

    def interior_energy(scale: Any) -> Any:
        return jnp.sum(extension.extend(scale * direction).displacement ** 3)

    gradient = jax.grad(interior_energy)(jnp.asarray(1.0))
    step = 1.0e-4
    finite_difference = (
        interior_energy(jnp.asarray(1.0 + step))
        - interior_energy(jnp.asarray(1.0 - step))
    ) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1.0e-5)


def test_mesh_motion_routes_scenario_2() -> None:
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


@pytest.mark.parametrize(
    "route", (Route.HARMONIC, Route.LINEAR_ELASTICITY, Route.WINSLOW, Route.MMPDE)
)
def test_every_route_converges_for_tiny_boundary_moves(route: Any) -> None:
    """Solve criteria are relative to the motion, not absolute coordinate floors.

    Every route solves for the displacement normalized by its largest boundary
    entry (reported as `displacement_scale`), so a 1e-9 move converges like an
    O(0.1) one and reports its residual in coordinate units.
    """
    points, triangles, boundary = _square(4)
    edge = points[boundary]
    direction = np.stack((0.2 * edge[:, 1], 0.1 * np.sin(np.pi * edge[:, 0])), 1)
    interior = np.setdiff1d(np.arange(points.shape[0]), boundary)
    extension = _extension(points, triangles, boundary, route)
    # A uniform monitor: MMPDE then moves the interior only with the boundary.
    monitor = _GaussianMonitor(jnp.asarray(0.0)) if route is Route.MMPDE else None
    linear = route in (Route.HARMONIC, Route.LINEAR_ELASTICITY)
    amplitudes = (1.0e-9, 1.0e-5, 1.0e-1) if linear else (1.0e-5, 1.0e-1)
    responses = []
    for amplitude in amplitudes:
        result = extension.extend(amplitude * direction, monitor=monitor)
        assert bool(result.successful), (amplitude, np.asarray(result.status))
        np.testing.assert_allclose(
            result.displacement_scale, amplitude * np.max(np.abs(direction)), rtol=1e-15
        )
        # The residual scales with the motion (MMPDE's carries 1 / tau).
        assert float(result.residual_norm) <= 1.0e-6 * float(result.displacement_scale)
        response = np.asarray(result.displacement)[interior] / amplitude
        assert float(np.max(np.abs(response))) > 1.0e-3
        responses.append(response)
    if linear:
        # Linear routes are homogeneous in the boundary data.
        for response in responses[:-1]:
            np.testing.assert_allclose(response, responses[-1], rtol=0.0, atol=1e-9)
    else:
        # A small move follows the route's linearization.
        scale = float(np.max(np.abs(responses[-1])))
        assert float(np.max(np.abs(responses[0] - responses[-1]))) < 0.2 * scale


def _periodic_strip(count: Any) -> Any:
    """Lifted unit square, periodic in x: the x = 1 column twins the x = 0 one."""

    points, triangles, _ = _square(count)
    kept = np.flatnonzero(points[:, 0] == 0.0)
    twin = np.flatnonzero(points[:, 0] == 1.0)
    kept = kept[np.argsort(points[kept, 1])]
    twin = twin[np.argsort(points[twin, 1])]
    translations = np.broadcast_to((1.0, 0.0), (kept.size, 2))
    boundary = np.flatnonzero((points[:, 1] == 0.0) | (points[:, 1] == 1.0))
    return points, triangles, boundary, (kept, twin, translations)


def test_periodic_extension_solves_harmonic_motion_on_the_quotient() -> None:
    count = 6
    points, triangles, boundary, pairs = _periodic_strip(count)
    kept, twin, _ = pairs
    extension = D.FiniteElementMotionExtension(
        points,
        (("triangle", triangles),),
        boundary,
        policy=D.FiniteElementMeshMotionPolicy(route=Route.HARMONIC),
        periodic_vertex_pairs=pairs,
    )
    # Bottom wall fixed; the top wall slides by a lattice-periodic profile.
    prescribed = np.zeros((boundary.size, 2))
    top = points[boundary, 1] == 1.0
    prescribed[top, 0] = 0.08 * np.sin(2.0 * np.pi * points[boundary[top], 0])
    prescribed[top, 1] = 0.05 * np.cos(2.0 * np.pi * points[boundary[top], 0])

    result = eqx.filter_jit(extension.extend)(prescribed)

    assert bool(result.successful)
    displacement = np.asarray(result.displacement)
    np.testing.assert_array_equal(displacement[twin], displacement[kept])
    # A prescribed class takes its first member's value; corner twins agree
    # with their own prescription to rounding (sin(0) against sin(2 pi)).
    np.testing.assert_allclose(displacement[boundary], prescribed, rtol=0.0, atol=1e-15)

    # Independent reference: weld the twins into the quotient mesh and solve
    # its graph Laplacian (edge weights 1/length) densely.
    weld = np.arange(points.shape[0])
    weld[twin] = kept
    edges = np.sort(
        np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        ),
        axis=1,
    )
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    welded = np.sort(weld[edges], axis=1)
    welded, unique = np.unique(welded, axis=0, return_index=True)
    weights = 1.0 / lengths[unique]
    laplacian = np.zeros((points.shape[0], points.shape[0]))
    for (a, b), weight in zip(welded, weights, strict=True):
        laplacian[a, a] += weight
        laplacian[b, b] += weight
        laplacian[a, b] -= weight
        laplacian[b, a] -= weight
    quotient = np.unique(weld)
    fixed = np.unique(weld[boundary])
    free = np.setdiff1d(quotient, fixed)
    values = np.zeros((points.shape[0], 2))
    values[weld[boundary]] = prescribed
    reference = np.linalg.solve(
        laplacian[np.ix_(free, free)],
        -laplacian[np.ix_(free, fixed)] @ values[fixed],
    )
    np.testing.assert_allclose(displacement[free], reference, rtol=0.0, atol=1e-9)
    # Seam columns move: the coupling is exercised, not a fixed seam.
    assert float(np.max(np.abs(displacement[kept]))) > 1.0e-3


@pytest.mark.parametrize("route", (Route.LINEAR_ELASTICITY, Route.WINSLOW))
def test_periodic_extension_keeps_twins_exact_on_every_route(route: Any) -> None:
    points, triangles, boundary, pairs = _periodic_strip(5)
    kept, twin, _ = pairs
    extension = D.FiniteElementMotionExtension(
        points,
        (("triangle", triangles),),
        boundary,
        policy=D.FiniteElementMeshMotionPolicy(route=route),
        periodic_vertex_pairs=pairs,
    )
    prescribed = np.zeros((boundary.size, 2))
    top = points[boundary, 1] == 1.0
    prescribed[top, 0] = 0.06 * np.sin(2.0 * np.pi * points[boundary[top], 0])

    result = eqx.filter_jit(extension.extend)(prescribed)

    assert bool(result.successful)
    displacement = np.asarray(result.displacement)
    np.testing.assert_array_equal(displacement[twin], displacement[kept])
    assert float(np.max(np.abs(displacement[kept]))) > 1.0e-4
    assert bool(_orientation_preserved(points, triangles, displacement).valid)


def test_periodic_extension_refuses_non_lattice_twins_and_torn_prescriptions() -> None:
    points, triangles, boundary, pairs = _periodic_strip(4)
    kept, twin, translations = pairs
    with pytest.raises(ValueError, match="lattice translates"):
        D.FiniteElementMotionExtension(
            points,
            (("triangle", triangles),),
            boundary,
            periodic_vertex_pairs=(kept, twin, translations * 0.5),
        )
    extension = D.FiniteElementMotionExtension(
        points,
        (("triangle", triangles),),
        boundary,
        periodic_vertex_pairs=pairs,
    )
    # The two lifted copies of a wall corner disagree: not a periodic motion.
    torn = np.zeros((boundary.size, 2))
    torn[np.flatnonzero(boundary == twin[-1]), 0] = 0.01
    assert not bool(extension.extend(torn).successful)
