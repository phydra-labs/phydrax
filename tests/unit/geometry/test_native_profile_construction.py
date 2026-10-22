"""Native profile construction: solved sketches, rational sweeps, and offsets.

Oracles are closed-form and independent of the construction: the slot is a
2 x 1 rectangle with two radius-1/2 end caps and a radius-1/5 hole; the
rational quadratic NURBS ellipse with corner weights sqrt(1/2) represents the
ellipse ``(a cos t, b sin t)`` up to binary rounding of that weight, so its
area is ``pi a b`` and its perimeter is the trapezoid sum of the analytic
speed (spectrally convergent for this periodic integrand). Revolved volumes and
areas follow Pappus; the parallel body of a convex meridian at signed
distance ``d`` within its reach has area ``A + L d + pi d^2`` and, being
symmetric about its centroid line, centroid radius ``R``.
"""

import math
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from itertools import permutations
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._external_resource import ResourceLimits
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry.brep import _correspondence as correspondence_owner
from phydrax.geometry.brep import BRepQueryBudget, BRepQueryResourceError
from phydrax.geometry.brep._constructors import reversed_bspline
from phydrax.geometry.brep._correspondence import certify_curve_surface
from phydrax.geometry.brep._measure import (
    face_quadrature,
    green_nodes,
    prepare_quadrature,
    surface_quadrature_u_breaks,
    trimmed_face_rules,
)
from phydrax.geometry.brep._patches import SurfaceIsoparametricCurve
from phydrax.geometry.brep._placed import PlacedSurface
from phydrax.meshing.qualification import rational_ellipse_profile, revolved_ellipse
from phydrax.units import METER, MILLIMETER, UnitDefinition
from tests._support.cad_models import native_offset_plane
from tests._support.patch_derivatives import (
    ANCHOR_ABSOLUTE_ERROR,
    assert_adjacent_order_partial,
    CarrierFamily,
    closed_form_partial,
    derivative_carrier,
    parameter_partial,
)


_CONTRACT = phx.SpatialCoordinateContract.si()
_SLOT_AREA = 2.0 + math.pi * (0.5**2 - 0.2**2)
_RADIUS, _A, _B = 3.0, 1.0, 0.5


def _slot_sketch() -> phx.geometry.Sketch:
    geometry = phx.geometry
    return geometry.Sketch(
        np.asarray(
            [
                [-0.9, -0.45],
                [1.0, -0.5],
                [1.0, 0.5],
                [-1.1, 0.55],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        lines=np.asarray([[0, 1], [2, 3]]),
        circle_centers=np.asarray([4, 5, 6]),
        circle_radii=np.asarray([0.45, 0.55, 0.25]),
        arcs=np.asarray([[0, 1, 2], [1, 3, 0]]),
        constraints=(
            geometry.FixedPoint(4, (1.0, 0.0)),
            geometry.FixedPoint(5, (-1.0, 0.0)),
            geometry.FixedPoint(6, (0.0, 0.0)),
            geometry.FixedPoint(1, (1.0, -0.5)),
            geometry.FixedPoint(2, (1.0, 0.5)),
            geometry.Radius(0, 0.5),
            geometry.Radius(1, 0.5),
            geometry.Radius(2, 0.2),
            geometry.Horizontal(0),
            geometry.Horizontal(1),
            geometry.PointDistance(0, 1, 2.0),
            geometry.PointDistance(2, 3, 2.0),
        ),
        feature_id="slot-sketch",
    )


def _ellipse_perimeter(a: float, b: float) -> float:
    angles = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    return float(np.mean(np.hypot(a * np.sin(angles), b * np.cos(angles))) * 2.0 * np.pi)


def _measures(model: phx.geometry.BRepModel) -> phx.geometry.brep.BRepMeasureResult:
    return phx.geometry.prepare_brep_query(model).measures


def test_profile_spline_reversal_retains_live_native_parameter_leaves() -> None:
    source = phx.geometry.BSplineCurve(
        [[1.0, 2.0], [3.0, -1.0], [4.0, 2.0]],
        [1.0, 0.75, 1.5],
        [2.0, 2.0, 3.5, 7.0, 7.0],
        1,
    )
    parameters = jnp.asarray([2.5, 4.0, 6.0])

    def reverse_values(curve: phx.geometry.BSplineCurve) -> jax.Array:
        return reversed_bspline(curve).evaluate(-parameters)

    expected = source.evaluate(parameters)
    np.testing.assert_allclose(reverse_values(source), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        eqx.filter_jit(reverse_values)(source), expected, rtol=0.0, atol=1e-12
    )
    reversed_source = eqx.filter_jit(reversed_bspline)(source)
    np.testing.assert_array_equal(reversed_source.knots, -source.knots[::-1])
    np.testing.assert_array_equal(reversed_source.weights, source.weights[::-1])
    twice = reversed_bspline(reversed_source)
    np.testing.assert_array_equal(twice.control_points, source.control_points)
    np.testing.assert_array_equal(twice.knots, source.knots)
    direction = jnp.asarray([[0.2, -0.1], [0.3, 0.4], [-0.5, 0.7]])

    def values(controls: jax.Array) -> jax.Array:
        return reverse_values(
            eqx.tree_at(lambda curve: curve.control_points, source, controls)
        )

    _, tangent = jax.jvp(values, (source.control_points,), (direction,))
    _, original_tangent = jax.jvp(
        lambda controls: eqx.tree_at(
            lambda curve: curve.control_points, source, controls
        ).evaluate(parameters),
        (source.control_points,),
        (direction,),
    )
    np.testing.assert_allclose(tangent, original_tangent, rtol=0.0, atol=1e-12)
    probe = jnp.asarray([[0.1, 0.4], [-0.2, 0.7], [0.3, -0.6]])
    _, transpose = jax.vjp(values, source.control_points)
    assert float(jnp.vdot(transpose(probe)[0], direction)) == pytest.approx(
        float(jnp.vdot(probe, tangent)), abs=1e-12
    )
    weight_direction = jnp.asarray([0.1, -0.2, 0.3])
    knot_direction = jnp.asarray([0.2, 0.2, -0.1, 0.4, 0.4])

    def leaf_values(weights: jax.Array, knots: jax.Array, *, reverse: bool) -> jax.Array:
        curve = eqx.tree_at(
            lambda value: (value.weights, value.knots), source, (weights, knots)
        )
        return reverse_values(curve) if reverse else curve.evaluate(parameters)

    primals = (source.weights, source.knots)
    tangents = (weight_direction, knot_direction)
    _, reversed_tangent = jax.jvp(
        lambda weights, knots: leaf_values(weights, knots, reverse=True),
        primals,
        tangents,
    )
    _, source_tangent = jax.jvp(
        lambda weights, knots: leaf_values(weights, knots, reverse=False),
        primals,
        tangents,
    )
    np.testing.assert_allclose(reversed_tangent, source_tangent, rtol=0.0, atol=1e-12)
    _, leaf_transpose = jax.vjp(
        lambda weights, knots: leaf_values(weights, knots, reverse=True), *primals
    )
    weight_cotangent, knot_cotangent = leaf_transpose(probe)
    assert float(
        jnp.vdot(weight_cotangent, weight_direction)
        + jnp.vdot(knot_cotangent, knot_direction)
    ) == pytest.approx(float(jnp.vdot(probe, reversed_tangent)), abs=1e-12)


@pytest.mark.parametrize(
    "plane",
    [
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfilePlane((4.0, -3.0, 2.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    ],
    ids=["native-frame", "placed-frame"],
)
def test_nonunit_spline_profile_frame_retains_source_bank_and_volume(
    plane: phx.geometry.ProfilePlane,
    tmp_path: Path,
) -> None:
    controls = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    source = phx.geometry.BSplineCurve(
        controls,
        np.full(5, 2.0),
        [2.0, 2.0, 2.5, 4.0, 5.0, 7.0, 7.0],
        1,
    )
    profile = phx.geometry.PlanarProfile(
        plane,
        phx.geometry.ProfileLoop(((0.0, 0.0),), (source,)),
    )
    height = 0.75
    model = phx.geometry.brep_extrusion(
        profile,
        height * plane.normal,
        coordinate_contract=_CONTRACT,
        source_id="nonunit-source-framed-extrusion",
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    patch = next(
        value
        for value in model.patches
        if isinstance(value, phx.geometry.ExtrusionSurface)
    )
    assert isinstance(patch.curve, phx.geometry.BSplineCurve)
    np.testing.assert_array_equal(patch.curve.control_points, plane.point(controls))
    np.testing.assert_array_equal(patch.curve.weights, source.weights)
    np.testing.assert_array_equal(patch.curve.knots, source.knots)
    parameters = jnp.asarray([2.25, 3.0, 4.5, 6.0])
    np.testing.assert_allclose(
        patch.evaluate(jnp.stack((parameters, jnp.full_like(parameters, 0.25)), axis=-1)),
        plane.point(np.asarray(source.evaluate(parameters))) + 0.25 * plane.normal,
        rtol=0.0,
        atol=1e-12,
    )
    # Independent rectangle area times physical extrusion height, not carrier
    # tessellation or a second evaluation of the source quadrature.
    assert float(_measures(model).solid_volumes[0]) == pytest.approx(
        2.0 * height, abs=1e-12
    )
    restored = phx.interchange.load_brep_archive(
        phx.interchange.save_brep_archive(model, tmp_path / "framed-source.phx").path
    )
    assert restored.model_id == model.model_id
    assert restored.source_id == model.source_id
    restored_patch = next(
        value
        for value in restored.patches
        if isinstance(value, phx.geometry.ExtrusionSurface)
    )
    assert eqx.tree_equal(restored_patch, patch)
    assert float(_measures(restored).solid_volumes[0]) == pytest.approx(
        2.0 * height, abs=1e-12
    )


@pytest.mark.parametrize("placed", [False, True], ids=["source", "placed-source"])
def test_native_green_measure_respects_original_c0_surface_knots(placed: bool) -> None:
    # The degree-one source traces this closed polygon.
    # Its unequal native span widths make an unsplit inner Green rule inexact.
    polygon = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.25, 0.25],
            [1.125, 0.75],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    curve = phx.geometry.BSplineCurve(
        np.concatenate((polygon, polygon[:1])),
        np.ones(7),
        [0.0, 0.0, 0.2, 1.0, 1.7, 2.0, 2.4, 3.0, 3.0],
        1,
    )
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop((tuple(polygon[0]),), (curve,)),
    )
    height = 0.5
    model = phx.geometry.brep_extrusion(
        profile,
        (0.0, 0.0, height),
        coordinate_contract=_CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
        source_id="source-native-green-c0-knots",
    )
    assert model.geometry is not None
    patches = model.patches
    if placed:
        rotation = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        patches = tuple(PlacedSurface(patch, rotation, np.zeros(3)) for patch in patches)
    # Shoelace independently computes the polygon's area; all vertices are
    # binary rationals, and it also isolates each face's signed divergence.
    area = 0.5 * sum(
        first[0] * last[1] - first[1] * last[0]
        for first, last in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    )
    expected = [0.0, area * height / 3.0, 2.0 * area * height / 3.0]
    from tests._support.coordinate_references import coordinate_reference_budget

    policy = phx.geometry.BRepQueryPolicy()
    with coordinate_reference_budget() as coordinate_budget:
        _, measures = prepare_quadrature(
            model, model.geometry, policy, coordinate_budget=coordinate_budget
        )
        assert float(measures.solid_volumes[0]) == pytest.approx(area * height, rel=1e-12)
        rules = trimmed_face_rules(model, policy, coordinate_budget=coordinate_budget)
        budget = BRepQueryBudget(
            policy.maximum_operations, policy.maximum_points, policy.maximum_scratch_bytes
        )
        for face, (parameters, weights) in enumerate(rules):
            charts, rule_weights = np.asarray(parameters), np.asarray(weights)
            if placed:
                bounds = np.asarray(model.parameter_bounds)[face]
                charts, rule_weights = green_nodes(
                    model.geometry,
                    face,
                    bounds,
                    policy.quadrature_order,
                    policy.quadrature_subdivisions,
                    surface_quadrature_u_breaks(
                        patches[face], bounds, budget, coordinate_budget=coordinate_budget
                    ),
                    budget,
                )
            points, vector_areas, _ = face_quadrature(
                patches[face], float(model.orientation[face]), charts, rule_weights
            )
            signed_volume = np.sum(points * vector_areas) / 3.0
            assert signed_volume == pytest.approx(expected[face], abs=1e-12)
        assert float(measures.solid_volume_error_estimates[0]) <= 1e-12
        with pytest.raises(BRepQueryResourceError) as refusal:
            prepare_quadrature(
                model,
                model.geometry,
                phx.geometry.BRepQueryPolicy(maximum_points=1),
                coordinate_budget=coordinate_budget,
            )
        assert refusal.value.resource == "points"
        assert refusal.value.remaining == 1


def _assert_live_sweep_derivatives(
    patch: phx.geometry.ExtrusionSurface | phx.geometry.RevolutionSurface,
) -> None:
    curve = patch.curve
    assert isinstance(curve, phx.geometry.BSplineCurve)
    parameters = jnp.asarray([0.125, 0.4], dtype=jnp.float64)
    direction = jnp.asarray([0.3, -0.2, 0.7], dtype=jnp.float64)
    delta = jnp.broadcast_to(direction, curve.control_points.shape)
    probe = jnp.asarray([0.2, 0.6, -0.1], dtype=jnp.float64)

    def point(controls: jax.Array) -> jax.Array:
        source = eqx.tree_at(lambda value: value.control_points, curve, controls)
        return eqx.tree_at(lambda value: value.curve, patch, source).evaluate(parameters)

    _, tangent = jax.jvp(point, (curve.control_points,), (delta,))
    if isinstance(patch, phx.geometry.ExtrusionSurface):
        expected = direction
    else:
        angle = float(parameters[0])
        axis = np.asarray(patch.axis_direction)
        value = np.asarray(direction)
        expected = (
            value * math.cos(angle)
            + np.cross(axis, value) * math.sin(angle)
            + axis * (axis @ value) * (1.0 - math.cos(angle))
        )
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(expected), rtol=0.0, atol=1e-12
    )
    gradient = jax.grad(lambda controls: jnp.vdot(probe, point(controls)))(
        curve.control_points
    )
    assert float(jnp.vdot(gradient, delta)) == pytest.approx(
        float(jnp.vdot(probe, tangent)), abs=1e-12
    )
    step = 1e-6
    difference = (
        point(curve.control_points + step * delta)
        - point(curve.control_points - step * delta)
    ) / (2 * step)
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(difference), rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        jax.jit(point)(curve.control_points),
        point(curve.control_points),
        rtol=0.0,
        atol=1e-12,
    )

    def source_point(weights: jax.Array, knots: jax.Array) -> jax.Array:
        source = eqx.tree_at(
            lambda value: (value.weights, value.knots), curve, (weights, knots)
        )
        return eqx.tree_at(lambda value: value.curve, patch, source).evaluate(parameters)

    weight_direction = jnp.linspace(-0.2, 0.3, curve.weights.size)
    # Translating all native knots preserves every authored multiplicity.
    knot_direction = jnp.full_like(curve.knots, 0.1)
    primals, tangents = (curve.weights, curve.knots), (weight_direction, knot_direction)
    value, source_tangent = jax.jvp(source_point, primals, tangents)
    np.testing.assert_allclose(
        jax.jit(source_point)(*primals), value, rtol=0.0, atol=1e-12
    )
    _, transpose = jax.vjp(source_point, *primals)
    weight_cotangent, knot_cotangent = transpose(probe)
    assert float(
        jnp.vdot(weight_cotangent, weight_direction)
        + jnp.vdot(knot_cotangent, knot_direction)
    ) == pytest.approx(float(jnp.vdot(probe, source_tangent)), abs=1e-12)
    difference = (
        source_point(
            curve.weights + step * weight_direction, curve.knots + step * knot_direction
        )
        - source_point(
            curve.weights - step * weight_direction, curve.knots - step * knot_direction
        )
    ) / (2 * step)
    np.testing.assert_allclose(source_tangent, difference, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("family", ["offset", "revolution", "bspline"])
@pytest.mark.parametrize(
    "axes",
    [
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ],
    ids=["u", "v", "uu", "uv", "vv", "uuu", "uuv", "uvv", "vvv"],
)
def test_patch_parameter_derivatives_match_adjacent_order_and_closed_form(
    family: CarrierFamily, axes: tuple[int, ...]
) -> None:
    patch = derivative_carrier(family)
    parameters = jnp.asarray([0.37, 0.41], dtype=jnp.float64)
    lower = parameter_partial(patch.evaluate, axes[:-1])
    higher = parameter_partial(patch.evaluate, axes)
    # Preserve the offset distance check's stricter relative tolerance.
    assert_adjacent_order_partial(
        lower, higher, parameters, axes[-1], rtol=1e-9 if family == "offset" else 1e-8
    )
    # Independent formulas anchor every family and every tested derivative;
    # neither positions nor another AD implementation supplies the reference.
    np.testing.assert_allclose(
        higher(parameters),
        closed_form_partial(family, parameters, axes),
        rtol=0.0,
        atol=ANCHOR_ABSOLUTE_ERROR,
    )
    if 0 in axes and 1 in axes:
        # Check every ordering of the third-order mixed partial, not just
        # reversal (uuv, uvu, vuu and uvv, vuv, vvu).
        for ordering in sorted(set(permutations(axes)) - {axes}):
            np.testing.assert_allclose(
                parameter_partial(patch.evaluate, ordering)(parameters),
                higher(parameters),
                rtol=0.0,
                atol=ANCHOR_ABSOLUTE_ERROR,
            )


@pytest.mark.parametrize("family", ["offset", "revolution", "bspline"])
@pytest.mark.parametrize(
    "axes", [(0,), (0, 1), (0, 1, 1)], ids=["first", "second-mixed", "third-mixed"]
)
def test_patch_derivative_oracle_rejects_corrupted_order(
    family: CarrierFamily, axes: tuple[int, ...]
) -> None:
    patch = derivative_carrier(family)
    parameters = jnp.asarray([0.37, 0.41], dtype=jnp.float64)
    lower = parameter_partial(patch.evaluate, axes[:-1])
    higher = parameter_partial(patch.evaluate, axes)

    def corrupted(value: jax.Array) -> jax.Array:
        return higher(value) + jnp.asarray([0.001, 0.0, 0.0], dtype=jnp.float64)

    # The order-n defect is detected against a correct order-(n-1) field;
    # retaining correct positions alone cannot make this oracle pass.
    with pytest.raises(AssertionError):
        assert_adjacent_order_partial(
            lower,
            corrupted,
            parameters,
            axes[-1],
            rtol=1e-9 if family == "offset" else 1e-8,
        )


def test_solved_mixed_line_arc_circle_sketch_lowers_to_exact_face() -> None:
    sketch = _slot_sketch()
    solution = sketch.solve()
    assert bool(solution.converged)
    points, radii = np.asarray(solution.points), np.asarray(solution.circle_radii)
    for circle, start, end in ((0, 1, 2), (1, 3, 0)):
        center = points[(4, 5)[circle]]
        np.testing.assert_allclose(
            np.linalg.norm(points[[start, end]] - center, axis=1),
            radii[circle],
            atol=1e-9,
        )
    model = sketch.to_model(solution, coordinate_contract=_CONTRACT)
    assert model.source_id == "slot-sketch"
    geometry = model.geometry
    assert geometry is not None
    assert len(geometry.face_loops) == 1 and len(geometry.face_loops[0]) == 2
    conics = [
        curve
        for curve in geometry.pcurves
        if isinstance(curve, phx.geometry.BSplineCurve) and curve.degree == 2
    ]
    assert len(conics) == 2
    for conic, (circle, start, end) in zip(conics, ((0, 1, 2), (1, 3, 0)), strict=True):
        interval = np.asarray(conic.knots)[[conic.degree, -conic.degree - 1]]
        np.testing.assert_array_equal(
            conic.evaluate(jnp.asarray(interval)), points[[start, end]]
        )
        samples = conic.evaluate(
            jnp.asarray(np.linspace(interval[0], interval[1], 9, dtype=np.float64))
        )
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(samples) - points[(4, 5)[circle]], axis=-1),
            radii[circle],
            atol=1e-9,
        )
    for coedge in geometry.face_loops[0][0]:
        pcurve = geometry.pcurves[coedge]
        if not (
            isinstance(pcurve, phx.geometry.LineCurve)
            or isinstance(pcurve, phx.geometry.BSplineCurve)
            and pcurve.degree == 1
        ):
            continue
        edge = geometry.coedge_edges[coedge]
        interval = np.asarray(geometry.edge_ranges)[edge]
        endpoints = np.asarray(geometry.vertex_points)[list(geometry.edge_vertices[edge])]
        carrier = geometry.curves[geometry.edge_curves[edge]]
        np.testing.assert_array_equal(carrier.evaluate(jnp.asarray(interval)), endpoints)
        parameter = jnp.asarray(0.5 * (interval[0] + interval[1]), dtype=jnp.float64)
        _, tangent = jax.jvp(carrier.evaluate, (parameter,), (jnp.ones_like(parameter),))
        np.testing.assert_allclose(
            tangent,
            (endpoints[1] - endpoints[0]) / (interval[1] - interval[0]),
            atol=1e-12,
        )
    assert any(isinstance(curve, phx.geometry.CircleCurve) for curve in geometry.pcurves)
    assert float(_measures(model).face_areas[0]) == pytest.approx(_SLOT_AREA, rel=1e-7)


def test_mixed_sketch_extrusion_preserves_source_through_archive(tmp_path: Path) -> None:
    sketch = _slot_sketch()
    profile = sketch.to_profile(sketch.solve())
    model = phx.geometry.brep_extrusion(
        profile,
        (0.0, 0.0, 0.5),
        coordinate_contract=_CONTRACT,
        source_id=sketch.feature_id,
    )
    volume = float(_measures(model).solid_volumes[0])
    assert volume == pytest.approx(0.5 * _SLOT_AREA, rel=1e-7)
    receipt = phx.interchange.save_brep_archive(model, tmp_path / "slot.phx")
    restored = phx.interchange.load_brep_archive(receipt.path)
    assert restored.model_id == model.model_id == receipt.model_id
    assert restored.source_id == sketch.feature_id
    assert float(_measures(restored).solid_volumes[0]) == volume


def test_solved_mixed_sketch_radius_has_live_jvp_and_vjp() -> None:
    sketch = _slot_sketch()
    radius_constraint = sketch.constraints[7]
    assert isinstance(radius_constraint, phx.geometry.Radius)

    def solved_radius(radius: jax.Array) -> jax.Array:
        constraint = eqx.tree_at(lambda value: value.radius, radius_constraint, radius)
        revised = eqx.tree_at(lambda value: value.constraints[7], sketch, constraint)
        return revised.solve().circle_radii[2]

    radius = radius_constraint.radius
    primal, tangent = jax.jvp(solved_radius, (radius,), (jnp.ones_like(radius),))
    reverse = jax.grad(solved_radius)(radius)
    step = 1e-5
    difference = (solved_radius(radius + step) - solved_radius(radius - step)) / (
        2 * step
    )
    assert float(primal) == pytest.approx(0.2, abs=1e-9)
    assert float(tangent) == pytest.approx(1.0, abs=1e-7)
    assert float(reverse) == pytest.approx(float(tangent), abs=1e-10)
    assert float(difference) == pytest.approx(float(tangent), abs=1e-7)
    assert float(jax.jit(solved_radius)(radius)) == pytest.approx(float(primal), abs=1e-9)
    assert float(jax.jit(jax.grad(solved_radius))(radius)) == pytest.approx(
        float(reverse), abs=1e-10
    )


def test_open_or_branched_sketch_curves_are_refused() -> None:
    sketch = phx.geometry.Sketch(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]),
        lines=np.asarray([[0, 1], [1, 2], [2, 3]]),
        circle_centers=np.asarray([4]),
        circle_radii=np.asarray([0.2]),
    )
    with pytest.raises(ValueError, match="closed cycles"):
        sketch.to_profile(sketch.solve())
    with pytest.raises(ValueError, match="to_profile or to_model"):
        sketch.to_source(sketch.solve())


def test_rational_spline_extrusion_has_exact_carriers_and_volume(tmp_path: Path) -> None:
    a, b, height = 1.5, 0.75, 2.0
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(), rational_ellipse_profile((0.0, 0.0), a, b)
    )
    model = phx.geometry.brep_extrusion(
        profile,
        (0.0, 0.0, height),
        coordinate_contract=_CONTRACT,
        source_id="extruded-ellipse",
    )
    geometry = model.geometry
    assert geometry is not None
    source = profile.outer.segments[0]
    assert isinstance(source, phx.geometry.BSplineCurve)
    caps = [
        pcurve
        for pcurve in geometry.pcurves
        if isinstance(pcurve, phx.geometry.BSplineCurve)
    ]
    assert len(caps) == 2
    for cap in caps:
        np.testing.assert_array_equal(
            np.asarray(cap.control_points), np.asarray(source.control_points)
        )
        np.testing.assert_array_equal(np.asarray(cap.weights), np.asarray(source.weights))
    assert (
        sum(isinstance(patch, phx.geometry.ExtrusionSurface) for patch in model.patches)
        == 1
    )
    lateral_patch = next(
        patch
        for patch in model.patches
        if isinstance(patch, phx.geometry.ExtrusionSurface)
    )
    _assert_live_sweep_derivatives(lateral_patch)
    measures = _measures(model)
    assert float(measures.solid_volumes[0]) == pytest.approx(
        math.pi * a * b * height, rel=1e-7
    )
    lateral = _ellipse_perimeter(a, b) * height
    assert float(np.sum(measures.face_areas)) == pytest.approx(
        lateral + 2.0 * math.pi * a * b, rel=1e-7
    )
    restored = phx.interchange.load_brep_archive(
        phx.interchange.save_brep_archive(model, tmp_path / "extruded-ellipse.phx").path
    )
    assert restored.model_id == model.model_id
    assert restored.source_id == model.source_id
    assert float(_measures(restored).solid_volumes[0]) == float(measures.solid_volumes[0])


def test_rational_spline_revolution_matches_pappus(tmp_path: Path) -> None:
    model = revolved_ellipse()
    assert len(model.patches) == 1
    assert isinstance(model.patches[0], phx.geometry.RevolutionSurface)
    _assert_live_sweep_derivatives(model.patches[0])
    measures = _measures(model)
    perimeter = _ellipse_perimeter(_A, _B)
    assert float(measures.solid_volumes[0]) == pytest.approx(
        2.0 * math.pi * _RADIUS * math.pi * _A * _B, rel=1e-7
    )
    assert float(measures.face_areas[0]) == pytest.approx(
        2.0 * math.pi * _RADIUS * perimeter, rel=1e-7
    )
    restored = phx.interchange.load_brep_archive(
        phx.interchange.save_brep_archive(model, tmp_path / "revolved-ellipse.phx").path
    )
    assert restored.model_id == model.model_id
    assert restored.source_id == model.source_id
    assert float(_measures(restored).solid_volumes[0]) == float(measures.solid_volumes[0])


@pytest.mark.parametrize("distance", [0.2, -0.1], ids=["outward", "inward-within-reach"])
def test_normal_offset_of_rational_revolution_is_exact_parallel_body(
    distance: float,
    tmp_path: Path,
) -> None:
    # The source's own tessellation is covered by the Pappus test; the offset
    # is tessellated with the default policy here.
    source = revolved_ellipse(phx.geometry.BRepTessellationPolicy(realize=False))
    model = phx.geometry.brep_offset(
        source, distance, source_id="offset-ellipse-torus"
    ).model
    assert model.source_id == "offset-ellipse-torus"
    (patch,) = model.patches
    assert isinstance(patch, phx.geometry.OffsetSurface)
    assert eqx.tree_equal(patch.base, source.patches[0])
    geometry = model.geometry
    assert geometry is not None
    assert all(isinstance(curve, SurfaceIsoparametricCurve) for curve in geometry.curves)
    for coedge, edge in enumerate(geometry.coedge_edges):
        curve_index = geometry.edge_curves[edge]
        assert curve_index >= 0
        first, last = np.asarray(geometry.edge_ranges)[edge]
        parameters = jnp.linspace(first, last, 5)
        lifted = patch.evaluate(geometry.pcurves[coedge].evaluate(parameters))
        curve = geometry.curves[curve_index]
        assert isinstance(curve, SurfaceIsoparametricCurve)
        boundary = curve.evaluate(parameters)
        np.testing.assert_allclose(
            np.asarray(lifted), np.asarray(boundary), rtol=0.0, atol=1e-12
        )
    perimeter = _ellipse_perimeter(_A, _B)
    meridian_area = math.pi * _A * _B + perimeter * distance + math.pi * distance**2
    query = phx.geometry.prepare_brep_query(model)
    measures = query.measures
    assert float(measures.solid_volumes[0]) == pytest.approx(
        2.0 * math.pi * _RADIUS * meridian_area, rel=1e-6
    )
    assert float(measures.face_areas[0]) == pytest.approx(
        2.0 * math.pi * _RADIUS * (perimeter + 2.0 * math.pi * distance), rel=1e-6
    )
    membership = query.contains(
        np.asarray(
            [
                [_RADIUS, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [_RADIUS + _A + distance + 0.25, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    np.testing.assert_array_equal(membership.inside, [True, False, False])
    assert np.all(
        np.asarray(membership.status) == phx.geometry.BRepProjectionStatus.UNIQUE
    )
    restored = phx.interchange.load_brep_archive(
        phx.interchange.save_brep_archive(model, tmp_path / "offset.phx").path
    )
    assert restored.model_id == model.model_id
    assert restored.source_id == model.source_id
    restored_geometry = restored.geometry
    assert restored_geometry is not None
    assert restored_geometry.geometry_id == geometry.geometry_id
    parameters = jnp.asarray([0.7, 0.125], dtype=jnp.float64)
    base_point = patch.base.evaluate(parameters)
    point = patch.evaluate(parameters)
    expected_normal = (point - base_point) / patch.distance

    def offset_point(value: jax.Array) -> jax.Array:
        return eqx.tree_at(lambda surface: surface.distance, patch, value).evaluate(
            parameters
        )

    _, tangent = jax.jvp(
        offset_point, (patch.distance,), (jnp.ones_like(patch.distance),)
    )
    np.testing.assert_allclose(tangent, expected_normal, rtol=0.0, atol=1e-12)
    assert float(jnp.linalg.norm(tangent)) == pytest.approx(1.0, abs=1e-12)
    probe = jnp.asarray([0.3, -0.2, 0.7], dtype=jnp.float64)
    reverse = jax.grad(lambda value: jnp.vdot(probe, offset_point(value)))(patch.distance)
    assert float(reverse) == pytest.approx(float(jnp.vdot(probe, tangent)), abs=1e-12)
    step = 1e-6
    difference = (
        offset_point(patch.distance + step) - offset_point(patch.distance - step)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, difference, rtol=1e-9, atol=1e-9)


def _meridian_signed_distance(rho: float, height: float) -> float:
    """Signed half-plane distance to the meridian ellipse, negative inside.

    The foot point is the best of 4096 samples refined by Newton steps on the
    stationarity condition ``(e(t) - q) . e'(t) = 0`` of the analytic ellipse.
    """
    x, y = rho - _RADIUS, height
    angles = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    t = float(
        angles[np.argmin(np.hypot(_A * np.cos(angles) - x, _B * np.sin(angles) - y))]
    )
    for _ in range(32):
        cosine, sine = math.cos(t), math.sin(t)
        gap = (_A * cosine - x, _B * sine - y)
        slope = (-_A * sine, _B * cosine)
        rate = gap[0] * slope[0] + gap[1] * slope[1]
        curvature = (
            slope[0] ** 2 + slope[1] ** 2 - gap[0] * _A * cosine - gap[1] * _B * sine
        )
        t -= rate / curvature
    distance = math.hypot(_A * math.cos(t) - x, _B * math.sin(t) - y)
    return -distance if (x / _A) ** 2 + (y / _B) ** 2 < 1.0 else distance


@pytest.mark.parametrize(
    "distance", [0.0, 0.2, -0.1], ids=["revolution", "outward", "inward-within-reach"]
)
def test_closed_meridian_revolution_containment_matches_parallel_body(
    distance: float,
) -> None:
    # Within reach, the body bounded by the revolved parallel meridian at
    # signed distance d is {s < d}, s the signed distance to the ellipse in
    # the meridian half-plane, and its boundary distance is |s - d|.
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    model = revolved_ellipse(lazy)
    if distance:
        model = phx.geometry.brep_offset(model, distance, tessellation=lazy).model
    rng = np.random.default_rng(20261010)
    samples = [
        *zip(
            rng.uniform(0.0, 4.6, 32),
            rng.uniform(-0.8, 0.8, 32),
            rng.uniform(0.0, 2.0 * math.pi, 32),
            strict=True,
        ),
        # On the axis, and on or beside the u = 0 seam half-plane; height 0
        # aims the radial half-line at the v = 0 meridian seam point.
        (0.0, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (_RADIUS, 0.0, 0.0),
        (_RADIUS, 0.0, 1.0e-12),
        (_RADIUS, 0.0, -1.0e-12),
        (_RADIUS + _A + distance - 0.05, 0.0, 0.0),
        (_RADIUS + _A + distance + 0.05, 0.0, 0.0),
        (_RADIUS + _A + distance - 0.05, 1.0e-12, -1.0e-12),
        (_RADIUS - _A - distance + 0.05, 0.0, 0.0),
        (_RADIUS - _A - distance - 0.05, -1.0e-12, 0.0),
        (_RADIUS, _B + distance - 0.05, 2.0 * math.pi - 1.0e-9),
        (_RADIUS, -_B - distance - 0.05, math.pi),
    ]
    points = np.asarray(
        [
            [rho * math.cos(angle), rho * math.sin(angle), height]
            for rho, height, angle in samples
        ],
        dtype=np.float64,
    )
    gaps = np.asarray(
        [_meridian_signed_distance(math.hypot(x, y), z) - distance for x, y, z in points]
    )
    clear = np.abs(gaps) > 1.0e-3
    assert np.count_nonzero(clear) >= 40
    query = phx.geometry.prepare_brep_query(model)
    membership = query.contains(points[clear])
    np.testing.assert_array_equal(membership.inside, gaps[clear] < 0.0)
    assert np.all(
        np.asarray(membership.status) == phx.geometry.BRepProjectionStatus.UNIQUE
    )
    exact = np.abs(gaps[clear])
    assert np.all(np.asarray(membership.distance_lower_bounds) <= exact + 1.0e-12)
    assert np.all(np.asarray(membership.distance_upper_bounds) >= exact - 1.0e-12)
    # The meridian's outer and inner vertices lie on the boundary.
    boundary = query.contains(
        np.asarray(
            [
                [_RADIUS + _A + distance, 0.0, 0.0],
                [0.0, _RADIUS - _A - distance, 0.0],
            ],
            dtype=np.float64,
        )
    )
    np.testing.assert_array_equal(boundary.inside, [True, True])
    assert np.all(
        np.asarray(boundary.status) == phx.geometry.BRepProjectionStatus.AMBIGUOUS
    )


def test_normal_offset_refuses_sharp_edges_and_focal_distances() -> None:
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    box = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=_CONTRACT,
        tessellation=lazy,
    )
    offset = phx.geometry.brep_offset(box, 0.25, tessellation=lazy)
    assert float(_measures(offset.model).solid_volumes[0]) == pytest.approx(
        2.5 * 3.5 * 4.5, rel=1e-12
    )
    with pytest.raises(ValueError, match="face vanishes"):
        phx.geometry.brep_offset(box, -1.0, tessellation=lazy)
    extruded = phx.geometry.brep_extrusion(
        phx.geometry.PlanarProfile(
            phx.geometry.ProfilePlane(), rational_ellipse_profile((0.0, 0.0), 1.0, 0.5)
        ),
        (0.0, 0.0, 1.0),
        coordinate_contract=_CONTRACT,
        tessellation=lazy,
    )
    with pytest.raises(ValueError, match="blend faces"):
        phx.geometry.brep_offset(extruded, 0.1)
    # The meridian's minimum curvature radius is b^2 / a = 1/4; an inward
    # offset by 0.3 passes its focal set and must not be published.
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    with pytest.raises(ValueError, match="unresolved"):
        phx.geometry.brep_offset(revolved_ellipse(lazy), -0.3)


@pytest.mark.parametrize("distance", [-0.25, 0.25])
def test_sharp_planar_prism_offset_matches_inradius_closed_form(
    distance: float,
) -> None:
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    # The 3-4-5 triangle has area 6 and inradius 1. Its parallel triangles
    # are similar with scale (1+d); the height gains two plane offsets.
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(((0.0, 0.0), (3.0, 0.0), (0.0, 4.0))),
    )
    prism = phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT, tessellation=lazy
    )
    offset = phx.geometry.brep_offset(prism, distance, tessellation=lazy)
    assert float(_measures(offset.model).solid_volumes[0]) == pytest.approx(
        6.0 * (1.0 + distance) ** 2 * (2.0 + 2.0 * distance), rel=2e-12
    )


@pytest.mark.parametrize("distance", [-0.25, 0.25])
def test_concave_l_prism_offset_matches_miter_closed_form(distance: float) -> None:
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(
            ((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0))
        ),
    )
    prism = phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT, tessellation=lazy
    )
    offset = phx.geometry.brep_offset(prism, distance, tessellation=lazy)
    # Five convex and one reflex right-angle corners give coefficient 4.
    expected_area = 5.0 + 12.0 * distance + 4.0 * distance**2
    assert float(_measures(offset.model).solid_volumes[0]) == pytest.approx(
        expected_area * (2.0 + 2.0 * distance), rel=2e-12
    )


def test_planar_pocket_offset_refuses_collision_pair() -> None:
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(
            (
                (0.0, 0.0),
                (3.0, 0.0),
                (3.0, 3.0),
                (2.0, 3.0),
                (2.0, 1.0),
                (1.0, 1.0),
                (1.0, 3.0),
                (0.0, 3.0),
            )
        ),
    )
    pocket = phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 3.0), coordinate_contract=_CONTRACT, tessellation=lazy
    )
    with pytest.raises(ValueError, match="collision pair"):
        phx.geometry.brep_offset(pocket, 0.625, tessellation=lazy)


@pytest.mark.parametrize(
    "distance", [-0.26, -0.5], ids=["just-past-focal", "far-past-focal"]
)
def test_inward_offset_beyond_minimum_curvature_radius_refuses(distance: float) -> None:
    # The convex-meridian embedding proof requires a forward parallel tangent,
    # |d| kappa_max < 1 with minimum curvature radius 1/4; past it the isotopy
    # meets the focal set and the certificate must refuse, not publish.
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    with pytest.raises(ValueError, match="unresolved"):
        phx.geometry.brep_offset(revolved_ellipse(lazy), distance, tessellation=lazy)


@pytest.mark.parametrize(
    ("profile", "offset_lifts"),
    [("g1-ellipse", True), ("cornered-closure", False)],
)
def test_closed_meridian_seam_lifts_exactly_only_with_its_closure_proof(
    profile: str, offset_lifts: bool
) -> None:
    # The v = 0 isoline must equal the v = 1 coedge exactly. A revolution
    # needs endpoint position equality; an offset also needs the closure's
    # end legs positively parallel, or the normals there disagree.
    if profile == "g1-ellipse":
        weight = math.sqrt(0.5)
        x, a, b = _RADIUS, _A, _B
        meridian = phx.geometry.BSplineCurve(
            [
                [x + a, 0.0, 0.0],
                [x + a, 0.0, b],
                [x, 0.0, b],
                [x - a, 0.0, b],
                [x - a, 0.0, 0.0],
                [x - a, 0.0, -b],
                [x, 0.0, -b],
                [x + a, 0.0, -b],
                [x + a, 0.0, 0.0],
            ],
            [1.0, weight, 1.0, weight, 1.0, weight, 1.0, weight, 1.0],
            [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
            2,
        )
    else:
        # Closed, but the end legs (-1, 0, 0.5) and (1, 0, 0.5) are not parallel.
        meridian = phx.geometry.BSplineCurve(
            [
                [4.0, 0.0, 0.0],
                [3.0, 0.0, 0.5],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, -0.5],
                [4.0, 0.0, 0.0],
            ],
            np.ones(5),
            [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0],
            1,
        )
    revolution = phx.geometry.RevolutionSurface(
        meridian, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    )
    box = np.asarray([[0.0, 0.0], [2.0 * math.pi, 1.0]])
    opposite = phx.geometry.LineCurve(np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0]))
    for surface, lifts in (
        (revolution, True),
        (phx.geometry.OffsetSurface(revolution, 0.1), offset_lifts),
    ):
        seam = SurfaceIsoparametricCurve(surface, 1, 0.0)
        # A zero tolerance requires equality of the represented carriers.
        result = certify_curve_surface(
            seam,
            opposite,
            surface,
            box,
            0.0,
            2.0 * math.pi,
            point=np.asarray([4.0, 0.0, 0.0]),
            tolerance=0.0,
        )
        assert (result.deviation_bound == 0.0 and not result.unresolved) is lifts


@pytest.fixture
def correspondence_budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=128 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000_000, maximum_memory_bytes=128 << 20
        )
        with budget.activate():
            yield budget


def _correspondence_sources() -> tuple[
    SurfaceIsoparametricCurve,
    phx.geometry.LineCurve,
    phx.geometry.BSplineSurfacePatch,
    np.ndarray,
]:
    # A genuine nonzero source gap needs the interval proof, not a generating
    # identity: the face's z coordinate is u*v/1024 and the edge's is zero.
    return (
        SurfaceIsoparametricCurve(
            phx.geometry.PlanePatch((0, 0, 0), (1, 1, 0), (0, 0, 1)),
            1, 0.0, parameter_range=(0.0, 1.0),
        ),
        phx.geometry.LineCurve((0.0, 0.0), (1.0, 1.0)),
        phx.geometry.BSplineSurfacePatch(
            (((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 1 / 1024))),
            np.ones((2, 2)),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
            1,
            1,
        ),
        np.asarray(((0.0, 0.0), (1.0, 1.0))),
    )


def test_native_correspondence_reuses_only_one_budget(
    correspondence_budget: CoordinateEnclosureBudget,
) -> None:
    sources = _correspondence_sources()
    point = np.zeros(3)
    with (
        patch.object(
            correspondence_owner, "_execute_correspondence_workset",
            wraps=correspondence_owner._execute_correspondence_workset,
        ) as execute,
        patch.object(
            correspondence_owner, "_generating_correspondence",
            wraps=correspondence_owner._generating_correspondence,
        ) as generating,
    ):
        original = certify_curve_surface(
            *sources, 0.0, 1.0, point=point, tolerance=0.1,
            budget=correspondence_budget,
        )
        assert original.complete and original.deviation_bound > 0.0
        retained = correspondence_budget.retained_basis_bytes
        repeated = certify_curve_surface(
            *sources, 0.0, 1.0, point=point, tolerance=0.1,
            budget=correspondence_budget,
        )
        assert repeated is original
        assert correspondence_budget.retained_basis_bytes == retained
        assert generating.call_count == 2
        assert execute.call_count == 1
        # Equal source data does not carry a certificate into a new request.
        with HostBatchTicket(100_000_000, bytes_upper=128 << 20) as ticket:
            cold = CoordinateEnclosureBudget(
                ticket, maximum_work_units=100_000_000, maximum_memory_bytes=128 << 20
            )
            with cold.activate():
                independent = certify_curve_surface(
                    *sources, 0.0, 1.0, point=point, tolerance=0.1, budget=cold,
                )
        assert independent == original and independent is not original
        assert execute.call_count == 2


@pytest.mark.parametrize(
    "changed",
    ("curve", "mutable", "pcurve", "surface", "static", "box", "point", "range",
     "tolerance", "maximum_intervals"),
)
def test_native_correspondence_key_binds_live_sources_and_controls(
    changed: str, correspondence_budget: CoordinateEnclosureBudget,
) -> None:
    curve, pcurve, surface, box = _correspondence_sources()
    plane = curve.surface
    assert isinstance(plane, phx.geometry.PlanePatch)
    point = np.zeros(3)
    mutable_origin = np.zeros(3)
    if changed == "mutable":
        mutable_plane = eqx.tree_at(lambda source: source.origin, plane, mutable_origin)
        curve = eqx.tree_at(lambda source: source.surface, curve, mutable_plane)
    first, last, tolerance, maximum_intervals = 0.125, 0.875, 0.1, 128
    with patch.object(
        correspondence_owner, "_execute_correspondence_workset",
        wraps=correspondence_owner._execute_correspondence_workset,
    ) as execute:
        original = certify_curve_surface(
            curve, pcurve, surface, box, first, last, point=point,
            tolerance=tolerance, maximum_intervals=maximum_intervals,
            budget=correspondence_budget,
        )
        assert original.complete
        if changed == "curve":
            changed_plane = eqx.tree_at(
                lambda source: source.origin, plane, plane.origin.at[2].set(1 / 2048)
            )
            curve = eqx.tree_at(lambda source: source.surface, curve, changed_plane)
        elif changed == "mutable":
            mutable_origin[2] = 1 / 2048
        elif changed == "pcurve":
            pcurve = eqx.tree_at(lambda source: source.direction, pcurve,
                                 pcurve.direction.at[0].set(0.99))
        elif changed == "surface":
            surface = eqx.tree_at(
                lambda source: source.control_points, surface,
                surface.control_points.at[1, 1, 2].set(1 / 512),
            )
        elif changed == "static":
            curve = SurfaceIsoparametricCurve(
                plane, 1, 1 / 2048, parameter_range=(0.0, 1.0)
            )
        elif changed == "box":
            box[1, 0] = 0.99
        elif changed == "point":
            point[2] = 1 / 2048
        elif changed == "range":
            first, last = 0.25, 0.75
        elif changed == "tolerance":
            tolerance = 0.05
        else:
            maximum_intervals = 256
        result = certify_curve_surface(
            curve, pcurve, surface, box, first, last, point=point,
            tolerance=tolerance, maximum_intervals=maximum_intervals,
            budget=correspondence_budget,
        )
        assert result.complete and result is not original
        assert execute.call_count == 2
        assert len(correspondence_budget.brep_correspondence_cache) == 2


def test_native_correspondence_does_not_reuse_incomplete_proofs(
    correspondence_budget: CoordinateEnclosureBudget,
) -> None:
    sources = _correspondence_sources()
    with patch.object(
        correspondence_owner, "_execute_correspondence_workset",
        wraps=correspondence_owner._execute_correspondence_workset,
    ) as execute:
        for _ in range(2):
            result = certify_curve_surface(
                *sources, 0.0, 1.0, point=np.zeros(3), tolerance=1e-12,
                maximum_intervals=1, budget=correspondence_budget,
            )
            assert not result.complete and result.unresolved
        assert execute.call_count == 2
        assert not correspondence_budget.brep_correspondence_cache


def test_native_correspondence_hit_preserves_control_and_range_checks(
    correspondence_budget: CoordinateEnclosureBudget,
) -> None:
    curve, _, surface, box = _correspondence_sources()
    pcurve = phx.geometry.BSplineCurve.bezier(((0, 0), (1, 1)), (1, 1))
    original = certify_curve_surface(
        curve, pcurve, surface, box, 0.0, 1.0, point=np.zeros(3), tolerance=0.1,
        budget=correspondence_budget,
    )
    assert original.complete
    with pytest.raises(ValueError, match="controls"):
        certify_curve_surface(
            curve, pcurve, surface, box, 0.0, 1.0, point=np.zeros(3), tolerance=-1.0,
            budget=correspondence_budget,
        )
    with pytest.raises(ValueError):
        certify_curve_surface(
            curve, pcurve, surface, box, -0.1, 1.0, point=np.zeros(3), tolerance=0.1,
            budget=correspondence_budget,
        )


def test_native_correspondence_hit_preserves_byte_admission(
    correspondence_budget: CoordinateEnclosureBudget,
) -> None:
    sources = _correspondence_sources()
    original = certify_curve_surface(
        *sources, 0.0, 1.0, point=np.zeros(3), tolerance=0.1,
        budget=correspondence_budget,
    )
    assert original.complete
    with correspondence_budget.bound_stage(
        100_000_000, correspondence_budget.retained_basis_bytes,
    ):
        with pytest.raises(CoordinateEnclosureResourceError):
            certify_curve_surface(
                *sources, 0.0, 1.0, point=np.zeros(3), tolerance=0.1,
                budget=correspondence_budget,
            )


def test_native_correspondence_cold_proof_preserves_work_refusal() -> None:
    sources = _correspondence_sources()
    with HostBatchTicket(0, bytes_upper=128 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=128 << 20
        )
        with budget.activate(), pytest.raises(CoordinateEnclosureResourceError):
            certify_curve_surface(
                *sources, 0.0, 1.0, point=np.zeros(3), tolerance=0.1, budget=budget,
            )
        assert not budget.brep_correspondence_cache


@pytest.mark.parametrize("warm", (False, True), ids=("new-proof", "cache-hit"))
def test_native_correspondence_polls_deadline_before_return(
    warm: bool, correspondence_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _correspondence_sources()
    if warm:
        assert certify_curve_surface(
            *sources, 0.0, 1.0, point=np.zeros(3), tolerance=0.1,
            budget=correspondence_budget,
        ).complete
    expired = warm
    original_checkpoint = HostBatchTicket.checkpoint
    original_execute = correspondence_owner._execute_correspondence_workset

    def checkpoint(ticket: HostBatchTicket) -> None:
        original_checkpoint(ticket)
        if expired:
            raise TimeoutError("correspondence deadline")

    def execute(*args, **kwargs):
        nonlocal expired
        result = original_execute(*args, **kwargs)
        expired = True
        return result

    monkeypatch.setattr(HostBatchTicket, "checkpoint", checkpoint)
    monkeypatch.setattr(correspondence_owner, "_execute_correspondence_workset", execute)
    with pytest.raises(TimeoutError, match="correspondence deadline"):
        certify_curve_surface(
            *sources, 0.0, 1.0, point=np.zeros(3), tolerance=0.1,
            budget=correspondence_budget,
        )
    assert len(correspondence_budget.brep_correspondence_cache) == int(warm)
    expired = False


@pytest.mark.parametrize("format_name", ["step", "iges", "brep"])
@pytest.mark.parametrize("sweep", ["extrusion", "revolution"])
def test_exact_rational_sweep_external_round_trip(
    format_name: Literal["step", "iges", "brep"],
    sweep: Literal["extrusion", "revolution"],
    tmp_path: Path,
) -> None:
    if sweep == "revolution":
        model = revolved_ellipse()
        expected_volume = 2.0 * math.pi * _RADIUS * math.pi * _A * _B
    else:
        profile = phx.geometry.PlanarProfile(
            phx.geometry.ProfilePlane(), rational_ellipse_profile((0.0, 0.0), _A, _B)
        )
        model = phx.geometry.brep_extrusion(
            profile,
            (0.0, 0.0, 0.75),
            coordinate_contract=_CONTRACT,
            source_id="external-extruded-ellipse",
        )
        expected_volume = math.pi * _A * _B * 0.75
    source_id = model.model_id
    geometry = model.geometry
    assert geometry is not None
    geometry_id = geometry.geometry_id
    policy = phx.interchange.CadImportPolicy(
        _CONTRACT,
        ResourceLimits(
            max_bytes=1 << 24,
            max_depth=64,
            max_nodes=200_000,
            max_attributes=4_000_000,
            max_losses=0,
        ),
    )
    path = tmp_path / ("ellipse." + format_name)
    match format_name:
        case "step":
            published = phx.interchange.write_step(model, path)
            imported = phx.interchange.read_step(path, policy, trusted_root=tmp_path)
        case "iges":
            published = phx.interchange.write_iges(model, path)
            imported = phx.interchange.read_iges(path, policy, trusted_root=tmp_path)
        case "brep":
            published = phx.interchange.write_brep_text(model, path)
            imported = phx.interchange.read_brep_text(
                path,
                policy,
                trusted_root=tmp_path,
                source_length_unit=model.coordinate_contract.length_unit,
            )
    assert published.report.valid
    assert not published.approximations
    assert imported.report.valid
    assert imported.coverage.pcurves_fitted == 0
    restored = imported.model
    restored_geometry = restored.geometry
    assert restored_geometry is not None
    assert restored_geometry.topology().num_solids == geometry.topology().num_solids
    assert restored_geometry.topology().num_faces == geometry.topology().num_faces
    assert restored_geometry.topology().num_edges == geometry.topology().num_edges
    assert float(_measures(restored).solid_volumes[0]) == pytest.approx(
        expected_volume, rel=1e-7
    )
    assert model.model_id == source_id
    assert geometry.geometry_id == geometry_id


@pytest.mark.parametrize("format_name", ["step", "iges"])
@pytest.mark.parametrize(
    ("unit", "factor"),
    [(METER, 1.0), (MILLIMETER, 1000.0)],
    ids=["meters", "millimeters"],
)
def test_exact_offset_surface_external_round_trip_preserves_source_and_units(
    format_name: Literal["step", "iges"],
    unit: UnitDefinition,
    factor: float,
    tmp_path: Path,
) -> None:
    limits = ResourceLimits(
        max_bytes=1 << 24,
        max_depth=64,
        max_nodes=200_000,
        max_attributes=4_000_000,
        max_losses=0,
    )
    model = native_offset_plane(phx.interchange.CadImportPolicy(_CONTRACT, limits))
    source_id = model.model_id
    geometry = model.geometry
    assert geometry is not None
    geometry_id = geometry.geometry_id
    contract = phx.SpatialCoordinateContract(unit)
    policy = phx.interchange.CadImportPolicy(contract, limits)
    path = tmp_path / ("offset." + format_name)
    match format_name:
        case "step":
            published = phx.interchange.write_step(model, path)
            imported = phx.interchange.read_step(path, policy, trusted_root=tmp_path)
        case "iges":
            published = phx.interchange.write_iges(model, path)
            imported = phx.interchange.read_iges(path, policy, trusted_root=tmp_path)
    assert published.report.valid and not published.approximations
    assert imported.report.valid and imported.coverage.pcurves_fitted == 0
    restored = imported.model
    assert restored.coordinate_contract.spatial_id == contract.spatial_id
    (patch,) = restored.patches
    assert isinstance(patch, phx.geometry.OffsetSurface)
    assert isinstance(patch.base, phx.geometry.PlanePatch)
    assert float(patch.distance) == pytest.approx(0.25 * factor, rel=0.0, abs=1e-12)
    uv = factor * jnp.asarray([[0.2, 0.7], [1.7, 0.2]], dtype=jnp.float64)
    expected = factor * np.asarray([[0.2, 0.7, 0.75], [1.7, 0.2, 0.75]])
    np.testing.assert_allclose(patch.evaluate(uv), expected, rtol=0.0, atol=1e-10)
    assert float(_measures(restored).face_areas[0]) == pytest.approx(
        2.0 * factor**2, rel=1e-10
    )
    assert restored.topology.num_faces == 1 and restored.topology.num_edges == 4
    assert model.model_id == source_id and geometry.geometry_id == geometry_id


def test_native_profile_paths_never_import_external_kernels(tmp_path: Path) -> None:
    # Imported kernels are process-global, so only a fresh interpreter isolates
    # these native paths from external-kernel tests sharing a worker.
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import phydrax as phx
        from phydrax._external_resource import ResourceLimits
        from phydrax.meshing.qualification import rational_ellipse_profile
        from tests.unit.geometry.test_native_profile_construction import (
            _CONTRACT,
            _slot_sketch,
        )

        root = Path(sys.argv[1])
        geometry, interchange = phx.geometry, phx.interchange
        # Native constructions stay lazy: tessellation does not choose which
        # module owns a path. Imports realize coarse surfaces, which keeps the
        # native tessellator in the contract and costs less than validating an
        # unrealized import exactly.
        lazy = geometry.BRepTessellationPolicy(realize=False)
        coarse = geometry.BRepTessellationPolicy(
            linear_deflection=0.25, angular_deflection=0.8
        )
        sketch = _slot_sketch()
        solution = sketch.solve()
        sketch.to_model(solution, coordinate_contract=_CONTRACT, tessellation=lazy)
        slot = geometry.brep_extrusion(
            sketch.to_profile(solution),
            (0.0, 0.0, 0.5),
            coordinate_contract=_CONTRACT,
            source_id=sketch.feature_id,
            tessellation=lazy,
        )
        interchange.load_brep_archive(
            interchange.save_brep_archive(slot, root / "slot.phx").path
        )
        extruded = geometry.brep_extrusion(
            geometry.PlanarProfile(
                geometry.ProfilePlane(), rational_ellipse_profile((0.0, 0.0), 1.0, 0.5)
            ),
            (0.0, 0.0, 0.75),
            coordinate_contract=_CONTRACT,
            tessellation=lazy,
        )
        revolved = geometry.brep_revolution(
            geometry.PlanarProfile(
                geometry.ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                rational_ellipse_profile((3.0, 0.0), 1.0, 0.5),
            ),
            (0.0, 0.0),
            (0.0, 1.0),
            coordinate_contract=_CONTRACT,
            tessellation=lazy,
        )
        geometry.brep_offset(revolved, 0.2, tessellation=lazy)
        policy = interchange.CadImportPolicy(
            _CONTRACT,
            ResourceLimits(
                max_bytes=1 << 24,
                max_depth=64,
                max_nodes=200_000,
                max_attributes=4_000_000,
                max_losses=0,
            ),
            tessellation=coarse,
        )
        unit = extruded.coordinate_contract.length_unit
        for name, write, read in (
            ("step", interchange.write_step, interchange.read_step),
            ("iges", interchange.write_iges, interchange.read_iges),
            ("brep", interchange.write_brep_text, None),
        ):
            path = root / ("ellipse." + name)
            write(extruded, path)
            if read is None:
                interchange.read_brep_text(
                    path, policy, trusted_root=root, source_length_unit=unit
                )
            else:
                read(path, policy, trusted_root=root)
        loaded = sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in ("OCP", "OCC", "build123d")
        )
        print(",".join(loaded))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert completed.stdout.strip() == ""
