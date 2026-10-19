"""Parallel-cylinder lens: independent circular-segment area and volume oracle."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.geometry.brep._differentiable import _FixedTopologyBRepKernel
from phydrax.geometry.brep._fillet import brep_chamfer, brep_fillet
from phydrax.geometry.brep._patches import CylinderPatch


# Empirical `test_parallel_cylinder_fillet_matches_circle_segment_closed_forms`: matched max |delta|=8.88178e-16; joint allowance fraction=4.44089e-06, margin=99.9996%.
_PARALLEL_FILLET_VOLUME_CHANGE_ATOL = 2e-10
# Empirical `test_parallel_cylinder_fillet_matches_circle_segment_closed_forms`: matched max |delta|=5.21805e-15; joint allowance fraction=2.60902e-05, margin=99.9974%.
_PARALLEL_FILLET_AREA_CHANGE_ATOL = 2e-10
# Empirical `test_parallel_cylinder_chamfer_matches_circle_segment_closed_form`: matched max |delta|=4.996e-15; joint allowance fraction=2.498e-05, margin=99.9975%.
_PARALLEL_CHAMFER_VOLUME_CHANGE_ATOL = 2e-10
# Empirical `test_parallel_fillet_radius_derivative_has_seam_evidence_at_fd_points`: matched max |delta|=1.74329e-10; joint allowance fraction=0.000871645, margin=99.9128%.
_PARALLEL_FILLET_RADIUS_DERIVATIVE_ATOL = 2e-7


_CONTRACT = phx.SpatialCoordinateContract.si()
_LAZY = phx.geometry.BRepTessellationPolicy(realize=False)


def _lens() -> tuple[phx.geometry.BRepModel, int]:
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop(
            ((0.0, -4.0), (0.0, 4.0)),
            (phx.geometry.ProfileArc((-3.0, 0.0)), phx.geometry.ProfileArc((3.0, 0.0))),
        ),
    )
    model = phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    geometry = model.require_geometry()
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        uses = [
            face
            for face, loops in enumerate(geometry.face_loops)
            if any(
                geometry.coedge_edges[coedge] == edge for loop in loops for coedge in loop
            )
        ]
        if (
            first != last
            and len(uses) == 2
            and all(isinstance(model.patches[face], CylinderPatch) for face in uses)
            and float(geometry.vertex_points[first, 1]) == 4.0
        ):
            return model, edge
    raise RuntimeError(
        "LensFixture.edge must contain the upper cylinder junction; got no matching incidence"
    )


def _circle_integral(
    center: tuple[float, float], radius: float, start: float, end: float, /
) -> float:
    return (
        radius**2 * (end - start)
        + center[0] * radius * (math.sin(end) - math.sin(start))
        - center[1] * radius * (math.cos(end) - math.cos(start))
    )


def _fillet_removed(radius: float, /) -> tuple[float, float]:
    top_angle = math.atan2(4.0, 3.0)
    center_y = math.sqrt((5.0 - radius) ** 2 - 9.0)
    angle = math.atan2(center_y, 3.0)
    integral = (
        _circle_integral((-3.0, 0.0), 5.0, angle, top_angle)
        + _circle_integral((3.0, 0.0), 5.0, math.pi - top_angle, math.pi - angle)
        - _circle_integral((0.0, center_y), radius, angle, math.pi - angle)
    )
    removed_area = integral / 2.0
    perimeter_change = radius * (math.pi - 2.0 * angle) - 10.0 * (top_angle - angle)
    return removed_area, perimeter_change


def test_parallel_cylinder_fillet_matches_circle_segment_closed_forms() -> None:
    source, edge = _lens()
    old = phx.geometry.prepare_brep_query(source).measures
    result = brep_fillet(source, (source.edge_ids[edge],), 0.25, tessellation=_LAZY)
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    new = phx.geometry.prepare_brep_query(model).measures
    removed_area, perimeter_change = _fillet_removed(0.25)
    assert float(new.solid_volumes[0] - old.solid_volumes[0]) == pytest.approx(
        -2.0 * removed_area, abs=_PARALLEL_FILLET_VOLUME_CHANGE_ATOL
    )
    assert float(np.sum(new.face_areas) - np.sum(old.face_areas)) == pytest.approx(
        2.0 * perimeter_change - 2.0 * removed_area, abs=_PARALLEL_FILLET_AREA_CHANGE_ATOL
    )
    assert result.g1_angle_bounds.shape == (2, 2)
    assert np.all(np.asarray(result.g1_angle_bounds)[:, 1] < 1e-6)


def test_parallel_cylinder_chamfer_matches_circle_segment_closed_form() -> None:
    source, edge = _lens()
    a, b = 0.25, 0.375
    angle = math.atan2(4.0, 3.0)
    first, last = angle - a / 5.0, math.pi - angle + b / 5.0
    p = (-3.0 + 5.0 * math.cos(first), 5.0 * math.sin(first))
    q = (3.0 + 5.0 * math.cos(last), 5.0 * math.sin(last))
    integral = (
        _circle_integral((-3.0, 0.0), 5.0, first, angle)
        + _circle_integral((3.0, 0.0), 5.0, math.pi - angle, last)
        + q[0] * p[1]
        - q[1] * p[0]
    )
    removed = integral / 2.0
    old = phx.geometry.prepare_brep_query(source).measures
    result = brep_chamfer(source, (source.edge_ids[edge],), (a, b), tessellation=_LAZY)
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    new = phx.geometry.prepare_brep_query(model).measures
    assert float(new.solid_volumes[0] - old.solid_volumes[0]) == pytest.approx(
        -2.0 * removed, abs=_PARALLEL_CHAMFER_VOLUME_CHANGE_ATOL
    )


def test_parallel_fillet_radius_derivative_has_seam_evidence_at_fd_points() -> None:
    source, edge = _lens()
    result = brep_fillet(source, (source.edge_ids[edge],), 0.25, tessellation=_LAZY)
    assert result.status == "certified", result.reason
    radius_source = result.source
    assert radius_source is not None
    compiled = radius_source.compile(coordinate_unit=_CONTRACT.length_unit)
    kernel = compiled.kernel
    assert isinstance(kernel, _FixedTopologyBRepKernel)
    index = compiled.schema.index(compiled.schema.parameter_ids[0])

    def volume(radius: Array) -> Array:
        return kernel.measure(compiled.state.replace_at(index, radius))

    step = 2.0**-15
    samples = []
    for radius in (0.25 - step, 0.25, 0.25 + step):
        state = compiled.state.replace_at(index, jnp.asarray(radius, dtype=jnp.float64))
        realized = kernel.realize(state)
        assert float(realized.seam_residual) <= float(realized.seam_bound)
        samples.append(float(kernel.measure(state)))
    finite_difference = (samples[2] - samples[0]) / (2.0 * step)
    independent = (
        -2.0
        * (_fillet_removed(0.25 + step)[0] - _fillet_removed(0.25 - step)[0])
        / (2.0 * step)
    )
    automatic = float(jax.grad(volume)(jnp.asarray(0.25, dtype=jnp.float64)))
    assert automatic == pytest.approx(
        finite_difference, abs=_PARALLEL_FILLET_RADIUS_DERIVATIVE_ATOL
    )
    assert automatic == pytest.approx(
        independent, abs=_PARALLEL_FILLET_RADIUS_DERIVATIVE_ATOL
    )
