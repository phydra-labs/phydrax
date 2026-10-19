"""Analytic edge-shaping oracles independent of the B-Rep builders."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.geometry.brep._differentiable import _FixedTopologyBRepKernel
from phydrax.geometry.brep._fillet import brep_chamfer, brep_fillet


# Empirical `test_single_box_edge_fillet_matches_area_and_volume_closed_forms`: matched max |delta|=3.55271e-15; joint allowance fraction=7.26635e-05, margin=99.9927%.
_BOX_FILLET_VOLUME_RTOL = 2e-12
# Empirical `test_single_box_edge_fillet_matches_area_and_volume_closed_forms`: matched max |delta|=1.42109e-14; joint allowance fraction=0.000136527, margin=99.9863%.
_BOX_FILLET_SURFACE_AREA_RTOL = 2e-12
# Empirical `test_single_box_edge_chamfer_matches_setback_closed_form`: matched max |delta|=3.55271e-15; joint allowance fraction=7.32518e-05, margin=99.9927%.
_BOX_CHAMFER_VOLUME_RTOL = 2e-12
# Empirical `test_single_box_edge_chamfer_matches_setback_closed_form`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_BOX_CHAMFER_SURFACE_AREA_RTOL = 2e-12
# Empirical `test_fillet_radius_derivative_uses_linked_native_trims_and_seam_evidence`: matched max |delta|=4.71267e-12; joint allowance fraction=0.054268, margin=94.5732%.
_FILLET_RADIUS_VOLUME_DERIVATIVE_RTOL = 2e-10
# Empirical `test_concave_plane_pair_fillet_uses_material_side_and_closed_form`: matched max |delta|=3.55271e-15; joint allowance fraction=0.000168746, margin=99.9831%.
_CONCAVE_FILLET_VOLUME_RTOL = 2e-12
# Empirical `test_concave_plane_pair_fillet_uses_material_side_and_closed_form`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CONCAVE_FILLET_SURFACE_AREA_RTOL = 2e-12


_CONTRACT = phx.SpatialCoordinateContract.si()


def _box() -> phx.geometry.BRepModel:
    return phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=_CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )


def _vertical_edge(model: phx.geometry.BRepModel) -> int:
    geometry = model.require_geometry()
    for index, (first, last) in enumerate(geometry.edge_vertices):
        direction = np.asarray(
            geometry.vertex_points[last] - geometry.vertex_points[first]
        )
        if direction[0] == direction[1] == 0.0 and abs(direction[2]) == 4.0:
            return index
    raise RuntimeError(
        "BoxFixture.edge must contain a vertical source edge; got no matching incidence"
    )


def test_single_box_edge_fillet_matches_area_and_volume_closed_forms() -> None:
    box = _box()
    edge = _vertical_edge(box)
    radius = 0.25
    result = brep_fillet(
        box,
        (box.edge_ids[edge],),
        radius,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    measure = phx.geometry.prepare_brep_query(model).measures
    removed_cross_section = (1.0 - math.pi / 4.0) * radius**2
    assert float(measure.solid_volumes[0]) == pytest.approx(
        24.0 - 4.0 * removed_cross_section, rel=_BOX_FILLET_VOLUME_RTOL
    )
    # Two caps lose the removed cross-section; two side strips are replaced
    # by one quarter-cylinder whose lateral area is pi*r*L/2.
    expected_area = (
        52.0 - 2.0 * removed_cross_section + 4.0 * radius * (math.pi / 2.0 - 2.0)
    )
    assert float(np.sum(measure.face_areas)) == pytest.approx(
        expected_area, rel=_BOX_FILLET_SURFACE_AREA_RTOL
    )
    assert result.g1_angle_bounds.shape == (2, 2)
    assert np.all(np.asarray(result.g1_angle_bounds)[:, 0] == 0.0)


def test_single_box_edge_chamfer_matches_setback_closed_form() -> None:
    box = _box()
    edge = _vertical_edge(box)
    a, b = 0.25, 0.5
    result = brep_chamfer(
        box,
        (box.edge_ids[edge],),
        (a, b),
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    measure = phx.geometry.prepare_brep_query(model).measures
    assert float(measure.solid_volumes[0]) == pytest.approx(
        24.0 - a * b * 4.0 / 2.0, rel=_BOX_CHAMFER_VOLUME_RTOL
    )
    expected_area = 52.0 - a * b + 4.0 * (math.hypot(a, b) - a - b)
    assert float(np.sum(measure.face_areas)) == pytest.approx(
        expected_area, rel=_BOX_CHAMFER_SURFACE_AREA_RTOL
    )
    assert result.g1_angle_bounds.shape == (0, 2)


def test_multiple_blended_edges_at_a_vertex_refuse_corner_patch() -> None:
    box = _box()
    geometry = box.require_geometry()
    first_edge = _vertical_edge(box)
    first_vertices = set(geometry.edge_vertices[first_edge])
    second_edge = next(
        index
        for index, vertices in enumerate(geometry.edge_vertices)
        if index != first_edge and first_vertices & set(vertices)
    )
    result = brep_fillet(box, (box.edge_ids[first_edge], box.edge_ids[second_edge]), 0.25)
    assert result.status == "unsupported"
    assert "corner patch" in result.reason
    assert result.model is None


def test_fillet_refuses_disappearing_neighboring_face() -> None:
    box = _box()
    result = brep_fillet(
        box,
        (box.edge_ids[_vertical_edge(box)],),
        2.0,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    assert result.status == "unsupported"
    assert "face vanishes" in result.reason


def test_fillet_radius_derivative_uses_linked_native_trims_and_seam_evidence() -> None:
    box = _box()
    edge = _vertical_edge(box)
    result = brep_fillet(
        box,
        (box.edge_ids[edge],),
        0.25,
        tessellation=phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.2, angular_deflection=0.4
        ),
    )
    assert result.status == "certified", result.reason
    source = result.source
    assert source is not None
    compiled = source.compile(coordinate_unit=_CONTRACT.length_unit)
    kernel = compiled.kernel
    assert isinstance(kernel, _FixedTopologyBRepKernel)
    index = compiled.schema.index(compiled.schema.parameter_ids[0])

    def volume(radius: Array) -> Array:
        return kernel.measure(compiled.state.replace_at(index, radius))

    step = 2.0**-12
    samples = []
    for radius in (0.25 - step, 0.25, 0.25 + step):
        state = compiled.state.replace_at(index, jnp.asarray(radius, dtype=jnp.float64))
        realized = kernel.realize(state)
        assert float(realized.seam_residual) <= float(realized.seam_bound)
        samples.append(float(kernel.measure(state)))
    finite_difference = (samples[2] - samples[0]) / (2 * step)
    automatic = float(jax.grad(volume)(jnp.asarray(0.25, dtype=jnp.float64)))
    expected = -8.0 * (1.0 - math.pi / 4.0) * 0.25
    assert automatic == pytest.approx(expected, rel=_FILLET_RADIUS_VOLUME_DERIVATIVE_RTOL)
    assert finite_difference == pytest.approx(
        expected, rel=_FILLET_RADIUS_VOLUME_DERIVATIVE_RTOL
    )


def test_nonanalytic_face_adjacency_refuses_with_unsupported_status() -> None:
    curve = phx.geometry.BSplineCurve(
        np.asarray(
            (
                (2.0, 0.0),
                (2.0, 1.0),
                (0.0, 1.0),
                (-2.0, 1.0),
                (-2.0, 0.0),
                (-2.0, -1.0),
                (0.0, -1.0),
                (2.0, -1.0),
                (2.0, 0.0),
            ),
            dtype=np.float64,
        ),
        np.asarray(
            (
                1.0,
                math.sqrt(0.5),
                1.0,
                math.sqrt(0.5),
                1.0,
                math.sqrt(0.5),
                1.0,
                math.sqrt(0.5),
                1.0,
            ),
            dtype=np.float64,
        ),
        np.asarray(
            (0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0), dtype=np.float64
        ),
        2,
    )
    source = phx.geometry.brep_extrusion(
        phx.geometry.PlanarProfile(
            phx.geometry.ProfilePlane(),
            phx.geometry.ProfileLoop(((2.0, 0.0),), (curve,)),
        ),
        (0.0, 0.0, 1.0),
        coordinate_contract=_CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    geometry = source.require_geometry()
    edge = next(
        index
        for index in range(len(geometry.edge_curves))
        if len(
            {
                face
                for face, loops in enumerate(geometry.face_loops)
                if any(
                    geometry.coedge_edges[coedge] == index
                    for loop in loops
                    for coedge in loop
                )
            }
        )
        == 2
    )
    result = brep_fillet(source, (source.edge_ids[edge],), 0.25)
    assert result.status == "unsupported"
    assert "analytic face adjacency" in result.reason
    assert result.model is None


def test_concave_plane_pair_fillet_uses_material_side_and_closed_form() -> None:
    lazy = phx.geometry.BRepTessellationPolicy(realize=False)
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(
            ((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0))
        ),
    )
    source = phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT, tessellation=lazy
    )
    geometry = source.require_geometry()
    edge = next(
        index
        for index, (first, last) in enumerate(geometry.edge_vertices)
        if first != last
        and np.array_equal(np.asarray(geometry.vertex_points[first])[:2], (1.0, 1.0))
        and np.array_equal(np.asarray(geometry.vertex_points[last])[:2], (1.0, 1.0))
    )
    radius = 0.25
    result = brep_fillet(source, (source.edge_ids[edge],), radius, tessellation=lazy)
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    measure = phx.geometry.prepare_brep_query(model).measures
    added = (1.0 - math.pi / 4.0) * radius**2
    assert float(measure.solid_volumes[0]) == pytest.approx(
        10.0 + 2.0 * added, rel=_CONCAVE_FILLET_VOLUME_RTOL
    )
    assert float(np.sum(measure.face_areas)) == pytest.approx(
        34.0 + 2.0 * added + 2.0 * radius * (math.pi / 2.0 - 2.0),
        rel=_CONCAVE_FILLET_SURFACE_AREA_RTOL,
    )
    assert result.sensitivity_status == "available"
