"""Closed-form wedge-solid oracles for native oblique plane-pair fillets."""

import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep._constructors import Builder, finish
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._fillet import brep_fillet
from phydrax.geometry.brep._intersection import NativePeriodEndpoint
from phydrax.geometry.brep._intersection_curve import (
    IntersectionCurve,
    IntersectionPCurve,
)
from phydrax.geometry.brep._patches import CylinderPatch, LineCurve, PlanePatch
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations


# Empirical `test_oblique_wedge_fillet_matches_independent_volume_moment_oracle`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_OBLIQUE_WEDGE_BASE_VOLUME_RTOL = 2e-12
# Empirical `test_oblique_wedge_fillet_matches_independent_volume_moment_oracle`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_OBLIQUE_FILLET_VOLUME_RTOL = 2e-10
# Empirical `test_oblique_wedge_fillet_matches_independent_volume_moment_oracle`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_OBLIQUE_FILLET_VOLUME_ATOL = 2e-10
# Empirical `test_oblique_wedge_fillet_matches_independent_volume_moment_oracle`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_OBLIQUE_FILLET_SURFACE_AREA_RTOL = 2e-10
# Empirical `test_oblique_wedge_fillet_matches_independent_volume_moment_oracle`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_OBLIQUE_FILLET_SURFACE_AREA_ATOL = 2e-10
# Empirical `test_oblique_cap_retains_shared_native_atlas_and_exact_source_incidence`: matched max |delta|=4.44089e-16; joint allowance fraction=2.22045e-06, margin=99.9998%.
_OBLIQUE_CAP_PLANE_RESIDUAL_ATOL = 2e-10
# Empirical `test_oblique_cap_retains_shared_native_atlas_and_exact_source_incidence`: matched max |delta|=2.72699e-14; joint allowance fraction=0.000136349, margin=99.9864%.
_OBLIQUE_CAP_CIRCLE_RESIDUAL_ATOL = 2e-10


_CONTRACT = phx.SpatialCoordinateContract.si()
_POLICY = phx.geometry.BRepTessellationPolicy(realize=False)


def _wedge() -> tuple[phx.geometry.BRepModel, int]:
    """Author x∈[0,2], y∈[0,3], z∈[0,4+x] by explicit source incidence.

    Face UV coordinates and orientation are authored independently of the
    fillet's offset-plane and intersection-atlas construction. The top plane's
    raw rational normal is (1,0,-1), with outward orientation -1.
    """
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 3.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 4.0),
            (2.0, 0.0, 6.0),
            (2.0, 3.0, 6.0),
            (0.0, 3.0, 4.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (
            PlanePatch((0, 0, 0), (0, 0, 1), (0, 1, 0)),
            ((0, 0), (4, 3)),
            (0, 4, 7, 3),
            ((0, 0), (4, 0), (4, 3), (0, 3)),
            1,
            "source:x_min",
        ),
        (
            PlanePatch((2, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((0, 0), (3, 6)),
            (1, 2, 6, 5),
            ((0, 0), (3, 0), (3, 6), (0, 6)),
            1,
            "source:x_max",
        ),
        (
            PlanePatch((0, 0, 0), (1, 0, 0), (0, 0, 1)),
            ((0, 0), (2, 6)),
            (0, 1, 5, 4),
            ((0, 0), (2, 0), (2, 6), (0, 4)),
            1,
            "source:y_min",
        ),
        (
            PlanePatch((0, 3, 0), (0, 0, 1), (1, 0, 0)),
            ((0, 0), (6, 2)),
            (3, 7, 6, 2),
            ((0, 0), (4, 0), (6, 2), (0, 2)),
            1,
            "source:y_max",
        ),
        (
            PlanePatch((0, 0, 0), (0, 1, 0), (1, 0, 0)),
            ((0, 0), (3, 2)),
            (0, 3, 2, 1),
            ((0, 0), (3, 0), (3, 2), (0, 2)),
            1,
            "source:bottom",
        ),
        (
            PlanePatch((0, 0, 4), (0, 1, 0), (1, 0, 1)),
            ((0, 0), (3, 2)),
            (4, 7, 6, 5),
            ((0, 0), (3, 0), (3, 2), (0, 2)),
            -1,
            "source:oblique_top",
        ),
    )
    builder = Builder()
    for point in points:
        builder.vertex(point)
    edges: dict[frozenset[int], int] = {}
    preparations = TrimRequestPreparations()
    with coordinate_source_request((points, faces)) as budget:
        for patch, box, vertices, coordinates, orientation, tag in faces:
            uv = {
                vertex: np.asarray(value, dtype=np.float64)
                for vertex, value in zip(vertices, coordinates, strict=True)
            }
            loop: list[int] = []
            for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
                key = frozenset((start, end))
                if key not in edges:
                    edges[key] = builder.edge(
                        LineCurve(points[start], points[end] - points[start]),
                        0.0,
                        1.0,
                        start,
                        end,
                    )
                edge = edges[key]
                first, last = builder.edge_vertices[edge]
                loop.append(
                    builder.coedge(
                        edge,
                        1 if first == start else -1,
                        LineCurve(uv[first], uv[last] - uv[first]),
                        preparations=preparations,
                        budget=budget,
                    )
                )
            builder.face(
                patch,
                np.asarray(box, dtype=np.float64),
                [loop],
                tag,
                None,
                orientation=orientation,
            )
        model = finish(
            builder,
            solid=True,
            coordinate_contract=_CONTRACT,
            source_id="oblique-fillet-wedge-source",
            tessellation=_POLICY,
            preparations=preparations,
            budget=budget,
        )
    return model, edges[frozenset((0, 4))]


def test_oblique_wedge_fillet_matches_independent_volume_moment_oracle() -> None:
    source, edge = _wedge()
    source_volume = float(
        phx.geometry.prepare_brep_query(source).measures.solid_volumes[0]
    )
    assert source_volume == pytest.approx(30.0, rel=_OBLIQUE_WEDGE_BASE_VOLUME_RTOL)
    radius = 0.25
    result = brep_fillet(source, (source.edge_ids[edge],), radius, tessellation=_POLICY)
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    # The removed planar corner has area A and first x moment A*xbar.
    # Integrating the independently specified wedge height 4+x gives 4*A+A*xbar;
    # no production carrier, offset, or trim construction enters this oracle.
    area = (1.0 - math.pi / 4.0) * radius**2
    xbar = radius * (5.0 / 6.0 - math.pi / 4.0) / (1.0 - math.pi / 4.0)
    expected = 30.0 - area * (4.0 + xbar)
    measures = phx.geometry.prepare_brep_query(model).measures
    actual = float(measures.solid_volumes[0])
    assert actual == pytest.approx(
        expected, rel=_OBLIQUE_FILLET_VOLUME_RTOL, abs=_OBLIQUE_FILLET_VOLUME_ATOL
    )
    # Side strips integrate the authored cap heights; the oblique cap has
    # constant metric factor sqrt(2), and the cylinder's height is 4+x.
    source_area = 12.0 + 18.0 + 10.0 + 10.0 + 6.0 + 6.0 * math.sqrt(2.0)
    side_loss = 8.0 * radius + radius**2 / 2.0
    cap_loss = (1.0 + math.sqrt(2.0)) * area
    cylinder_area = radius * (2.0 * math.pi + radius * (math.pi / 2.0 - 1.0))
    expected_area = source_area - side_loss - cap_loss + cylinder_area
    assert float(np.sum(measures.face_areas)) == pytest.approx(
        expected_area,
        rel=_OBLIQUE_FILLET_SURFACE_AREA_RTOL,
        abs=_OBLIQUE_FILLET_SURFACE_AREA_ATOL,
    )
    assert result.g1_angle_bounds.shape == (2, 2)
    bounds = np.asarray(result.g1_angle_bounds)
    assert np.all(bounds[:, 0] == 0.0)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds[:, 1] < 1e-10)
    assert result.source is None
    assert result.sensitivity_status == "unsupported"
    assert "oblique-cap sensitivity" in result.sensitivity_reason


def test_oblique_cap_retains_shared_native_atlas_and_exact_source_incidence() -> None:
    source, selected = _wedge()
    radius = 0.25
    result = brep_fillet(
        source, (source.edge_ids[selected],), radius, tessellation=_POLICY
    )
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    geometry = model.require_geometry()
    assert model.physical_tags[:-1] == source.physical_tags
    assert all(
        patch is original
        for patch, original in zip(model.patches[:-1], source.patches, strict=True)
    )
    blend_face = len(model.patches) - 1
    cylinder = model.patches[blend_face]
    assert isinstance(cylinder, CylinderPatch)
    assert float(cylinder.radius) == radius
    assert len(geometry.vertex_points) == (
        len(source.require_geometry().vertex_points) + 2
    )
    assert all(len(uses) == 2 for uses in model.topology.edge_faces)
    cap_faces = tuple(
        source.physical_tags.index(tag) for tag in ("source:bottom", "source:oblique_top")
    )
    assert model.trim_domains[blend_face] is not None
    assert np.all(np.isfinite(np.asarray(model.coedge_deviation_bounds)))
    for cap_face in cap_faces:
        assert model.trim_domains[cap_face] is not None
        shared = set(model.topology.face_edges[cap_face]) & set(
            model.topology.face_edges[blend_face]
        )
        assert len(shared) == 1
        edge = next(iter(shared))
        carrier = geometry.curves[geometry.edge_curves[edge]]
        assert isinstance(carrier, IntersectionCurve)
        assert carrier.first.patch is source.patches[cap_face]
        assert carrier.second.patch is cylinder
        assert carrier.fully_certified
        assert np.all(carrier.certified)
        assert np.all(carrier.nodes_certified)
        assert np.all(carrier.transitions_certified)
        assert np.all(np.asarray(carrier.contraction) < 1.0)
        assert carrier.start_kind == carrier.end_kind == "boundary"
        assert np.array_equal(np.asarray(carrier.node_axes)[[0, -1]], (2, 2))
        for face, side in ((cap_face, "first"), (blend_face, "second")):
            coedge = next(
                c
                for loop in geometry.face_loops[face]
                for c in loop
                if geometry.coedge_edges[c] == edge
            )
            pcurve = geometry.pcurves[coedge]
            assert isinstance(pcurve, IntersectionPCurve)
            assert pcurve.curve is carrier
            assert pcurve.side == side
            assert not pcurve.reversed
        if source.physical_tags[cap_face] == "source:oblique_top":
            points = np.asarray(
                carrier.evaluate(jnp.linspace(0.0, float(carrier.num_charts), 17)).point
            )
            # A plane section of the independent implicit cylinder equation is
            # an ellipse, not a fitted circle in the oblique plane's metric.
            np.testing.assert_allclose(
                points[:, 2] - points[:, 0],
                4.0,
                atol=_OBLIQUE_CAP_PLANE_RESIDUAL_ATOL,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                (points[:, 0] - radius) ** 2 + (points[:, 1] - radius) ** 2,
                radius**2,
                atol=_OBLIQUE_CAP_CIRCLE_RESIDUAL_ATOL,
                rtol=0.0,
            )
    # The unaffected source carriers survive by source incidence, even though
    # deleting the selected source edge changes their represented edge indices.
    old = source.require_geometry()
    affected = {selected}
    for cap_face in cap_faces:
        for edge in source.topology.face_edges[cap_face]:
            if set(old.edge_vertices[edge]) & set(old.edge_vertices[selected]):
                affected.add(edge)
    retained = {id(curve) for curve in geometry.curves}
    for edge, curve_index in enumerate(old.edge_curves):
        if edge not in affected:
            assert id(old.curves[curve_index]) in retained
    exact_trims: set[Fraction] = set()
    for source_edge in affected - {selected}:
        source_curve = old.curves[old.edge_curves[source_edge]]
        target_edge = next(
            edge
            for edge, curve_index in enumerate(geometry.edge_curves)
            if geometry.curves[curve_index] is source_curve
        )
        for endpoint, root in enumerate(geometry.edge_endpoint_roots[target_edge]):
            if root is None:
                continue
            assert isinstance(root, NativePeriodEndpoint)
            assert root.curve is source_curve
            assert root.turns == 0
            exact_trims.add(root.rational)
            for coedge, edge in enumerate(geometry.coedge_edges):
                if edge != target_edge:
                    continue
                use = geometry.coedge_endpoint_roots[coedge][endpoint]
                assert isinstance(use, NativePeriodEndpoint)
                assert use.curve is geometry.pcurves[coedge]
                assert use.rational == root.rational
                assert use.turns == 0
    assert exact_trims == {
        Fraction(1, 12),
        Fraction(11, 12),
        Fraction(1, 8),
        Fraction(7, 8),
    }


def test_oblique_wedge_fillet_refuses_neighbor_consumption_not_obliqueness() -> None:
    source, edge = _wedge()
    result = brep_fillet(source, (source.edge_ids[edge],), 2.0, tessellation=_POLICY)
    assert result.status == "unsupported"
    assert result.model is None
    assert "neighboring edge" in result.reason or "face vanishes" in result.reason


def test_oblique_wedge_requested_radius_range_has_whole_source_clearance_proof() -> None:
    source, edge = _wedge()
    result = brep_fillet(
        source,
        (source.edge_ids[edge],),
        0.25,
        tessellation=_POLICY,
        include_parameter_range=True,
    )
    assert result.status == "certified", result.reason
    certificate = result.parameter_range
    assert certificate is not None and certificate.parameter == "radius"
    # Independent conservative bound: a full circle centered at (r,r) has
    # x support 2r, so x<2 proves remote-face clearance while r<1. The
    # selected quarter-circle reaches its neighboring trim at r=2.
    assert certificate.interval == (0.0, 1.0)
    assert certificate.end_status == ("attained_exactly", "certified_inner_bound")
    assert certificate.evidence
    refusal = brep_fillet(source, (source.edge_ids[edge],), 2.0, tessellation=_POLICY)
    assert refusal.status == "unsupported"
    assert refusal.parameter_range is not None
    assert refusal.parameter_range.interval == certificate.interval
