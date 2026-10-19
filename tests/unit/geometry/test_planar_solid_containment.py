#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Planar parity contracts through the certificate and prepared-query boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep import (
    brep_box,
    brep_extrusion,
    BRepModel,
    BRepTessellationPolicy,
    PlanarProfile,
    prepare_brep_query,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._containment import certify_solid_containment


@pytest.fixture(scope="module", params=[-20, 0, 20], ids=["small", "unit", "large"])
def grazing_prism(request: pytest.FixtureRequest) -> tuple[BRepModel, float]:
    """Extrude a polygon with vertices on all three former fixed parity rays."""
    scale = float(np.ldexp(1.0, request.param))
    # From (0, 0, 0), the former directions hit vertical edges at the first
    # two listed positive vertices and the top-cap vertex at the third.
    polygon = np.asarray(
        [
            (-2.0, -2.0),
            (2.0, -2.0),
            (1.0, 0.317),
            (0.487, 1.0),
            (0.211, 0.719),
            (-2.0, 2.0),
        ],
        dtype=np.float64,
    )
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, -scale)), ProfileLoop.polygon(scale * polygon)
    )
    model = brep_extrusion(
        profile,
        (0.0, 0.0, 2.0 * scale),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    return model, scale


def test_polygon_vertex_grazing_has_certified_parity(
    grazing_prism: tuple[BRepModel, float],
) -> None:
    model, _ = grazing_prism
    certificate = certify_solid_containment(
        model,
        np.zeros((3,), dtype=np.float64),
        0,
        maximum_boxes=1,
        maximum_depth=0,
    )
    assert certificate.complete
    assert certificate.inside
    assert certificate.crossings % 2 == 1
    assert certificate.unresolved_faces == ()
    assert not certificate.resource_exhausted


def test_prepared_query_polygon_vertex_grazing_is_resolved(
    grazing_prism: tuple[BRepModel, float],
) -> None:
    model, _ = grazing_prism
    result = prepare_brep_query(model).contains(np.zeros((1, 3), dtype=np.float64))
    assert bool(result.inside[0])
    assert not bool(result.unresolved[0])
    assert not bool(result.resource_exhausted[0])


@pytest.mark.parametrize(
    ("permutation", "signs"),
    [
        pytest.param((0, 1, 2), (1, 1, 1), id="identity"),
        pytest.param((2, 0, 1), (-1, 1, -1), id="signed-cycle"),
        pytest.param((1, 0, 2), (1, -1, 1), id="signed-swap"),
    ],
)
@pytest.mark.parametrize("exponent", [-20, 0, 20], ids=["small", "unit", "large"])
def test_planar_parity_is_invariant_under_binary_transforms(
    permutation: tuple[int, int, int],
    signs: tuple[int, int, int],
    exponent: int,
) -> None:
    scale = float(np.ldexp(1.0, exponent))
    upper = np.asarray([2.0, 4.0, 8.0], dtype=np.float64)[list(permutation)]
    signed_upper = scale * upper * np.asarray(signs, dtype=np.float64)
    model = brep_box(
        np.minimum(0.0, signed_upper),
        np.maximum(0.0, signed_upper),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    # The center is interior under every signed permutation and binary scale.
    # No distance tolerance is used: all coordinates are represented exactly.
    certificate = certify_solid_containment(
        model,
        0.5 * signed_upper,
        0,
        maximum_boxes=1,
        maximum_depth=0,
    )
    assert certificate.complete and certificate.inside
    assert certificate.crossings % 2 == 1
    assert not certificate.resource_exhausted


@pytest.fixture(scope="module")
def polygonal_cavity() -> BRepModel:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, -1.0)),
        ProfileLoop.polygon(((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0))),
        (ProfileLoop.polygon(((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5))),),
    )
    return brep_extrusion(
        profile,
        (0.0, 0.0, 2.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )


@pytest.mark.parametrize(
    ("point", "inside"),
    [
        pytest.param((1.0, 0.5, 0.0), True, id="vertex-height-material"),
        pytest.param((0.0, 0.0, 0.0), False, id="cavity-interior"),
        pytest.param((0.0, 0.0, -1.0), False, id="cap-plane-outside-trim"),
    ],
)
def test_polygon_half_open_winding_respects_holes(
    polygonal_cavity: BRepModel,
    point: tuple[float, float, float],
    inside: bool,
) -> None:
    certificate = certify_solid_containment(
        polygonal_cavity,
        np.asarray(point, dtype=np.float64),
        0,
        maximum_boxes=1,
        maximum_depth=0,
    )
    assert certificate.complete
    assert certificate.inside is inside
    assert not certificate.resource_exhausted


@pytest.mark.parametrize(
    "point",
    [
        pytest.param((0.5, 0.0, 0.0), id="hole-face"),
        pytest.param((2.0, 2.0, 0.0), id="outer-edge"),
        pytest.param((2.0, 2.0, 1.0), id="outer-vertex"),
        pytest.param((1.0, 0.5, 1.0), id="cap-face"),
    ],
)
def test_polygon_boundary_does_not_establish_parity(
    polygonal_cavity: BRepModel,
    point: tuple[float, float, float],
) -> None:
    certificate = certify_solid_containment(
        polygonal_cavity,
        np.asarray(point, dtype=np.float64),
        0,
        maximum_boxes=1,
        maximum_depth=0,
    )
    assert not certificate.complete
    assert not certificate.inside
    assert certificate.unresolved_faces
    assert not certificate.resource_exhausted


def test_planar_parity_retains_operation_refusal(
    polygonal_cavity: BRepModel,
) -> None:
    certificate = certify_solid_containment(
        polygonal_cavity,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        0,
        maximum_boxes=1,
        maximum_depth=0,
        maximum_operations=1,
    )
    assert not certificate.complete
    assert certificate.resource_exhausted
    assert certificate.owner_operations <= 1
    assert certificate.unresolved_faces


def test_zero_operation_allowance_refuses_before_bounding_box_decision(
    polygonal_cavity: BRepModel,
) -> None:
    certificate = certify_solid_containment(
        polygonal_cavity,
        np.asarray([4.0, 0.0, 0.0], dtype=np.float64),
        0,
        maximum_boxes=1,
        maximum_depth=0,
        maximum_operations=0,
    )
    assert not certificate.complete
    assert certificate.resource_exhausted
    assert certificate.owner_operations == 0
    assert certificate.unresolved_faces == polygonal_cavity.topology.solid_faces[0]
