# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Quadratic closed-cell injectivity is earned from the actual source Jacobian."""

from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from phydrax.discretization._cell_geometry_validity import CellValidityCertificate
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    coordinate_expressions,
    Polynomial,
    RationalPolynomial,
)
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.geometry._embedding_cells import Cell, vertices
from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateLimits
from phydrax.geometry._mapped_embedding import (
    _quadratic_rank_injective,
    injective,
    root_injective,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mapped_embedding import complex_cube_map
from tests._support.mesh_certificate_fixtures import TETRAHEDRON
from tests._support.meshing_fixtures import finding_checks


type Transform = Callable[[np.ndarray], np.ndarray]


def _sheared(points: np.ndarray) -> np.ndarray:
    """(u, v + 4u^2, w): injective with unit Jacobian determinant, not monotone."""
    result = points.copy()
    result[:, 1] += 4.0 * points[:, 0] ** 2
    return result


def _folded(points: np.ndarray) -> np.ndarray:
    """(3u - 2u^2, v, w): u = 1/2 and u = 1 share an image; det changes sign."""
    result = points.copy()
    result[:, 0] = 3.0 * points[:, 0] - 2.0 * points[:, 0] ** 2
    return result


def _touching(points: np.ndarray) -> np.ndarray:
    """(u^2, v, w): injective, but its determinant vanishes on the face u = 0."""
    result = points.copy()
    result[:, 0] = points[:, 0] ** 2
    return result


def _p2_tetrahedron(
    transform: Transform,
) -> tuple[CellMesh, CellGeometrySpec, CellGeometryElement, tuple[tuple[Fraction, ...], ...]]:
    element = coordinate_lagrange_element("tetrahedron", 2)
    points = transform(np.asarray(element.reference_nodes, dtype=np.float64))
    mesh = CellMesh.from_tetrahedra(
        transform(TETRAHEDRON), np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    geometry = CellGeometrySpec(
        {"tetrahedra": element},
        {"tetrahedra": np.arange(points.shape[0], dtype=np.int32)[None]},
        points,
    )
    local = tuple(tuple(Fraction(float(value)) for value in row) for row in points)
    return mesh, geometry, element, local


def _root_cell(
    element: CellGeometryElement,
    local: tuple[tuple[Fraction, ...], ...],
    budget: CoordinateEnclosureBudget,
) -> Cell:
    coordinates = coordinate_expressions(element, local, budget=budget)
    assert coordinates is not None
    return Cell(
        coordinates, "simplex", "tetrahedron", 3, (0, 1, 2, 3), vertices("tetrahedron")
    )


_SHALLOW = MeshCertificateLimits(
    maximum_subdivision_depth=3, maximum_subdivision_pieces=10_000
)


def test_quadratic_rank_proves_what_root_monotonicity_cannot_at_depth_zero() -> None:
    _, _, element, local = _p2_tetrahedron(_sheared)
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    with coordinate_reference_budget(limits) as budget:
        cell = _root_cell(element, local, budget)
        # Bounds only: the general route's fixed projection is not monotone here.
        assert injective(cell, limits, EmbeddingState([], []), budget=budget) is not None
        assert _quadratic_rank_injective(cell, limits, 1, budget=budget) == (True, 1)
        state = EmbeddingState([], [])
        assert root_injective(cell, element, local, limits, state, budget=budget) is None
        assert state.subdivision_pieces == 1


def test_quadratic_rank_respects_the_shared_piece_allowance() -> None:
    _, _, element, local = _p2_tetrahedron(_sheared)
    limits = MeshCertificateLimits(maximum_subdivision_pieces=4)
    with coordinate_reference_budget(limits) as budget:
        cell = _root_cell(element, local, budget)
        assert _quadratic_rank_injective(cell, limits, 0, budget=budget) == (False, 0)
        state = EmbeddingState([], [])
        state.subdivision_pieces = limits.maximum_subdivision_pieces
        assert (
            root_injective(cell, element, local, limits, state, budget=budget)
            == "local_injectivity_piece_budget"
        )
        assert state.subdivision_pieces == limits.maximum_subdivision_pieces


@pytest.mark.parametrize(
    "transform", (_folded, _touching), ids=("sign-change-fold", "boundary-singular")
)
def test_singular_quadratic_jacobians_never_earn_rank(transform: Transform) -> None:
    _, _, element, local = _p2_tetrahedron(transform)
    with coordinate_reference_budget(_SHALLOW) as budget:
        cell = _root_cell(element, local, budget)
        proved, work = _quadratic_rank_injective(cell, _SHALLOW, 10_000, budget=budget)
        assert not proved and 0 < work <= 10_000
        state = EmbeddingState([], [])
        assert (
            root_injective(cell, element, local, _SHALLOW, state, budget=budget)
            is not None
        )


def test_fabricated_same_binding_validity_does_not_certify_a_fold() -> None:
    mesh, geometry, _, _ = _p2_tetrahedron(_folded)
    honest = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    assert honest.invalid_count == 1
    forged = CellValidityCertificate(
        np.zeros_like(np.asarray(honest.status)),
        np.ones((1,), dtype=np.float64),
        np.ones((1,), dtype=np.float64),
        np.asarray(honest.depth),
        block_names=honest.block_names,
        block_offsets=honest.block_offsets,
        unsupported_block_names=honest.unsupported_block_names,
        unresolved_reasons=(),
        geometry_id=honest.geometry_id,
        geometry_layout_id=honest.geometry_layout_id,
        topology_id=honest.topology_id,
        policy_id=honest.policy_id,
    )
    assert forged.all_certified
    certificate = phx.geometry.certify_global_embedding(
        mesh, geometry, forged, limits=_SHALLOW
    )
    assert certificate.status != "certified"
    refusals = finding_checks(certificate, "unresolved") | finding_checks(
        certificate, "violated"
    )
    assert any(
        check.startswith("local_injectivity") or check == "mapped_self_contact"
        for check in refusals
    )


def test_sheared_p2_cell_has_no_local_injectivity_finding_at_depth_zero() -> None:
    mesh, geometry, _, _ = _p2_tetrahedron(_sheared)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    certificate = phx.geometry.certify_global_embedding(
        mesh, geometry, validity, limits=MeshCertificateLimits(maximum_subdivision_depth=0)
    )
    assert "mapped_local_injectivity" in certificate.evaluated_checks
    refusals = finding_checks(certificate, "unresolved") | finding_checks(
        certificate, "violated"
    )
    assert not any(check.startswith("local_injectivity") for check in refusals)


def test_embedded_quadratic_maps_earn_rank_from_a_maximal_minor() -> None:
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    curve = Cell(
        ({(1,): Fraction(1)}, {(2,): Fraction(1, 3), (1,): Fraction(1, 7)}),
        "box",
        "interval",
        1,
        (0, 1),
        vertices("interval"),
    )
    surface = Cell(
        (
            {(1, 0): Fraction(1)},
            {(0, 1): Fraction(1)},
            {(2, 0): Fraction(1, 3), (0, 2): Fraction(2, 5), (1, 1): Fraction(1, 7)},
        ),
        "simplex",
        "triangle",
        2,
        (0, 1, 2),
        vertices("triangle"),
    )
    with coordinate_reference_budget(limits) as budget:
        assert _quadratic_rank_injective(curve, limits, 1, budget=budget) == (True, 1)
        assert _quadratic_rank_injective(surface, limits, 1, budget=budget) == (True, 1)


def test_nonquadratic_rational_and_pyramid_maps_keep_the_general_route() -> None:
    # Cubing has a positive determinant on this box but identifies 1 +/- i*sqrt(3).
    cubic = Cell(
        complex_cube_map(1), "box", "quadrilateral", 2, (0, 1, 2, 3),
        vertices("quadrilateral"),
    )
    linear: Polynomial = {(1, 0): Fraction(1)}
    quotient = Cell(
        (
            RationalPolynomial(linear, {(0, 0): Fraction(1), (0, 1): Fraction(1, 8)}),
            {(0, 1): Fraction(1)},
        ),
        "simplex",
        "triangle",
        2,
        (0, 1, 2),
        vertices("triangle"),
    )
    pyramid = Cell(
        (
            {(1, 0, 0): Fraction(1)},
            {(0, 1, 0): Fraction(1)},
            {(0, 0, 1): Fraction(1)},
        ),
        "box",
        "pyramid",
        3,
        (0, 1, 2, 3, 4),
        vertices("pyramid"),
    )
    with coordinate_reference_budget(_SHALLOW) as budget:
        for cell in (cubic, quotient, pyramid):
            assert _quadratic_rank_injective(cell, _SHALLOW, 10_000, budget=budget) == (
                False,
                0,
            )
