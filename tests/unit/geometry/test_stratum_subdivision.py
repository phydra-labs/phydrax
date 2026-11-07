from __future__ import annotations

from fractions import Fraction
from typing import Literal, Never

import numpy as np
import numpy.typing as npt
import pytest
import scipy.optimize

import phydrax.geometry._stratum_subdivision as subdivision
from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    compose_expression,
    constant,
    evaluate_expression,
    Expression,
    multiply,
    scale,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._global_embedding import _stratum_residuals
from phydrax.geometry._mapped_embedding import (
    _float_search_preconditioner,
    _singular_control_hull,
    monotone_preconditioner,
    prove_mapped_cell_injective,
)
from phydrax.geometry._restricted_embedding import restriction_trace_equal
from phydrax.geometry._stratum_subdivision import (
    glued_facet_monotone,
    StratumSubdivisionCache,
    subdivided_stratum_separation,
)


type _Point = tuple[Fraction, ...]

_PRISM = tuple(
    (Fraction(u), Fraction(v), Fraction(w))
    for w in (0, 1)
    for u, v in ((0, 0), (1, 0), (0, 1))
)
_TETRAHEDRON = tuple(
    (Fraction(u), Fraction(v), Fraction(w))
    for u, v, w in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
)
_HEXAHEDRON = tuple(
    (Fraction(u), Fraction(v), Fraction(w))
    for u, v, w in (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
)


def _bend(
    shear: Fraction = Fraction(1, 10), lift: Fraction = Fraction(1, 5)
) -> tuple[Expression, ...]:
    """A smooth injective map of space shared by every cell (equal traces)."""
    x, y, z = axes(3)
    return (
        add(x, scale(multiply(y, y), shear)),
        y,
        add(z, scale(multiply(x, y), lift)),
    )


def _cell(
    global_id: int,
    kind: str,
    vertex_ids: tuple[int, ...],
    corners: tuple[_Point, ...],
    physical: tuple[Expression, ...],
    /,
) -> EmbeddingCell:
    """``physical`` composed with the straight cell through ``corners``."""
    references = {
        "prism": _PRISM,
        "hexahedron": _HEXAHEDRON,
    }.get(kind, _TETRAHEDRON)
    u, v, w = axes(3)
    origin = corners[0]
    if kind in ("prism", "hexahedron"):
        edges = (
            (corners[1], corners[3], corners[4])
            if kind == "hexahedron"
            else (corners[1], corners[2], corners[3])
        )
    else:
        edges = corners[1:]
    straight = tuple(
        add(
            constant(origin[axis], 3),
            add(
                scale(u, edges[0][axis] - origin[axis]),
                add(
                    scale(v, edges[1][axis] - origin[axis]),
                    scale(w, edges[2][axis] - origin[axis]),
                ),
            ),
        )
        for axis in range(3)
    )
    coordinates = tuple(
        compose_expression(expression, straight) for expression in physical
    )
    images = tuple(
        tuple(evaluate_expression(expression, point) for expression in coordinates)
        for point in references
    )
    return EmbeddingCell(
        global_id=global_id,
        block_name=kind,
        local_index=global_id,
        kind=kind,
        dimension=3,
        domain="prism"
        if kind == "prism"
        else "box"
        if kind == "hexahedron"
        else "simplex",
        vertex_indices=tuple(range(len(vertex_ids))),
        vertex_ids=vertex_ids,
        reference_vertices=references,
        coordinate_expressions=coordinates,
        corner_images=images,
        affine_offset=None,
        affine_matrix=None,
        image_lower=tuple(
            float(min(point[axis] for point in images)) - 1.0 for axis in range(3)
        ),
        image_upper=tuple(
            float(max(point[axis] for point in images)) + 1.0 for axis in range(3)
        ),
        facet_routes=tuple(
            tuple(int(index) for index in face)
            for face in reference_cell_topology(kind).entities[2]
        ),
    )


def _p(x: float, y: float, z: float, /) -> _Point:
    return Fraction(x), Fraction(y), Fraction(z)


def _layer_prism(
    global_id: int,
    ids: tuple[int, ...],
    base: tuple[tuple[float, float], ...],
    /,
) -> EmbeddingCell:
    corners = (*(_p(*point, 0) for point in base), *(_p(*point, 1) for point in base))
    return _cell(global_id, "prism", ids, corners, _bend())


@pytest.mark.parametrize(
    ("kind", "corners"), [("tetrahedron", _TETRAHEDRON), ("prism", _PRISM)]
)
def test_vertex_residual_controls_equal_exact_composition(
    kind: str,
    corners: tuple[_Point, ...],
) -> None:
    cell = _cell(10, kind, tuple(range(len(corners))), corners, _bend())
    for vertex in cell.vertex_ids:
        residuals = _stratum_residuals(cell, (vertex,))
        assert residuals is not None
        piece = subdivision._root_piece(cell, (vertex,), residuals)
        assert piece is not None
        composed = tuple(
            compose_expression(residual, cell.coordinates) for residual in residuals
        )
        degree = subdivision._degree(composed, piece.groups, cell.dimension)
        assert degree == piece.residual_degree
        assert subdivision._controls(composed, piece.groups, degree, cell.dimension) == (
            piece.residual
        )


def test_geometry_control_cache_is_bound_to_the_cell_operand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(12, "tetrahedron", tuple(range(4)), _TETRAHEDRON, _bend())
    changed = _cell(
        12, "tetrahedron", tuple(range(4)), _TETRAHEDRON, _bend(Fraction(1, 8))
    )
    cache = StratumSubdivisionCache()
    original = subdivision._controls
    calls = 0

    def count_controls(
        polynomials: tuple[Expression, ...],
        groups: tuple[int, ...],
        degree: tuple[int, ...],
        dimension: int,
        /,
    ) -> subdivision._Controls | None:
        nonlocal calls
        calls += 1
        return original(polynomials, groups, degree, dimension)

    monkeypatch.setattr(subdivision, "_controls", count_controls)
    first = subdivision._root_piece(
        cell, (0,), _stratum_residuals(cell, (0,)) or (), cache=cache
    )
    assert first is not None
    assert calls == 1
    second = subdivision._root_piece(
        cell, (1,), _stratum_residuals(cell, (1,)) or (), cache=cache
    )
    assert second is not None
    assert calls == 1
    other = subdivision._root_piece(
        changed, (0,), _stratum_residuals(changed, (0,)) or (), cache=cache
    )
    assert other is not None
    assert calls == 2
    assert other.geometry != first.geometry


def test_batched_vertex_cones_prove_only_exact_opposite_sides() -> None:
    from phydrax.geometry._global_embedding import _VertexConeFilter

    first = _cell(20, "tetrahedron", (0, 1, 2, 3), _TETRAHEDRON, _bend())
    opposite = _cell(
        21,
        "tetrahedron",
        (0, 4, 5, 6),
        tuple(tuple(-value for value in point) for point in _TETRAHEDRON),
        _bend(),
    )
    overlap = _cell(22, "tetrahedron", (0, 7, 8, 9), _TETRAHEDRON, _bend())
    x, y, z = _bend()
    moved = _cell(
        23,
        "tetrahedron",
        (0, 10, 11, 12),
        _TETRAHEDRON,
        (add(x, constant(Fraction(1, 10), 3)), y, z),
    )
    remote = _cell(
        24,
        "tetrahedron",
        (13, 14, 15, 16),
        tuple((point[0] + 3, point[1], point[2]) for point in _TETRAHEDRON),
        _bend(),
    )
    duplicate = _cell(25, "tetrahedron", (17, 18, 19, 20), _TETRAHEDRON, _bend())
    prepared = _VertexConeFilter.prepare(
        (first, opposite, overlap, moved, remote, duplicate)
    )
    if prepared is None:
        pytest.skip("native meshcore is unavailable")
    result = prepared.certify(
        np.asarray([0, 0], dtype=np.intp), np.asarray([1, 2], dtype=np.intp)
    )
    assert result.tolist() == [True, False]
    assert prepared.matching_corners(
        np.asarray([0, 0, 0], dtype=np.intp),
        np.asarray([1, 2, 3], dtype=np.intp),
    ).tolist() == [True, False, False]
    assert prepared.certify_disjoint(
        np.asarray([0, 0], dtype=np.intp),
        np.asarray([4, 5], dtype=np.intp),
    ).tolist() == [True, False]
    residuals = _stratum_residuals(first, (0,))
    assert residuals is not None
    assert subdivided_stratum_separation(
        first, opposite, (0,), residuals, StratumSubdivisionCache()
    ) == (1, 0)


def test_batched_vertex_dispatch_matches_full_exact_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._cell_mesh import CellBlock, CellMesh
    from phydrax.discretization.fem._reference import lagrange_element
    from phydrax.geometry import _global_embedding as embedding
    from phydrax.geometry._embedding_cells import prepare_embedding_cells

    element = lagrange_element("tetrahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)

    def mapped(points: np.ndarray) -> np.ndarray:
        result = points.copy()
        result[:, 0] += points[:, 1] ** 2 / 8
        result[:, 2] += points[:, 0] * points[:, 1] / 8
        return result

    vertices = np.concatenate((np.zeros((1, 3)), np.eye(3), -np.eye(3)), axis=0)
    mesh = CellMesh(
        mapped(vertices),
        (CellBlock("tet", "tetrahedron", np.asarray(((0, 1, 2, 3), (0, 4, 5, 6)))),),
    )
    bank = np.concatenate((mapped(nodes), mapped(-nodes)))
    geometry = CellGeometrySpec(
        {"tet": element},
        {"tet": np.arange(2 * len(nodes)).reshape(2, len(nodes))},
        bank,
    )
    cells = prepare_embedding_cells(mesh, geometry)
    prepared = embedding._VertexConeFilter.prepare(cells)
    if prepared is None:
        pytest.skip("native meshcore is unavailable")
    left = np.asarray([0], dtype=np.intp)
    right = np.asarray([1], dtype=np.intp)
    assert prepared.certify(left, right).tolist() == [True]
    assert prepared.matching_corners(left, right).tolist() == [True]
    values: list[str] = []
    findings = embedding._nonperiodic_contacts(mesh, geometry, cells, values)
    assert not findings
    assert any("shared-stratum:1:cone-pieces:1" in value for value in values)
    monkeypatch.setattr(embedding._VertexConeFilter, "prepare", lambda *_: None)
    expected_values: list[str] = []
    expected = embedding._nonperiodic_contacts(mesh, geometry, cells, expected_values)
    assert findings == expected
    assert values == expected_values


_FIRST = _layer_prism(0, (0, 1, 2, 3, 4, 5), ((0, 0), (1, 0), (0, 1)))


def test_curved_prisms_sharing_a_quadrilateral_meet_only_in_it() -> None:
    # The glued second prism completes a convex quadrilateral base: one
    # constant preconditioner makes the glued map monotone.
    neighbour = _layer_prism(1, (1, 6, 2, 4, 7, 5), ((1, 0), (1, 1), (0, 1)))
    # One proof cache: the first prism's Jacobian controls serve both contacts.
    cache = StratumSubdivisionCache()
    assert glued_facet_monotone(_FIRST, neighbour, (1, 2, 4, 5), cache)
    # A positively oriented neighbour folded back over the first prism is refused.
    folded = _layer_prism(2, (1, 2, 6, 4, 5, 7), ((1, 0), (0, 1), (0.25, 0.25)))
    assert not glued_facet_monotone(_FIRST, folded, (1, 2, 4, 5), cache)


def test_prism_boundary_separation_checks_all_premises() -> None:
    from dataclasses import replace

    from phydrax.geometry._global_embedding import _prism_shared_facet_boundary_separation

    neighbour = _layer_prism(1, (1, 6, 2, 4, 7, 5), ((1, 0), (1, 1), (0, 1)))
    folded = _layer_prism(2, (1, 2, 6, 4, 5, 7), ((1, 0), (0, 1), (0.25, 0.25)))
    shared = (1, 2, 4, 5)
    assert prove_mapped_cell_injective(_FIRST) is not None
    assert prove_mapped_cell_injective(neighbour) is not None
    assert (
        _prism_shared_facet_boundary_separation(
            _FIRST, neighbour, shared, first_injective=True, second_injective=True
        )
        is not None
    )
    assert (
        _prism_shared_facet_boundary_separation(
            _FIRST, neighbour, shared, first_injective=False, second_injective=True
        )
        is None
    )
    assert (
        _prism_shared_facet_boundary_separation(
            _FIRST, folded, shared, first_injective=True, second_injective=True
        )
        is None
    )
    x, y, z = _bend()
    bump = scale(multiply(y, add(constant(1, 3), scale(y, -1))), Fraction(1, 10))
    corners = (
        *(_p(*point, 0) for point in ((1, 0), (1, 1), (0, 1))),
        *(_p(*point, 1) for point in ((1, 0), (1, 1), (0, 1))),
    )
    nonconforming = _cell(3, "prism", (1, 6, 2, 4, 7, 5), corners, (add(x, bump), y, z))
    assert restriction_trace_equal(_FIRST, nonconforming, shared) is False
    assert (
        _prism_shared_facet_boundary_separation(
            _FIRST, nonconforming, shared, first_injective=True, second_injective=True
        )
        is None
    )
    routes = list(neighbour.facet_routes)
    index = next(i for i, face in enumerate(routes) if len(face) == 4)
    a, b, c, d = routes[index]
    routes[index] = (a, c, b, d)
    malformed = replace(neighbour, facet_routes=tuple(routes))
    assert (
        _prism_shared_facet_boundary_separation(
            _FIRST, malformed, shared, first_injective=True, second_injective=True
        )
        is None
    )


def test_curved_hexahedra_shared_nonplanar_facet_and_overlap() -> None:
    first_corners = tuple(_p(float(x), float(y), float(z)) for x, y, z in _HEXAHEDRON)
    next_corners = tuple(_p(float(x + 1), float(y), float(z)) for x, y, z in _HEXAHEDRON)
    ids = (0, 1, 2, 3, 4, 5, 6, 7)
    next_ids = (1, 8, 9, 2, 5, 10, 11, 6)
    first = _cell(20, "hexahedron", ids, first_corners, _bend())
    neighbour = _cell(21, "hexahedron", next_ids, next_corners, _bend())
    overlap_corners = tuple(
        _p(float(1 - x), float(y), float(z)) for x, y, z in _HEXAHEDRON
    )
    overlap = _cell(22, "hexahedron", next_ids, overlap_corners, _bend())
    cache = StratumSubdivisionCache()
    assert glued_facet_monotone(first, neighbour, (1, 2, 5, 6), cache)
    assert not glued_facet_monotone(first, overlap, (1, 2, 5, 6), cache)
    assert restriction_trace_equal(first, neighbour, (1, 2, 5, 6)) is True
    x, y, z = _bend()
    _, parameter_y, _ = axes(3)
    bump = scale(
        multiply(parameter_y, add(constant(1, 3), scale(parameter_y, -1))),
        Fraction(1, 10),
    )
    distorted = _cell(23, "hexahedron", next_ids, next_corners, (add(x, bump), y, z))
    # Identical facet corners alone do not establish the intervening trace.
    assert all(
        first.corner_images[first.vertex_ids.index(vertex)]
        == distorted.corner_images[distorted.vertex_ids.index(vertex)]
        for vertex in (1, 2, 5, 6)
    )
    assert restriction_trace_equal(first, distorted, (1, 2, 5, 6)) is False


def test_curved_tetrahedron_on_a_prism_cap_meets_it_only_in_the_cap() -> None:
    corners = (_p(0, 0, 1), _p(1, 0, 1), _p(0, 1, 1), _p(0.25, 0.25, 1.5))
    cap = _cell(3, "simplex:3", (3, 4, 5, 8), corners, _bend())
    cache = StratumSubdivisionCache()
    assert glued_facet_monotone(_FIRST, cap, (3, 4, 5), cache)
    # A positively oriented tetrahedron sunk into the prism overlaps it.
    sunk = (corners[0], corners[2], corners[1], _p(0.25, 0.25, 0.5))
    inverted = _cell(4, "simplex:3", (3, 5, 4, 8), sunk, _bend())
    assert not glued_facet_monotone(_FIRST, inverted, (3, 4, 5), cache)


def test_prism_and_tetrahedron_sharing_a_vertex_are_separated_by_subdivision() -> None:
    corners = (_p(1, 0, 1), _p(2, 0, 1), _p(1.5, 0.5, 1), _p(1.5, 0.125, 1.75))
    neighbour = _cell(5, "simplex:3", (4, 9, 10, 11), corners, _bend())
    residuals = _stratum_residuals(_FIRST, (4,))
    assert residuals is not None
    counts = subdivided_stratum_separation(
        _FIRST, neighbour, (4,), residuals, StratumSubdivisionCache()
    )
    assert counts is not None and sum(counts) > 0


def test_batched_straight_edge_cones_reject_overlap_and_curved_trace() -> None:
    from phydrax.geometry._global_embedding import _VertexConeFilter

    first = _cell(60, "tetrahedron", (0, 1, 2, 3), _TETRAHEDRON, _bend())
    opposite_corners = (_p(0, 0, 0), _p(1, 0, 0), _p(0, -1, 0), _p(0, 0, -1))
    opposite = _cell(61, "tetrahedron", (0, 1, 4, 5), opposite_corners, _bend())
    overlap = _cell(62, "tetrahedron", (0, 1, 6, 7), _TETRAHEDRON, _bend())
    x, y, z = _bend()
    u, _, _ = axes(3)
    curved_z = add(
        z, scale(multiply(u, add(constant(1, 3), scale(u, -1))), Fraction(1, 8))
    )
    curved = _cell(63, "tetrahedron", (0, 1, 8, 9), opposite_corners, (x, y, curved_z))
    prepared = _VertexConeFilter.prepare((first, opposite, overlap, curved))
    if prepared is None:
        pytest.skip("native meshcore is unavailable")
    left = np.asarray([0, 0, 0], dtype=np.intp)
    right = np.asarray([1, 2, 3], dtype=np.intp)
    decisions = prepared.certify_straight_edges(left, right)
    assert decisions.tolist() == [True, False, False]
    assert prepared.matching_corners(left, right).tolist() == [True, False, True]
    assert restriction_trace_equal(first, curved, (0, 1)) is False


def test_batched_curved_edge_plane_requires_exact_trace_and_opposite_sides() -> None:
    from phydrax.geometry._global_embedding import _VertexConeFilter

    x, y, z = axes(3)
    bend = multiply(x, add(constant(1, 3), scale(x, -1)))
    physical = (x, y, add(z, scale(bend, Fraction(1, 8))))
    altered = (x, y, add(z, scale(bend, Fraction(1, 4))))
    first_corners = (_p(0, 0, 0), _p(1, 0, 0), _p(0, 1, 0), _p(0, 1, 1))
    other_corners = (_p(0, 0, 0), _p(1, 0, 0), _p(0, -1, 0), _p(0, -1, 1))
    first = _cell(70, "tetrahedron", (0, 1, 2, 3), first_corners, physical)
    opposite = _cell(71, "tetrahedron", (0, 1, 4, 5), other_corners, physical)
    overlap = _cell(72, "tetrahedron", (0, 1, 6, 7), first_corners, physical)
    nonconforming = _cell(73, "tetrahedron", (0, 1, 8, 9), other_corners, altered)
    prepared = _VertexConeFilter.prepare((first, opposite, overlap, nonconforming))
    if prepared is None:
        pytest.skip("native meshcore is unavailable")
    left = np.asarray([0, 0, 0], dtype=np.intp)
    right = np.asarray([1, 2, 3], dtype=np.intp)
    assert prepared.certify_curved_edges(left, right).tolist() == [True, False, False]
    assert restriction_trace_equal(first, opposite, (0, 1)) is True
    assert restriction_trace_equal(first, nonconforming, (0, 1)) is False


def test_affine_edge_residual_controls_match_exact_composition() -> None:
    x, y, z = axes(3)
    residuals = (add(x, scale(y, Fraction(2))), add(z, scale(x, Fraction(-3))))
    cell = _cell(41, "tetrahedron", (0, 1, 2, 3), _TETRAHEDRON, _bend())
    piece = subdivision._root_piece(cell, (0, 1), residuals)
    assert piece is not None
    composed = tuple(
        compose_expression(residual, cell.coordinates) for residual in residuals
    )
    assert piece.residual == subdivision._controls(
        composed, piece.groups, piece.residual_degree, cell.dimension
    )


def test_simplex_glued_controls_cache_binds_cell_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _cell(50, "tetrahedron", (0, 1, 2, 3), _TETRAHEDRON, _bend())
    corners = (_p(1, 1, 1), _p(1, 0, 0), _p(0, 1, 0), _p(0, 0, 1))
    second = _cell(51, "tetrahedron", (4, 1, 2, 3), corners, _bend())
    changed = _cell(51, "tetrahedron", (4, 1, 2, 3), corners, _bend(Fraction(1, 8)))
    cache = StratumSubdivisionCache()
    original = subdivision._jacobian_controls
    seen: list[EmbeddingCell] = []

    def counted(cell: EmbeddingCell) -> tuple | None:
        seen.append(cell)
        return original(cell)

    monkeypatch.setattr(subdivision, "_jacobian_controls", counted)
    expected = subdivision.glued_face_monotone(first, second, (1, 2, 3))
    seen.clear()
    assert subdivision.glued_face_monotone(first, second, (1, 2, 3), cache) == expected
    first_count = len(seen)
    assert subdivision.glued_face_monotone(first, second, (1, 2, 3), cache) == expected
    assert len(seen) == first_count
    expected_changed = subdivision.glued_face_monotone(first, changed, (1, 2, 3))
    seen.clear()
    assert (
        subdivision.glued_face_monotone(first, changed, (1, 2, 3), cache)
        == expected_changed
    )
    assert all(cell is not second for cell in seen)
    assert any(cell is changed for cell in seen)
    entry = cache.corner_affine_inverses[id(first)]
    assert entry[0] is first and entry[1] is not None
    for opposite in range(4):
        reference_edges = subdivision._edge_matrix(first.reference_vertices, opposite)
        reference_inverse = subdivision._inverse(reference_edges)
        assert reference_inverse is not None
        physical_edges = subdivision._edge_matrix(first.corner_images, opposite)
        assert (
            subdivision._inverse(subdivision._product(physical_edges, reference_inverse))
            == entry[1]
        )


def test_fixed_glue_excludes_overlap_and_preserves_exact_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _cell(80, "tetrahedron", (0, 1, 2, 3), _TETRAHEDRON, _bend())
    neighbour_corners = (_p(1, 1, 1), _p(1, 0, 0), _p(0, 1, 0), _p(0, 0, 1))
    neighbour = _cell(81, "tetrahedron", (4, 1, 2, 3), neighbour_corners, _bend())
    overlap = _cell(82, "tetrahedron", (5, 1, 2, 3), _TETRAHEDRON, _bend())
    shared = (1, 2, 3)
    assert restriction_trace_equal(first, neighbour, shared) is True
    assert subdivision._fixed_glued_monotone(
        first, neighbour, set(shared), StratumSubdivisionCache()
    )
    assert not subdivision._fixed_glued_monotone(
        first, overlap, set(shared), StratumSubdivisionCache()
    )
    expected = subdivision.glued_face_monotone(first, neighbour, shared)
    monkeypatch.setattr(subdivision, "_fixed_glued_monotone", lambda *_: False)
    assert subdivision.glued_face_monotone(first, neighbour, shared) == expected
    assert not subdivision.glued_face_monotone(first, overlap, shared)


def test_subdivision_eviction_recomputes_exact_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    residuals = _stratum_residuals(_FIRST, (4,))
    assert residuals is not None
    root = subdivision._root_piece(_FIRST, (4,), residuals)
    assert root is not None
    monkeypatch.setattr(subdivision, "_CHILD_CACHE_ENTRIES", 1)
    cache = StratumSubdivisionCache()
    key = (_FIRST.global_id, (4,))
    original = subdivision._children(cache, key, root)
    subdivision._children(cache, key, original[0])
    assert len(cache.children) == 1
    recomputed = subdivision._children(cache, key, root)
    for first, second in zip(original, recomputed, strict=True):
        assert first.vertices == second.vertices
        assert first.geometry == second.geometry
        assert first.residual == second.residual
        np.testing.assert_array_equal(first.points, second.points)
    assert len(cache.children) == 1


def test_subdivision_proof_counts_do_not_depend_on_cache_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corners = (_p(1, 0, 1), _p(2, 0, 1), _p(1.5, 0.5, 1), _p(1.5, 0.125, 1.75))
    neighbour = _cell(5, "simplex:3", (4, 9, 10, 11), corners, _bend())
    residuals = _stratum_residuals(_FIRST, (4,))
    assert residuals is not None
    expected = subdivided_stratum_separation(
        _FIRST, neighbour, (4,), residuals, StratumSubdivisionCache()
    )
    monkeypatch.setattr(subdivision, "_CHILD_CACHE_ENTRIES", 1)
    actual = subdivided_stratum_separation(
        _FIRST, neighbour, (4,), residuals, StratumSubdivisionCache()
    )
    assert expected is not None
    assert actual == expected


def test_sheared_prism_is_injective_by_a_searched_preconditioner() -> None:
    # J = I + 3 u e3 e1^T: the centroid inverse leaves sym(M J) indefinite at
    # u = 1, the searched constant preconditioner J(1/2)^-1 does not.
    x, y, z = axes(3)
    cell = _cell(
        6,
        "prism",
        (0, 1, 2, 3, 4, 5),
        _PRISM,
        (x, y, add(z, scale(multiply(x, x), Fraction(3, 2)))),
    )
    witness = prove_mapped_cell_injective(cell)
    assert witness is not None
    assert witness.projection_matrix != tuple(
        tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
    )


# Glued prism-facet controls of an adaptive layer whose Kelley cutting planes
# stall at bound 1.5e-8: each new cut is violated by less than the linear
# program's feasibility tolerance, so the program point never moves.
_STALLED_VALUES = (
    -1.0,
    -0.9599999999999989,
    -0.9599999999999987,
    -0.9599999999999975,
    -0.02000000000000124,
    -0.020000000000001128,
    -1.1830814106161824e-15,
    -3.469446951953614e-18,
    0.0,
    0.02,
    0.020000000000000004,
    0.02000000000000118,
    0.020000000000001183,
    0.9599999999999975,
    1.0,
)
_STALLED_CONTROLS = (
    ((8, 8, 12), (8, 14, 12), (0, 0, 4)),
    ((6, 7, 12), (6, 13, 12), (2, 3, 4)),
    ((8, 8, 9), (8, 14, 9), (0, 0, 10)),
    ((6, 7, 9), (6, 13, 9), (2, 3, 10)),
    ((8, 8, 11), (8, 14, 4), (0, 0, 12)),
    ((6, 7, 11), (6, 13, 4), (2, 3, 12)),
    ((8, 0, 12), (8, 8, 12), (0, 0, 4)),
    ((6, 2, 12), (6, 6, 12), (2, 1, 4)),
    ((8, 0, 4), (8, 8, 12), (0, 0, 5)),
    ((6, 2, 4), (6, 6, 12), (2, 1, 5)),
    ((8, 0, 9), (8, 8, 9), (0, 0, 10)),
    ((6, 2, 9), (6, 6, 9), (2, 1, 10)),
)


def test_preconditioner_search_stops_when_cuts_fall_below_lp_feasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = np.asarray(_STALLED_VALUES, dtype=np.float64)[
        np.asarray(_STALLED_CONTROLS)
    ]
    programs = 0
    linprog = scipy.optimize.linprog

    def counted(
        objective: npt.NDArray[np.float64],
        *,
        A_ub: npt.NDArray[np.float64],
        b_ub: npt.NDArray[np.float64],
        bounds: list[tuple[float | None, float | None]],
        method: Literal["highs"],
        options: dict[str, float],
    ) -> scipy.optimize.OptimizeResult:
        nonlocal programs
        programs += 1
        # The stall runs thousands of programs; a converged search needs few.
        assert programs <= 100, "the cutting-plane search did not stop"
        assert set(options) == {"primal_feasibility_tolerance"}
        return linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method=method,
            options={
                "primal_feasibility_tolerance": options["primal_feasibility_tolerance"]
            },
        )

    monkeypatch.setattr(scipy.optimize, "linprog", counted)
    proposal = _float_search_preconditioner(controls)
    # No positive margin was attained: the search reports no proposal (callers
    # treat it as not found), never a positivity claim.
    assert proposal is None
    assert programs > 1


def _exact_controls(
    *matrices: tuple[tuple[int | Fraction, ...], ...],
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    return tuple(
        tuple(tuple(Fraction(value) for value in row) for row in matrix)
        for matrix in matrices
    )


def _floats(
    controls: tuple[tuple[tuple[Fraction, ...], ...], ...],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [[[float(value) for value in row] for row in matrix] for matrix in controls]
    )


def test_singular_control_hull_refutes_preconditioners_without_programs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # det(I) = det(R) = 1, yet (I + R) / 2 = diag(0, 0, 1) is singular: no M
    # keeps every convex combination nonsingular, so none makes sym(M C) definite.
    controls = _exact_controls(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((1, Fraction(1, 2), 0), (0, 1, 0), (0, 0, 1)),
    )

    def refuse(
        objective: npt.NDArray[np.float64],
        *,
        A_ub: npt.NDArray[np.float64],
        b_ub: npt.NDArray[np.float64],
        bounds: list[tuple[float | None, float | None]],
        method: Literal["highs"],
        options: dict[str, float],
    ) -> Never:
        raise AssertionError("the exact refutation must precede the programs")

    monkeypatch.setattr(scipy.optimize, "linprog", refuse)
    assert _singular_control_hull(controls, _floats(controls))
    assert monotone_preconditioner(controls) is None


def test_singular_control_hull_leaves_feasible_controls_to_the_search() -> None:
    controls = _exact_controls(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, Fraction(1, 2), 0), (0, 1, 0), (0, 0, 1)),
        ((2, 0, 0), (Fraction(-1, 3), 1, 0), (0, 0, Fraction(1, 2))),
    )
    assert not _singular_control_hull(controls, _floats(controls))
    found = monotone_preconditioner(controls)
    assert found is not None
