#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Declared approximation domains transported through bound cavity charts."""

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

import phydrax.geometry._implicit_cavity_atlas as atlas_owner
import phydrax.meshing._adaptation as adaptation_owner
from phydrax._meshcore import TetMeshArrays
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._implicit_cavity_atlas import (
    build_implicit_cavity_atlas,
    CavityAtlasRefusal,
    CavitySnapshot,
    ImplicitCavityAtlas,
    prepare_implicit_cavity_edit,
    Tet,
)
from phydrax.meshing._tetra_metric import CurvedCavityTransition
from tests._support.coordinate_references import coordinate_reference_budget


# Dyadic bipyramid around the origin plus one exterior tetrahedron below a
# bottom face. Vertices: 0 anchor-plane center, 1-3 rim, 4 bottom, 5 top, 6 exterior.
_RIM = ((1.0, 0.0, 0.0), (-0.5, 0.75, 0.0), (-0.5, -0.75, 0.0))
_CONES = tuple(
    row for side in range(3) for row in (
        (0, 5, 1 + side, 1 + (side + 1) % 3),
        (0, 4, 1 + (side + 1) % 3, 1 + side),
    )
)
_EXTERIOR = (6, 4, 1, 2)
_ANCHOR = (Fraction(0), Fraction(0), Fraction(-1, 4))


def _points(bottom: float, top: float, exterior: float = -1.25) -> np.ndarray:
    return np.asarray(
        ((0.0, 0.0, 0.0), *_RIM, (0.0, 0.0, bottom), (0.0, 0.0, top), (0.5, 0.5, exterior))
    )


def _positive(points: np.ndarray, rows: tuple[tuple[int, ...], ...]) -> np.ndarray:
    cells = np.asarray(rows, dtype=np.int32)
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0
    cells[negative] = cells[negative][:, (0, 2, 1, 3)]
    return cells


def _mesh(points: np.ndarray, cells: np.ndarray) -> TetMeshArrays:
    empty = np.empty((0, 3), dtype=np.int32)
    return TetMeshArrays(
        points, cells, np.zeros(cells.shape[0], dtype=np.int32), empty,
        np.empty((0,), dtype=np.int32), np.empty((0, 2), dtype=np.int32),
        np.empty((0,), dtype=np.int32), np.zeros(points.shape[0], dtype=np.int8),
        np.ones(points.shape[0]),
    )


def _cavity(points: np.ndarray, identifiers: tuple[int, ...]) -> CavitySnapshot:
    """The whole bipyramid with its moving top patch and rim."""
    cones = _positive(points, _CONES)
    return CavitySnapshot(
        tuple(
            (Fraction(float(row[0])), Fraction(float(row[1])), Fraction(float(row[2])))
            for row in points[:6]
        ),
        tuple((int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in cones),
        tuple((5, 1 + side, 1 + (side + 1) % 3) for side in range(3)),
        ("top",) * 3,
        identifiers,
        ((1, 2, "rim-ab"), (2, 3, "rim-bc"), (3, 1, "rim-ca")),
    )


def _atlas(
    before: np.ndarray,
    after: np.ndarray,
    chronology: int,
    predecessor: str,
    budget: CoordinateEnclosureBudget | None = None,
) -> ImplicitCavityAtlas:
    if budget is None:
        with coordinate_reference_budget() as owned:
            return _atlas(before, after, chronology, predecessor, owned)
    return build_implicit_cavity_atlas(
        prepare_implicit_cavity_edit(
            _cavity(before, tuple(range(10 * chronology, 10 * chronology + 6))),
            _cavity(after, tuple(range(10 * chronology + 10, 10 * chronology + 16))),
            source_id="dyadic-bipyramid", discovery_id="dyadic-discovery",
            chronology_index=chronology,
            predecessor_id=predecessor or ("" if not chronology else "root"),
            budget=budget,
        ),
        chart_axes=(0, 1), anchor=_ANCHOR, budget=budget,
    )


def _record(
    atlas: ImplicitCavityAtlas,
    before: np.ndarray,
    after: np.ndarray,
    *,
    after_cells: np.ndarray | None = None,
) -> CurvedCavityTransition:
    chronology = atlas.chronology_index
    cells_before = np.vstack((_positive(before, _CONES), _positive(before, (_EXTERIOR,))))
    cells_after = (
        np.vstack((_positive(after, _CONES), _positive(after, (_EXTERIOR,))))
        if after_cells is None else after_cells
    )
    before_ids = np.asarray((*range(10 * chronology, 10 * chronology + 6), 999), dtype=np.int64)
    after_ids = np.asarray(
        (*range(10 * chronology + 10, 10 * chronology + 16), 999), dtype=np.int64
    )
    return CurvedCavityTransition(
        atlas, _mesh(before, cells_before), _mesh(after, cells_after), before_ids, after_ids
    )


def _outward(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    faces = cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape(-1, 3)
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    return faces[counts[inverse.reshape(-1)] == 1]


def _oriented(corners: np.ndarray) -> set[bytes]:
    return {
        min(row.tobytes(), row[(1, 2, 0), :].tobytes(), row[(2, 0, 1), :].tobytes())
        for row in np.ascontiguousarray(corners)
    }


def _source_cells(points: np.ndarray) -> np.ndarray:
    return np.vstack((_positive(points, _CONES), _positive(points, (_EXTERIOR,)))).astype(np.int64)


def _transport(
    source: np.ndarray, history: tuple[CurvedCavityTransition, ...]
) -> PiecewiseLinearDomain:
    cells = _source_cells(source)
    faces = _outward(source, cells)
    declared = PiecewiseLinearDomain(
        source, faces, np.tile((0, -1), (faces.shape[0], 1)), ("material",),
        source_id="dyadic-bipyramid",
    )
    with coordinate_reference_budget() as budget:
        domain = adaptation_owner._transported_curved_domain(
            declared, source, cells, history, budget=budget
        )
    assert domain.region_ids == declared.region_ids
    assert domain.source_id == declared.source_id
    return domain


def test_bound_chart_chain_transports_the_oriented_declared_boundary() -> None:
    source, first, second = _points(-1.0, 1.0), _points(-1.0, 1.5), _points(-1.0, 1.75)
    chart = _atlas(source, first, 0, "")
    successor = _atlas(first, second, 1, chart.chart_id)
    domain = _transport(source, (_record(chart, source, first), _record(successor, first, second)))
    assert np.all(domain.facet_regions == (0, -1))
    expected = _outward(second, _source_cells(second))
    assert _oriented(domain.vertices[domain.facets]) == _oriented(second[expected])


def test_same_corners_with_other_connectivity_refuse() -> None:
    source, moved = _points(-1.0, 1.0), _points(-1.0, 1.5)
    chart = _atlas(source, moved, 0, "")
    # Replace one top cone by a cell through the same corners: the corner set
    # is unchanged, but its oriented volume no longer equals the chart image.
    cells = np.vstack((_positive(moved, _CONES), _positive(moved, (_EXTERIOR,))))
    cells[0] = _positive(moved, ((4, 5, 1, 2),))[0]
    with pytest.raises(ValueError):
        _transport(source, (_record(chart, source, moved, after_cells=cells),))
    # A reversed (negative) cell with the same corners refuses as well.
    reversed_cells = np.vstack((_positive(moved, _CONES), _positive(moved, (_EXTERIOR,))))
    reversed_cells[0] = reversed_cells[0][[0, 2, 1, 3]]
    with pytest.raises(ValueError):
        _transport(source, (_record(chart, source, moved, after_cells=reversed_cells),))


def test_unowned_boundary_and_exterior_changes_refuse() -> None:
    source, first, second = _points(-1.0, 1.0), _points(-1.0, 1.5), _points(-1.0, 1.75)
    chart = _atlas(source, first, 0, "")
    drifted = _points(-1.0, 1.5, exterior=-1.5)
    successor = _atlas(drifted, second, 1, chart.chart_id)
    # The second edit starts from an oriented boundary no chart produced.
    with pytest.raises(ValueError, match="every boundary change to belong to a certified chart"):
        _transport(source, (_record(chart, source, first), _record(successor, drifted, second)))
    with pytest.raises(ValueError, match="every boundary change to belong to a certified chart"):
        _transport(_points(-1.0, 1.0, exterior=-1.5), (_record(chart, source, first),))
    # A chart may not carry an edit of a cell outside its bound cavity.
    with pytest.raises(ValueError, match="outside chart 0 to stay unchanged"):
        _transport(source, (_record(chart, source, _points(-1.0, 1.5, exterior=-1.5)),))
    # An exterior cell reversed with the same corners is not unchanged either.
    moved = _points(-1.0, 1.5)
    cells = np.vstack((_positive(moved, _CONES), _positive(moved, (_EXTERIOR,))))
    cells[-1] = cells[-1][[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="outside chart 0 to stay unchanged"):
        _transport(source, (_record(chart, source, moved, after_cells=cells),))


def _exact_cells(points: np.ndarray) -> tuple[Tet, ...]:
    def exact(vertex: int) -> tuple[Fraction, Fraction, Fraction]:
        x, y, z = (Fraction(float(value)) for value in points[vertex])
        return x, y, z

    return tuple(
        (exact(a), exact(b), exact(c), exact(d))
        for a, b, c, d in _positive(points, _CONES).tolist()
    )


_OLD_IDS, _NEW_IDS = tuple(range(6)), tuple(range(10, 16))


def _bind(
    atlas: ImplicitCavityAtlas,
    old_cells: tuple[Tet, ...],
    new_cells: tuple[Tet, ...],
    budget: CoordinateEnclosureBudget,
) -> int:
    """Bind and return the work it charged."""
    before = budget.work_units
    atlas_owner.bind_implicit_cavity_cells(
        old_cells, _OLD_IDS, new_cells, _NEW_IDS, atlas, budget
    )
    return budget.work_units - before


def test_earned_cavity_binding_fits_where_an_unissued_proof_cannot() -> None:
    source, moved = _points(-1.0, 1.0), _points(-1.0, 1.5)
    old_cells, new_cells = _exact_cells(source), _exact_cells(moved)
    with coordinate_reference_budget() as budget:
        atlas = _atlas(source, moved, 0, "", budget)
        built = budget.work_units
        reused = _bind(atlas, old_cells, new_cells, budget)
        # A distinct but equal atlas object never inherits the earned proof.
        proved = _bind(dataclasses.replace(atlas), old_cells, new_cells, budget)
    assert 0 < reused < proved
    # One request envelope: the build (whose replay note records the request
    # limits) is identical, and only the binding stage allowance varies.
    for allowance, fits in ((reused + proved - 1, False), (reused + proved, True)):
        with coordinate_reference_budget() as budget:
            atlas = _atlas(source, moved, 0, "", budget)
            assert budget.work_units == built
            with budget.bound_stage(allowance, budget.maximum_memory_bytes):
                # The earned binding fits on its own ledger...
                assert _bind(atlas, old_cells, new_cells, budget) == reused
                copy = dataclasses.replace(atlas)
                # ...while the same cells under an unissued atlas need the real proof.
                if fits:
                    assert _bind(copy, old_cells, new_cells, budget) == proved
                else:
                    with pytest.raises(CoordinateEnclosureResourceError):
                        _bind(copy, old_cells, new_cells, budget)
    # Another request ledger pays a full proof as well (with its own cold caches).
    with coordinate_reference_budget() as other:
        assert _bind(copy, old_cells, new_cells, other) >= proved


def test_earned_cavity_binding_refuses_changed_values_and_cells() -> None:
    source, moved = _points(-1.0, 1.0), _points(-1.0, 1.5)
    old_cells, new_cells = _exact_cells(source), _exact_cells(moved)
    flipped = (
        (new_cells[0][0], new_cells[0][2], new_cells[0][1], new_cells[0][3]),
        *new_cells[1:],
    )
    for tampered in ("orientation", "coordinates"):
        with coordinate_reference_budget() as budget:
            atlas = _atlas(source, moved, 0, "", budget)
            with pytest.raises(CavityAtlasRefusal):
                if tampered == "orientation":
                    _bind(atlas, old_cells, flipped, budget)
                else:
                    _bind(atlas, _exact_cells(_points(-1.0, 1.25)), new_cells, budget)
    with coordinate_reference_budget() as budget:
        atlas = _atlas(source, moved, 0, "", budget)
        # Mutate the earned atlas object's public rational in place: the same
        # objects now carry other values, so the earned binding cannot apply.
        apex = next(
            value for tet in atlas.new_tets for point in tet for value in point
            if value == Fraction(3, 2)
        )
        apex._numerator = 7
        with pytest.raises(CavityAtlasRefusal):
            _bind(atlas, old_cells, new_cells, budget)
