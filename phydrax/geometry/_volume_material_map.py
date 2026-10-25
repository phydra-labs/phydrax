# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cavity-local chronological material maps for metric topology edits.

Every committed edit of an adaptation retains its exact removed and created
cells by cell ID, vertex IDs and binary64 vertex coordinates. A curved edit
also retains its paired exact affine cones. Replaying the chain from the source
mesh proves identity outside every cavity by unchanged cell ID *and* exact
geometry, and must reproduce the target mesh exactly. Domain-preserving edits
are the identity material map: both cavities carry one oriented boundary
2-chain, so the new cavity tiles the old one. A curved edit is the certified
cone-wise affine map of its cavity, bound by the coned-boundary proof. Each
edit's exact material pieces come from one shared exact arrangement in its old
physical frame. All contents compose per cavity; no full-domain atlas or
refinement exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Never, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np
from jax import Array

from phydrax.linalg._small_batched import exact_adjugate_action, prepare_exact_adjugate_3

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._host_storage import declared_integer_storage_bytes, declared_object_storage_bytes
from .._meshcore import MeshcoreError, MeshcoreStatus
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
)
from ._exact_polyhedral_geometry import (
    arrange_tetrahedra,
    ArrangementVertices,
    exact_vertices,
    loop_volume_contribution,
    star_determinant,
)
from ._exact_power_refinement import _rne_error
from ._planar_primitives import plane_key, rational_points, reserve_fraction_work


if TYPE_CHECKING:
    from .._meshcore import ExactTetrahedronArrangement
    from ._implicit_cavity_atlas import ImplicitCavityAtlas


Point = tuple[Fraction, Fraction, Fraction]
Tet = tuple[Point, Point, Point, Point]
Face = tuple[Point, ...]
FacetParent = tuple[int, Face, int]
_CellKey = tuple[tuple[int, Point], ...]
_FaceKey = tuple[tuple[int, Point], ...]
type _ExactCells = tuple[
    tuple[int, ...], tuple[tuple[int, int, int, int], ...], tuple[Tet, ...]
]


class VolumeMaterialStatus(StrEnum):
    FOREIGN_ENDPOINT = "foreign_endpoint"
    STALE_IDENTITY = "stale_identity"
    MISSING_ANCESTRY = "missing_ancestry"
    INVALID_CAVITY = "invalid_cavity"
    COVERAGE_GAP = "coverage_gap"
    RESOURCE_LIMIT = "resource_limit"
    UNSUPPORTED_SPACE = "unsupported_space"
    INFEASIBLE_PROJECTION = "infeasible_projection"


class VolumeMaterialMapError(ValueError):
    """A refused material chain, with its typed status."""

    def __init__(self, status: VolumeMaterialStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _refuse(status: VolumeMaterialStatus, message: str) -> Never:
    raise VolumeMaterialMapError(status, message)


def _exact_point(values: Sequence[float]) -> Point:
    return (
        Fraction(float(values[0])),
        Fraction(float(values[1])),
        Fraction(float(values[2])),
    )


def _exact_tet(points: np.ndarray) -> Tet:
    return (
        _exact_point(points[0]),
        _exact_point(points[1]),
        _exact_point(points[2]),
        _exact_point(points[3]),
    )


def _cell_key(vertex_ids: Sequence[int], tet: Tet) -> _CellKey:
    """Orientation-free identity of one cell: its vertex IDs and exact corners."""
    return tuple(sorted(zip((int(value) for value in vertex_ids), tet, strict=True)))


def _exact_cell_key_shell_storage_upper(identity: int, key: _CellKey, /) -> int:
    """Bound a state cell's containers and IDs, whose points are shared by vertex."""
    storage = declared_object_storage_bytes(identity) + declared_object_storage_bytes(key)
    for pair in key:
        storage += declared_object_storage_bytes(pair)
        storage += declared_object_storage_bytes(pair[0])
    return storage


def _active_point_map_value_storage(value: tuple[Point, int], /) -> int:
    """Bound one active-vertex value; the point payload has its own shared owner."""
    return declared_object_storage_bytes(value) + declared_object_storage_bytes(value[1])


def _string_point(point: Sequence[Fraction]) -> tuple[str, ...]:
    return tuple(str(value) for value in point)


class VolumeMaterialEdit(StrictModule, NonTrainableState):
    """One committed cavity edit: exact removed and created cells.

    Rows of ``*_points`` list the four binary64 corners of positive
    tetrahedra, aligned with ``*_vertex_ids``. A curved edit retains its paired
    exact cones ``source_charts[k] -> target_charts[k]`` from ``chart_anchor``
    and each cone's actual outward parent facet ``(cell_id, face, orientation)``;
    a domain-preserving edit has no charts and is the identity material map.
    """

    source_cell_ids: Array
    source_vertex_ids: Array
    source_points: Array
    target_cell_ids: Array
    target_vertex_ids: Array
    target_points: Array
    source_charts: tuple[Tet, ...]
    target_charts: tuple[Tet, ...]
    chart_anchor: Point | None
    source_parents: tuple[FacetParent, ...]
    target_parents: tuple[FacetParent, ...]
    chart_id: str = eqx.field(static=True)
    edit_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_cell_ids: np.ndarray,
        source_vertex_ids: np.ndarray,
        source_points: np.ndarray,
        target_cell_ids: np.ndarray,
        target_vertex_ids: np.ndarray,
        target_points: np.ndarray,
        /,
        *,
        source_charts: tuple[Tet, ...] = (),
        target_charts: tuple[Tet, ...] = (),
        chart_anchor: Point | None = None,
        source_parents: tuple[FacetParent, ...] = (),
        target_parents: tuple[FacetParent, ...] = (),
        chart_id: str = "",
    ) -> None:
        banks = []
        for ids, vertices, points in (
            (source_cell_ids, source_vertex_ids, source_points),
            (target_cell_ids, target_vertex_ids, target_points),
        ):
            ids_ = np.asarray(ids, dtype=np.int64).reshape(-1)
            vertices_ = np.asarray(vertices, dtype=np.int64)
            points_ = np.asarray(points, dtype=np.float64)
            if (
                vertices_.shape != (ids_.size, 4)
                or points_.shape != (ids_.size, 4, 3)
                or not np.all(np.isfinite(points_))
                or np.unique(ids_).size != ids_.size
            ):
                _refuse(
                    VolumeMaterialStatus.INVALID_CAVITY,
                    "volume_material_edit.cells must be unique finite tetrahedra with vertex IDs",
                )
            banks.append((ids_, vertices_, points_))
        curved = bool(source_charts)
        if (
            curved != bool(target_charts)
            or len(source_charts) != len(target_charts)
            or curved != (chart_anchor is not None)
            or curved != bool(chart_id)
            or len(source_parents) != len(source_charts)
            or len(target_parents) != len(target_charts)
            or not banks[0][0].size
            or not banks[1][0].size
        ):
            _refuse(
                VolumeMaterialStatus.MISSING_ANCESTRY,
                "volume_material_edit must remove and create cells, pairing complete cones with their anchor, parents and chart ID",
            )
        (
            (self.source_cell_ids, self.source_vertex_ids, self.source_points),
            (
                self.target_cell_ids,
                self.target_vertex_ids,
                self.target_points,
            ),
        ) = (tuple(jax.device_put(value) for value in bank) for bank in banks)
        self.source_charts = tuple(source_charts)
        self.target_charts = tuple(target_charts)
        self.chart_anchor = chart_anchor
        self.source_parents = tuple(source_parents)
        self.target_parents = tuple(target_parents)
        self.chart_id = chart_id
        self.edit_id = canonical_fingerprint(
            {
                "kind": "volume-material-edit",
                "cells": array_tree_fingerprint(
                    tuple(value for bank in banks for value in bank)
                ),
                "charts": tuple(
                    tuple(tuple(_string_point(point) for point in tet) for tet in charts)
                    for charts in (self.source_charts, self.target_charts)
                ),
                "anchor": None if chart_anchor is None else _string_point(chart_anchor),
                "parents": tuple(
                    tuple(
                        (cell, tuple(_string_point(point) for point in face), sign)
                        for cell, face, sign in parents
                    )
                    for parents in (self.source_parents, self.target_parents)
                ),
                "chart": chart_id,
            }
        )

    @property
    def curved(self) -> bool:
        return bool(self.source_charts)

    def exact_cells(
        self, side: str, /, *, budget: CoordinateEnclosureBudget
    ) -> _ExactCells:
        """Return one side's cell IDs, vertex IDs and exact tetrahedra."""
        ids, vertices, points = (
            (self.source_cell_ids, self.source_vertex_ids, self.source_points)
            if side == "source"
            else (self.target_cell_ids, self.target_vertex_ids, self.target_points)
        )
        count = points.shape[0]
        # Admit device-to-host copies and the identifier/tuple shells before
        # touching numerical leaves. The exact-coordinate owner then admits
        # Fraction storage from the actual binary64 exponent range.
        budget.reserve(
            0,
            3 * 256
            + 8 * (points.size + vertices.size + ids.size)
            + count * (1024 + 5 * declared_integer_storage_bytes(64)),
        )
        bank = np.asarray(points, dtype=np.float64)
        exact = rational_points(bank.reshape((-1, 3)), budget=budget)
        corners = tuple((row[0], row[1], row[2]) for row in exact)
        return (
            tuple(int(value) for value in np.asarray(ids, dtype=np.int64)),
            tuple(
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                for row in np.asarray(vertices, dtype=np.int64)
            ),
            tuple(
                (
                    corners[start],
                    corners[start + 1],
                    corners[start + 2],
                    corners[start + 3],
                )
                for start in range(0, 4 * count, 4)
            ),
        )


def _endpoint_cells(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
) -> tuple[dict[int, _CellKey], int]:
    """Bind endpoint cells and admit their retained exact-map storage bound."""
    from ..discretization._exact_algebraic_geometry import rational_coefficient
    from ..discretization.fem._reference_types import FiniteElementSpec

    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    if any(
        not isinstance(element, FiniteElementSpec)
        or element.cell_kind != "tetrahedron"
        or element.degree != 1
        for element in elements
    ) or any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
        _refuse(
            VolumeMaterialStatus.FOREIGN_ENDPOINT,
            "volume_material_chain.endpoints require affine tetrahedral geometry",
        )
    exact = exact_vertices(mesh, geometry, budget=budget)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result: dict[int, _CellKey] = {}
    # Share valid endpoint points by global vertex ID; preserve disagreements
    # for the chain proof.
    point_by_vertex: dict[int, Point | None] = {}
    preflight_storage = 0
    for block, route in zip(mesh.blocks, routes, strict=True):
        cells = np.asarray(block.vertices)
        budget.reserve(16 * cells.shape[0], 1024 * cells.shape[0])
        preflight_storage += 1024 * cells.shape[0]
        for identity, corners, nodes in zip(
            np.asarray(block.global_ids).tolist(),
            cells,
            np.asarray(route),
            strict=True,
        ):
            identifiers = []
            coordinates = []
            for corner, node in zip(corners, nodes, strict=True):
                vertex = int(vertex_ids[int(corner)])
                point = tuple(
                    Fraction(
                        rational_coefficient(
                            value, role="volume material endpoint coordinate"
                        )
                    )
                    for value in exact[int(node)]
                )
                if len(point) != 3:
                    _refuse(
                        VolumeMaterialStatus.FOREIGN_ENDPOINT,
                        "volume_material_chain.endpoints must be three-dimensional",
                    )
                if vertex in point_by_vertex:
                    known = point_by_vertex[vertex]
                    if known is not None and known == point:
                        point = known
                    elif known is not None:
                        point_by_vertex[vertex] = None
                else:
                    point_by_vertex[vertex] = point
                identifiers.append(vertex)
                coordinates.append(point)
            tet = tuple(coordinates)
            if len(tet) != 4:
                _refuse(
                    VolumeMaterialStatus.FOREIGN_ENDPOINT,
                    "volume_material_chain.endpoints require four tetrahedral corners",
                )
            if int(identity) in result:
                _refuse(
                    VolumeMaterialStatus.FOREIGN_ENDPOINT,
                    "volume_material_chain.endpoint cell IDs must be unique",
                )
            result[int(identity)] = _cell_key(identifiers, tet)
    point_by_vertex.clear()
    del point_by_vertex
    # Reuse the endpoint preflight, topping it up only to the measured
    # escaping map owner bound.
    storage_upper = budget.live_object_storage_upper(
        result, include_retained=True
    )
    if storage_upper > preflight_storage:
        budget.reserve(0, storage_upper - preflight_storage)
    return result, storage_upper


def _cell_order(mesh: CellMesh) -> np.ndarray:
    """Cell IDs in concatenated block order."""
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def _oriented_boundary(
    vertices: Sequence[Sequence[int]],
    tets: Sequence[Tet],
    budget: CoordinateEnclosureBudget,
) -> dict[_FaceKey, int]:
    """Positive cells' oriented boundary 2-chain; interior faces must cancel."""
    incidences: dict[_FaceKey, int] = {}
    for ids, tet in zip(vertices, tets, strict=True):
        if star_determinant(tet, budget=budget) <= 0:
            _refuse(
                VolumeMaterialStatus.INVALID_CAVITY,
                "volume_material_edit.cells must be positive tetrahedra",
            )
        for local in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
            budget.reserve(4, 512)
            corners = tuple((int(ids[index]), tet[index]) for index in local)
            order = sorted(range(3), key=lambda position: corners[position])
            inversions = sum(
                order[first] > order[second] for first, second in ((0, 1), (0, 2), (1, 2))
            )
            key = tuple(corners[position] for position in order)
            incidences[key] = incidences.get(key, 0) + (-1 if inversions % 2 else 1)
    boundary = {key: value for key, value in incidences.items() if value}
    if any(abs(value) != 1 for value in boundary.values()):
        _refuse(
            VolumeMaterialStatus.INVALID_CAVITY,
            "volume_material_edit.cavity facets must be manifold",
        )
    return boundary


def _certify_edit(
    edit: VolumeMaterialEdit,
    atlas: ImplicitCavityAtlas | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[_ExactCells, _ExactCells]:
    """Prove one edit's cavities and return its exact source and target cells.

    Returned Fraction coordinates may escape into chain state; the caller owns
    the temporary scope through reconciliation and promotes surviving keys.
    """
    from ._implicit_cavity_atlas import (
        _cover_coned_boundary,
        bind_implicit_cavity_cells,
        CavityAtlasRefusal,
        CavityFacetParent,
    )

    source = edit.exact_cells("source", budget=budget)
    target = edit.exact_cells("target", budget=budget)
    source_ids, source_vertices, source_tets = source
    target_ids, target_vertices, target_tets = target
    if not edit.curved:
        # Positive chains with one boundary 2-chain have equal multiplicity; the
        # source cavity is embedded, so the target cavity tiles the same region.
        if _oriented_boundary(source_vertices, source_tets, budget) != _oriented_boundary(
            target_vertices, target_tets, budget
        ):
            _refuse(
                VolumeMaterialStatus.COVERAGE_GAP,
                "volume_material_edit.domain-preserving cavities must share their oriented boundary",
            )
        return source, target
    anchor = edit.chart_anchor
    if anchor is None:
        _refuse(
            VolumeMaterialStatus.MISSING_ANCESTRY,
            "volume_material_edit.charts require their anchor",
        )
    try:
        if atlas is not None:
            # The committed atlas must be exactly the edit's retained cones;
            # its binding to these actual cells is earned or reproved here.
            budget.reserve(24 * (len(edit.source_charts) + len(edit.target_charts)))
            if (
                atlas.chart_id != edit.chart_id
                or atlas.old_tets != edit.source_charts
                or atlas.new_tets != edit.target_charts
                or atlas.anchor != anchor
                or tuple(
                    (parent.cell_id, parent.vertices, parent.orientation)
                    for parent in atlas.source_boundary_parents
                )
                != edit.source_parents
                or tuple(
                    (parent.cell_id, parent.vertices, parent.orientation)
                    for parent in atlas.target_boundary_parents
                )
                != edit.target_parents
            ):
                _refuse(
                    VolumeMaterialStatus.MISSING_ANCESTRY,
                    "volume_material_edit must retain its committed cavity atlas",
                )
            bind_implicit_cavity_cells(
                source_tets, source_ids, target_tets, target_ids, atlas, budget
            )
            return source, target
        for tets, ids, charts, parents in (
            (source_tets, source_ids, edit.source_charts, edit.source_parents),
            (target_tets, target_ids, edit.target_charts, edit.target_parents),
        ):
            _cover_coned_boundary(
                tets,
                ids,
                charts,
                tuple(
                    CavityFacetParent(cell, face, sign) for cell, face, sign in parents
                ),
                anchor,
                budget,
            )
    except CavityAtlasRefusal as error:
        raise VolumeMaterialMapError(
            VolumeMaterialStatus.COVERAGE_GAP, str(error)
        ) from error
    return source, target


class VolumeMaterialChain(StrictModule, NonTrainableState):
    """Chronological cavity-local material maps bound from source to target.

    ``maximum_work_units``/``maximum_scratch_bytes`` are the publishing
    request's limits; cold reconstruction reproves every edit under them and
    rebuilds every derived bank. Edit ``e`` owns density weights
    ``weight_offsets[e]:weight_offsets[e + 1]`` (see
    `volume_material_edit_weights`), pieces
    ``piece_offsets[e]:piece_offsets[e + 1]`` with routes
    ``(target, source, chart)``, cone maps
    ``chart_offsets[e]:chart_offsets[e + 1]`` and rounded shared arrangement
    vertices ``piece_vertex_offsets[e]:piece_vertex_offsets[e + 1]``; piece
    ``p``'s simplices are the rows
    ``piece_simplex_offsets[p]:piece_simplex_offsets[p + 1]`` of
    ``piece_simplex_vertices``, edit-local indices into those vertices.
    """

    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    edits: tuple[VolumeMaterialEdit, ...]
    original_source_id: str = eqx.field(static=True)
    discovery_id: str = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
    weight_offsets: Array
    weight_targets: Array
    weight_sources: Array
    weight_mantissas: Array
    weight_exponents: Array
    weight_mantissa_errors: Array
    target_measures: Array
    target_measure_errors: Array
    piece_offsets: Array
    piece_routes: Array
    piece_simplex_offsets: Array
    piece_simplex_vertices: Array
    piece_vertex_offsets: Array
    piece_vertices: Array
    chart_offsets: Array
    chart_linear: Array
    chart_translation: Array
    chart_jacobians: Array
    chain_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        edits: tuple[VolumeMaterialEdit, ...],
        /,
        *,
        original_source_id: str,
        discovery_id: str,
        maximum_work_units: int,
        maximum_scratch_bytes: int,
        budget: CoordinateEnclosureBudget,
        atlases: Mapping[str, ImplicitCavityAtlas] | None = None,
    ) -> None:
        """Certify the chain; committed in-request atlases bind their cones.

        Without ``atlases`` (stored reconstruction) every curved edit reproves
        its coned cover from the retained exact inputs.

        Exact endpoint ownership, shared-point reference counts, and per-edit
        credit promotion form one transaction. Keep their helpers local so
        intermediate maps cannot escape their shared admission lifetime.
        """
        if not original_source_id or not discovery_id:
            _refuse(
                VolumeMaterialStatus.MISSING_ANCESTRY,
                "volume_material_chain must retain the original source and discovery",
            )
        if not edits or not any(edit.curved for edit in edits):
            _refuse(
                VolumeMaterialStatus.MISSING_ANCESTRY,
                "volume_material_chain requires at least one curved edit",
            )
        if maximum_work_units < 1 or maximum_scratch_bytes < 1:
            _refuse(
                VolumeMaterialStatus.RESOURCE_LIMIT,
                "volume_material_chain must retain positive reproof limits",
            )
        with budget.live_storage() as state_storage:
            with budget.temporary_scope() as endpoint_storage:
                state, _ = _endpoint_cells(source_mesh, source_geometry, budget)
                target, target_storage = _endpoint_cells(target_mesh, target_geometry, budget)
                target_points: dict[int, Point | None] = {}
                for key in target.values():
                    for vertex, point in key:
                        if vertex not in target_points:
                            target_points[vertex] = point
                        else:
                            known = target_points[vertex]
                            if known is not None and known != point:
                                target_points[vertex] = None
                # State cell keys share one exact point per active vertex ID.
                # The owner counts preserve that point until its last cell retires.
                active: dict[int, tuple[Point, int]] = {}
                point_owners: dict[int, tuple[Point, int, int]] = {}
                point_owner_records_storage = 0
                point_storage_upper = 0
                active_value_storage = 0
                state_key_storage = 0

                def point_owner_record_storage(
                    record: tuple[Point, int, int],
                ) -> int:
                    return (
                        declared_integer_storage_bytes(64)
                        + declared_object_storage_bytes(record)
                        + declared_object_storage_bytes(record[1])
                        + declared_object_storage_bytes(record[2])
                    )

                def attach(key: _CellKey) -> _CellKey:
                    nonlocal point_owner_records_storage
                    nonlocal point_storage_upper
                    nonlocal active_value_storage
                    normalized = []
                    for vertex, point in key:
                        current = active.get(vertex)
                        if current is not None and current[0] != point:
                            _refuse(
                                VolumeMaterialStatus.STALE_IDENTITY,
                                "volume_material_chain vertex IDs must name one active exact point",
                            )
                        endpoint_point = target_points.get(vertex)
                        retained_point = (
                            endpoint_point
                            if current is None
                            and endpoint_point is not None
                            and endpoint_point == point
                            else point if current is None else current[0]
                        )
                        if current is None:
                            owner_id = id(retained_point)
                            owner = point_owners.get(owner_id)
                            if owner is None:
                                # The endpoint map already owns this exact
                                # object; active references must not charge it twice.
                                size = (
                                    0 if retained_point is endpoint_point
                                    else budget.live_object_storage_upper(
                                        retained_point, include_retained=True
                                    )
                                )
                                record = (retained_point, 1, size)
                                point_owners[owner_id] = record
                                point_owner_records_storage += point_owner_record_storage(record)
                                point_storage_upper += size
                            else:
                                if owner[0] is not retained_point:
                                    raise RuntimeError(
                                        "volume_material_chain active point owner identity was reused"
                                    )
                                point_owner_records_storage -= point_owner_record_storage(owner)
                                record = (retained_point, owner[1] + 1, owner[2])
                                point_owners[owner_id] = record
                                point_owner_records_storage += point_owner_record_storage(record)
                            value = (retained_point, 1)
                        else:
                            active_value_storage -= _active_point_map_value_storage(current)
                            value = (retained_point, current[1] + 1)
                        active_value_storage += _active_point_map_value_storage(
                            (retained_point, value[1])
                        )
                        active[vertex] = value
                        normalized.append((vertex, retained_point))
                    return tuple(normalized)

                def detach(key: _CellKey) -> None:
                    nonlocal point_owner_records_storage
                    nonlocal point_storage_upper
                    nonlocal active_value_storage
                    for vertex, _ in key:
                        current = active[vertex]
                        active_value_storage -= _active_point_map_value_storage(current)
                        if current[1] == 1:
                            del active[vertex]
                            owner_id = id(current[0])
                            owner = point_owners[owner_id]
                            if owner[1] == 1:
                                del point_owners[owner_id]
                                point_owner_records_storage -= point_owner_record_storage(owner)
                                point_storage_upper -= owner[2]
                            else:
                                point_owner_records_storage -= point_owner_record_storage(owner)
                                record = (owner[0], owner[1] - 1, owner[2])
                                point_owners[owner_id] = record
                                point_owner_records_storage += point_owner_record_storage(record)
                        else:
                            value = (current[0], current[1] - 1)
                            active[vertex] = value
                            active_value_storage += _active_point_map_value_storage(value)

                def live_state_upper() -> int:
                    return (
                        target_storage
                        + state_key_storage
                        + point_storage_upper
                        + point_owner_records_storage
                        + declared_object_storage_bytes(state)
                        + declared_object_storage_bytes(active)
                        + active_value_storage
                        + declared_object_storage_bytes(point_owners)
                        + declared_object_storage_bytes(target_points)
                    )

                # The owner preflight and exact-key shells are charged once;
                # escaped point tuples are separately admitted by the live owner.
                budget.reserve(4 * len(state), 4 * 256 * len(state))
                for identity in tuple(state):
                    key = attach(state[identity])
                    state[identity] = key
                    state_key_storage += _exact_cell_key_shell_storage_upper(identity, key)
                state_storage.promote_temporary_storage(
                    endpoint_storage, live_state_upper()
                )
            if state:
                del identity, key

            def certify_and_apply(edit: VolumeMaterialEdit) -> None:
                nonlocal state_key_storage
                with budget.temporary_scope() as edit_storage:
                    source_cells, target_cells = _certify_edit(
                        edit,
                        None if atlases is None else atlases.get(edit.chart_id),
                        budget,
                    )
                    for side, (ids, vertices, tets) in (
                        ("source", source_cells), ("target", target_cells)
                    ):
                        budget.reserve(
                            4 * len(ids), 4 * 256 * len(ids) if side == "target" else 0
                        )
                        for identity, corners, tet in zip(ids, vertices, tets, strict=True):
                            key = _cell_key(corners, tet)
                            if side == "source":
                                if identity not in state:
                                    _refuse(
                                        VolumeMaterialStatus.MISSING_ANCESTRY,
                                        "volume_material_edit removes a cell absent from the chain state",
                                    )
                                stored = state.pop(identity)
                                if stored != key:
                                    _refuse(
                                        VolumeMaterialStatus.STALE_IDENTITY,
                                        "volume_material_edit cell ID retains changed geometry",
                                    )
                                state_key_storage -= _exact_cell_key_shell_storage_upper(
                                    identity, stored
                                )
                                detach(stored)
                            else:
                                if identity in state:
                                    _refuse(
                                        VolumeMaterialStatus.STALE_IDENTITY,
                                        "volume_material_edit creates an existing cell ID",
                                    )
                                key = attach(key)
                                state[identity] = key
                                state_key_storage += _exact_cell_key_shell_storage_upper(
                                    identity, key
                                )
                    total = live_state_upper()
                    if total >= state_storage.bound:
                        state_storage.promote_temporary_storage(edit_storage, total)
                    else:
                        state_storage.set_bound(total)

            for edit in edits:
                if edit.curved and atlases is not None and edit.chart_id not in atlases:
                    _refuse(
                        VolumeMaterialStatus.MISSING_ANCESTRY,
                        "volume_material_chain must bind every committed curved edit to its atlas",
                    )
                certify_and_apply(edit)
            budget.reserve(len(state))
            if state != target:
                _refuse(
                    VolumeMaterialStatus.STALE_IDENTITY
                    if state.keys() == target.keys()
                    else VolumeMaterialStatus.FOREIGN_ENDPOINT,
                    "volume_material_chain must reproduce the target cells by ID and exact geometry",
                )
            state.clear()
            active.clear()
            point_owners.clear()
            target_points.clear()
            point_owner_records_storage = 0
            point_storage_upper = 0
            active_value_storage = 0
            state_key_storage = 0
            state_storage.set_bound(live_state_upper())
            bank = _prepare_material_bank(target, _cell_order(target_mesh), edits, budget)
            target.clear()
            del target
            state_storage.set_bound(
                declared_object_storage_bytes(state)
                + declared_object_storage_bytes(active)
                + declared_object_storage_bytes(point_owners)
                + declared_object_storage_bytes(target_points)
            )
            del certify_and_apply, live_state_upper, attach, detach
            del point_owner_record_storage, state, active, point_owners, target_points
            state_storage.set_bound(0)
        published = {name: jax.device_put(getattr(bank, name)) for name in _MaterialBank.__slots__}
        # Device storage narrower than binary64 (JAX x64 off) rounds the
        # weight mantissas and target measures; their published bounds absorb
        # that exact rounding, stored rounded upward.
        for values, bounds in (
            ("weight_mantissas", "weight_mantissa_errors"),
            ("target_measures", "target_measure_errors"),
        ):
            exact = getattr(bank, values)
            stored = np.asarray(published[values], dtype=np.float64)
            if not np.array_equal(stored, exact):
                widened = np.nextafter(
                    getattr(bank, bounds) + np.nextafter(np.abs(stored - exact), np.inf),
                    np.inf,
                )
                narrow = widened.astype(published[bounds].dtype)
                narrow = np.where(
                    narrow.astype(np.float64) < widened,
                    np.nextafter(narrow, np.asarray(np.inf, dtype=narrow.dtype)),
                    narrow,
                )
                published[bounds] = jax.device_put(narrow)
        for name, value in published.items():
            setattr(self, name, value)
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.source_geometry = source_geometry
        self.target_geometry = target_geometry
        self.edits = edits
        self.original_source_id = original_source_id
        self.discovery_id = discovery_id
        self.maximum_work_units = int(maximum_work_units)
        self.maximum_scratch_bytes = int(maximum_scratch_bytes)
        self.chain_id = canonical_fingerprint(
            {
                "kind": "volume-material-chain",
                "source": (source_mesh.mesh_id, array_tree_fingerprint(source_geometry)),
                "target": (target_mesh.mesh_id, array_tree_fingerprint(target_geometry)),
                "edits": tuple(edit.edit_id for edit in edits),
                "original_source": original_source_id,
                "discovery": discovery_id,
                "limits": (self.maximum_work_units, self.maximum_scratch_bytes),
            }
        )

    @classmethod
    def from_stored_fields(
        cls,
        value: VolumeMaterialChain,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> VolumeMaterialChain:
        """Reprove every stored edit, including each curved coned cover."""
        rebuilt = cls(
            value.source_mesh,
            value.source_geometry,
            value.target_mesh,
            value.target_geometry,
            tuple(
                VolumeMaterialEdit(
                    np.asarray(edit.source_cell_ids),
                    np.asarray(edit.source_vertex_ids),
                    np.asarray(edit.source_points),
                    np.asarray(edit.target_cell_ids),
                    np.asarray(edit.target_vertex_ids),
                    np.asarray(edit.target_points),
                    source_charts=edit.source_charts,
                    target_charts=edit.target_charts,
                    chart_anchor=edit.chart_anchor,
                    source_parents=edit.source_parents,
                    target_parents=edit.target_parents,
                    chart_id=edit.chart_id,
                )
                for edit in value.edits
            ),
            original_source_id=value.original_source_id,
            discovery_id=value.discovery_id,
            maximum_work_units=value.maximum_work_units,
            maximum_scratch_bytes=value.maximum_scratch_bytes,
            budget=budget,
        )
        if rebuilt.chain_id != value.chain_id:
            _refuse(
                VolumeMaterialStatus.STALE_IDENTITY,
                "volume_material_chain.identity must bind its stored edits",
            )
        return rebuilt

    def reconstruction_preparation_bounds(self) -> tuple[int, int, int]:
        """Cold reproof runs under the publishing request's own limits."""
        return self.maximum_work_units, self.maximum_scratch_bytes, 0

    def require_bound(
        self,
        source: CellMesh,
        source_geometry: CellGeometrySpec,
        target: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None:
        """Bind actual endpoint meshes and geometry content to this chain."""
        if (
            source.mesh_id != self.source_mesh.mesh_id
            or target.mesh_id != self.target_mesh.mesh_id
            or not tree_content_equal(source_geometry, self.source_geometry)
            or not tree_content_equal(target_geometry, self.target_geometry)
        ):
            _refuse(
                VolumeMaterialStatus.FOREIGN_ENDPOINT,
                "volume_material_chain must bind actual endpoint meshes and geometry",
            )


def _tet_volume(tet: Tet, budget: CoordinateEnclosureBudget) -> Fraction:
    determinant = star_determinant(tet, budget=budget)
    if determinant <= 0:
        _refuse(
            VolumeMaterialStatus.INVALID_CAVITY,
            "volume_material_edit cells must be positive tetrahedra",
        )
    reserve_fraction_work(((determinant,),), 1, 1, 1, budget=budget)
    return determinant / 6


@dataclass(frozen=True, slots=True)
class VolumeMaterialChartMap:
    """Exact affine cone map ``x = linear @ y + offset`` and its Jacobian."""

    linear: tuple[Point, Point, Point]
    offset: Point
    jacobian: Fraction


@dataclass(frozen=True, slots=True)
class VolumeMaterialEditPieces:
    """Exact material pieces of one edit on its shared exact arrangement.

    Piece ``p`` is ``s ∩ C_k ∩ A_k^{-1}(t)`` (``s ∩ t`` for the identity map)
    in the edit's old physical frame, with explicit route
    ``routes[p] = (target, source, chart)``, ``chart = -1`` for the identity
    map, and exact old-frame volume ``volumes[p]``. Rows
    ``simplex_offsets[p]:simplex_offsets[p + 1]`` of ``simplices`` are its
    positive pulling tetrahedra over edit-local vertex rows: exact homogeneous
    ``exact_vertices`` ``(X, Y, Z, W)`` with ``W > 0`` and their binary64
    roundings ``vertices``, shared by every piece of the edit. The
    triangulation and vertex order are internal, never scientific identity.
    """

    routes: np.ndarray
    volumes: tuple[Fraction, ...]
    simplex_offsets: np.ndarray
    simplices: np.ndarray
    vertices: np.ndarray
    exact_vertices: tuple[tuple[int, int, int, int], ...]
    charts: tuple[VolumeMaterialChartMap, ...]
    source_volumes: tuple[Fraction, ...]
    target_volumes: tuple[Fraction, ...]


def _edges(tet: Tet) -> tuple[Point, Point, Point]:
    base = tet[0]
    first, second, third = (
        (point[0] - base[0], point[1] - base[1], point[2] - base[2]) for point in tet[1:]
    )
    return first, second, third


def _cone_map(
    origin: Tet,
    image: Tet,
    budget: CoordinateEnclosureBudget,
) -> VolumeMaterialChartMap:
    """The exact affine map of one cone onto its paired image cone."""
    reserve_fraction_work(origin + image, 24, 24, 2, budget=budget)
    adjugate = prepare_exact_adjugate_3(_edges(origin), budget=budget)
    if adjugate is None or adjugate.determinant <= 0:
        _refuse(
            VolumeMaterialStatus.INVALID_CAVITY,
            "volume_material_edit cones must be positive",
        )
    image_edges = _edges(image)
    # Column j of origin^{-1}: the action on the unit vector e_j.
    columns = tuple(
        exact_adjugate_action(
            adjugate,
            tuple(Fraction(int(axis == row)) for axis in range(3)),
            budget=budget,
        )
        for row in range(3)
    )
    reserve_fraction_work(tuple(columns) + image_edges, 54, 9, 4, budget=budget)
    linear_rows = []
    for a in range(3):
        row = []
        for b in range(3):
            row.append(
                sum((image_edges[m][a] * columns[b][m] for m in range(3)), Fraction(0))
            )
        linear_rows.append((row[0], row[1], row[2]))
    linear = (linear_rows[0], linear_rows[1], linear_rows[2])
    offset = tuple(
        image[0][a] - sum((linear[a][b] * origin[0][b] for b in range(3)), Fraction(0))
        for a in range(3)
    )
    image_adjugate = prepare_exact_adjugate_3(image_edges, budget=budget)
    if image_adjugate is None or image_adjugate.determinant <= 0:
        _refuse(
            VolumeMaterialStatus.INVALID_CAVITY,
            "volume_material_edit cones must be positive",
        )
    return VolumeMaterialChartMap(
        linear,
        (offset[0], offset[1], offset[2]),
        image_adjugate.determinant / adjugate.determinant,
    )

type Plane = tuple[Fraction, Fraction, Fraction, Fraction]
type _PlaneTerms = tuple[tuple[int, Fraction], ...]


def _pulled(
    plane: Plane, chart: VolumeMaterialChartMap, budget: CoordinateEnclosureBudget
) -> Plane:
    """The plane ``n @ (L y + t) + d <= 0`` in source coordinates ``y``."""
    reserve_fraction_work((plane, *chart.linear, chart.offset), 24, 4, 4, budget=budget)
    normal = tuple(
        sum((chart.linear[a][b] * plane[a] for a in range(3)), Fraction(0))
        for b in range(3)
    )
    offset = sum((plane[a] * chart.offset[a] for a in range(3)), plane[3])
    return normal[0], normal[1], normal[2], offset


# One new index, face, side value or side row: its dictionary slot and key.
_TABLE_SLOT_BYTES = 256
# Three term pairs, two tuples, first-list growth, and the construction iterator.
_FACE_TERMS_BYTES = 512


class _Cells:
    """Exact tetrahedra of one edit with their shared points, faces and bounds.

    Cavity cells, cones and images share corners and faces. Each distinct
    exact point is hashed once and afterwards named by its index. Each
    distinct face has one canonical plane (``plane_key`` normalizes the
    normal's first nonzero component, so it does not depend on corner order
    or owner), used outward by every owning tetrahedron and negated when the
    opposite corner lies on its positive side, exactly as
    ``tetrahedron_planes`` orients it. A face side at a point is evaluated
    once and shared by every owner; at the face's own corners it is exactly
    zero. Rows are assembled once per owner and point, then index lookups.
    """

    def __init__(self, budget: CoordinateEnclosureBudget) -> None:
        self.budget = budget
        self.points: list[Point] = []
        self.point_index: dict[Point, int] = {}
        self.tets: list[Tet] = []
        self.corners: list[tuple[int, int, int, int]] = []
        self.bounds: list[tuple[Point, Point]] = []
        self.cell_faces: list[tuple[tuple[int, ...], tuple[bool, ...]] | None] = []
        self.face_index: dict[tuple[int, int, int], int] = {}
        self.plane_index: dict[Plane, int] = {}
        self.face_planes: list[Plane] = []
        self.face_terms: list[tuple[_PlaneTerms, int]] = []
        self.values: dict[tuple[int, int], Fraction] = {}
        self.rows_cache: dict[tuple[int, int], tuple[Fraction, ...]] = {}

    def point(self, point: Point) -> int:
        """Index of one exact point; charges hashing its three coordinates."""
        self.budget.reserve(3)
        index = self.point_index.get(point)
        if index is None:
            self.budget.reserve(0, _TABLE_SLOT_BYTES)
            index = len(self.points)
            self.point_index[point] = index
            self.points.append(point)
        return index

    def add(self, tet: Tet) -> int:
        """Register one tetrahedron: its corner indices and exact axis bounds."""
        corners = (
            self.point(tet[0]),
            self.point(tet[1]),
            self.point(tet[2]),
            self.point(tet[3]),
        )
        # Three minima and three maxima of four values per axis.
        self.budget.reserve(18, _TABLE_SLOT_BYTES)
        lower = tuple(min(point[axis] for point in tet) for axis in range(3))
        upper = tuple(max(point[axis] for point in tet) for axis in range(3))
        self.tets.append(tet)
        self.corners.append(corners)
        self.bounds.append(
            ((lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2]))
        )
        self.cell_faces.append(None)
        return len(self.tets) - 1

    def disjoint(self, first: int, second: int) -> bool:
        """Whether axis bounds exclude a positive-volume intersection."""
        (first_lower, first_upper), (second_lower, second_upper) = (
            self.bounds[first],
            self.bounds[second],
        )
        for axis in range(3):
            self.budget.reserve(3)
            if max(first_lower[axis], second_lower[axis]) >= min(
                first_upper[axis], second_upper[axis]
            ):
                return True
        return False

    def _evaluate(self, face: int, points: Sequence[int]) -> None:
        """Evaluate one face plane at every point not yet decided, in one batch."""
        missing = []
        for point in points:
            self.budget.reserve(1)
            if (face, point) not in self.values and point not in missing:
                missing.append(point)
        if not missing:
            return
        plane = self.face_planes[face]
        terms, operations = self.face_terms[face]
        if not terms:
            raise RuntimeError("A prepared face plane must have a nonzero normal.")
        exact = tuple(self.points[point] for point in missing)
        # Remaining visits still bound every additive term; four operand
        # occurrences retain the existing conservative rational bit-growth rule.
        reserve_fraction_work(
            (plane, *exact), operations * len(missing), len(missing), 4,
            budget=self.budget,
        )
        self.budget.reserve(0, _TABLE_SLOT_BYTES * len(missing))
        for point, coordinates in zip(missing, exact, strict=True):
            axis, coefficient = terms[0]
            value = (
                coordinates[axis]
                if coefficient == 1
                else coefficient * coordinates[axis]
            )
            term_index = 1
            while term_index < len(terms):
                axis, coefficient = terms[term_index]
                term = (
                    coordinates[axis]
                    if coefficient == 1
                    else coefficient * coordinates[axis]
                )
                value += term
                term_index += 1
            if plane[3] != 0:
                value += plane[3]
            self.values[face, point] = value

    def faces(self, cell: int) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        """Face indices opposite each corner and whether each is used negated."""
        prepared = self.cell_faces[cell]
        if prepared is not None:
            return prepared
        tet, corners = self.tets[cell], self.corners[cell]
        faces, negated = [], []
        for opposite in range(4):
            key = tuple(sorted(corners[index] for index in range(4) if index != opposite))
            self.budget.reserve(3)
            face = self.face_index.get((key[0], key[1], key[2]))
            if face is None:
                prepared_key = plane_key(
                    tuple(tet[index] for index in range(4) if index != opposite),
                    budget=self.budget,
                )
                if prepared_key is None:
                    raise ValueError(
                        "tetrahedron.face must have rank two; got deficient rank"
                    )
                raw_plane = prepared_key[0]
                plane: Plane = (raw_plane[0], raw_plane[1], raw_plane[2], raw_plane[3])
                self.budget.reserve(4, 3 * _TABLE_SLOT_BYTES)
                face = self.plane_index.get(plane)
                if face is None:
                    # Admit the plane slot and sparse affine metadata before
                    # building or retaining either representation.
                    self.budget.reserve(7, _TABLE_SLOT_BYTES + _FACE_TERMS_BYTES)
                    terms: _PlaneTerms = tuple(
                        (axis, plane[axis])
                        for axis in range(3)
                        if plane[axis] != 0
                    )
                    operations = (
                        1 + sum(coefficient != 1 for _, coefficient in terms)
                        + len(terms) - 1 + (plane[3] != 0)
                    )
                    face = len(self.face_planes)
                    self.plane_index[plane] = face
                    self.face_planes.append(plane)
                    self.face_terms.append((terms, operations))
                self.face_index[key[0], key[1], key[2]] = face
                # A canonical plane through three exact corners vanishes there.
                for point in key:
                    self.values[face, point] = Fraction(0)
            self._evaluate(face, (corners[opposite],))
            side = self.values[face, corners[opposite]]
            if side == 0:
                raise ValueError(
                    "tetrahedron.vertices must have positive volume rank; got zero"
                )
            faces.append(face)
            negated.append(side > 0)
        prepared = (tuple(faces), tuple(negated))
        self.cell_faces[cell] = prepared
        return prepared

    def rows(self, cell: int, points: Sequence[int]) -> tuple[tuple[Fraction, ...], ...]:
        """Outward side rows of one registered tetrahedron at indexed points."""
        faces, negated = self.faces(cell)
        missing = []
        for point in points:
            self.budget.reserve(1)
            if (cell, point) not in self.rows_cache and point not in missing:
                missing.append(point)
        if missing:
            for face in faces:
                self._evaluate(face, missing)
            # One lookup and one sign per face of every new row.
            self.budget.reserve(8 * len(missing), _TABLE_SLOT_BYTES * len(missing))
            for point in missing:
                self.rows_cache[cell, point] = tuple(
                    -self.values[face, point] if flip else self.values[face, point]
                    for face, flip in zip(faces, negated, strict=True)
                )
        return tuple(self.rows_cache[cell, point] for point in points)


def _cutting(
    rows: Sequence[tuple[Fraction, ...]],
    budget: CoordinateEnclosureBudget,
) -> tuple[int, ...] | None:
    """Columns of ``rows`` cutting the hull of their points; ``None`` when one separates.

    A plane nonnegative at every point meets the hull in at most a face, so
    the intersection with its halfspace has zero volume. A plane nonpositive
    at every point contains the hull and cannot cut it. Each column's entries
    are charged before it is examined; a separating column ends the scan.
    """
    cutting = []
    for column in range(len(rows[0])):
        budget.reserve(len(rows))
        if all(row[column] >= 0 for row in rows):
            return None
        if any(row[column] > 0 for row in rows):
            cutting.append(column)
    return tuple(cutting)


def _decide(
    cells: _Cells, first: int, second: int, budget: CoordinateEnclosureBudget
) -> tuple[int, ...] | None:
    """Columns of ``first``'s planes cutting ``second``; ``None`` when either separates.

    Exact sides at the other cell's corners decide both directions. ``None``
    when one plane of either cell has the other on its outer side, so the pair
    meets in at most a face; an empty tuple means ``first`` contains ``second``.
    """
    cut = _cutting(cells.rows(first, cells.corners[second]), budget)
    if cut is None or _cutting(cells.rows(second, cells.corners[first]), budget) is None:
        return None
    return cut


# A job inherits its initial tetrahedron's four faces and every ancestral and
# own cut; material planning adds at most four cell or cone cuts and, for a
# curved edit, at most four pulled-target cuts.
_MAXIMUM_HALFSPACES = 12


class _Arrangement:
    """One edit's deterministic exact clipping DAG in its old physical frame.

    Initial tetrahedra are registered old-frame cells (sources, cones and
    identity-edit targets) with their exact corners and signed outward face
    planes. A job clips an initial tetrahedron (parent ``-(s + 1)``) or an
    earlier job's immutable convex result by signed plane references. Planes
    are interned by their canonical equation, unit first nonzero normal
    component, so a cell face and a pulled-back target face on one geometric
    plane share one numeric equation; routes and plane provenance stay with
    the caller. Each new table entry admits one dictionary slot.
    """

    def __init__(self, cells: _Cells) -> None:
        self.cells = cells
        self.budget = cells.budget
        self.points: list[Point] = []
        self.point_index: dict[int, int] = {}
        self.planes: list[Plane] = []
        self.plane_index: dict[Plane, int] = {}
        self.face_index: dict[int, int] = {}
        self.pulled_index: dict[tuple[int, int], int] = {}
        self.tetrahedra: list[tuple[int, int, int, int]] = []
        self.tetrahedron_planes: list[tuple[int, int, int, int]] = []
        self.initial: dict[int, int] = {}
        self.parents: list[int] = []
        self.depths: list[int] = []
        self.cut_offsets: list[int] = [0]
        self.cut_planes: list[int] = []

    def plane(self, plane: Plane) -> int:
        """1-based index of one canonical plane equation; charges hashing it."""
        self.budget.reserve(4)
        index = self.plane_index.get(plane)
        if index is None:
            self.budget.reserve(0, _TABLE_SLOT_BYTES)
            self.planes.append(plane)
            index = len(self.planes)
            self.plane_index[plane] = index
        return index

    def cuts(self, cell: int, columns: tuple[int, ...]) -> tuple[int, ...]:
        """Signed references of ``cell``'s outward faces in ``columns``."""
        faces, negated = self.cells.faces(cell)
        result = []
        for column in columns:
            face = faces[column]
            self.budget.reserve(1)
            index = self.face_index.get(face)
            if index is None:
                index = self.plane(self.cells.face_planes[face])
                self.budget.reserve(0, _TABLE_SLOT_BYTES)
                self.face_index[face] = index
            result.append(-index if negated[column] else index)
        return tuple(result)

    def pulled(
        self,
        cell: int,
        columns: tuple[int, ...],
        chart: int,
        cone_map: VolumeMaterialChartMap,
    ) -> tuple[int, ...]:
        """Signed references of ``cell``'s outward faces in ``columns`` pulled back by ``chart``.

        Pulling back is linear in the plane, so each face is pulled once per
        chart from its canonical equation and interned canonically; the
        owner's outward sign applies afterwards.
        """
        faces, negated = self.cells.faces(cell)
        result = []
        for column in columns:
            face = faces[column]
            self.budget.reserve(2)
            reference = self.pulled_index.get((face, chart))
            if reference is None:
                plane = _pulled(self.cells.face_planes[face], cone_map, self.budget)
                self.budget.reserve(3)
                pivot = next((axis for axis in range(3) if plane[axis]), None)
                if pivot is None:
                    raise ValueError(
                        "volume_material_edit pulled target faces must keep a nonzero normal; "
                        "got a singular cone map"
                    )
                divisor = plane[pivot]
                reserve_fraction_work((plane,), 4, 4, 2, budget=self.budget)
                index = self.plane(
                    (
                        plane[0] / divisor,
                        plane[1] / divisor,
                        plane[2] / divisor,
                        plane[3] / divisor,
                    )
                )
                reference = index if divisor > 0 else -index
                self.budget.reserve(0, _TABLE_SLOT_BYTES)
                self.pulled_index[face, chart] = reference
            result.append(-reference if negated[column] else reference)
        return tuple(result)

    def tetrahedron(self, cell: int) -> int:
        """Parent reference ``-(s + 1)`` of one registered old-frame cell."""
        self.budget.reserve(1)
        index = self.initial.get(cell)
        if index is None:
            corners = []
            for point in self.cells.corners[cell]:
                self.budget.reserve(1)
                local = self.point_index.get(point)
                if local is None:
                    self.budget.reserve(0, _TABLE_SLOT_BYTES)
                    local = len(self.points)
                    self.point_index[point] = local
                    self.points.append(self.cells.points[point])
                corners.append(local)
            planes = self.cuts(cell, (0, 1, 2, 3))
            self.budget.reserve(0, 2 * _TABLE_SLOT_BYTES)
            index = len(self.tetrahedra)
            self.initial[cell] = index
            self.tetrahedra.append((corners[0], corners[1], corners[2], corners[3]))
            self.tetrahedron_planes.append((planes[0], planes[1], planes[2], planes[3]))
        return -index - 1

    def job(self, parent: int, cuts: tuple[int, ...]) -> int:
        """Index of one job clipping ``parent`` by the signed references ``cuts``."""
        depth = (4 if parent < 0 else self.depths[parent]) + len(cuts)
        if depth > _MAXIMUM_HALFSPACES:
            raise RuntimeError(
                "volume_material_edit arrangement jobs must inherit at most "
                f"{_MAXIMUM_HALFSPACES} halfspaces; got {depth}"
            )
        self.budget.reserve(1 + len(cuts), _TABLE_SLOT_BYTES + 16 * len(cuts))
        self.parents.append(parent)
        self.depths.append(depth)
        self.cut_planes.extend(cuts)
        self.cut_offsets.append(len(self.cut_planes))
        return len(self.parents) - 1

    def intersection(self, cut: int, parent: int) -> int:
        """Job of ``cut ∩ parent``: ``parent`` clipped by all four outward faces of ``cut``.

        Both cells are initial tetrahedra, so the native pool recognizes the
        four signed references as exactly ``cut``'s declared planes, a purely
        geometric reuse, and screens the pair in both directions from exact
        sides at both cells' corners: a separating face empties the job, a
        cell inside the other is published whole (``cut`` first), and
        otherwise whichever cell fewer of the other's faces cut is clipped by
        exactly those faces (``parent`` on ties).
        """
        planes = self.tetrahedron_planes[-self.tetrahedron(cut) - 1]
        return self.job(self.tetrahedron(parent), planes)

    def execute(self) -> ExactTetrahedronArrangement:
        """Run every job of the edit in one native exact pool."""
        jobs, cuts, initial = len(self.parents), len(self.cut_planes), len(self.tetrahedra)
        # Seven host input banks and the two equation tuples before packing.
        self.budget.reserve(
            0,
            9 * 256
            + 8 * (len(self.points) + len(self.planes))
            + 32 * initial
            + 4 * jobs
            + 8 * (jobs + 1)
            + 4 * cuts,
        )
        return arrange_tetrahedra(
            tuple(self.points),
            tuple(self.planes),
            np.asarray(self.tetrahedra, dtype=np.int32).reshape((-1, 4)),
            np.asarray(self.tetrahedron_planes, dtype=np.int32).reshape((-1, 4)),
            np.asarray(self.parents, dtype=np.int32),
            np.asarray(self.cut_offsets, dtype=np.int64),
            np.asarray(self.cut_planes, dtype=np.int32),
            budget=self.budget,
        )


def volume_material_edit_pieces(
    edit: VolumeMaterialEdit,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> VolumeMaterialEditPieces:
    """Exact material pieces of one certified edit on one shared exact arrangement.

    Identity edits route ``s ∩ t`` and curved edits ``s ∩ C_k ∩ A_k^{-1}(t)``,
    all in the edit's old physical frame. After exact axis bounds, a cell
    pair is one job clipping the source by all four outward faces of the
    target or cone; the native pool recognizes that cut set as the declared
    initial tetrahedron and screens the pair by exact plane sides at both
    cells' corners: a separating face plane leaves no volume, a cell inside
    the other is published whole, and otherwise only the planes cutting the
    pair clip whichever cell they cut fewer times. A curved edit clips each
    source cell by a cone at most once, ``s ∩ C_k``. Each (target, image
    cone) pair is decided once, in the new frame, for every source under the
    cone: a separated pair routes nothing, a target containing the image cone
    routes ``s ∩ C_k`` itself, and otherwise only the target faces cutting
    the image cone are pulled back to clip that shared parent. All jobs of
    the edit run in one native exact pool; pairs meeting in at most a face
    publish no piece. Every source cell's pieces sum to its volume and every
    target cell's mapped pieces to its volume.
    """
    _, _, sources = edit.exact_cells("source", budget=budget)
    _, _, targets = edit.exact_cells("target", budget=budget)
    source_volumes = tuple(_tet_volume(tet, budget) for tet in sources)
    target_volumes = tuple(_tet_volume(tet, budget) for tet in targets)
    cells = _Cells(budget)
    source_cells = tuple(cells.add(tet) for tet in sources)
    target_cells = tuple(cells.add(tet) for tet in targets)
    plan = _Arrangement(cells)
    routes: list[tuple[int, int, int]] = []
    jobs: list[int] = []
    charts: list[VolumeMaterialChartMap] = []
    if not edit.curved:
        for target, tet in enumerate(target_cells):
            for source, other in enumerate(source_cells):
                if cells.disjoint(tet, other):
                    continue
                routes.append((target, source, -1))
                jobs.append(plan.intersection(tet, other))
    else:
        for origin, image in zip(edit.source_charts, edit.target_charts, strict=True):
            charts.append(_cone_map(origin, image, budget))
        cone_cells = tuple(cells.add(tet) for tet in edit.source_charts)
        image_cells = tuple(cells.add(tet) for tet in edit.target_charts)
        parents: dict[tuple[int, int], int | None] = {}

        def cone_parent(chart: int, source: int) -> int | None:
            """Job of ``s ∩ C_k``, planned at most once; ``None`` for disjoint bounds."""
            budget.reserve(2)
            if (chart, source) not in parents:
                origin, other = cone_cells[chart], source_cells[source]
                parents[chart, source] = (
                    None
                    if cells.disjoint(other, origin)
                    else plan.intersection(origin, other)
                )
            return parents[chart, source]

        for target, tet in enumerate(target_cells):
            for chart, image in enumerate(image_cells):
                if cells.disjoint(tet, image):
                    continue
                cutting = _decide(cells, tet, image, budget)
                if cutting is None:
                    continue
                # A target face containing the image cone contains every s ∩ C_k.
                pulled = plan.pulled(tet, cutting, chart, charts[chart])
                for source in range(len(source_cells)):
                    parent = cone_parent(chart, source)
                    if parent is None:
                        continue
                    routes.append((target, source, chart))
                    jobs.append(plan.job(parent, pulled) if pulled else parent)
    return _arranged_pieces(
        plan, routes, jobs, tuple(charts), source_volumes, target_volumes, budget
    )


def _arranged_pieces(
    plan: _Arrangement,
    routes: list[tuple[int, int, int]],
    jobs: list[int],
    charts: tuple[VolumeMaterialChartMap, ...],
    source_volumes: tuple[Fraction, ...],
    target_volumes: tuple[Fraction, ...],
    budget: CoordinateEnclosureBudget,
) -> VolumeMaterialEditPieces:
    """Exact volumes and positive pulling simplices of every routed job.

    One native call publishes every job's welded outward faces; empty and
    zero-measure jobs publish none and route no piece. A face's signed
    origin-cone contribution is evaluated once per welded loop modulo
    rotation and reused, negated, for the reversed loop of a neighbouring
    piece, whose canonical plane reference must agree. A piece's exact volume
    is the sum over its faces. Its pulling triangulation cones every face not
    incident to the piece's least vertex from that vertex and fans the face
    from its first vertex, so every corner is an existing exact vertex and
    no center is averaged; outward strictly convex loops make every
    tetrahedron positive. Only vertices of routed simplices are rounded, once
    each: a vertex proven equal to an input point gathers that point's
    rounding, every other vertex divides its exact integers.
    """
    if not plan.parents:
        _refuse(
            VolumeMaterialStatus.COVERAGE_GAP,
            "volume_material_edit pieces must exactly partition both cavities",
        )
    try:
        arrangement = plan.execute()
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        _refuse(
            VolumeMaterialStatus.RESOURCE_LIMIT,
            f"volume_material_edit exact arrangement refused its resources; got {error}",
        )
    cell_faces = np.asarray(arrangement.cell_face_offsets, dtype=np.int64)
    face_offsets = np.asarray(arrangement.face_offsets, dtype=np.int64)
    face_vertices = np.asarray(arrangement.face_vertices, dtype=np.int64)
    face_planes = np.asarray(arrangement.face_planes, dtype=np.int64)
    inputs = np.asarray(arrangement.vertex_input_index, dtype=np.int64)
    vertices = ArrangementVertices(arrangement, budget=budget)
    if (
        cell_faces.shape != (len(plan.parents) + 1,)
        or face_offsets.shape != (face_planes.size + 1,)
        or inputs.shape != (vertices.count,)
    ):
        raise ValueError(
            "volume_material_edit arrangement banks must cover every job, face and vertex"
        )
    contributions: dict[tuple[int, ...], tuple[int, Fraction]] = {}

    def contribution(loop: list[int], reference: int) -> Fraction:
        """Six times the signed origin-cone volume over one welded face loop."""
        # Native publication already rotates every loop to its least output
        # vertex. Copy that canonical key; do not find and rotate it again.
        budget.reserve(len(loop))
        key = tuple(loop)
        sign, cached = 1, contributions.get(key)
        if cached is None:
            sign, cached = -1, contributions.get((key[0], *reversed(key[1:])))
        if cached is not None:
            if cached[0] != sign * reference:
                _refuse(
                    VolumeMaterialStatus.INVALID_CAVITY,
                    "volume_material_edit matching arrangement faces must share their plane",
                )
            return cached[1] if sign > 0 else -cached[1]
        value = loop_volume_contribution(
            tuple(vertices[vertex] for vertex in key),
            plan.planes[abs(reference) - 1],
            budget=budget,
        )
        budget.reserve(0, _TABLE_SLOT_BYTES + 8 * len(key))
        contributions[key] = (reference, value)
        return value

    kept: list[tuple[int, int, int]] = []
    volumes: list[Fraction] = []
    offsets = [0]
    rows: list[tuple[int, int, int, int]] = []
    for route, job in zip(routes, jobs, strict=True):
        budget.reserve(1)
        first, last = int(cell_faces[job]), int(cell_faces[job + 1])
        if first == last:
            continue
        bounds = face_offsets[first : last + 1].tolist()
        entries = face_vertices[bounds[0] : bounds[-1]].tolist()
        references = face_planes[first:last].tolist()
        budget.reserve(len(entries) + 2 * (last - first))
        loops = [
            entries[bounds[face] - bounds[0] : bounds[face + 1] - bounds[0]]
            for face in range(last - first)
        ]
        faces = tuple(
            contribution(loop, reference)
            for loop, reference in zip(loops, references, strict=True)
        )
        reserve_fraction_work((faces,), len(faces) + 1, 2, len(faces) + 1, budget=budget)
        six = sum(faces, Fraction(0))
        if six <= 0:
            _refuse(
                VolumeMaterialStatus.INVALID_CAVITY,
                "volume_material_edit arrangement pieces must have positive volume",
            )
        apex = min(entries)
        for loop in loops:
            budget.reserve(len(loop))
            if apex in loop:
                continue
            budget.reserve(len(loop) - 2, 128 * (len(loop) - 2))
            rows.extend(
                (apex, loop[0], loop[position], loop[position + 1])
                for position in range(1, len(loop) - 1)
            )
        kept.append(route)
        volumes.append(six / 6)
        offsets.append(len(rows))
    _require_partition(kept, volumes, charts, source_volumes, target_volumes, budget)
    count = len(rows)
    # Simplex corner rows, their sorted unique vertices and inverse, the
    # edit-local rows, and the route and offset banks.
    budget.reserve(4 * count, 5 * 256 + count * (32 + 32 + 32 + 16) + 20 * len(kept))
    corners, inverse = np.unique(
        np.asarray(rows, dtype=np.int64).reshape(-1), return_inverse=True
    )
    simplices = inverse.reshape((-1, 4)).astype(np.int32)
    gathered = inputs[corners]
    original = gathered >= 0
    reused = gathered[original].tolist()
    # Proven input points gather their own exact rounding, once each.
    budget.reserve(3 * len(reused), 256 + corners.size * 24 + len(reused) * (24 + 3 * 32))
    rounded = np.empty((corners.size, 3), dtype=np.float64)
    rounded[original] = np.asarray(
        [plan.points[point] for point in reused], dtype=object
    ).reshape((-1, 3)).astype(np.float64)
    rounded[~original] = vertices.floats(corners[~original])
    exact = tuple(vertices[vertex] for vertex in corners.tolist())
    return VolumeMaterialEditPieces(
        np.asarray(kept, dtype=np.int32).reshape((-1, 3)),
        tuple(volumes),
        np.asarray(offsets, dtype=np.int64),
        simplices,
        rounded,
        exact,
        charts,
        source_volumes,
        target_volumes,
    )


def _require_partition(
    routes: Sequence[tuple[int, int, int]],
    volumes: Sequence[Fraction],
    charts: Sequence[VolumeMaterialChartMap],
    source_volumes: Sequence[Fraction],
    target_volumes: Sequence[Fraction],
    budget: CoordinateEnclosureBudget,
) -> None:
    """Every source cell's old-frame pieces and target cell's mapped pieces sum exactly."""
    sources = [Fraction(0)] * len(source_volumes)
    targets = [Fraction(0)] * len(target_volumes)
    for (target, source, chart), volume in zip(routes, volumes, strict=True):
        jacobian = Fraction(1) if chart < 0 else charts[chart].jacobian
        reserve_fraction_work(((volume, jacobian),), 3, 2, 4, budget=budget)
        sources[source] += volume
        targets[target] += volume * jacobian
    if sources != list(source_volumes) or targets != list(target_volumes):
        _refuse(
            VolumeMaterialStatus.COVERAGE_GAP,
            "volume_material_edit pieces must exactly partition both cavities",
        )


@dataclass(frozen=True, slots=True)
class _MaterialBank:
    """Per-edit exact density weights and binary64 shared-vertex piece simplices of one chain."""

    weight_offsets: np.ndarray
    weight_targets: np.ndarray
    weight_sources: np.ndarray
    weight_mantissas: np.ndarray
    weight_exponents: np.ndarray
    weight_mantissa_errors: np.ndarray
    target_measures: np.ndarray
    target_measure_errors: np.ndarray
    piece_offsets: np.ndarray
    piece_routes: np.ndarray
    piece_simplex_offsets: np.ndarray
    piece_simplex_vertices: np.ndarray
    piece_vertex_offsets: np.ndarray
    piece_vertices: np.ndarray
    chart_offsets: np.ndarray
    chart_linear: np.ndarray
    chart_translation: np.ndarray
    chart_jacobians: np.ndarray


def _key_volume(key: _CellKey, budget: CoordinateEnclosureBudget) -> Fraction:
    """Exact volume of one bound cell; keys are vertex-ID sorted, so unsigned."""
    determinant = star_determinant(
        (key[0][1], key[1][1], key[2][1], key[3][1]), budget=budget
    )
    if not determinant:
        _refuse(
            VolumeMaterialStatus.INVALID_CAVITY,
            "volume_material_chain cells must be nondegenerate",
        )
    reserve_fraction_work(((determinant,),), 1, 1, 1, budget=budget)
    return abs(determinant) / 6


@dataclass(frozen=True, slots=True)
class _EditBlock:
    """One edit's piece routes, weights, edit-local simplex rows, rounded vertices and cone maps."""

    routes: np.ndarray
    weight_targets: np.ndarray
    weight_sources: np.ndarray
    weight_mantissas: np.ndarray
    weight_exponents: np.ndarray
    weight_errors: np.ndarray
    simplex_counts: np.ndarray
    simplices: np.ndarray
    vertices: np.ndarray
    linear: np.ndarray
    translation: np.ndarray
    jacobians: np.ndarray


def _edit_block_bytes(
    pieces: int, pairs: int, simplices: int, vertices: int, charts: int
) -> int:
    """The block owner and its list slot, twelve ndarray headers and backing bytes."""
    return (
        272
        + 12 * 256
        + pieces * (12 + 8)
        + pairs * 28
        + simplices * 16
        + vertices * 24
        + charts * (72 + 24 + 8)
    )


# Weight exponents stay far inside int32 so a scaled kernel's zero sentinel
# (-2**30) and its exponent sums never collide with a real weight.
_WEIGHT_EXPONENT_LIMIT = 1 << 28


def _weight_pair(value: Fraction, /) -> tuple[float, int, float]:
    """``value = (mantissa + delta) * 2**exponent`` with ``mantissa`` in ``[1/2, 1)``.

    ``mantissa`` is the nearest binary64 to the exactly power-of-two scaled
    positive rational, so no weight overflows or underflows whatever its
    magnitude; the returned error bounds ``|delta|`` upward.
    """
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    # The bit lengths place value / 2**exponent in (1/2, 2).
    if (
        numerator >= denominator << exponent
        if exponent >= 0
        else numerator << -exponent >= denominator
    ):
        exponent += 1
    scaled = value * Fraction(2) ** -exponent
    mantissa = float(scaled)
    if mantissa == 1.0:
        # Rounded up to one: the same value is the half mantissa one binade up.
        exponent += 1
        scaled /= 2
        mantissa = 0.5
    if abs(exponent) >= _WEIGHT_EXPONENT_LIMIT:
        _refuse(
            VolumeMaterialStatus.RESOURCE_LIMIT,
            "volume_material_chain weight exponents must stay below 2**28 in magnitude",
        )
    return mantissa, exponent, _rne_error(scaled)


def _prepare_material_bank(
    target: Mapping[int, _CellKey],
    target_order: np.ndarray,
    edits: tuple[VolumeMaterialEdit, ...],
    budget: CoordinateEnclosureBudget,
) -> _MaterialBank:
    """Each edit's exact density weights and binary64 pieces; nothing is composed.

    Edit ``e`` maps the uniform densities of its removed cells ``s`` to its
    created cells ``t`` by ``W_ts = P_ts / |t|``, where ``P_ts`` is the exact
    source-frame volume of every piece routed from ``s`` to ``t``. Both
    partitions are proven exact per edit, so the chronological product of the
    edits is the exact material operator; untouched cells keep weight one.
    Each ``W_ts`` is published once as a binary64 mantissa in ``[1/2, 1)``, an
    exact integer exponent and the mantissa's certified rounding bound, so no
    weight overflows or underflows. Each edit's shared exact arrangement
    vertices are rounded once for finite-element quadrature and its piece
    simplices are edit-local vertex rows; weights never use the rounded
    vertices.

    Every edit's exact pieces and arrangement scratch retire with its own
    temporary scope; only the edit's binary64 block outlives it, admitted to
    one live bound, and blocks are concatenated once at the end.
    """
    blocks: list[_EditBlock] = []

    def fold(edit: VolumeMaterialEdit, live: CoordinateLiveStorage) -> None:
        """Group one edit's pieces by (target, source) and round its weights."""
        prepared = volume_material_edit_pieces(edit, budget=budget)
        pieces = prepared.routes.shape[0]
        # The route rows as Python ints and one grouping slot per piece.
        budget.reserve(2 * pieces, 256 + 320 * pieces)
        groups: dict[tuple[int, int], list[Fraction]] = {}
        for (target_index, source_index, _), volume in zip(
            prepared.routes.tolist(), prepared.volumes, strict=True
        ):
            groups.setdefault((target_index, source_index), []).append(volume)
        pairs = sorted(groups)
        # Sorting the pair keys, then the integer scaling and rounding steps.
        budget.reserve(
            len(pairs) * max(len(pairs), 1).bit_length() + 4 * len(pairs),
            256 + 64 * len(pairs),
        )
        weights = []
        for target_index, source_index in pairs:
            volumes = groups[(target_index, source_index)]
            measure = prepared.target_volumes[target_index]
            reserve_fraction_work(
                ((*volumes, measure),), len(volumes), 1, len(volumes) + 1, budget=budget
            )
            weights.append(_weight_pair(sum(volumes, Fraction(0)) / measure))
        simplices = prepared.simplices.shape[0]
        vertices = prepared.vertices.shape[0]
        charts = len(prepared.charts)
        # Route visits and thirteen cone-map roundings, five weight banks;
        # the edit's own simplex rows and rounded vertices move into the
        # block uncopied.
        budget.reserve(3 * pieces + 13 * charts + 5 * len(pairs))
        live.set_bound(
            live.bound
            + _edit_block_bytes(pieces, len(pairs), simplices, vertices, charts)
        )
        pair_rows = np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
        block = _EditBlock(
            prepared.routes,
            pair_rows[:, 0].copy(),
            pair_rows[:, 1].copy(),
            np.asarray([mantissa for mantissa, _, _ in weights], dtype=np.float64),
            np.asarray([exponent for _, exponent, _ in weights], dtype=np.int32),
            np.asarray([error for _, _, error in weights], dtype=np.float64),
            np.diff(prepared.simplex_offsets),
            prepared.simplices,
            prepared.vertices,
            np.empty((charts, 3, 3), dtype=np.float64),
            np.empty((charts, 3), dtype=np.float64),
            np.empty((charts,), dtype=np.float64),
        )
        for position, chart in enumerate(prepared.charts):
            block.linear[position] = [
                [float(value) for value in row] for row in chart.linear
            ]
            block.translation[position] = [float(value) for value in chart.offset]
            block.jacobians[position] = float(chart.jacobian)
        blocks.append(block)

    with budget.live_storage() as live:
        # The block list container.
        live.set_bound(256)
        for edit in edits:
            with budget.temporary_scope():
                fold(edit, live)
        pieces = sum(block.routes.shape[0] for block in blocks)
        pairs = sum(block.weight_targets.shape[0] for block in blocks)
        simplices = sum(block.simplices.shape[0] for block in blocks)
        vertices = sum(block.vertices.shape[0] for block in blocks)
        charts = sum(block.jacobians.shape[0] for block in blocks)
        # The published bank outlives this owner: eighteen ndarray headers
        # and their actual bytes, admitted before any of them is allocated.
        budget.reserve(
            target_order.size + 13 * len(blocks),
            18 * 256
            + 16 * target_order.size
            + 8 * (4 * len(blocks) + 4)
            + 28 * pairs
            + 8 * (pieces + 1)
            + 12 * pieces
            + 16 * simplices
            + 24 * vertices
            + 104 * charts,
        )
        with budget.temporary_scope():
            budget.reserve(0, 256 + 64 * target_order.size)
            target_volumes = [
                _key_volume(target[int(identity)], budget)
                for identity in target_order.tolist()
            ]
            budget.reserve(2 * len(target_volumes))
            target_measures = np.asarray(
                [float(value) for value in target_volumes], dtype=np.float64
            )
            target_errors = np.asarray(
                [_rne_error(value) for value in target_volumes], dtype=np.float64
            )

        def offsets(counts: list[int]) -> np.ndarray:
            result = np.zeros((len(counts) + 1,), dtype=np.int64)
            result[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
            return result

        def joined(name: str, shape: tuple[int, ...], dtype: type) -> np.ndarray:
            return np.concatenate(
                [np.empty(shape, dtype=dtype), *(getattr(block, name) for block in blocks)]
            )

        return _MaterialBank(
            offsets([block.weight_targets.shape[0] for block in blocks]),
            joined("weight_targets", (0,), np.int32),
            joined("weight_sources", (0,), np.int32),
            joined("weight_mantissas", (0,), np.float64),
            joined("weight_exponents", (0,), np.int32),
            joined("weight_errors", (0,), np.float64),
            target_measures,
            target_errors,
            offsets([block.routes.shape[0] for block in blocks]),
            joined("routes", (0, 3), np.int32),
            offsets([int(count) for block in blocks for count in block.simplex_counts]),
            joined("simplices", (0, 4), np.int32),
            offsets([block.vertices.shape[0] for block in blocks]),
            joined("vertices", (0, 3), np.float64),
            offsets([block.jacobians.shape[0] for block in blocks]),
            joined("linear", (0, 3, 3), np.float64),
            joined("translation", (0, 3), np.float64),
            joined("jacobians", (0,), np.float64),
        )


def volume_material_edit_weights(
    chain: VolumeMaterialChain,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-edit density weights ``(offsets, targets, sources, mantissas, exponents, errors)``.

    Edit ``e`` owns rows ``offsets[e]:offsets[e + 1]``; row ``r`` maps the
    density of the edit's removed cell ``sources[r]`` to its created cell
    ``targets[r]`` (edit-local positions in ``source_cell_ids`` and
    ``target_cell_ids``) with exact weight ``W = (mantissas[r] + d) *
    2**exponents[r]``, ``|d| <= errors[r]``.
    """
    return (
        np.asarray(chain.weight_offsets, dtype=np.int64),
        np.asarray(chain.weight_targets, dtype=np.int32),
        np.asarray(chain.weight_sources, dtype=np.int32),
        np.asarray(chain.weight_mantissas, dtype=np.float64),
        np.asarray(chain.weight_exponents, dtype=np.int32),
        np.asarray(chain.weight_mantissa_errors, dtype=np.float64),
    )


def volume_material_target_measures(
    chain: VolumeMaterialChain,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Rounded exact target cell volumes in block order, with their rounding errors."""
    return (
        np.asarray(chain.target_measures, dtype=np.float64),
        np.asarray(chain.target_measure_errors, dtype=np.float64),
    )


__all__ = [
    "VolumeMaterialChain",
    "VolumeMaterialChartMap",
    "VolumeMaterialEdit",
    "VolumeMaterialEditPieces",
    "VolumeMaterialMapError",
    "VolumeMaterialStatus",
    "volume_material_edit_pieces",
    "volume_material_edit_weights",
    "volume_material_target_measures",
]
