#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native-clipped, exactly constructed physical-cell sphere correspondence.

Native arbitrary-integer halfplane clipping consumes original rational rows
after exact positive denominator clearing. Original supporting-plane pairs
reconstruct every retained vertex exactly; boundary/feasibility checks and
complete rational partitions of BOTH physical reference meshes decide
acceptance. Projective target reference maps retain their actual quotients.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .. import _meshcore
from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._interval_enclosure import interval_divide
from ..geometry._mesh_certificates import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
)
from ..geometry._sphere_material_atlas import (
    SphereMaterialCellAtlas,
    SphereProjectiveReferenceMap,
    SphereProjectiveTriangleBounds,
)
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    CoordinateEnclosureBudget,
    Expression,
    ExpressionComposition,
    outward,
)


if TYPE_CHECKING:
    from ._cell_geometry_transfer import CellGeometryTransitionPolicy

type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _Plane = tuple[Fraction, Fraction, Fraction]
type _FloatArray = NDArray[np.float64]
type _IntArray = NDArray[np.int64]


class PreparedSphereChartPiece(StrictModule, NonTrainableState):
    """Exact source-reference triangle with its genuine rational target map."""

    source_cell_global_id: int = eqx.field(static=True)
    target_cell_global_id: int = eqx.field(static=True)
    source_row: int = eqx.field(static=True)
    target_row: int = eqx.field(static=True)
    source_atlas_row: int = eqx.field(static=True)
    target_atlas_row: int = eqx.field(static=True)
    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    projective_map: SphereProjectiveReferenceMap
    exact_source_reference_vertices: _Triangle = eqx.field(static=True)
    exact_target_reference_vertices: _Triangle = eqx.field(static=True)
    source_reference_vertices: Array
    target_reference_vertices: Array
    projective_bounds: SphereProjectiveTriangleBounds = eqx.field(static=True)
    source_reference_area: Fraction = eqx.field(static=True)
    target_reference_area: Fraction = eqx.field(static=True)
    native_boundary_labels: tuple[int, ...] = eqx.field(static=True)
    native_polygon_vertex_indices: tuple[int, int, int] = eqx.field(static=True)
    native_clip_id: str = eqx.field(static=True)
    displacement_bound: float = eqx.field(static=True)
    piece_id: str = eqx.field(static=True)


class PreparedSphereChartOccurrence(StrictModule, NonTrainableState):
    """Unmerged scientific occurrence and its complete exact overlap pieces."""

    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    piece_indices: tuple[int, ...] = eqx.field(static=True)
    source_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    target_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    coverage_status: str = eqx.field(static=True)


class PreparedSphereChartDeformation(StrictModule, NonTrainableState):
    """Accepted actual full-sphere transition, never a latitude/longitude alias."""

    source_atlas: SphereMaterialCellAtlas
    target_atlas: SphereMaterialCellAtlas
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    pieces: tuple[PreparedSphereChartPiece, ...]
    occurrences: tuple[PreparedSphereChartOccurrence, ...]
    source_reference_coverage: tuple[Fraction, ...] = eqx.field(static=True)
    target_reference_coverage: tuple[Fraction, ...] = eqx.field(static=True)
    source_fidelity_bounds: Array
    target_fidelity_bounds: Array
    source_cell_measures: Array
    target_cell_measures: Array
    source_measure_errors: Array
    target_measure_errors: Array
    source_validity: CellValidityCertificate
    target_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    maximum_displacement_bound: float = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    retained_bytes_upper: int = eqx.field(static=True)
    deformation_id: str = eqx.field(static=True)

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_domain_coverage(self) -> str:
        return (
            "certified"
            if self.source_atlas.coverage_status == "certified"
            and all(area == Fraction(1, 2) for area in self.source_reference_coverage)
            else "unresolved"
        )

    @property
    def target_domain_coverage(self) -> str:
        return (
            "certified"
            if self.target_atlas.coverage_status == "certified"
            and all(area == Fraction(1, 2) for area in self.target_reference_coverage)
            else "unresolved"
        )

    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None:
        if (
            source_mesh.mesh_id,
            target_mesh.mesh_id,
            cell_geometry_id(source_geometry),
            cell_geometry_id(target_geometry),
        ) != (
            self.source_mesh.mesh_id,
            self.target_mesh.mesh_id,
            self.source_geometry_id,
            self.target_geometry_id,
        ):
            raise ValueError(
                "Sphere correspondence is stale for these actual coordinate maps or cells."
            )
        for atlas, mesh, geometry, validity, embedding in (
            (
                self.source_atlas,
                source_mesh,
                source_geometry,
                self.source_validity,
                self.source_embedding,
            ),
            (
                self.target_atlas,
                target_mesh,
                target_geometry,
                self.target_validity,
                self.target_embedding,
            ),
        ):
            atlas.require_bound(atlas.domain, mesh, geometry, atlas.coordinate_contract)
            validity.require_bound(geometry, mesh=mesh)
            embedding.binding.require(mesh, geometry)
            if not validity.all_certified or embedding.status != "certified":
                raise ValueError(
                    "Sphere correspondence requires actually valid globally embedded coordinate maps."
                )
        if (
            self.source_domain_coverage != "certified"
            or self.target_domain_coverage != "certified"
        ):
            raise ValueError(
                "Sphere correspondence lacks exact COMPLETE source and target partitions."
            )
        source_coverage = [Fraction(0) for _ in self.source_reference_coverage]
        target_coverage = [Fraction(0) for _ in self.target_reference_coverage]
        for piece in self.pieces:
            source_row = self.source_atlas.cell_row(piece.source_cell_global_id)
            target_row = self.target_atlas.cell_row(piece.target_cell_global_id)
            if (
                piece.source_atlas_row,
                piece.target_atlas_row,
                piece.source_row,
                piece.target_row,
            ) != (
                source_row,
                target_row,
                int(np.asarray(self.source_atlas.physical_rows)[source_row]),
                int(np.asarray(self.target_atlas.physical_rows)[target_row]),
            ):
                raise ValueError(
                    "Sphere piece physical rows do not retain their scientific cell identities."
                )
            if (
                piece.projective_map.source_cell_global_id,
                piece.projective_map.target_cell_global_id,
                piece.geometry_entity_id,
                piece.occurrence_path,
            ) != (
                piece.source_cell_global_id,
                piece.target_cell_global_id,
                self.source_atlas.geometry_entity_ids[source_row],
                self.source_atlas.occurrence_paths[source_row],
            ):
                raise ValueError(
                    "Sphere piece rational map or occurrence changes its scientific endpoints."
                )
            piece.projective_map.require_bound(self.source_atlas, self.target_atlas)
            bounds = piece.projective_map.certify_triangle(
                piece.exact_source_reference_vertices
            )
            vertices = piece.exact_source_reference_vertices
            target: _Triangle = (
                _project(piece.projective_map, vertices[0]),
                _project(piece.projective_map, vertices[1]),
                _project(piece.projective_map, vertices[2]),
            )
            if (
                target != piece.exact_target_reference_vertices
                or bounds != piece.projective_bounds
                or piece.source_reference_area
                != _area(piece.exact_source_reference_vertices)
                or piece.target_reference_area
                != _area(piece.exact_target_reference_vertices)
            ):
                raise ValueError(
                    "Sphere piece changes its exact reference construction or rational map."
                )
            if not np.array_equal(
                np.asarray(piece.source_reference_vertices),
                _representatives(piece.exact_source_reference_vertices),
            ) or not np.array_equal(
                np.asarray(piece.target_reference_vertices),
                _representatives(piece.exact_target_reference_vertices),
            ):
                raise ValueError(
                    "Sphere piece representatives no longer match the exact constructions."
                )
            source_coverage[source_row] += piece.source_reference_area
            target_coverage[target_row] += piece.target_reference_area
        if (
            tuple(source_coverage) != self.source_reference_coverage
            or tuple(target_coverage) != self.target_reference_coverage
        ):
            raise ValueError(
                "Sphere pieces no longer provide their exact complete coverage evidence."
            )


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _tokens(vertices: tuple[_Point, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple((value.numerator, value.denominator) for value in point)
        for point in vertices
    )


def _representatives(vertices: _Triangle) -> _FloatArray:
    return np.asarray(
        [[float(value) for value in point] for point in vertices], dtype=np.float64
    )


def _area(vertices: tuple[_Point, ...]) -> Fraction:
    return (
        sum(
            (
                first[0] * second[1] - first[1] * second[0]
                for first, second in zip(
                    vertices, vertices[1:] + vertices[:1], strict=True
                )
            ),
            Fraction(0),
        )
        / 2
    )


def _project(mapping: SphereProjectiveReferenceMap, point: _Point) -> _Point:
    homogeneous = tuple(
        row[0] + row[1] * point[0] + row[2] * point[1]
        for row in mapping.exact_coefficients
    )
    denominator = sum(homogeneous, Fraction(0))
    if denominator <= 0 or any(value < 0 for value in homogeneous):
        raise ValueError(
            "An exact sphere overlap leaves its target cone or positive reference denominator."
        )
    return homogeneous[1] / denominator, homogeneous[2] / denominator


def _direction_boxes(atlas: SphereMaterialCellAtlas) -> tuple[_FloatArray, _FloatArray]:
    directions = np.asarray(atlas.directions, dtype=np.float64)
    low, high = interval_divide(
        (np.min(directions, axis=1), np.max(directions, axis=1)),
        (
            np.asarray(atlas.hemisphere_lower)[:, None],
            np.asarray(atlas.direction_norm_upper)[:, None],
        ),
    )
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def _cone_candidates(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    source_rows: _IntArray,
    target_rows: _IntArray,
    memory: int,
    /,
) -> tuple[_IntArray, _IntArray]:
    """Discard only certified halfspace-separated or zero-area cone pairs."""
    from .._geometry_predicates import orient3d, PredicateMode
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(0, source_rows.size)
    keep = np.ones(source_rows.shape, dtype=np.bool_)
    banks = (np.asarray(source.directions), np.asarray(target.directions))
    orientations = tuple(
        np.asarray(
            [1 if value > 0 else -1 for value in atlas.exact_determinants],
            dtype=np.int8,
        )
        for atlas in (source, target)
    )
    chunk = min(4096, memory // 2048)
    if chunk < 1:
        raise ValueError(
            "Sphere cone candidate filtering exceeds its original resident memory limit."
        )
    for start in range(0, source_rows.size, chunk):
        stop = min(start + chunk, source_rows.size)
        with ledger.temporary_scope() if ledger is not None else nullcontext():
            if ledger is not None:
                ledger.reserve(0, (stop - start) * 2048)
            rows = (source_rows[start:stop], target_rows[start:stop])
            rays = (banks[0][rows[0]], banks[1][rows[1]])
            for owner, other in ((0, 1), (1, 0)):
                predicate = orient3d(
                    np.zeros((3,), dtype=np.float64),
                    rays[owner][:, (0, 1, 2), None, :],
                    rays[owner][:, (1, 2, 0), None, :],
                    rays[other][:, None, :, :],
                    mode=PredicateMode.EXACT,
                )
                signs = (
                    np.asarray(predicate.signs)
                    * orientations[owner][rows[owner], None, None]
                )
                # Resolve uncertain filtered signs before forming projective
                # coefficient maps. A closed exterior halfspace admits only
                # zero-area contact, which cannot contribute reference coverage.
                separated = np.any(
                    np.all(np.asarray(predicate.certain) & (signs <= 0), axis=-1), axis=-1
                )
                keep[start:stop] &= ~separated
    if ledger is not None:
        ledger.reserve(0, source_rows.size * np.dtype(np.int64).itemsize)
    kept = np.flatnonzero(keep)
    if ledger is not None:
        ledger.reserve(0, 2 * kept.size * np.dtype(np.int64).itemsize)
    return source_rows[kept], target_rows[kept]


def _candidates(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    limit: int,
    memory: int,
) -> tuple[_IntArray, _IntArray]:
    source_low, source_high = _direction_boxes(source)
    target_low, target_high = _direction_boxes(target)
    source_bvh = prepare_bvh(source_low, source_high, dtype=jnp.float64)
    target_bvh = prepare_bvh(target_low, target_high, dtype=jnp.float64)
    targets: list[_IntArray] = []
    sources: list[_IntArray] = []
    count = 0
    for target_rows, source_rows in bvh_overlap_pair_blocks(target_bvh, source_bvh):
        count += target_rows.size
        if count > limit or count * 32 > memory:
            raise ValueError(
                "Sphere native candidate pairs exceed their work or resident memory limit."
            )
        targets.append(np.asarray(target_rows, dtype=np.int64))
        sources.append(np.asarray(source_rows, dtype=np.int64))
    if not sources:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    source_rows, target_rows = np.concatenate(sources), np.concatenate(targets)
    order = np.lexsort((source_rows, target_rows))
    return _cone_candidates(
        source, target, source_rows[order], target_rows[order], memory
    )


def _planes(mapping: SphereProjectiveReferenceMap) -> tuple[_Plane, ...]:
    return (
        (Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
    ) + tuple((-row[1], -row[2], row[0]) for row in mapping.exact_coefficients)


def _support(label: int, planes: tuple[_Plane, ...]) -> _Plane:
    if label < 0 or label >= len(planes):
        raise ValueError("Native polygon names an unknown original rational halfplane.")
    return planes[label]


def _native_polygon(
    mapping: SphereProjectiveReferenceMap,
    memory: int,
    maximum_work: int,
    coordinate_budget: CoordinateEnclosureBudget | None,
) -> tuple[tuple[_Point, ...], tuple[int, ...], int]:
    planes = _planes(mapping)
    supports, labels, work = _meshcore.clip_reference_triangle_exact(
        planes,
        maximum_scratch_bytes=memory,
        maximum_work_units=maximum_work,
    )
    count = supports.shape[0]
    if count < 3:
        return (), (), work
    edge_labels = tuple(int(value) for value in labels)
    vertices: list[_Point] = []
    source_corners = {
        (0, 1): (Fraction(0), Fraction(0)),
        (0, 2): (Fraction(0), Fraction(1)),
        (1, 2): (Fraction(1), Fraction(0)),
    }
    for slot in range(count):
        first_label, second_label = map(int, supports[slot])
        first = _support(first_label, planes)
        second = _support(second_label, planes)
        # Native support identities, not coordinate coincidence, name the
        # already-authored corners of the original reference triangle.
        point = source_corners.get(
            (min(first_label, second_label), max(first_label, second_label))
        )
        if point is None:
            with (
                coordinate_budget.temporary_scope()
                if coordinate_budget is not None
                else nullcontext()
            ):
                construction = prepare_exact_small_linear_actions(
                    ((first[0], first[1]), (second[0], second[1])),
                    ((first[2],), (second[2],)),
                    coordinate_budget=coordinate_budget,
                )
            work += construction.operation_count
            if construction.actions is None:
                raise ValueError(
                    "Native exact boundary cannot reconstruct a nonsingular original rational vertex."
                )
            point = construction.actions[0][0], construction.actions[1][0]
        if any(
            normal[0] * point[0] + normal[1] * point[1] > normal[2] for normal in planes
        ) or any(value < 0 or value > 1 for value in point):
            raise ValueError(
                "Native exact topology fails original rational halfplane feasibility."
            )
        vertices.append(point)
    polygon = tuple(vertices)
    if len(set(polygon)) != len(polygon) or _area(polygon) <= 0:
        raise ValueError(
            "Native exact polygon fails positive distinct-vertex construction."
        )
    for slot in range(count):
        first, second, third = (
            polygon[slot - 1],
            polygon[slot],
            polygon[(slot + 1) % count],
        )
        if (second[0] - first[0]) * (third[1] - second[1]) - (second[1] - first[1]) * (
            third[0] - second[0]
        ) <= 0:
            raise ValueError(
                "Native exact polygon is not the strictly convex original rational intersection."
            )
        support = _support(edge_labels[slot], planes)
        if any(
            support[0] * point[0] + support[1] * point[1] != support[2]
            for point in (second, third)
        ):
            raise ValueError(
                "Native exact edge does not retain its original rational boundary support."
            )
    return polygon, edge_labels, work


def _physical_order(atlas: SphereMaterialCellAtlas, values: Array) -> _FloatArray:
    result = np.empty(atlas.num_charts, dtype=np.float64)
    result[np.asarray(atlas.physical_rows, dtype=np.int64)] = np.asarray(
        values, dtype=np.float64
    )
    return result


def prepare_sphere_chart_deformation(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    /,
    *,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_pieces: int = 1_000_000,
    maximum_work_units: int = 10_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    prepared_target_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
) -> PreparedSphereChartDeformation:
    """Certify actual closed material cells and retained projective overlap pieces."""
    from ._cell_geometry_transfer import _certified_cell_measures
    from ._coordinate_enclosure import coordinate_enclosure_budget

    if not isinstance(source_atlas, SphereMaterialCellAtlas) or not isinstance(
        target_atlas, SphereMaterialCellAtlas
    ):
        raise TypeError(
            "Sphere deformation requires actual admitted material atlas owners."
        )
    candidates_limit = _positive(maximum_candidate_pairs, "maximum_candidate_pairs")
    pieces_limit = _positive(maximum_pieces, "maximum_pieces")
    work_limit = _positive(maximum_work_units, "maximum_work_units")
    memory_limit = _positive(maximum_memory_bytes, "maximum_memory_bytes")
    if coordinate_budget is None:
        coordinate_budget = coordinate_enclosure_budget(work_limit, memory_limit)
    if any(
        not math.isfinite(value) or value < 0
        for value in (maximum_fidelity, maximum_displacement)
    ):
        raise ValueError(
            "Sphere deformation fidelity/displacement limits must be finite and nonnegative."
        )
    for atlas in (source_atlas, target_atlas):
        atlas.require_bound(
            atlas.domain,
            atlas.source_mesh,
            atlas.source_geometry,
            atlas.coordinate_contract,
        )
        atlas.source_mesh.require_dense("Sphere material deformation")
    if (
        source_atlas.domain_id,
        source_atlas.source_id,
        source_atlas.source_revision,
        source_atlas.coordinate_contract.spatial_id,
    ) != (
        target_atlas.domain_id,
        target_atlas.source_id,
        target_atlas.source_revision,
        target_atlas.coordinate_contract.spatial_id,
    ):
        raise ValueError(
            "Sphere source and target atlases must retain one authoritative revision, domain and spatial contract."
        )
    source_groups = set(
        zip(source_atlas.geometry_entity_ids, source_atlas.occurrence_paths, strict=True)
    )
    target_groups = set(
        zip(target_atlas.geometry_entity_ids, target_atlas.occurrence_paths, strict=True)
    )
    if source_groups != target_groups:
        raise ValueError("Sphere correspondence cannot omit or merge source occurrences.")
    count = source_atlas.num_charts + target_atlas.num_charts
    if coordinate_budget is not None:
        if not isinstance(coordinate_budget, CoordinateEnclosureBudget):
            raise TypeError(
                "Sphere deformation requires the supplied original coordinate ledger."
            )
        coordinate_budget.reserve(count * 64, count * 1024)
    retained = count * 1024
    work = count * 64
    if retained >= memory_limit or work > work_limit:
        raise ValueError(
            "Sphere correspondence exceeds its bounded preparation footprint."
        )
    source_bounds = _physical_order(source_atlas, source_atlas.source_fidelity_bounds)
    target_bounds = _physical_order(target_atlas, target_atlas.source_fidelity_bounds)
    if max(float(np.max(source_bounds)), float(np.max(target_bounds))) > maximum_fidelity:
        raise ValueError(
            "Actual source or target polynomial coordinate fidelity exceeds the sphere bound."
        )
    source_candidates, target_candidates = _candidates(
        source_atlas, target_atlas, candidates_limit, memory_limit - retained
    )
    retained += source_candidates.nbytes + target_candidates.nbytes
    if work + source_candidates.size * 64 > work_limit:
        raise ValueError(
            "Sphere exact coefficient candidates exceed their preparation work budget."
        )
    pieces: list[PreparedSphereChartPiece] = []
    source_coverage = [Fraction(0) for _ in range(source_atlas.num_charts)]
    target_coverage = [Fraction(0) for _ in range(target_atlas.num_charts)]
    maximum_bound = 0.0
    coordinate_maps: (
        tuple[
            tuple[tuple[Expression, ...], ...],
            tuple[tuple[Expression, ...], ...],
        ]
        | None
    ) = None
    for source_index, target_index in zip(
        source_candidates, target_candidates, strict=True
    ):
        with (
            coordinate_budget.temporary_scope()
            if coordinate_budget is not None
            else nullcontext()
        ):
            source_row, target_row = int(source_index), int(target_index)
            group = (
                source_atlas.geometry_entity_ids[source_row],
                source_atlas.occurrence_paths[source_row],
            )
            if group != (
                target_atlas.geometry_entity_ids[target_row],
                target_atlas.occurrence_paths[target_row],
            ):
                continue
            source_id = int(np.asarray(source_atlas.cell_global_ids)[source_row])
            target_id = int(np.asarray(target_atlas.cell_global_ids)[target_row])
            mapping = source_atlas.projective_reference_map(
                source_id, target_atlas, target_id, coordinate_budget=coordinate_budget
            )
            if work_limit - work - 64 <= 0:
                raise ValueError(
                    "Sphere correspondence exhausts work before native exact clipping."
                )
            polygon, labels, clipping_work = _native_polygon(
                mapping,
                memory_limit - retained,
                work_limit - work - 64,
                coordinate_budget,
            )
            work += clipping_work + 64
            if work > work_limit:
                raise ValueError(
                    "Sphere exact clipping/construction exceeds its declared work budget."
                )
            if not polygon:
                continue
            clip_id = canonical_fingerprint(
                {
                    "kind": "native-sphere-exact-boundary",
                    "map": mapping.map_id,
                    "vertices": _tokens(polygon),
                    "support_labels": labels,
                }
            )
            physical_source = int(np.asarray(source_atlas.physical_rows)[source_row])
            physical_target = int(np.asarray(target_atlas.physical_rows)[target_row])
            # Both coordinate carriers approximate the same authoritative radial
            # point selected by the exact projective map. Their certified
            # source-fidelity radii therefore give a complete triangle-inequality
            # displacement bound without expanding a removable rational
            # difference. Fall back to the coefficient proof only when that
            # independently earned bound is too coarse for this policy.
            if coordinate_budget is not None:
                coordinate_budget.reserve(1, 512)
            displacement = outward(
                Fraction(float(source_bounds[physical_source]))
                + Fraction(float(target_bounds[physical_target])),
                math.inf,
            )
            displacement_map: tuple[Expression, ...] | None = None
            if displacement > maximum_displacement:
                if coordinate_maps is None:
                    coordinate_maps = (
                        _sphere_full_coordinate_expressions(
                            source_atlas, coordinate_budget
                        ),
                        _sphere_full_coordinate_expressions(
                            target_atlas, coordinate_budget
                        ),
                    )
                displacement_map = _sphere_displacement_map(
                    coordinate_maps[0][physical_source],
                    coordinate_maps[1][physical_target],
                    mapping,
                    coordinate_budget,
                )
            for slot in range(1, len(polygon) - 1):
                vertices: _Triangle = polygon[0], polygon[slot], polygon[slot + 1]
                bounds, target_vertices, source_area, target_area = (
                    _sphere_piece_geometry(mapping, vertices, coordinate_budget)
                )
                if displacement_map is not None:
                    displacement = _sphere_full_map_displacement(
                        displacement_map, vertices, coordinate_budget
                    )
                maximum_bound = max(maximum_bound, displacement)
                if displacement > maximum_displacement:
                    raise ValueError(
                        "Actual full-coordinate sphere deformation exceeds its whole-piece displacement bound."
                    )
                if source_area <= 0 or target_area <= 0:
                    raise ValueError(
                        "Sphere overlap triangle must have exact positive source and target reference area."
                    )
                fraction_bytes = sum(
                    (value.numerator.bit_length() + value.denominator.bit_length() + 7)
                    // 8
                    + 128
                    for point in (*vertices, *target_vertices)
                    for value in point
                )
                retained += 2048 + fraction_bytes
                if retained > memory_limit or len(pieces) >= pieces_limit:
                    raise ValueError(
                        "Sphere retained exact overlap artifacts exceed memory or piece limits."
                    )
                piece_id = canonical_fingerprint(
                    {
                        "kind": "sphere-exact-reference-piece",
                        "clip": clip_id,
                        "source": _tokens(vertices),
                        "target": _tokens(target_vertices),
                        "indices": (0, slot, slot + 1),
                    }
                )
                if coordinate_budget is not None:
                    coordinate_budget.retain_basis(
                        (
                            vertices,
                            target_vertices,
                            mapping.exact_coefficients,
                            mapping.exact_denominator,
                            mapping.orientation_ratio,
                            bounds.denominator_lower,
                            bounds.denominator_upper,
                            bounds.jacobian_lower,
                            bounds.jacobian_upper,
                            source_area,
                            target_area,
                        )
                    )
                    coordinate_budget.reserve(2)
                pieces.append(
                    PreparedSphereChartPiece(
                        source_cell_global_id=source_id,
                        target_cell_global_id=target_id,
                        source_row=physical_source,
                        target_row=physical_target,
                        source_atlas_row=source_row,
                        target_atlas_row=target_row,
                        geometry_entity_id=group[0],
                        occurrence_path=group[1],
                        projective_map=mapping,
                        exact_source_reference_vertices=vertices,
                        exact_target_reference_vertices=target_vertices,
                        source_reference_vertices=jnp.asarray(
                            _representatives(vertices), dtype=jnp.float64
                        ),
                        target_reference_vertices=jnp.asarray(
                            _representatives(target_vertices), dtype=jnp.float64
                        ),
                        projective_bounds=bounds,
                        source_reference_area=source_area,
                        target_reference_area=target_area,
                        native_boundary_labels=labels,
                        native_polygon_vertex_indices=(0, slot, slot + 1),
                        native_clip_id=clip_id,
                        displacement_bound=displacement,
                        piece_id=piece_id,
                    )
                )
                source_coverage[source_row] += source_area
                target_coverage[target_row] += target_area
    if any(value != Fraction(1, 2) for value in (*source_coverage, *target_coverage)):
        raise ValueError(
            "Native exact sphere correspondence has a source/target gap or double cover; COMPLETE rational reference coverage required."
        )
    validity_policy = (
        CellValidityPolicy(
            maximum_piece_count=min(work_limit, memory_limit // 1024),
            maximum_bernstein_nodes=min(work_limit, memory_limit // 1024),
        )
        if validity_policy is None
        else validity_policy
    )
    limits = (
        MeshCertificateLimits(
            maximum_candidate_pairs=candidates_limit,
            maximum_source_samples=work_limit,
            maximum_distance_evaluations=work_limit,
            maximum_subdivision_pieces=min(work_limit, memory_limit // 1024),
            maximum_bernstein_nodes=min(work_limit, memory_limit // 1024),
        )
        if certificate_limits is None
        else certificate_limits
    )
    validities: list[CellValidityCertificate] = []
    embeddings: list[GlobalEmbeddingCertificate] = []
    measures: list[_FloatArray] = []
    errors: list[_FloatArray] = []
    with coordinate_budget.activate() if coordinate_budget is not None else nullcontext():
        for atlas, prepared in (
            (source_atlas, prepared_source_certificates),
            (target_atlas, prepared_target_certificates),
        ):
            if prepared is None:
                validity = certify_cell_geometry_validity(
                    atlas.source_geometry, mesh=atlas.source_mesh, policy=validity_policy
                )
                embedding = certify_global_embedding(
                    atlas.source_mesh, atlas.source_geometry, validity, limits=limits
                )
            else:
                validity, embedding = prepared
                validity.require_bound(atlas.source_geometry, mesh=atlas.source_mesh)
                embedding.binding.require(atlas.source_mesh, atlas.source_geometry)
                if (
                    validity.policy_id != validity_policy.policy_id
                    or embedding.binding.limits_id != limits.limits_id
                ):
                    raise ValueError(
                        "Prepared sphere certificates change the original owning proof controls."
                    )
            if not validity.all_certified or embedding.status != "certified":
                raise ValueError(
                    "Sphere deformation requires independent actual source/target validity and global embedding."
                )
            values, bounds, _ = _certified_cell_measures(
                atlas.source_mesh,
                atlas.source_geometry,
                absolute_tolerance=measure_absolute_tolerance,
                relative_tolerance=measure_relative_tolerance,
                maximum_work=maximum_measure_work,
                maximum_subcells=maximum_measure_subcells,
                maximum_binomial_terms=maximum_binomial_terms,
            )
            if (
                np.any(~np.isfinite(values))
                or np.any(~np.isfinite(bounds))
                or np.any(bounds < 0)
                or np.any(values <= bounds)
            ):
                raise ValueError(
                    "Sphere physical cell measures require quantitative strictly positive enclosures."
                )
            validities.append(validity)
            embeddings.append(embedding)
            measures.append(values)
            errors.append(bounds)
    occurrences: list[PreparedSphereChartOccurrence] = []
    for entity, path in sorted(source_groups):
        indices = tuple(
            index
            for index, piece in enumerate(pieces)
            if (piece.geometry_entity_id, piece.occurrence_path) == (entity, path)
        )
        source_ids = tuple(
            int(identifier)
            for identifier, group in zip(
                np.asarray(source_atlas.cell_global_ids),
                zip(
                    source_atlas.geometry_entity_ids,
                    source_atlas.occurrence_paths,
                    strict=True,
                ),
                strict=True,
            )
            if group == (entity, path)
        )
        target_ids = tuple(
            int(identifier)
            for identifier, group in zip(
                np.asarray(target_atlas.cell_global_ids),
                zip(
                    target_atlas.geometry_entity_ids,
                    target_atlas.occurrence_paths,
                    strict=True,
                ),
                strict=True,
            )
            if group == (entity, path)
        )
        occurrences.append(
            PreparedSphereChartOccurrence(
                geometry_entity_id=entity,
                occurrence_path=path,
                piece_indices=indices,
                source_cell_global_ids=source_ids,
                target_cell_global_ids=target_ids,
                coverage_status="certified",
            )
        )
    identity = canonical_fingerprint(
        {
            "kind": "prepared-sphere-projective-deformation",
            "source": source_atlas.atlas_id,
            "target": target_atlas.atlas_id,
            "pieces": tuple(piece.piece_id for piece in pieces),
            "source_areas": _tokens(
                tuple((area, Fraction(0)) for area in source_coverage)
            ),
            "target_areas": _tokens(
                tuple((area, Fraction(0)) for area in target_coverage)
            ),
            "measure_values": array_tree_fingerprint(tuple(measures)),
            "measure_errors": array_tree_fingerprint(tuple(errors)),
            "embedding": tuple(value.certificate_id for value in embeddings),
        }
    )
    return PreparedSphereChartDeformation(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        source_mesh=source_atlas.source_mesh,
        target_mesh=target_atlas.source_mesh,
        source_geometry=source_atlas.source_geometry,
        target_geometry=target_atlas.source_geometry,
        source_geometry_id=source_atlas.source_geometry_id,
        target_geometry_id=target_atlas.source_geometry_id,
        source_topology_id=source_atlas.source_topology_id,
        target_topology_id=target_atlas.source_topology_id,
        domain_id=source_atlas.domain_id,
        pieces=tuple(pieces),
        occurrences=tuple(occurrences),
        source_reference_coverage=tuple(source_coverage),
        target_reference_coverage=tuple(target_coverage),
        source_fidelity_bounds=jnp.asarray(source_bounds, dtype=jnp.float64),
        target_fidelity_bounds=jnp.asarray(target_bounds, dtype=jnp.float64),
        source_cell_measures=jnp.asarray(measures[0], dtype=jnp.float64),
        target_cell_measures=jnp.asarray(measures[1], dtype=jnp.float64),
        source_measure_errors=jnp.asarray(errors[0], dtype=jnp.float64),
        target_measure_errors=jnp.asarray(errors[1], dtype=jnp.float64),
        source_validity=validities[0],
        target_validity=validities[1],
        source_embedding=embeddings[0],
        target_embedding=embeddings[1],
        maximum_displacement_bound=maximum_bound,
        candidate_pair_count=int(source_candidates.size),
        work_units=work,
        retained_bytes_upper=retained,
        deformation_id=identity,
    )


__all__ = [
    "PreparedSphereChartPiece",
    "PreparedSphereChartOccurrence",
    "PreparedSphereChartDeformation",
    "SphereGeometryReconstruction",
    "prepare_sphere_chart_deformation",
    "reconstruct_sphere_material_cell_geometry",
]


class SphereGeometryReconstruction(StrictModule, NonTrainableState):
    """Actual degree-preserving radial-source reconstruction, without UV ghost cells."""

    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    geometry: CellGeometrySpec
    vertex_coordinates: Array
    target_atlas: SphereMaterialCellAtlas
    target_validity: CellValidityCertificate
    target_embedding: GlobalEmbeddingCertificate
    cell_global_ids: Array
    cell_patches: Array
    fidelity_bounds: Array
    node_enclosure_bounds: Array
    node_owner_cells: Array
    node_owner_locals: Array
    continuity_residual: float = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    preparation_work_units: int = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    def require_bound(
        self, source: SphereMaterialCellAtlas, target_mesh: CellMesh, /
    ) -> None:
        source.require_bound(
            source.domain,
            source.source_mesh,
            source.source_geometry,
            source.coordinate_contract,
        )
        self.target_atlas.require_bound(
            source.domain, target_mesh, self.geometry, source.coordinate_contract
        )
        if (
            self.source_geometry_id,
            self.source_topology_id,
            self.target_geometry_id,
            self.target_topology_id,
            self.domain_id,
            self.source_id,
            self.source_revision,
        ) != (
            source.source_geometry_id,
            source.source_topology_id,
            cell_geometry_id(self.geometry),
            target_mesh.topology_id,
            source.domain_id,
            source.source_id,
            source.source_revision,
        ):
            raise ValueError(
                "Sphere reconstruction is stale for its actual source, target or material revision."
            )
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
        )
        patches = np.empty(identifiers.size, dtype=np.int32)
        patches[np.asarray(self.target_atlas.physical_rows, dtype=np.int64)] = np.asarray(
            self.target_atlas.patches
        )
        if (
            not np.array_equal(np.asarray(self.cell_global_ids), identifiers)
            or not np.array_equal(np.asarray(self.cell_patches), patches)
            or not np.array_equal(
                np.asarray(self.vertex_coordinates), np.asarray(target_mesh.coordinates)
            )
            or not np.array_equal(
                np.asarray(self.fidelity_bounds),
                _physical_order(
                    self.target_atlas, self.target_atlas.source_fidelity_bounds
                ),
            )
        ):
            raise ValueError(
                "Sphere reconstruction metadata is stale for its actual full-map material cells."
            )
        self.target_validity.require_bound(self.geometry, mesh=target_mesh)
        self.target_embedding.binding.require(target_mesh, self.geometry)
        if (
            not self.target_validity.all_certified
            or self.target_embedding.status != "certified"
        ):
            raise ValueError(
                "Sphere reconstruction lacks actual validity and global embedding."
            )


def reconstruct_sphere_material_cell_geometry(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
    *,
    cell_patches: np.ndarray,
    corner_source_dimensions: np.ndarray,
    corner_source_indices: np.ndarray,
    corner_source_parameters: np.ndarray,
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    corner_source_entity_ids: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: CellGeometryTransitionPolicy,
    coordinate_budget: CoordinateEnclosureBudget,
    certificate_limits: MeshCertificateLimits,
    validity_policy: CellValidityPolicy,
) -> SphereGeometryReconstruction:
    """Reconstruct the original radial source at real target coordinate nodes.

    Original source expressions and coordinate banks remain bound in source_atlas.
    The successor's complete coordinate degree is retained, while its new full
    polynomial map receives independent fidelity, validity and embedding proofs.
    """
    from .._meshcore import charge_native_geometry_queries
    from ..geometry._meshing_domain import _interval_vector_norm, _matrix_norm_upper
    from ..geometry._sphere_material_atlas import (
        _corner_direction,
        _source_image_enclosure,
    )
    from ._cell_geometry_transfer import (
        _place,
        _simplex_blocks,
        _vertex_rows,
        CellGeometryTransitionPolicy,
        nested_geometry_degree,
    )

    source_atlas.require_bound(
        source_atlas.domain,
        source_atlas.source_mesh,
        source_atlas.source_geometry,
        source_atlas.coordinate_contract,
    )
    if (
        not isinstance(policy, CellGeometryTransitionPolicy)
        or policy.reconstruction != "bounded_chart_deformation"
    ):
        raise ValueError(
            "Sphere reconstruction requires the explicit bounded chart-deformation policy."
        )
    if not isinstance(coordinate_budget, CoordinateEnclosureBudget):
        raise TypeError(
            "Sphere reconstruction requires the original coordinate enclosure budget."
        )
    if not isinstance(certificate_limits, MeshCertificateLimits) or not isinstance(
        validity_policy, CellValidityPolicy
    ):
        raise TypeError(
            "Sphere reconstruction requires original certificate and validity controls."
        )
    if not np.isfinite(maximum_fidelity) or maximum_fidelity < 0:
        raise ValueError(
            "Sphere reconstruction requires finite nonnegative authored fidelity."
        )
    degree = nested_geometry_degree(
        source_atlas.source_mesh, source_atlas.source_geometry
    )
    blocks = _simplex_blocks(target_mesh, target_layout, "sphere target")
    if any(
        block.element.cell_kind != "triangle" or block.degree != degree
        for block in blocks
    ):
        raise ValueError(
            "Sphere reconstruction must retain the complete original coordinate degree."
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in target_mesh.blocks]
    )
    count = cells.shape[0]
    patches = np.asarray(cell_patches)
    dimensions, indices = (
        np.asarray(corner_source_dimensions),
        np.asarray(corner_source_indices),
    )
    parameters = np.asarray(corner_source_parameters, dtype=np.float64)
    if (
        patches.shape != (count,)
        or not np.issubdtype(patches.dtype, np.integer)
        or np.any(patches < 0)
        or np.any(patches >= len(source_atlas.domain.patches))
        or dimensions.shape != (count, 3)
        or not np.issubdtype(dimensions.dtype, np.integer)
        or indices.shape != (count, 3)
        or not np.issubdtype(indices.dtype, np.integer)
        or parameters.shape != (count, 3, 2)
        or len(corner_source_occurrence_paths) != count
        or len(corner_source_entity_ids) != count
        or any(
            len(row) != 3
            for row in (*corner_source_occurrence_paths, *corner_source_entity_ids)
        )
    ):
        raise ValueError(
            "Sphere reconstruction strata must follow real target canonical corners."
        )
    domain = source_atlas.domain
    lookup = tuple(
        {
            (path, index): row
            for row, (path, index) in enumerate(zip(paths, ids, strict=True))
        }
        for paths, ids in zip(
            domain.source_occurrences, domain.source_indices, strict=True
        )
    )
    before = coordinate_budget.work_units
    coordinate_budget.reserve(count * 64, count * 2048)
    planned_evaluations = 3 * count + sum(
        block.cell_ids.size * block.element.reference_nodes.shape[0] for block in blocks
    )
    if planned_evaluations > policy.maximum_evaluations:
        raise ValueError(
            "Sphere coordinate reconstruction exceeds original evaluation allowance."
        )
    charge_native_geometry_queries(planned_evaluations)
    directions = np.empty((count, 3, 3), dtype=np.float64)
    for row in range(count):
        patch = int(patches[row])
        for corner in range(3):
            dimension = int(dimensions[row, corner])
            if not 0 <= dimension <= 2:
                raise ValueError("Sphere reconstruction requires actual source strata.")
            index = lookup[dimension].get(
                (corner_source_occurrence_paths[row][corner], int(indices[row, corner]))
            )
            if (
                index is None
                or domain.entity_id(dimension, index)
                != corner_source_entity_ids[row][corner]
            ):
                raise ValueError(
                    "Sphere reconstruction changes source entity or occurrence identity."
                )
            directions[row, corner], _ = _corner_direction(
                domain, patch, dimension, index, parameters[row, corner]
            )
    corner_reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    prescribed_routes: list[_IntArray] | None = []
    for block in blocks:
        nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
        matches = np.all(nodes[:, None, :] == corner_reference[None, :, :], axis=-1)
        if not np.all(np.sum(matches, axis=1) == 1):
            prescribed_routes = None
            break
        prescribed_routes.append(np.asarray(np.argmax(matches, axis=1), dtype=np.int64))
    values: list[_FloatArray] = []
    errors: list[float] = []
    first, evaluations = 0, 3 * count
    if prescribed_routes is not None:
        for block, route in zip(blocks, prescribed_routes, strict=True):
            rows = block.cell_ids.size
            coordinate_budget.reserve(rows * route.size * 3, rows * route.size * 3 * 32)
            values.append(
                np.asarray(target_mesh.coordinates)[cells[first : first + rows][:, route]]
            )
            evaluations += rows * route.size
            first += rows
    else:
        for block in blocks:
            nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
            barycentric = np.column_stack((1 - np.sum(nodes, axis=1), nodes))
            rows = block.cell_ids.size
            coordinate_budget.reserve(
                rows * nodes.shape[0] * 128, rows * nodes.shape[0] * 3 * 32
            )
            physical = np.empty((rows, nodes.shape[0], 3), dtype=np.float64)
            for local in range(rows):
                row = first + local
                patch = int(patches[row])
                slot = int(
                    np.asarray(source_atlas.source_slots)[
                        np.flatnonzero(np.asarray(source_atlas.patches) == patch)[0]
                    ]
                )
                rays = barycentric @ directions[row]
                normalized = rays / np.linalg.norm(rays, axis=-1)[:, None]
                unit = normalized @ np.asarray(source_atlas.axes)[slot].T
                points = np.broadcast_to(
                    np.asarray(source_atlas.centers)[slot], unit.shape
                ).copy()
                for radius in np.asarray(source_atlas.radius_terms)[slot]:
                    points += radius * unit
                physical[local] = (
                    points @ np.asarray(source_atlas.rotations)[slot].T
                    + np.asarray(source_atlas.translations)[slot]
                )
                # Prescribed vertices are actual scientific edit coordinates. Retaining
                # them exactly avoids a second rounding epoch at shared corners.
                for corner, reference in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))):
                    selected = np.flatnonzero(np.all(nodes == reference, axis=1))
                    physical[local, selected] = np.asarray(target_mesh.coordinates)[
                        cells[row, corner]
                    ]
                low, high = _source_image_enclosure(domain.patches[patch].surface, rays)
                point_error = max(
                    sum(
                        (
                            max(
                                abs(
                                    Fraction(float(physical[local, node, axis]))
                                    - Fraction(float(low[node, axis]))
                                ),
                                abs(
                                    Fraction(float(physical[local, node, axis]))
                                    - Fraction(float(high[node, axis]))
                                ),
                            )
                            for axis in range(3)
                        ),
                        Fraction(0),
                    )
                    for node in range(nodes.shape[0])
                )
                # Exact barycentric ray arithmetic separately encloses host dot-product
                # rounding; no fitted radial polynomial is used as the source.
                ray_error = Fraction(0)
                for node in range(nodes.shape[0]):
                    weights = (
                        1
                        - Fraction(float(nodes[node, 0]))
                        - Fraction(float(nodes[node, 1])),
                        Fraction(float(nodes[node, 0])),
                        Fraction(float(nodes[node, 1])),
                    )
                    exact = tuple(
                        sum(
                            (
                                weights[corner]
                                * Fraction(float(directions[row, corner, axis]))
                                for corner in range(3)
                            ),
                            Fraction(0),
                        )
                        for axis in range(3)
                    )
                    ray_error = max(
                        ray_error,
                        sum(
                            (
                                abs(exact[axis] - Fraction(float(rays[node, axis])))
                                for axis in range(3)
                            ),
                            Fraction(0),
                        ),
                    )
                floor = float(np.min(_interval_vector_norm((rays, rays))[0]))
                if Fraction(floor) <= ray_error:
                    raise ValueError(
                        "Sphere coordinate reconstruction has an unresolved radial normalization."
                    )
                radius = sum(
                    (
                        abs(Fraction(float(term)))
                        for term in np.asarray(source_atlas.radius_terms)[slot]
                    ),
                    Fraction(0),
                )
                amplification = (
                    radius
                    * Fraction(_matrix_norm_upper(np.asarray(source_atlas.axes)[slot]))
                    * Fraction(
                        _matrix_norm_upper(np.asarray(source_atlas.rotations)[slot])
                    )
                )
                error = outward(
                    point_error + 6 * amplification * ray_error / Fraction(floor),
                    math.inf,
                )
                errors.append(error)
                evaluations += nodes.shape[0]
            values.append(physical)
            first += rows
    if evaluations > policy.maximum_evaluations:
        raise ValueError(
            "Sphere coordinate reconstruction exceeds original evaluation allowance."
        )
    if evaluations != planned_evaluations:
        raise ValueError(
            "Sphere reconstruction evaluation accounting disagrees with the admitted node batch."
        )
    placed = _place(blocks, values, target_layout.coordinates.shape[0])
    extent = max(1.0, float(np.max(np.abs(placed.coordinates), initial=0.0)))
    if placed.continuity > policy.continuity_tolerance * extent:
        raise ValueError(
            "Original sphere source realizations disagree at shared coordinate nodes."
        )
    elements, routes, _ = target_layout.resolve(target_mesh)
    names = tuple(block.name for block in target_mesh.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, elements, strict=True)),
        dict(zip(names, routes, strict=True)),
        placed.coordinates,
    )
    if target_mesh.periodic_topology is not None:
        geometry = geometry.with_periodic_source(target_mesh)
    with coordinate_budget.activate(), coordinate_budget.temporary_scope():
        atlas = SphereMaterialCellAtlas(
            domain,
            target_mesh,
            geometry,
            source_atlas.coordinate_contract,
            cell_patches=patches,
            corner_source_dimensions=dimensions,
            corner_source_indices=indices,
            corner_source_parameters=parameters,
            corner_source_occurrence_paths=corner_source_occurrence_paths,
            corner_source_entity_ids=corner_source_entity_ids,
            maximum_fidelity=maximum_fidelity,
            maximum_cells=count,
            maximum_work_units=coordinate_budget.maximum_work_units,
            maximum_coordinate_memory_bytes=coordinate_budget.maximum_memory_bytes,
            coordinate_budget=coordinate_budget,
        )
    if prescribed_routes is not None:
        # Corner-only layouts already retain the prescribed native edit points.
        # The fresh atlas above earned their full original-source enclosure;
        # there is no interior barycentric-ray rounding term at these nodes.
        errors.extend(
            float(value)
            for value in _physical_order(atlas, atlas.source_corner_enclosure_bounds)
        )
    with coordinate_budget.activate(), coordinate_budget.temporary_scope():
        validity = certify_cell_geometry_validity(
            geometry, mesh=target_mesh, policy=validity_policy
        )
        embedding = certify_global_embedding(
            target_mesh, geometry, validity, limits=certificate_limits
        )
    if not validity.all_certified or embedding.status != "certified":
        raise ValueError(
            "Sphere reconstruction fails actual full-map validity or global embedding."
        )
    bounds = _physical_order(atlas, atlas.source_fidelity_bounds)
    identity = canonical_fingerprint(
        {
            "kind": "sphere-radial-coordinate-reconstruction",
            "source": source_atlas.atlas_id,
            "target": atlas.atlas_id,
            "policy": policy.policy_id,
            "node_errors": array_tree_fingerprint(np.asarray(errors, dtype=np.float64)),
            "validity": validity.certificate_id,
            "embedding": embedding.certificate_id,
        }
    )
    return SphereGeometryReconstruction(
        source_geometry_id=source_atlas.source_geometry_id,
        target_geometry_id=cell_geometry_id(geometry),
        source_topology_id=source_atlas.source_topology_id,
        target_topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        geometry=geometry,
        vertex_coordinates=jnp.asarray(
            _vertex_rows(target_mesh, blocks, placed.coordinates)
        ),
        target_atlas=atlas,
        target_validity=validity,
        target_embedding=embedding,
        cell_global_ids=jnp.asarray(np.concatenate([block.cell_ids for block in blocks])),
        cell_patches=jnp.asarray(patches),
        fidelity_bounds=jnp.asarray(bounds),
        node_enclosure_bounds=jnp.asarray(np.asarray(errors, dtype=np.float64)),
        node_owner_cells=jnp.asarray(placed.owner_cells),
        node_owner_locals=jnp.asarray(placed.owner_locals),
        continuity_residual=placed.continuity,
        evaluation_count=evaluations,
        preparation_work_units=coordinate_budget.work_units - before,
        reconstruction_id=identity,
    )


def _sphere_piece_geometry(
    mapping: SphereProjectiveReferenceMap,
    vertices: _Triangle,
    coordinate_budget: CoordinateEnclosureBudget | None,
    /,
) -> tuple[SphereProjectiveTriangleBounds, _Triangle, Fraction, Fraction]:
    """Admit the real complete projective piece using its actual operand heights."""
    from ..linalg._hermitian_spectral import (
        _fraction_matrix_profile,
        _reserve_fraction_work,
    )

    if coordinate_budget is not None:
        coordinate_budget.admit_work_bound(138 + 15)
        numerator_bits, denominator_bits = _fraction_matrix_profile(
            (mapping.exact_coefficients, vertices), coordinate_budget
        )
        # Homogeneous three-term sums, two reference quotients, degree-three
        # Jacobian quotients and the two three-edge areas have bounded scalar
        # operation counts (61 + 51 + 26). Each rational denominator is a
        # product of at most the 15 original denominators before cancellation.
        _reserve_fraction_work(
            coordinate_budget, 138, 256, 32 * (numerator_bits + 15 * denominator_bits + 1)
        )
    bounds = mapping.certify_triangle(vertices)
    target: _Triangle = (
        _project(mapping, vertices[0]),
        _project(mapping, vertices[1]),
        _project(mapping, vertices[2]),
    )
    return bounds, target, _area(vertices), _area(target)


def _sphere_full_coordinate_expressions(
    atlas: SphereMaterialCellAtlas,
    coordinate_budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[Expression, ...], ...]:
    """Retain the actual immutable coefficient maps in scientific physical-cell order."""
    from ._cell_geometry_transfer import (
        _mapped_coordinate_expressions,
        _mapped_geometry_cells,
    )

    with coordinate_budget.activate():
        cells = _mapped_geometry_cells(atlas.source_mesh, atlas.source_geometry)
        expressions: list[tuple[Expression, ...]] = []
        for element, bank in cells:
            with coordinate_budget.temporary_scope():
                values = _mapped_coordinate_expressions(element, bank)
            expressions.append(values)
    return tuple(expressions)


def _sphere_displacement_map(
    source: tuple[Expression, ...],
    target: tuple[Expression, ...],
    mapping: SphereProjectiveReferenceMap,
    coordinate_budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Expression, ...]:
    """Subtract the complete coordinate maps once in the original source chart."""
    from ._coordinate_enclosure import (
        add,
        expression_add,
        expression_scale,
        multiply,
        RationalPolynomial,
        scale,
    )

    with coordinate_budget.activate():
        composition = ExpressionComposition(mapping.reference_expressions())
        differences: list[Expression] = []
        for old, new in zip(source, target, strict=True):
            if (
                composition.composition is not None
                or isinstance(old, RationalPolynomial)
                or isinstance(new, RationalPolynomial)
            ):
                difference = expression_add(composition(new), expression_scale(old, -1))
            else:
                numerator, denominator = composition._polynomial(new)
                numerator = add(numerator, scale(multiply(old, denominator), -1))
                # The original homogeneous denominator is already certified by
                # the projective owner. An enclosure needs its exact ratio, not
                # an expensive removable-factor search on every coordinate.
                difference = (
                    RationalPolynomial(numerator, denominator) if numerator else {}
                )
            differences.append(difference)
        return tuple(differences)


def _sphere_full_map_displacement(
    displacement_map: tuple[Expression, ...],
    vertices: _Triangle,
    coordinate_budget: CoordinateEnclosureBudget,
    /,
) -> float:
    """Enclose the complete exact displacement on an entire material piece."""
    from ..geometry._sphere_material_atlas import _float_enclosure
    from ._coordinate_enclosure import (
        add,
        axes,
        constant,
        expression_bernstein_coefficients,
        RationalPolynomial,
        scale,
    )

    with coordinate_budget.activate(), coordinate_budget.temporary_scope():
        coordinate_budget.reserve(len(displacement_map))
        if all(
            not isinstance(value, RationalPolynomial) and not value
            for value in displacement_map
        ):
            # A canonical empty coefficient support is the zero polynomial on
            # every affine subpiece, independently of mesh or chart identity.
            return 0.0
        variables = axes(2)
        reference = tuple(
            add(
                constant(vertices[0][axis], 2),
                add(
                    scale(variables[0], vertices[1][axis] - vertices[0][axis]),
                    scale(variables[1], vertices[2][axis] - vertices[0][axis]),
                ),
            )
            for axis in range(2)
        )
        # Exact affine substitution commutes with subtraction. The source-chart
        # difference retains both full coordinate maps and is shared by all
        # triangles of this projective overlap, including exact cancellation.
        at_source = ExpressionComposition(reference)
        bound = Fraction(0)
        for difference in displacement_map:
            if not isinstance(difference, RationalPolynomial) and not difference:
                continue
            if isinstance(difference, RationalPolynomial):
                polynomial_action = at_source.composition
                if polynomial_action is None:
                    raise RuntimeError(
                        "A material piece lost its affine polynomial action."
                    )
                image = RationalPolynomial(
                    polynomial_action(difference.numerator),
                    polynomial_action(difference.denominator),
                )
            else:
                image = at_source(difference)
            controls = expression_bernstein_coefficients(image, "simplex", 2)
            bound += max(abs(value) for value in controls)
        # Exact coefficient cancellation, not an identity-case bypass. The
        # canonical realization enclosure preserves a genuinely exact zero.
        return _float_enclosure(bound)[1]
