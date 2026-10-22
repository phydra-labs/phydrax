#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Store chart correspondence data and validate its retained scientific bindings."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..typing import checked, ConvertibleToArray
from ..linalg._hermitian_spectral import reserve_fraction_work
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_transition_types import MaterialChartDeformation
from ._cell_geometry_validity import cell_geometry_id, CellValidityCertificate
from ._cell_mesh import CellMesh
from ._chart_geometry import area
from ._coordinate_budget import CoordinateEnclosureBudget


type _ExactPoint = tuple[Fraction, Fraction]
type _ExactTriangle = tuple[_ExactPoint, _ExactPoint, _ExactPoint]
type _IntArray = NDArray[np.int64]


class SurfaceChartResourceError(RuntimeError):
    """refused chart preparation, preserving owner resource evidence."""

    def __init__(self, message: str, counts: tuple[tuple[str, int, int], ...], /) -> None:
        self.resource_counts = counts
        super().__init__(message)


class SurfaceChartWitness(StrictModule, NonTrainableState):
    """Scientific cell-ID keyed affine reference-to--chart binding."""

    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    cell_global_ids: Array
    patches: Array
    charts: Array
    geometry_entity_ids: tuple[str, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_id: str,
        topology_id: str,
        domain_id: str,
        cell_global_ids: ConvertibleToArray,
        patches: ConvertibleToArray,
        charts: ConvertibleToArray,
        geometry_entity_ids: tuple[str, ...],
        occurrence_paths: tuple[tuple[str, ...], ...],
    ) -> None:
        ids, patch = (np.asarray(cell_global_ids), np.asarray(patches))
        uv = np.asarray(charts, dtype=np.float64)
        if (
            ids.ndim != 1
            or not ids.size
            or (not np.issubdtype(ids.dtype, np.integer))
            or np.any(ids < 0)
            or (np.unique(ids).size != ids.size)
        ):
            raise ValueError(
                "surface_chart.init.binding must satisfy the declared geometry and cell identities; got binding cells must be unique nonnegative scientific integer IDs."
            )
        if (
            patch.shape != ids.shape
            or not np.issubdtype(patch.dtype, np.integer)
            or np.any(patch < 0)
            or (uv.shape != (ids.size, 3, 2))
            or (not np.all(np.isfinite(uv)))
        ):
            raise ValueError(
                "surface_chart.init.binding must satisfy the declared geometry and cell identities; got binding patches and finite triangle charts must align with cell IDs."
            )
        if (
            len(geometry_entity_ids) != ids.size
            or len(occurrence_paths) != ids.size
            or any(
                (
                    not isinstance(entity, str) or not entity
                    for entity in geometry_entity_ids
                )
            )
            or any(
                (
                    not isinstance(path, tuple)
                    or any((not isinstance(part, str) or not part for part in path))
                    for path in occurrence_paths
                )
            )
        ):
            raise ValueError(
                "surface_chart.init.binding must satisfy the declared geometry and cell identities; got Witnesses require explicit entity IDs and occurrence paths."
            )
        if any(
            (
                not isinstance(value, str) or not value
                for value in (geometry_id, topology_id, domain_id)
            )
        ):
            raise ValueError(
                "surface_chart.init.binding must satisfy the declared geometry and cell identities; got binding geometry, topology and domain IDs must be explicit."
            )
        order = np.argsort(ids, kind="stable")
        self.geometry_id, self.topology_id, self.domain_id = (
            geometry_id,
            topology_id,
            domain_id,
        )
        self.cell_global_ids = jnp.asarray(ids[order], dtype=jnp.int64)
        self.patches = jnp.asarray(patch[order], dtype=jnp.int32)
        self.charts = jnp.asarray(uv[order], dtype=jnp.float64)
        self.geometry_entity_ids = tuple((geometry_entity_ids[row] for row in order))
        self.occurrence_paths = tuple((occurrence_paths[row] for row in order))
        self.witness_id = canonical_fingerprint(
            {
                "kind": "surface-chart-witness",
                "geometry": geometry_id,
                "topology": topology_id,
                "domain": domain_id,
                "cells": array_tree_fingerprint(self.cell_global_ids),
                "patches": array_tree_fingerprint(self.patches),
                "charts": array_tree_fingerprint(self.charts),
                "entities": self.geometry_entity_ids,
                "occurrences": self.occurrence_paths,
            }
        )


class PreparedSurfaceChartPiece(StrictModule, NonTrainableState):
    """One native material intersection in both physical references."""

    source_row: int = eqx.field(static=True)
    target_row: int = eqx.field(static=True)
    source_cell_global_id: int = eqx.field(static=True)
    target_cell_global_id: int = eqx.field(static=True)
    exact_source_reference_vertices: _ExactTriangle = eqx.field(static=True)
    exact_target_reference_vertices: _ExactTriangle = eqx.field(static=True)
    source_reference_area: Fraction = eqx.field(static=True)
    target_reference_area: Fraction = eqx.field(static=True)
    native_boundary_labels: tuple[int, ...] = eqx.field(static=True)
    exact_native_source_polygon: tuple[_ExactPoint, ...] = eqx.field(static=True)
    native_polygon_vertex_indices: tuple[int, int, int] = eqx.field(static=True)
    piece_id: str = eqx.field(static=True)


class PreparedSurfaceChartOccurrence(StrictModule, NonTrainableState):
    """native material overlap, independent of the physical UV map."""

    patch: int = eqx.field(static=True)
    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    source_rows: Array
    target_rows: Array
    exact_source_material_charts: tuple[_ExactTriangle, ...] = eqx.field(static=True)
    exact_target_material_charts: tuple[_ExactTriangle, ...] = eqx.field(static=True)
    pieces: tuple[PreparedSurfaceChartPiece, ...]
    root_pieces: tuple[PreparedSurfaceChartPiece, ...]
    material_root_witness_id: str = eqx.field(static=True)
    displacement_bounds: Array
    source_domain_coverage: str = eqx.field(static=True)
    target_domain_coverage: str = eqx.field(static=True)


class PreparedSurfaceChartDeformation(MaterialChartDeformation, NonTrainableState):
    """whole bounded old-to-target certificate, independently guarded in 3D.

    Physical measures, errors and fidelity arrays use concatenated physical mesh
    rows. Occurrence row maps bridge that order to scientific material
    cells. Displacement bounds belong to individual native overlap pieces."""

    source_witness: SurfaceChartWitness
    target_witness: SurfaceChartWitness
    material_root_witness: SurfaceChartWitness
    material_root_chart_cover: MeshingDomainBoundarySource | None
    target_chart_cover: MeshingDomainBoundarySource | None
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    occurrences: tuple[PreparedSurfaceChartOccurrence, ...]
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    source_fidelity_bounds: Array
    target_fidelity_bounds: Array
    source_polynomial_chord_bounds: Array
    source_corner_enclosure_bounds: Array
    source_continuous_chord_bounds: Array
    source_cell_measures: Array
    target_cell_measures: Array
    source_measure_errors: Array
    target_measure_errors: Array
    source_validity: CellValidityCertificate
    target_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    maximum_displacement_bound: float = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    deformation_id: str = eqx.field(static=True)

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_domain_coverage(self) -> str:
        return (
            "certified"
            if all(
                (item.source_domain_coverage == "certified" for item in self.occurrences)
            )
            else "unresolved"
        )

    @property
    def target_domain_coverage(self) -> str:
        return (
            "certified"
            if all(
                (item.target_domain_coverage == "certified" for item in self.occurrences)
            )
            else "unresolved"
        )

    def material_charts(self, *, target: bool) -> NDArray[np.object_]:
        """material bank in endpoint scientific physical row order."""
        mesh = self.target_mesh if target else self.source_mesh
        result = np.empty((chart_cell_ids(mesh).size, 3, 2), dtype=object)
        seen = np.zeros(result.shape[0], dtype=np.bool_)
        for occurrence in self.occurrences:
            rows = np.asarray(
                occurrence.target_rows if target else occurrence.source_rows,
                dtype=np.int64,
            )
            charts = (
                occurrence.exact_target_material_charts
                if target
                else occurrence.exact_source_material_charts
            )
            if len(charts) != rows.size or np.any(seen[rows]):
                raise ValueError(
                    "surface_chart.material_charts.binding must satisfy the declared geometry and cell identities; got Material binding has duplicated or missing scientific cell rows."
                )
            result[rows] = np.asarray(charts, dtype=object)
            seen[rows] = True
        if not np.all(seen):
            raise ValueError(
                "surface_chart.material_charts.binding must satisfy the declared geometry and cell identities; got Material binding does not cover its scientific physical endpoint."
            )
        return result

    @checked
    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
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
                "surface_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Surface deformation is stale for these meshes or coordinate maps."
            )
        if (
            self.source_domain_coverage != "certified"
            or self.target_domain_coverage != "certified"
        ):
            raise ValueError(
                "surface_chart.require_bound.pieces must satisfy the material reference contract; got Surface deformation lacks whole UV coverage."
            )
        if (
            not self.occurrences
            or {item.patch for item in self.occurrences}
            != set(map(int, np.asarray(self.source_witness.patches)))
            or len({item.patch for item in self.occurrences}) != len(self.occurrences)
        ):
            raise ValueError(
                "surface_chart.require_bound.pieces must satisfy the material reference contract; got Surface deformation omits or merges declared patch occurrences."
            )
        for item in self.occurrences:
            if item.material_root_witness_id != self.material_root_witness.witness_id:
                raise ValueError(
                    "surface_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Material occurrence changes its source root binding."
                )
            for mesh, witness, rows, charts in (
                (
                    source_mesh,
                    self.source_witness,
                    item.source_rows,
                    item.exact_source_material_charts,
                ),
                (
                    target_mesh,
                    self.target_witness,
                    item.target_rows,
                    item.exact_target_material_charts,
                ),
            ):
                row_map = np.asarray(rows)
                physical_ids = chart_cell_ids(mesh)
                if (
                    row_map.shape != (len(charts),)
                    or not np.issubdtype(row_map.dtype, np.integer)
                    or np.any(row_map < 0)
                    or np.any(row_map >= physical_ids.size)
                    or (np.unique(row_map).size != row_map.size)
                ):
                    raise ValueError(
                        "surface_chart.require_bound.pieces must satisfy the material reference contract; got material row maps alter their scientific physical cell axes."
                    )
                ids = physical_ids[row_map]
                declared = np.asarray(witness.cell_global_ids)
                witness_rows = np.searchsorted(declared, ids)
                if (
                    np.any(witness_rows >= declared.size)
                    or not np.array_equal(declared[witness_rows], ids)
                    or np.any(np.asarray(witness.patches)[witness_rows] != item.patch)
                    or any(
                        (
                            witness.geometry_entity_ids[row] != item.geometry_entity_id
                            or witness.occurrence_paths[row] != item.occurrence_path
                            for row in witness_rows
                        )
                    )
                ):
                    raise ValueError(
                        "surface_chart.require_bound.pieces must satisfy the material reference contract; got material cells alter an source entity or occurrence."
                    )
            root_selection = np.flatnonzero(
                np.asarray(self.material_root_witness.patches) == item.patch
            )
            root_charts = exact_chart_bank(
                np.asarray(self.material_root_witness.charts)[root_selection],
                root_selection.size,
                budget=budget,
            )
            root_ids = np.asarray(self.material_root_witness.cell_global_ids)[
                root_selection
            ]
            source_ids = chart_cell_ids(source_mesh)[
                np.asarray(item.source_rows, dtype=np.int64)
            ]
            if item.root_pieces:
                require_material_partition(
                    item.root_pieces,
                    root_charts,
                    item.exact_source_material_charts,
                    root_ids,
                    source_ids,
                    budget=budget,
                )
            elif root_charts != item.exact_source_material_charts or not np.array_equal(
                root_ids, source_ids
            ):
                raise ValueError(
                    "surface_chart.require_bound.pieces must satisfy the material reference contract; got Material source loses its source root ancestry partition."
                )
            require_material_partition(
                item.pieces,
                item.exact_source_material_charts,
                item.exact_target_material_charts,
                source_ids,
                chart_cell_ids(target_mesh)[np.asarray(item.target_rows, dtype=np.int64)],
                budget=budget,
            )
        for mesh, target in ((source_mesh, False), (target_mesh, True)):
            rows = np.concatenate(
                [
                    np.asarray(
                        item.target_rows if target else item.source_rows, dtype=np.int64
                    )
                    for item in self.occurrences
                ]
            )
            if not np.array_equal(
                np.sort(rows), np.arange(chart_cell_ids(mesh).size, dtype=np.int64)
            ):
                raise ValueError(
                    "surface_chart.require_bound.pieces must satisfy the material reference contract; got material occurrences omit or duplicate scientific physical cells."
                )
        for mesh, geometry, validity, embedding in (
            (source_mesh, source_geometry, self.source_validity, self.source_embedding),
            (target_mesh, target_geometry, self.target_validity, self.target_embedding),
        ):
            validity.require_bound(geometry, mesh=mesh)
            embedding.binding.require(mesh, geometry)
            if not validity.all_certified or embedding.status != "certified":
                raise ValueError(
                    "surface_chart.require_bound.pieces must satisfy the material reference contract; got Surface deformation requires valid globally embedded geometry."
                )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PreparedSurfaceChartDeformation],
        value: PreparedSurfaceChartDeformation,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PreparedSurfaceChartDeformation:
        """Re-prove retained partitions on the supplied actual caller ledger."""
        value.require_bound(
            value.source_mesh,
            value.source_geometry,
            value.target_mesh,
            value.target_geometry,
            budget=budget,
        )
        return value


def chart_cell_ids(mesh: CellMesh) -> _IntArray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def material_point(chart: _ExactTriangle, point: _ExactPoint) -> _ExactPoint:
    x, y = point
    values = tuple(
        (
            (1 - x - y) * chart[0][axis] + x * chart[1][axis] + y * chart[2][axis]
            for axis in range(2)
        )
    )
    return (values[0], values[1])


def material_reference(chart: _ExactTriangle, point: _ExactPoint) -> _ExactPoint:
    a = tuple((chart[1][axis] - chart[0][axis] for axis in range(2)))
    b = tuple((chart[2][axis] - chart[0][axis] for axis in range(2)))
    delta = tuple((point[axis] - chart[0][axis] for axis in range(2)))
    determinant = a[0] * b[1] - a[1] * b[0]
    if not determinant:
        raise ValueError(
            "surface_chart.material_reference.reference_map must satisfy the regular reference-map contract; got A source material cell is singular."
        )
    return (
        (delta[0] * b[1] - delta[1] * b[0]) / determinant,
        (a[0] * delta[1] - a[1] * delta[0]) / determinant,
    )


def material_piece_id(
    source_id: int,
    target_id: int,
    source_vertices: _ExactTriangle,
    target_vertices: _ExactTriangle,
    polygon: tuple[_ExactPoint, ...],
    labels: tuple[int, ...],
    slots: tuple[int, int, int],
) -> str:

    def tokens(
        points: tuple[_ExactPoint, ...],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(
            (
                tuple(((value.numerator, value.denominator) for value in point))
                for point in points
            )
        )

    return canonical_fingerprint(
        {
            "kind": "native-exact-surface-material-piece",
            "source": source_id,
            "target": target_id,
            "source_reference": tokens(source_vertices),
            "target_reference": tokens(target_vertices),
            "native_polygon": tokens(polygon),
            "supports": labels,
            "fan": slots,
        }
    )


def require_material_partition(
    pieces: tuple[PreparedSurfaceChartPiece, ...],
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    source_ids: _IntArray,
    target_ids: _IntArray,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> None:
    """Authenticate native supporting faces and both whole partitions."""
    source_areas, target_areas = (
        [Fraction(0)] * len(source),
        [Fraction(0)] * len(target),
    )
    fans: set[tuple[int, int, tuple[int, int, int]]] = set()
    for piece in pieces:
        old, new = (piece.source_row, piece.target_row)
        if (
            not 0 <= old < len(source)
            or not 0 <= new < len(target)
            or piece.source_cell_global_id != int(source_ids[old])
            or (piece.target_cell_global_id != int(target_ids[new]))
        ):
            raise ValueError(
                "surface_chart.require_material_partition.binding must satisfy the declared geometry and cell identities; got material piece changes its source scientific cell binding."
            )
        # Admit term-visit upper bounds for the three reference solves,
        # supporting planes, polygon predicates, fan images and area sums.
        # The bit bound includes division and products of exact input ratios.
        values = (*source[old], *target[new], *piece.exact_native_source_polygon)
        bits = max(
            (
                max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                for point in values for value in point
            ),
            default=1,
        )
        vertices = len(piece.exact_native_source_polygon)
        reserve_fraction_work(
            budget, 145 + 47 * vertices, 96 + 24 * vertices, 32 * bits + 64
        )
        mapped = (
            material_reference(target[new], source[old][0]),
            material_reference(target[new], source[old][1]),
            material_reference(target[new], source[old][2]),
        )
        origin = mapped[0]
        first = tuple((mapped[1][axis] - origin[axis] for axis in range(2)))
        second = tuple((mapped[2][axis] - origin[axis] for axis in range(2)))
        planes = (
            (Fraction(-1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(-1), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(1)),
            (-first[0], -second[0], origin[0]),
            (-first[1], -second[1], origin[1]),
            (first[0] + first[1], second[0] + second[1], 1 - origin[0] - origin[1]),
        )
        polygon = piece.exact_native_source_polygon
        labels = piece.native_boundary_labels
        if (
            len(polygon) < 3
            or len(set(polygon)) != len(polygon)
            or len(labels) != len(polygon)
            or (area(polygon) <= 0)
        ):
            raise ValueError(
                "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got Native material polygon loses its positive distinct-vertex construction."
            )
        for slot, label in enumerate(labels):
            if not 0 <= label < len(planes):
                raise ValueError(
                    "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got Native material boundary names an absent source supporting face."
                )
            point, next_point = (polygon[slot], polygon[(slot + 1) % len(polygon)])
            previous = polygon[slot - 1]
            if (point[0] - previous[0]) * (next_point[1] - point[1]) - (
                point[1] - previous[1]
            ) * (next_point[0] - point[0]) <= 0:
                raise ValueError(
                    "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got The native material polygon is not the strictly convex source intersection."
                )
            if any((a * point[0] + b * point[1] > c for a, b, c in planes)):
                raise ValueError(
                    "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got Native material polygon leaves an source rational halfplane."
                )
            a, b, c = planes[label]
            if any((a * p[0] + b * p[1] != c for p in (point, next_point))):
                raise ValueError(
                    "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got A native material boundary changes its source rational supporting face."
                )
        slots = piece.native_polygon_vertex_indices
        if (
            slots[0] != 0
            or slots[2] != slots[1] + 1
            or (not 1 <= slots[1] < len(polygon) - 1)
        ):
            raise ValueError(
                "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got A native material piece loses its ordered polygon fan partition."
            )
        key = (old, new, slots)
        if key in fans:
            raise ValueError(
                "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got A native material fan piece is duplicated."
            )
        fans.add(key)
        old_vertices = (polygon[slots[0]], polygon[slots[1]], polygon[slots[2]])
        new_vertices = (
            material_point(mapped, old_vertices[0]),
            material_point(mapped, old_vertices[1]),
            material_point(mapped, old_vertices[2]),
        )
        if (
            old_vertices != piece.exact_source_reference_vertices
            or new_vertices != piece.exact_target_reference_vertices
        ):
            raise ValueError(
                "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got Native material piece alters its source/target reference map."
            )
        old_area, new_area = (abs(area(old_vertices)), abs(area(new_vertices)))
        if (
            not old_area
            or not new_area
            or old_area != piece.source_reference_area
            or (new_area != piece.target_reference_area)
        ):
            raise ValueError(
                "surface_chart.require_material_partition.measures must satisfy the positive physical measure contract; got Native material piece changes its positive reference areas."
            )
        if piece.piece_id != material_piece_id(
            piece.source_cell_global_id,
            piece.target_cell_global_id,
            old_vertices,
            new_vertices,
            polygon,
            labels,
            slots,
        ):
            raise ValueError(
                "surface_chart.require_material_partition.binding must satisfy the declared geometry and cell identities; got A native material piece changes its source scientific identity."
            )
        source_areas[old] += old_area
        target_areas[new] += new_area
    if any((area != Fraction(1, 2) for area in (*source_areas, *target_areas))):
        raise ValueError(
            "surface_chart.require_material_partition.pieces must satisfy the material reference contract; got native material pieces lack whole source reference partitions."
        )


def exact_chart_bank(
    values: ConvertibleToArray, count: int,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[_ExactTriangle, ...]:
    bank = np.asarray(values, dtype=object)
    if bank.shape != (count, 3, 2):
        raise ValueError(
            "surface_chart.exact_chart_bank.binding must satisfy the declared geometry and cell identities; got material charts must align with their scientific cell binding."
        )
    charts: list[_ExactTriangle] = []
    for triangle in bank:
        points: list[_ExactPoint] = []
        for point in triangle:
            exact = cast(
                _ExactPoint,
                tuple(
                    (
                        value if isinstance(value, Fraction) else Fraction(float(value))
                        for value in point
                    )
                ),
            )
            points.append(exact)
        charts.append((points[0], points[1], points[2]))
    bits = max(
        (
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            for chart in charts for point in chart for value in point
        ),
        default=1,
    )
    reserve_fraction_work(budget, 13 * len(charts), 13 * len(charts), 8 * bits + 8)
    if any((not area(chart) for chart in charts)):
        raise ValueError(
            "surface_chart.exact_chart_bank.reference_map must satisfy the regular reference-map contract; got A material chart is singular."
        )
    return tuple(charts)
