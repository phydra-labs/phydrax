#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Accepted bounded surface-map deformation through exact material correspondence.

Native rational supporting faces partition retained original material charts.
Physical UV/source maps and nonlinear trim-ribbon coverage remain independent;
no rounded material carrier or sampled map is a completeness premise.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from typing import cast, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import charge_native_geometry_queries, exact_orient2d
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._mesh_certificates import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
    MeshCertificateBinding,
    MeshCertificateLimits,
    SourceFidelityCertificate,
)
from ..geometry._meshing_domain import (
    _verify_chart_chain,
    MeshingDomain,
    MeshingDomainBoundarySource,
    PatchCurveUse,
    PatchPoleUse,
)
from ..geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
)
from ..geometry.brep._patches import LineCurve
from ..typing import ConvertibleToArray
from ._cell_geometry import (
    _require_scalar_coordinate_element,
    BarycentricCellGeometryElement,
    CellGeometryElement,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ._cell_mesh import CellBlock, CellMesh
from ._coordinate_enclosure import (
    add,
    axes,
    constant,
    coordinate_expressions,
    expression_add,
    expression_bernstein_coefficients,
    expression_evaluate,
    expression_node_count,
    expression_scale,
    outward,
    scale,
)


if TYPE_CHECKING:
    from ._cell_geometry_transfer import SurfaceGeometryReconstruction
    from ._coordinate_enclosure import Polynomial
    from .fem._reference import FiniteElementSpec

type _FloatArray = NDArray[np.float64]
type _ExactPoint = tuple[Fraction, Fraction]
type _ExactTriangle = tuple[_ExactPoint, _ExactPoint, _ExactPoint]
type _IntArray = NDArray[np.int64]


class SurfaceChartResourceError(RuntimeError):
    """Actual refused chart preparation, preserving owner resource evidence."""

    def __init__(self, message: str, counts: tuple[tuple[str, int, int], ...], /) -> None:
        self.resource_counts = counts
        super().__init__(message)


class SurfaceChartWitness(StrictModule, NonTrainableState):
    """Scientific cell-ID keyed affine reference-to-authoritative-chart witness."""

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
        ids, patch = np.asarray(cell_global_ids), np.asarray(patches)
        uv = np.asarray(charts, dtype=np.float64)
        if (
            ids.ndim != 1
            or not ids.size
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
        ):
            raise ValueError(
                "Witness cells must be unique nonnegative scientific integer IDs."
            )
        if (
            patch.shape != ids.shape
            or not np.issubdtype(patch.dtype, np.integer)
            or np.any(patch < 0)
            or uv.shape != (ids.size, 3, 2)
            or not np.all(np.isfinite(uv))
        ):
            raise ValueError(
                "Witness patches and finite triangle charts must align with cell IDs."
            )
        if (
            len(geometry_entity_ids) != ids.size
            or len(occurrence_paths) != ids.size
            or any(
                not isinstance(entity, str) or not entity
                for entity in geometry_entity_ids
            )
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(part, str) or not part for part in path)
                for path in occurrence_paths
            )
        ):
            raise ValueError(
                "Witnesses require explicit entity IDs and occurrence paths."
            )
        if any(
            not isinstance(value, str) or not value
            for value in (geometry_id, topology_id, domain_id)
        ):
            raise ValueError(
                "Witness geometry, topology and domain IDs must be explicit."
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
        self.geometry_entity_ids = tuple(geometry_entity_ids[row] for row in order)
        self.occurrence_paths = tuple(occurrence_paths[row] for row in order)
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
    """One exact native material intersection in both physical references."""

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
    """Exact native material overlap, independent of the physical UV map."""

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


class PreparedSurfaceChartDeformation(StrictModule, NonTrainableState):
    """Complete bounded old-to-target certificate, independently guarded in 3D.

    Physical measures, errors and fidelity arrays use concatenated physical mesh
    rows. Occurrence row maps bridge that order to scientific exact material
    cells. Displacement bounds belong to individual exact native overlap pieces.
    """

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
                item.source_domain_coverage == "certified" for item in self.occurrences
            )
            else "unresolved"
        )

    @property
    def target_domain_coverage(self) -> str:
        return (
            "certified"
            if all(
                item.target_domain_coverage == "certified" for item in self.occurrences
            )
            else "unresolved"
        )

    def material_charts(self, *, target: bool) -> NDArray[np.object_]:
        """Exact material bank in actual endpoint scientific physical row order."""
        mesh = self.target_mesh if target else self.source_mesh
        result = np.empty((_cell_ids(mesh).size, 3, 2), dtype=object)
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
                    "Material witness has duplicated or missing scientific cell rows."
                )
            result[rows] = np.asarray(charts, dtype=object)
            seen[rows] = True
        if not np.all(seen):
            raise ValueError(
                "Material witness does not cover its scientific physical endpoint."
            )
        return result

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
                "Surface deformation is stale for these meshes or coordinate maps."
            )
        if (
            self.source_domain_coverage != "certified"
            or self.target_domain_coverage != "certified"
        ):
            raise ValueError(
                "Surface deformation lacks complete authoritative UV coverage."
            )
        if (
            not self.occurrences
            or {item.patch for item in self.occurrences}
            != set(map(int, np.asarray(self.source_witness.patches)))
            or len({item.patch for item in self.occurrences}) != len(self.occurrences)
        ):
            raise ValueError(
                "Surface deformation omits or merges declared patch occurrences."
            )
        for item in self.occurrences:
            if item.material_root_witness_id != self.material_root_witness.witness_id:
                raise ValueError("Material occurrence changes its original root witness.")
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
                physical_ids = _cell_ids(mesh)
                if (
                    row_map.shape != (len(charts),)
                    or not np.issubdtype(row_map.dtype, np.integer)
                    or np.any(row_map < 0)
                    or np.any(row_map >= physical_ids.size)
                    or np.unique(row_map).size != row_map.size
                ):
                    raise ValueError(
                        "Exact material row maps alter their scientific physical cell axes."
                    )
                ids = physical_ids[row_map]
                declared = np.asarray(witness.cell_global_ids)
                witness_rows = np.searchsorted(declared, ids)
                if (
                    np.any(witness_rows >= declared.size)
                    or not np.array_equal(declared[witness_rows], ids)
                    or np.any(np.asarray(witness.patches)[witness_rows] != item.patch)
                    or any(
                        witness.geometry_entity_ids[row] != item.geometry_entity_id
                        or witness.occurrence_paths[row] != item.occurrence_path
                        for row in witness_rows
                    )
                ):
                    raise ValueError(
                        "Exact material cells alter an original source entity or occurrence."
                    )
            root_selection = np.flatnonzero(
                np.asarray(self.material_root_witness.patches) == item.patch
            )
            root_charts = _exact_chart_bank(
                np.asarray(self.material_root_witness.charts)[root_selection],
                root_selection.size,
            )
            root_ids = np.asarray(self.material_root_witness.cell_global_ids)[
                root_selection
            ]
            source_ids = _cell_ids(source_mesh)[
                np.asarray(item.source_rows, dtype=np.int64)
            ]
            if item.root_pieces:
                _require_material_partition(
                    item.root_pieces,
                    root_charts,
                    item.exact_source_material_charts,
                    root_ids,
                    source_ids,
                )
            elif root_charts != item.exact_source_material_charts or not np.array_equal(
                root_ids, source_ids
            ):
                raise ValueError(
                    "Material source loses its original root ancestry partition."
                )
            _require_material_partition(
                item.pieces,
                item.exact_source_material_charts,
                item.exact_target_material_charts,
                source_ids,
                _cell_ids(target_mesh)[np.asarray(item.target_rows, dtype=np.int64)],
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
                np.sort(rows), np.arange(_cell_ids(mesh).size, dtype=np.int64)
            ):
                raise ValueError(
                    "Exact material occurrences omit or duplicate scientific physical cells."
                )
        for mesh, geometry, validity, embedding in (
            (source_mesh, source_geometry, self.source_validity, self.source_embedding),
            (target_mesh, target_geometry, self.target_validity, self.target_embedding),
        ):
            validity.require_bound(geometry, mesh=mesh)
            embedding.binding.require(mesh, geometry)
            if not validity.all_certified or embedding.status != "certified":
                raise ValueError(
                    "Surface deformation requires valid globally embedded geometry."
                )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _coordinate_elements(
    elements: tuple[CellGeometryElement, ...],
) -> tuple[
    FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    ...,
]:
    """Resolve the actual canonical polynomial/rational scalar source owners."""
    from .fem._reference import FiniteElementSpec

    resolved = []
    for candidate in elements:
        element = _require_scalar_coordinate_element(candidate, "Surface coordinates")
        if element.cell_kind != "triangle" or element.topological_dimension != 2:
            raise ValueError(
                "Surface coordinate owners must use actual triangle reference cells."
            )
        if isinstance(element, FiniteElementSpec) and (
            element.conformity != "H1"
            or element.value_shape
            or element.mapping != "identity"
        ):
            raise ValueError(
                "Surface coordinates require scalar identity-mapped H1 elements."
            )
        resolved.append(element)
    return tuple(resolved)


def _cell_ids(mesh: CellMesh) -> _IntArray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def _same_periodic_chart_use(
    domain: MeshingDomain,
    patch: int,
    first: tuple[float, ...],
    second: tuple[float, ...],
    /,
) -> bool:
    """Compare only exact authored deck shifts, never physical coincidences."""
    for a, b, period in zip(
        first, second, domain.patches[patch].surface.periods, strict=True
    ):
        difference = Fraction(b) - Fraction(a)
        if not difference:
            continue
        if period is None or (difference / Fraction(period)).denominator != 1:
            return False
    return True


def _witness_order(
    witness: SurfaceChartWitness,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
) -> _IntArray:
    if not isinstance(witness, SurfaceChartWitness) or (
        witness.geometry_id,
        witness.topology_id,
        witness.domain_id,
    ) != (cell_geometry_id(geometry), mesh.topology_id, domain.domain_id):
        raise ValueError("Chart witness is stale for the geometry, topology or domain.")
    ids = _cell_ids(mesh)
    declared = np.asarray(witness.cell_global_ids)
    if not np.array_equal(np.sort(ids), declared):
        raise ValueError(
            "Chart witness must cover every scientific physical cell ID exactly once."
        )
    patches = np.asarray(witness.patches)
    if np.any(patches >= len(domain.patches)) or any(
        entity != domain.entity_id(2, int(patch))
        or path != domain.source_occurrences[2][int(patch)]
        for entity, path, patch in zip(
            witness.geometry_entity_ids, witness.occurrence_paths, patches, strict=True
        )
    ):
        raise ValueError(
            "Chart witness has stale canonical geometry entities or occurrence paths."
        )
    order = np.asarray(np.searchsorted(declared, ids), dtype=np.int64)
    vertex_charts: dict[tuple[int, int], tuple[float, float]] = {}
    chart_vertices: dict[tuple[int, tuple[float, float]], int] = {}
    offset = 0
    for block in mesh.blocks:
        for vertices in np.asarray(block.vertices):
            witness_row = int(order[offset])
            patch = int(patches[witness_row])
            for vertex, chart in zip(
                vertices, np.asarray(witness.charts)[witness_row], strict=True
            ):
                key = patch, int(np.asarray(mesh.vertex_global_ids)[int(vertex)])
                value = (float(chart[0]), float(chart[1]))
                if key in vertex_charts and not _same_periodic_chart_use(
                    domain, patch, vertex_charts[key], value
                ):
                    raise ValueError(
                        "Shared physical vertices disagree in the declared occurrence chart."
                    )
                vertex_charts[key] = value
                chart_key = patch, value
                if chart_key in chart_vertices and chart_vertices[chart_key] != key[1]:
                    raise ValueError(
                        "One occurrence chart vertex cannot merge distinct scientific physical vertices."
                    )
                chart_vertices[chart_key] = key[1]
            offset += 1
    return order


def _carrier(
    witness: SurfaceChartWitness, physical_order: _IntArray, patch: int
) -> tuple[CellMesh, _IntArray, _FloatArray]:
    selected = np.flatnonzero(np.asarray(witness.patches) == patch)
    rows_by_witness = np.empty(physical_order.size, dtype=np.int64)
    rows_by_witness[physical_order] = np.arange(physical_order.size, dtype=np.int64)
    charts = np.asarray(witness.charts)[selected]
    signs = exact_orient2d(charts[:, 0], charts[:, 1], charts[:, 2])
    if np.any(signs == 0):
        raise ValueError("A surface chart must be a nondegenerate UV triangle.")
    permutations = np.tile(np.arange(3, dtype=np.int32), (selected.size, 1))
    permutations[signs < 0] = np.asarray([0, 2, 1], dtype=np.int32)
    ordered = np.take_along_axis(charts, permutations[..., None], axis=1)
    points, inverse = np.unique(ordered.reshape((-1, 2)), axis=0, return_inverse=True)
    mesh = CellMesh.from_triangles(
        points,
        inverse.reshape((-1, 3)),
        cell_global_ids=np.asarray(witness.cell_global_ids)[selected],
    )
    reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)[
        permutations
    ]
    return mesh, rows_by_witness[selected], reference


def _exact_area(vertices: tuple[_ExactPoint, ...]) -> Fraction:
    return (
        sum(
            (
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


def _material_point(chart: _ExactTriangle, point: _ExactPoint) -> _ExactPoint:
    x, y = point
    values = tuple(
        (1 - x - y) * chart[0][axis] + x * chart[1][axis] + y * chart[2][axis]
        for axis in range(2)
    )
    return values[0], values[1]


def _material_reference(chart: _ExactTriangle, point: _ExactPoint) -> _ExactPoint:
    a = tuple(chart[1][axis] - chart[0][axis] for axis in range(2))
    b = tuple(chart[2][axis] - chart[0][axis] for axis in range(2))
    delta = tuple(point[axis] - chart[0][axis] for axis in range(2))
    determinant = a[0] * b[1] - a[1] * b[0]
    if not determinant:
        raise ValueError("An original exact material cell is singular.")
    return (
        (delta[0] * b[1] - delta[1] * b[0]) / determinant,
        (a[0] * delta[1] - a[1] * delta[0]) / determinant,
    )


def _material_reference_arguments(vertices: _ExactTriangle) -> tuple[Polynomial, ...]:
    variables = axes(2)
    return tuple(
        add(
            constant(vertices[0][axis], 2),
            add(
                scale(variables[0], vertices[1][axis] - vertices[0][axis]),
                scale(variables[1], vertices[2][axis] - vertices[0][axis]),
            ),
        )
        for axis in range(2)
    )


def _exact_retained_bytes(value: object, seen: set[int] | None = None) -> int:
    """Actual unique host storage of retained exact scientific metadata."""
    identities = set() if seen is None else seen
    identity = id(value)
    if identity in identities:
        return 0
    identities.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, Fraction):
        return (
            size
            + _exact_retained_bytes(value.numerator, identities)
            + _exact_retained_bytes(value.denominator, identities)
        )
    if isinstance(value, tuple):
        return size + sum(_exact_retained_bytes(item, identities) for item in value)
    if isinstance(value, dict):
        return size + sum(
            _exact_retained_bytes(item, identities)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, StrictModule):
        return size + _exact_retained_bytes(value.__dict__, identities)
    return size


def _material_piece_id(
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
            tuple((value.numerator, value.denominator) for value in point)
            for point in points
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


def _require_material_partition(
    pieces: tuple[PreparedSurfaceChartPiece, ...],
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    source_ids: _IntArray,
    target_ids: _IntArray,
) -> None:
    """Authenticate exact native supporting faces and both complete partitions."""
    source_areas, target_areas = [Fraction(0)] * len(source), [Fraction(0)] * len(target)
    fans: set[tuple[int, int, tuple[int, int, int]]] = set()
    for piece in pieces:
        old, new = piece.source_row, piece.target_row
        if (
            not 0 <= old < len(source)
            or not 0 <= new < len(target)
            or piece.source_cell_global_id != int(source_ids[old])
            or piece.target_cell_global_id != int(target_ids[new])
        ):
            raise ValueError(
                "Exact material piece changes its original scientific cell binding."
            )
        mapped = (
            _material_reference(target[new], source[old][0]),
            _material_reference(target[new], source[old][1]),
            _material_reference(target[new], source[old][2]),
        )
        origin = mapped[0]
        first = tuple(mapped[1][axis] - origin[axis] for axis in range(2))
        second = tuple(mapped[2][axis] - origin[axis] for axis in range(2))
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
            or _exact_area(polygon) <= 0
        ):
            raise ValueError(
                "Native exact material polygon loses its positive distinct-vertex construction."
            )
        for slot, label in enumerate(labels):
            if not 0 <= label < len(planes):
                raise ValueError(
                    "Native material boundary names an absent original supporting face."
                )
            point, next_point = polygon[slot], polygon[(slot + 1) % len(polygon)]
            previous = polygon[slot - 1]
            if (point[0] - previous[0]) * (next_point[1] - point[1]) - (
                point[1] - previous[1]
            ) * (next_point[0] - point[0]) <= 0:
                raise ValueError(
                    "The native material polygon is not the strictly convex original intersection."
                )
            if any(a * point[0] + b * point[1] > c for a, b, c in planes):
                raise ValueError(
                    "Native material polygon leaves an original rational halfplane."
                )
            a, b, c = planes[label]
            if any(a * p[0] + b * p[1] != c for p in (point, next_point)):
                raise ValueError(
                    "A native material boundary changes its original rational supporting face."
                )
        slots = piece.native_polygon_vertex_indices
        if (
            slots[0] != 0
            or slots[2] != slots[1] + 1
            or not 1 <= slots[1] < len(polygon) - 1
        ):
            raise ValueError(
                "A native material piece loses its canonical polygon fan partition."
            )
        key = old, new, slots
        if key in fans:
            raise ValueError("An exact native material fan piece is duplicated.")
        fans.add(key)
        old_vertices = polygon[slots[0]], polygon[slots[1]], polygon[slots[2]]
        new_vertices = (
            _material_point(mapped, old_vertices[0]),
            _material_point(mapped, old_vertices[1]),
            _material_point(mapped, old_vertices[2]),
        )
        if (
            old_vertices != piece.exact_source_reference_vertices
            or new_vertices != piece.exact_target_reference_vertices
        ):
            raise ValueError(
                "Native material piece alters its original exact source/target reference map."
            )
        old_area, new_area = (
            abs(_exact_area(old_vertices)),
            abs(_exact_area(new_vertices)),
        )
        if (
            not old_area
            or not new_area
            or old_area != piece.source_reference_area
            or new_area != piece.target_reference_area
        ):
            raise ValueError(
                "Native material piece changes its positive exact reference areas."
            )
        if piece.piece_id != _material_piece_id(
            piece.source_cell_global_id,
            piece.target_cell_global_id,
            old_vertices,
            new_vertices,
            polygon,
            labels,
            slots,
        ):
            raise ValueError(
                "An exact native material piece changes its original scientific identity."
            )
        source_areas[old] += old_area
        target_areas[new] += new_area
    if any(area != Fraction(1, 2) for area in (*source_areas, *target_areas)):
        raise ValueError(
            "Exact native material pieces lack COMPLETE original reference partitions."
        )


def _exact_chart_bank(
    values: ConvertibleToArray, count: int
) -> tuple[_ExactTriangle, ...]:
    bank = np.asarray(values, dtype=object)
    if bank.shape != (count, 3, 2):
        raise ValueError(
            "Exact material charts must align with their scientific cell witness."
        )
    charts: list[_ExactTriangle] = []
    for triangle in bank:
        points: list[_ExactPoint] = []
        for point in triangle:
            exact = cast(
                _ExactPoint,
                tuple(
                    value if isinstance(value, Fraction) else Fraction(float(value))
                    for value in point
                ),
            )
            points.append(exact)
        charts.append((points[0], points[1], points[2]))
    if any(not _exact_area(chart) for chart in charts):
        raise ValueError("An exact material chart is singular.")
    return tuple(charts)


def _native_material_candidates(
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    maximum_pairs: int,
    maximum_memory: int,
) -> tuple[_IntArray, _IntArray]:
    from .._bvh import bvh_overlap_pair_blocks, prepare_bvh

    boxes = []
    for charts in (source, target):
        lower = np.asarray(
            [
                [
                    outward(min(point[axis] for point in chart), -math.inf)
                    for axis in range(2)
                ]
                for chart in charts
            ]
        )
        upper = np.asarray(
            [
                [
                    outward(max(point[axis] for point in chart), math.inf)
                    for axis in range(2)
                ]
                for chart in charts
            ]
        )
        boxes.append(prepare_bvh(lower, upper, dtype=jnp.float64))
    left, right, count = [], [], 0
    for target_rows, source_rows in bvh_overlap_pair_blocks(boxes[1], boxes[0]):
        count += source_rows.size
        if count > maximum_pairs or count * 32 > maximum_memory:
            raise SurfaceChartResourceError(
                "Exact material candidate routing exceeds the original pair/storage allowance.",
                (
                    ("material_candidate_pairs", count, maximum_pairs),
                    ("material_candidate_bytes", count * 32, maximum_memory),
                ),
            )
        left.append(np.asarray(source_rows, dtype=np.int64))
        right.append(np.asarray(target_rows, dtype=np.int64))
    if not left:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(left), np.concatenate(right)


def _original_coefficient_limits(work: int, memory: int) -> tuple[int, int]:
    """Never run host coefficients past the actual ambient original allowance."""
    from .._meshcore import current_native_execution_budget

    budget = current_native_execution_budget()
    if budget is None:
        return work, memory
    allowance = budget.remaining()
    budget.charge(work=0)
    return min(work, allowance.remaining_work_units), min(
        memory, allowance.remaining_scratch_bytes
    )


def _prepare_contained_material_pieces(
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    source_ids: _IntArray,
    target_ids: _IntArray,
    maximum_pairs: int,
    maximum_work: int,
    maximum_memory: int,
    /,
) -> tuple[tuple[PreparedSurfaceChartPiece, ...], int, int, int] | None:
    """Construct an exact whole-child partition already proved by chart closure."""
    candidates = _native_material_candidates(
        source, target, maximum_pairs, maximum_memory
    )
    by_target: dict[int, list[int]] = {}
    for old, new in zip(*candidates, strict=True):
        by_target.setdefault(int(new), []).append(int(old))
    pieces = []
    source_areas = [Fraction(0)] * len(source)
    target_areas = [Fraction(0)] * len(target)
    retained_objects: set[int] = set()
    retained = candidates[0].nbytes + candidates[1].nbytes
    work = 0
    references: _ExactTriangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    for new in range(len(target)):
        selected: tuple[int, _ExactTriangle] | None = None
        for old in sorted(by_target.get(new, ()), key=lambda row: int(source_ids[row])):
            old_vertices = cast(
                _ExactTriangle,
                tuple(_material_reference(source[old], point) for point in target[new]),
            )
            work += 18
            if _exact_area(old_vertices) > 0 and all(
                min(u, v, 1 - u - v) >= 0 for u, v in old_vertices
            ):
                selected = old, old_vertices
                break
        if selected is None:
            return None
        old, old_vertices = selected
        mapped = cast(
            _ExactTriangle,
            tuple(_material_reference(target[new], point) for point in source[old]),
        )
        origin = mapped[0]
        first = (
            mapped[1][0] - origin[0],
            mapped[1][1] - origin[1],
        )
        second = (
            mapped[2][0] - origin[0],
            mapped[2][1] - origin[1],
        )
        planes = (
            (Fraction(-1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(-1), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(1)),
            (-first[0], -second[0], origin[0]),
            (-first[1], -second[1], origin[1]),
            (
                first[0] + first[1],
                second[0] + second[1],
                1 - origin[0] - origin[1],
            ),
        )
        labels = []
        for point, following in zip(
            old_vertices, old_vertices[1:] + old_vertices[:1], strict=True
        ):
            supported = [
                index
                for index, (a, b, c) in enumerate(planes)
                if a * point[0] + b * point[1] == c
                and a * following[0] + b * following[1] == c
            ]
            if not supported:
                return None
            labels.append(min(supported))
        new_vertices = tuple(_material_point(mapped, point) for point in old_vertices)
        if new_vertices != references:
            return None
        old_area = _exact_area(old_vertices)
        new_area = Fraction(1, 2)
        labels_ = tuple(labels)
        identity = _material_piece_id(
            int(source_ids[old]),
            int(target_ids[new]),
            old_vertices,
            references,
            old_vertices,
            labels_,
            (0, 1, 2),
        )
        piece = PreparedSurfaceChartPiece(
            source_row=old,
            target_row=new,
            source_cell_global_id=int(source_ids[old]),
            target_cell_global_id=int(target_ids[new]),
            exact_source_reference_vertices=old_vertices,
            exact_target_reference_vertices=references,
            source_reference_area=old_area,
            target_reference_area=new_area,
            native_boundary_labels=labels_,
            piece_id=identity,
            exact_native_source_polygon=old_vertices,
            native_polygon_vertex_indices=(0, 1, 2),
        )
        retained += _exact_retained_bytes(piece, retained_objects)
        pieces.append(piece)
        source_areas[old] += old_area
        target_areas[new] += new_area
        work += 64
    if any(area != Fraction(1, 2) for area in (*source_areas, *target_areas)):
        return None
    if work > maximum_work or retained > maximum_memory:
        raise SurfaceChartResourceError(
            "Contained material partition exceeds original resources.",
            (
                ("material_work", work, maximum_work),
                ("material_bytes", retained, maximum_memory),
            ),
        )
    charge_native_geometry_queries(0, work_units=work)
    return tuple(pieces), work, candidates[0].size, retained


def _prepare_exact_material_pieces(
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    source_ids: _IntArray,
    target_ids: _IntArray,
    maximum_pairs: int,
    maximum_work: int,
    maximum_memory: int,
    /,
    *,
    partitions_certified: bool = False,
) -> tuple[tuple[PreparedSurfaceChartPiece, ...], int, int, int]:
    """Native exact halfplane topology plus owning exact coefficient construction."""
    from .. import _meshcore
    from ..linalg._small_batched import prepare_exact_small_linear_actions
    from ._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    maximum_work, maximum_memory = _original_coefficient_limits(
        maximum_work, maximum_memory
    )
    if maximum_work <= 0 or maximum_memory <= 0 or maximum_pairs <= 0:
        raise SurfaceChartResourceError(
            "No original material overlap resources remain.",
            (("material_work", 1, maximum_work), ("material_bytes", 1, maximum_memory)),
        )
    if partitions_certified:
        contained = _prepare_contained_material_pieces(
            source,
            target,
            source_ids,
            target_ids,
            maximum_pairs,
            maximum_work,
            maximum_memory,
        )
        if contained is not None:
            return contained
    embedding_work, embedding_pairs = 0, 0
    if source != target and not partitions_certified:
        for charts, ids in ((source, source_ids), (target, target_ids)):
            embedding = _prepare_exact_material_pieces(
                charts,
                charts,
                ids,
                ids,
                maximum_pairs - embedding_pairs,
                maximum_work - embedding_work,
                maximum_memory,
            )
            embedding_work += embedding[1]
            embedding_pairs += embedding[2]
            del embedding
    ledger = CoordinateEnclosureBudget(maximum_work, maximum_memory)
    native_work = 0
    retained_objects: set[int] = set()
    pieces = []
    source_areas, target_areas = [Fraction(0)] * len(source), [Fraction(0)] * len(target)
    candidates = _native_material_candidates(
        source, target, maximum_pairs - embedding_pairs, maximum_memory
    )
    retained = candidates[0].nbytes + candidates[1].nbytes
    try:
        with ledger.activate():
            ledger.reserve(embedding_work)
            ledger.reserve(0, retained)
            for old_value, new_value in zip(*candidates, strict=True):
                old, new = int(old_value), int(new_value)
                ledger.reserve(64)
                mapped = (
                    _material_reference(target[new], source[old][0]),
                    _material_reference(target[new], source[old][1]),
                    _material_reference(target[new], source[old][2]),
                )
                origin = mapped[0]
                first = tuple(mapped[1][axis] - origin[axis] for axis in range(2))
                second = tuple(mapped[2][axis] - origin[axis] for axis in range(2))
                planes = (
                    (Fraction(-1), Fraction(0), Fraction(0)),
                    (Fraction(0), Fraction(-1), Fraction(0)),
                    (Fraction(1), Fraction(1), Fraction(1)),
                    (-first[0], -second[0], origin[0]),
                    (-first[1], -second[1], origin[1]),
                    (
                        first[0] + first[1],
                        second[0] + second[1],
                        1 - origin[0] - origin[1],
                    ),
                )
                remaining = maximum_work - ledger.work_units
                if remaining <= 0 or maximum_memory - retained <= 0:
                    raise SurfaceChartResourceError(
                        "Exact native material clipping has no remaining resources.",
                        (
                            ("material_work", ledger.work_units + 1, maximum_work),
                            ("material_bytes", retained + 1, maximum_memory),
                        ),
                    )
                supports, labels, clipping_work = _meshcore.clip_reference_triangle_exact(
                    planes,
                    maximum_scratch_bytes=maximum_memory - retained,
                    maximum_work_units=remaining,
                )
                native_work += clipping_work
                ledger.reserve(clipping_work)
                if supports.shape[0] < 3:
                    continue
                polygon_points: list[_ExactPoint] = []
                for left, right in supports:
                    a, b = planes[int(left)], planes[int(right)]
                    construction = prepare_exact_small_linear_actions(
                        ((a[0], a[1]), (b[0], b[1])),
                        ((a[2],), (b[2],)),
                        coordinate_budget=ledger,
                    )
                    if construction.actions is None:
                        raise ValueError(
                            "Native exact material support has no nonsingular source construction."
                        )
                    point = construction.actions[0][0], construction.actions[1][0]
                    if any(x * point[0] + y * point[1] > z for x, y, z in planes):
                        raise ValueError(
                            "Native exact material support violates an original halfplane."
                        )
                    polygon_points.append(point)
                polygon = tuple(polygon_points)
                if len(set(polygon)) != len(polygon) or _exact_area(polygon) <= 0:
                    raise ValueError(
                        "Native exact material polygon lacks a positive distinct-vertex partition."
                    )
                for slot, label in enumerate(labels):
                    a, b, c = planes[int(label)]
                    if any(
                        a * point[0] + b * point[1] != c
                        for point in (polygon[slot], polygon[(slot + 1) % len(polygon)])
                    ):
                        raise ValueError(
                            "Native exact material edge changes its original support."
                        )
                for slot in range(1, len(polygon) - 1):
                    old_vertices = polygon[0], polygon[slot], polygon[slot + 1]
                    new_vertices = (
                        _material_point(mapped, old_vertices[0]),
                        _material_point(mapped, old_vertices[1]),
                        _material_point(mapped, old_vertices[2]),
                    )
                    old_area, new_area = (
                        abs(_exact_area(old_vertices)),
                        abs(_exact_area(new_vertices)),
                    )
                    if not old_area or not new_area:
                        raise ValueError("An exact material piece is collapsed.")
                    identity = _material_piece_id(
                        int(source_ids[old]),
                        int(target_ids[new]),
                        old_vertices,
                        new_vertices,
                        polygon,
                        tuple(map(int, labels)),
                        (0, slot, slot + 1),
                    )
                    piece = PreparedSurfaceChartPiece(
                        source_row=old,
                        target_row=new,
                        source_cell_global_id=int(source_ids[old]),
                        target_cell_global_id=int(target_ids[new]),
                        exact_source_reference_vertices=old_vertices,
                        exact_target_reference_vertices=new_vertices,
                        source_reference_area=old_area,
                        target_reference_area=new_area,
                        native_boundary_labels=tuple(map(int, labels)),
                        piece_id=identity,
                        exact_native_source_polygon=polygon,
                        native_polygon_vertex_indices=(0, slot, slot + 1),
                    )
                    size = _exact_retained_bytes(piece, retained_objects)
                    ledger.reserve(24, size)
                    retained += size
                    pieces.append(piece)
                    source_areas[old] += old_area
                    target_areas[new] += new_area
    except CoordinateEnclosureResourceError as error:
        raise SurfaceChartResourceError(
            "Exact native material coefficient construction exhausts its original budget.",
            ((error.resource, error.requested, error.limit),),
        ) from error
    if any(area != Fraction(1, 2) for area in (*source_areas, *target_areas)):
        raise ValueError(
            "Native exact material overlap has a gap or double cover; COMPLETE source and target partitions required."
        )
    work = ledger.work_units
    if work > maximum_work:
        raise SurfaceChartResourceError(
            "Exact material clipping exceeds its aggregate original work allowance.",
            (("material_work", work, maximum_work),),
        )
    charge_native_geometry_queries(0, work_units=work - native_work - embedding_work)
    return tuple(pieces), work, candidates[0].size + embedding_pairs, retained


def _material_displacement_bounds(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_rows: _IntArray,
    target_rows: _IntArray,
    pieces: tuple[PreparedSurfaceChartPiece, ...],
    maximum_work: int,
    maximum_memory: int,
) -> tuple[_FloatArray, int]:
    """Continuous original-map difference on every actual native material piece."""
    from ._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
        expression_compose,
    )

    maximum_work, maximum_memory = _original_coefficient_limits(
        maximum_work, maximum_memory
    )
    if maximum_work <= 0 or maximum_memory <= 0:
        raise SurfaceChartResourceError(
            "Material-map difference has no remaining original coefficient resources.",
            (
                ("material_difference_work", 1, maximum_work),
                ("material_difference_bytes", 1, maximum_memory),
            ),
        )
    ledger = CoordinateEnclosureBudget(maximum_work, maximum_memory)
    maps = []
    try:
        with ledger.activate():
            for mesh, geometry, rows in (
                (source_mesh, source_geometry, source_rows),
                (target_mesh, target_geometry, target_rows),
            ):
                elements, routes, _ = geometry.resolve(mesh)
                controls = geometry.source_coordinates()
                needed = set(map(int, rows))
                expressions = {}
                physical_row = 0
                for element, route in zip(elements, routes, strict=True):
                    for row in np.asarray(route):
                        if physical_row in needed:
                            values = coordinate_expressions(
                                element, tuple(controls[int(index)] for index in row)
                            )
                            if values is None:
                                raise ValueError(
                                    "Material correspondence requires actual coordinate source expressions."
                                )
                            expressions[physical_row] = values
                        physical_row += 1
                maps.append(expressions)
            bounds = []
            for piece in pieces:
                left = _material_reference_arguments(
                    piece.exact_source_reference_vertices
                )
                right = _material_reference_arguments(
                    piece.exact_target_reference_vertices
                )
                bound = Fraction(0)
                for old, new in zip(
                    maps[0][int(source_rows[piece.source_row])],
                    maps[1][int(target_rows[piece.target_row])],
                    strict=True,
                ):
                    difference = expression_add(
                        expression_compose(old, left),
                        expression_scale(expression_compose(new, right), -1),
                    )
                    bound += max(
                        abs(value)
                        for value in expression_bernstein_coefficients(
                            difference, "simplex", 2
                        )
                    )
                bounds.append(outward(bound, math.inf))
    except CoordinateEnclosureResourceError as error:
        raise SurfaceChartResourceError(
            "Continuous material-map difference exhausts its original coefficient budget.",
            ((error.resource, error.requested, error.limit),),
        ) from error
    return np.asarray(bounds, dtype=np.float64), ledger.work_units


def _source_owned_domain_cover(
    domain: MeshingDomain,
    patch: int,
    carrier: CellMesh,
    source: MeshingDomainBoundarySource,
    maximum_pairs: int,
    maximum_memory_bytes: int,
    /,
) -> int:
    """Retain the original trim ribbons and prove equality of their UV carrier."""
    if source.domain.domain_id != domain.domain_id:
        raise ValueError("The source chart cover changes its authored trim authority.")
    records = [record for record in source.chart_triangulations if record[0] == patch]
    if len(records) != 1:
        raise ValueError(
            "The source occurrence requires exactly one authoritative chart triangulation."
        )
    (
        _,
        charts,
        points,
        cells,
        boundary,
        provenance,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
    ) = records[0]
    findings, counts = [], []
    valid, _, _, _ = _verify_chart_chain(
        domain,
        patch,
        charts,
        points,
        cells,
        boundary,
        provenance,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
        findings,
        counts,
        maximum_pairs=maximum_pairs,
        reserve_queries=charge_native_geometry_queries,
        topology_only=True,
    )
    charge_native_geometry_queries(
        0, work_units=sum(completed for _, completed, _ in counts)
    )
    if any(
        finding.check in ("source_trim_pair_capacity", "source_trim_arc_capacity")
        for finding in findings
    ):
        raise SurfaceChartResourceError(
            "The original trim proof exhausts its authored resources.", tuple(counts)
        )
    if not valid:
        raise ValueError("The original rooted trim chart cover is not certified.")
    original = CellMesh.from_triangles(charts, cells)
    overlap = prepare_common_refinement(
        original,
        carrier,
        policy=CommonRefinementPolicy(
            coverage=CommonRefinementCoverage.COMPLETE,
            maximum_candidate_pairs=maximum_pairs,
            maximum_accepted_pairs=maximum_pairs,
            maximum_memory_bytes=maximum_memory_bytes,
            overlap_simplices=True,
        ),
    )
    if overlap.status is CommonRefinementStatus.RESOURCE_LIMIT:
        raise SurfaceChartResourceError(
            overlap.evidence.reason,
            (
                (
                    "source_trim_overlap_pairs",
                    overlap.evidence.candidate_pair_count,
                    maximum_pairs,
                ),
                (
                    "source_trim_overlap_bytes",
                    overlap.evidence.working_bytes,
                    maximum_memory_bytes,
                ),
            ),
        )
    if not overlap.succeeded:
        raise ValueError(
            f"Source-owned trim carrier coverage refused: {overlap.status.name}."
        )
    return overlap.evidence.candidate_pair_count


def _domain_cover(domain: MeshingDomain, patch: int, carrier: CellMesh) -> None:
    """Infer no trims: prove the actual chain against exact authored line uses."""
    surface = domain.patches[patch].surface
    points = np.asarray(carrier.coordinates)
    box = np.stack((np.min(points, axis=0), np.max(points, axis=0)))
    surface.validate_parameter_box(box)
    # Seam copies and collapsed sides remain distinct chart witnesses.
    # The oriented source chain below establishes coverage.
    cells = np.asarray(carrier.blocks[0].vertices)
    chain: dict[tuple[int, int], int] = {}
    for first, second in np.concatenate(
        (cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])
    ):
        edge, reverse = (int(first), int(second)), (int(second), int(first))
        if chain.get(reverse, 0):
            chain[reverse] -= 1
        else:
            chain[edge] = chain.get(edge, 0) + 1
    remaining = {edge: count for edge, count in chain.items() if count}
    boundary: list[tuple[int, int]] = []
    provenance: list[tuple[int, int, float, float]] = []
    for loop_index, loop in enumerate(domain.patches[patch].loops):
        for use_index, use in enumerate(loop):
            if isinstance(use, PatchPoleUse):
                origin = tuple(Fraction(float(value)) for value in use.start)
                direction = tuple(
                    Fraction(float(end)) - first
                    for first, end in zip(origin, use.end, strict=True)
                )
                first_parameter, last_parameter = 0.0, 1.0
            elif (
                isinstance(use, PatchCurveUse)
                and isinstance(use.pcurve, LineCurve)
                and not any(
                    root is not None
                    for root in (
                        use.first_root,
                        use.last_root,
                        use.start_vertex_root,
                        use.end_vertex_root,
                    )
                )
                and use.trim_curve is None
            ):
                origin = tuple(
                    Fraction(float(value)) for value in np.asarray(use.pcurve.origin)
                )
                direction = tuple(
                    Fraction(float(value)) for value in np.asarray(use.pcurve.direction)
                )
                first_parameter, last_parameter = use.first, use.last
            else:
                raise ValueError(
                    "General rooted trims require their actual source chart-cover owner."
                )
            axis = next((axis for axis, value in enumerate(direction) if value), None)
            if axis is None:
                raise ValueError("A chart trim line must be nondegenerate.")
            entries = []
            for edge, count in remaining.items():
                parameters = []
                for vertex in edge:
                    point = tuple(Fraction(float(value)) for value in points[vertex])
                    parameter = (point[axis] - origin[axis]) / direction[axis]
                    if any(
                        point[k] != origin[k] + parameter * direction[k] for k in range(2)
                    ):
                        break
                    parameters.append(parameter)
                if (
                    len(parameters) == 2
                    and min(Fraction(first_parameter), Fraction(last_parameter))
                    <= min(parameters)
                    and max(parameters)
                    <= max(Fraction(first_parameter), Fraction(last_parameter))
                    and (parameters[1] - parameters[0])
                    * (Fraction(last_parameter) - Fraction(first_parameter))
                    > 0
                ):
                    if count != 1:
                        raise ValueError("Chart boundary is multiply covered.")
                    entries.append((parameters, edge))
            entries.sort(
                key=lambda item: item[0][0], reverse=last_parameter < first_parameter
            )
            for parameters, edge in entries:
                if any(Fraction(float(value)) != value for value in parameters):
                    raise ValueError(
                        "Exact chart trim parameters are not representable by the coverage owner."
                    )
                boundary.append(edge)
                provenance.append(
                    (loop_index, use_index, float(parameters[0]), float(parameters[1]))
                )
    physical = domain.evaluate(np.full(points.shape[0], patch, dtype=np.int32), points)
    findings, counts = [], []
    required = np.zeros((points.shape[0],), dtype=np.bool_)
    valid, _, _, _ = _verify_chart_chain(
        domain,
        patch,
        points,
        physical,
        cells,
        np.asarray(boundary, dtype=np.int32).reshape((-1, 2)),
        np.asarray(provenance, dtype=np.float64).reshape((-1, 4)),
        required,
        np.empty((0,), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        findings,
        counts,
    )
    if not valid or findings:
        raise ValueError(
            "Charts do not completely cover the authoritative source trim domain."
        )


def _exact_chart_partition_proof(
    source: SurfaceChartWitness, target: SurfaceChartWitness, /
) -> int:
    """Prove the target positive chart complex has the exact source boundary."""

    def records(
        witness: SurfaceChartWitness, /
    ) -> tuple[
        dict[int, dict[tuple[_ExactPoint, _ExactPoint], int]],
        dict[int, Fraction],
        int,
    ]:
        charts = _exact_chart_bank(witness.charts, witness.cell_global_ids.shape[0])
        boundaries: dict[int, dict[tuple[_ExactPoint, _ExactPoint], int]] = {}
        areas: dict[int, Fraction] = {}
        work = 0
        for patch, triangle in zip(np.asarray(witness.patches), charts, strict=True):
            area = _exact_area(triangle)
            if area <= 0:
                raise ValueError("Restricted chart partition has a nonpositive cell.")
            patch_ = int(patch)
            areas[patch_] = areas.get(patch_, Fraction(0)) + area
            boundary = boundaries.setdefault(patch_, {})
            for first, second in zip(triangle, triangle[1:] + triangle[:1], strict=True):
                reverse = second, first
                if boundary.get(reverse, 0):
                    boundary[reverse] -= 1
                    if boundary[reverse] == 0:
                        del boundary[reverse]
                else:
                    edge = first, second
                    boundary[edge] = boundary.get(edge, 0) + 1
                work += 1
        return boundaries, areas, work

    source_boundary, source_area, source_work = records(source)
    target_boundary, target_area, target_work = records(target)
    if (
        source_boundary != target_boundary
        or source_area != target_area
        or source.domain_id != target.domain_id
    ):
        raise ValueError(
            "Target restricted charts do not exactly partition the source chart complex."
        )
    return source_work + target_work


def _rational_sqrt_enclosure(value: Fraction, /) -> tuple[Fraction, Fraction]:
    """Tight exact radical bounds without renewing the coefficient ledger."""
    if value <= 0:
        raise ValueError("A measure radical requires a positive value.")
    exponent = (value.numerator.bit_length() - value.denominator.bit_length() + 1) // 2
    upper = Fraction(2) ** exponent
    while upper * upper < value:
        upper *= 2
    while (upper / 2) * (upper / 2) >= value:
        upper /= 2
    for _ in range(8):
        upper = (upper + value / upper) / 2
    return value / upper, upper


def _validity_measure_enclosures(
    validity: CellValidityCertificate, /
) -> tuple[np.ndarray, np.ndarray]:
    """Positive physical triangle-area enclosures from whole-cell Gram bounds."""
    measures = []
    errors = []
    values = tuple(
        zip(
            np.asarray(validity.determinant_lower),
            np.asarray(validity.determinant_upper),
            strict=True,
        )
    )
    for lower, upper in values:
        if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0:
            raise ValueError("Validity evidence lacks a positive finite Gram bound.")
        # Exact rational Newton bounds consume the determinant certificate
        # without reopening its coefficient-work ledger.
        low = _rational_sqrt_enclosure(Fraction(float(lower)))[0] / 2
        high = _rational_sqrt_enclosure(Fraction(float(upper)))[1] / 2
        midpoint = (low + high) / 2
        value = float(midpoint)
        error = (high - low) / 2 + abs(Fraction(value) - midpoint)
        measures.append(value)
        errors.append(outward(error, math.inf) if error else 0.0)
    return (
        np.asarray(measures, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def _renew_restricted_surface_validity(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    policy: CellValidityPolicy,
    source_witness: SurfaceChartWitness,
    target_witness: SurfaceChartWitness,
    /,
) -> tuple[
    CellValidityCertificate,
    tuple[tuple[int, Fraction] | None, ...],
]:
    """Reuse unchanged exact restriction rows and certify only born chart cells."""
    source_validity.require_bound(source_geometry, mesh=source_mesh)
    if (
        source_validity.policy_id != policy.policy_id
        or not source_validity.all_certified
        or source_geometry.restriction_source is None
        or target_geometry.restriction_source is None
        or source_geometry.restriction_source.source_geometry_id
        != target_geometry.restriction_source.source_geometry_id
        or source_geometry.restriction_source.source_topology_id
        != target_geometry.restriction_source.source_topology_id
        or not np.array_equal(
            np.asarray(source_geometry.coordinates),
            np.asarray(target_geometry.coordinates),
        )
    ):
        raise ValueError(
            "Restricted validity reuse requires one positive exact source authority."
        )
    source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
    source_rows: dict[int, tuple[int, str, tuple[int, ...], tuple[int, ...]]] = {}
    cursor = 0
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids)
    for block, element, routes in zip(
        source_mesh.blocks, source_elements, source_routes, strict=True
    ):
        for local_row, (identifier, vertices, route) in enumerate(
            zip(
                np.asarray(block.global_ids),
                np.asarray(block.vertices),
                np.asarray(routes),
                strict=True,
            )
        ):
            source_rows[int(identifier)] = (
                cursor + local_row,
                element.element_id,
                tuple(map(int, route)),
                tuple(map(int, source_vertex_ids[vertices])),
            )
        cursor += block.cell_count
    source_witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(source_witness.cell_global_ids))
    }
    target_witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(target_witness.cell_global_ids))
    }

    def restriction_scale(
        source_chart: np.ndarray, target_chart: np.ndarray, /
    ) -> Fraction | None:
        source = tuple(
            tuple(Fraction(float(value)) for value in point) for point in source_chart
        )
        target = tuple(
            tuple(Fraction(float(value)) for value in point) for point in target_chart
        )
        a, b, c = source
        determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if determinant <= 0:
            return None
        references = []
        for point in target:
            delta = point[0] - a[0], point[1] - a[1]
            u = (delta[0] * (c[1] - a[1]) - delta[1] * (c[0] - a[0])) / determinant
            v = ((b[0] - a[0]) * delta[1] - (b[1] - a[1]) * delta[0]) / determinant
            if min(u, v, 1 - u - v) < 0:
                return None
            references.append((u, v))
        scale = (references[1][0] - references[0][0]) * (
            references[2][1] - references[0][1]
        ) - (references[1][1] - references[0][1]) * (references[2][0] - references[0][0])
        return scale if scale > 0 else None

    target_elements, target_routes, _ = target_geometry.resolve(target_mesh)
    target_vertex_ids = np.asarray(target_mesh.vertex_global_ids)
    changed_masks: list[np.ndarray] = []
    inherited_rows: list[list[tuple[int, Fraction] | None]] = []
    source_ids = sorted(source_witness_rows)
    source_patches = np.asarray(source_witness.patches)
    source_charts = np.asarray(source_witness.charts)
    target_patches = np.asarray(target_witness.patches)
    target_charts = np.asarray(target_witness.charts)
    for block, element, routes in zip(
        target_mesh.blocks, target_elements, target_routes, strict=True
    ):
        changed = np.ones((block.cell_count,), dtype=np.bool_)
        inherited: list[tuple[int, Fraction] | None] = []
        for row, (identifier, vertices, route) in enumerate(
            zip(
                np.asarray(block.global_ids),
                np.asarray(block.vertices),
                np.asarray(routes),
                strict=True,
            )
        ):
            identifier_ = int(identifier)
            previous = source_rows.get(identifier_)
            inheritance: tuple[int, Fraction] | None = None
            if previous is not None and previous[1:] == (
                element.element_id,
                tuple(map(int, route)),
                tuple(map(int, target_vertex_ids[vertices])),
            ):
                inheritance = previous[0], Fraction(1)
            else:
                target_row = target_witness_rows[identifier_]
                for source_id in source_ids:
                    source_row = source_witness_rows[source_id]
                    if source_patches[source_row] != target_patches[target_row]:
                        continue
                    scale = restriction_scale(
                        source_charts[source_row], target_charts[target_row]
                    )
                    if scale is not None:
                        inheritance = source_rows[source_id][0], scale
                        break
            if inheritance is not None:
                changed[row] = False
            inherited.append(inheritance)
        changed_masks.append(changed)
        inherited_rows.append(inherited)
    changed_blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            np.asarray(block.vertices)[mask],
            global_ids=np.asarray(block.global_ids)[mask],
        )
        for block, mask in zip(target_mesh.blocks, changed_masks, strict=True)
        if np.any(mask)
    )
    changed_by_name = {
        block.name: mask
        for block, mask in zip(target_mesh.blocks, changed_masks, strict=True)
        if np.any(mask)
    }
    changed_validity = None
    if changed_blocks:
        changed_mesh = CellMesh(
            np.asarray(target_mesh.coordinates),
            changed_blocks,
            vertex_global_ids=target_mesh.vertex_global_ids,
            numeric_version=target_mesh.numeric_version,
        )
        element_lookup = dict(
            zip(target_geometry.block_names, target_geometry.elements, strict=True)
        )
        route_lookup = dict(
            zip(target_geometry.block_names, target_geometry.geometry_dofs, strict=True)
        )
        origin = target_geometry.restriction_source
        if origin is None:
            raise RuntimeError("Changed restricted cells lost their source record.")
        source_blocks = origin.block_source_blocks
        changed_origin = CellGeometryRestrictionSource(
            origin.source_geometry_id,
            origin.source_topology_id,
            {
                name: np.asarray(origin.block_parent_cell_ids[name])[mask]
                for name, mask in changed_by_name.items()
            },
            {
                name: np.asarray(origin.block_parent_vertex_ids[name])[mask]
                for name, mask in changed_by_name.items()
            },
            block_source_blocks=None
            if source_blocks is None
            else {name: source_blocks[name] for name in changed_by_name},
        )
        changed_geometry = CellGeometrySpec(
            {name: element_lookup[name] for name in changed_by_name},
            {
                name: np.asarray(route_lookup[name])[mask]
                for name, mask in changed_by_name.items()
            },
            target_geometry.coordinates,
            restriction_source=changed_origin,
        )
        changed_validity = certify_cell_geometry_validity(
            changed_geometry, mesh=changed_mesh, policy=policy
        )
        if not changed_validity.all_certified:
            raise ValueError("Born restricted surface cells are not certified valid.")
    status: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    depth: list[int] = []
    measure_routes: list[tuple[int, Fraction] | None] = []
    changed_cursor = 0
    target_offsets = [0]
    for block, mask, inherited in zip(
        target_mesh.blocks, changed_masks, inherited_rows, strict=True
    ):
        for is_changed, inheritance in zip(mask, inherited, strict=True):
            scale = Fraction(1)
            if is_changed:
                if changed_validity is None:
                    raise RuntimeError("Changed validity rows lost their certificate.")
                row = changed_cursor
                changed_cursor += 1
                certificate = changed_validity
                measure_routes.append(None)
            else:
                if inheritance is None:
                    raise RuntimeError("Inherited validity row lost its source proof.")
                row, scale = inheritance
                certificate = source_validity
                measure_routes.append((row, scale))
            status.append(int(np.asarray(certificate.status)[row]))
            low = Fraction(float(np.asarray(certificate.determinant_lower)[row]))
            high = Fraction(float(np.asarray(certificate.determinant_upper)[row]))
            determinant_scale = scale * scale
            lower.append(outward(low * determinant_scale, -math.inf))
            upper.append(outward(high * determinant_scale, math.inf))
            depth.append(int(np.asarray(certificate.depth)[row]))
        target_offsets.append(len(status))
    certificate = CellValidityCertificate(
        np.asarray(status, dtype=np.int32),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(depth, dtype=np.int32),
        block_names=tuple(block.name for block in target_mesh.blocks),
        block_offsets=tuple(target_offsets),
        unsupported_block_names=(),
        unresolved_reasons=(),
        geometry_id=cell_geometry_id(target_geometry),
        geometry_layout_id=target_geometry.geometry_layout_id,
        topology_id=target_mesh.topology_id,
        policy_id=policy.policy_id,
    )
    return certificate, tuple(measure_routes)


def _source_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SurfaceChartWitness,
    order: _IntArray,
    domain: MeshingDomain,
    maximum_work: int,
) -> tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray, int]:
    from ._cell_geometry_transfer import _surface_node_enclosures

    unresolved_elements, routes, _ = geometry.resolve(mesh)
    elements = _coordinate_elements(unresolved_elements)
    local_coordinates = geometry.source_coordinates()
    total, defects, corners, chords = [], [], [], []
    work, offset = 0, 0
    variables = axes(2)
    reference_corners = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for local in np.asarray(route, dtype=np.int64):
            expressions = coordinate_expressions(
                element, tuple(local_coordinates[index] for index in local)
            )
            if expressions is None:
                raise ValueError(
                    "Source coordinate map lacks canonical exact source expressions."
                )
            work += (
                sum(
                    expression_node_count(expression, "simplex", 2)
                    for expression in expressions
                )
                + element.local_dof_count
            )
            if work > maximum_work:
                raise SurfaceChartResourceError(
                    "Surface source expression enclosure exceeds its work budget.",
                    (("source_expression_work", work, maximum_work),),
                )
            row = int(order[offset])
            patch = int(np.asarray(witness.patches)[row])
            chart = np.asarray(witness.charts)[row : row + 1]
            low, high = _surface_node_enclosures(
                domain, patch, chart, np.asarray(reference_corners, dtype=np.float64)
            )
            corner_values = tuple(
                tuple(
                    expression_evaluate(expression, point) for expression in expressions
                )
                for point in reference_corners
            )
            defect = Fraction(0)
            for component, expression in enumerate(expressions):
                values = tuple(point[component] for point in corner_values)
                chord = add(
                    constant(values[0], 2),
                    add(
                        scale(variables[0], values[1] - values[0]),
                        scale(variables[1], values[2] - values[0]),
                    ),
                )
                difference = expression_add(expression, expression_scale(chord, -1))
                defect += max(
                    abs(value)
                    for value in expression_bernstein_coefficients(
                        difference, "simplex", 2
                    )
                )
            corner_error = max(
                sum(
                    (
                        max(
                            abs(value - Fraction(float(lower))),
                            abs(value - Fraction(float(upper))),
                        )
                        for value, lower, upper in zip(
                            point, low[0, slot], high[0, slot], strict=True
                        )
                    ),
                    Fraction(0),
                )
                for slot, point in enumerate(corner_values)
            )
            chord_bound = float(domain.interpolation_bounds(patch, chart)[0])
            if not math.isfinite(chord_bound) or chord_bound < 0:
                raise ValueError(
                    "Authoritative source lacks a continuous chord enclosure."
                )
            total.append(outward(defect + corner_error + Fraction(chord_bound), math.inf))
            defects.append(outward(defect, math.inf))
            corners.append(outward(corner_error, math.inf))
            chords.append(chord_bound)
            offset += 1
    return (
        np.asarray(total, dtype=np.float64),
        np.asarray(defects, dtype=np.float64),
        np.asarray(corners, dtype=np.float64),
        np.asarray(chords, dtype=np.float64),
        work,
    )


def prepare_surface_chart_deformation(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    reconstruction: SurfaceGeometryReconstruction,
    domain: MeshingDomain,
    /,
    *,
    source_witness: SurfaceChartWitness,
    target_witness: SurfaceChartWitness,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_evaluations: int = 1_000_000,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    source_chart_cover: MeshingDomainBoundarySource | None = None,
    source_material_charts: ConvertibleToArray | None = None,
    target_material_charts: ConvertibleToArray | None = None,
    material_root_witness: SurfaceChartWitness | None = None,
    target_chart_cover: MeshingDomainBoundarySource | None = None,
    prepared_source_fidelity: SourceFidelityCertificate | None = None,
    prepared_source_restriction_id: str | None = None,
) -> PreparedSurfaceChartDeformation:
    """Prepare the full bounded transition; any unavailable premise refuses it."""
    from ._cell_geometry_transfer import (
        _certified_cell_measures,
        SurfaceGeometryReconstruction,
    )

    if not isinstance(domain, MeshingDomain) or not isinstance(
        reconstruction, SurfaceGeometryReconstruction
    ):
        raise TypeError(
            "An authoritative domain and actual surface reconstruction are required."
        )
    evaluations = _positive_integer(maximum_evaluations, "maximum_evaluations")
    candidates = _positive_integer(maximum_candidate_pairs, "maximum_candidate_pairs")
    memory = _positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
    for value in (maximum_fidelity, maximum_displacement):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "Continuous fidelity and displacement limits must be finite and nonnegative."
            )
    target_geometry = reconstruction.geometry
    for mesh in (source_mesh, target_mesh):
        if (
            not isinstance(mesh, CellMesh)
            or mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
            or any(block.cell_kind != "triangle" for block in mesh.blocks)
        ):
            raise ValueError(
                "Surface deformation requires explicit physical triangle meshes in 3D."
            )
        mesh.require_dense("Surface chart deformation")
        # Physical quotient copies retain their explicit occurrence UV uses;
        # the periodic controller separately certifies the actual quotient.
    if (
        reconstruction.source_geometry_id,
        reconstruction.source_topology_id,
        reconstruction.target_topology_id,
        reconstruction.domain_id,
    ) != (
        cell_geometry_id(source_geometry),
        source_mesh.topology_id,
        target_mesh.topology_id,
        domain.domain_id,
    ):
        raise ValueError(
            "Target reconstruction is stale for this source, topology or domain."
        )
    source_order = _witness_order(source_witness, source_mesh, source_geometry, domain)
    if (prepared_source_fidelity is None) != (prepared_source_restriction_id is None):
        raise ValueError(
            "Prepared source fidelity and restriction identity must be supplied together."
        )
    if prepared_source_fidelity is not None:
        prepared_source_fidelity.binding.require(source_mesh, source_geometry)
        restriction = source_geometry.restriction_source
        if (
            restriction is None
            or restriction.restriction_source_id != prepared_source_restriction_id
        ):
            raise ValueError("Prepared source fidelity names another exact restriction.")
        binding = prepared_source_fidelity.binding
        if (
            (binding.source_id, binding.source_revision)
            != (domain.source_id, domain.source_revision)
            or source_witness.geometry_id != binding.geometry_id
            or source_witness.topology_id != binding.topology_id
            or source_witness.domain_id != domain.domain_id
            or prepared_source_fidelity.status != "certified"
            or prepared_source_fidelity.mesh_to_source_semantics != "certified"
            or prepared_source_fidelity.source_to_mesh_semantics != "certified"
            or any(
                value != 0.0
                for value in (
                    prepared_source_fidelity.mesh_to_source_upper,
                    prepared_source_fidelity.mesh_to_source_lower,
                    prepared_source_fidelity.source_to_mesh_upper,
                    prepared_source_fidelity.source_to_mesh_lower,
                )
            )
        ):
            raise ValueError(
                "Prepared exact source fidelity is stale, nonzero, or differently governed."
            )
    target_order = _witness_order(target_witness, target_mesh, target_geometry, domain)
    source_patches, target_patches = (
        set(map(int, np.asarray(source_witness.patches))),
        set(map(int, np.asarray(target_witness.patches))),
    )
    if source_patches != target_patches:
        raise ValueError(
            "Source and target must cover the same separate patch occurrences."
        )
    cell_count = source_order.size + target_order.size
    # Include carrier connectivity, coordinates, maps and retained certificate arrays.
    unresolved_source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
    source_elements = _coordinate_elements(unresolved_source_elements)
    polynomial_work = sum(
        np.asarray(route).shape[0]
        * element.local_dof_count
        * 3
        * math.comb(element.degree + 2, 2)
        for element, route in zip(source_elements, source_routes, strict=True)
    )
    if source_chart_cover is None:
        boundary_work = sum(
            9 * int(np.sum(np.asarray(witness.patches) == patch)) ** 2
            for witness in (source_witness, target_witness)
            for patch in source_patches
        )
    else:
        # The retained authority verifies its boundary once; COMPLETE native
        # overlap transfers that theorem, rather than re-proving every cell pair.
        boundary_work = sum(
            record[4].shape[0] * (record[4].shape[0] - 1) // 2
            for record in source_chart_cover.chart_triangulations
            if record[0] in source_patches
        )
    preparation_work = cell_count * 24 + polynomial_work + boundary_work
    if cell_count * 1024 > memory or preparation_work > evaluations:
        raise SurfaceChartResourceError(
            "Surface chart preparation exceeds resources before artifacts.",
            (
                ("chart_preparation_work_bound", preparation_work, evaluations),
                ("chart_preparation_storage_bound", cell_count * 1024, memory),
            ),
        )
    if prepared_source_fidelity is None:
        source_bounds, polynomial, corner, chord, work = _source_fidelity(
            source_mesh,
            source_geometry,
            source_witness,
            source_order,
            domain,
            evaluations,
        )
    else:
        source_bounds = np.zeros((source_order.size,), dtype=np.float64)
        polynomial = source_bounds.copy()
        corner = source_bounds.copy()
        chord = source_bounds.copy()
        work = 0
    reconstruction_ids = np.asarray(reconstruction.cell_global_ids)
    if np.unique(
        reconstruction_ids
    ).size != reconstruction_ids.size or not np.array_equal(
        np.sort(reconstruction_ids), np.sort(_cell_ids(target_mesh))
    ):
        raise ValueError(
            "Target reconstruction cell IDs do not cover the physical target."
        )
    lookup = {int(identifier): row for row, identifier in enumerate(reconstruction_ids)}
    target_bound_rows = np.asarray(
        [lookup[int(identifier)] for identifier in _cell_ids(target_mesh)], dtype=np.int64
    )
    target_bounds = np.asarray(reconstruction.fidelity_bounds, dtype=np.float64)[
        target_bound_rows
    ]
    if (
        np.any(~np.isfinite(target_bounds))
        or np.any(target_bounds < 0)
        or max(float(np.max(source_bounds)), float(np.max(target_bounds)))
        > maximum_fidelity
    ):
        raise ValueError(
            "Continuous source or target fidelity exceeds the accepted bound."
        )
    for physical_row, witness_row in enumerate(target_order):
        row = lookup[int(_cell_ids(target_mesh)[physical_row])]
        if (
            reconstruction.cell_geometry_entity_ids[row]
            != target_witness.geometry_entity_ids[witness_row]
            or reconstruction.cell_occurrence_paths[row]
            != target_witness.occurrence_paths[witness_row]
        ):
            raise ValueError(
                "Target reconstruction and witness occurrence identities disagree."
            )
        if int(np.asarray(reconstruction.cell_patches)[row]) != int(
            np.asarray(target_witness.patches)[witness_row]
        ) or not np.array_equal(
            np.asarray(reconstruction.cell_charts)[row],
            np.asarray(target_witness.charts)[witness_row],
        ):
            raise ValueError(
                "Target witness changes the reconstruction's authoritative reference-to-chart mapping."
            )
    limits = (
        MeshCertificateLimits(
            maximum_candidate_pairs=candidates,
            maximum_source_samples=evaluations,
            maximum_distance_evaluations=evaluations,
            maximum_subdivision_pieces=min(evaluations, memory // 1024),
            maximum_bernstein_nodes=min(evaluations, memory // 1024),
        )
        if certificate_limits is None
        else certificate_limits
    )
    validity_policy_ = (
        CellValidityPolicy(
            maximum_piece_count=min(evaluations, memory // 1024),
            maximum_bernstein_nodes=min(evaluations, memory // 1024),
        )
        if validity_policy is None
        else validity_policy
    )
    if prepared_source_certificates is None:
        source_validity = certify_cell_geometry_validity(
            source_geometry, mesh=source_mesh, policy=validity_policy_
        )
        source_embedding = certify_global_embedding(
            source_mesh, source_geometry, source_validity, limits=limits
        )
    else:
        source_validity, source_embedding = prepared_source_certificates
        source_validity.require_bound(source_geometry, mesh=source_mesh)
        source_embedding.binding.require(source_mesh, source_geometry)
        if (
            source_validity.policy_id != validity_policy_.policy_id
            or source_embedding.binding.limits_id != limits.limits_id
        ):
            raise ValueError(
                "Prepared source certificates must retain the original validity and embedding policies."
            )
    if prepared_source_certificates is not None:
        target_validity, target_measure_routes = _renew_restricted_surface_validity(
            source_mesh,
            source_geometry,
            source_validity,
            target_mesh,
            target_geometry,
            validity_policy_,
            source_witness,
            target_witness,
        )
    else:
        target_validity = certify_cell_geometry_validity(
            target_geometry, mesh=target_mesh, policy=validity_policy_
        )
        target_measure_routes = None
    if prepared_source_certificates is not None:
        partition_work = _exact_chart_partition_proof(source_witness, target_witness)
        relation = target_mesh.topology.incidences[1].relation
        valid = np.asarray(relation.valid, dtype=np.bool_)
        incident = np.bincount(
            np.asarray(relation.source_indices)[valid],
            minlength=target_mesh.entity_set(1).count,
        )
        binding = MeshCertificateBinding(
            target_mesh,
            target_geometry,
            source_embedding.binding.coordinate_scope,
            limits,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
        )
        target_embedding = GlobalEmbeddingCertificate(
            binding,
            target_validity.certificate_id,
            (),
            (
                "restricted_source_embedding",
                "target_chart_topology",
                "target_cell_validity",
            ),
            cell_count=target_mesh.entity_set(2).count,
            boundary_facet_count=int(np.count_nonzero(incident == 1)),
            shell_count=source_embedding.shell_count,
            candidate_pair_count=partition_work,
            ray_test_count=0,
            subdivision_piece_count=0,
        )
    else:
        target_embedding = certify_global_embedding(
            target_mesh, target_geometry, target_validity, limits=limits
        )
    validities = [source_validity, target_validity]
    embeddings = [source_embedding, target_embedding]
    if any(not validity.all_certified for validity in validities) or any(
        embedding.status != "certified" for embedding in embeddings
    ):
        raise ValueError(
            "Surface deformation requires independently valid globally embedded source and target maps."
        )
    if (source_material_charts is None) != (target_material_charts is None):
        raise ValueError(
            "Material correspondence requires both exact endpoint chart banks."
        )
    root_witness = (
        source_witness if material_root_witness is None else material_root_witness
    )
    if root_witness.domain_id != domain.domain_id:
        raise ValueError("Material root witness changes its original source domain.")
    if source_material_charts is not None and material_root_witness is None:
        raise ValueError(
            "Authored material correspondence requires its retained original root chart witness."
        )
    source_material = _exact_chart_bank(
        source_witness.charts
        if source_material_charts is None
        else source_material_charts,
        source_order.size,
    )
    target_material = _exact_chart_bank(
        target_witness.charts
        if target_material_charts is None
        else target_material_charts,
        target_order.size,
    )
    root_material = _exact_chart_bank(
        root_witness.charts, root_witness.cell_global_ids.shape[0]
    )
    material_bytes = _exact_retained_bytes(
        (source_material, target_material, root_material)
    )
    occurrences, retained, remaining_candidates = (
        [],
        cell_count * 1024 + material_bytes,
        candidates,
    )
    if retained >= memory:
        raise SurfaceChartResourceError(
            "Exact scientific material banks exhaust the original retained allowance.",
            (("exact_material_retained_bytes", retained, memory),),
        )
    source_rows_by_witness = np.empty(source_order.size, dtype=np.int64)
    source_rows_by_witness[source_order] = np.arange(source_order.size, dtype=np.int64)
    maximum_bound = 0.0
    for patch in sorted(
        source_patches,
        key=lambda index: (
            domain.entity_id(2, index),
            domain.source_occurrences[2][index],
        ),
    ):
        target_carrier, target_rows, _ = _carrier(target_witness, target_order, patch)
        root_carrier, _, _ = _carrier(
            root_witness,
            np.arange(root_witness.cell_global_ids.shape[0], dtype=np.int64),
            patch,
        )
        old_selection = np.flatnonzero(np.asarray(source_witness.patches) == patch)
        source_rows = source_rows_by_witness[old_selection]
        new_selection = np.flatnonzero(np.asarray(target_witness.patches) == patch)
        root_selection = np.flatnonzero(np.asarray(root_witness.patches) == patch)
        old_charts = tuple(source_material[int(row)] for row in old_selection)
        new_charts = tuple(target_material[int(row)] for row in new_selection)
        root_charts = tuple(root_material[int(row)] for row in root_selection)
        old_ids = np.asarray(source_witness.cell_global_ids)[old_selection]
        new_ids = np.asarray(target_witness.cell_global_ids)[new_selection]
        root_ids = np.asarray(root_witness.cell_global_ids)[root_selection]
        if not root_charts:
            raise ValueError("Material root witness omits an original source occurrence.")
        if prepared_source_certificates is None:
            if source_chart_cover is None:
                _domain_cover(domain, patch, root_carrier)
            else:
                remaining_candidates -= _source_owned_domain_cover(
                    domain,
                    patch,
                    root_carrier,
                    source_chart_cover,
                    remaining_candidates,
                    memory - retained,
                )
            target_cover = (
                source_chart_cover if target_chart_cover is None else target_chart_cover
            )
            if target_cover is None:
                _domain_cover(domain, patch, target_carrier)
            else:
                remaining_candidates -= _source_owned_domain_cover(
                    domain,
                    patch,
                    target_carrier,
                    target_cover,
                    remaining_candidates,
                    memory - retained,
                )
        # The root premise is independently transported to every current source
        # cell by a genuine COMPLETE exact native partition, not a UV proxy.
        root_pieces = ()
        if old_charts != root_charts or not np.array_equal(old_ids, root_ids):
            root_pieces, root_work, root_pairs, root_bytes = (
                _prepare_exact_material_pieces(
                    root_charts,
                    old_charts,
                    root_ids,
                    old_ids,
                    remaining_candidates,
                    evaluations - work,
                    memory - retained,
                    partitions_certified=prepared_source_certificates is not None,
                )
            )
            remaining_candidates -= root_pairs
            work += root_work
            retained += root_bytes
        pieces, overlap_work, pair_count, piece_bytes = _prepare_exact_material_pieces(
            old_charts,
            new_charts,
            old_ids,
            new_ids,
            remaining_candidates,
            evaluations - work,
            memory - retained,
            partitions_certified=prepared_source_certificates is not None,
        )
        remaining_candidates -= pair_count
        work += overlap_work
        retained += piece_bytes
        if prepared_source_fidelity is not None:
            displacement = np.zeros((len(pieces),), dtype=np.float64)
            difference_work = 0
        else:
            displacement, difference_work = _material_displacement_bounds(
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
                source_rows,
                target_rows,
                pieces,
                evaluations - work,
                memory - retained,
            )
        work += difference_work
        charge_native_geometry_queries(0, work_units=difference_work)
        maximum_bound = max(maximum_bound, float(np.max(displacement, initial=0.0)))
        if maximum_bound > maximum_displacement:
            raise ValueError(
                "Old-to-target whole-material-overlap displacement exceeds the accepted bound."
            )
        occurrences.append(
            PreparedSurfaceChartOccurrence(
                patch=patch,
                geometry_entity_id=domain.entity_id(2, patch),
                occurrence_path=domain.source_occurrences[2][patch],
                source_rows=jnp.asarray(source_rows, dtype=jnp.int64),
                target_rows=jnp.asarray(target_rows, dtype=jnp.int64),
                exact_source_material_charts=old_charts,
                exact_target_material_charts=new_charts,
                pieces=pieces,
                root_pieces=root_pieces,
                material_root_witness_id=root_witness.witness_id,
                displacement_bounds=jnp.asarray(displacement, dtype=jnp.float64),
                source_domain_coverage="certified",
                target_domain_coverage="certified",
            )
        )
    if prepared_source_certificates is not None:
        source_measures, source_errors = _validity_measure_enclosures(source_validity)
        if target_measure_routes is None or any(
            route is None for route in target_measure_routes
        ):
            target_measures, target_errors = _validity_measure_enclosures(target_validity)
        else:
            target_values = []
            target_measure_bounds = []
            for route in target_measure_routes:
                if route is None:
                    raise RuntimeError("Target measure route unexpectedly vanished.")
                source_row, scale = route
                midpoint = Fraction(float(source_measures[source_row])) * scale
                value = float(midpoint)
                error = Fraction(float(source_errors[source_row])) * scale + abs(
                    Fraction(value) - midpoint
                )
                target_values.append(value)
                target_measure_bounds.append(outward(error, math.inf) if error else 0.0)
            target_measures = np.asarray(target_values, dtype=np.float64)
            target_errors = np.asarray(target_measure_bounds, dtype=np.float64)
    else:
        source_measures, source_errors, _ = _certified_cell_measures(
            source_mesh,
            source_geometry,
            absolute_tolerance=measure_absolute_tolerance,
            relative_tolerance=measure_relative_tolerance,
            maximum_work=maximum_measure_work,
            maximum_subcells=maximum_measure_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
        )
        target_measures, target_errors, _ = _certified_cell_measures(
            target_mesh,
            target_geometry,
            absolute_tolerance=measure_absolute_tolerance,
            relative_tolerance=measure_relative_tolerance,
            maximum_work=maximum_measure_work,
            maximum_subcells=maximum_measure_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
        )
    if (
        any(
            np.any(~np.isfinite(values))
            for values in (source_measures, target_measures, source_errors, target_errors)
        )
        or np.any(source_errors < 0)
        or np.any(target_errors < 0)
        or np.any(source_measures <= source_errors)
        or np.any(target_measures <= target_errors)
    ):
        raise ValueError(
            "Physical surface areas must have quantitative strictly positive enclosures."
        )
    cover_ids = tuple(
        None
        if cover is None
        else canonical_fingerprint(
            {
                "domain": cover.domain.domain_id,
                "patches": cover.patches,
                "resolution": cover.resolution,
                "records": array_tree_fingerprint(cover.chart_triangulations),
            }
        )
        for cover in (
            source_chart_cover,
            source_chart_cover if target_chart_cover is None else target_chart_cover,
        )
    )
    identity = canonical_fingerprint(
        {
            "kind": "prepared-surface-chart-deformation",
            "source": source_witness.witness_id,
            "target": target_witness.witness_id,
            "material_root": root_witness.witness_id,
            "chart_covers": cover_ids,
            "reconstruction": reconstruction.reconstruction_id,
            "prepared_source_fidelity": (
                None
                if prepared_source_fidelity is None
                else (
                    prepared_source_fidelity.certificate_id,
                    prepared_source_restriction_id,
                )
            ),
            "root_partitions": tuple(
                tuple(piece.piece_id for piece in item.root_pieces)
                for item in occurrences
            ),
            "overlaps": tuple(
                tuple(piece.piece_id for piece in item.pieces) for item in occurrences
            ),
            "source_bounds": array_tree_fingerprint(source_bounds),
            "target_bounds": array_tree_fingerprint(target_bounds),
            "source_measures": array_tree_fingerprint(source_measures),
            "target_measures": array_tree_fingerprint(target_measures),
            "source_errors": array_tree_fingerprint(source_errors),
            "target_errors": array_tree_fingerprint(target_errors),
            "embedding": tuple(item.certificate_id for item in embeddings),
        }
    )
    return PreparedSurfaceChartDeformation(
        source_witness=source_witness,
        target_witness=target_witness,
        material_root_witness=root_witness,
        material_root_chart_cover=source_chart_cover,
        target_chart_cover=source_chart_cover
        if target_chart_cover is None
        else target_chart_cover,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        occurrences=tuple(occurrences),
        source_geometry_id=source_witness.geometry_id,
        target_geometry_id=target_witness.geometry_id,
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        source_fidelity_bounds=jnp.asarray(source_bounds, dtype=jnp.float64),
        target_fidelity_bounds=jnp.asarray(target_bounds, dtype=jnp.float64),
        source_polynomial_chord_bounds=jnp.asarray(polynomial, dtype=jnp.float64),
        source_corner_enclosure_bounds=jnp.asarray(corner, dtype=jnp.float64),
        source_continuous_chord_bounds=jnp.asarray(chord, dtype=jnp.float64),
        source_cell_measures=jnp.asarray(source_measures, dtype=jnp.float64),
        target_cell_measures=jnp.asarray(target_measures, dtype=jnp.float64),
        source_measure_errors=jnp.asarray(source_errors, dtype=jnp.float64),
        target_measure_errors=jnp.asarray(target_errors, dtype=jnp.float64),
        source_validity=validities[0],
        target_validity=validities[1],
        source_embedding=embeddings[0],
        target_embedding=embeddings[1],
        maximum_displacement_bound=maximum_bound,
        evaluation_count=work,
        deformation_id=identity,
    )


__all__ = [
    "SurfaceChartWitness",
    "PreparedSurfaceChartPiece",
    "PreparedSurfaceChartOccurrence",
    "PreparedSurfaceChartDeformation",
    "prepare_surface_chart_deformation",
]
