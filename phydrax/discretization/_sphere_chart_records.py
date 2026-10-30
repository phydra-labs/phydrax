from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, Identifier
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import cell_geometry_id, CellValidityCertificate
from ._cell_mesh import CellMesh


if TYPE_CHECKING:
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ..geometry._sphere_material_atlas import (
        SphereMaterialCellAtlas,
        SphereMaterialFidelityCertificate,
        SphereProjectiveReferenceMap,
        SphereProjectiveTriangleBounds,
    )
else:

    @runtime_checkable
    class _SphereCertificateBinding(Protocol):
        def require(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None: ...

    @runtime_checkable
    class _SphereFidelityBinding(Protocol):
        def require(
            self, mesh: CellMesh, geometry: CellGeometrySpec, source: object, /
        ) -> None: ...

    @runtime_checkable
    class _GlobalEmbeddingCertificate(Protocol):
        binding: _SphereCertificateBinding

        certificate_id: str
        validity_certificate_id: str
        status: str

    @runtime_checkable
    class _SphereMaterialFidelityCertificate(Protocol):
        binding: _SphereFidelityBinding

        certificate_id: str
        status: str

    @runtime_checkable
    class _SphereProjectiveReferenceMap(Protocol):
        exact_coefficients: tuple[tuple[Fraction, ...], ...]
        source_atlas_id: str
        target_atlas_id: str
        source_cell_global_id: int
        target_cell_global_id: int
        geometry_entity_id: str
        occurrence_path: tuple[str, ...]
        orientation_ratio: Fraction

        def evaluate(
            self, point: tuple[Fraction, Fraction], /
        ) -> tuple[Fraction, Fraction]: ...

        map_id: str

    @runtime_checkable
    class _SphereProjectiveTriangleBounds(Protocol):
        denominator_lower: Fraction
        denominator_upper: Fraction
        jacobian_lower: Fraction
        jacobian_upper: Fraction

    @runtime_checkable
    class _SphereMaterialCellAtlas(Protocol):
        source_profile: object
        source_mesh: CellMesh
        source_geometry: CellGeometrySpec
        source_fidelity_certificate: _SphereMaterialFidelityCertificate
        source_validity: CellValidityCertificate
        source_embedding: _GlobalEmbeddingCertificate
        cell_global_ids: Array
        physical_rows: Array
        exact_directions: tuple[tuple[tuple[Fraction, ...], ...], ...]
        geometry_entity_id: str
        occurrence_path: tuple[str, ...]
        source_geometry_id: str
        source_topology_id: str
        domain_id: str
        coverage_status: str
        atlas_id: str

        def cell_row(self, cell_global_id: int, /) -> int: ...

        def require_bound(
            self, mesh: CellMesh, geometry: CellGeometrySpec, /
        ) -> None: ...

    GlobalEmbeddingCertificate = _GlobalEmbeddingCertificate
    SphereMaterialCellAtlas = _SphereMaterialCellAtlas
    SphereMaterialFidelityCertificate = _SphereMaterialFidelityCertificate
    SphereProjectiveReferenceMap = _SphereProjectiveReferenceMap
    SphereProjectiveTriangleBounds = _SphereProjectiveTriangleBounds

_Point = tuple[Fraction, Fraction]
_Triangle = tuple[_Point, _Point, _Point]


def _signed_area(polygon: tuple[_Point, ...], /) -> Fraction:
    return (
        sum(
            (
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


def _map_triangle(
    projective_map: SphereProjectiveReferenceMap, triangle: _Triangle, /
) -> _Triangle:
    coefficients = tuple(
        tuple(Fraction(value) for value in row)
        for row in projective_map.exact_coefficients
    )
    if len(coefficients) != 3 or any(len(row) != 3 for row in coefficients):
        raise ValueError(
            "Sphere projective maps require a nonsingular exact 3-by-3 action."
        )
    mapped = []
    for x, y in triangle:
        barycentric = (Fraction(1) - x - y, x, y)
        homogeneous = tuple(
            sum(
                (coefficients[row][column] * barycentric[column] for column in range(3)),
                Fraction(0),
            )
            for row in range(3)
        )
        total = sum(homogeneous, Fraction(0))
        if not total:
            raise ValueError("Sphere projective map has a zero homogeneous denominator.")
        mapped.append((homogeneous[1] / total, homogeneous[2] / total))
    return (mapped[0], mapped[1], mapped[2])


def _cross(first: _Point, second: _Point, point: _Point, /) -> Fraction:
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )


def _triangle_intersection_area(first: _Triangle, second: _Triangle, /) -> Fraction:
    polygon = list(first)
    orientation = 1 if _signed_area(second) > 0 else -1
    for edge_start, edge_end in zip(second, (*second[1:], second[0]), strict=True):
        if not polygon:
            return Fraction(0)
        output: list[_Point] = []
        previous = polygon[-1]
        previous_distance = orientation * _cross(edge_start, edge_end, previous)
        for current in polygon:
            current_distance = orientation * _cross(edge_start, edge_end, current)
            if (current_distance >= 0) != (previous_distance >= 0):
                ratio = previous_distance / (previous_distance - current_distance)
                output.append(
                    (
                        previous[0] + ratio * (current[0] - previous[0]),
                        previous[1] + ratio * (current[1] - previous[1]),
                    )
                )
            if current_distance >= 0:
                output.append(current)
            previous, previous_distance = current, current_distance
        polygon = []
        for point in output:
            if not polygon or point != polygon[-1]:
                polygon.append(point)
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon.pop()
    return abs(_signed_area(tuple(polygon))) if len(polygon) >= 3 else Fraction(0)


def _inside_reference(point: _Point, /) -> bool:
    return point[0] >= 0 and point[1] >= 0 and point[0] + point[1] <= 1


class PreparedSphereChartPiece(NamedTuple):
    source_cell_global_id: int
    target_cell_global_id: int
    source_row: int
    target_row: int
    source_atlas_row: int
    target_atlas_row: int
    geometry_entity_id: str
    occurrence_path: tuple[str, ...]
    projective_map: SphereProjectiveReferenceMap
    exact_source_reference_vertices: _Triangle
    exact_target_reference_vertices: _Triangle
    source_reference_vertices: Array
    target_reference_vertices: Array
    projective_bounds: SphereProjectiveTriangleBounds
    source_reference_area: Fraction
    target_reference_area: Fraction
    native_boundary_labels: tuple[int, ...]
    native_polygon_vertex_indices: tuple[int, int, int]
    native_clip_id: str
    displacement_bound: float
    piece_id: str


class PreparedSphereChartOccurrence(NamedTuple):
    geometry_entity_id: str
    occurrence_path: tuple[str, ...]
    piece_indices: tuple[int, ...]
    source_cell_global_ids: tuple[int, ...]
    target_cell_global_ids: tuple[int, ...]
    coverage_status: str


class PreparedSphereChartDeformation(StrictModule, NonTrainableState):
    """Exact full-sphere material overlap and endpoint-bound map deformation."""

    __strict_contract__ = True

    source_atlas: SphereMaterialCellAtlas
    target_atlas: SphereMaterialCellAtlas
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    source_geometry_id: Identifier = eqx.field(static=True)
    target_geometry_id: Identifier = eqx.field(static=True)
    source_topology_id: Identifier = eqx.field(static=True)
    target_topology_id: Identifier = eqx.field(static=True)
    domain_id: Identifier = eqx.field(static=True)
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
    source_fidelity_certificate: SphereMaterialFidelityCertificate
    target_fidelity_certificate: SphereMaterialFidelityCertificate
    maximum_displacement_bound: float = eqx.field(static=True)
    deformation_id: Identifier = eqx.field(static=True)

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_domain_coverage(self) -> str:
        return "certified" if self._coverage_is_complete(target=False) else "unresolved"

    @property
    def target_domain_coverage(self) -> str:
        return "certified" if self._coverage_is_complete(target=True) else "unresolved"

    def _coverage_is_complete(self, *, target: bool) -> bool:
        atlas = self.target_atlas if target else self.source_atlas
        areas = (
            self.target_reference_coverage if target else self.source_reference_coverage
        )
        return (
            atlas.coverage_status == "certified"
            and bool(areas)
            and all(value == Fraction(1, 2) for value in areas)
        )

    @checked
    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None:
        from ..geometry._global_embedding import (
            GlobalEmbeddingCertificate as EmbeddingCertificate,
        )
        from ..geometry._sphere_material_atlas import (
            SphereMaterialCellAtlas as MaterialAtlas,
            SphereMaterialFidelityCertificate as FidelityCertificate,
        )

        if not isinstance(self.source_atlas, MaterialAtlas) or not isinstance(
            self.target_atlas, MaterialAtlas
        ):
            raise TypeError("Sphere deformation atlases require SphereMaterialCellAtlas.")
        if not isinstance(self.source_embedding, EmbeddingCertificate) or not isinstance(
            self.target_embedding, EmbeddingCertificate
        ):
            raise TypeError(
                "Sphere deformation embeddings require GlobalEmbeddingCertificate."
            )
        if not isinstance(
            self.source_fidelity_certificate, FidelityCertificate
        ) or not isinstance(self.target_fidelity_certificate, FidelityCertificate):
            raise TypeError(
                "Sphere deformation fidelity requires SphereMaterialFidelityCertificate."
            )
        if (
            source_mesh.mesh_id,
            target_mesh.mesh_id,
            cell_geometry_id(source_geometry),
            cell_geometry_id(target_geometry),
            source_mesh.topology_id,
            target_mesh.topology_id,
        ) != (
            self.source_mesh.mesh_id,
            self.target_mesh.mesh_id,
            self.source_geometry_id,
            self.target_geometry_id,
            self.source_topology_id,
            self.target_topology_id,
        ):
            raise ValueError(
                "Sphere deformation is stale for these endpoint meshes or maps."
            )
        if not self.pieces or not self.occurrences:
            raise ValueError(
                "Sphere deformation requires exact material pieces and occurrences."
            )
        source_ids = tuple(
            int(identifier)
            for block in source_mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        )
        target_ids = tuple(
            int(identifier)
            for block in target_mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        )
        source_coverage = [Fraction(0) for _ in source_ids]
        target_coverage = [Fraction(0) for _ in target_ids]
        source_atlas_ids = np.asarray(self.source_atlas.cell_global_ids, dtype=np.int64)
        target_atlas_ids = np.asarray(self.target_atlas.cell_global_ids, dtype=np.int64)
        source_atlas_rows = np.asarray(self.source_atlas.physical_rows, dtype=np.int64)
        target_atlas_rows = np.asarray(self.target_atlas.physical_rows, dtype=np.int64)
        source_triangles: dict[int, list[_Triangle]] = {}
        target_triangles: dict[int, list[_Triangle]] = {}
        seen_piece_ids: set[str] = set()
        for piece in self.pieces:
            if (
                not 0 <= piece.source_row < len(source_ids)
                or not 0 <= piece.target_row < len(target_ids)
                or not 0 <= piece.source_atlas_row < len(source_atlas_ids)
                or not 0 <= piece.target_atlas_row < len(target_atlas_ids)
                or int(source_ids[piece.source_row]) != piece.source_cell_global_id
                or int(target_ids[piece.target_row]) != piece.target_cell_global_id
                or int(source_atlas_ids[piece.source_atlas_row])
                != piece.source_cell_global_id
                or int(target_atlas_ids[piece.target_atlas_row])
                != piece.target_cell_global_id
                or int(source_atlas_rows[piece.source_atlas_row]) != piece.source_row
                or int(target_atlas_rows[piece.target_atlas_row]) != piece.target_row
                or piece.piece_id in seen_piece_ids
            ):
                raise ValueError(
                    "Sphere material piece changes or duplicates physical cell identity."
                )
            seen_piece_ids.add(piece.piece_id)
            projective = piece.projective_map
            if (
                projective.source_atlas_id != self.source_atlas.atlas_id
                or projective.target_atlas_id != self.target_atlas.atlas_id
                or projective.source_cell_global_id != piece.source_cell_global_id
                or projective.target_cell_global_id != piece.target_cell_global_id
                or projective.geometry_entity_id != piece.geometry_entity_id
                or projective.occurrence_path != piece.occurrence_path
                or not projective.orientation_ratio
            ):
                raise ValueError(
                    "Sphere projective map changes atlas or source occurrence identity."
                )
            bounds = piece.projective_bounds
            for lower, upper in (
                (bounds.denominator_lower, bounds.denominator_upper),
                (bounds.jacobian_lower, bounds.jacobian_upper),
            ):
                if lower <= 0 <= upper:
                    raise ValueError(
                        "Sphere projective map is singular on a retained overlap piece."
                    )
            mapped = _map_triangle(projective, piece.exact_source_reference_vertices)
            if mapped != piece.exact_target_reference_vertices:
                raise ValueError(
                    "Sphere projective map changes its exact target reference triangle."
                )
            source_triangle = piece.exact_source_reference_vertices
            target_triangle = piece.exact_target_reference_vertices
            if any(
                not _inside_reference(point)
                for point in (*source_triangle, *target_triangle)
            ):
                raise ValueError("Sphere material piece leaves a reference simplex.")
            source_area = abs(_signed_area(source_triangle))
            target_area = abs(_signed_area(target_triangle))
            if (
                not source_area
                or not target_area
                or source_area != piece.source_reference_area
                or target_area != piece.target_reference_area
                or not np.isfinite(piece.displacement_bound)
                or piece.displacement_bound < 0.0
                or not piece.native_clip_id
                or not piece.native_boundary_labels
                or any(label < 0 for label in piece.native_boundary_labels)
                or piece.native_polygon_vertex_indices[0] != 0
                or piece.native_polygon_vertex_indices[2]
                != piece.native_polygon_vertex_indices[1] + 1
                or not 1
                <= piece.native_polygon_vertex_indices[1]
                < len(piece.native_boundary_labels)
            ):
                raise ValueError(
                    "Sphere material piece has invalid exact measures, clipping ancestry, or displacement."
                )
            source_triangles.setdefault(piece.source_row, []).append(source_triangle)
            target_triangles.setdefault(piece.target_row, []).append(target_triangle)
            source_coverage[piece.source_row] += source_area
            target_coverage[piece.target_row] += target_area
        for groups in (source_triangles, target_triangles):
            for triangles in groups.values():
                for first in range(len(triangles)):
                    for second in range(first + 1, len(triangles)):
                        if _triangle_intersection_area(
                            triangles[first], triangles[second]
                        ):
                            raise ValueError(
                                "Sphere material pieces overlap in a reference cell."
                            )
        if (
            tuple(source_coverage) != self.source_reference_coverage
            or tuple(target_coverage) != self.target_reference_coverage
        ):
            raise ValueError(
                "Sphere reference coverage bank disagrees with its exact pieces."
            )
        if any(value != Fraction(1, 2) for value in (*source_coverage, *target_coverage)):
            raise ValueError(
                "Sphere material pieces do not completely cover every reference cell."
            )
        occurrence_piece_indices: set[int] = set()
        for occurrence in self.occurrences:
            indices = occurrence.piece_indices
            if (
                not occurrence.geometry_entity_id
                or any(
                    not isinstance(value, str) or not value
                    for value in occurrence.occurrence_path
                )
                or not indices
                or len(set(indices)) != len(indices)
                or any(index < 0 or index >= len(self.pieces) for index in indices)
                or occurrence_piece_indices.intersection(indices)
            ):
                raise ValueError(
                    "Sphere occurrence rows must form a unique partition of overlap pieces."
                )
            occurrence_piece_indices.update(indices)
            occurrence_pieces = tuple(self.pieces[index] for index in indices)
            if tuple(
                sorted({piece.source_cell_global_id for piece in occurrence_pieces})
            ) != tuple(sorted(occurrence.source_cell_global_ids)) or tuple(
                sorted({piece.target_cell_global_id for piece in occurrence_pieces})
            ) != tuple(sorted(occurrence.target_cell_global_ids)):
                raise ValueError(
                    "Sphere occurrence cell IDs disagree with their retained pieces."
                )
            for piece in occurrence_pieces:
                if (
                    piece.geometry_entity_id != occurrence.geometry_entity_id
                    or piece.occurrence_path != occurrence.occurrence_path
                ):
                    raise ValueError(
                        "Sphere occurrence changes a piece's source entity or occurrence path."
                    )
        if occurrence_piece_indices != set(range(len(self.pieces))):
            raise ValueError(
                "Sphere occurrences do not cover every retained overlap piece."
            )
        if not self._coverage_is_complete(target=False) or not self._coverage_is_complete(
            target=True
        ):
            raise ValueError(
                "Sphere material atlases do not certify complete full-sphere coverage."
            )
        for atlas, mesh, geometry in (
            (self.source_atlas, source_mesh, source_geometry),
            (self.target_atlas, target_mesh, target_geometry),
        ):
            if atlas.domain_id != self.domain_id:
                raise ValueError(
                    "Sphere material atlas changes its retained source domain."
                )
            atlas.require_bound(mesh, geometry)
        for mesh, geometry, validity, embedding, fidelity, atlas in (
            (
                source_mesh,
                source_geometry,
                self.source_validity,
                self.source_embedding,
                self.source_fidelity_certificate,
                self.source_atlas,
            ),
            (
                target_mesh,
                target_geometry,
                self.target_validity,
                self.target_embedding,
                self.target_fidelity_certificate,
                self.target_atlas,
            ),
        ):
            validity.require_bound(geometry, mesh=mesh)
            embedding.binding.require(mesh, geometry)
            fidelity.binding.require(mesh, geometry, atlas.source_profile)
            if embedding.status != "certified" or fidelity.status != "certified":
                raise ValueError(
                    "Sphere deformation requires certified embedding and source fidelity."
                )
        for values in (
            self.source_fidelity_bounds,
            self.target_fidelity_bounds,
            self.source_cell_measures,
            self.target_cell_measures,
            self.source_measure_errors,
            self.target_measure_errors,
        ):
            host = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(host)) or np.any(host < 0.0):
                raise ValueError(
                    "Sphere fidelity and measure evidence must be finite and nonnegative."
                )
        if (
            not np.isfinite(self.maximum_displacement_bound)
            or self.maximum_displacement_bound < 0.0
        ):
            raise ValueError("Sphere displacement bound must be finite and nonnegative.")
        if (
            max(piece.displacement_bound for piece in self.pieces)
            > self.maximum_displacement_bound
        ):
            raise ValueError(
                "Sphere deformation bound is smaller than a piecewise displacement."
            )
        self.source_geometry.resolve(source_mesh)
        self.target_geometry.resolve(target_mesh)
        expected = canonical_fingerprint(
            {
                "kind": "prepared-sphere-chart-deformation",
                "source_atlas": getattr(self.source_atlas, "atlas_id", None),
                "target_atlas": getattr(self.target_atlas, "atlas_id", None),
                "source_geometry": self.source_geometry_id,
                "target_geometry": self.target_geometry_id,
                "source_mesh": source_mesh.mesh_id,
                "target_mesh": target_mesh.mesh_id,
                "domain": self.domain_id,
                "pieces": tuple(piece.piece_id for piece in self.pieces),
                "certificates": (
                    self.source_validity.certificate_id,
                    self.target_validity.certificate_id,
                    self.source_embedding.certificate_id,
                    self.target_embedding.certificate_id,
                    self.source_fidelity_certificate.certificate_id,
                    self.target_fidelity_certificate.certificate_id,
                ),
                "source_fidelity": self.source_fidelity_bounds,
                "target_fidelity": self.target_fidelity_bounds,
                "source_cell_measures": array_tree_fingerprint(self.source_cell_measures),
                "target_cell_measures": array_tree_fingerprint(self.target_cell_measures),
                "source_measure_errors": array_tree_fingerprint(
                    self.source_measure_errors
                ),
                "target_measure_errors": array_tree_fingerprint(
                    self.target_measure_errors
                ),
                "occurrences": tuple(
                    (
                        item.geometry_entity_id,
                        item.occurrence_path,
                        item.piece_indices,
                        item.source_cell_global_ids,
                        item.target_cell_global_ids,
                    )
                    for item in self.occurrences
                ),
                "displacement": self.maximum_displacement_bound,
            }
        )
        if expected != self.deformation_id:
            raise ValueError("Retained sphere deformation identity is stale.")

    @classmethod
    def from_stored_fields(
        cls, value: PreparedSphereChartDeformation, /
    ) -> PreparedSphereChartDeformation:
        """Cold-rebuild both atlases and their exact common refinement."""
        from ..geometry._sphere_material_atlas import (
            prepare_sphere_chart_deformation,
            SphereMaterialCellAtlas,
        )

        if not isinstance(value, PreparedSphereChartDeformation):
            raise TypeError("value must be PreparedSphereChartDeformation.")
        source = SphereMaterialCellAtlas.from_stored_fields(value.source_atlas)
        target = SphereMaterialCellAtlas.from_stored_fields(value.target_atlas)
        rebuilt = prepare_sphere_chart_deformation(source, target)
        if rebuilt.deformation_id != value.deformation_id:
            raise ValueError(
                "Stored sphere deformation does not reproduce its exact chart overlap."
            )
        return rebuilt


__all__ = [
    "PreparedSphereChartDeformation",
    "PreparedSphereChartOccurrence",
    "PreparedSphereChartPiece",
]
