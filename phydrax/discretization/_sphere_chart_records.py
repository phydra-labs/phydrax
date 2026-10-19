#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Store chart correspondence data and validate its retained scientific bindings."""

from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction

import equinox as eqx
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..geometry._sphere_material_atlas import (
    SphereMaterialCellAtlas,
    SphereProjectiveReferenceMap,
    SphereProjectiveTriangleBounds,
)
from ..typing import checked
from . import _chart_geometry as chart_geometry
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_transition_types import MaterialChartDeformation
from ._cell_geometry_validity import cell_geometry_id, CellValidityCertificate
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget


type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _FloatArray = NDArray[np.float64]


class PreparedSphereChartPiece(StrictModule, NonTrainableState):
    """source-reference triangle with its rational target map."""

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
    """Unmerged scientific occurrence and its whole overlap pieces."""

    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    piece_indices: tuple[int, ...] = eqx.field(static=True)
    source_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    target_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    coverage_status: str = eqx.field(static=True)


class PreparedSphereChartDeformation(MaterialChartDeformation, NonTrainableState):
    """Accepted full-sphere transition, never a latitude/longitude alias."""

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
            and all((area == Fraction(1, 2) for area in self.source_reference_coverage))
            else "unresolved"
        )

    @property
    def target_domain_coverage(self) -> str:
        return (
            "certified"
            if self.target_atlas.coverage_status == "certified"
            and all((area == Fraction(1, 2) for area in self.target_reference_coverage))
            else "unresolved"
        )

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
                "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere correspondence is stale for these coordinate maps or cells."
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
                    "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere correspondence requires valid globally embedded coordinate maps."
                )
        if (
            self.source_domain_coverage != "certified"
            or self.target_domain_coverage != "certified"
        ):
            raise ValueError(
                "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere correspondence lacks whole source and target partitions."
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
                    "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere piece physical rows do not retain their scientific cell identities."
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
                    "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere piece rational map or occurrence changes its scientific endpoints."
                )
            with budget.temporary_scope() if budget is not None else nullcontext():
                piece.projective_map.require_bound(
                    self.source_atlas, self.target_atlas, budget=budget
                )
            bounds = piece.projective_map.certify_triangle(
                piece.exact_source_reference_vertices
            )
            vertices = piece.exact_source_reference_vertices
            target: _Triangle = (
                chart_geometry.projective_point(
                    piece.projective_map.exact_coefficients,
                    vertices[0],
                    invalid_message="sphere_chart.project.pieces must remain inside the positive reference cone; got nonpositive denominator or negative barycentric coordinate.",
                ),
                chart_geometry.projective_point(
                    piece.projective_map.exact_coefficients,
                    vertices[1],
                    invalid_message="sphere_chart.project.pieces must remain inside the positive reference cone; got nonpositive denominator or negative barycentric coordinate.",
                ),
                chart_geometry.projective_point(
                    piece.projective_map.exact_coefficients,
                    vertices[2],
                    invalid_message="sphere_chart.project.pieces must remain inside the positive reference cone; got nonpositive denominator or negative barycentric coordinate.",
                ),
            )
            if (
                target != piece.exact_target_reference_vertices
                or bounds != piece.projective_bounds
                or piece.source_reference_area
                != chart_geometry.area(piece.exact_source_reference_vertices)
                or (
                    piece.target_reference_area
                    != chart_geometry.area(piece.exact_target_reference_vertices)
                )
            ):
                raise ValueError(
                    "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere piece changes its reference construction or rational map."
                )
            if not np.array_equal(
                np.asarray(piece.source_reference_vertices),
                reference_representatives(piece.exact_source_reference_vertices),
            ) or not np.array_equal(
                np.asarray(piece.target_reference_vertices),
                reference_representatives(piece.exact_target_reference_vertices),
            ):
                raise ValueError(
                    "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere piece representatives no longer match the constructions."
                )
            source_coverage[source_row] += piece.source_reference_area
            target_coverage[target_row] += piece.target_reference_area
        if (
            tuple(source_coverage) != self.source_reference_coverage
            or tuple(target_coverage) != self.target_reference_coverage
        ):
            raise ValueError(
                "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere pieces no longer provide their whole coverage evidence."
            )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PreparedSphereChartDeformation],
        value: PreparedSphereChartDeformation,
        /,
    ) -> PreparedSphereChartDeformation:
        """Validate retained endpoints without rebuilding numerical chart data."""
        value.require_bound(
            value.source_mesh,
            value.source_geometry,
            value.target_mesh,
            value.target_geometry,
        )
        return value


def reference_representatives(vertices: _Triangle) -> _FloatArray:
    return np.asarray(
        [[float(value) for value in point] for point in vertices], dtype=np.float64
    )


class SphereGeometryReconstruction(StrictModule, NonTrainableState):
    """degree-preserving radial-source reconstruction, without UV ghost cells."""

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

    @checked
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
                "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction is stale for its source, target or material revision."
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
            or (
                not np.array_equal(
                    np.asarray(self.vertex_coordinates),
                    np.asarray(target_mesh.coordinates),
                )
            )
            or (
                not np.array_equal(
                    np.asarray(self.fidelity_bounds),
                    physical_order(
                        self.target_atlas, self.target_atlas.source_fidelity_bounds
                    ),
                )
            )
        ):
            raise ValueError(
                "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction metadata is stale for its full-map material cells."
            )
        self.target_validity.require_bound(self.geometry, mesh=target_mesh)
        self.target_embedding.binding.require(target_mesh, self.geometry)
        if (
            not self.target_validity.all_certified
            or self.target_embedding.status != "certified"
        ):
            raise ValueError(
                "sphere_chart.require_bound.validity must satisfy the positive cell validity contract; got Sphere reconstruction lacks validity and global embedding."
            )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SphereGeometryReconstruction], value: SphereGeometryReconstruction, /
    ) -> SphereGeometryReconstruction:
        """Validate the stored target atlas and coordinate identities."""
        mesh = value.target_atlas.source_mesh
        value.target_atlas.require_bound(
            value.target_atlas.domain,
            mesh,
            value.geometry,
            value.target_atlas.coordinate_contract,
        )
        value.target_validity.require_bound(value.geometry, mesh=mesh)
        value.target_embedding.binding.require(mesh, value.geometry)
        if (
            value.target_geometry_id != cell_geometry_id(value.geometry)
            or value.target_topology_id != mesh.topology_id
        ):
            raise ValueError(
                "sphere_reconstruction.binding must retain its coordinate map and topology; got inconsistent restored identity"
            )
        return value


def physical_order(atlas: SphereMaterialCellAtlas, values: Array) -> _FloatArray:
    result = np.empty(atlas.num_charts, dtype=np.float64)
    result[np.asarray(atlas.physical_rows, dtype=np.int64)] = np.asarray(
        values, dtype=np.float64
    )
    return result
