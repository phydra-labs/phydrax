#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Support geometry and location status shared by discrete field views.

Structured reconstructions cover a tensor box; affine simplicial meshes derive
their region from the mesh; an explicit geometry is admitted only with evidence
that the reconstruction covers it.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from math import factorial, isfinite
from typing import Any, final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import ConvertibleToArray
from ._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ._cell_geometry import (
    CellGeometryCoordinateMap,
    CellGeometryElement,
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
)
from ._cell_geometry_validity import cell_geometry_id
from ._cell_locator import AbstractCellLocator, CellLocationResult, CellLocationStatus
from ._cell_mesh import CellBlock, CellMesh, SimplicialConnectivity
from ._hexahedral import HexahedralConnectivity
from ._mapped_locator import PreparedMappedCellLocator, PreparedOwnerLocalMappedCellMap
from ._polyhedral_locator import PreparedPolyhedralCellLocator, PreparedPolyhedralCellMap
from ._reference_cell import reference_cell_topology
from ._simplicial_locator import PreparedSimplicialCellLocator
from ._views import FieldQueryStatus


if TYPE_CHECKING:
    from ..geometry._domain_certification import DomainCoverageCertificate
    from ..geometry._mapped_reference_domain import MappedReferenceDomain
    from ..graph._partition import GraphPartitionPlan
    from ..meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
    from ..meshing._device_adaptation import OwnerLocalPeriodicAdaptiveResult
    from ..meshing._distribution import (
        OwnerLocalGraphMigrationEvidence,
        ShardedOwnerDirectoryData,
    )
    from ..meshing._lineage import VertexInterpolationStencil
    from ..meshing._scope import MeshingScope
    from ._exact_power_geometry import ExactPowerCellGeometrySource
    from ._periodic_cell import PeriodicCell

_COMPACT_OWNER_LOCAL_RESTORE = object()


class OwnerLocalCellBlock(StrictModule, NonTrainableState):
    """One compact, source-bound block in an owner-local cell geometry view."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    cell_rows: Array
    cell_ids: Array
    vertices: Array
    vertex_valid: Array
    coordinate_routes: Array
    element: CellGeometryElement
    element_rows: Array
    source_cell_ids: Array
    chart_arguments: tuple[Any, ...] | None
    cell_measures: Array
    polyhedron_face_ids: Array | None
    polyhedron_face_vertex_loops: tuple[Array, ...] | None
    polyhedron_faces: tuple[tuple[Array, ...], ...] | None
    polyhedron_face_signs: tuple[Array, ...] | None
    cell_vertex_shifts: Array | None

    def __init__(self, record: Mapping[str, Any], /) -> None:
        from ._nested_reference_types import reference_chart_arguments_identity

        name = str(record["name"])
        kind = str(record["cell_kind"])
        cell_rows = np.asarray(record["cell_rows"], dtype=np.int64)
        cell_ids = np.asarray(record["cell_ids"], dtype=np.int64)
        vertices = np.asarray(record["vertices"], dtype=np.int32)
        vertex_valid = np.asarray(record["vertex_valid"], dtype=np.bool_)
        routes = np.asarray(record["coordinate_routes"], dtype=np.int32)
        element = record["element"]
        if not isinstance(element, CellGeometryElement):
            raise TypeError("Owner-local cell blocks require a canonical cell element.")
        element_id = element.element_id
        if not isinstance(element_id, str) or not element_id:
            raise ValueError(
                "Owner-local cell blocks require a canonical element identity."
            )
        element_rows = np.asarray(record["element_rows"], dtype=np.int32)
        source_cell_ids = np.asarray(
            record.get("source_cell_ids", cell_ids), dtype=np.int64
        )
        raw_cell_vertex_shifts = record.get("cell_vertex_shifts")
        cell_vertex_shifts = (
            None
            if raw_cell_vertex_shifts is None
            else np.asarray(raw_cell_vertex_shifts, dtype=np.int64)
        )
        if cell_vertex_shifts is not None and (
            cell_vertex_shifts.ndim != 3 or cell_vertex_shifts.shape[:2] != vertices.shape
        ):
            raise ValueError(
                "Owner-local vertex image shifts do not match cell connectivity."
            )
        measures = np.asarray(record["cell_measures"], dtype=np.float64)
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            chart_arguments = tuple(
                element.chart_arguments(row) for row in range(cell_rows.size)
            )
            if (
                tuple(int(value) for value in element.target_cell_ids)
                != tuple(int(value) for value in cell_ids)
                or tuple(int(value) for value in element.source_cell_ids)
                != tuple(int(value) for value in source_cell_ids)
                or not np.array_equal(element_rows, np.arange(cell_rows.size))
            ):
                raise ValueError(
                    "Owner-local restricted charts lost stable source ancestry."
                )
            supplied_charts = record.get("chart_arguments")
            if supplied_charts is not None and reference_chart_arguments_identity(
                tuple(supplied_charts)
            ) != reference_chart_arguments_identity(chart_arguments):
                raise ValueError(
                    "Owner-local chart arguments differ from their compact element."
                )
        else:
            chart_arguments = record.get("chart_arguments")
            if chart_arguments is not None:
                raise ValueError(
                    "Only compact per-cell chart elements may carry chart arguments."
                )
        if (
            not name
            or not kind
            or cell_rows.ndim != 1
            or cell_ids.shape != cell_rows.shape
            or element_rows.shape != cell_rows.shape
            or source_cell_ids.shape != cell_rows.shape
            or measures.shape != cell_rows.shape
            or vertices.ndim != 2
            or vertices.shape[0] != cell_rows.size
            or vertex_valid.shape != vertices.shape
            or routes.ndim != 2
            or routes.shape[0] != cell_rows.size
            or np.any(cell_rows < 0)
            or np.any(element_rows < 0)
            or np.any(~np.isfinite(measures) | (measures <= 0.0))
            or np.any(vertices[~vertex_valid] != -1)
            or np.any(vertices[vertex_valid] < 0)
            or np.any(routes < -1)
        ):
            raise ValueError("Owner-local cell block arrays are malformed.")
        if element.cell_kind != kind:
            raise ValueError(
                "Owner-local coordinate element does not match its cell block."
            )
        if kind == "polyhedron":
            face_ids = np.asarray(record["face_ids"], dtype=np.int64)
            cell_face_offsets = np.asarray(record["cell_face_offsets"], dtype=np.int64)
            cell_face_values = np.asarray(record["cell_face_values"], dtype=np.int64)
            cell_face_signs = np.asarray(record["cell_face_signs"], dtype=np.int8)
            face_vertex_offsets = np.asarray(
                record["face_vertex_offsets"], dtype=np.int64
            )
            face_vertex_values = np.asarray(record["face_vertex_values"], dtype=np.int32)
            polyhedron_face_vertex_loops = tuple(
                jnp.asarray(
                    face_vertex_values[
                        face_vertex_offsets[face] : face_vertex_offsets[face + 1]
                    ]
                )
                for face in range(face_ids.size)
            )
            if (
                face_ids.ndim != 1
                or np.unique(face_ids).size != face_ids.size
                or cell_face_offsets.shape != (cell_rows.size + 1,)
                or cell_face_offsets[0] != 0
                or np.any(np.diff(cell_face_offsets) < 4)
                or cell_face_offsets[-1] != cell_face_values.size
                or cell_face_signs.shape != cell_face_values.shape
                or face_vertex_offsets.shape != (face_ids.size + 1,)
                or face_vertex_offsets[0] != 0
                or np.any(np.diff(face_vertex_offsets) < 3)
                or face_vertex_offsets[-1] != face_vertex_values.size
                or np.any(cell_face_values < 0)
                or np.any(cell_face_values >= face_ids.size)
                or np.any(~np.isin(cell_face_signs, (-1, 1)))
                or np.any(face_vertex_values < 0)
                or np.any(face_vertex_values >= int(record["local_vertex_count"]))
            ):
                raise ValueError("Owner-local polyhedral face incidence is malformed.")
            faces, signs = [], []
            for cell in range(cell_rows.size):
                start, stop = cell_face_offsets[cell : cell + 2]
                cell_faces = []
                for face in cell_face_values[start:stop]:
                    face_start, face_stop = face_vertex_offsets[face : face + 2]
                    cell_faces.append(
                        jnp.asarray(face_vertex_values[face_start:face_stop])
                    )
                faces.append(tuple(cell_faces))
                signs.append(jnp.asarray(cell_face_signs[start:stop]))
            polyhedron_face_ids = jnp.asarray(face_ids)
            polyhedron_faces, polyhedron_face_signs = tuple(faces), tuple(signs)
        else:
            if any(
                record.get(name) is not None
                for name in (
                    "face_ids",
                    "cell_face_offsets",
                    "cell_face_values",
                    "cell_face_signs",
                    "face_vertex_offsets",
                    "face_vertex_values",
                )
            ):
                raise ValueError(
                    "Only polyhedron blocks may carry authored face incidence."
                )
            polyhedron_face_ids = polyhedron_face_vertex_loops = None
            polyhedron_faces = polyhedron_face_signs = None
        self.name = name
        self.cell_kind = kind
        self.polyhedron_face_vertex_loops = polyhedron_face_vertex_loops
        self.cell_rows, self.cell_ids = jnp.asarray(cell_rows), jnp.asarray(cell_ids)
        self.vertices, self.vertex_valid = (
            jnp.asarray(vertices),
            jnp.asarray(vertex_valid),
        )
        self.coordinate_routes = jnp.asarray(routes)
        self.element, self.element_rows = element, jnp.asarray(element_rows)
        self.chart_arguments = chart_arguments
        self.source_cell_ids = jnp.asarray(source_cell_ids)
        self.cell_measures = jnp.asarray(measures)
        self.cell_vertex_shifts = (
            None if cell_vertex_shifts is None else jnp.asarray(cell_vertex_shifts)
        )
        self.polyhedron_face_ids = polyhedron_face_ids
        self.polyhedron_faces, self.polyhedron_face_signs = (
            polyhedron_faces,
            polyhedron_face_signs,
        )


def _reconstruct_owner_local_geometry(
    vertex_coordinates: Array,
    vertex_ids: Array,
    coordinates: Array,
    blocks: tuple[OwnerLocalCellBlock, ...],
    source_map: CellGeometryCoordinateMap | ExactPowerCellGeometrySource | None,
    periodic_vertex_roots: Array | None,
    periodic_vertex_windings: Array | None,
    /,
) -> tuple[CellMesh | None, CellGeometrySpec | None]:
    from ._exact_power_geometry import ExactPowerCellGeometrySource

    if not any(np.asarray(block.cell_ids).size for block in blocks):
        return None, None
    points = np.asarray(vertex_coordinates, dtype=np.float64)
    identifiers = np.asarray(vertex_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[0] != identifiers.size:
        raise ValueError("Owner-local vertex coordinates and IDs are not aligned.")
    dense_blocks: list[CellBlock] = []
    polyhedra: dict[str, tuple[tuple[np.ndarray, ...], ...]] = {}
    polyhedron_ids: dict[str, np.ndarray] = {}
    for block in blocks:
        if not np.asarray(block.cell_ids).size:
            continue
        if block.cell_kind != "polyhedron":
            dense_blocks.append(
                CellBlock(
                    block.name,
                    block.cell_kind,
                    block.vertices,
                    vertex_valid=block.vertex_valid,
                    global_ids=block.cell_ids,
                )
            )
            continue
        if block.polyhedron_faces is None or block.polyhedron_face_signs is None:
            raise ValueError("Owner-local polyhedron lacks canonical face incidence.")
        oriented_cells = []
        for faces, signs in zip(
            block.polyhedron_faces, block.polyhedron_face_signs, strict=True
        ):
            oriented_cells.append(
                tuple(
                    np.asarray(face, dtype=np.int32)
                    if int(sign) > 0
                    else np.asarray(face, dtype=np.int32)[::-1]
                    for face, sign in zip(faces, np.asarray(signs), strict=True)
                )
            )
        polyhedra[block.name] = tuple(oriented_cells)
        polyhedron_ids[block.name] = np.asarray(block.cell_ids, dtype=np.int64)
    if polyhedra:
        mesh = CellMesh.from_mixed_3d(
            points,
            tuple(dense_blocks),
            polyhedra=polyhedra,
            vertex_global_ids=identifiers,
            polyhedral_cell_global_ids=polyhedron_ids,
        )
    else:
        mesh = CellMesh(
            points,
            tuple(dense_blocks),
            vertex_global_ids=identifiers,
        )
    if isinstance(source_map, ExactPowerCellGeometrySource):
        context = source_map.periodic_source_context
        if context is not None:
            from ..meshing._periodic import PeriodicPointOrbits
            from ._periodic_topology import PeriodicMeshTopology

            periodic_source = context.periodic_source
            if not isinstance(periodic_source, PeriodicPointOrbits):
                raise TypeError(
                    "Periodic power geometry requires its exact point-orbit source."
                )
            if periodic_vertex_roots is None or periodic_vertex_windings is None:
                raise ValueError("Periodic power vertex quotient ancestry is missing.")
            roots = np.asarray(periodic_vertex_roots, dtype=np.int64)
            windings = np.asarray(periodic_vertex_windings, dtype=np.int64)
            if (
                roots.shape != identifiers.shape
                or windings.ndim != 2
                or windings.shape[0] != identifiers.size
                or windings.shape[1] != periodic_source.cell.ambient_dimension
            ):
                raise ValueError("Periodic power vertex quotient ancestry is incomplete.")
            representatives = np.empty(identifiers.shape, dtype=np.int32)
            relative_windings = np.empty_like(windings)
            for root in np.unique(roots):
                rows = np.flatnonzero(roots == root)
                representative = min(
                    (int(row) for row in rows),
                    key=lambda row: (int(identifiers[row]), row),
                )
                representatives[rows] = representative
                relative_windings[rows] = windings[rows] - windings[representative]
            quotient = PeriodicMeshTopology(
                mesh,
                periodic_source.cell,
                representatives,
                relative_windings,
                exact_coordinate_source=source_map,
            )
            connectivity = mesh.connectivity
            mesh = CellMesh(
                mesh.coordinates,
                mesh.blocks,
                vertex_global_ids=mesh.vertex_global_ids,
                polyhedral_connectivity=(
                    connectivity
                    if isinstance(connectivity, PolyhedralConnectivity)
                    else None
                ),
                numeric_version=mesh.numeric_version,
                periodic_topology=quotient,
            )
        if context is not None:
            source_map.require_periodic_mesh(mesh)
        geometry = CellGeometrySpec.power(mesh, source_map)
        return mesh, geometry
    element_map = {block.name: block.element for block in blocks}
    route_map = {
        block.name: np.asarray(block.coordinate_routes, dtype=np.int32)
        for block in blocks
    }
    if source_map is not None:
        if any(np.any(routes < 0) for routes in route_map.values()):
            raise ValueError(
                "Mapped owner-local geometry has incomplete coordinate routes."
            )
        geometry = CellGeometrySpec(
            element_map,
            route_map,
            coordinates,
            source_map=source_map,
            source_mesh=mesh,
        )
    elif any(block.cell_kind == "polyhedron" for block in blocks):
        geometry = CellGeometrySpec.affine(mesh)
    else:
        geometry = CellGeometrySpec(element_map, route_map, coordinates)
    return mesh, geometry


class OwnerLocalCellGeometry(StrictModule, NonTrainableState):
    """Versioned compact source-bound cell geometry owned by one rank.

    ``source_mesh_id`` is the authored source/lattice scope, not the current
    target mesh ID. The compact view stores scientific source/maps and current
    owner rows only; its PyTree and result identities intentionally change.
    """

    part_name: str = eqx.field(static=True)
    part_id: str = eqx.field(static=True)
    local_view_id: str = eqx.field(static=True)
    local_mesh_id: str | None = eqx.field(static=True)
    local_topology_id: str | None = eqx.field(static=True)
    local_geometry_id: str | None = eqx.field(static=True)
    distribution_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_map_id: str | None = eqx.field(static=True)
    source_value: Any
    source_geometry_authority: Any
    owner_directory_data: ShardedOwnerDirectoryData
    owner_rank: int = eqx.field(static=True)
    serial_source_view: bool = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    coordinate_contract_id: str = eqx.field(static=True)
    specification_id: str | None = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract
    topological_dimension: int = eqx.field(static=True)
    cell_rows: Array
    cell_ids: Array
    owned: Array
    cell_measures: Array
    owned_cell_ids: Array
    owned_cell_owner_ranks: Array
    owned_cell_weights: Array
    cell_neighbor_offsets: Array
    cell_neighbor_ids: Array
    cell_neighbor_owner_ranks: Array
    cell_neighbor_weights: Array
    vertex_rows: Array
    vertex_ids: Array
    vertex_coordinates: Array
    exact_vertex_coordinates: Any | None
    coordinate_rows: Array
    coordinates: Array
    exact_coordinates: Any
    source_coordinate_stencil: VertexInterpolationStencil | None
    exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None
    entity_set_ids: tuple[tuple[int, str], ...] = eqx.field(static=True)
    local_entity_ids: tuple[tuple[int, Array], ...]
    entity_vertex_offsets: tuple[tuple[int, Array], ...]
    entity_vertex_values: tuple[tuple[int, Array], ...]
    cell_key_offsets: Array
    cell_key_source_ids: Array
    cell_key_relative_windings: Array
    cell_owner_ranks: Array
    support_cell_mask: Array
    vertex_source_ids: Array
    source_point_ids: Array
    source_owner_ranks: Array
    source_owned: Array
    vertex_owner_ranks: Array
    vertex_absolute_windings: Array
    candidate_revision_id: str | None = eqx.field(static=True)
    certified_periodic_margin: Any | None
    periodic_shell_certificate: Any | None
    adaptive_result: OwnerLocalPeriodicAdaptiveResult | None
    graph_migration_evidence: OwnerLocalGraphMigrationEvidence | None
    periodic_face_coverage_id: str | None = eqx.field(static=True)
    periodic_topology_id: str | None = eqx.field(static=True)
    periodic_cell_ids: Array | None
    vertex_representative_ids: Array | None
    vertex_windings: Array | None
    cell_windings: Array | None
    source_context_id: str | None = eqx.field(static=True)
    periodic_lift_id: str | None = eqx.field(static=True)
    periodic_lift_matrix: Array | None
    coordinates_are_lifted: bool = eqx.field(static=True)
    support_fractional_lower: Array | None
    support_fractional_upper: Array | None
    blocks: tuple[OwnerLocalCellBlock, ...]

    def __init__(self, record: Mapping[str, Any], /) -> None:
        from ..lifecycle import Composition, CompositionEntry, CompositionRebindReceipt
        from ..meshing._distribution import (
            ShardedOwnerDirectoryData,
            ShardedOwnerDirectoryWitness,
        )

        identity_fields = (
            "part_name",
            "part_id",
            "distribution_id",
            "source_id",
            "source_revision",
            "source_mesh_id",
            "source_topology_id",
            "source_geometry_id",
            "geometry_layout_id",
            "coordinate_contract_id",
        )
        identities = {name: str(record[name]) for name in identity_fields}
        if any(not value for value in identities.values()):
            raise ValueError("Owner-local geometry requires complete global identities.")
        specification_id = record.get("specification_id")
        if specification_id is not None and (
            not isinstance(specification_id, str) or not specification_id
        ):
            raise ValueError(
                "Owner-local specification identity must be non-empty or None."
            )
        source_map_id = record.get("source_map_id")
        if source_map_id is not None and (
            not isinstance(source_map_id, str) or not source_map_id
        ):
            raise ValueError("Owner-local source-map identity must be non-empty or None.")
        coordinate_contract = record["coordinate_contract"]
        dimension = record["topological_dimension"]
        if (
            not isinstance(coordinate_contract, SpatialCoordinateContract)
            or coordinate_contract.spatial_id != identities["coordinate_contract_id"]
            or isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, (int, np.integer))
            or int(dimension) not in (2, 3)
        ):
            raise ValueError("Owner-local geometry frame or dimension is invalid.")
        serial_source_view = record.get("serial_source_view", False)
        if not isinstance(serial_source_view, bool):
            raise TypeError("serial_source_view must be an explicit bool.")
        raw_rank = record.get("rank", -1)
        owner_rank = (
            int(raw_rank)
            if isinstance(raw_rank, (int, np.integer))
            and not isinstance(raw_rank, (bool, np.bool_))
            and raw_rank >= 0
            else -1
        )
        cell_ids = np.asarray(record.get("cell_ids", np.empty((0,), dtype=np.int64)))
        owned = np.asarray(record.get("owned", np.empty((0,), dtype=np.bool_)))
        vertex_ids = np.asarray(record.get("vertex_ids", np.empty((0,), dtype=np.int64)))
        cell_owner_ranks = np.asarray(
            record.get("cell_owner_ranks", np.empty((0,), dtype=np.int32))
        )
        vertex_owner_ranks = np.asarray(
            record.get("vertex_owner_ranks", np.empty((0,), dtype=np.int32))
        )
        source_point_ids = np.asarray(
            record.get("source_point_ids", np.empty((0,), dtype=np.int64))
        )
        source_owner_ranks = np.asarray(
            record.get("source_owner_ranks", np.empty((0,), dtype=np.int32))
        )
        source_owned = np.asarray(
            record.get("source_owned", np.empty((0,), dtype=np.bool_))
        )
        source_context_id = record.get("source_context_id")
        boundary_keys = {
            "part_authority",
            "accepted_composition",
            "owner_directory_witness",
        }
        has_boundary_context = any(key in record for key in boundary_keys)
        restored_compact_context = (
            record.get("_compact_restore_validation") is _COMPACT_OWNER_LOCAL_RESTORE
        )
        if has_boundary_context == restored_compact_context:
            raise TypeError(
                "Owner-local records require either publication context or collective restoration."
            )
        part_authority = record.get("part_authority")
        accepted_composition = record.get("accepted_composition")
        receipt = record.get("acceptance_receipt")
        owner_directory_witness = record.get("owner_directory_witness")
        if has_boundary_context:
            if not isinstance(part_authority, CompositionEntry) or not isinstance(
                accepted_composition, Composition
            ):
                raise TypeError(
                    "Owner-local publication requires its accepted part entry and Composition."
                )
            if (
                accepted_composition.entry(part_authority.entry_id).record_id
                != part_authority.record_id
                or part_authority.revision_id != identities["part_id"]
                or part_authority.structure_id != identities["source_topology_id"]
            ):
                raise ValueError(
                    "Accepted part authority does not bind this source revision."
                )
            if receipt is not None:
                if not isinstance(receipt, CompositionRebindReceipt):
                    raise TypeError(
                        "acceptance_receipt must be CompositionRebindReceipt or None."
                    )
                if (
                    not receipt.published
                    or not receipt.boundary_accepted
                    or receipt.composition.composition_id
                    != accepted_composition.composition_id
                ):
                    raise ValueError(
                        "Owner-local part authority is not at an accepted boundary."
                    )
            if not isinstance(owner_directory_witness, ShardedOwnerDirectoryWitness):
                raise TypeError(
                    "Owner-local publication requires ShardedOwnerDirectoryWitness."
                )
            directory_entries = tuple(
                entry
                for entry in accepted_composition.entries
                if entry.value is owner_directory_witness
            )
            if len(directory_entries) != 1:
                raise ValueError(
                    "Sharded owner directory is not an accepted CompositionEntry value."
                )
            directory_entry = directory_entries[0]
            source_entry_ids = {
                dependency.entry_id
                for dependency in part_authority.dependencies
                if dependency.entry_id != directory_entry.entry_id
            }
            if len(source_entry_ids) != 1:
                raise ValueError(
                    "Accepted part authority requires one scientific source entry."
                )
            source_entry = accepted_composition.entry(next(iter(source_entry_ids)))
            source_dependencies = tuple(
                dependency
                for dependency in part_authority.dependencies
                if dependency.entry_id == source_entry.entry_id
            )
            if (
                not isinstance(source_entry, CompositionEntry)
                or not source_dependencies
                or any(
                    source_entry.binding(dependency.facet).bound_id != dependency.bound_id
                    for dependency in source_dependencies
                )
            ):
                raise ValueError(
                    "Accepted source entry does not bind the local part authority."
                )
            if (
                owner_directory_witness.source_authority_id != part_authority.record_id
                or not any(
                    dependency.entry_id == directory_entry.entry_id
                    and dependency.facet == "structure"
                    and dependency.bound_id
                    == directory_entry.structure_id
                    == owner_directory_witness.directory_id
                    for dependency in part_authority.dependencies
                )
            ):
                raise ValueError(
                    "Accepted part authority does not bind its sharded owner directory."
                )
            if not serial_source_view:
                owner_directory_witness.require_local_sources(
                    part_authority.record_id,
                    owner_rank,
                    source_point_ids,
                    source_owner_ranks,
                    source_owned,
                )
            owner_directory_witness.require_local_view(
                part_authority,
                accepted_composition,
                receipt,
                identities["part_id"],
                identities["source_mesh_id"],
                identities["source_topology_id"],
                identities["source_geometry_id"],
                source_map_id,
                owner_rank,
                cell_ids,
                cell_owner_ranks,
                owned,
                vertex_ids,
                vertex_owner_ranks,
            )
            source_value = None if serial_source_view else source_entry.value
            if not serial_source_view and source_value is None:
                raise ValueError("Owner-local source entry lacks scientific source data.")
            source_authority = part_authority.value
        else:
            if any(
                key in record
                for key in (
                    "acceptance_receipt",
                    "part_authority",
                    "accepted_composition",
                )
            ):
                raise TypeError(
                    "Compact restoration cannot contain partial lifecycle context."
                )
            if not isinstance(
                record.get("owner_directory_data"), ShardedOwnerDirectoryData
            ):
                raise TypeError(
                    "Compact restoration requires data-only ShardedOwnerDirectoryData."
                )
            owner_directory_witness = None
            part_authority = None
            accepted_composition = None
            receipt = None
            source_value = record.get("source_value")
            source_authority = record.get("source_geometry_authority")
            if serial_source_view and source_value is not None:
                raise ValueError(
                    "Serial compact views cannot retain a whole source carrier."
                )

        if has_boundary_context:
            if not isinstance(owner_directory_witness, ShardedOwnerDirectoryWitness):
                raise TypeError(
                    "Owner-local publication requires ShardedOwnerDirectoryWitness."
                )
            owner_directory_data = ShardedOwnerDirectoryData.from_witness(
                owner_directory_witness
            )
        else:
            owner_directory_data = record["owner_directory_data"]
        owner_directory_data._validate_local_row_shapes(
            owner_rank,
            identities["distribution_id"],
            source_map_id,
            source_context_id,
            source_point_ids,
            source_owner_ranks,
            source_owned,
            cell_ids,
            cell_owner_ranks,
            owned,
            vertex_ids,
            vertex_owner_ranks,
        )
        geometry_authority = None
        if serial_source_view:
            if has_boundary_context:
                from ..meshing._assembly import MeshPart
                from ..meshing._result import CellMeshingResult

                if not isinstance(part_authority, CompositionEntry):
                    raise TypeError(
                        "Serial publication requires its accepted part entry."
                    )
                serial_part = part_authority.value
                if (
                    not isinstance(serial_part, MeshPart)
                    or not isinstance(serial_part.carrier, CellMeshingResult)
                    or serial_part.name != identities["part_name"]
                    or serial_part.part_id != identities["part_id"]
                ):
                    raise TypeError(
                        "A serial source view must bind its exact local MeshPart revision."
                    )
                serial_result = serial_part.carrier
                geometry_authority = serial_result.geometry.source_map
                if (
                    serial_result.mesh.mesh_id != identities["source_mesh_id"]
                    or serial_result.mesh.topology_id != identities["source_topology_id"]
                    or cell_geometry_id(serial_result.geometry)
                    != identities["source_geometry_id"]
                    or serial_result.geometry.source_map_id != source_map_id
                ):
                    raise ValueError(
                        "Serial source view changed its accepted mesh/map identity."
                    )
            else:
                geometry_authority = source_authority
                from ._exact_power_geometry import ExactPowerCellGeometrySource

                if geometry_authority is not None and not isinstance(
                    geometry_authority,
                    (CellGeometryCoordinateMap, ExactPowerCellGeometrySource),
                ):
                    raise TypeError("Serial compact source maps must be exact map data.")
        else:
            from ..meshing._assembly import MeshPart
            from ..meshing._result import CellMeshingResult
            from ..meshing.providers._native_periodic import NativePeriodicSource
            from ._exact_power_geometry import ExactPowerCellGeometrySource

            if isinstance(
                source_value,
                (
                    OwnerLocalCellGeometry,
                    MeshPart,
                    CellMeshingResult,
                    Composition,
                    CompositionEntry,
                    CompositionRebindReceipt,
                    ShardedOwnerDirectoryData,
                    ShardedOwnerDirectoryWitness,
                ),
            ):
                raise TypeError(
                    "Owner-local source values cannot retain carriers or issuer objects."
                )
            if source_value is not None and not isinstance(
                source_value,
                (
                    NativePeriodicSource,
                    CellGeometryCoordinateMap,
                    ExactPowerCellGeometrySource,
                ),
            ):
                raise TypeError(
                    "Owner-local source values must be compact source/map data."
                )
            if record.get("periodic_topology_id") is not None:
                from ..meshing._periodic import PeriodicPointOrbits

                if not isinstance(source_value, NativePeriodicSource) or not isinstance(
                    source_value.domain, PeriodicPointOrbits
                ):
                    raise TypeError(
                        "Periodic owner-local views require authored point-orbit data."
                    )

            if isinstance(
                source_authority,
                (
                    OwnerLocalCellGeometry,
                    Composition,
                    CompositionEntry,
                    CompositionRebindReceipt,
                    CellMeshingResult,
                    ShardedOwnerDirectoryData,
                    ShardedOwnerDirectoryWitness,
                ),
            ):
                raise TypeError(
                    "Owner-local geometry authority cannot retain carriers or issuer objects."
                )
            if isinstance(source_authority, MeshPart):
                if not isinstance(source_authority.carrier, CellMeshingResult):
                    raise TypeError("Source MeshPart lacks a cell-meshing result.")
                source_authority = source_authority.carrier.geometry.source_map
                if isinstance(
                    source_authority,
                    (
                        OwnerLocalCellGeometry,
                        MeshPart,
                        CellMeshingResult,
                        Composition,
                        CompositionEntry,
                        CompositionRebindReceipt,
                        ShardedOwnerDirectoryData,
                        ShardedOwnerDirectoryWitness,
                    ),
                ):
                    raise TypeError(
                        "MeshPart source maps cannot retain carriers or issuer objects."
                    )
            if isinstance(source_authority, NativePeriodicSource):
                source_authority.validate_restored()
                if (
                    source_authority.source_id != identities["source_id"]
                    or source_authority.source_revision != identities["source_revision"]
                    or source_map_id is not None
                ):
                    raise ValueError(
                        "Native periodic source identity differs from this local view."
                    )
                if source_context_id != source_authority.binding_id:
                    raise ValueError(
                        "Native periodic source context does not match its local quotient."
                    )
            elif type(source_authority) is ExactPowerCellGeometrySource:
                if (
                    source_authority.source_id != identities["source_id"]
                    or source_authority.source_revision != identities["source_revision"]
                    or source_map_id != source_authority.source_map_id
                ):
                    raise ValueError(
                        "Exact power source map identity differs from this local view."
                    )
                context = source_authority.periodic_source_context
                expected_context_id = None if context is None else context.context_id
                if source_context_id != expected_context_id:
                    raise ValueError(
                        "Exact power source context differs from this local source view."
                    )
                if record.get("periodic_topology_id") is not None and context is None:
                    raise ValueError(
                        "Periodic power geometry lacks its exact source image context."
                    )
                geometry_authority = source_authority
            elif isinstance(source_authority, CellGeometryCoordinateMap):
                if (
                    source_authority.source_id != identities["source_id"]
                    or source_authority.source_revision != identities["source_revision"]
                    or source_map_id != source_authority.source_map_id
                ):
                    raise ValueError(
                        "Coordinate map identity differs from this local source view."
                    )
                if source_context_id is not None:
                    raise ValueError(
                        "Non-periodic coordinate maps cannot bind periodic source contexts."
                    )
                geometry_authority = source_authority
            else:
                raise TypeError(
                    "Owner-local source data has an unsupported concrete type."
                )
            if source_value is not None and (
                getattr(source_value, "source_id", identities["source_id"])
                != identities["source_id"]
                or getattr(source_value, "source_revision", identities["source_revision"])
                != identities["source_revision"]
            ):
                raise ValueError(
                    "Stored source data differs from its scientific identity."
                )
        if has_boundary_context:
            if not isinstance(part_authority, CompositionEntry) or not isinstance(
                accepted_composition, Composition
            ):
                raise TypeError("Owner-local publication requires its accepted context.")
            if part_authority.semantics_id != identities["source_geometry_id"]:
                raise ValueError(
                    "Accepted part authority does not bind the exact source geometry."
                )
            bound_entries = tuple(
                (dependency, accepted_composition.entry(dependency.entry_id))
                for dependency in part_authority.dependencies
            )
            source_mesh_bound = serial_source_view or any(
                dependency.facet == "structure"
                and entry.structure_id == identities["source_mesh_id"]
                for dependency, entry in bound_entries
            )
            source_revision_bound = serial_source_view or any(
                dependency.facet == "revision"
                and entry.revision_id == identities["source_revision"]
                for dependency, entry in bound_entries
            )
            if not source_mesh_bound or not source_revision_bound:
                raise ValueError(
                    "Accepted part authority lacks source mesh/revision dependencies."
                )
        if serial_source_view:
            source_geometry_authority = geometry_authority
        else:
            source_geometry_authority = source_authority
        cell_rows = np.asarray(record["cell_rows"], dtype=np.int64)
        measures = np.asarray(record["cell_measures"], dtype=np.float64)
        vertex_rows = np.asarray(record["vertex_rows"], dtype=np.int64)
        vertex_coordinates = np.asarray(record["vertex_coordinates"], dtype=np.float64)
        coordinate_rows = np.asarray(record["coordinate_rows"], dtype=np.int64)
        coordinates = np.asarray(record["coordinates"], dtype=np.float64)
        raw_exact_coordinates = record["exact_coordinates"]
        exact_coordinates = tuple(
            tuple(value for value in row) for row in raw_exact_coordinates
        )
        exact_float = (
            np.empty(coordinates.shape, dtype=np.float64)
            if not exact_coordinates and coordinates.shape[0] == 0
            else np.asarray(exact_coordinates, dtype=np.float64)
        )
        if (
            cell_rows.ndim != 1
            or cell_ids.shape != cell_rows.shape
            or owned.shape != cell_rows.shape
            or measures.shape != cell_rows.shape
            or np.unique(cell_ids).size != cell_ids.size
            or np.unique(cell_rows).size != cell_rows.size
            or np.any(cell_rows < 0)
            or np.any(~np.isfinite(measures) | (measures <= 0.0))
            or vertex_rows.ndim != 1
            or vertex_ids.shape != vertex_rows.shape
            or np.unique(vertex_ids).size != vertex_ids.size
            or vertex_coordinates.ndim != 2
            or vertex_coordinates.shape[0] != vertex_rows.size
            or coordinates.ndim != 2
            or coordinate_rows.shape != (coordinates.shape[0],)
            or exact_float.shape != coordinates.shape
            or not np.array_equal(exact_float, coordinates)
            or not np.all(np.isfinite(vertex_coordinates))
            or not np.all(np.isfinite(coordinates))
            or np.any(vertex_rows < 0)
            or np.any(coordinate_rows < 0)
        ):
            raise ValueError("Owner-local source geometry arrays are malformed.")
        block_records = tuple(
            {**value, "local_vertex_count": vertex_rows.size}
            for value in record["blocks"]
        )
        blocks = tuple(OwnerLocalCellBlock(value) for value in block_records)
        block_ids = (
            np.concatenate(
                tuple(np.asarray(block.cell_ids, dtype=np.int64) for block in blocks)
            )
            if blocks
            else np.empty((0,), dtype=np.int64)
        )
        if not np.array_equal(np.sort(block_ids), np.sort(cell_ids)):
            raise ValueError(
                "Owner-local blocks must cover every local cell ID exactly once."
            )
        for block in blocks:
            routes = np.asarray(block.coordinate_routes, dtype=np.int64)
            vertices = np.asarray(block.vertices, dtype=np.int64)
            valid_vertices = np.asarray(block.vertex_valid, dtype=np.bool_)
            if np.any(routes >= coordinates.shape[0]) or np.any(
                vertices[valid_vertices] >= vertex_rows.size
            ):
                raise ValueError(
                    "Owner-local cell routes exceed compact coordinate banks."
                )
        entity_set_ids = tuple(
            sorted(
                (int(dimension_), str(identifier))
                for dimension_, identifier in dict(record["entity_set_ids"]).items()
            )
        )
        raw_entity_ids = dict(record["local_entity_ids"])
        if set(raw_entity_ids) != {dimension_ for dimension_, _ in entity_set_ids}:
            raise ValueError(
                "Owner-local entity IDs must match the declared entity sets."
            )
        local_entity_ids = tuple(
            (
                dimension_,
                jnp.asarray(np.asarray(raw_entity_ids[dimension_], dtype=np.int64)),
            )
            for dimension_, _ in entity_set_ids
        )
        for dimension_, entity_ids in local_entity_ids:
            values = np.asarray(entity_ids)
            if (
                dimension_ < 0
                or dimension_ > int(dimension)
                or values.ndim != 1
                or np.unique(values).size != values.size
            ):
                raise ValueError("Owner-local entity identity rows are malformed.")
        if not np.array_equal(
            np.asarray(dict(local_entity_ids)[int(dimension)]), cell_ids
        ) or not np.array_equal(np.asarray(dict(local_entity_ids)[0]), vertex_ids):
            raise ValueError(
                "Owner-local vertex/cell IDs do not match their entity sets."
            )
        raw_entity_vertex_offsets = dict(record["entity_vertex_offsets"])
        raw_entity_vertex_values = dict(record["entity_vertex_values"])
        if set(raw_entity_vertex_offsets) != set(raw_entity_vertex_values) or not set(
            raw_entity_vertex_offsets
        ).issubset(dict(local_entity_ids)):
            raise ValueError("Owner-local entity vertex routes do not match entity sets.")
        entity_vertex_offsets = []
        entity_vertex_values = []
        for dimension_, ids in entity_set_ids:
            entity_count = np.asarray(dict(local_entity_ids)[dimension_]).size
            if dimension_ == 0 and dimension_ not in raw_entity_vertex_offsets:
                offsets = np.arange(entity_count + 1, dtype=np.int64)
                values = np.arange(entity_count, dtype=np.int32)
            else:
                if dimension_ not in raw_entity_vertex_offsets:
                    raise ValueError("Owner-local entity incidence is incomplete.")
                offsets = np.asarray(raw_entity_vertex_offsets[dimension_])
                values = np.asarray(raw_entity_vertex_values[dimension_])
                if offsets.dtype.kind not in "iu" or values.dtype.kind not in "iu":
                    raise TypeError(
                        "Owner-local entity incidence requires integer routes."
                    )
                offsets = offsets.astype(np.int64, copy=False)
                values = values.astype(np.int32, copy=False)
            if (
                offsets.shape != (entity_count + 1,)
                or offsets[0] != 0
                or offsets[-1] != values.size
                or np.any(np.diff(offsets) < 1)
                or np.any(values < 0)
                or np.any(values >= vertex_rows.size)
            ):
                raise ValueError("Owner-local entity vertex incidence is malformed.")
            entity_vertex_offsets.append((dimension_, jnp.asarray(offsets)))
            entity_vertex_values.append((dimension_, jnp.asarray(values)))
        periodic_topology_id = record.get("periodic_topology_id")
        if periodic_topology_id is not None and (
            not isinstance(periodic_topology_id, str) or not periodic_topology_id
        ):
            raise ValueError("Periodic owner-local topology identity must be non-empty.")
        periodic_cell_ids = record.get("periodic_cell_ids")
        representatives = record.get("vertex_representative_ids")
        vertex_windings = record.get("vertex_windings")
        cell_windings = record.get("cell_windings")
        periodic_arrays = tuple(
            None if value is None else np.asarray(value, dtype=np.int64)
            for value in (
                periodic_cell_ids,
                representatives,
                vertex_windings,
                cell_windings,
            )
        )
        if periodic_topology_id is None and any(
            value is not None for value in periodic_arrays
        ):
            raise ValueError("Periodic local arrays require their topology identity.")
        if periodic_topology_id is not None and (
            periodic_arrays[0] is None
            or periodic_arrays[0].shape != cell_ids.shape
            or periodic_arrays[1] is None
            or periodic_arrays[1].shape != vertex_ids.shape
            or periodic_arrays[2] is None
            or periodic_arrays[2].ndim != 2
            or periodic_arrays[2].shape[0] != vertex_ids.size
            or periodic_arrays[3] is None
            or periodic_arrays[3].ndim != 2
            or periodic_arrays[3].shape[0] != cell_ids.size
        ):
            raise ValueError(
                "Periodic owner-local identities do not align with cells and vertices."
            )
        if (
            periodic_topology_id is not None
            and not serial_source_view
            and (specification_id is None)
        ):
            raise ValueError(
                "Owner-local periodic views require their exact specification identity."
            )
        periodic_rank = 0 if periodic_arrays[2] is None else periodic_arrays[2].shape[1]
        source_context_id = record.get("source_context_id")
        if source_context_id is not None and (
            not isinstance(source_context_id, str) or not source_context_id
        ):
            raise ValueError("Owner-local source context identity must be non-empty.")
        periodic_lift_id = record.get("periodic_lift_id")
        raw_lift_matrix = record.get("periodic_lift_matrix")
        periodic_lift_matrix = (
            None
            if raw_lift_matrix is None
            else np.asarray(raw_lift_matrix, dtype=np.float64)
        )
        coordinates_are_lifted = record.get("coordinates_are_lifted")
        if coordinates_are_lifted is None and periodic_topology_id is None:
            coordinates_are_lifted = True
        if (
            not isinstance(coordinates_are_lifted, (bool, np.bool_))
            or periodic_topology_id is not None
            and (
                not isinstance(periodic_lift_id, str)
                or not periodic_lift_id
                or periodic_lift_matrix is None
                or periodic_lift_matrix.shape != (periodic_rank, coordinates.shape[1])
                or not np.all(np.isfinite(periodic_lift_matrix))
            )
            or periodic_topology_id is None
            and (periodic_lift_id is not None or periodic_lift_matrix is not None)
        ):
            raise ValueError("Owner-local periodic lift binding is malformed.")
        fractional_lower = record.get("support_fractional_lower")
        fractional_upper = record.get("support_fractional_upper")
        support_fractional_lower = (
            None
            if fractional_lower is None
            else np.asarray(fractional_lower, dtype=np.float64)
        )
        support_fractional_upper = (
            None
            if fractional_upper is None
            else np.asarray(fractional_upper, dtype=np.float64)
        )
        if (support_fractional_lower is None) != (support_fractional_upper is None):
            raise ValueError(
                "Owner-local support AABB endpoints must be supplied together."
            )
        if (
            support_fractional_lower is not None
            and support_fractional_upper is not None
            and (
                support_fractional_lower.shape != (cell_ids.size, periodic_rank)
                or support_fractional_upper.shape != support_fractional_lower.shape
                or np.any(~np.isfinite(support_fractional_lower))
                or np.any(~np.isfinite(support_fractional_upper))
                or np.any(support_fractional_lower > support_fractional_upper)
            )
        ):
            raise ValueError("Owner-local fractional support AABBs are malformed.")
        raw_key_offsets = record.get("cell_key_offsets")
        raw_key_sources = record.get("cell_key_source_ids")
        raw_key_windings = record.get("cell_key_relative_windings")
        if raw_key_offsets is None and periodic_topology_id is None:
            cell_key_offsets = np.arange(cell_ids.size + 1, dtype=np.int64)
            cell_key_source_ids = cell_ids.copy()
            cell_key_relative_windings = np.empty((cell_ids.size, 0), dtype=np.int64)
        else:
            cell_key_offsets = np.asarray(raw_key_offsets)
            cell_key_source_ids = np.asarray(raw_key_sources)
            cell_key_relative_windings = np.asarray(raw_key_windings)
            if (
                cell_key_offsets.dtype.kind not in "iu"
                or cell_key_source_ids.dtype.kind not in "iu"
                or cell_key_relative_windings.dtype.kind not in "iu"
            ):
                raise TypeError(
                    "Periodic cell keys must use integer source IDs and windings."
                )
            cell_key_offsets = cell_key_offsets.astype(np.int64, copy=False)
            cell_key_source_ids = cell_key_source_ids.astype(np.int64, copy=False)
            cell_key_relative_windings = cell_key_relative_windings.astype(
                np.int64, copy=False
            )
        if (
            cell_key_offsets.shape != (cell_ids.size + 1,)
            or cell_key_offsets[0] != 0
            or cell_key_offsets[-1] != cell_key_source_ids.size
            or np.any(np.diff(cell_key_offsets) < 1)
            or cell_key_relative_windings.shape
            != (cell_key_source_ids.size, periodic_rank)
        ):
            raise ValueError("Owner-local semantic cell keys are malformed.")
        for cell in range(cell_ids.size):
            start, stop = cell_key_offsets[cell : cell + 2]
            keys = tuple(
                (
                    int(source_id),
                    tuple(
                        int(value)
                        for value in np.asarray(winding, dtype=np.int64)
                        .reshape((-1,))
                        .tolist()
                    ),
                )
                for source_id, winding in zip(
                    cell_key_source_ids[start:stop],
                    cell_key_relative_windings[start:stop],
                    strict=True,
                )
            )
            if keys != tuple(sorted(set(keys))):
                raise ValueError(
                    "Owner-local cell keys must be unique and lexicographically ordered."
                )
        if serial_source_view:
            support_cell_mask = np.asarray(
                record.get("support_cell_mask", np.ones(cell_ids.shape, dtype=np.bool_))
            )
        else:
            support_cell_mask = np.asarray(record["support_cell_mask"])
        vertex_source_ids = np.asarray(
            record.get("vertex_source_ids", vertex_ids), dtype=np.int64
        )
        if not serial_source_view:
            if isinstance(source_authority, NativePeriodicSource):
                from ..meshing._periodic import PeriodicPointOrbits

                if isinstance(source_authority.domain, PeriodicPointOrbits):
                    source_ids = np.sort(
                        np.asarray(source_authority.domain.point_ids, dtype=np.int64)
                    )
                    if not np.array_equal(
                        np.sort(np.asarray(source_point_ids, dtype=np.int64)),
                        source_ids,
                    ):
                        raise ValueError(
                            "Local periodic source IDs differ from their typed point bank."
                        )
            elif isinstance(source_authority, ExactPowerCellGeometrySource):
                context = source_authority.periodic_source_context
                if context is not None:
                    source_ids = np.sort(
                        np.asarray(context.periodic_source.point_ids, dtype=np.int64)
                    )
                    if not np.array_equal(
                        np.sort(np.asarray(source_point_ids, dtype=np.int64)),
                        source_ids,
                    ):
                        raise ValueError(
                            "Exact power source IDs differ from their local periodic context."
                        )
        vertex_absolute_windings = np.asarray(
            record.get(
                "vertex_absolute_windings",
                np.zeros((vertex_ids.size, periodic_rank), dtype=np.int64),
            )
        )
        raw_candidate_revision_id = record.get("candidate_revision_id")
        candidate_revision_id = (
            None if raw_candidate_revision_id is None else str(raw_candidate_revision_id)
        )
        certified_periodic_margin = record.get("certified_periodic_margin")
        periodic_shell_certificate = record.get("periodic_shell_certificate")
        adaptive_result = record.get("adaptive_result")
        graph_migration_evidence = record.get("graph_migration_evidence")
        if adaptive_result is not None:
            from ..meshing._device_adaptation import OwnerLocalPeriodicAdaptiveResult
            from ..meshing.providers._native_periodic import NativePeriodicSource

            if not isinstance(adaptive_result, OwnerLocalPeriodicAdaptiveResult):
                raise TypeError(
                    "adaptive_result must be OwnerLocalPeriodicAdaptiveResult or None."
                )
            if not isinstance(source_authority, NativePeriodicSource):
                raise ValueError(
                    "Only exact periodic Delaunay views may bind an adaptive result."
                )
        from ..graph._partition import GraphPartitionPlan
        from ..meshing._distribution import OwnerLocalGraphMigrationEvidence

        if graph_migration_evidence is None:
            if (
                adaptive_result is not None
                and adaptive_result.graph_partition_plan is not None
            ):
                raise ValueError(
                    "An explicit graph plan requires applied owner-local migration evidence."
                )
        elif (
            not isinstance(graph_migration_evidence, OwnerLocalGraphMigrationEvidence)
            or graph_migration_evidence.local_rank != owner_rank
            or graph_migration_evidence.part_count != owner_directory_data.part_count
            or graph_migration_evidence.group_id != owner_directory_data.group_id
        ):
            raise TypeError("graph_migration_evidence has the wrong local owner binding.")
        elif adaptive_result is None:
            if graph_migration_evidence.target_part_id != identities["part_id"]:
                raise ValueError("Graph migration evidence targets a different part.")
            if (
                graph_migration_evidence.target_directory_id
                != identities["distribution_id"]
            ):
                raise ValueError(
                    "Graph migration evidence targets a different owner directory."
                )
            if (
                cell_ids.ndim != 1
                or owned.shape != cell_ids.shape
                or owned.dtype != np.dtype(np.bool_)
            ):
                raise ValueError(
                    "Owner-local cell ownership rows are malformed for graph evidence."
                )
            if not np.array_equal(
                np.asarray(graph_migration_evidence.target_cell_ids, dtype=np.int64),
                np.sort(cell_ids[owned], kind="stable"),
            ):
                raise ValueError(
                    "Graph migration evidence owned cell IDs differ from this local view."
                )
        else:
            plan = adaptive_result.graph_partition_plan
            if (
                not isinstance(plan, GraphPartitionPlan)
                or graph_migration_evidence.result.plan_id != plan.plan_id
                or graph_migration_evidence.target_part_id
                != adaptive_result.source_part_id
                or graph_migration_evidence.target_directory_id
                != adaptive_result.source_distribution_id
                or not np.array_equal(
                    np.asarray(graph_migration_evidence.target_cell_ids, dtype=np.int64),
                    np.asarray(adaptive_result.source_cell_ids, dtype=np.int64),
                )
            ):
                raise ValueError(
                    "Graph migration evidence does not bind the adaptive source shard."
                )
        if (
            not isinstance(serial_source_view, bool)
            or owner_rank < 0
            or cell_owner_ranks.shape != cell_ids.shape
            or cell_owner_ranks.dtype.kind not in "iu"
            or not np.array_equal(
                owned, cell_owner_ranks.astype(np.int64) == int(owner_rank)
            )
            or support_cell_mask.shape != cell_ids.shape
            or support_cell_mask.dtype != np.dtype(np.bool_)
            or vertex_source_ids.shape != vertex_ids.shape
            or source_point_ids.ndim != 1
            or source_point_ids.dtype.kind not in "iu"
            or np.any(source_point_ids < 0)
            or np.unique(source_point_ids).size != source_point_ids.size
            or source_owner_ranks.shape != source_point_ids.shape
            or source_owner_ranks.dtype.kind not in "iu"
            or np.any(source_owner_ranks < 0)
            or source_owned.shape != source_point_ids.shape
            or source_owned.dtype != np.dtype(np.bool_)
            or not np.array_equal(
                source_owned,
                source_owner_ranks.astype(np.int64, copy=False) == int(owner_rank),
            )
            or vertex_owner_ranks.shape != vertex_ids.shape
            or vertex_owner_ranks.dtype.kind not in "iu"
            or np.any(vertex_owner_ranks < 0)
            or vertex_absolute_windings.dtype.kind not in "iu"
            or vertex_absolute_windings.shape != (vertex_ids.size, periodic_rank)
            or (candidate_revision_id is None and not serial_source_view)
            or (candidate_revision_id is not None and not candidate_revision_id)
        ):
            raise ValueError(
                "Owner-local candidate/support ownership metadata is malformed."
            )
        raw_coordinate_stencil = record.get("source_coordinate_stencil")
        raw_coordinate_offsets = record.get("exact_vertex_coordinate_offsets")
        if (raw_coordinate_stencil is None) != (raw_coordinate_offsets is None):
            raise ValueError(
                "Periodic source-coordinate stencil and exact offsets must be supplied together."
            )
        source_coordinate_stencil = raw_coordinate_stencil
        exact_vertex_coordinate_offsets = raw_coordinate_offsets
        if source_coordinate_stencil is not None:
            from ..meshing._lineage import VertexInterpolationStencil

            if not isinstance(source_coordinate_stencil, VertexInterpolationStencil):
                raise TypeError(
                    "source_coordinate_stencil must be VertexInterpolationStencil or None."
                )
            if not isinstance(exact_vertex_coordinate_offsets, tuple):
                raise TypeError(
                    "Exact vertex coordinate offsets must be an immutable tuple."
                )
            exact_vertex_coordinate_offsets = tuple(
                tuple(point) for point in exact_vertex_coordinate_offsets
            )
            stencil_source_ids = np.asarray(
                source_coordinate_stencil.source_global_ids, dtype=np.int64
            )
            stencil_valid = np.asarray(source_coordinate_stencil.valid, dtype=np.bool_)
            if (
                not source_coordinate_stencil.preserves_constants
                or not np.array_equal(
                    np.asarray(
                        source_coordinate_stencil.target_global_ids, dtype=np.int64
                    ),
                    vertex_ids,
                )
                or stencil_source_ids.shape != stencil_valid.shape
                or len(exact_vertex_coordinate_offsets) != vertex_ids.size
                or any(
                    len(point) != coordinates.shape[1]
                    or any(not isinstance(value, Fraction) for value in point)
                    for point in exact_vertex_coordinate_offsets
                )
            ):
                raise ValueError(
                    "Cumulative periodic coordinate ancestry differs from its exact source bank."
                )
            if np.any(~np.isin(stencil_source_ids[stencil_valid], source_point_ids)):
                raise ValueError(
                    "Cumulative periodic coordinate ancestry references roots outside "
                    "its exact source bank."
                )
        periodic_owner_route = periodic_topology_id is not None and not serial_source_view
        periodic_delaunay_route = periodic_owner_route and isinstance(
            source_authority, NativePeriodicSource
        )
        periodic_power_route = periodic_owner_route and isinstance(
            source_authority, ExactPowerCellGeometrySource
        )
        if periodic_owner_route and not (periodic_delaunay_route or periodic_power_route):
            raise TypeError(
                "Periodic owner-local geometry requires a typed source authority."
            )
        if (
            periodic_delaunay_route
            and vertex_ids.size
            and source_coordinate_stencil is None
        ):
            raise ValueError(
                "Nonempty periodic Delaunay views require exact source-coordinate ancestry."
            )
        if source_coordinate_stencil is not None and not periodic_delaunay_route:
            raise ValueError(
                "Cumulative source-coordinate ancestry only applies to periodic Delaunay views."
            )
        if (certified_periodic_margin is None) != (periodic_shell_certificate is None):
            raise ValueError(
                "Periodic shell margin and certificate must be supplied together."
            )
        if (
            periodic_topology_id is not None
            and not serial_source_view
            and (
                candidate_revision_id is None
                or certified_periodic_margin is None
                or periodic_shell_certificate is None
            )
        ):
            raise ValueError(
                "Native periodic owner-local views require the shell certificate."
            )
        periodic_face_coverage_id = None
        graph_owned_ids = np.zeros((0,), dtype=np.int64)
        graph_owner_ranks = np.zeros((0,), dtype=np.int32)
        graph_weights = np.zeros((0,), dtype=np.int64)
        graph_offsets = np.zeros((1,), dtype=np.int64)
        graph_neighbor_ids = np.zeros((0,), dtype=np.int64)
        graph_neighbor_owner_ranks = np.zeros((0,), dtype=np.int32)
        graph_neighbor_weights = np.zeros((0,), dtype=np.int64)
        if periodic_topology_id is not None and not serial_source_view:
            if not isinstance(periodic_shell_certificate, Mapping):
                raise ValueError(
                    "Native periodic graph rows require the shell proof record."
                )
            periodic_face_coverage_id = periodic_shell_certificate.get(
                "quotient_face_coverage_id"
            )
            if (
                not isinstance(periodic_face_coverage_id, str)
                or not periodic_face_coverage_id
            ):
                raise ValueError(
                    "Native periodic graph rows require exact face coverage."
                )
            graph_owned_ids = np.asarray(record.get("owned_cell_ids"))
            raw_graph_owner_ranks = np.asarray(record.get("owned_cell_owner_ranks"))
            raw_graph_weights = np.asarray(record.get("owned_cell_weights"))
            graph_offsets = np.asarray(record.get("cell_neighbor_offsets"))
            graph_neighbor_ids = np.asarray(record.get("cell_neighbor_ids"))
            raw_neighbor_owner_ranks = np.asarray(record.get("cell_neighbor_owner_ranks"))
            raw_neighbor_weights = np.asarray(record.get("cell_neighbor_weights"))
            if (
                graph_owned_ids.dtype.kind not in "iu"
                or raw_graph_owner_ranks.dtype.kind not in "iu"
                or raw_graph_weights.dtype.kind not in "iu"
                or graph_offsets.dtype.kind not in "iu"
                or graph_neighbor_ids.dtype.kind not in "iu"
                or raw_neighbor_owner_ranks.dtype.kind not in "iu"
                or raw_neighbor_weights.dtype.kind not in "iu"
            ):
                raise TypeError(
                    "Owner-local periodic graph identities, offsets, and weights must be integer arrays."
                )
            graph_integer_banks = (
                graph_owned_ids,
                raw_graph_owner_ranks,
                raw_graph_weights,
                graph_offsets,
                graph_neighbor_ids,
                raw_neighbor_owner_ranks,
                raw_neighbor_weights,
            )
            if any(
                value.dtype.kind == "u"
                and value.size
                and int(np.max(value)) > np.iinfo(np.int64).max
                for value in graph_integer_banks
            ):
                raise ValueError(
                    "Owner-local periodic graph IDs and work weights must fit int64."
                )
            if any(
                value.size
                and (np.any(value < 0) or int(np.max(value)) > np.iinfo(np.int32).max)
                for value in (raw_graph_owner_ranks, raw_neighbor_owner_ranks)
            ):
                raise ValueError("Owner-local periodic graph owner slots must fit int32.")
            graph_weights = raw_graph_weights.astype(np.int64, copy=False)
            graph_owned_ids = graph_owned_ids.astype(np.int64, copy=False)
            graph_owner_ranks = raw_graph_owner_ranks.astype(np.int32, copy=False)
            graph_offsets = graph_offsets.astype(np.int64, copy=False)
            graph_neighbor_ids = graph_neighbor_ids.astype(np.int64, copy=False)
            graph_neighbor_owner_ranks = raw_neighbor_owner_ranks.astype(
                np.int32, copy=False
            )
            graph_neighbor_weights = raw_neighbor_weights.astype(np.int64, copy=False)
            expected_graph_ids = np.sort(cell_ids[owned], kind="stable")
            part_count = int(owner_directory_data.part_count)
            cell_row_by_id = {
                int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
            }
            graph_rows = np.asarray(
                [
                    cell_row_by_id.get(int(identifier), -1)
                    for identifier in graph_owned_ids
                ],
                dtype=np.int64,
            )
            if (
                not np.array_equal(graph_owned_ids, expected_graph_ids)
                or np.any(graph_rows < 0)
                or not np.array_equal(
                    graph_owner_ranks,
                    np.full(graph_owned_ids.shape, int(owner_rank), dtype=np.int32),
                )
                or graph_weights.shape != graph_owned_ids.shape
                or np.any(graph_weights <= 0)
                or graph_offsets.shape != (graph_owned_ids.size + 1,)
                or graph_offsets[0] != 0
                or np.any(np.diff(graph_offsets) < 0)
                or graph_offsets[-1] != graph_neighbor_ids.size
                or graph_neighbor_owner_ranks.shape != graph_neighbor_ids.shape
                or graph_neighbor_weights.shape != graph_neighbor_ids.shape
                or np.any(graph_neighbor_weights <= 0)
                or np.any(
                    (graph_neighbor_owner_ranks < 0)
                    | (graph_neighbor_owner_ranks >= part_count)
                )
                or np.any(graph_rows >= cell_ids.size)
                or not np.array_equal(cell_ids[graph_rows], graph_owned_ids)
                or any(
                    np.unique(graph_neighbor_ids[left:right]).size != right - left
                    for left, right in zip(
                        graph_offsets[:-1], graph_offsets[1:], strict=True
                    )
                )
            ):
                raise ValueError(
                    "Owner-local periodic graph rows differ from accepted cells."
                )
        for block in blocks:
            shifts = block.cell_vertex_shifts
            if shifts is not None and shifts.shape != (
                block.cell_ids.size,
                block.vertices.shape[1],
                periodic_rank,
            ):
                raise ValueError(
                    "Periodic cell-vertex shifts do not match the owner-local block."
                )
        exact_vertex_coordinates = record.get("exact_vertex_coordinates")
        exact_vertex = (
            None
            if exact_vertex_coordinates is None
            else tuple(tuple(value for value in row) for row in exact_vertex_coordinates)
        )
        if exact_vertex is not None and (
            np.asarray(exact_vertex, dtype=np.float64).shape != vertex_coordinates.shape
            or not np.array_equal(
                np.asarray(exact_vertex, dtype=np.float64), vertex_coordinates
            )
        ):
            raise ValueError(
                "Owner-local exact vertex coordinates differ from their execution shadows."
            )
        local_mesh, local_geometry = _reconstruct_owner_local_geometry(
            vertex_coordinates,
            vertex_ids,
            coordinates,
            blocks,
            geometry_authority,
            None if periodic_arrays[1] is None else periodic_arrays[1],
            None if periodic_arrays[2] is None else periodic_arrays[2],
        )
        if source_coordinate_stencil is not None and (
            local_mesh is None
            or source_coordinate_stencil.target_entity_set_id
            != local_mesh.entity_set(0).entity_set_id
        ):
            raise ValueError(
                "Cumulative periodic coordinate stencil targets a different local vertex set."
            )
        local_mesh_id = None if local_mesh is None else local_mesh.mesh_id
        local_topology_id = None if local_mesh is None else local_mesh.topology_id
        local_geometry_id = (
            None if local_geometry is None else cell_geometry_id(local_geometry)
        )
        for field, expected in (
            ("local_mesh_id", local_mesh_id),
            ("local_topology_id", local_topology_id),
            ("local_geometry_id", local_geometry_id),
        ):
            if field in record and record[field] != expected:
                raise ValueError(
                    f"Stored {field} differs from reconstructed local geometry."
                )
        directory_id = identities["distribution_id"]
        from ._exact_algebraic_geometry import coefficient_payload

        exact_vertex_payload = (
            None
            if exact_vertex is None
            else tuple(
                tuple(coefficient_payload(value) for value in row) for row in exact_vertex
            )
        )
        exact_coordinate_payload = tuple(
            tuple(coefficient_payload(value) for value in row)
            for row in exact_coordinates
        )
        exact_offset_payload = (
            None
            if exact_vertex_coordinate_offsets is None
            else tuple(
                tuple(coefficient_payload(value) for value in point)
                for point in exact_vertex_coordinate_offsets
            )
        )
        from ._nested_reference_types import reference_chart_arguments_identity

        local_view_id = canonical_fingerprint(
            {
                "kind": "owner-local-cell-geometry-view",
                "owner_rank": int(owner_rank),
                "serial_source_view": serial_source_view,
                "global": (
                    identities["part_id"],
                    identities["source_id"],
                    identities["source_revision"],
                    specification_id,
                    identities["source_mesh_id"],
                    identities["source_topology_id"],
                    identities["source_geometry_id"],
                    source_map_id,
                    identities["geometry_layout_id"],
                ),
                "distribution": identities["distribution_id"],
                "directory_source_manifest": owner_directory_data.source_manifest_id,
                "local_geometry": (
                    local_mesh_id,
                    local_topology_id,
                    local_geometry_id,
                ),
                "owner_directory": directory_id,
                "source_shard": (
                    source_point_ids,
                    source_owner_ranks,
                    source_owned,
                ),
                "cells": (cell_rows, cell_ids, owned, measures),
                "vertices": (
                    vertex_rows,
                    vertex_ids,
                    vertex_coordinates,
                    exact_vertex_payload,
                ),
                "coordinates": (
                    coordinate_rows,
                    coordinates,
                    exact_coordinate_payload,
                ),
                "coordinate_ancestry": (
                    None
                    if source_coordinate_stencil is None
                    else (
                        source_coordinate_stencil.stencil_id,
                        exact_offset_payload,
                    )
                ),
                "entities": (entity_set_ids, local_entity_ids),
                "periodic": (periodic_topology_id, periodic_arrays),
                "source_context": source_context_id,
                "periodic_lift": (
                    periodic_lift_id,
                    periodic_lift_matrix,
                    bool(coordinates_are_lifted),
                ),
                "support_fractional_aabb": (
                    support_fractional_lower,
                    support_fractional_upper,
                ),
                "entity_incidence": (entity_vertex_offsets, entity_vertex_values),
                "cell_keys": (
                    cell_key_offsets,
                    cell_key_source_ids,
                    cell_key_relative_windings,
                    cell_owner_ranks,
                    support_cell_mask,
                ),
                "vertex_sources": (vertex_source_ids, vertex_absolute_windings),
                "source_points": (
                    source_point_ids,
                    source_owner_ranks,
                    source_owned,
                ),
                "vertex_owners": vertex_owner_ranks,
                "candidate_revision": candidate_revision_id,
                "periodic_shell": (
                    certified_periodic_margin,
                    periodic_shell_certificate,
                ),
                "periodic_face_coverage": periodic_face_coverage_id,
                "owned_cell_graph": (
                    graph_owned_ids,
                    graph_owner_ranks,
                    graph_weights,
                    graph_offsets,
                    graph_neighbor_ids,
                    graph_neighbor_owner_ranks,
                    graph_neighbor_weights,
                ),
                "blocks": tuple(
                    (
                        block.name,
                        block.cell_kind,
                        block.cell_rows,
                        block.cell_ids,
                        block.source_cell_ids,
                        block.vertices,
                        block.vertex_valid,
                        block.coordinate_routes,
                        block.element.element_id,
                        block.element_rows,
                        reference_chart_arguments_identity(block.chart_arguments),
                        block.polyhedron_face_ids,
                        block.polyhedron_faces,
                        block.polyhedron_face_signs,
                        block.cell_vertex_shifts,
                    )
                    for block in blocks
                ),
            }
        )
        if adaptive_result is not None and (
            adaptive_result.result_id == ""
            or adaptive_result.target_local_view_id != local_view_id
            or adaptive_result.target_topology_id != identities["source_topology_id"]
            or adaptive_result.target_geometry_id != identities["source_geometry_id"]
            or adaptive_result.target_source_map_id != source_map_id
            or adaptive_result.target_source_context_id != source_context_id
            or adaptive_result.target_distribution_id != identities["distribution_id"]
            or adaptive_result.target_part_id != identities["part_id"]
            or adaptive_result.target_owner_rank != int(owner_rank)
            or adaptive_result.target_owner_directory_id != identities["distribution_id"]
            or adaptive_result.target_periodic_face_coverage_id
            != periodic_face_coverage_id
            or adaptive_result.target_periodic_topology_id != periodic_topology_id
            or not isinstance(periodic_shell_certificate, Mapping)
            or periodic_shell_certificate.get("size_evidence_id")
            != adaptive_result.size_evidence.evidence_id
        ):
            raise ValueError(
                "Periodic adaptive result is not bound to this accepted target view."
            )
        self.part_name, self.part_id = identities["part_name"], identities["part_id"]
        self.local_mesh_id = local_mesh_id
        self.local_topology_id = local_topology_id
        self.local_geometry_id = local_geometry_id
        self.local_view_id = local_view_id
        self.distribution_id = identities["distribution_id"]
        self.source_id, self.source_revision = (
            identities["source_id"],
            identities["source_revision"],
        )
        self.source_mesh_id = identities["source_mesh_id"]
        self.source_topology_id = identities["source_topology_id"]
        self.source_geometry_id = identities["source_geometry_id"]
        self.source_map_id = source_map_id
        self.source_value = source_value
        self.source_geometry_authority = source_geometry_authority
        self.owner_directory_data = owner_directory_data
        self.owner_rank = int(owner_rank)
        self.geometry_layout_id = identities["geometry_layout_id"]
        self.coordinate_contract_id = identities["coordinate_contract_id"]
        self.specification_id = specification_id
        self.coordinate_contract = coordinate_contract
        self.topological_dimension = int(dimension)
        self.cell_rows, self.cell_ids, self.owned, self.cell_measures = tuple(
            jnp.asarray(value) for value in (cell_rows, cell_ids, owned, measures)
        )
        self.vertex_rows, self.vertex_ids = (
            jnp.asarray(vertex_rows),
            jnp.asarray(vertex_ids),
        )
        self.vertex_coordinates = jnp.asarray(vertex_coordinates)
        self.exact_vertex_coordinates = exact_vertex
        self.coordinate_rows, self.coordinates = (
            jnp.asarray(coordinate_rows),
            jnp.asarray(coordinates),
        )
        self.exact_coordinates = exact_coordinates
        self.source_coordinate_stencil = source_coordinate_stencil
        self.exact_vertex_coordinate_offsets = exact_vertex_coordinate_offsets
        self.entity_set_ids, self.local_entity_ids = entity_set_ids, local_entity_ids
        self.periodic_topology_id = periodic_topology_id
        self.periodic_face_coverage_id = periodic_face_coverage_id
        self.source_context_id = source_context_id
        self.periodic_lift_id = periodic_lift_id
        self.periodic_lift_matrix = (
            None if periodic_lift_matrix is None else jnp.asarray(periodic_lift_matrix)
        )
        self.coordinates_are_lifted = bool(coordinates_are_lifted)
        self.support_fractional_lower = (
            None
            if support_fractional_lower is None
            else jnp.asarray(support_fractional_lower)
        )
        self.support_fractional_upper = (
            None
            if support_fractional_upper is None
            else jnp.asarray(support_fractional_upper)
        )
        self.periodic_cell_ids = (
            None if periodic_arrays[0] is None else jnp.asarray(periodic_arrays[0])
        )
        self.vertex_representative_ids = (
            None if periodic_arrays[1] is None else jnp.asarray(periodic_arrays[1])
        )
        self.vertex_windings = (
            None if periodic_arrays[2] is None else jnp.asarray(periodic_arrays[2])
        )
        self.cell_windings = (
            None if periodic_arrays[3] is None else jnp.asarray(periodic_arrays[3])
        )
        self.vertex_owner_ranks = jnp.asarray(vertex_owner_ranks, dtype=jnp.int32)
        self.source_point_ids = jnp.asarray(source_point_ids)
        self.source_owner_ranks = jnp.asarray(source_owner_ranks, dtype=jnp.int32)
        self.source_owned = jnp.asarray(source_owned)
        self.entity_vertex_offsets, self.entity_vertex_values = (
            tuple(entity_vertex_offsets),
            tuple(entity_vertex_values),
        )
        self.cell_key_offsets = jnp.asarray(cell_key_offsets)
        self.cell_key_source_ids = jnp.asarray(cell_key_source_ids)
        self.cell_key_relative_windings = jnp.asarray(cell_key_relative_windings)
        self.cell_owner_ranks = jnp.asarray(cell_owner_ranks, dtype=jnp.int32)
        self.support_cell_mask = jnp.asarray(support_cell_mask)
        self.owned_cell_ids = jnp.asarray(graph_owned_ids)
        self.owned_cell_owner_ranks = jnp.asarray(graph_owner_ranks, dtype=jnp.int32)
        self.owned_cell_weights = jnp.asarray(graph_weights)
        self.cell_neighbor_offsets = jnp.asarray(graph_offsets)
        self.cell_neighbor_ids = jnp.asarray(graph_neighbor_ids)
        self.cell_neighbor_owner_ranks = jnp.asarray(
            graph_neighbor_owner_ranks, dtype=jnp.int32
        )
        self.cell_neighbor_weights = jnp.asarray(graph_neighbor_weights)
        self.vertex_source_ids = jnp.asarray(vertex_source_ids)
        self.vertex_absolute_windings = jnp.asarray(vertex_absolute_windings)
        self.candidate_revision_id = candidate_revision_id
        self.serial_source_view = serial_source_view
        self.certified_periodic_margin = certified_periodic_margin
        self.periodic_shell_certificate = periodic_shell_certificate
        self.adaptive_result = adaptive_result
        self.graph_migration_evidence = graph_migration_evidence
        self.blocks = blocks

    @classmethod
    def from_owner_local_result(
        cls, record: Mapping[str, Any], /
    ) -> OwnerLocalCellGeometry:
        """Build from an accepted publication boundary; issuer graphs are not retained."""
        if not all(
            key in record
            for key in (
                "part_authority",
                "accepted_composition",
                "owner_directory_witness",
            )
        ):
            raise TypeError(
                "Owner-local publication requires its accepted boundary context."
            )
        return cls(record)

    def reconstruct_local_geometry(
        self, /
    ) -> tuple[CellMesh | None, CellGeometrySpec | None]:
        """Rebuild the canonical local carrier and its exact coordinate map."""
        from ._exact_power_geometry import ExactPowerCellGeometrySource

        source_map = self.source_geometry_authority
        if not isinstance(
            source_map, (CellGeometryCoordinateMap, ExactPowerCellGeometrySource)
        ):
            source_map = None
        mesh, geometry = _reconstruct_owner_local_geometry(
            self.vertex_coordinates,
            self.vertex_ids,
            self.coordinates,
            self.blocks,
            source_map,
            self.vertex_representative_ids,
            self.vertex_windings,
        )
        if (
            (None if mesh is None else mesh.mesh_id) != self.local_mesh_id
            or (None if mesh is None else mesh.topology_id) != self.local_topology_id
            or (None if geometry is None else cell_geometry_id(geometry))
            != self.local_geometry_id
        ):
            raise ValueError(
                "Reconstructed local geometry changed its canonical identities."
            )
        return mesh, geometry

    def require_owner_directory_current(
        self,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str = "owner_local_cell_geometry_directory",
    ) -> None:
        """Rebuild and compare the scientific owner directory for this local view."""
        self.owner_directory_data.require_collective_current(
            execution_group,
            self.owner_rank,
            self.source_point_ids,
            self.source_owner_ranks,
            self.source_owned,
            self.cell_ids,
            self.cell_owner_ranks,
            self.owned,
            self.vertex_ids,
            self.vertex_owner_ranks,
            distribution_id=self.distribution_id,
            source_map_id=self.source_map_id,
            source_context_id=self.source_context_id,
            axis_name=axis_name,
        )

    def owner_ranks_for_ids(
        self,
        identifiers: ConvertibleToArray,
        /,
        *,
        directory_kind: str = "source-points",
    ) -> np.ndarray:
        """Resolve IDs already present in this local view's owner rows."""
        match directory_kind:
            case "source-points":
                bank_ids, bank_owners = self.source_point_ids, self.source_owner_ranks
            case "mesh-cells":
                bank_ids, bank_owners = self.cell_ids, self.cell_owner_ranks
            case "mesh-vertices":
                bank_ids, bank_owners = self.vertex_ids, self.vertex_owner_ranks
            case _:
                raise ValueError("Unknown local owner directory kind.")
        values = np.asarray(identifiers)
        ids = np.asarray(bank_ids, dtype=np.int64)
        owners = np.asarray(bank_owners, dtype=np.int32)
        if (
            values.ndim != 1
            or values.dtype.kind not in "iu"
            or np.any(values < 0)
            or owners.shape != ids.shape
        ):
            raise ValueError("Local owner lookup IDs or rows are malformed.")
        lookup = {
            int(identifier): int(owner)
            for identifier, owner in zip(ids, owners, strict=True)
        }
        try:
            return np.asarray(
                [lookup[int(identifier)] for identifier in values],
                dtype=np.int32,
            )
        except KeyError as error:
            raise ValueError(
                "Owner lookup ID is outside this rank's local source/mesh shard."
            ) from error

    def resolve_owner_ranks(
        self,
        identifiers: ConvertibleToArray,
        execution_group: ExecutionGroup,
        /,
        *,
        directory_kind: str = "source-points",
        axis_name: str = "owner_local_cell_geometry_lookup",
    ) -> np.ndarray:
        from ..meshing._distribution import _owner_directory_lookup

        self.require_owner_directory_current(
            execution_group, axis_name=f"{axis_name}_current"
        )
        directory = self.owner_directory_data
        match directory_kind:
            case "source-points":
                shard_ids, shard_owners = (
                    directory.source_directory_ids,
                    directory.source_directory_owner_ranks,
                )
            case "mesh-cells":
                shard_ids, shard_owners = (
                    directory.cell_directory_ids,
                    directory.cell_directory_owner_ranks,
                )
            case "mesh-vertices":
                shard_ids, shard_owners = (
                    directory.vertex_directory_ids,
                    directory.vertex_directory_owner_ranks,
                )
            case _:
                raise ValueError("Unknown owner directory kind.")
        query_ids = np.asarray(identifiers)
        return _owner_directory_lookup(
            query_ids,
            np.asarray(shard_ids, dtype=np.int64),
            np.asarray(shard_owners, dtype=np.int32),
            directory.rank,
            directory.part_count,
            execution_group,
            axis_name,
            directory_kind,
        )

    def scope(
        self,
        dimension: int,
        entity_ids: ConvertibleToArray,
        /,
        *,
        entity_set_id: str | None = None,
    ) -> MeshingScope:
        """Build a scope only from entity rows resident in this compact local view."""
        from ..meshing._scope import MeshingEntityKind, MeshingScope

        resolved_set_id, available = self.entity_binding(dimension)
        if entity_set_id is not None and entity_set_id != resolved_set_id:
            raise ValueError("Entity set does not belong to this local mesh view.")
        identifiers = np.asarray(entity_ids)
        if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
            raise TypeError("Owner-local scope IDs must be an integer vector.")
        scope = MeshingScope(
            self.part_name,
            self.part_id,
            MeshingEntityKind.MESH,
            dimension,
            resolved_set_id,
            identifiers,
        )
        if not np.all(np.isin(np.asarray(scope.entity_ids), available)):
            raise ValueError(
                "Owner-local scope contains IDs outside its local entity bank."
            )
        return scope

    @classmethod
    def from_stored_fields(
        cls,
        value: OwnerLocalCellGeometry,
        execution_group: ExecutionGroup,
        /,
        *,
        current_specification: SurfaceMeshingSpec | VolumeMeshingSpec | None = None,
        graph_partition_plan: GraphPartitionPlan | None = None,
        axis_name: str = "owner_local_cell_geometry_restore",
    ) -> OwnerLocalCellGeometry:
        """Rebuild scientific rows after current-group and exact-source replay."""
        import jax

        from ..graph._partition import GraphPartitionPlan
        from ..meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
        from ..meshing._distribution import _owner_group_fail_if_any
        from ..meshing._periodic import PeriodicPointOrbits
        from ..meshing.providers._native_periodic import (
            _owner_local_periodic_source_content_identity,
            _require_owner_local_periodic_adaptive_target_current,
            _require_owner_local_periodic_view_current,
            NativePeriodicSource,
        )
        from ._exact_power_geometry import ExactPowerCellGeometrySource

        if not isinstance(value, cls):
            raise TypeError("value must be OwnerLocalCellGeometry.")
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be the current ExecutionGroup.")
        if not isinstance(axis_name, str) or not axis_name.strip():
            raise ValueError("axis_name must be a non-empty string.")
        if graph_partition_plan is not None and not isinstance(
            graph_partition_plan, GraphPartitionPlan
        ):
            raise TypeError("graph_partition_plan must be GraphPartitionPlan or None.")
        if value.specification_id is not None and (
            not isinstance(current_specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
            or current_specification.specification_id != value.specification_id
        ):
            raise ValueError(
                "Periodic restore requires its current exact meshing specification."
            )
        if value.adaptive_result is not None and (
            not isinstance(current_specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
            or current_specification.specification_id
            != value.adaptive_result.specification_id
        ):
            raise ValueError(
                "Adaptive periodic restore requires its current exact specification."
            )
        members = execution_group.spec.process_indices
        devices = execution_group.devices
        try:
            local_rank = members.index(jax.process_index())
        except ValueError:
            local_rank = -1
        local_error = None
        if (
            len(devices) != len(members)
            or tuple(device.process_index for device in devices) != members
            or not execution_group.is_member
            or local_rank < 0
            or local_rank != value.owner_rank
        ):
            local_error = "Stored owner-local view does not match its current group rank."
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            axis_name,
            "stored owner-local view rank",
        )

        try:
            if not isinstance(value.coordinate_contract, SpatialCoordinateContract):
                raise TypeError(
                    "Stored owner-local coordinate contract has the wrong type."
                )
            coordinate_contract = SpatialCoordinateContract.from_dict(
                value.coordinate_contract.to_dict()
            )
            if coordinate_contract.spatial_id != value.coordinate_contract_id:
                raise ValueError(
                    "Stored owner-local coordinate contract identity changed."
                )
            local_error = None
        except Exception as error:
            coordinate_contract = value.coordinate_contract
            local_error = str(error)
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            axis_name,
            "stored owner-local coordinate contract",
        )

        source_value = value.source_value
        source_authority = value.source_geometry_authority
        try:
            if isinstance(source_value, NativePeriodicSource):
                source_value = NativePeriodicSource.from_stored_fields(source_value)
            elif value.periodic_topology_id is not None and not value.serial_source_view:
                raise TypeError(
                    "Stored periodic view lacks its actual NativePeriodicSource data."
                )
            if isinstance(source_authority, NativePeriodicSource):
                source_authority = NativePeriodicSource.from_stored_fields(
                    source_authority
                )
            elif type(source_authority) is ExactPowerCellGeometrySource:
                target_periodic_cell = None
                if source_authority.periodic_source_context is not None:
                    if not isinstance(
                        source_value, NativePeriodicSource
                    ) or not isinstance(source_value.domain, PeriodicPointOrbits):
                        raise ValueError(
                            "Stored exact periodic power map requires its native point-orbit source."
                        )
                    target_periodic_cell = source_value.domain.cell
                source_authority = ExactPowerCellGeometrySource.from_stored_fields(
                    source_authority,
                    target_periodic_cell=target_periodic_cell,
                )
                context = source_authority.periodic_source_context
                if (
                    source_authority.source_id != value.source_id
                    or source_authority.source_revision != value.source_revision
                    or source_authority.source_map_id != value.source_map_id
                    or (None if context is None else context.context_id)
                    != value.source_context_id
                ):
                    raise ValueError(
                        "Stored exact power source map changed its local binding."
                    )
            elif isinstance(source_authority, CellGeometryCoordinateMap):
                if (
                    source_authority.source_id != value.source_id
                    or source_authority.source_revision != value.source_revision
                    or source_authority.source_map_id != value.source_map_id
                ):
                    raise ValueError(
                        "Stored coordinate map differs from its local source identity."
                    )
            elif isinstance(source_authority, OwnerLocalCellGeometry):
                raise ValueError(
                    "Nested owner-local source views require their own current-group restore."
                )
            elif source_authority is not None:
                raise TypeError(
                    "Stored owner-local source map has an unsupported concrete type."
                )
            if source_value is not None and (
                getattr(source_value, "source_id", value.source_id) != value.source_id
                or getattr(source_value, "source_revision", value.source_revision)
                != value.source_revision
            ):
                raise ValueError(
                    "Stored scientific source data changed its source identity."
                )
            local_error = None
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            axis_name,
            "stored owner-local source and geometry",
        )

        replayed_candidate = None
        source_content_id = None
        if value.periodic_topology_id is not None and not value.serial_source_view:
            if (
                not isinstance(source_value, NativePeriodicSource)
                or not isinstance(source_value.domain, PeriodicPointOrbits)
                or source_value.point_owner_ranks is None
            ):
                raise TypeError(
                    "Stored periodic source lacks exact owner-local point data."
                )
            source_content_id = _owner_local_periodic_source_content_identity(
                source_value,
                local_rank,
                execution_group,
                f"{axis_name}_source",
            )
            if source_content_id != value.owner_directory_data.source_manifest_id:
                local_error = (
                    "Restored source data differs from its owner-directory manifest."
                )
            else:
                local_error = None
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                "stored source-directory manifest",
            )

        try:
            value.require_owner_directory_current(
                execution_group,
                axis_name=f"{axis_name}_directory",
            )
            local_error = None
        except Exception as error:
            local_error = str(error)
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            axis_name,
            "stored owner-directory currentness",
        )

        if value.periodic_topology_id is not None and not value.serial_source_view:
            try:
                if (
                    not isinstance(source_value, NativePeriodicSource)
                    or source_content_id is None
                    or not isinstance(
                        source_authority,
                        (NativePeriodicSource, ExactPowerCellGeometrySource),
                    )
                ):
                    raise TypeError(
                        "Periodic restore lacks actual source and exact map data."
                    )
                if current_specification is None:
                    raise ValueError(
                        "Periodic restore requires its current typed specification."
                    )
                replayed_candidate = _require_owner_local_periodic_view_current(
                    value,
                    source_value,
                    source_authority,
                    source_content_id,
                    current_specification,
                    execution_group,
                    axis_name,
                    graph_partition_plan=graph_partition_plan,
                )
                local_error = None
            except Exception as error:
                local_error = str(error)
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                "stored periodic source/quotient replay",
            )

        record = {
            "part_name": value.part_name,
            "part_id": value.part_id,
            "local_mesh_id": value.local_mesh_id,
            "local_topology_id": value.local_topology_id,
            "local_geometry_id": value.local_geometry_id,
            "distribution_id": value.distribution_id,
            "source_id": value.source_id,
            "source_revision": value.source_revision,
            "source_mesh_id": value.source_mesh_id,
            "source_topology_id": value.source_topology_id,
            "source_geometry_id": value.source_geometry_id,
            "source_map_id": value.source_map_id,
            "source_context_id": value.source_context_id,
            "geometry_layout_id": value.geometry_layout_id,
            "coordinate_contract_id": value.coordinate_contract_id,
            "specification_id": value.specification_id,
            "coordinate_contract": coordinate_contract,
            "topological_dimension": value.topological_dimension,
            "cell_rows": value.cell_rows,
            "cell_ids": value.cell_ids,
            "owned": value.owned,
            "cell_measures": value.cell_measures,
            "cell_owner_ranks": value.cell_owner_ranks,
            "owned_cell_ids": value.owned_cell_ids,
            "owned_cell_owner_ranks": value.owned_cell_owner_ranks,
            "owned_cell_weights": value.owned_cell_weights,
            "cell_neighbor_offsets": value.cell_neighbor_offsets,
            "cell_neighbor_ids": value.cell_neighbor_ids,
            "cell_neighbor_owner_ranks": value.cell_neighbor_owner_ranks,
            "cell_neighbor_weights": value.cell_neighbor_weights,
            "support_cell_mask": value.support_cell_mask,
            "vertex_rows": value.vertex_rows,
            "vertex_ids": value.vertex_ids,
            "vertex_coordinates": value.vertex_coordinates,
            "exact_vertex_coordinates": value.exact_vertex_coordinates,
            "coordinate_rows": value.coordinate_rows,
            "coordinates": value.coordinates,
            "exact_coordinates": value.exact_coordinates,
            "source_coordinate_stencil": value.source_coordinate_stencil,
            "exact_vertex_coordinate_offsets": value.exact_vertex_coordinate_offsets,
            "entity_set_ids": dict(value.entity_set_ids),
            "local_entity_ids": dict(value.local_entity_ids),
            "entity_vertex_offsets": dict(value.entity_vertex_offsets),
            "entity_vertex_values": dict(value.entity_vertex_values),
            "cell_key_offsets": value.cell_key_offsets,
            "cell_key_source_ids": value.cell_key_source_ids,
            "cell_key_relative_windings": value.cell_key_relative_windings,
            "vertex_source_ids": value.vertex_source_ids,
            "source_point_ids": value.source_point_ids,
            "source_owner_ranks": value.source_owner_ranks,
            "source_owned": value.source_owned,
            "vertex_owner_ranks": value.vertex_owner_ranks,
            "vertex_absolute_windings": value.vertex_absolute_windings,
            "candidate_revision_id": value.candidate_revision_id,
            "certified_periodic_margin": value.certified_periodic_margin,
            "periodic_shell_certificate": value.periodic_shell_certificate,
            "adaptive_result": value.adaptive_result,
            "graph_migration_evidence": value.graph_migration_evidence,
            "periodic_topology_id": value.periodic_topology_id,
            "periodic_face_coverage_id": value.periodic_face_coverage_id,
            "periodic_cell_ids": value.periodic_cell_ids,
            "vertex_representative_ids": value.vertex_representative_ids,
            "vertex_windings": value.vertex_windings,
            "cell_windings": value.cell_windings,
            "periodic_lift_id": value.periodic_lift_id,
            "periodic_lift_matrix": value.periodic_lift_matrix,
            "coordinates_are_lifted": value.coordinates_are_lifted,
            "support_fractional_lower": value.support_fractional_lower,
            "support_fractional_upper": value.support_fractional_upper,
            "blocks": value.blocks,
            "rank": local_rank,
            "serial_source_view": value.serial_source_view,
            "source_value": source_value,
            "source_geometry_authority": source_authority,
            "owner_directory_data": value.owner_directory_data,
        }
        record["_compact_restore_validation"] = _COMPACT_OWNER_LOCAL_RESTORE
        try:
            renewed = cls(record)
            if renewed.local_view_id != value.local_view_id:
                raise ValueError("Reconstructed compact view changed its local identity.")
            local_error = None
        except Exception as error:
            renewed = None
            local_error = str(error)
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            axis_name,
            "stored owner-local view reconstruction",
        )
        if renewed is None:
            raise RuntimeError(
                "Collectively accepted owner-local reconstruction returned no view."
            )
        if renewed.adaptive_result is not None:
            if replayed_candidate is None or current_specification is None:
                local_error = (
                    "Adaptive periodic restore lacks current source/specification inputs."
                )
            else:
                try:
                    if (
                        source_content_id is None
                        or not isinstance(source_value, NativePeriodicSource)
                        or not isinstance(
                            source_authority,
                            (NativePeriodicSource, ExactPowerCellGeometrySource),
                        )
                    ):
                        raise ValueError(
                            "Adaptive periodic restore lacks current typed source/map data."
                        )
                    _require_owner_local_periodic_adaptive_target_current(
                        renewed,
                        source_value,
                        source_authority,
                        source_content_id,
                        current_specification,
                        execution_group,
                        local_rank,
                        axis_name=f"{axis_name}_target",
                    )
                    renewed.adaptive_result.revalidate(
                        replayed_candidate,
                        current_specification,
                        execution_group,
                        local_rank,
                        renewed,
                        graph_partition_plan,
                    )
                    local_error = None
                except Exception as error:
                    local_error = str(error)
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                "stored periodic adaptive replay",
            )
        return renewed

    def require_source(
        self,
        *,
        part_id: str,
        source_mesh_id: str,
        source_topology_id: str,
        source_geometry_id: str,
        source_map_id: str | None,
    ) -> None:
        expected = (
            part_id,
            source_mesh_id,
            source_topology_id,
            source_geometry_id,
            source_map_id,
        )
        actual = (
            self.part_id,
            self.source_mesh_id,
            self.source_topology_id,
            self.source_geometry_id,
            self.source_map_id,
        )
        if actual != expected:
            raise ValueError(
                "Owner-local geometry does not bind the exact source revision."
            )

    def entity_binding(self, dimension: int, /) -> tuple[str, np.ndarray]:
        sets = dict(self.entity_set_ids)
        entities = dict(self.local_entity_ids)
        if dimension not in sets or dimension not in entities:
            raise ValueError("Owner-local view does not expose that entity dimension.")
        return sets[dimension], np.asarray(entities[dimension], dtype=np.int64)

    def point_coordinates(self, entity_ids: ConvertibleToArray, /) -> Array:
        identifiers = np.asarray(entity_ids)
        if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
            raise TypeError("Owner-local point IDs must be an integer vector.")
        ids = np.asarray(self.vertex_ids, dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        positions = np.searchsorted(ids[order], identifiers.astype(np.int64))
        if np.any(positions >= order.size):
            raise ValueError("Owner-local point IDs are outside the compact vertex bank.")
        rows = order[positions]
        if not np.array_equal(ids[rows], identifiers.astype(np.int64)):
            raise ValueError("Owner-local point IDs are outside the compact vertex bank.")
        rows_ = jnp.asarray(rows, dtype=jnp.int32)
        points = self.vertex_coordinates[rows_]
        if self.coordinates_are_lifted or self.periodic_lift_matrix is None:
            return points
        shifts = jnp.asarray(self.vertex_absolute_windings)[rows_]
        return points + shifts @ self.periodic_lift_matrix

    def entity_vertex_loops(
        self, dimension: int, entity_ids: ConvertibleToArray, /
    ) -> tuple[np.ndarray, ...]:
        identifiers = np.asarray(entity_ids)
        if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
            raise TypeError("Owner-local entity IDs must be an integer vector.")
        local_ids = dict(self.local_entity_ids).get(dimension)
        offsets = dict(self.entity_vertex_offsets).get(dimension)
        values = dict(self.entity_vertex_values).get(dimension)
        if local_ids is None or offsets is None or values is None:
            raise ValueError("Owner-local entity incidence is unavailable.")
        ids = np.asarray(local_ids, dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        positions = np.searchsorted(ids[order], identifiers.astype(np.int64))
        if np.any(positions >= order.size):
            raise ValueError(
                "Owner-local entity IDs are outside the compact incidence bank."
            )
        rows = order[positions]
        if not np.array_equal(ids[rows], identifiers.astype(np.int64)):
            raise ValueError(
                "Owner-local entity IDs are outside the compact incidence bank."
            )
        offsets_ = np.asarray(offsets, dtype=np.int64)
        values_ = np.asarray(values, dtype=np.int32)
        return tuple(values_[offsets_[row] : offsets_[row + 1]] for row in rows)


@final
class OwnerLocalFieldSupport(StrictModule, NonTrainableState):
    """Exact field-query support bound to one owner-local source view."""

    local_view_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_map_id: str | None = eqx.field(static=True)
    source_context_id: str | None = eqx.field(static=True)
    owner_directory_id: str = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    cell_ids: Array
    locator_id: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        view: OwnerLocalCellGeometry,
        locator: AbstractCellLocator,
        block_name: str,
        /,
    ) -> None:
        if not isinstance(view, OwnerLocalCellGeometry):
            raise TypeError("view must be an OwnerLocalCellGeometry.")
        if not isinstance(block_name, str) or not block_name:
            raise ValueError("block_name must be a non-empty string.")
        if not isinstance(
            locator, (PreparedMappedCellLocator, PreparedPolyhedralCellLocator)
        ):
            raise TypeError(
                "Owner-local field support requires a mapped or polyhedral locator."
            )
        cell_map = locator.cell_map
        if not isinstance(
            cell_map, (PreparedOwnerLocalMappedCellMap, PreparedPolyhedralCellMap)
        ):
            raise TypeError(
                "Owner-local field support requires a source-bound local map."
            )
        block = next((item for item in view.blocks if item.name == block_name), None)
        if block is None:
            raise ValueError(f"Owner-local geometry has no block {block_name!r}.")
        ids = np.asarray(cell_map.cell_global_ids, dtype=np.int64)
        expected = np.asarray(block.cell_ids, dtype=np.int64)
        if (
            cell_map.local_view_id != view.local_view_id
            or cell_map.source_mesh_id != view.source_mesh_id
            or cell_map.source_geometry_id != view.source_geometry_id
            or cell_map.source_map_id != view.source_map_id
            or ids.shape != expected.shape
            or not np.array_equal(ids, expected)
        ):
            raise ValueError(
                "Owner-local field locator is stale or does not cover the exact "
                "source-bound block."
            )
        directory_id = view.distribution_id
        local_ids = np.asarray(view.cell_ids, dtype=np.int64)
        order = np.argsort(local_ids, kind="stable")
        positions = np.searchsorted(local_ids[order], ids)
        if ids.size and np.any(positions >= order.size):
            raise ValueError("Owner-local field block is outside its compact cell bank.")
        rows = order[positions]
        local_owners = np.asarray(view.cell_owner_ranks, dtype=np.int32)
        if ids.size and (
            not np.array_equal(local_ids[rows], ids)
            or local_owners.shape != local_ids.shape
            or np.any(local_owners[rows] < 0)
        ):
            raise ValueError(
                "Owner-local field block has missing or duplicate owner evidence."
            )
        self.local_view_id = view.local_view_id
        self.source_mesh_id = view.source_mesh_id
        self.source_topology_id = view.source_topology_id
        self.source_map_id = view.source_map_id
        self.source_context_id = view.source_context_id
        self.owner_directory_id = directory_id
        self.block_name = block_name
        self.cell_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.locator_id = locator.locator_id
        self.ambient_dimension = cell_map.ambient_dimension
        self.support_id = canonical_fingerprint(
            {
                "kind": "owner-local-field-query-support",
                "local_view": view.local_view_id,
                "source_mesh": view.source_mesh_id,
                "source_topology": view.source_topology_id,
                "source_map": view.source_map_id,
                "source_context": view.source_context_id,
                "owner_directory": directory_id,
                "block": block_name,
                "cell_ids": ids,
                "locator": locator.locator_id,
            }
        )

    @property
    def kind(self) -> Any:
        from ..geometry import GeometryKind

        return GeometryKind.REGION

    def require_kernel(self, kernel: Any, /) -> None:
        locator = getattr(kernel, "locator", None)
        cell_map = getattr(locator, "cell_map", None)
        if (
            locator is None
            or getattr(locator, "locator_id", None) != self.locator_id
            or getattr(cell_map, "local_view_id", None) != self.local_view_id
            or getattr(cell_map, "source_map_id", None) != self.source_map_id
            or not np.array_equal(
                np.asarray(getattr(cell_map, "cell_global_ids", ()), dtype=np.int64),
                np.asarray(self.cell_ids, dtype=np.int64),
            )
        ):
            raise ValueError(
                "Owner-local field support does not bind the reconstruction locator."
            )


def cell_location_status(status: Array, /) -> Array:
    """Map `CellLocationStatus` codes onto field query statuses."""
    return jnp.where(
        status == int(CellLocationStatus.LOCATED),
        int(FieldQueryStatus.VALID),
        jnp.where(
            status == int(CellLocationStatus.OUTSIDE),
            int(FieldQueryStatus.OUTSIDE_SUPPORT),
            jnp.where(
                status == int(CellLocationStatus.NONFINITE),
                int(FieldQueryStatus.NONFINITE),
                int(FieldQueryStatus.LOCATION_FAILED),
            ),
        ),
    ).astype(jnp.int32)


def simplicial_mesh_support_geometry(
    locator: AbstractCellLocator, support_id: str, /
) -> Any:
    """Compile the exact region covered by affine simplicial or planar-facet tensor cells."""
    from ..geometry.simplicial import MeshRegion
    from ..geometry.simplicial._io import planar_region_from_triangles
    from .fem._reference import FiniteElementSpec

    if isinstance(locator, PreparedPolyhedralCellLocator):
        return locator.support_geometry(support_id)
    if not isinstance(
        locator, (PreparedSimplicialCellLocator, PreparedMappedCellLocator)
    ):
        raise TypeError(
            "Derived mesh support requires a canonical FE or polyhedral locator."
        )
    cell_map = locator.cell_map
    element = cell_map.coordinate_element
    if not isinstance(element, FiniteElementSpec) or element.degree != 1:
        raise ValueError(
            "Curved or composed cell maps require explicit evidence-backed support_geometry."
        )
    coordinates = np.asarray(locator.coordinates, dtype=np.float64)
    cells = np.asarray(cell_map.coordinate_dofs, dtype=np.int64)
    feature_id = f"finite-element-support:{support_id}"
    match element.cell_kind:
        case "triangle":
            source = planar_region_from_triangles(
                coordinates, cells, recenter=False, feature_id=feature_id
            )
        case "tetrahedron":
            faces = np.concatenate(
                (
                    cells[:, [1, 2, 3]],
                    cells[:, [0, 3, 2]],
                    cells[:, [0, 1, 3]],
                    cells[:, [0, 2, 1]],
                )
            )
            keys = np.sort(faces, axis=1)
            _, inverse, counts = np.unique(
                keys, axis=0, return_inverse=True, return_counts=True
            )
            boundary = faces[counts[inverse.reshape((-1,))] == 1]
            used, compact = np.unique(boundary, return_inverse=True)
            source = MeshRegion(
                coordinates[used],
                compact.reshape(boundary.shape).astype(np.int32),
                feature_id=feature_id,
            )
        case "quadrilateral" | "hexahedron":
            return _tensor_mesh_support_geometry(
                coordinates,
                cells,
                element.cell_kind,
                feature_id,
                locator,
            )
        case "interval":
            from ..geometry.simplicial._support_region import compile_simplicial_support

            return compile_simplicial_support(locator, support_id)
        case kind:
            if kind.startswith("simplex:"):
                from ..geometry.simplicial._support_region import (
                    compile_simplicial_support,
                )

                return compile_simplicial_support(locator, support_id)
            if kind.startswith("tensor:"):
                return _tensor_mesh_support_geometry(
                    coordinates, cells, kind, feature_id, locator
                )
            raise ValueError(
                f"The support of {kind!r} FE cells is not derived; pass an explicit support_geometry."
            )
    return source.compile()


def _tensor_mesh_support_geometry(
    coordinates: np.ndarray,
    cells: np.ndarray,
    cell_kind: str,
    feature_id: str,
    locator: AbstractCellLocator,
    /,
) -> Any:
    from ..geometry.simplicial import MeshRegion
    from ..geometry.simplicial._io import planar_region_from_triangles

    reference = reference_cell_topology(cell_kind)
    if reference.dimension == 1:
        from ..geometry.simplicial._support_region import compile_simplicial_support

        return compile_simplicial_support(locator, feature_id)
    if reference.dimension == 2:
        triangles = cells[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)]
        return planar_region_from_triangles(
            coordinates,
            triangles.reshape((-1, 3)),
            recenter=False,
            feature_id=feature_id,
        ).compile()
    if reference.dimension != 3:
        raise ValueError("Tensor support regions are prepared for 2-D/3-D mesh cells.")
    faces = cells[:, np.asarray(reference.entities[2], dtype=np.int32)]
    keys = np.sort(faces.reshape((-1, 4)), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Tensor support requires manifold faces.")
    exterior = counts[inverse] == 1
    boundary = faces.reshape((-1, 4))[exterior].copy()
    parent = np.repeat(np.arange(cells.shape[0]), faces.shape[1])[exterior]
    points = coordinates[boundary]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=2), axis=1)
    if np.any(norm <= np.finfo(np.float64).eps * np.maximum(scale**2, 1.0)):
        raise ValueError("Tensor support boundary faces must be nondegenerate.")
    residual = np.abs(np.sum((points[:, 3] - points[:, 0]) * normals, axis=1)) / norm
    if np.any(residual > 1e-10 * np.maximum(scale, 1.0)):
        raise ValueError("Warped tensor faces need an explicit curved support_geometry.")
    centers = np.mean(coordinates[cells[parent]], axis=1)
    inward = np.sum(normals * (np.mean(points, axis=1) - centers), axis=1) < 0.0
    boundary[inward] = boundary[inward][:, (0, 3, 2, 1)]
    triangles = boundary[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)].reshape(
        (-1, 3)
    )
    used, compact = np.unique(triangles, return_inverse=True)
    return MeshRegion(
        coordinates[used],
        compact.reshape(triangles.shape).astype(np.int32),
        feature_id=feature_id,
    ).compile()


def verify_mesh_support_geometry(
    geometry: Any, locator: AbstractCellLocator, /, *, tolerance: float
) -> None:
    from ..geometry import GeometryCapability
    from ..geometry._contracts import CompiledGeometry
    from .fem._cell_map import PreparedFiniteElementCellMap

    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("support_geometry must be a CompiledGeometry.")
    if not isinstance(
        locator,
        (
            PreparedPolyhedralCellLocator,
            PreparedSimplicialCellLocator,
            PreparedMappedCellLocator,
        ),
    ):
        raise TypeError("Support verification requires a canonical cell locator.")
    coordinates = jnp.asarray(locator.coordinates)
    field = np.asarray(geometry.boundary_field(coordinates))
    if np.any(field > tolerance):
        raise ValueError("FE mesh vertices lie outside the support geometry.")
    if not geometry.has_capability(GeometryCapability.INTERIOR_MEASURE):
        raise ValueError(
            "An explicit support_geometry needs an interior measure to evidence that "
            "the FE mesh covers it."
        )
    if isinstance(locator, PreparedPolyhedralCellLocator):
        native_support = locator.support_geometry("coverage-verification")
        if not isinstance(
            native_support, CompiledGeometry
        ) or not native_support.has_capability(GeometryCapability.INTERIOR_MEASURE):
            raise ValueError("Polyhedral mesh support has no exact interior measure.")
        mesh_measure = float(np.asarray(native_support.measure))
    else:
        cell_map = locator.cell_map
        if not isinstance(cell_map, PreparedFiniteElementCellMap):
            raise ValueError(
                "Mapped source supports require DomainCoverageCertificate evidence."
            )
        if cell_map.coordinate_element.degree != 1:
            raise ValueError(
                "Covering evidence is exact only for affine FE cell maps; curved maps "
                "need an explicit inverse provider with its own support evidence."
            )
        reference = jnp.full(
            (cell_map.cell_count, cell_map.reference_dimension),
            1.0 / (cell_map.reference_dimension + 1),
            dtype=coordinates.dtype,
        )
        evaluation = cell_map.evaluate(
            coordinates, jnp.arange(cell_map.cell_count), reference
        )
        reference_topology = reference_cell_topology(
            cell_map.coordinate_element.cell_kind
        )
        tensor = reference_topology.name in (
            "quadrilateral",
            "hexahedron",
        ) or reference_topology.name.startswith("tensor:")
        if tensor:
            from .._polynomial._orthogonal import legendre_rule_data

            axis_rule = legendre_rule_data(2)
            nodes = 0.5 * (np.asarray(axis_rule.nodes) + 1.0)
            weights = 0.5 * np.asarray(axis_rule.weights)
            indices = np.asarray(
                tuple(product(range(2), repeat=cell_map.reference_dimension)),
                dtype=np.int32,
            )
            references = np.tile(nodes[indices], (cell_map.cell_count, 1))
            cell_indices = np.repeat(
                np.arange(cell_map.cell_count, dtype=np.int32), indices.shape[0]
            )
            evaluation = cell_map.evaluate(
                coordinates, jnp.asarray(cell_indices), jnp.asarray(references)
            )
            quadrature_weights = np.tile(
                np.prod(weights[indices], axis=1), cell_map.cell_count
            )
            mesh_measure = float(
                np.sum(np.abs(np.asarray(evaluation.determinant)) * quadrature_weights)
            )
        else:
            mesh_measure = float(
                np.sum(np.abs(np.asarray(evaluation.determinant)))
            ) / factorial(cell_map.reference_dimension)
    geometry_measure = float(np.asarray(geometry.measure))
    if abs(mesh_measure - geometry_measure) > tolerance * max(1.0, geometry_measure):
        raise ValueError(
            "The FE mesh measure differs from the support geometry measure; the "
            "mesh does not cover the geometry."
        )


def tensor_box_support_geometry(
    lower: np.ndarray,
    upper: np.ndarray,
    support_geometry: Any,
    /,
    *,
    feature_id: str,
    tolerance: float,
) -> Any:
    """Return the compiled support box, or verify an explicit box geometry.

    `support_geometry=None` compiles an `Orthotope` spanning `[lower, upper]`.
    An explicit region is admitted only when its bounds equal the box and its
    interior measure equals the box measure: a region inside the box's bounding
    box with the box's measure is the box.
    """
    from ..geometry import CompiledGeometry, GeometryCapability, GeometryKind, Orthotope

    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        lower_.ndim != 1
        or upper_.shape != lower_.shape
        or not np.all(np.isfinite(lower_) & np.isfinite(upper_))
        or np.any(upper_ <= lower_)
    ):
        raise ValueError("A support box needs finite, ordered lower and upper corners.")
    limit = float(tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError("support_tolerance must be finite and non-negative.")
    if support_geometry is None:
        return Orthotope(
            0.5 * (lower_ + upper_), upper_ - lower_, feature_id=feature_id
        ).compile()
    if not isinstance(support_geometry, CompiledGeometry):
        raise TypeError("support_geometry must be a CompiledGeometry or None.")
    if support_geometry.kind is not GeometryKind.REGION:
        raise ValueError("support_geometry must be a region geometry.")
    if support_geometry.ambient_dimension != lower_.size:
        raise ValueError("support_geometry dimension must equal the box dimension.")
    if not support_geometry.has_capability(GeometryCapability.INTERIOR_MEASURE):
        raise ValueError(
            "An explicit support_geometry needs an interior measure to evidence "
            "that it is the reconstruction's tensor box."
        )
    box = np.stack((lower_, upper_))
    bounds = np.asarray(support_geometry.bounds, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(box))))
    if bounds.shape != box.shape or np.max(np.abs(bounds - box)) > limit * scale:
        raise ValueError("support_geometry bounds differ from the reconstruction box.")
    box_measure = float(np.prod(upper_ - lower_))
    measure = float(np.asarray(support_geometry.measure))
    if abs(measure - box_measure) > limit * max(1.0, box_measure):
        raise ValueError(
            "support_geometry measure differs from the reconstruction box; the "
            "geometry is not the tensor box the reconstruction covers."
        )
    return support_geometry


class _OwnerLocalPeriodicTargetMaterialization(NamedTuple):
    mesh: CellMesh | None
    geometry: CellGeometrySpec | None
    source_map: CellGeometryCoordinateMap | None
    source_coordinate_stencil: VertexInterpolationStencil | None
    vertex_coordinates: np.ndarray
    exact_vertex_coordinates: tuple[tuple[Fraction, ...], ...]
    exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None
    vertex_source_ids: np.ndarray
    vertex_absolute_windings: np.ndarray
    blocks: tuple[OwnerLocalCellBlock, ...]
    cell_key_offsets: np.ndarray
    cell_key_source_ids: np.ndarray
    cell_key_relative_windings: np.ndarray
    cell_windings: np.ndarray
    exact_cell_measures: tuple[Fraction, ...]


def _materialize_owner_local_periodic_target(
    view: OwnerLocalCellGeometry,
    periodic_cell: PeriodicCell,
    exact_root_coordinates: tuple[tuple[int, tuple[Fraction, ...]], ...],
    target_root_float_coordinates: np.ndarray,
    target_source_entity_set_id: str,
    target_vertex_source_ids: np.ndarray,
    target_vertex_absolute_windings: np.ndarray,
    /,
    *,
    source_map: CellGeometryCoordinateMap | None = None,
) -> _OwnerLocalPeriodicTargetMaterialization:
    from ..meshing._lineage import VertexInterpolationStencil
    from ._exact_power_geometry import ExactPowerCellGeometrySource
    from ._periodic_exact import identification_orders, reduce_periodic_exponents

    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("view must be OwnerLocalCellGeometry.")
    if not isinstance(periodic_cell, PeriodicCell):
        raise TypeError("periodic_cell must be PeriodicCell.")
    if (
        view.periodic_topology_id is None
        or view.periodic_lift_id != periodic_cell.cell_id
        or not view.coordinates_are_lifted
        or view.periodic_lift_matrix is None
        or not np.array_equal(
            np.asarray(view.periodic_lift_matrix, dtype=np.float64),
            np.asarray(periodic_cell.vectors, dtype=np.float64),
        )
    ):
        raise ValueError("Motion target lattice differs from the current quotient view.")
    dimension = periodic_cell.ambient_dimension
    periodic_rank = int(np.asarray(periodic_cell.vectors).shape[0])
    if not isinstance(exact_root_coordinates, tuple):
        raise TypeError("exact_root_coordinates must be an immutable exact bank.")
    root_ids: list[int] = []
    root_points: dict[int, tuple[Fraction, ...]] = {}
    for record in exact_root_coordinates:
        if not isinstance(record, tuple) or len(record) != 2:
            raise TypeError("Exact root-coordinate rows must be (ID, point) tuples.")
        identifier, point = record
        if (
            isinstance(identifier, (bool, np.bool_))
            or not isinstance(identifier, (int, np.integer))
            or int(identifier) < 0
            or not isinstance(point, tuple)
            or len(point) != dimension
            or any(not isinstance(value, Fraction) for value in point)
        ):
            raise TypeError("Exact root-coordinate rows are malformed.")
        key = int(identifier)
        if key in root_points:
            raise ValueError("Exact root-coordinate IDs must be unique.")
        root_ids.append(key)
        root_points[key] = point
    if root_ids != sorted(root_ids):
        raise ValueError("Exact root-coordinate rows must be sorted by stable source ID.")
    root_float = np.asarray(target_root_float_coordinates, dtype=np.float64)
    if (
        root_float.shape != (len(root_ids), dimension)
        or np.any(~np.isfinite(root_float))
        or any(
            tuple(Fraction.from_float(float(value)) for value in root_float[row])
            != root_points[identifier]
            for row, identifier in enumerate(root_ids)
        )
    ):
        raise ValueError("Exact and execution root-coordinate banks disagree.")
    if (
        not isinstance(target_source_entity_set_id, str)
        or not target_source_entity_set_id.strip()
    ):
        raise ValueError("target_source_entity_set_id must be non-empty.")
    root_float_by_id = {
        identifier: root_float[row] for row, identifier in enumerate(root_ids)
    }
    vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
    current_roots = np.asarray(view.vertex_source_ids, dtype=np.int64)
    target_roots = np.asarray(target_vertex_source_ids, dtype=np.int64)
    target_windings = np.asarray(target_vertex_absolute_windings, dtype=np.int64)
    if (
        not np.array_equal(target_roots, current_roots)
        or target_roots.shape != vertex_ids.shape
        or target_windings.shape != (vertex_ids.size, periodic_rank)
        or any(int(identifier) not in root_points for identifier in target_roots)
    ):
        raise ValueError(
            "Owner-local motion changed source-root identity or omitted a lifted root."
        )
    lattice_exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in np.asarray(periodic_cell.vectors, dtype=np.float64)
    )
    target_coordinate_stencil: VertexInterpolationStencil | None = None
    target_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None = None
    if source_map is None:
        if (
            view.source_map_id is not None
            or view.vertex_ids.size
            and isinstance(view.source_geometry_authority, ExactPowerCellGeometrySource)
        ):
            raise ValueError(
                "Periodic power motion requires a freshly rebuilt exact source map."
            )
        if vertex_ids.size:
            source_stencil = view.source_coordinate_stencil
            source_offsets = view.exact_vertex_coordinate_offsets
            if source_stencil is None or source_offsets is None:
                raise ValueError(
                    "Nonempty periodic Delaunay motion lacks cumulative exact ancestry."
                )
            old_windings = np.asarray(view.vertex_absolute_windings, dtype=np.int64)
            delta_windings = target_windings - old_windings
            if old_windings.shape != target_windings.shape:
                raise ValueError(
                    "Periodic motion winding rows differ from the source view."
                )
            target_coordinate_offsets = tuple(
                tuple(
                    source_offsets[row][axis]
                    + sum(
                        (
                            int(delta_windings[row, generator])
                            * lattice_exact[generator][axis]
                            for generator in range(periodic_rank)
                        ),
                        Fraction(0),
                    )
                    for axis in range(dimension)
                )
                for row in range(vertex_ids.size)
            )
            target_coordinate_stencil = VertexInterpolationStencil(
                target_source_entity_set_id,
                source_stencil.target_entity_set_id,
                source_stencil.target_global_ids,
                source_stencil.source_global_ids,
                source_stencil.weights,
                source_stencil.valid,
                preserves_constants=source_stencil.preserves_constants,
                exact_weights=source_stencil.exact_weights,
            )
            exact_vertex_rows = []
            vertex_coordinates = np.empty((vertex_ids.size, dimension), dtype=np.float64)
            stencil_source_ids = np.asarray(
                target_coordinate_stencil.source_global_ids, dtype=np.int64
            )
            stencil_valid = np.asarray(target_coordinate_stencil.valid, dtype=np.bool_)
            for row, weights in enumerate(target_coordinate_stencil.exact_weights):
                coordinate = list(target_coordinate_offsets[row])
                active_ids = []
                for identifier, weight, valid in zip(
                    stencil_source_ids[row], weights, stencil_valid[row], strict=True
                ):
                    if not valid:
                        continue
                    source_point = root_points.get(int(identifier))
                    if source_point is None:
                        raise ValueError(
                            "Cumulative motion stencil references an unknown source site."
                        )
                    active_ids.append(int(identifier))
                    for axis, value in enumerate(source_point):
                        coordinate[axis] += weight * value
                exact_point = tuple(coordinate)
                if (
                    sum(
                        (
                            weight
                            for weight, valid in zip(
                                weights, stencil_valid[row], strict=True
                            )
                            if valid
                        ),
                        Fraction(0),
                    )
                    != 1
                ):
                    raise ValueError(
                        "Cumulative motion stencil does not preserve constants."
                    )
                exact_vertex_rows.append(exact_point)
                if (
                    len(active_ids) == 1
                    and weights[int(np.flatnonzero(stencil_valid[row])[0])] == 1
                    and all(value == 0 for value in target_coordinate_offsets[row])
                ):
                    vertex_coordinates[row] = root_float_by_id[active_ids[0]]
                else:
                    vertex_coordinates[row] = tuple(float(value) for value in exact_point)
            exact_vertex_coordinates = tuple(exact_vertex_rows)
        else:
            exact_vertex_coordinates = ()
            vertex_coordinates = np.zeros((0, dimension), dtype=np.float64)
    else:
        target_coordinate_stencil = None
        target_coordinate_offsets = None
        if type(source_map) is not ExactPowerCellGeometrySource:
            raise TypeError(
                "Periodic power motion requires ExactPowerCellGeometrySource."
            )
        if (
            view.source_map_id is None
            or source_map.source_id != view.source_id
            or source_map.periodic_source_context is None
        ):
            raise ValueError("Fresh exact power source map is not bound to this source.")
        context = source_map.periodic_source_context
        context.require_periodic_cell(periodic_cell)
        context_source = context.periodic_source
        context_ids = np.asarray(context_source.point_ids, dtype=np.int64)
        context_points = np.asarray(context_source.points, dtype=np.float64)
        if context_points.shape != (context_ids.size, dimension) or any(
            int(identifier) not in root_points
            or tuple(Fraction.from_float(float(value)) for value in context_points[row])
            != root_points[int(identifier)]
            for row, identifier in enumerate(context_ids)
        ):
            raise ValueError("Fresh exact power context differs from moved source roots.")
        exact_vertex_coordinates = tuple(source_map.source_coordinates())
        if len(exact_vertex_coordinates) != vertex_ids.size:
            raise ValueError("Fresh exact power map differs from the target vertex bank.")
        vertex_coordinates = np.asarray(source_map.rounded_vertices, dtype=np.float64)
        if vertex_coordinates.shape != (vertex_ids.size, dimension):
            raise ValueError("Fresh exact power source-map shadows have the wrong shape.")
    if any(
        len(point) != dimension or any(not isinstance(value, Fraction) for value in point)
        for point in exact_vertex_coordinates
    ):
        raise TypeError("Materialized periodic coordinates must remain exact rationals.")
    for root in np.unique(target_roots):
        rows = np.flatnonzero(target_roots == root)
        if rows.size < 2:
            continue
        reference = min(
            (int(row) for row in rows),
            key=lambda row: (int(vertex_ids[row]), row),
        )
        for row_value in rows:
            row = int(row_value)
            expected = tuple(
                exact_vertex_coordinates[reference][axis]
                + sum(
                    (
                        int(
                            target_windings[row, generator]
                            - target_windings[reference, generator]
                        )
                        * lattice_exact[generator][axis]
                        for generator in range(periodic_rank)
                    ),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            if exact_vertex_coordinates[row] != expected:
                raise ValueError(
                    "Materialized periodic orbit rows do not differ by their exact deck action."
                )
    if source_map is not None and not np.array_equal(
        np.asarray(source_map.rounded_vertices, dtype=np.float64),
        vertex_coordinates,
    ):
        raise ValueError(
            "Fresh exact power source map differs from its rounded target vertices."
        )
    if np.any(~np.isfinite(vertex_coordinates)):
        raise ValueError("Materialized periodic coordinate shadows must be finite.")
    cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
    exact_measures: list[Fraction | None] = [None] * cell_ids.size
    cell_vertices_by_row: dict[int, np.ndarray] = {}
    target_blocks: list[OwnerLocalCellBlock] = []
    for block in view.blocks:
        rows = np.asarray(block.cell_rows, dtype=np.int64)
        block_ids = np.asarray(block.cell_ids, dtype=np.int64)
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        if (
            rows.shape != block_ids.shape
            or vertices.shape[0] != rows.size
            or valid.shape != vertices.shape
            or np.any(rows < 0)
            or np.any(rows >= cell_ids.size)
            or not np.array_equal(cell_ids[rows], block_ids)
        ):
            raise ValueError("Periodic motion block rows differ from current cell IDs.")
        updated_measures = np.empty((rows.size,), dtype=np.float64)
        updated_shifts = np.zeros((*vertices.shape, periodic_rank), dtype=np.int64)
        for local_row, global_row in enumerate(rows):
            vertex_rows = vertices[local_row][valid[local_row]]
            if (
                not vertex_rows.size
                or np.any(vertex_rows < 0)
                or np.any(vertex_rows >= vertex_ids.size)
                or np.unique(vertex_rows).size != vertex_rows.size
                or exact_measures[int(global_row)] is not None
            ):
                raise ValueError("Periodic motion cell incidence is malformed.")
            cell_vertices_by_row[int(global_row)] = vertex_rows.copy()
            if block.cell_kind == "triangle" and dimension == 2:
                first, second, third = (
                    exact_vertex_coordinates[int(vertex)] for vertex in vertex_rows
                )
                left = tuple(second[axis] - first[axis] for axis in range(2))
                right = tuple(third[axis] - first[axis] for axis in range(2))
                measure = (left[0] * right[1] - left[1] * right[0]) / 2
            elif block.cell_kind == "tetrahedron" and dimension == 3:
                first, second, third, fourth = (
                    exact_vertex_coordinates[int(vertex)] for vertex in vertex_rows
                )
                left = tuple(second[axis] - first[axis] for axis in range(3))
                middle = tuple(third[axis] - first[axis] for axis in range(3))
                right = tuple(fourth[axis] - first[axis] for axis in range(3))
                measure = (
                    left[0] * (middle[1] * right[2] - middle[2] * right[1])
                    - left[1] * (middle[0] * right[2] - middle[2] * right[0])
                    + left[2] * (middle[0] * right[1] - middle[1] * right[0])
                ) / 6
            elif block.cell_kind == "polyhedron" and dimension == 3:
                if block.polyhedron_faces is None or block.polyhedron_face_signs is None:
                    raise ValueError(
                        "Periodic power motion lacks oriented polyhedron faces."
                    )
                volume_six = Fraction(0)
                faces = block.polyhedron_faces[local_row]
                signs = np.asarray(block.polyhedron_face_signs[local_row], dtype=np.int8)
                if len(faces) != signs.size:
                    raise ValueError("Periodic power face signs differ from face loops.")
                for face, sign in zip(faces, signs, strict=True):
                    loop = np.asarray(face, dtype=np.int64)
                    if sign < 0:
                        loop = loop[::-1]
                    if loop.size < 3:
                        raise ValueError("Periodic power motion has a degenerate face.")
                    origin = exact_vertex_coordinates[int(loop[0])]
                    for index_ in range(1, loop.size - 1):
                        left = exact_vertex_coordinates[int(loop[index_])]
                        right = exact_vertex_coordinates[int(loop[index_ + 1])]
                        cross = (
                            left[1] * right[2] - left[2] * right[1],
                            left[2] * right[0] - left[0] * right[2],
                            left[0] * right[1] - left[1] * right[0],
                        )
                        volume_six += sum(
                            (origin[axis] * cross[axis] for axis in range(3)),
                            Fraction(0),
                        )
                measure = volume_six / 6
            else:
                raise ValueError(
                    f"Periodic motion does not support exact measure for {block.cell_kind!r}."
                )
            if measure <= 0:
                raise ValueError("Periodic motion inverted or collapsed an exact cell.")
            exact_measures[int(global_row)] = measure
            updated_measures[local_row] = float(measure)
            updated_shifts[local_row][valid[local_row]] = target_windings[
                vertices[local_row][valid[local_row]]
            ]
        if block.cell_vertex_shifts is None:
            raise ValueError("Periodic owner-local block lost its lifted vertex rows.")
        updated_block = eqx.tree_at(
            lambda value: value.cell_measures,
            block,
            jnp.asarray(updated_measures),
        )
        updated_block = eqx.tree_at(
            lambda value: value.cell_vertex_shifts,
            updated_block,
            jnp.asarray(updated_shifts),
        )
        target_blocks.append(updated_block)
    if cell_ids.size and (
        len(cell_vertices_by_row) != cell_ids.size
        or any(measure is None for measure in exact_measures)
    ):
        raise ValueError("Periodic motion incidence does not cover every local cell.")
    exact_cell_measures = tuple(
        measure for measure in exact_measures if measure is not None
    )
    key_offsets = np.zeros((cell_ids.size + 1,), dtype=np.int64)
    key_sources: list[int] = []
    key_windings: list[tuple[int, ...]] = []
    cell_windings = np.zeros((cell_ids.size, periodic_rank), dtype=np.int64)
    orders = identification_orders(periodic_cell)
    for row in range(cell_ids.size):
        vertices = cell_vertices_by_row[row]
        anchor = min(
            (int(vertex) for vertex in vertices),
            key=lambda vertex: (
                int(target_roots[vertex]),
                tuple(int(value) for value in target_windings[vertex]),
            ),
        )
        cell_windings[row] = target_windings[anchor]
        relative = reduce_periodic_exponents(
            target_windings[vertices] - target_windings[anchor][None, :],
            orders,
        )
        keys = tuple(
            sorted(
                (
                    int(target_roots[int(vertex)]),
                    tuple(int(value) for value in winding),
                )
                for vertex, winding in zip(vertices, relative, strict=True)
            )
        )
        if len(set(keys)) != len(keys):
            raise ValueError("Periodic motion collapsed distinct quotient vertices.")
        key_sources.extend(source for source, _ in keys)
        key_windings.extend(winding for _, winding in keys)
        key_offsets[row + 1] = len(key_sources)
    key_source_array = np.asarray(key_sources, dtype=np.int64)
    key_winding_array = np.asarray(key_windings, dtype=np.int64).reshape(
        (-1, periodic_rank)
    )
    if (
        not np.array_equal(key_offsets, np.asarray(view.cell_key_offsets, dtype=np.int64))
        or not np.array_equal(
            key_source_array, np.asarray(view.cell_key_source_ids, dtype=np.int64)
        )
        or not np.array_equal(
            key_winding_array,
            np.asarray(view.cell_key_relative_windings, dtype=np.int64),
        )
    ):
        raise ValueError(
            "Periodic motion changed quotient cell identity without an orbit transition."
        )
    local_mesh, geometry = _reconstruct_owner_local_geometry(
        jnp.asarray(vertex_coordinates),
        view.vertex_ids,
        jnp.asarray(vertex_coordinates),
        tuple(target_blocks),
        source_map,
        jnp.asarray(target_roots),
        jnp.asarray(target_windings),
    )
    return _OwnerLocalPeriodicTargetMaterialization(
        local_mesh,
        geometry,
        source_map,
        target_coordinate_stencil,
        vertex_coordinates,
        exact_vertex_coordinates,
        target_coordinate_offsets,
        target_roots,
        target_windings,
        tuple(target_blocks),
        key_offsets,
        key_source_array,
        key_winding_array,
        cell_windings,
        exact_cell_measures,
    )


@final
class MappedFieldSupport(StrictModule, NonTrainableState):
    """Exact query support bound to mapped-domain coverage and one locator.

    An optional original source query may refine wall membership only when it
    has the same physical source identity and revision as the mapped domain.
    """

    mesh: CellMesh
    geometry: CellGeometrySpec
    domain: MappedReferenceDomain
    coverage: DomainCoverageCertificate
    locator: PreparedMappedCellLocator
    source_query: Any
    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    coverage_id: str = eqx.field(static=True)
    cell_map_id: str = eqx.field(static=True)
    locator_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    source_query_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        locator: PreparedMappedCellLocator,
        coverage: DomainCoverageCertificate,
        /,
        *,
        source_query: Any = None,
    ) -> None:
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Mapped field support requires its exact CellMesh and CellGeometrySpec."
            )
        if not isinstance(locator, PreparedMappedCellLocator):
            raise TypeError("Mapped field support requires PreparedMappedCellLocator.")
        from ..geometry._domain_certification import DomainCoverageCertificate
        from ..geometry._mapped_reference_domain import MappedReferenceDomain

        if not isinstance(coverage, DomainCoverageCertificate):
            raise TypeError("Mapped field support requires DomainCoverageCertificate.")
        if not isinstance(coverage.domain, MappedReferenceDomain):
            raise TypeError("Mapped field coverage must retain MappedReferenceDomain.")
        domain = coverage.domain
        if (
            coverage.status != "certified"
            or coverage.coverage_method != "mapped_reference_charts"
        ):
            raise ValueError(
                "Mapped field support requires certified mapped-reference coverage."
            )
        query_id = None
        if source_query is not None:
            from ..geometry import GeometryCapability, GeometryKind
            from ..geometry._contracts import (
                compiled_geometry_identity,
                CompiledGeometry,
            )

            if (
                not isinstance(source_query, CompiledGeometry)
                or source_query.kind is not GeometryKind.REGION
                or source_query.ambient_dimension != mesh.ambient_dimension
            ):
                raise TypeError(
                    "source_query must be a same-dimensional region CompiledGeometry."
                )
            source_query.require_valid()
            source_query.require(GeometryCapability.REGION_QUERY)
            source_kernel = source_query.kernel
            source_model = getattr(source_kernel, "model", None)
            source_id = getattr(
                source_model, "source_id", getattr(source_kernel, "source_id", None)
            )
            source_revision = getattr(
                source_model,
                "source_revision",
                getattr(source_kernel, "source_revision", None),
            )
            if source_id is None or source_revision is None:
                raise ValueError(
                    "Original source query must retain source_id and source_revision for composition."
                )
            if (source_id, source_revision) != (domain.source_id, domain.source_revision):
                raise ValueError(
                    "Original source query identity or revision differs from MappedReferenceDomain."
                )
            query_id = compiled_geometry_identity(source_query)
        elements, routes, coordinates = geometry.resolve(mesh)
        cell_map = locator.cell_map
        if (
            cell_map.topology_id != mesh.topology_id
            or cell_map.geometry_layout_id != geometry.geometry_layout_id
            or cell_map.cell_count != sum(block.cell_count for block in mesh.blocks)
            or not np.array_equal(
                np.asarray(locator.coordinates), np.asarray(coordinates)
            )
        ):
            raise ValueError(
                "Mapped support locator does not bind the exact full mesh geometry."
            )
        if cell_map.block_index is None:
            if cell_map.block_name != "whole-mesh" or any(
                element.element_id != cell_map.coordinate_element.element_id
                for element in elements
            ):
                raise ValueError(
                    "Whole-mesh mapped support needs one exact coordinate element identity."
                )
            expected_routes = np.concatenate(
                tuple(np.asarray(route) for route in routes), axis=0
            )
        else:
            block_index = cell_map.block_index
            if (
                block_index < 0
                or block_index >= len(mesh.blocks)
                or len(mesh.blocks) != 1
                or elements[block_index].element_id
                != cell_map.coordinate_element.element_id
            ):
                raise ValueError(
                    "Mapped support locator must cover the complete certified mesh."
                )
            expected_routes = np.asarray(routes[block_index])
        if not np.array_equal(np.asarray(cell_map.coordinate_dofs), expected_routes):
            raise ValueError(
                "Mapped support locator routes differ from the retained CellGeometrySpec."
            )
        coverage.require_bound(
            mesh,
            geometry,
            domain,
            coverage.cell_regions,
            embedding=coverage.embedding,
        )
        self.mesh, self.geometry, self.domain, self.coverage = (
            mesh,
            geometry,
            domain,
            coverage,
        )
        self.source_query = source_query
        self.locator = locator
        self.mesh_id = mesh.mesh_id
        self.geometry_id = cell_geometry_id(geometry)
        self.domain_id = domain.domain_id
        self.coverage_id = coverage.certificate_id
        self.cell_map_id = cell_map.cell_map_id
        self.source_query_id = query_id
        self.locator_id = locator.locator_id
        self.support_id = canonical_fingerprint(
            {
                "kind": "mapped-field-query-support",
                "mesh": self.mesh_id,
                "geometry": self.geometry_id,
                "domain": self.domain_id,
                "coverage": self.coverage_id,
                "cell_map": self.cell_map_id,
                "locator": self.locator_id,
                "source": (domain.source_id, domain.source_revision),
                "source_query": query_id,
            }
        )

    @property
    def kind(self) -> Any:
        from ..geometry import GeometryKind

        return GeometryKind.REGION

    @property
    def ambient_dimension(self) -> int:
        return self.mesh.ambient_dimension

    @property
    def bounds(self) -> Array:
        lower = jnp.min(self.locator.cell_lower, axis=0)
        upper = jnp.max(self.locator.cell_upper, axis=0)
        return jnp.stack((lower, upper))

    def require_locator(self, locator: object, /) -> None:
        if (
            not isinstance(locator, PreparedMappedCellLocator)
            or locator.locator_id != self.locator_id
            or locator.cell_map.cell_map_id != self.cell_map_id
            or locator.cell_map.topology_id != self.mesh.topology_id
            or locator.cell_map.geometry_layout_id != self.geometry.geometry_layout_id
        ):
            raise ValueError(
                "Mapped field query locator no longer matches its certified support."
            )
        self.coverage.require_bound(
            self.mesh,
            self.geometry,
            self.domain,
            self.coverage.cell_regions,
            embedding=self.coverage.embedding,
        )
        if self.source_query is not None:
            self.source_query.require_valid()
            from ..geometry._contracts import compiled_geometry_identity

            if compiled_geometry_identity(self.source_query) != self.source_query_id:
                raise ValueError(
                    "Original source query no longer matches its mapped support binding."
                )

    def require_kernel(self, kernel: object, /) -> None:
        locator = getattr(kernel, "locator", None)
        self.require_locator(locator)


@final
class MappedMeshSupportQueryResult(StrictModule, NonTrainableState):
    """Pointwise membership classifications backed by a certified mapped cell mesh."""

    location: CellLocationResult
    inside: Array
    boundary: Array
    outside: Array
    successful: Array
    unresolved: Array
    ambiguous: Array
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        location: CellLocationResult,
        inside: ConvertibleToArray,
        boundary: ConvertibleToArray,
        outside: ConvertibleToArray,
        successful: ConvertibleToArray,
        unresolved: ConvertibleToArray,
        ambiguous: ConvertibleToArray,
        support_id: str,
        /,
    ) -> None:
        if not isinstance(location, CellLocationResult):
            raise TypeError("location must be CellLocationResult.")
        count = location.status.shape[0]
        masks = tuple(
            np.asarray(value, dtype=np.bool_)
            for value in (inside, boundary, outside, successful, unresolved, ambiguous)
        )
        if any(value.shape != (count,) for value in masks):
            raise ValueError(
                "Mapped mesh membership masks must provide one value per query point."
            )
        if np.any(masks[0] & masks[2]) or np.any(masks[1] & ~masks[0]):
            raise ValueError("Mapped mesh membership classifications are inconsistent.")
        if not np.array_equal(masks[3], masks[0] | masks[2]):
            raise ValueError(
                "Mapped mesh successful membership must be inside or outside."
            )
        status = np.asarray(location.status, dtype=np.int32)
        complete = np.asarray(location.candidates_complete, dtype=np.bool_)
        expected_ambiguous = status == int(CellLocationStatus.AMBIGUOUS)
        expected_inside = (status == int(CellLocationStatus.LOCATED)) & complete
        expected_outside = (status == int(CellLocationStatus.OUTSIDE)) & complete
        if (
            status.shape != (count,)
            or complete.shape != (count,)
            or np.any(expected_ambiguous & ~complete)
            or np.any(masks[0] & ~expected_inside)
            or np.any(masks[2] & ~expected_outside)
            or not np.array_equal(masks[5], expected_ambiguous)
        ):
            raise ValueError(
                "Mapped membership must preserve canonical locator status and completeness."
            )
        if (
            np.any(masks[3] & masks[4])
            or np.any(masks[3] & masks[5])
            or np.any(masks[4] & masks[5])
            or not np.array_equal(masks[4], ~(masks[3] | masks[5]))
        ):
            raise ValueError(
                "Resolved, unresolved, and ambiguous statuses must partition queries."
            )
        self.location = location
        self.inside, self.boundary, self.outside = tuple(
            jnp.asarray(value) for value in masks[:3]
        )
        self.successful, self.unresolved, self.ambiguous = tuple(
            jnp.asarray(value) for value in masks[3:]
        )
        self.support_id = str(support_id)


def _point_on_reference_face(
    point: np.ndarray,
    face: np.ndarray,
    tolerance: float,
    /,
) -> bool:
    dimension = point.size
    if dimension == 1:
        if face.shape != (1, 1):
            raise ValueError("Mapped interval boundary queries require point faces.")
        return abs(float(point[0] - face[0, 0])) <= tolerance
    if dimension == 2:
        delta = face[1] - face[0]
        length_squared = float(delta @ delta)
        if length_squared <= np.finfo(np.float64).tiny:
            raise ValueError("Reference-cell boundary edge is degenerate.")
        parameter = float((point - face[0]) @ delta) / length_squared
        closest = face[0] + parameter * delta
        return (
            -tolerance <= parameter <= 1.0 + tolerance
            and float(np.linalg.norm(point - closest)) <= tolerance
        )
    if dimension != 3 or face.shape[0] < 3:
        raise ValueError("Mapped source boundary query requires 1-D/2-D/3-D cell faces.")
    origin = face[0]
    normal = np.cross(face[1] - origin, face[2] - origin)
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("Reference-cell boundary face is degenerate.")
    if abs(float((point - origin) @ normal)) / norm > tolerance:
        return False
    drop = int(np.argmax(np.abs(normal)))
    keep = tuple(axis for axis in range(3) if axis != drop)
    projected = face[:, keep]
    query = point[list(keep)]
    area = sum(
        projected[index, 0] * projected[(index + 1) % face.shape[0], 1]
        - projected[(index + 1) % face.shape[0], 0] * projected[index, 1]
        for index in range(face.shape[0])
    )
    if area == 0.0:
        raise ValueError("Reference-cell boundary face projection is degenerate.")
    orientation = 1.0 if area > 0.0 else -1.0
    for index in range(face.shape[0]):
        first = projected[index]
        second = projected[(index + 1) % face.shape[0]]
        edge = second - first
        side = edge[0] * (query[1] - first[1]) - edge[1] * (query[0] - first[0])
        if orientation * side < -tolerance * max(1.0, float(np.linalg.norm(edge))):
            return False
    return True


def _boundary_facet_flags(
    mesh: CellMesh, facet_dimension: int, /
) -> dict[tuple[int, ...], bool]:
    """Return canonical exterior flags keyed by complete facet vertex identity."""
    connectivity = mesh.connectivity
    if isinstance(connectivity, IntervalConnectivity):
        if facet_dimension != 0:
            raise ValueError("Interval support boundaries are zero-dimensional.")
        row_arity = 1
        rows = np.arange(connectivity.vertex_count, dtype=np.int64)[:, None]
        exterior = np.asarray(connectivity.boundary_vertices, dtype=np.bool_)
    elif isinstance(connectivity, SimplicialConnectivity):
        if facet_dimension != connectivity.dimension - 1:
            raise ValueError(
                "Simplicial support facet dimension does not match its topology."
            )
        row_arity = facet_dimension + 1
        rows = np.asarray(connectivity.entities[facet_dimension], dtype=np.int64)
        exterior = np.asarray(
            connectivity.boundary_masks[facet_dimension], dtype=np.bool_
        )
    elif isinstance(connectivity, PolygonalConnectivity):
        if facet_dimension != 1:
            raise ValueError("Polygonal support boundaries require edge facets.")
        row_arity = 2
        rows = np.asarray(connectivity.edges, dtype=np.int64)
        exterior = np.asarray(connectivity.boundary_edges, dtype=np.bool_)
    elif isinstance(connectivity, TetrahedralConnectivity):
        if facet_dimension != 2:
            raise ValueError("Tetrahedral support boundaries require triangular facets.")
        row_arity = 3
        rows = np.asarray(connectivity.faces, dtype=np.int64)
        exterior = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    elif isinstance(connectivity, HexahedralConnectivity):
        if facet_dimension != 2:
            raise ValueError(
                "Hexahedral support boundaries require quadrilateral facets."
            )
        row_arity = 4
        rows = np.asarray(connectivity.faces, dtype=np.int64)
        exterior = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    elif isinstance(connectivity, PolyhedralConnectivity):
        if facet_dimension != 2:
            raise ValueError("Polyhedral support boundaries require planar-face facets.")
        offsets = np.reshape(
            np.asarray(connectivity.face_vertex_offsets, dtype=np.int64), (-1,)
        )
        values = np.reshape(
            np.asarray(connectivity.face_vertex_values, dtype=np.int64), (-1,)
        )
        exterior = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
        if (
            offsets.shape != (exterior.size + 1,)
            or offsets[0] != 0
            or offsets[-1] != values.size
        ):
            raise ValueError("Polyhedral boundary face routes are incomplete.")
        return {
            tuple(
                sorted(
                    int(values[position])
                    for position in range(int(offsets[index]), int(offsets[index + 1]))
                )
            ): bool(exterior[index])
            for index in range(exterior.size)
        }
    else:
        raise TypeError("Mapped support requires canonical cell-complex facet incidence.")
    if (
        rows.ndim != 2
        or rows.shape != (exterior.size, row_arity)
        or exterior.shape != (exterior.size,)
    ):
        raise ValueError(
            "Mapped support connectivity lacks complete exterior-facet records."
        )
    rows = np.reshape(rows, (exterior.size, row_arity))
    return {
        tuple(sorted(int(vertex) for vertex in row)): bool(is_exterior)
        for row, is_exterior in zip(rows, exterior, strict=True)
    }


def mapped_mesh_support_query(
    support: MappedFieldSupport,
    points: ConvertibleToArray,
    /,
    *,
    source_query: Any = None,
) -> MappedMeshSupportQueryResult:
    """Classify points through mapped coverage and the canonical complete locator.

    A complete OUTSIDE status is required before a point is outside. Inverse-map
    exhaustion and incomplete BVH candidates stay unresolved or ambiguous.
    An optional original source query is usable only when retained at preparation
    and a resolved disagreement remains unresolved.
    """
    if not isinstance(support, MappedFieldSupport):
        raise TypeError("support must be MappedFieldSupport.")
    if source_query is None:
        source_query = support.source_query
    elif support.source_query_id is None:
        raise ValueError(
            "Original source query was not bound to this mapped support certificate."
        )
    else:
        from ..geometry._contracts import compiled_geometry_identity

        if compiled_geometry_identity(source_query) != support.source_query_id:
            raise ValueError(
                "Original source query was not bound to this mapped support certificate."
            )
    support.require_locator(support.locator)
    sites = np.asarray(points, dtype=np.float64)
    dimension = support.mesh.ambient_dimension
    if sites.ndim != 2 or sites.shape[1] != dimension:
        raise ValueError(
            "Mapped support query points must match the certified physical dimension."
        )
    location = support.locator.locate(sites)
    status = np.asarray(location.status, dtype=np.int32)
    complete = np.asarray(location.candidates_complete, dtype=np.bool_)
    inside = (status == int(CellLocationStatus.LOCATED)) & complete
    outside = (status == int(CellLocationStatus.OUTSIDE)) & complete
    successful = inside | outside
    ambiguous = status == int(CellLocationStatus.AMBIGUOUS)
    unresolved = ~(successful | ambiguous)
    boundary = np.zeros((sites.shape[0],), dtype=np.bool_)
    if np.any(inside):
        topological_dimension = support.locator.cell_map.reference_dimension
        if support.mesh.topological_dimension != topological_dimension:
            raise ValueError(
                "Mapped support cell map and mesh topological dimensions differ."
            )
        exterior_facets = _boundary_facet_flags(support.mesh, topological_dimension - 1)
        offsets = np.cumsum(
            (0, *(block.cell_count for block in support.mesh.blocks)), dtype=np.int64
        )
        candidates = np.asarray(location.candidate_cells, dtype=np.int64)
        references = np.asarray(location.candidate_reference, dtype=np.float64)
        tolerance = support.locator.policy.reference_tolerance
        for point_index in np.flatnonzero(inside):
            for candidate_index in range(candidates.shape[1]):
                cell = int(candidates[point_index, candidate_index])
                if cell < 0:
                    continue
                block_index = int(np.searchsorted(offsets[1:], cell, side="right"))
                if block_index >= len(support.mesh.blocks):
                    raise ValueError(
                        "Mapped locator returned a cell outside its source mesh."
                    )
                block = support.mesh.blocks[block_index]
                if not isinstance(block, CellBlock):
                    raise TypeError(
                        "Mapped support membership requires reference-cell blocks."
                    )
                local_cell = cell - int(offsets[block_index])
                topology = reference_cell_topology(block.cell_kind)
                reference = references[point_index, candidate_index]
                cell_vertices = np.asarray(block.vertices[local_cell], dtype=np.int64)
                for local_face in topology.entities[topological_dimension - 1]:
                    face_vertices = cell_vertices[np.asarray(local_face, dtype=np.int64)]
                    face_key = tuple(sorted(int(value) for value in face_vertices))
                    is_exterior = exterior_facets.get(face_key)
                    if is_exterior is None:
                        raise ValueError(
                            "Mapped support cell face is absent from canonical connectivity."
                        )
                    if is_exterior and _point_on_reference_face(
                        reference,
                        np.asarray(topology.vertices, dtype=np.float64)[
                            np.asarray(local_face, dtype=np.int64)
                        ],
                        tolerance,
                    ):
                        boundary[point_index] = True
                        break
                if boundary[point_index]:
                    break
    if source_query is not None:
        from ..geometry import GeometryCapability
        from ..geometry._contracts import CompiledGeometry

        if not isinstance(source_query, CompiledGeometry):
            raise TypeError(
                "source_query must be a resolved CompiledGeometry region query or None."
            )
        if source_query.ambient_dimension != dimension:
            raise ValueError(
                "Original source query and mapped support dimensions differ."
            )
        source_query.require_valid()
        source_query.require(GeometryCapability.REGION_QUERY)
        finite = np.all(np.isfinite(sites), axis=1)
        source_inside = np.zeros((sites.shape[0],), dtype=np.bool_)
        source_boundary = np.zeros((sites.shape[0],), dtype=np.bool_)
        valid_rows = np.flatnonzero(finite)
        if valid_rows.size:
            selected_points = jnp.asarray(sites[valid_rows])
            membership = np.asarray(source_query.contains(selected_points))
            margin = np.asarray(
                source_query.boundary_field(selected_points), dtype=np.float64
            )
            bounds = np.asarray(source_query.bounds, dtype=np.float64)
            if (
                membership.shape != (valid_rows.size,)
                or membership.dtype != np.dtype(np.bool_)
                or margin.shape != (valid_rows.size,)
                or np.any(~np.isfinite(margin))
                or bounds.shape != (2, dimension)
                or np.any(~np.isfinite(bounds))
            ):
                raise RuntimeError(
                    "Original source query returned incomplete membership data."
                )
            source_inside[valid_rows] = membership
            scale = float(np.linalg.norm(bounds[1] - bounds[0]))
            threshold = float(
                np.asarray(source_query.tolerance.threshold(jnp.asarray(scale)))
            )
            source_boundary[valid_rows] = np.abs(margin) <= threshold
        disagreement = (successful & (inside != source_inside) & ~source_boundary) | (
            successful & outside & source_boundary
        )
        unresolved |= disagreement
        inside = inside & (source_inside | source_boundary)
        outside = outside & ~source_inside & ~source_boundary
        boundary = (boundary & inside) | (inside & source_boundary)
        successful = inside | outside
    return MappedMeshSupportQueryResult(
        location,
        inside,
        boundary,
        outside,
        successful,
        unresolved,
        ambiguous,
        support.support_id,
    )


def prepare_mapped_field_support(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    locator: PreparedMappedCellLocator,
    coverage: DomainCoverageCertificate,
    /,
    *,
    source_query: Any = None,
) -> MappedFieldSupport:
    """Prepare exact mapped support for field queries and optional wall tests.

    The retained coverage certificate is revalidated whenever support is used.
    """
    return MappedFieldSupport(
        mesh, geometry, locator, coverage, source_query=source_query
    )


__all__ = [
    "MappedFieldSupport",
    "OwnerLocalCellBlock",
    "OwnerLocalCellGeometry",
    "MappedMeshSupportQueryResult",
    "cell_location_status",
    "mapped_mesh_support_query",
    "prepare_mapped_field_support",
    "simplicial_mesh_support_geometry",
    "tensor_box_support_geometry",
    "verify_mesh_support_geometry",
]
