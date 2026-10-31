#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quotient-aware periodic mesh consumers.

The stored cell mesh remains the validated lifted cover. These helpers expose
its quotient explicitly to ownership and adaptation consumers without inferring
identity from coordinates.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, pairwise, product
from typing import Any, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
)
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ..discretization._partition import CellAdjacency
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import (
    apply_periodic,
    identification_id,
    identification_orders,
    reduce_periodic_exponents,
)
from ..discretization._periodic_topology import (
    _canonical_unordered_form,
    _entity_vertex_rows,
    _interleaved_key,
    _orientation_orders,
    PeriodicMeshTopology,
)
from ..typing import checked
from ._lineage import VertexInterpolationStencil


if TYPE_CHECKING:
    from ._topology_edit import CellTopologyEdit


class PeriodicPointOrbits(StrictModule, NonTrainableState):
    """Authored source sites in one full-rank periodic lattice."""

    cell: PeriodicCell
    points: Array
    fractional_points: Array
    point_ids: Array
    ambient_dimension: int = eqx.field(static=True)
    orbits_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        points: ArrayLike,
        point_ids: ArrayLike,
        /,
    ) -> None:
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell lattice.")
        if not cell.fully_periodic or cell.rank != cell.ambient_dimension:
            raise ValueError(
                "Periodic point-orbit Delaunay requires a fully periodic full-rank lattice."
            )
        if cell.ambient_dimension not in (2, 3):
            raise ValueError(
                "Periodic point-orbit Delaunay supports ambient dimension two or three."
            )
        coordinates = np.asarray(points, dtype=np.float64)
        raw_ids = np.asarray(point_ids)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != cell.ambient_dimension
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError(
                "Periodic source points must form a finite ambient point bank."
            )
        if raw_ids.shape != (coordinates.shape[0],) or raw_ids.dtype.kind not in "iu":
            raise TypeError(
                "point_ids must be one integer semantic identity per source point."
            )
        if (
            raw_ids.dtype.kind == "u"
            and raw_ids.size
            and int(np.max(raw_ids)) > np.iinfo(np.int64).max
        ):
            raise ValueError("point_ids must fit in int64.")
        identifiers = raw_ids.astype(np.int64, copy=False)
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("point_ids must be unique nonnegative source identities.")
        fractional = np.asarray(cell.fractional(coordinates), dtype=np.float64)
        if fractional.shape != coordinates.shape or np.any(~np.isfinite(fractional)):
            raise ValueError(
                "Periodic source fractional coordinates must be finite and aligned."
            )
        self.cell = cell
        self.points = jnp.asarray(coordinates)
        self.fractional_points = jnp.asarray(fractional)
        self.point_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.ambient_dimension = cell.ambient_dimension
        order = np.argsort(identifiers, kind="stable")
        self.orbits_id = canonical_fingerprint(
            {
                "kind": "periodic-point-orbits",
                "cell": cell.cell_id,
                "point_ids": array_tree_fingerprint(identifiers[order]),
                "points": array_tree_fingerprint(coordinates[order]),
            }
        )

    @classmethod
    def from_stored_fields(cls, value: PeriodicPointOrbits, /) -> PeriodicPointOrbits:
        """Rebuild point orbits through their canonical periodic-cell owner."""
        if not isinstance(value, cls):
            raise TypeError("value must be PeriodicPointOrbits.")
        cell = PeriodicCell.from_stored_fields(value.cell)
        renewed = cls(cell, value.points, value.point_ids)
        if renewed.orbits_id != value.orbits_id:
            raise ValueError(
                "Stored periodic point-orbit fields do not reproduce their identity."
            )
        return renewed


class PeriodicMeshConstruction(StrictModule, NonTrainableState):
    """Native lifted cell complex and its validated quotient descriptor."""

    mesh: CellMesh
    geometry: CellGeometrySpec | None
    construction_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec | None = None,
    ) -> None:
        if not isinstance(mesh, CellMesh) or mesh.periodic_topology is None:
            raise TypeError(
                "A periodic construction requires a CellMesh with PeriodicMeshTopology."
            )
        mesh.periodic_topology.require_lifted(
            mesh.topology, mesh.topology_id, mesh.blocks
        )
        mesh.periodic_topology.require_coordinates(mesh.coordinates)
        if geometry is not None:
            if not isinstance(geometry, CellGeometrySpec):
                raise TypeError("geometry must be a CellGeometrySpec when supplied.")
            geometry.resolve(mesh)
        self.mesh = mesh
        self.geometry = geometry
        self.construction_id = canonical_fingerprint(
            {
                "kind": "periodic-mesh-construction",
                "mesh": mesh.mesh_id,
                "periodic_topology": mesh.periodic_topology.periodic_topology_id,
                "geometry": None if geometry is None else geometry.geometry_layout_id,
            }
        )


class OwnerLocalPeriodicCandidate(StrictModule, NonTrainableState):
    """One rank's exact local periodic candidate and collective closure receipts.

    Generated candidates key every vertex and cell key by a resident source
    site. Adapted candidates additionally carry the cumulative exact vertex
    ancestry: midpoint vertices are keyed by themselves, and their exact
    coordinates are the ancestry-weighted resident source sites plus exact
    lattice offsets.
    """

    local_mesh: CellMesh | None
    periodic_cell: PeriodicCell
    source_point_coordinates: Array
    source_point_ids: Array
    source_owner_ranks: Array
    source_owned: Array
    source_point_exact_coordinates: tuple[tuple[Fraction, ...], ...]
    cell_ids: Array
    cell_owner_ranks: Array
    cell_key_offsets: Array
    cell_key_source_ids: Array
    cell_key_relative_windings: Array
    cell_windings: Array
    vertex_ids: Array
    vertex_owner_ranks: Array
    vertex_source_ids: Array
    vertex_absolute_windings: Array
    exact_vertex_coordinates: tuple[tuple[Fraction, ...], ...]
    exact_cell_measures: tuple[Fraction, ...]
    vertex_coordinate_stencil: VertexInterpolationStencil | None
    exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None
    owned_cell_ids: Array
    owned_cell_owner_ranks: Array
    owned_cell_weights: Array
    cell_neighbor_offsets: Array
    cell_neighbor_ids: Array
    cell_neighbor_owner_ranks: Array
    cell_neighbor_weights: Array
    rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_distribution_id: str = eqx.field(static=True)
    source_part_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    coordinate_contract_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    face_coverage_id: str = eqx.field(static=True)
    candidate_revision_id: str = eqx.field(static=True)
    source_local_mesh_id: str | None = eqx.field(static=True)
    source_local_topology_id: str | None = eqx.field(static=True)
    source_local_geometry_id: str | None = eqx.field(static=True)
    periodic_cell_id: str = eqx.field(static=True)
    source_context_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        rank: int,
        part_count: int,
        dimension: int,
        local_mesh: CellMesh | None,
        periodic_cell: PeriodicCell,
        /,
        *,
        source_id: str,
        source_revision: str,
        source_manifest_id: str,
        source_mesh_id: str,
        source_distribution_id: str,
        source_part_id: str,
        source_geometry_id: str,
        coordinate_contract_id: str,
        source_topology_id: str,
        periodic_topology_id: str,
        face_coverage_id: str,
        candidate_revision_id: str,
        periodic_cell_id: str,
        source_context_id: str,
        geometry_layout_id: str,
        source_point_coordinates: ArrayLike,
        source_point_ids: ArrayLike,
        source_owner_ranks: ArrayLike,
        source_owned: ArrayLike,
        source_point_exact_coordinates: tuple[tuple[Fraction, ...], ...],
        cell_ids: ArrayLike,
        cell_owner_ranks: ArrayLike,
        cell_key_offsets: ArrayLike,
        cell_key_source_ids: ArrayLike,
        cell_key_relative_windings: ArrayLike,
        cell_windings: ArrayLike,
        vertex_ids: ArrayLike,
        vertex_owner_ranks: ArrayLike,
        vertex_source_ids: ArrayLike,
        vertex_absolute_windings: ArrayLike,
        exact_vertex_coordinates: tuple[tuple[Fraction, ...], ...],
        exact_cell_measures: tuple[Fraction, ...],
        owned_cell_ids: ArrayLike,
        owned_cell_owner_ranks: ArrayLike,
        owned_cell_weights: ArrayLike,
        cell_neighbor_offsets: ArrayLike,
        cell_neighbor_ids: ArrayLike,
        cell_neighbor_owner_ranks: ArrayLike,
        cell_neighbor_weights: ArrayLike,
        vertex_coordinate_stencil: VertexInterpolationStencil | None = None,
        exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None = None,
    ) -> None:
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(part_count, bool)
            or not isinstance(part_count, int)
            or part_count < 1
            or not 0 <= rank < part_count
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension not in (2, 3)
        ):
            raise ValueError(
                "Owner-local periodic candidate rank or dimension is invalid."
            )
        if (
            not isinstance(periodic_cell, PeriodicCell)
            or periodic_cell.cell_id != periodic_cell_id
            or periodic_cell.ambient_dimension != dimension
            or not periodic_cell.fully_periodic
        ):
            raise ValueError(
                "Owner-local candidate periodic cell identity is inconsistent."
            )
        identities = (
            source_id,
            source_revision,
            source_distribution_id,
            source_part_id,
            source_geometry_id,
            coordinate_contract_id,
            source_manifest_id,
            source_mesh_id,
            source_topology_id,
            periodic_topology_id,
            face_coverage_id,
            candidate_revision_id,
            periodic_cell_id,
            source_context_id,
            geometry_layout_id,
        )
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError(
                "Owner-local periodic candidate identities must be nonempty."
            )
        raw_points = np.asarray(source_point_coordinates)
        if raw_points.dtype.kind not in "iuf":
            raise TypeError(
                "Owner-local periodic coordinates must be real numeric values."
            )
        points = raw_points.astype(np.float64, copy=False)
        integer_banks = tuple(
            np.asarray(value)
            for value in (
                source_point_ids,
                source_owner_ranks,
                cell_ids,
                cell_owner_ranks,
                cell_key_offsets,
                cell_key_source_ids,
                vertex_ids,
                vertex_owner_ranks,
                vertex_source_ids,
                owned_cell_ids,
                owned_cell_owner_ranks,
                cell_neighbor_offsets,
                cell_neighbor_ids,
                cell_neighbor_owner_ranks,
                cell_neighbor_weights,
            )
        )
        if any(value.dtype.kind not in "iu" for value in integer_banks):
            raise TypeError(
                "Owner-local periodic identities and offsets must be integer arrays."
            )
        if any(
            value.dtype.kind == "u"
            and value.size
            and int(np.max(value)) > np.iinfo(np.int64).max
            for value in integer_banks
        ):
            raise ValueError("Owner-local periodic identities must fit int64.")
        winding_banks = tuple(
            np.asarray(value)
            for value in (
                cell_key_relative_windings,
                cell_windings,
                vertex_absolute_windings,
            )
        )
        if any(value.dtype.kind != "i" for value in winding_banks):
            raise TypeError(
                "Owner-local periodic windings must be signed integer arrays."
            )
        owned_raw = np.asarray(source_owned)
        if owned_raw.dtype.kind != "b":
            raise TypeError("source_owned must be a boolean owner mask.")
        point_ids = integer_banks[0].astype(np.int64, copy=False)
        point_owners_raw = integer_banks[1]
        cell_ids_value = integer_banks[2].astype(np.int64, copy=False)
        cell_owners_raw = integer_banks[3]
        key_offsets = integer_banks[4].astype(np.int64, copy=False)
        key_sources = integer_banks[5].astype(np.int64, copy=False)
        vertex_ids_value = integer_banks[6].astype(np.int64, copy=False)
        vertex_owners_raw = integer_banks[7]
        vertex_sources = integer_banks[8].astype(np.int64, copy=False)
        owned_cell_ids_value = integer_banks[9].astype(np.int64, copy=False)
        owned_cell_owner_raw = integer_banks[10]
        neighbor_offsets = integer_banks[11].astype(np.int64, copy=False)
        neighbor_ids = integer_banks[12].astype(np.int64, copy=False)
        neighbor_owner_raw = integer_banks[13]
        neighbor_weights = integer_banks[14].astype(np.int64, copy=False)
        raw_owned_weights = np.asarray(owned_cell_weights)
        if raw_owned_weights.dtype.kind not in "iu":
            raise TypeError("Owned periodic cell work must use integer weights.")
        if (
            raw_owned_weights.dtype.kind == "u"
            and raw_owned_weights.size
            and int(np.max(raw_owned_weights)) > np.iinfo(np.int64).max
        ):
            raise ValueError("Owned periodic cell work must fit int64.")
        owned_weights = raw_owned_weights.astype(np.int64, copy=False)
        owner_banks = (
            point_owners_raw,
            cell_owners_raw,
            vertex_owners_raw,
            owned_cell_owner_raw,
            neighbor_owner_raw,
        )
        if any(
            value.size
            and (np.any(value < 0) or int(np.max(value)) > np.iinfo(np.int32).max)
            for value in owner_banks
        ):
            raise ValueError("Owner-local periodic ranks must fit nonnegative int32.")
        point_owners = point_owners_raw.astype(np.int32, copy=False)
        point_owned = owned_raw.astype(np.bool_, copy=False)
        cell_owners = cell_owners_raw.astype(np.int32, copy=False)
        key_windings = winding_banks[0].astype(np.int64, copy=False)
        cell_winding_values = winding_banks[1].astype(np.int64, copy=False)
        vertex_owners = vertex_owners_raw.astype(np.int32, copy=False)
        vertex_windings = winding_banks[2].astype(np.int64, copy=False)
        owned_cell_owner_ranks = owned_cell_owner_raw.astype(np.int32, copy=False)
        neighbor_owner_ranks = neighbor_owner_raw.astype(np.int32, copy=False)
        if (
            points.ndim != 2
            or points.shape[1] != dimension
            or np.any(~np.isfinite(points))
            or point_ids.shape != (points.shape[0],)
            or np.unique(point_ids).size != point_ids.size
            or point_owners.shape != point_ids.shape
            or point_owned.shape != point_ids.shape
            or np.any((point_owners < 0) | (point_owners >= part_count))
            or not np.array_equal(point_owned, point_owners == rank)
            or len(source_point_exact_coordinates) != point_ids.size
            or any(
                len(row) != dimension
                or any(not isinstance(value, Fraction) for value in row)
                for row in source_point_exact_coordinates
            )
            or any(
                Fraction.from_float(float(points[row, axis]))
                != source_point_exact_coordinates[row][axis]
                for row in range(point_ids.size)
                for axis in range(dimension)
            )
        ):
            raise ValueError("Owner-local periodic source-point banks are inconsistent.")
        if (
            cell_owners.shape != cell_ids_value.shape
            or np.unique(cell_ids_value).size != cell_ids_value.size
            or np.any(np.diff(cell_ids_value) < 0)
            or np.any((cell_owners < 0) | (cell_owners >= part_count))
            or key_offsets.shape != (cell_ids_value.size + 1,)
            or key_offsets[0] != 0
            or np.any(np.diff(key_offsets) != dimension + 1)
            or key_offsets[-1] != key_sources.size
            or key_windings.shape != (key_sources.size, dimension)
            or cell_winding_values.shape != (cell_ids_value.size, dimension)
        ):
            raise ValueError("Owner-local periodic cell-key banks are inconsistent.")
        if (
            vertex_owners.shape != vertex_ids_value.shape
            or np.unique(vertex_ids_value).size != vertex_ids_value.size
            or np.any((vertex_owners < 0) | (vertex_owners >= part_count))
            or vertex_sources.shape != vertex_ids_value.shape
            or vertex_windings.shape != (vertex_ids_value.size, dimension)
            or len(exact_vertex_coordinates) != vertex_ids_value.size
            or any(
                len(row) != dimension
                or any(not isinstance(value, Fraction) for value in row)
                for row in exact_vertex_coordinates
            )
        ):
            raise ValueError("Owner-local periodic vertex banks are inconsistent.")
        if len(exact_cell_measures) != cell_ids_value.size or any(
            not isinstance(value, Fraction) or value <= 0 for value in exact_cell_measures
        ):
            raise ValueError(
                "Owner-local periodic exact cell measures are absent or nonpositive."
            )
        ancestry_payload = _candidate_vertex_ancestry(
            point_ids,
            source_point_exact_coordinates,
            vertex_ids_value,
            vertex_sources,
            vertex_windings,
            tuple(
                tuple(Fraction.from_float(float(value)) for value in row)
                for row in np.asarray(periodic_cell.vectors, dtype=np.float64)
            ),
            key_sources,
            exact_vertex_coordinates,
            vertex_coordinate_stencil,
            exact_vertex_coordinate_offsets,
        )
        expected_owned_ids = np.sort(cell_ids_value[cell_owners == rank], kind="stable")
        if (
            not np.array_equal(owned_cell_ids_value, expected_owned_ids)
            or owned_cell_owner_ranks.shape != owned_cell_ids_value.shape
            or not np.array_equal(
                owned_cell_owner_ranks,
                np.full(owned_cell_ids_value.shape, rank, dtype=np.int32),
            )
            or owned_weights.shape != owned_cell_ids_value.shape
            or np.any(owned_weights <= 0)
            or neighbor_offsets.shape != (owned_cell_ids_value.size + 1,)
            or neighbor_offsets[0] != 0
            or np.any(np.diff(neighbor_offsets) < 0)
            or neighbor_offsets[-1] != neighbor_ids.size
            or neighbor_owner_ranks.shape != neighbor_ids.shape
            or neighbor_weights.shape != neighbor_ids.shape
            or np.any(neighbor_weights <= 0)
            or np.any((neighbor_owner_ranks < 0) | (neighbor_owner_ranks >= part_count))
            or any(
                np.unique(neighbor_ids[left:right]).size != right - left
                for left, right in zip(
                    neighbor_offsets[:-1], neighbor_offsets[1:], strict=True
                )
            )
        ):
            raise ValueError("Owner-local periodic graph CSR is inconsistent.")
        owned_rows = np.searchsorted(cell_ids_value, owned_cell_ids_value)
        if not np.array_equal(cell_ids_value[owned_rows], owned_cell_ids_value):
            raise ValueError("Owned periodic graph IDs are absent from local cells.")
        if local_mesh is None:
            if cell_ids_value.size or vertex_ids_value.size:
                raise ValueError(
                    "A nonempty periodic candidate requires its local CellMesh."
                )
        elif (
            not isinstance(local_mesh, CellMesh)
            or local_mesh.topological_dimension != dimension
            or not np.array_equal(
                np.asarray(local_mesh.vertex_global_ids, dtype=np.int64),
                vertex_ids_value,
            )
            or not np.array_equal(
                np.asarray(local_mesh.entity_set(dimension).entity_ids, dtype=np.int64),
                cell_ids_value,
            )
            or not np.array_equal(
                np.asarray(local_mesh.coordinates, dtype=np.float64),
                np.asarray(
                    [[float(value) for value in row] for row in exact_vertex_coordinates],
                    dtype=np.float64,
                ),
            )
        ):
            raise ValueError(
                "Owner-local periodic candidate does not match its local CellMesh."
            )
        if local_mesh is None:
            source_local_mesh_id = None
            source_local_topology_id = None
            source_local_geometry_id = None
        else:
            from ..discretization._cell_geometry import CellGeometrySpec
            from ..discretization._cell_geometry_validity import cell_geometry_id

            source_local_mesh_id = local_mesh.mesh_id
            source_local_topology_id = local_mesh.topology_id
            source_local_geometry_id = cell_geometry_id(
                CellGeometrySpec.affine(local_mesh)
            )
        self.local_mesh = local_mesh
        self.source_point_coordinates = jnp.asarray(points)
        self.source_point_ids = jnp.asarray(point_ids)
        self.source_owner_ranks = jnp.asarray(point_owners)
        self.source_owned = jnp.asarray(point_owned)
        self.source_point_exact_coordinates = source_point_exact_coordinates
        self.cell_ids = jnp.asarray(cell_ids_value)
        self.cell_owner_ranks = jnp.asarray(cell_owners)
        self.cell_key_offsets = jnp.asarray(key_offsets)
        self.cell_key_source_ids = jnp.asarray(key_sources)
        self.cell_key_relative_windings = jnp.asarray(key_windings)
        self.cell_windings = jnp.asarray(cell_winding_values)
        self.vertex_ids = jnp.asarray(vertex_ids_value)
        self.vertex_owner_ranks = jnp.asarray(vertex_owners)
        self.vertex_source_ids = jnp.asarray(vertex_sources)
        self.vertex_absolute_windings = jnp.asarray(vertex_windings)
        self.exact_vertex_coordinates = exact_vertex_coordinates
        self.exact_cell_measures = exact_cell_measures
        self.vertex_coordinate_stencil = vertex_coordinate_stencil
        self.exact_vertex_coordinate_offsets = exact_vertex_coordinate_offsets
        self.owned_cell_ids = jnp.asarray(owned_cell_ids_value)
        self.owned_cell_owner_ranks = jnp.asarray(owned_cell_owner_ranks, dtype=jnp.int32)
        self.owned_cell_weights = jnp.asarray(owned_weights, dtype=jnp.int64)
        self.cell_neighbor_offsets = jnp.asarray(neighbor_offsets)
        self.cell_neighbor_ids = jnp.asarray(neighbor_ids)
        self.cell_neighbor_weights = jnp.asarray(neighbor_weights)
        self.cell_neighbor_owner_ranks = jnp.asarray(
            neighbor_owner_ranks, dtype=jnp.int32
        )
        self.rank = rank
        self.part_count = part_count
        self.periodic_cell = periodic_cell
        self.dimension = dimension
        self.source_id = source_id
        self.source_revision = source_revision
        self.source_manifest_id = source_manifest_id
        self.source_distribution_id = source_distribution_id
        self.source_part_id = source_part_id
        self.source_mesh_id = source_mesh_id
        self.source_local_mesh_id = source_local_mesh_id
        self.source_local_topology_id = source_local_topology_id
        self.source_local_geometry_id = source_local_geometry_id
        self.source_geometry_id = source_geometry_id
        self.coordinate_contract_id = coordinate_contract_id
        self.source_topology_id = source_topology_id
        self.periodic_topology_id = periodic_topology_id
        self.face_coverage_id = face_coverage_id
        self.candidate_revision_id = candidate_revision_id
        self.periodic_cell_id = periodic_cell_id
        self.source_context_id = source_context_id
        self.geometry_layout_id = geometry_layout_id
        from ..discretization._exact_algebraic_geometry import coefficient_payload

        self.candidate_id = canonical_fingerprint(
            {
                "kind": "owner-local-periodic-candidate",
                "rank": rank,
                "part_count": part_count,
                "source_id": source_id,
                "source_revision": source_revision,
                "dimension": dimension,
                "source_manifest": source_manifest_id,
                "source_topology": source_topology_id,
                "source_distribution": source_distribution_id,
                "source_part": source_part_id,
                "source_geometry": source_geometry_id,
                "source_mesh": source_mesh_id,
                "source_local_mesh": source_local_mesh_id,
                "source_local_topology": source_local_topology_id,
                "source_local_geometry": source_local_geometry_id,
                "coordinate_contract": coordinate_contract_id,
                "periodic_topology": periodic_topology_id,
                "face_coverage": face_coverage_id,
                "candidate_revision": candidate_revision_id,
                "periodic_cell": periodic_cell_id,
                "source_context": source_context_id,
                "geometry_layout": geometry_layout_id,
                "local_mesh": None if local_mesh is None else local_mesh.mesh_id,
                "source_points": array_tree_fingerprint(
                    (points, point_ids, point_owners, point_owned)
                ),
                "source_point_exact_coordinates": tuple(
                    tuple(coefficient_payload(value) for value in row)
                    for row in source_point_exact_coordinates
                ),
                "cell_keys": array_tree_fingerprint(
                    (key_offsets, key_sources, key_windings, cell_winding_values)
                ),
                "vertices": array_tree_fingerprint(
                    (
                        vertex_ids_value,
                        vertex_owners,
                        vertex_sources,
                        vertex_windings,
                    )
                ),
                "exact_vertex_coordinates": tuple(
                    tuple(coefficient_payload(value) for value in row)
                    for row in exact_vertex_coordinates
                ),
                "exact_cell_measures": tuple(
                    coefficient_payload(value) for value in exact_cell_measures
                ),
                "owned_cell_graph": array_tree_fingerprint(
                    (
                        owned_cell_ids_value,
                        owned_cell_owner_ranks,
                        owned_weights,
                        neighbor_offsets,
                        neighbor_ids,
                        neighbor_owner_ranks,
                        neighbor_weights,
                    )
                ),
                "vertex_ancestry": ancestry_payload,
            }
        )


def _candidate_vertex_ancestry(
    point_ids: np.ndarray,
    source_point_exact_coordinates: tuple[tuple[Fraction, ...], ...],
    vertex_ids: np.ndarray,
    vertex_sources: np.ndarray,
    vertex_windings: np.ndarray,
    lattice: tuple[tuple[Fraction, ...], ...],
    key_sources: np.ndarray,
    exact_vertex_coordinates: tuple[tuple[Fraction, ...], ...],
    stencil: VertexInterpolationStencil | None,
    offsets: tuple[tuple[Fraction, ...], ...] | None,
    /,
) -> Any:
    """Validate vertex and cell-key roots; return the ancestry identity payload."""
    from ..discretization._exact_algebraic_geometry import coefficient_payload
    from ._lineage import VertexInterpolationStencil

    point_by_id = {
        int(identifier): source_point_exact_coordinates[row]
        for row, identifier in enumerate(point_ids.tolist())
    }
    referenced = {int(identifier) for identifier in (*key_sources, *vertex_sources)}
    if stencil is None:
        if offsets is not None:
            raise ValueError("Exact vertex offsets require their ancestry stencil.")
        absent = sorted(referenced.difference(point_by_id))
        if absent:
            raise ValueError(
                "Owner-local periodic cell keys or vertices reference "
                f"{len(absent)} source points outside the resident point "
                f"bank (first {absent[:8]})."
            )
        return None
    if (
        not isinstance(stencil, VertexInterpolationStencil)
        or offsets is None
        or len(offsets) != vertex_ids.size
        or not np.array_equal(
            np.asarray(stencil.target_global_ids, dtype=np.int64), vertex_ids
        )
        or stencil.exact_weights is None
    ):
        raise ValueError(
            "Adapted periodic candidate ancestry does not bind its resident vertices."
        )
    source_ids = np.asarray(stencil.source_global_ids, dtype=np.int64)
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    for row, weights in enumerate(stencil.exact_weights):
        coordinate = list(offsets[row])
        weight_sum = Fraction(0)
        for identifier, weight, active in zip(
            source_ids[row].tolist(), weights, valid[row].tolist(), strict=True
        ):
            if not active:
                continue
            root = point_by_id.get(int(identifier))
            if root is None:
                raise ValueError(
                    "Adapted periodic vertex ancestry references a source site "
                    "outside the resident point bank."
                )
            weight_sum += weight
            for axis, value in enumerate(root):
                coordinate[axis] += weight * value
        if weight_sum != 1 or tuple(coordinate) != tuple(exact_vertex_coordinates[row]):
            raise ValueError(
                "Adapted periodic vertex ancestry does not reproduce its exact vertex."
            )
    # Adapted vertices are keyed by themselves or by the root vertex of their
    # periodic orbit, which may be resident on another owner only. Every image
    # of such an orbit determines the root exactly (its vertex minus its lattice
    # winding), so a non-site key is admissible when a resident vertex is keyed
    # by it, and all resident images of one orbit must agree on that root.
    vertex_keys = {int(identifier) for identifier in vertex_sources}
    absent = sorted(
        referenced.difference(point_by_id)
        .difference(map(int, vertex_ids))
        .difference(vertex_keys)
    )
    if absent:
        raise ValueError(
            "Owner-local periodic cell keys reference "
            f"{len(absent)} keys that are neither resident source sites nor "
            f"adapted resident vertices or their orbit roots (first {absent[:8]})."
        )
    row_by_vertex = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    orbit_roots: dict[int, tuple[Fraction, ...]] = {}
    for row, key in enumerate(vertex_sources.tolist()):
        key = int(key)
        if key in point_by_id:
            continue
        # A resident vertex used as a key must be the root of its own orbit.
        if key in row_by_vertex and int(vertex_sources[row_by_vertex[key]]) != key:
            raise ValueError(
                "Adapted periodic vertex is keyed by a vertex that is not an orbit root."
            )
        root = tuple(
            exact_vertex_coordinates[row][axis]
            - sum(
                (
                    int(vertex_windings[row, generator]) * lattice[generator][axis]
                    for generator in range(len(lattice))
                ),
                Fraction(0),
            )
            for axis in range(len(exact_vertex_coordinates[row]))
        )
        if orbit_roots.setdefault(key, root) != root:
            raise ValueError(
                "Adapted periodic vertex images of one orbit disagree on its root."
            )
    return {
        "sources": array_tree_fingerprint((source_ids, valid)),
        "weights": tuple(
            tuple(coefficient_payload(value) for value in row)
            for row in stencil.exact_weights
        ),
        "offsets": tuple(
            tuple(coefficient_payload(value) for value in row) for row in offsets
        ),
    }


class _PeriodicEntityLayout(NamedTuple):
    orbit_rows: np.ndarray
    representative_rows: np.ndarray
    entity_ids: np.ndarray


class _PeriodicCellLayout(NamedTuple):
    orbit_rows: np.ndarray
    representative_rows: np.ndarray
    entity_ids: np.ndarray


def _entity_layout(mesh: CellMesh, degree: int, /) -> _PeriodicEntityLayout:
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic entity layout requires periodic mesh topology.")
    topology.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    if isinstance(degree, bool) or not 0 <= degree <= mesh.topological_dimension:
        raise ValueError("Entity degree is outside the mesh topology.")
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lower, upper = map(int, lifted_offsets[degree : degree + 2])
    quotient_lower, quotient_upper = map(int, quotient_offsets[degree : degree + 2])
    orbit_rows = np.asarray(topology.orbit_indices, dtype=np.int64)[lower:upper]
    orbit_rows = orbit_rows - quotient_lower
    representative_rows = np.asarray(topology.representatives, dtype=np.int64)[
        quotient_lower:quotient_upper
    ]
    entity_ids = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
    quotient_ids = np.asarray(
        topology.quotient.entity_sets[degree].entity_ids, dtype=np.int64
    )
    if (
        orbit_rows.shape != entity_ids.shape
        or representative_rows.shape != quotient_ids.shape
        or np.any((orbit_rows < 0) | (orbit_rows >= quotient_ids.size))
        or np.any((representative_rows < 0) | (representative_rows >= entity_ids.size))
        or np.unique(quotient_ids).size != quotient_ids.size
    ):
        raise ValueError(
            "Periodic quotient tables do not match their lifted entity sets."
        )
    return _PeriodicEntityLayout(orbit_rows, representative_rows, quotient_ids)


class PeriodicEntityOrbitMap(StrictModule, NonTrainableState):
    """Lifted entity-to-quotient actions with packed ragged corner permutations.

    ``corner_permutation_offsets``/``corner_permutations`` retain one explicit
    canonical-corner permutation per lifted entity without padding variable-size
    polygon or polyhedron rows. ``winding_shifts`` are relative deck exponents
    from each representative.
    """

    degree: int = eqx.field(static=True)
    lifted_entity_ids: Array
    quotient_entity_ids: Array
    lifted_to_quotient: Array
    orientation_signs: Array
    winding_shifts: Array
    corner_permutation_offsets: Array
    corner_permutations: Array
    representative_lifted_rows: Array
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_lifted_topology_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    orbit_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        degree: int,
        layout: _PeriodicEntityLayout,
        orientation_signs: ArrayLike,
        winding_shifts: ArrayLike,
        corner_permutations: tuple[ArrayLike, ...],
        /,
    ) -> None:
        topology = mesh.periodic_topology
        if topology is None:
            raise ValueError("Periodic entity maps require periodic mesh topology.")
        signs = np.asarray(orientation_signs, dtype=np.int32)
        shifts = np.asarray(winding_shifts, dtype=np.int64)
        corners = tuple(np.asarray(row, dtype=np.int32) for row in corner_permutations)
        if (
            signs.shape != layout.orbit_rows.shape
            or shifts.ndim != 2
            or shifts.shape[0] != layout.orbit_rows.size
            or len(corners) != layout.orbit_rows.size
            or np.any(np.abs(signs) != 1)
            or any(
                row.ndim != 1
                or row.size == 0
                or not np.array_equal(np.sort(row), np.arange(row.size, dtype=np.int32))
                for row in corners
            )
        ):
            raise ValueError("Periodic entity actions must match the lifted entity set.")
        corner_offsets = np.zeros((len(corners) + 1,), dtype=np.int64)
        np.cumsum([row.size for row in corners], out=corner_offsets[1:])
        corner_values = (
            np.concatenate(corners).astype(np.int32, copy=False)
            if corners
            else np.zeros((0,), dtype=np.int32)
        )
        self.degree = int(degree)
        self.lifted_entity_ids = jnp.asarray(
            mesh.entity_set(self.degree).entity_ids, dtype=jnp.int64
        )
        self.quotient_entity_ids = jnp.asarray(layout.entity_ids, dtype=jnp.int64)
        self.lifted_to_quotient = jnp.asarray(layout.orbit_rows, dtype=jnp.int32)
        self.orientation_signs = jnp.asarray(signs, dtype=jnp.int32)
        self.winding_shifts = jnp.asarray(shifts, dtype=jnp.int64)
        self.corner_permutation_offsets = jnp.asarray(corner_offsets, dtype=jnp.int64)
        self.corner_permutations = jnp.asarray(corner_values, dtype=jnp.int32)
        self.representative_lifted_rows = jnp.asarray(
            layout.representative_rows, dtype=jnp.int32
        )
        self.source_mesh_id = mesh.mesh_id
        self.source_topology_id = mesh.topology_id
        self.source_lifted_topology_id = mesh.lifted_topology_id
        self.periodic_topology_id = topology.periodic_topology_id
        self.entity_set_id = mesh.entity_set(self.degree).entity_set_id
        self.orbit_map_id = canonical_fingerprint(
            {
                "kind": "periodic-oriented-entity-orbit-map",
                "source_mesh": self.source_mesh_id,
                "source_topology": self.source_topology_id,
                "lifted_topology": self.source_lifted_topology_id,
                "periodic_topology": self.periodic_topology_id,
                "entity_set": self.entity_set_id,
                "degree": self.degree,
                "lifted_to_quotient": array_tree_fingerprint(layout.orbit_rows),
                "orientation_signs": array_tree_fingerprint(signs),
                "winding_shifts": array_tree_fingerprint(shifts),
                "corner_permutation_offsets": array_tree_fingerprint(corner_offsets),
                "corner_permutations": array_tree_fingerprint(corner_values),
            }
        )

    def corner_permutation_row(self, row: int, /) -> np.ndarray:
        """Return the unpadded lifted-local corner order for one entity row."""
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be one integer lifted-entity row.")
        index = int(row)
        offsets = np.asarray(self.corner_permutation_offsets, dtype=np.int64)
        if not 0 <= index < offsets.size - 1:
            raise ValueError("row is outside the lifted entity map.")
        values = np.asarray(self.corner_permutations, dtype=np.int32)
        return values[int(offsets[index]) : int(offsets[index + 1])]

    def require_source(self, mesh: CellMesh, /) -> None:
        topology = mesh.periodic_topology if isinstance(mesh, CellMesh) else None
        if (
            topology is None
            or mesh.mesh_id != self.source_mesh_id
            or mesh.topology_id != self.source_topology_id
            or mesh.lifted_topology_id != self.source_lifted_topology_id
            or topology.periodic_topology_id != self.periodic_topology_id
            or mesh.entity_set(self.degree).entity_set_id != self.entity_set_id
        ):
            raise ValueError(
                "Periodic entity map belongs to a different source mesh revision."
            )

    def lift_oriented_scalars(self, quotient_values: ArrayLike, /) -> np.ndarray:
        values = np.asarray(quotient_values)
        orbit = np.asarray(self.lifted_to_quotient, dtype=np.int64)
        if values.ndim == 0 or values.shape[0] != self.quotient_entity_ids.shape[0]:
            raise ValueError(
                "Quotient coefficients must match the quotient entity count."
            )
        signs = np.asarray(self.orientation_signs, dtype=values.dtype)
        signs = signs.reshape((signs.size,) + (1,) * (values.ndim - 1))
        return values[orbit] * signs

    def collapse_oriented_scalars(self, lifted_values: ArrayLike, /) -> np.ndarray:
        values = np.asarray(lifted_values)
        orbit = np.asarray(self.lifted_to_quotient, dtype=np.int64)
        representatives = np.asarray(self.representative_lifted_rows, dtype=np.int64)
        if values.ndim == 0 or values.shape[0] != orbit.size:
            raise ValueError("Lifted coefficients must match the lifted entity count.")
        signs = np.asarray(self.orientation_signs, dtype=values.dtype)
        signs = signs.reshape((signs.size,) + (1,) * (values.ndim - 1))
        canonical = values * signs
        quotient = canonical[representatives]
        expanded = self.lift_oriented_scalars(quotient)
        if values.dtype.kind in "fc":
            consistent = np.allclose(values, expanded, rtol=1.0e-12, atol=1.0e-14)
        else:
            consistent = np.array_equal(values, expanded)
        if not consistent:
            raise ValueError(
                "Lifted coefficients are not deck-equivariant on entity orbits."
            )
        return quotient


def periodic_entity_orbit_map(mesh: CellMesh, degree: int, /) -> PeriodicEntityOrbitMap:
    """Expose quotient IDs and packed per-row orientation, corner, and winding maps."""
    layout = _entity_layout(mesh, degree)
    topology = mesh.periodic_topology
    if topology is None:
        raise RuntimeError("Periodic entity layout lost its quotient topology.")
    entity_vertices = _entity_vertex_rows(mesh)[degree]
    if len(entity_vertices) != layout.orbit_rows.size:
        raise ValueError(
            "Periodic entity corner rows do not match the lifted entity set."
        )
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    root_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[representatives]
    vertex_shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
    orders = identification_orders(topology.cell)
    offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lower, upper = map(int, offsets[degree : degree + 2])
    quotient_lower = int(quotient_offsets[degree])
    stored_signs = np.asarray(topology.orbit_orientations, dtype=np.int32)[lower:upper]
    shifts = np.asarray(topology.orbit_shifts, dtype=np.int64)[lower:upper]
    key_offsets = np.asarray(topology.entity_key_offsets, dtype=np.int64)
    key_values = np.asarray(topology.entity_key_values, dtype=np.int64)
    entity_ids = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
    cell_kind_by_id = {
        int(identifier): block.cell_kind
        for block in mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    corner_rows: list[np.ndarray] = []
    for row, vertex_row in enumerate(entity_vertices):
        vertices = np.asarray(vertex_row, dtype=np.int64)
        identifiers = root_ids[vertices]
        exponents = vertex_shifts[vertices]
        fixed_entities = fixed[vertices]
        quotient_row = quotient_lower + int(layout.orbit_rows[row])
        expected = key_values[key_offsets[quotient_row] : key_offsets[quotient_row + 1]]
        cell_kind = (
            cell_kind_by_id[int(entity_ids[row])]
            if degree == mesh.topological_dimension
            else ""
        )
        if degree == mesh.topological_dimension and cell_kind == "polyhedron":
            key, _, permutation = _canonical_unordered_form(
                identifiers, exponents, fixed_entities, orders
            )
            if not np.array_equal(np.asarray(key, dtype=np.int64), expected):
                raise ValueError(
                    "Periodic polyhedron corner keys disagree with the quotient entity."
                )
            if int(stored_signs[row]) != 1:
                raise ValueError(
                    "Periodic polyhedron orientation must preserve its lifted cell orientation."
                )
        else:
            simplex = degree == mesh.topological_dimension and cell_kind.startswith(
                ("simplex:", "triangle", "tetrahedron")
            )
            permutation = None
            for order, orientation in _orientation_orders(vertices.size, simplex=simplex):
                # Same moving-vertex anchor convention as the quotient keys.
                candidate, _ = _interleaved_key(
                    identifiers, exponents, fixed_entities, order, orders
                )
                if np.array_equal(np.asarray(candidate, dtype=np.int64), expected):
                    permutation_rows = np.asarray(order, dtype=np.int64)
                    if int(orientation) != int(stored_signs[row]):
                        raise ValueError(
                            "Periodic corner permutation has an inconsistent orientation."
                        )
                    permutation = permutation_rows.astype(np.int32)
                    break
            if permutation is None:
                raise ValueError("Periodic entity has no canonical corner permutation.")
        corner_rows.append(np.asarray(permutation, dtype=np.int32))
    return PeriodicEntityOrbitMap(
        mesh, degree, layout, stored_signs, shifts, tuple(corner_rows)
    )


def periodic_quotient_facet_cell_classes(
    mesh: CellMesh, cell_classes: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quotient incidence of every lifted facet: count, lowest and highest cell class.

    A lifted cover cuts the quotient along its outer boundary, where a facet has
    one incident lifted cell although its orbit is interior. Each quotient cell is
    counted once through its representative lift, so every facet orbit reports its
    quotient incidences (a cell meeting one facet orbit through two of its own
    facets counts twice). Values are returned per lifted facet row.
    """
    dimension = mesh.topological_dimension
    facet_map = periodic_entity_orbit_map(mesh, dimension - 1)
    facet_orbits = np.asarray(facet_map.lifted_to_quotient, dtype=np.int64)
    cell_representatives = np.asarray(
        periodic_entity_orbit_map(mesh, dimension).representative_lifted_rows,
        dtype=np.int64,
    )
    classes = np.asarray(cell_classes, dtype=np.int64)
    if classes.shape != (mesh.entity_set(dimension).count,):
        raise ValueError("cell_classes must align with the lifted cell rows.")
    relation = mesh.topology.incidences[dimension - 1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    facets = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    cells = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    representative = np.zeros((classes.size,), dtype=np.bool_)
    representative[cell_representatives] = True
    selected = representative[cells]
    orbit_rows = facet_orbits[facets[selected]]
    incident_classes = classes[cells[selected]]
    orbit_count = int(facet_map.quotient_entity_ids.size)
    incident = np.bincount(orbit_rows, minlength=orbit_count)
    lowest = np.full((orbit_count,), np.iinfo(np.int64).max, dtype=np.int64)
    highest = np.full((orbit_count,), -1, dtype=np.int64)
    np.minimum.at(lowest, orbit_rows, incident_classes)
    np.maximum.at(highest, orbit_rows, incident_classes)
    return incident[facet_orbits], lowest[facet_orbits], highest[facet_orbits]


def transfer_periodic_p1_vertices(
    source: CellMesh,
    target: CellMesh,
    stencil: VertexInterpolationStencil,
    source_quotient_values: ArrayLike,
    /,
) -> np.ndarray:
    """Apply a P1 scalar interpolation in lifted space and descend by orbit.

    The source quotient field is expanded with the exact entity orientation
    action, interpolated through the source-bound vertex stencil, then checked
    for target deck equivariance before quotient values are returned.
    """
    from ._lineage import VertexInterpolationStencil

    if not isinstance(stencil, VertexInterpolationStencil):
        raise TypeError("stencil must be VertexInterpolationStencil.")
    source_map = periodic_entity_orbit_map(source, 0)
    target_map = periodic_entity_orbit_map(target, 0)
    source_map.require_source(source)
    target_map.require_source(target)
    source_ids = np.asarray(source_map.lifted_entity_ids, dtype=np.int64)
    target_ids = np.asarray(target_map.lifted_entity_ids, dtype=np.int64)
    if (
        stencil.source_entity_set_id != source.entity_set(0).entity_set_id
        or stencil.target_entity_set_id != target.entity_set(0).entity_set_id
        or not np.array_equal(np.asarray(stencil.target_global_ids), target_ids)
    ):
        raise ValueError(
            "Periodic P1 stencil endpoints do not match the source and target meshes."
        )
    source_lifted = source_map.lift_oriented_scalars(source_quotient_values)
    target_lifted = stencil.apply(source_ids, source_lifted)
    return target_map.collapse_oriented_scalars(target_lifted)


def _periodic_cell_layout(mesh: CellMesh, /) -> _PeriodicCellLayout | None:
    if mesh.periodic_topology is None:
        return None
    layout = _entity_layout(mesh, mesh.topological_dimension)
    block_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    cell_ids = np.asarray(
        mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
    )
    if not np.array_equal(block_ids, cell_ids):
        raise ValueError("Periodic cell orbits must use canonical lifted cell ordering.")
    return _PeriodicCellLayout(*layout)


class PeriodicOrbitClosure(StrictModule, NonTrainableState):
    """Orbit-complete entity marks bound to one exact lifted source revision."""

    requested_global_ids: Array
    closed_global_ids: Array
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_lifted_topology_id: str = eqx.field(static=True)
    periodic_topology_id: str | None = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        degree: int,
        requested_global_ids: ArrayLike,
        closed_global_ids: ArrayLike,
        /,
    ) -> None:
        requested = np.asarray(requested_global_ids, dtype=np.int64)
        closed = np.asarray(closed_global_ids, dtype=np.int64)
        if (
            requested.ndim != 1
            or closed.ndim != 1
            or np.unique(closed).size != closed.size
            or np.any(~np.isin(requested, closed))
        ):
            raise ValueError(
                "Periodic orbit closure IDs must be unique and contain every request."
            )
        self.requested_global_ids = jnp.asarray(requested, dtype=jnp.int64)
        self.closed_global_ids = jnp.asarray(closed, dtype=jnp.int64)
        self.source_mesh_id = mesh.mesh_id
        self.source_topology_id = mesh.topology_id
        self.source_lifted_topology_id = mesh.lifted_topology_id
        self.periodic_topology_id = (
            None
            if mesh.periodic_topology is None
            else mesh.periodic_topology.periodic_topology_id
        )
        self.entity_set_id = mesh.entity_set(degree).entity_set_id
        self.degree = int(degree)
        self.closure_id = canonical_fingerprint(
            {
                "kind": "periodic-orbit-closure",
                "source_mesh": self.source_mesh_id,
                "source_topology": self.source_topology_id,
                "lifted_topology": self.source_lifted_topology_id,
                "periodic_topology": self.periodic_topology_id,
                "entity_set": self.entity_set_id,
                "degree": self.degree,
                "requested": array_tree_fingerprint(requested),
                "closed": array_tree_fingerprint(closed),
            }
        )

    def require_source(self, mesh: CellMesh, degree: int, /) -> np.ndarray:
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if (
            mesh.mesh_id != self.source_mesh_id
            or mesh.topology_id != self.source_topology_id
            or mesh.lifted_topology_id != self.source_lifted_topology_id
            or isinstance(degree, bool)
            or int(degree) != self.degree
            or mesh.entity_set(self.degree).entity_set_id != self.entity_set_id
            or (
                None
                if mesh.periodic_topology is None
                else mesh.periodic_topology.periodic_topology_id
            )
            != self.periodic_topology_id
        ):
            raise ValueError(
                "Periodic orbit closure belongs to a different source mesh revision."
            )
        return np.asarray(self.closed_global_ids, dtype=np.int64)


def periodic_orbit_closure(
    mesh: CellMesh, degree: int, entity_global_ids: ArrayLike, /
) -> PeriodicOrbitClosure:
    """Return source-bound orbit-complete marks, preserving winding identity.

    Distinct entities with identical representatives but different winding keys
    remain distinct unless the quotient topology places them in the same orbit.
    The result must be checked with :meth:`PeriodicOrbitClosure.require_source`
    before an edit is executed, so marks cannot migrate across revisions.
    """
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be a CellMesh.")
    if isinstance(degree, bool) or not 0 <= degree <= mesh.topological_dimension:
        raise ValueError("Entity degree is outside the mesh topology.")
    supplied = np.asarray(entity_global_ids)
    if supplied.ndim != 1 or supplied.dtype.kind not in "iu":
        raise TypeError("entity_global_ids must be a rank-one integer array.")
    identifiers = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
    if identifiers.size == 0:
        if supplied.size:
            raise ValueError("Periodic mark references an absent entity.")
        empty = np.zeros((0,), dtype=np.int64)
        return PeriodicOrbitClosure(mesh, degree, empty, empty)
    order = np.argsort(identifiers, kind="stable")
    positions = np.searchsorted(identifiers[order], supplied.astype(np.int64, copy=False))
    bounded = np.minimum(positions, identifiers.size - 1)
    rows = order[bounded]
    if np.any(identifiers[rows] != supplied):
        raise ValueError("Periodic mark references an entity outside the lifted mesh.")
    requested = identifiers[np.unique(rows)]
    if mesh.periodic_topology is None:
        closed = requested
    else:
        layout = _entity_layout(mesh, degree)
        marked_orbits = np.unique(layout.orbit_rows[rows])
        closed = identifiers[np.isin(layout.orbit_rows, marked_orbits)]
    return PeriodicOrbitClosure(mesh, degree, requested, closed)


class PeriodicRefinementEdgeLock(StrictModule, NonTrainableState):
    """One selected lifted edge per marked cell, closed over periodic orbits."""

    cell_global_ids: Array
    selected_edge_global_ids: Array
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_lifted_topology_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    cell_entity_set_id: str = eqx.field(static=True)
    edge_entity_set_id: str = eqx.field(static=True)
    lock_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        cell_global_ids: ArrayLike,
        selected_edge_global_ids: ArrayLike,
        /,
    ) -> None:
        cells = np.asarray(cell_global_ids, dtype=np.int64)
        edges = np.asarray(selected_edge_global_ids, dtype=np.int64)
        if (
            cells.ndim != 1
            or edges.shape != cells.shape
            or np.unique(cells).size != cells.size
        ):
            raise ValueError(
                "A periodic edge lock needs one selected edge per unique cell."
            )
        if mesh.periodic_topology is None:
            raise ValueError("A periodic edge lock requires periodic source topology.")
        if cells.size:
            order = np.argsort(cells, kind="stable")
            cells, edges = cells[order], edges[order]
        self.cell_global_ids = jnp.asarray(cells, dtype=jnp.int64)
        self.selected_edge_global_ids = jnp.asarray(edges, dtype=jnp.int64)
        self.source_mesh_id = mesh.mesh_id
        self.source_topology_id = mesh.topology_id
        self.source_lifted_topology_id = mesh.lifted_topology_id
        self.periodic_topology_id = mesh.periodic_topology.periodic_topology_id
        self.cell_entity_set_id = mesh.entity_set(
            mesh.topological_dimension
        ).entity_set_id
        self.edge_entity_set_id = mesh.entity_set(1).entity_set_id
        self.lock_id = canonical_fingerprint(
            {
                "kind": "periodic-refinement-edge-lock",
                "source_mesh": self.source_mesh_id,
                "source_topology": self.source_topology_id,
                "lifted_topology": self.source_lifted_topology_id,
                "periodic_topology": self.periodic_topology_id,
                "cell_entity_set": self.cell_entity_set_id,
                "edge_entity_set": self.edge_entity_set_id,
                "cells": array_tree_fingerprint(cells),
                "edges": array_tree_fingerprint(edges),
            }
        )

    def require_source(self, mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if (
            mesh.periodic_topology is None
            or mesh.mesh_id != self.source_mesh_id
            or mesh.topology_id != self.source_topology_id
            or mesh.lifted_topology_id != self.source_lifted_topology_id
            or mesh.periodic_topology.periodic_topology_id != self.periodic_topology_id
            or mesh.entity_set(mesh.topological_dimension).entity_set_id
            != self.cell_entity_set_id
            or mesh.entity_set(1).entity_set_id != self.edge_entity_set_id
        ):
            raise ValueError(
                "Periodic edge lock belongs to a different source mesh revision."
            )
        cells = np.asarray(self.cell_global_ids, dtype=np.int64)
        edges = np.asarray(self.selected_edge_global_ids, dtype=np.int64)
        validated = prepare_periodic_refinement_edge_lock(mesh, cells, edges)
        if validated.lock_id != self.lock_id:
            raise ValueError("Periodic edge lock contents are inconsistent.")
        return cells, edges


def prepare_periodic_refinement_edge_lock(
    mesh: CellMesh,
    cell_global_ids: ArrayLike,
    selected_edge_global_ids: ArrayLike,
    /,
) -> PeriodicRefinementEdgeLock:
    """Validate per-cell edge choices and bind them to full periodic orbits.

    Cell IDs and edge IDs are aligned input vectors. Every selected cell orbit
    must be complete and select one quotient edge orbit; the selected lifted
    edges themselves must cover their complete quotient edge orbits.
    """
    if not isinstance(mesh, CellMesh) or mesh.periodic_topology is None:
        raise ValueError("Periodic edge selection requires a periodic CellMesh.")
    dimension = mesh.topological_dimension
    cell_kind = {2: "triangle", 3: "tetrahedron"}.get(dimension)
    if cell_kind is None or any(block.cell_kind != cell_kind for block in mesh.blocks):
        raise ValueError(
            "Periodic bisection edge locks require triangle or tetrahedron topology."
        )
    supplied_cells = np.asarray(cell_global_ids)
    supplied_edges = np.asarray(selected_edge_global_ids)
    if (
        supplied_cells.ndim != 1
        or supplied_cells.dtype.kind not in "iu"
        or supplied_edges.shape != supplied_cells.shape
        or supplied_edges.dtype.kind not in "iu"
    ):
        raise TypeError("Cell and selected edge IDs must be aligned integer vectors.")
    cells = supplied_cells.astype(np.int64, copy=False)
    edges = supplied_edges.astype(np.int64, copy=False)
    if np.unique(cells).size != cells.size:
        raise ValueError("Periodic bisection marks must contain each lifted cell once.")

    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    if cell_ids.size == 0 or edge_ids.size == 0:
        raise ValueError("Periodic bisection requires nonempty cells and edges.")
    cell_order = np.argsort(cell_ids, kind="stable")
    edge_order = np.argsort(edge_ids, kind="stable")
    cell_positions = np.searchsorted(cell_ids[cell_order], cells)
    edge_positions = np.searchsorted(edge_ids[edge_order], edges)
    if np.any(cell_positions >= cell_ids.size) or np.any(edge_positions >= edge_ids.size):
        raise ValueError("Periodic bisection references an absent cell or edge.")
    cell_rows = cell_order[cell_positions]
    edge_rows = edge_order[edge_positions]
    if np.any(cell_ids[cell_rows] != cells) or np.any(edge_ids[edge_rows] != edges):
        raise ValueError("Periodic bisection references an absent cell or edge.")

    cell_layout = _periodic_cell_layout(mesh)
    edge_layout = _entity_layout(mesh, 1)
    if cell_layout is None:
        raise RuntimeError("Periodic cell edge lock lost its quotient layout.")
    requested_cells = periodic_orbit_closure(mesh, dimension, cells).require_source(
        mesh, dimension
    )
    if not np.array_equal(np.sort(requested_cells), np.sort(cells)):
        raise ValueError(
            "Periodic bisection cell marks must contain complete cell orbits."
        )
    connectivity_edges = _cell_edge_rows(mesh)
    edge_count = dimension * (dimension + 1) // 2
    if connectivity_edges.shape != (cell_ids.size, edge_count):
        raise ValueError(
            "Periodic cell-edge routes do not match lifted simplex identities."
        )
    if np.any(~np.any(connectivity_edges[cell_rows] == edge_rows[:, None], axis=1)):
        raise ValueError(
            "Each selected bisection edge must be incident to its marked cell."
        )

    cell_orbits = cell_layout.orbit_rows[cell_rows]
    edge_orbits = edge_layout.orbit_rows[edge_rows]
    for orbit in np.unique(cell_orbits):
        members = cell_orbits == orbit
        if np.unique(edge_orbits[members]).size != 1:
            raise ValueError(
                "Cells in one periodic orbit selected different quotient edges."
            )
    requested_edges = periodic_orbit_closure(mesh, 1, edges).require_source(mesh, 1)
    if not np.array_equal(np.sort(requested_edges), np.sort(np.unique(edges))):
        raise ValueError(
            "Selected bisection edges must cover complete periodic edge orbits."
        )
    return PeriodicRefinementEdgeLock(mesh, cells, edges)


def _cell_edge_rows(mesh: CellMesh, /) -> np.ndarray:
    """Return native edge rows for cells in canonical cell-ID order."""
    dimension = mesh.topological_dimension
    if dimension not in (2, 3):
        raise ValueError("Periodic edge routes require triangles or tetrahedra.")
    entity_rows = _entity_vertex_rows(mesh)
    lifted_edges = entity_rows[1]
    edge_row_by_vertices = {
        tuple(sorted(int(value) for value in vertices)): row
        for row, vertices in enumerate(lifted_edges)
    }
    block_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    if block_cell_ids.size != len(entity_rows[dimension]):
        raise ValueError("Periodic cell connectivity rows do not match block identities.")
    block_row_by_id = {
        int(identifier): row for row, identifier in enumerate(block_cell_ids)
    }
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if set(block_row_by_id) != set(map(int, cell_ids)):
        raise ValueError("Periodic edge routes do not cover canonical cell identities.")
    rows: list[tuple[int, ...]] = []
    for identifier in cell_ids:
        vertices = entity_rows[dimension][block_row_by_id[int(identifier)]]
        if len(vertices) != dimension + 1:
            raise ValueError("Periodic bisection requires simplex cell vertices.")
        pairs = tuple(combinations((int(value) for value in vertices), 2))
        try:
            rows.append(
                tuple(edge_row_by_vertices[tuple(sorted(pair))] for pair in pairs)
            )
        except KeyError as error:
            raise ValueError(
                "Periodic cell edge is absent from canonical connectivity."
            ) from error
    return np.asarray(rows, dtype=np.int64)


def _periodic_quotient_adjacency(mesh: CellMesh, /) -> CellAdjacency:
    if mesh.periodic_topology is None:
        raise ValueError("Periodic quotient adjacency requires periodic mesh topology.")
    topology = mesh.periodic_topology.quotient
    dimension = mesh.topological_dimension
    if dimension <= 0:
        raise ValueError(
            "Periodic cell adjacency requires positive topological dimension."
        )
    incidence = topology.incidences[dimension - 1].relation
    valid = np.asarray(incidence.valid, dtype=np.bool_).reshape(-1)
    facets = np.asarray(incidence.source_indices, dtype=np.int64).reshape(-1)[valid]
    cells = np.asarray(incidence.target_indices, dtype=np.int64).reshape(-1)[valid]
    cell_count = topology.entity_sets[dimension].count
    if (
        facets.shape != cells.shape
        or np.any((facets < 0) | (facets >= topology.entity_sets[dimension - 1].count))
        or np.any((cells < 0) | (cells >= cell_count))
    ):
        raise ValueError("Periodic quotient incidence contains invalid facet routes.")
    if facets.size == 0:
        return CellAdjacency(np.zeros((0, 2), dtype=np.int64), cell_count)
    order = np.lexsort((cells, facets))
    facets, cells = facets[order], cells[order]
    starts = np.concatenate(
        ([0], np.flatnonzero(facets[1:] != facets[:-1]) + 1, [facets.size])
    )
    pairs: list[tuple[int, int]] = []
    for start, stop in pairwise(starts):
        incident = np.unique(cells[int(start) : int(stop)])
        if incident.size > 2:
            raise ValueError(
                "A periodic quotient facet is incident to more than two cells."
            )
        if incident.size == 2:
            pairs.append((int(incident[0]), int(incident[1])))
    return CellAdjacency(np.asarray(pairs, dtype=np.int64).reshape((-1, 2)), cell_count)


def _periodic_lifted_adjacency(
    mesh: CellMesh, layout: _PeriodicCellLayout | None = None, /
) -> CellAdjacency:
    resolved = _periodic_cell_layout(mesh) if layout is None else layout
    if resolved is None:
        raise ValueError("Periodic lifted adjacency requires periodic mesh topology.")
    quotient = _periodic_quotient_adjacency(mesh)
    groups = tuple(
        np.flatnonzero(resolved.orbit_rows == row)
        for row in range(resolved.entity_ids.size)
    )
    pairs: list[np.ndarray] = []
    for first, second in quotient.undirected_pairs():
        left, right = groups[int(first)], groups[int(second)]
        if left.size == 0 or right.size == 0:
            raise ValueError("A periodic quotient cell has no lifted representative.")
        pairs.append(
            np.stack(
                (np.repeat(left, right.size), np.tile(right, left.size)),
                axis=1,
            )
        )
    lifted_count = resolved.orbit_rows.size
    values = np.concatenate(pairs, axis=0) if pairs else np.zeros((0, 2), dtype=np.int64)
    return CellAdjacency(values, lifted_count)


def _periodic_quotient_weights(
    lifted_weights: ArrayLike, layout: _PeriodicCellLayout, /
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(lifted_weights, dtype=np.float64)
    if (
        weights.shape != layout.orbit_rows.shape
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError(
            "Periodic cell weights must be finite, positive, and lifted-order."
        )
    counts = np.bincount(layout.orbit_rows, minlength=layout.entity_ids.size)
    if np.any(counts == 0):
        raise ValueError("Every periodic quotient cell requires a lifted representative.")
    quotient_weights = (
        np.bincount(layout.orbit_rows, weights=weights, minlength=layout.entity_ids.size)
        / counts
    )
    if not np.allclose(
        weights,
        quotient_weights[layout.orbit_rows],
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("Load weights must agree across each periodic cell orbit.")
    return quotient_weights, quotient_weights[layout.orbit_rows] / counts[
        layout.orbit_rows
    ]


def _periodic_collapse_owners(
    lifted_owners: ArrayLike, layout: _PeriodicCellLayout, /
) -> np.ndarray:
    owners = np.asarray(lifted_owners, dtype=np.int64)
    if owners.shape != layout.orbit_rows.shape:
        raise ValueError("Periodic ownership must follow lifted cell order.")
    count = layout.entity_ids.size
    lower = np.full((count,), np.iinfo(np.int64).max, dtype=np.int64)
    upper = np.full((count,), -1, dtype=np.int64)
    np.minimum.at(lower, layout.orbit_rows, owners)
    np.maximum.at(upper, layout.orbit_rows, owners)
    if np.any(lower != upper):
        raise ValueError("Periodic cell orbits must share one owner.")
    return lower


def _periodic_expand_owners(
    quotient_owners: ArrayLike, layout: _PeriodicCellLayout, /
) -> np.ndarray:
    owners = np.asarray(quotient_owners, dtype=np.int32)
    if owners.shape != (layout.entity_ids.size,):
        raise ValueError("Quotient ownership must cover every periodic cell orbit.")
    return owners[layout.orbit_rows]


def _periodic_quotient_cell_centroids(
    mesh: CellMesh, layout: _PeriodicCellLayout, /
) -> np.ndarray:
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic centroid preparation requires periodic mesh topology.")
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    block_counts = np.asarray([block.cell_count for block in mesh.blocks], dtype=np.int64)
    block_offsets = np.concatenate(([0], np.cumsum(block_counts)))
    representative_rows = layout.representative_rows
    block_rows = np.searchsorted(block_offsets[1:], representative_rows, side="right")
    centroids = np.empty(
        (layout.entity_ids.size, mesh.ambient_dimension), dtype=np.float64
    )
    for quotient_row, (block_index, lifted_row) in enumerate(
        zip(block_rows, representative_rows, strict=True)
    ):
        block = mesh.blocks[int(block_index)]
        local_row = int(lifted_row - block_offsets[int(block_index)])
        vertices = np.asarray(block.vertices, dtype=np.int64)[local_row]
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)[local_row]
        active = vertices[valid]
        if active.size == 0:
            raise ValueError("Periodic cell centroids require active vertices.")
        anchor = vertex_shifts[int(active[0])]
        home_coordinates = apply_periodic(
            topology.cell,
            coordinates[active],
            np.broadcast_to(-anchor, (active.size, anchor.size)),
        )
        centroids[quotient_row] = np.mean(home_coordinates, axis=0)
    return centroids


def _stencil_orbit_signature(
    representatives: np.ndarray,
    shifts: np.ndarray,
    fixed: np.ndarray,
    weights: np.ndarray,
    orders: np.ndarray,
    /,
) -> tuple[tuple[tuple[int, tuple[int, ...], int, int], ...], np.ndarray]:
    coefficients = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(weights)
    )
    active = np.asarray([value != 0 for value in coefficients], dtype=np.bool_)
    if not np.any(active):
        raise ValueError("A periodic target vertex requires a nonzero source stencil.")
    identifiers = np.asarray(representatives, dtype=np.int64)[active]
    exponents = np.asarray(shifts, dtype=np.int64)[active]
    fixed_rows = np.asarray(fixed, dtype=np.bool_)[active]
    exact_weights = tuple(
        value for value, use in zip(coefficients, active, strict=True) if use
    )
    exponents = reduce_periodic_exponents(exponents, orders)
    exponents[fixed_rows] = 0
    choices = tuple(
        np.unique(exponents[~fixed_rows[:, axis], axis])
        if np.any(~fixed_rows[:, axis])
        else np.zeros((1,), dtype=np.int64)
        for axis in range(orders.size)
    )
    candidates = []
    for candidate in product(*choices):
        anchor = np.asarray(candidate, dtype=np.int64)
        relative = reduce_periodic_exponents(exponents - anchor, orders)
        entries = tuple(
            sorted(
                (
                    int(identifier),
                    tuple(int(value) for value in shift),
                    weight.numerator,
                    weight.denominator,
                )
                for identifier, shift, weight in zip(
                    identifiers, relative, exact_weights, strict=True
                )
            )
        )
        candidates.append((entries, tuple(int(value) for value in anchor), anchor))
    signature, _, anchor = min(candidates, key=lambda value: (value[0], value[1]))
    return signature, anchor


def periodic_topology_after_edit(
    source: CellMesh,
    target: CellMesh,
    stencil: VertexInterpolationStencil,
    /,
    *,
    vertex_orbit_keys: ArrayLike | None = None,
    vertex_windings: ArrayLike | None = None,
    source_entity_degree: int | None = None,
    exact_coordinate_source: ExactPlcCellGeometrySource | None = None,
) -> PeriodicMeshTopology | None:
    """Rebuild a quotient descriptor for an unbound lifted topology edit target.

    The default key is derived from source global IDs, winding, fixed-generator
    status, and exact interpolation weights; a target vertex that retains a
    source global ID keeps that source vertex orbit. Generated vertices whose source
    stencil omits a discrete ancestry coordinate may instead pass explicit
    ``vertex_orbit_keys``/``vertex_windings`` plus ``source_entity_degree``.
    Each explicit key starts with a source quotient entity ID; remaining columns
    carry source-owned ancestry such as a boundary-layer level. Its winding must
    occur on that source entity orbit.

    A producer that owns the orbit identity of every target vertex (for example
    an orbit-locked local metric kernel whose generated vertices are exact deck
    images of one root) passes keys and windings without ``source_entity_degree``.
    Equal keys then form one vertex orbit and windings are the absolute deck
    exponents of each member; located interpolation stencils are not orbit
    evidence in this mode. Coordinates validate the resulting identity bit for
    bit through the canonical PeriodicMeshTopology constructor in every mode;
    ``exact_coordinate_source`` is the target's exact vertex bank when new
    vertices are canonical roundings of exact quotient-chart translates rather
    than translates of their rounded roots.
    """
    if not isinstance(source, CellMesh) or not isinstance(target, CellMesh):
        raise TypeError(
            "Periodic topology edits require source and target CellMesh values."
        )
    source_topology = source.periodic_topology
    if source_topology is None:
        return None
    source_topology.require_lifted(
        source.topology, source.lifted_topology_id, source.blocks
    )
    source_topology.require_coordinates(source.coordinates)
    if target.periodic_topology is not None:
        raise ValueError(
            "target must be an unbound lifted mesh before periodic rebinding."
        )
    from ._lineage import VertexInterpolationStencil

    if not isinstance(stencil, VertexInterpolationStencil):
        raise TypeError("stencil must be a VertexInterpolationStencil.")
    source_vertices = source.entity_set(0)
    target_vertices = target.entity_set(0)
    if (
        stencil.source_entity_set_id != source_vertices.entity_set_id
        or stencil.target_entity_set_id != target_vertices.entity_set_id
    ):
        raise ValueError(
            "Vertex interpolation stencil endpoints do not match the edit meshes."
        )
    target_ids = np.asarray(target_vertices.entity_ids, dtype=np.int64)
    if not np.array_equal(np.asarray(stencil.target_global_ids), target_ids):
        raise ValueError(
            "Periodic edit stencils must follow canonical target vertex order."
        )
    source_ids = np.asarray(source_vertices.entity_ids, dtype=np.int64)
    source_order = np.argsort(source_ids, kind="stable")
    source_sorted = source_ids[source_order]
    stencil_source_ids = np.asarray(stencil.source_global_ids, dtype=np.int64)
    stencil_valid = np.asarray(stencil.valid, dtype=np.bool_)
    stencil_weights = np.asarray(stencil.weights, dtype=np.float64)
    if not np.all(np.any(stencil_valid & (stencil_weights != 0.0), axis=1)):
        raise ValueError(
            "Every periodic target vertex requires source interpolation ancestry."
        )
    positions = np.minimum(
        np.searchsorted(source_sorted, stencil_source_ids), source_ids.size - 1
    )
    source_rows = source_order[positions]
    if np.any(stencil_valid & (source_sorted[positions] != stencil_source_ids)):
        raise ValueError("Periodic edit stencil references an absent source vertex.")

    representatives = np.asarray(source_topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(source_topology.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(source_topology.vertex_fixed_generators, dtype=np.bool_)
    if (
        representatives.shape != source_ids.shape
        or shifts.shape[0] != source_ids.size
        or fixed.shape != shifts.shape
    ):
        raise ValueError(
            "Source periodic vertex orbits do not cover the lifted source mesh."
        )
    orders = identification_orders(source_topology.cell)
    signatures: list[
        tuple[int, ...] | tuple[tuple[int, tuple[int, ...], int, int], ...]
    ] = []
    anchors = np.empty((target_ids.size, orders.size), dtype=np.int64)
    explicit = vertex_orbit_keys is not None or vertex_windings is not None
    if source_entity_degree is not None and not explicit:
        raise ValueError(
            "A source entity degree requires explicit periodic vertex orbit keys."
        )
    if explicit:
        if (
            vertex_orbit_keys is None
            or vertex_windings is None
            or (
                source_entity_degree is not None
                and (
                    isinstance(source_entity_degree, bool)
                    or not isinstance(source_entity_degree, (int, np.integer))
                    or not 0 <= int(source_entity_degree) <= source.topological_dimension
                )
            )
        ):
            raise ValueError(
                "Explicit periodic vertex ancestry requires orbit keys, windings, and an optional valid source entity degree."
            )
        keys = np.asarray(vertex_orbit_keys)
        windings = np.asarray(vertex_windings)
        if (
            keys.ndim != 2
            or keys.shape[0] != target_ids.size
            or keys.shape[1] < (2 if source_entity_degree is not None else 1)
            or keys.dtype.kind not in "iu"
            or windings.shape != (target_ids.size, orders.size)
            or windings.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Explicit periodic vertex keys and windings must cover every target vertex."
            )
        if (
            keys.dtype.kind == "u"
            and keys.size
            and int(np.max(keys)) > np.iinfo(np.int64).max
        ):
            raise OverflowError("Periodic ancestry keys must fit in int64.")
        keys = keys.astype(np.int64, copy=False)
        windings = reduce_periodic_exponents(
            windings.astype(np.int64, copy=False), orders
        )
        if np.any(keys < 0):
            raise ValueError(
                "Periodic vertex ancestry keys must be nonnegative source identities."
            )
        signatures = [tuple(int(value) for value in row) for row in keys]
        anchors[:] = windings
    if explicit and source_entity_degree is not None:
        degree = int(source_entity_degree)
        source_entity_ids = np.asarray(
            source_topology.quotient.entity_sets[degree].entity_ids, dtype=np.int64
        )
        source_entity_order = np.argsort(source_entity_ids, kind="stable")
        source_entity_sorted = source_entity_ids[source_entity_order]
        key_positions = np.searchsorted(source_entity_sorted, keys[:, 0])
        bounded = np.minimum(key_positions, source_entity_ids.size - 1)
        quotient_rows = source_entity_order[bounded]
        if np.any(source_entity_sorted[bounded] != keys[:, 0]):
            raise ValueError(
                "Periodic vertex ancestry references an absent quotient source entity."
            )
        lifted_offsets = np.asarray(source_topology.lifted_offsets, dtype=np.int64)
        quotient_offsets = np.asarray(source_topology.quotient_offsets, dtype=np.int64)
        lower, upper = map(int, lifted_offsets[degree : degree + 2])
        quotient_lower = int(quotient_offsets[degree])
        source_orbits = np.asarray(source_topology.orbit_indices, dtype=np.int64)[
            lower:upper
        ]
        source_orbits -= quotient_lower
        source_windings = np.asarray(source_topology.orbit_shifts, dtype=np.int64)[
            lower:upper
        ]
        available: list[set[tuple[int, ...]]] = [
            set() for _ in range(source_entity_ids.size)
        ]
        for row in range(source_orbits.size):
            orbit = int(source_orbits[row])
            winding = source_windings[row, :]
            available[orbit].add(tuple(int(value) for value in winding))
        winding_mismatch = False
        for row in range(quotient_rows.size):
            orbit = int(quotient_rows[row])
            winding = windings[row, :]
            if tuple(int(value) for value in winding) not in available[orbit]:
                winding_mismatch = True
                break
        if winding_mismatch:
            raise ValueError(
                "Target vertex winding is absent from its source entity orbit."
            )
    elif not explicit:
        root_ids = source_ids[representatives]
        weights = np.asarray(stencil.weights, dtype=np.float64)
        retained_positions = np.minimum(
            np.searchsorted(source_sorted, target_ids), source_ids.size - 1
        )
        retained = source_sorted[retained_positions] == target_ids
        unit_weight = np.ones((1,), dtype=np.float64)
        for target_row in range(target_ids.size):
            if retained[target_row]:
                # A retained vertex keeps its identity and therefore its orbit;
                # its transfer stencil (for example after relocation) is located
                # independently per lifted image and is not orbit evidence.
                rows = source_order[retained_positions[target_row : target_row + 1]]
                row_weights = unit_weight
            else:
                active = stencil_valid[target_row]
                rows = source_rows[target_row, active]
                row_weights = weights[target_row, active]
            signature, anchor = _stencil_orbit_signature(
                root_ids[rows],
                shifts[rows],
                fixed[rows],
                row_weights,
                orders,
            )
            signatures.append(signature)
            anchors[target_row] = anchor
    grouped: dict[
        tuple[int, ...] | tuple[tuple[int, tuple[int, ...], int, int], ...],
        list[int],
    ] = {}
    for row, signature in enumerate(signatures):
        grouped.setdefault(signature, []).append(row)
    target_representatives = np.empty((target_ids.size,), dtype=np.int32)
    target_shifts = np.empty((target_ids.size, orders.size), dtype=np.int64)
    for rows in grouped.values():
        representative = min(rows, key=lambda row: int(target_ids[row]))
        target_representatives[rows] = representative
        target_shifts[rows] = reduce_periodic_exponents(
            anchors[rows] - anchors[representative], orders
        )
    allocator = {}
    for degree, next_id in enumerate(source_topology.allocator_next_ids):
        if next_id >= 0:
            target_entity_ids = np.asarray(
                target.entity_set(degree).entity_ids, dtype=np.int64
            )
            allocator[degree] = max(
                int(next_id), int(np.max(target_entity_ids, initial=-1)) + 1
            )
    return PeriodicMeshTopology(
        target,
        source_topology.cell,
        target_representatives,
        target_shifts,
        entity_allocator_next_ids=allocator or None,
        exact_coordinate_source=exact_coordinate_source,
    )


def _periodic_edge_operation_signatures(
    mesh: CellMesh, endpoint_vertex_ids: ArrayLike, /
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]:
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic edge operations require a bound target mesh.")
    endpoints = np.asarray(endpoint_vertex_ids)
    if endpoints.ndim != 2 or endpoints.shape[1] != 2 or endpoints.dtype.kind not in "iu":
        raise TypeError("Refinement edge endpoints must be an integer (n, 2) array.")
    endpoints = endpoints.astype(np.int64, copy=False)
    if np.any(endpoints[:, 0] == endpoints[:, 1]):
        raise ValueError("A refinement edge must have two distinct endpoint identities.")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    positions = np.searchsorted(vertex_ids[order], endpoints)
    if np.any(positions >= vertex_ids.size):
        raise ValueError("A refinement edge endpoint is absent from the target mesh.")
    rows = order[positions]
    if not np.array_equal(vertex_ids[rows], endpoints):
        raise ValueError("A refinement edge endpoint is absent from the target mesh.")
    layout = _entity_layout(mesh, 0)
    quotient_ids = layout.entity_ids[layout.orbit_rows[rows]]
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)[rows]
    orders = identification_orders(topology.cell)
    signatures = []
    for row in range(endpoints.shape[0]):
        candidates = []
        for anchor in shifts[row]:
            relative = reduce_periodic_exponents(shifts[row] - anchor, orders)
            candidates.append(
                tuple(
                    sorted(
                        (
                            int(quotient_ids[row, endpoint]),
                            tuple(int(value) for value in relative[endpoint]),
                        )
                        for endpoint in range(2)
                    )
                )
            )
        signatures.append(min(candidates))
    return tuple(signatures)


class PeriodicCellEditCandidate(StrictModule, NonTrainableState):
    """Fully rebound periodic mesh staged against one immutable source revision."""

    source: CellMesh
    target: CellMesh
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_lifted_topology_id: str = eqx.field(static=True)
    source_periodic_topology_id: str = eqx.field(static=True)
    edge_lock_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMesh,
        target: CellMesh,
        edge_lock_id: str,
        /,
    ) -> None:
        if source.periodic_topology is None or target.periodic_topology is None:
            raise ValueError(
                "A periodic edit candidate requires bound source and target meshes."
            )
        self.source = source
        self.target = target
        self.source_mesh_id = source.mesh_id
        self.source_topology_id = source.topology_id
        self.source_lifted_topology_id = source.lifted_topology_id
        self.source_periodic_topology_id = source.periodic_topology.periodic_topology_id
        self.edge_lock_id = str(edge_lock_id)
        self.candidate_id = canonical_fingerprint(
            {
                "kind": "periodic-cell-edit-candidate",
                "source_mesh": self.source_mesh_id,
                "source_lifted_topology": self.source_lifted_topology_id,
                "source_periodic_topology": self.source_periodic_topology_id,
                "source_topology": self.source_topology_id,
                "target_mesh": target.mesh_id,
                "target_topology": target.topology_id,
                "target_periodic_topology": target.periodic_topology.periodic_topology_id,
                "edge_lock": self.edge_lock_id,
            }
        )

    def require_source(self, source: CellMesh, /) -> CellMesh:
        if (
            not isinstance(source, CellMesh)
            or source.mesh_id != self.source_mesh_id
            or source.topology_id != self.source_topology_id
            or source.lifted_topology_id != self.source_lifted_topology_id
            or source.periodic_topology is None
            or source.periodic_topology.periodic_topology_id
            != self.source_periodic_topology_id
        ):
            raise ValueError(
                "Periodic edit candidate belongs to a different source revision."
            )
        topology = self.target.periodic_topology
        if topology is None:
            raise ValueError("Periodic edit target lost its quotient descriptor.")
        topology.require_lifted(
            self.target.topology,
            self.target.lifted_topology_id,
            self.target.blocks,
        )
        topology.require_coordinates(self.target.coordinates)
        return self.target

    def rollback(self, source: CellMesh, /) -> CellMesh:
        self.require_source(source)
        return source


def stage_periodic_refinement_candidate(
    source: CellMesh,
    target: CellMesh,
    stencil: VertexInterpolationStencil,
    edge_lock: PeriodicRefinementEdgeLock,
    executed_parent_cell_global_ids: ArrayLike,
    executed_source_edge_global_ids: ArrayLike,
    executed_edge_vertex_global_ids: ArrayLike,
    /,
    *,
    vertex_orbit_keys: ArrayLike | None = None,
    vertex_windings: ArrayLike | None = None,
    source_entity_degree: int | None = None,
    exact_coordinate_source: ExactPlcCellGeometrySource | None = None,
) -> PeriodicCellEditCandidate:
    """Stage a full orbit of marked and conformity-closure bisections.

    Initial locked parent/source-edge choices must occur in the executed rows.
    Every split is checked by its stable endpoint IDs and periodic winding, so
    generated closure edges need no fabricated edge identity. A rejected stage
    leaves the immutable source untouched.
    """
    if not isinstance(edge_lock, PeriodicRefinementEdgeLock):
        raise TypeError("edge_lock must be PeriodicRefinementEdgeLock.")
    expected_cells, expected_edges = edge_lock.require_source(source)
    parents = np.asarray(executed_parent_cell_global_ids)
    source_edges = np.asarray(executed_source_edge_global_ids)
    endpoints = np.asarray(executed_edge_vertex_global_ids)
    if (
        parents.ndim != 1
        or parents.dtype.kind not in "iu"
        or source_edges.shape != parents.shape
        or source_edges.dtype.kind not in "i"
        or endpoints.shape != (parents.size, 2)
        or endpoints.dtype.kind not in "iu"
    ):
        raise TypeError(
            "Executed split IDs and endpoint pairs must be aligned integer arrays."
        )
    parents = parents.astype(np.int64, copy=False)
    source_edges = source_edges.astype(np.int64, copy=False)
    endpoints = endpoints.astype(np.int64, copy=False)
    if np.any(source_edges < -1):
        raise ValueError("Only generated non-source edges may use source edge ID -1.")

    for cell_id, edge_id in zip(expected_cells, expected_edges, strict=True):
        if not np.any((parents == cell_id) & (source_edges == edge_id)):
            raise ValueError(
                "An initially locked bisection choice is absent from the staged edit."
            )

    source_cell_ids = np.asarray(
        source.entity_set(source.topological_dimension).entity_ids, dtype=np.int64
    )
    cell_order = np.argsort(source_cell_ids, kind="stable")
    parent_positions = np.searchsorted(source_cell_ids[cell_order], parents)
    if np.any(parent_positions >= source_cell_ids.size) or not np.array_equal(
        source_cell_ids[
            cell_order[np.minimum(parent_positions, source_cell_ids.size - 1)]
        ],
        parents,
    ):
        raise ValueError("Bisection parent IDs must retain their source-cell ancestry.")
    source_edge_ids = np.asarray(source.entity_set(1).entity_ids, dtype=np.int64)
    source_edge_order = np.argsort(source_edge_ids, kind="stable")
    source_edge_positions = np.searchsorted(
        source_edge_ids[source_edge_order], source_edges, side="left"
    )
    source_edge_vertices = np.stack(_entity_vertex_rows(source)[1])
    if source_edge_vertices.shape != (source_edge_ids.size, 2):
        raise ValueError("Source periodic simplex edge identities are unavailable.")
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    source_edge_pairs = np.sort(source_vertex_ids[source_edge_vertices], axis=1)
    pair_to_edge_ids: dict[tuple[int, int], int] = {
        (int(source_edge_pairs[row, 0]), int(source_edge_pairs[row, 1])): int(
            source_edge_ids[row]
        )
        for row in range(source_edge_ids.size)
    }
    for row, source_edge_id in enumerate(source_edges):
        pair = tuple(sorted(int(value) for value in endpoints[row]))
        if source_edge_id >= 0:
            if source_edge_positions[row] >= source_edge_ids.size:
                raise ValueError("Bisection references an absent source edge ID.")
            source_edge_row = source_edge_order[source_edge_positions[row]]
            if (
                source_edge_ids[source_edge_row] != source_edge_id
                or tuple(int(value) for value in source_edge_pairs[source_edge_row])
                != pair
            ):
                raise ValueError(
                    "Source edge IDs must match their exact endpoint identities."
                )
        elif pair in pair_to_edge_ids:
            raise ValueError("An existing source edge must retain its source edge ID.")
    lifted_source_edges = np.unique(source_edges[source_edges >= 0])
    if lifted_source_edges.size:
        closed_edges = periodic_orbit_closure(
            source, 1, lifted_source_edges
        ).require_source(source, 1)
        if not np.array_equal(np.sort(closed_edges), np.sort(lifted_source_edges)):
            raise ValueError(
                "Executed source-edge splits must include complete edge orbits."
            )
    descriptor = periodic_topology_after_edit(
        source,
        target,
        stencil,
        vertex_orbit_keys=vertex_orbit_keys,
        vertex_windings=vertex_windings,
        source_entity_degree=source_entity_degree,
        exact_coordinate_source=exact_coordinate_source,
    )
    if descriptor is None:
        raise ValueError("A periodic refinement candidate requires a periodic source.")
    entity_ids = {
        degree: np.asarray(target.entity_set(degree).entity_ids, dtype=np.int64)
        for degree in range(target.topological_dimension + 1)
    }
    bound_target = CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=target.numeric_version,
        periodic_topology=descriptor,
    )
    descriptor.require_lifted(
        bound_target.topology,
        bound_target.lifted_topology_id,
        bound_target.blocks,
    )
    descriptor.require_coordinates(bound_target.coordinates)
    if bound_target.lifted_topology_id != target.lifted_topology_id:
        raise ValueError(
            "Periodic rebinding changed the staged lifted topology identity."
        )
    edge_signatures = _periodic_edge_operation_signatures(bound_target, endpoints)
    source_cell_layout = _periodic_cell_layout(source)
    if source_cell_layout is None:
        raise RuntimeError("Periodic candidate source lost its quotient cell layout.")
    parent_rows = cell_order[parent_positions]
    operation_signatures: dict[int, list[tuple[tuple[int, tuple[int, ...]], ...]]] = {
        int(cell_id): [] for cell_id in source_cell_ids
    }
    for parent_id, signature in zip(parents, edge_signatures, strict=True):
        operation_signatures[int(parent_id)].append(signature)
    for orbit in np.unique(source_cell_layout.orbit_rows[parent_rows]):
        member_ids = source_cell_ids[source_cell_layout.orbit_rows == orbit]
        member_operations = [
            tuple(sorted(operation_signatures[int(cell_id)])) for cell_id in member_ids
        ]
        if not member_operations[0] or any(
            operations != member_operations[0] for operations in member_operations[1:]
        ):
            raise ValueError(
                "Every lifted cell orbit must execute identical edge operations."
            )
    candidate = PeriodicCellEditCandidate(source, bound_target, edge_lock.lock_id)
    candidate.require_source(source)
    return candidate


def stage_periodic_topology_edit(
    source: CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    exact_coordinate_source: ExactPlcCellGeometrySource | None = None,
) -> CellMesh | None:
    """Build and validate a periodic target before any topology edit is published.

    Relation signatures are compared over every source entity orbit at each
    dimension. Thus split, collapse, flip, and relocation records either cover
    every lifted image with the same quotient operation or reject the entire
    immutable candidate. ``exact_coordinate_source`` is the target's exact
    quotient-chart vertex bank when generated vertices are not binary64.
    """
    if not isinstance(source, CellMesh):
        raise TypeError("source must be a CellMesh.")
    if source.periodic_topology is None:
        return None
    from ._lineage import VertexInterpolationStencil
    from ._topology_edit import (
        CellTopologyEdit,
        entity_keys,
        key_rows,
        TopologyEditBlock,
    )

    if not isinstance(edit, CellTopologyEdit):
        raise TypeError("edit must be CellTopologyEdit.")

    blocks = []
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock):
            raise ValueError("Periodic topology edits require simplex target blocks.")
        blocks.append(
            CellBlock(
                block.name,
                block.cell_kind,
                block.cells,
                global_ids=block.cell_ids,
            )
        )
    target_kind = {2: "triangle", 3: "tetrahedron"}.get(source.topological_dimension)
    if (
        target_kind is None
        or not blocks
        or any(block.cell_kind != target_kind for block in blocks)
    ):
        raise ValueError(
            "Periodic topology edits require a full-dimensional triangle or tetrahedron target."
        )
    target = CellMesh(
        edit.coordinates,
        blocks,
        vertex_global_ids=edit.vertex_global_ids,
    )
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        edit.vertex_global_ids,
        edit.stencil_sources,
        edit.stencil_weights,
        edit.stencil_valid,
        exact_weights=edit.stencil_exact_weights,
    )
    descriptor = periodic_topology_after_edit(
        source,
        target,
        stencil,
        vertex_orbit_keys=edit.vertex_orbit_keys,
        vertex_windings=edit.vertex_windings,
        source_entity_degree=edit.source_entity_degree,
        exact_coordinate_source=exact_coordinate_source,
    )
    if descriptor is None:
        raise ValueError(
            "Periodic source edits must retain a periodic target descriptor."
        )
    target_entity_ids = {
        degree: np.asarray(target.entity_set(degree).entity_ids, dtype=np.int64)
        for degree in range(target.topological_dimension + 1)
    }
    bound = CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids=target_entity_ids,
        numeric_version=target.numeric_version,
        periodic_topology=descriptor,
    )
    descriptor.require_lifted(bound.topology, bound.lifted_topology_id, bound.blocks)
    descriptor.require_coordinates(bound.coordinates)
    if bound.lifted_topology_id != target.lifted_topology_id:
        raise ValueError("Periodic edit rebinding changed the lifted target identity.")

    if len(edit.relations) != source.topological_dimension + 1:
        raise ValueError(
            "Periodic topology edits need one lineage relation per entity degree."
        )
    for degree, relation in enumerate(edit.relations):
        source_keys = np.asarray(relation.source_keys, dtype=np.int64)
        target_keys = np.asarray(relation.target_keys, dtype=np.int64)
        kinds = np.asarray(relation.kinds, dtype=np.int32)
        if (
            source_keys.ndim != 2
            or target_keys.ndim != 2
            or source_keys.shape[0] != target_keys.shape[0]
            or kinds.shape != (source_keys.shape[0],)
            or int(relation.dimension) != degree
        ):
            raise ValueError("Periodic edit lineage relations are malformed.")
        source_rows = key_rows(entity_keys(source, degree), source_keys)
        target_rows = key_rows(entity_keys(bound, degree), target_keys)
        if np.any(source_rows < 0) or np.any(target_rows < 0):
            raise ValueError(
                "Periodic edit lineage names an absent source or target entity."
            )
        source_layout = _entity_layout(source, degree)
        target_layout = _entity_layout(bound, degree)
        target_quotient_ids = target_layout.entity_ids[
            target_layout.orbit_rows[target_rows]
        ]
        operations: list[list[tuple[int, int]]] = [
            [] for _ in range(source_layout.orbit_rows.size)
        ]
        for source_row, target_id, kind in zip(
            source_rows, target_quotient_ids, kinds, strict=True
        ):
            operations[int(source_row)].append((int(target_id), int(kind)))
        touched_orbits = np.unique(source_layout.orbit_rows[source_rows])
        for orbit in touched_orbits:
            member_rows = np.flatnonzero(source_layout.orbit_rows == orbit)
            signatures = [tuple(sorted(operations[int(row)])) for row in member_rows]
            if not signatures[0] or any(
                signature != signatures[0] for signature in signatures[1:]
            ):
                raise ValueError(
                    "Periodic edit lineage must apply the same operation to every lifted entity orbit."
                )
    return bound


def _periodic_geometry_after_reorder(
    source: CellMesh, target: CellMesh, /
) -> CellGeometrySpec | None:
    topology = source.periodic_topology
    if topology is None:
        raise ValueError("A periodic geometry rebind requires periodic source topology.")
    geometry = topology._actual_geometry
    if geometry is None:
        return None

    elements, routes, coordinates = geometry.resolve(source)
    source_by_name = {
        block.name: (block, element, np.asarray(route, dtype=np.int32))
        for block, element, route in zip(source.blocks, elements, routes, strict=True)
    }
    if set(source_by_name) != {block.name for block in target.blocks}:
        raise ValueError("Periodic geometry blocks changed during canonicalization.")
    target_elements = {}
    target_routes = {}
    for block in target.blocks:
        source_block, element, route = source_by_name[block.name]
        source_ids = np.asarray(source_block.global_ids, dtype=np.int64)
        source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}

        target_ids = np.asarray(block.global_ids, dtype=np.int64)
        if source_rows.keys() != {int(identifier) for identifier in target_ids}:
            raise ValueError(
                "Periodic geometry cell identities changed during canonicalization."
            )
        rows = np.asarray(
            [source_rows[int(identifier)] for identifier in target_ids],
            dtype=np.int64,
        )
        if not np.array_equal(
            np.asarray(source_block.vertices)[rows], np.asarray(block.vertices)
        ):
            raise ValueError(
                "Periodic geometry connectivity changed during canonicalization."
            )
        if source_block.cell_kind != block.cell_kind:
            raise ValueError(
                "Periodic geometry cell kinds changed during canonicalization."
            )
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            element = PerCellRestrictedCellGeometryElement(
                element.source_element,
                element.cell_kind,
                target_ids,
                tuple(element.source_cell_ids[int(row)] for row in rows),
                tuple(element.chart_arguments(int(row)) for row in rows),
            )
        target_elements[block.name] = element
        target_routes[block.name] = route[rows]
    if not np.array_equal(np.asarray(coordinates), np.asarray(target.coordinates)):
        raise ValueError("Periodic geometry coordinates changed during canonicalization.")
    return CellGeometrySpec(target_elements, target_routes, coordinates)


def _periodic_entity_global_ids_after_reorder(
    source: CellMesh,
    target: CellMesh,
    transported: PeriodicMeshTopology,
    anchored: PeriodicMeshTopology,
    /,
) -> dict[int, np.ndarray]:
    """Carry quotient IDs while checking transported and source-gauge orbits."""
    lifted_entity_ids = {
        degree: np.asarray(target.entity_set(degree).entity_ids, dtype=np.int64)
        for degree in range(1, target.topological_dimension)
    }
    transported_mesh = CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids=lifted_entity_ids,
        numeric_version=target.numeric_version,
        periodic_topology=transported,
    )
    anchored_mesh = CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids=lifted_entity_ids,
        numeric_version=target.numeric_version,
        periodic_topology=anchored,
    )
    entity_global_ids = {}
    for degree in range(source.topological_dimension + 1):
        source_layout = _entity_layout(source, degree)
        transported_layout = _entity_layout(transported_mesh, degree)
        anchored_layout = _entity_layout(anchored_mesh, degree)
        source_entity_ids = np.asarray(source_layout.entity_ids, dtype=np.int64)
        source_lifted_ids = np.asarray(
            source.entity_set(degree).entity_ids, dtype=np.int64
        )
        target_lifted_ids = np.asarray(
            target.entity_set(degree).entity_ids, dtype=np.int64
        )
        source_rows_by_id = {
            int(identifier): row for row, identifier in enumerate(source_lifted_ids)
        }
        if source_rows_by_id.keys() != {
            int(identifier) for identifier in target_lifted_ids
        }:
            raise ValueError(
                "Periodic canonicalization must retain every lifted entity identity."
            )
        source_to_transported = np.full(source_entity_ids.shape, -1, dtype=np.int64)
        transported_to_source = np.full(
            transported_layout.entity_ids.shape, -1, dtype=np.int64
        )
        source_to_anchored = np.full(source_entity_ids.shape, -1, dtype=np.int64)
        anchored_to_source = np.full(anchored_layout.entity_ids.shape, -1, dtype=np.int64)
        for target_row, identifier in enumerate(target_lifted_ids):
            source_row = source_rows_by_id[int(identifier)]
            source_orbit = int(source_layout.orbit_rows[source_row])
            transported_orbit = int(transported_layout.orbit_rows[target_row])
            anchored_orbit = int(anchored_layout.orbit_rows[target_row])
            if (
                source_to_transported[source_orbit] not in (-1, transported_orbit)
                or transported_to_source[transported_orbit] not in (-1, source_orbit)
                or source_to_anchored[source_orbit] not in (-1, anchored_orbit)
                or anchored_to_source[anchored_orbit] not in (-1, source_orbit)
            ):
                raise ValueError(
                    "Periodic canonicalization changed a lifted entity orbit."
                )
            source_to_transported[source_orbit] = transported_orbit
            transported_to_source[transported_orbit] = source_orbit
            source_to_anchored[source_orbit] = anchored_orbit
            anchored_to_source[anchored_orbit] = source_orbit
        if (
            np.any(source_to_transported < 0)
            or np.any(transported_to_source < 0)
            or np.any(source_to_anchored < 0)
            or np.any(anchored_to_source < 0)
        ):
            raise ValueError(
                "Periodic canonicalization changed the quotient entity partition."
            )
        entity_global_ids[degree] = source_entity_ids[anchored_to_source]
    return entity_global_ids


def _periodic_topology_after_reorder(
    source: CellMesh, target: CellMesh, /
) -> PeriodicMeshTopology:
    """Rebind quotient tables while retaining source vertex roots and winding."""
    if not isinstance(source, CellMesh) or not isinstance(target, CellMesh):
        raise TypeError(
            "Periodic canonicalization requires source and target CellMesh values."
        )
    source_topology = source.periodic_topology
    if source_topology is None:
        raise ValueError("Periodic canonicalization requires periodic source topology.")
    if target.periodic_topology is not None:
        raise ValueError(
            "Periodic canonicalization target must be an unbound lifted mesh."
        )
    source_topology.require_lifted(source.topology, source.topology_id, source.blocks)
    source_topology.require_coordinates(source.coordinates)

    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    source_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(source_vertex_ids)
    }
    target_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(target_vertex_ids)
    }
    if source_vertex_rows.keys() != target_vertex_rows.keys():
        raise ValueError(
            "Periodic canonicalization must retain every lifted vertex identity."
        )
    source_rows = np.asarray(
        [source_vertex_rows[int(identifier)] for identifier in target_vertex_ids],
        dtype=np.int64,
    )
    if not np.array_equal(
        np.asarray(source.coordinates)[source_rows], np.asarray(target.coordinates)
    ):
        raise ValueError("Periodic canonicalization cannot change vertex coordinates.")

    source_representatives = np.asarray(
        source_topology.vertex_representatives, dtype=np.int64
    )
    source_shifts = np.asarray(source_topology.vertex_shifts, dtype=np.int32)
    source_root_ids = source_vertex_ids[source_representatives]
    target_representatives = np.asarray(
        [target_vertex_rows[int(source_root_ids[row])] for row in source_rows],
        dtype=np.int32,
    )
    target_shifts = source_shifts[source_rows]
    if not np.array_equal(
        target_vertex_ids[target_representatives], source_root_ids[source_rows]
    ):
        raise ValueError("Periodic canonicalization changed source vertex occurrence.")

    from ._lineage import VertexInterpolationStencil

    stencil_sources = target_vertex_ids[:, None]
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        target_vertex_ids,
        stencil_sources,
        np.ones(stencil_sources.shape, dtype=np.float64),
        np.ones(stencil_sources.shape, dtype=np.bool_),
    )
    transported = periodic_topology_after_edit(source, target, stencil)
    if transported is None:
        raise ValueError("Periodic canonicalization lost its periodic topology owner.")
    anchored = PeriodicMeshTopology(
        target,
        source_topology.cell,
        target_representatives,
        target_shifts,
    )
    entity_global_ids = _periodic_entity_global_ids_after_reorder(
        source, target, transported, anchored
    )
    allocator = {
        degree: next_id
        for degree, next_id in enumerate(source_topology.allocator_next_ids)
        if next_id >= 0
    }
    rebound = PeriodicMeshTopology(
        target,
        source_topology.cell,
        target_representatives,
        target_shifts,
        entity_global_ids=entity_global_ids,
        entity_allocator_next_ids=allocator or None,
        actual_geometry=_periodic_geometry_after_reorder(source, target),
    )
    if (
        not np.array_equal(
            np.asarray(source_topology.entity_key_offsets),
            np.asarray(rebound.entity_key_offsets),
        )
        or not np.array_equal(
            np.asarray(source_topology.entity_key_values),
            np.asarray(rebound.entity_key_values),
        )
        or source_topology.quotient.topology_id != rebound.quotient.topology_id
        or identification_id(source_topology.cell) != identification_id(rebound.cell)
    ):
        raise ValueError(
            "Periodic canonicalization changed lattice, quotient IDs, or incidence."
        )
    return rebound


@checked
def periodic_delaunay_mesh(
    source: PeriodicPointOrbits,
    /,
    *,
    initial_margin: float = 1.0,
) -> PeriodicMeshConstruction:
    """Construct a real periodic quotient Delaunay mesh with the native kernel."""
    if np.asarray(source.point_ids).size == 0:
        raise ValueError("Global periodic Delaunay requires at least one authored point.")
    dimension = source.ambient_dimension
    margin = float(initial_margin)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("initial_margin must be finite and positive.")
    point_ids = np.asarray(source.point_ids, dtype=np.int64)
    point_order = np.argsort(point_ids, kind="stable")
    point_ids = point_ids[point_order]
    points = np.ascontiguousarray(
        np.asarray(source.points, dtype=np.float64)[point_order]
    )
    fractional = np.ascontiguousarray(
        np.asarray(source.fractional_points, dtype=np.float64)[point_order]
    )
    cell = source.cell
    lattice = np.ascontiguousarray(np.asarray(cell.vectors, dtype=np.float64))
    inverse = np.ascontiguousarray(np.asarray(cell.inverse_vectors, dtype=np.float64))
    from .._meshcore import periodic_delaunay

    triangulation = periodic_delaunay(
        points, fractional, lattice, inverse, initial_margin=margin
    )
    cells = np.asarray(triangulation.cells)
    shifts = np.asarray(triangulation.shifts)
    if cells.dtype.kind not in "iu":
        raise TypeError("Native periodic Delaunay cells must be integer vertex rows.")
    if cells.ndim == 1:
        if cells.size % (dimension + 1):
            raise ValueError("Native periodic Delaunay returned incomplete simplex rows.")
        cells = cells.reshape((-1, dimension + 1))
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] != dimension + 1:
        raise ValueError("Native periodic Delaunay returned an invalid simplex table.")
    if shifts.dtype.kind not in "iu":
        raise TypeError("Native periodic Delaunay shifts must be integers.")
    if shifts.ndim == 1:
        if shifts.size != cells.shape[0] * (dimension + 1) * dimension:
            raise ValueError("Native periodic Delaunay returned incomplete winding rows.")
        shifts = shifts.reshape((cells.shape[0], dimension + 1, dimension))
    elif shifts.ndim == 2:
        if shifts.shape != (cells.shape[0], (dimension + 1) * dimension):
            raise ValueError("Native periodic Delaunay returned invalid winding rows.")
        shifts = shifts.reshape((cells.shape[0], dimension + 1, dimension))
    elif shifts.shape != (cells.shape[0], dimension + 1, dimension):
        raise ValueError("Native periodic Delaunay returned invalid winding rows.")
    cells = cells.astype(np.int64, copy=False)
    shifts = shifts.astype(np.int64, copy=False)
    if np.any((cells < 0) | (cells >= point_ids.size)):
        raise ValueError("Native periodic Delaunay references an absent source orbit.")
    if not np.array_equal(np.unique(cells), np.arange(point_ids.size, dtype=np.int64)):
        raise ValueError(
            "Native periodic Delaunay did not retain every authored point orbit."
        )
    base_shifts = -np.floor(fractional).astype(np.int64)
    if np.any(np.abs(shifts) > np.iinfo(np.int32).max):
        raise OverflowError(
            "Native periodic Delaunay winding exceeds the mesh topology index range."
        )
    image_keys = {
        (int(point_ids[int(site)]), tuple(int(value) for value in shift))
        for site, shift in zip(
            cells.reshape(-1), shifts.reshape((-1, dimension)), strict=True
        )
    }
    keys_by_point: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for key in image_keys:
        keys_by_point.setdefault(key[0], []).append(key)
    representative_key: dict[int, tuple[int, tuple[int, ...]]] = {}
    point_row_by_id = {int(identifier): row for row, identifier in enumerate(point_ids)}
    for identifier, keys in keys_by_point.items():
        row = point_row_by_id[identifier]
        base_key = (identifier, tuple(int(value) for value in base_shifts[row]))
        representative_key[identifier] = base_key if base_key in image_keys else min(keys)
    ordered_keys = tuple(sorted(image_keys))
    representative_keys = set(representative_key.values())
    maximum_id = int(np.max(point_ids, initial=-1))
    generated_count = len(image_keys) - len(representative_keys)
    index_limit = int(np.iinfo(np.int64).max)
    if maximum_id > index_limit - generated_count - cells.shape[0] - 1:
        raise OverflowError(
            "Periodic semantic identities exceed the int64 mesh ID range."
        )
    vertex_ids = np.empty((len(ordered_keys),), dtype=np.int64)
    vertex_rows = {key: row for row, key in enumerate(ordered_keys)}
    root_rows = np.empty((len(ordered_keys),), dtype=np.int32)
    vertex_shifts = np.empty((len(ordered_keys), dimension), dtype=np.int64)
    root_windings = np.empty((len(ordered_keys), dimension), dtype=np.int64)
    source_rows = np.empty((len(ordered_keys),), dtype=np.int64)
    next_vertex_id = maximum_id + 1
    for row, key in enumerate(ordered_keys):
        identifier, winding = key
        root_key = representative_key[identifier]
        root_row = vertex_rows[root_key]
        root_rows[row] = root_row
        vertex_shifts[row] = np.asarray(winding, dtype=np.int64) - np.asarray(
            root_key[1], dtype=np.int64
        )
        vertex_ids[row] = identifier if key == root_key else next_vertex_id
        if key != root_key:
            next_vertex_id += 1
        source_rows[row] = point_row_by_id[identifier]
        root_windings[row] = np.asarray(root_key[1], dtype=np.int64)
    # One canonical lift owner: each root is the exact image of its authored
    # point, and every other image is the exact image of its root, so the
    # topology's orbit witness reproduces coordinates bit for bit.
    coordinates = apply_periodic(
        cell,
        apply_periodic(cell, points[source_rows], root_windings),
        vertex_shifts,
    )
    if np.any(np.abs(vertex_shifts) > np.iinfo(np.int32).max):
        raise OverflowError(
            "Relative periodic winding exceeds the mesh topology index range."
        )
    lifted_cells = np.empty(cells.shape, dtype=np.int32)
    for cell_row in range(cells.shape[0]):
        for local_vertex in range(dimension + 1):
            site = int(cells[cell_row, local_vertex])
            key = (
                int(point_ids[site]),
                tuple(int(value) for value in shifts[cell_row, local_vertex]),
            )
            lifted_cells[cell_row, local_vertex] = vertex_rows[key]
    first_cell_id = int(np.max(vertex_ids, initial=-1)) + 1
    if first_cell_id > index_limit - cells.shape[0]:
        raise OverflowError("Periodic cell identities exceed the int64 mesh ID range.")
    cell_ids = np.arange(first_cell_id, first_cell_id + cells.shape[0], dtype=np.int64)
    cell_kind = "triangle" if dimension == 2 else "tetrahedron"
    block = CellBlock("periodic", cell_kind, lifted_cells, global_ids=cell_ids)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(lifted, cell, root_rows, vertex_shifts)
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )
    return PeriodicMeshConstruction(mesh)


__all__ = [
    "PeriodicMeshConstruction",
    "PeriodicCellEditCandidate",
    "PeriodicOrbitClosure",
    "PeriodicEntityOrbitMap",
    "PeriodicPointOrbits",
    "PeriodicRefinementEdgeLock",
    "periodic_delaunay_mesh",
    "periodic_orbit_closure",
    "periodic_entity_orbit_map",
    "transfer_periodic_p1_vertices",
    "prepare_periodic_refinement_edge_lock",
    "stage_periodic_refinement_candidate",
    "stage_periodic_topology_edit",
    "periodic_topology_after_edit",
]
