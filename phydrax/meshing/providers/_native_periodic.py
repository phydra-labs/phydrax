#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from math import prod
from numbers import Integral
from typing import (
    Any,
    cast,
    final,
    NamedTuple,
    overload,
    TYPE_CHECKING,
    TypeAlias,
    TypedDict,
)

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._geometry_predicates import PredicateMode
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_complex import TetrahedralConnectivity
from ...discretization._cell_geometry import CellGeometrySpec
from ...discretization._cell_mesh import CellMesh, SimplicialConnectivity
from ...discretization._periodic_topology import PeriodicMeshTopology
from ...meshing._certification_schedule import MeshCertificationSchedule
from ...meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from ...meshing._controls import FeatureKind, RegionControl
from ...meshing._local_metric import execute_local_metric_adaptation
from ...meshing._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from ...meshing._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshZone,
    MeshZoneRole,
)
from ...meshing._plc_source import _numeric_entity_ids
from ...meshing._result import MeshingComplianceReport
from ...meshing._scope import MeshingEntityKind, MeshingScope
from ...meshing._sizing import UniformSizeControl
from ...meshing._tetra_metric import execute_tetra_metric_adaptation
from ...meshing._topology_edit import CellTopologyEdit
from ...typing import checked
from .._canonical import canonicalize_cell_mesh, certify_cell_mesh
from .._distribution import _owner_local_multiset_identity
from .._periodic import (
    periodic_delaunay_mesh,
    periodic_entity_orbit_map,
    PeriodicPointOrbits,
    stage_periodic_topology_edit,
)
from ._native_options import NativeMeshingOptions
from ._native_periodic_constraints import (
    validate_periodic_mesh_constraints,
    validate_periodic_source_controls,
)
from ._native_periodic_power import periodic_power_mesh
from ._native_periodic_size import (
    periodic_metric_from_uniform_controls,
    periodic_size_compliance,
)


if TYPE_CHECKING:
    from ..._execution_runtime import ExecutionGroup
    from ...discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ...discretization._periodic_cell import PeriodicCell
    from ...discretization._view_support import OwnerLocalCellGeometry
    from ...graph._distributed_partition import DistributedGraphPartitionResult
    from ...graph._partition import GraphPartitionPlan
    from ...lifecycle import Composition, CompositionEntry
    from .._assembly import MeshAssembly
    from .._device_adaptation import (
        OwnerLocalAdaptivePartTransition,
        OwnerLocalPeriodicAdaptiveExecution,
        OwnerLocalPeriodicAdaptiveResult,
    )
    from .._distribution import (
        OwnerLocalGraphMigrationEvidence,
        OwnerLocalGraphPartitionEvidence,
        ShardedOwnerDirectoryWitness,
    )
    from .._lineage import VertexInterpolationStencil
    from .._periodic import OwnerLocalPeriodicCandidate

_OwnerLocalPeriodicFaceKey: TypeAlias = tuple[tuple[int, tuple[int, ...]], ...]
_OwnerLocalPeriodicFaceRow: TypeAlias = tuple[tuple[_OwnerLocalPeriodicFaceKey, int], ...]
_OwnerLocalPeriodicFaceRows: TypeAlias = tuple[_OwnerLocalPeriodicFaceRow, ...]


class _PeriodicGraphVertexCoordinateAncestryPayload(TypedDict):
    source_entity_set_id: str
    preserves_constants: bool
    source_ids: list[int]
    weight_bits: list[int]
    exact_weights: list[list[int]]
    exact_offsets: list[list[int]]


class _PeriodicGraphVertexPayload(TypedDict):
    vertex_id: int
    coordinate_bits: list[int]
    exact_coordinates: list[list[int]]
    source_id: int
    owner_rank: int
    absolute_winding: list[int]
    representative_id: int
    quotient_winding: list[int]
    coordinate_ancestry: _PeriodicGraphVertexCoordinateAncestryPayload | None


class _PeriodicGraphCellPayload(TypedDict):
    cell_id: int
    source_owner: int
    target_owner: int
    vertices: list[_PeriodicGraphVertexPayload]
    cell_vertex_shifts: list[list[int]]
    key_source_ids: list[int]
    key_relative_windings: list[list[int]]
    cell_winding: list[int]
    measure: list[int]
    support_lower_bits: list[int]
    support_upper_bits: list[int]
    neighbors: list[int]
    neighbor_owners: list[int]
    neighbor_weights: list[int]


class _PeriodicGraphSourcePointPayload(TypedDict):
    point_id: int
    owner_rank: int
    coordinate_bits: list[int]


class _PeriodicGraphMigrationPayload(TypedDict):
    cells: list[_PeriodicGraphCellPayload]
    source_points: list[_PeriodicGraphSourcePointPayload]


class _PeriodicHaloCellPayload(TypedDict):
    cell: _PeriodicGraphCellPayload
    source_points: list[_PeriodicGraphSourcePointPayload]


class _OwnerLocalPeriodicMeshAuthority(NamedTuple):
    face_coverage_id: str
    local_cell_faces: _OwnerLocalPeriodicFaceRows
    owned_cell_rows: np.ndarray
    topology_id: str
    geometry_id: str
    periodic_topology_id: str
    part_id: str
    owner_directory: ShardedOwnerDirectoryWitness
    exact_volume: Fraction
    expected_volume: Fraction


class _OwnerLocalPeriodicSuccessorAuthority(NamedTuple):
    source_entry: CompositionEntry
    source_composition: Composition
    mesh_authority: _OwnerLocalPeriodicMeshAuthority
    part_authority: CompositionEntry
    bound_witness: ShardedOwnerDirectoryWitness
    accepted_composition: Composition


@final
class NativePeriodicSource(StrictModule, NonTrainableState):
    """Bind periodic sites or a represented quotient mesh to owned source scopes."""

    domain: PeriodicPointOrbits | CellMesh
    point_owner_ranks: np.ndarray | None
    site_weights: np.ndarray | None
    cell_regions: np.ndarray | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = (
        eqx.field(static=True)
    )
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: PeriodicPointOrbits | CellMesh,
        source_id: str,
        source_revision: str,
        /,
        *,
        cell_regions: ArrayLike | None = None,
        site_weights: ArrayLike | None = None,
        point_owner_ranks: ArrayLike | None = None,
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        if not source or not revision:
            raise ValueError(
                "source_id and source_revision must be non-empty physical source identities."
            )
        dimension_banks: list[np.ndarray] = [
            np.empty((0,), dtype=np.int64) for _ in range(4)
        ]
        key_banks: list[tuple[tuple[str, tuple[str, ...]], ...]] = [() for _ in range(4)]
        set_ids: list[str] = ["" for _ in range(4)]
        point_owners = None
        match domain:
            case PeriodicPointOrbits():
                dimension = domain.cell.ambient_dimension
                weights = None
                if site_weights is not None:
                    raw_weights = np.asarray(site_weights)
                    if (
                        raw_weights.shape != (np.asarray(domain.point_ids).size,)
                        or raw_weights.dtype.kind not in "iuf"
                    ):
                        raise ValueError(
                            "site_weights must be one signed finite real per authored periodic point ID."
                        )
                    weights = np.array(raw_weights, dtype=np.float64, copy=True)
                    if not np.all(np.isfinite(weights)):
                        raise ValueError("site_weights must be finite.")
                    weights.setflags(write=False)
                if point_owner_ranks is not None:
                    raw_owners = np.asarray(point_owner_ranks)
                    if (
                        raw_owners.shape != np.asarray(domain.point_ids).shape
                        or raw_owners.dtype.kind not in "iu"
                        or np.any(raw_owners < 0)
                        or np.any(raw_owners > np.iinfo(np.int32).max)
                    ):
                        raise ValueError(
                            "point_owner_ranks must be one nonnegative group rank per local authored site."
                        )
                    point_owners = np.array(raw_owners, dtype=np.int32, copy=True)
                    point_owners.setflags(write=False)
                if cell_regions is not None:
                    raise ValueError(
                        "Periodic point orbits have no authored material-region table."
                    )
                regions = None
                domain_id = domain.orbits_id
                point_ids = np.sort(np.asarray(domain.point_ids, dtype=np.int64))
                dimension_banks[0] = np.array(point_ids, copy=True)
                key_banks[0] = tuple(
                    (
                        f"periodic-point:{domain.cell.cell_id}:{int(point_id)}",
                        (),
                    )
                    for point_id in point_ids.tolist()
                )
                set_ids[0] = domain.orbits_id
                fundamental_id = _numeric_entity_ids(
                    "periodic-fundamental-domain",
                    (domain.cell.cell_id,),
                    ((),),
                )
                dimension_banks[dimension] = np.asarray(fundamental_id, dtype=np.int64)
                key_banks[dimension] = ((domain.cell.cell_id, ()),)
                set_ids[dimension] = domain.cell.cell_id
            case CellMesh():
                if site_weights is not None:
                    raise ValueError(
                        "site_weights require an authored PeriodicPointOrbits source, not a represented CellMesh."
                    )
                if point_owner_ranks is not None:
                    raise ValueError(
                        "point_owner_ranks apply only to owner-local periodic point sources."
                    )
                weights = None
                periodic = domain.periodic_topology
                if not isinstance(periodic, PeriodicMeshTopology):
                    raise ValueError(
                        "A represented periodic source must carry PeriodicMeshTopology."
                    )
                periodic.require_lifted(
                    domain.topology, domain.topology_id, domain.blocks
                )
                periodic.require_coordinates(domain.coordinates)
                dimension = domain.ambient_dimension
                count = sum(block.cell_count for block in domain.blocks)
                if cell_regions is None:
                    regions = None
                else:
                    raw_regions = np.asarray(cell_regions)
                    if (
                        raw_regions.shape != (count,)
                        or raw_regions.dtype.kind not in "iu"
                    ):
                        raise ValueError(
                            "cell_regions must be one integer material ID per represented cell."
                        )
                    if raw_regions.size and (
                        np.any(raw_regions < 0)
                        or np.any(raw_regions > np.iinfo(np.int32).max)
                    ):
                        raise ValueError(
                            "cell_regions must fit nonnegative int32 material IDs."
                        )
                    regions = np.array(raw_regions, dtype=np.int32, copy=True)
                domain_id = canonical_fingerprint(
                    {
                        "kind": "represented-periodic-mesh-source",
                        "mesh_id": domain.mesh_id,
                        "numeric_version": domain.numeric_version,
                        "mesh_topology": domain.topology_id,
                        "coordinates": array_tree_fingerprint(domain.coordinates),
                        "periodic_topology": periodic.periodic_topology_id,
                    }
                )
                for entity_dimension in range(4):
                    if entity_dimension <= domain.topological_dimension:
                        entity_set = domain.entity_set(entity_dimension)
                        identifiers = np.sort(
                            np.asarray(entity_set.entity_ids, dtype=np.int64)
                        )
                        dimension_banks[entity_dimension] = np.array(
                            identifiers, copy=True
                        )
                        key_banks[entity_dimension] = tuple(
                            (
                                f"mesh:{domain.mesh_id}:d{entity_dimension}:id:{int(identifier)}",
                                (),
                            )
                            for identifier in identifiers.tolist()
                        )
                        set_ids[entity_dimension] = entity_set.entity_set_id
                    else:
                        set_ids[entity_dimension] = canonical_fingerprint(
                            {
                                "kind": "empty-periodic-mesh-source-scope",
                                "mesh_id": domain.mesh_id,
                                "numeric_version": domain.numeric_version,
                                "dimension": entity_dimension,
                            }
                        )
            case _:
                raise TypeError("domain must be PeriodicPointOrbits or CellMesh.")

        for entity_dimension in range(4):
            identifiers = np.asarray(dimension_banks[entity_dimension], dtype=np.int64)
            if identifiers.ndim != 1 or np.any(identifiers < 0):
                raise ValueError(
                    "Periodic source scope IDs must be nonnegative integer vectors."
                )
            if np.unique(identifiers).size != identifiers.size:
                raise ValueError(
                    "Periodic source scope IDs must be unique within each dimension."
                )
            if len(key_banks[entity_dimension]) != identifiers.size:
                raise ValueError(
                    "Periodic source key banks must align with their scope IDs."
                )
            identifiers.setflags(write=False)
            dimension_banks[entity_dimension] = identifiers
            if not set_ids[entity_dimension]:
                set_ids[entity_dimension] = canonical_fingerprint(
                    {
                        "kind": "empty-periodic-source-scope",
                        "source": (source, revision),
                        "domain": domain_id,
                        "dimension": entity_dimension,
                    }
                )

        if regions is not None:
            regions = np.ascontiguousarray(regions, dtype=np.int32)
            regions.flags.writeable = False
        self.domain = domain
        self.site_weights = weights
        self.point_owner_ranks = point_owners
        self.cell_regions = regions
        self.source_id = source
        self.source_revision = revision
        self.ambient_dimension = dimension
        self.scope_entity_id_banks = tuple(dimension_banks)
        self.scope_entity_key_banks = tuple(key_banks)
        self.scope_entity_set_ids = tuple(set_ids)
        site_weight_identity = None
        point_owner_identity = None
        if isinstance(domain, PeriodicPointOrbits):
            site_order = np.argsort(
                np.asarray(domain.point_ids, dtype=np.int64), kind="stable"
            )
            site_weight_identity = (
                None if weights is None else array_tree_fingerprint(weights[site_order])
            )
            point_owner_identity = (
                None
                if point_owners is None
                else array_tree_fingerprint(point_owners[site_order])
            )
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-periodic-source",
                "source_id": source,
                "source_revision": revision,
                "domain": domain_id,
                "scope_ids": tuple(
                    array_tree_fingerprint(values)
                    for values in self.scope_entity_id_banks
                ),
                "scope_keys": self.scope_entity_key_banks,
                "scope_sets": self.scope_entity_set_ids,
                "cell_regions": None
                if regions is None
                else array_tree_fingerprint(regions),
                "site_weights": site_weight_identity,
                "point_owner_ranks": point_owner_identity,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be one integer source dimension.")
        index = int(dimension)
        if not 0 <= index < 4:
            raise ValueError("Periodic source dimensions must lie in 0..3.")
        return self.scope_entity_id_banks[index]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be one integer source dimension.")
        index = int(dimension)
        if not 0 <= index < 4:
            raise ValueError("Periodic source dimensions must lie in 0..3.")
        return self.scope_entity_set_ids[index]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be one integer source dimension.")
        index = int(dimension)
        if not 0 <= index < 4:
            raise ValueError("Periodic source dimensions must lie in 0..3.")
        if isinstance(entity_id, (bool, np.bool_)) or not isinstance(entity_id, Integral):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[index]
        position = int(np.searchsorted(identifiers, int(entity_id)))
        if position >= identifiers.size or int(identifiers[position]) != int(entity_id):
            raise ValueError("entity_id is absent from this periodic source dimension.")
        return self.scope_entity_key_banks[index][position]

    def validate_restored(self, /) -> None:
        rebuilt = NativePeriodicSource(
            self.domain,
            self.source_id,
            self.source_revision,
            cell_regions=(
                None
                if isinstance(self.domain, PeriodicPointOrbits)
                else self.cell_regions
            ),
            site_weights=(
                self.site_weights
                if isinstance(self.domain, PeriodicPointOrbits)
                else None
            ),
            point_owner_ranks=(
                self.point_owner_ranks
                if isinstance(self.domain, PeriodicPointOrbits)
                else None
            ),
        )
        if (
            rebuilt.binding_id != self.binding_id
            or (rebuilt.cell_regions is None) != (self.cell_regions is None)
            or (
                rebuilt.cell_regions is not None
                and self.cell_regions is not None
                and not np.array_equal(rebuilt.cell_regions, self.cell_regions)
            )
            or (rebuilt.site_weights is None) != (self.site_weights is None)
            or (
                rebuilt.site_weights is not None
                and self.site_weights is not None
                and not np.array_equal(rebuilt.site_weights, self.site_weights)
            )
        ):
            raise ValueError("Stored native periodic source binding changed.")

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativePeriodicSource],
        value: NativePeriodicSource,
        /,
    ) -> NativePeriodicSource:
        """Rebuild periodic source scopes and validate their immutable identity."""
        if type(value) is not cls:
            raise TypeError("value must be NativePeriodicSource.")
        if isinstance(value.domain, PeriodicPointOrbits):
            domain = PeriodicPointOrbits.from_stored_fields(value.domain)
            renewed = cls(
                domain,
                value.source_id,
                value.source_revision,
                site_weights=value.site_weights,
                point_owner_ranks=value.point_owner_ranks,
            )
        elif isinstance(value.domain, CellMesh):
            renewed = cls(
                value.domain,
                value.source_id,
                value.source_revision,
                cell_regions=value.cell_regions,
            )
        else:
            raise TypeError("Stored native periodic source has an unsupported domain.")
        if renewed.binding_id != value.binding_id:
            raise ValueError(
                "Rebuilt native periodic source changed its binding identity."
            )
        renewed.validate_restored()
        return renewed


def _cell_regions_in_entity_order(mesh: CellMesh, values: np.ndarray, /) -> np.ndarray:
    entity_ids = np.asarray(
        mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
    )
    block_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    regions = np.asarray(values)
    if (
        regions.ndim != 1
        or regions.dtype.kind not in "iu"
        or regions.shape != block_ids.shape
        or block_ids.shape != entity_ids.shape
        or np.unique(block_ids).size != block_ids.size
        or not np.array_equal(np.sort(block_ids), np.sort(entity_ids))
    ):
        raise ValueError(
            "Periodic cell-region values must bind every lifted cell ID exactly once."
        )
    by_id = {
        int(identifier): int(region)
        for identifier, region in zip(block_ids, regions, strict=True)
    }
    return np.asarray(
        [by_id[int(identifier)] for identifier in entity_ids], dtype=np.int64
    )


def _periodic_quotient_entity_set_id(mesh: CellMesh, degree: int, /) -> str:
    if mesh.periodic_topology is None:
        raise ValueError("Periodic quotient scopes require attached periodic topology.")
    return mesh.periodic_topology.quotient.entity_sets[degree].entity_set_id


def _periodic_quotient_entity_ids(mesh: CellMesh, degree: int, /) -> np.ndarray:
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic quotient scopes require attached periodic topology.")
    return np.asarray(topology.quotient.entity_sets[degree].entity_ids, dtype=np.int64)


def _lifted_rows_for_quotient_ids(
    mesh: CellMesh, degree: int, entity_ids: np.ndarray, /
) -> np.ndarray:
    orbit_map = periodic_entity_orbit_map(mesh, degree)
    orbit_map.require_source(mesh)
    quotient_ids = np.asarray(orbit_map.quotient_entity_ids, dtype=np.int64)
    selected_ids = np.asarray(entity_ids, dtype=np.int64)
    if (
        selected_ids.ndim != 1
        or selected_ids.size == 0
        or np.unique(selected_ids).size != selected_ids.size
    ):
        raise ValueError(
            "Periodic quotient controls must select unique quotient entity IDs."
        )
    row_by_id = {int(identifier): row for row, identifier in enumerate(quotient_ids)}
    if any(int(identifier) not in row_by_id for identifier in selected_ids):
        raise ValueError(
            "Periodic quotient control references an absent quotient entity ID."
        )
    selected_rows = np.asarray(
        [row_by_id[int(identifier)] for identifier in selected_ids], dtype=np.int64
    )
    lifted_rows = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    if lifted_rows.size != mesh.entity_set(degree).entity_ids.size:
        raise ValueError("Periodic quotient map does not span the lifted entity bank.")
    return np.flatnonzero(np.isin(lifted_rows, selected_rows))


def _quotient_ids_for_lifted_ids(
    mesh: CellMesh, degree: int, entity_ids: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    orbit_map = periodic_entity_orbit_map(mesh, degree)
    orbit_map.require_source(mesh)
    lifted_ids = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
    requested_ids = np.asarray(entity_ids, dtype=np.int64)
    if (
        requested_ids.ndim != 1
        or requested_ids.size == 0
        or np.unique(requested_ids).size != requested_ids.size
    ):
        raise ValueError("Periodic lifted controls must select unique lifted entity IDs.")
    row_by_id = {int(identifier): row for row, identifier in enumerate(lifted_ids)}
    if any(int(identifier) not in row_by_id for identifier in requested_ids):
        raise ValueError("Periodic lifted control references an absent lifted entity ID.")
    requested_rows = np.asarray(
        [row_by_id[int(identifier)] for identifier in requested_ids], dtype=np.int64
    )
    quotient_ids = np.asarray(orbit_map.quotient_entity_ids, dtype=np.int64)
    lifted_to_quotient = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    selected_quotient_ids = np.unique(quotient_ids[lifted_to_quotient[requested_rows]])
    complete_rows = _lifted_rows_for_quotient_ids(mesh, degree, selected_quotient_ids)
    return selected_quotient_ids, complete_rows


def _target_cell_regions(
    source: CellMesh, target: CellMesh, edit: Any, source_values: np.ndarray, /
) -> np.ndarray:
    degree = source.topological_dimension
    source_ids = np.asarray(source.entity_set(degree).entity_ids, dtype=np.int64)
    target_ids = np.asarray(target.entity_set(degree).entity_ids, dtype=np.int64)
    source_regions = _cell_regions_in_entity_order(source, source_values)
    by_source_id = {
        int(identifier): int(region)
        for identifier, region in zip(source_ids, source_regions, strict=True)
    }
    inherited: dict[int, set[int]] = {}
    for identifier in target_ids:
        value = by_source_id.get(int(identifier))
        if value is not None:
            inherited.setdefault(int(identifier), set()).add(value)
    relation = edit.relations[degree]
    source_keys = np.asarray(relation.source_keys, dtype=np.int64).reshape(-1)
    target_keys = np.asarray(relation.target_keys, dtype=np.int64).reshape(-1)
    for source_id, target_id in zip(source_keys, target_keys, strict=True):
        if int(source_id) not in by_source_id:
            raise ValueError(
                "Periodic material lineage references an absent source cell."
            )
        inherited.setdefault(int(target_id), set()).add(by_source_id[int(source_id)])
    result = np.empty((target_ids.size,), dtype=np.int64)
    for row, identifier in enumerate(target_ids):
        regions = inherited.get(int(identifier), set())
        if len(regions) != 1:
            raise ValueError(
                "Periodic material lineage must preserve one region per target cell."
            )
        result[row] = next(iter(regions))
    target_map = periodic_entity_orbit_map(target, degree)
    target_map.require_source(target)
    for quotient_row in range(target_map.quotient_entity_ids.size):
        members = np.flatnonzero(
            np.asarray(target_map.lifted_to_quotient, dtype=np.int64) == quotient_row
        )
        if np.unique(result[members]).size != 1:
            raise ValueError(
                "Periodic target cell material labels must agree over every orbit."
            )
    return result


def _periodic_target_size_scopes(
    source: CellMesh,
    target: CellMesh,
    edit: Any,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> tuple[np.ndarray, ...]:
    source_map = periodic_entity_orbit_map(source, 0)
    target_map = periodic_entity_orbit_map(target, 0)
    source_map.require_source(source)
    target_map.require_source(target)
    source_ids = np.asarray(source_map.lifted_entity_ids, dtype=np.int64)
    target_ids = np.asarray(target_map.lifted_entity_ids, dtype=np.int64)
    stencil_ids = np.asarray(edit.vertex_global_ids, dtype=np.int64)
    sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    if (
        not np.array_equal(stencil_ids, target_ids)
        or sources.shape != valid.shape
        or sources.shape[0] != target_ids.size
    ):
        raise ValueError(
            "Periodic metric edit stencil does not match its target vertex bank."
        )
    source_qids = np.asarray(source_map.quotient_entity_ids, dtype=np.int64)[
        np.asarray(source_map.lifted_to_quotient, dtype=np.int64)
    ]
    source_q_by_id = {
        int(identifier): int(quotient_id)
        for identifier, quotient_id in zip(source_ids, source_qids, strict=True)
    }
    target_qids = np.asarray(target_map.quotient_entity_ids, dtype=np.int64)[
        np.asarray(target_map.lifted_to_quotient, dtype=np.int64)
    ]
    result: list[np.ndarray] = []
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if len(controls) != len(specification.size_controls):
        raise ValueError(
            "Periodic metric adaptation supports only uniform size controls."
        )
    for control in controls:
        selected = set(map(int, np.asarray(control.scope.entity_ids, dtype=np.int64)))
        membership: dict[int, set[bool]] = {}
        for row, quotient_id in enumerate(target_qids):
            support = sources[row, valid[row]]
            if support.size == 0:
                raise ValueError(
                    "Every periodic target vertex requires source interpolation ancestry."
                )
            try:
                source_orbits = {
                    source_q_by_id[int(identifier)] for identifier in support
                }
            except KeyError as error:
                raise ValueError(
                    "Periodic target stencil references a non-source vertex ID."
                ) from error
            membership.setdefault(int(quotient_id), set()).add(source_orbits <= selected)
        if any(len(values) != 1 for values in membership.values()):
            raise ValueError(
                "Periodic size-control ancestry is not equivariant over target vertex orbits."
            )
        result.append(
            np.asarray(
                sorted(
                    identifier
                    for identifier, values in membership.items()
                    if True in values
                ),
                dtype=np.int64,
            )
        )
    return tuple(result)


def _periodic_metric_controls(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    mesh: CellMesh,
    source_regions: np.ndarray | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, RegionControl]]:
    dimension = mesh.topological_dimension
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_classes = (
        np.zeros((cell_ids.size,), dtype=np.int64)
        if source_regions is None
        else _cell_regions_in_entity_order(mesh, source_regions)
    )
    edge_classes = np.zeros((edge_ids.size,), dtype=np.int64)
    protected_edges = np.zeros((edge_ids.size,), dtype=np.bool_)
    fixed_vertices = np.full(
        (vertex_ids.size,), isinstance(source.domain, PeriodicPointOrbits), dtype=np.bool_
    )
    region_by_code: dict[int, RegionControl] = {}
    disabled_cells = np.zeros((cell_ids.size,), dtype=np.bool_)
    cell_map = periodic_entity_orbit_map(mesh, dimension)
    cell_lifted_qids = np.asarray(cell_map.quotient_entity_ids, dtype=np.int64)[
        np.asarray(cell_map.lifted_to_quotient, dtype=np.int64)
    ]
    if specification.region_controls:
        if not isinstance(source.domain, CellMesh) or source_regions is None:
            raise ValueError(
                "Periodic material controls require a represented periodic source mesh."
            )
        row_by_id = {int(identifier): row for row, identifier in enumerate(cell_ids)}
        for control in specification.region_controls:
            scope = control.scope
            if (
                scope.source_id != source.source_id
                or scope.source_revision != source.source_revision
                or scope.entity_dimension != dimension
                or scope.entity_set_id != mesh.entity_set(dimension).entity_set_id
            ):
                raise ValueError(
                    "Periodic region controls must bind the represented source-cell entity set."
                )
            selected = np.asarray(scope.entity_ids, dtype=np.int64)
            if selected.size == 0 or any(
                int(identifier) not in row_by_id for identifier in selected
            ):
                raise ValueError(
                    "Periodic region controls must select existing lifted source cells."
                )
            requested_rows = np.asarray(
                [row_by_id[int(identifier)] for identifier in selected], dtype=np.int64
            )
            selected_quotient_ids, rows = _quotient_ids_for_lifted_ids(
                mesh, dimension, selected
            )
            if not np.array_equal(np.sort(requested_rows), rows):
                raise ValueError(
                    "Periodic material region scopes must contain complete cell orbits."
                )
            codes = np.unique(cell_classes[rows])
            if codes.size != 1:
                raise ValueError(
                    "Each periodic region control must agree with its source material IDs."
                )
            code = int(codes[0])
            previous = region_by_code.get(code)
            if previous is not None and (
                previous.region_name,
                previous.material_id,
                previous.role,
            ) != (control.region_name, control.material_id, control.role):
                raise ValueError(
                    "One periodic material ID cannot name multiple physical regions."
                )
            code_rows = np.flatnonzero(cell_classes == code)
            code_quotient_ids = set(map(int, cell_lifted_qids[code_rows]))
            if code_quotient_ids != set(map(int, selected_quotient_ids)):
                raise ValueError(
                    "Region-control scopes must cover all source quotient cells with their material ID."
                )
            region_by_code[code] = control
            if not control.meshing_enabled:
                disabled_cells[rows] = True
    for quotient_row in range(cell_map.quotient_entity_ids.size):
        members = np.flatnonzero(
            np.asarray(cell_map.lifted_to_quotient, dtype=np.int64) == quotient_row
        )
        if np.unique(cell_classes[members]).size != 1:
            raise ValueError(
                "Periodic material IDs must agree over every source cell orbit."
            )
    if np.any(disabled_cells):
        connectivity = mesh.connectivity
        if isinstance(connectivity, SimplicialConnectivity):
            cell_edges = np.asarray(connectivity.cell_entities[1], dtype=np.int64)
        elif isinstance(connectivity, TetrahedralConnectivity):
            face_edges = np.asarray(connectivity.face_edges, dtype=np.int64)
            cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int64)
            cell_edges = np.stack(
                tuple(
                    np.unique(face_edges[cell_faces[cell_row]])
                    for cell_row in range(cell_faces.shape[0])
                )
            )
        else:
            raise ValueError(
                "Periodic material exclusions require simplex edge incidence."
            )
        protected_edges[np.unique(cell_edges[disabled_cells])] = True
        cell_vertices = np.concatenate(
            tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
        )
        fixed_vertices[np.unique(cell_vertices[disabled_cells])] = True
    if specification.protected_features:
        if not isinstance(source.domain, CellMesh):
            raise ValueError(
                "Periodic protected features require a represented source mesh."
            )
        for class_id, feature in enumerate(specification.protected_features, start=1):
            scope = feature.scope
            if (
                scope.source_id != source.source_id
                or scope.source_revision != source.source_revision
                or not feature.hard
                or feature.maximum_deviation != 0.0
            ):
                raise ValueError(
                    "Periodic metric adaptation requires hard exact feature controls on the represented mesh."
                )
            if (
                feature.feature_kind is FeatureKind.CORNER
                and scope.entity_dimension == 0
                and scope.entity_set_id == mesh.entity_set(0).entity_set_id
            ):
                _, rows = _quotient_ids_for_lifted_ids(
                    mesh, 0, np.asarray(scope.entity_ids, dtype=np.int64)
                )
                fixed_vertices[rows] = True
                continue
            if (
                feature.feature_kind
                not in (FeatureKind.CURVE, FeatureKind.MATERIAL_INTERFACE)
                or scope.entity_dimension != 1
                or scope.entity_set_id != mesh.entity_set(1).entity_set_id
            ):
                raise ValueError(
                    "Periodic triangle features must be exact corners or edge features."
                )
            _, rows = _quotient_ids_for_lifted_ids(
                mesh, 1, np.asarray(scope.entity_ids, dtype=np.int64)
            )
            if np.any((edge_classes[rows] != 0) & (edge_classes[rows] != class_id)):
                raise ValueError("Periodic exact feature classes overlap incompatibly.")
            edge_classes[rows] = class_id
            protected_edges[rows] = True
    edge_map = periodic_entity_orbit_map(mesh, 1)
    for quotient_row in range(edge_map.quotient_entity_ids.size):
        members = np.flatnonzero(
            np.asarray(edge_map.lifted_to_quotient, dtype=np.int64) == quotient_row
        )
        if (
            np.unique(edge_classes[members]).size != 1
            or np.unique(protected_edges[members]).size != 1
        ):
            raise ValueError(
                "Periodic feature controls must agree over every edge orbit."
            )
    vertex_map = periodic_entity_orbit_map(mesh, 0)
    for quotient_row in range(vertex_map.quotient_entity_ids.size):
        members = np.flatnonzero(
            np.asarray(vertex_map.lifted_to_quotient, dtype=np.int64) == quotient_row
        )
        if np.unique(fixed_vertices[members]).size != 1:
            raise ValueError(
                "Periodic fixed-vertex controls must agree over every vertex orbit."
            )
    if dimension == 2:
        incidence = mesh.topology.incidences[1].relation
        valid = np.asarray(incidence.valid, dtype=np.bool_)
        edge_rows = np.asarray(incidence.source_indices, dtype=np.int64)[valid]
        cell_rows = np.asarray(incidence.target_indices, dtype=np.int64)[valid]
        incident_counts = np.bincount(edge_rows, minlength=edge_ids.size)
        required_class = incident_counts == 1
        for edge_row in np.flatnonzero(incident_counts == 2):
            adjacent_cells = cell_rows[edge_rows == edge_row]
            if np.unique(cell_classes[adjacent_cells]).size > 1:
                required_class[edge_row] = True
        reserved_class = len(specification.protected_features) + 1
        edge_classes[required_class & (edge_classes == 0)] = reserved_class
    return cell_classes, edge_classes, protected_edges, fixed_vertices, region_by_code


def _adapt_periodic_triangles(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    mesh: CellMesh,
    source_regions: np.ndarray | None,
    metric_controls: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[int, RegionControl],
    ],
    /,
) -> tuple[
    CellMesh,
    tuple[np.ndarray, ...],
    np.ndarray | None,
    Any,
    dict[int, RegionControl],
]:
    if mesh.topological_dimension != 2 or any(
        block.cell_kind != "triangle" for block in mesh.blocks
    ):
        raise ValueError("Periodic local metric transactions require a triangle mesh.")
    cell_classes, edge_classes, protected_edges, fixed_vertices, region_controls = (
        metric_controls
    )
    source_ids = (
        np.asarray(source.scope_entity_ids(0), dtype=np.int64)
        if isinstance(source.domain, PeriodicPointOrbits)
        else _periodic_quotient_entity_ids(mesh, 0)
    )
    source_size_entity_set_id = (
        source.scope_entity_set_id(0)
        if isinstance(source.domain, PeriodicPointOrbits)
        else _periodic_quotient_entity_set_id(mesh, 0)
    )
    metric = periodic_metric_from_uniform_controls(
        mesh,
        specification,
        source.source_id,
        source.source_revision,
        source_ids,
        source_size_entity_set_id,
    )
    outcome = execute_local_metric_adaptation(
        mesh,
        metric,
        cell_classes=cell_classes,
        edge_classes=edge_classes,
        protected_edges=protected_edges,
        fixed_vertices=fixed_vertices,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=True,
    )
    if not outcome.evidence.converged:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Periodic metric adaptation did not meet the unit-edge criterion; "
            f"{outcome.evidence.out_of_range_edges} edges remain outside the accepted range.",
            stage="periodic-metric-adaptation",
        )
    target = stage_periodic_topology_edit(mesh, outcome.edit)
    if target is None or target.periodic_topology is None:
        raise ValueError("Periodic metric edit lost its quotient topology owner.")
    target_scopes = _periodic_target_size_scopes(
        mesh, target, outcome.edit, specification
    )
    target_regions = (
        None
        if source_regions is None
        else _target_cell_regions(mesh, target, outcome.edit, source_regions)
    )
    return target, target_scopes, target_regions, outcome.edit, region_controls


def _adapt_periodic_tetrahedra(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    mesh: CellMesh,
    source_regions: np.ndarray | None,
    metric_controls: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[int, RegionControl],
    ],
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> tuple[
    CellMesh,
    tuple[np.ndarray, ...],
    np.ndarray | None,
    CellTopologyEdit,
    dict[int, RegionControl],
]:
    """Adapt a certified periodic P1 tetra source with the shared metric kernel.

    Boundary facets and material interfaces come from source incidence and
    region labels; this route has no separate facet-feature controls.
    """
    if (
        not isinstance(specification, VolumeMeshingSpec)
        or mesh.topological_dimension != 3
        or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
    ):
        raise ValueError(
            "Periodic tetra metric adaptation requires a 3D tetra volume request."
        )
    cell_classes, edge_classes, protected_edges, fixed_vertices, region_controls = (
        metric_controls
    )
    source_ids = (
        np.asarray(source.scope_entity_ids(0), dtype=np.int64)
        if isinstance(source.domain, PeriodicPointOrbits)
        else _periodic_quotient_entity_ids(mesh, 0)
    )
    source_size_entity_set_id = (
        source.scope_entity_set_id(0)
        if isinstance(source.domain, PeriodicPointOrbits)
        else _periodic_quotient_entity_set_id(mesh, 0)
    )
    metric = periodic_metric_from_uniform_controls(
        mesh,
        specification,
        source.source_id,
        source.source_revision,
        source_ids,
        source_size_entity_set_id,
    )
    certified_source = certify_cell_mesh(
        mesh,
        coordinate_contract,
        geometry=CellGeometrySpec.affine(mesh),
    )
    outcome = execute_tetra_metric_adaptation(
        certified_source,
        metric,
        cell_classes=cell_classes,
        facet_classes=np.zeros((mesh.entity_set(2).entity_ids.size,), dtype=np.int64),
        edge_classes=edge_classes,
        protected_edges=protected_edges,
        fixed_vertices=fixed_vertices,
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=True,
    )
    if not outcome.evidence.converged:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Periodic tetra metric adaptation did not meet the unit-edge criterion; "
            f"status={outcome.evidence.status.value}, "
            f"{outcome.evidence.out_of_range_edges} edges remain outside the "
            "accepted range.",
            stage="periodic-tetra-metric-adaptation",
        )
    target = stage_periodic_topology_edit(mesh, outcome.edit)
    if target is None or target.periodic_topology is None:
        raise ValueError("Periodic tetra metric edit lost its quotient topology owner.")
    target_scopes = _periodic_target_size_scopes(
        mesh, target, outcome.edit, specification
    )
    target_regions = (
        None
        if source_regions is None
        else _target_cell_regions(mesh, target, outcome.edit, source_regions)
    )
    return target, target_scopes, target_regions, outcome.edit, region_controls


def _owner_local_source_bank(
    points_value: ArrayLike,
    point_ids_value: ArrayLike,
    point_owner_ranks_value: ArrayLike,
    source_owned_value: ArrayLike,
    rank: int,
    execution_group: ExecutionGroup,
    source_manifest_id: str,
    axis_name: str,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    from ..._execution_runtime import ExecutionGroup
    from .._distribution import (
        _build_sharded_owner_directory,
        _owner_directory_lookup,
        _owner_exchange_message_banks,
        _owner_group_fail_if_any,
    )

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    points = np.array(points_value, dtype=np.float64, copy=True)
    raw_ids = np.asarray(point_ids_value)
    raw_owners = np.asarray(point_owner_ranks_value)
    raw_owned = np.asarray(source_owned_value)
    members = execution_group.spec.process_indices
    parts = len(members)
    local_error = None
    if (
        points.ndim != 2
        or points.shape[1] not in (2, 3)
        or not np.all(np.isfinite(points))
        or raw_ids.shape != (points.shape[0],)
        or raw_ids.dtype.kind not in "iu"
        or raw_owners.shape != raw_ids.shape
        or raw_owners.dtype.kind not in "iu"
        or raw_owned.shape != raw_ids.shape
        or raw_owned.dtype != np.dtype(np.bool_)
        or isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or not 0 <= int(rank) < parts
        or not source_manifest_id
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        local_error = "Owner-local periodic source banks are malformed."
    if (
        raw_ids.dtype.kind == "u"
        and raw_ids.size
        and int(np.max(raw_ids)) > np.iinfo(np.int64).max
    ):
        local_error = "Owner-local periodic source IDs exceed int64."
    if raw_owners.size and (np.any(raw_owners < 0) or np.any(raw_owners >= parts)):
        local_error = "Owner-local periodic source ranks are outside the group."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic source bank"
    )
    rank_ = int(rank)
    identifiers = raw_ids.astype(np.int64, copy=False)
    owners = raw_owners.astype(np.int32, copy=False)
    owned = np.asarray(raw_owned, dtype=np.bool_)
    if np.unique(identifiers).size != identifiers.size or np.any(
        owned != (owners == rank_)
    ):
        local_error = "Owner-local periodic source ownership is inconsistent."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic source ownership"
    )
    order = np.argsort(identifiers, kind="stable")
    identifiers = identifiers[order]
    owners = owners[order]
    owned = owned[order]
    points = points[order]
    source_shard = _build_sharded_owner_directory(
        identifiers[owned],
        rank_,
        parts,
        execution_group,
        axis_name,
        source_manifest_id,
        "source-points",
    )
    shard_ids, shard_owners, source_directory_id, total_count, _ = source_shard
    resolved_owners = _owner_directory_lookup(
        identifiers,
        shard_ids,
        shard_owners,
        rank_,
        parts,
        execution_group,
        axis_name,
        "source-points",
    )
    local_error = (
        None
        if np.array_equal(resolved_owners, owners)
        else "Local periodic source owners differ from the collective directory."
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic source lookup"
    )

    outgoing_requests = []
    for owner in range(parts):
        selected = (owners == owner) & (owner != rank_)
        outgoing_requests.append(
            identifiers[selected, None]
            if np.any(selected)
            else np.zeros((0, 1), dtype=np.int64)
        )
    incoming_requests = _owner_exchange_message_banks(
        tuple(outgoing_requests), rank_, execution_group, axis_name
    )
    owned_rows = {
        int(identifier): row for row, identifier in enumerate(identifiers) if owned[row]
    }
    outgoing_points = [
        np.zeros((0, points.shape[1]), dtype=np.float64) for _ in range(parts)
    ]
    local_error = None
    for requester, requests in incoming_requests.items():
        request_ids = np.asarray(requests, dtype=np.int64)
        if request_ids.ndim != 2 or request_ids.shape[1] != 1:
            local_error = "Periodic source coordinate request rows are malformed."
            continue
        positions = [
            owned_rows.get(int(identifier), -1) for identifier in request_ids[:, 0]
        ]
        if any(position < 0 for position in positions):
            local_error = "A periodic source owner lacks a requested authored site."
            continue
        outgoing_points[requester] = points[np.asarray(positions, dtype=np.int64)]
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic source coordinate request"
    )
    incoming_points = _owner_exchange_message_banks(
        tuple(outgoing_points), rank_, execution_group, axis_name
    )
    local_rows = {int(identifier): row for row, identifier in enumerate(identifiers)}
    local_error = None
    for owner, values in incoming_points.items():
        coordinates = np.asarray(values, dtype=np.float64)
        requested_ids = np.asarray(outgoing_requests[owner], dtype=np.int64)[:, 0]
        if coordinates.shape != (requested_ids.size, points.shape[1]):
            local_error = (
                "Periodic source coordinate replies do not match their requests."
            )
            continue
        for identifier, coordinate in zip(requested_ids, coordinates, strict=True):
            row = local_rows[int(identifier)]
            if not np.array_equal(points[row], coordinate):
                local_error = (
                    "A periodic source ghost coordinate differs from its exact owner."
                )
                continue
            points[row] = coordinate
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic source coordinates"
    )
    return (
        points,
        identifiers,
        owners,
        shard_ids,
        shard_owners,
        source_directory_id,
        total_count,
    )


def _owner_local_periodic_bounds(
    triangulation: Any,
    dimension: int,
    local_owned_sites: bool,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray] | None, ...], float]:
    from .._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    record = np.zeros((2 + 2 * dimension,), dtype=np.float64)
    local_error = None
    has_owned = bool(triangulation.has_owned_sites)
    if has_owned != local_owned_sites:
        local_error = "Native periodic bounds do not match local source ownership."
    if has_owned:
        lower = np.asarray(triangulation.owner_fractional_lower, dtype=np.float64)
        upper = np.asarray(triangulation.owner_fractional_upper, dtype=np.float64)
        margin = float(triangulation.certified_margin)
        if (
            lower.shape != (dimension,)
            or upper.shape != (dimension,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
            or not np.isfinite(margin)
            or margin <= 0.0
        ):
            local_error = "Native periodic owner bounds or margin are malformed."
        else:
            record[0] = 1.0
            record[1] = margin
            record[2 : 2 + dimension] = lower
            record[2 + dimension :] = upper
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        axis_name,
        "native periodic owner bounds",
    )
    gathered = _owner_group_allgather(record, execution_group, axis_name)
    bounds: list[tuple[np.ndarray, np.ndarray] | None] = []
    margins: list[float] = []
    for row in gathered:
        flag = row[0]
        if flag not in (0.0, 1.0):
            raise ValueError("Collective native owner-bound flags are malformed.")
        if flag == 0.0:
            bounds.append(None)
            continue
        lower = np.asarray(row[2 : 2 + dimension], dtype=np.float64)
        upper = np.asarray(row[2 + dimension :], dtype=np.float64)
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError("Collective native owner AABBs are malformed.")
        bounds.append((lower, upper))
        margins.append(float(row[1]))
    if not margins:
        raise ValueError("Periodic owner-local generation has no owned source sites.")
    return tuple(bounds), min(margins)


def _owner_local_periodic_support_exchange(
    triangulation: Any,
    points: np.ndarray,
    point_ids: np.ndarray,
    source_owned: np.ndarray,
    owner_bounds: tuple[tuple[np.ndarray, np.ndarray] | None, ...],
    rank: int,
    execution_group: ExecutionGroup,
    candidate_revision: int,
    lattice: np.ndarray,
    axis_name: str,
    /,
) -> tuple[tuple[tuple[int, np.ndarray, np.ndarray], ...], str]:
    from ... import _meshcore as meshcore
    from .._distribution import (
        _owner_exchange_message_banks,
        _owner_group_fail_if_any,
        _periodic_support_owner_routes,
    )

    dimension = points.shape[1]
    arity = dimension + 1
    local_error = None
    vertices = np.asarray(triangulation.vertices, dtype=np.int64)
    shifts = np.asarray(triangulation.shifts, dtype=np.int32)
    support = np.asarray(triangulation.support_cell_mask, dtype=np.bool_)
    lower = np.asarray(triangulation.support_fractional_lower, dtype=np.float64)
    upper = np.asarray(triangulation.support_fractional_upper, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != arity
        or shifts.shape != (vertices.shape[0], arity, dimension)
        or support.shape != (vertices.shape[0],)
        or lower.shape != (vertices.shape[0], dimension)
        or upper.shape != lower.shape
        or np.any((vertices < 0) | (vertices >= point_ids.size))
        or np.any(~np.isfinite(lower) | ~np.isfinite(upper))
        or np.any(lower > upper)
    ):
        local_error = "Native periodic support candidates are malformed."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic support candidates"
    )
    support_rows = np.flatnonzero(support)
    routes = _periodic_support_owner_routes(
        lower[support_rows],
        upper[support_rows],
        owner_bounds,
        (True,) * dimension,
    )
    float_width = arity * dimension
    integer_width = 1 + arity + arity * dimension
    outgoing_points = [
        np.zeros((0, float_width), dtype=np.float64) for _ in range(len(owner_bounds))
    ]
    outgoing_identities = [
        np.zeros((0, integer_width), dtype=np.int64) for _ in range(len(owner_bounds))
    ]
    for owner, route_rows in enumerate(routes):
        if owner == rank or route_rows.size == 0:
            continue
        cell_rows = support_rows[route_rows]
        cell_vertices = vertices[cell_rows]
        cell_points = points[cell_vertices]
        cell_ids = point_ids[cell_vertices]
        cell_shifts = shifts[cell_rows]
        outgoing_points[owner] = cell_points.reshape((-1, float_width))
        identities = np.empty((cell_rows.size, integer_width), dtype=np.int64)
        identities[:, 0] = cell_rows
        identities[:, 1 : 1 + arity] = cell_ids
        identities[:, 1 + arity :] = cell_shifts.reshape(
            (cell_rows.size, arity * dimension)
        )
        outgoing_identities[owner] = identities
    incoming_points = _owner_exchange_message_banks(
        tuple(outgoing_points), rank, execution_group, axis_name
    )
    incoming_identities = _owner_exchange_message_banks(
        tuple(outgoing_identities), rank, execution_group, axis_name
    )
    local_error = (
        None
        if set(incoming_points) == set(incoming_identities)
        else "Periodic support query point/identity channels disagree."
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic support request channels"
    )

    owned_points = points[source_owned]
    owned_ids = point_ids[source_owned]
    owned_rows = {int(identifier): row for row, identifier in enumerate(owned_ids)}
    response_ids = [np.zeros((0, 1), dtype=np.int64) for _ in range(len(owner_bounds))]
    response_points = [
        np.zeros((0, dimension), dtype=np.float64) for _ in range(len(owner_bounds))
    ]
    hit_records: list[tuple[int, int, int, tuple[int, ...], int]] = []
    for requester, identity_bank in incoming_identities.items():
        identities = np.asarray(identity_bank, dtype=np.int64)
        coordinate_bank = np.asarray(incoming_points[requester], dtype=np.float64)
        if (
            identities.ndim != 2
            or identities.shape[1] != integer_width
            or coordinate_bank.shape != (identities.shape[0], float_width)
        ):
            local_error = "Periodic support request banks are malformed."
            continue
        cell_points = coordinate_bank.reshape((-1, arity, dimension))
        cell_ids = identities[:, 1 : 1 + arity]
        cell_shifts = (
            identities[:, 1 + arity :]
            .reshape((-1, arity, dimension))
            .astype(np.int32, copy=False)
        )
        try:
            hits = meshcore.periodic_owner_support_query(
                cell_points,
                cell_ids,
                cell_shifts,
                owned_points,
                owned_ids,
                lattice,
                candidate_revision,
            )
        except Exception as error:
            local_error = f"Exact periodic support query failed: {error}"
            continue
        if (
            hits.candidate_revision != candidate_revision
            or np.any(np.asarray(hits.predicate_signs, dtype=np.int8) < 0)
            or np.any(
                (np.asarray(hits.cell_rows, dtype=np.int64) < 0)
                | (np.asarray(hits.cell_rows, dtype=np.int64) >= identities.shape[0])
            )
        ):
            local_error = "Exact periodic support query returned stale or invalid hits."
            continue
        hit_rows = np.asarray(hits.cell_rows, dtype=np.int64)
        hit_ids = np.asarray(hits.point_global_ids, dtype=np.int64)
        hit_shifts = np.asarray(hits.image_shifts, dtype=np.int32)
        hit_signs = np.asarray(hits.predicate_signs, dtype=np.int8)
        hit_records.extend(
            (
                requester,
                int(identities[int(cell_row), 0]),
                int(identifier),
                tuple(
                    int(value)
                    for value in np.asarray(winding, dtype=np.int64)
                    .reshape((-1,))
                    .tolist()
                ),
                int(sign),
            )
            for cell_row, identifier, winding, sign in zip(
                hit_rows, hit_ids, hit_shifts, hit_signs, strict=True
            )
        )
        unique_ids = np.unique(hit_ids)
        positions = [owned_rows.get(int(identifier), -1) for identifier in unique_ids]
        if any(position < 0 for position in positions):
            local_error = "A periodic support owner cannot resolve a returned site ID."
            continue
        response_ids[requester] = unique_ids[:, None]
        response_points[requester] = owned_points[np.asarray(positions, dtype=np.int64)]
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic exact support query"
    )
    incoming_ids = _owner_exchange_message_banks(
        tuple(response_ids), rank, execution_group, axis_name
    )
    incoming_coordinates = _owner_exchange_message_banks(
        tuple(response_points), rank, execution_group, axis_name
    )
    local_error = (
        None
        if set(incoming_ids) == set(incoming_coordinates)
        else "Periodic support reply point/identity channels disagree."
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic support reply channels"
    )
    offers = tuple(
        (
            owner,
            np.asarray(incoming_ids[owner], dtype=np.int64).reshape(-1),
            np.asarray(incoming_coordinates[owner], dtype=np.float64),
        )
        for owner in sorted(incoming_ids)
    )
    query_digest = canonical_fingerprint(
        {
            "candidate_revision": candidate_revision,
            "local_rank": rank,
            "requests": tuple(
                (
                    owner,
                    outgoing_identities[owner],
                    outgoing_points[owner],
                )
                for owner in range(len(owner_bounds))
                if owner != rank and outgoing_identities[owner].shape[0]
            ),
            "hits": tuple(sorted(hit_records)),
        }
    )
    return offers, query_digest


def _owner_local_periodic_delaunay_candidate(
    source: NativePeriodicSource,
    source_owner_ranks: ArrayLike,
    source_owned: ArrayLike,
    rank: int,
    execution_group: ExecutionGroup,
    source_manifest_id: str,
    /,
    *,
    initial_margin: float = 1.0,
    axis_name: str = "periodic_owner_closure",
) -> dict[str, Any]:
    from ... import _meshcore as meshcore
    from ..._execution_runtime import ExecutionGroup
    from .._distribution import (
        _owner_directory_lookup,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )
    from .._periodic import PeriodicPointOrbits

    if not isinstance(source, NativePeriodicSource) or not isinstance(
        source.domain, PeriodicPointOrbits
    ):
        raise TypeError(
            "Owner-local periodic Delaunay requires NativePeriodicSource point orbits."
        )
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if (
        isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or not source_manifest_id
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise ValueError("Owner-local periodic execution identity is malformed.")
    dimension = source.ambient_dimension
    margin = float(initial_margin)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("initial_margin must be finite and positive.")
    identity_digest = canonical_fingerprint(
        {
            "source_manifest": source_manifest_id,
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "periodic_cell": source.domain.cell.cell_id,
            "dimension": dimension,
        }
    )
    identity_records = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(identity_digest), dtype=np.uint8),
        execution_group,
        axis_name,
    )
    if any(
        np.asarray(record, dtype=np.uint8).tobytes().hex() != identity_digest
        for record in identity_records
    ):
        raise ValueError(
            "Owner-local periodic ranks supplied different source authority."
        )
    (
        points,
        point_ids,
        point_owners,
        source_directory_ids,
        source_directory_owners,
        source_directory_id,
        source_count,
    ) = _owner_local_source_bank(
        source.domain.points,
        source.domain.point_ids,
        source_owner_ranks,
        source_owned,
        int(rank),
        execution_group,
        source_manifest_id,
        axis_name,
    )
    if source_count == 0:
        raise ValueError(
            "Owner-local periodic generation requires authored source sites."
        )
    rank_ = int(rank)
    owned = point_owners == rank_
    lattice = np.ascontiguousarray(
        np.asarray(source.domain.cell.vectors, dtype=np.float64)
    )
    inverse = np.ascontiguousarray(
        np.asarray(source.domain.cell.inverse_vectors, dtype=np.float64)
    )
    candidate_revision = 0
    collective_rounds: list[str] = []
    while True:
        fractional = np.ascontiguousarray(
            np.asarray(source.domain.cell.fractional(points), dtype=np.float64)
        )
        triangulation = meshcore.periodic_delaunay_local(
            points,
            fractional,
            point_ids,
            point_owners,
            rank_,
            lattice,
            inverse,
            initial_margin=margin,
        )
        owner_bounds, global_margin = _owner_local_periodic_bounds(
            triangulation,
            dimension,
            bool(np.any(owned)),
            execution_group,
            axis_name,
        )
        if np.any(owned) and np.asarray(triangulation.cell_ids).size == 0:
            local_error = "Native periodic Delaunay omitted cells for an owned site."
        else:
            local_error = None
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic local triangulation"
        )
        offers, local_query_digest = _owner_local_periodic_support_exchange(
            triangulation,
            points,
            point_ids,
            owned,
            owner_bounds,
            rank_,
            execution_group,
            candidate_revision,
            lattice,
            axis_name,
        )
        offered_ids = (
            np.concatenate(tuple(ids for _, ids, _ in offers))
            if offers
            else np.zeros((0,), dtype=np.int64)
        )
        offered_owners = (
            np.concatenate(
                tuple(
                    np.full((ids.size,), owner, dtype=np.int32)
                    for owner, ids, _ in offers
                )
            )
            if offers
            else np.zeros((0,), dtype=np.int32)
        )
        resolved_owners = _owner_directory_lookup(
            offered_ids,
            source_directory_ids,
            source_directory_owners,
            rank_,
            len(execution_group.spec.process_indices),
            execution_group,
            axis_name,
            "source-points",
        )
        local_error = (
            None
            if np.array_equal(resolved_owners, offered_owners)
            else "Periodic support replies disagree with source ownership."
        )
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic support ownership"
        )
        local_rows = {int(identifier): row for row, identifier in enumerate(point_ids)}
        new_records: dict[int, tuple[int, np.ndarray]] = {}
        local_error = None
        for owner, identifiers, coordinates in offers:
            if coordinates.shape != (identifiers.size, dimension):
                local_error = "Periodic support coordinate replies are malformed."
                continue
            for identifier, coordinate in zip(identifiers, coordinates, strict=True):
                row = local_rows.get(int(identifier))
                if row is not None:
                    if point_owners[row] != owner or not np.array_equal(
                        points[row], coordinate
                    ):
                        local_error = (
                            "Periodic support returned a conflicting authored point."
                        )
                    else:
                        points[row] = coordinate
                else:
                    previous = new_records.get(int(identifier))
                    if previous is not None and (
                        previous[0] != owner
                        or not np.array_equal(previous[1], coordinate)
                    ):
                        local_error = (
                            "Periodic support returned conflicting point owners."
                        )
                    else:
                        new_records[int(identifier)] = (owner, coordinate.copy())
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic support point merge"
        )
        new_ids = np.asarray(sorted(new_records), dtype=np.int64)
        new_points = (
            np.asarray([new_records[int(identifier)][1] for identifier in new_ids])
            if new_ids.size
            else np.zeros((0, dimension), dtype=np.float64)
        )
        new_owners = (
            np.asarray(
                [new_records[int(identifier)][0] for identifier in new_ids],
                dtype=np.int32,
            )
            if new_ids.size
            else np.zeros((0,), dtype=np.int32)
        )
        global_new_counts = _owner_group_allgather(
            np.asarray([new_ids.size], dtype=np.int64), execution_group, axis_name
        )
        digest_records = _owner_group_allgather(
            np.frombuffer(bytes.fromhex(local_query_digest), dtype=np.uint8),
            execution_group,
            axis_name,
        )
        collective_rounds.append(
            canonical_fingerprint(
                {
                    "candidate_revision": candidate_revision,
                    "rank_query_digests": tuple(
                        np.asarray(record, dtype=np.uint8).tobytes().hex()
                        for record in digest_records
                    ),
                }
            )
        )
        if int(np.sum(global_new_counts)) == 0:
            break
        if new_ids.size:
            points = np.concatenate((points, new_points), axis=0)
            point_ids = np.concatenate((point_ids, new_ids))
            point_owners = np.concatenate((point_owners, new_owners))
            order = np.argsort(point_ids, kind="stable")
            points, point_ids, point_owners = (
                points[order],
                point_ids[order],
                point_owners[order],
            )
            owned = point_owners == rank_
        candidate_revision += 1
    shell_certificate = {
        "source_manifest_id": source_manifest_id,
        "source_directory_id": source_directory_id,
        "periodic_cell_id": source.domain.cell.cell_id,
        "candidate_revision": candidate_revision,
        "candidate_revision_id": canonical_fingerprint(
            {
                "kind": "owner-local-periodic-candidate",
                "source_manifest": source_manifest_id,
                "source_directory": source_directory_id,
                "periodic_cell": source.domain.cell.cell_id,
                "support_rounds": tuple(collective_rounds),
            }
        ),
        "certified_periodic_margin": global_margin,
        "owner_fractional_bounds": tuple(
            None if bounds is None else (bounds[0].copy(), bounds[1].copy())
            for bounds in owner_bounds
        ),
        "support_rounds": tuple(collective_rounds),
    }
    return {
        "triangulation": triangulation,
        "points": points,
        "point_ids": point_ids,
        "point_owner_ranks": point_owners,
        "source_owned": owned,
        "source_count": source_count,
        "source_directory_id": source_directory_id,
        "source_manifest_id": source_manifest_id,
        "candidate_revision_id": shell_certificate["candidate_revision_id"],
        "certified_periodic_margin": global_margin,
        "source_directory_bucket_ids": source_directory_ids,
        "source_directory_bucket_owners": source_directory_owners,
        "periodic_shell_certificate": shell_certificate,
    }


def _owner_local_periodic_delaunay_mesh_data(
    source: NativePeriodicSource,
    candidate: dict[str, Any],
    rank: int,
    /,
) -> dict[str, Any]:
    from fractions import Fraction

    from ...discretization._cell_mesh import CellBlock
    from ...discretization._periodic_topology import _entity_vertex_rows
    from ...discretization.fem._reference import lagrange_element

    if not isinstance(source.domain, PeriodicPointOrbits):
        raise TypeError("Owner-local Delaunay output requires periodic point orbits.")
    domain = source.domain
    triangulation = candidate["triangulation"]
    points = np.asarray(candidate["points"], dtype=np.float64)
    point_ids = np.asarray(candidate["point_ids"], dtype=np.int64)
    point_owners = np.asarray(candidate["point_owner_ranks"], dtype=np.int32)
    source_owned = np.asarray(candidate["source_owned"], dtype=np.bool_)
    dimension = domain.ambient_dimension
    arity = dimension + 1
    raw_cells = np.asarray(triangulation.vertices, dtype=np.int64)
    raw_shifts = np.asarray(triangulation.shifts, dtype=np.int64)
    raw_cell_ids = np.asarray(triangulation.cell_ids, dtype=np.int64)
    raw_cell_owners = np.asarray(triangulation.cell_owner_ranks, dtype=np.int32)
    raw_support = np.asarray(triangulation.support_cell_mask, dtype=np.bool_)
    if (
        points.ndim != 2
        or points.shape[1] != dimension
        or point_ids.shape != (points.shape[0],)
        or point_owners.shape != point_ids.shape
        or source_owned.shape != point_ids.shape
        or not np.array_equal(source_owned, point_owners == rank)
        or raw_cells.ndim != 2
        or raw_cells.shape[1] != arity
        or raw_shifts.shape != (raw_cells.shape[0], arity, dimension)
        or raw_cell_ids.shape != (raw_cells.shape[0],)
        or raw_cell_owners.shape != raw_cell_ids.shape
        or raw_support.shape != raw_cell_ids.shape
    ):
        raise ValueError("Owner-local periodic Delaunay output arrays are malformed.")
    if np.any((raw_cell_owners == rank) & ~raw_support):
        raise ValueError("A locally owned periodic cell is absent from support closure.")
    selected = np.flatnonzero(raw_support)
    cell_order = np.argsort(raw_cell_ids[selected], kind="stable")
    selected = selected[cell_order]
    cell_ids = raw_cell_ids[selected]
    cell_owner_ranks = raw_cell_owners[selected]
    source_rows = raw_cells[selected]
    cell_vertex_shifts = raw_shifts[selected]
    raw_key_ids = np.asarray(triangulation.cell_key_ids, dtype=np.int64)[selected]
    raw_relative = np.asarray(triangulation.cell_key_relative_shifts, dtype=np.int64)[
        selected
    ]
    if raw_key_ids.shape != (selected.size, arity) or raw_relative.shape != (
        selected.size,
        arity,
        dimension,
    ):
        raise ValueError("Native periodic quotient cell keys are malformed.")

    vertex_keys = tuple(
        sorted(
            {
                (
                    int(point_ids[int(source_rows[cell, local])]),
                    tuple(int(value) for value in cell_vertex_shifts[cell, local]),
                )
                for cell in range(selected.size)
                for local in range(arity)
            }
        )
    )
    vertex_key_text = tuple(
        canonical_fingerprint(
            {
                "kind": "periodic-delaunay-lifted-vertex",
                "source": source.source_id,
                "revision": source.source_revision,
                "periodic_cell": domain.cell.cell_id,
                "site": identifier,
                "winding": winding,
            }
        )
        for identifier, winding in vertex_keys
    )
    unsorted_vertex_ids = np.asarray(
        _numeric_entity_ids(
            "periodic-delaunay-lifted-vertex",
            vertex_key_text,
            tuple(() for _ in vertex_key_text),
        ),
        dtype=np.int64,
    )
    vertex_order = np.argsort(unsorted_vertex_ids, kind="stable")
    vertex_ids = unsorted_vertex_ids[vertex_order]
    vertex_keys = tuple(vertex_keys[int(row)] for row in vertex_order)
    vertex_row_by_key = {key: row for row, key in enumerate(vertex_keys)}
    source_row_by_id = {int(identifier): row for row, identifier in enumerate(point_ids)}
    lattice = np.asarray(domain.cell.vectors, dtype=np.float64)
    exact_vertex_coordinates: list[tuple[Fraction, ...]] = []
    vertex_coordinates = np.empty((len(vertex_keys), dimension), dtype=np.float64)
    for vertex, (identifier, winding) in enumerate(vertex_keys):
        source_row = source_row_by_id[identifier]
        exact_coordinates = []
        for axis in range(dimension):
            base = Fraction.from_float(float(points[source_row, axis]))
            translation = sum(
                (
                    Fraction.from_float(float(lattice[lattice_axis, axis]))
                    * winding[lattice_axis]
                    for lattice_axis in range(dimension)
                ),
                Fraction(0),
            )
            exact = base + translation
            exact_coordinates.append(exact)
            vertex_coordinates[vertex, axis] = (
                float(points[source_row, axis]) if translation == 0 else float(exact)
            )
        exact_vertex_coordinates.append(tuple(exact_coordinates))
    exact_vertex_bank = tuple(exact_vertex_coordinates)
    vertex_source_ids = np.asarray(
        [identifier for identifier, _ in vertex_keys], dtype=np.int64
    )
    vertex_absolute_windings = np.asarray(
        [winding for _, winding in vertex_keys], dtype=np.int64
    ).reshape((-1, dimension))
    owner_by_point_id = {
        int(identifier): int(owner)
        for identifier, owner in zip(point_ids, point_owners, strict=True)
    }
    vertex_owner_ranks = np.asarray(
        [owner_by_point_id[int(identifier)] for identifier in vertex_source_ids],
        dtype=np.int32,
    )
    vertex_rows = np.arange(vertex_ids.size, dtype=np.int64)
    coordinate_rows = np.arange(vertex_ids.size, dtype=np.int64)
    cells = np.empty((selected.size, arity), dtype=np.int32)
    cell_measures = np.empty((selected.size,), dtype=np.float64)
    exact_cell_measures: list[Fraction] = []
    cell_windings = np.empty((selected.size, dimension), dtype=np.int64)
    for cell in range(selected.size):
        exact_cell_points = []
        anchor_local = min(
            range(arity),
            key=lambda local: (
                int(point_ids[int(source_rows[cell, local])]),
                tuple(int(value) for value in cell_vertex_shifts[cell, local]),
            ),
        )
        cell_windings[cell] = cell_vertex_shifts[cell, anchor_local]
        for local in range(arity):
            key = (
                int(point_ids[int(source_rows[cell, local])]),
                tuple(int(value) for value in cell_vertex_shifts[cell, local]),
            )
            cells[cell, local] = vertex_row_by_key[key]
            exact_cell_points.append(exact_vertex_bank[int(cells[cell, local])])
        origin = exact_cell_points[0]
        edges = tuple(
            tuple(
                exact_cell_points[row][axis] - origin[axis] for axis in range(dimension)
            )
            for row in range(1, arity)
        )
        if dimension == 2:
            determinant = edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0]
            measure = determinant / 2
        else:
            first, second, third = edges
            determinant = (
                first[0] * (second[1] * third[2] - second[2] * third[1])
                - first[1] * (second[0] * third[2] - second[2] * third[0])
                + first[2] * (second[0] * third[1] - second[1] * third[0])
            )
            measure = determinant / 6
        if measure <= 0:
            raise ValueError("Native periodic simplex has nonpositive exact measure.")
        exact_cell_measures.append(measure)
        cell_measures[cell] = float(measure)

    cell_rows = np.arange(cell_ids.size, dtype=np.int64)
    cell_kind = "triangle" if dimension == 2 else "tetrahedron"
    element = lagrange_element(cell_kind, 1)
    geometry_layout_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-simplex-layout",
            "cell_kind": cell_kind,
            "element": element.element_id,
            "dimension": dimension,
        }
    )
    blocks: tuple[dict[str, Any], ...]
    local_entity_ids: dict[int, np.ndarray] = {}
    entity_vertex_offsets: dict[int, np.ndarray] = {}
    entity_vertex_values: dict[int, np.ndarray] = {}
    mesh = None
    if cell_ids.size:
        from ...discretization._cell_mesh import CellMesh

        block = CellBlock("periodic", cell_kind, cells, global_ids=cell_ids)
        mesh = CellMesh(
            vertex_coordinates,
            (block,),
            vertex_global_ids=vertex_ids,
        )
        entity_vertices = _entity_vertex_rows(mesh)
        for degree, rows_at_degree in enumerate(entity_vertices):
            entity_set = mesh.entity_set(degree)
            identifiers = np.asarray(entity_set.entity_ids, dtype=np.int64)
            if degree == 0:
                ordered_rows = np.arange(vertex_ids.size, dtype=np.int64)
            elif degree == dimension:
                ordered_rows = np.arange(cell_ids.size, dtype=np.int64)
                if not np.array_equal(identifiers, cell_ids):
                    raise ValueError(
                        "Local periodic simplex row order changed its cell IDs."
                    )
            else:
                ordered_rows = np.argsort(identifiers, kind="stable")
            local_entity_ids[degree] = identifiers[ordered_rows]
            offsets = np.zeros((ordered_rows.size + 1,), dtype=np.int64)
            values: list[np.ndarray] = []
            for local_row, entity_row in enumerate(ordered_rows):
                vertices_ = np.asarray(rows_at_degree[int(entity_row)], dtype=np.int64)
                values.append(vertices_.astype(np.int32, copy=False))
                offsets[local_row + 1] = offsets[local_row] + vertices_.size
            entity_vertex_offsets[degree] = offsets
            entity_vertex_values[degree] = (
                np.concatenate(values).astype(np.int32, copy=False)
                if values
                else np.zeros((0,), dtype=np.int32)
            )
        blocks = (
            {
                "name": "periodic",
                "cell_kind": cell_kind,
                "cell_rows": cell_rows,
                "cell_ids": cell_ids,
                "vertices": cells,
                "vertex_valid": np.ones(cells.shape, dtype=np.bool_),
                "coordinate_routes": cells,
                "element": element,
                "element_rows": cell_rows.astype(np.int32),
                "source_cell_ids": cell_ids,
                "chart_arguments": None,
                "cell_measures": cell_measures,
                "face_ids": None,
                "cell_face_offsets": None,
                "cell_face_values": None,
                "cell_face_signs": None,
                "face_vertex_offsets": None,
                "face_vertex_values": None,
                "cell_vertex_shifts": cell_vertex_shifts,
            },
        )
    else:
        for degree in range(dimension + 1):
            local_entity_ids[degree] = np.zeros((0,), dtype=np.int64)
            entity_vertex_offsets[degree] = np.zeros((1,), dtype=np.int64)
            entity_vertex_values[degree] = np.zeros((0,), dtype=np.int32)
        blocks = ()
    key_offsets = np.arange(cell_ids.size + 1, dtype=np.int64) * arity
    support_lower = np.asarray(triangulation.support_fractional_lower, dtype=np.float64)[
        selected
    ]
    support_upper = np.asarray(triangulation.support_fractional_upper, dtype=np.float64)[
        selected
    ]
    return {
        "cell_rows": cell_rows,
        "cell_ids": cell_ids,
        "owned": cell_owner_ranks == rank,
        "cell_measures": cell_measures,
        "exact_cell_measures": tuple(exact_cell_measures),
        "cell_owner_ranks": cell_owner_ranks,
        "support_cell_mask": np.ones(cell_ids.shape, dtype=np.bool_),
        "vertices": cells,
        "vertex_valid": np.ones(cells.shape, dtype=np.bool_),
        "vertex_rows": vertex_rows,
        "vertex_ids": vertex_ids,
        "vertex_coordinates": vertex_coordinates,
        "exact_vertex_coordinates": exact_vertex_bank,
        "vertex_source_ids": vertex_source_ids,
        "vertex_owner_ranks": vertex_owner_ranks,
        "vertex_absolute_windings": vertex_absolute_windings,
        "coordinate_rows": coordinate_rows,
        "coordinates": vertex_coordinates.copy(),
        "exact_coordinates": exact_vertex_bank,
        "entity_set_ids": {},
        "local_entity_ids": local_entity_ids,
        "entity_vertex_offsets": entity_vertex_offsets,
        "entity_vertex_values": entity_vertex_values,
        "cell_key_offsets": key_offsets,
        "cell_key_source_ids": raw_key_ids.reshape(-1),
        "cell_key_relative_windings": raw_relative.reshape((-1, dimension)),
        "periodic_cell_ids": cell_ids.copy(),
        "vertex_representative_ids": vertex_source_ids.copy(),
        "vertex_windings": vertex_absolute_windings.copy(),
        "cell_windings": cell_windings,
        "support_fractional_lower": support_lower,
        "support_fractional_upper": support_upper,
        "geometry_layout_id": geometry_layout_id,
        "blocks": blocks,
        "local_mesh": mesh,
        "cell_kind": cell_kind,
        "cell_vertex_shifts": cell_vertex_shifts,
        "vertex_keys": vertex_keys,
    }


def _owner_local_periodic_candidate_from_mesh_data(
    source: NativePeriodicSource,
    candidate_revision_id: str,
    mesh_data: dict[str, Any],
    mesh_authority: _OwnerLocalPeriodicMeshAuthority,
    source_entry: CompositionEntry,
    source_content_id: str,
    coordinate_contract_id: str,
    rank: int,
    part_count: int,
    dimension: int,
    periodic_cell: PeriodicCell,
    /,
) -> OwnerLocalPeriodicCandidate:
    from ...discretization._periodic_cell import PeriodicCell
    from ...lifecycle import CompositionEntry
    from .._periodic import OwnerLocalPeriodicCandidate, PeriodicPointOrbits

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.point_owner_ranks is None
        or not isinstance(mesh_authority, _OwnerLocalPeriodicMeshAuthority)
        or not isinstance(source_entry, CompositionEntry)
        or not isinstance(source_content_id, str)
        or not source_content_id
        or not isinstance(periodic_cell, PeriodicCell)
        or periodic_cell.cell_id != source.domain.cell.cell_id
        or source.domain.ambient_dimension != dimension
    ):
        raise TypeError(
            "Owner-local periodic candidate construction is not source-bound."
        )
    point_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
    points = np.asarray(source.domain.points, dtype=np.float64)
    point_owners = np.asarray(source.point_owner_ranks, dtype=np.int32)
    exact_points = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in points
    )
    return OwnerLocalPeriodicCandidate(
        rank,
        part_count,
        dimension,
        mesh_data["local_mesh"],
        periodic_cell,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_manifest_id=source_content_id,
        source_mesh_id=source_entry.structure_id,
        source_distribution_id=mesh_authority.owner_directory.directory_id,
        source_part_id=mesh_authority.part_id,
        source_geometry_id=mesh_authority.geometry_id,
        coordinate_contract_id=coordinate_contract_id,
        source_topology_id=mesh_authority.topology_id,
        periodic_topology_id=mesh_authority.periodic_topology_id,
        face_coverage_id=mesh_authority.face_coverage_id,
        candidate_revision_id=candidate_revision_id,
        periodic_cell_id=periodic_cell.cell_id,
        source_context_id=source.binding_id,
        geometry_layout_id=mesh_data["geometry_layout_id"],
        source_point_coordinates=points,
        source_point_ids=point_ids,
        source_owner_ranks=point_owners,
        source_owned=point_owners == rank,
        source_point_exact_coordinates=exact_points,
        cell_ids=mesh_data["cell_ids"],
        cell_owner_ranks=mesh_data["cell_owner_ranks"],
        cell_key_offsets=mesh_data["cell_key_offsets"],
        cell_key_source_ids=mesh_data["cell_key_source_ids"],
        cell_key_relative_windings=mesh_data["cell_key_relative_windings"],
        cell_windings=mesh_data["cell_windings"],
        vertex_ids=mesh_data["vertex_ids"],
        vertex_owner_ranks=mesh_data["vertex_owner_ranks"],
        vertex_source_ids=mesh_data["vertex_source_ids"],
        vertex_absolute_windings=mesh_data["vertex_absolute_windings"],
        exact_vertex_coordinates=tuple(mesh_data["exact_vertex_coordinates"]),
        exact_cell_measures=tuple(mesh_data["exact_cell_measures"]),
        owned_cell_ids=mesh_data["graph_cell_ids"],
        owned_cell_owner_ranks=mesh_data["graph_cell_owner_ranks"],
        owned_cell_weights=mesh_data["graph_cell_weights"],
        cell_neighbor_offsets=mesh_data["graph_cell_offsets"],
        cell_neighbor_ids=mesh_data["graph_neighbor_ids"],
        cell_neighbor_owner_ranks=mesh_data["graph_neighbor_owner_ranks"],
        cell_neighbor_weights=mesh_data["graph_neighbor_weights"],
    )


def _owner_local_periodic_exchange_json_messages(
    outgoing_by_owner: tuple[bytes | None, ...],
    local_rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> dict[int, bytes]:
    from ..._execution_runtime import ExecutionGroup
    from .._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    if (
        not isinstance(execution_group, ExecutionGroup)
        or not isinstance(local_rank, Integral)
        or isinstance(local_rank, bool)
        or not 0 <= int(local_rank) < len(execution_group.spec.process_indices)
        or len(outgoing_by_owner) != len(execution_group.spec.process_indices)
        or outgoing_by_owner[int(local_rank)] is not None
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise ValueError("Periodic owner messages do not match their execution group.")
    rank = int(local_rank)
    local_width = max(
        (len(message) for message in outgoing_by_owner if message is not None),
        default=0,
    )
    widths = np.asarray(
        _owner_group_allgather(
            np.asarray([local_width], dtype=np.int64),
            execution_group,
            f"{axis_name}_width",
        ),
        dtype=np.int64,
    ).reshape(-1)
    if widths.shape != (len(outgoing_by_owner),) or np.any(widths < 0):
        raise ValueError("Periodic owner message widths are malformed.")
    width = max(1, int(np.max(widths, initial=0)))
    outgoing = []
    for message in outgoing_by_owner:
        if message is None:
            outgoing.append(np.zeros((0, width), dtype=np.uint8))
            continue
        if not message:
            raise ValueError("Periodic owner messages cannot be empty.")
        bank = np.zeros((1, width), dtype=np.uint8)
        bank[0, : len(message)] = np.frombuffer(message, dtype=np.uint8)
        outgoing.append(bank)
    incoming = _owner_exchange_message_banks(
        tuple(outgoing),
        rank,
        execution_group,
        f"{axis_name}_payload",
    )
    messages: dict[int, bytes] = {}
    local_error = None
    try:
        for sender, rows in incoming.items():
            values = np.asarray(rows, dtype=np.uint8)
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError("Periodic owner message rows have the wrong width.")
            decoded = tuple(bytes(row).rstrip(b"\0") for row in values)
            if any(not value for value in decoded):
                raise ValueError("Periodic owner message payload is empty.")
            if sender in messages:
                raise ValueError("Periodic owner sent duplicate directed messages.")
            if len(decoded) != 1:
                raise ValueError("Periodic owner sent an unexpected message count.")
            messages[sender] = decoded[0]
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_decode",
        "periodic owner message decoding",
    )
    return messages


def _owner_local_periodic_vertex_owners_from_cells(
    mesh_data: dict[str, Any],
    target_cell_ids: ArrayLike,
    target_cell_owner_ranks: ArrayLike,
    rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> dict[int, int]:
    from ..._execution_runtime import ExecutionGroup
    from .._distribution import (
        _owner_group_fail_if_any,
        _owner_group_minimum_owner_ranks,
    )

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    parts = len(execution_group.spec.process_indices)
    local_error = None
    query_ids = np.zeros((0,), dtype=np.int64)
    owner_candidates = np.zeros((0, 2), dtype=np.int64)
    try:
        cell_ids = np.asarray(mesh_data["cell_ids"])
        cells = np.asarray(mesh_data["vertices"])
        vertex_ids = np.asarray(mesh_data["vertex_ids"])
        selected_cells = np.asarray(target_cell_ids)
        selected_owners = np.asarray(target_cell_owner_ranks)
        if (
            not 0 <= rank < parts
            or cell_ids.ndim != 1
            or cell_ids.dtype.kind not in "iu"
            or np.unique(cell_ids).size != cell_ids.size
            or cells.ndim != 2
            or cells.shape[0] != cell_ids.size
            or cells.dtype.kind not in "iu"
            or vertex_ids.ndim != 1
            or vertex_ids.dtype.kind not in "iu"
            or np.unique(vertex_ids).size != vertex_ids.size
            or np.any(cells < 0)
            or np.any(cells >= vertex_ids.size)
            or selected_cells.ndim != 1
            or selected_cells.dtype.kind not in "iu"
            or np.unique(selected_cells).size != selected_cells.size
            or selected_owners.shape != selected_cells.shape
            or selected_owners.dtype.kind not in "iu"
            or np.any(selected_owners < 0)
            or np.any(selected_owners >= parts)
        ):
            raise ValueError("Periodic cell/vertex incidence owner rows are malformed.")
        cell_ids = cell_ids.astype(np.int64, copy=False)
        cells = cells.astype(np.int64, copy=False)
        vertex_ids = vertex_ids.astype(np.int64, copy=False)
        selected_cells = selected_cells.astype(np.int64, copy=False)
        selected_owners = selected_owners.astype(np.int32, copy=False)
        row_by_id = {
            int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
        }
        candidate_rows: list[tuple[int, int]] = []
        query_values: set[int] = set()
        for cell_id, owner in zip(
            selected_cells.tolist(), selected_owners.tolist(), strict=True
        ):
            if int(cell_id) not in row_by_id:
                raise ValueError("Periodic owner cell row is absent from its local mesh.")
            for vertex_row in cells[row_by_id[int(cell_id)]].tolist():
                identifier = int(vertex_ids[int(vertex_row)])
                query_values.add(identifier)
                candidate_rows.append((identifier, int(owner)))
        query_ids = np.asarray(sorted(query_values), dtype=np.int64)
        owner_candidates = np.asarray(candidate_rows, dtype=np.int64).reshape((-1, 2))
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_incidence",
        "periodic vertex-owner incidence",
    )
    owners = _owner_group_minimum_owner_ranks(
        query_ids,
        owner_candidates,
        rank,
        execution_group,
        f"{axis_name}_reduction",
    )
    return {
        int(identifier): int(owner)
        for identifier, owner in zip(query_ids.tolist(), owners.tolist(), strict=True)
    }


def _owner_local_periodic_graph_vertex_owners(
    mesh_data: dict[str, Any],
    result: DistributedGraphPartitionResult,
    rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> dict[int, int]:
    from ...graph._distributed_partition import DistributedGraphPartitionResult

    if not isinstance(result, DistributedGraphPartitionResult):
        raise TypeError("result must be DistributedGraphPartitionResult.")
    return _owner_local_periodic_vertex_owners_from_cells(
        mesh_data,
        result.entity_ids,
        result.owners,
        rank,
        execution_group,
        axis_name,
    )


def _owner_local_periodic_graph_neighbor_owners(
    query_ids_by_old_owner: tuple[tuple[int, ...], ...],
    result: DistributedGraphPartitionResult,
    rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> dict[int, int]:
    from ..._execution_runtime import ExecutionGroup
    from ...graph._distributed_partition import DistributedGraphPartitionResult
    from .._distribution import _owner_exchange_message_banks, _owner_group_fail_if_any

    if (
        not isinstance(result, DistributedGraphPartitionResult)
        or not isinstance(execution_group, ExecutionGroup)
        or len(query_ids_by_old_owner) != len(execution_group.spec.process_indices)
        or not 0 <= rank < len(query_ids_by_old_owner)
    ):
        raise ValueError("Periodic graph neighbor owner lookup is not group-bound.")
    parts = len(query_ids_by_old_owner)
    local_labels = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            np.asarray(result.entity_ids, dtype=np.int64).tolist(),
            np.asarray(result.owners, dtype=np.int32).tolist(),
            strict=True,
        )
    }
    local_error = None
    requests: list[np.ndarray] = []
    expected: dict[int, set[int]] = {}
    try:
        for old_owner, identifiers in enumerate(query_ids_by_old_owner):
            unique_ids = tuple(sorted(set(int(value) for value in identifiers)))
            expected[old_owner] = set(unique_ids)
            if old_owner == rank:
                continue
            if unique_ids and any(value < 0 for value in unique_ids):
                raise ValueError("Periodic graph neighbor IDs must be nonnegative.")
        requests = [
            (
                np.asarray(sorted(values), dtype=np.int64).reshape((-1, 1))
                if owner != rank and values
                else np.zeros((0, 1), dtype=np.int64)
            )
            for owner, values in enumerate(expected.values())
        ]
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_request_validation",
        "periodic graph neighbor owner request",
    )
    incoming = _owner_exchange_message_banks(
        tuple(requests),
        rank,
        execution_group,
        f"{axis_name}_requests",
    )
    local_error = None
    replies: list[tuple[int, np.ndarray]] = []
    try:
        for sender, rows in incoming.items():
            values = np.asarray(rows, dtype=np.int64)
            if values.ndim != 2 or values.shape[1] != 1:
                raise ValueError("Periodic graph neighbor owner requests are malformed.")
            labels = []
            for identifier_value in values[:, 0].tolist():
                identifier = int(identifier_value)
                if identifier not in local_labels:
                    raise ValueError(
                        "Periodic graph neighbor request names an unknown cell."
                    )
                labels.append((identifier, local_labels[identifier]))
            replies_for_sender = np.asarray(labels, dtype=np.int64).reshape((-1, 2))
            replies.append((sender, replies_for_sender))
        reply_banks = [np.zeros((0, 2), dtype=np.int64) for _ in range(parts)]
        for sender, rows in replies:
            reply_banks[sender] = rows
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        reply_banks = [np.zeros((0, 2), dtype=np.int64) for _ in range(parts)]
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_reply_validation",
        "periodic graph neighbor owner response",
    )
    received = _owner_exchange_message_banks(
        tuple(reply_banks),
        rank,
        execution_group,
        f"{axis_name}_replies",
    )
    local_error = None
    owner_by_cell: dict[int, int] = {}
    try:
        for identifier in expected.get(rank, set()):
            if identifier not in local_labels:
                raise ValueError(
                    "Local periodic graph neighbor is not owned by this rank."
                )
            owner_by_cell[identifier] = local_labels[identifier]
        for sender, rows in received.items():
            values = np.asarray(rows, dtype=np.int64)
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("Periodic graph neighbor owner replies are malformed.")
            actual = set()
            for identifier_value, owner_value in values.tolist():
                identifier, owner = int(identifier_value), int(owner_value)
                if identifier in owner_by_cell or not 0 <= owner < parts:
                    raise ValueError(
                        "Periodic graph neighbor owner replies are inconsistent."
                    )
                owner_by_cell[identifier] = owner
                actual.add(identifier)
            if actual != expected.get(sender, set()):
                raise ValueError(
                    "Periodic graph neighbor owner replies omit requested cells."
                )
        if set(owner_by_cell) != set().union(*expected.values()):
            raise ValueError("Periodic graph neighbor owner lookup is incomplete.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_verify",
        "periodic graph neighbor owner verification",
    )
    return owner_by_cell


def _migrate_owner_local_periodic_delaunay_data(
    source: NativePeriodicSource,
    candidate: OwnerLocalPeriodicCandidate,
    mesh_data: dict[str, Any],
    graph_result: DistributedGraphPartitionResult,
    rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[NativePeriodicSource, dict[str, Any], np.ndarray]:
    import json

    from ..._execution_runtime import ExecutionGroup
    from ...discretization._cell_mesh import CellBlock, CellMesh
    from ...discretization._periodic_topology import _entity_vertex_rows
    from ...discretization.fem._reference import lagrange_element
    from ...graph._distributed_partition import DistributedGraphPartitionResult
    from ...meshing._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )
    from .._lineage import VertexInterpolationStencil
    from .._periodic import OwnerLocalPeriodicCandidate, PeriodicPointOrbits

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or not isinstance(candidate, OwnerLocalPeriodicCandidate)
        or not isinstance(graph_result, DistributedGraphPartitionResult)
        or not isinstance(execution_group, ExecutionGroup)
        or source.point_owner_ranks is None
        or candidate.rank != rank
        or candidate.part_count != len(execution_group.spec.process_indices)
        or not graph_result.feasible
        or graph_result.plan_id == ""
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise TypeError("Periodic graph migration requires a current Delaunay candidate.")
    parts = candidate.part_count
    dimension = candidate.dimension
    arity = dimension + 1
    if source.site_weights is not None and np.any(
        np.asarray(source.site_weights, dtype=np.float64) != 0.0
    ):
        raise ValueError("Weighted periodic sites cannot enter Delaunay graph migration.")

    def pair(value: Fraction, /) -> list[int]:
        if not isinstance(value, Fraction):
            raise TypeError(
                "Periodic graph migration requires exact rational coordinates."
            )
        return [int(value.numerator), int(value.denominator)]

    def exact_pair(value: object, /) -> Fraction:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(component, bool) or not isinstance(component, int)
                for component in value
            )
            or value[1] <= 0
        ):
            raise ValueError("Periodic graph rational payload is malformed.")
        exact = Fraction(value[0], value[1])
        if (exact.numerator, exact.denominator) != (value[0], value[1]):
            raise ValueError("Periodic graph rational payload is not canonical.")
        return exact

    def json_object(value: object, /) -> dict[str, object]:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError("Periodic graph message must be a string-keyed object.")
        return value

    def json_list(value: object, /) -> list[object]:
        if not isinstance(value, list):
            raise ValueError("Periodic graph message field must be a list.")
        return value

    def json_integer(value: object, /) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Periodic graph message integer is malformed.")
        return value

    def json_integers(value: object, /) -> list[int]:
        return [json_integer(item) for item in json_list(value)]

    def json_uint64s(value: object, /) -> list[int]:
        values = json_integers(value)
        if any(item < 0 or item > np.iinfo(np.uint64).max for item in values):
            raise ValueError("Periodic graph bit payload is outside uint64.")
        return values

    def json_fraction_rows(value: object, row_count: int, /) -> list[list[list[int]]]:
        rows = []
        for row in json_list(value):
            exact_row = []
            for item in json_list(row):
                exact_pair(item)
                exact_row.append(json_integers(item))
            rows.append(exact_row)
        if len(rows) != row_count or any(len(row) != dimension for row in rows):
            raise ValueError("Periodic graph exact coordinate rows are malformed.")
        return rows

    def decode_message(payload: bytes, /) -> _PeriodicGraphMigrationPayload:
        raw_value: object = json.loads(payload)
        raw = json_object(raw_value)
        if set(raw) != {"cells", "source_points"}:
            raise ValueError(
                "Periodic graph message fields differ from the canonical schema."
            )
        decoded_cells: list[_PeriodicGraphCellPayload] = []
        for raw_cell_value in json_list(raw["cells"]):
            raw_cell = json_object(raw_cell_value)
            cell_id = json_integer(raw_cell.get("cell_id"))
            source_owner = json_integer(raw_cell.get("source_owner"))
            target_owner = json_integer(raw_cell.get("target_owner"))
            vertices: list[_PeriodicGraphVertexPayload] = []
            for raw_vertex_value in json_list(raw_cell.get("vertices")):
                raw_vertex = json_object(raw_vertex_value)
                if set(raw_vertex) != {
                    "vertex_id",
                    "coordinate_bits",
                    "exact_coordinates",
                    "source_id",
                    "owner_rank",
                    "absolute_winding",
                    "representative_id",
                    "quotient_winding",
                    "coordinate_ancestry",
                }:
                    raise ValueError(
                        "Periodic graph vertex fields differ from its canonical schema."
                    )
                vertex_id = json_integer(raw_vertex["vertex_id"])
                exact_coordinates = json_fraction_rows(
                    [raw_vertex["exact_coordinates"]], 1
                )[0]
                coordinate_bits = json_uint64s(raw_vertex["coordinate_bits"])
                absolute_winding = json_integers(raw_vertex["absolute_winding"])
                quotient_winding = json_integers(raw_vertex["quotient_winding"])
                source_id = json_integer(raw_vertex["source_id"])
                owner_rank = json_integer(raw_vertex["owner_rank"])
                representative_id = json_integer(raw_vertex["representative_id"])
                raw_ancestry = raw_vertex["coordinate_ancestry"]
                coordinate_ancestry: _PeriodicGraphVertexCoordinateAncestryPayload | None
                if raw_ancestry is None:
                    coordinate_ancestry = None
                else:
                    ancestry = json_object(raw_ancestry)
                    if set(ancestry) != {
                        "source_entity_set_id",
                        "preserves_constants",
                        "source_ids",
                        "weight_bits",
                        "exact_weights",
                        "exact_offsets",
                    }:
                        raise ValueError(
                            "Periodic graph coordinate ancestry fields are malformed."
                        )
                    source_entity_set_id = ancestry["source_entity_set_id"]
                    preserves_constants = ancestry["preserves_constants"]
                    source_ids = json_integers(ancestry["source_ids"])
                    if (
                        not isinstance(source_entity_set_id, str)
                        or not source_entity_set_id
                        or type(preserves_constants) is not bool
                    ):
                        raise ValueError(
                            "Periodic graph coordinate ancestry identity is invalid."
                        )
                    weight_bits = json_uint64s(ancestry["weight_bits"])
                    exact_weight_rows = json_list(ancestry["exact_weights"])
                    exact_weights: list[list[int]] = []
                    for exact_value in exact_weight_rows:
                        exact = exact_pair(exact_value)
                        exact_weights.append(
                            [int(exact.numerator), int(exact.denominator)]
                        )
                    exact_offset_rows = json_list(ancestry["exact_offsets"])
                    exact_offsets: list[list[int]] = []
                    for exact_value in exact_offset_rows:
                        exact = exact_pair(exact_value)
                        exact_offsets.append(
                            [int(exact.numerator), int(exact.denominator)]
                        )
                    if (
                        len(source_ids) == 0
                        or any(value < 0 for value in source_ids)
                        or any(
                            second <= first
                            for first, second in zip(
                                source_ids, source_ids[1:], strict=False
                            )
                        )
                        or len(weight_bits) != len(source_ids)
                        or len(exact_weights) != len(source_ids)
                        or len(exact_offsets) != dimension
                    ):
                        raise ValueError(
                            "Periodic graph coordinate ancestry rows are misaligned."
                        )
                    coordinate_ancestry = {
                        "source_entity_set_id": source_entity_set_id,
                        "preserves_constants": preserves_constants,
                        "source_ids": source_ids,
                        "weight_bits": weight_bits,
                        "exact_weights": exact_weights,
                        "exact_offsets": exact_offsets,
                    }
                if (
                    vertex_id < 0
                    or source_id < 0
                    or representative_id < 0
                    or not 0 <= owner_rank < parts
                    or len(coordinate_bits) != dimension
                    or len(absolute_winding) != dimension
                    or len(quotient_winding) != dimension
                ):
                    raise ValueError("Periodic graph vertex payload is malformed.")
                vertices.append(
                    {
                        "vertex_id": vertex_id,
                        "coordinate_bits": coordinate_bits,
                        "exact_coordinates": exact_coordinates,
                        "source_id": source_id,
                        "owner_rank": owner_rank,
                        "absolute_winding": absolute_winding,
                        "representative_id": representative_id,
                        "quotient_winding": quotient_winding,
                        "coordinate_ancestry": coordinate_ancestry,
                    }
                )
            cell_vertex_shifts = [
                json_integers(row)
                for row in json_list(raw_cell.get("cell_vertex_shifts"))
            ]
            key_source_ids = json_integers(raw_cell.get("key_source_ids"))
            key_relative_windings = [
                json_integers(row)
                for row in json_list(raw_cell.get("key_relative_windings"))
            ]
            cell_winding = json_integers(raw_cell.get("cell_winding"))
            measure_payload = json_list(raw_cell.get("measure"))
            exact_measure = exact_pair(measure_payload)
            support_lower_bits = json_uint64s(raw_cell.get("support_lower_bits"))
            support_upper_bits = json_uint64s(raw_cell.get("support_upper_bits"))
            neighbors = json_integers(raw_cell.get("neighbors"))
            neighbor_owners = json_integers(raw_cell.get("neighbor_owners"))
            neighbor_weights = json_integers(raw_cell.get("neighbor_weights"))
            if (
                cell_id < 0
                or source_owner < 0
                or source_owner >= parts
                or target_owner < 0
                or target_owner >= parts
                or len(vertices) != arity
                or len({value["vertex_id"] for value in vertices}) != arity
                or len(cell_vertex_shifts) != arity
                or any(len(row) != dimension for row in cell_vertex_shifts)
                or len(key_source_ids) != arity
                or len(key_relative_windings) != arity
                or any(len(row) != dimension for row in key_relative_windings)
                or len(cell_winding) != dimension
                or exact_measure <= 0
                or len(support_lower_bits) not in (0, dimension)
                or len(support_upper_bits) != len(support_lower_bits)
                or len(neighbors) != len(neighbor_owners)
                or len(neighbors) != len(neighbor_weights)
                or len(set(neighbors)) != len(neighbors)
                or cell_id in neighbors
                or any(value < 0 for value in neighbors)
                or any(not 0 <= value < parts for value in neighbor_owners)
                or any(value <= 0 for value in neighbor_weights)
            ):
                raise ValueError("Periodic graph cell payload is malformed.")
            decoded_cells.append(
                {
                    "cell_id": cell_id,
                    "source_owner": source_owner,
                    "target_owner": target_owner,
                    "vertices": vertices,
                    "cell_vertex_shifts": cell_vertex_shifts,
                    "key_source_ids": key_source_ids,
                    "key_relative_windings": key_relative_windings,
                    "cell_winding": cell_winding,
                    "measure": [exact_measure.numerator, exact_measure.denominator],
                    "support_lower_bits": support_lower_bits,
                    "support_upper_bits": support_upper_bits,
                    "neighbors": neighbors,
                    "neighbor_owners": neighbor_owners,
                    "neighbor_weights": neighbor_weights,
                }
            )
        decoded_points: list[_PeriodicGraphSourcePointPayload] = []
        for raw_point_value in json_list(raw["source_points"]):
            raw_point = json_object(raw_point_value)
            point_id = json_integer(raw_point.get("point_id"))
            owner_rank = json_integer(raw_point.get("owner_rank"))
            coordinate_bits = json_uint64s(raw_point.get("coordinate_bits"))
            if (
                point_id < 0
                or not 0 <= owner_rank < parts
                or len(coordinate_bits) != dimension
            ):
                raise ValueError("Periodic graph source-point payload is malformed.")
            decoded_points.append(
                {
                    "point_id": point_id,
                    "owner_rank": owner_rank,
                    "coordinate_bits": coordinate_bits,
                }
            )
        return {"cells": decoded_cells, "source_points": decoded_points}

    local_error = None
    local_message: bytes | None = None
    outgoing_by_owner: list[bytes | None] = [None] * parts
    try:
        if source.domain.cell.cell_id != candidate.periodic_cell_id:
            raise ValueError("Periodic graph source cell differs from its candidate.")
        point_ids = np.asarray(candidate.source_point_ids, dtype=np.int64)
        point_coordinates = np.asarray(
            candidate.source_point_coordinates, dtype=np.float64
        )
        point_owners = np.asarray(candidate.source_owner_ranks, dtype=np.int32)
        point_bits = point_coordinates.view(np.uint64)
        source_point_payloads_by_id: dict[int, _PeriodicGraphSourcePointPayload] = {}
        for row, identifier in enumerate(point_ids.tolist()):
            if any(
                Fraction.from_float(float(point_coordinates[row, axis])) != exact_value
                for axis, exact_value in enumerate(
                    candidate.source_point_exact_coordinates[row]
                )
            ):
                raise ValueError("Graph source point exact/float shadows differ.")
            source_point_payloads_by_id[int(identifier)] = {
                "point_id": int(identifier),
                "owner_rank": int(point_owners[row]),
                "coordinate_bits": [int(value) for value in point_bits[row].tolist()],
            }
        cell_ids = np.asarray(mesh_data["cell_ids"], dtype=np.int64)
        cell_owners = np.asarray(mesh_data["cell_owner_ranks"], dtype=np.int32)
        cells = np.asarray(mesh_data["vertices"], dtype=np.int64)
        vertex_ids = np.asarray(mesh_data["vertex_ids"], dtype=np.int64)
        vertex_coordinates = np.asarray(mesh_data["vertex_coordinates"], dtype=np.float64)
        vertex_bits = vertex_coordinates.view(np.uint64)
        exact_vertices = tuple(mesh_data["exact_vertex_coordinates"])
        vertex_sources = np.asarray(mesh_data["vertex_source_ids"], dtype=np.int64)
        vertex_absolute_windings = np.asarray(
            mesh_data["vertex_absolute_windings"], dtype=np.int64
        )
        vertex_representatives = np.asarray(
            mesh_data["vertex_representative_ids"], dtype=np.int64
        )
        vertex_windings = np.asarray(mesh_data["vertex_windings"], dtype=np.int64)
        source_stencil = mesh_data.get("source_coordinate_stencil")
        source_offsets_value = mesh_data.get("exact_vertex_coordinate_offsets")
        stencil_row_by_vertex_id: dict[int, int] = {}
        exact_stencil_offsets: tuple[tuple[Fraction, ...], ...] = ()
        if source_stencil is None:
            if vertex_ids.size or source_offsets_value is not None:
                raise ValueError(
                    "Periodic graph vertices require exact coordinate ancestry."
                )
        else:
            if (
                not isinstance(source_stencil, VertexInterpolationStencil)
                or source_offsets_value is None
            ):
                raise TypeError("Periodic graph coordinate ancestry is incomplete.")
            stencil_vertex_ids = np.asarray(
                source_stencil.target_global_ids, dtype=np.int64
            )
            stencil_source_ids = np.asarray(
                source_stencil.source_global_ids, dtype=np.int64
            )
            stencil_valid = np.asarray(source_stencil.valid, dtype=np.bool_)
            stencil_weights = np.asarray(source_stencil.weights, dtype=np.float64)
            exact_stencil_offsets = tuple(tuple(row) for row in source_offsets_value)
            if (
                not np.array_equal(stencil_vertex_ids, vertex_ids)
                or stencil_source_ids.ndim != 2
                or stencil_source_ids.shape != stencil_valid.shape
                or stencil_source_ids.shape != stencil_weights.shape
                or len(source_stencil.exact_weights) != vertex_ids.size
                or len(exact_stencil_offsets) != vertex_ids.size
                or any(
                    len(source_stencil.exact_weights[row]) != stencil_source_ids.shape[1]
                    for row in range(vertex_ids.size)
                )
                or any(
                    len(offset) != dimension
                    or any(not isinstance(value, Fraction) for value in offset)
                    for offset in exact_stencil_offsets
                )
            ):
                raise ValueError(
                    "Periodic graph coordinate ancestry rows do not align with vertices."
                )
            stencil_row_by_vertex_id = {
                int(identifier): row
                for row, identifier in enumerate(stencil_vertex_ids.tolist())
            }
        cell_vertex_shifts = np.asarray(mesh_data["cell_vertex_shifts"], dtype=np.int64)
        key_offsets = np.asarray(mesh_data["cell_key_offsets"], dtype=np.int64)
        key_sources = np.asarray(mesh_data["cell_key_source_ids"], dtype=np.int64)
        key_windings = np.asarray(mesh_data["cell_key_relative_windings"], dtype=np.int64)
        cell_windings = np.asarray(mesh_data["cell_windings"], dtype=np.int64)
        exact_measures = tuple(mesh_data["exact_cell_measures"])
        raw_support_lower = mesh_data["support_fractional_lower"]
        raw_support_upper = mesh_data["support_fractional_upper"]
        if (raw_support_lower is None) != (raw_support_upper is None):
            raise ValueError("Periodic cell support bounds are only partially present.")
        support_lower = (
            None
            if raw_support_lower is None
            else np.asarray(raw_support_lower, dtype=np.float64).view(np.uint64)
        )
        support_upper = (
            None
            if raw_support_upper is None
            else np.asarray(raw_support_upper, dtype=np.float64).view(np.uint64)
        )
        support_width = 0 if support_lower is None else dimension
        if support_lower is not None and (
            support_upper is None
            or support_lower.shape != (cell_ids.size, dimension)
            or support_upper.shape != support_lower.shape
        ):
            raise ValueError("Periodic cell support bound rows are malformed.")
        graph_ids = np.asarray(mesh_data["graph_cell_ids"], dtype=np.int64)
        graph_offsets = np.asarray(mesh_data["graph_cell_offsets"], dtype=np.int64)
        graph_neighbors = np.asarray(mesh_data["graph_neighbor_ids"], dtype=np.int64)
        graph_neighbor_owners = np.asarray(
            mesh_data["graph_neighbor_owner_ranks"], dtype=np.int32
        )
        graph_neighbor_weights = np.asarray(
            mesh_data["graph_neighbor_weights"], dtype=np.int64
        )
        cell_row_by_id = {
            int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
        }
        graph_row_by_id = {
            int(identifier): row for row, identifier in enumerate(graph_ids.tolist())
        }
        vertex_target_owners = _owner_local_periodic_graph_vertex_owners(
            mesh_data,
            graph_result,
            rank,
            execution_group,
            f"{axis_name}_vertex_owners",
        )
        cell_payloads_by_owner: list[list[_PeriodicGraphCellPayload]] = [
            [] for _ in range(parts)
        ]
        source_points_by_owner: list[dict[int, _PeriodicGraphSourcePointPayload]] = [
            {} for _ in range(parts)
        ]
        for graph_row, cell_id_value in enumerate(
            np.asarray(graph_result.entity_ids, dtype=np.int64).tolist()
        ):
            cell_id = int(cell_id_value)
            target_owner = int(np.asarray(graph_result.owners)[graph_row])
            if cell_id not in cell_row_by_id or cell_id not in graph_row_by_id:
                raise ValueError(
                    "Graph partition cell row is absent from local mesh data."
                )
            cell_row = cell_row_by_id[cell_id]
            row_owner = int(cell_owners[cell_row])
            if row_owner != rank:
                raise ValueError("Graph result contains a nonowned source cell.")
            graph_index = graph_row_by_id[cell_id]
            vertex_payloads: list[_PeriodicGraphVertexPayload] = []
            for vertex_row_value in cells[cell_row].tolist():
                vertex_row = int(vertex_row_value)
                vertex_id = int(vertex_ids[vertex_row])
                if vertex_id not in vertex_target_owners:
                    raise ValueError(
                        "Graph vertex owner proof omits a local cell vertex."
                    )
                coordinate_ancestry: (
                    _PeriodicGraphVertexCoordinateAncestryPayload | None
                ) = None
                if source_stencil is not None:
                    stencil_row = stencil_row_by_vertex_id[vertex_id]
                    coordinate_ancestry = _periodic_vertex_ancestry_payload(
                        source_stencil,
                        stencil_row,
                        exact_stencil_offsets[stencil_row],
                    )
                vertex_payloads.append(
                    {
                        "vertex_id": vertex_id,
                        "coordinate_bits": [
                            int(value) for value in vertex_bits[vertex_row].tolist()
                        ],
                        "exact_coordinates": [
                            pair(value) for value in exact_vertices[vertex_row]
                        ],
                        "source_id": int(vertex_sources[vertex_row]),
                        "owner_rank": int(vertex_target_owners[vertex_id]),
                        "absolute_winding": [
                            int(value)
                            for value in vertex_absolute_windings[vertex_row].tolist()
                        ],
                        "representative_id": int(vertex_representatives[vertex_row]),
                        "quotient_winding": [
                            int(value) for value in vertex_windings[vertex_row].tolist()
                        ],
                        "coordinate_ancestry": coordinate_ancestry,
                    }
                )
            key_start, key_stop = map(int, key_offsets[cell_row : cell_row + 2])
            neighbor_start, neighbor_stop = map(
                int, graph_offsets[graph_index : graph_index + 2]
            )
            record: _PeriodicGraphCellPayload = {
                "cell_id": cell_id,
                "source_owner": rank,
                "target_owner": target_owner,
                "vertices": vertex_payloads,
                "cell_vertex_shifts": [
                    [int(value) for value in row.tolist()]
                    for row in cell_vertex_shifts[cell_row]
                ],
                "key_source_ids": [
                    int(value) for value in key_sources[key_start:key_stop].tolist()
                ],
                "key_relative_windings": [
                    [int(value) for value in row.tolist()]
                    for row in key_windings[key_start:key_stop]
                ],
                "cell_winding": [
                    int(value) for value in cell_windings[cell_row].tolist()
                ],
                "measure": pair(exact_measures[cell_row]),
                "support_lower_bits": (
                    []
                    if support_lower is None
                    else [int(value) for value in support_lower[cell_row].tolist()]
                ),
                "support_upper_bits": (
                    []
                    if support_upper is None
                    else [int(value) for value in support_upper[cell_row].tolist()]
                ),
                "neighbors": [
                    int(value)
                    for value in graph_neighbors[neighbor_start:neighbor_stop].tolist()
                ],
                "neighbor_owners": [
                    int(value)
                    for value in graph_neighbor_owners[
                        neighbor_start:neighbor_stop
                    ].tolist()
                ],
                "neighbor_weights": [
                    int(value)
                    for value in graph_neighbor_weights[
                        neighbor_start:neighbor_stop
                    ].tolist()
                ],
            }
            cell_payloads_by_owner[target_owner].append(record)
            for vertex in vertex_payloads:
                source_ids = {vertex["source_id"]}
                ancestry = vertex["coordinate_ancestry"]
                if ancestry is not None:
                    source_ids.update(ancestry["source_ids"])
                for source_id in source_ids:
                    if source_id not in source_point_payloads_by_id:
                        raise ValueError(
                            "Graph coordinate ancestry root is absent from its local source bank."
                        )
                    source_points_by_owner[target_owner][source_id] = (
                        source_point_payloads_by_id[source_id]
                    )
        for owner, cell_payloads in enumerate(cell_payloads_by_owner):
            if not cell_payloads:
                continue
            payload: _PeriodicGraphMigrationPayload = {
                "cells": cell_payloads,
                "source_points": [
                    source_points_by_owner[owner][identifier]
                    for identifier in sorted(source_points_by_owner[owner])
                ],
            }
            message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
            if owner == rank:
                local_message = message
            else:
                outgoing_by_owner[owner] = message
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_encode",
        "periodic graph cell migration encoding",
    )
    incoming_messages = _owner_local_periodic_exchange_json_messages(
        tuple(outgoing_by_owner),
        rank,
        execution_group,
        f"{axis_name}_cells",
    )
    payload_messages: list[tuple[int, bytes]] = []
    if local_message is not None:
        payload_messages.append((rank, local_message))
    payload_messages.extend(
        (sender, payload) for sender, payload in incoming_messages.items()
    )
    cell_records: dict[int, _PeriodicGraphCellPayload] = {}
    source_point_records: dict[int, _PeriodicGraphSourcePointPayload] = {}
    local_error = None
    try:
        for sender, message in payload_messages:
            payload = decode_message(message)
            for point in payload["source_points"]:
                point_id = point["point_id"]
                previous = source_point_records.get(point_id)
                if previous is not None and previous != point:
                    raise ValueError(
                        "Graph migration received conflicting source point rows."
                    )
                source_point_records[point_id] = point
            for cell in payload["cells"]:
                if (
                    cell["source_owner"] != sender
                    or cell["target_owner"] != rank
                    or cell["cell_id"] in cell_records
                ):
                    raise ValueError(
                        "Graph migration received a duplicate or misrouted cell."
                    )
                cell_records[cell["cell_id"]] = cell
        for point_id_value, owner_value, coordinate_row in zip(
            np.asarray(candidate.source_point_ids, dtype=np.int64).tolist(),
            np.asarray(candidate.source_owner_ranks, dtype=np.int32).tolist(),
            np.asarray(candidate.source_point_coordinates, dtype=np.float64)
            .view(np.uint64)
            .tolist(),
            strict=True,
        ):
            if owner_value != rank:
                continue
            owned_point: _PeriodicGraphSourcePointPayload = {
                "point_id": int(point_id_value),
                "owner_rank": int(owner_value),
                "coordinate_bits": [int(value) for value in coordinate_row],
            }
            previous = source_point_records.get(int(point_id_value))
            if previous is not None and previous != owned_point:
                raise ValueError("Graph migration changed an owned source point.")
            source_point_records[int(point_id_value)] = owned_point
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_decode",
        "periodic graph cell migration decoding",
    )
    counts = np.asarray(
        _owner_group_allgather(
            np.asarray(
                (np.asarray(graph_result.entity_ids).size, len(cell_records)),
                dtype=np.int64,
            ),
            execution_group,
            f"{axis_name}_cell_counts",
        ),
        dtype=np.int64,
    )
    if (
        counts.shape != (parts, 2)
        or np.any(counts < 0)
        or int(np.sum(counts[:, 0])) != int(np.sum(counts[:, 1]))
    ):
        _owner_group_fail_if_any(
            "Graph migration did not preserve the global owned-cell count.",
            execution_group,
            f"{axis_name}_cell_count_failure",
            "periodic graph migration cell coverage",
        )
    query_ids_by_owner: list[set[int]] = [set() for _ in range(parts)]
    for cell in cell_records.values():
        for identifier, owner in zip(
            cell["neighbors"], cell["neighbor_owners"], strict=True
        ):
            query_ids_by_owner[owner].add(identifier)
    neighbor_owner_by_id = _owner_local_periodic_graph_neighbor_owners(
        tuple(tuple(sorted(values)) for values in query_ids_by_owner),
        graph_result,
        rank,
        execution_group,
        f"{axis_name}_neighbor_owners",
    )
    vertex_records: dict[int, _PeriodicGraphVertexPayload] = {}
    local_error = None
    try:
        for cell in cell_records.values():
            for vertex in cell["vertices"]:
                previous = vertex_records.get(vertex["vertex_id"])
                if previous is not None and previous != vertex:
                    raise ValueError(
                        "Graph migration received conflicting quotient vertices."
                    )
                vertex_records[vertex["vertex_id"]] = vertex
                source_ids = {vertex["source_id"]}
                ancestry = vertex["coordinate_ancestry"]
                if ancestry is not None:
                    source_ids.update(ancestry["source_ids"])
                if not source_ids.issubset(source_point_records):
                    raise ValueError(
                        "Graph migration omitted an exact coordinate ancestry root."
                    )
            cell["neighbor_owners"] = [
                neighbor_owner_by_id[identifier] for identifier in cell["neighbors"]
            ]
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_geometry",
        "periodic graph migrated source/vertex binding",
    )
    target_cell_ids = np.asarray(sorted(cell_records), dtype=np.int64)
    target_vertex_ids = np.asarray(sorted(vertex_records), dtype=np.int64)
    empty_vertex_error = (
        "An empty graph owner received unowned vertex rows."
        if target_cell_ids.size == 0 and target_vertex_ids.size
        else None
    )
    _owner_group_fail_if_any(
        empty_vertex_error,
        execution_group,
        f"{axis_name}_empty_vertices",
        "periodic graph vertex owner coverage",
    )
    point_ids = np.asarray(sorted(source_point_records), dtype=np.int64)
    point_bits = np.asarray(
        [
            source_point_records[int(identifier)]["coordinate_bits"]
            for identifier in point_ids.tolist()
        ],
        dtype=np.uint64,
    ).reshape((-1, dimension))
    points = point_bits.view(np.float64)
    point_owner_ranks = np.asarray(
        [
            source_point_records[int(identifier)]["owner_rank"]
            for identifier in point_ids.tolist()
        ],
        dtype=np.int32,
    )
    local_error = None
    try:
        if any(
            int(source_id) not in source_point_records
            for cell in cell_records.values()
            for source_id in cell["key_source_ids"]
        ):
            raise ValueError("Graph migration omitted a cell-key source site.")
        from ._native_periodic_power import (
            _owner_local_periodic_site_weights_for_support,
        )

        migrated_site_weights = _owner_local_periodic_site_weights_for_support(
            source,
            point_ids,
            point_owner_ranks,
            rank,
            execution_group,
            axis_name=f"{axis_name}_site_weights",
        )
        new_source = NativePeriodicSource(
            PeriodicPointOrbits(source.domain.cell, points, point_ids),
            source.source_id,
            source.source_revision,
            site_weights=migrated_site_weights,
            point_owner_ranks=point_owner_ranks,
        )
        vertex_row_by_id = {
            int(identifier): row
            for row, identifier in enumerate(target_vertex_ids.tolist())
        }
        cells = np.asarray(
            [
                [
                    vertex_row_by_id[vertex["vertex_id"]]
                    for vertex in cell_records[int(cell_id)]["vertices"]
                ]
                for cell_id in target_cell_ids.tolist()
            ],
            dtype=np.int32,
        ).reshape((-1, arity))
        exact_vertex_coordinates = tuple(
            tuple(
                exact_pair(pair_value)
                for pair_value in vertex_records[int(identifier)]["exact_coordinates"]
            )
            for identifier in target_vertex_ids.tolist()
        )
        target_ancestry_rows: list[_PeriodicGraphVertexCoordinateAncestryPayload] = []
        for identifier in target_vertex_ids.tolist():
            ancestry_row = vertex_records[int(identifier)]["coordinate_ancestry"]
            if ancestry_row is None:
                raise ValueError(
                    "Migrated periodic vertices lost exact source interpolation ancestry."
                )
            target_ancestry_rows.append(ancestry_row)
        vertex_coordinate_bits = np.asarray(
            [
                vertex_records[int(identifier)]["coordinate_bits"]
                for identifier in target_vertex_ids.tolist()
            ],
            dtype=np.uint64,
        ).reshape((-1, dimension))
        vertex_coordinates = vertex_coordinate_bits.view(np.float64)
        expected_vertex_coordinates = np.asarray(
            tuple(
                tuple(float(value) for value in point)
                for point in exact_vertex_coordinates
            ),
            dtype=np.float64,
        ).reshape((-1, dimension))
        if not np.array_equal(vertex_coordinates, expected_vertex_coordinates):
            raise ValueError(
                "Migrated periodic vertex binary64 shadows differ from exact coordinates."
            )
        vertex_source_ids = np.asarray(
            [
                vertex_records[int(identifier)]["source_id"]
                for identifier in target_vertex_ids.tolist()
            ],
            dtype=np.int64,
        )
        vertex_owner_ranks = np.asarray(
            [
                vertex_records[int(identifier)]["owner_rank"]
                for identifier in target_vertex_ids.tolist()
            ],
            dtype=np.int32,
        )
        vertex_absolute_windings = np.asarray(
            [
                vertex_records[int(identifier)]["absolute_winding"]
                for identifier in target_vertex_ids.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, dimension))
        vertex_representative_ids = np.asarray(
            [
                vertex_records[int(identifier)]["representative_id"]
                for identifier in target_vertex_ids.tolist()
            ],
            dtype=np.int64,
        )
        vertex_windings = np.asarray(
            [
                vertex_records[int(identifier)]["quotient_winding"]
                for identifier in target_vertex_ids.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, dimension))
        exact_cell_measures = tuple(
            exact_pair(cell_records[int(identifier)]["measure"])
            for identifier in target_cell_ids.tolist()
        )
        cell_measures = np.asarray(
            [float(value) for value in exact_cell_measures], dtype=np.float64
        )
        cell_vertex_shifts = np.asarray(
            [
                cell_records[int(identifier)]["cell_vertex_shifts"]
                for identifier in target_cell_ids.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, arity, dimension))
        cell_windings = np.asarray(
            [
                cell_records[int(identifier)]["cell_winding"]
                for identifier in target_cell_ids.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, dimension))
        cell_key_offsets = np.zeros((target_cell_ids.size + 1,), dtype=np.int64)
        cell_key_sources: list[int] = []
        cell_key_windings: list[list[int]] = []
        for row, identifier in enumerate(target_cell_ids.tolist()):
            record = cell_records[int(identifier)]
            cell_key_sources.extend(record["key_source_ids"])
            cell_key_windings.extend(record["key_relative_windings"])
            cell_key_offsets[row + 1] = cell_key_offsets[row] + arity
        cell_key_source_ids = np.asarray(cell_key_sources, dtype=np.int64)
        cell_key_relative_windings = np.asarray(
            cell_key_windings, dtype=np.int64
        ).reshape((-1, dimension))
        if any(
            len(cell_records[int(identifier)]["support_lower_bits"]) != support_width
            for identifier in target_cell_ids.tolist()
        ):
            raise ValueError(
                "Periodic graph migration support-bound rows disagree with the "
                "source mesh support contract."
            )
        support_fractional_lower = (
            None
            if support_width == 0
            else np.asarray(
                [
                    cell_records[int(identifier)]["support_lower_bits"]
                    for identifier in target_cell_ids.tolist()
                ],
                dtype=np.uint64,
            )
            .reshape((-1, dimension))
            .view(np.float64)
        )
        support_fractional_upper = (
            None
            if support_width == 0
            else np.asarray(
                [
                    cell_records[int(identifier)]["support_upper_bits"]
                    for identifier in target_cell_ids.tolist()
                ],
                dtype=np.uint64,
            )
            .reshape((-1, dimension))
            .view(np.float64)
        )
        graph_cell_offsets = np.zeros((target_cell_ids.size + 1,), dtype=np.int64)
        graph_neighbor_ids: list[int] = []
        graph_neighbor_owners: list[int] = []
        graph_neighbor_weights: list[int] = []
        for row, identifier in enumerate(target_cell_ids.tolist()):
            record = cell_records[int(identifier)]
            order = np.argsort(
                np.asarray(record["neighbors"], dtype=np.int64), kind="stable"
            )
            graph_neighbor_ids.extend(
                np.asarray(record["neighbors"], dtype=np.int64)[order].tolist()
            )
            graph_neighbor_owners.extend(
                np.asarray(record["neighbor_owners"], dtype=np.int32)[order].tolist()
            )
            graph_neighbor_weights.extend(
                np.asarray(record["neighbor_weights"], dtype=np.int64)[order].tolist()
            )
            graph_cell_offsets[row + 1] = len(graph_neighbor_ids)
        graph_neighbor_id_array = np.asarray(graph_neighbor_ids, dtype=np.int64)
        graph_neighbor_owner_array = np.asarray(graph_neighbor_owners, dtype=np.int32)
        graph_neighbor_weight_array = np.asarray(graph_neighbor_weights, dtype=np.int64)
        cell_rows = np.arange(target_cell_ids.size, dtype=np.int64)
        vertex_rows = np.arange(target_vertex_ids.size, dtype=np.int64)
        local_mesh: CellMesh | None = None
        local_entity_ids: dict[int, np.ndarray] = {}
        entity_vertex_offsets: dict[int, np.ndarray] = {}
        entity_vertex_values: dict[int, np.ndarray] = {}
        cell_kind = "triangle" if dimension == 2 else "tetrahedron"
        element = lagrange_element(cell_kind, 1)
        blocks: tuple[dict[str, Any], ...] = ()
        if target_cell_ids.size:
            if target_vertex_ids.size == 0:
                raise ValueError("Nonempty graph cells require local quotient vertices.")
            block = CellBlock("periodic", cell_kind, cells, global_ids=target_cell_ids)
            local_mesh = CellMesh(
                vertex_coordinates,
                (block,),
                vertex_global_ids=target_vertex_ids,
                numeric_version="0",
            )
            entity_vertices = _entity_vertex_rows(local_mesh)
            for degree, rows_at_degree in enumerate(entity_vertices):
                entity_set = local_mesh.entity_set(degree)
                identifiers = np.asarray(entity_set.entity_ids, dtype=np.int64)
                if degree == 0:
                    ordered_rows = np.arange(target_vertex_ids.size, dtype=np.int64)
                elif degree == dimension:
                    ordered_rows = np.arange(target_cell_ids.size, dtype=np.int64)
                    if not np.array_equal(identifiers, target_cell_ids):
                        raise ValueError(
                            "Migrated graph cell IDs changed local row order."
                        )
                else:
                    ordered_rows = np.argsort(identifiers, kind="stable")
                local_entity_ids[degree] = identifiers[ordered_rows]
                offsets = np.zeros((ordered_rows.size + 1,), dtype=np.int64)
                values: list[np.ndarray] = []
                for local_row, entity_row in enumerate(ordered_rows):
                    vertices = np.asarray(rows_at_degree[int(entity_row)], dtype=np.int64)
                    values.append(vertices.astype(np.int32, copy=False))
                    offsets[local_row + 1] = offsets[local_row] + vertices.size
                entity_vertex_offsets[degree] = offsets
                entity_vertex_values[degree] = (
                    np.concatenate(values).astype(np.int32, copy=False)
                    if values
                    else np.zeros((0,), dtype=np.int32)
                )
            blocks = (
                {
                    "name": "periodic",
                    "cell_kind": cell_kind,
                    "cell_rows": cell_rows,
                    "cell_ids": target_cell_ids,
                    "vertices": cells,
                    "vertex_valid": np.ones(cells.shape, dtype=np.bool_),
                    "coordinate_routes": cells,
                    "element": element,
                    "element_rows": cell_rows.astype(np.int32),
                    "source_cell_ids": target_cell_ids,
                    "chart_arguments": None,
                    "cell_measures": cell_measures,
                    "face_ids": None,
                    "cell_face_offsets": None,
                    "cell_face_values": None,
                    "cell_face_signs": None,
                    "face_vertex_offsets": None,
                    "face_vertex_values": None,
                    "cell_vertex_shifts": cell_vertex_shifts,
                },
            )
        else:
            if target_vertex_ids.size:
                raise ValueError("Empty graph owners cannot retain quotient vertices.")
            for degree in range(dimension + 1):
                local_entity_ids[degree] = np.zeros((0,), dtype=np.int64)
                entity_vertex_offsets[degree] = np.zeros((1,), dtype=np.int64)
                entity_vertex_values[degree] = np.zeros((0,), dtype=np.int32)
        target_coordinate_stencil: VertexInterpolationStencil | None = None
        target_exact_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None = None
        if local_mesh is not None:
            target_coordinate_stencil, target_exact_coordinate_offsets = (
                _periodic_ancestry_stencil(
                    target_ancestry_rows,
                    target_vertex_ids,
                    local_mesh.entity_set(0).entity_set_id,
                    dimension,
                )
            )
            target_root_coordinates = tuple(
                (
                    int(point_ids[row]),
                    tuple(Fraction.from_float(float(value)) for value in points[row]),
                )
                for row in np.argsort(point_ids, kind="stable")
            )
            if (
                _evaluate_owner_local_periodic_coordinate_ancestry(
                    target_coordinate_stencil,
                    target_exact_coordinate_offsets,
                    target_root_coordinates,
                )
                != exact_vertex_coordinates
            ):
                raise ValueError(
                    "Migrated exact periodic coordinate ancestry does not reproduce target vertices."
                )
        new_mesh_data = dict(mesh_data)
        new_mesh_data.update(
            {
                "cell_rows": cell_rows,
                "cell_ids": target_cell_ids,
                "owned": np.ones(target_cell_ids.shape, dtype=np.bool_),
                "cell_measures": cell_measures,
                "exact_cell_measures": exact_cell_measures,
                "cell_owner_ranks": np.full(target_cell_ids.shape, rank, dtype=np.int32),
                "support_cell_mask": np.ones(target_cell_ids.shape, dtype=np.bool_),
                "vertices": cells,
                "vertex_valid": np.ones(cells.shape, dtype=np.bool_),
                "vertex_rows": vertex_rows,
                "vertex_ids": target_vertex_ids,
                "vertex_coordinates": vertex_coordinates,
                "exact_vertex_coordinates": exact_vertex_coordinates,
                "source_coordinate_stencil": target_coordinate_stencil,
                "exact_vertex_coordinate_offsets": (target_exact_coordinate_offsets),
                "vertex_source_ids": vertex_source_ids,
                "vertex_owner_ranks": vertex_owner_ranks,
                "vertex_absolute_windings": vertex_absolute_windings,
                "coordinate_rows": vertex_rows.copy(),
                "coordinates": vertex_coordinates.copy(),
                "exact_coordinates": exact_vertex_coordinates,
                "local_entity_ids": local_entity_ids,
                "entity_vertex_offsets": entity_vertex_offsets,
                "entity_vertex_values": entity_vertex_values,
                "cell_key_offsets": cell_key_offsets,
                "cell_key_source_ids": cell_key_source_ids,
                "cell_key_relative_windings": cell_key_relative_windings,
                "periodic_cell_ids": target_cell_ids.copy(),
                "vertex_representative_ids": vertex_representative_ids,
                "vertex_windings": vertex_windings,
                "cell_windings": cell_windings,
                "support_fractional_lower": support_fractional_lower,
                "support_fractional_upper": support_fractional_upper,
                "blocks": blocks,
                "local_mesh": local_mesh,
                "cell_kind": cell_kind,
                "cell_vertex_shifts": cell_vertex_shifts,
                "vertex_keys": tuple(
                    (
                        int(vertex_source_ids[row]),
                        tuple(
                            int(value) for value in vertex_absolute_windings[row].tolist()
                        ),
                    )
                    for row in range(target_vertex_ids.size)
                ),
                "graph_cell_ids": target_cell_ids.copy(),
                "graph_cell_owner_ranks": np.full(
                    target_cell_ids.shape, rank, dtype=np.int32
                ),
                "graph_cell_weights": np.ones(target_cell_ids.shape, dtype=np.int64),
                "graph_cell_offsets": graph_cell_offsets,
                "graph_neighbor_ids": graph_neighbor_id_array,
                "graph_neighbor_owner_ranks": graph_neighbor_owner_array,
                "graph_neighbor_weights": graph_neighbor_weight_array,
            }
        )
        new_domain = PeriodicPointOrbits(source.domain.cell, points, point_ids)
        new_source = NativePeriodicSource(
            new_domain,
            source.source_id,
            source.source_revision,
            point_owner_ranks=point_owner_ranks,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        new_source = source
        new_mesh_data = mesh_data
        target_cell_ids = np.empty((0,), dtype=np.int64)
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_materialize",
        "periodic graph migrated geometry",
    )
    return new_source, new_mesh_data, target_cell_ids


def _migrate_owner_local_periodic_power_source(
    source: NativePeriodicSource,
    point_ids: ArrayLike,
    point_owner_ranks: ArrayLike,
    source_owned: ArrayLike,
    graph_result: DistributedGraphPartitionResult,
    rank: int,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str,
) -> tuple[NativePeriodicSource, np.ndarray, np.ndarray]:
    import jax

    from ..._execution_runtime import ExecutionGroup
    from ...graph._distributed_partition import DistributedGraphPartitionResult
    from .._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
        _owner_local_multiset_identity,
    )
    from .._periodic import PeriodicPointOrbits

    members = execution_group.spec.process_indices
    parts = len(members)
    dimension = source.ambient_dimension
    local_error = None
    ids = np.empty((0,), dtype=np.int64)
    owners = np.empty((0,), dtype=np.int32)
    owned = np.empty((0,), dtype=np.bool_)
    points = np.empty((0, dimension), dtype=np.float64)
    weights: np.ndarray | None = None
    graph_ids = np.empty((0,), dtype=np.int64)
    graph_owners = np.empty((0,), dtype=np.int32)
    owned_ids = np.empty((0,), dtype=np.int64)
    domain: PeriodicPointOrbits | None = None
    try:
        if (
            not isinstance(source, NativePeriodicSource)
            or source.point_owner_ranks is None
            or not isinstance(graph_result, DistributedGraphPartitionResult)
            or not isinstance(execution_group, ExecutionGroup)
            or not execution_group.is_member
            or not 0 <= rank < parts
            or members[rank] != int(jax.process_index())
            or not isinstance(axis_name, str)
            or not axis_name.strip()
        ):
            raise TypeError(
                "Periodic power source migration requires current typed owners."
            )
        domain_candidate = source.domain
        if not isinstance(domain_candidate, PeriodicPointOrbits):
            raise TypeError("Periodic power source migration requires point orbits.")
        domain = domain_candidate
        raw_ids = np.asarray(point_ids)
        raw_owners = np.asarray(point_owner_ranks)
        raw_owned = np.asarray(source_owned)
        if (
            raw_ids.dtype.kind not in "iu"
            or raw_ids.ndim != 1
            or (
                raw_ids.dtype.kind == "u"
                and raw_ids.size
                and int(np.max(raw_ids)) > np.iinfo(np.int64).max
            )
            or raw_owners.dtype.kind not in "iu"
            or raw_owned.dtype != np.dtype(np.bool_)
        ):
            raise TypeError("Periodic power source site identity arrays are invalid.")
        ids = raw_ids.astype(np.int64, copy=False)
        owners = raw_owners.astype(np.int32, copy=False)
        owned = raw_owned.astype(np.bool_, copy=False)
        points = np.asarray(domain.points, dtype=np.float64)
        if source.site_weights is not None:
            weights = np.asarray(source.site_weights, dtype=np.float64)
        graph_ids = np.asarray(graph_result.entity_ids, dtype=np.int64)
        graph_owners = np.asarray(graph_result.owners, dtype=np.int32)
        owned_ids = np.sort(ids[owned], kind="stable")
        if (
            np.unique(ids).size != ids.size
            or owners.shape != ids.shape
            or owned.shape != ids.shape
            or not np.array_equal(owned, owners == rank)
            or not np.array_equal(
                np.asarray(source.point_owner_ranks, dtype=np.int32), owners
            )
            or points.shape != (ids.size, dimension)
            or np.any(~np.isfinite(points))
            or weights is not None
            and (weights.shape != ids.shape or np.any(~np.isfinite(weights)))
            or graph_ids.ndim != 1
            or graph_owners.shape != graph_ids.shape
            or np.unique(graph_ids).size != graph_ids.size
            or np.setdiff1d(graph_ids, owned_ids).size
            or np.any((graph_owners < 0) | (graph_owners >= parts))
            or not graph_result.feasible
        ):
            raise ValueError(
                "Periodic power graph labels differ from owned source sites."
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_input",
        "periodic power source ownership",
    )
    if domain is None:
        raise RuntimeError("Periodic power source validation lost its typed point bank.")

    weights_present = weights is not None
    schema_id = canonical_fingerprint(
        {
            "source": source.source_id,
            "revision": source.source_revision,
            "periodic_cell": domain.cell.cell_id,
            "dimension": dimension,
            "weights_present": weights_present,
            "graph_plan": graph_result.plan_id,
        }
    )
    schema_rows = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(schema_id), dtype=np.uint8),
        execution_group,
        f"{axis_name}_schema",
    )
    if any(
        np.asarray(value, dtype=np.uint8).tobytes().hex() != schema_id
        for value in schema_rows
    ):
        raise ValueError("Periodic power source migration schemas differ across ranks.")

    source_row_by_id = {
        int(identifier): row for row, identifier in enumerate(ids.tolist())
    }
    point_bits = points.view(np.uint64)
    weight_bits = None if weights is None else weights.view(np.uint64)
    graph_target_owner_by_id = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            graph_ids.tolist(), graph_owners.tolist(), strict=True
        )
    }
    target_owner_by_id: dict[int, tuple[int, bool]] = {}
    for identifier_value in owned_ids.tolist():
        identifier = int(identifier_value)
        target_owner = graph_target_owner_by_id.get(identifier)
        active = target_owner is not None
        if target_owner is None:
            digest = bytes.fromhex(
                canonical_fingerprint(("periodic-source-point-owner", identifier))
            )
            target_owner = digest[0] % parts
        target_owner_by_id[identifier] = (target_owner, active)

    width = 1 + dimension + int(weights_present) + 1
    outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_records: dict[int, tuple[tuple[int, ...], int | None, bool]] = {}
    local_error = None
    try:
        for identifier_value in owned_ids.tolist():
            identifier = int(identifier_value)
            source_row = source_row_by_id[identifier]
            target_owner, active = target_owner_by_id[identifier]
            coordinate_bits = tuple(int(value) for value in point_bits[source_row])
            weight_bit = None if weight_bits is None else int(weight_bits[source_row])
            record = (coordinate_bits, weight_bit, active)
            message = np.empty((width,), dtype=np.int64)
            message[0] = identifier
            message[1 : 1 + dimension] = np.asarray(
                coordinate_bits, dtype=np.uint64
            ).view(np.int64)
            if weight_bit is not None:
                message[1 + dimension] = np.asarray(weight_bit, dtype=np.uint64).view(
                    np.int64
                )
            message[-1] = int(active)
            if target_owner == rank:
                local_records[identifier] = record
            else:
                outgoing_rows[target_owner].append(message)
        outgoing = tuple(
            np.stack(rows) if rows else np.zeros((0, width), dtype=np.int64)
            for rows in outgoing_rows
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        outgoing = tuple(np.zeros((0, width), dtype=np.int64) for _ in range(parts))
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_encode",
        "periodic power source rows",
    )
    incoming = _owner_exchange_message_banks(
        outgoing, rank, execution_group, f"{axis_name}_rows"
    )
    local_error = None
    target_ids = np.empty((0,), dtype=np.int64)
    target_owners = np.empty((0,), dtype=np.int32)
    target_active_cell_ids = np.empty((0,), dtype=np.int64)
    target_points = np.empty((0, dimension), dtype=np.float64)
    target_weights = None if not weights_present else np.empty((0,), dtype=np.float64)
    try:
        for messages in incoming.values():
            bank = np.asarray(messages, dtype=np.int64)
            if bank.ndim != 2 or bank.shape[1] != width:
                raise ValueError(
                    "Periodic power site migration messages have wrong layout."
                )
            for message in bank:
                identifier = int(message[0])
                coordinate_bits = tuple(
                    int(value)
                    for value in message[1 : 1 + dimension].view(np.uint64).tolist()
                )
                weight_bit = (
                    None
                    if not weights_present
                    else int(
                        np.asarray(message[1 + dimension], dtype=np.int64).view(np.uint64)
                    )
                )
                active_value = int(message[-1])
                if active_value not in (0, 1) or identifier in local_records:
                    raise ValueError(
                        "Periodic power site migration duplicated or malformed a source ID."
                    )
                local_records[identifier] = (
                    coordinate_bits,
                    weight_bit,
                    bool(active_value),
                )
        target_ids = np.asarray(sorted(local_records), dtype=np.int64)
        target_owners = np.full(target_ids.shape, rank, dtype=np.int32)
        target_active_cell_ids = np.asarray(
            sorted(
                identifier for identifier, record in local_records.items() if record[2]
            ),
            dtype=np.int64,
        )
        target_points = (
            np.asarray(
                [local_records[int(identifier)][0] for identifier in target_ids.tolist()],
                dtype=np.uint64,
            )
            .reshape((-1, dimension))
            .view(np.float64)
        )
        if weights_present:
            target_weights = np.asarray(
                [local_records[int(identifier)][1] for identifier in target_ids.tolist()],
                dtype=np.uint64,
            ).view(np.float64)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        target_ids = np.empty((0,), dtype=np.int64)
        target_owners = np.empty((0,), dtype=np.int32)
        target_active_cell_ids = np.empty((0,), dtype=np.int64)
        target_points = np.empty((0, dimension), dtype=np.float64)
        target_weights = np.empty((0,), dtype=np.float64) if weights_present else None
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_decode",
        "periodic power source row coverage",
    )

    source_owner_records = tuple(
        (int(identifier), int(target_owner_by_id[int(identifier)][0]))
        for identifier in owned_ids.tolist()
    )
    target_owner_records = tuple(
        (int(identifier), rank) for identifier in target_ids.tolist()
    )
    source_value_records = tuple(
        (
            int(identifier),
            (
                tuple(
                    int(value) for value in point_bits[source_row_by_id[int(identifier)]]
                ),
                None
                if weight_bits is None
                else int(weight_bits[source_row_by_id[int(identifier)]]),
            ),
        )
        for identifier in owned_ids.tolist()
    )
    target_value_records = tuple(
        (
            int(identifier),
            (record[0], record[1]),
        )
        for identifier, record in sorted(local_records.items())
    )
    identity_header = {
        "periodic_cell": domain.cell.cell_id,
        "weights_present": weights_present,
        "dimension": dimension,
    }
    source_owner_id = _owner_local_multiset_identity(
        "periodic-power-site-owner-map",
        identity_header,
        source_owner_records,
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_source_owners",
    )
    target_owner_id = _owner_local_multiset_identity(
        "periodic-power-site-owner-map",
        identity_header,
        target_owner_records,
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_target_owners",
    )
    source_content_id = _owner_local_multiset_identity(
        "periodic-power-site-content",
        identity_header,
        source_value_records,
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_source_content",
    )
    target_content_id = _owner_local_multiset_identity(
        "periodic-power-site-content",
        identity_header,
        target_value_records,
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_target_content",
    )
    local_error = (
        None
        if source_owner_id == target_owner_id and source_content_id == target_content_id
        else "Periodic power repartition changed site ownership or exact source bits."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_identity",
        "periodic power source migration identity",
    )
    local_error = None
    try:
        migrated_source = NativePeriodicSource(
            PeriodicPointOrbits(domain.cell, target_points, target_ids),
            source.source_id,
            source.source_revision,
            site_weights=target_weights,
            point_owner_ranks=target_owners,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        migrated_source = source
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_source_build",
        "periodic power migrated source",
    )
    return migrated_source, target_ids, target_active_cell_ids


def _partition_owner_local_periodic_power_cells(
    mesh_data: dict[str, Any],
    plan: GraphPartitionPlan,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str,
) -> DistributedGraphPartitionResult:
    import jax

    from ..._execution_runtime import ExecutionGroup
    from ...graph._distributed_partition import (
        DistributedGraphPartitionResult,
        DistributedWeightedCSRGraph,
        partition_graph_distributed,
    )
    from ...graph._partition import GraphPartitionPlan
    from .._distribution import _owner_group_fail_if_any

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    members = execution_group.spec.process_indices
    try:
        rank = members.index(jax.process_index())
    except ValueError as error:
        raise ValueError(
            "The current process is outside the power graph group."
        ) from error
    local_error = None
    graph = None
    try:
        if (
            not execution_group.is_member
            or tuple(device.process_index for device in execution_group.devices)
            != members
            or not isinstance(plan, GraphPartitionPlan)
            or plan.part_count != len(members)
            or not isinstance(axis_name, str)
            or not axis_name.strip()
        ):
            raise ValueError("Periodic power graph plan and owner group differ.")
        graph = DistributedWeightedCSRGraph(
            mesh_data["graph_cell_offsets"],
            mesh_data["graph_neighbor_ids"],
            entity_ids=mesh_data["graph_cell_ids"],
            owner_ranks=mesh_data["graph_cell_owner_ranks"],
            neighbor_owner_ranks=mesh_data["graph_neighbor_owner_ranks"],
            edge_weights=mesh_data["graph_neighbor_weights"],
            vertex_weights=mesh_data["graph_cell_weights"],
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_input",
        "periodic power graph inputs",
    )
    if graph is None:
        raise RuntimeError("Periodic power graph construction returned no graph.")
    result = partition_graph_distributed(
        graph, plan, execution_group, axis_name=axis_name
    )
    local_error = None
    try:
        if (
            not isinstance(result, DistributedGraphPartitionResult)
            or not result.feasible
            or result.plan_id != plan.plan_id
            or not np.array_equal(
                np.asarray(result.entity_ids, dtype=np.int64),
                np.asarray(mesh_data["graph_cell_ids"], dtype=np.int64),
            )
            or np.asarray(result.owners).shape
            != np.asarray(mesh_data["graph_cell_ids"]).shape
        ):
            raise ValueError("Periodic power labels do not bind local owned cells.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_result",
        "periodic power graph partition",
    )
    del rank
    return result


def _owner_local_periodic_adaptive_mesh_data(
    adaptation: OwnerLocalPeriodicAdaptiveExecution,
    periodic_cell: PeriodicCell,
    rank: int,
    /,
) -> dict[str, Any]:
    from ...discretization._periodic_cell import PeriodicCell
    from ...discretization._periodic_topology import _entity_vertex_rows
    from ...discretization.fem._reference import lagrange_element
    from .._device_adaptation import OwnerLocalPeriodicAdaptiveExecution

    if not isinstance(adaptation, OwnerLocalPeriodicAdaptiveExecution):
        raise TypeError("adaptation must be OwnerLocalPeriodicAdaptiveExecution.")
    if not isinstance(periodic_cell, PeriodicCell):
        raise TypeError("periodic_cell must be PeriodicCell.")
    transition = adaptation.part_transition
    if transition is None:
        raise ValueError("Periodic adaptation has no committed local part transition.")
    dimension = periodic_cell.ambient_dimension
    metadata = adaptation.cell_metadata
    cell_ids = np.asarray(metadata.cell_ids, dtype=np.int64)
    cell_owner_ranks = np.asarray(transition.target_cell_owner_ranks, dtype=np.int32)
    exact_cell_measures = tuple(adaptation.exact_cell_measures)
    if (
        cell_ids.shape != cell_owner_ranks.shape
        or len(exact_cell_measures) != cell_ids.size
        or tuple(int(value) for value in transition.target_cell_ids)
        != tuple(int(value) for value in cell_ids)
    ):
        raise ValueError("Periodic target cell rows disagree with their exact metadata.")
    if np.any(cell_owner_ranks != rank):
        raise ValueError("A local periodic adaptive target contains a remote owned cell.")
    target_mesh = transition.target_mesh
    if (target_mesh is None) != (cell_ids.size == 0):
        raise ValueError("Periodic target mesh emptiness differs from its cell rows.")
    vertex_ids = np.asarray(adaptation.target_vertex_ids, dtype=np.int64)
    vertex_source_ids = np.asarray(adaptation.vertex_source_ids, dtype=np.int64)
    vertex_absolute_windings = np.asarray(
        adaptation.vertex_absolute_windings, dtype=np.int64
    )
    vertex_owner_ranks = np.asarray(adaptation.vertex_owner_ranks, dtype=np.int32)
    exact_vertex_coordinates = tuple(adaptation.exact_vertex_coordinates)
    if (
        vertex_source_ids.shape != vertex_ids.shape
        or vertex_owner_ranks.shape != vertex_ids.shape
        or vertex_absolute_windings.shape != (vertex_ids.size, dimension)
        or len(exact_vertex_coordinates) != vertex_ids.size
        or any(len(point) != dimension for point in exact_vertex_coordinates)
    ):
        raise ValueError("Periodic target vertex rows disagree with exact ancestry.")
    exact_coordinates_shadow = np.asarray(
        tuple(
            tuple(float(value) for value in point) for point in exact_vertex_coordinates
        ),
        dtype=np.float64,
    ).reshape((-1, dimension))
    if target_mesh is not None and (
        target_mesh.topological_dimension != dimension
        or not np.array_equal(
            np.asarray(target_mesh.vertex_global_ids, dtype=np.int64), vertex_ids
        )
        or not np.array_equal(
            np.asarray(target_mesh.coordinates, dtype=np.float64),
            exact_coordinates_shadow,
        )
    ):
        raise ValueError("Periodic target mesh differs from its exact vertex bank.")
    key_offsets = np.asarray(metadata.cell_key_offsets, dtype=np.int64)
    key_source_ids = np.asarray(metadata.cell_key_source_ids, dtype=np.int64)
    key_relative_windings = np.asarray(
        metadata.cell_key_relative_windings, dtype=np.int64
    )
    cell_windings = np.asarray(metadata.cell_windings, dtype=np.int64)
    if (
        key_offsets.shape != (cell_ids.size + 1,)
        or key_offsets[0] != 0
        or key_offsets[-1] != key_source_ids.size
        or key_relative_windings.shape != (key_source_ids.size, dimension)
        or cell_windings.shape != (cell_ids.size, dimension)
    ):
        raise ValueError("Periodic target quotient-cell keys are malformed.")
    cell_rows = np.arange(cell_ids.size, dtype=np.int64)
    cell_kind = "triangle" if dimension == 2 else "tetrahedron"
    element = lagrange_element(cell_kind, 1)
    geometry_layout_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-simplex-layout",
            "cell_kind": cell_kind,
            "element": element.element_id,
            "dimension": dimension,
        }
    )
    local_entity_ids: dict[int, np.ndarray] = {}
    entity_vertex_offsets: dict[int, np.ndarray] = {}
    entity_vertex_values: dict[int, np.ndarray] = {}
    blocks: tuple[dict[str, Any], ...] = ()
    vertices = np.zeros((0, dimension + 1), dtype=np.int32)
    vertex_valid = np.zeros(vertices.shape, dtype=np.bool_)
    if target_mesh is not None:
        target_entity_ids = np.asarray(
            target_mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        if not np.array_equal(target_entity_ids, cell_ids):
            raise ValueError("Periodic target mesh changed its canonical cell order.")
        row_by_id = {
            int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
        }
        vertices = np.full((cell_ids.size, dimension + 1), -1, dtype=np.int32)
        vertex_valid = np.zeros(vertices.shape, dtype=np.bool_)
        origins = np.asarray(transition.target_cell_origins, dtype=np.int64)
        if origins.shape != cell_ids.shape:
            raise ValueError("Periodic target lineage origins do not match its cells.")
        target_blocks: list[dict[str, Any]] = []
        exact_measure_values = np.asarray(
            tuple(float(value) for value in exact_cell_measures), dtype=np.float64
        )
        for block in target_mesh.blocks:
            block_ids = np.asarray(block.global_ids, dtype=np.int64)
            try:
                block_rows = np.asarray(
                    [row_by_id[int(identifier)] for identifier in block_ids],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    "Periodic target block contains an unknown cell ID."
                ) from error
            block_vertices = np.asarray(block.vertices, dtype=np.int64)
            if (
                block.cell_kind != cell_kind
                or block_vertices.shape != (block_ids.size, dimension + 1)
                or np.any(block_vertices < 0)
                or np.any(block_vertices >= vertex_ids.size)
            ):
                raise ValueError("Periodic target block is not a canonical simplex.")
            vertices[block_rows] = block_vertices.astype(np.int32)
            vertex_valid[block_rows] = True
            shifts = vertex_absolute_windings[block_vertices]
            target_blocks.append(
                {
                    "name": block.name,
                    "cell_kind": block.cell_kind,
                    "cell_rows": block_rows,
                    "cell_ids": block_ids,
                    "vertices": block_vertices.astype(np.int32),
                    "vertex_valid": np.ones(block_vertices.shape, dtype=np.bool_),
                    "coordinate_routes": block_vertices.astype(np.int32),
                    "element": element,
                    "element_rows": np.arange(block_ids.size, dtype=np.int32),
                    "source_cell_ids": origins[block_rows],
                    "chart_arguments": None,
                    "cell_measures": exact_measure_values[block_rows],
                    "face_ids": None,
                    "cell_face_offsets": None,
                    "cell_face_values": None,
                    "cell_face_signs": None,
                    "face_vertex_offsets": None,
                    "face_vertex_values": None,
                    "cell_vertex_shifts": shifts,
                }
            )
        if not target_blocks or np.any(~vertex_valid) or np.any(vertices < 0):
            raise ValueError("Periodic target blocks do not cover every target cell.")
        blocks = tuple(target_blocks)
        entity_vertices = _entity_vertex_rows(target_mesh)
        for degree, rows_at_degree in enumerate(entity_vertices):
            identifiers = np.asarray(
                target_mesh.entity_set(degree).entity_ids, dtype=np.int64
            )
            if degree == 0:
                order = np.arange(identifiers.size, dtype=np.int64)
                if not np.array_equal(identifiers, vertex_ids):
                    raise ValueError("Periodic target vertex entity order changed.")
            elif degree == dimension:
                order = np.arange(identifiers.size, dtype=np.int64)
                if not np.array_equal(identifiers, cell_ids):
                    raise ValueError("Periodic target cell entity order changed.")
            else:
                order = np.argsort(identifiers, kind="stable")
            local_entity_ids[degree] = identifiers[order]
            offsets = np.zeros((order.size + 1,), dtype=np.int64)
            values: list[np.ndarray] = []
            for local_row, entity_row in enumerate(order):
                values.append(np.asarray(rows_at_degree[int(entity_row)], dtype=np.int32))
                offsets[local_row + 1] = offsets[local_row] + values[-1].size
            entity_vertex_offsets[degree] = offsets
            entity_vertex_values[degree] = (
                np.concatenate(values).astype(np.int32, copy=False)
                if values
                else np.zeros((0,), dtype=np.int32)
            )
    else:
        for degree in range(dimension + 1):
            local_entity_ids[degree] = np.zeros((0,), dtype=np.int64)
            entity_vertex_offsets[degree] = np.zeros((1,), dtype=np.int64)
            entity_vertex_values[degree] = np.zeros((0,), dtype=np.int32)
    coordinates = (
        exact_coordinates_shadow
        if target_mesh is None
        else np.asarray(target_mesh.coordinates, dtype=np.float64)
    )
    return {
        "cell_rows": cell_rows,
        "cell_ids": cell_ids,
        "owned": cell_owner_ranks == rank,
        "cell_measures": np.asarray(
            tuple(float(value) for value in exact_cell_measures), dtype=np.float64
        ),
        "exact_cell_measures": exact_cell_measures,
        "cell_owner_ranks": cell_owner_ranks,
        "support_cell_mask": np.ones(cell_ids.shape, dtype=np.bool_),
        "vertices": vertices,
        "vertex_valid": vertex_valid,
        "vertex_rows": np.arange(vertex_ids.size, dtype=np.int64),
        "vertex_ids": vertex_ids,
        "vertex_coordinates": coordinates,
        "exact_vertex_coordinates": exact_vertex_coordinates,
        "vertex_source_ids": vertex_source_ids,
        "vertex_owner_ranks": vertex_owner_ranks,
        "vertex_absolute_windings": vertex_absolute_windings,
        "coordinate_rows": np.arange(vertex_ids.size, dtype=np.int64),
        "coordinates": coordinates.copy(),
        "exact_coordinates": exact_vertex_coordinates,
        "entity_set_ids": {},
        "local_entity_ids": local_entity_ids,
        "entity_vertex_offsets": entity_vertex_offsets,
        "entity_vertex_values": entity_vertex_values,
        "cell_key_offsets": key_offsets,
        "cell_key_source_ids": key_source_ids,
        "cell_key_relative_windings": key_relative_windings,
        "periodic_cell_ids": cell_ids.copy(),
        "vertex_representative_ids": vertex_source_ids.copy(),
        "vertex_windings": vertex_absolute_windings.copy(),
        "cell_windings": cell_windings,
        "support_fractional_lower": None,
        "support_fractional_upper": None,
        "geometry_layout_id": geometry_layout_id,
        "blocks": blocks,
        "local_mesh": target_mesh,
        "cell_kind": cell_kind,
        "cell_vertex_shifts": (
            vertex_absolute_windings[vertices]
            if vertices.size
            else np.zeros((0, dimension + 1, dimension), dtype=np.int64)
        ),
        "vertex_keys": tuple(
            (
                int(vertex_source_ids[row]),
                tuple(int(value) for value in vertex_absolute_windings[row]),
            )
            for row in range(vertex_ids.size)
        ),
    }


def _compose_owner_local_periodic_coordinate_ancestry(
    source_stencil: VertexInterpolationStencil | None,
    exact_source_offsets: tuple[tuple[Fraction, ...], ...],
    transition: OwnerLocalAdaptivePartTransition,
    /,
) -> tuple[VertexInterpolationStencil | None, tuple[tuple[Fraction, ...], ...]]:
    from .._device_adaptation import OwnerLocalAdaptivePartTransition
    from .._lineage import VertexInterpolationStencil

    if not isinstance(transition, OwnerLocalAdaptivePartTransition):
        raise TypeError("transition must be OwnerLocalAdaptivePartTransition.")
    target_mesh = transition.target_mesh
    source_mesh = transition.source_mesh
    if target_mesh is None:
        if transition.target_cell_ids.size or transition.target_vertex_parent_ids.size:
            raise ValueError("Empty periodic target retains source transition rows.")
        if source_stencil is None and exact_source_offsets:
            raise ValueError("An empty periodic source has nonempty coordinate offsets.")
        return None, ()
    if source_mesh is None:
        raise ValueError("A nonempty periodic target lacks its source mesh.")
    if not isinstance(source_stencil, VertexInterpolationStencil):
        raise TypeError(
            "Nonempty periodic transitions require cumulative vertex ancestry."
        )
    if not isinstance(exact_source_offsets, tuple):
        raise TypeError("exact_source_offsets must be an immutable tuple.")
    source_ids = np.asarray(source_stencil.target_global_ids, dtype=np.int64)
    source_global_ids = np.asarray(source_stencil.source_global_ids, dtype=np.int64)
    source_weights = np.asarray(source_stencil.weights, dtype=np.float64)
    source_valid = np.asarray(source_stencil.valid, dtype=np.bool_)
    if (
        source_global_ids.ndim != 2
        or source_weights.shape != source_global_ids.shape
        or source_valid.shape != source_global_ids.shape
        or len(source_stencil.exact_weights) != source_ids.size
        or len(exact_source_offsets) != source_ids.size
        or any(
            any(not isinstance(value, Fraction) for value in point)
            for point in exact_source_offsets
        )
    ):
        raise ValueError("Cumulative periodic source-coordinate ancestry is malformed.")
    ambient_dimension = len(exact_source_offsets[0]) if exact_source_offsets else 0
    if ambient_dimension <= 0 or any(
        len(point) != ambient_dimension for point in exact_source_offsets
    ):
        raise ValueError(
            "Cumulative periodic source offsets have inconsistent dimensions."
        )
    source_row_by_vertex = {
        int(identifier): row for row, identifier in enumerate(source_ids.tolist())
    }
    current_source_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    try:
        source_rows = np.asarray(
            [source_row_by_vertex[int(identifier)] for identifier in current_source_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "Adaptive source mesh lost cumulative coordinate ancestry."
        ) from error
    current_exact_rows = tuple(
        source_stencil.exact_weights[int(row)] for row in source_rows
    )
    current_offsets = tuple(exact_source_offsets[int(row)] for row in source_rows)
    current_stencil = VertexInterpolationStencil(
        source_stencil.source_entity_set_id,
        source_mesh.entity_set(0).entity_set_id,
        current_source_ids,
        source_global_ids[source_rows],
        source_weights[source_rows],
        source_valid[source_rows],
        exact_weights=current_exact_rows,
    )
    current_row_by_vertex = {
        int(identifier): row for row, identifier in enumerate(current_source_ids.tolist())
    }
    transition_stencil = transition.vertex_stencil
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    if transition_stencil is None:
        try:
            selected = np.asarray(
                [current_row_by_vertex[int(identifier)] for identifier in target_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "Unchanged periodic target introduced a vertex without ancestry."
            ) from error
        composed_rows = tuple(
            {
                int(current_stencil.source_global_ids[row, column]): exact_weight
                for column, (exact_weight, valid) in enumerate(
                    zip(
                        current_stencil.exact_weights[row],
                        np.asarray(current_stencil.valid)[row],
                        strict=True,
                    )
                )
                if valid
            }
            for row in selected
        )
        composed_offsets = tuple(current_offsets[int(row)] for row in selected)
        target_entity_set_id = target_mesh.entity_set(0).entity_set_id
    else:
        if (
            transition_stencil.source_entity_set_id
            != current_stencil.target_entity_set_id
            or transition_stencil.target_entity_set_id
            != target_mesh.entity_set(0).entity_set_id
            or not np.array_equal(
                np.asarray(transition_stencil.target_global_ids, dtype=np.int64),
                target_ids,
            )
        ):
            raise ValueError(
                "Periodic transition stencil does not bind its local meshes."
            )
        intermediate_sources = np.asarray(
            transition_stencil.source_global_ids, dtype=np.int64
        )
        intermediate_valid = np.asarray(transition_stencil.valid, dtype=np.bool_)
        intermediate_exact = transition_stencil.exact_weights
        cumulative_sources: list[dict[int, Fraction]] = []
        cumulative_offsets: list[tuple[Fraction, ...]] = []
        current_global_sources = np.asarray(
            current_stencil.source_global_ids, dtype=np.int64
        )
        current_valid = np.asarray(current_stencil.valid, dtype=np.bool_)
        for target_row in range(target_ids.size):
            coefficients: dict[int, Fraction] = {}
            offset = [Fraction(0) for _ in range(ambient_dimension)]
            for source_vertex_id, exact_weight, valid in zip(
                intermediate_sources[target_row],
                intermediate_exact[target_row],
                intermediate_valid[target_row],
                strict=True,
            ):
                if not valid:
                    continue
                source_vertex_row = current_row_by_vertex.get(int(source_vertex_id))
                if source_vertex_row is None:
                    raise ValueError(
                        "Periodic transition stencil references an unknown source vertex."
                    )
                for source_id, source_fraction, source_active in zip(
                    current_global_sources[source_vertex_row],
                    current_stencil.exact_weights[source_vertex_row],
                    current_valid[source_vertex_row],
                    strict=True,
                ):
                    if source_active:
                        key = int(source_id)
                        coefficients[key] = (
                            coefficients.get(key, Fraction(0))
                            + exact_weight * source_fraction
                        )
                for axis, value in enumerate(current_offsets[source_vertex_row]):
                    offset[axis] += exact_weight * value
            coefficients = {
                identifier: value
                for identifier, value in coefficients.items()
                if value != 0
            }
            if not coefficients or sum(coefficients.values(), Fraction(0)) != 1:
                raise ValueError(
                    "Composed periodic source-coordinate rows must preserve constants."
                )
            cumulative_sources.append(coefficients)
            cumulative_offsets.append(tuple(offset))
        composed_rows = tuple(cumulative_sources)
        composed_offsets = tuple(cumulative_offsets)
        target_entity_set_id = transition_stencil.target_entity_set_id
    if len(composed_rows) != target_ids.size or len(composed_offsets) != target_ids.size:
        raise ValueError(
            "Composed periodic coordinate rows do not match target vertices."
        )
    width = max((len(row) for row in composed_rows), default=0)
    if target_ids.size and width == 0:
        raise ValueError("A nonempty periodic target has no source-coordinate ancestry.")
    width = max(width, 1)
    output_sources = np.full((target_ids.size, width), -1, dtype=np.int64)
    output_weights = np.zeros((target_ids.size, width), dtype=np.float64)
    output_valid = np.zeros((target_ids.size, width), dtype=np.bool_)
    exact_rows: list[tuple[Fraction, ...]] = []
    for row, coefficients in enumerate(composed_rows):
        ordered = tuple(sorted(coefficients.items()))
        exact_row = [Fraction(0) for _ in range(width)]
        for column, (identifier, value) in enumerate(ordered):
            output_sources[row, column] = identifier
            output_weights[row, column] = float(value)
            output_valid[row, column] = True
            exact_row[column] = value
        exact_rows.append(tuple(exact_row))
    composed_stencil = VertexInterpolationStencil(
        source_stencil.source_entity_set_id,
        target_entity_set_id,
        target_ids,
        output_sources,
        output_weights,
        output_valid,
        exact_weights=tuple(exact_rows),
    )
    return composed_stencil, composed_offsets


def _initial_owner_local_periodic_coordinate_ancestry(
    source: NativePeriodicSource,
    mesh_data: dict[str, Any],
    periodic_cell: PeriodicCell,
    /,
) -> tuple[VertexInterpolationStencil | None, tuple[tuple[Fraction, ...], ...]]:
    from ...discretization._periodic_cell import PeriodicCell
    from .._lineage import VertexInterpolationStencil

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or not isinstance(periodic_cell, PeriodicCell)
        or source.domain.cell.cell_id != periodic_cell.cell_id
    ):
        raise TypeError(
            "Initial periodic coordinate ancestry requires typed source operands."
        )
    vertex_ids = np.asarray(mesh_data["vertex_ids"], dtype=np.int64)
    if not vertex_ids.size:
        if mesh_data["local_mesh"] is not None:
            raise ValueError("Empty periodic source vertex bank has a nonempty mesh.")
        return None, ()
    mesh = mesh_data["local_mesh"]
    if not isinstance(mesh, CellMesh):
        raise TypeError("Nonempty periodic coordinate ancestry requires CellMesh.")
    roots = np.asarray(mesh_data["vertex_source_ids"], dtype=np.int64)
    windings = np.asarray(mesh_data["vertex_absolute_windings"], dtype=np.int64)
    exact_vertices = tuple(mesh_data["exact_vertex_coordinates"])
    point_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
    points = np.asarray(source.domain.points, dtype=np.float64)
    if (
        roots.shape != vertex_ids.shape
        or windings.shape != (vertex_ids.size, periodic_cell.vectors.shape[0])
        or len(exact_vertices) != vertex_ids.size
        or point_ids.shape != points.shape[:1]
        or points.shape != (point_ids.size, periodic_cell.ambient_dimension)
        or np.unique(point_ids).size != point_ids.size
    ):
        raise ValueError("Initial periodic coordinate ancestry banks are misaligned.")
    point_row_by_id = {
        int(identifier): row for row, identifier in enumerate(point_ids.tolist())
    }
    exact_points = tuple(
        tuple(Fraction.from_float(float(value)) for value in point) for point in points
    )
    lattice_exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in np.asarray(periodic_cell.vectors, dtype=np.float64)
    )
    offsets: list[tuple[Fraction, ...]] = []
    for row, root in enumerate(roots):
        source_row = point_row_by_id.get(int(root))
        if source_row is None:
            raise ValueError("Periodic lifted vertex references an unknown source site.")
        offset = tuple(
            sum(
                (
                    int(windings[row, generator]) * lattice_exact[generator][axis]
                    for generator in range(len(lattice_exact))
                ),
                Fraction(0),
            )
            for axis in range(periodic_cell.ambient_dimension)
        )
        expected = tuple(
            exact_points[source_row][axis] + offset[axis]
            for axis in range(periodic_cell.ambient_dimension)
        )
        if tuple(exact_vertices[row]) != expected:
            raise ValueError(
                "Native periodic vertex lost its exact source-site/deck ancestry."
            )
        offsets.append(offset)
    source_ids = roots.reshape((-1, 1))
    weights = np.ones(source_ids.shape, dtype=np.float64)
    valid = np.ones(source_ids.shape, dtype=np.bool_)
    stencil = VertexInterpolationStencil(
        source.scope_entity_set_id(0),
        mesh.entity_set(0).entity_set_id,
        vertex_ids,
        source_ids,
        weights,
        valid,
        exact_weights=tuple((Fraction(1),) for _ in range(vertex_ids.size)),
    )
    return stencil, tuple(offsets)


def _evaluate_owner_local_periodic_coordinate_ancestry(
    stencil: VertexInterpolationStencil,
    exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...],
    exact_root_coordinates: tuple[tuple[int, tuple[Fraction, ...]], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    from .._lineage import VertexInterpolationStencil

    if not isinstance(stencil, VertexInterpolationStencil):
        raise TypeError("stencil must be VertexInterpolationStencil.")
    if not stencil.preserves_constants:
        raise ValueError("Periodic source-coordinate ancestry must preserve constants.")
    roots: dict[int, tuple[Fraction, ...]] = {}
    previous_id = -1
    for identifier, point in exact_root_coordinates:
        if (
            isinstance(identifier, (bool, np.bool_))
            or not isinstance(identifier, (int, np.integer))
            or int(identifier) <= previous_id
            or not isinstance(point, tuple)
            or not point
            or any(not isinstance(value, Fraction) for value in point)
        ):
            raise ValueError(
                "Exact periodic root coordinates are not canonically ordered."
            )
        key = int(identifier)
        roots[key] = point
        previous_id = key
    if not roots:
        raise ValueError("Periodic coordinate ancestry requires exact root coordinates.")
    offsets = tuple(tuple(point) for point in exact_vertex_coordinate_offsets)
    if len(offsets) != len(stencil.target_global_ids):
        raise ValueError("Exact periodic coordinate offsets do not match stencil rows.")
    dimension = len(next(iter(roots.values())))
    if any(len(point) != dimension for point in roots.values()) or any(
        len(point) != dimension or any(not isinstance(value, Fraction) for value in point)
        for point in offsets
    ):
        raise ValueError("Exact periodic root/offset dimensions are inconsistent.")
    source_ids = np.asarray(stencil.source_global_ids, dtype=np.int64)
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    if source_ids.shape != valid.shape or source_ids.ndim != 2:
        raise ValueError("Periodic coordinate stencil rows are malformed.")
    coordinates: list[tuple[Fraction, ...]] = []
    for row in range(source_ids.shape[0]):
        exact_weights = stencil.exact_weights[row]
        if len(exact_weights) != source_ids.shape[1]:
            raise ValueError("Periodic coordinate weights do not align with source rows.")
        if (
            sum(
                (
                    weight
                    for weight, active in zip(exact_weights, valid[row], strict=True)
                    if active
                ),
                Fraction(0),
            )
            != 1
        ):
            raise ValueError(
                "Periodic exact coordinate stencil row does not preserve constants."
            )
        coordinate = list(offsets[row])
        for identifier, weight, active in zip(
            source_ids[row], exact_weights, valid[row], strict=True
        ):
            if not active:
                continue
            source_point = roots.get(int(identifier))
            if source_point is None:
                raise ValueError(
                    "Periodic coordinate stencil references an unknown source root."
                )
            for axis, value in enumerate(source_point):
                coordinate[axis] += weight * value
        coordinates.append(tuple(coordinate))
    return tuple(coordinates)


def _periodic_vertex_ancestry_payload(
    stencil: VertexInterpolationStencil,
    row: int,
    exact_offsets: tuple[Fraction, ...],
    /,
) -> _PeriodicGraphVertexCoordinateAncestryPayload:
    """Encode one vertex's active exact source row, independent of its sender.

    A stencil's target entity set and padded row width belong to the resident
    mesh of one rank, so every rank sharing a vertex must publish only the
    vertex's canonical active row (sorted by source ID) and exact offset.
    """
    source_ids = np.asarray(stencil.source_global_ids, dtype=np.int64)[row]
    valid = np.asarray(stencil.valid, dtype=np.bool_)[row]
    weight_bits = np.asarray(stencil.weights, dtype=np.float64)[row].view(np.uint64)
    exact_weights = stencil.exact_weights[row]
    if any(not isinstance(value, Fraction) for value in exact_weights) or any(
        not isinstance(value, Fraction) for value in exact_offsets
    ):
        raise TypeError("Periodic vertex ancestry requires exact rational rows.")
    active = sorted(
        (
            (int(identifier), int(bits), exact)
            for identifier, bits, exact, is_valid in zip(
                source_ids, weight_bits, exact_weights, valid, strict=True
            )
            if is_valid
        ),
        key=lambda entry: entry[0],
    )
    return {
        "source_entity_set_id": stencil.source_entity_set_id,
        "preserves_constants": stencil.preserves_constants,
        "source_ids": [identifier for identifier, _, _ in active],
        "weight_bits": [bits for _, bits, _ in active],
        "exact_weights": [
            [int(exact.numerator), int(exact.denominator)] for _, _, exact in active
        ],
        "exact_offsets": [
            [int(value.numerator), int(value.denominator)] for value in exact_offsets
        ],
    }


def _periodic_ancestry_stencil(
    rows: Sequence[_PeriodicGraphVertexCoordinateAncestryPayload],
    target_vertex_ids: np.ndarray,
    target_entity_set_id: str,
    dimension: int,
    /,
) -> tuple[VertexInterpolationStencil, tuple[tuple[Fraction, ...], ...]]:
    """Bind canonical vertex ancestry rows to the receiving rank's resident mesh."""
    from .._lineage import VertexInterpolationStencil

    def exact_value(raw: object, /) -> Fraction:
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
            or raw[1] <= 0
        ):
            raise ValueError("Periodic exact ancestry scalar is malformed.")
        return Fraction(raw[0], raw[1])

    if not rows or len(rows) != target_vertex_ids.size:
        raise ValueError("Periodic ancestry rows do not cover the resident vertices.")
    metadata = (rows[0]["source_entity_set_id"], rows[0]["preserves_constants"])
    if any(
        (row["source_entity_set_id"], row["preserves_constants"]) != metadata
        for row in rows
    ):
        raise ValueError("Periodic vertex ancestry rows disagree on source identity.")
    width = max(len(row["source_ids"]) for row in rows)
    if width == 0:
        raise ValueError("Periodic vertex ancestry has no active source rows.")
    source_ids = np.full((len(rows), width), -1, dtype=np.int64)
    weight_bits = np.zeros((len(rows), width), dtype=np.uint64)
    valid = np.zeros((len(rows), width), dtype=np.bool_)
    exact_weights: list[tuple[Fraction, ...]] = []
    exact_offsets: list[tuple[Fraction, ...]] = []
    for index, row in enumerate(rows):
        count = len(row["source_ids"])
        if (
            count == 0
            or len(row["weight_bits"]) != count
            or len(row["exact_weights"]) != count
            or len(row["exact_offsets"]) != dimension
            or any(
                second <= first
                for first, second in zip(
                    row["source_ids"], row["source_ids"][1:], strict=False
                )
            )
        ):
            raise ValueError("Periodic vertex ancestry row is not canonical.")
        source_ids[index, :count] = row["source_ids"]
        weight_bits[index, :count] = row["weight_bits"]
        valid[index, :count] = True
        exact_weights.append(
            tuple(exact_value(value) for value in row["exact_weights"])
            + (Fraction(0),) * (width - count)
        )
        exact_offsets.append(tuple(exact_value(value) for value in row["exact_offsets"]))
    stencil = VertexInterpolationStencil(
        metadata[0],
        target_entity_set_id,
        target_vertex_ids,
        source_ids,
        weight_bits.view(np.float64),
        valid,
        preserves_constants=metadata[1],
        exact_weights=tuple(exact_weights),
    )
    return stencil, tuple(exact_offsets)


def _owner_local_periodic_delaunay_view_mesh_data(
    view: OwnerLocalCellGeometry,
    source_authority: NativePeriodicSource,
    periodic_cell: PeriodicCell,
    /,
) -> dict[str, Any]:
    from ...discretization._periodic_cell import PeriodicCell
    from ...discretization._view_support import (
        _materialize_owner_local_periodic_target,
        OwnerLocalCellGeometry,
    )
    from .._periodic import PeriodicPointOrbits

    if (
        not isinstance(view, OwnerLocalCellGeometry)
        or not isinstance(source_authority, NativePeriodicSource)
        or not isinstance(source_authority.domain, PeriodicPointOrbits)
        or not isinstance(periodic_cell, PeriodicCell)
        or source_authority.domain.cell.cell_id != periodic_cell.cell_id
    ):
        raise TypeError(
            "Periodic target replay requires typed source/view/lattice values."
        )
    domain = source_authority.domain
    point_ids = np.asarray(domain.point_ids, dtype=np.int64)
    point_coordinates = np.asarray(domain.points, dtype=np.float64)
    point_order = np.argsort(point_ids, kind="stable")
    exact_root_coordinates = tuple(
        (
            int(point_ids[row]),
            tuple(Fraction.from_float(float(value)) for value in point_coordinates[row]),
        )
        for row in point_order
    )
    root_float_coordinates = point_coordinates[point_order]
    materialized = _materialize_owner_local_periodic_target(
        view,
        periodic_cell,
        exact_root_coordinates,
        root_float_coordinates,
        source_authority.scope_entity_set_id(0),
        np.asarray(view.vertex_source_ids, dtype=np.int64),
        np.asarray(view.vertex_absolute_windings, dtype=np.int64),
    )
    mesh = materialized.mesh
    geometry = materialized.geometry
    cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
    vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
    dimension = view.topological_dimension
    exact_cell_measures = materialized.exact_cell_measures
    expected_exact_vertices = (
        ()
        if view.exact_vertex_coordinates is None
        else tuple(tuple(point) for point in view.exact_vertex_coordinates)
    )
    if view.adaptive_result is None:
        accepted_cell_measure_match = np.array_equal(
            np.asarray(view.cell_measures, dtype=np.float64),
            np.asarray(
                tuple(float(value) for value in exact_cell_measures),
                dtype=np.float64,
            ),
        )
        accepted_cell_id_match = np.array_equal(
            np.asarray(view.cell_ids, dtype=np.int64), cell_ids
        )
    else:
        # The adaptive result binds this owner's target cells; the compact view
        # adds read-only halo cells around them.
        owned_rows = np.flatnonzero(np.asarray(view.owned, dtype=np.bool_))
        measure_by_id = dict(
            zip(
                (int(value) for value in view.adaptive_result.target_cell_ids),
                tuple(view.adaptive_result.target_cell_measures),
                strict=True,
            )
        )
        accepted_cell_id_match = len(measure_by_id) == owned_rows.size and set(
            measure_by_id
        ) == {int(cell_ids[row]) for row in owned_rows}
        accepted_cell_measure_match = accepted_cell_id_match and all(
            measure_by_id[int(cell_ids[row])] == exact_cell_measures[int(row)]
            for row in owned_rows
        )
    if (
        len(exact_cell_measures) != cell_ids.size
        or not accepted_cell_id_match
        or not accepted_cell_measure_match
    ):
        adaptive_ids = (
            None
            if view.adaptive_result is None
            else np.asarray(view.adaptive_result.target_cell_ids, dtype=np.int64)
        )
        raise ValueError(
            "Stored periodic target cells differ from their exact cell measures "
            f"({len(exact_cell_measures)} measures for {cell_ids.size} cells; "
            f"cell IDs match: {accepted_cell_id_match}; "
            f"measures match: {accepted_cell_measure_match}; adaptive target "
            f"cells: {None if adaptive_ids is None else adaptive_ids.tolist()}; "
            f"view cells: {cell_ids.tolist()})."
        )
    if materialized.exact_vertex_coordinates != expected_exact_vertices or not (
        np.array_equal(
            materialized.vertex_coordinates,
            np.asarray(view.vertex_coordinates),
        )
    ):
        raise ValueError(
            "Stored periodic target vertices differ from their exact coordinates."
        )
    view_stencil = view.source_coordinate_stencil
    view_offsets = view.exact_vertex_coordinate_offsets
    if vertex_ids.size == 0:
        # A real empty owner has vacuous coordinate ancestry: it may store
        # either no ancestry or an ancestry over zero target rows.
        ancestry_matches = (
            materialized.source_coordinate_stencil is None
            and (
                view_stencil is None
                or np.asarray(view_stencil.target_global_ids).size == 0
            )
            and not materialized.exact_vertex_coordinate_offsets
            and not view_offsets
        )
    else:
        ancestry_matches = (
            (materialized.source_coordinate_stencil is None) == (view_stencil is None)
            and (
                view_stencil is None
                or materialized.source_coordinate_stencil is None
                or materialized.source_coordinate_stencil.stencil_id
                == view_stencil.stencil_id
            )
            and materialized.exact_vertex_coordinate_offsets == view_offsets
        )
    if not ancestry_matches:
        raise ValueError(
            "Stored periodic target differs from its exact coordinate ancestry."
        )
    if (mesh is None) != (cell_ids.size == 0) or (geometry is None) != (
        cell_ids.size == 0
    ):
        raise ValueError("Periodic target geometry emptiness differs from its cell bank.")
    if mesh is not None and (
        not np.array_equal(
            np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64),
            cell_ids,
        )
        or not np.array_equal(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64), vertex_ids
        )
    ):
        raise ValueError("Periodic target mesh changed its canonical entity rows.")
    vertex_count = vertex_ids.size
    cell_rows = np.arange(cell_ids.size, dtype=np.int64)
    cell_kind = "triangle" if dimension == 2 else "tetrahedron"
    vertices = np.full((cell_ids.size, dimension + 1), -1, dtype=np.int32)
    vertex_valid = np.zeros(vertices.shape, dtype=np.bool_)
    block_records: list[dict[str, Any]] = []
    row_covered = np.zeros((cell_ids.size,), dtype=np.bool_)
    for block in materialized.blocks:
        block_rows = np.asarray(block.cell_rows, dtype=np.int64)
        block_ids = np.asarray(block.cell_ids, dtype=np.int64)
        block_vertices = np.asarray(block.vertices, dtype=np.int32)
        block_valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        if (
            block.cell_kind != cell_kind
            or block_rows.shape != block_ids.shape
            or block_vertices.shape != (block_ids.size, dimension + 1)
            or block_valid.shape != block_vertices.shape
            or np.any(block_rows < 0)
            or np.any(block_rows >= cell_ids.size)
            or np.any(row_covered[block_rows])
            or not np.array_equal(cell_ids[block_rows], block_ids)
            or not np.all(block_valid)
        ):
            raise ValueError("Adaptive periodic target is not a complete simplex view.")
        vertices[block_rows] = block_vertices
        vertex_valid[block_rows] = block_valid
        row_covered[block_rows] = True
        if block.cell_vertex_shifts is None:
            raise ValueError("Periodic target simplex block lost its lifted image rows.")
        block_records.append(
            {
                "name": block.name,
                "cell_kind": block.cell_kind,
                "cell_rows": block_rows,
                "cell_ids": block_ids,
                "vertices": block_vertices,
                "vertex_valid": block_valid,
                "coordinate_routes": np.asarray(block.coordinate_routes, dtype=np.int32),
                "element": block.element,
                "element_rows": np.asarray(block.element_rows, dtype=np.int32),
                "source_cell_ids": np.asarray(block.source_cell_ids, dtype=np.int64),
                "chart_arguments": block.chart_arguments,
                "cell_measures": np.asarray(block.cell_measures, dtype=np.float64),
                "face_ids": None,
                "cell_face_offsets": None,
                "cell_face_values": None,
                "cell_face_signs": None,
                "face_vertex_offsets": None,
                "face_vertex_values": None,
                "cell_vertex_shifts": np.asarray(
                    block.cell_vertex_shifts, dtype=np.int64
                ),
            }
        )
    if cell_ids.size and (not np.all(row_covered) or not block_records):
        raise ValueError("Adaptive periodic target blocks do not cover all local cells.")
    exact_vertices = materialized.exact_vertex_coordinates
    if len(exact_vertices) != vertex_count:
        raise ValueError("Periodic target exact vertex bank is incomplete.")
    measures = np.asarray(
        tuple(float(value) for value in exact_cell_measures), dtype=np.float64
    )
    return {
        "cell_rows": cell_rows,
        "cell_ids": cell_ids,
        "owned": np.asarray(view.owned, dtype=np.bool_),
        "cell_measures": measures,
        "exact_cell_measures": exact_cell_measures,
        "cell_owner_ranks": np.asarray(view.cell_owner_ranks, dtype=np.int32),
        "support_cell_mask": np.asarray(view.support_cell_mask, dtype=np.bool_),
        "vertices": vertices,
        "vertex_valid": vertex_valid,
        "vertex_rows": np.asarray(view.vertex_rows, dtype=np.int64),
        "vertex_ids": vertex_ids,
        "vertex_coordinates": materialized.vertex_coordinates,
        "exact_vertex_coordinates": materialized.exact_vertex_coordinates,
        "vertex_source_ids": materialized.vertex_source_ids,
        "vertex_owner_ranks": np.asarray(view.vertex_owner_ranks, dtype=np.int32),
        "vertex_absolute_windings": materialized.vertex_absolute_windings,
        "coordinate_rows": np.asarray(view.coordinate_rows, dtype=np.int64),
        "coordinates": materialized.vertex_coordinates.copy(),
        "exact_coordinates": materialized.exact_vertex_coordinates,
        "entity_set_ids": dict(view.entity_set_ids),
        "local_entity_ids": dict(view.local_entity_ids),
        "entity_vertex_offsets": dict(view.entity_vertex_offsets),
        "entity_vertex_values": dict(view.entity_vertex_values),
        "cell_key_offsets": materialized.cell_key_offsets,
        "cell_key_source_ids": materialized.cell_key_source_ids,
        "cell_key_relative_windings": materialized.cell_key_relative_windings,
        "periodic_cell_ids": np.asarray(view.periodic_cell_ids, dtype=np.int64),
        "vertex_representative_ids": materialized.vertex_source_ids.copy(),
        "vertex_windings": materialized.vertex_absolute_windings.copy(),
        "cell_windings": materialized.cell_windings,
        "support_fractional_lower": (
            None
            if view.support_fractional_lower is None
            else np.asarray(view.support_fractional_lower, dtype=np.float64)
        ),
        "support_fractional_upper": (
            None
            if view.support_fractional_upper is None
            else np.asarray(view.support_fractional_upper, dtype=np.float64)
        ),
        "geometry_layout_id": view.geometry_layout_id,
        "graph_cell_ids": np.asarray(view.owned_cell_ids, dtype=np.int64),
        "graph_cell_owner_ranks": np.asarray(view.owned_cell_owner_ranks, dtype=np.int32),
        "graph_cell_weights": np.asarray(view.owned_cell_weights, dtype=np.int64),
        "graph_cell_offsets": np.asarray(view.cell_neighbor_offsets, dtype=np.int64),
        "graph_neighbor_ids": np.asarray(view.cell_neighbor_ids, dtype=np.int64),
        "graph_neighbor_owner_ranks": np.asarray(
            view.cell_neighbor_owner_ranks, dtype=np.int32
        ),
        "graph_neighbor_weights": np.asarray(view.cell_neighbor_weights, dtype=np.int64),
        "source_coordinate_stencil": materialized.source_coordinate_stencil,
        "exact_vertex_coordinate_offsets": (materialized.exact_vertex_coordinate_offsets),
        "blocks": tuple(block_records),
        "local_mesh": mesh,
        "cell_kind": cell_kind,
        "cell_vertex_shifts": (
            materialized.vertex_absolute_windings[vertices]
            if vertices.size
            else np.zeros((0, dimension + 1, dimension), dtype=np.int64)
        ),
        "vertex_keys": tuple(
            (
                int(materialized.vertex_source_ids[row]),
                tuple(int(value) for value in materialized.vertex_absolute_windings[row]),
            )
            for row in range(vertex_count)
        ),
    }


def _owner_local_periodic_delaunay_neighbor_halo(
    mesh_data: dict[str, Any],
    candidate: dict[str, Any],
    source: NativePeriodicSource,
    rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> None:
    import json

    from ..._execution_runtime import ExecutionGroup
    from ...discretization._cell_mesh import CellBlock, CellMesh
    from ...discretization._periodic_topology import _entity_vertex_rows
    from ...discretization.fem._reference import lagrange_element
    from .._distribution import _owner_exchange_message_banks, _owner_group_fail_if_any
    from .._periodic import PeriodicPointOrbits

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or not isinstance(execution_group, ExecutionGroup)
    ):
        raise TypeError("Periodic neighbor closure requires canonical Delaunay inputs.")
    dimension = source.ambient_dimension
    arity = dimension + 1
    cell_ids = np.asarray(mesh_data["cell_ids"], dtype=np.int64)
    cell_owners = np.asarray(mesh_data["cell_owner_ranks"], dtype=np.int32)
    neighbor_ids = np.asarray(mesh_data.get("graph_neighbor_ids"), dtype=np.int64)
    neighbor_owners = np.asarray(
        mesh_data.get("graph_neighbor_owner_ranks"), dtype=np.int32
    )
    owner_count = len(execution_group.spec.process_indices)
    local_error = None
    if (
        dimension not in (2, 3)
        or cell_ids.ndim != 1
        or np.unique(cell_ids).size != cell_ids.size
        or cell_owners.shape != cell_ids.shape
        or neighbor_ids.ndim != 1
        or neighbor_owners.shape != neighbor_ids.shape
    ):
        local_error = "Periodic Delaunay neighbor closure has malformed graph rows."
    local_cell_rows = {
        int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
    }
    missing_by_owner: dict[int, set[int]] = {}
    if local_error is None:
        for identifier, owner in zip(neighbor_ids, neighbor_owners, strict=True):
            cell_id = int(identifier)
            owner_rank = int(owner)
            if owner_rank < 0 or owner_rank >= owner_count:
                local_error = "Periodic face graph names an unknown neighbor owner."
            elif cell_id in local_cell_rows:
                if int(cell_owners[local_cell_rows[cell_id]]) != owner_rank:
                    local_error = "Resident periodic neighbor has conflicting ownership."
            elif owner_rank == rank:
                local_error = (
                    "An owned periodic neighbor cell is absent from its owner view."
                )
            else:
                missing_by_owner.setdefault(owner_rank, set()).add(cell_id)
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic neighbor halo premises"
    )

    if "source_coordinate_stencil" not in mesh_data:
        try:
            if np.asarray(mesh_data["vertex_ids"], dtype=np.int64).size:
                support_source = NativePeriodicSource(
                    PeriodicPointOrbits(
                        source.domain.cell,
                        np.asarray(candidate["points"], dtype=np.float64),
                        np.asarray(candidate["point_ids"], dtype=np.int64),
                    ),
                    source.source_id,
                    source.source_revision,
                    point_owner_ranks=np.asarray(
                        candidate["point_owner_ranks"], dtype=np.int32
                    ),
                )
            else:
                support_source = source
            stencil, offsets = _initial_owner_local_periodic_coordinate_ancestry(
                support_source, mesh_data, source.domain.cell
            )
            mesh_data["source_coordinate_stencil"] = stencil
            mesh_data["exact_vertex_coordinate_offsets"] = (
                offsets if stencil is not None else None
            )
            local_error = None
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
    else:
        local_error = None
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic neighbor coordinate ancestry"
    )
    vertex_ids = np.asarray(mesh_data["vertex_ids"], dtype=np.int64)
    source_stencil = mesh_data["source_coordinate_stencil"]
    exact_offsets = mesh_data["exact_vertex_coordinate_offsets"]
    stencil_row_by_id: dict[int, int] = {}
    if source_stencil is not None:
        target_ids = np.asarray(source_stencil.target_global_ids, dtype=np.int64)
        if (
            target_ids.ndim != 1
            or target_ids.size != vertex_ids.size
            or not np.array_equal(target_ids, vertex_ids)
            or exact_offsets is None
            or len(exact_offsets) != vertex_ids.size
        ):
            local_error = "Periodic neighbor interpolation ancestry is not vertex-bound."
        else:
            stencil_row_by_id = {
                int(identifier): row for row, identifier in enumerate(target_ids)
            }
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic neighbor ancestry binding"
    )
    if source_stencil is not None and exact_offsets is None:
        raise ValueError("Periodic neighbor exact interpolation offsets are missing.")

    def exact_pair(value: object, /) -> list[int]:
        if not isinstance(value, Fraction):
            raise TypeError("Periodic neighbor exact payloads require Fractions.")
        return [int(value.numerator), int(value.denominator)]

    vertex_source_ids = np.asarray(mesh_data["vertex_source_ids"], dtype=np.int64)
    vertex_owner_ranks = np.asarray(mesh_data["vertex_owner_ranks"], dtype=np.int32)
    vertex_absolute_windings = np.asarray(
        mesh_data["vertex_absolute_windings"], dtype=np.int64
    )
    vertex_representatives = np.asarray(
        mesh_data["vertex_representative_ids"], dtype=np.int64
    )
    vertex_windings = np.asarray(mesh_data["vertex_windings"], dtype=np.int64)
    vertex_bits = np.asarray(mesh_data["vertex_coordinates"], dtype=np.float64).view(
        np.uint64
    )
    exact_vertex_coordinates = mesh_data["exact_vertex_coordinates"]
    cells = np.asarray(mesh_data["vertices"], dtype=np.int64)
    cell_vertex_shifts = np.asarray(mesh_data["cell_vertex_shifts"], dtype=np.int64)
    key_offsets = np.asarray(mesh_data["cell_key_offsets"], dtype=np.int64)
    key_source_ids = np.asarray(mesh_data["cell_key_source_ids"], dtype=np.int64)
    key_relative = np.asarray(mesh_data["cell_key_relative_windings"], dtype=np.int64)
    exact_cell_measures = tuple(mesh_data["exact_cell_measures"])
    cell_windings = np.asarray(mesh_data["cell_windings"], dtype=np.int64)
    raw_support_lower = mesh_data["support_fractional_lower"]
    raw_support_upper = mesh_data["support_fractional_upper"]
    local_error = None
    if raw_support_lower is None and raw_support_upper is None:
        support_lower = None
        support_upper = None
    elif raw_support_lower is None or raw_support_upper is None:
        local_error = "Periodic cell support bounds are only partially present."
        support_lower = None
        support_upper = None
    else:
        support_lower = np.asarray(raw_support_lower, dtype=np.float64).view(np.uint64)
        support_upper = np.asarray(raw_support_upper, dtype=np.float64).view(np.uint64)
        if (
            support_lower.shape != (cell_ids.size, dimension)
            or support_upper.shape != support_lower.shape
        ):
            local_error = "Periodic cell support bound rows are malformed."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic neighbor support bounds"
    )

    def cell_payload(cell_row: int, target_owner: int, /) -> _PeriodicGraphCellPayload:
        vertex_payloads: list[_PeriodicGraphVertexPayload] = []
        for vertex_row_value in cells[cell_row].tolist():
            vertex_row = int(vertex_row_value)
            vertex_id = int(vertex_ids[vertex_row])
            ancestry: _PeriodicGraphVertexCoordinateAncestryPayload | None = None
            if source_stencil is not None:
                if exact_offsets is None:
                    raise ValueError(
                        "Periodic neighbor exact interpolation offsets are missing."
                    )
                stencil_row = stencil_row_by_id[vertex_id]
                ancestry = _periodic_vertex_ancestry_payload(
                    source_stencil, stencil_row, exact_offsets[stencil_row]
                )
            vertex_payloads.append(
                {
                    "vertex_id": vertex_id,
                    "coordinate_bits": [
                        int(value) for value in vertex_bits[vertex_row].tolist()
                    ],
                    "exact_coordinates": [
                        exact_pair(value)
                        for value in exact_vertex_coordinates[vertex_row]
                    ],
                    "source_id": int(vertex_source_ids[vertex_row]),
                    "owner_rank": int(vertex_owner_ranks[vertex_row]),
                    "absolute_winding": [
                        int(value)
                        for value in vertex_absolute_windings[vertex_row].tolist()
                    ],
                    "representative_id": int(vertex_representatives[vertex_row]),
                    "quotient_winding": [
                        int(value) for value in vertex_windings[vertex_row].tolist()
                    ],
                    "coordinate_ancestry": ancestry,
                }
            )
        key_start, key_stop = map(int, key_offsets[cell_row : cell_row + 2])
        return {
            "cell_id": int(cell_ids[cell_row]),
            "source_owner": int(cell_owners[cell_row]),
            "target_owner": int(target_owner),
            "vertices": vertex_payloads,
            "cell_vertex_shifts": [
                [int(value) for value in row.tolist()]
                for row in cell_vertex_shifts[cell_row]
            ],
            "key_source_ids": [
                int(value) for value in key_source_ids[key_start:key_stop].tolist()
            ],
            "key_relative_windings": [
                [int(value) for value in row.tolist()]
                for row in key_relative[key_start:key_stop]
            ],
            "cell_winding": [int(value) for value in cell_windings[cell_row].tolist()],
            "measure": exact_pair(exact_cell_measures[cell_row]),
            "support_lower_bits": (
                []
                if support_lower is None
                else [int(value) for value in support_lower[cell_row].tolist()]
            ),
            "support_upper_bits": (
                []
                if support_upper is None
                else [int(value) for value in support_upper[cell_row].tolist()]
            ),
            "neighbors": [],
            "neighbor_owners": [],
            "neighbor_weights": [],
        }

    # Halo cells reference authored sites outside a receiver's support bank
    # (for example after graph migration); each response carries their
    # canonical sender-independent rows with the authored owner rank.
    bank_point_ids = np.asarray(candidate["point_ids"], dtype=np.int64)
    bank_points = np.ascontiguousarray(
        np.asarray(candidate["points"], dtype=np.float64).reshape(
            (bank_point_ids.size, dimension)
        )
    )
    bank_owner_ranks = np.asarray(candidate["point_owner_ranks"], dtype=np.int32)
    source_point_bank: dict[int, _PeriodicGraphSourcePointPayload] = {
        int(identifier): {
            "point_id": int(identifier),
            "owner_rank": int(owner),
            "coordinate_bits": [int(value) for value in bits],
        }
        for identifier, owner, bits in zip(
            bank_point_ids.tolist(),
            bank_owner_ranks.tolist(),
            bank_points.view(np.uint64).tolist(),
            strict=True,
        )
    }

    def referenced_source_points(
        cell: _PeriodicGraphCellPayload, /
    ) -> list[_PeriodicGraphSourcePointPayload]:
        # Adapted vertices are their own quotient keys; their exact source-site
        # roots are the active rows of the cumulative coordinate ancestry. Only
        # ancestry-free (initial) vertices and their cell keys are source sites.
        identifiers: set[int] = set()
        ancestry_free = True
        for vertex in cell["vertices"]:
            ancestry = vertex["coordinate_ancestry"]
            if ancestry is None:
                identifiers.add(int(vertex["source_id"]))
            else:
                ancestry_free = False
                identifiers.update(int(value) for value in ancestry["source_ids"])
        if ancestry_free:
            identifiers.update(int(value) for value in cell["key_source_ids"])
        absent = sorted(identifiers.difference(source_point_bank))
        if absent:
            raise ValueError(
                "Periodic neighbor-cell owner lacks referenced source points "
                f"{absent[:8]}."
            )
        return [source_point_bank[identifier] for identifier in sorted(identifiers)]

    outgoing_requests = [
        np.asarray(sorted(missing_by_owner.get(owner, ())), dtype=np.int64).reshape(
            (-1, 1)
        )
        for owner in range(owner_count)
    ]
    incoming_requests = _owner_exchange_message_banks(
        tuple(outgoing_requests), rank, execution_group, axis_name
    )
    response_banks = [np.zeros((0, 1), dtype=np.uint8) for _ in range(owner_count)]
    local_error = None
    for requester, bank in incoming_requests.items():
        requests = np.asarray(bank, dtype=np.int64)
        if requests.ndim != 2 or requests.shape[1] != 1:
            local_error = "Periodic neighbor-cell requests are malformed."
            continue
        encoded = bytearray()
        for raw_cell_id in requests[:, 0]:
            cell_id = int(raw_cell_id)
            row = local_cell_rows.get(cell_id)
            if row is None or int(cell_owners[row]) != rank:
                local_error = "Periodic neighbor-cell owner lacks its exact cell row."
                continue
            try:
                cell = cell_payload(row, int(requester))
                payload: _PeriodicHaloCellPayload = {
                    "cell": cell,
                    "source_points": referenced_source_points(cell),
                }
                raw = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                encoded.extend(len(raw).to_bytes(8, byteorder="big", signed=False))
                encoded.extend(raw)
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
        response_banks[requester] = (
            np.frombuffer(bytes(encoded), dtype=np.uint8).reshape((-1, 1)).copy()
            if encoded
            else np.zeros((0, 1), dtype=np.uint8)
        )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic neighbor-cell packing"
    )
    incoming_payloads = _owner_exchange_message_banks(
        tuple(response_banks), rank, execution_group, axis_name
    )

    def decode_payloads(bank: object, /) -> list[dict[str, Any]]:
        raw = np.asarray(bank, dtype=np.uint8).reshape(-1).tobytes()
        records: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(raw):
            if cursor + 8 > len(raw):
                raise ValueError("Periodic neighbor-cell byte stream is truncated.")
            size = int.from_bytes(raw[cursor : cursor + 8], "big", signed=False)
            cursor += 8
            if size == 0 or cursor + size > len(raw):
                raise ValueError("Periodic neighbor-cell payload has invalid length.")
            record = json.loads(raw[cursor : cursor + size])
            cursor += size
            if not isinstance(record, dict):
                raise TypeError("Periodic neighbor-cell payload must be an object.")
            records.append(record)
        return records

    expected_cell_fields = {
        "cell_id",
        "source_owner",
        "target_owner",
        "vertices",
        "cell_vertex_shifts",
        "key_source_ids",
        "key_relative_windings",
        "cell_winding",
        "measure",
        "support_lower_bits",
        "support_upper_bits",
        "neighbors",
        "neighbor_owners",
        "neighbor_weights",
    }
    expected_vertex_fields = {
        "vertex_id",
        "coordinate_bits",
        "exact_coordinates",
        "source_id",
        "owner_rank",
        "absolute_winding",
        "representative_id",
        "quotient_winding",
        "coordinate_ancestry",
    }
    incoming_cells: dict[int, _PeriodicGraphCellPayload] = {}
    resident_source_points = dict(source_point_bank)
    local_error = None
    for owner, bank in incoming_payloads.items():
        try:
            responses = decode_payloads(bank)
            if any(
                set(response) != {"cell", "source_points"}
                or not isinstance(response["cell"], dict)
                or not isinstance(response["source_points"], list)
                for response in responses
            ):
                raise ValueError("Neighbor-cell response violates its exact schema.")
            records = [response["cell"] for response in responses]
            expected = set(
                int(value) for value in outgoing_requests[owner].reshape(-1).tolist()
            )
            identifiers = [int(record.get("cell_id", -1)) for record in records]
            if len(set(identifiers)) != len(identifiers) or set(identifiers) != expected:
                raise ValueError("Neighbor-cell payload IDs differ from their requests.")
            for record in records:
                if (
                    set(record) != expected_cell_fields
                    or record["source_owner"] != owner
                    or record["target_owner"] != rank
                    or not isinstance(record["vertices"], list)
                    or len(record["vertices"]) != arity
                    or any(
                        not isinstance(vertex, dict)
                        or set(vertex) != expected_vertex_fields
                        for vertex in record["vertices"]
                    )
                ):
                    raise ValueError("Neighbor-cell payload violates its exact schema.")
                incoming_cells[int(record["cell_id"])] = cast(
                    _PeriodicGraphCellPayload, record
                )
            for response in responses:
                for point in response["source_points"]:
                    if (
                        not isinstance(point, dict)
                        or set(point) != {"point_id", "owner_rank", "coordinate_bits"}
                        or not isinstance(point["coordinate_bits"], list)
                        or len(point["coordinate_bits"]) != dimension
                        or not 0 <= int(point["owner_rank"]) < owner_count
                    ):
                        raise ValueError(
                            "Neighbor-cell source-point row violates its exact schema."
                        )
                    identifier = int(point["point_id"])
                    previous = resident_source_points.get(identifier)
                    if previous is not None and previous != point:
                        raise ValueError(
                            "Periodic neighbor cells disagree on an authored source "
                            f"point ({identifier})."
                        )
                    resident_source_points[identifier] = cast(
                        _PeriodicGraphSourcePointPayload, point
                    )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic neighbor-cell validation"
    )

    try:
        cell_records: dict[int, _PeriodicGraphCellPayload] = {
            int(identifier): cell_payload(row, rank)
            for row, identifier in enumerate(cell_ids.tolist())
        }
        if set(cell_records).intersection(incoming_cells):
            raise ValueError("Periodic neighbor request returned a resident cell twice.")
        cell_records.update(incoming_cells)
        vertex_records: dict[int, _PeriodicGraphVertexPayload] = {}
        for cell in cell_records.values():
            for vertex in cell["vertices"]:
                vertex_id = int(vertex["vertex_id"])
                previous = vertex_records.get(vertex_id)
                if previous is not None and previous != vertex:
                    differing = sorted(
                        name
                        for name, value in vertex.items()
                        if dict(previous.items()).get(name) != value
                    )
                    raise ValueError(
                        "Periodic neighbor cells disagree on shared vertex geometry "
                        f"(vertex {vertex_id}, fields {differing})."
                    )
                vertex_records[vertex_id] = vertex

        def exact_fraction(raw: object, /) -> Fraction:
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) for value in raw
                )
                or raw[1] <= 0
            ):
                raise ValueError("Periodic exact scalar payload is malformed.")
            return Fraction(raw[0], raw[1])

        cell_order = np.asarray(sorted(cell_records), dtype=np.int64)
        vertex_order = np.asarray(sorted(vertex_records), dtype=np.int64)
        vertex_row_by_id = {
            int(identifier): row for row, identifier in enumerate(vertex_order.tolist())
        }
        cells = np.asarray(
            [
                [
                    vertex_row_by_id[int(vertex["vertex_id"])]
                    for vertex in cell_records[int(identifier)]["vertices"]
                ]
                for identifier in cell_order.tolist()
            ],
            dtype=np.int32,
        ).reshape((-1, arity))
        cell_vertex_shifts_ = np.asarray(
            [
                cell_records[int(identifier)]["cell_vertex_shifts"]
                for identifier in cell_order.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, arity, dimension))
        cell_owner_ranks_ = np.asarray(
            [
                int(cell_records[int(identifier)]["source_owner"])
                for identifier in cell_order.tolist()
            ],
            dtype=np.int32,
        )
        exact_vertex_bank = tuple(
            tuple(
                exact_fraction(value)
                for value in vertex_records[int(identifier)]["exact_coordinates"]
            )
            for identifier in vertex_order.tolist()
        )
        vertex_coordinate_bits = np.asarray(
            [
                vertex_records[int(identifier)]["coordinate_bits"]
                for identifier in vertex_order.tolist()
            ],
            dtype=np.uint64,
        ).reshape((-1, dimension))
        vertex_coordinates_ = vertex_coordinate_bits.view(np.float64)
        vertex_source_ids_ = np.asarray(
            [
                int(vertex_records[int(identifier)]["source_id"])
                for identifier in vertex_order.tolist()
            ],
            dtype=np.int64,
        )
        vertex_owner_ranks_ = np.asarray(
            [
                int(vertex_records[int(identifier)]["owner_rank"])
                for identifier in vertex_order.tolist()
            ],
            dtype=np.int32,
        )
        vertex_absolute_windings_ = np.asarray(
            [
                vertex_records[int(identifier)]["absolute_winding"]
                for identifier in vertex_order.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, dimension))
        vertex_representatives_ = np.asarray(
            [
                int(vertex_records[int(identifier)]["representative_id"])
                for identifier in vertex_order.tolist()
            ],
            dtype=np.int64,
        )
        vertex_windings_ = np.asarray(
            [
                vertex_records[int(identifier)]["quotient_winding"]
                for identifier in vertex_order.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, dimension))
        exact_cell_measures_ = tuple(
            exact_fraction(cell_records[int(identifier)]["measure"])
            for identifier in cell_order.tolist()
        )
        for cell_row, identifier in enumerate(cell_order.tolist()):
            cell = cell_records[int(identifier)]
            if (
                len(cell["key_source_ids"]) != arity
                or len(cell["key_relative_windings"]) != arity
                or len(cell["cell_winding"]) != dimension
                or len(cell["support_lower_bits"]) != len(cell["support_upper_bits"])
                or len(cell["support_lower_bits"]) not in (0, dimension)
            ):
                raise ValueError("Periodic neighbor cell has malformed exact keys.")
            exact_points = tuple(exact_vertex_bank[int(row)] for row in cells[cell_row])
            origin = exact_points[0]
            edges = tuple(
                tuple(
                    exact_points[edge + 1][axis] - origin[axis]
                    for axis in range(dimension)
                )
                for edge in range(dimension)
            )
            if dimension == 2:
                determinant = edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0]
                measure = determinant / 2
            else:
                first, second, third = edges
                determinant = (
                    first[0] * (second[1] * third[2] - second[2] * third[1])
                    - first[1] * (second[0] * third[2] - second[2] * third[0])
                    + first[2] * (second[0] * third[1] - second[1] * third[0])
                )
                measure = determinant / 6
            if measure <= 0 or measure != exact_cell_measures_[cell_row]:
                raise ValueError("Periodic neighbor cell exact measure is inconsistent.")
        support_widths = {
            len(cell_records[int(identifier)]["support_lower_bits"])
            for identifier in cell_order.tolist()
        }
        support_upper_widths = {
            len(cell_records[int(identifier)]["support_upper_bits"])
            for identifier in cell_order.tolist()
        }
        if support_widths != support_upper_widths or len(support_widths) > 1:
            raise ValueError("Periodic neighbor support-bound rows are inconsistent.")
        if support_widths == {dimension}:
            support_lower_ = (
                np.asarray(
                    [
                        cell_records[int(identifier)]["support_lower_bits"]
                        for identifier in cell_order.tolist()
                    ],
                    dtype=np.uint64,
                )
                .reshape((-1, dimension))
                .view(np.float64)
            )
            support_upper_ = (
                np.asarray(
                    [
                        cell_records[int(identifier)]["support_upper_bits"]
                        for identifier in cell_order.tolist()
                    ],
                    dtype=np.uint64,
                )
                .reshape((-1, dimension))
                .view(np.float64)
            )
            if (
                np.any(~np.isfinite(support_lower_))
                or np.any(~np.isfinite(support_upper_))
                or np.any(support_lower_ > support_upper_)
            ):
                raise ValueError("Periodic neighbor support bounds are invalid.")
        elif support_widths == {0}:
            support_lower_ = None
            support_upper_ = None
        elif not support_widths and not cell_order.size:
            support_lower_ = (
                None if support_lower is None else support_lower.view(np.float64)
            )
            support_upper_ = (
                None if support_upper is None else support_upper.view(np.float64)
            )
        else:
            raise ValueError("Periodic neighbor support-bound rows are inconsistent.")
        key_source_ids_ = np.asarray(
            [
                cell_records[int(identifier)]["key_source_ids"]
                for identifier in cell_order.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, arity))
        key_relative_ = np.asarray(
            [
                cell_records[int(identifier)]["key_relative_windings"]
                for identifier in cell_order.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, arity, dimension))
        cell_windings_ = np.asarray(
            [
                cell_records[int(identifier)]["cell_winding"]
                for identifier in cell_order.tolist()
            ],
            dtype=np.int64,
        ).reshape((-1, dimension))
        ancestry_rows: list[_PeriodicGraphVertexCoordinateAncestryPayload] = []
        for identifier in vertex_order.tolist():
            ancestry_row = vertex_records[int(identifier)]["coordinate_ancestry"]
            if ancestry_row is not None:
                ancestry_rows.append(ancestry_row)
        if ancestry_rows and len(ancestry_rows) != vertex_order.size:
            raise ValueError("Periodic neighbor vertices have incomplete ancestry.")
        source_stencil_: VertexInterpolationStencil | None = None
        exact_offsets_: tuple[tuple[Fraction, ...], ...] | None = None
        cell_kind = str(mesh_data["cell_kind"])
        element = lagrange_element(cell_kind, 1)
        cell_rows_ = np.arange(cell_order.size, dtype=np.int64)
        vertex_rows_ = np.arange(vertex_order.size, dtype=np.int64)
        cell_measures_ = np.asarray(
            [float(value) for value in exact_cell_measures_], dtype=np.float64
        )
        local_entity_ids: dict[int, np.ndarray] = {}
        entity_vertex_offsets: dict[int, np.ndarray] = {}
        entity_vertex_values: dict[int, np.ndarray] = {}
        local_mesh = None
        blocks: tuple[dict[str, Any], ...]
        if cell_order.size:
            block = CellBlock("periodic", cell_kind, cells, global_ids=cell_order)
            local_mesh = CellMesh(
                vertex_coordinates_,
                (block,),
                vertex_global_ids=vertex_order,
            )
            if ancestry_rows:
                source_stencil_, exact_offsets_ = _periodic_ancestry_stencil(
                    ancestry_rows,
                    vertex_order,
                    local_mesh.entity_set(0).entity_set_id,
                    dimension,
                )
            entity_vertices = _entity_vertex_rows(local_mesh)
            for degree, rows_at_degree in enumerate(entity_vertices):
                identifiers = np.asarray(
                    local_mesh.entity_set(degree).entity_ids, dtype=np.int64
                )
                if degree == 0:
                    ordered_rows = np.arange(vertex_order.size, dtype=np.int64)
                elif degree == dimension:
                    ordered_rows = np.arange(cell_order.size, dtype=np.int64)
                    if not np.array_equal(identifiers, cell_order):
                        raise ValueError("Periodic neighbor cell row order changed.")
                else:
                    ordered_rows = np.argsort(identifiers, kind="stable")
                local_entity_ids[degree] = identifiers[ordered_rows]
                offsets = np.zeros((ordered_rows.size + 1,), dtype=np.int64)
                values: list[np.ndarray] = []
                for local_row, entity_row in enumerate(ordered_rows):
                    vertices_ = np.asarray(
                        rows_at_degree[int(entity_row)], dtype=np.int64
                    )
                    values.append(vertices_.astype(np.int32, copy=False))
                    offsets[local_row + 1] = offsets[local_row] + vertices_.size
                entity_vertex_offsets[degree] = offsets
                entity_vertex_values[degree] = (
                    np.concatenate(values).astype(np.int32, copy=False)
                    if values
                    else np.zeros((0,), dtype=np.int32)
                )
            blocks = (
                {
                    "name": "periodic",
                    "cell_kind": cell_kind,
                    "cell_rows": cell_rows_,
                    "cell_ids": cell_order,
                    "vertices": cells,
                    "vertex_valid": np.ones(cells.shape, dtype=np.bool_),
                    "coordinate_routes": cells,
                    "element": element,
                    "element_rows": cell_rows_.astype(np.int32),
                    "source_cell_ids": cell_order,
                    "chart_arguments": None,
                    "cell_measures": cell_measures_,
                    "face_ids": None,
                    "cell_face_offsets": None,
                    "cell_face_values": None,
                    "cell_face_signs": None,
                    "face_vertex_offsets": None,
                    "face_vertex_values": None,
                    "cell_vertex_shifts": cell_vertex_shifts_,
                },
            )
        else:
            for degree in range(dimension + 1):
                local_entity_ids[degree] = np.zeros((0,), dtype=np.int64)
                entity_vertex_offsets[degree] = np.zeros((1,), dtype=np.int64)
                entity_vertex_values[degree] = np.zeros((0,), dtype=np.int32)
            blocks = ()
        graph_fields = (
            "graph_cell_ids",
            "graph_cell_offsets",
            "graph_neighbor_ids",
            "graph_neighbor_owner_ranks",
            "graph_neighbor_weights",
            "graph_cell_owner_ranks",
            "graph_cell_weights",
        )
        expanded_mesh_data = {
            "cell_rows": cell_rows_,
            "cell_ids": cell_order,
            "owned": cell_owner_ranks_ == rank,
            "cell_measures": cell_measures_,
            "exact_cell_measures": exact_cell_measures_,
            "cell_owner_ranks": cell_owner_ranks_,
            "support_cell_mask": np.ones(cell_order.shape, dtype=np.bool_),
            "vertices": cells,
            "vertex_valid": np.ones(cells.shape, dtype=np.bool_),
            "vertex_rows": vertex_rows_,
            "vertex_ids": vertex_order,
            "vertex_coordinates": vertex_coordinates_,
            "exact_vertex_coordinates": exact_vertex_bank,
            "vertex_source_ids": vertex_source_ids_,
            "vertex_owner_ranks": vertex_owner_ranks_,
            "vertex_absolute_windings": vertex_absolute_windings_,
            "coordinate_rows": vertex_rows_,
            "coordinates": vertex_coordinates_.copy(),
            "exact_coordinates": exact_vertex_bank,
            "entity_set_ids": {},
            "local_entity_ids": local_entity_ids,
            "entity_vertex_offsets": entity_vertex_offsets,
            "entity_vertex_values": entity_vertex_values,
            "cell_key_offsets": np.arange(cell_order.size + 1, dtype=np.int64) * arity,
            "cell_key_source_ids": key_source_ids_.reshape(-1),
            "cell_key_relative_windings": key_relative_.reshape((-1, dimension)),
            "periodic_cell_ids": cell_order.copy(),
            "vertex_representative_ids": vertex_representatives_,
            "vertex_windings": vertex_windings_,
            "cell_windings": cell_windings_,
            "support_fractional_lower": support_lower_,
            "support_fractional_upper": support_upper_,
            "geometry_layout_id": mesh_data["geometry_layout_id"],
            "blocks": blocks,
            "local_mesh": local_mesh,
            "cell_kind": cell_kind,
            "cell_vertex_shifts": cell_vertex_shifts_,
            "vertex_keys": tuple(
                (
                    int(vertex_source_ids_[row]),
                    tuple(int(value) for value in vertex_absolute_windings_[row]),
                )
                for row in range(vertex_order.size)
            ),
            "source_coordinate_stencil": source_stencil_,
            "exact_vertex_coordinate_offsets": exact_offsets_,
        }
        for field in graph_fields:
            expanded_mesh_data[field] = mesh_data[field]
        graph_neighbor_ids = np.asarray(
            expanded_mesh_data["graph_neighbor_ids"], dtype=np.int64
        )
        graph_neighbor_owners = np.asarray(
            expanded_mesh_data["graph_neighbor_owner_ranks"], dtype=np.int32
        )
        expanded_rows = {
            int(identifier): row for row, identifier in enumerate(cell_order.tolist())
        }
        neighbor_rows = np.asarray(
            [expanded_rows.get(int(identifier), -1) for identifier in graph_neighbor_ids],
            dtype=np.int64,
        )
        if np.any(neighbor_rows < 0) or not np.array_equal(
            cell_owner_ranks_[neighbor_rows], graph_neighbor_owners
        ):
            raise ValueError("Periodic owner-local view omits a face-neighbor cell.")
        mesh_data.update(expanded_mesh_data)
        local_error = None
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic neighbor halo materialization"
    )
    from ._native_periodic_power import _owner_local_periodic_site_weights_for_support

    union_ids = np.asarray(sorted(resident_source_points), dtype=np.int64)
    union_owner_ranks = np.asarray(
        [resident_source_points[int(value)]["owner_rank"] for value in union_ids],
        dtype=np.int32,
    )
    union_site_weights = _owner_local_periodic_site_weights_for_support(
        source,
        union_ids,
        union_owner_ranks,
        rank,
        execution_group,
        axis_name=f"{axis_name}_halo_site_weights",
    )
    if union_ids.size == bank_point_ids.size:
        return
    union_points = (
        np.asarray(
            [
                resident_source_points[int(value)]["coordinate_bits"]
                for value in union_ids
            ],
            dtype=np.uint64,
        )
        .reshape((-1, dimension))
        .view(np.float64)
    )
    # Halo-only sites are non-owned support of this view; their owner ranks are
    # the authored owners carried by the sending cell owner.
    mesh_data["expanded_source_authority"] = NativePeriodicSource(
        PeriodicPointOrbits(source.domain.cell, union_points, union_ids),
        source.source_id,
        source.source_revision,
        site_weights=union_site_weights,
        point_owner_ranks=union_owner_ranks,
    )
    # The published view binds its source rows to this same unioned bank.
    candidate["points"] = union_points
    candidate["point_ids"] = union_ids
    candidate["point_owner_ranks"] = union_owner_ranks
    candidate["source_owned"] = union_owner_ranks == rank


def _owner_local_periodic_mesh_authority(
    mesh_data: dict[str, Any],
    source: NativePeriodicSource,
    candidate: dict[str, Any],
    source_entry: CompositionEntry,
    source_content_id: str,
    coordinate_contract_id: str,
    specification_id: str,
    route: str,
    source_map_id: str | None,
    source_context_id: str | None,
    rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> _OwnerLocalPeriodicMeshAuthority:
    from ..._execution_runtime import ExecutionGroup
    from ...lifecycle import CompositionEntry
    from .._distribution import (
        _owner_group_fail_if_any,
        ShardedOwnerDirectoryWitness,
    )

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source_entry, CompositionEntry)
        or not isinstance(execution_group, ExecutionGroup)
    ):
        raise TypeError("Periodic mesh identity requires canonical source/group inputs.")
    if not isinstance(source.domain, PeriodicPointOrbits):
        raise TypeError("Periodic mesh identity requires periodic point orbits.")
    domain = source.domain
    exact_volume, expected_volume = _owner_local_require_periodic_volume(
        tuple(mesh_data["exact_cell_measures"]),
        np.asarray(mesh_data["owned"], dtype=np.bool_),
        domain.cell,
        execution_group,
        axis_name,
    )
    face_coverage_id, local_cell_faces, owned_cell_rows = (
        _owner_local_periodic_face_graph(
            mesh_data,
            domain.cell,
            rank,
            execution_group,
            source.source_id,
            source.source_revision,
            source_content_id,
            f"{axis_name}_faces",
        )
    )
    if route == "periodic_delaunay":
        _owner_local_periodic_delaunay_neighbor_halo(
            mesh_data, candidate, source, rank, execution_group, axis_name
        )
        cell_ids = np.asarray(mesh_data["cell_ids"], dtype=np.int64)
        graph_cell_ids = np.asarray(mesh_data["graph_cell_ids"], dtype=np.int64)
        owned_cell_rows = np.searchsorted(cell_ids, graph_cell_ids).astype(
            np.int64, copy=False
        )
        _owner_group_fail_if_any(
            None
            if owned_cell_rows.shape == graph_cell_ids.shape
            and np.all(owned_cell_rows < cell_ids.size)
            and np.array_equal(cell_ids[owned_cell_rows], graph_cell_ids)
            else "Periodic graph rows differ after neighbor halo materialization.",
            execution_group,
            axis_name,
            "periodic neighbor halo graph rows",
        )
    cell_ids = np.asarray(mesh_data["cell_ids"], dtype=np.int64)
    key_offsets = np.asarray(mesh_data["cell_key_offsets"], dtype=np.int64)
    key_sources = np.asarray(mesh_data["cell_key_source_ids"], dtype=np.int64)
    key_windings = np.asarray(mesh_data["cell_key_relative_windings"], dtype=np.int64)
    topology_payload = tuple(
        (
            int(cell_ids[row]),
            tuple(
                (
                    int(key_sources[index]),
                    tuple(int(value) for value in key_windings[index]),
                )
                for index in range(int(key_offsets[row]), int(key_offsets[row + 1]))
            ),
            local_cell_faces[int(np.searchsorted(owned_cell_rows, row))],
        )
        for row in owned_cell_rows
    )
    topology_id = _owner_local_multiset_identity(
        "periodic-quotient-topology",
        {
            "periodic_cell": domain.cell.cell_id,
            "periodic_axes": domain.cell.periodic_axes,
            "lattice_vectors": array_tree_fingerprint(
                np.asarray(domain.cell.vectors, dtype=np.float64)
            ),
            "dimension": domain.ambient_dimension,
        },
        tuple((record[0], record) for record in topology_payload),
        source.source_id,
        None,
        execution_group,
        f"{axis_name}_topology",
    )
    vertex_owner_ranks = np.asarray(mesh_data["vertex_owner_ranks"], dtype=np.int32)
    owned_vertex_rows = np.flatnonzero(vertex_owner_ranks == rank)
    exact_vertex_coordinates = mesh_data["exact_vertex_coordinates"]
    vertex_ids = np.asarray(mesh_data["vertex_ids"], dtype=np.int64)
    vertex_source_ids = np.asarray(mesh_data["vertex_source_ids"], dtype=np.int64)
    vertex_absolute_windings = np.asarray(
        mesh_data["vertex_absolute_windings"], dtype=np.int64
    )
    vertex_coordinate_bits = np.asarray(
        mesh_data["vertex_coordinates"], dtype=np.float64
    ).view(np.uint64)
    if vertex_coordinate_bits.shape != (vertex_ids.size, domain.ambient_dimension):
        raise ValueError("Periodic exact geometry coordinates do not match vertex IDs.")
    geometry_records = tuple(
        (
            int(vertex_ids[row]),
            (
                tuple(
                    (value.numerator, value.denominator)
                    for value in exact_vertex_coordinates[int(row)]
                ),
                tuple(int(value) for value in vertex_coordinate_bits[int(row)]),
                int(vertex_source_ids[int(row)]),
                tuple(int(value) for value in vertex_absolute_windings[int(row)]),
            ),
        )
        for row in owned_vertex_rows
    )
    geometry_id = _owner_local_multiset_identity(
        "periodic-exact-geometry",
        {
            "coordinate_contract": coordinate_contract_id,
            "periodic_cell": domain.cell.cell_id,
            "dimension": domain.ambient_dimension,
            "source_content": source_content_id,
        },
        geometry_records,
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_geometry",
    )
    owner_directory = ShardedOwnerDirectoryWitness(
        execution_group,
        rank,
        source_content_id,
        "periodic-owner-directory",
        candidate["point_ids"],
        candidate["point_owner_ranks"],
        candidate["source_owned"],
        mesh_data["cell_ids"],
        mesh_data["cell_owner_ranks"],
        mesh_data["vertex_ids"],
        mesh_data["vertex_owner_ranks"],
        source_map_id=source_map_id,
        source_context_id=source_context_id,
        axis_name=f"{axis_name}_directory",
    )
    periodic_topology_id = canonical_fingerprint(
        {
            "kind": "periodic-owner-local-topology",
            "topology": topology_id,
            "periodic_cell": domain.cell.cell_id,
        }
    )
    part_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-part",
            "source_manifest": source_content_id,
            "topology": topology_id,
            "geometry": geometry_id,
            "distribution": owner_directory.directory_id,
            "specification": specification_id,
            "route": route,
            "face_coverage_id": face_coverage_id,
        }
    )
    return _OwnerLocalPeriodicMeshAuthority(
        face_coverage_id,
        local_cell_faces,
        owned_cell_rows,
        topology_id,
        geometry_id,
        periodic_topology_id,
        part_id,
        owner_directory,
        exact_volume,
        expected_volume,
    )


def _owner_local_periodic_successor_authority(
    mesh_authority: _OwnerLocalPeriodicMeshAuthority,
    source_authority_value: NativePeriodicSource | ExactPowerCellGeometrySource,
    source_entry: CompositionEntry,
    source_composition: Composition,
    route: str,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str,
) -> _OwnerLocalPeriodicSuccessorAuthority:
    from ..._execution_runtime import ExecutionGroup
    from ...discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ...lifecycle import Composition, CompositionEntry
    from .._distribution import ShardedOwnerDirectoryWitness

    if not isinstance(mesh_authority, _OwnerLocalPeriodicMeshAuthority):
        raise TypeError("mesh_authority must be _OwnerLocalPeriodicMeshAuthority.")
    if not isinstance(
        source_authority_value,
        (NativePeriodicSource, ExactPowerCellGeometrySource),
    ):
        raise TypeError("source_authority_value must be typed periodic source authority.")
    if (
        not isinstance(source_entry, CompositionEntry)
        or not isinstance(source_composition, Composition)
        or not isinstance(execution_group, ExecutionGroup)
    ):
        raise TypeError(
            "Periodic successor authority requires accepted source/group values."
        )
    if (
        route not in ("periodic_delaunay", "periodic_power")
        or not isinstance(axis_name, str)
        or not axis_name.strip()
        or source_composition.entry(source_entry.entry_id).record_id
        != source_entry.record_id
    ):
        raise ValueError("Periodic successor authority inputs are inconsistent.")
    owner_directory = mesh_authority.owner_directory
    if not isinstance(owner_directory, ShardedOwnerDirectoryWitness):
        raise TypeError(
            "Periodic successor directory must be ShardedOwnerDirectoryWitness."
        )
    directory_entry = owner_directory.composition_entry()
    part_authority = CompositionEntry(
        source_authority_value,
        entry_id="periodic-part",
        role="topology",
        owner_id=f"phydrax.meshing.providers.native_periodic.{route}",
        structure_id=mesh_authority.topology_id,
        revision_id=mesh_authority.part_id,
        semantics_id=mesh_authority.geometry_id,
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
            directory_entry.binding("structure"),
        ),
    )
    bound_witness = owner_directory.bind_source_authority(
        part_authority.record_id,
        execution_group,
        axis_name=f"{axis_name}_directory_binding",
    )
    bound_entry = bound_witness.composition_entry()
    accepted_composition = Composition(
        (*source_composition.entries, bound_entry, part_authority),
        boundary_id=mesh_authority.part_id,
    )
    if (
        accepted_composition.entry(part_authority.entry_id).record_id
        != part_authority.record_id
        or bound_witness.source_authority_id != part_authority.record_id
    ):
        raise ValueError(
            "Periodic successor composition lost its current part authority."
        )
    return _OwnerLocalPeriodicSuccessorAuthority(
        source_entry,
        source_composition,
        mesh_authority,
        part_authority,
        bound_witness,
        accepted_composition,
    )


def _require_owner_local_periodic_adaptive_target_current(
    view: OwnerLocalCellGeometry,
    source: NativePeriodicSource,
    source_authority: NativePeriodicSource,
    source_content_id: str,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    execution_group: ExecutionGroup,
    local_rank: int,
    /,
    *,
    axis_name: str,
) -> None:
    from ..._execution_runtime import ExecutionGroup
    from ...discretization._view_support import OwnerLocalCellGeometry
    from ...meshing._distribution import _owner_group_fail_if_any

    if (
        not isinstance(view, OwnerLocalCellGeometry)
        or not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.point_owner_ranks is None
        or not isinstance(source_authority, NativePeriodicSource)
        or not isinstance(source_authority.domain, PeriodicPointOrbits)
        or source_authority.point_owner_ranks is None
        or source_authority.source_id != source.source_id
        or source_authority.source_revision != source.source_revision
        or source_authority.binding_id != view.source_context_id
        or not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
        or not isinstance(execution_group, ExecutionGroup)
        or view.adaptive_result is None
        or view.owner_rank != local_rank
        or specification.specification_id != view.adaptive_result.specification_id
        or not isinstance(axis_name, str)
        or not axis_name.strip()
        or not isinstance(source_content_id, str)
        or not source_content_id
        or view.owner_directory_data.source_manifest_id != source_content_id
        or view.owner_directory_data.group_id != execution_group.spec.group_id
        or view.owner_directory_data.rank != local_rank
        or view.owner_directory_data.part_count
        != len(execution_group.spec.process_indices)
    ):
        raise TypeError(
            "Adaptive periodic target replay requires current typed operands."
        )
    source_entry, source_composition, replayed_content_id = (
        _owner_local_periodic_source_authority(
            source,
            execution_group,
            f"{axis_name}_source_authority",
        )
    )
    _owner_group_fail_if_any(
        None
        if replayed_content_id == source_content_id
        else "Replayed adaptive source content differs from the local owner view.",
        execution_group,
        f"{axis_name}_source_content",
        "stored adaptive periodic source content",
    )
    mesh_error = None
    try:
        mesh_data = _owner_local_periodic_delaunay_view_mesh_data(
            view,
            source_authority,
            source.domain.cell,
        )
    except Exception as error:
        mesh_error = f"{type(error).__name__}: {error}"
        mesh_data = {}
    _owner_group_fail_if_any(
        mesh_error,
        execution_group,
        axis_name,
        "stored adaptive target exact mesh",
    )
    candidate = {
        "point_ids": np.asarray(view.source_point_ids, dtype=np.int64),
        "point_owner_ranks": np.asarray(view.source_owner_ranks, dtype=np.int32),
        "source_owned": np.asarray(view.source_owned, dtype=np.bool_),
    }
    mesh_authority = _owner_local_periodic_mesh_authority(
        mesh_data,
        source,
        candidate,
        source_entry,
        source_content_id,
        view.coordinate_contract_id,
        specification.specification_id,
        "periodic_delaunay",
        view.source_map_id,
        view.source_context_id,
        local_rank,
        execution_group,
        f"{axis_name}_authority",
    )
    successor = _owner_local_periodic_successor_authority(
        mesh_authority,
        source_authority,
        source_entry,
        source_composition,
        "periodic_delaunay",
        execution_group,
        axis_name=f"{axis_name}_successor",
    )
    shell = view.periodic_shell_certificate
    expected_entity_set_ids = {
        degree: canonical_fingerprint(
            {
                "kind": "periodic-owner-local-entity-set",
                "topology": mesh_authority.topology_id,
                "dimension": degree,
            }
        )
        for degree in range(view.topological_dimension + 1)
    }
    local_error = None
    if (
        mesh_authority.face_coverage_id != view.periodic_face_coverage_id
        or mesh_authority.topology_id != view.source_topology_id
        or mesh_authority.geometry_id != view.source_geometry_id
        or mesh_authority.periodic_topology_id != view.periodic_topology_id
        or mesh_authority.part_id != view.part_id
        or mesh_authority.owner_directory.directory_id != view.distribution_id
        or successor.bound_witness.directory_id != view.distribution_id
        or successor.part_authority.semantics_id != view.source_geometry_id
        or successor.part_authority.structure_id != view.source_topology_id
        or successor.bound_witness.source_authority_id
        != successor.part_authority.record_id
        or expected_entity_set_ids != dict(view.entity_set_ids)
        or not isinstance(shell, dict)
        or shell.get("quotient_face_coverage_id") != mesh_authority.face_coverage_id
        or shell.get("exact_fundamental_volume")
        != (
            mesh_authority.exact_volume.numerator,
            mesh_authority.exact_volume.denominator,
        )
        or shell.get("expected_fundamental_volume")
        != (
            mesh_authority.expected_volume.numerator,
            mesh_authority.expected_volume.denominator,
        )
        or shell.get("owner_local_candidate_id") != view.adaptive_result.candidate_id
        or shell.get("size_evidence_id") != view.adaptive_result.size_evidence.evidence_id
    ):
        local_error = "Replayed adaptive periodic target authority changed."
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        axis_name,
        "stored adaptive target quotient authority",
    )


def _owner_local_periodic_source_content_identity(
    source: NativePeriodicSource,
    rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> str:
    from .._periodic import PeriodicPointOrbits

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.point_owner_ranks is None
    ):
        raise TypeError("Periodic source-content identity requires owned point orbits.")
    point_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
    points = np.asarray(source.domain.points, dtype=np.float64)
    owners = np.asarray(source.point_owner_ranks, dtype=np.int32)
    weights = (
        None
        if source.site_weights is None
        else np.asarray(source.site_weights, dtype=np.float64)
    )
    if (
        point_ids.ndim != 1
        or points.shape != (point_ids.size, source.ambient_dimension)
        or owners.shape != point_ids.shape
        or (weights is not None and weights.shape != point_ids.shape)
    ):
        raise ValueError("Periodic source-content banks are not aligned.")
    local_rows = np.flatnonzero(owners == rank)
    records = tuple(
        (
            int(point_ids[row]),
            (
                tuple(
                    (value.numerator, value.denominator)
                    for value in (
                        Fraction.from_float(float(component)) for component in points[row]
                    )
                ),
                tuple(int(value) for value in points[row].view(np.uint64)),
                (
                    (0, 1)
                    if weights is None
                    else (
                        Fraction.from_float(float(weights[row])).numerator,
                        Fraction.from_float(float(weights[row])).denominator,
                    )
                ),
                (
                    0
                    if weights is None
                    else int(np.asarray(weights[row], dtype=np.float64).view(np.uint64))
                ),
            ),
        )
        for row in local_rows
    )
    return _owner_local_multiset_identity(
        "periodic-point-source-content",
        {
            "periodic_cell": source.domain.cell.cell_id,
            "dimension": source.ambient_dimension,
            "weighted": weights is not None,
        },
        records,
        source.source_id,
        source.source_revision,
        execution_group,
        axis_name,
    )


def _owner_local_require_periodic_volume(
    exact_cell_measures: tuple[Fraction, ...],
    owned: ArrayLike,
    cell: PeriodicCell,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[Fraction, Fraction]:

    from .._distribution import _owner_group_allgather, _owner_group_fail_if_any

    values = tuple(exact_cell_measures)
    owned_mask = np.asarray(owned, dtype=np.bool_)
    local_error = None
    if (
        owned_mask.ndim != 1
        or owned_mask.shape != (len(values),)
        or any(not isinstance(value, Fraction) for value in values)
    ):
        local_error = "Exact periodic cell measures are not aligned with owned rows."
    local_total = sum(
        (value for value, is_owned in zip(values, owned_mask, strict=True) if is_owned),
        Fraction(0),
    )
    if local_total < 0:
        local_error = "Exact periodic owner volume is negative."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "exact periodic owner volume"
    )
    encoded = f"{local_total.numerator}/{local_total.denominator}".encode("ascii")
    lengths = _owner_group_allgather(
        np.asarray([len(encoded)], dtype=np.int64), execution_group, axis_name
    ).reshape(-1)
    width = int(np.max(lengths, initial=0))
    local_bytes = np.zeros((width,), dtype=np.uint8)
    local_bytes[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    gathered = _owner_group_allgather(local_bytes, execution_group, axis_name)
    total = Fraction(0)
    for row, length in zip(gathered, lengths, strict=True):
        numerator_bytes, denominator_bytes = (
            np.asarray(row[: int(length)], dtype=np.uint8).tobytes().split(b"/", 1)
        )
        total += Fraction(int(numerator_bytes), int(denominator_bytes))
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in vectors
    )
    if len(exact) == 2:
        determinant = exact[0][0] * exact[1][1] - exact[0][1] * exact[1][0]
    elif len(exact) == 3:
        first, second, third = exact
        determinant = (
            first[0] * (second[1] * third[2] - second[2] * third[1])
            - first[1] * (second[0] * third[2] - second[2] * third[0])
            + first[2] * (second[0] * third[1] - second[1] * third[0])
        )
    else:
        raise ValueError("Periodic volume proof requires a 2D or 3D lattice.")
    expected = abs(determinant)
    if expected <= 0 or total != expected:
        raise ValueError(
            "Owner-local periodic cells do not exactly cover one fundamental cell."
        )
    return total, expected


def _canonical_periodic_face_key(
    source_ids: ArrayLike,
    absolute_windings: ArrayLike,
    orders: ArrayLike,
    /,
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], int]:
    from ...discretization._periodic_exact import reduce_periodic_exponents

    identifiers = np.asarray(source_ids, dtype=np.int64)
    windings = np.asarray(absolute_windings, dtype=np.int64)
    orders_ = np.asarray(orders, dtype=np.int64)
    if (
        identifiers.ndim != 1
        or windings.ndim != 2
        or windings.shape[0] != identifiers.size
        or orders_.shape != (windings.shape[1],)
        or identifiers.size < 2
    ):
        raise ValueError("Periodic face quotient keys are malformed.")
    anchor = min(
        range(identifiers.size),
        key=lambda row: (
            int(identifiers[row]),
            tuple(int(value) for value in windings[row]),
        ),
    )
    relative = np.asarray(
        reduce_periodic_exponents(windings - windings[anchor][None, :], orders_),
        dtype=np.int64,
    )
    cycle = tuple(
        (
            int(identifier),
            tuple(
                int(value)
                for value in np.asarray(exponent, dtype=np.int64).reshape((-1,)).tolist()
            ),
        )
        for identifier, exponent in zip(identifiers, relative, strict=True)
    )
    if len(set(cycle)) != len(cycle):
        raise ValueError("Periodic face quotient keys repeat an exact vertex.")

    if len(cycle) == 2:
        ordered = tuple(sorted(cycle))
        return (ordered, 1) if cycle == ordered else (ordered, -1)

    def minimal_rotation(
        values: tuple[tuple[int, tuple[int, ...]], ...], /
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return min(tuple(values[index:] + values[:index]) for index in range(len(values)))

    forward = minimal_rotation(cycle)
    reverse = minimal_rotation(tuple(reversed(cycle)))
    if forward == reverse:
        raise ValueError("Periodic face orientation is indeterminate.")
    return (forward, 1) if forward < reverse else (reverse, -1)


def _owner_local_require_periodic_face_coverage(
    face_records: tuple[
        tuple[tuple[tuple[int, tuple[int, ...]], ...], int, int, int], ...
    ],
    owned_cell_ids: np.ndarray,
    dimension: int,
    rank: int,
    execution_group: ExecutionGroup,
    source_id: str,
    source_revision: str,
    source_content_id: str,
    periodic_cell_id: str,
    axis_name: str,
    /,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    parts = len(execution_group.spec.process_indices)
    owned_ids = np.asarray(owned_cell_ids, dtype=np.int64)
    local_error = None
    if owned_ids.ndim != 1 or np.unique(owned_ids).size != owned_ids.size:
        local_error = "Owned periodic cell IDs must be one unique integer row."
    local_maximum = max(
        (
            len(
                tuple(
                    value
                    for identifier, winding in key
                    for value in (identifier, *winding)
                )
            )
            for key, _, _, _ in face_records
        ),
        default=0,
    )
    widths = _owner_group_allgather(
        np.asarray([local_maximum], dtype=np.int64),
        execution_group,
        axis_name,
    ).reshape(-1)
    width = int(np.max(widths, initial=0))
    row_width = 4 + width
    outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_rows: list[np.ndarray] = []
    for key, sign, cell_id, cell_owner in face_records:
        if (
            sign not in (-1, 1)
            or len(key) < 2
            or any(len(winding) != dimension for _, winding in key)
            or not 0 <= cell_owner < parts
            or cell_owner != rank
        ):
            local_error = "Periodic quotient face incidence is malformed."
            continue
        flattened = tuple(
            value for identifier, winding in key for value in (identifier, *winding)
        )
        destination_hash = canonical_fingerprint(
            {"kind": "periodic-face-shard", "key": key}
        )
        destination = int(destination_hash[:16], 16) % parts
        row = np.zeros((row_width,), dtype=np.int64)
        row[:4] = (sign, cell_id, cell_owner, len(flattened))
        row[4 : 4 + len(flattened)] = np.asarray(flattened, dtype=np.int64)
        if destination == rank:
            local_rows.append(row)
        else:
            outgoing_rows[destination].append(row)
    outgoing = tuple(
        np.stack(rows) if rows else np.zeros((0, row_width), dtype=np.int64)
        for rows in outgoing_rows
    )
    incoming = _owner_exchange_message_banks(outgoing, rank, execution_group, axis_name)
    if local_error is None:
        local_rows.extend(
            np.asarray(bank, dtype=np.int64)[index]
            for bank in incoming.values()
            for index in range(np.asarray(bank).shape[0])
        )
    incidences: dict[
        tuple[tuple[int, tuple[int, ...]], ...], list[tuple[int, int, int]]
    ] = {}
    for row in local_rows:
        sign, cell_id, cell_owner, length = map(int, row[:4])
        values = np.asarray(row[4 : 4 + length], dtype=np.int64)
        if (
            sign not in (-1, 1)
            or cell_owner < 0
            or cell_owner >= parts
            or length < 2 * (dimension + 1)
            or length % (dimension + 1)
            or length > width
        ):
            local_error = "Sharded periodic face-key rows are malformed."
            continue
        stride = dimension + 1
        key = tuple(
            (
                int(values[offset]),
                tuple(int(value) for value in values[offset + 1 : offset + stride]),
            )
            for offset in range(0, length, stride)
        )
        incidences.setdefault(key, []).append((sign, cell_id, cell_owner))
    if local_error is None and any(
        len(records) != 2 or sum(record[0] for record in records) != 0
        for records in incidences.values()
    ):
        local_error = (
            "Periodic quotient faces must have exactly two oppositely oriented cells."
        )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic quotient face coverage"
    )

    neighbor_outgoing: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_neighbor_rows: list[np.ndarray] = []
    for records in incidences.values():
        (
            (first_sign, first_id, first_owner),
            (
                second_sign,
                second_id,
                second_owner,
            ),
        ) = records
        del first_sign, second_sign
        for cell_id, owner, neighbor_id, neighbor_owner in (
            (first_id, first_owner, second_id, second_owner),
            (second_id, second_owner, first_id, first_owner),
        ):
            row = np.asarray([cell_id, neighbor_id, neighbor_owner], dtype=np.int64)
            if owner == rank:
                local_neighbor_rows.append(row)
            else:
                neighbor_outgoing[owner].append(row)
    neighbor_messages = tuple(
        np.stack(rows) if rows else np.zeros((0, 3), dtype=np.int64)
        for rows in neighbor_outgoing
    )
    neighbor_incoming = _owner_exchange_message_banks(
        neighbor_messages, rank, execution_group, axis_name
    )
    local_neighbor_rows.extend(
        np.asarray(bank, dtype=np.int64)[index]
        for bank in neighbor_incoming.values()
        for index in range(np.asarray(bank).shape[0])
    )
    adjacency: dict[int, dict[tuple[int, int], int]] = {
        int(cell_id): {} for cell_id in owned_ids
    }
    for row in local_neighbor_rows:
        cell_id, neighbor_id, neighbor_owner = map(int, row)
        if cell_id not in adjacency or not 0 <= neighbor_owner < parts:
            local_error = "Periodic cell adjacency returned to a nonowner."
            continue
        if cell_id != neighbor_id:
            key = (neighbor_id, neighbor_owner)
            adjacency[cell_id][key] = adjacency[cell_id].get(key, 0) + 1
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic quotient adjacency"
    )
    ordered_owned_ids = np.sort(owned_ids, kind="stable")
    offsets = np.zeros((ordered_owned_ids.size + 1,), dtype=np.int64)
    neighbor_ids: list[int] = []
    neighbor_owners: list[int] = []
    neighbor_weights: list[int] = []
    for row, cell_id in enumerate(ordered_owned_ids):
        neighbors = sorted(adjacency[int(cell_id)].items())
        neighbor_ids.extend(neighbor for (neighbor, _), _ in neighbors)
        neighbor_owners.extend(owner for (_, owner), _ in neighbors)
        neighbor_weights.extend(weight for _, weight in neighbors)
        offsets[row + 1] = len(neighbor_ids)
    neighbor_id_array = np.asarray(neighbor_ids, dtype=np.int64)
    neighbor_owner_array = np.asarray(neighbor_owners, dtype=np.int32)
    neighbor_weight_array = np.asarray(neighbor_weights, dtype=np.int64)
    local_face_records = tuple(
        (
            key,
            tuple(sorted((sign, cell_id) for sign, cell_id, _ in records)),
        )
        for key, records in incidences.items()
    )
    face_counts = _owner_group_allgather(
        np.asarray([len(local_face_records)], dtype=np.int64),
        execution_group,
        axis_name,
    ).reshape(-1)
    if int(np.sum(face_counts)) == 0:
        raise ValueError("Periodic quotient face proof has no cell incidence.")
    face_coverage_id = _owner_local_multiset_identity(
        "periodic-face-coverage",
        {
            "periodic_cell": periodic_cell_id,
            "dimension": dimension,
            "source_content": source_content_id,
        },
        local_face_records,
        source_id,
        source_revision,
        execution_group,
        axis_name,
    )
    return (
        face_coverage_id,
        ordered_owned_ids,
        offsets,
        neighbor_id_array,
        neighbor_owner_array,
        neighbor_weight_array,
    )


def _owner_local_periodic_face_graph(
    mesh_data: dict[str, Any],
    periodic_cell: PeriodicCell,
    rank: int,
    execution_group: ExecutionGroup,
    source_id: str,
    source_revision: str,
    source_content_id: str,
    axis_name: str,
    /,
) -> tuple[str, _OwnerLocalPeriodicFaceRows, np.ndarray]:
    from ...discretization._periodic_exact import identification_orders

    dimension = periodic_cell.ambient_dimension
    cell_ids = np.asarray(mesh_data["cell_ids"], dtype=np.int64)
    owned_cell_rows = np.flatnonzero(np.asarray(mesh_data["owned"], dtype=np.bool_))
    vertex_source_ids = np.asarray(mesh_data["vertex_source_ids"], dtype=np.int64)
    vertex_absolute_windings = np.asarray(
        mesh_data["vertex_absolute_windings"], dtype=np.int64
    )
    cell_vertices = np.asarray(mesh_data["vertices"], dtype=np.int64)
    cell_vertex_valid = np.asarray(mesh_data["vertex_valid"], dtype=np.bool_)
    orders = identification_orders(periodic_cell)
    blocks = tuple(mesh_data["blocks"])
    if blocks and blocks[0]["cell_kind"] == "polyhedron":
        block_rows: dict[int, tuple[dict[str, Any], int]] = {}
        for block_record in blocks:
            for local_row, global_row in enumerate(
                np.asarray(block_record["cell_rows"], dtype=np.int64)
            ):
                row = int(global_row)
                if row in block_rows:
                    raise ValueError(
                        "Periodic polyhedron cell row occurs in multiple blocks."
                    )
                block_rows[row] = (block_record, local_row)
        if set(block_rows) != set(range(cell_ids.size)):
            raise ValueError("Periodic polyhedron blocks do not cover local cell rows.")
        face_rows = []
        for row in owned_cell_rows:
            block_record, local_row = block_rows[int(row)]
            face_offsets = np.asarray(block_record["face_vertex_offsets"], dtype=np.int64)
            face_values = np.asarray(block_record["face_vertex_values"], dtype=np.int64)
            cell_face_offsets = np.asarray(
                block_record["cell_face_offsets"], dtype=np.int64
            )
            cell_face_values = np.asarray(
                block_record["cell_face_values"], dtype=np.int64
            )
            cell_face_signs = np.asarray(block_record["cell_face_signs"], dtype=np.int8)
            cell_faces = []
            for incidence in range(
                int(cell_face_offsets[local_row]),
                int(cell_face_offsets[local_row + 1]),
            ):
                face = int(cell_face_values[incidence])
                start, stop = map(int, face_offsets[face : face + 2])
                loop = face_values[start:stop]
                face_key, orientation = _canonical_periodic_face_key(
                    vertex_source_ids[loop],
                    vertex_absolute_windings[loop],
                    orders,
                )
                cell_faces.append(
                    (face_key, int(cell_face_signs[incidence]) * orientation)
                )
            face_rows.append(tuple(sorted(cell_faces)))
        local_cell_faces = tuple(face_rows)
    else:
        face_rows = []
        for row in owned_cell_rows:
            local_vertices = cell_vertices[row][cell_vertex_valid[row]]
            if local_vertices.size != dimension + 1:
                raise ValueError("Periodic simplex cell has invalid face arity.")
            cell_faces = []
            for omitted in range(dimension + 1):
                face_vertices = local_vertices[
                    np.asarray(
                        [index for index in range(dimension + 1) if index != omitted],
                        dtype=np.int64,
                    )
                ]
                face_key, orientation = _canonical_periodic_face_key(
                    vertex_source_ids[face_vertices],
                    vertex_absolute_windings[face_vertices],
                    orders,
                )
                sign = (1 if omitted % 2 == 0 else -1) * orientation
                cell_faces.append((face_key, sign))
            face_rows.append(tuple(sorted(cell_faces)))
        local_cell_faces = tuple(face_rows)
    cell_owner_ranks = np.asarray(mesh_data["cell_owner_ranks"], dtype=np.int32)
    face_records = tuple(
        (
            key,
            sign,
            int(cell_ids[int(owned_cell_rows[position])]),
            int(cell_owner_ranks[int(owned_cell_rows[position])]),
        )
        for position, cell_faces in enumerate(local_cell_faces)
        for key, sign in cell_faces
    )
    (
        face_coverage_id,
        graph_cell_ids,
        graph_cell_offsets,
        graph_neighbor_ids,
        graph_neighbor_owner_ranks,
        graph_neighbor_weights,
    ) = _owner_local_require_periodic_face_coverage(
        face_records,
        cell_ids[owned_cell_rows],
        dimension,
        rank,
        execution_group,
        source_id,
        source_revision,
        source_content_id,
        periodic_cell.cell_id,
        axis_name,
    )
    cell_row_by_id = {
        int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
    }
    graph_cell_rows = np.asarray(
        [cell_row_by_id.get(int(identifier), -1) for identifier in graph_cell_ids],
        dtype=np.int64,
    )
    if np.any(graph_cell_rows < 0) or not np.array_equal(
        cell_ids[graph_cell_rows], graph_cell_ids
    ):
        raise ValueError("Periodic graph rows differ from owned cell identities.")
    mesh_data["graph_cell_ids"] = graph_cell_ids
    mesh_data["graph_cell_offsets"] = graph_cell_offsets
    mesh_data["graph_neighbor_ids"] = graph_neighbor_ids
    mesh_data["graph_neighbor_owner_ranks"] = graph_neighbor_owner_ranks
    mesh_data["graph_neighbor_weights"] = graph_neighbor_weights
    mesh_data["graph_cell_owner_ranks"] = np.full(
        graph_cell_ids.shape, rank, dtype=np.int32
    )
    mesh_data["graph_cell_weights"] = np.ones(graph_cell_ids.shape, dtype=np.int64)
    return face_coverage_id, local_cell_faces, owned_cell_rows


def _owner_local_periodic_source_authority(
    source: NativePeriodicSource,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[CompositionEntry, Composition, str]:
    import jax

    from ..._execution_runtime import ExecutionGroup
    from ...lifecycle import Composition, CompositionEntry
    from .._distribution import _owner_group_allgather

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.point_owner_ranks is None
        or not isinstance(execution_group, ExecutionGroup)
    ):
        raise TypeError(
            "Owner-local source authority requires typed periodic point shards."
        )
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise ValueError("axis_name must be a non-empty string.")
    members = execution_group.spec.process_indices
    process_index = int(jax.process_index())
    if not execution_group.is_member or process_index not in members:
        raise ValueError("Source authority process is outside its execution group.")
    local_rank = members.index(process_index)
    domain = source.domain
    common = canonical_fingerprint(
        {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "periodic_cell": domain.cell.cell_id,
            "periodic_axes": domain.cell.periodic_axes,
            "lattice_vectors": array_tree_fingerprint(
                np.asarray(domain.cell.vectors, dtype=np.float64)
            ),
            "dimension": domain.ambient_dimension,
            "has_site_weights": source.site_weights is not None,
        }
    )
    common_records = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(common), dtype=np.uint8),
        execution_group,
        axis_name,
    )
    if any(
        np.asarray(record, dtype=np.uint8).tobytes().hex() != common
        for record in common_records
    ):
        raise ValueError(
            "Owner-local periodic ranks disagree on source or lattice identity."
        )
    source_content_id = _owner_local_periodic_source_content_identity(
        source,
        local_rank,
        execution_group,
        f"{axis_name}_content",
    )
    point_ids = np.asarray(domain.point_ids, dtype=np.int64)
    owners = np.asarray(source.point_owner_ranks, dtype=np.int32)
    local_site_rows = np.flatnonzero(owners == local_rank)
    source_entity_id = _owner_local_multiset_identity(
        "periodic-point-source-entities",
        {
            "periodic_cell": domain.cell.cell_id,
            "dimension": domain.ambient_dimension,
        },
        tuple((int(point_ids[row]), ()) for row in local_site_rows),
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_entities",
    )
    semantics_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-source",
            "common_source": common,
            "source_entities": source_entity_id,
        }
    )
    entry = CompositionEntry(
        source,
        entry_id="periodic-source",
        role="topology",
        owner_id="phydrax.meshing.providers.native_periodic",
        structure_id=domain.cell.cell_id,
        revision_id=source.source_revision,
        semantics_id=semantics_id,
    )
    composition = Composition((entry,), boundary_id=semantics_id)
    return entry, composition, source_content_id


class _OwnerLocalPeriodicSourceRefresh(NamedTuple):
    source_entry: CompositionEntry
    source_composition: Composition
    source_content_id: str
    candidate: dict[str, Any]
    final_source: NativePeriodicSource
    mesh_data: dict[str, Any]
    exact_root_coordinates: tuple[tuple[int, tuple[Fraction, ...]], ...]


def _refresh_owner_local_periodic_delaunay_source(
    source: NativePeriodicSource,
    execution_group: ExecutionGroup,
    local_rank: int,
    /,
    *,
    axis_name: str = "periodic_owner_motion_source",
) -> _OwnerLocalPeriodicSourceRefresh:
    from ..._execution_runtime import ExecutionGroup
    from .._periodic import PeriodicPointOrbits

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.point_owner_ranks is None
        or not isinstance(execution_group, ExecutionGroup)
    ):
        raise TypeError(
            "Periodic Delaunay source refresh requires owned periodic point rows."
        )
    if (
        isinstance(local_rank, bool)
        or not isinstance(local_rank, Integral)
        or not 0 <= int(local_rank) < len(execution_group.spec.process_indices)
    ):
        raise ValueError("local_rank is outside the periodic owner group.")
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise ValueError("axis_name must be a non-empty string.")
    if source.site_weights is not None and np.any(
        np.asarray(source.site_weights, dtype=np.float64) != 0.0
    ):
        raise ValueError(
            "Weighted periodic sites retain power-diagram identity; Delaunay motion refuses."
        )
    point_owners = np.asarray(source.point_owner_ranks, dtype=np.int32)
    source_entry, source_composition, source_content_id = (
        _owner_local_periodic_source_authority(
            source, execution_group, f"{axis_name}_source"
        )
    )
    candidate = _owner_local_periodic_delaunay_candidate(
        source,
        point_owners,
        point_owners == int(local_rank),
        int(local_rank),
        execution_group,
        source_content_id,
        initial_margin=1.0,
        axis_name=f"{axis_name}_closure",
    )

    final_domain = PeriodicPointOrbits(
        source.domain.cell,
        candidate["points"],
        candidate["point_ids"],
    )
    final_source = NativePeriodicSource(
        final_domain,
        source.source_id,
        source.source_revision,
        point_owner_ranks=candidate["point_owner_ranks"],
    )
    mesh_data = _owner_local_periodic_delaunay_mesh_data(
        final_source,
        candidate,
        int(local_rank),
    )
    point_ids = np.asarray(final_domain.point_ids, dtype=np.int64)
    point_coordinates = np.asarray(final_domain.points, dtype=np.float64)
    point_order = np.argsort(point_ids, kind="stable")
    exact_root_coordinates = tuple(
        (
            int(point_ids[row]),
            tuple(Fraction.from_float(float(value)) for value in point_coordinates[row]),
        )
        for row in point_order
    )
    return _OwnerLocalPeriodicSourceRefresh(
        source_entry,
        source_composition,
        source_content_id,
        candidate,
        final_source,
        mesh_data,
        exact_root_coordinates,
    )


def _require_owner_local_periodic_view_current(
    view: OwnerLocalCellGeometry,
    source: NativePeriodicSource,
    source_authority: NativePeriodicSource | ExactPowerCellGeometrySource,
    source_manifest_id: str,
    current_specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
    *,
    graph_partition_plan: GraphPartitionPlan | None = None,
) -> OwnerLocalPeriodicCandidate | None:
    from ..._execution_runtime import ExecutionGroup
    from ...discretization._cell_geometry_validity import cell_geometry_id
    from ...discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ...discretization._periodic_exact import identification_id
    from ...discretization._view_support import OwnerLocalCellGeometry
    from ...graph._partition import GraphPartitionPlan
    from .._distribution import (
        _owner_group_fail_if_any,
        _owner_local_multiset_identity,
        OwnerLocalGraphMigrationEvidence,
        ShardedOwnerDirectoryData,
        ShardedOwnerDirectoryWitness,
    )
    from .._periodic import OwnerLocalPeriodicCandidate
    from ._native_periodic_power import (
        _owner_local_periodic_power_candidate,
        _owner_local_periodic_power_mesh_data,
    )

    if (
        not isinstance(view, OwnerLocalCellGeometry)
        or not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.point_owner_ranks is None
        or not isinstance(
            source_authority,
            (NativePeriodicSource, ExactPowerCellGeometrySource),
        )
        or not isinstance(current_specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
        or current_specification.specification_id != view.specification_id
        or not isinstance(execution_group, ExecutionGroup)
        or not isinstance(axis_name, str)
        or not axis_name.strip()
        or not isinstance(view.owner_directory_data, ShardedOwnerDirectoryData)
        or view.owner_directory_data.source_manifest_id != source_manifest_id
        or view.owner_directory_data.group_id != execution_group.spec.group_id
        or view.owner_directory_data.rank != view.owner_rank
        or view.owner_directory_data.part_count
        != len(execution_group.spec.process_indices)
        or source.source_id != view.source_id
        or source.source_revision != view.source_revision
        or not isinstance(source_manifest_id, str)
        or not source_manifest_id
    ):
        raise TypeError(
            "Periodic restore requires its current typed source/view authority."
        )
    power_request = (
        isinstance(current_specification, VolumeMeshingSpec)
        and current_specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
        and set(current_specification.target.cell_families.required) == {"polyhedron"}
    )
    route_error = None
    if type(source_authority) is ExactPowerCellGeometrySource:
        route = "periodic_power"
        if not power_request:
            route_error = (
                "An exact power map requires its typed polyhedral power specification."
            )
    elif isinstance(source_authority, NativePeriodicSource):
        route = "periodic_power" if power_request else "periodic_delaunay"
        if power_request and (view.source_map_id is not None or view.vertex_ids.size):
            route_error = "A nonempty periodic power view lacks its exact power map."
    else:
        route = ""
        route_error = "Stored periodic part authority has no concrete source owner."
    plan_error = route_error
    if graph_partition_plan is not None and not isinstance(
        graph_partition_plan, GraphPartitionPlan
    ):
        plan_error = "Current periodic graph plan is not GraphPartitionPlan."
    elif (graph_partition_plan is None) != (view.graph_migration_evidence is None):
        plan_error = "Current periodic graph plan differs from stored migration evidence."
    elif (
        graph_partition_plan is not None
        and route == "periodic_delaunay"
        and (
            view.adaptive_result is None
            or not isinstance(
                view.adaptive_result.graph_partition_plan, GraphPartitionPlan
            )
            or view.adaptive_result.graph_partition_plan.plan_id
            != graph_partition_plan.plan_id
        )
    ):
        plan_error = "Current periodic graph plan differs from the adaptive recipe."
    elif graph_partition_plan is not None and route == "periodic_power":
        graph_evidence = view.graph_migration_evidence
        if (
            view.adaptive_result is not None
            or not isinstance(graph_evidence, OwnerLocalGraphMigrationEvidence)
            or graph_evidence.result.plan_id != graph_partition_plan.plan_id
        ):
            plan_error = (
                "Current periodic power graph plan differs from its migration evidence."
            )
    _owner_group_fail_if_any(
        plan_error,
        execution_group,
        f"{axis_name}_graph_plan",
        "stored periodic graph plan binding",
    )
    rank = view.owner_rank
    source_content_id = _owner_local_periodic_source_content_identity(
        source,
        rank,
        execution_group,
        f"{axis_name}_source_content",
    )
    _owner_group_fail_if_any(
        None
        if source_content_id == source_manifest_id
        else "Restored periodic source content differs from its owner-directory root.",
        execution_group,
        f"{axis_name}_source_content_binding",
        "stored periodic source-content binding",
    )
    source_entry, _, authority_content_id = _owner_local_periodic_source_authority(
        source,
        execution_group,
        f"{axis_name}_source_authority",
    )
    _owner_group_fail_if_any(
        None
        if authority_content_id == source_content_id
        else "Rebuilt periodic source authority changed its source content.",
        execution_group,
        f"{axis_name}_source_authority",
        "stored periodic source authority",
    )
    local_error: str | None = None
    candidate: dict[str, Any] | None = None
    mesh_data: dict[str, Any] | None = None
    source_map: ExactPowerCellGeometrySource | None = None
    source_map_id: str | None = None
    source_context_id: str | None = None
    shell_certificate: dict[str, Any] | None = None
    periodic_cell = None
    if route == "periodic_delaunay":
        try:
            if (
                not isinstance(source.domain, PeriodicPointOrbits)
                or source.point_owner_ranks is None
            ):
                raise TypeError(
                    "Stored periodic Delaunay source lacks owned point orbits."
                )
            candidate = _owner_local_periodic_delaunay_candidate(
                source,
                source.point_owner_ranks,
                np.asarray(source.point_owner_ranks, dtype=np.int32) == rank,
                rank,
                execution_group,
                source_content_id,
                initial_margin=1.0,
                axis_name=f"{axis_name}_delaunay",
            )
            final_domain = PeriodicPointOrbits(
                source.domain.cell, candidate["points"], candidate["point_ids"]
            )
            final_source = NativePeriodicSource(
                final_domain,
                source.source_id,
                source.source_revision,
                point_owner_ranks=candidate["point_owner_ranks"],
            )
            mesh_data = _owner_local_periodic_delaunay_mesh_data(
                final_source, candidate, rank
            )
            source_context_id = final_source.binding_id
            periodic_cell = source.domain.cell
            shell_certificate = dict(candidate["periodic_shell_certificate"])
            if graph_partition_plan is not None:
                from .._distribution import (
                    _partition_owner_local_periodic_candidate,
                    OwnerLocalGraphMigrationEvidence,
                )

                if view.adaptive_result is None or source_entry is None:
                    raise ValueError("Graph migration restore lacks its adaptive recipe.")
                pre_graph_authority = _owner_local_periodic_mesh_authority(
                    mesh_data,
                    source,
                    candidate,
                    source_entry,
                    source_content_id,
                    view.coordinate_contract_id,
                    view.adaptive_result.specification_id,
                    "periodic_delaunay",
                    view.source_map_id,
                    source_context_id,
                    rank,
                    execution_group,
                    f"{axis_name}_graph_source_authority",
                )
                final_source = mesh_data.get("expanded_source_authority", final_source)
                source_context_id = final_source.binding_id
                source_candidate = _owner_local_periodic_candidate_from_mesh_data(
                    final_source,
                    candidate["candidate_revision_id"],
                    mesh_data,
                    pre_graph_authority,
                    source_entry,
                    source_content_id,
                    view.coordinate_contract_id,
                    rank,
                    len(execution_group.spec.process_indices),
                    periodic_cell.ambient_dimension,
                    periodic_cell,
                )
                graph_result = _partition_owner_local_periodic_candidate(
                    source_candidate,
                    graph_partition_plan,
                    execution_group,
                    axis_name=f"{axis_name}_graph_partition",
                )
                final_source, mesh_data, target_cell_ids = (
                    _migrate_owner_local_periodic_delaunay_data(
                        final_source,
                        source_candidate,
                        mesh_data,
                        graph_result,
                        rank,
                        execution_group,
                        f"{axis_name}_graph_migration",
                    )
                )
                if not isinstance(final_source.domain, PeriodicPointOrbits):
                    raise TypeError(
                        "Periodic graph migration lost point-orbit authority."
                    )
                final_domain = final_source.domain
                candidate = dict(candidate)
                candidate["points"] = np.asarray(final_domain.points, dtype=np.float64)
                candidate["point_ids"] = np.asarray(
                    final_domain.point_ids, dtype=np.int64
                )
                candidate["point_owner_ranks"] = np.asarray(
                    final_source.point_owner_ranks, dtype=np.int32
                )
                candidate["source_owned"] = candidate["point_owner_ranks"] == rank
                source_context_id = final_source.binding_id
                migrated_authority = _owner_local_periodic_mesh_authority(
                    mesh_data,
                    source,
                    candidate,
                    source_entry,
                    source_content_id,
                    view.coordinate_contract_id,
                    view.adaptive_result.specification_id,
                    "periodic_delaunay",
                    view.source_map_id,
                    source_context_id,
                    rank,
                    execution_group,
                    f"{axis_name}_graph_target_authority",
                )
                final_source = mesh_data.get("expanded_source_authority", final_source)
                source_context_id = final_source.binding_id
                local_error = (
                    None
                    if migrated_authority.topology_id == pre_graph_authority.topology_id
                    and migrated_authority.geometry_id == pre_graph_authority.geometry_id
                    and migrated_authority.face_coverage_id
                    == pre_graph_authority.face_coverage_id
                    else "Replayed graph migration changed exact periodic geometry or coverage."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    f"{axis_name}_graph_identity",
                    "stored periodic graph migration identity",
                )
                renewed_evidence = OwnerLocalGraphMigrationEvidence(
                    graph_result,
                    source_candidate.candidate_id,
                    pre_graph_authority.part_id,
                    pre_graph_authority.face_coverage_id,
                    pre_graph_authority.owner_directory.directory_id,
                    migrated_authority.part_id,
                    migrated_authority.owner_directory.directory_id,
                    execution_group.spec.group_id,
                    rank,
                    len(execution_group.spec.process_indices),
                    target_cell_ids,
                )
                local_error = (
                    None
                    if view.graph_migration_evidence is not None
                    and renewed_evidence.evidence_id
                    == view.graph_migration_evidence.evidence_id
                    else "Replayed graph migration evidence differs from the stored owner result."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    f"{axis_name}_graph_evidence",
                    "stored periodic graph migration evidence",
                )
            if (
                not isinstance(source_authority, NativePeriodicSource)
                or source_authority.binding_id != final_source.binding_id
            ):
                raise ValueError(
                    "Replayed Delaunay closure changed its accepted source context."
                )
        except Exception as error:
            local_error = str(error)
    elif route == "periodic_power":
        try:
            if not isinstance(source.domain, PeriodicPointOrbits):
                raise TypeError("Stored periodic power source lacks point orbits.")
            power = _owner_local_periodic_power_candidate(
                source,
                rank,
                execution_group,
                source_content_id,
                axis_name=f"{axis_name}_power",
            )
            candidate = power["candidate"]
            source_map = power["source_map"]
            if source_map is None:
                source_authority_value = power["source"]
                if (
                    not isinstance(source_authority, NativePeriodicSource)
                    or view.source_map_id is not None
                    or view.vertex_ids.size
                    or source_authority.binding_id != source_authority_value.binding_id
                ):
                    raise ValueError(
                        "Replayed empty power shard changed its source authority."
                    )
                source_context_id = source_authority_value.binding_id
            else:
                if (
                    type(source_authority) is not ExactPowerCellGeometrySource
                    or source_authority.source_id != source.source_id
                    or source_authority.source_revision != source.source_revision
                    or source_authority.source_map_id != source_map.source_map_id
                ):
                    raise ValueError(
                        "Replayed exact power map changed its source identity."
                    )
                source_map_id = source_map.source_map_id
                context = source_map.periodic_source_context
                authority_context = source_authority.periodic_source_context
                if (
                    context is None
                    or authority_context is None
                    or context.context_id != authority_context.context_id
                ):
                    raise ValueError(
                        "Replayed exact power map lost its periodic context."
                    )
                source_context_id = context.context_id
                source_authority_value = source_map
            mesh_data = _owner_local_periodic_power_mesh_data(power, rank)
            periodic_cell = source.domain.cell
            shell_certificate = dict(candidate["periodic_shell_certificate"])
            if graph_partition_plan is not None:
                from .._distribution import (
                    _owner_local_multiset_identity,
                    OwnerLocalGraphMigrationEvidence,
                )

                graph_evidence = view.graph_migration_evidence
                local_error = None
                try:
                    if (
                        not isinstance(graph_evidence, OwnerLocalGraphMigrationEvidence)
                        or graph_evidence.result.plan_id != graph_partition_plan.plan_id
                        or not graph_evidence.result.feasible
                        or graph_evidence.target_part_id != view.part_id
                        or graph_evidence.target_directory_id != view.distribution_id
                        or graph_evidence.periodic_face_coverage_id
                        != view.periodic_face_coverage_id
                        or graph_evidence.group_id != execution_group.spec.group_id
                    ):
                        raise ValueError(
                            "Stored periodic power graph evidence differs from its target view."
                        )
                    graph_ids = np.asarray(
                        graph_evidence.result.entity_ids, dtype=np.int64
                    )
                    graph_owners = np.asarray(
                        graph_evidence.result.owners, dtype=np.int32
                    )
                    owned_rows = np.asarray(candidate["source_owned"], dtype=np.bool_)
                    source_ids = np.asarray(candidate["point_ids"], dtype=np.int64)[
                        owned_rows
                    ]
                    source_owners = np.asarray(
                        candidate["point_owner_ranks"], dtype=np.int32
                    )[owned_rows]
                    if (
                        graph_ids.ndim != 1
                        or graph_owners.shape != graph_ids.shape
                        or not np.array_equal(np.sort(graph_ids), np.sort(source_ids))
                    ):
                        raise ValueError(
                            "Stored power graph labels differ from current owned sites."
                        )
                    graph_owner_records = tuple(
                        (int(identifier), int(owner))
                        for identifier, owner in zip(
                            graph_ids.tolist(), graph_owners.tolist(), strict=True
                        )
                    )
                    current_owner_records = tuple(
                        (int(identifier), int(owner))
                        for identifier, owner in zip(
                            source_ids.tolist(), source_owners.tolist(), strict=True
                        )
                    )
                except Exception as error:
                    local_error = f"{type(error).__name__}: {error}"
                    graph_evidence = None
                    graph_owner_records = ()
                    current_owner_records = ()
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    f"{axis_name}_power_graph_rows",
                    "stored periodic power graph rows",
                )
                if graph_evidence is None:
                    raise RuntimeError(
                        "Collective power graph validation returned no evidence."
                    )
                owner_header = {
                    "periodic_cell": periodic_cell.cell_id,
                    "graph_plan": graph_partition_plan.plan_id,
                }
                graph_owner_map_id = _owner_local_multiset_identity(
                    "periodic-power-cell-owner-map",
                    owner_header,
                    graph_owner_records,
                    source.source_id,
                    source.source_revision,
                    execution_group,
                    f"{axis_name}_power_graph_result",
                )
                current_owner_map_id = _owner_local_multiset_identity(
                    "periodic-power-cell-owner-map",
                    owner_header,
                    current_owner_records,
                    source.source_id,
                    source.source_revision,
                    execution_group,
                    f"{axis_name}_power_graph_current",
                )
                local_error = (
                    None
                    if graph_owner_map_id == current_owner_map_id
                    and np.array_equal(
                        np.sort(
                            np.asarray(graph_evidence.target_cell_ids, dtype=np.int64)
                        ),
                        np.sort(np.asarray(mesh_data["cell_ids"], dtype=np.int64)),
                    )
                    else "Stored periodic power graph owner map differs from its exact cells."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    f"{axis_name}_power_graph_identity",
                    "stored periodic power graph identity",
                )
        except Exception as error:
            local_error = str(error)
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        axis_name,
        "stored native periodic candidate replay",
    )
    if (
        candidate is None
        or mesh_data is None
        or periodic_cell is None
        or source_context_id is None
        or shell_certificate is None
    ):
        raise RuntimeError(
            "Collective periodic candidate replay returned no local result."
        )
    if (
        source_map_id != view.source_map_id
        or source_context_id != view.source_context_id
        or candidate["candidate_revision_id"] != view.candidate_revision_id
        or not np.array_equal(
            np.asarray(candidate["point_ids"], dtype=np.int64),
            np.asarray(view.source_point_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(candidate["point_owner_ranks"], dtype=np.int32),
            np.asarray(view.source_owner_ranks, dtype=np.int32),
        )
        or not np.array_equal(
            np.asarray(candidate["source_owned"], dtype=np.bool_),
            np.asarray(view.source_owned, dtype=np.bool_),
        )
    ):
        local_error = (
            "Replayed periodic source/support shard changed its accepted identity."
        )
    else:
        local_error = None
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        axis_name,
        "stored periodic source/support identity",
    )
    face_coverage_id, local_cell_faces, owned_cell_rows = (
        _owner_local_periodic_face_graph(
            mesh_data,
            periodic_cell,
            rank,
            execution_group,
            source.source_id,
            source.source_revision,
            source_content_id,
            f"{axis_name}_faces",
        )
    )
    shell_certificate["quotient_face_coverage_id"] = face_coverage_id
    cell_ids = np.asarray(mesh_data["cell_ids"], dtype=np.int64)
    cell_key_offsets = np.asarray(mesh_data["cell_key_offsets"], dtype=np.int64)
    cell_key_sources = np.asarray(mesh_data["cell_key_source_ids"], dtype=np.int64)
    cell_key_windings = np.asarray(
        mesh_data["cell_key_relative_windings"], dtype=np.int64
    )
    topology_payload = tuple(
        (
            int(cell_ids[row]),
            tuple(
                (
                    int(cell_key_sources[index]),
                    tuple(int(value) for value in cell_key_windings[index]),
                )
                for index in range(
                    int(cell_key_offsets[row]), int(cell_key_offsets[row + 1])
                )
            ),
            local_cell_faces[position],
        )
        for position, row in enumerate(owned_cell_rows)
    )
    topology_id = _owner_local_multiset_identity(
        "periodic-quotient-topology",
        {
            "periodic_cell": periodic_cell.cell_id,
            "periodic_axes": periodic_cell.periodic_axes,
            "lattice_vectors": array_tree_fingerprint(
                np.asarray(periodic_cell.vectors, dtype=np.float64)
            ),
            "dimension": periodic_cell.ambient_dimension,
        },
        tuple((record[0], record) for record in topology_payload),
        source.source_id,
        None,
        execution_group,
        f"{axis_name}_topology",
    )
    vertex_owner_ranks = np.asarray(mesh_data["vertex_owner_ranks"], dtype=np.int32)
    owned_vertex_rows = np.flatnonzero(vertex_owner_ranks == rank)
    exact_vertex_coordinates = tuple(mesh_data["exact_vertex_coordinates"])
    vertex_ids = np.asarray(mesh_data["vertex_ids"], dtype=np.int64)
    vertex_coordinate_bits = np.asarray(
        mesh_data["vertex_coordinates"], dtype=np.float64
    ).view(np.uint64)
    if vertex_coordinate_bits.shape != (
        vertex_ids.size,
        periodic_cell.ambient_dimension,
    ):
        raise ValueError("Replayed exact geometry float bits do not match vertex IDs.")
    vertex_source_ids = np.asarray(mesh_data["vertex_source_ids"], dtype=np.int64)
    vertex_absolute_windings = np.asarray(
        mesh_data["vertex_absolute_windings"], dtype=np.int64
    )
    geometry_records = tuple(
        (
            int(vertex_ids[row]),
            (
                tuple(
                    (value.numerator, value.denominator)
                    for value in exact_vertex_coordinates[int(row)]
                ),
                tuple(int(value) for value in vertex_coordinate_bits[int(row)]),
                int(vertex_source_ids[row]),
                tuple(int(value) for value in vertex_absolute_windings[int(row)]),
            ),
        )
        for row in owned_vertex_rows
    )
    geometry_id = _owner_local_multiset_identity(
        "periodic-exact-geometry",
        {
            "coordinate_contract": view.coordinate_contract_id,
            "periodic_cell": periodic_cell.cell_id,
            "dimension": periodic_cell.ambient_dimension,
            "source_content": source_content_id,
        },
        geometry_records,
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_geometry",
    )
    periodic_topology_id = canonical_fingerprint(
        {
            "kind": "periodic-owner-local-topology",
            "topology": topology_id,
            "periodic_cell": periodic_cell.cell_id,
        }
    )
    entity_set_ids = {
        degree: canonical_fingerprint(
            {
                "kind": "periodic-owner-local-entity-set",
                "topology": topology_id,
                "dimension": degree,
            }
        )
        for degree in range(periodic_cell.ambient_dimension + 1)
    }
    if route == "periodic_delaunay":
        from fractions import Fraction

        adaptive_result = view.adaptive_result
        source_distribution_id = view.distribution_id
        source_part_id = view.part_id
        if adaptive_result is not None:
            source_directory = ShardedOwnerDirectoryWitness(
                execution_group,
                rank,
                source_manifest_id,
                "periodic-owner-directory",
                candidate["point_ids"],
                candidate["point_owner_ranks"],
                candidate["source_owned"],
                mesh_data["cell_ids"],
                mesh_data["cell_owner_ranks"],
                mesh_data["vertex_ids"],
                mesh_data["vertex_owner_ranks"],
                source_map_id=source_map_id,
                source_context_id=source_context_id,
                axis_name=f"{axis_name}_source_directory",
            )
            source_distribution_id = source_directory.directory_id
            source_part_id = canonical_fingerprint(
                {
                    "kind": "owner-local-periodic-part",
                    "source_manifest": source_content_id,
                    "topology": topology_id,
                    "geometry": geometry_id,
                    "distribution": source_distribution_id,
                    "specification": adaptive_result.specification_id,
                    "route": "periodic_delaunay",
                    "face_coverage_id": face_coverage_id,
                }
            )
            local_error = (
                None
                if (
                    source_distribution_id == adaptive_result.source_distribution_id
                    and source_part_id == adaptive_result.source_part_id
                    and topology_id == adaptive_result.source_topology_id
                    and geometry_id == adaptive_result.source_geometry_id
                    and periodic_topology_id
                    == adaptive_result.source_periodic_topology_id
                    and source.source_id == adaptive_result.source_id
                    and source.source_revision == adaptive_result.source_revision
                )
                else "Replayed adaptive source part authority changed."
            )
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                f"{axis_name}_source_authority",
                "stored periodic adaptive source authority",
            )

        source_point_exact = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in np.asarray(candidate["points"], dtype=np.float64)
        )
        replayed_candidate = OwnerLocalPeriodicCandidate(
            rank,
            len(execution_group.spec.process_indices),
            periodic_cell.ambient_dimension,
            mesh_data["local_mesh"],
            periodic_cell,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_manifest_id=source_content_id,
            source_mesh_id=view.source_mesh_id,
            source_distribution_id=source_distribution_id,
            source_part_id=source_part_id,
            source_geometry_id=geometry_id,
            coordinate_contract_id=view.coordinate_contract_id,
            source_topology_id=topology_id,
            periodic_topology_id=periodic_topology_id,
            face_coverage_id=face_coverage_id,
            candidate_revision_id=candidate["candidate_revision_id"],
            periodic_cell_id=periodic_cell.cell_id,
            source_context_id=source_context_id,
            geometry_layout_id=mesh_data["geometry_layout_id"],
            source_point_coordinates=candidate["points"],
            source_point_ids=candidate["point_ids"],
            source_owner_ranks=candidate["point_owner_ranks"],
            source_owned=candidate["source_owned"],
            source_point_exact_coordinates=source_point_exact,
            cell_ids=mesh_data["cell_ids"],
            cell_owner_ranks=mesh_data["cell_owner_ranks"],
            cell_key_offsets=mesh_data["cell_key_offsets"],
            cell_key_source_ids=mesh_data["cell_key_source_ids"],
            cell_key_relative_windings=mesh_data["cell_key_relative_windings"],
            cell_windings=mesh_data["cell_windings"],
            vertex_ids=mesh_data["vertex_ids"],
            vertex_owner_ranks=mesh_data["vertex_owner_ranks"],
            vertex_source_ids=mesh_data["vertex_source_ids"],
            vertex_absolute_windings=mesh_data["vertex_absolute_windings"],
            exact_vertex_coordinates=mesh_data["exact_vertex_coordinates"],
            exact_cell_measures=tuple(mesh_data["exact_cell_measures"]),
            owned_cell_ids=mesh_data["graph_cell_ids"],
            owned_cell_owner_ranks=mesh_data["graph_cell_owner_ranks"],
            owned_cell_weights=mesh_data["graph_cell_weights"],
            cell_neighbor_offsets=mesh_data["graph_cell_offsets"],
            cell_neighbor_ids=mesh_data["graph_neighbor_ids"],
            cell_neighbor_owner_ranks=mesh_data["graph_neighbor_owner_ranks"],
            cell_neighbor_weights=mesh_data["graph_neighbor_weights"],
        )
        shell_certificate["owner_local_candidate_id"] = replayed_candidate.candidate_id
        if adaptive_result is not None:
            local_error = (
                None
                if (
                    replayed_candidate.candidate_id == adaptive_result.candidate_id
                    and replayed_candidate.source_local_mesh_id
                    == adaptive_result.source_local_mesh_id
                    and replayed_candidate.source_local_topology_id
                    == adaptive_result.source_local_topology_id
                    and replayed_candidate.source_local_geometry_id
                    == adaptive_result.source_local_geometry_id
                    and replayed_candidate.geometry_layout_id
                    == adaptive_result.source_geometry_layout_id
                    and replayed_candidate.coordinate_contract_id
                    == adaptive_result.coordinate_contract_id
                    and replayed_candidate.source_mesh_id
                    == adaptive_result.source_mesh_id
                )
                else "Replayed periodic source candidate differs from adaptive result."
            )
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                axis_name,
                "stored periodic adaptive source candidate",
            )
            return replayed_candidate
    local_error = None
    if (
        topology_id != view.source_topology_id
        or geometry_id != view.source_geometry_id
        or periodic_topology_id != view.periodic_topology_id
        or face_coverage_id != view.periodic_face_coverage_id
        or canonical_fingerprint(shell_certificate)
        != canonical_fingerprint(view.periodic_shell_certificate)
    ):
        local_error = "Replayed periodic topology, geometry, or shell evidence changed."
    array_pairs = (
        ("cell_rows", "cell_rows"),
        ("cell_ids", "cell_ids"),
        ("owned", "owned"),
        ("cell_measures", "cell_measures"),
        ("cell_owner_ranks", "cell_owner_ranks"),
        ("vertex_rows", "vertex_rows"),
        ("vertex_ids", "vertex_ids"),
        ("vertex_coordinates", "vertex_coordinates"),
        ("coordinate_rows", "coordinate_rows"),
        ("coordinates", "coordinates"),
        ("vertex_source_ids", "vertex_source_ids"),
        ("vertex_owner_ranks", "vertex_owner_ranks"),
        ("vertex_absolute_windings", "vertex_absolute_windings"),
        ("cell_key_offsets", "cell_key_offsets"),
        ("cell_key_source_ids", "cell_key_source_ids"),
        ("cell_key_relative_windings", "cell_key_relative_windings"),
        ("periodic_cell_ids", "periodic_cell_ids"),
        ("vertex_representative_ids", "vertex_representative_ids"),
        ("vertex_windings", "vertex_windings"),
        ("cell_windings", "cell_windings"),
        ("support_fractional_lower", "support_fractional_lower"),
        ("support_fractional_upper", "support_fractional_upper"),
    )
    for view_name, data_name in array_pairs:
        expected = mesh_data[data_name]
        stored = getattr(view, view_name)
        if stored is None or not np.array_equal(np.asarray(stored), np.asarray(expected)):
            local_error = f"Stored periodic {view_name} differs from native replay."
            break
    expected_exact_vertex = (
        None
        if not np.asarray(mesh_data["vertex_ids"]).size
        else tuple(tuple(row) for row in mesh_data["exact_vertex_coordinates"])
    )
    if view.exact_vertex_coordinates != expected_exact_vertex or tuple(
        view.exact_coordinates
    ) != tuple(mesh_data["exact_coordinates"]):
        local_error = "Stored exact coordinate banks differ from native replay."
    if local_error is None:
        for view_name, data_name in (
            ("local_entity_ids", "local_entity_ids"),
            ("entity_vertex_offsets", "entity_vertex_offsets"),
            ("entity_vertex_values", "entity_vertex_values"),
        ):
            stored = dict(getattr(view, view_name))
            expected = mesh_data[data_name]
            if set(stored) != set(expected) or any(
                not np.array_equal(np.asarray(stored[key]), np.asarray(expected[key]))
                for key in expected
            ):
                local_error = f"Stored periodic {view_name} differs from native replay."
                break
    if local_error is None:
        expected_blocks = tuple(mesh_data["blocks"])
        if len(view.blocks) != len(expected_blocks):
            local_error = "Stored periodic block count differs from native replay."
        for block, expected in zip(view.blocks, expected_blocks):
            if local_error is not None:
                break
            if (
                block.name != expected["name"]
                or block.cell_kind != expected["cell_kind"]
                or block.element.element_id != expected["element"].element_id
                or canonical_fingerprint(block.chart_arguments)
                != canonical_fingerprint(expected.get("chart_arguments"))
            ):
                local_error = (
                    "Stored periodic cell element identity differs from native replay."
                )
                break
            for name in (
                "cell_rows",
                "cell_ids",
                "vertices",
                "vertex_valid",
                "coordinate_routes",
                "element_rows",
                "source_cell_ids",
                "cell_measures",
                "cell_vertex_shifts",
            ):
                stored = getattr(block, name)
                replayed = expected.get(name)
                if (stored is None) != (replayed is None) or (
                    stored is not None
                    and not np.array_equal(np.asarray(stored), np.asarray(replayed))
                ):
                    local_error = (
                        f"Stored periodic block {name} differs from native replay."
                    )
                    break
            if local_error is not None:
                break
            if block.cell_kind == "polyhedron":
                face_offsets = np.asarray(expected["face_vertex_offsets"], dtype=np.int64)
                face_values = np.asarray(expected["face_vertex_values"], dtype=np.int32)
                expected_loops = tuple(
                    face_values[face_offsets[row] : face_offsets[row + 1]]
                    for row in range(face_offsets.size - 1)
                )
                cell_face_offsets = np.asarray(
                    expected["cell_face_offsets"], dtype=np.int64
                )
                cell_face_values = np.asarray(
                    expected["cell_face_values"], dtype=np.int64
                )
                expected_faces = tuple(
                    tuple(
                        expected_loops[int(face)]
                        for face in cell_face_values[
                            cell_face_offsets[row] : cell_face_offsets[row + 1]
                        ]
                    )
                    for row in range(cell_face_offsets.size - 1)
                )
                expected_signs = tuple(
                    np.asarray(expected["cell_face_signs"], dtype=np.int8)[
                        cell_face_offsets[row] : cell_face_offsets[row + 1]
                    ]
                    for row in range(cell_face_offsets.size - 1)
                )
                if (
                    not np.array_equal(
                        np.asarray(block.polyhedron_face_ids),
                        np.asarray(expected["face_ids"], dtype=np.int64),
                    )
                    or len(block.polyhedron_face_vertex_loops or ())
                    != len(expected_loops)
                    or any(
                        not np.array_equal(np.asarray(left), right)
                        for left, right in zip(
                            block.polyhedron_face_vertex_loops or (),
                            expected_loops,
                            strict=True,
                        )
                    )
                    or len(block.polyhedron_faces or ()) != len(expected_faces)
                    or any(
                        len(stored_cell) != len(replayed_cell)
                        or any(
                            not np.array_equal(np.asarray(left), right)
                            for left, right in zip(
                                stored_cell, replayed_cell, strict=True
                            )
                        )
                        for stored_cell, replayed_cell in zip(
                            block.polyhedron_faces or (),
                            expected_faces,
                            strict=True,
                        )
                    )
                    or len(block.polyhedron_face_signs or ()) != len(expected_signs)
                    or any(
                        not np.array_equal(np.asarray(left), right)
                        for left, right in zip(
                            block.polyhedron_face_signs or (),
                            expected_signs,
                            strict=True,
                        )
                    )
                ):
                    local_error = (
                        "Stored periodic polyhedral incidence differs from native replay."
                    )
                    break
    if local_error is None:
        for view_name, data_name in (
            ("owned_cell_ids", "graph_cell_ids"),
            ("owned_cell_owner_ranks", "graph_cell_owner_ranks"),
            ("owned_cell_weights", "graph_cell_weights"),
            ("cell_neighbor_offsets", "graph_cell_offsets"),
            ("cell_neighbor_ids", "graph_neighbor_ids"),
            ("cell_neighbor_owner_ranks", "graph_neighbor_owner_ranks"),
            ("cell_neighbor_weights", "graph_neighbor_weights"),
        ):
            if not np.array_equal(
                np.asarray(getattr(view, view_name)),
                np.asarray(mesh_data[data_name]),
            ):
                local_error = f"Stored periodic {view_name} differs from face/CSR replay."
                break
    if local_error is None:
        try:
            local_mesh, local_geometry = view.reconstruct_local_geometry()
            replay_mesh = mesh_data["local_mesh"]
            if (local_mesh is None) != (replay_mesh is None):
                raise ValueError("Replayed local mesh emptiness changed.")
            if local_mesh is None:
                if local_geometry is not None or view.local_geometry_id is not None:
                    raise ValueError("Empty periodic shard gained a local geometry.")
            else:
                if replay_mesh is None:
                    raise ValueError("Replayed nonempty mesh is absent.")
                if (
                    local_mesh.topological_dimension != replay_mesh.topological_dimension
                    or (local_mesh.periodic_topology is None)
                    != (replay_mesh.periodic_topology is None)
                    or (
                        local_mesh.periodic_topology is None
                        and local_mesh.topology_id != replay_mesh.topology_id
                    )
                    or (
                        local_mesh.periodic_topology is not None
                        and replay_mesh.periodic_topology is not None
                        and (
                            identification_id(local_mesh.periodic_topology.cell)
                            != identification_id(replay_mesh.periodic_topology.cell)
                            or local_mesh.periodic_topology.lifted_topology_id
                            != replay_mesh.periodic_topology.lifted_topology_id
                        )
                    )
                    or not np.array_equal(
                        np.asarray(local_mesh.coordinates),
                        np.asarray(replay_mesh.coordinates),
                    )
                    or not np.array_equal(
                        np.asarray(local_mesh.vertex_global_ids),
                        np.asarray(replay_mesh.vertex_global_ids),
                    )
                    or tuple(
                        (block.name, block.cell_kind, block.block_id)
                        for block in local_mesh.blocks
                    )
                    != tuple(
                        (block.name, block.cell_kind, block.block_id)
                        for block in replay_mesh.blocks
                    )
                ):
                    raise ValueError("Replayed owner-local mesh connectivity changed.")
                replay_geometry = (
                    CellGeometrySpec.power(replay_mesh, source_map)
                    if source_map is not None
                    else CellGeometrySpec.affine(replay_mesh)
                )
                if (
                    local_geometry is None
                    or cell_geometry_id(local_geometry)
                    != cell_geometry_id(replay_geometry)
                    or cell_geometry_id(local_geometry) != view.local_geometry_id
                ):
                    raise ValueError("Replayed owner-local coordinate map changed.")
        except Exception as error:
            local_error = str(error)
    if local_error is None:
        if (
            view.geometry_layout_id != mesh_data["geometry_layout_id"]
            or dict(view.entity_set_ids) != entity_set_ids
            or view.periodic_lift_id != periodic_cell.cell_id
            or not np.array_equal(
                np.asarray(view.periodic_lift_matrix),
                np.asarray(periodic_cell.vectors),
            )
            or not view.coordinates_are_lifted
            or view.candidate_revision_id != candidate["candidate_revision_id"]
            or view.certified_periodic_margin != candidate["certified_periodic_margin"]
        ):
            local_error = "Stored periodic view metadata differs from native replay."
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        axis_name,
        "stored periodic geometry and quotient replay",
    )


def _owner_local_periodic_candidate_from_accepted_view(
    view: OwnerLocalCellGeometry,
    source: NativePeriodicSource,
    mesh_data: dict[str, Any],
    source_content_id: str,
    rank: int,
    part_count: int,
    /,
) -> OwnerLocalPeriodicCandidate:
    from ...discretization._view_support import OwnerLocalCellGeometry
    from .._periodic import OwnerLocalPeriodicCandidate, PeriodicPointOrbits

    if (
        not isinstance(view, OwnerLocalCellGeometry)
        or not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.point_owner_ranks is None
        or not isinstance(source_content_id, str)
        or not source_content_id
        or source_content_id != view.owner_directory_data.source_manifest_id
        or view.source_map_id is not None
        or view.source_context_id != source.binding_id
        or view.source_id != source.source_id
        or view.source_revision != source.source_revision
        or view.owner_directory_data.part_count != part_count
        or view.candidate_revision_id is None
        or view.periodic_topology_id is None
        or view.periodic_face_coverage_id is None
    ):
        raise TypeError("Accepted Delaunay graph source is not a current typed view.")
    source_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
    source_owners = np.asarray(source.point_owner_ranks, dtype=np.int32)
    source_points = np.asarray(source.domain.points, dtype=np.float64)
    view_source_ids = np.asarray(view.source_point_ids, dtype=np.int64)
    view_source_owners = np.asarray(view.source_owner_ranks, dtype=np.int32)
    view_owner_by_id = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            view_source_ids.tolist(), view_source_owners.tolist(), strict=True
        )
    }
    if (
        source_owners.shape != source_ids.shape
        or source_points.shape != (source_ids.size, source.ambient_dimension)
        or set(view_owner_by_id) != set(source_ids.tolist())
        or any(
            view_owner_by_id[int(identifier)] != int(owner)
            for identifier, owner in zip(
                source_ids.tolist(), source_owners.tolist(), strict=True
            )
        )
    ):
        raise ValueError("Accepted Delaunay source rows differ from their local view.")
    exact_source_points = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in source_points
    )
    local_mesh = mesh_data["local_mesh"]
    if (
        None if local_mesh is None else local_mesh.mesh_id
    ) != view.local_mesh_id or not np.array_equal(
        np.asarray(mesh_data["cell_ids"], dtype=np.int64),
        np.asarray(view.cell_ids, dtype=np.int64),
    ):
        raise ValueError("Accepted Delaunay mesh data changed the local carrier.")
    # Adapted views key midpoint vertices by themselves; their exact roots are
    # the cumulative coordinate ancestry carried with the accepted mesh data.
    # An owner without resident vertices has no ancestry to carry.
    stencil = (
        mesh_data["source_coordinate_stencil"]
        if view.adaptive_result is not None and np.asarray(view.vertex_ids).size
        else None
    )
    return OwnerLocalPeriodicCandidate(
        rank,
        part_count,
        source.ambient_dimension,
        local_mesh,
        source.domain.cell,
        source_id=view.source_id,
        source_revision=view.source_revision,
        source_manifest_id=source_content_id,
        source_mesh_id=view.source_mesh_id,
        source_distribution_id=view.distribution_id,
        source_part_id=view.part_id,
        source_geometry_id=view.source_geometry_id,
        coordinate_contract_id=view.coordinate_contract_id,
        source_topology_id=view.source_topology_id,
        periodic_topology_id=view.periodic_topology_id,
        face_coverage_id=view.periodic_face_coverage_id,
        candidate_revision_id=view.candidate_revision_id,
        periodic_cell_id=source.domain.cell.cell_id,
        source_context_id=source.binding_id,
        geometry_layout_id=view.geometry_layout_id,
        source_point_coordinates=source_points,
        source_point_ids=source_ids,
        source_owner_ranks=source_owners,
        source_owned=source_owners == rank,
        source_point_exact_coordinates=exact_source_points,
        cell_ids=mesh_data["cell_ids"],
        cell_owner_ranks=mesh_data["cell_owner_ranks"],
        cell_key_offsets=mesh_data["cell_key_offsets"],
        cell_key_source_ids=mesh_data["cell_key_source_ids"],
        cell_key_relative_windings=mesh_data["cell_key_relative_windings"],
        cell_windings=mesh_data["cell_windings"],
        vertex_ids=mesh_data["vertex_ids"],
        vertex_owner_ranks=mesh_data["vertex_owner_ranks"],
        vertex_source_ids=mesh_data["vertex_source_ids"],
        vertex_absolute_windings=mesh_data["vertex_absolute_windings"],
        exact_vertex_coordinates=tuple(mesh_data["exact_vertex_coordinates"]),
        exact_cell_measures=tuple(mesh_data["exact_cell_measures"]),
        owned_cell_ids=mesh_data["graph_cell_ids"],
        owned_cell_owner_ranks=mesh_data["graph_cell_owner_ranks"],
        owned_cell_weights=mesh_data["graph_cell_weights"],
        cell_neighbor_offsets=mesh_data["graph_cell_offsets"],
        cell_neighbor_ids=mesh_data["graph_neighbor_ids"],
        cell_neighbor_owner_ranks=mesh_data["graph_neighbor_owner_ranks"],
        cell_neighbor_weights=mesh_data["graph_neighbor_weights"],
        vertex_coordinate_stencil=stencil,
        exact_vertex_coordinate_offsets=(
            None
            if stencil is None
            else tuple(mesh_data["exact_vertex_coordinate_offsets"])
        ),
    )


def _owner_local_periodic_source_from_view(
    view: OwnerLocalCellGeometry,
    /,
) -> tuple[NativePeriodicSource, ExactPowerCellGeometrySource | None]:
    from fractions import Fraction

    from ...discretization._exact_power_geometry import (
        ExactPowerCellGeometrySource,
    )
    from ...discretization._view_support import OwnerLocalCellGeometry
    from .._periodic import PeriodicPointOrbits

    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("view must be OwnerLocalCellGeometry.")
    source = view.source_value
    source_geometry = view.source_geometry_authority
    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.source_id != view.source_id
        or source.source_revision != view.source_revision
    ):
        raise TypeError("Periodic local view lacks its typed authored source operands.")
    source.validate_restored()
    view_ids = np.asarray(view.source_point_ids, dtype=np.int64)
    view_owners = np.asarray(view.source_owner_ranks, dtype=np.int32)
    view_owned = np.asarray(view.source_owned, dtype=np.bool_)
    parts = view.owner_directory_data.part_count

    def require_view_bank(
        domain: PeriodicPointOrbits,
        owners: np.ndarray,
        /,
    ) -> None:
        domain_ids = np.asarray(domain.point_ids, dtype=np.int64)
        if (
            domain.cell.cell_id != source.domain.cell.cell_id
            or domain_ids.ndim != 1
            or owners.shape != domain_ids.shape
            or view_ids.shape != view_owners.shape
            or view_ids.shape != view_owned.shape
            or np.unique(domain_ids).size != domain_ids.size
            or np.unique(view_ids).size != view_ids.size
            or set(domain_ids.tolist()) != set(view_ids.tolist())
            or np.any(owners < 0)
            or np.any(owners >= parts)
            or np.any(view_owners < 0)
            or np.any(view_owners >= parts)
            or not np.array_equal(view_owned, view_owners == view.owner_rank)
        ):
            raise ValueError("Periodic local source owner banks are malformed.")
        owner_by_id = {
            int(identifier): int(owner)
            for identifier, owner in zip(
                view_ids.tolist(), view_owners.tolist(), strict=True
            )
        }
        if any(
            owner_by_id[int(identifier)] != int(owner)
            for identifier, owner in zip(
                domain_ids.tolist(), owners.tolist(), strict=True
            )
        ):
            raise ValueError("Periodic local source owners differ from the view.")

    def require_authored_rows(
        domain: PeriodicPointOrbits,
        owners: np.ndarray,
        weights: np.ndarray | None,
        /,
    ) -> None:
        if domain.cell.cell_id != source.domain.cell.cell_id:
            raise ValueError("Periodic authored source uses a different lattice.")
        domain_ids = np.asarray(domain.point_ids, dtype=np.int64)
        domain_points = np.asarray(domain.points, dtype=np.float64)
        authored_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
        authored_points = np.asarray(source.domain.points, dtype=np.float64)
        if (
            domain_points.shape != (domain_ids.size, source.ambient_dimension)
            or authored_points.shape != (authored_ids.size, source.ambient_dimension)
            or owners.shape != domain_ids.shape
        ):
            raise ValueError("Periodic authored source arrays are malformed.")
        domain_row_by_id = {
            int(identifier): row for row, identifier in enumerate(domain_ids.tolist())
        }
        authored_owner_by_id = (
            None
            if source.point_owner_ranks is None
            else {
                int(identifier): int(owner)
                for identifier, owner in zip(
                    authored_ids.tolist(),
                    np.asarray(source.point_owner_ranks, dtype=np.int32).tolist(),
                    strict=True,
                )
            }
        )
        owner_by_id = {
            int(identifier): int(owner)
            for identifier, owner in zip(
                domain_ids.tolist(), owners.tolist(), strict=True
            )
        }
        authored_weight_values = (
            None
            if source.site_weights is None
            else np.asarray(source.site_weights, dtype=np.float64)
        )
        if (
            authored_weight_values is not None
            and authored_weight_values.shape != authored_ids.shape
        ):
            raise ValueError("Periodic authored site weights differ from site IDs.")
        if authored_weight_values is None and weights is not None:
            raise ValueError("Periodic source authority introduced site weights.")
        if (
            authored_weight_values is not None
            and weights is None
            and np.any(authored_weight_values != 0.0)
        ):
            raise ValueError("Periodic source authority dropped nonzero site weights.")
        authority_weights = (
            None if weights is None else np.asarray(weights, dtype=np.float64)
        )
        if authority_weights is not None and authority_weights.shape != domain_ids.shape:
            raise ValueError("Periodic source authority weights differ from site IDs.")
        for row, identifier_value in enumerate(authored_ids.tolist()):
            identifier = int(identifier_value)
            domain_row = domain_row_by_id.get(identifier)
            if domain_row is None:
                raise ValueError("Periodic source authority omitted an authored site.")
            if not np.array_equal(
                authored_points[row].view(np.uint64),
                domain_points[domain_row].view(np.uint64),
            ):
                raise ValueError("Periodic source authority changed an authored site.")
            if (
                authored_owner_by_id is not None
                and authored_owner_by_id[identifier] != owner_by_id[identifier]
            ):
                raise ValueError(
                    "Periodic source authority changed authored site ownership."
                )
            if authored_weight_values is not None and authority_weights is not None:
                if int(
                    np.asarray(authored_weight_values[row], dtype=np.float64).view(
                        np.uint64
                    )
                ) != int(
                    np.asarray(authority_weights[domain_row], dtype=np.float64).view(
                        np.uint64
                    )
                ):
                    raise ValueError(
                        "Periodic source authority changed an authored weight."
                    )

    if source_geometry is None:
        if (
            view.source_map_id is not None
            or view.source_context_id != source.binding_id
            or source.point_owner_ranks is None
        ):
            raise ValueError("Periodic source view map/context binding is stale.")
        owners = np.asarray(source.point_owner_ranks, dtype=np.int32)
        require_view_bank(source.domain, owners)
        require_authored_rows(source.domain, owners, source.site_weights)
        return source, None

    if isinstance(source_geometry, NativePeriodicSource):
        source_geometry.validate_restored()
        if (
            not isinstance(source_geometry.domain, PeriodicPointOrbits)
            or source_geometry.source_id != source.source_id
            or source_geometry.source_revision != source.source_revision
            or source_geometry.point_owner_ranks is None
            or view.source_map_id is not None
            or source_geometry.binding_id != view.source_context_id
        ):
            raise ValueError(
                "Periodic support source differs from its compact local view."
            )
        owners = np.asarray(source_geometry.point_owner_ranks, dtype=np.int32)
        require_view_bank(source_geometry.domain, owners)
        require_authored_rows(
            source_geometry.domain,
            owners,
            None
            if source_geometry.site_weights is None
            else np.asarray(source_geometry.site_weights, dtype=np.float64),
        )
        return source_geometry, None

    if type(source_geometry) is not ExactPowerCellGeometrySource:
        raise TypeError(
            "Periodic local geometry authority must be a native source or exact power map."
        )
    context = source_geometry.periodic_source_context
    if (
        context is None
        or not isinstance(context.periodic_source, PeriodicPointOrbits)
        or source_geometry.source_id != source.source_id
        or source_geometry.source_revision != source.source_revision
        or source_geometry.source_map_id != view.source_map_id
        or context.context_id != view.source_context_id
        or context.periodic_source.cell.cell_id != source.domain.cell.cell_id
        or type(context.weights_present) is not bool
    ):
        raise ValueError(
            "Exact power source operands differ from the compact local view."
        )
    periodic_source = context.periodic_source
    context_ids = np.asarray(periodic_source.point_ids, dtype=np.int64)
    exact_weights = tuple(context.exact_source_weights)
    if len(exact_weights) != context_ids.size or any(
        not isinstance(value, Fraction) for value in exact_weights
    ):
        raise ValueError("Exact periodic power weights differ from the source site bank.")
    sorted_ids = np.sort(context_ids, kind="stable")
    weight_by_id = {
        int(identifier): float(weight)
        for identifier, weight in zip(sorted_ids.tolist(), exact_weights, strict=True)
    }
    weights = (
        None
        if not context.weights_present
        else np.asarray(
            [weight_by_id[int(identifier)] for identifier in context_ids.tolist()],
            dtype=np.float64,
        )
    )
    view_owner_by_id = {
        int(identifier): int(owner)
        for identifier, owner in zip(view_ids.tolist(), view_owners.tolist(), strict=True)
    }
    try:
        owners = np.asarray(
            [view_owner_by_id[int(identifier)] for identifier in context_ids.tolist()],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError(
            "Exact power context site is absent from its local owner bank."
        ) from error
    require_view_bank(periodic_source, owners)
    require_authored_rows(periodic_source, owners, weights)
    expanded_source = NativePeriodicSource(
        periodic_source,
        source.source_id,
        source.source_revision,
        site_weights=weights,
        point_owner_ranks=owners,
    )
    expanded_source.validate_restored()
    return expanded_source, source_geometry


def _repartition_owner_local_periodic_assembly(
    assembly: MeshAssembly,
    partition_evidence: OwnerLocalGraphPartitionEvidence,
    execution_group: ExecutionGroup,
    /,
    *,
    current_specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    graph_partition_plan: GraphPartitionPlan,
    axis_name: str = "owner_local_periodic_repartition",
) -> MeshAssembly:
    import jax

    from ..._execution_runtime import ExecutionGroup
    from ...discretization._cell_geometry_validity import cell_geometry_id
    from ...discretization._exact_power_geometry import (
        ExactPowerCellGeometrySource,
    )
    from ...discretization._view_support import OwnerLocalCellGeometry
    from ...graph._partition import GraphPartitionPlan
    from ...meshing._assembly import MeshAssembly
    from .._distribution import (
        _owner_group_fail_if_any,
        _owner_group_fail_with_reasons,
        OwnerLocalGraphMigrationEvidence,
        OwnerLocalGraphPartitionEvidence,
    )
    from ._native_periodic_power import (
        _owner_local_periodic_power_candidate,
        _owner_local_periodic_power_mesh_data,
    )

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    members = execution_group.spec.process_indices
    try:
        rank = members.index(jax.process_index())
    except ValueError as error:
        raise ValueError(
            "The current process is outside the periodic graph group."
        ) from error
    local_error = None
    view: OwnerLocalCellGeometry | None = None
    source: NativePeriodicSource | None = None
    source_map: ExactPowerCellGeometrySource | None = None
    source_entry: CompositionEntry | None = None
    source_composition: Composition | None = None
    coordinate_contract: SpatialCoordinateContract | None = None
    source_domain: PeriodicPointOrbits | None = None
    source_candidate_revision_id: str | None = None
    source_face_coverage_id: str | None = None
    source_shell_certificate: Mapping[str, Any] | None = None
    route = ""
    try:
        if (
            not isinstance(assembly, MeshAssembly)
            or not isinstance(partition_evidence, OwnerLocalGraphPartitionEvidence)
            or not isinstance(
                current_specification, (SurfaceMeshingSpec, VolumeMeshingSpec)
            )
            or not isinstance(graph_partition_plan, GraphPartitionPlan)
            or not isinstance(axis_name, str)
            or not axis_name.strip()
            or not execution_group.is_member
            or assembly.owner_local is not True
            or assembly.local_rank != rank
            or len(assembly.parts) != 1
            or graph_partition_plan.part_count != len(members)
        ):
            raise TypeError(
                "Periodic repartition requires a current owner-local assembly."
            )
        part = assembly.parts[0]
        if not isinstance(part.carrier, OwnerLocalCellGeometry):
            raise TypeError("Periodic repartition requires OwnerLocalCellGeometry.")
        view = part.carrier
        if (
            view.owner_rank != rank
            or view.owner_directory_data.rank != rank
            or view.owner_directory_data.group_id != execution_group.spec.group_id
            or view.owner_directory_data.part_count != len(members)
            or view.part_id != part.part_id
            or view.coordinate_contract_id != part.coordinate_contract.spatial_id
            or view.specification_id != current_specification.specification_id
            or view.periodic_topology_id is None
            or view.periodic_face_coverage_id is None
            or partition_evidence.source_view_id != view.local_view_id
            or partition_evidence.periodic_face_coverage_id
            != view.periodic_face_coverage_id
            or partition_evidence.owner_directory_id != view.distribution_id
            or partition_evidence.result.plan_id != graph_partition_plan.plan_id
            or not partition_evidence.result.feasible
            or not np.array_equal(
                np.asarray(partition_evidence.result.entity_ids, dtype=np.int64),
                np.asarray(view.owned_cell_ids, dtype=np.int64),
            )
            or np.asarray(partition_evidence.result.owners).shape
            != np.asarray(view.owned_cell_ids).shape
        ):
            raise ValueError("Graph labels do not bind this current periodic owner view.")
        coordinate_contract = part.coordinate_contract
        authority = view.source_geometry_authority
        power_specification = (
            isinstance(current_specification, VolumeMeshingSpec)
            and current_specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
            and set(current_specification.target.cell_families.required) == {"polyhedron"}
        )
        if isinstance(authority, ExactPowerCellGeometrySource):
            if not power_specification:
                raise ValueError(
                    "Exact periodic power geometry requires a polyhedral specification."
                )
            route = "periodic_power"
        elif isinstance(authority, NativePeriodicSource) and power_specification:
            if (
                view.cell_ids.size
                or view.vertex_ids.size
                or view.source_map_id is not None
            ):
                raise ValueError(
                    "Nonempty periodic power geometry requires its exact source map."
                )
            route = "periodic_power"
        elif isinstance(authority, NativePeriodicSource):
            route = "periodic_delaunay"
        else:
            raise TypeError("Periodic owner view has no concrete source authority.")
        source, source_map = _owner_local_periodic_source_from_view(view)
        source_domain_candidate = source.domain
        if not isinstance(source_domain_candidate, PeriodicPointOrbits):
            raise TypeError("Periodic repartition source lacks point orbits.")
        source_domain = source_domain_candidate
        if (
            source.source_id != view.source_id
            or source.source_revision != view.source_revision
            or source_domain.cell.cell_id != view.periodic_lift_id
            or source.ambient_dimension != view.topological_dimension
            or source.point_owner_ranks is None
        ):
            raise ValueError("Periodic source and accepted owner view differ.")
        if (
            route == "periodic_power"
            and source_map is None
            and (view.cell_ids.size or view.vertex_ids.size)
        ) or (route == "periodic_delaunay" and source_map is not None):
            raise ValueError("Periodic owner route lacks its exact source map.")
        if (
            route == "periodic_delaunay"
            and source.site_weights is not None
            and np.any(np.asarray(source.site_weights, dtype=np.float64) != 0.0)
        ):
            raise ValueError("Weighted periodic sources require exact power geometry.")
        source_candidate_revision_id = view.candidate_revision_id
        source_face_coverage_id = view.periodic_face_coverage_id
        source_shell_certificate = view.periodic_shell_certificate
        if (
            source_candidate_revision_id is None
            or source_face_coverage_id is None
            or not isinstance(source_shell_certificate, Mapping)
        ):
            raise ValueError(
                "Accepted periodic view lacks its current shell certificate."
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_input",
        "periodic graph repartition source",
    )
    if (
        view is None
        or source is None
        or source_domain is None
        or coordinate_contract is None
        or source_candidate_revision_id is None
        or source_face_coverage_id is None
        or source_shell_certificate is None
    ):
        raise RuntimeError(
            "Periodic repartition passed validation without typed source state."
        )
    view.require_owner_directory_current(
        execution_group,
        axis_name=f"{axis_name}_source_directory",
    )
    authored_source = view.source_value
    if not isinstance(authored_source, NativePeriodicSource):
        raise TypeError("Periodic repartition lacks its authored source value.")
    source_entry, source_composition, source_content_id = (
        _owner_local_periodic_source_authority(
            authored_source,
            execution_group,
            f"{axis_name}_source_authority",
        )
    )
    _owner_group_fail_if_any(
        None
        if source_content_id == view.owner_directory_data.source_manifest_id
        else "Current periodic source content differs from its owner directory.",
        execution_group,
        f"{axis_name}_source_content",
        "periodic repartition source content",
    )
    if source_entry is None or source_composition is None:
        raise RuntimeError("Periodic repartition rebuilt no typed source authority.")

    source_candidate_state: dict[str, Any] = {}
    source_mesh_data: dict[str, Any] = {}
    current_source = source
    source_candidate: OwnerLocalPeriodicCandidate | None = None
    local_error = None
    if route == "periodic_delaunay":
        try:
            source_mesh_data = _owner_local_periodic_delaunay_view_mesh_data(
                view, source, source_domain.cell
            )
            source_candidate = _owner_local_periodic_candidate_from_accepted_view(
                view, source, source_mesh_data, source_content_id, rank, len(members)
            )
            source_candidate_state = {
                "points": np.asarray(source_domain.points, dtype=np.float64),
                "point_ids": np.asarray(source_domain.point_ids, dtype=np.int64),
                "point_owner_ranks": np.asarray(source.point_owner_ranks, dtype=np.int32),
                "source_owned": np.asarray(source.point_owner_ranks, dtype=np.int32)
                == rank,
                "candidate_revision_id": source_candidate_revision_id,
                "certified_periodic_margin": view.certified_periodic_margin,
                "periodic_shell_certificate": dict(source_shell_certificate),
            }
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
    else:
        try:
            current_power = _owner_local_periodic_power_candidate(
                source,
                rank,
                execution_group,
                source_content_id,
                axis_name=f"{axis_name}_power_source",
            )
            current_source = current_power["source"]
            current_map = current_power["source_map"]
            source_mesh_data = _owner_local_periodic_power_mesh_data(current_power, rank)
            source_mesh, source_geometry = view.reconstruct_local_geometry()
            generated_mesh = current_power["mesh"]
            generated_geometry = current_power["geometry"]
            current_exact_vertices = (
                ()
                if view.exact_vertex_coordinates is None
                else tuple(tuple(point) for point in view.exact_vertex_coordinates)
            )
            generated_exact_vertices = tuple(
                tuple(point) for point in source_mesh_data["exact_vertex_coordinates"]
            )
            if (
                (None if generated_mesh is None else generated_mesh.mesh_id)
                != view.local_mesh_id
                or (None if source_mesh is None else source_mesh.mesh_id)
                != view.local_mesh_id
                or (
                    None
                    if generated_geometry is None
                    else cell_geometry_id(generated_geometry)
                )
                != view.local_geometry_id
                or (
                    None if source_geometry is None else cell_geometry_id(source_geometry)
                )
                != view.local_geometry_id
                or not np.array_equal(
                    np.asarray(source_mesh_data["cell_ids"], dtype=np.int64),
                    np.asarray(view.cell_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(source_mesh_data["cell_owner_ranks"], dtype=np.int32),
                    np.asarray(view.cell_owner_ranks, dtype=np.int32),
                )
                or not np.array_equal(
                    np.asarray(source_mesh_data["graph_cell_ids"], dtype=np.int64),
                    np.asarray(view.owned_cell_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(
                        source_mesh_data["graph_cell_owner_ranks"], dtype=np.int32
                    ),
                    np.asarray(view.owned_cell_owner_ranks, dtype=np.int32),
                )
                or not np.array_equal(
                    np.asarray(source_mesh_data["graph_cell_weights"]),
                    np.asarray(view.owned_cell_weights),
                )
                or not np.array_equal(
                    np.asarray(source_mesh_data["graph_cell_offsets"], dtype=np.int64),
                    np.asarray(view.cell_neighbor_offsets, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(source_mesh_data["graph_neighbor_ids"], dtype=np.int64),
                    np.asarray(view.cell_neighbor_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(
                        source_mesh_data["graph_neighbor_owner_ranks"], dtype=np.int32
                    ),
                    np.asarray(view.cell_neighbor_owner_ranks, dtype=np.int32),
                )
                or not np.array_equal(
                    np.asarray(
                        source_mesh_data["graph_neighbor_weights"], dtype=np.int64
                    ),
                    np.asarray(view.cell_neighbor_weights, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(source_mesh_data["vertex_ids"], dtype=np.int64),
                    np.asarray(view.vertex_ids, dtype=np.int64),
                )
                or not np.array_equal(
                    np.asarray(
                        source_mesh_data["vertex_coordinates"], dtype=np.float64
                    ).view(np.uint64),
                    np.asarray(view.vertex_coordinates, dtype=np.float64).view(np.uint64),
                )
                or generated_exact_vertices != current_exact_vertices
                or not np.array_equal(
                    np.asarray(
                        tuple(
                            float(value)
                            for value in source_mesh_data["exact_cell_measures"]
                        ),
                        dtype=np.float64,
                    ),
                    np.asarray(view.cell_measures, dtype=np.float64),
                )
                or current_power["candidate"]["candidate_revision_id"]
                != view.candidate_revision_id
            ):
                raise ValueError(
                    "Regenerated periodic power source differs from its accepted view."
                )
            if current_map is None:
                if view.source_map_id is not None or view.vertex_ids.size:
                    raise ValueError(
                        "Nonempty periodic power view lost its exact source map."
                    )
            elif (
                current_map.source_map_id != view.source_map_id
                or current_map.periodic_source_context is None
                or current_map.periodic_source_context.context_id
                != view.source_context_id
            ):
                raise ValueError(
                    "Regenerated exact power map differs from its accepted context."
                )
            source_candidate_state = dict(current_power["candidate"])
            source_candidate_state.update(
                {
                    "points": np.asarray(current_source.domain.points, dtype=np.float64),
                    "point_ids": np.asarray(
                        current_source.domain.point_ids, dtype=np.int64
                    ),
                    "point_owner_ranks": np.asarray(
                        current_source.point_owner_ranks, dtype=np.int32
                    ),
                    "source_owned": np.asarray(
                        current_source.point_owner_ranks, dtype=np.int32
                    )
                    == rank,
                }
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_with_reasons(
        local_error, execution_group, f"{axis_name}_source_replay"
    )

    source_content_id = _owner_local_periodic_source_content_identity(
        current_source,
        rank,
        execution_group,
        f"{axis_name}_source_content",
    )
    target_mesh_data: dict[str, Any] = {}
    target_candidate_state: dict[str, Any] = {}
    target_source = current_source
    target_source_map_id: str | None = None
    target_source_context_id: str | None = None
    target_source_authority: NativePeriodicSource | ExactPowerCellGeometrySource
    target_cell_ids = np.empty((0,), dtype=np.int64)
    if route == "periodic_delaunay":
        candidate = source_candidate
        if candidate is None:
            raise RuntimeError("Periodic Delaunay candidate was not constructed.")
        target_source, target_mesh_data, target_cell_ids = (
            _migrate_owner_local_periodic_delaunay_data(
                current_source,
                candidate,
                source_mesh_data,
                partition_evidence.result,
                rank,
                execution_group,
                f"{axis_name}_delaunay",
            )
        )
        local_error = None
        target_domain: PeriodicPointOrbits | None = None
        try:
            target_domain_candidate = target_source.domain
            if not isinstance(target_domain_candidate, PeriodicPointOrbits):
                raise TypeError("Migrated Delaunay source lost its point-orbit bank.")
            target_domain = target_domain_candidate
            target_candidate_state = dict(source_candidate_state)
            target_candidate_state.update(
                {
                    "points": np.asarray(target_domain.points, dtype=np.float64),
                    "point_ids": np.asarray(target_domain.point_ids, dtype=np.int64),
                    "point_owner_ranks": np.asarray(
                        target_source.point_owner_ranks, dtype=np.int32
                    ),
                    "source_owned": np.asarray(
                        target_source.point_owner_ranks, dtype=np.int32
                    )
                    == rank,
                }
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_delaunay_target",
            "periodic Delaunay target source",
        )
        if target_domain is None:
            raise RuntimeError("Periodic Delaunay target source was not rebuilt.")
        target_source_authority = target_source
        target_source_map_id = None
        target_source_context_id = target_source.binding_id
    else:
        local_error = None
        target_site_ids = np.empty((0,), dtype=np.int64)
        target_active_cell_ids = np.empty((0,), dtype=np.int64)
        try:
            (
                migrated_source,
                target_site_ids,
                target_active_cell_ids,
            ) = _migrate_owner_local_periodic_power_source(
                current_source,
                source_candidate_state["point_ids"],
                source_candidate_state["point_owner_ranks"],
                source_candidate_state["source_owned"],
                partition_evidence.result,
                rank,
                execution_group,
                axis_name=f"{axis_name}_power_sites",
            )
            target_power = _owner_local_periodic_power_candidate(
                migrated_source,
                rank,
                execution_group,
                source_content_id,
                axis_name=f"{axis_name}_power_target",
            )
            target_power_source = target_power["source"]
            if not isinstance(target_power_source, NativePeriodicSource):
                raise TypeError(
                    "Regenerated periodic power source is not NativePeriodicSource."
                )
            target_domain_candidate = target_power_source.domain
            if not isinstance(target_domain_candidate, PeriodicPointOrbits):
                raise TypeError(
                    "Regenerated periodic power source lost its point orbits."
                )
            target_domain = target_domain_candidate
            if target_power_source.point_owner_ranks is None:
                raise ValueError("Regenerated periodic power source lacks owner ranks.")
            target_source = target_power_source
            target_owned_site_ids = np.asarray(target_domain.point_ids, dtype=np.int64)[
                np.asarray(target_power_source.point_owner_ranks, dtype=np.int32) == rank
            ]
            if not np.array_equal(
                np.sort(target_owned_site_ids),
                np.sort(np.asarray(target_site_ids, dtype=np.int64)),
            ):
                raise ValueError(
                    "Periodic power migration omitted or duplicated authored sites."
                )
            target_mesh_data = _owner_local_periodic_power_mesh_data(target_power, rank)
            target_cell_ids = np.asarray(target_mesh_data["cell_ids"], dtype=np.int64)
            if not np.array_equal(
                target_cell_ids,
                np.asarray(target_active_cell_ids, dtype=np.int64),
            ):
                raise ValueError(
                    "Periodic power migration changed active site-cell ownership."
                )
            target_candidate_state = dict(target_power["candidate"])
            target_candidate_state.update(
                {
                    "points": np.asarray(target_domain.points, dtype=np.float64),
                    "point_ids": np.asarray(target_domain.point_ids, dtype=np.int64),
                    "point_owner_ranks": np.asarray(
                        target_power_source.point_owner_ranks, dtype=np.int32
                    ),
                    "source_owned": np.asarray(
                        target_power_source.point_owner_ranks, dtype=np.int32
                    )
                    == rank,
                }
            )
            target_map = target_power["source_map"]
            if target_map is None:
                if (
                    target_cell_ids.size
                    or np.asarray(target_mesh_data["vertex_ids"]).size
                ):
                    raise ValueError(
                        "Nonempty periodic power target lacks its exact source map."
                    )
                target_source_authority = target_source
                target_source_map_id = None
                target_source_context_id = target_source.binding_id
            else:
                if target_map.periodic_source_context is None:
                    raise ValueError(
                        "Periodic power target lost its exact source context."
                    )
                target_source_authority = target_map
                target_source_map_id = target_map.source_map_id
                target_source_context_id = target_map.periodic_source_context.context_id
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_power_migration",
            "periodic power graph migration",
        )

    target_content_id = _owner_local_periodic_source_content_identity(
        target_source,
        rank,
        execution_group,
        f"{axis_name}_target_content",
    )
    _owner_group_fail_if_any(
        None
        if target_content_id == source_content_id
        else "Periodic graph repartition changed authored site coordinates or weights.",
        execution_group,
        f"{axis_name}_source_identity",
        "periodic graph source identity preservation",
    )
    target_mesh_authority = _owner_local_periodic_mesh_authority(
        target_mesh_data,
        target_source,
        target_candidate_state,
        source_entry,
        target_content_id,
        coordinate_contract.spatial_id,
        current_specification.specification_id,
        route,
        target_source_map_id,
        target_source_context_id,
        rank,
        execution_group,
        f"{axis_name}_target_authority",
    )
    target_source = target_mesh_data.get("expanded_source_authority", target_source)
    target_source_context_id = target_source.binding_id
    target_source_authority = target_source
    _owner_group_fail_if_any(
        None
        if target_mesh_authority.topology_id == view.source_topology_id
        and target_mesh_authority.geometry_id == view.source_geometry_id
        and target_mesh_authority.periodic_topology_id == view.periodic_topology_id
        and target_mesh_authority.face_coverage_id == view.periodic_face_coverage_id
        else "Periodic graph repartition changed global topology, geometry, or quotient faces.",
        execution_group,
        f"{axis_name}_global_identity",
        "periodic graph global identity",
    )
    local_error = None
    migration_evidence: OwnerLocalGraphMigrationEvidence | None = None
    try:
        migration_evidence = OwnerLocalGraphMigrationEvidence(
            partition_evidence.result,
            source_candidate_revision_id,
            view.part_id,
            source_face_coverage_id,
            view.distribution_id,
            target_mesh_authority.part_id,
            target_mesh_authority.owner_directory.directory_id,
            execution_group.spec.group_id,
            rank,
            len(members),
            target_cell_ids,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_evidence",
        "periodic graph migration evidence",
    )
    if migration_evidence is None:
        raise RuntimeError("Periodic graph migration evidence was not constructed.")
    return _owner_local_periodic_assembly_from_state(
        target_source,
        current_specification,
        coordinate_contract,
        source_entry,
        source_composition,
        target_source_authority,
        route,
        target_candidate_state,
        target_mesh_data,
        target_mesh_authority,
        target_source_map_id,
        target_source_context_id,
        rank,
        execution_group,
        target_mesh_data.get("source_coordinate_stencil"),
        target_mesh_data.get("exact_vertex_coordinate_offsets"),
        migration_evidence,
        None,
        graph_partition_plan=graph_partition_plan,
        source_adaptive_result=view.adaptive_result,
        axis_name=f"{axis_name}_publication",
    )


def _owner_local_periodic_adaptive_target(
    adaptive_execution: OwnerLocalPeriodicAdaptiveExecution,
    source_coordinate_stencil: VertexInterpolationStencil | None,
    exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None,
    coordinate_root_bank: tuple[tuple[int, tuple[Fraction, ...]], ...],
    source: NativePeriodicSource,
    candidate: dict[str, Any],
    source_entry: CompositionEntry,
    source_content_id: str,
    coordinate_contract_id: str,
    specification_id: str,
    route: str,
    source_map_id: str | None,
    source_context_id: str | None,
    rank: int,
    execution_group: ExecutionGroup,
    periodic_cell: PeriodicCell,
    /,
) -> tuple[
    dict[str, Any],
    _OwnerLocalPeriodicMeshAuthority,
    VertexInterpolationStencil | None,
    tuple[tuple[Fraction, ...], ...] | None,
]:
    """Accept one size-controlled adaptation and bind its exact target authority."""
    from .._distribution import _owner_group_fail_if_any

    local_error = adaptive_execution.refusal
    if local_error is None and not adaptive_execution.size_evidence.compliant:
        local_error = "Periodic Delaunay target does not satisfy its exact size controls."
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        "periodic_owner_adaptation_acceptance",
        "periodic exact size compliance",
    )
    target_error = None
    target_mesh_data: dict[str, Any]
    try:
        target_mesh_data = _owner_local_periodic_adaptive_mesh_data(
            adaptive_execution,
            periodic_cell,
            rank,
        )
    except Exception as error:
        target_error = f"{type(error).__name__}: {error}"
        target_mesh_data = {}
    _owner_group_fail_if_any(
        target_error,
        execution_group,
        "periodic_owner_adaptation_target",
        "periodic target local mesh",
    )
    ancestry_error = None
    if adaptive_execution.part_transition is None:
        if np.asarray(target_mesh_data["vertex_ids"]).size:
            ancestry_error = "Periodic target vertices lack a committed exact transition."
        source_coordinate_stencil = None
        exact_vertex_coordinate_offsets = None
    else:
        try:
            (
                source_coordinate_stencil,
                exact_vertex_coordinate_offsets,
            ) = _compose_owner_local_periodic_coordinate_ancestry(
                source_coordinate_stencil,
                ()
                if exact_vertex_coordinate_offsets is None
                else exact_vertex_coordinate_offsets,
                adaptive_execution.part_transition,
            )
            expected_coordinates = tuple(
                tuple(point) for point in target_mesh_data["exact_vertex_coordinates"]
            )
            replayed_coordinates = (
                ()
                if source_coordinate_stencil is None
                else _evaluate_owner_local_periodic_coordinate_ancestry(
                    source_coordinate_stencil,
                    exact_vertex_coordinate_offsets,
                    coordinate_root_bank,
                )
            )
            if replayed_coordinates != expected_coordinates:
                raise ValueError(
                    "Exact periodic midpoint ancestry differs from native target coordinates."
                )
            if source_coordinate_stencil is None:
                exact_vertex_coordinate_offsets = None
        except Exception as error:
            ancestry_error = f"{type(error).__name__}: {error}"
            source_coordinate_stencil = None
            exact_vertex_coordinate_offsets = None
    _owner_group_fail_if_any(
        ancestry_error,
        execution_group,
        "periodic_owner_adaptation_ancestry",
        "periodic exact midpoint ancestry",
    )
    target_mesh_data["source_coordinate_stencil"] = source_coordinate_stencil
    target_mesh_data["exact_vertex_coordinate_offsets"] = exact_vertex_coordinate_offsets
    mesh_authority = _owner_local_periodic_mesh_authority(
        target_mesh_data,
        source,
        candidate,
        source_entry,
        source_content_id,
        coordinate_contract_id,
        specification_id,
        route,
        source_map_id,
        source_context_id,
        rank,
        execution_group,
        "periodic_owner_adaptation_target",
    )
    # Neighbor-halo closure rebinds the cumulative ancestry to the expanded
    # resident vertex bank.
    return (
        target_mesh_data,
        mesh_authority,
        target_mesh_data["source_coordinate_stencil"],
        target_mesh_data["exact_vertex_coordinate_offsets"],
    )


def _owner_local_periodic_assembly_from_state(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    coordinate_contract: SpatialCoordinateContract,
    source_entry: CompositionEntry,
    source_composition: Composition,
    source_authority_value: NativePeriodicSource | ExactPowerCellGeometrySource,
    owner_local_route: str,
    candidate: dict[str, Any],
    mesh_data: dict[str, Any],
    mesh_authority: _OwnerLocalPeriodicMeshAuthority,
    source_map_id: str | None,
    source_context_id: str | None,
    rank: int,
    execution_group: ExecutionGroup,
    source_coordinate_stencil: VertexInterpolationStencil | None,
    exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None,
    graph_migration_evidence: OwnerLocalGraphMigrationEvidence | None,
    adaptive_execution: OwnerLocalPeriodicAdaptiveExecution | None,
    /,
    *,
    graph_partition_plan: GraphPartitionPlan | None = None,
    source_adaptive_result: OwnerLocalPeriodicAdaptiveResult | None = None,
    axis_name: str,
) -> MeshAssembly:
    from ..._execution_runtime import ExecutionGroup
    from ...discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ...discretization._view_support import OwnerLocalCellGeometry
    from ...graph._partition import GraphPartitionPlan
    from ...lifecycle import Composition, CompositionEntry
    from ...meshing._assembly import MeshAssembly, MeshPart, MeshPartManifest
    from .._device_adaptation import (
        OwnerLocalPeriodicAdaptiveExecution,
        OwnerLocalPeriodicAdaptiveResult,
    )
    from .._distribution import _owner_group_fail_if_any

    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
        or not isinstance(coordinate_contract, SpatialCoordinateContract)
        or not isinstance(source_entry, CompositionEntry)
        or not isinstance(source_composition, Composition)
        or not isinstance(
            source_authority_value,
            (NativePeriodicSource, ExactPowerCellGeometrySource),
        )
        or owner_local_route not in ("periodic_delaunay", "periodic_power")
        or not isinstance(mesh_authority, _OwnerLocalPeriodicMeshAuthority)
        or not isinstance(execution_group, ExecutionGroup)
        or not execution_group.is_member
        or not isinstance(axis_name, str)
        or not axis_name.strip()
        or adaptive_execution is not None
        and not isinstance(adaptive_execution, OwnerLocalPeriodicAdaptiveExecution)
        or source_adaptive_result is not None
        and not isinstance(source_adaptive_result, OwnerLocalPeriodicAdaptiveResult)
        or graph_partition_plan is not None
        and not isinstance(graph_partition_plan, GraphPartitionPlan)
    ):
        raise TypeError("Owner-local periodic publication requires typed state.")
    from .._periodic import PeriodicPointOrbits

    source_domain_value = source.domain
    if not isinstance(source_domain_value, PeriodicPointOrbits):
        raise TypeError(
            "Owner-local periodic publication requires periodic point orbits."
        )
    source_domain = source_domain_value
    if (
        source_adaptive_result is not None
        and graph_partition_plan is None
        or adaptive_execution is not None
        and source_adaptive_result is not None
    ):
        raise ValueError("Periodic adaptive publication has inconsistent rebind state.")
    shell_certificate = dict(candidate["periodic_shell_certificate"])
    shell_certificate["exact_fundamental_volume"] = (
        mesh_authority.exact_volume.numerator,
        mesh_authority.exact_volume.denominator,
    )
    shell_certificate["expected_fundamental_volume"] = (
        mesh_authority.expected_volume.numerator,
        mesh_authority.expected_volume.denominator,
    )
    shell_certificate["quotient_face_coverage_id"] = mesh_authority.face_coverage_id
    candidate["periodic_shell_certificate"] = shell_certificate
    successor_authority = _owner_local_periodic_successor_authority(
        mesh_authority,
        source_authority_value,
        source_entry,
        source_composition,
        owner_local_route,
        execution_group,
        axis_name=f"{axis_name}_successor",
    )
    part_authority = successor_authority.part_authority
    bound_witness = successor_authority.bound_witness
    accepted_composition = successor_authority.accepted_composition
    dimension = source.ambient_dimension
    part_name = source.source_id
    entity_set_ids = {
        degree: canonical_fingerprint(
            {
                "kind": "periodic-owner-local-entity-set",
                "topology": mesh_authority.topology_id,
                "dimension": degree,
            }
        )
        for degree in range(dimension + 1)
    }
    record: dict[str, Any] = {
        "part_name": part_name,
        "part_id": mesh_authority.part_id,
        "distribution_id": bound_witness.directory_id,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_mesh_id": source_entry.structure_id,
        "source_topology_id": mesh_authority.topology_id,
        "source_geometry_id": mesh_authority.geometry_id,
        "source_map_id": source_map_id,
        "geometry_layout_id": mesh_data["geometry_layout_id"],
        "coordinate_contract_id": coordinate_contract.spatial_id,
        "specification_id": specification.specification_id,
        "coordinate_contract": coordinate_contract,
        "topological_dimension": dimension,
        "cell_rows": mesh_data["cell_rows"],
        "cell_ids": mesh_data["cell_ids"],
        "owned": mesh_data["owned"],
        "cell_measures": mesh_data["cell_measures"],
        "cell_owner_ranks": mesh_data["cell_owner_ranks"],
        "owned_cell_ids": mesh_data["graph_cell_ids"],
        "owned_cell_owner_ranks": mesh_data["graph_cell_owner_ranks"],
        "owned_cell_weights": mesh_data["graph_cell_weights"],
        "cell_neighbor_offsets": mesh_data["graph_cell_offsets"],
        "cell_neighbor_ids": mesh_data["graph_neighbor_ids"],
        "cell_neighbor_owner_ranks": mesh_data["graph_neighbor_owner_ranks"],
        "cell_neighbor_weights": mesh_data["graph_neighbor_weights"],
        "support_cell_mask": mesh_data["support_cell_mask"],
        "vertex_rows": mesh_data["vertex_rows"],
        "vertex_ids": mesh_data["vertex_ids"],
        "vertex_coordinates": mesh_data["vertex_coordinates"],
        "exact_vertex_coordinates": (
            None
            if not np.asarray(mesh_data["vertex_ids"]).size
            else mesh_data["exact_vertex_coordinates"]
        ),
        "coordinate_rows": mesh_data["coordinate_rows"],
        "coordinates": mesh_data["coordinates"],
        "exact_coordinates": mesh_data["exact_coordinates"],
        "source_coordinate_stencil": source_coordinate_stencil,
        "exact_vertex_coordinate_offsets": exact_vertex_coordinate_offsets,
        "entity_set_ids": entity_set_ids,
        "local_entity_ids": mesh_data["local_entity_ids"],
        "entity_vertex_offsets": mesh_data["entity_vertex_offsets"],
        "entity_vertex_values": mesh_data["entity_vertex_values"],
        "cell_key_offsets": mesh_data["cell_key_offsets"],
        "cell_key_source_ids": mesh_data["cell_key_source_ids"],
        "cell_key_relative_windings": mesh_data["cell_key_relative_windings"],
        "vertex_source_ids": mesh_data["vertex_source_ids"],
        "source_point_ids": candidate["point_ids"],
        "source_owner_ranks": candidate["point_owner_ranks"],
        "source_owned": candidate["source_owned"],
        "vertex_owner_ranks": mesh_data["vertex_owner_ranks"],
        "vertex_absolute_windings": mesh_data["vertex_absolute_windings"],
        "candidate_revision_id": candidate["candidate_revision_id"],
        "certified_periodic_margin": candidate["certified_periodic_margin"],
        "periodic_shell_certificate": candidate["periodic_shell_certificate"],
        "periodic_topology_id": mesh_authority.periodic_topology_id,
        "periodic_cell_ids": mesh_data["periodic_cell_ids"],
        "vertex_representative_ids": mesh_data["vertex_representative_ids"],
        "vertex_windings": mesh_data["vertex_windings"],
        "cell_windings": mesh_data["cell_windings"],
        "source_context_id": source_context_id,
        "periodic_lift_id": source_domain.cell.cell_id,
        "periodic_lift_matrix": np.asarray(source_domain.cell.vectors, dtype=np.float64),
        "coordinates_are_lifted": True,
        "support_fractional_lower": mesh_data["support_fractional_lower"],
        "support_fractional_upper": mesh_data["support_fractional_upper"],
        "blocks": mesh_data["blocks"],
        "rank": rank,
        "serial_source_view": False,
        "part_authority": part_authority,
        "accepted_composition": accepted_composition,
        "owner_directory_witness": bound_witness,
        "graph_migration_evidence": (
            None
            if adaptive_execution is not None or source_adaptive_result is not None
            else graph_migration_evidence
        ),
    }
    local_error = None
    local_assembly: MeshAssembly | None = None
    try:
        local_geometry = OwnerLocalCellGeometry.from_owner_local_result(record)
        if adaptive_execution is not None:
            adaptive_result = OwnerLocalPeriodicAdaptiveResult.from_execution(
                adaptive_execution, local_geometry
            )
            record["adaptive_result"] = adaptive_result
            record["graph_migration_evidence"] = graph_migration_evidence
            local_geometry = OwnerLocalCellGeometry.from_owner_local_result(record)
        elif source_adaptive_result is not None:
            if graph_partition_plan is None:
                raise RuntimeError(
                    "Adaptive ownership migration requires its accepted graph plan."
                )
            rebound_result = source_adaptive_result.rebind_after_ownership_migration(
                local_geometry,
                graph_partition_plan,
                execution_group,
                axis_name=f"{axis_name}_adaptive_rebind",
            )
            record["adaptive_result"] = rebound_result
            record["graph_migration_evidence"] = graph_migration_evidence
            local_geometry = OwnerLocalCellGeometry.from_owner_local_result(record)
        part = MeshPart(
            part_name, local_geometry, coordinate_contract=coordinate_contract
        )
        manifest = MeshPartManifest(
            part_name,
            mesh_authority.part_id,
            coordinate_contract,
            dimension,
            dimension,
            source_topology_id=mesh_authority.topology_id,
            source_geometry_id=mesh_authority.geometry_id,
            distribution_id=bound_witness.directory_id,
        )
        local_assembly = MeshAssembly((part,), global_parts=(manifest,), local_rank=rank)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_publication",
        "owner-local periodic assembly",
    )
    if local_assembly is None:
        raise RuntimeError(
            "Owner-local periodic assembly returned without collective acceptance."
        )
    return local_assembly


@final
class PreparedNativePeriodic(StrictModule, NonTrainableState):
    source: NativePeriodicSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativePeriodicSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        if not isinstance(source, NativePeriodicSource):
            raise TypeError("source must be NativePeriodicSource.")
        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError(
                "specification must be SurfaceMeshingSpec or VolumeMeshingSpec."
            )
        if not isinstance(options, NativeMeshingOptions):
            raise TypeError("options must be NativeMeshingOptions.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-periodic",
                "source_binding": source.binding_id,
                "specification": specification.specification_id,
                "route": options.route,
                "coordinate_contract": coordinate_contract.spatial_id,
            }
        )

    def _execute_owner_local(
        self,
        execution_group: ExecutionGroup,
        /,
        *,
        graph_partition_plan: GraphPartitionPlan | None = None,
    ) -> MeshAssembly:
        import jax

        from ..._execution_runtime import ExecutionGroup
        from ...graph._partition import GraphPartitionPlan
        from ...meshing._distribution import (
            _owner_group_allgather,
            _owner_group_fail_if_any,
        )
        from .._periodic import PeriodicPointOrbits

        if (
            not isinstance(execution_group, ExecutionGroup)
            or self.source.point_owner_ranks is None
            or not isinstance(self.source.domain, PeriodicPointOrbits)
        ):
            raise TypeError(
                "Owner-local periodic execution requires an ExecutionGroup and typed point ownership."
            )
        if graph_partition_plan is not None and not isinstance(
            graph_partition_plan, GraphPartitionPlan
        ):
            raise TypeError("graph_partition_plan must be GraphPartitionPlan or None.")
        members = execution_group.spec.process_indices
        devices = execution_group.devices
        if (
            not execution_group.is_member
            or tuple(device.process_index for device in devices) != members
            or jax.process_index() not in members
        ):
            raise ValueError(
                "Owner-local execution does not match its registered process group."
            )
        rank = members.index(jax.process_index())
        domain = self.source.domain
        owner_local_route = self.options.route
        if owner_local_route not in ("periodic_delaunay", "periodic_power"):
            raise ValueError("Owner-local periodic execution requires a periodic route.")
        dimension = domain.ambient_dimension
        common_request = canonical_fingerprint(
            {
                "source_id": self.source.source_id,
                "source_revision": self.source.source_revision,
                "periodic_cell": domain.cell.cell_id,
                "specification": self.specification.specification_id,
                "route": owner_local_route,
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "graph_partition_plan": (
                    None if graph_partition_plan is None else graph_partition_plan.plan_id
                ),
            }
        )
        request_records = _owner_group_allgather(
            np.frombuffer(bytes.fromhex(common_request), dtype=np.uint8),
            execution_group,
            "periodic_owner_request",
        )
        if any(
            np.asarray(record, dtype=np.uint8).tobytes().hex() != common_request
            for record in request_records
        ):
            raise ValueError("Owner-local periodic ranks supplied different requests.")
        if graph_partition_plan is not None and (
            graph_partition_plan.part_count != len(members)
        ):
            _owner_group_fail_if_any(
                "GraphPartitionPlan part count differs from the owner group.",
                execution_group,
                "periodic_owner_request",
                "periodic graph partition plan",
            )
        source_entry, source_composition, source_content_id = (
            _owner_local_periodic_source_authority(
                self.source, execution_group, "periodic_owner_source"
            )
        )
        source_coordinate_stencil: VertexInterpolationStencil | None = None
        exact_vertex_coordinate_offsets: tuple[tuple[Fraction, ...], ...] | None = None
        coordinate_root_bank: tuple[tuple[int, tuple[Fraction, ...]], ...] = ()
        if owner_local_route == "periodic_delaunay":
            local_nonzero_weight = bool(
                self.source.site_weights is not None
                and np.any(np.asarray(self.source.site_weights, dtype=np.float64) != 0.0)
            )
            weight_flags = _owner_group_allgather(
                np.asarray([local_nonzero_weight], dtype=np.bool_),
                execution_group,
                "periodic_owner_weights",
            )
            if np.any(np.asarray(weight_flags, dtype=np.bool_)):
                raise ValueError(
                    "Periodic Delaunay is unweighted; nonzero site weights require periodic_power."
                )
            candidate = _owner_local_periodic_delaunay_candidate(
                self.source,
                self.source.point_owner_ranks,
                np.asarray(self.source.point_owner_ranks, dtype=np.int32) == rank,
                rank,
                execution_group,
                source_content_id,
                initial_margin=1.0,
                axis_name="periodic_owner_closure",
            )
            controls = tuple(self.specification.size_controls)
            top_scope = (
                self.specification.scope
                if isinstance(self.specification, SurfaceMeshingSpec)
                else self.specification.boundary_scope
            )
            local_error = None
            for control in controls:
                if not isinstance(control, UniformSizeControl):
                    local_error = "Owner-local periodic controls must be uniform sizes."
                    continue
                if (
                    control.scope.source_id != self.source.source_id
                    or control.scope.source_revision != self.source.source_revision
                    or control.scope.scope_id != top_scope.scope_id
                ):
                    local_error = (
                        "Owner-local periodic size controls must bind the complete "
                        "authored fundamental-domain scope."
                    )
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                "periodic_owner_controls",
                "periodic size-control identity",
            )
            from ._native_periodic_power import (
                _owner_local_periodic_site_weights_for_support,
            )

            final_domain = PeriodicPointOrbits(
                domain.cell, candidate["points"], candidate["point_ids"]
            )
            final_site_weights = _owner_local_periodic_site_weights_for_support(
                self.source,
                candidate["point_ids"],
                candidate["point_owner_ranks"],
                rank,
                execution_group,
                axis_name="periodic_owner_source_weights",
            )
            final_source = NativePeriodicSource(
                final_domain,
                self.source.source_id,
                self.source.source_revision,
                site_weights=final_site_weights,
                point_owner_ranks=candidate["point_owner_ranks"],
            )
            source_authority_value: (
                NativePeriodicSource | ExactPowerCellGeometrySource
            ) = final_source
            source_map_id = None
            source_context_id = final_source.binding_id
            mesh_data = _owner_local_periodic_delaunay_mesh_data(
                final_source, candidate, rank
            )
            source_coordinate_stencil, initial_offsets = (
                _initial_owner_local_periodic_coordinate_ancestry(
                    final_source, mesh_data, domain.cell
                )
            )
            exact_vertex_coordinate_offsets = (
                initial_offsets if source_coordinate_stencil is not None else None
            )
            mesh_data["source_coordinate_stencil"] = source_coordinate_stencil
            mesh_data["exact_vertex_coordinate_offsets"] = exact_vertex_coordinate_offsets
            point_ids = np.asarray(final_domain.point_ids, dtype=np.int64)
            point_coordinates = np.asarray(final_domain.points, dtype=np.float64)
            point_order = np.argsort(point_ids, kind="stable")
            coordinate_root_bank = tuple(
                (
                    int(point_ids[row]),
                    tuple(
                        Fraction.from_float(float(value))
                        for value in point_coordinates[row]
                    ),
                )
                for row in point_order
            )
        else:
            from ._native_periodic_power import (
                _owner_local_periodic_power_candidate,
                _owner_local_periodic_power_mesh_data,
            )

            power = _owner_local_periodic_power_candidate(
                self.source,
                rank,
                execution_group,
                source_content_id,
                axis_name="periodic_owner_power",
            )
            candidate = power["candidate"]
            source_map = power["source_map"]
            if source_map is None:
                source_authority_value = power["source"]
                source_map_id = None
                source_context_id = source_authority_value.binding_id
            else:
                source_authority_value = source_map
                source_map_id = source_map.source_map_id
                context = source_map.periodic_source_context
                if context is None:
                    raise ValueError(
                        "Owner-local periodic power lost its exact source context."
                    )
                source_context_id = context.context_id
            mesh_data = _owner_local_periodic_power_mesh_data(power, rank)
            # Power cells are the authored sites' Laguerre cells; they carry no
            # Delaunay coordinate ancestry.
            source_coordinate_stencil = None
            exact_vertex_coordinate_offsets = None
        cell_ids = np.asarray(mesh_data["cell_ids"], dtype=np.int64)
        cell_owner_ranks = np.asarray(mesh_data["cell_owner_ranks"], dtype=np.int32)
        if owner_local_route == "periodic_delaunay":
            owned_source_ids = np.asarray(candidate["point_ids"], dtype=np.int64)[
                np.asarray(candidate["source_owned"], dtype=np.bool_)
            ]
            owned_vertex_sources = np.asarray(
                mesh_data["vertex_source_ids"], dtype=np.int64
            )[np.asarray(mesh_data["vertex_owner_ranks"], dtype=np.int32) == rank]
            local_error = (
                None
                if np.setdiff1d(owned_source_ids, owned_vertex_sources).size == 0
                else "An owned periodic Delaunay site has no local quotient vertex."
            )
        else:
            owned_source_ids = np.asarray(candidate["point_ids"], dtype=np.int64)[
                np.asarray(candidate["source_owned"], dtype=np.bool_)
            ]
            owned_cells = cell_ids[cell_owner_ranks == rank]
            local_error = (
                None
                if np.unique(owned_cells).size == owned_cells.size
                and np.setdiff1d(owned_cells, owned_source_ids).size == 0
                else "An owner-local power cell lacks one unique authored site."
            )
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            "periodic_owner_acceptance",
            "periodic source coverage",
        )
        mesh_authority = _owner_local_periodic_mesh_authority(
            mesh_data,
            self.source,
            candidate,
            source_entry,
            source_content_id,
            self.coordinate_contract.spatial_id,
            self.specification.specification_id,
            owner_local_route,
            source_map_id,
            source_context_id,
            rank,
            execution_group,
            "periodic_owner_acceptance",
        )
        if owner_local_route == "periodic_delaunay":
            # The Delaunay neighbor halo may have grown the resident source bank.
            final_source = mesh_data.get("expanded_source_authority", final_source)
            source_authority_value = final_source
            source_context_id = final_source.binding_id
            final_domain = final_source.domain
            source_coordinate_stencil = mesh_data.get(
                "source_coordinate_stencil", source_coordinate_stencil
            )
            exact_vertex_coordinate_offsets = mesh_data.get(
                "exact_vertex_coordinate_offsets", exact_vertex_coordinate_offsets
            )
            point_ids = np.asarray(final_domain.point_ids, dtype=np.int64)
            point_coordinates = np.asarray(final_domain.points, dtype=np.float64)
            coordinate_root_bank = tuple(
                (
                    int(point_ids[row]),
                    tuple(
                        Fraction.from_float(float(value))
                        for value in point_coordinates[row]
                    ),
                )
                for row in np.argsort(point_ids, kind="stable")
            )
        face_coverage_id = mesh_authority.face_coverage_id
        owner_directory = mesh_authority.owner_directory
        shell_certificate = dict(candidate["periodic_shell_certificate"])
        shell_certificate["exact_fundamental_volume"] = (
            mesh_authority.exact_volume.numerator,
            mesh_authority.exact_volume.denominator,
        )
        shell_certificate["expected_fundamental_volume"] = (
            mesh_authority.expected_volume.numerator,
            mesh_authority.expected_volume.denominator,
        )
        shell_certificate["quotient_face_coverage_id"] = face_coverage_id
        candidate["periodic_shell_certificate"] = shell_certificate
        adaptive_execution: OwnerLocalPeriodicAdaptiveExecution | None = None
        graph_migration_evidence: OwnerLocalGraphMigrationEvidence | None = None
        if owner_local_route == "periodic_delaunay":
            local_periodic_candidate = _owner_local_periodic_candidate_from_mesh_data(
                final_source,
                candidate["candidate_revision_id"],
                mesh_data,
                mesh_authority,
                source_entry,
                source_content_id,
                self.coordinate_contract.spatial_id,
                rank,
                len(members),
                dimension,
                domain.cell,
            )
            if graph_partition_plan is not None:
                from .._distribution import (
                    _partition_owner_local_periodic_candidate,
                    OwnerLocalGraphMigrationEvidence,
                )

                source_candidate_id = local_periodic_candidate.candidate_id
                source_mesh_authority = mesh_authority
                graph_result = _partition_owner_local_periodic_candidate(
                    local_periodic_candidate,
                    graph_partition_plan,
                    execution_group,
                    axis_name="periodic_owner_graph_partition",
                )
                final_source, mesh_data, target_cell_ids = (
                    _migrate_owner_local_periodic_delaunay_data(
                        final_source,
                        local_periodic_candidate,
                        mesh_data,
                        graph_result,
                        rank,
                        execution_group,
                        "periodic_owner_graph_migration",
                    )
                )
                if not isinstance(final_source.domain, PeriodicPointOrbits):
                    raise TypeError(
                        "Periodic graph migration lost its point-orbit source."
                    )
                final_domain = final_source.domain
                candidate = dict(candidate)
                candidate["points"] = np.asarray(final_domain.points, dtype=np.float64)
                candidate["point_ids"] = np.asarray(
                    final_domain.point_ids, dtype=np.int64
                )
                candidate["point_owner_ranks"] = np.asarray(
                    final_source.point_owner_ranks, dtype=np.int32
                )
                candidate["source_owned"] = candidate["point_owner_ranks"] == rank
                source_authority_value = final_source
                source_context_id = final_source.binding_id
                # The compact view keeps the authored source as its source value;
                # migration lives only in the actual geometry authority. The
                # migrated bank must still carry the same authored content.
                _, _, renewed_source_content_id = _owner_local_periodic_source_authority(
                    final_source,
                    execution_group,
                    "periodic_owner_graph_source",
                )
                if renewed_source_content_id != source_content_id:
                    raise ValueError(
                        "Graph repartition changed authored periodic source identity."
                    )
                source_coordinate_stencil = mesh_data["source_coordinate_stencil"]
                exact_vertex_coordinate_offsets = mesh_data[
                    "exact_vertex_coordinate_offsets"
                ]
                point_ids = np.asarray(final_domain.point_ids, dtype=np.int64)
                point_coordinates = np.asarray(final_domain.points, dtype=np.float64)
                coordinate_root_bank = tuple(
                    (
                        int(point_ids[row]),
                        tuple(
                            Fraction.from_float(float(value))
                            for value in point_coordinates[row]
                        ),
                    )
                    for row in np.argsort(point_ids, kind="stable")
                )
                migrated_authority = _owner_local_periodic_mesh_authority(
                    mesh_data,
                    self.source,
                    candidate,
                    source_entry,
                    source_content_id,
                    self.coordinate_contract.spatial_id,
                    self.specification.specification_id,
                    owner_local_route,
                    source_map_id,
                    source_context_id,
                    rank,
                    execution_group,
                    "periodic_owner_graph_migration_authority",
                )
                final_source = mesh_data.get("expanded_source_authority", final_source)
                source_authority_value = final_source
                source_context_id = final_source.binding_id
                final_domain = final_source.domain
                source_coordinate_stencil = mesh_data.get(
                    "source_coordinate_stencil", source_coordinate_stencil
                )
                exact_vertex_coordinate_offsets = mesh_data.get(
                    "exact_vertex_coordinate_offsets", exact_vertex_coordinate_offsets
                )
                local_error = (
                    None
                    if migrated_authority.topology_id == source_mesh_authority.topology_id
                    and migrated_authority.geometry_id
                    == source_mesh_authority.geometry_id
                    and migrated_authority.face_coverage_id
                    == source_mesh_authority.face_coverage_id
                    else "Graph migration changed periodic topology, exact geometry, or facet coverage."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    "periodic_owner_graph_migration_identity",
                    "periodic graph migration identity",
                )
                mesh_authority = migrated_authority
                face_coverage_id = mesh_authority.face_coverage_id
                owner_directory = mesh_authority.owner_directory
                graph_migration_evidence = OwnerLocalGraphMigrationEvidence(
                    graph_result,
                    source_candidate_id,
                    source_mesh_authority.part_id,
                    source_mesh_authority.face_coverage_id,
                    source_mesh_authority.owner_directory.directory_id,
                    mesh_authority.part_id,
                    owner_directory.directory_id,
                    execution_group.spec.group_id,
                    rank,
                    len(members),
                    target_cell_ids,
                )
                local_periodic_candidate = _owner_local_periodic_candidate_from_mesh_data(
                    final_source,
                    candidate["candidate_revision_id"],
                    mesh_data,
                    mesh_authority,
                    source_entry,
                    source_content_id,
                    self.coordinate_contract.spatial_id,
                    rank,
                    len(members),
                    dimension,
                    domain.cell,
                )
            shell_certificate = dict(candidate["periodic_shell_certificate"])
            shell_certificate["owner_local_candidate_id"] = (
                local_periodic_candidate.candidate_id
            )
            candidate["periodic_shell_certificate"] = shell_certificate
            # A base specification without size controls publishes the exact
            # owner-local Delaunay candidate; size controls request adaptation.
            if controls:
                from .._device_adaptation import _adapt_owner_local_periodic

                adaptive_execution = _adapt_owner_local_periodic(
                    local_periodic_candidate,
                    self.specification,
                    execution_group,
                    rank,
                    axis_name="periodic_owner_adaptation",
                    graph_partition_plan=graph_partition_plan,
                )
                (
                    mesh_data,
                    mesh_authority,
                    source_coordinate_stencil,
                    exact_vertex_coordinate_offsets,
                ) = _owner_local_periodic_adaptive_target(
                    adaptive_execution,
                    source_coordinate_stencil,
                    exact_vertex_coordinate_offsets,
                    coordinate_root_bank,
                    self.source,
                    candidate,
                    source_entry,
                    source_content_id,
                    self.coordinate_contract.spatial_id,
                    self.specification.specification_id,
                    owner_local_route,
                    source_map_id,
                    source_context_id,
                    rank,
                    execution_group,
                    domain.cell,
                )
                final_source = mesh_data.get("expanded_source_authority", final_source)
                source_authority_value = final_source
                source_context_id = final_source.binding_id
                face_coverage_id = mesh_authority.face_coverage_id
                owner_directory = mesh_authority.owner_directory
                shell_certificate = dict(candidate["periodic_shell_certificate"])
                shell_certificate["exact_fundamental_volume"] = (
                    mesh_authority.exact_volume.numerator,
                    mesh_authority.exact_volume.denominator,
                )
                shell_certificate["expected_fundamental_volume"] = (
                    mesh_authority.expected_volume.numerator,
                    mesh_authority.expected_volume.denominator,
                )
                shell_certificate["quotient_face_coverage_id"] = face_coverage_id
                shell_certificate["size_evidence_id"] = (
                    adaptive_execution.size_evidence.evidence_id
                )
                candidate["periodic_shell_certificate"] = shell_certificate
        else:
            if graph_partition_plan is not None:
                from .._distribution import OwnerLocalGraphMigrationEvidence

                source_mesh_authority = mesh_authority
                source_candidate_id = str(candidate["candidate_revision_id"])
                graph_result = _partition_owner_local_periodic_power_cells(
                    mesh_data,
                    graph_partition_plan,
                    execution_group,
                    axis_name="periodic_owner_power_graph_partition",
                )
                migrated_source, target_site_ids, target_active_cell_ids = (
                    _migrate_owner_local_periodic_power_source(
                        power["source"],
                        candidate["point_ids"],
                        candidate["point_owner_ranks"],
                        candidate["source_owned"],
                        graph_result,
                        rank,
                        execution_group,
                        axis_name="periodic_owner_power_source_migration",
                    )
                )
                power = _owner_local_periodic_power_candidate(
                    migrated_source,
                    rank,
                    execution_group,
                    source_content_id,
                    axis_name="periodic_owner_power_graph_regeneration",
                )
                candidate = power["candidate"]
                final_power_source = power["source"]
                regenerated_owned_ids = np.asarray(
                    final_power_source.domain.point_ids, dtype=np.int64
                )[
                    np.asarray(final_power_source.point_owner_ranks, dtype=np.int32)
                    == rank
                ]
                local_error = (
                    None
                    if np.array_equal(
                        np.sort(regenerated_owned_ids),
                        np.sort(target_site_ids),
                    )
                    else "Rebuilt periodic power source omitted or duplicated an authored site."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    "periodic_owner_power_source_rows",
                    "periodic power source row coverage",
                )
                renewed_source_entry, renewed_source_composition, renewed_content_id = (
                    _owner_local_periodic_source_authority(
                        final_power_source,
                        execution_group,
                        "periodic_owner_power_source_rebind",
                    )
                )
                local_error = (
                    None
                    if renewed_content_id == source_content_id
                    else "Periodic power ownership migration changed authored source identity."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    "periodic_owner_power_source_rebind",
                    "periodic power source identity",
                )
                source_entry = renewed_source_entry
                source_composition = renewed_source_composition
                source_content_id = renewed_content_id
                source_map = power["source_map"]
                if source_map is None:
                    source_authority_value = final_power_source
                    source_map_id = None
                    source_context_id = final_power_source.binding_id
                else:
                    source_authority_value = source_map
                    source_map_id = source_map.source_map_id
                    context = source_map.periodic_source_context
                    if context is None:
                        raise ValueError(
                            "Repartitioned periodic power lost its exact source context."
                        )
                    source_context_id = context.context_id
                mesh_data = _owner_local_periodic_power_mesh_data(power, rank)
                local_error = (
                    None
                    if np.array_equal(
                        np.sort(np.asarray(mesh_data["cell_ids"], dtype=np.int64)),
                        np.sort(np.asarray(target_active_cell_ids, dtype=np.int64)),
                    )
                    else "Rebuilt periodic power cells differ from graph target ownership."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    "periodic_owner_power_graph_cells",
                    "periodic power graph cell ownership",
                )
                migrated_authority = _owner_local_periodic_mesh_authority(
                    mesh_data,
                    self.source,
                    candidate,
                    source_entry,
                    source_content_id,
                    self.coordinate_contract.spatial_id,
                    self.specification.specification_id,
                    owner_local_route,
                    source_map_id,
                    source_context_id,
                    rank,
                    execution_group,
                    "periodic_owner_power_graph_authority",
                )
                local_error = (
                    None
                    if migrated_authority.topology_id == source_mesh_authority.topology_id
                    and migrated_authority.geometry_id
                    == source_mesh_authority.geometry_id
                    and migrated_authority.periodic_topology_id
                    == source_mesh_authority.periodic_topology_id
                    and migrated_authority.face_coverage_id
                    == source_mesh_authority.face_coverage_id
                    else "Periodic power graph repartition changed quotient geometry or coverage."
                )
                _owner_group_fail_if_any(
                    local_error,
                    execution_group,
                    "periodic_owner_power_graph_identity",
                    "periodic power graph identity",
                )
                mesh_authority = migrated_authority
                face_coverage_id = mesh_authority.face_coverage_id
                owner_directory = mesh_authority.owner_directory
                graph_migration_evidence = OwnerLocalGraphMigrationEvidence(
                    graph_result,
                    source_candidate_id,
                    source_mesh_authority.part_id,
                    source_mesh_authority.face_coverage_id,
                    source_mesh_authority.owner_directory.directory_id,
                    mesh_authority.part_id,
                    owner_directory.directory_id,
                    execution_group.spec.group_id,
                    rank,
                    len(members),
                    mesh_data["graph_cell_ids"],
                )
                shell_certificate = dict(candidate["periodic_shell_certificate"])
                shell_certificate["exact_fundamental_volume"] = (
                    mesh_authority.exact_volume.numerator,
                    mesh_authority.exact_volume.denominator,
                )
                shell_certificate["expected_fundamental_volume"] = (
                    mesh_authority.expected_volume.numerator,
                    mesh_authority.expected_volume.denominator,
                )
                shell_certificate["quotient_face_coverage_id"] = face_coverage_id
                candidate["periodic_shell_certificate"] = shell_certificate
        return _owner_local_periodic_assembly_from_state(
            self.source,
            self.specification,
            self.coordinate_contract,
            source_entry,
            source_composition,
            source_authority_value,
            owner_local_route,
            candidate,
            mesh_data,
            mesh_authority,
            source_map_id,
            source_context_id,
            rank,
            execution_group,
            source_coordinate_stencil,
            exact_vertex_coordinate_offsets,
            graph_migration_evidence,
            adaptive_execution,
            graph_partition_plan=graph_partition_plan,
            axis_name="periodic_owner",
        )

    @overload
    def execute(self, /) -> NativeMeshConstruction: ...

    @overload
    def execute(
        self,
        /,
        *,
        execution_group: ExecutionGroup,
        graph_partition_plan: GraphPartitionPlan | None = None,
    ) -> MeshAssembly: ...

    def execute(
        self,
        /,
        *,
        execution_group: ExecutionGroup | None = None,
        graph_partition_plan: GraphPartitionPlan | None = None,
    ) -> NativeMeshConstruction | MeshAssembly:
        from ..._execution_runtime import ExecutionGroup
        from ...graph._partition import GraphPartitionPlan

        if execution_group is not None and not isinstance(
            execution_group, ExecutionGroup
        ):
            raise TypeError("execution_group must be the current ExecutionGroup.")
        if graph_partition_plan is not None and not isinstance(
            graph_partition_plan, GraphPartitionPlan
        ):
            raise TypeError("graph_partition_plan must be GraphPartitionPlan or None.")
        if graph_partition_plan is not None and execution_group is None:
            raise ValueError("graph_partition_plan requires an ExecutionGroup.")
        if self.source.point_owner_ranks is not None:
            if execution_group is None:
                raise ValueError(
                    "Owner-local periodic plans require the current ExecutionGroup."
                )
            return self._execute_owner_local(
                execution_group,
                graph_partition_plan=graph_partition_plan,
            )
        if execution_group is not None:
            raise ValueError(
                "execution_group applies only to an owner-local periodic source."
            )
        if graph_partition_plan is not None:
            raise ValueError(
                "graph_partition_plan applies only to an owner-local periodic source."
            )
        constructed_geometry = None
        if isinstance(self.source.domain, PeriodicPointOrbits):
            constructed = (
                periodic_power_mesh(
                    self.source.domain,
                    self.source.site_weights,
                    source_id=self.source.source_id,
                    source_revision=self.source.source_revision,
                )
                if self.options.route == "periodic_power"
                else periodic_delaunay_mesh(self.source.domain)
            )
            mesh = canonicalize_cell_mesh(constructed.mesh)
            constructed_geometry = constructed.geometry
            if constructed_geometry is not None:
                constructed_geometry.resolve(mesh)
            if self.options.route == "periodic_power":
                from ...discretization._exact_power_geometry import (
                    ExactPowerCellGeometrySource,
                )

                if constructed_geometry is None:
                    raise ValueError(
                        "Periodic power construction requires its exact geometry map."
                    )
                source_map = constructed_geometry.source_map
                if not isinstance(source_map, ExactPowerCellGeometrySource):
                    raise TypeError(
                        "Periodic power geometry must retain ExactPowerCellGeometrySource."
                    )
                source_map.require_periodic_mesh(mesh)
            source_mesh = None
            source_geometry = None
            source_ids = np.asarray(self.source.scope_entity_ids(0), dtype=np.int64)
            source_size_entity_set_id = self.source.scope_entity_set_id(0)
            source_regions = None
        else:
            original_mesh = self.source.domain
            mesh = canonicalize_cell_mesh(original_mesh)
            source_mesh = mesh
            source_geometry = CellGeometrySpec.affine(mesh)
            source_ids = _periodic_quotient_entity_ids(mesh, 0)
            source_size_entity_set_id = _periodic_quotient_entity_set_id(mesh, 0)
            source_regions = None
            if self.source.cell_regions is not None:
                original_regions = _cell_regions_in_entity_order(
                    original_mesh, self.source.cell_regions
                )
                original_ids = np.asarray(
                    original_mesh.entity_set(
                        original_mesh.topological_dimension
                    ).entity_ids,
                    dtype=np.int64,
                )
                region_by_id = {
                    int(identifier): int(region)
                    for identifier, region in zip(
                        original_ids, original_regions, strict=True
                    )
                }
                source_regions = np.asarray(
                    [
                        region_by_id[int(identifier)]
                        for block in mesh.blocks
                        for identifier in np.asarray(block.global_ids, dtype=np.int64)
                    ],
                    dtype=np.int64,
                )
        target_scope_ids = None
        if self.options.route == "periodic_power":
            region_by_code = {}
            target_regions = None
        else:
            metric_controls = _periodic_metric_controls(
                self.source, self.specification, mesh, source_regions
            )
            region_by_code = metric_controls[4]
            if mesh.topological_dimension == 2:
                (
                    mesh,
                    target_scope_ids,
                    target_regions,
                    _,
                    region_by_code,
                ) = _adapt_periodic_triangles(
                    self.source,
                    self.specification,
                    mesh,
                    source_regions,
                    metric_controls,
                )
            elif mesh.topological_dimension == 3:
                (
                    mesh,
                    target_scope_ids,
                    target_regions,
                    _,
                    region_by_code,
                ) = _adapt_periodic_tetrahedra(
                    self.source,
                    self.specification,
                    mesh,
                    source_regions,
                    metric_controls,
                    self.coordinate_contract,
                )
            else:
                raise ValueError(
                    "Periodic Delaunay metric adaptation requires dimension two or three."
                )
        region_by_target_id = None
        if target_regions is not None:
            old_cell_ids = np.asarray(
                mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
            )
            region_by_target_id = {
                int(identifier): int(region)
                for identifier, region in zip(old_cell_ids, target_regions, strict=True)
            }
        canonical_mesh = canonicalize_cell_mesh(mesh)
        if canonical_mesh is not mesh:
            mesh = canonical_mesh
            if region_by_target_id is not None:
                canonical_cell_ids = np.asarray(
                    mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
                )
                if set(region_by_target_id) != set(map(int, canonical_cell_ids)):
                    raise ValueError(
                        "Periodic material IDs changed during canonicalization."
                    )
                target_regions = np.asarray(
                    [
                        region_by_target_id[int(identifier)]
                        for identifier in canonical_cell_ids
                    ],
                    dtype=np.int64,
                )
        if self.options.route == "periodic_power":
            requested, achieved, issues = (), (), ()
        else:
            requested, achieved, issues = periodic_size_compliance(
                mesh,
                self.specification,
                self.source.source_id,
                self.source.source_revision,
                source_ids,
                source_entity_set_id=source_size_entity_set_id,
                target_scope_entity_ids=target_scope_ids,
            )
        compliance = MeshingComplianceReport(
            self.specification.specification_id,
            issues=issues,
            requested=requested,
            achieved=achieved,
        )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("periodic"),
            self.source.source_id,
            self.source.source_revision,
        )
        attributes = ()
        zones = []
        if target_regions is not None:
            cell_entities = mesh.entity_set(mesh.topological_dimension)
            scope = MeshingScope.whole(mesh, mesh.topological_dimension)
            if scope.entity_set_id != cell_entities.entity_set_id:
                raise ValueError(
                    "Periodic material scope does not match the adapted cell set."
                )
            attributes = (
                MeshAttribute(
                    "periodic-material-id",
                    MeshAttributeRole.MATERIAL,
                    scope,
                    target_regions,
                ),
            )
            target_cell_ids = np.asarray(cell_entities.entity_ids, dtype=np.int64)
            for code, control in sorted(region_by_code.items()):
                ids = target_cell_ids[np.asarray(target_regions, dtype=np.int64) == code]
                if ids.size == 0:
                    raise ValueError(
                        "Periodic adaptation removed a declared material region."
                    )
                zone_scope = MeshingScope(
                    mesh.mesh_id,
                    mesh.numeric_version,
                    MeshingEntityKind.MESH,
                    mesh.topological_dimension,
                    cell_entities.entity_set_id,
                    ids,
                )
                zones.append(
                    MeshZone(
                        control.region_name,
                        MeshZoneRole.REGION,
                        zone_scope,
                        material_id=control.material_id,
                        region_role=control.role,
                    )
                )
        if self.options.route == "periodic_power":
            if constructed_geometry is None:
                raise ValueError(
                    "Periodic power construction requires its exact geometry map."
                )
            geometry = constructed_geometry
        else:
            geometry = CellGeometrySpec.affine(mesh)
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=NativeMeshOrganization(
                attributes=attributes, zones=tuple(zones)
            ),
            source_mesh=source_mesh,
            source_geometry=source_geometry,
        )


@checked
def prepare_native_periodic(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedNativePeriodic:
    if not isinstance(source, NativePeriodicSource):
        raise TypeError("source must be NativePeriodicSource.")
    source.validate_restored()
    if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
        raise TypeError("specification must be SurfaceMeshingSpec or VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions) or options.route not in (
        "periodic_delaunay",
        "periodic_power",
    ):
        raise ValueError("options must select periodic_delaunay or periodic_power.")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    dimension = source.ambient_dimension
    owner_local = source.point_owner_ranks is not None
    if (
        specification.target.topological_dimension != dimension
        or specification.target.ambient_dimension != dimension
    ):
        raise ValueError(
            "Periodic meshing requires a full-dimensional target matching the source lattice."
        )
    periodic_power = options.route == "periodic_power"
    if periodic_power:
        if (
            not isinstance(source.domain, PeriodicPointOrbits)
            or dimension != 3
            or not isinstance(specification, VolumeMeshingSpec)
            or specification.fill_strategy is not VolumeFillStrategy.POLYHEDRAL
            or set(specification.target.cell_families.required) != {"polyhedron"}
        ):
            raise ValueError(
                "Periodic power requires a 3D point-orbit source and an exclusively polyhedral volume target."
            )
        if specification.size_controls:
            raise ValueError(
                "Periodic power has no site-size adaptation route; size controls must be absent."
            )
        required_kind = "polyhedron"
    else:
        required_kind = {2: "triangle", 3: "tetrahedron"}.get(dimension)
        if (
            required_kind is None
            or required_kind not in specification.target.cell_families.required
        ):
            raise ValueError(
                "Periodic Delaunay requires the corresponding simplex family as a required target."
            )
    size_controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if len(size_controls) != len(specification.size_controls):
        raise ValueError("Periodic adaptation supports only uniform size controls.")
    if specification.maximum_deviation is not None:
        raise ValueError(
            "Periodic point or mesh sources have no analytic CAD deviation authority."
        )
    scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
    domain_entity_ids = np.asarray(source.scope_entity_ids(dimension), dtype=np.int64)
    if (
        scope.entity_dimension != dimension
        or scope.entity_set_id != source.scope_entity_set_id(dimension)
        or not np.array_equal(
            np.sort(np.asarray(scope.entity_ids, dtype=np.int64)),
            np.sort(domain_entity_ids),
        )
    ):
        raise ValueError(
            "Periodic top-dimensional scope must bind the authored fundamental-domain entity bank."
        )
    if isinstance(source.domain, PeriodicPointOrbits):
        point_ids = np.asarray(source.scope_entity_ids(0), dtype=np.int64)
        if owner_local:
            if (
                specification.protected_features
                or specification.region_controls
                or specification.patch_controls
                or specification.periodic_constraints
                or specification.layer_controls
                or (
                    isinstance(specification, VolumeMeshingSpec)
                    and (specification.region_seeds or specification.hole_seeds)
                )
            ):
                raise ValueError(
                    "Periodic Delaunay point-orbit generation does not yet own protected, material, patch, layered, or additional source-pair controls."
                )
            for control in size_controls:
                if (
                    control.scope.source_id != source.source_id
                    or control.scope.source_revision != source.source_revision
                    or control.scope.scope_id != scope.scope_id
                ):
                    raise ValueError(
                        "Owner-local periodic size controls must bind the complete authored fundamental-domain scope."
                    )
        else:
            validate_periodic_source_controls(
                source.source_id,
                source.source_revision,
                point_ids,
                specification,
            )
        if (
            not owner_local
            and not periodic_power
            and source.site_weights is not None
            and np.any(np.asarray(source.site_weights, dtype=np.float64) != 0.0)
        ):
            raise ValueError(
                "Periodic Delaunay is unweighted; nonzero site weights require periodic_power."
            )
        if not periodic_power and not owner_local:
            if any(
                control.scope.entity_set_id != source.scope_entity_set_id(0)
                for control in size_controls
            ):
                raise ValueError(
                    "Periodic point size controls must use the authored point-orbit entity set."
                )
            selected = np.unique(
                np.concatenate(
                    tuple(
                        np.asarray(control.scope.entity_ids, dtype=np.int64)
                        for control in size_controls
                    )
                    + (np.zeros((0,), dtype=np.int64),)
                )
            )
            if not np.array_equal(selected, np.sort(point_ids)):
                raise ValueError(
                    "Periodic size controls must cover every authored point orbit."
                )
    else:
        if periodic_power:
            raise ValueError(
                "Periodic power requires authored periodic point-orbit source identities."
            )
        mesh = source.domain
        if any(block.cell_kind != required_kind for block in mesh.blocks):
            raise ValueError(
                "Represented periodic sources must use the required simplex family."
            )
        if (
            specification.patch_controls
            or specification.layer_controls
            or (
                isinstance(specification, VolumeMeshingSpec)
                and (specification.region_seeds or specification.hole_seeds)
            )
        ):
            raise ValueError(
                "Periodic source patch, layer, and seed controls require an authored geometry-domain owner."
            )
        validate_periodic_mesh_constraints(
            mesh,
            source.source_id,
            source.source_revision,
            tuple(specification.periodic_constraints),
        )
        source_ids = _periodic_quotient_entity_ids(mesh, 0)
        size_entity_set_id = _periodic_quotient_entity_set_id(mesh, 0)
        if any(
            control.scope.entity_set_id != size_entity_set_id for control in size_controls
        ):
            raise ValueError(
                "Periodic mesh size controls must use its quotient vertex entity set."
            )
        periodic_metric_from_uniform_controls(
            mesh,
            specification,
            source.source_id,
            source.source_revision,
            source_ids,
            size_entity_set_id,
        )
        _periodic_metric_controls(source, specification, mesh, source.cell_regions)
    return PreparedNativePeriodic(source, specification, options, coordinate_contract)


def rebuild_owner_local_periodic_assembly_from_shards(
    stored_shards: Iterable[OwnerLocalCellGeometry],
    execution_group: ExecutionGroup,
    /,
    *,
    current_specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    graph_partition_plan: GraphPartitionPlan | None = None,
    axis_name: str = "periodic_shard_reattachment",
) -> MeshAssembly:
    """Reattach streamed archived source shards to a current owner group.

    The stream contains disjoint stored local views. Only authored owned point
    rows cross the reattachment boundary; current owners are recomputed from
    stable site IDs, and the existing native route rebuilds support and graph
    migration before the new local assembly is returned.
    """
    import json

    import jax

    from ..._execution_runtime import ExecutionGroup
    from ...discretization._periodic_cell import PeriodicCell
    from ...discretization._view_support import OwnerLocalCellGeometry
    from ...graph._partition import GraphPartitionPlan
    from ...meshing._assembly import MeshAssembly
    from ...meshing._contracts import (
        SurfaceMeshingSpec,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from .._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    if (
        not isinstance(execution_group, ExecutionGroup)
        or not execution_group.is_member
        or not isinstance(current_specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
        or graph_partition_plan is not None
        and not isinstance(graph_partition_plan, GraphPartitionPlan)
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise TypeError("Periodic shard reattachment requires typed current inputs.")
    members = execution_group.spec.process_indices
    devices = execution_group.devices
    try:
        rank = members.index(jax.process_index())
    except ValueError as error:
        raise ValueError(
            "The current process is outside the reattachment group."
        ) from error
    if (
        tuple(device.process_index for device in devices) != members
        or graph_partition_plan is not None
        and graph_partition_plan.part_count != len(members)
    ):
        raise ValueError("Periodic shard reattachment group and graph plan differ.")
    power_request = (
        isinstance(current_specification, VolumeMeshingSpec)
        and current_specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
        and set(current_specification.target.cell_families.required) == {"polyhedron"}
    )
    iterator = iter(stored_shards)
    local_header: dict[str, Any] | None = None
    local_point_records: dict[int, tuple[tuple[int, ...], int | None]] = {}
    local_shard_count = 0
    while True:
        shard: OwnerLocalCellGeometry | None = None
        local_error = None
        try:
            shard = next(iterator)
        except StopIteration:
            pass
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        present = shard is not None
        presence = np.asarray(
            _owner_group_allgather(
                np.asarray([int(present)], dtype=np.int32),
                execution_group,
                f"{axis_name}_shard_presence",
            ),
            dtype=np.int32,
        ).reshape(-1)
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_shard_read",
            "periodic archived shard stream",
        )
        if presence.shape != (len(members),) or np.any(presence < 0):
            raise ValueError("Periodic shard stream presence record is malformed.")
        if not np.any(presence):
            break
        local_error = None
        try:
            if present:
                from .._periodic import PeriodicPointOrbits

                if not isinstance(shard, OwnerLocalCellGeometry):
                    raise TypeError("Shard stream values must be OwnerLocalCellGeometry.")
                authored_source = shard.source_value
                if (
                    not isinstance(authored_source, NativePeriodicSource)
                    or not isinstance(authored_source.domain, PeriodicPointOrbits)
                    or authored_source.source_id != shard.source_id
                    or authored_source.source_revision != shard.source_revision
                ):
                    raise TypeError(
                        "Archived periodic shard lacks its typed authored source."
                    )
                source, source_map = _owner_local_periodic_source_from_view(shard)
                source_authority = shard.source_geometry_authority
                if source_map is not None:
                    if not power_request:
                        raise ValueError(
                            "Archived exact power map differs from its typed request."
                        )
                    route = "periodic_power"
                elif isinstance(source_authority, NativePeriodicSource):
                    if power_request:
                        if (
                            shard.source_map_id is not None
                            or np.asarray(shard.vertex_ids).size
                            or shard.blocks
                        ):
                            raise ValueError(
                                "Nonempty periodic power geometry lacks its exact power map."
                            )
                        route = "periodic_power"
                    else:
                        route = "periodic_delaunay"
                else:
                    raise TypeError(
                        "Archived periodic shard has no concrete source geometry authority."
                    )
                if (
                    not isinstance(source.domain, PeriodicPointOrbits)
                    or source.point_owner_ranks is None
                    or shard.serial_source_view
                    or shard.periodic_topology_id is None
                ):
                    raise TypeError(
                        "Archived shard must retain its exact periodic point source."
                    )
                periodic_cell = source.domain.cell
                points = np.asarray(source.domain.points, dtype=np.float64)
                point_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
                point_owners = np.asarray(source.point_owner_ranks, dtype=np.int32)
                shard_site_ids = np.asarray(shard.source_point_ids, dtype=np.int64)
                shard_site_owners = np.asarray(shard.source_owner_ranks, dtype=np.int32)
                shard_site_owned = np.asarray(shard.source_owned, dtype=np.bool_)
                owner_by_id = {
                    int(identifier): int(owner)
                    for identifier, owner in zip(
                        shard_site_ids.tolist(),
                        shard_site_owners.tolist(),
                        strict=True,
                    )
                }
                owned_by_id = {
                    int(identifier): bool(owned_value)
                    for identifier, owned_value in zip(
                        shard_site_ids.tolist(),
                        shard_site_owned.tolist(),
                        strict=True,
                    )
                }
                if (
                    points.shape != (point_ids.size, source.ambient_dimension)
                    or point_owners.shape != point_ids.shape
                    or shard_site_owners.shape != shard_site_ids.shape
                    or shard_site_owned.shape != shard_site_ids.shape
                    or set(point_ids.tolist()) != set(shard_site_ids.tolist())
                    or any(
                        owner_by_id[int(identifier)] != int(owner)
                        or owned_by_id[int(identifier)]
                        != (int(owner) == shard.owner_rank)
                        for identifier, owner in zip(
                            point_ids.tolist(), point_owners.tolist(), strict=True
                        )
                    )
                ):
                    raise ValueError(
                        "Archived source point ownership rows are malformed."
                    )
                owned = np.asarray(
                    [owned_by_id[int(identifier)] for identifier in point_ids.tolist()],
                    dtype=np.bool_,
                )
                weights = (
                    None
                    if source.site_weights is None
                    else np.asarray(source.site_weights, dtype=np.float64)
                )
                if authored_source.site_weights is not None and weights is None:
                    authored_weights = np.asarray(
                        authored_source.site_weights, dtype=np.float64
                    )
                    if np.any(authored_weights != 0.0):
                        raise ValueError(
                            "Periodic source authority dropped nonzero site weights."
                        )
                    weights = np.zeros(point_ids.shape, dtype=np.float64)
                if weights is not None and weights.shape != point_ids.shape:
                    raise ValueError("Archived source weights differ from site IDs.")
                if (
                    route == "periodic_delaunay"
                    and weights is not None
                    and np.any(weights != 0.0)
                ):
                    raise ValueError(
                        "Weighted periodic sources require exact power geometry."
                    )
                adaptive = shard.adaptive_result
                specification_id = shard.specification_id
                if (
                    not isinstance(specification_id, str)
                    or not specification_id
                    or adaptive is not None
                    and adaptive.specification_id != specification_id
                ):
                    raise ValueError(
                        "Archived periodic specification identity is missing or stale."
                    )
                graph_plan_id = (
                    None
                    if adaptive is None or adaptive.graph_partition_plan is None
                    else adaptive.graph_partition_plan.plan_id
                )
                if shard.graph_migration_evidence is not None:
                    evidence_plan_id = shard.graph_migration_evidence.result.plan_id
                    if graph_plan_id is not None and evidence_plan_id != graph_plan_id:
                        raise ValueError("Archived graph evidence differs from its plan.")
                    graph_plan_id = evidence_plan_id
                vectors = np.asarray(periodic_cell.vectors, dtype=np.float64)
                origin = np.asarray(periodic_cell.origin, dtype=np.float64)
                header: dict[str, Any] = {
                    "source_content_id": shard.owner_directory_data.source_manifest_id,
                    "source_id": source.source_id,
                    "source_revision": source.source_revision,
                    "source_mesh_id": shard.source_mesh_id,
                    "source_topology_id": shard.source_topology_id,
                    "source_geometry_id": shard.source_geometry_id,
                    "periodic_topology_id": shard.periodic_topology_id,
                    "coordinate_contract": shard.coordinate_contract.to_dict(),
                    "coordinate_contract_id": shard.coordinate_contract_id,
                    "specification_id": specification_id,
                    "graph_partition_plan_id": graph_plan_id,
                    "route": route,
                    "weighted": weights is not None,
                    "periodic_cell": {
                        "cell_id": periodic_cell.cell_id,
                        "vectors_shape": list(vectors.shape),
                        "vectors_bits": [
                            int(value) for value in vectors.view(np.uint64).reshape(-1)
                        ],
                        "origin_bits": [
                            int(value) for value in origin.view(np.uint64).reshape(-1)
                        ],
                        "periodic_axes": list(periodic_cell.periodic_axes),
                        "condition": float(
                            periodic_cell.certified_condition_number
                        ).hex(),
                    },
                }
                header_bytes = json.dumps(
                    header, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
                if local_header is not None:
                    prior_bytes = json.dumps(
                        local_header, sort_keys=True, separators=(",", ":")
                    ).encode("ascii")
                    if prior_bytes != header_bytes:
                        raise ValueError("Archived periodic source shards disagree.")
                else:
                    local_header = header
                point_bits = points.view(np.uint64)
                weight_bits = None if weights is None else weights.view(np.uint64)
                for row in np.flatnonzero(owned).tolist():
                    identifier = int(point_ids[row])
                    record = (
                        tuple(int(value) for value in point_bits[row].tolist()),
                        None if weight_bits is None else int(weight_bits[row]),
                    )
                    if identifier in local_point_records:
                        raise ValueError(
                            "An authored source site occurs in multiple stored owner shards."
                        )
                    local_point_records[identifier] = record
                local_shard_count += 1
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis_name}_shard_validation",
            "periodic archived source shard",
        )
    shard_counts = np.asarray(
        _owner_group_allgather(
            np.asarray([local_shard_count], dtype=np.int64),
            execution_group,
            f"{axis_name}_shard_counts",
        ),
        dtype=np.int64,
    ).reshape(-1)
    if shard_counts.shape != (len(members),) or int(np.sum(shard_counts)) == 0:
        raise ValueError("Periodic shard stream did not provide a source declaration.")
    local_header_bytes = (
        b""
        if local_header is None
        else json.dumps(local_header, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    )
    lengths = np.asarray(
        _owner_group_allgather(
            np.asarray([len(local_header_bytes)], dtype=np.int64),
            execution_group,
            f"{axis_name}_header_lengths",
        ),
        dtype=np.int64,
    ).reshape(-1)
    if lengths.shape != (len(members),) or np.any(lengths < 0):
        raise ValueError("Periodic source declaration lengths are malformed.")
    header_width = max(1, int(np.max(lengths, initial=0)))
    local_header_row = np.zeros((header_width,), dtype=np.uint8)
    if local_header_bytes:
        local_header_row[: len(local_header_bytes)] = np.frombuffer(
            local_header_bytes, dtype=np.uint8
        )
    header_rows = np.asarray(
        _owner_group_allgather(
            local_header_row,
            execution_group,
            f"{axis_name}_header_values",
        ),
        dtype=np.uint8,
    )
    common_headers = tuple(
        bytes(header_rows[row, : int(lengths[row])])
        for row in range(len(members))
        if lengths[row] > 0
    )
    local_error = None
    try:
        if not common_headers or any(
            value != common_headers[0] for value in common_headers
        ):
            raise ValueError("Archived source declarations disagree across shards.")
        header_value = json.loads(common_headers[0])
        if not isinstance(header_value, dict):
            raise ValueError("Periodic source declaration is not a JSON object.")
        header = header_value
        source_id = str(header["source_id"])
        source_revision = str(header["source_revision"])
        scope = (
            current_specification.scope
            if isinstance(current_specification, SurfaceMeshingSpec)
            else current_specification.boundary_scope
        )
        if (
            not source_id
            or not source_revision
            or scope.source_id != source_id
            or scope.source_revision != source_revision
            or current_specification.specification_id != header["specification_id"]
            or header["graph_partition_plan_id"]
            != (None if graph_partition_plan is None else graph_partition_plan.plan_id)
        ):
            raise ValueError(
                "Current periodic source/specification differs from its archive."
            )
        if graph_partition_plan is not None and (
            graph_partition_plan.part_count != len(members)
        ):
            raise ValueError("Current graph plan does not match the owner group.")
        cell_header = header["periodic_cell"]
        if not isinstance(cell_header, dict):
            raise ValueError("Periodic cell declaration is malformed.")
        vectors_shape = tuple(int(value) for value in cell_header["vectors_shape"])
        vectors_bits = np.asarray(cell_header["vectors_bits"], dtype=np.uint64)
        origin_bits = np.asarray(cell_header["origin_bits"], dtype=np.uint64)
        if (
            len(vectors_shape) != 2
            or vectors_shape[0] <= 0
            or vectors_shape[1] != vectors_shape[0]
            or vectors_bits.size != prod(vectors_shape)
            or origin_bits.shape != (vectors_shape[1],)
        ):
            raise ValueError("Periodic source lattice declaration has invalid shape.")
        periodic_cell = PeriodicCell(
            vectors_bits.reshape(vectors_shape).view(np.float64),
            origin=origin_bits.view(np.float64),
            periodic_axes=tuple(bool(value) for value in cell_header["periodic_axes"]),
            maximum_condition_number=float.fromhex(str(cell_header["condition"])),
        )
        if periodic_cell.cell_id != cell_header["cell_id"]:
            raise ValueError(
                "Restored periodic lattice differs from its exact source ID."
            )
        coordinate_contract = SpatialCoordinateContract.from_dict(
            header["coordinate_contract"]
        )
        if coordinate_contract.spatial_id != header["coordinate_contract_id"]:
            raise ValueError("Restored spatial coordinate contract changed identity.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        source_id = ""
        source_revision = ""
        periodic_cell = None
        coordinate_contract = None
        header = {}
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_declaration",
        "periodic shard source declaration",
    )
    if periodic_cell is None or coordinate_contract is None:
        raise RuntimeError(
            "Periodic shard declaration returned no typed lattice contract."
        )
    dimension = periodic_cell.ambient_dimension
    weighted = bool(header["weighted"])
    width = 1 + dimension + int(weighted)
    parts = len(members)
    outgoing_rows: list[list[np.ndarray]] = [[] for _ in range(parts)]
    local_point_rows: dict[int, tuple[tuple[int, ...], int | None]] = {}
    local_error = None
    try:
        for identifier, (coordinate_bits, weight_bits) in local_point_records.items():
            digest = bytes.fromhex(
                canonical_fingerprint(("periodic-source-point-owner", identifier))
            )
            owner = digest[0] % parts
            identifier_bits = int(np.asarray(identifier, dtype=np.int64).view(np.uint64))
            row_values = [
                identifier_bits,
                *coordinate_bits,
                *(() if weight_bits is None else (weight_bits,)),
            ]
            if len(row_values) != width:
                raise ValueError(
                    "Periodic source point row does not match its declaration."
                )
            row = np.asarray(row_values, dtype=np.uint64)
            if owner == rank:
                local_point_rows[identifier] = (coordinate_bits, weight_bits)
            else:
                outgoing_rows[owner].append(row)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_source_encode",
        "periodic current source owner routing",
    )
    outgoing = tuple(
        np.stack(rows) if rows else np.zeros((0, width), dtype=np.uint64)
        for rows in outgoing_rows
    )
    incoming = _owner_exchange_message_banks(
        outgoing,
        rank,
        execution_group,
        f"{axis_name}_source_points",
    )
    local_error = None
    try:
        for rows in incoming.values():
            values = np.asarray(rows, dtype=np.uint64)
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError("Periodic source owner rows have an invalid layout.")
            for row in values:
                identifier = int(np.asarray(row[0], dtype=np.uint64).view(np.int64))
                coordinate_bits = tuple(int(value) for value in row[1 : 1 + dimension])
                weight_bits = int(row[-1]) if weighted else None
                if identifier in local_point_rows:
                    raise ValueError(
                        "Periodic source owner routing duplicated a site ID."
                    )
                local_point_rows[identifier] = (coordinate_bits, weight_bits)
        point_ids = np.asarray(sorted(local_point_rows), dtype=np.int64)
        point_bits = np.asarray(
            [local_point_rows[int(identifier)][0] for identifier in point_ids],
            dtype=np.uint64,
        ).reshape((-1, dimension))
        points = point_bits.view(np.float64)
        site_weights = (
            None
            if not weighted
            else np.asarray(
                [local_point_rows[int(identifier)][1] for identifier in point_ids],
                dtype=np.uint64,
            ).view(np.float64)
        )
        point_owner_ranks = np.full(point_ids.shape, rank, dtype=np.int32)
        source = NativePeriodicSource(
            PeriodicPointOrbits(periodic_cell, points, point_ids),
            source_id,
            source_revision,
            site_weights=site_weights,
            point_owner_ranks=point_owner_ranks,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        source = None
        point_ids = np.empty((0,), dtype=np.int64)
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_source_build",
        "periodic reattached source bank",
    )
    if source is None:
        raise RuntimeError("Periodic source reattachment produced no local source.")
    _, _, source_content_id = _owner_local_periodic_source_authority(
        source,
        execution_group,
        f"{axis_name}_source_authority",
    )
    local_error = (
        None
        if source_content_id == header["source_content_id"]
        else "Streamed source rows changed the authored periodic source content."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_source_identity",
        "periodic reattached source identity",
    )
    if header["route"] == "periodic_delaunay":
        options = NativeMeshingOptions("periodic_delaunay")
    elif header["route"] == "periodic_power":
        options = NativeMeshingOptions("periodic_power")
    else:
        raise ValueError("Archived periodic route is not supported.")
    prepared = prepare_native_periodic(
        source,
        current_specification,
        options,
        coordinate_contract,
    )
    assembly = prepared.execute(
        execution_group=execution_group,
        graph_partition_plan=graph_partition_plan,
    )
    if not isinstance(assembly, MeshAssembly) or not assembly.owner_local:
        raise TypeError("Periodic shard reattachment did not produce MeshAssembly.")
    if len(assembly.parts) != 1:
        raise ValueError("Periodic reattachment requires one local assembly part.")
    view = assembly.parts[0].carrier
    local_error = (
        None
        if isinstance(view, OwnerLocalCellGeometry)
        else "Periodic reattachment produced the wrong local geometry carrier."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_result_view",
        "periodic changed-placement local view",
    )
    if not isinstance(view, OwnerLocalCellGeometry):
        raise RuntimeError("Periodic reattachment accepted no local geometry carrier.")
    view.require_owner_directory_current(
        execution_group,
        axis_name=f"{axis_name}_result_directory",
    )
    local_error = (
        None
        if (
            view.owner_directory_data.source_manifest_id == header["source_content_id"]
            and view.owner_directory_data.group_id == execution_group.spec.group_id
            and view.owner_directory_data.part_count == len(members)
            and view.owner_directory_data.rank == rank
            and view.source_id == header["source_id"]
            and view.source_revision == header["source_revision"]
            and view.source_mesh_id == header["source_mesh_id"]
            and view.source_topology_id == header["source_topology_id"]
            and view.source_geometry_id == header["source_geometry_id"]
            and view.periodic_topology_id == header["periodic_topology_id"]
            and view.coordinate_contract_id == header["coordinate_contract_id"]
            and view.owner_rank == rank
        )
        else "Reattached periodic assembly changed a global source/topology/geometry identity."
    )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_result_identity",
        "periodic changed-placement assembly identity",
    )
    return assembly


__all__ = [
    "NativePeriodicSource",
    "PreparedNativePeriodic",
    "prepare_native_periodic",
    "rebuild_owner_local_periodic_assembly_from_shards",
]
