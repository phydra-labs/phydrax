#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify native initial meshes collectively from a source-discriminated premise.

A surface premise binds authored patches, charts and trims; a planar premise
binds a worker-count-independent tile complex of an authored planar region.
Owner-local proofs use the premise; cross-owner checks use bounded packets or
tile containment; logical content remains on the sharded array substrate.
Source-stratum ordering, duplicate admission and incidence joins share one
numerical program; only bounded owning-host inputs and scalar counts cross
the host boundary, and retained content keeps the canonical byte recipes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import assert_never, final, Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec
from numpy.typing import NDArray

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._cell_mesh as _owner_phydrax_discretization__cell_mesh
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _owner_phydrax_discretization__coordinate_enclosure
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._meshing_domain as _dependency_phydrax_geometry__meshing_domain
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.meshing._audit as _owner_phydrax_meshing__audit
import phydrax.meshing._audit_topology as _dependency_phydrax_meshing__audit_topology
import phydrax.meshing._collective_organization as _owner_phydrax_meshing__collective_organization
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._contracts as _owner_phydrax_meshing__contracts
import phydrax.meshing._device_adaptation as _dependency_phydrax_meshing__device_adaptation
import phydrax.meshing._device_generation as _dependency_phydrax_meshing__device_generation
import phydrax.meshing._distributed_generation as _dependency_phydrax_meshing__distributed_generation
import phydrax.meshing._distributed_planar_generation as planar_generation_owner
import phydrax.meshing._domain as _dependency_phydrax_meshing__domain
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._organization as _owner_phydrax_meshing__organization
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._publication_lowering as _owner_phydrax_meshing__publication_lowering
import phydrax.meshing._result as _owner_phydrax_meshing__result
import phydrax.meshing._scope as _owner_phydrax_meshing__scope
import phydrax.meshing._surface_construction as surface_construction_owner
import phydrax.meshing._trace as _owner_phydrax_meshing__trace
import phydrax.meshing.providers._native_schedules as _dependency_phydrax_meshing_providers__native_schedules
import phydrax.meshing.providers._native_sources as _dependency_phydrax_meshing_providers__native_sources
import phydrax.meshing.providers._native_surface_association as surface_association_owner
from phydrax.meshing.providers._native_planar_domain import (
    _association_distances,
    _field_sizes,
    source_arrays,
)

from .._execution_runtime import owner_execution_mesh, process_owner_devices
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
    tree_content_equal,
)
from .._meshcore import exact_orient2d, MeshcoreError, NativeExecutionScope
from .._physical import SpatialCoordinateContract
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..typing import checked
from ._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from ._collective_failure import (
    agreed_local_step,
    collective_failure_barrier,
    collective_failure_decision,
)
from ._compliance_failure import compliance_failure
from ._controls import FeatureKind
from ._result import AbstractCollectiveMeshTheorem
from ._sizing import size_statistic_quantile, SizeControlStrength, UniformSizeControl


_SURFACE_CHECKS = (
    "authored_source_binding",
    "chart_trim_coverage",
    "continuous_source_fidelity",
    "local_embedding",
    "physical_controls",
    "unique_cell_ownership",
    "shared_vertex_geometry",
    "reciprocal_source_facets",
    "cross_owner_embedding",
)
# Same positions as the surface checks: coverage, ownership, shared geometry,
# reciprocal facets and cross-owner disjointness keep their indices.
_PLANAR_CHECKS = (
    "authored_source_binding",
    "tile_region_coverage",
    "source_carrier_fidelity",
    "local_embedding",
    "physical_controls",
    "unique_cell_ownership",
    "shared_vertex_geometry",
    "reciprocal_source_facets",
    "tile_containment",
)


def _layout_key(
    layout: _dependency_phydrax_meshing__device_generation.DeviceGenerationLayout, /
) -> tuple[int, int, int, int, int]:
    return (
        layout.vertex_capacity,
        layout.cell_capacity,
        layout.candidate_capacity,
        layout.maximum_work_units,
        layout.neighborhood_depth,
    )


def _immutable_owners(owners: NDArray[np.int32], rows: int, name: str, /) -> NDArray[np.int32]:
    value = np.array(owners, dtype=np.int32, copy=True)
    if value.shape != (rows,) or np.any(value < 0):
        raise ValueError(
            f"initial.premise.{name} must be nonnegative int32 rows aligned with generation units; got incompatible owners"
        )
    value.flags.writeable = False
    return value


@final
class InitialSurfacePremise(StrictModule, NonTrainableState):
    """Authored parametric surface domain, request, schedule and patch owners."""

    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain
    specification: _dependency_phydrax_meshing__contracts.SurfaceMeshingSpec
    authored_source: (
        _dependency_phydrax_meshing_providers__native_sources.NativeSurfaceSource
    )
    schedule: (
        _dependency_phydrax_meshing_providers__native_schedules.NativeSurfaceSchedule
    )
    layout: _dependency_phydrax_meshing__device_generation.DeviceGenerationLayout
    patch_owners: NDArray[np.int32]
    premise_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
        specification: _dependency_phydrax_meshing__contracts.SurfaceMeshingSpec,
        schedule: _dependency_phydrax_meshing_providers__native_schedules.NativeSurfaceSchedule,
        layout: _dependency_phydrax_meshing__device_generation.DeviceGenerationLayout,
        patch_owners: NDArray[np.int32],
        /,
    ) -> None:
        if compiled.specification_id != specification.specification_id:
            raise ValueError(
                "initial.premise.compiled must bind its specification; got a different request"
            )
        owners = _immutable_owners(patch_owners, compiled.patches.size, "patch_owners")
        self.compiled = compiled
        self.specification = specification
        self.authored_source = (
            _dependency_phydrax_meshing_providers__native_sources.NativeSurfaceSource(
                compiled.domain
            )
        )
        self.schedule = schedule
        self.layout = layout
        self.patch_owners = owners
        self.premise_id = canonical_fingerprint(
            {
                "kind": "initial-surface-premise",
                "compiled": compiled.compiled_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "layout": _layout_key(layout),
                "owners": array_tree_fingerprint(owners),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: InitialSurfacePremise, /) -> Self:
        """Rebuild the premise through its constructor and its identity."""
        rebuilt = cls(
            value.compiled,
            value.specification,
            value.schedule,
            value.layout,
            value.patch_owners,
        )
        if rebuilt.premise_id != value.premise_id:
            raise ValueError(
                "InitialSurfacePremise.premise_id must match its stored inputs; got changed stored content"
            )
        return rebuilt

    @property
    def source_id(self) -> str:
        return self.compiled.domain.source_id

    @property
    def source_revision(self) -> str:
        return self.compiled.domain.source_revision

    @property
    def source_domain_id(self) -> str:
        return self.compiled.domain.domain_id

    @property
    def generation_id(self) -> str:
        return self.compiled.compiled_id

    @property
    def definitions_id(self) -> str:
        return canonical_fingerprint(
            (self.compiled.compiled_id, self.specification.specification_id)
        )

    @property
    def block_name(self) -> str:
        return "surface"

    @property
    def ambient_dimension(self) -> int:
        return 3

    def identity_fields(self) -> dict[str, str]:
        """Source-theorem identity fields of this premise kind."""
        return {"compiled": self.compiled.compiled_id}


@final
class InitialPlanarPremise(StrictModule, NonTrainableState):
    """Authored planar region tile complex, layout and tile owners."""

    tiles: planar_generation_owner.PlanarTileComplex
    layout: _dependency_phydrax_meshing__device_generation.DeviceGenerationLayout
    tile_owners: NDArray[np.int32]
    premise_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        tiles: planar_generation_owner.PlanarTileComplex,
        layout: _dependency_phydrax_meshing__device_generation.DeviceGenerationLayout,
        tile_owners: NDArray[np.int32],
        /,
    ) -> None:
        owners = _immutable_owners(tile_owners, tiles.tiles.shape[0], "tile_owners")
        self.tiles = tiles
        self.layout = layout
        self.tile_owners = owners
        self.premise_id = canonical_fingerprint(
            {
                "kind": "initial-planar-premise",
                "tiles": tiles.compiled_id,
                "layout": _layout_key(layout),
                "owners": array_tree_fingerprint(owners),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: InitialPlanarPremise, /) -> Self:
        """Rebuild the premise through its constructor and its identity."""
        rebuilt = cls(
            value._current_tiles(),
            value.layout,
            value.tile_owners,
        )
        if rebuilt.premise_id != value.premise_id:
            raise ValueError(
                "InitialPlanarPremise.premise_id must match its stored inputs; got changed stored content"
            )
        return rebuilt

    def require_current_domain(self) -> PiecewiseLinearDomain:
        """Authenticate authored input and retained tile banks at a proof boundary."""
        return type(self).from_stored_fields(self).tiles.region_domain()

    def _current_tiles(self) -> planar_generation_owner.PlanarTileComplex:
        """Rebuild and compare, rather than silently repairing a changed bank."""
        _dependency_phydrax_meshing_providers__native_sources.NativePlanarSource.from_stored_fields(
            self.tiles.source
        )
        rebuilt = planar_generation_owner.PlanarTileComplex.from_stored_fields(self.tiles)
        if not tree_content_equal(rebuilt, self.tiles):
            raise ValueError(
                "InitialPlanarPremise.tiles must match their authored source, request and schedule; got changed retained banks"
            )
        return rebuilt

    @property
    def authored_source(
        self,
    ) -> _dependency_phydrax_meshing_providers__native_sources.NativePlanarSource:
        return self.tiles.source

    @property
    def specification(self) -> _dependency_phydrax_meshing__contracts.SurfaceMeshingSpec:
        return self.tiles.specification

    @property
    def schedule(self) -> planar_generation_owner.PlanarTileSchedule:
        return self.tiles.schedule

    @property
    def source_id(self) -> str:
        return self.tiles.source.source_id

    @property
    def source_revision(self) -> str:
        return self.tiles.source.source_revision

    @property
    def source_domain_id(self) -> str:
        return self.tiles.source.binding_id

    @property
    def generation_id(self) -> str:
        return self.tiles.compiled_id

    @property
    def definitions_id(self) -> str:
        return canonical_fingerprint(
            (self.tiles.compiled_id, self.tiles.specification.specification_id)
        )

    @property
    def block_name(self) -> str:
        return "planar"

    @property
    def ambient_dimension(self) -> int:
        return 2

    def identity_fields(self) -> dict[str, str]:
        """Source-theorem identity fields of this premise kind."""
        return {"tiles": self.tiles.compiled_id}


type InitialMeshPremise = InitialSurfacePremise | InitialPlanarPremise


import phydrax._identity as _owner_phydrax__identity


@final
class _LocalProof(StrictModule):
    certificate_ids: tuple[str, ...] = eqx.field(static=True)
    patch_checks: NDArray[np.bool_]
    vertex_residuals: NDArray[np.float64] | None = None


@final
class _InitialOwnerProof(StrictModule):
    local: _LocalProof
    sharding: NamedSharding = eqx.field(static=True)


@final
class _InitialContent(StrictModule):
    arrays: dict[str, Array]
    keys: tuple[Array, ...]
    identifiers: tuple[Array, ...]
    owners: tuple[Array, ...]
    counts: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    coordinate_geometry_id: str = eqx.field(static=True)
    checks: Array


@final
class _InitialContactProof(StrictModule):
    checks: Array
    verdict: Array
    summary: NDArray[np.bool_]
    raw_findings: Array
    pairs: int = eqx.field(static=True)


@final
class _InitialIdentity(StrictModule):
    organization_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    source_evidence_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


@final
class _InitialOwnerRejection(StrictModule):
    failure: _leaf_phydrax_meshing__failures.MeshingFailure


@final
class _ContactFindings(StrictModule):
    rows: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    tested: int = eqx.field(static=True)


def _remote_contact_findings(
    points: NDArray[np.float64],
    ids: NDArray[np.int64],
    cell_ids: NDArray[np.int64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    remote_points: NDArray[np.float64],
    remote_ids: NDArray[np.int64],
    capacity: int,
    tested: int,
    maximum_candidate_pairs: int,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    /,
) -> _ContactFindings:
    findings: list[tuple[int, int, int]] = []
    for packet_index in range(remote_points.shape[0]):
        triangle = remote_points[packet_index, :, :]
        identities = remote_ids[packet_index, :]
        if identities[3] < 0:
            continue
        candidates = np.flatnonzero(
            np.all(lower <= np.max(triangle, axis=0), axis=1)
            & np.all(upper >= np.min(triangle, axis=0), axis=1)
        )
        tested += candidates.size
        if tested > maximum_candidate_pairs:
            findings.append((-1, -1, 2))
            continue
        for start in range(0, candidates.size, capacity):
            batch = candidates[start : start + capacity]
            joined_ids = np.concatenate((ids[batch].reshape(-1), identities[:3]))
            unique, inverse = np.unique(joined_ids, return_inverse=True)
            coordinates = np.empty((unique.size, 3), dtype=np.float64)
            coordinates[inverse] = np.concatenate(
                (points[batch].reshape((-1, 3)), triangle)
            )
            a = inverse[: 3 * batch.size].reshape((-1, 3))
            b = np.broadcast_to(inverse[-3:], a.shape)
            hit, certain = (
                _dependency_phydrax_meshing__audit_topology.triangle_pairs_intersect(
                    coordinates, a, b, budget=budget
                )
            )
            for cell, intersects, decided in zip(
                cell_ids[batch], hit, certain, strict=True
            ):
                if intersects or not decided:
                    findings.append((int(cell), int(identities[3]), 1 if decided else 2))
    return _ContactFindings(tuple(findings), tested)


@final
class _CollectiveSourcePayload(StrictModule):
    """Bind source declarations, physical controls, and retained collective content."""

    source: str = eqx.field(static=True)
    revision: str = eqx.field(static=True)
    generation: tuple[tuple[str, str], ...] = eqx.field(static=True)
    specification: str = eqx.field(static=True)
    limits: str = eqx.field(static=True)
    physical_angles: NDArray[np.float64]
    content: str = eqx.field(static=True)
    organization: str = eqx.field(static=True)

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "collective-initial-source-theorem",
                "source": self.source,
                "revision": self.revision,
                **dict(self.generation),
                "specification": self.specification,
                "limits": self.limits,
                "physical_angles": array_tree_fingerprint(self.physical_angles),
                "content": self.content,
                "organization": self.organization,
            }
        )


_INITIAL_ORGANIZATION_KEYS = (
    "initial/source_dimensions",
    "initial/source_indices",
    "initial/source_parameters",
    "initial/cell_patches",
    "initial/edge_curves",
)


@final
class _CornerSourceBinding(StrictModule):
    rows: NDArray[np.int64]


@final
class _CurveSourceBinding(StrictModule):
    rows: NDArray[np.int64]


@final
class _PatchSourceBinding(StrictModule):
    rows: NDArray[np.int64]


@final
class _SourceBindingRows(StrictModule):
    local_patches: np.ndarray
    vertex_count: int = eqx.field(static=True)


def _collective_source_payload(
    premise: InitialSurfacePremise | InitialPlanarPremise,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    minimum_angle: float,
    maximum_normal_angle: float,
    content_id: str,
    organization_id: str,
    /,
) -> _CollectiveSourcePayload:
    return _CollectiveSourcePayload(
        premise.source_id,
        premise.source_revision,
        tuple(premise.identity_fields().items()),
        premise.specification.specification_id,
        limits.limits_id,
        np.asarray((minimum_angle, maximum_normal_angle), dtype=np.float64),
        content_id,
        organization_id,
    )


def _source_binding_contract(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    /,
    *,
    partition_index: int | None = None,
) -> _SourceBindingRows | None:
    domain = prepared.compiled.domain
    target = prepared.specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 2
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"triangle"}
        or families.allowed_transitions
        or families.allow_mixed
    ):
        raise compliance_failure(
            "initial.target must declare affine surface triangles; got incompatible target semantics",
            stage="distributed-initial-certification",
        )
    if (
        generated.source_id != domain.source_id
        or generated.source_revision != domain.source_revision
        or generated.specification_id != prepared.specification.specification_id
        or prepared.compiled.specification_id != generated.specification_id
    ):
        raise compliance_failure(
            "initial.source_binding must match the source revision and specification; got different identities",
            stage="distributed-initial-certification",
        )
    construction = generated.construction
    part = jax.process_index() if partition_index is None else partition_index
    local_patches = prepared.compiled.patches[prepared.patch_owners == part]
    if construction is None:
        if local_patches.size or generated.vertex_ids.size or generated.cell_ids.size:
            raise compliance_failure(
                "initial.construction must contain the assigned patch rows; got missing construction",
                stage="distributed-initial-certification",
            )
        return None
    nv, nc = construction.vertices.shape[0], construction.triangles.shape[0]
    if (
        generated.vertex_ids.shape != (nv,)
        or generated.vertex_owners.shape != (nv,)
        or generated.cell_ids.shape != (nc,)
        or np.unique(generated.vertex_ids).size != nv
        or np.unique(generated.cell_ids).size != nc
        or np.any(generated.vertex_ids < 0)
        or np.any(generated.vertex_ids >= generated.global_vertex_count)
        or np.any(generated.cell_ids < 0)
        or np.any(generated.cell_ids >= generated.global_cell_count)
        or np.any(generated.vertex_owners < 0)
        or np.any(generated.vertex_owners >= prepared.partition_count)
        or construction.triangle_patches.shape != (nc,)
        or construction.triangle_parameters.shape != (nc, 3, 2)
        or not np.array_equal(np.unique(construction.triangle_patches), local_patches)
    ):
        raise compliance_failure(
            "initial.generated_rows must have unique in-range IDs and assigned patch incidence; got inconsistent rows",
            stage="distributed-initial-certification",
        )
    return _SourceBindingRows(local_patches, nv)


def _validate_source_charts(
    construction: surface_construction_owner.SurfaceConstruction,
    binding: _SourceBindingRows,
    /,
) -> _SourceBindingRows:
    local_patches = binding.local_patches
    for (
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
    ) in construction.chart_triangulations:
        if patch not in local_patches:
            raise compliance_failure(
                "initial.chart.patch must belong to this partition; got an unassigned patch",
                stage="distributed-initial-certification",
            )
        if (
            charts.ndim != 2
            or charts.shape[1] != 2
            or points.shape != (charts.shape[0], 3)
            or cells.ndim != 2
            or cells.shape[1] != 3
            or np.any(cells < 0)
            or np.any(cells >= charts.shape[0])
        ):
            raise compliance_failure(
                "initial.chart.incidence must align chart coordinates, points, and triangle indices; got invalid incidence",
                stage="distributed-initial-certification",
            )
        if (
            restriction_required.shape != (charts.shape[0],)
            or restriction_required.dtype != np.bool_
            or restriction_vertices.ndim != 1
            or restriction_edges.shape != (restriction_vertices.size, 2)
            or restriction_parameters.shape != (restriction_vertices.size, 2)
        ):
            raise compliance_failure(
                "initial.chart.restrictions must align vertices, edges, and parameters; got inconsistent restriction rows",
                stage="distributed-initial-certification",
            )
    return binding


def _curve_binding_coordinates(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    construction: surface_construction_owner.SurfaceConstruction,
    binding: _CurveSourceBinding,
    /,
) -> NDArray[np.float64]:
    rows = binding.rows
    indices = construction.vertex_source_indices
    parameters = construction.vertex_parameters
    normalized = np.empty((rows.size, 1), dtype=np.float64)
    for curve in np.unique(indices[rows]).tolist():
        selected = np.flatnonzero(indices[rows] == curve)
        owner_patch, owner_loop, owner_position = domain.curve_owners[curve]
        use = domain.patches[owner_patch].loops[owner_loop][owner_position]
        match use:
            case _dependency_phydrax_geometry__meshing_domain.PatchCurveUse():
                normalized[selected, 0] = (parameters[rows[selected], 0] - use.first) / (
                    use.last - use.first
                )
            case _:
                raise compliance_failure(
                    "initial.curve.use must be PatchCurveUse; got a different patch loop entry",
                    stage="distributed-initial-certification",
                )
    if np.any(~np.isfinite(normalized)) or np.any(
        (normalized <= 0.0) | (normalized >= 1.0)
    ):
        raise compliance_failure(
            "initial.curve.parameters must be finite and strictly interior; got invalid normalized parameters",
            stage="distributed-initial-certification",
        )
    return np.asarray(
        domain.curve_atlas.map(
            jnp.asarray(indices[rows], dtype=jnp.int32),
            jnp.asarray(normalized, dtype=jnp.float64),
        ),
        dtype=np.float64,
    ).reshape((-1, 3))


def _source_binding(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    /,
    *,
    partition_index: int | None = None,
) -> None:
    binding = _source_binding_contract(
        prepared, generated, partition_index=partition_index
    )
    construction = generated.construction
    if binding is None or construction is None:
        return
    binding = _validate_source_charts(construction, binding)
    domain = prepared.compiled.domain
    nv = binding.vertex_count
    local_patches = binding.local_patches
    dimensions = construction.vertex_source_dimensions
    indices = construction.vertex_source_indices
    parameters = construction.vertex_parameters
    if (
        dimensions.shape != (nv,)
        or indices.shape != (nv,)
        or parameters.shape != (nv, 2)
        or np.any(dimensions < 0)
        or np.any(dimensions > 2)
    ):
        raise compliance_failure(
            "initial.vertex.strata must align source dimensions, indices, and parameters; got inconsistent rows",
            stage="distributed-initial-certification",
        )
    strata = (
        _CornerSourceBinding(np.flatnonzero(dimensions == 0)),
        _CurveSourceBinding(np.flatnonzero(dimensions == 1)),
        _PatchSourceBinding(np.flatnonzero(dimensions == 2)),
    )
    bounds = (domain.corner_points.shape[0], len(domain.curves), len(domain.patches))
    for stratum, upper in zip(strata, bounds, strict=True):
        rows = stratum.rows
        if np.any(indices[rows] < 0) or np.any(indices[rows] >= upper):
            raise compliance_failure(
                "initial.vertex.source_indices must lie within the declared stratum; got out-of-range indices",
                stage="distributed-initial-certification",
            )
        match stratum:
            case _CornerSourceBinding():
                exact = domain.corner_points[indices[rows]]
            case _CurveSourceBinding():
                exact = _curve_binding_coordinates(domain, construction, stratum)
            case _PatchSourceBinding():
                if np.any(~np.isfinite(parameters[rows])) or not np.all(
                    np.isin(indices[rows], local_patches)
                ):
                    raise compliance_failure(
                        "initial.patch.vertices must have finite parameters on assigned patches; got invalid patch rows",
                        stage="distributed-initial-certification",
                    )
                exact = domain.evaluate(indices[rows], parameters[rows])
            case _:
                assert_never(stratum)
        if not np.allclose(
            construction.vertices[rows],
            exact,
            rtol=0.0,
            atol=128 * np.finfo(np.float64).eps * domain.scale,
        ):
            raise compliance_failure(
                "initial.vertex.coordinates must reproduce source-stratum coordinates; got discrepancies beyond the rounding bound",
                stage="distributed-initial-certification",
            )


def _physical_controls(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    construction: surface_construction_owner.SurfaceConstruction,
    minimum_angle: float,
    maximum_normal_angle: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Recompute physical quantities from source charts and generated coordinates."""
    compiled, domain = prepared.compiled, prepared.compiled.domain
    points = construction.vertices[construction.triangles]
    charts = construction.triangle_parameters
    first = np.roll(points, -1, axis=1) - points
    second = np.roll(points, -2, axis=1) - points
    squared_first = np.sum(first * first, axis=2)
    squared_second = np.sum(second * second, axis=2)
    product = np.sum(first * second, axis=2)
    metric = compiled.source_sizing.metric
    if metric is not None:
        values, _ = metric.sample(
            construction.vertices,
            prepared.specification.limits.maximum_work_units,
            budget=budget,
        )
        edges = construction.triangles[:, np.asarray(((0, 1), (1, 2), (2, 0)))].reshape(
            (-1, 2)
        )
        bound = metric.control.maximum_metric_edge_length or 1.0
        if np.any(
            np.asarray(
                _dependency_phydrax_meshing__metric.metric_edge_lengths(
                    values, construction.vertices, edges
                )
            )
            > bound
        ):
            raise compliance_failure(
                "initial.metric.edges must satisfy the physical metric bound; got overlong edges",
                stage="distributed-initial-certification",
            )
        mean = np.asarray(
            _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
                values[construction.triangles],
                np.full(construction.triangles.shape, 1.0 / 3.0, dtype=np.float64),
            )
        )
        squared_first = np.sum(first * (mean[:, None] @ first[..., None])[..., 0], axis=2)
        squared_second = np.sum(
            second * (mean[:, None] @ second[..., None])[..., 0], axis=2
        )
        product = np.sum(first * (mean[:, None] @ second[..., None])[..., 0], axis=2)
    cosine = product / np.sqrt(
        np.maximum(squared_first * squared_second, np.finfo(np.float64).tiny)
    )
    if np.any(
        np.arccos(np.clip(cosine, -1.0, 1.0))
        < minimum_angle - 256 * np.finfo(np.float64).eps
    ):
        raise compliance_failure(
            "initial.minimum_angle must satisfy the physical angle control; got smaller angles",
            stage="distributed-initial-certification",
        )
    for patch in np.unique(construction.triangle_patches).tolist():
        rows = np.flatnonzero(construction.triangle_patches == patch)
        uv = charts[rows]
        center = np.mean(uv, axis=1)
        source_center = domain.evaluate(np.full(rows.size, patch, dtype=np.int64), center)
        sizes = compiled.source_sizing.evaluate(
            domain, patch, center, source_center, budget=budget
        )
        midpoint_uv = 0.5 * (np.roll(uv, -1, axis=1) + np.roll(uv, -2, axis=1))
        midpoint = domain.evaluate(
            np.full(3 * rows.size, patch, dtype=np.int64), midpoint_uv.reshape((-1, 2))
        ).reshape((-1, 3, 3))
        ends_a, ends_b = (
            np.roll(points[rows], -1, axis=1),
            np.roll(points[rows], -2, axis=1),
        )
        lengths = np.linalg.norm(midpoint - ends_a, axis=2) + np.linalg.norm(
            midpoint - ends_b, axis=2
        )
        if np.any(lengths > sizes[:, None] * (1.0 + 1.0e-9)):
            raise compliance_failure(
                "initial.sizing.edges must satisfy source sizing controls; got overlong edges",
                stage="distributed-initial-certification",
            )
        normals, regular = domain.oriented_normals(
            np.full(rows.size, patch, dtype=np.int64), center
        )
        face = np.cross(
            points[rows, 1] - points[rows, 0], points[rows, 2] - points[rows, 0]
        )
        if np.any(~regular) or np.any(np.sum(face * normals, axis=1) <= 0.0):
            raise compliance_failure(
                "initial.cells.orientation must agree with regular source normals; got irregular or reversed cells",
                stage="distributed-initial-certification",
            )
        requested = min(
            maximum_normal_angle,
            float(compiled.patch_normal_angles[np.searchsorted(compiled.patches, patch)]),
        )
        if np.isfinite(requested) and np.any(
            domain.normal_turn_bounds(patch, uv) > requested
        ):
            raise compliance_failure(
                "initial.normal_turn must satisfy the continuous angle control; got excessive turning",
                stage="distributed-initial-certification",
            )


def _source_feature_controls(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    construction: surface_construction_owner.SurfaceConstruction,
    /,
) -> None:
    """Certify the native parameter chains of source curves."""
    compiled, domain = prepared.compiled, prepared.compiled.domain
    for curve, parameters in construction.curve_parameters:
        if (
            parameters.ndim != 1
            or parameters.size < 2
            or parameters[0] != 0.0
            or parameters[-1] != 1.0
            or np.any(~np.isfinite(parameters))
            or np.any(np.diff(parameters) <= 0.0)
        ):
            raise compliance_failure(
                "initial.feature.parameters must cover the declared curve interval in order; got invalid chains",
                stage="distributed-initial-certification",
            )
        rows = np.flatnonzero(construction.curve_edge_curves == curve)
        if rows.size != parameters.size - 1:
            raise compliance_failure(
                "initial.feature.edges must contain one edge per parameter interval; got inconsistent edge counts",
                stage="distributed-initial-certification",
            )
        points = np.array(
            domain.curve_atlas.map(
                jnp.full((parameters.size,), curve, dtype=jnp.int32),
                jnp.asarray(parameters[:, None], dtype=jnp.float64),
            ),
            dtype=np.float64,
            copy=True,
        ).reshape((-1, 3))
        source = domain.curves[curve]
        points[0], points[-1] = (
            domain.corner_points[source.start],
            domain.corner_points[source.end],
        )
        expected = np.stack((points[:-1], points[1:]), axis=1)
        actual = construction.vertices[construction.curve_edges[rows]]
        if not np.allclose(
            actual, expected, rtol=0.0, atol=128 * np.finfo(np.float64).eps * domain.scale
        ):
            raise compliance_failure(
                "initial.feature.incidence must reproduce the source curve chain; got different endpoint coordinates",
                stage="distributed-initial-certification",
            )
        row = np.searchsorted(compiled.curves, curve)
        if np.any(
            np.linalg.norm(actual[:, 1] - actual[:, 0], axis=1)
            > compiled.curve_sizes[row] * (1.0 + 1.0e-9)
        ):
            raise compliance_failure(
                "initial.feature.sizing must satisfy the physical source control; got overlong feature edges",
                stage="distributed-initial-certification",
            )
        if np.any(
            domain.curve_interpolation_bounds(curve, parameters)
            > compiled.curve_deviations[row]
        ):
            raise compliance_failure(
                "initial.feature.fidelity must satisfy the requested interpolation bound; got excessive curve deviation",
                stage="distributed-initial-certification",
            )


def _local_proof(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    minimum_angle: float,
    maximum_normal_angle: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    partition_index: int | None = None,
) -> _LocalProof:
    _source_binding(prepared, generated, partition_index=partition_index)
    construction = generated.construction
    patch_checks = np.zeros(prepared.compiled.patches.shape, dtype=np.bool_)
    if construction is None:
        return _LocalProof((), patch_checks)
    mesh = CellMesh(
        construction.vertices,
        (
            CellBlock(
                "surface",
                "triangle",
                construction.triangles,
                global_ids=generated.cell_ids,
            ),
        ),
        vertex_global_ids=generated.vertex_ids,
    )
    geometry = CellGeometrySpec.affine(mesh, budget=budget)
    validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
        geometry, mesh=mesh, budget=budget
    )
    embedding = global_embedding_owner.certify_global_embedding(
        mesh, geometry, validity, limits=limits, budget=budget
    )
    if embedding.status != "certified":
        raise compliance_failure(
            "initial.owner.embedding must be certified; got violated or unresolved geometry predicates",
            stage="distributed-initial-certification",
        )
    certificates = [validity.certificate_id, embedding.certificate_id]
    for patch in np.unique(construction.triangle_patches).tolist():
        rows = np.flatnonzero(construction.triangle_patches == patch)
        patch_mesh = CellMesh(
            construction.vertices,
            (
                CellBlock(
                    "surface",
                    "triangle",
                    construction.triangles[rows],
                    global_ids=generated.cell_ids[rows],
                ),
            ),
            vertex_global_ids=generated.vertex_ids,
        )
        patch_geometry = CellGeometrySpec.affine(patch_mesh, budget=budget)
        source = domain_boundary_owner.MeshingDomainBoundarySource(
            prepared.compiled.domain,
            (patch,),
            chart_triangulations=tuple(
                value for value in construction.chart_triangulations if value[0] == patch
            ),
        )
        cover = source.boundary_chart_cover(limits.maximum_source_samples, budget=budget)
        if not cover.complete or cover.semantics != "certified":
            raise compliance_failure(
                "initial.chart.coverage must certify the declared source trims; got uncertified or missing coverage",
                stage="distributed-initial-certification",
            )
        tolerance = float(
            prepared.compiled.patch_deviations[
                np.searchsorted(prepared.compiled.patches, patch)
            ]
        )
        # An unbounded request still requires a finite independently established
        # continuous enclosure, not an infinite or sampled success criterion.
        if not np.isfinite(tolerance):
            tolerance = (
                float(np.max(cover.deviation_bounds, initial=0.0))
                + 1024 * np.finfo(np.float64).eps * prepared.compiled.domain.scale
            )
        fidelity = source_fidelity_owner.certify_source_fidelity(
            patch_mesh,
            patch_geometry,
            source,
            tolerance=tolerance,
            limits=limits,
            budget=budget,
        )
        if fidelity.status != "certified":
            raise compliance_failure(
                "initial.cells.fidelity must certify two-sided source fidelity; got an uncertified bound",
                stage="distributed-initial-certification",
            )
        certificates.extend((cover.cover_id, fidelity.certificate_id))
        patch_checks[np.searchsorted(prepared.compiled.patches, patch)] = True
    _source_feature_controls(prepared, construction)
    _physical_controls(
        prepared, construction, minimum_angle, maximum_normal_angle, budget=budget
    )
    return _LocalProof(tuple(certificates), patch_checks)


@dataclass(frozen=True, slots=True)
class _InitialOwnerRows:
    """One owner's generated rows in the premise-neutral collective layout.

    ``vertex_banks`` become ``initial/{name}`` vertex banks, ``cell_banks``
    ``initial/cell_{name}`` and ``curve_banks`` ``initial/edge_{name}``. Curve
    edges carry sorted global vertex IDs, their oriented ``curve_along`` rows and
    source curve/edge values; ``curve_uses[c]`` is the declared facet incidence
    of source curve/edge ``c``. ``work_checks`` mark the generation units this
    owner certified (patches or tiles).
    """

    coordinates: NDArray[np.float64]
    vertex_ids: NDArray[np.int64]
    vertex_owners: NDArray[np.int32]
    corners: NDArray[np.int64]
    cell_ids: NDArray[np.int64]
    vertex_banks: tuple[tuple[str, np.ndarray, float | int], ...]
    cell_banks: tuple[tuple[str, np.ndarray, float | int], ...]
    curve_edges: NDArray[np.int64]
    curve_along: NDArray[np.int64]
    curve_values: NDArray[np.int64]
    curve_banks: tuple[tuple[str, NDArray[np.float64]], ...]
    curve_uses: NDArray[np.int32]
    work_checks: NDArray[np.bool_]
    global_vertex_count: int
    global_cell_count: int


def _rows_or_empty(
    value: np.ndarray | None, shape: tuple[int, ...], dtype: type, /
) -> np.ndarray:
    return np.empty(shape, dtype=dtype) if value is None else np.asarray(value, dtype=dtype)


def _surface_owner_rows(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    local: _LocalProof,
    /,
) -> _InitialOwnerRows:
    """Surface chart, trim and curve facts of one owner's generated rows."""
    construction = generated.construction
    present = construction is not None

    def field(name: str) -> np.ndarray | None:
        return None if construction is None else getattr(construction, name)

    uses = np.asarray(
        [
            sum(
                patch in prepared.compiled.patches
                for patch, _ in prepared.compiled.domain.curve_uses(curve)
            )
            for curve in range(len(prepared.compiled.domain.curves))
        ],
        dtype=np.int32,
    )
    return _InitialOwnerRows(
        coordinates=_rows_or_empty(field("vertices"), (0, 3), np.float64),
        vertex_ids=generated.vertex_ids,
        vertex_owners=generated.vertex_owners,
        corners=(
            generated.vertex_ids[construction.triangles]
            if present
            else np.empty((0, 3), dtype=np.int64)
        ),
        cell_ids=generated.cell_ids,
        vertex_banks=(
            (
                "source_dimensions",
                _rows_or_empty(field("vertex_source_dimensions"), (0,), np.int64),
                -1,
            ),
            (
                "source_indices",
                _rows_or_empty(field("vertex_source_indices"), (0,), np.int64),
                -1,
            ),
            (
                "source_parameters",
                _rows_or_empty(field("vertex_parameters"), (0, 2), np.float64),
                0,
            ),
        ),
        cell_banks=(
            ("patches", _rows_or_empty(field("triangle_patches"), (0,), np.int64), -1),
            ("charts", _rows_or_empty(field("triangle_charts"), (0, 2), np.float64), 0),
            (
                "parameters",
                _rows_or_empty(field("triangle_parameters"), (0, 3, 2), np.float64),
                0,
            ),
            (
                "deviations",
                _rows_or_empty(field("triangle_deviations"), (0,), np.float64),
                0,
            ),
            (
                "deviation_bounds",
                _rows_or_empty(field("triangle_deviation_bounds"), (0,), np.float64),
                0,
            ),
            (
                "normal_bounds",
                _rows_or_empty(field("triangle_normal_bounds"), (0,), np.float64),
                0,
            ),
            (
                "boundary_occurrences",
                _rows_or_empty(field("boundary_occurrences"), (0, 3, 3), np.int32),
                -1,
            ),
        ),
        curve_edges=(
            np.sort(generated.vertex_ids[construction.curve_edges], axis=1)
            if present
            else np.empty((0, 2), dtype=np.int64)
        ),
        curve_along=(
            generated.vertex_ids[construction.curve_edges]
            if present
            else np.empty((0, 2), dtype=np.int64)
        ),
        curve_values=_rows_or_empty(field("curve_edge_curves"), (0,), np.int64),
        curve_banks=(
            (
                "deviations",
                _rows_or_empty(field("curve_edge_deviations"), (0,), np.float64),
            ),
            (
                "deviation_bounds",
                _rows_or_empty(field("curve_edge_deviation_bounds"), (0,), np.float64),
            ),
        ),
        curve_uses=uses,
        work_checks=local.patch_checks,
        global_vertex_count=generated.global_vertex_count,
        global_cell_count=generated.global_cell_count,
    )


@jax.jit
def _initial_content_values(
    raw: dict[str, Array],
    vertex_banks: dict[str, Array],
    cell_banks: dict[str, Array],
    curve_banks: dict[str, Array],
    /,
) -> tuple[
    dict[str, Array],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
    Array,
    Array,
]:
    """Fuse source-stratum joins on placed arrays, without host mesh assembly."""
    vertex_ids, vertex_owners = raw["vertex_ids"], raw["vertex_owners"]
    raw_coordinates = raw["coordinates"]
    cell_ids, raw_corners = raw["cell_ids"], raw["corners"]
    parts = raw["work_checks"].shape[0]
    vertex_capacity = vertex_ids.shape[0] // parts
    cell_capacity = cell_ids.shape[0] // parts
    ranks = jnp.repeat(jnp.arange(parts, dtype=jnp.int32), cell_capacity)
    cell_keys, cell_order, nc = (
        _dependency_phydrax_meshing__device_adaptation._canonical_row_values(
            cell_ids[:, None], cell_ids >= 0
        )
    )
    vertex_keys, vertex_order, nv = (
        _dependency_phydrax_meshing__device_adaptation._canonical_row_values(
            vertex_ids[:, None], vertex_ids >= 0
        )
    )
    ordered = jnp.argsort(
        jnp.where(vertex_ids >= 0, vertex_ids, jnp.iinfo(jnp.int64).max), stable=True
    )
    duplicate = (vertex_ids[ordered[1:]] >= 0) & (
        vertex_ids[ordered[1:]] == vertex_ids[ordered[:-1]]
    )
    shared = jnp.all(
        ~duplicate[:, None]
        | (raw_coordinates[ordered[1:]] == raw_coordinates[ordered[:-1]])
    )
    shared &= jnp.all(
        ~duplicate | (vertex_owners[ordered[1:]] == vertex_owners[ordered[:-1]])
    )
    owned_vertex = vertex_ids >= 0
    owned_vertex &= vertex_owners == jnp.repeat(
        jnp.arange(parts, dtype=jnp.int32), vertex_capacity
    )
    unique_ownership = jnp.sum(owned_vertex) == nv
    unique_ownership &= (
        (jnp.sum(cell_ids >= 0) == nc)
        & (nc == raw["global_counts"][1])
        & (nv == raw["global_counts"][0])
    )
    canonical_corners = jnp.where(
        (jnp.arange(cell_keys.shape[0]) < nc)[:, None], raw_corners[cell_order], -1
    )
    canonical_coordinates = jnp.where(
        (jnp.arange(vertex_keys.shape[0]) < nv)[:, None],
        raw_coordinates[vertex_order],
        0.0,
    )
    arrays = {
        "cell_global_ids": cell_keys[:, 0],
        "cell_vertices": canonical_corners,
        "vertex_global_ids": vertex_keys[:, 0],
        "coordinates": canonical_coordinates,
        "cell_owners": ranks[cell_order],
    }
    for name, bank in vertex_banks.items():
        arrays[f"initial/{name}"] = bank[vertex_order]
        equal = bank[ordered[1:]] == bank[ordered[:-1]]
        if jnp.issubdtype(bank.dtype, jnp.floating):
            equal |= jnp.isnan(bank[ordered[1:]]) & jnp.isnan(bank[ordered[:-1]])
        if equal.ndim == 2:
            equal = jnp.all(equal, axis=1)
        shared &= jnp.all(~duplicate | equal)
    for name, bank in cell_banks.items():
        arrays[f"initial/cell_{name}"] = bank[cell_order]
    edge_columns = np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int32)
    edge_rows = jnp.sort(canonical_corners[:, edge_columns], axis=2).reshape((-1, 2))
    edge_valid = jnp.repeat(jnp.arange(cell_keys.shape[0]) < nc, 3)
    edge_keys, edge_order, ne = (
        _dependency_phydrax_meshing__device_adaptation._canonical_row_values(
            edge_rows, edge_valid
        )
    )
    edge_ids = jnp.where(
        jnp.arange(edge_keys.shape[0]) < ne,
        jnp.arange(edge_keys.shape[0], dtype=jnp.int64),
        -1,
    )
    edge_owners = jnp.repeat(ranks[cell_order], 3)[edge_order]
    positions = entity_keys_owner.key_positions(
        jnp.where(edge_keys >= 0, edge_keys, jnp.iinfo(jnp.int64).max), edge_rows
    )
    incidence = (
        jnp.zeros(edge_ids.shape, dtype=jnp.int32)
        .at[jnp.maximum(positions, 0)]
        .add(edge_valid.astype(jnp.int32))
    )
    directed = canonical_corners[:, edge_columns].reshape((-1, 2))
    sign = jnp.where(directed[:, 0] < directed[:, 1], 1, -1) * jnp.tile(
        jnp.asarray((1, -1, 1), dtype=jnp.int32), cell_keys.shape[0]
    )
    orientation = (
        jnp.zeros(edge_ids.shape, dtype=jnp.int32)
        .at[jnp.maximum(positions, 0)]
        .add(jnp.where(edge_valid, sign, 0))
    )
    curve_keys, curve_values = raw["curve_edges"], raw["curve_values"]
    curve_table, curve_order, curve_count = (
        _dependency_phydrax_meshing__device_adaptation._canonical_row_values(
            curve_keys, curve_values >= 0
        )
    )
    curve_positions = entity_keys_owner.key_positions(
        jnp.where(curve_table >= 0, curve_table, jnp.iinfo(jnp.int64).max), edge_keys
    )
    edge_curves = jnp.where(
        curve_positions >= 0,
        curve_values[curve_order[jnp.maximum(curve_positions, 0)]],
        -1,
    )
    expected = jnp.where(
        edge_curves >= 0,
        raw["curve_uses"][jnp.maximum(edge_curves, 0)],
        2,
    )
    reciprocal = jnp.all((jnp.arange(edge_ids.shape[0]) >= ne) | (incidence == expected))
    curve_rows = entity_keys_owner.key_positions(
        jnp.where(edge_keys >= 0, edge_keys, jnp.iinfo(jnp.int64).max), curve_table
    )
    reciprocal &= jnp.all(
        (jnp.arange(curve_table.shape[0]) >= curve_count) | (curve_rows >= 0)
    )
    reciprocal &= jnp.all((incidence != 2) | (orientation == 0))
    arrays.update(
        {
            "entity_global_ids_1": edge_ids,
            "entity_vertices_1": edge_keys,
            "initial/edge_curves": edge_curves,
            "initial/edge_physical": incidence == 1,
        }
    )
    for name, bank in curve_banks.items():
        arrays[f"initial/edge_{name}"] = bank[
            curve_order[jnp.maximum(curve_positions, 0)]
        ]
    arrays["initial/edge_along"] = raw["curve_along"][
        curve_order[jnp.maximum(curve_positions, 0)]
    ]
    keys, identifiers, owners = (
        (vertex_keys, edge_keys, cell_keys),
        (vertex_keys[:, 0], edge_ids, cell_keys[:, 0]),
        (vertex_owners[vertex_order], edge_owners, ranks[cell_order]),
    )
    counts = (nv, ne, nc)
    for degree, count in enumerate(counts):
        arrays[f"entity_active_{degree}"] = jnp.arange(keys[degree].shape[0]) < count
    arrays["source_basis"] = jnp.where(
        (jnp.arange(cell_keys.shape[0]) < nc)[:, None],
        jnp.broadcast_to(raw["source_basis"], (cell_keys.shape[0], 32)),
        0,
    )
    covered = jnp.all(jnp.sum(raw["work_checks"].astype(jnp.int32), axis=0) == 1)
    checks = raw["checks"]
    checks = (
        checks.at[:, 1]
        .set(covered)
        .at[:, 5]
        .set(unique_ownership)
        .at[:, 6]
        .set(shared)
        .at[:, 7]
        .set(reciprocal)
    )
    return arrays, keys, identifiers, owners, jnp.stack(counts), checks


def _collective_initial_content(
    owner_rows: _InitialOwnerRows,
    sharding: NamedSharding,
    /,
    *,
    vertex_capacity: int,
    cell_capacity: int,
    block: str,
    width: int,
) -> tuple[
    dict[str, Array],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[int, ...],
    str,
    str,
    Array,
]:
    """Place owning-host rows once, then fuse joins before the unchanged hashes."""
    element = _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
        "triangle", 1
    )
    basis_id = canonical_fingerprint(
        {
            "signature": _owner_phydrax_discretization__coordinate_enclosure.coordinate_source_signature(
                element
            ),
            "arrays": array_tree_fingerprint(element),
        }
    )
    parts = jax.process_count()
    raw = {
        "vertex_ids": _global_rows(owner_rows.vertex_ids, vertex_capacity, sharding, -1),
        "vertex_owners": _global_rows(
            owner_rows.vertex_owners, vertex_capacity, sharding, -1
        ),
        "coordinates": _global_rows(owner_rows.coordinates, vertex_capacity, sharding),
        "cell_ids": _global_rows(owner_rows.cell_ids, cell_capacity, sharding, -1),
        "corners": _global_rows(owner_rows.corners, cell_capacity, sharding, -1),
        "curve_edges": _global_rows(
            owner_rows.curve_edges, 3 * cell_capacity, sharding, -1
        ),
        "curve_values": _global_rows(
            owner_rows.curve_values, 3 * cell_capacity, sharding, -1
        ),
        "curve_along": _global_rows(
            owner_rows.curve_along, 3 * cell_capacity, sharding, -1
        ),
        "curve_uses": jax.device_put(owner_rows.curve_uses),
        "work_checks": jax.make_array_from_process_local_data(
            sharding,
            owner_rows.work_checks[None],
            (parts, owner_rows.work_checks.size),
        ),
        "checks": jax.make_array_from_process_local_data(
            sharding,
            np.ones((1, len(_SURFACE_CHECKS)), dtype=np.bool_),
            (parts, len(_SURFACE_CHECKS)),
        ),
        "global_counts": jax.device_put(
            np.asarray(
                (owner_rows.global_vertex_count, owner_rows.global_cell_count),
                dtype=np.int64,
            )
        ),
        "source_basis": jax.device_put(
            np.frombuffer(bytes.fromhex(basis_id), dtype=np.uint8)
        ),
    }
    vertex_banks = {
        name: _global_rows(value, vertex_capacity, sharding, fill)
        for name, value, fill in owner_rows.vertex_banks
    }
    cell_banks = {
        name: _global_rows(value, cell_capacity, sharding, fill)
        for name, value, fill in owner_rows.cell_banks
    }
    curve_banks = {
        name: _global_rows(value, 3 * cell_capacity, sharding)
        for name, value in owner_rows.curve_banks
    }
    arrays, keys, identifiers, owners, count_values, checks = _initial_content_values(
        raw, vertex_banks, cell_banks, curve_banks
    )
    counts = tuple(int(value) for value in np.asarray(jax.device_get(count_values)))
    nv, ne, nc = counts
    canonical_corners = arrays["cell_vertices"]
    canonical_coordinates = arrays["coordinates"]
    shapes = {
        "cell_global_ids": (nc,),
        "cell_vertices": (nc, 3),
        "vertex_global_ids": (nv,),
        "entity_global_ids_1": (ne,),
        "entity_vertices_1": (ne, 2),
    }
    topology_id = canonical_fingerprint(
        {
            "kind": "cell-mesh-topology",
            "dimension": 2,
            "blocks": [(block, "triangle")],
            "arrays": logical_array_value_collection_digest(
                {name: arrays[name] for name in shapes}, logical_shapes=shapes
            ),
        }
    )
    geometry_id = canonical_fingerprint(
        {
            "kind": "cell-mesh-geometry",
            "topology": topology_id,
            "ambient_dimension": width,
            "coordinates": logical_array_value_collection_digest(
                {"coordinates": canonical_coordinates},
                logical_shapes={"coordinates": (nv, width)},
            ),
        }
    )
    arrays.update(
        {
            "geometry/coordinate_ids": identifiers[0],
            "geometry/coordinates": canonical_coordinates,
            "geometry/coordinate_owners": owners[0],
            f"geometry/cell_ids/{block}": identifiers[2],
            f"geometry/routes/{block}": canonical_corners,
            f"geometry/source_basis/{block}": arrays.pop("source_basis"),
        }
    )
    return arrays, keys, identifiers, owners, counts, topology_id, geometry_id, checks


def _initial_coordinate_geometry_id(
    arrays: dict[str, Array],
    counts: tuple[int, ...],
    topology_id: str,
    premise: InitialSurfacePremise | InitialPlanarPremise,
    /,
) -> str:
    """Bind the affine coordinate basis, coefficient IDs, values, and routes."""
    block = premise.block_name
    shapes = {
        "geometry/coordinate_ids": (counts[0],),
        "geometry/coordinates": (counts[0], premise.ambient_dimension),
        f"geometry/cell_ids/{block}": (counts[2],),
        f"geometry/routes/{block}": (counts[2], 3),
        f"geometry/source_basis/{block}": (counts[2], 32),
    }
    digest = logical_array_value_collection_digest(
        {name: arrays[name] for name in shapes}, logical_shapes=shapes
    )
    return canonical_fingerprint(
        {
            "kind": "collective-initial-coordinate-geometry",
            "topology": topology_id,
            "arrays": digest,
        }
    )


@partial(
    jax.jit,
    static_argnames=("vertex_count", "cell_count", "capacity", "parts", "neighborhood_depth"),
)
def _initial_closures(
    arrays: dict[str, Array],
    *,
    vertex_count: int,
    cell_count: int,
    capacity: int,
    parts: int,
    neighborhood_depth: int,
) -> tuple[dict[str, Array], Array]:
    """Earn the requested whole-star reach, retaining bounded owner-local packets."""
    ids, corners = arrays["cell_global_ids"], arrays["cell_vertices"]
    table = jnp.where(
        jnp.arange(arrays["vertex_global_ids"].shape[0]) < vertex_count,
        arrays["vertex_global_ids"],
        jnp.iinfo(jnp.int64).max,
    )
    rows = jnp.searchsorted(table, corners)
    active = jnp.arange(ids.shape[0]) < cell_count

    def one(part: Array) -> tuple[Array, Array, Array]:
        own_cells = active & (arrays["cell_owners"] == part)
        def expand(_: int, selected: Array) -> Array:
            vertices = (
                jnp.zeros(table.shape, dtype=jnp.bool_)
                .at[jnp.minimum(rows, table.shape[0] - 1)]
                .max(selected[:, None])
            )
            return active & jnp.any(
                vertices[jnp.minimum(rows, table.shape[0] - 1)], axis=1
            )

        # A finite cell graph reaches its fixed point within cell_count steps.
        # This cap preserves every requested layer without unbounded empty work.
        selected = jax.lax.fori_loop(
            0, min(neighborhood_depth, cell_count), expand, own_cells
        )
        count = jnp.sum(selected, dtype=jnp.int32)
        order = jnp.nonzero(selected, size=capacity, fill_value=0)[0]
        return order, jnp.arange(capacity) < count, count <= capacity

    order, valid, fits = jax.vmap(one)(jnp.arange(parts, dtype=jnp.int32))
    packets = {
        "cell_ids": jnp.where(valid, ids[order], -1),
        "cell_vertices": jnp.where(valid[..., None], corners[order], -1),
        "cell_valid": valid,
        "cell_owner": arrays["cell_owners"][order],
        "cell_coordinates": arrays["coordinates"][
            jnp.minimum(rows[order], table.shape[0] - 1)
        ],
    }
    return {f"closure/{name}": value for name, value in packets.items()}, jnp.all(fits)


_INITIAL_INPUT_FIELDS = (
    ("vertices", np.float64, (3,)),
    ("vertex_source_dimensions", np.int64, ()),
    ("vertex_source_indices", np.int64, ()),
    ("vertex_parameters", np.float64, (2,)),
    ("chart_restriction_vertices", np.int64, ()),
    ("chart_restriction_edges", np.int64, (2,)),
    ("chart_restriction_endpoint_parameters", np.float64, (2, 2)),
    ("chart_restriction_parameters", np.int64, (2,)),
    ("triangles", np.int64, (3,)),
    ("triangle_patches", np.int64, ()),
    ("triangle_charts", np.float64, (2,)),
    ("triangle_parameters", np.float64, (3, 2)),
    ("triangle_deviations", np.float64, ()),
    ("triangle_deviation_bounds", np.float64, ()),
    ("triangle_normal_bounds", np.float64, ()),
    ("curve_edges", np.int64, (2,)),
    ("curve_edge_curves", np.int64, ()),
    ("curve_edge_deviations", np.float64, ()),
    ("curve_edge_deviation_bounds", np.float64, ()),
    ("unresolved", np.bool_, ()),
    ("boundary_occurrences", np.int32, (3, 3)),
)


def _initial_input_values(
    construction: surface_construction_owner.SurfaceConstruction | None,
    /,
) -> tuple[np.ndarray, ...]:
    """Return replay inputs in the retained field order with explicit source access."""
    if construction is None:
        return tuple(
            np.empty((0, *tail), dtype=dtype) for _, dtype, tail in _INITIAL_INPUT_FIELDS
        )
    values = (
        construction.vertices,
        construction.vertex_source_dimensions,
        construction.vertex_source_indices,
        construction.vertex_parameters,
        construction.chart_restriction_vertices,
        construction.chart_restriction_edges,
        construction.chart_restriction_endpoint_parameters,
        construction.chart_restriction_parameters,
        construction.triangles,
        construction.triangle_patches,
        construction.triangle_charts,
        construction.triangle_parameters,
        construction.triangle_deviations,
        construction.triangle_deviation_bounds,
        construction.triangle_normal_bounds,
        construction.curve_edges,
        construction.curve_edge_curves,
        construction.curve_edge_deviations,
        construction.curve_edge_deviation_bounds,
        construction.unresolved,
        construction.boundary_occurrences,
    )
    return tuple(
        np.asarray(value, dtype=dtype)
        for value, (_, dtype, _) in zip(values, _INITIAL_INPUT_FIELDS, strict=True)
    )


def _retain_initial_inputs(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    sharding: NamedSharding,
    /,
) -> dict[str, Array]:
    """Retain native chart and trim inputs for cold replay."""
    construction = generated.construction
    parts = prepared.partition_count
    result: dict[str, Array] = {}

    def retain(name: str, value: np.ndarray) -> None:
        counts = np.asarray(
            multihost_utils.process_allgather(np.int64(value.shape[0]), tiled=False)
        )
        capacity = max(1, int(np.max(counts)))
        result[f"initial/input/{name}"] = _global_rows(value, capacity, sharding).reshape(
            (parts, capacity, *value.shape[1:])
        )
        result[f"initial/input_count/{name}"] = jax.make_array_from_process_local_data(
            sharding, np.asarray((value.shape[0],), dtype=np.int64), (parts,)
        )

    for (name, _, _), value in zip(
        _INITIAL_INPUT_FIELDS, _initial_input_values(construction), strict=True
    ):
        retain(name, value)
    retain("vertex_ids", generated.vertex_ids)
    retain("vertex_owners", generated.vertex_owners)
    retain("cell_ids", generated.cell_ids)
    retain("shared_curve_ids", generated.shared_curve_ids)
    charts = (
        {}
        if construction is None
        else {row[0]: row[1:] for row in construction.chart_triangulations}
    )
    for patch in prepared.compiled.patches.tolist():
        for index, (name, dtype, tail) in enumerate(
            (
                ("charts", np.float64, (2,)),
                ("points", np.float64, (3,)),
                ("cells", np.int64, (3,)),
                ("boundary", np.int64, (2,)),
                ("provenance", np.float64, (4,)),
                ("restriction_required", np.bool_, ()),
                ("restriction_vertices", np.int64, ()),
                ("restriction_edges", np.int64, (2,)),
                ("restriction_parameters", np.int64, (2,)),
            )
        ):
            value = (
                np.empty((0, *tail), dtype=dtype)
                if patch not in charts
                else np.asarray(charts[patch][index], dtype=dtype)
            )
            retain(f"chart/{patch}/{name}", value)
    curves = {} if construction is None else dict(construction.curve_parameters)
    for curve in prepared.compiled.curves.tolist():
        retain(f"curve/{curve}", curves.get(curve, np.empty((0,), dtype=np.float64)))
    integers = np.asarray(
        (
            0 if construction is None else construction.rounds,
            0 if construction is None else construction.inserted,
            0 if construction is None else construction.refused,
            0 if construction is None else construction.flips,
            0 if construction is None else construction.geometry_queries,
            0 if construction is None else construction.work_units,
            0 if construction is None else construction.incomplete_refinement,
            generated.neighbor_packet_count,
            generated.neighbor_transferred_bytes,
            generated.metadata_bytes,
        ),
        dtype=np.int64,
    )
    result["initial/input_metadata"] = jax.make_array_from_process_local_data(
        sharding, integers[None], (parts, integers.size)
    )
    result["initial/input_minimum_angle"] = jax.make_array_from_process_local_data(
        sharding,
        np.asarray(
            (np.inf if construction is None else construction.minimum_angle,),
            dtype=np.float64,
        ),
        (parts,),
    )
    result["initial/input_present"] = jax.make_array_from_process_local_data(
        sharding, np.asarray((construction is not None,), dtype=np.bool_), (parts,)
    )
    return result


def _placement() -> NamedSharding:
    mesh = owner_execution_mesh(process_owner_devices())
    return NamedSharding(mesh, PartitionSpec(mesh.axis_names[0]))


def _global_rows(
    local: np.ndarray,
    capacity: int,
    sharding: NamedSharding,
    fill: int | float | bool = 0,
    /,
) -> Array:
    if local.shape[0] > capacity:
        raise compliance_failure(
            "initial.rows must fit the reserved capacity; got excess rows",
            stage="distributed-initial-certification",
        )
    padded = np.full((capacity, *local.shape[1:]), fill, dtype=local.dtype)
    padded[: local.shape[0]] = local
    return jax.make_array_from_process_local_data(
        sharding, padded, (jax.process_count() * capacity, *local.shape[1:])
    )


def _exchange(
    local: np.ndarray, first: int, second: int, sharding: NamedSharding, /
) -> np.ndarray:
    """One bounded candidate packet, with no global target host materialization."""
    data = jax.make_array_from_process_local_data(
        sharding, local[None], (jax.process_count(), *local.shape)
    )
    permutation = tuple(
        (rank, second if rank == first else first if rank == second else rank)
        for rank in range(jax.process_count())
    )
    axis = sharding.mesh.axis_names[0]

    @partial(
        jax.shard_map,
        mesh=sharding.mesh,
        in_specs=PartitionSpec(axis),
        out_specs=PartitionSpec(axis),
        check_vma=False,
    )
    def send(value: Array) -> Array:
        return jax.lax.ppermute(value, axis, permutation)

    received = jax.jit(send)(data)
    return np.asarray(jax.device_get(received.addressable_shards[0].data))[0]


def _cross_owner_contacts(
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    maximum_scratch_bytes: int,
    sharding: NamedSharding,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[NDArray[np.int64], int]:
    construction = generated.construction
    points = (
        np.empty((0, 3, 3), dtype=np.float64)
        if construction is None
        else construction.vertices[construction.triangles]
    )
    ids = (
        np.empty((0, 3), dtype=np.int64)
        if construction is None
        else generated.vertex_ids[construction.triangles]
    )
    lower = np.min(points, axis=1) if points.size else np.empty((0, 3), dtype=np.float64)
    upper = np.max(points, axis=1) if points.size else np.empty((0, 3), dtype=np.float64)
    box = (
        np.stack((np.min(lower, axis=0), np.max(upper, axis=0)))
        if points.size
        else np.asarray(((np.inf,) * 3, (-np.inf,) * 3))
    )
    boxes = np.asarray(multihost_utils.process_allgather(box, tiled=False))
    # Packets have distinct int64 identity and binary64 coordinate channels.
    # Separate integer channels preserve IDs above the binary64 integer precision limit.
    capacity = min(256, maximum_scratch_bytes // (4 * (9 * 8 + 4 * 8)))
    if capacity < 1:
        raise compliance_failure(
            "initial.contact_packet must fit at least one candidate within the scratch budget; got zero capacity",
            stage="distributed-initial-certification",
        )
    findings: list[tuple[int, int, int]] = []
    tested = 0
    for first, second in combinations(range(jax.process_count()), 2):
        if np.any(boxes[first, 1] < boxes[second, 0]) or np.any(
            boxes[second, 1] < boxes[first, 0]
        ):
            continue
        rank = jax.process_index()
        other = second if rank == first else first
        selected = (
            np.flatnonzero(
                np.all(lower <= boxes[other, 1], axis=1)
                & np.all(upper >= boxes[other, 0], axis=1)
            )
            if rank in (first, second)
            else np.empty((0,), dtype=np.int64)
        )
        counts = np.asarray(
            multihost_utils.process_allgather(np.int64(selected.size), tiled=False)
        )
        rounds = (max(int(counts[first]), int(counts[second])) + capacity - 1) // capacity
        for step in range(rounds):
            rows = selected[step * capacity : (step + 1) * capacity]
            geometry_packet = np.zeros((capacity, 3, 3), dtype=np.float64)
            identity_packet = np.full((capacity, 4), -1, dtype=np.int64)
            geometry_packet[: rows.size] = points[rows]
            identity_packet[: rows.size, :3] = ids[rows]
            identity_packet[: rows.size, 3] = generated.cell_ids[rows]
            remote_points = _exchange(geometry_packet, first, second, sharding)
            remote_ids = _exchange(identity_packet, first, second, sharding)
            if rank != first:
                continue
            packet = _remote_contact_findings(
                points,
                ids,
                generated.cell_ids,
                lower,
                upper,
                remote_points,
                remote_ids,
                capacity,
                tested,
                limits.maximum_candidate_pairs,
                budget,
            )
            findings.extend(packet.rows)
            tested = packet.tested
    return np.asarray(findings, dtype=np.int64).reshape((-1, 3)), tested

def _premise_checks(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
    | planar_generation_owner.PreparedDistributedPlanarGeneration,
    /,
) -> tuple[str, ...]:
    match prepared:
        case _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration():
            return _SURFACE_CHECKS
        case planar_generation_owner.PreparedDistributedPlanarGeneration():
            return _PLANAR_CHECKS
        case _:
            assert_never(prepared)


def _triangle_angles(corners: NDArray[np.float64], /) -> NDArray[np.float64]:
    """Interior angles (radians) of ``(n, 3, d)`` triangle corners."""
    first = np.roll(corners, -1, axis=1) - corners
    second = np.roll(corners, 1, axis=1) - corners
    lengths = np.linalg.norm(first, axis=2) * np.linalg.norm(second, axis=2)
    cosine = np.sum(first * second, axis=2) / np.where(lengths > 0.0, lengths, 1.0)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _planar_edge_parameters(
    edge_rows: NDArray[np.int64],
    vertices: NDArray[np.int64],
    dimensions: NDArray[np.int64],
    indices: NDArray[np.int64],
    parameters: NDArray[np.float64],
    source_edges: NDArray[np.int64],
    /,
) -> NDArray[np.float64]:
    """Binary64 source-edge parameter of each local vertex row, NaN if foreign."""
    edge = source_edges[edge_rows]
    on_edge = (dimensions[vertices] == 1) & (indices[vertices] == edge_rows)
    start = (dimensions[vertices] == 0) & (indices[vertices] == edge[:, 0])
    end = (dimensions[vertices] == 0) & (indices[vertices] == edge[:, 1])
    return np.where(
        on_edge,
        parameters[vertices],
        np.where(start, 0.0, np.where(end, 1.0, np.nan)),
    )


def _planar_local_proof(
    prepared: planar_generation_owner.PreparedDistributedPlanarGeneration,
    generated: planar_generation_owner.DistributedPlanarConstruction,
    minimum_angle: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _LocalProof:
    """Certify one owner's tiles against the authored region and its controls."""
    tiles = prepared.tiles
    source = tiles.source
    stage = "distributed-initial-certification"
    if (
        generated.compiled_id != tiles.compiled_id
        or generated.source_id != source.source_id
        or generated.source_revision != source.source_revision
        or generated.specification_id != tiles.specification.specification_id
    ):
        raise compliance_failure(
            "initial.source_binding must match the tile complex, source revision and specification; got different identities",
            stage=stage,
        )
    target = tiles.specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 2
        or target.ambient_dimension != 2
        or target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"triangle"}
        or families.allowed_transitions
        or families.allow_mixed
    ):
        raise compliance_failure(
            "initial.target must declare affine planar triangles; got incompatible target semantics",
            stage=stage,
        )
    local_tiles = np.flatnonzero(prepared.tile_owners == jax.process_index())
    coordinates = generated.coordinates
    triangles = generated.triangles
    nv, nc = generated.vertex_ids.size, generated.cell_ids.size
    if (
        coordinates.shape != (nv, 2)
        or generated.vertex_owners.shape != (nv,)
        or generated.vertex_dimensions.shape != (nv,)
        or generated.vertex_indices.shape != (nv,)
        or generated.vertex_parameters.shape != (nv,)
        or triangles.shape != (nc, 3)
        or generated.cell_tiles.shape != (nc,)
        or np.unique(generated.vertex_ids).size != nv
        or np.unique(generated.cell_ids).size != nc
        or np.any(generated.vertex_ids < 0)
        or np.any(generated.vertex_ids >= generated.global_vertex_count)
        or np.any(generated.cell_ids < 0)
        or np.any(generated.cell_ids >= generated.global_cell_count)
        or np.any(generated.vertex_owners < 0)
        or np.any(generated.vertex_owners >= prepared.partition_count)
        or np.any(triangles < 0)
        or np.any(triangles >= nv)
        or not np.array_equal(generated.local_tiles, local_tiles)
        or not np.array_equal(np.unique(generated.cell_tiles), local_tiles)
        or not np.all(np.isfinite(coordinates))
    ):
        raise compliance_failure(
            "initial.generated_rows must have unique in-range IDs and assigned tile incidence; got inconsistent rows",
            stage=stage,
        )
    corners = coordinates[triangles]
    orientation = exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
    if np.any(orientation <= 0) or (
        np.unique(np.sort(triangles, axis=1), axis=0).shape[0] != nc
    ):
        raise compliance_failure(
            "initial.local_embedding must have distinct positively oriented cells; got inverted, degenerate or repeated cells",
            stage=stage,
        )
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    areas = 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    tile_count = tiles.tiles.shape[0]
    tile_area = np.bincount(generated.cell_tiles, weights=areas, minlength=tile_count)
    tile_cells = np.bincount(generated.cell_tiles, minlength=tile_count)
    scale = max(1.0, float(np.max(np.abs(tiles.origin))) + tiles.span) ** 2
    tolerance = 64.0 * np.finfo(np.float64).eps * scale * np.maximum(tile_cells, 1)
    covered = np.abs(tile_area - tiles.tile_weights) <= tolerance
    if not np.all(covered[local_tiles]):
        raise compliance_failure(
            "initial.tile_region_coverage must reproduce every owned tile's region area; got an uncovered or overlapping tile",
            stage=stage,
        )
    tile_checks = np.zeros((tile_count,), dtype=np.bool_)
    tile_checks[local_tiles] = True
    vertices, edges, loops = source_arrays(source)
    dimensions = generated.vertex_dimensions
    indices = generated.vertex_indices
    corner_rows = dimensions == 0
    edge_rows = dimensions == 1
    if (
        np.any((dimensions < 0) | (dimensions > 2))
        or np.any(corner_rows & ((indices < 0) | (indices >= vertices.shape[0])))
        or np.any(edge_rows & ((indices < 0) | (indices >= edges.shape[0])))
        or np.any((dimensions == 2) & (indices != 0))
        or not np.array_equal(
            coordinates[corner_rows], vertices[indices[corner_rows]]
        )
    ):
        raise compliance_failure(
            "initial.authored_source_binding must bind vertices to source vertices, edges or the region; got foreign provenance",
            stage=stage,
        )
    constrained = dimensions < 2
    residuals = np.zeros((nv,), dtype=np.float64)
    residuals[constrained] = _association_distances(
        coordinates[constrained],
        vertices,
        edges,
        dimensions[constrained].astype(np.int8),
        indices[constrained],
        budget=budget,
    )
    bounds = np.where(
        edge_rows, tiles.source_bounds[np.clip(indices, 0, edges.shape[0] - 1)], 0.0
    )
    if np.any(residuals[constrained] > bounds[constrained]):
        raise compliance_failure(
            "initial.source_carrier_fidelity must keep constrained vertices within their source bounds; got excess deviation",
            stage=stage,
        )
    carried = generated.edge_sources >= 0
    pairs = generated.edge_vertices[carried]
    carried_edges = generated.edge_sources[carried]
    for column in (0, 1):
        if np.any(
            np.isnan(
                _planar_edge_parameters(
                    carried_edges,
                    pairs[:, column],
                    dimensions,
                    indices,
                    generated.vertex_parameters,
                    edges,
                )
            )
        ):
            raise compliance_failure(
                "initial.authored_source_binding must bind constrained edge endpoints to their source edge; got a foreign endpoint",
                stage=stage,
            )
    if nc and minimum_angle > 0.0:
        achieved = float(np.min(_triangle_angles(corners)))
        if achieved < minimum_angle:
            raise compliance_failure(
                f"initial.physical_controls must satisfy the hard minimum angle; got {achieved} < {minimum_angle}",
                stage=stage,
            )
    certificates = tuple(
        canonical_fingerprint(
            {
                "kind": "planar-tile-certificate",
                "tiles": tiles.compiled_id,
                "tile": int(tile),
                "cells": array_tree_fingerprint(
                    np.sort(generated.cell_ids[generated.cell_tiles == tile])
                ),
                "area": float(tile_area[tile]),
            }
        )
        for tile in local_tiles.tolist()
    )
    return _LocalProof(certificates, tile_checks, residuals)


def _planar_owner_rows(
    prepared: planar_generation_owner.PreparedDistributedPlanarGeneration,
    generated: planar_generation_owner.DistributedPlanarConstruction,
    local: _LocalProof,
    /,
) -> _InitialOwnerRows:
    """Planar vertex provenance, region cells and source-edge facts of one owner."""
    tiles = prepared.tiles
    _, edges, loops = source_arrays(tiles.source)
    residuals = local.vertex_residuals
    if residuals is None:
        raise ValueError(
            "initial.planar_proof must retain constrained vertex residuals; got None"
        )
    dimensions = generated.vertex_dimensions
    parameters = np.full((dimensions.size, 2), np.nan, dtype=np.float64)
    parameters[dimensions == 1, 0] = generated.vertex_parameters[dimensions == 1]
    carried = generated.edge_sources >= 0
    pairs = generated.edge_vertices[carried]
    values = generated.edge_sources[carried]
    first, second = (
        _planar_edge_parameters(
            values,
            pairs[:, column],
            dimensions,
            generated.vertex_indices,
            generated.vertex_parameters,
            edges,
        )
        for column in (0, 1)
    )
    along = np.where((first <= second)[:, None], pairs, pairs[:, ::-1])
    ids = generated.vertex_ids
    return _InitialOwnerRows(
        coordinates=generated.coordinates,
        vertex_ids=ids,
        vertex_owners=generated.vertex_owners,
        corners=ids[generated.triangles],
        cell_ids=generated.cell_ids,
        vertex_banks=(
            ("source_dimensions", dimensions.astype(np.int64), -1),
            ("source_indices", generated.vertex_indices.astype(np.int64), -1),
            ("source_parameters", parameters, 0),
            ("source_residuals", residuals, 0.0),
        ),
        cell_banks=(
            ("patches", np.zeros((generated.cell_ids.size,), dtype=np.int64), -1),
            ("tiles", generated.cell_tiles.astype(np.int64), -1),
        ),
        curve_edges=np.sort(ids[pairs], axis=1).reshape((-1, 2)),
        curve_along=ids[along].reshape((-1, 2)),
        curve_values=values.astype(np.int64),
        curve_banks=(
            (
                "deviations",
                np.maximum(residuals[pairs[:, 0]], residuals[pairs[:, 1]]),
            ),
            ("deviation_bounds", tiles.source_bounds[values]),
        ),
        curve_uses=np.where(loops >= 0, 1, 2).astype(np.int32),
        work_checks=local.patch_checks,
        global_vertex_count=generated.global_vertex_count,
        global_cell_count=generated.global_cell_count,
    )


def _planar_tile_findings(
    prepared: planar_generation_owner.PreparedDistributedPlanarGeneration,
    generated: planar_generation_owner.DistributedPlanarConstruction,
    /,
) -> tuple[NDArray[np.int64], int]:
    """Cells outside their tile's closed lattice box.

    Distinct tiles meet only on lattice lines, so containment of every cell in
    its own closed box proves cross-owner interiors disjoint without exchanging
    candidate pairs; the tested pair count is therefore zero.
    """
    tiles = prepared.tiles
    corners = generated.coordinates[generated.triangles]
    low = tiles.origin + tiles.span * tiles.tiles[generated.cell_tiles].astype(np.float64)
    high = low + tiles.span
    inside = np.all(
        (corners >= low[:, None, :]) & (corners <= high[:, None, :]), axis=(1, 2)
    )
    outside = np.flatnonzero(~inside)
    findings = np.stack(
        (
            generated.cell_ids[outside],
            generated.cell_tiles[outside],
            np.ones((outside.size,), dtype=np.int64),
        ),
        axis=1,
    ).astype(np.int64)
    return findings.reshape((-1, 3)), 0


class InitialCollectiveMeshEvidence(AbstractCollectiveMeshTheorem, NonTrainableState):
    """Certify source binding and retain collective initial mesh content.

    ``premise`` discriminates the authored source: a parametric surface with
    patch owners, or a planar region tile complex with tile owners.
    """

    premise: InitialSurfacePremise | InitialPlanarPremise
    cell_kind: Literal["triangle"] = eqx.field(static=True)
    partition_checks: Array
    global_checks: Array
    logical_arrays: tuple[tuple[str, Array], ...]
    entity_keys: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    entity_owners: tuple[Array, ...]
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    coordinate_geometry_id: str = eqx.field(static=True)
    global_organization_id: str = eqx.field(static=True)
    source_evidence_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    checks: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    minimum_angle: float = eqx.field(static=True)
    maximum_normal_angle: float = eqx.field(static=True)
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    content_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
        | planar_generation_owner.PreparedDistributedPlanarGeneration,
        generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction
        | planar_generation_owner.DistributedPlanarConstruction,
        /,
        *,
        limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
        | None = None,
        minimum_angle: float,
        maximum_normal_angle: float = np.inf,
    ) -> None:
        limits_ = (
            _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits()
            if limits is None
            else limits
        )
        match prepared, generated:
            case (
                _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration(),
                _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction(),
            ):
                premise: InitialSurfacePremise | InitialPlanarPremise = (
                    InitialSurfacePremise(
                        prepared.compiled,
                        prepared.specification,
                        prepared.schedule,
                        prepared.layout,
                        prepared.patch_owners,
                    )
                )
            case (
                planar_generation_owner.PreparedDistributedPlanarGeneration(),
                planar_generation_owner.DistributedPlanarConstruction(),
            ):
                if not math.isinf(maximum_normal_angle):
                    raise ValueError(
                        "initial.maximum_normal_angle must be unbounded for planar regions; got a normal control"
                    )
                premise = InitialPlanarPremise(
                    prepared.tiles, prepared.layout, prepared.tile_owners
                )
            case _:
                raise TypeError(
                    "initial.prepared must pair with its own generated rows; got "
                    f"{type(prepared).__name__} with {type(generated).__name__}"
                )
        with NativeExecutionScope(limits_.native_limits()) as scope:
            ticket = scope.batch(0)
            ticket.settle()
            budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits_.maximum_work_units,
                maximum_memory_bytes=limits_.maximum_scratch_bytes,
            )
            with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
                self._initialize(
                    premise,
                    prepared,
                    generated,
                    limits_=limits_,
                    minimum_angle=minimum_angle,
                    maximum_normal_angle=maximum_normal_angle,
                    scope=scope,
                    budget=budget,
                )

    def _attempt_local_proof(
        self,
        prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
        | planar_generation_owner.PreparedDistributedPlanarGeneration,
        generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction
        | planar_generation_owner.DistributedPlanarConstruction,
        limits_: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
        minimum_angle: float,
        maximum_normal_angle: float,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
        /,
    ) -> _LocalProof | _InitialOwnerRejection:
        try:
            with budget.measurement():
                remaining = scope.remaining()
                ticket = scope.batch(
                    min(
                        limits_.maximum_work_units - budget.work_units,
                        remaining.remaining_work_units,
                    ),
                    queries_upper=min(
                        limits_.maximum_distance_evaluations,
                        remaining.remaining_geometry_queries,
                    ),
                    bytes_upper=min(
                        limits_.maximum_scratch_bytes, remaining.remaining_managed_bytes
                    ),
                )
                with budget.bind_ticket(ticket), ticket.native_target():
                    match prepared, generated:
                        case (
                            _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration(),
                            _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction(),
                        ):
                            return _local_proof(
                                prepared,
                                generated,
                                limits_,
                                minimum_angle,
                                maximum_normal_angle,
                                budget=budget,
                            )
                        case (
                            planar_generation_owner.PreparedDistributedPlanarGeneration(),
                            planar_generation_owner.DistributedPlanarConstruction(),
                        ):
                            return _planar_local_proof(
                                prepared, generated, minimum_angle, budget=budget
                            )
                        case _:
                            raise TypeError(
                                "initial.prepared must pair with its own generated rows; got mismatched kinds"
                            )
        except (
            MeshcoreError,
            _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
        ) as error:
            return _InitialOwnerRejection(
                _dependency_phydrax_meshing__distributed_generation.initial_owner_failure(
                    error, "distributed-initial-certification"
                )
            )
        except _leaf_phydrax_meshing__failures.MeshingFailure as error:
            return _InitialOwnerRejection(error)
        except ValueError as error:
            return _InitialOwnerRejection(
                compliance_failure(str(error), stage="distributed-initial-certification")
            )

    def _certify_local_owners(
        self,
        premise: InitialSurfacePremise | InitialPlanarPremise,
        prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
        | planar_generation_owner.PreparedDistributedPlanarGeneration,
        generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction
        | planar_generation_owner.DistributedPlanarConstruction,
        limits_: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
        minimum_angle: float,
        maximum_normal_angle: float,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
        /,
    ) -> _InitialOwnerProof:
        if (
            not np.isfinite(minimum_angle)
            or minimum_angle < 0.0
            or minimum_angle >= np.pi / 3.0
        ):
            raise ValueError(
                f"initial.minimum_angle must be finite and lie in [0, pi/3); got {minimum_angle}"
            )
        if np.isnan(maximum_normal_angle) or maximum_normal_angle <= 0.0:
            raise ValueError(
                f"initial.maximum_normal_angle must be positive and not NaN; got {maximum_normal_angle}"
            )
        binding = canonical_fingerprint(
            (
                premise.premise_id,
                limits_.limits_id,
                array_tree_fingerprint(
                    np.asarray((minimum_angle, maximum_normal_angle), dtype=np.float64)
                ),
            )
        )
        bindings = np.asarray(
            multihost_utils.process_allgather(
                np.frombuffer(bytes.fromhex(binding), dtype=np.uint8), tiled=False
            )
        )
        if np.any(bindings != bindings[0]):
            raise compliance_failure(
                "initial.owner_bindings must agree on source, controls, limits, and capacities; got different fingerprints",
                stage="distributed-initial-certification",
            )
        attempt = self._attempt_local_proof(
            prepared,
            generated,
            limits_,
            minimum_angle,
            maximum_normal_angle,
            scope,
            budget,
        )
        sharding = _placement()
        match attempt:
            case _LocalProof():
                local_category = None
            case _InitialOwnerRejection():
                local_category = attempt.failure.category
            case _:
                assert_never(attempt)
        decision = collective_failure_decision(local_category)
        if decision.category is not None:
            rejected_codes = jax.make_array_from_process_local_data(
                sharding,
                np.asarray((decision.code,), dtype=np.int32),
                (jax.process_count(),),
            )
            raw_cells = _global_rows(
                generated.cell_ids, prepared.layout.cell_capacity, sharding, -1
            )
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                decision.category,
                "initial.owner_checks must pass on every partition; got a rejected source certification",
                stage="distributed-initial-certification",
                logical_findings=(
                    ("initial/failed_owner_category", rejected_codes),
                    ("initial/rejected_cell_ids", raw_cells),
                ),
            )
        match attempt:
            case _LocalProof():
                return _InitialOwnerProof(attempt, sharding)
            case _InitialOwnerRejection():
                raise attempt.failure
            case _:
                assert_never(attempt)

    def _prepare_collective_content(
        self,
        premise: InitialSurfacePremise | InitialPlanarPremise,
        prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
        | planar_generation_owner.PreparedDistributedPlanarGeneration,
        generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction
        | planar_generation_owner.DistributedPlanarConstruction,
        owner: _InitialOwnerProof,
        /,
    ) -> _InitialContent:
        local, sharding = owner.local, owner.sharding
        match prepared, generated:
            case (
                _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration(),
                _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction(),
            ):
                owner_rows = _surface_owner_rows(prepared, generated, local)
            case (
                planar_generation_owner.PreparedDistributedPlanarGeneration(),
                planar_generation_owner.DistributedPlanarConstruction(),
            ):
                owner_rows = _planar_owner_rows(prepared, generated, local)
            case _:
                raise TypeError(
                    "initial.prepared must pair with its own generated rows; got mismatched kinds"
                )
        # The layout caps admit construction; immutable evidence retains only
        # the collectively required rows, not every unused construction slot.
        row_counts = np.asarray(
            multihost_utils.process_allgather(
                np.asarray(
                    (owner_rows.vertex_ids.size, owner_rows.cell_ids.size),
                    dtype=np.int64,
                ),
                tiled=False,
            )
        ).reshape((-1, 2))
        vertex_rows, cell_rows = (int(value) for value in np.max(row_counts, axis=0))
        if (
            vertex_rows > premise.layout.vertex_capacity
            or cell_rows > premise.layout.cell_capacity
        ):
            raise compliance_failure(
                "initial.owner_rows must fit the declared construction capacities; got excess rows",
                stage="distributed-initial-certification",
            )
        arrays, keys, identifiers, owners, counts, topology_id, geometry_id, checks = (
            _collective_initial_content(
                owner_rows,
                sharding,
                vertex_capacity=max(1, vertex_rows),
                cell_capacity=max(1, cell_rows),
                block=premise.block_name,
                width=premise.ambient_dimension,
            )
        )
        request_limits = premise.specification.limits
        quotas = (
            ("vertices", counts[0], request_limits.maximum_vertices),
            ("edges", counts[1], request_limits.maximum_edges),
            ("faces", counts[2], request_limits.maximum_faces),
            ("cells", counts[2], request_limits.maximum_cells),
            (
                "connectivity_entries",
                3 * counts[2],
                request_limits.maximum_connectivity_entries,
            ),
            (
                "data_bytes",
                8 * premise.ambient_dimension * counts[0] + 12 * counts[2],
                request_limits.maximum_data_bytes,
            ),
        )
        exceeded = tuple(name for name, measured, maximum in quotas if measured > maximum)
        if exceeded:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "initial.resources must remain within global request quotas; got exceeded quotas: "
                + ", ".join(exceeded),
                stage="distributed-initial-resources",
                requested=tuple((name, float(maximum)) for name, _, maximum in quotas),
                achieved=tuple((name, float(measured)) for name, measured, _ in quotas),
            )
        coordinate_geometry_id = _initial_coordinate_geometry_id(
            arrays, counts, topology_id, premise
        )
        return _InitialContent(
            arrays,
            keys,
            identifiers,
            owners,
            counts,
            topology_id,
            geometry_id,
            coordinate_geometry_id,
            checks,
        )

    def _certify_collective_contacts(
        self,
        prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
        | planar_generation_owner.PreparedDistributedPlanarGeneration,
        generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction
        | planar_generation_owner.DistributedPlanarConstruction,
        limits_: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
        content: _InitialContent,
        owner: _InitialOwnerProof,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
        /,
    ) -> _InitialContactProof:
        sharding, checks = owner.sharding, content.checks
        try:
            with budget.measurement():
                remaining = scope.remaining()
                ticket = scope.batch(
                    min(
                        limits_.maximum_work_units - budget.work_units,
                        remaining.remaining_work_units,
                    ),
                    queries_upper=min(
                        limits_.maximum_distance_evaluations,
                        remaining.remaining_geometry_queries,
                    ),
                    bytes_upper=min(
                        limits_.maximum_scratch_bytes, remaining.remaining_managed_bytes
                    ),
                )
                with budget.bind_ticket(ticket), ticket.native_target():
                    match prepared, generated:
                        case (
                            _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration(),
                            _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction(),
                        ):
                            findings, pairs = _cross_owner_contacts(
                                generated,
                                limits_,
                                prepared.specification.limits.maximum_scratch_bytes,
                                sharding,
                                budget=budget,
                            )
                        case (
                            planar_generation_owner.PreparedDistributedPlanarGeneration(),
                            planar_generation_owner.DistributedPlanarConstruction(),
                        ):
                            findings, pairs = _planar_tile_findings(prepared, generated)
                        case _:
                            raise TypeError(
                                "initial.prepared must pair with its own generated rows; got mismatched kinds"
                            )
        except (
            MeshcoreError,
            _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
        ) as error:
            failure = (
                _dependency_phydrax_meshing__distributed_generation.initial_owner_failure(
                    error, "distributed-initial-contacts"
                )
            )
            collective_failure_barrier(
                failure,
                message="An owner rejected initial native construction; no generated part was published.",
                stage="distributed-initial-commit",
            )
            raise failure from error
        collective_failure_barrier(
            None,
            message="An owner rejected initial native construction; no generated part was published.",
            stage="distributed-initial-commit",
        )
        cross = np.asarray(
            multihost_utils.process_allgather(
                np.bool_(findings.shape[0] == 0), tiled=False
            )
        )
        pair_counts = np.asarray(
            multihost_utils.process_allgather(np.int64(pairs), tiled=False)
        )
        cross = cross & (np.sum(pair_counts) <= limits_.maximum_candidate_pairs)
        checks = checks.at[:, -1].set(jnp.asarray(cross))
        verdict = jnp.all(checks, axis=0)
        summary = np.asarray(jax.device_get(verdict), dtype=np.bool_)
        finding_capacity = max(
            1,
            int(
                np.max(
                    np.asarray(
                        multihost_utils.process_allgather(
                            np.int64(findings.shape[0]), tiled=False
                        )
                    )
                )
            ),
        )
        raw_findings = _global_rows(findings, finding_capacity, sharding, -1)
        if not np.all(summary):
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
                "initial.collective_checks must pass; got failed checks: "
                + ", ".join(
                    name
                    for name, passed in zip(
                        _premise_checks(prepared), summary, strict=True
                    )
                    if not passed
                ),
                stage="distributed-initial-certification",
                logical_findings=(
                    ("initial/cross_owner_findings", raw_findings),
                    ("initial/partition_checks", checks),
                ),
            )
        return _InitialContactProof(checks, verdict, summary, raw_findings, pairs)

    def _retain_collective_content(
        self,
        premise: InitialSurfacePremise | InitialPlanarPremise,
        prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
        | planar_generation_owner.PreparedDistributedPlanarGeneration,
        generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction
        | planar_generation_owner.DistributedPlanarConstruction,
        limits_: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
        minimum_angle: float,
        maximum_normal_angle: float,
        content: _InitialContent,
        owner: _InitialOwnerProof,
        contact: _InitialContactProof,
        /,
    ) -> _InitialIdentity:
        arrays, counts = content.arrays, content.counts
        topology_id, geometry_id = content.topology_id, content.geometry_id
        local, sharding = owner.local, owner.sharding
        checks, summary = contact.checks, contact.summary
        raw_findings, pairs = contact.raw_findings, contact.pairs
        arrays["initial/cross_owner_findings"] = raw_findings
        arrays["initial/partition_checks"] = checks
        arrays["initial/intersection_pairs"] = jax.make_array_from_process_local_data(
            sharding, np.asarray((pairs,), dtype=np.int64), (jax.process_count(),)
        )
        certificate_rows = np.asarray(
            [
                np.frombuffer(bytes.fromhex(value), dtype=np.uint8)
                for value in local.certificate_ids
            ],
            dtype=np.uint8,
        ).reshape((-1, 32))
        match prepared, generated:
            case (
                _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration(),
                _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction(),
            ):
                certificate_capacity = 2 + 2 * prepared.compiled.patches.size
                arrays.update(_retain_initial_inputs(prepared, generated, sharding))
            case (
                planar_generation_owner.PreparedDistributedPlanarGeneration(),
                planar_generation_owner.DistributedPlanarConstruction(),
            ):
                certificate_capacity = prepared.tiles.tiles.shape[0]
            case _:
                raise TypeError(
                    "initial.prepared must pair with its own generated rows; got mismatched kinds"
                )
        arrays["initial/certificate_ids"] = _global_rows(
            certificate_rows, certificate_capacity, sharding
        )
        arrays["initial/certificate_counts"] = jax.make_array_from_process_local_data(
            sharding,
            np.asarray((certificate_rows.shape[0],), dtype=np.int64),
            (jax.process_count(),),
        )
        closures, fits = _initial_closures(
            arrays,
            vertex_count=counts[0],
            cell_count=counts[2],
            capacity=min(premise.layout.cell_capacity, counts[2]),
            parts=jax.process_count(),
            neighborhood_depth=premise.layout.neighborhood_depth,
        )
        if not bool(jax.device_get(fits)):
            raise compliance_failure(
                "initial.closure must fit the declared fixed capacity; got excess closure rows",
                stage="distributed-initial-certification",
            )
        arrays.update(
            {name: jax.device_put(value, sharding) for name, value in closures.items()}
        )

        membership = _owner_phydrax_meshing__collective_organization.initial_organization_membership(
            premise, arrays, counts
        )
        arrays.update(membership)
        organization_names = (
            *_INITIAL_ORGANIZATION_KEYS,
            *(name for name, _ in membership),
        )
        organization_id = canonical_fingerprint(
            {
                "kind": "collective-initial-source-organization",
                "source": premise.source_domain_id,
                "specification": premise.specification.specification_id,
                "content": logical_array_value_collection_digest(
                    {name: arrays[name] for name in organization_names}
                ),
            }
        )
        content_id = logical_array_value_collection_digest(arrays)
        source_evidence_id = _collective_source_payload(
            premise,
            limits_,
            minimum_angle,
            maximum_normal_angle,
            content_id,
            organization_id,
        ).fingerprint()
        mesh_id = canonical_fingerprint(
            {"kind": "cell-mesh", "topology": topology_id, "geometry": geometry_id}
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "collective-native-initial-evidence",
                "mesh": mesh_id,
                "source_theorem": source_evidence_id,
                "content": content_id,
                "checks": list(zip(_premise_checks(prepared), summary.tolist(), strict=True)),
            }
        )
        return _InitialIdentity(
            organization_id, content_id, source_evidence_id, mesh_id, evidence_id
        )

    def _initialize(
        self,
        premise: InitialSurfacePremise | InitialPlanarPremise,
        prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration
        | planar_generation_owner.PreparedDistributedPlanarGeneration,
        generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction
        | planar_generation_owner.DistributedPlanarConstruction,
        /,
        *,
        limits_: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
        minimum_angle: float,
        maximum_normal_angle: float,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        owner = self._certify_local_owners(
            premise,
            prepared,
            generated,
            limits_,
            minimum_angle,
            maximum_normal_angle,
            scope,
            budget,
        )
        content = self._prepare_collective_content(premise, prepared, generated, owner)
        contact = self._certify_collective_contacts(
            prepared, generated, limits_, content, owner, scope, budget
        )
        identity = self._retain_collective_content(
            premise,
            prepared,
            generated,
            limits_,
            minimum_angle,
            maximum_normal_angle,
            content,
            owner,
            contact,
        )
        self.premise = premise
        self.partition_checks = contact.checks
        self.global_checks = contact.verdict
        self.logical_arrays = tuple(sorted(content.arrays.items()))
        self.entity_keys = content.keys
        self.entity_ids = content.identifiers
        self.entity_owners = content.owners
        self.global_entity_counts = content.counts
        self.topology_id = content.topology_id
        self.geometry_id = content.geometry_id
        self.coordinate_geometry_id = content.coordinate_geometry_id
        self.global_organization_id = identity.organization_id
        self.source_evidence_id = identity.source_evidence_id
        self.mesh_id = identity.mesh_id
        self.partition_count = jax.process_count()
        self.checks = _premise_checks(prepared)
        self.evidence_id = identity.evidence_id
        self.minimum_angle = minimum_angle
        self.maximum_normal_angle = maximum_normal_angle
        self.cell_kind = "triangle"
        self.limits = limits_
        self.content_id = identity.content_id

    def require_passed(self) -> None:
        if not bool(jax.device_get(jnp.all(self.global_checks))):
            raise compliance_failure(
                "initial.global_checks must pass; got a rejected verdict",
                stage="distributed-initial-certification",
            )

    def require_current(self) -> None:
        """Collective immutable-content replay before any asymmetric constructor."""
        with agreed_local_step(stage="distributed-initial-certification"):
            self.require_passed()
            arrays = dict(self.logical_arrays)
            block = self.premise.block_name
            # Check the recipe's bank accesses before either distributed digest.
            for name in (
                "geometry/coordinate_ids",
                "geometry/coordinates",
                f"geometry/cell_ids/{block}",
                f"geometry/routes/{block}",
                f"geometry/source_basis/{block}",
            ):
                if name not in arrays:
                    raise compliance_failure(
                        f"initial.logical_arrays must contain {name!r}; got absent bank",
                        stage="distributed-initial-certification",
                    )
        content_id = logical_array_value_collection_digest(arrays)
        with agreed_local_step(stage="distributed-initial-certification"):
            if content_id != self.content_id:
                raise compliance_failure(
                    "initial.content_id must match retained numerical content; got a different digest",
                    stage="distributed-initial-certification",
                )
        coordinate_geometry_id = _initial_coordinate_geometry_id(
            arrays, self.global_entity_counts, self.topology_id, self.premise
        )
        with agreed_local_step(stage="distributed-initial-certification"):
            if coordinate_geometry_id != self.coordinate_geometry_id:
                raise compliance_failure(
                    "initial.coordinate_geometry_id must match the basis and coefficient banks; got a different fingerprint",
                    stage="distributed-initial-certification",
                )
            expected = _collective_source_payload(
                self.premise,
                self.limits,
                self.minimum_angle,
                self.maximum_normal_angle,
                self.content_id,
                self.global_organization_id,
            ).fingerprint()
            if expected != self.source_evidence_id:
                raise compliance_failure(
                    "initial.source_evidence_id must match the domain and physical controls; got a different fingerprint",
                    stage="distributed-initial-certification",
                )


@checked
def restore_initial_construction(
    evidence: InitialCollectiveMeshEvidence,
    partition_index: int,
    /,
) -> _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction:
    """Restore the numerical surface inputs used by initial certification."""

    if not 0 <= partition_index < evidence.partition_count:
        raise ValueError(
            f"initial.partition_index must lie in the retained process group; got {partition_index}"
        )
    premise = evidence.premise
    if not isinstance(premise, InitialSurfacePremise):
        raise ValueError(
            "initial.restore_construction must replay a surface premise; got a planar tile premise"
        )
    banks = dict(evidence.logical_arrays)

    def local(name: str) -> np.ndarray:
        return np.asarray(
            jax.device_get(
                _owner_phydrax_meshing__publication_lowering.addressable(
                    banks[name], partition_index
                )
            )
        )

    def rows(name: str) -> np.ndarray:
        count = int(local(f"initial/input_count/{name}"))
        values = local(f"initial/input/{name}")
        if not 0 <= count <= values.shape[0]:
            raise compliance_failure(
                f"initial.input_count must lie within packet {name!r}; got {count}",
                stage="distributed-initial-certification",
            )
        return values[:count]

    metadata = local("initial/input_metadata")
    construction = None
    if bool(local("initial/input_present")):
        arrays = {name: rows(name) for name, _, _ in _INITIAL_INPUT_FIELDS}
        charts = []
        for patch in premise.compiled.patches.tolist():
            values = rows(f"chart/{patch}/charts")
            if values.shape[0]:
                charts.append(
                    (
                        patch,
                        values,
                        rows(f"chart/{patch}/points"),
                        rows(f"chart/{patch}/cells"),
                        rows(f"chart/{patch}/boundary"),
                        rows(f"chart/{patch}/provenance"),
                        rows(f"chart/{patch}/restriction_required"),
                        rows(f"chart/{patch}/restriction_vertices"),
                        rows(f"chart/{patch}/restriction_edges"),
                        rows(f"chart/{patch}/restriction_parameters"),
                    )
                )
        curves = []
        for curve in premise.compiled.curves.tolist():
            values = rows(f"curve/{curve}")
            if values.shape[0]:
                curves.append((curve, values))
        construction = surface_construction_owner.SurfaceConstruction(
            **arrays,
            chart_triangulations=tuple(charts),
            curve_parameters=tuple(curves),
            minimum_angle=float(local("initial/input_minimum_angle")),
            rounds=int(metadata[0]),
            inserted=int(metadata[1]),
            refused=int(metadata[2]),
            flips=int(metadata[3]),
            geometry_queries=int(metadata[4]),
            work_units=int(metadata[5]),
            incomplete_refinement=bool(metadata[6]),
        )
    return _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction(
        construction,
        rows("vertex_ids"),
        rows("vertex_owners"),
        rows("cell_ids"),
        evidence.global_entity_counts[0],
        evidence.global_entity_counts[-1],
        rows("shared_curve_ids"),
        int(metadata[7]),
        int(metadata[8]),
        int(metadata[9]),
        None,
        premise.source_id,
        premise.source_revision,
        premise.specification.specification_id,
    )


def _lower_initial_mesh(
    evidence: InitialCollectiveMeshEvidence,
    receipts: dict[str, np.ndarray],
    geometry_projection: _dependency_phydrax_discretization__cell_geometry.CellGeometryStorageProjection,
    /,
) -> CellMesh:

    valid = receipts["entity/0/valid"]
    vertex_ids = receipts["entity/0/ids"][valid]
    coordinates = receipts["coordinates"][valid]
    active = receipts["closure/cell_valid"]
    cell_ids = receipts["closure/cell_ids"][active]
    corners = receipts["closure/cell_vertices"][active]
    cells = np.searchsorted(vertex_ids, corners).astype(np.int32)
    blocks = (
        ()
        if cell_ids.size == 0
        else (
            CellBlock(
                evidence.premise.block_name,
                evidence.cell_kind,
                cells,
                global_ids=cell_ids,
            ),
        )
    )
    probe = (
        None
        if not blocks
        else CellMesh(coordinates, blocks, vertex_global_ids=vertex_ids)
    )
    identifiers, owners = [], []
    boundary_entities = []
    for degree in range(3):
        mask = receipts[f"entity/{degree}/valid"]
        positions = (
            np.empty((0,), dtype=np.int64)
            if probe is None
            else entity_keys_owner.key_rows(
                receipts[f"entity/{degree}/keys"][mask],
                entity_keys_owner.entity_keys(probe, degree),
            )
        )
        identifiers.append(receipts[f"entity/{degree}/ids"][mask][positions])
        owners.append(receipts[f"entity/{degree}/owners"][mask][positions])
        boundary_entities.append(
            receipts[f"entity/{degree}/physical_boundary"][mask][positions]
        )
    edge_mask = receipts["entity/1/valid"]
    receipt_edge_keys = receipts["entity/1/keys"][edge_mask]
    edge_keys = (
        receipt_edge_keys if probe is None else entity_keys_owner.entity_keys(probe, 1)
    )
    positions = (
        np.empty((0,), dtype=np.int64)
        if probe is None
        else entity_keys_owner.key_rows(receipt_edge_keys, edge_keys)
    )
    physical = receipts["initial/edge_physical"][edge_mask][positions]
    facets = np.sort(
        corners[:, np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int32)], axis=2
    )
    facet_rows = (
        np.empty((0, 3), dtype=np.int64)
        if facets.shape[0] == 0
        else entity_keys_owner.key_rows(edge_keys, facets.reshape((-1, 2))).reshape(
            (-1, 3)
        )
    )
    edge_incidence = np.bincount(facet_rows.reshape(-1), minlength=physical.size)
    complete = np.all((edge_incidence[facet_rows] == 2) | physical[facet_rows], axis=1)
    storage = _owner_phydrax_discretization__cell_mesh.CellMeshStorage(
        evidence.global_entity_counts,
        identifiers,
        owners,
        partition_index=jax.process_index(),
        partition_count=evidence.partition_count,
        local_coordinates=coordinates,
        local_blocks=blocks,
        global_coordinate_count=evidence.global_entity_counts[0],
        coordinate_global_ids=vertex_ids,
        coordinate_owner=owners[0],
        geometry_projection=geometry_projection,
        local_physical_boundary_facets=physical,
        local_physical_boundary_entities=boundary_entities,
        local_neighborhood_complete=complete,
        neighborhood_depth=evidence.premise.layout.neighborhood_depth,
        logical_topology_id=evidence.topology_id,
        logical_geometry_id=evidence.geometry_id,
        logical_coordinate_geometry_id=evidence.coordinate_geometry_id,
        evidence_id=evidence.evidence_id,
        logical_arrays=evidence.logical_arrays,
    )
    return CellMesh(
        coordinates,
        blocks,
        storage=storage,
        numeric_version=evidence.premise.source_revision,
    )


def _lower_initial_construction(
    mesh: CellMesh,
    receipts: dict[str, np.ndarray],
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    minimum_angle: float,
    /,
) -> surface_construction_owner.SurfaceConstruction:

    vertex_mask, cell_mask = receipts["entity/0/valid"], receipts["entity/2/valid"]
    vertices = np.asarray(mesh.vertex_global_ids)
    order = entity_keys_owner.key_rows(
        receipts["entity/0/ids"][vertex_mask, None], vertices[:, None]
    )
    cell_order = entity_keys_owner.key_rows(
        receipts["entity/2/ids"][cell_mask, None],
        np.asarray(mesh.blocks[0].global_ids)[:, None],
    )
    edge_mask = receipts["entity/1/valid"] & (receipts["initial/edge_curves"] >= 0)
    along = receipts["initial/edge_along"][edge_mask]
    original = generated.construction
    if original is None:
        restriction_vertices = np.empty((0,), dtype=np.int64)
        restriction_edges = np.empty((0, 2), dtype=np.int64)
        restriction_endpoint_parameters = np.empty((0, 2, 2), dtype=np.float64)
        restriction_parameters = np.empty((0, 2), dtype=np.int64)
    else:
        restriction_vertices = np.searchsorted(
            vertices,
            generated.vertex_ids[original.chart_restriction_vertices],
        ).astype(np.int64)
        restriction_edges = np.searchsorted(
            vertices,
            generated.vertex_ids[original.chart_restriction_edges],
        ).astype(np.int64)
        restriction_endpoint_parameters = original.chart_restriction_endpoint_parameters
        restriction_parameters = original.chart_restriction_parameters
    return surface_construction_owner.SurfaceConstruction(
        vertices=np.asarray(mesh.coordinates),
        vertex_source_dimensions=receipts["initial/source_dimensions"][vertex_mask][
            order
        ],
        vertex_source_indices=receipts["initial/source_indices"][vertex_mask][order],
        vertex_parameters=receipts["initial/source_parameters"][vertex_mask][order],
        chart_restriction_vertices=restriction_vertices,
        chart_restriction_edges=restriction_edges,
        chart_restriction_endpoint_parameters=restriction_endpoint_parameters,
        chart_restriction_parameters=restriction_parameters,
        triangles=np.asarray(mesh.blocks[0].vertices),
        triangle_patches=receipts["initial/cell_patches"][cell_mask][cell_order],
        triangle_charts=receipts["initial/cell_charts"][cell_mask][cell_order],
        triangle_parameters=receipts["initial/cell_parameters"][cell_mask][cell_order],
        triangle_deviations=receipts["initial/cell_deviations"][cell_mask][cell_order],
        triangle_deviation_bounds=receipts["initial/cell_deviation_bounds"][cell_mask][
            cell_order
        ],
        triangle_normal_bounds=receipts["initial/cell_normal_bounds"][cell_mask][
            cell_order
        ],
        curve_edges=np.searchsorted(vertices, along).astype(np.int32),
        curve_edge_curves=receipts["initial/edge_curves"][edge_mask],
        curve_edge_deviations=receipts["initial/edge_deviations"][edge_mask],
        curve_edge_deviation_bounds=receipts["initial/edge_deviation_bounds"][edge_mask],
        unresolved=np.zeros((mesh.entity_set(2).count,), dtype=np.bool_),
        boundary_occurrences=receipts["initial/cell_boundary_occurrences"][cell_mask][
            cell_order
        ],
        chart_triangulations=() if original is None else original.chart_triangulations,
        curve_parameters=() if original is None else original.curve_parameters,
        minimum_angle=minimum_angle,
        rounds=0 if original is None else original.rounds,
        inserted=0 if original is None else original.inserted,
        refused=0 if original is None else original.refused,
        flips=0 if original is None else original.flips,
        geometry_queries=0 if original is None else original.geometry_queries,
        work_units=0 if original is None else original.work_units,
        incomplete_refinement=False
        if original is None
        else original.incomplete_refinement,
    )


def _initial_region_boundaries(
    evidence: InitialCollectiveMeshEvidence,
    mesh: CellMesh,
    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...],
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...],
    /,
) -> tuple[_dependency_phydrax_meshing__organization.RegionBoundaryEvidence, ...]:
    """Bind authored region sides to the final globally scoped patch identities."""

    premise = evidence.premise
    if not isinstance(premise, InitialSurfacePremise):
        raise ValueError(
            "initial.region_boundaries must bind authored surface regions; got a planar tile premise"
        )
    domain, specification = premise.compiled.domain, premise.specification
    result = []
    for region in range(len(domain.regions)):
        identifier = domain.entity_id(3, region)
        sides = tuple(
            (
                patch.patch_id,
                1 if patch.source_adjacent_region_ids[0] == identifier else -1,
            )
            for patch in patches
            if patch.source_adjacent_region_ids is not None
            and identifier in patch.source_adjacent_region_ids
        )
        if not sides:
            continue
        controls = tuple(
            control
            for control in specification.region_controls
            if control.scope.entity_dimension == 3
            and region
            in domain.resolve_indices(
                3, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
            )
        )
        control = controls[0] if controls else None
        name = (
            f"region:{identifier}"
            if control is None
            else control.region_name
            if control.scope.global_entity_ids.shape[0] == 1
            else f"{control.region_name}:{identifier}"
        )
        selected = tuple(label for label in labels if label.name == name)
        if len(selected) != 1:
            raise ValueError(
                "initial.region.boundary must have one logical boundary label; got a different label count"
            )
        scope = _owner_phydrax_meshing__scope.MeshingScope(
            domain.source_id,
            domain.source_revision,
            _owner_phydrax_meshing__scope.MeshingEntityKind.GEOMETRY,
            3,
            domain.entity_set_id(3),
            np.asarray((domain.scope_indices(3)[region],), dtype=np.int64),
        )
        result.append(
            _dependency_phydrax_meshing__organization.RegionBoundaryEvidence(
                mesh,
                scope,
                identifier,
                domain.domain_id if control is None else control.control_id,
                None if control is None else control.material_id,
                None if control is None else control.role,
                selected[0],
                sides,
            )
        )
    return tuple(result)


@checked
def publish_distributed_surface_initial(
    prepared: _dependency_phydrax_meshing__distributed_generation.PreparedDistributedSurfaceGeneration,
    generated: _dependency_phydrax_meshing__distributed_generation.DistributedSurfaceConstruction,
    source: _dependency_phydrax_meshing_providers__native_sources.NativeSurfaceSource,
    coordinate_contract: SpatialCoordinateContract,
    provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo,
    /,
    *,
    minimum_angle: float,
    maximum_normal_angle: float = np.inf,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    plan_id: str | None = None,
) -> _owner_phydrax_meshing__result.CellMeshingResult:
    """Accept the authored source theorem before lowering any result constructor."""

    if source.domain.domain_id != prepared.compiled.domain.domain_id:
        raise ValueError(
            "initial.publication.source must match the compiled domain; got a different domain identity"
        )
    evidence = InitialCollectiveMeshEvidence(
        prepared,
        generated,
        minimum_angle=minimum_angle,
        maximum_normal_angle=maximum_normal_angle,
        limits=limits,
    )
    projection = _owner_phydrax_meshing__publication_lowering.PreparedPublicationLowering(
        None, evidence, None, vertex_capacity=prepared.layout.vertex_capacity
    ).execute()
    projection.require_source(evidence.logical_arrays)
    storage_binding = _owner_phydrax_meshing__result.CollectiveMeshStorageBinding(
        evidence, evidence.logical_arrays, evidence.global_entity_counts
    )
    scopes = _owner_phydrax_meshing__organization.prepare_mesh_organization_scopes(
        evidence, evidence, projection, evidence.premise.source_revision
    )
    receipts = {
        name: np.asarray(jax.device_get(value))
        for name, value in projection.addressable_arrays(jax.process_index())
    }
    local_angle = (
        np.inf if generated.construction is None else generated.construction.minimum_angle
    )
    achieved_angle = float(
        np.min(
            np.asarray(
                multihost_utils.process_allgather(np.float64(local_angle), tiled=False)
            )
        )
    )
    quality_target = prepared.specification.quality_target
    compliance = _owner_phydrax_meshing__result.MeshingComplianceReport(
        prepared.specification.specification_id,
        requested=()
        if quality_target is None
        else (("minimum_angle", quality_target.minimum_angle),),
        achieved=(("minimum_angle", achieved_angle),),
    )
    result = None
    failure = None
    try:
        mesh = _lower_initial_mesh(evidence, receipts, projection.geometry)
        if mesh.storage is None:
            raise ValueError("initial.mesh.storage must retain logical storage; got None")
        geometry = mesh.storage.restore_geometry()
        associations = ()
        if np.any(receipts["closure/cell_valid"]):
            construction = _lower_initial_construction(
                mesh, receipts, generated, minimum_angle
            )
            association_limits = prepared.specification.limits
            ticket = HostBatchTicket(
                association_limits.maximum_work_units,
                queries_upper=association_limits.maximum_geometry_queries,
                bytes_upper=association_limits.maximum_scratch_bytes,
            )
            association_budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=association_limits.maximum_work_units,
                maximum_memory_bytes=association_limits.maximum_scratch_bytes,
            )
            with ticket, association_budget.measurement(), association_budget.activate():
                associations, _, _ = surface_association_owner.associations(
                    source, mesh, construction, budget=association_budget
                )
        patches, zones, labels, attributes = (
            _owner_phydrax_meshing__organization.lower_mesh_organization(
                evidence, mesh, dict(evidence.logical_arrays), scope_projections=scopes
            )
        )
        region_boundaries = _initial_region_boundaries(evidence, mesh, patches, labels)
        audit = _owner_phydrax_meshing__audit.audit_cell_mesh(
            mesh,
            geometry,
            policy=_owner_phydrax_meshing__audit.CellMeshAuditPolicy(
                require_complete_association=True
            ),
            patches=patches,
            labels=labels,
            zones=zones,
            attributes=attributes,
            associations=associations,
        )
        audit.require_passed()
        audit.require_decided()
        trace = _owner_phydrax_meshing__trace.MeshingTrace(
            (
                _owner_phydrax_meshing__trace.MeshingStageReport(
                    _owner_phydrax_meshing__trace.MeshingStageKind.SURFACE_MESHING,
                    _owner_phydrax_meshing__trace.MeshingStageStatus.PASSED,
                    input_ids=(
                        prepared.compiled.compiled_id,
                        prepared.specification.specification_id,
                    ),
                    output_ids=(evidence.evidence_id, audit.report_id),
                ),
            )
        )
        result = _owner_phydrax_meshing__result.CellMeshingResult(
            mesh,
            geometry,
            coordinate_contract,
            audit,
            audit.quality,
            compliance,
            trace,
            provider,
            _owner_phydrax_meshing__result.MeshingRuntimeInfo(
                provider.provider_id,
                provider.version,
                _owner_phydrax_meshing__contracts.MeshingExecutionMode.IN_PROCESS,
                deterministic=True,
                enforced_limits=(
                    "source_theorem",
                    "collective_acceptance",
                    "fixed_capacity_closure",
                ),
            ),
            _owner_phydrax_meshing__contracts.MeshingDerivativeMode.NONDIFFERENTIABLE,
            _owner_phydrax__identity.SemanticProvenance(
                {
                    "kind": "native-distributed-initial-publication",
                    "source": source.binding_id,
                    "theorem": evidence.evidence_id,
                    "plan": plan_id,
                }
            ),
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            region_boundary_evidence=region_boundaries,
            collective_evidence=evidence,
            scope_projections=scopes,
            storage_binding=storage_binding,
        )
    except (
        MeshcoreError,
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
    ) as error:
        failure = (
            _dependency_phydrax_meshing__distributed_generation.initial_owner_failure(
                error, "distributed-initial-publication"
            )
        )
    except _leaf_phydrax_meshing__failures.MeshingFailure as error:
        failure = error
    except ValueError as error:
        failure = compliance_failure(
            str(error), stage="distributed-initial-certification"
        )
    collective_failure_barrier(
        failure,
        message="An owner rejected initial native construction; no generated part was published.",
        stage="distributed-initial-commit",
    )
    if result is None:
        raise ValueError(
            "initial.publication.result must be present on every partition; got None"
        )
    return result


def _process_values(value: np.ndarray | np.generic, /) -> np.ndarray:
    return np.asarray(multihost_utils.process_allgather(value, tiled=False))


def _order_statistic(values: NDArray[np.float64], rank: int, /) -> float:
    """``rank``-th smallest (0-based) nonnegative binary64 value over all owners.

    Nonnegative binary64 values order like their bit patterns, so bisection on
    the pattern needs only one exchanged count per round and no value gather.
    """
    patterns = np.sort(np.ascontiguousarray(values, dtype=np.float64).view(np.uint64))
    low, high = 0, int(np.asarray(np.inf, dtype=np.float64).view(np.uint64))
    while low < high:
        middle = (low + high) // 2
        local = np.int64(np.searchsorted(patterns, np.uint64(middle), side="right"))
        if int(np.sum(_process_values(local))) >= rank + 1:
            high = middle
        else:
            low = middle + 1
    return float(np.asarray(low, dtype=np.uint64).view(np.float64))


def _process_quantile(values: NDArray[np.float64], quantile: float, count: int, /) -> float:
    """Linear-interpolated global quantile of distributed nonnegative values."""
    position = quantile * (count - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, count - 1)
    first = _order_statistic(values, lower)
    second = first if upper == lower else _order_statistic(values, upper)
    return first + (second - first) * (position - lower)


def _planar_owned_edges(
    prepared: planar_generation_owner.PreparedDistributedPlanarGeneration,
    generated: planar_generation_owner.DistributedPlanarConstruction,
    /,
) -> NDArray[np.int64]:
    """Local edge rows counted once over owners: a seam edge belongs to its lower tile's owner."""
    triangles = generated.triangles
    edges = np.unique(
        np.sort(
            triangles[:, np.asarray(((0, 1), (1, 2), (0, 2)), dtype=np.int64)].reshape(
                (-1, 2)
            ),
            axis=1,
        ),
        axis=0,
    ).reshape((-1, 2))
    seams = generated.edge_seams >= 0
    seam_edges = generated.edge_vertices[seams]
    seam_rows = generated.edge_seams[seams]
    if seam_rows.size == 0:
        return edges
    lower_tiles = np.min(prepared.tiles.seam_tiles[seam_rows], axis=1)
    foreign = seam_edges[prepared.tile_owners[lower_tiles] != jax.process_index()]
    if foreign.shape[0] == 0:
        return edges
    width = np.int64(max(generated.vertex_ids.size, 1))
    keep = ~np.isin(edges[:, 0] * width + edges[:, 1], foreign[:, 0] * width + foreign[:, 1])
    return edges[keep]


def _planar_distributed_compliance(
    prepared: planar_generation_owner.PreparedDistributedPlanarGeneration,
    generated: planar_generation_owner.DistributedPlanarConstruction,
    owned_residual: float,
    /,
) -> _owner_phydrax_meshing__result.MeshingComplianceReport:
    """Planar size, quality and carrier statistics over all owners' owned rows.

    Quantities and names follow the serial planar compliance; extrema and sums
    exchange one scalar per quantity and quantiles use an exact distributed
    order-statistic search.
    """
    tiles = prepared.tiles
    specification = tiles.specification
    policy = specification.size_compliance
    points = generated.coordinates
    edges = _planar_owned_edges(prepared, generated)
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    lengths = np.linalg.norm(b - a, axis=1)
    local = np.min(
        _field_sizes(
            np.concatenate((a, b, 0.5 * (a + b))),
            tiles.region_size,
            tiles.patch_segments,
            tiles.patch_sizes,
            tiles.gradation_slope,
        ).reshape(3, -1),
        axis=0,
    )
    endpoint = _field_sizes(
        points,
        tiles.region_size,
        tiles.patch_segments,
        tiles.patch_sizes,
        tiles.gradation_slope,
    )
    gradation = 1.0 + float(
        np.max(
            _process_values(
                np.float64(
                    np.max(
                        np.abs(endpoint[edges[:, 1]] - endpoint[edges[:, 0]]) / lengths,
                        initial=0.0,
                    )
                )
            )
        )
    )
    constrained = generated.edge_sources >= 0
    constrained_pairs = generated.edge_vertices[constrained]
    constrained_sources = generated.edge_sources[constrained]
    width = np.int64(max(generated.vertex_ids.size, 1))
    edge_keys = edges[:, 0] * width + edges[:, 1]
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise TypeError(
                "specification.size_controls must contain UniformSizeControl; got an unsupported control"
            )
        if control.scope.entity_dimension == 1:
            pairs = constrained_pairs[
                np.isin(constrained_sources, np.asarray(control.scope.entity_ids))
            ]
            rows = np.flatnonzero(np.isin(edge_keys, pairs[:, 0] * width + pairs[:, 1]))
        else:
            governed = local >= tiles.region_size
            any_governed = bool(np.any(_process_values(np.bool_(np.any(governed)))))
            rows = np.flatnonzero(governed) if any_governed else np.arange(edges.shape[0])
        selected, selected_local = lengths[rows], local[rows]
        count = int(np.sum(_process_values(np.int64(selected.size))))
        key = f"size:{control.control_id}"
        requested.append((f"{key}:target_size", control.target_size))
        requested.extend(
            (f"{key}:{name}", value)
            for name, value in (
                ("minimum_size", control.minimum_size),
                ("maximum_size", control.maximum_size),
                ("maximum_growth_rate", control.maximum_growth_rate),
            )
            if value is not None
        )
        minimum = float(
            np.min(_process_values(np.float64(np.min(selected, initial=np.inf))))
        )
        maximum = float(np.max(_process_values(np.float64(np.max(selected, initial=0.0)))))
        ratio = float(
            np.max(
                _process_values(np.float64(np.max(selected / selected_local, initial=0.0)))
            )
        )
        achieved.extend(
            (
                (f"{key}:edge_count", float(count)),
                (f"{key}:minimum_edge", minimum),
                (f"{key}:maximum_edge", maximum),
                (f"{key}:maximum_edge_to_local_size", ratio),
                (f"{key}:size_field_growth_rate", gradation),
            )
        )
        statistics = {
            name: _process_quantile(selected, size_statistic_quantile(name), count)
            for name in policy.target_statistics
        }
        achieved.extend((f"{key}:{name}_edge", value) for name, value in statistics.items())
        if control.strength is SizeControlStrength.SOFT:
            continue
        tolerance = policy.tolerance
        issues.extend(
            f"target_size_{name}:{control.control_id}"
            for name, value in statistics.items()
            if abs(value - control.target_size) > tolerance(control.target_size)
        )
        oversized = bool(
            np.any(
                _process_values(
                    np.bool_(
                        np.any(
                            selected
                            > selected_local
                            + policy.absolute_tolerance
                            + policy.relative_tolerance * selected_local
                        )
                    )
                )
            )
        )
        checks = (
            (
                "minimum_size",
                control.minimum_size is not None
                and minimum < control.minimum_size - tolerance(control.minimum_size),
            ),
            (
                "maximum_size",
                control.maximum_size is not None
                and maximum > control.maximum_size + tolerance(control.maximum_size),
            ),
            ("local_size", oversized),
            (
                "maximum_growth_rate",
                control.maximum_growth_rate is not None
                and gradation
                > control.maximum_growth_rate + tolerance(control.maximum_growth_rate),
            ),
        )
        issues.extend(f"{name}:{control.control_id}" for name, failed in checks if failed)
    corners = points[generated.triangles]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    areas = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    maximum_area = float(np.max(_process_values(np.float64(np.max(areas, initial=0.0)))))
    local_angle = (
        float(np.min(_triangle_angles(corners))) if corners.shape[0] else math.inf
    )
    minimum_angle = float(np.min(_process_values(np.float64(local_angle))))
    steiner = int(np.sum(_process_values(np.int64(generated.steiner_count))))
    complete = bool(np.all(_process_values(np.bool_(generated.refinement_complete))))
    rounding = float(np.max(_process_values(np.float64(owned_residual))))
    admissible = float(np.max(tiles.source_bounds, initial=0.0))
    requested.extend(
        (
            ("maximum_triangle_area", tiles.maximum_area),
            ("maximum_steiner_points", tiles.maximum_steiner),
            ("maximum_constraint_rounding", admissible),
        )
    )
    achieved.extend(
        (
            ("maximum_triangle_area", maximum_area),
            ("minimum_angle", minimum_angle),
            ("steiner_points", steiner),
            ("maximum_constraint_rounding", rounding),
            ("seam_rounds", generated.seam_rounds),
        )
    )
    quality = specification.quality_target
    if quality is not None:
        requested.append(("minimum_angle", quality.minimum_angle))
        if quality.hard and minimum_angle < quality.minimum_angle:
            issues.append(f"minimum_angle:{quality.target_id}")
    if not complete:
        issues.append("refinement_limit:refinement_limit")
    if rounding > admissible:
        issues.append("constraint_rounding")
    for feature in specification.protected_features:
        key = f"protected:{feature.feature_id}:maximum_deviation"
        requested.append((key, feature.maximum_deviation))
        deviation = 0.0 if feature.feature_kind is FeatureKind.CORNER else rounding
        achieved.append((key, deviation))
        if feature.hard and deviation > feature.maximum_deviation:
            issues.append(key)
    return _owner_phydrax_meshing__result.MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _planar_associations(
    premise: InitialPlanarPremise,
    mesh: CellMesh,
    receipts: dict[str, np.ndarray],
    region_distance: float,
    /,
) -> tuple[GeometryAssociation, ...]:
    """Source vertex/edge/region associations of one owner's lowered closure."""
    from ._association_routes import ordered_vertices

    source = premise.authored_source
    revision = source.source_revision
    vertices, source_edges, _ = source_arrays(source)
    directions = vertices[source_edges[:, 1]] - vertices[source_edges[:, 0]]
    vertex_mask = receipts["entity/0/valid"]
    dimensions = receipts["initial/source_dimensions"][vertex_mask].astype(np.int8)
    indices = receipts["initial/source_indices"][vertex_mask].astype(np.int64)
    parameters = receipts["initial/source_parameters"][vertex_mask]
    residuals = receipts["initial/source_residuals"][vertex_mask]
    distances = np.where(dimensions < 2, residuals, region_distance)
    edge_mask = receipts["entity/1/valid"]
    positions = entity_keys_owner.key_rows(
        receipts["entity/1/keys"][edge_mask], entity_keys_owner.entity_keys(mesh, 1)
    )
    edge_curves = receipts["initial/edge_curves"][edge_mask][positions]
    edge_deviations = receipts["initial/edge_deviations"][edge_mask][positions]
    edge_set = mesh.entity_set(1)
    carried = np.flatnonzero(edge_curves >= 0)
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[carried]
    order = np.argsort(edge_ids, kind="stable")
    carried, edge_ids = carried[order], edge_ids[order]
    sources = edge_curves[carried].astype(np.int64)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    ordered = ordered_vertices(mesh, 1)
    if ordered is None:
        raise ValueError(
            "initial.planar_edge_orientation must retain ordered mesh incidence; got unsupported edges"
        )
    first, second = ordered[carried].T
    orientations = np.sign(
        np.sum((coordinates[second] - coordinates[first]) * directions[sources], axis=1)
    ).astype(np.int8)
    cell_set = mesh.entity_set(2)
    cell_count = cell_set.count
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.REGION,
    )
    entity_id = _dependency_phydrax_meshing_providers__native_sources.source_entity_id
    return (
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            cell_set.entity_set_id,
            cell_set.entity_ids,
            tuple(entity_id(revision, "region", 0) for _ in range(cell_count)),
            np.full((cell_count,), region_distance, dtype=np.float64),
            exact=region_distance == 0.0,
            source_dimensions=np.full(cell_count, 2, dtype=np.int8),
            source_indices=np.zeros(cell_count, dtype=np.int64),
            source_entity_roles=(GeometrySourceEntityRole.REGION,) * cell_count,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            edge_set.entity_set_id,
            edge_ids,
            tuple(entity_id(revision, "edge", int(edge)) for edge in sources.tolist()),
            edge_deviations[carried],
            exact=bool(np.all(edge_deviations[carried] == 0.0)),
            orientations=orientations,
            source_dimensions=np.full(edge_ids.size, 1, dtype=np.int8),
            source_indices=sources,
            source_entity_roles=(GeometrySourceEntityRole.EDGE,) * edge_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(0).entity_set_id,
            mesh.vertex_global_ids,
            tuple(
                entity_id(revision, ("vertex", "edge", "region")[int(kind)], int(index))
                for kind, index in zip(dimensions, indices, strict=True)
            ),
            distances,
            resolved=np.ones(dimensions.shape, dtype=np.bool_),
            ambiguous=np.zeros(dimensions.shape, dtype=np.bool_),
            exact=bool(np.all(distances == 0.0)),
            source_dimensions=dimensions,
            source_indices=indices,
            parameters=parameters,
            source_entity_roles=tuple(roles[int(kind)] for kind in dimensions),
        ),
    )


@checked
def publish_distributed_planar_initial(
    prepared: planar_generation_owner.PreparedDistributedPlanarGeneration,
    generated: planar_generation_owner.DistributedPlanarConstruction,
    coordinate_contract: SpatialCoordinateContract,
    provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo,
    /,
    *,
    minimum_angle: float,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    plan_id: str | None = None,
) -> _owner_phydrax_meshing__result.CellMeshingResult:
    """Accept the planar tile theorem collectively, then lower owner-local results."""
    evidence = InitialCollectiveMeshEvidence(
        prepared, generated, minimum_angle=minimum_angle, limits=limits
    )
    premise = evidence.premise
    if not isinstance(premise, InitialPlanarPremise):
        raise TypeError(
            "initial.planar_publication must retain a planar premise; got another premise"
        )
    projection = _owner_phydrax_meshing__publication_lowering.PreparedPublicationLowering(
        None, evidence, None, vertex_capacity=prepared.layout.vertex_capacity
    ).execute()
    projection.require_source(evidence.logical_arrays)
    storage_binding = _owner_phydrax_meshing__result.CollectiveMeshStorageBinding(
        evidence, evidence.logical_arrays, evidence.global_entity_counts
    )
    scopes = _owner_phydrax_meshing__organization.prepare_mesh_organization_scopes(
        evidence, evidence, projection, premise.source_revision
    )
    receipts = {
        name: np.asarray(jax.device_get(value))
        for name, value in projection.addressable_arrays(jax.process_index())
    }
    _, _, loops = source_arrays(premise.authored_source)
    edge_mask = receipts["entity/1/valid"]
    curves = receipts["initial/edge_curves"][edge_mask]
    region_edges = (curves >= 0) & (loops[np.maximum(curves, 0)] >= 0)
    region_distance = float(
        np.max(
            _process_values(
                np.float64(
                    np.max(
                        receipts["initial/edge_deviations"][edge_mask][region_edges],
                        initial=0.0,
                    )
                )
            )
        )
    )
    vertex_mask = receipts["entity/0/valid"]
    owned = vertex_mask & (receipts["entity/0/owners"] == jax.process_index())
    owned_residual = float(
        np.max(
            np.where(
                receipts["initial/source_dimensions"][owned] < 2,
                receipts["initial/source_residuals"][owned],
                0.0,
            ),
            initial=0.0,
        )
    )
    compliance = _planar_distributed_compliance(prepared, generated, owned_residual)
    result = None
    failure = None
    try:
        mesh = _lower_initial_mesh(evidence, receipts, projection.geometry)
        if mesh.storage is None:
            raise ValueError("initial.mesh.storage must retain logical storage; got None")
        geometry = mesh.storage.restore_geometry()
        associations = (
            _planar_associations(premise, mesh, receipts, region_distance)
            if np.any(receipts["closure/cell_valid"])
            else ()
        )
        patches, zones, labels, attributes = (
            _owner_phydrax_meshing__organization.lower_mesh_organization(
                evidence, mesh, dict(evidence.logical_arrays), scope_projections=scopes
            )
        )
        audit = _owner_phydrax_meshing__audit.audit_cell_mesh(
            mesh,
            geometry,
            policy=_owner_phydrax_meshing__audit.CellMeshAuditPolicy(
                require_complete_association=True
            ),
            patches=patches,
            labels=labels,
            zones=zones,
            attributes=attributes,
            associations=associations,
        )
        audit.require_passed()
        audit.require_decided()
        trace = _owner_phydrax_meshing__trace.MeshingTrace(
            (
                _owner_phydrax_meshing__trace.MeshingStageReport(
                    _owner_phydrax_meshing__trace.MeshingStageKind.SURFACE_MESHING,
                    _owner_phydrax_meshing__trace.MeshingStageStatus.PASSED,
                    input_ids=(
                        premise.generation_id,
                        premise.specification.specification_id,
                    ),
                    output_ids=(evidence.evidence_id, audit.report_id),
                    created_count=generated.global_cell_count,
                ),
            )
        )
        result = _owner_phydrax_meshing__result.CellMeshingResult(
            mesh,
            geometry,
            coordinate_contract,
            audit,
            audit.quality,
            compliance,
            trace,
            provider,
            _owner_phydrax_meshing__result.MeshingRuntimeInfo(
                provider.provider_id,
                provider.version,
                _owner_phydrax_meshing__contracts.MeshingExecutionMode.IN_PROCESS,
                deterministic=True,
                enforced_limits=(
                    "source_theorem",
                    "collective_acceptance",
                    "fixed_capacity_closure",
                    "worker_count_independent_tiles",
                ),
            ),
            _owner_phydrax_meshing__contracts.MeshingDerivativeMode.NONDIFFERENTIABLE,
            _owner_phydrax__identity.SemanticProvenance(
                {
                    "kind": "native-distributed-planar-initial-publication",
                    "source": premise.authored_source.binding_id,
                    "tiles": premise.generation_id,
                    "theorem": evidence.evidence_id,
                    "plan": plan_id,
                }
            ),
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            region_boundary_evidence=(),
            collective_evidence=evidence,
            scope_projections=scopes,
            storage_binding=storage_binding,
        )
    except (
        MeshcoreError,
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
    ) as error:
        failure = _dependency_phydrax_meshing__distributed_generation.initial_owner_failure(
            error, "distributed-initial-publication"
        )
    except _leaf_phydrax_meshing__failures.MeshingFailure as error:
        failure = error
    except ValueError as error:
        failure = compliance_failure(str(error), stage="distributed-initial-certification")
    collective_failure_barrier(
        failure,
        message="An owner rejected owner-local planar publication; no generated tile was published.",
        stage="distributed-initial-commit",
    )
    if result is None:
        raise ValueError(
            "initial.publication.result must be present on every partition; got None"
        )
    return result


__all__ = [
    "InitialCollectiveMeshEvidence",
    "InitialMeshPremise",
    "InitialPlanarPremise",
    "InitialSurfacePremise",
    "publish_distributed_planar_initial",
    "publish_distributed_surface_initial",
]
