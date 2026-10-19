# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import resource
import socket
import sys
from collections.abc import (
    Callable,
)
from dataclasses import (
    dataclass,
    replace,
)
from pathlib import (
    Path,
)
from time import (
    perf_counter,
)
from typing import (
    Any,
    final,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import (
    multihost_utils,
)
from jax.sharding import (
    PartitionSpec,
)

import phydrax as phx
from phydrax import (
    ArrayArchiveLimits,
    canonical_fingerprint,
)
from phydrax.discretization import (
    AdaptiveSimplexPolicy,
    CellMeshStorage,
    coarsen_adaptive_simplex_parts,
    FiniteElementTopologyTransfer,
    make_owner_local_field_array,
    prepare_owner_local_halo_plans,
    refine_adaptive_simplex_parts,
)
from phydrax.discretization.fem import (
    FiniteElementPlan,
    FiniteElementPrecisionPolicy,
)
from phydrax.discretization.finite_volume import (
    execute_owner_local_finite_volume_advance,
    PreparedUnstructuredFiniteVolumeDynamics,
)
from phydrax.execution import (
    ExecutionGroup,
    ExecutionGroupSpec,
    ExecutionRuntime,
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
)
from phydrax.lifecycle import (
    assemble_distributed_checkpoint_from_repository,
    HPCFilesystemProfile,
    MeshingCheckpointState,
    MeshingFieldIndexBank,
    MeshingFieldStateKind,
    MeshingFieldStateRole,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
    prepare_meshing_field_index_binding,
    prepare_meshing_field_owner,
    project_meshing_field_history,
    publish_process_meshing_checkpoint,
    read_meshing_checkpoint_states,
    read_meshing_source_closure,
    rebind_meshing_field_history,
    recertify_restored_meshing_source,
    recover_meshing_field_history,
    register_meshing_source_artifacts,
    restore_meshing_checkpoint,
    TopologyRestartPolicy,
    TopologyRestartRelation,
    validate_meshing_field_declarations,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    CollectiveMeshEvidence,
    commit_partitioned_adaptive_simplex,
    commit_partitioned_mesh_restart,
    NativeMeshingPhaseMeasurement,
    PlcAssociationTransfer,
    prepare_graph_restart_proposal,
    prepare_ownerlocal_migration_state,
    prepare_partitioned_mesh_adaptation,
    repack_simplex_restart,
    restore_partitioned_mesh_state,
    SimplexGraphRestartProposal,
    SimplexRestartRepack,
    SimplexRestartRepackProof,
)
from phydrax.meshing.qualification import (
    bind_distributed_checkpoint_source_arrays,
    distributed_checkpoint_host_array_names,
    distributed_field_templates,
    prepare_distributed_checkpoint_mesh_arrays,
    prepare_distributed_checkpoint_source_closure,
    rebind_distributed_checkpoint_field_declaration,
)
from phydrax.typing import parse
from tools.meshing_qualification._controls import (
    _DISTRIBUTED_ARCHIVE_LIMITS,
    _DISTRIBUTED_FIELD_ROLES,
    _DISTRIBUTED_GRAPH_IMBALANCE,
    _distributed_qualification_requirements,
    _DISTRIBUTED_REPOSITORY_POLICY,
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _result_identity,
    FailureRecord,
)
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.metric_adaptation import (
    _mesh_triangles,
)
from tools.meshing_qualification.scenarios.triangulation import (
    _simplex_measures,
)


type _DistributedFields = dict[str, jax.Array | np.ndarray]


_DISTRIBUTED_PART = "native-unit-square"


def _distributed_qualification_topology(args: QualificationControls) -> dict[str, object]:
    """Refuse unavailable real placements before any scientific work begins."""

    requirements = _distributed_qualification_requirements()["required_arguments"]
    for name, rule in requirements.items():
        value = args.to_record()[name]
        if "minimum" in rule and (type(value) is not int or value < rule["minimum"]):
            raise ValueError(
                f"_distributed_qualification_topology.{name} must be an integer at least the declared minimum; got {value!r}."
            )
        if "finite" in rule and (
            not np.isfinite(value) or value <= rule["exclusive_minimum"]
        ):
            raise ValueError(
                f"_distributed_qualification_topology.{name} must be finite and greater than the declared exclusive minimum; got {value!r}."
            )
        if "choices" in rule and value not in rule["choices"]:
            raise ValueError(
                f"_distributed_qualification_topology.{name} must be an admitted frozen choice; got {value!r}."
            )
        if rule.get("nullable") is False and value is None:
            raise ValueError(
                f"_distributed_qualification_topology.{name} must be non-null for multiple-process qualification; got {value!r}."
            )
    if args.distributed_processes != args.distributed_parts:
        raise ValueError(
            f"_distributed_qualification_topology.distributed_processes must equal distributed_parts; got {(args.distributed_processes, args.distributed_parts)!r}."
        )
    local = tuple(jax.local_devices())
    world = tuple(jax.devices())
    match jax.process_count():
        case args.distributed_processes:
            role = "producer"
        case args.restart_parts:
            role = "restore"
        case _:
            raise ValueError(
                f"_distributed_qualification_topology.process_count must equal the requested producer or restart count; got {jax.process_count()!r}."
            )
    if len(local) != 1:
        raise ValueError(
            f"_distributed_qualification_topology.local_devices must contain one device per process; got {len(local)!r}."
        )
    devices = [
        {
            "id": device.id,
            "process_index": device.process_index,
            "platform": device.platform,
            "kind": device.device_kind,
        }
        for device in world
    ]
    return {
        "role": role,
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "global_device_count": jax.device_count(),
        "local_device_count": len(local),
        "distributed_parts": args.distributed_parts,
        "restart_parts": args.restart_parts,
        "generation_placement": "one-native-host-provider-result-per-producer-process",
        "source_devices": devices if role == "producer" else None,
        "restart_devices": devices if role == "restore" else None,
        "simulated_topology": False,
    }


def _distributed_qualification_source(
    args: QualificationControls,
) -> tuple[Any, Any, Any]:
    """Author a bounded native source; never replace provider output by a fixture mesh."""
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        ((0, 1, 2, 3),),
        feature_id="native-unit-square",
    )
    revision = canonical_fingerprint(
        {
            "kind": "distributed-authored-planar-source",
            "vertices": np.asarray(region.vertices).tolist(),
            "loops": [[0, 1, 2, 3]],
            "coordinate_contract": phx.SpatialCoordinateContract.si().spatial_id,
        }
    )
    source = phx.meshing.NativePlanarSource(region, revision)
    scope = phx.meshing.MeshingScope(
        "native-unit-square",
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "native-unit-square-region",
        np.asarray((0,), dtype=np.int64),
    )
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        edges=8,
        faces=8,
        cells=8,
        connectivity_entries=64,
        data_bytes=1024,
        work_units=32,
        cavity_cells_limit=args.distributed_cavity_capacity,
        geometry_queries=64,
        scratch_bytes=4096,
    )
    size = 1.0 / args.resolution
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=np.radians(25.0), hard=True
        ),
        limits=limits,
    )
    return (
        source,
        request,
        phx.meshing.NativeMeshingOptions("planar_constrained_delaunay"),
    )


def _distributed_qualification_generate(
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    started: float,
) -> _DistributedGeneratedSource:
    """Freeze one authored source/request/options and consume exactly its accepted provider result."""

    source, request, options = StageLedger(
        record, deadline=Deadline(started + args.timeout)
    ).run("source_preparation", lambda: _distributed_qualification_source(args))
    record["source"] = {
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "request_id": request.specification_id,
        "coordinate_contract_id": phx.SpatialCoordinateContract.si().spatial_id,
        "control_ids": [control.control_id for control in request.size_controls],
        "quality_target_id": request.quality_target.target_id,
        "limits_id": request.limits.limits_id,
    }
    provider = phx.meshing.NativeMeshingProvider(options)
    plan = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "geometry_constraint_preparation",
        lambda: provider.plan(
            source,
            request,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            record_phase=phases.append,
        ),
    )
    result = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "native_generation_publication_certification",
        lambda: plan.execute(record_phase=phases.append),
    )
    result.audit.require_passed()
    if not result.compliance.passed or result.certification is None:
        raise qualification_failure(
            "_distributed_qualification_generate",
            ("Native distributed source lacks source/control acceptance."),
            {"result": result},
        )
    result.certification.require_passed()
    cells = _mesh_triangles(result.mesh)
    coordinates = np.asarray(result.mesh.coordinates)
    measures = _simplex_measures(coordinates, cells)
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError(
            f"_distributed_qualification_generate.connectivity must be PolygonalConnectivity; got {type(connectivity).__name__!r}."
        )
    edge_rows = np.asarray(connectivity.boundary_edges)
    edges = np.asarray(connectivity.edges)[edge_rows]
    perimeter = float(
        np.sum(
            np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
        )
    )
    area = float(np.sum(measures))
    if (
        np.any(measures <= 0.0)
        or abs(area - 1.0) > 1.0e-10
        or abs(perimeter - 4.0) > 1.0e-10
    ):
        raise qualification_failure(
            "_distributed_qualification_generate",
            (
                "Provider source disagrees with independent unit-square area/orientation/boundary."
            ),
            {"area": area, "measures": measures, "perimeter": perimeter},
        )
    record["generation"] = {
        **_result_identity(result),
        "plan_id": plan.plan_id,
        "certification_id": result.certification.report_id,
        "independent_area": area,
        "independent_perimeter": perimeter,
    }
    return _DistributedGeneratedSource(
        author=source,
        request=request,
        options=options,
        source=result,
        certification=result.certification,
    )


def _distributed_qualification_entity_cells(mesh: Any, /) -> np.ndarray:
    """Cell vertex rows in the mesh's cell entity order across all blocks."""
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    vertices = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    order = np.argsort(ids, kind="stable")
    entity_ids = np.asarray(
        mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
    )
    return vertices[order[np.searchsorted(ids[order], entity_ids)]]


def _distributed_qualification_fe_declaration() -> tuple[tuple[Any, ...], Any]:
    """Declare the one physical P1 temperature field and its precision policy."""

    return (
        tuple(field for _, field in distributed_field_templates()),
        FiniteElementPrecisionPolicy(),
    )


def _distributed_qualification_fe(
    carrier: Any,
    fields: tuple[Any, ...],
    precision: Any,
    numeric_version: str,
) -> Any:
    """Prepare the declared physical field space on one accepted carrier."""

    return FiniteElementPlan(
        carrier.mesh,
        fields,
        coordinate_spec=carrier.geometry,
        precision_policy=precision,
    ).prepare(numeric_version=numeric_version)


def _distributed_qualification_policy(
    policy: Any, distribution: Any, partition: Any
) -> Any:
    """Rebind one frozen adaptation policy to an accepted distribution and partition request."""
    return phx.meshing.MeshAdaptationPolicy(
        policy.route,
        protected_scopes=policy.protected_scopes,
        compatibility=policy.compatibility,
        predicate_mode=policy.predicate_mode,
        maximum_passes=policy.maximum_passes,
        maximum_closure_iterations=policy.maximum_closure_iterations,
        relocation=policy.relocation,
        limits=policy.limits,
        audit_policy=policy.audit_policy,
        device_policy=policy.device_policy,
        provider=policy.provider,
        provider_options=policy.provider_options,
        distribution=distribution,
        partition_policy=partition,
        association_transfer=policy.association_transfer,
        geometry_transition=policy.geometry_transition,
    )


def _distributed_qualification_histories(
    fields: _DistributedFields,
    association: str,
) -> tuple[jax.Array, ...]:
    """Allocated history banks of one association in their repack history-index order."""
    rows = sorted(
        (row for row in _DISTRIBUTED_FIELD_ROLES if row[5] == association),
        key=lambda row: row[6],
    )
    return tuple(jnp.asarray(fields[row[1]]) for row in rows)


def _distributed_qualification_project(
    carrier: Any,
    values: _DistributedFields,
    baselines: _DistributedFields,
    fe_fields: tuple[Any, ...],
    precision: Any,
    mesh_epoch: int,
) -> tuple[Any, tuple[Any, ...], _DistributedFields]:
    """Project every compact physical role into its stable-ID allocated history bank."""

    physical = _distributed_qualification_fe(
        carrier, fe_fields, precision, f"accepted-{mesh_epoch}"
    )
    space = physical.field_spaces[0]
    material = values["material-temperature"]
    compact_values = {
        **values,
        "essential-mask": physical.dof_maps[0].boundary_dof_mask,
        "material-cell-average": jnp.mean(
            material[jnp.asarray(_distributed_qualification_entity_cells(carrier.mesh))],
            axis=1,
        ),
    }
    roles: list[Any] = []
    fields: _DistributedFields = {
        "accepted-epoch": jnp.asarray(mesh_epoch, dtype=jnp.int64)
    }
    for (
        name,
        raw_name,
        kind,
        raw_kind,
        bank,
        association,
        _,
        indexed,
        _,
        units,
    ) in _DISTRIBUTED_FIELD_ROLES:
        value = compact_values[name]
        field_space = space if indexed else None
        value_units = None if units is None else (phx.units.KELVIN,)
        compact_index = prepare_meshing_field_index_binding(
            carrier,
            parse(bank, MeshingFieldIndexBank, "bank"),
            field_space=field_space,
        )
        raw_index = prepare_meshing_field_index_binding(
            carrier,
            parse(f"epoch/{association}_ids", MeshingFieldIndexBank, "bank"),
        )
        compact = MeshingFieldStateRole(
            name,
            "temperature",
            parse(kind, MeshingFieldStateKind, "role"),
            value.shape,
            np.dtype(value.dtype),
            mesh_epoch,
            value_units=value_units,
            index_binding=compact_index,
        )
        raw = MeshingFieldStateRole(
            raw_name,
            "temperature",
            parse(raw_kind, MeshingFieldStateKind, "role"),
            raw_index.index_values.shape,
            np.dtype(value.dtype),
            mesh_epoch,
            value_units=value_units,
            index_binding=raw_index,
        )
        baseline = baselines.get(raw_name)
        if baseline is None:
            index_values = raw_index.index_values
            if not isinstance(index_values, jax.Array):
                raise TypeError(
                    f"_distributed_qualification_project.index_values must remain a placed JAX array; got {type(index_values).__name__!r}."
                )
            baseline = jnp.zeros(raw.shape, dtype=raw.dtype, device=index_values.sharding)
        projected = project_meshing_field_history(
            compact,
            value,
            raw,
            baseline,
            carrier,
            field_space=field_space,
            index_source=carrier,
        )
        baselines[raw_name] = projected
        roles.extend((compact, raw))
        fields[name] = value
        fields[raw_name] = projected
    roles.append(
        MeshingFieldStateRole(
            "accepted-epoch",
            "temperature",
            "state-epoch",
            (),
            np.dtype(np.int64),
            mesh_epoch,
        )
    )
    return physical, tuple(roles), fields


def _distributed_qualification_recover(
    carrier: Any,
    receipt: Any,
    roles: tuple[Any, ...],
    fields: _DistributedFields,
    space: Any,
    proof: SimplexRestartRepackProof,
) -> tuple[tuple[Any, ...], _DistributedFields]:
    """Rebind allocated histories through one repack transfer record and recover every compact role."""

    by_name = {role.entry_name: role for role in roles}
    recovered_roles: list[Any] = []
    recovered: _DistributedFields = {"accepted-epoch": fields["accepted-epoch"]}
    for (
        name,
        raw_name,
        _,
        _,
        bank,
        _,
        history_index,
        indexed,
        _,
        _,
    ) in _DISTRIBUTED_FIELD_ROLES:
        field_space = space if indexed else None
        index = prepare_meshing_field_index_binding(
            carrier,
            parse(bank, MeshingFieldIndexBank, "bank"),
            field_space=field_space,
        )
        compact = replace(
            by_name[name], shape=index.index_values.shape, index_binding=index
        )
        raw, payload = rebind_meshing_field_history(
            by_name[raw_name],
            fields[raw_name],
            carrier,
            receipt,
            history_index=history_index,
            proof=proof,
        )
        recovered[name] = recover_meshing_field_history(
            raw,
            payload,
            compact,
            carrier,
            receipt,
            history_index=history_index,
            field_space=field_space,
            proof=proof,
        )
        recovered[raw_name] = payload
        recovered_roles.extend((compact, raw))
    recovered_roles.append(by_name["accepted-epoch"])
    return tuple(recovered_roles), recovered


def _distributed_qualification_repository(root: Path, /) -> tuple[Any, Any]:
    """Open the shared accepted-checkpoint repository under its frozen limits."""

    profile = HPCFilesystemProfile(
        "distributed-qualification-posix",
        "declared-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    repository = POSIXArtifactRepository(
        root / "repository",
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=_DISTRIBUTED_REPOSITORY_POLICY["maximum_chunk_bytes"],
            maximum_metadata_bytes=_DISTRIBUTED_REPOSITORY_POLICY[
                "maximum_metadata_bytes"
            ],
        ),
    )
    return repository, ArrayArchiveLimits(
        max_members=_DISTRIBUTED_ARCHIVE_LIMITS["max_members"],
        max_manifest_nesting=_DISTRIBUTED_ARCHIVE_LIMITS["max_manifest_nesting"],
        max_manifest_bytes=_DISTRIBUTED_ARCHIVE_LIMITS["max_manifest_bytes"],
        max_central_directory_bytes=_DISTRIBUTED_ARCHIVE_LIMITS[
            "max_central_directory_bytes"
        ],
    )


def _distributed_qualification_refuses(call: Callable[[], object], /) -> bool:
    """Check whether the transfer proof refuses a deliberately tampered transfer record."""
    try:
        call()
    except ValueError:
        return True
    return False


@final
@dataclass(frozen=True, slots=True)
class _DistributedGeneratedSource:
    """Accepted native source with the certification established at provider publication."""

    author: phx.meshing.NativePlanarSource
    request: phx.meshing.SurfaceMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    source: phx.meshing.CellMeshingResult
    certification: phx.meshing.MeshCertificationReport


@final
@dataclass(frozen=True, slots=True)
class _DistributedEpochState:
    """Accepted physical roles and raw allocator state after the topology epochs."""

    current: phx.meshing.CellMeshingResult
    distribution: phx.meshing.MeshDistribution
    policy: phx.meshing.MeshAdaptationPolicy
    partition: phx.meshing.MeshPartitionPolicy
    roles: tuple[MeshingFieldStateRole, ...]
    fields: _DistributedFields
    raw_layout: phx.discretization.AdaptiveSimplexLayout
    raw_state: phx.discretization.AdaptiveSimplexState


@final
@dataclass(frozen=True, slots=True)
class _DistributedAcceptedEpoch:
    """One accepted topology publication with its stable raw history-slot identities."""

    result: phx.meshing.MeshAdaptationResult
    current: phx.meshing.CellMeshingResult
    distribution: phx.meshing.MeshDistribution
    values: _DistributedFields
    raw_layout: phx.discretization.AdaptiveSimplexLayout
    raw_state: phx.discretization.AdaptiveSimplexState
    ids: tuple[jax.Array, jax.Array]
    evidence: phx.meshing.CollectiveMeshEvidence
    storage: CellMeshStorage
    transfer: FiniteElementTopologyTransfer


@final
@dataclass(frozen=True, slots=True)
class _DistributedGraphPreparation:
    """Prepared graph epoch, repacked history, and consumed restart proof."""

    adaptation: phx.meshing.PreparedMeshAdaptation
    partitioned: phx.meshing.PartitionedAdaptiveSimplex
    proposal: SimplexGraphRestartProposal
    receipt: SimplexRestartRepack
    proof: SimplexRestartRepackProof


@final
@dataclass(frozen=True, slots=True)
class _DistributedGraphState:
    """Accepted graph results and recovered physical roles for checkpoint publication."""

    results: tuple[phx.meshing.MeshAdaptationResult, ...]
    physical: phx.discretization.FiniteElementDiscretization
    roles: tuple[MeshingFieldStateRole, ...]
    fields: _DistributedFields
    parts: phx.discretization.AdaptiveSimplexParts
    errors: dict[str, float]
    transition: phx.meshing.MeshDistributionTransition
    migration_counts: jax.Array


@final
@dataclass(frozen=True, slots=True)
class _DistributedCheckpointBinding:
    """Scientific source closure and its named shared-array aliases."""

    result: phx.meshing.MeshAdaptationResult
    target: phx.meshing.CellMeshingResult
    closure: dict[str, Any]
    repository: phx.lifecycle.POSIXArtifactRepository
    limits: phx.ArrayArchiveLimits
    bindings: dict[str, str]
    bank: dict[str, jax.Array]


@final
@dataclass(frozen=True, slots=True)
class _DistributedCheckpointPayload:
    """Bound scientific checkpoint and the arrays published by this process."""

    state: phx.lifecycle.MeshingCheckpointState
    arrays: dict[str, jax.Array]


@final
@dataclass(frozen=True, slots=True)
class _DistributedRestoredFields:
    """Decoded accepted fields with their independently checked physical references."""

    target: phx.meshing.CellMeshingResult
    fields: _DistributedFields
    final: phx.discretization.FiniteElementDiscretization
    expected: jax.Array
    cells: np.ndarray
    errors: dict[str, float]
    mask_exact: bool
    archive_content_id: str
    archive_path: Path
    reopened_roles: tuple[str, ...]


@final
@dataclass(frozen=True, slots=True)
class _DistributedFEContinuation:
    """Continued FE solve measurements consumed by inventory and resource reporting."""

    solve_error: float
    free: int
    warm: list[float]


@final
@dataclass(frozen=True, slots=True)
class _DistributedInventories:
    """Independent accepted integrals and their analytic normalization references."""

    integrals: dict[str, float]
    references: dict[str, float]


def _distributed_accept_topology_epoch(
    args: QualificationControls,
    current: phx.meshing.CellMeshingResult,
    values: _DistributedFields,
    policy: phx.meshing.MeshAdaptationPolicy,
    distribution: phx.meshing.MeshDistribution,
    partition: phx.meshing.MeshPartitionPolicy,
    previous_ids: tuple[jax.Array, jax.Array] | None,
    record: dict[str, Any],
    started: float,
    epoch: int,
    operation: str,
) -> _DistributedAcceptedEpoch:
    """Execute and accept one owner-local edit while preserving raw history identities."""

    meshing = phx.meshing
    partitioned = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        f"owner_local_preparation_{epoch}",
        lambda: prepare_partitioned_mesh_adaptation(
            meshing.prepare_mesh_adaptation(
                current,
                meshing.MarkedMeshAdaptation(),
                policy=_distributed_qualification_policy(policy, distribution, partition),
            )
        ),
    )
    states = partitioned.states
    # Every active cell is marked: retained Morton ownership stays within its balance
    # tolerance only because each part refines or coarsens uniformly.
    operation_kernel = (
        coarsen_adaptive_simplex_parts
        if operation == "coarsen"
        else refine_adaptive_simplex_parts
    )
    kernel = lambda: operation_kernel(
        partitioned.layout, partitioned.parts, states, states.mesh.cell_active
    )
    updated = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        f"owner_local_{operation}_{epoch}", lambda: jax.block_until_ready(kernel())
    )
    results = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        f"owner_local_publication_{epoch}",
        lambda: (
            commit_partitioned_adaptive_simplex(partitioned, updated.state).local_results
        ),
    )
    if (
        len(results) != 1
        or results[0].status is not meshing.MeshAdaptationStatus.COMPLETE
    ):
        raise qualification_failure(
            "_distributed_qualification_epochs",
            (f"owner-local {operation} epoch {epoch} was not accepted by this process."),
            {"meshing": meshing, "results": results},
        )
    result = results[0]
    transfer, transition, current = (
        result.transfer,
        result.distribution,
        result.target,
    )
    storage, evidence = current.mesh.storage, current.collective_evidence
    if (
        transfer is None
        or transition is None
        or storage is None
        or not isinstance(evidence, CollectiveMeshEvidence)
    ):
        raise qualification_failure(
            "_distributed_qualification_epochs",
            (
                "Accepted owner-local publication omitted its transfer, distribution, storage, or proof."
            ),
            {
                "CollectiveMeshEvidence": CollectiveMeshEvidence,
                "evidence": evidence,
                "storage": storage,
                "transfer": transfer,
                "transition": transition,
            },
        )
    values = {name: transfer.apply(value) for name, value in values.items()}
    distribution = transition.target
    evidence.require_passed()
    current.audit.require_passed()
    raw_layout, raw_state, _ = restore_partitioned_mesh_state(storage, evidence)
    ids = (raw_state.mesh.vertex_ids, raw_state.mesh.cell_ids)
    if previous_ids is not None and any(
        old.shape != new.shape
        or not bool(jax.device_get(jnp.all((old < 0) | (old == new))))
        for old, new in zip(previous_ids, ids, strict=True)
    ):
        raise qualification_failure(
            "_distributed_qualification_epochs",
            (
                "A topology epoch changed the identity meaning of retained raw history slots."
            ),
            {"ids": ids, "previous_ids": previous_ids},
        )
    return _DistributedAcceptedEpoch(
        result=result,
        current=current,
        distribution=distribution,
        values=values,
        raw_layout=raw_layout,
        raw_state=raw_state,
        ids=ids,
        evidence=evidence,
        storage=storage,
        transfer=transfer,
    )


def _distributed_project_topology_epoch(
    accepted: _DistributedAcceptedEpoch,
    baselines: _DistributedFields,
    fe_fields: tuple[phx.discretization.FiniteElementFieldSpec, ...],
    precision: phx.discretization.FiniteElementPrecisionPolicy,
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
    epoch: int,
    operation: str,
) -> tuple[tuple[MeshingFieldStateRole, ...], _DistributedFields]:
    """Project every physical role and record its accepted affine-transfer evidence."""

    current, result = accepted.current, accepted.result
    values, distribution, raw_state = (
        accepted.values,
        accepted.distribution,
        accepted.raw_state,
    )
    evidence, storage, transfer = accepted.evidence, accepted.storage, accepted.transfer
    physical, roles, fields = StageLedger(
        record, deadline=Deadline(started + args.timeout)
    ).run(
        f"all_field_history_projection_{epoch}",
        lambda: _distributed_qualification_project(
            current, values, baselines, fe_fields, precision, mesh_epoch
        ),
    )
    dofs = physical.dof_maps[0].dof_coordinates
    error = float(
        jnp.max(jnp.abs(values["temperature"] - (1.0 + dofs[:, 0] + 2.0 * dofs[:, 1])))
    )
    if error > 1.0e-10:
        raise qualification_failure(
            "_distributed_qualification_epochs",
            ("Accepted H1 transfer lost the exactly representable affine temperature."),
            {"error": error},
        )
    record["epochs"].append(
        {
            "epoch": epoch,
            "operation": operation,
            "status": result.status.value,
            "result_id": current.result_id,
            "source_result_id": result.source.result_id,
            "topology_id": current.mesh.topology_id,
            "evidence_id": evidence.evidence_id,
            "lineage_id": None if result.lineage is None else result.lineage.lineage_id,
            "transfer_id": transfer.transfer_id,
            "distribution_kind": distribution.evidence.kind.value,
            "local_cells": current.mesh.entity_set(2).count,
            "global_cells": storage.global_entity_counts[-1],
            "h1_affine_transfer_error": error,
            "accepted_cross_epoch": True,
            "allocator_cursors_local": np.asarray(
                raw_state.cursors.addressable_shards[0].data
            ).tolist(),
            "allocator_clocks_local": np.asarray(
                raw_state.clocks.addressable_shards[0].data
            ).tolist(),
            "allocator_counters_local": np.asarray(
                raw_state.counters.addressable_shards[0].data
            ).tolist(),
            "field_roles": [role.entry_name for role in roles],
        }
    )
    return roles, fields


def _distributed_qualification_epochs(
    args: QualificationControls,
    author: Any,
    request: Any,
    source: Any,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedEpochState:
    """Accept native owner-local refine/coarsen epochs and project every physical role."""

    meshing = phx.meshing
    domain = source.certification.request.domain
    association = PlcAssociationTransfer(
        domain,
        source.coordinate_contract,
        author.source_revision,
        edge_vertices=np.asarray(author.region.edges, dtype=np.int64),
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=np.asarray(domain.facet_regions, dtype=np.int64),
    )
    partition = meshing.MeshPartitionPolicy(
        meshing.MeshPartitionKind.MORTON,
        args.distributed_parts,
        halo_width=args.distributed_halo_width,
    )
    distribution = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "native_morton_partition",
        lambda: meshing.prepare_mesh_distribution(
            meshing.MeshPart(_DISTRIBUTED_PART, source), policy=partition
        ),
    )
    # Each refine epoch bisects every marked cell at most once.
    growth = 2 ** (args.adaptation_rounds + 1)
    cells = sum(block.cell_count for block in source.mesh.blocks)
    policy = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        compatibility=args.distributed_bisection_compatibility,
        device_policy=AdaptiveSimplexPolicy(
            vertex_capacity=min(
                args.capacity,
                args.distributed_slack * growth * source.mesh.coordinates.shape[0],
            ),
            cell_capacity=min(
                args.capacity,
                args.distributed_cavity_capacity,
                args.distributed_slack * growth * cells,
            ),
        ),
        distribution=distribution,
        partition_policy=partition,
        association_transfer=association,
        audit_policy=meshing.CellMeshAuditPolicy(
            watertight_boundary=meshing.CellMeshAuditDisposition.REJECT
        ),
        limits=request.limits,
    )
    fe_fields, precision = _distributed_qualification_fe_declaration()
    points = (
        _distributed_qualification_fe(
            source, fe_fields, precision, f"accepted-{mesh_epoch}"
        )
        .dof_maps[0]
        .dof_coordinates
    )
    temperature = 1.0 + points[:, 0] + 2.0 * points[:, 1]
    values: _DistributedFields = {
        "temperature": temperature,
        "material-temperature": 37.0 + 3.0 * points[:, 0] - points[:, 1],
        "essential-values": temperature,
    }
    current, baselines = source, {}
    previous_ids: tuple[jax.Array, jax.Array] | None = None
    for epoch, operation in enumerate(
        ("refine", *(("coarsen", "refine") * args.adaptation_rounds))
    ):
        accepted = _distributed_accept_topology_epoch(
            args,
            current,
            values,
            policy,
            distribution,
            partition,
            previous_ids,
            record,
            started,
            epoch,
            operation,
        )
        current, distribution, values = (
            accepted.current,
            accepted.distribution,
            accepted.values,
        )
        previous_ids = accepted.ids
        roles, fields = _distributed_project_topology_epoch(
            accepted,
            baselines,
            fe_fields,
            precision,
            args,
            record,
            started,
            mesh_epoch,
            epoch,
            operation,
        )
    return _DistributedEpochState(
        current=current,
        distribution=distribution,
        policy=policy,
        partition=partition,
        roles=roles,
        fields=fields,
        raw_layout=accepted.raw_layout,
        raw_state=accepted.raw_state,
    )


def _distributed_prepare_graph_epoch(
    args: QualificationControls,
    accepted: _DistributedEpochState,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedGraphPreparation:
    """Bind physical history to the forest and execute the proved graph restart repack."""

    current, distribution, policy, partition = (
        accepted.current,
        accepted.distribution,
        accepted.policy,
        accepted.partition,
    )
    roles, fields, raw_layout, raw_state = (
        accepted.roles,
        accepted.fields,
        accepted.raw_layout,
        accepted.raw_state,
    )
    meshing = phx.meshing
    graph = meshing.MeshPartitionPolicy(
        meshing.MeshPartitionKind.GRAPH,
        args.distributed_parts,
        halo_width=partition.halo_width,
        maximum_imbalance=_DISTRIBUTED_GRAPH_IMBALANCE,
        curve_depth=partition.curve_depth,
        metis_seed=partition.metis_seed,
    )
    adaptation = meshing.prepare_mesh_adaptation(
        current,
        meshing.MarkedMeshAdaptation(),
        policy=_distributed_qualification_policy(policy, distribution, graph),
    )
    partitioned = prepare_partitioned_mesh_adaptation(adaptation)
    devices = tuple(partitioned.parts.mesh.devices.flat)

    def bucket(
        values: jax.Array | np.ndarray, capacity: int, fill: int | float
    ) -> jax.Array:
        if values.shape[0] > capacity:
            raise qualification_failure(
                "bucket",
                ("An indexed physical field exceeds its declared native bucket."),
                {"capacity": capacity, "values": values},
            )
        local = jnp.pad(
            jnp.asarray(values), ((0, capacity - values.shape[0]),), constant_values=fill
        )
        return make_owner_local_field_array(
            (local,),
            (jax.process_index(),),
            devices,
            axis_name=partitioned.parts.axis_name,
        )

    by_name = {role.entry_name: role for role in roles}

    def bound_indices(role: MeshingFieldStateRole) -> jax.Array | np.ndarray:
        binding = role.index_binding
        if binding is None:
            raise qualification_failure(
                "_distributed_prepare_graph_epoch",
                "Distributed graph field must retain its physical index binding; got None",
                {"role": role},
            )
        return binding.index_values

    bound = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "graph_field_forest_binding",
        lambda: prepare_ownerlocal_migration_state(
            partitioned,
            raw_state,
            cell_field_ids=bucket(
                bound_indices(by_name["material-cell-average"]),
                raw_layout.cell_capacity,
                -1,
            ),
            cell_fields=bucket(
                fields["material-cell-average"], raw_layout.cell_capacity, 0.0
            ),
            vertex_field_ids=bucket(
                bound_indices(by_name["temperature"]),
                raw_layout.vertex_capacity,
                -1,
            ),
            vertex_fields=bucket(fields["temperature"], raw_layout.vertex_capacity, 0.0),
            cell_epochs=jnp.full(
                raw_state.mesh.cell_ids.shape,
                mesh_epoch,
                dtype=jnp.int64,
                device=raw_state.mesh.cell_ids.sharding,
            ),
            cell_history=_distributed_qualification_histories(fields, "cell"),
            vertex_history=_distributed_qualification_histories(fields, "vertex"),
        ),
    )
    cell_owners, vertex_owners = bound.forest_cell_owners, bound.forest_vertex_owners
    if cell_owners is None or vertex_owners is None:
        raise qualification_failure(
            "_distributed_qualification_graph",
            ("Native migration binding omitted its reassembled forest ownership."),
            {"cell_owners": cell_owners, "vertex_owners": vertex_owners},
        )
    proposal = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "native_graph_restart_proposal",
        lambda: prepare_graph_restart_proposal(
            partitioned.parts,
            raw_state,
            bound,
            graph,
            vertex_weights=jnp.where(
                bound.workset.cell_valid, 1 + bound.workset.cell_ids % 3, 0
            ).astype(jnp.int64),
        ),
    )
    receipt = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "native_graph_restart_repartition",
        lambda: repack_simplex_restart(
            ExecutionRuntime.current().root_group,
            raw_layout,
            raw_state,
            cell_owners,
            vertex_owners,
            proposal.requested_target_owners,
            cell_history=bound.cell_history,
            vertex_history=bound.vertex_history,
            source_result=current,
            graph_proposal=proposal,
        ),
    )
    proof = SimplexRestartRepackProof(receipt)
    proof.require_receipt(receipt)
    return _DistributedGraphPreparation(
        adaptation=adaptation,
        partitioned=partitioned,
        proposal=proposal,
        receipt=receipt,
        proof=proof,
    )


def _distributed_recover_graph_epoch(
    args: QualificationControls,
    accepted: _DistributedEpochState,
    prepared: _DistributedGraphPreparation,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedGraphState:
    """Accept the graph publication and verify every recovered physical role."""

    meshing = phx.meshing
    adaptation, receipt = prepared.adaptation, prepared.receipt
    roles, fields = accepted.roles, accepted.fields
    results = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "graph_restart_publication",
        lambda: commit_partitioned_mesh_restart(receipt, adaptation).local_results,
    )
    if (
        len(results) != 1
        or results[0].status is not meshing.MeshAdaptationStatus.COMPLETE
    ):
        raise qualification_failure(
            "_distributed_qualification_graph",
            ("The native graph restart epoch was not accepted by this process."),
            {"meshing": meshing, "results": results},
        )
    target, transition = results[0].target, results[0].distribution
    evidence, migration_counts = target.collective_evidence, receipt.cell_migration_counts
    if evidence is None or transition is None or migration_counts is None:
        raise qualification_failure(
            "_distributed_qualification_graph",
            (
                "Graph restart publication omitted its proof, distribution, or migration counts."
            ),
            {
                "evidence": evidence,
                "migration_counts": migration_counts,
                "transition": transition,
            },
        )
    evidence.require_passed()
    target.audit.require_passed()
    fe_fields, precision = _distributed_qualification_fe_declaration()
    physical = _distributed_qualification_fe(
        target, fe_fields, precision, f"accepted-{mesh_epoch}"
    )
    roles, recovered = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "graph_field_history_recovery",
        lambda: _distributed_qualification_recover(
            target, receipt, roles, fields, physical.field_spaces[0], prepared.proof
        ),
    )
    mask = physical.dof_maps[0].boundary_dof_mask
    if not bool(jnp.array_equal(recovered["essential-mask"], mask)):
        raise qualification_failure(
            "_distributed_qualification_graph",
            (
                "Recovered essential mask differs from the repartitioned physical boundary."
            ),
            {"mask": mask, "recovered": recovered},
        )
    if not bool(jax.device_get(jnp.any(~mask))):
        raise qualification_failure(
            "_distributed_qualification_graph",
            ("The graph epoch lost every physical interior FE unknown."),
            {"mask": mask},
        )
    points = physical.dof_maps[0].dof_coordinates
    temperature = 1.0 + points[:, 0] + 2.0 * points[:, 1]
    vertices = target.mesh.coordinates
    material = 37.0 + 3.0 * vertices[:, 0] - vertices[:, 1]
    references = {
        "temperature": temperature,
        "essential-values": temperature,
        "material-temperature": material,
        "material-cell-average": jnp.mean(
            material[jnp.asarray(_distributed_qualification_entity_cells(target.mesh))],
            axis=1,
        ),
    }
    errors = {
        name: float(jnp.max(jnp.abs(recovered[name] - reference)))
        for name, reference in references.items()
    }
    if max(errors.values()) > 1e-12:
        raise qualification_failure(
            "_distributed_qualification_graph",
            (
                "Hot graph history recovery disagrees with the independent physical references."
            ),
            {"errors": errors},
        )
    return _DistributedGraphState(
        results=results,
        physical=physical,
        roles=roles,
        fields=recovered,
        parts=prepared.partitioned.parts,
        errors=errors,
        transition=transition,
        migration_counts=migration_counts,
    )


def _distributed_qualification_graph(
    args: QualificationControls,
    accepted: _DistributedEpochState,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedGraphState:
    """Replay a native graph restart repartition and recover every physical role across it."""
    prepared = _distributed_prepare_graph_epoch(
        args, accepted, record, started, mesh_epoch
    )
    recovered = _distributed_recover_graph_epoch(
        args, accepted, prepared, record, started, mesh_epoch
    )
    proposal, receipt, proof = prepared.proposal, prepared.receipt, prepared.proof
    target, errors = recovered.results[0].target, recovered.errors
    transition = recovered.transition
    migration_counts = recovered.migration_counts
    counts = np.asarray(migration_counts.addressable_shards[0].data).tolist()
    valid = np.asarray(proposal.workset.cell_valid.addressable_shards[0].data)
    record["migration"] = {
        "status": "accepted",
        "route": "native-graph-restart-repack",
        "distribution_kind": transition.target.evidence.kind.value,
        "provenance": transition.target.evidence.provenance,
        "graph_parts_local": np.asarray(proposal.parts.addressable_shards[0].data)[
            valid
        ].tolist(),
        "cell_migration_counts_local": counts,
        "receipt_id": proof.content_id,
        "result_id": target.result_id,
        "local_cells": target.mesh.entity_set(2).count,
        "hot_field_reference_errors": errors,
        "essential_mask_exact": True,
        "saved_cursors_local": np.asarray(
            receipt.saved_cursors.addressable_shards[0].data
        ).tolist(),
        "saved_clocks_local": np.asarray(
            receipt.saved_clocks.addressable_shards[0].data
        ).tolist(),
        "saved_counters_local": np.asarray(
            receipt.saved_counters.addressable_shards[0].data
        ).tolist(),
        "recovered_roles": sorted(recovered.fields),
    }
    record["communication"] = {
        "status": "measured-owner-message-counts",
        "cell_migration_counts_local": counts,
        "message_payload_bytes": None,
        "payload_byte_status": "unmeasured",
    }
    return recovered


def _distributed_bind_checkpoint_source(
    args: QualificationControls,
    generated: _DistributedGeneratedSource,
    graph: _DistributedGraphState,
    mesh_epoch: int,
) -> _DistributedCheckpointBinding:
    """Bind the accepted source closure to the scientific shared-array alias bank."""

    result = graph.results[0]
    target = result.target
    fe_fields, precision = _distributed_qualification_fe_declaration()
    closure = prepare_distributed_checkpoint_source_closure(
        generated.author,
        generated.request,
        generated.options,
        generated.source,
        graph.results,
        graph.physical,
        graph.roles,
        graph.fields,
        fe_fields,
        precision,
        generated.certification,
        part_name=_DISTRIBUTED_PART,
        mesh_epoch=mesh_epoch,
    )
    repository, limits = _distributed_qualification_repository(
        args.distributed_repository_root()
    )
    source_arrays = bind_distributed_checkpoint_source_arrays(closure, limits)
    missing = source_arrays.missing
    if missing:
        raise qualification_failure(
            "_distributed_qualification_publish",
            (f"global source records lack scientific alias roles: {missing}"),
            {"missing": missing},
        )
    return _DistributedCheckpointBinding(
        result=result,
        target=target,
        closure=closure,
        repository=repository,
        limits=limits,
        bindings=source_arrays.bindings,
        bank=source_arrays.bank,
    )


def _distributed_bind_checkpoint_arrays(
    args: QualificationControls,
    author: phx.meshing.NativePlanarSource,
    binding: _DistributedCheckpointBinding,
    graph: _DistributedGraphState,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedCheckpointPayload:
    """Bind logical mesh and raw allocated history arrays to the scientific checkpoint."""

    result, target, closure = binding.result, binding.target, binding.closure
    limits, bindings, bank = binding.limits, binding.bindings, binding.bank
    fields, parts = graph.fields, graph.parts
    mesh_arrays = prepare_distributed_checkpoint_mesh_arrays(
        target, fields, parts, part_count=args.distributed_parts, mesh_epoch=mesh_epoch
    )
    arrays, array_roles, evidence = (
        mesh_arrays.arrays,
        mesh_arrays.array_roles,
        mesh_arrays.evidence,
    )
    state = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "checkpoint_scientific_binding",
        lambda: MeshingCheckpointState(
            source_topology_id=evidence.source.mesh.topology_id,
            source_geometry_id=evidence.source.audit.geometry_id,
            topology_id=target.mesh.topology_id,
            geometry_id=target.audit.geometry_id,
            result_id=target.result_id,
            source_revision_id=author.source_revision,
            coordinate_contract_id=target.coordinate_contract.spatial_id,
            lineage_id=None if result.lineage is None else result.lineage.lineage_id,
            epoch_id=canonical_fingerprint(
                {"accepted": target.result_id, "epoch": mesh_epoch}
            ),
            evidence_id=evidence.evidence_id,
            transfer_id=None if result.transfer is None else result.transfer.transfer_id,
            request_id=evidence.preparation.request.request_id,
            mesh_epoch=mesh_epoch,
            array_roles=tuple(array_roles.items()),
            source_closure=closure,
            source_owner_index=jax.process_index(),
            source_array_bindings=bindings,
            source_logical_arrays=bank,
            limits=limits,
        ),
    )
    return _DistributedCheckpointPayload(state=state, arrays=arrays)


def _distributed_qualification_publish(
    args: QualificationControls,
    generated: _DistributedGeneratedSource,
    graph: _DistributedGraphState,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> None:
    """Publish this process's accepted owner-local result through the source closure."""

    author, request = generated.author, generated.request
    binding = _distributed_bind_checkpoint_source(args, generated, graph, mesh_epoch)
    payload = _distributed_bind_checkpoint_arrays(
        args, author, binding, graph, record, started, mesh_epoch
    )
    target, closure, repository, limits = (
        binding.target,
        binding.closure,
        binding.repository,
        binding.limits,
    )
    state, arrays = payload.state, payload.arrays
    space = graph.physical.field_spaces[0]
    identities = {
        "checkpoint_id": canonical_fingerprint(
            {
                "kind": "distributed-qualification-scientific-checkpoint",
                "request": record["request_id"],
            }
        ),
        "execution_id": canonical_fingerprint(
            record["actual_topology"]["source_devices"]
        ),
        "analysis_plan_id": request.specification_id,
        "numeric_revision_id": author.source_revision,
    }
    publication = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "process_local_checkpoint",
        lambda: publish_process_meshing_checkpoint(
            repository,
            identities["checkpoint_id"],
            identities["execution_id"],
            state,
            arrays,
            analysis_plan_id=identities["analysis_plan_id"],
            numeric_revision_id=identities["numeric_revision_id"],
            writer_id=f"distributed-qualification-process-{jax.process_index()}",
            limits=limits,
        ),
    )
    multihost_utils.sync_global_devices("distributed-qualification-checkpoint-published")
    manifest = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "checkpoint_exact_coverage_assembly",
        lambda: assemble_distributed_checkpoint_from_repository(
            repository,
            identities["checkpoint_id"],
            identities["analysis_plan_id"],
            identities["numeric_revision_id"],
            identities["execution_id"],
            expected_process_count=jax.process_count(),
            limits=limits,
        ),
    )
    record["checkpoint"] = {
        "status": "published",
        **identities,
        "manifest_id": manifest.manifest_id,
        "state_id": state.checkpoint_state_id,
        "source_closure_id": state.source_closure_id,
        "result_id": target.result_id,
        "field_space_id": space.field_space_id,
        "process_index": jax.process_index(),
        "process_shards": len(publication.shards),
        "mesh_epoch": mesh_epoch,
        "scientific_roles": [
            list(role) for role in state.array_roles if role[1] != "source-record"
        ],
        "nonaddressable_arrays": sum(
            not value.is_fully_addressable for value in arrays.values()
        ),
        "closure_roles": sorted(closure),
        "accepted_data_roles": sorted(closure["accepted_data"]),
    }


def _distributed_qualification_produce(
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    started: float,
    mesh_epoch: int,
) -> None:
    """Generate, adapt, repartition, and publish this producer process's accepted epoch."""

    generated = _distributed_qualification_generate(args, record, phases, started)
    source = generated.source
    record["field_declarations"] = [
        {
            **declaration,
            "source_field_id": canonical_fingerprint(
                {
                    "declaration": declaration,
                    "source_result": source.result_id,
                    "entity_set": source.mesh.entity_set(
                        0 if declaration["association"] == "vertex" else 2
                    ).entity_set_id,
                }
            ),
        }
        for declaration in _distributed_qualification_requirements()["fields"]
    ]
    epochs = _distributed_qualification_epochs(
        args, generated.author, generated.request, source, record, started, mesh_epoch
    )
    graph = _distributed_qualification_graph(args, epochs, record, started, mesh_epoch)
    _distributed_qualification_publish(
        args, generated, graph, record, started, mesh_epoch
    )


def _distributed_qualification_launch(
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> list[dict[str, Any]]:
    """Run the producer half in JAX processes and retain every process record."""

    # Fresh repository root: concurrent writers must not open an unprovisioned or reused directory.
    (args.distributed_repository_root() / "repository").mkdir(
        parents=True, exist_ok=False
    )
    command = [
        sys.executable,
        "-m",
        "tools.meshing_qualification",
        "--scenario",
        "native-distributed-lifecycle",
    ]
    for name in _distributed_qualification_requirements()["required_arguments"]:
        command += [f"--{name.replace('_', '-')}", str(args.to_record()[name])]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    root = str(Path(__file__).resolve().parents[1])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (root, *((environment["PYTHONPATH"],) if environment.get("PYTHONPATH") else ()))
    )
    environment.setdefault("PHYDRAX_CPU_COLLECTIVES", "gloo")
    record_directory = args.distributed_repository_root() / "producer-records"
    record_directory.mkdir(exist_ok=False)
    environment["PHYDRAX_MESHING_RECORD_DIRECTORY"] = str(record_directory.resolve())

    def launch() -> tuple[Any, ...]:
        try:
            return launch_local_processes(
                command,
                LocalProcessLaunchPlan(args.distributed_processes, coordinator),
                environment=environment,
                cwd=os.getcwd(),
            )
        except LocalProcessLaunchError as failure:
            # Failed producers still print their phase-resolved records.
            return failure.results

    results = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "producer_processes", launch
    )
    producers: list[dict[str, Any]] = []
    for result in results:
        result_path = record_directory / f"process-{result.process_id}.json"
        if not result_path.is_file():
            raise qualification_failure(
                "_distributed_qualification_launch",
                (
                    f"Producer process {result.process_id} exited {result.returncode} without its record: "
                    + result.stderr.decode("utf-8", errors="replace")[-4000:]
                ),
                {"result_path": result_path},
            )
        producers.append(json.loads(result_path.read_text(encoding="utf-8")))
    record["producer_processes"] = producers
    for result, producer in zip(results, producers, strict=True):
        if producer["status"] != "passed":
            raise qualification_failure(
                "_distributed_qualification_launch",
                (
                    f"Producer process {result.process_id} failed in {producer['failure_phase']}: "
                    f"{producer['failure_type']}: {producer['failure']}"
                ),
                {"producer": producer},
            )
    return producers


def _distributed_qualification_restore(
    args: QualificationControls,
    record: dict[str, Any],
    producers: list[dict[str, Any]],
    started: float,
) -> tuple[Any, Any, Any, Any]:
    """Restore every producer owner into a fresh process set with a changed owner count."""

    register_meshing_source_artifacts()
    repository, limits = _distributed_qualification_repository(
        args.distributed_repository_root()
    )
    published = [producer["checkpoint"] for producer in producers]
    identity_names = (
        "checkpoint_id",
        "execution_id",
        "analysis_plan_id",
        "numeric_revision_id",
    )
    if any(
        {name: item[name] for name in identity_names}
        != {name: published[0][name] for name in identity_names}
        for item in published
    ):
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Producer processes published different accepted checkpoint identities."),
            {"identity_names": identity_names, "published": published},
        )
    expected = {item["process_index"]: item["state_id"] for item in published}
    manifest = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "restore_checkpoint_coverage_assembly",
        lambda: assemble_distributed_checkpoint_from_repository(
            repository,
            published[0]["checkpoint_id"],
            published[0]["analysis_plan_id"],
            published[0]["numeric_revision_id"],
            published[0]["execution_id"],
            expected_process_count=args.distributed_processes,
            limits=limits,
        ),
    )
    states = read_meshing_checkpoint_states(repository, manifest, limits=limits)
    if {
        state.source_owner_index: state.checkpoint_state_id for state in states
    } != expected:
        raise qualification_failure(
            "_distributed_qualification_restore",
            (
                "Restored owner records differ from the producer-published scientific states."
            ),
            {"expected": expected, "states": states},
        )
    names = {name for state in states for name, _ in state.array_roles}
    host_names: set[str] = set()
    for owner in states:
        recipe = owner.source_recipe_json
        if recipe is None:
            raise qualification_failure(
                "_distributed_qualification_restore",
                ("Every restored owner requires its retained source recipe."),
                {"recipe": recipe},
            )
        host_names |= distributed_checkpoint_host_array_names(owner, recipe, limits)
    group = ExecutionRuntime.current().root_group
    relation = TopologyRestartRelation(
        published[0]["execution_id"],
        canonical_fingerprint(record["actual_topology"]["restart_devices"]),
        "bitwise",
    )
    checkpoint = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "changed_owner_count_restore",
        lambda: restore_meshing_checkpoint(
            repository,
            manifest,
            {name: group.named_sharding(PartitionSpec()) for name in names - host_names},
            relation,
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=expected,
            source_owner_index=0,
            limits=limits,
            maximum_host_packet_bytes=args.distributed_host_packet_bytes,
        ),
    )
    state = checkpoint.state
    arrays = checkpoint.arrays
    admission = checkpoint.admission
    if (
        not admission.admitted
        or len(state.source_owner_states) != args.distributed_processes
    ):
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Changed owner-count restore did not admit every producer owner."),
            {"admission": admission, "args": args, "state": state},
        )
    closure = state.source_closure
    if closure is None:
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Restored checkpoint lost its meshing source closure."),
            {"closure": closure},
        )
    source = closure["generation_part"].carrier
    renewed = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "source_recertification",
        lambda: recertify_restored_meshing_source(
            closure["certification_inputs"],
            source.mesh,
            source.geometry,
            source.audit,
            archive_limits=limits,
        ),
    )
    renewed.require_passed()
    if renewed.report_id != closure["report"].report_id:
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Restored source recertification differs from the original theorem."),
            {"closure": closure, "renewed": renewed},
        )
    accepted = closure["collective_target"]
    declaration = closure["field_declarations"][_DISTRIBUTED_PART]
    physical, _ = prepare_meshing_field_owner(
        declaration, accepted, index_source=accepted
    )
    if physical.field_spaces[0].field_space_id != published[0]["field_space_id"]:
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Restored physical field space differs from the published owner-0 space."),
            {"physical": physical, "published": published},
        )
    if int(closure["accepted_data"]["fields"]["accepted-epoch"]) != state.mesh_epoch:
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Restored physical state epoch differs from its checkpoint epoch."),
            {"closure": closure, "state": state},
        )
    record["checkpoint"] = {
        "status": "restored",
        **{name: published[0][name] for name in identity_names},
        "manifest_id": manifest.manifest_id,
        "state_ids": {str(owner): value for owner, value in expected.items()},
        "source_process_count": len(states),
        "restored_process_count": jax.process_count(),
        "source_result_ids": {
            str(item["process_index"]): item["result_id"] for item in published
        },
        "source_closure_ids": {
            str(item["process_index"]): item["source_closure_id"] for item in published
        },
        "restored_field_space_id": physical.field_spaces[0].field_space_id,
        "source_closure_id": state.source_closure_id,
        "recertified_report_id": renewed.report_id,
    }
    return state, arrays, closure, accepted


def _distributed_qualification_repack(
    args: QualificationControls,
    closure: Any,
    accepted: Any,
    arrays: dict[str, Any],
    record: dict[str, Any],
    started: float,
) -> tuple[Any, Any, Any, Any, _DistributedFields]:
    """Repack the restored allocated forest onto the changed owner count and recover all roles."""

    meshing = phx.meshing
    fields = closure["accepted_data"]["fields"]
    layout, raw, _ = restore_partitioned_mesh_state(
        accepted.mesh.storage, accepted.collective_evidence
    )
    owners = arrays["cell_owners"]
    receipt = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "changed_owner_count_repack",
        lambda: repack_simplex_restart(
            ExecutionRuntime.current().root_group,
            layout,
            raw,
            owners,
            arrays["vertex_owners"],
            # Every restored cell is requested by the single changed owner.
            jnp.zeros_like(owners),
            cell_history=_distributed_qualification_histories(fields, "cell"),
            vertex_history=_distributed_qualification_histories(fields, "vertex"),
            source_result=accepted,
        ),
    )
    proof = SimplexRestartRepackProof(receipt)
    proof.require_receipt(receipt)
    counts = receipt.cell_migration_counts
    if (
        counts is None
        or not bool(jnp.all(receipt.status == 0))
        or counts.shape != (args.distributed_parts, args.restart_parts)
        or int(jnp.sum(counts)) != int(jnp.sum(raw.mesh.cell_active))
    ):
        raise qualification_failure(
            "_distributed_qualification_repack",
            ("Changed owner-count repack did not move every accepted cell exactly once."),
            {"args": args, "counts": counts, "raw": raw, "receipt": receipt},
        )
    corrupt = replace(
        receipt,
        vertex_history=(
            receipt.vertex_history[0].at[0, 0].add(0.25),
            *receipt.vertex_history[1:],
        ),
    )
    tamper = {
        "fresh_proof_refused": _distributed_qualification_refuses(
            lambda: SimplexRestartRepackProof(corrupt)
        ),
        "bound_proof_refused": _distributed_qualification_refuses(
            lambda: proof.require_receipt(corrupt)
        ),
    }
    if not all(tamper.values()):
        raise qualification_failure(
            "_distributed_qualification_repack",
            ("A tampered physical history transfer record was accepted."),
            {"tamper": tamper},
        )
    results = closure["accepted_data"]["adaptation_results"]
    old_policy = results[0].policy
    old_partition = old_policy.partition_policy
    partition = meshing.MeshPartitionPolicy(
        meshing.MeshPartitionKind.PROVIDER,
        args.restart_parts,
        halo_width=old_partition.halo_width,
        maximum_imbalance=old_partition.maximum_imbalance,
        curve_depth=old_partition.curve_depth,
        metis_seed=old_partition.metis_seed,
    )
    prepared = meshing.prepare_mesh_adaptation(
        accepted,
        meshing.MarkedMeshAdaptation(),
        policy=_distributed_qualification_policy(
            old_policy, results[0].distribution.target, partition
        ),
    )
    restored = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "changed_owner_count_publication",
        lambda: commit_partitioned_mesh_restart(receipt, prepared).local_results,
    )
    if len(restored) != 1:
        raise qualification_failure(
            "_distributed_qualification_repack",
            ("The changed owner count must publish exactly one accepted local result."),
            {"restored": restored},
        )
    result = restored[0]
    target, transition = result.target, result.distribution
    storage, evidence = target.mesh.storage, target.collective_evidence
    if storage is None or evidence is None or transition is None:
        raise qualification_failure(
            "_distributed_qualification_repack",
            (
                "Changed owner-count publication omitted its storage, proof, or distribution."
            ),
            {"evidence": evidence, "storage": storage, "transition": transition},
        )
    stable = np.unique(
        np.asarray(raw.mesh.vertex_ids)[np.asarray(raw.mesh.vertex_active)]
    )
    if (
        storage.partition_count != args.restart_parts
        or evidence.partition_count != args.restart_parts
        or transition.source.partition.part_count != args.distributed_parts
        or transition.target.partition.part_count != args.restart_parts
        or not np.array_equal(np.sort(np.asarray(target.mesh.vertex_global_ids)), stable)
        or not all(
            np.array_equal(np.asarray(saved), np.asarray(value))
            for saved, value in (
                (receipt.saved_cursors, raw.cursors),
                (receipt.saved_clocks, raw.clocks),
                (receipt.saved_counters, raw.counters),
            )
        )
    ):
        raise qualification_failure(
            "_distributed_qualification_repack",
            (
                "Changed owner-count publication lost stable IDs, ownership, or allocator clocks."
            ),
            {
                "args": args,
                "evidence": evidence,
                "raw": raw,
                "receipt": receipt,
                "stable": stable,
                "storage": storage,
                "target": target,
                "transition": transition,
            },
        )
    declaration = closure["field_declarations"][_DISTRIBUTED_PART]
    prepared_fe = _distributed_qualification_fe(
        target,
        declaration.finite_element_fields,
        declaration.precision_policy,
        declaration.numeric_version,
    )
    roles, recovered = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "restored_field_history_recovery",
        lambda: _distributed_qualification_recover(
            target,
            receipt,
            declaration.state_roles,
            fields,
            prepared_fe.field_spaces[0],
            proof,
        ),
    )
    renewed = rebind_distributed_checkpoint_field_declaration(
        declaration, target, prepared_fe, roles
    )
    new_closure = {
        **closure,
        "collective_target": target,
        "field_declarations": {_DISTRIBUTED_PART: renewed},
        "accepted_data": {"adaptation_results": restored, "fields": recovered},
    }
    validate_meshing_field_declarations(new_closure)
    record["restart"] = {
        "status": "full-accepted-result-reconstructed",
        "source_parts": args.distributed_parts,
        "destination_parts": args.restart_parts,
        "receipt_id": proof.content_id,
        "restored_kind": transition.target.evidence.kind.value,
        "result_id": target.result_id,
        "tamper_refusal": tamper,
        "source_result_id": accepted.result_id,
        "field_space_id": prepared_fe.field_spaces[0].field_space_id,
        "saved_cursors": np.asarray(receipt.saved_cursors).tolist(),
        "saved_clocks": np.asarray(receipt.saved_clocks).tolist(),
        "saved_counters": np.asarray(receipt.saved_counters).tolist(),
        "recovered_roles": sorted(recovered),
        "full_accepted_result_reconstructed": True,
    }
    record["stable_ids"] = {
        "vertex_count": stable.size,
        "preserved": True,
        "allocator_clocks_preserved": True,
    }
    return target, new_closure, receipt, proof, recovered


def _distributed_reopen_physical_fields(
    args: QualificationControls,
    new_closure: dict[str, Any],
    record: dict[str, Any],
    started: float,
) -> _DistributedRestoredFields:
    """Rearchive and decode accepted scientific fields, then check physical row meaning."""

    _, limits = _distributed_qualification_repository(args.distributed_repository_root())
    archived = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "restored_closure_rearchive",
        lambda: write_meshing_source_closure(
            args.distributed_repository_root() / "restored-closure.zip",
            new_closure,
            limits=limits,
        ),
    )
    reopened = read_meshing_source_closure(
        archived.path, expected_content_id=archived.content_id, limits=limits
    )
    target = reopened["collective_target"]
    fields = reopened["accepted_data"]["fields"]
    final_space, _ = prepare_meshing_field_owner(
        reopened["field_declarations"][_DISTRIBUTED_PART], target, index_source=target
    )
    if not isinstance(final_space, phx.discretization.FiniteElementDiscretization):
        raise TypeError(
            f"_distributed_reopen_physical_fields.final_space must be a FiniteElementDiscretization; got {type(final_space).__name__!r}."
        )
    restored_fields = new_closure["accepted_data"]["fields"]
    if any(
        not np.array_equal(np.asarray(fields[name]), np.asarray(restored_fields[name]))
        for name in restored_fields
    ):
        raise qualification_failure(
            "_distributed_qualification_continue",
            ("Rearchived accepted fields differ from the restored accepted fields."),
            {"fields": fields, "restored_fields": restored_fields},
        )
    points = final_space.dof_maps[0].dof_coordinates
    expected = 1.0 + points[:, 0] + 2.0 * points[:, 1]
    cells = _distributed_qualification_entity_cells(target.mesh)
    vertices = target.mesh.coordinates
    material = 37.0 + 3.0 * vertices[:, 0] - vertices[:, 1]
    errors = {
        "temperature": float(jnp.max(jnp.abs(fields["temperature"] - expected))),
        "essential-values": float(
            jnp.max(jnp.abs(fields["essential-values"] - expected))
        ),
        "material-temperature": float(
            jnp.max(jnp.abs(fields["material-temperature"] - material))
        ),
        "material-cell-average": float(
            jnp.max(
                jnp.abs(
                    fields["material-cell-average"]
                    - jnp.mean(material[jnp.asarray(cells)], axis=1)
                )
            )
        ),
    }
    mask_exact = bool(
        jnp.array_equal(
            fields["essential-mask"], final_space.dof_maps[0].boundary_dof_mask
        )
    )
    if not mask_exact:
        raise qualification_failure(
            "_distributed_qualification_continue",
            ("Restored essential mask differs from the accepted physical boundary."),
            {"mask_exact": mask_exact},
        )
    return _DistributedRestoredFields(
        target=target,
        fields=fields,
        final=final_space,
        expected=expected,
        cells=cells,
        errors=errors,
        mask_exact=mask_exact,
        archive_content_id=archived.content_id,
        archive_path=archived.path,
        reopened_roles=tuple(sorted(reopened)),
    )


@eqx.filter_jit
def _continued_distributed_fe_system(
    problem: phx.equations.CompiledFiniteElementProblem,
    /,
) -> tuple[phx.linalg.LinearSystem, object]:
    """Assemble once in the restored FE layout's declared capacity bucket."""
    return problem.linear_system()


@eqx.filter_jit
def _continued_distributed_fe_solve(
    problem: phx.equations.CompiledFiniteElementProblem,
    operator: phx.linalg.LinearSystem,
    rhs: object,
    policy: phx.linalg.LinearSolvePolicy,
    /,
) -> tuple[object, phx.linalg.LinearSolveResult]:
    """Solve and expand in the restored FE layout's declared capacity bucket."""
    solved = phx.linalg.solve(operator, rhs, policy=policy)
    return problem.expand(solved.value), solved


def _distributed_continue_fe(
    args: QualificationControls,
    restored: _DistributedRestoredFields,
    record: dict[str, Any],
    started: float,
) -> _DistributedFEContinuation:
    """Continue the nonempty restored Dirichlet problem and check the affine reference."""

    _ledger = StageLedger(record)
    final_space, fields, expected, errors = (
        restored.final,
        restored.fields,
        restored.expected,
        restored.errors,
    )
    form = phx.equations.FiniteElementForm(
        "restored-temperature",
        "temperature",
        (phx.equations.DiffusionAction("temperature", 1.0),),
    )
    problem = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "continued_fe_preparation",
        lambda: phx.equations.compile_finite_element_problem(
            form,
            final_space,
            constraint=phx.discretization.dirichlet_constraint(
                final_space, "temperature", boundary_mask=fields["essential-mask"]
            ),
            dirichlet_values=fields["essential-values"],
        ),
    )
    operator, rhs = _continued_distributed_fe_system(problem)
    policy = phx.linalg.LinearSolvePolicy(
        tolerance=phx.linalg.TolerancePolicy(relative=1e-12, absolute=1e-13)
    )
    values, solved = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "continued_fe_solve",
        lambda: jax.block_until_ready(
            _continued_distributed_fe_solve(problem, operator, rhs, policy)
        ),
    )
    warm = []
    for repeat in range(args.repeats):
        name = f"continued_fe_warm_solve_{repeat}"
        StageLedger(record, deadline=Deadline(started + args.timeout)).run(
            name,
            lambda: jax.block_until_ready(
                _continued_distributed_fe_solve(problem, operator, rhs, policy)
            ),
        )
        warm.append(record["stages_seconds"][name])

    solve_error = _ledger.run(
        "continued_fe_validation",
        lambda: float(np.max(np.abs(np.asarray(values) - np.asarray(expected)))),
    )
    free = int(np.count_nonzero(~np.asarray(final_space.dof_maps[0].boundary_dof_mask)))
    if (
        not np.all(np.asarray(solved.successful))
        or free == 0
        or solve_error > 1e-10
        or max(errors.values()) > 1e-12
    ):
        raise qualification_failure(
            "_distributed_qualification_continue",
            (
                "Restored physical fields or continued FE solve disagree with the affine reference."
            ),
            {
                "errors": errors,
                "free": free,
                "solve_error": solve_error,
                "solved": solved,
            },
        )
    return _DistributedFEContinuation(solve_error=solve_error, free=free, warm=warm)


def _distributed_check_restored_inventories(
    restored: _DistributedRestoredFields,
    continued: _DistributedFEContinuation,
    record: dict[str, Any],
) -> _DistributedInventories:
    """Check independent physical inventories and publish restored FE acceptance evidence."""

    _ledger = StageLedger(record)
    target, cells, fields, errors = (
        restored.target,
        restored.cells,
        restored.fields,
        restored.errors,
    )
    mask_exact = restored.mask_exact
    free, solve_error, warm = continued.free, continued.solve_error, continued.warm
    triangles = _ledger.run(
        "restored_field_integral_validation",
        lambda: np.asarray(target.mesh.coordinates)[cells],
    )
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / 2
    integrals = {
        "area": float(np.sum(area)),
        "temperature": float(
            np.sum(area * np.mean(np.asarray(fields["temperature"])[cells], axis=1))
        ),
        "material-temperature": float(
            np.sum(
                area * np.mean(np.asarray(fields["material-temperature"])[cells], axis=1)
            )
        ),
        "material-cell-average": float(
            np.sum(area * np.asarray(fields["material-cell-average"]))
        ),
    }
    references = {
        "area": 1.0,
        "temperature": 2.5,
        "material-temperature": 38.0,
        "material-cell-average": 38.0,
    }
    if any(abs(integrals[name] - value) > 1e-12 for name, value in references.items()):
        raise qualification_failure(
            "_distributed_qualification_continue",
            ("Restored physical inventories disagree with their independent integrals."),
            {"integrals": integrals, "references": references},
        )
    record["archive"] = {
        "content_id": restored.archive_content_id,
        "path": str(restored.archive_path),
        "reopened_roles": list(restored.reopened_roles),
    }
    record["restored_fields"] = {
        "affine_reference_errors": errors,
        "essential_mask_exact": mask_exact,
        "reopened_state_roles": sorted(fields),
        "integrals": integrals,
        "independent_references": references,
    }
    record["continued_fe"] = {
        "status": "executed",
        "physical_problem": "restored-Dirichlet-Laplace-affine-temperature",
        "free_dofs": free,
        "solve_max_error": solve_error,
        "warm_samples_seconds": warm,
        "coefficient_shape": list(fields["temperature"].shape),
        "continued_from_restart": True,
    }
    record["physical_error"] = solve_error
    return _DistributedInventories(integrals=integrals, references=references)


def _distributed_continue_fv(
    args: QualificationControls,
    restored: _DistributedRestoredFields,
    inventory: _DistributedInventories,
    record: dict[str, Any],
    started: float,
) -> None:
    """Advance restored material-density Euler state with owner-local halo exchange."""

    _ledger = StageLedger(record)
    target, fields = restored.target, restored.fields
    integrals, references = inventory.integrals, inventory.references
    mesh = target.mesh
    storage = mesh.storage

    system = _ledger.run("continued_fv_preparation", lambda: phx.equations.EulerSystem(2))
    if storage is None:
        raise qualification_failure(
            "_distributed_continue_fv",
            "Restored FV mesh.storage must retain owner-local closure evidence; got None",
            {"mesh": mesh},
        )
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        mesh,
        component_names=system.component_names,
        neighborhood_complete=storage.local_neighborhood_complete,
        physical_boundary_faces=storage.local_physical_boundary_facets,
        closure_evidence_id=storage.evidence_id,
    ).prepare(owned_face_capacity=4 * storage.global_entity_counts[1])
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.SlipWallBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    dynamics = PreparedUnstructuredFiniteVolumeDynamics(
        system,
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        boundaries,
    )
    count = dynamics.discretization.cell_centers.shape[0]
    # Density carries the restored material-cell history; the advance must move it while conserving inventory.
    density = (
        jnp.asarray(fields["material-cell-average"]) / references["material-cell-average"]
    )
    seed = system.primitive_to_conserved(
        jnp.stack((density, jnp.zeros(count), jnp.zeros(count), jnp.ones(count)), axis=1)
    )[None]
    devices = tuple(jax.devices())
    group = ExecutionGroup(
        ExecutionGroupSpec(
            canonical_fingerprint(
                {
                    "kind": "distributed-qualification-restored-group",
                    "devices": [(device.process_index, device.id) for device in devices],
                }
            ),
            tuple(sorted({device.process_index for device in devices})),
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("parts", args.restart_parts),),
        ),
        devices,
    )

    halos = _ledger.run(
        "continued_fv_halo_preparation",
        lambda: prepare_owner_local_halo_plans(
            (mesh,),
            2,
            devices=devices,
            axis_name="parts",
            message_capacity=min(
                args.distributed_halo_capacity, storage.entity_global_ids[2].size
            ),
        ),
    )
    advance = StageLedger(record, deadline=Deadline(started + args.timeout)).run(
        "continued_owner_local_fv",
        lambda: jax.block_until_ready(
            execute_owner_local_finite_volume_advance(
                ((dynamics, halos[0]),),
                seed,
                jnp.asarray(0.0),
                jnp.asarray(0.01),
                group,
                axis_name="parts",
            )
        ),
    )
    drift = float(jnp.max(jnp.abs(advance.inventory_after - advance.inventory_before)))
    changed = int(
        np.count_nonzero(np.asarray(advance.state[0][:, 0]) != np.asarray(seed[0][:, 0]))
    )
    if not bool(advance.finite & advance.admissible) or drift > 1e-12 or changed == 0:
        raise qualification_failure(
            "_distributed_qualification_continue",
            ("Continued FV advance is inadmissible, nonconservative, or stationary."),
            {"advance": advance, "changed": changed, "drift": drift},
        )
    record["continued_fv"] = {
        "status": "executed",
        "physical_problem": "restored-material-density-slip-wall-Euler",
        "changed_cells": changed,
        "inventory_drift": drift,
        "system_id": system.system_id,
        "continued_from_restart": True,
    }
    record["conservation"] = {
        "status": "measured",
        "material_cell_inventory": integrals["material-cell-average"],
        "fv_inventory_drift": drift,
    }


def _distributed_qualification_continue(
    args: QualificationControls,
    new_closure: dict[str, Any],
    record: dict[str, Any],
    started: float,
) -> float:
    """Rearchive the restored closure and continue nonempty FE and FV physics from it."""
    restored = _distributed_reopen_physical_fields(args, new_closure, record, started)
    continued = _distributed_continue_fe(args, restored, record, started)
    inventory = _distributed_check_restored_inventories(restored, continued, record)
    _distributed_continue_fv(args, restored, inventory, record, started)
    return continued.solve_error


def _distributed_qualification_lifecycle(
    args: QualificationControls,
) -> dict[str, object]:
    """Publish from producer processes, then restore with a changed owner count.

    A single-process invocation launches ``distributed_processes`` producer
    processes with the same frozen request, then restores, rearchives and
    continues the published accepted state in its own fresh process.
    """

    started = perf_counter()
    phases: list[NativeMeshingPhaseMeasurement] = []
    requirements = _distributed_qualification_requirements()
    frozen = {name: args.to_record()[name] for name in requirements["required_arguments"]}
    record: dict[str, Any] = {
        "scenario": "native-distributed-lifecycle",
        "status": "failed",
        "request": frozen,
        "request_id": None,
        "runtime": _runtime_identity(),
        "frozen_targets": requirements,
        "source": None,
        "generation": None,
        "actual_topology": None,
        "epochs": [],
        "stages_seconds": {},
        "stage_peak_process_bytes": {},
        "attempts": [],
        "timeout_scope": "native-provider-limits-and-stage-deadline-checks; external-command-owns-hard-deadline",
        "compiler": {
            "status": "unmeasured",
            "temporary_bytes": None,
            "output_bytes": None,
            "argument_bytes": None,
            "generated_code_bytes": None,
        },
        "communication": {
            "status": "unmeasured",
            "scope": "included-in-owner-local-execution",
        },
        "migration": {"status": "not-executed"},
        "checkpoint": {"status": "not-executed"},
        "restart": {"status": "not-executed"},
        "conservation": {"status": "unmeasured"},
        "physical_error": None,
        "time_to_target_error_seconds": None,
        "independent_scientific_certification": "unassessed",
        "mandatory_workflow_completion": "incomplete",
        "active_phase": "actual_placement_admission",
        "nonclaims": requirements["nonclaims"],
    }
    try:
        record["request_id"] = canonical_fingerprint(frozen)
        topology = _distributed_qualification_topology(args)
        record["actual_topology"] = topology
        # Refine/coarsen epochs plus the graph restart epoch.
        mesh_epoch = 2 * args.adaptation_rounds + 2
        match topology["role"]:
            case "producer":
                _distributed_qualification_produce(
                    args, record, phases, started, mesh_epoch
                )
                record["mandatory_workflow_completion"] = "producer-published"
            case "restore":
                producers = _distributed_qualification_launch(args, record, started)
                _, arrays, closure, accepted = _distributed_qualification_restore(
                    args, record, producers, started
                )
                _, new_closure, _, _, _ = _distributed_qualification_repack(
                    args, closure, accepted, arrays, record, started
                )
                physical_error = _distributed_qualification_continue(
                    args, new_closure, record, started
                )
                if physical_error > args.target_error:
                    raise qualification_failure(
                        "_distributed_qualification_lifecycle",
                        (
                            "Continued distributed physics missed the frozen physical-error target."
                        ),
                        {"args": args, "record": record},
                    )
                record["time_to_target_error_seconds"] = perf_counter() - started
                record["independent_scientific_certification"] = "passed"
                record["mandatory_workflow_completion"] = "complete"
            case role:
                raise ValueError(
                    f"_distributed_qualification_lifecycle.role must be 'producer' or 'restore'; got {role!r}."
                )
        record["status"] = "passed"
    except Exception as error:
        native_timeout = isinstance(error, phx.meshing.MeshingFailure) and (
            error.category is phx.meshing.MeshingFailureCategory.TIMED_OUT
        )
        record["status"] = (
            "timeout" if isinstance(error, TimeoutError) or native_timeout else "failed"
        )
        detail = FailureRecord.from_exception(error, record["active_phase"]).to_record()
        record.update(detail)
        record["attempts"].append(
            {
                "phase": record["active_phase"],
                "status": record["status"],
                **detail,
            }
        )
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
        record["process_peak_scope"] = "process-lifetime"
        record["device_memory"] = [
            {
                "device_id": device.id,
                "process_index": device.process_index,
                "memory_statistics": statistics,
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "status": "unavailable" if statistics is None else "backend-reported",
            }
            for device in jax.local_devices()
            for statistics in (device.memory_stats(),)
        ]
        if record["end_to_end_seconds"] > args.timeout:
            record["status"] = "timeout"
            record.update(
                {
                    **FailureRecord(
                        record["active_phase"],
                        "TimeoutError",
                        "Frozen distributed lifecycle wall deadline exceeded.",
                    ).to_record(),
                    "failure_category": "wall-deadline",
                    "requested": {"wall_seconds": args.timeout},
                    "achieved": {"wall_seconds": record["end_to_end_seconds"]},
                }
            )
        record["counts"] = {
            "passed": int(record["status"] == "passed"),
            "failed": int(record["status"] == "failed"),
            "timeout": int(record["status"] == "timeout"),
            "uncertainty_or_refusal_is_positive": False,
        }
    return record


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _distributed_qualification_lifecycle(args)
