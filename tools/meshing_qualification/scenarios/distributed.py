# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from collections.abc import (
    Callable,
    Iterator,
    Mapping,
)
from contextlib import contextmanager
from dataclasses import (
    asdict,
    dataclass,
    replace,
)
from functools import (
    cache,
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
    TypedDict,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import (
    multihost_utils,
)
from jax.sharding import (
    NamedSharding,
    PartitionSpec,
)

import phydrax as phx
from phydrax import (
    ArrayArchiveLimits,
    canonical_fingerprint,
)
from phydrax._fingerprint import array_tree_fingerprint
from phydrax._meshcore import MeshcoreError, MeshcoreStatus
from phydrax.discretization import (
    AdaptiveSimplexPolicy,
    AdaptiveSimplexStatus,
    CellMeshStorage,
    coarsen_adaptive_simplex_parts,
    FiniteElementTopologyTransfer,
    make_owner_local_field_array,
    prepare_owner_local_halo_plans,
    refine_adaptive_simplex_parts,
    resolve_owner_local_field_values,
)
from phydrax.discretization.fem import (
    FiniteElementExecutionLimits,
    FiniteElementPlan,
    FiniteElementPrecisionPolicy,
    OwnerLocalFiniteElementDiscretization,
    prepare_owner_local_finite_element,
)
from phydrax.discretization.fem._distributed import (
    prepare_owner_local_finite_element_execution,
)
from phydrax.discretization.finite_volume import (
    execute_owner_local_finite_volume_advance,
    PreparedUnstructuredFiniteVolumeDynamics,
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.discretization.finite_volume._unstructured_dynamics import (
    prepare_owner_local_finite_volume_execution,
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
    MeshingFieldIndexBinding,
    MeshingFieldStateKind,
    MeshingFieldStateRole,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
    prepare_meshing_field_index_binding,
    prepare_meshing_field_owner,
    project_meshing_field_history,
    publish_process_meshing_checkpoint,
    read_meshing_checkpoint_states,
    rebind_meshing_field_history,
    recover_meshing_field_history,
    register_meshing_source_artifacts,
    RepositoryCorruptionError,
    restore_meshing_checkpoint,
    TopologyRestartPolicy,
    TopologyRestartRelation,
    validate_meshing_field_declarations,
)
from phydrax.lifecycle._meshing_field_records import (
    _owner_local_recovery,
    MeshingRandomStream,
)
from phydrax.linalg import (
    DistributedKrylovPolicy,
    DistributedLinearOperator,
    solve_distributed_pcg,
)
from phydrax.meshing import (
    CollectiveMeshEvidence,
    commit_partitioned_adaptive_simplex,
    commit_partitioned_mesh_restart,
    DeviceGenerationLayout,
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
from phydrax.meshing._collective_failure import (
    agreed_local,
    agreed_local_step,
    collective_failure_decision,
)
from phydrax.meshing._device_adaptation import (
    partitioned_transfer_sources,
    repartition_partitioned_mesh_adaptation,
)
from phydrax.meshing._distributed_planar_generation import (
    PlanarTileSchedule,
    prepare_distributed_planar_generation,
    PreparedDistributedPlanarGeneration,
)
from phydrax.meshing._distribution import prepare_partitioned_rebalance_proposal
from phydrax.meshing._distribution_migration import accept_ownerlocal_migration
from phydrax.meshing._initial_certification import (
    InitialCollectiveMeshEvidence,
    InitialPlanarPremise,
)
from phydrax.meshing._restart_distribution import SimplexRestartInvalidReason
from phydrax.meshing._result import require_original_meshing_source
from phydrax.meshing.qualification import (
    bind_distributed_checkpoint_source_arrays,
    distributed_field_templates,
    DistributedRestorePlacement,
    prepare_distributed_checkpoint_mesh_arrays,
    prepare_distributed_checkpoint_source_closure,
    prepare_distributed_restore_placement,
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
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    Deadline,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios import _distributed_euler as euler_history
from tools.meshing_qualification.scenarios.triangulation import (
    _simplex_measures,
)


type _DistributedFields = dict[str, jax.Array | np.ndarray]


class _DistributedEulerModel(TypedDict):
    """Persisted physical definitions; no executable caches or provider handles."""

    system: phx.equations.EulerSystem
    reconstruction: phx.discretization.PiecewiseConstantReconstruction
    flux: phx.discretization.RusanovFluxPlan
    boundary: phx.discretization.SlipWallBoundary
    time_step: jax.Array | np.ndarray


_DISTRIBUTED_PART = "native-unit-square"
# The orchestrating invocation sets this for each independently launched process set.
_DISTRIBUTED_ROLE = "PHYDRAX_DISTRIBUTED_QUALIFICATION_ROLE"
_DISTRIBUTED_REQUEST = "PHYDRAX_DISTRIBUTED_QUALIFICATION_REQUEST"
_PRODUCER_PUBLICATION = "producer-publication.json"
# Declared root seed of the material-history probe stream. Producers persist
# its key data as an rng-key bank; restarted owners draw only from that bank.
_DISTRIBUTED_RNG_SEED = 20261020
# Rademacher signs drawn by one owned cell per probe event (at most 32 bits).
_DISTRIBUTED_RNG_PROBES = 8
# Future probe events compared between uninterrupted and restored owners.
_DISTRIBUTED_RNG_STEPS = 2
# Sketch agreement relative to the cell inventory: summation order only.
_DISTRIBUTED_RNG_RELATIVE_TOLERANCE = 1.0e-10
# Probed restored history must match its independent cell means to this norm.
_DISTRIBUTED_RNG_RESIDUAL_NORM = 1.0e-12


@contextmanager
def _distributed_stage_boundary(
    record: dict[str, Any], name: str
) -> Iterator[None]:
    """Agree only a ledger's local boundary, never its scientific callback."""
    failure: Exception | None = None
    category = None
    categories = phx.meshing.MeshingFailureCategory
    try:
        yield
    except Exception as error:
        failure = error
        record.setdefault("stage_boundary_failures", []).append(
            FailureRecord.from_exception(error, name).to_record()
        )
        match error:
            case TimeoutError() | MeshcoreError(status=MeshcoreStatus.TIMEOUT):
                category = categories.TIMED_OUT
            case phx.meshing.MeshingFailure():
                category = error.category
            case MemoryError() | MeshcoreError(status=MeshcoreStatus.CAPACITY_EXCEEDED):
                category = categories.RESOURCE_EXHAUSTED
            case ValueError():
                category = categories.LINEAGE_FAILED
            case _:
                category = categories.PROVIDER_EXECUTION_FAILED
    decision = collective_failure_decision(category)
    # A deadline remains authoritative even if another rank simultaneously
    # rejects a different local boundary. Preserve a local native timeout intact.
    timeout_code = tuple(categories).index(categories.TIMED_OUT) + 1
    selected = (
        categories.TIMED_OUT
        if np.any(decision.rejected == timeout_code)
        else decision.category
    )
    if selected is not None:
        if failure is not None and category is selected:
            raise failure
        raise phx.meshing.MeshingFailure(
            selected,
            "An owning process rejected the qualification stage boundary; "
            "no process may enter the next scientific collective.",
            stage=name,
        )


def _distributed_stage_ledger(
    record: dict[str, Any], *, deadline: Deadline | None = None
) -> StageLedger:
    return StageLedger(
        record,
        deadline=deadline,
        local_boundary=(
            (lambda name: _distributed_stage_boundary(record, name))
            if os.environ.get(_DISTRIBUTED_ROLE) in ("producer", "consumer")
            else None
        ),
    )


def _distributed_qualification_request(args: QualificationControls) -> dict[str, Any]:
    """Retain every declared control, including explicit nullable resource limits."""
    controls = args.to_record()
    return {
        name: controls[name]
        for name in _distributed_qualification_requirements()["required_arguments"]
    }


def _distributed_qualification_topology(args: QualificationControls) -> dict[str, object]:
    """Refuse unavailable real placements before any scientific work begins."""

    requirements = _distributed_qualification_requirements()["required_arguments"]
    for name, rule in requirements.items():
        value = args.to_record()[name]
        if value is None and rule.get("nullable") is True:
            continue
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
    producers, consumers = args.distributed_processes, args.restart_parts
    if producers != args.distributed_parts:
        raise ValueError(
            f"_distributed_qualification_topology.distributed_processes must equal distributed_parts; got {(producers, args.distributed_parts)!r}."
        )
    # The leading source-owner axis must split evenly over a changed owner
    # count. One consumer runs the same owner-local restart and continued
    # physics as a routine case, never a changed multiple-process claim.
    if consumers < 1 or consumers == producers or producers % consumers:
        raise ValueError(
            "_distributed_qualification_topology.restart_parts must be a changed "
            f"owner count dividing distributed_processes; got {(producers, consumers)!r}."
        )
    role = os.environ.get(_DISTRIBUTED_ROLE, "orchestrator")
    expected = {"orchestrator": 1, "producer": producers, "consumer": consumers}
    if role not in expected or jax.process_count() != expected[role]:
        raise ValueError(
            f"_distributed_qualification_topology.process_count must match the launched role; got role={role!r}, process_count={jax.process_count()!r}."
        )
    if role != "orchestrator":
        parent_request = os.environ.get(_DISTRIBUTED_REQUEST)
        if parent_request is None:
            raise ValueError(
                "_distributed_qualification_topology.request must retain the "
                "explicit parent request before child scientific work."
            )
        retained = json.loads(parent_request)
        if not isinstance(retained, dict) or canonical_fingerprint(
            retained
        ) != canonical_fingerprint(_distributed_qualification_request(args)):
            raise ValueError(
                "_distributed_qualification_topology.request must match the "
                "exact parent request and resource controls."
            )
    local = tuple(jax.local_devices())
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
        for device in jax.devices()
    ]
    return {
        "role": role,
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "global_device_count": jax.device_count(),
        "local_device_count": len(local),
        "distributed_parts": args.distributed_parts,
        "restart_parts": consumers,
        "generation_placement": "owner-local-native-planar-tiles-per-producer-process",
        "source_devices": devices if role == "producer" else None,
        "restart_devices": devices if role == "consumer" else None,
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
        maximum_work_units=args.maximum_work_units,
        maximum_geometry_queries=args.maximum_geometry_queries,
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
    """Generate the unchanged authored source owner-locally on its fixed tile complex.

    Tiles, seams and every constraint point are fixed from source, request and
    schedule; each producer refines only its own tiles and the provider's
    collective initial theorem accepts or refuses all owners together.
    """

    source, request, options = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run("source_preparation", lambda: _distributed_qualification_source(args))
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
    schedule = PlanarTileSchedule()
    layout = DeviceGenerationLayout(
        vertex_capacity=args.capacity,
        cell_capacity=args.capacity,
        candidate_capacity=args.capacity,
        maximum_work_units=request.limits.maximum_work_units,
        neighborhood_depth=args.distributed_halo_width,
    )
    provider = phx.meshing.NativeMeshingProvider(options)

    def plan_generation() -> tuple[Any, Any]:
        prepared = prepare_distributed_planar_generation(source, request, schedule, layout)
        return prepared, provider.plan(
            source,
            request,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            record_phase=phases.append,
            initial_partition=prepared,
        )

    prepared, plan = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "geometry_constraint_preparation", plan_generation
    )
    result = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "native_generation_publication_certification",
        lambda: plan.execute(record_phase=phases.append),
    )
    type(result.audit).require_passed(result.audit)
    evidence, storage = result.collective_evidence, result.mesh.storage
    if (
        not result.compliance.passed
        or storage is None
        or not isinstance(evidence, InitialCollectiveMeshEvidence)
        or not isinstance(evidence.premise, InitialPlanarPremise)
        or evidence.premise.generation_id != prepared.tiles.compiled_id
        or evidence.premise.layout.neighborhood_depth != args.distributed_halo_width
        or storage.neighborhood_depth != args.distributed_halo_width
    ):
        raise qualification_failure(
            "_distributed_qualification_generate",
            "Owner-local planar generation lacks its collective tile theorem or control acceptance.",
            {"result": result},
        )
    evidence.require_current()
    mesh = result.mesh
    cells = _distributed_qualification_entity_cells(mesh)
    coordinates = np.asarray(mesh.coordinates)
    owned_cells = np.asarray(storage.entity_owned[-1], dtype=np.bool_)
    measures = _simplex_measures(coordinates, cells[owned_cells])
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError(
            f"_distributed_qualification_generate.connectivity must be PolygonalConnectivity; got {type(connectivity).__name__!r}."
        )
    physical = storage.local_physical_boundary_facets
    if physical is None:
        raise qualification_failure(
            "_distributed_qualification_generate",
            "Owner-local planar storage lacks its physical boundary receipts.",
            {"storage": storage},
        )
    boundary = np.asarray(physical, dtype=np.bool_) & np.asarray(
        storage.entity_owned[1], dtype=np.bool_
    )
    edges = np.asarray(connectivity.edges)[boundary]
    local_measures = np.asarray(
        (
            float(np.sum(measures)),
            float(
                np.sum(
                    np.linalg.norm(
                        coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1
                    )
                )
            ),
        ),
        dtype=np.float64,
    )
    area, perimeter = (float(value) for value in _distributed_process_sum(local_measures))
    declared = _distributed_qualification_requirements()["source"]
    positive = bool(
        np.all(_distributed_process_sum(np.asarray((int(np.sum(measures <= 0.0)),))) == 0)
    )
    if (
        not positive
        or abs(area - declared["area"]) > 1.0e-10
        or abs(perimeter - declared["perimeter"]) > 1.0e-10
    ):
        raise qualification_failure(
            "_distributed_qualification_generate",
            "Owned tile cells disagree with the independent authored area/orientation/boundary.",
            {"area": area, "perimeter": perimeter, "positive": positive},
        )
    achieved = dict(result.compliance.achieved)
    record["generation"] = {
        **_result_identity(result),
        "plan_id": plan.plan_id,
        "method": "owner-local-native-planar-tile-cdt",
        "tile_complex_id": prepared.tiles.compiled_id,
        "tile_schedule_id": schedule.schedule_id,
        "tile_span_sizes": schedule.tile_span_sizes,
        "tile_count": int(prepared.tiles.tiles.shape[0]),
        "tile_owners": prepared.tile_owners.tolist(),
        "local_tiles": int(np.sum(prepared.tile_owners == jax.process_index())),
        "owned_cells_local": int(np.sum(owned_cells)),
        "closure_cells_local": int(owned_cells.size),
        "neighborhood_depth": storage.neighborhood_depth,
        "global_cells": storage.global_entity_counts[-1],
        "global_vertices": storage.global_entity_counts[0],
        "seam_rounds": achieved.get("seam_rounds"),
        "initial_evidence_id": evidence.evidence_id,
        # Placement-independent scientific identity: equal for every owner count.
        "scientific_generation_id": canonical_fingerprint(
            {
                "tiles": prepared.tiles.compiled_id,
                "topology": evidence.topology_id,
                "geometry": evidence.geometry_id,
            }
        ),
        "preparation_receipt_id": np.asarray(
            prepared.preparation_evidence.receipt_id
        ).tobytes().hex(),
        "independent_area": area,
        "independent_perimeter": perimeter,
    }
    return _DistributedGeneratedSource(
        author=source,
        request=request,
        options=options,
        source=result,
        generation=prepared,
        domain=prepared.tiles.region_domain(),
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

    plan = FiniteElementPlan(
        carrier.mesh,
        fields,
        coordinate_spec=carrier.geometry,
        precision_policy=precision,
    )
    return FiniteElementPlan.prepare(plan, numeric_version=numeric_version)


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
    transition: phx.meshing.CellMeshTransition | None = None,
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
        if kind == "rng-counter":
            # Probe counters follow allocator lineage, never value projection.
            continue
        value = compact_values[name]
        field_space = space if indexed else None
        value_units = (
            euler_history.state_units()
            if name == euler_history.ENTRY
            else None if units is None else (phx.units.KELVIN,)
        )
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
            (*raw_index.index_values.shape, *value.shape[1:]),
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
            transition=transition,
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
    recovered: _DistributedFields = {}
    for (
        name,
        raw_name,
        kind,
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
        raw, payload = rebind_meshing_field_history(
            by_name[raw_name],
            fields[raw_name],
            carrier,
            receipt,
            history_index=history_index,
            proof=proof,
        )
        # The compact role inherits the authenticated raw schema and stream.
        compact = MeshingFieldStateRole(
            name,
            raw.field_name,
            parse(kind, MeshingFieldStateKind, "role"),
            (*index.index_values.shape, *raw.shape[2:]),
            raw.dtype,
            raw.state_epoch,
            value_units=raw.value_units,
            index_binding=index,
            random_stream=raw.random_stream,
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
    # Unindexed epoch and root-key state move unchanged with the accepted record.
    for role in roles:
        if role.role in ("state-epoch", "rng-key"):
            recovered_roles.append(role)
            recovered[role.entry_name] = fields[role.entry_name]
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
    """Accepted owner-local planar source, its tile generation and authored region."""

    author: phx.meshing.NativePlanarSource
    request: phx.meshing.SurfaceMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    source: phx.meshing.CellMeshingResult
    generation: PreparedDistributedPlanarGeneration
    domain: Any


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
    rng: _DistributedRngBank
    euler_clock: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class _DistributedRngBank:
    """Probe-stream counters of every allocated raw slot of one allocator epoch.

    ``counters[p, s]`` names the next unconsumed draw event of the stable cell
    ID held by slot ``s`` of part ``p`` in ``raw_state``; ``key_data`` is the
    persisted uint32 root key of the declared stream.
    """

    key_data: jax.Array
    raw_state: phx.discretization.AdaptiveSimplexState
    counters: jax.Array


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
    prepared: phx.discretization.AdaptiveSimplexState
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
    euler_clock: dict[str, Any]


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
class _DistributedRestoredCheckpoint:
    """Producer owners restored onto this changed process set."""

    state: phx.lifecycle.MeshingCheckpointState
    arrays: dict[str, jax.Array | np.ndarray]
    closure: Mapping[str, Any]
    accepted: phx.meshing.CellMeshingResult
    placement: DistributedRestorePlacement


@final
@dataclass(frozen=True, slots=True)
class _DistributedRestartState:
    """This process's changed-placement accepted result and recovered physical roles."""

    result: phx.meshing.MeshAdaptationResult
    target: phx.meshing.CellMeshingResult
    receipt: SimplexRestartRepack
    closure: dict[str, Any]
    fields: _DistributedFields
    physical: phx.discretization.FiniteElementDiscretization


@final
@dataclass(frozen=True, slots=True)
class _DistributedPhysicsState:
    """Physical inputs on either a generated or a cold-restored owner closure."""

    target: phx.meshing.CellMeshingResult
    fields: _DistributedFields
    finite_element_fields: tuple[phx.discretization.FiniteElementFieldSpec, ...]
    precision_policy: phx.discretization.FiniteElementPrecisionPolicy
    numeric_version: str
    devices: tuple[jax.Device, ...]
    time: jax.Array
    steps: jax.Array
    euler_model: _DistributedEulerModel


@final
@dataclass(frozen=True, slots=True)
class _DistributedFEContinuation:
    """Continued FE solve measurements consumed by inventory and resource reporting."""

    solve_error: float
    free: int
    warm: list[float]
    values: jax.Array

@final
@dataclass(frozen=True, slots=True)
class _DistributedEulerAdvance:
    """Accepted conserved state, physical clock and globally owned inventory."""

    values: jax.Array
    time: jax.Array
    steps: jax.Array
    inventory: jax.Array



@final
@dataclass(frozen=True, slots=True)
class _DistributedInventories:
    """Independent accepted integrals and their analytic normalization references."""

    integrals: dict[str, float]
    references: dict[str, float]


@cache
def _distributed_rng_stream() -> MeshingRandomStream:
    """Declared address of the randomized material-history probe stream."""
    return MeshingRandomStream(
        key_entry_name="rng-root-key",
        implementation="threefry2x32",
        namespace="phydrax.meshing.qualification.distributed",
        operation="material-history-probe",
        target=("temperature", "material-cell-average"),
        purpose="rademacher",
    )


def _distributed_rng_root_key() -> jax.Array:
    """Key data of the declared typed root key; restarts read the persisted bank."""
    return jax.random.key_data(
        jax.random.key(
            _DISTRIBUTED_RNG_SEED, impl=_distributed_rng_stream().implementation
        )
    )


def _distributed_rng_row() -> tuple[
    str, str, str, str, str, str, int, bool, str, str | None
]:
    """The registry row of the cell probe counters."""
    (row,) = (row for row in _DISTRIBUTED_FIELD_ROLES if row[2] == "rng-counter")
    return row


def _distributed_collective_require(
    passed: bool, field: str, expectation: str, values: dict[str, Any], /
) -> None:
    """Refuse on every process when any process fails an accepted-state invariant."""
    failed = _distributed_process_sum(np.asarray((int(not passed),), dtype=np.int64))
    if int(failed[0]):
        raise qualification_failure(field, expectation, values)


def _distributed_rng_shard(
    value: jax.Array | np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Part indices and rows of this process's distinct shards of a part array."""
    if isinstance(value, np.ndarray):
        return np.arange(value.shape[0]), value
    parts: dict[int, np.ndarray] = {}
    for shard in value.addressable_shards:
        start = shard.index[0].start
        parts.setdefault(0 if start is None else int(start), np.asarray(shard.data))
    starts = sorted(parts)
    indices = np.concatenate(
        [np.arange(start, start + parts[start].shape[0]) for start in starts]
    )
    return indices, np.concatenate([parts[start] for start in starts])


def _distributed_rng_local(value: jax.Array | np.ndarray, /) -> np.ndarray:
    """The one raw part row this producer or changed owner holds."""
    indices, rows = _distributed_rng_shard(value)
    if indices.shape != (1,):
        raise qualification_failure(
            "_distributed_rng_local",
            "Probe counters advance only on one addressable raw part per process.",
            {"parts": indices.tolist()},
        )
    return rows[0]


def _distributed_place_owner_history(row: np.ndarray, like: jax.Array, /) -> jax.Array:
    """Place this process's raw row with the part sharding of an allocator array."""
    (shard,) = like.addressable_shards
    return jax.make_array_from_single_device_arrays(
        (like.shape[0], *row.shape), like.sharding, [jax.device_put(row[None], shard.device)]
    )


def _distributed_rng_mix(values: np.ndarray, /) -> np.ndarray:
    """SplitMix64 finalizer: a bijective wrapping mix of 64-bit words."""
    words = values.astype(np.uint64)
    words = (words ^ (words >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    words = (words ^ (words >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return words ^ (words >> np.uint64(31))


def _distributed_rng_reduce(words: np.ndarray, /) -> dict[str, object]:
    """Placement-independent multiset digest of 64-bit words over the process set."""
    local = np.asarray(
        (words.size, np.sum(words, dtype=np.uint64), np.bitwise_xor.reduce(words)),
        dtype=np.uint64,
    )
    gathered = np.asarray(
        multihost_utils.process_allgather(local, tiled=False), dtype=np.uint64
    ).reshape((jax.process_count(), 3))
    return {
        "count": int(np.sum(gathered[:, 0])),
        "sum": f"{int(np.sum(gathered[:, 1], dtype=np.uint64)):016x}",
        "xor": f"{int(np.bitwise_xor.reduce(gathered[:, 2])):016x}",
    }


def _distributed_rng_records(ids: np.ndarray, counters: np.ndarray, /) -> dict[str, object]:
    """Digest of allocated (stable cell ID, counter) records over the process set."""
    live = ids >= 0
    return _distributed_rng_reduce(
        _distributed_rng_mix(
            ids[live].astype(np.uint64) ^ _distributed_rng_mix(counters[live])
        )
    )


def _distributed_rng_bank_digest(bank: _DistributedRngBank, /) -> dict[str, object]:
    """Digest of every allocated slot counter of one probe bank."""
    return _distributed_rng_records(
        _distributed_rng_local(bank.raw_state.mesh.cell_ids),
        _distributed_rng_local(bank.counters),
    )


def _distributed_rng_owned_cells(
    carrier: phx.meshing.CellMeshingResult, history: jax.Array | np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Owned cells' stable IDs, compact rows, areas, history and independent means."""
    mesh, storage = carrier.mesh, carrier.mesh.storage
    if storage is None:
        raise qualification_failure(
            "_distributed_rng_owned_cells",
            "Probe owners require owner-local mesh storage.",
            {"carrier": carrier},
        )
    ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    owned = np.asarray(storage.entity_global_ids[-1])[
        np.asarray(storage.entity_owned[-1], dtype=np.bool_)
    ]
    rows = np.flatnonzero(np.isin(ids, owned))
    cells = _distributed_qualification_entity_cells(mesh)[rows]
    triangles = np.asarray(mesh.coordinates)[cells]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    areas = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / 2
    reference = np.mean(37.0 + 3.0 * triangles[..., 0] - triangles[..., 1], axis=1)
    values = np.asarray(history, dtype=np.float64)[rows]
    return ids[rows], rows, areas, values, reference


def _distributed_rng_probe(
    stream: MeshingRandomStream,
    key_data: jax.Array | np.ndarray,
    ids: np.ndarray,
    counters: np.ndarray,
    areas: np.ndarray,
    history: np.ndarray,
    reference: np.ndarray,
    /,
) -> dict[str, object]:
    """Randomized Rademacher residual and inventory sketch of owned cell history.

    Owned cell ``c`` draws signs ``z_c`` from its stable-ID stream at its
    counter. The process set reduces the inventory sketch ``S = sum_c z_c a_c
    h_c`` and the residual sketch ``r = sum_c z_c a_c (h_c - m_c)`` against the
    independent cell mean ``m``; ``mean(r**2)`` is the Hutchinson estimate of
    the squared residual norm. The draw digest binds every (ID, counter, signs).
    """
    probes = _DISTRIBUTED_RNG_PROBES
    if ids.size:
        keys = stream.entity_keys(key_data, ids, counters)
        signs = np.asarray(
            jax.vmap(
                lambda key: jax.random.rademacher(key, (probes,), dtype=jnp.int8)
            )(keys),
            dtype=np.float64,
        )
    else:
        signs = np.zeros((0, probes), dtype=np.float64)
    weighted = areas * history
    sketch = _distributed_process_sum(signs.T @ weighted)
    residual = _distributed_process_sum(signs.T @ (areas * (history - reference)))
    inventory = _distributed_process_sum(np.asarray((np.sum(weighted),)))
    starts = _distributed_process_sum(
        np.asarray((np.count_nonzero(counters == 0),), dtype=np.int64)
    )
    bits = np.sum(
        (signs > 0).astype(np.uint64) << np.arange(probes, dtype=np.uint64),
        axis=1,
        dtype=np.uint64,
    )
    draws = _distributed_rng_mix(
        ids.astype(np.uint64)
        ^ _distributed_rng_mix(counters.astype(np.uint64))
        ^ _distributed_rng_mix(bits)
    )
    estimate = float(np.mean(residual**2))
    probe = {
        "draws": _distributed_rng_reduce(draws),
        "counters": _distributed_rng_records(ids, counters),
        "first_stream_events": int(starts[0]),
        "probes": probes,
        "sketch": sketch.tolist(),
        "residual_estimate": estimate,
        "inventory": float(inventory[0]),
    }
    if not np.sqrt(estimate) <= _DISTRIBUTED_RNG_RESIDUAL_NORM:
        raise qualification_failure(
            "_distributed_rng_probe",
            "The randomized residual probe found history off its independent cell means.",
            {"probe": probe},
        )
    return probe


def _distributed_rng_draw(
    stream: MeshingRandomStream,
    bank: _DistributedRngBank,
    carrier: phx.meshing.CellMeshingResult,
    history: jax.Array | np.ndarray,
    /,
) -> tuple[_DistributedRngBank, dict[str, object]]:
    """Consume one probe event on every owned cell and advance its raw counter."""
    ids, _, areas, values, reference = _distributed_rng_owned_cells(carrier, history)
    raw_ids = _distributed_rng_local(bank.raw_state.mesh.cell_ids)
    live = np.flatnonzero(
        _distributed_rng_local(bank.raw_state.mesh.cell_active) & (raw_ids >= 0)
    )
    order = np.argsort(raw_ids[live], kind="stable")
    keys = raw_ids[live][order]
    aligned = keys.size == ids.size and np.array_equal(keys, np.sort(ids))
    _distributed_collective_require(
        aligned,
        "_distributed_rng_draw",
        "Owned compact cells must be exactly the active raw records of each part.",
        {"owned": int(ids.size), "active_records": int(keys.size)},
    )
    slots = live[order[np.searchsorted(keys, ids)]]
    counters = _distributed_rng_local(bank.counters)
    _distributed_collective_require(
        not np.any(counters[slots] == np.iinfo(np.uint32).max),
        "_distributed_rng_draw",
        "A probe counter would wrap its uint32 event range.",
        {"maximum": int(np.max(counters, initial=0))},
    )
    probe = _distributed_rng_probe(
        stream, bank.key_data, ids, counters[slots], areas, values, reference
    )
    advanced = counters.copy()
    advanced[slots] += 1
    placed = _distributed_place_owner_history(advanced, bank.raw_state.mesh.cell_ids)
    return replace(bank, counters=placed), probe


def _distributed_rng_require_nonreplay(
    previous: np.ndarray, candidate: np.ndarray, retained: np.ndarray, /
) -> None:
    """Refuse counters that would redraw consumed events of a retained stable ID."""
    if np.any(retained & (candidate < previous)):
        raise ValueError(
            "_distributed_rng_transport.counters must never decrease for a retained stable ID; got a replayed consumed stream address"
        )


def _distributed_rng_transport(
    bank: _DistributedRngBank,
    prepared: phx.discretization.AdaptiveSimplexState,
    edited: phx.discretization.AdaptiveSimplexState,
    /,
) -> tuple[_DistributedRngBank, dict[str, object]]:
    """Carry slot counters through one allocator edit by its declared lineage.

    Retained slots keep their stable ID and counter: refined parents freeze
    their streams and a coarsened parent resumes its own retained counter, so
    its consumed events are never redrawn. Retired children keep their final
    counters and never reactivate. A fresh slot must hold an ID issued by this
    edit (at or above the prepared cursor, never consumed) whose parent was
    active before the edit or is itself fresh; it starts at counter zero.
    """
    local = _distributed_rng_local
    before, after = local(prepared.mesh.cell_ids), local(edited.mesh.cell_ids)
    aligned = before.shape == after.shape and np.array_equal(
        local(bank.raw_state.mesh.cell_ids), before
    )
    _distributed_collective_require(
        aligned,
        "_distributed_rng_transport",
        "Probe counters must align with the prepared allocator slots of this edit.",
        {"slots": before.shape},
    )
    counters = local(bank.counters)
    was_active, active = local(prepared.mesh.cell_active), local(edited.mesh.cell_active)
    was_retired, retired = local(prepared.retired), local(edited.retired)
    # Parts share one global ID cursor: earlier IDs lie below the prepared one.
    issued, cursor = _distributed_process_maximum(
        np.asarray((local(prepared.cursors)[3], local(edited.cursors)[3]), np.int64)
    ).tolist()
    retained = before >= 0
    fresh = (after >= 0) & ~retained
    parents = local(edited.parents) // 2
    parent = np.where(fresh & (parents >= 0), parents, 0)
    lineage = (
        (parents >= 0)
        & (after[parent] >= 0)
        & (after[parent] < after)
        & ((retained & was_active) | fresh)[parent]
    )
    valid = not (
        np.any(retained & (after != before))
        or np.any(fresh & ((after < issued) | (after >= cursor) | ~lineage))
        or np.any(was_retired & active)
    )
    candidate = np.where(retained, counters, 0).astype(np.uint32)
    _distributed_rng_require_nonreplay(counters, candidate, retained)
    # Forged coarsening that restarts resumed parents at zero must be refused.
    resumed = retained & ~was_active & active & (counters > 0)
    forged = candidate.copy()
    forged[resumed] = 0
    refused = _distributed_qualification_refuses(
        lambda: _distributed_rng_require_nonreplay(counters, forged, retained)
    )
    _distributed_collective_require(
        valid and refused == bool(np.any(resumed)),
        "_distributed_rng_transport",
        "An allocator edit broke stable-ID counter lineage or admitted a forged replay.",
        {"valid": valid, "forged_refused": refused},
    )
    names = (
        "surviving_cells",
        "refined_parents",
        "retired_children",
        "reactivated_parents",
        "resumed_parents",
        "fresh_children",
        "forged_reset_owners",
        "forged_reset_refusals",
    )
    counts = _distributed_process_sum(
        np.asarray(
            (
                np.count_nonzero(retained & was_active & active),
                np.count_nonzero(retained & was_active & ~active & ~retired),
                np.count_nonzero(was_active & retired),
                np.count_nonzero(retained & ~was_active & active),
                np.count_nonzero(resumed),
                np.count_nonzero(fresh),
                int(np.any(resumed)),
                int(refused),
            ),
            dtype=np.int64,
        )
    )
    lineage_record: dict[str, object] = dict(zip(names, counts.tolist(), strict=True))
    lineage_record["policy"] = "retained-slot-counter; fresh-child-zero"
    lineage_record["parent_coarsening_nonreplay"] = (
        "exercised" if counts[4] else "not-exercised-no-resumed-parent"
    )
    placed = _distributed_place_owner_history(candidate, edited.mesh.cell_ids)
    return _DistributedRngBank(bank.key_data, edited, placed), lineage_record


def _distributed_rng_advance(bank: _DistributedRngBank, /) -> _DistributedRngBank:
    """Advance every active raw record by one event, as an accepted probe would."""
    ids = _distributed_rng_local(bank.raw_state.mesh.cell_ids)
    active = _distributed_rng_local(bank.raw_state.mesh.cell_active) & (ids >= 0)
    counters = _distributed_rng_local(bank.counters).copy()
    _distributed_collective_require(
        not np.any(counters[active] == np.iinfo(np.uint32).max),
        "_distributed_rng_advance",
        "A probe counter would wrap its uint32 event range.",
        {"maximum": int(np.max(counters, initial=0))},
    )
    counters[active] += 1
    placed = _distributed_place_owner_history(counters, bank.raw_state.mesh.cell_ids)
    return replace(bank, counters=placed)


def _distributed_rng_roles(
    carrier: phx.meshing.CellMeshingResult,
    bank: _DistributedRngBank,
    mesh_epoch: int,
    /,
) -> tuple[tuple[MeshingFieldStateRole, ...], _DistributedFields]:
    """Raw counter and root-key roles of one accepted epoch's probe stream."""
    stream = _distributed_rng_stream()
    _, raw_name, _, raw_kind, _, association, _, _, _, _ = _distributed_rng_row()
    index = prepare_meshing_field_index_binding(
        carrier, parse(f"epoch/{association}_ids", MeshingFieldIndexBank, "bank")
    )
    values = index.index_values
    aligned = (
        isinstance(values, jax.Array)
        and values.shape == bank.counters.shape
        and np.array_equal(
            _distributed_rng_local(values),
            _distributed_rng_local(bank.raw_state.mesh.cell_ids),
        )
    )
    _distributed_collective_require(
        aligned,
        "_distributed_rng_roles",
        "Probe counters must bind the accepted epoch's allocated cell bank.",
        {"shape": bank.counters.shape},
    )
    counter = MeshingFieldStateRole(
        raw_name,
        "temperature",
        parse(raw_kind, MeshingFieldStateKind, "role"),
        values.shape,
        np.dtype(bank.counters.dtype),
        mesh_epoch,
        index_binding=index,
        random_stream=stream,
    )
    key = MeshingFieldStateRole(
        stream.key_entry_name,
        "temperature",
        "rng-key",
        stream.key_shape,
        np.dtype(np.uint32),
        mesh_epoch,
        random_stream=stream,
    )
    return (counter, key), {raw_name: bank.counters, stream.key_entry_name: bank.key_data}


def _distributed_rng_generation(
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    prepared: phx.discretization.AdaptiveSimplexState,
    source: phx.meshing.CellMeshingResult,
    history: np.ndarray,
) -> _DistributedRngBank:
    """Draw the generated cells' first probe event on their prepared raw slots."""
    stream = _distributed_rng_stream()
    ids = prepared.mesh.cell_ids
    initial = _DistributedRngBank(
        jnp.asarray(_distributed_rng_root_key()),
        prepared,
        jnp.zeros(ids.shape, dtype=jnp.uint32, device=ids.sharding),
    )
    bank, draw = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "rng_generation_probe_draw",
        lambda: _distributed_rng_draw(stream, initial, source, history),
    )
    record["rng"] = {
        "status": "producer-epochs",
        "stream": asdict(stream),
        "seed": _DISTRIBUTED_RNG_SEED,
        "root_key_data": np.asarray(bank.key_data).tolist(),
        "generation": draw,
        "epochs": [],
    }
    return bank


def _distributed_rng_repack(
    receipt: SimplexRestartRepack,
    target: phx.meshing.CellMeshingResult,
    fields: _DistributedFields,
    /,
) -> dict[str, object]:
    """Prove a repack moved every counter record and active cell counter once.

    Saved records count on their saved authority and target records on their
    repacked owner; the owned compact counters recovered on the target must
    equal the saved active records, whatever the source and target placement.
    """
    name, raw_name, _, _, _, _, index, _, _, _ = _distributed_rng_row()
    saved_parts, saved_ids = _distributed_rng_shard(receipt.saved_states.mesh.cell_ids)
    _, saved_owners = _distributed_rng_shard(receipt.saved_cell_owners)
    _, saved_active = _distributed_rng_shard(receipt.saved_states.mesh.cell_active)
    _, saved = _distributed_rng_shard(receipt.saved_cell_history[index])
    authority = saved_owners == saved_parts[:, None]
    parts, target_ids = _distributed_rng_shard(receipt.states.mesh.cell_ids)
    _, owners = _distributed_rng_shard(receipt.cell_owners)
    _, moved = _distributed_rng_shard(fields[raw_name])
    ids, rows, _, _, _ = _distributed_rng_owned_cells(
        target, fields["material-cell-average"]
    )
    records = _distributed_rng_records(np.where(authority, saved_ids, -1), saved)
    active = _distributed_rng_records(
        np.where(authority & saved_active, saved_ids, -1), saved
    )
    repacked = _distributed_rng_records(
        np.where(owners == parts[:, None], target_ids, -1), moved
    )
    owned = _distributed_rng_records(ids, np.asarray(fields[name])[rows])
    if records != repacked or active != owned:
        raise qualification_failure(
            "_distributed_rng_repack",
            "A repack lost, duplicated or changed a probe counter record.",
            {"active": active, "owned": owned, "records": records, "repacked": repacked},
        )
    return {"allocated_records": repacked, "active_cells": owned}


def _distributed_rng_continue(
    stream: MeshingRandomStream,
    carrier: phx.meshing.CellMeshingResult,
    fields: _DistributedFields,
    steps: int,
    /,
) -> list[dict[str, object]]:
    """Draw the next probe events of every owned cell from its recovered counter."""
    name = _distributed_rng_row()[0]
    ids, rows, areas, values, reference = _distributed_rng_owned_cells(
        carrier, fields["material-cell-average"]
    )
    counters = np.asarray(fields[name])[rows]
    _distributed_collective_require(
        counters.dtype == np.uint32
        and not np.any(counters > np.iinfo(np.uint32).max - steps),
        "_distributed_rng_continue",
        "Recovered probe counters must be uint32 with room for the continued events.",
        {"dtype": str(counters.dtype)},
    )
    key_data = fields[stream.key_entry_name]
    return [
        _distributed_rng_probe(
            stream, key_data, ids, counters + step, areas, values, reference
        )
        for step in range(steps)
    ]


def _distributed_rng_agree(
    uninterrupted: list[dict[str, Any]], restored: list[dict[str, Any]], /
) -> bool:
    """Exact draw digests and order-tolerant sketches of two future sequences."""

    def close(first: float, second: float, scale: float) -> bool:
        return abs(first - second) <= _DISTRIBUTED_RNG_RELATIVE_TOLERANCE * scale

    return len(uninterrupted) == len(restored) and all(
        left["draws"] == right["draws"]
        and left["counters"] == right["counters"]
        and left["first_stream_events"] == right["first_stream_events"]
        and left["probes"] == right["probes"]
        and close(left["inventory"], right["inventory"], abs(left["inventory"]))
        and all(
            close(first, second, abs(left["inventory"]))
            for first, second in zip(left["sketch"], right["sketch"], strict=True)
        )
        for left, right in zip(uninterrupted, restored, strict=True)
    )


def _distributed_rng_role_refusals(
    carrier: phx.meshing.CellMeshingResult, /
) -> dict[str, bool]:
    """Check that RNG state roles admit only unsigned unitless addressed counters."""
    stream = _distributed_rng_stream()
    index = prepare_meshing_field_index_binding(carrier, "mesh/cell_ids")
    shape = index.index_values.shape

    def counter(
        dtype: np.dtype,
        units: tuple[Any, ...] | None = None,
        address: MeshingRandomStream | None = stream,
    ) -> Callable[[], object]:
        return lambda: MeshingFieldStateRole(
            "rng-counter",
            "temperature",
            "rng-counter",
            shape,
            dtype,
            0,
            value_units=units,
            index_binding=index,
            random_stream=address,
        )

    def refused(call: Callable[[], object]) -> bool:
        try:
            call()
        except (TypeError, ValueError):
            return True
        return False

    return {
        "unsigned_counter_admitted": not refused(counter(np.dtype(np.uint32))),
        "float_counter_refused": refused(counter(np.dtype(np.float64))),
        "signed_counter_refused": refused(counter(np.dtype(np.int64))),
        "unit_counter_refused": refused(
            counter(np.dtype(np.uint32), units=(phx.units.KELVIN,))
        ),
        "unaddressed_counter_refused": refused(counter(np.dtype(np.uint32), address=None)),
        "signed_root_key_refused": refused(
            lambda: MeshingFieldStateRole(
                stream.key_entry_name,
                "temperature",
                "rng-key",
                stream.key_shape,
                np.dtype(np.int32),
                0,
                random_stream=stream,
            )
        ),
    }


def _distributed_transfer_vertex_fields(
    args: QualificationControls,
    result: phx.meshing.MeshAdaptationResult,
    transfer: FiniteElementTopologyTransfer,
    values: _DistributedFields,
) -> _DistributedFields:
    """Carry every vertex field through the accepted stencil transfer.

    The transfer reads the accepted predecessor closure, then predecessor
    vertices beyond it; those values come from their accepted owners in bounded
    packets, never from extrapolation or a host-global table. Collective.
    """

    source = result.source.mesh
    storage = source.storage
    if storage is None:
        raise qualification_failure(
            "_distributed_qualification_epochs",
            "Accepted owner-local predecessor omitted its storage.",
            {"source": source},
        )
    names = tuple(name for name in values if name != euler_history.ENTRY)
    extension, owners = partitioned_transfer_sources(result)
    closure = jnp.stack([jnp.asarray(values[name]) for name in names], axis=1)
    ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    stored_rows = order[
        np.searchsorted(
            ids[order], np.asarray(storage.entity_global_ids[0], dtype=np.int64)
        )
    ]
    (remote,) = resolve_owner_local_field_values(
        (source,),
        (closure[stored_rows],),
        (extension,),
        (owners,),
        0,
        source_topology_id=storage.logical_topology_id,
        devices=tuple(jax.devices()),
        axis_name="parts",
        message_capacity=args.distributed_halo_capacity,
    )
    sources = jnp.concatenate(
        (jax.device_put(closure, next(iter(remote.devices()))), remote), axis=0
    )
    return {
        name: transfer.apply(sources[:, column]) for column, name in enumerate(names)
    }


def _distributed_accept_topology_epoch(
    args: QualificationControls,
    current: phx.meshing.CellMeshingResult,
    values: _DistributedFields,
    euler_system: phx.equations.EulerSystem,
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
    partitioned = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
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
        partitioned.layout, type(partitioned).execution_parts(partitioned), states, states.mesh.cell_active
    )
    updated = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"owner_local_{operation}_{epoch}", lambda: jax.block_until_ready(kernel())
    )
    results = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
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
    euler_values = values[euler_history.ENTRY]
    values = _distributed_transfer_vertex_fields(args, result, transfer, values)
    distribution = transition.target
    type(evidence).require_passed(evidence)
    type(current.audit).require_passed(current.audit)
    raw_layout, raw_state, _ = restore_partitioned_mesh_state(storage, evidence)
    moved_euler, euler_record = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"euler_conservative_transfer_{epoch}",
        lambda: euler_history.transfer_history(
            result, states, raw_state, euler_values, tuple(jax.devices()),
            message_capacity=args.distributed_halo_capacity,
            system=euler_system,
        ),
    )
    values[euler_history.ENTRY] = moved_euler
    record.setdefault("euler_transfers", []).append(euler_record)
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
        prepared=states,
        ids=ids,
        evidence=evidence,
        storage=storage,
        transfer=transfer,
    )


def _distributed_project_topology_epoch(
    accepted: _DistributedAcceptedEpoch,
    baselines: _DistributedFields,
    rng: _DistributedRngBank,
    fe_fields: tuple[phx.discretization.FiniteElementFieldSpec, ...],
    precision: phx.discretization.FiniteElementPrecisionPolicy,
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
    epoch: int,
    operation: str,
) -> tuple[tuple[MeshingFieldStateRole, ...], _DistributedFields, _DistributedRngBank]:
    """Project every physical role, carry and draw the probe counters, and record it."""

    current, result = accepted.current, accepted.result
    values, distribution, raw_state = (
        accepted.values,
        accepted.distribution,
        accepted.raw_state,
    )
    evidence, storage, transfer = accepted.evidence, accepted.storage, accepted.transfer
    carried, lineage = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"rng_counter_lineage_transport_{epoch}",
        lambda: _distributed_rng_transport(rng, accepted.prepared, raw_state),
    )
    physical, roles, fields = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"all_field_history_projection_{epoch}",
        lambda: _distributed_qualification_project(
            current, values, baselines, fe_fields, precision, mesh_epoch,
            transition=result.transition,
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
    advanced, draw = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"rng_probe_draw_{epoch}",
        lambda: _distributed_rng_draw(
            _distributed_rng_stream(), carried, current, fields["material-cell-average"]
        ),
    )
    rng_roles, rng_fields = _distributed_rng_roles(current, advanced, mesh_epoch)
    roles, fields = (*roles, *rng_roles), {**fields, **rng_fields}
    record["rng"]["epochs"].append(
        {"epoch": epoch, "operation": operation, "lineage": lineage, "draw": draw}
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
    return roles, fields, advanced


def _distributed_initial_field_histories(
    source: phx.meshing.CellMeshingResult,
    physical: phx.discretization.FiniteElementDiscretization,
    fields: _DistributedFields,
    prepared: phx.discretization.AdaptiveSimplexState,
) -> _DistributedFields:
    """Retain accepted generated-cell/vertex values before their first retirement."""
    result: _DistributedFields = {}
    for name, raw_name, kind, _, bank, association, _, indexed, _, _ in _DISTRIBUTED_FIELD_ROLES:
        if kind == "rng-counter":
            continue
        source_index = prepare_meshing_field_index_binding(
            source, parse(bank, MeshingFieldIndexBank, "bank"),
            field_space=physical.field_spaces[0] if indexed else None,
        )
        raw_ids = (
            prepared.mesh.vertex_ids if association == "vertex" else prepared.mesh.cell_ids
        )
        ids = _distributed_rng_local(raw_ids)
        source_ids = np.asarray(source_index.index_values)
        order = np.argsort(source_ids, kind="stable")
        live = np.flatnonzero(ids >= 0)
        positions = np.searchsorted(source_ids[order], ids[live])
        valid = bool(np.all(positions < source_ids.size))
        if valid:
            valid = np.array_equal(source_ids[order[positions]], ids[live])
        _distributed_collective_require(
            valid, "_distributed_initial_field_histories",
            "Initial allocated records must retain their accepted source field identities.",
            {"entry": name},
        )
        values = np.asarray(fields[name])
        rows = np.zeros((ids.size, *values.shape[1:]), dtype=values.dtype)
        rows[live] = values[order[positions]]
        result[raw_name] = _distributed_place_owner_history(rows, raw_ids)
    return result


def _distributed_graph_field_bucket(
    values: jax.Array | np.ndarray,
    capacity: int,
    fill: int | float,
    parts: phx.discretization.AdaptiveSimplexParts,
) -> jax.Array:
    """Place actual compact field rows in this owner's declared native bucket."""
    if values.shape[0] > capacity:
        raise qualification_failure(
            "_distributed_graph_field_bucket",
            "An indexed physical field exceeds its declared native bucket.",
            {"capacity": capacity, "shape": values.shape},
        )
    padding = ((0, capacity - values.shape[0]), *((0, 0),) * (values.ndim - 1))
    local = jnp.pad(jnp.asarray(values), padding, constant_values=fill)
    return make_owner_local_field_array(
        (local,),
        (jax.process_index(),),
        tuple(parts.mesh.devices.flat),
        axis_name=parts.axis_name,
    )


def _distributed_storage_field_values(
    carrier: phx.meshing.CellMeshingResult,
    value: jax.Array | np.ndarray,
    degree: int,
) -> np.ndarray:
    """Order locally resident physical coefficients by their storage identities."""
    storage = carrier.mesh.storage
    if storage is None:
        raise ValueError("prebalance.field_source must retain owner-local storage")
    ids = np.asarray(
        carrier.mesh.vertex_global_ids
        if degree == 0
        else type(carrier.mesh).entity_set(carrier.mesh, degree).entity_ids,
        dtype=np.int64,
    )
    stored = np.asarray(storage.entity_global_ids[degree], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    positions = np.searchsorted(ids[order], stored)
    if (
        value.shape[0] != ids.size
        or np.any(positions >= ids.size)
        or not np.array_equal(ids[order[positions]], stored)
    ):
        raise ValueError(
            "prebalance.fields must bind every resident scientific entity by identity"
        )
    return np.asarray(value)[order[positions]]


def _distributed_owned_field_content(
    carrier: phx.meshing.CellMeshingResult,
    name: str,
    value: jax.Array | np.ndarray,
    degree: int,
) -> dict[str, object]:
    """Independently digest owned bytes without gathering mesh or field banks."""
    with agreed_local_step(stage=f"prebalance-owned-content-{name}"):
        storage = carrier.mesh.storage
        if storage is None:
            raise ValueError("prebalance.field_source must retain owner-local storage")
        values = _distributed_storage_field_values(carrier, value, degree)
        ids = np.asarray(storage.entity_global_ids[degree], dtype=np.int64)
        owned = np.flatnonzero(
            np.asarray(storage.entity_owner[degree]) == storage.partition_index
        )
        prefix = (
            name.encode()
            + values.dtype.str.encode()
            + repr(values.shape[1:]).encode()
        )
        words = np.empty((owned.size, 4), dtype=np.uint64)
        for index, row in enumerate(owned):
            digest = hashlib.sha256(
                prefix + ids[row].tobytes() + values[row].tobytes()
            ).digest()
            words[index] = np.frombuffer(digest, dtype=np.uint64)
    return _distributed_rng_reduce(words.reshape(-1))


def _distributed_graph_compact_values(
    ids: np.ndarray, values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Recover compact rows from accepted cell/corner packets by stable identity."""
    if values.shape[: ids.ndim] != ids.shape:
        raise ValueError("prebalance.fields must align with accepted packet identities")
    rows = values.reshape((ids.size, *values.shape[ids.ndim :]))
    order = np.argsort(ids.reshape(-1), kind="stable")
    keys, rows = ids.reshape(-1)[order], rows[order]
    positions = np.searchsorted(keys, query)
    if np.any(positions >= keys.size) or not np.array_equal(keys[positions], query):
        raise ValueError("prebalance.fields must cover every published compact identity")
    # Shared vertex corners may repeat; selecting one must not hide differing bytes.
    row_bytes = np.ascontiguousarray(rows).view(np.uint8).reshape(
        (keys.size, int(np.prod(rows.shape[1:])) * rows.dtype.itemsize)
    )
    if np.any(
        (keys[1:] == keys[:-1]) & np.any(row_bytes[1:] != row_bytes[:-1], axis=1)
    ):
        raise ValueError("prebalance.fields must retain identical shared packet bytes")
    return rows[positions]


@final
@dataclass(frozen=True, slots=True)
class _DistributedPrebalance:
    source: phx.meshing.CellMeshingResult
    distribution: phx.meshing.MeshDistribution
    partition: phx.meshing.MeshPartitionPolicy
    policy: phx.meshing.MeshAdaptationPolicy
    physical: phx.discretization.FiniteElementDiscretization
    fields: _DistributedFields


def _distributed_prebalance_provider(
    args: QualificationControls,
    source: phx.meshing.CellMeshingResult,
    distribution: phx.meshing.MeshDistribution,
    partition: phx.meshing.MeshPartitionPolicy,
    policy: phx.meshing.MeshAdaptationPolicy,
    physical: phx.discretization.FiniteElementDiscretization,
    fields: _DistributedFields,
    clock: Mapping[str, Any],
    fe_fields: tuple[phx.discretization.FiniteElementFieldSpec, ...],
    precision: phx.discretization.FiniteElementPrecisionPolicy,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedPrebalance:
    """Migrate uneven provider ownership after warm solves, before adaptation.

    This is a canonical initial-source migration, not an adaptive restart/repack.
    All six accepted physical roles and the accepted clock survive unchanged;
    histories and the probe RNG are still initialized once at first retirement.
    """
    imbalance = float(_distributed_replicated(distribution.evidence.imbalance))
    if (
        partition.kind is not phx.meshing.MeshPartitionKind.PROVIDER
        or imbalance <= partition.maximum_imbalance
    ):
        record["generation_prebalance"] = {
            "status": "not-required",
            "provider_imbalance": imbalance,
            "maximum_imbalance": partition.maximum_imbalance,
        }
        return _DistributedPrebalance(
            source, distribution, partition, policy, physical, fields
        )
    ledger = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout))
    with agreed_local_step(stage="provider-prebalance-declaration"):
        schema = {
            row[0]: 0 if row[5] == "vertex" else source.mesh.topological_dimension
            for row in _DISTRIBUTED_FIELD_ROLES
            if row[2] != "rng-counter"
        }
        if set(fields) != set(schema):
            raise ValueError(
                "prebalance.fields must include every accepted physical state role"
            )
        graph = phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.GRAPH,
            partition.part_count,
            halo_width=partition.halo_width,
            maximum_imbalance=partition.maximum_imbalance,
            curve_depth=partition.curve_depth,
            metis_seed=partition.metis_seed,
        )
        graph_policy = _distributed_qualification_policy(policy, distribution, graph)
    adaptation = ledger.run(
        "provider_prebalance_preparation",
        lambda: phx.meshing.prepare_mesh_adaptation(
            source, phx.meshing.MarkedMeshAdaptation(), policy=graph_policy
        ),
    )
    prepared = ledger.run(
        "provider_prebalance_forest",
        lambda: prepare_partitioned_mesh_adaptation(adaptation),
    )
    raw = prepared.states
    live_parts = type(prepared).execution_parts(prepared)
    vertex_names = (
        "temperature", "material-temperature", "essential-values", "essential-mask"
    )
    with agreed_local_step(stage="provider-prebalance-field-packets"):
        storage = source.mesh.storage
        if storage is None:
            raise ValueError("prebalance.source must retain owner-local storage")
        cell_degree = source.mesh.topological_dimension
        cell_values = np.concatenate(
            (
                _distributed_storage_field_values(
                    source, fields["material-cell-average"], cell_degree
                )[:, None],
                _distributed_storage_field_values(
                    source, fields[euler_history.ENTRY], cell_degree
                ),
            ),
            axis=1,
        )
        vertex_values = np.stack(
            tuple(
                _distributed_storage_field_values(source, fields[name], 0)
                for name in vertex_names
            ),
            axis=1,
        )
    bound = ledger.run(
        "provider_prebalance_field_binding",
        lambda: prepare_ownerlocal_migration_state(
            prepared,
            raw,
            cell_field_ids=_distributed_graph_field_bucket(
                storage.entity_global_ids[cell_degree],
                prepared.layout.cell_capacity, -1, live_parts,
            ),
            cell_fields=_distributed_graph_field_bucket(
                cell_values, prepared.layout.cell_capacity, 0.0, live_parts
            ),
            vertex_field_ids=_distributed_graph_field_bucket(
                storage.entity_global_ids[0],
                prepared.layout.vertex_capacity, -1, live_parts,
            ),
            vertex_fields=_distributed_graph_field_bucket(
                vertex_values, prepared.layout.vertex_capacity, 0.0, live_parts
            ),
            cell_epochs=jnp.full(
                raw.mesh.cell_ids.shape, mesh_epoch,
                dtype=jnp.int64, device=raw.mesh.cell_ids.sharding,
            ),
        ),
    )
    proposal = ledger.run(
        "provider_prebalance_native_graph",
        lambda: prepare_partitioned_rebalance_proposal(prepared, bound, graph),
    )
    repartitioned = ledger.run(
        "provider_prebalance_repartition",
        lambda: repartition_partitioned_mesh_adaptation(
            prepared, raw, bound, proposal
        ),
    )
    with agreed_local_step(stage="provider-prebalance-migration-admission"):
        migration = repartitioned.migration
        if migration is None:
            raise ValueError("prebalance.repartition must retain canonical migration")
    accepted = ledger.run(
        "provider_prebalance_migration_acceptance",
        lambda: accept_ownerlocal_migration(migration.migration),
    )
    publication = ledger.run(
        "provider_prebalance_publication",
        lambda: commit_partitioned_adaptive_simplex(repartitioned, migration.states),
    )
    with agreed_local_step(stage="provider-prebalance-publication-admission"):
        if len(publication.local_results) != 1:
            raise ValueError("prebalance.publication must retain one local result")
        result = publication.local_results[0]
        target, transition = result.target, result.distribution
        if (
            result.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
            or transition is None
            or source.collective_evidence is None
            or not isinstance(target.collective_evidence, CollectiveMeshEvidence)
            or target.collective_evidence.migration_record is None
            or target.collective_evidence.topology_id
            != source.collective_evidence.topology_id
        ):
            raise ValueError(
                "prebalance.publication must preserve topology through canonical migration"
            )
        rebound = _distributed_qualification_policy(policy, transition.target, graph)

    def recover_fields() -> _DistributedFields:
        with agreed_local_step(stage="provider-prebalance-compact-fields"):
            valid = _distributed_rng_local(accepted.workset.cell_valid)
            cells = _distributed_graph_compact_values(
                _distributed_rng_local(accepted.workset.cell_ids)[valid],
                _distributed_rng_local(accepted.cell_fields)[valid],
                np.asarray(
                    type(target.mesh).entity_set(target.mesh, cell_degree).entity_ids
                ),
            )
            vertices = _distributed_graph_compact_values(
                _distributed_rng_local(accepted.workset.cell_vertices)[valid],
                _distributed_rng_local(accepted.vertex_fields)[valid],
                np.asarray(target.mesh.vertex_global_ids),
            )
            return {
                **{
                    name: jnp.asarray(vertices[:, column], dtype=fields[name].dtype)
                    for column, name in enumerate(vertex_names)
                },
                "material-cell-average": jnp.asarray(
                    cells[:, 0], dtype=fields["material-cell-average"].dtype
                ),
                euler_history.ENTRY: jnp.asarray(
                    cells[:, 1:], dtype=fields[euler_history.ENTRY].dtype
                ),
            }

    moved = ledger.run("provider_prebalance_compact_fields", recover_fields)
    content: dict[str, object] = {}
    for name in sorted(fields):
        degree = schema[name]
        before = ledger.run(
            f"provider_prebalance_field_before_{name}",
            lambda: _distributed_owned_field_content(source, name, fields[name], degree),
        )
        after = ledger.run(
            f"provider_prebalance_field_after_{name}",
            lambda: _distributed_owned_field_content(target, name, moved[name], degree),
        )
        _distributed_collective_require(
            before == after,
            "_distributed_prebalance_provider",
            "Native graph prebalance must preserve every owned physical field bitwise.",
            {"entry": name, "before": before, "after": after},
        )
        content[name] = {"before": before, "after": after}
    moved_physical = ledger.run(
        "provider_prebalance_physical_fields",
        lambda: _distributed_qualification_fe(
            target, fe_fields, precision, f"accepted-{mesh_epoch}"
        ),
    )
    ledger.run(
        "provider_prebalance_euler_clock",
        lambda: _distributed_euler_clock(target, moved, clock),
    )
    record["generation_prebalance"] = {
        "status": "native-graph-repartitioned-before-adaptation",
        "trigger": "actual-provider-imbalance-exceeds-unchanged-policy",
        "provider_imbalance": imbalance,
        "maximum_imbalance": graph.maximum_imbalance,
        "vertex_weights": "one-per-active-cell",
        "source_result_id": source.result_id,
        "result_id": target.result_id,
        "source_evidence_id": source.collective_evidence.evidence_id,
        "evidence_id": target.collective_evidence.evidence_id,
        "graph_policy_id": graph.policy_id,
        "proposal_id": proposal.proposal_id,
        "migration_id": target.collective_evidence.migration_record.migration_id,
        "topology_id": target.collective_evidence.topology_id,
        "owned_field_content": content,
        "physical_clock": "unchanged-accepted-object-no-time-advance",
        "rng": "declared-root-and-addresses-unchanged-initialized-once-at-first-retirement",
    }
    return _DistributedPrebalance(
        target, transition.target, graph, rebound, moved_physical, moved
    )


def _distributed_qualification_epochs(
    args: QualificationControls,
    author: Any,
    request: Any,
    source: Any,
    domain: Any,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedEpochState:
    """Accept native owner-local refine/coarsen epochs and project every physical role."""

    meshing = phx.meshing
    association = PlcAssociationTransfer(
        domain,
        source.coordinate_contract,
        author.source_revision,
        edge_vertices=np.asarray(author.region.edges, dtype=np.int64),
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=np.asarray(domain.facet_regions, dtype=np.int64),
    )
    # Provider ownership starts with the tile owners' accepted depth-one closure.
    # A required initial graph migration and the later graph restart are explicit.
    partition = meshing.MeshPartitionPolicy(
        meshing.MeshPartitionKind.PROVIDER,
        args.distributed_parts,
        halo_width=args.distributed_halo_width,
    )
    distribution = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "owner_local_tile_ownership",
        lambda: meshing.prepare_mesh_distribution(
            meshing.MeshPart(_DISTRIBUTED_PART, source), policy=partition
        ),
    )
    # Each refine epoch bisects every marked cell at most once. Capacities use
    # the collective global counts so every owner binds the same policy.
    growth = 2 ** (args.adaptation_rounds + 1)
    global_vertices = source.mesh.storage.global_entity_counts[0]
    cells = source.mesh.storage.global_entity_counts[-1]
    policy = meshing.MeshAdaptationPolicy(
        meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        compatibility=args.distributed_bisection_compatibility,
        device_policy=AdaptiveSimplexPolicy(
            vertex_capacity=min(
                args.capacity,
                args.distributed_slack * growth * global_vertices,
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
    initial_physical = _distributed_qualification_fe(
        source, fe_fields, precision, f"accepted-{mesh_epoch}"
    )
    points = initial_physical.dof_maps[0].dof_coordinates
    temperature = 1.0 + points[:, 0] + 2.0 * points[:, 1]
    values: _DistributedFields = {
        "temperature": temperature,
        "material-temperature": 37.0 + 3.0 * points[:, 0] - points[:, 1],
        "essential-values": temperature,
    }
    current, baselines = source, {}
    # The generated cells' own history probes their first stream event.
    generated_history = np.mean(
        np.asarray(values["material-temperature"])[
            _distributed_qualification_entity_cells(source.mesh)
        ],
        axis=1,
    )
    initial_fields = {
        **values,
        "essential-mask": (
            (points[:, 0] == 0.0)
            | (points[:, 0] == 1.0)
            | (points[:, 1] == 0.0)
            | (points[:, 1] == 1.0)
        ),
        "material-cell-average": jnp.asarray(generated_history),
    }
    initial_fields["temperature"] = jnp.where(
        initial_fields["essential-mask"], temperature, 0.0
    )
    density = jnp.asarray(generated_history) / 38.0
    euler_model: _DistributedEulerModel = {
        "system": phx.equations.EulerSystem(2),
        "reconstruction": phx.discretization.PiecewiseConstantReconstruction(),
        "flux": phx.discretization.RusanovFluxPlan(),
        "boundary": phx.discretization.SlipWallBoundary(),
        "time_step": jnp.asarray(0.01, dtype=jnp.float64),
    }
    initial_fields[euler_history.ENTRY] = euler_model["system"].primitive_to_conserved(
        jnp.stack((density, jnp.zeros_like(density), jnp.zeros_like(density), jnp.ones_like(density)), axis=1)
    )
    initial = _DistributedPhysicsState(
        source, initial_fields, fe_fields, precision, f"accepted-{mesh_epoch}",
        tuple(jax.devices()), jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(0, dtype=jnp.int64),
        euler_model,
    )
    solved = _distributed_continue_fe(
        args, initial, record, started, prefix="initial", continued_from_restart=False
    )
    # The accepted FE coefficients, rather than a regenerated analytic field,
    # enter every subsequent topology transfer and checkpoint.
    values["temperature"] = solved.values
    initial_fields["temperature"] = solved.values
    inventory = _distributed_check_restored_inventories(
        initial, record, record_name="initial_fields"
    )
    euler = _distributed_continue_fv(
        args, initial, inventory, record, started,
        prefix="initial", continued_from_restart=False,
    )
    values[euler_history.ENTRY] = euler.values
    initial_fields[euler_history.ENTRY] = euler.values
    euler_clock = {
        "time": euler.time,
        "steps": euler.steps,
        "time_units": phx.units.SECOND,
        "inventory": euler.inventory,
        "state_entry": euler_history.ENTRY,
        "component_names": euler_history.COMPONENTS,
        "model": euler_model,
        "model_id": canonical_fingerprint(array_tree_fingerprint(euler_model)),
    }
    prebalanced = _distributed_prebalance_provider(
        args, source, distribution, partition, policy, initial_physical,
        initial_fields, euler_clock, fe_fields, precision, record, started, mesh_epoch,
    )
    source, current = prebalanced.source, prebalanced.source
    distribution, partition, policy = (
        prebalanced.distribution, prebalanced.partition, prebalanced.policy
    )
    initial_physical, initial_fields = prebalanced.physical, prebalanced.fields
    values = {name: initial_fields[name] for name in values}
    generated_history = np.asarray(initial_fields["material-cell-average"])
    rng: _DistributedRngBank | None = None
    previous_ids: tuple[jax.Array, jax.Array] | None = None
    for epoch, operation in enumerate(
        ("refine", *(("coarsen", "refine") * args.adaptation_rounds))
    ):
        accepted = _distributed_accept_topology_epoch(
            args,
            current,
            values,
            euler_model["system"],
            policy,
            distribution,
            partition,
            previous_ids,
            record,
            started,
            epoch,
            operation,
        )
        if rng is None:
            baselines.update(
                _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
                    "initial_field_history_binding",
                    lambda: _distributed_initial_field_histories(
                        source, initial_physical, initial_fields, accepted.prepared
                    ),
                )
            )
            rng = _distributed_rng_generation(
                args, record, started, accepted.prepared, source, generated_history
            )
        current, distribution, values = (
            accepted.current,
            accepted.distribution,
            accepted.values,
        )
        previous_ids = accepted.ids
        roles, fields, rng = _distributed_project_topology_epoch(
            accepted,
            baselines,
            rng,
            fe_fields,
            precision,
            args,
            record,
            started,
            mesh_epoch,
            epoch,
            operation,
        )
    if rng is None:
        raise qualification_failure(
            "_distributed_qualification_epochs",
            "No topology epoch drew from the probe stream.",
            {"adaptation_rounds": args.adaptation_rounds},
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
        rng=rng,
        euler_clock=euler_clock,
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
        if not isinstance(binding, MeshingFieldIndexBinding):
            raise qualification_failure(
                "_distributed_prepare_graph_epoch",
                "Distributed graph field must retain its physical index binding; got None",
                {"role": role},
            )
        return binding.index_values

    bound = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
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
    proposal = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
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
    receipt = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
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

    def local_values(value: jax.Array) -> list[Any]:
        return [np.asarray(shard.data).tolist() for shard in value.addressable_shards]

    def packet_inventory(values: tuple[jax.Array, ...]) -> list[dict[str, Any]]:
        return [
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": value.size * value.dtype.itemsize,
                "local_bytes": sum(shard.data.nbytes for shard in value.addressable_shards),
            }
            for value in values
        ]

    status_codes = [
        int(code)
        for shard in receipt.status.addressable_shards
        for code in np.asarray(shard.data).reshape(-1)
    ]
    record["graph_restart_packet"] = {
        "status_local": status_codes,
        "status_reasons_local": [AdaptiveSimplexStatus(code).name for code in status_codes],
        "invalid_reason_masks_local": local_values(receipt.invalid_reason_masks),
        "duplicate_record_conflicts_local": local_values(receipt.duplicate_record_conflicts),
        "duplicate_record_schema": receipt.duplicate_record_schema(),
        "invalid_reason_bits": {
            bit.name: int(bit) for bit in SimplexRestartInvalidReason
        },
        "cell_capacity": receipt.layout.cell_capacity,
        "vertex_capacity": receipt.layout.vertex_capacity,
        "protected_edge_capacity": receipt.layout.protected_edge_capacity,
        "saved_cell_history": packet_inventory(receipt.saved_cell_history),
        "saved_vertex_history": packet_inventory(receipt.saved_vertex_history),
        "cell_history": packet_inventory(receipt.cell_history),
        "vertex_history": packet_inventory(receipt.vertex_history),
        "saved_counters_local": local_values(receipt.saved_counters),
        "saved_clocks_local": local_values(receipt.saved_clocks),
        "saved_cursors_local": local_values(receipt.saved_cursors),
        "counters_local": local_values(receipt.states.counters),
        "clocks_local": local_values(receipt.states.clocks),
        "cells_before_local": local_values(receipt.cells_before),
        "cells_after_local": local_values(receipt.cells_after),
        "vertices_before_local": local_values(receipt.vertices_before),
        "vertices_after_local": local_values(receipt.vertices_after),
    }
    reason_mask, reason_counts, reason_first = jax.device_get(
        receipt.invalid_reason_summary()
    )
    record["graph_restart_packet"].update(
        invalid_reason_mask=int(reason_mask),
        invalid_reason_target_counts=np.asarray(reason_counts).tolist(),
        invalid_reason_first_target_indices=np.asarray(reason_first).tolist(),
    )
    proof = _distributed_stage_ledger(
        record, deadline=Deadline(started + args.timeout)
    ).run("graph_restart_proof", lambda: SimplexRestartRepackProof(receipt))
    SimplexRestartRepackProof.require_receipt(proof, receipt)
    return _DistributedGraphPreparation(
        adaptation=adaptation,
        partitioned=partitioned,
        proposal=proposal,
        receipt=receipt,
        proof=proof,
    )


def _distributed_field_difference(
    role: MeshingFieldStateRole,
    actual: np.ndarray,
    expected: np.ndarray,
    identifiers: np.ndarray,
    owners: np.ndarray,
    *,
    comparison: str,
    tolerance: float,
) -> dict[str, Any]:
    """Retain typed, deterministic identity evidence before collective refusal."""
    if actual.shape != expected.shape or actual.shape[0] != identifiers.size:
        raise ValueError("Graph field comparison requires matching entity/component shapes.")
    binding = role.index_binding
    if not isinstance(binding, MeshingFieldIndexBinding):
        raise ValueError("Graph field comparison requires an entity identity binding.")
    different = actual != expected
    if np.issubdtype(actual.dtype, np.inexact):
        error = np.abs(actual - expected)
        error = np.where(np.isfinite(error), error, np.inf)
    else:
        # Exact counters must not wrap unsigned subtraction or round through float64.
        error = np.asarray(
            [abs(int(a) - int(b)) for a, b in zip(
                actual.reshape(-1), expected.reshape(-1), strict=True
            )], dtype=object,
        ).reshape(actual.shape)
    components = int(np.prod(actual.shape[1:]))
    row_error = np.max(error.reshape((identifiers.size, components)), axis=1)
    row_different = np.any(different.reshape((identifiers.size, components)), axis=1)
    differing = np.flatnonzero(row_different)
    first = (
        int(differing[np.lexsort((owners[differing], identifiers[differing]))[0]])
        if differing.size else None
    )
    maximum = max(row_error.tolist(), default=0)
    return {
        "role": role.entry_name,
        "state_epoch": role.state_epoch,
        "dtype": str(actual.dtype),
        "entity_bank": binding.bank_name,
        "comparison": comparison,
        "tolerance": tolerance,
        "max_error": maximum,
        "different_entities": int(differing.size),
        "outside_tolerance_entities": int(np.count_nonzero(row_error > tolerance)),
        "first_differing_global_id": None if first is None else int(identifiers[first]),
        "first_differing_owner": None if first is None else int(owners[first]),
        "first_actual": None if first is None else actual[first].tolist(),
        "first_expected": None if first is None else expected[first].tolist(),
    }


def _distributed_graph_transport_reference(
    accepted: _DistributedEpochState,
    receipt: SimplexRestartRepack,
    name: str,
    degree: int,
    queries: np.ndarray,
) -> np.ndarray:
    """Route accepted compact values, independently of projected/repacked histories."""
    with agreed_local_step(stage="graph-transport-reference-admission"):
        storage = accepted.current.mesh.storage
        if storage is None:
            raise ValueError("Accepted graph predecessor requires current compact ownership.")
        ids = np.asarray(storage.entity_global_ids[degree])
        owned = np.asarray(storage.entity_owned[degree])
        values = accepted.fields[name]
        if values.shape[0] != ids.size:
            raise ValueError("Accepted compact history must match current entity rows.")
    capacity = int(np.max(multihost_utils.process_allgather(np.int64(max(ids.size, 1)))))
    with agreed_local_step(stage="graph-transport-reference-packets"):
        parts = type(receipt).execution_parts(receipt)
        placement = NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))
        packet_ids = np.full((1, capacity), -1, dtype=np.int64)
        packet_ids[0, :ids.size] = np.where(owned, ids, -1)
        packet_values = np.zeros((1, capacity, *values.shape[1:]), dtype=values.dtype)
        packet_values[0, :ids.size] = np.asarray(values)
    identifiers = jax.make_array_from_process_local_data(
        placement, packet_ids, (parts.part_count, capacity)
    )
    owners = jax.make_array_from_process_local_data(
        placement, np.full((1, capacity), storage.partition_index, dtype=np.int32),
        (parts.part_count, capacity),
    )
    payload = jax.make_array_from_process_local_data(
        placement, packet_values, (parts.part_count, capacity, *values.shape[1:])
    )
    return np.asarray(_owner_local_recovery(receipt, identifiers, owners, payload, queries))


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
    results = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "graph_restart_publication",
        lambda: commit_partitioned_mesh_restart(
            receipt, adaptation, proof=prepared.proof
        ).local_results,
    )
    with agreed_local_step(stage="graph-restart-publication-admission"):
        if (
            len(results) != 1
            or results[0].status is not meshing.MeshAdaptationStatus.COMPLETE
        ):
            raise qualification_failure(
                "_distributed_qualification_graph",
                "The native graph restart epoch was not accepted by this process.",
                {"meshing": meshing, "results": results},
            )
        target, transition = results[0].target, results[0].distribution
        evidence = target.collective_evidence
        migration_counts = receipt.cell_migration_counts
        if evidence is None or transition is None or migration_counts is None:
            raise qualification_failure(
                "_distributed_qualification_graph",
                "Graph restart publication omitted its proof, distribution, or migration counts.",
                {
                    "evidence": evidence,
                    "migration_counts": migration_counts,
                    "transition": transition,
                },
            )
        type(evidence).require_passed(evidence)
        type(target.audit).require_passed(target.audit)
    fe_fields, precision = _distributed_qualification_fe_declaration()
    physical = agreed_local(
        lambda: _distributed_qualification_fe(
            target, fe_fields, precision, f"accepted-{mesh_epoch}"
        ),
        stage="graph-physical-field-preparation",
    )
    roles, recovered = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "graph_field_history_recovery",
        lambda: _distributed_qualification_recover(
            target, receipt, roles, fields, physical.field_spaces[0], prepared.proof
        ),
    )
    with agreed_local_step(stage="graph-field-comparison-admission"):
        storage = target.mesh.storage
        if storage is None:
            raise ValueError("Graph recovery requires target entity ownership.")
        by_name = {role.entry_name: role for role in roles}
        record["graph_field_inventory"] = [
            {"role": role.entry_name, "state_epoch": role.state_epoch,
             "dtype": str(role.dtype), "shape": list(role.shape)}
            for role in roles
        ]
    comparisons: list[dict[str, Any]] = []
    record["graph_field_comparisons"] = comparisons
    for row in _DISTRIBUTED_FIELD_ROLES:
        name, association = row[0], row[5]
        if name not in accepted.fields:
            continue
        degree = 0 if association == "vertex" else target.mesh.topological_dimension
        ids = np.asarray(storage.entity_global_ids[degree])
        expected = _distributed_graph_transport_reference(accepted, receipt, name, degree, ids)
        with agreed_local_step(stage="graph-transport-reference-comparison"):
            comparisons.append(_distributed_field_difference(
                by_name[name], np.asarray(recovered[name]), expected, ids,
                np.asarray(storage.entity_owner[degree]),
                comparison="accepted-compact-transport", tolerance=0.0,
            ))
    with agreed_local_step(stage="graph-physical-field-recovery"):
        mask = np.asarray(physical.dof_maps[0].boundary_dof_mask)
        recovered_mask = np.asarray(recovered["essential-mask"])
        comparisons.append(_distributed_field_difference(
            by_name["essential-mask"], recovered_mask, mask,
            np.asarray(storage.entity_global_ids[0]), np.asarray(storage.entity_owner[0]),
            comparison="independent-physical-reference", tolerance=0.0,
        ))
        if not np.array_equal(recovered_mask, mask):
            raise qualification_failure(
                "_distributed_qualification_graph",
                "Recovered essential mask differs from the repartitioned physical boundary.",
                {
                    "expected_shape": tuple(mask.shape),
                    "recovered_shape": tuple(recovered_mask.shape),
                    "mismatched_scientific_ids": (
                        np.asarray(target.mesh.vertex_global_ids)[recovered_mask != mask]
                        if recovered_mask.shape == mask.shape
                        else np.empty((0,), dtype=np.int64)
                    ).tolist(),
                },
            )
        if not np.any(~mask):
            raise qualification_failure(
                "_distributed_qualification_graph",
                "The graph epoch lost every physical interior FE unknown.",
                {"mask": mask},
            )
        points = np.asarray(physical.dof_maps[0].dof_coordinates)
        temperature = 1.0 + points[:, 0] + 2.0 * points[:, 1]
        vertices = np.asarray(target.mesh.coordinates)
        material = 37.0 + 3.0 * vertices[:, 0] - vertices[:, 1]
        references = {
            "temperature": temperature,
            "essential-values": temperature,
            "material-temperature": material,
            "material-cell-average": np.mean(
                material[_distributed_qualification_entity_cells(target.mesh)], axis=1
            ),
        }
        errors = {
            name: float(np.max(np.abs(np.asarray(recovered[name]) - reference)))
            for name, reference in references.items()
        }
        for name, reference in references.items():
            degree = target.mesh.topological_dimension if name == "material-cell-average" else 0
            comparisons.append(_distributed_field_difference(
                by_name[name], np.asarray(recovered[name]), reference,
                np.asarray(storage.entity_global_ids[degree]),
                np.asarray(storage.entity_owner[degree]),
                comparison="independent-physical-reference", tolerance=1e-12,
            ))
        transport_errors = {
            item["role"]: item["max_error"]
            for item in comparisons if item["comparison"] == "accepted-compact-transport"
        }
        if any(value != 0 for value in transport_errors.values()):
            raise qualification_failure(
                "_distributed_qualification_graph",
                "Graph history transport changed accepted compact field values.",
                transport_errors,
            )
        if max(errors.values()) > 1e-12:
            raise qualification_failure(
                "_distributed_qualification_graph",
                "Hot graph history recovery disagrees with the independent physical references.",
                errors,
            )
    time, steps = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "graph_euler_history_recovery",
        lambda: _distributed_euler_clock(target, recovered, accepted.euler_clock),
    )
    record["euler_graph"] = {
        "receipt_id": prepared.proof.content_id,
        "time": float(_distributed_replicated(time)),
        "steps": int(_distributed_replicated(steps)),
        "inventory": _distributed_replicated(accepted.euler_clock["inventory"]).tolist(),
    }
    return _DistributedGraphState(
        results=results,
        physical=physical,
        roles=roles,
        fields=recovered,
        parts=type(prepared.partitioned).execution_parts(prepared.partitioned),
        errors=errors,
        transition=transition,
        migration_counts=migration_counts,
        euler_clock=accepted.euler_clock,
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
    record["rng"]["graph"] = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "graph_rng_counter_recovery",
        lambda: _distributed_rng_repack(receipt, target, recovered.fields),
    )
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


def _distributed_bind_checkpoint_closure(
    args: QualificationControls,
    closure: dict[str, Any],
    result: phx.meshing.MeshAdaptationResult,
) -> _DistributedCheckpointBinding:
    """Bind one accepted source closure to the scientific shared-array alias bank."""

    repository, limits = _distributed_qualification_repository(
        args.distributed_repository_root()
    )
    source_arrays = bind_distributed_checkpoint_source_arrays(closure, limits)
    missing = source_arrays.missing
    if missing:
        raise qualification_failure(
            "_distributed_bind_checkpoint_closure",
            f"global source records lack scientific alias roles: {source_arrays.missing_details}",
            {"missing": missing},
        )
    return _DistributedCheckpointBinding(
        result=result,
        target=result.target,
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
    fields: _DistributedFields,
    parts: phx.discretization.AdaptiveSimplexParts,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> _DistributedCheckpointPayload:
    """Bind logical mesh and raw allocated history arrays to the scientific checkpoint."""

    result, target, closure = binding.result, binding.target, binding.closure
    limits, bindings, bank = binding.limits, binding.bindings, binding.bank
    mesh_arrays = prepare_distributed_checkpoint_mesh_arrays(
        target, fields, parts, part_count=parts.part_count, mesh_epoch=mesh_epoch
    )
    arrays, array_roles, evidence = (
        mesh_arrays.arrays,
        mesh_arrays.array_roles,
        mesh_arrays.evidence,
    )
    state = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
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


def _distributed_publish_checkpoint(
    args: QualificationControls,
    binding: _DistributedCheckpointBinding,
    payload: _DistributedCheckpointPayload,
    identities: dict[str, str],
    record: dict[str, Any],
    started: float,
    stage: str,
) -> tuple[dict[str, object], phx.lifecycle.CheckpointManifest]:
    """Publish this process's shards, then assemble the exact all-owner manifest."""

    repository, limits = binding.repository, binding.limits
    state, arrays = payload.state, payload.arrays
    publication = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        stage,
        lambda: publish_process_meshing_checkpoint(
            repository,
            identities["checkpoint_id"],
            identities["execution_id"],
            state,
            arrays,
            analysis_plan_id=identities["analysis_plan_id"],
            numeric_revision_id=identities["numeric_revision_id"],
            writer_id=f"distributed-qualification-{stage}-{jax.process_index()}",
            limits=limits,
        ),
    )
    multihost_utils.sync_global_devices(f"distributed-qualification-{stage}-published")
    manifest = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"{stage}_exact_coverage_assembly",
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
    return {
        **identities,
        "manifest_id": manifest.manifest_id,
        "state_id": state.checkpoint_state_id,
        "source_closure_id": state.source_closure_id,
        "result_id": binding.target.result_id,
        "process_index": jax.process_index(),
        "process_shards": len(publication.shards),
        "scientific_roles": [
            list(role) for role in state.array_roles if role[1] != "source-record"
        ],
        "nonaddressable_arrays": sum(
            not value.is_fully_addressable for value in arrays.values()
        ),
        "closure_roles": sorted(binding.closure),
        "accepted_data_roles": sorted(binding.closure["accepted_data"]),
    }, manifest


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

    def bind_source() -> _DistributedCheckpointBinding:
        with agreed_local_step(stage="checkpoint-source-binding"):
            fe_fields, precision = _distributed_qualification_fe_declaration()
            closure = prepare_distributed_checkpoint_source_closure(
                author,
                request,
                generated.options,
                generated.generation,
                graph.results,
                graph.physical,
                graph.roles,
                graph.fields,
                fe_fields,
                precision,
                part_name=_DISTRIBUTED_PART,
                mesh_epoch=mesh_epoch,
            )
            closure["accepted_data"]["euler_clock"] = graph.euler_clock
            return _distributed_bind_checkpoint_closure(args, closure, graph.results[0])

    binding = _distributed_stage_ledger(
        record, deadline=Deadline(started + args.timeout)
    ).run("checkpoint_source_binding", bind_source)
    payload = _distributed_bind_checkpoint_arrays(
        args, author, binding, graph.fields, graph.parts, record, started, mesh_epoch
    )
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
    published, _ = _distributed_publish_checkpoint(
        args, binding, payload, identities, record, started, "process_local_checkpoint"
    )
    record["checkpoint"] = {
        "status": "published",
        **published,
        "field_space_id": graph.physical.field_spaces[0].field_space_id,
        "mesh_epoch": mesh_epoch,
    }


def _distributed_collective_refusal(
    call: Callable[[], object], /
) -> dict[str, object] | None:
    """Run one deliberately corrupted collective operation and describe this refusal."""
    try:
        call()
    except phx.meshing.MeshingFailure as error:
        return {
            "exception": "MeshingFailure",
            "category": error.category.value,
            "stage": error.stage,
            "message": str(error)[:2000],
        }
    except (ValueError, RepositoryCorruptionError) as error:
        return {
            "exception": type(error).__name__,
            "category": None,
            "stage": None,
            "message": str(error)[:2000],
        }
    return None


def _distributed_refusal_probe(
    args: QualificationControls,
    epochs: _DistributedEpochState,
    record: dict[str, Any],
    started: float,
) -> None:
    """Corrupt only the last owner's next epoch; every owner must refuse its commit.

    The retained accepted epoch is then rechecked and continues into the graph
    repartition, so the refusal publishes nothing and rolls back to it.
    """

    meshing = phx.meshing
    current = epochs.current
    retained = current.collective_evidence
    if not isinstance(retained, CollectiveMeshEvidence):
        raise qualification_failure(
            "_distributed_refusal_probe",
            "The accepted owner-local epoch lost its collective evidence.",
            {"retained": retained},
        )
    partitioned = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "collective_refusal_probe_preparation",
        lambda: prepare_partitioned_mesh_adaptation(
            meshing.prepare_mesh_adaptation(
                current,
                meshing.MarkedMeshAdaptation(),
                policy=_distributed_qualification_policy(
                    epochs.policy, epochs.distribution, epochs.partition
                ),
            )
        ),
    )
    states = partitioned.states
    updated = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "collective_refusal_probe_coarsen",
        lambda: jax.block_until_ready(
            coarsen_adaptive_simplex_parts(
                partitioned.layout, type(partitioned).execution_parts(partitioned), states, states.mesh.cell_active
            )
        ),
    )
    owner = args.distributed_parts - 1
    mesh = updated.state.mesh
    corrupt = (jnp.arange(mesh.coordinates.shape[0]) == owner)[
        :, None, None
    ] & mesh.vertex_active[..., None]
    tampered = eqx.tree_at(
        lambda value: value.mesh.coordinates,
        updated.state,
        jnp.where(corrupt, mesh.coordinates + 0.125, mesh.coordinates),
    )
    # The refused epoch's counters advance only tentatively, as its probe would.
    def euler_state_id() -> str:
        raw = epochs.fields[euler_history.RAW_ENTRY]
        local_raw = (
            raw.addressable_shards[0].data
            if isinstance(raw, jax.Array) and not raw.is_fully_addressable
            else raw
        )
        return canonical_fingerprint(
            array_tree_fingerprint(
                (
                    epochs.fields[euler_history.ENTRY],
                    local_raw,
                    _distributed_replicated(jnp.asarray(epochs.euler_clock["time"])),
                    _distributed_replicated(jnp.asarray(epochs.euler_clock["steps"])),
                )
            )
        )

    retained_euler = euler_state_id()
    retained_rng = _distributed_rng_bank_digest(epochs.rng)
    tentative = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "collective_refusal_probe_rng_tentative",
        lambda: _distributed_rng_bank_digest(
            _distributed_rng_advance(
                _distributed_rng_transport(epochs.rng, states, tampered)[0]
            )
        ),
    )
    refusal = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "collective_refusal_probe_commit",
        lambda: _distributed_collective_refusal(
            lambda: commit_partitioned_adaptive_simplex(partitioned, tampered)
        ),
    )
    if refusal is None:
        raise qualification_failure(
            "_distributed_refusal_probe",
            "A commit holding one owner's corrupted geometry was published.",
            {"owner": owner},
        )
    type(retained).require_passed(retained)
    type(current.audit).require_passed(current.audit)
    restored_rng = _distributed_rng_bank_digest(epochs.rng)
    if restored_rng != retained_rng or tentative == retained_rng:
        raise qualification_failure(
            "_distributed_refusal_probe",
            "A refused epoch must leave the retained probe counters unchanged.",
            {"retained": retained_rng, "restored": restored_rng, "tentative": tentative},
        )
    _distributed_collective_require(
        retained_euler == euler_state_id(),
        "_distributed_refusal_probe",
        "A refused topology epoch must retain all accepted Euler components and its physical clock.",
        {"retained_euler_state_id": retained_euler},
    )
    record["euler_refusal_rollback"] = {
        "status": "accepted-euler-state-and-clock-retained",
        "local_retained_state_id": retained_euler,
    }
    record["rng"]["refusal_rollback"] = {
        "status": "rolled-back",
        "tentative_counters": tentative,
        "retained_counters": restored_rng,
    }
    record["collective_refusal"] = {
        "status": "collectively-refused",
        "corrupted_owner": owner,
        "corruption": "active-vertex-coordinates+0.125",
        "operation": "owner-local-coarsen-commit",
        "local_refusal": refusal,
        "published_results": 0,
        "retained_result_id": current.result_id,
        "retained_evidence_id": retained.evidence_id,
        "retained_epoch": record["epochs"][-1]["epoch"],
        "retained_matches_last_accepted": current.result_id
        == record["epochs"][-1]["result_id"],
    }


def _distributed_qualification_produce(
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    started: float,
    mesh_epoch: int,
) -> None:
    """Generate, adapt, refuse a corrupt epoch, repartition, and publish this owner."""

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
        args,
        generated.author,
        generated.request,
        source,
        generated.domain,
        record,
        started,
        mesh_epoch,
    )
    _distributed_refusal_probe(args, epochs, record, started)
    graph = _distributed_qualification_graph(args, epochs, record, started, mesh_epoch)
    _distributed_qualification_publish(
        args, generated, graph, record, started, mesh_epoch
    )
    # The uninterrupted owners draw the next probe events of the published
    # counters; the restored owners must reproduce them before advancing.
    target = graph.results[0].target
    record["rng"]["uninterrupted"] = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "rng_uninterrupted_continuation",
        lambda: _distributed_rng_continue(
            _distributed_rng_stream(), target, graph.fields, _DISTRIBUTED_RNG_STEPS
        ),
    )
    fe_fields, precision = _distributed_qualification_fe_declaration()
    physics = _DistributedPhysicsState(
        target, graph.fields, fe_fields, precision, f"accepted-{mesh_epoch}",
        tuple(graph.parts.mesh.devices.flat),
        jnp.asarray(graph.euler_clock["time"]),
        jnp.asarray(graph.euler_clock["steps"]),
        graph.euler_clock["model"],
    )
    inventory = _distributed_check_restored_inventories(
        physics, record, record_name="uninterrupted_fields"
    )
    _distributed_continue_fv(
        args, physics, inventory, record, started,
        prefix="uninterrupted", continued_from_restart=False,
    )
    refusals = _distributed_rng_role_refusals(target)
    if not all(refusals.values()):
        raise qualification_failure(
            "_distributed_qualification_produce",
            "RNG state roles must admit only unsigned, unitless, addressed counters.",
            {"refusals": refusals},
        )
    record["rng"]["role_refusals"] = refusals
    record["rng"]["status"] = "producer-uninterrupted-reference"


def _distributed_qualification_launch(
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    role: str,
) -> list[dict[str, Any]]:
    """Retain unchanged child records and exits before the launch deadline check.

    Reporting a completed launch is not additional scientific work: the original
    deadline still rejects a late launch, but must not discard its rank evidence.
    """

    count = {"producer": args.distributed_processes, "consumer": args.restart_parts}[
        role
    ]
    command = [
        sys.executable,
        "-m",
        "tools.meshing_qualification",
        "--scenario",
        "native-distributed-lifecycle",
    ]
    frozen = _distributed_qualification_request(args)
    for name, value in frozen.items():
        # Optional resource controls preserve the explicit null in the frozen
        # request; only actual integer limits have an integer CLI argument.
        if value is not None:
            command += [f"--{name.replace('_', '-')}", str(value)]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    root = str(Path(__file__).resolve().parents[1])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (root, *((environment["PYTHONPATH"],) if environment.get("PYTHONPATH") else ()))
    )
    environment.setdefault("PHYDRAX_CPU_COLLECTIVES", "gloo")
    environment[_DISTRIBUTED_ROLE] = role
    environment[_DISTRIBUTED_REQUEST] = json.dumps(frozen, sort_keys=True)
    record_directory = args.distributed_repository_root() / f"{role}-records"
    record_directory.mkdir(parents=True, exist_ok=False)
    environment["PHYDRAX_MESHING_RECORD_DIRECTORY"] = str(record_directory.resolve())

    processes: list[dict[str, Any]] = []
    by_rank: dict[int, dict[str, Any]] = {}
    record_errors: list[dict[str, Any]] = []

    def launch() -> tuple[Any, ...]:
        try:
            results = launch_local_processes(
                command,
                LocalProcessLaunchPlan(count, coordinator),
                environment=environment,
                cwd=os.getcwd(),
            )
        except LocalProcessLaunchError as failure:
            # Failed processes still write their phase-resolved records.
            results = failure.results
        record[f"{role}_process_results"] = [
            {
                "process_id": result.process_id,
                "returncode": result.returncode,
                "stderr": result.stderr.decode("utf-8", errors="replace"),
                "record_path": str(record_directory / f"process-{result.process_id}.json"),
            }
            for result in results
        ]
        record[f"{role}_processes"] = processes
        record[f"{role}_process_record_errors"] = record_errors
        # Read every rank before raising for any missing/malformed record. In
        # particular, StageLedger's post-call timeout must not erase this evidence.
        for rank in sorted(set(range(count)) | {result.process_id for result in results}):
            result_path = record_directory / f"process-{rank}.json"
            try:
                process = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(process, dict):
                    raise ValueError("The process record must be a JSON object.")
            except (OSError, UnicodeError, ValueError) as error:
                record_errors.append(
                    {
                        "process_id": rank,
                        "record_path": str(result_path),
                        **FailureRecord.from_exception(error, f"{role}_processes").to_record(),
                    }
                )
                continue
            by_rank[rank] = process
            processes.append(process)
        return results

    results = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"{role}_processes", launch
    )
    for result in results:
        process = by_rank.get(result.process_id)
        if process is None:
            continue
        if process["status"] != "passed":
            raise qualification_failure(
                "_distributed_qualification_launch",
                (
                    f"{role.capitalize()} process {result.process_id} failed in {process['failure_phase']}: "
                    f"{process['failure_type']}: {process['failure']}"
                ),
                {"process": process},
            )
        if result.returncode != 0:
            raise qualification_failure(
                "_distributed_qualification_launch",
                (
                    f"{role.capitalize()} process {result.process_id} exited {result.returncode} "
                    f"after recording status {process['status']!r}: "
                    + result.stderr.decode("utf-8", errors="replace")[-4000:]
                ),
                {"process": process},
            )
    if record_errors:
        error = record_errors[0]
        result = next(
            (item for item in results if item.process_id == error["process_id"]), None
        )
        raise qualification_failure(
            "_distributed_qualification_launch",
            (
                f"{role.capitalize()} process {error['process_id']} "
                + (
                    f"exited {result.returncode} without a readable record: "
                    + result.stderr.decode("utf-8", errors="replace")[-4000:]
                    if result is not None
                    else "has no launch exit result or readable record."
                )
            ),
            {"record_errors": record_errors},
        )
    if (
        len(processes) != count
        or sorted(result.process_id for result in results) != list(range(count))
    ):
        raise qualification_failure(
            "_distributed_qualification_launch",
            f"The {role} launch returned {len(processes)} records and "
            f"{len(results)} exit results for {count} processes.",
            {"processes": processes, "results": record[f"{role}_process_results"]},
        )
    request_id = canonical_fingerprint(frozen)
    for rank, process in by_rank.items():
        child_request = process.get("request")
        if (
            not isinstance(child_request, dict)
            or canonical_fingerprint(child_request) != request_id
            or process.get("request_id") != request_id
        ):
            raise qualification_failure(
                "_distributed_qualification_launch",
                f"The {role} process {rank} must retain the exact parent request "
                "and resource controls with their canonical identity.",
                {"process": process, "expected_request": frozen},
            )
    return processes


def _distributed_replicated(value: jax.Array, /) -> np.ndarray:
    """Read one small collective result through an explicit replicated placement."""
    if value.is_fully_addressable:
        return np.asarray(value)
    replicated = ExecutionRuntime.current().root_group.named_sharding(PartitionSpec())
    return np.asarray(jax.device_put(value, replicated))


def _distributed_process_sum(values: np.ndarray, /) -> np.ndarray:
    """Sum one bounded process-local vector over the process set in rank order."""
    gathered = np.asarray(multihost_utils.process_allgather(values, tiled=False))
    return np.sum(gathered.reshape((jax.process_count(), *values.shape)), axis=0)


def _distributed_process_maximum(values: np.ndarray, /) -> np.ndarray:
    """Maximum of one bounded process-local vector over the process set."""
    gathered = np.asarray(multihost_utils.process_allgather(values, tiled=False))
    return np.max(gathered.reshape((jax.process_count(), *values.shape)), axis=0)


@jax.jit
def _distributed_unique_id_moments(
    ids: jax.Array, active: jax.Array, /
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Count and moments of distinct active stable IDs in one logical record bank."""
    sentinel = jnp.iinfo(ids.dtype).max
    keys = jnp.sort(jnp.where(active, ids, sentinel).reshape(-1))
    first = (keys != sentinel) & jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), keys[1:] != keys[:-1])
    )
    values = jnp.where(first, keys, 0).astype(jnp.int64)
    return (
        jnp.sum(first, dtype=jnp.int64),
        jnp.sum(values),
        jnp.sum(values * values),
    )


def _distributed_qualification_restore(
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> _DistributedRestoredCheckpoint:
    """Restore the producer owners onto this changed process set without a host gather.

    Every consumer admits every owner record, places shared logical banks
    split across the consumer group, and reconstructs an actually addressable
    archived owner's closure. A tampered binding is refused by every process.
    """

    register_meshing_source_artifacts()
    root = args.distributed_repository_root()
    repository, limits = _distributed_qualification_repository(root)
    published = json.loads((root / _PRODUCER_PUBLICATION).read_text(encoding="utf-8"))[
        "checkpoints"
    ]
    identity_names = (
        "checkpoint_id",
        "execution_id",
        "analysis_plan_id",
        "numeric_revision_id",
    )
    expected = {int(item["process_index"]): item["state_id"] for item in published}
    if set(expected) != set(range(args.distributed_processes)) or any(
        {name: item[name] for name in identity_names}
        != {name: published[0][name] for name in identity_names}
        for item in published
    ):
        raise qualification_failure(
            "_distributed_qualification_restore",
            "Producer publication must name one checkpoint and every producer owner once.",
            {"published": published},
        )
    owner_zero = next(item for item in published if int(item["process_index"]) == 0)
    manifest = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "restore_checkpoint_coverage_assembly",
        lambda: assemble_distributed_checkpoint_from_repository(
            repository,
            owner_zero["checkpoint_id"],
            owner_zero["analysis_plan_id"],
            owner_zero["numeric_revision_id"],
            owner_zero["execution_id"],
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
    group = ExecutionRuntime.current().root_group
    relation = TopologyRestartRelation(
        owner_zero["execution_id"],
        canonical_fingerprint(record["actual_topology"]["restart_devices"]),
        "bitwise",
    )
    policy = TopologyRestartPolicy(allow_topology_change=True)
    placement = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "owner_local_restore_placement",
        lambda: prepare_distributed_restore_placement(
            states, manifest, group, limits
        ),
    )
    source_owner_index = placement.decoded_owners[0]
    source_owner = next(
        item for item in published if int(item["process_index"]) == source_owner_index
    )
    last = args.distributed_processes - 1
    swapped = {**expected, 0: expected[last], last: expected[0]}
    binding_refusal = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "consumer_binding_refusal_probe",
        lambda: _distributed_collective_refusal(
            lambda: restore_meshing_checkpoint(
                repository,
                manifest,
                placement.shardings,
                relation,
                policy,
                expected_state_id=swapped,
                source_owner_index=source_owner_index,
                limits=limits,
                maximum_host_packet_bytes=args.distributed_host_packet_bytes,
                decoded_source_owners=placement.decoded_owners,
            )
        ),
    )
    if binding_refusal is None:
        raise qualification_failure(
            "_distributed_qualification_restore",
            "A restore bound to swapped owner identities was admitted.",
            {"swapped": swapped},
        )
    checkpoint = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "changed_owner_count_restore",
        lambda: restore_meshing_checkpoint(
            repository,
            manifest,
            placement.shardings,
            relation,
            policy,
            expected_state_id=expected,
            source_owner_index=source_owner_index,
            limits=limits,
            maximum_host_packet_bytes=args.distributed_host_packet_bytes,
            decoded_source_owners=placement.decoded_owners,
        ),
    )
    state, arrays, admission = checkpoint.state, checkpoint.arrays, checkpoint.admission
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
    accepted = closure["collective_target"]
    original = require_original_meshing_source(accepted)
    if (
        not isinstance(original, InitialCollectiveMeshEvidence)
        or not isinstance(original.premise, InitialPlanarPremise)
        or original.premise.authored_source.binding_id
        != closure["generation_source"].binding_id
        or original.premise.schedule.schedule_id
        != closure["generation_schedule"].schedule_id
        or original.premise.specification.specification_id
        != closure["generation_specification"].specification_id
    ):
        raise qualification_failure(
            "_distributed_qualification_restore",
            "Restored target lost its owner-local planar root or authored generation inputs.",
            {"closure": closure, "original": original},
        )
    premise = original.premise

    def recertify() -> str:
        # Replay retained content and independently authenticate the authored
        # source and every retained P-independent tile bank.
        original.require_current()
        premise.require_current_domain()
        return premise.generation_id

    renewed = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "source_recertification", recertify
    )
    record["source_recertification"] = {
        "initial_evidence_id": original.evidence_id,
        "source_evidence_id": original.source_evidence_id,
        "tile_complex_id": renewed,
    }
    declaration = closure["field_declarations"][_DISTRIBUTED_PART]
    physical, _ = prepare_meshing_field_owner(
        declaration, accepted, index_source=accepted
    )
    if not isinstance(physical, phx.discretization.FiniteElementDiscretization):
        raise qualification_failure(
            "_distributed_qualification_restore",
            "Restored temperature must retain its actual finite-element owner.",
            {"physical": physical},
        )
    if physical.field_spaces[0].field_space_id != source_owner["field_space_id"]:
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Restored physical field space differs from its published source-owner space."),
            {"physical": physical, "published": source_owner},
        )
    if int(closure["accepted_data"]["fields"]["accepted-epoch"]) != state.mesh_epoch:
        raise qualification_failure(
            "_distributed_qualification_restore",
            ("Restored physical state epoch differs from its checkpoint epoch."),
            {"closure": closure, "state": state},
        )
    restored = {
        name: value for name, value in arrays.items() if isinstance(value, jax.Array)
    }
    summary = {
        kind: {
            "arrays": sum(value == kind for value in placement.placements.values()),
            "global_bytes": sum(
                placement.global_bytes[name]
                for name, value in placement.placements.items()
                if value == kind
            ),
            "declared_addressable_bytes": sum(
                placement.addressable_bytes[name]
                for name, value in placement.placements.items()
                if value == kind
            ),
            "measured_addressable_bytes": sum(
                shard.data.nbytes
                for name, value in placement.placements.items()
                if value == kind
                for shard in restored[name].addressable_shards
            ),
        }
        for kind in ("owner-sharded", "replicated", "process-local-owner-record")
    }
    record["checkpoint"] = {
        "status": "restored",
        **{name: owner_zero[name] for name in identity_names},
        "manifest_id": manifest.manifest_id,
        "state_ids": {str(owner): value for owner, value in sorted(expected.items())},
        "source_process_count": len(states),
        "restored_process_count": jax.process_count(),
        "decoded_source_owners": list(placement.decoded_owners),
        "addressable_source_owners": list(placement.source_owners),
        "selected_source_owner": source_owner_index,
        "source_result_ids": {
            str(item["process_index"]): item["result_id"] for item in published
        },
        "restored_field_space_id": physical.field_spaces[0].field_space_id,
        "source_closure_id": state.source_closure_id,
        "recertified_tile_complex_id": renewed,
        "binding_refusal": binding_refusal,
        "nonaddressable_arrays": sum(
            not value.is_fully_addressable for value in restored.values()
        ),
        "placement": summary,
        "host_record_bytes": sum(
            value.nbytes for value in arrays.values() if isinstance(value, np.ndarray)
        ),
        "maximum_host_packet_bytes": args.distributed_host_packet_bytes,
    }
    return _DistributedRestoredCheckpoint(state, arrays, closure, accepted, placement)


def _distributed_qualification_repack(
    args: QualificationControls,
    restored: _DistributedRestoredCheckpoint,
    record: dict[str, Any],
    started: float,
) -> _DistributedRestartState:
    """Move every accepted record onto the changed owner count and recover all roles.

    Each changed owner adopts a contiguous run of the producers' native graph
    parts. A receipt whose saved authority is corrupted must be refused by
    every process before any changed placement is published.
    """

    meshing = phx.meshing
    closure, accepted, arrays = restored.closure, restored.accepted, restored.arrays
    fields = closure["accepted_data"]["fields"]
    producers, consumers = args.distributed_processes, args.restart_parts
    storage, evidence = accepted.mesh.storage, accepted.collective_evidence
    if storage is None or not isinstance(evidence, CollectiveMeshEvidence):
        raise qualification_failure(
            "_distributed_qualification_repack",
            "Restored owner state must retain its published raw allocator evidence.",
            {"storage": storage, "evidence": evidence},
        )
    layout, raw, _ = restore_partitioned_mesh_state(storage, evidence)
    owners, vertex_owners = arrays["cell_owners"], arrays["vertex_owners"]
    if not isinstance(owners, jax.Array) or not isinstance(vertex_owners, jax.Array):
        raise qualification_failure(
            "_distributed_qualification_repack",
            "Restored authority banks must retain their distributed array placement.",
            {"cell_owners": owners, "vertex_owners": vertex_owners},
        )
    requested = jnp.where(owners >= 0, owners // (producers // consumers), 0).astype(
        jnp.int32
    )
    group = ExecutionRuntime.current().root_group
    cell_history = _distributed_qualification_histories(fields, "cell")
    vertex_history = _distributed_qualification_histories(fields, "vertex")
    receipt = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "changed_owner_count_repack",
        lambda: repack_simplex_restart(
            group,
            layout,
            raw,
            owners,
            vertex_owners,
            requested,
            cell_history=cell_history,
            vertex_history=vertex_history,
            source_result=accepted,
        ),
    )
    restart_packet: dict[str, Any] = {
        "status_local": [
            np.asarray(shard.data).tolist() for shard in receipt.status.addressable_shards
        ],
        "invalid_reason_masks_local": [
            np.asarray(shard.data).tolist()
            for shard in receipt.invalid_reason_masks.addressable_shards
        ],
        "duplicate_record_conflicts_local": [
            np.asarray(shard.data).tolist()
            for shard in receipt.duplicate_record_conflicts.addressable_shards
        ],
        "duplicate_record_schema": receipt.duplicate_record_schema(),
        "invalid_reason_bits": {
            bit.name: int(bit) for bit in SimplexRestartInvalidReason
        },
    }
    record["restart_packet"] = restart_packet
    reason_mask, reason_counts, reason_first = jax.device_get(
        receipt.invalid_reason_summary()
    )
    restart_packet.update(
        invalid_reason_mask=int(reason_mask),
        invalid_reason_target_counts=np.asarray(reason_counts).tolist(),
        invalid_reason_first_target_indices=np.asarray(reason_first).tolist(),
    )
    proof = _distributed_stage_ledger(
        record, deadline=Deadline(started + args.timeout)
    ).run("changed_owner_count_proof", lambda: SimplexRestartRepackProof(receipt))
    SimplexRestartRepackProof.require_receipt(proof, receipt)
    counts = receipt.cell_migration_counts
    if (
        counts is None
        or not bool(jnp.all(receipt.status == 0))
        or counts.shape != (producers, consumers)
        or int(jnp.sum(counts)) != int(jnp.sum(raw.mesh.cell_active))
    ):
        raise qualification_failure(
            "_distributed_qualification_repack",
            ("Changed owner-count repack did not move every accepted cell exactly once."),
            {"args": args, "counts": counts, "raw": raw, "receipt": receipt},
        )
    migration = _distributed_replicated(counts)
    received = np.sum(migration, axis=0)
    corrupt = replace(
        receipt,
        vertex_history=(
            receipt.vertex_history[0].at[0, 0].add(0.25),
            *receipt.vertex_history[1:],
        ),
    )
    rng_index = _distributed_rng_row()[6]
    # A counter rewound or skipped in transit would replay or drop probe events.
    replayed = replace(
        receipt,
        cell_history=tuple(
            value.at[0, 0].add(1) if index == rng_index else value
            for index, value in enumerate(receipt.cell_history)
        ),
    )
    euler_index = next(
        row[6] for row in _DISTRIBUTED_FIELD_ROLES if row[0] == euler_history.ENTRY
    )
    changed_euler = replace(
        receipt,
        cell_history=tuple(
            value.at[0, 0, 1].add(0.125) if index == euler_index else value
            for index, value in enumerate(receipt.cell_history)
        ),
    )
    tamper = {
        "fresh_proof_refused": _distributed_qualification_refuses(
            lambda: SimplexRestartRepackProof(corrupt)
        ),
        "bound_proof_refused": _distributed_qualification_refuses(
            lambda: SimplexRestartRepackProof.require_receipt(proof, corrupt)
        ),
        "rng_counter_fresh_proof_refused": _distributed_qualification_refuses(
            lambda: SimplexRestartRepackProof(replayed)
        ),
        "rng_counter_bound_proof_refused": _distributed_qualification_refuses(
            lambda: SimplexRestartRepackProof.require_receipt(proof, replayed)
        ),
        "euler_state_fresh_proof_refused": _distributed_qualification_refuses(
            lambda: SimplexRestartRepackProof(changed_euler)
        ),
        "euler_state_bound_proof_refused": _distributed_qualification_refuses(
            lambda: SimplexRestartRepackProof.require_receipt(proof, changed_euler)
        ),
    }
    if not all(tamper.values()):
        raise qualification_failure(
            "_distributed_qualification_repack",
            ("A tampered physical history transfer record was accepted."),
            {"tamper": tamper},
        )
    results = closure["accepted_data"]["adaptation_results"]
    with agreed_local_step(stage="restart-historical-source-admission"):
        sources = []
        for owner in restored.placement.source_owners:
            state = restored.state.source_owner_states[owner]
            source_closure = state.source_closure
            if state.source_owner_index != owner or source_closure is None:
                raise ValueError("restart.historical_sources must retain every assigned authenticated archive owner")
            sources.append(source_closure["collective_target"])
        historical_sources = tuple(sources)
    old_policy = results[0].policy
    old_partition = old_policy.partition_policy
    partition = meshing.MeshPartitionPolicy(
        meshing.MeshPartitionKind.PROVIDER,
        consumers,
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
    lost = jnp.where(owners >= 0, (owners + 1) % producers, owners).astype(jnp.int32)
    refused = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "consumer_collective_refusal_probe",
        lambda: repack_simplex_restart(
            group,
            layout,
            raw,
            lost,
            vertex_owners,
            requested,
            cell_history=cell_history,
            vertex_history=vertex_history,
            source_result=accepted,
        ),
    )
    refused_status = bool(jnp.all(refused.status != 0))
    commit_refusal = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "consumer_collective_refusal_commit",
        lambda: _distributed_collective_refusal(
            lambda: commit_partitioned_mesh_restart(
                refused, prepared, proof=SimplexRestartRepackProof(refused)
            )
        ),
    )
    if not refused_status or commit_refusal is None:
        raise qualification_failure(
            "_distributed_qualification_repack",
            "A repack with corrupted saved authority was not refused by every owner.",
            {"commit_refusal": commit_refusal, "refused_status": refused_status},
        )
    published = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "changed_owner_count_publication",
        lambda: commit_partitioned_mesh_restart(
            receipt, prepared, proof=proof, historical_sources=historical_sources
        ).local_results,
    )
    if len(published) != 1:
        raise qualification_failure(
            "_distributed_qualification_repack",
            ("Each changed owner must publish exactly one accepted local result."),
            {"published": published},
        )
    result = published[0]
    target, transition = result.target, result.distribution
    storage, evidence = target.mesh.storage, target.collective_evidence
    if (
        storage is None
        or not isinstance(evidence, CollectiveMeshEvidence)
        or transition is None
    ):
        raise qualification_failure(
            "_distributed_qualification_repack",
            (
                "Changed owner-count publication omitted its storage, proof, or distribution."
            ),
            {"evidence": evidence, "storage": storage, "transition": transition},
        )
    type(evidence).require_passed(evidence)
    type(target.audit).require_passed(target.audit)
    source_moments = _distributed_unique_id_moments(
        raw.mesh.vertex_ids, raw.mesh.vertex_active
    )
    owned_vertices = np.asarray(storage.entity_global_ids[0])[
        np.asarray(storage.entity_owned[0])
    ].astype(np.int64)
    target_moments = _distributed_process_sum(
        np.asarray(
            (
                owned_vertices.size,
                np.sum(owned_vertices),
                np.sum(owned_vertices * owned_vertices),
            ),
            dtype=np.int64,
        )
    )
    stable = [int(value) for value in source_moments] == target_moments.tolist()
    allocator = all(
        bool(jnp.array_equal(saved, value))
        for saved, value in (
            (receipt.saved_cursors, raw.cursors),
            (receipt.saved_clocks, raw.clocks),
            (receipt.saved_counters, raw.counters),
        )
    )
    if (
        storage.partition_count != consumers
        or evidence.partition_count != consumers
        or transition.source.partition.part_count != producers
        or transition.target.partition.part_count != consumers
        or np.unique(owned_vertices).size != owned_vertices.size
        or not stable
        or not allocator
    ):
        raise qualification_failure(
            "_distributed_qualification_repack",
            (
                "Changed owner-count publication lost stable IDs, ownership, or allocator clocks."
            ),
            {
                "allocator": allocator,
                "source_moments": source_moments,
                "stable": stable,
                "storage": storage,
                "target_moments": target_moments,
                "transition": transition,
            },
        )
    declaration = closure["field_declarations"][_DISTRIBUTED_PART]
    physical = _distributed_qualification_fe(
        target,
        declaration.finite_element_fields,
        declaration.precision_policy,
        declaration.numeric_version,
    )
    roles, recovered = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "restored_field_history_recovery",
        lambda: _distributed_qualification_recover(
            target,
            receipt,
            declaration.state_roles,
            fields,
            physical.field_spaces[0],
            proof,
        ),
    )
    renewed = rebind_distributed_checkpoint_field_declaration(
        declaration, target, physical, roles
    )
    new_closure: dict[str, Any] = {
        **closure,
        "collective_target": target,
        "field_declarations": {_DISTRIBUTED_PART: renewed},
        "accepted_data": {
            **closure["accepted_data"],
            "adaptation_results": published,
            "historical_owner_results": historical_sources,
            "fields": recovered,
        },
    }
    validate_meshing_field_declarations(new_closure)
    euler_clock: Mapping[str, Any] = closure["accepted_data"]["euler_clock"]
    euler_time, euler_steps = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "restored_euler_history_recovery",
        lambda: _distributed_euler_clock(target, recovered, euler_clock),
    )
    record["euler_restart"] = {
        "receipt_id": proof.content_id,
        "time": float(_distributed_replicated(euler_time)),
        "steps": int(_distributed_replicated(euler_steps)),
        "inventory": _distributed_replicated(
            jnp.asarray(euler_clock["inventory"])
        ).tolist(),
    }
    points = np.asarray(physical.dof_maps[0].dof_coordinates)
    expected = 1.0 + points[:, 0] + 2.0 * points[:, 1]
    cells = _distributed_qualification_entity_cells(target.mesh)
    vertices = np.asarray(target.mesh.coordinates)
    material = 37.0 + 3.0 * vertices[:, 0] - vertices[:, 1]
    # The physical boundary of the authored unit square, independent of ownership.
    boundary = (
        (points[:, 0] == 0.0)
        | (points[:, 0] == 1.0)
        | (points[:, 1] == 0.0)
        | (points[:, 1] == 1.0)
    )
    local_errors = np.asarray(
        (
            np.max(np.abs(np.asarray(recovered["temperature"]) - expected)),
            np.max(np.abs(np.asarray(recovered["essential-values"]) - expected)),
            np.max(np.abs(np.asarray(recovered["material-temperature"]) - material)),
            np.max(
                np.abs(
                    np.asarray(recovered["material-cell-average"])
                    - np.mean(material[cells], axis=1)
                )
            ),
            float(
                not np.array_equal(np.asarray(recovered["essential-mask"]), boundary)
            ),
        ),
        dtype=np.float64,
    )
    global_errors = _distributed_process_maximum(local_errors)
    errors = dict(
        zip(
            (
                "temperature",
                "essential-values",
                "material-temperature",
                "material-cell-average",
            ),
            global_errors[:4].tolist(),
            strict=True,
        )
    )
    if max(errors.values()) > 1e-12 or global_errors[4] != 0.0:
        raise qualification_failure(
            "_distributed_qualification_repack",
            (
                "Restored owner-local fields disagree with the independent physical references."
            ),
            {"errors": errors, "mask_mismatch": global_errors[4]},
        )
    record["restart"] = {
        "status": "owner-local-accepted-result-published",
        "source_parts": producers,
        "destination_parts": consumers,
        "placement_route": "contiguous-native-graph-part-merge",
        "receipt_id": proof.content_id,
        "restored_kind": transition.target.evidence.kind.value,
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "partition_index": storage.partition_index,
        "local_cells": target.mesh.entity_set(2).count,
        "owned_cells": int(np.count_nonzero(np.asarray(storage.entity_owned[2]))),
        "global_cells": storage.global_entity_counts[-1],
        "cell_migration_counts": migration.tolist(),
        "received_cells": received.tolist(),
        "received_cell_imbalance": float(np.max(received) / np.mean(received)),
        "tamper_refusal": tamper,
        "collective_refusal": {
            "corruption": "saved-cell-authority-shifted-one-owner",
            "all_status_refused": refused_status,
            "commit_refusal": commit_refusal,
        },
        "source_result_id": accepted.result_id,
        "field_space_id": physical.field_spaces[0].field_space_id,
        "recovered_roles": sorted(recovered),
        "reference_errors": errors,
        "essential_mask_matches_physical_boundary": True,
    }
    record["stable_ids"] = {
        "vertex_count": target_moments[0].item(),
        "preserved": stable,
        "allocator_clocks_preserved": allocator,
        "comparison": "distinct active source IDs equal disjoint owned target IDs (count, sum, sum of squares)",
    }
    record["communication"] = {
        "status": "measured-owner-message-counts",
        "cell_migration_counts": migration.tolist(),
        "message_payload_bytes": None,
        "payload_byte_status": "unmeasured",
    }
    return _DistributedRestartState(
        result, target, receipt, new_closure, recovered, physical
    )


def _distributed_rearchive_closure(
    restart: _DistributedRestartState, mesh_epoch: int,
) -> dict[str, Any]:
    """Archive current accepted arrays, retaining prior owners only as provenance."""
    declaration = restart.closure["field_declarations"][_DISTRIBUTED_PART]
    roles = declaration.state_roles
    names = tuple(role.entry_name for role in roles)
    required = {
        name for row in _DISTRIBUTED_FIELD_ROLES for name in row[:2]
    } | {"accepted-epoch", "rng-root-key"}
    if len(names) != len(set(names)) or set(names) != required:
        raise ValueError("restart.rearchive requires each physical field and history role exactly once")
    if set(restart.fields) != required:
        raise ValueError("restart.rearchive fields must match the complete accepted role inventory")
    if restart.result.target.result_id != restart.target.result_id:
        raise ValueError("restart.rearchive result must own the current accepted target")
    for role in roles:
        binding = role.index_binding
        if role.state_epoch != mesh_epoch or (
            binding is not None and binding.source_result_id != restart.target.result_id
        ):
            raise ValueError("restart.rearchive roles must retain the current accepted epoch and field owner")
    accepted = restart.closure["accepted_data"]
    closure = {
        **restart.closure,
        "collective_target": restart.target,
        "accepted_data": {
            **accepted,
            "adaptation_results": (restart.result,),
            "fields": {name: restart.fields[name] for name in names},
        },
    }
    validate_meshing_field_declarations(closure)
    return closure


def _distributed_rearchive(
    args: QualificationControls,
    restart: _DistributedRestartState,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> None:
    """Republish the restarted epoch from every changed owner, then reread its states."""

    with agreed_local_step(stage="rearchive-current-scientific-closure"):
        closure = _distributed_rearchive_closure(restart, mesh_epoch)
        binding = _distributed_bind_checkpoint_closure(args, closure, restart.result)
    payload = _distributed_bind_checkpoint_arrays(
        args,
        restart.closure["generation_source"],
        binding,
        restart.fields,
        type(restart.receipt).execution_parts(restart.receipt),
        record,
        started,
        mesh_epoch,
    )
    source = record["checkpoint"]
    identities = {
        "checkpoint_id": canonical_fingerprint(
            {
                "kind": "distributed-qualification-restarted-checkpoint",
                "request": record["request_id"],
                "source_checkpoint": source["checkpoint_id"],
            }
        ),
        "execution_id": canonical_fingerprint(
            record["actual_topology"]["restart_devices"]
        ),
        "analysis_plan_id": source["analysis_plan_id"],
        "numeric_revision_id": source["numeric_revision_id"],
    }
    published, manifest = _distributed_publish_checkpoint(
        args, binding, payload, identities, record, started, "restored_closure_rearchive"
    )
    states = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "restored_closure_reread",
        lambda: read_meshing_checkpoint_states(
            binding.repository, manifest, limits=binding.limits
        ),
    )
    reread = {state.source_owner_index: state.checkpoint_state_id for state in states}
    if (
        set(reread) != set(range(args.restart_parts))
        or reread[jax.process_index()] != payload.state.checkpoint_state_id
        or any(state.mesh_epoch != mesh_epoch for state in states)
    ):
        raise qualification_failure(
            "_distributed_rearchive",
            "The restarted owners' republished states differ from their accepted states.",
            {"published": published, "reread": reread},
        )
    record["rearchive"] = {
        "status": "published-and-reread",
        **published,
        "reread_state_ids": {str(owner): value for owner, value in sorted(reread.items())},
    }


def _distributed_dirichlet_action(
    policy: DistributedKrylovPolicy,
) -> Callable[..., tuple[jax.Array, ...]]:
    """Owner-local constrained Laplace solve with exactly-once global pairings."""

    def action(
        program: OwnerLocalFiniteElementDiscretization,
        values: jax.Array,
        essential: jax.Array,
        reference: jax.Array,
        initial: jax.Array,
    ) -> tuple[jax.Array, ...]:
        stiffness = program.operator("stiffness")
        owned = program.owned_mask
        shape = program.execution_vector_shape

        def apply(vector: jax.Array) -> jax.Array:
            interior = stiffness.mv(jnp.where(essential, 0.0, vector))
            return jnp.where(owned, jnp.where(essential, vector, interior), 0.0)

        operator = DistributedLinearOperator(
            apply,
            apply,
            shape,
            shape,
            operator_id=canonical_fingerprint(
                {"kind": "dirichlet-stiffness", "program": program.prepared_id}
            ),
        )
        lifted = stiffness.mv(jnp.where(essential, values, 0.0))
        rhs = jnp.where(owned, jnp.where(essential, values, -lifted), 0.0)
        solved = solve_distributed_pcg(
            operator,
            rhs,
            program.pairing,
            policy,
            initial=jnp.where(owned, initial, 0.0),
        )
        true_residual = rhs - operator.mv(solved.value)
        true_residual_norm = jnp.sqrt(jnp.maximum(
            jnp.real(program.pairing.inner(true_residual, true_residual)), 0.0
        ))
        rhs_norm = jnp.sqrt(jnp.maximum(
            jnp.real(program.pairing.inner(rhs, rhs)), 0.0
        ))
        threshold = policy.absolute_tolerance + policy.relative_tolerance * rhs_norm
        error = jax.lax.pmax(
            jnp.max(jnp.where(owned, jnp.abs(solved.value - reference), 0.0)),
            program.axis_name,
        )
        free = jax.lax.psum(
            jnp.sum(owned & ~essential, dtype=jnp.int64), program.axis_name
        )
        accepted = program.pairing.global_all(
            program.collective_certificate()
            & solved.converged
            & jnp.all(jnp.isfinite(solved.value))
        )
        return (
            program.halo.exchange(
                solved.value, program.halo.partition_index, axis_name=program.axis_name
            ),
            error,
            free,
            accepted,
            solved.iterations,
            solved.residual_norm,
            true_residual_norm,
            threshold,
            solved.breakdown,
            solved.converged,
        )

    return action


def _distributed_continue_fe(
    args: QualificationControls,
    state: _DistributedPhysicsState,
    record: dict[str, Any],
    started: float,
    *,
    prefix: str = "continued",
    continued_from_restart: bool = True,
) -> _DistributedFEContinuation:
    """Solve the Dirichlet problem on generated or restored owner-local closures."""

    target, fields = state.target, state.fields
    devices = state.devices
    halos = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"{prefix}_fe_halo_preparation",
        lambda: prepare_owner_local_halo_plans(
            (target.mesh,),
            0,
            devices=devices,
            axis_name="parts",
            message_capacity=args.distributed_halo_capacity,
        ),
    )
    program = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"{prefix}_fe_preparation",
        lambda: prepare_owner_local_finite_element(
            FiniteElementPlan(
                target.mesh,
                state.finite_element_fields,
                coordinate_spec=target.geometry,
                precision_policy=state.precision_policy,
            ),
            halos[0],
            axis_name="parts",
            numeric_version=state.numeric_version,
        ),
    )
    points = np.asarray(program.local_discretization.dof_maps[0].dof_coordinates)
    if program.local_dof_count != fields["temperature"].shape[0]:
        raise qualification_failure(
            "_distributed_continue_fe",
            "The owner-local FE closure differs from the recovered physical field rows.",
            {"program": program},
        )
    reference = jnp.asarray(1.0 + points[:, 0] + 2.0 * points[:, 1])
    inputs = (
        (jnp.asarray(fields["essential-values"]),),
        (jnp.asarray(fields["essential-mask"]),),
        (reference,),
        (jnp.asarray(fields["temperature"]),),
    )
    # Unequal owner closures share one bucket bounded by the frozen controls.
    limits = FiniteElementExecutionLimits(
        args.capacity,
        args.capacity,
        9 * args.capacity,
        args.distributed_halo_capacity,
        args.distributed_host_packet_bytes,
    )
    # Conjugate gradients terminate within the global unknown count.
    action = _distributed_dirichlet_action(
        DistributedKrylovPolicy(
            program.global_dof_count, relative_tolerance=1e-12, absolute_tolerance=1e-13
        )
    )

    execution = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"{prefix}_fe_execution_preparation",
        lambda: prepare_owner_local_finite_element_execution(
            (program,),
            inputs,
            action,
            devices,
            execution_limits=limits,
            budget=None,
        ),
    )

    def solve() -> tuple[jax.Array, ...]:
        return jax.block_until_ready(execution())

    (
        solution, error, free, accepted, iterations, residual,
        true_residual, threshold, breakdown, converged,
    ) = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"{prefix}_fe_solve", solve
    )
    warm = []
    for repeat in range(args.repeats):
        name = f"{prefix}_fe_warm_solve_{repeat}"
        _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(name, solve)
        warm.append(record["stages_seconds"][name])

    def local(value: jax.Array) -> np.ndarray:
        return np.asarray(value.addressable_shards[0].data)[0]

    solve_error = float(local(error))
    free_dofs = int(local(free))
    record[f"{prefix}_fe_solver"] = {
        "dtype": str(solution.dtype),
        "reported_residual_norm": float(local(residual)),
        "true_residual_norm": float(local(true_residual)),
        "residual_threshold": float(local(threshold)),
        "breakdown": bool(local(breakdown)),
        "converged": bool(local(converged)),
        "collectively_accepted": bool(local(accepted)),
        "affine_forward_max_error": solve_error,
        "forward_error_bound": None,
    }
    if not bool(local(accepted)) or free_dofs == 0 or solve_error > 1e-10:
        raise qualification_failure(
            "_distributed_continue_fe",
            (
                "Continued owner-local FE solve was refused or disagrees with the affine reference."
            ),
            {"accepted": local(accepted), "free": free_dofs, "solve_error": solve_error},
        )
    record[f"{prefix}_fe"] = {
        "status": "executed",
        "physical_problem": "Dirichlet-Laplace-affine-temperature",
        "route": "owner-local-distributed-pcg",
        "global_space_id": program.global_space_id,
        "local_space_id": program.local_space_id,
        "global_dofs": program.global_dof_count,
        "local_dofs": program.local_dof_count,
        "owned_dofs": int(np.count_nonzero(np.asarray(program.owned_mask))),
        "free_dofs": free_dofs,
        "solve_max_error": solve_error,
        "iterations": int(local(iterations)),
        "residual_norm": float(local(residual)),
        "collectively_accepted": True,
        "warm_samples_seconds": warm,
        "continued_from_restart": continued_from_restart,
    }
    return _DistributedFEContinuation(
        solve_error=solve_error, free=free_dofs, warm=warm,
        values=solution.addressable_shards[0].data[0, : program.local_dof_count],
    )


def _distributed_check_restored_inventories(
    restart: _DistributedPhysicsState,
    record: dict[str, Any],
    *,
    record_name: str = "restored_fields",
) -> _DistributedInventories:
    """Reduce owned-cell inventories across owners and check independent integrals."""

    target, fields = restart.target, restart.fields
    storage = target.mesh.storage
    if storage is None:
        raise qualification_failure(
            "_distributed_check_restored_inventories",
            "Restored mesh.storage must retain owner-local ownership; got None",
            {"target": target},
        )
    cells = _distributed_stage_ledger(record).run(
        f"{record_name}_integral_validation",
        lambda: _distributed_qualification_entity_cells(target.mesh),
    )
    owned = np.isin(
        np.asarray(target.mesh.entity_set(2).entity_ids),
        np.asarray(storage.entity_global_ids[2])[np.asarray(storage.entity_owned[2])],
    )
    triangles = np.asarray(target.mesh.coordinates)[cells]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / 2
    names = ("area", "temperature", "material-temperature", "material-cell-average")
    local = np.asarray(
        (
            np.sum(area[owned]),
            np.sum(
                (area * np.mean(np.asarray(fields["temperature"])[cells], axis=1))[owned]
            ),
            np.sum(
                (
                    area
                    * np.mean(np.asarray(fields["material-temperature"])[cells], axis=1)
                )[owned]
            ),
            np.sum((area * np.asarray(fields["material-cell-average"]))[owned]),
        ),
        dtype=np.float64,
    )
    integrals = dict(zip(names, _distributed_process_sum(local).tolist(), strict=True))
    owned_cells = int(
        _distributed_process_sum(np.asarray(np.count_nonzero(owned), dtype=np.int64))
    )
    references = {
        "area": 1.0,
        "temperature": 2.5,
        "material-temperature": 38.0,
        "material-cell-average": 38.0,
    }
    if owned_cells != storage.global_entity_counts[-1] or any(
        abs(integrals[name] - value) > 1e-12 for name, value in references.items()
    ):
        raise qualification_failure(
            "_distributed_check_restored_inventories",
            ("Restored physical inventories disagree with their independent integrals."),
            {"integrals": integrals, "owned_cells": owned_cells, "references": references},
        )
    record[record_name] = {
        "reopened_state_roles": sorted(fields),
        "integrals": integrals,
        "independent_references": references,
        "owned_cell_cover": owned_cells,
        "reduction": "owned-cell partial sums over the process set",
        "euler_inventory": euler_history.inventory(
            target, fields[euler_history.ENTRY], system=restart.euler_model["system"]
        ).tolist(),
    }
    return _DistributedInventories(integrals=integrals, references=references)


def _distributed_euler_clock(
    carrier: phx.meshing.CellMeshingResult,
    fields: _DistributedFields,
    clock: Mapping[str, Any],
) -> tuple[jax.Array, jax.Array]:
    """Validate the persisted physical clock against its actual accepted Euler bank."""
    schemas = (("time", (), np.float64), ("steps", (), np.int64), ("inventory", (4,), np.float64))
    valid = all(
        isinstance(clock.get(name), (jax.Array, np.ndarray))
        and clock[name].shape == shape
        and clock[name].dtype == np.dtype(dtype)
        for name, shape, dtype in schemas
    )
    valid = (
        valid
        and clock.get("state_entry") == euler_history.ENTRY
        and tuple(clock.get("component_names", ())) == euler_history.COMPONENTS
        and isinstance(clock.get("time_units"), phx.units.UnitDefinition)
        and clock["time_units"] == phx.units.SECOND
    )
    model = clock.get("model")
    valid = (
        valid
        and isinstance(model, Mapping)
        and type(model.get("system")) is phx.equations.EulerSystem
        and model["system"].dimension == 2
        and type(model.get("reconstruction")) is phx.discretization.PiecewiseConstantReconstruction
        and type(model.get("flux")) is phx.discretization.RusanovFluxPlan
        and type(model.get("boundary")) is phx.discretization.SlipWallBoundary
        and isinstance(model.get("time_step"), (jax.Array, np.ndarray))
        and model["time_step"].shape == ()
        and model["time_step"].dtype == np.dtype(np.float64)
    )
    _distributed_collective_require(
        valid,
        "_distributed_euler_clock",
        "Euler checkpoint state must retain its four components, SI units and numerical clock.",
        {"clock": clock},
    )
    assert isinstance(model, Mapping)
    time_step = float(_distributed_replicated(jnp.asarray(model["time_step"])))
    _distributed_collective_require(
        clock.get("model_id") == canonical_fingerprint(array_tree_fingerprint(model))
        and bool(np.isfinite(time_step))
        and time_step > 0,
        "_distributed_euler_clock",
        "Restored Euler dynamics must retain the producer's actual model and time step.",
        {"model_id": clock.get("model_id"), "time_step": time_step},
    )
    time, steps = jnp.asarray(clock["time"]), jnp.asarray(clock["steps"])
    physical_time = float(_distributed_replicated(time))
    accepted_steps = int(_distributed_replicated(steps))
    measured = euler_history.inventory(
        carrier, fields[euler_history.ENTRY], system=model["system"]
    )
    expected = _distributed_replicated(jnp.asarray(clock["inventory"]))
    _distributed_collective_require(
        bool(np.isfinite(physical_time))
        and physical_time > 0
        and accepted_steps >= 1
        and bool(np.all(np.abs(measured - expected) <= euler_history.INVENTORY_TOLERANCE)),
        "_distributed_euler_clock",
        "Accepted Euler content and time must survive migration and cold restore without reinitialization.",
        {"inventory": measured.tolist(), "expected": expected.tolist(), "time": physical_time, "steps": accepted_steps},
    )
    return time, steps


def _distributed_accept_continued_fields(
    restart: _DistributedRestartState,
    finite_element: _DistributedFEContinuation,
    euler: _DistributedEulerAdvance,
    mesh_epoch: int,
) -> _DistributedRestartState:
    """Republish every advanced field, raw history, RNG state and physical clock together."""
    with agreed_local_step(stage="continued-all-field-acceptance-admission"):
        declaration = restart.closure["field_declarations"][_DISTRIBUTED_PART]
        baselines = {
            role.entry_name: restart.fields[role.entry_name]
            for role in declaration.state_roles
            if role.index_binding is not None
            and role.index_binding.bank_name.startswith("epoch/")
        }
    physical, roles, fields = _distributed_qualification_project(
        restart.target,
        {
            **restart.fields,
            "temperature": finite_element.values,
            euler_history.ENTRY: euler.values,
        },
        baselines,
        declaration.finite_element_fields,
        declaration.precision_policy,
        mesh_epoch,
    )
    with agreed_local_step(stage="continued-all-field-acceptance-roles"):
        rng_roles = tuple(
            role for role in declaration.state_roles
            if role.role in ("rng-key", "rng-counter")
        )
        roles = (*roles, *rng_roles)
        fields.update({role.entry_name: restart.fields[role.entry_name] for role in rng_roles})
    renewed = rebind_distributed_checkpoint_field_declaration(
        declaration, restart.target, physical, roles
    )
    with agreed_local_step(stage="continued-all-field-acceptance-publication"):
        accepted = restart.closure["accepted_data"]
        clock = {
            **accepted["euler_clock"],
            "time": euler.time,
            "steps": euler.steps,
            "inventory": euler.inventory,
        }
        closure = {
            **restart.closure,
            "field_declarations": {_DISTRIBUTED_PART: renewed},
            "accepted_data": {**accepted, "fields": fields, "euler_clock": clock},
        }
    validate_meshing_field_declarations(closure)
    _distributed_euler_clock(restart.target, fields, clock)
    return replace(restart, fields=fields, physical=physical, closure=closure)


def _distributed_continue_fv(
    args: QualificationControls,
    physics: _DistributedPhysicsState,
    inventory: _DistributedInventories,
    record: dict[str, Any],
    started: float,
    *,
    prefix: str = "continued",
    continued_from_restart: bool = True,
) -> _DistributedEulerAdvance:
    """Advance the actual accepted Euler bank and its physical clock."""

    _ledger = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout))
    target, fields = physics.target, physics.fields
    integrals = inventory.integrals
    mesh = target.mesh
    storage = mesh.storage

    system = physics.euler_model["system"]
    if storage is None:
        raise qualification_failure(
            "_distributed_continue_fv",
            "Restored FV mesh.storage must retain owner-local closure evidence; got None",
            {"mesh": mesh},
        )
    def prepare() -> tuple[
        UnstructuredFiniteVolumeDiscretization, PreparedUnstructuredFiniteVolumeDynamics
    ]:
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
                name: physics.euler_model["boundary"]
                for name in geometry.boundary_patch_names
            },
        )
        dynamics = PreparedUnstructuredFiniteVolumeDynamics(
            system,
            geometry,
            phx.discretization.UnstructuredFiniteVolumeMethodPlan(
                physics.euler_model["reconstruction"],
                physics.euler_model["flux"],
            ),
            boundaries,
        )
        return geometry, dynamics

    geometry, dynamics = _ledger.run(f"{prefix}_fv_preparation", prepare)
    count = geometry.cell_centers.shape[0]
    seed = jnp.asarray(fields[euler_history.ENTRY])
    if seed.shape != (count, 4):
        raise qualification_failure(
            "_distributed_continue_fv",
            "Accepted Euler history must retain density, both momenta and total energy.",
            {"shape": seed.shape, "cells": count},
        )
    devices = physics.devices
    ranks = tuple(
        index
        for index, device in enumerate(devices)
        if device.process_index == jax.process_index()
    )
    if ranks != (storage.partition_index,):
        raise qualification_failure(
            "_distributed_continue_fv",
            "The restored owner is not placed on this process's execution device.",
            {"partition_index": storage.partition_index, "ranks": ranks},
        )
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
            mesh_axes=(("parts", len(devices)),),
        ),
        devices,
    )
    halos = _ledger.run(
        f"{prefix}_fv_halo_preparation",
        lambda: prepare_owner_local_halo_plans(
            (mesh,),
            2,
            devices=devices,
            axis_name="parts",
            message_capacity=args.distributed_halo_capacity,
        ),
    )
    # One collective host sync agrees the padded execution buckets of unequal
    # restored closures; padding is inactive storage, never physical cells.
    execution = _ledger.run(
        f"{prefix}_fv_execution_preparation",
        lambda: prepare_owner_local_finite_volume_execution(
            ((dynamics, halos[0]),),
            group,
            axis_name="parts",
            maximum_cells=storage.global_entity_counts[-1],
            maximum_faces=4 * storage.global_entity_counts[1],
            maximum_message_entries=args.distributed_halo_capacity,
            maximum_storage_bytes=args.maximum_memory_bytes,
        ),
    )
    state = execution.pack_state((seed,))
    advance = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        f"{prefix}_owner_local_fv",
        lambda: jax.block_until_ready(
            execute_owner_local_finite_volume_advance(
                execution, state, physics.time, jnp.asarray(physics.euler_model["time_step"])
            )
        ),
    )
    drift = float(
        np.max(
            np.abs(
                _distributed_replicated(advance.inventory_after)
                - _distributed_replicated(advance.inventory_before)
            )
        )
    )
    owned = np.asarray(geometry.cell_owned)
    updated = np.asarray(execution.unpack_state(advance.state)[0])
    changed = int(
        _distributed_process_sum(
            np.asarray(
                np.count_nonzero(updated[owned, 0] != np.asarray(seed)[owned, 0]),
                dtype=np.int64,
            )
        )
    )
    admissible = bool(
        np.all(_distributed_replicated(advance.finite & advance.admissible))
    )
    if not admissible or drift > 1e-12 or changed == 0:
        raise qualification_failure(
            "_distributed_continue_fv",
            ("Continued FV advance is inadmissible, nonconservative, or stationary."),
            {"admissible": admissible, "changed": changed, "drift": drift},
        )
    accepted_values = _ledger.run(
        f"{prefix}_fv_accepted_ghost_synchronization",
        lambda: euler_history.synchronize(
            target, updated, devices, message_capacity=args.distributed_halo_capacity
        ),
    )
    accepted_time = physics.time + jnp.asarray(physics.euler_model["time_step"])
    accepted_steps = physics.steps + jnp.asarray(1, dtype=jnp.int64)
    record[f"{prefix}_fv"] = {
        "status": "executed",
        "physical_problem": "material-density-slip-wall-Euler",
        "route": "owner-local-halo-exchange-fixed-capacity-buckets",
        "execution_layout": {
            "logical_closure_cells_local": int(count),
            "cell_capacity": int(execution.cell_capacity),
            "padding_is_inactive_storage": True,
        },
        "changed_cells": changed,
        "inventory_drift": drift,
        "system_id": system.system_id,
        "model_id": canonical_fingerprint(array_tree_fingerprint(physics.euler_model)),
        "continued_from_restart": continued_from_restart,
        "state_entry": euler_history.ENTRY,
        "component_names": list(euler_history.COMPONENTS),
        "inventory_before": _distributed_replicated(advance.inventory_before).tolist(),
        "inventory_after": _distributed_replicated(advance.inventory_after).tolist(),
        "time_before": float(_distributed_replicated(physics.time)),
        "time_after": float(_distributed_replicated(accepted_time)),
        "steps_before": int(_distributed_replicated(physics.steps)),
        "steps_after": int(_distributed_replicated(accepted_steps)),
        "state_moments": euler_history.moments(target, accepted_values).tolist(),
    }
    record["conservation"] = {
        "status": "measured",
        "material_cell_inventory": integrals["material-cell-average"],
        "fv_inventory_drift": drift,
    }
    return _DistributedEulerAdvance(
        accepted_values, accepted_time, accepted_steps, advance.inventory_after
    )


def _distributed_rng_restart(
    args: QualificationControls,
    restart: _DistributedRestartState,
    record: dict[str, Any],
    started: float,
) -> _DistributedRestartState:
    """Reproduce the uninterrupted probe events from restored counters, then advance.

    The persisted stream address and root key must be the declared ones and
    the changed-placement repack must move every counter record once. The
    restored owners' next draws must equal the uninterrupted producers' draws;
    every counter then advances past the consumed events, so the republished
    state never replays them.
    """
    name, raw_name, *_ = _distributed_rng_row()
    fields, receipt, steps = restart.fields, restart.receipt, _DISTRIBUTED_RNG_STEPS
    stream = _distributed_rng_stream()
    declaration = restart.closure["field_declarations"][_DISTRIBUTED_PART]
    persisted = {
        role.entry_name: role.random_stream
        for role in declaration.state_roles
        if role.random_stream is not None
    }
    key_data, raw_values = fields[stream.key_entry_name], fields[raw_name]
    if (
        persisted != {name: stream, raw_name: stream, stream.key_entry_name: stream}
        or not np.array_equal(np.asarray(key_data), np.asarray(_distributed_rng_root_key()))
        or not isinstance(raw_values, jax.Array)
    ):
        raise qualification_failure(
            "_distributed_rng_restart",
            "The restored probe stream must persist the declared address and root key.",
            {"persisted": persisted},
        )
    repack = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "restored_rng_counter_recovery",
        lambda: _distributed_rng_repack(receipt, restart.target, fields),
    )
    publication = json.loads(
        (args.distributed_repository_root() / _PRODUCER_PUBLICATION).read_text(
            encoding="utf-8"
        )
    )
    restored = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "rng_restored_continuation",
        lambda: _distributed_rng_continue(stream, restart.target, fields, steps),
    )
    if not _distributed_rng_agree(publication["rng_uninterrupted"], restored):
        raise qualification_failure(
            "_distributed_rng_restart",
            "Restored owners drew different future probe events than the uninterrupted run.",
            {"restored": restored, "uninterrupted": publication["rng_uninterrupted"]},
        )
    part = int(_distributed_rng_shard(receipt.states.mesh.cell_ids)[0][0])
    live = (
        (_distributed_rng_local(receipt.states.mesh.cell_ids) >= 0)
        & (_distributed_rng_local(receipt.cell_owners) == part)
        & _distributed_rng_local(receipt.states.mesh.cell_active)
    )
    counters = _distributed_rng_local(raw_values).copy()
    compact = jnp.asarray(fields[name])
    room = np.iinfo(np.uint32).max - steps
    _distributed_collective_require(
        not np.any(counters[live] > room) and not bool(jnp.any(compact > room)),
        "_distributed_rng_restart",
        "A restored probe counter would wrap its uint32 event range.",
        {"maximum": int(np.max(counters, initial=0))},
    )
    counters[live] += steps
    advanced: _DistributedFields = {
        **fields,
        name: compact + steps,
        raw_name: _distributed_place_owner_history(counters, raw_values),
    }
    closure = {
        **restart.closure,
        "accepted_data": {**restart.closure["accepted_data"], "fields": advanced},
    }
    validate_meshing_field_declarations(closure)
    record["rng"] = {
        "status": "restored-draws-reproduced-and-advanced",
        "stream": asdict(stream),
        "root_key_data": np.asarray(key_data).tolist(),
        "repack": repack,
        "restored": restored,
        "agrees_with_uninterrupted": True,
        "advanced_events": steps,
    }
    return replace(restart, closure=closure, fields=advanced)


def _distributed_euler_agreement(
    reference: Mapping[str, Any], continued: Mapping[str, Any], /
) -> dict[str, object]:
    """Compare actual uninterrupted and cold advances at the same accepted time."""
    schemas = (("inventory_before", (4,)), ("inventory_after", (4,)), ("state_moments", (2, 4)))
    arrays = tuple(
        (np.asarray(reference[name]), np.asarray(continued[name]), shape)
        for name, shape in schemas
    )
    _distributed_collective_require(
        all(left.shape == right.shape == shape for left, right, shape in arrays),
        "_distributed_euler_agreement",
        "Continued Euler diagnostics must cover all four conserved components.",
        {"reference": reference, "continued": continued},
    )
    defect = max(float(np.max(np.abs(left - right))) for left, right, _ in arrays)
    identities = (
        "system_id", "model_id", "state_entry", "component_names",
        "time_before", "time_after", "steps_before", "steps_after",
    )
    _distributed_collective_require(
        all(reference[name] == continued[name] for name in identities)
        and bool(np.isfinite(defect))
        and defect <= euler_history.INVENTORY_TOLERANCE,
        "_distributed_euler_agreement",
        "Cold Euler continuation must agree with the uninterrupted accepted-state advance.",
        {"maximum_moment_difference": defect, "reference": reference, "continued": continued},
    )
    return {
        "status": "uninterrupted-and-restored-euler-advances-agree",
        "maximum_moment_difference": defect,
        "inventory_tolerance": euler_history.INVENTORY_TOLERANCE,
        "comparison": "owned-volume first and second moments of all four components",
        "time_before": continued["time_before"],
        "time_after": continued["time_after"],
        "steps_before": continued["steps_before"],
        "steps_after": continued["steps_after"],
    }


def _distributed_qualification_consume(
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    mesh_epoch: int,
) -> float:
    """Restore, repartition, republish and continue physics on this changed owner."""
    restored = _distributed_qualification_restore(args, record, started)
    repacked = _distributed_qualification_repack(args, restored, record, started)
    restart = _distributed_rng_restart(args, repacked, record, started)
    declaration = restart.closure["field_declarations"][_DISTRIBUTED_PART]
    clock = restart.closure["accepted_data"]["euler_clock"]
    time, steps = _distributed_euler_clock(restart.target, restart.fields, clock)
    physics = _DistributedPhysicsState(
        restart.target,
        restart.fields,
        declaration.finite_element_fields,
        declaration.precision_policy,
        declaration.numeric_version,
        tuple(type(restart.receipt).execution_parts(restart.receipt).mesh.devices.flat),
        time,
        steps,
        clock["model"],
    )
    continued = _distributed_continue_fe(args, physics, record, started)
    inventory = _distributed_check_restored_inventories(physics, record)
    euler = _distributed_continue_fv(args, physics, inventory, record, started)
    publication = json.loads(
        (args.distributed_repository_root() / _PRODUCER_PUBLICATION).read_text(
            encoding="utf-8"
        )
    )
    record["euler_continuation"] = _distributed_euler_agreement(
        publication["euler_uninterrupted"], record["continued_fv"]
    )
    continued_state = _distributed_stage_ledger(record, deadline=Deadline(started + args.timeout)).run(
        "continued_all_field_acceptance",
        lambda: _distributed_accept_continued_fields(
            restart, continued, euler, mesh_epoch
        ),
    )
    _distributed_rearchive(args, continued_state, record, started, mesh_epoch)
    return continued.solve_error


def _distributed_identical(records: list[dict[str, Any]], path: tuple[str, ...], /) -> Any:
    """Return the one value every process record reports for a collective result."""
    values = []
    for item in records:
        value: Any = item
        for key in path:
            value = value[key]
        values.append(value)
    if any(
        canonical_fingerprint(value) != canonical_fingerprint(values[0])
        for value in values
    ):
        raise qualification_failure(
            "_distributed_identical",
            f"Process records disagree on the collective result {'/'.join(path)}.",
            {"values": values},
        )
    return values[0]


def _distributed_ordered(
    records: list[dict[str, Any]], role: str, count: int, /
) -> list[dict[str, Any]]:
    """Order one launched process set by rank and require its actual topology."""
    ordered = sorted(records, key=lambda item: item["actual_topology"]["process_index"])
    if [item["actual_topology"]["process_index"] for item in ordered] != list(
        range(count)
    ) or any(
        item["actual_topology"]["role"] != role
        or item["actual_topology"]["process_count"] != count
        or item["actual_topology"]["simulated_topology"]
        for item in ordered
    ):
        raise qualification_failure(
            "_distributed_ordered",
            f"The {role} records must cover one real {count}-process set once.",
            {"records": [item["actual_topology"] for item in ordered]},
        )
    return ordered


def _distributed_verify_producers(
    args: QualificationControls, producers: list[dict[str, Any]], /
) -> dict[str, Any]:
    """Require one collective producer epoch, refusal and checkpoint publication."""
    ordered = _distributed_ordered(producers, "producer", args.distributed_processes)
    for path in (
        ("request_id",),
        ("generation", "result_id"),
        ("initial_fe", "global_space_id"),
        ("initial_fe", "solve_max_error"),
        ("initial_fe", "collectively_accepted"),
        ("initial_fv", "changed_cells"),
        ("initial_fv", "inventory_drift"),
        ("initial_fields", "integrals"),
        ("uninterrupted_fv", "inventory_before"),
        ("uninterrupted_fv", "inventory_after"),
        ("uninterrupted_fv", "state_moments"),
        ("uninterrupted_fv", "time_after"),
        ("uninterrupted_fv", "steps_after"),
        ("euler_graph",),
        ("checkpoint", "checkpoint_id"),
        ("checkpoint", "execution_id"),
        ("checkpoint", "analysis_plan_id"),
        ("checkpoint", "numeric_revision_id"),
        ("checkpoint", "manifest_id"),
        ("migration", "receipt_id"),
        ("collective_refusal", "status"),
        ("collective_refusal", "local_refusal", "category"),
        ("rng",),
    ):
        _distributed_identical(ordered, path)
    if any(
        not item["collective_refusal"]["retained_matches_last_accepted"]
        or item["checkpoint"]["process_index"] != item["actual_topology"]["process_index"]
        for item in ordered
    ):
        raise qualification_failure(
            "_distributed_verify_producers",
            "A producer lost its retained epoch or published another owner's checkpoint.",
            {"producers": [item["checkpoint"] for item in ordered]},
        )
    epochs = ordered[0]["rng"]["epochs"]
    coarsened = [item["lineage"] for item in epochs if item["operation"] == "coarsen"]
    if not coarsened or any(
        item["resumed_parents"] == 0
        or item["forged_reset_owners"] != item["forged_reset_refusals"]
        for item in coarsened
    ):
        raise qualification_failure(
            "_distributed_verify_producers",
            "Every coarsening epoch must resume consumed parent streams and refuse their reset.",
            {"coarsened": coarsened},
        )
    return {
        "checkpoints": [item["checkpoint"] for item in ordered],
        "rng_uninterrupted": ordered[0]["rng"]["uninterrupted"],
        "euler_uninterrupted": ordered[0]["uninterrupted_fv"],
    }


def _distributed_verify_consumers(
    args: QualificationControls,
    record: dict[str, Any],
    producers: list[dict[str, Any]],
    consumers: list[dict[str, Any]],
) -> float:
    """Require one collective changed-placement result and bind every process record."""
    producers = _distributed_ordered(producers, "producer", args.distributed_processes)
    ordered = _distributed_ordered(consumers, "consumer", args.restart_parts)
    for path in (
        ("request_id",),
        ("checkpoint", "manifest_id"),
        ("checkpoint", "state_ids"),
        ("checkpoint", "binding_refusal", "exception"),
        ("restart", "receipt_id"),
        ("restart", "cell_migration_counts"),
        ("restart", "reference_errors"),
        ("restart", "global_cells"),
        ("restart", "collective_refusal", "commit_refusal", "exception"),
        ("stable_ids",),
        ("rearchive", "manifest_id"),
        ("rearchive", "reread_state_ids"),
        ("continued_fe", "global_space_id"),
        ("continued_fe", "global_dofs"),
        ("continued_fe", "free_dofs"),
        ("continued_fe", "solve_max_error"),
        ("restored_fields", "integrals"),
        ("continued_fv", "inventory_drift"),
        ("continued_fv", "changed_cells"),
        ("continued_fv", "inventory_before"),
        ("continued_fv", "inventory_after"),
        ("continued_fv", "state_moments"),
        ("continued_fv", "time_after"),
        ("continued_fv", "steps_after"),
        ("euler_restart",),
        ("euler_continuation",),
        ("rng",),
    ):
        _distributed_identical(ordered, path)
    sharded = [item["checkpoint"]["placement"]["owner-sharded"] for item in ordered]
    if (
        sum(item["restart"]["owned_cells"] for item in ordered)
        != ordered[0]["restart"]["global_cells"]
        or any(
            item["restart"]["partition_index"] != item["actual_topology"]["process_index"]
            or item["rearchive"]["reread_state_ids"][str(item["rearchive"]["process_index"])]
            != item["rearchive"]["state_id"]
            or item["checkpoint"]["nonaddressable_arrays"] == 0
            for item in ordered
        )
        or any(
            placement["global_bytes"] != sharded[0]["global_bytes"]
            or placement["measured_addressable_bytes"]
            != placement["declared_addressable_bytes"]
            or placement["measured_addressable_bytes"] >= placement["global_bytes"]
            for placement in sharded
        )
        or sum(placement["measured_addressable_bytes"] for placement in sharded)
        != sharded[0]["global_bytes"]
    ):
        raise qualification_failure(
            "_distributed_verify_consumers",
            (
                "Changed owners must partition cells and shared banks exactly once, "
                "republish their own states, and hold nonaddressable arrays."
            ),
            {
                "consumers": [
                    {
                        "checkpoint": item["checkpoint"],
                        "rearchive": item["rearchive"],
                        "restart": item["restart"],
                    }
                    for item in ordered
                ]
            },
        )
    first = producers[0]
    record["source"] = first["source"]
    record["generation"] = first["generation"]
    record["initial_fe"] = [item["initial_fe"] for item in producers]
    record["initial_fv"] = [item["initial_fv"] for item in producers]
    record["initial_fields"] = first["initial_fields"]
    record["euler_transfers"] = [item["euler_transfers"] for item in producers]
    record["euler_graph"] = first["euler_graph"]
    record["euler_restart"] = ordered[0]["euler_restart"]
    record["euler_continuation"] = ordered[0]["euler_continuation"]
    record["field_declarations"] = first["field_declarations"]
    record["epochs"] = first["epochs"]
    record["collective_refusal"] = {
        "producer_commit": [item["collective_refusal"] for item in producers],
        "consumer_binding": [item["checkpoint"]["binding_refusal"] for item in ordered],
        "consumer_commit": [item["restart"]["collective_refusal"] for item in ordered],
    }
    record["migration"] = {
        "status": "accepted",
        "producer_graph_repartition": [item["migration"] for item in producers],
        "changed_owner_count": {
            "route": ordered[0]["restart"]["placement_route"],
            "cell_migration_counts": ordered[0]["restart"]["cell_migration_counts"],
            "received_cells": ordered[0]["restart"]["received_cells"],
            "received_cell_imbalance": ordered[0]["restart"]["received_cell_imbalance"],
        },
    }
    record["checkpoint"] = {
        "status": "published-restored-republished",
        "producer_publications": [item["checkpoint"] for item in producers],
        "consumer_restores": [item["checkpoint"] for item in ordered],
        "consumer_republications": [item["rearchive"] for item in ordered],
    }
    record["restart"] = {
        "status": "changed-owner-count-restart-accepted",
        "source_parts": args.distributed_processes,
        "destination_parts": args.restart_parts,
        "owners": [item["restart"] for item in ordered],
    }
    record["stable_ids"] = ordered[0]["stable_ids"]
    record["restored_fields"] = ordered[0]["restored_fields"]
    record["conservation"] = ordered[0]["conservation"]
    if not _distributed_rng_agree(
        producers[0]["rng"]["uninterrupted"], ordered[0]["rng"]["restored"]
    ):
        raise qualification_failure(
            "_distributed_verify_consumers",
            "Restored probe draws differ from the uninterrupted producers' future draws.",
            {"consumer": ordered[0]["rng"], "producer": producers[0]["rng"]},
        )
    record["rng"] = {
        "status": "uninterrupted-and-restored-future-draws-agree",
        "producer": producers[0]["rng"],
        "consumer": ordered[0]["rng"],
    }
    record["continued_fe"] = [item["continued_fe"] for item in ordered]
    record["continued_fv"] = [item["continued_fv"] for item in ordered]
    record["uninterrupted_fv"] = [item["uninterrupted_fv"] for item in producers]
    record["communication"] = {
        "status": "measured-owner-message-counts",
        "producer": [item["communication"] for item in producers],
        "consumer": [item["communication"] for item in ordered],
        "payload_byte_status": "unmeasured",
    }
    return max(item["continued_fe"]["solve_max_error"] for item in ordered)


def _distributed_qualification_orchestrate(
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> float:
    """Launch independent producer and changed-placement consumer process sets."""
    root = args.distributed_repository_root()
    # Fresh repository root: concurrent writers must not open an unprovisioned or reused directory.
    (root / "repository").mkdir(parents=True, exist_ok=False)
    producers = _distributed_qualification_launch(args, record, started, "producer")
    publication = _distributed_verify_producers(args, producers)
    (root / _PRODUCER_PUBLICATION).write_text(
        json.dumps(publication, indent=2, sort_keys=True), encoding="utf-8"
    )
    consumers = _distributed_qualification_launch(args, record, started, "consumer")
    return _distributed_verify_consumers(args, record, producers, consumers)


def _distributed_qualification_lifecycle(
    args: QualificationControls,
) -> dict[str, object]:
    """Publish from producer processes, then restart on a changed process set.

    A single-process invocation orchestrates two independently launched JAX
    process sets: ``distributed_processes`` producers generate, adapt, refuse a
    corrupt epoch, repartition and publish; ``restart_parts`` consumers restore
    owner-local shards, repack, republish and continue FE and FV physics.
    """

    started = perf_counter()
    phases: list[NativeMeshingPhaseMeasurement] = []
    requirements = _distributed_qualification_requirements()
    frozen = _distributed_qualification_request(args)
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
        "rng": {"status": "not-executed"},
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
            case "consumer":
                record["physical_error"] = _distributed_qualification_consume(
                    args, record, started, mesh_epoch
                )
            case "orchestrator":
                physical_error = _distributed_qualification_orchestrate(
                    args, record, started
                )
                if physical_error > args.target_error:
                    raise qualification_failure(
                        "_distributed_qualification_lifecycle",
                        (
                            "Continued distributed physics missed the frozen physical-error target."
                        ),
                        {"args": args, "record": record},
                    )
                record["physical_error"] = physical_error
                record["time_to_target_error_seconds"] = perf_counter() - started
                record["independent_scientific_certification"] = "passed"
                # A one-owner restart runs the same owner-local implementation
                # but is not a changed multiple-process placement.
                record["mandatory_workflow_completion"] = (
                    "complete"
                    if args.restart_parts >= 2
                    else "routine-single-owner-restart-nonclaim"
                )
            case role:
                raise ValueError(
                    f"_distributed_qualification_lifecycle.role must be 'orchestrator', 'producer' or 'consumer'; got {role!r}."
                )
        record["status"] = "passed"
    except Exception as error:
        detail = record_failure(record, error, record["active_phase"])
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
        record["peak_process_resident_bytes"] = peak_resident_bytes()
        record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
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
