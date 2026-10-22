# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Scientific source and raw-state array bindings for distributed qualification."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, final, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec, Sharding, SingleDeviceSharding

from phydrax import ArrayArchiveLimits, BranchDifferentiationPolicy, StrictModule
from phydrax._frozendict import frozendict
from phydrax._model._structure import (
    model_recipe_array_inventory,
    prepare_model_recipe_catalog,
)
from phydrax.discretization import (
    AdaptiveSimplexParts,
    FieldTracePolicy,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPrecisionPolicy,
    lagrange_element,
)
from phydrax.execution import ExecutionGroup
from phydrax.lifecycle import (
    CheckpointManifest,
    MeshingArrayRole,
    MeshingCheckpointState,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
)
from phydrax.lifecycle._distributed_checkpoint import (
    _meshing_checkpoint_record_paths,
    _validate_meshing_owner_states,
    meshing_checkpoint_array_layouts,
)
from phydrax.lifecycle._meshing_checkpoint_binding import source_logical_array_name
from phydrax.lifecycle._meshing_sources import (
    meshing_source_local_restore_phase,
    meshing_source_shared_array_aliases,
    MeshingSourceClosure,
    register_meshing_source_artifacts,
)
from phydrax.meshing import (
    CellMeshingResult,
    CollectiveMeshEvidence,
    MeshAdaptationResult,
    NativeMeshingOptions,
    NativePlanarSource,
    SurfaceMeshingSpec,
)
from phydrax.meshing._device_adaptation import restore_partitioned_mesh_state
from phydrax.meshing._distributed_planar_generation import (
    PreparedDistributedPlanarGeneration,
)
from phydrax.meshing._distribution_forest import forest_vertex_record_owners
from phydrax.typing import checked
from phydrax.units import KELVIN


def distributed_field_templates() -> tuple[tuple[str, FiniteElementFieldSpec], ...]:
    """Return the physical temperature declaration for material-history binding."""
    return (
        (
            "material-history",
            FiniteElementFieldSpec("temperature", lagrange_element("triangle", 1)),
        ),
    )


@final
class DistributedCheckpointSourceArrays(StrictModule):
    """Named source arrays and missing scientific roles at the archive boundary."""

    bank: dict[str, jax.Array]
    bindings: dict[str, str]
    missing: list[str]
    missing_details: tuple[tuple[str, str, tuple[int, ...], str], ...]


@final
class DistributedCheckpointMeshArrays(StrictModule):
    """Logical mesh and raw allocator arrays with explicit checkpoint roles."""

    arrays: dict[str, jax.Array]
    array_roles: dict[str, MeshingArrayRole]
    evidence: CollectiveMeshEvidence


@checked
def prepare_distributed_checkpoint_source_closure(
    author: NativePlanarSource,
    request: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    generation: PreparedDistributedPlanarGeneration,
    results: tuple[MeshAdaptationResult, ...],
    physical: FiniteElementDiscretization,
    roles: tuple[MeshingFieldStateRole, ...],
    fields: Mapping[str, jax.Array | np.ndarray],
    fe_fields: tuple[FiniteElementFieldSpec, ...],
    precision: FiniteElementPrecisionPolicy,
    *,
    part_name: str,
    mesh_epoch: int,
) -> dict[str, Any]:
    """Build and register the accepted scientific source closure for archival.

    The owner-local planar root retains its authored generation inputs (source,
    request, tile schedule and layout); its collective theorem is the target's
    original source, never a fabricated serial certification report.
    """
    generation.require_binding(author, request)
    target = results[0].target
    space = physical.field_spaces[0]
    declaration = MeshingFiniteElementFieldDeclaration(
        part_name=part_name,
        topology_id=target.mesh.topology_id,
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_space_ids=frozendict({"temperature": space.field_space_id}),
        value_units=frozendict({"temperature": (KELVIN,)}),
        maximum_derivative_orders=frozendict({"temperature": 1}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version=f"accepted-{mesh_epoch}",
        finite_element_fields=fe_fields,
        precision_policy=precision,
    )
    closure = {
        "generation_source": author,
        "generation_specification": request,
        "generation_schedule": generation.schedule,
        "generation_layout": generation.layout,
        "generation_options": options,
        "collective_target": target,
        "field_declarations": {part_name: declaration},
        "accepted_data": {"adaptation_results": results, "fields": fields},
    }
    register_meshing_source_artifacts()
    return closure


@checked
def bind_distributed_checkpoint_source_arrays(
    closure: Mapping[str, Any], limits: ArrayArchiveLimits
) -> DistributedCheckpointSourceArrays:
    """Bind model recipe arrays to the source closure's scientific alias bank."""
    _, catalog, values = prepare_model_recipe_catalog(
        closure, prefix="source-record", limits=limits
    )
    bank, aliases = meshing_source_shared_array_aliases(
        MeshingSourceClosure.from_records(closure)
    )
    bindings = {
        name: aliases[id(value)] for name, value in values.items() if id(value) in aliases
    }
    missing = [
        name
        for name, value in values.items()
        if isinstance(value, jax.Array)
        and not value.is_fully_addressable
        and name not in bindings
    ]
    return DistributedCheckpointSourceArrays(
        bank=bank,
        bindings=bindings,
        missing=missing,
        missing_details=tuple(
            (entry.name, entry.path, entry.shape, entry.dtype)
            for entry in catalog
            if entry.name in missing
        ),
    )


@checked
def prepare_distributed_checkpoint_mesh_arrays(
    target: CellMeshingResult,
    fields: Mapping[str, jax.Array | np.ndarray],
    parts: AdaptiveSimplexParts,
    *,
    part_count: int,
    mesh_epoch: int,
) -> DistributedCheckpointMeshArrays:
    """Place logical mesh and allocated history arrays on their scientific owners."""
    storage, evidence = target.mesh.storage, target.collective_evidence
    if storage is None or not isinstance(evidence, CollectiveMeshEvidence):
        raise TypeError(
            "qualification.distributed.target must provide mesh storage and "
            "CollectiveMeshEvidence; "
            f"got storage={type(storage).__name__}, evidence={type(evidence).__name__}"
        )
    _, raw_state, _ = restore_partitioned_mesh_state(storage, evidence)
    cell_ids = raw_state.mesh.cell_ids
    ranks = jnp.arange(part_count, dtype=jnp.int32)[:, None]
    vertex_owners = forest_vertex_record_owners(parts, raw_state.mesh.vertex_ids)
    # Raw allocator leaves share forest placement; logical leaves retain storage ownership.
    allocated = {
        "lineage": raw_state.parents,
        "epoch": jnp.full(cell_ids.shape, mesh_epoch, dtype=jnp.int64),
        "evidence": evidence.partition_checks,
        "cell_owners": jnp.where(cell_ids >= 0, ranks, -1),
        "vertex_owners": vertex_owners,
        "allocated-temperature": fields["allocated-temperature"],
        "allocated-material-temperature": fields["allocated-material-temperature"],
    }
    logical = dict(storage.logical_arrays)
    arrays = {
        "vertex_ids": logical["vertex_global_ids"],
        "coordinates": logical["coordinates"],
        "topology": logical["cell_vertices"],
        **{
            name: jax.device_put(value, cell_ids.sharding)
            for name, value in allocated.items()
        },
    }
    array_roles: dict[str, MeshingArrayRole] = {
        "vertex_ids": "entity-ids",
        "coordinates": "geometry",
        "topology": "topology",
        "lineage": "lineage",
        "epoch": "epoch",
        "evidence": "evidence",
        "cell_owners": "state",
        "vertex_owners": "state",
        "allocated-temperature": "state",
        "allocated-material-temperature": "state",
    }
    return DistributedCheckpointMeshArrays(
        arrays=arrays, array_roles=array_roles, evidence=evidence
    )


@checked
def distributed_checkpoint_host_array_names(
    owner: MeshingCheckpointState, recipe: str, limits: ArrayArchiveLimits
) -> set[str]:
    """Classify restored host-only source arrays by their registered recipe backends."""
    return {
        owner.source_array_name(entry.name)
        for entry in model_recipe_array_inventory(
            json.loads(recipe), prefix="source-record", limits=limits
        )
        if entry.backend == "numpy"
    }


type DistributedRestorePlacementKind = Literal[
    "owner-sharded", "replicated", "process-local-owner-record"
]


@final
@dataclass(frozen=True, slots=True)
class DistributedRestorePlacement:
    """Restart destinations and the logical bytes each one places on this process.

    Shared logical banks and scientific arrays are split by their leading axis
    across the restart group when it divides evenly. ``source_owners`` retains
    every archived owner row currently addressable on this process. Every assigned
    typed source record is reconstructed so restart retains complete associations.
    """

    shardings: dict[str, Sharding]
    placements: dict[str, DistributedRestorePlacementKind]
    global_bytes: dict[str, int]
    addressable_bytes: dict[str, int]
    decoded_owners: tuple[int, ...]
    source_owners: tuple[int, ...]


def _distributed_restore_owner_layouts(
    states: Sequence[MeshingCheckpointState],
    manifest: CheckpointManifest,
    group: ExecutionGroup,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[dict[str, tuple[tuple[int, ...], np.dtype[Any]]], tuple[int, ...]]:
    """Admit archived owner coverage against actual current device-slice maps."""
    with meshing_source_local_restore_phase():
        owner_count = len(_meshing_checkpoint_record_paths(manifest))
        if len(states) != owner_count or tuple(
            state.source_owner_index for state in states
        ) != tuple(range(owner_count)):
            raise ValueError(
                "qualification.distributed.states must contain each archived source "
                "owner exactly once in canonical order."
            )
        _validate_meshing_owner_states(
            states,
            {owner: state.checkpoint_state_id for owner, state in enumerate(states)},
            0,
        )
        aliases = states[0].source_logical_array_names
        projection_names = {
            source_logical_array_name(name)
            for name in aliases if "/geometry_projection/" in name
        }
        if not projection_names:
            raise ValueError(
                "qualification.distributed.projection must retain its authenticated "
                "collective geometry storage banks."
            )
        layouts = meshing_checkpoint_array_layouts(
            manifest, tuple(source_logical_array_name(name) for name in aliases),
            limits=limits,
        )
        (axis,) = group.mesh.axis_names
        sharding = group.named_sharding(PartitionSpec(axis))
        source_owners: tuple[int, ...] | None = None
        for name in sorted(projection_names):
            shape, _ = layouts[name]
            if not shape or shape[0] != owner_count:
                raise ValueError(
                    "qualification.distributed.projection must retain the complete "
                    "archived source-owner axis; got foreign storage layout."
                )
            rows: list[int] = []
            for index in sharding.addressable_devices_indices_map(shape).values():
                selection = index[0]
                if not isinstance(selection, slice) or selection.step not in (None, 1):
                    raise ValueError(
                        "qualification.distributed.projection must expose contiguous "
                        "current source-owner slices."
                    )
                rows.extend(range(*selection.indices(owner_count)))
            current = tuple(sorted(rows))
            if (
                not current or len(set(current)) != len(current)
                or (source_owners is not None and current != source_owners)
            ):
                raise ValueError(
                    "qualification.distributed.projection must bind the same unique "
                    "process-addressable source owners in every geometry bank."
                )
            source_owners = current
        if source_owners is None:
            raise ValueError(
                "qualification.distributed.projection must retain current owner storage."
            )
        local_coverage = np.zeros(owner_count, dtype=np.int32)
        local_coverage[np.asarray(source_owners, dtype=np.int64)] = 1
    coverage = np.asarray(
        multihost_utils.process_allgather(local_coverage, tiled=False)
    ).reshape((-1, owner_count))
    with meshing_source_local_restore_phase():
        if not np.all(np.sum(coverage, axis=0) == 1):
            raise ValueError(
                "qualification.distributed.projection must cover every archived "
                "source owner exactly once across the current process set."
            )
    return layouts, source_owners


@checked
def prepare_distributed_restore_placement(
    states: Sequence[MeshingCheckpointState],
    manifest: CheckpointManifest,
    group: ExecutionGroup,
    limits: ArrayArchiveLimits,
    /,
) -> DistributedRestorePlacement:
    """Reconstruct historical-owner placement before reading scientific payloads."""
    layouts, source_owners = _distributed_restore_owner_layouts(
        states, manifest, group, limits
    )
    with meshing_source_local_restore_phase():
        # Each assigned archived closure contributes its authenticated scientific
        # rows; a representative root alone cannot reconstruct changed ownership.
        decoded = source_owners
        bank = set(layouts)
        names = {
            name for owner in decoded for name, _ in states[owner].array_roles
        } | bank
        host: set[str] = set()
        records: set[str] = set()
        for owner in decoded:
            state = states[owner]
            recipe = state.source_recipe_json
            if recipe is None:
                raise ValueError(
                    "qualification.distributed.states must retain every decoded owner's source recipe; "
                    f"got none for owner {owner}"
                )
            host |= distributed_checkpoint_host_array_names(state, recipe, limits)
            records |= {
                state.source_array_name(entry.name)
                for entry in model_recipe_array_inventory(
                    json.loads(recipe), prefix="source-record", limits=limits
                )
            } - bank
        layouts.update(
            meshing_checkpoint_array_layouts(
                manifest, tuple(sorted(names - host - bank)), limits=limits
            )
        )
        (axis,) = group.mesh.axis_names
        count = int(group.mesh.shape[axis])
        local_devices = sum(
            device.process_index == jax.process_index() for device in group.devices
        )
        process_local = SingleDeviceSharding(jax.local_devices()[0])
        shardings: dict[str, Sharding] = {}
        placements: dict[str, DistributedRestorePlacementKind] = {}
        global_bytes: dict[str, int] = {}
        addressable_bytes: dict[str, int] = {}
        for name, (shape, dtype) in layouts.items():
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            global_bytes[name] = size
            if name in records:
                shardings[name] = process_local
                placements[name] = "process-local-owner-record"
                addressable_bytes[name] = size
            elif shape and shape[0] % count == 0:
                shardings[name] = group.named_sharding(PartitionSpec(axis))
                placements[name] = "owner-sharded"
                addressable_bytes[name] = size // count * local_devices
            else:
                shardings[name] = group.named_sharding(PartitionSpec())
                placements[name] = "replicated"
                addressable_bytes[name] = size
        return DistributedRestorePlacement(
            shardings, placements, global_bytes, addressable_bytes, decoded, source_owners
        )


@checked
def rebind_distributed_checkpoint_field_declaration(
    declaration: MeshingFiniteElementFieldDeclaration,
    target: CellMeshingResult,
    physical: FiniteElementDiscretization,
    roles: tuple[MeshingFieldStateRole, ...],
) -> MeshingFiniteElementFieldDeclaration:
    """Bind a restored field declaration to the accepted target and recovered roles."""
    return replace(
        declaration,
        topology_id=target.mesh.topology_id,
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_space_ids=frozendict(
            {"temperature": physical.field_spaces[0].field_space_id}
        ),
        state_roles=roles,
    )
