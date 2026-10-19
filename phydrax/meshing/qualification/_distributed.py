# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Scientific source and raw-state array bindings for distributed qualification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, final

import jax
import jax.numpy as jnp
import numpy as np

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
from phydrax.lifecycle import (
    MeshingArrayRole,
    MeshingCheckpointState,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
)
from phydrax.lifecycle._meshing_sources import (
    meshing_source_shared_array_aliases,
    MeshingSourceClosure,
    register_meshing_source_artifacts,
)
from phydrax.meshing import (
    CellMeshingResult,
    CollectiveMeshEvidence,
    MeshAdaptationResult,
    MeshPart,
    NativeMeshingOptions,
    NativePlanarSource,
    SurfaceMeshingSpec,
)
from phydrax.meshing._certification import MeshCertificationReport
from phydrax.meshing._device_adaptation import restore_partitioned_mesh_state
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
    source: CellMeshingResult,
    results: tuple[MeshAdaptationResult, ...],
    physical: FiniteElementDiscretization,
    roles: tuple[MeshingFieldStateRole, ...],
    fields: Mapping[str, jax.Array | np.ndarray],
    fe_fields: tuple[FiniteElementFieldSpec, ...],
    precision: FiniteElementPrecisionPolicy,
    certification: MeshCertificationReport,
    *,
    part_name: str,
    mesh_epoch: int,
) -> dict[str, Any]:
    """Build and register the accepted scientific source closure for archival."""
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
        "certification_inputs": certification.request,
        "report": certification,
        "associations": source.associations,
        "generation_part": MeshPart(part_name, source),
        "generation_source": author,
        "generation_specification": request,
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
    _, _, values = prepare_model_recipe_catalog(
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
        bank=bank, bindings=bindings, missing=missing
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
