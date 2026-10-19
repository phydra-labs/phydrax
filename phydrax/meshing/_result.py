#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Publish mesh results with bound scientific evidence and sharded storage identities."""

from __future__ import annotations

import abc
from dataclasses import fields
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax
import phydrax.discretization._adaptive_simplex as _dependency_phydrax_discretization__adaptive_simplex
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._embedding_state as _owner_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._surface_source_support as _dependency_phydrax_geometry__surface_source_support
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._audit as _dependency_phydrax_meshing__audit
import phydrax.meshing._certification as _dependency_phydrax_meshing__certification
import phydrax.meshing._certification as _owner_phydrax_meshing__certification
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization_records
import phydrax.meshing._quality as _dependency_phydrax_meshing__quality
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace

from .._fingerprint import (
    canonical_fingerprint,
    logical_array_value_collection_digest,
    tree_content_equal,
)
from .._identity import SemanticProvenance
from .._meshcore import NativeExecutionScope
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..geometry.surface import SurfaceModel
from ..interchange import AdapterReport
from ..typing import Bool, checked, Dim, Size
from ._measurements import NativeExecutionRecord


if TYPE_CHECKING:
    import phydrax.meshing._bisection_kernel_host
    import phydrax.meshing._device_adaptation
    import phydrax.meshing._initial_certification

    from ._adaptation import MeshAdaptationResult, PreparedMeshAdaptation
    from ._bisection import BisectionUniformRefinement
    from ._distribution import PartitionedRebalanceProposal
    from ._distribution_migration import PartitionedEpochMigration


_BISECTION_CERTIFICATE_CHECKS = (
    "source_roots",
    "subdivision",
    "midpoint_geometry",
    "owned_cell_identity",
    "reciprocal_facets",
    "shared_geometry",
    "collective_status",
)


import phydrax._validation as _owner_phydrax__validation


@final
class _CollectiveSourcePreparation(StrictModule):
    cold: bool
    preparation: PreparedMeshAdaptation
    initial: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    raw_blocks: tuple[CellBlock, ...]
    saved_publication_blocks: tuple[CellBlock, ...] | None
    uniform_refinement: BisectionUniformRefinement | None
    exterior: Array
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout
    neighbor_pairs: tuple[tuple[int, int], ...]
    axis_name: str
    placement_arrays: tuple[tuple[str, Array], ...]
    source: CellMeshingResult
    witness: phydrax.meshing._distribution.AffineBisectionSourceWitness | None
    source_entities: tuple[
        tuple[np.ndarray | Array, np.ndarray | Array, np.ndarray | Array], ...
    ]
    initial_owners: Array
    migration: PartitionedEpochMigration | None
    rebalance_proposal: PartitionedRebalanceProposal | None


@final
class _CollectiveScientificSource(StrictModule):
    original: (
        CellMeshingResult
        | phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence
    )
    scientific_id: str
    premise_id: str
    source_embedding: global_embedding_owner.GlobalEmbeddingCertificate | None
    source_audit_id: str
    scientific_binding: dict[str, str | None | list[str]]
    witness: phydrax.meshing._distribution.AffineBisectionSourceWitness


@final
class _CollectiveSubdivision(StrictModule):
    source_cache_id: str
    initial_state_id: str
    native_arrays: tuple[tuple[str, Array], ...]
    native_receipt_id: str
    partition_checks: Array
    verdicts: Array
    summary: np.ndarray
    owners: Array


@final
class _CollectivePackedEpoch(StrictModule):
    compiled_states: (
        _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    )
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    publication_exterior: Array
    publication_source_blocks: tuple[CellBlock, ...]
    owners: Array


@final
class _CollectivePublication(StrictModule):
    arrays: tuple[tuple[str, Array], ...]
    entity_keys: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    entity_owners: tuple[Array, ...]
    counts: tuple[int, ...]
    presentation_names: tuple[str, ...] | None
    topology_id: str
    geometry_id: str
    organization_arrays: tuple[tuple[str, Array], ...]
    geometry_arrays: tuple[tuple[str, Array], ...]
    coordinate_geometry_id: str
    global_organization_id: str
    source_evidence_id: str
    target_mesh_id: str


def _prepare_collective_source(
    partitioned: phydrax.meshing.PartitionedAdaptiveSimplex | CollectiveMeshEvidence,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CollectiveSourcePreparation:
    # Lazy: _result -> _bisection_kernel_host -> _lineage import cycle.
    import phydrax.meshing._bisection_kernel_host

    # Lazy: _result -> collective/distribution owners -> _bisection -> _lineage import cycle.
    import phydrax.meshing._collective_geometry
    import phydrax.meshing._collective_organization
    import phydrax.meshing._device_adaptation
    import phydrax.meshing._distribution
    import phydrax.meshing._distribution_forest
    import phydrax.meshing._distribution_migration

    # Lazy: _result -> _initial_certification -> _canonical -> _lineage import cycle.
    import phydrax.meshing._initial_certification

    cold = isinstance(partitioned, CollectiveMeshEvidence)
    saved_publication_blocks: tuple[CellBlock, ...] | None = None
    match partitioned:
        case CollectiveMeshEvidence():
            partitioned.require_passed()
            preparation = partitioned.preparation
            initial = partitioned.initial_states
            raw_source_blocks = partitioned.raw_source_blocks
            saved_publication_blocks = partitioned.publication_source_blocks
            if (
                not isinstance(saved_publication_blocks, tuple)
                or not saved_publication_blocks
            ):
                raise ValueError(
                    "collective_source.publication_source_blocks must retain nonempty source block declarations; got absent declarations"
                )
            if any(
                not isinstance(block, CellBlock) for block in saved_publication_blocks
            ):
                raise ValueError(
                    "collective_source.publication_source_blocks must contain CellBlock declarations; got an invalid block kind"
                )
            uniform_refinement = partitioned.uniform_refinement
            exterior = partitioned.source_exterior
            layout = partitioned.layout
            neighbor_pairs = partitioned.neighbor_pairs
            axis_name = partitioned.axis_name
            placement_arrays = tuple(
                (name, value)
                for name, value in partitioned.logical_arrays
                if name.startswith(("placement/", "rebalance/"))
            )
            source = preparation.source
            migration = (
                None
                if partitioned.migration_record is None
                else partitioned.migration_record.replay(scope=scope, budget=budget)
            )
            rebalance_proposal = partitioned.rebalance_proposal
            if source.result_id != partitioned.source.result_id:
                raise ValueError(
                    "collective_source.preparation.source must bind the accepted predecessor; got changed result_id"
                )
            witness = phydrax.meshing._distribution.AffineBisectionSourceWitness(
                source,
                uniform_refinement=uniform_refinement,
                scope=scope,
                budget=budget,
            )
            source_entities = phydrax.meshing._device_adaptation.source_entity_tables(
                preparation
            )
            initial_owners = dict(placement_arrays).get(
                "placement/initial_solver_cell_owners"
            )
            if initial_owners is None:
                initial_owners = (
                    phydrax.meshing._device_adaptation.source_solver_cell_owners(
                        preparation, initial
                    )
                )
        case phydrax.meshing._device_adaptation.PartitionedAdaptiveSimplex():
            preparation = partitioned.prepared.adaptation
            initial = partitioned.states
            uniform_refinement = partitioned.prepared.anchor.start.uniform_refinement
            declared_blocks = partitioned.prepared.anchor.source_blocks
            raw_source_blocks = (
                partitioned.prepared.anchor.source.mesh.blocks
                if declared_blocks is None
                else declared_blocks
            )
            exterior = partitioned.source_exterior
            layout = partitioned.layout
            neighbor_pairs = partitioned.parts.neighbor_pairs
            axis_name = partitioned.parts.axis_name
            placement_arrays = partitioned.placement_arrays
            source = preparation.source
            witness = partitioned.source_witness
            source_entities = partitioned.source_entities
            initial_owners = partitioned.solver_cell_owners
            migration = partitioned.migration
            rebalance_proposal = partitioned.rebalance_proposal
        case _:
            raise TypeError(
                "collective_source.partitioned must be a partitioned preparation or retained evidence; got unsupported type"
            )
    if not isinstance(raw_source_blocks, tuple) or not raw_source_blocks:
        raise ValueError(
            "collective_source.raw_source_blocks must retain nonempty source block declarations; got absent declarations"
        )
    if migration is not None:
        if not cold:
            phydrax.meshing._distribution_migration.validate_migrated_epoch(
                migration,
                migration.source_partitioned.prepared,
                states,
                initial,
                exterior,
                migration.slot_origins,
            )
        if rebalance_proposal is None or not bool(
            jax.device_get(
                jnp.array_equal(
                    migration.requested_target_owners,
                    rebalance_proposal.requested_target_owners,
                )
            )
        ):
            raise ValueError(
                "collective_migration.proposal must bind the transported ownership; got mismatched evidence"
            )
        phydrax.meshing._distribution.replay_partitioned_rebalance_proposal(
            rebalance_proposal, axis_name=axis_name
        )
    kind = "triangle" if layout.dimension == 2 else "tetrahedron"
    raw_blocks = tuple(
        block for block in raw_source_blocks if isinstance(block, CellBlock)
    )
    if len(raw_blocks) != len(raw_source_blocks) or any(
        block.cell_kind != kind for block in raw_blocks
    ):
        raise ValueError(
            "collective_source.raw_source_blocks must match the allocated simplex family; got incompatible declarations"
        )

    return _CollectiveSourcePreparation(
        cold=cold,
        preparation=preparation,
        initial=initial,
        raw_blocks=raw_blocks,
        saved_publication_blocks=saved_publication_blocks,
        uniform_refinement=uniform_refinement,
        exterior=exterior,
        layout=layout,
        neighbor_pairs=neighbor_pairs,
        axis_name=axis_name,
        placement_arrays=placement_arrays,
        source=source,
        witness=witness,
        source_entities=source_entities,
        initial_owners=initial_owners,
        migration=migration,
        rebalance_proposal=rebalance_proposal,
    )


def _bind_collective_scientific_source(
    prepared: _CollectiveSourcePreparation,
    partitioned: phydrax.meshing.PartitionedAdaptiveSimplex | CollectiveMeshEvidence,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
) -> _CollectiveScientificSource:
    saved = partitioned if isinstance(partitioned, CollectiveMeshEvidence) else None
    witness = prepared.witness
    original = require_original_meshing_source(
        prepared.source
        if prepared.uniform_refinement is None
        else prepared.uniform_refinement.source,
    )
    scientific_binding: dict[str, str | None | list[str]]
    match original:
        case phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence():
            scientific_id = original.evidence_id
            premise_id = original.evidence_id
            replayed_premise_id = premise_id
            source_embedding = None
            if saved is not None and saved.source_embedding is not None:
                raise ValueError(
                    "collective_source.embedding must belong to a serial source; got an initial collective theorem"
                )
            source_audit_id = original.evidence_id
            scientific_binding = {
                "kind": "independent-native-initial-theorem",
                "theorem": original.evidence_id,
                "source": original.source_evidence_id,
                "compiled": original.compiled.compiled_id,
                "specification": original.specification.specification_id,
            }
        case CellMeshingResult():
            if witness is None or witness.embedding is None:
                raise ValueError(
                    "collective_source.embedding must carry its prepared proof; got absent evidence"
                )
            witness.embedding.binding.require(original.mesh, original.geometry)
            if witness.embedding.status != "certified":
                raise ValueError(
                    "collective_source.embedding must be certified; got "
                    f"{witness.embedding.status}"
                )
            scientific_id = original.result_id
            replayed_premise_id = (
                original.certification.report_id
                if original.certification is not None
                else canonical_fingerprint(
                    {
                        "kind": "audited-meshing-source-embedding",
                        "mesh": original.mesh.mesh_id,
                        "geometry": _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                            original.geometry
                        ),
                        "audit": original.audit.report_id,
                        "embedding": witness.embedding.certificate_id,
                    }
                )
            )
            source_embedding = witness.embedding
            premise_id = replayed_premise_id
            if saved is not None:
                source_embedding, premise_id = (
                    CollectiveMeshEvidence._replay_source_embedding(
                        original, saved.source_embedding, witness.embedding
                    )
                )
            source_audit_id = original.audit.report_id
            scientific_binding = {
                "kind": "audited-meshing-source-embedding",
                "premise": premise_id,
                "embedding": source_embedding.certificate_id,
                "source_regions": None
                if original.region_evidence is None
                else original.region_evidence.evidence_id,
                "region_boundaries": [
                    value.evidence_id for value in original.region_boundary_evidence
                ],
                "associations": [value.association_id for value in original.associations],
            }
        case _:
            raise TypeError(
                "collective_source.original must resolve to initial evidence or a serial result; got unsupported source kind"
            )
    if saved is not None and saved.source_premise_id != premise_id:
        raise ValueError(
            "collective_source.premise_id must bind its retained source proof; got changed identity"
        )
    if witness is None:
        raise ValueError(
            "collective_source.witness must retain the affine source map; got absent evidence"
        )
    if (
        witness.source_result_id != prepared.source.result_id
        or witness.source_geometry_id != prepared.source.mesh.geometry_id
        or witness.source_layout_id != prepared.source.geometry.geometry_layout_id
        or witness.source_coordinate_geometry_id
        != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            prepared.source.geometry
        )
        or witness.scientific_source_result_id != scientific_id
        or witness.scientific_certification_id != replayed_premise_id
        or witness.predecessor_evidence_id
        != (
            None
            if prepared.source.collective_evidence is None
            else prepared.source.collective_evidence.evidence_id
        )
        or states.cursors.shape != prepared.initial.cursors.shape
        or states.cursors.shape != (prepared.initial.mesh.cell_ids.shape[0], 4)
        or prepared.exterior.shape != prepared.initial.mesh.cells.shape
        or prepared.exterior.dtype != jnp.bool_
    ):
        raise ValueError(
            "collective_source.witness must bind source, placement and state axes; got mismatched evidence"
        )
    return _CollectiveScientificSource(
        original=original,
        scientific_id=scientific_id,
        premise_id=premise_id,
        source_embedding=source_embedding,
        source_audit_id=source_audit_id,
        scientific_binding=scientific_binding,
        witness=witness,
    )


def _prove_collective_subdivision(
    prepared: _CollectiveSourcePreparation,
    partitioned: phydrax.meshing.PartitionedAdaptiveSimplex | CollectiveMeshEvidence,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CollectiveSubdivision:
    source_cache_id = phydrax.meshing._device_adaptation.source_cache_digest(
        prepared.source_entities
    )
    initial_state_id = logical_array_value_collection_digest(
        {
            jax.tree_util.keystr(path): value
            for path, value in jax.tree_util.tree_flatten_with_path(prepared.initial)[0]
            if isinstance(value, Array)
        }
    )
    if (
        not prepared.cold
        and isinstance(
            partitioned,
            phydrax.meshing._device_adaptation.PartitionedAdaptiveSimplex,
        )
        and (
            source_cache_id != partitioned.source_cache_id
            or initial_state_id != partitioned.initial_state_id
        )
    ):
        raise ValueError(
            "collective_source.cache must bind the initial numerical state and source entities; got changed digests"
        )
    midpoint_valid, midpoint_signs = (
        phydrax.meshing._device_adaptation.require_partitioned_exact_restrictions(
            prepared.layout, states, prepared.initial, scope=scope, budget=budget
        )
    )
    native_arrays = (
        ("native/midpoint_valid", midpoint_valid),
        ("native/midpoint_signs", midpoint_signs),
    )
    native_receipt_id = logical_array_value_collection_digest(dict(native_arrays))
    match partitioned:
        case CollectiveMeshEvidence():
            partition_checks = (
                phydrax.meshing._distribution.replay_logical_bisection_checks(
                    states,
                    prepared.initial,
                    prepared.exterior,
                    neighbor_pairs=prepared.neighbor_pairs,
                    axis_name=prepared.axis_name,
                )
            )
        case phydrax.meshing._device_adaptation.PartitionedAdaptiveSimplex():
            partition_checks = phydrax.meshing._distribution.compiled_bisection_checks(
                partitioned.parts,
                states,
                prepared.initial,
                prepared.exterior,
            )
        case _:
            raise TypeError(
                "collective_source.partitioned must retain its preparation kind; got changed numerical proof"
            )
    verdicts = jnp.all(partition_checks, axis=0)
    summary = np.asarray(jax.device_get(verdicts), dtype=np.bool_)
    if not np.all(summary):
        raise ValueError(
            "collective_subdivision.checks must pass every global check; got failed subdivision checks"
        )
    match partitioned:
        case CollectiveMeshEvidence():
            owners, owner_status = (
                phydrax.meshing._distribution_forest.replay_logical_solver_cell_owners(
                    states,
                    prepared.initial,
                    prepared.initial_owners,
                    neighbor_pairs=prepared.neighbor_pairs,
                    axis_name=prepared.axis_name,
                )
            )
        case phydrax.meshing._device_adaptation.PartitionedAdaptiveSimplex():
            owners, owner_status = (
                phydrax.meshing._distribution_forest.inherit_solver_cell_owners(
                    partitioned.parts,
                    states,
                    prepared.initial,
                    prepared.initial_owners,
                )
            )
        case _:
            raise TypeError(
                "collective_source.partitioned must retain its preparation kind; got changed owner proof"
            )
    if bool(jax.device_get(jnp.any(owner_status != 0))):
        raise ValueError(
            "collective_subdivision.owners must inherit accepted predecessor ancestry; got failed owner status"
        )
    if prepared.migration is not None:
        owners = prepared.migration.solver_cell_owners
    return _CollectiveSubdivision(
        source_cache_id=source_cache_id,
        initial_state_id=initial_state_id,
        native_arrays=native_arrays,
        native_receipt_id=native_receipt_id,
        partition_checks=partition_checks,
        verdicts=verdicts,
        summary=summary,
        owners=owners,
    )


def _pack_collective_epoch(
    prepared: _CollectiveSourcePreparation,
    proof: _CollectiveSubdivision,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CollectivePackedEpoch:
    compiled_states = states
    states, publication_exterior, publication_source_blocks = (
        phydrax.meshing._device_adaptation.packed_publication_states(
            prepared.preparation,
            prepared.layout,
            prepared.initial,
            compiled_states,
            prepared.exterior,
            prepared.uniform_refinement,
            raw_source_blocks=prepared.raw_blocks,
            initial_solver_cell_owners=prepared.initial_owners,
            scope=scope,
            budget=budget,
        )
    )
    if not isinstance(publication_source_blocks, tuple) or not publication_source_blocks:
        raise ValueError(
            "collective_publication.source_blocks must retain nonempty source block declarations; got absent declarations"
        )
    if any(
        not isinstance(block, CellBlock)
        or block.cell_kind
        != ("triangle" if prepared.layout.dimension == 2 else "tetrahedron")
        for block in publication_source_blocks
    ):
        raise ValueError(
            "collective_publication.source_blocks must match the allocated simplex family; got incompatible declarations"
        )
    saved_blocks = prepared.saved_publication_blocks
    if saved_blocks is not None:
        if tuple(block.block_id for block in publication_source_blocks) != tuple(
            block.block_id for block in saved_blocks
        ):
            raise ValueError(
                "collective_publication.source_blocks must reproduce retained source block order; got changed block identities"
            )
        publication_source_blocks = saved_blocks
    phydrax.meshing._bisection_kernel_host.uniform_count(
        8 * states.mesh.cell_ids.size + len(publication_source_blocks),
        8 * states.mesh.cell_ids.size,
        budget=budget,
    )
    used = (states.mesh.cell_ids >= 0) & (states.mesh.cell_active | ~states.retired)
    if states.blocks.shape != states.mesh.cell_ids.shape or not jnp.issubdtype(
        states.blocks.dtype, jnp.integer
    ):
        raise ValueError(
            "collective_publication.blocks must have integer indices on the allocated cell axis; got incompatible shape or dtype"
        )
    invalid = used & (
        (states.blocks < 0) | (states.blocks >= len(publication_source_blocks))
    )
    if bool(jax.device_get(jnp.any(invalid))):
        raise ValueError(
            "collective_publication.blocks must index a declared source block for every nonretired cell; got an out-of-range index"
        )
    owners = proof.owners
    if prepared.uniform_refinement is not None:
        sentinel = jnp.iinfo(jnp.int64).max
        raw_keys = jnp.where(
            compiled_states.mesh.cell_ids >= 0,
            compiled_states.mesh.cell_ids,
            sentinel,
        )
        positions = jax.vmap(jnp.searchsorted)(raw_keys, states.mesh.cell_ids)
        safe = jnp.minimum(positions, raw_keys.shape[1] - 1)
        inherited = jnp.take_along_axis(owners, safe, axis=1)
        present = jnp.take_along_axis(raw_keys, safe, axis=1) == states.mesh.cell_ids
        # A restored uniform root inherits the same minimum of
        # source-leaf solver owners as a binary coarsening. Physical
        # placement rank is not scientific solver ownership.
        roots = jnp.asarray(prepared.uniform_refinement.parent_ids)
        siblings = jnp.asarray(prepared.uniform_refinement.child_ids)
        root_positions = jnp.searchsorted(roots, states.mesh.cell_ids)
        root_safe = jnp.minimum(root_positions, roots.shape[0] - 1)
        root_present = roots[root_safe] == states.mesh.cell_ids
        child_ids = jnp.broadcast_to(siblings, (raw_keys.shape[0], *siblings.shape))
        child_positions = jax.vmap(jnp.searchsorted)(
            raw_keys, child_ids.reshape((raw_keys.shape[0], -1))
        )
        child_positions = child_positions.reshape(child_ids.shape)
        child_safe = jnp.minimum(child_positions, raw_keys.shape[1] - 1)
        child_keys = jax.vmap(lambda keys, positions: keys[positions])(
            raw_keys, child_safe
        )
        child_owners = jax.vmap(lambda values, positions: values[positions])(
            owners, child_safe
        )
        root_complete = jnp.all((child_keys == child_ids) & (child_owners >= 0), axis=-1)
        complete = jnp.take_along_axis(root_complete, root_safe, axis=1)
        newly_restored = states.mesh.cell_active & ~present
        if bool(jax.device_get(jnp.any(newly_restored & ~(root_present & complete)))):
            raise ValueError(
                "collective_publication.owners must retain every source-leaf owner for a restored root; got missing root support"
            )
        restored_owners = jnp.take_along_axis(
            jnp.min(child_owners, axis=-1), root_safe, axis=1
        )
        owners = jnp.where(
            states.mesh.cell_active,
            jnp.where(present, inherited, restored_owners),
            -1,
        )
    return _CollectivePackedEpoch(
        compiled_states=compiled_states,
        states=states,
        publication_exterior=publication_exterior,
        publication_source_blocks=publication_source_blocks,
        owners=owners,
    )


def _prepare_collective_publication(
    prepared: _CollectiveSourcePreparation,
    scientific: _CollectiveScientificSource,
    packed: _CollectivePackedEpoch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CollectivePublication:
    logical = phydrax.meshing._device_adaptation.logical_publication(
        prepared.source,
        packed.states,
        packed.owners,
        packed.publication_exterior,
        prepared.layout,
        placement_arrays=prepared.placement_arrays,
        original_source=None
        if prepared.uniform_refinement is None
        else prepared.uniform_refinement.source,
        uniform_refinement=prepared.uniform_refinement,
        # The serial commit binds the adapted target to its preparation.
        numeric_version=f"adaptation:{prepared.preparation.prepared_id}",
        publication_cell_owners=packed.owners if prepared.migration is not None else None,
        scope=scope,
        budget=budget,
    )
    logical_values = dict(logical.arrays)
    capacity = logical_values["cell_global_ids"].shape[0]
    active = packed.states.mesh.cell_active.reshape(-1)
    order = jnp.argsort(
        jnp.where(
            active, packed.states.mesh.cell_ids.reshape(-1), jnp.iinfo(jnp.int64).max
        ),
        stable=True,
    )[:capacity]
    classes = packed.states.cell_classes.reshape(-1)[order]
    facets = packed.states.facet_classes.reshape(
        (-1, packed.states.mesh.cells.shape[-1])
    )[order]
    placement = logical_values["cell_global_ids"].sharding
    organization_content = logical_array_value_collection_digest(
        {
            "cell_classes": jax.device_put(classes, placement),
            "facet_classes": jax.device_put(facets, placement),
        },
        logical_shapes={
            "cell_classes": (logical.counts[-1],),
            "facet_classes": (logical.counts[-1], packed.states.mesh.cells.shape[-1]),
        },
    )

    organization_arrays, global_organization_id = (
        phydrax.meshing._collective_organization.build_collective_organization_witness(
            scientific.original,
            packed.states,
            logical.entity_keys,
            logical.entity_ids,
            logical.counts,
            uniform_refinement=prepared.uniform_refinement,
            budget=budget,
        )
    )

    geometry_arrays, coordinate_geometry_id = (
        phydrax.meshing._collective_geometry.build_collective_geometry_witness(
            scientific.original,
            packed.states,
            organization_arrays,
            logical.entity_ids,
            logical.entity_owners,
            logical.counts,
            uniform_refinement=prepared.uniform_refinement,
            budget=budget,
        )
    )
    global_organization_id = canonical_fingerprint(
        {
            "kind": "collective-source-organization",
            "scientific_occurrences": global_organization_id,
            "target_class_content": organization_content,
        }
    )
    source_evidence_id = canonical_fingerprint(
        {
            "kind": "collective-source-evidence",
            "accepted_source": prepared.source.result_id,
            "scientific_source": scientific.scientific_id,
            "scientific_binding": scientific.scientific_binding,
            "predecessor": scientific.witness.predecessor_evidence_id,
            "source_premise": scientific.premise_id,
            "global_organization": global_organization_id,
        }
    )
    target_mesh_id = canonical_fingerprint(
        {
            "kind": "cell-mesh",
            "topology": logical.topology_id,
            "geometry": logical.geometry_id,
        }
    )
    return _CollectivePublication(
        arrays=logical.arrays,
        entity_keys=logical.entity_keys,
        entity_ids=logical.entity_ids,
        entity_owners=logical.entity_owners,
        counts=logical.counts,
        topology_id=logical.topology_id,
        geometry_id=logical.geometry_id,
        presentation_names=None
        if logical.presentation is None
        else logical.presentation.names,
        organization_arrays=organization_arrays,
        geometry_arrays=geometry_arrays,
        coordinate_geometry_id=coordinate_geometry_id,
        global_organization_id=global_organization_id,
        source_evidence_id=source_evidence_id,
        target_mesh_id=target_mesh_id,
    )


class AbstractCollectiveMeshTheorem(StrictModule):
    """Represent passed global mesh evidence for owner-local publication.

    Publication accepts subdivision evidence or independent initial evidence."""

    logical_arrays: eqx.AbstractVar[tuple[tuple[str, Array], ...]]
    entity_keys: eqx.AbstractVar[tuple[Array, ...]]
    entity_ids: eqx.AbstractVar[tuple[Array, ...]]
    entity_owners: eqx.AbstractVar[tuple[Array, ...]]
    topology_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]
    global_entity_counts: eqx.AbstractVar[tuple[int, ...]]
    coordinate_geometry_id: eqx.AbstractVar[str]
    source_evidence_id: eqx.AbstractVar[str]
    global_organization_id: eqx.AbstractVar[str]
    mesh_id: eqx.AbstractVar[str]
    partition_count: eqx.AbstractVar[int]
    evidence_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def require_passed(self) -> None:
        """Refuse failed global checks at the collective publication barrier."""


class _CollectivePartitionDim(Dim, minimum=1):
    """Partition axis of one collective subdivision theorem."""


class CollectiveMeshEvidence(AbstractCollectiveMeshTheorem, NonTrainableState):
    """Retain partition-indexed subdivision checks and publication arrays.

    Numerical states and findings remain sharded. Raw and published source block
    declarations retain separate identities when uniform roots are restored.
    ``evidence_id`` identifies the scientific subdivision theorem.
    ``stored_content_id`` binds every retained field and raw numerical bank for
    restoration and is anchored by the enclosing result's storage binding."""

    __strict_contract__ = True

    partition_checks: Bool[_CollectivePartitionDim, Literal[7]]
    global_checks: Bool[Literal[7]]
    source: CellMeshingResult
    preparation: phydrax.meshing._adaptation.PreparedMeshAdaptation
    initial_states: (
        _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    )
    compiled_states: (
        _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    )
    raw_source_blocks: tuple[CellBlock, ...]
    publication_source_blocks: tuple[CellBlock, ...]
    uniform_refinement: phydrax.meshing._bisection.BisectionUniformRefinement | None
    source_exterior: Array
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout
    neighbor_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    logical_arrays: tuple[tuple[str, Array], ...]
    entity_keys: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    entity_owners: tuple[Array, ...]
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    presentation_names: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    coordinate_geometry_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_audit_id: str = eqx.field(static=True)
    source_embedding: global_embedding_owner.GlobalEmbeddingCertificate | None
    source_premise_id: str = eqx.field(static=True)
    source_evidence_id: str = eqx.field(static=True)
    global_organization_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    partition_count: Size[_CollectivePartitionDim] = eqx.field(static=True)
    migration_record: phydrax.meshing._migration_record.CollectiveMigrationRecord | None
    rebalance_proposal: phydrax.meshing._distribution.PartitionedRebalanceProposal | None
    checks: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    stored_content_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        partitioned: phydrax.meshing.PartitionedAdaptiveSimplex | CollectiveMeshEvidence,
        states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
        /,
        *,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        self._initialize(partitioned, states, scope=scope, budget=budget)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CollectiveMeshEvidence], value: CollectiveMeshEvidence, /
    ) -> CollectiveMeshEvidence:
        """Authenticate retained fields without repeating charged proof generation.

        Mathematical replay remains the explicit partitioned-evidence validator's
        responsibility. The enclosing result authenticates the storage binding.
        """
        # Lazy: source registration resolves meshing declarations after import.
        from .._model._structure import restore_registered_fields
        from ..lifecycle._meshing_source_records import stored_record_content_id

        if type(value) is not cls:
            raise TypeError(
                "CollectiveMeshEvidence.stored_fields must have the reconstruction "
                f"type; got {type(value).__name__} for {cls.__name__}"
            )
        content_id = stored_record_content_id(value, excluded="stored_content_id")
        if not tree_content_equal(content_id, value.stored_content_id):
            raise ValueError(
                "CollectiveMeshEvidence.stored_content_id must match retained field and bank bindings; got changed stored content"
            )
        value.require_passed()
        stored = {member.name: getattr(value, member.name) for member in fields(value)}
        stored["stored_content_id"] = content_id
        rebuilt = restore_registered_fields(cls, stored)
        for member in fields(value):
            if not tree_content_equal(
                getattr(rebuilt, member.name), getattr(value, member.name)
            ):
                raise ValueError(
                    f"CollectiveMeshEvidence.{member.name} must match authenticated retained content; got changed stored content"
                )
        return rebuilt

    @staticmethod
    def _replay_source_embedding(
        source: CellMeshingResult,
        saved: global_embedding_owner.GlobalEmbeddingCertificate | None,
        renewed: global_embedding_owner.GlobalEmbeddingCertificate,
        /,
    ) -> tuple[global_embedding_owner.GlobalEmbeddingCertificate, str]:

        if saved is None:
            raise ValueError(
                "collective_source.embedding must retain its measured proof; got absent evidence"
            )
        certification = source.certification
        limits = (
            _owner_phydrax_geometry__embedding_state.MeshCertificateLimits()
            if certification is None or certification.embedding is None
            else certification.request.limits
        )
        for embedding in (saved, renewed):
            _owner_phydrax_meshing__certification.require_prepared_embedding(
                source.mesh, source.geometry, source.audit.validity, embedding, limits
            )
            for name, maximum in (
                ("source_expression_work_units", limits.maximum_work_units),
                ("source_expression_peak_bytes", limits.maximum_scratch_bytes),
            ):
                quantity = _owner_phydrax__validation.nonnegative_integer(
                    object.__getattribute__(embedding, name), f"collective_source.{name}"
                )
                if quantity > maximum:
                    raise ValueError(
                        "collective_source.embedding must fit its certificate controls; "
                        f"got {name}={quantity} > {maximum}"
                    )
        historical = frozenset(
            {
                "source_expression_work_units",
                "source_expression_peak_bytes",
                "certificate_id",
            }
        )
        for field in fields(saved):
            if field.name not in historical and not tree_content_equal(
                object.__getattribute__(saved, field.name),
                object.__getattribute__(renewed, field.name),
            ):
                raise ValueError(
                    "collective_source.embedding must reproduce retained proof fields; "
                    f"got changed {field.name}"
                )
        if certification is not None:
            if certification.embedding is not None and not tree_content_equal(
                saved, certification.embedding
            ):
                raise ValueError(
                    "collective_source.embedding must bind its source certification; "
                    "got changed measured proof"
                )
            premise_id = certification.report_id
        else:
            premise_id = canonical_fingerprint(
                {
                    "kind": "audited-meshing-source-embedding",
                    "mesh": source.mesh.mesh_id,
                    "geometry": _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                        source.geometry
                    ),
                    "audit": source.audit.report_id,
                    "embedding": saved.certificate_id,
                }
            )
        return saved, premise_id

    def _initialize(
        self,
        partitioned: phydrax.meshing.PartitionedAdaptiveSimplex | CollectiveMeshEvidence,
        states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
        /,
        *,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        prepared = _prepare_collective_source(
            partitioned, states, scope=scope, budget=budget
        )
        lanes = prepared.initial.mesh.cell_ids.size + states.mesh.cell_ids.size
        phydrax.meshing._bisection_kernel_host.uniform_count(
            8 * lanes + len(prepared.raw_blocks), 8 * lanes, budget=budget
        )
        for raw in (prepared.initial, states):
            if raw.blocks.shape != raw.mesh.cell_ids.shape or not jnp.issubdtype(
                raw.blocks.dtype, jnp.integer
            ):
                raise ValueError(
                    "collective_source.blocks must have integer indices on the allocated cell axis; got incompatible shape or dtype"
                )
            used = (raw.mesh.cell_ids >= 0) & (raw.mesh.cell_active | ~raw.retired)
            invalid = used & ((raw.blocks < 0) | (raw.blocks >= len(prepared.raw_blocks)))
            if bool(jax.device_get(jnp.any(invalid))):
                raise ValueError(
                    "collective_source.blocks must index a declared source block for every nonretired cell; got an out-of-range index"
                )
        scientific = _bind_collective_scientific_source(prepared, partitioned, states)
        proof = _prove_collective_subdivision(
            prepared, partitioned, states, scope=scope, budget=budget
        )
        packed = _pack_collective_epoch(
            prepared, proof, states, scope=scope, budget=budget
        )
        publication = _prepare_collective_publication(
            prepared, scientific, packed, scope=scope, budget=budget
        )
        self.partition_checks = proof.partition_checks
        self.global_checks = proof.verdicts
        self.source = prepared.source
        self.preparation = prepared.preparation
        self.initial_states = prepared.initial
        self.compiled_states = packed.compiled_states
        self.raw_source_blocks = prepared.raw_blocks
        self.publication_source_blocks = packed.publication_source_blocks
        self.uniform_refinement = prepared.uniform_refinement
        self.source_exterior = prepared.exterior
        self.layout = prepared.layout
        self.neighbor_pairs = prepared.neighbor_pairs
        self.axis_name = prepared.axis_name
        self.logical_arrays = tuple(
            sorted(
                (
                    *publication.arrays,
                    *publication.organization_arrays,
                    *publication.geometry_arrays,
                    *proof.native_arrays,
                )
            )
        )
        self.entity_keys = publication.entity_keys
        self.entity_ids = publication.entity_ids
        self.entity_owners = publication.entity_owners
        self.global_entity_counts = publication.counts
        if publication.presentation_names is None:
            raise ValueError(
                "collective_publication.presentation must bind prepared block order; got absent evidence"
            )
        self.presentation_names = publication.presentation_names
        self.topology_id = publication.topology_id
        self.geometry_id = publication.geometry_id
        self.coordinate_geometry_id = publication.coordinate_geometry_id
        self.source_mesh_id = prepared.source.mesh.mesh_id
        self.source_audit_id = scientific.source_audit_id
        self.source_embedding = scientific.source_embedding
        self.source_premise_id = scientific.premise_id
        self.source_evidence_id = publication.source_evidence_id
        self.global_organization_id = publication.global_organization_id
        self.mesh_id = publication.target_mesh_id
        self.partition_count = proof.partition_checks.shape[0]
        # Lazy: portable migration records depend on completed live class bodies.
        from ._migration_record import CollectiveMigrationRecord

        self.migration_record = (
            None
            if prepared.migration is None
            else CollectiveMigrationRecord(prepared.migration)
        )
        self.rebalance_proposal = prepared.rebalance_proposal
        self.checks = _BISECTION_CERTIFICATE_CHECKS
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "collective-affine-bisection-evidence",
                **(
                    {}
                    if self.migration_record is None
                    else {"migration": self.migration_record.migration_id}
                ),
                "source_mesh": prepared.source.mesh.mesh_id,
                "source_audit": scientific.source_audit_id,
                "source_routing": proof.source_cache_id,
                "rebalance": None
                if prepared.rebalance_proposal is None
                else prepared.rebalance_proposal.proposal_id,
                "initial_numerical_state": proof.initial_state_id,
                "compiled_numerical_state": logical_array_value_collection_digest(
                    {
                        jax.tree_util.keystr(path): value
                        for path, value in jax.tree_util.tree_flatten_with_path(
                            packed.compiled_states
                        )[0]
                        if isinstance(value, Array)
                    }
                ),
                "raw_source_blocks": [block.block_id for block in prepared.raw_blocks],
                "publication_source_blocks": [
                    block.block_id for block in packed.publication_source_blocks
                ],
                "uniform_source": None
                if prepared.uniform_refinement is None
                else prepared.uniform_refinement.lineage_id,
                "target_mesh": publication.target_mesh_id,
                "coordinate_geometry": publication.coordinate_geometry_id,
                "native_midpoint_receipts": proof.native_receipt_id,
                "source_evidence": publication.source_evidence_id,
                "global_organization": publication.global_organization_id,
                "partition_count": self.partition_count,
                "numerical_receipts": logical_array_value_collection_digest(
                    dict(self.logical_arrays)
                ),
                "checks": list(zip(self.checks, proof.summary.tolist(), strict=True)),
            }
        )
        # Lazy: the registered source graph resolves after meshing declarations.
        from ..lifecycle._meshing_source_records import stored_record_content_id

        self.stored_content_id = stored_record_content_id(
            self, excluded="stored_content_id"
        )

    def require_passed(self) -> None:
        """Refuse failed global checks at the collective publication barrier."""
        summary = np.asarray(jax.device_get(self.global_checks), dtype=np.bool_)
        if not np.all(summary):
            failed = tuple(
                name
                for name, passed in zip(self.checks, summary, strict=True)
                if not passed
            )
            raise ValueError(
                "collective_evidence.global_checks must pass every check; got failed checks: "
                + ", ".join(failed)
            )


class MeshingComplianceReport(StrictModule, NonTrainableState):
    specification_id: str = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        specification_id: str,
        /,
        *,
        issues: tuple[str, ...] = (),
        requested: tuple[tuple[str, float], ...] = (),
        achieved: tuple[tuple[str, float], ...] = (),
    ) -> None:
        specification = str(specification_id).strip()
        if not specification:
            raise ValueError(
                "meshing_compliance.specification_id must be nonempty; got an empty identity"
            )
        issues_ = tuple(str(value) for value in issues)
        requested_ = tuple((str(name), float(value)) for name, value in requested)
        achieved_ = tuple((str(name), float(value)) for name, value in achieved)
        self.specification_id = specification
        self.passed = not issues_
        self.issues = issues_
        self.requested = requested_
        self.achieved = achieved_
        self.report_id = canonical_fingerprint(
            {
                "kind": "meshing-compliance-report",
                "specification": specification,
                "issues": issues_,
                "requested": requested_,
                "achieved": achieved_,
            }
        )


class MeshingRuntimeInfo(StrictModule, NonTrainableState):
    provider_id: str = eqx.field(static=True)
    actual_version: str = eqx.field(static=True)
    execution_mode: _dependency_phydrax_meshing__contracts.MeshingExecutionMode = (
        eqx.field(static=True)
    )
    deterministic: bool = eqx.field(static=True)
    enforced_limits: tuple[str, ...] = eqx.field(static=True)
    unenforced_limits: tuple[str, ...] = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        actual_version: str,
        execution_mode: _dependency_phydrax_meshing__contracts.MeshingExecutionMode,
        /,
        *,
        deterministic: bool,
        enforced_limits: tuple[str, ...] = (),
        unenforced_limits: tuple[str, ...] = (),
    ) -> None:
        provider = str(provider_id).strip()
        version = str(actual_version).strip()
        if not provider or not version:
            raise ValueError(
                "meshing_runtime.provider_id must be nonempty together with actual_version; "
                f"got provider_id={provider!r}, actual_version={version!r}"
            )
        if not isinstance(
            execution_mode, _dependency_phydrax_meshing__contracts.MeshingExecutionMode
        ):
            raise TypeError(
                "meshing_runtime.execution_mode must be MeshingExecutionMode; got an invalid selector kind"
            )
        self.provider_id = provider
        self.actual_version = version
        self.execution_mode = execution_mode
        self.deterministic = bool(deterministic)
        self.enforced_limits = tuple(str(value) for value in enforced_limits)
        self.unenforced_limits = tuple(str(value) for value in unenforced_limits)
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "meshing-runtime-info",
                "provider": provider,
                "actual_version": version,
                "execution_mode": execution_mode.value,
                "deterministic": bool(deterministic),
                "enforced_limits": self.enforced_limits,
                "unenforced_limits": self.unenforced_limits,
            }
        )


def _require_bound_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
    trace: _dependency_phydrax_meshing__trace.MeshingTrace,
    runtime: MeshingRuntimeInfo,
    certification: _dependency_phydrax_meshing__certification.MeshCertificationReport
    | None,
    /,
) -> None:
    """Refuse evidence of other inputs and mandatory checks left undecided."""

    geometry_id = (
        _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            geometry
        )
    )
    audit.require_decided()
    if certification is not None:
        if (
            certification.mesh_id != mesh.mesh_id
            or certification.geometry_id != geometry_id
            or certification.audit_report_id != audit.report_id
        ):
            raise ValueError(
                "result.certification must bind the publication mesh, geometry and audit; got mismatched certification evidence"
            )
        certification.require_passed()
    binding = trace.binding
    if binding is not None and (
        binding.topology_id != mesh.topology_id
        or binding.geometry_id != geometry_id
        or binding.geometry_layout_id != geometry.geometry_layout_id
        or binding.runtime_id != runtime.runtime_id
        or audit.policy_id not in binding.policy_ids
    ):
        raise ValueError(
            "result.trace.binding must bind topology, coordinates, layout, audit policy and runtime; got mismatched trace evidence"
        )


class CollectiveMeshStorageBinding(StrictModule, NonTrainableState):
    """Admit scientific bank contents before owner-local construction."""

    evidence: AbstractCollectiveMeshTheorem
    arrays: tuple[tuple[str, Array], ...]
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        evidence: AbstractCollectiveMeshTheorem,
        arrays: tuple[tuple[str, Array], ...],
        global_entity_counts: tuple[int, ...],
        /,
    ) -> None:
        evidence.require_passed()
        if global_entity_counts != evidence.global_entity_counts:
            raise ValueError(
                "collective_storage.global_entity_counts must match accepted evidence; got mismatched counts"
            )
        stored = dict(arrays)
        theorem = dict(evidence.logical_arrays)
        if any(name not in stored for name in theorem):
            raise ValueError(
                "collective_storage.arrays must contain every accepted scientific bank; got missing array names"
            )
        actual = {name: stored[name] for name in theorem}
        digest = logical_array_value_collection_digest(actual)
        if digest != logical_array_value_collection_digest(theorem):
            raise ValueError(
                "collective_storage.arrays must reproduce accepted scientific bytes; got changed content digest"
            )
        self.evidence = evidence
        self.arrays = tuple(sorted(actual.items()))
        self.global_entity_counts = global_entity_counts
        self.content_id = canonical_fingerprint(
            {
                "kind": "accepted-collective-storage-bank",
                "evidence": evidence.evidence_id,
                "counts": global_entity_counts,
                "actual_bytes": digest,
            }
        )

    @checked
    def require_storage(
        self,
        mesh: CellMesh,
        evidence: AbstractCollectiveMeshTheorem,
        /,
    ) -> None:
        storage = mesh.storage
        if storage is None or (
            evidence.evidence_id != self.evidence.evidence_id
            or evidence.mesh_id != self.evidence.mesh_id
            or evidence.coordinate_geometry_id != self.evidence.coordinate_geometry_id
            or storage.global_entity_counts != self.global_entity_counts
        ):
            raise ValueError(
                "collective_storage.mesh.storage must bind the admitted collective source; got mismatched evidence or counts"
            )
        actual = dict(storage.logical_arrays)
        if any(
            name not in actual or actual[name] is not value for name, value in self.arrays
        ):
            raise ValueError(
                "collective_storage.mesh.storage.logical_arrays must retain admitted array objects; got missing or replaced banks"
            )


@checked
def restore_collective_mesh_storage_binding(
    mesh: CellMesh,
    evidence: AbstractCollectiveMeshTheorem,
    /,
) -> CollectiveMeshStorageBinding:
    """Admit restored scientific array contents before binding local storage."""
    if mesh.storage is None:
        raise ValueError(
            "collective_storage.mesh.storage must be an owner-local carrier; got absent storage"
        )
    return CollectiveMeshStorageBinding(
        evidence,
        mesh.storage.logical_arrays,
        mesh.storage.global_entity_counts,
    )


@final
class _ResultInputs(StrictModule):
    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport
    quality: _dependency_phydrax_meshing__quality.CellQualityReport
    compliance: MeshingComplianceReport
    trace: _dependency_phydrax_meshing__trace.MeshingTrace
    provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo
    runtime: MeshingRuntimeInfo
    derivative_mode: _dependency_phydrax_meshing__contracts.MeshingDerivativeMode
    provenance: SemanticProvenance
    boundary: SurfaceModel | None
    patches: tuple[_dependency_phydrax_meshing__organization_records.MeshPatch, ...]
    zones: tuple[_dependency_phydrax_meshing__organization_records.MeshZone, ...]
    labels: tuple[_dependency_phydrax_meshing__organization_records.MeshLabel, ...]
    attributes: tuple[
        _dependency_phydrax_meshing__organization_records.MeshAttribute, ...
    ]
    associations: tuple[association_state_owner.GeometryAssociation, ...]
    adapter_reports: tuple[AdapterReport, ...]
    certification: (
        _dependency_phydrax_meshing__certification.MeshCertificationReport | None
    )
    region_evidence: (
        _dependency_phydrax_meshing__organization_records.RegionMeshingEvidence | None
    )
    region_boundary_evidence: tuple[
        _dependency_phydrax_meshing__organization_records.RegionBoundaryEvidence, ...
    ]
    collective_evidence: AbstractCollectiveMeshTheorem | None
    scope_projections: (
        tuple[tuple[str, _dependency_phydrax_meshing__scope.MeshScopeProjection], ...]
        | None
    )
    attribute_projections: (
        tuple[
            tuple[
                str,
                _dependency_phydrax_meshing__organization_records.MeshAttributeProjection,
            ],
            ...,
        ]
        | None
    )
    storage_binding: CollectiveMeshStorageBinding | None
    collective_certificates: (
        tuple[
            global_embedding_owner.GlobalEmbeddingCertificate,
            domain_coverage_owner.DomainCoverageCertificate,
        ]
        | None
    )
    execution_evidence: NativeExecutionRecord | None
    surface_source: (
        _dependency_phydrax_geometry__surface_source_support.SurfaceSourceCharts | None
    )


@final
class _ResultCarrier(StrictModule):
    storage_binding: CollectiveMeshStorageBinding | None
    collective_certificates: (
        tuple[
            global_embedding_owner.GlobalEmbeddingCertificate,
            domain_coverage_owner.DomainCoverageCertificate,
        ]
        | None
    )


@final
class _ResultVerdicts(StrictModule):
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport
    quality: _dependency_phydrax_meshing__quality.CellQualityReport
    compliance: MeshingComplianceReport
    trace: _dependency_phydrax_meshing__trace.MeshingTrace
    runtime: MeshingRuntimeInfo


@final
class _ResultOrganization(StrictModule):
    patches: tuple[_dependency_phydrax_meshing__organization_records.MeshPatch, ...]
    zones: tuple[_dependency_phydrax_meshing__organization_records.MeshZone, ...]
    labels: tuple[_dependency_phydrax_meshing__organization_records.MeshLabel, ...]
    attributes: tuple[
        _dependency_phydrax_meshing__organization_records.MeshAttribute, ...
    ]
    associations: tuple[association_state_owner.GeometryAssociation, ...]
    adapter_reports: tuple[AdapterReport, ...]
    boundary_regions: tuple[
        _dependency_phydrax_meshing__organization_records.RegionBoundaryEvidence, ...
    ]


@final
class _ResultIdentity(StrictModule):
    result_id: str


def _prepare_result_carrier(
    inputs: _ResultInputs,
    /,
) -> _ResultCarrier:
    storage_binding = inputs.storage_binding
    if inputs.execution_evidence is not None:
        inputs.execution_evidence.require_valid()
    if inputs.surface_source is not None:
        if inputs.certification is None or inputs.certification.fidelity is None:
            raise ValueError(
                "result.surface_source must bind a continuous source certificate; got absent fidelity certification"
            )
        if (
            inputs.surface_source.coordinate_contract.spatial_id
            != inputs.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "result.surface_source.coordinate_contract must match publication coordinates; got a different spatial_id"
            )
        source = _dependency_phydrax_geometry__surface_source_support.require_surface_bound_source(
            inputs.certification.request.source,
            "result.surface_source.certification_source",
        )
        inputs.surface_source.require_bound(inputs.mesh, inputs.geometry, source)

    if inputs.mesh.storage is not None:
        if inputs.collective_evidence is None:
            raise ValueError(
                "result.collective_evidence must be present for owner-local storage; got absent evidence"
            )
        if (
            inputs.collective_evidence.mesh_id != inputs.mesh.mesh_id
            or inputs.collective_evidence.partition_count
            != inputs.mesh.storage.partition_count
            or inputs.collective_evidence.evidence_id != inputs.mesh.storage.evidence_id
            or inputs.collective_evidence.coordinate_geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                inputs.geometry
            )
        ):
            raise ValueError(
                "result.collective_evidence must bind mesh, geometry and partition storage; got mismatched publication evidence"
            )
        if storage_binding is None:
            if any(
                not value.is_fully_addressable
                for _, value in inputs.mesh.storage.logical_arrays
            ):
                inputs.collective_evidence.require_passed()
                raise ValueError(
                    "result.storage_binding must admit nonaddressable scientific banks on all owners; got absent admission"
                )
            storage_binding = restore_collective_mesh_storage_binding(
                inputs.mesh, inputs.collective_evidence
            )
        else:
            inputs.collective_evidence.require_passed()
        storage_binding.require_storage(inputs.mesh, inputs.collective_evidence)
    elif inputs.collective_evidence is not None:
        raise ValueError(
            "result.collective_evidence must be absent for dense serial storage; got owner-local evidence"
        )
    if inputs.collective_certificates is not None:
        _require_collective_certificates(
            inputs.mesh, inputs.geometry, inputs.collective_certificates
        )
    return _ResultCarrier(
        storage_binding=storage_binding,
        collective_certificates=inputs.collective_certificates,
    )


def _require_result_verdicts(
    inputs: _ResultInputs,
    /,
) -> _ResultVerdicts:
    inputs.geometry.resolve(inputs.mesh)
    if inputs.audit.mesh_id != inputs.mesh.mesh_id:
        raise ValueError(
            "result.audit.mesh_id must match the publication mesh; got mismatched identity"
        )
    expected_scope = (
        _dependency_phydrax_meshing__audit.CellMeshAuditScope.DENSE_SERIAL
        if inputs.mesh.storage is None
        else _dependency_phydrax_meshing__audit.CellMeshAuditScope.OWNER_LOCAL_CLOSURE
    )
    if inputs.audit.audit_scope != expected_scope or inputs.audit.storage_id != (
        None if inputs.mesh.storage is None else inputs.mesh.storage.storage_id
    ):
        raise ValueError(
            "result.audit.storage must match the accepted mesh carrier and scope; got mismatched storage evidence"
        )
    if (
        inputs.audit.geometry_layout_id != inputs.geometry.geometry_layout_id
        or inputs.audit.geometry_id
        != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            inputs.geometry
        )
    ):
        raise ValueError(
            "result.audit.geometry must bind result geometry values and layout; got mismatched geometry evidence"
        )
    inputs.audit.require_passed()
    if inputs.quality.report_id != inputs.audit.quality.report_id:
        raise ValueError(
            "result.quality.report_id must match the mesh audit; got mismatched quality evidence"
        )
    if not inputs.compliance.passed:
        raise ValueError(
            "result.compliance must pass the requested conditions; got failed compliance"
        )
    if not inputs.trace.successful:
        raise ValueError(
            "result.trace must record successful stages; got an unsuccessful trace"
        )
    if inputs.runtime.provider_id != inputs.provider.provider_id:
        raise ValueError(
            "result.runtime.provider_id must match the publication provider; got mismatched provider identity"
        )
    _require_bound_evidence(
        inputs.mesh,
        inputs.geometry,
        inputs.audit,
        inputs.trace,
        inputs.runtime,
        inputs.certification,
    )
    if not isinstance(
        inputs.derivative_mode,
        _dependency_phydrax_meshing__contracts.MeshingDerivativeMode,
    ):
        raise TypeError(
            "result.derivative_mode must be MeshingDerivativeMode; got "
            f"{type(inputs.derivative_mode).__name__}"
        )
    return _ResultVerdicts(
        audit=inputs.audit,
        quality=inputs.quality,
        compliance=inputs.compliance,
        trace=inputs.trace,
        runtime=inputs.runtime,
    )


def _prepare_result_organization(
    inputs: _ResultInputs,
    /,
) -> _ResultOrganization:
    patches_ = tuple(inputs.patches)
    zones_ = _dependency_phydrax_meshing__organization_records.validate_mesh_zones(
        tuple(inputs.zones)
    )
    labels_ = _dependency_phydrax_meshing__organization_records.validate_mesh_labels(
        tuple(inputs.labels)
    )
    if not all(
        isinstance(value, _dependency_phydrax_meshing__organization_records.MeshPatch)
        for value in patches_
    ):
        raise TypeError(
            "result.patches must contain MeshPatch values; got an invalid patch kind"
        )
    if inputs.region_evidence is not None:
        inputs.region_evidence.require_current(
            inputs.mesh, zones_, patches_, geometry=inputs.geometry
        )
    boundary_regions = tuple(inputs.region_boundary_evidence)
    if not all(
        isinstance(
            value,
            _dependency_phydrax_meshing__organization_records.RegionBoundaryEvidence,
        )
        for value in boundary_regions
    ):
        raise TypeError(
            "result.region_boundary_evidence must contain RegionBoundaryEvidence values; got an invalid evidence kind"
        )
    source_regions = {value.source_region_id for value in boundary_regions}
    if len(source_regions) != len(boundary_regions):
        raise ValueError(
            "result.region_boundary_evidence.source_region_id must be unique; got duplicate identities"
        )
    for value in boundary_regions:
        value.require_current(inputs.mesh, labels_, patches_)
    for patch in patches_:
        if patch.source_adjacent_region_ids is not None and any(
            region is not None and region not in source_regions
            for region in patch.source_adjacent_region_ids
        ):
            raise ValueError(
                "result.patches.source_adjacent_region_ids must have region boundary evidence; got an unbound source region"
            )
    if not all(
        isinstance(value, _dependency_phydrax_meshing__organization_records.MeshAttribute)
        for value in inputs.attributes
    ):
        raise TypeError(
            "result.attributes must contain MeshAttribute values; got an invalid attribute kind"
        )
    if not all(
        isinstance(value, association_state_owner.GeometryAssociation)
        for value in inputs.associations
    ):
        raise TypeError(
            "result.associations must contain GeometryAssociation values; got an invalid association kind"
        )
    if not all(isinstance(value, AdapterReport) for value in inputs.adapter_reports):
        raise TypeError(
            "result.adapter_reports must contain AdapterReport values; got an invalid report kind"
        )
    if (
        inputs.boundary is not None
        and inputs.boundary.metadata.coordinate_contract.spatial_id
        != inputs.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "result.boundary.coordinate_contract must match publication coordinates; got a different spatial_id"
        )
    evidence_issues = _dependency_phydrax_meshing__audit.mesh_evidence_issues(
        inputs.mesh,
        inputs.boundary,
        patches_,
        zones_,
        labels_,
        inputs.attributes,
        inputs.associations,
    )
    if evidence_issues:
        raise ValueError(
            "result.organization must bind the mesh; got " + "; ".join(evidence_issues)
        )
    if inputs.audit.evidence_id != _dependency_phydrax_meshing__audit.evidence_id(
        patches_,
        zones_,
        labels_,
        inputs.attributes,
        inputs.associations,
    ):
        raise ValueError(
            "result.organization must match the audited evidence; got changed organization or associations"
        )
    return _ResultOrganization(
        patches=patches_,
        zones=zones_,
        labels=labels_,
        boundary_regions=boundary_regions,
        attributes=tuple(inputs.attributes),
        associations=tuple(inputs.associations),
        adapter_reports=tuple(inputs.adapter_reports),
    )


def _prepare_result_identity(
    inputs: _ResultInputs,
    organization: _ResultOrganization,
    /,
) -> _ResultIdentity:
    # Lazy: _result -> _initial_certification -> _canonical -> _lineage import cycle.
    import phydrax.meshing._initial_certification

    if inputs.mesh.storage is not None:
        # Lazy: high organization validation reaches collective owners and _lineage.
        import phydrax.meshing._organization as _dependency_phydrax_meshing__organization

        match inputs.collective_evidence:
            case CollectiveMeshEvidence():
                original = require_original_meshing_source(
                    inputs.collective_evidence.source
                    if inputs.collective_evidence.uniform_refinement is None
                    else inputs.collective_evidence.uniform_refinement.source,
                )
                _dependency_phydrax_meshing__organization.validate_local_mesh_organization(
                    original,
                    inputs.mesh,
                    dict(inputs.collective_evidence.logical_arrays),
                    organization.patches,
                    organization.zones,
                    organization.labels,
                    inputs.attributes,
                    scope_projections=inputs.scope_projections,
                    attribute_projections=inputs.attribute_projections,
                )
                source_identity = {
                    "source_mesh": inputs.collective_evidence.source_mesh_id,
                    "source_audit": inputs.collective_evidence.source_audit_id,
                }
            case phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence():
                if (
                    inputs.compliance.specification_id
                    != inputs.collective_evidence.specification.specification_id
                ):
                    raise ValueError(
                        "result.compliance.specification_id must match the authored request; got mismatched initial specification"
                    )
                if inputs.scope_projections is None:
                    raise ValueError(
                        "result.scope_projections must retain prepared all-owner organization scopes; got absent projections"
                    )
                _dependency_phydrax_meshing__organization.validate_local_mesh_organization(
                    inputs.collective_evidence,
                    inputs.mesh,
                    dict(inputs.collective_evidence.logical_arrays),
                    organization.patches,
                    organization.zones,
                    organization.labels,
                    inputs.attributes,
                    scope_projections=inputs.scope_projections,
                    attribute_projections=inputs.attribute_projections,
                )
                source_identity = {
                    "source_domain": inputs.collective_evidence.compiled.domain.domain_id,
                    "compiled_domain": inputs.collective_evidence.compiled.compiled_id,
                    "specification": inputs.collective_evidence.specification.specification_id,
                }
            case _:
                raise TypeError(
                    "result.collective_evidence must be a supported collective proof; got unknown theorem kind"
                )
        result_id = canonical_fingerprint(
            {
                "kind": "owner-local-cell-meshing-result",
                "mesh": inputs.mesh.mesh_id,
                "topology": inputs.mesh.topology_id,
                "geometry": _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                    inputs.geometry
                ),
                "geometry_layout": inputs.geometry.geometry_layout_id,
                "coordinate_contract": inputs.coordinate_contract.spatial_id,
                **source_identity,
                "source_evidence": inputs.collective_evidence.source_evidence_id,
                "organization": inputs.collective_evidence.global_organization_id,
                "collective": inputs.collective_evidence.evidence_id,
                "provider": inputs.provider.provider_id,
                "derivative_mode": inputs.derivative_mode.value,
                "collective_certificates": _collective_certificate_ids(
                    inputs.collective_certificates
                ),
                "surface_source": None
                if inputs.surface_source is None
                else inputs.surface_source.source_chart_id,
            }
        )
        return _ResultIdentity(result_id)
    result_id = canonical_fingerprint(
        {
            "kind": "cell-meshing-result",
            "mesh": inputs.mesh.mesh_id,
            "geometry_layout": inputs.geometry.geometry_layout_id,
            "geometry": inputs.audit.geometry_id,
            "coordinate_contract": inputs.coordinate_contract.spatial_id,
            "boundary": None if inputs.boundary is None else inputs.boundary.model_id,
            "patches": [value.patch_id for value in organization.patches],
            "zones": [value.zone_id for value in organization.zones],
            "labels": [value.label_id for value in organization.labels],
            "attributes": [value.attribute_id for value in inputs.attributes],
            "associations": [value.association_id for value in inputs.associations],
            "audit": inputs.audit.report_id,
            "quality": inputs.quality.report_id,
            "compliance": inputs.compliance.report_id,
            "certification": (
                None if inputs.certification is None else inputs.certification.report_id
            ),
            "regions": (
                None
                if inputs.region_evidence is None
                else inputs.region_evidence.evidence_id
            ),
            "region_boundaries": tuple(
                value.evidence_id for value in organization.boundary_regions
            ),
            "collective": (
                None
                if inputs.collective_evidence is None
                else inputs.collective_evidence.evidence_id
            ),
            "trace": inputs.trace.trace_id,
            "adapter_reports": [value.report_id for value in inputs.adapter_reports],
            "provider": inputs.provider.provider_id,
            "runtime": inputs.runtime.runtime_id,
            "derivative_mode": inputs.derivative_mode.value,
            "provenance": inputs.provenance.semantic_id,
            "collective_certificates": None,
            "surface_source": None
            if inputs.surface_source is None
            else inputs.surface_source.source_chart_id,
        }
    )
    return _ResultIdentity(result_id)


class CellMeshingResult(StrictModule, NonTrainableState):
    """Successful meshing publication with bound, decided evidence.

    Construction refuses a failed audit, a REJECT-governed check left
    unresolved, a failed or unresolved trace stage, and (when supplied) a route
    certification that is not passed or not bound to this mesh and audit.
    ``result_id`` identifies the scientific publication independently of rank.
    ``stored_content_id`` anchors its collective rank-local storage binding,
    authenticated by the archive content ID, or is absent for other publications.
    """

    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    boundary: SurfaceModel | None
    patches: tuple[_dependency_phydrax_meshing__organization_records.MeshPatch, ...]
    zones: tuple[_dependency_phydrax_meshing__organization_records.MeshZone, ...]
    labels: tuple[_dependency_phydrax_meshing__organization_records.MeshLabel, ...]
    attributes: tuple[
        _dependency_phydrax_meshing__organization_records.MeshAttribute, ...
    ]
    associations: tuple[association_state_owner.GeometryAssociation, ...]
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport
    quality: _dependency_phydrax_meshing__quality.CellQualityReport
    compliance: MeshingComplianceReport
    certification: (
        _dependency_phydrax_meshing__certification.MeshCertificationReport | None
    )
    region_evidence: (
        _dependency_phydrax_meshing__organization_records.RegionMeshingEvidence | None
    )
    region_boundary_evidence: tuple[
        _dependency_phydrax_meshing__organization_records.RegionBoundaryEvidence, ...
    ]
    collective_evidence: AbstractCollectiveMeshTheorem | None
    trace: _dependency_phydrax_meshing__trace.MeshingTrace
    adapter_reports: tuple[AdapterReport, ...]
    provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo
    runtime: MeshingRuntimeInfo
    derivative_mode: _dependency_phydrax_meshing__contracts.MeshingDerivativeMode = (
        eqx.field(static=True)
    )
    storage_binding: CollectiveMeshStorageBinding | None
    provenance: SemanticProvenance
    result_id: str = eqx.field(static=True)
    stored_content_id: str | None = eqx.field(static=True)
    collective_certificates: (
        tuple[
            global_embedding_owner.GlobalEmbeddingCertificate,
            domain_coverage_owner.DomainCoverageCertificate,
        ]
        | None
    )
    execution_evidence: NativeExecutionRecord | None
    surface_source: (
        _dependency_phydrax_geometry__surface_source_support.SurfaceSourceCharts | None
    )

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
        quality: _dependency_phydrax_meshing__quality.CellQualityReport,
        compliance: MeshingComplianceReport,
        trace: _dependency_phydrax_meshing__trace.MeshingTrace,
        provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo,
        runtime: MeshingRuntimeInfo,
        derivative_mode: _dependency_phydrax_meshing__contracts.MeshingDerivativeMode,
        provenance: SemanticProvenance,
        /,
        *,
        boundary: SurfaceModel | None = None,
        patches: tuple[
            _dependency_phydrax_meshing__organization_records.MeshPatch, ...
        ] = (),
        zones: tuple[
            _dependency_phydrax_meshing__organization_records.MeshZone, ...
        ] = (),
        labels: tuple[
            _dependency_phydrax_meshing__organization_records.MeshLabel, ...
        ] = (),
        attributes: tuple[
            _dependency_phydrax_meshing__organization_records.MeshAttribute, ...
        ] = (),
        associations: tuple[association_state_owner.GeometryAssociation, ...] = (),
        adapter_reports: tuple[AdapterReport, ...] = (),
        certification: _dependency_phydrax_meshing__certification.MeshCertificationReport
        | None = None,
        region_evidence: _dependency_phydrax_meshing__organization_records.RegionMeshingEvidence
        | None = None,
        region_boundary_evidence: tuple[
            _dependency_phydrax_meshing__organization_records.RegionBoundaryEvidence, ...
        ] = (),
        collective_evidence: AbstractCollectiveMeshTheorem | None = None,
        scope_projections: tuple[
            tuple[str, _dependency_phydrax_meshing__scope.MeshScopeProjection], ...
        ]
        | None = None,
        attribute_projections: tuple[
            tuple[
                str,
                _dependency_phydrax_meshing__organization_records.MeshAttributeProjection,
            ],
            ...,
        ]
        | None = None,
        storage_binding: CollectiveMeshStorageBinding | None = None,
        collective_certificates: tuple[
            global_embedding_owner.GlobalEmbeddingCertificate,
            domain_coverage_owner.DomainCoverageCertificate,
        ]
        | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
        surface_source: _dependency_phydrax_geometry__surface_source_support.SurfaceSourceCharts
        | None = None,
    ) -> None:
        inputs = _ResultInputs(
            mesh=mesh,
            geometry=geometry,
            coordinate_contract=coordinate_contract,
            audit=audit,
            quality=quality,
            compliance=compliance,
            trace=trace,
            provider=provider,
            runtime=runtime,
            derivative_mode=derivative_mode,
            provenance=provenance,
            boundary=boundary,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            adapter_reports=adapter_reports,
            certification=certification,
            region_evidence=region_evidence,
            region_boundary_evidence=region_boundary_evidence,
            collective_evidence=collective_evidence,
            scope_projections=scope_projections,
            attribute_projections=attribute_projections,
            storage_binding=storage_binding,
            collective_certificates=collective_certificates,
            execution_evidence=execution_evidence,
            surface_source=surface_source,
        )
        carrier = _prepare_result_carrier(inputs)
        verdicts = _require_result_verdicts(inputs)
        organization = _prepare_result_organization(inputs)
        identity = _prepare_result_identity(inputs, organization)
        self.mesh = inputs.mesh
        self.geometry = inputs.geometry
        self.coordinate_contract = inputs.coordinate_contract
        self.boundary = inputs.boundary
        self.patches = organization.patches
        self.zones = organization.zones
        self.labels = organization.labels
        self.attributes = organization.attributes
        self.associations = organization.associations
        self.audit = verdicts.audit
        self.quality = verdicts.quality
        self.compliance = verdicts.compliance
        self.certification = inputs.certification
        self.region_evidence = inputs.region_evidence
        self.region_boundary_evidence = organization.boundary_regions
        self.collective_evidence = inputs.collective_evidence
        self.trace = verdicts.trace
        self.adapter_reports = organization.adapter_reports
        self.provider = inputs.provider
        self.runtime = verdicts.runtime
        self.derivative_mode = inputs.derivative_mode
        self.provenance = inputs.provenance
        self.execution_evidence = inputs.execution_evidence
        self.surface_source = inputs.surface_source
        self.collective_certificates = carrier.collective_certificates
        self.storage_binding = carrier.storage_binding
        self.result_id = identity.result_id
        self.stored_content_id = (
            inputs.collective_evidence.stored_content_id
            if isinstance(inputs.collective_evidence, CollectiveMeshEvidence)
            else None
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CellMeshingResult], value: CellMeshingResult, /
    ) -> CellMeshingResult:
        """Reconstruct a result using retained scientific fields and projections.

        Organization projections retain their global array bindings. The
        publication constructor revalidates the fields and regenerates result_id.
        """
        if type(value) is not cls:
            raise TypeError(
                "CellMeshingResult.stored_fields must have the reconstruction type; "
                f"got {type(value).__name__} for {cls.__name__}"
            )
        stored_content_id = (
            value.collective_evidence.stored_content_id
            if isinstance(value.collective_evidence, CollectiveMeshEvidence)
            else None
        )
        if not tree_content_equal(stored_content_id, value.stored_content_id):
            raise ValueError(
                "CellMeshingResult.stored_content_id must match its retained collective storage binding; got changed child content"
            )
        # Lazy: retained projection recovery reaches collective owners and _lineage.
        import phydrax.meshing._organization as _dependency_phydrax_meshing__organization

        scopes, attributes = (
            _dependency_phydrax_meshing__organization.retained_mesh_organization_projections(
                value
            )
        )
        rebuilt = cls(
            value.mesh,
            value.geometry,
            value.coordinate_contract,
            value.audit,
            value.quality,
            value.compliance,
            value.trace,
            value.provider,
            value.runtime,
            value.derivative_mode,
            value.provenance,
            boundary=value.boundary,
            patches=value.patches,
            zones=value.zones,
            labels=value.labels,
            attributes=value.attributes,
            associations=value.associations,
            surface_source=value.surface_source,
            adapter_reports=value.adapter_reports,
            certification=value.certification,
            region_evidence=value.region_evidence,
            region_boundary_evidence=value.region_boundary_evidence,
            collective_evidence=value.collective_evidence,
            storage_binding=value.storage_binding,
            collective_certificates=value.collective_certificates,
            execution_evidence=value.execution_evidence,
            scope_projections=scopes,
            attribute_projections=attributes,
        )
        if rebuilt.result_id != value.result_id:
            raise ValueError(
                "CellMeshingResult.result_id must match regenerated publication; "
                f"got {value.result_id!r} instead of {rebuilt.result_id!r}"
            )
        return rebuilt

    @checked
    def with_execution_evidence(
        self, record: NativeExecutionRecord, /
    ) -> CellMeshingResult:
        """Attach a validated execution record without rebuilding scientific fields.

        The record is excluded from result_id. Reconstruct only this root from its
        direct children so shared mesh, geometry and evidence retain object identity
        and traced geometry is not fingerprinted again."""
        record.require_valid()
        target = (jax.tree_util.GetAttrKey("execution_evidence"),)
        children, structure = jax.tree_util.tree_flatten_with_path(
            self, is_leaf=lambda value: value is not self
        )
        if sum(path == target for path, _ in children) != 1:
            raise RuntimeError(
                "result.execution_evidence must have one root PyTree slot; got missing or duplicate slots"
            )
        return jax.tree_util.tree_unflatten(
            structure,
            [record if path == target else value for path, value in children],
        )


def _collective_certificate_ids(
    certificates: tuple[
        global_embedding_owner.GlobalEmbeddingCertificate,
        domain_coverage_owner.DomainCoverageCertificate,
    ]
    | None,
    /,
) -> tuple[str, str] | None:
    return (
        None
        if certificates is None
        else (certificates[0].certificate_id, certificates[1].certificate_id)
    )


def _require_collective_certificates(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    certificates: tuple[
        global_embedding_owner.GlobalEmbeddingCertificate,
        domain_coverage_owner.DomainCoverageCertificate,
    ],
    /,
) -> None:
    """Admit only target-bound, wholly certified collective embedding and coverage."""
    if mesh.storage is None:
        raise ValueError(
            "result.collective_certificates must belong to owner-local storage; got a dense serial mesh"
        )
    if (
        type(certificates) is not tuple
        or len(certificates) != 2
        or not isinstance(
            certificates[0],
            global_embedding_owner.GlobalEmbeddingCertificate,
        )
        or not isinstance(
            certificates[1],
            domain_coverage_owner.DomainCoverageCertificate,
        )
    ):
        raise TypeError(
            "result.collective_certificates must be (GlobalEmbeddingCertificate, DomainCoverageCertificate); got an invalid certificate pair"
        )
    embedding, coverage = certificates
    embedding.binding.require(mesh, geometry)
    coverage.binding.require(mesh, geometry)
    if (
        embedding.status != "certified"
        or coverage.status != "certified"
        or coverage.embedding_certificate_id != embedding.certificate_id
        or mesh.storage.evidence_id not in coverage.premise_certificate_ids
    ):
        raise ValueError(
            "result.collective_certificates must certify and bind this owner-local partition; got failed status or mismatched evidence"
        )
    if any(
        value is None
        for value in (
            *coverage.achieved_region_measures,
            *coverage.achieved_region_measure_bounds,
            *coverage.integration_error_bounds,
        )
    ):
        raise ValueError(
            "result.collective_certificates.coverage must retain required region integrals; got absent measures or bounds"
        )


@checked
def require_original_meshing_source(
    result: CellMeshingResult,
    /,
) -> (
    CellMeshingResult
    | phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence
):
    """Validate the execution chain and resolve its scientific source."""

    # Lazy: _result -> _initial_certification -> _canonical -> _lineage import cycle.
    import phydrax.meshing._initial_certification

    active: set[int] = set()
    resolved: dict[
        int,
        CellMeshingResult
        | phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence,
    ] = {}

    def resolve(
        current: CellMeshingResult,
    ) -> (
        CellMeshingResult
        | phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence
    ):
        identity = id(current)
        if identity in active:
            raise ValueError(
                "meshing_source.predecessor must be acyclic; got a repeated active source"
            )
        if identity in resolved:
            return resolved[identity]
        active.add(identity)
        try:
            evidence = current.collective_evidence
            match evidence:
                case phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence():
                    evidence.require_passed()
                    storage = current.mesh.storage
                    if (
                        storage is None
                        or storage.evidence_id != evidence.evidence_id
                        or current.mesh.mesh_id != evidence.mesh_id
                        or _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                            current.geometry
                        )
                        != evidence.coordinate_geometry_id
                        or current.compliance.specification_id
                        != evidence.specification.specification_id
                    ):
                        raise ValueError(
                            "meshing_source.collective_evidence must bind initial storage and specification; got mismatched source evidence"
                        )
                    owner: (
                        CellMeshingResult
                        | phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence
                    ) = evidence
                case CollectiveMeshEvidence():
                    evidence.require_passed()
                    predecessor = evidence.source
                    storage = current.mesh.storage
                    if (
                        evidence.mesh_id != current.mesh.mesh_id
                        or evidence.topology_id != current.mesh.topology_id
                        or evidence.geometry_id != current.mesh.geometry_id
                        or evidence.coordinate_geometry_id
                        != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                            current.geometry
                        )
                        or storage is None
                        or evidence.evidence_id != storage.evidence_id
                        or evidence.partition_count != storage.partition_count
                        or evidence.source_mesh_id != predecessor.mesh.mesh_id
                    ):
                        raise ValueError(
                            "meshing_source.collective_evidence must bind predecessor topology, geometry and storage; got mismatched source evidence"
                        )
                    predecessor_owner = resolve(predecessor)
                    uniform = evidence.uniform_refinement
                    owner = (
                        predecessor_owner if uniform is None else resolve(uniform.source)
                    )
                    audit_id = (
                        owner.evidence_id
                        if isinstance(
                            owner,
                            phydrax.meshing._initial_certification.InitialCollectiveMeshEvidence,
                        )
                        else owner.audit.report_id
                    )
                    if evidence.source_audit_id != audit_id:
                        raise ValueError(
                            "meshing_source.collective_evidence.source_audit_id must bind the resolved source; got mismatched audit identity"
                        )
                case None:
                    certification = current.certification
                    current.geometry.resolve(current.mesh)
                    current.audit.require_passed()
                    current.audit.validity.require_bound(
                        current.geometry, mesh=current.mesh
                    )
                    _require_bound_evidence(
                        current.mesh,
                        current.geometry,
                        current.audit,
                        current.trace,
                        current.runtime,
                        certification,
                    )
                    if (
                        current.mesh.storage is not None
                        or current.audit.mesh_id != current.mesh.mesh_id
                        or current.audit.geometry_id
                        != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                            current.geometry
                        )
                        or current.audit.geometry_layout_id
                        != current.geometry.geometry_layout_id
                    ):
                        raise ValueError(
                            "meshing_source.audit must bind the dense mesh and coordinate map; got mismatched evidence"
                        )
                    if certification is not None:
                        certification.request.validate_source_integrity()
                        if (
                            certification.mesh_id != current.mesh.mesh_id
                            or certification.topology_id != current.mesh.topology_id
                            or certification.geometry_id
                            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                                current.geometry
                            )
                            or certification.geometry_layout_id
                            != current.geometry.geometry_layout_id
                            or certification.audit_report_id != current.audit.report_id
                        ):
                            raise ValueError(
                                "meshing_source.certification must bind the mesh/map/audit; got mismatched evidence"
                            )
                    owner = current
                case _:
                    raise TypeError(
                        "meshing_source.collective_evidence must be a supported collective proof; got an unknown theorem kind"
                    )
            resolved[identity] = owner
            return owner
        finally:
            active.remove(identity)

    return resolve(result)


@checked
def prepare_meshing_source_embedding(
    source: CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[global_embedding_owner.GlobalEmbeddingCertificate, str]:
    """Prepare the embedding certificate of an audited serial source."""

    certification = source.certification
    embedding = (
        global_embedding_owner.certify_global_embedding(
            source.mesh, source.geometry, source.audit.validity, budget=budget
        )
        if certification is None or certification.embedding is None
        else certification.embedding
    )
    embedding.binding.require(source.mesh, source.geometry)
    if (
        embedding.status != "certified"
        or embedding.validity_certificate_id != source.audit.validity.certificate_id
    ):
        raise ValueError(
            "meshing_source.embedding must certify the audited coordinate map; got "
            f"{embedding.status}"
        )
    premise_id = (
        certification.report_id
        if certification is not None
        else canonical_fingerprint(
            {
                "kind": "audited-meshing-source-embedding",
                "mesh": source.mesh.mesh_id,
                "geometry": _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                    source.geometry
                ),
                "audit": source.audit.report_id,
                "embedding": embedding.certificate_id,
            }
        )
    )
    return embedding, premise_id


@final
class CollectiveMeshIdentity(StrictModule, NonTrainableState):
    """Partition-independent topology, lineage, and hierarchy identities."""

    target_topology_id: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)


@final
class PartitionedMeshAdaptationResult(StrictModule, NonTrainableState):
    """Owner-local results and the collective identity of their global adaptation."""

    local_results: tuple[MeshAdaptationResult, ...]
    identity: CollectiveMeshIdentity


__all__ = [
    "CellMeshingResult",
    "CollectiveMeshEvidence",
    "CollectiveMeshIdentity",
    "MeshingComplianceReport",
    "MeshingRuntimeInfo",
    "PartitionedMeshAdaptationResult",
]
