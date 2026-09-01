#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class TaxonomyStatus(IntEnum):
    SUCCESS = 0
    VERSION_MISMATCH = 1
    UNKNOWN_TAXON = 2
    DELETED_TAXON = 3
    INVALID_TREE = 4
    CAPACITY_EXCEEDED = 5
    EMPTY_INPUT = 6


def _tree_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "supplied taxonomy-tree validation",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.GRAPH,
        conditioning_statement=(
            "Taxon nodes, parent-to-child routes, merged IDs, deleted IDs, and a topological "
            "order are supplied by one declared database version."
        ),
        truncation_statement="Every valid node and route is checked; none is truncated.",
        capacity_semantics=(
            "Node, edge, merged-ID, and deleted-ID capacities are explicit array shapes."
        ),
        assumptions=("Merged IDs point directly to active IDs in the same release.",),
        nondifferentiable_outputs=(
            "relation",
            "taxon_ids",
            "rank_codes",
            "merged_taxon_ids",
            "deleted_taxon_ids",
            "status",
        ),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _resolution_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "versioned taxonomy-ID resolution",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement="Query IDs and taxonomy data identify the exact same release.",
        truncation_statement="All query IDs are resolved or receive an explicit failure status.",
        capacity_semantics="Query capacity is fixed by the query array shape.",
        assumptions=("Merged IDs were validated to point directly to active IDs.",),
        nondifferentiable_outputs=("taxon_indices", "resolved_taxon_ids", "status"),
        input_dtype="int32",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _lineage_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact taxonomy-lineage tracing",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.ARRAY,
        conditioning_statement="Input indices belong to one validated rooted taxonomy tree.",
        truncation_statement=(
            "A lineage longer than max_depth fails and is never returned as a truncated lineage."
        ),
        capacity_semantics="Lineage width is the caller-declared max_depth.",
        assumptions=(),
        nondifferentiable_outputs=("lineage_indices", "lineage_valid", "status"),
        input_dtype="int32",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


class TaxonomyVersion(StrictModule, NonTrainableState):
    """Canonical identity of one taxonomy database release and content snapshot."""

    namespace: str = eqx.field(static=True)
    release: str = eqx.field(static=True)
    content_fingerprint: str = eqx.field(static=True)
    version_id: str = eqx.field(static=True)

    def __init__(self, namespace: str, release: str, content_fingerprint: str, /):
        namespace_ = str(namespace).strip()
        release_ = str(release).strip()
        content_ = str(content_fingerprint).strip()
        if not namespace_ or not release_ or not content_:
            raise ValueError(
                "Taxonomy namespace, release, and content fingerprint are required."
            )
        self.namespace = namespace_
        self.release = release_
        self.content_fingerprint = content_
        self.version_id = canonical_fingerprint(
            {
                "kind": "taxonomy-version",
                "namespace": namespace_,
                "release": release_,
                "content": content_,
            }
        )


class TaxonomyTree(StrictModule, NonTrainableState):
    """Versioned rooted taxonomy whose only graph topology is an EdgeRelation."""

    taxon_ids: Array
    rank_codes: Array
    node_valid: Array
    relation: EdgeRelation
    root_index: Array
    merged_taxon_ids: Array
    merged_into_ids: Array
    merged_valid: Array
    deleted_taxon_ids: Array
    deleted_valid: Array
    version: TaxonomyVersion
    method_contract: BioinformaticsMethodContract
    taxonomy_id: str = eqx.field(static=True)

    def __init__(
        self,
        taxon_ids: ArrayLike,
        rank_codes: ArrayLike,
        node_valid: ArrayLike,
        relation: EdgeRelation,
        root_index: ArrayLike,
        merged_taxon_ids: ArrayLike,
        merged_into_ids: ArrayLike,
        merged_valid: ArrayLike,
        deleted_taxon_ids: ArrayLike,
        deleted_valid: ArrayLike,
        version: TaxonomyVersion,
        method_contract: BioinformaticsMethodContract,
        taxonomy_id: str,
        /,
    ):
        ids = jnp.asarray(taxon_ids, dtype=jnp.int32)
        ranks = jnp.asarray(rank_codes, dtype=jnp.int32)
        active = jnp.asarray(node_valid, dtype=bool)
        root = jnp.asarray(root_index, dtype=jnp.int32).reshape(())
        merged = jnp.asarray(merged_taxon_ids, dtype=jnp.int32)
        merged_into = jnp.asarray(merged_into_ids, dtype=jnp.int32)
        merged_mask = jnp.asarray(merged_valid, dtype=bool)
        deleted = jnp.asarray(deleted_taxon_ids, dtype=jnp.int32)
        deleted_mask = jnp.asarray(deleted_valid, dtype=bool)
        if ids.ndim != 1 or ids.shape != ranks.shape or ids.shape != active.shape:
            raise ValueError("Taxonomy node arrays must be matching rank-1 arrays.")
        if relation.source_size != ids.size or relation.target_size != ids.size:
            raise ValueError("Taxonomy relation spaces must match node capacity.")
        if merged.ndim != 1 or merged_into.shape != merged.shape:
            raise ValueError("Merged source and target ID arrays must match.")
        if merged_mask.shape != merged.shape:
            raise ValueError("merged_valid must match merged-ID capacity.")
        if deleted.ndim != 1 or deleted_mask.shape != deleted.shape:
            raise ValueError("Deleted ID arrays must match.")
        if not taxonomy_id:
            raise ValueError("taxonomy_id must be non-empty.")
        self.taxon_ids = ids
        self.rank_codes = ranks
        self.node_valid = active
        self.relation = relation
        self.root_index = root
        self.merged_taxon_ids = merged
        self.merged_into_ids = merged_into
        self.merged_valid = merged_mask
        self.deleted_taxon_ids = deleted
        self.deleted_valid = deleted_mask
        self.version = version
        self.method_contract = method_contract
        self.taxonomy_id = taxonomy_id

    @property
    def capacity(self) -> int:
        return int(self.taxon_ids.shape[0])


class TaxonomyBuildResult(StrictModule):
    taxonomy: TaxonomyTree
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


class TaxonomyResolutionResult(StrictModule):
    taxon_indices: Array
    resolved_taxon_ids: Array
    was_merged: Array
    was_deleted: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    version_id: str = eqx.field(static=True)


class TaxonomyLineageResult(StrictModule):
    lineage_indices: Array
    lineage_taxon_ids: Array
    lineage_valid: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    taxonomy_id: str = eqx.field(static=True)


def build_taxonomy_tree(
    taxon_ids: ArrayLike,
    rank_codes: ArrayLike,
    parent_indices: ArrayLike,
    child_indices: ArrayLike,
    edge_valid: ArrayLike,
    root_index: ArrayLike,
    topological_order: ArrayLike,
    order_valid: ArrayLike,
    version: TaxonomyVersion,
    /,
    *,
    node_valid: ArrayLike | None = None,
    merged_taxon_ids: ArrayLike | None = None,
    merged_into_ids: ArrayLike | None = None,
    merged_valid: ArrayLike | None = None,
    deleted_taxon_ids: ArrayLike | None = None,
    deleted_valid: ArrayLike | None = None,
) -> TaxonomyBuildResult:
    """Validate and package a supplied versioned taxonomy and ID migration tables."""
    ids = jnp.asarray(taxon_ids, dtype=jnp.int32)
    ranks = jnp.asarray(rank_codes, dtype=jnp.int32)
    if ids.ndim != 1 or ranks.shape != ids.shape:
        raise ValueError("taxon_ids and rank_codes must be matching rank-1 arrays.")
    node_capacity = ids.shape[0]
    if node_capacity < 1:
        raise ValueError("Taxonomy node capacity must be positive.")
    active = (
        jnp.ones((node_capacity,), dtype=bool)
        if node_valid is None
        else jnp.asarray(node_valid, dtype=bool)
    )
    if active.shape != ids.shape:
        raise ValueError("node_valid must match taxonomy node capacity.")
    relation = EdgeRelation(
        parent_indices,
        child_indices,
        source_size=node_capacity,
        target_size=node_capacity,
        valid=edge_valid,
    )
    root = jnp.asarray(root_index, dtype=jnp.int32).reshape(())
    order = jnp.asarray(topological_order, dtype=jnp.int32)
    supplied_order = jnp.asarray(order_valid, dtype=bool)
    if order.shape != ids.shape or supplied_order.shape != ids.shape:
        raise ValueError("Taxonomy topological order must match node capacity.")

    merged = (
        jnp.empty((0,), dtype=jnp.int32)
        if merged_taxon_ids is None
        else jnp.asarray(merged_taxon_ids, dtype=jnp.int32)
    )
    merged_into = (
        jnp.empty((0,), dtype=jnp.int32)
        if merged_into_ids is None
        else jnp.asarray(merged_into_ids, dtype=jnp.int32)
    )
    if merged.ndim != 1 or merged_into.shape != merged.shape:
        raise ValueError("Merged ID arrays must be matching rank-1 arrays.")
    merged_mask = (
        jnp.ones(merged.shape, dtype=bool)
        if merged_valid is None
        else jnp.asarray(merged_valid, dtype=bool)
    )
    if merged_mask.shape != merged.shape:
        raise ValueError("merged_valid must match merged-ID capacity.")
    deleted = (
        jnp.empty((0,), dtype=jnp.int32)
        if deleted_taxon_ids is None
        else jnp.asarray(deleted_taxon_ids, dtype=jnp.int32)
    )
    if deleted.ndim != 1:
        raise ValueError("deleted_taxon_ids must be rank-1.")
    deleted_mask = (
        jnp.ones(deleted.shape, dtype=bool)
        if deleted_valid is None
        else jnp.asarray(deleted_valid, dtype=bool)
    )
    if deleted_mask.shape != deleted.shape:
        raise ValueError("deleted_valid must match deleted-ID capacity.")

    safe_root = jnp.clip(root, 0, max(node_capacity - 1, 0))
    root_ok = (root >= 0) & (root < node_capacity) & active[safe_root]
    safe_parent = jnp.where(relation.valid, relation.source_indices, 0)
    safe_child = jnp.where(relation.valid, relation.target_indices, 0)
    endpoint_ok = jnp.all((~relation.valid) | (active[safe_parent] & active[safe_child]))
    self_edge_count = jnp.sum(
        relation.valid & (safe_parent == safe_child), dtype=jnp.int32
    )
    in_degree = jax.ops.segment_sum(
        relation.valid.astype(jnp.int32), safe_child, node_capacity
    )
    parent_count_ok = jnp.all(
        (~active)
        | (jnp.arange(node_capacity, dtype=jnp.int32) == root)
        | (in_degree == 1)
    ) & (in_degree[safe_root] == 0)

    order_in_bounds = (order >= 0) & (order < node_capacity)
    safe_order = jnp.clip(order, 0, max(node_capacity - 1, 0))
    order_usable = supplied_order & order_in_bounds
    occurrence = jnp.sum(
        jax.nn.one_hot(safe_order, node_capacity, dtype=jnp.int32)
        * order_usable[:, None].astype(jnp.int32),
        axis=0,
        dtype=jnp.int32,
    )
    order_ok = jnp.all(occurrence == active.astype(jnp.int32)) & jnp.all(
        (~supplied_order) | order_in_bounds
    )
    positions = jnp.arange(node_capacity, dtype=jnp.int32)
    ranks_in_order = jnp.full((node_capacity,), node_capacity, dtype=jnp.int32)
    ranks_in_order = ranks_in_order.at[safe_order].min(
        jnp.where(order_usable, positions, node_capacity)
    )
    back_edge_count = jnp.sum(
        relation.valid & (ranks_in_order[safe_parent] >= ranks_in_order[safe_child]),
        dtype=jnp.int32,
    )

    active_same = (ids[:, None] == ids[None, :]) & active[:, None] & active[None, :]
    lower_nodes = jnp.arange(node_capacity)[:, None] > jnp.arange(node_capacity)[None, :]
    duplicate_active = jnp.sum(active_same & lower_nodes, dtype=jnp.int32)
    merged_same = (
        (merged[:, None] == merged[None, :]) & merged_mask[:, None] & merged_mask[None, :]
    )
    lower_merged = jnp.arange(merged.size)[:, None] > jnp.arange(merged.size)[None, :]
    duplicate_merged = jnp.sum(merged_same & lower_merged, dtype=jnp.int32)
    deleted_same = (
        (deleted[:, None] == deleted[None, :])
        & deleted_mask[:, None]
        & deleted_mask[None, :]
    )
    lower_deleted = jnp.arange(deleted.size)[:, None] > jnp.arange(deleted.size)[None, :]
    duplicate_deleted = jnp.sum(deleted_same & lower_deleted, dtype=jnp.int32)
    merged_target_active = jnp.any(
        (merged_into[:, None] == ids[None, :]) & active[None, :], axis=1
    )
    migration_conflict = jnp.any(
        merged_mask
        & (
            (~merged_target_active)
            | jnp.any((merged[:, None] == ids[None, :]) & active[None, :], axis=1)
            | jnp.any(
                (merged[:, None] == deleted[None, :]) & deleted_mask[None, :],
                axis=1,
            )
        )
    ) | jnp.any(
        deleted_mask
        & jnp.any((deleted[:, None] == ids[None, :]) & active[None, :], axis=1)
    )

    valid = (
        root_ok
        & endpoint_ok
        & parent_count_ok
        & order_ok
        & (self_edge_count == 0)
        & (back_edge_count == 0)
        & (duplicate_active == 0)
        & (duplicate_merged == 0)
        & (duplicate_deleted == 0)
        & (~migration_conflict)
    )
    contract = _tree_contract()
    identity = canonical_fingerprint(
        {
            "kind": "taxonomy-tree",
            "version": version.version_id,
            "arrays": array_tree_fingerprint(
                (
                    ids,
                    ranks,
                    active,
                    relation.source_indices,
                    relation.target_indices,
                    relation.valid,
                    root,
                    merged,
                    merged_into,
                    merged_mask,
                    deleted,
                    deleted_mask,
                )
            ),
        }
    )
    taxonomy = TaxonomyTree(
        ids,
        ranks,
        active,
        relation,
        root,
        merged,
        merged_into,
        merged_mask,
        deleted,
        deleted_mask,
        version,
        contract,
        identity,
    )
    status = jnp.where(
        valid,
        int(TaxonomyStatus.SUCCESS),
        int(TaxonomyStatus.INVALID_TREE),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        (
            jnp.sum(active, dtype=jnp.int32),
            jnp.sum(relation.valid, dtype=jnp.int32),
            self_edge_count,
            back_edge_count,
            duplicate_active + duplicate_merged + duplicate_deleted,
            jnp.asarray(migration_conflict, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return TaxonomyBuildResult(taxonomy, valid, status, evidence, contract)


def resolve_taxon_ids(
    taxonomy: TaxonomyTree,
    query_taxon_ids: ArrayLike,
    /,
    *,
    database_version: TaxonomyVersion,
) -> TaxonomyResolutionResult:
    """Resolve active, merged, deleted, and unknown IDs under exact version identity."""
    query = jnp.asarray(query_taxon_ids, dtype=jnp.int32)
    active_match = (query[..., None] == taxonomy.taxon_ids) & taxonomy.node_valid
    active_found = jnp.any(active_match, axis=-1)
    active_index = jnp.argmax(active_match.astype(jnp.int32), axis=-1)

    if taxonomy.merged_taxon_ids.size > 0:
        merged_match = (
            query[..., None] == taxonomy.merged_taxon_ids
        ) & taxonomy.merged_valid
        merged_found = jnp.any(merged_match, axis=-1)
        merged_slot = jnp.argmax(merged_match.astype(jnp.int32), axis=-1)
        merged_target = taxonomy.merged_into_ids[merged_slot]
        target_match = (
            merged_target[..., None] == taxonomy.taxon_ids
        ) & taxonomy.node_valid
        merged_index = jnp.argmax(target_match.astype(jnp.int32), axis=-1)
    else:
        merged_found = jnp.zeros(query.shape, dtype=bool)
        merged_index = jnp.zeros(query.shape, dtype=jnp.int32)

    if taxonomy.deleted_taxon_ids.size > 0:
        deleted = jnp.any(
            (query[..., None] == taxonomy.deleted_taxon_ids) & taxonomy.deleted_valid,
            axis=-1,
        )
    else:
        deleted = jnp.zeros(query.shape, dtype=bool)
    version_match = jnp.asarray(
        database_version.version_id == taxonomy.version.version_id, dtype=bool
    )
    resolved = (active_found | merged_found) & (~deleted) & version_match
    index = jnp.where(active_found, active_index, merged_index)
    index = jnp.where(resolved, index, -1).astype(jnp.int32)
    safe_index = jnp.clip(index, 0, max(taxonomy.capacity - 1, 0))
    resolved_id = jnp.where(resolved, taxonomy.taxon_ids[safe_index], -1)
    status = jnp.where(
        ~version_match,
        int(TaxonomyStatus.VERSION_MISMATCH),
        jnp.where(
            deleted,
            int(TaxonomyStatus.DELETED_TAXON),
            jnp.where(
                resolved,
                int(TaxonomyStatus.SUCCESS),
                int(TaxonomyStatus.UNKNOWN_TAXON),
            ),
        ),
    ).astype(jnp.int32)
    contract = _resolution_contract()
    evidence = jnp.asarray(
        (
            query.size,
            jnp.sum(resolved, dtype=jnp.int32),
            jnp.sum(resolved & merged_found, dtype=jnp.int32),
            jnp.sum(deleted, dtype=jnp.int32),
            jnp.sum((~resolved) & (~deleted) & version_match, dtype=jnp.int32),
            jnp.asarray(~version_match, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return TaxonomyResolutionResult(
        index,
        resolved_id,
        resolved & merged_found,
        deleted & version_match,
        resolved,
        status,
        evidence,
        contract,
        taxonomy.version.version_id,
    )


def trace_taxonomy_lineages(
    taxonomy: TaxonomyTree,
    taxon_indices: ArrayLike,
    /,
    *,
    max_depth: int,
) -> TaxonomyLineageResult:
    """Trace node-to-root lineages, failing any lineage that exceeds max_depth."""
    depth = int(max_depth)
    if depth < 1:
        raise ValueError("max_depth must be positive.")
    query = jnp.asarray(taxon_indices, dtype=jnp.int32)
    flat = query.reshape((-1,))
    capacity = taxonomy.capacity
    in_bounds = (flat >= 0) & (flat < capacity)
    safe_query = jnp.clip(flat, 0, max(capacity - 1, 0))
    query_valid = in_bounds & taxonomy.node_valid[safe_query]

    if taxonomy.relation.capacity > 0:
        safe_parent = jnp.where(
            taxonomy.relation.valid, taxonomy.relation.source_indices, 0
        )
        safe_child = jnp.where(
            taxonomy.relation.valid, taxonomy.relation.target_indices, 0
        )
        parent_match = (
            jnp.arange(capacity, dtype=jnp.int32)[:, None] == safe_child[None, :]
        ) & taxonomy.relation.valid[None, :]
        has_parent = jnp.any(parent_match, axis=1)
        parent_slot = jnp.argmax(parent_match.astype(jnp.int32), axis=1)
        parent_index = safe_parent[parent_slot]
    else:
        has_parent = jnp.zeros((capacity,), dtype=bool)
        parent_index = jnp.arange(capacity, dtype=jnp.int32)

    lineage = jnp.full((flat.size, depth), -1, dtype=jnp.int32)
    lineage_valid = jnp.zeros((flat.size, depth), dtype=bool)
    current = safe_query
    continuing = query_valid

    def body(slot: int, state: tuple[Array, Array, Array, Array]):
        indices, valid, node, active = state
        indices = indices.at[:, slot].set(jnp.where(active, node, -1))
        valid = valid.at[:, slot].set(active)
        next_active = active & has_parent[node]
        next_node = jnp.where(next_active, parent_index[node], node)
        return indices, valid, next_node, next_active

    lineage, lineage_valid, current, continuing = jax.lax.fori_loop(
        0, depth, body, (lineage, lineage_valid, current, continuing)
    )
    overflow = continuing
    per_query_valid = query_valid & (~overflow)
    lineage_valid = lineage_valid & per_query_valid[:, None]
    lineage = jnp.where(lineage_valid, lineage, -1)
    safe_lineage = jnp.clip(lineage, 0, max(capacity - 1, 0))
    lineage_ids = jnp.where(lineage_valid, taxonomy.taxon_ids[safe_lineage], -1)
    output_shape = query.shape + (depth,)
    lineage = lineage.reshape(output_shape)
    lineage_ids = lineage_ids.reshape(output_shape)
    lineage_valid = lineage_valid.reshape(output_shape)
    valid = per_query_valid.reshape(query.shape)
    status = jnp.where(
        overflow.reshape(query.shape),
        int(TaxonomyStatus.CAPACITY_EXCEEDED),
        jnp.where(
            valid,
            int(TaxonomyStatus.SUCCESS),
            int(TaxonomyStatus.UNKNOWN_TAXON),
        ),
    ).astype(jnp.int32)
    contract = _lineage_contract()
    evidence = jnp.asarray(
        (
            flat.size,
            jnp.sum(per_query_valid, dtype=jnp.int32),
            jnp.sum(overflow, dtype=jnp.int32),
            jnp.sum(~query_valid, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return TaxonomyLineageResult(
        lineage,
        lineage_ids,
        lineage_valid,
        valid,
        status,
        evidence,
        contract,
        taxonomy.taxonomy_id,
    )


__all__ = [
    "TaxonomyBuildResult",
    "TaxonomyLineageResult",
    "TaxonomyResolutionResult",
    "TaxonomyStatus",
    "TaxonomyTree",
    "TaxonomyVersion",
    "build_taxonomy_tree",
    "resolve_taxon_ids",
    "trace_taxonomy_lineages",
]
