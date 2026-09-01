#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


SAMPLE_NODE = 1
NULL_INDEX = -1


class TreeSequenceStatus(IntEnum):
    SUCCESS = 0
    POSITION_OUT_OF_BOUNDS = 1
    MULTIPLE_PARENTS = 2
    INVALID_TOPOLOGY = 3
    NONFINITE = 4


class NodeTable(StrictModule):
    flags: Array
    time: Array
    population: Array
    individual: Array

    def __init__(
        self,
        flags: ArrayLike,
        time: ArrayLike,
        /,
        *,
        population: ArrayLike | None = None,
        individual: ArrayLike | None = None,
    ):
        flags_ = jnp.asarray(flags)
        time_ = jnp.asarray(time)
        if flags_.ndim != 1 or time_.shape != flags_.shape:
            raise ValueError("Node flags and times must be equal-length vectors.")
        if not jnp.issubdtype(flags_.dtype, jnp.integer):
            raise TypeError("Node flags must contain integers.")
        if not jnp.issubdtype(time_.dtype, jnp.inexact):
            time_ = time_.astype(float)
        population_ = (
            jnp.full(flags_.shape, NULL_INDEX, dtype=jnp.int32)
            if population is None
            else jnp.asarray(population)
        )
        individual_ = (
            jnp.full(flags_.shape, NULL_INDEX, dtype=jnp.int32)
            if individual is None
            else jnp.asarray(individual)
        )
        if population_.shape != flags_.shape or individual_.shape != flags_.shape:
            raise ValueError(
                "Node population/individual must contain one value per node."
            )
        if not jnp.issubdtype(population_.dtype, jnp.integer) or not jnp.issubdtype(
            individual_.dtype, jnp.integer
        ):
            raise TypeError("Node population and individual IDs must be integers.")
        if not np.all(np.isfinite(np.asarray(time_))):
            raise ValueError("Node times must be finite.")
        self.flags = flags_.astype(jnp.int32)
        self.time = time_
        self.population = population_.astype(jnp.int32)
        self.individual = individual_.astype(jnp.int32)

    @property
    def count(self) -> int:
        return int(self.flags.shape[0])


class EdgeTable(StrictModule):
    left: Array
    right: Array
    parent: Array
    child: Array

    def __init__(
        self,
        left: ArrayLike,
        right: ArrayLike,
        parent: ArrayLike,
        child: ArrayLike,
        /,
    ):
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        parent_ = jnp.asarray(parent)
        child_ = jnp.asarray(child)
        shape = left_.shape
        if left_.ndim != 1 or any(
            value.shape != shape for value in (right_, parent_, child_)
        ):
            raise ValueError("Edge columns must be equal-length vectors.")
        if not jnp.issubdtype(parent_.dtype, jnp.integer) or not jnp.issubdtype(
            child_.dtype, jnp.integer
        ):
            raise TypeError("Edge parent and child columns must contain integers.")
        if not jnp.issubdtype(left_.dtype, jnp.inexact):
            left_ = left_.astype(float)
        if not jnp.issubdtype(right_.dtype, jnp.inexact):
            right_ = right_.astype(float)
        host_left = np.asarray(left_)
        host_right = np.asarray(right_)
        if (
            not np.all(np.isfinite(host_left))
            or not np.all(np.isfinite(host_right))
            or np.any(host_left < 0.0)
            or np.any(host_right <= host_left)
        ):
            raise ValueError(
                "Every edge must have a finite nonempty non-negative interval."
            )
        self.left = left_
        self.right = right_
        self.parent = parent_.astype(jnp.int32)
        self.child = child_.astype(jnp.int32)

    @property
    def count(self) -> int:
        return int(self.left.shape[0])


class SiteTable(StrictModule):
    position: Array
    ancestral_state: Array

    def __init__(self, position: ArrayLike, ancestral_state: ArrayLike, /):
        position_ = jnp.asarray(position)
        ancestral_ = jnp.asarray(ancestral_state)
        if position_.ndim != 1 or ancestral_.shape != position_.shape:
            raise ValueError(
                "Site positions and ancestral states must be equal-length vectors."
            )
        if not jnp.issubdtype(ancestral_.dtype, jnp.integer):
            raise TypeError("Ancestral states must be numeric integer allele codes.")
        if not jnp.issubdtype(position_.dtype, jnp.inexact):
            position_ = position_.astype(float)
        host = np.asarray(position_)
        if (
            not np.all(np.isfinite(host))
            or np.any(host < 0.0)
            or np.any(np.diff(host) <= 0.0)
        ):
            raise ValueError(
                "Site positions must be finite, non-negative, and increasing."
            )
        self.position = position_
        self.ancestral_state = ancestral_.astype(jnp.int32)

    @property
    def count(self) -> int:
        return int(self.position.shape[0])


class MutationTable(StrictModule):
    site: Array
    node: Array
    derived_state: Array
    parent: Array
    time: Array

    def __init__(
        self,
        site: ArrayLike,
        node: ArrayLike,
        derived_state: ArrayLike,
        /,
        *,
        parent: ArrayLike | None = None,
        time: ArrayLike | None = None,
    ):
        site_ = jnp.asarray(site)
        node_ = jnp.asarray(node)
        derived_ = jnp.asarray(derived_state)
        if site_.ndim != 1 or node_.shape != site_.shape or derived_.shape != site_.shape:
            raise ValueError("Mutation columns must be equal-length vectors.")
        if any(
            not jnp.issubdtype(value.dtype, jnp.integer)
            for value in (site_, node_, derived_)
        ):
            raise TypeError("Mutation site/node/state columns must contain integers.")
        parent_ = (
            jnp.full(site_.shape, NULL_INDEX, dtype=jnp.int32)
            if parent is None
            else jnp.asarray(parent)
        )
        time_ = (
            jnp.full(site_.shape, jnp.nan, dtype=float)
            if time is None
            else jnp.asarray(time)
        )
        if parent_.shape != site_.shape or time_.shape != site_.shape:
            raise ValueError("Mutation parent/time must contain one value per mutation.")
        if not jnp.issubdtype(parent_.dtype, jnp.integer):
            raise TypeError("Mutation parent must contain integers.")
        if not jnp.issubdtype(time_.dtype, jnp.inexact):
            time_ = time_.astype(float)
        host_time = np.asarray(time_)
        if np.any(np.isinf(host_time)):
            raise ValueError("Mutation times may be finite or NaN for unknown.")
        self.site = site_.astype(jnp.int32)
        self.node = node_.astype(jnp.int32)
        self.derived_state = derived_.astype(jnp.int32)
        self.parent = parent_.astype(jnp.int32)
        self.time = time_

    @property
    def count(self) -> int:
        return int(self.site.shape[0])


class TreeSequenceTables(StrictModule):
    nodes: NodeTable
    edges: EdgeTable
    sites: SiteTable
    mutations: MutationTable
    sequence_length: Array

    def __init__(
        self,
        nodes: NodeTable,
        edges: EdgeTable,
        sequence_length: float | ArrayLike,
        /,
        *,
        sites: SiteTable | None = None,
        mutations: MutationTable | None = None,
    ):
        if not isinstance(nodes, NodeTable) or not isinstance(edges, EdgeTable):
            raise TypeError("nodes and edges must be NodeTable and EdgeTable values.")
        sites_ = (
            SiteTable(jnp.zeros((0,), dtype=float), jnp.zeros((0,), dtype=jnp.int32))
            if sites is None
            else sites
        )
        mutations_ = (
            MutationTable(
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
            )
            if mutations is None
            else mutations
        )
        if not isinstance(sites_, SiteTable) or not isinstance(mutations_, MutationTable):
            raise TypeError("sites and mutations must be numeric table values.")
        length = jnp.asarray(sequence_length)
        if length.shape != () or jnp.iscomplexobj(length):
            raise ValueError("sequence_length must be a real scalar.")
        if not jnp.issubdtype(length.dtype, jnp.inexact):
            length = length.astype(float)
        host_length = float(np.asarray(length))
        parent = np.asarray(edges.parent)
        child = np.asarray(edges.child)
        if not np.isfinite(host_length) or host_length <= 0.0:
            raise ValueError("sequence_length must be finite and positive.")
        if np.any(np.asarray(edges.right) > host_length):
            raise ValueError("Edge intervals must lie within sequence_length.")
        if (
            np.any(parent < 0)
            or np.any(parent >= nodes.count)
            or np.any(child < 0)
            or np.any(child >= nodes.count)
        ):
            raise ValueError("Edge node indices are outside the node table.")
        if np.any(parent == child):
            raise ValueError("An edge cannot connect a node to itself.")
        if np.any(np.asarray(nodes.time)[parent] <= np.asarray(nodes.time)[child]):
            raise ValueError("Every parent must be older than its child.")
        if sites_.count and (np.any(np.asarray(sites_.position) >= host_length)):
            raise ValueError("Site positions must lie within sequence_length.")
        mutation_site = np.asarray(mutations_.site)
        mutation_node = np.asarray(mutations_.node)
        mutation_parent = np.asarray(mutations_.parent)
        if (
            np.any(mutation_site < 0)
            or np.any(mutation_site >= sites_.count)
            or np.any(mutation_node < 0)
            or np.any(mutation_node >= nodes.count)
            or np.any(mutation_parent >= mutations_.count)
        ):
            raise ValueError("Mutation indices refer outside their numeric tables.")
        self.nodes = nodes
        self.edges = edges
        self.sites = sites_
        self.mutations = mutations_
        self.sequence_length = length


class MarginalTree(StrictModule):
    parent: Array
    active_node: Array
    root_mask: Array
    active_edge: Array
    root_count: Array
    total_branch_length: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(TreeSequenceStatus.SUCCESS))


class TreeSequenceSummary(StrictModule):
    breakpoints: Array
    root_count: Array
    total_branch_length: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def tree_count(self) -> int:
        return int(self.root_count.shape[0])

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid & (self.status == int(TreeSequenceStatus.SUCCESS)))


def _tree_contract(method_name: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.GRAPH,
        conditioning_statement="Conditioned on validated numeric node, edge, site, and mutation tables.",
        truncation_statement="No active edge, node, root, site, or mutation is truncated.",
        capacity_semantics="Node and edge table lengths are the exact finite capacities.",
        assumptions=("Parent time is strictly greater than child time.",),
        nondifferentiable_outputs=(
            "parent",
            "active_node",
            "root_mask",
            "active_edge",
            "status",
            "valid",
        ),
    )


def marginal_tree(
    tables: TreeSequenceTables, position: float | ArrayLike, /
) -> MarginalTree:
    """Materialize a numeric marginal forest; multiple roots are first-class."""
    if not isinstance(tables, TreeSequenceTables):
        raise TypeError("tables must be TreeSequenceTables.")
    coordinate = jnp.asarray(position, dtype=tables.sequence_length.dtype)
    if coordinate.shape != ():
        raise ValueError("position must be scalar.")
    in_bounds = (
        jnp.isfinite(coordinate)
        & (coordinate >= 0.0)
        & (coordinate < tables.sequence_length)
    )
    active_edge = (
        (tables.edges.left <= coordinate) & (coordinate < tables.edges.right) & in_bounds
    )
    node_count = tables.nodes.count
    incoming = (
        jnp.zeros((node_count,), dtype=jnp.int32)
        .at[tables.edges.child]
        .add(active_edge.astype(jnp.int32))
    )
    parent = (
        jnp.zeros((node_count,), dtype=jnp.int32)
        .at[tables.edges.child]
        .max(jnp.where(active_edge, tables.edges.parent + 1, 0))
        - 1
    )
    multiple_parents = jnp.any(incoming > 1)
    incident = jnp.zeros((node_count,), dtype=bool)
    incident = incident.at[tables.edges.parent].max(active_edge)
    incident = incident.at[tables.edges.child].max(active_edge)
    samples = (tables.nodes.flags & SAMPLE_NODE) != 0
    active_node = incident | samples
    root_mask = active_node & (incoming == 0)
    root_count = jnp.sum(root_mask).astype(jnp.int32)
    branch_length = jnp.sum(
        jnp.where(
            active_edge,
            tables.nodes.time[tables.edges.parent]
            - tables.nodes.time[tables.edges.child],
            0.0,
        )
    )
    valid = in_bounds & ~multiple_parents & (root_count > 0) & jnp.isfinite(branch_length)
    status = jnp.where(
        ~in_bounds,
        int(TreeSequenceStatus.POSITION_OUT_OF_BOUNDS),
        jnp.where(
            multiple_parents,
            int(TreeSequenceStatus.MULTIPLE_PARENTS),
            jnp.where(
                root_count == 0,
                int(TreeSequenceStatus.INVALID_TOPOLOGY),
                jnp.where(
                    jnp.isfinite(branch_length),
                    int(TreeSequenceStatus.SUCCESS),
                    int(TreeSequenceStatus.NONFINITE),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return MarginalTree(
        parent,
        active_node,
        root_mask,
        active_edge,
        root_count,
        branch_length,
        valid,
        status,
        jnp.asarray((jnp.sum(active_edge), root_count), dtype=jnp.int32),
        _tree_contract("marginal-tree-materialization"),
    )


def summarize_tree_sequence(tables: TreeSequenceTables, /) -> TreeSequenceSummary:
    """Summarize every nonempty marginal interval including multiple-root forests."""
    if not isinstance(tables, TreeSequenceTables):
        raise TypeError("tables must be TreeSequenceTables.")
    breakpoints = np.unique(
        np.concatenate(
            (
                np.asarray((0.0, float(np.asarray(tables.sequence_length)))),
                np.asarray(tables.edges.left),
                np.asarray(tables.edges.right),
            )
        )
    )
    roots: list[Array] = []
    branch_lengths: list[Array] = []
    valid: list[Array] = []
    status: list[Array] = []
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        tree = marginal_tree(tables, 0.5 * (left + right))
        roots.append(tree.root_count)
        branch_lengths.append(tree.total_branch_length)
        valid.append(tree.valid)
        status.append(tree.status)
    root_values = jnp.stack(roots)
    branch_values = jnp.stack(branch_lengths)
    valid_values = jnp.stack(valid)
    status_values = jnp.stack(status)
    return TreeSequenceSummary(
        jnp.asarray(breakpoints),
        root_values,
        branch_values,
        valid_values,
        status_values,
        jnp.stack((root_values, valid_values.astype(jnp.int32)), axis=-1),
        _tree_contract("tree-sequence-numeric-summary"),
    )


__all__ = [
    "NULL_INDEX",
    "SAMPLE_NODE",
    "EdgeTable",
    "MarginalTree",
    "MutationTable",
    "NodeTable",
    "SiteTable",
    "TreeSequenceStatus",
    "TreeSequenceSummary",
    "TreeSequenceTables",
    "marginal_tree",
    "summarize_tree_sequence",
]
