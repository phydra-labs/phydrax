#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule


class TreeTopologyStatus(IntEnum):
    SUCCESS = 0
    EMPTY = 1
    INVALID_PARENT = 2
    ROOT_COUNT = 3
    CYCLE_OR_DISCONNECTED = 4
    CHILD_CAPACITY_EXCEEDED = 5
    TIP_MISMATCH = 6


class TreeTopologyEvidence(StrictModule):
    """Array-valued structural evidence for a rooted numeric topology."""

    node_count: Array
    edge_count: Array
    tip_count: Array
    maximum_out_degree: Array
    unique_root: Array
    parent_indices_in_range: Array
    acyclic: Array
    connected: Array
    traversal_complete: Array
    child_capacity_sufficient: Array


class TreeTopology(StrictModule):
    """One rooted tree encoded entirely by numeric, nondifferentiable arrays.

    ``parent_indices[root_index] == -1``. Every other parent is a node index.
    Children are padded with ``-1`` and selected by ``child_mask``. Traversals
    contain every node exactly once, with descendants preceding ancestors in
    ``postorder`` and ancestors preceding descendants in ``preorder``.
    """

    parent_indices: Array
    child_indices: Array
    child_mask: Array
    postorder: Array
    preorder: Array
    tip_indices: Array
    internal_indices: Array
    root_index: Array
    valid: Array
    status: Array
    evidence: TreeTopologyEvidence
    node_count: int = eqx.field(static=True)
    tip_count: int = eqx.field(static=True)
    internal_count: int = eqx.field(static=True)
    child_capacity: int = eqx.field(static=True)


class TreeTopologyBatch(StrictModule):
    """Fixed-capacity batch of rooted numeric tree topologies."""

    parent_indices: Array
    child_indices: Array
    child_mask: Array
    postorder: Array
    preorder: Array
    tip_indices: Array
    tip_mask: Array
    internal_indices: Array
    internal_mask: Array
    root_indices: Array
    valid: Array
    status: Array
    evidence: TreeTopologyEvidence
    batch_size: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    tip_capacity: int = eqx.field(static=True)
    internal_capacity: int = eqx.field(static=True)
    child_capacity: int = eqx.field(static=True)


def _invalid_topology(
    parent: np.ndarray,
    child_capacity: int,
    status: TreeTopologyStatus,
    *,
    maximum_out_degree: int = 0,
    parent_in_range: bool = False,
    unique_root: bool = False,
    acyclic: bool = False,
    connected: bool = False,
) -> TreeTopology:
    count = int(parent.size)
    safe_count = max(count, 1)
    safe_parent = parent if count else np.asarray([-1], dtype=np.int32)
    order = np.arange(safe_count, dtype=np.int32)
    children = np.full((safe_count, child_capacity), -1, dtype=np.int32)
    mask = np.zeros((safe_count, child_capacity), dtype=bool)
    evidence = TreeTopologyEvidence(
        node_count=jnp.asarray(count, dtype=jnp.int32),
        edge_count=jnp.asarray(max(count - 1, 0), dtype=jnp.int32),
        tip_count=jnp.asarray(0, dtype=jnp.int32),
        maximum_out_degree=jnp.asarray(maximum_out_degree, dtype=jnp.int32),
        unique_root=jnp.asarray(unique_root),
        parent_indices_in_range=jnp.asarray(parent_in_range),
        acyclic=jnp.asarray(acyclic),
        connected=jnp.asarray(connected),
        traversal_complete=jnp.asarray(False),
        child_capacity_sufficient=jnp.asarray(maximum_out_degree <= child_capacity),
    )
    return TreeTopology(
        parent_indices=jnp.asarray(safe_parent, dtype=jnp.int32),
        child_indices=jnp.asarray(children),
        child_mask=jnp.asarray(mask),
        postorder=jnp.asarray(order),
        preorder=jnp.asarray(order),
        tip_indices=jnp.zeros((0,), dtype=jnp.int32),
        internal_indices=jnp.zeros((0,), dtype=jnp.int32),
        root_index=jnp.asarray(0, dtype=jnp.int32),
        valid=jnp.asarray(False),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence=evidence,
        node_count=safe_count,
        tip_count=0,
        internal_count=0,
        child_capacity=child_capacity,
    )


def tree_topology(
    parent_indices: ArrayLike,
    /,
    *,
    child_capacity: int | None = None,
    tip_indices: ArrayLike | None = None,
) -> TreeTopology:
    """Validate and lower one rooted parent array to bounded traversals.

    Structural failures are represented by ``valid=False`` and ``status``.
    An insufficient explicit child capacity is therefore observable and never
    truncates a polytomy.
    """

    parent = np.asarray(parent_indices, dtype=np.int32)
    if parent.ndim != 1:
        raise ValueError("parent_indices must be a rank-one numeric array.")
    count = int(parent.size)
    if child_capacity is not None and int(child_capacity) < 0:
        raise ValueError("child_capacity must be nonnegative.")
    if count == 0:
        capacity = 0 if child_capacity is None else int(child_capacity)
        return _invalid_topology(parent, capacity, TreeTopologyStatus.EMPTY)

    roots = np.flatnonzero(parent == -1)
    parent_in_range = bool(np.all((parent == -1) | ((parent >= 0) & (parent < count))))
    if not parent_in_range:
        capacity = 0 if child_capacity is None else int(child_capacity)
        return _invalid_topology(
            parent,
            capacity,
            TreeTopologyStatus.INVALID_PARENT,
            parent_in_range=False,
            unique_root=roots.size == 1,
        )
    if roots.size != 1:
        capacity = 0 if child_capacity is None else int(child_capacity)
        return _invalid_topology(
            parent,
            capacity,
            TreeTopologyStatus.ROOT_COUNT,
            parent_in_range=True,
            unique_root=False,
        )

    root = int(roots[0])
    child_lists: list[list[int]] = [[] for _ in range(count)]
    for node, parent_node in enumerate(parent.tolist()):
        if node != root:
            child_lists[parent_node].append(node)
    maximum_out_degree = max(len(children) for children in child_lists)
    capacity = maximum_out_degree if child_capacity is None else int(child_capacity)
    if maximum_out_degree > capacity:
        return _invalid_topology(
            parent,
            capacity,
            TreeTopologyStatus.CHILD_CAPACITY_EXCEEDED,
            maximum_out_degree=maximum_out_degree,
            parent_in_range=True,
            unique_root=True,
        )

    preorder_values: list[int] = []
    postorder_values: list[int] = []
    color = np.zeros((count,), dtype=np.int8)

    def visit(node: int) -> bool:
        if color[node] == 1:
            return False
        if color[node] == 2:
            return True
        color[node] = 1
        preorder_values.append(node)
        for child in child_lists[node]:
            if not visit(child):
                return False
        color[node] = 2
        postorder_values.append(node)
        return True

    acyclic = visit(root)
    connected = bool(np.all(color == 2))
    if not acyclic or not connected:
        return _invalid_topology(
            parent,
            capacity,
            TreeTopologyStatus.CYCLE_OR_DISCONNECTED,
            maximum_out_degree=maximum_out_degree,
            parent_in_range=True,
            unique_root=True,
            acyclic=acyclic,
            connected=connected,
        )

    children = np.full((count, capacity), -1, dtype=np.int32)
    child_mask = np.zeros((count, capacity), dtype=bool)
    for node, values in enumerate(child_lists):
        if values:
            children[node, : len(values)] = values
            child_mask[node, : len(values)] = True
    inferred_tips = np.asarray(
        [node for node, values in enumerate(child_lists) if not values], dtype=np.int32
    )
    if tip_indices is not None:
        supplied_tips = np.asarray(tip_indices, dtype=np.int32)
        if supplied_tips.ndim != 1:
            raise ValueError("tip_indices must be rank one.")
        if not np.array_equal(np.sort(supplied_tips), inferred_tips):
            result = _invalid_topology(
                parent,
                capacity,
                TreeTopologyStatus.TIP_MISMATCH,
                maximum_out_degree=maximum_out_degree,
                parent_in_range=True,
                unique_root=True,
                acyclic=True,
                connected=True,
            )
            return result
        tips = supplied_tips
    else:
        tips = inferred_tips
    internals = np.asarray(
        [node for node, values in enumerate(child_lists) if values], dtype=np.int32
    )
    evidence = TreeTopologyEvidence(
        node_count=jnp.asarray(count, dtype=jnp.int32),
        edge_count=jnp.asarray(count - 1, dtype=jnp.int32),
        tip_count=jnp.asarray(tips.size, dtype=jnp.int32),
        maximum_out_degree=jnp.asarray(maximum_out_degree, dtype=jnp.int32),
        unique_root=jnp.asarray(True),
        parent_indices_in_range=jnp.asarray(True),
        acyclic=jnp.asarray(True),
        connected=jnp.asarray(True),
        traversal_complete=jnp.asarray(True),
        child_capacity_sufficient=jnp.asarray(True),
    )
    return TreeTopology(
        parent_indices=jnp.asarray(parent),
        child_indices=jnp.asarray(children),
        child_mask=jnp.asarray(child_mask),
        postorder=jnp.asarray(postorder_values, dtype=jnp.int32),
        preorder=jnp.asarray(preorder_values, dtype=jnp.int32),
        tip_indices=jnp.asarray(tips),
        internal_indices=jnp.asarray(internals),
        root_index=jnp.asarray(root, dtype=jnp.int32),
        valid=jnp.asarray(True),
        status=jnp.asarray(int(TreeTopologyStatus.SUCCESS), dtype=jnp.int32),
        evidence=evidence,
        node_count=count,
        tip_count=int(tips.size),
        internal_count=int(internals.size),
        child_capacity=capacity,
    )


def tree_topology_batch(
    parent_indices: ArrayLike,
    /,
    *,
    child_capacity: int | None = None,
    tip_capacity: int | None = None,
) -> TreeTopologyBatch:
    """Lower a rank-two parent array to one fixed-capacity topology batch."""

    parents = np.asarray(parent_indices, dtype=np.int32)
    if parents.ndim != 2:
        raise ValueError("Batched parent_indices must have shape (batch, nodes).")
    batch_size, node_count = parents.shape
    inferred_child_capacity = 0
    if child_capacity is None:
        for parent in parents:
            if parent.size:
                valid_parent = parent[(parent >= 0) & (parent < node_count)]
                if valid_parent.size:
                    inferred_child_capacity = max(
                        inferred_child_capacity,
                        int(np.max(np.bincount(valid_parent, minlength=node_count))),
                    )
    capacity = inferred_child_capacity if child_capacity is None else int(child_capacity)
    lowered = [tree_topology(parent, child_capacity=capacity) for parent in parents]
    maximum_tips = max((tree.tip_count for tree in lowered), default=0)
    resolved_tip_capacity = maximum_tips if tip_capacity is None else int(tip_capacity)
    if resolved_tip_capacity < 0:
        raise ValueError("tip_capacity must be nonnegative.")
    internal_capacity = node_count

    tips = np.full((batch_size, resolved_tip_capacity), -1, dtype=np.int32)
    tip_mask = np.zeros_like(tips, dtype=bool)
    internals = np.full((batch_size, internal_capacity), -1, dtype=np.int32)
    internal_mask = np.zeros_like(internals, dtype=bool)
    valid = []
    status = []
    for batch_index, tree in enumerate(lowered):
        tree_valid = bool(np.asarray(tree.valid))
        if tree.tip_count > resolved_tip_capacity:
            tree_valid = False
            tree_status = int(TreeTopologyStatus.CHILD_CAPACITY_EXCEEDED)
        else:
            tree_status = int(np.asarray(tree.status))
            if tree.tip_count:
                tips[batch_index, : tree.tip_count] = np.asarray(tree.tip_indices)
                tip_mask[batch_index, : tree.tip_count] = True
            if tree.internal_count:
                internals[batch_index, : tree.internal_count] = np.asarray(
                    tree.internal_indices
                )
                internal_mask[batch_index, : tree.internal_count] = True
        valid.append(tree_valid)
        status.append(tree_status)

    evidence = TreeTopologyEvidence(
        node_count=jnp.asarray([tree.node_count for tree in lowered], dtype=jnp.int32),
        edge_count=jnp.asarray(
            [int(np.asarray(tree.evidence.edge_count)) for tree in lowered],
            dtype=jnp.int32,
        ),
        tip_count=jnp.asarray([tree.tip_count for tree in lowered], dtype=jnp.int32),
        maximum_out_degree=jnp.asarray(
            [int(np.asarray(tree.evidence.maximum_out_degree)) for tree in lowered],
            dtype=jnp.int32,
        ),
        unique_root=jnp.asarray(
            [bool(np.asarray(tree.evidence.unique_root)) for tree in lowered]
        ),
        parent_indices_in_range=jnp.asarray(
            [bool(np.asarray(tree.evidence.parent_indices_in_range)) for tree in lowered]
        ),
        acyclic=jnp.asarray(
            [bool(np.asarray(tree.evidence.acyclic)) for tree in lowered]
        ),
        connected=jnp.asarray(
            [bool(np.asarray(tree.evidence.connected)) for tree in lowered]
        ),
        traversal_complete=jnp.asarray(
            [bool(np.asarray(tree.evidence.traversal_complete)) for tree in lowered]
        ),
        child_capacity_sufficient=jnp.asarray(
            [
                bool(np.asarray(tree.evidence.child_capacity_sufficient))
                and tree.tip_count <= resolved_tip_capacity
                for tree in lowered
            ]
        ),
    )
    return TreeTopologyBatch(
        parent_indices=jnp.asarray(parents),
        child_indices=jnp.stack([tree.child_indices for tree in lowered], axis=0),
        child_mask=jnp.stack([tree.child_mask for tree in lowered], axis=0),
        postorder=jnp.stack([tree.postorder for tree in lowered], axis=0),
        preorder=jnp.stack([tree.preorder for tree in lowered], axis=0),
        tip_indices=jnp.asarray(tips),
        tip_mask=jnp.asarray(tip_mask),
        internal_indices=jnp.asarray(internals),
        internal_mask=jnp.asarray(internal_mask),
        root_indices=jnp.stack([tree.root_index for tree in lowered]),
        valid=jnp.asarray(valid),
        status=jnp.asarray(status, dtype=jnp.int32),
        evidence=evidence,
        batch_size=batch_size,
        node_count=node_count,
        tip_capacity=resolved_tip_capacity,
        internal_capacity=internal_capacity,
        child_capacity=capacity,
    )


__all__ = [
    "TreeTopology",
    "TreeTopologyBatch",
    "TreeTopologyEvidence",
    "TreeTopologyStatus",
    "tree_topology",
    "tree_topology_batch",
]
