#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.graph import EquivariantGraphConvolution, GraphIR
from phydrax.nn.layers import Linear

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class StructureModelStatus(IntEnum):
    """Array-valued macromolecular model status codes."""

    SUCCESS = 0
    EMPTY_SUPPORT = 1
    NONFINITE = 2


_STRUCTURE_CONTRACT = BioinformaticsMethodContract(
    "learned-macromolecular-structure-representation",
    MethodKind.LEARNED,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.STRUCTURED,
    conditioning_statement="Scalar channels use invariant distances; vector channels use relative positions.",
    truncation_statement="No nodes or supplied edges are truncated.",
    capacity_semantics="Node and edge capacities are explicit padded axes with observable masks.",
    assumptions=("Edges refer only to in-capacity nodes of the same padded case.",),
    nondifferentiable_outputs=("topology indices", "validity status"),
    input_dtype="real coordinates and features; integer topology",
    compute_dtype="feature dtype",
    output_dtype="feature dtype",
)


class MacromolecularBatch(StrictModule):
    """Padded numeric macromolecular graphs with dynamic topology leaves.

    Host strings, files, parsers, and third-party molecular objects are excluded.
    ``senders`` and ``receivers`` are case-local integer indices of shape
    ``(batch, edge_capacity)``. Invalid edge slots are masked and never routed.
    """

    record_ids: Array
    positions: Array
    node_features: Array
    node_mask: Array
    case_mask: Array
    senders: Array
    receivers: Array
    edge_mask: Array
    edge_features: Array | None
    spatial_dimension: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    node_capacity: int = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        record_ids: Array,
        positions: Array,
        node_features: Array,
        node_mask: Array,
        case_mask: Array,
        senders: Array,
        receivers: Array,
        edge_mask: Array,
        *,
        edge_features: Array | None = None,
    ):
        ids = jnp.asarray(record_ids)
        coordinates = jnp.asarray(positions)
        features = jnp.asarray(node_features)
        nodes = jnp.asarray(node_mask, dtype=bool)
        cases = jnp.asarray(case_mask, dtype=bool)
        source = jnp.asarray(senders)
        target = jnp.asarray(receivers)
        edges = jnp.asarray(edge_mask, dtype=bool)
        if ids.ndim != 1 or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError("record_ids must be a rank-one integer array.")
        if coordinates.ndim != 3 or not jnp.issubdtype(coordinates.dtype, jnp.floating):
            raise ValueError("positions must have shape (batch, nodes, dimension).")
        if features.ndim != 3 or not jnp.issubdtype(features.dtype, jnp.floating):
            raise ValueError("node_features must have shape (batch, nodes, channels).")
        if (
            coordinates.shape[:2] != features.shape[:2]
            or nodes.shape != coordinates.shape[:2]
        ):
            raise ValueError(
                "Position, feature, and node-mask leading shapes must agree."
            )
        batch, node_capacity = nodes.shape
        if ids.shape != (batch,) or cases.shape != (batch,):
            raise ValueError("record_ids and case_mask must match the batch axis.")
        if (
            source.ndim != 2
            or source.shape != target.shape
            or source.shape != edges.shape
        ):
            raise ValueError(
                "senders, receivers, and edge_mask must share shape (batch, edges)."
            )
        if source.shape[0] != batch:
            raise ValueError("Topology batch axis must match node arrays.")
        if not jnp.issubdtype(source.dtype, jnp.integer) or not jnp.issubdtype(
            target.dtype, jnp.integer
        ):
            raise TypeError("Topology indices must have integer dtype.")
        if int(coordinates.shape[-1]) <= 0 or int(features.shape[-1]) <= 0:
            raise ValueError("Spatial and feature dimensions must be positive.")
        active_nodes = nodes & cases[:, None]
        active_edges = edges & cases[:, None]
        safe_source = jnp.where(active_edges, source, 0)
        safe_target = jnp.where(active_edges, target, 0)
        invalid_index = active_edges & (
            (safe_source < 0)
            | (safe_source >= node_capacity)
            | (safe_target < 0)
            | (safe_target >= node_capacity)
        )
        safe_source = eqx.error_if(
            safe_source,
            jnp.any(invalid_index),
            "Active macromolecular edge index exceeds node capacity.",
        )
        safe_target = eqx.error_if(
            safe_target,
            jnp.any(invalid_index),
            "Active macromolecular edge index exceeds node capacity.",
        )
        referenced_padding = active_edges & (
            ~jnp.take_along_axis(active_nodes, safe_source, axis=1)
            | ~jnp.take_along_axis(active_nodes, safe_target, axis=1)
        )
        active_edges = eqx.error_if(
            active_edges,
            jnp.any(referenced_padding),
            "Active macromolecular edges cannot reference padded nodes.",
        )
        normalized_edge_features = None
        if edge_features is not None:
            normalized_edge_features = jnp.asarray(edge_features)
            if (
                normalized_edge_features.ndim < 2
                or normalized_edge_features.shape[:2] != edges.shape
            ):
                raise ValueError(
                    "edge_features must begin with the topology batch and edge axes."
                )
            if not jnp.issubdtype(normalized_edge_features.dtype, jnp.floating):
                raise TypeError("edge_features must have real floating dtype.")
        self.record_ids = ids
        self.positions = jnp.where(active_nodes[..., None], coordinates, 0.0)
        self.node_features = jnp.where(active_nodes[..., None], features, 0.0)
        self.node_mask = active_nodes
        self.case_mask = cases
        self.senders = jax.lax.stop_gradient(safe_source)
        self.receivers = jax.lax.stop_gradient(safe_target)
        self.edge_mask = active_edges
        self.edge_features = normalized_edge_features
        self.spatial_dimension = int(coordinates.shape[-1])
        self.feature_count = int(features.shape[-1])
        self.node_capacity = int(node_capacity)
        self.edge_capacity = int(edges.shape[1])

    def graph_ir(self, node_features: Array | None = None, /) -> GraphIR:
        """Lower the padded batch to the native sparse graph representation."""
        features = (
            self.node_features if node_features is None else jnp.asarray(node_features)
        )
        expected = self.node_mask.shape + (int(features.shape[-1]),)
        if features.ndim != 3 or features.shape != expected:
            raise ValueError(
                "Replacement node features must preserve batch and node axes."
            )
        batch = int(self.node_mask.shape[0])
        offsets = (
            jnp.arange(batch, dtype=self.senders.dtype)[:, None] * self.node_capacity
        )
        senders = (self.senders + offsets).reshape((-1,))
        receivers = (self.receivers + offsets).reshape((-1,))
        nodes = {
            "features": features.reshape((-1, int(features.shape[-1]))),
            "positions": self.positions.reshape((-1, self.spatial_dimension)),
        }
        edges = None
        if self.edge_features is not None:
            edges = {
                "features": self.edge_features.reshape((batch * self.edge_capacity, -1))
            }
        return GraphIR(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=jnp.full((batch,), self.node_capacity, dtype=jnp.int32),
            n_edge=jnp.full((batch,), self.edge_capacity, dtype=jnp.int32),
            node_mask=self.node_mask.reshape((-1,)),
            edge_mask=self.edge_mask.reshape((-1,)),
            graph_mask=self.case_mask,
        )


class StructureModelResult(StrictModule):
    """Invariant scalar and equivariant vector macromolecular representations."""

    node_embeddings: Array
    vector_features: Array
    coordinate_updates: Array
    pooled_embedding: Array
    node_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        node_embeddings: Array,
        vector_features: Array,
        coordinate_updates: Array,
        node_mask: Array,
    ):
        scalar = jnp.asarray(node_embeddings)
        vector = jnp.asarray(vector_features)
        updates = jnp.asarray(coordinate_updates)
        mask = jnp.asarray(node_mask, dtype=bool)
        if scalar.ndim != 3 or scalar.shape[:2] != mask.shape:
            raise ValueError("Node embeddings must have shape (batch, nodes, channels).")
        if vector.ndim != 4 or vector.shape[:2] != mask.shape:
            raise ValueError(
                "Vector features must have shape (batch, nodes, dimension, channels)."
            )
        if updates.shape != vector.shape[:3]:
            raise ValueError("Coordinate updates must match vector spatial axes.")
        scalar = jnp.where(mask[..., None], scalar, 0.0)
        vector = jnp.where(mask[..., None, None], vector, 0.0)
        updates = jnp.where(mask[..., None], updates, 0.0)
        counts = jnp.sum(mask, axis=1, dtype=jnp.int32)
        pooled = jnp.sum(scalar, axis=1) / jnp.maximum(counts, 1)[:, None].astype(
            scalar.dtype
        )
        finite = jnp.all(jnp.isfinite(scalar)) & jnp.all(jnp.isfinite(vector))
        support = jnp.any(mask)
        self.node_embeddings = scalar
        self.vector_features = vector
        self.coordinate_updates = updates
        self.pooled_embedding = pooled
        self.node_mask = mask
        self.valid = finite & support
        self.status = jnp.where(
            ~support,
            jnp.asarray(StructureModelStatus.EMPTY_SUPPORT, dtype=jnp.int32),
            jnp.where(
                finite,
                jnp.asarray(StructureModelStatus.SUCCESS, dtype=jnp.int32),
                jnp.asarray(StructureModelStatus.NONFINITE, dtype=jnp.int32),
            ),
        )
        self.evidence = jnp.stack(
            (jnp.sum(mask, dtype=jnp.int32), jnp.asarray(mask.size, dtype=jnp.int32))
        )
        self.method_contract = _STRUCTURE_CONTRACT


class EquivariantStructureEncoder(StrictModule):
    """Native SE(n)-equivariant message-passing wrapper for macromolecules."""

    input_projection: Linear
    convolutions: tuple[EquivariantGraphConvolution, ...]
    scalar_updates: tuple[Linear, ...]
    vector_projection: Linear
    input_feature_count: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_feature_count: int,
        hidden_size: int,
        /,
        *,
        depth: int = 2,
        normalize_messages: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        inputs = int(input_feature_count)
        width = int(hidden_size)
        depth_ = int(depth)
        if inputs <= 0 or width <= 0 or depth_ <= 0:
            raise ValueError("Feature counts, hidden_size, and depth must be positive.")
        keys = jr.split(key, depth_ + 2)
        self.input_projection = Linear(
            in_size=inputs,
            out_size=width,
            activation=jax.nn.silu,
            rwf=False,
            key=keys[0],
        )
        self.convolutions = tuple(
            EquivariantGraphConvolution(normalize=normalize_messages)
            for _ in range(depth_)
        )
        self.scalar_updates = tuple(
            Linear(
                in_size=width,
                out_size=width,
                activation=jax.nn.silu,
                rwf=False,
                key=layer_key,
            )
            for layer_key in keys[1 : depth_ + 1]
        )
        self.vector_projection = Linear(
            in_size=width,
            out_size="scalar",
            activation=None,
            use_bias=False,
            rwf=False,
            key=keys[-1],
        )
        self.input_feature_count = inputs
        self.hidden_size = width

    def __call__(self, batch: MacromolecularBatch, /) -> StructureModelResult:
        if not isinstance(batch, MacromolecularBatch):
            raise TypeError("batch must be a MacromolecularBatch.")
        if batch.feature_count != self.input_feature_count:
            raise ValueError("Macromolecular feature count does not match the encoder.")
        scalar = self.input_projection(batch.node_features)
        vector = jnp.zeros(
            scalar.shape[:2] + (batch.spatial_dimension, self.hidden_size),
            dtype=scalar.dtype,
        )
        for convolution, update in zip(
            self.convolutions, self.scalar_updates, strict=True
        ):
            graph = convolution(batch.graph_ir(scalar))
            message = graph.nodes["scalar"].reshape(scalar.shape)
            vector_message = graph.nodes["vector"].reshape(vector.shape)
            scalar = jnp.where(
                batch.node_mask[..., None],
                scalar + update(message),
                0.0,
            )
            vector = jnp.where(
                batch.node_mask[..., None, None], vector + vector_message, 0.0
            )
        coordinate_updates = self.vector_projection(vector)
        return StructureModelResult(
            scalar,
            vector,
            coordinate_updates,
            batch.node_mask,
        )


__all__ = [
    "EquivariantStructureEncoder",
    "MacromolecularBatch",
    "StructureModelResult",
    "StructureModelStatus",
]
