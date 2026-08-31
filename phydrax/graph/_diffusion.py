#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._ir import GraphIR


GraphPayloadKind: TypeAlias = Literal["nodes", "edges"]


def _optional_array_equal(left: Array | None, right: Array | None, /) -> bool:
    if left is None or right is None:
        return left is right
    return bool(jnp.array_equal(left, right))


def _payload(graph: GraphIR, kind: GraphPayloadKind, key: str, /) -> Array:
    container = graph.nodes if kind == "nodes" else graph.edges
    if not isinstance(container, Mapping) or key not in container:
        raise ValueError(f"Graph {kind} payload has no {key!r} array.")
    value = jnp.asarray(container[key])
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        raise TypeError("Diffused graph payloads must have an inexact dtype.")
    return value


def _replace(graph: GraphIR, kind: GraphPayloadKind, key: str, value: Array, /) -> GraphIR:
    container = graph.nodes if kind == "nodes" else graph.edges
    assert isinstance(container, Mapping)
    updated = {**container, key: value}
    return graph.replace(
        nodes=updated if kind == "nodes" else graph.nodes,
        edges=updated if kind == "edges" else graph.edges,
        validate=False,
    )


class FixedTopologyGraphDiffusion(StrictModule):
    """Gaussian diffusion of one floating graph payload on immutable topology."""

    template: GraphIR
    process: Any
    payload_kind: GraphPayloadKind = eqx.field(static=True)
    payload_key: str = eqx.field(static=True)
    payload_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        template: GraphIR,
        process: Any,
        /,
        *,
        payload_kind: GraphPayloadKind,
        payload_key: str,
        process_id: str | None = None,
    ):
        from ..stochastic._gaussian_diffusion import AbstractGaussianDiffusion

        if not isinstance(template, GraphIR):
            raise TypeError("template must be a GraphIR.")
        if not isinstance(process, AbstractGaussianDiffusion):
            raise TypeError("process must implement AbstractGaussianDiffusion.")
        if payload_kind not in ("nodes", "edges") or not payload_key:
            raise ValueError("payload_kind/key are invalid.")
        value = _payload(template, payload_kind, payload_key)
        if process.state_shape != (int(value.size),):
            raise ValueError("Gaussian process dimension must equal flattened payload size.")
        self.template = template
        self.process = process
        self.payload_kind = payload_kind
        self.payload_key = payload_key
        self.payload_shape = tuple(value.shape)
        self.process_id = process_id or canonical_fingerprint(
            {
                "kind": "fixed-topology-graph-diffusion",
                "process_id": process.process_id,
                "payload_kind": payload_kind,
                "payload_key": payload_key,
                "payload_shape": list(value.shape),
            }
        )

    def _require_topology(self, graph: GraphIR, /) -> Array:
        if not isinstance(graph, GraphIR):
            raise TypeError("graph must be a GraphIR.")
        if not (
            _optional_array_equal(graph.senders, self.template.senders)
            and _optional_array_equal(graph.receivers, self.template.receivers)
            and bool(jnp.array_equal(graph.n_node, self.template.n_node))
            and bool(jnp.array_equal(graph.n_edge, self.template.n_edge))
            and _optional_array_equal(graph.node_mask, self.template.node_mask)
            and _optional_array_equal(graph.edge_mask, self.template.edge_mask)
            and _optional_array_equal(graph.graph_mask, self.template.graph_mask)
        ):
            raise ValueError("Graph diffusion requires exactly the template topology.")
        value = _payload(graph, self.payload_kind, self.payload_key)
        if value.shape != self.payload_shape:
            raise ValueError("Graph payload shape differs from the template.")
        return value

    def perturb(self, graph: GraphIR, key: Key[Array, ""], /, *, time) -> GraphIR:
        value = self._require_topology(graph)
        perturbed = self.process.perturb(key, value.reshape((-1,)), t1=time)
        result = perturbed.reshape(self.payload_shape)
        mask = graph.node_mask if self.payload_kind == "nodes" else graph.edge_mask
        if mask is not None:
            expanded = jnp.asarray(mask, dtype=bool).reshape(
                mask.shape + (1,) * (result.ndim - mask.ndim)
            )
            result = jnp.where(expanded, result, value)
        return _replace(graph, self.payload_kind, self.payload_key, result)

    def conditional_score(self, perturbed: GraphIR, clean: GraphIR, /, *, time) -> Array:
        noisy = self._require_topology(perturbed).reshape((-1,))
        source = self._require_topology(clean).reshape((-1,))
        score = self.process.conditional_score(noisy, source, t1=time).reshape(
            self.payload_shape
        )
        mask = clean.node_mask if self.payload_kind == "nodes" else clean.edge_mask
        if mask is None:
            return score
        expanded = jnp.asarray(mask, dtype=bool).reshape(
            mask.shape + (1,) * (score.ndim - mask.ndim)
        )
        return jnp.where(expanded, score, 0.0)


def graph_denoising_loss(
    diffusion: FixedTopologyGraphDiffusion,
    score_model: Any,
    clean: GraphIR,
    key: Key[Array, ""],
    /,
    *,
    time,
) -> Array:
    noise_key, model_key = jr.split(key)
    perturbed = diffusion.perturb(clean, noise_key, time=time)
    target = diffusion.conditional_score(perturbed, clean, time=time)
    prediction = jnp.asarray(score_model(perturbed, time, key=model_key))
    if prediction.shape != target.shape:
        raise ValueError("Graph score model output must match the diffused payload.")
    mask = clean.node_mask if diffusion.payload_kind == "nodes" else clean.edge_mask
    if mask is None:
        return jnp.mean((prediction - target) ** 2)
    expanded = jnp.asarray(mask, dtype=bool).reshape(
        mask.shape + (1,) * (prediction.ndim - mask.ndim)
    )
    expanded = jnp.broadcast_to(expanded, prediction.shape)
    count = jnp.sum(expanded)
    count = eqx.error_if(count, count <= 0, "Graph diffusion mask is empty.")
    return jnp.sum(jnp.where(expanded, (prediction - target) ** 2, 0.0)) / count


__all__ = ["FixedTopologyGraphDiffusion", "GraphPayloadKind", "graph_denoising_loss"]
