#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bridges between operator samples and canonical graph topology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...graph._ir import GraphIR
from ...graph._operator_topology import (
    broadcast_operator_topology,
    operator_graph_fingerprint,
    operator_topology_fingerprint,
    OperatorTopology,
    OperatorTopologyEntity,
    OperatorTopologyKind,
    OperatorTopologySite,
    pad_operator_topology,
    slice_operator_topology,
    stack_operator_topologies,
    take_operator_topology,
)
from .data import FunctionSamples, OperatorBatch
from .field import OperatorFieldSpec


def _scatter_samples(
    values: PyTree[Any],
    mapping: Array,
    valid: Array,
    entity_count: int,
    leading_shape: tuple[int, ...],
    /,
) -> PyTree[Any]:
    flat_mapping = mapping.reshape((-1,))
    flat_valid = valid.reshape((-1,))
    safe_mapping = jnp.maximum(flat_mapping, 0)
    leading_ndim = len(leading_shape)

    def scatter(leaf: Any) -> Array:
        array = jnp.asarray(leaf)
        if tuple(array.shape[:leading_ndim]) != leading_shape:
            raise ValueError(
                f"Sample value must start with shape {leading_shape}; got {array.shape}."
            )
        trailing = tuple(int(size) for size in array.shape[leading_ndim:])
        flattened = array.reshape((-1,) + trailing)
        mask = flat_valid.reshape((-1,) + (1,) * len(trailing))
        output = jnp.zeros((entity_count,) + trailing, dtype=array.dtype)
        updates = jnp.where(mask, flattened, jnp.zeros((), dtype=array.dtype))
        if jnp.issubdtype(array.dtype, jnp.bool_):
            return output.at[safe_mapping].max(updates)
        return output.at[safe_mapping].add(updates)

    return jax.tree_util.tree_map(scatter, values)


def scatter_operator_graph_entities(
    samples: FunctionSamples,
    sample_values: PyTree[Any],
    /,
    *,
    case_shape: Sequence[int] | None = None,
) -> PyTree[Any]:
    """Scatter topology-aligned sample values to canonical graph entities."""
    if samples.topology is None:
        raise ValueError("FunctionSamples has no native topology.")
    target_cases = (
        samples.topology.case_shape
        if case_shape is None
        else tuple(int(size) for size in case_shape)
    )
    topology = broadcast_operator_topology(samples.topology, target_cases)
    mapping = topology.absolute_sample_entities()
    sample_mask = samples.mask_array(case_shape=target_cases)
    valid = (mapping >= 0) & sample_mask
    return _scatter_samples(
        sample_values,
        mapping,
        valid,
        topology.entity_count,
        target_cases + samples.sample_shape,
    )


def materialize_operator_fields(
    batch: OperatorBatch,
    fields: Sequence[OperatorFieldSpec],
    /,
) -> GraphIR:
    """Scatter every named source field onto one shared canonical topology."""
    if not isinstance(batch, OperatorBatch):
        raise TypeError("materialize_operator_fields requires an OperatorBatch.")
    specs = tuple(fields)
    if not specs or any(not isinstance(field, OperatorFieldSpec) for field in specs):
        raise TypeError(
            "materialize_operator_fields requires non-empty OperatorFieldSpec fields."
        )
    bound: list[tuple[str, FunctionSamples]] = []
    for field in specs:
        if field.is_source:
            assert field.source_name is not None
            if field.source_name not in batch.inputs:
                if field.required:
                    raise KeyError(
                        f"Missing required source field {field.source_name!r}."
                    )
            else:
                bound.append((field.name, batch.input(field.source_name)))
        if field.is_target:
            assert field.query_name is not None
            if field.query_name not in batch.queries:
                if field.required:
                    raise KeyError(f"Missing required query {field.query_name!r}.")
            else:
                bound.append((f"query:{field.name}", batch.query(field.query_name)))
    if not bound or bound[0][1].topology is None:
        raise ValueError("Named operator fields require attached native topology.")
    first_topology = bound[0][1].topology
    assert first_topology is not None
    if first_topology.entity != "node":
        raise ValueError("Named cochain fields must be represented by graph nodes.")
    for name, samples in bound:
        topology = samples.topology
        if (
            topology is None
            or topology.kind != "cell_complex"
            or topology.entity != "node"
            or topology.graph_fingerprint != first_topology.graph_fingerprint
        ):
            raise ValueError(
                f"Field {name!r} does not share the canonical cell-complex topology."
            )
    topology = broadcast_operator_topology(first_topology, batch.case_shape)
    if not isinstance(topology.graph.nodes, Mapping):
        raise ValueError("Cell-complex topology requires named graph-node metadata.")
    nodes = dict(topology.graph.nodes)
    for field in specs:
        if not field.is_source:
            continue
        assert field.source_name is not None
        if field.source_name not in batch.inputs:
            continue
        samples = batch.input(field.source_name)
        if samples.values is None:
            raise ValueError(f"Source field {field.name!r} has no sampled values.")
        nodes[f"field:{field.name}"] = scatter_operator_graph_entities(
            samples,
            samples.values,
            case_shape=batch.case_shape,
        )
        nodes[f"field_mask:{field.name}"] = scatter_operator_graph_entities(
            samples,
            samples.mask_array(case_shape=batch.case_shape),
            case_shape=batch.case_shape,
        ).astype(bool)
    return topology.graph.replace(nodes=nodes, validate=True)


def operator_graph_from_samples(
    samples: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int] | None = None,
) -> GraphIR:
    """Materialize sample values and geometry on their native canonical graph."""
    if samples.topology is None:
        raise ValueError("FunctionSamples has no native topology.")
    target_cases = (
        samples.topology.case_shape
        if case_shape is None
        else tuple(int(size) for size in case_shape)
    )
    topology = broadcast_operator_topology(samples.topology, target_cases)
    mapping = topology.absolute_sample_entities()
    sample_mask = samples.mask_array(case_shape=target_cases)
    valid = (mapping >= 0) & sample_mask
    leading_shape = target_cases + samples.sample_shape
    entity_count = topology.entity_count
    coordinates = _scatter_samples(
        samples.coordinates_array(case_shape=target_cases),
        mapping,
        valid,
        entity_count,
        leading_shape,
    )
    quadrature = _scatter_samples(
        samples.quadrature(case_shape=target_cases),
        mapping,
        valid,
        entity_count,
        leading_shape,
    )
    if topology.entity == "node":
        payload = topology.graph.nodes
    elif topology.entity == "edge":
        payload = topology.graph.edges
    else:
        payload = topology.graph.globals
    entities: dict[str, Any]
    if isinstance(payload, Mapping):
        entities = dict(payload)
    else:
        entities = {}
        if payload is not None:
            entities["topology"] = payload
    entities.update(
        {
            "coordinates": coordinates,
            "quadrature_weights": quadrature,
            "sample_mask": _scatter_samples(
                sample_mask,
                mapping,
                valid,
                entity_count,
                leading_shape,
            ).astype(bool),
        }
    )
    if samples.values is not None:
        entities["features"] = _scatter_samples(
            samples.values,
            mapping,
            valid,
            entity_count,
            leading_shape,
        )
    if topology.entity == "node":
        return topology.graph.replace(nodes=entities, validate=True)
    if topology.entity == "edge":
        return topology.graph.replace(edges=entities, validate=True)
    return topology.graph.replace(globals=entities, validate=True)


def gather_operator_graph_entities(
    samples: FunctionSamples,
    entity_values: PyTree[Any],
    /,
    *,
    case_shape: Sequence[int] | None = None,
) -> PyTree[Any]:
    """Gather graph-entity output into the sample shape and mask padding."""
    if samples.topology is None:
        raise ValueError("FunctionSamples has no native topology.")
    target_cases = (
        samples.topology.case_shape
        if case_shape is None
        else tuple(int(size) for size in case_shape)
    )
    topology = broadcast_operator_topology(samples.topology, target_cases)
    mapping = topology.absolute_sample_entities()
    sample_mask = samples.mask_array(case_shape=target_cases)
    valid = (mapping >= 0) & sample_mask
    flat_mapping = mapping.reshape((-1,))
    safe_mapping = jnp.maximum(flat_mapping, 0)
    output_shape = target_cases + samples.sample_shape

    def gather(leaf: Any) -> Array:
        array = jnp.asarray(leaf)
        if int(array.shape[0]) != topology.entity_count:
            raise ValueError(
                f"Graph-entity values require leading size {topology.entity_count}; "
                f"got {array.shape[0]}."
            )
        trailing = tuple(int(size) for size in array.shape[1:])
        gathered = array[safe_mapping].reshape(output_shape + trailing)
        mask = valid.reshape(output_shape + (1,) * len(trailing))
        return jnp.where(mask, gathered, 0)

    return jax.tree_util.tree_map(gather, entity_values)


__all__ = [
    "OperatorTopology",
    "OperatorTopologyEntity",
    "OperatorTopologyKind",
    "OperatorTopologySite",
    "broadcast_operator_topology",
    "gather_operator_graph_entities",
    "materialize_operator_fields",
    "operator_graph_fingerprint",
    "operator_graph_from_samples",
    "operator_topology_fingerprint",
    "pad_operator_topology",
    "scatter_operator_graph_entities",
    "slice_operator_topology",
    "stack_operator_topologies",
    "take_operator_topology",
]
