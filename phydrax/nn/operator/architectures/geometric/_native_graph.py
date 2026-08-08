#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""OperatorBatch adapter for native graph and simplicial processors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from phydrax._doc import DOC_KEY0
from phydrax.graph._ir import GraphIR
from phydrax.nn._keys import EvalKey
from phydrax.nn.operator.data import OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel
from phydrax.nn.operator.topology import (
    gather_operator_graph_entities,
    operator_graph_from_samples,
)


class NativeGraphOperator(AbstractOperatorModel):
    """Run a ``GraphIR`` processor directly on an operator batch's topology.

    Source values, coordinates, quadrature, masks, and any pre-existing graph-node
    metadata are materialized on the canonical graph. The wrapped processor must
    return a ``GraphIR``; its selected node field is gathered at the query's native
    topology sites. Source and query mappings may select different nodes, but they
    must reference the same graph representation.
    """

    operator_architecture = "GraphNeuralOperator"

    processor: Callable[[GraphIR], GraphIR]
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    source_name: str | None = eqx.field(static=True)
    input_key: str = eqx.field(static=True)
    output_key: str | None = eqx.field(static=True)

    def __init__(
        self,
        processor: Callable[[GraphIR], GraphIR],
        /,
        *,
        in_size: int | tuple[int, ...] | Literal["scalar"],
        out_size: int | tuple[int, ...] | Literal["scalar"],
        source_name: str | None = None,
        input_key: str = "features",
        output_key: str | None = None,
    ):
        if not callable(processor):
            raise TypeError("NativeGraphOperator processor must be callable.")
        if not str(input_key):
            raise ValueError("input_key must be non-empty.")
        if output_key is not None and not str(output_key):
            raise ValueError("output_key must be non-empty when provided.")
        self.processor = processor
        self.in_size = in_size
        self.out_size = out_size
        self.source_name = None if source_name is None else str(source_name)
        self.input_key = str(input_key)
        self.output_key = None if output_key is None else str(output_key)

    def _source(self, batch: OperatorBatch, /):
        if self.source_name is not None:
            return batch.input(self.source_name)
        candidates = tuple(
            (name, samples)
            for name, samples in batch.inputs.items()
            if samples.values is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "NativeGraphOperator requires source_name when the batch does not "
                "contain exactly one valued input."
            )
        return candidates[0][1]

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        source = self._source(batch)
        query = batch.require_single_query()
        if source.topology is None or query.topology is None:
            raise ValueError(
                "NativeGraphOperator requires native topology on source and query."
            )
        if source.topology.graph_fingerprint != query.topology.graph_fingerprint:
            raise ValueError(
                "NativeGraphOperator source and query topology must reference the "
                "same canonical graph."
            )
        if source.topology.entity != "node" or query.topology.entity != "node":
            raise ValueError(
                "NativeGraphOperator requires node-entity topology mappings."
            )

        graph = operator_graph_from_samples(source, case_shape=batch.case_shape)
        if self.input_key != "features":
            if not isinstance(graph.nodes, Mapping):
                raise TypeError("Materialized operator graph nodes must be a mapping.")
            nodes = dict(graph.nodes)
            nodes[self.input_key] = nodes.pop("features")
            graph = graph.replace(nodes=nodes, validate=False)

        output_graph = self.processor(graph)
        if not isinstance(output_graph, GraphIR):
            raise TypeError("NativeGraphOperator processor must return a GraphIR.")
        output_nodes: Any = output_graph.nodes
        if self.output_key is not None:
            if not isinstance(output_nodes, Mapping):
                raise TypeError(
                    "output_key requires the processor to return mapping-valued nodes."
                )
            if self.output_key not in output_nodes:
                raise KeyError(
                    f"Processor output nodes have no field {self.output_key!r}."
                )
            output_nodes = output_nodes[self.output_key]
        elif isinstance(output_nodes, Mapping):
            if self.input_key not in output_nodes:
                raise ValueError(
                    "Mapping-valued processor output requires output_key or a field "
                    f"named {self.input_key!r}."
                )
            output_nodes = output_nodes[self.input_key]

        gathered = jnp.asarray(
            gather_operator_graph_entities(
                query,
                output_nodes,
                case_shape=batch.case_shape,
            )
        )
        if self.out_size == "scalar" and gathered.ndim == (
            len(batch.case_shape) + len(query.sample_shape) + 1
        ):
            if int(gathered.shape[-1]) != 1:
                raise ValueError(
                    "Scalar NativeGraphOperator output must have no channel axis or "
                    "one trailing channel."
                )
            gathered = gathered[..., 0]
        return gathered

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("NativeGraphOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(batch, key=key)


__all__ = ["NativeGraphOperator"]
