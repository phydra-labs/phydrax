#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

from phydrax.domain import Domain, DomainFunction

from ...._differentiation import DerivativeContract, DerivativeRegularity, DerivativeRoute
from ...._doc import DOC_KEY0
from ...._external_runtime import _require_execution
from ...._model._array import AbstractArrayModel
from ...._model._component import ExecutionCapabilities, ModelExecutionContract
from ...._model._ports import (
    ModelPorts,
    PortBindingEvidence,
    PortMapping,
    resolve_port_mapping,
)
from ...._trainable import fixed_field
from ..._base import _AbstractBaseModel
from ..._contracts import model_regularity
from ..._keys import EvalKey
from ..._loss import ModelWithLoss
from ..._utils import _get_size
from ..data import FunctionSamples, OperatorBatch, OperatorPrediction
from ..engine import AbstractOperatorModel


def _bind_trained_context(
    trained: Any,
    port_mapping: PortMapping | None,
    owner_ports: ModelPorts | None,
    /,
) -> tuple[PortBindingEvidence, str, str]:
    """Select one task query and target field of a trained operator by port ID."""
    if port_mapping is None or owner_ports is None:
        raise ValueError(
            "OperatorContextModel requires an explicit port_mapping and owner_ports: "
            f"{type(trained).__name__} declares model ports."
        )
    if not isinstance(port_mapping, PortMapping):
        raise TypeError("port_mapping must be a PortMapping.")
    if not isinstance(owner_ports, ModelPorts):
        raise TypeError("owner_ports must be ModelPorts.")
    if len(port_mapping.inputs) != 1 or len(port_mapping.outputs) != 1:
        raise ValueError(
            "OperatorContextModel port_mapping binds exactly one query input port "
            "and one output field port."
        )
    task = trained.task
    queries = {query.value_port().port_id: query for query in task.queries}
    fields = {field.value_port().port_id: field for field in task.target_fields}
    query_port_id = port_mapping.inputs[0][0]
    field_port_id = port_mapping.outputs[0][0]
    if query_port_id not in queries:
        raise ValueError(
            f"port_mapping input {query_port_id} is not a query port of the operator."
        )
    if field_port_id not in fields:
        raise ValueError(
            f"port_mapping output {field_port_id} is not a target field port of the "
            "operator."
        )
    query = queries[query_port_id]
    field = fields[field_port_id]
    evidence = resolve_port_mapping(
        ModelPorts(inputs=(query.value_port(),), outputs=(field.value_port(),)),
        owner_ports,
        port_mapping,
    )
    return evidence, query.name, field.name


class OperatorContextModel(_AbstractBaseModel):
    """Differentiable point-query view with fixed neural-operator sources.

    The bridge preserves named source and case metadata, replaces one selected query,
    and extracts one field. A port-declaring `TrainedOperator` is bound through an
    explicit `port_mapping` against `owner_ports`: its single input pair binds one
    operator query port and its single output pair binds one operator target field
    port, and the audited `port_binding` is kept. Raw operators without ports select
    by `query_name` and their own output `field_name`.
    Scalar coordinate arguments and already-stacked coordinate arrays are both accepted,
    so the resulting callable composes directly with PhydraX differential operators.
    The source ``batch`` is FIXED data; only the operator's parameters train.
    ``execution`` holds the operator's declared `ExecutionCapabilities`; every
    prediction is admitted against them before the operator is invoked, and a
    host-only operator makes the context host-only with no derivative route.
    """

    operator: Any
    batch: OperatorBatch = fixed_field()
    query_name: str
    field_name: str | None
    port_binding: PortBindingEvidence | None
    execution: ExecutionCapabilities
    coord_dim: int
    in_size: int
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(
        self,
        operator: Any,
        batch: OperatorBatch,
        /,
        *,
        query_name: str | None = None,
        field_name: str | None = None,
        port_mapping: PortMapping | None = None,
        owner_ports: ModelPorts | None = None,
        coord_dim: int | None = None,
    ):
        from ..training._trained_operator import TrainedOperator

        if not isinstance(batch, OperatorBatch):
            raise TypeError("OperatorContextModel requires an OperatorBatch.")
        base_operator = (
            operator.model if isinstance(operator, ModelWithLoss) else operator
        )
        port_binding = None
        if isinstance(base_operator, TrainedOperator):
            if query_name is not None or field_name is not None:
                raise ValueError(
                    "A TrainedOperator context selects its query and field through "
                    "port_mapping, not query_name or field_name."
                )
            port_binding, query_name, field_name = _bind_trained_context(
                base_operator, port_mapping, owner_ports
            )
        elif port_mapping is not None or owner_ports is not None:
            raise ValueError(
                "OperatorContextModel received a port_mapping, but "
                f"{type(base_operator).__name__} declares no model ports."
            )
        if query_name is None:
            resolved_query = batch.single_query_name()
        else:
            resolved_query = str(query_name)
            if resolved_query not in batch.queries:
                raise KeyError(
                    f"Unknown context query {resolved_query!r}; expected one of {tuple(batch.queries)!r}."
                )
        query = batch.query(resolved_query)
        if coord_dim is None:
            if query.axes:
                dimension = len(query.axes)
            elif query.coordinates is not None:
                dimension = query.coordinates.shape[-1]
            else:
                raise ValueError("Operator query geometry has no coordinate dimension.")
        else:
            dimension = int(coord_dim)
        if dimension <= 0:
            raise ValueError("coord_dim must be positive.")

        if isinstance(base_operator, TrainedOperator):
            resolved_field = field_name
            out_size = base_operator.task.field_by_name[resolved_field].channels
            execution = base_operator.execution_plan.execution
        elif isinstance(base_operator, AbstractOperatorModel):
            declared = base_operator.operator_output_specs
            available = tuple(declared)
            resolved_field = (
                available[0] if field_name is None and len(available) == 1 else field_name
            )
            if resolved_field is None or resolved_field not in declared:
                raise ValueError("field_name is required for a multi-output operator.")
            out_size = declared[str(resolved_field)].channels
            execution = base_operator.model_execution_contract().execution
        else:
            if not callable(base_operator):
                raise TypeError("operator must be callable.")
            resolved_field = field_name
            out_size = base_operator.out_size
            execution = (
                base_operator.model_execution_contract().execution
                if isinstance(base_operator, AbstractArrayModel)
                else ExecutionCapabilities("native-jax")
            )

        self.operator = base_operator
        self.batch = batch
        self.query_name = resolved_query
        self.field_name = None if resolved_field is None else str(resolved_field)
        self.port_binding = port_binding
        self.execution = execution
        self.coord_dim = dimension
        self.in_size = dimension
        self.out_size = out_size

    def _coordinates(self, values: tuple[Any, ...], /) -> Array:
        if len(values) == 1:
            coordinates = jnp.asarray(values[0])
            if coordinates.ndim == 0 and self.coord_dim == 1:
                return coordinates.reshape((1,))
            if coordinates.ndim >= 1 and coordinates.shape[-1] == self.coord_dim:
                return coordinates
            if self.coord_dim == 1:
                return coordinates[..., None]
        if len(values) != self.coord_dim:
            raise ValueError(
                f"OperatorContextModel expects {self.coord_dim} scalar coordinates "
                "or one array with that trailing size."
            )
        broadcast = jnp.broadcast_arrays(*(jnp.asarray(value) for value in values))
        return jnp.stack(broadcast, axis=-1)

    def _prediction(
        self,
        operator_batch: OperatorBatch,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        from ..training._trained_operator import TrainedOperator

        _require_execution(self.execution, operator_batch, key)

        if isinstance(self.operator, TrainedOperator):
            prepared = self.operator.prepare_prevalidated(operator_batch)
            prediction = self.operator.predict_prepared(prepared, key=key)
        elif isinstance(self.operator, AbstractOperatorModel):
            prediction = self.operator.predict_prevalidated(operator_batch, key=key)
        else:
            result = self.operator(operator_batch, key=key)
            if not isinstance(result, OperatorPrediction):
                return jnp.asarray(result)
            prediction = result
        assert self.field_name is not None
        return jnp.asarray(prediction.field(self.field_name).values)

    def __call__(
        self,
        *values: Any,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        coordinates = self._coordinates(values)
        point_shape = tuple(coordinates.shape[:-1])
        query = FunctionSamples(
            values=None,
            coordinates=coordinates.reshape((-1, self.coord_dim)),
        )
        queries = dict(self.batch.queries)
        queries[self.query_name] = query
        operator_batch = OperatorBatch(
            inputs=self.batch.inputs,
            queries=queries,
            case_axes=self.batch.case_axes,
            case_shape=self.batch.case_shape,
        )
        output = self._prediction(operator_batch, key=key)
        channel_shape = () if self.out_size == "scalar" else (_get_size(self.out_size),)
        return output.reshape(self.batch.case_shape + point_shape + channel_shape)

    def _value_regularity(self) -> DerivativeRegularity | None:
        # The fixed sources and the replaced query layout are data; the coordinates
        # enter the operator's own declared map.
        return model_regularity(self.operator)

    def model_execution_contract(self) -> ModelExecutionContract:
        """Return the context contract under the operator's capabilities."""
        if self.execution.host_only:
            return self._execution_contract(
                DerivativeContract(route=DerivativeRoute.STOPPED),
                execution=self.execution,
            )
        return super().model_execution_contract()

    def domain_function(
        self,
        domain: Domain,
        coordinate_labels: str | Sequence[str],
        /,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DomainFunction:
        """Bind this context as a coordinate-aware ``DomainFunction``."""
        labels = (
            (str(coordinate_labels),)
            if isinstance(coordinate_labels, str)
            else tuple(str(label) for label in coordinate_labels)
        )
        if len(labels) not in (1, self.coord_dim):
            raise ValueError(
                f"Expected one vector label or {self.coord_dim} scalar coordinate labels, got {len(labels)}."
            )
        if len(set(labels)) != len(labels):
            raise ValueError("coordinate_labels must be unique.")
        unknown = tuple(label for label in labels if label not in domain.labels)
        if unknown:
            raise KeyError(f"Domain has no coordinate labels {unknown!r}.")
        return DomainFunction(
            domain=domain,
            deps=labels,
            func=self,
            metadata=metadata,
        )


def bind_operator_context(
    operator: Any,
    batch: OperatorBatch,
    /,
    *,
    query_name: str | None = None,
    field_name: str | None = None,
    port_mapping: PortMapping | None = None,
    owner_ports: ModelPorts | None = None,
    coord_dim: int | None = None,
) -> OperatorContextModel:
    """Return a differentiable point-query view with fixed sources.

    A `TrainedOperator` is selected through `port_mapping` against `owner_ports`;
    raw operators select by `query_name` and `field_name`.
    """
    return OperatorContextModel(
        operator,
        batch,
        query_name=query_name,
        field_name=field_name,
        port_mapping=port_mapping,
        owner_ports=owner_ports,
        coord_dim=coord_dim,
    )


__all__ = ["OperatorContextModel", "bind_operator_context"]
