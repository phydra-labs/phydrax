#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

from ...._doc import DOC_KEY0
from ....domain._domain import _AbstractDomain
from ....domain._function import DomainFunction
from ..._utils import _get_size
from ..core._base import _AbstractBaseModel, _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._loss import ModelWithLoss
from ..core._operator import FunctionSamples, OperatorBatch, OperatorPrediction


class OperatorContextModel(_AbstractBaseModel):
    """Differentiable point-query view with fixed neural-operator sources.

    The bridge preserves named source and case metadata, replaces one selected query,
    and can extract one named field from raw, multi-output, or task-bound operators.
    Scalar coordinate arguments and already-stacked coordinate arrays are both accepted,
    so the resulting callable composes directly with PhydraX differential operators.
    """

    operator: Any
    batch: OperatorBatch
    query_name: str
    field_name: str | None
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
        coord_dim: int | None = None,
    ):
        from ...operator_training._trained_operator import TrainedOperator

        if not isinstance(batch, OperatorBatch):
            raise TypeError("OperatorContextModel requires an OperatorBatch.")
        if query_name is None:
            resolved_query = batch.single_query_name()
        else:
            resolved_query = str(query_name)
            if resolved_query not in batch.queries:
                raise KeyError(
                    f"Unknown context query {resolved_query!r}; "
                    f"expected one of {tuple(batch.queries)!r}."
                )
        query = batch.query(resolved_query)
        if coord_dim is None:
            if query.axes:
                dimension = len(query.axes)
            elif query.coordinates is not None:
                dimension = int(query.coordinates.shape[-1])
            else:
                raise ValueError("Operator query geometry has no coordinate dimension.")
        else:
            dimension = int(coord_dim)
        if dimension <= 0:
            raise ValueError("coord_dim must be positive.")

        base_operator = operator.model if isinstance(operator, ModelWithLoss) else operator
        if isinstance(base_operator, TrainedOperator):
            declared = base_operator.task.field_by_name
            available = tuple(base_operator.output_field_map.values())
            resolved_field = available[0] if field_name is None and len(available) == 1 else field_name
            if resolved_field is None or resolved_field not in available:
                raise ValueError(
                    "field_name is required for a multi-output task-bound operator."
                )
            field = declared[str(resolved_field)]
            out_size = field.channels
        elif isinstance(base_operator, _AbstractOperatorModel):
            declared = base_operator.operator_output_specs
            available = tuple(declared)
            resolved_field = available[0] if field_name is None and len(available) == 1 else field_name
            if resolved_field is None or resolved_field not in declared:
                raise ValueError("field_name is required for a multi-output operator.")
            out_size = declared[str(resolved_field)].channels
        else:
            if not callable(base_operator):
                raise TypeError("operator must be callable.")
            resolved_field = field_name
            out_size = base_operator.out_size

        self.operator = base_operator
        self.batch = batch
        self.query_name = resolved_query
        self.field_name = None if resolved_field is None else str(resolved_field)
        self.coord_dim = dimension
        self.in_size = dimension
        self.out_size = out_size

    def _coordinates(self, values: tuple[Any, ...], /) -> Array:
        if len(values) == 1:
            coordinates = jnp.asarray(values[0])
            if coordinates.ndim == 0 and self.coord_dim == 1:
                return coordinates.reshape((1,))
            if coordinates.ndim >= 1 and int(coordinates.shape[-1]) == self.coord_dim:
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
        from ...operator_training._trained_operator import TrainedOperator

        if isinstance(self.operator, TrainedOperator):
            prepared = self.operator.prepare_prevalidated(operator_batch)
            prediction = self.operator.predict_prepared(prepared, key=key)
        elif isinstance(self.operator, _AbstractOperatorModel):
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
        point_shape = tuple(int(size) for size in coordinates.shape[:-1])
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

    def domain_function(
        self,
        domain: _AbstractDomain,
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
                f"Expected one vector label or {self.coord_dim} scalar coordinate "
                f"labels, got {len(labels)}."
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
    coord_dim: int | None = None,
) -> OperatorContextModel:
    """Return a differentiable named point-query view with fixed sources."""
    return OperatorContextModel(
        operator,
        batch,
        query_name=query_name,
        field_name=field_name,
        coord_dim=coord_dim,
    )


__all__ = ["OperatorContextModel", "bind_operator_context"]
