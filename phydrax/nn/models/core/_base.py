#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal, TypeAlias

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array

from ...._doc import DOC_KEY0
from ...._strict import AbstractAttribute, StrictModule
from ._keys import EvalKey
from ._operator import OperatorBatch, OperatorOutputSpec, OperatorPrediction
from ._operator_domain import (
    operator_domain_view_from_coord_separable,
    operator_domain_view_from_graph,
    operator_domain_view_from_points,
    operator_domain_view_from_ragged_series,
    operator_domain_view_from_trajectory,
)


DomainInputMode: TypeAlias = Literal["flat", "structured"]


class _AbstractBaseModel(StrictModule):
    """Abstract base class for callable models with defined input and output sizes."""

    in_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]
    out_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]

    @abstractmethod
    def __call__(
        self,
        x: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    _supports_structured_input: ClassVar[bool] = False
    _supports_blockwise_input: ClassVar[bool] = False
    _supports_axis_batch_input: ClassVar[bool] = False
    _warn_on_auto_fallback: ClassVar[bool] = False
    _domain_input_mode: ClassVar[DomainInputMode] = "flat"

    @classmethod
    def supports_structured_input(cls) -> bool:
        return cls._supports_structured_input

    @classmethod
    def supports_blockwise_input(cls) -> bool:
        return cls._supports_blockwise_input

    def warn_on_auto_fallback(self) -> bool:
        return bool(self._warn_on_auto_fallback)

    def supports_axis_batch_input(self) -> bool:
        return bool(self._supports_axis_batch_input)

    @classmethod
    def domain_input_mode(cls) -> DomainInputMode:
        return cls._domain_input_mode

    def __loss__(
        self,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> Array:
        del key, iter_
        return jnp.array(0.0, dtype=float)

    def add_model_loss(
        self,
        penalty: Callable[..., Any],
        /,
        *,
        weight: Any = 1.0,
        label: str | None = None,
    ) -> Any:
        """Return a model wrapper that contributes an extra scalar objective term."""
        from ._loss import add_model_loss

        return add_model_loss(self, penalty, weight=weight, label=label)


class _AbstractStructuredInputModel(_AbstractBaseModel):
    """Abstract base for models whose concrete structured-input schema is model-specific."""

    _supports_structured_input: ClassVar[bool] = True
    _domain_input_mode: ClassVar[DomainInputMode] = "structured"

    @abstractmethod
    def __call__(
        self,
        x: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError


OperatorPredictionBuilder: TypeAlias = Callable[
    [Any, OperatorBatch, EvalKey],
    OperatorPrediction,
]


class _AbstractOperatorModel(_AbstractStructuredInputModel):
    """Abstract model that consumes full operator source/query metadata."""

    _supports_axis_batch_input: ClassVar[bool] = True
    _operator_prediction_builder: ClassVar[OperatorPredictionBuilder | None] = None
    @property
    def operator_contract(self):
        """Return the configured runtime contract derived from this model instance."""
        from ._operator_architecture_status import operator_instance_contract

        return operator_instance_contract(self)


    @abstractmethod
    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        """Return statically declared named output fields."""
        if self.out_size == "scalar":
            spec = OperatorOutputSpec("scalar")
        elif isinstance(self.out_size, int):
            spec = OperatorOutputSpec(self.out_size)
        else:
            channels = 1
            for size in self.out_size:
                channels *= int(size)
            spec = OperatorOutputSpec(channels)
        return {"output": spec}


    def predict_prevalidated(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Evaluate a batch whose static runtime contract was checked on the host."""
        if self._operator_prediction_builder is not None:
            return self._operator_prediction_builder(self, batch, key)
        values = self.__call_operator_batch__(batch, key=key)
        query_name = batch.single_query_name()
        return OperatorPrediction.from_field(
            "output",
            values,
            query_name,
            batch.query(query_name),
            spec=self.operator_output_specs["output"],
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )


    def predict(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Evaluate and retain named query, case-axis, and output metadata."""
        self.operator_contract.validate(batch).require_runtime()
        return self.predict_prevalidated(batch, key=key)

    def __call_axis_batch__(
        self,
        batch: Any,
        deps: tuple[str, ...],
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Any | None = None,
        **kwargs: Any,
    ) -> cx.Field:
        from ....domain._structure import CoordSeparableBatch, PointsBatch
        from ....domain._trajectory_dataset import TRAJECTORY_CASE_INDEX_KEY
        from ....domain.graph._batch import GraphBatch

        del iter_, kwargs
        if isinstance(batch, PointsBatch):
            if not deps:
                raise ValueError("Neural-operator point batches require dependencies.")
            first_payload = batch.points[deps[0]]
            if isinstance(first_payload, Mapping) and all(
                name in first_payload for name in ("series", "time", "mask", "length")
            ):
                view = operator_domain_view_from_ragged_series(batch, deps[0])
                view.require_compatible(self)
                values = jnp.asarray(self.__call_operator_batch__(view.batch, key=key))
                return view.layouts["query"].restore(values)
            if TRAJECTORY_CASE_INDEX_KEY in batch.points:
                query_label = deps[-1]
                source_deps = deps[:-1] if len(deps) > 1 else deps
                view = operator_domain_view_from_trajectory(
                    batch,
                    inputs={dep: dep for dep in source_deps},
                    query_label=query_label,
                )
                view.require_compatible(self)
                values = jnp.asarray(self.__call_operator_batch__(view.batch, key=key))
                return view.layouts["query"].restore(values)
            query_label = deps[-1]
            query_value = batch.points[query_label]
            if not isinstance(query_value, cx.Field):
                raise TypeError(
                    "Automatic point-domain operator dispatch requires one Field "
                    "for the final query dependency."
                )
            named_query_axes = tuple(
                dim for dim in query_value.dims if dim is not None
            )
            structure_axes = tuple(batch.structure.axis_names or ())
            sample_axes = tuple(
                axis for axis in structure_axes if axis in named_query_axes
            )
            if len(sample_axes) != 1:
                raise ValueError(
                    "Automatic point-domain operator dispatch requires one query "
                    "sample axis; use operator_domain_view_from_points explicitly."
                )
            sample_axis = sample_axes[0]
            case_axes = tuple(axis for axis in structure_axes if axis != sample_axis)
            source_deps = deps[:-1] if len(deps) > 1 else deps
            input_coordinates = {
                dep: query_label
                for dep in source_deps
                if dep == query_label
                or (
                    isinstance(batch.points[dep], cx.Field)
                    and sample_axis in batch.points[dep].dims
                )
            }
            view = operator_domain_view_from_points(
                batch,
                inputs={dep: dep for dep in source_deps},
                queries={"query": query_label},
                input_coordinates=input_coordinates,
                case_axes=case_axes,
            )
            view.require_compatible(self)
            values = jnp.asarray(self.__call_operator_batch__(view.batch, key=key))
            return view.layouts["query"].restore(values)
        if isinstance(batch, GraphBatch):
            graph_axis = batch.structure.axis_for(batch.graph_label)
            if graph_axis is None:
                raise ValueError("GraphBatch graph labels require a sampled entity axis.")
            query_labels = tuple(
                dep
                for dep in deps
                if dep != batch.graph_label
                and isinstance(batch.points[dep], cx.Field)
                and graph_axis in batch.points[dep].dims
            )
            source_deps = tuple(dep for dep in deps if dep not in query_labels)
            if not source_deps:
                raise ValueError(
                    "Graph-domain operator dispatch requires a source dependency."
                )
            view = operator_domain_view_from_graph(
                batch,
                inputs={dep: dep for dep in source_deps},
                query_labels=query_labels,
            )
            view.require_compatible(self)
            values = jnp.asarray(self.__call_operator_batch__(view.batch, key=key))
            return view.layouts["query"].restore(values)
        if not isinstance(batch, CoordSeparableBatch):
            raise TypeError(
                "Neural-operator axis-batch execution requires a supported "
                "structured domain batch."
            )
        coordinate_labels = tuple(
            dep for dep in deps if isinstance(batch.points[dep], tuple)
        )
        inputs = {
            dep: dep for dep in deps if not isinstance(batch.points[dep], tuple)
        }
        view = operator_domain_view_from_coord_separable(
            batch,
            inputs=inputs,
            queries={"query": coordinate_labels},
        )
        view.require_compatible(self)
        values = jnp.asarray(self.__call_operator_batch__(view.batch, key=key))
        return view.layouts["query"].restore(values)
