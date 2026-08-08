#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array

from ..._doc import DOC_KEY0
from ..._model import AxisModelEvaluator, ModelBinding
from ...domain.operator import (
    operator_domain_view_from_graph,
    operator_domain_view_from_grid,
    operator_domain_view_from_points,
    operator_domain_view_from_ragged_series,
    operator_domain_view_from_trajectory,
)
from .._base import _AbstractStructuredInputModel
from .._keys import EvalKey
from .data import OperatorBatch, OperatorOutputSpec, OperatorPrediction
from .protocols import OperatorModel, OperatorPredictionBuilder


class AbstractOperatorModel(
    _AbstractStructuredInputModel,
    AxisModelEvaluator,
    OperatorModel,
):
    """Abstract model that consumes full operator source/query metadata."""

    _input_binding: ClassVar[ModelBinding] = ModelBinding.axis()
    _operator_prediction_builder: ClassVar[OperatorPredictionBuilder | None] = None
    operator_architecture: ClassVar[str | None] = None
    _operator_contract_builder: ClassVar[Callable[[Any], Any] | None] = None
    _operator_contract_configuration: ClassVar[
        Callable[[Any], Mapping[str, Any] | tuple[tuple[str, Any], ...]] | None
    ] = None

    @property
    def operator_contract(self):
        """Return the runtime contract declared by this concrete operator engine."""
        from .catalog import (
            _reconcile_instance_contract,
            operator_architecture_contract,
        )

        if self._operator_contract_builder is not None:
            return _reconcile_instance_contract(
                self,
                self._operator_contract_builder(self),
            )
        if self.operator_architecture is None:
            raise TypeError(
                f"{type(self).__name__} does not declare an operator architecture."
            )
        configuration = (
            ()
            if self._operator_contract_configuration is None
            else self._operator_contract_configuration(self)
        )
        return _reconcile_instance_contract(
            self,
            operator_architecture_contract(
                self.operator_architecture,
                configuration=configuration,
            ),
        )

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

    def evaluate(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Evaluate and retain named query, case-axis, and output metadata."""
        self.operator_contract.validate(batch).require_runtime()
        return self.predict_prevalidated(batch, key=key)

    def predict(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Compatibility spelling for :meth:`evaluate` during runtime migration."""
        return self.evaluate(batch, key=key)

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
        from phydrax.domain import GridBatch, PointBatch, TRAJECTORY_CASE_INDEX_KEY
        from phydrax.domain.graph import GraphBatch

        del iter_, kwargs
        if isinstance(batch, PointBatch):
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
            named_query_axes = tuple(dim for dim in query_value.dims if dim is not None)
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
        if not isinstance(batch, GridBatch):
            raise TypeError(
                "Neural-operator axis-batch execution requires a supported "
                "structured domain batch."
            )
        coordinate_labels = tuple(
            dep for dep in deps if isinstance(batch.points[dep], tuple)
        )
        inputs = {dep: dep for dep in deps if not isinstance(batch.points[dep], tuple)}
        view = operator_domain_view_from_grid(
            batch,
            inputs=inputs,
            queries={"query": coordinate_labels},
        )
        view.require_compatible(self)
        values = jnp.asarray(self.__call_operator_batch__(view.batch, key=key))
        return view.layouts["query"].restore(values)


__all__ = ["AbstractOperatorModel"]
