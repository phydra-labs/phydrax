#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Degree-aware metric residual constraints on cochain complexes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import BatchEvaluator, DomainComponent, DomainFunction, PointSampling
from phydrax.domain.graph import (
    cochain_field_spec,
    CochainCells,
    GRAPH_ENTITY_INDEX_KEY,
    GRAPH_GRAPH_INDEX_KEY,
    graph_trajectory_default_quadrature_total_weight,
    GraphBatch,
    with_cochain_field_spec,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..graph import cochain_metric_reduce, CochainMetricReduction, CochainResidualProgram
from ..nn.models.wrappers._graph import _full_node_batch


def _component_degree(component: DomainComponent, /) -> int:
    selected = tuple(
        selector.degree
        for label in component.domain.labels
        if isinstance(
            selector := component.spec.selection_for(label),
            CochainCells,
        )
    )
    if len(selected) != 1:
        raise ValueError(
            "CochainResidualTerm requires exactly one CochainCells selector."
        )
    return selected[0]


def _squared_cell_values(field: cx.Field, batch: GraphBatch, /) -> Array:
    axis = batch.structure.axis_for(batch.graph_label)
    if axis is None:
        raise ValueError("GraphBatch has no sampling axis for its graph label.")
    if axis not in field.dims:
        raise ValueError(
            f"Residual field dims {field.dims!r} omit graph sampling axis {axis!r}."
        )
    named = tuple(dim for dim in field.dims if dim is not None)
    if named != (axis,):
        raise ValueError(
            "Cochain residuals must have one named graph sampling axis; "
            f"got dims {field.dims!r}."
        )
    axis_index = field.dims.index(axis)
    data = jnp.moveaxis(jnp.asarray(field.data), axis_index, 0)
    squared = jnp.real(jnp.conj(data) * data)
    if squared.ndim > 1:
        squared = jnp.sum(squared, axis=tuple(range(1, squared.ndim)))
    return squared


def _hodge_weights(batch: GraphBatch, /) -> Array:
    payload = batch.points[batch.graph_label]
    if not isinstance(payload, Mapping) or "hodge_star" not in payload:
        raise ValueError(
            "Cochain metric reduction requires graph.nodes['hodge_star'] metadata."
        )
    weight = payload["hodge_star"]
    if not isinstance(weight, cx.Field):
        raise TypeError("Sampled cochain hodge_star values must be coordax.Fields.")
    return jnp.asarray(weight.data, dtype=float).reshape((-1,))


def _trajectory_segment_weights(
    component: DomainComponent,
    batch: GraphBatch,
    /,
) -> Array | None:
    weight = graph_trajectory_default_quadrature_total_weight(component, batch)
    if weight is None:
        return None
    values = jnp.asarray(weight.data, dtype=float).reshape((-1,))
    return values * float(values.shape[0])


class _ProgramDomainOutput(StrictModule, BatchEvaluator):
    program: CochainResidualProgram
    fields: frozendict[str, DomainFunction]
    output_name: str

    def __init__(
        self,
        program: CochainResidualProgram,
        fields: Mapping[str, DomainFunction],
        output_name: str,
        /,
    ):
        self.program = program
        self.fields = frozendict(fields)
        self.output_name = str(output_name)

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, GraphBatch) or batch.component_kind != "nodes":
            raise TypeError("Cochain residual programs require node-backed GraphBatch data.")
        axis = batch.structure.axis_for(batch.graph_label)
        if axis is None:
            raise ValueError("GraphBatch has no graph sampling axis.")
        full_batch = _full_node_batch(batch)
        arrays: dict[str, Array] = {}
        for name, field in self.fields.items():
            evaluated = field(full_batch, key=key, **kwargs)
            if not isinstance(evaluated, cx.Field) or axis not in evaluated.dims:
                raise TypeError(
                    f"Program input field {name!r} must return a field on axis {axis!r}."
                )
            arrays[name] = jnp.moveaxis(
                jnp.asarray(evaluated.data),
                evaluated.dims.index(axis),
                0,
            )
        full_output = self.program(batch.graph, arrays, key=key, **kwargs)[
            self.output_name
        ]
        entity_field = batch.points.get(GRAPH_ENTITY_INDEX_KEY)
        if not isinstance(entity_field, cx.Field):
            raise TypeError("GraphBatch is missing canonical entity indices.")
        indices = jnp.asarray(entity_field.data, dtype=jnp.int32)
        selected = full_output[indices]
        return cx.Field(
            selected,
            dims=(axis,) + (None,) * (selected.ndim - 1),
        )


def cochain_residual_field(
    program: CochainResidualProgram,
    fields: Mapping[str, DomainFunction],
    output: str,
    /,
) -> DomainFunction:
    """Bind one shared residual-program output to graph-backed domain fields."""
    if not isinstance(program, CochainResidualProgram):
        raise TypeError("cochain_residual_field requires a CochainResidualProgram.")
    if frozenset(fields) != frozenset(program.input_specs):
        raise ValueError("Program field names must exactly match its declared input schema.")
    output_name = str(output)
    if output_name not in program.output_specs:
        raise KeyError(f"Unknown cochain residual output {output_name!r}.")
    if not fields:
        raise ValueError("Cochain residual fields must be non-empty.")

    normalized: dict[str, DomainFunction] = {}
    base = next(iter(fields.values()))
    if not isinstance(base, DomainFunction):
        raise TypeError("Program inputs must be DomainFunctions.")
    for name, expected in program.input_specs.items():
        field = fields[name]
        if not isinstance(field, DomainFunction):
            raise TypeError(f"Program input {name!r} is not a DomainFunction.")
        if field.domain.labels != base.domain.labels:
            raise ValueError("All cochain residual fields must share one domain.")
        actual = cochain_field_spec(field)
        if actual != expected:
            raise ValueError(
                f"Program input {name!r} semantics do not match its declared schema."
            )
        normalized[name] = field
    deps = tuple(
        label
        for label in base.domain.labels
        if any(label in field.deps for field in normalized.values())
    )
    result = DomainFunction(
        domain=base.domain,
        deps=deps,
        func=_ProgramDomainOutput(program, normalized, output_name),
        metadata={},
    )
    return with_cochain_field_spec(result, program.output_specs[output_name])


class CochainResidualTerm(AbstractSamplingTerm):
    """Metric-aware sampled residual loss for a declared cochain degree.

    Every sampled graph or graph-time segment contributes equally. The cell
    reduction is selected by ``reduction``:

    - ``graph_mean``: arithmetic mean over cells;
    - ``metric_mean``: Hodge-star-weighted mean over cells;
    - ``metric_sum``: Hodge-star-weighted physical sum over cells.
    """

    fields: tuple[str, ...]
    component: DomainComponent
    sampling: PointSampling
    weight: Array
    label: str | None
    over: None
    reduction: CochainMetricReduction
    degree: int
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction]
    sampling_mode: str
    fixed_batch: GraphBatch | None

    def __init__(
        self,
        *,
        component: DomainComponent,
        residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
        sampling: PointSampling,
        fields: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        reduction: CochainMetricReduction = "graph_mean",
        sampling_mode: str = "resample",
        fixed_batch: GraphBatch | None = None,
        fixed_batch_key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(component, DomainComponent):
            raise TypeError("CochainResidualTerm requires one DomainComponent.")
        if component.where or component.where_all is not None or component.weight_all is not None:
            raise ValueError(
                "CochainResidualTerm does not accept component filters or weights; "
                "encode masks in CochainCells and scalar weighting in the term."
            )
        if reduction not in ("graph_mean", "metric_mean", "metric_sum"):
            raise ValueError(
                "reduction must be 'graph_mean', 'metric_mean', or 'metric_sum'."
            )
        mode = str(sampling_mode).lower()
        if mode not in ("resample", "fixed"):
            raise ValueError("sampling_mode must be either 'resample' or 'fixed'.")
        if mode == "resample" and fixed_batch is not None:
            raise ValueError("fixed_batch is only valid when sampling_mode='fixed'.")

        self.fields = () if fields is None else tuple(fields)
        self.component = component
        if not isinstance(sampling, PointSampling):
            raise TypeError("CochainResidualTerm requires PointSampling.")
        self.sampling = sampling
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        self.over = None
        self.reduction = reduction
        self.degree = _component_degree(component)
        self.residual = residual
        self.sampling_mode = mode
        self.fixed_batch = (
            self._sample_once(key=fixed_batch_key)
            if mode == "fixed" and fixed_batch is None
            else fixed_batch
        )

    @classmethod
    def from_operator(
        cls,
        *,
        component: DomainComponent,
        operator: Callable[..., DomainFunction],
        fields: str | Sequence[str],
        sampling: PointSampling,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        reduction: CochainMetricReduction = "graph_mean",
        sampling_mode: str = "resample",
        fixed_batch: GraphBatch | None = None,
        fixed_batch_key: Key[Array, ""] = DOC_KEY0,
    ) -> "CochainResidualTerm":
        names = (
            (fields,)
            if isinstance(fields, str)
            else tuple(fields)
        )

        def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return operator(*(functions[name] for name in names))

        return cls(
            component=component,
            residual=residual,
            sampling=sampling,
            fields=names,
            weight=weight,
            label=label,
            reduction=reduction,
            sampling_mode=sampling_mode,
            fixed_batch=fixed_batch,
            fixed_batch_key=fixed_batch_key,
        )

    @classmethod
    def from_program(
        cls,
        *,
        component: DomainComponent,
        program: CochainResidualProgram,
        field_map: Mapping[str, str],
        output: str,
        sampling: PointSampling,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        reduction: CochainMetricReduction = "graph_mean",
        sampling_mode: str = "resample",
        fixed_batch: GraphBatch | None = None,
        fixed_batch_key: Key[Array, ""] = DOC_KEY0,
    ) -> "CochainResidualTerm":
        if frozenset(field_map) != frozenset(program.input_specs):
            raise ValueError(
                "field_map names must exactly match the program input schema."
            )
        output_name = str(output)
        if output_name not in program.output_specs:
            raise KeyError(f"Unknown cochain residual output {output_name!r}.")
        names_by_input = {name: str(field_map[name]) for name in program.input_specs}
        constraint_names = tuple(dict.fromkeys(names_by_input.values()))

        def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return cochain_residual_field(
                program,
                {
                    input_name: functions[field_name]
                    for input_name, field_name in names_by_input.items()
                },
                output_name,
            )

        return cls(
            component=component,
            residual=residual,
            sampling=sampling,
            fields=constraint_names,
            weight=weight,
            label=label,
            reduction=reduction,
            sampling_mode=sampling_mode,
            fixed_batch=fixed_batch,
            fixed_batch_key=fixed_batch_key,
        )


    def _sample_once(self, *, key: Key[Array, ""] = DOC_KEY0) -> GraphBatch:
        batch = self.component.sample(self.sampling, key=key)
        if not isinstance(batch, GraphBatch):
            raise TypeError("CochainResidualTerm sampling must return a GraphBatch.")
        return batch

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> GraphBatch:
        if self.sampling_mode == "fixed":
            if self.fixed_batch is None:
                raise ValueError("sampling_mode='fixed' requires a fixed GraphBatch.")
            return self.fixed_batch
        return self._sample_once(key=key)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: GraphBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        residual = self.residual(functions)
        if not isinstance(residual, DomainFunction):
            raise TypeError("Cochain residual factories must return a DomainFunction.")
        spec = cochain_field_spec(residual)
        if spec.degree != self.degree:
            raise ValueError(
                f"Residual degree {spec.degree} does not match CochainCells degree "
                f"{self.degree}."
            )

        selected = self.sample(key=key) if batch is None else batch
        if not isinstance(selected, GraphBatch):
            raise TypeError("CochainResidualTerm requires a GraphBatch.")
        evaluated = residual(selected, key=key, **kwargs)
        if not isinstance(evaluated, cx.Field):
            raise TypeError("Cochain residual evaluation must return a coordax.Field.")
        values = _squared_cell_values(evaluated, selected)
        metric = _hodge_weights(selected)
        graph_field = selected.points[GRAPH_GRAPH_INDEX_KEY]
        if not isinstance(graph_field, cx.Field):
            raise TypeError("Graph batch graph indices must be a coordax.Field.")
        graph_index = jnp.asarray(graph_field.data, dtype=jnp.int32).reshape((-1,))
        if metric.shape != values.shape or graph_index.shape != values.shape:
            raise ValueError("Residual, metric, and graph-index cell shapes must agree.")

        reduced = cochain_metric_reduce(
            values,
            metric,
            graph_index,
            n_graph=int(selected.graph.n_node.shape[0]),
            reduction=self.reduction,
            segment_weight=_trajectory_segment_weights(self.component, selected),
        )
        return self.weight * jnp.asarray(reduced, dtype=float).reshape(())


__all__ = ["CochainResidualTerm", "cochain_residual_field"]
