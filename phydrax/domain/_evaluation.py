#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from .._doc import DOC_KEY0
from .._strict import StrictModule
from ._structure import GridBatch, PointBatch


@dataclass(frozen=True, slots=True)
class FunctionBinding:
    """Explicit runtime-context contract for a pointwise domain function."""

    pass_key: bool = False
    pass_iter: bool = False

    def call(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        /,
        *,
        key: Any,
        iter_: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        call_kwargs = kwargs
        if (self.pass_key and key is not None) or (
            self.pass_iter and iter_ is not None
        ):
            call_kwargs = dict(kwargs)
            if self.pass_key and key is not None:
                call_kwargs["key"] = key
            if self.pass_iter and iter_ is not None:
                call_kwargs["iter_"] = iter_
        return function(*args, **call_kwargs)


class PointwiseEvaluator(StrictModule):
    """Typed adapter for a callable evaluated over domain coordinate points."""

    function: Callable[..., Any]
    binding: FunctionBinding

    def __init__(
        self,
        function: Callable[..., Any],
        /,
        *,
        binding: FunctionBinding | None = None,
    ):
        if not callable(function):
            raise TypeError("PointwiseEvaluator.function must be callable.")
        if binding is not None and not isinstance(binding, FunctionBinding):
            raise TypeError("PointwiseEvaluator.binding must be a FunctionBinding.")
        self.function = function
        self.binding = FunctionBinding() if binding is None else binding

    def __call__(self, *args: Any, key=None, iter_=None, **kwargs: Any) -> Any:
        coordinate_positions = tuple(
            index for index, arg in enumerate(args) if isinstance(arg, tuple)
        )
        if not coordinate_positions:
            return self.binding.call(
                self.function,
                args,
                key=key,
                iter_=iter_,
                kwargs=kwargs,
            )

        coordinate_values = tuple(
            jnp.asarray(coordinate).reshape((-1,))
            for index in coordinate_positions
            for coordinate in args[index]
        )
        if not coordinate_values:
            return self.binding.call(
                self.function,
                args,
                key=key,
                iter_=iter_,
                kwargs=kwargs,
            )

        def call_point(*values: Any) -> Any:
            point_args = list(args)
            offset = 0
            for index in coordinate_positions:
                count = len(args[index])
                point_args[index] = jnp.stack(values[offset : offset + count])
                offset += count
            return self.binding.call(
                self.function,
                tuple(point_args),
                key=key,
                iter_=iter_,
                kwargs=kwargs,
            )

        mapped = call_point
        for position in reversed(range(len(coordinate_values))):
            in_axes = tuple(
                0 if index == position else None
                for index in range(len(coordinate_values))
            )
            mapped = jax.vmap(mapped, in_axes=in_axes, out_axes=0)
        return mapped(*coordinate_values)


class BatchEvaluator(abc.ABC):
    """Callable protocol for evaluators that consume a structured domain batch."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise TypeError(
            "This evaluator requires structured batch evaluation via __call_batch__."
        )

    @abc.abstractmethod
    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        raise NotImplementedError


class AxisBatchEvaluator(abc.ABC):
    """Protocol for models that evaluate a complete named-axis batch directly."""

    @abc.abstractmethod
    def __call_axis_batch__(
        self,
        batch: Any,
        deps: tuple[str, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        raise NotImplementedError


def resolve_batch_evaluator(evaluator: Callable, /) -> BatchEvaluator | None:
    """Return an evaluator that explicitly implements the batch protocol."""
    return evaluator if isinstance(evaluator, BatchEvaluator) else None


def _first_field_leaf(tree: PyTree[Any]) -> cx.Field:
    leaves = jax.tree_util.tree_leaves(tree, is_leaf=lambda x: isinstance(x, cx.Field))
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            return leaf
    raise ValueError("Expected at least one coordax.Field leaf.")


def _unwrap_fields_to_data(tree: PyTree[Any]) -> PyTree[Any]:
    return jax.tree_util.tree_map(
        lambda x: x.data if isinstance(x, cx.Field) else x,
        tree,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )


def _axis_size(points: Mapping[str, PyTree[Any]], axis: str, /) -> int:
    leaves = jax.tree_util.tree_leaves(points, is_leaf=lambda x: isinstance(x, cx.Field))
    for leaf in leaves:
        if isinstance(leaf, cx.Field) and axis in leaf.named_shape:
            return int(leaf.named_shape[axis])
    raise ValueError(f"Cannot infer size for axis {axis!r} from points.")


def _reorder_named_axes(field: cx.Field, axis_order: tuple[str, ...]) -> cx.Field:
    dims = field.dims
    if not dims:
        return field
    named_dims = [dim for dim in dims if dim is not None]
    if not named_dims:
        return field
    target_named = [dim for dim in axis_order if dim in named_dims]
    target_named.extend(dim for dim in named_dims if dim not in axis_order)
    index_by_dim = {dim: i for i, dim in enumerate(dims) if dim is not None}
    permutation = [index_by_dim[dim] for dim in target_named]
    permutation.extend(i for i, dim in enumerate(dims) if dim is None)
    if permutation == list(range(len(dims))):
        return field
    return cx.Field(
        jnp.transpose(jnp.asarray(field.data), permutation),
        dims=tuple(dims[i] for i in permutation),
    )


def _singleton_axis_for_label(structure: Any, label: str, /) -> str | None:
    axis_names = structure.axis_names
    if axis_names is None:
        return None
    for block, axis in zip(structure.blocks, axis_names, strict=True):
        if label in block:
            return axis if len(block) == 1 else None
    return None


def _dedupe_axes(axis_names: Sequence[str], /) -> tuple[str, ...]:
    return tuple(dict.fromkeys(axis_names))


def _as_blockwise_arg(value: Any, /) -> Array | tuple[Array, ...] | None:
    if isinstance(value, cx.Field):
        return jnp.asarray(value.data)
    if isinstance(value, tuple) and all(isinstance(item, cx.Field) for item in value):
        return tuple(jnp.asarray(item.data) for item in value)
    return None


def _batch_parts(
    batch: Any,
    /,
) -> tuple[
    Mapping[str, PyTree[Any]],
    Any | None,
    Any | None,
    Mapping[str, tuple[str, ...]] | None,
]:
    from .graph._batch import GraphBatch

    if isinstance(batch, (PointBatch, GraphBatch)):
        return batch.points, batch.structure, None, None
    if isinstance(batch, GridBatch):
        return (
            batch.points,
            None,
            batch.dense_structure,
            batch.coord_axes_by_label,
        )
    return batch, None, None, None


def _validate_domain_points(
    points: Mapping[str, PyTree[Any]],
    domain_labels: tuple[str, ...],
    /,
) -> None:
    missing = tuple(label for label in domain_labels if label not in points)
    if missing:
        raise KeyError(
            f"Missing labels {missing!r} in points; expected at least {domain_labels!r}."
        )


def _mapped_axis_prefix(
    values: Array,
    axes: Sequence[str],
    points: Mapping[str, PyTree[Any]],
    /,
) -> tuple[str, ...]:
    used: list[str] = []
    shape_index = 0
    started = False
    for axis in axes:
        if shape_index >= values.ndim:
            break
        if int(values.shape[shape_index]) == _axis_size(points, axis):
            used.append(axis)
            shape_index += 1
            started = True
            continue
        if started:
            break
    return tuple(used)


def complete_batch_axes(
    field: cx.Field,
    batch: Any,
    domain_labels: tuple[str, ...],
    /,
) -> cx.Field:
    """Broadcast a field over every sampled domain axis and restore canonical order."""
    points, structure, dense_structure, coord_axes_by_label = _batch_parts(batch)
    if dense_structure is not None:
        if dense_structure.axis_names is None:
            raise ValueError("GridBatch.dense_structure must be canonicalized.")
        out = field
        for label in domain_labels:
            coord_axes = (
                None
                if coord_axes_by_label is None
                else coord_axes_by_label.get(label)
            )
            if coord_axes is not None:
                for axis in coord_axes:
                    if axis not in out.named_dims:
                        out = out * cx.Field(
                            jnp.ones((_axis_size(points, axis),), dtype=float),
                            dims=(axis,),
                        )
                continue
            axis = dense_structure.axis_for(label)
            if axis is not None and axis not in out.named_dims:
                out = out * cx.Field(
                    jnp.ones((_axis_size(points, axis),), dtype=float),
                    dims=(axis,),
                )
        return _reorder_named_axes(out, dense_structure.axis_names)

    if structure is None:
        return field
    if structure.axis_names is None:
        raise ValueError("PointBatch.structure must be canonicalized.")
    out = field
    for label in domain_labels:
        axis = structure.axis_for(label)
        if axis is None or axis in out.named_dims:
            continue
        source = _first_field_leaf(points[label])
        if axis not in source.named_shape:
            raise ValueError(
                f"Cannot infer size for sampling axis {axis!r} from points[{label!r}]."
            )
        out = out * cx.Field(
            jnp.ones((int(source.named_shape[axis]),), dtype=float),
            dims=(axis,),
        )
    return _reorder_named_axes(out, structure.axis_names)


def try_blockwise_evaluation(
    evaluator: Callable[..., Any],
    deps: tuple[str, ...],
    batch: Any,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> tuple[cx.Field | None, str | None]:
    """Try one model call over independent batch axes without point materialization."""
    if not deps:
        return None, "blockwise model execution requires non-empty dependencies."

    points, structure, dense_structure, coord_axes_by_label = _batch_parts(batch)
    args: list[Any] = []
    axis_order: list[str] = []

    if dense_structure is None:
        if structure is None:
            return None, "blockwise model execution requires a structured batch."
        if structure.axis_names is None:
            return None, "SampleLayout must be canonicalized for blockwise execution."
        for dep in deps:
            axis = _singleton_axis_for_label(structure, dep)
            if axis is None:
                return None, f"dependency {dep!r} is not sampled in a singleton block."
            arg = _as_blockwise_arg(points[dep])
            if arg is None:
                return None, f"dependency {dep!r} has an unsupported blockwise value."
            if isinstance(arg, tuple):
                return None, f"dependency {dep!r} unexpectedly has grid-axis values."
            args.append(arg)
            axis_order.append(axis)
    else:
        if dense_structure.axis_names is None:
            return None, "GridBatch.dense_structure must be canonicalized."
        for dep in deps:
            dep_axes = (
                None
                if coord_axes_by_label is None
                else coord_axes_by_label.get(dep)
            )
            arg = _as_blockwise_arg(points[dep])
            if arg is None:
                return None, f"dependency {dep!r} has an unsupported blockwise value."
            if dep_axes is not None:
                if not isinstance(arg, tuple):
                    return None, f"dependency {dep!r} requires grid-axis tuple values."
                args.append(arg)
                axis_order.extend(dep_axes)
                continue
            axis = _singleton_axis_for_label(dense_structure, dep)
            if axis is None:
                return None, f"dependency {dep!r} is not in a singleton dense block."
            if isinstance(arg, tuple):
                return None, f"dependency {dep!r} unexpectedly has grid-axis values."
            args.append(arg)
            axis_order.append(axis)

    axes = _dedupe_axes(axis_order)
    values = jnp.asarray(evaluator(*args, key=key, **kwargs))
    if values.ndim < len(axes):
        return None, "model output rank is smaller than its blockwise axis rank."
    return cx.Field(values, dims=axes + (None,) * (values.ndim - len(axes))), None


def evaluate_pointwise_callable(
    evaluator: Callable[..., Any],
    *,
    deps: tuple[str, ...],
    domain_labels: tuple[str, ...],
    points: Any,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: Mapping[str, Any] | None = None,
) -> cx.Field:
    """Evaluate a coordinate callable on a mapping, point batch, or grid batch."""
    call_kwargs = {} if kwargs is None else kwargs
    points_map, structure, dense_structure, coord_axes_by_label = _batch_parts(points)
    _validate_domain_points(points_map, domain_labels)

    if dense_structure is not None:
        if dense_structure.axis_names is None:
            raise ValueError("GridBatch.dense_structure must be canonicalized.")
        mapped_blocks = tuple(
            block
            for block in dense_structure.blocks
            if any(label in deps for label in block)
        )
        mapped_axes = tuple(
            dense_structure.axis_for(block[0])
            for block in mapped_blocks
        )
        if any(axis is None for axis in mapped_axes):
            raise ValueError("GridBatch dense axes must be canonicalized.")

        if not deps:
            values = jnp.asarray(evaluator(key=key, **call_kwargs))
        else:
            dep_values = tuple(_unwrap_fields_to_data(points_map[dep]) for dep in deps)

            def _call(*args: Any):
                return evaluator(*args, key=key, **call_kwargs)

            mapped = _call
            for block in reversed(mapped_blocks):
                mapped = jax.vmap(
                    mapped,
                    in_axes=tuple(0 if dep in block else None for dep in deps),
                    out_axes=0,
                )
            values = jnp.asarray(mapped(*dep_values))

        axis_order = [axis for axis in mapped_axes if axis is not None]
        if coord_axes_by_label is not None:
            for dep in deps:
                axis_order.extend(coord_axes_by_label.get(dep, ()))
        used_axes = _mapped_axis_prefix(values, axis_order, points_map)
        out = cx.Field(
            values,
            dims=used_axes + (None,) * (values.ndim - len(used_axes)),
        )
        return complete_batch_axes(out, points, domain_labels)

    if not deps:
        values = jnp.asarray(evaluator(key=key, **call_kwargs))
        out = cx.Field(values, dims=(None,) * values.ndim)
    else:
        out = cx.cmap(evaluator, out_axes="leading")(
            *(points_map[dep] for dep in deps),
            key=key,
            **call_kwargs,
        )
    if not isinstance(out, cx.Field):
        raise TypeError("DomainFunction evaluators must return a coordax.Field.")
    return complete_batch_axes(out, points, domain_labels)


def evaluate_domain_function(
    evaluator: Callable[..., Any],
    *,
    deps: tuple[str, ...],
    domain_labels: tuple[str, ...],
    points: Any,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: Mapping[str, Any] | None = None,
) -> cx.Field:
    """Evaluate one bound domain field through its declared evaluator protocol."""
    call_kwargs = {} if kwargs is None else kwargs
    batch_evaluator = resolve_batch_evaluator(evaluator)
    if batch_evaluator is not None:
        from .graph._batch import GraphBatch

        if isinstance(points, (PointBatch, GridBatch, GraphBatch)):
            out = batch_evaluator.__call_batch__(points, key=key, **call_kwargs)
            if not isinstance(out, cx.Field):
                raise TypeError("Batch evaluators must return a coordax.Field.")
            return complete_batch_axes(out, points, domain_labels)
    return evaluate_pointwise_callable(
        evaluator,
        deps=deps,
        domain_labels=domain_labels,
        points=points,
        key=key,
        kwargs=call_kwargs,
    )


__all__ = [
    "AxisBatchEvaluator",
    "BatchEvaluator",
    "FunctionBinding",
    "PointwiseEvaluator",
    "complete_batch_axes",
    "evaluate_domain_function",
    "evaluate_pointwise_callable",
    "resolve_batch_evaluator",
    "try_blockwise_evaluation",
]
