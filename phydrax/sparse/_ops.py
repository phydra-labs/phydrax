#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array
from jax.typing import ArrayLike

from .. import ein
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import AnyShape, Bool, Float, Int32, parse, Scalar
from ._execution import _sum_segments, RelationAccumulation, RelationExecutionState
from ._relation import EdgeRelation, RowRelation, SparseRelation


RouteReduction: TypeAlias = Literal["sum", "mean", "max", "min"]


def _require_prefix(
    name: str,
    array: Array,
    prefix: tuple[int, ...],
    /,
) -> tuple[int, ...]:
    if array.ndim < len(prefix) or tuple(array.shape[: len(prefix)]) != prefix:
        raise ValueError(f"{name} must begin with shape {prefix}; got {array.shape}.")
    return tuple(array.shape[len(prefix) :])


def _expanded_mask(valid: Array, payload_ndim: int, /) -> Array:
    return valid.reshape(valid.shape + (1,) * int(payload_ndim))


def mask_routes(relation: SparseRelation, values: Any, /) -> Any:
    """Make invalid route payloads numerically inert without changing their layout."""
    route_shape = relation.route_shape

    def mask_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_shape = _require_prefix("Route values", array, route_shape)
        valid = _expanded_mask(relation.valid, len(payload_shape))
        return jnp.where(valid, array, jnp.zeros((), dtype=array.dtype))

    return jtu.tree_map(mask_leaf, values)


def _gather_indexed_leaf(
    name: str,
    indices: Array,
    valid: Array,
    input_shape: tuple[int, ...],
    value: Any,
    /,
    *,
    ordered: bool,
) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix(name, array, input_shape)
    safe = jnp.where(valid, indices, 0)
    gathered = (
        jax.lax.map(lambda index: array[index], safe)
        if ordered and safe.size
        else array[safe]
    )
    mask = _expanded_mask(valid, len(payload_shape))
    return jnp.where(mask, gathered, jnp.zeros((), dtype=gathered.dtype))


def _gather_row_events(
    relation: RowRelation,
    flattened: Array,
    events: Array,
    payload_shape: tuple[int, ...],
    /,
    *,
    ordered: bool,
) -> Array:
    """Gather case-local events shaped ``(cases, targets_per_case * width)``."""
    if ordered and relation.capacity:

        def gather_case(source: Array, indices: Array) -> Array:
            # Cases are independent; sequence individual target/slot events
            # within each case, not whole rows with duplicate indices.
            return jax.lax.map(lambda index: source[index], indices)

        gathered = jax.vmap(gather_case)(flattened, events)
    else:
        case_indices = jnp.arange(relation.num_cases, dtype=jnp.int32)[:, None]
        gathered = flattened[case_indices, events]
    gathered = gathered.reshape(relation.route_shape + payload_shape)
    valid = _expanded_mask(relation.valid, len(payload_shape))
    return jnp.where(valid, gathered, jnp.zeros((), dtype=gathered.dtype))


def _gather_row_leaf(relation: RowRelation, value: Any, /, *, ordered: bool) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Row source values", array, relation.input_shape)
    flattened = array.reshape((relation.num_cases, relation.source_size) + payload_shape)
    events = jnp.where(relation.valid, relation.source_indices, 0).reshape(
        (relation.num_cases, relation.targets_per_case * relation.width)
    )
    return _gather_row_events(relation, flattened, events, payload_shape, ordered=ordered)


def _gather_row_targets(relation: RowRelation, value: Any, /, *, ordered: bool) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Row target values", array, relation.output_shape)
    targets = relation.targets_per_case
    flattened = array.reshape((relation.num_cases, targets) + payload_shape)
    rows = jnp.repeat(jnp.arange(targets, dtype=jnp.int32), relation.width)
    events = jnp.broadcast_to(rows, (relation.num_cases, rows.shape[0]))
    return _gather_row_events(relation, flattened, events, payload_shape, ordered=ordered)


def gather_routes(
    relation: SparseRelation,
    values: Any,
    /,
    *,
    accumulation: RelationAccumulation = "fast",
) -> Any:
    """Gather source payloads onto routes using the relation's canonical source axis.

    Gathered values are identical under every policy. Ordered policies gather one
    route event at a time, so the automatic transpose adds duplicate-source
    cotangents in route storage order rather than by unordered GPU scatter
    additions; ordinary JVP, VJP and mixed derivatives remain available.
    """
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    ordered = accumulation != "fast"
    if isinstance(relation, EdgeRelation):
        return jtu.tree_map(
            lambda value: _gather_indexed_leaf(
                "Edge source values",
                relation.source_indices,
                relation.valid,
                relation.input_shape,
                value,
                ordered=ordered,
            ),
            values,
        )
    if isinstance(relation, RowRelation):
        return jtu.tree_map(
            lambda value: _gather_row_leaf(relation, value, ordered=ordered), values
        )
    raise TypeError("relation must be an EdgeRelation or RowRelation.")


def _extreme_fill(dtype: jnp.dtype, reduction: RouteReduction, /) -> Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise TypeError(f"Route reduction {reduction!r} does not support complex values.")
    if jnp.issubdtype(dtype, jnp.bool_):
        return jnp.asarray(reduction == "min", dtype=dtype)
    if jnp.issubdtype(dtype, jnp.inexact):
        value = -jnp.inf if reduction == "max" else jnp.inf
        return jnp.asarray(value, dtype=dtype)
    limits = jnp.iinfo(dtype)
    value = limits.min if reduction == "max" else limits.max
    return jnp.asarray(value, dtype=dtype)


def _finalize_reduction(
    reduced: Array,
    counts: Array,
    payload_ndim: int,
    reduction: RouteReduction,
    /,
) -> Array:
    expanded_counts = counts.reshape(counts.shape + (1,) * payload_ndim)
    if reduction == "mean":
        denominator = jnp.maximum(expanded_counts, 1).astype(reduced.dtype)
        return reduced / denominator
    if reduction in ("max", "min"):
        return jnp.where(expanded_counts > 0, reduced, jnp.zeros_like(reduced))
    return reduced


def _edge_counts(relation: EdgeRelation, /) -> Array:
    """Exact number of valid routes of every edge-relation target."""
    return jax.ops.segment_sum(
        relation.valid.astype(jnp.int32),
        jnp.where(relation.valid, relation.target_indices, 0),
        relation.target_size,
    )


def _reduce_edge_leaf(
    relation: EdgeRelation,
    value: Any,
    reduction: RouteReduction,
    accumulation: RelationAccumulation,
    /,
) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Edge route values", array, relation.route_shape)
    payload_ndim = len(payload_shape)
    valid = _expanded_mask(relation.valid, payload_ndim)
    safe_target = jnp.where(relation.valid, relation.target_indices, 0)
    counts = _edge_counts(relation)
    if reduction in ("sum", "mean"):
        if accumulation == "fast":
            material = jnp.where(valid, array, jnp.zeros((), dtype=array.dtype))
            reduced = jax.ops.segment_sum(material, safe_target, relation.target_size)
        else:
            high, correction = _sum_segments(
                array,
                safe_target,
                relation.valid,
                relation.target_size,
                accumulation=accumulation,
            )
            reduced = high + correction if accumulation == "compensated" else high
    elif reduction == "max":
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jax.ops.segment_max(material, safe_target, relation.target_size)
    else:
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jax.ops.segment_min(material, safe_target, relation.target_size)
    return _finalize_reduction(reduced, counts, payload_ndim, reduction)


def _reduce_row_leaf(
    relation: RowRelation,
    value: Any,
    reduction: RouteReduction,
    accumulation: RelationAccumulation,
    /,
) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Row route values", array, relation.route_shape)
    payload_ndim = len(payload_shape)
    valid = _expanded_mask(relation.valid, payload_ndim)
    route_axis = len(relation.route_shape) - 1
    counts = jnp.sum(relation.valid, axis=-1, dtype=jnp.int32)
    if reduction in ("sum", "mean"):
        if accumulation == "fast":
            material = jnp.where(valid, array, jnp.zeros((), dtype=array.dtype))
            reduced = jnp.sum(material, axis=route_axis)
        else:
            row_count = prod(relation.output_shape)
            rows = array.reshape((row_count, relation.width) + payload_shape)
            active = relation.valid.reshape((row_count, relation.width))
            slots = jnp.zeros((relation.width,), dtype=jnp.int32)

            def reduce_row(row: Array, mask: Array) -> Array:
                high, correction = _sum_segments(
                    row, slots, mask, 1, accumulation=accumulation
                )
                total = high + correction if accumulation == "compensated" else high
                return total[0]

            reduced = jax.vmap(reduce_row)(rows, active).reshape(
                relation.output_shape + payload_shape
            )
    elif reduction == "max":
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jnp.max(material, axis=route_axis)
    else:
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jnp.min(material, axis=route_axis)
    return _finalize_reduction(reduced, counts, payload_ndim, reduction)


def route_reduce(
    relation: SparseRelation,
    values: Any,
    /,
    *,
    reduction: RouteReduction = "sum",
    accumulation: RelationAccumulation = "fast",
) -> Any:
    """Reduce route payloads only onto the relation's declared target axis.

    Ordered sum/mean accumulation follows the existing route storage order
    (within each row for a RowRelation), without sorting or preparing new state.
    The policy controls this forward reduction, not every downstream adjoint.
    An empty target's sum is an exact zero; its mean, minimum and maximum hold
    an inert zero fill that is not a statistic. ``route_support`` reports the
    route counts and the targets whose normalized statistics are defined.
    """
    reduction = parse(reduction, RouteReduction, "reduction")
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    if accumulation == "compensated" and reduction not in ("sum", "mean"):
        raise ValueError("Compensated accumulation supports sum and mean only.")
    if isinstance(relation, EdgeRelation):
        return jtu.tree_map(
            lambda value: _reduce_edge_leaf(relation, value, reduction, accumulation),
            values,
        )
    if isinstance(relation, RowRelation):
        return jtu.tree_map(
            lambda value: _reduce_row_leaf(relation, value, reduction, accumulation),
            values,
        )
    raise TypeError("relation must be an EdgeRelation or RowRelation.")


def _coefficient_array(
    relation: SparseRelation,
    coefficients: ArrayLike,
    /,
) -> Array:
    values = jnp.asarray(coefficients)
    route_ndim = len(relation.route_shape)
    if (
        values.ndim < route_ndim
        or tuple(values.shape[-route_ndim:]) != relation.route_shape
    ):
        raise ValueError(
            f"Sparse coefficients must end in route shape {relation.route_shape}; got {values.shape}."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype("float64")
    batch_ndim = values.ndim - route_ndim
    valid = relation.valid.reshape((1,) * batch_ndim + relation.valid.shape)
    return jnp.where(valid, values, jnp.zeros((), dtype=values.dtype))


def _flatten_operator_batch(
    values: Any,
    batch_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
    name: str,
    /,
) -> Any:
    count = 1
    for size in batch_shape:
        count *= size

    def flatten_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        prefix = batch_shape + event_shape
        payload_shape = _require_prefix(name, array, prefix)
        return array.reshape((count,) + event_shape + payload_shape)

    return jtu.tree_map(flatten_leaf, values)


def _restore_operator_batch(values: Any, batch_shape: tuple[int, ...], /) -> Any:
    return jtu.tree_map(
        lambda value: jnp.asarray(value).reshape(batch_shape + value.shape[1:]),
        values,
    )


def linear_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
    *,
    accumulation: RelationAccumulation = "fast",
) -> Any:
    """Apply a scalar-coefficient map with an explicit accumulation policy.

    Ordered modes also sequence source-route gathers, so their automatic
    transposes accumulate duplicate source contributions in a fixed order.
    """
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    weights = _coefficient_array(relation, coefficients)
    route_ndim = len(relation.route_shape)
    batch_shape = tuple(weights.shape[:-route_ndim])
    if batch_shape:
        flattened_weights = weights.reshape((-1,) + relation.route_shape)
        flattened_values = _flatten_operator_batch(
            values,
            batch_shape,
            relation.input_shape,
            "Batched sparse source values",
        )
        output = jax.vmap(
            lambda weight, value: linear_apply(
                relation, weight, value, accumulation=accumulation
            )
        )(flattened_weights, flattened_values)
        return _restore_operator_batch(output, batch_shape)
    gathered = gather_routes(relation, values, accumulation=accumulation)

    def weight_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_ndim = array.ndim - len(relation.route_shape)
        expanded = weights.reshape(weights.shape + (1,) * payload_ndim)
        return array * expanded

    messages = jtu.tree_map(weight_leaf, gathered)
    return route_reduce(relation, messages, accumulation=accumulation)


def _flatten_row_targets(relation: RowRelation, values: Any, /) -> Any:
    def flatten_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_shape = _require_prefix("Row target values", array, relation.output_shape)
        return array.reshape(
            (relation.num_cases * relation.targets_per_case,) + payload_shape
        )

    return jtu.tree_map(flatten_leaf, values)


def _restore_row_sources(relation: RowRelation, values: Any, /) -> Any:
    def restore_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_shape = _require_prefix(
            "Flattened row source values",
            array,
            (relation.num_cases * relation.source_size,),
        )
        return array.reshape(relation.input_shape + payload_shape)

    return jtu.tree_map(restore_leaf, values)


def _linear_reverse_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
    *,
    conjugate: bool,
) -> Any:
    weights = _coefficient_array(relation, coefficients)
    route_ndim = len(relation.route_shape)
    batch_shape = tuple(weights.shape[:-route_ndim])
    if batch_shape:
        flattened_weights = weights.reshape((-1,) + relation.route_shape)
        flattened_values = _flatten_operator_batch(
            values,
            batch_shape,
            relation.output_shape,
            "Batched sparse target values",
        )
        output = jax.vmap(
            lambda weight, value: _linear_reverse_apply(
                relation,
                weight,
                value,
                conjugate=conjugate,
            )
        )(flattened_weights, flattened_values)
        return _restore_operator_batch(output, batch_shape)
    if conjugate:
        weights = jnp.conj(weights)
    if isinstance(relation, EdgeRelation):
        return linear_apply(relation.transpose(), weights, values)
    if isinstance(relation, RowRelation):
        edge_relation = relation.as_edge_relation()
        flattened_targets = _flatten_row_targets(relation, values)
        flattened_weights = weights.reshape((-1,))
        sources = linear_apply(
            edge_relation.transpose(),
            flattened_weights,
            flattened_targets,
        )
        return _restore_row_sources(relation, sources)
    raise TypeError("relation must be an EdgeRelation or RowRelation.")


def linear_transpose_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
) -> Any:
    """Apply the algebraic transpose of a sparse linear map."""
    return _linear_reverse_apply(
        relation,
        coefficients,
        values,
        conjugate=False,
    )


def linear_adjoint_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
) -> Any:
    """Apply the conjugate adjoint of a sparse linear map."""
    return _linear_reverse_apply(
        relation,
        coefficients,
        values,
        conjugate=True,
    )


def _block_coefficient_array(
    relation: SparseRelation,
    coefficients: ArrayLike,
    /,
) -> Array:
    """Validate explicit route-major matrix fibers, never scalar batches."""
    weights = jnp.asarray(coefficients)
    route_ndim = len(relation.route_shape)
    if (
        weights.ndim != route_ndim + 2
        or weights.shape[:route_ndim] != relation.route_shape
        or min(weights.shape[-2:]) < 1
    ):
        raise ValueError(
            "Block coefficients must have shape "
            f"{relation.route_shape} + (target_fiber, source_fiber), "
            f"with positive fiber sizes; got {weights.shape}."
        )
    if not jnp.issubdtype(weights.dtype, jnp.inexact):
        weights = weights.astype(jnp.float64)
    return mask_routes(relation, weights)


def block_linear_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: ArrayLike,
    /,
) -> Array:
    """Apply route matrices to values shaped ``(*input_shape, r_s, *payload)``.

    Coefficients have shape ``(*route_shape, r_t, r_s)``. Repeated routes
    accumulate; invalid routes, including nonfinite padding, are inert.
    This explicit matrix-fiber operation does not reinterpret scalar batches.
    """
    weights = _block_coefficient_array(relation, coefficients)
    array = jnp.asarray(values)
    payload = _require_prefix(
        "Block source values", array, relation.input_shape + (weights.shape[-1],)
    )
    gathered = gather_routes(relation, array)
    route_count = relation.valid.size
    # Flatten only route and independent payload axes; preserve both fiber axes.
    messages = ein.contract(
        "roi,rip->rop",
        weights.reshape((route_count,) + weights.shape[-2:]),
        gathered.reshape((route_count, weights.shape[-1], prod(payload))),
    )
    return route_reduce(
        relation, messages.reshape(relation.route_shape + (weights.shape[-2],) + payload)
    )


def _block_linear_reverse_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: ArrayLike,
    /,
    *,
    conjugate: bool,
) -> Array:
    weights = _block_coefficient_array(relation, coefficients)
    array = jnp.asarray(values)
    payload = _require_prefix(
        "Block target values", array, relation.output_shape + (weights.shape[-2],)
    )
    reversed_weights = jnp.swapaxes(weights, -2, -1)
    if conjugate:
        reversed_weights = jnp.conj(reversed_weights)
    edge = relation if isinstance(relation, EdgeRelation) else relation.as_edge_relation()
    sources = block_linear_apply(
        edge.transpose(),
        reversed_weights.reshape(edge.route_shape + reversed_weights.shape[-2:]),
        array.reshape((edge.target_size, weights.shape[-2]) + payload),
    )
    return sources.reshape(relation.input_shape + (weights.shape[-1],) + payload)


def block_linear_transpose_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: ArrayLike,
    /,
) -> Array:
    """Apply the algebraic transpose, transposing but not conjugating fibers."""
    return _block_linear_reverse_apply(relation, coefficients, values, conjugate=False)


def block_linear_adjoint_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: ArrayLike,
    /,
) -> Array:
    """Apply the Euclidean conjugate transpose of route matrix fibers."""
    return _block_linear_reverse_apply(relation, coefficients, values, conjugate=True)


_ROUTE_OWNER_MESSAGE = (
    "relation must be an EdgeRelation, RowRelation or RelationExecutionState."
)


class RouteSupport(NonTrainableState, StrictModule):
    """Valid-route support of every relation target.

    ``counts`` holds the exact number of valid routes of each target.
    ``successful`` reports whether the executing owner reduced every target: a
    direct relation always does, while a prepared ``RelationExecutionState``
    reports its grouping admission. ``available`` marks the targets whose
    route-normalized statistics are defined, namely nonempty targets of a
    successful execution. Route means and norms hold an inert zero fill
    elsewhere; that fill owns no derivative and is not a statistic, so a
    capacity failure is never mistaken for an empty or zero-valued target.
    """

    __strict_contract__ = True

    counts: Int32[AnyShape]
    available: Bool[AnyShape]
    successful: Bool[Scalar]

    def __init__(
        self, counts: ArrayLike, available: ArrayLike, successful: ArrayLike
    ) -> None:
        counts_array = parse(jnp.asarray(counts), Int32[AnyShape], "counts")
        available_array = parse(jnp.asarray(available), Bool[AnyShape], "available")
        successful_array = parse(jnp.asarray(successful), Bool[Scalar], "successful")
        if available_array.shape != counts_array.shape:
            raise ValueError("available must have the shape of counts.")
        self.counts = counts_array
        self.available = available_array
        self.successful = successful_array


class RouteRootMeanSquare(NonTrainableState, StrictModule):
    """Root mean square of route payload norms on every relation target.

    ``mean_square`` averages each valid route's squared Euclidean payload norm
    over the target's valid routes (normalized by route count, not payload
    entries) and ``value`` is the root mean square. Both hold zero where
    ``support.available`` is false. ``value`` is evaluated on scaled payloads
    and stays representable where the unscaled ``mean_square`` underflows or
    overflows.

    ``mean_square`` is a polynomial of the route values, differentiable to every
    order everywhere. ``value`` is analytic on ``regular`` targets: available,
    with a positive finite payload scale. On an available target whose routes
    are exactly zero, the norm has directional derivatives but no linear
    derivative, so derivatives of ``value`` there are refused as NaN on that
    target's own routes, also under a zero seed because a linear derivative
    cannot depend on its seed. No epsilon is added; request ``mean_square``
    where derivatives at zero are needed. Unavailable targets are inert.
    """

    __strict_contract__ = True

    value: Float[AnyShape]
    mean_square: Float[AnyShape]
    regular: Bool[AnyShape]
    support: RouteSupport

    def __init__(
        self,
        value: ArrayLike,
        mean_square: ArrayLike,
        regular: ArrayLike,
        support: RouteSupport,
    ) -> None:
        value_array = parse(jnp.asarray(value), Float[AnyShape], "value")
        mean_square_array = parse(
            jnp.asarray(mean_square), Float[AnyShape], "mean_square"
        )
        regular_array = parse(jnp.asarray(regular), Bool[AnyShape], "regular")
        if not isinstance(support, RouteSupport):
            raise TypeError("support must be a RouteSupport.")
        shape = support.counts.shape
        if (
            value_array.shape != shape
            or mean_square_array.shape != shape
            or regular_array.shape != shape
        ):
            raise ValueError(f"Root-mean-square fields must have target shape {shape}.")
        if value_array.dtype != mean_square_array.dtype:
            raise TypeError("value and mean_square must have the same dtype.")
        self.value = value_array
        self.mean_square = mean_square_array
        self.regular = regular_array
        self.support = support


def _route_shape(relation: SparseRelation | RelationExecutionState, /) -> tuple[int, ...]:
    if isinstance(relation, RelationExecutionState):
        return (relation.route_capacity,)
    if isinstance(relation, (EdgeRelation, RowRelation)):
        return relation.route_shape
    raise TypeError(_ROUTE_OWNER_MESSAGE)


def _route_valid(relation: SparseRelation | RelationExecutionState, /) -> Array:
    if isinstance(relation, RelationExecutionState):
        return relation.relation.valid
    return relation.valid


def _masked_routes(
    relation: SparseRelation | RelationExecutionState,
    name: str,
    values: ArrayLike,
    /,
) -> Array:
    array = jnp.asarray(values)
    payload_shape = _require_prefix(name, array, _route_shape(relation))
    valid = _expanded_mask(_route_valid(relation), len(payload_shape))
    return jnp.where(valid, array, jnp.zeros((), dtype=array.dtype))


def _gather_targets(
    relation: SparseRelation | RelationExecutionState,
    values: ArrayLike,
    /,
    *,
    ordered: bool,
) -> Array:
    """Gather target payloads onto routes, the transpose of route reduction."""
    if isinstance(relation, RelationExecutionState):
        array = jnp.asarray(values)
        payload_shape = _require_prefix("Target values", array, relation.output_shape)
        edge = relation.relation
        return _gather_indexed_leaf(
            "Target values",
            edge.target_indices,
            edge.valid,
            edge.output_shape,
            array.reshape(edge.output_shape + payload_shape),
            ordered=ordered,
        )
    if isinstance(relation, EdgeRelation):
        return _gather_indexed_leaf(
            "Edge target values",
            relation.target_indices,
            relation.valid,
            relation.output_shape,
            values,
            ordered=ordered,
        )
    if isinstance(relation, RowRelation):
        return _gather_row_targets(relation, values, ordered=ordered)
    raise TypeError(_ROUTE_OWNER_MESSAGE)


def _segment_reduce(
    relation: SparseRelation | RelationExecutionState,
    values: Array,
    reduction: Literal["sum", "max"],
    accumulation: RelationAccumulation,
    /,
) -> Array:
    if isinstance(relation, RelationExecutionState):
        # Reuse the prepared canonical grouping; its admission is reported
        # through RouteSupport.successful rather than this zero-filled result.
        reduced, _ = relation.reduce(
            values, reduction=reduction, accumulation=accumulation, output="dense"
        )
        return jnp.asarray(reduced)
    return route_reduce(relation, values, reduction=reduction, accumulation=accumulation)


def route_support(relation: SparseRelation | RelationExecutionState, /) -> RouteSupport:
    """Return exact valid-route counts and normalized-statistic availability.

    Compute it once per topology and reuse it with ``route_reduce`` or
    ``RelationExecutionState.reduce`` means, whose unavailable targets hold the
    same inert zero fill. A prepared execution reports its grouping admission.
    """
    if isinstance(relation, RelationExecutionState):
        counts = _edge_counts(relation.relation).reshape(relation.output_shape)
        successful = relation.groups.evidence.successful
    elif isinstance(relation, EdgeRelation):
        counts = _edge_counts(relation)
        successful = jnp.asarray(True)
    elif isinstance(relation, RowRelation):
        counts = jnp.sum(relation.valid, axis=-1, dtype=jnp.int32)
        successful = jnp.asarray(True)
    else:
        raise TypeError(_ROUTE_OWNER_MESSAGE)
    return RouteSupport(counts, (counts > 0) & successful, successful)


def _route_dot(
    relation: SparseRelation | RelationExecutionState,
    left: ArrayLike,
    right: ArrayLike,
    accumulation: RelationAccumulation,
    /,
) -> Array:
    route_shape = _route_shape(relation)
    left_routes = _masked_routes(relation, "Left route values", left)
    right_routes = _masked_routes(relation, "Right route values", right)
    if right_routes.shape != left_routes.shape:
        raise ValueError(
            f"Right route values must have shape {left_routes.shape}; "
            f"got {right_routes.shape}."
        )
    route_count = prod(route_shape)
    width = prod(left_routes.shape[len(route_shape) :])
    if accumulation == "fast":
        products = ein.contract(
            "rp,rp->r",
            left_routes.reshape((route_count, width)),
            right_routes.reshape((route_count, width)),
        )
        return _segment_reduce(
            relation, products.reshape(route_shape), "sum", accumulation
        )
    # Ordered policies reduce each payload component over routes, then fold the
    # components of every target in C order under the same policy.
    products = (left_routes * right_routes).reshape(route_shape + (width,))
    components = _segment_reduce(relation, products, "sum", accumulation)
    high, correction = _sum_segments(
        jnp.moveaxis(components, -1, 0),
        jnp.zeros((width,), dtype=jnp.int32),
        jnp.ones((width,), dtype=jnp.bool_),
        1,
        accumulation=accumulation,
    )
    total = high + correction if accumulation == "compensated" else high
    return total[0]


def route_dot(
    relation: SparseRelation | RelationExecutionState,
    left: ArrayLike,
    right: ArrayLike,
    /,
    *,
    accumulation: RelationAccumulation = "fast",
) -> tuple[Array, RouteSupport]:
    """Sum bilinear route inner products onto every target.

    ``left`` and ``right`` are route-aligned with identical payload shapes;
    target ``t`` receives ``sum_{r -> t} sum_payload left[r] * right[r]`` over its
    valid routes. No complex conjugation is applied; pass ``jnp.conj(left)`` for
    the Hermitian product. Invalid routes are inert even with nonfinite
    payloads, and an empty target's sum is an exact zero. ``"fast"`` contracts
    each route payload densely before reducing; ordered policies reduce each
    payload component over routes in storage order (stable grouped order for a
    prepared execution) and fold components in C order under the same policy.
    A prepared execution takes route-major values shaped
    ``(route_capacity, *payload)``; its admission is in the returned support.
    """
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    return _route_dot(relation, left, right, accumulation), route_support(relation)


def route_rms(
    relation: SparseRelation | RelationExecutionState,
    values: ArrayLike,
    /,
    *,
    accumulation: RelationAccumulation = "fast",
) -> RouteRootMeanSquare:
    """Root mean square of route payload norms with support and regularity.

    Each target's routes are divided by the target's maximum absolute payload
    component, a gradient-free constant on which the norm is exactly
    homogeneous, so the root neither underflows nor overflows for tiny or huge
    payloads and every derivative order equals that of the unscaled norm. The
    scaled squares, and separately the unscaled squares of ``mean_square``, are
    summed like ``route_dot`` under ``accumulation``. See
    ``RouteRootMeanSquare`` for the zero-norm and empty-target contract;
    nonfinite active payloads yield nonfinite values.
    """
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    route_shape = _route_shape(relation)
    routes = _masked_routes(relation, "Route values", values)
    if not jnp.issubdtype(routes.dtype, jnp.inexact):
        raise TypeError("route_rms requires floating or complex route values.")
    payload_ndim = routes.ndim - len(route_shape)
    complex_values = jnp.issubdtype(routes.dtype, jnp.complexfloating)
    observed = jax.lax.stop_gradient(routes)
    magnitude = (
        jnp.maximum(jnp.abs(observed.real), jnp.abs(observed.imag))
        if complex_values
        else jnp.abs(observed)
    )
    real_dtype = magnitude.dtype
    one = jnp.ones((), dtype=real_dtype)
    zero = jnp.zeros((), dtype=real_dtype)
    route_scale = jnp.max(
        magnitude, axis=tuple(range(len(route_shape), routes.ndim)), initial=zero
    )
    scale = _segment_reduce(relation, route_scale, "max", "fast")
    safe_scale = jnp.where(scale > 0, scale, one)
    divisor = jnp.where(
        _route_valid(relation),
        _gather_targets(relation, safe_scale, ordered=False),
        one,
    )
    scaled = routes / divisor.reshape(route_shape + (1,) * payload_ndim)
    support = route_support(relation)
    available = support.available
    divisors = jnp.maximum(support.counts, 1).astype(real_dtype)

    def mean_of_squares(payload: Array, /) -> Array:
        squares = _route_dot(
            relation,
            payload,
            jnp.conj(payload) if complex_values else payload,
            accumulation,
        )
        return jnp.real(squares) / divisors

    # Unavailable targets take a unit square-root argument, so their zero fill
    # owns no derivative; available all-zero targets keep sqrt(0), whose
    # derivative is the declared refusal.
    root = safe_scale * jnp.sqrt(jnp.where(available, mean_of_squares(scaled), one))
    return RouteRootMeanSquare(
        jnp.where(available, root, zero),
        jnp.where(available, mean_of_squares(routes), zero),
        available & (scale > 0) & jnp.isfinite(scale),
        support,
    )


def route_multiply(
    relation: SparseRelation | RelationExecutionState,
    route_values: ArrayLike,
    target_values: ArrayLike,
    /,
    *,
    accumulation: RelationAccumulation = "fast",
) -> Array:
    """Multiply route payloads by their target's payload.

    Returns ``route_values[r] * target_values[target(r)]`` on valid routes and
    zero on invalid routes. The target payload shape must equal the leading
    route payload axes and broadcasts over the remaining ones (for example a
    per-target scalar over route vectors). Ordered policies gather one route
    event at a time, ordering the target cotangent accumulation.
    """
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    route_shape = _route_shape(relation)
    routes = _masked_routes(relation, "Route values", route_values)
    factors = _gather_targets(relation, target_values, ordered=accumulation != "fast")
    payload_shape = routes.shape[len(route_shape) :]
    factor_shape = factors.shape[len(route_shape) :]
    if payload_shape[: len(factor_shape)] != factor_shape:
        raise ValueError(
            f"Target payload {factor_shape} must equal the leading route payload "
            f"axes of {payload_shape}."
        )
    expanded = factors.reshape(
        factors.shape + (1,) * (len(payload_shape) - len(factor_shape))
    )
    return routes * expanded


def route_matvec(
    relation: SparseRelation | RelationExecutionState,
    target_matrices: ArrayLike,
    route_values: ArrayLike,
    /,
    *,
    accumulation: RelationAccumulation = "fast",
) -> Array:
    """Apply every target's matrix to its routes' vectors.

    ``target_matrices`` has shape ``(*target_shape, m, n)`` and ``route_values``
    has shape ``(*route_shape, n, *payload)``. The result, shaped
    ``(*route_shape, m, *payload)``, is ``matrix[target(r)] @ values[r]`` (column
    vector convention; swap the matrix axes for row vectors) on valid routes and
    zero on invalid routes. Ordered policies gather one route event at a time,
    ordering the matrix cotangent accumulation; each route's product is an
    ordinary dense contraction.
    """
    accumulation = parse(accumulation, RelationAccumulation, "accumulation")
    route_shape = _route_shape(relation)
    matrices = jnp.asarray(target_matrices)
    target_shape = relation.output_shape
    if (
        matrices.ndim != len(target_shape) + 2
        or matrices.shape[: len(target_shape)] != target_shape
        or min(matrices.shape[-2:]) < 1
    ):
        raise ValueError(
            f"Target matrices must have shape {target_shape} + (m, n) with positive "
            f"m and n; got {matrices.shape}."
        )
    rows, columns = matrices.shape[-2:]
    vectors = _masked_routes(relation, "Route values", route_values)
    payload_shape = _require_prefix("Route values", vectors, route_shape + (columns,))
    gathered = _gather_targets(relation, matrices, ordered=accumulation != "fast")
    route_count = prod(route_shape)
    product = ein.contract(
        "rij,rjp->rip",
        gathered.reshape((route_count, rows, columns)),
        vectors.reshape((route_count, columns, prod(payload_shape))),
    )
    return product.reshape(route_shape + (rows,) + payload_shape)


__all__ = [
    "RouteReduction",
    "RouteRootMeanSquare",
    "RouteSupport",
    "block_linear_adjoint_apply",
    "block_linear_apply",
    "block_linear_transpose_apply",
    "gather_routes",
    "linear_adjoint_apply",
    "linear_apply",
    "linear_transpose_apply",
    "mask_routes",
    "route_dot",
    "route_matvec",
    "route_multiply",
    "route_reduce",
    "route_rms",
    "route_support",
]
