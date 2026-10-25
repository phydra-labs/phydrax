#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private chronological material-chain action in exact power-of-two scaled arithmetic.

The action is ``Y = b * A(M)(a * x)``: ``A(M)`` is the chronological product of
the per-edit density maps ``W = M * 2**E`` on a token-slot state (see
`_material_schedule`), ``a`` is one or the reciprocal source measure and ``b``
is the bound target measure, optionally divided by a denominator. Every value
is carried as a pair ``(m, e)`` with ``m`` a normalized mantissa in the value
dtype (``|m|`` in ``[1/2, 1)``) and ``e`` an int32 exponent, so no intermediate
overflows or underflows; IEEE values are split and composed bit-exactly (JAX's
``frexp``/``ldexp`` are not used: they mishandle subnormals on some backends).

``material_chain_p`` evaluates any derivative of the scalar pairing
``Phi = sum_S <ct_S, Y_S>`` over truncated multilinear jets: each group
(``x``, ``M``, ``v``, ``d``, ``s`` and ``ct``) carries components indexed by
subsets of ``k`` directions, and outputs are jet components ``Y_S`` or
gradients ``dPhi/dg_S``. The JVP of any output adds one direction and the
transpose of any linear use adds one direction, so the primitive is closed
under every order of forward and reverse differentiation. Value-only tangents
(every other group's tangent a symbolic zero) reuse the same order directly.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping, Sequence
from math import factorial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.core import ShapedArray
from jax.dtypes import canonicalize_dtype
from jax.extend import core as jax_core
from jax.interpreters import ad, batching, mlir

from ..._execution_sampling import compiled_memory_estimate
from ..._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    prepare_replay_schedule,
)
from ...sparse import (
    EdgeRelation,
    gather_routes,
    InjectiveRouteWindows,
    route_reduce,
    route_replace,
)


# Exponent of a zero pair; real exponents stay below 2**29 in magnitude.
ZERO_EXPONENT = -(1 << 30)
_EXPONENT_LIMIT = 1 << 29

_GROUPS = ("x", "M", "v", "d", "s", "ct")
_GRADIENT_GROUPS = ("x", "M", "v", "d", "s")

type Pair = tuple[Array, Array]
type Jet = dict[int, Pair]


class _Format(NamedTuple):
    unsigned: Any
    width: int
    fraction_bits: int
    bias: int
    field: int


def _format(dtype: Any, /) -> _Format:
    info = jnp.finfo(dtype)
    width = int(info.bits)
    fraction_bits = int(info.nmant)
    exponent_bits = width - 1 - fraction_bits
    unsigned = {16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64}[width]
    return _Format(
        unsigned, width, fraction_bits, (1 << (exponent_bits - 1)) - 1, (1 << exponent_bits) - 1
    )


def split(value: Array, /) -> Pair:
    """Bit-exact ``value = m * 2**e`` with ``|m|`` in ``[1/2, 1)``, subnormals included.

    Zero gives ``(±0, ZERO_EXPONENT)``; infinities and NaN give ``(value, 0)``.
    """
    form = _format(value.dtype)
    unsigned = form.unsigned
    one = unsigned(1)
    fraction_mask = (one << form.fraction_bits) - one
    bits = lax.bitcast_convert_type(value, unsigned)
    sign = bits & (one << (form.width - 1))
    field = (bits >> form.fraction_bits) & unsigned(form.field)
    fraction = bits & fraction_mask
    lead = (form.width - 1) - lax.clz(fraction).astype(jnp.int32)
    shift = jnp.clip(form.fraction_bits - lead, 0, form.fraction_bits + 1)
    normalized = (fraction << shift.astype(unsigned)) & fraction_mask
    subnormal = field == 0
    mantissa_bits = (
        sign
        | (unsigned(form.bias - 1) << form.fraction_bits)
        | jnp.where(subnormal, normalized, fraction)
    )
    exponent = jnp.where(
        subnormal, 2 - form.bias - shift, field.astype(jnp.int32) - (form.bias - 1)
    )
    zero = subnormal & (fraction == 0)
    special = field == unsigned(form.field)
    mantissa = jnp.where(
        zero | special, value, lax.bitcast_convert_type(mantissa_bits, value.dtype)
    )
    return mantissa, jnp.where(
        zero, ZERO_EXPONENT, jnp.where(special, 0, exponent)
    ).astype(jnp.int32)


def compose(mantissa: Array, exponent: Array, /) -> Array:
    """Bit-exact round-to-nearest-even ``m * 2**e`` with gradual underflow and overflow."""
    form = _format(mantissa.dtype)
    unsigned = form.unsigned
    one = unsigned(1)
    fraction_mask = (one << form.fraction_bits) - one
    bits = lax.bitcast_convert_type(mantissa, unsigned)
    sign = bits & (one << (form.width - 1))
    field = (bits >> form.fraction_bits) & unsigned(form.field)
    fraction = bits & fraction_mask
    significand = fraction | (one << form.fraction_bits)
    biased = jnp.clip(exponent, -_EXPONENT_LIMIT, _EXPONENT_LIMIT) - 1 + form.bias
    normal = (
        sign
        | (jnp.clip(biased, 1, form.field - 1).astype(unsigned) << form.fraction_bits)
        | fraction
    )
    shift = jnp.clip(1 - biased, 1, form.fraction_bits + 2).astype(unsigned)
    quotient = significand >> shift
    remainder = significand & ((one << shift) - one)
    halfway = one << (shift - one)
    round_up = (remainder > halfway) | (
        (remainder == halfway) & ((quotient & one) == one)
    )
    gradual = sign | (quotient + round_up.astype(unsigned))
    infinite = sign | (unsigned(form.field) << form.fraction_bits)
    result = lax.bitcast_convert_type(
        jnp.where(
            biased >= form.field,
            infinite,
            jnp.where(biased >= 1, normal, gradual),
        ),
        mantissa.dtype,
    )
    nonfinite = field == unsigned(form.field)
    vanished = (mantissa == 0) | (exponent == ZERO_EXPONENT)
    return jnp.where(
        nonfinite, mantissa, jnp.where(vanished, mantissa * 0, result)
    )


def _normalize(mantissa: Array, exponent: Array, /) -> Pair:
    """Renormalize ``mantissa * 2**exponent`` for any finite or nonfinite mantissa."""
    normalized, shift = split(mantissa)
    return normalized, jnp.where(mantissa == 0, ZERO_EXPONENT, exponent + shift)


def _multiply(left: Pair, right: Pair, /) -> Pair:
    """Product of two normalized pairs: one mantissa rounding, then exact renormalization."""
    return _normalize(left[0] * right[0], left[1] + right[1])


def _reciprocal(value: Pair, /) -> Pair:
    """``1 / value``: the normalized mantissa's reciprocal lies in ``(1, 2]``."""
    return _normalize(1 / value[0], -value[1])


def _power_of_two(shift: Array, dtype: Any, /) -> Array:
    """Exact ``2**-shift`` for ``0 <= shift <= bias - 1`` (always normal)."""
    return compose(jnp.full(shift.shape, 0.5, dtype=dtype), 1 - shift)


def _alignment_limit(dtype: Any, /) -> int:
    """Largest exponent gap kept: a lane ``|m| >= 1/2`` stays normal after scaling."""
    return _format(dtype).bias - 2


def _aligned(mantissa: Array, exponent: Array, maximum: Array, /) -> Array:
    """Scale lanes to the common exponent ``maximum``; normal lanes only.

    A finite nonzero lane more than ``bias - 2`` binades below ``maximum`` has
    magnitude below the least normal in those units and is dropped; the
    certified tables charge one least normal for every such lane.
    """
    gap = maximum - exponent
    limit = _alignment_limit(mantissa.dtype)
    kept = (mantissa != 0) & jnp.isfinite(mantissa) & (gap <= limit)
    scale = _power_of_two(jnp.clip(gap, 0, limit), mantissa.dtype)
    return jnp.where(
        kept, mantissa * scale, jnp.where(jnp.isfinite(mantissa), 0, mantissa)
    )


def _live_exponent(value: Pair, /) -> Array:
    return jnp.where(
        (value[0] != 0) & jnp.isfinite(value[0]), value[1], ZERO_EXPONENT
    )


def _sum_axis(value: Pair, axis: int, /) -> Pair:
    """Aligned sum of pairs along ``axis``."""
    exponents = _live_exponent(value)
    maximum = jnp.max(exponents, axis=axis, keepdims=True)
    total = jnp.sum(_aligned(value[0], value[1], maximum), axis=axis)
    return _normalize(total, jnp.squeeze(maximum, axis))


def _sum_pairs(values: Sequence[Pair], /) -> Pair:
    shape = jnp.broadcast_shapes(*(value[0].shape for value in values))
    stacked = (
        jnp.stack([jnp.broadcast_to(value[0], shape) for value in values]),
        jnp.stack([jnp.broadcast_to(value[1], shape) for value in values]),
    )
    return _sum_axis(stacked, 0)


def _lane_relation(destinations: Array, valid: Array, count: int, /) -> EdgeRelation:
    lanes = destinations.shape[0]
    return EdgeRelation(
        jnp.arange(lanes, dtype=jnp.int32),
        destinations,
        source_size=lanes,
        target_size=count,
        valid=valid,
    )


def _segment_sum(value: Pair, destinations: Array, valid: Array, count: int, /) -> Pair:
    """Aligned per-destination sums of lane pairs ``[lanes, ...]``."""
    relation = _lane_relation(destinations, valid, count)
    exponents = _live_exponent(value)
    maximum = route_reduce(relation, exponents, reduction="max")
    live = route_reduce(relation, (exponents != ZERO_EXPONENT).astype(jnp.int32))
    maximum = jnp.where(live > 0, maximum, ZERO_EXPONENT)
    lane_maximum = gather_routes(relation.transpose(), maximum)
    total = route_reduce(relation, _aligned(value[0], value[1], lane_maximum))
    return _normalize(total, maximum)


def _zero_pair(shape: tuple[int, ...], dtype: Any, /) -> Pair:
    return (
        jnp.zeros(shape, dtype=dtype),
        jnp.full(shape, ZERO_EXPONENT, dtype=jnp.int32),
    )


def _gather(relation: EdgeRelation, value: Pair, /) -> Pair:
    mantissa, exponent = gather_routes(relation, value)
    return mantissa, jnp.where(relation.valid.reshape(
        relation.valid.shape + (1,) * (exponent.ndim - 1)
    ), exponent, ZERO_EXPONENT)


def _replace(
    windows: InjectiveRouteWindows, window: Array, state: Pair, values: Pair, /
) -> Pair:
    return (
        route_replace(windows, state[0], values[0], window=window),
        route_replace(windows, state[1], values[1], window=window),
    )


# ---------------------------------------------------------------------------
# Jet algebra over subsets of directions (bit masks).


def _subsets(mask: int, /) -> Iterator[int]:
    """Every submask of ``mask``, including zero and ``mask``."""
    sub = mask
    while True:
        yield sub
        if sub == 0:
            return
        sub = (sub - 1) & mask


def _convolve(
    left: Mapping[int, Pair], right: Mapping[int, Pair], masks: Sequence[int], /
) -> Jet:
    """``out_S = sum over disjoint A | B = S of left_A * right_B``."""
    result: Jet = {}
    for mask in masks:
        terms = [
            _multiply(left[part], right[mask ^ part])
            for part in _subsets(mask)
            if part in left and mask ^ part in right
        ]
        if terms:
            result[mask] = terms[0] if len(terms) == 1 else _sum_pairs(terms)
    return result


def _cover(left: Sequence[int], right: Sequence[int], /) -> tuple[int, ...]:
    return tuple(sorted({a | b for a in left for b in right if not a & b}))


def _set_partitions(mask: int, /) -> tuple[tuple[int, ...], ...]:
    """Partitions of the directions in ``mask`` into nonempty blocks."""
    bits = [1 << index for index in range(mask.bit_length()) if mask >> index & 1]
    if not bits:
        return ((),)
    first, rest = bits[0], sum(bits[1:])
    result = []
    for part in _subsets(rest):
        for remainder in _set_partitions(rest ^ part):
            result.append((first | part, *remainder))
    return tuple(result)


def _constant(value: float, like: Array, /) -> Pair:
    return split(jnp.full(like.shape, value, dtype=like.dtype))


def _reciprocal_jets(values: Jet, order: int, /) -> Jet:
    """Jets of ``1 / s`` by Faa di Bruno over set partitions."""
    base = values[0]
    inverse = _reciprocal(base)
    powers = [inverse]
    result: Jet = {0: inverse}
    for mask in range(1, 1 << order):
        terms = []
        for partition in _set_partitions(mask):
            if not all(block in values for block in partition):
                continue
            while len(powers) <= len(partition):
                powers.append(_multiply(powers[-1], inverse))
            term = _multiply(
                _constant(
                    float((-1) ** len(partition) * factorial(len(partition))), base[0]
                ),
                powers[len(partition)],
            )
            for block in partition:
                term = _multiply(term, values[block])
            terms.append(term)
        if terms:
            result[mask] = terms[0] if len(terms) == 1 else _sum_pairs(terms)
    return result


def _reciprocal_gradient(
    values: Jet, adjoint: Jet, wanted: Sequence[int], order: int, /
) -> Jet:
    """``dPhi/ds_R`` from ``dPhi/dr_S`` for ``r = 1 / s`` jets."""
    base = values[0]
    inverse = _reciprocal(base)
    powers = [inverse]
    result: Jet = {}
    for target in wanted:
        terms = []
        for mask, cotangent in adjoint.items():
            if target & ~mask:
                continue
            for partition in _set_partitions(mask):
                blocks = list(partition)
                if target:
                    if target not in blocks:
                        continue
                    blocks.remove(target)
                if not all(block in values for block in blocks):
                    continue
                size = len(partition)
                need = size + (1 if target else 2)
                while len(powers) < need:
                    powers.append(_multiply(powers[-1], inverse))
                coefficient = (-1) ** size * factorial(size)
                if not target:
                    coefficient *= -(size + 1)
                term = _multiply(
                    _constant(float(coefficient), base[0]), powers[need - 1]
                )
                for block in blocks:
                    term = _multiply(term, values[block])
                terms.append(_multiply(term, cotangent))
        if terms:
            result[target] = terms[0] if len(terms) == 1 else _sum_pairs(terms)
    return result


# ---------------------------------------------------------------------------
# Static configuration.


class _Config(NamedTuple):
    order: int
    input_scaling: str  # "none" or "reciprocal"
    output_scaling: str  # "measure" or "measure-over-denominator"
    present: tuple[tuple[str, tuple[int, ...]], ...]
    outputs: tuple[tuple[str, int], ...]
    shapes: tuple[tuple[str, tuple[int, ...], str], ...]
    compute: str
    schedule: Any  # PyTreeDef of the schedule operands
    replay: tuple[int, int]
    # Set only by the compiled-memory preflight, which measures this kernel
    # without admitting its bytes.
    preflight: bool = False


def _config(params: Mapping[str, Any], /) -> _Config:
    return _Config(**params)


def _present(config: _Config, /) -> dict[str, tuple[int, ...]]:
    return dict(config.present)


def _shape(config: _Config, group: str, /) -> tuple[tuple[int, ...], Any]:
    for name, shape, dtype in config.shapes:
        if name == group:
            return shape, jnp.dtype(dtype)
    raise KeyError(group)


def _unpack(
    operands: Sequence[Array], config: _Config, /
) -> tuple[dict[str, dict[int, Array]], Any]:
    present = _present(config)
    position = 0
    groups: dict[str, dict[int, Array]] = {}
    for group in _GROUPS:
        groups[group] = {}
        for mask in present[group]:
            groups[group][mask] = operands[position]
            position += 1
    schedule = jax.tree_util.tree_unflatten(config.schedule, operands[position:])
    return groups, schedule


def _pack(
    groups: Mapping[str, Mapping[int, Any]], schedule_leaves: Sequence[Any], /
) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], list[Any]]:
    present = tuple(
        (group, tuple(sorted(groups.get(group, {})))) for group in _GROUPS
    )
    operands = [
        groups[group][mask] for group, masks in present for mask in masks
    ]
    return present, [*operands, *schedule_leaves]


# ---------------------------------------------------------------------------
# Evaluation.


def _closure(start: Sequence[int], weights: Sequence[int], full: int, /) -> tuple[int, ...]:
    found = set(start)
    while True:
        grown = found | {a | b for a in weights for b in found if not a & b and (a | b) <= full}
        if grown == found:
            return tuple(sorted(found))
        found = grown


def _adjoint_closure(
    start: Sequence[int], weights: Sequence[int], /
) -> tuple[int, ...]:
    found = set(start)
    while True:
        grown = found | {
            target ^ part
            for target in found
            for part in weights
            if part & target == part
        }
        if grown == found:
            return tuple(sorted(found))
        found = grown


def _chunk_weights(
    mantissas: Jet, schedule: Any, chunk: Array, /
) -> tuple[Jet, Array, EdgeRelation]:
    lanes = jax.lax.dynamic_index_in_dim(
        schedule.weight_windows.target_indices, chunk, keepdims=False
    )
    exponents = jax.lax.dynamic_index_in_dim(
        schedule.pair_exponents, chunk, keepdims=False
    )
    count = schedule.weight_windows.target_size
    valid = lanes < count
    relation = EdgeRelation(
        jnp.where(valid, lanes, 0),
        jnp.arange(lanes.shape[0], dtype=jnp.int32),
        source_size=max(count, 1),
        target_size=lanes.shape[0],
        valid=valid,
    )
    weights = {}
    for mask, value in mantissas.items():
        mantissa, exponent = _gather(relation, value)
        weights[mask] = (mantissa, jnp.where(mantissa != 0, exponent + exponents, ZERO_EXPONENT))
    return weights, valid, relation


def _chunk_lanes(schedule: Any, chunk: Array, /) -> dict[str, Any]:
    read = jax.lax.dynamic_index_in_dim(
        schedule.read_windows.target_indices, chunk, keepdims=False
    )
    write = jax.lax.dynamic_index_in_dim(
        schedule.write_windows.target_indices, chunk, keepdims=False
    )
    sources = jax.lax.dynamic_index_in_dim(schedule.pair_sources, chunk, keepdims=False)
    targets = jax.lax.dynamic_index_in_dim(schedule.pair_targets, chunk, keepdims=False)
    slots = schedule.read_windows.target_size
    read_valid = read < slots
    read_relation = EdgeRelation(
        jnp.where(read_valid, read, 0),
        jnp.arange(read.shape[0], dtype=jnp.int32),
        source_size=slots,
        target_size=read.shape[0],
        valid=read_valid,
    )
    write_valid = write < slots
    write_relation = EdgeRelation(
        jnp.where(write_valid, write, 0),
        jnp.arange(write.shape[0], dtype=jnp.int32),
        source_size=slots,
        target_size=write.shape[0],
        valid=write_valid,
    )
    return {
        "read": read_relation,
        "write": write_relation,
        "sources": sources,
        "targets": targets,
    }


def _forward_chunk(
    state: Jet,
    chunk: Array,
    mantissas: Jet,
    schedule: Any,
    masks: Sequence[int],
    /,
) -> tuple[Jet, Jet]:
    """One chunk: gather every read slot, then overwrite every written slot."""
    lanes = _chunk_lanes(schedule, chunk)
    weights, valid, _ = _chunk_weights(mantissas, schedule, chunk)
    old = {mask: _gather(lanes["read"], value) for mask, value in state.items()}
    capacity = lanes["write"].target_size
    pair_relation = EdgeRelation(
        lanes["sources"],
        jnp.arange(valid.shape[0], dtype=jnp.int32),
        source_size=lanes["read"].target_size,
        target_size=valid.shape[0],
        valid=valid,
    )
    channels = next(iter(state.values()))[0].shape[1:]
    new: Jet = {}
    for mask in masks:
        terms = []
        for part in _subsets(mask):
            rest = mask ^ part
            if part in weights and rest in old:
                lane_values = _gather(pair_relation, old[rest])
                weight = weights[part]
                terms.append(
                    _multiply(
                        (
                            weight[0].reshape(weight[0].shape + (1,) * len(channels)),
                            weight[1].reshape(weight[1].shape + (1,) * len(channels)),
                        ),
                        lane_values,
                    )
                )
        if not terms:
            continue
        stacked = (
            jnp.concatenate([term[0] for term in terms]),
            jnp.concatenate([term[1] for term in terms]),
        )
        destinations = jnp.concatenate([lanes["targets"]] * len(terms))
        lane_valid = jnp.concatenate([valid] * len(terms))
        new[mask] = _segment_sum(stacked, destinations, lane_valid, capacity)
    updated = dict(state)
    for mask, value in state.items():
        updated[mask] = _replace(
            schedule.write_windows,
            chunk,
            value,
            new.get(mask, _zero_pair(value[0].shape[:0] + (capacity,) + channels, value[0].dtype)),
        )
    return updated, old


def _reverse_chunk(
    adjoint: Jet,
    chunk: Array,
    mantissas: Jet,
    schedule: Any,
    masks: Sequence[int],
    old: Jet | None,
    weight_masks: Sequence[int],
    gradients: Jet,
    /,
) -> tuple[Jet, Jet]:
    """Transpose one chunk: read and clear written slots, then fill read slots.

    Every read slot holds zero before the fill: a token's adjoint comes only
    from its single reader, and a later token reusing the slot was cleared by
    the chunk that wrote it.
    """
    lanes = _chunk_lanes(schedule, chunk)
    weights, valid, _ = _chunk_weights(mantissas, schedule, chunk)
    written = {mask: _gather(lanes["write"], value) for mask, value in adjoint.items()}
    channels = next(iter(adjoint.values()))[0].shape[1:]
    capacity = lanes["read"].target_size
    cleared = {
        mask: _replace(
            schedule.write_windows,
            chunk,
            value,
            _zero_pair((lanes["write"].target_size,) + channels, value[0].dtype),
        )
        for mask, value in adjoint.items()
    }
    target_relation = EdgeRelation(
        lanes["targets"],
        jnp.arange(valid.shape[0], dtype=jnp.int32),
        source_size=lanes["write"].target_size,
        target_size=valid.shape[0],
        valid=valid,
    )
    lane_adjoint = {mask: _gather(target_relation, value) for mask, value in written.items()}
    expand = (1,) * len(channels)
    updated = dict(cleared)
    for mask in masks:
        terms = []
        for target, value in lane_adjoint.items():
            part = target ^ mask
            if part & mask or part not in weights:
                continue
            weight = weights[part]
            terms.append(
                _multiply(
                    (weight[0].reshape(weight[0].shape + expand), weight[1].reshape(weight[1].shape + expand)),
                    value,
                )
            )
        if not terms:
            continue
        stacked = (
            jnp.concatenate([term[0] for term in terms]),
            jnp.concatenate([term[1] for term in terms]),
        )
        destinations = jnp.concatenate([lanes["sources"]] * len(terms))
        lane_valid = jnp.concatenate([valid] * len(terms))
        filled = _segment_sum(stacked, destinations, lane_valid, capacity)
        base = updated.get(mask)
        if base is None:
            shape = next(iter(adjoint.values()))[0].shape
            base = _zero_pair(shape, next(iter(adjoint.values()))[0].dtype)
        updated[mask] = _replace(schedule.read_windows, chunk, base, filled)
    if old is not None and weight_masks:
        source_relation = EdgeRelation(
            lanes["sources"],
            jnp.arange(valid.shape[0], dtype=jnp.int32),
            source_size=capacity,
            target_size=valid.shape[0],
            valid=valid,
        )
        lane_old = {mask: _gather(source_relation, value) for mask, value in old.items()}
        exponents = jax.lax.dynamic_index_in_dim(schedule.pair_exponents, chunk, keepdims=False)
        new_gradients = dict(gradients)
        for part in weight_masks:
            terms = []
            for rest, forward in lane_old.items():
                if rest & part or (rest | part) not in lane_adjoint:
                    continue
                product = _multiply(forward, lane_adjoint[rest | part])
                flat = (
                    product[0].reshape((product[0].shape[0], -1)),
                    product[1].reshape((product[1].shape[0], -1)),
                )
                terms.append(flat)
            if not terms:
                continue
            stacked = (
                jnp.concatenate([term[0] for term in terms], axis=1),
                jnp.concatenate([term[1] for term in terms], axis=1),
            )
            lane = _sum_axis(stacked, 1)
            lane = (
                jnp.where(valid, lane[0], 0),
                jnp.where(valid & (lane[0] != 0), lane[1] + exponents, ZERO_EXPONENT),
            )
            new_gradients[part] = _replace(
                schedule.weight_windows, chunk, gradients[part], lane
            )
        gradients = new_gradients
    return updated, gradients


def _replay_profile(
    schedule: Any,
    channels: int,
    compute: Any,
    state_masks: int,
    adjoint_masks: int,
    gradient_masks: int,
    terms: int,
    /,
) -> tuple[int, int, int, int]:
    """``(steps, checkpoint, tape, fixed)`` bytes of the weight-gradient replay peak."""
    steps = max(int(schedule.pair_sources.shape[0]), 1)
    slots = int(schedule.read_windows.target_size)
    read = int(schedule.read_windows.target_indices.shape[1])
    write = int(schedule.write_windows.target_indices.shape[1])
    pairs = int(schedule.pair_sources.shape[1])
    stored = int(schedule.weight_windows.target_size)
    width = jnp.dtype(compute).itemsize + 4
    checkpoint = max(slots * channels * state_masks * width, 1)
    tape = read * channels * state_masks * width
    fixed = (
        slots * channels * (state_masks + adjoint_masks) * width
        + stored * gradient_masks * width
        + pairs * channels * (adjoint_masks + 3 * terms) * width
        + (read + write) * channels * max(state_masks, adjoint_masks) * width
    )
    return steps, checkpoint, tape, fixed


def _uniform_split(
    profile: tuple[int, int, int, int], operations_budget: int, /
) -> tuple[int, int, int, int]:
    """``(length, blocks, peak, operations)``: the least peak within the budget.

    With no split inside the budget, the fewest executed updates (one block).
    """
    steps, checkpoint, tape, fixed = profile
    best: tuple[int, int, int, int] = (steps, 1, checkpoint + steps * tape + fixed, 2 * steps)
    for count in range(1, steps + 1):
        length = -(-steps // count)
        if -(-steps // length) != count:
            continue
        operations = 2 * count * length
        peak = count * checkpoint + length * tape + fixed
        if operations <= operations_budget and (
            best[3] > operations_budget or peak < best[2]
        ):
            best = (length, count, peak, operations)
    return best


def material_chain_replay_evidence(
    schedule: Any,
    channels: int,
    compute: Any,
    replay: tuple[int, int],
    /,
    *,
    state_masks: int = 1,
    adjoint_masks: int = 1,
    gradient_masks: int = 1,
    terms: int = 1,
) -> tuple[int, int, int, int]:
    """Weight-gradient replay ``(length, blocks, logical_peak, operations)``; host only.

    The logical peak selects the split; it is not the admitted resource.
    Every live component is a ``(mantissa, int32 exponent)`` pair of width
    ``w = itemsize + 4``. With ``L`` slots, ``C`` channels, read/write/pair
    lane capacities ``R``, ``W``, ``P`` and ``P_total`` stored weights, the
    logical peak holds together: ``blocks`` checkpoints (``L C ns w`` each),
    the recompute tape of one block (``length R C ns w``), the recompute carry
    (``L C ns w``), the adjoint (``L C na w``), the weight gradients
    (``P_total nm w``) and one chunk's lanes (``P C (na + 3 terms) w + (R + W)
    C max(ns, na) w``). Execution pads to uniform blocks, so it runs
    ``2 * blocks * length`` chunk updates. This is the uniform split of least
    logical peak within ``replay``'s operation budget, by arithmetic alone:
    nothing is prepared or admitted. Execution admits its executed updates
    before the replay schedule owner splits the chunks, then admits XLA's
    compiled temporaries for the replaying kernel (the logical peak only on a
    backend without compiled memory analysis) against ``replay``'s bytes.
    """
    return _uniform_split(
        _replay_profile(
            schedule, channels, compute, state_masks, adjoint_masks, gradient_masks, terms
        ),
        replay[1],
    )


def _replay_split(
    profile: tuple[int, int, int, int], replay: tuple[int, int], /
) -> tuple[int, int, int]:
    """Uniform replay ``(length, blocks, logical_peak)`` within the operation budget.

    The least-logical-peak split's executed updates are admitted before any
    schedule preparation; `prepare_replay_schedule` then splits the chunks
    under the unchanged operation budget with that split's checkpoint
    allowance, and its longest block's executed updates are re-admitted.
    """
    steps, checkpoint, tape, fixed = profile
    length, blocks, _, operations = _uniform_split(profile, replay[1])
    _admit_operations(operations, replay)
    owner = prepare_replay_schedule(
        steps, checkpoint, AdaptiveReplayPreparationPolicy(blocks * checkpoint, replay[1])
    )
    length = max(owner.block_lengths)
    blocks = -(-steps // length)
    _admit_operations(2 * blocks * length, replay)
    return length, blocks, blocks * checkpoint + length * tape + fixed


def _admit_operations(operations: int, replay: tuple[int, int], /) -> None:
    if operations > replay[1]:
        raise ValueError(
            f"Material chain replay runs {operations} chunk updates; exceeds the schedule operation budget {replay[1]}."
        )


@functools.lru_cache(maxsize=64)
def _compiled_replay_bytes(
    params: tuple[tuple[str, Any], ...],
    operands: tuple[tuple[tuple[int, ...], str], ...],
    /,
) -> int | None:
    """XLA's temporary bytes for this replaying kernel, from the compiled-memory owner.

    The measured program is the same kernel with byte admission disabled;
    ``None`` when the backend reports no memory analysis.
    """
    compiled = _compiled.lower(
        *(jax.ShapeDtypeStruct(shape, jnp.dtype(dtype)) for shape, dtype in operands),
        params=params,
    ).compile()
    estimate = compiled_memory_estimate(compiled)
    return None if estimate is None else estimate.temporary_bytes


def _admit_replay_bytes(
    operands: Sequence[Any], params: Mapping[str, Any], logical_peak: int, replay: tuple[int, int], /
) -> None:
    """Admit the replaying kernel's compiled temporaries against ``replay``'s bytes.

    Backends without a compiled memory analysis admit the logical peak
    (checkpoints, tape, carry, adjoint, weight gradients and lanes) instead;
    the refusal names which evidence it used.
    """
    # The byte ceiling decides admission, not the program or replay split.
    # Reuse one measured executable across callers with different ceilings.
    preflight = {**params, "replay": (0, replay[1]), "preflight": True}
    measured = _compiled_replay_bytes(
        tuple(sorted(preflight.items())),
        tuple((tuple(value.shape), jnp.dtype(value.dtype).str) for value in operands),
    )
    peak = logical_peak if measured is None else measured
    if peak > replay[0]:
        evidence = (
            "logical model; no compiled memory analysis"
            if measured is None
            else "compiled temporaries"
        )
        raise ValueError(
            f"Material chain replay peak {peak} bytes ({evidence}) exceeds the checkpoint byte budget {replay[0]}."
        )


def _pad_chunks(schedule: Any, total: int, /) -> Any:
    """Append inert chunks: every lane invalid, so they read and write nothing."""

    def pad(value: Array, fill: int) -> Array:
        extra = total - value.shape[0]
        if extra <= 0:
            return value
        return jnp.concatenate(
            [value, jnp.full((extra,) + value.shape[1:], fill, dtype=value.dtype)]
        )

    return type(schedule)._padded(
        schedule,
        read=pad(schedule.read_windows.target_indices, schedule.read_windows.target_size),
        write=pad(schedule.write_windows.target_indices, schedule.write_windows.target_size),
        weights=pad(
            schedule.weight_windows.target_indices, schedule.weight_windows.target_size
        ),
        sources=pad(schedule.pair_sources, 0),
        targets=pad(schedule.pair_targets, 0),
        exponents=pad(schedule.pair_exponents, 0),
    )


def _evaluate(*operands: Array, **params: Any) -> list[Array]:
    config = _config(params)
    order = config.order
    full = (1 << order) - 1
    groups, schedule = _unpack(operands, config)
    compute = jnp.dtype(config.compute)
    pairs: dict[str, Jet] = {
        group: {mask: split(value.astype(compute)) for mask, value in components.items()}
        for group, components in groups.items()
    }
    (source_count, channels), _ = _shape(config, "x")
    (target_count, _), _ = _shape(config, "ct")
    slots = schedule.read_windows.target_size
    chunks = schedule.pair_sources.shape[0]
    wanted = {group: [] for group in ("y", *_GRADIENT_GROUPS)}
    for group, mask in config.outputs:
        wanted[group].append(mask)
    gradient_wanted = any(wanted[group] for group in _GRADIENT_GROUPS)
    every = tuple(range(full + 1))

    # Endpoint jets.
    if config.input_scaling == "reciprocal":
        scaling = _reciprocal_jets(pairs["s"], order)
        scaling_columns = {
            mask: (value[0][:, None], value[1][:, None]) for mask, value in scaling.items()
        }
        inputs = _convolve(scaling_columns, pairs["x"], every)
    else:
        scaling = {}
        inputs = dict(pairs["x"])
    if config.output_scaling == "measure-over-denominator":
        inverse = _reciprocal_jets(pairs["d"], order)
        output = _convolve(pairs["v"], inverse, every)
    else:
        inverse = {}
        output = dict(pairs["v"])
    output_columns = {
        mask: (value[0][:, None], value[1][:, None]) for mask, value in output.items()
    }

    weight_masks = tuple(sorted(pairs["M"]))
    state_masks = _closure(tuple(inputs), weight_masks, full) or (0,)

    def initial_state() -> Jet:
        state = {}
        for mask in state_masks:
            if mask in inputs:
                value = inputs[mask]
                state[mask] = (
                    jnp.concatenate(
                        [value[0], jnp.zeros((slots - source_count, channels), dtype=compute)]
                    ),
                    jnp.concatenate(
                        [
                            value[1],
                            jnp.full((slots - source_count, channels), ZERO_EXPONENT, jnp.int32),
                        ]
                    ),
                )
            else:
                state[mask] = _zero_pair((slots, channels), compute)
        return state

    final_relation = EdgeRelation(
        schedule.final_windows.target_indices[0],
        jnp.arange(target_count, dtype=jnp.int32),
        source_size=slots,
        target_size=target_count,
    )
    need_forward = bool(wanted["y"] or wanted["v"] or wanted["d"])
    replay_weights = bool(wanted["M"])
    results: dict[tuple[str, int], Pair] = {}

    def run_forward(state: Jet) -> Jet:
        def body(carry: Jet, chunk: Array) -> tuple[Jet, None]:
            carry, _ = _forward_chunk(carry, chunk, pairs["M"], schedule, state_masks)
            return carry, None

        return lax.scan(body, state, jnp.arange(chunks, dtype=jnp.int32))[0]

    final: Jet = {}
    if need_forward and not replay_weights:
        state = run_forward(initial_state())
        final = {mask: _gather(final_relation, value) for mask, value in state.items()}
    if wanted["y"]:
        values = _convolve(output_columns, final, wanted["y"]) if final else {}
        for mask in wanted["y"]:
            results[("y", mask)] = values.get(mask, _zero_pair((target_count, channels), compute))

    if gradient_wanted:
        cotangents = pairs["ct"]
        # dPhi/dzf_B = sum over A disjoint B of b_A ct_{A | B}.
        seed_masks = sorted(
            {
                target ^ part
                for target in cotangents
                for part in output
                if part & target == part
            }
        )
        seeds: Jet = {}
        for mask in seed_masks:
            terms = [
                _multiply(output_columns[target ^ mask], cotangents[target])
                for target in cotangents
                if target & mask == mask and (target ^ mask) in output_columns
            ]
            if terms:
                seeds[mask] = terms[0] if len(terms) == 1 else _sum_pairs(terms)
        # An absent adjoint still carries one canonical-zero component, so a
        # structurally zero transpose returns exact zeros.
        adjoint_masks = _adjoint_closure(tuple(seeds), weight_masks) or (0,)
        final_windows = schedule.final_windows

        def initial_adjoint() -> Jet:
            adjoint = {}
            for mask in adjoint_masks:
                base = _zero_pair((slots, channels), compute)
                if mask in seeds:
                    base = _replace(final_windows, jnp.int32(0), base, seeds[mask])
                adjoint[mask] = base
            return adjoint

        gradient_state = {
            mask: _zero_pair((schedule.weight_windows.target_size,), compute)
            for mask in wanted["M"]
        }
        if replay_weights:
            profile = _replay_profile(
                schedule,
                channels,
                compute,
                len(state_masks),
                len(adjoint_masks),
                len(wanted["M"]),
                len(weight_masks),
            )
            length, blocks, logical_peak = _replay_split(profile, config.replay)
            if not config.preflight:
                _admit_replay_bytes(operands, params, logical_peak, config.replay)
            padded = _pad_chunks(schedule, length * blocks)

            def forward_block(carry: Jet, block: Array) -> tuple[Jet, Jet]:
                def inner(state: Jet, step: Array) -> tuple[Jet, None]:
                    state, _ = _forward_chunk(
                        state, block * length + step, pairs["M"], padded, state_masks
                    )
                    return state, None

                end = lax.scan(inner, carry, jnp.arange(length, dtype=jnp.int32))[0]
                return end, carry

            end, checkpoints = lax.scan(
                forward_block, initial_state(), jnp.arange(blocks, dtype=jnp.int32)
            )
            final = {mask: _gather(final_relation, value) for mask, value in end.items()}
            if wanted["y"]:
                values = _convolve(output_columns, final, wanted["y"])
                for mask in wanted["y"]:
                    results[("y", mask)] = values.get(
                        mask, _zero_pair((target_count, channels), compute)
                    )

            def reverse_block(
                carry: tuple[Jet, Jet], block: tuple[Array, Jet]
            ) -> tuple[tuple[Jet, Jet], None]:
                index, start = block

                def inner(state: Jet, step: Array) -> tuple[Jet, Jet]:
                    return _forward_chunk(
                        state, index * length + step, pairs["M"], padded, state_masks
                    )

                _, olds = lax.scan(inner, start, jnp.arange(length, dtype=jnp.int32))

                def backward(
                    inner_carry: tuple[Jet, Jet], step: tuple[Array, Jet]
                ) -> tuple[tuple[Jet, Jet], None]:
                    position, old = step
                    adjoint, gradients = inner_carry
                    return (
                        _reverse_chunk(
                            adjoint,
                            index * length + position,
                            pairs["M"],
                            padded,
                            adjoint_masks,
                            old,
                            wanted["M"],
                            gradients,
                        ),
                        None,
                    )

                carry, _ = lax.scan(
                    backward,
                    carry,
                    (jnp.arange(length, dtype=jnp.int32), olds),
                    reverse=True,
                )
                return carry, None

            (adjoint, gradient_state), _ = lax.scan(
                reverse_block,
                (initial_adjoint(), gradient_state),
                (jnp.arange(blocks, dtype=jnp.int32), checkpoints),
                reverse=True,
            )
        else:

            def backward(
                carry: tuple[Jet, Jet], chunk: Array
            ) -> tuple[tuple[Jet, Jet], None]:
                adjoint, gradients = carry
                return (
                    _reverse_chunk(
                        adjoint, chunk, pairs["M"], schedule, adjoint_masks, None, (), gradients
                    ),
                    None,
                )

            (adjoint, gradient_state), _ = lax.scan(
                backward,
                (initial_adjoint(), gradient_state),
                jnp.arange(chunks, dtype=jnp.int32),
                reverse=True,
            )
        for mask in wanted["M"]:
            results[("M", mask)] = gradient_state[mask]
        source_adjoint = {
            mask: (value[0][:source_count], value[1][:source_count])
            for mask, value in adjoint.items()
        }
        # Input endpoint: u = a * x.
        if wanted["x"]:
            if config.input_scaling == "reciprocal":
                for mask in wanted["x"]:
                    terms = [
                        _multiply(scaling_columns[target ^ mask], source_adjoint[target])
                        for target in source_adjoint
                        if target & mask == mask and (target ^ mask) in scaling_columns
                    ]
                    results[("x", mask)] = (
                        _sum_pairs(terms) if len(terms) > 1 else terms[0]
                    ) if terms else _zero_pair((source_count, channels), compute)
            else:
                for mask in wanted["x"]:
                    results[("x", mask)] = source_adjoint.get(
                        mask, _zero_pair((source_count, channels), compute)
                    )
        if wanted["s"]:
            scaling_adjoint = _channel_contraction(pairs["x"], source_adjoint, every)
            gradient = _reciprocal_gradient(pairs["s"], scaling_adjoint, wanted["s"], order)
            for mask in wanted["s"]:
                results[("s", mask)] = gradient.get(mask, _zero_pair((source_count,), compute))
        if wanted["v"] or wanted["d"]:
            output_adjoint = _channel_contraction(final, cotangents, every)
            if config.output_scaling == "measure-over-denominator":
                for mask in wanted["v"]:
                    terms = [
                        _multiply(inverse[target ^ mask], output_adjoint[target])
                        for target in output_adjoint
                        if target & mask == mask and (target ^ mask) in inverse
                    ]
                    results[("v", mask)] = (
                        _sum_pairs(terms) if len(terms) > 1 else terms[0]
                    ) if terms else _zero_pair((target_count,), compute)
                if wanted["d"]:
                    inverse_adjoint: Jet = {}
                    for mask in every:
                        terms = [
                            _multiply(pairs["v"][target ^ mask], output_adjoint[target])
                            for target in output_adjoint
                            if target & mask == mask and (target ^ mask) in pairs["v"]
                        ]
                        if terms:
                            inverse_adjoint[mask] = (
                                _sum_pairs(terms) if len(terms) > 1 else terms[0]
                            )
                    gradient = _reciprocal_gradient(
                        pairs["d"], inverse_adjoint, wanted["d"], order
                    )
                    for mask in wanted["d"]:
                        results[("d", mask)] = gradient.get(
                            mask, _zero_pair((target_count,), compute)
                        )
            else:
                for mask in wanted["v"]:
                    results[("v", mask)] = output_adjoint.get(
                        mask, _zero_pair((target_count,), compute)
                    )
    outputs = []
    for group, mask in config.outputs:
        value = results[(group, mask)]
        shape, dtype = _output_aval(config, group)
        composed = compose(value[0], value[1])
        outputs.append(composed.astype(dtype).reshape(shape))
    return outputs


def _channel_contraction(left: Jet, adjoint: Jet, masks: Sequence[int], /) -> Jet:
    """``out_A = sum over B disjoint A and channels of left_B * adjoint_{A | B}``."""
    result: Jet = {}
    for mask in masks:
        terms = []
        for rest, value in left.items():
            if rest & mask or (rest | mask) not in adjoint:
                continue
            product = _multiply(value, adjoint[rest | mask])
            terms.append(
                (
                    product[0].reshape((product[0].shape[0], -1)),
                    product[1].reshape((product[1].shape[0], -1)),
                )
            )
        if terms:
            stacked = (
                jnp.concatenate([term[0] for term in terms], axis=1),
                jnp.concatenate([term[1] for term in terms], axis=1),
            )
            result[mask] = _sum_axis(stacked, 1)
    return result


def _output_aval(config: _Config, group: str, /) -> tuple[tuple[int, ...], Any]:
    if group == "y":
        shape, _ = _shape(config, "ct")
        return shape, jnp.dtype(config.compute)
    return _shape(config, group)


# ---------------------------------------------------------------------------
# Primitive registration and rules.

material_chain_p = jax_core.Primitive("phydrax_material_chain")
material_chain_p.multiple_results = True


@functools.partial(jax.jit, static_argnames=("params",))
def _compiled(*operands: Array, params: tuple[tuple[str, Any], ...]) -> list[Array]:
    return _evaluate(*operands, **dict(params))


def _impl(*operands: Array, **params: Any) -> list[Array]:
    return _compiled(*operands, params=tuple(sorted(params.items())))


def _abstract(*operands: ShapedArray, **params: Any) -> list[ShapedArray]:
    config = _config(params)
    return [
        ShapedArray(*_output_aval(config, group)) for group, _ in config.outputs
    ]


material_chain_p.def_impl(_impl)
material_chain_p.def_abstract_eval(_abstract)
mlir.register_lowering(
    material_chain_p, mlir.lower_fun(_evaluate, multiple_results=True)
)


def _bind(
    groups: Mapping[str, Mapping[int, Array]],
    schedule_leaves: Sequence[Array],
    params: Mapping[str, Any],
    /,
    **changes: Any,
) -> list[Array]:
    present, operands = _pack(groups, schedule_leaves)
    return material_chain_p.bind(*operands, **{**params, "present": present, **changes})


def _split_operands(
    values: Sequence[Any], config: _Config, /
) -> tuple[dict[str, dict[int, Any]], list[Any]]:
    present = _present(config)
    position = 0
    groups: dict[str, dict[int, Any]] = {}
    for group in _GROUPS:
        groups[group] = {}
        for mask in present[group]:
            groups[group][mask] = values[position]
            position += 1
    return groups, list(values[position:])


def _depends_on_values(group: str, /) -> bool:
    """Outputs linear in ``x``: every one except ``dPhi/dx``."""
    return group != "x"


def _jvp(
    primals: Sequence[Array], tangents: Sequence[Array | ad.Zero], **params: Any
) -> tuple[list[Array], list[Array | ad.Zero]]:
    config = _config(params)
    outputs = material_chain_p.bind(*primals, **params)
    groups, leaves = _split_operands(primals, config)
    tangent_groups, _ = _split_operands(tangents, config)
    active = {
        group: {mask: value for mask, value in components.items() if not isinstance(value, ad.Zero)}
        for group, components in tangent_groups.items()
    }
    avals = [ShapedArray(*_output_aval(config, group)) for group, _ in config.outputs]
    if not any(active[group] for group in ("M", "v", "d", "s")):
        # Value or cotangent tangents only: every output is linear in each, so
        # the tangent is the same order with that group replaced.
        result: list[Array | ad.Zero] = [ad.Zero(aval) for aval in avals]
        for group in ("x", "ct"):
            if not active[group]:
                continue
            selected = [
                index
                for index, (output, _) in enumerate(config.outputs)
                if (group == "x" and _depends_on_values(output))
                or (group == "ct" and output != "y")
            ]
            if not selected:
                continue
            replaced = {**groups, group: active[group]}
            values = _bind(
                replaced,
                leaves,
                params,
                outputs=tuple(config.outputs[index] for index in selected),
            )
            for index, value in zip(selected, values, strict=True):
                current = result[index]
                result[index] = value if isinstance(current, ad.Zero) else current + value
        return outputs, result
    direction = 1 << config.order
    extended: dict[str, dict[int, Array]] = {}
    for group in _GROUPS:
        components: dict[int, Array] = {}
        if group == "ct":
            for mask, value in groups[group].items():
                components[mask | direction] = value
            for mask, value in active[group].items():
                components[mask] = value
        else:
            components.update(groups[group])
            for mask, value in active[group].items():
                components[mask | direction] = value
        extended[group] = components
    tangent_outputs = tuple(
        ("y", mask | direction) if group == "y" else (group, mask)
        for group, mask in config.outputs
    )
    values = _bind(
        extended, leaves, params, order=config.order + 1, outputs=tangent_outputs
    )
    return outputs, list(values)


def _transpose(
    cotangents: Sequence[Array | ad.Zero], *operands: Any, **params: Any
) -> list[Array | ad.Zero | None]:
    config = _config(params)
    groups, leaves = _split_operands(operands, config)
    if any(ad.is_undefined_primal(leaf) for leaf in leaves):
        raise ValueError("Material-chain schedule arrays are not linear operands.")
    direction = 1 << config.order
    extended: dict[str, dict[int, Array]] = {group: {} for group in _GROUPS}
    linear: list[tuple[str, int]] = []
    for group in _GROUPS:
        for mask, value in groups[group].items():
            if ad.is_undefined_primal(value):
                linear.append((group, mask))
            elif group == "ct":
                extended[group][mask | direction] = value
            else:
                extended[group][mask] = value
    any_cotangent = False
    for (group, mask), cotangent in zip(config.outputs, cotangents, strict=True):
        if isinstance(cotangent, ad.Zero):
            continue
        any_cotangent = True
        if group == "y":
            extended["ct"][mask] = cotangent
        else:
            extended[group][mask | direction] = cotangent
    result: list[Array | ad.Zero | None] = [None] * len(operands)
    positions = {}
    position = 0
    present = _present(config)
    for group in _GROUPS:
        for mask in present[group]:
            positions[(group, mask)] = position
            position += 1
    if not linear:
        return result
    if not any_cotangent:
        for key in linear:
            result[positions[key]] = ad.Zero(operands[positions[key]].aval)
        return result
    requested = tuple(
        ("y", mask | direction) if group == "ct" else (group, mask)
        for group, mask in linear
    )
    values = _bind(
        extended, leaves, params, order=config.order + 1, outputs=requested
    )
    for key, value in zip(linear, values, strict=True):
        result[positions[key]] = value
    return result


def _batch(
    operands: Sequence[Array], axes: Sequence[int | None], **params: Any
) -> tuple[list[Array], list[int]]:
    """Map the primitive over the batch axis; every batched operand is sliced per entry."""
    size = next(
        operand.shape[axis] for operand, axis in zip(operands, axes, strict=True) if axis is not None
    )
    moved = [
        operand if axis is None else jnp.moveaxis(operand, axis, 0)
        for operand, axis in zip(operands, axes, strict=True)
    ]
    batched = [axis is not None for axis in axes]

    def one(index: Array) -> list[Array]:
        selected = [
            operand[index] if flag else operand
            for operand, flag in zip(moved, batched, strict=True)
        ]
        return material_chain_p.bind(*selected, **params)

    stacked = lax.map(one, jnp.arange(size))
    return list(stacked), [0] * len(stacked)


ad.primitive_jvps[material_chain_p] = _jvp
ad.primitive_transposes[material_chain_p] = _transpose
batching.primitive_batchers[material_chain_p] = _batch


# ---------------------------------------------------------------------------
# Public helpers for the FV owner.


def _real_channels(values: Array, rows: int, /) -> tuple[Array, Any]:
    """``[rows, *payload]`` real or complex values as real channels ``[rows, C]``."""
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        stacked = jnp.stack(
            [jnp.real(values), jnp.imag(values)], axis=-1
        ).reshape((rows, -1))
        return stacked, values.dtype
    return values.reshape((rows, -1)), values.dtype


def _restore(channels: Array, rows: int, payload: tuple[int, ...], dtype: Any, /) -> Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        pairs = channels.reshape((rows, *payload, 2))
        return lax.complex(pairs[..., 0], pairs[..., 1]).astype(dtype)
    return channels.reshape((rows, *payload))


def _action(
    schedule: Any,
    mantissas: Array,
    measures: Array,
    values: Array,
    /,
    *,
    denominators: Array | None,
    source_measures: Array | None,
    transpose: bool,
    replay: tuple[int, int],
) -> Array:
    if denominators is not None and source_measures is not None:
        raise ValueError(
            "material chain action takes target denominators or source measures, not both"
        )
    rows = schedule.source_count if not transpose else schedule.target_count
    out_rows = schedule.target_count if not transpose else schedule.source_count
    payload = tuple(values.shape[1:])
    real, dtype = _real_channels(values, rows)
    if not jnp.issubdtype(real.dtype, jnp.floating):
        real = real.astype(canonicalize_dtype(np.float64))
    compute = real.dtype
    channels = real.shape[1]
    leaves, treedef = jax.tree_util.tree_flatten(schedule)
    groups: dict[str, dict[int, Array]] = {group: {} for group in _GROUPS}
    # Every numeric operand executes in the compute dtype; the casts sit
    # outside the primitive, so AD carries cotangents back to stored dtypes.
    groups["M"][0] = mantissas.astype(compute)
    groups["v"][0] = measures.astype(compute)
    if denominators is not None:
        groups["d"][0] = denominators.astype(compute)
    if source_measures is not None:
        groups["s"][0] = source_measures.astype(compute)
    shapes = (
        ("x", (schedule.source_count, channels), str(compute)),
        ("M", tuple(mantissas.shape), str(compute)),
        ("v", tuple(measures.shape), str(compute)),
        ("d", (schedule.target_count,), str(compute)),
        ("s", (schedule.source_count,), str(compute)),
        ("ct", (schedule.target_count, channels), str(compute)),
    )
    if transpose:
        groups["ct"][0] = real
        outputs = (("x", 0),)
    else:
        groups["x"][0] = real
        outputs = (("y", 0),)
    present, operands = _pack(groups, leaves)
    (result,) = material_chain_p.bind(
        *operands,
        order=0,
        input_scaling="none" if source_measures is None else "reciprocal",
        output_scaling="measure" if denominators is None else "measure-over-denominator",
        present=present,
        outputs=outputs,
        shapes=shapes,
        compute=str(compute),
        schedule=treedef,
        replay=replay,
    )
    return _restore(result, out_rows, payload, dtype)


def material_chain_apply(
    schedule: Any,
    mantissas: Array,
    measures: Array,
    values: Array,
    /,
    *,
    denominators: Array | None = None,
    source_measures: Array | None = None,
    replay: tuple[int, int],
) -> Array:
    """``measures / denominators * A(values)`` or ``measures * A(values / source_measures)``.

    ``values`` is ``[sources, *payload]`` real or complex; the result is
    ``[targets, *payload]``. At most one of ``denominators`` and
    ``source_measures`` is given; with neither, the result is
    ``measures * A(values)``.
    """
    return _action(
        schedule,
        mantissas,
        measures,
        values,
        denominators=denominators,
        source_measures=source_measures,
        transpose=False,
        replay=replay,
    )


def material_chain_transpose(
    schedule: Any,
    mantissas: Array,
    measures: Array,
    cotangents: Array,
    /,
    *,
    denominators: Array | None = None,
    replay: tuple[int, int],
) -> Array:
    """Algebraic transpose in values of `material_chain_apply` (no source measures)."""
    return _action(
        schedule,
        mantissas,
        measures,
        cotangents,
        denominators=denominators,
        source_measures=None,
        transpose=True,
        replay=replay,
    )


__all__ = [
    "ZERO_EXPONENT",
    "compose",
    "material_chain_apply",
    "material_chain_p",
    "material_chain_replay_evidence",
    "material_chain_transpose",
    "split",
]
