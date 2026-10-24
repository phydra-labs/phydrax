#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Streaming APIC route moments with a native per-lane CUDA schedule.

One primitive owns the three Cartesian APIC route moments

* ``0`` velocity ``sum_r w_r v_r`` with shape ``(particle, component)``,
* ``1`` velocity gradient ``sum_r v_r g_r`` with shape
  ``(particle, component, gradient)``,
* ``2`` affine moment ``sum_r w_r v_r x_r`` with shape
  ``(particle, component, offset)``,

where ``v_r`` is the gathered grid velocity of a route. ``_reference`` is the
canonical JAX meaning: row-relation gather semantics (invalid routes gather
exact zeros, negative valid indices wrap once, remaining out-of-range valid
indices clamp to the source boundary) followed by one route-axis sum per moment.
Invalid routes zero their weight and velocity but never their gradient or offset
metadata, so ``0 * NaN`` remains NaN, and zero weights are never skipped.

On an admitted CUDA compilation target the primal lowers to one Mosaic GPU
kernel with a bounded particle/component thread schedule: every thread
owns one flattened ``(particle, component)`` lane, adjacent threads own adjacent
lanes, and each thread streams the routes of its particle in storage order into
scalar accumulators (seven for the full Cartesian moment set, four when only the
velocity and affine moment are live). Lanes beyond the last particle are
predicated off before any load or store; every gather address is selected to be
in bounds before it is issued. Other targets, dtypes and shapes lower the
reference. Derivatives always differentiate the reference, batching with any
mapped read operand maps the reference, and a dead-code rule rebinds only live
moments with unread operands removed.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src import dispatch
from jax._src.core import ShapedArray
from jax._src.interpreters import partial_eval as pe
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike

from ...backends._mosaic import lower_mosaic_or_reference


type Tangent = Array | ad.Zero
type Primal = Array | ad.UndefinedPrimal

_VELOCITY = 0
_VELOCITY_GRADIENT = 1
_AFFINE_MOMENT = 2
_ALL_OUTPUTS = (_VELOCITY, _VELOCITY_GRADIENT, _AFFINE_MOMENT)
# Operand positions: indices, valid, weights, gradients, offsets, grid_velocity.
_FLOATING_OPERANDS = (2, 3, 4, 5)
# One warpgroup per CTA; one thread per flattened (particle, component) lane.
_TILE = 128
# Bound emitted code and register pressure to the qualified local stencil sizes.
# Wider stencils and physical axes retain the ordinary JAX reference.
_MAXIMUM_ROUTES = 27
_INT32_LIMIT = (1 << 31) - 1


def _check_outputs(outputs: tuple[int, ...], /) -> None:
    if not isinstance(outputs, tuple) or any(
        type(output) is not int for output in outputs
    ):
        raise TypeError("Streaming APIC outputs must be a tuple of integer roles.")
    if (
        not outputs
        or any(output not in _ALL_OUTPUTS for output in outputs)
        or any(left >= right for left, right in zip(outputs, outputs[1:]))
    ):
        raise ValueError(
            "Streaming APIC outputs must be a nonempty increasing subset of (0, 1, 2)."
        )


def _needed(outputs: tuple[int, ...], /) -> tuple[bool, ...]:
    """Which of the six operands the requested moments read."""
    weighted = _VELOCITY in outputs or _AFFINE_MOMENT in outputs
    return (
        True,
        True,
        weighted,
        _VELOCITY_GRADIENT in outputs,
        _AFFINE_MOMENT in outputs,
        True,
    )


def _reference(
    indices: Array,
    valid: Array,
    weights: Array,
    gradients: Array,
    offsets: Array,
    grid_velocity: Array,
    *,
    outputs: tuple[int, ...],
) -> tuple[Array, ...]:
    """Canonical gather, elementwise products and route sums of live moments.

    Operands not read by ``outputs`` are never touched and may be placeholders.
    """
    # Exactly the row-relation gather of ``gather_routes`` for one case.
    safe = jnp.where(valid, indices, 0)
    gathered = grid_velocity[safe]
    route_velocity = jnp.where(
        valid[..., None], gathered, jnp.zeros((), dtype=gathered.dtype)
    )
    if _VELOCITY in outputs or _AFFINE_MOMENT in outputs:
        # One shared weighted route load feeds both velocity and affine moment.
        weighted_velocity = jnp.where(valid, weights, 0.0)[..., None] * route_velocity
    results = []
    for output in outputs:
        if output == _VELOCITY:
            results.append(jnp.sum(weighted_velocity, axis=1))
        elif output == _VELOCITY_GRADIENT:
            results.append(
                jnp.sum(route_velocity[..., :, None] * gradients[:, :, None, :], axis=1)
            )
        else:
            results.append(
                jnp.sum(weighted_velocity[..., :, None] * offsets[:, :, None, :], axis=1)
            )
    return tuple(results)


def _abstract(
    indices: ShapedArray,
    valid: ShapedArray,
    weights: ShapedArray,
    gradients: ShapedArray,
    offsets: ShapedArray,
    grid_velocity: ShapedArray,
    *,
    outputs: tuple[int, ...],
) -> tuple[ShapedArray, ...]:
    _check_outputs(outputs)
    needed = _needed(outputs)
    if indices.ndim != 2:
        raise ValueError("Streaming APIC indices must have particle and route axes.")
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError(
            "Streaming APIC indices must be a (particle, route) integer array."
        )
    if valid.shape != indices.shape:
        raise ValueError("Streaming APIC validity must have the indices shape.")
    if valid.dtype != jnp.bool_:
        raise TypeError("Streaming APIC validity must be a boolean array like indices.")
    if grid_velocity.ndim != 2:
        raise ValueError(
            "Streaming APIC grid velocity must have source and component axes."
        )
    if not jnp.issubdtype(grid_velocity.dtype, jnp.inexact):
        raise TypeError(
            "Streaming APIC grid velocity must be a (source, component) array."
        )
    dtype = grid_velocity.dtype
    if needed[2] and weights.shape != indices.shape:
        raise ValueError("Streaming APIC weights must have the indices shape.")
    if needed[2] and weights.dtype != dtype:
        raise TypeError("Streaming APIC weights must match indices and the grid dtype.")
    for position, operand in ((3, gradients), (4, offsets)):
        if not needed[position]:
            continue
        if operand.ndim != 3 or operand.shape[:2] != indices.shape:
            raise ValueError(
                "Streaming APIC gradients and offsets must have particle, route, and axis dimensions."
            )
        if operand.dtype != dtype:
            raise TypeError(
                "Streaming APIC gradients and offsets must be (particle, route, axis) "
                "arrays of the grid dtype."
            )
    particles = indices.shape[0]
    components = grid_velocity.shape[1]
    shapes = {
        _VELOCITY: (particles, components),
        _VELOCITY_GRADIENT: (particles, components, gradients.shape[-1])
        if needed[3]
        else None,
        _AFFINE_MOMENT: (particles, components, offsets.shape[-1]) if needed[4] else None,
    }
    return tuple(ShapedArray(shapes[output], dtype) for output in outputs)


def _admitted(operands: Sequence[Array], outputs: tuple[int, ...], /) -> bool:
    """Native shape/dtype profile; anything else lowers the reference."""
    indices, _, weights, gradients, offsets, grid_velocity = operands
    needed = _needed(outputs)
    dtype = grid_velocity.dtype
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        return False
    if indices.dtype != jnp.dtype(jnp.int32):
        return False
    particles, routes = indices.shape
    sources, components = grid_velocity.shape
    widths = [1]
    if needed[3]:
        widths.append(gradients.shape[-1])
    if needed[4]:
        widths.append(offsets.shape[-1])
    if (
        particles < 1
        or not 1 <= routes <= _MAXIMUM_ROUTES
        or sources < 1
        or not 1 <= components <= 3
        or min(widths) < 1
        or max(widths) > 3
    ):
        return False
    # Lane ids, gather addresses and route offsets stay in signed 32-bit range.
    return (
        particles * components <= _INT32_LIMIT - _TILE
        and sources * components <= _INT32_LIMIT
        and particles * routes * max(widths) <= _INT32_LIMIT
        and all(
            operand.dtype == dtype
            for operand, use in zip((weights, gradients, offsets), needed[2:5])
            if use
        )
    )


def _streaming_kernel(
    operands: tuple[Array, ...],
    *,
    outputs: tuple[int, ...],
    particles: int,
    routes: int,
    components: int,
    sources: int,
    gradient_width: int,
    offset_width: int,
) -> tuple[Array, ...]:
    from jax.experimental import pallas as pl
    from jax.experimental.mosaic import gpu as mgpu
    from jax.experimental.pallas import mosaic_gpu as plgpu
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import arith, memref, vector

    dtype = operands[-1].dtype
    needed = _needed(outputs)
    lane_count = particles * components
    tiles = -(-lane_count // _TILE)
    layout = plgpu.Layout.WG_STRIDED((_TILE,), vec_size=1)
    shapes = {
        _VELOCITY: (particles, components),
        _VELOCITY_GRADIENT: (particles, components, gradient_width),
        _AFFINE_MOMENT: (particles, components, offset_width),
    }
    out_type = tuple(jax.ShapeDtypeStruct(shapes[output], dtype) for output in outputs)

    @plgpu.inline_mgpu(
        arg_types=(layout,) + (plgpu.RefType(),) * (len(operands) + len(outputs))
    )
    def stream(_: object, lane_array: mgpu.FragmentedArray, *refs: ir.Value) -> None:
        index = ir.IndexType.get()
        i32 = ir.IntegerType.get_signless(32)
        real = ir.F32Type.get() if dtype == jnp.dtype(jnp.float32) else ir.F64Type.get()
        predicate = arith.CmpIPredicate
        reference_iterator = iter(refs)
        indices_ref = next(reference_iterator)
        valid_ref = next(reference_iterator)
        weights_ref = next(reference_iterator) if needed[2] else None
        gradients_ref = next(reference_iterator) if needed[3] else None
        offsets_ref = next(reference_iterator) if needed[4] else None
        grid_ref = next(reference_iterator)
        output_refs = dict(zip(outputs, reference_iterator, strict=True))

        # WG_STRIDED with vec_size=1 gives each of the 128 threads one lane.
        (register,) = lane_array.registers.flat
        lane = vector.extract(register, [], [0])
        # Tail lanes past the last particle issue no loads and no stores.
        with mgpu.when(arith.cmpi(predicate.slt, lane, mgpu.c(lane_count, i32))):
            lane_index = arith.index_cast(index, lane)
            width = mgpu.c(components, index)
            particle = arith.divui(lane_index, width)
            component = arith.remui(lane_index, width)
            zero = mgpu.c(0.0, real)
            index_zero = mgpu.c(0, i32)
            source_count = mgpu.c(sources, i32)

            def accumulate(position: ir.Value, totals: list[ir.Value]) -> list[ir.Value]:
                """One route of the lane in storage order; same carry order."""
                route = memref.load(indices_ref, [particle, position])
                is_valid = memref.load(valid_ref, [particle, position])
                # Match reference retrieval: normalize negative indices once,
                # then clamp. Invalid routes still issue only a safe row-0 load.
                safe = arith.select(is_valid, route, index_zero)
                normalized = arith.select(
                    arith.cmpi(predicate.slt, safe, index_zero),
                    arith.addi(safe, source_count),
                    safe,
                )
                bounded = arith.minsi(
                    arith.maxsi(normalized, index_zero), mgpu.c(sources - 1, i32)
                )
                source = arith.index_cast(index, bounded)
                route_velocity = arith.select(
                    is_valid, memref.load(grid_ref, [source, component]), zero
                )
                # Python binding only (emits nothing): velocity and affine
                # outputs both imply a weights operand, which rebinds it.
                weighted = zero
                if weights_ref is not None:
                    weight = arith.select(
                        is_valid, memref.load(weights_ref, [particle, position]), zero
                    )
                    weighted = arith.mulf(weight, route_velocity)
                carried = iter(totals)
                updated: list[ir.Value] = []
                if _VELOCITY in outputs:
                    updated.append(arith.addf(next(carried), weighted))
                if gradients_ref is not None:
                    for axis in range(gradient_width):
                        term = memref.load(
                            gradients_ref, [particle, position, mgpu.c(axis, index)]
                        )
                        updated.append(
                            arith.addf(next(carried), arith.mulf(route_velocity, term))
                        )
                if offsets_ref is not None:
                    for axis in range(offset_width):
                        term = memref.load(
                            offsets_ref, [particle, position, mgpu.c(axis, index)]
                        )
                        updated.append(
                            arith.addf(next(carried), arith.mulf(weighted, term))
                        )
                return updated

            # Keep route order, but expose the bounded loop to instruction
            # scheduling; paired measurements favored this over a rolled loop.
            initial = [zero] * (
                int(_VELOCITY in outputs)
                + (gradient_width if gradients_ref is not None else 0)
                + (offset_width if offsets_ref is not None else 0)
            )
            values = initial
            for slot in range(routes):
                values = accumulate(mgpu.c(slot, index), values)
            totals = iter(values)
            if _VELOCITY in outputs:
                memref.store(next(totals), output_refs[_VELOCITY], [particle, component])
            for output, axes, present in (
                (_VELOCITY_GRADIENT, gradient_width, gradients_ref is not None),
                (_AFFINE_MOMENT, offset_width, offsets_ref is not None),
            ):
                if present:
                    for axis in range(axes):
                        memref.store(
                            next(totals),
                            output_refs[output],
                            [particle, component, mgpu.c(axis, index)],
                        )

    def kernel(*refs: jax.Ref) -> None:
        start = jax.lax.axis_index("lanes") * _TILE
        lanes = start + plgpu.broadcasted_iota(jnp.int32, (_TILE,), 0, layout=layout)
        stream(plgpu.layout_cast(lanes, layout), *refs)

    itemsize = dtype.itemsize
    widths = int(needed[2]) + gradient_width * needed[3] + offset_width * needed[4]
    flops = (
        lane_count
        * routes
        * (
            2 * (_VELOCITY in outputs)
            + (_VELOCITY not in outputs and needed[2])
            + 2 * gradient_width * needed[3]
            + 2 * offset_width * needed[4]
        )
    )
    result = plgpu.kernel(
        kernel,
        out_type=out_type,
        grid=(tiles,),
        grid_names=("lanes",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        cost_estimate=pl.CostEstimate(
            flops=flops,
            transcendentals=0,
            bytes_accessed=lane_count * routes * (5 + itemsize * (1 + widths))
            + sum(int(np.prod(value.shape)) for value in out_type) * itemsize,
        ),
    )(*operands)
    return tuple(result)


def _native(
    indices: Array,
    valid: Array,
    weights: Array,
    gradients: Array,
    offsets: Array,
    grid_velocity: Array,
    *,
    outputs: tuple[int, ...],
) -> tuple[Array, ...]:
    operands = (indices, valid, weights, gradients, offsets, grid_velocity)
    if not _admitted(operands, outputs):
        return _reference(*operands, outputs=outputs)
    needed = _needed(outputs)
    # A boolean GMEM reference reads the existing mask without a conversion buffer.
    kernel_operands = (
        (indices, valid)
        + tuple(operand for operand, use in zip(operands[2:5], needed[2:5]) if use)
        + (grid_velocity,)
    )
    return _streaming_kernel(
        kernel_operands,
        outputs=outputs,
        particles=indices.shape[0],
        routes=indices.shape[1],
        components=grid_velocity.shape[1],
        sources=grid_velocity.shape[0],
        gradient_width=gradients.shape[-1] if needed[3] else 0,
        offset_width=offsets.shape[-1] if needed[4] else 0,
    )


_moments_p = core.Primitive("phydrax_apic_streaming_moments")
_moments_p.multiple_results = True
_moments_p.def_impl(partial(dispatch.apply_primitive, _moments_p))
_moments_p.def_abstract_eval(_abstract)
mlir.register_lowering(_moments_p, mlir.lower_fun(_reference, multiple_results=True))
mlir.register_lowering(
    _moments_p,
    partial(lower_mosaic_or_reference, _native, _reference),
    platform="cuda",
)


def _jvp(
    primals: tuple[Array, ...],
    tangents: tuple[Tangent, ...],
    *,
    outputs: tuple[int, ...],
) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
    result = tuple(_moments_p.bind(*primals, outputs=outputs))
    needed = _needed(outputs)
    active = tuple(
        position
        for position in _FLOATING_OPERANDS
        if needed[position] and not isinstance(tangents[position], ad.Zero)
    )
    if not active:
        return result, tuple(ad.Zero(jax.typeof(value)) for value in result)

    # Tangents belong to the owning JAX reference, never to the native kernel.
    def reference(*values: Array) -> tuple[Array, ...]:
        arguments = list(primals)
        for position, value in zip(active, values):
            arguments[position] = value
        return _reference(*arguments, outputs=outputs)

    _, tangent = jax.jvp(
        reference,
        tuple(primals[position] for position in active),
        tuple(tangents[position] for position in active),
    )
    return result, tuple(tangent)


def _transpose(
    cotangents: tuple[Tangent, ...],
    *operands: Primal,
    outputs: tuple[int, ...],
) -> tuple[Tangent | None, ...]:
    """Transpose the reference with the known route operands held fixed."""

    live_outputs = tuple(
        output
        for output, cotangent in zip(outputs, cotangents, strict=True)
        if not isinstance(cotangent, ad.Zero)
    )
    if not live_outputs:
        return tuple(
            ad.Zero(value.aval) if isinstance(value, ad.UndefinedPrimal) else None
            for value in operands
        )

    def reference(*values: Array) -> tuple[Array, ...]:
        iterator = iter(values)
        arguments = tuple(
            next(iterator) if isinstance(value, ad.UndefinedPrimal) else value
            for value in operands
        )
        return _reference(*arguments, outputs=live_outputs)

    templates = tuple(
        jax.ShapeDtypeStruct(value.aval.shape, value.aval.dtype)
        for value in operands
        if isinstance(value, ad.UndefinedPrimal)
    )
    pullback = jax.linear_transpose(reference, *templates)
    result = iter(
        pullback(tuple(value for value in cotangents if not isinstance(value, ad.Zero)))
    )
    return tuple(
        next(result) if isinstance(value, ad.UndefinedPrimal) else None
        for value in operands
    )


def _placeholder(dtype: jnp.dtype, /) -> Array:
    return jnp.zeros((), dtype=dtype)


def _batch(
    operands: tuple[Array, ...],
    axes: tuple[int | None, ...],
    *,
    outputs: tuple[int, ...],
) -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
    needed = _needed(outputs)
    operands = tuple(
        operand if use or axis is None else _placeholder(operand.dtype)
        for operand, axis, use in zip(operands, axes, needed)
    )
    axes = tuple(axis if use else None for axis, use in zip(axes, needed))
    if all(axis is None for axis in axes):
        result = _moments_p.bind(*operands, outputs=outputs)
        return tuple(result), (None,) * len(outputs)
    # Any mapped read operand maps the canonical reference exactly.
    result = jax.vmap(partial(_reference, outputs=outputs), in_axes=axes)(*operands)
    return tuple(result), (0,) * len(outputs)


def _placeholder_literal(variable: core.Var | core.Literal, /) -> core.Literal:
    """A 0-d zero literal of an unread operand's dtype."""
    aval = variable.aval
    if not isinstance(aval, ShapedArray):
        raise TypeError("Streaming APIC operands must be shaped arrays.")
    return core.Literal(np.zeros((), dtype=aval.dtype), ShapedArray((), aval.dtype))


def _dead_code_elimination(
    used_outputs: list[bool], live_inputs: list[bool], equation: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
    del live_inputs
    if not any(used_outputs):
        return [False] * len(equation.invars), None
    outputs = tuple(
        output for output, used in zip(equation.params["outputs"], used_outputs) if used
    )
    needed = _needed(outputs)
    # Unread operands become scalar literals, so their producers (for example
    # weight gradients of an energy-only consumer) are eliminated upstream.
    invariables = [
        variable if use else _placeholder_literal(variable)
        for variable, use in zip(equation.invars, needed)
    ]
    return list(needed), equation.replace(
        invars=invariables,
        outvars=[
            variable for variable, used in zip(equation.outvars, used_outputs) if used
        ],
        params={**equation.params, "outputs": outputs},
    )


ad.primitive_jvps[_moments_p] = _jvp
ad.primitive_transposes[_moments_p] = _transpose
batching.primitive_batchers[_moments_p] = _batch
pe.dce_rules[_moments_p] = _dead_code_elimination


def streaming_moments(
    indices: ArrayLike,
    valid: ArrayLike,
    weights: ArrayLike,
    gradients: ArrayLike,
    offsets: ArrayLike,
    grid_velocity: ArrayLike,
    /,
) -> tuple[Array, Array, Array]:
    """Velocity, velocity-gradient and affine route moments of one route table.

    Shapes: ``indices``/``valid``/``weights`` ``(particle, route)``,
    ``gradients`` ``(particle, route, gradient)``, ``offsets``
    ``(particle, route, offset)`` and flat ``grid_velocity`` ``(source,
    component)``. Floating operands sharing one dtype bind the streaming
    primitive; other dtype combinations keep the reference under ordinary JAX
    promotion. Unused results are eliminated together with the operands only
    they read.
    """
    operands = tuple(
        jnp.asarray(value)
        for value in (indices, valid, weights, gradients, offsets, grid_velocity)
    )
    floating = operands[2:]
    if not jnp.issubdtype(floating[-1].dtype, jnp.inexact) or any(
        value.dtype != floating[-1].dtype for value in floating
    ):
        velocity, velocity_gradient, affine_moment = _reference(
            *operands, outputs=_ALL_OUTPUTS
        )
        return velocity, velocity_gradient, affine_moment
    velocity, velocity_gradient, affine_moment = _moments_p.bind(
        *operands, outputs=_ALL_OUTPUTS
    )
    return velocity, velocity_gradient, affine_moment


__all__ = ["streaming_moments"]
