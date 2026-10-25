#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax._src import ad_util, dispatch
from jax._src.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ..backends._mosaic import lower_mosaic_or_reference
from ._compensated import _accumulate_pallas, _accumulate_reference, two_sum


type Tangent = Array | ad.Zero
type Primal = Array | ad.UndefinedPrimal


def _forward(flat: Array, high: Array, correction: Array) -> tuple[Array, Array]:
    return _accumulate_reference(flat, (high, correction))


def _transpose_reference(
    high: Array, correction: Array, *, length: int
) -> tuple[Array, Array, Array]:
    zero = jnp.zeros_like(high)
    terms = jnp.zeros((length,) + high.shape, high.dtype)
    return jax.linear_transpose(_forward, terms, zero, zero)((high, correction))


def _transpose_step_jaxpr(shape: tuple[int, ...], dtype: jnp.dtype) -> core.ClosedJaxpr:
    def forward_step(high: Array, correction: Array, term: Array) -> tuple[Array, Array]:
        next_high, error = two_sum(high, term)
        return next_high, correction + error

    specification = jax.ShapeDtypeStruct(shape, dtype)
    transpose_step = jax.linear_transpose(
        forward_step, specification, specification, specification
    )
    adjoint = jax.make_jaxpr(transpose_step)((specification, specification))
    supported = (jax.lax.add_p, jax.lax.sub_p, jax.lax.neg_p, ad_util.add_any_p)
    if (
        adjoint.effects
        or adjoint.consts
        or any(
            equation.primitive not in supported
            or equation.params
            or any(
                not isinstance(value.aval, ShapedArray)
                or value.aval.dtype != dtype
                or value.aval.shape != shape
                for value in (*equation.invars, *equation.outvars)
            )
            for equation in adjoint.jaxpr.eqns
        )
    ):
        raise NotImplementedError(
            "The compensated one-step adjoint must contain only "
            "same-shaped real add/subtract/negate operations."
        )
    # Generate JAX's exact add/subtract adjoint without algebraic cancellation.
    # Normalize only its same-shaped real cotangent adds for Mosaic lowering.
    equations = [
        equation.replace(primitive=jax.lax.add_p)
        if equation.primitive is ad_util.add_any_p
        else equation
        for equation in adjoint.jaxpr.eqns
    ]
    return core.ClosedJaxpr(adjoint.jaxpr.replace(eqns=equations), adjoint.consts)


def _transpose_real(
    high: Array, correction: Array, *, length: int
) -> tuple[Array, Array, Array]:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    output_shape = high.shape
    lane_count = high.size
    full_tiles, remainder = divmod(lane_count, 128)

    def kernel(
        high_ref: jax.Ref,
        correction_ref: jax.Ref,
        terms_ref: jax.Ref,
        output_high_ref: jax.Ref,
        output_correction_ref: jax.Ref,
    ) -> None:
        program = jax.lax.axis_index("lanes")

        def run(start: Array, *, vector: bool) -> None:
            window = pl.ds(start, 128) if vector else start
            initial_high = high_ref[window]
            initial_correction = correction_ref[window]
            if vector:
                layout = plgpu.Layout.WG_STRIDED((128,), vec_size=1)
                initial_high = plgpu.layout_cast(initial_high, layout)
                initial_correction = plgpu.layout_cast(initial_correction, layout)
            transpose_step = core.jaxpr_as_fun(
                _transpose_step_jaxpr(initial_high.shape, initial_high.dtype)
            )

            def step(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
                next_high, next_correction, term = transpose_step(*carry)
                terms_ref[length - 1 - index, window] = term
                return next_high, next_correction

            high, correction = jax.lax.fori_loop(
                0, length, step, (initial_high, initial_correction)
            )
            output_high_ref[window] = high
            output_correction_ref[window] = correction

        if full_tiles:

            @pl.when(program < full_tiles)
            def vector_tile() -> None:
                run(program * 128, vector=True)

        if remainder:

            @pl.when(program >= full_tiles)
            def scalar_lane() -> None:
                run(full_tiles * 128 + program - full_tiles, vector=False)

    shape = jax.ShapeDtypeStruct((lane_count,), high.dtype)
    terms, high, correction = plgpu.kernel(
        kernel,
        out_type=(jax.ShapeDtypeStruct((length, lane_count), high.dtype), shape, shape),
        grid=(full_tiles + remainder,),
        grid_names=("lanes",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        cost_estimate=pl.CostEstimate(
            flops=len(_transpose_step_jaxpr((), high.dtype).jaxpr.eqns)
            * length
            * lane_count,
            transcendentals=0,
            bytes_accessed=(length + 4) * lane_count * high.dtype.itemsize,
        ),
    )(high.reshape((lane_count,)), correction.reshape((lane_count,)))
    return (
        terms.reshape((length,) + output_shape),
        high.reshape(output_shape),
        correction.reshape(output_shape),
    )


def _transpose_native(
    high: Array, correction: Array, *, length: int
) -> tuple[Array, Array, Array]:
    components = 2 if jnp.issubdtype(high.dtype, jnp.complexfloating) else 1
    if length == 0 or high.size == 0 or length * high.size * components > (1 << 31) - 1:
        return _transpose_reference(high, correction, length=length)
    if not jnp.issubdtype(high.dtype, jnp.complexfloating):
        return _transpose_real(high, correction, length=length)
    results = _transpose_real(
        jnp.stack((high.real, high.imag), axis=-1),
        jnp.stack((correction.real, correction.imag), axis=-1),
        length=length,
    )
    terms, initial_high, initial_correction = results
    return (
        jax.lax.complex(terms[..., 0], terms[..., 1]),
        jax.lax.complex(initial_high[..., 0], initial_high[..., 1]),
        jax.lax.complex(initial_correction[..., 0], initial_correction[..., 1]),
    )


def _forward_abstract(
    flat: ShapedArray, high: ShapedArray, correction: ShapedArray
) -> tuple[ShapedArray, ShapedArray]:
    if not flat.ndim or flat.shape[1:] != high.shape or high.shape != correction.shape:
        raise ValueError("Compensated recurrence operands have inconsistent shapes.")
    if flat.dtype != high.dtype or high.dtype != correction.dtype:
        raise TypeError("Compensated recurrence operands must have identical dtypes.")
    return high, correction


def _forward_native(flat: Array, high: Array, correction: Array) -> tuple[Array, Array]:
    components = 2 if jnp.issubdtype(flat.dtype, jnp.complexfloating) else 1
    if flat.shape[0] == 0 or high.size == 0 or flat.size * components > (1 << 31) - 1:
        return _forward(flat, high, correction)
    return _accumulate_pallas(flat, (high, correction))


_forward_p = core.Primitive("phydrax_compensated_accumulate_linear")
_forward_p.multiple_results = True
_forward_p.def_impl(partial(dispatch.apply_primitive, _forward_p))
_forward_p.def_abstract_eval(_forward_abstract)
mlir.register_lowering(_forward_p, mlir.lower_fun(_forward, multiple_results=True))
mlir.register_lowering(
    _forward_p,
    partial(lower_mosaic_or_reference, _forward_native, _forward),
    platform="cuda",
)

_transpose_p = core.Primitive("phydrax_compensated_accumulate_transpose")
_transpose_p.multiple_results = True
_transpose_p.def_impl(partial(dispatch.apply_primitive, _transpose_p))


def _transpose_abstract(
    high: ShapedArray, correction: ShapedArray, *, length: int
) -> tuple[ShapedArray, ShapedArray, ShapedArray]:
    if length < 0 or high.shape != correction.shape:
        raise ValueError("Compensated transpose operands have inconsistent shapes.")
    if high.dtype != correction.dtype:
        raise TypeError("Compensated transpose operands must have identical dtypes.")
    return ShapedArray((length,) + high.shape, high.dtype), high, correction


_transpose_p.def_abstract_eval(_transpose_abstract)
mlir.register_lowering(
    _transpose_p, mlir.lower_fun(_transpose_reference, multiple_results=True)
)
mlir.register_lowering(
    _transpose_p,
    partial(lower_mosaic_or_reference, _transpose_native, _transpose_reference),
    platform="cuda",
)


def _reference_jvp(
    function: Callable[..., tuple[Array, ...]],
    primals: tuple[Array, ...],
    tangents: tuple[Tangent, ...],
) -> tuple[Array, ...]:
    active = tuple(not isinstance(tangent, ad.Zero) for tangent in tangents)

    def reference(*arguments: Array) -> tuple[Array, ...]:
        iterator = iter(arguments)
        return function(
            *(next(iterator) if use else value for value, use in zip(primals, active))
        )

    return jax.jvp(
        reference,
        tuple(value for value, use in zip(primals, active) if use),
        tuple(value for value, use in zip(tangents, active) if use),
    )[1]


def _forward_jvp(
    primals: tuple[Array, Array, Array], tangents: tuple[Tangent, Tangent, Tangent]
) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
    result = tuple(_forward_p.bind(*primals))
    if all(isinstance(value, ad.Zero) for value in tangents):
        return result, tuple(ad.Zero(jax.typeof(value)) for value in result)
    if not isinstance(tangents[0], ad.Zero):
        tangent = tuple(
            _forward_p.bind(*(ad.instantiate_zeros(value) for value in tangents))
        )
    elif isinstance(tangents[1], ad.Zero):
        tangent = (ad.Zero(jax.typeof(result[0])), tangents[2])
    else:
        tangent = _reference_jvp(_forward, primals, tangents)
    return result, tangent


def _forward_transpose(
    cotangents: tuple[Tangent, Tangent], flat: Primal, high: Primal, correction: Primal
) -> tuple[Array | None, ...]:
    length = flat.aval.shape[0] if isinstance(flat, ad.UndefinedPrimal) else flat.shape[0]
    result = _transpose_p.bind(
        *(ad.instantiate_zeros(value) for value in cotangents), length=length
    )
    return tuple(
        value if ad.is_undefined_primal(primal) else None
        for value, primal in zip(result, (flat, high, correction))
    )


def _transpose_jvp(
    primals: tuple[Array, Array], tangents: tuple[Tangent, Tangent], *, length: int
) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
    result = tuple(_transpose_p.bind(*primals, length=length))
    if all(isinstance(value, ad.Zero) for value in tangents):
        return result, tuple(ad.Zero(jax.typeof(value)) for value in result)
    if all(not isinstance(value, ad.Zero) for value in tangents):
        tangent = tuple(_transpose_p.bind(*tangents, length=length))
    else:
        tangent = _reference_jvp(
            partial(_transpose_reference, length=length), primals, tangents
        )
        if isinstance(tangents[1], ad.Zero):
            tangent = (*tangent[:2], ad.Zero(jax.typeof(result[2])))
    return result, tangent


def _transpose_transpose(
    cotangents: tuple[Tangent, Tangent, Tangent],
    high: Primal,
    correction: Primal,
    *,
    length: int,
) -> tuple[Array | None, ...]:
    high_aval = high.aval if ad.is_undefined_primal(high) else jax.typeof(high)
    zero = jnp.zeros(high_aval.shape, high_aval.dtype)
    result = jax.linear_transpose(
        partial(_transpose_reference, length=length), zero, zero
    )(tuple(ad.instantiate_zeros(value) for value in cotangents))
    return tuple(
        value if ad.is_undefined_primal(primal) else None
        for value, primal in zip(result, (high, correction))
    )


def _forward_batch(
    operands: tuple[Array, Array, Array], axes: tuple[int | None, int | None, int | None]
) -> tuple[tuple[Array, ...], tuple[int, int]]:
    size = next(
        value.shape[axis] for value, axis in zip(operands, axes) if axis is not None
    )
    flat, high, correction = (
        batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes)
    )
    result = _forward_p.bind(jnp.moveaxis(flat, 0, 1), high, correction)
    return tuple(result), (0, 0)


def _transpose_batch(
    operands: tuple[Array, Array], axes: tuple[int | None, int | None], *, length: int
) -> tuple[tuple[Array, ...], tuple[int, int, int]]:
    size = next(
        value.shape[axis] for value, axis in zip(operands, axes) if axis is not None
    )
    operands = tuple(
        batching.bdim_at_front(value, axis, size) for value, axis in zip(operands, axes)
    )
    return tuple(_transpose_p.bind(*operands, length=length)), (1, 0, 0)


ad.primitive_jvps[_forward_p] = _forward_jvp
ad.primitive_transposes[_forward_p] = _forward_transpose
batching.primitive_batchers[_forward_p] = _forward_batch
ad.primitive_jvps[_transpose_p] = _transpose_jvp
ad.primitive_transposes[_transpose_p] = _transpose_transpose
batching.primitive_batchers[_transpose_p] = _transpose_batch


def accumulate_native(flat: Array, high: Array, correction: Array) -> tuple[Array, Array]:
    """Apply the seeded ordered recurrence with its exact linear AD rules."""
    result = _forward_p.bind(flat, high, correction)
    return result[0], result[1]
