#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src import ad_util, core
from jax._src.ad_checkpoint import transpose_jaxpr
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe, pxla


# A separate call primitive is necessary: lax.cond batches to select, while
# custom_batching.sequential_vmap does not support reverse-mode differentiation.
# Keep the JAX interpreter boundary here; physical models never depend on it.
_domain_call = core.Primitive("phydrax_domain_call")
_domain_call.multiple_results = True


def _bind(call, arguments):
    closed = core.ClosedJaxpr(pe.convert_constvars_jaxpr(call.jaxpr), ())
    return _domain_call.bind(*call.consts, *arguments, call=closed)


def _implementation(*arguments, call):
    return core.jaxpr_as_fun(call)(*arguments)


def _abstract(*avals, call):
    del avals
    return call.out_avals, call.effects


def _batch(arguments, dimensions, *, call):
    mapped = tuple(
        None if axis is None else jnp.moveaxis(value, axis, 0)
        for value, axis in zip(arguments, dimensions, strict=True)
    )

    def lane(values):
        inputs = tuple(
            original if axis is None else value
            for original, value, axis in zip(arguments, values, dimensions, strict=True)
        )
        # Bind again, rather than evaluate the jaxpr directly, so nested vmap
        # axes retain the same lane-local conditional semantics.
        return _domain_call.bind(*inputs, call=call)

    outputs = jax.lax.map(lane, mapped)
    return outputs, (0,) * len(outputs)


def _jvp(primals, tangents, *, call):
    values = _domain_call.bind(*primals, call=call)
    nonzero = tuple(not isinstance(value, ad_util.Zero) for value in tangents)
    if not any(nonzero):
        return values, [ad_util.p2tz(value) for value in values]
    differentiated, output_nonzero = ad.jvp_jaxpr(call, nonzero, False)
    # Separate the known primal outputs from the linear tangent outputs. This
    # lets ordinary partial evaluation retain primal residuals and transpose the
    # tangent call, instead of leaving an opaque custom_vmap derivative primitive.
    tangent_jaxpr, used_constants, used_inputs = pe.dce_jaxpr_consts(
        differentiated.jaxpr,
        (False,) * len(values) + (True,) * sum(output_nonzero),
    )
    tangent_call = core.ClosedJaxpr(
        tangent_jaxpr,
        [
            value
            for value, used in zip(differentiated.consts, used_constants, strict=True)
            if used
        ],
    )
    inputs = (
        *primals,
        *(value for value, present in zip(tangents, nonzero, strict=True) if present),
    )
    tangent_values = iter(
        _bind(
            tangent_call,
            tuple(value for value, used in zip(inputs, used_inputs, strict=True) if used),
        )
    )
    return values, [
        next(tangent_values) if present else ad_util.p2tz(value)
        for value, present in zip(values, output_nonzero, strict=True)
    ]


def _transpose(cotangents, *arguments, call):
    linear = tuple(ad.is_undefined_primal(value) for value in arguments)
    zero = tuple(isinstance(value, ad_util.Zero) for value in cotangents)
    transposed, input_zero = transpose_jaxpr(call, linear, zero)
    inputs = (
        *(
            value
            for value, is_linear in zip(arguments, linear, strict=True)
            if not is_linear
        ),
        *(value for value, is_zero in zip(cotangents, zero, strict=True) if not is_zero),
    )
    values = iter(_bind(transposed, inputs))
    zeros = iter(input_zero)
    return [
        (ad_util.Zero(value.aval) if next(zeros) else next(values)) if is_linear else None
        for value, is_linear in zip(arguments, linear, strict=True)
    ]


_domain_call.def_impl(_implementation)
_domain_call.def_effectful_abstract_eval(_abstract)
batching.primitive_batchers[_domain_call] = _batch
ad.primitive_jvps[_domain_call] = _jvp
ad.primitive_transposes[_domain_call] = _transpose
pxla.register_initial_style_primitive(_domain_call)
mlir.register_lowering(
    _domain_call, mlir.lower_fun(_implementation, multiple_results=True)
)


def domain_cond(predicate, true_function, false_function, operand=None):
    """Scalar conditional whose mapped lanes never execute the other branch.

    Closure values are explicit primitive operands, including model parameters;
    JVP and transpose calls use the same sequential batching rule. Neither trial
    states nor residuals are clipped, replaced with valid states, or sanitized.
    """
    call, output_shape, static = eqx.filter_make_jaxpr(
        lambda: jax.lax.cond(predicate, true_function, false_function, operand)
    )()
    values = _bind(call, ())
    dynamic = jax.tree.unflatten(jax.tree.structure(output_shape), values)
    return eqx.combine(dynamic, static)
