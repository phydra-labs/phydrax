#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax._src import ad_util, core, effects, source_info_util
from jax._src.ad_checkpoint import transpose_jaxpr
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe
from jax._src.util import partition_list
from jax.typing import ArrayLike


_BranchResult = TypeVar("_BranchResult")


# A separate call primitive is necessary: lax.cond batches to select, while
# custom_batching.sequential_vmap does not support reverse-mode differentiation.
# Keep the JAX interpreter boundary here; physical models never depend on it.
_domain_call = core.Primitive("phydrax_domain_call")
_domain_call.multiple_results = True


def _bind(call: core.ClosedJaxpr, arguments: Sequence[Any]) -> list[Any]:
    closed = core.ClosedJaxpr(pe.convert_constvars_jaxpr(call.jaxpr), ())
    return _domain_call.bind(*call.consts, *arguments, call=closed)


def _implementation(*arguments: Any, call: core.ClosedJaxpr) -> list[Any]:
    return core.jaxpr_as_fun(call)(*arguments)


def _abstract(
    *avals: core.AbstractValue, call: core.ClosedJaxpr
) -> tuple[list[core.AbstractValue], core.Effects]:
    del avals
    return call.out_avals, call.effects


def _batch(
    arguments: Sequence[Any],
    dimensions: Sequence[int | None],
    *,
    call: core.ClosedJaxpr,
) -> tuple[list[Any], tuple[int | None, ...]]:
    input_mapped = tuple(axis is not None for axis in dimensions)
    known_call, mapped_call, output_mapped, residual_avals = (
        pe.partial_eval_jaxpr_nounits(call, input_mapped, False)
    )
    known_values = _bind(
        known_call,
        tuple(
            value
            for value, present in zip(arguments, input_mapped, strict=True)
            if not present
        ),
    )
    known_count = len(known_values) - len(residual_avals)
    known_outputs = iter(known_values[:known_count])
    residuals = known_values[known_count:]
    mapped = tuple(
        jnp.moveaxis(value, axis, 0)
        for value, axis in zip(arguments, dimensions, strict=True)
        if axis is not None
    )

    def lane(values: tuple[Array, ...]) -> list[Any]:
        # Binding retains outer mapped axes and their lane-local conditionals.
        return _bind(mapped_call, (*residuals, *values))

    mapped_outputs = iter(jax.lax.map(lane, mapped))
    # In jacfwd's tangent-basis vmap, primal outputs have no tangent-axis
    # dependence. Marking them mapped violates its out_axes=None contract.
    outputs = [
        next(mapped_outputs) if present else next(known_outputs)
        for present in output_mapped
    ]
    return outputs, tuple(0 if present else None for present in output_mapped)


def _jvp(
    primals: Sequence[Any],
    tangents: Sequence[Any],
    *,
    call: core.ClosedJaxpr,
) -> tuple[list[Any], list[Any]]:
    nonzero = tuple(not isinstance(value, ad_util.Zero) for value in tangents)
    if not any(nonzero):
        values = _domain_call.bind(*primals, call=call)
        return values, [ad_util.p2tz(value) for value in values]
    differentiated, output_nonzero = ad.jvp_jaxpr(call, nonzero, False)
    # Bind primal and tangent outputs together. Partial evaluation saves the
    # accepted branch's nonlinear residuals once, preserving effect order.
    inputs = (
        *primals,
        *(value for value, present in zip(tangents, nonzero, strict=True) if present),
    )
    outputs = _bind(differentiated, inputs)
    values = outputs[: len(call.out_avals)]
    tangent_values = iter(outputs[len(call.out_avals) :])
    return values, [
        next(tangent_values) if present else ad_util.p2tz(value)
        for value, present in zip(values, output_nonzero, strict=True)
    ]


def _partial_eval(
    trace: pe.JaxprTrace, *tracers: pe.JaxprTracer, call: core.ClosedJaxpr
) -> list[Any]:
    unknown = tuple(not tracer.pval.is_known() for tracer in tracers)
    known_call, linear_call, output_unknown, residual_avals = (
        pe.partial_eval_jaxpr_nounits(call, unknown, False)
    )
    known_values = _bind(
        known_call,
        tuple(tracer.pval.get_known() for tracer in tracers if tracer.is_known()),
    )
    known_count = len(known_values) - len(residual_avals)
    known_outputs = iter(known_values[:known_count])
    # Residuals precede unknown operands in JAX's partial-evaluated jaxpr.
    # Keep both calls behind the lane-local boundary: evaluating the known
    # jaxpr directly would let vmap execute a rejected scientific branch.
    inputs = [
        *(trace.new_instantiated_const(value) for value in linear_call.consts),
        *(trace.new_instantiated_const(value) for value in known_values[known_count:]),
        *(tracer for tracer, present in zip(tracers, unknown, strict=True) if present),
    ]
    closed = core.ClosedJaxpr(pe.convert_constvars_jaxpr(linear_call.jaxpr), ())
    outputs = [
        pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
        for aval in closed.out_avals
    ]
    recipe = pe.new_eqn_recipe(
        trace,
        inputs,
        outputs,
        _domain_call,
        dict(call=closed),
        core.positional_effects(closed),
        source_info_util.current(),
    )
    for output in outputs:
        output.recipe = recipe
    if effects.partial_eval_kept_effects.filter_in(closed.effects):
        parents: list[core.Tracer] = list(inputs)
        trace.effect_handles.append(pe.EffectHandle(parents, recipe))
    unknown_outputs = iter(outputs)
    return [
        next(unknown_outputs) if present else next(known_outputs)
        for present in output_unknown
    ]


def _partial_eval_custom(
    saveable: Callable[..., Any],
    unknown_inputs: Sequence[bool],
    instantiated_inputs: Sequence[bool],
    eqn: core.JaxprEqn,
) -> tuple[core.JaxprEqn, core.JaxprEqn, list[bool], list[bool], list[core.Var]]:
    """Split one call for rematerialization without staging its known outputs.

    Without this rule, any unknown operand stages the whole call, so the primal
    outputs of a differentiated call become unknown and downstream primal
    values reach transposition as linear inputs. Both halves remain domain calls.
    """
    known, staged, unknown_outputs, instantiated_outputs, residual_count = (
        pe.partial_eval_jaxpr_custom(
            eqn.params["call"].jaxpr,
            unknown_inputs,
            instantiated_inputs,
            False,
            False,
            saveable,
        )
    )
    known_inputs, _ = partition_list(unknown_inputs, eqn.invars)
    known_binders, _ = partition_list(unknown_outputs, eqn.outvars)
    _, staged_inputs = partition_list(instantiated_inputs, eqn.invars)
    _, staged_binders = partition_list(instantiated_outputs, eqn.outvars)
    residuals = [core.Var(variable.aval) for variable in staged.invars[:residual_count]]
    known_eqn = pe.new_jaxpr_eqn(
        known_inputs,
        [*known_binders, *residuals],
        _domain_call,
        dict(call=core.ClosedJaxpr(known, ())),
        core.eqn_effects(known, known_inputs),
        eqn.source_info,
        eqn.ctx,
    )
    staged_eqn = pe.new_jaxpr_eqn(
        [*residuals, *staged_inputs],
        staged_binders,
        _domain_call,
        dict(call=core.ClosedJaxpr(staged, ())),
        core.eqn_effects(staged, [*residuals, *staged_inputs]),
        eqn.source_info,
        eqn.ctx,
    )
    forwarded = [
        variable
        for variable, instantiated in zip(eqn.invars, instantiated_inputs, strict=True)
        if isinstance(variable, core.Var) and not instantiated
    ]
    return (
        known_eqn,
        staged_eqn,
        unknown_outputs,
        instantiated_outputs,
        forwarded + residuals,
    )


def _transpose(
    cotangents: Sequence[Any], *arguments: Any, call: core.ClosedJaxpr
) -> list[Any]:
    linear = tuple(ad.is_undefined_primal(value) for value in arguments)
    zero = tuple(isinstance(value, ad_util.Zero) for value in cotangents)
    transposed, input_zero, output_tree = transpose_jaxpr(call, linear, zero)
    inputs = (
        *(
            value
            for value, is_linear in zip(arguments, linear, strict=True)
            if not is_linear
        ),
        *(value for value, is_zero in zip(cotangents, zero, strict=True) if not is_zero),
    )
    cotangent_values, logs = jax.tree.unflatten(output_tree, _bind(transposed, inputs))
    if jax.tree.leaves(logs):
        raise RuntimeError("Domain conditional transpose produced unexpected logs.")
    values = iter(cotangent_values)
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
pe.custom_partial_eval_rules[_domain_call] = _partial_eval
pe.partial_eval_jaxpr_custom_rules[_domain_call] = _partial_eval_custom
mlir.register_lowering(
    _domain_call, mlir.lower_fun(_implementation, multiple_results=True)
)


def domain_cond(
    predicate: ArrayLike,
    true_function: Callable[[Any], _BranchResult],
    false_function: Callable[[Any], _BranchResult],
    operand: object = None,
) -> _BranchResult:
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
