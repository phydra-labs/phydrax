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
from jax._src import ad_util, core, effects, source_info_util, util
from jax._src.ad_checkpoint import transpose_jaxpr
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe, remat
from jax._src.lax.control_flow import conditionals
from jax.typing import ArrayLike


_BranchResult = TypeVar("_BranchResult")


# Retain the guard boundary through JAX transformations: native cond's select
# batching evaluates refused operands. Pure branches use same-branch reference
# operands instead; effectful branches retain lane-local ordered execution.
_domain_call = core.Primitive("phydrax_domain_call")
_domain_call.multiple_results = True
_pure_cond = core.Primitive("phydrax_pure_domain_cond")
_pure_cond.multiple_results = True
_pure_cond.skip_canonicalization = True


def _bind(call: core.ClosedJaxpr, arguments: Sequence[Any]) -> list[Any]:
    # Keep the converted jaxpr's identity: JAX caches AD/partial evaluation by
    # this object. Rewrapping it would retrace every nested guard transform.
    closed = pe.convert_constvars_jaxpr(call)
    return _domain_call.bind(*call.consts, *arguments, call=closed)


def _implementation(*arguments: Any, call: core.ClosedJaxpr) -> list[Any]:
    return core.jaxpr_as_fun(call)(*arguments)


def _abstract(
    *avals: core.AbstractValue, call: core.ClosedJaxpr
) -> tuple[list[core.AbstractValue], core.Effects]:
    del avals
    return call.out_avals, call.effects


def _sequential_batch(
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


@util.weakref_lru_cache
def _vectorized_call(call: core.Jaxpr) -> core.Jaxpr:
    def replace_parameter(value: Any) -> Any:
        return _vectorized_call(value) if isinstance(value, core.Jaxpr) else value

    equations = [
        equation.replace(
            primitive=(
                _pure_cond
                if equation.primitive is conditionals.cond_p
                else equation.primitive
            ),
            params=jax.tree.map(
                replace_parameter,
                equation.params,
                is_leaf=lambda value: isinstance(value, core.Jaxpr),
            ),
        )
        for equation in call.eqns
    ]
    return call.replace(eqns=equations)


def _batch(
    axis_data: batching.AxisData,
    arguments: Sequence[Any],
    dimensions: Sequence[int | None],
    *,
    call: core.ClosedJaxpr,
) -> tuple[list[Any], tuple[int | None, ...]]:
    if call.effects:
        return _sequential_batch(arguments, dimensions, call=call)
    mapped = tuple(axis is not None for axis in dimensions)
    batched_call, output_mapped = batching.batch_jaxpr(
        _vectorized_call(call), axis_data, mapped, False
    )
    values = _bind(
        batched_call,
        tuple(
            value if axis is None else jnp.moveaxis(value, axis, 0)
            for value, axis in zip(arguments, dimensions, strict=True)
        ),
    )
    return values, tuple(0 if present else None for present in output_mapped)


def _lane_select(mask: Array, accepted: Array, refused: Array) -> Array:
    expanded = mask.reshape(mask.shape + (1,) * (accepted.ndim - mask.ndim))
    return jnp.where(expanded, accepted, refused)


def _pure_cond_implementation(
    *arguments: Any, branches: Sequence[core.Jaxpr], **params: Any
) -> list[Any]:
    return conditionals.cond_p.bind(*arguments, branches=branches, **params)


def _reference_branch(
    axis_data: batching.AxisData,
    active: Array,
    operands: Sequence[Any],
    branch: core.Jaxpr,
    input_mapped: tuple[bool, ...],
) -> list[Any]:
    def evaluate(_: None) -> list[Any]:
        reference = jnp.argmax(active.astype(jnp.int32))
        safe_operands = [
            _lane_select(active, value, value[reference]) if present else value
            for value, present in zip(operands, input_mapped, strict=True)
        ]
        batched_branch, _ = batching.batch_jaxpr(branch, axis_data, input_mapped, True)
        return _bind(batched_branch, safe_operands)

    def empty(_: None) -> list[Array]:
        values = []
        for aval in branch.out_avals:
            if not isinstance(aval, core.ShapedArray):
                raise TypeError("Vectorized domain branches require array outputs.")
            values.append(
                batching.bdim_at_front(
                    jnp.zeros(aval.shape, dtype=aval.dtype),
                    None,
                    axis_data.size,
                    mesh_axis=axis_data.explicit_mesh_axis,
                )
            )
        return values

    # This guard also protects outer vmap axes. Native lax.cond would
    # otherwise batch an outer all-refused lane to an eager select.
    return domain_cond(jnp.any(active), evaluate, empty)


def _pure_cond_batch(
    axis_data: batching.AxisData,
    arguments: Sequence[Any],
    dimensions: Sequence[int | None],
    *,
    branches: Sequence[core.Jaxpr],
    **params: Any,
) -> tuple[list[Any], Sequence[int | None]]:
    if dimensions[0] is None:
        return batching.fancy_primitive_batchers[conditionals.cond_p](
            axis_data, arguments, dimensions, branches=branches, **params
        )
    index = batching.bdim_at_front(
        arguments[0],
        dimensions[0],
        axis_data.size,
        mesh_axis=axis_data.explicit_mesh_axis,
    )
    input_mapped = tuple(axis is not None for axis in dimensions[1:])
    # Only declared lane operands need reference substitution; shared numerical
    # operands remain explicit and dynamic without acquiring a broadcast axis.
    operands = [
        value
        if axis is None
        else batching.bdim_at_front(
            value, axis, axis_data.size, mesh_axis=axis_data.explicit_mesh_axis
        )
        for value, axis in zip(arguments[1:], dimensions[1:], strict=True)
    ]
    branch_outputs = [
        _reference_branch(
            axis_data, index == branch_index, operands, branch, input_mapped
        )
        for branch_index, branch in enumerate(branches)
    ]
    outputs = branch_outputs[0]
    for branch_index, values in enumerate(branch_outputs[1:], start=1):
        outputs = [
            _lane_select(index == branch_index, accepted, refused)
            for accepted, refused in zip(values, outputs, strict=True)
        ]
    return outputs, (0,) * len(outputs)


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


def _linearize(
    is_vjp: bool,
    nonzero: Sequence[bool],
    *primals: Any,
    call: core.ClosedJaxpr,
) -> tuple[list[Any], Sequence[bool], tuple[Any, ...], Any, Callable[..., list[Any]]]:
    # Use JAX's cached call linearization directly. Falling back through the
    # JVP rule re-enters legacy partial evaluation and hoists nested scans again.
    forward_call, forward_tree, output_nonzero, forwarded, tangent_call = (
        ad.linearize_jaxpr(call, nonzero, is_vjp=is_vjp)
    )
    primal_avals, residual_avals, structured_avals = forward_tree.unpack()
    values = _bind(forward_call, primals)
    primal_count = len(primal_avals)
    residual_end = primal_count + len(residual_avals)
    materialized = iter(values[primal_count:residual_end])
    original_inputs = (*call.consts, *primals)
    residuals = tuple(
        next(materialized) if index is None else original_inputs[index]
        for index in forwarded
    )
    structured = structured_avals.update(values[residual_end:]).unflatten()

    def linearized(
        saved: Sequence[Any], structured_saved: Any, *tangents: Any
    ) -> list[Any]:
        tangent_values = iter(
            _bind(
                tangent_call,
                (
                    *saved,
                    *(
                        value
                        for value, present in zip(tangents, nonzero, strict=True)
                        if present
                    ),
                    *jax.tree.leaves(structured_saved),
                ),
            )
        )
        return [
            next(tangent_values) if present else ad_util.Zero(aval.to_tangent_aval())
            for aval, present in zip(call.out_avals, output_nonzero, strict=True)
        ]

    return values[:primal_count], output_nonzero, residuals, structured, linearized


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
    closed = pe.convert_constvars_jaxpr(linear_call)
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


def _rematerialization_parameters(
    _unknown_inputs: Sequence[bool],
    _instantiated_inputs: Sequence[bool],
    _known_outputs: Sequence[bool],
    _instantiated_outputs: Sequence[bool],
    _forwarded_residuals: int,
    _total_residuals: int,
    known: dict[str, Any],
    staged: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The closed-call rule splits primal/tangent outputs while retaining both
    # calls behind this lane-local primitive; there are no other parameters.
    return known, staged


def _rematerialization_partial_eval(
    saveable: Callable[..., pe.RematCases_],
    unknown_inputs: Sequence[bool],
    instantiated_inputs: Sequence[bool],
    equation: core.JaxprEqn,
) -> pe.PartialEvalCustomResult:
    return pe.closed_call_partial_eval_custom_rule(
        "call",
        _rematerialization_parameters,
        saveable,
        list(unknown_inputs),
        list(instantiated_inputs),
        equation,
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


def _dead_code_elimination(
    used_outputs: list[bool], equation: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
    # Delegate liveness and effect retention to JAX's closed-call owner. Only
    # its parameter spelling differs from this lane-local native primitive.
    parameters = dict(equation.params)
    parameters["call_jaxpr"] = parameters.pop("call")
    used_inputs, pruned = pe.dce_jaxpr_closed_call_rule(
        used_outputs, equation.replace(params=parameters)
    )
    if pruned is None:
        return used_inputs, None
    parameters = dict(pruned.params)
    parameters["call"] = parameters.pop("call_jaxpr")
    return used_inputs, pruned.replace(params=parameters)


_domain_call.def_impl(_implementation)
_domain_call.def_effectful_abstract_eval(_abstract)
batching.fancy_primitive_batchers[_domain_call] = _batch
ad.primitive_jvps[_domain_call] = _jvp
ad.primitive_linearizations[_domain_call] = _linearize
ad.primitive_transposes[_domain_call] = _transpose
pe.custom_partial_eval_rules[_domain_call] = _partial_eval
pe.partial_eval_jaxpr_custom_rules[_domain_call] = _rematerialization_partial_eval
pe.dce_rules[_domain_call] = _dead_code_elimination
mlir.register_lowering(
    _domain_call,
    mlir.lower_fun(_implementation, multiple_results=True),
    inline=False,
)

# Only the batching semantics differ from scalar JAX cond. Every derivative,
# partial-evaluation, refusal callback, and lowering remains with its owner.
_pure_cond.def_impl(_pure_cond_implementation)
_pure_cond.def_effectful_abstract_eval(conditionals.cond_p.abstract_eval)
batching.fancy_primitive_batchers[_pure_cond] = _pure_cond_batch
ad.primitive_jvps[_pure_cond] = ad.primitive_jvps[conditionals.cond_p]
ad.primitive_linearizations[_pure_cond] = ad.primitive_linearizations[conditionals.cond_p]
ad.fancy_transposes[_pure_cond] = ad.fancy_transposes[conditionals.cond_p]
pe.custom_partial_eval_rules[_pure_cond] = pe.custom_partial_eval_rules[
    conditionals.cond_p
]
pe.partial_eval_jaxpr_custom_rules[_pure_cond] = pe.partial_eval_jaxpr_custom_rules[
    conditionals.cond_p
]
pe.dce_rules[_pure_cond] = pe.dce_rules[conditionals.cond_p]
remat.rules[_pure_cond] = remat.rules[conditionals.cond_p]
core.custom_typechecks[_pure_cond] = core.custom_typechecks[conditionals.cond_p]
mlir.register_lowering(
    _pure_cond, mlir._lowerings[conditionals.cond_p].rule, inline=False
)


def domain_cond(
    predicate: ArrayLike,
    true_function: Callable[[Any], _BranchResult],
    false_function: Callable[[Any], _BranchResult],
    operand: object = None,
) -> _BranchResult:
    """Scalar conditional preserving scientific refusal under mapped evaluation.

    Closure values are explicit primitive operands, including model parameters.
    Pure mapped branches replace only discarded evaluation operands with a
    reference from the same branch; trial states and selected outputs are not
    changed. Effectful branches execute once per selected lane in order.
    """
    call, output_shape, static = eqx.filter_make_jaxpr(
        lambda: jax.lax.cond(predicate, true_function, false_function, operand)
    )()
    values = _bind(call, ())
    dynamic = jax.tree.unflatten(jax.tree.structure(output_shape), values)
    return eqx.combine(dynamic, static)
