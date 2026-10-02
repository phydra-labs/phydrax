# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import final

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._differentiation import DerivativeAdmission
from ..._strict import StrictModule
from ...typing import (
    AnyShape,
    as_array,
    as_host_array,
    Bool,
    Float,
    HostFloat,
    Inexact,
    parse,
    Scope,
    Size,
)
from ._jet import jet_series_normalized
from ._taylor_contracts import (
    TaylorContractionSchedule,
    TaylorEventDims,
    TaylorRequestDim,
)
from ._taylor_planning import TaylorContractionPlan


@final
class TaylorContractionResult(StrictModule):
    __strict_contract__ = True

    primal: Inexact[TaylorEventDims]
    values: Inexact[TaylorRequestDim, TaylorEventDims]
    finite: Bool[TaylorRequestDim]
    derivative_valid: Bool[TaylorRequestDim]
    plan: TaylorContractionPlan = eqx.field(static=True)
    evidence: tuple[DerivativeAdmission, ...] = eqx.field(static=True)

    def __init__(
        self,
        primal: Array,
        values: Array,
        finite: Array,
        derivative_valid: Array,
        plan: TaylorContractionPlan,
    ) -> None:
        scope = Scope()
        parse(len(plan.requests), Size[TaylorRequestDim], "request count", scope=scope)
        self.primal = parse(primal, Inexact[TaylorEventDims], "primal", scope=scope)
        self.values = parse(
            values, Inexact[TaylorRequestDim, TaylorEventDims], "values", scope=scope
        )
        self.finite = parse(finite, Bool[TaylorRequestDim], "finite", scope=scope)
        self.derivative_valid = parse(
            derivative_valid, Bool[TaylorRequestDim], "derivative_valid", scope=scope
        )
        if self.values.dtype != self.primal.dtype:
            raise TypeError("Contraction values must have the primal output dtype.")
        self.plan = plan
        self.evidence = plan.executed_admissions


def _curve_coefficients(
    schedule: TaylorContractionSchedule,
    direction_ids: tuple[str, ...],
) -> tuple[tuple[int, ...], ...]:
    positions = {identifier: index for index, identifier in enumerate(direction_ids)}
    coefficients = [[0] * len(direction_ids) for _ in range(schedule.order)]
    for identifier, degree, scale in schedule.coefficients:
        coefficients[degree - 1][positions[identifier]] = scale
    return tuple(tuple(row) for row in coefficients)


def _bind_direction(component: ArrayLike, primal: Float[AnyShape]) -> Float[AnyShape]:
    # Validate the source category before the one input-dtype conversion. Host
    # preparation never reads traced/device numerical values.
    if isinstance(component, (Array, jax_core.Tracer)):
        source = parse(component, Float[AnyShape], "direction component")
    else:
        source = as_host_array(component, HostFloat[AnyShape], "direction component")
    if source.shape != primal.shape:
        raise ValueError("Direction components must match their primal shapes.")
    return parse(
        jnp.asarray(source, dtype=primal.dtype), Float[AnyShape], "direction component"
    )


def _admit_extraction_precision(
    plan: TaylorContractionPlan,
    inputs: tuple[Array, ...],
    output_dtype: DTypeLike,
) -> None:
    """Keep static normalization and integer weights in the nominal normal range."""
    precisions = tuple(jnp.finfo(value.dtype) for value in inputs) + (
        jnp.finfo(output_dtype),
    )
    integer_limit = min(int(precision.max) for precision in precisions)
    normal_limit = min(
        Fraction(1, 1) / Fraction.from_float(float(precision.tiny))
        for precision in precisions
    )
    for recipe in plan.recipes:
        normalization = (
            math.factorial(recipe.request.order)
            if recipe.strategy == "linear"
            else math.prod(
                math.factorial(count) for count in recipe.request.multiplicities
            )
        )
        if normalization > normal_limit or normalization > integer_limit:
            raise ValueError(
                "Taylor normalization exceeds the declared dtype normal range; "
                "use wider input and output precision."
            )
    for terms in plan.extractions:
        if any(
            abs(term.numerator) > integer_limit or term.denominator > integer_limit
            for term in terms
        ):
            raise ValueError(
                "Taylor extraction weights exceed the declared dtype range; "
                "use wider input and output precision."
            )


def _bind_inputs(
    primals: tuple[ArrayLike, ...],
    directions: Mapping[str, tuple[ArrayLike, ...]],
    plan: TaylorContractionPlan,
) -> tuple[tuple[Array, ...], dict[str, tuple[Array, ...]]]:
    if not isinstance(primals, tuple) or not primals:
        raise ValueError("primals must be a nonempty tuple of real arrays.")
    if set(directions) != set(plan.direction_ids):
        raise ValueError("Direction bindings must exactly match the plan direction IDs.")
    inputs = tuple(
        as_array(primal, Float[AnyShape], "Taylor primal") for primal in primals
    )
    bound: dict[str, tuple[Array, ...]] = {}
    for identifier in plan.direction_ids:
        components = directions[identifier]
        if not isinstance(components, tuple) or len(components) != len(inputs):
            raise ValueError(
                "Each direction must bind one component per primal argument."
            )
        bound[identifier] = tuple(
            _bind_direction(component, primal)
            for component, primal in zip(components, inputs, strict=True)
        )
    return inputs, bound


def _admit_worksets(
    function: Callable[..., Array],
    inputs: tuple[Array, ...],
    plan: TaylorContractionPlan,
) -> tuple[dict[int, list[int]], dict[int, tuple[int, ...]], jax.ShapeDtypeStruct]:
    resources = plan.policy.resources
    groups: dict[int, list[int]] = {}
    for index, schedule in enumerate(plan.schedules):
        groups.setdefault(schedule.order, []).append(index)
    selected_orders: dict[int, tuple[int, ...]] = {
        order: tuple(
            sorted(
                {
                    extraction.coefficient_order
                    for request in plan.extractions
                    for extraction in request
                    if plan.schedules[extraction.schedule_index].order == order
                }
            )
        )
        for order in groups
    }
    input_size = sum(primal.size for primal in inputs)
    direction_storage = input_size * len(plan.direction_ids)
    largest_workset = max(
        min(resources.workset_size, len(indices)) * order
        for order, indices in groups.items()
    )
    # Integer coefficient tuples are static host preparation. Only converted
    # argument-dtype matrices belong to numerical scratch, once per argument.
    matrix_storage = max(
        len(indices) * order * len(plan.direction_ids) * len(inputs)
        for order, indices in groups.items()
    )
    input_buffers = direction_storage + largest_workset * input_size + matrix_storage
    preparation_buffers = 2 * direction_storage
    if max(input_buffers, preparation_buffers) > resources.max_logical_buffer_elements:
        raise ValueError(
            "Taylor input worksets exceed the logical-buffer resource limit."
        )
    abstract = jax.eval_shape(function, *inputs)
    if not isinstance(abstract, jax.ShapeDtypeStruct):
        raise TypeError("Taylor function must return one inexact array.")
    if not jnp.issubdtype(abstract.dtype, jnp.inexact):
        raise TypeError(
            "Taylor function outputs must have real or complex floating dtype."
        )
    _admit_extraction_precision(plan, inputs, abstract.dtype)
    output_size = math.prod(abstract.shape)
    stored_coefficients = sum(
        len(indices) * len(selected_orders[order]) for order, indices in groups.items()
    )
    jet_scratch = max(
        min(resources.workset_size, len(indices)) * (order + 1)
        for order, indices in groups.items()
    )
    # Conservatively cover retained coefficients, accumulators, their stack and
    # finiteness mask, one primal and live full-order Jet output scratch.
    output_buffers = output_size * (
        stored_coefficients + 3 * len(plan.requests) + 2 + jet_scratch
    )
    if (
        input_buffers + output_buffers + 4 * len(plan.requests)
        > resources.max_logical_buffer_elements
    ):
        raise ValueError(
            "Taylor output worksets exceed the logical-buffer resource limit."
        )
    return groups, selected_orders, abstract


def _extract_contractions(
    primal: Array,
    results: dict[int, tuple[Array, int]],
    selected_orders: dict[int, tuple[int, ...]],
    plan: TaylorContractionPlan,
) -> TaylorContractionResult:
    values = []
    for request in plan.extractions:
        value = jnp.zeros_like(primal)
        for extraction in request:
            order = plan.schedules[extraction.schedule_index].order
            position = selected_orders[order].index(extraction.coefficient_order)
            group, lane = results[extraction.schedule_index]
            coefficient = group[lane, position]
            weight = jnp.asarray(extraction.numerator, dtype=primal.dtype) / jnp.asarray(
                extraction.denominator, dtype=primal.dtype
            )
            value = value + weight * coefficient
        values.append(value)
    stacked = jnp.stack(tuple(values))
    finite = jnp.all(jnp.isfinite(stacked), axis=tuple(range(1, stacked.ndim))) & jnp.all(
        jnp.isfinite(primal)
    )
    admitted = jnp.asarray(
        tuple(item.supported for item in plan.executed_admissions), dtype=jnp.bool_
    )
    return TaylorContractionResult(primal, stacked, finite, finite & admitted, plan)


def evaluate_taylor_contractions(
    function: Callable[..., Array],
    primals: tuple[ArrayLike, ...],
    directions: Mapping[str, tuple[ArrayLike, ...]],
    plan: TaylorContractionPlan,
) -> TaylorContractionResult:
    """Execute certified scalar curves without multivariate derivative tensors.

    Curves sharing a schedule execute once at their maximum required order.
    Material schedule groups use bounded ``lax.map`` worksets. The callable,
    parameters, input values, and direction values remain dynamic and may be
    differentiated. Logical series/output limits do not claim bounds on XLA
    compiler temporaries. Native Jet primitive failures are never caught.
    """
    if not isinstance(plan, TaylorContractionPlan):
        raise TypeError("plan must be TaylorContractionPlan.")
    inputs, bound = _bind_inputs(primals, directions, plan)
    groups, selected_orders, abstract = _admit_worksets(function, inputs, plan)
    resources = plan.policy.resources
    direction_matrices = tuple(
        jnp.stack(
            tuple(
                bound[identifier][argument].reshape(-1)
                for identifier in plan.direction_ids
            )
        )
        for argument in range(len(inputs))
    )
    del bound
    primal: Array | None = None
    results: dict[int, tuple[Array, int]] = {}
    for order, indices in groups.items():
        extraction_orders = selected_orders[order]
        metadata = tuple(
            _curve_coefficients(plan.schedules[index], plan.direction_ids)
            for index in indices
        )

        def evaluate_curve(matrices: tuple[Array, ...]) -> tuple[Array, Array]:
            argument_series = tuple(
                (matrix @ direction_matrix).reshape((order, *argument.shape))
                for matrix, direction_matrix, argument in zip(
                    matrices, direction_matrices, inputs, strict=True
                )
            )
            series = tuple(
                tuple(coefficients[degree] for degree in range(order))
                for coefficients in argument_series
            )
            curve_primal, coefficients = jet_series_normalized(function, inputs, series)
            if (
                curve_primal.shape != abstract.shape
                or curve_primal.dtype != abstract.dtype
            ):
                raise ValueError(
                    "Taylor function output layout changed during Jet execution."
                )
            return curve_primal, jnp.stack(
                tuple(coefficients[degree - 1] for degree in extraction_orders)
            )

        def evaluate_coefficients(matrices: tuple[Array, ...]) -> Array:
            return evaluate_curve(matrices)[1]

        remaining = indices
        if primal is None:
            first_matrices = tuple(
                jnp.asarray(metadata[0], dtype=argument.dtype) for argument in inputs
            )
            primal, first_coefficients = evaluate_curve(first_matrices)
            del first_matrices
            results[indices[0]] = (first_coefficients[None], 0)
            remaining = indices[1:]
            metadata = metadata[1:]
        if not remaining:
            continue
        # Convert only the remaining host metadata; device slicing would copy
        # full matrices merely to drop the separately evaluated first lane.
        coefficient_matrices = tuple(
            jnp.asarray(metadata, dtype=argument.dtype) for argument in inputs
        )
        if len(remaining) == 1:
            group_values = evaluate_coefficients(
                tuple(matrix[0] for matrix in coefficient_matrices)
            )[None]
        else:
            group_values = jax.lax.map(
                evaluate_coefficients,
                coefficient_matrices,
                batch_size=min(resources.workset_size, len(remaining)),
            )
        for position, index in enumerate(remaining):
            results[index] = (group_values, position)
        del coefficient_matrices
    if primal is None:
        raise ValueError("A Taylor plan must execute at least one curve.")
    return _extract_contractions(primal, results, selected_orders, plan)
