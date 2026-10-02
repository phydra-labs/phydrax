#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...domain import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
)
from ...domain._derivative import DerivativeRuleProvider
from ...domain._function import _drop_model_construction_certificates
from ...typing import PRNGKey
from ._requests import _EAGER_REGULARITY_POLICY, DerivativeStep, field_regularity
from ._runtime import get_partial_eval_cache
from ._taylor_contracts import TaylorContractionRequest
from ._taylor_execution import evaluate_taylor_contractions
from ._taylor_planning import plan_taylor_contractions, TaylorContractionPlan


def _coordinate_identity(source: DomainFunction, step: DerivativeStep) -> tuple[str, int]:
    coordinate = source.domain.coordinate(step.variable)
    if not coordinate.differentiable:
        raise TypeError(f"Coordinate {step.variable!r} is not differentiable.")
    if coordinate.kind == "scalar":
        dimension = 1
    elif coordinate.kind == "array" and coordinate.event_shape is not None:
        if len(coordinate.event_shape) != 1:
            raise ValueError("Taylor coordinate derivatives require rank-one events.")
        dimension = coordinate.event_shape[0]
    else:
        raise TypeError("Taylor coordinate derivatives require dense coordinates.")
    axis = 0 if step.axis is None else step.axis
    if step.axis is None and dimension != 1:
        raise ValueError("A vector-coordinate derivative requires an explicit axis.")
    if axis >= dimension:
        raise ValueError("Taylor derivative axis is outside the declared coordinate.")
    # Variable bindings and component indices belong to the source domain, not shape.
    return f"{step.variable}[{axis}]", axis


def _flatten_arguments(
    args: tuple[ArrayLike | tuple[ArrayLike, ...], ...],
) -> tuple[tuple[Array, ...], tuple[int, ...]]:
    counts = tuple(len(arg) if isinstance(arg, tuple) else 0 for arg in args)
    arrays = tuple(
        jnp.asarray(value)
        for arg in args
        for value in (arg if isinstance(arg, tuple) else (arg,))
    )
    return arrays, counts


def _rebuild_arguments(
    arrays: tuple[Array, ...], counts: tuple[int, ...]
) -> tuple[Array | tuple[Array, ...], ...]:
    rebuilt: list[Array | tuple[Array, ...]] = []
    position = 0
    for count in counts:
        if count:
            rebuilt.append(arrays[position : position + count])
            position += count
        else:
            rebuilt.append(arrays[position])
            position += 1
    return tuple(rebuilt)


@final
class _TaylorPartialEvaluator(StrictModule, DerivativeRuleProvider):
    source: DomainFunction
    steps: tuple[DerivativeStep, ...] = eqx.field(static=True)
    plan: TaylorContractionPlan = eqx.field(static=True)

    def __init__(
        self,
        source: DomainFunction,
        steps: tuple[DerivativeStep, ...],
        plan: TaylorContractionPlan,
        /,
    ) -> None:
        if not isinstance(source, DomainFunction):
            raise TypeError("source must be a DomainFunction.")
        if not steps or any(
            not isinstance(step, DerivativeStep) or step.kind != "partial"
            for step in steps
        ):
            raise ValueError("A Taylor partial evaluator requires partial steps.")
        if not isinstance(plan, TaylorContractionPlan):
            raise TypeError("plan must be a TaylorContractionPlan.")
        self.source = source
        self.steps = steps
        self.plan = plan

    def __call__(
        self,
        *args: Any,
        key: PRNGKey | None = None,
        **kwargs: Any,
    ) -> Array:
        from ._domain_ops import _runtime_signature

        cache = get_partial_eval_cache()
        cache_key = (
            id(self.source.func),
            self.steps,
            self.plan.plan_id,
            _runtime_signature(args),
            _runtime_signature(key),
            _runtime_signature(kwargs),
        )
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        if len(args) != len(self.source.deps):
            raise ValueError("Taylor arguments do not match the source dependencies.")
        variables = frozenset(step.variable for step in self.steps)
        positions = tuple(
            position
            for position, label in enumerate(self.source.deps)
            if label in variables
        )
        primals, counts = _flatten_arguments(
            tuple(args[position] for position in positions)
        )
        directions: dict[str, tuple[Array, ...]] = {}
        for step in self.steps:
            name, axis = _coordinate_identity(self.source, step)
            if name in directions:
                continue
            argument = positions.index(self.source.deps.index(step.variable))
            start = sum(count or 1 for count in counts[:argument])
            index = start + axis if counts[argument] else start
            tangent = jnp.ones_like(primals[index])
            if (
                not counts[argument]
                and self.source.domain.coordinate(step.variable).kind != "scalar"
            ):
                tangent = (
                    jnp.zeros_like(primals[index])
                    .at[..., axis]
                    .set(jnp.asarray(1, dtype=primals[index].dtype))
                )
            directions[name] = tuple(
                tangent if position == index else jnp.zeros_like(primal)
                for position, primal in enumerate(primals)
            )

        def function(*values: Array) -> Array:
            replacements = _rebuild_arguments(values, counts)
            call_args = list(args)
            for position, replacement in zip(positions, replacements, strict=True):
                call_args[position] = replacement
            return jnp.asarray(self.source.func(*call_args, key=key, **kwargs))

        result = evaluate_taylor_contractions(function, primals, directions, self.plan)
        value = eqx.error_if(
            result.values[0],
            ~jnp.all(result.finite) | ~jnp.all(result.derivative_valid),
            "Taylor coordinate contraction failed its numerical or derivative contract.",
        )
        if cache is not None:
            cache[cache_key] = value
        return value

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        del function
        return _TaylorPartialRule(self.source, self.steps)


@final
@dataclass(frozen=True, slots=True, eq=False)
class _TaylorPartialRule(DerivativeRule):
    source: DomainFunction
    steps: tuple[DerivativeStep, ...]

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        if backend != "jet":
            return None
        step = DerivativeStep("partial", var, axis, order, backend)
        return make_taylor_partial(
            self.source, (*self.steps, step), mode=mode, basis=basis, periodic=periodic
        )

    def derive_path(
        self,
        steps: tuple[DerivativeStep, ...],
        /,
        *,
        mode: DerivativeMode,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        if any(step.kind != "partial" or step.backend != "jet" for step in steps):
            return None
        return make_taylor_partial(
            self.source, (*self.steps, *steps), mode=mode, basis=basis, periodic=periodic
        )


def make_taylor_partial(
    source: DomainFunction,
    steps: tuple[DerivativeStep, ...],
    /,
    *,
    mode: DerivativeMode,
    basis: DerivativeBasis,
    periodic: bool,
) -> DomainFunction:
    rule = source.derivative_rule
    if rule is not None:
        native = rule.derive_path(steps, mode=mode, basis=basis, periodic=periodic)
        if native is not None:
            if not isinstance(native, DomainFunction):
                raise TypeError(
                    "DerivativeRule.derive_path must return a DomainFunction or None."
                )
            return native
    multiplicities: dict[str, int] = {}
    for step in steps:
        if step.kind != "partial":
            raise ValueError("Taylor partial paths must contain only partial steps.")
        name, _ = _coordinate_identity(source, step)
        multiplicities[name] = multiplicities.get(name, 0) + step.order
    names = tuple(sorted(multiplicities))
    request = TaylorContractionRequest(
        names, tuple(multiplicities[name] for name in names)
    )
    plan = plan_taylor_contractions(
        (request,),
        regularity=field_regularity(source),
        regularity_policy=_EAGER_REGULARITY_POLICY,
    )
    return DomainFunction(
        domain=source.domain,
        deps=source.deps,
        metadata=_drop_model_construction_certificates(source.metadata),
        func=_TaylorPartialEvaluator(source, steps, plan),
    )
