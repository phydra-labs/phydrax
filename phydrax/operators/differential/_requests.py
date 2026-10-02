#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._differentiation import (
    _REGULARITY_UNDECLARED,
    admit_regularity,
    ComponentAuthority,
    DerivativeAdmission,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
)
from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._validation import canonical_identifier, nonnegative_integer, positive_integer
from ...domain import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
)
from ...domain._function import (
    _ConstCallable,
    BinaryFieldEvaluator,
    SwapAxesFieldEvaluator,
    UnaryFieldEvaluator,
)
from ...domain._model_function import ConcatenatedModelEvaluator
from ...logging import emit
from ...typing import parse


# Direct eager differentiation has no owner beyond its caller, who asked for the
# derivative: almost-everywhere derivatives execute with their conditions and
# only proven degeneracy is rejected.
_EAGER_REGULARITY_POLICY = RegularityPolicy(allow_almost_everywhere=True)
_CONSTANT_REGULARITY = DerivativeRegularity.smooth(degree_bound=0)
_ABSOLUTE_VALUE_REGULARITY = DerivativeRegularity.piecewise_polynomial(
    continuity=0, degree_bound=1
)
_ADMISSION_HINTS = {
    "regularity-degenerate": (
        "the declared piecewise-polynomial degree makes this derivative vanish "
        "almost everywhere"
    ),
    "almost-everywhere-not-allowed": (
        "acknowledge almost-everywhere derivatives with "
        "RegularityPolicy(allow_almost_everywhere=True)"
    ),
    "regularity-undeclared": (
        "declare the model's value regularity or admit it explicitly with "
        "RegularityPolicy(allow_undeclared=True)"
    ),
}


def _combined_field_regularity(
    function: BinaryFieldEvaluator, /
) -> DerivativeRegularity | None:
    left = field_regularity(function.a)
    right = field_regularity(function.b)
    if left is None or right is None:
        return None
    if function.op in (operator.add, operator.sub):
        return left.add(right)
    if function.op in (operator.mul, operator.matmul):
        return left.multiply(right)
    if function.op is operator.truediv and isinstance(function.b.func, _ConstCallable):
        return left
    return None


def _unary_regularity(op: Callable[[Any], Any], /) -> DerivativeRegularity | None:
    if op is operator.abs:
        return _ABSOLUTE_VALUE_REGULARITY
    # Imported here: `phydrax.nn` depends on the differential operators.
    from ...nn.activations import activation_regularity

    return activation_regularity(op)


def _callable_regularity(function: Any, /) -> DerivativeRegularity | None:
    # Imported here: discretization depends on the differential operators.
    from ...discretization._views import DiscreteFieldEvaluator
    from ...domain._evaluation import PointwiseEvaluator
    from ._taylor_domain import _TaylorPartialEvaluator

    if isinstance(function, PointwiseEvaluator):
        return _callable_regularity(function.function)
    if isinstance(function, ConcatenatedModelEvaluator):
        return _callable_regularity(function.raw_model)
    if isinstance(function, AbstractArrayModel):
        return function.model_execution_contract().regularity
    if isinstance(function, _TaylorPartialEvaluator):
        source_regularity = field_regularity(function.source)
        return (
            None
            if source_regularity is None
            else source_regularity.differentiate(
                sum(step.order for step in function.steps)
            )
        )
    if isinstance(function, DiscreteFieldEvaluator):
        return function.regularity
    if isinstance(function, _ConstCallable):
        return _CONSTANT_REGULARITY
    if isinstance(function, BinaryFieldEvaluator):
        return _combined_field_regularity(function)
    if isinstance(function, SwapAxesFieldEvaluator):
        return _callable_regularity(function.func)
    if isinstance(function, UnaryFieldEvaluator):
        inner = _callable_regularity(function.func)
        outer = _unary_regularity(function.op)
        return None if inner is None or outer is None else inner.compose(outer)
    return None


def field_regularity(field: DomainFunction, /) -> DerivativeRegularity | None:
    """Return the declared value regularity of a domain field, or `None`.

    The regularity is composed structurally over the field's evaluation tree:
    bound models contribute their `model_execution_contract().regularity`,
    discrete field views contribute their reconstruction regularity,
    constants are degree-zero polynomials, sums and differences take the maximum
    degree bound, products add degree bounds, division by a constant preserves
    the numerator, and known pointwise maps compose. Any node whose regularity
    is undeclared, including an opaque callable, makes the field undeclared, so
    a declared result is an upper bound that never hides a cancellation.
    """
    if not isinstance(field, DomainFunction):
        raise TypeError("field must be a DomainFunction.")
    return _callable_regularity(field.func)


def admit_field_derivative(
    field: str | None,
    regularity: DerivativeRegularity | None,
    variables: tuple[str, ...],
    order: int,
    /,
    *,
    authority: ComponentAuthority | None,
    policy: RegularityPolicy,
) -> DerivativeAdmission:
    """Admit one value derivative of a field, raising `ValueError` on rejection."""
    admission = admit_regularity(
        regularity,
        DifferentiationRequest(
            (DerivativeSurface.INPUT,), order=order, authority=authority
        ),
        route=DerivativeRoute.DIRECT,
        policy=policy,
    )
    if admission.supported:
        return admission
    owner = "direct" if authority is None else f"{authority.value}-authority"
    reasons = "; ".join(
        f"{reason} ({_ADMISSION_HINTS[reason]})" if reason in _ADMISSION_HINTS else reason
        for reason in admission.reasons
    )
    subject = "the field" if field is None else f"field {field!r}"
    raise ValueError(
        f"Order-{order} derivative of {subject} with respect to "
        f"{', '.join(dict.fromkeys(variables))} is not admitted for {owner} "
        f"differentiation: {reasons}."
    )


def admit_direct_derivative(field: DomainFunction, var: str, order: int, /) -> None:
    """Admit an eager derivative outside scientific preparation.

    Proven degeneracy raises `ValueError`; almost-everywhere derivatives execute,
    and an undeclared field regularity is recorded as a
    `derivative.regularity.undeclared` event.
    """
    if var not in field.deps:
        return
    admission = admit_field_derivative(
        None,
        field_regularity(field),
        (var,),
        order,
        authority=None,
        policy=_EAGER_REGULARITY_POLICY,
    )
    if _REGULARITY_UNDECLARED in admission.conditions:
        emit(
            "DEBUG",
            "derivative.regularity.undeclared",
            "Direct derivative of a field with undeclared regularity",
            variable=var,
            order=order,
        )


DerivativeStepKind: TypeAlias = Literal["partial", "laplacian"]


@final
@dataclass(frozen=True, slots=True)
class DerivativeStep:
    """One ordered operation on a source field's coordinate derivative path."""

    kind: DerivativeStepKind
    variable: str
    axis: int | None = None
    order: int = 1
    backend: DerivativeBackend = "ad"

    def __post_init__(self) -> None:
        kind = parse(self.kind, DerivativeStepKind, "kind")
        canonical_identifier(self.variable, "variable")
        positive_integer(self.order, "order")
        parse(self.backend, DerivativeBackend, "backend")
        if self.axis is not None:
            nonnegative_integer(self.axis, "axis")
        match kind:
            case "partial":
                pass
            case "laplacian":
                if self.axis is not None or self.order != 2:
                    raise ValueError("A Laplacian step has no axis and has order two.")
            case _:
                assert_never(kind)


@final
@dataclass(frozen=True, slots=True)
class DerivativeRequest:
    """An ordered coordinate derivative of one named residual field."""

    field: str
    steps: tuple[DerivativeStep, ...]
    admission: DerivativeAdmission | None = None

    def __post_init__(self) -> None:
        canonical_identifier(self.field, "field")
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple of DerivativeStep values.")
        if not self.steps:
            raise ValueError("A derivative request requires at least one step.")
        if any(not isinstance(step, DerivativeStep) for step in self.steps):
            raise TypeError("steps must contain DerivativeStep values.")
        if self.admission is not None and not isinstance(
            self.admission, DerivativeAdmission
        ):
            raise TypeError("admission must be a DerivativeAdmission or None.")

    @property
    def contracted_laplacian(self) -> bool:
        return any(step.kind == "laplacian" for step in self.steps)

    @property
    def order(self) -> int:
        return sum(step.order for step in self.steps)

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(step.variable for step in self.steps)

    @property
    def explicitly_uses_jet(self) -> bool:
        return any(step.backend == "jet" for step in self.steps)


@dataclass(frozen=True, slots=True, eq=False)
class _RecordedField:
    source: DomainFunction
    field: str
    requests: list[DerivativeRequest]
    regularity: DerivativeRegularity | None
    authority: ComponentAuthority | None
    policy: RegularityPolicy

    def record(self, request: DerivativeRequest, /) -> None:
        # A derivative along a variable the field does not depend on vanishes by
        # independence and requires no regularity.
        if request.variables <= frozenset(self.source.deps):
            request = replace(
                request,
                admission=admit_field_derivative(
                    self.field,
                    self.regularity,
                    tuple(step.variable for step in request.steps),
                    request.order,
                    authority=self.authority,
                    policy=self.policy,
                ),
            )
        self.requests.append(request)


class _RequestRecorderRule(DerivativeRule):
    def __init__(
        self,
        recorded: _RecordedField,
        /,
        *,
        prefix: tuple[DerivativeStep, ...] = (),
    ) -> None:
        self.recorded = recorded
        self.prefix = prefix

    def _result(self, step: DerivativeStep, /) -> DomainFunction:
        source = self.recorded.source
        steps = (*self.prefix, step)
        self.recorded.record(DerivativeRequest(self.recorded.field, steps))
        return DomainFunction(
            domain=source.domain,
            deps=source.deps,
            func=source.func,
            metadata=source.metadata,
            derivative_rule=_RequestRecorderRule(self.recorded, prefix=steps),
        )

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
        del mode, basis, periodic
        return self._result(DerivativeStep("partial", var, axis, order, backend))

    def derive_laplacian(
        self,
        *,
        var: str,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, basis, periodic
        return self._result(DerivativeStep("laplacian", var, order=2, backend=backend))


def trace_derivative_requests(
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
    functions: Mapping[str, DomainFunction],
    /,
    *,
    authority: ComponentAuthority | None = None,
    policy: RegularityPolicy | None = None,
) -> tuple[DerivativeRequest, ...]:
    """Trace derivative requirements without evaluating a batch.

    Every recorded request is admitted against the regularity of its field
    (`field_regularity`) at its accumulated order, and the admission is attached
    to the request. `authority` is the authority of the owner that will execute
    the derivatives; `None` denotes direct differentiation outside scientific
    preparation. The default `policy` is `RegularityPolicy()` for an owner
    authority and, for direct differentiation, a policy admitting
    almost-everywhere derivatives with their conditions. Proven degeneracy is
    always rejected. A rejected request raises `ValueError` while tracing, before
    any batch is evaluated.
    """
    if authority is not None and not isinstance(authority, ComponentAuthority):
        raise TypeError("authority must be a ComponentAuthority or None.")
    if policy is None:
        policy_ = _EAGER_REGULARITY_POLICY if authority is None else RegularityPolicy()
    elif isinstance(policy, RegularityPolicy):
        policy_ = policy
    else:
        raise TypeError("policy must be a RegularityPolicy or None.")
    recorded: list[DerivativeRequest] = []
    traced = {
        name: function.with_derivative_rule(
            _RequestRecorderRule(
                _RecordedField(
                    function,
                    name,
                    recorded,
                    field_regularity(function),
                    authority,
                    policy_,
                )
            )
        )
        for name, function in functions.items()
    }
    result = residual(traced)
    if not isinstance(result, DomainFunction):
        raise TypeError("A ResidualPenalty condition must return a DomainFunction.")
    return tuple(dict.fromkeys(recorded))


DerivativeExecutionStrategy: TypeAlias = Literal["reverse", "forward", "jvp", "jet"]


class DerivativeExecutionPlan(StrictModule):
    """Static execution recommendation for one traced residual derivative set."""

    requests: tuple[DerivativeRequest, ...] = eqx.field(static=True)
    strategy: DerivativeExecutionStrategy = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    contracted_laplacian: bool = eqx.field(static=True)
    directional: bool = eqx.field(static=True)

    def __init__(
        self,
        requests: tuple[DerivativeRequest, ...],
        strategy: DerivativeExecutionStrategy,
        /,
        *,
        directional: bool = False,
    ) -> None:
        if not requests:
            raise ValueError("DerivativeExecutionPlan requires derivative requests.")
        strategy = parse(strategy, DerivativeExecutionStrategy, "strategy")
        self.requests = tuple(requests)
        self.strategy = strategy
        self.maximum_order = max(request.order for request in requests)
        self.variable_count = len(
            set().union(*(request.variables for request in requests))
        )
        self.contracted_laplacian = any(
            request.contracted_laplacian for request in requests
        )
        self.directional = bool(directional)


def plan_derivative_execution(
    requests: tuple[DerivativeRequest, ...],
    /,
    *,
    output_size: int | None = None,
    coordinate_size: int | None = None,
    directional: bool = False,
) -> DerivativeExecutionPlan:
    """Choose a non-approximating AD strategy from derivative request shape."""
    values = tuple(requests)
    if not values:
        raise ValueError("At least one derivative request is required.")
    maximum_order = max(request.order for request in values)
    contracted = any(request.contracted_laplacian for request in values)
    if any(request.explicitly_uses_jet for request in values):
        strategy: DerivativeExecutionStrategy = "jet"
    elif directional or contracted or maximum_order >= 2:
        strategy = "jvp"
    elif (
        output_size is not None
        and coordinate_size is not None
        and int(output_size) > int(coordinate_size)
    ):
        strategy = "forward"
    else:
        strategy = "reverse"
    return DerivativeExecutionPlan(values, strategy, directional=directional)


class FusedDerivativeEvaluation(StrictModule):
    value: Any
    first_derivatives: tuple[Any, ...]
    diagonal_second_derivatives: tuple[Any, ...]
    plan: DerivativeExecutionPlan | None
    first_axes: tuple[int, ...] = eqx.field(static=True)
    second_axes: tuple[int, ...] = eqx.field(static=True)


def evaluate_fused_coordinate_derivatives(
    function: Callable[[Array], Any],
    point: Array,
    /,
    *,
    first_axes: tuple[int, ...] = (),
    second_axes: tuple[int, ...] = (),
) -> FusedDerivativeEvaluation:
    """Evaluate a value, Jacobian columns, and requested Hessian diagonal entries."""
    if not callable(function):
        raise TypeError("function must be callable.")
    point_ = jnp.asarray(point)
    if point_.ndim != 1:
        raise ValueError("Fused coordinate derivatives require one rank-one point.")
    first = tuple(first_axes)
    second = tuple(second_axes)
    if any(axis < 0 or axis >= point_.size for axis in first + second):
        raise ValueError("Fused derivative axis is out of range.")
    value, pushforward = jax.linearize(function, point_)

    def direction(axis: int, /) -> Array:
        return jnp.zeros_like(point_).at[axis].set(1.0)

    if first:
        first_directions = jnp.stack(tuple(direction(axis) for axis in first))
        first_stacked = jax.vmap(pushforward)(first_directions)
        first_values = tuple(
            jax.tree.map(lambda leaf, index=index: leaf[index], first_stacked)
            for index in range(len(first))
        )
    else:
        first_values = ()

    def second_direction(tangent: Array) -> Any:
        def first_direction(current: Array) -> Any:
            return jax.jvp(function, (current,), (tangent,))[1]

        return jax.jvp(first_direction, (point_,), (tangent,))[1]

    if second:
        second_directions = jnp.stack(tuple(direction(axis) for axis in second))
        second_stacked = jax.vmap(second_direction)(second_directions)
        second_values = tuple(
            jax.tree.map(lambda leaf, index=index: leaf[index], second_stacked)
            for index in range(len(second))
        )
    else:
        second_values = ()
    requests = tuple(
        DerivativeRequest(
            "__fused__", (DerivativeStep("partial", "__coordinate__", axis),)
        )
        for axis in first
    ) + tuple(
        DerivativeRequest(
            "__fused__", (DerivativeStep("partial", "__coordinate__", axis, 2),)
        )
        for axis in second
    )
    output_size = sum(jnp.size(leaf) for leaf in jax.tree.leaves(value))
    plan = (
        plan_derivative_execution(
            requests,
            output_size=output_size,
            coordinate_size=point_.size,
            directional=True,
        )
        if requests
        else None
    )
    return FusedDerivativeEvaluation(
        value,
        first_values,
        second_values,
        plan,
        first,
        second,
    )


__all__ = [
    "DerivativeExecutionPlan",
    "DerivativeExecutionStrategy",
    "DerivativeRequest",
    "DerivativeStep",
    "DerivativeStepKind",
    "FusedDerivativeEvaluation",
    "evaluate_fused_coordinate_derivatives",
    "plan_derivative_execution",
    "trace_derivative_requests",
]
