#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

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
    if isinstance(function, ConcatenatedModelEvaluator):
        return _callable_regularity(function.raw_model)
    if isinstance(function, AbstractArrayModel):
        return function.model_execution_contract().regularity
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


@dataclass(frozen=True, slots=True)
class DerivativeRequest:
    """One derivative of a named residual field requested by an operator.

    `admission` records the regularity admission made while tracing the request
    (`None` when the field does not depend on a differentiated variable, so the
    derivative vanishes by independence, or when the request was built directly).
    """

    field: str
    variable: str
    axes: tuple[int | None, ...]
    laplacian_count: int = 0
    variable_path: tuple[str, ...] = ()
    backends: tuple[DerivativeBackend, ...] = ()
    laplacian_variables: tuple[str, ...] = ()
    laplacian_backends: tuple[DerivativeBackend, ...] = ()
    admission: DerivativeAdmission | None = None

    @property
    def contracted_laplacian(self) -> bool:
        return self.laplacian_count > 0

    @property
    def order(self) -> int:
        return len(self.axes) + 2 * self.laplacian_count

    @property
    def variables(self) -> frozenset[str]:
        return frozenset((self.variable, *self.variable_path, *self.laplacian_variables))

    @property
    def explicitly_uses_jet(self) -> bool:
        return "jet" in (*self.backends, *self.laplacian_backends)


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
                    (*request.variable_path, *request.laplacian_variables),
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
        prefix: tuple[int | None, ...] = (),
        prefix_variables: tuple[str, ...] = (),
        prefix_backends: tuple[DerivativeBackend, ...] = (),
        prefix_laplacian_variables: tuple[str, ...] = (),
        prefix_laplacian_backends: tuple[DerivativeBackend, ...] = (),
    ):
        self.recorded = recorded
        self.prefix = prefix
        self.prefix_variables = prefix_variables
        self.prefix_backends = prefix_backends
        self.prefix_laplacian_variables = prefix_laplacian_variables
        self.prefix_laplacian_backends = prefix_laplacian_backends

    def _result(
        self,
        *,
        prefix: tuple[int | None, ...],
        prefix_variables: tuple[str, ...],
        prefix_backends: tuple[DerivativeBackend, ...],
        prefix_laplacian_variables: tuple[str, ...],
        prefix_laplacian_backends: tuple[DerivativeBackend, ...],
    ) -> DomainFunction:
        source = self.recorded.source
        return DomainFunction(
            domain=source.domain,
            deps=source.deps,
            func=source.func,
            metadata=source.metadata,
            derivative_rule=_RequestRecorderRule(
                self.recorded,
                prefix=prefix,
                prefix_variables=prefix_variables,
                prefix_backends=prefix_backends,
                prefix_laplacian_variables=prefix_laplacian_variables,
                prefix_laplacian_backends=prefix_laplacian_backends,
            ),
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
        order_ = int(order)
        axes = self.prefix + (axis,) * order_
        variables = self.prefix_variables + (var,) * order_
        backends = self.prefix_backends + (backend,) * order_
        self.recorded.record(
            DerivativeRequest(
                field=self.recorded.field,
                variable=var,
                axes=axes,
                laplacian_count=len(self.prefix_laplacian_variables),
                variable_path=variables,
                backends=backends,
                laplacian_variables=self.prefix_laplacian_variables,
                laplacian_backends=self.prefix_laplacian_backends,
            )
        )
        return self._result(
            prefix=axes,
            prefix_variables=variables,
            prefix_backends=backends,
            prefix_laplacian_variables=self.prefix_laplacian_variables,
            prefix_laplacian_backends=self.prefix_laplacian_backends,
        )

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
        laplacian_variables = self.prefix_laplacian_variables + (var,)
        laplacian_backends = self.prefix_laplacian_backends + (backend,)
        self.recorded.record(
            DerivativeRequest(
                field=self.recorded.field,
                variable=var,
                axes=self.prefix,
                laplacian_count=len(laplacian_variables),
                variable_path=self.prefix_variables,
                backends=self.prefix_backends,
                laplacian_variables=laplacian_variables,
                laplacian_backends=laplacian_backends,
            )
        )
        return self._result(
            prefix=self.prefix,
            prefix_variables=self.prefix_variables,
            prefix_backends=self.prefix_backends,
            prefix_laplacian_variables=laplacian_variables,
            prefix_laplacian_backends=laplacian_backends,
        )


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


DerivativeExecutionStrategy = Literal["reverse", "forward", "jvp", "jet"]


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
    ):
        if not requests:
            raise ValueError("DerivativeExecutionPlan requires derivative requests.")
        if strategy not in ("reverse", "forward", "jvp", "jet"):
            raise ValueError("Unknown derivative execution strategy.")
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

    def second_direction(tangent):
        def first_direction(current):
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
        DerivativeRequest("__fused__", "__coordinate__", (axis,)) for axis in first
    ) + tuple(
        DerivativeRequest("__fused__", "__coordinate__", (axis, axis)) for axis in second
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
    "FusedDerivativeEvaluation",
    "evaluate_fused_coordinate_derivatives",
    "plan_derivative_execution",
    "trace_derivative_requests",
]
