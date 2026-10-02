#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic, transported, and affine-jump trace relations across identified seams.

For a `PeriodicIdentification` with source (lower) face ``a`` and target (upper)
face ``b`` along coordinate ``x_c`` one declaration states

```text
T[u_target](b, y) - Gamma S[u_source](a, y) = g(y),
```

where ``S`` and ``T`` are certified constant-coefficient jet actions along ``x_c``
(by default the ``order``-th coordinate derivative), ``Gamma`` is a scalar or an
event-linear transport, and ``g`` is the seam target. ``Gamma = 1`` with ``g = 0`` is
ordinary periodicity, ``Gamma = -1`` antiperiodicity, ``Gamma = exp(i k L)`` Bloch
matching, and ``g != 0`` an affine jump.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._doc import DOC_KEY0
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..domain import (
    DomainComponent,
    DomainFunction,
    UnaryFieldEvaluator,
)
from ..domain.decomposition._cover import PairedSupport
from ..domain.decomposition._periodic import PeriodicIdentification
from ..operators.differential._domain_ops import partial_n
from ..typing import checked
from ._base import AbstractResidualCondition
from ._functional import EventLinearMap
from ._ir import (
    AbstractConditionOperator,
    ArrayCodomain,
    codomains_compatible,
    Condition,
    ConditionCodomain,
    FieldCodomain,
    FieldSpec,
    OperatorCapabilities,
    OperatorLinearization,
    ProductFieldSpec,
)
from ._relations import Equality
from ._subdomain import _field_name, FieldBinding


class JetAction(StrictModule):
    """Certified constant-coefficient jet ``sum_k c_k d^k u / dx_c^k``.

    The coordinate ``x_c`` is the identified coordinate of the seam that applies
    the action. Coefficients are fixed numbers, so the action is linear in ``u``
    by construction; a conserved flux ``k du/dx`` with constant conductivity ``k``
    is ``JetAction({1: k})``.
    """

    orders: tuple[int, ...] = eqx.field(static=True)
    coefficients: Array
    action_id: str = eqx.field(static=True)
    unit: bool = eqx.field(static=True)

    def __init__(self, terms: Mapping[int, ArrayLike], /) -> None:
        if not isinstance(terms, Mapping) or not terms:
            raise TypeError(
                "JetAction terms must be a nonempty order->coefficient mapping."
            )
        orders = tuple(sorted(terms))
        if any(isinstance(order, bool) or not isinstance(order, int) for order in orders):
            raise TypeError("JetAction orders must be integers.")
        if orders[0] < 0:
            raise ValueError("JetAction orders must be nonnegative.")
        coefficients = jnp.stack([jnp.asarray(terms[order]) for order in orders])
        if coefficients.ndim != 1:
            raise ValueError("JetAction coefficients must be scalars.")
        if not jnp.issubdtype(coefficients.dtype, jnp.inexact):
            coefficients = coefficients.astype(jnp.float64)
        if not bool(jnp.all(jnp.isfinite(coefficients))):
            raise ValueError("JetAction coefficients must be finite.")
        if bool(jnp.any(coefficients == 0)):
            raise ValueError("JetAction coefficients must be nonzero.")
        self.orders = orders
        self.coefficients = coefficients
        # A single unit coefficient is the bare derivative and is never multiplied.
        self.unit = len(orders) == 1 and bool(coefficients[0] == 1)
        self.action_id = canonical_fingerprint(
            {
                "kind": "periodic-jet-action",
                "orders": orders,
                "coefficients": array_tree_fingerprint(coefficients),
            }
        )

    @classmethod
    def derivative(cls, order: int, /) -> JetAction:
        """The pure coordinate derivative of one order."""
        return cls({order: 1.0})

    @property
    def max_order(self) -> int:
        return self.orders[-1]

    @checked
    def apply(
        self, u: DomainFunction, identification: PeriodicIdentification, /
    ) -> DomainFunction:
        """Apply the jet along the identified coordinate of ``u``'s domain."""
        result: DomainFunction | None = None
        for index, order in enumerate(self.orders):
            derivative = partial_n(
                u, var=identification.label, axis=identification.component, order=order
            )
            term = (
                derivative
                if self.unit
                else _promoted(derivative, _PromotedConstant(self.coefficients[index]))
            )
            result = term if result is None else result + term
        if result is None:
            raise RuntimeError("A validated JetAction lost its terms.")
        return result


class _PromotedConstant(StrictModule):
    """Scale by, or subtract, a seam constant in the promoted dtype of the value.

    Real jet coefficients, transports, and targets act on complex (Bloch) fields.
    The constant is cast explicitly to the promoted dtype of the evaluated value,
    so the relation does not depend on implicit real-to-complex promotion.
    """

    constant: Array
    subtract: bool = eqx.field(static=True)

    def __init__(self, constant: Array, /, *, subtract: bool = False) -> None:
        self.constant = constant
        self.subtract = subtract

    def __call__(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        # Constants adopt the field precision. A complex constant makes a real
        # field complex at that same precision; nothing widens the field.
        dtype = _adopted_dtype(self.constant, value_)
        constant = self.constant.astype(dtype)
        value_ = value_.astype(dtype)
        return value_ - constant if self.subtract else constant * value_


class _EventContract(StrictModule):
    """Refuse a trace whose pointwise event shape or dtype differs from the declaration."""

    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str | None = eqx.field(static=True)

    @checked
    def __init__(self, value: ArrayCodomain, /) -> None:
        self.shape = value.shape
        self.dtype = value.dtype

    def __call__(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != self.shape:
            raise ValueError(
                f"Periodic trace has event shape {value_.shape}, but the declaration "
                f"states {self.shape}; declare the field's actual value codomain."
            )
        if self.dtype is not None and value_.dtype.name != self.dtype:
            raise ValueError(
                f"Periodic trace has dtype {value_.dtype.name}, but the declaration "
                f"states {self.dtype}."
            )
        return value_


def _promoted(value: DomainFunction, op: _PromotedConstant, /) -> DomainFunction:
    return DomainFunction(
        domain=value.domain,
        deps=value.deps,
        func=UnaryFieldEvaluator(value.func, op),
        metadata={},
    )


def _checked(value: DomainFunction, contract: _EventContract, /) -> DomainFunction:
    return DomainFunction(
        domain=value.domain,
        deps=value.deps,
        func=UnaryFieldEvaluator(value.func, contract),
        metadata=value.metadata,
    )


def _adopted_dtype(constant: Array, value: Array, /) -> np.dtype:
    """Field-precision dtype for applying ``constant`` to ``value``."""
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        return np.promote_types(constant.dtype, value.dtype)
    if jnp.issubdtype(constant.dtype, jnp.complexfloating) and not jnp.issubdtype(
        value.dtype, jnp.complexfloating
    ):
        return np.result_type(value.dtype, np.complex64)
    return np.dtype(value.dtype)


class _EventTransport(StrictModule):
    """Apply an event-linear transport at the field precision."""

    transport: EventLinearMap

    @checked
    def __init__(self, transport: EventLinearMap, /) -> None:
        self.transport = transport

    def __call__(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        dtype = _adopted_dtype(self.transport.matrix, value_)
        return ein.contract(
            "oi,...i->...o", self.transport.matrix.astype(dtype), value_.astype(dtype)
        )


def _transported(
    value: DomainFunction, transport: Array | EventLinearMap, /
) -> DomainFunction:
    if isinstance(transport, EventLinearMap):
        return DomainFunction(
            domain=value.domain,
            deps=value.deps,
            func=UnaryFieldEvaluator(value.func, _EventTransport(transport)),
            metadata={},
        )
    return _promoted(value, _PromotedConstant(transport))


class Periodic(AbstractResidualCondition):
    """Transported trace relation across one identified periodic seam.

    **Arguments:**

    - `fields`: One field name for a same-field seam, or the canonical
      ``(source_field, target_field)`` pair; the source is evaluated on the lower
      face and the target on the upper face. A `LocalFieldRef` binds a decomposed
      patch field.
    - `pairing`: A ``"periodic-interface"`` `PairedSupport` from
      `PeriodicIdentification.pairing()` or a periodic Cartesian cover.
    - `order`: Coordinate derivative order of the matched trace. It states that
      order only; declare one condition per required order.
    - `transport`: Scalar ``Gamma`` (``-1`` antiperiodic, a unit complex phase for
      Bloch matching) or an `EventLinearMap` acting on the source trace's event.
    - `target`: Affine seam target ``g`` (constant or a `DomainFunction`).
    - `value`: Explicit event codomain of the matched trace; scalar by default.
    - `trace_actions`: Optional certified ``(source_action, target_action)``
      `JetAction` pair replacing the ``order``-th derivative, e.g. a constant
      constitutive flux. ``order`` must then be zero.
    - `label`: Optional diagnostic label.
    """

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    pairing: PairedSupport
    identification: PeriodicIdentification
    source_field: str = eqx.field(static=True)
    target_field: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    transport: Array | EventLinearMap
    target: DomainFunction
    target_constant: Array | None
    value: ArrayCodomain
    source_action: JetAction
    target_action: JetAction
    label: str | None = eqx.field(static=True)
    homogeneous: bool = eqx.field(static=True)
    identity_transport: bool = eqx.field(static=True)
    _condition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        fields: FieldBinding | tuple[FieldBinding, FieldBinding],
        pairing: PairedSupport,
        /,
        *,
        order: int = 0,
        transport: ArrayLike | EventLinearMap = 1.0,
        target: DomainFunction | ArrayLike = 0.0,
        value: ArrayCodomain | None = None,
        trace_actions: tuple[JetAction, JetAction] | None = None,
        label: str | None = None,
    ) -> None:
        identification = pairing.identification
        if pairing.topology != "periodic-interface" or identification is None:
            raise ValueError("Periodic requires a periodic-interface pairing.")
        if isinstance(fields, tuple):
            if len(fields) != 2:
                raise ValueError("fields must be one name or a (source, target) pair.")
            source = _field_name(fields[0])
            target_name = _field_name(fields[1])
            if source == target_name:
                raise ValueError(
                    "A same-field seam binds its field once; pass one field name."
                )
            names: tuple[str, ...] = (source, target_name)
        else:
            source = target_name = _field_name(fields)
            names = (source,)
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError("order must be a nonnegative integer.")
        value_ = ArrayCodomain() if value is None else value
        if trace_actions is None:
            source_action = target_action = JetAction.derivative(order)
        else:
            if order != 0:
                raise ValueError("Explicit trace_actions replace order; use order=0.")
            # Nominal items in general-container annotations remain static-only;
            # the declaration owns this pair's container, arity, and item kinds.
            if (
                not isinstance(trace_actions, tuple)
                or len(trace_actions) != 2
                or not all(isinstance(action, JetAction) for action in trace_actions)
            ):
                raise TypeError(
                    "trace_actions must be a (source_action, target_action) JetAction pair."
                )
            source_action, target_action = trace_actions
        if isinstance(transport, EventLinearMap):
            if len(value_.shape) != 1 or transport.matrix.shape != (
                value_.shape[0],
                value_.shape[0],
            ):
                raise ValueError(
                    "An EventLinearMap transport requires a matching rank-one value codomain."
                )
            transport_: Array | EventLinearMap = transport
        else:
            transport_ = jnp.asarray(transport)
            if transport_.shape:
                raise ValueError(
                    "Array transports must be scalar; use EventLinearMap for event maps."
                )
            if not jnp.issubdtype(transport_.dtype, jnp.inexact):
                transport_ = transport_.astype(jnp.float64)
            if not bool(jnp.isfinite(transport_)) or bool(transport_ == 0):
                raise ValueError("A scalar transport must be finite and nonzero.")
        if value_.dtype is not None and not jnp.issubdtype(
            np.dtype(value_.dtype), jnp.complexfloating
        ):
            constants = (
                transport_.matrix
                if isinstance(transport_, EventLinearMap)
                else transport_,
                source_action.coefficients,
                target_action.coefficients,
            )
            if any(jnp.iscomplexobj(constant) for constant in constants):
                raise ValueError(
                    f"A real declared value dtype {value_.dtype} cannot carry complex "
                    "transport or jet coefficients; declare a complex value dtype."
                )
        target_, target_constant = _seam_target(pairing, target, value_)
        self.fields = names
        self.on = pairing.component
        self.pairing = pairing
        self.identification = identification
        self.source_field = source
        self.target_field = target_name
        self.order = order
        self.transport = transport_
        self.target = target_
        self.target_constant = target_constant
        self.value = value_
        self.source_action = source_action
        self.target_action = target_action
        self.label = None if label is None else str(label)
        # Structural facts decided once on the host, so compiled consumers never
        # synchronize on them.
        self.homogeneous = self.target_constant is not None and bool(
            jnp.all(self.target_constant == 0)
        )
        if isinstance(transport_, EventLinearMap):
            matrix = np.asarray(transport_.matrix)
            self.identity_transport = bool(
                np.array_equal(matrix, np.eye(matrix.shape[0]))
            )
        else:
            self.identity_transport = bool(np.asarray(transport_) == 1)
        self._condition_id = canonical_fingerprint(
            {
                "kind": "periodic-condition",
                "fields": self.fields,
                "pairing": self.pairing.pairing_id,
                "identification": self.identification.revision,
                "source_action": self.source_action.action_id,
                "target_action": self.target_action.action_id,
                "transport": (
                    self.transport.map_id
                    if isinstance(self.transport, EventLinearMap)
                    else array_tree_fingerprint(self.transport)
                ),
                "value_axes": [
                    [axis.name, axis.size, axis.labels] for axis in self.value.axes
                ],
                "value_dtype": self.value.dtype,
                "constant_target": (
                    None
                    if self.target_constant is None
                    else array_tree_fingerprint(self.target_constant)
                ),
                "target_dependencies": self.target.deps,
                "label": self.label,
            }
        )

    @property
    def max_order(self) -> int:
        return max(self.source_action.max_order, self.target_action.max_order)

    @property
    def same_field(self) -> bool:
        return self.source_field == self.target_field

    @checked
    def action(self, source: DomainFunction, target: DomainFunction, /) -> DomainFunction:
        """Linear seam action ``T[target](b) - Gamma S[source](a)`` on the seam."""
        contract = _EventContract(self.value)
        upper = _checked(
            self.pairing.trace(
                self.target_action.apply(target, self.identification),
                side=self.pairing.target_side,
            ),
            contract,
        )
        lower = _transported(
            _checked(
                self.pairing.trace(
                    self.source_action.apply(source, self.identification),
                    side=self.pairing.source_side,
                ),
                contract,
            ),
            self.transport,
        )
        # Promote both traces explicitly to the common kind of every seam constant
        # (jet coefficients and transport): a complex constant on either side makes
        # both traces complex, while real constants keep the field precision. The
        # two traces must otherwise share one precision.
        transport_dtype = (
            self.transport.matrix.dtype
            if isinstance(self.transport, EventLinearMap)
            else self.transport.dtype
        )
        common = jnp.ones(
            (),
            np.result_type(
                transport_dtype,
                self.source_action.coefficients.dtype,
                self.target_action.coefficients.dtype,
            ),
        )
        upper = _promoted(upper, _PromotedConstant(common))
        lower = _promoted(lower, _PromotedConstant(common))
        return upper - lower

    def residual(self, functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        missing = tuple(name for name in self.fields if name not in functions)
        if missing:
            raise KeyError(f"Missing periodic fields {missing!r}.")
        action = self.action(functions[self.source_field], functions[self.target_field])
        if self.target_constant is None:
            return action - self.target
        return _promoted(action, _PromotedConstant(self.target_constant, subtract=True))

    @checked
    def as_condition(
        self,
        *,
        fields: ProductFieldSpec | Sequence[FieldSpec] | None = None,
        codomain: ConditionCodomain | None = None,
        condition_id: str | None = None,
    ) -> Condition:
        """Lower to a certified linear trace action with an explicit affine target."""
        support = (
            self.identification.domain.component()
            if self.pairing.self_seam
            and self.identification.domain.same_support(self.on.domain)
            else self.on
        )
        source_fields = (
            ProductFieldSpec(
                tuple(FieldSpec(name, FieldCodomain(support)) for name in self.fields)
            )
            if fields is None
            else fields
            if isinstance(fields, ProductFieldSpec)
            else ProductFieldSpec(tuple(fields))
        )
        if source_fields.sources != self.fields:
            raise ValueError("Periodic lowering must preserve its declared field order.")
        declared = FieldCodomain(self.on, self.value)
        output = declared if codomain is None else codomain
        if not codomains_compatible(output, declared):
            raise ValueError(
                "Periodic conditions lower to the field codomain of their seam and "
                "declared value; a codomain override must be compatible with both."
            )
        identifier = condition_id or self.label or self.condition_id
        return Condition(
            identifier,
            source_fields,
            PeriodicTraceAction(self),
            output,
            Equality(self.target),
            label=self.label,
        )

    @property
    def condition_id(self) -> str:
        """Identity of the geometry, action, fiber, and fixed target declaration."""
        return self._condition_id


def _field_value_issue(field: DomainFunction, value: ArrayCodomain, /) -> str | None:
    """Stage one abstract point to validate the declared field event, without execution."""
    arguments: list[jax.ShapeDtypeStruct] = []
    for label in field.deps:
        coordinate = field.domain.coordinate(label)
        if coordinate.event_shape is None or coordinate.dtype is None:
            return "requires dense typed input coordinates"
        arguments.append(
            jax.ShapeDtypeStruct(coordinate.event_shape, jnp.dtype(coordinate.dtype))
        )
    output = jax.eval_shape(field.func, *arguments, key=DOC_KEY0)
    if not isinstance(output, jax.ShapeDtypeStruct) or output.shape != value.shape:
        return "has an incompatible value event shape"
    if value.dtype is not None and output.dtype.name != value.dtype:
        return "has an incompatible value dtype"
    return None


def _seam_target(
    pairing: PairedSupport,
    target: DomainFunction | ArrayLike,
    value: ArrayCodomain,
    /,
) -> tuple[DomainFunction, Array | None]:
    domain = pairing.component.domain
    if isinstance(target, DomainFunction):
        if not target.domain.same_support(domain):
            raise ValueError("Periodic target must live on the seam domain.")
        traced = bool(target.deps) and set(target.deps) <= set(
            pairing.coordinate_labels(pairing.source_side)
        )
        restricted = pairing.trace(target, side=pairing.source_side) if traced else target
        issue = _field_value_issue(restricted, value)
        if issue is not None:
            raise ValueError(f"Periodic target {issue}.")
        return restricted, None
    array = jnp.asarray(target)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.float64)
    if array.shape not in ((), value.shape):
        raise ValueError(
            f"Periodic target shape {array.shape} does not match the value codomain {value.shape}."
        )
    if value.dtype is not None:
        dtype = jnp.dtype(value.dtype)
        if jnp.iscomplexobj(array) and not jnp.issubdtype(dtype, jnp.complexfloating):
            raise ValueError("A complex periodic target requires a complex value dtype.")
        array = array.astype(dtype)
    array = jnp.broadcast_to(array, value.shape)
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError("Periodic targets must be finite.")
    return DomainFunction(domain=domain, deps=(), func=array), array


class PeriodicTraceAction(AbstractConditionOperator):
    """Certified linear seam action of one or several `Periodic` declarations.

    A single declaration acts as one seam field; a sequence acts as the ordered
    product of their seam fields. The action excludes the affine targets, which the
    lowered `Equality` carries. No function-space adjoint is claimed.
    """

    conditions: tuple[Periodic, ...]
    product: bool = eqx.field(static=True)
    capabilities: OperatorCapabilities = eqx.field(static=True)

    def __init__(self, conditions: Periodic | Sequence[Periodic], /) -> None:
        # A nominal-or-sequence union remains static-only at checked boundaries.
        product = not isinstance(conditions, Periodic)
        values = tuple(conditions) if product else (conditions,)
        if not values or any(not isinstance(value, Periodic) for value in values):
            raise TypeError("PeriodicTraceAction requires Periodic declarations.")
        self.conditions = values
        self.product = product
        self.capabilities = OperatorCapabilities(is_linear=True)

    def apply(
        self, values: Mapping[str, Any], /, *, key: Any | None = None, **kwargs: Any
    ) -> DomainFunction | tuple[DomainFunction, ...]:
        del key, kwargs
        actions = tuple(
            condition.action(
                values[condition.source_field], values[condition.target_field]
            )
            for condition in self.conditions
        )
        return actions if self.product else actions[0]

    def linear_action(
        self, values: Mapping[str, Any], /, *, key: Any | None = None, **kwargs: Any
    ) -> DomainFunction | tuple[DomainFunction, ...]:
        return self.apply(values, key=key, **kwargs)

    def adjoint_action(
        self, value: Any, /, *, key: Any | None = None, **kwargs: Any
    ) -> NoReturn:
        del value, key, kwargs
        raise TypeError(
            "Periodic trace actions require a representation provider for adjoints."
        )

    def linearize(
        self, values: Mapping[str, Any], /, *, key: Any | None = None, **kwargs: Any
    ) -> OperatorLinearization:
        del values, key, kwargs
        raise TypeError("A globally linear periodic action does not need linearization.")


__all__ = ["JetAction", "Periodic", "PeriodicTraceAction"]
