#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from ._base import AbstractCondition, AbstractMomentCondition, AbstractResidualCondition
from ._ir import (
    AbstractConditionOperator,
    ArrayCodomain,
    Condition,
    ConditionCodomain,
    FieldCodomain,
    FieldSpec,
    OperatorCapabilities,
    OperatorLinearization,
    ProductFieldSpec,
    validate_codomain_value,
)
from ._relations import Equality


def _exact_keys(
    values: Mapping[str, Any], expected: tuple[str, ...], /, *, kind: str
) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{kind} values must be a mapping.")
    keys = tuple(values.keys())
    missing = tuple(name for name in expected if name not in values)
    extra = tuple(name for name in keys if name not in expected)
    if missing or extra:
        raise ValueError(
            f"{kind} fields do not match the declaration; missing={missing!r}, "
            f"extra={extra!r}."
        )


def _bind_sources(
    fields: ProductFieldSpec, values: Mapping[str, Any], /
) -> tuple[frozendict[str, Any], frozendict[str, Any]]:
    _exact_keys(values, fields.sources, kind="Bound source")
    source = frozendict(
        {
            field.source: validate_codomain_value(
                field.codomain, values[field.source], path=f"source {field.source!r}"
            )
            for field in fields.fields
        }
    )
    local = frozendict({field.name: source[field.source] for field in fields.fields})
    return source, local


def _validate_local_values(
    fields: ProductFieldSpec, values: Mapping[str, Any], /, *, kind: str
) -> frozendict[str, Any]:
    _exact_keys(values, fields.names, kind=kind)
    return frozendict(
        {
            field.name: validate_codomain_value(
                field.codomain,
                values[field.name],
                path=f"{kind.lower()} {field.name!r}",
            )
            for field in fields.fields
        }
    )


class BoundCondition(StrictModule):
    """A condition with every declared external source bound exactly once."""

    condition: Condition
    source: frozendict[str, Any]
    values: frozendict[str, Any]
    bound_id: str = eqx.field(static=True)

    def __init__(self, condition: Condition, values: Mapping[str, Any], /):
        if not isinstance(condition, Condition):
            raise TypeError("BoundCondition.condition must be a Condition.")
        source, local = _bind_sources(condition.fields, values)
        self.condition = condition
        self.source = source
        self.values = local
        self.bound_id = canonical_fingerprint(
            {
                "kind": "bound-condition",
                "condition": condition.condition_id,
                "field_spec": condition.fields.field_spec_id,
                "sources": condition.fields.sources,
            }
        )

    @property
    def condition_id(self) -> str:
        return self.condition.condition_id

    @property
    def operator(self) -> AbstractConditionOperator:
        return self.condition.operator

    @property
    def codomain(self) -> ConditionCodomain:
        return self.condition.codomain

    @property
    def relation(self):
        return self.condition.relation

    @property
    def evidence(self) -> Any:
        return self.condition.evidence

    def apply(self, /, *, key: Any | None = None, **kwargs: Any) -> Any:
        value = self.operator.apply(self.values, key=key, **kwargs)
        return validate_codomain_value(
            self.codomain, value, path=f"condition {self.condition_id!r} output"
        )

    def linear_action(
        self,
        values: Mapping[str, Any] | None = None,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.operator.capabilities.is_linear:
            raise TypeError("This condition operator has no certified linear action.")
        arguments = (
            self.values
            if values is None
            else _validate_local_values(
                self.condition.fields, values, kind="Linear-action"
            )
        )
        value = self.operator.linear_action(arguments, key=key, **kwargs)
        return validate_codomain_value(
            self.codomain,
            value,
            path=f"condition {self.condition_id!r} linear action",
        )

    def adjoint_action(
        self,
        value: Any,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> frozendict[str, Any]:
        if not self.operator.capabilities.has_adjoint:
            raise TypeError("This condition operator has no certified adjoint action.")
        cotangent = validate_codomain_value(
            self.codomain, value, path=f"condition {self.condition_id!r} cotangent"
        )
        result = self.operator.adjoint_action(cotangent, key=key, **kwargs)
        return _validate_local_values(
            self.condition.fields, result, kind="Adjoint-action"
        )

    def linearize(
        self, /, *, key: Any | None = None, **kwargs: Any
    ) -> OperatorLinearization:
        if not self.operator.capabilities.has_linearization:
            raise TypeError("This condition operator has no certified linearization.")
        result = self.operator.linearize(self.values, key=key, **kwargs)
        if not isinstance(result, OperatorLinearization):
            raise TypeError("Condition linearize() must return OperatorLinearization.")
        value = validate_codomain_value(
            self.codomain,
            result.value,
            path=f"condition {self.condition_id!r} linearization value",
        )
        return OperatorLinearization(value, result.tangent_operator)


class _LegacyResidualOperator(AbstractConditionOperator):
    condition: AbstractResidualCondition
    capabilities: OperatorCapabilities = eqx.field(static=True)

    def __init__(self, condition: AbstractResidualCondition, /):
        self.condition = condition
        self.capabilities = OperatorCapabilities()

    def apply(self, values, /, *, key=None, **kwargs):
        del key, kwargs
        return self.condition.residual(values)

    def linear_action(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError("Legacy residual callables do not certify linearity.")

    def adjoint_action(self, value, /, *, key=None, **kwargs):
        del value, key, kwargs
        raise TypeError("Legacy residual callables do not certify an adjoint.")

    def linearize(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError("Legacy residual callables do not certify a linearization.")


class _LegacyMomentOperator(AbstractConditionOperator):
    condition: AbstractMomentCondition
    capabilities: OperatorCapabilities = eqx.field(static=True)

    def __init__(self, condition: AbstractMomentCondition, /):
        self.condition = condition
        self.capabilities = OperatorCapabilities()

    def apply(self, values, /, *, key=None, **kwargs):
        if "reduction" not in kwargs:
            raise TypeError(
                "Moment evaluation requires reduction=PreparedLinearReduction."
            )
        reduction = kwargs.pop("reduction")
        return reduction.apply(self.condition.integrand(values), key=key, **kwargs)

    def linear_action(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError("Legacy moment callables do not certify linearity.")

    def adjoint_action(self, value, /, *, key=None, **kwargs):
        del value, key, kwargs
        raise TypeError("Legacy moment callables do not certify an adjoint.")

    def linearize(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError("Legacy moment callables do not certify a linearization.")


def bind_condition(condition: Condition, values: Mapping[str, Any], /) -> BoundCondition:
    return BoundCondition(condition, values)


def lower_condition(
    condition: AbstractCondition | Condition,
    /,
    *,
    fields: ProductFieldSpec | Sequence[FieldSpec] | None = None,
    codomain: ConditionCodomain | None = None,
    condition_id: str | None = None,
) -> Condition:
    """Expose an existing condition through the typed condition IR."""
    if isinstance(condition, Condition):
        if fields is not None or codomain is not None or condition_id is not None:
            raise ValueError(
                "An existing Condition cannot be re-declared while lowering."
            )
        return condition
    source_fields = (
        ProductFieldSpec(
            tuple(
                FieldSpec(name, FieldCodomain(condition.on)) for name in condition.fields
            )
        )
        if fields is None
        else fields
        if isinstance(fields, ProductFieldSpec)
        else ProductFieldSpec(fields)
    )
    if source_fields.names != condition.fields:
        raise ValueError("Legacy lowering must preserve constructor field order.")
    identifier = (
        str(condition_id)
        if condition_id is not None
        else condition.label
        or f"legacy:{type(condition).__module__}.{type(condition).__qualname__}:"
        + ",".join(condition.fields)
    )
    if isinstance(condition, AbstractResidualCondition):
        output = FieldCodomain(condition.on) if codomain is None else codomain
        if not isinstance(output, FieldCodomain):
            raise TypeError("Legacy residual conditions require a FieldCodomain output.")
        return Condition(
            identifier,
            source_fields,
            _LegacyResidualOperator(condition),
            output,
            Equality(),
            label=condition.label,
        )
    if isinstance(condition, AbstractMomentCondition):
        output = (
            ArrayCodomain.from_shape(condition.target.shape, dtype=condition.target.dtype)
            if codomain is None
            else codomain
        )
        if not isinstance(output, ArrayCodomain):
            raise TypeError("Legacy moment conditions require an ArrayCodomain output.")
        return Condition(
            identifier,
            source_fields,
            _LegacyMomentOperator(condition),
            output,
            Equality(condition.target),
            label=condition.label,
        )
    raise TypeError("Expected a typed or legacy condition declaration.")


__all__ = ["BoundCondition", "bind_condition", "lower_condition"]
