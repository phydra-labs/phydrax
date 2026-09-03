#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.domain import ComponentSum, Domain, DomainComponent, DomainFunction

from .._strict import AbstractAttribute, StrictModule


ConditionSupport = DomainComponent | ComponentSum


def _fields(value: str | Sequence[str], /) -> tuple[str, ...]:
    fields = (value,) if isinstance(value, str) else tuple(str(name) for name in value)
    if not fields or any(not name for name in fields):
        raise ValueError("Condition fields must be non-empty names.")
    if len(set(fields)) != len(fields):
        raise ValueError("Condition fields must be unique.")
    return fields


def _validate_support(value: Any, /) -> ConditionSupport:
    if not isinstance(value, (DomainComponent, ComponentSum)):
        raise TypeError("Condition support must be a DomainComponent or ComponentSum.")
    return value


def _condition_functions(
    fields: tuple[str, ...],
    functions: Mapping[str, DomainFunction],
    /,
) -> tuple[DomainFunction, ...]:
    missing = tuple(name for name in fields if name not in functions)
    if missing:
        raise KeyError(f"Missing condition fields {missing!r}.")
    values = tuple(functions[name] for name in fields)
    if any(not isinstance(value, DomainFunction) for value in values):
        raise TypeError("Condition fields must map to DomainFunction values.")
    return values


def _same_support(left: Domain, right: Domain, /) -> bool:
    if left is right:
        return True
    if any(
        isinstance(leaf, jax_core.Tracer)
        for domain in (left, right)
        for leaf in jax.tree_util.tree_leaves(domain)
    ):
        return left.schema_compatible(right)
    return left.same_support(right)


class AbstractCondition(StrictModule):
    """Declarative scientific condition independent of numerical realization."""

    fields: AbstractAttribute[tuple[str, ...]]
    on: AbstractAttribute[ConditionSupport]
    label: AbstractAttribute[str | None]

    def as_condition(
        self,
        *,
        fields: Any = None,
        codomain: Any = None,
        condition_id: str | None = None,
    ):
        """Lower this legacy declaration to the typed condition IR."""
        from ._lowering import lower_condition

        return lower_condition(
            self,
            fields=fields,
            codomain=codomain,
            condition_id=condition_id,
        )


class AbstractResidualCondition(AbstractCondition):
    """Condition represented by a pointwise residual field."""

    @abstractmethod
    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        raise NotImplementedError


class AbstractMomentCondition(AbstractCondition):
    """Condition represented by an integrated moment target."""

    target: AbstractAttribute[Array]

    @abstractmethod
    def integrand(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        raise NotImplementedError


class Residual(AbstractResidualCondition):
    """Generic residual condition over named domain functions."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: ConditionSupport
    operator: Callable[..., DomainFunction] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        fields: str | Sequence[str],
        on: ConditionSupport,
        operator: Callable[..., DomainFunction],
        /,
        *,
        label: str | None = None,
    ):
        if not callable(operator):
            raise TypeError("Residual condition operator must be callable.")
        self.fields = _fields(fields)
        self.on = _validate_support(on)
        self.operator = operator
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        result = self.operator(*_condition_functions(self.fields, functions))
        if not isinstance(result, DomainFunction):
            raise TypeError("Residual condition operators must return a DomainFunction.")
        if not _same_support(result.domain, self.on.domain):
            raise ValueError("Residual domain is incompatible with condition support.")
        return result


class Moment(AbstractMomentCondition):
    """Generic integrated moment equality over named domain functions."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: ConditionSupport
    operator: Callable[..., DomainFunction] = eqx.field(static=True)
    target: Array
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        fields: str | Sequence[str],
        on: ConditionSupport,
        operator: Callable[..., DomainFunction],
        /,
        *,
        target: ArrayLike = 0.0,
        label: str | None = None,
    ):
        if not callable(operator):
            raise TypeError("Moment condition operator must be callable.")
        self.fields = _fields(fields)
        self.on = _validate_support(on)
        self.operator = operator
        self.target = jnp.asarray(target)
        self.label = None if label is None else str(label)

    def integrand(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        result = self.operator(*_condition_functions(self.fields, functions))
        if not isinstance(result, DomainFunction):
            raise TypeError("Moment condition operators must return a DomainFunction.")
        if not _same_support(result.domain, self.on.domain):
            raise ValueError("Moment integrand domain is incompatible with its support.")
        return result


class Observation(AbstractResidualCondition):
    """Observed target field compared with an operator on model fields."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: ConditionSupport
    operator: Callable[..., DomainFunction] = eqx.field(static=True)
    target: DomainFunction
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        fields: str | Sequence[str],
        on: ConditionSupport,
        target: DomainFunction,
        /,
        *,
        operator: Callable[..., DomainFunction] | None = None,
        label: str | None = None,
    ):
        resolved_fields = _fields(fields)
        if not isinstance(target, DomainFunction):
            raise TypeError("Observation target must be a DomainFunction.")
        if not target.domain.same_support(on.domain):
            raise ValueError("Observation target domain is incompatible with support.")
        if operator is None:
            if len(resolved_fields) != 1:
                raise ValueError("The identity observation operator requires one field.")
            operator = lambda value: value
        if not callable(operator):
            raise TypeError("Observation operator must be callable.")
        self.fields = resolved_fields
        self.on = _validate_support(on)
        self.operator = operator
        self.target = target
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        prediction = self.operator(*_condition_functions(self.fields, functions))
        if not isinstance(prediction, DomainFunction):
            raise TypeError("Observation operators must return a DomainFunction.")
        return prediction - self.target


__all__ = [
    "AbstractCondition",
    "AbstractMomentCondition",
    "AbstractResidualCondition",
    "ConditionSupport",
    "Moment",
    "Observation",
    "Residual",
]
