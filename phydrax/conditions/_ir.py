#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp

from phydrax.domain import DomainFunction

from .._strict import AbstractAttribute, StrictModule
from ._base import _same_support, _validate_support, ConditionSupport


if TYPE_CHECKING:
    from ._relations import ConditionRelation
from .._fingerprint import canonical_fingerprint


class ValueAxis(StrictModule):
    """One ordered finite value axis of an array-valued field."""

    name: str = eqx.field(static=True)
    size: int = eqx.field(static=True)
    labels: tuple[str, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        size: int,
        /,
        *,
        labels: Sequence[str] | None = None,
    ):
        name_ = str(name)
        size_ = int(size)
        labels_ = None if labels is None else tuple(str(label) for label in labels)
        if not name_:
            raise ValueError("Value-axis names must be non-empty.")
        if size_ <= 0:
            raise ValueError("Value-axis sizes must be positive.")
        if labels_ is not None:
            if len(labels_) != size_:
                raise ValueError("Value-axis labels must have exactly axis.size entries.")
            if any(not label for label in labels_) or len(set(labels_)) != len(labels_):
                raise ValueError("Value-axis labels must be non-empty and unique.")
        self.name = name_
        self.size = size_
        self.labels = labels_


class ArrayCodomain(StrictModule):
    """A finite tensor codomain with named, ordered axes and an optional dtype."""

    axes: tuple[ValueAxis, ...]
    dtype: str | None = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[ValueAxis] = (),
        /,
        *,
        dtype: Any | None = None,
    ):
        axes_ = tuple(axes)
        if any(not isinstance(axis, ValueAxis) for axis in axes_):
            raise TypeError("ArrayCodomain axes must be ValueAxis instances.")
        names = tuple(axis.name for axis in axes_)
        if len(set(names)) != len(names):
            raise ValueError("ArrayCodomain axis names must be unique.")
        self.axes = axes_
        self.dtype = None if dtype is None else jnp.dtype(dtype).name

    @classmethod
    def from_shape(
        cls,
        shape: Sequence[int],
        /,
        *,
        axis_names: Sequence[str] | None = None,
        dtype: Any | None = None,
    ) -> ArrayCodomain:
        shape_ = tuple(int(size) for size in shape)
        if any(size <= 0 for size in shape_):
            raise ValueError("ArrayCodomain dimensions must be positive.")
        names = (
            tuple(f"axis_{index}" for index in range(len(shape_)))
            if axis_names is None
            else tuple(str(name) for name in axis_names)
        )
        if len(names) != len(shape_):
            raise ValueError("axis_names must have exactly one name per dimension.")
        return cls(
            tuple(
                ValueAxis(name, size) for name, size in zip(names, shape_, strict=True)
            ),
            dtype=dtype,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)


class FieldCodomain(StrictModule):
    """A field over an explicit support with a declared finite fiber codomain."""

    support: ConditionSupport
    value: ArrayCodomain

    def __init__(
        self,
        support: ConditionSupport,
        value: ArrayCodomain | None = None,
        /,
    ):
        if value is not None and not isinstance(value, ArrayCodomain):
            raise TypeError("FieldCodomain.value must be an ArrayCodomain.")
        self.support = _validate_support(support)
        self.value = ArrayCodomain() if value is None else value


class ProductCodomain(StrictModule):
    """An ordered Cartesian product with no implicit leaf broadcasting."""

    factors: tuple[ConditionCodomain, ...]

    def __init__(self, factors: Sequence[ConditionCodomain], /):
        factors_ = tuple(factors)
        if not factors_:
            raise ValueError("ProductCodomain requires at least one factor.")
        if any(not _is_codomain(factor) for factor in factors_):
            raise TypeError("ProductCodomain factors must be condition codomains.")
        self.factors = factors_


ConditionCodomain: TypeAlias = ArrayCodomain | FieldCodomain | ProductCodomain


def _is_codomain(value: Any, /) -> bool:
    return isinstance(value, (ArrayCodomain, FieldCodomain, ProductCodomain))


def validate_codomain_value(
    codomain: ConditionCodomain,
    value: Any,
    /,
    *,
    path: str = "value",
) -> Any:
    """Validate and normalize one value without changing its declared shape."""
    if isinstance(codomain, ArrayCodomain):
        array = jnp.asarray(value)
        if array.shape != codomain.shape:
            raise ValueError(
                f"{path} has shape {array.shape}; expected exactly {codomain.shape}."
            )
        if codomain.dtype is not None and array.dtype.name != codomain.dtype:
            raise TypeError(
                f"{path} has dtype {array.dtype.name!r}; expected {codomain.dtype!r}."
            )
        return array
    if isinstance(codomain, FieldCodomain):
        if not isinstance(value, DomainFunction):
            raise TypeError(f"{path} must be a DomainFunction.")
        if not _same_support(value.domain, codomain.support.domain):
            raise ValueError(f"{path} has a domain incompatible with its field support.")
        return value
    if not isinstance(codomain, ProductCodomain):
        raise TypeError("codomain must be a condition codomain.")
    if not isinstance(value, tuple):
        raise TypeError(f"{path} must be a tuple for a ProductCodomain.")
    if len(value) != len(codomain.factors):
        raise ValueError(
            f"{path} has {len(value)} product leaves; expected {len(codomain.factors)}."
        )
    return tuple(
        validate_codomain_value(factor, leaf, path=f"{path}[{index}]")
        for index, (factor, leaf) in enumerate(zip(codomain.factors, value, strict=True))
    )


def codomains_compatible(left: ConditionCodomain, right: ConditionCodomain, /) -> bool:
    if isinstance(left, ArrayCodomain) and isinstance(right, ArrayCodomain):
        return left.axes == right.axes and left.dtype == right.dtype
    if isinstance(left, FieldCodomain) and isinstance(right, FieldCodomain):
        return codomains_compatible(left.value, right.value) and _same_support(
            left.support.domain, right.support.domain
        )
    if isinstance(left, ProductCodomain) and isinstance(right, ProductCodomain):
        return len(left.factors) == len(right.factors) and all(
            codomains_compatible(a, b)
            for a, b in zip(left.factors, right.factors, strict=True)
        )
    return False


class FieldSpec(StrictModule):
    """One named operator input and its externally bound source."""

    name: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    codomain: ConditionCodomain

    def __init__(
        self,
        name: str,
        codomain: ConditionCodomain,
        /,
        *,
        source: str | None = None,
    ):
        name_ = str(name)
        source_ = name_ if source is None else str(source)
        if not name_ or not source_:
            raise ValueError("FieldSpec names and sources must be non-empty.")
        if not _is_codomain(codomain):
            raise TypeError("FieldSpec.codomain must be a condition codomain.")
        self.name = name_
        self.source = source_
        self.codomain = codomain


class ProductFieldSpec(StrictModule):
    """An ordered, source-unique collection of condition input specifications."""

    fields: tuple[FieldSpec, ...]
    field_spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[FieldSpec],
        /,
        *,
        field_spec_id: str | None = None,
    ):
        fields_ = tuple(fields)
        if not fields_:
            raise ValueError("ProductFieldSpec requires at least one field.")
        if any(not isinstance(field, FieldSpec) for field in fields_):
            raise TypeError("ProductFieldSpec entries must be FieldSpec instances.")
        names = tuple(field.name for field in fields_)
        sources = tuple(field.source for field in fields_)
        if len(set(names)) != len(names):
            raise ValueError("Condition field names must be unique.")
        if len(set(sources)) != len(sources):
            raise ValueError("Condition field sources must be unique.")
        generated = canonical_fingerprint(
            {
                "kind": "condition-field-spec",
                "fields": tuple(
                    {
                        "name": field.name,
                        "source": field.source,
                        "codomain": _codomain_schema(field.codomain),
                    }
                    for field in fields_
                ),
            }
        )
        identifier = generated if field_spec_id is None else str(field_spec_id)
        if not identifier:
            raise ValueError("ProductFieldSpec.field_spec_id must be non-empty.")
        self.fields = fields_
        self.field_spec_id = identifier

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(field.source for field in self.fields)


def _codomain_schema(codomain: ConditionCodomain, /) -> dict[str, Any]:
    if isinstance(codomain, ArrayCodomain):
        return {
            "kind": "array",
            "axes": tuple((axis.name, axis.size, axis.labels) for axis in codomain.axes),
            "dtype": codomain.dtype,
        }
    if isinstance(codomain, FieldCodomain):
        return {
            "kind": "field",
            "domain_labels": codomain.support.domain.labels,
            "value": _codomain_schema(codomain.value),
        }
    if isinstance(codomain, ProductCodomain):
        return {
            "kind": "product",
            "factors": tuple(_codomain_schema(factor) for factor in codomain.factors),
        }
    raise TypeError("Expected a condition codomain.")


class OperatorCapabilities(StrictModule):
    """Explicitly certified actions implemented by a condition operator."""

    is_linear: bool = eqx.field(static=True)
    has_adjoint: bool = eqx.field(static=True)
    has_linearization: bool = eqx.field(static=True)
    uses_randomness: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        is_linear: bool = False,
        has_adjoint: bool = False,
        has_linearization: bool = False,
        uses_randomness: bool = False,
    ):
        linear = bool(is_linear)
        adjoint = bool(has_adjoint)
        linearization = bool(has_linearization)
        if adjoint and not linear:
            raise ValueError("An adjoint action requires a certified linear operator.")
        if linear and linearization:
            raise ValueError(
                "Globally linear operators expose linear_action directly, not linearize."
            )
        self.is_linear = linear
        self.has_adjoint = adjoint
        self.has_linearization = linearization
        self.uses_randomness = bool(uses_randomness)


class AbstractConditionOperator(StrictModule):
    """Typed action contract for a condition declaration."""

    capabilities: AbstractAttribute[OperatorCapabilities]

    @abstractmethod
    def apply(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def linear_action(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def adjoint_action(
        self,
        value: Any,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def linearize(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> OperatorLinearization:
        raise NotImplementedError


class CallableConditionOperator(AbstractConditionOperator):
    """Apply-only wrapper; raw callables cannot certify linear or adjoint actions."""

    function: Callable[..., Any] = eqx.field(static=True)
    capabilities: OperatorCapabilities = eqx.field(static=True)

    def __init__(self, function: Callable[..., Any], /):
        if not callable(function):
            raise TypeError("CallableConditionOperator requires a callable.")
        self.function = function
        self.capabilities = OperatorCapabilities()

    def apply(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.function(values, key=key, **kwargs)

    def linear_action(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        del values, key, kwargs
        raise TypeError("Raw callable operators do not certify a linear action.")

    def adjoint_action(
        self,
        value: Any,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        del value, key, kwargs
        raise TypeError("Raw callable operators do not certify an adjoint action.")

    def linearize(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> OperatorLinearization:
        del values, key, kwargs
        raise TypeError("Raw callable operators do not certify a linearization.")


class OperatorLinearization(StrictModule):
    """A primal operator value and its certified tangent operator."""

    value: Any
    tangent_operator: AbstractConditionOperator

    def __init__(self, value: Any, tangent_operator: AbstractConditionOperator, /):
        if not isinstance(tangent_operator, AbstractConditionOperator):
            raise TypeError("A linearization tangent must be a condition operator.")
        if not tangent_operator.capabilities.is_linear:
            raise ValueError("A linearization tangent operator must certify linearity.")
        self.value = value
        self.tangent_operator = tangent_operator


class ConditionQuantifier(str, Enum):
    deterministic = "deterministic"
    samplewise = "samplewise"
    almost_sure = "almost_sure"
    expectation = "expectation"
    chance = "chance"


class Condition(StrictModule):
    """A typed relation on the result of an operator over ordered field sources."""

    condition_id: str = eqx.field(static=True)
    fields: ProductFieldSpec
    operator: AbstractConditionOperator
    codomain: ConditionCodomain
    relation: ConditionRelation
    evidence: Any
    quantifier: ConditionQuantifier = eqx.field(static=True)
    probability_level: float | None = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        condition_id: str,
        fields: ProductFieldSpec | Sequence[FieldSpec],
        operator: AbstractConditionOperator,
        codomain: ConditionCodomain,
        relation: ConditionRelation,
        /,
        *,
        quantifier: ConditionQuantifier = ConditionQuantifier.deterministic,
        probability_level: float | None = None,
        label: str | None = None,
        evidence: Any = None,
    ):
        from ._relations import validate_relation

        identifier = str(condition_id)
        fields_ = (
            fields if isinstance(fields, ProductFieldSpec) else ProductFieldSpec(fields)
        )
        if not identifier:
            raise ValueError("Condition.condition_id must be non-empty.")
        if not isinstance(operator, AbstractConditionOperator):
            raise TypeError("Condition.operator must be an AbstractConditionOperator.")
        if not _is_codomain(codomain):
            raise TypeError("Condition.codomain must be a condition codomain.")
        quantifier_ = ConditionQuantifier(quantifier)
        if quantifier_ is ConditionQuantifier.chance:
            if probability_level is None:
                raise ValueError("Chance conditions require probability_level.")
            level = float(probability_level)
            if not 0.0 < level <= 1.0:
                raise ValueError("Chance probability_level must lie in (0, 1].")
        else:
            if probability_level is not None:
                raise ValueError("probability_level is valid only for chance conditions.")
            level = None
        validate_relation(relation, codomain)
        self.condition_id = identifier
        self.fields = fields_
        self.operator = operator
        self.codomain = codomain
        self.relation = relation
        self.evidence = evidence
        self.quantifier = quantifier_
        self.probability_level = level
        self.label = None if label is None else str(label)


__all__ = [
    "AbstractConditionOperator",
    "ArrayCodomain",
    "CallableConditionOperator",
    "Condition",
    "ConditionCodomain",
    "ConditionQuantifier",
    "FieldCodomain",
    "FieldSpec",
    "OperatorCapabilities",
    "OperatorLinearization",
    "ProductCodomain",
    "ProductFieldSpec",
    "ValueAxis",
    "codomains_compatible",
    "validate_codomain_value",
]
