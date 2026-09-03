#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any, TypeAlias

import equinox as eqx

from .._strict import StrictModule
from ._ir import (
    ArrayCodomain,
    codomains_compatible,
    ConditionCodomain,
    FieldCodomain,
    ProductCodomain,
    validate_codomain_value,
)


class AbstractConditionRelation(StrictModule):
    """A typed relation over one declared condition codomain."""

    @abstractmethod
    def validate(self, codomain: ConditionCodomain, /) -> None:
        raise NotImplementedError


class Equality(AbstractConditionRelation):
    """Exact equality to a codomain-shaped target, or to the additive zero."""

    target: Any
    has_target: bool = eqx.field(static=True)

    def __init__(self, target: Any = None, /):
        self.target = target
        self.has_target = target is not None

    def validate(self, codomain: ConditionCodomain, /) -> None:
        if self.has_target:
            validate_codomain_value(codomain, self.target, path="equality target")


class Inequality(AbstractConditionRelation):
    """Pointwise lower and/or upper bounds with exact codomain shapes."""

    lower: Any
    upper: Any
    has_lower: bool = eqx.field(static=True)
    has_upper: bool = eqx.field(static=True)
    strict_lower: bool = eqx.field(static=True)
    strict_upper: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        lower: Any = None,
        upper: Any = None,
        strict_lower: bool = False,
        strict_upper: bool = False,
    ):
        has_lower = lower is not None
        has_upper = upper is not None
        if not has_lower and not has_upper:
            raise ValueError("Inequality requires a lower or upper bound.")
        if strict_lower and not has_lower:
            raise ValueError("strict_lower requires a lower bound.")
        if strict_upper and not has_upper:
            raise ValueError("strict_upper requires an upper bound.")
        self.lower = lower
        self.upper = upper
        self.has_lower = has_lower
        self.has_upper = has_upper
        self.strict_lower = bool(strict_lower)
        self.strict_upper = bool(strict_upper)

    def validate(self, codomain: ConditionCodomain, /) -> None:
        if self.has_lower:
            validate_codomain_value(codomain, self.lower, path="inequality lower bound")
        if self.has_upper:
            validate_codomain_value(codomain, self.upper, path="inequality upper bound")


class ConeKind(str, Enum):
    nonnegative = "nonnegative"
    nonpositive = "nonpositive"
    second_order = "second_order"
    positive_semidefinite = "positive_semidefinite"
    simplex = "simplex"


class ConeMembership(AbstractConditionRelation):
    """Membership in a named closed cone (or in the probability simplex)."""

    cone: ConeKind = eqx.field(static=True)
    axis: int = eqx.field(static=True)

    def __init__(self, cone: ConeKind, /, *, axis: int = -1):
        self.cone = ConeKind(cone)
        self.axis = int(axis)

    def validate(self, codomain: ConditionCodomain, /) -> None:
        fiber = _finite_fiber(codomain)
        if self.cone in (ConeKind.second_order, ConeKind.simplex):
            if not fiber.shape:
                raise ValueError(f"{self.cone.value} membership requires a value axis.")
            axis = self.axis if self.axis >= 0 else len(fiber.shape) + self.axis
            if axis < 0 or axis >= len(fiber.shape):
                raise ValueError("ConeMembership.axis is outside the finite fiber rank.")
            if self.cone is ConeKind.second_order and fiber.shape[axis] < 2:
                raise ValueError(
                    "A second-order cone axis must contain at least two values."
                )
        if self.cone is ConeKind.positive_semidefinite:
            if len(fiber.shape) < 2 or fiber.shape[-1] != fiber.shape[-2]:
                raise ValueError(
                    "Positive-semidefinite membership requires square trailing axes."
                )


class Complementarity(AbstractConditionRelation):
    """Complementarity of two equally typed nonnegative product factors."""

    def __init__(self):
        pass

    def validate(self, codomain: ConditionCodomain, /) -> None:
        if not isinstance(codomain, ProductCodomain) or len(codomain.factors) != 2:
            raise TypeError("Complementarity requires a two-factor ProductCodomain.")
        left, right = codomain.factors
        if not codomains_compatible(left, right):
            raise ValueError("Complementarity factors must have compatible codomains.")


class NoisyObservation(AbstractConditionRelation):
    """A finite observation with explicitly shaped, positive noise scales."""

    observed: Any
    noise_scale: Any

    def __init__(self, observed: Any, noise_scale: Any, /):
        self.observed = observed
        self.noise_scale = noise_scale

    def validate(self, codomain: ConditionCodomain, /) -> None:
        if not isinstance(codomain, ArrayCodomain):
            raise TypeError("NoisyObservation currently requires an ArrayCodomain.")
        observed = validate_codomain_value(codomain, self.observed, path="observed value")
        noise = validate_codomain_value(
            codomain, self.noise_scale, path="observation noise scale"
        )
        del observed
        if bool((noise <= 0).any()):
            raise ValueError("Observation noise scales must be strictly positive.")


def _finite_fiber(codomain: ConditionCodomain, /) -> ArrayCodomain:
    if isinstance(codomain, ArrayCodomain):
        return codomain
    if isinstance(codomain, FieldCodomain):
        return codomain.value
    raise TypeError("This relation requires one array or field codomain, not a product.")


ConditionRelation: TypeAlias = (
    Equality | Inequality | ConeMembership | Complementarity | NoisyObservation
)


def validate_relation(
    relation: ConditionRelation, codomain: ConditionCodomain, /
) -> None:
    if not isinstance(relation, AbstractConditionRelation):
        raise TypeError("Condition.relation must be a typed condition relation.")
    relation.validate(codomain)


__all__ = [
    "AbstractConditionRelation",
    "Complementarity",
    "ConditionRelation",
    "ConeKind",
    "ConeMembership",
    "Equality",
    "Inequality",
    "NoisyObservation",
    "validate_relation",
]
