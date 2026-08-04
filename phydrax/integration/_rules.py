#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._numerics import (
    clenshaw_curtis_data,
    gauss_kronrod_data,
    gauss_legendre_data,
    QuadratureRuleData,
    tanh_sinh_data,
)
from .._strict import StrictModule


class ReferenceCellData(NamedTuple):
    """Points and weights on a canonical reference cell."""

    points: Array
    weights: Array
    embedded_weights: Array | None
    cell: str


class GaussLegendreRule(StrictModule):
    """Order-``n`` Gauss--Legendre quadrature on ``[-1, 1]``."""

    order: int = eqx.field(static=True)

    def __init__(self, order: int = 32):
        order_ = int(order)
        gauss_legendre_data(order_)
        self.order = order_

    def data(self) -> QuadratureRuleData:
        return gauss_legendre_data(self.order)


class GaussKronrodRule(StrictModule):
    """Embedded Gauss--Kronrod quadrature on ``[-1, 1]``."""

    order: int = eqx.field(static=True)

    def __init__(self, order: int = 21):
        order_ = int(order)
        gauss_kronrod_data(order_)
        self.order = order_

    def data(self) -> QuadratureRuleData:
        return gauss_kronrod_data(self.order)


class ClenshawCurtisRule(StrictModule):
    """Nested endpoint-including Clenshaw--Curtis quadrature."""

    level: int = eqx.field(static=True)

    def __init__(self, level: int = 5):
        level_ = int(level)
        if level_ < 1:
            raise ValueError("Clenshaw--Curtis level must be positive.")
        self.level = level_

    @property
    def order(self) -> int:
        return 2**self.level + 1

    def data(self) -> QuadratureRuleData:
        return clenshaw_curtis_data(self.order)


class TanhSinhRule(StrictModule):
    """Double-exponential quadrature with ``20 × level + 1`` nodes."""

    level: int = eqx.field(static=True)

    def __init__(self, level: int = 3):
        level_ = int(level)
        if level_ < 1:
            raise ValueError("Tanh--sinh level must be positive.")
        self.level = level_

    @property
    def order(self) -> int:
        return 20 * self.level + 1

    def data(self) -> QuadratureRuleData:
        return tanh_sinh_data(self.order)


IntervalRule: TypeAlias = (
    GaussLegendreRule | GaussKronrodRule | ClenshawCurtisRule | TanhSinhRule
)


def interval_rule_data(rule: IntervalRule, /) -> QuadratureRuleData:
    if isinstance(rule, GaussLegendreRule):
        return rule.data()
    if isinstance(rule, GaussKronrodRule):
        return rule.data()
    if isinstance(rule, ClenshawCurtisRule):
        return rule.data()
    if isinstance(rule, TanhSinhRule):
        return rule.data()
    raise TypeError(f"Unsupported interval rule {type(rule).__name__}.")


def _unit_interval_data(rule: IntervalRule, /) -> tuple[Array, Array, Array | None]:
    data = interval_rule_data(rule)
    points = 0.5 * (data.nodes + 1.0)
    weights = 0.5 * data.weights
    embedded = None
    if data.embedded_weights is not None:
        embedded = 0.5 * data.embedded_weights
    return points, weights, embedded


class ReferenceIntervalRule(StrictModule):
    """Quadrature on the unit reference interval."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule | None = None):
        self.rule = GaussLegendreRule() if rule is None else rule

    def materialize(self) -> ReferenceCellData:
        points, weights, embedded = _unit_interval_data(self.rule)
        return ReferenceCellData(points[:, None], weights, embedded, "interval")


class ReferenceTriangleRule(StrictModule):
    """Duffy-mapped tensor quadrature on the unit right triangle."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule | None = None):
        self.rule = GaussLegendreRule(8) if rule is None else rule

    def materialize(self) -> ReferenceCellData:
        axis, weights, _ = _unit_interval_data(self.rule)
        first, second = jnp.meshgrid(axis, axis, indexing="ij")
        points = jnp.stack((first, (1.0 - first) * second), axis=-1)
        combined = weights[:, None] * weights[None, :] * (1.0 - first)
        return ReferenceCellData(
            points.reshape((-1, 2)), combined.reshape((-1,)), None, "triangle"
        )


class ReferenceQuadrilateralRule(StrictModule):
    """Tensor quadrature on the unit square."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule | None = None):
        self.rule = GaussLegendreRule(8) if rule is None else rule

    def materialize(self) -> ReferenceCellData:
        axis, weights, _ = _unit_interval_data(self.rule)
        first, second = jnp.meshgrid(axis, axis, indexing="ij")
        points = jnp.stack((first, second), axis=-1)
        combined = weights[:, None] * weights[None, :]
        return ReferenceCellData(
            points.reshape((-1, 2)), combined.reshape((-1,)), None, "quadrilateral"
        )


class ReferenceTetrahedronRule(StrictModule):
    """Duffy-mapped tensor quadrature on the unit tetrahedron."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule | None = None):
        self.rule = GaussLegendreRule(6) if rule is None else rule

    def materialize(self) -> ReferenceCellData:
        axis, weights, _ = _unit_interval_data(self.rule)
        first, second, third = jnp.meshgrid(axis, axis, axis, indexing="ij")
        one_minus_first = 1.0 - first
        one_minus_second = 1.0 - second
        points = jnp.stack(
            (
                first,
                one_minus_first * second,
                one_minus_first * one_minus_second * third,
            ),
            axis=-1,
        )
        jacobian = one_minus_first**2 * one_minus_second
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * jacobian
        )
        return ReferenceCellData(
            points.reshape((-1, 3)), combined.reshape((-1,)), None, "tetrahedron"
        )


class ReferenceHexahedronRule(StrictModule):
    """Tensor quadrature on the unit cube."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule | None = None):
        self.rule = GaussLegendreRule(6) if rule is None else rule

    def materialize(self) -> ReferenceCellData:
        axis, weights, _ = _unit_interval_data(self.rule)
        first, second, third = jnp.meshgrid(axis, axis, axis, indexing="ij")
        points = jnp.stack((first, second, third), axis=-1)
        combined = (
            weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        )
        return ReferenceCellData(
            points.reshape((-1, 3)), combined.reshape((-1,)), None, "hexahedron"
        )


ReferenceRule: TypeAlias = (
    ReferenceIntervalRule
    | ReferenceTriangleRule
    | ReferenceQuadrilateralRule
    | ReferenceTetrahedronRule
    | ReferenceHexahedronRule
)


def reference_rule_data(rule: ReferenceRule, /) -> ReferenceCellData:
    if isinstance(rule, ReferenceIntervalRule):
        return rule.materialize()
    if isinstance(rule, ReferenceTriangleRule):
        return rule.materialize()
    if isinstance(rule, ReferenceQuadrilateralRule):
        return rule.materialize()
    if isinstance(rule, ReferenceTetrahedronRule):
        return rule.materialize()
    if isinstance(rule, ReferenceHexahedronRule):
        return rule.materialize()
    raise TypeError(f"Unsupported reference-cell rule {type(rule).__name__}.")


__all__ = [
    "ClenshawCurtisRule",
    "GaussKronrodRule",
    "GaussLegendreRule",
    "IntervalRule",
    "ReferenceCellData",
    "ReferenceHexahedronRule",
    "ReferenceIntervalRule",
    "ReferenceQuadrilateralRule",
    "ReferenceRule",
    "ReferenceTetrahedronRule",
    "ReferenceTriangleRule",
    "TanhSinhRule",
    "interval_rule_data",
    "reference_rule_data",
]
