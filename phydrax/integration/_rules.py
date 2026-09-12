#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple, Sequence, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._numerics import (
    AdaptiveCubatureRuleData,
    clenshaw_curtis_data,
    gauss_kronrod_data,
    gauss_legendre_data,
    genz_malik_rule_data,
    QuadratureRuleData,
    tanh_sinh_data,
    tensor_product_cubature_rule_data,
)
from .._polynomial._cubature import (
    cubature_rule_data,
    CubatureReference,
    CubatureRuleData,
)
from .._polynomial._gaussian_cubature import (
    gaussian_cubature_rule_data,
    GaussianCubatureFamily,
)
from .._polynomial._orthogonal import (
    legendre_rule_data,
    OrthogonalRuleData,
    standard_normal_hermite_rule_data,
)
from .._strict import StrictModule


class ReferenceCellData(NamedTuple):
    """Points and weights on a canonical reference cell."""

    points: Array
    weights: Array
    embedded_weights: Array | None
    cell: str


class CubatureRule(StrictModule):
    """Curated positive cubature on a canonical multidimensional reference."""

    prepared: CubatureRuleData
    requested_degree: int = eqx.field(static=True)
    allow_duffy_fallback: bool = eqx.field(static=True)

    def __init__(
        self,
        reference: CubatureReference,
        degree: int,
        /,
        *,
        allow_duffy_fallback: bool = True,
        maximum_rule_bytes: int = 64 * 1024**2,
    ):
        prepared = cubature_rule_data(
            reference,
            degree,
            allow_duffy_fallback=allow_duffy_fallback,
            maximum_rule_bytes=maximum_rule_bytes,
        )
        self.prepared = prepared
        self.requested_degree = int(degree)
        self.allow_duffy_fallback = bool(allow_duffy_fallback)

    @property
    def reference_domain(self) -> str:
        return self.prepared.reference_domain

    @property
    def family(self) -> str:
        return self.prepared.family

    @property
    def exact_degree(self) -> int:
        return self.prepared.exact_degree

    @property
    def num_points(self) -> int:
        return int(self.prepared.weights.shape[0])

    @property
    def measure_mass(self) -> float:
        return self.prepared.measure_mass

    @property
    def storage_bytes(self) -> int:
        return self.prepared.storage_bytes

    @property
    def source_id(self) -> str:
        return self.prepared.source_id

    @property
    def rule_id(self) -> str:
        return self.prepared.rule_id

    def materialize(self) -> ReferenceCellData:
        return ReferenceCellData(
            self.prepared.points,
            self.prepared.weights,
            None,
            self.reference_domain,
        )


class GaussianCubatureRule(StrictModule):
    """Positive total-degree cubature for a multivariate standard normal."""

    prepared: CubatureRuleData
    requested_degree: int = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        degree: int,
        /,
        *,
        family: GaussianCubatureFamily = "auto",
        maximum_points: int = 65_536,
        maximum_rule_bytes: int = 64 * 1024**2,
    ):
        self.prepared = gaussian_cubature_rule_data(
            dimension,
            degree,
            family=family,
            maximum_points=maximum_points,
            maximum_rule_bytes=maximum_rule_bytes,
        )
        self.requested_degree = int(degree)

    @property
    def dimension(self) -> int:
        return int(self.prepared.points.shape[1])

    @property
    def family(self) -> str:
        return self.prepared.family

    @property
    def exact_degree(self) -> int:
        return self.prepared.exact_degree

    @property
    def num_points(self) -> int:
        return int(self.prepared.weights.shape[0])

    @property
    def storage_bytes(self) -> int:
        return self.prepared.storage_bytes

    @property
    def source_id(self) -> str:
        return self.prepared.source_id

    @property
    def rule_id(self) -> str:
        return self.prepared.rule_id


class GenzMalikRule(StrictModule):
    """Embedded fully symmetric cubature on ``[-1, 1]^dimension``."""

    prepared: AdaptiveCubatureRuleData
    degree: int = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        degree: int = 9,
        *,
        maximum_points: int = 65_536,
        maximum_rule_bytes: int = 64 * 1024**2,
    ):
        self.prepared = genz_malik_rule_data(
            dimension,
            degree,
            maximum_points=maximum_points,
            maximum_rule_bytes=maximum_rule_bytes,
        )
        self.degree = int(degree)

    @property
    def dimension(self) -> int:
        return self.prepared.dimension

    @property
    def family(self) -> str:
        return self.prepared.family

    @property
    def exact_degree(self) -> int:
        if self.prepared.exact_degree is None:
            raise RuntimeError("Genz--Malik exact degree is unavailable.")
        return self.prepared.exact_degree

    @property
    def embedded_degree(self) -> int:
        if self.prepared.embedded_degree is None:
            raise RuntimeError("Genz--Malik embedded degree is unavailable.")
        return self.prepared.embedded_degree

    @property
    def num_points(self) -> int:
        return self.prepared.num_points

    @property
    def storage_bytes(self) -> int:
        return self.prepared.storage_bytes

    @property
    def source_id(self) -> str:
        return self.prepared.source_id

    @property
    def rule_id(self) -> str:
        return self.prepared.rule_id

    @property
    def weight_l1_norm(self) -> float:
        return self.prepared.weight_l1_norm

    @property
    def negative_weight_mass(self) -> float:
        return self.prepared.negative_weight_mass


class GaussLegendreRule(StrictModule):
    """Order-``n`` Gauss--Legendre quadrature on ``[-1, 1]``."""

    order: int = eqx.field(static=True)

    def __init__(self, order: int = 32):
        order_ = int(order)
        gauss_legendre_data(order_)
        self.order = order_

    @property
    def exact_degree(self) -> int:
        return 2 * self.order - 1

    def data(self) -> QuadratureRuleData:
        return gauss_legendre_data(self.order)


class GaussLobattoLegendreRule(StrictModule):
    """Order-``n`` Gauss--Lobatto--Legendre quadrature on ``[-1, 1]``."""

    order: int = eqx.field(static=True)

    def __init__(self, order: int = 8):
        order_ = int(order)
        legendre_rule_data(order_, "lobatto")
        self.order = order_

    @property
    def exact_degree(self) -> int:
        return 2 * self.order - 3

    def data(self) -> QuadratureRuleData:
        data = legendre_rule_data(self.order, "lobatto")
        return QuadratureRuleData(
            data.nodes,
            data.weights,
            None,
            data.exact_degree,
        )


class GaussHermiteRule(StrictModule):
    """Order-``n`` Gaussian quadrature for standard-normal expectations."""

    order: int = eqx.field(static=True)

    def __init__(self, order: int = 16):
        data = standard_normal_hermite_rule_data(order)
        self.order = int(data.nodes.shape[0])

    def data(self) -> OrthogonalRuleData:
        return standard_normal_hermite_rule_data(self.order)


class GaussKronrodRule(StrictModule):
    """Embedded Gauss--Kronrod quadrature on ``[-1, 1]``."""

    order: int = eqx.field(static=True)

    def __init__(self, order: int = 21):
        order_ = int(order)
        gauss_kronrod_data(order_)
        self.order = order_

    def data(self) -> QuadratureRuleData:
        return gauss_kronrod_data(self.order)


class TensorProductCubatureRule(StrictModule):
    """Tensor product of embedded Gauss--Kronrod interval rules."""

    prepared: AdaptiveCubatureRuleData
    rules: tuple[GaussKronrodRule, ...]

    def __init__(
        self,
        rules: GaussKronrodRule | Sequence[GaussKronrodRule],
        /,
        *,
        dimension: int | None = None,
        maximum_points: int = 65_536,
        maximum_rule_bytes: int = 64 * 1024**2,
    ):
        if isinstance(rules, GaussKronrodRule):
            if dimension is None:
                raise ValueError(
                    "dimension is required when one GaussKronrodRule is repeated."
                )
            if isinstance(dimension, bool) or not isinstance(dimension, Integral):
                raise TypeError("Tensor cubature dimension must be an integer.")
            dimension_ = int(dimension)
            if dimension_ < 1:
                raise ValueError("Tensor cubature dimension must be positive.")
            axis_rules = (rules,) * dimension_
        else:
            if dimension is not None:
                raise ValueError(
                    "dimension must not be given with one rule per tensor axis."
                )
            axis_rules = tuple(rules)
            if not axis_rules:
                raise ValueError("Tensor cubature requires at least one axis rule.")
        if any(not isinstance(rule, GaussKronrodRule) for rule in axis_rules):
            raise TypeError(
                "Tensor cubature axes must use embedded GaussKronrodRule instances."
            )
        self.prepared = tensor_product_cubature_rule_data(
            tuple(rule.data() for rule in axis_rules),
            source_ids=tuple(f"gauss-kronrod:{rule.order}" for rule in axis_rules),
            maximum_points=maximum_points,
            maximum_rule_bytes=maximum_rule_bytes,
        )
        self.rules = axis_rules

    @property
    def dimension(self) -> int:
        return self.prepared.dimension

    @property
    def family(self) -> str:
        return self.prepared.family

    @property
    def exact_degree(self) -> int | None:
        return self.prepared.exact_degree

    @property
    def embedded_degree(self) -> int | None:
        return self.prepared.embedded_degree

    @property
    def num_points(self) -> int:
        return self.prepared.num_points

    @property
    def storage_bytes(self) -> int:
        return self.prepared.storage_bytes

    @property
    def source_id(self) -> str:
        return self.prepared.source_id

    @property
    def rule_id(self) -> str:
        return self.prepared.rule_id

    @property
    def weight_l1_norm(self) -> float:
        return self.prepared.weight_l1_norm

    @property
    def negative_weight_mass(self) -> float:
        return self.prepared.negative_weight_mass


AdaptiveCubatureRule: TypeAlias = GenzMalikRule | TensorProductCubatureRule


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
    GaussLegendreRule
    | GaussLobattoLegendreRule
    | GaussKronrodRule
    | ClenshawCurtisRule
    | TanhSinhRule
)
ProbabilityRule: TypeAlias = GaussHermiteRule | GaussianCubatureRule


def probability_rule_data(
    rule: ProbabilityRule, /
) -> OrthogonalRuleData | CubatureRuleData:
    if isinstance(rule, GaussHermiteRule):
        return rule.data()
    if isinstance(rule, GaussianCubatureRule):
        return rule.prepared
    raise TypeError(f"Unsupported probability rule {type(rule).__name__}.")


def interval_rule_data(rule: IntervalRule, /) -> QuadratureRuleData:
    if isinstance(rule, GaussLegendreRule):
        return rule.data()
    if isinstance(rule, GaussLobattoLegendreRule):
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


class ReferencePrismRule(StrictModule):
    """Triangle-times-interval quadrature on the unit prism."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule | None = None):
        self.rule = GaussLegendreRule(6) if rule is None else rule

    def materialize(self) -> ReferenceCellData:
        triangle = ReferenceTriangleRule(self.rule).materialize()
        axis, weights, _ = _unit_interval_data(self.rule)
        triangle_count = triangle.points.shape[0]
        points = jnp.concatenate(
            (
                jnp.broadcast_to(
                    triangle.points[:, None, :],
                    (triangle_count, axis.shape[0], 2),
                ),
                jnp.broadcast_to(
                    axis[None, :, None],
                    (triangle_count, axis.shape[0], 1),
                ),
            ),
            axis=-1,
        )
        combined = triangle.weights[:, None] * weights[None, :]
        return ReferenceCellData(
            points.reshape((-1, 3)), combined.reshape((-1,)), None, "prism"
        )


class ReferencePyramidRule(StrictModule):
    """Collapsed-cube quadrature on a square-base unit pyramid."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule | None = None):
        self.rule = GaussLegendreRule(8) if rule is None else rule

    def materialize(self) -> ReferenceCellData:
        axis, weights, _ = _unit_interval_data(self.rule)
        first, second, height = jnp.meshgrid(axis, axis, axis, indexing="ij")
        scale = 1.0 - height
        points = jnp.stack(
            (
                scale * first + 0.5 * height,
                scale * second + 0.5 * height,
                height,
            ),
            axis=-1,
        )
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * scale**2
        )
        return ReferenceCellData(
            points.reshape((-1, 3)), combined.reshape((-1,)), None, "pyramid"
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
    | ReferencePrismRule
    | ReferencePyramidRule
    | ReferenceHexahedronRule
    | CubatureRule
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
    if isinstance(rule, ReferencePrismRule):
        return rule.materialize()
    if isinstance(rule, ReferencePyramidRule):
        return rule.materialize()
    if isinstance(rule, ReferenceHexahedronRule):
        return rule.materialize()
    if isinstance(rule, CubatureRule):
        return rule.materialize()
    raise TypeError(f"Unsupported reference-cell rule {type(rule).__name__}.")


__all__ = [
    "AdaptiveCubatureRule",
    "ClenshawCurtisRule",
    "CubatureRule",
    "GaussKronrodRule",
    "GenzMalikRule",
    "GaussianCubatureRule",
    "GaussHermiteRule",
    "GaussLegendreRule",
    "GaussLobattoLegendreRule",
    "IntervalRule",
    "ProbabilityRule",
    "ReferenceCellData",
    "ReferenceHexahedronRule",
    "ReferenceIntervalRule",
    "ReferencePrismRule",
    "ReferencePyramidRule",
    "ReferenceQuadrilateralRule",
    "ReferenceRule",
    "ReferenceTetrahedronRule",
    "ReferenceTriangleRule",
    "TensorProductCubatureRule",
    "TanhSinhRule",
    "interval_rule_data",
    "probability_rule_data",
    "reference_rule_data",
]
