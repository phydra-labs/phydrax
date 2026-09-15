#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


RegulatorName = Literal["optimized", "exponential", "power-law"]


class FunctionalRGStatus(enum.IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    POLE_ENCOUNTERED = 2
    CAPACITY_EXCEEDED = 3
    NOT_CONVERGED = 4


class Regulator(StrictModule, NonTrainableState):
    """Dimensionless bosonic cutoff shape in ``R_k=Z_k k^2 r(p^2/k^2)``."""

    power: float = eqx.field(static=True)
    family: RegulatorName = eqx.field(static=True)
    regulator_id: str = eqx.field(static=True)

    def __init__(self, family: RegulatorName, /, *, power: float = 1.0):
        if family not in ("optimized", "exponential", "power-law"):
            raise ValueError("Unknown functional-RG regulator family.")
        power_ = float(power)
        if not np.isfinite(power_) or power_ <= 0.0:
            raise ValueError("Regulator power must be finite and positive.")
        if family == "optimized" and power_ != 1.0:
            raise ValueError("The optimized regulator has no shape parameter.")
        if family == "exponential" and power_ < 1.0:
            raise ValueError("Exponential regulator power must be at least one.")
        if family == "power-law" and power_ <= 1.0:
            raise ValueError("Power-law regulators require power greater than one.")
        self.power = power_
        self.family = family
        self.regulator_id = canonical_fingerprint(
            {"kind": "wetterich-regulator", "family": family, "power": power_}
        )

    @classmethod
    def optimized(cls) -> "Regulator":
        return cls("optimized")

    @classmethod
    def exponential(cls, power: float = 1.0) -> "Regulator":
        return cls("exponential", power=power)

    @classmethod
    def power_law(cls, power: float = 2.0) -> "Regulator":
        return cls("power-law", power=power)

    def shape(self, momentum_squared: ArrayLike, /) -> Array:
        y = jnp.asarray(momentum_squared)
        if self.family == "optimized":
            return jnp.maximum(1.0 - y, 0.0)
        if self.family == "exponential":
            z = y**self.power
            safe = jnp.where(jnp.abs(z) < 1.0e-5, 1.0, z)
            regular = safe / jnp.expm1(safe)
            series = 1.0 - 0.5 * z + z * z / 12.0
            return jnp.where(jnp.abs(z) < 1.0e-5, series, regular)
        return y ** (-self.power)

    def derivative(self, momentum_squared: ArrayLike, /) -> Array:
        y = jnp.asarray(momentum_squared)
        if self.family == "optimized":
            return jnp.where(y < 1.0, -1.0, 0.0)
        if self.family == "exponential":
            z = y**self.power
            dz = self.power * y ** (self.power - 1.0)
            safe = jnp.where(jnp.abs(z) < 1.0e-4, 1.0, z)
            denominator = jnp.expm1(safe)
            regular = dz * (denominator - safe * jnp.exp(safe)) / denominator**2
            series = dz * (-0.5 + z / 6.0 - z**3 / 180.0)
            return jnp.where(jnp.abs(z) < 1.0e-4, series, regular)
        return -self.power * y ** (-self.power - 1.0)


class ThresholdIntegral(StrictModule):
    value: Array
    coarse_value: Array
    quadrature_error: Array
    tail_indicator: Array
    minimum_inverse_propagator: Array
    finite: Array
    admissible: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class ThresholdQuadraturePlan(StrictModule, NonTrainableState):
    """Fixed Gauss--Legendre radial rule for dimensionless FRG thresholds."""

    __hash__ = object.__hash__

    nodes: Array
    weights: Array
    coarse_nodes: Array
    coarse_weights: Array
    dimension: float = eqx.field(static=True)
    momentum_upper: float = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    maximum_nodes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: float,
        /,
        *,
        quadrature_order: int = 48,
        momentum_upper: float = 24.0,
        maximum_nodes: int = 512,
    ):
        dimension_ = float(dimension)
        order = int(quadrature_order)
        upper = float(momentum_upper)
        capacity = int(maximum_nodes)
        coarse_order = max(2, order // 2)
        if (
            not np.isfinite(dimension_)
            or dimension_ <= 1.0
            or order < 4
            or not np.isfinite(upper)
            or upper < 1.0
            or capacity <= 0
            or order + coarse_order > capacity
        ):
            raise ValueError("Threshold dimension, interval, or node budget is invalid.")
        fine_rule = gauss_legendre_data(order)
        coarse_rule = gauss_legendre_data(coarse_order)

        def map_rule(rule):
            return (
                0.5 * upper * (jnp.asarray(rule.nodes) + 1.0),
                0.5 * upper * jnp.asarray(rule.weights),
            )

        nodes, weights = map_rule(fine_rule)
        coarse_nodes, coarse_weights = map_rule(coarse_rule)
        self.nodes = nodes
        self.weights = weights
        self.coarse_nodes = coarse_nodes
        self.coarse_weights = coarse_weights
        self.dimension = dimension_
        self.momentum_upper = upper
        self.quadrature_order = order
        self.maximum_nodes = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-radial-threshold-quadrature",
                "dimension": dimension_,
                "momentum_upper": upper,
                "quadrature_order": order,
                "maximum_nodes": capacity,
                "nodes": array_tree_fingerprint(np.asarray(nodes)),
                "weights": array_tree_fingerprint(np.asarray(weights)),
            }
        )

    def _integral(
        self,
        regulator: Regulator,
        nodes: Array,
        weights: Array,
        mass_squared: Array,
        anomalous_dimension: Array,
    ) -> tuple[Array, Array]:
        y = nodes.reshape((nodes.size,) + (1,) * mass_squared.ndim)
        weight = weights.reshape((weights.size,) + (1,) * mass_squared.ndim)
        shape = regulator.shape(y)
        derivative = regulator.derivative(y)
        inverse = y + shape + mass_squared
        numerator = (2.0 - anomalous_dimension) * shape - 2.0 * y * derivative
        density = 0.5 * y ** (0.5 * self.dimension - 1.0) * numerator / inverse
        value = contract("q...,q...->...", weight, density)
        return value, jnp.min(inverse, axis=0)

    def evaluate(
        self,
        regulator: Regulator,
        mass_squared: ArrayLike,
        anomalous_dimension: ArrayLike = 0.0,
        /,
    ) -> ThresholdIntegral:
        if not isinstance(regulator, Regulator):
            raise TypeError("regulator must be a Regulator.")
        if (
            regulator.family == "power-law"
            and regulator.power <= 0.5 * self.dimension - 1.0
        ):
            raise ValueError(
                "Power-law threshold tail is not integrable in this dimension."
            )
        mass = jnp.asarray(mass_squared)
        eta = jnp.asarray(anomalous_dimension)
        mass, eta = jnp.broadcast_arrays(mass, eta)
        fine, minimum = self._integral(regulator, self.nodes, self.weights, mass, eta)
        coarse, coarse_minimum = self._integral(
            regulator, self.coarse_nodes, self.coarse_weights, mass, eta
        )
        if regulator.family == "optimized":
            prefactor = 2.0 / self.dimension * (1.0 - eta / (self.dimension + 2.0))
            exact = prefactor / (1.0 + mass)
            fine = exact
            coarse = exact
            minimum = 1.0 + mass
            coarse_minimum = minimum
            tail = jnp.zeros_like(fine)
        else:
            y = jnp.asarray(self.momentum_upper, dtype=mass.dtype)
            shape = regulator.shape(y)
            derivative = regulator.derivative(y)
            inverse = y + shape + mass
            edge = (
                0.5
                * y ** (0.5 * self.dimension - 1.0)
                * jnp.abs((2.0 - eta) * shape - 2.0 * y * derivative)
                / jnp.abs(inverse)
            )
            tail = y * edge
        quadrature_error = jnp.abs(fine - coarse)
        minimum = jnp.minimum(minimum, coarse_minimum)
        finite = (
            jnp.isfinite(fine)
            & jnp.isfinite(coarse)
            & jnp.isfinite(tail)
            & jnp.isfinite(minimum)
        )
        admissible = finite & (minimum > 0.0)
        status = jnp.where(
            admissible,
            int(FunctionalRGStatus.SUCCESS),
            jnp.where(
                finite,
                int(FunctionalRGStatus.POLE_ENCOUNTERED),
                int(FunctionalRGStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return ThresholdIntegral(
            fine,
            coarse,
            quadrature_error,
            tail,
            minimum,
            finite,
            admissible,
            status,
            self.plan_id,
        )


__all__ = [
    "FunctionalRGStatus",
    "Regulator",
    "RegulatorName",
    "ThresholdIntegral",
    "ThresholdQuadraturePlan",
]
