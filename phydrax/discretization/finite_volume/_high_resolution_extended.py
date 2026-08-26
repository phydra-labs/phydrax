#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FilterADPolicy: TypeAlias = Literal["exact", "frozen", "smooth", "forbid"]


def _lagrange_coefficients(offsets: tuple[int, ...], point: float, /) -> np.ndarray:
    coefficients = np.ones(len(offsets))
    for i, offset in enumerate(offsets):
        for j, other in enumerate(offsets):
            if i != j:
                coefficients[i] *= (point - other) / (offset - other)
    return coefficients


class TENOQualification(StrictModule, NonTrainableState):
    order: int = eqx.field(static=True)
    stencil_width: int = eqx.field(static=True)
    constant_residual: float = eqx.field(static=True)
    polynomial_residual: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class _HighOrderTENOPlan(StrictModule, NonTrainableState):
    """Sixth/eighth-order smooth-exact targeted reconstruction."""

    order: int = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    full_coefficients: Array
    candidate_offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    candidate_coefficients: tuple[Array, ...]
    cutoff: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    qualification: TENOQualification
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        order: Literal[6, 8],
        /,
        *,
        cutoff: float = 1e-6,
        epsilon: float = 1e-14,
    ):
        if order == 6:
            offsets = (-2, -1, 0, 1, 2, 3)
            candidates = ((-2, -1, 0), (-1, 0, 1), (0, 1, 2), (0, 1, 2, 3))
        elif order == 8:
            offsets = (-3, -2, -1, 0, 1, 2, 3, 4)
            candidates = (
                (-3, -2, -1),
                (-2, -1, 0),
                (-1, 0, 1),
                (0, 1, 2, 3),
                (-1, 0, 1, 2),
                (0, 1, 2, 3, 4),
            )
        else:
            raise ValueError("High-order TENO supports orders six and eight.")
        cutoff_ = float(cutoff)
        epsilon_ = float(epsilon)
        if not np.isfinite(cutoff_) or cutoff_ <= 0.0 or cutoff_ >= 1.0:
            raise ValueError("TENO cutoff must lie in (0, 1).")
        if not np.isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("TENO epsilon must be positive.")
        full = _lagrange_coefficients(offsets, 0.5)
        candidate_coefficients = tuple(
            jnp.asarray(_lagrange_coefficients(value, 0.5)) for value in candidates
        )
        constant_residual = abs(np.sum(full) - 1.0)
        polynomial_residual = 0.0
        for degree in range(order):
            exact = 0.5**degree
            reconstructed = sum(
                coefficient * offset**degree
                for coefficient, offset in zip(full, offsets, strict=True)
            )
            polynomial_residual = max(polynomial_residual, abs(reconstructed - exact))
        passed = constant_residual <= 1e-12 and polynomial_residual <= 1e-10
        report_id = canonical_fingerprint(
            {
                "kind": "high-order-teno-qualification",
                "order": order,
                "constant_residual": constant_residual,
                "polynomial_residual": polynomial_residual,
            }
        )
        self.order = order
        self.offsets = offsets
        self.full_coefficients = jnp.asarray(full)
        self.candidate_offsets = candidates
        self.candidate_coefficients = candidate_coefficients
        self.cutoff = cutoff_
        self.epsilon = epsilon_
        self.qualification = TENOQualification(
            order,
            len(offsets),
            constant_residual,
            polynomial_residual,
            passed,
            report_id,
        )
        self.plan_id = canonical_fingerprint(
            {"kind": "high-order-teno", "qualification": report_id, "cutoff": cutoff_}
        )

    @property
    def radius(self) -> int:
        return max(abs(value) for value in self.offsets)

    @staticmethod
    def _smoothness(window: Array, /) -> Array:
        first = jnp.diff(window, axis=1)
        second = jnp.diff(first, axis=1)
        return jnp.sum(first**2, axis=1) + jnp.sum(second**2, axis=1)

    def _side(self, values: Array, sign: int, /) -> Array:
        count = values.shape[0]
        full_offsets = (
            self.offsets if sign > 0 else tuple(-value + 1 for value in self.offsets)
        )
        full_indices = jnp.clip(
            jnp.arange(count)[:, None] + jnp.asarray(full_offsets)[None, :],
            0,
            count - 1,
        )
        full_window = values[full_indices]
        full_value = jnp.sum(
            self.full_coefficients.reshape((1, -1) + (1,) * (values.ndim - 1))
            * full_window,
            axis=1,
        )
        candidate_values = []
        smoothness = []
        for offsets, coefficients in zip(
            self.candidate_offsets, self.candidate_coefficients, strict=True
        ):
            oriented = offsets if sign > 0 else tuple(-value + 1 for value in offsets)
            indices = jnp.clip(
                jnp.arange(count)[:, None] + jnp.asarray(oriented)[None, :],
                0,
                count - 1,
            )
            window = values[indices]
            candidate_values.append(
                jnp.sum(
                    coefficients.reshape((1, -1) + (1,) * (values.ndim - 1)) * window,
                    axis=1,
                )
            )
            smoothness.append(self._smoothness(window))
        beta = jnp.stack(tuple(smoothness), axis=1)
        tau = jnp.max(beta, axis=1, keepdims=True) - jnp.min(beta, axis=1, keepdims=True)
        gamma = (1.0 + tau / (beta + self.epsilon)) ** 6
        normalized = gamma / jnp.sum(gamma, axis=1, keepdims=True)
        active = normalized >= self.cutoff
        all_active = jnp.all(active, axis=1)
        weights = active / jnp.maximum(jnp.sum(active, axis=1, keepdims=True), 1)
        candidates = jnp.stack(tuple(candidate_values), axis=1)
        targeted = jnp.sum(weights * candidates, axis=1)
        return jnp.where(all_active, full_value, targeted)

    def reconstruct(self, values: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(values)
        if value.ndim < 1 or value.shape[0] < len(self.offsets):
            raise ValueError("High-order TENO input is shorter than its stencil.")
        return self._side(value, 1), self._side(value, -1)


class ExplicitStabilizationPlan(StrictModule, NonTrainableState):
    """Separate conservative nearest-neighbor filter with declared AD semantics."""

    strength: float = eqx.field(static=True)
    ad_policy: FilterADPolicy = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        strength: float,
        /,
        *,
        ad_policy: FilterADPolicy = "frozen",
        periodic: bool = False,
    ):
        strength_ = float(strength)
        if not np.isfinite(strength_) or strength_ < 0.0 or strength_ > 0.5:
            raise ValueError("Filter strength must lie in [0, 0.5].")
        if ad_policy not in ("exact", "frozen", "smooth", "forbid"):
            raise ValueError("Unknown filter AD policy.")
        self.strength = strength_
        self.ad_policy = ad_policy
        self.periodic = bool(periodic)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "explicit-stabilization",
                "strength": strength_,
                "ad_policy": ad_policy,
                "periodic": bool(periodic),
            }
        )

    def apply(self, values: ArrayLike, sensor: ArrayLike | None = None, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim < 1 or value.shape[0] < 3:
            raise ValueError("Explicit filter requires at least three values.")
        if self.periodic:
            left = jnp.roll(value, 1, axis=0)
            right = jnp.roll(value, -1, axis=0)
        else:
            left = jnp.concatenate((value[:1], value[:-1]), axis=0)
            right = jnp.concatenate((value[1:], value[-1:]), axis=0)
        curvature = value - 0.5 * (left + right)
        sensor_ = jnp.ones(value.shape[:1]) if sensor is None else jnp.asarray(sensor)
        if sensor_.shape != value.shape[:1]:
            raise ValueError("Filter sensor must have one value per leading site.")
        if self.ad_policy == "forbid" and isinstance(value, jax.core.Tracer):
            raise ValueError("Filter AD policy forbids differentiation.")
        if self.ad_policy == "frozen":
            sensor_ = jax.lax.stop_gradient(sensor_)
        elif self.ad_policy == "smooth":
            sensor_ = jax.nn.sigmoid(sensor_)
        update = (
            self.strength
            * sensor_.reshape(sensor_.shape + (1,) * (value.ndim - 1))
            * curvature
        )
        filtered = value - update
        correction = jnp.mean(value - filtered, axis=0, keepdims=True)
        return filtered + correction


__all__ = [
    "ExplicitStabilizationPlan",
    "FilterADPolicy",
    "TENOQualification",
]
