#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


WENOOrder: TypeAlias = Literal[3, 5]


def _nonlinear_weights(
    smoothness: tuple[Array, ...],
    optimal: tuple[float, ...],
    epsilon: float,
    power: int,
) -> tuple[Array, ...]:
    alpha = tuple(
        weight / (epsilon + beta) ** power
        for weight, beta in zip(optimal, smoothness, strict=True)
    )
    total = sum(alpha)
    return tuple(value / total for value in alpha)


def _weno3(values: Array, epsilon: float, power: int) -> tuple[Array, Array]:
    um1 = jnp.roll(values, 1, axis=0)
    up1 = jnp.roll(values, -1, axis=0)
    up2 = jnp.roll(values, -2, axis=0)
    left_candidates = (
        0.5 * (-um1 + 3.0 * values),
        0.5 * (values + up1),
    )
    left_smoothness = ((values - um1) ** 2, (up1 - values) ** 2)
    left_weights = _nonlinear_weights(
        left_smoothness,
        (1.0 / 3.0, 2.0 / 3.0),
        epsilon,
        power,
    )
    left = left_weights[0] * left_candidates[0] + left_weights[1] * left_candidates[1]
    right_candidates = (
        0.5 * (-up2 + 3.0 * up1),
        0.5 * (up1 + values),
    )
    right_smoothness = ((up1 - up2) ** 2, (values - up1) ** 2)
    right_weights = _nonlinear_weights(
        right_smoothness,
        (1.0 / 3.0, 2.0 / 3.0),
        epsilon,
        power,
    )
    right = (
        right_weights[0] * right_candidates[0] + right_weights[1] * right_candidates[1]
    )
    return left, right


def _weno5(values: Array, epsilon: float, power: int) -> tuple[Array, Array]:
    um2 = jnp.roll(values, 2, axis=0)
    um1 = jnp.roll(values, 1, axis=0)
    up1 = jnp.roll(values, -1, axis=0)
    up2 = jnp.roll(values, -2, axis=0)
    up3 = jnp.roll(values, -3, axis=0)
    left_candidates = (
        (2.0 * um2 - 7.0 * um1 + 11.0 * values) / 6.0,
        (-um1 + 5.0 * values + 2.0 * up1) / 6.0,
        (2.0 * values + 5.0 * up1 - up2) / 6.0,
    )
    left_smoothness = (
        13.0 / 12.0 * (um2 - 2.0 * um1 + values) ** 2
        + 0.25 * (um2 - 4.0 * um1 + 3.0 * values) ** 2,
        13.0 / 12.0 * (um1 - 2.0 * values + up1) ** 2 + 0.25 * (um1 - up1) ** 2,
        13.0 / 12.0 * (values - 2.0 * up1 + up2) ** 2
        + 0.25 * (3.0 * values - 4.0 * up1 + up2) ** 2,
    )
    left_weights = _nonlinear_weights(
        left_smoothness,
        (0.1, 0.6, 0.3),
        epsilon,
        power,
    )
    left = (
        left_weights[0] * left_candidates[0]
        + left_weights[1] * left_candidates[1]
        + left_weights[2] * left_candidates[2]
    )
    right_candidates = (
        (-up3 + 5.0 * up2 + 2.0 * up1) / 6.0,
        (2.0 * up2 + 5.0 * up1 - values) / 6.0,
        (11.0 * up1 - 7.0 * values + 2.0 * um1) / 6.0,
    )
    right_smoothness = (
        13.0 / 12.0 * (up3 - 2.0 * up2 + up1) ** 2
        + 0.25 * (up3 - 4.0 * up2 + 3.0 * up1) ** 2,
        13.0 / 12.0 * (up2 - 2.0 * up1 + values) ** 2 + 0.25 * (up2 - values) ** 2,
        13.0 / 12.0 * (up1 - 2.0 * values + um1) ** 2
        + 0.25 * (3.0 * up1 - 4.0 * values + um1) ** 2,
    )
    right_weights = _nonlinear_weights(
        right_smoothness,
        (0.1, 0.6, 0.3),
        epsilon,
        power,
    )
    right = (
        right_weights[0] * right_candidates[0]
        + right_weights[1] * right_candidates[1]
        + right_weights[2] * right_candidates[2]
    )
    return left, right


class WENOReconstructionPlan(StrictModule, NonTrainableState):
    """Periodic scalar WENO face reconstruction shared by FD and FV methods."""

    order: WENOOrder = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    power: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        order: WENOOrder = 5,
        /,
        *,
        epsilon: float = 1e-6,
        power: int = 2,
    ):
        if order not in (3, 5):
            raise ValueError("WENO order must be 3 or 5.")
        epsilon_ = float(epsilon)
        power_ = int(power)
        if not np.isfinite(epsilon_) or epsilon_ <= 0.0 or power_ <= 0:
            raise ValueError("WENO epsilon and power must be positive.")
        self.order = order
        self.epsilon = epsilon_
        self.power = power_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "weno-reconstruction",
                "order": order,
                "epsilon": epsilon_,
                "power": power_,
            }
        )

    @property
    def radius(self) -> int:
        return 1 if self.order == 3 else 3

    def reconstruct(self, values: ArrayLike, /) -> tuple[Array, Array]:
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[0] < (3 if self.order == 3 else 6):
            raise ValueError("WENO input leading axis is too short.")
        return (
            _weno3(array, self.epsilon, self.power)
            if self.order == 3
            else _weno5(array, self.epsilon, self.power)
        )


__all__ = ["WENOOrder", "WENOReconstructionPlan"]
