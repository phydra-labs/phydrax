#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule


class RobustLossEvaluation(StrictModule):
    rho: Array
    first: Array
    second: Array
    convex_model: Array


class AbstractRobustLoss(StrictModule):
    @property
    @abc.abstractmethod
    def loss_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, squared_norm: Any, /) -> RobustLossEvaluation:
        raise NotImplementedError


class IdentityLoss(AbstractRobustLoss):
    @property
    def loss_id(self) -> str:
        return "identity"

    def evaluate(self, squared_norm, /):
        value = jnp.asarray(squared_norm)
        return RobustLossEvaluation(
            value,
            jnp.ones_like(value),
            jnp.zeros_like(value),
            jnp.asarray(True),
        )


class HuberLoss(AbstractRobustLoss):
    delta: float = eqx.field(static=True)

    def __init__(self, delta: float = 1.0, /):
        value = float(delta)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Huber delta must be finite and positive.")
        self.delta = value

    @property
    def loss_id(self) -> str:
        return f"huber/{self.delta}"

    def evaluate(self, squared_norm, /):
        value = jnp.asarray(squared_norm)
        threshold = self.delta * self.delta
        root = jnp.sqrt(jnp.maximum(value, 1e-30))
        inside = value <= threshold
        rho = jnp.where(inside, value, 2.0 * self.delta * root - threshold)
        first = jnp.where(inside, 1.0, self.delta / root)
        second = jnp.where(inside, 0.0, -0.5 * self.delta / (root**3))
        return RobustLossEvaluation(rho, first, second, jnp.asarray(True))


class SoftL1Loss(AbstractRobustLoss):
    scale: float = eqx.field(static=True)

    def __init__(self, scale: float = 1.0, /):
        value = float(scale)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Soft-L1 scale must be finite and positive.")
        self.scale = value

    @property
    def loss_id(self) -> str:
        return f"soft-l1/{self.scale}"

    def evaluate(self, squared_norm, /):
        value = jnp.asarray(squared_norm)
        scaled = value / (self.scale * self.scale)
        root = jnp.sqrt(1.0 + scaled)
        rho = 2.0 * self.scale * self.scale * (root - 1.0)
        first = 1.0 / root
        second = -0.5 / (self.scale * self.scale * root**3)
        return RobustLossEvaluation(rho, first, second, jnp.asarray(True))


class CauchyLoss(AbstractRobustLoss):
    scale: float = eqx.field(static=True)

    def __init__(self, scale: float = 1.0, /):
        value = float(scale)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Cauchy scale must be finite and positive.")
        self.scale = value

    @property
    def loss_id(self) -> str:
        return f"cauchy/{self.scale}"

    def evaluate(self, squared_norm, /):
        value = jnp.asarray(squared_norm)
        scale_squared = self.scale * self.scale
        denominator = 1.0 + value / scale_squared
        rho = scale_squared * jnp.log(denominator)
        first = 1.0 / denominator
        second = -1.0 / (scale_squared * denominator**2)
        return RobustLossEvaluation(rho, first, second, second >= 0.0)


class ArctanLoss(AbstractRobustLoss):
    scale: float = eqx.field(static=True)

    def __init__(self, scale: float = 1.0, /):
        value = float(scale)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Arctan scale must be finite and positive.")
        self.scale = value

    @property
    def loss_id(self) -> str:
        return f"arctan/{self.scale}"

    def evaluate(self, squared_norm, /):
        value = jnp.asarray(squared_norm)
        scale_squared = self.scale * self.scale
        scaled = value / scale_squared
        denominator = 1.0 + scaled * scaled
        rho = scale_squared * jnp.arctan(scaled)
        first = 1.0 / denominator
        second = -2.0 * scaled / (scale_squared * denominator**2)
        return RobustLossEvaluation(rho, first, second, second >= 0.0)


class TukeyLoss(AbstractRobustLoss):
    scale: float = eqx.field(static=True)

    def __init__(self, scale: float = 1.0, /):
        value = float(scale)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Tukey scale must be finite and positive.")
        self.scale = value

    @property
    def loss_id(self) -> str:
        return f"tukey/{self.scale}"

    def evaluate(self, squared_norm, /):
        value = jnp.asarray(squared_norm)
        scale_squared = self.scale * self.scale
        scaled = value / scale_squared
        inside = scaled <= 1.0
        remainder = 1.0 - scaled
        rho = jnp.where(
            inside,
            scale_squared / 3.0 * (1.0 - remainder**3),
            scale_squared / 3.0,
        )
        first = jnp.where(inside, remainder**2, 0.0)
        second = jnp.where(inside, -2.0 * remainder / scale_squared, 0.0)
        return RobustLossEvaluation(rho, first, second, second >= 0.0)


class ScaledLoss(AbstractRobustLoss):
    loss: AbstractRobustLoss
    scale: float = eqx.field(static=True)

    def __init__(self, loss: AbstractRobustLoss, scale: float, /):
        if not isinstance(loss, AbstractRobustLoss):
            raise TypeError("loss must be AbstractRobustLoss.")
        value = float(scale)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Loss scale must be finite and positive.")
        self.loss = loss
        self.scale = value

    @property
    def loss_id(self) -> str:
        return f"scaled/{self.scale}/{self.loss.loss_id}"

    def evaluate(self, squared_norm, /):
        result = self.loss.evaluate(squared_norm)
        return RobustLossEvaluation(
            self.scale * result.rho,
            self.scale * result.first,
            self.scale * result.second,
            result.convex_model,
        )


def squared_tree_norm(value: PyTree[Any], /) -> Array:
    return sum(jnp.real(jnp.vdot(leaf, leaf)) for leaf in jax.tree.leaves(value))


def robustify_residual(
    residual: PyTree[Any],
    loss: AbstractRobustLoss,
    /,
) -> tuple[PyTree[Array], RobustLossEvaluation]:
    if not isinstance(loss, AbstractRobustLoss):
        raise TypeError("loss must be AbstractRobustLoss.")
    squared = squared_tree_norm(residual)
    evaluation = loss.evaluate(squared)
    factor = jnp.sqrt(jnp.maximum(evaluation.rho, 0.0) / jnp.maximum(squared, 1e-30))
    transformed = jax.tree.map(lambda value: factor * value, residual)
    return transformed, evaluation


def robust_normal_weight(
    residual: PyTree[Any],
    loss: AbstractRobustLoss,
    /,
) -> tuple[Array, Array]:
    evaluation = loss.evaluate(squared_tree_norm(residual))
    first = jnp.maximum(evaluation.first, 0.0)
    curvature = evaluation.first + 2.0 * evaluation.second * squared_tree_norm(residual)
    return jnp.sqrt(first), jnp.maximum(curvature, 0.0)


__all__ = [
    "AbstractRobustLoss",
    "ArctanLoss",
    "CauchyLoss",
    "HuberLoss",
    "IdentityLoss",
    "RobustLossEvaluation",
    "ScaledLoss",
    "SoftL1Loss",
    "TukeyLoss",
    "robust_normal_weight",
    "robustify_residual",
    "squared_tree_norm",
]
