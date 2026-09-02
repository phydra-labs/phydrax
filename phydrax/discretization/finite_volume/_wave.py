#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


WaveLimiterKind: TypeAlias = Literal["minmod", "mc", "superbee", "van_leer"]


class WaveDecomposition(StrictModule):
    """Directional waves, speeds, and left/right fluctuations."""

    waves: Array
    speeds: Array
    left_fluctuation: Array
    right_fluctuation: Array

    def __init__(
        self,
        waves: Array,
        speeds: Array,
        left_fluctuation: Array,
        right_fluctuation: Array,
        /,
    ):
        waves_ = jnp.asarray(waves)
        speeds_ = jnp.asarray(speeds)
        left_ = jnp.asarray(left_fluctuation)
        right_ = jnp.asarray(right_fluctuation)
        if (
            waves_.shape[:-2] != speeds_.shape[:-1]
            or waves_.shape[-1] != speeds_.shape[-1]
        ):
            raise ValueError("Wave families and speeds must align.")
        if left_.shape != waves_.shape[:-1] or right_.shape != left_.shape:
            raise ValueError("Wave fluctuations must match the state batch shape.")
        self.waves = waves_
        self.speeds = speeds_
        self.left_fluctuation = left_
        self.right_fluctuation = right_


class AbstractWavePropagationPlan(StrictModule, NonTrainableState):
    """Normal interface decomposition into propagating fluctuations."""

    wave_plan_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    fwave: bool = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)

    @abc.abstractmethod
    def decompose(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
        *,
        auxiliary_left: Array | None = None,
        auxiliary_right: Array | None = None,
    ) -> WaveDecomposition:
        raise NotImplementedError


class RoeWavePropagationPlan(AbstractWavePropagationPlan):
    """Roe state-wave decomposition with upwind fluctuations."""

    entropy_fix: float = eqx.field(static=True)

    def __init__(self, *, entropy_fix: float = 0.0):
        fix = float(entropy_fix)
        if fix < 0.0:
            raise ValueError("entropy_fix must be non-negative.")
        self.entropy_fix = fix
        self.conservative = True
        self.fwave = False
        self.differentiability = "almost_everywhere"
        self.wave_plan_id = canonical_fingerprint(
            {"kind": "roe-wave-propagation", "entropy_fix": fix}
        )

    def decompose(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
        *,
        auxiliary_left: Array | None = None,
        auxiliary_right: Array | None = None,
    ) -> WaveDecomposition:
        del auxiliary_left, auxiliary_right
        left_matrix, right_matrix, speeds = system.eigensystem(
            left, right, int(axis), args
        )
        amplitudes = ein.contract("...ij,...j->...i", left_matrix, right - left)
        waves = right_matrix * amplitudes[..., None, :]
        negative = jnp.minimum(speeds, 0.0)
        positive = jnp.maximum(speeds, 0.0)
        if self.entropy_fix > 0.0:
            width = self.entropy_fix * jnp.maximum(
                jnp.max(jnp.abs(speeds), axis=-1, keepdims=True), 1.0
            )
            absolute = jnp.where(
                jnp.abs(speeds) < width,
                0.5 * (speeds**2 / width + width),
                jnp.abs(speeds),
            )
            negative = 0.5 * (speeds - absolute)
            positive = 0.5 * (speeds + absolute)
        left_fluctuation = jnp.sum(negative[..., None, :] * waves, axis=-1)
        right_fluctuation = jnp.sum(positive[..., None, :] * waves, axis=-1)
        return WaveDecomposition(waves, speeds, left_fluctuation, right_fluctuation)


class WaveFamilyLimiterPlan(StrictModule, NonTrainableState):
    """Upwind wave-family limiter for a line of interface decompositions."""

    kind: WaveLimiterKind = eqx.field(static=True)
    limiter_id: str = eqx.field(static=True)

    def __init__(self, kind: WaveLimiterKind = "mc", /):
        if kind not in ("minmod", "mc", "superbee", "van_leer"):
            raise ValueError("Unknown wave-family limiter.")
        self.kind = kind
        self.limiter_id = canonical_fingerprint(
            {"kind": "wave-family-limiter", "method": kind}
        )

    def _phi(self, ratio: Array, /) -> Array:
        if self.kind == "minmod":
            return jnp.maximum(0.0, jnp.minimum(1.0, ratio))
        if self.kind == "mc":
            return jnp.maximum(
                0.0,
                jnp.minimum(jnp.minimum(2.0 * ratio, 0.5 * (1.0 + ratio)), 2.0),
            )
        if self.kind == "superbee":
            return jnp.maximum(
                0.0,
                jnp.maximum(
                    jnp.minimum(2.0 * ratio, 1.0),
                    jnp.minimum(ratio, 2.0),
                ),
            )
        return jnp.where(ratio > 0.0, 2.0 * ratio / (1.0 + ratio), 0.0)

    def limit(self, decomposition: WaveDecomposition, axis: int, /) -> WaveDecomposition:
        waves = jnp.moveaxis(decomposition.waves, int(axis), 0)
        speeds = jnp.moveaxis(decomposition.speeds, int(axis), 0)
        previous = jnp.roll(waves, 1, axis=0)
        following = jnp.roll(waves, -1, axis=0)
        upstream = jnp.where((speeds >= 0.0)[..., None, :], previous, following)
        denominator = jnp.sum(waves**2, axis=-2)
        ratio = jnp.sum(upstream * waves, axis=-2) / jnp.where(
            denominator == 0.0, 1.0, denominator
        )
        limited = waves * self._phi(ratio)[..., None, :]
        limited = jnp.moveaxis(limited, 0, int(axis))
        negative = jnp.minimum(decomposition.speeds, 0.0)
        positive = jnp.maximum(decomposition.speeds, 0.0)
        left = jnp.sum(negative[..., None, :] * limited, axis=-1)
        right = jnp.sum(positive[..., None, :] * limited, axis=-1)
        return WaveDecomposition(limited, decomposition.speeds, left, right)


class TransverseWaveSolverPlan(StrictModule, NonTrainableState):
    """Split one normal fluctuation through a transverse eigensystem."""

    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.plan_id = canonical_fingerprint({"kind": "transverse-wave-solver"})

    def split(
        self,
        system: Any,
        left: Array,
        right: Array,
        fluctuation: Array,
        transverse_axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        left_matrix, right_matrix, speeds = system.eigensystem(
            left, right, int(transverse_axis), args
        )
        amplitudes = ein.contract("...ij,...j->...i", left_matrix, fluctuation)
        waves = right_matrix * amplitudes[..., None, :]
        negative = jnp.sum(jnp.minimum(speeds, 0.0)[..., None, :] * waves, axis=-1)
        positive = jnp.sum(jnp.maximum(speeds, 0.0)[..., None, :] * waves, axis=-1)
        return negative, positive


__all__ = [
    "AbstractWavePropagationPlan",
    "RoeWavePropagationPlan",
    "TransverseWaveSolverPlan",
    "WaveDecomposition",
    "WaveFamilyLimiterPlan",
    "WaveLimiterKind",
]
