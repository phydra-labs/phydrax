#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...stochastic import WienerRealization


PhaseFieldNoiseKind = Literal["allen-cahn", "cahn-hilliard"]


class PhaseFieldNoiseEvidence(StrictModule):
    increment: Array
    weighted_mass: Array
    finite: Array
    conservative: Array
    successful: Array
    noise_plan_id: str = eqx.field(static=True)


class PhaseFieldNoisePlan(StrictModule, NonTrainableState):
    """One replayable finite-rank spatial Wiener forcing."""

    realization: WienerRealization
    basis: Array
    amplitude: Array
    conservation_weights: Array | None
    kind: PhaseFieldNoiseKind = eqx.field(static=True)
    noise_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PhaseFieldNoiseKind,
        realization: WienerRealization,
        basis: ArrayLike,
        /,
        *,
        amplitude: ArrayLike = 1.0,
        conservation_weights: ArrayLike | None = None,
    ):
        if kind not in ("allen-cahn", "cahn-hilliard"):
            raise ValueError("Unknown phase-field noise kind.")
        if not isinstance(realization, WienerRealization):
            raise TypeError("realization must be WienerRealization.")
        modes = np.asarray(basis)
        scale = np.asarray(amplitude)
        if (
            modes.ndim != 2
            or modes.shape[1:] != realization.noise_shape
            or np.any(~np.isfinite(modes))
            or scale.shape != ()
            or not np.isfinite(scale)
            or scale < 0.0
        ):
            raise ValueError("Phase-field stochastic basis or amplitude is invalid.")
        weights = (
            None if conservation_weights is None else np.asarray(conservation_weights)
        )
        if weights is not None and (
            weights.shape != (modes.shape[0],)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("Noise conservation weights must be positive per DOF.")
        if kind == "cahn-hilliard" and weights is None:
            raise ValueError(
                "Cahn-Hilliard stochastic forcing requires conservation weights."
            )
        self.kind = kind
        self.realization = realization
        self.basis = jnp.asarray(modes)
        self.amplitude = jnp.asarray(scale)
        self.conservation_weights = None if weights is None else jnp.asarray(weights)
        self.noise_plan_id = canonical_fingerprint(
            {
                "kind": "phase-field-noise-plan",
                "noise_kind": kind,
                "realization": realization.realization_id,
                "basis": array_tree_fingerprint(modes),
                "amplitude": float(scale),
                "weights": None if weights is None else array_tree_fingerprint(weights),
            }
        )

    @classmethod
    def thermal(
        cls,
        kind: PhaseFieldNoiseKind,
        realization: WienerRealization,
        basis: ArrayLike,
        /,
        *,
        temperature: float,
        boltzmann_constant: float,
        mobility_scale: float = 1.0,
        conservation_weights: ArrayLike | None = None,
    ) -> PhaseFieldNoisePlan:
        values = (float(temperature), float(boltzmann_constant), float(mobility_scale))
        if (
            any(not np.isfinite(value) or value < 0.0 for value in values)
            or values[1] <= 0.0
        ):
            raise ValueError("Thermal phase-field noise parameters are invalid.")
        amplitude = np.sqrt(2.0 * values[0] * values[1] * values[2])
        return cls(
            kind,
            realization,
            basis,
            amplitude=amplitude,
            conservation_weights=conservation_weights,
        )

    def transfer(
        self,
        prolongation: ArrayLike,
        /,
        *,
        conservation_weights: ArrayLike | None = None,
    ) -> PhaseFieldNoisePlan:
        """Transfer the same global Wiener modes to a successor FE epoch."""

        operator = jnp.asarray(prolongation)
        if operator.ndim != 2 or operator.shape[1] != self.basis.shape[0]:
            raise ValueError(
                "Noise-basis transfer must map source DOFs to successor DOFs."
            )
        basis = operator @ self.basis
        weights = conservation_weights if self.kind == "cahn-hilliard" else None
        if self.kind == "cahn-hilliard" and weights is None:
            raise ValueError(
                "Conservative stochastic transfer requires successor integration weights."
            )
        return PhaseFieldNoisePlan(
            self.kind,
            self.realization,
            basis,
            amplitude=self.amplitude,
            conservation_weights=weights,
        )

    def increment(
        self,
        start: ArrayLike,
        end: ArrayLike,
        /,
        *,
        dtype=jnp.float64,
    ) -> PhaseFieldNoiseEvidence:
        lower = jnp.asarray(start)
        upper = jnp.asarray(end, dtype=lower.dtype)
        if lower.shape != () or upper.shape != ():
            raise ValueError("Phase-field Wiener interval bounds must be scalar.")
        coefficients = self.realization.increments(lower, upper, dtype=dtype)
        increment = self.amplitude.astype(dtype) * ein.contract(
            "dr,r->d", self.basis.astype(dtype), coefficients
        )
        if self.conservation_weights is None:
            weighted_mass = jnp.asarray(0.0, dtype=increment.dtype)
            conservative = jnp.asarray(True)
        else:
            weights = self.conservation_weights.astype(increment.dtype)
            total_weight = jnp.sum(weights)
            weighted_mean = jnp.sum(weights * increment) / total_weight
            increment = increment - weighted_mean
            weighted_mass = jnp.sum(weights * increment)
            tolerance = (
                128.0
                * jnp.finfo(increment.dtype).eps
                * jnp.maximum(jnp.sum(jnp.abs(weights * increment)), 1.0)
            )
            conservative = jnp.abs(weighted_mass) <= tolerance
        finite = jnp.all(jnp.isfinite(increment))
        return PhaseFieldNoiseEvidence(
            increment,
            weighted_mass,
            finite,
            conservative,
            finite & conservative,
            self.noise_plan_id,
        )


__all__ = [
    "PhaseFieldNoiseEvidence",
    "PhaseFieldNoiseKind",
    "PhaseFieldNoisePlan",
]
