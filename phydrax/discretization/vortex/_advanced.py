#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ReformulatedVPMRate3D(StrictModule):
    strength_rate: Array
    core_radius_rate: Array
    dilation_rate: Array
    conservation_residual: Array
    finite: Array
    formulation_id: str = eqx.field(static=True)


class ReformulatedVPMPlan3D(StrictModule, NonTrainableState):
    """Explicit f/g family for coupled vector-strength and core evolution."""

    f: float = eqx.field(static=True)
    g: float = eqx.field(static=True)
    h_strength: float = eqx.field(static=True)
    h_core: float = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)

    def __init__(self, f: float = 0.0, g: float = 0.2, /):
        f_, g_ = float(f), float(g)
        denominator = 1.0 + 3.0 * f_
        if not math.isfinite(f_) or not math.isfinite(g_) or abs(denominator) <= 1.0e-14:
            raise ValueError("rVPM f/g parameters are invalid.")
        h_strength = (1.0 - 3.0 * g_) / denominator
        h_core = (f_ + g_) / denominator
        self.f = f_
        self.g = g_
        self.h_strength = h_strength
        self.h_core = h_core
        self.formulation_id = canonical_fingerprint(
            {
                "kind": "reformulated-vpm-3d",
                "f": f_,
                "g": g_,
                "h_strength": h_strength,
                "h_core": h_core,
            }
        )

    def rate(
        self,
        strength: ArrayLike,
        stretching: ArrayLike,
        core_radius: ArrayLike,
        /,
    ) -> ReformulatedVPMRate3D:
        gamma = jnp.asarray(strength)
        stretch = jnp.asarray(stretching, dtype=gamma.dtype)
        core = jnp.asarray(core_radius, dtype=gamma.dtype)
        if (
            gamma.ndim != 2
            or gamma.shape[1] != 3
            or stretch.shape != gamma.shape
            or core.shape != gamma.shape[:1]
        ):
            raise ValueError(
                "rVPM arrays require strength/stretching (N,3) and core (N,)."
            )
        norm_squared = jnp.sum(gamma * gamma, axis=-1)
        safe_norm = jnp.maximum(norm_squared, jnp.finfo(gamma.dtype).tiny)
        projected_stretch = jnp.sum(gamma * stretch, axis=-1) / safe_norm
        dilation = self.h_core * projected_stretch
        strength_rate = self.h_strength * stretch + (1.0 - self.h_strength) * (
            stretch - 3.0 * dilation[:, None] * gamma
        )
        core_rate = -dilation * core
        identity = self.h_strength + 3.0 * self.h_core
        residual = jnp.asarray(identity - 1.0, dtype=gamma.dtype)
        finite = jnp.all(jnp.isfinite(strength_rate)) & jnp.all(jnp.isfinite(core_rate))
        return ReformulatedVPMRate3D(
            strength_rate,
            core_rate,
            dilation,
            residual,
            finite,
            self.formulation_id,
        )


class VortexRelaxationResult3D(StrictModule):
    strength: Array
    alignment_before: Array
    alignment_after: Array
    magnitude_residual: Array
    finite: Array
    relaxation_id: str = eqx.field(static=True)


class PedrizzettiRelaxationPlan3D(StrictModule, NonTrainableState):
    fraction: float = eqx.field(static=True)
    preserve_magnitude: bool = eqx.field(static=True)
    relaxation_id: str = eqx.field(static=True)

    def __init__(self, fraction: float, /, *, preserve_magnitude: bool = False):
        fraction_ = float(fraction)
        if not math.isfinite(fraction_) or not 0.0 <= fraction_ <= 1.0:
            raise ValueError("Relaxation fraction must lie in [0, 1].")
        self.fraction = fraction_
        self.preserve_magnitude = bool(preserve_magnitude)
        self.relaxation_id = canonical_fingerprint(
            {
                "kind": "pedrizzetti-relaxation-3d",
                "fraction": fraction_,
                "preserve_magnitude": bool(preserve_magnitude),
            }
        )

    def apply(
        self, strength: ArrayLike, vorticity: ArrayLike, /
    ) -> VortexRelaxationResult3D:
        gamma = jnp.asarray(strength)
        omega = jnp.asarray(vorticity, dtype=gamma.dtype)
        if gamma.ndim != 2 or gamma.shape[1] != 3 or omega.shape != gamma.shape:
            raise ValueError("Relaxation requires matching (N,3) strength and vorticity.")
        tiny = jnp.finfo(gamma.dtype).tiny
        gamma_norm = jnp.linalg.norm(gamma, axis=-1)
        omega_norm = jnp.linalg.norm(omega, axis=-1)
        gamma_unit = gamma / jnp.maximum(gamma_norm, tiny)[:, None]
        omega_unit = omega / jnp.maximum(omega_norm, tiny)[:, None]
        blend = (1.0 - self.fraction) * gamma_unit + self.fraction * omega_unit
        blend_norm = jnp.linalg.norm(blend, axis=-1)
        if self.preserve_magnitude:
            relaxed = gamma_norm[:, None] * blend / jnp.maximum(blend_norm, tiny)[:, None]
        else:
            relaxed = (1.0 - self.fraction) * gamma + self.fraction * gamma_norm[
                :, None
            ] * omega_unit
        alignment_before = jnp.sum(gamma_unit * omega_unit, axis=-1)
        relaxed_unit = (
            relaxed / jnp.maximum(jnp.linalg.norm(relaxed, axis=-1), tiny)[:, None]
        )
        alignment_after = jnp.sum(relaxed_unit * omega_unit, axis=-1)
        magnitude_residual = (
            jnp.max(jnp.abs(jnp.linalg.norm(relaxed, axis=-1) - gamma_norm))
            if self.preserve_magnitude
            else jnp.asarray(0.0, dtype=gamma.dtype)
        )
        finite = jnp.all(jnp.isfinite(relaxed))
        return VortexRelaxationResult3D(
            relaxed,
            alignment_before,
            alignment_after,
            magnitude_residual,
            finite,
            self.relaxation_id,
        )


__all__ = [
    "PedrizzettiRelaxationPlan3D",
    "ReformulatedVPMPlan3D",
    "ReformulatedVPMRate3D",
    "VortexRelaxationResult3D",
]
