#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.particle import ParticleBox
from ....discretization.vortex._capabilities import VortexDiffusionCapabilities
from ....discretization.vortex._interfaces import (
    AbstractPreparedVortexDiffusion,
    AbstractVortexDiffusionPlan,
    VortexDiffusionDiagnostics,
    VortexDiffusionEvaluation,
)
from ....discretization.vortex._source import VortexSourceState


class ParticleStrengthExchangeEvidence(StrictModule):
    smoothing_scale: Array
    support_radius: Array
    active_pair_count: Array
    total_rate_defect: Array
    cutoff_tail_bound: Array
    maximum_pair_flux: Array
    stable_step: Array
    periodic: bool = eqx.field(static=True)


class GaussianParticleStrengthExchangePlan(AbstractVortexDiffusionPlan):
    """Symmetric Gaussian particle-strength exchange for integrated vorticity."""

    smoothing_scale: float = eqx.field(static=True)
    cutoff_factor: float = eqx.field(static=True)
    maximum_interactions: int = eqx.field(static=True)
    box: ParticleBox | None
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    capabilities: VortexDiffusionCapabilities

    def __init__(
        self,
        dimension: int,
        smoothing_scale: float,
        /,
        *,
        cutoff_factor: float = 4.0,
        maximum_interactions: int = 1_000_000,
        box: ParticleBox | None = None,
    ):
        dimension_ = int(dimension)
        epsilon = float(smoothing_scale)
        cutoff = float(cutoff_factor)
        maximum = int(maximum_interactions)
        if dimension_ not in (2, 3):
            raise ValueError("PSE requires dimension 2 or 3.")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("PSE smoothing_scale must be finite and positive.")
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("PSE cutoff_factor must be finite and positive.")
        if maximum <= 0:
            raise ValueError("maximum_interactions must be positive.")
        if box is not None:
            if not isinstance(box, ParticleBox) or box.ambient_dimension != dimension_:
                raise ValueError("PSE ParticleBox dimension is incompatible.")
            widths = np.asarray(box.widths)
            periodic = np.asarray(box.periodic)
            if np.any(periodic & (cutoff * epsilon >= 0.5 * widths)):
                raise ValueError(
                    "Periodic PSE support must be less than half each period."
                )
        self.smoothing_scale = epsilon
        self.cutoff_factor = cutoff
        self.maximum_interactions = maximum
        self.box = box
        periodic_domain = box is not None and bool(np.any(np.asarray(box.periodic)))
        self.capabilities = VortexDiffusionCapabilities(
            dimension_,
            required_source_fields=(
                "positions",
                "strength",
                "active_mask",
                "volume",
            ),
            domain="periodic" if periodic_domain else "free-space",
            derivatives=(
                "source-position",
                "source-strength",
                "source-volume",
            ),
            acceleration="direct",
        )
        self.dimension = dimension_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-particle-strength-exchange",
                "dimension": dimension_,
                "smoothing_scale": epsilon,
                "cutoff_factor": cutoff,
                "maximum_interactions": maximum,
                "box": None if box is None else box.box_id,
                "capabilities": self.capabilities.capabilities_id,
            }
        )

    def prepare(
        self,
        /,
        *,
        capacity: int,
        dimension: int,
    ) -> PreparedGaussianParticleStrengthExchange:
        capacity_ = int(capacity)
        if int(dimension) != self.dimension or capacity_ <= 0:
            raise ValueError("PSE capacity/dimension is incompatible.")
        pairs = capacity_ * (capacity_ - 1) // 2
        if pairs > self.maximum_interactions:
            raise ValueError("PSE pair count exceeds maximum_interactions.")
        # Runtime activity belongs to VortexSourceState, not the plan.
        left, right = np.triu_indices(capacity_, k=1)
        return PreparedGaussianParticleStrengthExchange(
            self,
            jnp.asarray(left, dtype=jnp.int32),
            jnp.asarray(right, dtype=jnp.int32),
        )


class PreparedGaussianParticleStrengthExchange(AbstractPreparedVortexDiffusion):
    plan: GaussianParticleStrengthExchangePlan
    left: Array
    right: Array
    capabilities: VortexDiffusionCapabilities
    dimension: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: GaussianParticleStrengthExchangePlan, left: Array, right: Array, /
    ):
        capacity = int(
            max(int(jnp.max(left, initial=0)), int(jnp.max(right, initial=0))) + 1
        )
        self.plan = plan
        self.left = left
        self.right = right
        self.dimension = plan.dimension
        self.capacity = capacity
        self.pair_capacity = int(left.size)
        self.backend_id = plan.plan_id
        self.capabilities = plan.capabilities
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-gaussian-pse",
                "plan": plan.plan_id,
                "capacity": capacity,
                "pairs": self.pair_capacity,
            }
        )

    def evaluate(
        self,
        source: VortexSourceState,
        viscosity: ArrayLike,
        /,
    ) -> VortexDiffusionEvaluation:
        if not isinstance(source, VortexSourceState):
            raise TypeError("source must be VortexSourceState.")
        if source.capacity != self.capacity or source.dimension != self.dimension:
            raise ValueError("PSE source does not match prepared capacity/dimension.")
        if source.volume is None:
            raise ValueError("PSE requires source volume.")
        positions = source.safe_positions()
        strengths = source.safe_strength()
        volumes = source.safe_volume()
        viscosity_ = jnp.asarray(viscosity, dtype=positions.dtype)
        if viscosity_.shape != ():
            raise ValueError("PSE viscosity must be scalar.")
        active = source.active_mask
        inputs_finite = (
            jnp.all(jnp.where(active[:, None], jnp.isfinite(positions), True))
            & jnp.all(
                jnp.where(
                    active if self.dimension == 2 else active[:, None],
                    jnp.isfinite(strengths),
                    True,
                )
            )
            & jnp.all(jnp.where(active, jnp.isfinite(volumes) & (volumes > 0.0), True))
            & jnp.isfinite(viscosity_)
            & (viscosity_ >= 0.0)
        )
        safe_positions = jnp.where(active[:, None], positions, 0.0)
        safe_strengths = jnp.where(
            active if self.dimension == 2 else active[:, None], strengths, 0.0
        )
        safe_volumes = jnp.where(active, volumes, 1.0)
        displacement = safe_positions[self.left] - safe_positions[self.right]
        if self.plan.box is not None:
            displacement = self.plan.box.minimum_image(displacement)
        squared = jnp.sum(displacement * displacement, axis=-1)
        epsilon = jnp.asarray(self.plan.smoothing_scale, dtype=positions.dtype)
        scaled_squared = squared / epsilon**2
        pair_active = (
            active[self.left]
            & active[self.right]
            & (scaled_squared < self.plan.cutoff_factor**2)
        )
        normalization = 1.0 / (
            (jnp.pi ** (0.5 * self.dimension)) * epsilon**self.dimension
        )
        kernel = normalization * jnp.exp(-scaled_squared)
        omega_left = safe_strengths[self.left] / (
            safe_volumes[self.left]
            if self.dimension == 2
            else safe_volumes[self.left, None]
        )
        omega_right = safe_strengths[self.right] / (
            safe_volumes[self.right]
            if self.dimension == 2
            else safe_volumes[self.right, None]
        )
        prefactor = (
            4.0
            * viscosity_
            * safe_volumes[self.left]
            * safe_volumes[self.right]
            * kernel
            / epsilon**2
        )
        flux = (
            prefactor * (omega_right - omega_left)
            if self.dimension == 2
            else prefactor[:, None] * (omega_right - omega_left)
        )
        flux = jnp.where(
            pair_active if self.dimension == 2 else pair_active[:, None], flux, 0.0
        )
        rate = jnp.zeros_like(safe_strengths)
        rate = rate.at[self.left].add(flux)
        rate = rate.at[self.right].add(-flux)
        total_rate = jnp.sum(rate, axis=0)
        rate_scale = jnp.maximum(jnp.sum(jnp.abs(rate), axis=0), 1.0)
        defect = jnp.max(jnp.abs(total_rate) / rate_scale)
        outputs_finite = jnp.all(jnp.isfinite(rate))
        conservative = defect <= 128 * jnp.finfo(rate.dtype).eps
        successful = inputs_finite & outputs_finite & conservative
        maximum_flux = jnp.max(jnp.abs(flux), initial=0.0)
        stable_step = jnp.where(
            viscosity_ > 0.0,
            0.125 * epsilon**2 / jnp.maximum(viscosity_, jnp.finfo(rate.dtype).tiny),
            jnp.asarray(jnp.inf, dtype=rate.dtype),
        )
        backend = ParticleStrengthExchangeEvidence(
            epsilon,
            self.plan.cutoff_factor * epsilon,
            jnp.sum(pair_active, dtype=jnp.int32),
            defect,
            jnp.exp(-(self.plan.cutoff_factor**2)),
            maximum_flux,
            stable_step,
            self.plan.box is not None,
        )
        diagnostics = VortexDiffusionDiagnostics(
            jnp.asarray(self.capacity, dtype=jnp.int32),
            jnp.sum(pair_active, dtype=jnp.int32),
            total_rate,
            inputs_finite,
            outputs_finite,
            jnp.asarray(True),
            conservative,
            successful,
            backend,
        )
        return VortexDiffusionEvaluation(
            rate,
            successful,
            self.backend_id,
            canonical_fingerprint(
                {"kind": "gaussian-pse-evaluation", "prepared": self.prepared_id}
            ),
            diagnostics,
        )


__all__ = [
    "GaussianParticleStrengthExchangePlan",
    "ParticleStrengthExchangeEvidence",
    "PreparedGaussianParticleStrengthExchange",
]
