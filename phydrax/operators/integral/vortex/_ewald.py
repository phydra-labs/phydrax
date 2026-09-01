#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.vortex._capabilities import VortexVelocityCapabilities
from ....discretization.vortex._compatibility import (
    request_fields,
    validate_vortex_velocity_evaluation,
    VortexVelocityCompatibility,
)
from ....discretization.vortex._interfaces import (
    AbstractPreparedVortexVelocity,
    AbstractVortexVelocityPlan,
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
from ....discretization.vortex._precision import VortexPrecisionPolicy
from ....discretization.vortex._source import VortexSourceState, VortexTargetState


class PeriodicVortexEwaldDiagnostics(StrictModule):
    total_strength: Array
    compatibility_residual: Array
    real_image_count: Array
    reciprocal_mode_count: Array
    real_tail_bound: Array
    reciprocal_tail_bound: Array
    splitting_parameter: Array
    compatible: Array


class PeriodicVortexEwaldPlan(AbstractVortexVelocityPlan):
    """Screened real/reciprocal periodic vortex authority in two or three dimensions."""

    periods: Array
    splitting_parameter: float = eqx.field(static=True)
    real_image_radius: int = eqx.field(static=True)
    reciprocal_mode_radius: int = eqx.field(static=True)
    compatibility_tolerance: float = eqx.field(static=True)
    precision: VortexPrecisionPolicy
    real_shifts: Array
    reciprocal_vectors: Array
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    capabilities: VortexVelocityCapabilities

    def __init__(
        self,
        periods: ArrayLike,
        /,
        *,
        splitting_parameter: float,
        real_image_radius: int,
        reciprocal_mode_radius: int,
        compatibility_tolerance: float = 1.0e-12,
        precision: VortexPrecisionPolicy | None = None,
    ):
        period_host = np.asarray(periods, dtype=float)
        if (
            period_host.ndim != 1
            or period_host.size not in (2, 3)
            or np.any(~np.isfinite(period_host))
            or np.any(period_host <= 0.0)
        ):
            raise ValueError("Ewald periods require a finite positive 2- or 3-vector.")
        dimension = int(period_host.size)
        alpha = float(splitting_parameter)
        real_radius = int(real_image_radius)
        mode_radius = int(reciprocal_mode_radius)
        tolerance = float(compatibility_tolerance)
        if (
            not math.isfinite(alpha)
            or alpha <= 0.0
            or real_radius < 0
            or mode_radius <= 0
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Ewald splitting, truncation, or tolerance is invalid.")
        integer_shifts = np.asarray(
            tuple(
                itertools.product(range(-real_radius, real_radius + 1), repeat=dimension)
            ),
            dtype=float,
        )
        mode_tuples = tuple(
            mode
            for mode in itertools.product(
                range(-mode_radius, mode_radius + 1), repeat=dimension
            )
            if any(value != 0 for value in mode)
        )
        reciprocal = 2.0 * np.pi * np.asarray(mode_tuples, dtype=float) / period_host
        precision_ = VortexPrecisionPolicy() if precision is None else precision
        self.periods = jnp.asarray(period_host)
        self.splitting_parameter = alpha
        self.real_image_radius = real_radius
        self.reciprocal_mode_radius = mode_radius
        self.compatibility_tolerance = tolerance
        self.precision = precision_
        self.real_shifts = jnp.asarray(integer_shifts * period_host)
        self.reciprocal_vectors = jnp.asarray(reciprocal)
        self.dimension = dimension
        self.capabilities = VortexVelocityCapabilities(
            dimension,
            required_source_fields=("positions", "strength", "active_mask"),
            supported_fields=("velocity", "velocity_gradient", "vorticity"),
            domain="periodic",
            precision=precision_,
            derivatives=("source-position", "source-strength", "target-position"),
            target_topologies=("same-support", "arbitrary-targets"),
            acceleration="ewald",
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-vortex-ewald-plan",
                "periods": period_host.tolist(),
                "splitting_parameter": alpha,
                "real_image_radius": real_radius,
                "reciprocal_mode_radius": mode_radius,
                "compatibility_tolerance": tolerance,
                "precision": precision_.policy_id,
            }
        )

    def prepare(
        self,
        /,
        *,
        source_capacity: int,
        target_capacity: int | None = None,
        source_kind: str = "particle",
        target_topology: str = "same-support",
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> "PreparedPeriodicVortexEwald":
        targets = (
            int(source_capacity) if target_capacity is None else int(target_capacity)
        )
        compatibility = VortexVelocityCompatibility(
            self.capabilities,
            source_capacity=int(source_capacity),
            target_capacity=targets,
            source_kind=source_kind,
            target_topology=target_topology,
            requested_fields=request_fields(request),
        )
        return PreparedPeriodicVortexEwald(self, compatibility)


class PreparedPeriodicVortexEwald(AbstractPreparedVortexVelocity):
    plan: PeriodicVortexEwaldPlan
    compatibility: VortexVelocityCompatibility
    dimension: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: VortexVelocityCapabilities

    def __init__(
        self, plan: PeriodicVortexEwaldPlan, compatibility: VortexVelocityCompatibility, /
    ):
        self.plan = plan
        self.compatibility = compatibility
        self.dimension = plan.dimension
        self.source_capacity = compatibility.source_capacity
        self.target_capacity = compatibility.target_capacity
        self.backend_id = plan.plan_id
        self.capabilities = plan.capabilities
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-vortex-ewald",
                "plan": plan.plan_id,
                "compatibility": compatibility.compatibility_id,
            }
        )

    def _velocity(
        self, source: VortexSourceState, targets: Array, target_identity: Array | None, /
    ) -> Array:
        positions = source.safe_positions()
        strength = source.safe_strength()
        active = source.active_mask
        displacement = (
            targets[:, None, None, :]
            - positions[None, :, None, :]
            - self.plan.real_shifts[None, None, :, :]
        )
        squared = jnp.sum(displacement * displacement, axis=-1)
        zero_shift = jnp.all(self.plan.real_shifts == 0.0, axis=-1)
        self_pair = jnp.zeros(squared.shape, dtype=bool)
        if target_identity is not None:
            source_index = jnp.arange(source.capacity, dtype=jnp.int32)
            self_pair = (
                target_identity[:, None, None] == source_index[None, :, None]
            ) & zero_shift[None, None, :]
        coincident = (squared == 0.0) & ~self_pair & active[None, :, None]
        safe_squared = jnp.where(self_pair | coincident, 1.0, squared)
        alpha = jnp.asarray(self.plan.splitting_parameter, dtype=targets.dtype)
        if self.dimension == 2:
            screened = jnp.exp(-(alpha**2) * safe_squared) / safe_squared
            perpendicular = jnp.stack(
                (-displacement[..., 1], displacement[..., 0]), axis=-1
            )
            real_pair = (
                strength[None, :, None, None]
                * screened[..., None]
                * perpendicular
                / (2.0 * jnp.pi)
            )
        else:
            radius = jnp.sqrt(safe_squared)
            screened = (
                jax.scipy.special.erfc(alpha * radius)
                + 2.0
                * alpha
                * radius
                / jnp.sqrt(jnp.pi)
                * jnp.exp(-(alpha**2) * safe_squared)
            ) / (safe_squared * radius)
            cross = jnp.cross(strength[None, :, None, :], displacement)
            real_pair = screened[..., None] * cross / (4.0 * jnp.pi)
        real_pair = jnp.where(
            (active[None, :, None] & ~self_pair & ~coincident)[..., None], real_pair, 0.0
        )
        real_velocity = jnp.sum(real_pair, axis=(1, 2))
        base_displacement = targets[:, None, :] - positions[None, :, :]
        wave = self.plan.reciprocal_vectors.astype(targets.dtype)
        squared_wave = jnp.sum(wave * wave, axis=-1)
        phase = jnp.exp(
            1j
            * jnp.sum(base_displacement[:, :, None, :] * wave[None, None, :, :], axis=-1)
        )
        filter_ = jnp.exp(-squared_wave / (4.0 * alpha**2)) / squared_wave
        volume = jnp.prod(self.plan.periods.astype(targets.dtype))
        if self.dimension == 2:
            direction = 1j * jnp.stack((wave[:, 1], -wave[:, 0]), axis=-1)
            reciprocal_pair = (
                strength[None, :, None, None]
                * phase[..., None]
                * filter_[None, None, :, None]
                * direction[None, None, :, :]
                / volume
            )
        else:
            direction = 1j * jnp.cross(wave[None, None, :, :], strength[None, :, None, :])
            reciprocal_pair = (
                phase[..., None] * filter_[None, None, :, None] * direction / volume
            )
        reciprocal_pair = jnp.where(active[None, :, None, None], reciprocal_pair, 0.0)
        reciprocal_velocity = jnp.real(jnp.sum(reciprocal_pair, axis=(1, 2)))
        return real_velocity + reciprocal_velocity

    def evaluate(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
        /,
        *,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        source, target = validate_vortex_velocity_evaluation(
            self.capabilities, self.compatibility, source, target, request
        )
        total = jnp.sum(source.safe_strength(), axis=0)
        scale = jnp.maximum(jnp.sum(jnp.abs(source.safe_strength()), axis=0), 1.0)
        residual = jnp.max(jnp.abs(total) / scale)
        compatible = residual <= self.plan.compatibility_tolerance
        positions = eqx.error_if(
            target.positions,
            ~compatible,
            "Periodic Ewald evaluation requires zero total integrated vorticity.",
        )
        velocity_all = self._velocity(source, positions, target.source_indices)
        gradient_all = None
        if request.velocity_gradient or request.vorticity:
            gradient_all = jax.vmap(
                jax.jacfwd(lambda point: self._velocity(source, point[None, :], None)[0])
            )(positions)
        if request.vorticity:
            if self.dimension == 2:
                vorticity_all = gradient_all[:, 1, 0] - gradient_all[:, 0, 1]
            else:
                vorticity_all = jnp.stack(
                    (
                        gradient_all[:, 2, 1] - gradient_all[:, 1, 2],
                        gradient_all[:, 0, 2] - gradient_all[:, 2, 0],
                        gradient_all[:, 1, 0] - gradient_all[:, 0, 1],
                    ),
                    axis=-1,
                )
        else:
            vorticity_all = None
        velocity = velocity_all if request.velocity else None
        gradient = gradient_all if request.velocity_gradient else None
        real_extent = self.plan.real_image_radius * jnp.min(self.plan.periods)
        wave_extent = jnp.max(jnp.linalg.norm(self.plan.reciprocal_vectors, axis=-1))
        real_tail = jnp.exp(-((self.plan.splitting_parameter * real_extent) ** 2))
        reciprocal_tail = jnp.exp(
            -(wave_extent**2) / (4.0 * self.plan.splitting_parameter**2)
        )
        finite = jnp.all(jnp.isfinite(velocity_all))
        if gradient_all is not None:
            finite = finite & jnp.all(jnp.isfinite(gradient_all))
        successful = compatible & finite
        backend = PeriodicVortexEwaldDiagnostics(
            total,
            residual,
            jnp.asarray(self.plan.real_shifts.shape[0], dtype=jnp.int32),
            jnp.asarray(self.plan.reciprocal_vectors.shape[0], dtype=jnp.int32),
            real_tail,
            reciprocal_tail,
            jnp.asarray(self.plan.splitting_parameter, dtype=positions.dtype),
            compatible,
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.asarray(target.capacity, dtype=jnp.int32),
            jnp.asarray(
                source.capacity
                * target.capacity
                * (
                    self.plan.real_shifts.shape[0] + self.plan.reciprocal_vectors.shape[0]
                ),
                dtype=jnp.int32,
            ),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=positions.dtype),
            jnp.all(jnp.isfinite(source.safe_positions()))
            & jnp.all(jnp.isfinite(source.safe_strength())),
            finite,
            jnp.asarray(True),
            successful,
            backend,
        )
        return VortexVelocityEvaluation(
            velocity,
            gradient,
            vorticity_all,
            successful,
            self.backend_id,
            canonical_fingerprint(
                {
                    "kind": "periodic-vortex-ewald-evaluation",
                    "prepared": self.prepared_id,
                    "request": request.request_id,
                }
            ),
            diagnostics,
        )


__all__ = [
    "PeriodicVortexEwaldDiagnostics",
    "PeriodicVortexEwaldPlan",
    "PreparedPeriodicVortexEwald",
]
