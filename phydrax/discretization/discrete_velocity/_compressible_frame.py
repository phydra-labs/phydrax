#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._exponential_family import solve_finite_support_mean
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._compressible_contracts import CompressibleKineticPopulationState
from ._compressible_rules import (
    CompressibleVelocityRule,
    shift_compressible_velocity_rule,
)
from ._positive_kinetic import PositiveCompressibleKineticPlan


class IntegerKineticFramePlan(StrictModule, NonTrainableState):
    base_rule: CompressibleVelocityRule
    shifted_rule: CompressibleVelocityRule
    shift: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rule: CompressibleVelocityRule,
        shift: tuple[int, ...],
        /,
    ):
        if not isinstance(rule, CompressibleVelocityRule):
            raise TypeError("rule must be a CompressibleVelocityRule.")
        frame = tuple(int(value) for value in shift)
        self.base_rule = rule
        self.shifted_rule = shift_compressible_velocity_rule(rule, frame)
        self.shift = frame
        self.plan_id = canonical_fingerprint(
            {
                "kind": "integer-kinetic-frame",
                "base_rule": rule.rule_id,
                "shifted_rule": self.shifted_rule.rule_id,
                "shift": list(frame),
            }
        )


class KineticFrameRemapEvidence(StrictModule):
    mass_defect: Array
    momentum_defect: Array
    energy_defect: Array
    feature_defect: Array
    entropy_change: Array
    minimum_population: Array
    retained_guided_moments: Array
    successful: Array


class KineticFrameRemapResult(StrictModule):
    candidate: CompressibleKineticPopulationState
    previous: CompressibleKineticPopulationState
    evidence: KineticFrameRemapEvidence


def remap_kinetic_frame(
    source_model: PositiveCompressibleKineticPlan,
    target_model: PositiveCompressibleKineticPlan,
    state: CompressibleKineticPopulationState,
    /,
) -> KineticFrameRemapResult:
    """Remap positive populations between compatible prepared velocity frames."""
    if not isinstance(source_model, PositiveCompressibleKineticPlan) or not isinstance(
        target_model, PositiveCompressibleKineticPlan
    ):
        raise TypeError("source_model and target_model must be positive kinetic plans.")
    if source_model.rule.model_kind != target_model.rule.model_kind:
        raise ValueError("Frame remapping requires equal kinetic model kinds.")
    if source_model.rule.dual_dimension != target_model.rule.dual_dimension:
        raise ValueError("Frame remapping requires equal guided feature dimensions.")
    if source_model.layout.layout_id != target_model.layout.layout_id:
        raise ValueError("Frame remapping requires identical population layouts.")
    source_macro = source_model.moments(state)
    source_particle = state.population("particle")
    source_mean = (
        source_particle @ source_model.rule.guided_features.astype(source_particle.dtype)
    ) / source_macro.density[..., None]
    solved = solve_finite_support_mean(
        target_model.family,
        source_mean,
    )
    target_particle = source_macro.density[..., None] * solved.probabilities
    target_populations: list[Array] = [target_particle]
    if len(target_model.layout.fields) == 2:
        if len(source_model.layout.fields) != 2:
            raise ValueError("Frame remapping cannot create internal energy implicitly.")
        internal_total = jnp.sum(state.population("internal-energy"), axis=-1)
        target_populations.append(internal_total[..., None] * solved.probabilities)
    frame = jnp.broadcast_to(
        jnp.asarray(target_model.rule.frame_shift, dtype=source_particle.dtype),
        source_macro.density.shape + (target_model.rule.dimension,),
    )
    candidate = CompressibleKineticPopulationState(
        tuple(target_populations),
        solved.conversion.natural.values,
        jnp.full(source_macro.density.shape, 2.0, dtype=source_particle.dtype),
        frame,
        state.frame_temperature_scale,
        target_model.layout,
        target_model.model_id,
        target_model.rule.rule_id,
    )
    target_macro = target_model.moments(candidate)
    feature_reconstruction = (
        target_particle @ target_model.rule.guided_features.astype(target_particle.dtype)
    ) / target_macro.density[..., None]
    feature_defect = jnp.max(jnp.abs(feature_reconstruction - source_mean), axis=-1)
    mass_defect = target_macro.density - source_macro.density
    momentum_defect = jnp.max(
        jnp.abs(target_macro.momentum - source_macro.momentum), axis=-1
    )
    energy_defect = target_macro.total_energy - source_macro.total_energy
    source_entropy = jnp.sum(
        source_particle * jnp.log(source_particle / source_model.rule.base_probabilities),
        axis=-1,
    )
    target_entropy = jnp.sum(
        target_particle * jnp.log(target_particle / target_model.rule.base_probabilities),
        axis=-1,
    )
    minimum = jnp.min(
        jnp.stack(tuple(jnp.min(value, axis=-1) for value in target_populations), axis=0),
        axis=0,
    )
    scale = jnp.maximum(jnp.abs(source_macro.total_energy), 1.0)
    tolerance = (
        jnp.maximum(
            1024.0 * jnp.finfo(source_particle.dtype).eps,
            8.0 * target_model.family.solve_plan.residual_tolerance,
        )
        * scale
    )
    successful = (
        solved.evidence.successful
        & target_macro.admissible
        & (minimum > 0.0)
        & (jnp.abs(mass_defect) <= tolerance)
        & (momentum_defect <= tolerance)
        & (jnp.abs(energy_defect) <= tolerance)
        & (feature_defect <= tolerance)
    )
    evidence = KineticFrameRemapEvidence(
        mass_defect=mass_defect,
        momentum_defect=momentum_defect,
        energy_defect=energy_defect,
        feature_defect=feature_defect,
        entropy_change=target_entropy - source_entropy,
        minimum_population=minimum,
        retained_guided_moments=successful & (feature_defect <= tolerance),
        successful=successful,
    )
    return KineticFrameRemapResult(candidate, state, evidence)


class AdaptiveGaugePlan(StrictModule, NonTrainableState):
    reference_temperature: float = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    maximum_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_temperature: float = 1.0,
        minimum_scale: float = 0.25,
        maximum_scale: float = 4.0,
    ):
        reference = float(reference_temperature)
        lower = float(minimum_scale)
        upper = float(maximum_scale)
        if any(
            not np.isfinite(value) or value <= 0.0 for value in (reference, lower, upper)
        ):
            raise ValueError("Gauge temperatures and scales must be finite and positive.")
        if lower >= upper:
            raise ValueError("minimum_scale must be below maximum_scale.")
        self.reference_temperature = reference
        self.minimum_scale = lower
        self.maximum_scale = upper
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-kinetic-gauge",
                "reference_temperature": reference,
                "minimum_scale": lower,
                "maximum_scale": upper,
            }
        )

    def gauge(
        self,
        velocity: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        frame = jnp.asarray(velocity)
        thermal = jnp.asarray(temperature, dtype=frame.dtype)
        if frame.shape[:-1] != thermal.shape:
            raise ValueError("velocity and temperature batch shapes must match.")
        raw_scale = jnp.sqrt(thermal / self.reference_temperature)
        supported = (
            jnp.all(jnp.isfinite(frame), axis=-1)
            & jnp.isfinite(raw_scale)
            & (raw_scale >= self.minimum_scale)
            & (raw_scale <= self.maximum_scale)
        )
        scale = jnp.minimum(
            jnp.maximum(raw_scale, self.minimum_scale), self.maximum_scale
        )
        return frame, scale, supported

    def physical_velocities(
        self,
        rule: CompressibleVelocityRule,
        frame_velocity: ArrayLike,
        temperature_scale: ArrayLike,
        /,
    ) -> Array:
        if not isinstance(rule, CompressibleVelocityRule):
            raise TypeError("rule must be a CompressibleVelocityRule.")
        frame = jnp.asarray(frame_velocity)
        scale = jnp.asarray(temperature_scale, dtype=frame.dtype)
        if frame.shape[:-1] != scale.shape or frame.shape[-1] != rule.dimension:
            raise ValueError("Gauge frame and scale shapes are incompatible.")
        return frame[..., None, :] + scale[..., None, None] * rule.velocities


__all__ = [
    "AdaptiveGaugePlan",
    "IntegerKineticFramePlan",
    "KineticFrameRemapEvidence",
    "KineticFrameRemapResult",
    "remap_kinetic_frame",
]
