#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)


class SmoothCompressibleD2VForcingStatus(IntEnum):
    """Outcome of one coupled particle/energy population source transaction."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NONPOSITIVE_DENSITY = 2
    NONFINITE_CANDIDATE = 3
    INVALID_CANDIDATE = 4
    NONPOSITIVE_CANDIDATE = 5
    SOURCE_MOMENT_MISMATCH = 6
    INVALID_TIME_STEP = 7


class SmoothCompressibleD2VForcingEvidence(StrictModule):
    """Step-integrated moment, population, and work evidence for a D2V17 kick."""

    body_force: Array
    midpoint_velocity: Array
    acceleration_work_increment: Array
    volumetric_heating_increment: Array
    target_source_moments: Array
    recovered_source_moments: Array
    source_moment_residual: Array
    maximum_absolute_source_moment_residual: Array
    particle_source_second_moment: Array
    energy_source_first_moment: Array
    minimum_particle_population: Array
    minimum_total_energy_population: Array
    macroscopic_admissible: Array
    finite: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SmoothCompressibleD2VForcingResult(StrictModule):
    """Candidate, accepted state, and accepted conserved increment for one kick."""

    candidate_state: SmoothCompressibleKineticState
    accepted_state: SmoothCompressibleKineticState
    particle_population_increment: Array
    total_energy_population_increment: Array
    mass_momentum_energy_increment: Array
    evidence: SmoothCompressibleD2VForcingEvidence
    successful: Array
    rollback_applied: Array
    plan_id: str = eqx.field(static=True)


def _validate_method(method: SmoothCompressibleD2VKineticMethod, /) -> None:
    if not isinstance(method, SmoothCompressibleD2VKineticMethod):
        raise TypeError("method must be a SmoothCompressibleD2VKineticMethod.")
    quadrature = method.quadrature
    if (
        quadrature.name != "D2V17"
        or quadrature.dimension != 2
        or quadrature.population_count != 17
        or quadrature.transport_kind != "integer_lattice"
    ):
        raise ValueError(
            "Spatial forcing requires the certified integer-lattice D2V17 method."
        )


def _apply_source(
    method: SmoothCompressibleD2VKineticMethod,
    state: SmoothCompressibleKineticState,
    time_step: ArrayLike,
    acceleration: tuple[float, float],
    volumetric_heating: float,
    plan_id: str,
    /,
) -> SmoothCompressibleD2VForcingResult:
    method.validate_state(state)
    particles = state.particle_populations
    energy_populations = state.total_energy_populations
    step = jnp.asarray(time_step, dtype=particles.dtype)
    if step.ndim != 0:
        raise ValueError("D2V17 forcing time_step must be scalar.")

    density = jnp.sum(particles, axis=-1)
    momentum = ein.contract("...q,qd->...d", particles, method.quadrature.velocities)
    acceleration_value = jnp.asarray(acceleration, dtype=particles.dtype)
    heating_value = jnp.asarray(volumetric_heating, dtype=particles.dtype)
    body_force = density[..., None] * acceleration_value
    momentum_increment = step * body_force
    midpoint_velocity = (momentum + 0.5 * momentum_increment) / density[..., None]
    acceleration_work_increment = step * ein.contract(
        "...d,...d->...", midpoint_velocity, body_force
    )
    heating_increment = step * jnp.broadcast_to(heating_value, density.shape)
    total_energy_increment = acceleration_work_increment + heating_increment

    target_particle_moments = jnp.concatenate(
        (jnp.zeros_like(density)[..., None], momentum_increment), axis=-1
    )
    particle_increment = ein.contract(
        "qm,...m->...q", method.particle_moment_lift, target_particle_moments
    )
    energy_increment = method.energy_moment_lift * total_energy_increment[..., None]
    candidate = SmoothCompressibleKineticState(
        particles + particle_increment,
        energy_populations + energy_increment,
    )

    recovered_particle_moments = ein.contract(
        "mq,...q->...m", method.particle_moment_matrix, particle_increment
    )
    recovered_energy_increment = jnp.sum(energy_increment, axis=-1)
    target_source_moments = jnp.concatenate(
        (target_particle_moments, total_energy_increment[..., None]), axis=-1
    )
    recovered_source_moments = jnp.concatenate(
        (recovered_particle_moments, recovered_energy_increment[..., None]), axis=-1
    )
    source_residual = recovered_source_moments - target_source_moments
    maximum_residual = jnp.max(jnp.abs(source_residual))
    particle_second_moment = ein.contract(
        "...q,qd,qe->...de",
        particle_increment,
        method.quadrature.velocities,
        method.quadrature.velocities,
    )
    energy_first_moment = ein.contract(
        "...q,qd->...d", energy_increment, method.quadrature.velocities
    )
    minimum_particles = jnp.min(candidate.particle_populations, axis=-1)
    minimum_energy = jnp.min(candidate.total_energy_populations, axis=-1)
    candidate_realizability = method.realizability(candidate)

    input_finite = (
        jnp.isfinite(step)
        & jnp.all(jnp.isfinite(particles))
        & jnp.all(jnp.isfinite(energy_populations))
    )
    step_valid = jnp.isfinite(step) & (step > 0.0)
    density_positive = jnp.all(density > 0.0)
    output_finite = (
        jnp.all(jnp.isfinite(candidate.particle_populations))
        & jnp.all(jnp.isfinite(candidate.total_energy_populations))
        & jnp.all(jnp.isfinite(target_source_moments))
        & jnp.all(jnp.isfinite(recovered_source_moments))
        & jnp.all(jnp.isfinite(source_residual))
        & jnp.all(jnp.isfinite(particle_second_moment))
        & jnp.all(jnp.isfinite(energy_first_moment))
        & jnp.all(jnp.isfinite(midpoint_velocity))
    )
    macroscopic_admissible = jnp.all(candidate_realizability.macroscopic_admissible)
    populations_positive = jnp.all(minimum_particles > 0.0) & jnp.all(
        minimum_energy > 0.0
    )
    source_scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(target_source_moments)),
            jnp.max(jnp.abs(recovered_source_moments)),
        ),
        1.0,
    )
    source_tolerance = 512.0 * jnp.finfo(particles.dtype).eps * source_scale
    moments_exact = maximum_residual <= source_tolerance
    successful = (
        input_finite
        & step_valid
        & density_positive
        & output_finite
        & macroscopic_admissible
        & populations_positive
        & moments_exact
    )

    status = jnp.asarray(int(SmoothCompressibleD2VForcingStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~moments_exact,
        int(SmoothCompressibleD2VForcingStatus.SOURCE_MOMENT_MISMATCH),
        status,
    )
    status = jnp.where(
        ~macroscopic_admissible,
        int(SmoothCompressibleD2VForcingStatus.INVALID_CANDIDATE),
        status,
    )
    status = jnp.where(
        ~populations_positive,
        int(SmoothCompressibleD2VForcingStatus.NONPOSITIVE_CANDIDATE),
        status,
    )
    status = jnp.where(
        ~output_finite,
        int(SmoothCompressibleD2VForcingStatus.NONFINITE_CANDIDATE),
        status,
    )
    status = jnp.where(
        ~density_positive,
        int(SmoothCompressibleD2VForcingStatus.NONPOSITIVE_DENSITY),
        status,
    )
    status = jnp.where(
        ~step_valid,
        int(SmoothCompressibleD2VForcingStatus.INVALID_TIME_STEP),
        status,
    )
    status = jnp.where(
        ~input_finite,
        int(SmoothCompressibleD2VForcingStatus.NONFINITE_INPUT),
        status,
    ).astype(jnp.int32)

    accepted = SmoothCompressibleKineticState(
        jnp.where(successful, candidate.particle_populations, particles),
        jnp.where(successful, candidate.total_energy_populations, energy_populations),
    )
    accepted_increment = jnp.where(
        successful, recovered_source_moments, jnp.zeros_like(recovered_source_moments)
    )
    evidence = SmoothCompressibleD2VForcingEvidence(
        body_force=body_force,
        midpoint_velocity=midpoint_velocity,
        acceleration_work_increment=acceleration_work_increment,
        volumetric_heating_increment=heating_increment,
        target_source_moments=target_source_moments,
        recovered_source_moments=recovered_source_moments,
        source_moment_residual=source_residual,
        maximum_absolute_source_moment_residual=maximum_residual,
        particle_source_second_moment=particle_second_moment,
        energy_source_first_moment=energy_first_moment,
        minimum_particle_population=minimum_particles,
        minimum_total_energy_population=minimum_energy,
        macroscopic_admissible=macroscopic_admissible,
        finite=input_finite & output_finite,
        status=status,
        successful=successful,
        plan_id=plan_id,
    )
    return SmoothCompressibleD2VForcingResult(
        candidate_state=candidate,
        accepted_state=accepted,
        particle_population_increment=particle_increment,
        total_energy_population_increment=energy_increment,
        mass_momentum_energy_increment=accepted_increment,
        evidence=evidence,
        successful=successful,
        rollback_applied=~successful,
        plan_id=plan_id,
    )


class ZeroSmoothCompressibleD2VForcingPlan(StrictModule):
    """Exact no-source transaction for a certified D2V17 kinetic method."""

    method: SmoothCompressibleD2VKineticMethod
    plan_id: str = eqx.field(static=True)

    def __init__(self, method: SmoothCompressibleD2VKineticMethod, /):
        _validate_method(method)
        self.method = method
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v17-zero-forcing",
                "method": method.method_id,
            }
        )

    def apply(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        /,
    ) -> SmoothCompressibleD2VForcingResult:
        return _apply_source(
            self.method,
            state,
            time_step,
            (0.0, 0.0),
            0.0,
            self.plan_id,
        )

    __call__ = apply


class SmoothCompressibleD2VBodyForcingPlan(StrictModule):
    """Coupled D2V17 body-acceleration and volumetric-heating source kick."""

    method: SmoothCompressibleD2VKineticMethod
    acceleration: tuple[float, float] = eqx.field(static=True)
    volumetric_heating: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: SmoothCompressibleD2VKineticMethod,
        /,
        *,
        acceleration: Sequence[float] = (0.0, 0.0),
        volumetric_heating: float = 0.0,
    ):
        _validate_method(method)
        acceleration_value = tuple(float(value) for value in acceleration)
        heating_value = float(volumetric_heating)
        if len(acceleration_value) != 2 or any(
            not np.isfinite(value) for value in acceleration_value
        ):
            raise ValueError("acceleration must contain two finite components.")
        if not np.isfinite(heating_value):
            raise ValueError("volumetric_heating must be finite.")
        self.method = method
        self.acceleration = acceleration_value
        self.volumetric_heating = heating_value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v17-body-forcing",
                "method": method.method_id,
                "acceleration": acceleration_value,
                "volumetric_heating": heating_value,
                "energy_work": "midpoint-velocity",
            }
        )

    def apply(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        /,
    ) -> SmoothCompressibleD2VForcingResult:
        return _apply_source(
            self.method,
            state,
            time_step,
            self.acceleration,
            self.volumetric_heating,
            self.plan_id,
        )

    __call__ = apply


__all__ = [
    "SmoothCompressibleD2VBodyForcingPlan",
    "SmoothCompressibleD2VForcingEvidence",
    "SmoothCompressibleD2VForcingResult",
    "SmoothCompressibleD2VForcingStatus",
    "ZeroSmoothCompressibleD2VForcingPlan",
]
