#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._hyperbolic_systems import AbstractAdmissibleSystem
from ...equations._materials import IdealGasMaterial
from ._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
    SmoothCompressibleRealizabilityEvidence,
)


class KineticShockSensorEvidence(StrictModule):
    """Smooth sensor terms and the mandatory finite-volume ownership decision."""

    pressure_jump: Array
    density_jump: Array
    nonequilibrium: Array
    score: Array
    finite_volume_weight: Array
    fv_owned: Array
    kinetic_eligible: Array


class KineticShockSensorPlan(StrictModule, NonTrainableState):
    """Pressure/density/non-equilibrium shock sensor with FV-only shock ownership."""

    material: IdealGasMaterial
    pressure_weight: float = eqx.field(static=True)
    density_weight: float = eqx.field(static=True)
    nonequilibrium_weight: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    transition_width: float = eqx.field(static=True)
    shock_owner: str = eqx.field(static=True)
    sensor_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: IdealGasMaterial,
        /,
        *,
        pressure_weight: float = 1.0,
        density_weight: float = 0.5,
        nonequilibrium_weight: float = 0.5,
        threshold: float = 0.2,
        transition_width: float = 0.025,
    ):
        if not isinstance(material, IdealGasMaterial):
            raise TypeError("Shock sensing requires an IdealGasMaterial.")
        values = tuple(
            float(value)
            for value in (
                pressure_weight,
                density_weight,
                nonequilibrium_weight,
                threshold,
                transition_width,
            )
        )
        pressure_weight_, density_weight_, nonequilibrium_weight_, threshold_, width = (
            values
        )
        if (
            any(not np.isfinite(value) for value in values)
            or pressure_weight_ < 0.0
            or density_weight_ < 0.0
            or nonequilibrium_weight_ < 0.0
            or pressure_weight_ + density_weight_ + nonequilibrium_weight_ <= 0.0
            or threshold_ <= 0.0
            or width <= 0.0
        ):
            raise ValueError(
                "Kinetic shock-sensor weights, threshold, or width are invalid."
            )
        self.material = material
        self.pressure_weight = pressure_weight_
        self.density_weight = density_weight_
        self.nonequilibrium_weight = nonequilibrium_weight_
        self.threshold = threshold_
        self.transition_width = width
        self.shock_owner = "finite_volume"
        self.sensor_id = canonical_fingerprint(
            {
                "kind": "kinetic-shock-sensor-v1",
                "material": material.material_id,
                "pressure_weight": pressure_weight_,
                "density_weight": density_weight_,
                "nonequilibrium_weight": nonequilibrium_weight_,
                "threshold": threshold_,
                "transition_width": width,
                "shock_owner": "finite_volume",
            }
        )

    def _pressure(self, conserved: Array, /) -> Array:
        density = conserved[..., 0]
        momentum = conserved[..., 1:-1]
        total_energy = conserved[..., -1]
        velocity = momentum / density[..., None]
        kinetic = 0.5 * oe.contract("...d,...d->...", momentum, velocity)
        return self.material.pressure(density, (total_energy - kinetic) / density)

    def evaluate(
        self,
        left_conserved: ArrayLike,
        right_conserved: ArrayLike,
        kinetic_state: SmoothCompressibleKineticState,
        kinetic_equilibrium: SmoothCompressibleKineticState,
        /,
    ) -> KineticShockSensorEvidence:
        left = jnp.asarray(left_conserved)
        right = jnp.asarray(right_conserved)
        if left.shape != right.shape or left.ndim == 0 or left.shape[-1] < 3:
            raise ValueError(
                "Shock-sensor face states must have equal trailing (D + 2) shapes."
            )
        if left.dtype != right.dtype or not jnp.issubdtype(left.dtype, jnp.inexact):
            raise TypeError("Shock-sensor face states must share one inexact dtype.")
        if (
            kinetic_state.particle_populations.shape
            != kinetic_equilibrium.particle_populations.shape
            or kinetic_state.total_energy_populations.shape
            != kinetic_equilibrium.total_energy_populations.shape
            or kinetic_state.particle_populations.shape[:-1] != left.shape[:-1]
        ):
            raise ValueError(
                "Shock-sensor kinetic fields must match the face batch shape."
            )
        left_pressure = self._pressure(left)
        right_pressure = self._pressure(right)
        invalid = (
            ~jnp.all(jnp.isfinite(left), axis=-1)
            | ~jnp.all(jnp.isfinite(right), axis=-1)
            | ~self.material.admissible(left[..., 0], left_pressure)
            | ~self.material.admissible(right[..., 0], right_pressure)
            | ~jnp.all(jnp.isfinite(kinetic_state.particle_populations), axis=-1)
            | ~jnp.all(jnp.isfinite(kinetic_state.total_energy_populations), axis=-1)
            | ~jnp.all(jnp.isfinite(kinetic_equilibrium.particle_populations), axis=-1)
            | ~jnp.all(
                jnp.isfinite(kinetic_equilibrium.total_energy_populations), axis=-1
            )
        )
        left = eqx.error_if(
            left,
            jnp.any(invalid),
            "Kinetic shock sensing requires finite admissible face and population states.",
        )
        tiny = jnp.finfo(left.dtype).tiny
        pressure_jump = jnp.abs(right_pressure - left_pressure) / jnp.maximum(
            jnp.abs(right_pressure) + jnp.abs(left_pressure), tiny
        )
        density_jump = jnp.abs(right[..., 0] - left[..., 0]) / jnp.maximum(
            jnp.abs(right[..., 0]) + jnp.abs(left[..., 0]), tiny
        )
        particle_delta = (
            kinetic_state.particle_populations - kinetic_equilibrium.particle_populations
        )
        energy_delta = (
            kinetic_state.total_energy_populations
            - kinetic_equilibrium.total_energy_populations
        )
        delta_norm = jnp.sqrt(
            jnp.sum(particle_delta**2, axis=-1) + jnp.sum(energy_delta**2, axis=-1)
        )
        equilibrium_norm = jnp.sqrt(
            jnp.sum(kinetic_equilibrium.particle_populations**2, axis=-1)
            + jnp.sum(kinetic_equilibrium.total_energy_populations**2, axis=-1)
        )
        nonequilibrium = delta_norm / jnp.maximum(equilibrium_norm, tiny)
        score = (
            self.pressure_weight * pressure_jump
            + self.density_weight * density_jump
            + self.nonequilibrium_weight * nonequilibrium
        )
        fv_weight = jax_sigmoid((score - self.threshold) / self.transition_width)
        fv_owned = score >= self.threshold
        return KineticShockSensorEvidence(
            pressure_jump=pressure_jump,
            density_jump=density_jump,
            nonequilibrium=nonequilibrium,
            score=score,
            finite_volume_weight=fv_weight,
            fv_owned=fv_owned,
            kinetic_eligible=~fv_owned,
        )


def jax_sigmoid(value: Array, /) -> Array:
    """Overflow-safe logistic used by the smooth ownership diagnostic."""

    return jnn.sigmoid(value)


class ConformingFVKineticState(StrictModule):
    """One interface-adjacent FV state and its kinetic neighbor."""

    finite_volume_conserved: Array
    kinetic: SmoothCompressibleKineticState

    def __init__(
        self,
        finite_volume_conserved: ArrayLike,
        kinetic: SmoothCompressibleKineticState,
        /,
    ):
        conserved = jnp.asarray(finite_volume_conserved)
        if not isinstance(kinetic, SmoothCompressibleKineticState):
            raise TypeError("kinetic must be SmoothCompressibleKineticState.")
        if (
            conserved.ndim == 0
            or conserved.shape[:-1] != kinetic.particle_populations.shape[:-1]
        ):
            raise ValueError(
                "FV and kinetic states must share the interface batch shape."
            )
        self.finite_volume_conserved = conserved
        self.kinetic = kinetic


class CommonFVKineticFluxEvidence(StrictModule):
    """One population flux and its exactly shared conservative moment flux."""

    particle_population_flux: Array
    total_energy_population_flux: Array
    common_conservative_flux: Array
    finite_volume_outward_flux: Array
    kinetic_outward_flux: Array
    flux_equality_residual: Array
    lifted_finite_volume_state: SmoothCompressibleKineticState
    recovered_lifted_conserved: Array
    moment_lift_residual: Array
    maximum_flux_equality_residual: Array
    maximum_moment_lift_residual: Array


class AtomicHybridUpdateEvidence(StrictModule):
    """All-or-nothing structured acceptance and rollback evidence."""

    accepted: Array
    rollback_applied: Array
    finite_volume_admissible: Array
    kinetic_realizability: SmoothCompressibleRealizabilityEvidence
    flux_equality_residual: Array
    moment_lift_residual: Array


class AtomicHybridUpdateResult(StrictModule):
    previous: ConformingFVKineticState
    candidate: ConformingFVKineticState
    committed: ConformingFVKineticState
    evidence: AtomicHybridUpdateEvidence


class FixedConformingFVKineticInterfacePlan(StrictModule, NonTrainableState):
    """Fixed conforming interface using one common population-derived flux.

    The normal is oriented from the finite-volume cell to the kinetic cell.
    Shocks are always finite-volume-owned; this plan has no runtime ownership
    switch. Candidate FV and both kinetic population fields commit atomically.
    """

    method: SmoothCompressibleD2VKineticMethod
    finite_volume_system: AbstractAdmissibleSystem
    normal: Array
    face_id: str = eqx.field(static=True)
    shock_owner: str = eqx.field(static=True)
    population_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: SmoothCompressibleD2VKineticMethod,
        finite_volume_system: AbstractAdmissibleSystem,
        normal: ArrayLike,
        face_id: str,
        /,
        *,
        population_floor: float = 0.0,
    ):
        if not isinstance(method, SmoothCompressibleD2VKineticMethod):
            raise TypeError("method must be SmoothCompressibleD2VKineticMethod.")
        if not isinstance(finite_volume_system, AbstractAdmissibleSystem):
            raise TypeError("finite_volume_system must be AbstractAdmissibleSystem.")
        if (
            finite_volume_system.dimension != method.quadrature.dimension
            or finite_volume_system.component_count != method.quadrature.dimension + 2
        ):
            raise ValueError(
                "FV system must use the matching compressible (D + 2) layout."
            )
        normal_ = np.asarray(normal)
        identifier = str(face_id)
        floor = float(population_floor)
        if normal_.shape != (method.quadrature.dimension,) or np.any(
            ~np.isfinite(normal_)
        ):
            raise ValueError("Fixed interface normal must have finite shape (D,).")
        length = float(np.linalg.norm(normal_))
        if not np.isfinite(length) or abs(length - 1.0) > 1e-12:
            raise ValueError("Fixed interface normal must be unit length.")
        if not identifier:
            raise ValueError("face_id must be non-empty.")
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("population_floor must be finite and non-negative.")
        self.method = method
        self.finite_volume_system = finite_volume_system
        self.normal = jnp.asarray(normal_, dtype=method.quadrature.velocities.dtype)
        self.face_id = identifier
        self.shock_owner = "finite_volume"
        self.population_floor = floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-conforming-fv-kinetic-interface-v1",
                "method": method.method_id,
                "finite_volume_system": finite_volume_system.system_id,
                "normal": array_tree_fingerprint(normal_),
                "face_id": identifier,
                "population_floor": floor,
                "shock_owner": "finite_volume",
            }
        )

    def common_flux(
        self,
        finite_volume_conserved: ArrayLike,
        kinetic_state: SmoothCompressibleKineticState,
        /,
    ) -> CommonFVKineticFluxEvidence:
        conserved = jnp.asarray(finite_volume_conserved)
        if (
            conserved.ndim == 0
            or conserved.shape[-1] != self.method.quadrature.dimension + 2
        ):
            raise ValueError("FV interface state must have trailing shape (D + 2,).")
        self.method.validate_state(kinetic_state)
        if conserved.shape[:-1] != kinetic_state.particle_populations.shape[:-1]:
            raise ValueError(
                "FV and kinetic interface states must share their batch shape."
            )
        lifted = self.method.equilibrium(conserved)
        normal_velocities = oe.contract(
            "qd,d->q", self.method.quadrature.velocities, self.normal
        )
        particles_upwind = jnp.where(
            normal_velocities >= 0.0,
            lifted.particle_populations,
            kinetic_state.particle_populations,
        )
        energy_upwind = jnp.where(
            normal_velocities >= 0.0,
            lifted.total_energy_populations,
            kinetic_state.total_energy_populations,
        )
        particle_flux = particles_upwind * normal_velocities
        energy_flux = energy_upwind * normal_velocities
        mass_flux = jnp.sum(particle_flux, axis=-1)
        momentum_flux = oe.contract(
            "...q,qd->...d", particle_flux, self.method.quadrature.velocities
        )
        total_energy_flux = jnp.sum(energy_flux, axis=-1)
        common = jnp.concatenate(
            (mass_flux[..., None], momentum_flux, total_energy_flux[..., None]), axis=-1
        )
        finite_volume_outward = common
        kinetic_outward = -common
        flux_residual = finite_volume_outward + kinetic_outward
        recovered_lift = self.method.moments(lifted).conserved
        lift_residual = recovered_lift - conserved
        return CommonFVKineticFluxEvidence(
            particle_population_flux=particle_flux,
            total_energy_population_flux=energy_flux,
            common_conservative_flux=common,
            finite_volume_outward_flux=finite_volume_outward,
            kinetic_outward_flux=kinetic_outward,
            flux_equality_residual=flux_residual,
            lifted_finite_volume_state=lifted,
            recovered_lifted_conserved=recovered_lift,
            moment_lift_residual=lift_residual,
            maximum_flux_equality_residual=jnp.max(jnp.abs(flux_residual)),
            maximum_moment_lift_residual=jnp.max(jnp.abs(lift_residual)),
        )

    def atomic_update(
        self,
        previous: ConformingFVKineticState,
        flux: CommonFVKineticFluxEvidence,
        finite_volume_scale: ArrayLike,
        kinetic_scale: ArrayLike,
        /,
    ) -> AtomicHybridUpdateResult:
        if not isinstance(previous, ConformingFVKineticState):
            raise TypeError("previous must be ConformingFVKineticState.")
        if not isinstance(flux, CommonFVKineticFluxEvidence):
            raise TypeError("flux must be CommonFVKineticFluxEvidence.")
        fv_scale = jnp.asarray(
            finite_volume_scale, dtype=previous.finite_volume_conserved.dtype
        )
        kinetic_scale_ = jnp.asarray(
            kinetic_scale, dtype=previous.kinetic.particle_populations.dtype
        )
        invalid_scale = jnp.any(~jnp.isfinite(fv_scale) | (fv_scale < 0.0)) | jnp.any(
            ~jnp.isfinite(kinetic_scale_) | (kinetic_scale_ < 0.0)
        )
        fv_scale = eqx.error_if(
            fv_scale,
            invalid_scale,
            "Hybrid interface update scales must be finite and non-negative.",
        )
        candidate = ConformingFVKineticState(
            previous.finite_volume_conserved
            - fv_scale[..., None] * flux.common_conservative_flux,
            SmoothCompressibleKineticState(
                previous.kinetic.particle_populations
                + kinetic_scale_[..., None] * flux.particle_population_flux,
                previous.kinetic.total_energy_populations
                + kinetic_scale_[..., None] * flux.total_energy_population_flux,
            ),
        )
        fv_valid = self.finite_volume_system.admissible(candidate.finite_volume_conserved)
        kinetic_evidence = self.method.realizability(
            candidate.kinetic, population_floor=self.population_floor
        )
        accepted = jnp.all(fv_valid) & kinetic_evidence.realizable
        committed = ConformingFVKineticState(
            jnp.where(
                accepted,
                candidate.finite_volume_conserved,
                previous.finite_volume_conserved,
            ),
            SmoothCompressibleKineticState(
                jnp.where(
                    accepted,
                    candidate.kinetic.particle_populations,
                    previous.kinetic.particle_populations,
                ),
                jnp.where(
                    accepted,
                    candidate.kinetic.total_energy_populations,
                    previous.kinetic.total_energy_populations,
                ),
            ),
        )
        return AtomicHybridUpdateResult(
            previous=previous,
            candidate=candidate,
            committed=committed,
            evidence=AtomicHybridUpdateEvidence(
                accepted=accepted,
                rollback_applied=~accepted,
                finite_volume_admissible=fv_valid,
                kinetic_realizability=kinetic_evidence,
                flux_equality_residual=flux.flux_equality_residual,
                moment_lift_residual=flux.moment_lift_residual,
            ),
        )


__all__ = [
    "AtomicHybridUpdateEvidence",
    "AtomicHybridUpdateResult",
    "CommonFVKineticFluxEvidence",
    "ConformingFVKineticState",
    "FixedConformingFVKineticInterfacePlan",
    "KineticShockSensorEvidence",
    "KineticShockSensorPlan",
]
