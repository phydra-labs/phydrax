#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import (
    ParticleInternalGeometry,
    PreparedRadialSpeciesTransport,
    RadialShellMeshPlan,
    RadialSpeciesTransportPlan,
)


class BatteryParticleEvaluation(StrictModule):
    """Conservative representative-particle transport and reconstruction evidence."""

    concentration_mol_m3: Array
    center_concentration_mol_m3: Array
    surface_concentration_mol_m3: Array
    average_concentration_mol_m3: Array
    face_molar_flux_mol_m2_s: Array
    amount_rate_mol_s: Array
    outer_amount_rate_mol_s: Array
    total_amount_mol: Array
    active_surface_area_m2: Array
    specific_surface_area_m2_m3: Array
    explicit_dt_limit_s: Array
    conservation_residual_mol_s: Array
    domain_valid: Array

    @property
    def successful(self) -> Array:
        """Whether geometry, material data, state, and conservative transport are valid."""
        return self.domain_valid


class BatteryParticlePlan(StrictModule, NonTrainableState):
    """Fixed spherical-shell topology for one battery representative particle."""

    mesh: RadialShellMeshPlan
    particle_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell_count: int,
        /,
        *,
        reference_faces: ArrayLike | None = None,
        particle_id: str = "particle",
    ):
        if isinstance(shell_count, bool) or not isinstance(shell_count, int):
            raise TypeError("shell_count must be an integer.")
        if (
            not isinstance(particle_id, str)
            or not particle_id
            or particle_id != particle_id.strip()
        ):
            raise ValueError("particle_id must be a non-empty canonical identifier.")
        mesh = RadialShellMeshPlan(
            ParticleInternalGeometry.SPHERE,
            shell_count,
            reference_faces=reference_faces,
            mesh_id=f"battery-particle:{particle_id}:radial-shells",
        )
        self.mesh = mesh
        self.particle_id = particle_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-particle-plan",
                "particle_id": particle_id,
                "mesh_id": mesh.mesh_id,
            }
        )

    @property
    def shell_count(self) -> int:
        return self.mesh.cell_count

    def prepare(self, /) -> "PreparedBatteryParticle":
        """Prepare the public radial species transport on this immutable topology."""
        return PreparedBatteryParticle(self)


class PreparedBatteryParticle(StrictModule, NonTrainableState):
    """Prepared representative particle with extensive electrode-scale amounts.

    State amounts are totals over every identical active-material particle in an
    electrode. The public radial transport is evaluated per representative particle,
    then multiplied by the particle multiplicity. This leaves concentrations and
    molar surface flux intensive while conserving electrode-scale lithium exactly.
    """

    plan: BatteryParticlePlan
    transport: PreparedRadialSpeciesTransport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: BatteryParticlePlan, /):
        if not isinstance(plan, BatteryParticlePlan):
            raise TypeError("plan must be a BatteryParticlePlan.")
        mesh = plan.mesh.prepare()
        transport = RadialSpeciesTransportPlan(
            1,
            plan_id=f"battery-particle:{plan.particle_id}:solid-lithium",
        ).prepare(mesh)
        self.plan = plan
        self.transport = transport
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-particle",
                "plan_id": plan.plan_id,
                "transport_id": transport.prepared_id,
            }
        )

    @property
    def shell_count(self) -> int:
        return self.plan.shell_count

    def shell_measures(self, particle_radius_m: ArrayLike, /) -> Array:
        """Return physical shell volumes for one representative particle."""
        radius = jnp.asarray(particle_radius_m)
        if radius.shape != ():
            raise ValueError("particle_radius_m must be scalar.")
        safe_radius = jnp.where(jnp.isfinite(radius) & (radius > 0.0), radius, 1.0)
        return self.transport.mesh.metrics(safe_radius[None]).cell_measures[0]

    def initial_amounts(
        self,
        concentration_mol_m3: ArrayLike,
        particle_radius_m: ArrayLike,
        particle_multiplicity: ArrayLike,
        /,
    ) -> Array:
        """Construct uniform extensive shell amounts over all represented particles."""
        concentration = jnp.asarray(concentration_mol_m3)
        radius = jnp.asarray(particle_radius_m)
        multiplicity = jnp.asarray(particle_multiplicity)
        if concentration.shape != () or radius.shape != () or multiplicity.shape != ():
            raise ValueError("Particle initialization inputs must be scalar.")
        invalid = (
            ~jnp.isfinite(concentration)
            | (concentration < 0.0)
            | ~jnp.isfinite(radius)
            | (radius <= 0.0)
            | ~jnp.isfinite(multiplicity)
            | (multiplicity <= 0.0)
        )
        concentration = eqx.error_if(
            concentration,
            invalid,
            "Battery particle initialization requires finite nonnegative concentration, "
            "positive radius, and positive multiplicity.",
        )
        return concentration * multiplicity * self.shell_measures(radius)

    def concentrations(
        self,
        amounts_mol: ArrayLike,
        particle_radius_m: ArrayLike,
        particle_multiplicity: ArrayLike,
        /,
    ) -> Array:
        """Reconstruct shell concentrations from extensive represented amounts."""
        amounts = jnp.asarray(amounts_mol)
        radius = jnp.asarray(particle_radius_m)
        multiplicity = jnp.asarray(particle_multiplicity)
        if amounts.ndim < 1 or amounts.shape[-1] != self.shell_count:
            raise ValueError(
                f"amounts_mol must have trailing shape ({self.shell_count},)."
            )
        if radius.shape != () or multiplicity.shape != ():
            raise ValueError("Particle radius and multiplicity must be scalar.")
        return amounts / (multiplicity * self.shell_measures(radius))

    def evaluate(
        self,
        amounts_mol: ArrayLike,
        /,
        *,
        particle_radius_m: ArrayLike,
        particle_multiplicity: ArrayLike,
        support_volume_m3: ArrayLike,
        diffusivity_m2_s: ArrayLike,
        outward_molar_flux_mol_m2_s: ArrayLike,
    ) -> BatteryParticleEvaluation:
        """Evaluate spherical diffusion with outward-positive solid molar flux."""
        amounts = jnp.asarray(amounts_mol)
        if amounts.ndim < 1 or amounts.shape[-1] != self.shell_count:
            raise ValueError(
                f"amounts_mol must have trailing shape ({self.shell_count},)."
            )
        leading_shape = amounts.shape[:-1]
        radius = jnp.asarray(particle_radius_m)
        multiplicity = jnp.asarray(particle_multiplicity)
        support_volume = jnp.asarray(support_volume_m3)
        diffusivity = jnp.asarray(diffusivity_m2_s)
        if any(value.shape != () for value in (radius, multiplicity, support_volume)):
            raise ValueError(
                "Particle radius, multiplicity, and support volume must be scalar."
            )
        if diffusivity.shape == ():
            diffusivity = jnp.broadcast_to(diffusivity, amounts.shape)
        elif diffusivity.shape == (self.shell_count,):
            diffusivity = jnp.broadcast_to(diffusivity, amounts.shape)
        elif diffusivity.shape != amounts.shape:
            raise ValueError(
                "diffusivity_m2_s must be scalar, have shape (shell,), or match "
                "the complete amount shape."
            )
        outward_flux = jnp.asarray(outward_molar_flux_mol_m2_s)
        if outward_flux.shape == ():
            outward_flux = jnp.broadcast_to(outward_flux, leading_shape)
        elif outward_flux.shape != leading_shape:
            raise ValueError(
                "outward_molar_flux_mol_m2_s must be scalar or match amount leading axes."
            )

        geometry_valid = (
            jnp.isfinite(radius)
            & (radius > 0.0)
            & jnp.isfinite(multiplicity)
            & (multiplicity > 0.0)
            & jnp.isfinite(support_volume)
            & (support_volume > 0.0)
        )
        diffusivity_valid = jnp.all(
            jnp.isfinite(diffusivity) & (diffusivity > 0.0), axis=-1
        )
        state_valid = jnp.all(jnp.isfinite(amounts) & (amounts >= 0.0), axis=-1)
        flux_valid = jnp.isfinite(outward_flux)
        active = state_valid & flux_valid & geometry_valid & diffusivity_valid
        safe_radius = jnp.where(geometry_valid, radius, 1.0)
        safe_multiplicity = jnp.where(geometry_valid, multiplicity, 1.0)
        safe_support_volume = jnp.where(geometry_valid, support_volume, 1.0)
        safe_diffusivity = jnp.where(
            jnp.isfinite(diffusivity) & (diffusivity > 0.0),
            diffusivity,
            1.0,
        )
        safe_amounts = jnp.where(jnp.isfinite(amounts), amounts, 0.0)
        safe_outward_flux = jnp.where(flux_valid, outward_flux, 0.0)

        metrics = self.transport.mesh.metrics(safe_radius[None])
        single_particle_shell_volume = metrics.cell_measures[0]
        per_particle_amount = safe_amounts / safe_multiplicity
        radial = self.transport.evaluate(
            per_particle_amount[..., :, None],
            outer_scale=safe_radius,
            storage_measure=single_particle_shell_volume,
            cell_diffusivity=safe_diffusivity[..., :, None],
            outer_molar_flux=safe_outward_flux[..., None],
            active_mask=active,
        )
        concentration = radial.concentrations[..., 0]
        total_amount_rate = radial.amount_rate[..., 0] * safe_multiplicity
        outer_amount_rate = radial.outer_amount_rate[..., 0] * safe_multiplicity
        face_flux = radial.face_molar_flux[..., 0]
        total_amount = jnp.sum(safe_amounts, axis=-1)
        active_surface_area = metrics.surface_measure[0] * safe_multiplicity
        specific_surface_area = active_surface_area / safe_support_volume
        represented_volume = jnp.sum(single_particle_shell_volume) * safe_multiplicity
        average_concentration = total_amount / represented_volume

        surface_distance = safe_radius * (1.0 - self.transport.mesh.reference_cells[-1])
        surface_concentration = (
            concentration[..., -1]
            - safe_outward_flux * surface_distance / safe_diffusivity[..., -1]
        )
        center_concentration = concentration[..., 0]
        conservation_residual = radial.conservation_defect[..., 0] * safe_multiplicity
        finite_output = (
            jnp.all(jnp.isfinite(concentration), axis=-1)
            & jnp.isfinite(center_concentration)
            & jnp.isfinite(surface_concentration)
            & jnp.isfinite(average_concentration)
            & jnp.all(jnp.isfinite(total_amount_rate), axis=-1)
            & jnp.isfinite(outer_amount_rate)
            & jnp.isfinite(active_surface_area)
            & jnp.isfinite(specific_surface_area)
            & jnp.isfinite(conservation_residual)
        )
        reconstructed_nonnegative = (
            jnp.all(concentration >= 0.0, axis=-1)
            & (surface_concentration >= 0.0)
            & (average_concentration >= 0.0)
        )
        domain_valid = (
            active & radial.successful & finite_output & reconstructed_nonnegative
        )
        return BatteryParticleEvaluation(
            concentration,
            center_concentration,
            surface_concentration,
            average_concentration,
            face_flux,
            total_amount_rate,
            outer_amount_rate,
            total_amount,
            active_surface_area,
            specific_surface_area,
            radial.explicit_dt_limit,
            conservation_residual,
            domain_valid,
        )


__all__ = [
    "BatteryParticleEvaluation",
    "BatteryParticlePlan",
    "PreparedBatteryParticle",
]
