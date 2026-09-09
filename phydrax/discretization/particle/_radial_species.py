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
from ._particle_internal_mesh import PreparedRadialShellMesh


class RadialSpeciesTransportPlan(StrictModule, NonTrainableState):
    """Static species layout for conservative radial amount transport."""

    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, species_count: int, /, *, plan_id: str | None = None):
        count = int(species_count)
        if count <= 0:
            raise ValueError("species_count must be positive.")
        generated = canonical_fingerprint(
            {"kind": "radial-species-transport-plan", "species_count": count}
        )
        self.species_count = count
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(self, mesh: PreparedRadialShellMesh, /) -> PreparedRadialSpeciesTransport:
        return prepare_radial_species_transport(self, mesh)


class RadialSpeciesTransportResult(StrictModule):
    """One matrix-free conservative radial species-amount evaluation."""

    concentrations: Array
    face_molar_flux: Array
    amount_rate: Array
    outer_amount_rate: Array
    explicit_dt_limit: Array
    conservation_defect: Array
    successful: Array


class PreparedRadialSpeciesTransport(StrictModule, NonTrainableState):
    """Prepared radial topology with runtime geometry and transport properties.

    Amounts have shape ``(..., shell, species)``. ``outer_scale`` and
    ``active_mask`` are scalars or have the exact leading shape. Storage measure
    has shape ``(shell,)`` or ``(..., shell)``. Cell diffusivity has shape
    ``(species,)``, ``(shell, species)``, ``(..., 1, species)``, or
    ``(..., shell, species)``. Outer molar flux has shape ``(species,)`` or
    ``(..., species)`` and is positive along the outward normal.
    """

    plan: RadialSpeciesTransportPlan
    mesh: PreparedRadialShellMesh
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RadialSpeciesTransportPlan,
        mesh: PreparedRadialShellMesh,
        /,
    ):
        if not isinstance(plan, RadialSpeciesTransportPlan):
            raise TypeError("plan must be a RadialSpeciesTransportPlan.")
        if not isinstance(mesh, PreparedRadialShellMesh):
            raise TypeError("mesh must be a PreparedRadialShellMesh.")
        self.plan = plan
        self.mesh = mesh
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-radial-species-transport",
                "plan": plan.plan_id,
                "mesh": mesh.prepared_id,
            }
        )

    def evaluate(
        self,
        amounts: ArrayLike,
        /,
        *,
        outer_scale: ArrayLike,
        storage_measure: ArrayLike,
        cell_diffusivity: ArrayLike,
        outer_molar_flux: ArrayLike,
        active_mask: ArrayLike,
    ) -> RadialSpeciesTransportResult:
        """Evaluate fluxes and amount rates without materializing an operator."""
        amount = jnp.asarray(amounts)
        shell_count = self.mesh.cell_capacity
        species_count = self.plan.species_count
        if amount.ndim < 2 or amount.shape[-2:] != (shell_count, species_count):
            raise ValueError(
                f"amounts must have trailing shape ({shell_count}, {species_count})."
            )
        leading_shape = amount.shape[:-2]

        scale = jnp.asarray(outer_scale)
        if scale.shape == ():
            scale = jnp.broadcast_to(scale, leading_shape)
        elif scale.shape != leading_shape:
            raise ValueError("outer_scale must be scalar or match the leading shape.")

        active = jnp.asarray(active_mask)
        if active.dtype != jnp.bool_:
            raise TypeError("active_mask must have boolean dtype.")
        if active.shape == ():
            active = jnp.broadcast_to(active, leading_shape)
        elif active.shape != leading_shape:
            raise ValueError("active_mask must be scalar or match the leading shape.")

        storage = jnp.asarray(storage_measure)
        storage_shape = leading_shape + (shell_count,)
        if storage.shape == (shell_count,):
            storage = jnp.broadcast_to(storage, storage_shape)
        elif storage.shape != storage_shape:
            raise ValueError("storage_measure must have shape (shell,) or (..., shell).")

        diffusivity = jnp.asarray(cell_diffusivity)
        amount_shape = leading_shape + (shell_count, species_count)
        if diffusivity.shape == (species_count,):
            diffusivity = jnp.broadcast_to(diffusivity, amount_shape)
        elif diffusivity.shape == (shell_count, species_count):
            diffusivity = jnp.broadcast_to(diffusivity, amount_shape)
        elif diffusivity.shape == leading_shape + (1, species_count):
            diffusivity = jnp.broadcast_to(diffusivity, amount_shape)
        elif diffusivity.shape != amount_shape:
            raise ValueError(
                "cell_diffusivity must have shape (species,), (shell, species), "
                "(..., 1, species), or (..., shell, species)."
            )

        outer_flux = jnp.asarray(outer_molar_flux)
        outer_flux_shape = leading_shape + (species_count,)
        if outer_flux.shape == (species_count,):
            outer_flux = jnp.broadcast_to(outer_flux, outer_flux_shape)
        elif outer_flux.shape != outer_flux_shape:
            raise ValueError(
                "outer_molar_flux must have shape (species,) or (..., species)."
            )

        amount_finite = jnp.all(jnp.isfinite(amount), axis=(-2, -1))
        scale_finite = jnp.isfinite(scale)
        storage_finite = jnp.all(jnp.isfinite(storage), axis=-1)
        diffusivity_finite = jnp.all(jnp.isfinite(diffusivity), axis=(-2, -1))
        outer_flux_finite = jnp.all(jnp.isfinite(outer_flux), axis=-1)
        active_values_valid = (
            (scale > 0.0)
            & jnp.all(storage > 0.0, axis=-1)
            & jnp.all(diffusivity >= 0.0, axis=(-2, -1))
        )
        inputs_valid = (
            amount_finite
            & scale_finite
            & storage_finite
            & diffusivity_finite
            & outer_flux_finite
            & (~active | active_values_valid)
        )

        safe_scale = jnp.where(active & scale_finite & (scale > 0.0), scale, 1.0)
        metrics_flat = self.mesh.metrics(safe_scale.reshape((-1,)))
        face_measure = metrics_flat.face_measures.reshape(
            leading_shape + (shell_count + 1,)
        )
        center_distance = metrics_flat.center_distances.reshape(
            leading_shape + (shell_count - 1,)
        )
        geometry_valid = jnp.all(
            jnp.isfinite(face_measure) & (face_measure >= 0.0),
            axis=-1,
        ) & jnp.all(
            jnp.isfinite(center_distance) & (center_distance > 0.0),
            axis=-1,
        )

        active_species = active[..., None, None]
        safe_amount = jnp.where(active_species & jnp.isfinite(amount), amount, 0.0)
        safe_storage = jnp.where(
            active[..., None] & jnp.isfinite(storage) & (storage > 0.0),
            storage,
            1.0,
        )
        safe_diffusivity = jnp.where(
            active_species & jnp.isfinite(diffusivity) & (diffusivity >= 0.0),
            diffusivity,
            0.0,
        )
        safe_outer_flux = jnp.where(
            active[..., None] & jnp.isfinite(outer_flux), outer_flux, 0.0
        )
        concentrations = jnp.where(
            active_species,
            safe_amount / safe_storage[..., :, None],
            0.0,
        )
        left_diffusivity = safe_diffusivity[..., :-1, :]
        right_diffusivity = safe_diffusivity[..., 1:, :]
        minimum_diffusivity = jnp.minimum(left_diffusivity, right_diffusivity)
        maximum_diffusivity = jnp.maximum(left_diffusivity, right_diffusivity)
        positive_maximum = maximum_diffusivity > 0.0
        safe_maximum = jnp.where(positive_maximum, maximum_diffusivity, 1.0)
        diffusivity_ratio = minimum_diffusivity / safe_maximum
        face_diffusivity = jnp.where(
            positive_maximum,
            2.0 * minimum_diffusivity / (1.0 + diffusivity_ratio),
            0.0,
        )
        interior_flux = (
            face_diffusivity
            * (concentrations[..., :-1, :] - concentrations[..., 1:, :])
            / center_distance[..., :, None]
        )
        center_flux = jnp.zeros(
            leading_shape + (1, species_count), dtype=interior_flux.dtype
        )
        face_molar_flux = jnp.concatenate(
            (center_flux, interior_flux, safe_outer_flux[..., None, :]), axis=-2
        )
        face_molar_flux = jnp.where(active_species, face_molar_flux, 0.0)

        integrated_face_rate = face_molar_flux * face_measure[..., :, None]
        amount_rate = integrated_face_rate[..., :-1, :] - integrated_face_rate[..., 1:, :]
        outer_amount_rate = integrated_face_rate[..., -1, :]
        amount_rate = jnp.where(active_species, amount_rate, 0.0)
        outer_amount_rate = jnp.where(active[..., None], outer_amount_rate, 0.0)

        interior_conductance = (
            face_diffusivity
            * face_measure[..., 1:-1, None]
            / center_distance[..., :, None]
        )
        degree = jnp.zeros_like(amount_rate)
        degree = degree.at[..., :-1, :].add(interior_conductance)
        degree = degree.at[..., 1:, :].add(interior_conductance)
        local_dt_limit = jnp.where(
            degree > 0.0,
            safe_storage[..., :, None] / degree,
            jnp.inf,
        )
        explicit_dt_limit = jnp.min(local_dt_limit, axis=(-2, -1))
        explicit_dt_limit = jnp.where(active, explicit_dt_limit, jnp.inf)

        conservation_defect = jnp.sum(amount_rate, axis=-2) + outer_amount_rate
        tolerance = 128.0 * jnp.finfo(conservation_defect.dtype).eps
        conservation_scale = jnp.maximum(
            jnp.sum(jnp.abs(amount_rate), axis=(-2, -1))
            + jnp.sum(jnp.abs(outer_amount_rate), axis=-1),
            1.0,
        )
        conservation_valid = jnp.isfinite(conservation_scale) & jnp.all(
            jnp.abs(conservation_defect) <= tolerance * conservation_scale[..., None],
            axis=-1,
        )
        outputs_finite = (
            jnp.all(jnp.isfinite(concentrations), axis=(-2, -1))
            & jnp.all(jnp.isfinite(face_molar_flux), axis=(-2, -1))
            & jnp.all(jnp.isfinite(amount_rate), axis=(-2, -1))
            & jnp.all(jnp.isfinite(outer_amount_rate), axis=-1)
            & jnp.all(jnp.isfinite(conservation_defect), axis=-1)
        )
        successful = (
            inputs_valid
            & geometry_valid
            & conservation_valid
            & outputs_finite
            & ~jnp.isnan(explicit_dt_limit)
        )
        return RadialSpeciesTransportResult(
            concentrations,
            face_molar_flux,
            amount_rate,
            outer_amount_rate,
            explicit_dt_limit,
            conservation_defect,
            successful,
        )


def prepare_radial_species_transport(
    plan: RadialSpeciesTransportPlan,
    mesh: PreparedRadialShellMesh,
    /,
) -> PreparedRadialSpeciesTransport:
    """Bind a species layout to one immutable radial shell topology."""
    return PreparedRadialSpeciesTransport(plan, mesh)


__all__ = [
    "PreparedRadialSpeciesTransport",
    "RadialSpeciesTransportPlan",
    "RadialSpeciesTransportResult",
    "prepare_radial_species_transport",
]
