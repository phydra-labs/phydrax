#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem_cohesion import (
    BagheriCapillaryBridgePlan,
    CompositeDEMCohesionPlan,
    DEMCohesionComponentHistory,
)


DEMBarrierGeometryPolicy: TypeAlias = Literal["planar", "isotropic_curvature"]
DEMBarrierCapillaryLaw: TypeAlias = Literal["linear", "bagheri"]


class DEMBarrierCapillaryPlan(StrictModule, NonTrainableState):
    """Certified sphere-surface capillary binding and liquid endpoint split."""

    barrier_id: str = eqx.field(static=True)
    geometry_policy: DEMBarrierGeometryPolicy = eqx.field(static=True)
    law: DEMBarrierCapillaryLaw = eqx.field(static=True)
    particle_liquid_fraction: float = eqx.field(static=True)
    initial_barrier_film_volume: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        barrier_id: str,
        *,
        geometry_policy: DEMBarrierGeometryPolicy,
        particle_liquid_fraction: float,
        initial_barrier_film_volume: float,
        law: DEMBarrierCapillaryLaw = "bagheri",
    ):
        identifier = str(barrier_id)
        if not identifier:
            raise ValueError("barrier_id must be nonempty.")
        if geometry_policy not in ("planar", "isotropic_curvature"):
            raise ValueError("geometry_policy must be 'planar' or 'isotropic_curvature'.")
        if law not in ("linear", "bagheri"):
            raise ValueError("law must be 'linear' or 'bagheri'.")
        fraction = float(particle_liquid_fraction)
        reservoir = float(initial_barrier_film_volume)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError("particle_liquid_fraction must lie in [0, 1].")
        if not np.isfinite(reservoir) or reservoir < 0.0:
            raise ValueError(
                "initial_barrier_film_volume must be finite and nonnegative."
            )
        self.barrier_id = identifier
        self.geometry_policy = geometry_policy
        self.law = law
        self.particle_liquid_fraction = fraction
        self.initial_barrier_film_volume = reservoir
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dem-barrier-capillary",
                "barrier_id": identifier,
                "geometry_policy": geometry_policy,
                "law": law,
                "particle_fraction": fraction,
                "initial_reservoir": reservoir,
            }
        )

    def effective_radius(
        self,
        particle_radius: ArrayLike,
        wall_curvature: ArrayLike,
        isotropy_defect: ArrayLike,
        curvature_valid: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-12,
    ) -> tuple[Array, Array, Array]:
        radius = jnp.asarray(particle_radius)
        curvature = jnp.asarray(wall_curvature, dtype=radius.dtype)
        defect = jnp.asarray(isotropy_defect, dtype=radius.dtype)
        certified = jnp.asarray(curvature_valid, dtype=bool)
        if curvature.shape != radius.shape or defect.shape != radius.shape:
            raise ValueError("Barrier curvature arrays must match particle radii.")
        if self.geometry_policy == "planar":
            certified = certified & (jnp.abs(curvature) <= tolerance)
            curvature = jnp.zeros_like(curvature)
        else:
            certified = certified & (defect <= tolerance)
        denominator = 1.0 / radius + curvature
        valid = (
            certified
            & jnp.isfinite(radius)
            & (radius > 0.0)
            & jnp.isfinite(denominator)
            & (denominator > tolerance)
        )
        effective = jnp.where(valid, 1.0 / denominator, 0.0)
        margin = jnp.minimum(denominator, tolerance - defect)
        if self.geometry_policy == "planar":
            margin = jnp.minimum(denominator, tolerance - jnp.abs(curvature))
        return effective, margin, valid


class DEMBarrierLiquidAllocation(StrictModule):
    bridge_volume: Array
    particle_withdrawal: Array
    barrier_withdrawal: Array
    film_volume: Array
    barrier_reservoir_volume: Array
    successful: Array


class DEMBarrierLiquidEvaluation(StrictModule):
    allocated_bridge_volume: Array
    released_bridge_volume: Array
    evaporated_bridge_volume: Array
    evaporated_ruptures: Array
    bridge_volume: Array
    particle_return: Array
    barrier_return: Array
    next_state: DEMLiquidState
    successful: Array


class DEMLiquidState(StrictModule):
    film_volume: Array
    barrier_reservoir_volume: Array
    cumulative_evaporated_volume: Array
    initial_total_volume: Array
    balance_residual: Array
    successful: Array


class DEMLiquidAllocation(StrictModule):
    bridge_volume: Array
    film_volume: Array
    successful: Array


class DEMLiquidEvaluation(StrictModule):
    allocated_bridge_volume: Array
    released_bridge_volume: Array
    evaporated_bridge_volume: Array
    evaporated_ruptures: Array
    bridge_volume: Array
    next_state: DEMLiquidState
    successful: Array


class ConservedLiquidBridgeProcessPlan(StrictModule, NonTrainableState):
    initial_film_volume: Array
    barrier_capillaries: tuple[DEMBarrierCapillaryPlan, ...]
    evaporation_flux: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_film_volume: ArrayLike,
        /,
        *,
        barrier_capillaries: Sequence[DEMBarrierCapillaryPlan] = (),
        evaporation_flux: float = 0.0,
        plan_id: str | None = None,
    ):
        initial = np.asarray(initial_film_volume)
        flux = float(evaporation_flux)
        if initial.ndim not in (0, 1):
            raise ValueError(
                "initial_film_volume must be scalar or particle-capacity shaped."
            )
        if np.any(~np.isfinite(initial)) or np.any(initial < 0.0):
            raise ValueError("initial_film_volume must be finite and nonnegative.")
        if not np.isfinite(flux) or flux < 0.0:
            raise ValueError("evaporation_flux must be finite and nonnegative.")
        barriers = tuple(barrier_capillaries)
        if any(not isinstance(value, DEMBarrierCapillaryPlan) for value in barriers):
            raise TypeError(
                "barrier_capillaries must contain DEMBarrierCapillaryPlan values."
            )
        barrier_ids = tuple(value.barrier_id for value in barriers)
        if len(set(barrier_ids)) != len(barrier_ids):
            raise ValueError("Barrier capillary IDs must be unique.")
        generated = canonical_fingerprint(
            {
                "kind": "conserved-liquid-bridge-process",
                "initial_film_volume": array_tree_fingerprint(initial),
                "evaporation_flux": flux,
                "barrier_capillaries": [value.plan_id for value in barriers],
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.initial_film_volume = jnp.asarray(initial)
        self.barrier_capillaries = barriers
        self.evaporation_flux = flux
        self.plan_id = identifier

    def initialize(
        self, capacity: int, dtype, active_mask: ArrayLike, /
    ) -> DEMLiquidState:
        count = int(capacity)
        active = jnp.asarray(active_mask, dtype=bool)
        if active.shape != (count,):
            raise ValueError("Liquid active mask must have particle-capacity shape.")
        if self.initial_film_volume.ndim == 0:
            film = jnp.full((count,), self.initial_film_volume, dtype=dtype)
        elif self.initial_film_volume.shape == (count,):
            film = self.initial_film_volume.astype(dtype)
        else:
            raise ValueError("initial_film_volume does not match particle capacity.")
        film = jnp.where(active, film, 0.0)
        zero = jnp.zeros((), dtype=dtype)
        reservoirs = jnp.asarray(
            tuple(
                value.initial_barrier_film_volume for value in self.barrier_capillaries
            ),
            dtype=dtype,
        )
        total = jnp.sum(film) + jnp.sum(reservoirs)
        return DEMLiquidState(film, reservoirs, zero, total, zero, jnp.asarray(True))

    def allocate(
        self,
        state: DEMLiquidState,
        left: Array,
        right: Array,
        requested_volume: Array,
        minimum_volume: Array,
        birth_candidates: Array,
        particle_capacity: int,
        /,
    ) -> DEMLiquidAllocation:
        if not isinstance(state, DEMLiquidState):
            raise TypeError("state must be DEMLiquidState.")
        request = jnp.where(birth_candidates, requested_volume, 0.0)
        half_request = 0.5 * request
        requested_by_particle = (
            jnp.zeros((particle_capacity,), dtype=request.dtype)
            .at[left]
            .add(half_request)
        )
        requested_by_particle = requested_by_particle.at[right].add(half_request)
        scale = jnp.where(
            requested_by_particle > 0.0,
            jnp.minimum(state.film_volume / requested_by_particle, 1.0),
            1.0,
        )
        allocated = request * jnp.minimum(scale[left], scale[right])
        allocated = jnp.where(allocated >= minimum_volume, allocated, 0.0)
        contribution = 0.5 * allocated
        withdrawal = (
            jnp.zeros((particle_capacity,), dtype=request.dtype)
            .at[left]
            .add(contribution)
        )
        withdrawal = withdrawal.at[right].add(contribution)
        film = state.film_volume - withdrawal
        tolerance = (
            64.0
            * jnp.finfo(film.dtype).eps
            * jnp.maximum(jnp.max(state.film_volume, initial=0.0), 1.0)
        )
        successful = (
            state.successful
            & jnp.all(jnp.isfinite(allocated))
            & jnp.all(jnp.isfinite(film))
            & jnp.all(film >= -tolerance)
        )
        return DEMLiquidAllocation(allocated, jnp.maximum(film, 0.0), successful)

    def advance(
        self,
        state: DEMLiquidState,
        allocation: DEMLiquidAllocation,
        component: DEMCohesionComponentHistory,
        left: Array,
        right: Array,
        released_volume: Array,
        surface_area: Array,
        minimum_volume: Array,
        step_size: ArrayLike,
        particle_capacity: int,
        /,
        *,
        additional_bridge_volume: ArrayLike = 0.0,
    ) -> tuple[DEMCohesionComponentHistory, DEMLiquidEvaluation]:
        if not isinstance(component, DEMCohesionComponentHistory):
            raise TypeError("component must be DEMCohesionComponentHistory.")
        release = jnp.where(released_volume > 0.0, released_volume, 0.0)
        returned = (
            jnp.zeros((particle_capacity,), dtype=release.dtype)
            .at[left]
            .add(0.5 * release)
        )
        returned = returned.at[right].add(0.5 * release)
        film = allocation.film_volume + returned
        dt = jnp.maximum(jnp.asarray(step_size, dtype=release.dtype), 0.0)
        raw_loss = self.evaporation_flux * surface_area * dt
        loss = jnp.where(
            component.active,
            jnp.minimum(raw_loss, component.bridge_volume),
            0.0,
        )
        remaining = component.bridge_volume - loss
        evaporated_rupture = (
            component.active & (remaining < minimum_volume) & (loss > 0.0)
        )
        loss = jnp.where(evaporated_rupture, component.bridge_volume, loss)
        remaining = jnp.where(evaporated_rupture, 0.0, remaining)
        active = component.active & ~evaporated_rupture
        next_component = DEMCohesionComponentHistory(
            active,
            jnp.where(active, remaining, 0.0),
            component.previous_gap,
            jnp.where(active, component.birth_step, -1),
        )
        cumulative_evaporated = state.cumulative_evaporated_volume + jnp.sum(loss)
        bridge_total = jnp.sum(next_component.bridge_volume)
        total = (
            jnp.sum(film)
            + jnp.sum(state.barrier_reservoir_volume)
            + bridge_total
            + jnp.sum(jnp.asarray(additional_bridge_volume, dtype=film.dtype))
            + cumulative_evaporated
        )
        residual = total - state.initial_total_volume
        scale = jnp.maximum(state.initial_total_volume, 1.0)
        tolerance = 128.0 * jnp.finfo(film.dtype).eps * scale
        successful = (
            state.successful
            & allocation.successful
            & jnp.all(jnp.isfinite(film))
            & jnp.all(jnp.isfinite(next_component.bridge_volume))
            & jnp.all(film >= 0.0)
            & jnp.all(next_component.bridge_volume >= 0.0)
            & jnp.isfinite(residual)
            & (jnp.abs(residual) <= tolerance)
        )
        next_state = DEMLiquidState(
            film,
            state.barrier_reservoir_volume,
            cumulative_evaporated,
            state.initial_total_volume,
            residual,
            successful,
        )
        evaluation = DEMLiquidEvaluation(
            allocation.bridge_volume,
            release,
            loss,
            evaporated_rupture,
            next_component.bridge_volume,
            next_state,
            successful,
        )
        return next_component, evaluation

    def allocate_barriers(
        self,
        state: DEMLiquidState,
        particle_indices: Array,
        barrier_indices: Array,
        requested_volume: Array,
        minimum_volume: Array,
        birth_candidates: Array,
        particle_capacity: int,
        /,
    ) -> DEMBarrierLiquidAllocation:
        """Allocate simultaneous particle-wall births without ordering bias."""

        if not isinstance(state, DEMLiquidState):
            raise TypeError("state must be DEMLiquidState.")
        width = int(requested_volume.shape[0])
        expected = (width,)
        if (
            particle_indices.shape != expected
            or barrier_indices.shape != expected
            or minimum_volume.shape != expected
            or birth_candidates.shape != expected
        ):
            raise ValueError("Barrier liquid route arrays must share one shape.")
        if state.barrier_reservoir_volume.shape != (len(self.barrier_capillaries),):
            raise ValueError("Liquid state has the wrong barrier reservoir shape.")
        fractions = jnp.asarray(
            tuple(value.particle_liquid_fraction for value in self.barrier_capillaries),
            dtype=requested_volume.dtype,
        )
        safe_barrier = jnp.maximum(barrier_indices, 0)
        safe_particle = jnp.maximum(particle_indices, 0)
        route_valid = (
            birth_candidates
            & (particle_indices >= 0)
            & (particle_indices < particle_capacity)
            & (barrier_indices >= 0)
            & (barrier_indices < len(self.barrier_capillaries))
        )
        request = jnp.where(route_valid, requested_volume, 0.0)
        particle_share = request * fractions[safe_barrier]
        barrier_share = request - particle_share
        particle_demand = (
            jnp.zeros((particle_capacity,), dtype=request.dtype)
            .at[safe_particle]
            .add(particle_share)
        )
        barrier_demand = (
            jnp.zeros((len(self.barrier_capillaries),), dtype=request.dtype)
            .at[safe_barrier]
            .add(barrier_share)
        )
        particle_scale = jnp.where(
            particle_demand > 0.0,
            jnp.minimum(state.film_volume / particle_demand, 1.0),
            1.0,
        )
        barrier_scale = jnp.where(
            barrier_demand > 0.0,
            jnp.minimum(
                state.barrier_reservoir_volume / barrier_demand,
                1.0,
            ),
            1.0,
        )
        allocation_scale = jnp.minimum(
            particle_scale[safe_particle], barrier_scale[safe_barrier]
        )
        allocated = request * allocation_scale
        allocated = jnp.where(allocated >= minimum_volume, allocated, 0.0)
        allocated_particle = allocated * fractions[safe_barrier]
        allocated_barrier = allocated - allocated_particle
        particle_withdrawal = (
            jnp.zeros((particle_capacity,), dtype=request.dtype)
            .at[safe_particle]
            .add(allocated_particle)
        )
        barrier_withdrawal = (
            jnp.zeros((len(self.barrier_capillaries),), dtype=request.dtype)
            .at[safe_barrier]
            .add(allocated_barrier)
        )
        film = state.film_volume - particle_withdrawal
        reservoir = state.barrier_reservoir_volume - barrier_withdrawal
        tolerance = (
            128.0
            * jnp.finfo(request.dtype).eps
            * jnp.maximum(state.initial_total_volume, 1.0)
        )
        successful = (
            state.successful
            & jnp.all(jnp.isfinite(allocated))
            & jnp.all(jnp.isfinite(film))
            & jnp.all(jnp.isfinite(reservoir))
            & jnp.all(film >= -tolerance)
            & jnp.all(reservoir >= -tolerance)
        )
        return DEMBarrierLiquidAllocation(
            allocated,
            particle_withdrawal,
            barrier_withdrawal,
            jnp.maximum(film, 0.0),
            jnp.maximum(reservoir, 0.0),
            successful,
        )

    def advance_barriers(
        self,
        state: DEMLiquidState,
        allocation: DEMBarrierLiquidAllocation,
        particle_indices: Array,
        barrier_indices: Array,
        bridge_volume: Array,
        released_volume: Array,
        surface_area: Array,
        minimum_volume: Array,
        step_size: ArrayLike,
        particle_capacity: int,
        /,
        *,
        other_bridge_volume: ArrayLike = 0.0,
    ) -> DEMBarrierLiquidEvaluation:
        """Commit returns/evaporation and certify the full liquid inventory."""

        if not isinstance(allocation, DEMBarrierLiquidAllocation):
            raise TypeError("allocation must be DEMBarrierLiquidAllocation.")
        width = int(bridge_volume.shape[0])
        expected = (width,)
        if (
            particle_indices.shape != expected
            or barrier_indices.shape != expected
            or released_volume.shape != expected
            or surface_area.shape != expected
            or minimum_volume.shape != expected
        ):
            raise ValueError("Barrier liquid evaluation arrays must share one shape.")
        safe_particle = jnp.maximum(particle_indices, 0)
        safe_barrier = jnp.maximum(barrier_indices, 0)
        fractions = jnp.asarray(
            tuple(value.particle_liquid_fraction for value in self.barrier_capillaries),
            dtype=bridge_volume.dtype,
        )
        release = jnp.maximum(released_volume, 0.0)
        particle_return_route = release * fractions[safe_barrier]
        barrier_return_route = release - particle_return_route
        particle_return = (
            jnp.zeros((particle_capacity,), dtype=bridge_volume.dtype)
            .at[safe_particle]
            .add(particle_return_route)
        )
        barrier_return = (
            jnp.zeros((len(self.barrier_capillaries),), dtype=bridge_volume.dtype)
            .at[safe_barrier]
            .add(barrier_return_route)
        )
        film = allocation.film_volume + particle_return
        reservoir = allocation.barrier_reservoir_volume + barrier_return
        dt = jnp.maximum(jnp.asarray(step_size, dtype=bridge_volume.dtype), 0.0)
        loss = jnp.minimum(
            self.evaporation_flux * surface_area * dt,
            jnp.maximum(bridge_volume, 0.0),
        )
        remaining = jnp.maximum(bridge_volume - loss, 0.0)
        evaporated_ruptures = (remaining < minimum_volume) & (loss > 0.0)
        loss = jnp.where(evaporated_ruptures, bridge_volume, loss)
        remaining = jnp.where(evaporated_ruptures, 0.0, remaining)
        cumulative = state.cumulative_evaporated_volume + jnp.sum(loss)
        total = (
            jnp.sum(film)
            + jnp.sum(reservoir)
            + jnp.sum(remaining)
            + jnp.sum(jnp.asarray(other_bridge_volume))
            + cumulative
        )
        residual = total - state.initial_total_volume
        tolerance = (
            256.0
            * jnp.finfo(bridge_volume.dtype).eps
            * jnp.maximum(state.initial_total_volume, 1.0)
        )
        successful = (
            state.successful
            & allocation.successful
            & jnp.all(jnp.isfinite(film))
            & jnp.all(jnp.isfinite(reservoir))
            & jnp.all(jnp.isfinite(remaining))
            & jnp.all(film >= 0.0)
            & jnp.all(reservoir >= 0.0)
            & jnp.all(remaining >= 0.0)
            & jnp.isfinite(residual)
            & (jnp.abs(residual) <= tolerance)
        )
        next_state = DEMLiquidState(
            film,
            reservoir,
            cumulative,
            state.initial_total_volume,
            residual,
            successful,
        )
        return DEMBarrierLiquidEvaluation(
            allocation.bridge_volume,
            release,
            loss,
            evaporated_ruptures,
            remaining,
            particle_return,
            barrier_return,
            next_state,
            successful,
        )


def conserved_bagheri_component(cohesion, /) -> tuple[BagheriCapillaryBridgePlan, int]:
    if isinstance(cohesion, BagheriCapillaryBridgePlan):
        if not cohesion.conserve_liquid:
            raise ValueError("Liquid process requires conserve_liquid=True.")
        return cohesion, 0
    if isinstance(cohesion, CompositeDEMCohesionPlan):
        matches = tuple(
            (component, index)
            for index, component in enumerate(cohesion.components)
            if isinstance(component, BagheriCapillaryBridgePlan)
            and component.conserve_liquid
        )
        if len(matches) != 1:
            raise ValueError(
                "Liquid process requires exactly one conserved Bagheri component."
            )
        return matches[0]
    raise ValueError("Liquid process requires a conserved Bagheri cohesion law.")


__all__ = [
    "DEMBarrierCapillaryLaw",
    "DEMBarrierCapillaryPlan",
    "DEMBarrierGeometryPolicy",
    "DEMBarrierLiquidAllocation",
    "DEMBarrierLiquidEvaluation",
    "ConservedLiquidBridgeProcessPlan",
    "DEMLiquidAllocation",
    "DEMLiquidEvaluation",
    "DEMLiquidState",
    "conserved_bagheri_component",
]
