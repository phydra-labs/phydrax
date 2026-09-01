#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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


class DEMLiquidState(StrictModule):
    film_volume: Array
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
    evaporation_flux: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_film_volume: ArrayLike,
        /,
        *,
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
        generated = canonical_fingerprint(
            {
                "kind": "conserved-liquid-bridge-process",
                "initial_film_volume": array_tree_fingerprint(initial),
                "evaporation_flux": flux,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.initial_film_volume = jnp.asarray(initial)
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
        total = jnp.sum(film)
        return DEMLiquidState(film, zero, total, zero, jnp.asarray(True))

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
        total = jnp.sum(film) + bridge_total + cumulative_evaporated
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
    "ConservedLiquidBridgeProcessPlan",
    "DEMLiquidAllocation",
    "DEMLiquidEvaluation",
    "DEMLiquidState",
    "conserved_bagheri_component",
]
