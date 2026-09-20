#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class SlipSpringPlan(StrictModule, NonTrainableState):
    maximum_particles: int = eqx.field(static=True)
    maximum_springs: int = eqx.field(static=True)
    inverse_temperature: float = eqx.field(static=True)
    stiffness: float = eqx.field(static=True)
    chemical_potential: float = eqx.field(static=True)
    maximum_extension: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_particles: int,
        maximum_springs: int,
        inverse_temperature: float,
        stiffness: float,
        chemical_potential: float,
        /,
        *,
        maximum_extension: float = math.inf,
    ):
        if int(maximum_particles) <= 1 or int(maximum_springs) <= 0:
            raise ValueError(
                "Slip-spring particle and spring capacities must be positive."
            )
        if (
            not math.isfinite(float(inverse_temperature))
            or float(inverse_temperature) <= 0.0
        ):
            raise ValueError("inverse_temperature must be finite and positive.")
        if not math.isfinite(float(stiffness)) or float(stiffness) <= 0.0:
            raise ValueError("stiffness must be finite and positive.")
        extension = float(maximum_extension)
        if not (math.isinf(extension) or (math.isfinite(extension) and extension > 0.0)):
            raise ValueError("maximum_extension must be positive or infinity.")
        self.maximum_particles = int(maximum_particles)
        self.maximum_springs = int(maximum_springs)
        self.inverse_temperature = float(inverse_temperature)
        self.stiffness = float(stiffness)
        self.chemical_potential = float(chemical_potential)
        self.maximum_extension = extension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "slip-spring",
                "particles": self.maximum_particles,
                "springs": self.maximum_springs,
                "beta": self.inverse_temperature,
                "stiffness": self.stiffness,
                "chemical_potential": self.chemical_potential,
                "maximum_extension": ("infinity" if math.isinf(extension) else extension),
            }
        )

    def prepare(self, allowed_pairs: ArrayLike, /) -> PreparedSlipSpring:
        return PreparedSlipSpring(self, allowed_pairs)


class SlipSpringState(StrictModule):
    endpoints: Array
    active_mask: Array
    spring_ids: Array
    next_spring_id: Array
    step_index: Array
    key_data: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class SlipSpringEventResult(StrictModule):
    candidate_state: SlipSpringState
    accepted_state: SlipSpringState
    attempted_birth: Array
    accepted: Array
    grand_potential_change: Array
    log_hastings_ratio: Array
    acceptance_probability: Array
    forward_proposal_probability: Array
    reverse_proposal_probability: Array
    spring_energy: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def _event_probabilities(active_count: Array, available_count: Array, capacity: int):
    birth_possible = (available_count > 0) & (active_count < capacity)
    death_possible = active_count > 0
    both = birth_possible & death_possible
    birth_probability = jnp.where(both, 0.5, jnp.where(birth_possible, 1.0, 0.0))
    death_probability = jnp.where(both, 0.5, jnp.where(death_possible, 1.0, 0.0))
    return birth_probability, death_probability


class PreparedSlipSpring(StrictModule, NonTrainableState):
    plan: SlipSpringPlan
    allowed_pairs: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SlipSpringPlan, allowed_pairs: ArrayLike, /):
        if not isinstance(plan, SlipSpringPlan):
            raise TypeError("plan must be SlipSpringPlan.")
        pairs = np.asarray(allowed_pairs, dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0:
            raise ValueError("allowed_pairs must have nonzero shape (pair, 2).")
        pairs = np.sort(pairs, axis=1)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("Slip-spring endpoints must be distinct.")
        if np.any(pairs < 0) or np.any(pairs >= plan.maximum_particles):
            raise ValueError("allowed_pairs exceed the particle capacity.")
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError(
                "allowed_pairs must be unique after endpoint canonicalization."
            )
        self.plan = plan
        self.allowed_pairs = jnp.asarray(pairs)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-slip-spring",
                "plan": plan.plan_id,
                "allowed_pairs": pairs,
            }
        )

    def initialize(
        self,
        key: Key,
        /,
        *,
        initial_pairs: ArrayLike | None = None,
    ) -> SlipSpringState:
        endpoints = np.full((self.plan.maximum_springs, 2), -1, dtype=np.int32)
        active = np.zeros((self.plan.maximum_springs,), dtype=np.bool_)
        spring_ids = np.full((self.plan.maximum_springs,), -1, dtype=np.int32)
        count = 0
        if initial_pairs is not None:
            requested = np.sort(np.asarray(initial_pairs, dtype=np.int32), axis=1)
            if requested.ndim != 2 or requested.shape[1] != 2:
                raise ValueError("initial_pairs must have shape (spring, 2).")
            if requested.shape[0] > self.plan.maximum_springs:
                raise ValueError("initial_pairs exceed maximum_springs.")
            if np.unique(requested, axis=0).shape[0] != requested.shape[0]:
                raise ValueError("initial_pairs must not contain duplicates.")
            allowed = {tuple(pair) for pair in np.asarray(self.allowed_pairs).tolist()}
            if any(tuple(pair) not in allowed for pair in requested.tolist()):
                raise ValueError("Every initial pair must belong to allowed_pairs.")
            count = requested.shape[0]
            endpoints[:count] = requested
            active[:count] = True
            spring_ids[:count] = np.arange(count, dtype=np.int32)
        return SlipSpringState(
            jnp.asarray(endpoints),
            jnp.asarray(active),
            jnp.asarray(spring_ids),
            jnp.asarray(count, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jr.key_data(key),
            jnp.asarray(True),
            self.prepared_id,
        )

    def spring_energy(
        self, state: SlipSpringState, unwrapped_positions: ArrayLike
    ) -> Array:
        positions = jnp.asarray(unwrapped_positions)
        if positions.shape != (self.plan.maximum_particles, 3):
            raise ValueError("unwrapped_positions must match (maximum_particles, 3).")
        safe_endpoints = jnp.where(state.active_mask[:, None], state.endpoints, 0)
        displacement = positions[safe_endpoints[:, 1]] - positions[safe_endpoints[:, 0]]
        energy = 0.5 * self.plan.stiffness * jnp.sum(displacement * displacement, axis=-1)
        return jnp.where(state.active_mask, energy, 0.0)

    def step(
        self, state: SlipSpringState, unwrapped_positions: ArrayLike, /
    ) -> SlipSpringEventResult:
        if (
            not isinstance(state, SlipSpringState)
            or state.prepared_id != self.prepared_id
        ):
            raise ValueError("Slip-spring state does not belong to this prepared plan.")
        positions = jnp.asarray(unwrapped_positions)
        if positions.shape != (self.plan.maximum_particles, 3):
            raise ValueError("unwrapped_positions must match (maximum_particles, 3).")
        key = jr.wrap_key_data(state.key_data)
        choice_key, pair_key, slot_key, accept_key, next_key = jr.split(key, 5)
        endpoints = jnp.sort(state.endpoints, axis=1)
        present = jnp.any(
            jnp.all(self.allowed_pairs[:, None, :] == endpoints[None, :, :], axis=-1)
            & state.active_mask[None, :],
            axis=1,
        )
        available = ~present
        active_count = jnp.sum(state.active_mask.astype(jnp.int32))
        available_count = jnp.sum(available.astype(jnp.int32))
        birth_probability, death_probability = _event_probabilities(
            active_count, available_count, self.plan.maximum_springs
        )
        event_possible = (active_count > 0) | (available_count > 0)
        attempted_birth = jr.uniform(choice_key) < birth_probability

        pair_logits = (
            jnp.where(available, 0.0, -jnp.inf)
            .at[0]
            .set(
                jnp.where(
                    available_count > 0, jnp.where(available[0], 0.0, -jnp.inf), 0.0
                )
            )
        )
        birth_pair_index = jr.categorical(pair_key, pair_logits)
        birth_pair = self.allowed_pairs[birth_pair_index]
        free_mask = ~state.active_mask
        free_logits = (
            jnp.where(free_mask, 0.0, -jnp.inf)
            .at[0]
            .set(
                jnp.where(jnp.any(free_mask), jnp.where(free_mask[0], 0.0, -jnp.inf), 0.0)
            )
        )
        birth_slot = jr.categorical(slot_key, free_logits)
        death_logits = (
            jnp.where(state.active_mask, 0.0, -jnp.inf)
            .at[0]
            .set(
                jnp.where(
                    active_count > 0, jnp.where(state.active_mask[0], 0.0, -jnp.inf), 0.0
                )
            )
        )
        death_slot = jr.categorical(slot_key, death_logits)
        selected_pair = jnp.where(attempted_birth, birth_pair, endpoints[death_slot])
        displacement = positions[selected_pair[1]] - positions[selected_pair[0]]
        spring_energy = 0.5 * self.plan.stiffness * jnp.sum(displacement * displacement)
        within_extension = (
            jnp.sqrt(jnp.sum(displacement * displacement)) <= self.plan.maximum_extension
        )

        post_active_birth = active_count + 1
        post_available_birth = available_count - 1
        _, reverse_death_birth = _event_probabilities(
            post_active_birth, post_available_birth, self.plan.maximum_springs
        )
        post_active_death = active_count - 1
        post_available_death = available_count + 1
        reverse_birth_death, _ = _event_probabilities(
            post_active_death, post_available_death, self.plan.maximum_springs
        )
        tiny = jnp.finfo(positions.dtype).tiny
        birth_forward = birth_probability / jnp.maximum(available_count, 1)
        birth_reverse = reverse_death_birth / jnp.maximum(post_active_birth, 1)
        death_forward = death_probability / jnp.maximum(active_count, 1)
        death_reverse = reverse_birth_death / jnp.maximum(post_available_death, 1)
        forward = jnp.where(attempted_birth, birth_forward, death_forward)
        reverse = jnp.where(attempted_birth, birth_reverse, death_reverse)
        grand_change = jnp.where(
            attempted_birth,
            spring_energy - self.plan.chemical_potential,
            -spring_energy + self.plan.chemical_potential,
        )
        log_hastings = (
            -self.plan.inverse_temperature * grand_change
            + jnp.log(jnp.maximum(reverse, tiny))
            - jnp.log(jnp.maximum(forward, tiny))
        )
        acceptance_probability = jnp.minimum(1.0, jnp.exp(log_hastings))
        accepted = (
            event_possible
            & (forward > 0.0)
            & (~attempted_birth | within_extension)
            & (jr.uniform(accept_key) < acceptance_probability)
        )

        birth_endpoints = state.endpoints.at[birth_slot].set(birth_pair)
        birth_active = state.active_mask.at[birth_slot].set(True)
        birth_ids = state.spring_ids.at[birth_slot].set(state.next_spring_id)
        death_endpoints = state.endpoints.at[death_slot].set(
            jnp.asarray([-1, -1], dtype=state.endpoints.dtype)
        )
        death_active = state.active_mask.at[death_slot].set(False)
        death_ids = state.spring_ids.at[death_slot].set(jnp.asarray(-1, dtype=jnp.int32))
        candidate_endpoints = jnp.where(attempted_birth, birth_endpoints, death_endpoints)
        candidate_active = jnp.where(attempted_birth, birth_active, death_active)
        candidate_ids = jnp.where(attempted_birth, birth_ids, death_ids)
        candidate_next_id = state.next_spring_id + attempted_birth.astype(jnp.int32)
        candidate = SlipSpringState(
            candidate_endpoints,
            candidate_active,
            candidate_ids,
            candidate_next_id,
            state.step_index + 1,
            jr.key_data(next_key),
            state.successful,
            self.prepared_id,
        )
        accepted_state = SlipSpringState(
            jnp.where(accepted, candidate.endpoints, state.endpoints),
            jnp.where(accepted, candidate.active_mask, state.active_mask),
            jnp.where(accepted, candidate.spring_ids, state.spring_ids),
            jnp.where(accepted, candidate.next_spring_id, state.next_spring_id),
            state.step_index + 1,
            jr.key_data(next_key),
            state.successful,
            self.prepared_id,
        )
        successful = (
            state.successful
            & jnp.isfinite(spring_energy)
            & jnp.isfinite(log_hastings)
            & (forward >= 0.0)
            & (reverse >= 0.0)
        )
        return SlipSpringEventResult(
            candidate,
            accepted_state,
            attempted_birth,
            accepted,
            grand_change,
            log_hastings,
            acceptance_probability,
            forward,
            reverse,
            spring_energy,
            successful,
            self.prepared_id,
        )


__all__ = [
    "PreparedSlipSpring",
    "SlipSpringEventResult",
    "SlipSpringPlan",
    "SlipSpringState",
]
