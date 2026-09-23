#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import (
    AtomisticDynamicsState,
    PreparedAtomisticDynamics,
    PreparedThermodynamicStateTable,
)
from ...discretization import PairSpringEvaluation
from ._active_polymers import (
    ChromatinState,
    ChromatinStepResult,
    PreparedChromatinDynamics,
)


class ChromatinAtomisticCouplingPlan(StrictModule, NonTrainableState):
    site_particle_ids: Array
    maximum_spring_energy: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_particle_ids: ArrayLike,
        /,
        *,
        maximum_spring_energy: float,
    ):
        identifiers = np.asarray(site_particle_ids)
        maximum = float(maximum_spring_energy)
        if (
            identifiers.ndim != 1
            or identifiers.size < 2
            or not np.issubdtype(identifiers.dtype, np.integer)
            or np.unique(identifiers).size != identifiers.size
            or not math.isfinite(maximum)
            or maximum <= 0.0
        ):
            raise ValueError("Chromatin-atomistic coupling definition is invalid.")
        self.site_particle_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.maximum_spring_energy = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "chromatin-atomistic-coupling-plan",
                "site_particle_ids": identifiers.astype(np.int64).tolist(),
                "maximum_spring_energy": maximum,
            }
        )

    def prepare(
        self,
        atomistic: PreparedAtomisticDynamics,
        chromatin: PreparedChromatinDynamics,
        /,
    ) -> "PreparedChromatinAtomisticCoupling":
        return PreparedChromatinAtomisticCoupling(self, atomistic, chromatin)


class ChromatinAtomisticState(StrictModule):
    atomistic: AtomisticDynamicsState
    chromatin: ChromatinState
    prepared_id: str = eqx.field(static=True)


class ChromatinAtomisticEvidence(StrictModule):
    chromatin_step: ChromatinStepResult
    initial_springs: PairSpringEvaluation
    final_springs: PairSpringEvaluation
    coupling_work: Array
    atomistic_successful: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class ChromatinAtomisticStepResult(StrictModule):
    candidate_state: ChromatinAtomisticState
    accepted_state: ChromatinAtomisticState
    evidence: ChromatinAtomisticEvidence
    successful: Array


class ChromatinAtomisticCheckpoint(StrictModule):
    state: ChromatinAtomisticState
    checkpoint_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedChromatinAtomisticCoupling(StrictModule, NonTrainableState):
    plan: ChromatinAtomisticCouplingPlan
    atomistic: PreparedAtomisticDynamics
    chromatin: PreparedChromatinDynamics
    site_slots: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ChromatinAtomisticCouplingPlan,
        atomistic: PreparedAtomisticDynamics,
        chromatin: PreparedChromatinDynamics,
        /,
    ):
        if not isinstance(plan, ChromatinAtomisticCouplingPlan):
            raise TypeError("plan must be ChromatinAtomisticCouplingPlan.")
        if not isinstance(atomistic, PreparedAtomisticDynamics):
            raise TypeError("atomistic must be PreparedAtomisticDynamics.")
        if not isinstance(chromatin, PreparedChromatinDynamics):
            raise TypeError("chromatin must be PreparedChromatinDynamics.")
        if chromatin.plan.ambient_dimension != 3:
            raise ValueError(
                "Atomistic chromatin coupling requires ambient dimension three."
            )
        if chromatin.plan.site_count != plan.site_particle_ids.size:
            raise ValueError("Chromatin site count and particle mapping differ.")
        particle_ids = np.asarray(atomistic.system.plan.particle_ids, dtype=np.int64)
        active = np.asarray(atomistic.system.active_mask, dtype=np.bool_)
        slot_by_id = {int(value): index for index, value in enumerate(particle_ids)}
        requested = np.asarray(plan.site_particle_ids, dtype=np.int64)
        if any(int(value) not in slot_by_id for value in requested):
            raise ValueError("Chromatin mapping references an absent particle ID.")
        slots = np.asarray(
            [slot_by_id[int(value)] for value in requested], dtype=np.int32
        )
        if not np.all(active[slots]):
            raise ValueError("Chromatin mapping references an inactive particle.")
        self.plan = plan
        self.atomistic = atomistic
        self.chromatin = chromatin
        self.site_slots = jnp.asarray(slots)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-chromatin-atomistic-coupling",
                "plan": plan.plan_id,
                "atomistic": atomistic.prepared_id,
                "chromatin": chromatin.prepared_id,
            }
        )

    def initialize(
        self,
        atomistic: AtomisticDynamicsState,
        chromatin: ChromatinState,
        /,
    ) -> ChromatinAtomisticState:
        if atomistic.prepared_dynamics_id != self.atomistic.prepared_id:
            raise ValueError("Atomistic state belongs to another dynamics runtime.")
        return ChromatinAtomisticState(atomistic, chromatin, self.prepared_id)

    def _site_positions(self, state: AtomisticDynamicsState, /) -> Array:
        return self.atomistic._unwrapped(state.kinematics, state.cell_vectors)[
            self.site_slots
        ]

    def _spring_forces(
        self, chromatin: ChromatinState, site_positions: Array, /
    ) -> tuple[PairSpringEvaluation, Array]:
        evaluation = self.chromatin.springs.evaluate(chromatin.relations, site_positions)
        forces = (
            jnp.zeros((self.atomistic.system.capacity, 3), dtype=site_positions.dtype)
            .at[self.site_slots]
            .add(evaluation.forces)
        )
        return evaluation, forces

    def _kick(
        self, state: AtomisticDynamicsState, forces: Array, half_step: Array, /
    ) -> AtomisticDynamicsState:
        momentum = state.kinematics.momenta + (
            half_step
            * self.atomistic.system.plan.units.force_to_momentum_rate
            * jnp.where(self.atomistic.system.mobile_mask[:, None], forces, 0.0)
        )
        kinematics = eqx.tree_at(lambda value: value.momenta, state.kinematics, momentum)
        return eqx.tree_at(lambda value: value.kinematics, state, kinematics)

    def step(
        self,
        state: ChromatinAtomisticState,
        thermodynamic: PreparedThermodynamicStateTable,
        key: Array,
        /,
    ) -> ChromatinAtomisticStepResult:
        if not isinstance(state, ChromatinAtomisticState):
            raise TypeError("state must be ChromatinAtomisticState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("Coupled state belongs to another prepared runtime.")
        thermodynamic.validate_dynamics(self.atomistic)
        dt = jnp.asarray(
            self.atomistic.integrator.step_size,
            dtype=state.atomistic.kinematics.positions.dtype,
        )
        site_positions = self._site_positions(state.atomistic)
        chromatin_step = self.chromatin.step(state.chromatin, site_positions, key, dt)
        initial_springs, initial_forces = self._spring_forces(
            chromatin_step.accepted_state, site_positions
        )
        pre_kicked = self._kick(state.atomistic, initial_forces, 0.5 * dt)
        atomistic_step = self.atomistic.step_detailed(pre_kicked, thermodynamic)
        candidate_atomistic = atomistic_step.accepted_state
        final_positions = self._site_positions(candidate_atomistic)
        final_springs, final_forces = self._spring_forces(
            chromatin_step.accepted_state, final_positions
        )
        post_kicked = self._kick(candidate_atomistic, final_forces, 0.5 * dt)
        coupling_work = final_springs.energy - initial_springs.energy
        successful = (
            chromatin_step.successful
            & atomistic_step.successful
            & initial_springs.successful
            & final_springs.successful
            & jnp.isfinite(coupling_work)
            & (initial_springs.energy <= self.plan.maximum_spring_energy)
            & (final_springs.energy <= self.plan.maximum_spring_energy)
        )
        candidate = ChromatinAtomisticState(
            post_kicked, chromatin_step.accepted_state, self.prepared_id
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        evidence = ChromatinAtomisticEvidence(
            chromatin_step,
            initial_springs,
            final_springs,
            coupling_work,
            atomistic_step.successful,
            successful,
            self.prepared_id,
        )
        return ChromatinAtomisticStepResult(candidate, accepted, evidence, successful)

    def _validate_state(self, state: ChromatinAtomisticState, /) -> None:
        if not isinstance(state, ChromatinAtomisticState):
            raise TypeError("state must be ChromatinAtomisticState.")
        if (
            state.prepared_id != self.prepared_id
            or state.atomistic.prepared_dynamics_id != self.atomistic.prepared_id
            or state.atomistic.force.program_id != self.atomistic.potential.prepared_id
        ):
            raise ValueError("Coupled state belongs to another prepared runtime.")
        template = self.chromatin.initialize()
        if jax.tree.structure(state.chromatin) != jax.tree.structure(template):
            raise ValueError(
                "Chromatin state structure does not match the prepared runtime."
            )
        supplied = jax.tree.leaves(state.chromatin)
        expected = jax.tree.leaves(template)
        if any(
            np.asarray(value).shape != np.asarray(reference).shape
            for value, reference in zip(supplied, expected, strict=True)
        ):
            raise ValueError("Chromatin state shapes do not match the prepared runtime.")
        arrays = []
        for leaf in jax.tree.leaves(state):
            try:
                arrays.append(np.asarray(leaf))
            except (TypeError, ValueError):
                if leaf is not state.atomistic.random_key:
                    raise
        if any(
            np.issubdtype(value.dtype, np.inexact) and np.any(~np.isfinite(value))
            for value in arrays
        ):
            raise ValueError("Coupled checkpoint state contains nonfinite values.")
        if (
            int(np.asarray(state.atomistic.step_index)) < 0
            or int(np.asarray(state.chromatin.step_index)) < 0
            or float(np.asarray(state.chromatin.time)) < 0.0
        ):
            raise ValueError("Coupled checkpoint clocks must be nonnegative.")

    def _checkpoint_id(self, state: ChromatinAtomisticState, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "chromatin-atomistic-checkpoint",
                "prepared": self.prepared_id,
                "thermodynamic_table": state.atomistic.thermodynamic_table_id,
                "force_program": state.atomistic.force.program_id,
                "random_key": array_tree_fingerprint(
                    jax.random.key_data(state.atomistic.random_key)
                ),
                "state": array_tree_fingerprint(state),
            }
        )

    def checkpoint(
        self, state: ChromatinAtomisticState, /
    ) -> ChromatinAtomisticCheckpoint:
        self._validate_state(state)
        identifier = self._checkpoint_id(state)
        return ChromatinAtomisticCheckpoint(state, identifier, self.prepared_id)

    def restore(
        self, checkpoint: ChromatinAtomisticCheckpoint, /
    ) -> ChromatinAtomisticState:
        if not isinstance(checkpoint, ChromatinAtomisticCheckpoint):
            raise TypeError("checkpoint must be a ChromatinAtomisticCheckpoint.")
        if checkpoint.prepared_id != self.prepared_id:
            raise ValueError("Checkpoint belongs to another coupling runtime.")
        self._validate_state(checkpoint.state)
        if checkpoint.checkpoint_id != self._checkpoint_id(checkpoint.state):
            raise ValueError("Checkpoint content identity does not match its state.")
        return checkpoint.state


__all__ = [
    "ChromatinAtomisticCheckpoint",
    "ChromatinAtomisticCouplingPlan",
    "ChromatinAtomisticEvidence",
    "ChromatinAtomisticState",
    "ChromatinAtomisticStepResult",
    "PreparedChromatinAtomisticCoupling",
]
