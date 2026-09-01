#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from .._dynamics import (
    AtomisticDynamicsState,
    AtomisticEnergyLedgerState,
    AtomisticForceState,
    PreparedAtomisticDynamics,
)
from ._collective_variable import CollectiveVariableProgram


class BiasKind(StrEnum):
    HARMONIC = "harmonic"
    FLAT_BOTTOM = "flat-bottom"
    LOWER_WALL = "lower-wall"
    UPPER_WALL = "upper-wall"
    MOVING = "moving"
    UMBRELLA = "umbrella"
    METADYNAMICS = "metadynamics"
    ABF = "abf"


_BIASED_CHECKPOINT_FORMAT = "phydrax-biased-atomistic-dynamics-checkpoint"


class AtomisticBiasPlan(StrictModule, NonTrainableState):
    kind: BiasKind = eqx.field(static=True)
    variables: CollectiveVariableProgram
    center: Array
    stiffness: Array
    width: Array
    rate: Array
    maximum_hills: int = eqx.field(static=True)
    grid_minimum: Array
    grid_maximum: Array
    grid_bins: int = eqx.field(static=True)
    bias_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: BiasKind,
        variables: CollectiveVariableProgram,
        /,
        *,
        center: ArrayLike = 0.0,
        stiffness: ArrayLike = 1.0,
        width: ArrayLike = 0.1,
        rate: ArrayLike = 0.0,
        maximum_hills: int = 0,
        grid_minimum: ArrayLike = 0.0,
        grid_maximum: ArrayLike = 1.0,
        grid_bins: int = 64,
    ):
        if not isinstance(kind, BiasKind) or not isinstance(
            variables, CollectiveVariableProgram
        ):
            raise TypeError("Bias requires kind and collective-variable program.")
        bins = int(grid_bins)
        hills = int(maximum_hills)
        if (
            bins <= 1
            or hills < 0
            or (kind is BiasKind.METADYNAMICS and hills == 0)
            or (kind is BiasKind.ABF and len(variables.variables) != 1)
        ):
            raise ValueError("Bias grid, hill capacity, or CV dimensionality is invalid.")
        self.kind = kind
        self.variables = variables
        self.center = jnp.asarray(center, dtype=float).reshape((-1,))
        self.stiffness = jnp.asarray(stiffness, dtype=float).reshape((-1,))
        self.width = jnp.asarray(width, dtype=float).reshape((-1,))
        self.rate = jnp.asarray(rate, dtype=float).reshape((-1,))
        self.maximum_hills = hills
        self.grid_minimum = jnp.asarray(grid_minimum, dtype=float).reshape((-1,))
        self.grid_maximum = jnp.asarray(grid_maximum, dtype=float).reshape((-1,))
        self.grid_bins = bins
        arrays = (
            self.center,
            self.stiffness,
            self.width,
            self.rate,
            self.grid_minimum,
            self.grid_maximum,
        )
        dimension = len(variables.variables)
        if any(value.size not in (1, dimension) for value in arrays[:4]):
            raise ValueError("Bias vectors must be scalar or align with the CV program.")
        if kind is BiasKind.ABF and any(value.size != 1 for value in arrays[4:]):
            raise ValueError("ABF grid bounds must be scalar.")
        if any(bool(jnp.any(~jnp.isfinite(value))) for value in arrays):
            raise ValueError("Bias parameters must be finite.")
        if bool(jnp.any(self.stiffness <= 0.0)) or bool(jnp.any(self.width <= 0.0)):
            raise ValueError("Bias stiffness and widths must be positive.")
        if bool(jnp.any(self.grid_maximum <= self.grid_minimum)):
            raise ValueError("Bias grid maximum must exceed its minimum.")
        self.bias_id = canonical_fingerprint(
            {
                "kind": "atomistic-bias-plan",
                "bias_kind": kind.value,
                "variables": variables.program_id,
                "center": np.asarray(self.center).tolist(),
                "stiffness": np.asarray(self.stiffness).tolist(),
                "width": np.asarray(self.width).tolist(),
                "rate": np.asarray(self.rate).tolist(),
                "maximum_hills": hills,
                "grid_minimum": np.asarray(self.grid_minimum).tolist(),
                "grid_maximum": np.asarray(self.grid_maximum).tolist(),
                "grid_bins": bins,
            }
        )

    def initialize(self, dtype=jnp.float64) -> "AtomisticBiasState":
        dimension = len(self.variables.variables)
        return AtomisticBiasState(
            jnp.zeros((self.maximum_hills, dimension), dtype=dtype),
            jnp.zeros((self.maximum_hills,), dtype=dtype),
            jnp.zeros((self.maximum_hills, dimension), dtype=dtype),
            jnp.zeros((self.maximum_hills,), dtype=bool),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((self.grid_bins,), dtype=jnp.int32),
            jnp.zeros((self.grid_bins,), dtype=dtype),
            jnp.zeros((self.grid_bins,), dtype=dtype),
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(True),
            self.bias_id,
        )


class AtomisticBiasState(StrictModule):
    hill_centers: Array
    hill_heights: Array
    hill_widths: Array
    hill_valid: Array
    hill_count: Array
    abf_counts: Array
    abf_force_sums: Array
    abf_potential: Array
    update_epoch: Array
    successful: Array
    bias_id: str = eqx.field(static=True)


class AtomisticBiasEvaluation(StrictModule):
    energy: Array
    forces: Array
    variables: Array
    successful: Array
    variable_gradients: Array
    state: AtomisticBiasState
    bias_id: str = eqx.field(static=True)


class PreparedAtomisticBias(StrictModule):
    plan: AtomisticBiasPlan
    dynamics: PreparedAtomisticDynamics
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: AtomisticBiasPlan, dynamics: PreparedAtomisticDynamics, /):
        self.plan = plan
        self.dynamics = dynamics
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-bias",
                "plan": plan.bias_id,
                "dynamics": dynamics.prepared_id,
            }
        )

    def energy(self, positions: Array, state: AtomisticBiasState, time: Array, /):
        values, valid = self.plan.variables.evaluate(
            positions, cell=self.dynamics.system.cell
        )
        center = self.plan.center.astype(values.dtype)
        if self.plan.kind is BiasKind.MOVING:
            center = center + self.plan.rate.astype(values.dtype) * time
        delta = values - center
        if self.plan.kind in (BiasKind.HARMONIC, BiasKind.MOVING, BiasKind.UMBRELLA):
            energy = 0.5 * jnp.sum(self.plan.stiffness * delta**2)
        elif self.plan.kind is BiasKind.FLAT_BOTTOM:
            displacement = jnp.sign(delta) * jnp.maximum(
                jnp.abs(delta) - self.plan.width, 0.0
            )
            energy = 0.5 * jnp.sum(self.plan.stiffness * displacement**2)
        elif self.plan.kind is BiasKind.LOWER_WALL:
            energy = 0.5 * jnp.sum(
                self.plan.stiffness * jnp.maximum(center - values, 0.0) ** 2
            )
        elif self.plan.kind is BiasKind.UPPER_WALL:
            energy = 0.5 * jnp.sum(
                self.plan.stiffness * jnp.maximum(values - center, 0.0) ** 2
            )
        elif self.plan.kind is BiasKind.METADYNAMICS:
            difference = values[None, :] - state.hill_centers
            gaussian = state.hill_heights * jnp.exp(
                -0.5
                * jnp.sum(
                    (
                        difference
                        / jnp.where(state.hill_widths > 0.0, state.hill_widths, 1.0)
                    )
                    ** 2,
                    axis=-1,
                )
            )
            energy = jnp.sum(jnp.where(state.hill_valid, gaussian, 0.0))
        else:
            value = values[0]
            minimum, maximum = self.plan.grid_minimum[0], self.plan.grid_maximum[0]
            coordinate = jnp.clip(
                (value - minimum) / (maximum - minimum) * (self.plan.grid_bins - 1),
                0.0,
                self.plan.grid_bins - 1.0,
            )
            left = jnp.floor(coordinate).astype(jnp.int32)
            right = jnp.minimum(left + 1, self.plan.grid_bins - 1)
            fraction = coordinate - left
            energy = (1.0 - fraction) * state.abf_potential[
                left
            ] + fraction * state.abf_potential[right]
        return energy, (values, valid & state.successful)

    def evaluate(
        self, positions: Array, state: AtomisticBiasState, time: Array, /
    ) -> AtomisticBiasEvaluation:
        (energy, auxiliary), gradient = jax.value_and_grad(
            lambda value: self.energy(value, state, time), has_aux=True
        )(positions)
        values, successful = auxiliary
        variable_gradients = (
            jax.jacrev(
                lambda value: self.plan.variables.evaluate(
                    value, cell=self.dynamics.system.cell
                )[0]
            )(positions)
            if self.plan.kind is BiasKind.ABF
            else jnp.zeros((values.shape[0],) + positions.shape, dtype=positions.dtype)
        )
        successful = (
            successful
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(variable_gradients))
        )
        return AtomisticBiasEvaluation(
            energy,
            -gradient,
            values,
            successful,
            variable_gradients,
            state,
            self.prepared_id,
        )

    def update(
        self,
        state: AtomisticBiasState,
        evaluation: AtomisticBiasEvaluation,
        physical_force: Array,
        /,
    ) -> AtomisticBiasState:
        if self.plan.kind is BiasKind.METADYNAMICS:
            available = state.hill_count < self.plan.maximum_hills
            index = jnp.minimum(state.hill_count, max(self.plan.maximum_hills - 1, 0))
            centers = state.hill_centers.at[index].set(evaluation.variables)
            heights = state.hill_heights.at[index].set(
                self.plan.stiffness.reshape((-1,))[0]
            )
            widths = state.hill_widths.at[index].set(
                jnp.broadcast_to(self.plan.width, evaluation.variables.shape)
            )
            valid = state.hill_valid.at[index].set(available)
            return eqx.tree_at(
                lambda item: (
                    item.hill_centers,
                    item.hill_heights,
                    item.hill_widths,
                    item.hill_valid,
                    item.hill_count,
                    item.update_epoch,
                    item.successful,
                ),
                state,
                (
                    centers,
                    heights,
                    widths,
                    valid,
                    state.hill_count + available.astype(jnp.int32),
                    state.update_epoch + 1,
                    state.successful & available,
                ),
            )
        if self.plan.kind is BiasKind.ABF:
            value = evaluation.variables[0]
            minimum, maximum = self.plan.grid_minimum[0], self.plan.grid_maximum[0]
            index = jnp.clip(
                jnp.floor(
                    (value - minimum) / (maximum - minimum) * self.plan.grid_bins
                ).astype(jnp.int32),
                0,
                self.plan.grid_bins - 1,
            )
            gradient = evaluation.variable_gradients[0]
            norm_squared = jnp.sum(gradient * gradient)
            valid = evaluation.successful & (norm_squared > 0.0)
            generalized_force = jnp.sum(gradient * physical_force) / jnp.where(
                norm_squared > 0.0, norm_squared, 1.0
            )
            count = state.abf_counts.at[index].add(valid.astype(jnp.int32))
            force_sum = state.abf_force_sums.at[index].add(
                jnp.where(valid, generalized_force, 0.0)
            )
            average = jnp.where(count > 0, force_sum / count, 0.0)
            spacing = (maximum - minimum) / self.plan.grid_bins
            potential = -jnp.cumsum(average) * spacing
            return eqx.tree_at(
                lambda item: (
                    item.abf_counts,
                    item.abf_force_sums,
                    item.abf_potential,
                    item.update_epoch,
                    item.successful,
                ),
                state,
                (
                    count,
                    force_sum,
                    potential,
                    state.update_epoch + valid.astype(jnp.int32),
                    state.successful & valid,
                ),
            )
        return state


class BiasedDynamicsState(StrictModule):
    base: AtomisticDynamicsState
    bias: AtomisticBiasState
    physical_force: AtomisticForceState
    prepared_id: str = eqx.field(static=True)


class BiasedDynamicsReplayResult(StrictModule):
    final_state: BiasedDynamicsState
    accepted: Array
    prepared_id: str = eqx.field(static=True)


class PreparedBiasedDynamics(StrictModule):
    base: PreparedAtomisticDynamics
    bias: PreparedAtomisticBias
    prepared_id: str = eqx.field(static=True)

    def __init__(self, base: PreparedAtomisticDynamics, bias: PreparedAtomisticBias, /):
        if bias.dynamics.prepared_id != base.prepared_id:
            raise ValueError("Bias belongs to another dynamics runtime.")
        self.base = base
        self.bias = bias
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-biased-dynamics",
                "base": base.prepared_id,
                "bias": bias.prepared_id,
            }
        )

    def initialize(self, state: AtomisticDynamicsState, /) -> BiasedDynamicsState:
        if state.prepared_dynamics_id != self.base.prepared_id:
            raise ValueError("Initial state belongs to another dynamics runtime.")
        bias_state = self.bias.plan.initialize(state.kinematics.positions.dtype)
        evaluation = self.bias.evaluate(
            state.kinematics.positions, bias_state, state.time
        )
        force = self._augment_force(state.force, evaluation)
        ledger = eqx.tree_at(
            lambda item: (
                item.initial_potential_energy,
                item.potential_energy,
                item.total_energy,
            ),
            state.energy,
            (
                state.energy.initial_potential_energy + evaluation.energy,
                state.energy.potential_energy + evaluation.energy,
                state.energy.total_energy + evaluation.energy,
            ),
        )
        base = eqx.tree_at(lambda item: (item.force, item.energy), state, (force, ledger))
        return BiasedDynamicsState(base, bias_state, state.force, self.prepared_id)

    @staticmethod
    def _augment_force(physical: AtomisticForceState, bias, /):
        return AtomisticForceState(
            physical.forces + bias.forces,
            physical.potential_energy + bias.energy,
            physical.term_energies,
            physical.atom_energy,
            physical.virial,
            physical.neighborhood_epoch,
            physical.position_epoch,
            physical.successful & bias.successful,
            physical.program_id,
        )

    def replay(
        self, state: BiasedDynamicsState, step_count: int, /
    ) -> "BiasedDynamicsReplayResult":
        count = int(step_count)
        if state.prepared_id != self.prepared_id or count < 0:
            raise ValueError("Biased replay state or step count is invalid.")
        current = state
        accepted = []
        for _ in range(count):
            next_state = self.step(current)
            accepted.append(next_state.base.step_index > current.base.step_index)
            current = next_state
        return BiasedDynamicsReplayResult(
            current, jnp.asarray(accepted, dtype=bool), self.prepared_id
        )

    def step(self, state: BiasedDynamicsState, /) -> BiasedDynamicsState:
        if state.prepared_id != self.prepared_id:
            raise ValueError("Biased state belongs to another dynamics runtime.")
        current_bias = self.bias.evaluate(
            state.base.kinematics.positions, state.bias, state.base.time
        )
        staged_force = self._augment_force(state.physical_force, current_bias)
        staged = eqx.tree_at(lambda item: item.force, state.base, staged_force)
        physical_step = self.base.step_detailed(staged)
        candidate = physical_step.candidate_state
        physical_force = candidate.force
        next_bias = self.bias.evaluate(
            candidate.kinematics.positions, state.bias, candidate.time
        )
        dt = jnp.asarray(
            self.base.integrator.step_size, dtype=candidate.kinematics.momenta.dtype
        )
        momentum = (
            candidate.kinematics.momenta
            + 0.5
            * dt
            * self.base.system.plan.units.force_to_momentum_rate
            * next_bias.forces
        )
        updated_bias = self.bias.update(state.bias, next_bias, physical_force.forces)
        committed_bias = self.bias.evaluate(
            candidate.kinematics.positions, updated_bias, candidate.time
        )
        kinematics = eqx.tree_at(
            lambda item: item.momenta, candidate.kinematics, momentum
        )
        force = self._augment_force(physical_force, committed_bias)
        kinetic = self.base.kinetic_energy(momentum)
        potential = force.potential_energy
        total = kinetic + potential
        bias_update_work = committed_bias.energy - next_bias.energy
        thermostat_heat = candidate.energy.thermostat_heat
        barostat_work = candidate.energy.barostat_work
        external_work = candidate.energy.external_work + bias_update_work
        constraint_work = candidate.energy.constraint_work
        work_increment = (
            thermostat_heat
            - state.base.energy.thermostat_heat
            + barostat_work
            - state.base.energy.barostat_work
            + external_work
            - state.base.energy.external_work
            + constraint_work
            - state.base.energy.constraint_work
        )
        balance = total - state.base.energy.total_energy - work_increment
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(total), jnp.abs(state.base.energy.total_energy)),
            1.0e-30,
        )
        ledger = AtomisticEnergyLedgerState(
            state.base.energy.initial_kinetic_energy,
            state.base.energy.initial_potential_energy,
            kinetic,
            potential,
            total,
            thermostat_heat,
            barostat_work,
            external_work,
            constraint_work,
            state.base.energy.cumulative_balance_residual + balance,
            jnp.abs(balance) / scale,
            candidate.energy.accepted_steps,
        )
        candidate = eqx.tree_at(
            lambda item: (item.kinematics, item.force, item.energy),
            candidate,
            (kinematics, force, ledger),
        )
        successful = (
            physical_step.successful
            & current_bias.successful
            & next_bias.successful
            & updated_bias.successful
            & committed_bias.successful
            & jnp.all(jnp.isfinite(momentum))
            & jnp.isfinite(total)
        )
        accepted_base = tree_where(successful, candidate, state.base)
        accepted_bias = tree_where(successful, updated_bias, state.bias)
        accepted_physical = tree_where(successful, physical_force, state.physical_force)
        return BiasedDynamicsState(
            accepted_base, accepted_bias, accepted_physical, self.prepared_id
        )


class BiasedDynamicsCheckpointPlan(StrictModule, NonTrainableState):
    dynamics: PreparedBiasedDynamics
    checkpoint_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedBiasedDynamics, /):
        if not isinstance(dynamics, PreparedBiasedDynamics):
            raise TypeError("dynamics must be PreparedBiasedDynamics.")
        self.dynamics = dynamics
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "biased-dynamics-checkpoint-plan",
                "dynamics": dynamics.prepared_id,
                "base": dynamics.base.prepared_id,
                "bias": dynamics.bias.prepared_id,
            }
        )


class BiasedDynamicsCheckpoint(StrictModule):
    state: BiasedDynamicsState
    payload_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


def write_biased_dynamics_checkpoint(
    path: str | Path,
    plan: BiasedDynamicsCheckpointPlan,
    state: BiasedDynamicsState,
    /,
) -> BiasedDynamicsCheckpoint:
    if not isinstance(plan, BiasedDynamicsCheckpointPlan) or not isinstance(
        state, BiasedDynamicsState
    ):
        raise TypeError("Biased checkpoint requires a checkpoint plan and runtime state.")
    if state.prepared_id != plan.dynamics.prepared_id:
        raise ValueError("Biased checkpoint state belongs to another runtime.")
    arrays: dict[str, object] = {}
    specification = pack_array_tree("runtime", state, arrays)
    payload_id = canonical_fingerprint(
        {
            "kind": "biased-dynamics-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "time": float(state.base.time),
            "step": int(state.base.step_index),
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    write_array_archive(
        path,
        manifest={
            "format": _BIASED_CHECKPOINT_FORMAT,
            "kind": "biased-atomistic-dynamics-runtime",
            "checkpoint_id": plan.checkpoint_id,
            "prepared_id": plan.dynamics.prepared_id,
            "base_id": plan.dynamics.base.prepared_id,
            "bias_id": plan.dynamics.bias.prepared_id,
            "state": specification,
            "payload_id": payload_id,
        },
        arrays=arrays,
    )
    return BiasedDynamicsCheckpoint(state, payload_id, plan.checkpoint_id)


def read_biased_dynamics_checkpoint(
    path: str | Path,
    plan: BiasedDynamicsCheckpointPlan,
    template: BiasedDynamicsState,
    /,
) -> BiasedDynamicsCheckpoint:
    if not isinstance(plan, BiasedDynamicsCheckpointPlan) or not isinstance(
        template, BiasedDynamicsState
    ):
        raise TypeError(
            "Biased checkpoint requires a checkpoint plan and state template."
        )
    if template.prepared_id != plan.dynamics.prepared_id:
        raise ValueError("Biased checkpoint template belongs to another runtime.")
    manifest, arrays = read_array_archive(path)
    expected = {
        "format",
        "kind",
        "checkpoint_id",
        "prepared_id",
        "base_id",
        "bias_id",
        "state",
        "payload_id",
        "arrays",
    }
    if set(manifest) != expected:
        raise ValueError(
            "Biased checkpoint manifest is not the canonical current format."
        )
    identities = {
        "format": _BIASED_CHECKPOINT_FORMAT,
        "kind": "biased-atomistic-dynamics-runtime",
        "checkpoint_id": plan.checkpoint_id,
        "prepared_id": plan.dynamics.prepared_id,
        "base_id": plan.dynamics.base.prepared_id,
        "bias_id": plan.dynamics.bias.prepared_id,
    }
    if any(manifest[name] != value for name, value in identities.items()):
        raise ValueError("Biased checkpoint identity does not match the runtime.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    if not isinstance(state, BiasedDynamicsState):
        raise TypeError("Checkpoint did not reconstruct BiasedDynamicsState.")
    payload_id = canonical_fingerprint(
        {
            "kind": "biased-dynamics-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "time": float(state.base.time),
            "step": int(state.base.step_index),
            "state": manifest["state"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if manifest["payload_id"] != payload_id:
        raise ValueError("Biased checkpoint payload identity is corrupt.")
    return BiasedDynamicsCheckpoint(state, payload_id, plan.checkpoint_id)


__all__ = [
    "AtomisticBiasEvaluation",
    "AtomisticBiasPlan",
    "AtomisticBiasState",
    "BiasKind",
    "BiasedDynamicsCheckpoint",
    "BiasedDynamicsCheckpointPlan",
    "BiasedDynamicsReplayResult",
    "BiasedDynamicsState",
    "PreparedAtomisticBias",
    "PreparedBiasedDynamics",
    "read_biased_dynamics_checkpoint",
    "write_biased_dynamics_checkpoint",
]
