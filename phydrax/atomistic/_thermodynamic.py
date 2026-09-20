#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._system import PreparedAtomisticSystem


AtomisticEnsembleKind = Literal["nve", "nvt", "npt"]
_ENSEMBLE_CODES = {"nve": 0, "nvt": 1, "npt": 2}


class AtomisticPhaseSpaceMeasurePlan(StrictModule, NonTrainableState):
    """Identity of the common coordinate, momentum, topology, and unit measure."""

    particle_capacity: int = eqx.field(static=True)
    scaled_entity_count: int = eqx.field(static=True)
    volume_coordinate_convention: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    coordinate_map_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        scaled_entity_count: int | None = None,
        volume_coordinate_convention: str = "molecular-center",
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be a PreparedAtomisticSystem.")
        entities = (
            len(system.molecule_labels)
            if scaled_entity_count is None
            else int(scaled_entity_count)
        )
        convention = str(volume_coordinate_convention)
        if entities <= 0 or convention != "molecular-center":
            raise ValueError(
                "Phase-space volume scaling requires a positive entity count "
                "and the supported 'molecular-center' convention."
            )
        self.particle_capacity = system.capacity
        self.scaled_entity_count = entities
        self.volume_coordinate_convention = convention
        self.system_id = system.prepared_id
        self.topology_id = system.topology.topology_id
        self.unit_system_id = system.plan.units.unit_system_id
        self.coordinate_map_id = system.coordinate_map.prepared_id
        self.measure_id = canonical_fingerprint(
            {
                "kind": "atomistic-phase-space-measure",
                "system": self.system_id,
                "topology": self.topology_id,
                "units": self.unit_system_id,
                "coordinate_map": self.coordinate_map_id,
                "particle_capacity": self.particle_capacity,
                "scaled_entity_count": entities,
                "volume_coordinate_convention": convention,
                "particle_ids": array_tree_fingerprint(system.plan.particle_ids),
                "masses": array_tree_fingerprint(system.plan.masses),
                "active": array_tree_fingerprint(system.active_mask),
                "mobile": array_tree_fingerprint(system.mobile_mask),
            }
        )


class AtomisticThermodynamicStatePlan(StrictModule, NonTrainableState):
    """One physical ensemble state on a declared common phase-space measure."""

    phase_space: AtomisticPhaseSpaceMeasurePlan
    controls: Array
    ensemble: AtomisticEnsembleKind = eqx.field(static=True)
    temperature: float | None = eqx.field(static=True)
    pressure: float | None = eqx.field(static=True)
    control_ids: tuple[str, ...] = eqx.field(static=True)
    bias_id: str | None = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        phase_space: AtomisticPhaseSpaceMeasurePlan,
        /,
        *,
        ensemble: AtomisticEnsembleKind,
        temperature: float | None = None,
        pressure: float | None = None,
        controls: ArrayLike | None = None,
        control_ids: tuple[str, ...] = (),
        bias_id: str | None = None,
        state_id: str | None = None,
    ):
        if not isinstance(phase_space, AtomisticPhaseSpaceMeasurePlan):
            raise TypeError("phase_space must be an AtomisticPhaseSpaceMeasurePlan.")
        if ensemble not in _ENSEMBLE_CODES:
            raise ValueError("ensemble must be 'nve', 'nvt', or 'npt'.")
        temperature_ = None if temperature is None else float(temperature)
        pressure_ = None if pressure is None else float(pressure)
        ids = tuple(control_ids)
        values = (
            jnp.zeros((0,), dtype=jnp.float64)
            if controls is None
            else jnp.asarray(controls, dtype=jnp.float64).reshape((-1,))
        )
        if (
            len(ids) != values.size
            or len(set(ids)) != len(ids)
            or any(
                not isinstance(name, str) or not name or name != name.strip()
                for name in ids
            )
            or bool(jnp.any(~jnp.isfinite(values)))
        ):
            raise ValueError("Thermodynamic control IDs and values are invalid.")
        if ensemble == "nve":
            valid_intensives = temperature_ is None and pressure_ is None
        elif ensemble == "nvt":
            valid_intensives = (
                temperature_ is not None
                and np.isfinite(temperature_)
                and temperature_ > 0.0
                and pressure_ is None
            )
        else:
            valid_intensives = (
                temperature_ is not None
                and np.isfinite(temperature_)
                and temperature_ > 0.0
                and pressure_ is not None
                and np.isfinite(pressure_)
            )
        if not valid_intensives:
            raise ValueError(
                "Thermodynamic intensive variables do not match the ensemble."
            )
        if bias_id is not None and (
            not isinstance(bias_id, str) or not bias_id or bias_id != bias_id.strip()
        ):
            raise ValueError("bias_id must be a canonical nonempty string or None.")
        generated = canonical_fingerprint(
            {
                "kind": "atomistic-thermodynamic-state",
                "phase_space": phase_space.measure_id,
                "ensemble": ensemble,
                "temperature": temperature_,
                "pressure": pressure_,
                "control_ids": list(ids),
                "controls": array_tree_fingerprint(values),
                "bias": bias_id,
            }
        )
        identifier = generated if state_id is None else str(state_id)
        if not identifier or identifier != identifier.strip():
            raise ValueError("state_id must be a canonical nonempty string.")
        self.phase_space = phase_space
        self.controls = values
        self.ensemble = ensemble
        self.temperature = temperature_
        self.pressure = pressure_
        self.control_ids = ids
        self.bias_id = bias_id
        self.state_id = identifier

    def prepare(self, dynamics: Any, /) -> "PreparedThermodynamicStateTable":
        return PreparedThermodynamicStateTable(dynamics, (self,))


class _ThermodynamicStateRows(StrictModule):
    state_index: Array
    beta: Array
    temperature: Array
    temperature_mask: Array
    pressure: Array
    hamiltonian_index: Array
    pressure_mask: Array
    controls: Array
    ensemble_code: Array
    valid: Array


class PreparedThermodynamicStateTable(StrictModule, NonTrainableState):
    """Fixed-shape numerical thermodynamic states bound to one dynamics runtime."""

    beta: Array
    temperature: Array
    temperature_mask: Array
    pressure: Array
    pressure_mask: Array
    controls: Array
    ensemble_code: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    control_ids: tuple[str, ...] = eqx.field(static=True)
    bias_ids: tuple[str | None, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    hamiltonian_index: Array
    state_count: int = eqx.field(static=True)
    control_count: int = eqx.field(static=True)
    phase_space_measure_id: str = eqx.field(static=True)
    declared_phase_space_measure_id: str = eqx.field(static=True)
    scaled_entity_count: int = eqx.field(static=True)
    volume_coordinate_convention: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    control_layout_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    constraint_manifold_id: str | None = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    requires_cross_evaluation: bool = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(self, dynamics: Any, states, /):
        from ._dynamics import PreparedAtomisticDynamics

        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        plans = tuple(states)
        if not plans or any(
            not isinstance(state, AtomisticThermodynamicStatePlan) for state in plans
        ):
            raise TypeError("states must contain AtomisticThermodynamicStatePlan values.")
        measure_id = plans[0].phase_space.measure_id
        measure = plans[0].phase_space
        control_ids = plans[0].control_ids
        if any(state.phase_space.measure_id != measure_id for state in plans):
            raise ValueError(
                "All thermodynamic states must share one phase-space measure."
            )
        if plans[0].phase_space.system_id != dynamics.system.prepared_id:
            raise ValueError("Thermodynamic states belong to another atomistic system.")
        constraint_manifold_id = (
            None if dynamics.constraints is None else dynamics.constraints.prepared_id
        )
        if (
            dynamics.system.topology.constraint_count > 0
            and constraint_manifold_id is None
        ):
            raise ValueError(
                "Thermodynamic binding requires a prepared executor for topology constraints."
            )
        executed_measure_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-phase-space-measure",
                "declared_measure": measure_id,
                "constraint_manifold": constraint_manifold_id,
            }
        )
        if any(state.control_ids != control_ids for state in plans):
            raise ValueError(
                "Thermodynamic states must share one ordered control layout."
            )
        if (
            control_ids != dynamics.potential.control_ids
            or canonical_fingerprint(
                {
                    "kind": "atomistic-control-layout",
                    "control_ids": list(control_ids),
                }
            )
            != dynamics.potential.control_layout_id
        ):
            raise ValueError(
                "Thermodynamic controls do not match the Hamiltonian control layout."
            )
        state_ids = tuple(state.state_id for state in plans)
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("Thermodynamic state IDs must be unique and ordered.")
        if any(state.ensemble == "npt" for state in plans):
            if dynamics.system.cell is None or not dynamics.system.cell.fully_periodic:
                raise ValueError("NPT states require a fully periodic atomistic system.")
        dtype = np.dtype(dynamics.system.plan.coordinate_dtype)
        boltzmann = dynamics.system.plan.units.boltzmann_constant
        temperature_mask = np.asarray(
            [state.temperature is not None for state in plans], dtype=np.bool_
        )
        pressure_mask = np.asarray(
            [state.pressure is not None for state in plans], dtype=np.bool_
        )
        temperature = np.asarray(
            [0.0 if state.temperature is None else state.temperature for state in plans],
            dtype=dtype,
        )
        pressure = np.asarray(
            [0.0 if state.pressure is None else state.pressure for state in plans],
            dtype=dtype,
        )
        beta = np.zeros_like(temperature)
        beta[temperature_mask] = 1.0 / (boltzmann * temperature[temperature_mask])
        controls = np.stack([np.asarray(state.controls, dtype=dtype) for state in plans])
        ensemble_code = np.asarray(
            [_ENSEMBLE_CODES[state.ensemble] for state in plans], dtype=np.int32
        )
        bias_ids = tuple(state.bias_id for state in plans)
        if len(set(bias_ids)) != 1:
            raise ValueError(
                "Thermodynamic states must share one embedded bias identity."
            )
        if bias_ids[0] is not None:
            raise ValueError(
                "Typed bias cross-evaluation is unavailable; nonempty bias IDs "
                "cannot be authenticated as embedded Hamiltonian terms."
            )
        requires_cross = bool(np.any(controls != controls[0]))
        hamiltonian_labels: dict[bytes, int] = {}
        hamiltonian_indices = []
        for control_row in controls:
            key = np.ascontiguousarray(control_row).tobytes()
            if key not in hamiltonian_labels:
                hamiltonian_labels[key] = len(hamiltonian_labels)
            hamiltonian_indices.append(hamiltonian_labels[key])
        hamiltonian_index = np.asarray(hamiltonian_indices, dtype=np.int32)
        layout_id = dynamics.potential.control_layout_id
        potential_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "atomistic-state-hamiltonian",
                    "program": dynamics.potential.prepared_id,
                    "control_layout": layout_id,
                    "controls": array_tree_fingerprint(control_row),
                    "bias": bias_id,
                }
            )
            for bias_id, control_row in zip(bias_ids, controls, strict=True)
        )
        reduced_convention_id = canonical_fingerprint(
            {
                "kind": "atomistic-configurational-reduced-potential",
                "expression": "beta*(potential_energy+pressure*volume)",
                "unit_id": "1",
                "unit_system": dynamics.system.plan.units.unit_system_id,
                "volume_coordinate_convention": measure.volume_coordinate_convention,
                "measure": executed_measure_id,
            }
        )
        table_id = canonical_fingerprint(
            {
                "kind": "prepared-thermodynamic-state-table",
                "dynamics": dynamics.prepared_id,
                "system": dynamics.system.prepared_id,
                "program": dynamics.potential.prepared_id,
                "measure": executed_measure_id,
                "declared_measure": measure_id,
                "scaled_entity_count": measure.scaled_entity_count,
                "volume_coordinate_convention": measure.volume_coordinate_convention,
                "constraint_manifold": constraint_manifold_id,
                "reduced_convention": reduced_convention_id,
                "units": dynamics.system.plan.units.unit_system_id,
                "layout": layout_id,
                "state_ids": list(state_ids),
                "bias_ids": list(bias_ids),
                "potential_ids": list(potential_ids),
                "beta": array_tree_fingerprint(beta),
                "temperature": array_tree_fingerprint(temperature),
                "temperature_mask": array_tree_fingerprint(temperature_mask),
                "pressure": array_tree_fingerprint(pressure),
                "pressure_mask": array_tree_fingerprint(pressure_mask),
                "controls": array_tree_fingerprint(controls),
                "ensemble_code": array_tree_fingerprint(ensemble_code),
                "hamiltonian_index": array_tree_fingerprint(hamiltonian_index),
            }
        )
        self.beta = jnp.asarray(beta)
        self.temperature = jnp.asarray(temperature)
        self.temperature_mask = jnp.asarray(temperature_mask)
        self.pressure = jnp.asarray(pressure)
        self.pressure_mask = jnp.asarray(pressure_mask)
        self.controls = jnp.asarray(controls)
        self.ensemble_code = jnp.asarray(ensemble_code)
        self.state_ids = state_ids
        self.control_ids = control_ids
        self.hamiltonian_index = jnp.asarray(hamiltonian_index)
        self.bias_ids = bias_ids
        self.potential_ids = potential_ids
        self.state_count = len(plans)
        self.control_count = len(control_ids)
        self.phase_space_measure_id = executed_measure_id
        self.declared_phase_space_measure_id = measure_id
        self.scaled_entity_count = measure.scaled_entity_count
        self.volume_coordinate_convention = measure.volume_coordinate_convention
        self.unit_system_id = dynamics.system.plan.units.unit_system_id
        self.system_id = dynamics.system.prepared_id
        self.program_id = dynamics.potential.prepared_id
        self.control_layout_id = layout_id
        self.dynamics_id = dynamics.prepared_id
        self.constraint_manifold_id = constraint_manifold_id
        self.reduced_convention_id = reduced_convention_id
        self.requires_cross_evaluation = requires_cross
        self.table_id = table_id

    def validate_dynamics(self, dynamics: Any, /) -> None:
        constraint_manifold_id = (
            None if dynamics.constraints is None else dynamics.constraints.prepared_id
        )
        if (
            dynamics.prepared_id != self.dynamics_id
            or dynamics.system.prepared_id != self.system_id
            or dynamics.potential.prepared_id != self.program_id
            or dynamics.system.plan.units.unit_system_id != self.unit_system_id
            or constraint_manifold_id != self.constraint_manifold_id
        ):
            raise ValueError("Thermodynamic state table belongs to another runtime.")

    def state_at_replica(self, state_at_replica: ArrayLike, /) -> _ThermodynamicStateRows:
        indices = jnp.asarray(state_at_replica, dtype=jnp.int32)
        valid = (indices >= 0) & (indices < self.state_count)
        safe = jnp.clip(indices, 0, self.state_count - 1)
        return _ThermodynamicStateRows(
            state_index=indices,
            beta=jnp.take(self.beta, safe, axis=0),
            temperature=jnp.take(self.temperature, safe, axis=0),
            temperature_mask=jnp.take(self.temperature_mask, safe, axis=0),
            pressure=jnp.take(self.pressure, safe, axis=0),
            hamiltonian_index=jnp.take(self.hamiltonian_index, safe, axis=0),
            pressure_mask=jnp.take(self.pressure_mask, safe, axis=0),
            controls=jnp.take(self.controls, safe, axis=0),
            ensemble_code=jnp.take(self.ensemble_code, safe, axis=0),
            valid=valid,
        )


__all__ = [
    "AtomisticEnsembleKind",
    "AtomisticPhaseSpaceMeasurePlan",
    "AtomisticThermodynamicStatePlan",
    "PreparedThermodynamicStateTable",
]
