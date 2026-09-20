#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authenticated nonequilibrium switching over native atomistic dynamics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._alchemical import PreparedControlledHamiltonian
from .._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from .._thermodynamic import (
    AtomisticPhaseSpaceMeasurePlan,
    AtomisticThermodynamicStatePlan,
    PreparedThermodynamicStateTable,
)


class AlchemicalSwitchingLineage(StrictModule, NonTrainableState):
    """Per-work sampling lineage matching the native statistical evidence model."""

    origin_ids: Array
    chain_ids: Array
    draw_indices: Array
    repeat_ids: Array
    dependence_ids: Array
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self, origin_ids, chain_ids, draw_indices, repeat_ids, dependence_ids, /
    ):
        arrays = tuple(
            np.asarray(value)
            for value in (
                origin_ids,
                chain_ids,
                draw_indices,
                repeat_ids,
                dependence_ids,
            )
        )
        if arrays[0].ndim != 1 or any(value.shape != arrays[0].shape for value in arrays):
            raise ValueError("Switching lineage arrays must be aligned vectors.")
        if any(not np.issubdtype(value.dtype, np.integer) for value in arrays):
            raise TypeError("Switching lineage arrays must contain integer identities.")
        if any(np.any(value < 0) for value in arrays):
            raise ValueError("Switching lineage identities must be non-negative.")
        identities = tuple(
            zip(
                arrays[0].tolist(),
                arrays[1].tolist(),
                arrays[2].tolist(),
                arrays[3].tolist(),
                strict=True,
            )
        )
        if len(set(identities)) != len(identities):
            raise ValueError(
                "Switching lineage (origin, chain, draw, repeat) identities must be unique."
            )
        canonical = tuple(value.astype(np.int64, copy=False) for value in arrays)
        for name, value in zip(
            ("origin_ids", "chain_ids", "draw_indices", "repeat_ids", "dependence_ids"),
            canonical,
            strict=True,
        ):
            setattr(self, name, jnp.asarray(value))
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "alchemical-switching-lineage",
                "arrays": array_tree_fingerprint(canonical),
            }
        )

    @property
    def sample_count(self) -> int:
        return self.origin_ids.shape[0]


class AlchemicalSwitchingRecord(StrictModule, NonTrainableState):
    """Executed dimensionless work in both physical protocol directions."""

    forward_work: Array
    reverse_work: Array
    forward_coverage: Array
    reverse_coverage: Array
    forward_lineage: AlchemicalSwitchingLineage
    reverse_lineage: AlchemicalSwitchingLineage
    forward_final_states: tuple[AtomisticDynamicsState, ...]
    reverse_final_states: tuple[AtomisticDynamicsState, ...]
    source_state_id: str = eqx.field(static=True)
    destination_state_id: str = eqx.field(static=True)
    source_potential_id: str = eqx.field(static=True)
    destination_potential_id: str = eqx.field(static=True)
    measure_ids: tuple[str, str] = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)
    thermodynamic_table_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    work_kind: str = eqx.field(static=True)
    forward_orientation: str = eqx.field(static=True)
    reverse_orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    work_id: str = eqx.field(static=True)


class AlchemicalSwitchingPlan(StrictModule, NonTrainableState):
    """Execute source↔destination control switching with native dynamics."""

    dynamics: PreparedAtomisticDynamics
    thermodynamic: PreparedThermodynamicStateTable
    qualification: Any
    hamiltonian: PreparedControlledHamiltonian
    source_state_index: int = eqx.field(static=True)
    destination_state_index: int = eqx.field(static=True)
    inverse_temperature: float = eqx.field(static=True)
    integration_steps: int = eqx.field(static=True)
    protocol_time: float = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        thermodynamic: PreparedThermodynamicStateTable,
        qualification: Any,
        source_state_index: int,
        destination_state_index: int,
        integration_steps: int,
        protocol_time: float,
        sample_count: int,
        /,
    ):
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
            raise TypeError("thermodynamic must be PreparedThermodynamicStateTable.")
        from ..sampling._multistate import AtomisticCanonicalSamplingQualification

        if not isinstance(qualification, AtomisticCanonicalSamplingQualification):
            raise TypeError(
                "qualification must be AtomisticCanonicalSamplingQualification."
            )
        if (
            qualification.dynamics_id != dynamics.prepared_id
            or qualification.thermodynamic_table_id != thermodynamic.table_id
            or qualification.measure_id != thermodynamic.phase_space_measure_id
        ):
            raise ValueError(
                "Canonical-sampling qualification is bound to another dynamics/table."
            )
        thermodynamic.validate_dynamics(dynamics)
        if not isinstance(dynamics.potential, PreparedControlledHamiltonian):
            raise TypeError("Switching dynamics require a PreparedControlledHamiltonian.")
        hamiltonian = dynamics.potential
        source = int(source_state_index)
        destination = int(destination_state_index)
        if (
            source < 0
            or source >= thermodynamic.state_count
            or destination < 0
            or destination >= thermodynamic.state_count
            or source == destination
        ):
            raise ValueError("Switching endpoints must be distinct prepared states.")
        steps = int(integration_steps)
        duration = float(protocol_time)
        count = int(sample_count)
        step_size = float(dynamics.integrator.step_size)
        if (
            steps <= 0
            or not math.isfinite(duration)
            or duration <= 0.0
            or count <= 0
            or not math.isclose(duration, steps * step_size, rel_tol=1.0e-12, abs_tol=0.0)
        ):
            raise ValueError(
                "protocol_time must equal integration_steps times the dynamics step size."
            )
        beta = np.asarray(thermodynamic.beta, dtype=np.float64)[[source, destination]]
        ensemble = np.asarray(thermodynamic.ensemble_code, dtype=np.int32)[
            [source, destination]
        ]
        if (
            np.any(~np.isfinite(beta))
            or np.any(beta <= 0.0)
            or not np.allclose(beta, beta[0], rtol=1.0e-12, atol=0.0)
            or np.any(ensemble != 1)
        ):
            raise ValueError(
                "Switching endpoints require one positive inverse temperature and NVT ensemble."
            )
        if thermodynamic.bias_ids[source] != thermodynamic.bias_ids[destination]:
            raise ValueError("Switching endpoints must share one exact bias identity.")
        if thermodynamic.bias_ids[source] is not None:
            raise ValueError(
                "Biased switching requires an explicitly integrated bias Hamiltonian."
            )
        self.dynamics = dynamics
        self.thermodynamic = thermodynamic
        self.qualification = qualification
        self.hamiltonian = hamiltonian
        self.source_state_index = source
        self.destination_state_index = destination
        self.inverse_temperature = float(beta[0])
        self.integration_steps = steps
        self.protocol_time = duration
        self.sample_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "alchemical-switching-plan",
                "dynamics": dynamics.prepared_id,
                "thermodynamic": thermodynamic.table_id,
                "qualification": qualification.qualification_id,
                "source_state": thermodynamic.state_ids[source],
                "destination_state": thermodynamic.state_ids[destination],
                "integration_steps": steps,
                "protocol_time": duration.hex(),
                "sample_count": count,
            }
        )

    @property
    def source_state_id(self) -> str:
        return self.thermodynamic.state_ids[self.source_state_index]

    @property
    def destination_state_id(self) -> str:
        return self.thermodynamic.state_ids[self.destination_state_index]

    def _protocol_table(self, reverse: bool, /) -> PreparedThermodynamicStateTable:
        start = self.destination_state_index if reverse else self.source_state_index
        stop = self.source_state_index if reverse else self.destination_state_index
        state_controls = np.asarray(self.thermodynamic.controls, dtype=np.float64)
        coordinate = np.linspace(float(start), float(stop), self.integration_steps + 1)
        lower = np.floor(coordinate).astype(np.int32)
        upper = np.ceil(coordinate).astype(np.int32)
        fraction = (coordinate - lower.astype("float64"))[:, None]
        controls = (1.0 - fraction) * state_controls[lower] + fraction * state_controls[
            upper
        ]
        temperature = float(np.asarray(self.thermodynamic.temperature)[start])
        phase_space = AtomisticPhaseSpaceMeasurePlan(self.dynamics.system)
        direction = "reverse" if reverse else "forward"
        plans = tuple(
            AtomisticThermodynamicStatePlan(
                phase_space,
                ensemble="nvt",
                temperature=temperature,
                controls=row,
                control_ids=self.hamiltonian.control_ids,
                bias_id=self.thermodynamic.bias_ids[start],
                state_id=f"{self.plan_id}:{direction}:{index}",
            )
            for index, row in enumerate(controls)
        )
        return PreparedThermodynamicStateTable(self.dynamics, plans)

    @staticmethod
    def _bind_state(
        state: AtomisticDynamicsState,
        table: PreparedThermodynamicStateTable,
        state_index: int,
        /,
    ) -> AtomisticDynamicsState:
        return AtomisticDynamicsState(
            time=state.time,
            step_index=state.step_index,
            kinematics=state.kinematics,
            species=state.species,
            cell_vectors=state.cell_vectors,
            neighborhood=state.neighborhood,
            neighborhood_cache=state.neighborhood_cache,
            force=state.force,
            constraint_lagrange=state.constraint_lagrange,
            constraint_position_residual=state.constraint_position_residual,
            constraint_velocity_residual=state.constraint_velocity_residual,
            thermostat_state=state.thermostat_state,
            barostat_state=state.barostat_state,
            random_key=state.random_key,
            energy=state.energy,
            last_status=state.last_status,
            last_rejection_reasons=state.last_rejection_reasons,
            thermodynamic_state_index=jnp.asarray(state_index, dtype=jnp.int32),
            thermodynamic_table_id=table.table_id,
            prepared_dynamics_id=state.prepared_dynamics_id,
        )

    def _execute_direction(
        self,
        initial_states: tuple[AtomisticDynamicsState, ...],
        lineage: AlchemicalSwitchingLineage,
        reverse: bool,
        /,
    ) -> tuple[Array, Array, tuple[AtomisticDynamicsState, ...]]:
        expected_index = (
            self.destination_state_index if reverse else self.source_state_index
        )
        if (
            len(initial_states) != self.sample_count
            or not isinstance(lineage, AlchemicalSwitchingLineage)
            or lineage.sample_count != self.sample_count
            or any(
                not isinstance(state, AtomisticDynamicsState) for state in initial_states
            )
        ):
            raise ValueError(
                "Switching states and authenticated lineage must match sample_count."
            )
        if not np.all(np.asarray(lineage.origin_ids, dtype=np.int64) == expected_index):
            raise ValueError(
                "Switching lineage origins must identify the expected endpoint."
            )
        for state in initial_states:
            if (
                state.prepared_dynamics_id != self.dynamics.prepared_id
                or state.thermodynamic_table_id != self.thermodynamic.table_id
                or int(np.asarray(state.thermodynamic_state_index)) != expected_index
            ):
                raise ValueError(
                    "Switching initial states must belong to the expected dynamics/table endpoint."
                )
        if any(
            int(np.asarray(state.step_index))
            != int(np.asarray(lineage.draw_indices)[index])
            for index, state in enumerate(initial_states)
        ):
            raise ValueError(
                "Switching lineage draw indices must match initial state steps."
            )
        table = self._protocol_table(reverse)
        work_values = []
        coverage = []
        final_states = []
        for initial in initial_states:
            state = self._bind_state(initial, table, 0)
            work = jnp.zeros((), dtype=state.kinematics.positions.dtype)
            successful = state.force.successful
            for next_index in range(1, self.integration_steps + 1):
                previous_external = state.energy.external_work
                rebase = self.dynamics.rebase_thermodynamic_state(
                    state, table, next_index
                )
                successful = successful & rebase.successful
                state = rebase.accepted_state
                increment = state.energy.external_work - previous_external
                work = work + jnp.where(rebase.successful, increment, 0.0)
                step = self.dynamics.step_detailed(state, table)
                successful = successful & step.successful
                state = step.accepted_state
            reduced_work = jnp.asarray(self.inverse_temperature, dtype=work.dtype) * work
            successful = successful & jnp.isfinite(reduced_work)
            work_values.append(jnp.where(successful, reduced_work, jnp.nan))
            coverage.append(successful)
            final_states.append(state)
        return (
            jnp.stack(tuple(work_values)),
            jnp.stack(tuple(coverage)),
            tuple(final_states),
        )

    def execute(
        self,
        forward_initial_states: Sequence[AtomisticDynamicsState],
        reverse_initial_states: Sequence[AtomisticDynamicsState],
        forward_lineage: AlchemicalSwitchingLineage,
        reverse_lineage: AlchemicalSwitchingLineage,
        /,
        *,
        producer_id: str,
        run_id: str,
    ) -> AlchemicalSwitchingRecord:
        producer = str(producer_id).strip()
        run = str(run_id).strip()
        if not producer or not run:
            raise ValueError("producer_id and run_id must be non-empty.")
        forward, forward_coverage, forward_states = self._execute_direction(
            tuple(forward_initial_states), forward_lineage, False
        )
        reverse, reverse_coverage, reverse_states = self._execute_direction(
            tuple(reverse_initial_states), reverse_lineage, True
        )
        payload = {
            "kind": "alchemical-switching-record",
            "plan": self.plan_id,
            "forward_work": array_tree_fingerprint(np.asarray(forward)),
            "reverse_work": array_tree_fingerprint(np.asarray(reverse)),
            "forward_coverage": array_tree_fingerprint(np.asarray(forward_coverage)),
            "reverse_coverage": array_tree_fingerprint(np.asarray(reverse_coverage)),
            "forward_lineage": forward_lineage.lineage_id,
            "reverse_lineage": reverse_lineage.lineage_id,
            "producer": producer,
            "run": run,
            "qualification": self.qualification.qualification_id,
            "sampling_exact": self.qualification.sampling_exact,
            "sampling_bias_bound": self.qualification.sampling_bias_bound,
        }
        successful = bool(
            np.all(np.asarray(forward_coverage)) and np.all(np.asarray(reverse_coverage))
        )
        return AlchemicalSwitchingRecord(
            forward_work=forward,
            reverse_work=reverse,
            forward_coverage=forward_coverage,
            reverse_coverage=reverse_coverage,
            forward_lineage=forward_lineage,
            reverse_lineage=reverse_lineage,
            forward_final_states=forward_states,
            reverse_final_states=reverse_states,
            source_state_id=self.source_state_id,
            destination_state_id=self.destination_state_id,
            source_potential_id=self.thermodynamic.potential_ids[self.source_state_index],
            destination_potential_id=self.thermodynamic.potential_ids[
                self.destination_state_index
            ],
            measure_ids=(
                self.thermodynamic.phase_space_measure_id,
                self.thermodynamic.phase_space_measure_id,
            ),
            unit_system_id=self.thermodynamic.unit_system_id,
            unit_id="1",
            producer_id=producer,
            run_id=run,
            schedule_id=self.hamiltonian.plan.schedule.schedule_id,
            hamiltonian_id=self.hamiltonian.prepared_id,
            thermodynamic_table_id=self.thermodynamic.table_id,
            qualification_id=self.qualification.qualification_id,
            sampling_exact=self.qualification.sampling_exact,
            sampling_bias_bound=self.qualification.sampling_bias_bound,
            work_kind="nonequilibrium-switching",
            forward_orientation="source-to-destination",
            reverse_orientation="destination-to-source",
            successful=successful,
            work_id=canonical_fingerprint(payload),
        )


__all__ = [
    "AlchemicalSwitchingLineage",
    "AlchemicalSwitchingPlan",
    "AlchemicalSwitchingRecord",
]
