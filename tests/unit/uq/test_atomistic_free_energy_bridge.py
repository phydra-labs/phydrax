from functools import cache
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig
from phydrax.atomistic._dynamics import AtomisticDynamicsState, VelocityVerletPlan
from tests._support.atomistic_dynamics_fixtures import atomistic_system, prepare_dynamics
from tests._support.atomistic_sampling_fixtures import nve_table, start, zero_field


def test_atomistic_free_energy_bridge_scenario_1() -> None:
    evaluation = phx.atomistic.AlchemicalReducedPotentialEvaluation(
        values=jnp.asarray([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]),
        energies=jnp.asarray([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]),
        coverage=jnp.ones((2, 4), dtype="bool"),
        successful=jnp.ones((2, 4), dtype="bool"),
        state_indices=jnp.asarray([0, 1]),
        inverse_temperatures=jnp.ones((2,)),
        controls=jnp.asarray([[0.0], [1.0]]),
        state_ids=("lambda-zero", "lambda-one"),
        potential_ids=("potential-zero", "potential-one"),
        bias_ids=(None, None),
        control_ids=("lambda",),
        measure_id="union-topology-measure",
        unit_system_id="reduced-unit-system",
        thermodynamic_table_id="thermodynamic-table",
        prepared_id="prepared-controlled-hamiltonian",
        schedule_id="alchemical-schedule",
    )
    dataset = phx.uq.reduced_potential_dataset_from_alchemical_evaluation(
        evaluation,
        jnp.ones((4,), dtype="bool"),
        jnp.asarray([0, 0, 1, 1]),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.arange(4),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.zeros((4,), dtype=jnp.int32),
        state_ids=evaluation.state_ids,
        bias_ids=evaluation.bias_ids,
        control_ids=evaluation.control_ids,
        inverse_temperatures=evaluation.inverse_temperatures,
        reduced_convention_id="beta-times-controlled-hamiltonian-energy",
        qualification_id="exact-controlled-hamiltonian-evaluation",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        potential_ids=evaluation.potential_ids,
        measure_id=evaluation.measure_id,
        unit_system_id=evaluation.unit_system_id,
        prepared_id=evaluation.prepared_id,
        thermodynamic_table_id=evaluation.thermodynamic_table_id,
        schedule_id=evaluation.schedule_id,
        run_id="cross-evaluation-run",
        unit_id="1",
    )
    np.testing.assert_array_equal(dataset.state_counts, [2, 2])
    assert dataset.potential_ids == evaluation.potential_ids
    assert dataset.unit_system_id == evaluation.unit_system_id
    with pytest.raises(ValueError, match="potential_ids"):
        phx.uq.reduced_potential_dataset_from_alchemical_evaluation(
            evaluation,
            jnp.ones((4,), dtype="bool"),
            jnp.asarray([0, 0, 1, 1]),
            jnp.zeros((4,), dtype=jnp.int32),
            jnp.arange(4),
            jnp.zeros((4,), dtype=jnp.int32),
            jnp.zeros((4,), dtype=jnp.int32),
            state_ids=evaluation.state_ids,
            bias_ids=evaluation.bias_ids,
            control_ids=evaluation.control_ids,
            inverse_temperatures=evaluation.inverse_temperatures,
            reduced_convention_id="beta-times-controlled-hamiltonian-energy",
            qualification_id="exact-controlled-hamiltonian-evaluation",
            sampling_exact=True,
            sampling_bias_bound=0.0,
            potential_ids=("wrong", "potential-one"),
            measure_id=evaluation.measure_id,
            unit_system_id=evaluation.unit_system_id,
            prepared_id=evaluation.prepared_id,
            thermodynamic_table_id=evaluation.thermodynamic_table_id,
            schedule_id=evaluation.schedule_id,
            run_id="cross-evaluation-run",
            unit_id="1",
        )
    forward = _switching_dataset("forward")
    paired = _switching_dataset("both")
    reverse = _switching_dataset("reverse")
    np.testing.assert_array_equal(forward.direction_counts, [4, 0])
    np.testing.assert_array_equal(paired.direction_counts, [4, 4])
    assert reverse.state_ids == ("destination-state", "source-state")
    np.testing.assert_allclose(
        phx.uq.free_energy_perturbation(forward).free_energies,
        [0.0, 1.25],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        phx.uq.bennett_acceptance_ratio(paired).free_energies,
        [0.0, 1.25],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        phx.uq.free_energy_perturbation(reverse).free_energies,
        [0.0, -1.25],
        atol=1.0e-12,
    )
    # Equal caller labels cannot stand in for the record's own bias identity.
    with pytest.raises(ValueError, match="bias_ids"):
        phx.uq.reduced_work_dataset_from_alchemical_switching(
            _switching_record(),
            direction="forward",
            source_origin_id=11,
            destination_origin_id=22,
            state_ids=("source-state", "destination-state"),
            potential_ids=("source-potential", "destination-potential"),
            measure_ids=("common-measure", "common-measure"),
            unit_system_id="reduced-unit-system",
            unit_id="1",
            producer_id="switching-producer",
            run_id="switching-run",
            schedule_id="switching-schedule",
            hamiltonian_id="controlled-hamiltonian",
            thermodynamic_table_id="switching-thermodynamic-table",
            bias_ids=("frozen-looking-label", "frozen-looking-label"),
        )


@cache
def _endpoint_states(count: int, /) -> tuple[AtomisticDynamicsState, ...]:
    """Real accepted endpoint states; the bridge checks their count only."""
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        system = atomistic_system([0, 1], [1.0, 1.0])
        dynamics = prepare_dynamics(system, (zero_field(),), VelocityVerletPlan(1.0e-3))
        state = start(
            dynamics, nve_table(dynamics), jnp.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        )
    return (state,) * count


def _switching_record() -> Any:
    forward_lineage = phx.atomistic.AlchemicalSwitchingLineage(
        jnp.full((4,), 11),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.arange(4),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.zeros((4,), dtype=jnp.int32),
    )
    reverse_lineage = phx.atomistic.AlchemicalSwitchingLineage(
        jnp.full((4,), 22),
        jnp.ones((4,), dtype=jnp.int32),
        jnp.arange(4),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.zeros((4,), dtype=jnp.int32),
    )
    return phx.atomistic.AlchemicalSwitchingRecord(
        forward_work=jnp.full((4,), 1.25),
        reverse_work=jnp.full((4,), -1.25),
        forward_coverage=jnp.ones((4,), dtype="bool"),
        reverse_coverage=jnp.ones((4,), dtype="bool"),
        forward_lineage=forward_lineage,
        reverse_lineage=reverse_lineage,
        forward_final_states=_endpoint_states(4),
        reverse_final_states=_endpoint_states(4),
        source_state_id="source-state",
        destination_state_id="destination-state",
        source_potential_id="source-potential",
        destination_potential_id="destination-potential",
        measure_ids=("common-measure", "common-measure"),
        unit_system_id="reduced-unit-system",
        inverse_temperature=1.0,
        unit_id="1",
        producer_id="switching-producer",
        run_id="switching-run",
        schedule_id="switching-schedule",
        hamiltonian_id="controlled-hamiltonian",
        thermodynamic_table_id="switching-thermodynamic-table",
        qualification_id="exact-nonequilibrium-endpoint-sampling",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        bias_ids=(None, None),
        bias_history_digests=(),
        work_kind="nonequilibrium-switching",
        forward_orientation="source-to-destination",
        reverse_orientation="destination-to-source",
        successful=True,
        work_id="switching-record",
    )


def _switching_dataset(direction: Any) -> Any:
    record = _switching_record()
    return phx.uq.reduced_work_dataset_from_alchemical_switching(
        record,
        direction=direction,
        source_origin_id=11,
        destination_origin_id=22,
        state_ids=(record.source_state_id, record.destination_state_id),
        potential_ids=(record.source_potential_id, record.destination_potential_id),
        measure_ids=record.measure_ids,
        unit_system_id=record.unit_system_id,
        unit_id=record.unit_id,
        producer_id=record.producer_id,
        run_id=record.run_id,
        schedule_id=record.schedule_id,
        hamiltonian_id=record.hamiltonian_id,
        thermodynamic_table_id=record.thermodynamic_table_id,
        bias_ids=(None, None),
    )
