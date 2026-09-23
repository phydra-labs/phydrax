import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic.sampling import AtomisticMultistateSegmentResult


def test_alchemical_cross_evaluation_bridge_preserves_exact_evidence():
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


def _switching_record():
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
        forward_final_states=(None,) * 4,
        reverse_final_states=(None,) * 4,
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
        work_kind="nonequilibrium-switching",
        forward_orientation="source-to-destination",
        reverse_orientation="destination-to-source",
        successful=True,
        work_id="switching-record",
    )


def _switching_dataset(direction):
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


def test_switching_bridge_builds_forward_and_bidirectional_estimator_inputs():
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


def test_multistate_segment_bridge_flattens_capacity_then_replica_order():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    reduced = jnp.arange(8.0).reshape((2, 2, 2))
    sample_shape = (2, 2)
    result = AtomisticMultistateSegmentResult(
        successor_state=None,
        reduced_potentials=reduced,
        coverage=jnp.ones_like(reduced, dtype="bool"),
        sample_active=jnp.ones(sample_shape, dtype="bool"),
        origin_state=jnp.asarray([[0, 1], [1, 0]]),
        state_at_replica=jnp.asarray([[0, 1], [1, 0]]),
        chain_index=jnp.asarray([[0, 1], [0, 1]]),
        draw_index=jnp.asarray([[0, 0], [1, 1]]),
        repeat_index=jnp.zeros(sample_shape, dtype=jnp.int32),
        dependence_group_index=jnp.zeros(sample_shape, dtype=jnp.int32),
        pair_indices=jnp.zeros((2, 1, 2), dtype=jnp.int32),
        exchange_attempted=jnp.zeros((2, 1), dtype="bool"),
        exchange_accepted=jnp.zeros((2, 1), dtype="bool"),
        exchange_log_acceptance=jnp.zeros((2, 1)),
        sams_attempted=jnp.zeros(sample_shape, dtype="bool"),
        sams_adapting=jnp.zeros(sample_shape, dtype="bool"),
        sams_changed=jnp.zeros(sample_shape, dtype="bool"),
        dynamics_accepted=jnp.ones(sample_shape, dtype="bool"),
        barostat_attempted=jnp.zeros(sample_shape, dtype="bool"),
        barostat_accepted=jnp.zeros(sample_shape, dtype="bool"),
        iteration_valid=jnp.ones((2,), dtype="bool"),
        count=jnp.asarray(2),
        start_watermark=jnp.asarray(0),
        stop_watermark=jnp.asarray(2),
        successful=jnp.asarray(True),
        units=units,
        run_id="multistate-run",
        measure_id="phase-space-measure",
        state_ids=("state-zero", "state-one"),
        potential_ids=("potential-zero", "potential-one"),
        bias_ids=(None, None),
        inverse_temperatures=jnp.ones((2,)),
        reduced_convention_id="beta-times-potential-energy",
        qualification_id="exact-multistate-sampling",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        producer_id="prepared-multistate-runtime",
        unit_id="1",
        segment_id="segment",
        predecessor_id="predecessor",
        runtime_id="prepared-multistate-runtime",
    )
    dataset = phx.uq.reduced_potential_dataset_from_multistate(result)
    np.testing.assert_array_equal(
        dataset.values,
        reduced.transpose((1, 0, 2)).reshape((2, 4)),
    )
    np.testing.assert_array_equal(dataset.origin_state, [0, 1, 1, 0])
    assert dataset.producer_id == result.producer_id
    assert dataset.bias_ids == result.bias_ids
    assert dataset.unit_system_id == units.unit_system_id
