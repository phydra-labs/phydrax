from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


sampling = phx.atomistic.sampling


def _prepared_runtime(temperatures, replica_count, *, exchange=False, sams=False):
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    topology = phx.atomistic.MolecularTopologyPlan(bonds=[[10, 20]])
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        topology=topology,
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.HarmonicBondPotential([100.0], [1.0])]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-7),
    ).prepare()
    measure = sampling.AtomisticPhaseSpaceMeasurePlan(system)
    states = tuple(
        sampling.AtomisticThermodynamicStatePlan(
            measure,
            ensemble="nvt",
            temperature=temperature,
            state_id=f"state-{index}",
        )
        for index, temperature in enumerate(temperatures)
    )
    table = sampling.PreparedThermodynamicStateTable(dynamics, states)
    exchange_plan = sampling.AtomisticReplicaExchangePlan(1) if exchange else None
    sams_plan = (
        sampling.AtomisticSAMSPlan(
            np.full(len(states), 1.0 / len(states)), adaptation_steps=8
        )
        if sams
        else None
    )
    plan = sampling.AtomisticMultistatePlan(
        table,
        np.arange(replica_count) + 100,
        qualification=sampling.AtomisticCanonicalSamplingQualification(
            dynamics,
            table,
            "focused-synthetic-kernel-qualification",
            sampling_exact=False,
            sampling_bias_bound=1.0,
        ),
        exchange=exchange_plan,
        sams=sams_plan,
        run_id="focused-multistate-runtime",
    )
    return dynamics, table, plan.prepare(dynamics)


def _initial_states(dynamics, table, positions):
    return tuple(
        dynamics.initialize_state(
            position,
            table,
            state_index=min(index, table.state_count - 1),
            velocity=jnp.zeros_like(position),
            key=jax.random.key(index + 11),
        )
        for index, position in enumerate(positions)
    )


def test_compiled_thermodynamic_rows_gather_numerically_under_jit():
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0, 4.0], 2, sams=True)
    rows = eqx.filter_jit(table.state_at_replica)(jnp.asarray([2, 0]))

    assert table.state_count == 3
    assert table.control_count == 0
    assert table.system_id == dynamics.system.prepared_id
    np.testing.assert_allclose(rows.temperature, [4.0, 1.0])
    np.testing.assert_allclose(rows.beta, [0.25, 1.0])
    np.testing.assert_array_equal(rows.valid, [True, True])
    assert rows.controls.shape == (2, 0)

    assert runtime.qualification.dynamics_id == dynamics.prepared_id
    assert runtime.qualification.thermodynamic_table_id == table.table_id
    alternate = sampling.AtomisticThermodynamicStatePlan(
        sampling.AtomisticPhaseSpaceMeasurePlan(dynamics.system),
        ensemble="nvt",
        temperature=1.0,
        state_id="alternate-qualification-target",
    ).prepare(dynamics)
    wrong_qualification = sampling.AtomisticCanonicalSamplingQualification(
        dynamics,
        alternate,
        "wrong-target-qualification",
        sampling_exact=False,
        sampling_bias_bound=1.0,
    )
    with pytest.raises(ValueError, match="another dynamics/table target"):
        sampling.AtomisticMultistatePlan(
            table,
            [100, 200],
            qualification=wrong_qualification,
            exchange=sampling.AtomisticReplicaExchangePlan(1),
            run_id="wrong-qualified-target",
        )


def test_exchange_ladder_dependence_and_rng_are_bound_to_run_identity():
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0], 2, exchange=True)
    np.testing.assert_array_equal(runtime.plan.dependence_group_indices, [0, 0])
    with pytest.raises(ValueError, match="dependence group"):
        sampling.AtomisticMultistatePlan(
            table,
            [100, 200],
            qualification=runtime.plan.qualification,
            exchange=sampling.AtomisticReplicaExchangePlan(1),
            dependence_group_indices=[0, 1],
            run_id="split-exchange-ladder",
        )
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lanes = _initial_states(dynamics, table, (position, position))
    first = runtime.initialize(lanes, [0, 1], jax.random.key(15))
    other = sampling.AtomisticMultistatePlan(
        table,
        [100, 200],
        qualification=runtime.plan.qualification,
        exchange=sampling.AtomisticReplicaExchangePlan(1),
        repeat_index=1,
        run_id="other-run-identity",
    ).prepare(dynamics)
    second = other.initialize(lanes, [0, 1], jax.random.key(15))
    assert not bool(jnp.array_equal(first.root_key, second.root_key))
    assert not bool(
        jnp.array_equal(first.dynamics.random_key, second.dynamics.random_key)
    )


def test_unbound_bias_and_missing_constraint_executor_are_rejected():
    dynamics, _, _ = _prepared_runtime([1.0, 2.0], 2, exchange=True)
    measure = sampling.AtomisticPhaseSpaceMeasurePlan(dynamics.system)
    with pytest.raises(ValueError, match="bias cross-evaluation"):
        sampling.AtomisticThermodynamicStatePlan(
            measure,
            ensemble="nvt",
            temperature=1.0,
            bias_id="unbound-bias",
        ).prepare(dynamics)

    units = phx.atomistic.AtomisticUnitSystem.reduced()
    topology = phx.atomistic.MolecularTopologyPlan(
        constraints=[[10, 20]], constraint_distances=[1.0]
    )
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        topology=topology,
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.1], [0.8], 2.0)]
    ).prepare(system)
    unconstrained = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    with pytest.raises(ValueError, match="prepared executor"):
        sampling.AtomisticThermodynamicStatePlan(
            sampling.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
        ).prepare(unconstrained)


def test_sams_adaptation_draws_are_not_equilibrium_samples():
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0, 3.0], 2, sams=True)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    initial = runtime.initialize(
        _initial_states(dynamics, table, (position, position)),
        [0, 1],
        jax.random.key(18),
    )
    result = sampling.AtomisticMultistateSegmentPlan(
        runtime, 9, 0, 0, runtime.initial_continuation_id
    ).run(initial)
    assert int(result.count) == 9
    assert bool(jnp.all(result.iteration_valid))
    assert not bool(jnp.any(result.sample_active[:8]))
    assert bool(jnp.all(result.sample_active[8]))
    assert not bool(jnp.any(result.coverage[:8]))
    assert bool(jnp.all(result.coverage[8]))
    assert bool(jnp.all(result.sams_adapting[:8]))
    assert not bool(jnp.any(result.sams_adapting[8]))
    np.testing.assert_array_equal(
        result.reduced_potentials[:8],
        jnp.zeros_like(result.reduced_potentials[:8]),
    )
    np.testing.assert_array_equal(result.inverse_temperatures, table.beta)
    assert result.qualification_id == runtime.plan.qualification.qualification_id
    assert not result.sampling_exact
    assert result.sampling_bias_bound == 1.0


def test_valid_rejected_exchange_consumes_counter_and_rebases_ledgers():
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0], 2, exchange=True)
    positions = (
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    )
    initial = runtime.initialize(
        _initial_states(dynamics, table, positions),
        [0, 1],
        jax.random.key(23),
    )
    iteration = runtime.iterate(initial)

    assert bool(iteration.successful)
    assert bool(iteration.exchange_attempted[0])
    assert not bool(iteration.exchange_accepted[0])
    assert int(iteration.state.exchange_action_counter) == 1
    np.testing.assert_array_equal(iteration.state.state_at_replica, [0, 1])
    np.testing.assert_array_equal(
        iteration.state.dynamics.force.position_epoch,
        iteration.state.dynamics.step_index,
    )
    np.testing.assert_allclose(
        iteration.state.dynamics.force.potential_energy,
        iteration.state.dynamics.energy.potential_energy,
    )
    np.testing.assert_allclose(
        iteration.state.dynamics.energy.total_energy,
        iteration.state.dynamics.energy.kinetic_energy
        + iteration.state.dynamics.energy.potential_energy,
    )


def test_accepted_temperature_swap_rescales_momenta_and_keeps_force_current():
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0], 2, exchange=True)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocity = jnp.asarray([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
    lanes = tuple(
        dynamics.initialize_state(
            position,
            table,
            state_index=index,
            velocity=velocity,
            key=jax.random.key(index + 40),
        )
        for index in range(2)
    )
    initial = runtime.initialize(lanes, [0, 1], jax.random.key(43))
    before = initial.dynamics.kinematics.momenta
    iteration = runtime.iterate(initial)

    assert bool(iteration.successful)
    assert bool(iteration.exchange_accepted[0])
    np.testing.assert_array_equal(iteration.state.state_at_replica, [1, 0])
    np.testing.assert_allclose(
        iteration.state.dynamics.kinematics.momenta[0],
        jnp.sqrt(2.0) * before[0],
    )
    np.testing.assert_allclose(
        iteration.state.dynamics.kinematics.momenta[1],
        before[1] / jnp.sqrt(2.0),
    )
    np.testing.assert_array_equal(
        iteration.state.dynamics.thermodynamic_state_index,
        iteration.state.state_at_replica,
    )
    np.testing.assert_array_equal(
        iteration.state.dynamics.force.position_epoch,
        iteration.state.dynamics.step_index,
    )
    np.testing.assert_allclose(
        iteration.state.dynamics.force.potential_energy,
        iteration.state.dynamics.energy.potential_energy,
    )


def test_invalid_iteration_rolls_back_every_action_counter():
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0], 2, exchange=True)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    initial = runtime.initialize(
        _initial_states(dynamics, table, (position, position)),
        [0, 1],
        jax.random.key(29),
    )
    invalid = eqx.tree_at(
        lambda value: value.dynamics.force.position_epoch,
        initial,
        -jnp.ones_like(initial.dynamics.force.position_epoch),
    )
    iteration = runtime.iterate(invalid)

    assert not bool(iteration.successful)
    assert int(iteration.state.iteration_index) == int(invalid.iteration_index)
    assert int(iteration.state.exchange_action_counter) == int(
        invalid.exchange_action_counter
    )
    np.testing.assert_array_equal(
        iteration.state.barostat_action_counter, invalid.barostat_action_counter
    )
    np.testing.assert_array_equal(
        iteration.state.sams_action_counter, invalid.sams_action_counter
    )
    np.testing.assert_array_equal(
        iteration.state.dynamics.kinematics.positions,
        invalid.dynamics.kinematics.positions,
    )


def test_r_and_k_are_distinct_and_failed_segment_padding_is_canonical(
    tmp_path: Path,
):
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0, 3.0], 2, sams=True)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    initial = runtime.initialize(
        _initial_states(dynamics, table, (position, position)),
        [0, 1],
        jax.random.key(31),
    )
    invalid = eqx.tree_at(
        lambda value: value.dynamics.force.position_epoch,
        initial,
        -jnp.ones_like(initial.dynamics.force.position_epoch),
    )
    segment_plan = sampling.AtomisticMultistateSegmentPlan(
        runtime,
        3,
        0,
        0,
        runtime.initial_continuation_id,
    )
    result = segment_plan.run(invalid)

    assert result.reduced_potentials.shape == (3, 3, 2)
    assert result.sample_active.shape == (3, 2)
    assert int(result.count) == 0
    assert not bool(jnp.any(result.iteration_valid))
    assert not bool(jnp.any(result.coverage))
    np.testing.assert_array_equal(
        result.reduced_potentials, jnp.zeros_like(result.reduced_potentials)
    )
    np.testing.assert_array_equal(
        result.origin_state, -jnp.ones_like(result.origin_state)
    )
    np.testing.assert_array_equal(
        result.state_at_replica, -jnp.ones_like(result.state_at_replica)
    )
    assert int(result.successor_state.iteration_index) == 0
    assert int(result.successor_state.segment_index) == 1

    checkpoint_plan = sampling.AtomisticMultistateCheckpointPlan(runtime, 3)
    path = tmp_path / "multistate-segment.phx"
    written = sampling.write_atomistic_multistate_checkpoint(
        path, checkpoint_plan, result
    )
    restored = sampling.read_atomistic_multistate_checkpoint(
        path, checkpoint_plan, invalid
    )
    assert written.payload_id == restored.payload_id
    assert written.manifest_id == result.segment_id == restored.manifest_id
    np.testing.assert_array_equal(
        restored.state.continuation_token,
        result.successor_state.continuation_token,
    )
    np.testing.assert_array_equal(restored.segment.sample_active, result.sample_active)

    with pytest.raises(ValueError, match="predecessor identity"):
        segment_plan.run(restored.state)


def test_multistate_segment_is_jittable_and_matches_eager():
    dynamics, table, runtime = _prepared_runtime([1.0, 2.0], 2, exchange=True)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    initial = runtime.initialize(
        _initial_states(dynamics, table, (position, position)),
        [0, 1],
        jax.random.key(71),
    )
    plan = sampling.AtomisticMultistateSegmentPlan(
        runtime,
        2,
        0,
        0,
        runtime.initial_continuation_id,
    )

    eager = plan.run(initial)
    compiled = eqx.filter_jit(plan.run)(initial)

    np.testing.assert_array_equal(compiled.sample_active, eager.sample_active)
    np.testing.assert_array_equal(
        compiled.reduced_potentials,
        eager.reduced_potentials,
    )
    np.testing.assert_array_equal(
        compiled.successor_state.state_at_replica,
        eager.successor_state.state_at_replica,
    )
    np.testing.assert_array_equal(
        compiled.successor_state.continuation_token,
        eager.successor_state.continuation_token,
    )
