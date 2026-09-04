from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.robotics._soft_observations import (
    prepare_soft_observation_plan,
    SoftEnergyLoadQueryPlan,
    SoftFrameQueryPlan,
    SoftObservationPlan,
    SoftReducedStateQueryPlan,
    SoftSensorPlan,
    SoftStrainQueryPlan,
    SoftTendonQueryPlan,
)
from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_plant import prepare_reduced_rod_plant
from phydrax.applications.solid_mechanics._rod_reconstruction import (
    prepare_rod_reconstruction,
    RodFrameQueryPlan,
    RodReconstructionPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import RodStrainBasisPlan
from phydrax.applications.solid_mechanics._rod_reduced_dynamics import (
    prepare_reduced_rod_dynamics,
)
from phydrax.applications.solid_mechanics._rod_reduced_integrators import (
    ReducedRodSemiImplicitVelocityEuler,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
)
from phydrax.applications.solid_mechanics._rod_tendon import (
    FrictionlessElasticTendonPlan,
    RodMaterialStation,
    TendonActuatorState,
    TendonRoutePlan,
)
from phydrax.dynamics import PlantRuntimeState, PlantStepContext


def _plant_and_tendon():
    dtype = jnp.float32
    nodes = jnp.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)),
        dtype=dtype,
    )
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            nodes,
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.ones((3,), dtype=dtype),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((80.0, 60.0, 40.0), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((8.0, 7.0, 6.0), dtype=dtype)),
                (1, 3, 3),
            ),
        )
    )
    basis = RodStrainBasisPlan.piecewise_constant(
        jnp.asarray((0.0, 1.0), dtype=dtype),
        dimension=3,
        component_scales=jnp.ones((6,), dtype=dtype),
    )
    reduction = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    dynamics = prepare_reduced_rod_dynamics(reduction)
    plant = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.05, energy_balance_tolerance=1.0e-4
        ),
    )
    route = TendonRoutePlan(
        (
            RodMaterialStation(0, 0.0, jnp.zeros((3,), dtype=dtype)),
            RodMaterialStation(1, 1.0, jnp.zeros((3,), dtype=dtype)),
        )
    )
    tendon = FrictionlessElasticTendonPlan(
        route,
        20.0,
        free_length_bounds=(0.5, 1.5),
        payout_rate_bounds=(-0.2, 0.2),
        tendon_length_bounds=(0.5, 1.5),
        maximum_tension=20.0,
    ).prepare(reduction)
    return plant, tendon


def _runtime(plant, *, time=0.0, epoch=0, key_seed=7):
    dtype = plant.initial_state.reduced_state.values.dtype
    return PlantRuntimeState(
        plant.initial_state,
        jnp.asarray(time, dtype=dtype),
        jnp.asarray(epoch, dtype=jnp.int32),
        jax.random.key(key_seed),
        plant.semantic_provenance.semantic_id,
        plant.numeric_revision.revision_id,
        plant.state_schema.schema_id,
        plant.execution_signature.signature_id,
    )


def _full_plan(tendon, *, sensor=None, ledger=False):
    frame_reconstruction = RodReconstructionPlan(
        RodFrameQueryPlan(jnp.asarray((0.23, 1.0), dtype=jnp.float32))
    )
    strain_reconstruction = RodReconstructionPlan(
        RodFrameQueryPlan(jnp.asarray((0.37,), dtype=jnp.float32))
    )
    return SoftObservationPlan(
        reduced_state=SoftReducedStateQueryPlan(),
        frame=SoftFrameQueryPlan(
            frame_reconstruction,
            twists=("body", "frame_world"),
        ),
        strain=SoftStrainQueryPlan(
            strain_reconstruction,
            include_total=True,
            include_reduced=True,
        ),
        tendon=SoftTendonQueryPlan(
            (tendon,),
            tendon_names=("center",),
            include_length=True,
            include_length_rate=True,
            include_tension=True,
            include_stored_energy=True,
        ),
        energy_load=SoftEnergyLoadQueryPlan(include_step_ledger=ledger),
        sensor=sensor,
    )


def test_exact_layout_arbitrary_arc_length_provenance_and_zero_noise_mechanics():
    plant, tendon = _plant_and_tendon()
    runtime = _runtime(plant)
    plan = prepare_soft_observation_plan(
        plant,
        _full_plan(
            tendon,
            sensor=SoftSensorPlan(
                "joint-encoder-and-shape",
                noise_standard_deviation=0.0,
                sample_period=0.0,
            ),
        ),
    )
    tendon_state = plan.bind_tendon_state(
        runtime, (TendonActuatorState(jnp.asarray(0.8, dtype=jnp.float32)),)
    )
    sensor_state = plan.initialize_sensor_state(jnp.asarray(0.0, dtype=jnp.float32))

    observation, candidate_sensor = plan.observe(
        runtime,
        tendon_state=tendon_state,
        sensor_state=sensor_state,
    )

    assert plan.layout.size == 75
    assert plan.layout.groups == (
        ("reduced_state", 0, 12),
        ("frame", 12, 50),
        ("strain", 50, 62),
        ("tendon", 62, 66),
        ("energy_load", 66, 75),
    )
    assert plan.layout.component_names[12:19] == (
        "frame[0].pose.qw",
        "frame[0].pose.qx",
        "frame[0].pose.qy",
        "frame[0].pose.qz",
        "frame[0].pose.x",
        "frame[0].pose.y",
        "frame[0].pose.z",
    )
    assert plan.layout.component_units[12:19] == (
        "1",
        "1",
        "1",
        "1",
        "m",
        "m",
        "m",
    )
    assert plan.layout.component_frames[12:19] == (
        "world<-material",
        "world<-material",
        "world<-material",
        "world<-material",
        "world",
        "world",
        "world",
    )
    assert plan.layout.component_names[62:66] == (
        "tendon[center].length",
        "tendon[center].length_rate",
        "tendon[center].tension",
        "tendon[center].stored_energy",
    )
    assert plan.layout.component_units[62:66] == ("m", "m/s", "N", "J")
    assert len(set(plan.layout.component_query_ids[12:50])) == 1
    assert len(set(plan.layout.component_query_ids[50:62])) == 1
    assert plan.layout.component_query_ids[12] != plan.layout.component_query_ids[50]

    frame_evaluation = plan.frame.reconstruction.evaluate(runtime.payload.reduced_state)
    strain_evaluation = plan.strain.reconstruction.evaluate(runtime.payload.reduced_state)
    assert jnp.array_equal(
        frame_evaluation.arc_lengths,
        jnp.asarray((0.23, 1.0), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        strain_evaluation.arc_lengths, jnp.asarray((0.37,), dtype=jnp.float32)
    )
    expected_frame = jnp.concatenate(
        (
            frame_evaluation.poses.reshape((-1,)),
            frame_evaluation.body_twists.reshape((-1,)),
            frame_evaluation.frame_velocities.reshape((-1,)),
        )
    )
    expected_strain = jnp.concatenate(
        (
            strain_evaluation.strains.reshape((-1,)),
            strain_evaluation.reduced_strains.reshape((-1,)),
        )
    )
    assert jnp.allclose(
        observation.values[plan.layout.slice_for("frame")], expected_frame
    )
    assert jnp.allclose(
        observation.values[plan.layout.slice_for("strain")], expected_strain
    )
    assert jnp.allclose(observation.values, observation.ideal_values)
    assert jnp.array_equal(observation.noise, jnp.zeros_like(observation.noise))
    assert jnp.array_equal(observation.bias, jnp.zeros_like(observation.bias))
    assert observation.fresh
    assert observation.valid
    assert observation.timestamp == runtime.time
    assert observation.epoch == runtime.step_index
    assert observation.semantic_provenance_id == runtime.semantic_provenance_id
    assert observation.numeric_revision_id == runtime.numeric_revision_id
    assert observation.state_schema_id == runtime.state_schema_id
    assert observation.execution_signature_id == runtime.execution_signature_id
    assert observation.query_plan_id == plan.query_plan_id
    assert observation.query_ids == plan.query_ids
    assert observation.sensor_id == "joint-encoder-and-shape"
    assert candidate_sensor is not sensor_state


def test_same_plant_key_replays_noise_and_sample_hold_is_explicit():
    plant, _ = _plant_and_tendon()
    runtime = _runtime(plant, key_seed=19)
    prepared = prepare_soft_observation_plan(
        plant,
        SoftObservationPlan(
            reduced_state=SoftReducedStateQueryPlan(),
            sensor=SoftSensorPlan(
                "reduced-encoder",
                noise_standard_deviation=0.05,
                sample_period=0.1,
            ),
        ),
    )
    source = prepared.initialize_sensor_state(0.02)

    first, first_candidate = prepared.observe(runtime, sensor_state=source)
    replay, replay_candidate = prepared.observe(runtime, sensor_state=source)

    assert jnp.array_equal(first.values, replay.values)
    assert jnp.array_equal(first.noise, replay.noise)
    assert jnp.array_equal(first.noise_key, replay.noise_key)
    assert jnp.array_equal(first_candidate.held_values, replay_candidate.held_values)
    assert jnp.allclose(first.values, first.ideal_values + first.bias + first.noise)

    held_runtime = _runtime(plant, time=0.05, epoch=1, key_seed=20)
    held, held_candidate = prepared.observe(held_runtime, sensor_state=first_candidate)
    assert held.sample_held
    assert not held.fresh
    assert held.sample_timestamp == first.sample_timestamp
    assert held.sample_epoch == first.sample_epoch
    assert held.age == pytest.approx(0.05)
    assert jnp.array_equal(held.values, first.values)

    due_runtime = _runtime(plant, time=0.11, epoch=2, key_seed=21)
    resampled, _ = prepared.observe(due_runtime, sensor_state=held_candidate)
    assert resampled.fresh
    assert not resampled.sample_held
    assert resampled.sample_timestamp == due_runtime.time
    assert resampled.sample_epoch == due_runtime.step_index
    assert not jnp.array_equal(resampled.noise_key, first.noise_key)


def test_foreign_plant_stale_tendon_and_query_mismatches_reject():
    plant, tendon = _plant_and_tendon()
    runtime = _runtime(plant)
    prepared = prepare_soft_observation_plan(plant, _full_plan(tendon))
    bound_tendon = prepared.bind_tendon_state(
        runtime, (TendonActuatorState(jnp.asarray(0.8, dtype=jnp.float32)),)
    )

    foreign = PlantRuntimeState(
        runtime.payload,
        runtime.time,
        runtime.step_index,
        runtime.key,
        runtime.semantic_provenance_id + ":foreign",
        runtime.numeric_revision_id,
        runtime.state_schema_id,
        runtime.execution_signature_id,
    )
    with pytest.raises(ValueError, match="different prepared plant"):
        prepared.observe(foreign, tendon_state=bound_tendon)

    newer = _runtime(plant, time=0.1, epoch=1, key_seed=8)
    with pytest.raises(Exception, match="Tendon actuator state is stale"):
        observation, _ = prepared.observe(newer, tendon_state=bound_tendon)
        jax.block_until_ready(observation.values)

    wrong_reconstruction = prepare_rod_reconstruction(
        plant.dynamics.reduction,
        RodReconstructionPlan(
            RodFrameQueryPlan(jnp.asarray((0.1, 0.9), dtype=jnp.float32))
        ),
    ).evaluate(runtime.payload.reduced_state)
    with pytest.raises(ValueError, match="different query"):
        prepared.observe(
            runtime,
            tendon_state=bound_tendon,
            frame_evaluation=wrong_reconstruction,
        )


def test_complete_energy_load_ledger_requires_the_current_accepted_step():
    plant, _ = _plant_and_tendon()
    parameters = plant.bind_parameters()
    reset = plant.reset(jax.random.key(31), parameters)
    context = PlantStepContext(
        reset.accepted_state.time,
        reset.accepted_state.time + jnp.asarray(0.01, dtype=jnp.float32),
        reset.accepted_state.step_index,
    )
    step = plant.step(context, reset.accepted_state, None, parameters)
    assert step.successful
    prepared = prepare_soft_observation_plan(
        plant,
        SoftObservationPlan(
            energy_load=SoftEnergyLoadQueryPlan(
                include_mechanics=True, include_step_ledger=True
            )
        ),
    )

    observation, sensor_state = prepared.observe(step.accepted_state, plant_step=step)

    ledger = step.evidence.integration_result.evidence.ledger
    assert sensor_state is None
    assert observation.valid
    assert observation.sensor_id == "mechanics-direct"
    assert prepared.layout.component_names[:3] == (
        "energy.potential",
        "energy.kinetic",
        "energy.total",
    )
    assert "ledger.source[elastic].power_before" in prepared.layout.component_names
    assert "ledger.source[kelvin_voigt].work" in prepared.layout.component_names
    assert "ledger.channel[elastic].power_after" in prepared.layout.component_names
    assert observation.values[
        prepared.layout.index_for("ledger.total_power_after")
    ] == pytest.approx(ledger.total_power_after)
    assert observation.values[
        prepared.layout.index_for("ledger.balance_residual")
    ] == pytest.approx(ledger.balance_residual)
    assert observation.values[
        prepared.layout.index_for("ledger.mechanical_energy_after")
    ] == pytest.approx(ledger.mechanical_energy_after)

    stale_runtime = PlantRuntimeState(
        step.accepted_state.payload,
        step.accepted_state.time + jnp.asarray(0.01, dtype=jnp.float32),
        step.accepted_state.step_index,
        step.accepted_state.key,
        *(
            step.accepted_state.semantic_provenance_id,
            step.accepted_state.numeric_revision_id,
            step.accepted_state.state_schema_id,
            step.accepted_state.execution_signature_id,
        ),
    )
    with pytest.raises(Exception, match="Energy/load ledger is stale"):
        stale, _ = prepared.observe(stale_runtime, plant_step=step)
        jax.block_until_ready(stale.values)
