#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.mechanics import (
    _growth as growth,
    _sarcomere as sarcomere,
)


def _growth_problem(*, target=5.0, maximum_increment=0.03):
    point_ids = ("lv-q0", "lv-q1")
    directions = np.asarray(
        (
            ((1.0, 2.0, 0.5), (-0.2, 1.0, 1.5)),
            ((1.0, -0.4, 0.3), (0.7, 0.2, -1.0)),
        )
    )
    plan = growth.GrowthPlan(
        point_ids,
        ("sheet-skew", "fiber-skew"),
        directions,
        target,
        1.0,
        0.01,
        minimum_cycles=1,
        maximum_log_increment=maximum_increment,
        maximum_log_magnitude=2.0,
        maximum_refinements=8,
    )
    epoch = growth.GrowthReferenceEpoch(
        "anatomy-A", "unloaded-reference-A", "tet-topology-A", point_ids
    )
    prepared = growth.prepare_growth(plan, epoch)
    return plan, prepared


def _constant_cycle(prepared, value, index=0):
    identity = np.eye(prepared.plan.dimension)
    tensor = value * np.broadcast_to(
        identity,
        (3, prepared.plan.point_count, prepared.plan.dimension, prepared.plan.dimension),
    )
    return growth.aggregate_growth_cycle(
        prepared, index, np.asarray((0.0, 400.0, 900.0)), tensor
    )


def test_homeostatic_cycle_keeps_positive_identity_growth_and_exact_split():
    _, prepared = _growth_problem()
    accumulator = growth.initialize_growth_cycle_accumulator(prepared)
    accumulator = growth.accumulate_growth_cycle(
        prepared, accumulator, _constant_cycle(prepared, 5.0)
    )
    state = growth.initialize_growth_state(prepared)
    stimulus = growth.evaluate_homeostatic_stimulus(prepared, accumulator)
    proposal = growth.propose_growth_step(prepared, state, accumulator, 20.0)
    evidence = growth.evaluate_growth_proposal(prepared, state, proposal)
    committed = growth.commit_growth_step(prepared, state, proposal, evidence)

    assert bool(stimulus.homeostatic)
    assert np.allclose(proposal.log_increment, 0.0)
    assert bool(evidence.passed)
    assert bool(committed.committed)

    total = np.asarray(
        (
            ((1.10, 0.12, 0.03), (0.01, 0.93, -0.04), (0.0, 0.08, 1.02)),
            ((0.96, -0.02, 0.07), (0.05, 1.08, 0.02), (0.01, 0.0, 1.03)),
        )
    )
    kinematics = growth.evaluate_growth_kinematics(prepared, committed.state, total)
    reconstructed = np.matmul(
        np.asarray(kinematics.elastic_deformation_gradient),
        np.asarray(kinematics.growth_deformation_gradient),
    )
    assert np.all(np.asarray(kinematics.growth_positive))
    assert np.all(np.asarray(kinematics.growth_jacobian) > 0.0)
    assert np.allclose(reconstructed, total, rtol=2e-5, atol=2e-6)


def test_slow_growth_requires_refinement_before_atomic_commit():
    _, prepared = _growth_problem(target=0.0, maximum_increment=0.03)
    accumulator = growth.initialize_growth_cycle_accumulator(prepared)
    accumulator = growth.accumulate_growth_cycle(
        prepared, accumulator, _constant_cycle(prepared, 2.0)
    )
    state = growth.initialize_growth_state(prepared)
    proposal = growth.propose_growth_step(prepared, state, accumulator, 10.0)
    evidence = growth.evaluate_growth_proposal(prepared, state, proposal)

    assert not bool(evidence.passed)
    assert int(evidence.status) == int(growth.GrowthStatus.INCREMENT_TOO_LARGE)
    rejected = growth.commit_growth_step(prepared, state, proposal, evidence)
    assert not bool(rejected.committed)
    assert rejected.state.state_id == state.state_id

    while not bool(evidence.passed):
        proposal = growth.refine_growth_proposal(prepared, state, accumulator, proposal)
        evidence = growth.evaluate_growth_proposal(prepared, state, proposal)
    result = growth.commit_growth_step(prepared, state, proposal, evidence)

    assert proposal.refinement_level > 0
    assert float(evidence.maximum_increment) <= prepared.plan.maximum_log_increment
    assert bool(result.committed)
    log_tensor = np.asarray(result.state.log_growth_tensor)
    eigenvalues = np.linalg.eigvalsh(log_tensor)
    psd_tolerance = (
        32.0
        * np.finfo(log_tensor.dtype).eps
        * max(1.0, float(np.max(np.abs(log_tensor))))
    )
    assert np.all(eigenvalues >= -psd_tolerance)


def test_reference_epoch_transfer_is_discrete_and_forces_all_rebuilds():
    source_plan, source = _growth_problem(target=0.0)
    source_state = growth.LogGrowthTensorState(
        np.asarray(
            (
                ((0.1, 0.02, 0.0), (0.02, -0.03, 0.0), (0.0, 0.0, 0.01)),
                ((-0.02, 0.0, 0.01), (0.0, 0.06, 0.0), (0.01, 0.0, 0.03)),
            )
        ),
        50.0,
        source.prepared_id,
    )
    source_accumulator = growth.initialize_growth_cycle_accumulator(source)
    target_ids = ("remesh-q0", "remesh-q1", "remesh-q2")
    transfer_weights = np.asarray(((1.0, 0.0), (0.25, 0.75), (0.0, 1.0)))
    target_directions = np.asarray(source_plan.reference_directions)[
        np.asarray((0, 1, 1))
    ]
    target_plan = growth.GrowthPlan(
        target_ids,
        source_plan.direction_ids,
        target_directions,
        np.zeros((3, 2)),
        np.ones((3, 2)),
        np.full((3, 2), 0.01),
    )
    target_epoch = growth.GrowthReferenceEpoch(
        "anatomy-B", "unloaded-reference-B", "tet-topology-B", target_ids
    )
    target = growth.prepare_growth(target_plan, target_epoch)
    candidate = growth.propose_growth_epoch_transfer(
        source, source_state, target, transfer_weights
    )
    evidence = growth.evaluate_growth_epoch_transfer(source, source_state, candidate)
    result = growth.commit_growth_epoch_transfer(
        source, source_state, source_accumulator, candidate, evidence
    )

    assert bool(evidence.passed)
    assert bool(result.committed)
    assert result.prepared.prepared_id == target.prepared_id
    assert result.accumulator.cycle_count == 0
    assert result.requirements.transfer_growth_state
    assert result.requirements.rebuild_mechanics_reference
    assert result.requirements.rebuild_cycle_aggregator
    assert result.requirements.rebuild_observation_operators
    assert not result.requirements.ordinary_gradient_supported
    with pytest.raises(ValueError, match="different anatomy/reference epoch"):
        growth.evaluate_growth_kinematics(
            source, result.state, np.broadcast_to(np.eye(3), (3, 3, 3))
        )

    source_log = jnp.asarray(source_state.log_growth_tensor)
    weights = jnp.asarray(transfer_weights)
    derivative = jax.grad(
        lambda values: jnp.sum(growth.discrete_growth_log_transfer(values, weights))
    )(source_log)
    assert np.allclose(derivative, 0.0)


def test_invalid_epoch_transfer_rolls_back_source_epoch_and_cycle_history():
    source_plan, source = _growth_problem(target=0.0)
    state = growth.initialize_growth_state(source)
    accumulator = growth.accumulate_growth_cycle(
        source,
        growth.initialize_growth_cycle_accumulator(source),
        _constant_cycle(source, 1.0),
    )
    target_epoch = growth.GrowthReferenceEpoch(
        "anatomy-C",
        "reference-C",
        "topology-C",
        source_plan.material_point_ids,
    )
    target = growth.prepare_growth(source_plan, target_epoch)
    candidate = growth.propose_growth_epoch_transfer(
        source, state, target, np.asarray(((0.8, 0.0), (0.0, 1.0)))
    )
    evidence = growth.evaluate_growth_epoch_transfer(source, state, candidate)
    result = growth.commit_growth_epoch_transfer(
        source, state, accumulator, candidate, evidence
    )

    assert not bool(evidence.passed)
    assert int(evidence.status) == int(growth.GrowthStatus.INVALID_TRANSFER)
    assert not bool(result.committed)
    assert result.prepared.prepared_id == source.prepared_id
    assert result.state.state_id == state.state_id
    assert result.accumulator.cycle_count == 1


def _sarcomere_plan(**overrides):
    parameters = dict(
        attachment_rate_per_ms=0.08,
        powerstroke_rate_per_ms=0.05,
        adp_release_rate_per_ms=0.03,
        atp_binding_rate_per_ms=0.04,
        calcium_half_saturation_mM=0.0005,
        calcium_cooperativity=2.0,
        atp_half_saturation=50.0,
        oxidative_adp_half_saturation=10.0,
        oxidative_pi_half_saturation=10.0,
        oxygen_half_saturation_kpa=2.0,
        oxygen_kinetic_floor=0.15,
        oxygen_per_atp=0.2,
        myosin_site_density=20.0,
        atp_free_energy=0.02,
        resting_length_mm=0.002,
        overlap_width_mm=0.0004,
        shortening_velocity_scale_mm_per_ms=0.0001,
        maximum_active_stress_kpa=50.0,
        balance_tolerance=2.0e-5,
    )
    parameters.update(overrides)
    return sarcomere.MeanFieldSarcomerePlan(**parameters)


def _coupling(oxygen, *, velocity=0.0):
    return sarcomere.SarcomereCouplingInputs(
        np.asarray((0.001, 0.001)),
        np.asarray((0.002, 0.002)),
        np.asarray((velocity, velocity)),
        np.asarray((oxygen, oxygen)),
        np.asarray((1.0, 1.0)),
    )


def test_mean_field_species_and_power_ledgers_close_with_oxygen_modulation():
    plan = _sarcomere_plan()
    fractions = np.asarray(((0.5, 0.5, 0.0, 0.0),) * 2)
    state = sarcomere.SarcomereState(
        fractions,
        np.full(2, 100.0),
        np.full(2, 40.0),
        np.full(2, 40.0),
        0.0,
        plan.plan_id,
    )
    high = sarcomere.step_mean_field_sarcomere(plan, state, _coupling(8.0), 1.0)
    low = sarcomere.step_mean_field_sarcomere(plan, state, _coupling(0.0), 1.0)

    assert bool(high.accepted)
    assert bool(low.accepted)
    assert float(high.evidence.population_sum_error) < 2e-6
    assert float(high.evidence.adenylate_balance_error) < 2e-5
    assert float(high.evidence.phosphoryl_balance_error) < 2e-5
    assert float(high.evidence.maximum_chemical_power_residual) < 2e-5
    assert float(high.evidence.maximum_total_power_residual) < 2e-5
    assert np.all(np.asarray(high.ledger.heat_power_kpa_per_ms) >= 0.0)
    assert np.all(
        np.asarray(high.outputs.modulation.oxygen_limitation)
        > np.asarray(low.outputs.modulation.oxygen_limitation)
    )
    assert np.all(
        np.asarray(high.outputs.atp_regeneration_pmol_per_mm3_ms)
        > np.asarray(low.outputs.atp_regeneration_pmol_per_mm3_ms)
    )
    assert np.all(
        np.asarray(high.extents.powerstroke_fraction)
        > np.asarray(low.extents.powerstroke_fraction)
    )


def test_invalid_oxygen_input_fails_closed_without_changing_state():
    plan = _sarcomere_plan()
    state = sarcomere.initialize_sarcomere_state(
        plan,
        (2,),
        atp_pmol_per_mm3=100.0,
        adp_pmol_per_mm3=20.0,
        phosphate_pmol_per_mm3=20.0,
    )
    result = sarcomere.step_mean_field_sarcomere(plan, state, _coupling(-1.0), 0.5)

    assert not bool(result.accepted)
    assert int(result.evidence.status) == int(
        sarcomere.SarcomereStatus.INVALID_COUPLING_INPUT
    )
    assert np.array_equal(result.state.crossbridge_fractions, state.crossbridge_fractions)
    assert float(result.state.time_ms) == float(state.time_ms)


def test_stochastic_molecular_fidelity_cannot_be_used_as_mean_field_mode():
    stochastic = sarcomere.StochasticMolecularSarcomereFidelity(1000, 8)
    assert stochastic.molecule_count == 1000
    with pytest.raises(TypeError, match="distinct route"):
        _sarcomere_plan(fidelity=stochastic)
