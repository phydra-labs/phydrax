#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import electrophysiology as ep
from phydrax.interchange import AdapterError, AdapterStatus, require_lossless


jax.config.update("jax_enable_x64", True)


def _single(*mechanisms, dt=0.05, scheme="backward-euler"):
    diameter = np.sqrt(100_000.0 / np.pi)
    morphology = ep.CellMorphologyPlan(
        "cell",
        [ep.CompartmentSpec("soma", None, diameter, diameter)],
    ).prepare()
    program = ep.MembraneProgram(mechanisms or (ep.PassiveLeak(0.3, -65.0),))
    return ep.CableSolverPlan(dt, scheme=scheme, residual_tolerance=1.0e-9).prepare(
        morphology, program
    )


def _branched():
    plan = ep.CellMorphologyPlan(
        "branched",
        [
            ep.CompartmentSpec("soma", None, 20.0, 20.0),
            ep.CompartmentSpec("trunk", "soma", 60.0, 3.0),
            ep.CompartmentSpec("left", "trunk", 80.0, 2.0),
            ep.CompartmentSpec("right", "trunk", 100.0, 1.5),
        ],
        branches=(
            ep.BranchSpec("left-branch", ("soma", "trunk", "left")),
            ep.BranchSpec("right-branch", ("soma", "trunk", "right")),
        ),
    )
    return plan.prepare()


def test_explicit_unit_conversion_and_dimension_validation():
    np.testing.assert_allclose(ep.convert_quantity(1.0, "V", "mV"), 1000.0)
    np.testing.assert_allclose(ep.convert_quantity(250.0, "pA", "nA"), 0.25)
    np.testing.assert_allclose(ep.convert_quantity(2.5, "mS", "uS"), 2500.0)
    np.testing.assert_allclose(ep.convert_quantity(1.0, "mM", "mol_per_m3"), 1.0)
    assert ep.ELECTROPHYSIOLOGY_UNITS.voltage == "mV"
    with pytest.raises(ValueError, match="Cannot convert"):
        ep.convert_quantity(1.0, "mV", "ms")
    with pytest.raises(ValueError, match="Unknown"):
        ep.conversion_factor("banana", "mV")


def test_morphology_has_stable_tree_schedule_and_axial_kirchhoff_operator():
    first = _branched()
    second = _branched()
    assert first.runtime_id == second.runtime_id
    assert first.plan.compartment_index("right") == 3
    np.testing.assert_array_equal(np.sort(first.elimination_order), [1, 2, 3])
    np.testing.assert_allclose(first.axial_laplacian_uS, first.axial_laplacian_uS.T)
    np.testing.assert_allclose(
        np.sum(first.axial_laplacian_uS, axis=1), 0.0, atol=1.0e-14
    )
    assert np.all(np.asarray(first.capacitance_nF) > 0.0)


@pytest.mark.parametrize(
    "scheme,theta", [("backward-euler", 1.0), ("crank-nicolson", 0.5)]
)
def test_passive_single_compartment_matches_analytic_theta_mode(scheme, theta):
    runtime = _single(ep.PassiveLeak(0.3, -65.0), dt=0.2, scheme=scheme)
    state = ep.initialize_cable_state(runtime, jnp.asarray([-40.0]))
    result = ep.step_cable(runtime, state, ep.zero_cable_inputs(runtime))
    capacitance_rate = float(runtime.morphology.capacitance_nF[0]) / 0.2
    conductance = 0.3 * float(runtime.morphology.membrane_area_um2[0]) * 1.0e-5
    expected = (
        (capacitance_rate - (1.0 - theta) * conductance) * -40.0 + conductance * -65.0
    ) / (capacitance_rate + theta * conductance)
    np.testing.assert_allclose(
        result.state.voltage_mV, [expected], rtol=2.0e-12, atol=2.0e-12
    )
    assert bool(result.evidence.successful)
    neutral = ep.zero_cable_inputs(runtime)

    def voltage_for_current(current):
        inputs = ep.CableStepInputs(
            jnp.asarray([current]),
            neutral.synaptic_conductance_uS,
            neutral.synaptic_current_offset_nA,
            neutral.voltage_clamp_mask,
            neutral.voltage_clamp_target_mV,
        )
        return ep.step_cable(runtime, state, inputs).state.voltage_mV[0]

    _, tangent = jax.jvp(
        voltage_for_current,
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )
    np.testing.assert_allclose(
        tangent,
        1.0 / (capacitance_rate + theta * conductance),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_branched_cable_satisfies_kirchhoff_balance_and_tree_dense_parity():
    morphology = _branched()
    runtime = ep.CableSolverPlan(0.1, residual_tolerance=1.0e-9).prepare(
        morphology, ep.MembraneProgram((ep.PassiveLeak(0.1, -68.0),))
    )
    state = ep.initialize_cable_state(runtime, jnp.asarray([-65.0, -64.0, -63.0, -62.0]))
    inputs = ep.zero_cable_inputs(runtime)
    inputs = ep.CableStepInputs(
        jnp.asarray([0.2, 0.0, 0.0, 0.0]),
        inputs.synaptic_conductance_uS,
        inputs.synaptic_current_offset_nA,
        inputs.voltage_clamp_mask,
        inputs.voltage_clamp_target_mV,
    )
    evaluation = ep.evaluate_membrane_program(
        runtime.program,
        state.membrane,
        morphology,
        state.voltage_mV,
        state.intracellular_mM,
        state.extracellular_mM,
    )
    matrix, right, _, _ = ep.assemble_cable_system(runtime, state, evaluation, inputs)
    tree = ep.tree_elimination_solve(jnp.diag(matrix), right, morphology)
    dense = ep.differentiable_dense_solve(matrix, right)
    result = ep.step_cable(runtime, state, inputs)
    np.testing.assert_allclose(tree, dense, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(result.evidence.kirchhoff_residual_nA, 0.0, atol=2.0e-11)
    np.testing.assert_allclose(
        result.evidence.charge_balance_residual_nA, 0.0, atol=2.0e-11
    )


def test_native_linear_forward_and_reverse_derivatives_match_finite_difference():
    matrix = jnp.asarray([[3.0, -0.4], [-0.4, 2.0]])
    right = jnp.asarray([0.7, -1.1])
    direction_matrix = jnp.asarray([[0.2, 0.1], [-0.3, -0.1]])
    direction_right = jnp.asarray([-0.4, 0.25])
    weights = jnp.asarray([1.3, -0.8])

    def objective(matrix_, right_):
        return jnp.dot(weights, ep.differentiable_dense_solve(matrix_, right_))

    gradients = jax.grad(objective, argnums=(0, 1))(matrix, right)
    analytic = jnp.sum(gradients[0] * direction_matrix) + jnp.dot(
        gradients[1], direction_right
    )
    epsilon = 1.0e-6
    finite_difference = (
        objective(matrix + epsilon * direction_matrix, right + epsilon * direction_right)
        - objective(
            matrix - epsilon * direction_matrix, right - epsilon * direction_right
        )
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-7, atol=2.0e-8)
    _, forward_tangent = jax.jvp(
        objective,
        (matrix, right),
        (direction_matrix, direction_right),
    )
    np.testing.assert_allclose(
        forward_tangent,
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-8,
    )


def test_exact_gate_update_and_hh_current_affinity():
    gate = jnp.asarray([0.2, 0.7])
    steady = jnp.asarray([0.8, 0.1])
    tau = jnp.asarray([2.0, 5.0])
    updated = ep.exact_affine_gate_update(gate, steady, tau, jnp.asarray(0.4))

    np.testing.assert_allclose(
        updated, steady + (gate - steady) * np.exp(-0.4 / tau), rtol=1.0e-13
    )
    runtime = _single(ep.HodgkinHuxleyNaK(), dt=0.01)
    state = ep.initialize_cable_state(runtime, jnp.asarray([-55.0]))
    evaluation = ep.evaluate_membrane_program(
        runtime.program,
        state.membrane,
        runtime.morphology,
        state.voltage_mV,
        state.intracellular_mM,
        state.extracellular_mM,
    )
    voltage_a = -45.0
    voltage_b = 5.0
    current_a = evaluation.conductance_uS * voltage_a + evaluation.current_offset_nA
    current_b = evaluation.conductance_uS * voltage_b + evaluation.current_offset_nA
    np.testing.assert_allclose(
        current_b - current_a,
        evaluation.conductance_uS * (voltage_b - voltage_a),
        rtol=1.0e-13,
    )


def test_nonfinite_updated_hh_gates_reject_entire_cable_transition():
    runtime = _single(ep.HodgkinHuxleyNaK(), dt=0.1)
    state = ep.initialize_cable_state(runtime, jnp.asarray([-65.0]))
    inputs = ep.CableStepInputs(
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([True]),
        jnp.asarray([-1.0e6]),
    )
    result = ep.step_cable(runtime, state, inputs)
    assert not bool(result.evidence.successful)
    assert int(result.evidence.status) & int(ep.CableSolveStatus.NONFINITE)
    np.testing.assert_array_equal(result.state.voltage_mV, state.voltage_mV)
    np.testing.assert_array_equal(result.state.membrane.gates, state.membrane.gates)
    np.testing.assert_array_equal(result.state.step_index, state.step_index)
    np.testing.assert_array_equal(result.state.time_ms, state.time_ms)


def test_hodgkin_huxley_program_is_excitable_under_inward_current():
    runtime = _single(ep.HodgkinHuxleyNaK(), ep.PassiveLeak(0.0, -65.0), dt=0.02)
    state = ep.initialize_cable_state(runtime, jnp.asarray([-65.0]))
    neutral = ep.zero_cable_inputs(runtime)

    def advance(carry, index):
        current = jnp.where((index >= 50) & (index < 300), 10.0, 0.0)
        inputs = ep.CableStepInputs(
            jnp.asarray([current]),
            neutral.synaptic_conductance_uS,
            neutral.synaptic_current_offset_nA,
            neutral.voltage_clamp_mask,
            neutral.voltage_clamp_target_mV,
        )
        result = ep.step_cable(runtime, carry, inputs)
        return result.state, result.state.voltage_mV[0]

    _, voltage = jax.lax.scan(advance, state, jnp.arange(1000))
    assert float(jnp.max(voltage)) > 0.0
    assert float(voltage[-1]) < float(jnp.max(voltage)) - 20.0


def test_current_and_voltage_clamp_signs_are_explicit():
    runtime = _single(ep.PassiveLeak(0.0, -65.0), dt=0.1)
    state = ep.initialize_cable_state(runtime, jnp.asarray([-65.0]))
    neutral = ep.zero_cable_inputs(runtime)
    inward = ep.CableStepInputs(
        jnp.asarray([1.0]),
        neutral.synaptic_conductance_uS,
        neutral.synaptic_current_offset_nA,
        neutral.voltage_clamp_mask,
        neutral.voltage_clamp_target_mV,
    )
    raised = ep.step_cable(runtime, state, inward)
    assert float(raised.state.voltage_mV[0]) > -65.0
    clamped_inputs = ep.CableStepInputs(
        jnp.asarray([0.0]),
        neutral.synaptic_conductance_uS,
        neutral.synaptic_current_offset_nA,
        jnp.asarray([True]),
        jnp.asarray([-30.0]),
    )
    clamped = ep.step_cable(runtime, state, clamped_inputs)
    np.testing.assert_allclose(clamped.state.voltage_mV, [-30.0], atol=1.0e-13)
    assert float(clamped.evidence.clamp_current_nA[0]) > 0.0


def test_synapse_network_preserves_exact_voltage_affinity_and_delay():
    plan = ep.SynapseNetworkPlan(
        2,
        2,
        3,
        2,
        0.1,
        connections=(
            ep.SynapseConnection(
                "exc",
                0,
                0,
                1,
                0,
                ep.ConductanceSynapse(3.0, 0.5, 0.0),
                delay_steps=1,
                weight=2.0,
            ),
            ep.SynapseConnection(
                "bias", 0, 1, 1, 1, ep.CurrentSynapse(2.0, -0.3), weight=1.0
            ),
        ),
    )
    runtime = plan.prepare()
    state = ep.initialize_synapse_network(runtime)
    spikes = jnp.asarray([[1.0, 1.0], [0.0, 0.0]])
    first = ep.evaluate_synapse_network_transition(runtime, state, spikes)
    state = ep.commit_synapse_network_transition(first, state)
    assert float(first.evidence.current_offset_nA[1, 1]) < 0.0
    second = ep.evaluate_synapse_network_transition(
        runtime, state, jnp.zeros_like(spikes)
    )
    conductance = second.evidence.conductance_uS[1, 0]
    offset = second.evidence.current_offset_nA[1, 0]
    assert float(conductance) > 0.0
    np.testing.assert_allclose(offset, -conductance * 0.0, atol=1.0e-14)
    voltage = -60.0
    assert float(conductance * voltage + offset) < 0.0


def test_dynamic_synaptogenesis_and_pair_stdp_are_candidate_commit_transitions():
    runtime = ep.SynapseNetworkPlan(2, 1, 2, 0, 1.0).prepare()
    relations = ep.initialize_synapse_network(runtime)
    event = ep.SynapseRelationEvent(
        int(ep.SynapseRelationEventKind.ACTIVATE),
        -1,
        0,
        0,
        1,
        0,
        int(ep.SynapseKind.CONDUCTANCE),
        0.4,
        0.2,
        0.0,
        5.0,
        0,
    )
    candidate = ep.evaluate_synapse_relation_event(runtime, relations, event)
    assert bool(candidate.successful)
    assert int(jnp.sum(relations.active)) == 0
    plasticity = ep.initialize_pair_stdp(runtime)
    relations, plasticity = ep.commit_synapse_relation_event_with_plasticity(
        candidate,
        relations,
        plasticity,
    )
    assert int(jnp.sum(relations.active)) == 1
    stdp_plan = ep.PairSTDPPlan(20.0, 20.0, 0.1, 0.05, 0.0, 1.0, trace_bound=1.5)
    pre = jnp.asarray([[1.0], [0.0]])
    post = jnp.asarray([[0.0], [0.0]])
    first = ep.evaluate_pair_stdp(runtime, stdp_plan, relations, plasticity, pre, post)
    relations, plasticity = ep.commit_pair_stdp(first, relations, plasticity)
    repeated_pre = ep.evaluate_pair_stdp(
        runtime,
        stdp_plan,
        relations,
        plasticity,
        pre,
        post,
    )
    relations, plasticity = ep.commit_pair_stdp(
        repeated_pre,
        relations,
        plasticity,
    )
    np.testing.assert_allclose(plasticity.pre_trace[0], 1.0, atol=1.0e-14)
    second = ep.evaluate_pair_stdp(
        runtime,
        stdp_plan,
        relations,
        plasticity,
        jnp.zeros_like(pre),
        jnp.asarray([[0.0], [1.0]]),
    )
    assert float(second.relations.weight[0]) > float(relations.weight[0])
    assert bool(second.evidence.trace_bound_satisfied)
    assert bool(second.evidence.weight_bound_satisfied)
    full_runtime = ep.SynapseNetworkPlan(
        2,
        1,
        1,
        0,
        1.0,
        connections=(
            ep.SynapseConnection(
                "occupied",
                0,
                0,
                1,
                0,
                ep.ConductanceSynapse(5.0, 0.2, 0.0),
            ),
        ),
    ).prepare()
    full_state = ep.initialize_synapse_network(full_runtime)
    rejected = ep.evaluate_synapse_relation_event(full_runtime, full_state, event)
    assert not bool(rejected.successful)
    assert int(rejected.status) == int(ep.SynapseStatus.CAPACITY_EXCEEDED)


def test_concentration_transfer_conserves_moles_charge_and_nernst_sign():
    runtime = ep.IonDynamicsPlan(
        (ep.IonSpecies("Na", 1), ep.IonSpecies("K", 1)),
        (2.0, 3.0),
        (10.0, 12.0),
        conservation_tolerance_mol=1.0e-18,
        charge_tolerance_C=1.0e-15,
    ).prepare()
    state = ep.initialize_ion_concentrations(
        runtime,
        jnp.asarray([[12.0, 13.0], [130.0, 125.0]]),
        jnp.asarray([[145.0, 140.0], [4.0, 5.0]]),
    )
    currents = jnp.asarray([[0.02, -0.01], [-0.03, 0.01]])
    candidate = ep.evaluate_ion_concentration_transition(
        runtime, state, currents, jnp.asarray(0.1)
    )
    assert bool(candidate.evidence.successful)
    np.testing.assert_allclose(
        candidate.evidence.conservation_residual_mol, 0.0, atol=2.0e-20
    )
    np.testing.assert_allclose(
        candidate.evidence.intracellular_charge_residual_C, 0.0, atol=2.0e-19
    )
    accepted = ep.commit_ion_concentration_transition(candidate, state)
    assert float(accepted.intracellular_mM[0, 0]) < float(state.intracellular_mM[0, 0])
    potential = ep.nernst_potential_mV(runtime, state)
    assert float(potential[0, 0]) > 0.0
    assert float(potential[1, 0]) < 0.0
    integer_candidate = ep.evaluate_ion_concentration_transition(
        runtime,
        state,
        jnp.ones(state.intracellular_mM.shape, dtype=jnp.int32),
        jnp.asarray(0.1),
    )
    assert bool(integer_candidate.evidence.successful)
    assert jnp.issubdtype(
        integer_candidate.proposed.intracellular_mM.dtype,
        jnp.inexact,
    )
    assert not np.array_equal(
        integer_candidate.proposed.intracellular_mM,
        state.intracellular_mM,
    )
    with pytest.raises(ValueError, match="scalar"):
        ep.evaluate_ion_concentration_transition(
            runtime,
            state,
            currents,
            jnp.asarray([0.1, 0.1]),
        )


def test_stochastic_channel_counts_are_reproducible_and_key_lineage_is_explicit():
    generator = jnp.asarray([[-0.2, 0.2], [0.1, -0.1]])
    runtime = ep.MarkovChannelPlan(generator, 3).prepare(0.5)
    state_a = ep.initialize_stochastic_channels(
        runtime,
        jnp.asarray([[80, 20], [50, 50], [100, 0]], dtype=jnp.int32),
        jax.random.key(91),
    )
    state_b = ep.initialize_stochastic_channels(
        runtime, state_a.counts, jax.random.key(91)
    )
    first = ep.evaluate_stochastic_channel_transition(runtime, state_a)
    second = ep.evaluate_stochastic_channel_transition(runtime, state_b)
    np.testing.assert_array_equal(first.proposed.counts, second.proposed.counts)
    np.testing.assert_array_equal(
        jax.random.key_data(first.lineage.parent_key),
        jax.random.key_data(state_a.key),
    )
    np.testing.assert_array_equal(
        jax.random.key_data(first.lineage.next_key),
        jax.random.key_data(first.proposed.key),
    )
    overflowing = np.asarray(
        [[2**31, 0], [1, 0], [1, 0]],
        dtype=np.uint32,
    )
    with pytest.raises(ValueError, match="int32"):
        ep.initialize_stochastic_channels(
            runtime,
            overflowing,
            jax.random.key(91),
        )
    np.testing.assert_array_equal(
        first.evidence.counts_before, first.evidence.counts_after
    )
    assert bool(first.evidence.successful)


def test_swc_parser_validates_topology_and_reports_stable_mapping():
    text = """
    # id type x y z radius parent
    1 1 0 0 0 5 -1
    2 3 10 0 0 1 1
    3 3 20 5 0 0.8 2
    4 3 20 -5 0 0.8 2
    """
    first = ep.parse_swc_text(text, "swc-cell")
    second = ep.parse_swc_text(text, "swc-cell")
    assert first.report.source_id == second.report.source_id
    assert first.report.status == AdapterStatus.DECLARED_LOSS
    assert {loss.path for loss in first.report.losses} == {
        "/nodes/*/absolute_xyz",
        "/nodes/*/type",
        "/root/length_um",
        "/nodes/*/radius",
    }
    assert first.evidence.stable_mapping == (
        (1, "swc-1"),
        (2, "swc-2"),
        (3, "swc-3"),
        (4, "swc-4"),
    )
    assert first.evidence.branch_count == 3
    with pytest.raises(AdapterError):
        require_lossless(first.report)
    translated = ep.parse_swc_text(
        text.replace("10 0 0", "11 0 0")
        .replace("20 5 0", "21 5 0")
        .replace("20 -5 0", "21 -5 0"),
        "swc-cell",
    )
    assert translated.report.source_id != first.report.source_id
    with pytest.raises(ValueError, match="missing parent"):
        ep.parse_swc_text("1 1 0 0 0 1 -1\n2 3 1 0 0 1 9", "bad")
    with pytest.raises(ValueError, match="exactly one"):
        ep.parse_swc_text("1 1 0 0 0 1 -1\n2 1 1 0 0 1 -1", "bad")


def test_rejected_cable_step_does_not_create_a_valid_recording():
    runtime = _single(
        ep.SodiumPotassiumPump(0.05, 10.0, 1.5),
        dt=0.1,
    )
    cable_state = ep.initialize_cable_state(
        runtime,
        jnp.asarray([-65.0]),
        intracellular_mM=jnp.asarray([[-1.0], [140.0]]),
        extracellular_mM=jnp.asarray([[145.0], [4.0]]),
    )
    protocol = ep.ElectrophysiologyProtocol(ep.RecordingPlan(("soma",), 2)).prepare(
        runtime
    )
    state = ep.initialize_experiment(protocol, cable_state)
    result = ep.step_experiment(protocol, state)
    assert not bool(result.cable.evidence.successful)
    assert int(result.recording_status) == int(ep.RecordingStatus.REJECTED_CABLE_STEP)
    assert int(result.state.recording.count) == 0
    assert not bool(jnp.any(result.state.recording.valid))


def test_checkpoint_replay_matches_uninterrupted_identity():
    runtime = _single(ep.PassiveLeak(0.2, -65.0), dt=0.1)
    protocol = ep.ElectrophysiologyProtocol(
        ep.RecordingPlan(("soma",), 20),
        current_clamps=(ep.CurrentClamp("pulse", "soma", 0.5, 0.2, 0.7),),
    ).prepare(runtime)
    initial = ep.initialize_experiment(
        protocol, ep.initialize_cable_state(runtime, jnp.asarray([-65.0]))
    )
    prefix = ep.run_experiment(protocol, initial, 4)
    checkpoint = ep.checkpoint_experiment(protocol, prefix.state)
    replay = ep.replay_experiment(protocol, checkpoint, 6)
    direct = ep.run_experiment(protocol, initial, 10)
    np.testing.assert_array_equal(
        replay.state.cable.voltage_mV, direct.state.cable.voltage_mV
    )
    np.testing.assert_array_equal(
        replay.state.recording.voltage_mV, direct.state.recording.voltage_mV
    )
    np.testing.assert_array_equal(
        replay.state.recording.valid, direct.state.recording.valid
    )
