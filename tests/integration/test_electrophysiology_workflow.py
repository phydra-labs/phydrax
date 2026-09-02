#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications import electrophysiology as ep


jax.config.update("jax_enable_x64", True)


def test_branched_multicell_ion_channel_plasticity_workflow_is_jittable():
    swc = ep.parse_swc_text(
        """
        1 1 0 0 0 6 -1
        2 3 20 0 0 2 1
        3 3 60 15 0 1 2
        4 3 60 -15 0 1 2
        """,
        "workflow-cell",
    )
    morphology = swc.morphology.prepare()
    program = ep.MembraneProgram(
        (
            ep.PassiveLeak(0.3, -65.0),
            ep.HodgkinHuxleyNaK(),
            ep.SodiumPotassiumPump(0.03, 10.0, 1.5),
        )
    )
    cable = ep.CableSolverPlan(
        0.025,
        scheme="crank-nicolson",
        residual_tolerance=1.0e-9,
    ).prepare(morphology, program)
    ion_runtime = ep.IonDynamicsPlan(
        (ep.IonSpecies("Na", 1), ep.IonSpecies("K", 1)),
        (0.8, 0.5, 0.3, 0.3),
        (8.0, 5.0, 3.0, 3.0),
    ).prepare()
    intracellular = jnp.asarray([[12.0] * 4, [140.0] * 4])
    extracellular = jnp.asarray([[145.0] * 4, [4.0] * 4])
    ion_states = tuple(
        ep.initialize_ion_concentrations(ion_runtime, intracellular, extracellular)
        for _ in range(2)
    )
    ion_state = jax.tree.map(lambda *values: jnp.stack(values), *ion_states)
    cable_states = tuple(
        ep.initialize_cable_state(
            cable,
            jnp.asarray([-65.0, -65.0, -65.0, -65.0]),
            intracellular_mM=intracellular,
            extracellular_mM=extracellular,
        )
        for _ in range(2)
    )
    cable_state = jax.tree.map(lambda *values: jnp.stack(values), *cable_states)
    network = ep.SynapseNetworkPlan(
        2,
        4,
        4,
        2,
        0.025,
        connections=(
            ep.SynapseConnection(
                "cell-0-to-cell-1",
                0,
                0,
                1,
                2,
                ep.ConductanceSynapse(3.0, 0.03, 0.0),
                delay_steps=1,
            ),
            ep.SynapseConnection(
                "cell-1-to-cell-0",
                1,
                0,
                0,
                3,
                ep.CurrentSynapse(4.0, -0.01),
                delay_steps=2,
            ),
        ),
    ).prepare()
    relation_state = ep.initialize_synapse_network(network)
    plasticity_plan = ep.PairSTDPPlan(20.0, 20.0, 0.005, 0.004, 0.1, 2.0)
    plasticity_state = ep.initialize_pair_stdp(network)
    channel = ep.MarkovChannelPlan(
        jnp.asarray([[-0.15, 0.15], [0.05, -0.05]]),
        2,
    ).prepare(0.025)
    channel_state = ep.initialize_stochastic_channels(
        channel,
        jnp.asarray([[900, 100], [800, 200]], dtype=jnp.int32),
        jax.random.key(2026),
    )

    def one_step(carry, index):
        cells, ions, relations, plasticity, channels = carry
        spikes = (
            jnp.zeros((2, 4)).at[index % 2, 0].set((index % 7 == 0).astype(jnp.float64))
        )
        synapse_candidate = ep.evaluate_synapse_network_transition(
            network, relations, spikes
        )
        propagated_relations = ep.commit_synapse_network_transition(
            synapse_candidate, relations
        )
        plasticity_candidate = ep.evaluate_pair_stdp(
            network,
            plasticity_plan,
            propagated_relations,
            plasticity,
            spikes,
            jnp.roll(spikes, 1, axis=0),
        )
        relations_next, plasticity_next = ep.commit_pair_stdp(
            plasticity_candidate,
            propagated_relations,
            plasticity,
        )
        zeros = jnp.zeros((2, 4))
        injected = zeros.at[0, 0].set(jnp.where(index < 10, 0.2, 0.0))
        inputs = ep.CableStepInputs(
            injected,
            synapse_candidate.evidence.conductance_uS,
            synapse_candidate.evidence.current_offset_nA,
            jnp.zeros((2, 4), dtype=bool),
            zeros,
        )
        cable_results = jax.vmap(
            lambda state, input_: ep.step_cable(cable, state, input_)
        )(cells, inputs)
        pump_ion_current = jax.vmap(
            lambda current: ep.sodium_potassium_pump_ion_currents(current)
        )(cable_results.membrane_evaluation.nonlinear_current_nA)
        ion_candidates = jax.vmap(
            lambda state, current: ep.evaluate_ion_concentration_transition(
                ion_runtime, state, current, jnp.asarray(0.025)
            )
        )(ions, pump_ion_current)
        ions_next = jax.vmap(ep.commit_ion_concentration_transition)(ion_candidates, ions)
        cells_next = ep.CableState(
            cable_results.state.voltage_mV,
            cable_results.state.membrane,
            ions_next.intracellular_mM,
            ions_next.extracellular_mM,
            cable_results.state.time_ms,
            cable_results.state.step_index,
        )
        channel_candidate = ep.evaluate_stochastic_channel_transition(channel, channels)
        channels_next = ep.commit_stochastic_channel_transition(
            channel_candidate, channels
        )
        evidence = (
            cable_results.evidence.successful,
            ion_candidates.evidence.successful,
            synapse_candidate.successful,
            plasticity_candidate.successful,
            channel_candidate.evidence.successful,
        )
        return (
            cells_next,
            ions_next,
            relations_next,
            plasticity_next,
            channels_next,
        ), evidence

    @jax.jit
    def run(initial):
        return jax.lax.scan(one_step, initial, jnp.arange(24))

    final, evidence = run(
        (cable_state, ion_state, relation_state, plasticity_state, channel_state)
    )
    assert bool(jnp.all(evidence[0]))
    assert bool(jnp.all(evidence[1]))
    assert bool(jnp.all(evidence[2]))
    assert bool(jnp.all(evidence[3]))
    assert bool(jnp.all(evidence[4]))
    assert final[0].voltage_mV.shape == (2, 4)
    assert final[1].intracellular_mM.shape == (2, 2, 4)
    assert final[2].active.shape == (4,)
    np.testing.assert_array_equal(
        np.sum(final[4].counts, axis=1),
        np.asarray([1000, 1000]),
    )
    assert swc.evidence.morphology_id == morphology.plan.plan_id
