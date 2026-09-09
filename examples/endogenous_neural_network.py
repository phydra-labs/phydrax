#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Drive an HH action potential through a delayed plastic synapse into LIF."""

import jax.numpy as jnp
import numpy as np

from phydrax.applications import electrophysiology as ep


def main() -> None:
    diameter = float(np.sqrt(100_000.0 / np.pi))
    morphology = ep.CellMorphologyPlan(
        "hh-soma", (ep.CompartmentSpec("soma", None, diameter, diameter),)
    ).prepare()
    hh = ep.CableSolverPlan(0.05, residual_tolerance=1.0e-8).prepare(
        morphology,
        ep.MembraneProgram((ep.HodgkinHuxleyNaK(),)),
    )
    target = ep.LeakyIntegrateAndFire(0.2, 0.01, -65.0, -50.0, -65.0, refractory_ms=2.0)
    synapses = ep.SynapseNetworkPlan(
        (1, 1),
        1,
        0.4,
        0.05,
        execution="event",
        connections=(
            ep.SynapseConnection(
                "hh-to-lif",
                0,
                0,
                1,
                0,
                ep.CurrentSynapse(3.0, -2.0),
                delay_ms=0.4,
                weight=1.5,
            ),
        ),
    )
    runtime = ep.NeuralNetworkPlan(
        (
            ep.NeuralCellPlan("hh", hh, threshold_mV=0.0, rearm_mV=-40.0),
            ep.NeuralCellPlan("target", target),
        ),
        synapses,
        learning=ep.PairSTDPPlan(20.0, 20.0, 0.02, 0.01, 0.0, 3.0),
        current_clamps=(("hh", ep.CurrentClamp("stimulus", "soma", 10.0, 0.0, 4.0)),),
        queue_capacity=32,
        spike_capacity=64,
        recording_capacity=240,
        maximum_events_per_step=64,
        root_subdivisions=2,
    ).prepare()
    first = ep.run_neural_network(runtime, ep.initialize_neural_network(runtime), 120)
    if bool(jnp.any(first.status != 0)):
        raise RuntimeError(f"First neural segment rejected: {first.status}")
    checkpoint = ep.checkpoint_neural_network(runtime, first.state)
    final = ep.run_neural_network(
        runtime, ep.restore_neural_network(runtime, checkpoint), 120
    )
    if bool(jnp.any(final.status != 0)):
        raise RuntimeError(f"Continued neural segment rejected: {final.status}")
    endpoints = final.state.spikes.endpoint[: final.state.spikes.count]
    if not bool(jnp.any(endpoints == 0)) or not bool(jnp.any(endpoints == 1)):
        raise RuntimeError(
            "Expected both an endogenous HH spike and a postsynaptic spike; "
            f"observed endpoints {endpoints.tolist()}."
        )
    print("spike times (ms):", final.state.spikes.time_ms[: final.state.spikes.count])
    print("spike endpoints:", endpoints)
    print("accepted synaptic weight:", final.state.relations.weight[0])
    print("final physical voltages (mV):", ep.neural_voltage(runtime, final.state))


if __name__ == "__main__":
    main()
