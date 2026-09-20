#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Representative jets, trigger replay, neutrino, flavor, and bubble workflows."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    analysis = phx.applications.collider_analysis
    jet_inputs = analysis.JetInputBatch(
        jnp.asarray([1]),
        jnp.asarray(
            [[[10.0, 6.0, 0.0, 8.0], [5.0, 3.0, 0.0, 4.0], [4.0, -4.0, 0.0, 0.0]]]
        ),
        jnp.ones((1, 3), dtype="bool"),
        source_collection_id="particles",
        momentum_unit_id="GeV",
    )
    jets = analysis.cluster_sequential_jets(
        analysis.JetDefinition(analysis.JetAlgorithm.ANTI_KT, 0.5), jet_inputs
    )

    trigger = phx.applications.trigger_replay
    menu = trigger.TriggerMenu(
        (
            trigger.TriggerLine(
                "physics",
                stream_names=("physics",),
                maximum_latency=1.0,
                maximum_resource_units=1.0,
            ),
        )
    )
    trigger_result = trigger.replay_trigger_menu(
        menu, jnp.asarray([1]), jnp.asarray([[True]])
    )

    neutrino = phx.applications.neutrino
    probabilities = neutrino.oscillation_probabilities(
        neutrino.NeutrinoOscillationParameters(
            theta12=0.59,
            theta13=0.15,
            theta23=0.78,
            delta_cp=-1.2,
            delta_m21_squared=7.5e-5,
            delta_m31_squared=2.5e-3,
            ordering=neutrino.NeutrinoMassOrdering.NORMAL,
        ),
        jnp.asarray([1.0]),
        jnp.asarray([295.0]),
        matter_density_g_cm3=2.6,
    )

    flavor = phx.applications.flavor_physics
    amplitude = flavor.evaluate_coherent_amplitude(
        flavor.CoherentAmplitudePlan(
            ("a", "b"),
            phase_convention_id="example",
            normalization_evidence_id="example-normalization",
        ),
        jnp.asarray([[1.0 + 0.0j, 0.0 + 1.0j]]),
        jnp.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
        jnp.asarray([1.0]),
    )

    phase = phx.applications.cosmology.phase_transitions
    bubble_plan = phase.ThinWallBubblePlan(
        surface_tension=1000.0,
        vacuum_energy_difference=0.0,
        inside_mass=1.0,
        outside_mass=100.0,
        time_step=0.01,
        maximum_steps=1,
        maximum_energy_residual=100.0,
    )
    bubble_state = phase.prepare_thin_wall_bubble(
        bubble_plan,
        1.0,
        0.0,
        phase.BubbleParticleEnsemble(
            jnp.asarray([[0.9, 0.0, 0.0]]),
            jnp.asarray([[10.0, 0.0, 0.0]]),
            jnp.asarray([1.0]),
            jnp.asarray([True]),
            jnp.asarray([True]),
        ),
    )
    bubble = phase.step_thin_wall_bubble(bubble_plan, bubble_state)

    if (
        not bool(jets.valid[0, 0])
        or not bool(trigger_result.successful)
        or not bool(probabilities.valid[0])
        or not bool(amplitude.valid)
        or not bool(bubble.accepted)
    ):
        raise RuntimeError(
            "Expanded HEP scientific workflow failed its declared support."
        )
    print(
        {
            "jet_count": int(jnp.sum(jets.active)),
            "trigger_accept": bool(trigger_result.post_prescale[0, 0]),
            "oscillation_unitarity": float(probabilities.row_unitarity_residual[0]),
            "amplitude_normalization": float(amplitude.normalization),
            "bubble_reflections": int(jnp.sum(bubble.reflected)),
        }
    )


if __name__ == "__main__":
    main()
