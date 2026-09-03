#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _provider():
    def evaluate(system, positions, cell):
        del system, cell
        energy = jnp.sum(positions * positions)
        return phx.atomistic.ExternalAtomisticEvaluation(
            energy,
            -2.0 * positions,
            None,
            jnp.asarray(True),
            "harmonic-reference",
        )

    return phx.atomistic.CallableBornOppenheimerProvider(evaluate, "harmonic-reference")


def _frame(system, positions, source):
    return phx.atomistic.AtomisticFrame(
        0.0,
        0,
        positions,
        system.plan.particle_ids,
        system_id=system.prepared_id,
        topology_id=system.topology.topology_id,
        unit_system_id=system.plan.units.unit_system_id,
        source_id=source,
    )


def _acquisition(frame, plan_id="seed-acquisition"):
    return phx.atomistic.AcquisitionRecord(
        frame=frame,
        descriptor=frame.positions.reshape((-1,)),
        component_scores=jnp.asarray([1.0, 0.0, 0.0]),
        source_index=0,
        score=1.0,
        reason="seed",
        model_id="seed-committee",
        plan_id=plan_id,
    )


def test_campaign_labels_retrains_qualifies_and_promotes():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20], [1, 1], [1.0, 1.0], units, atom_type_ids=[1, 1]
    ).prepare()
    provider = _provider()
    training_frame = _frame(
        system, jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), "training"
    )
    validation_frame = _frame(
        system, jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]), "validation"
    )
    training_label = phx.atomistic.label_atomistic_acquisitions(
        system, provider, (_acquisition(training_frame),), split="train"
    )[0]
    validation_label = phx.atomistic.label_atomistic_acquisitions(
        system,
        provider,
        (_acquisition(validation_frame, "validation-acquisition"),),
        split="validation",
    )[0]
    labels = phx.atomistic.AtomisticLabelSet((training_label, validation_label))
    state = phx.atomistic.AtomisticLearningCampaignState(labels)
    graph = phx.atomistic.AtomisticGraphExecutionPlan(
        1, backend="dense", maximum_dense_atoms=2
    )
    runtime_graph = phx.atomistic.AtomisticGraphExecutionPlan(1, backend="particle")
    reduction = phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0)
    campaign = phx.atomistic.AtomisticLearningCampaignPlan(
        system,
        provider,
        phx.atomistic.AcquisitionPlan(
            1, phx.atomistic.CommitteeAcquisitionScorePolicy(1.0, 1.0, 1.0)
        ),
        graph,
        runtime_graph,
        phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0),
        reduction,
    )
    candidate_frame = _frame(
        system, jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]), "candidate"
    )
    uncertainty = phx.atomistic.AtomisticUncertaintyEvidence(
        jnp.asarray(2.0),
        jnp.zeros((2, 3)),
        jnp.zeros((2,)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        jnp.asarray(True),
        "seed-committee",
    )

    def qualify(committee):
        evidence = phx.atomistic.AtomisticDynamicsClaimEvidence(
            phx.atomistic.AtomisticDynamicsQualificationClaim.FINITE_EXECUTION,
            committee.committee_id,
            True,
        )
        return phx.atomistic.AtomisticDynamicsQualificationResult(
            phx.discretization.ParticleMethodMaturity.EXPERIMENTAL,
            phx.atomistic.AtomisticDynamicsQualificationProfile(),
            (evidence,),
            True,
        )

    models = tuple(
        phx.nn.atomistic.PaiNNPotential(
            units.scale,
            cutoff=2.5,
            feature_count=4,
            interaction_count=1,
            radial_basis_count=3,
            maximum_species_id=1,
            key=jax.random.key(seed),
        )
        for seed in (3, 4)
    )
    result = phx.atomistic.run_atomistic_campaign_round(
        campaign,
        state,
        (candidate_frame,),
        (uncertainty,),
        models,
        (jax.random.key(10), jax.random.key(11)),
        qualify,
    )

    assert bool(result.successful & result.promoted)
    assert result.state.round_index == 1
    assert len(result.state.labels.records) == 3
    assert result.state.labels.revision.parent_digest == labels.revision.content_digest
    assert result.state.committee is not None
    assert result.lifecycle.run.status == "completed"
    assert len(result.lifecycle.models) == 2
    assert (
        result.lifecycle.run.numeric_revision_id
        == result.state.labels.revision.revision_id
    )
