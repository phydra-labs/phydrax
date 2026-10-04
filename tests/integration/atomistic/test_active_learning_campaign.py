#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _provider() -> Any:
    def evaluate(system: Any, positions: Any, cell: Any) -> Any:
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


def _frame(system: Any, positions: Any, source: Any) -> Any:
    return phx.atomistic.AtomisticFrame(
        0.0,
        0,
        positions,
        system.plan.particle_ids,
        system_id=system.prepared_id,
        topology_id=system.topology.topology_id,
        units=system.plan.units,
        source_id=source,
    )


def _acquisition(frame: Any, plan_id: Any = "seed-acquisition") -> Any:
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


def test_campaign_labels_retrains_qualifies_and_promotes() -> None:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[1, 1],
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

    def qualify(committee: Any) -> Any:
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
    lineage = result.state.labels.lineage
    assert lineage.revision_id == result.state.labels.revision.revision_id
    assert lineage.parent_revision_id == labels.revision.revision_id
    assert lineage.parent_lineage_id == labels.lineage.lineage_id
    assert labels.lineage.parent_lineage_id is None
    assert result.state.labels.revision.semantic_id == labels.revision.semantic_id
    assert result.state.committee is not None
    assert result.lifecycle.run.status == "completed"
    assert len(result.lifecycle.models) == 2
    assert len({model.numeric_revision_id for model in result.lifecycle.models}) == 2
    assert all(
        result.state.labels.label_set_id in model.association_ids
        for model in result.lifecycle.models
    )
    assert (
        result.lifecycle.run.numeric_revision_id
        == result.state.labels.revision.revision_id
    )


def _stress_provider(provider_id: str, *, with_stress: bool) -> Any:
    def evaluate(system: Any, positions: Any, cell: Any) -> Any:
        del system
        energy = jnp.sum(positions * positions)
        stress = None
        if with_stress:
            stress = 0.01 * energy * jnp.eye(3) + 0.002 * (cell + cell.T)
        return phx.atomistic.ExternalAtomisticEvaluation(
            energy, -2.0 * positions, stress, jnp.asarray(True), provider_id
        )

    return phx.atomistic.CallableBornOppenheimerProvider(evaluate, provider_id)


def test_campaign_retains_provider_stress_through_training_and_promotion() -> None:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    cell = 3.2 * jnp.eye(3)
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 8],
        # ty: ignore[invalid-argument-type]
        [1.0, 16.0],
        units,
        cell=phx.discretization.PeriodicCell(cell),
    ).prepare()
    stress_provider = _stress_provider("stress-reference", with_stress=True)

    def frame(positions: Any, source: str) -> Any:
        return phx.atomistic.AtomisticFrame(
            0.0,
            0,
            jnp.asarray(positions),
            system.plan.particle_ids,
            cell_vectors=cell,
            system_id=system.prepared_id,
            topology_id=system.topology.topology_id,
            units=system.plan.units,
            source_id=source,
        )

    unstressed_label = phx.atomistic.label_atomistic_acquisitions(
        system,
        _stress_provider("energy-force-reference", with_stress=False),
        (_acquisition(frame([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]], "seed")),),
        split="train",
    )[0]
    validation_label = phx.atomistic.label_atomistic_acquisitions(
        system,
        stress_provider,
        (
            _acquisition(
                frame([[0.0, 0.0, 0.0], [1.1, 0.0, 0.3]], "validation"),
                "validation-acquisition",
            ),
        ),
        split="validation",
    )[0]
    labels = phx.atomistic.AtomisticLabelSet((unstressed_label, validation_label))
    image_capacity = phx.discretization.ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=1024,
        maximum_degree=64,
        maximum_images=125,
    )
    graph = phx.atomistic.AtomisticGraphExecutionPlan(
        64, backend="particle", image_capacity=image_capacity
    )
    campaign = phx.atomistic.AtomisticLearningCampaignPlan(
        system,
        stress_provider,
        phx.atomistic.AcquisitionPlan(
            1, phx.atomistic.CommitteeAcquisitionScorePolicy(1.0, 1.0, 1.0)
        ),
        graph,
        phx.atomistic.AtomisticGraphExecutionPlan(
            64, backend="particle", image_capacity=image_capacity
        ),
        phx.atomistic.AtomisticTrainingPolicy(maximum_steps=1, learning_rate=1e-3),
        phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0),
    )
    candidate = frame([[0.0, 0.0, 0.0], [1.2, 0.4, 0.1]], "candidate")
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

    def qualify(committee: Any) -> Any:
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
        phx.nn.atomistic.MACEPotential(
            units.scale,
            phx.nn.atomistic.MACEArchitecture(
                species=(1, 8),
                cutoff=3.0,
                radial_basis_count=4,
                cutoff_power=5,
                channel_count=4,
                hidden_degree=1,
                edge_degree=2,
                interactions=("real-agnostic", "real-agnostic-residual"),
                correlations=(2, 2),
                radial_widths=(8,),
                readout_width=4,
                average_neighbor_count=2.0,
            ),
            # ty: ignore[invalid-argument-type]
            atomic_energies=[[-1.0, -2.0]],
            key=jax.random.key(seed),
        )
        for seed in (3, 4)
    )
    result = phx.atomistic.run_atomistic_campaign_round(
        campaign,
        phx.atomistic.AtomisticLearningCampaignState(labels),
        (candidate,),
        (uncertainty,),
        models,
        (jax.random.key(10), jax.random.key(11)),
        qualify,
    )

    assert bool(result.successful & result.promoted)
    acquired = result.labels[0]
    assert acquired.stress is not None
    problem = result.state.labels.training_problem(system, graph, cutoff=3.0)
    assert problem.training.stress is not None
    assert problem.training.stress_mask is not None
    assert problem.validation is not None
    assert problem.validation.stress is not None
    np.testing.assert_array_equal(problem.training.stress[1], acquired.stress)
    np.testing.assert_array_equal(
        problem.training.stress_mask,
        np.broadcast_to(np.asarray([False, True])[:, None, None], (2, 3, 3)),
    )
    np.testing.assert_array_equal(problem.validation.stress[0], validation_label.stress)
    for training, model in zip(
        result.training_results, result.lifecycle.models, strict=True
    ):
        assert training.problem_id == problem.problem_id
        assert training.stress_loss_history.shape == (1,)
        assert bool(jnp.isfinite(training.stress_loss_history[0]))
        assert float(training.stress_loss_history[0]) > 0.0
        assert training.problem_id in model.association_ids
        assert training.capabilities_id in model.association_ids
