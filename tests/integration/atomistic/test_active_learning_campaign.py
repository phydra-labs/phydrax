#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests._support.atomistic_bundles import (
    neighborhood_bundle_plan,
    role_capacity,
    search_capacity,
)
from tests._support.mace_deployment import bundle_resources


def _provider(calls: list[int] | None = None, *, successful: bool = True) -> Any:
    def evaluate(system: Any, positions: Any, cell: Any) -> Any:
        del system, cell
        if calls is not None:
            calls.append(1)
        energy = jnp.sum(positions * positions)
        return phx.atomistic.ExternalAtomisticEvaluation(
            energy,
            -2.0 * positions,
            None,
            jnp.asarray(successful),
            provider_label="harmonic-reference",
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


def _uncertainty() -> Any:
    return phx.atomistic.AtomisticUncertaintyEvidence(
        jnp.asarray(2.0),
        jnp.zeros((2, 3)),
        jnp.zeros((2,)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        jnp.asarray(True),
        "seed-committee",
    )


def _qualify(committee: Any) -> Any:
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


def _finite_system() -> Any:
    return phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[1, 1],
    ).prepare()


def _painn(system: Any, seed: int, *, cutoff: float = 2.5) -> Any:
    return phx.nn.atomistic.PaiNNPotential(
        system.plan.units.scale,
        cutoff=cutoff,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        maximum_species_id=1,
        key=jax.random.key(seed),
    )


def _finite_resources(model: Any) -> Any:
    """Explicit learned-graph resources of a finite pair in a nonperiodic frame."""
    return neighborhood_bundle_plan(
        phx.atomistic.LearnedGraphPotentialTerm(model).requirements.neighborhoods,
        search=phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(8, particles_per_cell=2, images=27),),
        role_capacities=(role_capacity(4, 4),),
        frame=phx.discretization.PeriodicCell(
            20.0 * np.eye(3), origin=np.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )


def _energy_force_objectives() -> tuple[Any, ...]:
    return (
        phx.atomistic.AtomisticEnergyObjective(),
        phx.atomistic.AtomisticForceObjective(),
    )


def _finite_campaign(
    system: Any, provider: Any, models: tuple[Any, ...], *, maximum_steps: int = 0
) -> Any:
    return phx.atomistic.AtomisticLearningCampaignPlan(
        system,
        provider,
        phx.atomistic.AcquisitionPlan(
            1, phx.atomistic.CommitteeAcquisitionScorePolicy(1.0, 1.0, 1.0)
        ),
        phx.atomistic.AtomisticGraphExecutionPlan(
            1, backend="dense", maximum_dense_atoms=2
        ),
        phx.atomistic.AtomisticGraphExecutionPlan(1, backend="particle"),
        phx.atomistic.AtomisticTrainingPolicy(maximum_steps=maximum_steps),
        phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0),
        objectives=_energy_force_objectives(),
        runtime_resources=(_finite_resources(models[0]),),
    )


def _seed_labels(system: Any, provider: Any) -> Any:
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
    return phx.atomistic.AtomisticLabelSet((training_label, validation_label))


def test_campaign_labels_retrains_qualifies_and_promotes() -> None:
    system = _finite_system()
    provider = _provider()
    labels = _seed_labels(system, provider)
    state = phx.atomistic.AtomisticLearningCampaignState(labels)
    models = (_painn(system, 3), _painn(system, 4))
    campaign = _finite_campaign(system, provider, models)
    candidate_frame = _frame(
        system, jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]), "candidate"
    )

    result = phx.atomistic.run_atomistic_campaign_round(
        campaign,
        state,
        (candidate_frame,),
        (_uncertainty(),),
        models,
        (jax.random.key(10), jax.random.key(11)),
        _qualify,
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
    assert len(result.state.committee.members) == 2
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
    # Members were fitted on the canonical lowering of the extended label set.
    problem = result.state.labels.training_problem(
        system, campaign.graph_execution, campaign.objectives, cutoff=2.5
    )
    assert problem.validation is not None
    assert problem.split_evidence is not None
    for training, model in zip(
        result.training_results, result.lifecycle.models, strict=True
    ):
        assert training.problem_id == problem.problem_id
        assert training.policy_id == campaign.training.policy_id
        assert training.problem_id in model.association_ids


def test_campaign_refuses_uncovered_runtime_member_before_labeling() -> None:
    system = _finite_system()
    calls: list[int] = []
    provider = _provider(calls)
    labels = _seed_labels(system, _provider())
    models = (_painn(system, 3), _painn(system, 4))
    campaign = _finite_campaign(system, provider, models)
    # A member of another cutoff declares another graph requirement.
    uncovered = (_painn(system, 3), _painn(system, 5, cutoff=2.0))

    with pytest.raises(ValueError, match="runtime resource"):
        phx.atomistic.run_atomistic_campaign_round(
            campaign,
            phx.atomistic.AtomisticLearningCampaignState(labels),
            (_frame(system, jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]), "c"),),
            (_uncertainty(),),
            uncovered,
            (jax.random.key(10), jax.random.key(11)),
            _qualify,
        )
    assert calls == []
    # One training record has no energy spread: a fitted energy scale is refused
    # at the start of every fit; a declared scale admits the same labels.
    with pytest.raises(ValueError, match="degenerate scale"):
        phx.atomistic.fit_atomistic_potential(
            models[0],
            labels.training_problem(
                system, campaign.graph_execution, campaign.objectives, cutoff=2.5
            ),
            phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0),
            key=jax.random.key(12),
        )
    declared = phx.atomistic.fit_atomistic_potential(
        models[0],
        labels.training_problem(
            system,
            campaign.graph_execution,
            (
                phx.atomistic.AtomisticEnergyObjective(scale=1.0),
                phx.atomistic.AtomisticForceObjective(),
            ),
            cutoff=2.5,
        ),
        phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0),
        key=jax.random.key(12),
    )
    assert bool(declared.successful)
    assert declared.normalization.fitted == (False, True)
    with pytest.raises(ValueError, match="label slots"):
        phx.atomistic.AtomisticLearningCampaignPlan(
            system,
            provider,
            campaign.acquisition,
            campaign.graph_execution,
            campaign.runtime_graph_execution,
            campaign.training,
            campaign.committee_reduction,
            objectives=(phx.atomistic.AtomisticStressObjective(),),
            runtime_resources=campaign.runtime_resources,
        )


def test_campaign_provider_failure_keeps_promoted_state() -> None:
    system = _finite_system()
    labels = _seed_labels(system, _provider())
    state = phx.atomistic.AtomisticLearningCampaignState(labels)
    models = (_painn(system, 3), _painn(system, 4))
    campaign = _finite_campaign(system, _provider(successful=False), models)

    result = phx.atomistic.run_atomistic_campaign_round(
        campaign,
        state,
        (_frame(system, jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]), "c"),),
        (_uncertainty(),),
        models,
        (jax.random.key(10), jax.random.key(11)),
        _qualify,
    )

    assert not bool(result.successful)
    assert not bool(result.promoted)
    assert result.state.state_id == state.state_id
    assert result.training_results == ()
    assert len(result.labels) == 1
    assert not bool(result.labels[0].successful)
    assert result.lifecycle.run.status == "failed"


def test_label_set_lineage_disjoint_validation() -> None:
    system = _finite_system()
    frame = _frame(system, jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), "shared")
    first = phx.atomistic.label_atomistic_acquisitions(
        system, _provider(), (_acquisition(frame),), split="train"
    )[0]

    def other(system: Any, positions: Any, cell: Any) -> Any:
        del system, cell
        return phx.atomistic.ExternalAtomisticEvaluation(
            jnp.sum(positions * positions) + 0.1,
            -2.0 * positions,
            None,
            jnp.asarray(True),
            provider_label="shifted-reference",
        )

    second = phx.atomistic.label_atomistic_acquisitions(
        system,
        phx.atomistic.CallableBornOppenheimerProvider(other, "shifted-reference"),
        (_acquisition(frame, "validation-acquisition"),),
        split="validation",
    )[0]
    labels = phx.atomistic.AtomisticLabelSet((first, second))
    graph = phx.atomistic.AtomisticGraphExecutionPlan(
        1, backend="dense", maximum_dense_atoms=2
    )

    source = labels.source(system)
    assert source.record_ids == (first.label_id, second.label_id)
    assert source.lineage_ids == (first.configuration_id, first.configuration_id)
    assert "stress" not in source.schema.field_names
    # One configuration labeled by two providers is one independent unit.
    with pytest.raises(ValueError, match="lineage"):
        labels.training_problem(system, graph, _energy_force_objectives(), cutoff=2.5)
    with pytest.raises(ValueError, match="another prepared"):
        labels.source(
            phx.atomistic.AtomisticSystemPlan(
                # ty: ignore[invalid-argument-type]
                [10, 20],
                # ty: ignore[invalid-argument-type]
                [1, 1],
                # ty: ignore[invalid-argument-type]
                [1.0, 2.0],
                phx.atomistic.AtomisticUnitSystem.reduced(),
                # ty: ignore[invalid-argument-type]
                atom_type_ids=[1, 1],
            ).prepare()
        )


def _stress_provider(provider_label: str, *, with_stress: bool) -> Any:
    def evaluate(system: Any, positions: Any, cell: Any) -> Any:
        del system
        energy = jnp.sum(positions * positions)
        stress = None
        if with_stress:
            stress = 0.01 * energy * jnp.eye(3) + 0.002 * (cell + cell.T)
        return phx.atomistic.ExternalAtomisticEvaluation(
            energy,
            -2.0 * positions,
            stress,
            jnp.asarray(True),
            provider_label=provider_label,
        )

    return phx.atomistic.CallableBornOppenheimerProvider(evaluate, provider_label)


def _periodic_system(cell: Any) -> Any:
    return phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 8],
        # ty: ignore[invalid-argument-type]
        [1.0, 16.0],
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        cell=phx.discretization.PeriodicCell(cell),
    ).prepare()


def _periodic_frame(system: Any, positions: Any, source: str, cell: Any) -> Any:
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


def test_label_source_keeps_frame_cells_and_tensile_stress() -> None:
    cell = 3.2 * jnp.eye(3)
    strained = jnp.asarray([[3.3, 0.0, 0.0], [0.1, 3.2, 0.0], [0.0, 0.0, 3.1]])
    system = _periodic_system(cell)
    units = system.plan.units
    stressed = phx.atomistic.label_atomistic_acquisitions(
        system,
        _stress_provider("stress-reference", with_stress=True),
        (
            _acquisition(
                _periodic_frame(
                    system, [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]], "strained", strained
                )
            ),
        ),
        split="train",
    )[0]
    unstressed = phx.atomistic.label_atomistic_acquisitions(
        system,
        _stress_provider("energy-force-reference", with_stress=False),
        (
            _acquisition(
                _periodic_frame(system, [[0.0, 0.0, 0.0], [1.1, 0.0, 0.3]], "x", None),
                "unstressed-acquisition",
            ),
        ),
        split="validation",
    )[0]
    labels = phx.atomistic.AtomisticLabelSet((stressed, unstressed))

    source = labels.source(system)
    schema = source.schema
    stress_field = schema.field("stress")
    assert (
        stress_field.convention
        == phx.atomistic.AtomisticStressConvention.CAUCHY_TENSION_POSITIVE.value
    )
    assert stress_field.unit is not None
    assert stress_field.unit.unit_id == units.pressure_unit.unit_id
    masses = schema.field("masses").unit
    assert masses is not None and masses.unit_id == units.mass_unit.unit_id
    assert schema.field("species").convention == "atomic-number"
    rows = source.read([0, 1])
    # A frame cell is kept; a frame without one takes the system cell.
    np.testing.assert_allclose(rows.values["cell"], np.stack([strained, cell]))
    np.testing.assert_array_equal(rows.values["periodic"], np.ones((2, 3), bool))
    np.testing.assert_allclose(rows.values["stress"][0], np.asarray(stressed.stress))
    np.testing.assert_allclose(rows.values["stress"][1], np.zeros((3, 3)))
    np.testing.assert_array_equal(rows.available["stress"], [True, False])
    np.testing.assert_array_equal(rows.values["species"], [1, 8, 1, 8])
    np.testing.assert_allclose(rows.values["masses"], [1.0, 16.0, 1.0, 16.0])
    np.testing.assert_allclose(
        rows.values["forces"],
        np.concatenate(
            [np.asarray(stressed.forces), np.asarray(unstressed.forces)], axis=0
        ),
    )
    # Every split shares the schema, so stress stays unavailable, not absent.
    validation = labels.source(system, split="validation")
    assert validation.schema.schema_id == schema.schema_id
    assert validation.record_ids == (unstressed.label_id,)
    with pytest.raises(ValueError, match="no 'train' records"):
        phx.atomistic.AtomisticLabelSet((unstressed,)).source(system, split="train")


def test_campaign_retains_provider_stress_through_training_and_promotion() -> None:
    cell = 3.2 * jnp.eye(3)
    system = _periodic_system(cell)
    units = system.plan.units
    stress_provider = _stress_provider("stress-reference", with_stress=True)

    def frame(positions: Any, source: str) -> Any:
        return _periodic_frame(system, positions, source, cell)

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
    # Periodic training payloads search routes in one declared cell envelope.
    graph = phx.atomistic.AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=image_capacity,
        search_cell=np.asarray(cell),
        search_margin=0.05,
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
    objectives = (
        phx.atomistic.AtomisticEnergyObjective(),
        phx.atomistic.AtomisticForceObjective(),
        phx.atomistic.AtomisticStressObjective(),
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
        objectives=objectives,
        runtime_resources=(
            neighborhood_bundle_plan(
                phx.atomistic.LearnedGraphPotentialTerm(
                    models[0]
                ).requirements.neighborhoods,
                **bundle_resources(),
            ),
        ),
    )
    candidate = frame([[0.0, 0.0, 0.0], [1.2, 0.4, 0.1]], "candidate")

    result = phx.atomistic.run_atomistic_campaign_round(
        campaign,
        phx.atomistic.AtomisticLearningCampaignState(labels),
        (candidate,),
        (_uncertainty(),),
        models,
        (jax.random.key(10), jax.random.key(11)),
        _qualify,
    )

    assert bool(result.successful & result.promoted)
    acquired = result.labels[0]
    assert acquired.stress is not None
    training_source = result.state.labels.source(system, split="train")
    rows = training_source.read([0, 1])
    np.testing.assert_allclose(rows.values["stress"][1], np.asarray(acquired.stress))
    np.testing.assert_array_equal(rows.available["stress"], [False, True])
    validation_rows = result.state.labels.source(system, split="validation").read([0])
    np.testing.assert_allclose(
        validation_rows.values["stress"][0], np.asarray(validation_label.stress)
    )
    problem = result.state.labels.training_problem(system, graph, objectives, cutoff=3.0)
    assert problem.label_slots == ("energy", "forces", "stress")
    assert problem.validation is not None
    for training, model in zip(
        result.training_results, result.lifecycle.models, strict=True
    ):
        assert training.problem_id == problem.problem_id
        assert training.training_objectives.shape == (1, 3)
        assert bool(jnp.isfinite(training.training_objectives[0, 2]))
        assert float(training.training_objectives[0, 2]) > 0.0
        assert training.problem_id in model.association_ids
        assert training.capabilities_id in model.association_ids
