import jax
import jax.numpy as jnp

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.reduced()
system = phx.atomistic.AtomisticSystemPlan(
    [10, 20], [1, 1], [1.0, 1.0], units, atom_type_ids=[1, 1]
).prepare()


def evaluate(system, positions, cell):
    del system, cell
    return phx.atomistic.ExternalAtomisticEvaluation(
        jnp.sum(positions * positions),
        -2.0 * positions,
        None,
        jnp.asarray(True),
        "harmonic-reference",
    )


provider = phx.atomistic.CallableBornOppenheimerProvider(evaluate, "harmonic-reference")


def frame(distance, source):
    return phx.atomistic.AtomisticFrame(
        0.0,
        0,
        jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        system.plan.particle_ids,
        system_id=system.prepared_id,
        topology_id=system.topology.topology_id,
        units=system.plan.units,
        source_id=source,
    )


def seed_record(value, split):
    acquisition = phx.atomistic.AcquisitionRecord(
        frame=value,
        descriptor=value.positions.reshape((-1,)),
        component_scores=jnp.asarray([1.0, 0.0, 0.0]),
        source_index=0,
        score=1.0,
        reason="seed",
        model_id="seed",
        plan_id=f"seed-{split}",
    )
    return phx.atomistic.label_atomistic_acquisitions(
        system, provider, (acquisition,), split=split
    )[0]


labels = phx.atomistic.AtomisticLabelSet(
    (
        seed_record(frame(1.0, "train"), "train"),
        seed_record(frame(1.1, "validation"), "validation"),
    )
)
dense_graph = phx.atomistic.AtomisticGraphExecutionPlan(
    1, backend="dense", maximum_dense_atoms=2
)
particle_graph = phx.atomistic.AtomisticGraphExecutionPlan(1, backend="particle")
reduction = phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0)
plan = phx.atomistic.AtomisticLearningCampaignPlan(
    system,
    provider,
    phx.atomistic.AcquisitionPlan(
        1, phx.atomistic.CommitteeAcquisitionScorePolicy(1.0, 1.0, 1.0)
    ),
    dense_graph,
    particle_graph,
    phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0),
    reduction,
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
    for seed in (2, 3)
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


result = phx.atomistic.run_atomistic_campaign_round(
    plan,
    phx.atomistic.AtomisticLearningCampaignState(labels),
    (frame(1.2, "candidate"),),
    (uncertainty,),
    models,
    (jax.random.key(4), jax.random.key(5)),
    qualify,
)
if not bool(result.successful & result.promoted):
    raise RuntimeError("Adaptive-learning campaign example did not qualify.")
print(result.state.round_index, result.state.labels.revision.revision_id)
