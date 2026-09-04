import jax.numpy as jnp

import phydrax as phx


system = phx.atomistic.AtomisticSystemPlan(
    [0, 1],
    [1, 1],
    [1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=[0, 0],
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(2).prepare(
    system.particles
)
relation = neighborhood.build(positions)
members = tuple(
    phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([epsilon], [1.0], 2.5)]
    ).prepare(system)
    for epsilon in (0.19, 0.20, 0.21)
)
committee = phx.atomistic.CommitteeAtomisticPotential(
    members,
    phx.atomistic.CommitteeReductionPolicy(0.1, 0.1, 0.1),
)
result = committee.evaluate(positions, relation)
blend = phx.atomistic.ConservativeUncertaintyBlend(committee, members[1])
blend_energy, blend_force, _, _ = blend.evaluate(positions, relation)
frame = phx.atomistic.AtomisticFrame(
    0.0,
    0,
    positions,
    system.plan.particle_ids,
    system_id=system.plan.system_id,
    topology_id=system.topology.topology_id,
    units=system.plan.units,
    source_id="committee-frame",
)
records = phx.atomistic.AcquisitionPlan(
    1, phx.atomistic.CommitteeAcquisitionScorePolicy(1.0, 1.0, 1.0)
).select((frame,), (result.uncertainty,))
if (
    not bool(result.successful)
    or not bool(jnp.isfinite(blend_energy))
    or not bool(jnp.all(jnp.isfinite(blend_force)))
    or len(records) != 1
):
    raise RuntimeError("committee uncertainty or acquisition failed")
print(
    float(result.energy),
    float(result.uncertainty.maximum_force_standard_deviation),
    records[0].frame.source_id,
)
