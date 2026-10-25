"""Committee uncertainty over independently valid neighborhood resources.

Every member owns the explicit neighborhood bundle of its own requirements;
``committee.resources`` initializes one certified state per distinct bundle
(replaying capacity-only failures on declared ladders) at the evaluated
geometry.
"""

import jax.numpy as jnp

import phydrax as phx


def finite_resources(
    program: phx.atomistic.AtomisticPotentialProgram,
) -> phx.atomistic.AtomisticNeighborhoodBundlePlan:
    """Explicit resources of one finite pair program in a nonperiodic frame."""
    requirements = program.requirements.neighborhoods
    return phx.atomistic.AtomisticNeighborhoodBundlePlan(
        requirements,
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "pairs",
                phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
                requirements,
                phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
                phx.discretization.ParticleImageCapacityLadder(
                    (
                        phx.discretization.ParticleImageCapacity(
                            maximum_particles_per_cell=2,
                            maximum_edges=8,
                            maximum_degree=2,
                            maximum_images=27,
                            maximum_case_edges=8,
                        ),
                    )
                ),
                skin=0.5,
            ),
        ),
        tuple(
            phx.atomistic.AtomisticRoleCapacityLadder(
                requirement,
                (
                    phx.atomistic.AtomisticRoleCapacity(
                        maximum_routes=4, maximum_degree=4
                    ),
                ),
            )
            for requirement in requirements
        ),
        frame=phx.discretization.PeriodicCell(
            20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )


system = phx.atomistic.AtomisticSystemPlan(
    # ty: ignore[invalid-argument-type]
    [0, 1],
    # ty: ignore[invalid-argument-type]
    [1, 1],
    # ty: ignore[invalid-argument-type]
    [1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    # ty: ignore[invalid-argument-type]
    atom_type_ids=[0, 0],
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
programs = tuple(
    phx.atomistic.AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [phx.atomistic.LennardJonesPotential([epsilon], [1.0], cutoff)]
    )
    for epsilon, cutoff in ((0.19, 2.5), (0.20, 2.5), (0.21, 2.0))
)
members = tuple(
    program.prepare(system, neighborhood_bundle=finite_resources(program))
    for program in programs
)
committee = phx.atomistic.CommitteeAtomisticPotential(
    members,
    phx.atomistic.CommitteeReductionPolicy(0.1, 0.1, 0.1),
)
resources = committee.resources(positions)
committee = resources.committee
result = committee.evaluate(positions, resources.neighborhoods)
blend = phx.atomistic.ConservativeUncertaintyBlend(committee, committee.members[1])
blend_energy, blend_force, _, _ = blend.evaluate(
    positions,
    resources.neighborhoods,
    resources.neighborhoods[committee.member_bundles[1]],
)
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
    or len(committee.bundles) != 2
    or resources.exhausted
):
    raise RuntimeError("committee uncertainty or acquisition failed")
print(
    float(result.energy),
    float(result.uncertainty.maximum_force_standard_deviation),
    records[0].frame.source_id,
)
