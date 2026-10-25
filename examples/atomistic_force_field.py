import jax.numpy as jnp
import numpy as np

import phydrax as phx


system_plan = phx.atomistic.AtomisticSystemPlan(
    np.asarray([0, 1, 2]),
    np.asarray([1, 1, 1]),
    np.asarray([1.0, 1.0, 1.0]),
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=np.asarray([0, 0, 0]),
)
program = phx.atomistic.AtomisticPotentialProgram(
    (
        phx.atomistic.MorsePotential(
            np.asarray([[0.2]]), np.asarray([[2.0]]), np.asarray([[1.0]]), 2.5
        ),
    )
)
# The Morse interaction-site pair view is served by one declared cell-list
# image source of its own entity inside an explicit nonperiodic search frame;
# no resource is inferred.
requirements = program.requirements.neighborhoods
(entity,) = {requirement.entity for requirement in requirements}
resources = phx.atomistic.AtomisticNeighborhoodBundlePlan(
    requirements,
    (
        phx.atomistic.AtomisticNeighborhoodSource(
            "morse-pairs",
            entity,
            requirements,
            phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=3,
                        maximum_edges=12,
                        maximum_degree=4,
                        maximum_images=27,
                        maximum_case_edges=12,
                    ),
                )
            ),
            skin=0.5,
        ),
    ),
    tuple(
        phx.atomistic.AtomisticRoleCapacityLadder(
            requirement,
            (phx.atomistic.AtomisticRoleCapacity(maximum_routes=6, maximum_degree=4),),
        )
        for requirement in requirements
    ),
    frame=phx.discretization.PeriodicCell(
        20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
    ),
)
force_field = phx.atomistic.AtomisticForceFieldPlan(
    system_plan,
    program,
    phx.atomistic.AtomisticNonbondedPolicy(2.5),
    phx.atomistic.AtomisticForceFieldProvenance(
        "native", ("example",), "custom", "explicit"
    ),
).prepare(neighborhood_bundle=resources)
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
# The force field's own prepared bundle is certified at this configuration.
bundle = force_field.potential.neighborhood_bundle
neighborhoods = bundle.initialize(
    phx.atomistic.bind_atomistic_bundle_geometry(bundle, force_field.system, positions)
)
if not bool(neighborhoods.successful[0]):
    raise RuntimeError("neighborhood bundle binding failed")
result = force_field.potential.evaluate(positions, neighborhoods)
if not bool(result.successful):
    raise RuntimeError("force-field evaluation failed")
print(float(result.energy), result.forces)
