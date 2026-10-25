from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax import AbstractArrayModel


class QuadraticFreeEnergy(AbstractArrayModel):
    stiffness: jax.Array
    offset: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, stiffness: Any, offset: Any) -> None:
        self.stiffness = jnp.asarray(stiffness)
        self.offset = jnp.asarray(offset)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value: Any, /, *, key: Any = None) -> Any:
        del key
        return jnp.asarray([0.5 * self.stiffness * value[0] ** 2 + self.offset])


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
    atom_type_ids=[0, 0],
).prepare()
program = phx.atomistic.AtomisticPotentialProgram(
    # ty: ignore[invalid-argument-type]
    [phx.atomistic.LennardJonesPotential([0.1], [1.0], 2.5)]
)
(pairs,) = program.requirements.neighborhoods
bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
    (pairs,),
    (
        phx.atomistic.AtomisticNeighborhoodSource(
            "lennard-jones-pairs",
            phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
            (pairs,),
            phx.atomistic.AtomisticDenseImageSearch(maximum_dense_routes=4),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=2,
                        maximum_edges=2,
                        maximum_degree=1,
                        maximum_images=1,
                        maximum_case_edges=2,
                    ),
                )
            ),
            skin=0.3,
        ),
    ),
    (
        phx.atomistic.AtomisticRoleCapacityLadder(
            pairs,
            (phx.atomistic.AtomisticRoleCapacity(maximum_routes=1, maximum_degree=1),),
        ),
    ),
    frame=phx.discretization.PeriodicCell(
        16.0 * jnp.eye(3), origin=jnp.full((3,), -8.0), periodic_axes=(False,) * 3
    ),
)
potential = program.prepare(system, neighborhood_bundle=bundle)
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system, potential, phx.atomistic.VelocityVerletPlan(1.0e-3)
).prepare()
thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
    phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
).prepare(dynamics)
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
state = dynamics.initialize_state(
    positions,
    thermodynamic,
    velocity=jnp.zeros_like(positions),
    key=jax.random.key(0),
)
distance = phx.atomistic.sampling.CollectiveVariablePlan(
    phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
).prepare(system)
variables = phx.atomistic.sampling.CollectiveVariableProgram(
    (distance,), names=("distance",)
)
form = phx.atomistic.sampling.LearnedFreeEnergyBiasForm(
    (QuadraticFreeEnergy(1.0, 2.0), QuadraticFreeEnergy(1.0, -3.0)),
    model_ids=("quadratic-a", "quadratic-b"),
    # ty: ignore[invalid-argument-type]
    reference=[1.0],
    trusted_uncertainty=1.0e-8,
    rejected_uncertainty=0.5,
)
bias = phx.atomistic.sampling.AtomisticBiasTerm(
    variables, form, name="learned-bias", force_group=1
).prepare(system)
energy, cvs, successful = bias.bias(
    phx.atomistic.sampling.CollectiveVariableGeometry(state.kinematics.positions), None
)
committee = form.committee(cvs.values)
if not bool(successful):
    raise RuntimeError("Learned free-energy bias example did not qualify.")
print(float(energy), float(committee.uncertainty), float(committee.trust))
