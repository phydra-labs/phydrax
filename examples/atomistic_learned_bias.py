import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._model import AbstractArrayModel


class QuadraticFreeEnergy(AbstractArrayModel):
    stiffness: jax.Array
    offset: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, stiffness, offset):
        self.stiffness = jnp.asarray(stiffness)
        self.offset = jnp.asarray(offset)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value, /, *, key=None):
        del key
        return jnp.asarray([0.5 * self.stiffness * value[0] ** 2 + self.offset])


units = phx.atomistic.AtomisticUnitSystem.reduced()
system = phx.atomistic.AtomisticSystemPlan(
    [10, 20], [1, 1], [1.0, 1.0], units, atom_type_ids=[0, 0]
).prepare()
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
    system.particles
)
potential = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.1], [1.0], 2.5)]
).prepare(system)
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system, potential, neighborhood, phx.atomistic.VelocityVerletPlan(1.0e-3)
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
state = dynamics.initialize_state(
    positions, velocity=jnp.zeros_like(positions), key=jax.random.key(0)
)
distance = phx.atomistic.sampling.CollectiveVariablePlan(
    phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
).prepare(system)
variables = phx.atomistic.sampling.CollectiveVariableProgram(
    (distance,), names=("distance",)
)
plan = phx.atomistic.sampling.LearnedFreeEnergyBiasPlan(
    variables,
    (QuadraticFreeEnergy(1.0, 2.0), QuadraticFreeEnergy(1.0, -3.0)),
    model_ids=("quadratic-a", "quadratic-b"),
    reference=[1.0],
    trusted_uncertainty=1.0e-8,
    rejected_uncertainty=0.5,
)
bias = plan.prepare(dynamics)
evaluation = bias.evaluate(state.kinematics.positions, plan.initialize(), state.time)
if not bool(evaluation.successful):
    raise RuntimeError("Learned free-energy bias example did not qualify.")
print(float(evaluation.energy), float(evaluation.uncertainty), float(evaluation.trust))
