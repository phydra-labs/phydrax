#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _ScaledCoordinate(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale=2.0):
        self.scale = jnp.asarray(scale)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value, /, *, key=None):
        del key
        return self.scale * value


def _runtime():
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
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(0)
    )
    return system, dynamics, state


def test_model_collective_variable_is_vector_contract_and_bias_differentiable():
    system, dynamics, state = _runtime()
    distance = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    source = phx.atomistic.sampling.CollectiveVariableProgram(
        (distance,), names=("distance",)
    )
    learned = phx.atomistic.sampling.ModelCollectiveVariableProgram(
        source,
        _ScaledCoordinate(),
        model_id="scaled-distance",
        names=("slow-distance",),
    )
    plan = phx.atomistic.sampling.AtomisticBiasPlan(
        phx.atomistic.sampling.BiasKind.HARMONIC,
        learned,
        center=[2.0],
        stiffness=[1.5],
    )
    bias = phx.atomistic.sampling.PreparedAtomisticBias(plan, dynamics)
    evaluation = bias.evaluate(state.kinematics.positions, plan.initialize(), state.time)

    assert learned.output_size == 1
    assert learned.names == ("slow-distance",)
    assert bool(evaluation.successful)
    assert evaluation.variables.shape == (1,)
    assert jnp.all(jnp.isfinite(evaluation.forces))
    numerical = jax.grad(
        lambda value: bias.energy(value, plan.initialize(), state.time)[0]
    )(state.kinematics.positions)
    assert jnp.allclose(evaluation.forces, -numerical)


def test_atomistic_cv_feature_library_matches_canonical_trajectory_layout():
    system, dynamics, state = _runtime()
    distance = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    program = phx.atomistic.sampling.CollectiveVariableProgram(
        (distance,), names=("distance",)
    )
    library = phx.atomistic.CollectiveVariableFeatureLibrary(dynamics, program)
    states = jnp.stack(
        (
            state.kinematics.positions,
            jnp.zeros_like(state.kinematics.positions),
        )
    )[None, ...]

    evaluation = library.evaluate(states)

    assert evaluation.values.shape == (1, 1)
    assert bool(evaluation.valid[0])
    assert jnp.allclose(evaluation.values[0, 0], 1.2)
