#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _QuadraticFreeEnergy(AbstractArrayModel):
    stiffness: jax.Array
    offset: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, stiffness, offset=0.0):
        self.stiffness = jnp.asarray(stiffness, dtype=jnp.float64)
        self.offset = jnp.asarray(offset, dtype=jnp.float64)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value, /, *, key=None):
        del key
        return jnp.asarray([0.5 * self.stiffness * value[0] ** 2 + self.offset])


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


def test_restrained_mean_force_estimator_records_finite_stiffness_gradient():
    centers = jnp.asarray([[-1.0], [0.0], [1.0]])
    plan = phx.atomistic.sampling.RestrainedMeanForcePlan(centers, [10.0])
    samples = centers[:, None, :] + jnp.asarray(
        [[[-0.2], [-0.1], [-0.15]], [[0.0], [0.1], [-0.1]], [[0.2], [0.1], [0.15]]]
    )

    data = phx.atomistic.sampling.estimate_restrained_free_energy_gradient(
        plan, samples, source_id="restrained-windows"
    )

    assert jnp.all(data.valid)
    assert jnp.all(jnp.isfinite(data.gradient_standard_error))
    assert data.free_energy_gradients[0, 0] > 0.0
    assert data.free_energy_gradients[2, 0] < 0.0


def test_free_energy_gradient_training_selects_scalar_model():
    centers = jnp.linspace(-1.5, 1.5, 21)[:, None]
    data = phx.atomistic.sampling.MeanForceData(
        centers,
        2.0 * centers,
        gradient_standard_error=jnp.full_like(centers, 0.1),
        source_id="quadratic-gradient",
    )
    result = phx.atomistic.sampling.fit_free_energy_model(
        _QuadraticFreeEnergy(0.3),
        data,
        jax.random.key(7),
        model_id="quadratic-free-energy",
        policy=phx.atomistic.sampling.FreeEnergyTrainingPolicy(
            maximum_steps=40, learning_rate=5.0e-2, validation_interval=5
        ),
    )

    assert bool(result.valid)
    assert result.validation_loss[-1] < 0.1
    assert jnp.isfinite(result.model(jnp.asarray([0.4]))[0])


def test_gauge_aligned_committee_produces_conservative_trusted_bias():
    system, dynamics, state = _runtime()
    distance = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    variables = phx.atomistic.sampling.CollectiveVariableProgram(
        (distance,), names=("distance",)
    )
    plan = phx.atomistic.sampling.LearnedFreeEnergyBiasPlan(
        variables,
        (_QuadraticFreeEnergy(1.0, 3.0), _QuadraticFreeEnergy(1.0, -2.0)),
        model_ids=("quadratic-a", "quadratic-b"),
        reference=[1.0],
        trusted_uncertainty=1.0e-8,
        rejected_uncertainty=0.5,
    )
    bias = plan.prepare(dynamics)
    evaluation = bias.evaluate(state.kinematics.positions, plan.initialize(), state.time)

    assert bool(evaluation.successful)
    assert evaluation.uncertainty < 1.0e-10
    assert jnp.allclose(evaluation.trust, 1.0)
    numerical = jax.grad(
        lambda value: bias.energy(value, plan.initialize(), state.time)[0]
    )(state.kinematics.positions)
    assert jnp.allclose(evaluation.forces, -numerical)

    runtime = phx.atomistic.sampling.PreparedBiasedDynamics(dynamics, bias)
    biased_state = runtime.initialize(state)
    assert bool(biased_state.bias.successful)
