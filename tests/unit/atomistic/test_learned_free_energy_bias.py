#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax._model import AbstractArrayModel
from phydrax._trainable import partition_parameters
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


class _QuadraticFreeEnergy(AbstractArrayModel):
    stiffness: jax.Array
    offset: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, stiffness: Any, offset: Any = 0.0) -> None:
        self.stiffness = jnp.asarray(stiffness, dtype=jnp.float64)
        self.offset = jnp.asarray(offset, dtype=jnp.float64)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value: Any, /, *, key: Any = None) -> Any:
        del key
        return jnp.asarray([0.5 * self.stiffness * value[0] ** 2 + self.offset])


def _runtime() -> Any:
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
    # Finite pair: explicit nonperiodic frame and the one-pair storage of the
    # former dense neighborhood.
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.1], [1.0], 2.5)]
        ),
        system,
        search=phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(8, particles_per_cell=2, images=27),),
        role_capacities=(role_capacity(1, 1),),
        frame=phx.discretization.PeriodicCell(
            20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
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
    return system, dynamics, thermodynamic, state


def test_runtime_bias_freezes_its_trained_committee_but_keeps_coordinate_response() -> (
    None
):
    model = _QuadraticFreeEnergy(2.0, 3.0)
    model_parameters, _, _ = partition_parameters(model)
    np.testing.assert_array_equal(model_parameters.stiffness, 2.0)
    form = phx.atomistic.sampling.LearnedFreeEnergyBiasForm(
        (model,),
        model_ids=("quadratic",),
        reference=jnp.asarray([1.0], dtype=jnp.float64),
    )
    parameters, model_state, fixed = partition_parameters(form)
    assert parameters is None and model_state is None
    point = jnp.asarray([2.0], dtype=jnp.float64)
    np.testing.assert_allclose(fixed.committee(point).mean, 3.0)

    def mean(values: Array) -> Array:
        return form.committee(values).mean

    np.testing.assert_array_equal(
        jax.grad(mean)(point), np.asarray([4.0], dtype=np.float64)
    )


def test_learned_free_energy_bias_scenario_1() -> None:
    centers = jnp.asarray([[-1.0], [0.0], [1.0]])
    # ty: ignore[invalid-argument-type]
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
    system, dynamics, thermodynamic, state = _runtime()
    distance = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    variables = phx.atomistic.sampling.CollectiveVariableProgram(
        (distance,), names=("distance",)
    )
    form = phx.atomistic.sampling.LearnedFreeEnergyBiasForm(
        (_QuadraticFreeEnergy(1.0, 3.0), _QuadraticFreeEnergy(1.0, -2.0)),
        model_ids=("quadratic-a", "quadratic-b"),
        # ty: ignore[invalid-argument-type]
        reference=[1.0],
        trusted_uncertainty=1.0e-8,
        rejected_uncertainty=0.5,
    )
    term = phx.atomistic.sampling.AtomisticBiasTerm(
        variables, form, name="learned-bias", force_group=1
    ).prepare(system)
    geometry = phx.atomistic.sampling.CollectiveVariableGeometry(
        state.kinematics.positions
    )
    energy, evaluation, successful = term.bias(geometry, None)
    committee = form.committee(evaluation.values)

    assert bool(successful)
    assert committee.uncertainty < 1.0e-10
    assert jnp.allclose(committee.trust, 1.0)
    gradient = jax.grad(
        lambda value: term.bias(
            phx.atomistic.sampling.CollectiveVariableGeometry(value), None
        )[0]
    )(state.kinematics.positions)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.isfinite(energy)
    assert not term.plan.stateful
