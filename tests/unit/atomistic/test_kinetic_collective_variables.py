#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._model import AbstractArrayModel
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


class _ScaledCoordinate(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale: Any = 2.0) -> None:
        self.scale = jnp.asarray(scale)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value: Any, /, *, key: Any = None) -> Any:
        del key
        return self.scale * value


def _runtime() -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray([10, 20], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([1.0, 1.0], dtype=np.float64),
        units,
        atom_type_ids=np.asarray([0, 0], dtype=np.int32),
    ).prepare()
    # Finite pair: explicit nonperiodic frame and the one-pair storage of the
    # former dense neighborhood.
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            (
                phx.atomistic.LennardJonesPotential(
                    np.asarray([0.1]), np.asarray([1.0]), 2.5
                ),
            )
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
    return system, dynamics, state


def test_kinetic_collective_variables_scenario_1() -> None:
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
    bias = phx.atomistic.sampling.AtomisticBiasTerm(
        learned,
        phx.atomistic.sampling.HarmonicBiasForm(jnp.asarray([2.0]), 1.5),
        name="slow-umbrella",
        force_group=1,
    ).prepare(system)
    term_state = bias.initial_term_state(jnp.float64)

    def energy(value: jax.Array) -> jax.Array:
        geometry = phx.atomistic.sampling.CollectiveVariableGeometry(value)
        return bias.bias(geometry, term_state)[0]

    _, evaluation, successful = bias.bias(
        phx.atomistic.sampling.CollectiveVariableGeometry(state.kinematics.positions),
        term_state,
    )

    assert learned.output_size == 1
    assert learned.names == ("slow-distance",)
    assert bool(successful)
    assert evaluation.values.shape == (1,)
    gradient = jax.grad(energy)(state.kinematics.positions)
    assert jnp.all(jnp.isfinite(gradient))
    step = 1e-6
    direction = jnp.zeros_like(gradient).at[1, 0].set(1.0)
    finite = (
        energy(state.kinematics.positions + step * direction)
        - energy(state.kinematics.positions - step * direction)
    ) / (2.0 * step)
    assert jnp.allclose(jnp.sum(gradient * direction), finite, rtol=1e-5)
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
