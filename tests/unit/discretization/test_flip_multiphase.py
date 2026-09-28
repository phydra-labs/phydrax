#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


_POSITION = jnp.asarray(
    [[0.20, 0.20], [0.22, 0.24], [0.62, 0.71], [0.21, 0.22], [0.24, 0.18], [0.60, 0.70]]
)
_VELOCITY = jnp.asarray(
    [[1.0, 0.2], [0.8, -0.1], [0.3, 0.5], [-0.6, 0.4], [-0.4, 0.1], [0.0, -0.2]]
)
_MASS = jnp.asarray([2.0, 1.5, 1.0, 0.5, 0.7, 0.3])
_PHASE = jnp.asarray([0, 0, 0, 1, 1, -1])
_ACTIVE = jnp.asarray([True, True, True, True, True, False])


def _evaluate(drag: Any, step_size: float = 0.1) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8, periodic=True) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(6), _MASS, ambient_dimension=2
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(operators).prepare(
        particles
    )
    plan = phx.discretization.flip.MultiphaseFLIPPlan(
        transfer,
        jnp.asarray([1000.0, 1.0]),
        jnp.asarray([1.0e-3, 1.8e-5]),
        drag,
    )
    population = phx.discretization.ParticlePopulationPlan(particles).initialize(
        active_mask=_ACTIVE, masses=jnp.where(_ACTIVE, _MASS, 0.0)
    )
    state = phx.discretization.flip.MultiphaseFLIPState(
        phx.discretization.flip.FLIPParticleState(_POSITION, _VELOCITY),
        population,
        _PHASE,
        jnp.zeros(finite_volume.cell_shape),
    )
    return plan.evaluate(state, step_size=step_size)


def test_per_phase_particle_to_grid_conserves_mass_and_momentum() -> None:
    result = _evaluate(None)

    assert result.successful
    for phase in (0, 1):
        members = _ACTIVE & (_PHASE == phase)
        np.testing.assert_allclose(
            result.phase_mass[phase], jnp.sum(jnp.where(members, _MASS, 0.0))
        )
        for axis in range(2):
            np.testing.assert_allclose(
                jnp.sum(result.phase_face_momentum[axis][phase]),
                jnp.sum(jnp.where(members, _MASS * _VELOCITY[:, axis], 0.0)),
                atol=1.0e-12,
            )
    np.testing.assert_allclose(result.mass_defect, 0.0, atol=1.0e-12)


def test_implicit_drag_exchanges_equal_opposite_impulse_and_decays_slip() -> None:
    coefficient = 40.0
    step = 0.1
    free = _evaluate(None, step)
    coupled = _evaluate(jnp.asarray([[0.0, coefficient], [coefficient, 0.0]]), step)

    assert coupled.successful
    for axis in range(2):
        np.testing.assert_allclose(
            jnp.sum(coupled.face_momentum[axis]),
            jnp.sum(free.face_momentum[axis]),
            atol=1.0e-12,
        )
        impulse = coupled.pairwise_impulse[axis]
        np.testing.assert_allclose(impulse[0, 1], -impulse[1, 0], atol=0.0)
        masses = coupled.phase_face_mass[axis]
        shared = (masses[0] > 0.0) & (masses[1] > 0.0)
        assert bool(jnp.any(shared))
        slip_before = free.phase_velocity[axis][1] - free.phase_velocity[axis][0]
        slip_after = coupled.phase_velocity[axis][1] - coupled.phase_velocity[axis][0]
        factor = 1.0 / (
            1.0
            + step
            * coefficient
            * (
                1.0 / jnp.where(shared, masses[0], 1.0)
                + 1.0 / jnp.where(shared, masses[1], 1.0)
            )
        )
        np.testing.assert_allclose(
            jnp.where(shared, slip_after, 0.0),
            jnp.where(shared, factor * slip_before, 0.0),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(coupled.momentum_defect, 0.0, atol=1.0e-12)


def test_drag_pair_work_is_nonpositive_and_zero_without_drag() -> None:
    coupled = _evaluate(jnp.asarray([[0.0, 5.0], [5.0, 0.0]]))
    free = _evaluate(None)

    assert bool(jnp.all(coupled.pairwise_work <= 1.0e-14))
    assert float(jnp.min(coupled.pairwise_work)) < 0.0
    np.testing.assert_allclose(free.pairwise_work, 0.0, atol=0.0)
