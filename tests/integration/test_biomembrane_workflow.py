#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cellular_mechanics._membrane import BiomembranePlan
from phydrax.discretization.lattice_boltzmann import ImmersedBoundaryForcingPlan


def _membrane():
    reference = np.asarray(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    reference = 0.08 * reference / np.sqrt(3.0) + 0.5
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    plan = BiomembranePlan(
        faces,
        bending_rigidity=0.2,
        spontaneous_curvature=2.0,
        global_area_modulus=0.5,
        volume_modulus=0.5,
        active_traction=0.01,
        mobility=1.0e-4,
        species_diffusivity=(0.02, 0.01),
        reaction_matrix=((-0.1, 0.05), (0.1, -0.05)),
        curvature_coupling=(0.02, -0.01),
    )
    return plan.prepare(reference)


def _fluid():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))
    return phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D3Q19()
    ).prepare()


def test_transport_thermal_mechanics_and_immersed_fluid_compose_end_to_end():
    membrane = _membrane()
    fluid = _fluid()
    forcing = ImmersedBoundaryForcingPlan(
        fluid, iteration_count=16, convergence_tolerance=1.0e-5
    )
    mass = jnp.asarray(((0.04, 0.02), (0.03, 0.01), (0.02, 0.03), (0.01, 0.04)))
    state = membrane.state(species_mass=mass)
    transported = membrane.diffuse_react(state, 1.0e-3)
    assert transported.evidence.successful
    np.testing.assert_allclose(transported.evidence.total_mass_residual, 0.0, atol=2.0e-8)

    thermal = membrane.thermal_step(
        transported.accepted_state,
        jnp.asarray((19, 23), dtype=jnp.uint32),
        1.0e-5,
        0.0,
        step_index=4,
    )
    assert thermal.evidence.successful
    velocity = (
        thermal.accepted_state.positions - transported.accepted_state.positions
    ) / 1.0e-5
    coupling = membrane.couple_immersed_boundary(
        thermal.accepted_state,
        velocity,
        forcing,
        jnp.zeros(fluid.grid.shape + (3,)),
        jnp.ones(fluid.grid.shape),
        1.0e-3,
    )
    assert coupling.successful
    np.testing.assert_allclose(coupling.force_balance_residual, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(
        coupling.membrane_force,
        -coupling.forcing.marker_force,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        coupling.work,
        jnp.sum(coupling.forcing.ledger.body_work),
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        coupling.total_force,
        coupling.mechanical_force + coupling.membrane_force,
        rtol=0.0,
        atol=0.0,
    )
    evaluation = membrane.evaluate(thermal.accepted_state)
    assert evaluation.valid
    assert evaluation.force.shape == coupling.membrane_force.shape
