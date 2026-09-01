#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.mpm._rigid_coupling import (
    RigidMPMCouplingMode,
    RigidMPMCouplingPlan,
)
from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._rigid_body import RigidBodySetPlan


def _mpm_problem(*, periodic=True):
    dimension = 2
    axes = tuple(
        phx.discretization.UniformAxisSpec(
            12,
            periodic=periodic,
            endpoint=not periodic,
        )
        for _ in range(dimension)
    )
    bounds = (
        jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,))))
        if periodic
        else jnp.asarray([[-0.25, -0.25], [1.25, 1.25]])
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y")).prepare(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]),
        jnp.asarray([0.01]),
        ambient_dimension=dimension,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    domain = phx.discretization.MPMParticleDomainPlan(
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        periodic=(periodic, periodic),
        support_margin=0.0 if periodic else 0.21,
    )
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    problem = phx.equations.MaterialPointProblemIR("coupled-solid", material)
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        domain,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        jnp.asarray([[0.4, 0.4]]),
        jnp.zeros((1, 2)),
        jnp.asarray([0.01]),
        arguments,
    )
    body_particles = ParticleSetPlan(
        jnp.asarray([10]),
        jnp.asarray([2.0]),
        ambient_dimension=2,
    ).prepare()
    bodies = RigidBodySetPlan(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
    ).prepare(body_particles)
    kinematics = bodies.kinematics(
        jnp.asarray([[0.3, 0.4]]),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    return compiled.dynamics, arguments, state, bodies, kinematics


def _coupling(dynamics, bodies, mode, **parameters):
    return RigidMPMCouplingPlan(
        jnp.asarray([0]),
        jnp.asarray([0]),
        jnp.zeros((1, 2)),
        ambient_dimension=2,
        mode=mode,
        local_normals=jnp.asarray([[1.0, 0.0]]),
        **parameters,
    ).prepare(dynamics, bodies)


def test_penalty_and_hard_weld_are_distinct_and_preserve_action_reaction():
    dynamics, arguments, state, bodies, kinematics = _mpm_problem()
    penalty = _coupling(
        dynamics,
        bodies,
        RigidMPMCouplingMode.PENALTY,
        stiffness=20.0,
        damping=0.5,
    )
    weld = _coupling(dynamics, bodies, RigidMPMCouplingMode.WELD)

    penalty_result = jax.jit(
        lambda mpm, coupling: penalty.evaluate(mpm, kinematics, coupling, 0.001)
    )(state, penalty.initialize_state())
    weld_result = weld.evaluate(state, kinematics, weld.initialize_state(), 0.001)

    assert bool(penalty_result.successful)
    assert bool(weld_result.successful)
    assert not bool(penalty_result.payload.hard[0])
    assert bool(weld_result.payload.hard[0])
    assert not bool(penalty_result.payload.unilateral[0])
    assert not bool(weld_result.payload.unilateral[0])
    assert jnp.linalg.norm(penalty_result.route_force[0]) > 0.0
    np.testing.assert_allclose(weld_result.route_force, 0.0, atol=0.0)
    np.testing.assert_allclose(penalty_result.action_reaction_residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        penalty_result.angular_action_reaction_residual, 0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        jnp.sum(penalty_result.grid_force, axis=(0, 1)),
        jnp.sum(penalty_result.particle_force, axis=0),
        rtol=1e-13,
        atol=1e-13,
    )
    assert bool(penalty_result.action_reaction_valid)
    assert bool(penalty_result.grid_scatter_valid)
    assert penalty_result.payload.validity_margin[0] > 0.0
    assert penalty_result.payload.feature_margin[0] > 0.0
    assert bool(penalty_result.certificate.finite)
    stepped = penalty.step_detailed(
        penalty.initialize_state(),
        state,
        kinematics,
        0.001,
        arguments,
    )
    assert bool(stepped.successful)
    assert int(stepped.candidate_state.cache_generation) == 1
    assert int(stepped.accepted_state.cache_generation) == 1
    assert bool(stepped.accepted_state.cache_valid)
    refreshed = penalty.evaluate(
        stepped.accepted_mpm_state,
        kinematics,
        stepped.accepted_state,
        0.001,
    )
    assert bool(refreshed.certificate.cache_hit)
    assert bool(refreshed.certificate.cache_coherent)


def test_impulse_mode_exposes_unilateral_payload_and_equal_opposite_impulse_load():
    dynamics, _, state, bodies, _ = _mpm_problem()
    kinematics = bodies.kinematics(
        jnp.asarray([[0.45, 0.4]]),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    impulse = _coupling(
        dynamics,
        bodies,
        RigidMPMCouplingMode.IMPULSE,
        restitution=0.25,
        activation_distance=0.02,
        baumgarte_factor=0.2,
    )
    result = impulse.evaluate(state, kinematics, impulse.initialize_state(), 0.001)

    assert bool(result.successful)
    assert bool(result.payload.unilateral[0])
    assert not bool(result.payload.hard[0])
    assert result.payload.gap[0] < 0.0
    assert result.route_impulse[0, 0] > 0.0
    np.testing.assert_allclose(result.action_reaction_residual, 0.0, atol=1e-13)
    assert bool(result.finite)


def test_mpm_stability_and_route_failures_roll_back_coupling_cache_and_material_state():
    dynamics, arguments, state, bodies, kinematics = _mpm_problem(periodic=False)
    coupling = _coupling(
        dynamics,
        bodies,
        RigidMPMCouplingMode.PENALTY,
        stiffness=10.0,
    )
    initial = coupling.initialize_state()

    unstable = coupling.step_detailed(
        initial,
        state,
        kinematics,
        10.0,
        arguments,
    )
    assert not bool(unstable.successful)
    for accepted, original in zip(
        jax.tree.leaves(unstable.accepted_state),
        jax.tree.leaves(initial),
        strict=True,
    ):
        np.testing.assert_array_equal(accepted, original)
    for accepted, original in zip(
        jax.tree.leaves(unstable.accepted_mpm_state.particles),
        jax.tree.leaves(state.particles),
        strict=True,
    ):
        np.testing.assert_array_equal(accepted, original)
    np.testing.assert_array_equal(
        unstable.accepted_mpm_state.last_status,
        unstable.mpm_result.accepted_state.last_status,
    )

    outside = eqx.tree_at(
        lambda value: value.particles.position,
        state,
        jnp.asarray([[1.24, 0.4]]),
    )
    route_failed = coupling.step_detailed(
        initial,
        outside,
        kinematics,
        0.001,
        arguments,
    )
    assert not bool(route_failed.successful)
    assert not bool(route_failed.evaluation.certificate.successful)
    assert int(route_failed.accepted_state.cache_generation) == 0
    assert int(route_failed.accepted_state.route_digest) == -1
    assert jnp.isfinite(route_failed.stability_margin)
    assert not bool(route_failed.stability_margin_valid)
    assert not bool(route_failed.route_margin_valid)
    np.testing.assert_array_equal(
        route_failed.accepted_mpm_state.last_status,
        route_failed.mpm_result.accepted_state.last_status,
    )
    np.testing.assert_array_equal(
        route_failed.accepted_mpm_state.particles.material_state,
        outside.particles.material_state,
    )
