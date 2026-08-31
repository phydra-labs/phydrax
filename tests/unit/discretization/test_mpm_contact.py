#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_sharp_rigid_contact_projects_normal_and_coulomb_impulse():
    geometry = phx.geometry.Circle((0.0, 0.0), 0.5).compile()
    plan = phx.discretization.RigidMPMContactPlan(
        geometry,
        phx.discretization.SharpCoulombMPMFrictionPlan(0.25),
        contact_band=0.02,
    )
    points = jnp.asarray([[0.49, 0.0], [0.8, 0.0]])
    velocity = jnp.asarray([[-1.0, 0.6], [-1.0, 0.0]])
    mass = jnp.asarray((2.0, 1.0))
    result = plan.apply(points, velocity, mass, 0.0, 0.01)

    assert bool(result.successful)
    assert bool(result.active_mask[0])
    assert not bool(result.active_mask[1])
    assert result.velocity[0, 0] >= -1e-12
    normal_impulse = abs(result.impulse[0])
    tangential_impulse = abs(result.velocity[0, 1] - velocity[0, 1]) * mass[0]
    assert tangential_impulse <= 0.25 * normal_impulse + 1e-12
    assert result.dissipation >= 0.0
    assert result.work <= 0.0


def test_smooth_rigid_contact_has_finite_geometry_velocity_derivative():
    geometry = phx.geometry.Circle((0.0, 0.0), 0.5).compile()
    plan = phx.discretization.RigidMPMContactPlan(
        geometry,
        phx.discretization.SmoothCoulombMPMFrictionPlan(0.3, regularization=1e-3),
        contact_band=0.02,
        smooth_normal_regularization=1e-3,
    )
    point = jnp.asarray([[0.495, 0.0]])
    mass = jnp.asarray((1.0,))

    def objective(velocity):
        return jnp.sum(plan.apply(point, velocity, mass, 0.0, 0.01).velocity ** 2)

    gradient = jax.grad(objective)(jnp.asarray([[-0.2, 0.1]]))
    assert jnp.all(jnp.isfinite(gradient))


def test_contact_and_prescribed_velocity_overlap_rejects_compilation():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    mask = jnp.ones(grid.vertices().shape + (2,), dtype=bool)
    boundary = phx.discretization.PrescribedGridVelocityPlan(mask)
    contact = phx.discretization.RigidMPMContactPlan(
        phx.geometry.Circle((0.0, 0.0), 0.5).compile(),
        phx.discretization.SharpCoulombMPMFrictionPlan(0.0),
        contact_band=0.1,
    )
    problem = phx.equations.MaterialPointProblemIR(
        "overlap",
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        phx.equations.compile_material_point_problem(
            problem,
            particles,
            splat,
            phx.discretization.ExplicitMPMMethodPlan(),
            phx.discretization.MPMParticleDomainPlan(
                jnp.asarray([[-0.5, -0.5], [0.5, 0.5]]),
                support_margin=0.4,
            ),
            boundary=boundary,
            contact=contact,
        )
