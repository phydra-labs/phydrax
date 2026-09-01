#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_one_dimensional_mpm_translation_and_visualization_output(tmp_path):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    position = jnp.asarray([[0.2], [0.4], [0.6]])
    volume = jnp.full((3,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), volume, ambient_dimension=1
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "one-dimensional",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(1),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0], [1.0]]),
            periodic=(True,),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = jnp.full((3, 1), 0.03)
    state = compiled.initialize_state(position, velocity, volume, arguments)
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.accepted_state.particles.position,
        position + 0.001 * velocity,
        rtol=1e-10,
        atol=1e-12,
    )
    assert not bool(detail.diagnostics.transfer.angular_momentum_valid)
    output = phx.solver.MPMOutputPlan(compiled, tmp_path / "particles.h5")
    output.append(detail.accepted_state)
    vtk = output.write_vtk_snapshot(tmp_path / "particles.vtu", detail.accepted_state)
    xdmf = (tmp_path / "particles.xdmf").read_text()
    assert '<Geometry GeometryType="XY">' in xdmf
    assert 'Dimensions="3 2"' in xdmf
    assert vtk.exists()


def test_runtime_lifecycle_mass_and_activity_are_authoritative():
    plan = phx.discretization.MPMParticleLifecyclePlan(3)
    lifecycle, valid = plan.initialize(
        jnp.asarray((10, 11, 12)),
        jnp.asarray((0.01, 0.02, 0.03)),
        jnp.asarray((True, True, False)),
    )
    assert bool(valid)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    position = jnp.asarray([[0.2], [0.4], [0.6]])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), jnp.ones((3,)), ambient_dimension=1
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "lifecycle",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(1),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0], [1.0]]),
            periodic=(True,),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position,
        jnp.zeros((3, 1)),
        jnp.asarray((0.01, 0.02, 0.01)),
        arguments,
        lifecycle_state=lifecycle,
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.diagnostics.transfer.particle_mass, 0.03, atol=1e-12
    )
    np.testing.assert_array_equal(
        detail.accepted_state.lifecycle_state.particle_ids,
        lifecycle.particle_ids,
    )
