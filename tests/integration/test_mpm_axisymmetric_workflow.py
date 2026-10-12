#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Public implicit and real damped-explicit axisymmetric workflows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from examples.material_point_axisymmetric import relax_cylinder


@pytest.mark.parametrize("kind", ("none", "b_bar"), ids=("local", "bbar"))
def test_axisymmetric_implicit_small_step_converges_with_finite_tangent(
    kind: phx.discretization.MPMVolumetricStabilizationKind,
) -> None:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(17, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),
        ),
        axis_names=("r", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0), (2.0, 1.0)), dtype=jnp.float64))
    position = jnp.asarray(
        ((0.027, 0.31), (0.043, 0.38), (0.36, 0.52)), dtype=jnp.float64
    )
    volume = jnp.full((3,), 0.01, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3, dtype=jnp.int32),
        volume,
        ambient_dimension=2,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "axisymmetric-implicit-consumer",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            geometry=phx.discretization.MPMGeometryPlan(
                "axisymmetric", spatial_dimension=2
            ),
            volumetric_stabilization=phx.discretization.MPMVolumetricStabilizationPlan(
                kind
            ),
        ),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.0, 0.0), (1.8, 1.0)), dtype=jnp.float64),
            periodic=(False, True),
            support_margin=(0.1875, 0.0),
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = jnp.stack(
        (0.03 * position[:, 0], jnp.zeros((3,), dtype=jnp.float64)), axis=-1
    )
    initial = compiled.initialize_state(position, velocity, volume, arguments)
    # The derivative duality oracle resolves O(1e-5) sensitivities; the default
    # absolute Krylov tolerance is appropriate for stepping, not this oracle.
    method = (
        phx.solver.ImplicitMPMMethodPlan(
            phx.nonlinear.NewtonKrylov(
                linear_policy=phx.linalg.LinearSolvePolicy(
                    phx.linalg.GMRES(restart=64),
                    tolerance=phx.linalg.TolerancePolicy(
                        relative=1e-12,
                        absolute=1e-15,
                        max_steps=512,
                    ),
                ),
            ),
            volumetric_stabilization=compiled.dynamics.method.volumetric_stabilization,
        )
        if kind == "b_bar"
        else None
    )
    implicit = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics, method)
    detail = implicit.step_detailed(initial, 0.001, arguments)
    assert bool(detail.successful)
    assert bool(detail.diagnostics.tangent_successful)
    assert float(detail.diagnostics.residual_norm) < 1e-8
    assert detail.accepted_state.particles.deformation_gradient.shape == (3, 3, 3)
    assert detail.accepted_state.particles.affine_velocity.shape == (3, 2, 2)
    assert bool(jnp.all(jnp.isfinite(detail.accepted_state.particles.first_piola)))
    assert bool(
        jnp.all(detail.accepted_state.particles.deformation_gradient[:, 2, 2] > 1.0)
    )
    np.testing.assert_allclose(detail.accepted_state.time, 0.001, rtol=0.0, atol=0.0)

    def hoop_stretch(scale: Array) -> Array:
        state = compiled.initialize_state(position, scale * velocity, volume, arguments)
        step = implicit.step_detailed(state, 0.001, arguments)
        return step.accepted_state.particles.deformation_gradient[:, 2, 2]

    scale = jnp.asarray(1.0, dtype=jnp.float64)
    direction = jnp.asarray(0.7, dtype=jnp.float64)
    cotangent = jnp.asarray((0.2, -0.4, 0.5), dtype=jnp.float64)
    _, tangent = jax.jvp(hoop_stretch, (scale,), (direction,))
    _, pullback = jax.vjp(hoop_stretch, scale)
    adjoint = pullback(cotangent)[0]
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert bool(jnp.isfinite(adjoint))
    assert float(jnp.linalg.norm(tangent)) > 1e-8
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        adjoint * direction,
        rtol=2e-9,
        atol=2e-12,
    )

    residual = implicit.prepare_residual(initial, 0.001, arguments)
    rng = np.random.default_rng(2851)
    residual_direction = jnp.asarray(
        0.01 * rng.normal(size=residual.initial_velocity.shape), dtype=jnp.float64
    )
    residual_value, valid = residual.evaluate(residual.initial_velocity)
    assert bool(valid)
    _, residual_jvp = jax.jvp(
        lambda value: residual.evaluate(value)[0],
        (residual.initial_velocity,),
        (residual_direction,),
    )
    epsilon = 1e-5
    plus, plus_valid = residual.evaluate(
        residual.initial_velocity + epsilon * residual_direction
    )
    minus, minus_valid = residual.evaluate(
        residual.initial_velocity - epsilon * residual_direction
    )
    assert bool(plus_valid) and bool(minus_valid)
    assert bool(jnp.all(jnp.isfinite(residual_value)))
    assert bool(jnp.all(jnp.isfinite(residual_jvp)))
    np.testing.assert_allclose(
        residual_jvp, (plus - minus) / (2.0 * epsilon), rtol=3e-7, atol=3e-10
    )


def test_lame_thick_cylinder_relaxes_from_unstressed_state() -> None:
    result = relax_cylinder(radial_cells=24, steps=4000, step_size=0.003)
    assert bool(jnp.all(result.accepted))
    assert int(result.final_state.accepted_step) == 4000
    np.testing.assert_allclose(result.final_state.time, 12.0, rtol=0.0, atol=2e-11)
    initial = np.asarray(result.initial_position)
    state = result.final_state.particles
    radius = initial[:, 0]
    volume = np.asarray(state.reference_volume)
    displacement = np.asarray(state.position)[:, 0] - radius

    # Independent plane-strain linear-elastic reference. The radial equilibrium
    # equation is d(sigma_r)/dr+(sigma_r-sigma_theta)/r=0. Its Lamé solution is
    # sigma_r=A-B/r^2, sigma_theta=A+B/r^2. Enforcing sigma_r(a)=-pi and
    # sigma_r(b)=-po gives A=(pi*a^2-po*b^2)/(b^2-a^2),
    # B=(pi-po)*a^2*b^2/(b^2-a^2). With u=c*r+d/r and epsilon_z=0,
    # c=A/[2(mu+lambda)], d=B/(2*mu). No MPM constitutive routine supplies this oracle.
    inner, outer, pressure_inner, pressure_outer = 1.0, 2.0, 1e-3, 0.0
    mu, lame_lambda = 1.0, 4.0 / 3.0
    a = (pressure_inner * inner**2 - pressure_outer * outer**2) / (outer**2 - inner**2)
    b = (pressure_inner - pressure_outer) * inner**2 * outer**2 / (outer**2 - inner**2)
    reference_displacement = a * radius / (2.0 * (mu + lame_lambda)) + b / (
        2.0 * mu * radius
    )
    reference_radial_stress = a - b / radius**2
    reference_hoop_stress = a + b / radius**2
    assert np.all(displacement > 0.0)
    assert np.linalg.norm(displacement) > 0.5 * np.linalg.norm(reference_displacement)
    np.testing.assert_allclose(
        volume.sum(), np.pi * (outer**2 - inner**2) * 0.25, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        np.asarray(state.position)[:, 1], initial[:, 1], rtol=0.0, atol=2e-12
    )

    # The equivalent pressure is distributed to the innermost particle layer,
    # offset dr/2 from the surface. At 24 cells, grid spacing/thickness=1/24;
    # a 10% relative L2 tolerance covers this surface smearing and quadratic-grid
    # stress recovery. p/mu=1e-3 makes finite-strain differences subordinate.
    # Exclude two boundary cells only for stress; displacement includes ALL rings.
    relative_displacement_error = np.sqrt(
        np.sum(volume * (displacement - reference_displacement) ** 2)
        / np.sum(volume * reference_displacement**2)
    )
    assert relative_displacement_error < 0.10
    deformation = np.asarray(state.deformation_gradient)
    kirchhoff = np.asarray(state.first_piola) @ np.swapaxes(deformation, -1, -2)
    cauchy = kirchhoff / np.linalg.det(deformation)[:, None, None]
    assert np.all(np.isfinite(cauchy))
    interior = (radius > inner + 2.0 / 24.0) & (radius < outer - 2.0 / 24.0)
    for actual, reference in (
        (cauchy[:, 0, 0], reference_radial_stress),
        (cauchy[:, 2, 2], reference_hoop_stress),
    ):
        relative_stress_error = np.sqrt(
            np.sum(volume[interior] * (actual[interior] - reference[interior]) ** 2)
            / np.sum(volume[interior] * reference[interior] ** 2)
        )
        assert relative_stress_error < 0.10

    # Establish actual relaxation rather than an analytic initialized equilibrium:
    # kinetic energy rises from rest and then decays; the late displacement is
    # stationary independently of agreement with the Lamé displacement.
    kinetic = np.asarray(result.kinetic_energy)
    mean_displacement = np.asarray(result.mean_radial_displacement)
    assert kinetic.max() > 0.0
    assert kinetic[-1] < 1e-3 * kinetic.max()
    assert abs(mean_displacement[-1] - mean_displacement[-201]) < 0.005 * abs(
        mean_displacement[-1]
    )
