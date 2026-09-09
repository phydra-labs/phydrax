#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.porous_media import (
    CoupledWaterHeatPlan,
    PorousBoundaryConditions,
    PorousMaterial,
    PorousThermalMaterial,
    RichardsPlan,
    VanGenuchtenMualem,
)
from phydrax.discretization import UnstructuredFiniteVolumePlan
from phydrax.discretization.finite_volume import (
    HybridDiffusionBoundary,
    HybridMimeticDiffusion,
)
from phydrax.linalg import DenseLU, LinearSolvePolicy
from phydrax.nonlinear import NewtonKrylov, NonlinearTermination


def _geometry():
    return UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    ).prepare()


def _nonlinear_method():
    return NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU()))


def _termination():
    return NonlinearTermination(
        absolute_residual=1.0e-11,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=30,
    )


def _exterior(discretization):
    return np.flatnonzero(np.asarray(discretization.neighbour_cells) < 0)


def test_hybrid_mimetic_rotated_tensor_is_affine_exact_and_conservative():
    discretization = _geometry()
    diffusion = HybridMimeticDiffusion(discretization)
    gradient = jnp.asarray([1.0, -2.0, 0.5])
    tensor = jnp.asarray(((2.0, 0.4, -0.2), (0.4, 1.5, 0.3), (-0.2, 0.3, 3.0)))
    cells = discretization.cell_centers @ gradient + 0.7
    faces = discretization.face_centers @ gradient + 0.7
    local = diffusion.local_fluxes(cells, faces, tensor)
    expected = -jnp.einsum("cfi,ij,j->cf", diffusion.outward_areas, tensor, gradient)
    np.testing.assert_allclose(
        local, jnp.where(diffusion.valid, expected, 0.0), atol=2.0e-14
    )
    interior = np.asarray(discretization.neighbour_cells) >= 0
    np.testing.assert_allclose(
        diffusion.continuity_residual(local)[interior], 0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(jnp.sum(local, axis=1), 0.0, atol=2.0e-14)


def test_hybrid_global_solve_recovers_affine_dirichlet_field():
    discretization = _geometry()
    diffusion = HybridMimeticDiffusion(discretization)
    gradient = jnp.asarray([0.3, -0.5, 0.9])
    expected_cells = discretization.cell_centers @ gradient + 2.0
    expected_faces = discretization.face_centers @ gradient + 2.0
    boundary = HybridDiffusionBoundary(
        discretization,
        dirichlet={
            int(face): float(expected_faces[face]) for face in _exterior(discretization)
        },
    )
    result = diffusion.solve(jnp.eye(3), boundary, policy=LinearSolvePolicy(DenseLU()))
    assert result.successful
    np.testing.assert_allclose(
        result.value[: expected_cells.size], expected_cells, atol=1.0e-12
    )
    np.testing.assert_allclose(
        result.value[expected_cells.size :], expected_faces, atol=1.0e-12
    )


def _hydrostatic_plan(*, anchored=True, thermal_feedback=False):
    discretization = _geometry()
    density = 1000.0
    gravity = -9.80665
    face_pressure = (
        1.0e5 + density * gravity * np.asarray(discretization.face_centers)[:, 2]
    )
    pressure_boundary = (
        {int(face): float(face_pressure[face]) for face in _exterior(discretization)}
        if anchored
        else None
    )
    material = PorousMaterial(
        0.3,
        jnp.asarray(
            ((2.0e-12, 0.4e-12, 0.0), (0.4e-12, 1.0e-12, 0.0), (0.0, 0.0, 3.0e-12))
        ),
        thermal_expansion_K_inverse=2.0e-4 if thermal_feedback else 0.0,
        viscosity_temperature_K_inverse=0.02 if thermal_feedback else 0.0,
    )
    plan = RichardsPlan(
        discretization,
        material,
        VanGenuchtenMualem(1.0e-5, 2.0),
        PorousBoundaryConditions(discretization, pressure_Pa=pressure_boundary),
        method=_nonlinear_method(),
        termination=_termination(),
    )
    pressure = 1.0e5 + density * gravity * discretization.cell_centers[:, 2]
    return plan, pressure, jnp.asarray(face_pressure)


def test_richards_hydrostatic_state_has_zero_flux_and_is_preserved():
    plan, pressure, face_pressure = _hydrostatic_plan()
    previous = plan.initialize(pressure, face_pressure)
    flux = plan.fluxes(previous.pressure_Pa, previous.face_pressure_Pa)
    np.testing.assert_allclose(flux.mass_face_rates, 0.0, atol=1.0e-15)
    result = plan.step(previous, 10.0)
    assert result.successful
    np.testing.assert_allclose(
        result.state.pressure_Pa, previous.pressure_Pa, atol=1.0e-9
    )
    np.testing.assert_allclose(result.residual, 0.0, atol=1.0e-10)


def test_saturated_closed_incompressible_richards_problem_fails_without_storage_floor():
    plan, pressure, face_pressure = _hydrostatic_plan(anchored=False)
    previous = plan.initialize(pressure, face_pressure)
    assert not plan.well_posed(previous.pressure_Pa)
    result = plan.step(previous, 1.0)
    assert not result.successful
    np.testing.assert_allclose(result.state.pressure_Pa, previous.pressure_Pa)


def test_unsaturated_richards_step_has_implicit_forward_and_reverse_derivatives():
    discretization = _geometry()
    boundary = PorousBoundaryConditions(
        discretization,
        pressure_Pa={int(face): -2.0e4 for face in _exterior(discretization)},
    )
    plan = RichardsPlan(
        discretization,
        PorousMaterial(0.3, 1.0e-12),
        VanGenuchtenMualem(1.0e-5, 2.0),
        boundary,
        gravity_m_s2=(0.0, 0.0, 0.0),
        method=_nonlinear_method(),
        termination=_termination(),
    )
    previous = plan.initialize(-2.0e4)

    def total_mass(source):
        result = plan.step(previous, 5.0, source_kg_s=jnp.asarray((source, 0.0)))
        return jnp.sum(result.state.water_mass_kg)

    value, tangent = jax.jvp(total_mass, (jnp.asarray(1.0e-5),), (jnp.asarray(1.0),))
    gradient = jax.grad(total_mass)(jnp.asarray(1.0e-5))
    finite_difference = (
        total_mass(jnp.asarray(1.01e-5)) - total_mass(jnp.asarray(0.99e-5))
    ) / 2.0e-7
    assert jnp.isfinite(value) & jnp.isfinite(tangent)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-5)
    np.testing.assert_allclose(gradient, tangent, rtol=2.0e-8)


def test_monolithic_water_heat_preserves_hydrostatic_isothermal_state():
    water, pressure, face_pressure = _hydrostatic_plan(thermal_feedback=True)
    thermal_boundary = HybridDiffusionBoundary(
        water.discretization,
        dirichlet={int(face): 300.0 for face in _exterior(water.discretization)},
    )
    coupled = CoupledWaterHeatPlan(
        water,
        PorousThermalMaterial(
            jnp.asarray(((2.0, 0.2, 0.0), (0.2, 1.5, 0.0), (0.0, 0.0, 3.0))),
            dry_conductivity_W_m_K=1.0,
        ),
        thermal_boundary,
    )
    previous = coupled.initialize(
        pressure,
        300.0,
        face_pressure_Pa=face_pressure,
        face_temperature_K=jnp.full(face_pressure.shape, 300.0),
    )
    result = coupled.step(previous, 2.0)
    assert result.successful
    np.testing.assert_allclose(
        result.state.pressure_Pa, previous.pressure_Pa, atol=1.0e-8
    )
    np.testing.assert_allclose(result.state.temperature_K, 300.0, atol=1.0e-10)
    np.testing.assert_allclose(result.residual, 0.0, atol=1.0e-9)
