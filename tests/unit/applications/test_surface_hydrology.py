#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.applications.porous_media import (
    CoupledWaterHeatPlan,
    PorousBoundaryConditions,
    PorousMaterial,
    PorousThermalMaterial,
    RichardsPlan,
    SurfaceRichardsPlan,
    SurfaceWaterHeatPlan,
    VanGenuchtenMualem,
)
from phydrax.applications.porous_media._surface_exchange import (
    OrthogonalDiffusiveWaveSurfacePlan,
)
from phydrax.discretization import CellMesh
from phydrax.discretization._boundary_trace import BoundarySurfaceTrace
from phydrax.discretization.finite_volume import (
    HybridDiffusionBoundary,
    UnstructuredFiniteVolumePlan,
)
from phydrax.linalg import DenseLU, LinearSolvePolicy
from phydrax.nonlinear import NewtonKrylov, NonlinearTermination


def _surface(*, lateral=True):
    vertices = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (2, 1, 1),
        ],
        dtype=float,
    )
    cells = (
        (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
        (
            (1, 2, 9, 8),
            (5, 10, 11, 6),
            (1, 8, 10, 5),
            (8, 9, 11, 10),
            (9, 2, 6, 11),
            (2, 1, 5, 6),
        ),
    )
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(
        CellMesh.from_polyhedra(vertices, cells)
    ).prepare()
    faces = np.flatnonzero(np.asarray(fv.area_vectors)[:, 2] > 0.9)
    trace = BoundarySurfaceTrace(fv, faces)
    return fv, OrthogonalDiffusiveWaveSurfacePlan(
        trace, spatial=SpatialCoordinateContract.si(), lateral=lateral
    )


def test_boundary_trace_parent_rates_cancel_and_transpose():
    fv, plan = _surface()
    rates = jnp.asarray([2.0, -3.0])
    np.testing.assert_allclose(
        jnp.sum(plan.trace.volume_content_rate(rates)) + jnp.sum(rates), 0
    )
    face_values = jnp.arange(fv.face_measures.size, dtype=float)
    np.testing.assert_allclose(
        jnp.vdot(plan.trace.gather(face_values), rates),
        jnp.vdot(face_values, plan.trace.scatter(rates)),
    )
    interior = np.flatnonzero(np.asarray(fv.neighbour_cells) >= 0)
    with pytest.raises(ValueError, match="interior"):
        BoundarySurfaceTrace(fv, interior)


def test_runoff_transfers_water_and_energy_without_creating_inventory():
    _, plan = _surface()
    initial = plan.initial_state(jnp.asarray([0.2, 0.1]), jnp.asarray([310.0, 290.0]))
    result = plan.step(initial, 0.01)
    assert result.successful
    assert result.state.volume[0] < initial.volume[0]
    assert result.state.volume[1] > initial.volume[1]
    np.testing.assert_allclose(
        jnp.sum(result.state.volume), jnp.sum(initial.volume), atol=1e-14
    )
    np.testing.assert_allclose(
        jnp.sum(result.state.energy), jnp.sum(initial.energy), rtol=1e-14
    )
    np.testing.assert_allclose(result.volume_residual, 0, atol=1e-14)
    np.testing.assert_allclose(result.energy_residual, 0, atol=1e-8)


def test_dry_reservoir_limits_shared_infiltration_rate_not_inventory():
    _, plan = _surface(lateral=False)
    initial = plan.initial_state(0.0)
    result = plan.step(initial, 10.0, rainfall=0.001, infiltration_demand=1.0)
    assert result.successful
    assert result.limited
    assert not result.derivative_available
    np.testing.assert_allclose(result.infiltration_rate, 0.001, atol=1e-15)
    np.testing.assert_allclose(result.state.volume, 0, atol=1e-15)
    np.testing.assert_allclose(result.volume_residual, 0, atol=1e-15)
    np.testing.assert_allclose(result.energy_residual, 0, atol=1e-8)


def test_exfiltration_carries_subsurface_enthalpy_into_surface():
    _, plan = _surface(lateral=False)
    initial = plan.initial_state(0.0)
    result = plan.step(
        initial, 2.0, infiltration_demand=-0.01, subsurface_temperature=320.0
    )
    np.testing.assert_allclose(result.state.volume, 0.02)
    np.testing.assert_allclose(plan.temperature(result.state), 320.0)
    np.testing.assert_allclose(
        jnp.sum(result.state.energy) + 2 * jnp.sum(result.infiltration_energy_rate),
        0,
        atol=1e-8,
    )


def test_wet_interior_rainfall_derivative_preserves_mass_balance():
    _, plan = _surface(lateral=False)
    initial = plan.initial_state(0.1)
    derivative = jax.grad(
        lambda rain: jnp.sum(plan.step(initial, 3.0, rainfall=rain).state.volume)
    )(0.001)
    np.testing.assert_allclose(
        derivative, 3.0 * jnp.sum(plan.projected_areas), rtol=1e-12
    )


def _equilibrium_subsurface(*, heat=False, surface_depth=0.1, dry_head=0.0):
    discretization, surface = _surface()
    density, gravity = 1000.0, 9.80665
    top_pressure = density * gravity * (surface_depth - dry_head)
    cell_pressure = top_pressure + density * gravity * (
        1.0 - discretization.cell_centers[:, 2]
    )
    face_pressure = top_pressure + density * gravity * (
        1.0 - discretization.face_centers[:, 2]
    )
    top = set(np.asarray(surface.trace.parent_faces).tolist())
    pressure = {
        int(face): float(face_pressure[face])
        for face in np.flatnonzero(np.asarray(discretization.neighbour_cells) < 0)
        if int(face) not in top
    }
    boundary = PorousBoundaryConditions(discretization, pressure_Pa=pressure)
    material = PorousMaterial(
        0.3,
        1.0e-12,
        viscosity_temperature_K_inverse=0.01 if heat else 0.0,
    )
    termination = NonlinearTermination(
        absolute_residual=1.0e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=30,
    )
    water = RichardsPlan(
        discretization,
        material,
        VanGenuchtenMualem(1.0e-5, 2.0),
        boundary,
        method=NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU())),
        termination=termination,
    )
    return water, surface, cell_pressure, face_pressure


def test_monolithic_wet_surface_richards_preserves_shared_hydrostatic_flux():
    water, surface, cell_pressure, face_pressure = _equilibrium_subsurface()
    previous = water.initialize(cell_pressure, face_pressure, temperature_K=300.0)
    surface_state = surface.initial_state(0.1, 300.0)
    result = SurfaceRichardsPlan(water, surface, termination=water.termination).step(
        previous, surface_state, 10.0
    )
    assert result.successful
    assert result.derivative_available
    np.testing.assert_allclose(
        result.porous.pressure_Pa, previous.pressure_Pa, atol=1.0e-8
    )
    np.testing.assert_allclose(result.surface.volume, surface_state.volume, atol=1.0e-12)
    np.testing.assert_allclose(result.exchange_mass_rate, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.mass_residual, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(result.complementarity_residual, 0.0, atol=1.0e-10)


def test_monolithic_dry_surface_richards_preserves_suction_complementarity():
    water, surface, cell_pressure, face_pressure = _equilibrium_subsurface(
        surface_depth=0.0, dry_head=0.1
    )
    previous = water.initialize(cell_pressure, face_pressure, temperature_K=300.0)
    surface_state = surface.initial_state(0.0, 300.0)
    result = SurfaceRichardsPlan(water, surface, termination=water.termination).step(
        previous, surface_state, 10.0
    )
    assert result.successful
    assert result.derivative_available
    np.testing.assert_allclose(result.surface.volume, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.dry_pressure_head, 0.1, atol=1.0e-10)
    np.testing.assert_allclose(result.exchange_mass_rate, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.complementarity_residual, 0.0, atol=1.0e-10)


def test_four_field_surface_water_heat_root_preserves_shared_equilibrium():
    water, surface, cell_pressure, face_pressure = _equilibrium_subsurface(heat=True)
    top = set(np.asarray(surface.trace.parent_faces).tolist())
    thermal_boundary = HybridDiffusionBoundary(
        water.discretization,
        dirichlet={
            int(face): 300.0
            for face in np.flatnonzero(
                np.asarray(water.discretization.neighbour_cells) < 0
            )
            if int(face) not in top
        },
    )
    coupled = CoupledWaterHeatPlan(
        water,
        PorousThermalMaterial(2.0),
        thermal_boundary,
    )
    previous = coupled.initialize(
        cell_pressure,
        300.0,
        face_pressure_Pa=face_pressure,
        face_temperature_K=jnp.full(face_pressure.shape, 300.0),
    )
    surface_state = surface.initial_state(0.1, 300.0)
    result = SurfaceWaterHeatPlan(coupled, surface).step(previous, surface_state, 5.0)
    assert result.successful
    assert result.derivative_available
    np.testing.assert_allclose(
        result.state.pressure_Pa, previous.pressure_Pa, atol=1.0e-8
    )
    np.testing.assert_allclose(result.state.temperature_K, 300.0, atol=1.0e-10)
    np.testing.assert_allclose(result.surface.volume, surface_state.volume, atol=1.0e-12)
    np.testing.assert_allclose(result.surface.energy, surface_state.energy, atol=1.0e-8)
    np.testing.assert_allclose(result.exchange_mass_rate, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.exchange_energy_rate, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.combined_mass_balance, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(result.combined_energy_balance, 0.0, atol=1.0e-7)
