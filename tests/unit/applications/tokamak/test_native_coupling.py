import numpy as np

import phydrax as phx


def _manufactured_equilibrium_plan(count=9):
    r = np.linspace(1.0, 3.0, count)
    z = np.linspace(-1.0, 1.0, count)
    return phx.applications.tokamak.FixedBoundaryGradShafranovPlan(
        r,
        z,
        phx.applications.tokamak.TokamakMagneticConvention.canonical(),
        relative_tolerance=1.0e-10,
        maximum_steps=256,
    ).prepare()


def _manufactured_current_boundary(plan):
    rr, zz = np.meshgrid(np.asarray(plan.r_m), np.asarray(plan.z_m))
    psi = (rr - 2.0) ** 2 + zz**2
    current = np.zeros_like(psi)
    mu0 = phx.applications.tokamak.DEFAULT_VACUUM_PERMEABILITY_H_M
    current[1:-1, 1:-1] = -(2.0 + 4.0 / rr[1:-1, 1:-1]) / (mu0 * rr[1:-1, 1:-1])
    boundary = np.zeros_like(psi)
    boundary[0] = psi[0]
    boundary[-1] = psi[-1]
    boundary[:, 0] = psi[:, 0]
    boundary[:, -1] = psi[:, -1]
    return current, boundary


def test_free_boundary_step_couples_circuit_boundary_and_equilibrium():
    equilibrium = _manufactured_equilibrium_plan()
    current, boundary = _manufactured_current_boundary(equilibrium)
    circuit = phx.circuit.CoupledInductancePlan(
        np.asarray([[1.0]]), np.asarray([[0.1]]), ("pf",)
    ).prepare()
    response_values = np.zeros(
        (equilibrium.z_m.size, equilibrium.r_m.size, 1), dtype=float
    )
    response = phx.applications.tokamak.AxisymmetricCoilResponsePlan(
        response_values,
        (phx.applications.tokamak.TokamakWindingRole.ACTIVE,),
        "synthetic-green-response",
        ("pf",),
    )
    free = phx.applications.tokamak.FreeBoundaryTokamakPlan(
        equilibrium, circuit, response
    ).prepare()
    result = free.step(free.state([0.0]), [0.0], current, boundary, 0.1)

    assert bool(result.successful)
    np.testing.assert_allclose(result.accepted_state.winding_current_a, [0.0])
    assert result.equilibrium.pde_residual_norm < 1.0e-6


def test_filament_green_response_is_finite_reciprocal_boundary_data():
    equilibrium = _manufactured_equilibrium_plan()
    response = phx.applications.tokamak.AxisymmetricCoilResponsePlan.from_filament_coils(
        equilibrium.r_m,
        equilibrium.z_m,
        (phx.applications.tokamak.AxisymmetricFilamentCoil("pf", 4.0, 0.0),),
        (phx.applications.tokamak.TokamakWindingRole.ACTIVE,),
        source_id="analytic-filament-green",
        minimum_distance_m=0.1,
    )

    values = response.boundary_flux_per_amp_wb_per_rad[..., 0]
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], values[-1], rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_array_equal(values[1:-1, 1:-1], 0.0)
    assert response.winding_ids == ("pf",)


def test_current_diffusion_preserves_flux_without_source_or_boundary_rate():
    theta = 2.0 * np.pi * np.arange(16) / 16
    contours = np.zeros((3, 16, 2))
    contours[0, :, 0] = 2.0
    for index, radius in enumerate((0.0, 0.3, 0.6)):
        contours[index, :, 0] = 2.0 + radius * np.cos(theta)
        contours[index, :, 1] = radius * np.sin(theta)
    geometry = phx.applications.tokamak.FluxSurfaceGeometry(
        np.asarray([0.0, 0.5, 0.9]),
        contours,
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 2.0, 4.0]),
        np.asarray([2.0, 2.0, 2.0]),
        np.asarray([0.0, 0.3, 0.6]),
        np.asarray([1.0, 1.5, 2.0]),
        "synthetic-equilibrium",
    )
    prepared = phx.applications.tokamak.CurrentDiffusionPlan(geometry).prepare()
    state = phx.applications.tokamak.CurrentDiffusionState([0.1, 0.2])
    result = prepared.step(state, 0.1, np.zeros(3), np.zeros(2), 0.0)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted_state.poloidal_flux_wb_per_rad,
        state.poloidal_flux_wb_per_rad,
    )
    np.testing.assert_allclose(result.balance_residual_wb, 0.0, atol=1.0e-14)
