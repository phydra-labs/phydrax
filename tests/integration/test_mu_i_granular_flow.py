#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization.flip._viscous import viscous_flip_stage


def test_small_mu_i_bagnold_steady_profile() -> None:
    """GDR MiDi (2004), EPJE 14:341–365, inclined uniform dense-layer flow.

    The independent analytic profile uses hydrostatic compression and mu(I)=
    tan(theta). This qualifies the pressure-lagged FLIP grid stress stage, not
    a moving-particle runout experiment. Binary interface/edge interpolation
    at 24 cells is first order; a 12% profile band allows surface placement
    error while rejecting omission of viscosity or the pressure dependence.
    """
    D = phx.discretization
    count, depth = 24, 0.75
    density, grain_density, diameter = 1500.0, 2500.0, 0.01
    mu_s, mu_2, I_0, slope = 0.38, 0.64, 0.279, 0.5
    gravity = 9.81
    cosine = 1.0 / np.sqrt(1.0 + slope**2)
    tangential_gravity = gravity * slope * cosine
    inertial_number = I_0 * (slope - mu_s) / (mu_2 - slope)
    coefficient = (
        inertial_number / diameter * np.sqrt(density * gravity * cosine / grain_density)
    )
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(4, periodic=True), D.UniformCellAxisSpec(count)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64))
    mac = D.MACOperatorPlan(D.FiniteVolumePlan(grid).prepare()).prepare()
    boundaries = D.MACBoundaryPlan(
        mac,
        (
            D.MACBoundarySide("y", "lower", "no-slip"),
            D.MACBoundarySide("y", "upper", "free-slip"),
        ),
    ).prepare()
    momentum = D.MACMomentumPlan(mac, boundaries=boundaries).prepare()
    plan = phx.solver.MACVariationalViscosityPlan(momentum, tolerance=1.0e-11)
    y = grid.structured_axes[1].interval_centers
    fluid = y < depth
    pressure = jnp.broadcast_to(
        jnp.where(fluid, density * gravity * cosine * (depth - y), 0.0),
        mac.discretization.cell_shape,
    )
    exact = (2.0 / 3.0) * coefficient * (depth**1.5 - jnp.maximum(depth - y, 0.0) ** 1.5)
    liquid = jnp.broadcast_to(fluid, mac.discretization.cell_shape)
    measures = D.finite_volume.MACFreeSurfaceViscousMeasurePlan(
        mac, density
    ).evaluate_cell_fraction(liquid.astype(jnp.float64), 0.0)
    initial = (
        jnp.broadcast_to(exact, mac.discretization.face_layouts[0].shape),
        jnp.zeros(mac.discretization.face_layouts[1].shape, dtype=jnp.float64),
    )
    law = phx.rheology.MuILaw()
    parameters = phx.rheology.MuIParameters(mu_s, mu_2, I_0, diameter, grain_density)
    method = D.FLIPMethodPlan(
        0.0, viscous_tolerance=1.0e-8, maximum_picard_iterations=100
    )
    stage = boundaries.evaluate(0.0)
    dt = jnp.asarray(0.1, dtype=jnp.float64)

    def advance(
        velocity: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], D.FLIPViscousEvidence]:
        forced = (velocity[0] + dt * tangential_gravity, velocity[1])
        return viscous_flip_stage(
            plan,
            law,
            parameters,
            method,
            measures,
            forced,
            pressure,
            liquid,
            dt,
            stage,
            jnp.asarray(True),
        )

    compiled_advance = eqx.filter_jit(advance)
    velocity = initial
    for _ in range(8):
        velocity, evidence = compiled_advance(velocity)
        assert evidence.successful, {
            "nonlinear_status": int(evidence.nonlinear.status),
            "fixed_point_residual": float(
                evidence.nonlinear.diagnostics.final_residual_norm
            ),
            "momentum_residual": float(evidence.relative_residual),
            "linear_status": int(evidence.linear.linear_status),
            "iterations": int(evidence.iterations),
        }
        assert evidence.pressure_admitted
        assert evidence.relative_residual <= method.viscous_tolerance
        np.testing.assert_array_equal(evidence.lagged_pressure_pa, pressure)
    difference = np.asarray(velocity[0][0, :18] - exact[:18])
    error = float(np.max(np.abs(difference)) / np.max(np.asarray(exact[:18])))
    assert error < 0.12, f"Measured 24-cell Bagnold relative profile error: {error:.8g}"
    # An omitted stress stage increases speed by eight body-force increments.
    unresisted_error = float(8.0 * dt * tangential_gravity / jnp.max(exact[:18]))
    assert unresisted_error > 0.12
