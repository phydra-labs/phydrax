#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax.applications.numerical_relativity._boundaries import (
    AnalyticBoundary,
    CharacteristicRadiativeBoundary,
    PeriodicBoundary,
)
from phydrax.applications.numerical_relativity._derivatives import (
    FourthOrderDerivatives,
)
from phydrax.applications.numerical_relativity._enforcement import (
    Z4cAlgebraicEnforcement,
)
from phydrax.applications.numerical_relativity._gauge import (
    GeodesicGauge,
    HarmonicGauge,
    MovingPunctureGauge,
)
from phydrax.applications.numerical_relativity._grid import FixedGridGeometry
from phydrax.applications.numerical_relativity._state import (
    flat_z4c_state,
    make_z4c_state,
)
from phydrax.applications.numerical_relativity._status import (
    NumericalRelativityStatus,
)
from phydrax.applications.numerical_relativity._temporal import FixedGridZ4cRuntime
from phydrax.applications.numerical_relativity._z4c import (
    evaluate_z4c_rhs,
    z4c_adm_geometry,
    Z4cSystem,
)
from phydrax.metrix._adm_exchange import StressEnergyProjection
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import KILOGRAM


def _system(*, constraint_damping=0.02, tolerance=1.0e-5):
    return Z4cSystem(
        RelativityScaleContract.geometric(KILOGRAM),
        RelativityConvention.canonical(),
        chart_id="cartesian",
        constraint_damping=constraint_damping,
        constraint_tolerance=tolerance,
    )


def test_z4c_requires_geometric_units_and_rejects_complex_fields():
    with pytest.raises(ValueError, match="G=c=1"):
        Z4cSystem(
            RelativityScaleContract.si(),
            RelativityConvention.canonical(),
            chart_id="cartesian",
        )

    flat = flat_z4c_state((5, 5, 5), grid_id="complex-rejection")
    with pytest.raises(TypeError, match="real"):
        make_z4c_state(
            flat.chi.astype(jnp.complex64) + 1.0j,
            flat.conformal_metric,
            flat.k_hat,
            flat.conformal_extrinsic_curvature,
            flat.theta,
            flat.conformal_connection,
            flat.lapse,
            flat.shift,
            flat.shift_driver,
            grid_id=flat.grid_id,
        )


def test_fourth_order_polynomial_identities_include_nonperiodic_edges():
    shape = (9, 8, 7)
    spacing = (0.2, 0.3, 0.4)
    operators = FourthOrderDerivatives(shape, spacing, boundary="one_sided")
    x = spacing[0] * jnp.arange(shape[0])
    y = spacing[1] * jnp.arange(shape[1])
    z = spacing[2] * jnp.arange(shape[2])
    x, y, z = jnp.meshgrid(x, y, z, indexing="ij")
    polynomial = x**4 - 2.0 * y**3 + 0.5 * z**2 + x * y

    np.testing.assert_allclose(
        operators.first(polynomial, 0), 4.0 * x**3 + y, rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        operators.second(polynomial, 1), -12.0 * y, rtol=3e-4, atol=3e-4
    )
    np.testing.assert_allclose(
        operators.mixed_second(polynomial, 0, 1), 1.0, rtol=3e-4, atol=3e-4
    )
    positive_speed = jnp.ones(shape)
    np.testing.assert_allclose(
        operators.upwind(polynomial, positive_speed, 0),
        4.0 * x**3 + y,
        rtol=3e-4,
        atol=3e-4,
    )

    dissipative = FourthOrderDerivatives(
        (8, 8, 8), (1.0, 1.0, 1.0), dissipation_strength=0.2
    )
    checkerboard = (-1.0) ** sum(
        jnp.meshgrid(
            jnp.arange(8), jnp.arange(8), jnp.arange(8), indexing="ij"
        )
    )
    damping = dissipative.dissipation(checkerboard)
    assert float(jnp.sum(damping * checkerboard)) < 0.0


def test_flat_vacuum_rhs_constraints_and_jit_shape():
    grid = FixedGridGeometry((5, 5, 5), (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), periodic=True)
    derivatives = FourthOrderDerivatives(grid.shape, grid.spacing)
    system = _system(tolerance=1.0e-6)
    gauge = GeodesicGauge()
    state = flat_z4c_state(grid.shape, grid_id=grid.grid_id)

    result = evaluate_z4c_rhs(
        system, grid, derivatives, gauge, state, snapshot_token=0
    )
    np.testing.assert_allclose(result.rates.values, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.constraints.maximum_norm, 0.0, atol=2e-6)
    assert bool(result.finite)
    assert bool(result.physically_valid)
    assert bool(result.constraints.qualified)

    compiled = jax.jit(
        lambda values: evaluate_z4c_rhs(
            system,
            grid,
            derivatives,
            gauge,
            state.with_values(values),
            snapshot_token=0,
        ).rates.values
    )(state.values)
    assert compiled.shape == state.values.shape
    np.testing.assert_allclose(compiled, 0.0, atol=2e-6)


def test_harmonic_and_linearized_tensor_wave_rates():
    shape = (9, 5, 5)
    spacing = (0.25, 1.0, 1.0)
    grid = FixedGridGeometry(shape, (0.0, 0.0, 0.0), spacing, periodic=True)
    derivatives = FourthOrderDerivatives(shape, spacing)
    system = _system(tolerance=1.0)
    flat = flat_z4c_state(shape, grid_id=grid.grid_id)
    wave = 1.0e-4 * jnp.sin(2.0 * jnp.pi * grid.coordinates[0] / (shape[0] * spacing[0]))
    metric = flat.conformal_metric.at[1, 1].add(wave).at[2, 2].add(-wave)
    k_hat = 2.0e-4 * jnp.ones(shape)
    state = make_z4c_state(
        flat.chi,
        metric,
        k_hat,
        flat.conformal_extrinsic_curvature,
        flat.theta,
        flat.conformal_connection,
        flat.lapse,
        flat.shift,
        flat.shift_driver,
        grid_id=grid.grid_id,
    )

    result = evaluate_z4c_rhs(
        system,
        grid,
        derivatives,
        HarmonicGauge(),
        state,
        snapshot_token=0,
    )
    np.testing.assert_allclose(result.rates.lapse, -k_hat, rtol=3e-4, atol=2e-7)
    expected_yy = -0.5 * derivatives.second(wave, 0)
    np.testing.assert_allclose(
        result.rates.conformal_extrinsic_curvature[1, 1],
        expected_yy,
        rtol=2e-2,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result.rates.conformal_extrinsic_curvature[2, 2],
        -expected_yy,
        rtol=2e-2,
        atol=2e-6,
    )
    driver = 3.0e-4 * jnp.ones_like(flat.shift_driver)
    puncture_state = make_z4c_state(
        flat.chi,
        flat.conformal_metric,
        k_hat,
        flat.conformal_extrinsic_curvature,
        flat.theta,
        flat.conformal_connection,
        flat.lapse,
        flat.shift,
        driver,
        grid_id=grid.grid_id,
    )
    puncture_rates = MovingPunctureGauge(
        driver_damping=1.5, advective=False
    ).rates(
        puncture_state,
        derivatives,
        jnp.zeros_like(flat.conformal_connection),
    )
    np.testing.assert_allclose(puncture_rates.lapse, -2.0 * k_hat)
    np.testing.assert_allclose(puncture_rates.shift, 0.75 * driver)
    np.testing.assert_allclose(puncture_rates.shift_driver, -1.5 * driver)


def test_stress_energy_and_constraint_damping_are_explicit():
    grid = FixedGridGeometry((5, 5, 5), (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), periodic=True)
    derivatives = FourthOrderDerivatives(grid.shape, grid.spacing)
    system = _system(constraint_damping=0.3, tolerance=1.0)
    gauge = GeodesicGauge()
    flat = flat_z4c_state(grid.shape, grid_id=grid.grid_id)
    geometry = z4c_adm_geometry(system, grid, flat, snapshot_token=11)
    density = 1.0e-4 * jnp.ones(grid.shape)
    projection = StressEnergyProjection(
        density,
        jnp.zeros(grid.shape + (3,)),
        jnp.zeros(grid.shape + (3, 3)),
        jnp.ones(grid.shape, dtype=bool),
        jnp.ones(grid.shape, dtype=bool),
        jnp.zeros(grid.shape),
        jnp.zeros(grid.shape),
        snapshot_token=geometry.snapshot_token,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id="constant-density",
    )
    sourced = evaluate_z4c_rhs(
        system,
        grid,
        derivatives,
        gauge,
        flat,
        snapshot_token=geometry.snapshot_token,
        stress_energy=projection,
    )
    np.testing.assert_allclose(
        sourced.rates.k_hat,
        0.5 * system.einstein_coupling * density,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        sourced.rates.theta,
        -system.einstein_coupling * density,
        rtol=2e-5,
    )
    assert bool(sourced.source_valid)
    later_geometry = z4c_adm_geometry(system, grid, flat, snapshot_token=12)
    assert later_geometry.geometry_lineage_id == geometry.geometry_lineage_id
    assert not bool(projection.compatible_with(later_geometry))
    stale = evaluate_z4c_rhs(
        system,
        grid,
        derivatives,
        gauge,
        flat,
        snapshot_token=later_geometry.snapshot_token,
        stress_energy=projection,
    )
    assert not bool(stale.source_valid)

    theta = 1.0e-4 * jnp.ones(grid.shape)
    damped_state = make_z4c_state(
        flat.chi,
        flat.conformal_metric,
        flat.k_hat,
        flat.conformal_extrinsic_curvature,
        theta,
        flat.conformal_connection,
        flat.lapse,
        flat.shift,
        flat.shift_driver,
        grid_id=grid.grid_id,
    )
    damped = evaluate_z4c_rhs(
        system,
        grid,
        derivatives,
        gauge,
        damped_state,
        snapshot_token=0,
    )
    expected = (4.0 / 3.0) * theta**2 - 0.6 * theta
    np.testing.assert_allclose(damped.rates.theta, expected, rtol=2e-5, atol=2e-8)


def test_boundary_evidence_enforcement_and_atomic_flat_step():
    grid = FixedGridGeometry((6, 6, 6), (-2.5, -2.5, -2.5), (1.0, 1.0, 1.0), periodic=False)
    derivatives = FourthOrderDerivatives(grid.shape, grid.spacing, boundary="one_sided")
    flat = flat_z4c_state(grid.shape, grid_id=grid.grid_id)

    analytic = AnalyticBoundary(lambda time, coordinates: flat.values, "flat", width=1)
    perturbed = flat.with_values(flat.values.at[18, 0].set(0.7))
    analytic_result = analytic.apply_state(jnp.asarray(0.0), perturbed, grid)
    assert bool(analytic_result.evidence.analytic)
    assert bool(analytic_result.evidence.successful)
    assert int(analytic_result.evidence.applied_points) > 0
    np.testing.assert_allclose(analytic_result.state.lapse[0], 1.0)

    asymptotic = np.asarray(flat.values[:, 0, 0, 0])
    radiative = CharacteristicRadiativeBoundary(asymptotic)
    rate = flat.with_values(jnp.zeros_like(flat.values))
    radiative_result = radiative.apply_rates(
        jnp.asarray(0.0), flat, rate, grid, derivatives
    )
    assert bool(radiative_result.evidence.characteristic)
    assert bool(radiative_result.evidence.successful)
    assert int(radiative_result.evidence.applied_points) > 0
    np.testing.assert_allclose(radiative_result.state.values, 0.0, atol=2e-6)

    metric = flat.conformal_metric * 1.01
    extrinsic = flat.conformal_extrinsic_curvature.at[0, 0].set(0.03)
    unconstrained = make_z4c_state(
        flat.chi,
        metric,
        flat.k_hat,
        extrinsic,
        flat.theta,
        flat.conformal_connection,
        flat.lapse,
        flat.shift,
        flat.shift_driver,
        grid_id=grid.grid_id,
    )
    enforcement = Z4cAlgebraicEnforcement()
    enforced = enforcement.apply(unconstrained)
    assert bool(enforced.evidence.successful)
    np.testing.assert_allclose(enforced.evidence.determinant_defect_after, 0.0, atol=2e-5)
    np.testing.assert_allclose(enforced.evidence.trace_defect_after, 0.0, atol=2e-6)

    periodic_grid = FixedGridGeometry((5, 5, 5), (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), periodic=True)
    runtime = FixedGridZ4cRuntime(
        _system(tolerance=1.0e-5),
        periodic_grid,
        FourthOrderDerivatives(periodic_grid.shape, periodic_grid.spacing),
        GeodesicGauge(),
        PeriodicBoundary(),
        enforcement,
        time_step=0.1,
        integrator="ssprk33",
    )
    initial = runtime.initialize(
        flat_z4c_state(periodic_grid.shape, grid_id=periodic_grid.grid_id)
    )
    observed_tokens = []

    def zero_stress_energy(stage_time, geometry):
        observed_tokens.append(int(geometry.snapshot_token))
        shape = geometry.leading_shape
        dtype = geometry.alpha.dtype
        return StressEnergyProjection(
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape + (3,), dtype=dtype),
            jnp.zeros(shape + (3, 3), dtype=dtype),
            jnp.ones(shape, dtype=bool),
            jnp.ones(shape, dtype=bool),
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
            snapshot_token=geometry.snapshot_token,
            geometry_lineage_id=geometry.geometry_lineage_id,
            convention_id=geometry.convention_id,
            scale_id=geometry.scale_id,
            topology_id=geometry.topology_id,
            projection_id="zero-stage-source",
        )

    proposal = runtime.evaluate(
        initial, stress_energy_provider=zero_stress_energy
    )
    assert observed_tokens == [1, 2, 3, 7]
    assert bool(proposal.successful)
    accepted = runtime.accept(proposal)
    assert int(accepted.step_index) == 1
    np.testing.assert_allclose(accepted.state.values, initial.state.values, atol=2e-6)
    rolled_back = runtime.accept(proposal, False)
    assert int(rolled_back.step_index) == 0
    np.testing.assert_allclose(rolled_back.state.values, initial.state.values)

    flat_periodic = initial.state
    constrained = make_z4c_state(
        flat_periodic.chi,
        flat_periodic.conformal_metric,
        flat_periodic.k_hat,
        flat_periodic.conformal_extrinsic_curvature,
        1.0e-2 * jnp.ones(periodic_grid.shape),
        flat_periodic.conformal_connection,
        flat_periodic.lapse,
        flat_periodic.shift,
        flat_periodic.shift_driver,
        grid_id=periodic_grid.grid_id,
    )
    constrained_initial = runtime.initialize(constrained)
    rejected = runtime.evaluate(constrained_initial)
    assert not bool(rejected.qualified)
    assert not bool(rejected.successful)
    assert int(rejected.status) & int(
        NumericalRelativityStatus.CONSTRAINT_TOLERANCE_EXCEEDED
    )
    constrained_rollback = runtime.accept(rejected)
    assert int(constrained_rollback.step_index) == 0
    np.testing.assert_allclose(
        constrained_rollback.state.values, constrained_initial.state.values
    )
