#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._trainable import combine_trainable, partition_trainable
from phydrax.applications.atmosphere._radiation import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)


SIGMA = 5.670374419e-8


def optics(
    *,
    sw=(0.0, 0.0, 0.0, 0.0),
    scatter=(0.0, 0.0, 0.0, 0.0),
    lw=(0.0, 0.0, 0.0, 0.0),
    asymmetry=(0.0, 0.0, 0.0, 0.0),
):
    return ColumnOpticalProperties(
        shortwave_absorption=sw,
        shortwave_scattering=scatter,
        longwave_absorption=lw,
        shortwave_asymmetry=asymmetry,
        reference_id="analytic-test-coefficients-not-observations",
    )


def test_transparent_column_preserves_boundary_fluxes_and_surface_kirchhoff_law():
    plan = ColumnRadiationPlan(optics(), surface_albedo=0.3, surface_emissivity=0.7)
    result = eqx.filter_jit(plan.evaluate)(
        jnp.array([210.0, 280.0]),
        jnp.array([10.0, 100.0]),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
        300.0,
        400.0,
        longwave_down=80.0,
    )
    assert result.successful
    np.testing.assert_allclose(result.shortwave_downward_flux, 400.0)
    np.testing.assert_allclose(result.shortwave_upward_flux, 120.0)
    np.testing.assert_allclose(result.longwave_downward_flux, 80.0)
    emitted = 0.7 * SIGMA * 300.0**4 + 0.3 * 80.0
    np.testing.assert_allclose(result.longwave_upward_flux, emitted)
    np.testing.assert_allclose(result.heating, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.surface_heating, 280.0 + 80.0 - emitted)
    np.testing.assert_allclose(result.space_heating, -result.surface_heating)
    np.testing.assert_allclose(result.budget_residual, 0.0, atol=1e-12)


def test_absorbing_slab_matches_independent_beer_and_lte_boundary_solution():
    sw_depth, lw_depth = 0.37, 0.82
    plan = ColumnRadiationPlan(
        optics(sw=(sw_depth, 0.0, 0.0, 0.0), lw=(lw_depth, 0.0, 0.0, 0.0)),
        surface_albedo=0.25,
        surface_emissivity=0.8,
    )
    result = plan.evaluate([240.0], [1.0], [0.0], [0.0], [0.0], 295.0, 350.0)
    assert result.successful
    sw_t, lw_t = np.exp(-2 * sw_depth), np.exp(-2 * lw_depth)
    lw_source = SIGMA * 240.0**4 * (1 - lw_t)
    sw_down = np.array([350.0, 350.0 * sw_t])
    sw_up = 0.25 * 350.0 * np.array([sw_t**2, sw_t])
    lw_surface = 0.8 * SIGMA * 295.0**4 + 0.2 * lw_source
    lw_up = np.array([lw_surface * lw_t + lw_source, lw_surface])
    lw_down = np.array([0.0, lw_source])
    np.testing.assert_allclose(result.shortwave_upward_flux, sw_up, rtol=1e-12)
    np.testing.assert_allclose(result.shortwave_downward_flux, sw_down, rtol=1e-12)
    np.testing.assert_allclose(result.longwave_upward_flux, lw_up, rtol=1e-12)
    np.testing.assert_allclose(result.longwave_downward_flux, lw_down, rtol=1e-12)
    net = sw_up + lw_up - sw_down - lw_down
    np.testing.assert_allclose(result.heating, net[1:] - net[:-1], atol=1e-11)
    np.testing.assert_allclose(result.budget_residual, 0.0, atol=1e-11)


def test_scattering_slab_matches_independent_two_stream_fundamental_solution():
    absorption, scattering, g, albedo = 0.23, 1.4, 0.6, 0.31
    plan = ColumnRadiationPlan(
        optics(
            sw=(absorption, 0, 0, 0),
            scatter=(scattering, 0, 0, 0),
            asymmetry=(g, 0, 0, 0),
        ),
        surface_albedo=albedo,
    )
    result = plan.evaluate([260.0], [1.0], [0.0], [0.0], [0.0], 290.0, 500.0)
    # Direct homogeneous ODE solution, not the implementation's adding recursion.
    b = scattering * (1 - g)
    a = 2 * absorption + b
    matrix = np.array([[-a, b], [-b, a]])
    k = np.sqrt(a * a - b * b)
    transfer = np.cosh(k) * np.eye(2) + np.sinh(k) / k * matrix
    up_top = (
        500.0
        * (albedo * transfer[0, 0] - transfer[1, 0])
        / (transfer[1, 1] - albedo * transfer[0, 1])
    )
    down_bottom, up_bottom = transfer @ np.array([500.0, up_top])
    assert result.successful
    np.testing.assert_allclose(
        result.shortwave_upward_flux, [up_top, up_bottom], rtol=2e-12
    )
    np.testing.assert_allclose(
        result.shortwave_downward_flux, [500.0, down_bottom], rtol=2e-12
    )


def test_conservative_scattering_thick_and_perfectly_reflecting_limits():
    scattering_depth = 1.0e20
    plan = ColumnRadiationPlan(
        optics(scatter=(scattering_depth, 0, 0, 0)),
        surface_albedo=1.0,
        surface_emissivity=0.0,
    )
    result = eqx.filter_jit(plan.evaluate)(
        [260.0, 270.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 300.0, 400.0
    )
    assert result.successful
    np.testing.assert_allclose(result.shortwave_upward_flux, 400.0, rtol=2e-12)
    np.testing.assert_allclose(result.shortwave_downward_flux, 400.0, rtol=2e-12)
    np.testing.assert_allclose(result.heating, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.surface_heating, 0.0, atol=1e-10)
    black_surface = eqx.tree_at(lambda p: p.surface_albedo, plan, jnp.asarray(0.0))
    transmitted = black_surface.evaluate(
        [260.0], [1.0], [0.0], [0.0], [0.0], 300.0, 400.0
    )
    assert transmitted.successful
    np.testing.assert_allclose(
        transmitted.shortwave_downward_flux[-1],
        400.0 / (1 + scattering_depth),
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        transmitted.shortwave_upward_flux[0],
        400.0 * scattering_depth / (1 + scattering_depth),
        rtol=2e-12,
    )


def test_opaque_blackbody_and_isothermal_bath_are_distinct_limits():
    plan = ColumnRadiationPlan(optics(lw=(1000.0, 0, 0, 0)), surface_emissivity=1.0)
    outgoing = plan.evaluate([250.0], [1.0], [0.0], [0.0], [0.0], 300.0, 0.0)
    assert outgoing.successful
    np.testing.assert_allclose(
        outgoing.longwave_upward_flux[0], SIGMA * 250.0**4, rtol=1e-12
    )
    np.testing.assert_allclose(
        outgoing.longwave_downward_flux[-1], SIGMA * 250.0**4, rtol=1e-12
    )
    np.testing.assert_array_equal(outgoing.shortwave_downward_flux, 0.0)
    # Isothermal atmosphere with empty space still cools at its top. A matching
    # incident blackbody bath instead gives exact thermal equilibrium everywhere.
    bath = ColumnRadiationPlan(optics(lw=(0.31, 0, 0, 0)), surface_emissivity=0.47)
    equilibrium = bath.evaluate(
        [280.0, 280.0],
        [1.0, 3.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        280.0,
        0.0,
        longwave_down=SIGMA * 280.0**4,
    )
    assert equilibrium.successful
    np.testing.assert_allclose(equilibrium.heating, 0.0, atol=2e-11)
    np.testing.assert_allclose(equilibrium.surface_heating, 0.0, atol=2e-11)
    np.testing.assert_allclose(equilibrium.space_heating, 0.0, atol=2e-11)


def test_composing_homogeneous_layers_preserves_both_band_solutions():
    plan = ColumnRadiationPlan(
        optics(
            sw=(0.04, 0.2, 0.3, 0.5),
            scatter=(0.02, 0, 4, 2),
            lw=(0.1, 0.8, 1.5, 2.0),
            asymmetry=(0, 0, 0.7, 0.4),
        ),
        surface_albedo=0.42,
        surface_emissivity=0.71,
    )
    weights = jnp.array([0.13, 0.39, 0.48])
    single = plan.evaluate(
        [260.0], [10.0], [0.4], [0.1], [0.05], 300.0, 430.0, longwave_down=15.0
    )
    split = plan.evaluate(
        jnp.full(3, 260.0),
        10 * weights,
        0.4 * weights,
        0.1 * weights,
        0.05 * weights,
        300.0,
        430.0,
        longwave_down=15.0,
    )
    assert single.successful & split.successful
    for whole, pieces in (
        (single.shortwave_upward_flux, split.shortwave_upward_flux),
        (single.shortwave_downward_flux, split.shortwave_downward_flux),
        (single.longwave_upward_flux, split.longwave_upward_flux),
        (single.longwave_downward_flux, split.longwave_downward_flux),
    ):
        np.testing.assert_allclose(
            pieces[jnp.array([0, 3])], whole, rtol=2e-12, atol=1e-11
        )
    np.testing.assert_allclose(jnp.sum(split.heating), single.heating[0], atol=1e-10)
    np.testing.assert_allclose(split.budget_residual, 0.0, atol=1e-10)


def test_water_cloud_and_solar_forcing_change_actual_band_transfers():
    plan = ColumnRadiationPlan(
        optics(
            lw=(0, 0.2, 1.0, 1.0), scatter=(0, 0, 20.0, 10.0), asymmetry=(0, 0, 0.8, 0.6)
        ),
        surface_albedo=0.1,
    )
    dry = plan.evaluate([240.0], [100.0], [0.0], [0.0], [0.0], 300.0, 400.0)
    moist = plan.evaluate([240.0], [100.0], [5.0], [0.0], [0.0], 300.0, 400.0)
    cloud = plan.evaluate([240.0], [100.0], [5.0], [0.2], [0.0], 300.0, 400.0)
    dark = plan.evaluate([240.0], [100.0], [5.0], [0.2], [0.0], 300.0, 0.0)
    brighter = plan.evaluate([240.0], [100.0], [5.0], [0.2], [0.0], 300.0, 800.0)
    assert (
        dry.successful
        & moist.successful
        & cloud.successful
        & dark.successful
        & brighter.successful
    )
    assert moist.longwave_upward_flux[0] < dry.longwave_upward_flux[0]
    assert moist.longwave_downward_flux[-1] > dry.longwave_downward_flux[-1]
    np.testing.assert_allclose(moist.shortwave_downward_flux, dry.shortwave_downward_flux)
    assert cloud.shortwave_upward_flux[0] > moist.shortwave_upward_flux[0]
    assert cloud.shortwave_downward_flux[-1] < moist.shortwave_downward_flux[-1]
    assert cloud.longwave_downward_flux[-1] > moist.longwave_downward_flux[-1]
    np.testing.assert_array_equal(dark.shortwave_upward_flux, 0.0)
    np.testing.assert_allclose(
        brighter.shortwave_upward_flux, 2 * cloud.shortwave_upward_flux
    )
    np.testing.assert_allclose(brighter.longwave_upward_flux, cloud.longwave_upward_flux)
    # No sign claim about total cloud forcing: SW reflection and LW absorption compete.


def test_invalid_batch_columns_are_rejected_atomically_without_poisoning_neighbors():
    plan = ColumnRadiationPlan(optics(sw=(0.1, 0, 0, 0), lw=(0.2, 0, 0, 0)))
    t = jnp.array([[[250.0, 280.0], [250.0, -1.0]], [[250.0, 280.0], [jnp.nan, 280.0]]])
    water = jnp.zeros((2, 2, 2)).at[1, 0, 0].set(2.0)
    result = eqx.filter_jit(plan.evaluate)(
        t, jnp.ones(2), water, jnp.zeros(2), jnp.zeros(2), 300.0, 400.0
    )
    np.testing.assert_array_equal(result.successful, [[True, False], [False, False]])
    single = plan.evaluate(
        [250.0, 280.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 300.0, 400.0
    )
    np.testing.assert_allclose(result.heating[0, 0], single.heating)
    for output in jax.tree.leaves(result)[:-1]:
        np.testing.assert_array_equal(output[~result.successful], 0.0)
    invalid = eqx.tree_at(lambda p: p.longwave_absorption_scale, plan, -jnp.ones(4))
    rejected = invalid.evaluate([260.0], [1.0], [0.0], [0.0], [0.0], 300.0, 400.0)
    assert not rejected.successful
    np.testing.assert_array_equal(rejected.upward_flux, 0.0)


def test_optical_zero_boundary_has_finite_correct_native_derivative():
    plan = ColumnRadiationPlan(
        optics(sw=(1, 0, 0, 0), lw=(1, 0, 0, 0)), surface_albedo=0.0
    )

    def observable(scale):
        calibrated = eqx.tree_at(
            lambda p: (p.shortwave_absorption_scale, p.longwave_absorption_scale),
            plan,
            (jnp.full(4, scale), jnp.full(4, scale)),
        )
        result = calibrated.evaluate([250.0], [2.0], [0.0], [0.0], [0.0], 300.0, 400.0)
        return jnp.stack(
            (result.shortwave_downward_flux[-1], result.longwave_upward_flux[0])
        )

    derivative = jax.jacfwd(observable)(0.0)
    np.testing.assert_allclose(
        derivative, [-1600.0, -4 * SIGMA * (300.0**4 - 250.0**4)], rtol=2e-12
    )
    trainable, fixed = partition_trainable(plan)
    restored = combine_trainable(trainable, fixed)
    differentiated = eqx.filter_grad(
        lambda p: (
            p.evaluate([250.0], [2.0], [0.0], [0.0], [0.0], 300.0, 400.0).space_heating
        )
    )(restored)
    assert jnp.isfinite(differentiated.longwave_absorption_scale).all()
    assert differentiated.longwave_absorption_scale[0] < 0.0


def test_falling_precipitation_is_neither_dry_gas_nor_suspended_cloud_optics():
    plan = ColumnRadiationPlan(
        optics(sw=(0.03, 0.1, 0.2, 0.3), scatter=(0.01, 0, 3, 2), lw=(0.07, 0.3, 4, 2)),
    )
    baseline = plan.evaluate([260.0], [10.0], [1.0], [0.2], [0.1], 300.0, 400.0)
    precipitating = plan.evaluate(
        [260.0],
        [15.0],
        [1.0],
        [0.2],
        [0.1],
        300.0,
        400.0,
        rain_mass=[3.0],
        snow_mass=2.0,
    )
    assert baseline.successful & precipitating.successful
    np.testing.assert_allclose(
        precipitating.upward_flux, baseline.upward_flux, rtol=1e-12
    )
    np.testing.assert_allclose(
        precipitating.downward_flux, baseline.downward_flux, rtol=1e-12
    )
    np.testing.assert_allclose(precipitating.heating, baseline.heating, atol=1e-11)
    rejected = plan.evaluate(
        [260.0],
        [10.0],
        [1.0],
        [0.2],
        [0.1],
        300.0,
        400.0,
        rain_mass=[9.0],
        snow_mass=0.0,
    )
    assert not rejected.successful
    np.testing.assert_array_equal(rejected.heating, 0.0)


def test_invalid_optical_data_and_layer_shapes_are_explicit_errors():
    with pytest.raises(ValueError, match="nonnegative"):
        optics(sw=(-1, 0, 0, 0))
    with pytest.raises(ValueError, match="asymmetry"):
        optics(asymmetry=(0, 0, 1.1, 0))
    with pytest.raises(ValueError, match="reference_id"):
        ColumnOpticalProperties(
            shortwave_absorption=[0] * 4,
            shortwave_scattering=[0] * 4,
            longwave_absorption=[0] * 4,
            reference_id="",
        )
    with pytest.raises(ValueError, match="common positive length"):
        ColumnRadiationPlan(optics()).evaluate(
            [250.0, 270.0], [1.0], [0.0], [0.0], [0.0], 300.0, 400.0
        )
