#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications._energy_series import (
    counter_to_intervals,
    EnergySeries,
    integrate_energy_series,
    rebin_energy_series,
)
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.units import derived_unit, JOULE, SECOND, UnitDefinition


WATT = derived_unit("W", ((JOULE, 1), (SECOND, -1)))
HOUR = UnitDefinition("h", SECOND.dimension, SECOND.reference_system_id, 3600)


def test_irregular_rebin_preserves_energy_and_parameter_derivative():
    support = SeriesSupport(jnp.array([0.0, 2.0, 5.0]), coordinate_id="time")
    target = SeriesSupport(jnp.array([0.0, 1.0, 3.0, 5.0]), coordinate_id="time")

    def rebin(values):
        series = EnergySeries(
            SampledSeries(support, values, alignment="edge", series_id="load"),
            quantity="active_power",
            unit=WATT,
            meaning="interval_average",
            time_unit=HOUR,
        )
        return rebin_energy_series(series, target)

    result = jax.jit(rebin)(jnp.array([3.0, 5.0]))
    assert jnp.allclose(result.samples.values, jnp.array([3.0, 4.0, 5.0]))
    total, unit = integrate_energy_series(result)
    assert unit.dimension == JOULE.dimension
    assert jnp.allclose(total, 21.0 * 3600)
    derivative = jax.grad(lambda values: integrate_energy_series(rebin(values))[0])(
        jnp.array([3.0, 5.0])
    )
    assert jnp.allclose(derivative, jnp.array([2.0, 3.0]) * 3600)


def test_incomplete_channel_is_not_imputed_or_integrated():
    support = SeriesSupport(jnp.array([0.0, 1.0, 2.0]), coordinate_id="time")
    source = EnergySeries(
        SampledSeries(
            support,
            jnp.array([[2.0, 7.0], [4.0, 9.0]]),
            alignment="edge",
            value_valid=jnp.array([[True, True], [True, False]]),
            series_id="meters",
        ),
        quantity="active_power",
        unit=WATT,
        meaning="interval_average",
    )
    result = rebin_energy_series(
        source, SeriesSupport(jnp.array([0.0, 2.0]), coordinate_id="time")
    )
    assert jnp.array_equal(result.samples.value_valid, jnp.array([[True, False]]))
    assert jnp.allclose(result.samples.values[:, 0], jnp.array([3.0]))
    with pytest.raises(eqx.EquinoxRuntimeError):
        integrate_energy_series(result)


def test_interval_integrals_subdivide_without_changing_total():
    source = EnergySeries(
        SampledSeries(
            SeriesSupport(jnp.array([0.0, 2.0, 5.0]), coordinate_id="time"),
            jnp.array([6.0, 15.0]),
            alignment="edge",
            series_id="energy",
        ),
        quantity="electrical_energy",
        unit=JOULE,
        meaning="interval_integral",
    )
    result = rebin_energy_series(
        source, SeriesSupport(jnp.array([0.0, 1.0, 3.0, 5.0]), coordinate_id="time")
    )
    assert jnp.allclose(result.samples.values, jnp.array([3.0, 8.0, 10.0]))
    assert jnp.allclose(integrate_energy_series(result)[0], 21.0)
    outside = rebin_energy_series(
        source, SeriesSupport(jnp.array([-1.0, 1.0, 6.0]), coordinate_id="time")
    )
    assert jnp.array_equal(outside.samples.value_valid, jnp.array([False, False]))


def test_counter_decrease_requires_explicit_consumption_evidence():
    source = EnergySeries(
        SampledSeries(
            SeriesSupport(jnp.array([0.0, 1.0, 2.0]), coordinate_id="time"),
            jnp.array([95.0, 3.0, 7.0]),
            series_id="counter",
        ),
        quantity="electrical_energy",
        unit=JOULE,
        meaning="cumulative",
    )
    with pytest.raises(eqx.EquinoxRuntimeError):
        counter_to_intervals(source)
    wrapped = counter_to_intervals(source, rollover=100.0)
    reset = counter_to_intervals(source, reset_increments=jnp.array([12.0, 0.0]))
    assert jnp.array_equal(wrapped.samples.values, jnp.array([8.0, 4.0]))
    assert jnp.array_equal(reset.samples.values, jnp.array([12.0, 4.0]))
