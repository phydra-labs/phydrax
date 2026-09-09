# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.geophysics import (
    GeophysicalFieldBinding,
    GeophysicalQuantity,
    GeophysicalTimeSpec,
    HybridPressureCoordinate,
    TemporalSupport,
)
from phydrax.dynamics import StateLayout
from phydrax.units import KELVIN, KILOPASCAL, ONE, PASCAL


def test_exchange_compatibility_is_physical_not_local_storage_or_unit_scale():
    source = GeophysicalQuantity(
        "pressure_source",
        "pressure",
        KILOPASCAL,
        axes=("station",),
        support_association="points",
    )
    target = GeophysicalQuantity(
        "pressure_target", "pressure", PASCAL, axes=("cell",), support_association="cells"
    )
    assert source.compatibility_id == target.compatibility_id
    np.testing.assert_allclose(
        target.from_si(source.to_si(jnp.array([100.0]))), [100000.0]
    )
    anomaly = GeophysicalQuantity("p_anomaly", "pressure_anomaly", PASCAL)
    assert source.compatibility_id != anomaly.compatibility_id
    vapor = GeophysicalQuantity("q", "specific_humidity", ONE)
    mixing = GeophysicalQuantity("r", "vapor_mixing_ratio", ONE)
    assert vapor.compatibility_id != mixing.compatibility_id
    with pytest.raises(ValueError, match="dimensions"):
        GeophysicalQuantity("temperature", "temperature", PASCAL)


def test_archive_binding_requires_actual_original_storage_and_untampered_semantics():
    layout = StateLayout((2,), component_names=("temperature", "pressure"))
    quantity = GeophysicalQuantity("temperature", "temperature", KELVIN)
    binding = GeophysicalFieldBinding(
        quantity, state_layout=layout, components=("temperature",)
    )
    restored = GeophysicalFieldBinding.from_dict(binding.to_dict(), state_layout=layout)
    assert restored.binding_id == binding.binding_id
    wrong = StateLayout((2,), component_names=("pressure", "temperature"))
    with pytest.raises(ValueError, match="storage"):
        GeophysicalFieldBinding.from_dict(binding.to_dict(), state_layout=wrong)
    forged = copy.deepcopy(binding.to_dict())
    forged["quantity"]["reference_configuration"] = "anomaly"
    with pytest.raises(ValueError, match="fingerprint"):
        GeophysicalFieldBinding.from_dict(forged, state_layout=layout)
    with pytest.raises(ValueError, match="component"):
        GeophysicalFieldBinding(quantity, state_layout=layout, components=("missing",))


@pytest.mark.parametrize(
    ("calendar", "dates", "elapsed"),
    [
        ("standard", ("1582-10-04", "1582-10-15"), 1.0),
        ("proleptic_gregorian", ("1582-10-04", "1582-10-15"), 11.0),
        ("noleap", ("2000-02-28", "2000-03-01"), 1.0),
        ("all_leap", ("1900-02-28", "1900-03-01"), 2.0),
        ("360_day", ("2001-02-30", "2001-03-01"), 1.0),
    ],
)
def test_calendar_elapsed_time_and_inverse(calendar, dates, elapsed):
    spec = GeophysicalTimeSpec(calendar, dates[0], "d")
    np.testing.assert_array_equal(spec.encode(dates), [0.0, elapsed])
    assert spec.decode([0.0, elapsed]) == tuple(date + "T00:00:00" for date in dates)


def test_calendar_invalid_dates_time_scale_and_epoch_identity():
    standard = GeophysicalTimeSpec("gregorian", "1582-10-04", "d")
    with pytest.raises(ValueError, match="transition"):
        standard.encode(["1582-10-10"])
    with pytest.raises(ValueError, match="Day"):
        GeophysicalTimeSpec("noleap").encode(["2000-02-29"])
    with pytest.raises(ValueError, match="leap seconds"):
        GeophysicalTimeSpec().encode(["2016-12-31T23:59:60"])
    with pytest.raises(ValueError, match="years/months"):
        GeophysicalTimeSpec(unit="year")
    spec = GeophysicalTimeSpec(epoch="2000-01-01T00:00:00.25")
    dates = ("1999-12-31T23:59:59.75", "2000-01-01T00:00:00.250001")
    np.testing.assert_allclose(spec.encode(dates), [-0.5, 0.000001], atol=1.0e-15)
    assert spec.decode(spec.encode(dates)) == dates
    forged = spec.to_dict()
    forged["calendar"] = "360_day"
    with pytest.raises(ValueError, match="fingerprint"):
        GeophysicalTimeSpec.from_dict(forged)


def test_clock_rejects_lossy_fractional_dates_instead_of_leaving_supported_range():
    final_date = "9999-12-31T23:59:59.999999"
    with pytest.raises(ValueError, match="representable"):
        GeophysicalTimeSpec().encode([final_date])
    close_epoch = GeophysicalTimeSpec(epoch="9999-12-31")
    assert close_epoch.decode(close_epoch.encode([final_date])) == (final_date,)


def test_accumulation_requires_bounds_but_allows_explicit_rolling_windows():
    with pytest.raises(ValueError, match="bounds"):
        TemporalSupport("accumulation")
    rolling = TemporalSupport("mean", ((0.0, 2.0), (1.0, 3.0)), "end")
    assert rolling.bounds == ((0.0, 2.0), (1.0, 3.0))
    assert TemporalSupport.from_dict(rolling.to_dict()) == rolling
    with pytest.raises(ValueError, match="increasing"):
        TemporalSupport("accumulation", ((3.0, 1.0),))


def test_hybrid_layer_measure_tracks_surface_pressure_and_derivative():
    coordinate = HybridPressureCoordinate([0.1, 0.05, 0.0], [0.0, 0.5, 1.0])
    ps = jnp.array([90000.0, 100000.0])
    pressure = coordinate.interfaces(ps)
    np.testing.assert_allclose(
        pressure, [[10000.0, 50000.0, 90000.0], [10000.0, 55000.0, 100000.0]]
    )
    mass = coordinate.layer_mass(ps)
    np.testing.assert_allclose(jnp.sum(mass, axis=-1) * 9.80665, ps - 10000.0)
    derivative = jax.jvp(coordinate.layer_mass, (ps,), (jnp.ones_like(ps),))[1]
    np.testing.assert_allclose(derivative, jnp.full((2, 2), 0.5 / 9.80665))
    assert np.all(coordinate.valid(ps))
    assert not bool(coordinate.valid(1000.0))
    with pytest.raises(ValueError, match="increase"):
        HybridPressureCoordinate([0.0, 0.0], [1.0, 0.0])


def test_hybrid_archive_retains_executed_precision_not_unrounded_input():
    coordinate = HybridPressureCoordinate(
        [0.1, 0.05, 0.0], [0.0, 0.5, 1.0], dtype="float32"
    )
    restored = HybridPressureCoordinate.from_dict(coordinate.to_dict())
    assert restored.coordinate_id == coordinate.coordinate_id
    assert restored.a.dtype == jnp.float32
    np.testing.assert_array_equal(
        restored.interfaces(jnp.float32(90000.0)),
        coordinate.interfaces(jnp.float32(90000.0)),
    )
    high_precision = HybridPressureCoordinate(
        [0.1, 0.05, 0.0], [0.0, 0.5, 1.0], dtype="float64"
    )
    assert high_precision.coordinate_id != coordinate.coordinate_id
