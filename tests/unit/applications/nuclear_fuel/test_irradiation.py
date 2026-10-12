from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax.applications.nuclear_fuel._irradiation import IrradiationHistory


def _history() -> IrradiationHistory:
    return IrradiationHistory(
        times=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        zone_edges=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        linear_power=np.asarray([[0.0, 0.0], [10.0, 20.0], [0.0, 0.0]], dtype=np.float64),
        fast_flux=np.asarray(
            [[0.0, 0.0], [2.0e17, 4.0e17], [0.0, 0.0]], dtype=np.float64
        ),
        coolant_inlet_temperature=np.asarray([300.0, 570.0, 300.0], dtype=np.float64),
        coolant_pressure=np.asarray([1.0e5, 1.55e7, 1.0e5], dtype=np.float64),
        coolant_mass_flux=np.asarray([100.0, 3000.0, 100.0], dtype=np.float64),
    )


def test_linear_irradiation_primitive_is_exact_between_breakpoints() -> None:
    history = _history()
    energy = jax.jit(lambda time: history.energy_per_length(time))(1.0)
    np.testing.assert_allclose(energy, [2.5, 5.0], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(history.energy_per_length(4.0), [20.0, 40.0])
    np.testing.assert_allclose(history.fast_fluence(4.0), [4.0e17, 8.0e17])


def test_irradiation_accounting_is_independent_of_step_partition() -> None:
    history = _history()
    split_energy = (
        1.0 * history.step_average_linear_power(0.0, 1.0)
        + 2.0 * history.step_average_linear_power(1.0, 3.0)
        + 1.0 * history.step_average_linear_power(3.0, 4.0)
    )
    split_fluence = (
        1.0 * history.step_average_fast_flux(0.0, 1.0)
        + 2.0 * history.step_average_fast_flux(1.0, 3.0)
        + 1.0 * history.step_average_fast_flux(3.0, 4.0)
    )
    np.testing.assert_allclose(split_energy, [20.0, 40.0], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(split_fluence, [4e17, 8e17], rtol=1e-15)


def test_history_coolant_and_flux_share_physical_time() -> None:
    conditions = _history().conditions(1.0)
    np.testing.assert_allclose(conditions.linear_power, [5.0, 10.0])
    np.testing.assert_allclose(conditions.fast_flux, [1e17, 2e17])
    np.testing.assert_allclose(conditions.coolant_inlet_temperature, 435.0)
    np.testing.assert_allclose(conditions.coolant_pressure, 7.8e6)
    np.testing.assert_allclose(conditions.coolant_mass_flux, 1550.0)


@pytest.mark.parametrize(
    "time", [-1.0, 5.0, np.nan], ids=["before", "after", "nonfinite"]
)
def test_history_queries_outside_declared_support_are_refused(time: float) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="within the declared time"):
        _history().energy_per_length(time)


@pytest.mark.parametrize("end", [1.0, 0.0], ids=["zero-duration", "backward"])
def test_nonpositive_source_interval_is_refused(end: float) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="strictly increasing"):
        _history().step_average_linear_power(1.0, end)
