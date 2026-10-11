"""Grain-face bubbles, grain growth and micro-cracking of UO2.

Independent references: the saturation coverage condition of Pastore et al.
(2013); the temperature-variation trigger of Barani et al. (2017)
micro-cracking.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nuclear_fuel import AinscoughGrainGrowth
from tests._support.nuclear_fuel_gas import (
    advance_gas,
    DAY,
    fission_gas_model,
)


pytestmark = pytest.mark.strict_jax


def test_venting_begins_only_once_coverage_reaches_saturation() -> None:
    model = fission_gas_model(mode_count=40)
    state = model.initial_state((), grain_radius=5.0e-6)
    coverages: list[float] = []
    ventings: list[float] = []
    burnup = 0.0
    for _ in range(80):
        burnup += 0.3
        result = advance_gas(
            model,
            state,
            temperature=1500.0,
            previous_temperature=1500.0,
            burnup=burnup,
            time_step=5.0 * DAY,
        )
        coverages.append(float(result.diagnostics.fractional_coverage))
        ventings.append(float(result.transfers.venting_release))
        state = result.state
    saturation = 0.5
    first = next(index for index, value in enumerate(ventings) if value > 0.0)
    assert first > 0
    assert all(value == 0.0 for value in ventings[:first])
    assert all(value < saturation for value in coverages[:first])
    assert np.all(np.diff(coverages[:first]) > 0.0)
    np.testing.assert_allclose(coverages[first:], saturation, rtol=1.0e-12)
    assert all(value > 0.0 for value in ventings[first:])


def test_burst_release_requires_a_temperature_variation() -> None:
    model = fission_gas_model(mode_count=40)
    state = model.initial_state((), grain_radius=5.0e-6)
    burnup = 0.0
    for _ in range(12):
        burnup += 0.2
        result = advance_gas(
            model,
            state,
            temperature=1600.0,
            previous_temperature=1600.0,
            burnup=burnup,
            time_step=5.0 * DAY,
        )
        assert float(result.transfers.burst_release) == 0.0
        assert float(result.state.grain_face_intactness) == 1.0
        state = result.state
    assert float(state.boundary_gas) > 0.0

    heated = advance_gas(
        model,
        state,
        temperature=2000.0,
        previous_temperature=1600.0,
        burnup=burnup,
        time_step=600.0,
    )
    assert float(heated.transfers.burst_release) > 0.0
    assert float(heated.state.grain_face_intactness) < 1.0
    assert float(heated.diagnostics.saturation_coverage) < 0.5

    held = advance_gas(
        model,
        heated.state,
        temperature=2000.0,
        previous_temperature=2000.0,
        burnup=burnup,
        time_step=600.0,
    )
    assert float(held.transfers.burst_release) == 0.0

    cooled = advance_gas(
        model,
        held.state,
        temperature=1200.0,
        previous_temperature=2000.0,
        burnup=burnup,
        time_step=600.0,
    )
    assert float(cooled.transfers.burst_release) > 0.0
    assert float(cooled.state.grain_face_intactness) < float(
        held.state.grain_face_intactness
    )


def test_prepared_grain_radius_cannot_shrink_an_empty_accepted_grain() -> None:
    model = fission_gas_model(mode_count=3)
    state = model.initial_state((2,), grain_radius=5.0e-6)
    result = model.advance(
        state,
        temperature=600.0,
        grain_radius=2.5e-6,
        fission_rate_density=0.0,
        gas_production_rate=0.0,
        burnup_mwd_per_kghm=0.0,
        hydrostatic_stress=-2.0e7,
        time_step=1.0,
        previous_temperature=600.0,
    )
    # Zero inventories make this a pure growth-monotonicity violation, not a
    # rejection that could accidentally be supplied by another gas guard.
    np.testing.assert_array_equal(np.asarray(result.retained_total), np.zeros(2))
    np.testing.assert_array_equal(
        np.asarray(result.admissible), np.zeros(2, dtype=np.bool_)
    )


def test_irradiation_drag_lowers_growth_limit_without_grain_shrinkage() -> None:
    law = AinscoughGrainGrowth()
    fresh_limit = law.limiting_radius(1773.0, burnup_mwd_per_kghm=0.0)
    irradiated_limit = law.limiting_radius(1773.0, burnup_mwd_per_kghm=20.0)
    # Primary Barani thesis (2020), Appendix Eq. A.7, bu in MWd/kgU.
    np.testing.assert_allclose(
        float(fresh_limit / irradiated_limit),
        1.0 + 0.002 * 20.0,
        rtol=1.0e-13,
    )
    initial = float((fresh_limit + irradiated_limit) / 2.0)
    fresh = law.advance_radius(initial, 1773.0, 30.0 * DAY, burnup_mwd_per_kghm=0.0)
    irradiated = law.advance_radius(initial, 1773.0, 30.0 * DAY, burnup_mwd_per_kghm=20.0)
    assert float(fresh) > initial
    np.testing.assert_allclose(float(irradiated), initial, rtol=1.0e-13)


def test_vented_cavities_preserve_volume_when_their_gas_enters_free_volume() -> None:
    model = fission_gas_model(mode_count=40)
    radius = 5.0e-6
    density = 2.0e13
    gas = 1.0e26
    vacancies = 1.0e8
    bubbles_per_volume = 1.5 / radius * density
    state = eqx.tree_at(
        lambda s: (s.boundary_gas, s.bubble_gas_atoms, s.bubble_vacancies, s.produced),
        model.initial_state((), grain_radius=radius),
        (
            jnp.asarray(gas),
            jnp.asarray(gas / bubbles_per_volume),
            jnp.asarray(vacancies),
            jnp.asarray(gas),
        ),
    )
    # Topology-only transition: no time evolution or production. Pastore
    # Table I volumes and shared-grain-face inventory determine initial volume.
    initial_volume = 8.5e-29 * gas + 4.09e-29 * vacancies * bubbles_per_volume
    result = advance_gas(
        model,
        state,
        temperature=1500.0,
        previous_temperature=1500.0,
        burnup=0.0,
        time_step=0.0,
        fission_rate=0.0,
        production=0.0,
    )
    assert float(result.transfers.venting_release) > 0.0
    assert float(result.connected_boundary_volume_increment) > 0.0
    np.testing.assert_allclose(
        float(result.intergranular_swelling), initial_volume, rtol=1.0e-12
    )
    closed_volume = 8.5e-29 * float(result.state.boundary_gas) + 4.09e-29 * float(
        result.state.bubble_vacancies
    ) * (1.5 / radius) * float(result.state.bubble_number_density)
    np.testing.assert_allclose(
        closed_volume + float(result.state.connected_boundary_volume),
        initial_volume,
        rtol=1.0e-12,
    )
    assert bool(result.admissible)
