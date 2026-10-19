"""Per-site helium diffusion-trapping kinetics of UO2 with a Henry surface condition."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import phydrax.applications.nuclear_fuel._helium as helium_impl
from phydrax.applications.nuclear_fuel import HeliumGasModel, HeliumTransportTable
from phydrax.materials import PropertyValidityRange


pytestmark = pytest.mark.strict_jax

BOLTZMANN_EV = 8.617333262145179e-5  # [eV/K]
# Luzzi et al. NED 330 (2018), Eq. 2 / Table 5: damaged-fuel helium diffusivity.
LUZZI_DIFFUSIVITY_PREFACTOR = 3.3e-10  # [m^2/s]
LUZZI_ACTIVATION_ENERGY = 1.64  # [eV]
# Cognini et al. NET 53 (2021), Eq. 11: Henry constant, published per MPa.
COGNINI_HENRY_PREFACTOR = 4.1e24 / 1.0e6  # [atoms/m^3/Pa]
COGNINI_ACTIVATION_ENERGY = 0.65  # [eV]
# Empirical `test_bubble_free_helium_step_and_tangent_match_spherical_resolvent`: matched max |delta|=589824; joint allowance fraction=0.00640233, margin=99.3598%.
_SPHERICAL_RESOLVENT_RELEASE_TANGENT_RTOL = 1.0e-12


def _inventory_roundoff_bound(radial_cell_count: int, *terms: float) -> float:
    """Bound two shell reductions and four scalar ledger operations."""

    # Each length-N dot product needs N products and at most N-1 additions.
    # The same stored inventory is reduced on each side of the step; allowing
    # 4*N+4 operations covers both reductions and subtraction/recombination.
    # Standard gamma_k = k*u/(1-k*u), u=eps/2, multiplies gross magnitudes,
    # not the potentially canceled net flow. No interface error enters this
    # inventory-defined transfer: that belongs to the separate flux certificate.
    unit_roundoff = float(np.finfo(np.float64).eps) / 2.0
    scaled_roundoff = (4 * radial_cell_count + 4) * unit_roundoff
    return scaled_roundoff / (1.0 - scaled_roundoff) * sum(abs(term) for term in terms)


def _assert_flux_certificate(result: helium_impl.HeliumGasStepResult) -> None:
    """Assert independent surface flux agrees within its reported error bound."""

    assert bool(result.linear_solve_successful)
    assert (
        0.0
        <= float(result.interface_residual_norm)
        <= float(result.linear_solve_residual_norm)
    )
    bound = float(result.boundary_transfer_error_bound)
    assert np.isfinite(bound)
    assert (
        abs(float(result.diffusion_to_boundary - result.boundary_flux_transfer)) <= bound
    )


@pytest.mark.parametrize(
    "temperature", [1100.0, 1500.0, 1750.0], ids=["1100K", "1500K", "1750K"]
)
def test_helium_diffusivity_matches_luzzi_damaged_fuel_fit(temperature: float) -> None:
    expected = LUZZI_DIFFUSIVITY_PREFACTOR * np.exp(
        -LUZZI_ACTIVATION_ENERGY / (BOLTZMANN_EV * temperature)
    )
    np.testing.assert_allclose(
        float(HeliumGasModel().diffusivity(temperature)), expected, rtol=1.0e-13
    )


@pytest.mark.parametrize(
    "temperature", [1100.0, 1500.0, 1750.0], ids=["1100K", "1500K", "1750K"]
)
def test_helium_henry_constant_converts_source_mpa_to_pa(temperature: float) -> None:
    expected = COGNINI_HENRY_PREFACTOR * np.exp(
        -COGNINI_ACTIVATION_ENERGY / (BOLTZMANN_EV * temperature)
    )
    np.testing.assert_allclose(
        float(HeliumGasModel().henry_constant(temperature)), expected, rtol=1.0e-13
    )


def test_helium_absorption_and_emission_close_signed_reference_ledger() -> None:
    model = HeliumGasModel()
    state = model.initial_state((), grain_radius=5.0e-6)
    absorbed = model.advance(
        state,
        temperature=1500.0,
        fission_rate_density=0.0,
        gas_production_rate=0.0,
        time_step=1.0e6,
        grain_radius=5.0e-6,
        bubble_radius=1.0e-9,
        bubble_number_density=2.08e20,
        boundary_helium_pressure=1.0e6,
    )
    assert float(absorbed.released_increment) < 0.0
    assert bool(absorbed.admissible)
    # Empty initial inventory and zero production make uptake exactly the
    # negative of the reported retained inventory, not a separately rounded flux.
    assert float(absorbed.released_increment) == -float(absorbed.retained_total)
    _assert_flux_certificate(absorbed)
    emitted = model.advance(
        absorbed.state,
        temperature=1500.0,
        fission_rate_density=0.0,
        gas_production_rate=0.0,
        time_step=1.0e6,
        grain_radius=5.0e-6,
        bubble_radius=1.0e-9,
        bubble_number_density=2.08e20,
        boundary_helium_pressure=0.0,
    )
    assert float(emitted.released_increment) > 0.0
    assert bool(emitted.admissible)
    np.testing.assert_allclose(
        float(emitted.retained_total + emitted.released_increment),
        float(absorbed.retained_total),
        rtol=0.0,
        atol=_inventory_roundoff_bound(
            model.radial_cell_count,
            float(emitted.retained_total),
            float(emitted.released_increment),
            float(absorbed.retained_total),
        ),
    )
    _assert_flux_certificate(emitted)


def test_helium_inventory_closure_does_not_admit_inconsistent_shell_flux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberately corrupt only a positive source-mean weight. Both native
    # solves remain successful and the state remains finite/nonnegative, but
    # the PDE shell inventory no longer telescopes to the independent flux.
    shell_factors = helium_impl._helium_shell_factors

    def inconsistent_factors(eta: jax.Array) -> helium_impl._HeliumShellFactors:
        factors = shell_factors(eta)
        return factors._replace(same_face_deficit=factors.same_face_deficit + 0.01)

    monkeypatch.setattr(helium_impl, "_helium_shell_factors", inconsistent_factors)
    model = HeliumGasModel()
    production = 2.2e16
    time_step = 86400.0
    result = model.advance(
        model.initial_state((), grain_radius=5.0e-6),
        temperature=1500.0,
        fission_rate_density=0.0,
        gas_production_rate=production,
        time_step=time_step,
        grain_radius=5.0e-6,
        bubble_radius=0.0,
        bubble_number_density=0.0,
        boundary_helium_pressure=0.0,
    )
    assert bool(result.linear_solve_successful)
    assert bool(jnp.all(result.state.dissolved_cells >= 0.0))
    assert bool(jnp.all(result.state.trapped_cells >= 0.0))
    np.testing.assert_allclose(
        float(result.retained_total + result.released_increment),
        production * time_step,
        rtol=0.0,
        atol=_inventory_roundoff_bound(
            model.radial_cell_count,
            float(result.retained_total),
            float(result.released_increment),
            production * time_step,
        ),
    )
    assert abs(
        float(result.diffusion_to_boundary - result.boundary_flux_transfer)
    ) > float(result.boundary_transfer_error_bound)
    assert not bool(result.admissible)


@pytest.mark.parametrize("radial_cell_count", [1, 2, 40], ids=["one", "two", "forty"])
def test_helium_nonuniform_inventory_and_grain_growth_preserve_signed_transfers(
    radial_cell_count: int,
) -> None:
    model = HeliumGasModel(radial_cell_count=radial_cell_count)
    state = model.initial_state((), grain_radius=5.0e-6)
    dissolved = jnp.linspace(1.0e22, 2.0e22, radial_cell_count, dtype=jnp.float64)
    trapped = jnp.linspace(0.5e22, 0.6e22, radial_cell_count, dtype=jnp.float64)
    retained_start = float((dissolved + trapped) @ model.cell_volume_weights)
    state = eqx.tree_at(
        lambda current: (
            current.dissolved_cells,
            current.trapped_cells,
            current.produced,
        ),
        state,
        (dissolved, trapped, jnp.asarray(retained_start, dtype=jnp.float64)),
    )
    production = 2.2e16
    time_step = 86400.0
    radius = 6.0e-6
    result = model.advance(
        state,
        temperature=1500.0,
        fission_rate_density=1.0e19,
        gas_production_rate=production,
        time_step=time_step,
        grain_radius=radius,
        bubble_radius=1.0e-9,
        bubble_number_density=2.08e20,
        boundary_helium_pressure=1.0e6,
    )
    assert bool(result.admissible)
    _assert_flux_certificate(result)
    produced = production * time_step
    scale = _inventory_roundoff_bound(
        radial_cell_count,
        retained_start,
        float(result.retained_total),
        float(result.released_increment),
        produced,
    )
    np.testing.assert_allclose(
        float(result.retained_total + result.released_increment),
        retained_start + produced,
        rtol=0.0,
        atol=scale,
    )
    np.testing.assert_allclose(
        float(result.grain_growth_sweeping),
        (1.0 - (5.0e-6 / radius) ** 3) * retained_start,
        rtol=0.0,
        atol=scale,
    )
    np.testing.assert_allclose(
        float(result.diffusion_to_boundary + result.grain_growth_sweeping),
        float(result.released_increment),
        rtol=0.0,
        atol=scale,
    )
    np.testing.assert_allclose(
        float(result.state.produced - result.state.net_released),
        float(result.retained_total),
        rtol=0.0,
        atol=scale,
    )


def test_declared_cold_helium_data_drive_absorption_without_hidden_defaults() -> None:
    # Synthetic data prove data injection, not scientific validity of cold fits.
    data = HeliumTransportTable(
        temperature=[300.0, 500.0],
        diffusivity=[1.0e-15, 1.0e-15],
        henry_constant=[1.0e14, 1.0e14],
        validity=PropertyValidityRange(
            {"temperature": (300.0, 500.0, "K")}, source_id="test-table"
        ),
        source_id="test-table",
    )
    model = HeliumGasModel(transport=data)
    result = model.advance(
        model.initial_state((), grain_radius=5.0e-6),
        temperature=400.0,
        fission_rate_density=0.0,
        gas_production_rate=0.0,
        time_step=1.0e6,
        grain_radius=5.0e-6,
        bubble_radius=0.0,
        bubble_number_density=0.0,
        boundary_helium_pressure=1.0e6,
    )
    assert float(result.released_increment) < 0.0
    assert bool(result.in_range)
    assert bool(result.admissible)
    assert not bool(HeliumGasModel().validity.contains(temperature=400.0))


def test_helium_model_with_supplied_transport_reports_the_table_source() -> None:
    validity = PropertyValidityRange(
        {"temperature": (300.0, 500.0, "K")}, source_id="measured-helium-table"
    )
    data = HeliumTransportTable(
        temperature=[300.0, 500.0],
        diffusivity=[1.0e-15, 1.0e-15],
        henry_constant=[1.0e14, 1.0e14],
        validity=validity,
        source_id="measured-helium-table",
    )
    model = HeliumGasModel(transport=data)
    assert model.source_id == "measured-helium-table"
    assert model.validity.range_id == validity.range_id
    assert HeliumGasModel().source_id != "measured-helium-table"


@pytest.mark.parametrize(
    ("diffusivity", "time_step"),
    [(1.0e-20, 1.0e-10), (1.0e-20, 0.05), (1.0e-12, 100.0)],
    ids=["vanishing-time", "cold-pressure-step", "near-equilibrium-hold"],
)
def test_helium_pressure_step_matches_positive_spherical_helmholtz_oracle(
    diffusivity: float, time_step: float
) -> None:
    # Synthetic cold data exercise the discretization, not a scientific cold fit.
    henry = 1.0e12
    radius = 5.0e-6
    pressure = 103322.18601133744
    data = HeliumTransportTable(
        temperature=[300.0, 500.0],
        diffusivity=[diffusivity, diffusivity],
        henry_constant=[henry, henry],
        validity=PropertyValidityRange(
            {"temperature": (300.0, 500.0, "K")}, source_id="cold-step-oracle"
        ),
        source_id="cold-step-oracle",
    )
    model = HeliumGasModel(radial_cell_count=2, transport=data)
    result = model.advance(
        model.initial_state((), grain_radius=radius),
        temperature=300.0,
        fission_rate_density=0.0,
        gas_production_rate=0.0,
        time_step=time_step,
        grain_radius=radius,
        bubble_radius=0.0,
        bubble_number_density=0.0,
        boundary_helium_pressure=pressure,
    )
    # Full-sphere BE Green function: G=sinh(z*x)/(x*sinh(z)),
    # mean(G)=3*(z*coth(z)-1)/z^2, independently of the radial partition.
    z = radius / np.sqrt(diffusivity * time_step)
    expected = henry * pressure * 3.0 * (z / np.tanh(z) - 1.0) / z**2
    np.testing.assert_allclose(float(result.retained_total), expected, rtol=1.0e-12)
    np.testing.assert_allclose(float(result.released_increment), -expected, rtol=1.0e-12)
    np.testing.assert_allclose(
        float(result.dissolved_concentration[-1]), henry * pressure, rtol=1.0e-14
    )
    assert bool(jnp.all(result.dissolved_concentration >= 0.0))
    assert bool(jnp.all(result.trapped_concentration >= 0.0))
    assert bool(jnp.all(result.state.dissolved_cells >= 0.0))
    assert bool(result.linear_solve_successful)
    assert bool(result.admissible)
    _assert_flux_certificate(result)


def test_zero_diffusivity_helium_pressure_jump_has_zero_uptake() -> None:
    data = HeliumTransportTable(
        temperature=[300.0, 500.0],
        diffusivity=[0.0, 0.0],
        henry_constant=[1.0e12, 1.0e12],
        validity=PropertyValidityRange(
            {"temperature": (300.0, 500.0, "K")}, source_id="immobile-step"
        ),
        source_id="immobile-step",
    )
    model = HeliumGasModel(radial_cell_count=2, transport=data)
    result = model.advance(
        model.initial_state((), grain_radius=5.0e-6),
        temperature=300.0,
        fission_rate_density=0.0,
        gas_production_rate=0.0,
        time_step=0.05,
        grain_radius=5.0e-6,
        bubble_radius=0.0,
        bubble_number_density=0.0,
        boundary_helium_pressure=103322.18601133744,
    )
    assert float(result.retained_total) == 0.0
    assert float(result.released_increment) == 0.0
    np.testing.assert_array_equal(np.asarray(result.state.dissolved_cells), np.zeros(2))
    np.testing.assert_allclose(
        float(result.dissolved_concentration[-1]), 1.0332218601133744e17
    )
    assert bool(result.linear_solve_successful)
    assert bool(result.admissible)
    assert float(result.boundary_flux_transfer) == 0.0
    assert float(result.interface_residual_norm) == 0.0
    _assert_flux_certificate(result)


# One-day isothermal helium step of an empty 5 um grain under 1 MPa of plenum
# helium with a uniform intragranular helium source.
STEP_TEMPERATURE = 1500.0  # [K]
STEP_TIME = 86400.0  # [s]
STEP_GRAIN_RADIUS = 5.0e-6  # [m]
STEP_FISSION_RATE = 1.0e19  # [fissions/m^3/s]
STEP_HELIUM_PRODUCTION = 2.2e16  # [atoms/m^3/s]
STEP_BOUNDARY_PRESSURE = 1.0e6  # [Pa]
# Intragranular bubbles that trap and re-solve helium.
STEP_BUBBLE_RADIUS = 1.0e-9  # [m]
STEP_BUBBLE_DENSITY = 2.08e20  # [1/m^3]


def _helium_step_balance(
    model: HeliumGasModel, bubble_radius: float, bubble_density: float
) -> tuple[jax.Array, jax.Array]:
    """Retained and released helium [atoms/m^3] and their temperature tangents."""
    state = model.initial_state((), grain_radius=STEP_GRAIN_RADIUS)

    def balance(temperature: jax.Array) -> jax.Array:
        result = model.advance(
            state,
            temperature=temperature,
            fission_rate_density=STEP_FISSION_RATE,
            gas_production_rate=STEP_HELIUM_PRODUCTION,
            time_step=STEP_TIME,
            grain_radius=STEP_GRAIN_RADIUS,
            bubble_radius=bubble_radius,
            bubble_number_density=bubble_density,
            boundary_helium_pressure=STEP_BOUNDARY_PRESSURE,
        )
        return jnp.stack([result.retained_total, result.released_increment])

    return jax.jvp(
        balance,
        (jnp.asarray(STEP_TEMPERATURE, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )


def _bubble_free_retained_reference(temperature: mp.mpf) -> mp.mpf:
    """Full-sphere backward-Euler resolvent of a uniform source [atoms/m^3].

    ``c - D dt lap(c) = beta dt`` from zero with ``c(a) = H p`` has the mean
    ``beta dt (1 - g) + H p g``, ``g = 3 (z coth z - 1) / z^2`` and
    ``z = a / sqrt(D dt)``, with D and H from the Luzzi and Cognini fits.
    """
    thermal = mp.mpf(BOLTZMANN_EV) * temperature
    diffusivity = mp.mpf(LUZZI_DIFFUSIVITY_PREFACTOR) * mp.exp(
        -mp.mpf(LUZZI_ACTIVATION_ENERGY) / thermal
    )
    surface = (
        mp.mpf(COGNINI_HENRY_PREFACTOR)
        * mp.exp(-mp.mpf(COGNINI_ACTIVATION_ENERGY) / thermal)
        * STEP_BOUNDARY_PRESSURE
    )
    z = STEP_GRAIN_RADIUS / mp.sqrt(diffusivity * STEP_TIME)
    mean_green = 3 * (z * mp.coth(z) - 1) / z**2
    produced = mp.mpf(STEP_HELIUM_PRODUCTION) * STEP_TIME
    return produced * (1 - mean_green) + surface * mean_green


@pytest.mark.parametrize("radial_cell_count", [2, 40], ids=["two-shells", "forty-shells"])
def test_bubble_free_helium_step_and_tangent_match_spherical_resolvent(
    radial_cell_count: int,
) -> None:
    # Uniform source and initial data are reproduced exactly at any shell count.
    value, tangent = _helium_step_balance(
        HeliumGasModel(radial_cell_count=radial_cell_count), 0.0, 0.0
    )
    with mp.workdps(50):
        temperature = mp.mpf(STEP_TEMPERATURE)
        expected = float(_bubble_free_retained_reference(temperature))
        expected_tangent = float(mp.diff(_bubble_free_retained_reference, temperature))
    produced = STEP_HELIUM_PRODUCTION * STEP_TIME
    np.testing.assert_allclose(
        np.asarray(value), [expected, produced - expected], rtol=1.0e-12, atol=0.0
    )
    retained_tangent, released_tangent = np.asarray(tangent)
    np.testing.assert_allclose(retained_tangent, expected_tangent, rtol=1.0e-12)
    # Inventory-defined release has the complementary temperature tangent.
    # Use the retained-tangent acceptance ceiling from e7b82d23c, this same
    # 50-decimal spherical-resolvent oracle and two-/forty-shell fixture.
    # Release is its complementary temperature tangent, not a second flux sum.
    np.testing.assert_allclose(
        released_tangent,
        -expected_tangent,
        rtol=_SPHERICAL_RESOLVENT_RELEASE_TANGENT_RTOL,
    )


def test_trapped_helium_temperature_tangent_conserves_atoms() -> None:
    value, tangent = _helium_step_balance(
        HeliumGasModel(), STEP_BUBBLE_RADIUS, STEP_BUBBLE_DENSITY
    )
    retained, released = np.asarray(value)
    retained_tangent, released_tangent = np.asarray(tangent)
    assert np.all(np.isfinite(tangent))
    # Production is temperature independent: retained + released = beta dt.
    np.testing.assert_allclose(
        retained + released,
        STEP_HELIUM_PRODUCTION * STEP_TIME,
        rtol=0.0,
        atol=_inventory_roundoff_bound(
            40, retained, released, STEP_HELIUM_PRODUCTION * STEP_TIME
        ),
    )
    np.testing.assert_allclose(
        retained_tangent + released_tangent,
        0.0,
        rtol=0.0,
        atol=_inventory_roundoff_bound(40, retained_tangent, released_tangent),
    )
    assert retained_tangent != 0.0


def test_trapped_helium_step_and_tangent_are_shell_count_independent() -> None:
    # With trapping, the uniform data still see one modified Helmholtz
    # resolvent, so the shell partition must not change the step.
    coarse = _helium_step_balance(
        HeliumGasModel(radial_cell_count=2), STEP_BUBBLE_RADIUS, STEP_BUBBLE_DENSITY
    )
    fine = _helium_step_balance(
        HeliumGasModel(radial_cell_count=40), STEP_BUBBLE_RADIUS, STEP_BUBBLE_DENSITY
    )
    for coarse_part, fine_part in zip(coarse, fine, strict=True):
        np.testing.assert_allclose(
            np.asarray(coarse_part), np.asarray(fine_part), rtol=1.0e-12, atol=0.0
        )
