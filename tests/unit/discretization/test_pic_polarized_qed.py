#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Polarization- and spin-resolved strong-field QED and spin precession.

Code units: c = ε₀ = 1, time in 1/ω₀ and fields in m ω₀ c/e for a 1 μm laser,
so the Schwinger field is ``a_S = mc²/(ħω₀) = 4.1·10⁵`` and α = 1/137.036
(q = m = ħ a_S with ħ = 4πα/a_S²).

Independent references:

- Seipt & King, PRA 102, 052805 (2020): the polarized LCFA channel rates in
  their Airy-function form (Eq. 38 for nonlinear Compton, Eq. 59 for
  Breit–Wheeler), evaluated with SciPy's Airy functions, and their small-``χ``
  channel asymptotics (Eq. 44), whose spin-flip part is the Sokolov–Ternov
  result: total flip rate ``(5√3/8) α χ³ mc²/(ħγ)`` and equilibrium
  polarization ``8/(5√3)`` antiparallel (electrons) to the magnetic field;
- Bargmann–Michel–Telegdi: in a pure magnetic field across the momentum the
  spin turns relative to the momentum at ``aγω_c``;
- classical synchrotron radiation (Jackson §14.6): the linear polarization
  along the acceleration is ``∫G/x ÷ ∫F/x = 3/5`` of the photon number and
  ``∫G ÷ ∫F = 3/4`` of the power, the ``χ → 0`` limit of the LCFA;
- Baier & Katkov: pair creation by photons polarized along the transverse
  field is half that of perpendicular photons as ``χ_γ → 0``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import airy

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax._strict import StrictModule
from phydrax.discretization.pic import (
    ELECTRON_MAGNETIC_MOMENT_ANOMALY,
    ExternalFieldSample,
    NonlinearBreitWheelerPlan,
    NonlinearComptonPlan,
    PIC_CODE_RELATIVITY,
    PICChargeModelPlan,
    PICSpeciesPlan,
    QEDCascadeProcess,
    QEDPhotonSpeciesPlan,
    QEDPolarizationModel,
    QEDTable,
    RelativisticPusher,
    RelativisticPushPlan,
)
from phydrax.units import CHARGE, UnitDefinition


D = phx.discretization
_ALPHA = Fraction(1000, 137036)
_SCHWINGER = 410000
_HBAR = 4 * Fraction(math.pi) * _ALPHA / _SCHWINGER**2
_MASS = float(_HBAR) * _SCHWINGER
_SCALE = ElectromagneticScaleContract.code_units(
    PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=_HBAR,
    boltzmann_constant=1,
    elementary_charge=Fraction(_MASS),
    electron_mass=Fraction(_MASS),
    vacuum_permittivity=1,
    constant_set_id="polarized-qed-test",
)
# α_q/τ_C in inverse code time: the dimensionless rates are per unit α mc²/(ħγ).
_RATE_SCALE = float(_ALPHA) * _SCHWINGER
_SIGNS = (1.0, -1.0)


@pytest.fixture(scope="module")
def tables() -> dict[tuple[str, str], QEDTable]:
    return {
        (process, polarization): QEDTable(
            process, maximum_chi=100.0, polarization=polarization
        )
        for process in ("nonlinear-compton", "nonlinear-breit-wheeler")
        for polarization in ("averaged", "positive", "negative")
    }


def _compton(
    tables: dict[tuple[str, str], QEDTable],
    polarization: QEDPolarizationModel,
    **options: Any,
) -> NonlinearComptonPlan:
    spin = (
        (
            tables["nonlinear-compton", "positive"],
            tables["nonlinear-compton", "negative"],
        )
        if polarization == "spin-and-photon-polarized"
        else None
    )
    return NonlinearComptonPlan(
        "lcfa",
        _SCALE,
        -_MASS,
        _MASS,
        tables["nonlinear-compton", "averaged"],
        maximum_chi=100.0,
        minimum_gamma=options.pop("minimum_gamma", 1.0),
        maximum_event_probability=options.pop("maximum_event_probability", 0.2),
        polarization=polarization,
        spin_tables=spin,
        **options,
    )


def _breit_wheeler(
    tables: dict[tuple[str, str], QEDTable], polarization: QEDPolarizationModel
) -> NonlinearBreitWheelerPlan:
    polarized = (
        None
        if polarization == "unpolarized"
        else (
            tables["nonlinear-breit-wheeler", "positive"],
            tables["nonlinear-breit-wheeler", "negative"],
        )
    )
    return NonlinearBreitWheelerPlan(
        _SCALE,
        _MASS,
        _MASS,
        tables["nonlinear-breit-wheeler", "averaged"],
        maximum_chi=100.0,
        polarization=polarization,
        polarized_tables=polarized,
    )


# -- Seipt–King references (Airy form, SciPy) -------------------------------------


def _airy_terms(z: float) -> tuple[float, float, float]:
    """``∫_z^∞ Ai``, ``Ai(z)/√z`` and ``2Ai'(z)/z``."""
    if z > 200.0:
        # Every term is below e^{−1800}.
        return 0.0, 0.0, 0.0
    ai, aip, _, _ = airy(np.float64(z))
    integral = quad(
        lambda t: airy(t)[0],
        z,
        np.inf,
        epsabs=1.0e-300,
        epsrel=1.0e-11,
        limit=500,
    )[0]
    return integral, ai / math.sqrt(z), 2.0 * aip / z


def _sk_compton(chi: float, s: float, spin: float, final: float, tau: float) -> float:
    """Seipt–King Eq. (38) per unit ``α/b``; spins projected on ``ê = v̂ × F̂``."""
    g = 1.0 + s**2 / (2.0 * (1.0 - s))
    integral, ai, aip = _airy_terms((s / (chi * (1.0 - s))) ** (2.0 / 3.0))
    a = 1.0 + spin * final + tau * spin * final * (1.0 - g)
    b = s * spin + s / (1.0 - s) * final + tau * (s / (1.0 - s) * spin + s * final)
    c = g + spin * final + tau * (1.0 + g * spin * final) / 2.0
    return -0.25 * (a * integral + b * ai + c * aip)


def _sk_pairs(
    chi: float, electron: float, tau: float, spin_positron: float, spin_electron: float
) -> float:
    """Seipt–King Eq. (59) per unit ``α/b`` with ``s`` the positron fraction.

    Seipt & King quantize both leptons along the lab magnetic field of a
    counter-propagating photon; with ``ê_± = v̂ × F̂_±`` of each lepton this is
    ``σ_p = −P₊`` (positron) and ``σ_q = P₋`` (electron).
    """
    s = 1.0 - electron
    sigma_p, sigma_q = -spin_positron, spin_electron
    g = 1.0 - 1.0 / (2.0 * s * (1.0 - s))
    integral, ai, aip = _airy_terms((chi * s * (1.0 - s)) ** (-2.0 / 3.0))
    a = 1.0 + sigma_p * sigma_q + tau * sigma_p * sigma_q * (1.0 - g)
    b = sigma_p / s - sigma_q / (1.0 - s) + tau * (sigma_q / s - sigma_p / (1.0 - s))
    c = (g + sigma_p * sigma_q) + tau * (1.0 + g * sigma_p * sigma_q) / 2.0
    return 0.25 * (a * integral + b * ai + c * aip)


_REFERENCE_NODES, _REFERENCE_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _panel_integral(function: Any, lower: float, upper: float, /) -> float:
    half = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    points = midpoint + half * _REFERENCE_NODES
    values = np.asarray([function(float(point)) for point in points])
    return float(half * np.dot(_REFERENCE_WEIGHTS, values))


def _integrate_range(function: Any, lower: float, upper: float) -> float:
    breaks = np.linspace(lower, upper, 25)
    return sum(
        _panel_integral(function, float(a), float(b))
        for a, b in zip(breaks[:-1], breaks[1:], strict=True)
    )


def _integrate(function: Any, chi: float) -> float:
    """``∫₀¹ f(s) ds`` on log-spaced panels around the spectral scale ``χ``.

    Below the first break the photon spectrum is ``∝ s^{−2/3}``, whose mass is
    ``3 s₀ f(s₀)`` (pair spectra vanish there exponentially).
    """
    scale = min(0.5, chi)
    breaks = np.unique(
        np.concatenate(
            (np.geomspace(1e-9 * scale, scale, 12), np.linspace(scale, 1.0 - 1e-12, 12))
        )
    )
    head = float(breaks[0])
    return 3.0 * head * function(head) + sum(
        _panel_integral(function, float(a), float(b))
        for a, b in zip(breaks[:-1], breaks[1:], strict=True)
    )


# -- tables ------------------------------------------------------------------------


@pytest.mark.parametrize("process", ["nonlinear-compton", "nonlinear-breit-wheeler"])
def test_polarized_tables_average_to_the_unpolarized_tables(
    tables: dict[tuple[str, str], QEDTable], process: str
) -> None:
    averaged = tables[process, "averaged"]
    positive = tables[process, "positive"]
    negative = tables[process, "negative"]
    chi = jnp.geomspace(0.05, 80.0, 23)
    np.testing.assert_allclose(
        0.5 * (positive.rate_function(chi) + negative.rate_function(chi)),
        averaged.rate_function(chi),
        rtol=1e-8,
    )
    fraction = jnp.linspace(0.01, 0.99, 41)
    grid_chi, grid_fraction = jnp.meshgrid(chi, fraction)
    # The spin-averaged (polarization-averaged) spectrum is the unpolarized one,
    # and so is the rate-weighted mixture of the two cumulative spectra.
    np.testing.assert_allclose(
        0.5
        * (
            positive.spectrum(grid_chi, grid_fraction)
            + negative.spectrum(grid_chi, grid_fraction)
        ),
        averaged.spectrum(grid_chi, grid_fraction),
        rtol=1e-12,
    )
    mixture = (
        positive.rate_function(grid_chi) * positive.cdf(grid_chi, grid_fraction)
        + negative.rate_function(grid_chi) * negative.cdf(grid_chi, grid_fraction)
    ) / (positive.rate_function(grid_chi) + negative.rate_function(grid_chi))
    np.testing.assert_allclose(mixture, averaged.cdf(grid_chi, grid_fraction), atol=2e-3)
    for table in (positive, negative):
        assert table.minimum_cdf_increment >= 0.0
        assert table.rate_interpolation_error < 1e-6


def test_polarized_rates_match_seipt_king_quadrature(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    for chi in (0.5, 5.0):
        for index, spin in enumerate(_SIGNS):
            expected = _integrate(
                lambda s, spin=spin: sum(
                    _sk_compton(chi, s, spin, final, tau)
                    for final in _SIGNS
                    for tau in _SIGNS
                ),
                chi,
            )
            table = tables["nonlinear-compton", ("positive", "negative")[index]]
            np.testing.assert_allclose(table.rate_function(chi), expected, rtol=1e-6)
    for chi in (0.5, 5.0):
        for index, tau in enumerate(_SIGNS):
            expected = _integrate(
                lambda e, tau=tau: sum(
                    _sk_pairs(chi, e, tau, up, down) for up in _SIGNS for down in _SIGNS
                ),
                chi,
            )
            table = tables["nonlinear-breit-wheeler", ("positive", "negative")[index]]
            np.testing.assert_allclose(table.rate_function(chi), expected, rtol=1e-6)
    # Photons polarized along the field create pairs at half the perpendicular
    # rate as χ_γ → 0 (Baier–Katkov).
    parallel = tables["nonlinear-breit-wheeler", "positive"].rate_function(0.02)
    perpendicular = tables["nonlinear-breit-wheeler", "negative"].rate_function(0.02)
    np.testing.assert_allclose(parallel / perpendicular, 0.5, rtol=0.01)


def test_channel_spectra_match_seipt_king(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    compton = _compton(tables, "spin-and-photon-polarized")
    pairs = _breit_wheeler(tables, "spin-and-photon-polarized")
    for chi, fraction in ((0.3, 0.02), (2.0, 0.4), (20.0, 0.9)):
        for spin in (1.0, -0.4):
            actual = np.asarray(compton.channel_spectrum(chi, fraction, spin))
            expected = [
                [_sk_compton(chi, fraction, spin, final, tau) for tau in _SIGNS]
                for final in _SIGNS
            ]
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-15)
        for tau in (1.0, -0.3):
            actual = np.asarray(pairs.channel_spectrum(chi, fraction, tau))
            expected = [
                [_sk_pairs(chi, fraction, tau, up, down) for down in _SIGNS]
                for up in _SIGNS
            ]
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-15)


# -- spin precession ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["boris", "vay", "higuera-cary"])
def test_tbmt_precession_turns_the_spin_at_the_anomaly_frequency(
    method: RelativisticPusher,
) -> None:
    pusher = RelativisticPushPlan(PIC_CODE_RELATIVITY, method=method)
    gamma, field, dt, steps = 50.0, 1.0, 0.05, 4000
    u = jnp.asarray([[math.sqrt(gamma**2 - 1.0), 0.0, 0.0]])
    electric = jnp.zeros((1, 3))
    magnetic = jnp.asarray([[0.0, 0.0, field]])
    charge = jnp.asarray([-1.0])
    active = jnp.asarray([True])

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        proper, spin, locked = carry
        pushed = pusher.push(proper, electric, magnetic, charge, active, dt)
        new = pushed.proper_velocity
        return (
            new,
            pusher.precess(
                spin,
                proper,
                new,
                electric,
                magnetic,
                charge,
                ELECTRON_MAGNETIC_MOMENT_ANOMALY,
                active,
                dt,
            ),
            pusher.precess(
                locked, proper, new, electric, magnetic, charge, 0.0, active, dt
            ),
        ), None

    start = u / jnp.linalg.norm(u)
    (final, spin, locked), _ = jax.lax.scan(step, (u, start, start), None, length=steps)
    direction = final / jnp.linalg.norm(final)
    # Without an anomaly the spin follows the momentum (Thomas locking).
    np.testing.assert_allclose(locked, direction, atol=1e-11)
    np.testing.assert_allclose(jnp.linalg.norm(spin), 1.0, rtol=1e-12)
    angle = math.atan2(
        float(jnp.cross(direction, spin)[0, 2]), float(jnp.sum(direction * spin))
    )
    cyclotron = field / gamma
    expected = ELECTRON_MAGNETIC_MOMENT_ANOMALY * gamma * cyclotron * steps * dt
    # The electron (q < 0) turns counterclockwise about B; so does its spin,
    # ahead of the momentum. The discretization error is O((ω_c Δt)²).
    assert expected > 0.1
    np.testing.assert_allclose(angle, expected, rtol=1e-6)


# -- Sokolov–Ternov ---------------------------------------------------------------------


def test_spin_flip_channels_reproduce_sokolov_ternov(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    plan = _compton(tables, "spin-and-photon-polarized")
    chi = 1.0e-4
    nodes, weights = np.polynomial.legendre.leggauss(32)
    edges = np.linspace(math.log(1e-10), math.log(80.0), 801)
    lower, upper = edges[:-1, None], edges[1:, None]
    x = (0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes).ravel()
    w = (0.5 * (upper - lower) * weights).ravel()
    delta = np.exp(x)
    a = 3.0 * chi * delta
    fraction = a / (2.0 + a)
    jacobian = w * 6.0 * chi * delta / (2.0 + a) ** 2
    flips = {}
    for spin in _SIGNS:
        channels = plan.channel_spectrum(
            jnp.full_like(fraction, chi), fraction, jnp.full_like(fraction, spin)
        )
        rates = np.einsum("n,nij->ij", jacobian, np.asarray(channels))
        # Seipt–King Eq. (44): flip channels ∝ χ³[15/16 − 5τ/6 + (√3/2)σ(1 − τ)].
        expected = [
            (15.0 / 16.0 - 5.0 * tau / 6.0 + 0.5 * math.sqrt(3.0) * spin * (1.0 - tau))
            / (2.0 * math.sqrt(3.0))
            for tau in _SIGNS
        ]
        final = 1 if spin > 0.0 else 0
        np.testing.assert_allclose(rates[final] / chi**3, expected, rtol=2e-3)
        flips[spin] = float(np.sum(rates[final]))
    total = flips[1.0] + flips[-1.0]
    np.testing.assert_allclose(total / chi**3, 5.0 * math.sqrt(3.0) / 8.0, rtol=2e-3)
    # Equilibrium polarization along ê = v̂ × F̂: −8/(5√3).
    np.testing.assert_allclose(
        (flips[-1.0] - flips[1.0]) / total, -8.0 / (5.0 * math.sqrt(3.0)), atol=1e-4
    )


def test_radiative_polarization_relaxes_to_the_lcfa_equilibrium(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    """Electrons held at χ = 1 across B = b ẑ (energy restored every step)."""
    plan = _compton(tables, "spin-and-photon-polarized", maximum_subcycles=16)
    count, gamma, dt, steps = 4096, 1000.0, 0.4, 80
    field = _SCHWINGER / gamma
    u = jnp.zeros((count, 3)).at[:, 0].set(math.sqrt(gamma**2 - 1.0))
    electric = jnp.zeros((count, 3))
    magnetic = jnp.zeros((count, 3)).at[:, 2].set(field)
    active = jnp.ones((count,), dtype=bool)
    ids = jnp.arange(count, dtype=jnp.uint32)
    zero_ids = jnp.zeros((count,), dtype=jnp.uint32)
    key = jax.random.key(7)
    np.testing.assert_allclose(
        plan.quantum_parameter(u, electric, magnetic)[1], 1.0, rtol=1e-6
    )
    np.testing.assert_allclose(plan.spin_axis(u, electric, magnetic)[:, 2], 1.0)

    @jax.jit
    def run(depth: jax.Array) -> jax.Array:
        def body(
            carry: tuple[jax.Array, jax.Array], step: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            spin, depth = carry
            result = plan.apply(
                u,
                electric,
                magnetic,
                dt,
                active,
                depth,
                plan.uniforms(jax.random.fold_in(key, step), zero_ids, ids),
                spin=spin,
            )
            spin = result.spin
            assert spin is not None
            return (spin, result.optical_depth), jnp.mean(spin[:, 2])

        _, history = jax.lax.scan(body, (jnp.zeros((count, 3)), depth), jnp.arange(steps))
        return history

    polarization = np.asarray(
        run(plan.initial_optical_depth(jax.random.key(8), zero_ids, ids))
    )
    up = _integrate(
        lambda s: sum(_sk_compton(1.0, s, 1.0, -1.0, tau) for tau in _SIGNS), 1.0
    )
    down = _integrate(
        lambda s: sum(_sk_compton(1.0, s, -1.0, 1.0, tau) for tau in _SIGNS), 1.0
    )
    equilibrium = (down - up) / (down + up)
    rate = _RATE_SCALE / gamma * (up + down)
    time = dt * np.arange(1, steps + 1)
    expected = equilibrium * (1.0 - np.exp(-rate * time))
    # Spins polarize antiparallel to B (Sokolov–Ternov direction) at the flip rate.
    assert equilibrium < -0.8 and rate * time[-1] > 3.0
    # Binomial noise of 4096 spins is ≤ 0.016.
    np.testing.assert_allclose(polarization, expected, atol=0.05)
    np.testing.assert_allclose(
        np.mean(polarization[-20:]), np.mean(expected[-20:]), atol=0.02
    )


# -- photon polarization -------------------------------------------------------------


@pytest.mark.parametrize("chi", [0.01, 1.0])
def test_emitted_photon_polarization_degree_matches_the_lcfa(
    tables: dict[tuple[str, str], QEDTable], chi: float
) -> None:
    plan = _compton(tables, "photon-polarized")
    count, gamma = 40000, 1000.0
    u = jnp.zeros((count, 3)).at[:, 0].set(math.sqrt(gamma**2 - 1.0))
    magnetic = jnp.zeros((count, 3)).at[:, 2].set(chi * _SCHWINGER / gamma)
    ids = jnp.arange(count, dtype=jnp.uint32)
    result = plan.apply(
        u,
        jnp.zeros((count, 3)),
        magnetic,
        1.0e-6,
        jnp.ones((count,), dtype=bool),
        jnp.full((count,), 1.0e-300),
        plan.uniforms(jax.random.key(3), jnp.zeros_like(ids), ids),
    )
    assert result.photon_stokes is not None and result.photon_axis is not None
    emitted = np.asarray(result.emitted[:, 0])
    assert emitted.all()
    stokes = np.asarray(result.photon_stokes[:, 0])
    fraction = np.asarray(result.photon_fraction[:, 0])
    # The polarization basis is the transverse force, here ±ŷ.
    np.testing.assert_allclose(np.abs(np.asarray(result.photon_axis[:, 0, 1])), 1.0)

    def moment(power: int, tau: float | None) -> float:
        return _integrate(
            lambda s: (
                s**power
                * sum(
                    _sk_compton(chi, s, 0.0, 0.0, value) * (1.0 if tau is None else value)
                    for value in _SIGNS
                )
            ),
            chi,
        )

    number = moment(0, 1.0) / moment(0, None)
    energy = moment(1, 1.0) / moment(1, None)
    # Sampling noise of 40000 photons: ≤ 0.005 (number), ≤ 0.008 (power).
    np.testing.assert_allclose(np.mean(stokes), number, atol=0.015)
    np.testing.assert_allclose(
        np.sum(stokes * fraction) / np.sum(fraction), energy, atol=0.025
    )
    if chi < 0.1:
        # The classical synchrotron degrees 3/5 (number) and 3/4 (power).
        np.testing.assert_allclose((number, energy), (0.6, 0.75), atol=0.02)


def test_pair_creation_follows_photon_polarization_and_polarizes_the_pair(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    plan = _breit_wheeler(tables, "spin-and-photon-polarized")
    count, energy, amplitude = 40000, 2000.0, 500.0
    # A photon against a crossed field: E = a x̂, B = a ŷ, k = −ε ẑ gives
    # E + c k̂×B = 2a x̂ and χ_γ = 2aε/a_S (in mc² units).
    chi = 2.0 * amplitude * energy / _SCHWINGER
    k = jnp.zeros((count, 3)).at[:, 2].set(-energy * _MASS)
    electric = jnp.zeros((count, 3)).at[:, 0].set(amplitude)
    magnetic = jnp.zeros((count, 3)).at[:, 1].set(amplitude)
    np.testing.assert_allclose(plan.quantum_parameter(k, electric, magnetic)[1], chi)
    ids = jnp.arange(count, dtype=jnp.uint32)
    for tau in _SIGNS:
        # Stokes relative to an axis rotated by 90° are (−τ, 0) in the local basis.
        stokes = jnp.zeros((count, 2)).at[:, 0].set(-tau)
        axis = jnp.zeros((count, 3)).at[:, 1].set(1.0)
        rate = plan.rate(k, electric, magnetic, stokes=stokes, polarization_axis=axis)
        expected = _integrate(
            lambda e, tau=tau: sum(
                _sk_pairs(chi, e, tau, up, down) for up in _SIGNS for down in _SIGNS
            ),
            chi,
        )
        np.testing.assert_allclose(rate, _RATE_SCALE / energy * expected, rtol=1e-6)
        result = plan.apply(
            k,
            electric,
            magnetic,
            1.0e-12,
            jnp.ones((count,), dtype=bool),
            jnp.full((count,), 1.0e-300),
            plan.uniforms(jax.random.key(5), jnp.zeros_like(ids), ids),
            stokes=stokes,
            polarization_axis=axis,
        )
        assert result.electron_spin is not None and result.positron_spin is not None
        assert bool(jnp.all(result.decayed))
        # ê₊ = k̂ × (E + c k̂×B)/|…| = −ŷ and ê₋ = +ŷ.
        positron = -np.asarray(result.positron_spin[:, 1])
        electron = np.asarray(result.electron_spin[:, 1])
        fraction = np.asarray(result.electron_fraction)
        # Positrons above half the photon energy (electron fraction below 1/2).
        for selected, lower, upper in (
            (np.ones_like(fraction, dtype=bool), 1e-12, 1.0 - 1e-12),
            (fraction < 0.5, 1e-12, 0.5),
        ):
            total = _integrate_range(
                lambda e, tau=tau: sum(
                    _sk_pairs(chi, e, tau, up, down) for up in _SIGNS for down in _SIGNS
                ),
                lower,
                upper,
            )
            mean_positron, mean_electron = (
                _integrate_range(
                    lambda e, tau=tau, lepton=lepton: sum(
                        (up if lepton == 0 else down) * _sk_pairs(chi, e, tau, up, down)
                        for up in _SIGNS
                        for down in _SIGNS
                    ),
                    lower,
                    upper,
                )
                / total
                for lepton in (0, 1)
            )
            count_selected = int(np.sum(selected))
            # Binomial noise of the selected ±1 spins, with a 4σ bound.
            bound = 4.0 / math.sqrt(count_selected)
            np.testing.assert_allclose(
                np.mean(positron[selected]), mean_positron, atol=bound
            )
            np.testing.assert_allclose(
                np.mean(electron[selected]), mean_electron, atol=bound
            )
        if tau < 0.0:
            # Pairs from perpendicular photons are created polarized antiparallel
            # to their own ê (the energy-integrated polarization of pairs from
            # parallel photons vanishes).
            assert np.mean(positron) < -0.1 and np.mean(electron) < -0.1


# -- PIC integration -------------------------------------------------------------------


class _UniformMagnetic(StrictModule):
    """Uniform ``B = b ẑ``."""

    strength: float = eqx.field(static=True)

    @property
    def source_id(self) -> str:
        return f"uniform-magnetic-{self.strength!r}"

    def external_fields(self, positions: Any, times: Any, /) -> ExternalFieldSample:
        magnetic = jnp.zeros((times.shape[0], 3)).at[:, 2].set(self.strength)
        return ExternalFieldSample(
            jnp.zeros_like(magnetic), magnetic, jnp.ones(times.shape, dtype=bool)
        )


def _species(
    capacity: int, sign: float, name: str, offset: int, dimension: int
) -> PICSpeciesPlan:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + capacity),
        jnp.ones((capacity,)),
        ambient_dimension=dimension,
    ).prepare()
    return PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )


def _cascade(
    tables: dict[tuple[str, str], QEDTable],
    polarization: QEDPolarizationModel,
    photon_capacity: int,
    dimension: int,
    **options: Any,
) -> QEDCascadeProcess:
    photons = QEDPhotonSpeciesPlan(
        photon_capacity,
        dimension,
        escape_lower=(-1.0e9,) * dimension,
        escape_upper=(1.0e9,) * dimension,
        energy_edges=tuple(np.geomspace(1.0, 1.0e5, 11) * _MASS),
    )
    return QEDCascadeProcess(
        _compton(tables, polarization, **options),
        photons,
        emitters=(0, 1),
        breit_wheeler=_breit_wheeler(tables, polarization),
        electron=0,
        positron=1,
        gather_species=0,
        minimum_photon_energy=2.0 * _MASS,
    )


def _reduced_run(process: QEDCascadeProcess, capacity: int, field: float) -> Any:
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(64, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [100.0]]))
    return phx.solver.ElectromagneticPICPlan(
        phx.solver.ReducedMaxwellPICFieldSolver(
            phx.solver.CompatibleMaxwell1DPlan(grid), D.pic.ReducedPICTransferPlan(grid)
        ),
        species=(
            _species(capacity, -1.0, "electrons", 0, 1),
            _species(capacity, 1.0, "positrons", 100000, 1),
        ),
        processes=(process,),
        ownership="subgrid-reaction",
        external_fields=(_UniformMagnetic(field),),
        key=jax.random.key(1),
    )


def _seed(run: Any, capacity: int, seeds: int, dt: float, gamma: float) -> Any:
    position = jnp.zeros((capacity, 1)).at[:seeds, 0].set(jnp.linspace(30.0, 70.0, seeds))
    active = jnp.arange(capacity) < seeds
    velocity = jnp.zeros((capacity, 3)).at[:seeds, 0].set(math.sqrt(1.0 - gamma**-2))
    mass = jnp.where(active, 1.0e-9 * _MASS, 0.0)
    return run.initialize(
        (position, position),
        (velocity, -velocity),
        dt,
        active_masks=(active, active),
        masses=(mass, mass),
    )


@eqx.filter_jit
def _advance(run: Any, state: Any, dt: float, steps: int) -> Any:
    def body(carry: Any, _: None) -> tuple[Any, jax.Array]:
        result = run.step_detailed(carry, dt)
        return result.accepted_state, result.successful

    return jax.lax.scan(body, state, None, length=steps)


def test_cascade_spins_precess_with_the_run_pusher(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    capacity, seeds, gamma, field, dt, steps = 16, 4, 10.0, 1.0, 0.5, 400
    # Leptons below minimum_gamma do not emit: the spin evolves by T-BMT alone.
    process = _cascade(
        tables, "spin-and-photon-polarized", 64, 1, minimum_gamma=2.0 * gamma
    )
    run = _reduced_run(process, capacity, field)
    state = _seed(run, capacity, seeds, dt, gamma)
    cascade = state.processes[0]
    for index in (0, 1):
        species = state.species[index]
        u = np.asarray(species.particles.proper_velocity)
        direction = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-300)
        cascade = process.polarize(cascade, index, species, direction)
    state = eqx.tree_at(lambda value: value.processes, state, (cascade,))
    state, successful = _advance(run, state, dt, steps)
    assert bool(jnp.all(successful))
    polarization = state.processes[0].polarization
    for index, sign in ((0, 1.0), (1, -1.0)):
        species = state.species[index]
        active = np.asarray(species.population.active)
        u = np.asarray(species.particles.proper_velocity)[active]
        spin = np.asarray(polarization.spins[index].spin)[active]
        direction = u / np.linalg.norm(u, axis=-1, keepdims=True)
        angle = np.arctan2(np.cross(direction, spin)[:, 2], np.sum(direction * spin, -1))
        cyclotron = field / gamma
        # Electrons (q < 0) and positrons turn in opposite senses; the spin leads
        # the momentum by aγω_c t in each (Boris phase error O((ω_cΔt)²)).
        expected = (
            sign * ELECTRON_MAGNETIC_MOMENT_ANOMALY * gamma * cyclotron * steps * dt
        )
        np.testing.assert_allclose(angle, expected, rtol=2e-3)
        np.testing.assert_allclose(np.linalg.norm(spin, axis=-1), 1.0, rtol=1e-12)


def test_spin_polarized_cascade_restarts_exactly(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    capacity, dt = 32, 0.005
    process = _cascade(tables, "spin-and-photon-polarized", 128, 1)
    run = _reduced_run(process, capacity, 2000.0)
    state, _ = _advance(run, _seed(run, capacity, 6, dt, 4000.0), dt, 40)
    polarization = state.processes[0].polarization
    assert int(jnp.sum(state.processes[0].photons.population.active)) > 0
    assert float(jnp.max(jnp.abs(polarization.spins[0].spin))) > 0.0
    restored = run.restore(run.checkpoint(state))
    for resumed, original in zip(
        jax.tree.leaves(_advance(run, restored, dt, 30)[0]),
        jax.tree.leaves(_advance(run, state, dt, 30)[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(resumed, original)
    other = _reduced_run(_cascade(tables, "photon-polarized", 128, 1), capacity, 2000.0)
    with pytest.raises(ValueError):
        other.restore(run.checkpoint(state))


def _spectral_run(
    tables: dict[tuple[str, str], QEDTable],
    velocity: tuple[float, float, float] | None,
    capacity: int,
) -> Any:
    spacing, cells = 1.0, 8
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(cells, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [cells * spacing] * 3]))
    bridge = D.StructuredCochainBridge(grid)
    species = (
        _species(capacity, -1.0, "electrons", 0, 3),
        _species(capacity, 1.0, "positrons", 100000, 3),
    )
    transfer = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=1)
    charged = tuple(
        D.ChargedParticlePlan(sign * jnp.ones((capacity,)), name).prepare(
            value.population.particles
        )
        for sign, name, value in (
            (-1.0, "electrons", species[0]),
            (1.0, "positrons", species[1]),
        )
    )
    transfers = tuple(transfer.prepare(value) for value in charged)
    currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)
    options: dict[str, Any] = {"charge_conservation": "update-with-rho"}
    if velocity is not None:
        options |= {"variant": "galilean", "galilean_velocity": velocity}
    solver = phx.solver.maxwell.spectral.SpectralMaxwellPlan(bridge, **options).prepare(
        transfers, currents
    )
    return phx.solver.ElectromagneticPICPlan(
        solver,
        species=species,
        processes=(_cascade(tables, "photon-polarized", 4 * capacity, 3),),
        ownership="subgrid-reaction",
        external_fields=(_UniformMagnetic(2000.0),),
        key=jax.random.key(11),
    )


def test_galilean_grid_qed_emission_equals_the_lab_grid(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    capacity, seeds, dt, steps = 16, 6, 0.005, 60
    grid_velocity = (0.5, 0.0, 0.0)
    runs = {
        "lab": _spectral_run(tables, None, capacity),
        "galilean": _spectral_run(tables, grid_velocity, capacity),
    }
    position = (
        jnp.zeros((capacity, 3))
        .at[:seeds]
        .set(
            jnp.stack(
                (
                    jnp.linspace(2.0, 6.0, seeds),
                    jnp.full(seeds, 4.0),
                    jnp.full(seeds, 4.0),
                ),
                axis=-1,
            )
        )
    )
    active = jnp.arange(capacity) < seeds
    velocity = jnp.zeros((capacity, 3)).at[:seeds, 0].set(math.sqrt(1.0 - 4000.0**-2))
    mass = jnp.where(active, 1.0e-9 * _MASS, 0.0)
    states = {
        name: _advance(
            run,
            run.initialize(
                (position, position),
                (velocity, -velocity),
                dt,
                active_masks=(active, active),
                masses=(mass, mass),
            ),
            dt,
            steps,
        )
        for name, run in runs.items()
    }
    for _, successful in states.values():
        assert bool(jnp.all(successful))
    lab, galilean = states["lab"][0], states["galilean"][0]
    time = float(galilean.time)
    shift = time * np.asarray(grid_velocity)
    for left, right in zip(lab.species, galilean.species, strict=True):
        np.testing.assert_array_equal(left.population.active, right.population.active)
        np.testing.assert_allclose(
            left.particles.proper_velocity,
            right.particles.proper_velocity,
            rtol=1e-9,
            atol=1e-6,
        )
        lab_position = np.asarray(left.particles.position)
        grid_position = np.asarray(right.particles.position) + shift
        # Species positions are periodic in the 8-cell box.
        difference = (grid_position - lab_position + 4.0) % 8.0 - 4.0
        np.testing.assert_allclose(
            np.where(np.asarray(left.population.active)[:, None], difference, 0.0),
            0.0,
            atol=1e-9,
        )
    left = lab.processes[0].photons
    right = galilean.processes[0].photons
    assert int(jnp.sum(left.population.active)) > 0
    np.testing.assert_array_equal(left.population.active, right.population.active)
    np.testing.assert_allclose(left.momentum, right.momentum, rtol=1e-9, atol=1e-12)
    live = np.asarray(left.population.active)[:, None]
    # Photons drift at c k̂ − v_grid in grid coordinates: lab = grid + v_grid t.
    np.testing.assert_allclose(
        np.where(live, np.asarray(right.position) + shift, 0.0),
        np.where(live, np.asarray(left.position), 0.0),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        lab.processes[0].polarization.photon_stokes,
        galilean.processes[0].polarization.photon_stokes,
        atol=1e-12,
    )


def test_polarized_configurations_are_validated(
    tables: dict[tuple[str, str], QEDTable],
) -> None:
    averaged = tables["nonlinear-compton", "averaged"]
    with pytest.raises(ValueError, match="spin_tables"):
        NonlinearComptonPlan(
            "lcfa",
            _SCALE,
            -_MASS,
            _MASS,
            averaged,
            maximum_chi=10.0,
            minimum_gamma=1.0,
            polarization="spin-and-photon-polarized",
        )
    with pytest.raises(ValueError, match="'positive' and 'negative'"):
        NonlinearComptonPlan(
            "lcfa",
            _SCALE,
            -_MASS,
            _MASS,
            averaged,
            maximum_chi=10.0,
            minimum_gamma=1.0,
            polarization="spin-and-photon-polarized",
            spin_tables=(averaged, averaged),
        )
    with pytest.raises(ValueError, match="averaged"):
        _compton(
            {
                **tables,
                ("nonlinear-compton", "averaged"): tables[
                    "nonlinear-compton", "positive"
                ],
            },
            "unpolarized",
        )
    photons = QEDPhotonSpeciesPlan(
        64, 1, escape_lower=(-1.0,), escape_upper=(1.0,), energy_edges=(0.0, 1.0)
    )
    with pytest.raises(ValueError, match="different polarizations"):
        QEDCascadeProcess(
            _compton(tables, "photon-polarized"),
            photons,
            emitters=(0,),
            breit_wheeler=_breit_wheeler(tables, "unpolarized"),
            electron=0,
            positron=1,
            gather_species=0,
        )
    unpolarized = QEDCascadeProcess(
        _compton(tables, "unpolarized"),
        photons,
        emitters=(0,),
        electron=0,
        positron=1,
        gather_species=0,
    )
    run = _reduced_run(_cascade(tables, "unpolarized", 64, 1), 32, 1.0)
    state = _seed(run, 32, 2, 0.01, 10.0)
    with pytest.raises(ValueError, match="spin-and-photon-polarized"):
        unpolarized.polarize(state.processes[0], 0, state.species[0], np.zeros((32, 3)))
    polarized = _cascade(tables, "spin-and-photon-polarized", 64, 1)
    polarized_run = _reduced_run(polarized, 32, 1.0)
    polarized_state = _seed(polarized_run, 32, 2, 0.01, 10.0)
    with pytest.raises(ValueError, match=r"\|S\| ≤ 1"):
        polarized.polarize(
            polarized_state.processes[0],
            0,
            polarized_state.species[0],
            np.full((32, 3), 1.0),
        )
