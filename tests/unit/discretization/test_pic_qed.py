#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strong-field QED tables, nonlinear Compton emission and Breit–Wheeler decay.

References, evaluated with SciPy's modified Bessel functions (independent of
`phydrax.special` and of the tables' quadrature): the locally constant field
spectra of Baier & Katkov / Ritus,

    Compton:  dN/dξ ∝ [(1 − ξ + 1/(1 − ξ)) K_{2/3}(δ) − ∫_δ^∞ K_{1/3}],  δ = 2ξ/(3χ(1 − ξ)),
    pairs:    dN/dξ ∝ [(ξ/(1 − ξ) + (1 − ξ)/ξ) K_{2/3}(δ) + ∫_δ^∞ K_{1/3}],  δ = 2/(3χξ(1 − ξ)),

normalized by ``1/(√3π)``; Erber's asymptotics (Rev. Mod. Phys. 38, 626,
1966, converted to ``χ_γ = 2χ_Erber``) ``T → (3/16)√(3/2) e^{−8/(3χ)}`` and
``T → 0.378 χ^{−1/3}``; the classical photon number ``K → 5χ/(2√3)``; and the
quantum-corrected Landau–Lifshitz drift of `RadiationReactionPlan` (Q1) for the
mean emitted power as ``χ → 0``.
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
from scipy.integrate import cumulative_trapezoid, quad
from scipy.special import kv
from scipy.stats import kstest

from phydrax import ElectromagneticScaleContract
from phydrax.discretization.pic import (
    NonlinearBreitWheelerPlan,
    NonlinearComptonPlan,
    PIC_CODE_RELATIVITY,
    QEDEventFlag,
    QEDTable,
    RadiationReactionPlan,
    RadiationReactionTables,
)
from phydrax.units import CHARGE, UnitDefinition


# Code units c = ε₀ = 1 with α = 1/137.036: q = m = 1 gives E_S = 1/ħ.
_HBAR = Fraction(1, 1) / (4 * Fraction(math.pi) * Fraction(1, 137036) * 1000)
_SCALE = ElectromagneticScaleContract.code_units(
    PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=_HBAR,
    boltzmann_constant=1,
    elementary_charge=1,
    electron_mass=1,
    vacuum_permittivity=1,
    constant_set_id="qed-test",
)
_CRITICAL = 1.0 / float(_HBAR)
_NORMALIZATION = 1.0 / (math.sqrt(3.0) * math.pi)


def _k(order: float, x: float) -> float:
    return float(kv(np.float64(order), np.float64(x)))


def _tail(order: float, x: float) -> float:
    return quad(lambda t: _k(order, t), x, np.inf, epsabs=0.0, epsrel=1e-12, limit=200)[0]


def _compton_density(chi: float, xi: float) -> float:
    delta = 2.0 * xi / (3.0 * chi * (1.0 - xi))
    return _NORMALIZATION * (
        (1.0 - xi + 1.0 / (1.0 - xi)) * _k(2.0 / 3.0, delta) - _tail(1.0 / 3.0, delta)
    )


def _pair_density(chi: float, xi: float) -> float:
    delta = 2.0 / (3.0 * chi * xi * (1.0 - xi))
    return _NORMALIZATION * (
        (xi / (1.0 - xi) + (1.0 - xi) / xi) * _k(2.0 / 3.0, delta)
        + _tail(1.0 / 3.0, delta)
    )


def _compton_total(chi: float) -> float:
    # Integrate in log δ, where the photon spectrum is smooth for every χ.
    def integrand(x: float) -> float:
        delta = math.exp(x)
        xi = 3.0 * chi * delta / (2.0 + 3.0 * chi * delta)
        return (
            _compton_density(chi, xi) * 6.0 * chi * delta / (2.0 + 3.0 * chi * delta) ** 2
        )

    breaks = np.linspace(math.log(1e-14), math.log(90.0), 40)
    total = sum(
        quad(integrand, float(a), float(b), epsabs=0.0, epsrel=1e-11, limit=200)[0]
        for a, b in zip(breaks[:-1], breaks[1:], strict=True)
    )
    # Below δ = 1e-14 the density in log δ is ∝ δ^{1/3}.
    return total + 3.0 * integrand(float(breaks[0]))


def _pair_total(chi: float) -> float:
    width = min(0.5, 2.0 * math.sqrt(chi))
    breaks = np.concatenate(
        (np.linspace(1e-9, 0.5 - width, 8), np.linspace(0.5 - width, 0.5, 20)[1:])
    )
    return 2.0 * sum(
        quad(
            lambda xi: _pair_density(chi, xi),
            float(a),
            float(b),
            epsabs=0.0,
            epsrel=1e-11,
            limit=200,
        )[0]
        for a, b in zip(breaks[:-1], breaks[1:], strict=True)
    )


def _reference_cdf(density: Any, lower: float, upper: float, count: int) -> Any:
    """Exact cumulative spectrum on a fine grid (trapezoid in ξ^{1/3})."""
    s = np.linspace(lower ** (1.0 / 3.0), upper ** (1.0 / 3.0), count)
    xi = s**3
    values = np.array(
        [density(float(value)) * 3.0 * float(v) ** 2 for value, v in zip(xi, s)],
        dtype=np.float64,
    )
    cumulative = np.asarray(cumulative_trapezoid(values, x=s, initial=0))
    return xi, cumulative / cumulative[-1]


@pytest.fixture(scope="module")
def compton_table() -> QEDTable:
    return QEDTable("nonlinear-compton", maximum_chi=100.0)


@pytest.fixture(scope="module")
def pair_table() -> QEDTable:
    return QEDTable("nonlinear-breit-wheeler", maximum_chi=2000.0)


def _compton(table: QEDTable, **options: Any) -> NonlinearComptonPlan:
    options.setdefault("maximum_chi", 100.0)
    options.setdefault("minimum_gamma", 1.0)
    return NonlinearComptonPlan(
        options.pop("model", "lcfa"), _SCALE, -1.0, 1.0, table, **options
    )


def _crossed(gamma: np.ndarray, chi: np.ndarray) -> tuple[Any, Any, Any]:
    """Leptons along x in a magnetic field along z with the requested χ."""
    count = gamma.shape[0]
    speed = np.sqrt(gamma**2 - 1.0)
    u = np.zeros((count, 3))
    u[:, 0] = speed
    b = np.zeros((count, 3))
    b[:, 2] = chi * _CRITICAL / speed
    return jnp.asarray(u), jnp.zeros((count, 3)), jnp.asarray(b)


def _uniforms(count: int, slots: int, seed: int) -> Any:
    return jnp.asarray(np.random.default_rng(seed).uniform(size=(count, slots, 2)))


@pytest.mark.parametrize("chi", [1.0e-3, 0.3, 5.0, 80.0], ids=lambda v: f"chi={v:g}")
def test_compton_rate_matches_direct_quadrature(
    compton_table: QEDTable, chi: float
) -> None:
    value = float(compton_table.rate_function(chi))
    np.testing.assert_allclose(value, _compton_total(chi), rtol=2e-7)


def test_compton_rate_has_the_classical_photon_number_limit(
    compton_table: QEDTable,
) -> None:
    chi = np.array([1.0e-9, 1.0e-7])
    np.testing.assert_allclose(
        compton_table.rate_function(chi), 5.0 * chi / (2.0 * math.sqrt(3.0)), rtol=1e-5
    )
    assert float(compton_table.rate_function(0.0)) == 0.0


@pytest.mark.parametrize("chi", [0.05, 1.0, 30.0, 1500.0], ids=lambda v: f"chi={v:g}")
def test_breit_wheeler_rate_matches_direct_quadrature(
    pair_table: QEDTable, chi: float
) -> None:
    np.testing.assert_allclose(
        float(pair_table.rate_function(chi)), _pair_total(chi), rtol=2e-7
    )


def test_breit_wheeler_rate_follows_erber_asymptotics(pair_table: QEDTable) -> None:
    small = np.array([4.0e-3, 2.0e-2])
    low = np.asarray(pair_table.rate_function(small)) / small
    # T/(0.2296 e^{−8/(3χ)}) = 1 − O(χ); below the table (χ < 0.01) the rate
    # continues on the asymptote.
    ratio = low / ((3.0 / 16.0) * math.sqrt(1.5) * np.exp(-8.0 / (3.0 * small)))
    np.testing.assert_allclose(ratio[0], 1.0, atol=2e-3)
    np.testing.assert_allclose(ratio[1], 1.0, atol=6e-3)
    assert ratio[1] < 1.0
    large = 1500.0
    high = float(pair_table.rate_function(large)) / large
    np.testing.assert_allclose(high / (0.378 * large ** (-1.0 / 3.0)), 1.0, atol=0.02)


@pytest.mark.parametrize("process", ["compton", "pairs"])
def test_tables_report_monotone_spectra_and_bounded_interpolation(
    compton_table: QEDTable, pair_table: QEDTable, process: str
) -> None:
    table = compton_table if process == "compton" else pair_table
    assert table.minimum_cdf_increment >= 0.0
    assert table.rate_interpolation_error <= 1e-6
    assert table.cdf_row_error + table.cdf_node_error <= 1e-3
    assert table.truncated_mass < 1e-20
    probability = jnp.linspace(0.0, 1.0, 101)
    for chi in (0.02, 0.7, 40.0):
        fraction = table.quantile(chi, probability)
        assert bool(jnp.all(jnp.diff(fraction) >= 0.0))
        np.testing.assert_allclose(table.cdf(chi, fraction), probability, atol=1e-12)


def test_table_construction_refuses_unresolved_spectra() -> None:
    with pytest.raises(ValueError, match="spectrum interpolation error"):
        QEDTable(
            "nonlinear-breit-wheeler",
            maximum_chi=10.0,
            spectrum_nodes=17,
            cdf_tolerance=1e-5,
        )
    with pytest.raises(ValueError, match="maximum_chi"):
        QEDTable("nonlinear-compton", maximum_chi=1e-6)


def test_photon_spectrum_passes_a_kolmogorov_smirnov_test_at_fixed_chi(
    compton_table: QEDTable,
) -> None:
    chi, count = 2.0, 20000
    plan = _compton(compton_table)
    uniform = np.random.default_rng(3).uniform(size=count)
    fraction = np.asarray(
        plan.sample_fraction(jnp.full((count,), chi), jnp.full((count,), 1e3), uniform)
    )
    grid, reference = _reference_cdf(
        lambda xi: _compton_density(chi, xi), 1e-12, 1.0 - 1e-9, 1500
    )
    statistic = kstest(fraction, lambda x: np.interp(x, grid, reference))
    assert statistic.pvalue > 1e-3
    assert statistic.statistic < 1.63 / math.sqrt(count)


def test_pair_spectrum_passes_a_kolmogorov_smirnov_test_at_fixed_chi(
    pair_table: QEDTable,
) -> None:
    count = 20000
    plan = NonlinearBreitWheelerPlan(_SCALE, 1.0, 1.0, pair_table, maximum_chi=2000.0)
    # Photons along x in B_z with χ_γ = 1.5 and zero optical depth all decay.
    energy = 1.0e3
    k = jnp.zeros((count, 3)).at[:, 0].set(energy)
    b = jnp.zeros((count, 3)).at[:, 2].set(1.5 * _CRITICAL / energy)
    result = plan.apply(
        k,
        jnp.zeros((count, 3)),
        b,
        1.0e-9,
        jnp.ones((count,), dtype=bool),
        jnp.zeros((count,)),
        jnp.asarray(np.random.default_rng(5).uniform(size=(count, 2))),
    )
    assert bool(jnp.all(result.decayed))
    np.testing.assert_allclose(result.quantum_parameter, 1.5, rtol=1e-12)
    grid, reference = _reference_cdf(
        lambda xi: _pair_density(1.5, min(xi, 1.0 - xi)), 1e-7, 1.0 - 1e-7, 3000
    )
    statistic = kstest(
        np.asarray(result.electron_fraction), lambda x: np.interp(x, grid, reference)
    )
    assert statistic.pvalue > 1e-3


@pytest.mark.parametrize("chi", [0.01, 0.05], ids=lambda v: f"chi={v:g}")
def test_small_chi_mean_emitted_power_equals_quantum_corrected_landau_lifshitz(
    compton_table: QEDTable, chi: float
) -> None:
    count, gamma = 200000, 2000.0
    plan = _compton(compton_table, maximum_event_probability=0.05)
    u, e, b = _crossed(np.full(count, gamma), np.full(count, chi))
    rate = float(plan.rate(u[:1], e[:1], b[:1])[0])
    dt = 0.04 / rate
    uniforms = _uniforms(count, plan.maximum_subcycles, 9)
    depth = -jnp.log1p(-jnp.asarray(np.random.default_rng(10).uniform(size=count)))
    result = eqx.filter_jit(lambda plan, *arguments: plan.apply(*arguments))(
        plan, u, e, b, dt, jnp.ones((count,), dtype=bool), depth, uniforms
    )
    power = float(jnp.sum(result.photon_energy)) / (count * dt)
    reaction = RadiationReactionPlan(
        "quantum-corrected-landau-lifshitz",
        _SCALE,
        -1.0,
        1.0,
        tables=RadiationReactionTables(maximum_chi=1.0),
        maximum_chi=1.0,
        minimum_gamma=1.0,
    )
    drift = reaction.apply(u[:1], e[:1], b[:1], dt, jnp.ones((1,), dtype=bool))
    expected = -float(drift.drift_rate[0])
    events = float(jnp.sum(result.emitted))
    # Relative standard error of a compound Poisson sum: √(⟨ω²⟩/⟨ω⟩²/n).
    spread = float(
        jnp.sqrt(jnp.sum(result.photon_energy**2) / events)
        / (jnp.sum(result.photon_energy) / events)
    )
    np.testing.assert_allclose(power, expected, rtol=4.0 * spread / math.sqrt(events))


@pytest.mark.parametrize("probability", [0.1, 0.05, 0.02], ids=lambda v: f"p={v:g}")
def test_emission_count_and_energy_do_not_depend_on_the_subcycle_probability(
    compton_table: QEDTable, probability: float
) -> None:
    # One expected emission per lepton, split into 10, 20 or 50 subcycles. At
    # χ = 0.01 the rate K(χ)/γ is independent of γ to O(χ), so recoil leaves the
    # expected count at W Δt; discarding the overshoot at each crossing would
    # bias it low by (1 − e^{−p})/p (−4.9 % at p = 0.1).
    count, gamma, chi = 60000, 2000.0, 0.01
    plan = _compton(
        compton_table, maximum_event_probability=probability, maximum_subcycles=64
    )
    u, e, b = _crossed(np.full(count, gamma), np.full(count, chi))
    rate = float(plan.rate(u[:1], e[:1], b[:1])[0])
    dt = 1.0 / rate
    depth = -jnp.log1p(-jnp.asarray(np.random.default_rng(12).uniform(size=count)))
    result = eqx.filter_jit(lambda plan, *arguments: plan.apply(*arguments))(
        plan,
        u,
        e,
        b,
        dt,
        jnp.ones((count,), dtype=bool),
        depth,
        _uniforms(count, 64, 13),
    )
    np.testing.assert_array_equal(result.subcycles, round(1.0 / probability))
    # A crossing still pending at the step end is a nonpositive optical depth.
    events = float(jnp.sum(result.emitted)) + float(jnp.sum(result.optical_depth <= 0.0))
    np.testing.assert_allclose(events / count, 1.0, rtol=4.0 / math.sqrt(count))
    reaction = RadiationReactionPlan(
        "quantum-corrected-landau-lifshitz",
        _SCALE,
        -1.0,
        1.0,
        tables=RadiationReactionTables(maximum_chi=1.0),
        maximum_chi=1.0,
        minimum_gamma=1.0,
    )
    power = -float(
        reaction.apply(u[:1], e[:1], b[:1], dt, jnp.ones((1,), dtype=bool)).drift_rate[0]
    )
    emitted = float(jnp.sum(result.emitted))
    energy = float(jnp.sum(result.photon_energy))
    spread = math.sqrt(float(jnp.sum(result.photon_energy**2)) / emitted) / (
        energy / emitted
    )
    # Recoil lowers the power by about half the ⟨ω⟩/ε ≈ 0.5 % lost per emission.
    np.testing.assert_allclose(
        energy / (count * dt), power, rtol=4.0 * spread / math.sqrt(emitted) + 5e-3
    )


def test_improved_lcfa_holds_the_infrared_spectrum_flat_below_its_threshold(
    compton_table: QEDTable,
) -> None:
    lcfa = _compton(compton_table)
    improved = _compton(compton_table, model="improved-lcfa")
    gamma, chi = np.array([4000.0]), np.array([0.5])
    u, e, b = _crossed(gamma, chi)
    # A constant field has no finite variation time: the models coincide.
    np.testing.assert_array_equal(improved.rate(u, e, b), lcfa.rate(u, e, b))
    # S = χτ/(8γτ_C) = 2 puts ω_LCFA near 1 % of the lepton energy.
    tau = jnp.asarray([1.28e5 * lcfa.compton_time])
    threshold, allowed = improved.infrared_threshold(chi, gamma, tau)
    xi_th = float(threshold[0])
    assert bool(allowed[0]) and 0.0 < xi_th < 0.05
    # Independent reference: LCFA spectrum above ξ_th plus the flat part below.
    above = quad(
        lambda xi: _compton_density(0.5, xi),
        xi_th,
        1.0,
        epsabs=0.0,
        epsrel=1e-10,
        limit=400,
    )[0]
    expected = lcfa.rate_scale / gamma[0] * (above + xi_th * _compton_density(0.5, xi_th))
    np.testing.assert_allclose(
        float(improved.rate(u, e, b, variation_time=tau)[0]), expected, rtol=2e-4
    )
    # The flat part is sampled uniformly on [0, ξ_th].
    count = 40000
    fraction = np.asarray(
        improved.sample_fraction(
            jnp.full((count,), 0.5),
            jnp.full((count,), gamma[0]),
            np.random.default_rng(1).uniform(size=count),
            variation_time=jnp.full((count,), tau[0]),
        )
    )
    below = fraction[fraction < xi_th]
    flat_mass = (
        xi_th
        * _compton_density(0.5, xi_th)
        / (above + xi_th * _compton_density(0.5, xi_th))
    )
    np.testing.assert_allclose(
        below.size / count, flat_mass, atol=4.0 * math.sqrt(flat_mass / count)
    )
    assert kstest(below / xi_th, "uniform").pvalue > 1e-3


@pytest.mark.parametrize("conservation", ["momentum", "energy"])
def test_emission_conserves_the_declared_quantity_and_hands_the_defect_to_the_field(
    compton_table: QEDTable, conservation: str
) -> None:
    plan = _compton(compton_table, conservation=conservation)
    count = 64
    gamma = np.geomspace(20.0, 2.0e4, count)
    u, e, b = _crossed(gamma, np.full(count, 1.0))
    result = plan.apply(
        u,
        e,
        b,
        1e-12,
        jnp.ones((count,), dtype=bool),
        jnp.zeros((count,)),
        _uniforms(count, 8, 2),
    )
    emitted = np.asarray(result.emitted[:, 0])
    assert emitted.all()
    k = np.asarray(result.photon_momentum[:, 0])
    omega = np.asarray(result.photon_energy[:, 0])
    after = np.asarray(result.proper_velocity)
    before = np.asarray(u)
    energy = gamma
    after_energy = np.sqrt(1.0 + np.sum(after**2, axis=-1))
    field_energy = np.asarray(result.field_energy)
    field_momentum = np.asarray(result.field_momentum)
    if conservation == "momentum":
        np.testing.assert_allclose(after + k, before, rtol=1e-14, atol=1e-10)
        # ε' + ω − ε > 0, O(m²c⁴/ε), checked against the unrounded definition.
        defect = (after_energy - energy) + omega
        # The unrounded definition cancels to about 4 ε_machine γ.
        assert np.all(np.abs(field_energy - defect) <= 1e-5 * defect + 1e-15 * energy)
        assert np.all(field_energy > 0.0)
        np.testing.assert_array_equal(field_momentum, 0.0)
    else:
        np.testing.assert_allclose(after_energy + omega, energy, rtol=1e-14)
        np.testing.assert_array_equal(field_energy, 0.0)
        np.testing.assert_allclose(
            field_momentum, after + k - before, rtol=1e-5, atol=1e-11
        )


def test_random_numbers_follow_particle_identity_not_storage_slot(
    compton_table: QEDTable,
) -> None:
    plan = _compton(compton_table)
    count = 32
    u, e, b = _crossed(np.full(count, 1e3), np.linspace(0.5, 3.0, count))
    key = jax.random.key(4)
    high = jnp.zeros((count,), dtype=jnp.uint32)
    low = jnp.arange(count, dtype=jnp.uint32)
    permutation = np.random.default_rng(0).permutation(count)

    def run(order: np.ndarray) -> Any:
        depth = plan.initial_optical_depth(key, high[order], low[order])
        return plan.apply(
            u[order],
            e[order],
            b[order],
            0.5 / float(plan.rate(u[:1], e[:1], b[:1])[0]),
            jnp.ones((count,), dtype=bool),
            depth,
            plan.uniforms(key, high[order], low[order]),
        )

    straight = run(np.arange(count))
    shuffled = run(permutation)
    inverse = np.argsort(permutation)
    for name in ("proper_velocity", "optical_depth", "photon_energy", "emitted"):
        np.testing.assert_array_equal(
            np.asarray(getattr(shuffled, name))[inverse], getattr(straight, name)
        )
    assert int(jnp.sum(straight.emitted)) > 0


def test_event_probability_cap_subcycles_and_refuses_beyond_its_bound(
    compton_table: QEDTable,
) -> None:
    plan = _compton(compton_table, maximum_event_probability=0.1, maximum_subcycles=4)
    u, e, b = _crossed(np.full(3, 1e3), np.array([1.0, 1.0, 200.0]))
    rate = float(plan.rate(u[:1], e[:1], b[:1])[0])
    active = jnp.ones((3,), dtype=bool)
    depth = jnp.full((3,), 50.0)
    fine = plan.apply(u, e, b, 0.25 / rate, active, depth, _uniforms(3, 4, 0))
    assert int(fine.subcycles[0]) == 3
    assert float(fine.event_probability[0]) <= 0.1
    assert int(fine.flags[2]) & QEDEventFlag.CHI_EXCEEDED
    assert not bool(fine.successful)
    coarse = plan.apply(
        u[:2], e[:2], b[:2], 0.5 / rate, active[:2], depth[:2], _uniforms(2, 4, 0)
    )
    assert int(coarse.subcycles[0]) == 4
    assert int(coarse.flags[0]) & QEDEventFlag.EVENT_PROBABILITY_EXCEEDED
    assert not bool(coarse.successful)
    slow = plan.apply(
        u[:2], e[:2], b[:2], 0.2 / rate, active[:2], depth[:2], _uniforms(2, 4, 0)
    )
    assert bool(slow.successful)
    np.testing.assert_array_equal(slow.subcycles, 2)


def test_lcfa_validity_evidence_flags_nonlocal_formation(compton_table: QEDTable) -> None:
    plan = _compton(compton_table)
    count = 16
    u, e, b = _crossed(np.full(count, 1e3), np.full(count, 2.0))
    arguments = (
        u,
        e,
        b,
        1e-12,
        jnp.ones((count,), dtype=bool),
        jnp.zeros((count,)),
        _uniforms(count, 8, 6),
    )
    local = plan.apply(*arguments)
    assert bool(jnp.all(local.formation_ratio == 0.0))
    assert not bool(jnp.any(local.flags & int(QEDEventFlag.OUTSIDE_LCFA_VALIDITY)))
    # A field varying over one Compton time is far from locally constant.
    nonlocal_ = plan.apply(
        *arguments, variation_time=jnp.full((count,), plan.compton_time)
    )
    ratio = np.asarray(nonlocal_.formation_ratio[:, 0])
    assert np.all(ratio > 1.0)
    assert bool(
        jnp.all(nonlocal_.event_flags[:, 0] & int(QEDEventFlag.OUTSIDE_LCFA_VALIDITY))
    )
    assert bool(jnp.all(nonlocal_.flags & int(QEDEventFlag.ONE_STEP_TRIDENT)))
    assert bool(nonlocal_.successful)


def test_plans_refuse_mismatched_tables_and_invalid_options(
    compton_table: QEDTable, pair_table: QEDTable
) -> None:
    with pytest.raises(ValueError, match="nonlinear-compton"):
        _compton(pair_table)
    with pytest.raises(ValueError, match="cover maximum_chi"):
        _compton(compton_table, maximum_chi=1e3)
    with pytest.raises(ValueError, match="model"):
        _compton(compton_table, model="baier-katkov")
    with pytest.raises(ValueError, match="conservation"):
        NonlinearBreitWheelerPlan(
            _SCALE,
            1.0,
            1.0,
            pair_table,
            maximum_chi=10.0,
            conservation="charge",  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="nonlinear-breit-wheeler"):
        NonlinearBreitWheelerPlan(_SCALE, 1.0, 1.0, compton_table, maximum_chi=10.0)
