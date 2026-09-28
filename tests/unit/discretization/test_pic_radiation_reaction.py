#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Radiation reaction: classical Landau–Lifshitz and quantum Fokker–Planck.

References: the Landau–Lifshitz force (Landau & Lifshitz, *The Classical
Theory of Fields*, §76) and its planar cooling solution in a uniform magnetic
field, ``γ(t) = coth(τ ω_B² t + arccoth γ₀)`` for ``u ⊥ B``; the quantum
corrections ``g(χ)`` and ``h(χ)`` and the Fokker–Planck drift
``A = −(2/3) α (mc²/ħ) χ² g(χ)`` and diffusion ``B = (2/3) α (mc²/ħ) γ h(χ)``
of Niel et al., Phys. Rev. E 97, 043209 (2018), evaluated here with SciPy's
Bessel functions, independent of `phydrax.special`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax.discretization.pic import (
    PIC_CODE_RELATIVITY,
    PICChargeModelPlan,
    PICRejectionReason,
    PICSpeciesPlan,
    RadiationReactionFlag,
    RadiationReactionModel,
    RadiationReactionPlan,
    RadiationReactionProcess,
    RadiationReactionTables,
    RelativisticPushPlan,
)
from phydrax.units import CHARGE, UnitDefinition


D = phx.discretization
# One physical species with q/m = −1 in code units (c = ε₀ = 1).
_CHARGE = -0.1
_MASS = 0.1
_TAU = _CHARGE**2 / (6.0 * math.pi * _MASS)


def _scale(
    reduced_planck_constant: Fraction, *, speed_of_light: int = 1
) -> ElectromagneticScaleContract:
    return ElectromagneticScaleContract.code_units(
        PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=speed_of_light,
        reduced_planck_constant=reduced_planck_constant,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="radiation-reaction-test",
    )


_CLASSICAL = _scale(Fraction(1, 10**8))


def _integral(integrand: Callable[[float], float]) -> float:
    # Niel's h integrand scales as χ³: only a relative tolerance is meaningful.
    return quad(integrand, 0, np.inf, limit=400, epsabs=0.0, epsrel=1e-12)[0]


def _bessel_k(order: float, argument: float) -> float:
    return float(kv(np.float64(order), np.float64(argument)))


def _niel_g(chi: float) -> float:
    def integrand(nu: float) -> float:
        d = 2.0 + 3.0 * nu * chi
        return (
            2.0 * nu**2 * _bessel_k(5.0 / 3.0, nu) / d**2
            + 4.0 * nu * (3.0 * nu * chi) ** 2 * _bessel_k(2.0 / 3.0, nu) / d**4
        )

    return 9.0 * math.sqrt(3.0) / (8.0 * math.pi) * _integral(integrand)


def _niel_h(chi: float) -> float:
    def integrand(nu: float) -> float:
        d = 2.0 + 3.0 * nu * chi
        return (
            2.0 * chi**3 * nu**3 * _bessel_k(5.0 / 3.0, nu) / d**3
            + 54.0 * chi**5 * nu**4 * _bessel_k(2.0 / 3.0, nu) / d**5
        )

    return 9.0 * math.sqrt(3.0) / (4.0 * math.pi) * _integral(integrand)


def _normalized_niel_h(chi: float) -> float:
    """Niel's ``h_N`` over its classical limit ``(55/(16√3)) χ³``."""
    return _niel_h(chi) / (55.0 / (16.0 * math.sqrt(3.0)) * chi**3)


@pytest.fixture(scope="module")
def tables() -> RadiationReactionTables:
    return RadiationReactionTables(maximum_chi=5.0)


def _plan(model: RadiationReactionModel, **kwargs: Any) -> RadiationReactionPlan:
    options: dict[str, Any] = {"maximum_chi": 1.0e-2, "minimum_gamma": 1.0}
    options.update(kwargs)
    scale = options.pop("scale", _CLASSICAL)
    return RadiationReactionPlan(model, scale, _CHARGE, _MASS, **options)


def _planar_state(gamma: np.ndarray, angle: np.ndarray) -> jax.Array:
    speed = np.sqrt(gamma**2 - 1.0)
    return jnp.asarray(
        np.stack((speed * np.cos(angle), speed * np.sin(angle), 0.0 * speed), axis=-1)
    )


def _uniform(count: int, value: tuple[float, float, float]) -> jax.Array:
    return jnp.broadcast_to(jnp.asarray(value), (count, 3))


# -- quantum corrections --------------------------------------------------------


@pytest.mark.parametrize("chi", [1.0e-3, 0.1, 1.0, 4.0], ids=lambda v: f"chi={v:g}")
def test_tables_reproduce_niel_quantum_corrections(
    tables: RadiationReactionTables, chi: float
) -> None:
    np.testing.assert_allclose(tables.power_correction(chi), _niel_g(chi), rtol=2e-6)
    np.testing.assert_allclose(
        tables.diffusion_correction(chi), _normalized_niel_h(chi), rtol=2e-6
    )


def test_tables_tend_to_the_classical_limit(tables: RadiationReactionTables) -> None:
    chi = jnp.asarray([0.0, 1.0e-9, 1.0e-7])
    # g = 1 − (55√3/16) χ + O(χ²) (Sokolov–Ternov).
    np.testing.assert_allclose(
        tables.power_correction(chi),
        1.0 - 55.0 * math.sqrt(3.0) / 16.0 * chi,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tables.diffusion_correction(chi),
        [1.0, _normalized_niel_h(1.0e-9), _normalized_niel_h(1.0e-7)],
        rtol=1e-10,
    )
    assert tables.interpolation_error < 1e-6
    assert tables.quadrature_error < 1e-12


# -- classical Landau–Lifshitz ----------------------------------------------------


def _planar_cooling(plan: RadiationReactionPlan, steps: int, horizon: float) -> Any:
    pusher = RelativisticPushPlan(_CLASSICAL.relativity, method="boris")
    dt = horizon / steps
    electric = jnp.zeros((1, 3))
    magnetic = jnp.asarray([[0.0, 0.0, 1.0]])
    active = jnp.asarray([True])

    def body(u: jax.Array, _: None) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        pushed = pusher.push(u, electric, magnetic, jnp.asarray([-1.0]), active, dt)
        result = plan.apply(pushed.proper_velocity, electric, magnetic, dt, active)
        return result.proper_velocity, (result.radiated_energy[0], result.successful)

    initial = _planar_state(np.asarray([50.0]), np.asarray([0.0]))
    final, (radiated, successful) = jax.lax.scan(body, initial, length=steps)
    return initial, final, radiated, successful


def test_planar_landau_lifshitz_cooling_converges_first_order_to_the_analytic_law() -> (
    None
):
    plan = _plan("landau-lifshitz-reduced")
    horizon = 2.0
    # u ⊥ B: du/dt = −τ ω_B² γ u, so γ(t) = coth(τ ω_B² t + arccoth γ₀), ω_B = 1.
    exact = 1.0 / math.tanh(_TAU * horizon + math.atanh(1.0 / 50.0))
    errors = []
    for steps in (50, 100, 200, 400):
        _, final, _, successful = _planar_cooling(plan, steps, horizon)
        assert bool(jnp.all(successful))
        gamma = float(jnp.sqrt(1.0 + jnp.sum(final**2)))
        errors.append(abs(gamma - exact))
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    np.testing.assert_allclose(ratios, 2.0, rtol=0.05)
    assert errors[-1] / exact < 1e-3


def test_radiated_energy_equals_the_kinetic_energy_removed() -> None:
    plan = _plan("landau-lifshitz-reduced")
    initial, final, radiated, _ = _planar_cooling(plan, 200, 2.0)
    kinetic_loss = _MASS * (
        jnp.sqrt(1.0 + jnp.sum(initial**2)) - jnp.sqrt(1.0 + jnp.sum(final**2))
    )
    assert bool(jnp.all(radiated > 0.0))
    np.testing.assert_allclose(jnp.sum(radiated), kinetic_loss, rtol=1e-12)


def _random_kinematics(count: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    keys = jr.split(jr.key(7), 3)
    return (
        20.0 * jr.normal(keys[0], (count, 3), dtype=jnp.float64),
        0.3 * jr.normal(keys[1], (count, 3), dtype=jnp.float64),
        jr.normal(keys[2], (count, 3), dtype=jnp.float64),
    )


def test_full_and_reduced_landau_lifshitz_agree_in_uniform_static_fields() -> None:
    proper, electric, magnetic = _random_kinematics(16)
    active = jnp.ones((16,), dtype=bool)
    dt = 1.0e-3
    reduced = _plan("landau-lifshitz-reduced").apply(
        proper, electric, magnetic, dt, active
    )
    zero_gradient = jnp.zeros((16, 3, 3))
    zero_rate = jnp.zeros((16, 3))
    full = _plan("landau-lifshitz").apply(
        proper,
        electric,
        magnetic,
        dt,
        active,
        electric_gradient=zero_gradient,
        magnetic_gradient=zero_gradient,
        electric_rate=zero_rate,
        magnetic_rate=zero_rate,
    )
    assert bool(reduced.successful) and bool(full.successful)
    np.testing.assert_array_equal(full.proper_velocity, reduced.proper_velocity)
    np.testing.assert_array_equal(full.radiated_energy, reduced.radiated_energy)


def test_full_landau_lifshitz_adds_the_convective_field_derivative_term() -> None:
    proper, electric, magnetic = _random_kinematics(8)
    keys = jr.split(jr.key(11), 4)
    gradient_e = jr.normal(keys[0], (8, 3, 3), dtype=jnp.float64)
    gradient_b = jr.normal(keys[1], (8, 3, 3), dtype=jnp.float64)
    rate_e = jr.normal(keys[2], (8, 3), dtype=jnp.float64)
    rate_b = jr.normal(keys[3], (8, 3), dtype=jnp.float64)
    active = jnp.ones((8,), dtype=bool)
    dt = 1.0e-4
    reduced = _plan("landau-lifshitz-reduced").apply(
        proper, electric, magnetic, dt, active
    )
    full = _plan("landau-lifshitz").apply(
        proper,
        electric,
        magnetic,
        dt,
        active,
        electric_gradient=gradient_e,
        magnetic_gradient=gradient_b,
        electric_rate=rate_e,
        magnetic_rate=rate_b,
    )
    u = np.asarray(proper)
    gamma = np.sqrt(1.0 + np.sum(u**2, axis=-1))
    v = u / gamma[:, None]
    # LL §76: τ q γ [(∂_t + v·∇)E + v × (∂_t + v·∇)B].
    convective_e = np.asarray(rate_e) + np.einsum("nij,nj->ni", gradient_e, v)
    convective_b = np.asarray(rate_b) + np.einsum("nij,nj->ni", gradient_b, v)
    term = _TAU * _CHARGE * gamma[:, None] * (convective_e + np.cross(v, convective_b))
    np.testing.assert_allclose(
        np.asarray(full.proper_velocity) - np.asarray(reduced.proper_velocity),
        dt * term / _MASS,
        rtol=1e-8,
        atol=1e-15,
    )


# -- quantum-corrected Landau–Lifshitz and Fokker–Planck -------------------------


def _quantum_scale(chi: float, gamma: float) -> ElectromagneticScaleContract:
    """Scale whose ħ gives quantum parameter ``chi`` at ``gamma`` in ``B = 1``."""
    # χ = γ β |q| B ħ / (m² c²) for u ⊥ B.
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    hbar = chi * _MASS**2 / (gamma * beta * abs(_CHARGE))
    return _scale(Fraction(hbar).limit_denominator(10**18))


def test_quantum_corrected_force_is_g_times_classical_and_classical_as_chi_vanishes(
    tables: RadiationReactionTables,
) -> None:
    gamma = np.asarray([200.0, 400.0, 800.0])
    proper = _planar_state(gamma, np.asarray([0.1, 1.2, 2.3]))
    electric = jnp.zeros((3, 3))
    magnetic = _uniform(3, (0.0, 0.0, 1.0))
    active = jnp.ones((3,), dtype=bool)
    for chi, scale in (
        (0.5, _quantum_scale(0.5, 400.0)),
        (1.0e-9, _quantum_scale(1.0e-9, 400.0)),
    ):
        options = {"scale": scale, "maximum_chi": 5.0}
        classical = _plan("landau-lifshitz-reduced", **options).apply(
            proper, electric, magnetic, 1.0e-6, active
        )
        quantum = _plan(
            "quantum-corrected-landau-lifshitz", tables=tables, **options
        ).apply(proper, electric, magnetic, 1.0e-6, active)
        chis = np.asarray(quantum.quantum_parameter)
        np.testing.assert_allclose(chis[1], chi, rtol=1e-6)
        np.testing.assert_allclose(
            quantum.drift_rate,
            np.asarray([_niel_g(value) for value in chis]) * classical.drift_rate,
            rtol=2e-6,
        )
    # χ ≈ 1e-9: the quantum model is the classical one to O(χ).
    np.testing.assert_allclose(
        quantum.proper_velocity, classical.proper_velocity, rtol=1e-12
    )


def test_fokker_planck_drift_and_diffusion_match_niel_coefficients(
    tables: RadiationReactionTables,
) -> None:
    scale = _quantum_scale(0.5, 1000.0)
    plan = _plan("stochastic-fokker-planck", scale=scale, tables=tables, maximum_chi=5.0)
    gamma = np.asarray([600.0, 1000.0, 1800.0])
    proper = _planar_state(gamma, np.asarray([0.0, 1.0, 2.0]))
    result = plan.apply(
        proper,
        jnp.zeros((3, 3)),
        _uniform(3, (0.0, 0.0, 1.0)),
        1.0e-7,
        jnp.ones((3,), dtype=bool),
        wiener=jnp.zeros((3,)),
    )
    chi = np.asarray(result.quantum_parameter)
    hbar = float(scale.reduced_planck_constant)
    alpha = _CHARGE**2 / (4.0 * math.pi * hbar)
    rate = (2.0 / 3.0) * alpha * _MASS / hbar
    np.testing.assert_allclose(
        result.drift_rate,
        [-rate * value**2 * _niel_g(value) for value in chi],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        result.diffusion,
        [rate * g * _niel_h(value) for g, value in zip(gamma, chi, strict=True)],
        rtol=2e-6,
    )


def test_fokker_planck_ensemble_obeys_the_mean_and_variance_equations(
    tables: RadiationReactionTables,
) -> None:
    scale = _quantum_scale(0.5, 1000.0)
    plan = _plan("stochastic-fokker-planck", scale=scale, tables=tables, maximum_chi=5.0)
    count = 100_000
    rng = np.random.default_rng(3)
    gamma = rng.uniform(900.0, 1100.0, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    increments = rng.standard_normal(count)
    # Antithetic pairs cancel the odd noise moments of one Euler–Maruyama step.
    gamma = np.concatenate((gamma, gamma))
    proper = _planar_state(gamma, np.concatenate((angle, angle)))
    wiener = jnp.asarray(np.concatenate((increments, -increments)))
    dt = 1.0e-4
    result = plan.apply(
        proper,
        jnp.zeros((2 * count, 3)),
        _uniform(2 * count, (0.0, 0.0, 1.0)),
        dt,
        jnp.ones((2 * count,), dtype=bool),
        wiener=wiener,
    )
    assert bool(result.successful)
    after = np.sqrt(1.0 + np.sum(np.asarray(result.proper_velocity) ** 2, axis=-1))
    drift = np.asarray(result.drift_rate)
    diffusion = np.asarray(result.diffusion)
    assert -np.mean(drift) * dt / np.mean(gamma) > 1e-5
    # d⟨γ⟩/dt = ⟨A⟩ and dσ²/dt = 2⟨(γ − ⟨γ⟩)A⟩ + ⟨B⟩ (Niel et al. 2018).
    np.testing.assert_allclose(
        (np.mean(after) - np.mean(gamma)) / dt, np.mean(drift), rtol=1e-3
    )
    covariance = np.mean((gamma - np.mean(gamma)) * drift)
    np.testing.assert_allclose(
        (np.var(after) - np.var(gamma)) / dt,
        2.0 * covariance + np.mean(diffusion),
        rtol=2e-2,
    )


def test_wiener_increments_follow_particle_identity_not_storage_slot(
    tables: RadiationReactionTables,
) -> None:
    plan = _plan(
        "stochastic-fokker-planck",
        scale=_quantum_scale(0.5, 1000.0),
        tables=tables,
        maximum_chi=5.0,
    )
    high = jnp.asarray([0, 0, 1, 7], dtype=jnp.uint32)
    low = jnp.asarray([0, 1, 0, 3], dtype=jnp.uint32)
    key = jr.key(5)
    draws = plan.wiener_increments(key, high, low)
    order = jnp.asarray([2, 0, 3, 1])
    np.testing.assert_array_equal(
        plan.wiener_increments(key, high[order], low[order]), draws[order]
    )
    assert len(set(np.asarray(draws).tolist())) == 4
    assert not np.allclose(plan.wiener_increments(jr.key(6), high, low), draws)


# -- support ----------------------------------------------------------------------


def test_support_flags_refuse_out_of_domain_steps_and_exempt_slow_particles() -> None:
    plan = _plan("landau-lifshitz-reduced", minimum_gamma=5.0, maximum_chi=1.0e-4)
    gamma = np.asarray([2.0, 50.0, 2000.0])
    proper = _planar_state(gamma, np.zeros(3))
    magnetic = _uniform(3, (0.0, 0.0, 1.0))
    result = plan.apply(
        proper,
        jnp.zeros((3, 3)),
        magnetic,
        1.0e-3,
        jnp.ones((3,), dtype=bool),
        grid_cutoff_frequency=20.0,
    )
    flags = [RadiationReactionFlag(int(value)) for value in result.flags]
    # γ = 2 is exempt: unchanged, no radiation, still supported.
    assert flags[0] == RadiationReactionFlag.BELOW_MINIMUM_GAMMA
    np.testing.assert_array_equal(result.proper_velocity[0], proper[0])
    assert float(result.radiated_energy[0]) == 0.0
    # γ = 50: ω_c = (3/2) γ² β ω_B ≈ 3750 is well above the grid cutoff 20.
    assert flags[1] == RadiationReactionFlag.NONE
    np.testing.assert_allclose(
        result.scale_separation[1], 1.5 * 50.0**2 * 0.9998 / 20.0, rtol=1e-3
    )
    # γ = 2000 exceeds the declared χ bound.
    assert RadiationReactionFlag.CHI_EXCEEDED in flags[2]
    np.testing.assert_array_equal(result.supported, [True, True, False])
    assert not bool(result.successful)

    coarse = _plan("landau-lifshitz-reduced", maximum_chi=1.0).apply(
        proper[1:2], jnp.zeros((1, 3)), magnetic[1:2], 1.0, jnp.asarray([True])
    )
    assert RadiationReactionFlag.STEP_LOSS_EXCEEDED in RadiationReactionFlag(
        int(coarse.flags[0])
    )
    unresolved = plan.apply(
        proper[1:2],
        jnp.zeros((1, 3)),
        magnetic[1:2],
        1.0e-3,
        jnp.asarray([True]),
        grid_cutoff_frequency=1000.0,
    )
    assert bool(unresolved.successful)
    assert not bool(unresolved.scale_separated[0])
    assert RadiationReactionFlag.SCALE_UNSEPARATED in RadiationReactionFlag(
        int(unresolved.flags[0])
    )


def test_model_inputs_and_tables_are_refused_outside_their_models(
    tables: RadiationReactionTables,
) -> None:
    with pytest.raises(ValueError, match="takes no tables"):
        _plan("landau-lifshitz-reduced", tables=tables)
    with pytest.raises(TypeError, match="requires tables"):
        _plan("quantum-corrected-landau-lifshitz")
    with pytest.raises(ValueError, match="cover maximum_chi"):
        _plan("stochastic-fokker-planck", tables=tables, maximum_chi=10.0)
    with pytest.raises(ValueError):
        _plan("landau-lifshitz-sokolov")  # ty: ignore[invalid-argument-type]
    proper, electric, magnetic = _random_kinematics(2)
    active = jnp.ones((2,), dtype=bool)
    with pytest.raises(ValueError, match="landau-lifshitz"):
        _plan("landau-lifshitz").apply(proper, electric, magnetic, 1e-3, active)
    with pytest.raises(ValueError, match="stochastic-fokker-planck"):
        _plan("stochastic-fokker-planck", tables=tables).apply(
            proper, electric, magnetic, 1e-3, active
        )


# -- PIC integration ----------------------------------------------------------------


def _species(sign: float, name: str, offset: int) -> PICSpeciesPlan:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + 2), jnp.ones((2,)), ambient_dimension=1
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


def _pic(process: RadiationReactionProcess, **kwargs: Any) -> Any:
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(16, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    solver = phx.solver.ReducedMaxwellPICFieldSolver(
        phx.solver.CompatibleMaxwell1DPlan(grid), D.pic.ReducedPICTransferPlan(grid)
    )
    options: dict[str, Any] = {"ownership": "subgrid-reaction"}
    options.update(kwargs)
    return phx.solver.ElectromagneticPICPlan(
        solver,
        species=(_species(-1.0, "electrons", 0), _species(1.0, "ions", 10)),
        processes=(process,),
        **options,
    )


def _pic_state(pic: Any, gamma: float, dt: float) -> Any:
    # Macroparticles of 1e-5 physical particles keep self-fields negligible.
    position = jnp.asarray([[0.2], [0.7]])
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    weight = jnp.full((2,), 1.0e-5 * _MASS)
    zero = jnp.zeros((16,))
    return pic.initialize(
        (position, position),
        (jnp.zeros((2, 3)).at[:, 0].set(beta), jnp.zeros((2, 3))),
        dt,
        masses=(weight, weight),
        magnetic=(zero, zero, zero + 1.0),
    )


def test_pic_radiation_reaction_ledger_closes_and_follows_planar_cooling() -> None:
    plan = _plan("landau-lifshitz-reduced")
    pic = _pic(RadiationReactionProcess(plan, 0))
    dt = 0.01
    state = _pic_state(pic, 50.0, dt)
    kinetic = [float(pic._kinetic(state.species))]
    radiated = 0.0
    defect = 0.0
    for _ in range(40):
        result = pic.step_detailed(state, dt)
        assert bool(result.successful)
        (ledger,) = result.diagnostics.processes
        assert ledger.radiation is not None
        assert float(ledger.radiation.minimum_scale_separation) > 10.0
        energy = result.diagnostics.energy
        radiated += float(energy.radiated)
        defect += float(energy.defect)
        state = result.accepted_state
    kinetic.append(float(pic._kinetic(state.species)))
    loss = kinetic[0] - kinetic[1]
    assert loss > 0.0
    np.testing.assert_allclose(radiated, loss, rtol=1e-6)
    assert abs(defect) < 1e-6 * radiated
    proper = np.asarray(state.species[0].particles.proper_velocity)
    gamma = np.sqrt(1.0 + np.sum(proper**2, axis=-1))
    # The half-step bootstrap keeps γ₀, so 40 reaction steps span 40 Δt.
    exact = 1.0 / math.tanh(_TAU * 40 * dt + math.atanh(1.0 / 50.0))
    np.testing.assert_allclose(gamma, exact, rtol=5e-3)


def test_pic_rejects_steps_whose_emission_the_grid_resolves() -> None:
    pic = _pic(RadiationReactionProcess(_plan("landau-lifshitz-reduced"), 0))
    dt = 0.01
    state = _pic_state(pic, 1.5, dt)
    result = pic.step_detailed(state, dt)
    assert not bool(result.successful)
    reason = PICRejectionReason(int(result.diagnostics.rejection_reason))
    assert reason == PICRejectionReason.RADIATION_OWNERSHIP
    (ledger,) = result.diagnostics.processes
    assert ledger.radiation is not None
    assert not bool(ledger.radiation.scale_separated)
    for left, right in zip(
        jax.tree.leaves(result.accepted_state), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_pic_refuses_mismatched_species_units_and_ownership() -> None:
    process = RadiationReactionProcess(_plan("landau-lifshitz-reduced"), 0)
    with pytest.raises(ValueError, match="overlaps"):
        _pic(process, ownership="resolved-field")
    with pytest.raises(ValueError, match="charge-to-mass"):
        _pic(RadiationReactionProcess(_plan("landau-lifshitz-reduced"), 1))
    fast = RadiationReactionPlan(
        "landau-lifshitz-reduced",
        _scale(Fraction(1, 10**8), speed_of_light=2),
        _CHARGE,
        _MASS,
        maximum_chi=1.0e-2,
        minimum_gamma=1.0,
    )
    with pytest.raises(ValueError, match="speed of light"):
        _pic(RadiationReactionProcess(fast, 0))


def test_pic_full_landau_lifshitz_keeps_a_restartable_field_history() -> None:
    reduced = _pic(RadiationReactionProcess(_plan("landau-lifshitz-reduced"), 0))
    full = _pic(RadiationReactionProcess(_plan("landau-lifshitz"), 0))
    dt = 0.01
    reduced_state = _pic_state(reduced, 50.0, dt)
    full_state = _pic_state(full, 50.0, dt)
    assert reduced_state.field_history is None
    assert full_state.field_history is not None
    for _ in range(3):
        reduced_result = reduced.step_detailed(reduced_state, dt)
        full_result = full.step_detailed(full_state, dt)
        assert bool(full_result.successful)
        reduced_state = reduced_result.accepted_state
        full_state = full_result.accepted_state
    history = full_state.field_history
    assert history is not None
    np.testing.assert_allclose(history.time, full_state.time - dt)
    # The static magnetic field dominates; self-field derivatives are tiny.
    np.testing.assert_allclose(
        full_state.species[0].particles.proper_velocity,
        reduced_state.species[0].particles.proper_velocity,
        rtol=1e-9,
    )
    restored = full.restore(full.checkpoint(full_state))
    for left, right in zip(
        jax.tree.leaves(restored), jax.tree.leaves(full_state), strict=True
    ):
        np.testing.assert_array_equal(left, right)
