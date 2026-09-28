#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Storage-ring radiation integrals, equilibrium, and radiative tracking.

References: the weak-focusing combined-function ring with field index ``n``
has the constant periodic solution ``η = ρ/(1 − n)``, ``β_x = ρ/√(1 − n)`` and
tunes ``(√(1 − n), √n)``, hence ``I₁ = 2πρ/(1 − n)``, ``I₄ = 2π(1 − 2n)/(ρ(1 − n))``
and ``I₅ = 2π/(ρ(1 − n)^{3/2})`` (Sands, SLAC-121, §5); the isomagnetic FODO
integrals from an independent SciPy integration of the Hill, dispersion, and
Twiss differential equations; ``U₀ = C_γ E⁴ I₂/(2π)`` with
``C_γ = 4π r_e/(3 (mₑc²)³)``, ``C_q = 55 ƛ_c/(32√3)`` from SciPy CODATA 2022;
the Mellin moments ``∫F(ξ)/ξ dξ = 5π/3``, ``⟨ξ⟩ = 8/(15√3)``,
``⟨ξ²⟩ = 11/27`` of the synchrotron function; and the combined-function
alternating-gradient ring's horizontal anti-damping ``J_x < 0`` (CERN PS).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.stats import poisson

from phydrax import ElectromagneticScaleContract
from phydrax.applications import accelerator


SCALE = ElectromagneticScaleContract.si()
LIGHT = constants.c
CHARGE = constants.e
REST = constants.m_e * LIGHT**2
ELECTRON_RADIUS = constants.physical_constants["classical electron radius"][0]
REDUCED_COMPTON = constants.physical_constants["reduced Compton wavelength"][0]


def _momentum(gamma: float) -> float:
    return math.sqrt(gamma * gamma - 1.0) * REST / LIGHT


def _fodo(
    rho: float,
    cells: int,
    gradient: float,
    *,
    length_scale: float = 1.0,
    voltage: float | None = None,
    harmonic: int = 1,
) -> accelerator.RingLattice:
    angle = math.pi / cells
    quad = 0.3 * length_scale
    drift = 0.5 * length_scale
    elements = [
        accelerator.RingElement(
            "quadrupole", "qf1", length=0.5 * quad, gradient=gradient
        ),
        accelerator.RingElement("drift", "d1", length=drift),
        accelerator.RingElement(
            "sector-bend", "b1", length=rho * angle, curvature=1.0 / rho
        ),
        accelerator.RingElement("drift", "d2", length=drift),
        accelerator.RingElement("quadrupole", "qd", length=quad, gradient=-gradient),
        accelerator.RingElement("drift", "d3", length=drift),
        accelerator.RingElement(
            "sector-bend", "b2", length=rho * angle, curvature=1.0 / rho
        ),
        accelerator.RingElement("drift", "d4", length=drift),
        accelerator.RingElement(
            "quadrupole", "qf2", length=0.5 * quad, gradient=gradient
        ),
    ]
    if voltage is not None:
        elements.append(
            accelerator.RingElement("rf-cavity", "rf", voltage=voltage, harmonic=harmonic)
        )
    return accelerator.RingLattice(elements, periodicity=cells)


def _plan(
    lattice: accelerator.RingLattice, gamma: float
) -> accelerator.RingRadiationPlan:
    return accelerator.RingRadiationPlan(
        lattice,
        SCALE,
        reference_rest_energy=REST,
        reference_momentum=_momentum(gamma),
        reference_charge=-CHARGE,
    )


def _bunch(
    coordinates: np.ndarray, identities: np.ndarray
) -> accelerator.AcceleratorBunch:
    count = coordinates.shape[0]
    return accelerator.AcceleratorBunch(
        jnp.asarray(coordinates, dtype=jnp.float64),
        jnp.ones((count,), dtype=jnp.float64),
        jnp.asarray(identities, dtype=jnp.int32),
        reference_rest_energy=REST,
        reference_momentum=_momentum(RAPID_GAMMA),
        reference_charge=-CHARGE,
        bunch_id="ring-bunch",
    )


# A 3 GeV isomagnetic FODO ring of 16 cells with 10 m bending radius.
RING_GAMMA = 3.0e9 * CHARGE / REST
RING_RHO = 10.0
RING_CELLS = 16
RING_GRADIENT = 1.2

# A compressed six-cell FODO ring (γ = 300) that loses 2 % of its energy per
# turn, so tracking reaches the radiation equilibrium within a few hundred turns
# while emitting only about 20 photons per particle and turn.
RAPID_GAMMA = 300.0
RAPID_LOSS_FRACTION = 0.02
RAPID_RHO = (4.0 * math.pi / 3.0) * ELECTRON_RADIUS * RAPID_GAMMA**3 / RAPID_LOSS_FRACTION
RAPID_SCALE = RAPID_RHO / 10.0
RAPID_CELLS = 6


def _rapid_lattice(*, with_rf: bool = True) -> accelerator.RingLattice:
    loss = RAPID_LOSS_FRACTION * RAPID_GAMMA * REST
    return _fodo(
        RAPID_RHO,
        RAPID_CELLS,
        0.85 / RAPID_SCALE**2,
        length_scale=RAPID_SCALE,
        voltage=2.0 * loss / CHARGE / RAPID_CELLS if with_rf else None,
    )


@pytest.fixture(scope="module")
def rapid_plan() -> accelerator.RingRadiationPlan:
    return _plan(_rapid_lattice(), RAPID_GAMMA)


# --------------------------------------------------------------------------- #
# Independent Hill-equation reference.
# --------------------------------------------------------------------------- #


def _hill_reference(lattice: accelerator.RingLattice) -> tuple[np.ndarray, float, float]:
    """Radiation integrals, tunes by integrating the Hill and Twiss ODEs with SciPy."""

    def transfer(element: accelerator.RingElement) -> np.ndarray:
        strength = element.curvature**2 + element.gradient

        def rhs(_: float, state: np.ndarray) -> np.ndarray:
            matrix = state.reshape(3, 3)
            generator = np.asarray(
                [[0.0, 1.0, 0.0], [-strength, 0.0, element.curvature], [0.0, 0.0, 0.0]]
            )
            return (generator @ matrix).ravel()

        solution = solve_ivp(
            rhs, (0.0, element.length), np.eye(3).ravel(), rtol=1e-12, atol=1e-14
        )
        return solution.y[:, -1].reshape(3, 3)

    elements = [value for value in lattice.elements if value.length > 0.0]
    cell = np.eye(3)
    for element in elements:
        cell = transfer(element) @ cell
    block = cell[:2, :2]
    cosine = 0.5 * np.trace(block)
    sine = math.copysign(math.sqrt(1.0 - cosine**2), block[0, 1])
    beta = block[0, 1] / sine
    alpha = (block[0, 0] - block[1, 1]) / (2.0 * sine)
    eta = np.linalg.solve(np.eye(2) - block, cell[:2, 2])
    state = np.asarray([eta[0], eta[1], beta, alpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for element in elements:
        h = element.curvature
        k1 = element.gradient
        strength = h * h + k1

        def rhs(
            _: float, y: np.ndarray, h: float = h, k1: float = k1, k: float = strength
        ) -> np.ndarray:
            d, dp, b, a = y[:4]
            g = (1.0 + a * a) / b
            curly = g * d * d + 2.0 * a * d * dp + b * dp * dp
            return np.asarray(
                [
                    dp,
                    -k * d + h,
                    -2.0 * a,
                    k * b - g,
                    1.0 / b,
                    h * d,
                    h * h,
                    abs(h) ** 3,
                    h * d * (h * h + 2.0 * k1),
                    abs(h) ** 3 * curly,
                ]
            )

        solution = solve_ivp(rhs, (0.0, element.length), state, rtol=1e-12, atol=1e-16)
        state = solution.y[:, -1]
    integrals = lattice.periodicity * state[5:]
    return integrals, lattice.periodicity * state[4] / (2.0 * math.pi), float(eta[0])


# --------------------------------------------------------------------------- #
# Integrals and equilibrium.
# --------------------------------------------------------------------------- #


def test_weak_focusing_ring_matches_closed_form_integrals() -> None:
    rho = 12.0
    index = 0.5
    periods = 12
    gradient = -index / rho**2
    lattice = accelerator.RingLattice(
        [
            accelerator.RingElement(
                "sector-bend",
                "combined",
                length=2.0 * math.pi * rho / periods,
                curvature=1.0 / rho,
                gradient=gradient,
            )
        ],
        periodicity=periods,
    )
    plan = _plan(lattice, RING_GAMMA)
    integrals = plan.integrals
    optics = plan.optics
    expected = (
        2.0 * math.pi * rho / (1.0 - index),
        2.0 * math.pi / rho,
        2.0 * math.pi / rho**2,
        2.0 * math.pi * (1.0 - 2.0 * index) / (rho * (1.0 - index)),
        2.0 * math.pi / (rho * (1.0 - index) ** 1.5),
    )
    actual = (integrals.i1, integrals.i2, integrals.i3, integrals.i4, integrals.i5)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        float(optics.horizontal_tune), math.sqrt(1.0 - index), rtol=1e-12
    )
    np.testing.assert_allclose(float(optics.vertical_tune), math.sqrt(index), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(optics.dispersion), rho / (1.0 - index), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(optics.beta_x), rho / math.sqrt(1.0 - index), rtol=1e-12
    )
    np.testing.assert_allclose(
        float(optics.momentum_compaction), 1.0 / (1.0 - index), rtol=1e-12
    )
    partition = np.asarray(plan.equilibrium.damping_partition)
    damping = (1.0 - 2.0 * index) / (1.0 - index)
    np.testing.assert_allclose(partition, (1.0 - damping, 1.0, 2.0 + damping), rtol=1e-12)


def test_isomagnetic_fodo_integrals_match_hill_equation_reference() -> None:
    lattice = _fodo(RING_RHO, RING_CELLS, RING_GRADIENT)
    plan = _plan(lattice, RING_GAMMA)
    reference, tune, dispersion = _hill_reference(lattice)
    integrals = plan.integrals
    actual = np.asarray(
        (integrals.i1, integrals.i2, integrals.i3, integrals.i4, integrals.i5)
    )
    np.testing.assert_allclose(actual, reference, rtol=1e-8)
    # Isomagnetic identities hold exactly.
    np.testing.assert_allclose(actual[1], 2.0 * math.pi / RING_RHO, rtol=1e-14)
    np.testing.assert_allclose(actual[2], 2.0 * math.pi / RING_RHO**2, rtol=1e-14)
    np.testing.assert_allclose(float(plan.optics.horizontal_tune), tune, rtol=1e-9)
    np.testing.assert_allclose(float(plan.optics.dispersion[0]), dispersion, rtol=1e-9)
    assert float(jnp.max(integrals.quadrature_error)) < 1e-12 * float(
        jnp.max(jnp.abs(actual))
    )
    np.testing.assert_allclose(
        float(plan.optics.momentum_compaction),
        reference[0] / lattice.circumference,
        rtol=1e-8,
    )


def test_fodo_equilibrium_matches_published_radiation_constants() -> None:
    lattice = _fodo(RING_RHO, RING_CELLS, RING_GRADIENT)
    plan = _plan(lattice, RING_GAMMA)
    reference, _, _ = _hill_reference(lattice)
    i1, i2, i3, i4, i5 = reference
    energy = RING_GAMMA * REST
    radiation_constant = 4.0 * math.pi * ELECTRON_RADIUS / (3.0 * REST**3)
    loss = radiation_constant * energy**4 * i2 / (2.0 * math.pi)
    quantum_constant = 55.0 * REDUCED_COMPTON / (32.0 * math.sqrt(3.0))
    partition = np.asarray((1.0 - i4 / i2, 1.0, 2.0 + i4 / i2))
    beta = math.sqrt(1.0 - 1.0 / RING_GAMMA**2)
    period = lattice.circumference / (beta * LIGHT)
    equilibrium = plan.equilibrium
    np.testing.assert_allclose(float(equilibrium.energy_loss_per_turn), loss, rtol=1e-8)
    np.testing.assert_allclose(
        np.asarray(equilibrium.damping_partition), partition, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(equilibrium.damping_times),
        2.0 * energy * period / (partition * loss),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        float(equilibrium.energy_spread),
        math.sqrt(quantum_constant * RING_GAMMA**2 * i3 / (partition[2] * i2)),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        float(equilibrium.horizontal_emittance),
        quantum_constant * RING_GAMMA**2 * i5 / (partition[0] * i2),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        float(equilibrium.photons_per_turn),
        5.0 * constants.alpha * RING_GAMMA * 2.0 * math.pi / (2.0 * math.sqrt(3.0)),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        float(equilibrium.critical_energy),
        1.5 * constants.hbar * LIGHT * RING_GAMMA**3 / RING_RHO,
        rtol=1e-8,
    )
    assert i1 > 0.0


def test_photon_spectrum_moments_follow_synchrotron_function() -> None:
    spectrum = accelerator.SynchrotronPhotonSpectrum()
    mean = 8.0 / (15.0 * math.sqrt(3.0))
    second = 11.0 / 27.0
    assert spectrum.quadrature_photon_number == pytest.approx(
        5.0 * math.pi / 3.0, rel=1e-12
    )
    assert spectrum.quadrature_mean_fraction == pytest.approx(mean, rel=1e-12)
    assert spectrum.quadrature_second_moment == pytest.approx(second, rel=1e-12)
    assert spectrum.table_mean_fraction == pytest.approx(mean, rel=1e-6)
    assert spectrum.table_second_moment == pytest.approx(second, rel=1e-5)
    assert spectrum.tail_bound < 1e-20
    count = 400_000
    fractions = np.asarray(
        spectrum.sample_fractions((np.arange(count, dtype=np.float64) + 0.5) / count)
    )
    assert np.all(np.diff(fractions) >= 0.0)
    assert fractions.mean() == pytest.approx(mean, rel=2e-4)
    assert (fractions**2).mean() == pytest.approx(second, rel=2e-3)
    # The median of F(ξ)/ξ lies well below the critical energy.
    assert np.median(fractions) == pytest.approx(
        float(spectrum.sample_fractions(0.5)), rel=1e-4
    )


# --------------------------------------------------------------------------- #
# Tracking.
# --------------------------------------------------------------------------- #


def test_zero_radiation_tracking_reproduces_symplectic_one_turn_map() -> None:
    plan = _plan(_rapid_lattice(with_rf=False), RAPID_GAMMA)
    turns = 40
    tracking = accelerator.RadiativeRingTrackingPlan(
        plan, turns, model="none", horizontal_aperture=1.0, vertical_aperture=1.0
    )
    generator = np.random.default_rng(3)
    coordinates = generator.normal(size=(16, 6)) * np.asarray(
        [1e-7, 1e-3, 1e-7, 1e-3, 1e-7, 1e-3]
    )
    bunch = _bunch(coordinates, np.arange(16))
    radiative = accelerator.track_radiative_ring(tracking, bunch)
    symplectic = accelerator.track_ring(
        accelerator.RingTrackingPlan(
            plan.one_turn, turns, horizontal_aperture=1.0, vertical_aperture=1.0
        ),
        bunch,
    )
    np.testing.assert_allclose(
        np.asarray(radiative.bunch.coordinates),
        np.asarray(symplectic.bunch.coordinates),
        rtol=1e-9,
        atol=1e-18,
    )
    evidence = radiative.evidence
    assert int(evidence.status) == int(accelerator.RadiativeRingTrackingStatus.SUCCESS)
    assert np.all(np.asarray(evidence.radiated_energy) == 0.0)
    assert np.all(np.asarray(evidence.photon_count) == 0.0)


def test_classical_mean_loss_per_turn_is_u0() -> None:
    loss = (4.0 * math.pi / 3.0) * ELECTRON_RADIUS * RING_GAMMA**4 * REST / RING_RHO
    lattice = _fodo(
        RING_RHO, RING_CELLS, RING_GRADIENT, voltage=2.0 * loss / CHARGE / RING_CELLS
    )
    plan = accelerator.RingRadiationPlan(
        lattice,
        SCALE,
        reference_rest_energy=REST,
        reference_momentum=_momentum(RING_GAMMA),
        reference_charge=-CHARGE,
    )
    tracking = accelerator.RadiativeRingTrackingPlan(
        plan, 1, model="classical", horizontal_aperture=1.0, vertical_aperture=1.0
    )
    bunch = accelerator.AcceleratorBunch(
        jnp.zeros((1, 6)),
        jnp.ones((1,)),
        jnp.zeros((1,), dtype=jnp.int32),
        reference_rest_energy=REST,
        reference_momentum=_momentum(RING_GAMMA),
        reference_charge=-CHARGE,
        bunch_id="reference",
    )
    evidence = accelerator.track_radiative_ring(tracking, bunch).evidence
    # The reference loses U₀(1 + O(U₀/E)) while its momentum sags within the turn.
    fraction = loss / (RING_GAMMA * REST)
    np.testing.assert_allclose(
        float(plan.equilibrium.energy_loss_per_turn), loss, rtol=1e-10
    )
    np.testing.assert_allclose(
        float(evidence.radiated_energy[0]), loss, rtol=2.0 * fraction
    )
    np.testing.assert_allclose(float(evidence.rf_energy[0]), loss, rtol=2.0 * fraction)


def test_classical_tracking_damps_at_partitioned_rates(
    rapid_plan: accelerator.RingRadiationPlan,
) -> None:
    turns = 60
    tracking = accelerator.RadiativeRingTrackingPlan(
        rapid_plan,
        turns,
        model="classical",
        horizontal_aperture=1.0,
        vertical_aperture=1.0,
    )
    optics = rapid_plan.optics
    generator = np.random.default_rng(7)
    count = 512
    beta_x = float(optics.beta_x[0])
    beta_y = float(optics.beta_y[0])
    coordinates = np.zeros((count, 6))
    coordinates[:, 0] = generator.normal(size=count) * math.sqrt(4e-12 * beta_x)
    coordinates[:, 1] = generator.normal(size=count) * math.sqrt(4e-12 / beta_x)
    coordinates[:, 2] = generator.normal(size=count) * math.sqrt(1e-12 * beta_y)
    coordinates[:, 3] = generator.normal(size=count) * math.sqrt(1e-12 / beta_y)
    coordinates[:, 4] = generator.normal(size=count) * 6e-4 * RAPID_SCALE
    coordinates[:, 5] = generator.normal(size=count) * 1e-5
    bunch = _bunch(coordinates, np.arange(count))
    evidence = accelerator.track_radiative_ring(tracking, bunch).evidence
    covariance = np.asarray(evidence.covariance)
    initial = np.cov(coordinates, rowvar=False, bias=True)
    form = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    coupled = np.kron(np.eye(2), form)

    def emittances(sigma: np.ndarray) -> np.ndarray:
        # Williamson normal form: the eigenvalues of Σ J are ±i ε_k, invariant
        # under the linear symplectic motion; dispersion couples x with (z, δ).
        planar = sigma[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])]
        modes = np.sort(np.abs(np.linalg.eigvals(planar @ coupled).imag))[::-2]
        vertical = math.sqrt(np.linalg.det(sigma[2:4, 2:4]))
        return np.asarray([modes[0], vertical, modes[1]])

    ratio = emittances(covariance[-1]) / emittances(initial)
    expected = np.exp(-2.0 * turns / np.asarray(rapid_plan.equilibrium.damping_turns))
    np.testing.assert_allclose(ratio, expected, rtol=0.03)


def test_stochastic_tracking_converges_to_equilibrium(
    rapid_plan: accelerator.RingRadiationPlan,
) -> None:
    turns = 250
    tracking = accelerator.RadiativeRingTrackingPlan(
        rapid_plan,
        turns,
        model="stochastic",
        horizontal_aperture=1.0,
        vertical_aperture=1.0,
        slices_per_bend=1,
        photon_overflow_probability=1e-10,
    )
    count = 1000
    bunch = _bunch(np.zeros((count, 6)), np.arange(count))
    evidence = accelerator.track_radiative_ring(
        tracking, bunch, key=jax.random.key(2026)
    ).evidence
    equilibrium = rapid_plan.equilibrium
    assert bool(evidence.accepted)
    assert int(evidence.photon_overflow_count) == 0
    # The per-slice photon capacity is the smallest meeting the overflow bound.
    slice_mean = float(equilibrium.photons_per_turn) / (2 * RAPID_CELLS)
    capacity = tracking.photon_capacity
    assert tracking.overflow_probability == pytest.approx(
        poisson.sf(capacity, slice_mean), rel=1e-9
    )
    assert tracking.overflow_probability <= 1e-10 < poisson.sf(capacity - 1, slice_mean)
    damping_x, _, _ = np.asarray(equilibrium.damping_turns)
    window = np.arange(turns - 50, turns) + 1
    predicted = float(equilibrium.horizontal_emittance) * (
        1.0 - np.exp(-2.0 * window / damping_x)
    )
    tracked = np.asarray(evidence.horizontal_emittance)[-50:]
    assert tracked.mean() == pytest.approx(predicted.mean(), rel=0.12)
    spread = np.asarray(evidence.energy_spread)[-50:]
    assert spread.mean() == pytest.approx(float(equilibrium.energy_spread), rel=0.05)
    # Collinear emission leaves an uncoupled ring vertically cold.
    assert float(evidence.vertical_emittance[-1]) == 0.0
    loss = float(equilibrium.energy_loss_per_turn)
    assert np.asarray(evidence.radiated_energy)[-50:].mean() == pytest.approx(
        loss, rel=0.01
    )
    assert np.asarray(evidence.photon_count)[-50:].mean() == pytest.approx(
        float(equilibrium.photons_per_turn), rel=0.01
    )


def test_photon_sampling_follows_particle_identity(
    rapid_plan: accelerator.RingRadiationPlan,
) -> None:
    tracking = accelerator.RadiativeRingTrackingPlan(
        rapid_plan,
        4,
        model="stochastic",
        horizontal_aperture=1.0,
        vertical_aperture=1.0,
        slices_per_bend=1,
    )
    count = 24
    identities = np.arange(100, 100 + count)
    key = jax.random.key(11)
    full = accelerator.track_radiative_ring(
        tracking, _bunch(np.zeros((count, 6)), identities), key=key
    )
    order = np.random.default_rng(5).permutation(count)
    permuted = accelerator.track_radiative_ring(
        tracking, _bunch(np.zeros((count, 6)), identities[order]), key=key
    )
    subset = accelerator.track_radiative_ring(
        tracking, _bunch(np.zeros((8, 6)), identities[:8]), key=key
    )
    final = np.asarray(full.bunch.coordinates)
    np.testing.assert_array_equal(np.asarray(permuted.bunch.coordinates), final[order])
    np.testing.assert_array_equal(np.asarray(subset.bunch.coordinates), final[:8])
    assert np.unique(final[:, 5]).size == count
    other = accelerator.track_radiative_ring(
        tracking, _bunch(np.zeros((count, 6)), identities), key=jax.random.key(12)
    )
    assert not np.array_equal(np.asarray(other.bunch.coordinates), final)


# --------------------------------------------------------------------------- #
# Refusals.
# --------------------------------------------------------------------------- #


def _combined_function_alternating_gradient(gradient: float) -> accelerator.RingLattice:
    rho = 10.0
    periods = 20
    length = rho * math.pi / periods
    return accelerator.RingLattice(
        [
            accelerator.RingElement(
                "sector-bend", "bf", length=length, curvature=1.0 / rho, gradient=gradient
            ),
            accelerator.RingElement(
                "sector-bend",
                "bd",
                length=length,
                curvature=1.0 / rho,
                gradient=-gradient,
            ),
        ],
        periodicity=periods,
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda: _plan(_combined_function_alternating_gradient(0.5), RING_GAMMA),
            "no radiation-damped equilibrium",
            id="horizontal-antidamping",
        ),
        pytest.param(
            lambda: _plan(_fodo(RING_RHO, RING_CELLS, 40.0), RING_GAMMA),
            "unstable",
            id="unstable-optics",
        ),
        pytest.param(
            lambda: accelerator.RingLattice(
                [
                    accelerator.RingElement("drift", "d", length=1.0),
                    accelerator.RingElement("quadrupole", "q", length=0.2, gradient=1.0),
                ]
            ),
            "at least one sector bend",
            id="no-bends",
        ),
        pytest.param(
            lambda: accelerator.RingLattice(
                [accelerator.RingElement("sector-bend", "b", length=1.0, curvature=0.1)],
                periodicity=4,
            ),
            "do not close",
            id="open-ring",
        ),
        pytest.param(
            lambda: _plan(_fodo(RING_RHO, RING_CELLS, RING_GRADIENT), 20.0),
            "ultrarelativistic",
            id="sub-threshold-lorentz-factor",
        ),
        pytest.param(
            lambda: _plan(
                _fodo(1.0e-9, RING_CELLS, RING_GRADIENT * 1e20, length_scale=1e-10),
                RING_GAMMA,
            ),
            "Quantum parameter",
            id="quantum-regime",
        ),
        pytest.param(
            lambda: accelerator.RadiativeRingTrackingPlan(
                _plan(
                    _fodo(RING_RHO, RING_CELLS, RING_GRADIENT, voltage=10.0), RING_GAMMA
                ),
                1,
                model="classical",
                horizontal_aperture=1.0,
                vertical_aperture=1.0,
            ),
            "cannot restore",
            id="insufficient-rf",
        ),
        pytest.param(
            lambda: accelerator.RadiativeRingTrackingPlan(
                _plan(_fodo(RING_RHO, RING_CELLS, RING_GRADIENT), RING_GAMMA),
                1,
                model="stochastic",
                horizontal_aperture=1.0,
                vertical_aperture=1.0,
            ),
            "needs RF cavities",
            id="radiating-without-rf",
        ),
    ],
)
def test_unsupported_rings_are_refused(build: Callable[[], object], message: str) -> None:
    with pytest.raises(accelerator.RingOpticsError, match=message):
        build()


def test_tracking_boundary_refusals(rapid_plan: accelerator.RingRadiationPlan) -> None:
    stochastic = accelerator.RadiativeRingTrackingPlan(
        rapid_plan, 2, model="stochastic", horizontal_aperture=1.0, vertical_aperture=1.0
    )
    classical = accelerator.RadiativeRingTrackingPlan(
        rapid_plan, 2, model="classical", horizontal_aperture=1.0, vertical_aperture=1.0
    )
    bunch = _bunch(np.zeros((4, 6)), np.arange(4))
    with pytest.raises(ValueError, match="requires a key"):
        accelerator.track_radiative_ring(stochastic, bunch)
    with pytest.raises(ValueError, match="consumes no key"):
        accelerator.track_radiative_ring(classical, bunch, key=jax.random.key(0))
    with pytest.raises(ValueError, match="unique particle_ids"):
        accelerator.track_radiative_ring(classical, _bunch(np.zeros((4, 6)), np.zeros(4)))
    foreign = accelerator.AcceleratorBunch(
        jnp.zeros((4, 6)),
        jnp.ones((4,)),
        jnp.arange(4, dtype=jnp.int32),
        reference_rest_energy=REST,
        reference_momentum=_momentum(2.0 * RAPID_GAMMA),
        reference_charge=-CHARGE,
        bunch_id="foreign",
    )
    with pytest.raises(ValueError, match="reference particle differs"):
        accelerator.track_radiative_ring(classical, foreign)
    tight = accelerator.RadiativeRingTrackingPlan(
        rapid_plan,
        2,
        model="classical",
        horizontal_aperture=1.0,
        vertical_aperture=1.0,
        maximum_bytes=1024,
    )
    with pytest.raises(accelerator.RadiativeRingTrackingResourceError):
        accelerator.track_radiative_ring(tight, bunch)


def test_ring_elements_refuse_foreign_parameters() -> None:
    with pytest.raises(ValueError, match="'drift' contract"):
        accelerator.RingElement("drift", "d", length=1.0, curvature=0.1)
    with pytest.raises(ValueError, match="'rf-cavity' contract"):
        accelerator.RingElement("rf-cavity", "rf", length=0.5, voltage=1.0, harmonic=1)
    with pytest.raises(ValueError):
        accelerator.RingElement("sextupole", "s", length=0.2)  # ty: ignore[invalid-argument-type]
