#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strong-field QED cascades in electromagnetic PIC.

Code units: c = ε₀ = 1, time in 1/ω₀ and fields in m ω₀ c/e for a 1 μm laser,
so the Schwinger field is ``a_S = mc²/(ħω₀) = 4.1·10⁵`` and α = 1/137.036
(q = m = ħ a_S with ħ = 4πα/a_S²).

The growth-rate reference is the cascade model of Grismayer et al., Phys. Rev.
E 95, 023210 (2017), Eqs. (9)–(10): in a uniform rotating electric field of
amplitude ``a_r`` the pair number grows as ``e^{Γt}`` with ``Γ`` the positive
root of ``2∫₀¹ dη (d²P/dt dη) W_p(η)/(Γ + W_p(η)) = Γ``, evaluated at the
recoil-free cycle averages ``γ = 4a_r/π`` and ``χ = a_r²/a_S`` (Grismayer
et al., Phys. Plasmas 23, 056706, 2016); the photon-emission and pair rates are
evaluated here with SciPy's Bessel functions.
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
from scipy.integrate import quad, simpson
from scipy.optimize import brentq
from scipy.special import kv

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax._strict import StrictModule
from phydrax.discretization.pic import (
    ExternalFieldSample,
    NonlinearBreitWheelerPlan,
    NonlinearComptonPlan,
    PIC_CODE_RELATIVITY,
    PICChargeModelPlan,
    PICRejectionReason,
    PICSpeciesPlan,
    QEDCascadeProcess,
    QEDPhotonSpeciesPlan,
    QEDTable,
    RadiationReactionPlan,
    RadiationReactionProcess,
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
    constant_set_id="qed-cascade-test",
)
_ALPHA_FLOAT = float(_ALPHA)


class _RotatingField(StrictModule):
    """Uniform ``E = a_r (cos t, sin t, 0)``, ``B = 0``."""

    amplitude: float = eqx.field(static=True)

    @property
    def source_id(self) -> str:
        return f"rotating-electric-{self.amplitude!r}"

    def external_fields(self, positions: Any, times: Any, /) -> ExternalFieldSample:
        del positions
        electric = self.amplitude * jnp.stack(
            (jnp.cos(times), jnp.sin(times), jnp.zeros_like(times)), axis=-1
        )
        return ExternalFieldSample(
            electric, jnp.zeros_like(electric), jnp.ones(times.shape, dtype=bool)
        )


class _UniformMagnetic(StrictModule):
    """Uniform ``B = b ẑ``: the pusher does no work, QED alone changes energies."""

    strength: float = eqx.field(static=True)

    @property
    def source_id(self) -> str:
        return f"uniform-magnetic-{self.strength!r}"

    def external_fields(self, positions: Any, times: Any, /) -> ExternalFieldSample:
        magnetic = jnp.zeros((times.shape[0], 3)).at[:, 2].set(self.strength)
        return ExternalFieldSample(
            jnp.zeros_like(magnetic), magnetic, jnp.ones(times.shape, dtype=bool)
        )


@pytest.fixture(scope="module")
def tables() -> tuple[QEDTable, QEDTable]:
    return (
        QEDTable("nonlinear-compton", maximum_chi=100.0),
        QEDTable("nonlinear-breit-wheeler", maximum_chi=100.0),
    )


def _species(capacity: int, sign: float, name: str, offset: int) -> PICSpeciesPlan:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + capacity),
        jnp.ones((capacity,)),
        ambient_dimension=1,
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
    tables: tuple[QEDTable, QEDTable],
    photon_capacity: int,
    *,
    minimum_photon_energy: float = 2.0,
    escape: tuple[float, float] = (-1.0e9, 1.0e9),
) -> QEDCascadeProcess:
    compton = NonlinearComptonPlan(
        "lcfa",
        _SCALE,
        -_MASS,
        _MASS,
        tables[0],
        maximum_chi=100.0,
        minimum_gamma=1.0,
        maximum_event_probability=0.2,
    )
    pairs = NonlinearBreitWheelerPlan(_SCALE, _MASS, _MASS, tables[1], maximum_chi=100.0)
    photons = QEDPhotonSpeciesPlan(
        photon_capacity,
        1,
        escape_lower=(escape[0],),
        escape_upper=(escape[1],),
        energy_edges=tuple(np.geomspace(1.0, 1.0e5, 11) * _MASS),
    )
    return QEDCascadeProcess(
        compton,
        photons,
        emitters=(0, 1),
        breit_wheeler=pairs,
        electron=0,
        positron=1,
        gather_species=0,
        minimum_photon_energy=minimum_photon_energy * _MASS,
    )


def _run(
    process: Any,
    capacity: int,
    field: Any,
    *,
    key: int = 1,
    extra: tuple[Any, ...] = (),
) -> phx.solver.ElectromagneticPICPlan:
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(64, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [100.0]]))
    return phx.solver.ElectromagneticPICPlan(
        phx.solver.ReducedMaxwellPICFieldSolver(
            phx.solver.CompatibleMaxwell1DPlan(grid), D.pic.ReducedPICTransferPlan(grid)
        ),
        species=(
            _species(capacity, -1.0, "electrons", 0),
            _species(capacity, 1.0, "positrons", 100000),
        ),
        processes=(process, *extra),
        ownership="subgrid-reaction",
        external_fields=(field,),
        key=jax.random.key(key),
    )


def _seed(run: Any, capacity: int, seeds: int, dt: float, gamma: float = 1.0) -> Any:
    """Neutral electron–positron seeds at rest (or moving along ±x)."""
    position = jnp.zeros((capacity, 1)).at[:seeds, 0].set(jnp.linspace(30.0, 70.0, seeds))
    active = jnp.arange(capacity) < seeds
    speed = math.sqrt(1.0 - 1.0 / gamma**2)
    velocity = jnp.zeros((capacity, 3)).at[:seeds, 0].set(speed)
    mass = jnp.where(active, 1.0e-9 * _MASS, 0.0)
    return run.initialize(
        (position, position),
        (velocity, -velocity),
        dt,
        active_masks=(active, active),
        masses=(mass, mass),
    )


@eqx.filter_jit
def _step(run: Any, state: Any, dt: float) -> Any:
    return run.step_detailed(state, dt)


@eqx.filter_jit
def _advance(run: Any, state: Any, dt: float, steps: int) -> tuple[Any, Any]:
    def body(carry: Any, _: None) -> tuple[Any, tuple[Any, Any]]:
        result = run.step_detailed(carry, dt)
        return result.accepted_state, (
            result.successful,
            jnp.sum(result.accepted_state.species[0].population.active),
        )

    return jax.lax.scan(body, state, None, length=steps)


def _identity(population: Any) -> np.ndarray:
    return (np.asarray(population.id_hi).astype(np.uint64) << np.uint64(32)) | np.asarray(
        population.id_lo
    ).astype(np.uint64)


def _stores(state: Any) -> tuple[float, float]:
    """Independent kinetic energy of both species and photon store energy."""
    kinetic = 0.0
    for species in state.species:
        active = np.asarray(species.population.active)
        mass = np.asarray(species.population.mass)[active]
        u = np.asarray(species.particles.proper_velocity)[active]
        kinetic += float(np.sum(mass * (np.sqrt(1.0 + np.sum(u**2, axis=-1)) - 1.0)))
    photons = state.processes[0].photons
    active = np.asarray(photons.population.active)
    bank = float(
        np.sum(
            np.asarray(photons.population.mass)[active]
            * np.linalg.norm(np.asarray(photons.momentum)[active], axis=-1)
        )
    )
    store = (
        bank
        + float(np.sum(photons.escaped_energy))
        + float(photons.escaped_unbinned_energy)
        + float(photons.untracked_energy)
    )
    return kinetic, store


def test_cascade_ledger_closes_with_field_exchange_and_pairs_deposit_in_their_step(
    tables: tuple[QEDTable, QEDTable],
) -> None:
    capacity, dt = 64, 0.005
    process = _cascade(tables, 256, escape=(35.0, 65.0))
    # γ = 4000 across B = 2000 m ω₀ c/e gives χ ≈ 20.
    run = _run(process, capacity, _UniformMagnetic(2000.0))
    state = _seed(run, capacity, 8, dt, gamma=4000.0)
    exchange = rest = 0.0
    pairs_seen = escaped = 0
    for _ in range(150):
        result = _step(run, state, dt)
        assert bool(result.successful)
        after = result.accepted_state
        kinetic_before, store_before = _stores(state)
        kinetic_after, store_after = _stores(after)
        energy = result.diagnostics.energy
        (ledger,) = result.diagnostics.processes
        (evidence,) = result.diagnostics.process_evidence
        radiation = ledger.radiation
        assert radiation is not None
        scale = kinetic_before + store_after
        # Independent stores: kinetic + photons + created rest − field supply.
        balance = (
            kinetic_after
            - kinetic_before
            + store_after
            - store_before
            + float(energy.created_rest_energy)
            - float(energy.field_exchange)
        )
        assert abs(balance) <= 1e-9 * scale
        np.testing.assert_allclose(
            float(energy.radiated),
            store_after - store_before,
            rtol=1e-9,
            atol=1e-12 * scale,
        )
        assert abs(float(energy.defect)) <= 1e-9 * scale
        assert abs(float(ledger.energy_defect)) <= 1e-9 * scale
        assert abs(float(ledger.charge_defect)) <= 1e-20
        # Pairs are created at their photon's step-start position and drift
        # within the creation step.
        created = int(evidence.decays)
        if created:
            photons = state.processes[0].photons
            # Retired slots keep their last identity; only live photons decay.
            photon_ids = np.where(
                np.asarray(photons.population.active),
                _identity(photons.population),
                np.uint64(2**64 - 1),
            )
            electrons = after.species[0]
            born = np.asarray(electrons.population.has_parent) & np.asarray(
                electrons.population.active
            )
            parents = (
                np.asarray(electrons.population.parent_hi).astype(np.uint64)
                << np.uint64(32)
            ) | np.asarray(electrons.population.parent_lo).astype(np.uint64)
            for slot in np.flatnonzero(born & np.isin(parents, photon_ids)):
                (source,) = np.flatnonzero(photon_ids == parents[slot])
                u = np.asarray(electrons.particles.proper_velocity)[slot]
                start = float(photons.position[source, 0])
                moved = float(electrons.particles.position[slot, 0]) - start
                np.testing.assert_allclose(
                    moved, dt * u[0] / math.sqrt(1.0 + u @ u), rtol=1e-9
                )
        pairs_seen += created
        escaped += int(evidence.escaped)
        exchange += float(energy.field_exchange)
        rest += float(energy.created_rest_energy)
        state = after
    assert pairs_seen > 0 and escaped > 0
    # Pair rest energy 2mc² per physical pair; the collinear defect is positive.
    np.testing.assert_allclose(
        rest, 2.0 * _MASS * 1.0e-9 * pairs_seen, rtol=1e-12
    )
    assert exchange > 0.0
    histogram = state.processes[0].photons.escaped_number
    assert float(jnp.sum(histogram)) + float(
        state.processes[0].photons.escaped_unbinned_number
    ) == pytest.approx(escaped * 1.0e-9, rel=1e-12)


def _permuted(state: Any, species: int, order: np.ndarray) -> Any:
    capacity = order.shape[0]
    values = list(state.species)
    values[species] = jax.tree.map(
        lambda leaf: leaf[order] if leaf.ndim and leaf.shape[0] == capacity else leaf,
        values[species],
    )
    return eqx.tree_at(lambda value: value.species, state, tuple(values))


def _by_identity(population: Any, *arrays: Any) -> tuple[np.ndarray, ...]:
    active = np.asarray(population.active)
    ids = _identity(population)[active]
    order = np.argsort(ids)
    return (ids[order],) + tuple(np.asarray(value)[active][order] for value in arrays)


def test_cascade_is_invariant_to_storage_slot_order(
    tables: tuple[QEDTable, QEDTable],
) -> None:
    dt, steps = 0.005, 200

    def evolve(capacity: int, photon_capacity: int, order: np.ndarray | None) -> Any:
        run = _run(_cascade(tables, photon_capacity), capacity, _UniformMagnetic(2000.0))
        state = _seed(run, capacity, 6, dt, gamma=4000.0)
        if order is not None:
            state = _permuted(_permuted(state, 0, order), 1, order)
        state, (successful, _) = _advance(run, state, dt, steps)
        assert bool(jnp.all(successful))
        return state

    reference = evolve(32, 256, None)
    permuted = evolve(32, 256, np.random.default_rng(2).permutation(32))
    decays = int(jnp.sum(reference.species[0].population.has_parent))
    assert decays > 0
    for other in (permuted,):
        for index in (0, 1):
            left = reference.species[index]
            right = other.species[index]
            expected = _by_identity(
                left.population, left.particles.position, left.particles.proper_velocity
            )
            actual = _by_identity(
                right.population,
                right.particles.position,
                right.particles.proper_velocity,
            )
            np.testing.assert_array_equal(actual[0], expected[0])
            for got, want in zip(actual[1:], expected[1:], strict=True):
                np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
        left = reference.processes[0].photons
        right = other.processes[0].photons
        expected = _by_identity(left.population, left.position, left.momentum)
        actual = _by_identity(right.population, right.position, right.momentum)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(actual[2], expected[2], rtol=1e-9)


def test_photon_capacity_refusal_rejects_the_whole_step(
    tables: tuple[QEDTable, QEDTable],
) -> None:
    capacity, dt = 16, 0.005
    run = _run(_cascade(tables, 16), capacity, _UniformMagnetic(2000.0))
    state = _seed(run, capacity, 8, dt, gamma=4000.0)
    for _ in range(400):
        result = _step(run, state, dt)
        if not bool(result.successful):
            break
        state = result.accepted_state
    else:
        pytest.fail("The photon bank never filled.")
    (evidence,) = result.diagnostics.process_evidence
    assert bool(evidence.photon_capacity_refused)
    assert int(result.diagnostics.rejection_reason) & PICRejectionReason.PROCESS
    for accepted, previous in zip(
        jax.tree.leaves(result.accepted_state), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(accepted, previous)


def test_restart_component_resumes_the_cascade_exactly(
    tables: tuple[QEDTable, QEDTable],
) -> None:
    capacity, dt = 32, 0.005
    process = _cascade(tables, 128)
    run = _run(process, capacity, _UniformMagnetic(2000.0))
    state, _ = _advance(run, _seed(run, capacity, 6, dt, gamma=4000.0), dt, 40)
    checkpoint = run.checkpoint(state)
    names = {value.name: value for value in checkpoint.components}
    assert names["process/0"].owner_id == process.process_id
    restored = run.restore(checkpoint)
    for resumed, original in zip(
        jax.tree.leaves(_advance(run, restored, dt, 30)[0]),
        jax.tree.leaves(_advance(run, state, dt, 30)[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(resumed, original)
    other = _run(
        _cascade(tables, 128, minimum_photon_energy=4.0),
        capacity,
        _UniformMagnetic(2000.0),
    )
    with pytest.raises(ValueError):
        other.restore(checkpoint)


def test_cascade_refuses_overlapping_ownership_and_incompatible_species(
    tables: tuple[QEDTable, QEDTable],
) -> None:
    reaction = RadiationReactionProcess(
        RadiationReactionPlan(
            "landau-lifshitz-reduced",
            _SCALE,
            -_MASS,
            _MASS,
            maximum_chi=1.0,
            minimum_gamma=1.0,
        ),
        0,
    )
    with pytest.raises(ValueError, match="exactly one claiming process"):
        _run(_cascade(tables, 64), 32, _UniformMagnetic(1.0), extra=(reaction,))
    with pytest.raises(ValueError, match="multiple of the gather species"):
        _run(_cascade(tables, 48), 32, _UniformMagnetic(1.0))
    swapped = QEDCascadeProcess(
        _cascade(tables, 64).compton,
        _cascade(tables, 64).photons,
        emitters=(0, 1),
        breit_wheeler=_cascade(tables, 64).breit_wheeler,
        electron=1,
        positron=0,
        gather_species=0,
    )
    with pytest.raises(ValueError, match="electron ratio"):
        _run(swapped, 32, _UniformMagnetic(1.0))


def _photon_spectrum(chi: float, eta: np.ndarray) -> np.ndarray:
    def tail(x: float) -> float:
        return quad(
            lambda t: float(kv(np.float64(1.0 / 3.0), np.float64(t))),
            x,
            np.inf,
            epsabs=0.0,
            epsrel=1e-10,
        )[0]

    delta = 2.0 * eta / (3.0 * chi * (1.0 - eta))
    return np.array(
        [
            (
                (1.0 - e + 1.0 / (1.0 - e))
                * float(kv(np.float64(2.0 / 3.0), np.float64(d)))
                - tail(d)
            )
            / (math.sqrt(3.0) * math.pi)
            for e, d in zip(eta, delta, strict=True)
        ]
    )


def _pair_rate(chi: float) -> float:
    def density(xi: float) -> float:
        delta = 2.0 / (3.0 * chi * xi * (1.0 - xi))
        integral = quad(
            lambda t: float(kv(np.float64(1.0 / 3.0), np.float64(t))),
            delta,
            np.inf,
            epsabs=0.0,
            epsrel=1e-10,
        )[0]
        return (
            (xi / (1.0 - xi) + (1.0 - xi) / xi)
            * float(kv(np.float64(2.0 / 3.0), np.float64(delta)))
            + integral
        ) / (math.sqrt(3.0) * math.pi)

    return 2.0 * quad(density, 1e-9, 0.5, epsabs=0.0, epsrel=1e-9, limit=200)[0]


def _grismayer_growth_rate(amplitude: float) -> float:
    gamma = 4.0 * amplitude / math.pi
    chi = amplitude**2 / _SCHWINGER
    scale = _ALPHA_FLOAT * _SCHWINGER
    eta = np.linspace(2.0 / gamma, 1.0 - 1e-6, 1201)
    emission = scale / gamma * _photon_spectrum(chi, eta)
    decay = np.array([scale / (value * gamma) * _pair_rate(value * chi) for value in eta])
    return brentq(
        lambda s: 2.0 * simpson(emission * decay / (s + decay), x=eta) - s, 1e-8, 10.0
    )


def test_rotating_field_cascade_grows_at_the_grismayer_rate(
    tables: tuple[QEDTable, QEDTable],
) -> None:
    amplitude, dt = 600.0, 0.05
    # Photons below 150 mc² have χ_γ < 0.2 here and practically never decay.
    process = _cascade(tables, 12288, minimum_photon_energy=150.0)
    run = _run(process, 1024, _RotatingField(amplitude), key=3)
    state = _seed(run, 1024, 16, dt)
    counts = []
    for _ in range(19):
        state, (successful, count) = _advance(run, state, dt, 100)
        assert bool(jnp.all(successful))
        counts.append(np.asarray(count))
    number = np.concatenate(counts).astype(np.float64)
    time = dt * np.arange(1, number.size + 1)
    late = time >= 40.0
    growth = np.polyfit(time[late], np.log(number[late]), 1)[0]
    assert number[-1] > 8 * number[0]
    np.testing.assert_allclose(growth, _grismayer_growth_rate(amplitude), rtol=0.25)
