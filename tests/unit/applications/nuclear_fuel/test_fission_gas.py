"""Per-site UO2 total fission gas model.

Independent reference: the documented atom ledger
``retained(t + dt) - retained(t) = beta dt - released`` and the reservoir
transfer balances published by ``FissionGasTransfers``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nuclear_fuel import (
    FissionGasModel,
    FissionGasSiteState,
    FissionGasStepResult,
)
from tests._support.nuclear_fuel_gas import (
    advance_gas,
    DAY,
    fission_gas_model,
    FISSION_RATE,
    PRODUCTION,
    XE_KR_YIELD,
)


pytestmark = pytest.mark.strict_jax


def test_compiled_exposure_conversion_preserves_zero_fission_cooldown() -> None:
    """Rounded exposure epochs must not create a negative shutdown increment."""
    model = fission_gas_model(mode_count=2)
    # Three independently recorded zero-fission cooldown sites. Their rounded
    # burnups equal the accepted values, but fused multiply/subtract yields
    # negative increments of order 1e-20 MWd/kgHM without the epoch boundary.
    energy = np.asarray(
        (22823679.818674423, 22831385.943654623, 22842248.81272396),
        dtype=np.float64,
    )
    accepted_burnup = energy / 8.64e10
    state = eqx.tree_at(
        lambda value: value.burnup_mwd_per_kghm,
        model.initial_state((3,), grain_radius=5e-6),
        jnp.asarray(accepted_burnup),
    )

    @eqx.filter_jit
    def cooldown(endpoint_energy: jax.Array) -> FissionGasStepResult:
        return advance_gas(
            model,
            state,
            temperature=600.0,
            previous_temperature=601.0,
            burnup=endpoint_energy / 8.64e10,
            time_step=0.234375,
            fission_rate=0.0,
            production=0.0,
        )

    result = cooldown(jnp.asarray(energy))
    np.testing.assert_array_equal(result.admissible, np.ones(3, dtype=np.bool_))
    np.testing.assert_array_equal(result.state.burnup_mwd_per_kghm, accepted_burnup)
    np.testing.assert_array_equal(
        result.state.effective_burnup_mwd_per_kghm, np.zeros(3, dtype=np.float64)
    )
    np.testing.assert_array_equal(result.state.produced, state.produced)
    np.testing.assert_array_equal(result.released_increment, np.zeros(3))
    np.testing.assert_array_equal(result.retained_total, np.zeros(3))
    # A genuinely decreasing endpoint remains refused, not clipped or forgiven.
    decreased = cooldown(jnp.asarray(energy - 864.0))
    np.testing.assert_array_equal(decreased.admissible, np.zeros(3, dtype=np.bool_))


def test_compiled_exposure_epoch_preserves_burnup_tangent() -> None:
    model = fission_gas_model(mode_count=2)
    energy = jnp.asarray(22823679.818674423, dtype=jnp.float64)
    state = eqx.tree_at(
        lambda value: value.burnup_mwd_per_kghm,
        model.initial_state((), grain_radius=5e-6),
        energy / 8.64e10,
    )

    @eqx.filter_jit
    def effective_burnup(endpoint_energy: jax.Array) -> jax.Array:
        return advance_gas(
            model,
            state,
            temperature=600.0,
            previous_temperature=601.0,
            burnup=endpoint_energy / 8.64e10,
            time_step=0.234375,
            fission_rate=0.0,
            production=0.0,
        ).state.effective_burnup_mwd_per_kghm

    value, tangent = jax.jvp(effective_burnup, (energy,), (jnp.ones_like(energy),))
    reverse = jax.grad(effective_burnup)(energy)
    np.testing.assert_array_equal(value, jnp.zeros_like(value))
    np.testing.assert_allclose(tangent, 1.0 / 8.64e10, rtol=1e-14)
    np.testing.assert_allclose(reverse, tangent, rtol=1e-14)


# Deterministic power cycle of three radial sites of one pellet: center,
# mid-radius and rim. The rim sees the self-shielding fission peak and stays
# below the HBS annealing threshold, so every reservoir and transfer channel
# of the model is exercised by accepted direct steps.
BASE_TEMPERATURE = np.asarray([1500.0, 1250.0, 900.0])  # [K]
RAMP_TEMPERATURE = np.asarray([2000.0, 1600.0, 1050.0])  # [K]
SHUTDOWN_TEMPERATURE = np.full(3, 600.0)  # [K]
RADIAL_FISSION_FACTOR = np.asarray([1.0, 1.1, 2.0])  # local over nominal fission rate
BASE_BURNUP_RATE = np.asarray([0.04, 0.05, 0.12]) / DAY  # [MWd/kgHM/s]
INITIAL_GRAIN_RADIUS = 5.0e-6  # [m]
FABRICATION_TEMPERATURE = 300.0  # [K]


@dataclass(frozen=True)
class PowerPhase:
    """Constant-temperature phase at a fraction of base power."""

    temperature: np.ndarray
    relative_power: float
    time_step: float
    steps: int


POWER_CYCLE = (
    PowerPhase(BASE_TEMPERATURE, 1.0, 5.0 * DAY, 20),  # base irradiation
    PowerPhase(RAMP_TEMPERATURE, 1.3, 0.1 * DAY, 1),  # ramp: heating burst
    PowerPhase(RAMP_TEMPERATURE, 1.3, 2.0 * DAY, 10),  # ramp hold
    PowerPhase(SHUTDOWN_TEMPERATURE, 0.0, 0.1 * DAY, 1),  # shutdown: cooling burst
    PowerPhase(BASE_TEMPERATURE, 1.0, 0.1 * DAY, 1),  # restart
    PowerPhase(BASE_TEMPERATURE, 1.0, 5.0 * DAY, 10),  # base irradiation
)


@dataclass(frozen=True)
class PowerCycleStep:
    """One direct public step: the accepted start, its result and its source."""

    before: FissionGasSiteState
    retained_before: np.ndarray
    result: FissionGasStepResult
    produced: np.ndarray


@pytest.fixture(scope="module")
def power_cycle() -> tuple[PowerCycleStep, ...]:
    model = fission_gas_model(mode_count=40)
    step = eqx.filter_jit(advance_gas)
    state = model.initial_state((3,), grain_radius=INITIAL_GRAIN_RADIUS)
    retained = np.zeros(3)
    previous_temperature = np.full(3, FABRICATION_TEMPERATURE)
    burnup = np.zeros(3)
    records: list[PowerCycleStep] = []
    for phase in POWER_CYCLE:
        fission_rate = phase.relative_power * FISSION_RATE * RADIAL_FISSION_FACTOR
        production = XE_KR_YIELD * fission_rate
        for _ in range(phase.steps):
            burnup = burnup + phase.relative_power * BASE_BURNUP_RATE * phase.time_step
            result = step(
                model,
                state,
                temperature=phase.temperature,
                previous_temperature=previous_temperature,
                burnup=burnup,
                time_step=phase.time_step,
                fission_rate=fission_rate,
                production=production,
            )
            records.append(
                PowerCycleStep(state, retained, result, production * phase.time_step)
            )
            state = result.state
            retained = np.asarray(result.retained_total)
            previous_temperature = phase.temperature
    return tuple(records)


def test_power_cycle_steps_are_admitted(
    power_cycle: tuple[PowerCycleStep, ...],
) -> None:
    rejected = [
        index
        for index, record in enumerate(power_cycle)
        if not bool(jnp.all(record.result.admissible))
    ]
    assert rejected == []


def test_power_cycle_exercises_every_transfer_channel(
    power_cycle: tuple[PowerCycleStep, ...],
) -> None:
    channels = (
        "intragranular_production",
        "hbs_production",
        "athermal_release",
        "grain_growth_sweeping",
        "hbs_depletion",
        "diffusion_to_boundary",
        "burst_release",
        "venting_release",
        "hbs_nucleation",
        "hbs_trapping",
        "hbs_resolution",
    )
    idle = [
        channel
        for channel in channels
        if not any(
            bool(jnp.any(getattr(record.result.transfers, channel) > 0.0))
            for record in power_cycle
        )
    ]
    assert idle == []


class Balance(NamedTuple):
    """One documented balance: reservoir change, net flow and rounding scale.

    ``scale`` bounds every inventory and gross flow entering the balance, so
    a closed balance differs from its flow only by rounding of that scale.
    """

    change: np.ndarray
    flow: np.ndarray
    scale: np.ndarray


def _site_scale(record: PowerCycleStep) -> np.ndarray:
    return np.maximum.reduce(
        [
            record.retained_before,
            np.asarray(record.result.retained_total),
            record.produced,
        ]
    )


def _gross(*flows: jax.Array) -> np.ndarray:
    return np.sum([np.abs(np.asarray(flow)) for flow in flows], axis=0)


def _site_atoms(record: PowerCycleStep) -> Balance:
    change = np.asarray(record.result.retained_total) - record.retained_before
    flow = record.produced - np.asarray(record.result.released_increment)
    return Balance(change, flow, _site_scale(record))


def _production_partition(record: PowerCycleStep) -> Balance:
    transfers = record.result.transfers
    split = (
        transfers.intragranular_production
        + transfers.hbs_production
        + transfers.athermal_release
    )
    return Balance(np.asarray(split), record.produced, record.produced)


def _release_partition(record: PowerCycleStep) -> Balance:
    transfers = record.result.transfers
    parts = (
        transfers.athermal_release,
        transfers.burst_release,
        transfers.venting_release,
    )
    return Balance(
        np.asarray(record.result.released_increment),
        np.asarray(sum(parts)),
        _gross(*parts),
    )


def _intragranular(state: FissionGasSiteState, retained: jax.Array) -> np.ndarray:
    return np.asarray(
        retained - state.boundary_gas - state.hbs_pore_gas - state.hbs_boundary_gas
    )


def _intragranular_reservoir(record: PowerCycleStep) -> Balance:
    # The grain inventory is the site total minus the explicit reservoirs, so
    # its rounding scale is that of the site ledger.
    result, transfers = record.result, record.result.transfers
    change = _intragranular(result.state, result.retained_total) - _intragranular(
        record.before, jnp.asarray(record.retained_before)
    )
    flow = (
        transfers.intragranular_production
        - transfers.grain_growth_sweeping
        - transfers.hbs_depletion
        - transfers.diffusion_to_boundary
    )
    return Balance(change, np.asarray(flow), _site_scale(record))


def _grain_face_reservoir(record: PowerCycleStep) -> Balance:
    transfers = record.result.transfers
    after, before = record.result.state.boundary_gas, record.before.boundary_gas
    flows = (
        transfers.grain_growth_sweeping,
        transfers.diffusion_to_boundary,
        -transfers.burst_release,
        -transfers.venting_release,
    )
    return Balance(
        np.asarray(after - before),
        np.asarray(sum(flows)),
        np.maximum.reduce([np.asarray(after), np.asarray(before), _gross(*flows)]),
    )


def _hbs_scale(record: PowerCycleStep) -> np.ndarray:
    after, before, transfers = record.result.state, record.before, record.result.transfers
    return np.maximum.reduce(
        [
            np.asarray(after.hbs_pore_gas + after.hbs_boundary_gas),
            np.asarray(before.hbs_pore_gas + before.hbs_boundary_gas),
            _gross(transfers.hbs_production, transfers.hbs_depletion),
            _gross(
                transfers.hbs_nucleation,
                transfers.hbs_trapping,
                transfers.hbs_resolution,
            ),
        ]
    )


def _hbs_storage(record: PowerCycleStep) -> Balance:
    after, before = record.result.state, record.before
    change = (
        after.hbs_pore_gas
        + after.hbs_boundary_gas
        - before.hbs_pore_gas
        - before.hbs_boundary_gas
    )
    transfers = record.result.transfers
    flow = transfers.hbs_production + transfers.hbs_depletion
    return Balance(np.asarray(change), np.asarray(flow), _hbs_scale(record))


def _hbs_pore_exchange(record: PowerCycleStep) -> Balance:
    # Pore/boundary exchange partitions the HBS storage, hence its scale.
    transfers = record.result.transfers
    change = record.result.state.hbs_pore_gas - record.before.hbs_pore_gas
    flow = transfers.hbs_nucleation + transfers.hbs_trapping - transfers.hbs_resolution
    return Balance(np.asarray(change), np.asarray(flow), _hbs_scale(record))


@pytest.mark.parametrize(
    "balance",
    [
        _site_atoms,
        _production_partition,
        _release_partition,
        _intragranular_reservoir,
        _grain_face_reservoir,
        _hbs_storage,
        _hbs_pore_exchange,
    ],
    ids=[
        "site-atom-ledger",
        "production-partition",
        "release-partition",
        "intragranular-reservoir",
        "grain-face-reservoir",
        "hbs-storage",
        "hbs-pore-exchange",
    ],
)
def test_power_cycle_balance_closes_every_step(
    power_cycle: tuple[PowerCycleStep, ...],
    balance: Callable[[PowerCycleStep], Balance],
) -> None:
    for index, record in enumerate(power_cycle):
        change, flow, scale = balance(record)
        mismatch = np.abs(change - flow)
        assert bool(np.all(mismatch <= 1.0e-12 * scale)), (index, mismatch, scale)


def test_power_cycle_retained_inventory_is_produced_minus_released(
    power_cycle: tuple[PowerCycleStep, ...],
) -> None:
    for index, record in enumerate(power_cycle):
        state = record.result.state
        np.testing.assert_allclose(
            np.asarray(record.result.retained_total),
            np.asarray(state.produced - state.released),
            rtol=1.0e-11,
            err_msg=f"step {index}",
        )


def test_admitted_power_cycle_inventories_are_nonnegative(
    power_cycle: tuple[PowerCycleStep, ...],
) -> None:
    for index, record in enumerate(power_cycle):
        result, state = record.result, record.result.state
        inventories = {
            "boundary_gas": state.boundary_gas,
            "hbs_pore_gas": state.hbs_pore_gas,
            "hbs_boundary_gas": state.hbs_boundary_gas,
            "bubble_gas_atoms": state.bubble_gas_atoms,
            "bubble_vacancies": state.bubble_vacancies,
            "hbs_pore_number_density": state.hbs_pore_number_density,
            "hbs_pore_variance": state.hbs_pore_variance,
            "hbs_pore_vacancies": state.hbs_pore_vacancies,
            "diffusion_to_boundary": result.transfers.diffusion_to_boundary,
            "venting_release": result.transfers.venting_release,
            "burst_release": result.transfers.burst_release,
            "intergranular_swelling": result.intergranular_swelling,
            "intragranular_swelling": result.intragranular_swelling,
            "hbs_swelling": result.hbs_swelling,
        }
        negative = [name for name, value in inventories.items() if jnp.any(value < 0.0)]
        assert negative == [], index


def test_admitted_power_cycle_coverage_stays_below_saturation(
    power_cycle: tuple[PowerCycleStep, ...],
) -> None:
    for index, record in enumerate(power_cycle):
        diagnostics = record.result.diagnostics
        assert bool(
            jnp.all(
                diagnostics.fractional_coverage
                <= diagnostics.saturation_coverage * (1.0 + 1.0e-12)
            )
        ), index


def test_admitted_power_cycle_grains_never_shrink(
    power_cycle: tuple[PowerCycleStep, ...],
) -> None:
    for index, record in enumerate(power_cycle):
        assert bool(
            jnp.all(record.result.state.grain_radius >= record.before.grain_radius)
        ), index


def _gas_loaded_state(
    model: FissionGasModel, sites: tuple[int, ...]
) -> FissionGasSiteState:
    state = model.initial_state(sites, grain_radius=5.0e-6)
    burnup = 0.0
    for _ in range(6):
        burnup += 0.5
        state = advance_gas(
            model,
            state,
            temperature=1500.0,
            previous_temperature=1500.0,
            burnup=burnup,
            time_step=10.0 * DAY,
        ).state
    return state


def test_advance_is_jit_and_vmap_consistent_over_sites() -> None:
    model = fission_gas_model(mode_count=40)
    state = _gas_loaded_state(model, (4,))
    temperature = jnp.asarray([900.0, 1300.0, 1700.0, 2000.0])
    previous = jnp.asarray([900.0, 1250.0, 1750.0, 1800.0])

    def step(site_state: FissionGasSiteState, t: jax.Array, tp: jax.Array) -> jax.Array:
        result = model.advance(
            site_state,
            temperature=t,
            grain_radius=model.grain_growth.advance_radius(
                site_state.grain_radius, t, 2.0 * DAY, burnup_mwd_per_kghm=4.0
            ),
            fission_rate_density=FISSION_RATE,
            gas_production_rate=PRODUCTION,
            burnup_mwd_per_kghm=4.0,
            hydrostatic_stress=-2.0e7,
            time_step=2.0 * DAY,
            previous_temperature=tp,
        )
        return jnp.stack(
            [
                result.released_increment,
                result.retained_total,
                result.intergranular_swelling,
                result.state.bubble_vacancies,
            ],
            axis=-1,
        )

    eager = step(state, temperature, previous)
    compiled = jax.jit(step)(state, temperature, previous)
    mapped = jax.vmap(step)(state, temperature, previous)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(mapped), np.asarray(eager), rtol=1.0e-12)


def test_step_is_differentiable_with_respect_to_temperature() -> None:
    model = fission_gas_model(mode_count=40)
    state = _gas_loaded_state(model, ())

    def outputs(temperature: jax.Array) -> jax.Array:
        result = model.advance(
            state,
            temperature=temperature,
            grain_radius=model.grain_growth.advance_radius(
                state.grain_radius, temperature, 2.0 * DAY, burnup_mwd_per_kghm=3.5
            ),
            fission_rate_density=FISSION_RATE,
            gas_production_rate=PRODUCTION,
            burnup_mwd_per_kghm=3.5,
            hydrostatic_stress=-2.0e7,
            time_step=2.0 * DAY,
            previous_temperature=1500.0,
        )
        return jnp.stack(
            [
                result.released_increment / PRODUCTION,
                result.intergranular_swelling,
                result.intragranular_swelling,
                result.state.grain_radius * 1.0e6,
            ]
        )

    jacobian = jax.jacfwd(outputs)(jnp.asarray(1650.0))
    gradient = jax.grad(lambda t: jnp.sum(outputs(t)))(jnp.asarray(1650.0))
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert bool(jnp.isfinite(gradient))
    np.testing.assert_allclose(float(gradient), float(jnp.sum(jacobian)), rtol=1.0e-8)
    # A hotter step releases more gas (thermal diffusion and micro-cracking).
    assert float(jacobian[0]) > 0.0


def test_model_refuses_mismatched_mode_count() -> None:
    state = fission_gas_model(mode_count=10).initial_state((2,), grain_radius=5.0e-6)
    with pytest.raises(ValueError, match="modes"):
        advance_gas(
            fission_gas_model(mode_count=12),
            state,
            temperature=1200.0,
            previous_temperature=1200.0,
            burnup=0.1,
            time_step=DAY,
        )


def test_unpowered_initial_state_has_finite_roots_and_zero_fgr() -> None:
    model = fission_gas_model(mode_count=40)
    result = advance_gas(
        model,
        model.initial_state((2,), grain_radius=5.0e-6),
        temperature=np.asarray([300.0, 1500.0]),
        previous_temperature=np.asarray([300.0, 1500.0]),
        burnup=0.0,
        time_step=600.0,
        fission_rate=0.0,
        production=0.0,
    )
    np.testing.assert_array_equal(np.asarray(result.retained_total), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(result.released_increment), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(result.cumulative_fgr), np.zeros(2))
    assert bool(jnp.all(result.admissible))
    assert bool(jnp.all(result.diagnostics.vacancy_root_converged))
    assert bool(jnp.all(result.diagnostics.hbs_root_converged))


def test_fabrication_diagnostics_report_unavailability_without_nan() -> None:
    # Fabrication state is a sensitivity input; NaN placeholders would make
    # every forward tangent NaN through ``NaN * 0`` products.
    diagnostics = fission_gas_model(mode_count=40).initial_diagnostics((3,))
    for leaf in jax.tree.leaves(diagnostics):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.all(jnp.isfinite(leaf)))
    assert not bool(jnp.any(diagnostics.hbs_root_converged))
    assert not bool(jnp.any(diagnostics.hbs_root_conditioning_available))
    np.testing.assert_array_equal(diagnostics.hbs_root_linear_status, np.full(3, -1))


@pytest.mark.parametrize(
    "time_step", [0.0, 0.1, 600.0], ids=["zero-time", "startup", "hold"]
)
def test_unpowered_site_jvp_skips_inactive_roots(time_step: float) -> None:
    model = fission_gas_model(mode_count=40)
    state = model.initial_state((), grain_radius=5.0e-6)

    def outputs(temperature: jax.Array) -> jax.Array:
        result = model.advance(
            state,
            temperature=temperature,
            grain_radius=model.grain_growth.advance_radius(
                state.grain_radius, temperature, time_step, burnup_mwd_per_kghm=0.0
            ),
            fission_rate_density=0.0,
            gas_production_rate=0.0,
            burnup_mwd_per_kghm=0.0,
            hydrostatic_stress=-2.0e7,
            time_step=time_step,
            previous_temperature=300.0,
        )
        return jnp.stack(
            [
                result.retained_total,
                result.released_increment,
                result.hbs_swelling,
                result.state.bubble_vacancies,
                result.state.grain_radius,
            ]
        )

    primal, tangent = jax.jit(lambda t: jax.jvp(outputs, (t,), (jnp.ones_like(t),)))(
        jnp.asarray(300.0)
    )
    assert bool(jnp.all(jnp.isfinite(primal)))
    np.testing.assert_array_equal(np.asarray(tangent), np.zeros(5))
    np.testing.assert_array_equal(np.asarray(primal[:4]), np.zeros(4))
