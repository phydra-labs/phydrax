#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cloud solve: initial refusal, route dispatch, saved-row evidence and Bjerknes forces."""

from __future__ import annotations

from typing import assert_never

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from .._validation import finite_real_scalar
from ._cloud import (
    BubbleCloudEvidence,
    BubbleCloudRates,
    BubbleCloudResult,
    BubbleCloudState,
    BubbleCloudTrajectory,
    cloud_radius,
    PreparedBubbleCloud,
)
from ._cloud_coupling import FMMCloudCoupling
from ._cloud_integration import (
    cloud_context,
    CloudContext,
    CloudIntegration,
    integrate_incompressible,
    refused_integration,
)
from ._cloud_retarded import integrate_retarded, pair_delays
from ._status import BubbleDynamicsStatus
from ._validity import BubbleValidityEvidence, neglected_terms


def _initial_status(context: CloudContext, /) -> Array:
    """Refusal of an inadmissible initial configuration before integration."""
    prepared = context.prepared
    plan = prepared.plan
    resources = plan.resources
    state = prepared.initial_state
    time = jnp.zeros(())
    terms = prepared.local_terms(state, time)
    equilibrium = jnp.all(jnp.concatenate([eq.admissible for eq in prepared.equilibria]))
    walls = jnp.all(
        jnp.concatenate([rates.wall.admissible for rates in terms.group_rates])
    )
    overlap = prepared.contact_ratio(state) <= resources.minimum_contact_ratio
    match plan.route:
        case "dense":
            condition, definite, spectral = prepared.coupling_spectrum(state, time)
            conditioned = definite & (condition <= resources.maximum_condition_number)
            coupled = jnp.asarray(True)
            growth_supported = jnp.asarray(True)
        case "fmm":
            coupling = prepared.coupling
            if (
                not isinstance(coupling, FMMCloudCoupling)
                or prepared.candidates is None
                or prepared.growth_limit is None
            ):
                raise ValueError("The FMM route requires a prepared FMM coupling.")
            conditioned = jnp.asarray(True)
            spectral = jnp.zeros(())
            coupled = (
                coupling.structure_successful
                & prepared.candidates.successful
                & prepared.rates(state, time).coupling_successful
            )
            growth_supported = jnp.max(cloud_radius(state)) <= prepared.growth_limit
        case _:
            assert_never(plan.route)
    match plan.coupling:
        case "incompressible":
            neutral = jnp.asarray(True)
        case "retarded":
            neutral = spectral < 1.0
        case _:
            assert_never(plan.coupling)
    status = jnp.asarray(int(BubbleDynamicsStatus.SUCCESS), dtype=jnp.int32)
    # Later entries take precedence: invalid equilibrium outranks certificate limits.
    ordered = (
        (~neutral, BubbleDynamicsStatus.NEUTRAL_UNSTABLE),
        (~conditioned, BubbleDynamicsStatus.ILL_CONDITIONED),
        (~coupled, BubbleDynamicsStatus.COUPLING_FAILURE),
        (overlap, BubbleDynamicsStatus.OVERLAP),
        (~growth_supported, BubbleDynamicsStatus.VALIDITY_EXCEEDED),
        (~(equilibrium & walls), BubbleDynamicsStatus.INVALID_EQUILIBRIUM),
    )
    for refused, code in ordered:
        status = jnp.where(refused, int(code), status)
    return status.astype(jnp.int32)


def _integrate(prepared: PreparedBubbleCloud, /) -> tuple[CloudContext, CloudIntegration]:
    context, flat = cloud_context(prepared)
    plan = prepared.plan
    initial_status = _initial_status(context)
    retarded = plan.coupling == "retarded"

    def run(value: Array) -> CloudIntegration:
        match plan.coupling:
            case "incompressible":
                return integrate_incompressible(context, value)
            case "retarded":
                return integrate_retarded(context, value)
            case _:
                assert_never(plan.coupling)

    def refuse(value: Array) -> CloudIntegration:
        return refused_integration(context, value, initial_status, retarded=retarded)

    successful = initial_status == int(BubbleDynamicsStatus.SUCCESS)
    return context, jax.lax.cond(successful, run, refuse, flat)


def _row_rates(
    prepared: PreparedBubbleCloud,
    integration: CloudIntegration,
    states: BubbleCloudState,
    times: Array,
    /,
) -> BubbleCloudRates:
    plan = prepared.plan
    match plan.coupling:
        case "incompressible":
            return jax.lax.map(
                lambda item: prepared.rates(item[0], item[1]), (states, times)
            )
        case "retarded":
            if integration.row_potential is None or integration.row_gradient is None:
                raise ValueError("Retarded rows require the retarded neighbour field.")
            return jax.lax.map(
                lambda item: prepared.retarded_rates(item[0], item[1], item[2], item[3]),
                (states, times, integration.row_potential, integration.row_gradient),
            )
        case _:
            assert_never(plan.coupling)


def _masked(values: Array, valid: Array, /) -> Array:
    shape = valid.shape + (1,) * (values.ndim - valid.ndim)
    return jnp.where(valid.reshape(shape), values, jnp.nan)


def _extreme(values: Array, valid: Array, /, *, maximum: bool) -> Array:
    shape = valid.shape + (1,) * (values.ndim - valid.ndim)
    mask = jnp.broadcast_to(valid.reshape(shape), values.shape)
    if maximum:
        return jnp.max(jnp.where(mask, values, -jnp.inf))
    return jnp.min(jnp.where(mask, values, jnp.inf))


def _concatenated(rates: BubbleCloudRates, name: str, /) -> Array:
    """One wall-pressure diagnostic of every group along the bubble axis."""
    match name:
        case "gas_pressure":
            parts = [group.wall.gas.pressure for group in rates.group_rates]
        case "gas_temperature":
            parts = [group.wall.gas.temperature for group in rates.group_rates]
        case "hard_core_margin":
            parts = [group.wall.gas.hard_core_margin for group in rates.group_rates]
        case "wall_pressure":
            parts = [group.wall.wall_pressure for group in rates.group_rates]
        case "far_field_pressure":
            parts = [group.wall.far_field_pressure for group in rates.group_rates]
        case "capillary_pressure":
            parts = [
                group.wall.interface.capillary_pressure for group in rates.group_rates
            ]
        case "wall_sound_speed":
            parts = [group.wall_sound_speed for group in rates.group_rates]
        case _:
            raise ValueError(f"Unknown wall diagnostic {name!r}.")
    return jnp.concatenate(parts, axis=1)


def _cloud_neglected_terms(prepared: PreparedBubbleCloud, /) -> tuple[str, ...]:
    plan = prepared.plan
    terms: list[str] = []
    for group in plan.groups:
        for term in neglected_terms(group.model.equation):
            if term not in terms and not (term == "translation" and plan.translating):
                terms.append(term)
    terms.extend(("neighbor-dipole-interaction", "neighbor-compressible-corrections"))
    if plan.coupling == "incompressible":
        terms.append("finite-sound-speed-coupling")
    if plan.translating:
        terms.extend(("neighbor-induced-liquid-velocity", "history-force"))
    return tuple(terms)


def _validity(
    prepared: PreparedBubbleCloud,
    rates: BubbleCloudRates,
    states: BubbleCloudState,
    valid: Array,
    /,
) -> BubbleValidityEvidence:
    plan = prepared.plan
    policy = plan.validity
    radius = jax.vmap(cloud_radius)(states)
    velocity = jnp.concatenate([member.wall_velocity for member in states.groups], axis=1)
    ambient = jnp.concatenate(
        [group.model.environment.ambient_pressure for group in plan.groups]
    )[None, :]
    mach = jnp.abs(velocity) / _concatenated(rates, "wall_sound_speed")
    laplace = jnp.where(
        ambient > 0.0,
        _concatenated(rates, "capillary_pressure")
        / jnp.where(ambient > 0.0, ambient, 1.0),
        jnp.inf,
    )
    temperature = _concatenated(rates, "gas_temperature")
    knudsen = policy.knudsen_number(
        temperature, _concatenated(rates, "gas_pressure"), radius
    )
    tolman = policy.tolman_ratio(radius)
    max_mach = _extreme(mach, valid, maximum=True)
    max_knudsen = None if knudsen is None else _extreme(knudsen, valid, maximum=True)
    max_tolman = None if tolman is None else _extreme(tolman, valid, maximum=True)
    within = max_mach <= policy.mach_limit
    if max_knudsen is not None:
        within = within & (max_knudsen <= policy.knudsen_limit)
    if max_tolman is not None:
        within = within & (max_tolman <= policy.tolman_ratio_limit)
    ratio = radius / prepared.equilibrium_radius[None, :]
    return BubbleValidityEvidence(
        max_mach,
        _extreme(_concatenated(rates, "wall_pressure"), valid, maximum=True),
        _extreme(ratio, valid, maximum=False),
        _extreme(ratio, valid, maximum=True),
        _extreme(temperature, valid, maximum=True),
        _extreme(_concatenated(rates, "hard_core_margin"), valid, maximum=False),
        _extreme(laplace, valid, maximum=True),
        max_knudsen,
        max_tolman,
        within,
        neglected_terms=_cloud_neglected_terms(prepared),
    )


def _kinetic_energy(prepared: PreparedBubbleCloud, state: BubbleCloudState, /) -> Array:
    """Incompressible liquid kinetic energy `2πρ Σ q_i (R_i Ṙ_i + Σ_j q_j/d_ij)`."""
    radius = cloud_radius(state)
    velocity = jnp.concatenate([member.wall_velocity for member in state.groups])
    flux = radius**2 * velocity
    field = prepared.coupling.field(state.position, flux)
    density = prepared.plan.liquid_density()
    return 2.0 * jnp.pi * density * jnp.sum(flux * (radius * velocity + field.potential))


def _coupling_extremes(
    prepared: PreparedBubbleCloud,
    states: BubbleCloudState,
    times: Array,
    valid: Array,
    /,
) -> tuple[Array, Array]:
    plan = prepared.plan
    match plan.route:
        case "dense":
            condition, _, spectral = jax.lax.map(
                lambda item: prepared.coupling_spectrum(item[0], item[1]), (states, times)
            )
            return _extreme(condition, valid, maximum=True), _extreme(
                spectral, valid, maximum=True
            )
        case "fmm":
            return jnp.full((), jnp.nan), jnp.full((), jnp.nan)
        case _:
            assert_never(plan.route)


def _trajectory(
    prepared: PreparedBubbleCloud,
    rates: BubbleCloudRates,
    states: BubbleCloudState,
    valid: Array,
    /,
) -> BubbleCloudTrajectory:
    count = prepared.plan.save_times.shape[0]
    saved_valid = valid[:count]

    def saved(values: Array) -> Array:
        return _masked(values[:count], saved_valid)

    saved_states = jax.tree.map(saved, states)
    return BubbleCloudTrajectory(
        prepared.plan.save_times,
        jax.vmap(cloud_radius)(saved_states),
        jnp.concatenate([member.wall_velocity for member in saved_states.groups], axis=1),
        saved(rates.acceleration),
        saved_states.position,
        saved_states.velocity,
        saved(_concatenated(rates, "gas_pressure")),
        saved(_concatenated(rates, "wall_pressure")),
        saved(_concatenated(rates, "far_field_pressure")),
        saved(rates.neighbor_pressure),
        saved(rates.primary_bjerknes),
        saved(rates.secondary_bjerknes),
        saved_states,
        saved_valid,
    )


def _retardation_ratio(prepared: PreparedBubbleCloud, /) -> Array:
    match prepared.plan.coupling:
        case "incompressible":
            return jnp.zeros(())
        case "retarded":
            return jnp.max(pair_delays(prepared)) / prepared.time_scale
        case _:
            assert_never(prepared.plan.coupling)


def _evidence(
    prepared: PreparedBubbleCloud,
    integration: CloudIntegration,
    rates: BubbleCloudRates,
    states: BubbleCloudState,
    ledgers: Array,
    times: Array,
    valid: Array,
    /,
) -> BubbleCloudEvidence:
    plan = prepared.plan
    kinetic = jax.lax.map(lambda state: _kinetic_energy(prepared, state), states)
    initial_kinetic = _kinetic_energy(prepared, prepared.initial_state)
    kinetic_change = kinetic[-1] - initial_kinetic
    terminal_ledger = ledgers[-1]
    contact = jax.lax.map(prepared.contact_ratio, states)
    condition, spectral = _coupling_extremes(prepared, states, times, valid)
    coupling = prepared.coupling
    candidates = prepared.candidates
    fmm = coupling.resource if isinstance(coupling, FMMCloudCoupling) else None
    exact = (
        plan.coupling == "incompressible"
        and not plan.translating
        and all(group.model.equation == "rayleigh_plesset" for group in plan.groups)
    )
    return BubbleCloudEvidence(
        integration.solver_ok,
        integration.accepted,
        integration.rejected,
        integration.event_kind,
        integration.end_time * prepared.time_scale,
        terminal_ledger[0],
        kinetic_change,
        kinetic_change - terminal_ledger[0],
        terminal_ledger[1],
        terminal_ledger[2],
        _extreme(contact, valid, maximum=False),
        condition,
        spectral,
        jnp.all(jnp.where(valid, rates.coupling_successful, True)),
        jnp.max(jnp.where(valid, rates.coupling_iterations, 0)),
        _extreme(rates.coupling_residual, valid, maximum=True),
        _retardation_ratio(prepared),
        integration.history_occupancy,
        jnp.zeros((), dtype=jnp.int32) if candidates is None else candidates.required,
        jnp.asarray(True) if candidates is None else candidates.successful,
        fmm,
        _validity(prepared, rates, states, valid),
        route=plan.route,
        coupling=plan.coupling,
        integrator=plan.integrator,
        work_identity_exact=exact,
    )


@eqx.filter_jit
def solve_bubble_cloud(prepared: PreparedBubbleCloud, /) -> BubbleCloudResult:
    """Integrate one prepared cloud and return its trajectory, status and evidence."""
    if not isinstance(prepared, PreparedBubbleCloud):
        raise TypeError("prepared must be a PreparedBubbleCloud.")
    plan = prepared.plan
    context, integration = _integrate(prepared)
    rows = jnp.concatenate((integration.saved, integration.end_state[None]), axis=0)
    times = jnp.concatenate(
        (plan.save_times, (integration.end_time * prepared.time_scale)[None])
    )
    valid = jnp.concatenate((integration.covered, jnp.asarray([True])))
    states, ledgers = jax.vmap(context.physical)(rows)
    rates = _row_rates(prepared, integration, states, times)
    return BubbleCloudResult(
        _trajectory(prepared, rates, states, valid),
        jax.tree.map(lambda leaf: leaf[-1], states),
        integration.end_time * prepared.time_scale,
        integration.status,
        _evidence(prepared, integration, rates, states, ledgers, times, valid),
        integration.emission,
        plan_id=plan.plan_id,
        bubble_ids=plan.bubble_ids,
    )


class BjerknesForceResult(StrictModule):
    """Time-averaged primary and secondary Bjerknes forces over one window.

    Forces (N) are trapezoidal time averages of the saved instantaneous forces
    `F₁ = −V ∇p_ac` and `F₂ = −V ∇p_nb` between consecutive saved samples inside
    `[start_time, end_time]`. `covered` requires every sample in the window to
    be valid and at least two samples; resolve the forcing period with the save
    schedule for a meaningful average.
    """

    start_time: Array
    end_time: Array
    primary: Array
    secondary: Array
    sample_count: Array
    covered: Array
    bubble_ids: tuple[int, ...] = eqx.field(static=True)

    @property
    def total(self) -> Array:
        """Primary plus secondary mean force of every bubble."""
        return self.primary + self.secondary


def mean_bjerknes_forces(
    result: BubbleCloudResult, start_time: float, end_time: float, /
) -> BjerknesForceResult:
    """Average the saved Bjerknes forces of `result` over `[start_time, end_time]`."""
    if not isinstance(result, BubbleCloudResult):
        raise TypeError("result must be a BubbleCloudResult.")
    start = finite_real_scalar(start_time, "start_time")
    end = finite_real_scalar(end_time, "end_time")
    if not end > start:
        raise ValueError("end_time must exceed start_time.")
    trajectory = result.trajectory
    times = trajectory.times
    inside = (times >= start) & (times <= end)
    pair = inside[1:] & inside[:-1]
    step = jnp.where(pair, times[1:] - times[:-1], 0.0)
    span = jnp.sum(step)

    def average(force: Array) -> Array:
        middle = 0.5 * (force[1:] + force[:-1])
        weighted = jnp.where(pair[:, None, None], middle, 0.0) * step[:, None, None]
        return jnp.sum(weighted, axis=0) / span

    count = jnp.sum(inside, dtype=jnp.int32)
    covered = jnp.all(jnp.where(inside, trajectory.valid, True)) & (count >= 2)
    return BjerknesForceResult(
        jnp.asarray(start, dtype=jnp.float64),
        jnp.asarray(end, dtype=jnp.float64),
        average(trajectory.primary_bjerknes),
        average(trajectory.secondary_bjerknes),
        count,
        covered,
        bubble_ids=result.bubble_ids,
    )


__all__ = ["BjerknesForceResult", "mean_bjerknes_forces", "solve_bubble_cloud"]
