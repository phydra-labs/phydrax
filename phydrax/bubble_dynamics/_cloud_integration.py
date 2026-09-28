#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nondimensional cloud state, native Diffrax events and the incompressible integration.

The cloud state is flattened per group with each member's inertial state
scales (as for a single bubble) and one global time scale, the fastest member
inertial time. Terminal conditions (overlap, overlap-certificate growth,
minimum radius, hard core, Mach, coupling conditioning or neutral stability,
invalid state, drive support and coupled-solve failure) are native Diffrax
events localized by Newton root finding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_never, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jax.flatten_util import ravel_pytree

from .._strict import StrictModule
from ..solver import DifferentialProblem, solve_diffrax
from ._cloud import (
    BubbleCloudRates,
    BubbleCloudState,
    cloud_radius,
    CloudIntegrator,
    PreparedBubbleCloud,
)
from ._emission import (
    far_field_emission,
    FarFieldEmissionEvidence,
    FarFieldEmissionResult,
)
from ._radial import BubbleState
from ._single import BubbleDifferentiation
from ._status import BubbleDynamicsStatus, BubbleEventKind


_CloudCondition: TypeAlias = Literal[
    "overlap",
    "growth",
    "minimum_radius",
    "hard_core",
    "mach",
    "conditioning",
    "neutral",
    "invalid",
    "support",
    "coupling",
]

_KIND_STATUS: dict[BubbleEventKind, BubbleDynamicsStatus] = {
    BubbleEventKind.NONE: BubbleDynamicsStatus.SUCCESS,
    BubbleEventKind.MINIMUM_RADIUS: BubbleDynamicsStatus.MINIMUM_RADIUS,
    BubbleEventKind.HARD_CORE: BubbleDynamicsStatus.HARD_CORE,
    BubbleEventKind.MACH_LIMIT: BubbleDynamicsStatus.MACH_LIMIT,
    BubbleEventKind.INVALID_STATE: BubbleDynamicsStatus.INVALID_STATE,
    BubbleEventKind.SUPPORT_EXIT: BubbleDynamicsStatus.SUPPORT_EXIT,
    BubbleEventKind.OVERLAP: BubbleDynamicsStatus.OVERLAP,
    BubbleEventKind.COUPLING_FAILURE: BubbleDynamicsStatus.COUPLING_FAILURE,
    BubbleEventKind.ILL_CONDITIONED: BubbleDynamicsStatus.ILL_CONDITIONED,
    BubbleEventKind.SUPPORT_GROWTH: BubbleDynamicsStatus.VALIDITY_EXCEEDED,
    BubbleEventKind.NEUTRAL_UNSTABLE: BubbleDynamicsStatus.NEUTRAL_UNSTABLE,
}


class CloudContext(StrictModule):
    """Nondimensional maps shared by the vector field, events and evidence."""

    prepared: PreparedBubbleCloud
    template: tuple[tuple[BubbleState, ...], Array, Array, Array]

    def physical(self, flat: Array, /) -> tuple[BubbleCloudState, Array]:
        """Physical cloud state and energy ledger of a flat nondimensional state."""
        prepared = self.prepared
        _, unravel = ravel_pytree(self.template)
        groups, position, velocity, ledger = unravel(flat)
        states = tuple(
            jax.tree.map(jnp.multiply, group, scale)
            for group, scale in zip(groups, prepared.state_scales, strict=True)
        )
        if prepared.plan.translating:
            current_position = position * prepared.length_scale
            current_velocity = velocity * prepared.velocity_scale
        else:
            current_position = prepared.initial_state.position
            current_velocity = prepared.initial_state.velocity
        return (
            BubbleCloudState(states, current_position, current_velocity),
            ledger * prepared.energy_scale,
        )

    def flatten(self, rates: BubbleCloudRates, /) -> Array:
        """Flat nondimensional time derivative of the state and ledger."""
        prepared = self.prepared
        time_scale = prepared.time_scale
        groups = tuple(
            jax.tree.map(lambda rate, scale: rate * time_scale / scale, group, scale)
            for group, scale in zip(
                rates.derivative.groups, prepared.state_scales, strict=True
            )
        )
        if prepared.plan.translating:
            position = rates.derivative.position * time_scale / prepared.length_scale
            velocity = rates.derivative.velocity * time_scale / prepared.velocity_scale
        else:
            position = self.template[1]
            velocity = self.template[2]
        ledger = (
            jnp.stack((rates.wall_work_rate, rates.dissipation_rate, rates.gas_heat_rate))
            * time_scale
            / prepared.energy_scale
        )
        return ravel_pytree((groups, position, velocity, ledger))[0]

    def rates(self, time: Array, flat: Array, /) -> BubbleCloudRates:
        """Coupled incompressible rates at nondimensional `time`."""
        state, _ = self.physical(flat)
        return self.prepared.rates(state, time * self.prepared.time_scale)

    def vector_field(self, time: Array, flat: Array, args: None, /) -> Array:
        """Incompressible cloud vector field in nondimensional form."""
        del args
        return self.flatten(self.rates(time, flat))


def cloud_context(prepared: PreparedBubbleCloud, /) -> tuple[CloudContext, Array]:
    """Context and flat initial state of one prepared cloud."""
    initial = prepared.initial_state
    groups = tuple(
        jax.tree.map(jnp.divide, group, scale)
        for group, scale in zip(initial.groups, prepared.state_scales, strict=True)
    )
    if prepared.plan.translating:
        position = initial.position / prepared.length_scale
        velocity = initial.velocity / prepared.velocity_scale
    else:
        position = jnp.zeros((0, 3))
        velocity = jnp.zeros((0, 3))
    template = (groups, position, velocity, jnp.zeros((3,)))
    flat, _ = ravel_pytree(template)
    return CloudContext(prepared, template), flat


class _CloudEventCondition(StrictModule):
    """One native Diffrax event condition on the nondimensional flat state."""

    context: CloudContext
    kind: _CloudCondition = eqx.field(static=True)

    def __call__(self, t: Array, y: Array, args: object, **kwargs: object) -> Array:
        del args, kwargs
        prepared = self.context.prepared
        plan = prepared.plan
        policy = plan.events
        state, _ = self.context.physical(y)
        time = t * prepared.time_scale
        match self.kind:
            case "overlap":
                return (
                    prepared.contact_ratio(state) - plan.resources.minimum_contact_ratio
                )
            case "growth":
                if prepared.growth_limit is None:
                    raise ValueError(
                        "The growth guard requires an FMM overlap certificate."
                    )
                return 1.0 - jnp.max(cloud_radius(state)) / prepared.growth_limit
            case "minimum_radius":
                if policy.minimum_radius_ratio is None:
                    raise ValueError(
                        "Minimum-radius event requires minimum_radius_ratio."
                    )
                ratio = cloud_radius(state) / prepared.equilibrium_radius
                return jnp.min(ratio) - policy.minimum_radius_ratio
            case "hard_core":
                terms = prepared.local_terms(state, time)
                margin = jnp.concatenate(
                    [rates.wall.gas.hard_core_margin for rates in terms.group_rates]
                )
                return jnp.min(margin) - policy.hard_core_margin
            case "mach":
                if policy.mach_limit is None:
                    raise ValueError("Mach event requires mach_limit.")
                terms = prepared.local_terms(state, time)
                speed = jnp.concatenate(
                    [rates.wall_sound_speed for rates in terms.group_rates]
                )
                return 1.0 - jnp.max((terms.velocity / (policy.mach_limit * speed)) ** 2)
            case "conditioning":
                condition, _, _ = prepared.coupling_spectrum(state, time)
                return jnp.log(plan.resources.maximum_condition_number) - jnp.log(
                    condition
                )
            case "neutral":
                _, _, spectral = prepared.coupling_spectrum(state, time)
                return 1.0 - spectral
            case "invalid":
                terms = prepared.local_terms(state, time)
                admissible = jnp.concatenate(
                    [rates.wall.admissible for rates in terms.group_rates]
                )
                return ~jnp.all(admissible)
            case "support":
                terms = prepared.local_terms(state, time)
                support = jnp.concatenate(
                    [rates.wall.drive.in_support for rates in terms.group_rates]
                )
                return ~jnp.all(support)
            case "coupling":
                # Boolean failure detection has no root derivative; blocking its
                # tangents keeps the FMM linear-solve options outside event AD.
                return ~self.context.rates(
                    jax.lax.stop_gradient(t), jax.lax.stop_gradient(y)
                ).coupling_successful
            case _:
                assert_never(self.kind)


def cloud_event(context: CloudContext, /) -> tuple[dfx.Event, tuple[int, ...]]:
    """Native terminal event of one cloud solve and the kind of each condition."""
    plan = context.prepared.plan
    policy = plan.events
    entries: list[tuple[_CloudCondition, bool | None, BubbleEventKind]] = [
        ("overlap", False, BubbleEventKind.OVERLAP)
    ]
    if plan.route == "fmm":
        entries.append(("growth", False, BubbleEventKind.SUPPORT_GROWTH))
    if policy.minimum_radius_ratio is not None:
        entries.append(("minimum_radius", False, BubbleEventKind.MINIMUM_RADIUS))
    entries.append(("hard_core", False, BubbleEventKind.HARD_CORE))
    if policy.mach_limit is not None:
        entries.append(("mach", False, BubbleEventKind.MACH_LIMIT))
    if plan.route == "dense":
        entries.append(("conditioning", False, BubbleEventKind.ILL_CONDITIONED))
    entries.append(("invalid", None, BubbleEventKind.INVALID_STATE))
    entries.append(("support", None, BubbleEventKind.SUPPORT_EXIT))
    match plan.coupling:
        case "incompressible":
            entries.append(("coupling", None, BubbleEventKind.COUPLING_FAILURE))
        case "retarded":
            entries.append(("neutral", False, BubbleEventKind.NEUTRAL_UNSTABLE))
        case _:
            assert_never(plan.coupling)
    tolerance = policy.event_tolerance
    event = dfx.Event(
        tuple(_CloudEventCondition(context, kind) for kind, _, _ in entries),
        root_finder=optx.Newton(rtol=tolerance, atol=tolerance),
        direction=tuple(direction for _, direction, _ in entries),
    )
    return event, tuple(int(event_kind) for _, _, event_kind in entries)


class CloudIntegration(StrictModule):
    """Route-independent outcome of one temporal integration (nondimensional time)."""

    end_time: Array
    end_state: Array
    saved: Array
    covered: Array
    status: Array
    event_kind: Array
    accepted: Array
    rejected: Array
    solver_ok: Array
    emission: FarFieldEmissionResult | None
    row_potential: Array | None
    row_gradient: Array | None
    history_occupancy: Array


def native_solver(integrator: CloudIntegrator, /) -> dfx.AbstractSolver:
    """Tsit5 (explicit) or Kvaerno5 (stiff) native Diffrax solver."""
    match integrator:
        case "explicit":
            return dfx.Tsit5()
        case "stiff":
            return dfx.Kvaerno5()
        case _:
            assert_never(integrator)


def native_adjoint(mode: BubbleDifferentiation, /) -> dfx.AbstractAdjoint:
    """Recursive-checkpoint (reverse) or forward-mode Diffrax adjoint."""
    match mode:
        case "reverse":
            return dfx.RecursiveCheckpointAdjoint()
        case "forward":
            return dfx.ForwardMode()
        case _:
            assert_never(mode)


def terminal_status(
    backend_successful: Array,
    exhausted: Array,
    event_terminated: Array,
    event_mask: object,
    kinds: tuple[int, ...],
    /,
) -> tuple[Array, Array]:
    """Terminal status and event kind from native solver and event evidence."""
    if not isinstance(event_mask, (tuple, list)):
        raise TypeError("A multi-condition event mask must be a sequence.")
    failed = ~backend_successful
    mask = jnp.stack(tuple(jnp.asarray(value, dtype=jnp.bool_) for value in event_mask))
    hit = event_terminated & jnp.any(mask) & ~failed
    kind = jnp.where(
        hit,
        jnp.asarray(kinds, dtype=jnp.int32)[jnp.argmax(mask)],
        int(BubbleEventKind.NONE),
    ).astype(jnp.int32)
    table = [int(BubbleDynamicsStatus.SOLVER_FAILURE)] * len(BubbleEventKind)
    for event_kind, status in _KIND_STATUS.items():
        table[int(event_kind)] = int(status)
    event_status = jnp.asarray(table, dtype=jnp.int32)[kind]
    status = jnp.where(
        failed,
        jnp.where(
            exhausted,
            int(BubbleDynamicsStatus.MAX_STEPS),
            int(BubbleDynamicsStatus.SOLVER_FAILURE),
        ),
        event_status,
    ).astype(jnp.int32)
    return status, kind


def saved_maximum_radius(
    context: CloudContext, saved: Array, covered: Array, end_state: Array, /
) -> Array:
    """Largest radius of every bubble over the covered saved rows and the terminal state."""
    radius = jax.vmap(lambda flat: cloud_radius(context.physical(flat)[0]))(saved)
    terminal = cloud_radius(context.physical(end_state)[0])
    return jnp.maximum(
        jnp.max(jnp.where(covered[:, None], radius, 0.0), axis=0), terminal
    )


def cloud_emission(
    context: CloudContext,
    evaluate: Callable[[Array], Array],
    end_time: Array,
    maximum_radius: Array,
    /,
) -> FarFieldEmissionResult | None:
    """Far-field monopole emission from the dense output, when requested."""
    prepared = context.prepared
    plan = prepared.plan
    if plan.emission is None:
        return None
    time_scale = prepared.time_scale

    def volume_rate(time: Array) -> Array:
        state, _ = context.physical(evaluate(time / time_scale))
        return (
            4.0
            * jnp.pi
            * cloud_radius(state) ** 2
            * jnp.concatenate([member.wall_velocity for member in state.groups])
        )

    return far_field_emission(
        plan.emission,
        volume_rate,
        prepared.initial_state.position,
        maximum_radius,
        plan.liquid_density(),
        plan.sound_speed(),
        jnp.zeros(()),
        end_time * time_scale,
    )


def integrate_incompressible(context: CloudContext, flat: Array, /) -> CloudIntegration:
    """One native Diffrax solve of the implicitly coupled incompressible cloud."""
    prepared = context.prepared
    plan = prepared.plan
    event, kinds = cloud_event(context)
    save_times = plan.save_times / prepared.time_scale
    problem = DifferentialProblem(
        context.vector_field,
        flat,
        t0=jnp.zeros(()),
        t1=save_times[-1],
        args=None,
        problem_id=f"{plan.plan_id}:cloud",
    )
    solution = solve_diffrax(
        problem,
        save_times=save_times[-1:],
        solver=native_solver(plan.integrator),
        adjoint=native_adjoint(plan.differentiation),
        event=event,
        rtol=plan.relative_tolerance,
        atol=plan.absolute_tolerance,
        dense=True,
        max_steps=plan.maximum_steps,
        throw=False,
        solver_configuration_id=f"{plan.plan_id}:{plan.integrator}",
    )
    terminal_ok = solution.terminal_valid & jnp.all(jnp.isfinite(solution.terminal_state))
    end_time = jnp.where(terminal_ok, solution.terminal_time, 0.0)
    end_state = jnp.where(terminal_ok, solution.terminal_state, flat)
    exhausted = solution.backend_result == dfx.RESULTS.max_steps_reached
    status, kind = terminal_status(
        solution.backend_successful,
        exhausted,
        solution.event_terminated,
        solution.event_mask,
        kinds,
    )
    covered = save_times <= end_time
    saved = solution.evaluate(jnp.clip(save_times, 0.0, end_time))
    maximum_radius = saved_maximum_radius(context, saved, covered, end_state)
    return CloudIntegration(
        end_time,
        end_state,
        saved,
        covered,
        status,
        kind,
        jnp.asarray(solution.stats["num_accepted_steps"], dtype=jnp.int32),
        jnp.asarray(solution.stats["num_rejected_steps"], dtype=jnp.int32),
        solution.backend_successful,
        cloud_emission(context, solution.evaluate, end_time, maximum_radius),
        None,
        None,
        jnp.zeros((), dtype=jnp.int32),
    )


def refused_integration(
    context: CloudContext, flat: Array, status: Array, /, *, retarded: bool
) -> CloudIntegration:
    """Outcome of a configuration refused before integration (initial state kept)."""
    plan = context.prepared.plan
    rows = plan.save_times.shape[0]
    count = plan.bubble_count
    emission = plan.emission
    if emission is None:
        refused_emission = None
    else:
        shape = (emission.times.shape[0], emission.observers.shape[0])
        refused_emission = FarFieldEmissionResult(
            emission.times,
            emission.observers,
            jnp.full(shape, jnp.nan),
            FarFieldEmissionEvidence(
                jnp.zeros(shape, dtype=jnp.bool_),
                jnp.zeros(()),
                jnp.full((), jnp.nan),
                jnp.asarray(False),
                jnp.asarray(False),
                evaluation_count=shape[0] * shape[1] * count,
            ),
            plan_id=emission.plan_id,
        )
    return CloudIntegration(
        jnp.zeros(()),
        flat,
        jnp.broadcast_to(flat, (rows, flat.shape[0])),
        jnp.zeros((rows,), dtype=jnp.bool_),
        status,
        jnp.asarray(int(BubbleEventKind.NONE), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.asarray(True),
        refused_emission,
        jnp.full((rows + 1, count), jnp.nan) if retarded else None,
        jnp.full((rows + 1, count, 3), jnp.nan) if retarded else None,
        jnp.zeros((), dtype=jnp.int32),
    )


__all__ = [
    "CloudContext",
    "CloudIntegration",
    "CloudIntegrator",
    "cloud_context",
    "cloud_emission",
    "cloud_event",
    "integrate_incompressible",
    "native_adjoint",
    "native_solver",
    "refused_integration",
    "saved_maximum_radius",
    "terminal_status",
]
