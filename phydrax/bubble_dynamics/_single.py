#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single radial bubble: plan, preparation, native integration and evidence.

The state is nondimensionalized with inertial scales and integrated with
`solver.solve_diffrax`. Terminal events and interface regime guards are native
Diffrax events localized by Newton root finding. A piecewise interface law is
integrated as a fixed-capacity sequence of smooth segments: each segment ends at
a localized guard crossing, the regime switches, and the next segment restarts
from the exact event state. `solver.solve_jump_differential` is the stochastic
jump owner and requires Poisson clocks, which a deterministic shell regime does
not have, so the deterministic segment sequence is owned here.
"""

from __future__ import annotations

from typing import assert_never, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field
from ..solver import DifferentialProblem, DifferentialSolution, solve_diffrax
from ..typing import parse
from ._contracts import AbstractBubblePressureDrive, BubbleScales
from ._events import BubbleEventPolicy, BubbleRegimeTape
from ._gas import sphere_volume
from ._radial import BubbleEquilibrium, BubbleState, RadialBubbleModel, RadialBubbleRates
from ._status import bubble_status_successful, BubbleDynamicsStatus, BubbleEventKind
from ._validity import BubbleValidityEvidence, BubbleValidityPolicy, neglected_terms


SingleBubbleIntegrator: TypeAlias = Literal["auto", "explicit", "stiff"]
ResolvedBubbleIntegrator: TypeAlias = Literal["explicit", "stiff"]
BubbleDifferentiation: TypeAlias = Literal["reverse", "forward"]
_ConditionKind: TypeAlias = Literal[
    "regime", "minimum_radius", "hard_core", "mach", "invalid", "support"
]

_KIND_STATUS = (
    BubbleDynamicsStatus.SUCCESS,
    BubbleDynamicsStatus.MINIMUM_RADIUS,
    BubbleDynamicsStatus.HARD_CORE,
    BubbleDynamicsStatus.MACH_LIMIT,
    BubbleDynamicsStatus.INVALID_STATE,
    BubbleDynamicsStatus.SUPPORT_EXIT,
    BubbleDynamicsStatus.SUCCESS,
    BubbleDynamicsStatus.DISSOLVED,
)


def _resolve_integrator(
    integrator: SingleBubbleIntegrator, model: RadialBubbleModel, /
) -> ResolvedBubbleIntegrator:
    match integrator:
        case "auto":
            return "stiff" if model.requires_stiff_integration else "explicit"
        case "explicit":
            return "explicit"
        case "stiff":
            return "stiff"
        case _:
            assert_never(integrator)


def _validated_save_times(save_times: ArrayLike, /) -> np.ndarray:
    times = np.asarray(save_times, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("save_times must be a non-empty rank-1 array.")
    if not np.all(np.isfinite(times)) or times[0] < 0.0 or times[-1] <= 0.0:
        raise ValueError("save_times must be finite, nonnegative and end after t = 0.")
    if times.shape[0] > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("save_times must be strictly increasing.")
    return times


def _positive_float(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


class SingleBubblePlan(StrictModule):
    """Static structure, save schedule, tolerances and event policy of one solve.

    `save_times` are absolute times in seconds measured from the initial state
    at `t = 0`; the solve ends at the last save time. `integrator="auto"` selects
    the stiff implicit route (Kvaerno5) when a composed law declares stiff
    internal dynamics and Tsit5 otherwise. The model and drive carry the
    trainable coefficients; replace them with `eqx.tree_at` to differentiate.
    `differentiation="reverse"` (recursive checkpointing) supports gradients
    and VJPs; `"forward"` supports JVPs and `jax.linearize`.
    """

    model: RadialBubbleModel
    drive: AbstractBubblePressureDrive
    save_times: Array = fixed_field()
    events: BubbleEventPolicy
    validity: BubbleValidityPolicy
    integrator: ResolvedBubbleIntegrator = eqx.field(static=True)
    differentiation: BubbleDifferentiation = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: RadialBubbleModel,
        drive: AbstractBubblePressureDrive,
        save_times: ArrayLike,
        /,
        *,
        integrator: SingleBubbleIntegrator = "auto",
        differentiation: BubbleDifferentiation = "reverse",
        relative_tolerance: float = 1.0e-8,
        absolute_tolerance: float = 1.0e-10,
        maximum_steps: int = 16384,
        events: BubbleEventPolicy | None = None,
        validity: BubbleValidityPolicy | None = None,
    ) -> None:
        if not isinstance(model, RadialBubbleModel):
            raise TypeError("model must be a RadialBubbleModel.")
        if not isinstance(drive, AbstractBubblePressureDrive):
            raise TypeError("drive must be an AbstractBubblePressureDrive.")
        times = _validated_save_times(save_times)
        selected = parse(integrator, SingleBubbleIntegrator, "integrator")
        policy = BubbleEventPolicy() if events is None else events
        if not isinstance(policy, BubbleEventPolicy):
            raise TypeError("events must be a BubbleEventPolicy or None.")
        support = BubbleValidityPolicy() if validity is None else validity
        if not isinstance(support, BubbleValidityPolicy):
            raise TypeError("validity must be a BubbleValidityPolicy or None.")
        rtol = _positive_float(relative_tolerance, "relative_tolerance")
        atol = _positive_float(absolute_tolerance, "absolute_tolerance")
        steps = int(maximum_steps)
        if steps <= 0:
            raise ValueError("maximum_steps must be positive.")
        resolved = _resolve_integrator(selected, model)
        mode = parse(differentiation, BubbleDifferentiation, "differentiation")
        self.model = model
        self.drive = drive
        self.save_times = jnp.asarray(times, dtype=jnp.float64)
        self.events = policy
        self.validity = support
        self.integrator = resolved
        self.differentiation = mode
        self.relative_tolerance = rtol
        self.absolute_tolerance = atol
        self.maximum_steps = steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-bubble-plan",
                "model": model.model_id,
                "drive": drive.drive_id,
                "save_times": times,
                "integrator": resolved,
                "differentiation": mode,
                "relative_tolerance": rtol,
                "absolute_tolerance": atol,
                "maximum_steps": steps,
                "events": policy.policy_id,
                "validity": support.policy_id,
            }
        )

    def prepare(
        self,
        equilibrium_radius: ArrayLike,
        /,
        *,
        initial_radius: ArrayLike | None = None,
        initial_wall_velocity: ArrayLike = 0.0,
    ) -> PreparedSingleBubble:
        """Equilibrium, initial state and nondimensional scales.

        The gas content is fixed by the Laplace balance at `equilibrium_radius`.
        With `initial_radius` the bubble starts displaced from equilibrium with
        the same gas state (polytropic laws follow their adiabat; caloric laws
        keep the ambient temperature).
        """
        model = self.model
        equilibrium = model.equilibrium(equilibrium_radius)
        radius = (
            equilibrium.state.radius
            if initial_radius is None
            else jnp.asarray(initial_radius, dtype=jnp.float64)
        )
        velocity = jnp.asarray(initial_wall_velocity, dtype=jnp.float64)
        state = BubbleState(
            radius,
            velocity,
            equilibrium.state.gas,
            equilibrium.state.liquid,
            equilibrium.state.interface,
        )
        regime = model.interface.initial_regime(radius, equilibrium.state.interface)
        scales = model.characteristic_scales(
            equilibrium, self.drive.characteristic_pressure(), velocity
        )
        state_scale = model.state_scale(equilibrium.state.gas, scales)
        initial_wall = model.wall_pressure(state, regime, jnp.zeros_like(radius), self.drive)
        admissible = (
            equilibrium.admissible
            & initial_wall.admissible
            & (radius > 0.0)
            & jnp.isfinite(velocity)
            & (scales.pressure > 0.0)
        )
        return PreparedSingleBubble(self, equilibrium, state, regime, scales, state_scale, admissible)


class PreparedSingleBubble(StrictModule):
    """Equilibrium, initial state, scales and admissibility of one solve."""

    plan: SingleBubblePlan
    equilibrium: BubbleEquilibrium
    initial_state: BubbleState
    initial_regime: Array
    scales: BubbleScales
    state_scale: BubbleState
    admissible: Array

    def parameter_id(self) -> str:
        """Host fingerprint of every dynamic model and drive coefficient.

        This is an explicit host boundary; call it outside traced code.
        """
        leaves = jax.tree.leaves(eqx.filter((self.plan.model, self.plan.drive), eqx.is_array))
        return canonical_fingerprint(
            {
                "kind": "single-bubble-parameters",
                "plan": self.plan.plan_id,
                "leaves": [np.asarray(leaf) for leaf in leaves],
            }
        )


class SingleBubbleTrajectory(StrictModule):
    """Requested saved trajectory; entries at unreached save times are NaN."""

    times: Array
    radius: Array
    wall_velocity: Array
    acceleration: Array
    gas_pressure: Array
    gas_temperature: Array
    wall_pressure: Array
    far_field_pressure: Array
    surface_tension: Array
    regime: Array
    states: BubbleState
    valid: Array


class BubbleDynamicsEvidence(StrictModule):
    """Solver work, events, energy ledger, validity and derivative evidence.

    `work_residual = ΔK − W` compares the liquid kinetic energy change
    `K = 2πρR³Ṙ²` with the integrated wall work `W = ∫ 4πR²Ṙ (p_L − p_∞) dt`.
    It is an exact identity of Rayleigh–Plesset (`work_identity_exact`), so it
    measures the time-integration error there; for the compressible equations
    `−work_residual` is the energy radiated acoustically.
    """

    solver_successful: Array
    accepted_steps: Array
    rejected_steps: Array
    segment_count: Array
    event_kind: Array
    event_time: Array
    regime_tape: BubbleRegimeTape
    wall_work: Array
    kinetic_energy_change: Array
    work_residual: Array
    dissipated_energy: Array
    gas_heat: Array
    min_inertia_fraction: Array
    validity: BubbleValidityEvidence
    derivative_available: Array
    work_identity_exact: bool = eqx.field(static=True)
    integrator: ResolvedBubbleIntegrator = eqx.field(static=True)
    equation: str = eqx.field(static=True)


class SingleBubbleResult(StrictModule):
    """Trajectory, exact terminal state, status and evidence of one solve."""

    trajectory: SingleBubbleTrajectory
    terminal_state: BubbleState
    terminal_time: Array
    terminal_regime: Array
    status: Array
    evidence: BubbleDynamicsEvidence
    plan_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether the solve reached a physical endpoint without refusal."""
        return bubble_status_successful(self.status) & self.evidence.solver_successful

    @property
    def completed(self) -> Array:
        """Whether the solve reached the final save time."""
        return self.status == int(BubbleDynamicsStatus.SUCCESS)

    def realization_id(self) -> str:
        """Host fingerprint of the computed realization (explicit host boundary)."""
        return canonical_fingerprint(
            {
                "kind": "single-bubble-realization",
                "plan": self.plan_id,
                "model": self.model_id,
                "status": int(self.status),
                "terminal_time": np.asarray(self.terminal_time),
                "radius": np.asarray(self.trajectory.radius),
                "wall_velocity": np.asarray(self.trajectory.wall_velocity),
            }
        )


class _SegmentCarry(StrictModule):
    time: Array
    state: Array
    regime: Array
    done: Array
    status: Array
    event_kind: Array
    event_time: Array
    count: Array
    tape_times: Array
    tape_from: Array
    tape_to: Array
    tape_guard: Array
    tape_radius: Array
    tape_transversality: Array
    saved: Array
    saved_regime: Array
    saved_valid: Array
    accepted: Array
    rejected: Array
    segments: Array
    solver_ok: Array


class _SolveContext(StrictModule):
    """Nondimensional maps shared by the vector field, events and evidence."""

    prepared: PreparedSingleBubble
    template: tuple[BubbleState, Array]

    def physical(self, flat: Array, /) -> tuple[BubbleState, Array]:
        _, unravel = ravel_pytree(self.template)
        scaled, ledger = unravel(flat)
        state = jax.tree.map(jnp.multiply, scaled, self.prepared.state_scale)
        return state, ledger * self.prepared.scales.energy

    def rates(self, time: Array, flat: Array, regime: Array, /) -> RadialBubbleRates:
        state, _ = self.physical(flat)
        prepared = self.prepared
        return prepared.plan.model.rates(
            state, regime, time * prepared.scales.time, prepared.plan.drive
        )

    def vector_field(self, time: Array, flat: Array, regime: Array, /) -> Array:
        rates = self.rates(time, flat, regime)
        scales = self.prepared.scales
        derivative = jax.tree.map(
            lambda rate, scale: rate * scales.time / scale,
            rates.derivative,
            self.prepared.state_scale,
        )
        ledger = (
            jnp.stack((rates.wall_work_rate, rates.dissipation_rate, rates.gas_heat_rate))
            * scales.time
            / scales.energy
        )
        return ravel_pytree((derivative, ledger))[0]


class _BubbleEventCondition(StrictModule):
    """One native Diffrax event condition on the nondimensional flat state."""

    context: _SolveContext
    kind: _ConditionKind = eqx.field(static=True)
    guard: int = eqx.field(static=True)

    def __call__(self, t: Array, y: Array, args: Array, **kwargs: object) -> Array:
        del kwargs
        prepared = self.context.prepared
        plan = prepared.plan
        model = plan.model
        policy = plan.events
        state, _ = self.context.physical(y)
        match self.kind:
            case "regime":
                guards = model.interface.regime_guards(state.radius, state.interface, args)
                return guards[self.guard] + policy.regime_hysteresis
            case "minimum_radius":
                if policy.minimum_radius_ratio is None:
                    raise ValueError("Minimum-radius event requires minimum_radius_ratio.")
                ratio = state.radius / prepared.equilibrium.state.radius
                return ratio - policy.minimum_radius_ratio
            case "hard_core":
                gas = model.gas.evaluate(
                    sphere_volume(state.radius),
                    4.0 * jnp.pi * state.radius**2 * state.wall_velocity,
                    state.gas,
                    model.environment,
                )
                return gas.hard_core_margin - policy.hard_core_margin
            case "mach":
                if policy.mach_limit is None:
                    raise ValueError("Mach event requires mach_limit.")
                rates = self.context.rates(t, y, args)
                return 1.0 - (state.wall_velocity / (policy.mach_limit * rates.wall_sound_speed)) ** 2
            case "invalid":
                wall = model.wall_pressure(state, args, t * prepared.scales.time, plan.drive)
                return ~wall.admissible
            case "support":
                wall = model.wall_pressure(state, args, t * prepared.scales.time, plan.drive)
                return ~wall.drive.in_support
            case _:
                assert_never(self.kind)


def _event_conditions(
    context: _SolveContext, /
) -> tuple[tuple[_BubbleEventCondition, ...], tuple[bool | None, ...], tuple[int, ...]]:
    policy = context.prepared.plan.events
    guard_count = context.prepared.plan.model.interface.guard_count
    entries: list[tuple[_ConditionKind, int, bool | None, BubbleEventKind]] = [
        ("regime", guard, False, BubbleEventKind.REGIME_TRANSITION)
        for guard in range(guard_count)
    ]
    if policy.minimum_radius_ratio is not None:
        entries.append(("minimum_radius", 0, False, BubbleEventKind.MINIMUM_RADIUS))
    entries.append(("hard_core", 0, False, BubbleEventKind.HARD_CORE))
    if policy.mach_limit is not None:
        entries.append(("mach", 0, False, BubbleEventKind.MACH_LIMIT))
    entries.append(("invalid", 0, None, BubbleEventKind.INVALID_STATE))
    entries.append(("support", 0, None, BubbleEventKind.SUPPORT_EXIT))
    conditions = tuple(
        _BubbleEventCondition(context, kind, guard) for kind, guard, _, _ in entries
    )
    directions = tuple(direction for _, _, direction, _ in entries)
    kinds = tuple(int(kind) for _, _, _, kind in entries)
    return conditions, directions, kinds


def _native_solver(integrator: ResolvedBubbleIntegrator, /) -> dfx.AbstractSolver:
    match integrator:
        case "explicit":
            return dfx.Tsit5()
        case "stiff":
            return dfx.Kvaerno5()
        case _:
            assert_never(integrator)


def _native_adjoint(mode: BubbleDifferentiation, /) -> dfx.AbstractAdjoint:
    match mode:
        case "reverse":
            return dfx.RecursiveCheckpointAdjoint()
        case "forward":
            return dfx.ForwardMode()
        case _:
            assert_never(mode)


def _segment(
    context: _SolveContext,
    event: dfx.Event,
    carry: _SegmentCarry,
    save_times: Array,
    /,
) -> DifferentialSolution:
    plan = context.prepared.plan
    problem = DifferentialProblem(
        context.vector_field,
        carry.state,
        t0=carry.time,
        t1=save_times[-1],
        args=carry.regime,
        problem_id=f"{plan.plan_id}:segment",
    )
    return solve_diffrax(
        problem,
        save_times=save_times[-1:],
        solver=_native_solver(plan.integrator),
        adjoint=_native_adjoint(plan.differentiation),
        event=event,
        rtol=plan.relative_tolerance,
        atol=plan.absolute_tolerance,
        dense=True,
        max_steps=plan.maximum_steps,
        throw=False,
        solver_configuration_id=f"{plan.plan_id}:{plan.integrator}",
    )


def _advance(
    context: _SolveContext,
    event: dfx.Event,
    kinds: tuple[int, ...],
    save_times: Array,
    carry: _SegmentCarry,
    /,
) -> _SegmentCarry:
    prepared = context.prepared
    plan = prepared.plan
    interface = plan.model.interface
    guard_count = interface.guard_count
    solution = _segment(context, event, carry, save_times)
    terminal_ok = solution.terminal_valid & jnp.all(jnp.isfinite(solution.terminal_state))
    end_time = jnp.where(terminal_ok, solution.terminal_time, carry.time)
    end_state = jnp.where(terminal_ok, solution.terminal_state, carry.state)
    failed = ~solution.backend_successful
    exhausted = solution.backend_result == dfx.RESULTS.max_steps_reached
    mask = jnp.stack(tuple(jnp.asarray(value, dtype=jnp.bool_) for value in solution.event_mask))
    hit = solution.event_terminated & jnp.any(mask)
    kind = jnp.where(
        hit & ~failed,
        jnp.asarray(kinds, dtype=jnp.int32)[jnp.argmax(mask)],
        int(BubbleEventKind.NONE),
    )
    transition = kind == int(BubbleEventKind.REGIME_TRANSITION)
    kind_status = jnp.asarray([int(value) for value in _KIND_STATUS], dtype=jnp.int32)[kind]
    status = jnp.where(
        failed,
        jnp.where(
            exhausted,
            int(BubbleDynamicsStatus.MAX_STEPS),
            int(BubbleDynamicsStatus.SOLVER_FAILURE),
        ),
        kind_status,
    ).astype(jnp.int32)

    values = solution.evaluate(jnp.clip(save_times, carry.time, end_time))
    covered = (save_times >= carry.time) & (save_times <= end_time)
    saved = jnp.where(covered[:, None], values, carry.saved)
    saved_regime = jnp.where(covered, carry.regime, carry.saved_regime)
    saved_valid = carry.saved_valid | covered

    regime = carry.regime
    count = carry.count
    tape = (
        carry.tape_times,
        carry.tape_from,
        carry.tape_to,
        carry.tape_guard,
        carry.tape_radius,
        carry.tape_transversality,
    )
    exceeded = jnp.asarray(False)
    if guard_count > 0:
        state, _ = context.physical(end_state)
        guard = jnp.argmax(mask[:guard_count]).astype(jnp.int32)
        new_regime = interface.regime_after_crossing(
            state.radius, state.wall_velocity, state.interface, regime, guard
        )
        capacity = plan.events.regime_capacity
        exceeded = transition & (count >= capacity)
        record = transition & ~exceeded
        slot = jnp.minimum(count, capacity - 1)
        entries = (
            end_time * prepared.scales.time,
            regime,
            new_regime,
            guard,
            state.radius,
            jnp.abs(state.wall_velocity) / prepared.scales.velocity,
        )
        tape = tuple(
            column.at[slot].set(jnp.where(record, entry.astype(column.dtype), column[slot]))
            for column, entry in zip(tape, entries, strict=True)
        )
        regime = jnp.where(record, new_regime, regime).astype(jnp.int32)
        count = count + record.astype(jnp.int32)
    status = jnp.where(
        exceeded, int(BubbleDynamicsStatus.REGIME_CAPACITY), status
    ).astype(jnp.int32)
    done = failed | ~transition | exceeded
    return _SegmentCarry(
        end_time,
        end_state,
        regime,
        done,
        status,
        jnp.where(transition & ~exceeded, carry.event_kind, kind).astype(jnp.int32),
        jnp.where(transition & ~exceeded, carry.event_time, end_time * prepared.scales.time),
        count,
        tape[0],
        tape[1],
        tape[2],
        tape[3],
        tape[4],
        tape[5],
        saved,
        saved_regime,
        saved_valid,
        carry.accepted + jnp.asarray(solution.stats["num_accepted_steps"], dtype=jnp.int32),
        carry.rejected + jnp.asarray(solution.stats["num_rejected_steps"], dtype=jnp.int32),
        carry.segments + 1,
        carry.solver_ok & ~failed,
    )


def _initial_carry(
    prepared: PreparedSingleBubble, flat: Array, save_count: int, /
) -> _SegmentCarry:
    guard_count = prepared.plan.model.interface.guard_count
    capacity = prepared.plan.events.regime_capacity if guard_count > 0 else 0
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    regime = prepared.initial_regime.astype(jnp.int32)
    return _SegmentCarry(
        jnp.asarray(0.0, dtype=jnp.float64),
        flat,
        regime,
        ~prepared.admissible,
        jnp.where(
            prepared.admissible,
            int(BubbleDynamicsStatus.SUCCESS),
            int(BubbleDynamicsStatus.INVALID_EQUILIBRIUM),
        ).astype(jnp.int32),
        jnp.asarray(int(BubbleEventKind.NONE), dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        zero_int,
        jnp.zeros((capacity,), dtype=jnp.float64),
        jnp.full((capacity,), regime, dtype=jnp.int32),
        jnp.full((capacity,), regime, dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.float64),
        jnp.zeros((capacity,), dtype=jnp.float64),
        jnp.broadcast_to(flat, (save_count, flat.shape[0])),
        jnp.full((save_count,), regime, dtype=jnp.int32),
        jnp.zeros((save_count,), dtype=jnp.bool_),
        zero_int,
        zero_int,
        zero_int,
        jnp.asarray(True),
    )


def _integrate(prepared: PreparedSingleBubble, /) -> tuple[_SolveContext, _SegmentCarry]:
    plan = prepared.plan
    initial_scaled = (
        jax.tree.map(jnp.divide, prepared.initial_state, prepared.state_scale),
        jnp.zeros((3,), dtype=jnp.float64),
    )
    flat, _ = ravel_pytree(initial_scaled)
    context = _SolveContext(prepared, initial_scaled)
    conditions, directions, kinds = _event_conditions(context)
    tolerance = plan.events.event_tolerance
    event = dfx.Event(
        conditions,
        root_finder=optx.Newton(rtol=tolerance, atol=tolerance),
        direction=directions,
    )
    save_times = plan.save_times / prepared.scales.time
    initial = _initial_carry(prepared, flat, save_times.shape[0])

    def advance(carry: _SegmentCarry) -> _SegmentCarry:
        return _advance(context, event, kinds, save_times, carry)

    def step(carry: _SegmentCarry, _: None) -> tuple[_SegmentCarry, None]:
        return jax.lax.cond(carry.done, lambda value: value, advance, carry), None

    segments = (
        plan.events.regime_capacity + 1 if plan.model.interface.guard_count > 0 else 1
    )
    final, _ = jax.lax.scan(step, initial, None, length=segments)
    return context, final


def _masked(values: Array, valid: Array, /) -> Array:
    shape = valid.shape + (1,) * (values.ndim - valid.ndim)
    return jnp.where(valid.reshape(shape), values, jnp.nan)


def _masked_extreme(values: Array, valid: Array, *, maximum: bool) -> Array:
    if maximum:
        return jnp.max(jnp.where(valid, values, -jnp.inf))
    return jnp.min(jnp.where(valid, values, jnp.inf))


def _validity_evidence(
    plan: SingleBubblePlan,
    equilibrium_radius: Array,
    states: BubbleState,
    rates: RadialBubbleRates,
    valid: Array,
    /,
) -> BubbleValidityEvidence:
    policy = plan.validity
    wall = rates.wall
    radius = states.radius
    mach = jnp.abs(states.wall_velocity) / rates.wall_sound_speed
    ambient = plan.model.environment.ambient_pressure
    laplace = jnp.where(
        ambient > 0.0,
        wall.interface.capillary_pressure / jnp.where(ambient > 0.0, ambient, 1.0),
        jnp.inf,
    )
    knudsen = policy.knudsen_number(wall.gas.temperature, wall.gas.pressure, radius)
    tolman = policy.tolman_ratio(radius)
    max_mach = _masked_extreme(mach, valid, maximum=True)
    max_knudsen = None if knudsen is None else _masked_extreme(knudsen, valid, maximum=True)
    max_tolman = None if tolman is None else _masked_extreme(tolman, valid, maximum=True)
    within = max_mach <= policy.mach_limit
    if max_knudsen is not None:
        within = within & (max_knudsen <= policy.knudsen_limit)
    if max_tolman is not None:
        within = within & (max_tolman <= policy.tolman_ratio_limit)
    return BubbleValidityEvidence(
        max_mach,
        _masked_extreme(wall.wall_pressure, valid, maximum=True),
        _masked_extreme(radius / equilibrium_radius, valid, maximum=False),
        _masked_extreme(radius / equilibrium_radius, valid, maximum=True),
        _masked_extreme(wall.gas.temperature, valid, maximum=True),
        _masked_extreme(wall.gas.hard_core_margin, valid, maximum=False),
        _masked_extreme(laplace, valid, maximum=True),
        max_knudsen,
        max_tolman,
        within,
        neglected_terms=neglected_terms(plan.model.equation),
    )


@eqx.filter_jit
def solve_single_bubble(prepared: PreparedSingleBubble, /) -> SingleBubbleResult:
    """Integrate one prepared bubble and return its trajectory and evidence."""
    if not isinstance(prepared, PreparedSingleBubble):
        raise TypeError("prepared must be a PreparedSingleBubble.")
    plan = prepared.plan
    model = plan.model
    context, final = _integrate(prepared)
    rows = jnp.concatenate((final.saved, final.state[None]), axis=0)
    row_regimes = jnp.concatenate((final.saved_regime, final.regime[None]))
    row_times = jnp.concatenate((plan.save_times, (final.time * prepared.scales.time)[None]))
    row_valid = jnp.concatenate((final.saved_valid, jnp.asarray([True])))
    states, ledgers = jax.vmap(context.physical)(rows)
    rates = jax.vmap(lambda state, regime, time: model.rates(state, regime, time, plan.drive))(
        states, row_regimes, row_times
    )
    count = plan.save_times.shape[0]
    saved_valid = row_valid[:count]
    saved_states = jax.tree.map(lambda leaf: _masked(leaf[:count], saved_valid), states)
    trajectory = SingleBubbleTrajectory(
        plan.save_times,
        saved_states.radius,
        saved_states.wall_velocity,
        _masked(rates.acceleration[:count], saved_valid),
        _masked(rates.wall.gas.pressure[:count], saved_valid),
        _masked(rates.wall.gas.temperature[:count], saved_valid),
        _masked(rates.wall.wall_pressure[:count], saved_valid),
        _masked(rates.wall.far_field_pressure[:count], saved_valid),
        _masked(rates.wall.interface.surface_tension[:count], saved_valid),
        jnp.where(saved_valid, row_regimes[:count], -1).astype(jnp.int32),
        saved_states,
        saved_valid,
    )
    terminal_state = jax.tree.map(lambda leaf: leaf[-1], states)
    terminal_ledger = ledgers[-1]
    density = model.far_field_density()
    kinetic = 2.0 * jnp.pi * density * states.radius**3 * states.wall_velocity**2
    initial_kinetic = (
        2.0
        * jnp.pi
        * density
        * prepared.initial_state.radius**3
        * prepared.initial_state.wall_velocity**2
    )
    kinetic_change = kinetic[-1] - initial_kinetic
    capacity = final.tape_times.shape[0]
    active = jnp.arange(capacity) < final.count
    tape = BubbleRegimeTape(
        final.tape_times,
        final.tape_from,
        final.tape_to,
        final.tape_guard,
        final.tape_radius,
        final.tape_transversality,
        active,
        final.count,
        final.status == int(BubbleDynamicsStatus.REGIME_CAPACITY),
    )
    transversal = jnp.all(
        jnp.where(active, final.tape_transversality >= plan.events.transversality_tolerance, True)
    )
    validity = _validity_evidence(
        plan, prepared.equilibrium.state.radius, states, rates, row_valid
    )
    successful = bubble_status_successful(final.status) & final.solver_ok
    evidence = BubbleDynamicsEvidence(
        final.solver_ok,
        final.accepted,
        final.rejected,
        final.segments,
        final.event_kind,
        final.event_time,
        tape,
        terminal_ledger[0],
        kinetic_change,
        kinetic_change - terminal_ledger[0],
        terminal_ledger[1],
        terminal_ledger[2],
        _masked_extreme(rates.inertia_fraction, row_valid, maximum=False),
        validity,
        successful & transversal,
        work_identity_exact=model.equation == "rayleigh_plesset",
        integrator=plan.integrator,
        equation=model.equation,
    )
    return SingleBubbleResult(
        trajectory,
        terminal_state,
        final.time * prepared.scales.time,
        final.regime,
        final.status,
        evidence,
        plan_id=plan.plan_id,
        model_id=model.model_id,
    )


__all__ = [
    "BubbleDifferentiation",
    "BubbleDynamicsEvidence",
    "PreparedSingleBubble",
    "SingleBubbleIntegrator",
    "SingleBubblePlan",
    "SingleBubbleResult",
    "SingleBubbleTrajectory",
    "solve_single_bubble",
]
