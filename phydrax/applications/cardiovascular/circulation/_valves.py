#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....dynamics import DAEComponent, DAEEquationBlock
from ._components import (
    _ConservationResidual,
    _incidence,
    _positive_scalar,
    _two_port_variables,
    _two_ports,
    PressureFlowComponent,
)


class ValveEventDirection(StrEnum):
    OPENING = "opening"
    CLOSING = "closing"
    NONE = "none"


class _SmoothValveResidual(StrictModule):
    open_resistance: Array
    closed_resistance: Array
    pressure_width: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        pressure_drop = jet.value("pressure_in") - jet.value("pressure_out")
        open_fraction = jax.nn.sigmoid(pressure_drop / self.pressure_width)
        resistance = self.closed_resistance + open_fraction * (
            self.open_resistance - self.closed_resistance
        )
        return pressure_drop - resistance * jet.value("flow_out")


class _ComplementarityValveResidual(StrictModule):
    open_resistance: Array
    pressure_scale: Array
    flow_scale: Array
    smoothing: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        flow = jet.value("flow_out")
        pressure_drop = jet.value("pressure_in") - jet.value("pressure_out")
        normalized_flow = flow / self.flow_scale
        normalized_slack = (
            self.open_resistance * flow - pressure_drop
        ) / self.pressure_scale
        return (
            jnp.sqrt(
                normalized_flow * normalized_flow
                + normalized_slack * normalized_slack
                + 2.0 * self.smoothing * self.smoothing
            )
            - normalized_flow
            - normalized_slack
        )


class _EventValveResidual(StrictModule):
    resistance: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        pressure_drop = jet.value("pressure_in") - jet.value("pressure_out")
        return pressure_drop - self.resistance * jet.value("flow_out")


class SmoothValve(PressureFlowComponent):
    """Everywhere-smooth valve for gradient-based fixed-topology simulation."""

    open_resistance: Array
    closed_resistance: Array
    pressure_width: Array

    def __init__(
        self,
        name: str,
        open_resistance: ArrayLike,
        closed_resistance: ArrayLike,
        /,
        *,
        pressure_width: ArrayLike = 0.05,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        opened, opened_host = _positive_scalar(open_resistance, "open_resistance")
        closed, closed_host = _positive_scalar(closed_resistance, "closed_resistance")
        width, width_host = _positive_scalar(pressure_width, "pressure_width")
        if closed_host <= opened_host:
            raise ValueError("closed_resistance must exceed open_resistance.")
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale),
            (
                DAEEquationBlock(
                    "conserve_flow",
                    _ConservationResidual(),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
                DAEEquationBlock(
                    "smooth_valve",
                    _SmoothValveResidual(opened, closed, width),
                    _incidence(
                        ("pressure_in", 0),
                        ("pressure_out", 0),
                        ("flow_out", 0),
                    ),
                ),
            ),
            _two_ports(),
        )
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="smooth-valve",
            parameters=(
                ("open_resistance_kPa_ms_per_mm3", opened_host),
                ("closed_resistance_kPa_ms_per_mm3", closed_host),
                ("pressure_width_kPa", width_host),
            ),
        )
        self.open_resistance = opened
        self.closed_resistance = closed
        self.pressure_width = width

    def open_fraction(self, pressure_drop: ArrayLike, /) -> Array:
        return jax.nn.sigmoid(jnp.asarray(pressure_drop) / self.pressure_width)

    def flow(self, pressure_drop: ArrayLike, /) -> Array:
        pressure = jnp.asarray(pressure_drop)
        fraction = self.open_fraction(pressure)
        resistance = self.closed_resistance + fraction * (
            self.open_resistance - self.closed_resistance
        )
        return pressure / resistance


class ComplementarityValve(PressureFlowComponent):
    """One-way valve using a scaled Fischer--Burmeister complementarity law."""

    open_resistance: Array
    smoothing: Array
    pressure_scale: Array
    flow_scale: Array

    def __init__(
        self,
        name: str,
        open_resistance: ArrayLike,
        /,
        *,
        smoothing: ArrayLike = 1.0e-8,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        opened, opened_host = _positive_scalar(open_resistance, "open_resistance")
        smoothing_ = jnp.asarray(smoothing)
        smoothing_host = float(smoothing_)
        if (
            smoothing_.shape != ()
            or not np.isfinite(smoothing_host)
            or smoothing_host < 0.0
        ):
            raise ValueError("smoothing must be a finite nonnegative scalar.")
        p_scale_value, p_scale = _positive_scalar(pressure_scale, "pressure_scale")
        q_scale_value, q_scale = _positive_scalar(flow_scale, "flow_scale")
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale),
            (
                DAEEquationBlock(
                    "conserve_flow",
                    _ConservationResidual(),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
                DAEEquationBlock(
                    "complementarity_valve",
                    _ComplementarityValveResidual(
                        opened,
                        p_scale_value,
                        q_scale_value,
                        smoothing_,
                    ),
                    _incidence(
                        ("pressure_in", 0),
                        ("pressure_out", 0),
                        ("flow_out", 0),
                    ),
                ),
            ),
            _two_ports(),
        )
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="complementarity-valve",
            parameters=(
                ("open_resistance_kPa_ms_per_mm3", opened_host),
                ("smoothing", smoothing_host),
                ("pressure_scale_kPa", p_scale),
                ("flow_scale_mm3_per_ms", q_scale),
            ),
        )
        self.open_resistance = opened
        self.smoothing = smoothing_
        self.pressure_scale = p_scale_value
        self.flow_scale = q_scale_value

    def complementarity_residual(
        self, pressure_drop: ArrayLike, flow: ArrayLike, /
    ) -> Array:
        normalized_flow = jnp.asarray(flow) / self.flow_scale
        normalized_slack = (
            self.open_resistance * jnp.asarray(flow) - jnp.asarray(pressure_drop)
        ) / self.pressure_scale
        return (
            jnp.sqrt(
                normalized_flow * normalized_flow
                + normalized_slack * normalized_slack
                + 2.0 * self.smoothing * self.smoothing
            )
            - normalized_flow
            - normalized_slack
        )


class EventValveState(StrictModule):
    """Committed discrete valve state at a fixed-topology differentiation boundary."""

    is_open: Array
    last_transition_time: Array
    transition_count: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        is_open: ArrayLike = False,
        last_transition_time: ArrayLike = -jnp.inf,
        transition_count: ArrayLike = 0,
        /,
    ) -> None:
        opened = jnp.asarray(is_open, dtype=bool)
        last = jnp.asarray(last_transition_time)
        count = jnp.asarray(transition_count, dtype=jnp.int32)
        if opened.shape != () or last.shape != () or count.shape != ():
            raise ValueError("Event valve state entries must be scalars.")
        opened_host = bool(opened)
        last_host = float(last)
        count_host = int(count)
        if np.isnan(last_host) or count_host < 0:
            raise ValueError("Event valve state is invalid.")
        self.is_open = opened
        self.last_transition_time = last
        self.transition_count = count
        self.state_id = canonical_fingerprint(
            {
                "kind": "event-valve-state",
                "is_open": opened_host,
                "last_transition_time": last_host.hex(),
                "transition_count": count_host,
            }
        )


class ValveEventCandidate(StrictModule):
    """Proposed but uncommitted deterministic valve transition."""

    previous_state: EventValveState
    candidate_state: EventValveState
    event_required: Array
    direction: Array
    event_time: Array
    pressure_drop: Array
    valve_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    source_state_id: str = eqx.field(static=True)


class EventValve(PressureFlowComponent):
    """Hysteretic event valve; commits rebuild coefficients, never DAE topology."""

    open_resistance: Array
    closed_resistance: Array
    opening_pressure: Array
    closing_pressure: Array
    minimum_dwell_time: Array
    state: EventValveState
    valve_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        open_resistance: ArrayLike,
        closed_resistance: ArrayLike,
        /,
        *,
        opening_pressure: ArrayLike = 0.0,
        closing_pressure: ArrayLike = -0.01,
        minimum_dwell_time: ArrayLike = 0.0,
        state: EventValveState | None = None,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        opened, opened_host = _positive_scalar(open_resistance, "open_resistance")
        closed, closed_host = _positive_scalar(closed_resistance, "closed_resistance")
        if closed_host <= opened_host:
            raise ValueError("closed_resistance must exceed open_resistance.")
        opening = jnp.asarray(opening_pressure)
        closing = jnp.asarray(closing_pressure)
        dwell = jnp.asarray(minimum_dwell_time)
        host = tuple(float(value) for value in (opening, closing, dwell))
        if (
            any(value.shape != () for value in (opening, closing, dwell))
            or any(not np.isfinite(value) for value in host)
            or host[0] <= host[1]
            or host[2] < 0.0
        ):
            raise ValueError(
                "Event thresholds require opening > closing and nonnegative dwell."
            )
        state_ = EventValveState() if state is None else state
        if not isinstance(state_, EventValveState):
            raise TypeError("state must be an EventValveState or None.")
        active_resistance = jnp.where(state_.is_open, opened, closed)
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale),
            (
                DAEEquationBlock(
                    "conserve_flow",
                    _ConservationResidual(),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
                DAEEquationBlock(
                    "event_valve",
                    _EventValveResidual(active_resistance),
                    _incidence(
                        ("pressure_in", 0),
                        ("pressure_out", 0),
                        ("flow_out", 0),
                    ),
                ),
            ),
            _two_ports(),
        )
        route_id = canonical_fingerprint(
            {
                "kind": "event-valve-route",
                "name": str(name),
                "open_resistance": opened_host.hex(),
                "closed_resistance": closed_host.hex(),
                "opening_pressure": host[0].hex(),
                "closing_pressure": host[1].hex(),
                "minimum_dwell_time": host[2].hex(),
            }
        )
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="event-valve",
            parameters=(
                ("valve_id", route_id),
                ("state", "open" if bool(state_.is_open) else "closed"),
            ),
        )
        self.open_resistance = opened
        self.closed_resistance = closed
        self.opening_pressure = opening
        self.closing_pressure = closing
        self.minimum_dwell_time = dwell
        self.state = state_
        self.valve_id = route_id

    @property
    def active_resistance(self) -> Array:
        return jnp.where(self.state.is_open, self.open_resistance, self.closed_resistance)

    def flow(self, pressure_drop: ArrayLike, /) -> Array:
        return jnp.asarray(pressure_drop) / self.active_resistance

    def propose_event(
        self,
        time: ArrayLike,
        pressure_drop: ArrayLike,
        /,
    ) -> ValveEventCandidate:
        time_ = jnp.asarray(time)
        pressure_ = jnp.asarray(pressure_drop)
        if time_.shape != () or pressure_.shape != ():
            raise ValueError("Event time and pressure drop must be scalars.")
        finite = jnp.isfinite(time_) & jnp.isfinite(pressure_)
        dwell_satisfied = (
            time_ - self.state.last_transition_time >= self.minimum_dwell_time
        )
        opening = (~self.state.is_open) & (pressure_ >= self.opening_pressure)
        closing = self.state.is_open & (pressure_ <= self.closing_pressure)
        event_required = jax.lax.stop_gradient(
            finite & dwell_satisfied & (opening | closing)
        )
        candidate_open = jnp.where(
            event_required,
            opening,
            self.state.is_open,
        )
        direction = jnp.where(
            event_required & opening,
            1,
            jnp.where(event_required & closing, -1, 0),
        ).astype(jnp.int8)
        candidate_state = EventValveState(
            candidate_open,
            jnp.where(event_required, time_, self.state.last_transition_time),
            self.state.transition_count + event_required.astype(jnp.int32),
        )
        candidate_id = canonical_fingerprint(
            {
                "kind": "valve-event-candidate",
                "valve": self.valve_id,
                "source_state": self.state.state_id,
                "event_time": float(time_).hex(),
                "pressure_drop": float(pressure_).hex(),
                "direction": int(direction),
            }
        )
        return ValveEventCandidate(
            self.state,
            candidate_state,
            event_required,
            direction,
            time_,
            pressure_,
            self.valve_id,
            candidate_id,
            self.state.state_id,
        )

    def commit_event(
        self,
        candidate: ValveEventCandidate,
        /,
        *,
        accept: ArrayLike = True,
    ) -> EventValve:
        if not isinstance(candidate, ValveEventCandidate):
            raise TypeError("candidate must be a ValveEventCandidate.")
        if candidate.valve_id != self.valve_id:
            raise ValueError("Valve event candidate belongs to another valve.")
        same_source = (
            candidate.source_state_id == self.state.state_id
            and candidate.previous_state.state_id == self.state.state_id
            and bool(candidate.previous_state.is_open == self.state.is_open)
            and bool(
                candidate.previous_state.last_transition_time
                == self.state.last_transition_time
            )
            and bool(
                candidate.previous_state.transition_count == self.state.transition_count
            )
        )
        if not same_source:
            raise ValueError(
                "Valve event candidate source state does not match current state."
            )
        accepted = jnp.asarray(accept, dtype=bool)
        if accepted.shape != ():
            raise ValueError("accept must be a scalar decision.")
        commit = accepted & candidate.event_required
        state = EventValveState(
            jnp.where(
                commit,
                candidate.candidate_state.is_open,
                self.state.is_open,
            ),
            jnp.where(
                commit,
                candidate.candidate_state.last_transition_time,
                self.state.last_transition_time,
            ),
            jnp.where(
                commit,
                candidate.candidate_state.transition_count,
                self.state.transition_count,
            ),
        )
        return EventValve(
            self.name,
            self.open_resistance,
            self.closed_resistance,
            opening_pressure=self.opening_pressure,
            closing_pressure=self.closing_pressure,
            minimum_dwell_time=self.minimum_dwell_time,
            state=state,
            pressure_scale=float(self.dae_component.variables[0].scale),
            flow_scale=float(self.dae_component.variables[2].scale),
        )


__all__ = [
    "ComplementarityValve",
    "EventValve",
    "EventValveState",
    "SmoothValve",
    "ValveEventCandidate",
    "ValveEventDirection",
]
