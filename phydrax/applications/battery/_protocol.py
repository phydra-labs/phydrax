#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import HeldInputPolicy, InputLayout


StopDirection: TypeAlias = Literal["below", "above"]


class BatteryTerminalConvention(StrictModule, NonTrainableState):
    """Immutable SI sign convention shared by battery models and protocols."""

    voltage_definition: str = eqx.field(static=True)
    current_definition: str = eqx.field(static=True)
    absorbed_power_definition: str = eqx.field(static=True)
    discharge_current_definition: str = eqx.field(static=True)
    interfacial_flux_definition: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self):
        self.voltage_definition = (
            "positive-terminal potential minus negative-terminal potential"
        )
        self.current_definition = "positive current enters the positive terminal"
        self.absorbed_power_definition = "terminal voltage times terminal current"
        self.discharge_current_definition = (
            "negative terminal current divided by electrode area"
        )
        self.interfacial_flux_definition = (
            "positive interfacial molar flux is oxidation and outward from the solid"
        )
        self.convention_id = canonical_fingerprint(
            {
                "kind": "battery-terminal-convention",
                "voltage": "V=V_positive-V_negative",
                "terminal_current": "I_t>0-enters-positive-terminal",
                "absorbed_power": "P_abs=V*I_t",
                "discharge_areal_current": "i_d=-I_t/A",
                "interfacial_flux": "j>0-oxidation-outward-solid",
            }
        )

    def terminal_voltage(
        self, positive_potential_v: ArrayLike, negative_potential_v: ArrayLike, /
    ) -> Array:
        return jnp.asarray(positive_potential_v) - jnp.asarray(negative_potential_v)

    def absorbed_power(self, voltage_v: ArrayLike, current_a: ArrayLike, /) -> Array:
        return jnp.asarray(voltage_v) * jnp.asarray(current_a)

    def discharge_areal_current(
        self, terminal_current_a: ArrayLike, electrode_area_m2: ArrayLike, /
    ) -> Array:
        area = jnp.asarray(electrode_area_m2)
        area = eqx.error_if(
            area,
            ~jnp.all(jnp.isfinite(area) & (area > 0.0)),
            "Battery electrode area must be finite and positive.",
        )
        return -jnp.asarray(terminal_current_a) / area


PASSIVE_TERMINAL_CONVENTION = BatteryTerminalConvention()

BATTERY_CURRENT_INPUT_LAYOUT = InputLayout(
    (1,),
    axes=("terminal_input",),
    component_names=("terminal_current_a",),
    roles=("control",),
    layout_id="battery:terminal-current-si-passive",
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _duration(value: float, /) -> float:
    if isinstance(value, bool):
        raise TypeError("Battery step duration must be a real scalar.")
    duration = float(value)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("Battery step duration must be finite and positive.")
    return duration


def _direction(value: StopDirection, /) -> StopDirection:
    if value not in ("below", "above"):
        raise ValueError("Stop direction must be 'below' or 'above'.")
    return value


class VoltageStopGuard(StrictModule, NonTrainableState):
    """Stop when terminal voltage crosses a dynamic lower or upper threshold."""

    direction: StopDirection = eqx.field(static=True)
    guard_id: str = eqx.field(static=True)

    def __init__(self, direction: StopDirection, /):
        direction_ = _direction(direction)
        self.direction = direction_
        self.guard_id = canonical_fingerprint(
            {
                "kind": "battery-stop-guard",
                "quantity": "voltage_v",
                "direction": direction_,
            }
        )

    @property
    def quantity(self) -> str:
        return "voltage_v"

    @property
    def reason(self) -> str:
        return f"voltage-{self.direction}"


class CurrentStopGuard(StrictModule, NonTrainableState):
    """Stop when passive terminal current crosses a dynamic threshold."""

    direction: StopDirection = eqx.field(static=True)
    guard_id: str = eqx.field(static=True)

    def __init__(self, direction: StopDirection, /):
        direction_ = _direction(direction)
        self.direction = direction_
        self.guard_id = canonical_fingerprint(
            {
                "kind": "battery-stop-guard",
                "quantity": "current_a",
                "direction": direction_,
            }
        )

    @property
    def quantity(self) -> str:
        return "current_a"

    @property
    def reason(self) -> str:
        return f"current-{self.direction}"


class TemperatureStopGuard(StrictModule, NonTrainableState):
    """Stop when cell temperature crosses a dynamic threshold."""

    direction: StopDirection = eqx.field(static=True)
    guard_id: str = eqx.field(static=True)

    def __init__(self, direction: StopDirection, /):
        direction_ = _direction(direction)
        self.direction = direction_
        self.guard_id = canonical_fingerprint(
            {
                "kind": "battery-stop-guard",
                "quantity": "temperature_k",
                "direction": direction_,
            }
        )

    @property
    def quantity(self) -> str:
        return "temperature_k"

    @property
    def reason(self) -> str:
        return f"temperature-{self.direction}"


class StoichiometryStopGuard(StrictModule, NonTrainableState):
    """Stop on one model-declared electrode stoichiometry coordinate."""

    component: str = eqx.field(static=True)
    direction: StopDirection = eqx.field(static=True)
    guard_id: str = eqx.field(static=True)

    def __init__(self, component: str, direction: StopDirection, /):
        component_ = _identifier(component, "Stoichiometry component")
        direction_ = _direction(direction)
        self.component = component_
        self.direction = direction_
        self.guard_id = canonical_fingerprint(
            {
                "kind": "battery-stop-guard",
                "quantity": f"stoichiometry:{component_}",
                "direction": direction_,
            }
        )

    @property
    def quantity(self) -> str:
        return f"stoichiometry:{self.component}"

    @property
    def reason(self) -> str:
        return f"stoichiometry:{self.component}-{self.direction}"


BatteryStopGuard: TypeAlias = (
    VoltageStopGuard | CurrentStopGuard | TemperatureStopGuard | StoichiometryStopGuard
)
_STOP_GUARD_TYPES = (
    VoltageStopGuard,
    CurrentStopGuard,
    TemperatureStopGuard,
    StoichiometryStopGuard,
)


def _guards(values: Sequence[BatteryStopGuard], /) -> tuple[BatteryStopGuard, ...]:
    resolved = tuple(values)
    if any(not isinstance(guard, _STOP_GUARD_TYPES) for guard in resolved):
        raise TypeError("Battery steps accept only typed battery stop guards.")
    return resolved


class CurrentStepPlan(StrictModule, NonTrainableState):
    """One interval-held prescribed-current phase without a captured amplitude."""

    duration_s: float = eqx.field(static=True)
    stop_guards: tuple[BatteryStopGuard, ...]
    label: str = eqx.field(static=True)
    step_id: str = eqx.field(static=True)

    def __init__(
        self,
        duration_s: float,
        /,
        *,
        stop_guards: Sequence[BatteryStopGuard] = (),
        label: str = "current",
    ):
        duration = _duration(duration_s)
        guards = _guards(stop_guards)
        label_ = _identifier(label, "Battery step label")
        self.duration_s = duration
        self.stop_guards = guards
        self.label = label_
        self.step_id = canonical_fingerprint(
            {
                "kind": "battery-current-step",
                "duration_s": duration,
                "label": label_,
                "guards": [guard.guard_id for guard in guards],
            }
        )


class RestStepPlan(StrictModule, NonTrainableState):
    """One interval-held zero-current phase."""

    duration_s: float = eqx.field(static=True)
    stop_guards: tuple[BatteryStopGuard, ...]
    label: str = eqx.field(static=True)
    step_id: str = eqx.field(static=True)

    def __init__(
        self,
        duration_s: float,
        /,
        *,
        stop_guards: Sequence[BatteryStopGuard] = (),
        label: str = "rest",
    ):
        duration = _duration(duration_s)
        guards = _guards(stop_guards)
        label_ = _identifier(label, "Battery step label")
        self.duration_s = duration
        self.stop_guards = guards
        self.label = label_
        self.step_id = canonical_fingerprint(
            {
                "kind": "battery-rest-step",
                "duration_s": duration,
                "label": label_,
                "guards": [guard.guard_id for guard in guards],
            }
        )


BatteryStepPlan: TypeAlias = CurrentStepPlan | RestStepPlan
_STEP_TYPES = (CurrentStepPlan, RestStepPlan)


class BatteryProtocolPlan(StrictModule, NonTrainableState):
    """Static current/rest topology, guard topology, and endpoint convention."""

    steps: tuple[BatteryStepPlan, ...]
    boundary_times_s: Array
    interval_current_indices: Array
    guard_step_indices: Array
    t0_s: float = eqx.field(static=True)
    node_side: Literal["left", "right"] = eqx.field(static=True)
    current_step_count: int = eqx.field(static=True)
    guard_count: int = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)

    def __init__(
        self,
        steps: Sequence[BatteryStepPlan],
        /,
        *,
        t0_s: float = 0.0,
        node_side: Literal["left", "right"] = "left",
    ):
        resolved = tuple(steps)
        if not resolved or any(not isinstance(step, _STEP_TYPES) for step in resolved):
            raise TypeError("BatteryProtocolPlan requires typed current/rest steps.")
        if node_side not in ("left", "right"):
            raise ValueError("Battery protocol node_side must be 'left' or 'right'.")
        start = float(t0_s)
        if not np.isfinite(start):
            raise ValueError("Battery protocol t0_s must be finite.")
        boundaries = [start]
        current_indices: list[int] = []
        guard_steps: list[int] = []
        current_count = 0
        for step_index, step in enumerate(resolved):
            boundaries.append(boundaries[-1] + step.duration_s)
            if isinstance(step, CurrentStepPlan):
                current_indices.append(current_count)
                current_count += 1
            else:
                current_indices.append(-1)
            guard_steps.extend((step_index,) * len(step.stop_guards))
        boundary_array = np.asarray(boundaries, dtype=float)
        if not np.all(np.isfinite(boundary_array)) or np.any(
            np.diff(boundary_array) <= 0.0
        ):
            raise ValueError("Battery protocol boundaries must be finite and increasing.")
        self.steps = resolved
        self.boundary_times_s = jax.lax.stop_gradient(jnp.asarray(boundary_array))
        self.interval_current_indices = jax.lax.stop_gradient(
            jnp.asarray(current_indices, dtype=jnp.int32)
        )
        self.guard_step_indices = jax.lax.stop_gradient(
            jnp.asarray(guard_steps, dtype=jnp.int32)
        )
        self.t0_s = start
        self.node_side = node_side
        self.current_step_count = current_count
        self.guard_count = len(guard_steps)
        self.protocol_id = canonical_fingerprint(
            {
                "kind": "battery-protocol",
                "steps": [step.step_id for step in resolved],
                "t0_s": start,
                "node_side": node_side,
                "terminal_convention": PASSIVE_TERMINAL_CONVENTION.convention_id,
            }
        )

    @property
    def t1_s(self) -> float:
        return float(np.asarray(self.boundary_times_s)[-1])

    @property
    def stop_guards(self) -> tuple[BatteryStopGuard, ...]:
        return tuple(guard for step in self.steps for guard in step.stop_guards)

    @property
    def transition_times_s(self) -> Array:
        return self.boundary_times_s

    def interval_index(self, time_s: ArrayLike, /) -> Array:
        time = jnp.asarray(time_s, dtype=self.boundary_times_s.dtype)
        side = "left" if self.node_side == "left" else "right"
        index = jnp.searchsorted(self.boundary_times_s, time, side=side) - 1
        return jnp.clip(index, 0, len(self.steps) - 1)

    def interval_currents(self, values: "BatteryProtocolValues", /) -> Array:
        self._validate_values(values)
        if self.current_step_count == 0:
            return jnp.zeros((len(self.steps),), dtype=values.current_amplitudes_a.dtype)
        indices = jnp.maximum(self.interval_current_indices, 0)
        selected = values.current_amplitudes_a[indices]
        return jnp.where(self.interval_current_indices >= 0, selected, 0.0)

    def current(self, time_s: ArrayLike, values: "BatteryProtocolValues", /) -> Array:
        currents = self.interval_currents(values)
        return currents[self.interval_index(time_s)]

    def input_policy(
        self,
        values: "BatteryProtocolValues",
        /,
        *,
        node_side: Literal["left", "right"] | None = None,
    ) -> HeldInputPolicy:
        currents = self.interval_currents(values)
        side = self.node_side if node_side is None else node_side
        if side not in ("left", "right"):
            raise ValueError("Battery input-policy node_side must be 'left' or 'right'.")
        return HeldInputPolicy(
            self.boundary_times_s,
            currents[:, None],
            input_layout=BATTERY_CURRENT_INPUT_LAYOUT,
            node_side=side,
            policy_id=f"battery-protocol:{self.protocol_id}:terminal-current:{side}",
        )

    def _validate_values(self, values: "BatteryProtocolValues", /) -> None:
        if not isinstance(values, BatteryProtocolValues):
            raise TypeError("values must be BatteryProtocolValues.")
        if values.protocol_id != self.protocol_id:
            raise ValueError(
                "Battery protocol values do not match this protocol topology."
            )


class BatteryProtocolValues(StrictModule):
    """Dynamic trainable current amplitudes and stop thresholds for one topology."""

    current_amplitudes_a: Array
    stop_thresholds: Array
    protocol_id: str = eqx.field(static=True)

    def __init__(
        self,
        protocol: BatteryProtocolPlan,
        current_amplitudes_a: ArrayLike,
        stop_thresholds: ArrayLike = (),
        /,
    ):
        if not isinstance(protocol, BatteryProtocolPlan):
            raise TypeError("protocol must be a BatteryProtocolPlan.")
        current_input = jnp.asarray(current_amplitudes_a)
        threshold_input = jnp.asarray(stop_thresholds)
        if jnp.iscomplexobj(current_input) or jnp.iscomplexobj(threshold_input):
            raise TypeError("Battery protocol values must be real-valued.")
        dtype = jnp.result_type(current_input, threshold_input, float)
        currents = current_input.astype(dtype)
        thresholds = threshold_input.astype(dtype)
        if currents.shape != (protocol.current_step_count,):
            raise ValueError(
                "current_amplitudes_a must contain one value per CurrentStepPlan."
            )
        if thresholds.shape != (protocol.guard_count,):
            raise ValueError("stop_thresholds must contain one value per stop guard.")
        if not jnp.issubdtype(currents.dtype, jnp.inexact):
            raise TypeError("Battery protocol values must be floating-point.")
        currents = eqx.error_if(
            currents,
            jnp.any(~jnp.isfinite(currents)),
            "Battery current amplitudes must be finite.",
        )
        thresholds = eqx.error_if(
            thresholds,
            jnp.any(~jnp.isfinite(thresholds)),
            "Battery stop thresholds must be finite.",
        )
        self.current_amplitudes_a = currents
        self.stop_thresholds = thresholds
        self.protocol_id = protocol.protocol_id


__all__ = [
    "BATTERY_CURRENT_INPUT_LAYOUT",
    "PASSIVE_TERMINAL_CONVENTION",
    "BatteryProtocolPlan",
    "BatteryProtocolValues",
    "BatteryStepPlan",
    "BatteryStopGuard",
    "BatteryTerminalConvention",
    "CurrentStepPlan",
    "CurrentStopGuard",
    "RestStepPlan",
    "StoichiometryStopGuard",
    "TemperatureStopGuard",
    "VoltageStopGuard",
]
