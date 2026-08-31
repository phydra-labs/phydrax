#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._mna import AbstractMNAComponent, MNAStamp


CircuitVariableRole: TypeAlias = Literal["differential", "algebraic"]


class CircuitElementStateLayout(StrictModule):
    size: int = eqx.field(static=True)
    roles: tuple[CircuitVariableRole, ...] = eqx.field(static=True)
    state_scale: Array
    rate_scale: Array
    residual_scale: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        roles: Sequence[CircuitVariableRole] = (),
        /,
        *,
        state_scale: ArrayLike | None = None,
        rate_scale: ArrayLike | None = None,
        residual_scale: ArrayLike | None = None,
    ):
        roles_ = tuple(roles)
        if any(value not in ("differential", "algebraic") for value in roles_):
            raise ValueError("Unknown circuit element state role.")
        size = len(roles_)

        def scale(value: ArrayLike | None, name: str) -> Array:
            result = (
                jnp.ones((size,)) if value is None else jnp.asarray(value, dtype=float)
            )
            if result.shape != (size,) or bool(
                jnp.any(~jnp.isfinite(result)) | jnp.any(result <= 0.0)
            ):
                raise ValueError(f"{name} must be a finite positive state-sized vector.")
            return result

        self.size = size
        self.roles = roles_
        self.state_scale = scale(state_scale, "state_scale")
        self.rate_scale = scale(rate_scale, "rate_scale")
        self.residual_scale = scale(residual_scale, "residual_scale")
        self.layout_id = canonical_fingerprint(
            {
                "kind": "circuit-element-state-layout",
                "roles": roles_,
                "size": size,
            }
        )


class CircuitElementEvaluation(StrictModule):
    terminal_currents: Array
    auxiliary_residual: Array

    def __init__(self, terminal_currents: ArrayLike, auxiliary_residual: ArrayLike, /):
        currents = jnp.asarray(terminal_currents)
        residual = jnp.asarray(auxiliary_residual)
        if currents.ndim != 1 or residual.ndim != 1:
            raise ValueError("Circuit element evaluation values must be vectors.")
        dtype = jnp.result_type(currents, residual, jnp.float64)
        self.terminal_currents = currents.astype(dtype)
        self.auxiliary_residual = residual.astype(dtype)


class AbstractImplicitCircuitLaw(StrictModule):
    terminal_count: int = eqx.field(static=True)
    voltage_rate_dependent: bool = eqx.field(static=True)
    state_layout: CircuitElementStateLayout
    law_id: str = eqx.field(static=True)

    @abstractmethod
    def evaluate(
        self,
        time: Array,
        terminal_voltages: Array,
        terminal_voltage_rates: Array,
        state: Array,
        state_rate: Array,
        inputs: Any,
        args: Any,
        /,
    ) -> CircuitElementEvaluation:
        raise NotImplementedError


class AbstractCircuitEnergyLaw(StrictModule):
    @abstractmethod
    def stored_energy(
        self, terminal_voltages: Array, state: Array, /, *, args: Any = None
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def dissipated_power(
        self,
        terminal_voltages: Array,
        terminal_currents: Array,
        state: Array,
        /,
        *,
        args: Any = None,
    ) -> Array:
        raise NotImplementedError


class AbstractCircuitNoiseLaw(StrictModule):
    @abstractmethod
    def spectral_factor(
        self,
        angular_frequency: Array,
        terminal_voltages: Array,
        state: Array,
        /,
        *,
        temperature: Array,
        args: Any = None,
    ) -> Array:
        raise NotImplementedError


class CircuitElement(AbstractMNAComponent):
    """Composed frequency, implicit, energy, and noise device capabilities."""

    implicit_law: AbstractImplicitCircuitLaw
    frequency_law: AbstractMNAComponent | None
    energy_law: AbstractCircuitEnergyLaw | None
    noise_law: AbstractCircuitNoiseLaw | None
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        implicit_law: AbstractImplicitCircuitLaw,
        /,
        *,
        frequency_law: AbstractMNAComponent | None = None,
        energy_law: AbstractCircuitEnergyLaw | None = None,
        noise_law: AbstractCircuitNoiseLaw | None = None,
        element_id: str,
    ):
        if not isinstance(implicit_law, AbstractImplicitCircuitLaw):
            raise TypeError("implicit_law must be AbstractImplicitCircuitLaw.")
        if frequency_law is not None and not isinstance(
            frequency_law, AbstractMNAComponent
        ):
            raise TypeError("frequency_law must be AbstractMNAComponent or None.")
        if frequency_law is not None and (
            frequency_law.terminal_count != implicit_law.terminal_count
        ):
            raise ValueError("Frequency and implicit terminal counts must match.")
        if energy_law is not None and not isinstance(
            energy_law, AbstractCircuitEnergyLaw
        ):
            raise TypeError("energy_law must be AbstractCircuitEnergyLaw or None.")
        if noise_law is not None and not isinstance(noise_law, AbstractCircuitNoiseLaw):
            raise TypeError("noise_law must be AbstractCircuitNoiseLaw or None.")
        identifier = str(element_id)
        if not identifier:
            raise ValueError("element_id must be non-empty.")
        self.implicit_law = implicit_law
        self.frequency_law = frequency_law
        self.energy_law = energy_law
        self.noise_law = noise_law
        self.element_id = identifier

    @property
    def terminal_count(self) -> int:
        return self.implicit_law.terminal_count

    @property
    def auxiliary_count(self) -> int:
        if self.frequency_law is None:
            return 0
        return self.frequency_law.auxiliary_count

    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        if self.frequency_law is None:
            raise ValueError(
                f"Circuit element {self.element_id!r} has no frequency-domain law."
            )
        return self.frequency_law.evaluate(angular_frequency)


class TwoTerminalConductanceLaw(AbstractImplicitCircuitLaw):
    conductance: Array

    def __init__(self, conductance: ArrayLike, /, *, law_id: str = "conductance"):
        value = jnp.asarray(conductance, dtype=float)
        if value.shape != () or bool(~jnp.isfinite(value)) or bool(value < 0.0):
            raise ValueError("conductance must be one finite nonnegative scalar.")
        self.conductance = value
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout()
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state, state_rate, inputs, args
        drop = terminal_voltages[0] - terminal_voltages[1]
        current = self.conductance * drop
        return CircuitElementEvaluation(jnp.asarray([current, -current]), jnp.zeros((0,)))


class TwoTerminalCapacitanceLaw(AbstractImplicitCircuitLaw):
    capacitance: Array

    def __init__(self, capacitance: ArrayLike, /, *, law_id: str = "capacitance"):
        value = jnp.asarray(capacitance, dtype=float)
        if value.shape != () or bool(~jnp.isfinite(value)) or bool(value <= 0.0):
            raise ValueError("capacitance must be one finite positive scalar.")
        self.capacitance = value
        self.terminal_count = 2
        self.voltage_rate_dependent = True
        self.state_layout = CircuitElementStateLayout()
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltages, state, state_rate, inputs, args
        current = self.capacitance * (
            terminal_voltage_rates[0] - terminal_voltage_rates[1]
        )
        return CircuitElementEvaluation(jnp.asarray([current, -current]), jnp.zeros((0,)))


class TwoTerminalInductanceLaw(AbstractImplicitCircuitLaw):
    inductance: Array

    def __init__(self, inductance: ArrayLike, /, *, law_id: str = "inductance"):
        value = jnp.asarray(inductance, dtype=float)
        if value.shape != () or bool(~jnp.isfinite(value)) or bool(value <= 0.0):
            raise ValueError("inductance must be one finite positive scalar.")
        self.inductance = value
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout(("differential",))
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, inputs, args
        current = state[0]
        residual = self.inductance * state_rate[0] - (
            terminal_voltages[0] - terminal_voltages[1]
        )
        return CircuitElementEvaluation(
            jnp.asarray([current, -current]), jnp.asarray([residual])
        )


class IndependentCurrentSourceLaw(AbstractImplicitCircuitLaw):
    current: Array
    input_key: str | None = eqx.field(static=True)

    def __init__(
        self,
        current: ArrayLike,
        /,
        *,
        input_key: str | None = None,
        law_id: str = "current-source",
    ):
        value = jnp.asarray(current, dtype=float)
        if value.shape != () or bool(~jnp.isfinite(value)):
            raise ValueError("current must be one finite scalar.")
        key = None if input_key is None else str(input_key)
        if key == "":
            raise ValueError("input_key must be non-empty when supplied.")
        self.current, self.input_key = value, key
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout()
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltages, terminal_voltage_rates, state, state_rate, args
        current = self.current
        if self.input_key is not None:
            if not isinstance(inputs, dict) or self.input_key not in inputs:
                raise ValueError(f"Current source requires input {self.input_key!r}.")
            current = current * jnp.asarray(inputs[self.input_key])
        return CircuitElementEvaluation(jnp.asarray([current, -current]), jnp.zeros((0,)))


class IndependentVoltageSourceLaw(AbstractImplicitCircuitLaw):
    voltage: Array
    input_key: str | None = eqx.field(static=True)

    def __init__(
        self,
        voltage: ArrayLike,
        /,
        *,
        input_key: str | None = None,
        law_id: str = "voltage-source",
    ):
        value = jnp.asarray(voltage, dtype=float)
        if value.shape != () or bool(~jnp.isfinite(value)):
            raise ValueError("voltage must be one finite scalar.")
        key = None if input_key is None else str(input_key)
        if key == "":
            raise ValueError("input_key must be non-empty when supplied.")
        self.voltage, self.input_key = value, key
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout(("algebraic",))
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state_rate, args
        voltage = self.voltage
        if self.input_key is not None:
            if not isinstance(inputs, dict) or self.input_key not in inputs:
                raise ValueError(f"Voltage source requires input {self.input_key!r}.")
            voltage = voltage * jnp.asarray(inputs[self.input_key])
        current = state[0]
        residual = terminal_voltages[0] - terminal_voltages[1] - voltage
        return CircuitElementEvaluation(
            jnp.asarray([current, -current]), jnp.asarray([residual])
        )


class ExponentialDiodeLaw(AbstractImplicitCircuitLaw):
    saturation_current: Array
    thermal_voltage: Array

    def __init__(
        self,
        saturation_current: ArrayLike,
        thermal_voltage: ArrayLike,
        /,
        *,
        law_id: str = "exponential-diode",
    ):
        saturation = jnp.asarray(saturation_current, dtype=float)
        thermal = jnp.asarray(thermal_voltage, dtype=float)
        if (
            saturation.shape != ()
            or thermal.shape != ()
            or bool(~jnp.isfinite(saturation))
            or bool(~jnp.isfinite(thermal))
            or bool(saturation <= 0.0)
            or bool(thermal <= 0.0)
        ):
            raise ValueError("Diode parameters must be finite positive scalars.")
        self.saturation_current, self.thermal_voltage = saturation, thermal
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout()
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state, state_rate, inputs, args
        voltage = terminal_voltages[0] - terminal_voltages[1]
        current = self.saturation_current * jnp.expm1(voltage / self.thermal_voltage)
        return CircuitElementEvaluation(jnp.asarray([current, -current]), jnp.zeros((0,)))


class SmoothSwitchLaw(AbstractImplicitCircuitLaw):
    on_conductance: Array
    off_conductance: Array
    sharpness: Array
    control_key: str = eqx.field(static=True)

    def __init__(
        self,
        on_conductance: ArrayLike,
        off_conductance: ArrayLike,
        /,
        *,
        control_key: str,
        sharpness: ArrayLike = 20.0,
        law_id: str = "smooth-switch",
    ):
        on = jnp.asarray(on_conductance, dtype=float)
        off = jnp.asarray(off_conductance, dtype=float)
        sharp = jnp.asarray(sharpness, dtype=float)
        key = str(control_key)
        if (
            on.shape != ()
            or off.shape != ()
            or sharp.shape != ()
            or bool(jnp.any(~jnp.isfinite(jnp.asarray([on, off, sharp]))))
            or bool(on <= 0.0)
            or bool(off < 0.0)
            or bool(sharp <= 0.0)
            or not key
        ):
            raise ValueError("Smooth switch parameters are invalid.")
        self.on_conductance, self.off_conductance, self.sharpness = on, off, sharp
        self.control_key = key
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout()
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state, state_rate, args
        if not isinstance(inputs, dict) or self.control_key not in inputs:
            raise ValueError(f"Smooth switch requires input {self.control_key!r}.")
        control = jnp.asarray(inputs[self.control_key])
        gate = jax.nn.sigmoid(self.sharpness * control)
        conductance = self.off_conductance + gate * (
            self.on_conductance - self.off_conductance
        )
        voltage = terminal_voltages[0] - terminal_voltages[1]
        current = conductance * voltage
        return CircuitElementEvaluation(jnp.asarray([current, -current]), jnp.zeros((0,)))


class VoltageControlledCurrentLaw(AbstractImplicitCircuitLaw):
    """Four-terminal transconductance with output terminals followed by control."""

    transconductance: Array

    def __init__(self, transconductance: ArrayLike, /, *, law_id: str = "vccs"):
        value = jnp.asarray(transconductance, dtype=float)
        if value.shape != () or bool(~jnp.isfinite(value)):
            raise ValueError("transconductance must be one finite scalar.")
        self.transconductance = value
        self.terminal_count = 4
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout()
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state, state_rate, inputs, args
        control = terminal_voltages[2] - terminal_voltages[3]
        current = self.transconductance * control
        return CircuitElementEvaluation(
            jnp.asarray([current, -current, 0.0, 0.0]), jnp.zeros((0,))
        )


class VoltageControlledVoltageLaw(AbstractImplicitCircuitLaw):
    """Four-terminal voltage gain with one algebraic output branch current."""

    gain: Array

    def __init__(self, gain: ArrayLike, /, *, law_id: str = "vcvs"):
        value = jnp.asarray(gain, dtype=float)
        if value.shape != () or bool(~jnp.isfinite(value)):
            raise ValueError("gain must be one finite scalar.")
        self.gain = value
        self.terminal_count = 4
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout(("algebraic",))
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state_rate, inputs, args
        current = state[0]
        output = terminal_voltages[0] - terminal_voltages[1]
        control = terminal_voltages[2] - terminal_voltages[3]
        return CircuitElementEvaluation(
            jnp.asarray([current, -current, 0.0, 0.0]),
            jnp.asarray([output - self.gain * control]),
        )


class IdealTransformerLaw(AbstractImplicitCircuitLaw):
    """Lossless four-terminal ideal transformer with two branch currents."""

    turns_ratio: Array

    def __init__(self, turns_ratio: ArrayLike, /, *, law_id: str = "ideal-transformer"):
        ratio = jnp.asarray(turns_ratio, dtype=float)
        if ratio.shape != () or bool(~jnp.isfinite(ratio)) or bool(ratio == 0.0):
            raise ValueError("turns_ratio must be one finite nonzero scalar.")
        self.turns_ratio = ratio
        self.terminal_count = 4
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout(("algebraic", "algebraic"))
        self.law_id = str(law_id)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state_rate, inputs, args
        primary_current, secondary_current = state
        primary_voltage = terminal_voltages[0] - terminal_voltages[1]
        secondary_voltage = terminal_voltages[2] - terminal_voltages[3]
        return CircuitElementEvaluation(
            jnp.asarray(
                [
                    primary_current,
                    -primary_current,
                    secondary_current,
                    -secondary_current,
                ]
            ),
            jnp.asarray(
                [
                    primary_voltage - self.turns_ratio * secondary_voltage,
                    secondary_current + self.turns_ratio * primary_current,
                ]
            ),
        )


def implicit_law_for(component: AbstractMNAComponent, /) -> AbstractImplicitCircuitLaw:
    """Return the causal law carried by one built-in or composed element."""
    from ._components import Capacitor, Inductor, Resistor

    if isinstance(component, CircuitElement):
        return component.implicit_law
    if isinstance(component, Resistor):
        return TwoTerminalConductanceLaw(
            1.0 / component.resistance,
            law_id=f"{component.component_id}/implicit",
        )
    if isinstance(component, Capacitor):
        return TwoTerminalCapacitanceLaw(
            component.capacitance,
            law_id=f"{component.component_id}/implicit",
        )
    if isinstance(component, Inductor):
        return TwoTerminalInductanceLaw(
            component.inductance,
            law_id=f"{component.component_id}/implicit",
        )
    raise ValueError(
        f"Component {type(component).__name__} has no causal implicit circuit law."
    )


__all__ = [
    "AbstractCircuitEnergyLaw",
    "AbstractCircuitNoiseLaw",
    "AbstractImplicitCircuitLaw",
    "CircuitElement",
    "CircuitElementEvaluation",
    "CircuitElementStateLayout",
    "CircuitVariableRole",
    "ExponentialDiodeLaw",
    "IndependentCurrentSourceLaw",
    "IndependentVoltageSourceLaw",
    "IdealTransformerLaw",
    "implicit_law_for",
    "SmoothSwitchLaw",
    "TwoTerminalCapacitanceLaw",
    "TwoTerminalConductanceLaw",
    "TwoTerminalInductanceLaw",
    "VoltageControlledCurrentLaw",
    "VoltageControlledVoltageLaw",
]
