#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import (
    DAEComponent,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEPort,
    DAEVariableBlock,
)
from ._process import (
    HeatFlowOrientation,
    MaterialFlowDirection,
    ThermofluidComponent,
    ThermofluidPortKind,
    ThermofluidPortSpec,
)


class HeatPortBridge(StrictModule):
    """Convert an external K-or-Celsius/W port into Kelvin and inward watts.

    DAE heat ports themselves always expose absolute Kelvin. The offset belongs
    at the external boundary, never in a potential-equality connection.
    """

    orientation: HeatFlowOrientation = eqx.field(static=True)
    temperature_offset: float = eqx.field(static=True)

    def __init__(
        self,
        orientation: HeatFlowOrientation,
        /,
        *,
        temperature_offset: float = 0.0,
    ) -> None:
        if not isinstance(orientation, HeatFlowOrientation):
            raise TypeError("orientation must be HeatFlowOrientation.")
        offset = float(temperature_offset)
        if not np.isfinite(offset):
            raise ValueError("temperature_offset must be finite.")
        self.orientation = orientation
        self.temperature_offset = offset

    def temperature_kelvin(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(value) + self.temperature_offset

    def heat_into_component(self, value: ArrayLike, /) -> Array:
        return int(self.orientation) * jnp.asarray(value)


class HeatConversionEvaluation(StrictModule):
    """Heating-side balance; positive environment heat is extracted heat."""

    electrical_power: Array
    delivered_heat: Array
    environment_heat: Array
    successful: Array


class HeatConversionLaw(StrictModule, abc.ABC):
    """Native array device law, independent of building or planning compilers."""

    __strict_abstract__ = True

    law_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self,
        electrical_power: ArrayLike,
        source_temperature: ArrayLike,
        supply_temperature: ArrayLike,
        /,
    ) -> HeatConversionEvaluation:
        raise NotImplementedError


def _conversion_evaluation(power, source, supply, factor, *, heat_pump):
    power, source, supply, factor = jnp.broadcast_arrays(
        jnp.asarray(power), jnp.asarray(source), jnp.asarray(supply), jnp.asarray(factor)
    )
    delivered = factor * power
    environment = delivered - power
    valid = (
        jnp.isfinite(power)
        & jnp.isfinite(factor)
        & (power >= 0)
        & jnp.isfinite(source)
        & (source > 0)
        & jnp.isfinite(supply)
        & (supply > 0)
        & jnp.isfinite(delivered)
        & jnp.isfinite(environment)
    )
    if heat_pump:
        valid = valid & (factor >= 1.0)
        # For positive temperature lift the constant COP must not exceed Carnot.
        # No clipping: unsupported operating points retain their energy balance.
        valid = valid & ((power == 0) | (factor * (supply - source) <= supply))
    else:
        valid = valid & (factor > 0.0) & (factor <= 1.0)
    return HeatConversionEvaluation(power, delivered, environment, valid)


class ConstantCOPHeatPumpLaw(HeatConversionLaw):
    coefficient_of_performance: Array

    def __init__(self, coefficient_of_performance: ArrayLike) -> None:
        cop = jnp.asarray(coefficient_of_performance)
        if cop.ndim != 0:
            raise ValueError("coefficient_of_performance must be scalar.")
        if not jnp.issubdtype(cop.dtype, jnp.inexact):
            cop = cop.astype(float)
        self.coefficient_of_performance = eqx.error_if(
            cop,
            ~jnp.isfinite(cop) | (cop < 1.0),
            "Heating coefficient_of_performance must be finite >= 1.",
        )
        self.law_id = canonical_fingerprint({"kind": "constant-cop-heat-pump"})

    def evaluate(
        self, electrical_power, source_temperature, supply_temperature, /
    ) -> HeatConversionEvaluation:
        return _conversion_evaluation(
            electrical_power,
            source_temperature,
            supply_temperature,
            self.coefficient_of_performance,
            heat_pump=True,
        )


class ResistiveHeatingLaw(HeatConversionLaw):
    efficiency: Array

    def __init__(self, efficiency: ArrayLike = 1.0) -> None:
        efficiency_value = jnp.asarray(efficiency)
        if efficiency_value.ndim != 0:
            raise ValueError("efficiency must be scalar.")
        if not jnp.issubdtype(efficiency_value.dtype, jnp.inexact):
            efficiency_value = efficiency_value.astype(float)
        self.efficiency = eqx.error_if(
            efficiency_value,
            ~jnp.isfinite(efficiency_value)
            | (efficiency_value <= 0.0)
            | (efficiency_value > 1.0),
            "Resistance heating efficiency must be finite in (0, 1].",
        )
        self.law_id = canonical_fingerprint({"kind": "resistive-heating"})

    def evaluate(
        self, electrical_power, source_temperature, supply_temperature, /
    ) -> HeatConversionEvaluation:
        return _conversion_evaluation(
            electrical_power,
            source_temperature,
            supply_temperature,
            self.efficiency,
            heat_pump=False,
        )


def _heat_port(name, temperature, flow, orientation):
    return (
        DAEPort(name, (temperature,), (flow,)),
        ThermofluidPortSpec(
            name,
            ThermofluidPortKind.HEAT,
            MaterialFlowDirection.BIDIRECTIONAL,
            state_pair="temperature-heat-flow",
            heat_flow_orientation=orientation,
        ),
    )


def thermal_capacitance_component(
    name: str,
    /,
    *,
    heat_capacity: float,
    port_count: int = 1,
    orientation: HeatFlowOrientation = HeatFlowOrientation.INTO_COMPONENT,
) -> ThermofluidComponent:
    """A lumped body: C dT/dt = sum(Q_into), with C in J/K and T in K.

    Initial temperature is a differential initial condition, not a boundary
    equation. Ports are named ``heat`` for one port, otherwise ``heat_0``, etc.
    """
    capacity = float(heat_capacity)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("heat_capacity must be finite and positive.")
    if not isinstance(port_count, int) or isinstance(port_count, bool) or port_count < 1:
        raise ValueError("port_count must be a positive integer.")
    flow_names = tuple(f"heat_flow_{index}" for index in range(port_count))
    ports = tuple(
        _heat_port(
            "heat" if port_count == 1 else f"heat_{index}",
            "temperature",
            flow,
            orientation,
        )
        for index, flow in enumerate(flow_names)
    )

    def energy_balance(time, jet, args):
        del time, args
        return capacity * jet.value("temperature", 1) - int(orientation) * sum(
            jet.value(flow) for flow in flow_names
        )

    return ThermofluidComponent(
        DAEComponent(
            name,
            (DAEVariableBlock("temperature", (), 1, 300.0),)
            + tuple(DAEVariableBlock(flow, (), 0, 1.0) for flow in flow_names),
            (
                DAEEquationBlock(
                    "energy_balance",
                    energy_balance,
                    (DAEDerivativeIncidence("temperature", 1),)
                    + tuple(DAEDerivativeIncidence(flow) for flow in flow_names),
                ),
            ),
            tuple(port[0] for port in ports),
        ),
        tuple(port[1] for port in ports),
        model_parameters=(("heat_capacity", capacity),),
    )


def thermal_conductor_component(
    name: str,
    /,
    *,
    conductance: float,
    left_orientation: HeatFlowOrientation = HeatFlowOrientation.INTO_COMPONENT,
    right_orientation: HeatFlowOrientation = HeatFlowOrientation.INTO_COMPONENT,
) -> ThermofluidComponent:
    """Massless heat exchange: Q_left,in = G (T_left - T_right)."""
    conductance_value = float(conductance)
    if not np.isfinite(conductance_value) or conductance_value <= 0:
        raise ValueError("conductance must be finite and positive.")
    left = _heat_port("left", "left_temperature", "left_heat_flow", left_orientation)
    right = _heat_port("right", "right_temperature", "right_heat_flow", right_orientation)

    def constitutive(time, jet, args):
        del time, args
        return int(left_orientation) * jet.value("left_heat_flow") - conductance_value * (
            jet.value("left_temperature") - jet.value("right_temperature")
        )

    def balance(time, jet, args):
        del time, args
        return int(left_orientation) * jet.value("left_heat_flow") + int(
            right_orientation
        ) * jet.value("right_heat_flow")

    return ThermofluidComponent(
        DAEComponent(
            name,
            tuple(
                DAEVariableBlock(
                    variable, (), 0, 300.0 if "temperature" in variable else 1.0
                )
                for variable in (
                    "left_temperature",
                    "left_heat_flow",
                    "right_temperature",
                    "right_heat_flow",
                )
            ),
            (
                DAEEquationBlock(
                    "heat_transfer",
                    constitutive,
                    tuple(
                        DAEDerivativeIncidence(variable)
                        for variable in (
                            "left_heat_flow",
                            "left_temperature",
                            "right_temperature",
                        )
                    ),
                ),
                DAEEquationBlock(
                    "energy_balance",
                    balance,
                    (
                        DAEDerivativeIncidence("left_heat_flow"),
                        DAEDerivativeIncidence("right_heat_flow"),
                    ),
                ),
            ),
            (left[0], right[0]),
        ),
        (left[1], right[1]),
        model_parameters=(("conductance", conductance_value),),
    )


def temperature_boundary_component(
    name: str,
    /,
    *,
    temperature: float,
    orientation: HeatFlowOrientation = HeatFlowOrientation.INTO_COMPONENT,
) -> ThermofluidComponent:
    """Infinite reservoir at prescribed Kelvin; its heat flow is solved."""
    target = float(temperature)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("temperature must be finite positive Kelvin.")
    port, typed = _heat_port("heat", "temperature", "heat_flow", orientation)

    def residual(time, jet, args):
        del time, args
        return jet.value("temperature") - target

    return ThermofluidComponent(
        DAEComponent(
            name,
            (
                DAEVariableBlock("temperature", (), 0, target),
                DAEVariableBlock("heat_flow", (), 0, 1.0),
            ),
            (
                DAEEquationBlock(
                    "temperature_boundary",
                    residual,
                    (DAEDerivativeIncidence("temperature"),),
                ),
            ),
            (port,),
        ),
        (typed,),
        model_parameters=(("temperature", target),),
    )


class _HeatDeliveryResidual(StrictModule):
    law: HeatConversionLaw
    electrical_power: Array

    def __call__(self, time, jet, args):
        del time, args
        result = self.law.evaluate(
            self.electrical_power,
            jet.value("source_temperature"),
            jet.value("supply_temperature"),
        )
        return jet.value("supply_heat_flow") + result.delivered_heat


class _HeatConversionBalance(StrictModule):
    electrical_power: Array

    def __call__(self, time, jet, args):
        del time, args
        return (
            self.electrical_power
            + jet.value("supply_heat_flow")
            + jet.value("environment_heat_flow")
        )


def heat_conversion_component(
    name: str,
    /,
    *,
    law: HeatConversionLaw,
    electrical_power: ArrayLike,
) -> ThermofluidComponent:
    """Two thermal terminals and prescribed electrical input, all in watts.

    ``supply`` exports useful heat; ``environment`` extracts source heat for a
    heat pump or rejects resistance losses. Both port flows are positive inward.
    Evaluate the same law on solved temperatures to inspect physical support;
    numerical DAE convergence alone does not certify a constitutive domain.
    """
    if not isinstance(law, HeatConversionLaw):
        raise TypeError("law must implement HeatConversionLaw.")
    power = jnp.asarray(electrical_power)
    if power.ndim != 0:
        raise ValueError("electrical_power must be scalar.")
    if not jnp.issubdtype(power.dtype, jnp.inexact):
        power = power.astype(float)
    power = eqx.error_if(
        power,
        ~jnp.isfinite(power) | (power < 0.0),
        "electrical_power must be finite and nonnegative.",
    )
    inward = HeatFlowOrientation.INTO_COMPONENT
    supply = _heat_port("supply", "supply_temperature", "supply_heat_flow", inward)
    environment = _heat_port(
        "environment", "source_temperature", "environment_heat_flow", inward
    )

    delivery = _HeatDeliveryResidual(law, power)
    balance = _HeatConversionBalance(power)

    return ThermofluidComponent(
        DAEComponent(
            name,
            tuple(
                DAEVariableBlock(variable, (), 0, 1.0)
                for variable in (
                    "supply_temperature",
                    "supply_heat_flow",
                    "source_temperature",
                    "environment_heat_flow",
                )
            ),
            (
                DAEEquationBlock(
                    "heat_delivery",
                    delivery,
                    tuple(
                        DAEDerivativeIncidence(variable)
                        for variable in (
                            "supply_heat_flow",
                            "source_temperature",
                            "supply_temperature",
                        )
                    ),
                ),
                DAEEquationBlock(
                    "energy_balance",
                    balance,
                    (
                        DAEDerivativeIncidence("supply_heat_flow"),
                        DAEDerivativeIncidence("environment_heat_flow"),
                    ),
                ),
            ),
            (supply[0], environment[0]),
        ),
        (supply[1], environment[1]),
        model_parameters=(("law_id", law.law_id),),
    )


__all__ = [
    "ConstantCOPHeatPumpLaw",
    "HeatConversionEvaluation",
    "HeatConversionLaw",
    "HeatPortBridge",
    "ResistiveHeatingLaw",
    "heat_conversion_component",
    "temperature_boundary_component",
    "thermal_capacitance_component",
    "thermal_conductor_component",
]
