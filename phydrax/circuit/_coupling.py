#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..dynamics._differential_algebraic import DAEStructure, DifferentialAlgebraicSystem
from ..dynamics._linear_descriptor import LinearDescriptorSystem
from ._dae import PreparedCircuitDAE
from ._models import AbstractScatteringComponent, ScatteringResponse
from ._ports import WavePort


class FieldPortModel(AbstractScatteringComponent):
    """Typed field-port scattering action with an optional causal realization."""

    component: AbstractScatteringComponent
    descriptor: LinearDescriptorSystem | None
    _ports: tuple[WavePort, ...]
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: AbstractScatteringComponent,
        /,
        *,
        descriptor: LinearDescriptorSystem | None = None,
        model_id: str,
    ):
        if not isinstance(component, AbstractScatteringComponent):
            raise TypeError("component must be AbstractScatteringComponent.")
        if descriptor is not None and not isinstance(descriptor, LinearDescriptorSystem):
            raise TypeError("descriptor must be LinearDescriptorSystem or None.")
        identifier = str(model_id)
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.component = component
        self.descriptor = descriptor
        self._ports = component.ports
        self.model_id = identifier

    @property
    def ports(self) -> tuple[WavePort, ...]:
        return self._ports

    def evaluate(self, angular_frequency: ArrayLike, /) -> ScatteringResponse:
        return self.component.evaluate(angular_frequency)


class ElectrothermalDiagnostics(StrictModule):
    circuit_residual_norm: Array
    thermal_residual: Array
    heat_power: Array
    ambient_loss: Array
    finite: Array


class PreparedElectrothermalCircuit(StrictModule):
    circuit: PreparedCircuitDAE
    system: DifferentialAlgebraicSystem
    heat_capacity: Array
    thermal_conductance: Array
    ambient_temperature: Array
    coupling_id: str = eqx.field(static=True)

    def initialize(
        self,
        circuit_state: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> Array:
        state = jnp.asarray(circuit_state, dtype=float)
        kelvin = jnp.asarray(temperature, dtype=float)
        if state.shape != (self.circuit.plan.layout.size,) or kelvin.shape != ():
            raise ValueError("Electrothermal initial values have wrong shapes.")
        if bool(jnp.any(~jnp.isfinite(state))) or bool(~jnp.isfinite(kelvin)):
            raise ValueError("Electrothermal initial values must be finite.")
        return jnp.concatenate((state, kelvin[None]))

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        state_rate: ArrayLike,
        args: Any = None,
        /,
    ) -> ElectrothermalDiagnostics:
        value, rate = jnp.asarray(state), jnp.asarray(state_rate)
        residual = self.system.evaluate(time, value, rate, args)
        temperature = value[-1]
        heat = _heat_value(args, jnp.asarray(time), value[:-1], temperature)
        loss = self.thermal_conductance * (temperature - self.ambient_temperature)
        return ElectrothermalDiagnostics(
            jnp.linalg.norm(residual[:-1]),
            residual[-1],
            heat,
            loss,
            jnp.all(jnp.isfinite(residual)),
        )


class _ElectrothermalResidual(StrictModule):
    circuit: PreparedCircuitDAE
    heat_capacity: Array
    thermal_conductance: Array
    ambient_temperature: Array

    def __call__(self, time: Array, state: Array, state_rate: Array, args: Any, /):
        circuit_state, temperature = state[:-1], state[-1]
        circuit_rate, temperature_rate = state_rate[:-1], state_rate[-1]
        circuit_args = {
            "inputs": args["inputs"]
            if isinstance(args, dict) and "inputs" in args
            else None,
            "args": {
                "temperature": temperature,
                "user": args["args"]
                if isinstance(args, dict) and "args" in args
                else args,
            },
        }
        circuit_residual = self.circuit.system.evaluate(
            time, circuit_state, circuit_rate, circuit_args
        )
        heat = _heat_value(args, time, circuit_state, temperature)
        thermal_residual = (
            self.heat_capacity * temperature_rate
            - heat
            + self.thermal_conductance * (temperature - self.ambient_temperature)
        )
        return jnp.concatenate((circuit_residual, thermal_residual[None]))


def _heat_value(
    args: Any, time: Array, circuit_state: Array, temperature: Array, /
) -> Array:
    if not isinstance(args, dict) or "heat_power" not in args:
        raise ValueError("Electrothermal execution requires a heat_power callable.")
    function = args["heat_power"]
    if not callable(function):
        raise TypeError("heat_power must be callable.")
    value = jnp.asarray(function(time, circuit_state, temperature, args))
    if value.shape != ():
        raise ValueError("heat_power must return one scalar.")
    return value


def prepare_electrothermal_circuit(
    circuit: PreparedCircuitDAE,
    heat_capacity: ArrayLike,
    thermal_conductance: ArrayLike,
    ambient_temperature: ArrayLike,
    /,
) -> PreparedElectrothermalCircuit:
    if not isinstance(circuit, PreparedCircuitDAE):
        raise TypeError("circuit must be PreparedCircuitDAE.")
    capacity = jnp.asarray(heat_capacity, dtype=float)
    conductance = jnp.asarray(thermal_conductance, dtype=float)
    ambient = jnp.asarray(ambient_temperature, dtype=float)
    if (
        capacity.shape != ()
        or conductance.shape != ()
        or ambient.shape != ()
        or bool(jnp.any(~jnp.isfinite(jnp.asarray([capacity, conductance, ambient]))))
        or bool(capacity <= 0.0)
        or bool(conductance < 0.0)
        or bool(ambient <= 0.0)
    ):
        raise ValueError("Electrothermal parameters must be finite physical scalars.")
    roles = circuit.plan.layout.roles + ("differential",)
    state_scale = jnp.concatenate((circuit.plan.state_scale, ambient[None]))
    rate_scale = jnp.concatenate((circuit.plan.rate_scale, ambient[None]))
    residual_scale = jnp.concatenate(
        (circuit.plan.residual_scale, (conductance * ambient + 1.0)[None])
    )
    coupling_id = canonical_fingerprint(
        {
            "kind": "prepared-electrothermal-circuit",
            "circuit": circuit.prepared_id,
        }
    )
    system = DifferentialAlgebraicSystem(
        _ElectrothermalResidual(circuit, capacity, conductance, ambient),
        state_shape=(circuit.plan.layout.size + 1,),
        structure=DAEStructure(roles, equation_roles=roles, component_axis=-1),
        state_scale=state_scale,
        state_rate_scale=rate_scale,
        residual_scale=residual_scale,
        system_id=coupling_id,
    )
    return PreparedElectrothermalCircuit(
        circuit, system, capacity, conductance, ambient, coupling_id
    )


__all__ = [
    "ElectrothermalDiagnostics",
    "FieldPortModel",
    "PreparedElectrothermalCircuit",
    "prepare_electrothermal_circuit",
]
