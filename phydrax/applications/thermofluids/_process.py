#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import (
    AcausalDAESource,
    DAEComponent,
    DAEConnection,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEPort,
    DAEVariableBlock,
)


class ThermofluidPortKind(StrEnum):
    MATERIAL = "material"
    HEAT = "heat"
    POWER = "power"
    SHAFT = "shaft"


class MaterialFlowDirection(StrEnum):
    INLET = "inlet"
    OUTLET = "outlet"
    BIDIRECTIONAL = "bidirectional"


class HeatFlowOrientation(IntEnum):
    """Sign converting a heat-port flow into heat entering its component."""

    INTO_COMPONENT = 1
    OUT_OF_COMPONENT = -1


class ThermofluidPortSpec(StrictModule):
    name: str = eqx.field(static=True)
    kind: ThermofluidPortKind = eqx.field(static=True)
    direction: MaterialFlowDirection = eqx.field(static=True)
    catalog_id: str = eqx.field(static=True)
    thermodynamics_id: str = eqx.field(static=True)
    state_pair: str = eqx.field(static=True)
    heat_flow_orientation: HeatFlowOrientation = eqx.field(static=True)
    mass_flow_orientation: int = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: ThermofluidPortKind,
        direction: MaterialFlowDirection,
        /,
        *,
        catalog_id: str = "none",
        thermodynamics_id: str = "none",
        state_pair: str = "none",
        heat_flow_orientation: HeatFlowOrientation = HeatFlowOrientation.INTO_COMPONENT,
        mass_flow_orientation: int = 1,
    ) -> None:
        name_value = str(name)
        if not name_value:
            raise ValueError("Thermofluid port name must be non-empty.")
        if not isinstance(kind, ThermofluidPortKind):
            raise TypeError("kind must be ThermofluidPortKind.")
        if not isinstance(direction, MaterialFlowDirection):
            raise TypeError("direction must be MaterialFlowDirection.")
        if not isinstance(heat_flow_orientation, HeatFlowOrientation):
            raise TypeError("heat_flow_orientation must be HeatFlowOrientation.")
        if isinstance(mass_flow_orientation, bool) or mass_flow_orientation not in (
            -1,
            1,
        ):
            raise ValueError("mass_flow_orientation must be -1 or +1.")
        catalog = str(catalog_id)
        thermodynamics = str(thermodynamics_id)
        pair = str(state_pair)
        if kind is ThermofluidPortKind.MATERIAL and (
            not catalog
            or catalog == "none"
            or not thermodynamics
            or thermodynamics == "none"
        ):
            raise ValueError("Material ports require catalog and thermodynamics IDs.")
        self.name = name_value
        self.kind = kind
        self.direction = direction
        self.catalog_id = catalog
        self.thermodynamics_id = thermodynamics
        self.state_pair = pair
        self.heat_flow_orientation = heat_flow_orientation
        self.mass_flow_orientation = int(mass_flow_orientation)
        self.port_id = canonical_fingerprint(
            {
                "kind": kind.value,
                "name": name_value,
                "direction": direction.value,
                "catalog": catalog,
                "thermodynamics": thermodynamics,
                "state_pair": pair,
                "heat_flow_orientation": int(heat_flow_orientation),
                "mass_flow_orientation": int(mass_flow_orientation),
            }
        )


class ThermofluidComponent(StrictModule):
    dae_component: DAEComponent
    ports: tuple[ThermofluidPortSpec, ...]
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        dae_component: DAEComponent,
        ports: tuple[ThermofluidPortSpec, ...],
        /,
        *,
        model_parameters: tuple[tuple[str, float | str], ...] = (),
    ) -> None:
        if not isinstance(dae_component, DAEComponent):
            raise TypeError("dae_component must be DAEComponent.")
        values = tuple(ports)
        if any(not isinstance(value, ThermofluidPortSpec) for value in values):
            raise TypeError("ports must contain ThermofluidPortSpec values.")
        if {value.name for value in values} != {
            value.name for value in dae_component.ports
        }:
            raise ValueError("Typed and DAE port names must match exactly.")
        self.dae_component = dae_component
        self.ports = values
        self.component_id = canonical_fingerprint(
            {
                "kind": "thermofluid-component",
                "name": dae_component.name,
                "ports": [value.port_id for value in values],
                "parameters": [list(value) for value in model_parameters],
            }
        )

    def port(self, name: str, /) -> ThermofluidPortSpec:
        for port in self.ports:
            if port.name == name:
                return port
        raise KeyError(f"Unknown thermofluid port {name!r}.")


class ThermofluidConnection(StrictModule):
    left_component: str = eqx.field(static=True)
    left_port: str = eqx.field(static=True)
    right_component: str = eqx.field(static=True)
    right_port: str = eqx.field(static=True)
    connection_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_component: str,
        left_port: str,
        right_component: str,
        right_port: str,
        /,
    ) -> None:
        values = tuple(
            str(value)
            for value in (
                left_component,
                left_port,
                right_component,
                right_port,
            )
        )
        if any(not value for value in values):
            raise ValueError("Thermofluid connection identifiers must be non-empty.")
        if values[:2] == values[2:]:
            raise ValueError("A thermofluid connection cannot connect a port to itself.")
        self.left_component, self.left_port, self.right_component, self.right_port = (
            values
        )
        self.connection_id = canonical_fingerprint(
            {"kind": "thermofluid-connection", "endpoints": list(values)}
        )


class ThermofluidProcessPlan(StrictModule):
    components: tuple[ThermofluidComponent, ...]
    connections: tuple[ThermofluidConnection, ...]
    source: AcausalDAESource
    process_model_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: tuple[ThermofluidComponent, ...],
        connections: tuple[ThermofluidConnection, ...],
        /,
    ) -> None:
        component_values = tuple(components)
        connection_values = tuple(connections)
        if not component_values or any(
            not isinstance(value, ThermofluidComponent) for value in component_values
        ):
            raise ValueError("ThermofluidProcessPlan requires typed components.")
        if any(
            not isinstance(value, ThermofluidConnection) for value in connection_values
        ):
            raise TypeError("connections must contain ThermofluidConnection values.")
        by_name = {value.dae_component.name: value for value in component_values}
        if len(by_name) != len(component_values):
            raise ValueError("Thermofluid component names must be unique.")
        dae_connections = []
        connected_ports: set[tuple[str, str]] = set()
        for connection in connection_values:
            if (
                connection.left_component not in by_name
                or connection.right_component not in by_name
            ):
                raise ValueError(
                    "Thermofluid connection references an unknown component."
                )
            left = by_name[connection.left_component].port(connection.left_port)
            right = by_name[connection.right_component].port(connection.right_port)
            _validate_connection(left, right)
            endpoints = (
                (connection.left_component, connection.left_port),
                (connection.right_component, connection.right_port),
            )
            if any(endpoint in connected_ports for endpoint in endpoints):
                raise ValueError(
                    "A process port may belong to only one pairwise connection; "
                    "use an explicit control volume or multiport heat body."
                )
            connected_ports.update(endpoints)
            if left.kind is ThermofluidPortKind.HEAT:
                orientations = (
                    int(left.heat_flow_orientation),
                    int(right.heat_flow_orientation),
                )
            elif left.kind is ThermofluidPortKind.MATERIAL:
                orientations = (left.mass_flow_orientation, right.mass_flow_orientation)
            else:
                orientations = (1, -1)
            dae_connections.append(
                DAEConnection(
                    (
                        f"{connection.left_component}.{connection.left_port}",
                        f"{connection.right_component}.{connection.right_port}",
                    ),
                    orientations,
                )
            )
        source = AcausalDAESource(
            tuple(value.dae_component for value in component_values),
            tuple(dae_connections),
        )
        self.components = component_values
        self.connections = connection_values
        self.source = source
        self.process_model_id = canonical_fingerprint(
            {
                "kind": "thermofluid-process",
                "components": [value.component_id for value in component_values],
                "connections": [value.connection_id for value in connection_values],
                "source": source.source_id,
            }
        )


def fixed_material_boundary_component(
    name: str,
    /,
    *,
    pressure: float,
    specific_enthalpy: float,
    mass_flow: float,
    catalog_id: str,
    thermodynamics_id: str,
    direction: MaterialFlowDirection,
) -> ThermofluidComponent:
    """Create a fully prescribed single-stream material boundary."""
    parameters = tuple(float(value) for value in (pressure, specific_enthalpy, mass_flow))
    if any(not np.isfinite(value) for value in parameters) or pressure <= 0.0:
        raise ValueError(
            "Boundary pressure, enthalpy, and flow must be finite and physical."
        )
    variables = tuple(
        DAEVariableBlock(variable, (), 0, scale)
        for variable, scale in (
            ("pressure", max(abs(pressure), 1.0)),
            ("specific_enthalpy", max(abs(specific_enthalpy), 1.0)),
            ("mass_flow", max(abs(mass_flow), 1.0)),
        )
    )

    def prescribed(variable, target):
        def residual(time: Array, jet, args: Any):
            del time, args
            return jet.value(variable) - target

        return residual

    equations = tuple(
        DAEEquationBlock(
            f"prescribe_{variable}",
            prescribed(variable, target),
            (DAEDerivativeIncidence(variable, 0),),
        )
        for variable, target in zip(
            ("pressure", "specific_enthalpy", "mass_flow"), parameters, strict=True
        )
    )
    dae_port = DAEPort(
        "material",
        ("pressure", "specific_enthalpy"),
        ("mass_flow",),
    )
    typed_port = ThermofluidPortSpec(
        "material",
        ThermofluidPortKind.MATERIAL,
        direction,
        catalog_id=catalog_id,
        thermodynamics_id=thermodynamics_id,
        state_pair="pressure-enthalpy",
        mass_flow_orientation=-1,
    )
    return ThermofluidComponent(
        DAEComponent(str(name), variables, equations, (dae_port,)),
        (typed_port,),
        model_parameters=tuple(
            zip(("pressure", "specific_enthalpy", "mass_flow"), parameters, strict=True)
        ),
    )


def isenthalpic_valve_component(
    name: str,
    /,
    *,
    pressure_ratio: float,
    catalog_id: str,
    thermodynamics_id: str,
) -> ThermofluidComponent:
    """Create a fixed-direction isenthalpic pressure-ratio valve."""
    ratio = float(pressure_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("pressure_ratio must be finite in (0, 1].")
    variable_names = (
        "inlet_pressure",
        "inlet_enthalpy",
        "inlet_mass_flow",
        "outlet_pressure",
        "outlet_enthalpy",
        "outlet_mass_flow",
    )
    variables = tuple(DAEVariableBlock(value, (), 0, 1.0) for value in variable_names)

    def pressure_residual(time, jet, args):
        del time, args
        return jet.value("outlet_pressure") - ratio * jet.value("inlet_pressure")

    def enthalpy_residual(time, jet, args):
        del time, args
        return jet.value("outlet_enthalpy") - jet.value("inlet_enthalpy")

    def mass_residual(time, jet, args):
        del time, args
        return jet.value("inlet_mass_flow") + jet.value("outlet_mass_flow")

    equations = (
        DAEEquationBlock(
            "pressure_drop",
            pressure_residual,
            (
                DAEDerivativeIncidence("inlet_pressure"),
                DAEDerivativeIncidence("outlet_pressure"),
            ),
        ),
        DAEEquationBlock(
            "isenthalpic",
            enthalpy_residual,
            (
                DAEDerivativeIncidence("inlet_enthalpy"),
                DAEDerivativeIncidence("outlet_enthalpy"),
            ),
        ),
        DAEEquationBlock(
            "mass_balance",
            mass_residual,
            (
                DAEDerivativeIncidence("inlet_mass_flow"),
                DAEDerivativeIncidence("outlet_mass_flow"),
            ),
        ),
    )
    inlet = DAEPort(
        "inlet",
        ("inlet_pressure", "inlet_enthalpy"),
        ("inlet_mass_flow",),
    )
    outlet = DAEPort(
        "outlet",
        ("outlet_pressure", "outlet_enthalpy"),
        ("outlet_mass_flow",),
    )
    typed = (
        ThermofluidPortSpec(
            "inlet",
            ThermofluidPortKind.MATERIAL,
            MaterialFlowDirection.INLET,
            catalog_id=catalog_id,
            thermodynamics_id=thermodynamics_id,
            state_pair="pressure-enthalpy",
        ),
        ThermofluidPortSpec(
            "outlet",
            ThermofluidPortKind.MATERIAL,
            MaterialFlowDirection.OUTLET,
            catalog_id=catalog_id,
            thermodynamics_id=thermodynamics_id,
            state_pair="pressure-enthalpy",
        ),
    )
    return ThermofluidComponent(
        DAEComponent(str(name), variables, equations, (inlet, outlet)),
        typed,
        model_parameters=(("pressure_ratio", ratio),),
    )


def _validate_connection(
    left: ThermofluidPortSpec,
    right: ThermofluidPortSpec,
) -> None:
    if left.kind is not right.kind:
        raise ValueError("Connected thermofluid ports must have the same kind.")
    if left.kind is ThermofluidPortKind.MATERIAL:
        if (
            left.catalog_id != right.catalog_id
            or left.thermodynamics_id != right.thermodynamics_id
            or left.state_pair != right.state_pair
        ):
            raise ValueError(
                "Connected material ports must share identity and state pair."
            )
        directions = {left.direction, right.direction}
        if directions != {
            MaterialFlowDirection.INLET,
            MaterialFlowDirection.OUTLET,
        }:
            raise ValueError("Material connections require one inlet and one outlet.")


__all__ = [
    "HeatFlowOrientation",
    "MaterialFlowDirection",
    "ThermofluidComponent",
    "ThermofluidConnection",
    "ThermofluidPortKind",
    "ThermofluidPortSpec",
    "ThermofluidProcessPlan",
    "fixed_material_boundary_component",
    "isenthalpic_valve_component",
]
