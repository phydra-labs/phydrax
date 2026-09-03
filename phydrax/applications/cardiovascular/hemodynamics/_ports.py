#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.lattice_boltzmann import LatticeBoltzmannDiscretization
from .._quantities import cardiovascular_quantity
from ..circulation._components import PressureFlowComponent
from ._domain import FixedWallLumenRegion


_PRESSURE = cardiovascular_quantity("pressure")
_FLOW = cardiovascular_quantity("volumetric_flow_rate")


class TerminalDirection(IntEnum):
    """Declared positive circulation direction at a 3D terminal."""

    INTO_LUMEN = -1
    OUT_OF_LUMEN = 1

    @property
    def outward_sign(self) -> int:
        """Map positive directed flow to positive outward surface flux."""

        return int(self.value)


class TerminalFace(StrictModule, NonTrainableState):
    """Axis-aligned exterior face and its explicit circulation orientation."""

    axis: str = eqx.field(static=True)
    side: str = eqx.field(static=True)
    direction: TerminalDirection = eqx.field(static=True)
    face_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: str,
        side: str,
        direction: TerminalDirection,
        /,
    ):
        axis_ = str(axis)
        side_ = str(side)
        if not axis_ or side_ not in ("lower", "upper"):
            raise ValueError("A terminal face requires an axis and lower/upper side.")
        if not isinstance(direction, TerminalDirection):
            raise TypeError("direction must be a TerminalDirection.")
        self.axis = axis_
        self.side = side_
        self.direction = direction
        self.face_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-terminal-face",
                "axis": axis_,
                "side": side_,
                "positive_flow": direction.name,
            }
        )


class CirculationPortBinding(StrictModule, NonTrainableState):
    """Identity-only binding to a circulation-owned pressure/flow DAE port.

    The binding does not copy or own any 0D state, storage, resistance, or
    compliance.  Those remain in the supplied ``PressureFlowComponent``.
    """

    component_id: str = eqx.field(static=True)
    component_name: str = eqx.field(static=True)
    component_port_name: str = eqx.field(static=True)
    qualified_port_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: PressureFlowComponent,
        component_port_name: str,
        /,
    ):
        if not isinstance(component, PressureFlowComponent):
            raise TypeError("component must be a circulation PressureFlowComponent.")
        port_name = str(component_port_name)
        port = component.port(port_name)
        if len(port.potentials) != 1 or len(port.flows) != 1:
            raise ValueError(
                "A hemodynamics terminal requires one circulation pressure and one flow."
            )
        qualified = component.port_id(port_name)
        self.component_id = component.component_id
        self.component_name = component.name
        self.component_port_name = port_name
        self.qualified_port_id = qualified
        self.binding_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-hemodynamics-circulation-port-binding",
                "component": component.component_id,
                "qualified_port": qualified,
                "pressure_variable": port.potentials[0],
                "flow_variable": port.flows[0],
                "pressure_quantity": _PRESSURE.quantity_id,
                "flow_quantity": _FLOW.quantity_id,
            }
        )


class PressureTerminalPort(StrictModule, NonTrainableState):
    """Pressure-controlled 3D boundary backed by one circulation-owned p/Q port."""

    face: TerminalFace
    circulation: CirculationPortBinding
    pressure_reference_kpa: float = eqx.field(static=True)
    terminal_id: str = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        terminal_id: str,
        face: TerminalFace,
        circulation: CirculationPortBinding,
        /,
        *,
        pressure_reference_kpa: float = 0.0,
    ):
        identifier = str(terminal_id)
        reference = float(pressure_reference_kpa)
        if not identifier:
            raise ValueError("terminal_id must be nonempty.")
        if not isinstance(face, TerminalFace):
            raise TypeError("face must be TerminalFace.")
        if not isinstance(circulation, CirculationPortBinding):
            raise TypeError("circulation must be CirculationPortBinding.")
        if not np.isfinite(reference):
            raise ValueError("pressure_reference_kpa must be finite.")
        self.face = face
        self.circulation = circulation
        self.pressure_reference_kpa = reference
        self.terminal_id = identifier
        self.port_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-pressure-terminal-port",
                "terminal_id": identifier,
                "face": face.face_id,
                "circulation": circulation.binding_id,
                "pressure_reference_kpa": reference,
                "pressure_quantity": _PRESSURE.quantity_id,
                "flow_quantity": _FLOW.quantity_id,
            }
        )


class FlowTerminalPort(StrictModule, NonTrainableState):
    """Flow-controlled 3D boundary backed by one circulation-owned p/Q port."""

    face: TerminalFace
    circulation: CirculationPortBinding
    pressure_reference_kpa: float = eqx.field(static=True)
    terminal_id: str = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        terminal_id: str,
        face: TerminalFace,
        circulation: CirculationPortBinding,
        /,
        *,
        pressure_reference_kpa: float = 0.0,
    ):
        identifier = str(terminal_id)
        reference = float(pressure_reference_kpa)
        if not identifier:
            raise ValueError("terminal_id must be nonempty.")
        if not isinstance(face, TerminalFace):
            raise TypeError("face must be TerminalFace.")
        if not isinstance(circulation, CirculationPortBinding):
            raise TypeError("circulation must be CirculationPortBinding.")
        if not np.isfinite(reference):
            raise ValueError("pressure_reference_kpa must be finite.")
        self.face = face
        self.circulation = circulation
        self.pressure_reference_kpa = reference
        self.terminal_id = identifier
        self.port_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-flow-terminal-port",
                "terminal_id": identifier,
                "face": face.face_id,
                "circulation": circulation.binding_id,
                "pressure_reference_kpa": reference,
                "pressure_quantity": _PRESSURE.quantity_id,
                "flow_quantity": _FLOW.quantity_id,
            }
        )


TerminalPort: TypeAlias = PressureTerminalPort | FlowTerminalPort


class PressureMeasurementDefinition(StrictModule, NonTrainableState):
    """Area-weighted gauge pressure over one immutable terminal region."""

    cell_weights_mm2: Array
    total_area_mm2: Array
    pressure_reference_kpa: Array
    terminal_id: str = eqx.field(static=True)
    quantity_spec_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        terminal_id: str,
        cell_weights_mm2: ArrayLike,
        /,
        *,
        pressure_reference_kpa: float,
    ):
        weights = np.asarray(cell_weights_mm2, dtype=float)
        reference = float(pressure_reference_kpa)
        identifier = str(terminal_id)
        if (
            weights.ndim != 3
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
            or not np.isfinite(reference)
            or not identifier
        ):
            raise ValueError("Pressure measurement definition is invalid.")
        self.cell_weights_mm2 = jnp.asarray(weights, dtype=jnp.float64)
        self.total_area_mm2 = jnp.asarray(np.sum(weights), dtype=jnp.float64)
        self.pressure_reference_kpa = jnp.asarray(reference, dtype=jnp.float64)
        self.terminal_id = identifier
        self.quantity_spec_id = _PRESSURE.quantity_id
        self.definition_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-terminal-pressure-measurement",
                "terminal": identifier,
                "weights": array_tree_fingerprint(weights),
                "pressure_reference_kpa": reference,
                "quantity": self.quantity_spec_id,
            }
        )

    def measure(self, gauge_pressure_kpa: ArrayLike, /) -> Array:
        pressure = jnp.asarray(gauge_pressure_kpa)
        if pressure.shape != self.cell_weights_mm2.shape:
            raise ValueError("Pressure field must match the terminal measurement grid.")
        weights = self.cell_weights_mm2.astype(pressure.dtype)
        return oe.contract("ijk,ijk->", weights, pressure) / jnp.sum(
            weights
        ) + self.pressure_reference_kpa.astype(pressure.dtype)


class FlowMeasurementDefinition(StrictModule, NonTrainableState):
    """Oriented area integral of cell velocity over one terminal region."""

    cell_weights_mm2: Array
    outward_normal: Array
    total_area_mm2: Array
    direction: TerminalDirection = eqx.field(static=True)
    terminal_id: str = eqx.field(static=True)
    quantity_spec_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        terminal_id: str,
        cell_weights_mm2: ArrayLike,
        outward_normal: ArrayLike,
        direction: TerminalDirection,
        /,
    ):
        weights = np.asarray(cell_weights_mm2, dtype=float)
        normal = np.asarray(outward_normal, dtype=float)
        identifier = str(terminal_id)
        if (
            weights.ndim != 3
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
            or normal.shape != (3,)
            or np.any(~np.isfinite(normal))
            or not np.isclose(np.sum(normal**2), 1.0)
            or not identifier
        ):
            raise ValueError("Flow measurement definition is invalid.")
        if not isinstance(direction, TerminalDirection):
            raise TypeError("direction must be TerminalDirection.")
        self.cell_weights_mm2 = jnp.asarray(weights, dtype=jnp.float64)
        self.outward_normal = jnp.asarray(normal, dtype=jnp.float64)
        self.total_area_mm2 = jnp.asarray(np.sum(weights), dtype=jnp.float64)
        self.direction = direction
        self.terminal_id = identifier
        self.quantity_spec_id = _FLOW.quantity_id
        self.definition_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-terminal-flow-measurement",
                "terminal": identifier,
                "weights": array_tree_fingerprint(weights),
                "outward_normal": normal.tolist(),
                "positive_flow": direction.name,
                "quantity": self.quantity_spec_id,
            }
        )

    def outward_flow_rate(self, velocity_mm_per_ms: ArrayLike, /) -> Array:
        velocity = jnp.asarray(velocity_mm_per_ms)
        if velocity.shape != self.cell_weights_mm2.shape + (3,):
            raise ValueError("Velocity field must match the terminal measurement grid.")
        normal = self.outward_normal.astype(velocity.dtype)
        weights = self.cell_weights_mm2.astype(velocity.dtype)
        normal_velocity = oe.contract("ijkd,d->ijk", velocity, normal)
        return oe.contract("ijk,ijk->", weights, normal_velocity)

    def directed_flow_rate(self, velocity_mm_per_ms: ArrayLike, /) -> Array:
        return self.direction.outward_sign * self.outward_flow_rate(velocity_mm_per_ms)


class TerminalMeasurements(StrictModule):
    """Fixed-order pressure, flow, and power observations at all terminals."""

    pressure_kpa: Array
    directed_flow_mm3_per_ms: Array
    outward_flow_mm3_per_ms: Array
    power_into_lumen_mg_mm2_per_ms3: Array
    terminal_ids: tuple[str, ...] = eqx.field(static=True)


class TerminalPortValues(StrictModule):
    """p/Q values owned by the coupled circulation solve in terminal order."""

    pressure_kpa: Array
    directed_flow_mm3_per_ms: Array

    def __init__(
        self,
        pressure_kpa: ArrayLike,
        directed_flow_mm3_per_ms: ArrayLike,
        /,
    ):
        pressure = jnp.asarray(pressure_kpa)
        flow = jnp.asarray(directed_flow_mm3_per_ms, dtype=pressure.dtype)
        if pressure.ndim != 1 or flow.shape != pressure.shape:
            raise ValueError("Terminal pressure and flow must be equal-length vectors.")
        if not jnp.issubdtype(pressure.dtype, jnp.inexact):
            raise TypeError("Terminal pressure and flow must have an inexact dtype.")
        self.pressure_kpa = pressure
        self.directed_flow_mm3_per_ms = flow


class PreparedTerminalMeasurements(StrictModule, NonTrainableState):
    """Frozen terminal regions and their pressure/flow measurement operators."""

    pressure_definitions: tuple[PressureMeasurementDefinition, ...]
    flow_definitions: tuple[FlowMeasurementDefinition, ...]
    terminal_ids: tuple[str, ...] = eqx.field(static=True)
    circulation_port_ids: tuple[str, ...] = eqx.field(static=True)
    outward_signs: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        pressure_definitions: tuple[PressureMeasurementDefinition, ...],
        flow_definitions: tuple[FlowMeasurementDefinition, ...],
        circulation_port_ids: tuple[str, ...],
        /,
    ):
        if not pressure_definitions or len(pressure_definitions) != len(flow_definitions):
            raise ValueError("Prepared terminal measurements require paired definitions.")
        terminal_ids = tuple(value.terminal_id for value in pressure_definitions)
        if terminal_ids != tuple(value.terminal_id for value in flow_definitions):
            raise ValueError("Pressure and flow definitions must share terminal order.")
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("Terminal identifiers must be unique.")
        if len(circulation_port_ids) != len(terminal_ids):
            raise ValueError("Every terminal must bind one circulation port.")
        self.pressure_definitions = pressure_definitions
        self.flow_definitions = flow_definitions
        self.terminal_ids = terminal_ids
        self.circulation_port_ids = tuple(circulation_port_ids)
        self.outward_signs = jnp.asarray(
            tuple(value.direction.outward_sign for value in flow_definitions),
            dtype=jnp.float64,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-terminal-measurements",
                "pressure_definitions": tuple(
                    value.definition_id for value in pressure_definitions
                ),
                "flow_definitions": tuple(
                    value.definition_id for value in flow_definitions
                ),
                "circulation_ports": circulation_port_ids,
            }
        )

    @property
    def terminal_count(self) -> int:
        return len(self.terminal_ids)

    @property
    def areas_mm2(self) -> Array:
        return jnp.stack(tuple(value.total_area_mm2 for value in self.flow_definitions))

    def measure(
        self,
        gauge_pressure_kpa: ArrayLike,
        velocity_mm_per_ms: ArrayLike,
        /,
    ) -> TerminalMeasurements:
        pressure = jnp.stack(
            tuple(
                value.measure(gauge_pressure_kpa) for value in self.pressure_definitions
            )
        )
        outward = jnp.stack(
            tuple(
                value.outward_flow_rate(velocity_mm_per_ms)
                for value in self.flow_definitions
            )
        )
        signs = self.outward_signs.astype(outward.dtype)
        directed = signs * outward
        power = -pressure * outward
        return TerminalMeasurements(
            pressure,
            directed,
            outward,
            power,
            self.terminal_ids,
        )

    def validate_values(self, values: TerminalPortValues, /) -> TerminalPortValues:
        if not isinstance(values, TerminalPortValues):
            raise TypeError("values must be TerminalPortValues.")
        if values.pressure_kpa.shape != (self.terminal_count,):
            raise ValueError("Terminal values do not match the prepared terminal order.")
        return values


class TerminalBalanceEvidence(StrictModule):
    """Per-terminal p/Q match plus control-volume volume and power balances."""

    pressure_residual_kpa: Array
    pressure_tolerance_kpa: Array
    pressure_balanced: Array
    flow_relative_defect: Array
    volume_relative_defect: Array
    power_relative_defect: Array
    measured_power_into_lumen: Array
    circulation_power_into_lumen: Array
    finite: Array
    flow_balanced: Array
    volume_balanced: Array
    power_balanced: Array
    passed: Array


def terminal_balance_evidence(
    prepared: PreparedTerminalMeasurements,
    measured: TerminalMeasurements,
    circulation_values: TerminalPortValues,
    /,
    *,
    storage_volume_change_mm3: ArrayLike,
    time_step_ms: ArrayLike,
    flow_relative_tolerance: float,
    pressure_absolute_tolerance_kpa: ArrayLike,
    volume_relative_tolerance: float,
    power_relative_tolerance: float,
) -> TerminalBalanceEvidence:
    """Audit coupling equality and one-step volume/power exchange."""

    if not isinstance(prepared, PreparedTerminalMeasurements):
        raise TypeError("prepared must be PreparedTerminalMeasurements.")
    if not isinstance(measured, TerminalMeasurements):
        raise TypeError("measured must be TerminalMeasurements.")
    values = prepared.validate_values(circulation_values)
    if measured.terminal_ids != prepared.terminal_ids:
        raise ValueError("Measurements and terminal plan identifiers do not match.")
    storage = jnp.asarray(storage_volume_change_mm3, dtype=measured.pressure_kpa.dtype)
    step = jnp.asarray(time_step_ms, dtype=measured.pressure_kpa.dtype)
    if storage.shape != () or step.shape != ():
        raise ValueError("Storage-volume change and time step must be scalars.")
    scalar_tolerances = tuple(
        float(value)
        for value in (
            flow_relative_tolerance,
            volume_relative_tolerance,
            power_relative_tolerance,
        )
    )
    pressure_tolerance_host = np.asarray(pressure_absolute_tolerance_kpa, dtype=float)
    if pressure_tolerance_host.shape == ():
        pressure_tolerance_host = np.full(
            (prepared.terminal_count,), float(pressure_tolerance_host)
        )
    if (
        pressure_tolerance_host.shape != (prepared.terminal_count,)
        or np.any(~np.isfinite(pressure_tolerance_host))
        or np.any(pressure_tolerance_host < 0.0)
        or any(not np.isfinite(value) or value < 0.0 for value in scalar_tolerances)
    ):
        raise ValueError("Terminal balance tolerances must be finite and nonnegative.")

    pressure = values.pressure_kpa.astype(measured.pressure_kpa.dtype)
    flow = values.directed_flow_mm3_per_ms.astype(measured.pressure_kpa.dtype)
    pressure_tolerance = jnp.asarray(
        pressure_tolerance_host, dtype=measured.pressure_kpa.dtype
    )
    pressure_residual = measured.pressure_kpa - pressure
    pressure_balanced = jnp.isfinite(pressure_residual) & (
        jnp.abs(pressure_residual) <= pressure_tolerance
    )
    signs = prepared.outward_signs.astype(measured.pressure_kpa.dtype)
    circulation_outward = signs * flow
    relative_floor = jnp.sqrt(jnp.finfo(flow.dtype).eps)
    flow_scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(flow)),
            jnp.max(jnp.abs(measured.directed_flow_mm3_per_ms)),
        ),
        relative_floor,
    )
    flow_relative = (
        jnp.max(jnp.abs(measured.directed_flow_mm3_per_ms - flow)) / flow_scale
    )

    measured_volume_terms = measured.outward_flow_mm3_per_ms * step
    volume_residual = storage + jnp.sum(measured_volume_terms)
    volume_scale = jnp.maximum(
        jnp.maximum(jnp.abs(storage), jnp.sum(jnp.abs(measured_volume_terms))),
        relative_floor,
    )
    volume_relative = jnp.abs(volume_residual) / volume_scale

    measured_power = jnp.sum(measured.power_into_lumen_mg_mm2_per_ms3)
    circulation_power = jnp.sum(-pressure * circulation_outward)
    power_scale = jnp.maximum(
        jnp.maximum(jnp.abs(measured_power), jnp.abs(circulation_power)),
        relative_floor,
    )
    power_relative = jnp.abs(measured_power - circulation_power) / power_scale
    finite = (
        jnp.all(jnp.isfinite(measured.pressure_kpa))
        & jnp.all(jnp.isfinite(measured.directed_flow_mm3_per_ms))
        & jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(flow))
        & jnp.isfinite(storage)
        & jnp.isfinite(step)
        & (step > 0.0)
        & jnp.isfinite(flow_relative)
        & jnp.isfinite(volume_relative)
        & jnp.isfinite(power_relative)
    )
    flow_balanced = finite & (flow_relative <= scalar_tolerances[0])
    volume_balanced = finite & (volume_relative <= scalar_tolerances[1])
    power_balanced = finite & (power_relative <= scalar_tolerances[2])
    return TerminalBalanceEvidence(
        pressure_residual,
        pressure_tolerance,
        pressure_balanced,
        flow_relative,
        volume_relative,
        power_relative,
        measured_power,
        circulation_power,
        finite,
        flow_balanced,
        volume_balanced,
        power_balanced,
        flow_balanced & volume_balanced & power_balanced & jnp.all(pressure_balanced),
    )


def prepare_terminal_measurements(
    discretization: LatticeBoltzmannDiscretization,
    lumen: FixedWallLumenRegion,
    terminals: tuple[TerminalPort, ...],
    /,
) -> PreparedTerminalMeasurements:
    """Freeze whole-face measurement regions against one D3Q19 lumen mask."""

    if not isinstance(discretization, LatticeBoltzmannDiscretization):
        raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
    if not isinstance(lumen, FixedWallLumenRegion):
        raise TypeError("lumen must be FixedWallLumenRegion.")
    if lumen.shape != discretization.grid.shape:
        raise ValueError("Lumen mask and LBM grid shapes do not match.")
    values = tuple(terminals)
    if not values or any(
        not isinstance(value, (PressureTerminalPort, FlowTerminalPort))
        for value in values
    ):
        raise ValueError("At least one typed pressure or flow terminal is required.")
    terminal_ids = tuple(value.terminal_id for value in values)
    faces = tuple((value.face.axis, value.face.side) for value in values)
    circulation_ids = tuple(value.circulation.qualified_port_id for value in values)
    if len(set(terminal_ids)) != len(values):
        raise ValueError("Terminal identifiers must be unique.")
    if len(set(faces)) != len(values):
        raise ValueError("Each exterior face may own at most one terminal.")
    if len(set(circulation_ids)) != len(values):
        raise ValueError("Each circulation p/Q port may bind at most one 3D terminal.")
    axis_names = discretization.grid.axis_names
    cell_area = float(discretization.cell_size) ** 2
    mask = np.asarray(lumen.fluid_mask, dtype=bool)
    pressure_definitions = []
    flow_definitions = []
    for terminal in values:
        face = terminal.face
        if face.axis not in axis_names:
            raise ValueError(f"Unknown terminal axis {face.axis!r}.")
        axis = axis_names.index(face.axis)
        if discretization.periodic[axis]:
            raise ValueError("A periodic grid face cannot be a cardiovascular terminal.")
        face_slice: list[object] = [slice(None)] * 3
        face_slice[axis] = 0 if face.side == "lower" else -1
        region = np.zeros(discretization.grid.shape, dtype=bool)
        region[tuple(face_slice)] = mask[tuple(face_slice)]
        if not np.any(region):
            raise ValueError(
                f"Terminal {terminal.terminal_id!r} has no fluid face cells."
            )
        weights = cell_area * region.astype(float)
        normal = np.zeros(3, dtype=float)
        normal[axis] = -1.0 if face.side == "lower" else 1.0
        pressure_definitions.append(
            PressureMeasurementDefinition(
                terminal.terminal_id,
                weights,
                pressure_reference_kpa=terminal.pressure_reference_kpa,
            )
        )
        flow_definitions.append(
            FlowMeasurementDefinition(
                terminal.terminal_id,
                weights,
                normal,
                face.direction,
            )
        )
    return PreparedTerminalMeasurements(
        tuple(pressure_definitions),
        tuple(flow_definitions),
        circulation_ids,
    )


__all__ = [
    "CirculationPortBinding",
    "FlowMeasurementDefinition",
    "FlowTerminalPort",
    "prepare_terminal_measurements",
    "PreparedTerminalMeasurements",
    "PressureMeasurementDefinition",
    "PressureTerminalPort",
    "TerminalBalanceEvidence",
    "terminal_balance_evidence",
    "TerminalDirection",
    "TerminalFace",
    "TerminalMeasurements",
    "TerminalPort",
    "TerminalPortValues",
]
