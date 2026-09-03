#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative one-dimensional blood-vessel dynamics and couplers.

The kernel units are millimetres, milliseconds, milligrams, and kilopascals.
The finite-volume state is cross-sectional area and signed volume flow.  Every
runtime object has fixed topology and every step is transactional: invalid
candidates are returned as evidence while the accepted state remains unchanged.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite, pi
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


VascularPortSide = Literal["inlet", "outlet"]


class VascularStepStatus(IntFlag):
    """Fail-closed status bits for a one-dimensional vessel step."""

    SUCCESS = 0
    NONFINITE = 1
    AREA_OUT_OF_DOMAIN = 2
    CFL_VIOLATION = 4
    INVALID_BOUNDARY = 8


class SquareRootTubeLaw(StrictModule, NonTrainableState):
    """Named elastic tube law ``p=p0+beta*(sqrt(A/A0)-1)``.

    The lower and upper area ratios are part of the model identity rather than
    numerical clipping thresholds.  Evaluation outside that declared physical
    validity interval is rejected by vessel and port operations.
    """

    reference_area_mm2: float = eqx.field(static=True)
    reference_pressure_kPa: float = eqx.field(static=True)
    stiffness_kPa: float = eqx.field(static=True)
    minimum_area_ratio: float = eqx.field(static=True)
    maximum_area_ratio: float = eqx.field(static=True)
    law_name: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_area_mm2: float,
        stiffness_kPa: float,
        /,
        *,
        reference_pressure_kPa: float = 0.0,
        minimum_area_ratio: float = 0.2,
        maximum_area_ratio: float = 5.0,
    ):
        area = float(reference_area_mm2)
        stiffness = float(stiffness_kPa)
        pressure = float(reference_pressure_kPa)
        minimum = float(minimum_area_ratio)
        maximum = float(maximum_area_ratio)
        if not all(
            isfinite(value) for value in (area, stiffness, pressure, minimum, maximum)
        ):
            raise ValueError("Tube-law parameters must be finite.")
        if area <= 0.0 or stiffness <= 0.0:
            raise ValueError("Reference area and stiffness must be positive.")
        if minimum <= 0.0 or maximum <= minimum:
            raise ValueError("Tube-law area-ratio bounds must satisfy 0 < min < max.")
        self.reference_area_mm2 = area
        self.reference_pressure_kPa = pressure
        self.stiffness_kPa = stiffness
        self.minimum_area_ratio = minimum
        self.maximum_area_ratio = maximum
        self.law_name = "square-root-elastic"
        self.law_id = canonical_fingerprint(
            {
                "kind": self.law_name,
                "reference_area_mm2": area,
                "reference_pressure_kPa": pressure,
                "stiffness_kPa": stiffness,
                "minimum_area_ratio": minimum,
                "maximum_area_ratio": maximum,
            }
        )

    @property
    def minimum_area_mm2(self) -> float:
        return self.minimum_area_ratio * self.reference_area_mm2

    @property
    def maximum_area_mm2(self) -> float:
        return self.maximum_area_ratio * self.reference_area_mm2

    def valid_area(self, area_mm2: ArrayLike, /) -> Array:
        area = jnp.asarray(area_mm2)
        return (
            jnp.isfinite(area)
            & (area >= self.minimum_area_mm2)
            & (area <= self.maximum_area_mm2)
        )

    def pressure(self, area_mm2: ArrayLike, /) -> Array:
        """Return transmural pressure; callers must enforce ``valid_area``."""
        area = jnp.asarray(area_mm2)
        return self.reference_pressure_kPa + self.stiffness_kPa * (
            jnp.sqrt(area / self.reference_area_mm2) - 1.0
        )

    def pressure_derivative(self, area_mm2: ArrayLike, /) -> Array:
        """Return ``dp/dA`` in kPa/mm² on the declared area domain."""
        area = jnp.asarray(area_mm2)
        return self.stiffness_kPa / (2.0 * jnp.sqrt(area * self.reference_area_mm2))

    def area(self, pressure_kPa: ArrayLike, /) -> Array:
        """Invert the tube law without clipping pressure to its validity range."""
        pressure = jnp.asarray(pressure_kPa)
        root_ratio = 1.0 + (pressure - self.reference_pressure_kPa) / self.stiffness_kPa
        return self.reference_area_mm2 * root_ratio * root_ratio

    def valid_pressure(self, pressure_kPa: ArrayLike, /) -> Array:
        pressure = jnp.asarray(pressure_kPa)
        area = self.area(pressure)
        positive_root = (
            1.0 + (pressure - self.reference_pressure_kPa) / self.stiffness_kPa
        )
        return jnp.isfinite(pressure) & (positive_root > 0.0) & self.valid_area(area)

    def wave_speed(self, area_mm2: ArrayLike, density_mg_per_mm3: ArrayLike, /) -> Array:
        """Return the Moens–Korteweg characteristic speed in mm/ms."""
        area = jnp.asarray(area_mm2)
        density = jnp.asarray(density_mg_per_mm3, dtype=area.dtype)
        return jnp.sqrt(area * self.pressure_derivative(area) / density)

    def characteristic_impedance(
        self,
        area_mm2: ArrayLike,
        density_mg_per_mm3: ArrayLike,
        /,
    ) -> Array:
        """Return local pressure/flow impedance in kPa·ms/mm³."""
        area = jnp.asarray(area_mm2)
        density = jnp.asarray(density_mg_per_mm3, dtype=area.dtype)
        return density * self.wave_speed(area, density) / area

    def pressure_potential(
        self,
        area_mm2: ArrayLike,
        density_mg_per_mm3: ArrayLike,
        /,
    ) -> Array:
        """Return the conservative momentum-flux pressure potential."""
        area = jnp.asarray(area_mm2)
        density = jnp.asarray(density_mg_per_mm3, dtype=area.dtype)
        return (
            self.stiffness_kPa
            * area**1.5
            / (3.0 * density * jnp.sqrt(self.reference_area_mm2))
        )


class Vascular1DPlan(StrictModule, NonTrainableState):
    """Discretization and fluid plan for one fixed one-dimensional vessel."""

    vessel_id: str = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    length_mm: float = eqx.field(static=True)
    step_size_ms: float = eqx.field(static=True)
    density_mg_per_mm3: float = eqx.field(static=True)
    dynamic_viscosity_mg_per_mm_ms: float = eqx.field(static=True)
    maximum_courant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vessel_id: str,
        cell_count: int,
        length_mm: float,
        step_size_ms: float,
        /,
        *,
        density_mg_per_mm3: float = 1.06,
        dynamic_viscosity_mg_per_mm_ms: float = 3.5e-3,
        maximum_courant: float = 0.9,
    ):
        identifier = str(vessel_id)
        count = int(cell_count)
        length = float(length_mm)
        step = float(step_size_ms)
        density = float(density_mg_per_mm3)
        viscosity = float(dynamic_viscosity_mg_per_mm_ms)
        courant = float(maximum_courant)
        if not identifier:
            raise ValueError("vessel_id must be non-empty.")
        if count < 2:
            raise ValueError("cell_count must be at least two.")
        if not all(
            isfinite(value) for value in (length, step, density, viscosity, courant)
        ):
            raise ValueError("Vessel plan scalars must be finite.")
        if length <= 0.0 or step <= 0.0 or density <= 0.0:
            raise ValueError("Length, step size, and density must be positive.")
        if viscosity < 0.0 or courant <= 0.0 or courant > 1.0:
            raise ValueError("Viscosity and maximum Courant number are invalid.")
        self.vessel_id = identifier
        self.cell_count = count
        self.length_mm = length
        self.step_size_ms = step
        self.density_mg_per_mm3 = density
        self.dynamic_viscosity_mg_per_mm_ms = viscosity
        self.maximum_courant = courant
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vascular-1d-plan-v1",
                "vessel_id": identifier,
                "cell_count": count,
                "length_mm": length,
                "step_size_ms": step,
                "density_mg_per_mm3": density,
                "dynamic_viscosity_mg_per_mm_ms": viscosity,
                "maximum_courant": courant,
            }
        )

    def prepare(self, tube_law: SquareRootTubeLaw, /) -> "PreparedVascular1D":
        if not isinstance(tube_law, SquareRootTubeLaw):
            raise TypeError("tube_law must be a SquareRootTubeLaw.")
        return PreparedVascular1D(
            plan=self,
            tube_law=tube_law,
            cell_length_mm=self.length_mm / self.cell_count,
            runtime_id=canonical_fingerprint(
                {
                    "kind": "prepared-vascular-1d-v1",
                    "plan": self.plan_id,
                    "tube_law": tube_law.law_id,
                }
            ),
        )


class PreparedVascular1D(StrictModule, NonTrainableState):
    """Prepared fixed-grid vascular runtime."""

    plan: Vascular1DPlan
    tube_law: SquareRootTubeLaw
    cell_length_mm: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)


class Vascular1DState(StrictModule):
    """Conservative cell averages at one accepted time boundary."""

    area_mm2: Array
    flow_mm3_per_ms: Array
    time_ms: Array
    step_index: Array


class VascularBoundaryState(StrictModule):
    """One ghost-cell state, expressed in the vessel's axial orientation."""

    area_mm2: Array
    flow_mm3_per_ms: Array


class VascularStepEvidence(StrictModule):
    """Conservation, CFL, domain, and commit evidence for one candidate."""

    mass_balance_residual_mm3: Array
    momentum_balance_residual_mm4_per_ms: Array
    maximum_courant: Array
    finite: Array
    area_valid: Array
    boundary_valid: Array
    status: Array
    successful: Array


class VascularStepResult(StrictModule):
    """Accepted state, uncommitted candidate, interface fluxes, and evidence."""

    state: Vascular1DState
    candidate: Vascular1DState
    mass_flux_mm3_per_ms: Array
    momentum_flux_mm4_per_ms2: Array
    evidence: VascularStepEvidence


def initialize_vascular_state(
    runtime: PreparedVascular1D,
    area_mm2: ArrayLike,
    flow_mm3_per_ms: ArrayLike,
    /,
) -> Vascular1DState:
    """Create a checked fixed-shape state at time zero."""
    if not isinstance(runtime, PreparedVascular1D):
        raise TypeError("runtime must be a PreparedVascular1D.")
    area = jnp.asarray(area_mm2)
    flow = jnp.asarray(flow_mm3_per_ms, dtype=area.dtype)
    expected = (runtime.plan.cell_count,)
    if area.shape != expected or flow.shape != expected:
        raise ValueError(f"Area and flow must both have shape {expected}.")
    area_host = np.asarray(area)
    flow_host = np.asarray(flow)
    if not np.all(np.isfinite(area_host)) or not np.all(np.isfinite(flow_host)):
        raise ValueError("Initial vascular state must be finite.")
    if not np.all(np.asarray(runtime.tube_law.valid_area(area))):
        raise ValueError("Initial area is outside the tube-law validity domain.")
    return Vascular1DState(
        area_mm2=area,
        flow_mm3_per_ms=flow,
        time_ms=jnp.asarray(0.0, dtype=area.dtype),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )


def _physical_flux(
    runtime: PreparedVascular1D,
    area_mm2: Array,
    flow_mm3_per_ms: Array,
) -> tuple[Array, Array, Array]:
    velocity = flow_mm3_per_ms / area_mm2
    wave_speed = runtime.tube_law.wave_speed(area_mm2, runtime.plan.density_mg_per_mm3)
    momentum = flow_mm3_per_ms * velocity + runtime.tube_law.pressure_potential(
        area_mm2, runtime.plan.density_mg_per_mm3
    )
    return flow_mm3_per_ms, momentum, jnp.abs(velocity) + wave_speed


def vascular_numerical_flux(
    runtime: PreparedVascular1D,
    left_area_mm2: ArrayLike,
    left_flow_mm3_per_ms: ArrayLike,
    right_area_mm2: ArrayLike,
    right_flow_mm3_per_ms: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Return a local Lax–Friedrichs flux for the conservative vessel state."""
    left_area = jnp.asarray(left_area_mm2)
    right_area = jnp.asarray(right_area_mm2, dtype=left_area.dtype)
    left_flow = jnp.asarray(left_flow_mm3_per_ms, dtype=left_area.dtype)
    right_flow = jnp.asarray(right_flow_mm3_per_ms, dtype=left_area.dtype)
    left_mass, left_momentum, left_speed = _physical_flux(runtime, left_area, left_flow)
    right_mass, right_momentum, right_speed = _physical_flux(
        runtime, right_area, right_flow
    )
    signal_speed = jnp.maximum(left_speed, right_speed)
    mass_flux = 0.5 * (left_mass + right_mass) - 0.5 * signal_speed * (
        right_area - left_area
    )
    momentum_flux = 0.5 * (left_momentum + right_momentum) - 0.5 * signal_speed * (
        right_flow - left_flow
    )
    return mass_flux, momentum_flux


def step_vascular_1d(
    runtime: PreparedVascular1D,
    state: Vascular1DState,
    left_boundary: VascularBoundaryState,
    right_boundary: VascularBoundaryState,
    /,
) -> VascularStepResult:
    """Advance one conservative finite-volume step and commit only valid results."""
    if not isinstance(runtime, PreparedVascular1D):
        raise TypeError("runtime must be a PreparedVascular1D.")
    count = runtime.plan.cell_count
    if state.area_mm2.shape != (count,) or state.flow_mm3_per_ms.shape != (count,):
        raise ValueError("State shape does not match the prepared vessel.")
    area_valid = jnp.all(runtime.tube_law.valid_area(state.area_mm2))
    boundary_valid = (
        jnp.all(runtime.tube_law.valid_area(left_boundary.area_mm2))
        & jnp.all(runtime.tube_law.valid_area(right_boundary.area_mm2))
        & jnp.all(jnp.isfinite(left_boundary.flow_mm3_per_ms))
        & jnp.all(jnp.isfinite(right_boundary.flow_mm3_per_ms))
    )
    safe_area = jnp.where(
        runtime.tube_law.valid_area(state.area_mm2),
        state.area_mm2,
        runtime.tube_law.reference_area_mm2,
    )
    left_area = jnp.where(
        runtime.tube_law.valid_area(left_boundary.area_mm2),
        left_boundary.area_mm2,
        runtime.tube_law.reference_area_mm2,
    )
    right_area = jnp.where(
        runtime.tube_law.valid_area(right_boundary.area_mm2),
        right_boundary.area_mm2,
        runtime.tube_law.reference_area_mm2,
    )
    extended_area = jnp.concatenate(
        (left_area.reshape(1), safe_area, right_area.reshape(1))
    )
    extended_flow = jnp.concatenate(
        (
            jnp.asarray(left_boundary.flow_mm3_per_ms).reshape(1),
            state.flow_mm3_per_ms,
            jnp.asarray(right_boundary.flow_mm3_per_ms).reshape(1),
        )
    )
    mass_flux, momentum_flux = vascular_numerical_flux(
        runtime,
        extended_area[:-1],
        extended_flow[:-1],
        extended_area[1:],
        extended_flow[1:],
    )
    dt = runtime.plan.step_size_ms
    dx = runtime.cell_length_mm
    area_candidate = safe_area - (dt / dx) * (mass_flux[1:] - mass_flux[:-1])
    friction = -(
        8.0
        * pi
        * runtime.plan.dynamic_viscosity_mg_per_mm_ms
        * state.flow_mm3_per_ms
        / (runtime.plan.density_mg_per_mm3 * safe_area)
    )
    flow_candidate = (
        state.flow_mm3_per_ms
        - (dt / dx) * (momentum_flux[1:] - momentum_flux[:-1])
        + dt * friction
    )
    candidate = Vascular1DState(
        area_mm2=area_candidate,
        flow_mm3_per_ms=flow_candidate,
        time_ms=state.time_ms + dt,
        step_index=state.step_index + jnp.asarray(1, dtype=state.step_index.dtype),
    )
    _, _, extended_signal_speed = _physical_flux(runtime, extended_area, extended_flow)
    maximum_courant = dt * jnp.max(extended_signal_speed) / dx
    candidate_area_valid = jnp.all(runtime.tube_law.valid_area(area_candidate))
    finite = (
        jnp.all(jnp.isfinite(state.area_mm2))
        & jnp.all(jnp.isfinite(state.flow_mm3_per_ms))
        & jnp.all(jnp.isfinite(area_candidate))
        & jnp.all(jnp.isfinite(flow_candidate))
        & jnp.all(jnp.isfinite(mass_flux))
        & jnp.all(jnp.isfinite(momentum_flux))
        & jnp.isfinite(maximum_courant)
    )
    cfl_valid = maximum_courant <= runtime.plan.maximum_courant
    successful = finite & area_valid & candidate_area_valid & boundary_valid & cfl_valid
    status = jnp.asarray(int(VascularStepStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(VascularStepStatus.NONFINITE)),
    )
    status = jnp.where(
        area_valid & candidate_area_valid,
        status,
        jnp.bitwise_or(status, int(VascularStepStatus.AREA_OUT_OF_DOMAIN)),
    )
    status = jnp.where(
        cfl_valid,
        status,
        jnp.bitwise_or(status, int(VascularStepStatus.CFL_VIOLATION)),
    )
    status = jnp.where(
        boundary_valid,
        status,
        jnp.bitwise_or(status, int(VascularStepStatus.INVALID_BOUNDARY)),
    )
    accepted = jax.tree.map(
        lambda proposed, prior: jnp.where(successful, proposed, prior), candidate, state
    )
    old_area_integral = dx * jnp.sum(safe_area)
    new_area_integral = dx * jnp.sum(area_candidate)
    mass_residual = (
        new_area_integral - old_area_integral + dt * (mass_flux[-1] - mass_flux[0])
    )
    old_flow_integral = dx * jnp.sum(state.flow_mm3_per_ms)
    new_flow_integral = dx * jnp.sum(flow_candidate)
    momentum_residual = (
        new_flow_integral
        - old_flow_integral
        + dt * (momentum_flux[-1] - momentum_flux[0])
        - dt * dx * jnp.sum(friction)
    )
    evidence = VascularStepEvidence(
        mass_balance_residual_mm3=mass_residual,
        momentum_balance_residual_mm4_per_ms=momentum_residual,
        maximum_courant=maximum_courant,
        finite=finite,
        area_valid=area_valid & candidate_area_valid,
        boundary_valid=boundary_valid,
        status=status,
        successful=successful,
    )
    return VascularStepResult(
        state=accepted,
        candidate=candidate,
        mass_flux_mm3_per_ms=mass_flux,
        momentum_flux_mm4_per_ms2=momentum_flux,
        evidence=evidence,
    )


class CharacteristicTerminal(StrictModule, NonTrainableState):
    """Linear characteristic load with an explicit, fixed impedance domain."""

    terminal_id: str = eqx.field(static=True)
    reference_pressure_kPa: float = eqx.field(static=True)
    load_impedance_kPa_ms_per_mm3: float = eqx.field(static=True)
    terminal_id_hash: str = eqx.field(static=True)

    def __init__(
        self,
        terminal_id: str,
        reference_pressure_kPa: float,
        load_impedance_kPa_ms_per_mm3: float,
        /,
    ):
        identifier = str(terminal_id)
        pressure = float(reference_pressure_kPa)
        impedance = float(load_impedance_kPa_ms_per_mm3)
        if not identifier:
            raise ValueError("terminal_id must be non-empty.")
        if not isfinite(pressure) or not isfinite(impedance) or impedance <= 0.0:
            raise ValueError("Terminal pressure must be finite and impedance positive.")
        self.terminal_id = identifier
        self.reference_pressure_kPa = pressure
        self.load_impedance_kPa_ms_per_mm3 = impedance
        self.terminal_id_hash = canonical_fingerprint(
            {
                "kind": "characteristic-terminal-v1",
                "terminal_id": identifier,
                "reference_pressure_kPa": pressure,
                "load_impedance_kPa_ms_per_mm3": impedance,
            }
        )


class TerminalReflection(StrictModule):
    """Pressure/flow wave evidence at a characteristic terminal."""

    reflection_coefficient: Array
    reflected_pressure_wave_kPa: Array
    terminal_pressure_kPa: Array
    terminal_flow_mm3_per_ms: Array
    successful: Array


def reflect_characteristic_wave(
    terminal: CharacteristicTerminal,
    incident_pressure_wave_kPa: ArrayLike,
    vessel_impedance_kPa_ms_per_mm3: ArrayLike,
    /,
) -> TerminalReflection:
    """Reflect one incident wave, refusing nonpositive characteristic impedance."""
    incident = jnp.asarray(incident_pressure_wave_kPa)
    vessel_impedance = jnp.asarray(vessel_impedance_kPa_ms_per_mm3, dtype=incident.dtype)
    valid = (
        jnp.isfinite(incident) & jnp.isfinite(vessel_impedance) & (vessel_impedance > 0.0)
    )
    safe_impedance = jnp.where(valid, vessel_impedance, 1.0)
    load_impedance = jnp.asarray(
        terminal.load_impedance_kPa_ms_per_mm3, dtype=incident.dtype
    )
    coefficient = (load_impedance - safe_impedance) / (load_impedance + safe_impedance)
    reflected = coefficient * incident
    pressure = terminal.reference_pressure_kPa + incident + reflected
    flow = (incident - reflected) / safe_impedance
    nan = jnp.asarray(jnp.nan, dtype=incident.dtype)
    return TerminalReflection(
        reflection_coefficient=jnp.where(valid, coefficient, nan),
        reflected_pressure_wave_kPa=jnp.where(valid, reflected, nan),
        terminal_pressure_kPa=jnp.where(valid, pressure, nan),
        terminal_flow_mm3_per_ms=jnp.where(valid, flow, nan),
        successful=valid,
    )


class VascularJunctionPlan(StrictModule, NonTrainableState):
    """Fixed branch ordering for a pressure-continuous conservative junction."""

    junction_id: str = eqx.field(static=True)
    branch_ids: tuple[str, ...] = eqx.field(static=True)
    branch_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, junction_id: str, branch_ids: tuple[str, ...], /):
        identifier = str(junction_id)
        branches = tuple(str(value) for value in branch_ids)
        if not identifier or len(branches) < 2:
            raise ValueError("A junction needs an ID and at least two branches.")
        if any(not value for value in branches) or len(set(branches)) != len(branches):
            raise ValueError("Junction branch IDs must be unique and non-empty.")
        self.junction_id = identifier
        self.branch_ids = branches
        self.branch_count = len(branches)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vascular-junction-v1",
                "junction_id": identifier,
                "branches": branches,
            }
        )


class VascularJunctionResult(StrictModule):
    """Common pressure and outward branch flows at a junction."""

    common_pressure_kPa: Array
    branch_flow_away_mm3_per_ms: Array
    conservation_residual_mm3_per_ms: Array
    pressure_residual_kPa: Array
    successful: Array


def solve_vascular_junction(
    plan: VascularJunctionPlan,
    incident_pressure_kPa: ArrayLike,
    characteristic_impedance_kPa_ms_per_mm3: ArrayLike,
    /,
    *,
    source_flow_into_junction_mm3_per_ms: ArrayLike = 0.0,
) -> VascularJunctionResult:
    """Solve a lossless characteristic junction with exact flow conservation."""
    incident = jnp.asarray(incident_pressure_kPa)
    impedance = jnp.asarray(characteristic_impedance_kPa_ms_per_mm3, dtype=incident.dtype)
    if incident.shape != (plan.branch_count,) or impedance.shape != (plan.branch_count,):
        raise ValueError("Junction arrays must match the fixed branch count.")
    source = jnp.asarray(
        source_flow_into_junction_mm3_per_ms, dtype=incident.dtype
    ).reshape(())
    valid = (
        jnp.all(jnp.isfinite(incident))
        & jnp.all(jnp.isfinite(impedance))
        & jnp.all(impedance > 0.0)
        & jnp.isfinite(source)
    )
    safe_impedance = jnp.where(
        jnp.isfinite(impedance) & (impedance > 0.0), impedance, 1.0
    )
    admittance = 1.0 / safe_impedance
    common_pressure = (source + jnp.sum(admittance * incident)) / jnp.sum(admittance)
    branch_flow = (common_pressure - incident) / safe_impedance
    conservation_residual = jnp.sum(branch_flow) - source
    reconstructed_pressure = incident + safe_impedance * branch_flow
    pressure_residual = jnp.max(jnp.abs(reconstructed_pressure - common_pressure))
    nan = jnp.asarray(jnp.nan, dtype=incident.dtype)
    return VascularJunctionResult(
        common_pressure_kPa=jnp.where(valid, common_pressure, nan),
        branch_flow_away_mm3_per_ms=jnp.where(valid, branch_flow, nan),
        conservation_residual_mm3_per_ms=jnp.where(valid, conservation_residual, nan),
        pressure_residual_kPa=jnp.where(valid, pressure_residual, nan),
        successful=valid,
    )


class Vascular0DPort(StrictModule, NonTrainableState):
    """Stable pressure/flow endpoint for coupling a vessel to a 0D component."""

    vessel_id: str = eqx.field(static=True)
    side: VascularPortSide = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(self, vessel_id: str, side: VascularPortSide, /):
        identifier = str(vessel_id)
        if not identifier or side not in ("inlet", "outlet"):
            raise ValueError("A vascular port needs a vessel ID and inlet/outlet side.")
        self.vessel_id = identifier
        self.side = side
        self.port_id = f"{identifier}.{side}"


class VascularPortCoupling(StrictModule):
    """Characteristic-compatible boundary state and signed coupling flow."""

    boundary: VascularBoundaryState
    pressure_kPa: Array
    axial_flow_mm3_per_ms: Array
    flow_into_vessel_mm3_per_ms: Array
    successful: Array


def couple_0d_pressure_port(
    port: Vascular0DPort,
    tube_law: SquareRootTubeLaw,
    density_mg_per_mm3: float,
    interior_area_mm2: ArrayLike,
    interior_flow_mm3_per_ms: ArrayLike,
    port_pressure_kPa: ArrayLike,
    /,
) -> VascularPortCoupling:
    """Impose 0D pressure through the incoming vessel characteristic.

    Axial flow is positive from vessel inlet to outlet. ``flow_into_vessel`` is
    positive from the connected 0D network into the vessel at either endpoint.
    """
    area = jnp.asarray(interior_area_mm2)
    flow = jnp.asarray(interior_flow_mm3_per_ms, dtype=area.dtype).reshape(())
    pressure = jnp.asarray(port_pressure_kPa, dtype=area.dtype).reshape(())
    density = float(density_mg_per_mm3)
    if not isfinite(density) or density <= 0.0:
        raise ValueError("density_mg_per_mm3 must be finite and positive.")
    area_scalar = area.reshape(())
    area_valid = tube_law.valid_area(area_scalar)
    pressure_valid = tube_law.valid_pressure(pressure)
    finite = jnp.isfinite(flow)
    valid = area_valid & pressure_valid & finite
    safe_area = jnp.where(area_valid, area_scalar, tube_law.reference_area_mm2)
    safe_pressure = jnp.where(pressure_valid, pressure, tube_law.reference_pressure_kPa)
    interior_pressure = tube_law.pressure(safe_area)
    impedance = tube_law.characteristic_impedance(safe_area, density)
    direction = 1.0 if port.side == "inlet" else -1.0
    axial_flow = flow + direction * (safe_pressure - interior_pressure) / impedance
    boundary_area = tube_law.area(safe_pressure)
    flow_into_vessel = direction * axial_flow
    nan = jnp.asarray(jnp.nan, dtype=area.dtype)
    boundary = VascularBoundaryState(
        area_mm2=jnp.where(valid, boundary_area, nan),
        flow_mm3_per_ms=jnp.where(valid, axial_flow, nan),
    )
    return VascularPortCoupling(
        boundary=boundary,
        pressure_kPa=jnp.where(valid, pressure, nan),
        axial_flow_mm3_per_ms=jnp.where(valid, axial_flow, nan),
        flow_into_vessel_mm3_per_ms=jnp.where(valid, flow_into_vessel, nan),
        successful=valid,
    )


__all__ = [
    "CharacteristicTerminal",
    "PreparedVascular1D",
    "SquareRootTubeLaw",
    "TerminalReflection",
    "Vascular0DPort",
    "Vascular1DPlan",
    "Vascular1DState",
    "VascularBoundaryState",
    "VascularJunctionPlan",
    "VascularJunctionResult",
    "VascularPortCoupling",
    "VascularPortSide",
    "VascularStepEvidence",
    "VascularStepResult",
    "VascularStepStatus",
    "couple_0d_pressure_port",
    "initialize_vascular_state",
    "reflect_characteristic_wave",
    "solve_vascular_junction",
    "step_vascular_1d",
    "vascular_numerical_flux",
]
