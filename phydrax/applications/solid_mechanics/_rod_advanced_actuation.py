#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, FunctionLinearOperator
from ._rod_dynamics import RodState
from ._rod_materials import (
    RodConstitutiveControl,
    RodConstitutiveResult,
    RodConstitutiveTrial,
)
from ._rod_reduction import PreparedReducedRod, ReducedRodState
from ._rod_tendon import _rotation_matrices


PressureExclusion: TypeAlias = str
MagneticExclusion: TypeAlias = str
ConstitutiveExclusion: TypeAlias = str

_PRESSURE_EXCLUSIONS = (
    "deformable_cross_sections",
    "interacting_chambers",
    "valves",
    "compressors",
    "leakage",
    "mass_flow_networks",
    "thermal_networks",
    "vacuum",
    "volumetric_bodies",
)
_INTRINSIC_EXCLUSIONS = (
    "hysteresis",
    "shape_memory_phase_change",
    "dielectric_field_solve",
    "thermal_networks",
    "swelling_transport",
)
_STIFFNESS_EXCLUSIONS = (
    "jamming",
    "hysteresis",
    "phase_change",
    "rate_dependent_modulus",
    "damping_modulation",
    "topology_change",
)
_MAGNETIC_EXCLUSIONS = (
    "nonlinear_ferromagnetics",
    "hysteresis",
    "mutual_fields",
    "maxwell_solves",
    "coupled_rl_circuits",
)


def _identifier(value: str, owner: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{owner} must be nonempty.")
    return identifier


def _positive_finite(value: float, owner: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{owner} must be finite and positive.")
    return result


def _finite_pair(value: tuple[float, float], owner: str, /) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{owner} must be a pair.")
    lower, upper = float(value[0]), float(value[1])
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError(f"{owner} must be finite and ordered.")
    return lower, upper


def _real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _vector_parameter(
    name: str, value: ArrayLike | float, size: int, dtype: np.dtype, /
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape == ():
        array = np.full((size,), array, dtype=dtype)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or have shape ({size},).")
    return array


def _validate_psd(name: str, matrices: np.ndarray, /) -> None:
    if matrices.shape[0] == 0:
        return
    tolerance = 500.0 * np.finfo(matrices.dtype).eps
    scale = max(1.0, float(np.max(np.abs(matrices))))
    if not np.allclose(
        matrices,
        np.swapaxes(matrices, -1, -2),
        rtol=tolerance,
        atol=tolerance * scale,
    ):
        raise ValueError(f"{name} must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrices) < -tolerance * scale):
        raise ValueError(f"{name} must be positive semidefinite.")


def _reject_requested_exclusions(owner: str, **requested: bool) -> None:
    selected = tuple(name for name, enabled in requested.items() if bool(enabled))
    if selected:
        joined = ", ".join(selected)
        raise ValueError(f"{owner} explicitly excludes: {joined}.")


def _all_finite(*values: Array) -> Array:
    result = jnp.asarray(True)
    for value in values:
        result = result & jnp.all(jnp.isfinite(value))
    return result


def _power_balanced(residual: Array, *powers: Array, tolerance: float) -> Array:
    scale = jnp.asarray(1.0, dtype=residual.dtype)
    for power in powers:
        scale = jnp.maximum(scale, jnp.abs(power))
    return jnp.abs(residual) <= jnp.asarray(tolerance, dtype=residual.dtype) * scale


def _transition(
    source: Array,
    target: Array,
    lower: Array,
    upper: Array,
    rise_rate: Array,
    fall_rate: Array,
    time_step: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    positive_step = jnp.isfinite(time_step) & (time_step > 0.0)
    safe_step = jnp.where(positive_step, time_step, jnp.ones_like(time_step))
    requested_rate = (target - source) / safe_step
    limited_rate = jnp.clip(requested_rate, -fall_rate, rise_rate)
    unconstrained = source + time_step * limited_rate
    candidate = jnp.clip(unconstrained, lower, upper)
    applied_rate = (candidate - source) / safe_step
    source_margin = jnp.min(jnp.minimum(source - lower, upper - source))
    candidate_margin = jnp.min(jnp.minimum(candidate - lower, upper - candidate))
    saturated = jnp.any(candidate != target)
    finite = _all_finite(
        source, target, candidate, requested_rate, applied_rate, time_step
    )
    valid = finite & positive_step & (source_margin >= 0.0) & (candidate_margin >= 0.0)
    return (
        candidate,
        requested_rate,
        applied_rate,
        source_margin,
        candidate_margin,
        saturated,
        valid,
    )


class RodTubeStation(StrictModule, NonTrainableState):
    """Fixed eccentric chamber centerline station in a segment material frame."""

    offset_material: Array
    segment_id: int = eqx.field(static=True)
    xi: float = eqx.field(static=True)
    station_id: str = eqx.field(static=True)

    def __init__(self, segment_id: int, xi: float, offset_material: ArrayLike, /):
        if isinstance(segment_id, bool) or int(segment_id) != segment_id:
            raise TypeError("segment_id must be an integer.")
        segment = int(segment_id)
        coordinate = float(xi)
        offset = _real_array("offset_material", offset_material, 1)
        if segment < 0:
            raise ValueError("segment_id must be nonnegative.")
        if not isfinite(coordinate) or coordinate < 0.0 or coordinate > 1.0:
            raise ValueError("xi must lie in [0, 1].")
        if offset.shape != (3,):
            raise ValueError("Reduced tube offsets must be spatial three-vectors.")
        self.offset_material = jnp.asarray(offset)
        self.segment_id = segment
        self.xi = coordinate
        self.station_id = canonical_fingerprint(
            {
                "kind": "reduced-rod-tube-material-station",
                "segment": segment,
                "xi": coordinate,
                "offset_material_m": array_tree_fingerprint(offset),
            }
        )


class ReducedTubeChamberPlan(StrictModule, NonTrainableState):
    """Rigid-cross-section eccentric tube reduction with closed end caps.

    The calibrated volume is ``dead_volume + sum(area[j] * span_length[j])``.
    It is deliberately not a deformable chamber or volumetric-body model.
    """

    stations: tuple[RodTubeStation, ...]
    cross_section_areas: Array
    dead_volume: float = eqx.field(static=True)
    minimum_volume: float = eqx.field(static=True)
    maximum_volume: float = eqx.field(static=True)
    ambient_pressure: float = eqx.field(static=True)
    minimum_span_length: float = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    unit_system: str = eqx.field(static=True)
    excluded_capabilities: tuple[PressureExclusion, ...] = eqx.field(static=True)

    def __init__(
        self,
        stations: tuple[RodTubeStation, ...],
        cross_section_areas: ArrayLike,
        dead_volume: float,
        /,
        *,
        volume_bounds: tuple[float, float],
        ambient_pressure: float,
        source_manifest_id: str,
        calibration_id: str,
        minimum_span_length: float = 1.0e-9,
        closed_caps: bool = True,
        deformable_cross_sections: bool = False,
        interacting_chambers: bool = False,
        valves: bool = False,
        compressors: bool = False,
        leakage: bool = False,
        mass_flow_networks: bool = False,
        thermal_networks: bool = False,
        vacuum: bool = False,
        volumetric_bodies: bool = False,
    ):
        if not isinstance(stations, tuple) or len(stations) < 2:
            raise ValueError("stations must contain at least two RodTubeStation values.")
        if not all(isinstance(station, RodTubeStation) for station in stations):
            raise TypeError("Every chamber station must be a RodTubeStation.")
        if not closed_caps:
            raise ValueError("Reduced tube pressure requires the closed-cap convention.")
        _reject_requested_exclusions(
            "ReducedTubeChamberPlan",
            deformable_cross_sections=deformable_cross_sections,
            interacting_chambers=interacting_chambers,
            valves=valves,
            compressors=compressors,
            leakage=leakage,
            mass_flow_networks=mass_flow_networks,
            thermal_networks=thermal_networks,
            vacuum=vacuum,
            volumetric_bodies=volumetric_bodies,
        )
        areas = _real_array("cross_section_areas", cross_section_areas, 1)
        if areas.shape != (len(stations) - 1,) or np.any(areas <= 0.0):
            raise ValueError("Every chamber span must have one positive area.")
        if any(station.offset_material.dtype != areas.dtype for station in stations):
            raise TypeError("Chamber stations and areas must share a dtype.")
        dead = _positive_finite(dead_volume, "dead_volume")
        minimum, maximum = _finite_pair(volume_bounds, "volume_bounds")
        ambient = float(ambient_pressure)
        if minimum <= 0.0 or not isfinite(ambient) or ambient < 0.0:
            raise ValueError(
                "Volume bounds must be positive and ambient pressure nonnegative."
            )
        minimum_span = _positive_finite(minimum_span_length, "minimum_span_length")
        manifest = _identifier(source_manifest_id, "source_manifest_id")
        calibration = _identifier(calibration_id, "calibration_id")
        self.stations = stations
        self.cross_section_areas = jnp.asarray(areas)
        self.dead_volume = dead
        self.minimum_volume = minimum
        self.maximum_volume = maximum
        self.ambient_pressure = ambient
        self.minimum_span_length = minimum_span
        self.source_manifest_id = manifest
        self.calibration_id = calibration
        self.unit_system = "SI:m,m^2,m^3,Pa,N,N*m,W"
        self.excluded_capabilities = _PRESSURE_EXCLUSIONS
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rigid-cross-section-reduced-tube-chamber-plan",
                "stations": tuple(station.station_id for station in stations),
                "areas_m2": array_tree_fingerprint(areas),
                "dead_volume_m3": dead,
                "volume_bounds_m3": (minimum, maximum),
                "ambient_pressure_pa": ambient,
                "minimum_span_length_m": minimum_span,
                "closed_caps": True,
                "source_manifest": manifest,
                "calibration": calibration,
                "units": self.unit_system,
                "exclusions": self.excluded_capabilities,
            }
        )

    def prepare(self, rod: PreparedReducedRod, /) -> "PreparedReducedTubeChamber":
        return PreparedReducedTubeChamber(self, rod)


class PreparedReducedTubeChamber(StrictModule, NonTrainableState):
    """Prepared tube volume coordinate and exact native/reduced dual actions."""

    plan: ReducedTubeChamberPlan
    reduction: PreparedReducedRod
    segment_ids: Array
    start_node_ids: Array
    end_node_ids: Array
    xis: Array
    offsets: Array
    volume_rate_space: ArraySpace
    workset_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ReducedTubeChamberPlan, rod: PreparedReducedRod, /):
        if not isinstance(plan, ReducedTubeChamberPlan):
            raise TypeError("plan must be a ReducedTubeChamberPlan.")
        if not isinstance(rod, PreparedReducedRod):
            raise TypeError("Reduced tube pressure requires a PreparedReducedRod.")
        if rod.rod.plan.dimension != 3:
            raise ValueError("Reduced tube pressure supports spatial rods only.")
        segment_ids = np.asarray(
            tuple(station.segment_id for station in plan.stations), dtype=np.int32
        )
        if np.any(segment_ids >= rod.rod.plan.segment_count):
            raise ValueError("A chamber station references an absent rod segment.")
        topology = np.asarray(rod.rod.plan.segment_node_ids)
        dtype = np.dtype(rod.rod.plan.rest_positions.dtype)
        if np.dtype(plan.cross_section_areas.dtype) != dtype:
            raise TypeError(
                "Chamber calibration dtype must match the prepared rod dtype."
            )
        xis = np.asarray(tuple(station.xi for station in plan.stations), dtype=dtype)
        offsets = np.stack(
            tuple(np.asarray(station.offset_material) for station in plan.stations)
        )
        workset_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-tube-volume-workset",
                "rod": rod.prepared_id,
                "chamber": plan.plan_id,
                "segments": array_tree_fingerprint(segment_ids),
                "topology": array_tree_fingerprint(topology[segment_ids]),
            }
        )
        self.plan = plan
        self.reduction = rod
        self.segment_ids = jnp.asarray(segment_ids)
        self.start_node_ids = jnp.asarray(topology[segment_ids, 0])
        self.end_node_ids = jnp.asarray(topology[segment_ids, 1])
        self.xis = jnp.asarray(xis)
        self.offsets = jnp.asarray(offsets)
        self.volume_rate_space = ArraySpace(
            (),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {"kind": "reduced-tube-volume-rate-space", "workset": workset_id}
            ),
        )
        self.workset_id = workset_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-cross-section-reduced-tube-chamber",
                "rod": rod.prepared_id,
                "plan": plan.plan_id,
                "workset": workset_id,
                "coefficient_space": rod.coefficient_space.space_id,
                "native_effort_space": rod.native_effort_space.space_id,
                "reduced_effort_space": rod.reduced_effort_space.space_id,
            }
        )

    def _native_state(self, state: ReducedRodState, /) -> RodState:
        self.reduction.validate_state(state)
        return self.reduction.lift(state)

    def _points_frames(self, native: RodState, /) -> tuple[Array, Array]:
        positions, orientations = self.reduction.rod.configuration_from_state(native)
        frames = _rotation_matrices(orientations, 3)[self.segment_ids]
        centers = (1.0 - self.xis)[:, None] * positions[self.start_node_ids] + self.xis[
            :, None
        ] * positions[self.end_node_ids]
        offsets_world = ein.contract("sij,sj->si", frames, self.offsets)
        return centers + offsets_world, frames

    def geometry(self, state: ReducedRodState, /) -> tuple[Array, Array, Array, Array]:
        native = self._native_state(state)
        points, frames = self._points_frames(native)
        spans = points[1:] - points[:-1]
        lengths = jnp.sqrt(jnp.sum(spans * spans, axis=-1))
        directions = spans / jnp.where(lengths > 0.0, lengths, 1.0)[:, None]
        return points, frames, lengths, directions

    def volume(self, state: ReducedRodState, /) -> Array:
        _, _, lengths, _ = self.geometry(state)
        return jnp.asarray(self.plan.dead_volume, dtype=lengths.dtype) + jnp.sum(
            self.plan.cross_section_areas * lengths
        )

    def native_volume_rate_operator(
        self, state: ReducedRodState, /
    ) -> FunctionLinearOperator:
        native = self._native_state(state)
        points, frames = self._points_frames(native)
        spans = points[1:] - points[:-1]
        lengths = jnp.sqrt(jnp.sum(spans * spans, axis=-1))
        directions = spans / jnp.where(lengths > 0.0, lengths, 1.0)[:, None]
        areas = self.plan.cross_section_areas
        xis = self.xis
        start_ids = self.start_node_ids
        end_ids = self.end_node_ids
        segment_ids = self.segment_ids
        offsets = self.offsets
        node_count = self.reduction.rod.plan.node_count
        segment_count = self.reduction.rod.plan.segment_count

        def station_velocity(velocity):
            linear, angular = velocity
            centers = (1.0 - xis)[:, None] * linear[start_ids] + xis[:, None] * linear[
                end_ids
            ]
            offsets_world = ein.contract("sij,sj->si", frames, offsets)
            angular_world = ein.contract("sij,sj->si", frames, angular[segment_ids])
            return centers + jnp.cross(angular_world, offsets_world)

        def action(velocity):
            velocities = station_velocity(velocity)
            span_rates = jnp.sum(directions * (velocities[1:] - velocities[:-1]), axis=-1)
            return jnp.sum(areas * span_rates)

        def transpose_action(covector):
            pressure = jnp.asarray(covector)
            span_efforts = pressure * areas[:, None] * directions
            station_efforts = jnp.zeros_like(points)
            station_efforts = station_efforts.at[:-1].add(-span_efforts)
            station_efforts = station_efforts.at[1:].add(span_efforts)
            forces = jnp.zeros((node_count, 3), dtype=points.dtype)
            forces = forces.at[start_ids].add((1.0 - xis)[:, None] * station_efforts)
            forces = forces.at[end_ids].add(xis[:, None] * station_efforts)
            material_efforts = ein.contract("sji,sj->si", frames, station_efforts)
            station_moments = jnp.cross(offsets, material_efforts)
            moments = jnp.zeros((segment_count, 3), dtype=points.dtype)
            moments = moments.at[segment_ids].add(station_moments)
            return forces, moments

        return FunctionLinearOperator(
            action,
            source=self.reduction.native_velocity_space,
            target=self.volume_rate_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "native-reduced-tube-volume-rate-operator",
                    "chamber": self.prepared_id,
                }
            ),
        )

    def reduced_volume_rate_operator(
        self, state: ReducedRodState, /
    ) -> FunctionLinearOperator:
        native_operator = self.native_volume_rate_operator(state)
        lift = self.reduction.lift_velocity_operator(state.coefficients)

        def action(rate):
            return native_operator.mv(lift.mv(rate))

        def transpose_action(covector):
            return lift.transpose_mv(native_operator.transpose_mv(covector))

        return FunctionLinearOperator(
            action,
            source=self.reduction.coefficient_space,
            target=self.volume_rate_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "reduced-tube-volume-rate-operator",
                    "chamber": self.prepared_id,
                }
            ),
        )

    def evaluate_mechanics(
        self, state: ReducedRodState, gauge_pressure: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        native = self._native_state(state)
        operator = self.native_volume_rate_operator(state)
        volume_rate = operator.mv(self.reduction.rod.velocity_from_state(native))
        forces, moments = self.reduction.native_effort_space.validate(
            operator.transpose_mv(gauge_pressure)
        )
        reduced_operator = self.reduced_volume_rate_operator(state)
        reduced_effort = self.reduction.reduced_effort_space.validate(
            reduced_operator.transpose_mv(gauge_pressure)
        )
        native_power = self.reduction.native_effort_space.pair(
            (forces, moments), self.reduction.rod.velocity_from_state(native)
        ).real
        reduced_power = self.reduction.reduced_effort_space.pair(
            reduced_effort, state.coefficient_velocities
        ).real
        mechanical_power = gauge_pressure * volume_rate
        native_residual = native_power - mechanical_power
        reduced_residual = reduced_power - mechanical_power
        return (
            self.volume(state),
            volume_rate,
            forces,
            moments,
            reduced_effort,
            native_power,
            reduced_power,
            native_residual,
            reduced_residual,
        )


class RegulatedTubePressureState(StrictModule):
    """Accepted scalar gauge pressure in pascals."""

    gauge_pressure: Array

    def __init__(self, gauge_pressure: ArrayLike, /):
        value = jnp.asarray(gauge_pressure)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("gauge_pressure must be a real inexact scalar.")
        self.gauge_pressure = value


class RegulatedTubePressureCommand(StrictModule):
    """Requested scalar gauge-pressure target in pascals."""

    target_gauge_pressure: Array

    def __init__(self, target_gauge_pressure: ArrayLike, /):
        value = jnp.asarray(target_gauge_pressure)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("target_gauge_pressure must be a real inexact scalar.")
        self.target_gauge_pressure = value


class RegulatedTubePressureEvaluation(StrictModule):
    """Regulator candidate, exact tube effort, and boundary-parameter power."""

    candidate_state: RegulatedTubePressureState
    requested_gauge_pressure: Array
    applied_gauge_pressure: Array
    requested_pressure_rate: Array
    applied_pressure_rate: Array
    volume: Array
    volume_rate: Array
    native_forces: Array
    native_moments: Array
    reduced_effort: Array
    interaction_energy: Array
    stored_power: Array
    source_power: Array
    mechanical_power: Array
    native_mechanical_power: Array
    reduced_mechanical_power: Array
    power_residual: Array
    native_virtual_work_residual: Array
    reduced_virtual_work_residual: Array
    pressure_margin: Array
    candidate_pressure_margin: Array
    volume_margin: Array
    span_margin: Array
    saturated: Array
    finite: Array
    within_domain: Array
    power_balanced: Array
    valid: Array
    electrical_power_available: bool = eqx.field(static=True)
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class RegulatedReducedTubePressurePlan(StrictModule, NonTrainableState):
    """Rate- and magnitude-bounded prescribed gauge-pressure capability."""

    chamber: ReducedTubeChamberPlan
    minimum_pressure: float = eqx.field(static=True)
    maximum_pressure: float = eqx.field(static=True)
    maximum_rise_rate: float = eqx.field(static=True)
    maximum_fall_rate: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    excluded_capabilities: tuple[PressureExclusion, ...] = eqx.field(static=True)

    def __init__(
        self,
        chamber: ReducedTubeChamberPlan,
        /,
        *,
        pressure_bounds: tuple[float, float],
        maximum_rise_rate: float,
        maximum_fall_rate: float,
        power_tolerance: float = 1.0e-6,
        valves: bool = False,
        compressors: bool = False,
        leakage: bool = False,
        thermal_networks: bool = False,
    ):
        if not isinstance(chamber, ReducedTubeChamberPlan):
            raise TypeError("chamber must be a ReducedTubeChamberPlan.")
        _reject_requested_exclusions(
            "RegulatedReducedTubePressurePlan",
            valves=valves,
            compressors=compressors,
            leakage=leakage,
            thermal_networks=thermal_networks,
        )
        minimum, maximum = _finite_pair(pressure_bounds, "pressure_bounds")
        if minimum < 0.0:
            raise ValueError("Regulated tube pressure does not support vacuum.")
        rise = _positive_finite(maximum_rise_rate, "maximum_rise_rate")
        fall = _positive_finite(maximum_fall_rate, "maximum_fall_rate")
        tolerance = _positive_finite(power_tolerance, "power_tolerance")
        self.chamber = chamber
        self.minimum_pressure = minimum
        self.maximum_pressure = maximum
        self.maximum_rise_rate = rise
        self.maximum_fall_rate = fall
        self.power_tolerance = tolerance
        self.excluded_capabilities = _PRESSURE_EXCLUSIONS
        self.plan_id = canonical_fingerprint(
            {
                "kind": "regulated-reduced-tube-gauge-pressure-plan",
                "chamber": chamber.plan_id,
                "pressure_bounds_pa": (minimum, maximum),
                "pressure_rates_pa_per_s": (fall, rise),
                "power_tolerance": tolerance,
                "exclusions": self.excluded_capabilities,
            }
        )

    def prepare(
        self, rod: PreparedReducedRod, /
    ) -> "PreparedRegulatedReducedTubePressureActuation":
        return PreparedRegulatedReducedTubePressureActuation(
            self, self.chamber.prepare(rod)
        )


class PreparedRegulatedReducedTubePressureActuation(StrictModule, NonTrainableState):
    """Prepared prescribed-pressure actuator; no valve/compressor power claim."""

    plan: RegulatedReducedTubePressurePlan
    chamber: PreparedReducedTubeChamber
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RegulatedReducedTubePressurePlan,
        chamber: PreparedReducedTubeChamber,
        /,
    ):
        if chamber.plan.plan_id != plan.chamber.plan_id:
            raise ValueError("Prepared chamber does not belong to the pressure plan.")
        provenance = canonical_fingerprint(
            {
                "kind": "regulated-reduced-tube-pressure-provenance",
                "plan": plan.plan_id,
                "chamber": chamber.prepared_id,
                "rod": chamber.reduction.prepared_id,
                "source_manifest": plan.chamber.source_manifest_id,
                "calibration": plan.chamber.calibration_id,
                "units": plan.chamber.unit_system,
                "exclusions": plan.excluded_capabilities,
            }
        )
        self.plan = plan
        self.chamber = chamber
        self.provenance_id = provenance
        self.actuation_id = canonical_fingerprint(
            {"kind": "prepared-regulated-reduced-tube-pressure", "provenance": provenance}
        )

    def initialize_state(self, gauge_pressure: ArrayLike) -> RegulatedTubePressureState:
        state = RegulatedTubePressureState(gauge_pressure)
        if (
            np.dtype(state.gauge_pressure.dtype)
            != self.chamber.reduction.coefficient_space.dtype
        ):
            raise TypeError("Pressure state dtype must match the reduced rod dtype.")
        return state

    def evaluate(
        self,
        rod_state: ReducedRodState,
        state: RegulatedTubePressureState,
        command: RegulatedTubePressureCommand,
        time_step: ArrayLike,
        /,
    ) -> RegulatedTubePressureEvaluation:
        if not isinstance(state, RegulatedTubePressureState):
            raise TypeError("state must be a RegulatedTubePressureState.")
        if not isinstance(command, RegulatedTubePressureCommand):
            raise TypeError("command must be a RegulatedTubePressureCommand.")
        dtype = self.chamber.reduction.coefficient_space.dtype
        if (
            np.dtype(state.gauge_pressure.dtype) != dtype
            or np.dtype(command.target_gauge_pressure.dtype) != dtype
        ):
            raise TypeError(
                "Pressure state and command must match the reduced rod dtype."
            )
        step = jnp.asarray(time_step, dtype=dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        lower = jnp.asarray(self.plan.minimum_pressure, dtype=dtype)
        upper = jnp.asarray(self.plan.maximum_pressure, dtype=dtype)
        transition = _transition(
            state.gauge_pressure[None],
            command.target_gauge_pressure[None],
            lower[None],
            upper[None],
            jnp.asarray((self.plan.maximum_rise_rate,), dtype=dtype),
            jnp.asarray((self.plan.maximum_fall_rate,), dtype=dtype),
            step,
        )
        candidate_value = transition[0][0]
        requested_rate = transition[1][0]
        pressure_rate = transition[2][0]
        candidate_state = RegulatedTubePressureState(candidate_value)
        mechanics = self.chamber.evaluate_mechanics(rod_state, candidate_value)
        (
            volume,
            volume_rate,
            forces,
            moments,
            reduced_effort,
            native_power,
            reduced_power,
            native_residual,
            reduced_residual,
        ) = mechanics
        interaction_energy = -candidate_value * volume
        stored_power = -candidate_value * volume_rate - volume * pressure_rate
        source_power = -volume * pressure_rate
        mechanical_power = candidate_value * volume_rate
        power_residual = stored_power + mechanical_power - source_power
        pressure_margin = transition[3]
        candidate_pressure_margin = transition[4]
        _, _, span_lengths, _ = self.chamber.geometry(rod_state)
        volume_margin = jnp.minimum(
            volume - self.plan.chamber.minimum_volume,
            self.plan.chamber.maximum_volume - volume,
        )
        span_margin = jnp.min(span_lengths - self.plan.chamber.minimum_span_length)
        finite = _all_finite(
            state.gauge_pressure,
            command.target_gauge_pressure,
            candidate_value,
            requested_rate,
            pressure_rate,
            volume,
            volume_rate,
            forces,
            moments,
            reduced_effort,
            interaction_energy,
            stored_power,
            source_power,
            mechanical_power,
            native_power,
            reduced_power,
            power_residual,
        )
        within_domain = transition[6] & (volume_margin >= 0.0) & (span_margin >= 0.0)
        balanced = (
            _power_balanced(
                power_residual,
                stored_power,
                source_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
            & _power_balanced(
                native_residual,
                native_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
            & _power_balanced(
                reduced_residual,
                reduced_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
        )
        return RegulatedTubePressureEvaluation(
            candidate_state,
            command.target_gauge_pressure,
            candidate_value,
            requested_rate,
            pressure_rate,
            volume,
            volume_rate,
            forces,
            moments,
            reduced_effort,
            interaction_energy,
            stored_power,
            source_power,
            mechanical_power,
            native_power,
            reduced_power,
            power_residual,
            native_residual,
            reduced_residual,
            pressure_margin,
            candidate_pressure_margin,
            volume_margin,
            span_margin,
            transition[5],
            finite,
            within_domain,
            balanced,
            finite & within_domain & balanced,
            False,
            self.actuation_id,
            self.provenance_id,
        )


class SealedTubePressureState(StrictModule):
    """Accepted dimensionless fixed-gas-charge scale; no mass-flow state."""

    charge_scale: Array

    def __init__(self, charge_scale: ArrayLike, /):
        value = jnp.asarray(charge_scale)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("charge_scale must be a real inexact scalar.")
        self.charge_scale = value


class SealedTubePressureEvaluation(StrictModule):
    """Sealed polytropic gas mechanics and exact stored-energy derivative."""

    candidate_state: SealedTubePressureState
    absolute_pressure: Array
    gauge_pressure: Array
    volume: Array
    volume_rate: Array
    native_forces: Array
    native_moments: Array
    reduced_effort: Array
    stored_energy: Array
    stored_power: Array
    source_power: Array
    mechanical_power: Array
    native_mechanical_power: Array
    reduced_mechanical_power: Array
    power_residual: Array
    native_virtual_work_residual: Array
    reduced_virtual_work_residual: Array
    volume_margin: Array
    span_margin: Array
    finite: Array
    within_domain: Array
    pressure_nonnegative: Array
    power_balanced: Array
    valid: Array
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class SealedReducedTubePressurePlan(StrictModule, NonTrainableState):
    """Fixed-charge polytropic gas law over one reduced tube coordinate."""

    chamber: ReducedTubeChamberPlan
    reference_absolute_pressure: float = eqx.field(static=True)
    reference_volume: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    excluded_capabilities: tuple[PressureExclusion, ...] = eqx.field(static=True)

    def __init__(
        self,
        chamber: ReducedTubeChamberPlan,
        reference_absolute_pressure: float,
        reference_volume: float,
        /,
        *,
        exponent: float = 1.0,
        power_tolerance: float = 1.0e-6,
        valves: bool = False,
        compressors: bool = False,
        leakage: bool = False,
        thermal_networks: bool = False,
    ):
        if not isinstance(chamber, ReducedTubeChamberPlan):
            raise TypeError("chamber must be a ReducedTubeChamberPlan.")
        _reject_requested_exclusions(
            "SealedReducedTubePressurePlan",
            valves=valves,
            compressors=compressors,
            leakage=leakage,
            thermal_networks=thermal_networks,
        )
        pressure = _positive_finite(
            reference_absolute_pressure, "reference_absolute_pressure"
        )
        volume = _positive_finite(reference_volume, "reference_volume")
        exponent_ = float(exponent)
        if pressure <= chamber.ambient_pressure:
            raise ValueError("Reference gas pressure must exceed ambient pressure.")
        if not isfinite(exponent_) or exponent_ < 0.0:
            raise ValueError("exponent must be finite and nonnegative.")
        tolerance = _positive_finite(power_tolerance, "power_tolerance")
        self.chamber = chamber
        self.reference_absolute_pressure = pressure
        self.reference_volume = volume
        self.exponent = exponent_
        self.power_tolerance = tolerance
        self.excluded_capabilities = _PRESSURE_EXCLUSIONS
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sealed-polytropic-reduced-tube-pressure-plan",
                "chamber": chamber.plan_id,
                "reference_absolute_pressure_pa": pressure,
                "reference_volume_m3": volume,
                "exponent": exponent_,
                "power_tolerance": tolerance,
                "exclusions": self.excluded_capabilities,
            }
        )

    def prepare(
        self, rod: PreparedReducedRod, /
    ) -> "PreparedSealedReducedTubePressureActuation":
        return PreparedSealedReducedTubePressureActuation(self, self.chamber.prepare(rod))


class PreparedSealedReducedTubePressureActuation(StrictModule, NonTrainableState):
    """Prepared autonomous sealed-gas actuator with no command channel."""

    plan: SealedReducedTubePressurePlan
    chamber: PreparedReducedTubeChamber
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SealedReducedTubePressurePlan,
        chamber: PreparedReducedTubeChamber,
        /,
    ):
        if chamber.plan.plan_id != plan.chamber.plan_id:
            raise ValueError("Prepared chamber does not belong to the sealed plan.")
        provenance = canonical_fingerprint(
            {
                "kind": "sealed-reduced-tube-pressure-provenance",
                "plan": plan.plan_id,
                "chamber": chamber.prepared_id,
                "rod": chamber.reduction.prepared_id,
                "source_manifest": plan.chamber.source_manifest_id,
                "calibration": plan.chamber.calibration_id,
                "units": plan.chamber.unit_system,
                "exclusions": plan.excluded_capabilities,
            }
        )
        self.plan = plan
        self.chamber = chamber
        self.provenance_id = provenance
        self.actuation_id = canonical_fingerprint(
            {"kind": "prepared-sealed-reduced-tube-pressure", "provenance": provenance}
        )

    def initialize_state(self, charge_scale: ArrayLike = 1.0) -> SealedTubePressureState:
        state = SealedTubePressureState(charge_scale)
        if (
            np.dtype(state.charge_scale.dtype)
            != self.chamber.reduction.coefficient_space.dtype
        ):
            raise TypeError("Sealed gas state dtype must match the reduced rod dtype.")
        return state

    def evaluate(
        self, rod_state: ReducedRodState, state: SealedTubePressureState, /
    ) -> SealedTubePressureEvaluation:
        if not isinstance(state, SealedTubePressureState):
            raise TypeError("state must be a SealedTubePressureState.")
        dtype = self.chamber.reduction.coefficient_space.dtype
        if np.dtype(state.charge_scale.dtype) != dtype:
            raise TypeError("Sealed gas state dtype must match the reduced rod dtype.")
        volume = self.chamber.volume(rod_state)
        reference_volume = jnp.asarray(self.plan.reference_volume, dtype=dtype)
        reference_pressure = jnp.asarray(
            self.plan.reference_absolute_pressure, dtype=dtype
        )
        ambient = jnp.asarray(self.plan.chamber.ambient_pressure, dtype=dtype)
        ratio = volume / reference_volume
        absolute_pressure = (
            state.charge_scale * reference_pressure * ratio ** (-self.plan.exponent)
        )
        gauge_pressure = absolute_pressure - ambient
        mechanics = self.chamber.evaluate_mechanics(rod_state, gauge_pressure)
        (
            _,
            volume_rate,
            forces,
            moments,
            reduced_effort,
            native_power,
            reduced_power,
            native_residual,
            reduced_residual,
        ) = mechanics
        if self.plan.exponent == 1.0:
            gas_energy = (
                -state.charge_scale
                * reference_pressure
                * reference_volume
                * jnp.log(ratio)
            )
        else:
            gas_energy = (
                state.charge_scale
                * reference_pressure
                * reference_volume
                * (ratio ** (1.0 - self.plan.exponent) - 1.0)
                / (self.plan.exponent - 1.0)
            )
        stored_energy = gas_energy + ambient * (volume - reference_volume)
        stored_power = -gauge_pressure * volume_rate
        source_power = jnp.asarray(0.0, dtype=dtype)
        mechanical_power = gauge_pressure * volume_rate
        power_residual = stored_power + mechanical_power - source_power
        _, _, span_lengths, _ = self.chamber.geometry(rod_state)
        volume_margin = jnp.minimum(
            volume - self.plan.chamber.minimum_volume,
            self.plan.chamber.maximum_volume - volume,
        )
        span_margin = jnp.min(span_lengths - self.plan.chamber.minimum_span_length)
        finite = _all_finite(
            state.charge_scale,
            absolute_pressure,
            gauge_pressure,
            volume,
            volume_rate,
            forces,
            moments,
            reduced_effort,
            stored_energy,
            stored_power,
            mechanical_power,
            native_power,
            reduced_power,
            power_residual,
        )
        within_domain = (
            (state.charge_scale > 0.0) & (volume_margin >= 0.0) & (span_margin >= 0.0)
        )
        pressure_nonnegative = gauge_pressure >= 0.0
        balanced = (
            _power_balanced(
                power_residual,
                stored_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
            & _power_balanced(
                native_residual,
                native_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
            & _power_balanced(
                reduced_residual,
                reduced_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
        )
        return SealedTubePressureEvaluation(
            state,
            absolute_pressure,
            gauge_pressure,
            volume,
            volume_rate,
            forces,
            moments,
            reduced_effort,
            stored_energy,
            stored_power,
            source_power,
            mechanical_power,
            native_power,
            reduced_power,
            power_residual,
            native_residual,
            reduced_residual,
            volume_margin,
            span_margin,
            finite,
            within_domain,
            pressure_nonnegative,
            balanced,
            finite & within_domain & pressure_nonnegative & balanced,
            self.actuation_id,
            self.provenance_id,
        )


class IntrinsicStrainActuationState(StrictModule):
    """Accepted fixed-shape intrinsic-strain activation vector."""

    activation: Array

    def __init__(self, activation: ArrayLike, /):
        value = jnp.asarray(activation)
        if value.ndim != 1 or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("activation must be a real inexact rank-one array.")
        self.activation = value


class IntrinsicStrainCommand(StrictModule):
    """Requested fixed-shape intrinsic-strain activation target."""

    target_activation: Array

    def __init__(self, target_activation: ArrayLike, /):
        value = jnp.asarray(target_activation)
        if value.ndim != 1 or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("target_activation must be a real inexact rank-one array.")
        self.target_activation = value


class IntrinsicStrainCandidate(StrictModule):
    candidate_state: IntrinsicStrainActuationState
    requested_rate: Array
    applied_rate: Array
    intrinsic_strain: Array
    intrinsic_strain_rate: Array
    control: RodConstitutiveControl
    source_margin: Array
    candidate_margin: Array
    saturated: Array
    finite: Array
    valid: Array
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class IntrinsicStrainEvaluation(StrictModule):
    """One controlled material trial; candidate state remains uncommitted."""

    candidate: IntrinsicStrainCandidate
    material_result: RodConstitutiveResult
    stored_energy: Array
    stored_power: Array
    source_power: Array
    mechanical_power: Array
    dissipation_power: Array
    power_residual: Array
    finite: Array
    power_balanced: Array
    valid: Array
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class IntrinsicStrainActuationPlan(StrictModule, NonTrainableState):
    """Calibrated sitewise intrinsic-strain modes with bounded activation."""

    mode_shapes: Array
    lower_activation: Array
    upper_activation: Array
    maximum_rise_rate: Array
    maximum_fall_rate: Array
    channel_count: int = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    excluded_capabilities: tuple[ConstitutiveExclusion, ...] = eqx.field(static=True)

    def __init__(
        self,
        mode_shapes: ArrayLike,
        /,
        *,
        activation_bounds: tuple[ArrayLike | float, ArrayLike | float] = (0.0, 1.0),
        maximum_rise_rate: ArrayLike | float,
        maximum_fall_rate: ArrayLike | float,
        source_manifest_id: str,
        calibration_id: str,
        hysteresis: bool = False,
        shape_memory_phase_change: bool = False,
        dielectric_field_solve: bool = False,
        thermal_networks: bool = False,
        swelling_transport: bool = False,
    ):
        _reject_requested_exclusions(
            "IntrinsicStrainActuationPlan",
            hysteresis=hysteresis,
            shape_memory_phase_change=shape_memory_phase_change,
            dielectric_field_solve=dielectric_field_solve,
            thermal_networks=thermal_networks,
            swelling_transport=swelling_transport,
        )
        modes = _real_array("mode_shapes", mode_shapes, 3)
        channels = int(modes.shape[2])
        if channels < 1:
            raise ValueError("Intrinsic strain requires at least one control channel.")
        dtype = modes.dtype
        lower = _vector_parameter(
            "lower activation", activation_bounds[0], channels, dtype
        )
        upper = _vector_parameter(
            "upper activation", activation_bounds[1], channels, dtype
        )
        rise = _vector_parameter("maximum_rise_rate", maximum_rise_rate, channels, dtype)
        fall = _vector_parameter("maximum_fall_rate", maximum_fall_rate, channels, dtype)
        if np.any(lower > upper) or np.any(rise <= 0.0) or np.any(fall <= 0.0):
            raise ValueError("Activation bounds and rate limits are invalid.")
        manifest = _identifier(source_manifest_id, "source_manifest_id")
        calibration = _identifier(calibration_id, "calibration_id")
        self.mode_shapes = jnp.asarray(modes)
        self.lower_activation = jnp.asarray(lower)
        self.upper_activation = jnp.asarray(upper)
        self.maximum_rise_rate = jnp.asarray(rise)
        self.maximum_fall_rate = jnp.asarray(fall)
        self.channel_count = channels
        self.source_manifest_id = manifest
        self.calibration_id = calibration
        self.excluded_capabilities = _INTRINSIC_EXCLUSIONS
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-intrinsic-rod-strain-plan",
                "mode_shapes": array_tree_fingerprint(modes),
                "activation_bounds": array_tree_fingerprint(
                    {"lower": lower, "upper": upper}
                ),
                "rate_limits": array_tree_fingerprint({"rise": rise, "fall": fall}),
                "source_manifest": manifest,
                "calibration": calibration,
                "units": "activation:1;stretch/shear:1;bend/twist:m^-1;s^-1",
                "exclusions": self.excluded_capabilities,
            }
        )

    def prepare(
        self, material: RodConstitutiveTrial, /
    ) -> "PreparedIntrinsicStrainActuation":
        return PreparedIntrinsicStrainActuation(self, material)


class PreparedIntrinsicStrainActuation(StrictModule, NonTrainableState):
    plan: IntrinsicStrainActuationPlan
    material: RodConstitutiveTrial
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self, plan: IntrinsicStrainActuationPlan, material: RodConstitutiveTrial, /
    ):
        if not isinstance(material, RodConstitutiveTrial):
            raise TypeError("material must be a prepared RodConstitutiveTrial.")
        expected = (
            material.workset.site_count,
            material.workset.component_count,
            plan.channel_count,
        )
        if plan.mode_shapes.shape != expected:
            raise ValueError(f"Intrinsic mode_shapes must have shape {expected}.")
        if plan.mode_shapes.dtype != material.workset.reference_strains.dtype:
            raise TypeError("Intrinsic modes must match the material workset dtype.")
        provenance = canonical_fingerprint(
            {
                "kind": "intrinsic-strain-actuation-provenance",
                "plan": plan.plan_id,
                "material": material.material_id,
                "workset": material.workset.workset_id,
                "source_manifest": plan.source_manifest_id,
                "calibration": plan.calibration_id,
                "exclusions": plan.excluded_capabilities,
            }
        )
        self.plan = plan
        self.material = material
        self.provenance_id = provenance
        self.actuation_id = canonical_fingerprint(
            {"kind": "prepared-intrinsic-strain-actuation", "provenance": provenance}
        )

    def initialize_state(
        self, activation: ArrayLike | None = None
    ) -> IntrinsicStrainActuationState:
        value = (
            self.plan.lower_activation if activation is None else jnp.asarray(activation)
        )
        state = IntrinsicStrainActuationState(value)
        self._validate_state_command(state.activation, "activation")
        return state

    def _validate_state_command(self, value: Array, owner: str) -> None:
        if value.shape != (self.plan.channel_count,):
            raise ValueError(f"{owner} must have shape ({self.plan.channel_count},).")
        if value.dtype != self.plan.mode_shapes.dtype:
            raise TypeError(f"{owner} must match the intrinsic mode dtype.")

    def candidate_control(
        self,
        state: IntrinsicStrainActuationState,
        command: IntrinsicStrainCommand,
        time_step: ArrayLike,
        /,
    ) -> IntrinsicStrainCandidate:
        if not isinstance(state, IntrinsicStrainActuationState):
            raise TypeError("state must be an IntrinsicStrainActuationState.")
        if not isinstance(command, IntrinsicStrainCommand):
            raise TypeError("command must be an IntrinsicStrainCommand.")
        self._validate_state_command(state.activation, "activation")
        self._validate_state_command(command.target_activation, "target_activation")
        step = jnp.asarray(time_step, dtype=self.plan.mode_shapes.dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        transition = _transition(
            state.activation,
            command.target_activation,
            self.plan.lower_activation,
            self.plan.upper_activation,
            self.plan.maximum_rise_rate,
            self.plan.maximum_fall_rate,
            step,
        )
        candidate_state = IntrinsicStrainActuationState(transition[0])
        intrinsic = ein.contract("sdc,c->sd", self.plan.mode_shapes, transition[0])
        intrinsic_rate = ein.contract("sdc,c->sd", self.plan.mode_shapes, transition[2])
        passive = self.material.initialize_control()
        control = RodConstitutiveControl(
            intrinsic,
            intrinsic_rate,
            passive.stiffness,
            passive.stiffness_rate,
            material_id=self.material.material_id,
            workset_id=self.material.workset.workset_id,
            control_id=canonical_fingerprint(
                {
                    "kind": "intrinsic-strain-rod-constitutive-control",
                    "actuation": self.actuation_id,
                    "workset": self.material.workset.workset_id,
                }
            ),
            intrinsic_owner_id=self.actuation_id,
        )
        finite = _all_finite(intrinsic, intrinsic_rate)
        return IntrinsicStrainCandidate(
            candidate_state,
            transition[1],
            transition[2],
            intrinsic,
            intrinsic_rate,
            control,
            transition[3],
            transition[4],
            transition[5],
            finite,
            finite & transition[6],
            self.actuation_id,
            self.provenance_id,
        )

    def evaluate(
        self,
        source_strain: ArrayLike,
        candidate_strain: ArrayLike,
        strain_rate: ArrayLike,
        source_history: ArrayLike,
        state: IntrinsicStrainActuationState,
        command: IntrinsicStrainCommand,
        time: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> IntrinsicStrainEvaluation:
        candidate = self.candidate_control(state, command, time_step)
        result = self.material(
            source_strain,
            candidate_strain,
            strain_rate,
            source_history,
            candidate.control,
            time,
            time_step,
        )
        finite = candidate.finite & result.evidence.finite
        return IntrinsicStrainEvaluation(
            candidate,
            result,
            result.stored_energy,
            result.stored_energy_rate,
            result.control_source_power,
            result.mechanical_power,
            result.viscous_dissipation_power,
            result.power_residual,
            finite,
            result.evidence.power_balanced,
            candidate.valid & result.evidence.valid,
            self.actuation_id,
            self.provenance_id,
        )


class VariableStiffnessState(StrictModule):
    """Accepted scalar interpolation activation."""

    activation: Array

    def __init__(self, activation: ArrayLike, /):
        value = jnp.asarray(activation)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("activation must be a real inexact scalar.")
        self.activation = value


class VariableStiffnessCommand(StrictModule):
    """Requested scalar stiffness interpolation target."""

    target_activation: Array

    def __init__(self, target_activation: ArrayLike, /):
        value = jnp.asarray(target_activation)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("target_activation must be a real inexact scalar.")
        self.target_activation = value


class VariableStiffnessCandidate(StrictModule):
    candidate_state: VariableStiffnessState
    requested_rate: Array
    applied_rate: Array
    effective_stiffness: Array
    stiffness_rate: Array
    control: RodConstitutiveControl
    source_margin: Array
    candidate_margin: Array
    minimum_eigenvalue: Array
    saturated: Array
    finite: Array
    psd: Array
    valid: Array
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class VariableStiffnessEvaluation(StrictModule):
    """One bounded PSD material trial; candidate state remains uncommitted."""

    candidate: VariableStiffnessCandidate
    material_result: RodConstitutiveResult
    stored_energy: Array
    stored_power: Array
    source_power: Array
    mechanical_power: Array
    dissipation_power: Array
    power_residual: Array
    finite: Array
    power_balanced: Array
    valid: Array
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class VariableStiffnessActuationPlan(StrictModule, NonTrainableState):
    """Reversible bounded interpolation between calibrated PSD tensors."""

    minimum_stiffness: Array
    maximum_stiffness: Array
    maximum_rise_rate: float = eqx.field(static=True)
    maximum_fall_rate: float = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    excluded_capabilities: tuple[ConstitutiveExclusion, ...] = eqx.field(static=True)

    def __init__(
        self,
        minimum_stiffness: ArrayLike,
        maximum_stiffness: ArrayLike,
        /,
        *,
        maximum_rise_rate: float,
        maximum_fall_rate: float,
        source_manifest_id: str,
        calibration_id: str,
        jamming: bool = False,
        hysteresis: bool = False,
        phase_change: bool = False,
        rate_dependent_modulus: bool = False,
        damping_modulation: bool = False,
        topology_change: bool = False,
    ):
        _reject_requested_exclusions(
            "VariableStiffnessActuationPlan",
            jamming=jamming,
            hysteresis=hysteresis,
            phase_change=phase_change,
            rate_dependent_modulus=rate_dependent_modulus,
            damping_modulation=damping_modulation,
            topology_change=topology_change,
        )
        minimum = _real_array("minimum_stiffness", minimum_stiffness, 3)
        maximum = _real_array("maximum_stiffness", maximum_stiffness, 3)
        if minimum.shape != maximum.shape or minimum.shape[1] != minimum.shape[2]:
            raise ValueError("Stiffness endpoints must be equal-shaped square tensors.")
        if minimum.dtype != maximum.dtype:
            raise TypeError("Stiffness endpoints must share a dtype.")
        _validate_psd("minimum_stiffness", minimum)
        _validate_psd("maximum_stiffness", maximum)
        _validate_psd("maximum_stiffness - minimum_stiffness", maximum - minimum)
        rise = _positive_finite(maximum_rise_rate, "maximum_rise_rate")
        fall = _positive_finite(maximum_fall_rate, "maximum_fall_rate")
        manifest = _identifier(source_manifest_id, "source_manifest_id")
        calibration = _identifier(calibration_id, "calibration_id")
        self.minimum_stiffness = jnp.asarray(minimum)
        self.maximum_stiffness = jnp.asarray(maximum)
        self.maximum_rise_rate = rise
        self.maximum_fall_rate = fall
        self.source_manifest_id = manifest
        self.calibration_id = calibration
        self.excluded_capabilities = _STIFFNESS_EXCLUSIONS
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bounded-psd-variable-rod-stiffness-plan",
                "endpoints": array_tree_fingerprint(
                    {"minimum": minimum, "maximum": maximum}
                ),
                "rate_limits": (fall, rise),
                "source_manifest": manifest,
                "calibration": calibration,
                "units": "stretch/shear:N;bend/twist:N*m^2;activation:1",
                "exclusions": self.excluded_capabilities,
            }
        )

    def prepare(
        self, material: RodConstitutiveTrial, /
    ) -> "PreparedVariableStiffnessActuation":
        return PreparedVariableStiffnessActuation(self, material)


class PreparedVariableStiffnessActuation(StrictModule, NonTrainableState):
    plan: VariableStiffnessActuationPlan
    material: RodConstitutiveTrial
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self, plan: VariableStiffnessActuationPlan, material: RodConstitutiveTrial, /
    ):
        if not isinstance(material, RodConstitutiveTrial):
            raise TypeError("material must be a prepared RodConstitutiveTrial.")
        expected = (
            material.workset.site_count,
            material.workset.component_count,
            material.workset.component_count,
        )
        if plan.minimum_stiffness.shape != expected:
            raise ValueError(f"Stiffness endpoints must have shape {expected}.")
        if plan.minimum_stiffness.dtype != material.workset.reference_strains.dtype:
            raise TypeError("Stiffness endpoints must match the material workset dtype.")
        provenance = canonical_fingerprint(
            {
                "kind": "variable-stiffness-actuation-provenance",
                "plan": plan.plan_id,
                "material": material.material_id,
                "workset": material.workset.workset_id,
                "source_manifest": plan.source_manifest_id,
                "calibration": plan.calibration_id,
                "exclusions": plan.excluded_capabilities,
            }
        )
        self.plan = plan
        self.material = material
        self.provenance_id = provenance
        self.actuation_id = canonical_fingerprint(
            {"kind": "prepared-variable-stiffness-actuation", "provenance": provenance}
        )

    def initialize_state(self, activation: ArrayLike = 0.0) -> VariableStiffnessState:
        state = VariableStiffnessState(activation)
        if state.activation.dtype != self.plan.minimum_stiffness.dtype:
            raise TypeError("Variable stiffness state must match the calibration dtype.")
        return state

    def candidate_control(
        self,
        state: VariableStiffnessState,
        command: VariableStiffnessCommand,
        time_step: ArrayLike,
        /,
    ) -> VariableStiffnessCandidate:
        if not isinstance(state, VariableStiffnessState):
            raise TypeError("state must be a VariableStiffnessState.")
        if not isinstance(command, VariableStiffnessCommand):
            raise TypeError("command must be a VariableStiffnessCommand.")
        dtype = self.plan.minimum_stiffness.dtype
        if state.activation.dtype != dtype or command.target_activation.dtype != dtype:
            raise TypeError("Variable stiffness state and command must match plan dtype.")
        step = jnp.asarray(time_step, dtype=dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        transition = _transition(
            state.activation[None],
            command.target_activation[None],
            jnp.asarray((0.0,), dtype=dtype),
            jnp.asarray((1.0,), dtype=dtype),
            jnp.asarray((self.plan.maximum_rise_rate,), dtype=dtype),
            jnp.asarray((self.plan.maximum_fall_rate,), dtype=dtype),
            step,
        )
        activation = transition[0][0]
        activation_rate = transition[2][0]
        delta = self.plan.maximum_stiffness - self.plan.minimum_stiffness
        stiffness = self.plan.minimum_stiffness + activation * delta
        stiffness_rate = activation_rate * delta
        passive = self.material.initialize_control()
        control = RodConstitutiveControl(
            passive.intrinsic_strain,
            passive.intrinsic_strain_rate,
            stiffness,
            stiffness_rate,
            material_id=self.material.material_id,
            workset_id=self.material.workset.workset_id,
            control_id=canonical_fingerprint(
                {
                    "kind": "variable-stiffness-rod-constitutive-control",
                    "actuation": self.actuation_id,
                    "workset": self.material.workset.workset_id,
                }
            ),
            stiffness_owner_id=self.actuation_id,
        )
        minimum_eigenvalue = jnp.min(jnp.linalg.eigvalsh(stiffness))
        tolerance = (
            500.0 * jnp.finfo(dtype).eps * jnp.maximum(1.0, jnp.max(jnp.abs(stiffness)))
        )
        psd = minimum_eigenvalue >= -tolerance
        finite = _all_finite(stiffness, stiffness_rate, minimum_eigenvalue)
        candidate_state = VariableStiffnessState(activation)
        return VariableStiffnessCandidate(
            candidate_state,
            transition[1][0],
            activation_rate,
            stiffness,
            stiffness_rate,
            control,
            transition[3],
            transition[4],
            minimum_eigenvalue,
            transition[5],
            finite,
            psd,
            finite & psd & transition[6],
            self.actuation_id,
            self.provenance_id,
        )

    def evaluate(
        self,
        source_strain: ArrayLike,
        candidate_strain: ArrayLike,
        strain_rate: ArrayLike,
        source_history: ArrayLike,
        state: VariableStiffnessState,
        command: VariableStiffnessCommand,
        time: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> VariableStiffnessEvaluation:
        candidate = self.candidate_control(state, command, time_step)
        result = self.material(
            source_strain,
            candidate_strain,
            strain_rate,
            source_history,
            candidate.control,
            time,
            time_step,
        )
        finite = candidate.finite & result.evidence.finite
        return VariableStiffnessEvaluation(
            candidate,
            result,
            result.stored_energy,
            result.stored_energy_rate,
            result.control_source_power,
            result.mechanical_power,
            result.viscous_dissipation_power,
            result.power_residual,
            finite,
            result.evidence.power_balanced,
            candidate.valid & result.evidence.valid,
            self.actuation_id,
            self.provenance_id,
        )


def combine_rod_constitutive_controls(
    first: RodConstitutiveControl,
    second: RodConstitutiveControl,
    /,
) -> RodConstitutiveControl:
    """Compose orthogonal control owners without evaluating material twice."""
    if not isinstance(first, RodConstitutiveControl) or not isinstance(
        second, RodConstitutiveControl
    ):
        raise TypeError("Both values must be RodConstitutiveControl instances.")
    if first.workset_id != second.workset_id or first.material_id != second.material_id:
        raise ValueError("Constitutive controls belong to different materials/worksets.")
    if first.intrinsic_owner_id is not None and second.intrinsic_owner_id is not None:
        raise ValueError("Overlapping intrinsic-strain owners require an explicit rule.")
    if first.stiffness_owner_id is not None and second.stiffness_owner_id is not None:
        raise ValueError("Overlapping stiffness owners require an explicit rule.")
    intrinsic_owner = first.intrinsic_owner_id or second.intrinsic_owner_id
    stiffness_owner = first.stiffness_owner_id or second.stiffness_owner_id
    intrinsic_source = first if first.intrinsic_owner_id is not None else second
    stiffness_source = first if first.stiffness_owner_id is not None else second
    return RodConstitutiveControl(
        intrinsic_source.intrinsic_strain,
        intrinsic_source.intrinsic_strain_rate,
        stiffness_source.stiffness,
        stiffness_source.stiffness_rate,
        material_id=first.material_id,
        workset_id=first.workset_id,
        control_id=canonical_fingerprint(
            {
                "kind": "composed-rod-constitutive-control",
                "workset": first.workset_id,
                "intrinsic_owner": intrinsic_owner,
                "stiffness_owner": stiffness_owner,
            }
        ),
        intrinsic_owner_id=intrinsic_owner,
        stiffness_owner_id=stiffness_owner,
    )


class MagneticCurrentState(StrictModule):
    """Accepted fixed-shape coil-current vector in amperes."""

    currents: Array

    def __init__(self, currents: ArrayLike, /):
        value = jnp.asarray(currents)
        if value.ndim != 1 or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("currents must be a real inexact rank-one array.")
        self.currents = value


class MagneticCurrentCommand(StrictModule):
    """Requested fixed-shape coil-current target in amperes."""

    target_currents: Array

    def __init__(self, target_currents: ArrayLike, /):
        value = jnp.asarray(target_currents)
        if value.ndim != 1 or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("target_currents must be a real inexact rank-one array.")
        self.target_currents = value


class MagneticActuationEvaluation(StrictModule):
    """Prescribed affine-field dipole mechanics and interaction-energy power."""

    candidate_state: MagneticCurrentState
    requested_current_rate: Array
    applied_current_rate: Array
    segment_centers: Array
    segment_moments_world: Array
    magnetic_field: Array
    segment_forces_world: Array
    segment_torques_world: Array
    native_forces: Array
    native_moments: Array
    reduced_effort: Array
    stored_energy: Array
    stored_power: Array
    source_power: Array
    mechanical_power: Array
    native_mechanical_power: Array
    reduced_mechanical_power: Array
    power_residual: Array
    native_virtual_work_residual: Array
    reduced_virtual_work_residual: Array
    source_margin: Array
    candidate_margin: Array
    position_margin: Array
    saturated: Array
    finite: Array
    within_domain: Array
    power_balanced: Array
    valid: Array
    electrical_power_available: bool = eqx.field(static=True)
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class AffineMagneticActuationPlan(StrictModule, NonTrainableState):
    """Calibrated source-free affine fields acting on rigid body dipoles."""

    uniform_field_per_current: Array
    field_gradient_per_current: Array
    field_origin: Array
    position_lower: Array
    position_upper: Array
    current_lower: Array
    current_upper: Array
    maximum_rise_rate: Array
    maximum_fall_rate: Array
    segment_dipoles_material: Array
    channel_count: int = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    excluded_capabilities: tuple[MagneticExclusion, ...] = eqx.field(static=True)

    def __init__(
        self,
        uniform_field_per_current: ArrayLike,
        field_gradient_per_current: ArrayLike,
        field_origin: ArrayLike,
        segment_dipoles_material: ArrayLike,
        /,
        *,
        position_bounds: tuple[ArrayLike, ArrayLike],
        current_bounds: tuple[ArrayLike | float, ArrayLike | float],
        maximum_rise_rate: ArrayLike | float,
        maximum_fall_rate: ArrayLike | float,
        source_manifest_id: str,
        calibration_id: str,
        power_tolerance: float = 1.0e-6,
        nonlinear_ferromagnetics: bool = False,
        hysteresis: bool = False,
        mutual_fields: bool = False,
        maxwell_solves: bool = False,
        coupled_rl_circuits: bool = False,
    ):
        _reject_requested_exclusions(
            "AffineMagneticActuationPlan",
            nonlinear_ferromagnetics=nonlinear_ferromagnetics,
            hysteresis=hysteresis,
            mutual_fields=mutual_fields,
            maxwell_solves=maxwell_solves,
            coupled_rl_circuits=coupled_rl_circuits,
        )
        uniform = _real_array("uniform_field_per_current", uniform_field_per_current, 2)
        gradient = _real_array(
            "field_gradient_per_current", field_gradient_per_current, 3
        )
        origin = _real_array("field_origin", field_origin, 1)
        dipoles = _real_array("segment_dipoles_material", segment_dipoles_material, 2)
        channels = int(uniform.shape[0])
        if (
            channels < 1
            or uniform.shape != (channels, 3)
            or gradient.shape != (channels, 3, 3)
            or origin.shape != (3,)
            or dipoles.shape[1:] != (3,)
        ):
            raise ValueError("Affine magnetic calibration shapes are invalid.")
        dtype = uniform.dtype
        if any(value.dtype != dtype for value in (gradient, origin, dipoles)):
            raise TypeError("Magnetic calibration arrays must share one dtype.")
        tolerance = 500.0 * np.finfo(dtype).eps
        scale = max(1.0, float(np.max(np.abs(gradient))))
        if not np.allclose(
            gradient,
            np.swapaxes(gradient, -1, -2),
            rtol=tolerance,
            atol=tolerance * scale,
        ) or np.any(np.abs(np.trace(gradient, axis1=-2, axis2=-1)) > tolerance * scale):
            raise ValueError(
                "Every affine field gradient must be symmetric and trace-free."
            )
        position_lower = _real_array("position lower bound", position_bounds[0], 1)
        position_upper = _real_array("position upper bound", position_bounds[1], 1)
        if (
            position_lower.shape != (3,)
            or position_upper.shape != (3,)
            or position_lower.dtype != dtype
            or position_upper.dtype != dtype
            or np.any(position_lower > position_upper)
        ):
            raise ValueError(
                "Position bounds must be ordered spatial vectors of calibration dtype."
            )
        current_lower = _vector_parameter(
            "current lower bound", current_bounds[0], channels, dtype
        )
        current_upper = _vector_parameter(
            "current upper bound", current_bounds[1], channels, dtype
        )
        rise = _vector_parameter("maximum_rise_rate", maximum_rise_rate, channels, dtype)
        fall = _vector_parameter("maximum_fall_rate", maximum_fall_rate, channels, dtype)
        if (
            np.any(current_lower > current_upper)
            or np.any(rise <= 0.0)
            or np.any(fall <= 0.0)
        ):
            raise ValueError("Current bounds and slew limits are invalid.")
        manifest = _identifier(source_manifest_id, "source_manifest_id")
        calibration = _identifier(calibration_id, "calibration_id")
        power_tolerance_ = _positive_finite(power_tolerance, "power_tolerance")
        self.uniform_field_per_current = jnp.asarray(uniform)
        self.field_gradient_per_current = jnp.asarray(gradient)
        self.field_origin = jnp.asarray(origin)
        self.position_lower = jnp.asarray(position_lower)
        self.position_upper = jnp.asarray(position_upper)
        self.current_lower = jnp.asarray(current_lower)
        self.current_upper = jnp.asarray(current_upper)
        self.maximum_rise_rate = jnp.asarray(rise)
        self.maximum_fall_rate = jnp.asarray(fall)
        self.segment_dipoles_material = jnp.asarray(dipoles)
        self.channel_count = channels
        self.source_manifest_id = manifest
        self.calibration_id = calibration
        self.power_tolerance = power_tolerance_
        self.excluded_capabilities = _MAGNETIC_EXCLUSIONS
        self.plan_id = canonical_fingerprint(
            {
                "kind": "affine-prescribed-field-magnetized-rod-plan",
                "field": array_tree_fingerprint(
                    {"uniform": uniform, "gradient": gradient, "origin": origin}
                ),
                "position_bounds_m": array_tree_fingerprint(
                    {"lower": position_lower, "upper": position_upper}
                ),
                "current_bounds_a": array_tree_fingerprint(
                    {"lower": current_lower, "upper": current_upper}
                ),
                "slew_a_per_s": array_tree_fingerprint({"rise": rise, "fall": fall}),
                "dipoles_a_m2": array_tree_fingerprint(dipoles),
                "source_manifest": manifest,
                "calibration": calibration,
                "units": "A,T,T/m,A*m^2,N,N*m,W",
                "exclusions": self.excluded_capabilities,
            }
        )

    def prepare(self, rod: PreparedReducedRod, /) -> "PreparedAffineMagneticActuation":
        return PreparedAffineMagneticActuation(self, rod)


class PreparedAffineMagneticActuation(StrictModule, NonTrainableState):
    plan: AffineMagneticActuationPlan
    reduction: PreparedReducedRod
    actuation_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(self, plan: AffineMagneticActuationPlan, rod: PreparedReducedRod, /):
        if not isinstance(rod, PreparedReducedRod):
            raise TypeError("Affine magnetic actuation requires a PreparedReducedRod.")
        if rod.rod.plan.dimension != 3:
            raise ValueError("Affine magnetic actuation supports spatial rods only.")
        if plan.segment_dipoles_material.shape != (rod.rod.plan.segment_count, 3):
            raise ValueError("There must be one material dipole per rod segment.")
        if plan.segment_dipoles_material.dtype != rod.coefficient_space.dtype:
            raise TypeError(
                "Magnetic calibration dtype must match the reduced rod dtype."
            )
        provenance = canonical_fingerprint(
            {
                "kind": "affine-magnetic-rod-actuation-provenance",
                "plan": plan.plan_id,
                "rod": rod.prepared_id,
                "native_effort_space": rod.native_effort_space.space_id,
                "reduced_effort_space": rod.reduced_effort_space.space_id,
                "source_manifest": plan.source_manifest_id,
                "calibration": plan.calibration_id,
                "exclusions": plan.excluded_capabilities,
            }
        )
        self.plan = plan
        self.reduction = rod
        self.provenance_id = provenance
        self.actuation_id = canonical_fingerprint(
            {"kind": "prepared-affine-magnetic-rod-actuation", "provenance": provenance}
        )

    def initialize_state(self, currents: ArrayLike | None = None) -> MagneticCurrentState:
        value = jnp.zeros_like(self.plan.current_lower) if currents is None else currents
        state = MagneticCurrentState(value)
        self._validate_currents(state.currents, "currents")
        return state

    def _validate_currents(self, currents: Array, owner: str) -> None:
        if currents.shape != (self.plan.channel_count,):
            raise ValueError(f"{owner} must have shape ({self.plan.channel_count},).")
        if currents.dtype != self.plan.uniform_field_per_current.dtype:
            raise TypeError(f"{owner} must match the field calibration dtype.")

    def evaluate(
        self,
        rod_state: ReducedRodState,
        state: MagneticCurrentState,
        command: MagneticCurrentCommand,
        time_step: ArrayLike,
        /,
    ) -> MagneticActuationEvaluation:
        if not isinstance(state, MagneticCurrentState):
            raise TypeError("state must be a MagneticCurrentState.")
        if not isinstance(command, MagneticCurrentCommand):
            raise TypeError("command must be a MagneticCurrentCommand.")
        self._validate_currents(state.currents, "currents")
        self._validate_currents(command.target_currents, "target_currents")
        self.reduction.validate_state(rod_state)
        dtype = self.plan.uniform_field_per_current.dtype
        step = jnp.asarray(time_step, dtype=dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        transition = _transition(
            state.currents,
            command.target_currents,
            self.plan.current_lower,
            self.plan.current_upper,
            self.plan.maximum_rise_rate,
            self.plan.maximum_fall_rate,
            step,
        )
        currents = transition[0]
        current_rate = transition[2]
        candidate_state = MagneticCurrentState(currents)
        native = self.reduction.lift(rod_state)
        positions, orientations = self.reduction.rod.configuration_from_state(native)
        frames = _rotation_matrices(orientations, 3)
        topology = self.reduction.rod.plan.segment_node_ids
        centers = 0.5 * (positions[topology[:, 0]] + positions[topology[:, 1]])
        dipoles_world = ein.contract(
            "sij,sj->si", frames, self.plan.segment_dipoles_material
        )
        relative = centers - self.plan.field_origin
        field_basis = self.plan.uniform_field_per_current[:, None, :] + ein.contract(
            "cij,sj->csi", self.plan.field_gradient_per_current, relative
        )
        field = ein.contract("c,csi->si", currents, field_basis)
        gradient = ein.contract(
            "c,cij->ij", currents, self.plan.field_gradient_per_current
        )
        segment_forces = ein.contract("ij,sj->si", gradient.T, dipoles_world)
        torques_world = jnp.cross(dipoles_world, field)
        forces = jnp.zeros_like(positions)
        forces = forces.at[topology[:, 0]].add(0.5 * segment_forces)
        forces = forces.at[topology[:, 1]].add(0.5 * segment_forces)
        moments = ein.contract("sji,sj->si", frames, torques_world)
        native_effort = self.reduction.native_effort_space.validate((forces, moments))
        reduced_effort = self.reduction.lift_effort_pullback_operator(
            rod_state.coefficients
        ).mv(native_effort)
        native_velocity = self.reduction.rod.velocity_from_state(native)
        native_power = self.reduction.native_effort_space.pair(
            native_effort, native_velocity
        ).real
        reduced_power = self.reduction.reduced_effort_space.pair(
            reduced_effort, rod_state.coefficient_velocities
        ).real
        mechanical_power = native_power
        stored_energy = -jnp.sum(dipoles_world * field)
        current_energy_gradient = -jnp.sum(
            dipoles_world[None, :, :] * field_basis, axis=(1, 2)
        )
        source_power = jnp.sum(current_energy_gradient * current_rate)
        stored_power = -mechanical_power + source_power
        power_residual = stored_power + mechanical_power - source_power
        native_residual = native_power - mechanical_power
        reduced_residual = reduced_power - mechanical_power
        position_margin = jnp.min(
            jnp.minimum(
                centers - self.plan.position_lower,
                self.plan.position_upper - centers,
            )
        )
        finite = _all_finite(
            state.currents,
            command.target_currents,
            currents,
            current_rate,
            centers,
            dipoles_world,
            field,
            segment_forces,
            torques_world,
            forces,
            moments,
            reduced_effort,
            stored_energy,
            stored_power,
            source_power,
            mechanical_power,
            native_power,
            reduced_power,
            power_residual,
        )
        within_domain = transition[6] & (position_margin >= 0.0)
        balanced = (
            _power_balanced(
                power_residual,
                stored_power,
                source_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
            & _power_balanced(
                native_residual,
                native_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
            & _power_balanced(
                reduced_residual,
                reduced_power,
                mechanical_power,
                tolerance=self.plan.power_tolerance,
            )
        )
        return MagneticActuationEvaluation(
            candidate_state,
            transition[1],
            current_rate,
            centers,
            dipoles_world,
            field,
            segment_forces,
            torques_world,
            forces,
            moments,
            reduced_effort,
            stored_energy,
            stored_power,
            source_power,
            mechanical_power,
            native_power,
            reduced_power,
            power_residual,
            native_residual,
            reduced_residual,
            transition[3],
            transition[4],
            position_margin,
            transition[5],
            finite,
            within_domain,
            balanced,
            finite & within_domain & balanced,
            False,
            self.actuation_id,
            self.provenance_id,
        )


__all__ = [
    "AffineMagneticActuationPlan",
    "IntrinsicStrainActuationPlan",
    "IntrinsicStrainActuationState",
    "IntrinsicStrainCandidate",
    "IntrinsicStrainCommand",
    "IntrinsicStrainEvaluation",
    "MagneticActuationEvaluation",
    "MagneticCurrentCommand",
    "MagneticCurrentState",
    "PreparedAffineMagneticActuation",
    "PreparedIntrinsicStrainActuation",
    "PreparedReducedTubeChamber",
    "PreparedRegulatedReducedTubePressureActuation",
    "PreparedSealedReducedTubePressureActuation",
    "PreparedVariableStiffnessActuation",
    "ReducedTubeChamberPlan",
    "RegulatedReducedTubePressurePlan",
    "RegulatedTubePressureCommand",
    "RegulatedTubePressureEvaluation",
    "RegulatedTubePressureState",
    "RodTubeStation",
    "SealedReducedTubePressurePlan",
    "SealedTubePressureEvaluation",
    "SealedTubePressureState",
    "VariableStiffnessActuationPlan",
    "VariableStiffnessCandidate",
    "VariableStiffnessCommand",
    "VariableStiffnessEvaluation",
    "VariableStiffnessState",
    "combine_rod_constitutive_controls",
]
