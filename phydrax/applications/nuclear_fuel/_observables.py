"""Identity-bound FE sensor forward operators and inventory-defined FGR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._dtype_names import inexact_result_type
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier
from ...discretization import CellBlock
from ...discretization.fem import (
    prepare_finite_element_point_interpolation,
    PreparedFiniteElementPointInterpolation,
)
from ...observation import CoordinateLayout, TheoryVector
from ...typing import (
    as_array,
    as_host_array,
    Bool,
    Dim,
    Float,
    HostFloat,
    parse,
    Scope,
    Size,
)
from ._bundles import FuelModelBundle
from ._geometry import FuelBodyGeometry, point_field_values
from ._identity import FUEL_GAS_SPECIES
from ._inventory import FuelGasLedger
from ._problem import PreparedFuelRod
from ._result import FuelRodResult
from ._sensitivity import FuelFixedGrid
from ._specification import FuelRodSpecification
from ._state import FuelRodState


type FuelObservableKind = Literal[
    "thermocouple-temperature",
    "centerline-temperature",
    "in-pile-pressure",
    "cold-pressure",
    "diameter-metal",
    "diameter-with-oxide",
    "elongation",
    "oxide-thickness",
    "hydrogen-ppm",
    "incremental-fgr",
    "cumulative-fgr",
]
type FuelSensorBody = Literal["fuel", "cladding"]

# Default reference temperature [K] (20 degC) of a cold measurement and the
# tolerance [K] within which the computed rod temperature must match it.
COLD_MEASUREMENT_TEMPERATURE_K = 293.15
COLD_MEASUREMENT_TOLERANCE_K = 1.0
# Absolute tolerance [m] on the physical position reconstructed from a sensor's
# FE reference coordinates; it rejects a wrong inverse map, not mesh error.
SENSOR_POINT_MAPPING_TOLERANCE_M = 1.0e-12

_COLD_OBSERVABLE_KINDS: tuple[FuelObservableKind, ...] = (
    "cold-pressure",
    "diameter-metal",
    "diameter-with-oxide",
)
_FGR_OBSERVABLE_KINDS: tuple[FuelObservableKind, ...] = (
    "incremental-fgr",
    "cumulative-fgr",
)


class FuelObservationDim(Dim):
    """Retained sensor/time sample coordinates."""


class FuelSensorDim(Dim):
    """Named detector coordinates."""


class FuelAxialEdgeDim(Dim):
    """Irradiation-zone edge coordinates of the prepared rod."""


@dataclass(frozen=True, slots=True)
class FuelSensor:
    """SI physical position, detector reference, and drift-valid time window.

    Coordinates refer to the reference rod; a thermocouple follows its material
    point. Elongation is the endpoint difference relative to the rig reference.
    Cold measurements require the *computed* whole-rod temperature to match
    ``cold_temperature``; they never overwrite the temperature fields.
    """

    sensor_id: str
    kind: FuelObservableKind
    radius: float = 0.0
    axial_position: float = 0.0
    response_time: float = 0.0
    drift_valid_window: tuple[float, float] = (0.0, np.inf)
    detector_reference: float = 0.0
    cold_temperature: float = COLD_MEASUREMENT_TEMPERATURE_K
    cold_tolerance: float = COLD_MEASUREMENT_TOLERANCE_K
    body: FuelSensorBody = "fuel"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sensor_id", canonical_identifier(self.sensor_id, "sensor_id")
        )
        object.__setattr__(
            self, "kind", parse(self.kind, FuelObservableKind, "observable kind")
        )
        object.__setattr__(self, "body", parse(self.body, FuelSensorBody, "sensor body"))
        if self.kind == "centerline-temperature" and self.body != "fuel":
            raise ValueError("Centerline temperature belongs to the fuel body.")
        if self.kind != "thermocouple-temperature" and self.response_time != 0.0:
            raise ValueError("Only a thermocouple sensor may have a response time.")
        values = (
            self.radius,
            self.axial_position,
            self.response_time,
            self.detector_reference,
            self.cold_temperature,
            self.cold_tolerance,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Sensor positions, response, and references must be finite.")
        if (
            self.radius < 0
            or self.axial_position < 0
            or self.response_time < 0
            or self.cold_temperature <= 0
            or self.cold_tolerance <= 0
        ):
            raise ValueError(
                "Sensor radius, axial position, and response time must be nonnegative and cold temperature and tolerance positive."
            )
        window_start, window_end = self.drift_valid_window
        if np.isnan(window_start) or np.isnan(window_end) or window_start > window_end:
            raise ValueError("Sensor drift-valid window must be ordered.")


class FuelGasAncestry(StrictModule):
    """Ancestor transfers plus the segment-start ledger, not a gas stock.

    Refabrication may remove free gas; historical released transfers remain in
    the ancestor ledger. Both ledgers use the ``FUEL_GAS_SPECIES`` axis.
    """

    ancestor: FuelGasLedger
    segment_initial: FuelGasLedger
    parent_rod_id: str = eqx.field(static=True)
    segment_rod_id: str = eqx.field(static=True)
    refabrication_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        ancestor: FuelGasLedger,
        segment_initial: FuelGasLedger,
        /,
        *,
        parent_rod_id: str,
        segment_rod_id: str,
        refabrication_id: str | None = None,
    ) -> None:
        if not isinstance(ancestor, FuelGasLedger) or not isinstance(
            segment_initial, FuelGasLedger
        ):
            raise TypeError("FGR ancestry requires FuelGasLedger inventories.")
        for ledger in (ancestor, segment_initial):
            if ledger.produced_moles.shape != (
                len(FUEL_GAS_SPECIES),
            ) or ledger.released_moles.shape != (len(FUEL_GAS_SPECIES),):
                raise ValueError("FGR ledgers must use the fuel gas species axis.")
        self.ancestor = ancestor
        self.segment_initial = segment_initial
        self.parent_rod_id = canonical_identifier(parent_rod_id, "parent_rod_id")
        self.segment_rod_id = canonical_identifier(segment_rod_id, "segment_rod_id")
        self.refabrication_id = (
            None
            if refabrication_id is None
            else canonical_identifier(refabrication_id, "refabrication_id")
        )

    def fraction(
        self, current: FuelGasLedger, /, *, cumulative: bool
    ) -> tuple[Array, Array]:
        """Xe+Kr released transfer / production; zero production is invalid."""
        indices = jnp.asarray(
            tuple(FUEL_GAS_SPECIES.index(name) for name in ("Kr", "Xe")), dtype=jnp.int32
        )
        produced_moles = current.produced_moles - self.segment_initial.produced_moles
        released_moles = current.released_moles - self.segment_initial.released_moles
        if cumulative:
            produced_moles = produced_moles + self.ancestor.produced_moles
            released_moles = released_moles + self.ancestor.released_moles
        denominator = jnp.sum(produced_moles[indices])
        numerator = jnp.sum(released_moles[indices])
        valid = (denominator > 0) & (numerator >= 0) & (numerator <= denominator)
        return jnp.where(
            valid, numerator / jnp.where(denominator > 0, denominator, 1.0), jnp.nan
        ), valid


def thermocouple_response(
    temperature: ArrayLike, times: ArrayLike, response_time: float, /
) -> Array:
    """Exact first-order sensor response to a piecewise-linear sampled signal.

    The sensor starts equilibrated with the first sample. Drift is applied
    separately so its validity window is not hidden in the lag response.
    """
    signal = jnp.asarray(temperature, dtype=inexact_result_type(temperature))
    time = jnp.asarray(times, dtype=signal.dtype)
    if signal.ndim != 1 or time.shape != signal.shape or signal.size == 0:
        raise ValueError("Thermocouple signal and time axes must agree and be nonempty.")
    if response_time < 0 or not np.isfinite(response_time):
        raise ValueError("Thermocouple response time must be finite and nonnegative.")
    time = eqx.error_if(
        time, jnp.any(jnp.diff(time) <= 0), "Sensor times must strictly increase."
    )
    if response_time == 0:
        return signal

    def advance(
        previous: Array, interval: tuple[Array, Array, Array]
    ) -> tuple[Array, Array]:
        dt, left, right = interval
        gain = -jnp.expm1(-dt / response_time)
        slope = (right - left) / dt
        value = (
            (1.0 - gain) * previous + gain * left + slope * (dt - response_time * gain)
        )
        return value, value

    _, response = jax.lax.scan(
        advance, signal[0], (jnp.diff(time), signal[:-1], signal[1:])
    )
    return jnp.concatenate((signal[:1], response))


class _FuelPointQuery(StrictModule):
    """Prepared FE point route; ``layer >= 0`` selects one layer of a layered body."""

    route: PreparedFiniteElementPointInterpolation
    layer: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    radial_node_count: int = eqx.field(static=True)

    def evaluate(self, coefficients: Array, /) -> Array:
        if self.layer >= 0:
            values = coefficients.reshape(
                (self.layer_count, self.radial_node_count) + coefficients.shape[1:]
            )[self.layer]
        else:
            values = coefficients
        return point_field_values(self.route, values)[0]


def _require_query(query: _FuelPointQuery | None, message: str, /) -> _FuelPointQuery:
    if query is None:
        raise ValueError(message)
    return query


def _physical_point_query(
    body: FuelBodyGeometry,
    radius: float,
    axial_position: float,
    edges: HostFloat[FuelAxialEdgeDim],
    /,
) -> _FuelPointQuery:
    """Prepare a host FE point route to one physical sensor position.

    The owning cell is found from physical coordinates and mesh connectivity;
    the reconstructed physical position must reproduce the requested point. A
    trace shared by cells is continuous, so the smallest global cell ID wins.
    """
    original_route = body.value_route
    if not isinstance(original_route, PreparedFiniteElementPointInterpolation):
        raise TypeError(
            "Observation preparation requires the body's FE point-interpolation route."
        )
    space = original_route.discretization
    block = space.mesh.blocks[0]
    if not isinstance(block, CellBlock):
        raise TypeError(
            "Fuel sensor queries require an interval or quadrilateral cell block."
        )
    corners = np.asarray(space.mesh.coordinates)[np.asarray(block.vertices)]
    point = np.asarray(
        [radius] if body.layered else [radius, axial_position], dtype=np.float64
    )
    low, high = corners.min(axis=1), corners.max(axis=1)
    containing = np.flatnonzero(np.all((point >= low) & (point <= high), axis=1))
    if containing.size == 0 or not edges[0] <= axial_position <= edges[-1]:
        raise ValueError("Sensor physical position lies outside its FE body.")
    selected = containing[np.argmin(np.asarray(block.global_ids)[containing])]
    extent = high[selected] - low[selected]
    if np.any(extent <= 0):
        raise ValueError("Sensor route cell has a degenerate physical extent.")
    local = ((point - low[selected]) / extent)[None, :]
    route = prepare_finite_element_point_interpolation(
        space,
        original_route.field_name,
        block.name,
        np.asarray([selected], dtype=np.int32),
        local,
    )
    if not np.allclose(
        np.asarray(route.reference_positions)[0],
        point,
        rtol=0.0,
        atol=SENSOR_POINT_MAPPING_TOLERANCE_M,
    ):
        raise ValueError(
            "Sensor inverse coordinates do not map to its physical position."
        )
    layer = (
        min(
            np.searchsorted(edges, axial_position, side="right") - 1, body.layer_count - 1
        )
        if body.layered
        else -1
    )
    return _FuelPointQuery(
        route=route,
        layer=int(layer),
        layer_count=body.layer_count,
        radial_node_count=body.radial_node_count,
    )


def _sensor_zone(edges: HostFloat[FuelAxialEdgeDim], axial_position: float, /) -> int:
    """Index of the irradiation zone containing ``axial_position``."""
    return min(
        int(np.searchsorted(edges, axial_position, side="right") - 1),
        edges.size - 2,
    )


def _sensor_query(
    prepared: PreparedFuelRod,
    specification: FuelRodSpecification,
    sensor: FuelSensor,
    edges: HostFloat[FuelAxialEdgeDim],
    ancestry: FuelGasAncestry | None,
    /,
) -> _FuelPointQuery | None:
    """FE point route of a field sensor; ``None`` for zone/inventory sensors."""
    match sensor.kind:
        case "thermocouple-temperature" | "centerline-temperature":
            radius = 0.0 if sensor.kind == "centerline-temperature" else sensor.radius
            body = (
                prepared.geometry.fuel
                if sensor.body == "fuel"
                else prepared.geometry.cladding
            )
            return _physical_point_query(body, radius, sensor.axial_position, edges)
        case "diameter-metal" | "diameter-with-oxide" | "elongation":
            return _physical_point_query(
                prepared.geometry.cladding,
                specification.cladding.outer_radius,
                sensor.axial_position,
                edges,
            )
        case "in-pile-pressure" | "cold-pressure" | "oxide-thickness" | "hydrogen-ppm":
            return None
        case "incremental-fgr" | "cumulative-fgr":
            if ancestry is None:
                raise ValueError(
                    "FGR observations require segment-start and ancestor inventories."
                )
            return None
        case _:
            assert_never(sensor.kind)


def _sensor_unit(kind: FuelObservableKind, /) -> str:
    match kind:
        case "thermocouple-temperature" | "centerline-temperature":
            return "K"
        case "in-pile-pressure" | "cold-pressure":
            return "Pa"
        case "hydrogen-ppm":
            return "ppm-by-mass"
        case "incremental-fgr" | "cumulative-fgr":
            return "1"
        case "diameter-metal" | "diameter-with-oxide" | "elongation" | "oxide-thickness":
            return "m"
        case _:
            assert_never(kind)


def _resolve_samples(
    sensors: tuple[FuelSensor, ...],
    samples: tuple[tuple[str, float], ...],
    grid: FuelFixedGrid,
    /,
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    """(sensor index, grid time index) of every sample, with its unit."""
    sensor_ids = tuple(sensor.sensor_id for sensor in sensors)
    selected: list[tuple[int, int]] = []
    units: list[str] = []
    for sensor_id, time in samples:
        if sensor_id not in sensor_ids or time not in grid.times:
            raise ValueError(
                "Observation samples require a named sensor and an exact frozen-grid time."
            )
        sensor_index = sensor_ids.index(sensor_id)
        selected.append((sensor_index, grid.times.index(time)))
        units.append(_sensor_unit(sensors[sensor_index].kind))
    return tuple(selected), tuple(units)


def _observation_plan_id(
    prepared: PreparedFuelRod,
    grid: FuelFixedGrid,
    sensors: tuple[FuelSensor, ...],
    layout: CoordinateLayout,
    ancestry: FuelGasAncestry | None,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "fuel-observation-plan",
            "prepared": prepared.prepared_id,
            "grid": grid.grid_id,
            "sensors": [
                (
                    sensor.sensor_id,
                    sensor.kind,
                    sensor.body,
                    sensor.radius,
                    sensor.axial_position,
                    sensor.response_time,
                    tuple(float(value).hex() for value in sensor.drift_valid_window),
                    sensor.detector_reference,
                    sensor.cold_temperature,
                    sensor.cold_tolerance,
                )
                for sensor in sensors
            ],
            "layout": layout.layout_id,
            "ancestry": None
            if ancestry is None
            else {
                "rods": (
                    ancestry.parent_rod_id,
                    ancestry.segment_rod_id,
                    ancestry.refabrication_id,
                ),
                "inventories": array_tree_fingerprint(
                    (ancestry.ancestor, ancestry.segment_initial)
                ),
            },
        }
    )


class FuelObservationResult(StrictModule):
    """Sensor samples with per-sample validity and units, in observation order."""

    __strict_contract__ = True
    product: TheoryVector
    valid: Bool[FuelObservationDim]
    successful: Array
    units: tuple[str, ...] = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class FuelSensorParameters(StrictModule):
    """Named detector gain, offset, and drift on an exact sensor layout.

    The same numerical tree can hold search coordinates in a ParameterSpace;
    positive-gain validity is judged when the nuisances are applied.
    """

    offset: Array
    drift: Array
    gain: Array
    layout: CoordinateLayout

    def __init__(
        self,
        layout: CoordinateLayout,
        /,
        *,
        offset: ArrayLike,
        drift: ArrayLike,
        gain: ArrayLike,
    ) -> None:
        scope = Scope()
        parse(layout.size, Size[FuelSensorDim], "sensor_count", scope=scope)
        self.offset = as_array(offset, Float[FuelSensorDim], "sensor_offset", scope=scope)
        self.drift = as_array(drift, Float[FuelSensorDim], "sensor_drift", scope=scope)
        self.gain = as_array(gain, Float[FuelSensorDim], "sensor_gain", scope=scope)
        self.layout = layout

    def validate(self, layout: CoordinateLayout, /) -> None:
        if self.layout.layout_id != layout.layout_id:
            raise ValueError(
                "Sensor nuisance layout must equal the observation sensor layout."
            )
        scope = Scope()
        parse(layout.size, Size[FuelSensorDim], "sensor_count", scope=scope)
        parse(self.offset, Float[FuelSensorDim], "sensor_offset", scope=scope)
        parse(self.drift, Float[FuelSensorDim], "sensor_drift", scope=scope)
        parse(self.gain, Float[FuelSensorDim], "sensor_gain", scope=scope)


class FuelObservationPlan(StrictModule):
    """Sensor/time sample coordinates bound to one prepared rod."""

    prepared: PreparedFuelRod
    specification: FuelRodSpecification
    grid: FuelFixedGrid = eqx.field(static=True)
    sensors: tuple[FuelSensor, ...] = eqx.field(static=True)
    queries: tuple[_FuelPointQuery | None, ...]
    reference_queries: tuple[_FuelPointQuery | None, ...]
    ancestry: FuelGasAncestry | None
    layout: CoordinateLayout
    sensor_layout: CoordinateLayout
    samples: tuple[tuple[int, int], ...] = eqx.field(static=True)
    zones: tuple[int, ...] = eqx.field(static=True)
    units: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedFuelRod,
        grid: FuelFixedGrid,
        sensors: tuple[FuelSensor, ...],
        samples: tuple[tuple[str, float], ...],
        /,
        *,
        ancestry: FuelGasAncestry | None = None,
    ) -> None:
        if (
            not isinstance(prepared, PreparedFuelRod)
            or grid.prepared_id != prepared.prepared_id
        ):
            raise ValueError("Observation grid must belong to the prepared rod.")
        specification = prepared.admission.specification
        if not isinstance(specification, FuelRodSpecification):
            raise TypeError(
                "Observation preparation requires the rod's FuelRodSpecification."
            )
        sensor_ids = tuple(sensor.sensor_id for sensor in sensors)
        if not sensors or len(set(sensor_ids)) != len(sensor_ids) or not samples:
            raise ValueError("Observation sensors must be unique and samples nonempty.")
        edges = as_host_array(
            prepared.admission.history.zone_edges,
            HostFloat[FuelAxialEdgeDim],
            "zone_edges",
        )
        queries: list[_FuelPointQuery | None] = []
        zones: list[int] = []
        for sensor in sensors:
            if not edges[0] <= sensor.axial_position <= edges[-1]:
                raise ValueError("Sensor axial position must lie in the rod.")
            zones.append(_sensor_zone(edges, sensor.axial_position))
            queries.append(
                _sensor_query(prepared, specification, sensor, edges, ancestry)
            )
        selected_samples, units = _resolve_samples(sensors, samples, grid)
        self.prepared = prepared
        self.grid = grid
        self.sensors = sensors
        self.specification = specification
        self.queries = tuple(queries)
        self.zones = tuple(zones)
        # RZ elongation subtracts the axial displacement of the cladding outer
        # surface at the bottom zone edge from that at the detector.
        self.reference_queries = tuple(
            _physical_point_query(
                prepared.geometry.cladding,
                specification.cladding.outer_radius,
                float(edges[0]),
                edges,
            )
            if sensor.kind == "elongation" and not prepared.geometry.layered
            else None
            for sensor in sensors
        )
        self.ancestry = ancestry
        self.samples = selected_samples
        self.units = units
        self.layout = CoordinateLayout(
            tuple(
                f"{prepared.prepared_id}:{sensor_id}@{float(time).hex()}"
                for sensor_id, time in samples
            )
        )
        self.sensor_layout = CoordinateLayout(
            tuple(f"{prepared.prepared_id}:{sensor_id}" for sensor_id in sensor_ids)
        )
        self.plan_id = _observation_plan_id(
            prepared, grid, sensors, self.layout, ancestry
        )

    def _evaluate_sensor(
        self, sensor_index: int, state: FuelRodState, parameters: FuelModelBundle, /
    ) -> tuple[Array, Array]:
        """Instantaneous value and validity of one sensor at one state."""
        sensor = self.sensors[sensor_index]
        query = self.queries[sensor_index]
        zone = self.zones[sensor_index]
        fields = state.fields
        valid = jnp.asarray(True)
        match sensor.kind:
            case "thermocouple-temperature" | "centerline-temperature":
                value = _require_query(
                    query, "A temperature sensor requires its prepared FE query."
                ).evaluate(
                    fields.temperature_fuel
                    if sensor.body == "fuel"
                    else fields.temperature_cladding
                )
            case "in-pile-pressure" | "cold-pressure":
                value = fields.pressure
            case "diameter-metal" | "diameter-with-oxide":
                value = self._cladding_diameter(
                    sensor,
                    _require_query(
                        query, "A diameter sensor requires its prepared FE query."
                    ),
                    state,
                    zone,
                    parameters,
                )
            case "elongation":
                value = self._elongation(sensor_index, state) - sensor.detector_reference
                if self.prepared.admission.axial_interaction != "coulomb":
                    valid = valid & jnp.all(fields.contact_pressure <= 0.0)
            case "oxide-thickness":
                value = self._surface_oxide(state, zone)
            case "hydrogen-ppm":
                value = self._hydrogen_concentration(state, zone, parameters)
            case "incremental-fgr" | "cumulative-fgr":
                if self.ancestry is None:
                    raise ValueError("FGR observations require gas ancestry.")
                value, valid = self.ancestry.fraction(
                    state.gas, cumulative=sensor.kind == "cumulative-fgr"
                )
            case _:
                assert_never(sensor.kind)
        if sensor.kind in _COLD_OBSERVABLE_KINDS:
            valid = valid & self.prepared.is_cooled(
                state,
                target_temperature=sensor.cold_temperature,
                tolerance=sensor.cold_tolerance,
            )
        return value, valid & jnp.isfinite(value)

    def _cladding_diameter(
        self,
        sensor: FuelSensor,
        query: _FuelPointQuery,
        state: FuelRodState,
        zone: int,
        parameters: FuelModelBundle,
        /,
    ) -> Array:
        """Outer diameter [m]: metal surface, optionally plus the oxide layer."""
        oxide = self._surface_oxide(state, zone)
        metal_loss = parameters.corrosion.metal_thickness_consumed(oxide)
        radius = (
            self.prepared.admission.specification.cladding.outer_radius
            + query.evaluate(state.fields.displacement_cladding)[0]
            - metal_loss
        )
        return 2.0 * (radius + (oxide if sensor.kind == "diameter-with-oxide" else 0.0))

    def _elongation(self, sensor_index: int, state: FuelRodState, /) -> Array:
        """Cladding elongation [m] between the rod bottom and the detector."""
        sensor = self.sensors[sensor_index]
        fields = state.fields
        if self.prepared.geometry.layered:
            edges = jnp.asarray(self.prepared.admission.history.zone_edges)
            lengths = jnp.clip(sensor.axial_position - edges[:-1], 0.0, jnp.diff(edges))
            return jnp.sum(fields.axial_strain_cladding * lengths)
        query = _require_query(
            self.queries[sensor_index],
            "An elongation detector requires its prepared FE query.",
        )
        reference_query = _require_query(
            self.reference_queries[sensor_index],
            "RZ elongation requires both physical detector endpoints.",
        )
        return (
            query.evaluate(fields.displacement_cladding)[1]
            - reference_query.evaluate(fields.displacement_cladding)[1]
        )

    def _hydrogen_concentration(
        self, state: FuelRodState, zone: int, parameters: FuelModelBundle, /
    ) -> Array:
        """Zone-average hydrogen [ppm by mass] weighted by remaining metal area."""
        body = self.prepared.geometry.cladding
        cladding = self.prepared.admission.specification.cladding
        consumed = parameters.corrosion.metal_thickness_consumed(
            state.cladding_sites.oxide_thickness
        )
        remaining_metal_fraction = (
            (cladding.outer_radius - consumed) ** 2 - cladding.inner_radius**2
        ) / (cladding.outer_radius**2 - cladding.inner_radius**2)
        weight = jnp.where(
            body.site_zone == zone, body.site_weight * remaining_metal_fraction, 0.0
        )
        return jnp.sum(weight * state.cladding_sites.hydrogen_ppm) / jnp.sum(weight)

    def _surface_oxide(self, state: FuelRodState, zone: int, /) -> Array:
        """Zone-average outer-surface oxide thickness [m]."""
        trace = self.prepared.geometry.cladding_outer
        weights = jnp.where(trace.zone == zone, trace.weight, 0.0)
        return jnp.sum(weights * self.prepared.surface_oxide(state)) / jnp.sum(weights)

    def _ancestry_matches(self, initial_state: FuelRodState, /) -> Array:
        """Whether the first state's gas ledger equals the segment-start ledger."""
        if self.ancestry is None:
            return jnp.asarray(True)
        initial = self.ancestry.segment_initial
        return jnp.all(
            initial_state.gas.produced_moles == initial.produced_moles
        ) & jnp.all(initial_state.gas.released_moles == initial.released_moles)

    def _sensor_samples(
        self,
        sensor_index: int,
        time_indices: tuple[int, ...],
        states: tuple[FuelRodState, ...],
        grid_times: Array,
        parameters: FuelModelBundle,
        /,
    ) -> tuple[tuple[int, Array, Array], ...]:
        """(time index, value, validity) of one sensor at its sampled times."""
        sensor = self.sensors[sensor_index]
        if sensor.kind == "thermocouple-temperature" and sensor.response_time > 0.0:
            # The lagged response at a sample depends on every earlier state.
            last = max(time_indices)
            evaluations = tuple(
                self._evaluate_sensor(sensor_index, state, parameters)
                for state in states[: last + 1]
            )
            response = thermocouple_response(
                jnp.stack(tuple(value for value, _ in evaluations)),
                grid_times[: last + 1],
                sensor.response_time,
            )
            return tuple(
                (time_index, response[time_index], evaluations[time_index][1])
                for time_index in time_indices
            )
        return tuple(
            (
                time_index,
                *self._evaluate_sensor(sensor_index, states[time_index], parameters),
            )
            for time_index in time_indices
        )

    def physical_signals(
        self,
        states: tuple[FuelRodState, ...],
        /,
        *,
        parameters: FuelModelBundle | None = None,
    ) -> tuple[Array, Array, Array]:
        """Nuisance-free sampled signals, their validity, and the grid times.

        Rows follow ``samples``. Validity includes the sensor drift window and,
        for FGR, agreement of the first state with the segment-start ledger.
        A lagged thermocouple sample carries the validity of its sampled state;
        a non-finite earlier value propagates into the lagged signal, which
        ``apply_sensor_nuisance`` rejects as non-finite.
        """
        checked = self.grid.validate_states(states)
        bundle = self.prepared.admission.bundle if parameters is None else parameters
        ancestry_valid = self._ancestry_matches(checked[0])
        signals: dict[tuple[int, int], Array] = {}
        validity: dict[tuple[int, int], Array] = {}
        grid_times = jnp.asarray(self.grid.times, dtype=checked[0].time.dtype)
        for sensor_index, sensor in enumerate(self.sensors):
            time_indices = tuple(
                time_index
                for sampled_sensor, time_index in self.samples
                if sampled_sensor == sensor_index
            )
            if not time_indices:
                continue
            for time_index, value, valid in self._sensor_samples(
                sensor_index, time_indices, checked, grid_times, bundle
            ):
                key = sensor_index, time_index
                signals[key] = value
                valid = (
                    valid
                    & (grid_times[time_index] >= sensor.drift_valid_window[0])
                    & (grid_times[time_index] <= sensor.drift_valid_window[1])
                )
                if sensor.kind in _FGR_OBSERVABLE_KINDS:
                    valid = valid & ancestry_valid
                validity[key] = valid
        return (
            jnp.stack(tuple(signals[key] for key in self.samples)),
            jnp.stack(tuple(validity[key] for key in self.samples)),
            grid_times,
        )

    def apply_sensor_nuisance(
        self,
        signals: Array,
        validity: Array,
        times: Array,
        /,
        *,
        sensor_nuisance: FuelSensorParameters | None = None,
    ) -> FuelObservationResult:
        """Measured value ``gain * signal + offset + drift * (t - t0)`` per sample."""
        if sensor_nuisance is not None:
            if not isinstance(sensor_nuisance, FuelSensorParameters):
                raise TypeError(
                    "Sensor nuisances must be FuelSensorParameters, not positional array columns."
                )
            sensor_nuisance.validate(self.sensor_layout)
        measured: list[Array] = []
        valid_rows: list[Array] = []
        for row, (sensor_index, time_index) in enumerate(self.samples):
            offset = (
                0.0 if sensor_nuisance is None else sensor_nuisance.offset[sensor_index]
            )
            drift = (
                0.0 if sensor_nuisance is None else sensor_nuisance.drift[sensor_index]
            )
            gain = 1.0 if sensor_nuisance is None else sensor_nuisance.gain[sensor_index]
            measured.append(
                gain * signals[row] + offset + drift * (times[time_index] - times[0])
            )
            valid_rows.append(validity[row] & (gain > 0.0))
        values = jnp.stack(measured)
        valid = jnp.stack(valid_rows) & jnp.isfinite(values)
        return FuelObservationResult(
            product=TheoryVector(values, self.layout, self.plan_id),
            valid=valid,
            successful=jnp.all(valid),
            units=self.units,
            grid_id=self.grid.grid_id,
            plan_id=self.plan_id,
        )

    def evaluate_states(
        self,
        states: tuple[FuelRodState, ...],
        /,
        *,
        sensor_nuisance: FuelSensorParameters | None = None,
        parameters: FuelModelBundle | None = None,
    ) -> FuelObservationResult:
        signals, validity, times = self.physical_signals(states, parameters=parameters)
        return self.apply_sensor_nuisance(
            signals, validity, times, sensor_nuisance=sensor_nuisance
        )

    def evaluate(
        self,
        result: FuelRodResult,
        /,
        *,
        sensor_nuisance: FuelSensorParameters | None = None,
        parameters: FuelModelBundle | None = None,
    ) -> FuelObservationResult:
        observation = self.evaluate_states(
            result.states, sensor_nuisance=sensor_nuisance, parameters=parameters
        )
        return eqx.tree_at(
            lambda record: record.successful,
            observation,
            observation.successful & result.successful,
        )
