#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ._components import (
    Compliance,
    MechanicsChamberCoupling,
    PeriodicElastance,
    PressureFlowComponent,
    StorageOwner,
    TimeVaryingElastance,
    WindkesselRCR,
)
from ._network import CirculationNetwork, PressureFlowConnection
from ._valves import ComplementarityValve, EventValve, SmoothValve


class SmoothValveRoute(StrictModule):
    open_resistance: Array
    closed_resistance: Array
    pressure_width: Array

    def __init__(
        self,
        open_resistance: ArrayLike = 1.0e-4,
        closed_resistance: ArrayLike = 100.0,
        pressure_width: ArrayLike = 0.02,
        /,
    ) -> None:
        self.open_resistance = jnp.asarray(open_resistance)
        self.closed_resistance = jnp.asarray(closed_resistance)
        self.pressure_width = jnp.asarray(pressure_width)
        SmoothValve(
            "validate_smooth_valve_route",
            self.open_resistance,
            self.closed_resistance,
            pressure_width=self.pressure_width,
        )


class ComplementarityValveRoute(StrictModule):
    open_resistance: Array
    smoothing: Array

    def __init__(
        self,
        open_resistance: ArrayLike = 1.0e-4,
        smoothing: ArrayLike = 1.0e-8,
        /,
    ) -> None:
        self.open_resistance = jnp.asarray(open_resistance)
        self.smoothing = jnp.asarray(smoothing)
        ComplementarityValve(
            "validate_complementarity_valve_route",
            self.open_resistance,
            smoothing=self.smoothing,
        )


class EventValveRoute(StrictModule):
    open_resistance: Array
    closed_resistance: Array
    opening_pressure: Array
    closing_pressure: Array
    minimum_dwell_time: Array

    def __init__(
        self,
        open_resistance: ArrayLike = 1.0e-4,
        closed_resistance: ArrayLike = 100.0,
        opening_pressure: ArrayLike = 0.0,
        closing_pressure: ArrayLike = -0.01,
        minimum_dwell_time: ArrayLike = 1.0,
        /,
    ) -> None:
        self.open_resistance = jnp.asarray(open_resistance)
        self.closed_resistance = jnp.asarray(closed_resistance)
        self.opening_pressure = jnp.asarray(opening_pressure)
        self.closing_pressure = jnp.asarray(closing_pressure)
        self.minimum_dwell_time = jnp.asarray(minimum_dwell_time)
        EventValve(
            "validate_event_valve_route",
            self.open_resistance,
            self.closed_resistance,
            opening_pressure=self.opening_pressure,
            closing_pressure=self.closing_pressure,
            minimum_dwell_time=self.minimum_dwell_time,
        )


ValveRoute = SmoothValveRoute | ComplementarityValveRoute | EventValveRoute


def _valve(name: str, route: ValveRoute, /) -> PressureFlowComponent:
    if isinstance(route, SmoothValveRoute):
        return SmoothValve(
            name,
            route.open_resistance,
            route.closed_resistance,
            pressure_width=route.pressure_width,
            pressure_scale=20.0,
            flow_scale=100.0,
        )
    if isinstance(route, ComplementarityValveRoute):
        return ComplementarityValve(
            name,
            route.open_resistance,
            smoothing=route.smoothing,
            pressure_scale=20.0,
            flow_scale=100.0,
        )
    if isinstance(route, EventValveRoute):
        return EventValve(
            name,
            route.open_resistance,
            route.closed_resistance,
            opening_pressure=route.opening_pressure,
            closing_pressure=route.closing_pressure,
            minimum_dwell_time=route.minimum_dwell_time,
            pressure_scale=20.0,
            flow_scale=100.0,
        )
    raise TypeError("route must be a concrete cardiovascular valve route.")


class ExternalChamberStorage(StrictModule):
    mechanics_chamber_id: str = eqx.field(static=True)
    initial_volume: Array

    def __init__(self, mechanics_chamber_id: str, initial_volume: ArrayLike, /) -> None:
        identifier = str(mechanics_chamber_id).strip()
        volume = jnp.asarray(initial_volume)
        if (
            not identifier
            or volume.shape != ()
            or not bool(jnp.isfinite(volume) & (volume >= 0.0))
        ):
            raise ValueError("External chamber storage requires an ID and finite volume.")
        self.mechanics_chamber_id = identifier
        self.initial_volume = volume


class ClosedLoopCirculation(StrictModule):
    """Closed 0D circulation model with explicit total-volume ownership."""

    network: CirculationNetwork
    chamber_names: tuple[str, ...] = eqx.field(static=True)
    external_storage: tuple[ExternalChamberStorage, ...]
    reference_total_volume: Array
    loop_id: str = eqx.field(static=True)

    def __init__(
        self,
        network: CirculationNetwork,
        chamber_names: Sequence[str],
        reference_total_volume: ArrayLike,
        /,
        *,
        external_storage: Sequence[ExternalChamberStorage] = (),
    ) -> None:
        if not isinstance(network, CirculationNetwork) or not network.closed:
            raise ValueError(
                "ClosedLoopCirculation requires a closed circulation network."
            )
        names = tuple(str(value) for value in chamber_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Closed-loop chamber names must be unique and non-empty.")
        for name in names:
            network.component(name)
        external = tuple(external_storage)
        if any(not isinstance(value, ExternalChamberStorage) for value in external):
            raise TypeError(
                "external_storage must contain ExternalChamberStorage values."
            )
        external_ids = tuple(value.mechanics_chamber_id for value in external)
        if len(set(external_ids)) != len(external_ids):
            raise ValueError("Mechanics chamber storage IDs must be unique.")
        mechanics_ids = tuple(
            component.mechanics_chamber_id
            for component in network.components
            if isinstance(component, MechanicsChamberCoupling)
        )
        if set(external_ids) != set(mechanics_ids):
            raise ValueError(
                "Every mechanics-owned coupling requires exactly one external storage record."
            )
        total = jnp.asarray(reference_total_volume)
        if total.shape != () or not bool(jnp.isfinite(total) & (total > 0.0)):
            raise ValueError("reference_total_volume must be finite and positive.")
        self.network = network
        self.chamber_names = names
        self.external_storage = external
        self.reference_total_volume = total
        self.loop_id = canonical_fingerprint(
            {
                "kind": "closed-loop-circulation",
                "network": network.network_id,
                "chambers": list(names),
                "external_storage": list(external_ids),
                "reference_total_volume": float(total).hex(),
            }
        )


def _ring_connections(
    component_names: Sequence[str],
    /,
) -> tuple[PressureFlowConnection, ...]:
    names = tuple(component_names)
    return tuple(
        PressureFlowConnection(left, "outlet", right, "inlet")
        for left, right in zip(names, names[1:] + names[:1], strict=True)
    )


def _initial_total(components: Sequence[PressureFlowComponent], /) -> Array:
    values = tuple(
        component.initial_value(variable)
        for component in components
        if component.storage_owner is StorageOwner.CIRCULATION
        for variable in component.storage_variable_names
    )
    if not values:
        raise ValueError("A closed circulation loop requires volume storage.")
    return jnp.sum(jnp.stack(values))


def systemic_closed_loop(
    *,
    cycle_length: ArrayLike = 800.0,
    valve_route: ValveRoute | None = None,
) -> ClosedLoopCirculation:
    """Build a left-heart/systemic circulation reference closed loop."""

    route = SmoothValveRoute() if valve_route is None else valve_route
    if not isinstance(
        route, (SmoothValveRoute, ComplementarityValveRoute, EventValveRoute)
    ):
        raise TypeError("valve_route must be a concrete valve route or None.")
    cycle = jnp.asarray(cycle_length)
    elastance = PeriodicElastance(
        8.0e-6,
        2.7e-4,
        cycle,
        0.38 * cycle,
    )
    components: tuple[PressureFlowComponent, ...] = (
        TimeVaryingElastance(
            "left_ventricle",
            elastance,
            unstressed_volume=10_000.0,
            initial_volume=135_000.0,
            pressure_scale=20.0,
            flow_scale=100.0,
            volume_scale=150_000.0,
        ),
        _valve("aortic_valve", route),
        WindkesselRCR(
            "systemic_arteries",
            0.005,
            11_000.0,
            0.12,
            unstressed_volume=70_000.0,
            initial_volume=81_000.0,
            pressure_scale=20.0,
            flow_scale=100.0,
            volume_scale=200_000.0,
        ),
        Compliance(
            "systemic_veins",
            55_000.0,
            unstressed_volume=2_500_000.0,
            initial_volume=2_555_000.0,
            pressure_scale=5.0,
            flow_scale=100.0,
            volume_scale=3_000_000.0,
        ),
        _valve("mitral_valve", route),
    )
    names = tuple(value.name for value in components)
    network = CirculationNetwork(components, _ring_connections(names))
    return ClosedLoopCirculation(
        network,
        ("left_ventricle",),
        _initial_total(components),
    )


def pulmonary_closed_loop(
    *,
    cycle_length: ArrayLike = 800.0,
    valve_route: ValveRoute | None = None,
) -> ClosedLoopCirculation:
    """Build a right-heart/pulmonary circulation reference closed loop."""

    route = SmoothValveRoute() if valve_route is None else valve_route
    if not isinstance(
        route, (SmoothValveRoute, ComplementarityValveRoute, EventValveRoute)
    ):
        raise TypeError("valve_route must be a concrete valve route or None.")
    cycle = jnp.asarray(cycle_length)
    components: tuple[PressureFlowComponent, ...] = (
        TimeVaryingElastance(
            "right_ventricle",
            PeriodicElastance(4.0e-6, 8.0e-5, cycle, 0.38 * cycle),
            unstressed_volume=15_000.0,
            initial_volume=140_000.0,
            pressure_scale=8.0,
            flow_scale=100.0,
            volume_scale=150_000.0,
        ),
        _valve("pulmonic_valve", route),
        WindkesselRCR(
            "pulmonary_arteries",
            0.001,
            35_000.0,
            0.018,
            unstressed_volume=80_000.0,
            initial_volume=97_500.0,
            pressure_scale=8.0,
            flow_scale=100.0,
            volume_scale=200_000.0,
        ),
        Compliance(
            "pulmonary_veins",
            25_000.0,
            unstressed_volume=300_000.0,
            initial_volume=312_500.0,
            pressure_scale=5.0,
            flow_scale=100.0,
            volume_scale=400_000.0,
        ),
        _valve("tricuspid_valve", route),
    )
    names = tuple(value.name for value in components)
    network = CirculationNetwork(components, _ring_connections(names))
    return ClosedLoopCirculation(
        network,
        ("right_ventricle",),
        _initial_total(components),
    )


def biventricular_closed_loop(
    *,
    cycle_length: ArrayLike = 800.0,
    valve_route: ValveRoute | None = None,
) -> ClosedLoopCirculation:
    """Build one volume-conserving systemic--pulmonary four-chamber ring."""

    route = SmoothValveRoute() if valve_route is None else valve_route
    if not isinstance(
        route, (SmoothValveRoute, ComplementarityValveRoute, EventValveRoute)
    ):
        raise TypeError("valve_route must be a concrete valve route or None.")
    cycle = jnp.asarray(cycle_length)
    components: tuple[PressureFlowComponent, ...] = (
        Compliance(
            "left_atrium",
            25_000.0,
            unstressed_volume=20_000.0,
            initial_volume=40_000.0,
            pressure_scale=5.0,
            flow_scale=100.0,
            volume_scale=80_000.0,
        ),
        _valve("mitral_valve", route),
        TimeVaryingElastance(
            "left_ventricle",
            PeriodicElastance(8.0e-6, 2.7e-4, cycle, 0.38 * cycle),
            unstressed_volume=10_000.0,
            initial_volume=110_000.0,
            pressure_scale=20.0,
            flow_scale=100.0,
            volume_scale=150_000.0,
        ),
        _valve("aortic_valve", route),
        WindkesselRCR(
            "systemic_vasculature",
            0.005,
            11_000.0,
            0.12,
            unstressed_volume=2_500_000.0,
            initial_volume=2_508_800.0,
            pressure_scale=20.0,
            flow_scale=100.0,
            volume_scale=3_000_000.0,
        ),
        _valve("tricuspid_valve", route),
        TimeVaryingElastance(
            "right_ventricle",
            PeriodicElastance(4.0e-6, 8.0e-5, cycle, 0.38 * cycle),
            unstressed_volume=15_000.0,
            initial_volume=215_000.0,
            pressure_scale=8.0,
            flow_scale=100.0,
            volume_scale=150_000.0,
        ),
        _valve("pulmonic_valve", route),
        WindkesselRCR(
            "pulmonary_vasculature",
            0.001,
            35_000.0,
            0.018,
            unstressed_volume=350_000.0,
            initial_volume=378_000.0,
            pressure_scale=8.0,
            flow_scale=100.0,
            volume_scale=600_000.0,
        ),
    )
    names = tuple(value.name for value in components)
    network = CirculationNetwork(components, _ring_connections(names))
    return ClosedLoopCirculation(
        network,
        ("left_atrium", "left_ventricle", "right_ventricle"),
        _initial_total(components),
    )


def replace_chamber_with_mechanics(
    model: ClosedLoopCirculation,
    chamber_name: str,
    coupling: MechanicsChamberCoupling,
    mechanics_initial_volume: ArrayLike,
    /,
) -> ClosedLoopCirculation:
    """Transfer one chamber's storage exclusively from circulation to mechanics."""

    if not isinstance(model, ClosedLoopCirculation):
        raise TypeError("model must be a ClosedLoopCirculation.")
    if chamber_name not in model.chamber_names:
        raise ValueError("chamber_name must identify a declared closed-loop chamber.")
    if not isinstance(coupling, MechanicsChamberCoupling):
        raise TypeError("coupling must be a MechanicsChamberCoupling.")
    original = model.network.component(chamber_name)
    if original.storage_owner is not StorageOwner.CIRCULATION:
        raise ValueError("Only a circulation-owned chamber can transfer storage.")
    if coupling.name != chamber_name:
        raise ValueError("Mechanics coupling must preserve the chamber component name.")
    if (
        coupling.storage_owner is not StorageOwner.MECHANICS
        or coupling.storage_variable_names
    ):
        raise ValueError(
            "Mechanics coupling must not duplicate a circulation volume state."
        )
    mechanics_storage = ExternalChamberStorage(
        coupling.mechanics_chamber_id,
        mechanics_initial_volume,
    )
    removed_volume = jnp.sum(
        jnp.stack(
            tuple(
                original.initial_value(variable)
                for variable in original.storage_variable_names
            )
        )
    )
    total = (
        model.reference_total_volume - removed_volume + mechanics_storage.initial_volume
    )
    external = model.external_storage + (mechanics_storage,)
    return ClosedLoopCirculation(
        model.network.replace_component(chamber_name, coupling),
        model.chamber_names,
        total,
        external_storage=external,
    )


__all__ = [
    "ClosedLoopCirculation",
    "ComplementarityValveRoute",
    "EventValveRoute",
    "ExternalChamberStorage",
    "SmoothValveRoute",
    "ValveRoute",
    "biventricular_closed_loop",
    "pulmonary_closed_loop",
    "replace_chamber_with_mechanics",
    "systemic_closed_loop",
]
