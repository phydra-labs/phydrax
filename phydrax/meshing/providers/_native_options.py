#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Closed native route selection and deterministic numerical schedules."""

from __future__ import annotations

from typing import final, Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.implicit._policy import (
    AdaptiveImplicitSurfacePolicy,
    ImplicitSurfacePolicy,
)
from ...typing import checked, parse
from .._hex_schedule import NativeHexGridSchedule
from .._polyhedral_generation import NativePolyhedralSchedule
from .._volume_generation import NativeVolumeSchedule
from . import _native_schedules as schedules_owner


NativeMeshingRoute: TypeAlias = Literal[
    "planar_constrained_delaunay",
    "implicit_surface",
    "implicit_restricted_delaunay",
    "implicit_adaptive_tetrahedral",
    "curve_arc_length",
    "parametric_surface",
    "reconstructed_surface",
    "plc_tetrahedral",
    "cad_tetrahedral",
    "periodic_delaunay",
    "periodic_power",
    "plc_restricted_power",
    "layer_core",
    "surface_envelope_tetrahedral",
    "structured_transfinite",
    "sweep",
    "image_material_tetrahedral",
    "planar_dual_quad",
    "plc_dual_hex",
    "plc_hex_dominant",
    "plc_balanced_grid_hex",
    "plc_frame_grid_hex",
    "mapped_balanced_grid_hex",
    "mapped_frame_grid_hex",
]

_ROUTE_SCHEDULE_FIELDS: dict[NativeMeshingRoute, tuple[str, ...]] = {
    "planar_constrained_delaunay": (),
    "reconstructed_surface": (),
    "implicit_surface": ("implicit_policy",),
    "implicit_restricted_delaunay": ("volume_schedule",),
    "implicit_adaptive_tetrahedral": ("implicit_policy", "volume_schedule"),
    "curve_arc_length": ("curve_schedule",),
    "parametric_surface": ("surface_schedule",),
    "plc_tetrahedral": ("volume_schedule",),
    "cad_tetrahedral": ("volume_schedule",),
    "periodic_delaunay": (),
    "periodic_power": (),
    "plc_restricted_power": ("polyhedral_schedule",),
    "layer_core": ("volume_schedule", "polyhedral_schedule"),
    "surface_envelope_tetrahedral": ("volume_schedule",),
    "structured_transfinite": ("structured_schedule",),
    "sweep": (),
    "image_material_tetrahedral": ("volume_schedule",),
    "planar_dual_quad": (),
    "plc_dual_hex": ("volume_schedule",),
    "plc_hex_dominant": ("volume_schedule", "hex_core_fraction"),
    "plc_balanced_grid_hex": ("volume_schedule", "grid_schedule"),
    "plc_frame_grid_hex": ("volume_schedule", "grid_schedule"),
    "mapped_balanced_grid_hex": ("grid_schedule",),
    "mapped_frame_grid_hex": ("grid_schedule",),
}


@final
class _ResolvedSchedules(StrictModule):
    implicit: AdaptiveImplicitSurfacePolicy | ImplicitSurfacePolicy | None
    curve: schedules_owner.NativeCurveSchedule | None
    surface: schedules_owner.NativeSurfaceSchedule | None
    volume: NativeVolumeSchedule | None
    polyhedral: NativePolyhedralSchedule | None
    structured: schedules_owner.NativeStructuredSchedule | None
    grid: NativeHexGridSchedule | None


def _resolve_schedules(
    route: NativeMeshingRoute,
    implicit: AdaptiveImplicitSurfacePolicy | ImplicitSurfacePolicy | None,
    curve: schedules_owner.NativeCurveSchedule | None,
    surface: schedules_owner.NativeSurfaceSchedule | None,
    volume: NativeVolumeSchedule | None,
    polyhedral: NativePolyhedralSchedule | None,
    structured: schedules_owner.NativeStructuredSchedule | None,
    grid: NativeHexGridSchedule | None,
) -> _ResolvedSchedules:
    fields = _ROUTE_SCHEDULE_FIELDS[route]
    if "implicit_policy" in fields and implicit is None:
        implicit = AdaptiveImplicitSurfacePolicy()
    if "curve_schedule" in fields and curve is None:
        curve = schedules_owner.NativeCurveSchedule()
    if "surface_schedule" in fields and surface is None:
        surface = schedules_owner.NativeSurfaceSchedule()
    if "volume_schedule" in fields and volume is None:
        volume = NativeVolumeSchedule()
    if "polyhedral_schedule" in fields and polyhedral is None:
        polyhedral = NativePolyhedralSchedule()
    if "structured_schedule" in fields and structured is None:
        structured = schedules_owner.NativeStructuredSchedule()
    match route:
        case "plc_balanced_grid_hex" | "mapped_balanced_grid_hex":
            grid_route = "balanced_grid"
        case "plc_frame_grid_hex" | "mapped_frame_grid_hex":
            grid_route = "frame_grid"
        case _:
            grid_route = None
    if grid_route is not None:
        grid = NativeHexGridSchedule(grid_route) if grid is None else grid
        if grid.route != grid_route:
            raise ValueError(
                "NativeMeshingOptions.grid_schedule.route must match selected route."
            )
    return _ResolvedSchedules(
        implicit, curve, surface, volume, polyhedral, structured, grid
    )


@final
class NativeMeshingOptions(StrictModule, NonTrainableState):
    """Algorithm choices; an omitted route is resolved from source and request."""

    route: NativeMeshingRoute | None = eqx.field(static=True)
    implicit_policy: AdaptiveImplicitSurfacePolicy | ImplicitSurfacePolicy | None = (
        eqx.field(static=True)
    )
    curve_schedule: schedules_owner.NativeCurveSchedule | None
    surface_schedule: schedules_owner.NativeSurfaceSchedule | None
    volume_schedule: NativeVolumeSchedule | None
    polyhedral_schedule: NativePolyhedralSchedule | None = eqx.field(static=True)
    structured_schedule: schedules_owner.NativeStructuredSchedule | None
    hex_core_fraction: float | None = eqx.field(static=True)
    grid_schedule: NativeHexGridSchedule | None
    options_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        route: NativeMeshingRoute | None = None,
        /,
        *,
        implicit_policy: AdaptiveImplicitSurfacePolicy
        | ImplicitSurfacePolicy
        | None = None,
        curve_schedule: schedules_owner.NativeCurveSchedule | None = None,
        surface_schedule: schedules_owner.NativeSurfaceSchedule | None = None,
        volume_schedule: NativeVolumeSchedule | None = None,
        polyhedral_schedule: NativePolyhedralSchedule | None = None,
        structured_schedule: schedules_owner.NativeStructuredSchedule | None = None,
        hex_core_fraction: float | None = None,
        grid_schedule: NativeHexGridSchedule | None = None,
    ) -> None:
        route_ = (
            None
            if route is None
            else parse(route, NativeMeshingRoute, "NativeMeshingOptions.route")
        )
        fields = () if route_ is None else _ROUTE_SCHEDULE_FIELDS[route_]
        supplied = (
            ("grid_schedule", grid_schedule),
            ("hex_core_fraction", hex_core_fraction),
            ("structured_schedule", structured_schedule),
            ("polyhedral_schedule", polyhedral_schedule),
            ("implicit_policy", implicit_policy),
            ("curve_schedule", curve_schedule),
            ("surface_schedule", surface_schedule),
            ("volume_schedule", volume_schedule),
        )
        for name, value in supplied:
            if route_ is not None and value is not None and name not in fields:
                raise ValueError(
                    f"NativeMeshingOptions.{name} must schedule the selected route."
                )
        if route_ == "implicit_adaptive_tetrahedral" and isinstance(
            implicit_policy, ImplicitSurfacePolicy
        ):
            raise ValueError(
                "implicit_adaptive_tetrahedral requires AdaptiveImplicitSurfacePolicy."
            )
        if hex_core_fraction is not None:
            if isinstance(hex_core_fraction, (bool, np.bool_)):
                raise TypeError("hex_core_fraction must be a real fraction.")
            fraction = float(hex_core_fraction)
            if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError("hex_core_fraction must lie in (0, 1].")
        else:
            fraction = 0.8 if route_ == "plc_hex_dominant" else None
        schedules = (
            _ResolvedSchedules(
                implicit_policy,
                curve_schedule,
                surface_schedule,
                volume_schedule,
                polyhedral_schedule,
                structured_schedule,
                grid_schedule,
            )
            if route_ is None
            else _resolve_schedules(
                route_,
                implicit_policy,
                curve_schedule,
                surface_schedule,
                volume_schedule,
                polyhedral_schedule,
                structured_schedule,
                grid_schedule,
            )
        )
        policy, curve = schedules.implicit, schedules.curve
        surface, volume = schedules.surface, schedules.volume
        polyhedral, structured, grid = (
            schedules.polyhedral,
            schedules.structured,
            schedules.grid,
        )
        self.route = route_
        self.implicit_policy = policy
        self.curve_schedule = curve
        self.surface_schedule = surface
        self.volume_schedule = volume
        self.polyhedral_schedule = polyhedral
        self.structured_schedule = structured
        self.hex_core_fraction = fraction
        self.grid_schedule = grid
        self.options_id = canonical_fingerprint(
            {
                "kind": "native-meshing-options",
                "route": route_,
                "implicit_policy": None if policy is None else repr(policy),
                "curve_schedule": None if curve is None else curve.schedule_id,
                "surface_schedule": None if surface is None else surface.schedule_id,
                "volume_schedule": None if volume is None else volume.schedule_id,
                "polyhedral_schedule": None
                if polyhedral is None
                else polyhedral.schedule_id,
                "structured_schedule": None
                if structured is None
                else structured.schedule_id,
                "hex_core_fraction": fraction,
                "grid_schedule": None if grid is None else grid.schedule_id,
            }
        )


__all__ = ["NativeMeshingOptions", "NativeMeshingRoute"]
