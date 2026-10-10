#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native meshing route selection and numerical scheduling.

`NativeMeshingOptions` selects one native construction route and schedules its
numerics. Physical requirements (sizes, fidelity, quality, limits) belong to
the meshing specification; options never relax or add them, and never carry
another engine's tuning vocabulary.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.implicit import AdaptiveImplicitSurfacePolicy, ImplicitSurfacePolicy
from ...typing import parse
from .._hex_generation import NativeHexGridSchedule
from .._polyhedral_generation import NativePolyhedralSchedule
from .._volume_generation import NativeVolumeSchedule


NativeMeshingRoute: TypeAlias = Literal[
    "planar_constrained_delaunay",
    "implicit_surface",
    "implicit_restricted_delaunay",
    "implicit_adaptive_tetrahedral",
    "curve_arc_length",
    "parametric_surface",
    "plc_tetrahedral",
    "periodic_delaunay",
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


def _positive_count(value: int, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@final
class NativeCurveSchedule(StrictModule, NonTrainableState):
    """Numerical schedule of arc-length curve discretization.

    ``quadrature_intervals`` and ``quadrature_order`` set the composite
    Gauss-Legendre rule of arc-length and size-density integration per curve;
    ``fidelity_samples`` is the number of interior samples of each interval at
    which chord deviation is measured.
    """

    quadrature_intervals: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    fidelity_samples: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        quadrature_intervals: int = 64,
        quadrature_order: int = 8,
        fidelity_samples: int = 8,
    ) -> None:
        intervals = _positive_count(quadrature_intervals, "quadrature_intervals")
        order = _positive_count(quadrature_order, "quadrature_order")
        samples = _positive_count(fidelity_samples, "fidelity_samples")
        self.quadrature_intervals = intervals
        self.quadrature_order = order
        self.fidelity_samples = samples
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-curve-schedule",
                "quadrature": [intervals, order],
                "fidelity_samples": samples,
            }
        )


@final
class NativeSurfaceSchedule(StrictModule, NonTrainableState):
    """Numerical schedule of chart-space surface generation.

    ``quality_angle_degrees`` is the physical minimum-angle aim of refinement
    (a requested hard quality target is enforced by compliance, never relaxed
    by the aim); ``maximum_rounds`` bounds the refinement batches;
    ``spacing_fraction`` of the local size is the smallest physical distance
    at which a refinement point may approach an existing vertex;
    ``curve_schedule`` discretizes the shared feature curves.
    """

    quality_angle_degrees: float = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)
    spacing_fraction: float = eqx.field(static=True)
    curve_schedule: NativeCurveSchedule
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        quality_angle_degrees: float = 22.0,
        maximum_rounds: int = 64,
        spacing_fraction: float = 0.25,
        curve_schedule: NativeCurveSchedule | None = None,
    ) -> None:
        angle = float(quality_angle_degrees)
        spacing = float(spacing_fraction)
        if not np.isfinite(angle) or not 0.0 <= angle < 34.0:
            raise ValueError("quality_angle_degrees must lie in [0, 34).")
        if not np.isfinite(spacing) or not 0.0 < spacing < 1.0:
            raise ValueError("spacing_fraction must lie in (0, 1).")
        if curve_schedule is not None and not isinstance(
            curve_schedule, NativeCurveSchedule
        ):
            raise TypeError("curve_schedule must be NativeCurveSchedule or None.")
        rounds = _positive_count(maximum_rounds, "maximum_rounds")
        curves = NativeCurveSchedule() if curve_schedule is None else curve_schedule
        self.quality_angle_degrees = angle
        self.maximum_rounds = rounds
        self.spacing_fraction = spacing
        self.curve_schedule = curves
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-surface-schedule",
                "quality_angle_degrees": angle,
                "maximum_rounds": rounds,
                "spacing_fraction": spacing,
                "curve_schedule": curves.schedule_id,
            }
        )


@final
class NativeStructuredSchedule(StrictModule, NonTrainableState):
    """Bounded variational optimization of declared transfinite blocks."""

    optimization_steps: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, *, optimization_steps: int = 0) -> None:
        if isinstance(optimization_steps, (bool, np.bool_)) or not isinstance(
            optimization_steps, (int, np.integer)
        ):
            raise TypeError("optimization_steps must be an integer.")
        if optimization_steps < 0:
            raise ValueError("optimization_steps must be nonnegative.")
        steps = int(optimization_steps)
        self.optimization_steps = steps
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-structured-schedule",
                "optimization_steps": steps,
            }
        )


@final
class NativeMeshingOptions(StrictModule, NonTrainableState):
    """One native route and its numerical schedule.

    ``implicit_policy`` schedules ``"implicit_surface"`` discovery:
    `AdaptiveImplicitSurfacePolicy` (the default) selects adaptive discovery,
    while `ImplicitSurfacePolicy` selects design-differentiable fixed-lattice
    realization. ``"implicit_adaptive_tetrahedral"`` combines the adaptive
    policy with ``volume_schedule``; its scalar interpretation stays unchanged.
    ``curve_schedule`` schedules ``"curve_arc_length"`` and ``surface_schedule``
    schedules ``"parametric_surface"``. ``volume_schedule`` schedules PLC,
    image/material, repaired-envelope, layer-core and dual-hex tetrahedral
    construction stages. ``polyhedral_schedule`` schedules restricted power
    cells; ``structured_schedule`` schedules transfinite block optimization.
    Physical power sites and weights belong to ``NativePolyhedralSource``,
    not tuning options. A schedule supplied to another route is refused.
    """

    route: NativeMeshingRoute = eqx.field(static=True)
    implicit_policy: AdaptiveImplicitSurfacePolicy | ImplicitSurfacePolicy | None = (
        eqx.field(static=True)
    )
    curve_schedule: NativeCurveSchedule | None
    surface_schedule: NativeSurfaceSchedule | None
    volume_schedule: NativeVolumeSchedule | None
    polyhedral_schedule: NativePolyhedralSchedule | None = eqx.field(static=True)
    structured_schedule: NativeStructuredSchedule | None
    hex_core_fraction: float | None = eqx.field(static=True)
    grid_schedule: NativeHexGridSchedule | None
    options_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: NativeMeshingRoute,
        /,
        *,
        implicit_policy: AdaptiveImplicitSurfacePolicy
        | ImplicitSurfacePolicy
        | None = None,
        curve_schedule: NativeCurveSchedule | None = None,
        surface_schedule: NativeSurfaceSchedule | None = None,
        volume_schedule: NativeVolumeSchedule | None = None,
        polyhedral_schedule: NativePolyhedralSchedule | None = None,
        structured_schedule: NativeStructuredSchedule | None = None,
        hex_core_fraction: float | None = None,
        grid_schedule: NativeHexGridSchedule | None = None,
    ) -> None:
        route_ = parse(route, NativeMeshingRoute, "route")
        if grid_schedule is not None:
            if not isinstance(grid_schedule, NativeHexGridSchedule):
                raise TypeError("grid_schedule must be NativeHexGridSchedule or None.")
            match route_:
                case (
                    "plc_balanced_grid_hex"
                    | "plc_frame_grid_hex"
                    | "mapped_balanced_grid_hex"
                    | "mapped_frame_grid_hex"
                ):
                    pass
                case _:
                    raise ValueError(
                        "grid_schedule schedules the declared hex-grid routes only."
                    )
        if hex_core_fraction is not None:
            if route_ != "plc_hex_dominant":
                raise ValueError(
                    "hex_core_fraction schedules the hex-dominant route only."
                )
            if isinstance(hex_core_fraction, (bool, np.bool_)):
                raise TypeError("hex_core_fraction must be a real fraction.")
            fraction = float(hex_core_fraction)
            if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError("hex_core_fraction must lie in (0, 1].")
        else:
            fraction = 0.8 if route_ == "plc_hex_dominant" else None
        if structured_schedule is not None and not isinstance(
            structured_schedule, NativeStructuredSchedule
        ):
            raise TypeError(
                "structured_schedule must be NativeStructuredSchedule or None."
            )
        if structured_schedule is not None and route_ != "structured_transfinite":
            raise ValueError("structured_schedule schedules transfinite blocks only.")
        if polyhedral_schedule is not None and not isinstance(
            polyhedral_schedule, NativePolyhedralSchedule
        ):
            raise TypeError(
                "polyhedral_schedule must be NativePolyhedralSchedule or None."
            )
        if polyhedral_schedule is not None and route_ != "plc_restricted_power":
            raise ValueError(
                "polyhedral_schedule schedules the restricted power route only."
            )
        if implicit_policy is not None and type(implicit_policy) not in (
            AdaptiveImplicitSurfacePolicy,
            ImplicitSurfacePolicy,
        ):
            raise TypeError(
                "implicit_policy must be AdaptiveImplicitSurfacePolicy, "
                "ImplicitSurfacePolicy, or None."
            )
        if curve_schedule is not None and not isinstance(
            curve_schedule, NativeCurveSchedule
        ):
            raise TypeError("curve_schedule must be NativeCurveSchedule or None.")
        if surface_schedule is not None and not isinstance(
            surface_schedule, NativeSurfaceSchedule
        ):
            raise TypeError("surface_schedule must be NativeSurfaceSchedule or None.")
        if surface_schedule is not None and route_ != "parametric_surface":
            raise ValueError(
                "surface_schedule schedules the parametric surface route only."
            )
        if volume_schedule is not None and not isinstance(
            volume_schedule, NativeVolumeSchedule
        ):
            raise TypeError("volume_schedule must be NativeVolumeSchedule or None.")
        if volume_schedule is not None:
            match route_:
                case (
                    "plc_tetrahedral"
                    | "layer_core"
                    | "surface_envelope_tetrahedral"
                    | "image_material_tetrahedral"
                    | "plc_dual_hex"
                    | "plc_hex_dominant"
                    | "plc_balanced_grid_hex"
                    | "plc_frame_grid_hex"
                    | "implicit_restricted_delaunay"
                    | "implicit_adaptive_tetrahedral"
                ):
                    pass
                case _:
                    raise ValueError(
                        "volume_schedule schedules native tetrahedral core routes only."
                    )
        surface = None
        volume = None
        polyhedral = None
        structured = None
        grid = None
        match route_:
            case "implicit_surface":
                if curve_schedule is not None:
                    raise ValueError("curve_schedule schedules the curve route only.")
                policy = (
                    AdaptiveImplicitSurfacePolicy()
                    if implicit_policy is None
                    else implicit_policy
                )
                schedule = None
            case "implicit_adaptive_tetrahedral":
                if curve_schedule is not None or (
                    implicit_policy is not None
                    and type(implicit_policy) is not AdaptiveImplicitSurfacePolicy
                ):
                    raise ValueError(
                        "Adaptive implicit volumes require an adaptive discovery policy."
                    )
                policy = (
                    AdaptiveImplicitSurfacePolicy()
                    if implicit_policy is None
                    else implicit_policy
                )
                schedule = None
                volume = (
                    NativeVolumeSchedule() if volume_schedule is None else volume_schedule
                )
            case "curve_arc_length":
                if implicit_policy is not None:
                    raise ValueError("implicit_policy schedules the implicit route only.")
                policy = None
                schedule = (
                    NativeCurveSchedule() if curve_schedule is None else curve_schedule
                )
            case (
                "planar_constrained_delaunay"
                | "periodic_delaunay"
                | "sweep"
                | "planar_dual_quad"
            ):
                if implicit_policy is not None or curve_schedule is not None:
                    raise ValueError(
                        f"The {route_} route has no numerical schedule options."
                    )
                policy = None
                schedule = None
            case "parametric_surface":
                if implicit_policy is not None or curve_schedule is not None:
                    raise ValueError(
                        "The parametric surface route is scheduled by surface_schedule."
                    )
                policy = None
                schedule = None
                surface = (
                    NativeSurfaceSchedule()
                    if surface_schedule is None
                    else surface_schedule
                )
            case (
                "plc_tetrahedral"
                | "layer_core"
                | "surface_envelope_tetrahedral"
                | "image_material_tetrahedral"
                | "plc_dual_hex"
                | "plc_hex_dominant"
                | "implicit_restricted_delaunay"
            ):
                if implicit_policy is not None or curve_schedule is not None:
                    raise ValueError(
                        f"The {route_} route is scheduled by volume_schedule."
                    )
                policy = None
                schedule = None
                volume = (
                    NativeVolumeSchedule() if volume_schedule is None else volume_schedule
                )
            case "plc_restricted_power":
                if implicit_policy is not None or curve_schedule is not None:
                    raise ValueError(
                        "The restricted power route is scheduled by polyhedral_schedule."
                    )
                policy = None
                schedule = None
                polyhedral = (
                    NativePolyhedralSchedule()
                    if polyhedral_schedule is None
                    else polyhedral_schedule
                )
            case "structured_transfinite":
                if implicit_policy is not None or curve_schedule is not None:
                    raise ValueError(
                        "Transfinite blocks are scheduled by structured_schedule."
                    )
                policy = None
                schedule = None
                structured = (
                    NativeStructuredSchedule()
                    if structured_schedule is None
                    else structured_schedule
                )
            case (
                "plc_balanced_grid_hex"
                | "plc_frame_grid_hex"
                | "mapped_balanced_grid_hex"
                | "mapped_frame_grid_hex"
            ):
                if implicit_policy is not None or curve_schedule is not None:
                    raise ValueError("Hex-grid routes are scheduled by grid_schedule.")
                policy = None
                schedule = None
                match route_:
                    case "plc_balanced_grid_hex" | "mapped_balanced_grid_hex":
                        grid = (
                            NativeHexGridSchedule("balanced_grid")
                            if grid_schedule is None
                            else grid_schedule
                        )
                        if grid.route != "balanced_grid":
                            raise ValueError(
                                "The balanced-grid route requires its declared grid schedule."
                            )
                    case "plc_frame_grid_hex" | "mapped_frame_grid_hex":
                        grid = (
                            NativeHexGridSchedule("frame_grid")
                            if grid_schedule is None
                            else grid_schedule
                        )
                        if grid.route != "frame_grid":
                            raise ValueError(
                                "The frame-grid route requires its declared grid schedule."
                            )
                    case _:
                        assert_never(route_)
                match route_:
                    case "plc_balanced_grid_hex" | "plc_frame_grid_hex":
                        volume = (
                            NativeVolumeSchedule()
                            if volume_schedule is None
                            else volume_schedule
                        )
                    case "mapped_balanced_grid_hex" | "mapped_frame_grid_hex":
                        volume = None
                    case _:
                        assert_never(route_)
            case _:
                assert_never(route_)
        self.route = route_
        self.implicit_policy = policy
        self.curve_schedule = schedule
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
                "curve_schedule": None if schedule is None else schedule.schedule_id,
                "surface_schedule": None if surface is None else surface.schedule_id,
                "volume_schedule": None if volume is None else volume.schedule_id,
                "polyhedral_schedule": None if polyhedral is None else asdict(polyhedral),
                "structured_schedule": None
                if structured is None
                else structured.schedule_id,
                "hex_core_fraction": fraction,
                "grid_schedule": None if grid is None else grid.schedule_id,
            }
        )


__all__ = [
    "NativeCurveSchedule",
    "NativeMeshingOptions",
    "NativeMeshingRoute",
    "NativeSurfaceSchedule",
    "NativeStructuredSchedule",
]
