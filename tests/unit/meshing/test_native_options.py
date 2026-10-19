"""Native route schedule admission and deterministic option identity."""

from collections.abc import Callable
from functools import partial
from typing import get_args

import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.meshing import (
    NativeCurveSchedule,
    NativeHexGridSchedule,
    NativeMeshingOptions,
    NativeMeshingRoute,
    NativePolyhedralSchedule,
    NativeStructuredSchedule,
    NativeSurfaceSchedule,
    NativeVolumeSchedule,
)


@pytest.mark.parametrize(
    "route", get_args(NativeMeshingRoute), ids=get_args(NativeMeshingRoute)
)
def test_every_native_route_resolves_deterministic_default_options(
    route: NativeMeshingRoute,
) -> None:
    assert (
        NativeMeshingOptions(route).options_id == NativeMeshingOptions(route).options_id
    )


@pytest.mark.parametrize(
    ("field", "factory"),
    [
        (
            "curve_schedule",
            partial(NativeMeshingOptions, curve_schedule=NativeCurveSchedule()),
        ),
        (
            "surface_schedule",
            partial(NativeMeshingOptions, surface_schedule=NativeSurfaceSchedule()),
        ),
        (
            "volume_schedule",
            partial(NativeMeshingOptions, volume_schedule=NativeVolumeSchedule()),
        ),
        (
            "polyhedral_schedule",
            partial(NativeMeshingOptions, polyhedral_schedule=NativePolyhedralSchedule()),
        ),
        (
            "structured_schedule",
            partial(NativeMeshingOptions, structured_schedule=NativeStructuredSchedule()),
        ),
        (
            "grid_schedule",
            partial(
                NativeMeshingOptions, grid_schedule=NativeHexGridSchedule("balanced_grid")
            ),
        ),
        ("hex_core_fraction", partial(NativeMeshingOptions, hex_core_fraction=0.5)),
    ],
    ids=["curve", "surface", "volume", "power", "structured", "grid", "hybrid"],
)
def test_planar_route_refuses_each_foreign_schedule_field(
    field: str,
    factory: Callable[[NativeMeshingRoute], NativeMeshingOptions],
) -> None:
    with pytest.raises(ValueError, match=field):
        factory("planar_constrained_delaunay")


@pytest.mark.parametrize(
    "route",
    ["plc_balanced_grid_hex", "mapped_balanced_grid_hex"],
    ids=["plc", "mapped"],
)
def test_balanced_route_refuses_a_frame_schedule(route: NativeMeshingRoute) -> None:
    with pytest.raises(ValueError, match="grid_schedule.route"):
        NativeMeshingOptions(route, grid_schedule=NativeHexGridSchedule("frame_grid"))


def test_polyhedral_options_bind_the_schedule_identity_like_other_routes() -> None:
    schedule = NativePolyhedralSchedule(maximum_refinement_steps=0)
    options = NativeMeshingOptions("plc_restricted_power", polyhedral_schedule=schedule)
    assert options.polyhedral_schedule is schedule
    assert options.options_id == canonical_fingerprint(
        {
            "kind": "native-meshing-options",
            "route": "plc_restricted_power",
            "implicit_policy": None,
            "curve_schedule": None,
            "surface_schedule": None,
            "volume_schedule": None,
            "polyhedral_schedule": schedule.schedule_id,
            "structured_schedule": None,
            "hex_core_fraction": None,
            "grid_schedule": None,
        }
    )
    other = NativeMeshingOptions(
        "plc_restricted_power",
        polyhedral_schedule=NativePolyhedralSchedule(maximum_refinement_steps=1),
    )
    assert options.options_id != other.options_id
