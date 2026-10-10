from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from tools.meshing_qualification import (
    _hybrid_qualification_original_layer_sweep_profile,
    _hybrid_qualification_original_wall_profile,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="exact layer certification requires meshcore"
)


def test_boundary_layer_scenario_1() -> None:
    for profile in ("plate5", "quadrilateral_plate3", "opposing_sloped_plates4"):
        _hybrid_qualification_original_wall_profile(profile)


def test_boundary_layer_scenario_2() -> None:
    for profile in (
        "sphere_radius04_subdivision2",
        "outward_box_count3",
        "concave_box_count3",
    ):
        _hybrid_qualification_original_wall_profile(profile)


def test_boundary_layer_scenario_3() -> None:
    for profile in (
        "wedge30_rejection",
        "open_box_floor_count4",
        "channel_gap005_rejection",
        "channel_gap005_policies",
    ):
        _hybrid_qualification_original_wall_profile(profile)


@pytest.mark.parametrize(
    ("thicknesses", "remaining_core"),
    (
        ((0.05, 0.10), True),
        ((0.25, 0.75), False),
    ),
    ids=("slab-and-core", "entire-solid-slab"),
)
def test_native_cad_extrusion_preserves_physical_wall_cap_and_volume(
    tmp_path: Any, thicknesses: tuple[float, ...], remaining_core: bool
) -> None:
    profile = "cad-slab-and-core" if remaining_core else "cad-entire-solid-slab"
    owners = _hybrid_qualification_original_layer_sweep_profile(
        profile, destination=tmp_path / "native-extrusion.brep"
    )
    np.testing.assert_array_equal(owners["control"].schedule.thicknesses, thicknesses)


@pytest.mark.parametrize(
    ("thickness", "opposite_walls", "reason"),
    (
        (1.25, False, "leaves its controlled solid"),
        (0.75, True, "slabs of different walls overlap"),
    ),
    ids=("outside-controlled-solid", "overlapping-wall-slabs"),
)
def test_native_cad_extrusion_refuses_invalid_slabs_atomically(
    tmp_path: Path,
    thickness: float,
    opposite_walls: bool,
    reason: str,
) -> None:
    profile = (
        "cad-overlapping-wall-slabs" if opposite_walls else "cad-outside-controlled-solid"
    )
    owners = _hybrid_qualification_original_layer_sweep_profile(
        profile, destination=tmp_path / "protected.brep"
    )
    assert reason in owners["rejection"]
    np.testing.assert_array_equal(owners["control"].schedule.thicknesses, (thickness,))


@pytest.mark.parametrize("periodicity", ("translation", "partial", "rotation"))
@pytest.mark.parametrize("quadrilateral", (False, True))
def test_periodic_layers_preserve_winding_cap_and_physical_schedule(
    periodicity: str,
    quadrilateral: bool,
) -> None:
    kind = "quadrilateral" if quadrilateral else "triangle"
    _hybrid_qualification_original_layer_sweep_profile(f"periodic-{periodicity}-{kind}")


def test_periodic_layers_preserve_a_convex_ridge_crossing_the_seam() -> None:
    _hybrid_qualification_original_layer_sweep_profile("periodic-convex-ridge")


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("perturb_cap", (False, True))
def test_exact_sweep_authors_independent_measured_reference_stations(
    periodic: bool,
    perturb_cap: bool,
) -> None:
    periodicity = "periodic" if periodic else "nonperiodic"
    outcome = "perturbed-cap" if perturb_cap else "accepted"
    _hybrid_qualification_original_layer_sweep_profile(
        f"exact-sweep-{periodicity}-{outcome}"
    )
