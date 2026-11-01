"""Partially trimmed cone and sphere quadric intersections through Boolean.

Each case cuts the intersection curve by a trim of one operand, so branches
end on exact trim crossings bound to curve-surface roots of the trim edges,
and the result survives a CAD archive round-trip.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep import brep_cone, brep_cylinder, brep_sphere
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._model import BRepModel
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)


def _boss(
    radius: float, top: float, base: float, coordinate_contract: SpatialCoordinateContract
) -> BRepModel:
    if top == radius:
        return brep_cylinder(
            radius,
            2.0,
            base_center=(base, 0.0, 1.0),
            axis=(1.0, 0.0, 0.0),
            coordinate_contract=coordinate_contract,
        )
    return brep_cone(
        radius,
        top,
        2.0,
        base_center=(base, 0.0, 1.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=coordinate_contract,
    )


@pytest.mark.parametrize(
    ("body", "boss", "branches"),
    (
        pytest.param(
            (1.0, 0.75),
            (0.5, 0.25, 0.75),
            5,
            id="cone-cone",
        ),
    ),
)
def test_boss_cut_by_its_own_cap_traces_trimmed_quadric_branches(
    tmp_path: Path,
    body: tuple[float, float],
    boss: tuple[float, float, float],
    branches: int,
) -> None:
    # A boss along x starts inside a vertical body: its cap disk cuts the
    # quartic intersection of the two lateral faces, which is traced over a
    # revolution angle and ends on the cap circle's exact roots.
    solid = (
        brep_cylinder(body[0], 2.0, coordinate_contract=_COORDINATES)
        if body[0] == body[1]
        else brep_cone(body[0], body[1], 2.0, coordinate_contract=_COORDINATES)
    )
    result = boolean_brep(solid, _boss(*boss, _COORDINATES), "difference")
    assert result.model.topology.num_solids == 1
    geometry = result.model.geometry
    assert geometry is not None
    assert len(geometry.face_loops) == 7
    traced = [curve for curve in geometry.curves if isinstance(curve, IntersectionCurve)]
    assert len(traced) == branches
    assert all(curve.fully_certified for curve in traced)
    assert all(
        (curve.start_kind, curve.end_kind) == ("boundary", "boundary") for curve in traced
    )
    for branch in traced:
        parameters = jnp.linspace(0.15, branch.num_charts - 0.15, 11)
        coupled = branch.evaluate_coupled(parameters)
        for region, chart in (
            (branch.first, coupled.first_parameters),
            (branch.second, coupled.second_parameters),
        ):
            np.testing.assert_allclose(
                region.patch.evaluate(chart), coupled.point, rtol=0.0, atol=1.0e-10
            )
        points = np.asarray(coupled.point)
        radii = body[0] + (body[1] - body[0]) * points[:, 2] / 2.0
        np.testing.assert_allclose(
            points[:, 0] ** 2 + points[:, 1] ** 2, radii**2, rtol=0.0, atol=1.0e-10
        )
    archive = tmp_path / "boss.phx"
    save_brep_archive(result.model, archive)
    restored = load_brep_archive(archive).geometry
    assert restored is not None
    assert tuple(
        None if root is None else root.root_id for root in restored.vertex_roots
    ) == tuple(None if root is None else root.root_id for root in geometry.vertex_roots)
    _cold_restore(archive)


@pytest.mark.parametrize("top_radius", (0.5, 0.25), ids=("cylinder", "cone"))
def test_sphere_cut_by_a_capped_quadric_traces_a_trimmed_branch(
    tmp_path: Path,
    top_radius: float,
) -> None:
    # The cylinder case obeys z = sqrt(1 - x). Both lateral quadrics meet
    # the same cap disk z = 1/2, whose trim crossings have x = 3/4.
    sphere = brep_sphere(1.0, axis=(0.0, 1.0, 0.0), coordinate_contract=_COORDINATES)
    cylinder = (
        brep_cylinder(
            0.5, 1.5, base_center=(0.5, 0.0, 0.5), coordinate_contract=_COORDINATES
        )
        if top_radius == 0.5
        else brep_cone(
            0.5,
            top_radius,
            1.5,
            base_center=(0.5, 0.0, 0.5),
            coordinate_contract=_COORDINATES,
        )
    )
    result = boolean_brep(sphere, cylinder, "difference")
    assert result.model.topology.num_solids == 1
    geometry = result.model.geometry
    assert geometry is not None
    assert len(geometry.face_loops) == 4
    traced = [curve for curve in geometry.curves if isinstance(curve, IntersectionCurve)]
    assert len(traced) == 3
    assert all(curve.fully_certified for curve in traced)
    for branch in traced:
        parameters = jnp.linspace(0.15, branch.num_charts - 0.15, 11)
        coupled = branch.evaluate_coupled(parameters)
        for region, chart in (
            (branch.first, coupled.first_parameters),
            (branch.second, coupled.second_parameters),
        ):
            np.testing.assert_allclose(
                region.patch.evaluate(chart), coupled.point, rtol=0.0, atol=1.0e-10
            )
        points = np.asarray(coupled.point)
        np.testing.assert_allclose(np.sum(points**2, axis=1), 1.0, rtol=0.0, atol=1.0e-10)
    archive = tmp_path / "sphere-cylinder.phx"
    save_brep_archive(result.model, archive)
    restored = load_brep_archive(archive).geometry
    assert restored is not None
    assert tuple(
        None if root is None else root.root_id for root in restored.vertex_roots
    ) == tuple(None if root is None else root.root_id for root in geometry.vertex_roots)
    _cold_restore(archive)


def _cold_restore(archive: Path) -> None:
    worker = (
        Path(__file__).resolve().parents[1]
        / "interchange"
        / "_native_cad_archive_worker.py"
    )
    subprocess.run(
        [sys.executable, str(worker), "read-brep", str(archive)],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )


def test_sphere_quadric_recognition_requires_exact_meridian_metric() -> None:
    from phydrax.geometry.brep._patches import CircleCurve, RevolutionSurface
    from phydrax.geometry.brep._quadric_tracing import _sphere_frame

    def surface(first: tuple[float, float, float]) -> RevolutionSurface:
        return RevolutionSurface(
            CircleCurve((0.0, 0.0, 0.0), first, (0.0, 0.0, 1.0), 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    assert _sphere_frame(surface((1.0, 0.0, 0.0))) is not None
    diagonal = float(np.sqrt(0.5))
    assert _sphere_frame(surface((diagonal, diagonal, 0.0))) is None
