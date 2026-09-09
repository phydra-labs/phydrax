#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Artifact and convergence boundaries; real external solves live in the benchmark."""

import re

import jax
import numpy as np
import pytest

from phydrax.interchange.dafoam import _request, run_dafoam
from phydrax.interchange.xfoil import (
    _geometry_bytes,
    _parse_polar,
    run_xfoil_point,
    XFOILOperatingPoint,
)
from tools.aerodynamic_design_adapter_benchmarks import (
    GEOMETRY_SOURCE,
    naca0012_coordinates,
    naca0012_dafoam_case,
)


# Native POLWRIT/PACC file syntax, not an executable pretending to be XFOIL.
_POLAR_HEADER = b"""
 XFOIL Version 6.99
 Calculated polar for: phydrax-airfoil
 1 1 Reynolds number fixed       Mach number fixed
 xtrf = 1.000 (top)    1.000 (bottom)
 Mach = 0.000     Re = 1.000 e 6     Ncrit = 9.000
 alpha CL CD CDp CM Top_Xtr Bot_Xtr
 ------- -------- -------- -------- -------- -------- --------
"""
_ROW = b" 2.000 0.2180 0.00580 0.00082 -0.0013 0.5240 0.7480\n"


def test_missing_viscous_pacc_point_is_missing_physics_not_zero_or_nan():
    point = XFOILOperatingPoint(2, 1e6)
    assert _parse_polar(_POLAR_HEADER, point) is None
    accepted = _parse_polar(_POLAR_HEADER + _ROW, point)
    assert accepted.lift == pytest.approx(0.218)
    assert accepted.drag == pytest.approx(0.0058)
    # A neighboring polar's row cannot be reused for an unconverged angle.
    with pytest.raises(ValueError, match="different angle"):
        _parse_polar(_POLAR_HEADER + _ROW, XFOILOperatingPoint(3, 1e6))
    extended = (
        (_POLAR_HEADER + _ROW)
        .replace(
            b"alpha CL CD CDp CM Top_Xtr Bot_Xtr",
            b"alpha CL CD CDp CM Top_Xtr Bot_Xtr Top_Itr Bot_Itr",
        )
        .replace(
            _ROW,
            _ROW.rstrip() + b" 31.2277 130.7723\n",
        )
    )
    extended_result = _parse_polar(extended, point)
    assert extended_result.drag == pytest.approx(0.0058)


def test_polar_conditions_and_nonfinite_values_fail_closed():
    point = XFOILOperatingPoint(2, 1e6)
    with pytest.raises(ValueError, match="conditions"):
        _parse_polar((_POLAR_HEADER + _ROW).replace(b"Re = 1.000", b"Re = 2.000"), point)
    with pytest.raises(ValueError, match="Nonfinite"):
        _parse_polar(_POLAR_HEADER + _ROW.replace(b"0.00580", b"NaN"), point)
    with pytest.raises(ValueError, match="multiple rows"):
        _parse_polar(_POLAR_HEADER + _ROW + _ROW, point)


def test_exact_requested_geometry_cannot_inject_engine_commands():
    points = naca0012_coordinates()
    original = _geometry_bytes(points)
    moved = list(points)
    moved[20] = (moved[20][0], moved[20][1] + 0.001)
    assert original != _geometry_bytes(moved)
    with pytest.raises((TypeError, ValueError)):
        _geometry_bytes([("0\nQUIT\n", 0)] * 20)
    with pytest.raises(ValueError, match="finite"):
        _geometry_bytes([(1.0, float("nan")), *points[1:]])


def test_prepared_case_identity_tracks_design_mesh_and_convergence_policy():
    files, options, design = naca0012_dafoam_case(16, 16)
    _, base = _request(files, options, design, GEOMETRY_SOURCE, True, 8 * 1024**2)
    _, changed = _request(
        files, options, {"patchV": [10, 3]}, GEOMETRY_SOURCE, True, 8 * 1024**2
    )
    assert changed["request_id"] != base["request_id"]
    stricter = {**options, "primalMinResTolDiff": 1.0}
    _, changed = _request(files, stricter, design, GEOMETRY_SOURCE, True, 8 * 1024**2)
    assert changed["request_id"] != base["request_id"]
    with pytest.raises(ValueError, match="relative POSIX"):
        _request(
            {**files, "../escape": b"x"},
            options,
            design,
            GEOMETRY_SOURCE,
            True,
            8 * 1024**2,
        )
    with pytest.raises(ValueError, match="Missing prepared"):
        _request(
            {k: v for k, v in files.items() if k != "constant/polyMesh/points"},
            options,
            design,
            GEOMETRY_SOURCE,
            True,
            8 * 1024**2,
        )
    with pytest.raises(ValueError, match="primalMinResTolDiff"):
        _request(
            files,
            {k: v for k, v in options.items() if k != "primalMinResTolDiff"},
            design,
            GEOMETRY_SOURCE,
            True,
            8 * 1024**2,
        )


def test_generated_public_geometry_mesh_has_closed_positive_volume_cells():
    files, _, _ = naca0012_dafoam_case(16, 16)

    def rows(name):
        text = files["constant/polyMesh/" + name].decode()
        body = text[text.index("}") + 1 :].strip()
        return body.splitlines()[2:-1]

    points = np.array(
        [[float(v) for v in row.strip("()").split()] for row in rows("points")]
    )
    faces = [
        tuple(int(v) for v in re.search(r"\((.*)\)", row)[1].split())
        for row in rows("faces")
    ]
    owner = np.array([int(row) for row in rows("owner")])
    neighbour = np.array([int(row) for row in rows("neighbour")])
    cells = int(max(owner)) + 1
    volume = np.zeros(cells)
    closure = np.zeros((cells, 3))
    counts = np.zeros(cells, dtype=int)
    for i, face in enumerate(faces):
        p = points[list(face)]
        area = 0.5 * (
            np.cross(p[1] - p[0], p[2] - p[0]) + np.cross(p[2] - p[0], p[3] - p[0])
        )
        contribution = np.dot(p.mean(axis=0), area) / 3
        volume[owner[i]] += contribution
        closure[owner[i]] += area
        counts[owner[i]] += 1
        if i < len(neighbour):
            volume[neighbour[i]] -= contribution
            closure[neighbour[i]] -= area
            counts[neighbour[i]] += 1
    assert np.all(volume > 0)
    np.testing.assert_allclose(closure, 0, atol=2e-13)
    assert np.all(counts == 6)


def test_aerodynamic_launches_refuse_argument_free_jax_tracing():
    @jax.jit
    def xfoil_trace():
        run_xfoil_point(None, (), None)
        return 1

    @jax.jit
    def dafoam_trace():
        run_dafoam(None, case_files={}, options={}, design={}, geometry_source="declared")
        return 1

    with pytest.raises(TypeError, match="JAX transformations"):
        xfoil_trace()
    with pytest.raises(TypeError, match="JAX transformations"):
        dafoam_trace()
