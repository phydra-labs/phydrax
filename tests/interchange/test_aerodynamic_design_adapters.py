#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Artifact and convergence boundaries; real external solves live in the benchmark."""

import json
import re
import sys
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from phydrax._external_runtime import pin_energy_executable
from phydrax._fingerprint import canonical_fingerprint
from phydrax.interchange import dafoam
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
        tuple(int(value) for value in re.search(r"\((.*)\)", row)[1].split())
        for row in rows("faces")
    ]
    owner = np.array([int(row) for row in rows("owner")])
    neighbor = np.array([int(row) for row in rows("neighbour")])
    cells = int(max(owner)) + 1
    volume = np.zeros(cells)
    closure = np.zeros((cells, 3))
    counts = np.zeros(cells, dtype="int64")
    for i, face in enumerate(faces):
        p = points[list(face)]
        area = 0.5 * (
            np.cross(p[1] - p[0], p[2] - p[0]) + np.cross(p[2] - p[0], p[3] - p[0])
        )
        contribution = np.dot(p.mean(axis=0), area) / 3
        volume[owner[i]] += contribution
        closure[owner[i]] += area
        counts[owner[i]] += 1
        if i < len(neighbor):
            volume[neighbor[i]] -= contribution
            closure[neighbor[i]] -= area
            counts[neighbor[i]] += 1
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


# A linear external response f = Σ_var A[f, var] · var stands in for the engine
# so the whole host path (request, worker payload, acceptance, staging, adjoint)
# runs without DAFoam.
_SENSITIVITIES = {
    ("CD", "patchV"): np.asarray([0.01, -0.2]),
    ("CD", "shape"): np.arange(6.0).reshape(2, 3),
    ("CL", "patchV"): np.asarray([0.3, 0.05]),
    ("CL", "shape"): -np.ones((2, 3)),
}


def _fake_runtime(tmp_path):
    package = tmp_path / "dafoam"
    (package / "mphys").mkdir(parents=True)
    files = (package / "pyDAFoam.py", package / "mphys" / "mphys_dafoam.py")
    for path in files:
        path.write_text("# pinned\n")
    return dafoam.DAFoamRuntime(
        pin_energy_executable(sys.executable, version="3", license_id="PSF-2.0"),
        str(package),
        tuple((str(path.resolve()), "0" * 64) for path in files),
        "GPL-3.0-or-later",
    )


def _fake_worker(runtime, *, accept):
    def run_energy_command(executable, argv, *, inputs, **options):
        request = json.loads(inputs["aerodynamic-request.json"])
        design = {
            name: np.asarray(values).reshape(request["design_shapes"][name])
            for name, values in request["design"].items()
        }
        functions = {
            name: float(
                sum(np.sum(_SENSITIVITIES[name, var] * design[var]) for var in design)
            )
            for name in ("CL", "CD")
        }
        realization = {
            "state_sha256": "1" * 64,
            "mesh_sha256": "2" * 64,
            "design_sha256": canonical_fingerprint(request["design"]),
        }
        payload = {
            "request_id": request["request_id"],
            "runtime_sha256": canonical_fingerprint(dict(runtime.implementation_files)),
            "functions": functions if accept else {},
            "total_derivatives": [
                {
                    "function": name,
                    "design_variable": var,
                    "values": _SENSITIVITIES[name, var].reshape(-1).tolist(),
                }
                for name in ("CL", "CD")
                for var in ("shape", "patchV")
            ]
            if accept
            else [],
            "adjoints": [
                {
                    "function": name,
                    "accepted": True,
                    **realization,
                    "adjoint_sha256": str(index) * 64,
                    "linear_iterations": 12,
                    "linear_residual_norm": 1e-12,
                    "petsc_converged_reason": 2,
                    "failure_reason": "",
                }
                for index, name in enumerate(("CL", "CD"))
            ]
            if accept
            else [],
            "mesh_accepted": True,
            "state_accepted": accept,
            **realization,
            "failure_reason": "" if accept else "Native primalFail rejected the primal.",
        }
        return SimpleNamespace(
            artifact=SimpleNamespace(artifact_id="3" * 64),
            output=lambda path: json.dumps(payload).encode(),
        )

    return run_energy_command


def _dafoam_action(tmp_path, options=None):
    files, base_options, _ = naca0012_dafoam_case(16, 16)
    base_options = {
        **base_options,
        "inputInfo": {
            **base_options["inputInfo"],
            "shape": {"type": "volCoord", "components": ["solver", "function"]},
        },
    }
    runtime = _fake_runtime(tmp_path)
    action = dafoam.DAFoamAdjointAction(
        runtime,
        case_files=files,
        options={**base_options, **(options or {})},
        design_shapes={"shape": (2, 3), "patchV": (2,)},
        geometry_source=GEOMETRY_SOURCE,
    )
    return runtime, action


def test_dafoam_staged_adjoint_contracts_shape_preserving_totals(tmp_path, monkeypatch):
    runtime, action = _dafoam_action(tmp_path)
    monkeypatch.setattr(dafoam, "run_energy_command", _fake_worker(runtime, accept=True))
    assert [spec.name for spec in action.input_schema] == ["patchV", "shape"]
    assert [spec.name for spec in action.output_schema] == ["CD", "CL"]

    patch = np.asarray([10.0, 2.0])
    shape = np.linspace(0.0, 1.0, 6).reshape(2, 3)
    stage = action.stage_primal(patch, shape)
    assert stage.accepted
    for output, name in zip(stage.outputs, ("CD", "CL"), strict=True):
        expected = np.sum(_SENSITIVITIES[name, "patchV"] * patch) + np.sum(
            _SENSITIVITIES[name, "shape"] * shape
        )
        assert float(output) == pytest.approx(expected)

    weights = (np.asarray(0.5), np.asarray(-2.0))
    patch_bar, shape_bar = action.apply_adjoint(stage, *weights)
    for bar, var in ((patch_bar, "patchV"), (shape_bar, "shape")):
        expected = sum(
            weight * _SENSITIVITIES[name, var]
            for weight, name in zip(weights, ("CD", "CL"), strict=True)
        )
        assert bar.shape == expected.shape
        np.testing.assert_allclose(bar, expected)

    other_runtime, other = _dafoam_action(tmp_path / "other", {"primalMinResTol": 1e-9})
    monkeypatch.setattr(
        dafoam, "run_energy_command", _fake_worker(other_runtime, accept=True)
    )
    with pytest.raises(ValueError, match="Replay mismatch"):
        action.apply_adjoint(other.stage_primal(patch, shape), *weights)


def test_dafoam_failed_primal_stages_evidence_without_an_adjoint(tmp_path, monkeypatch):
    runtime, action = _dafoam_action(tmp_path)
    monkeypatch.setattr(dafoam, "run_energy_command", _fake_worker(runtime, accept=False))
    stage = action.stage_primal(np.asarray([10.0, 2.0]), np.zeros((2, 3)))
    assert not stage.accepted and stage.outputs == ()
    assert "primalFail" in stage.failure_reason
    assert dict(stage.evidence)["state_sha256"] == "1" * 64
    with pytest.raises(ValueError, match="not accepted"):
        action.apply_adjoint(stage, 1.0, 1.0)


def test_dafoam_design_kinds_are_closed():
    files, options, design = naca0012_dafoam_case(16, 16)
    unsupported = {
        **options,
        "inputInfo": {"patchV": {**options["inputInfo"]["patchV"], "type": "shape"}},
    }
    with pytest.raises(ValueError, match="volCoord, patchVelocity, patchVar, field"):
        _request(files, unsupported, design, GEOMETRY_SOURCE, True, 8 * 1024**2)
