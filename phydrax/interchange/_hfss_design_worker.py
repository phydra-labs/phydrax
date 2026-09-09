#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Standalone Windows worker for pinned QDesignOptimizer/HFSS/Q3D/pyEPR.

Only the selected external interpreter imports this file. It has no phydrax or
JAX dependency and emits the complete ``result.json`` qualified by
``hfss_design``. Raw adaptive convergence and mesh-statistics exports remain
separate detached evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tomllib
from functools import partial
from pathlib import Path


_MAX_DEPENDENCY_FILES = 100000
_MAX_DEPENDENCY_BYTES = 2 * 1024 * 1024 * 1024


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _number(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    number = float(value)
    return number if math.isfinite(number) else None


def _real(value, label):
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise RuntimeError(f"Nonfinite {label} returned by the solver.")
    if abs(number.imag) > 1e-9 * max(1.0, abs(number.real)):
        raise RuntimeError(f"Materially complex {label} returned by the solver.")
    return float(number.real)


def _table(frame):
    return {
        "index": [str(value).strip() for value in frame.index],
        "columns": [str(value).strip() for value in frame.columns],
        "data": [[_number(value) for value in row] for row in frame.to_numpy()],
    }


def _evidence(setup, name):
    frame, raw = setup.get_convergence()
    if frame is None or not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Solver did not export adaptive convergence evidence.")
    convergence_path = Path(name + "-convergence.txt")
    mesh_path = Path(name + "-mesh.txt")
    convergence_path.write_text(raw, encoding="utf-8")
    setup.parent._design.ExportMeshStats(setup.name, "", str(mesh_path.resolve()), True)
    if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
        raise RuntimeError("Solver did not export nonempty mesh statistics.")
    return _table(frame)


def _distribution_evidence(name, expected_version):
    distribution = importlib.metadata.distribution(name)
    if distribution.version != expected_version:
        raise RuntimeError(
            f"Pinned dependency mismatch for {name}: "
            f"expected {expected_version}, observed {distribution.version}."
        )
    metadata = distribution.read_text("METADATA")
    record = distribution.read_text("RECORD")
    if not metadata or not record:
        raise RuntimeError(f"Dependency {name} has no auditable wheel metadata/RECORD.")
    files = distribution.files
    if files is None or len(files) > _MAX_DEPENDENCY_FILES:
        raise RuntimeError(f"Dependency {name} has an unbounded installation manifest.")
    identities = []
    total_bytes = 0
    for relative in sorted(files, key=str):
        path = Path(str(distribution.locate_file(relative)))
        if not path.is_file():
            raise RuntimeError(f"Dependency file is absent: {name}:{relative}")
        size = path.stat().st_size
        total_bytes += size
        if total_bytes > _MAX_DEPENDENCY_BYTES:
            raise RuntimeError(f"Dependency {name} exceeds the provenance byte bound.")
        identities.append((str(relative).replace("\\", "/"), size, _digest(path)))
    direct_url = distribution.read_text("direct_url.json")
    return {
        "version": distribution.version,
        "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
        "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        "direct_url_sha256": (
            hashlib.sha256(direct_url.encode()).hexdigest() if direct_url else None
        ),
        "content_sha256": _json_identity(identities),
        "file_count": len(identities),
        "total_bytes": total_bytes,
    }


def _verify_sources(request):
    identities = request["source_files_sha256"]
    for name, expected in identities.items():
        if _digest(Path("source") / name) != expected:
            raise RuntimeError(f"Detached source identity mismatch: {name}")
    if _json_identity(identities) != request["source_tree_identity"]:
        raise RuntimeError("Detached source tree identity mismatch.")
    lock = tomllib.loads(Path("source/poetry.lock").read_text(encoding="utf-8"))
    locked = {package["name"].lower(): package["version"] for package in lock["package"]}
    if any(
        locked.get(name) != version
        for name, version in request["dependency_profile"].items()
    ):
        raise RuntimeError(
            "Requested dependency profile disagrees with the pinned Poetry lock."
        )


def _ansys_processes(psutil):
    return [
        process
        for process in psutil.process_iter(["name"])
        if process.info["name"] and process.info["name"].lower() == "ansysedt.exe"
    ]


def main():
    if sys.platform != "win32":
        raise RuntimeError(
            "The pinned QDesignOptimizer profile requires Windows AEDT COM."
        )
    request = json.loads(Path("request.json").read_text(encoding="utf-8"))
    _verify_sources(request)
    python = request["python"]
    if _digest(sys.executable) != python["sha256"]:
        raise RuntimeError("Running Python executable no longer matches its pin.")
    observed_python = platform.python_version()
    if observed_python != python["version"] or sys.version_info[:2] not in (
        (3, 11),
        (3, 12),
    ):
        raise RuntimeError(
            f"Unsupported Python release: expected {python['version']}, observed {observed_python}."
        )
    packages = {
        name: _distribution_evidence(name, version)
        for name, version in request["dependency_profile"].items()
    }

    import psutil
    import win32api
    import win32con
    import win32job

    engine = request["aedt"]
    if _digest(engine["path"]) != engine["sha256"]:
        raise RuntimeError("AEDT executable no longer matches its pin.")
    # pyEPR COM can attach to an existing desktop. Refuse such a session rather
    # than changing a user's project or claiming process ownership we do not have.
    if _ansys_processes(psutil):
        raise RuntimeError(
            "An exclusive AEDT host is required: close all existing desktops."
        )

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(Path("source/src").resolve()))
    tutorial = Path("source/tutorials/examples_coupled_transmon_chip").resolve()
    sys.path.insert(0, str(tutorial))
    import design as tutorial_design
    import mini_studies
    import parameter_targets
    import qdesignoptimizer
    from qdesignoptimizer.design_analysis import DesignAnalysis
    from qdesignoptimizer.design_analysis_types import DesignAnalysisState, MeshingMap
    from qdesignoptimizer.utils.chip_generation import create_chip_base
    from qiskit_metal.analyses.quantization import LOManalysis

    if qdesignoptimizer.__version__ != "0.2.0":
        raise RuntimeError(
            "Detached QDesignOptimizer package version disagrees with its pin."
        )
    profile = request["profile"]
    design, gui = create_chip_base(
        "phydrax-transmon",
        tutorial_design.chip_type,
        open_gui=True,
        design_variables_file=str(tutorial / "design_variables.json"),
    )
    unknown = set(request["design_variables"]) - set(design.variables)
    if unknown:
        raise ValueError(f"Unknown upstream geometry variables: {sorted(unknown)}")
    design.variables.update(request["design_variables"])
    render = partial(tutorial_design.render_qiskit_metal_design, gui=gui)
    study = mini_studies.get_mini_study_qb_res(profile["group"])
    requested_modes = {window[0] for window in profile["mode_windows_hz"]}
    if set(study.modes) != requested_modes:
        raise RuntimeError(
            "Requested mode labels disagree with the pinned upstream mini-study."
        )
    study.nbr_passes = profile["max_passes"]
    study.delta_f = profile["frequency_tolerance_percent"]
    target_params = dict(parameter_targets.PARAM_TARGETS)
    state = DesignAnalysisState(design, render, target_params)

    job = win32job.CreateJobObject(None, None)
    information = win32job.QueryInformationJobObject(
        job, win32job.JobObjectExtendedLimitInformation
    )
    information["BasicLimitInformation"]["LimitFlags"] |= (
        win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    win32job.SetInformationJobObject(
        job, win32job.JobObjectExtendedLimitInformation, information
    )
    desktop_process = subprocess.Popen([engine["path"], "-NoSplash"])
    analysis = None
    lom = None
    try:
        process_handle = win32api.OpenProcess(
            win32con.PROCESS_ALL_ACCESS, False, desktop_process.pid
        )
        win32job.AssignProcessToJobObject(job, process_handle)
        win32api.CloseHandle(process_handle)
        analysis = DesignAnalysis(
            state,
            study,
            save_path="analysis",
            update_design_variables=False,
            meshing_map=[
                MeshingMap(
                    tutorial_design.CoupledLineTee,
                    tutorial_design.CoupledLineTee_mesh_names,
                )
            ],
        )
        owned = _ansys_processes(psutil)
        if (
            len(owned) != 1
            or owned[0].pid != desktop_process.pid
            or _digest(owned[0].exe()) != engine["sha256"]
        ):
            raise RuntimeError(
                "COM opened a different AEDT executable or multiple desktops."
            )
        desktop = analysis.pinfo.project.parent
        observed_version = desktop.get_version()
        if observed_version != engine["version"]:
            raise RuntimeError(f"AEDT release mismatch: {observed_version!r}")

        eigenmodes = analysis.run_eigenmodes()
        eigen_convergence = _evidence(analysis.setup, "eigenmode")
        chis, participation = analysis.run_epr()
        if chis is None or participation is None:
            raise RuntimeError("The selected mini-study did not produce EPR results.")
        epr_frequencies = analysis.epra.get_frequencies(numeric=True)
        modes = [mode for mode, _ in analysis.get_simulated_modes_sorted()]
        junctions = list(analysis.jj_setups_to_include_in_epr)

        # This is the exact implementation of the pinned upstream
        # CapacitanceMatrixStudy.simulate_capacitance_matrix method, kept here so
        # its live Q3D setup remains available for convergence/mesh export.
        cap_study = mini_studies.get_mini_study_resonator_capacitance(
            profile["group"]
        ).capacitance_matrix_studies[0]
        cap_study.nbr_passes = profile["max_passes"]
        cap_study.percent_error = profile["capacitance_tolerance_percent"]
        render(design, **cap_study.render_qiskit_metal_kwargs)
        lom = LOManalysis(design, "q3d")
        lom.sim.setup.max_passes = cap_study.nbr_passes
        lom.sim.setup.percent_error = cap_study.percent_error
        lom.sim.renderer.options["x_buffer_width_mm"] = cap_study.x_buffer_width_mm
        lom.sim.renderer.options["y_buffer_width_mm"] = cap_study.y_buffer_width_mm
        lom.sim.run(
            components=cap_study.qiskit_component_names,
            open_terminations=cap_study.open_pins,
        )
        capacitance = _table(lom.sim.capacitance_matrix)
        cap_convergence = _evidence(lom.sim.renderer.pinfo.setup, "capacitance")
        owned = _ansys_processes(psutil)
        if len(owned) != 1 or owned[0].pid != desktop_process.pid:
            raise RuntimeError("Q3D did not remain in the owned AEDT desktop.")

        output_names = (
            "eigenmode-convergence.txt",
            "eigenmode-mesh.txt",
            "capacitance-convergence.txt",
            "capacitance-mesh.txt",
        )
        exports = {name: _digest(name) for name in output_names}
        exports["request.json"] = _digest("request.json")
        payload = {
            "source_commit": request["source_commit"],
            "source_license_id": request["source_license_id"],
            "source_repository_url": request["source_repository_url"],
            "source_files_sha256": request["source_files_sha256"],
            "source_tree_identity": request["source_tree_identity"],
            "dependency_lock_sha256": request["source_files_sha256"]["poetry.lock"],
            "worker_sha256": _digest("worker.py"),
            "profile": profile,
            "python_version": observed_python,
            "python_sha256": python["sha256"],
            "python_license_id": python["license_id"],
            "python_source_url": python["source_url"],
            "engine_version": observed_version,
            "engine_sha256": engine["sha256"],
            "engine_license_id": engine["license_id"],
            "engine_source_url": engine["source_url"],
            "packages": packages,
            "qdesignoptimizer_version": qdesignoptimizer.__version__,
            "mode_identity_scope": "upstream-target-frequency-rank-with-disjoint-owner-windows",
            "mode_labels": modes,
            "junction_labels": junctions,
            "eigenmode_frequency_hz": [
                float(value) for value in eigenmodes["Freq. (Hz)"]
            ],
            "eigenmode_kappa_hz": [float(value) for value in eigenmodes["Kappas (Hz)"]],
            "epr_frequency_hz": [
                _real(value, "EPR frequency") * 1e6
                for value in epr_frequencies.iloc[:, 0]
            ],
            "chi_hz": [
                [_real(value, "EPR Kerr coefficient") * 1e6 for value in row]
                for row in chis.to_numpy()
            ],
            "participation": [
                [_real(value, "EPR participation ratio") for value in row]
                for row in participation.to_numpy()
            ],
            "capacitance_ff": capacitance,
            "eigenmode_convergence": eigen_convergence,
            "capacitance_convergence": cap_convergence,
            "mesh_identity_scope": "exported-statistics-and-adaptive-history-not-connectivity",
            "mesh_exports": exports,
            "mesh_identity": _json_identity(exports),
            "design_variables": dict(design.variables),
        }
        Path("result.json").write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True), encoding="utf-8"
        )
    finally:
        try:
            if lom is not None:
                lom.sim.renderer.stop()
            if analysis is not None:
                analysis.renderer.stop()
            gui.main_window.close()
        finally:
            win32api.CloseHandle(job)
            desktop_process.wait(timeout=30)
    if _digest(engine["path"]) != engine["sha256"]:
        raise RuntimeError("AEDT executable changed during the solve.")


if __name__ == "__main__":
    main()
