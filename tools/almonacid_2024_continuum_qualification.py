#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run native Almonacid muscle/aponeurosis against hashed external raw fields.

This comparison preserves float32 binary output and source nearest-cell-QP
boundary forces. Source ``energy_int`` is P:Grad(u), not stored energy. Passing
this script is numerical source equivalence, not biological or LBB validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from scipy.spatial import cKDTree

from phydrax.applications.skeletal_muscle.continuum import almonacid_2024_repository_case


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_id(record):
    payload = {key: value for key, value in record.items() if key != "content_id"}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def parameter_contract(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.strip().startswith("set "):
            key, value = line.strip()[4:].split("=", 1)
            values[key.strip()] = value.split("#", 1)[0].strip()
    for output_only in (
        "Output binary files main variables",
        "Output binary files tensors",
    ):
        values.pop(output_only)
    return values


def checked_file(directory, record, name):
    path = directory / name
    expected = record["artifacts"][name]
    if (
        path.is_symlink()
        or path.stat().st_size != expected["bytes"]
        or digest(path) != expected["sha256"]
    ):
        raise ValueError(f"External frozen artifact mismatch: {name}")
    return path


def measure(actual, reference, *, rtol, atol):
    actual, reference = np.asarray(actual), np.asarray(reference)
    difference = actual - reference
    finite = np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    scale = np.maximum(np.abs(reference), atol / rtol)
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), atol)
        ),
        "max_scaled_error": float(np.max(np.abs(difference) / scale)),
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "passed": bool(
            finite and np.all(np.abs(difference) <= atol + rtol * np.abs(reference))
        ),
    }


def compare_fields(prepared, dt, main, tensors, indices, rtol):
    fields = prepared.quadrature_fields(dt)
    response = fields["response"]

    def flat(value):
        array = np.asarray(value)
        return array.reshape((-1,) + array.shape[2:])[indices]

    def compare(value, reference, atol):
        return measure(flat(value), reference, rtol=rtol, atol=atol)

    # float32 storage and source Newton tolerances are explicit, not trace-level f64 criteria.
    comparisons = {
        "displacement_m": compare(fields["displacement_m"], main[:, 5:8], 2e-6),
        "velocity_m_per_s": compare(fields["velocity_m_per_s"], main[:, 8:11], 2e-4),
        "pressure_Pa": compare(fields["pressure_Pa"], main[:, 11], 20.0),
        "dilation": compare(fields["dilation"], main[:, 12], 2e-5),
        "volume_ratio": compare(response.volume_ratio, main[:, 4], 2e-5),
        "fiber_stretch": compare(response.fiber_stretch, main[:, 13], 2e-5),
        "isochoric_fiber_stretch": compare(
            response.isochoric_fiber_stretch, main[:, 14], 2e-5
        ),
        "normalized_total_strain_rate": compare(
            fields["normalized_total_strain_rate"], main[:, 15], 2e-4
        ),
        "normalized_isochoric_strain_rate": compare(
            response.normalized_strain_rate, main[:, 16], 2e-4
        ),
        "orientation": compare(fields["orientation"], main[:, 17:20], 2e-5),
        "deformation_gradient": compare(
            fields["deformation_gradient"], tensors[:, 5:14].reshape((-1, 3, 3)), 2e-5
        ),
        "kirchhoff_Pa": compare(
            response.kirchhoff_Pa, tensors[:, 14:23].reshape((-1, 3, 3)), 20.0
        ),
        "kirchhoff_active_Pa": compare(
            response.kirchhoff_active_Pa, tensors[:, 41:50].reshape((-1, 3, 3)), 20.0
        ),
        "kirchhoff_passive_fiber_Pa": compare(
            response.kirchhoff_passive_fiber_Pa,
            tensors[:, 50:59].reshape((-1, 3, 3)),
            20.0,
        ),
        "kirchhoff_base_Pa": compare(
            response.kirchhoff_base_Pa, tensors[:, 59:68].reshape((-1, 3, 3)), 20.0
        ),
    }
    return comparisons, fields


def qualify(directory, inputs, *, maximum_step=100, rtol=2e-3, save_native=None):
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("relative tolerance must be finite and positive.")
    record = json.loads((directory / "frozen-reference.json").read_text())
    if record["status"] != "frozen-executable-output" or record["case"] not in (
        "repository-default-fields",
        "fixed-end-activation",
        "passive-cyclic",
    ):
        raise ValueError("A completed supported external campaign is required.")
    inventory = json.loads((inputs / "reference_outputs/manifest.json").read_text())
    expected = [
        package["source_record_content_id"]
        for package in inventory["packages"]
        if package["case"] == record["case"]
    ]
    if (
        len(expected) != 1
        or record.get("content_id") != expected[0]
        or content_id(record) != expected[0]
    ):
        raise ValueError(
            "External campaign record identity is not frozen by the fixture."
        )
    source_inputs = {
        name: checked_file(directory, record, name)
        for name in (
            "parameters.prm",
            "control_points_activation.dat",
            "control_points_strain.dat",
        )
    }
    plan, parameters, history, dt, end = almonacid_2024_repository_case(
        inputs, protocol=record["case"]
    )
    if parameter_contract(inputs / "parameters.prm") != parameter_contract(
        source_inputs["parameters.prm"]
    ):
        raise ValueError("Native parameters differ from the executed source contract.")
    source_activation = np.loadtxt(source_inputs["control_points_activation.dat"])
    source_strain = np.loadtxt(source_inputs["control_points_strain.dat"])
    native_activation = np.column_stack((history.activation_time_s, history.activation))
    native_strain = np.column_stack((history.strain_time_s, history.engineering_strain))
    if not np.array_equal(native_activation, source_activation) or not np.array_equal(
        native_strain, source_strain
    ):
        raise ValueError("Native controls differ from the executed source input history.")
    if float(parameters.muscle.fat_fraction) != 0:
        raise ValueError(
            "Component-column comparison requires the pinned zero-fat source case."
        )
    prepared = plan.prepare(parameters)
    propose = eqx.filter_jit(lambda model, control: model.propose(control))
    main0 = np.fromfile(
        checked_file(directory, record, "outputs/cell_data_main-3d-000.data"), dtype="<f4"
    ).reshape((-1, 21))
    points = np.asarray(prepared.geometry.points_m).reshape((-1, 3))
    distance, indices = cKDTree(points).query(main0[:, :3])
    if (
        len(points) != len(main0)
        or len(np.unique(indices)) != len(indices)
        or np.max(distance) > 4e-8
    ):
        raise ValueError("Native/reference quadrature coordinates are not a bijection.")
    tissues = np.broadcast_to(
        np.asarray(prepared.geometry.tissue_ids)[:, None],
        prepared.geometry.weights_m3.shape,
    ).reshape(-1)
    if not np.array_equal(tissues[indices], main0[:, 20]):
        raise ValueError("Reference/native tissue dispatch differs.")
    energy_history = np.loadtxt(
        checked_file(directory, record, "outputs/energy_data-3d.csv"),
        delimiter=",",
        skiprows=1,
    )
    rows = []
    started = time.perf_counter()
    expected_steps = int(round((end - 1e-8) / dt))
    if maximum_step < 0 or maximum_step > expected_steps:
        raise ValueError("maximum_step is outside the pinned campaign.")
    for step in range(maximum_step + 1):
        print(
            json.dumps({"case": record["case"], "progress_step": step}),
            flush=True,
        )
        diagnostics = None
        if step:
            candidate = propose(prepared, history.sample(step * dt))
            jax.block_until_ready(candidate)
            if not bool(candidate.successful):
                failure = candidate.nonlinear_result.diagnostics
                rows.append(
                    {
                        "step": step,
                        "time_s": step * dt,
                        "native_solve_successful": False,
                        "nonlinear_status": int(candidate.nonlinear_result.status),
                        "initial_residual_norm": float(failure.initial_residual_norm),
                        "final_residual_norm": float(failure.final_residual_norm),
                        "nonlinear_iterations": int(failure.iterations),
                        "linear_iterations": int(failure.linear_iterations),
                        "final_linear_status": int(failure.final_linear_status),
                    }
                )
                break
            diagnostics = candidate.diagnostics
            prepared = candidate.commit(prepared)
        raw_main = np.fromfile(
            checked_file(directory, record, f"outputs/cell_data_main-3d-{step:03d}.data"),
            dtype="<f4",
        ).reshape((-1, 21))
        raw_tensor = np.fromfile(
            checked_file(
                directory, record, f"outputs/cell_data_tensors-3d-{step:03d}.data"
            ),
            dtype="<f4",
        ).reshape((-1, 69))
        if not np.array_equal(raw_main[:, :4], main0[:, :4]) or not np.array_equal(
            raw_tensor[:, :4], main0[:, :4]
        ):
            raise ValueError("External quadrature layout changed during the campaign.")
        comparisons, fields = compare_fields(
            prepared, dt, raw_main, raw_tensor, indices, rtol
        )
        row = {
            "step": step,
            "time_s": step * dt,
            "native_solve_successful": True,
            "fields": comparisons,
        }
        response = fields["response"]
        weights = np.asarray(fields["weights_m3"])
        volume = float(np.sum(np.asarray(response.volume_ratio) * weights))
        source_internal = float(
            np.sum(
                np.asarray(response.first_piola_Pa)
                * (np.asarray(fields["deformation_gradient"]) - np.eye(3))
                * weights[..., None, None]
            )
        )
        source_kinetic = float(
            0.5
            * np.asarray(parameters.density_kg_per_m3)
            * np.sum(np.asarray(fields["velocity_m_per_s"]) ** 2 * weights[..., None])
        )
        row["source_energy_density"] = measure(
            np.array((source_kinetic, source_internal)) / volume,
            energy_history[step, 1:3],
            rtol=rtol,
            atol=0.1,
        )
        row["source_volume_m3"] = measure(
            volume, energy_history[step, -1], rtol=rtol, atol=1e-8
        )
        traces = prepared.geometry.traces
        P = np.asarray(response.first_piola_Pa)[
            np.asarray(traces.cells)[:, None], np.asarray(traces.nearest_volume_qp)
        ]
        traction = np.sum(P * np.asarray(traces.normals)[..., None, :], axis=-1)
        face_force = np.sum(traction * np.asarray(traces.weights_m2)[..., None], axis=1)
        source_force = np.zeros((8, 3))
        boundaries = np.asarray(traces.boundary_ids)
        np.add.at(source_force, boundaries[boundaries >= 0], face_force[boundaries >= 0])
        force_file = checked_file(
            directory, record, f"outputs/force_data-3d-{step:03d}.csv"
        )
        force = np.loadtxt(force_file, delimiter=",", skiprows=1)
        row["source_boundary_force_N"] = measure(
            source_force[force[:, 0].astype(int)], force[:, 1:4], rtol=rtol, atol=0.02
        )
        if diagnostics is not None:
            row["diagnostics"] = {
                "free_force_residual_N": float(diagnostics.free_force_residual_N),
                "interface_displacement_jump_l2_m": float(
                    diagnostics.interface_displacement_jump_l2_m
                ),
                "interface_traction_jump_l2_Pa": float(
                    diagnostics.interface_traction_jump_l2_Pa
                ),
                "interface_power_defect_W": float(diagnostics.interface_power_defect_W),
                "pressure_weak_residual_m3": float(
                    diagnostics.mixed_space.pressure_weak_residual_m3
                ),
                "dilation_weak_residual_J": float(
                    diagnostics.mixed_space.dilation_weak_residual_J
                ),
                "pressure_jump_l2_Pa": float(diagnostics.mixed_space.pressure_jump_l2_Pa),
                "work_energy_residual_J": float(diagnostics.work_energy_residual_J),
                "source_stress_displacement_contraction_J": float(
                    diagnostics.source_stress_displacement_contraction_J
                ),
            }
        rows.append(row)
        if save_native is not None:
            save_native.mkdir(parents=True, exist_ok=True)
            state_arrays = {
                "displacement_m": np.asarray(prepared.state.displacement_m),
                "velocity_m_per_s": np.asarray(prepared.state.velocity_m_per_s),
                "pressure_coefficients_Pa": np.asarray(
                    prepared.state.pressure_coefficients_Pa
                ),
                "dilation_coefficients": np.asarray(prepared.state.dilation_coefficients),
                "deformation_gradient": np.asarray(prepared.state.deformation_gradient),
                "reference_velocity_gradient_per_s": np.asarray(
                    prepared.state.reference_velocity_gradient_per_s
                ),
            }
            np.savez(save_native / f"native-{step:03d}.npz", **state_arrays)
    all_passed = len(rows) == maximum_step + 1 and all(
        row["native_solve_successful"]
        and all(value["passed"] for value in row.get("fields", {}).values())
        and all(
            row.get(key, {"passed": False})["passed"]
            for key in (
                "source_boundary_force_N",
                "source_energy_density",
                "source_volume_m3",
            )
        )
        for row in rows
    )
    return {
        "source_commit": "0698e3d87d7261c81437d410dda161fe3b8efcbf",
        "case": record["case"],
        "reference_record_sha256": digest(directory / "frozen-reference.json"),
        "input_sha256": {
            name: digest(inputs / name)
            for name in (
                "parameters.prm",
                "control_points_activation.dat",
                "control_points_strain.dat",
            )
        },
        "plan_id": plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "numeric_revision": prepared.numeric_revision().revision_id,
        "source_equivalence_passed": all_passed,
        "full_101_sample_campaign": maximum_step == 100 and len(rows) == 101,
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_qp_coordinate_error_m": float(np.max(distance)),
        "source_binary_dtype": "float32-little-endian",
        "native_dtype": "float64",
        "inf_sup_qualified": False,
        "biological_validation": False,
        "full_mta_admitted": False,
        "limitations": [
            "Numerical source comparison only; not biological validation.",
            "No LBB lower-bound certificate or mesh/time refinement conclusion is implied.",
            "Source force uses nearest cell-QP stress; variational reactions are separate.",
            "Source internal energy is endpoint P:Grad(u), not stored energy.",
        ],
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-directory",
        type=Path,
        default=ROOT
        / ".tmp/skeletal-production-sources/flexodeal-reference-default-fields",
    )
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "tests/fixtures/flexodeal_0698e3d"
    )
    parser.add_argument("--maximum-step", type=int, default=100)
    parser.add_argument("--relative-tolerance", type=float, default=2e-3)
    parser.add_argument("--save-native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    result = qualify(
        args.reference_directory,
        args.inputs,
        maximum_step=args.maximum_step,
        rtol=args.relative_tolerance,
        save_native=args.save_native,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "samples"},
            indent=2,
            sort_keys=True,
        )
    )
    if not result["source_equivalence_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
