"""Deterministic 3D Laplace DP0 Galerkin capacitance benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


_TETRA_FACES = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
_TETRA_VERTICES = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)

_SPHERE_FACE_LADDER = (20, 80, 320, 1280)
_SPHERE_SUBDIVISIONS = {
    face_count: subdivision for subdivision, face_count in enumerate(_SPHERE_FACE_LADDER)
}
_SPHERE_ERROR_LIMITS = {
    20: 0.20,
    80: 0.10,
    320: 0.05,
    1280: 0.025,
}


def _sphere_face_count(case: str) -> int | None:
    if case == "icosphere":
        return 20
    prefix = "icosphere-"
    if not case.startswith(prefix):
        return None
    suffix = case.removeprefix(prefix)
    if not suffix.isdecimal():
        raise ValueError(f"[geometry] Unsupported sphere refinement {case!r}.")
    face_count = int(suffix)
    if face_count not in _SPHERE_SUBDIVISIONS:
        raise ValueError(
            f"[geometry] Unsupported sphere refinement face count {face_count}."
        )
    return face_count


def _region(case: str):
    if case == "tetrahedron":
        return phx.geometry.MeshRegion(_TETRA_VERTICES, _TETRA_FACES)
    if case == "two-tetrahedra":
        vertices = np.concatenate(
            (_TETRA_VERTICES, _TETRA_VERTICES + np.asarray([3.0, 0.0, 0.0]))
        )
        faces = np.concatenate((_TETRA_FACES, _TETRA_FACES + 4))
        return phx.geometry.MeshRegion(vertices, faces)
    sphere_faces = _sphere_face_count(case)
    if sphere_faces is not None:
        import trimesh

        mesh = trimesh.creation.icosphere(
            subdivisions=_SPHERE_SUBDIVISIONS[sphere_faces],
            radius=1.0,
        )
        if int(mesh.faces.shape[0]) != sphere_faces:
            raise ValueError(
                "[geometry] Deterministic icosphere refinement produced an "
                "unexpected face count."
            )
        return phx.geometry.MeshRegion(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    raise ValueError(f"[geometry] Unknown benchmark case {case!r}.")


def _selections(prepared, case: str):
    if case != "two-tetrahedra":
        return {
            "body": phx.discretization.EntitySelection(
                prepared.surface_entities,
                jnp.ones((prepared.face_count,), dtype=bool),
            )
        }
    return {
        "left": phx.discretization.EntitySelection(
            prepared.surface_entities,
            jnp.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        ),
        "right": phx.discretization.EntitySelection(
            prepared.surface_entities,
            jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
        ),
    }


def _case(case: str):
    region = _region(case)
    face_count = int(region.faces.shape[0])
    policy = phx.operators.LaplaceSingleLayerDP0GalerkinPolicy3D(
        singular_order=3,
        near_ratio=1.0,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        target_block_size=min(8, face_count),
        source_block_size=min(8, face_count),
        dense_oracle=phx.linalg.MaterializationPolicy(
            max_entries=max(face_count * face_count, 1),
            max_bytes=max(face_count * face_count * 8, 8),
        ),
    )
    started = time.perf_counter()
    prepared = phx.operators.prepare_laplace_single_layer_dp0_3d(region, policy=policy)
    preparation_seconds = time.perf_counter() - started

    vector = jnp.linspace(0.5, 1.5, face_count)
    started = time.perf_counter()
    action = prepared.strong_operator.mv(vector)
    jax.block_until_ready(action)
    action_seconds = time.perf_counter() - started
    dense_error = float(jnp.max(jnp.abs(action - prepared.dense_oracle.matrix @ vector)))

    started = time.perf_counter()
    result = phx.solver.solve_laplace_capacitance_3d(
        prepared,
        _selections(prepared, case),
    )
    jax.block_until_ready(result.capacitance)
    solve_seconds = time.perf_counter() - started
    statuses = [int(linear.status) for linear in result.linear_results]
    residuals = [
        float(linear.diagnostics.relative_residual) for linear in result.linear_results
    ]
    sphere_faces = _sphere_face_count(case)
    sphere_relative_error = (
        abs(float(result.capacitance[0, 0]) - 4.0 * np.pi) / (4.0 * np.pi)
        if sphere_faces is not None
        else None
    )
    failure_codes = []
    if not bool(result.valid):
        failure_codes.append("linear-solve")
    if dense_error > 1.0e-10:
        failure_codes.append("dense-action")
    if not bool(jnp.all(jnp.isfinite(result.capacitance))):
        failure_codes.append("nonfinite-capacitance")
    if (
        sphere_relative_error is not None
        and sphere_relative_error > _SPHERE_ERROR_LIMITS[sphere_faces]
    ):
        failure_codes.append("sphere-capacitance")
    passed = not failure_codes
    return {
        "case": case,
        "faces": face_count,
        "components": prepared.component_count,
        "conductors": len(result.conductor_names),
        "pair_counts": list(prepared.assembly_report.pair_counts),
        "pair_class_names": list(prepared.assembly_report.pair_class_names),
        "pair_class_maximum_errors": np.asarray(
            prepared.assembly_report.maximum_errors
        ).tolist(),
        "pair_class_tolerances": np.asarray(
            prepared.assembly_report.pair_class_tolerances
        ).tolist(),
        "pair_class_supported": np.asarray(
            prepared.assembly_report.pair_class_supported
        ).tolist(),
        "pair_class_evaluations": np.asarray(
            prepared.assembly_report.evaluations
        ).tolist(),
        "pair_class_workspace_bytes": list(
            prepared.assembly_report.pair_class_workspace_bytes
        ),
        "pair_class_resident_bytes": list(
            prepared.assembly_report.pair_class_resident_bytes
        ),
        "exception_count": prepared.assembly_report.exception_count,
        "resident_bytes": prepared.assembly_report.resident_bytes,
        "preparation_workspace_bytes": (
            prepared.assembly_report.preparation_workspace_bytes
        ),
        "action_workspace_bytes_per_rhs": (
            prepared.assembly_report.action_workspace_bytes_per_rhs
        ),
        "dense_oracle_bytes": prepared.assembly_report.dense_oracle_bytes,
        "dense_action_error": dense_error,
        "linear_statuses": statuses,
        "relative_residuals": residuals,
        "capacitance": np.asarray(result.capacitance).tolist(),
        "reciprocity_defect": float(result.capacitance_reciprocity_defect),
        "sphere_relative_capacitance_error": sphere_relative_error,
        "sphere_error_limit": (
            None if sphere_faces is None else _SPHERE_ERROR_LIMITS[sphere_faces]
        ),
        "failure_codes": failure_codes,
        "preparation_seconds": preparation_seconds,
        "action_seconds": action_seconds,
        "solve_seconds": solve_seconds,
        "passed": passed,
    }


def _sphere_refinement_evidence(records):
    sphere_records = [
        record for record in records if _sphere_face_count(record["case"]) is not None
    ]
    recorded_faces = [int(record["faces"]) for record in sphere_records]
    errors = [
        float(record["sphere_relative_capacitance_error"]) for record in sphere_records
    ]
    complete = tuple(recorded_faces) == _SPHERE_FACE_LADDER
    decreasing = complete and all(
        finer < coarser for coarser, finer in zip(errors, errors[1:])
    )
    failure_codes = []
    if not complete:
        failure_codes.append("sphere-ladder-incomplete")
    if complete and not decreasing:
        failure_codes.append("sphere-convergence")
    if any(
        error > _SPHERE_ERROR_LIMITS[faces]
        for faces, error in zip(recorded_faces, errors, strict=True)
    ):
        failure_codes.append("sphere-capacitance")
    return {
        "face_ladder": list(_SPHERE_FACE_LADDER),
        "recorded_faces": recorded_faces,
        "relative_capacitance_errors": errors,
        "strictly_decreasing": decreasing if complete else None,
        "complete": complete,
        "failure_codes": failure_codes,
        "passed": complete and not failure_codes,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 3D Laplace DP0 Galerkin capacitance."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/laplace_capacitance.json"),
    )
    return parser


def main():
    arguments = _parser().parse_args()
    cases = ["tetrahedron", "two-tetrahedra"]
    if not arguments.smoke:
        cases.extend(
            "icosphere" if faces == 20 else f"icosphere-{faces}"
            for faces in _SPHERE_FACE_LADDER
        )
    records = [_case(case) for case in cases]
    sphere_refinement = _sphere_refinement_evidence(records)
    payload = {
        "benchmark": "laplace-capacitance-3d",
        "backend": jax.default_backend(),
        "records": records,
        "sphere_refinement": sphere_refinement,
        "passed": all(record["passed"] for record in records)
        and (arguments.smoke or sphere_refinement["passed"]),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
