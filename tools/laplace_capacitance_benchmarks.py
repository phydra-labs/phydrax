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


def _region(case: str):
    if case == "tetrahedron":
        return phx.geometry.MeshRegion(_TETRA_VERTICES, _TETRA_FACES)
    if case == "two-tetrahedra":
        vertices = np.concatenate(
            (_TETRA_VERTICES, _TETRA_VERTICES + np.asarray([3.0, 0.0, 0.0]))
        )
        faces = np.concatenate((_TETRA_FACES, _TETRA_FACES + 4))
        return phx.geometry.MeshRegion(vertices, faces)
    if case == "icosphere":
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
        return phx.geometry.MeshRegion(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    raise ValueError(f"Unknown benchmark case {case!r}.")


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
    sphere_relative_error = (
        abs(float(result.capacitance[0, 0]) - 4.0 * np.pi) / (4.0 * np.pi)
        if case == "icosphere"
        else None
    )
    passed = bool(
        result.valid
        & (dense_error <= 1.0e-10)
        & jnp.all(jnp.isfinite(result.capacitance))
        & (True if sphere_relative_error is None else sphere_relative_error <= 0.2)
    )
    return {
        "case": case,
        "faces": face_count,
        "components": prepared.component_count,
        "conductors": len(result.conductor_names),
        "pair_counts": list(prepared.assembly_report.pair_counts),
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
        "preparation_seconds": preparation_seconds,
        "action_seconds": action_seconds,
        "solve_seconds": solve_seconds,
        "passed": passed,
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
        cases.append("icosphere")
    records = [_case(case) for case in cases]
    payload = {
        "benchmark": "laplace-capacitance-3d",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "records": records,
        "passed": all(record["passed"] for record in records),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
