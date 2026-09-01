"""Deterministic bounded zero-speed potential-flow hydrodynamics benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import trimesh

from phydrax.geometry import MeshRegion
from phydrax.operators.integral.layer_potential._free_surface_green3d import (
    FreeSurfaceGreenPolicy3D,
)
from phydrax.operators.integral.layer_potential._free_surface_hydrodynamics3d import (
    FreeSurfaceHydrodynamicsPolicy3D,
    prepare_free_surface_hydrodynamics_3d,
)
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
)
from phydrax.solver._potential_flow_hydrodynamics import (
    solve_potential_flow_hydrodynamics_3d,
)


def _policy(face_count: int, *, smoke: bool):
    radial = 8 if smoke else 16
    angular = 8 if smoke else 16
    return FreeSurfaceHydrodynamicsPolicy3D(
        green=FreeSurfaceGreenPolicy3D(
            radial_order_per_interval=radial,
            angular_order=angular,
            cutoff_clearance_factor=4.0,
            minimum_cutoff_root_ratio=2.0,
            maximum_wavenumber=200.0,
        ),
        galerkin=LaplaceSingleLayerDP0GalerkinPolicy3D(
            regular_order=3,
            singular_order=3,
            near_order=3,
            near_ratio=1.5,
            near_max_depth=1,
            absolute_tolerance=1.0e-2,
            relative_tolerance=5.0e-2,
            target_block_size=min(16, face_count),
            source_block_size=min(16, face_count),
        ),
        max_faces=face_count,
        max_dense_entries=2 * face_count * face_count,
        max_resident_bytes=256 * 1024 * 1024,
    )


def _case(*, depth: float | None, smoke: bool):
    mesh = trimesh.creation.icosphere(subdivisions=0 if smoke else 1, radius=0.5)
    mesh.apply_translation((0.0, 0.0, -2.0))
    region = MeshRegion(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces, dtype=np.int32),
        feature_id="potential-flow-benchmark-sphere",
    )
    face_count = int(mesh.faces.shape[0])
    started = time.perf_counter()
    prepared = prepare_free_surface_hydrodynamics_3d(
        region,
        2.0,
        gravity=9.81,
        depth=depth,
        frame_id="benchmark-z-up",
        unit_system_id="si-water",
        policy=_policy(face_count, smoke=smoke),
    )
    preparation_seconds = time.perf_counter() - started

    vector = jnp.linspace(0.2, 1.0, face_count).astype(jnp.complex128)
    started = time.perf_counter()
    forward = prepared.boundary_operator.mv(vector)
    transposed = prepared.boundary_operator.transpose_mv(vector)
    jax.block_until_ready((forward, transposed))
    action_seconds = time.perf_counter() - started
    exact_forward_error = float(
        jnp.max(jnp.abs(forward - prepared.boundary_operator.matrix @ vector))
    )
    exact_transpose_error = float(
        jnp.max(jnp.abs(transposed - prepared.boundary_operator.matrix.T @ vector))
    )

    started = time.perf_counter()
    result = solve_potential_flow_hydrodynamics_3d(
        prepared,
        fluid_density=1025.0,
        incident_headings=(0.0, 0.5 * np.pi),
        reciprocity_tolerance=0.25,
        radiated_power_tolerance=1.0e-7,
    )
    jax.block_until_ready((result.added_mass, result.excitation_loads))
    solve_seconds = time.perf_counter() - started
    added_diagonal = np.asarray(jnp.diag(result.added_mass))
    damping_diagonal = np.asarray(jnp.diag(result.radiation_damping))
    surge_sway_added_symmetry = abs(added_diagonal[0] - added_diagonal[1]) / max(
        abs(added_diagonal[0]), abs(added_diagonal[1]), np.finfo(float).tiny
    )
    surge_sway_damping_symmetry = abs(damping_diagonal[0] - damping_diagonal[1]) / max(
        abs(damping_diagonal[0]),
        abs(damping_diagonal[1]),
        np.finfo(float).tiny,
    )
    passed = bool(
        result.valid
        & (exact_forward_error <= 1.0e-12)
        & (exact_transpose_error <= 1.0e-12)
        & (surge_sway_added_symmetry <= 0.1)
        & (surge_sway_damping_symmetry <= 0.1)
    )
    return {
        "case": "infinite-depth" if depth is None else "finite-depth",
        "depth": depth,
        "faces": face_count,
        "degrees_of_freedom": prepared.degree_of_freedom_count,
        "frame_id": prepared.frame_id,
        "unit_system_id": prepared.unit_system_id,
        "dispersion_wavenumber": prepared.green.wavenumber,
        "dispersion_residual": float(prepared.green.dispersion.residual),
        "green_tail_bound": float(prepared.green.errors.spectral_tail_envelope_bound),
        "green_radial_nodes": prepared.green.resources.radial_node_count,
        "green_angular_nodes": prepared.green.resources.angular_node_count,
        "resident_bytes": prepared.assembly_report.resident_bytes,
        "preparation_workspace_bytes": (
            prepared.assembly_report.preparation_workspace_bytes
        ),
        "boundary_operator_bytes": prepared.assembly_report.boundary_operator_bytes,
        "trace_operator_bytes": prepared.assembly_report.trace_operator_bytes,
        "exact_forward_error": exact_forward_error,
        "exact_transpose_error": exact_transpose_error,
        "added_mass_diagonal": added_diagonal.tolist(),
        "damping_diagonal": damping_diagonal.tolist(),
        "added_mass_reciprocity_defect": float(result.added_mass_reciprocity_defect),
        "damping_reciprocity_defect": float(result.damping_reciprocity_defect),
        "minimum_radiated_power_eigenvalue": float(
            result.minimum_radiated_power_eigenvalue
        ),
        "radiated_power_nonnegative": bool(result.radiated_power_nonnegative),
        "surge_sway_added_symmetry": float(surge_sway_added_symmetry),
        "surge_sway_damping_symmetry": float(surge_sway_damping_symmetry),
        "linear_relative_residuals": [
            float(item.diagnostics.relative_residual)
            for item in (
                result.radiation_linear_results + result.diffraction_linear_results
            )
        ],
        "preparation_seconds": preparation_seconds,
        "action_seconds": action_seconds,
        "solve_seconds": solve_seconds,
        "passed": passed,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Benchmark bounded zero-speed 3D potential-flow hydrodynamics."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/potential_flow_hydrodynamics.json"),
    )
    return parser


def main():
    arguments = _parser().parse_args()
    records = [
        _case(depth=None, smoke=arguments.smoke),
        _case(depth=5.0, smoke=arguments.smoke),
    ]
    payload = {
        "benchmark": "zero-speed-potential-flow-hydrodynamics-3d",
        "backend": jax.default_backend(),
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
