#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.cardiovascular.anatomy._coordinates import (
    HarmonicCoordinateFields,
    HarmonicCoordinatePlan,
    HarmonicCoordinateSpec,
)
from phydrax.applications.cardiovascular.anatomy._microstructure import (
    VentricularMicrostructurePlan,
)
from phydrax.applications.cardiovascular.anatomy._roles import (
    CardiacBoundaryProfile,
    CardiacBoundaryRoles,
)
from phydrax.applications.cardiovascular.anatomy._surfaces import ChamberSurfacePlan
from phydrax.discretization import CellMesh


def _index(i: int, j: int, k: int, /, size: int) -> int:
    return i + size * (j + size * k)


def manufactured_lv_slab(subdivisions: int):
    if subdivisions < 2:
        raise ValueError("Qualification slab requires at least two subdivisions.")
    size = subdivisions + 1
    axis = np.linspace(0.0, 1.0, size)
    points = np.asarray(
        [
            (axis[i], axis[j], axis[k])
            for k in range(size)
            for j in range(size)
            for i in range(size)
        ]
    )
    cells = []
    for k in range(subdivisions):
        for j in range(subdivisions):
            for i in range(subdivisions):
                v000 = _index(i, j, k, size)
                v100 = _index(i + 1, j, k, size)
                v010 = _index(i, j + 1, k, size)
                v110 = _index(i + 1, j + 1, k, size)
                v001 = _index(i, j, k + 1, size)
                v101 = _index(i + 1, j, k + 1, size)
                v011 = _index(i, j + 1, k + 1, size)
                v111 = _index(i + 1, j + 1, k + 1, size)
                cells.extend(
                    (
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    )
                )
    mesh = CellMesh.from_tetrahedra(points, np.asarray(cells, dtype=np.int32))
    faces = np.asarray(mesh.connectivity.faces)
    face_points = points[faces]
    assignments = {
        "endocardium": np.flatnonzero(np.all(face_points[..., 0] == 0.0, axis=1)),
        "epicardium": np.flatnonzero(np.all(face_points[..., 0] == 1.0, axis=1)),
        "apex": np.flatnonzero(np.all(face_points[..., 2] == 0.0, axis=1)),
        "base": np.flatnonzero(np.all(face_points[..., 2] == 1.0, axis=1)),
        "posterior": np.flatnonzero(np.all(face_points[..., 1] == 0.0, axis=1)),
        "anterior": np.flatnonzero(np.all(face_points[..., 1] == 1.0, axis=1)),
    }
    shared = tuple(
        (first, second)
        for first in ("endocardium", "epicardium")
        for second in ("apex", "base", "posterior", "anterior")
    ) + tuple(
        (first, second)
        for first in ("apex", "base")
        for second in ("posterior", "anterior")
    )
    profile = CardiacBoundaryProfile(
        "qualified-affine-lv-slab",
        required_roles=tuple(assignments),
        connected_roles=tuple(assignments),
        disjoint_closure_pairs=(
            ("endocardium", "epicardium"),
            ("apex", "base"),
            ("posterior", "anterior"),
        ),
        shared_closure_pairs=shared,
        exhaustive=True,
    )
    return mesh, CardiacBoundaryRoles(mesh, assignments, profile=profile)


def qualify_affine_coordinates_and_microstructure(subdivisions: int) -> dict[str, object]:
    mesh, roles = manufactured_lv_slab(subdivisions)
    coordinate_plan = HarmonicCoordinatePlan(
        mesh,
        roles,
        (
            HarmonicCoordinateSpec("transmural", "endocardium", "epicardium"),
            HarmonicCoordinateSpec("longitudinal", "apex", "base"),
        ),
    )
    fields = coordinate_plan.prepare(numeric_version="qualification").solve().commit()
    points = np.asarray(mesh.coordinates)
    expected_values = np.stack((points[:, 0], points[:, 2]))
    expected_gradients = np.broadcast_to(
        np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))[:, None, :],
        fields.cell_gradients.shape,
    )
    value_error = float(np.max(np.abs(np.asarray(fields.nodal_values) - expected_values)))
    gradient_error = float(
        np.max(np.abs(np.asarray(fields.cell_gradients) - expected_gradients))
    )
    micro_plan = VentricularMicrostructurePlan("transmural", "longitudinal")
    microstructure = micro_plan.prepare(fields).build().commit()
    frame = np.asarray(microstructure.material_frame.matrix)
    orthonormality_error = float(
        np.max(np.abs(np.swapaxes(frame, -1, -2) @ frame - np.eye(3)))
    )
    orientation_error = float(np.max(np.abs(np.linalg.det(frame) - 1.0)))

    reversed_gradients = np.asarray(fields.cell_gradients).copy()
    reversed_gradients[fields.coordinate_index("longitudinal")] *= -1.0
    reversed_fields = HarmonicCoordinateFields(
        fields.names,
        fields.nodal_values,
        fields.cell_values,
        reversed_gradients,
        fields.dirichlet_masks,
        fields.evidence,
        fields_id="qualification-longitudinal-gauge-reversal",
    )
    reversed_microstructure = micro_plan.prepare(reversed_fields).build().commit()
    tensor_sign_error = float(
        np.max(
            np.abs(
                np.asarray(reversed_microstructure.fiber_structure_tensor)
                - np.asarray(microstructure.fiber_structure_tensor)
            )
        )
    )

    degenerate_gradients = np.asarray(fields.cell_gradients).copy()
    degenerate_gradients[fields.coordinate_index("longitudinal")] = degenerate_gradients[
        fields.coordinate_index("transmural")
    ]
    degenerate_fields = HarmonicCoordinateFields(
        fields.names,
        fields.nodal_values,
        fields.cell_values,
        degenerate_gradients,
        fields.dirichlet_masks,
        fields.evidence,
        fields_id="qualification-parallel-gradients",
    )
    degenerate_candidate = micro_plan.prepare(degenerate_fields).build()
    degeneracy_rejected = not bool(degenerate_candidate.evidence.all_successful)
    passed = bool(
        value_error <= 5.0e-6
        and gradient_error <= 5.0e-6
        and orthonormality_error <= 5.0e-6
        and orientation_error <= 5.0e-6
        and tensor_sign_error <= 5.0e-6
        and degeneracy_rejected
        and bool(fields.evidence.all_successful)
    )
    return {
        "case": "manufactured-affine-tetrahedral-lv-slab",
        "nodes": int(points.shape[0]),
        "tetrahedra": int(fields.cell_values.shape[1]),
        "boundary_roles_id": roles.roles_id,
        "coordinate_plan_id": coordinate_plan.plan_id,
        "coordinate_fields_id": fields.fields_id,
        "microstructure_id": microstructure.microstructure_id,
        "maximum_nodal_coordinate_error": value_error,
        "maximum_cell_gradient_error": gradient_error,
        "maximum_frame_orthonormality_error": orthonormality_error,
        "maximum_frame_orientation_error": orientation_error,
        "fiber_structure_tensor_sign_error": tensor_sign_error,
        "parallel_gradient_degeneracy_rejected": degeneracy_rejected,
        "passed": passed,
    }


def qualify_chamber_surface() -> dict[str, object]:
    points = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    triangles = jnp.asarray(((3, 2, 1), (2, 0, 3), (1, 3, 0), (0, 2, 1)))
    surface = ChamberSurfacePlan("manufactured-lv-cavity", points, triangles).prepare()
    candidate = surface.evaluate()
    result = candidate.commit()
    autodiff_derivative = jax.grad(lambda value: surface.evaluate(value).volume)(points)
    derivative_error = float(
        jnp.max(jnp.abs(autodiff_derivative - result.coordinate_derivative))
    )
    translated_volume = (
        surface.evaluate(points + jnp.asarray((17.0, -3.0, 5.0))).commit().volume
    )
    translation_error = float(jnp.abs(translated_volume - result.volume))
    volume_error = float(jnp.abs(result.volume - 1.0 / 6.0))
    reflected_rejected = not bool(
        surface.evaluate(points.at[:, 0].multiply(-1.0)).evidence.successful
    )
    passed = bool(
        volume_error <= 5.0e-7
        and derivative_error <= 5.0e-7
        and translation_error <= 5.0e-7
        and reflected_rejected
        and bool(result.evidence.successful)
        and bool(surface.topology_evidence.successful)
    )
    return {
        "case": "closed-oriented-tetrahedral-lv-cavity",
        "surface_id": surface.surface_id,
        "volume": float(result.volume),
        "analytic_volume_error": volume_error,
        "analytic_derivative_autodiff_error": derivative_error,
        "translation_volume_error": translation_error,
        "closure_residual_norm": float(result.evidence.closure_residual_norm),
        "translation_derivative_norm": float(result.evidence.translation_derivative_norm),
        "reflection_rejected": reflected_rejected,
        "passed": passed,
    }


def qualification(subdivisions: int = 2) -> dict[str, object]:
    cases = (
        qualify_affine_coordinates_and_microstructure(subdivisions),
        qualify_chamber_surface(),
    )
    return {"cases": cases, "passed": all(bool(case["passed"]) for case in cases)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify cardiovascular affine geometry and chamber evidence."
    )
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = qualification(arguments.subdivisions)
    if not bool(payload["passed"]):
        raise RuntimeError("Cardiovascular geometry qualification failed.")
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
