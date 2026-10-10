#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native wall -> advancing prism layers -> immutable PLC core -> diffusion.

The open wall cap does not stand in for the domain boundary: every remaining
side and top triangle is included explicitly in the core PLC. Shared vertices
are authoritative integer identities, not tolerance-welded coordinates. The
native provider publishes only after independent combined-domain coverage,
global embedding, mapped-cell validity and topology checks. No Gmsh/OCCT is
used. Run ``python examples/boundary_layer_core_mesh.py`` with meshcore built.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx


M = phx.meshing
D = phx.discretization


def exact_solution(points: Array, /) -> Array:
    return jnp.sum(points, axis=-1)


def mesh_layers_and_core() -> tuple[M.CellMeshingResult, M.BoundaryLayerMesh]:
    """Exact unit-box source, with a two-cell-per-edge native wall lattice."""
    values = np.linspace(0.0, 1.0, 3, dtype=np.float64)
    x, y = np.meshgrid(values, values, indexing="ij")
    wall_points = np.stack((x.ravel(), y.ravel(), np.zeros(9, dtype=np.float64)), axis=1)
    triangles = np.asarray(
        [
            row
            for i in range(2)
            for j in range(2)
            for a in (3 * i + j,)
            for row in ((a, a + 3, a + 4), (a, a + 4, a + 1))
        ],
        dtype=np.int64,
    )
    wall = D.CellMesh(wall_points, (D.CellBlock("wall", "triangle", triangles),))
    cells = wall.entity_set(2)
    wall_scope = M.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        cells.entity_ids,
    )
    schedule = M.LayerSchedule.geometric(2, 0.1, growth_rate=1.0)
    layers = M.prepare_boundary_layers(
        wall,
        M.BoundaryLayerControl(
            wall_scope, schedule, route=M.BoundaryLayerRoute.ADVANCING
        ),
    )
    if layers.cap is None:
        raise RuntimeError("The square wall must produce an open cap.")
    cap_points = np.asarray(layers.cap.coordinates, dtype=np.float64)
    cap = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
    )
    upper = cap_points.copy()
    upper[:, 2] = 1.0
    count = cap_points.shape[0]
    rim = (0, 3, 6, 7, 8, 5, 2, 1)
    sides = np.asarray(
        [
            triangle
            for a, b in zip(rim, (*rim[1:], rim[0]), strict=True)
            for triangle in ((a, b, b + count), (a, b + count, a + count))
        ],
        dtype=np.int64,
    )
    polygons = np.concatenate((cap, cap + count, sides))
    complex_ = M.PiecewiseLinearComplex(
        np.concatenate((cap_points, upper)),
        tuple(polygons),
        np.concatenate(
            (
                np.zeros(cap.shape[0], dtype=np.int64),
                np.ones(cap.shape[0] + sides.shape[0], dtype=np.int64),
            )
        ),
        np.asarray(((0, -1), (-1, 0)), dtype=np.int64),
        ("fluid",),
        boundary="fixed",
    )
    source = M.NativeLayerCoreSource(
        layers,
        complex_,
        "layered-unit-box",
        "reference",
        vertex_layer_ids=np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(count, -1, dtype=np.int64),
            )
        ),
        cap_polygon_ids=np.arange(cap.shape[0], dtype=np.int64),
        layer_regions=np.zeros(
            sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
        ),
    )
    core_scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "core-facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
        ),
        core_scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(core_scope, 1.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    return result, layers


def solve_diffusion(result: M.CellMeshingResult) -> float:
    """Mixed H1 patch test: -Laplacian u=0 and u=x+y+z on the box boundary."""
    field = D.FiniteElementFieldSpec(
        "u",
        {
            block.name: D.lagrange_element(block.cell_kind, 1)
            for block in result.mesh.blocks
        },
    )
    space = D.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "layer-core-diffusion", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=D.dirichlet_constraint(space, "u"),
        dirichlet_values=exact_solution,
    )
    operator, rhs = problem.linear_system()
    # The 1e-10 patch claim is a discretization check: a direct factorization
    # keeps the default GMRES stopping residual (about 4e-9 after mixed
    # refinement) from masquerading as patch-test error.
    solved = phx.linalg.solve(
        operator, rhs, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("The native mixed diffusion solve did not converge.")
    values = problem.expand(solved.value)
    points = space.dof_maps[0].dof_coordinates
    error = float(jnp.max(jnp.abs(values - exact_solution(points))))
    if error > 1.0e-10:
        raise RuntimeError(f"The native mixed diffusion patch error is {error}.")
    return error


def main() -> None:
    result, layers = mesh_layers_and_core()
    error = solve_diffusion(result)
    if result.certification is None or result.certification.coverage is None:
        raise RuntimeError("Combined coverage certification is mandatory.")
    print(
        json.dumps(
            {
                "result_id": result.result_id,
                "cell_counts": {
                    block.cell_kind: block.cell_count for block in result.mesh.blocks
                },
                "requested_thicknesses": layers.evidence.requested_thicknesses,
                "column_thicknesses": np.asarray(
                    layers.evidence.column_thicknesses
                ).tolist(),
                "active_column_counts": np.asarray(
                    layers.evidence.active_column_counts
                ).tolist(),
                "region_volumes": np.asarray(
                    result.certification.coverage.achieved_region_measures
                ).tolist(),
                "certification_passed": result.certification.passed,
                "diffusion_patch_max_error": error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
