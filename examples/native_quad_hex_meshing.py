# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native planar-dual quads and PLC-dual hexes, each consumed by FEM.

These dual routes are not a claim of general quad or hex generation.
Set PHYDRAX_MESHCORE_LIBRARY to the intended library and JAX_ENABLE_X64=true.
"""

import json
import os
from typing import Literal

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    PiecewiseLinearComplex,
    SizeControlStrength,
    SurfaceMeshingSpec,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePlanarSource,
    NativePlcSource,
)


def execute(
    source: NativePlanarSource | NativePlcSource,
    spec: SurfaceMeshingSpec | VolumeMeshingSpec,
    route: Literal["planar_dual_quad", "plc_dual_hex"],
    family: Literal["quadrilateral", "hexahedron"],
    expected_measure: float,
) -> dict[str, object]:
    options = NativeMeshingOptions(route)
    plan = NativeMeshingProvider(options).plan(
        source, spec, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    if plan.route != route:
        raise RuntimeError(f"Expected {route}, got {plan.route}")
    result = plan.execute()
    kinds = {b.cell_kind for b in result.mesh.blocks if b.cell_count}
    measures = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    if (
        kinds != {family}
        or not measures.size
        or not np.all(np.isfinite(measures))
        or np.any(measures <= 0.0)
        or abs(float(measures.sum()) - expected_measure) > 1e-10
    ):
        raise RuntimeError(f"{route} family or source measure mismatch")
    if (
        not result.audit.passed
        or result.certification is None
        or not result.certification.passed
    ):
        raise RuntimeError(f"{route} was not certified")
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element(family, 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        f"{route}-affine", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=lambda p: p[..., 0]
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(
        operator,
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=64),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1e-12, absolute=1e-14, max_steps=2000
            ),
        ),
    )
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    error = float(
        np.max(np.abs(values - np.asarray(space.dof_maps[0].dof_coordinates)[:, 0]))
    )
    if not bool(jnp.all(solved.successful)) or error > 2e-9:
        raise RuntimeError(f"{route} affine FE solve failed: {error=}")
    return {
        "route": route,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "topology_id": result.mesh.topology_id,
        "family": family,
        "cells": int(measures.size),
        "measure": float(measures.sum()),
        "affine_error": error,
    }


def main() -> None:
    if not os.environ.get("PHYDRAX_MESHCORE_LIBRARY"):
        raise RuntimeError(
            "Set PHYDRAX_MESHCORE_LIBRARY to the native library before running."
        )
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    region = PlanarMeshRegion(vertices, ((0, 1, 2, 3),), feature_id="square")
    quad_source = NativePlanarSource(
        region,
        "square-state",
        vertex_entity_ids=np.arange(100, 104, dtype=np.int64),
        edge_entity_ids=np.arange(200, 204, dtype=np.int64),
    )
    scope = MeshingScope.whole(quad_source, 2)
    quad_spec = SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("quadrilateral",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            UniformSizeControl(scope, 0.5, strength=SizeControlStrength.SOFT),
        ),
    )
    quad = execute(quad_source, quad_spec, "planar_dual_quad", "quadrilateral", 1.0)

    corners = np.asarray(
        [(x, y, z) for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=np.float64,
    )
    faces = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    complex_ = PiecewiseLinearComplex(
        corners,
        faces,
        np.arange(6, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        ("solid",),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"cube-vertex-{i}" for i in range(8)),
        vertex_occurrence_paths=((),) * 8,
        facet_ids=tuple(f"cube-face-{i}" for i in range(6)),
        facet_occurrence_paths=((),) * 6,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    hex_source = NativePlcSource(complex_, "cube", "cube-state")
    scope = MeshingScope.whole(hex_source, 2)
    hex_spec = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("hexahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 0.5, strength=SizeControlStrength.SOFT),
        ),
    )
    hexahedron = execute(hex_source, hex_spec, "plc_dual_hex", "hexahedron", 1.0)
    print(json.dumps({"dual_quad": quad, "dual_hex": hexahedron}, indent=2))


if __name__ == "__main__":
    main()
