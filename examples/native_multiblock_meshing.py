#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Two source-bound transfinite blocks glued into one pure-quad FEM complex.

Run with the worktree Python. The manufactured Poisson solution independently
checks interface sharing and Q1 convergence; no external mesher participates.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.geometry import LineCurve, ParametricCurveBoundarySource
from phydrax.meshing import (
    BlockInterfaceControl,
    CellMeshingResult,
    generate_structured_block,
    glue_structured_blocks,
    MultiblockConstruction,
    TransfiniteBlock,
    TransfiniteCurveControl,
)


def rectangle(name: str, left: float, right: float, n: int, /) -> TransfiniteBlock:
    return TransfiniteBlock(
        name,
        tuple(
            TransfiniteCurveControl(
                LineCurve(origin, direction), f"{name}:{edge}", (0.0, 1.0), n
            )
            for edge, origin, direction in (
                ("left", (left, 0.0), (0.0, 1.0)),
                ("right", (right, 0.0), (0.0, 1.0)),
                ("bottom", (left, 0.0), (right - left, 0.0)),
                ("top", (left, 1.0), (right - left, 0.0)),
            )
        ),
    )


def construct_multiblock(n: int, /) -> MultiblockConstruction:
    left = generate_structured_block(rectangle("left", 0.0, 0.5, n))
    right = generate_structured_block(rectangle("right", 0.5, 1.0, n))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    return glue_structured_blocks((left, right), (interface,))


def publish_multiblock(n: int, /) -> CellMeshingResult:
    blocks = (rectangle("left", 0.0, 0.5, n), rectangle("right", 0.5, 1.0, n))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    # The authoritative boundary is specified independently of generated nodes.
    curves = (
        LineCurve((0.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((0.0, 1.0), (1.0, 0.0)),
    )
    revision = str(phx.array_tree_fingerprint(curves)["sha256"])
    boundary = ParametricCurveBoundarySource(
        curves,
        ((0.0, 1.0),) * 4,
        source_id="unit-square",
        source_revision=revision,
        covering_radius=0.01,
    )
    source = phx.meshing.NativeStructuredSource(
        blocks,
        (interface,),
        "unit-square",
        revision,
        fidelity_source=boundary,
        maximum_deviation=0.08,
    )
    scope = phx.meshing.MeshingScope(
        "unit-square",
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "unit-square-domain",
        np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("quadrilateral",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 1.0 / n, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("structured_transfinite")
    )
    return provider.plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()


def manufactured_solution(points: Array, /) -> Array:
    x, y = points[..., 0], points[..., 1]
    return x * (1.0 - x) * y * (1.0 - y)


def manufactured_source(points: Array, _args: object, /) -> Array:
    x, y = points[..., 0], points[..., 1]
    return 2.0 * (x * (1.0 - x) + y * (1.0 - y))


def poisson_error(n: int, /) -> float:
    result = publish_multiblock(n)
    mesh = result.mesh
    element = phx.discretization.lagrange_element("quadrilateral", 1)
    field = phx.discretization.FiniteElementFieldSpec("u", element)
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.FiniteElementForm(
        "multiblock-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    manufactured_source, coefficient_id="manufactured-source"
                ),
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=0.0,
    )
    problem, rhs = compiled.linear_system()
    result = phx.linalg.solve(problem, rhs)
    if not bool(jnp.all(result.successful)):
        raise RuntimeError("Native multiblock FEM linear solve failed.")
    solution = compiled.expand(result.value)
    expected = manufactured_solution(mesh.coordinates)
    return float(jnp.max(jnp.abs(solution - expected)))


def main() -> None:
    coarse, fine = poisson_error(4), poisson_error(8)
    if not 0.0 < fine < coarse / 3.0:
        raise RuntimeError(f"Native multiblock FEM convergence failed: {coarse}, {fine}.")
    construction = construct_multiblock(4)
    routes = dict(construction.block_vertex_ids)
    np.testing.assert_array_equal(
        routes["left"].reshape(5, 5)[-1], routes["right"].reshape(5, 5)[0]
    )
    print(
        {
            "family": "quadrilateral",
            "shared_ids": 5,
            "coarse_error": coarse,
            "fine_error": fine,
            "refinement_ratio": coarse / fine,
            "mesh_id": construction.mesh.mesh_id,
        }
    )


if __name__ == "__main__":
    main()
