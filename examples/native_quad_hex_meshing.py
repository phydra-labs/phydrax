# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Generate native quadrilateral and hexahedral meshes, then solve diffusion.

A square and a unit cube use the planar-dual and PLC-dual native routes.
The Q1 field has affine Dirichlet data on every boundary. Run from the
repository root::

    JAX_ENABLE_X64=1 python -m examples.native_quad_hex_meshing

For curved sources, cavity profiles, and JSON qualification records, run
``python -m tools.meshing_qualification --scenario native-quad-hex``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx


def main() -> None:
    """Mesh two domains and print the mesh size and diffusion solution range."""
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    meshing = phx.meshing
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        ((0, 1, 2, 3),),
        feature_id="square",
    )
    planar = meshing.NativePlanarSource(region, "square-state")
    vertices = np.asarray(
        [(x, y, z) for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=np.float64,
    )
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    plc = meshing.PiecewiseLinearComplex(
        vertices,
        loops,
        np.arange(6, dtype=np.int64),
        np.asarray([(-1, 0)] * 6, dtype=np.int64),
        ("solid",),
    )
    volume = meshing.NativePlcSource(plc, "cube", "cube-state")
    for source, dimension, family, route in (
        (planar, 2, "quadrilateral", "planar_dual_quad"),
        (volume, 3, "hexahedron", "plc_dual_hex"),
    ):
        scope = meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            meshing.MeshingEntityKind.GEOMETRY,
            2,
            "boundary",
            np.arange(1 if dimension == 2 else 6, dtype=np.int64),
        )
        target = meshing.CellMeshingTarget(
            dimension,
            dimension,
            meshing.CellFamilyPolicy(required=(family,)),
        )
        sizing = (
            meshing.UniformSizeControl(
                scope, 0.5, strength=meshing.SizeControlStrength.SOFT
            ),
        )
        specification = (
            meshing.SurfaceMeshingSpec(
                target,
                scope,
                size_controls=sizing,
                planar_embedding=phx.geometry.PlanarEmbedding(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )
            if dimension == 2
            else meshing.VolumeMeshingSpec(
                target,
                scope,
                meshing.VolumeFillStrategy.MULTIZONE,
                size_controls=sizing,
            )
        )
        result = (
            meshing.NativeMeshingProvider(meshing.NativeMeshingOptions(route))
            .plan(
                source,
                specification,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
            )
            .execute()
        )
        field = phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element(family, 1),
        )
        space = phx.discretization.FiniteElementPlan(
            result.mesh,
            field,
            coordinate_spec=result.geometry,
        ).prepare()

        def boundary(points: Array) -> Array:
            return points[..., 0]

        problem = phx.equations.compile_finite_element_problem(
            phx.equations.FiniteElementForm(
                "native-diffusion",
                "u",
                (phx.equations.DiffusionAction("u", 1.0),),
            ),
            space,
            constraint=phx.discretization.dirichlet_constraint(space, "u"),
            dirichlet_values=boundary,
        )
        operator, rhs = problem.linear_system()
        solution = phx.linalg.solve(operator, rhs)
        if not bool(jnp.all(solution.successful)):
            raise RuntimeError("diffusion.solution must converge; got failed solve")
        values = problem.expand(solution.value)
        print(
            f"{family}: cells={sum(block.cell_count for block in result.mesh.blocks)}, "
            f"u range=[{float(jnp.min(values)):.3f}, {float(jnp.max(values)):.3f}]"
        )


if __name__ == "__main__":
    main()
