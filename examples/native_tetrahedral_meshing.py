#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native constrained tetrahedral meshing of a two-material box with a cavity.

The piecewise-linear complex is the box [0, 2] x [0, 1] x [0, 1] split by the
material interface x = 1 into the regions ``left`` and ``right``; the right
region contains the void cavity [1.25, 1.75] x [0.25, 0.75] x [0.25, 0.75].
Facets declare the region on each side of their oriented loops, so the native
route (``NativeMeshingOptions("plc_tetrahedral")``):

1. validates the complex exactly and recovers every facet, interface and
   cavity wall in a constrained tetrahedralization (``recover_plc_3d``);
2. classifies regions from the facet incidence, refines and improves;
3. publishes only after the independent ``volume_plc`` certification (global
   embedding and exact coverage of the declared facets and region volumes).

The mesh then carries a P1 solve of -Δu = f with the manufactured solution
u = sin(x) cosh(y) + z², whose exact Dirichlet data holds on every boundary
(outer walls and cavity). The report compares the region volumes with the
analytic ones and the discrete solution with u in the energy norm.

Run from the repository root::

    python examples/native_tetrahedral_meshing.py
"""

import json
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


M = phx.meshing

# Box corner i of [lower, upper]: x = bit 0, y = bit 1, z = bit 2; loops wind
# counterclockwise seen from outside.
BOX_LOOPS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def box(lower: Any, upper: Any) -> Any:
    return np.asarray(
        [[(upper if i >> k & 1 else lower)[k] for k in range(3)] for i in range(8)],
        dtype=np.float64,
    )


def complex_with_cavity() -> Any:
    """Two unit cubes sharing x = 1, plus a void cavity inside the right one."""
    left = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    right = box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    cavity = box((1.25, 0.25, 0.25), (1.75, 0.75, 0.75))
    # Both cubes share the interface vertices; keep one copy of each.
    vertices = np.concatenate((left, right[[1, 3, 5, 7]], cavity))
    right_index = {0: 1, 2: 3, 4: 5, 6: 7, 1: 8, 3: 9, 5: 10, 7: 11}
    loops, facets = [], []
    for loop in BOX_LOOPS:
        if loop != (1, 3, 7, 5):  # the left cube's x = 1 face is the interface
            loops.append(loop)
            facets.append(0)
        if loop != (0, 4, 6, 2):  # the right cube's x = 1 face is the interface
            loops.append(tuple(right_index[v] for v in loop))
            facets.append(1)
    loops.append((1, 3, 7, 5))  # interface, normal +x toward ``right``
    facets.append(2)
    loops.extend(tuple(v + 12 for v in loop) for loop in BOX_LOOPS)
    facets.extend([3] * 6)  # cavity walls: normal out of the void into ``right``
    regions = [[-1, 0], [-1, 1], [1, 0], [1, -1]]
    return M.PiecewiseLinearComplex(vertices, loops, facets, regions, ("left", "right"))


def generate(complex_: Any) -> Any:
    source = M.NativePlcSource(complex_, "cavity-box", "r1")
    scope = M.MeshingScope(
        "cavity-box",
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        2,
        "cavity-box-facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    interface = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        scope.entity_set_id,
        np.asarray([2], dtype=np.int64),
    )
    cavity = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        scope.entity_set_id,
        np.asarray([3], dtype=np.int64),
    )
    region_scopes = tuple(
        M.MeshingScope(
            source.source_id,
            source.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            3,
            "cavity-box-regions",
            np.asarray([region], dtype=np.int64),
        )
        for region in range(2)
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 0.2, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=(
            M.RegionControl(
                region_scopes[0], "left", "material-left", M.RegionRole.SOLID
            ),
            M.RegionControl(
                region_scopes[1], "right", "material-right", M.RegionRole.SOLID
            ),
        ),
        patch_controls=(
            M.PatchControl("material-interface", interface, ("left", "right")),
            M.PatchControl("cavity-wall", cavity, ("right",)),
        ),
        hole_seeds=(M.HoleSeed(np.asarray((1.5, 0.5, 0.5), dtype=np.float64), cavity),),
    )
    provider = M.NativeMeshingProvider(M.NativeMeshingOptions("plc_tetrahedral"))
    return provider.plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()


def exact_solution(points: Any) -> Any:
    return jnp.sin(points[..., 0]) * jnp.cosh(points[..., 1]) + points[..., 2] ** 2


def exact_gradient(points: Any) -> Any:
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return jnp.stack(
        (jnp.cos(x) * jnp.cosh(y), jnp.sin(x) * jnp.sinh(y), 2.0 * z), axis=-1
    )


def solve(mesh: Any) -> Any:
    """P1 solve of -Δu = -2 (the Laplacian of u is 2) with exact boundary data."""
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.FiniteElementForm(
        "cavity-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    lambda x, _: -2.0 * jnp.ones(x.shape[:-1]),
                    coefficient_id="manufactured-source",
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=exact_solution,
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("The P1 diffusion solve failed.")
    return space, problem.expand(solved.value)


def energy_error(space: Any, values: Any) -> Any:
    """||∇(u - u_h)||_L2 with a Duffy-mapped Gauss rule of 5^3 points."""
    rule = phx.integration.ReferenceTetrahedronRule(phx.integration.GaussLegendreRule(5))
    data = phx.integration.reference_rule_data(rule)
    geometry = space.evaluate_block_geometry(
        "u", 0, space.default_runtime.coordinates, data.points, data.weights
    )
    dofs = space.dof_maps[0].cell_dofs[0]
    discrete = phx.ein.contract("cqld,cl->cqd", geometry.physical_gradients, values[dofs])
    difference = exact_gradient(geometry.physical_points) - discrete
    return jnp.sqrt(
        jnp.sum(geometry.physical_weights * jnp.sum(difference * difference, axis=-1))
    )


def main() -> None:
    result = generate(complex_with_cavity())
    mesh = result.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    corners = points[cells]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    identifiers = np.asarray(mesh.entity_set(3).entity_ids)
    measured = {
        zone.name: float(np.sum(volumes[np.isin(identifiers, zone.scope.entity_ids)]))
        for zone in result.zones
    }
    analytic = {"left": 1.0, "right": 1.0 - 0.5**3}
    space, values = solve(mesh)
    nodal = float(jnp.max(jnp.abs(values - exact_solution(jnp.asarray(points)))))
    report = {
        "cells": int(cells.shape[0]),
        "vertices": int(points.shape[0]),
        "stages": {
            stage.stage.value: stage.status.value for stage in result.trace.stages
        },
        "region_volume_error": {
            name: abs(measured[name] - analytic[name]) for name in analytic
        },
        "minimum_cell_volume": float(np.min(volumes)),
        "energy_error": float(energy_error(space, values)),
        "maximum_nodal_error": nodal,
        "evidence": {
            name: value
            for name, value in result.compliance.achieved
            if name.startswith(
                ("construction:", "steiner", "minimum_dihedral", "slivers")
            )
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
