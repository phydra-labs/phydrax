#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact Delaunay, constrained Delaunay, Voronoi, and power diagrams on meshcore.

Workflow:

1. Delaunay triangulations of random points in 2D and 3D. Every simplex is
   checked with the EXACT ``orient2d``/``orient3d`` predicates (positive
   orientation) and against every input point with the EXACT
   ``incircle``/``insphere`` predicates (empty circumcircle/circumsphere).
2. A constrained Delaunay triangulation of a square plate with a square hole,
   refined by Ruppert/Chew insertion to a minimum angle and a maximum area
   within a Steiner-point budget.
3. Voronoi and power diagrams of random generators clipped to a box in 2D and
   3D; the cell measures sum to the box measure.
4. The refined triangulation is certified as a ``CellMesh`` and a P1 Poisson
   problem with affine Dirichlet data is solved on it, which P1 elements
   reproduce exactly.

The native meshcore library is required: install ``phydrax[meshcore]`` or set
``PHYDRAX_MESHCORE_LIBRARY`` to the built ``libphydrax_meshcore`` shared library.

Run with ``python examples/delaunay_voronoi.py``.
"""

import json
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


EXACT = phx.geometry.PredicateMode.EXACT
if phx.geometry.resolve_host_predicate_mode(EXACT) is not EXACT:
    raise SystemExit(
        "examples/delaunay_voronoi.py requires the native meshcore library: install "
        "phydrax[meshcore] or set PHYDRAX_MESHCORE_LIBRARY to libphydrax_meshcore."
    )

rng = np.random.default_rng(2026)


def verify_delaunay(points: Any) -> Any:
    """Triangulate and verify orientation and empty circumballs exactly."""
    triangulation = phx.geometry.DelaunayTriangulation(points)
    simplices = triangulation.simplices
    corners = tuple(points[simplices[:, index]] for index in range(simplices.shape[1]))
    # Every (simplex, point) pair: signs have shape (simplices, points).
    balls = tuple(corner[:, None, :] for corner in corners)
    if triangulation.dimension == 2:
        orientation = phx.geometry.orient2d(*corners, mode=EXACT)
        # ty: ignore[too-many-positional-arguments]
        inside = phx.geometry.incircle(*balls, points[None, :, :], mode=EXACT)
    else:
        orientation = phx.geometry.orient3d(*corners, mode=EXACT)
        # ty: ignore[too-many-positional-arguments]
        inside = phx.geometry.insphere(*balls, points[None, :, :], mode=EXACT)
    if not (np.all(orientation.certain) and np.all(inside.certain)):
        raise RuntimeError("EXACT predicates left an uncertain sign.")
    positive = int(np.count_nonzero(orientation.signs == 1))
    violations = int(np.count_nonzero(inside.signs > 0))
    # Each simplex's own vertices lie on its circumball.
    ties = int(np.count_nonzero(inside.signs == 0)) - simplices.size
    if positive != simplices.shape[0] or violations or ties:
        raise RuntimeError(
            f"{triangulation.dimension}D Delaunay verification failed: "
            f"{positive} positive of {simplices.shape[0]}, {violations} violations."
        )
    evidence = triangulation.evidence
    return {
        "points": points.shape[0],
        "simplices": simplices.shape[0],
        "in_ball_checks": inside.signs.size,
        "empty_ball_violations": violations,
        "positive_orientations": positive,
        "predicate_mode": inside.mode.value,
        "evidence_predicate_mode": evidence.predicate_mode.value,
        "route": evidence.route,
        "status": evidence.status,
    }


def verify_diagram(diagram: Any, box_measure: Any) -> Any:
    measures = diagram.cells.measures
    total = float(np.sum(measures))
    if abs(total - box_measure) > 1.0e-12 * box_measure or np.any(measures < 0.0):
        raise RuntimeError(f"{diagram.evidence.route} cells do not partition the box.")
    evidence = diagram.evidence
    return {
        "cells": diagram.cells.cell_count,
        "empty_cells": int(np.count_nonzero(measures == 0.0)),
        "redundant_generators": evidence.redundant_count,
        "measure_sum": total,
        "box_measure": box_measure,
        "route": evidence.route,
        "evidence_predicate_mode": evidence.predicate_mode.value,
    }


delaunay = {
    "2d": verify_delaunay(rng.random((160, 2))),
    "3d": verify_delaunay(rng.random((60, 3))),
}

diagrams = {}
for dimension, count in ((2, 48), (3, 32)):
    lower = np.zeros((dimension,), dtype=np.float64)
    upper = np.asarray((1.0, 2.0, 1.0)[:dimension], dtype=np.float64)
    box_measure = float(np.prod(upper - lower))
    generators = lower + (upper - lower) * rng.random((count, dimension))
    voronoi = phx.geometry.VoronoiDiagram(generators, box_lower=lower, box_upper=upper)
    power = phx.geometry.PowerDiagram(
        generators, 0.01 * rng.random(count), box_lower=lower, box_upper=upper
    )
    diagrams[f"voronoi_{dimension}d"] = verify_diagram(voronoi, box_measure)
    diagrams[f"power_{dimension}d"] = verify_diagram(power, box_measure)

# Square plate [0, 2]^2 with the square hole [0.75, 1.25]^2 (seeded at its center).
outer = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)))
inner = np.asarray(((0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)))
loop = np.arange(4, dtype=np.int32)
segments = np.concatenate(
    (
        np.stack((loop, (loop + 1) % 4), axis=1),
        4 + np.stack((loop, (loop + 1) % 4), axis=1),
    )
)
holes = np.asarray(((1.0, 1.0),))
min_angle, max_area = 28.0, 0.02
unrefined = phx.geometry.ConstrainedDelaunayTriangulation(
    np.concatenate((outer, inner)), segments, holes=holes
)
cdt = phx.geometry.ConstrainedDelaunayTriangulation(
    np.concatenate((outer, inner)),
    segments,
    holes=holes,
    min_angle=min_angle,
    max_area=max_area,
    max_steiner=2000,
)
corners = tuple(cdt.points[cdt.triangles[:, index]] for index in range(3))
orientation = phx.geometry.orient2d(*corners, mode=EXACT)
first, second = corners[1] - corners[0], corners[2] - corners[0]
areas = 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
plate_area = 4.0 - 0.25
if (
    cdt.evidence.status != "ok"
    or cdt.evidence.minimum_angle_degrees < min_angle
    or not np.all(orientation.signs == 1)
    or np.max(areas) > max_area
    or abs(float(np.sum(areas)) - plate_area) > 1.0e-12 * plate_area
):
    raise RuntimeError(
        "The refined constrained Delaunay triangulation failed its bounds."
    )

mesh = phx.discretization.CellMesh.from_triangles(cdt.points, cdt.triangles)
certified = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
target = certified.mesh
field = phx.discretization.FiniteElementFieldSpec(
    "u", phx.discretization.lagrange_element("triangle", 1)
)
space = phx.discretization.FiniteElementPlan(target, field).prepare()
constraint = phx.discretization.dirichlet_constraint(space, "u")
form = phx.equations.FiniteElementForm(
    "delaunay-affine-poisson", "u", (phx.equations.DiffusionAction("u", 1.0),)
)
problem = phx.equations.compile_finite_element_problem(
    form,
    space,
    constraint=constraint,
    dirichlet_values=lambda points: 1.0 + 2.0 * points[..., 0] - points[..., 1],
)
operator, rhs = problem.linear_system()
# A tight Krylov tolerance so the affine reproduction error measures P1 exactness.
solved = phx.linalg.solve(
    operator,
    rhs,
    control=phx.linalg.LinearSolveControl(
        relative_tolerance=1.0e-14, absolute_tolerance=1.0e-14
    ),
)
expected = 1.0 + 2.0 * target.coordinates[:, 0] - target.coordinates[:, 1]
error = float(jnp.max(jnp.abs(problem.expand(solved.value) - expected)))
if not (certified.audit.passed and bool(jnp.all(solved.successful)) and error < 1e-10):
    raise RuntimeError("The certified triangulation failed the affine Poisson solve.")

print(
    json.dumps(
        {
            "meshcore": cdt.evidence.provider_identity,
            "delaunay": delaunay,
            "diagrams": diagrams,
            "constrained_delaunay": {
                "status": cdt.evidence.status,
                "requested_minimum_angle_degrees": min_angle,
                "minimum_angle_degrees": cdt.evidence.minimum_angle_degrees,
                "unrefined_minimum_angle_degrees": (
                    unrefined.evidence.minimum_angle_degrees
                ),
                "maximum_area": float(np.max(areas)),
                "area": float(np.sum(areas)),
                "input_points": cdt.input_point_count,
                "steiner_points": cdt.evidence.steiner_count,
                "triangles": cdt.evidence.simplex_count,
                "unrefined_triangles": unrefined.evidence.simplex_count,
                "evidence_predicate_mode": cdt.evidence.predicate_mode.value,
            },
            "poisson": {
                "audit_passed": certified.audit.passed,
                "certified_minimum_angle_degrees": float(
                    np.degrees(certified.quality.minimum_angle)
                ),
                "vertices": target.coordinates.shape[0],
                "cells": target.blocks[0].cell_count,
                "solution_error": error,
            },
        },
        indent=2,
    )
)
