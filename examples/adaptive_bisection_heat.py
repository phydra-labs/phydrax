#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Adaptive steady heat conduction on native newest-vertex/Maubach bisection.

Solves -Δu = f on the unit square (triangles) and the unit cube (tetrahedra)
with the manufactured solution u = atan(α (|x - c| - r0)), whose steep front is
a circular (spherical) shell centered just outside the domain. Every cycle:

1. solves the P1 problem with the exact Dirichlet data
   (`compile_finite_element_problem`, `phx.linalg.solve`);
2. estimates the energy error per cell by Zienkiewicz-Zhu gradient recovery
   (`prepare_gradient_recovery`, `recovery_error_estimate`) and compares it
   with the true energy error (effectivity index);
3. marks the Dörfler bulk: the fewest largest indicators whose squared sum
   reaches theta times the total (`dorfler_mark`);
4. refines the marks with `NATIVE_BISECTION` (`MarkedMeshAdaptation` carrying
   the previous `hierarchy`) and carries the previous solution to the refined
   mesh through `result.transfer`, whose linear reproduction is measured.

Finally every cell of the adapted mesh is offered for coarsening: complete
bisection patches are removed until the source topology is restored exactly
(the status is PARTIAL because never-refined cells have no patch to merge).

Run from the repository root::

    python examples/adaptive_bisection_heat.py
"""

import itertools
import json

import jax.numpy as jnp
import numpy as np

import phydrax as phx


THETA = 0.5
SHARPNESS = 12.0
FRONT_RADIUS = 0.7
FRONT_CENTER = -0.05


def front(points):
    """Radius, profile argument, and first two radial derivatives of u."""
    offset = points - FRONT_CENTER
    radius = jnp.sqrt(jnp.sum(offset * offset, axis=-1))
    argument = SHARPNESS * (radius - FRONT_RADIUS)
    first = SHARPNESS / (1.0 + argument**2)
    second = -2.0 * SHARPNESS**2 * argument / (1.0 + argument**2) ** 2
    return offset, radius, argument, first, second


def exact_solution(points):
    return jnp.arctan(front(points)[2])


def exact_gradient(points):
    offset, radius, _, first, _ = front(points)
    return (first / radius)[..., None] * offset


def heat_source(points):
    """f = -Δu = -(u'' + (d - 1) u' / r) for the radial profile u(r)."""
    _, radius, _, first, second = front(points)
    return -(second + (points.shape[-1] - 1) * first / radius)


def unit_square(count):
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((count + 1) ** 2, dtype=np.int32).reshape((count + 1, count + 1))
    lower, right = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    upper, left = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate(
        (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
    )
    return phx.discretization.CellMesh.from_triangles(points, triangles)


def unit_cube(count):
    """Kuhn tetrahedra: six per lattice cube along the monotone vertex paths."""
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape((-1, 3))
    index = np.arange((count + 1) ** 3, dtype=np.int32).reshape((count + 1,) * 3)
    origins = index[:-1, :-1, :-1].ravel()
    strides = np.asarray(((count + 1) ** 2, count + 1, 1), dtype=np.int32)
    paths = np.asarray(tuple(itertools.permutations(range(3))), dtype=np.int32)
    steps = np.cumsum(strides[paths], axis=1)
    offsets = np.concatenate((np.zeros((6, 1), dtype=np.int32), steps), axis=1)
    cells = (origins[:, None, None] + offsets[None]).reshape((-1, 4))
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    cells[negative] = cells[negative][:, [1, 0, 2, 3]]
    return phx.discretization.CellMesh.from_tetrahedra(points, cells)


def solve(mesh):
    """P1 solve of -Δu = f with the exact Dirichlet data on every boundary vertex."""
    kind = mesh.blocks[0].cell_kind
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element(kind, 1)
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.FiniteElementForm(
        "bisection-heat",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    lambda x, _: heat_source(x), coefficient_id="shell-front-source"
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
        raise RuntimeError("The P1 heat solve failed.")
    return space, problem.expand(solved.value)


def energy_error(space, values):
    """True ||∇(u - u_h)||_L2 with a Duffy-mapped Gauss rule of 5^d points."""
    kind = space.mesh.blocks[0].cell_kind
    rule = (
        phx.integration.ReferenceTriangleRule
        if kind == "triangle"
        else phx.integration.ReferenceTetrahedronRule
    )(phx.integration.GaussLegendreRule(5))
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


def adapt(source, refine, coarsen, hierarchy):
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(refine, coarsen, hierarchy=hierarchy),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )


def adaptive_run(mesh, cycles):
    initial = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    current, hierarchy = initial, None
    slope = np.linspace(0.5, 1.5, mesh.ambient_dimension)
    history = []
    true_errors = []
    space, values = solve(current.mesh)
    for cycle in range(cycles + 1):
        recovery = phx.discretization.fem.prepare_gradient_recovery(space, "u")
        estimate, evidence = phx.discretization.fem.recovery_error_estimate(
            recovery, values
        )
        if not evidence.passed:
            raise RuntimeError("Gradient recovery failed on a vertex patch.")
        estimated = float(estimate.global_estimate)
        true_error = float(energy_error(space, values))
        true_errors.append(true_error)
        record = {
            "cycle": cycle,
            "cells": current.mesh.blocks[0].cell_count,
            "vertices": current.mesh.coordinates.shape[0],
            "estimated_error": estimated,
            "true_error": true_error,
            "effectivity_index": estimated / true_error,
            "audit_passed": current.audit.passed,
        }
        if not current.audit.passed or not 0.3 < estimated / true_error < 3.0:
            raise RuntimeError(f"Cycle {cycle} failed its audit or effectivity bound.")
        if cycle < cycles:
            marks = phx.discretization.fem.dorfler_mark(
                estimate.cell_indicators,
                THETA,
                cell_global_ids=current.mesh.blocks[0].global_ids,
            )
            refined = adapt(current, marks, (), hierarchy)
            transfer = refined.transfer
            if (
                refined.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
                or transfer is None
                or not transfer.preserves_linear
            ):
                raise RuntimeError(f"Cycle {cycle} refinement was not complete.")
            # The transferred solution interpolates u_h on the refined mesh; its
            # distance to the next solve is the update the new cells contribute.
            carried = transfer.apply(values)
            linear = transfer.apply(jnp.asarray(current.mesh.coordinates) @ slope)
            reproduction = float(
                jnp.max(jnp.abs(linear - refined.target.mesh.coordinates @ slope))
            )
            if reproduction > 1.0e-12:
                raise RuntimeError("The bisection transfer does not reproduce P1.")
            record |= {
                "marked_cells": marks.shape[0],
                "bisections": refined.evidence.bisections,
                "closure_iterations": refined.evidence.closure_iterations,
                "maximum_generation": refined.evidence.maximum_generation,
                "transfer_linear_reproduction_error": reproduction,
            }
            current, hierarchy = refined.target, refined.hierarchy
            space, values = solve(current.mesh)
            record["carried_solution_max_update"] = float(
                jnp.max(jnp.abs(values - carried))
            )
        history.append(record)
    if not all(b < a for a, b in itertools.pairwise(true_errors)):
        raise RuntimeError("The true energy error did not decrease every cycle.")
    everything = np.sort(np.asarray(current.mesh.blocks[0].global_ids))
    restored = adapt(current, (), everything, hierarchy)
    if restored.target.mesh.topology_id != initial.mesh.topology_id:
        raise RuntimeError("Coarsening did not restore the initial topology.")
    return {
        "cycles": history,
        "coarsening": {
            "status": restored.status.value,
            "coarsened_vertices": restored.evidence.coarsened_vertices,
            "coarsening_passes": restored.evidence.coarsening_passes,
            "restored_cells": restored.target.mesh.blocks[0].cell_count,
            "coarsened_topology_id": restored.target.mesh.topology_id,
            "initial_topology_id": initial.mesh.topology_id,
            "audit_passed": restored.target.audit.passed,
        },
    }


summary = {
    "theta": THETA,
    "triangles": adaptive_run(unit_square(8), 3),
    "tetrahedra": adaptive_run(unit_cube(2), 2),
}
print(json.dumps(summary, indent=2))
