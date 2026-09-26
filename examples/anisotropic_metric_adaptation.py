#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Hessian-driven anisotropic metric adaptation of a P1 internal-layer solve.

Workflow, repeated for three cycles on the unit square:

1. solve ``-laplace(u) = f`` with P1 elements for the manufactured internal layer
   ``u = tanh(k (x - 1/2))`` (exact Dirichlet data on the whole boundary);
2. recover the vertex Hessian of the discrete solution
   (``phx.discretization.fem.recover_hessian``);
3. build the ``L^2`` Loseille-Alauzet metric with an explicit target complexity,
   size bounds, and anisotropy bound (``phx.meshing.lp_metric_from_hessian``);
4. bind the metric to the certified vertices and bound and grade it with
   ``normalize_mesh_metric`` under an explicit ``MetricNormalizationPolicy``;
5. adapt with ``MetricMeshAdaptation`` on the ``NATIVE_METRIC_2D`` route and
   re-solve on the certified adapted mesh.

Every cycle reports the mesh size, the Riemannian edge lengths of the adapted mesh
in its metric (``metric_edge_lengths``), requested and achieved anisotropy, the
adaptation status, the error against the exact solution, and the audit result.

Run with ``python examples/anisotropic_metric_adaptation.py``.
"""

import json

import jax.numpy as jnp
import numpy as np

import phydrax as phx


LAYER = 25.0
COMPLEXITY = 120.0
MINIMUM_SIZE = 0.004
MAXIMUM_SIZE = 0.35
MAXIMUM_ANISOTROPY = 16.0
CYCLES = 3
_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)


def exact_solution(points):
    return jnp.tanh(LAYER * (points[..., 0] - 0.5))


def layer_source(points, args):
    """``f = -u''`` for ``u = tanh(k (x - 1/2))``."""
    value = jnp.tanh(LAYER * (points[..., 0] - 0.5))
    return 2.0 * LAYER**2 * value * (1.0 - value**2)


def square_mesh(count):
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((count + 1) ** 2, dtype=np.int32).reshape((count + 1, count + 1))
    lower, right = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    upper, left = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate(
        (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
    )
    return phx.discretization.CellMesh.from_triangles(points, triangles)


def scope_order(mesh):
    """Vertex rows in the (sorted global ID) row order of a vertex scope."""
    return np.argsort(np.asarray(mesh.vertex_global_ids), kind="stable")


def vertex_scope(mesh):
    vertices = mesh.entity_set(0)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )


def cell_vertices(mesh):
    return np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])


def dual_areas(mesh):
    """Barycentric dual area of every vertex row."""
    points = np.asarray(mesh.coordinates)
    cells = cell_vertices(mesh)
    first = points[cells[:, 1]] - points[cells[:, 0]]
    second = points[cells[:, 2]] - points[cells[:, 0]]
    area = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    volumes = np.zeros((points.shape[0],), dtype=np.float64)
    np.add.at(volumes, cells, np.repeat(area[:, None] / 3.0, 3, axis=1))
    return volumes


def stretch_ratio(trace, determinant):
    """``sqrt(lambda_max / lambda_min)`` of 2x2 SPD tensors from trace and determinant."""
    root = np.sqrt(np.maximum(trace**2 - 4.0 * determinant, 0.0))
    return np.sqrt((trace + root) / (trace - root))


def cell_anisotropy(mesh):
    """Singular-value ratio of the map from the equilateral triangle to each cell."""
    points = np.asarray(mesh.coordinates)
    cells = cell_vertices(mesh)
    first = points[cells[:, 1]] - points[cells[:, 0]]
    second = points[cells[:, 2]] - points[cells[:, 0]]
    # Columns (first, second) times the inverse of [[1, 1/2], [0, sqrt(3)/2]].
    third = (2.0 * second - first) / np.sqrt(3.0)
    frobenius = np.sum(first**2 + third**2, axis=1)
    determinant = first[:, 0] * third[:, 1] - first[:, 1] * third[:, 0]
    return stretch_ratio(frobenius, determinant**2)


def metric_anisotropy(values):
    tensors = np.asarray(values)
    trace = tensors[:, 0, 0] + tensors[:, 1, 1]
    determinant = tensors[:, 0, 0] * tensors[:, 1, 1] - tensors[:, 0, 1] ** 2
    return stretch_ratio(trace, determinant)


def solve(certified):
    mesh = certified.mesh
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    dof_coordinates = np.asarray(space.dof_maps[0].dof_coordinates)
    if not np.array_equal(dof_coordinates, np.asarray(mesh.coordinates)):
        raise RuntimeError("P1 degrees of freedom are not aligned with the vertex rows.")
    form = phx.equations.FiniteElementForm(
        "anisotropic-layer-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    layer_source, coefficient_id="tanh-layer-source"
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
    # A few hundred unknowns: an explicit direct factorization is exact and cheap,
    # and stays robust on the stretched cells the metric produces.
    solved = phx.linalg.solve(
        operator, rhs, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("The P1 layer solve failed.")
    values = problem.expand(solved.value)
    geometry = space.block_geometries[0][0]
    cell_dofs = jnp.asarray(space.dof_maps[0].cell_dofs[0])
    discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[cell_dofs])
    defect = exact_solution(geometry.physical_points) - discrete
    l2_error = float(jnp.sqrt(jnp.sum(geometry.physical_weights * defect**2)))
    nodal_error = float(jnp.max(jnp.abs(values - exact_solution(dof_coordinates))))
    return space, values, l2_error, nodal_error


def normalized_metric(certified, space, values):
    mesh = certified.mesh
    prepared = phx.discretization.fem.prepare_gradient_recovery(space, "u")
    hessian, recovery = phx.discretization.fem.recover_hessian(prepared, values)
    if not recovery.passed:
        raise RuntimeError("Hessian recovery failed on at least one vertex patch.")
    hessian = np.asarray(hessian, dtype=np.float64)
    hessian = 0.5 * (hessian + np.swapaxes(hessian, -1, -2))
    order = scope_order(mesh)
    volumes = dual_areas(mesh)[order]
    tensors, lp = phx.meshing.lp_metric_from_hessian(
        hessian[order],
        p=2.0,
        target_complexity=COMPLEXITY,
        minimum_size=MINIMUM_SIZE,
        maximum_size=MAXIMUM_SIZE,
        maximum_anisotropy=MAXIMUM_ANISOTROPY,
        vertex_volumes=volumes,
    )
    if not lp.passed:
        raise RuntimeError("The Lp metric did not meet its target complexity.")
    raw = phx.meshing.MeshMetricField(
        vertex_scope(mesh),
        tensors,
        minimum_size=MINIMUM_SIZE,
        maximum_size=MAXIMUM_SIZE,
        maximum_anisotropy=MAXIMUM_ANISOTROPY,
    )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.shape[0])
    edges = inverse[np.asarray(mesh.connectivity.edges)]
    metric, normalization = phx.meshing.normalize_mesh_metric(
        raw,
        policy=phx.meshing.MetricNormalizationPolicy(
            minimum_size=MINIMUM_SIZE,
            maximum_size=MAXIMUM_SIZE,
            maximum_anisotropy=MAXIMUM_ANISOTROPY,
            gradation=phx.meshing.MetricGradationPolicy(1.5, anisotropic=True),
        ),
        adjacency=edges,
        coordinates=np.asarray(mesh.coordinates)[order],
        vertex_volumes=volumes,
    )
    if not normalization.passed:
        raise RuntimeError("Metric gradation did not converge.")
    return metric, lp, normalization


def edge_length_summary(certified, metric):
    mesh = certified.mesh
    order = scope_order(mesh)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.shape[0])
    lengths = np.asarray(
        phx.meshing.metric_edge_lengths(
            metric.values,
            np.asarray(mesh.coordinates)[order],
            inverse[np.asarray(mesh.connectivity.edges)],
        )
    )
    unit = (lengths >= _LOWER) & (lengths <= _UPPER)
    return {
        "minimum": float(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "maximum": float(np.max(lengths)),
        "unit_fraction": float(np.mean(unit)),
    }


source = phx.meshing.certify_cell_mesh(square_mesh(8), phx.SpatialCoordinateContract.si())
space, values, l2_error, nodal_error = solve(source)
initial = {
    "vertices": source.mesh.coordinates.shape[0],
    "cells": source.mesh.entity_set(2).count,
    "l2_error": l2_error,
    "nodal_error": nodal_error,
}
policy = phx.meshing.MeshAdaptationPolicy(
    phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D, maximum_passes=24
)
current = source
cycles = []
for cycle in range(CYCLES):
    metric, lp, normalization = normalized_metric(current, space, values)
    adapted = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            current, phx.meshing.MetricMeshAdaptation(metric), policy=policy
        )
    )
    if not adapted.target.audit.passed or adapted.metric is None:
        raise RuntimeError("Metric adaptation did not certify its target.")
    current = adapted.target
    space, values, l2_error, nodal_error = solve(current)
    anisotropy = cell_anisotropy(current.mesh)
    evidence = adapted.evidence
    cycles.append(
        {
            "cycle": cycle,
            "vertices": current.mesh.coordinates.shape[0],
            "cells": current.mesh.entity_set(2).count,
            "status": adapted.status.value,
            "passes": evidence.passes,
            "operations": {
                "splits": evidence.splits,
                "collapses": evidence.collapses,
                "flips": evidence.flips,
                "relocations": evidence.relocations,
            },
            "metric_complexity": normalization.final_complexity,
            "lp_indefinite_tensors": lp.indefinite_tensor_count,
            "gradation_iterations": normalization.gradation.iterations,
            "gradation_modified_vertices": normalization.gradation.modified_count,
            "metric_edge_lengths": edge_length_summary(current, adapted.metric),
            "requested_anisotropy_maximum": float(
                np.max(metric_anisotropy(adapted.metric.values))
            ),
            "achieved_anisotropy": {
                "median": float(np.median(anisotropy)),
                "maximum": float(np.max(anisotropy)),
            },
            "l2_error": l2_error,
            "nodal_error": nodal_error,
            "audit_passed": current.audit.passed,
        }
    )

final = cycles[-1]
errors = [initial["l2_error"], *(record["l2_error"] for record in cycles)]
if min(record["metric_edge_lengths"]["unit_fraction"] for record in cycles) < 0.9:
    raise RuntimeError("An adapted mesh is not close to a unit mesh of its metric.")
if final["achieved_anisotropy"]["maximum"] < 5.0:
    raise RuntimeError("Adaptation did not produce anisotropic cells along the layer.")
if any(later >= earlier for earlier, later in zip(errors, errors[1:])):
    raise RuntimeError("A metric adaptation cycle did not reduce the layer error.")
if final["l2_error"] >= 0.05 * initial["l2_error"]:
    raise RuntimeError("Metric adaptation did not resolve the internal layer.")
print(json.dumps({"layer": LAYER, "initial": initial, "cycles": cycles}, indent=2))
