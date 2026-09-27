#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Solve, mark, refine, and coarsen on the device, then commit one certified mesh.

A Poisson problem with a steep interior layer is solved on the masked capacity
layout of a device adaptive simplex epoch. Each cycle marks the cells with the
largest P1 gradient jump energy, refines them (conformity closure on device),
coarsens the calmest cells, and re-solves: no host round trip and no
recompilation inside the capacity bucket. The epoch is committed once into a
certified `MeshAdaptationResult` whose target solve agrees with the masked one.
"""

import json
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def source_term(points: Any) -> Any:
    x, y = points[..., 0], points[..., 1]
    radius = jnp.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return 200.0 * jnp.exp(-400.0 * (radius - 0.25) ** 2)


def masked_solve(plan: Any, mesh: Any) -> Any:
    system = phx.discretization.assemble_masked_finite_element(plan, mesh)
    load = jnp.where(mesh.vertex_active, source_term(mesh.coordinates), 0.0)
    rhs = jnp.where(system.boundary_dofs, 0.0, system.mass.mv(load))
    operator = phx.discretization.constrain_masked_dofs(
        system.stiffness, system.boundary_dofs
    )
    solved = phx.linalg.solve(phx.linalg.LinearSystem(operator), rhs)
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("The masked Poisson solve failed.")
    return solved.value


def cell_indicator(mesh: Any, values: Any) -> Any:
    """Squared P1 gradient magnitude times cell area (masked lanes are zero)."""
    corners = mesh.coordinates[mesh.cells]
    spans = corners[:, 1:] - corners[:, :1]
    area = 0.5 * (spans[:, 0, 0] * spans[:, 1, 1] - spans[:, 0, 1] * spans[:, 1, 0])
    safe = jnp.where(mesh.cell_active, area, 1.0)
    local = values[mesh.cells]
    rise = local[:, 1:] - local[:, :1]
    gradient = jnp.stack(
        (
            rise[:, 0] * spans[:, 1, 1] - rise[:, 1] * spans[:, 0, 1],
            rise[:, 1] * spans[:, 0, 0] - rise[:, 0] * spans[:, 1, 0],
        ),
        axis=1,
    ) / (2.0 * safe[:, None])
    energy = jnp.sum(gradient * gradient, axis=1) * area
    return jnp.where(mesh.cell_active, energy, 0.0)


axis = np.linspace(0.0, 1.0, 9)
points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
index = np.arange(81, dtype=np.int32).reshape((9, 9))
lower, right = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
upper, left = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
triangles = np.concatenate(
    (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
)
source = phx.meshing.certify_cell_mesh(
    phx.discretization.CellMesh.from_triangles(points, triangles),
    phx.SpatialCoordinateContract.si(),
)
policy = phx.meshing.MeshAdaptationPolicy(
    phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
    device_policy=phx.discretization.AdaptiveSimplexPolicy(
        vertex_capacity=4096, cell_capacity=8192
    ),
)
prepared = phx.meshing.prepare_adaptive_simplex(source, policy=policy)
layout, state = prepared.layout, prepared.state
plan = phx.discretization.MaskedFiniteElementPlan(state.mesh)
history = []
for cycle in range(5):
    values = masked_solve(plan, state.mesh)
    indicator = cell_indicator(state.mesh, values)
    active = state.mesh.cell_active
    threshold = 0.25 * jnp.max(indicator)
    refined = phx.discretization.refine_adaptive_simplex(
        layout, state, active & (indicator > threshold)
    )
    # Cells that stayed active through the refinement keep their slot and
    # indicator; the calmest of them are offered for coarsening.
    calm = refined.state.mesh.cell_active & active & (indicator < 1.0e-3 * threshold)
    coarsened = phx.discretization.coarsen_adaptive_simplex(layout, refined.state, calm)
    for report in (refined.report, coarsened.report):
        if bool(report.failed):
            raise RuntimeError(f"Device adaptation refused: status {int(report.status)}.")
    state = coarsened.state
    history.append(
        {
            "cycle": cycle,
            "bisections": int(refined.report.operations),
            "restored_cells": int(coarsened.report.operations),
            "active_cells": int(jnp.sum(state.mesh.cell_active)),
            "minimum_quality": float(refined.report.minimum_quality),
        }
    )

adapted = phx.meshing.commit_adaptive_simplex(prepared, state)
values = masked_solve(plan, state.mesh)
target = adapted.target.mesh
identifiers = np.asarray(state.mesh.vertex_ids)
slots = np.searchsorted(
    np.where(identifiers >= 0, identifiers, np.iinfo(np.int64).max),
    np.asarray(target.vertex_global_ids),
)
field = phx.discretization.FiniteElementFieldSpec(
    "u", phx.discretization.lagrange_element("triangle", 1)
)
space = phx.discretization.FiniteElementPlan(target, field).prepare()
boundary = space.boundary_dof_mask
compact = phx.linalg.solve(
    phx.linalg.LinearSystem(
        phx.discretization.constrain_masked_dofs(space.stiffness, boundary)
    ),
    jnp.where(boundary, 0.0, space.mass.mv(source_term(target.coordinates))),
).value
difference = float(jnp.max(jnp.abs(values[slots] - compact)))
if difference > 1.0e-9 * float(jnp.max(jnp.abs(compact))):
    raise RuntimeError("The masked solve differs from the committed compact solve.")
print(
    json.dumps(
        {
            "cycles": history,
            "status": adapted.status.value,
            "committed_cells": target.blocks[0].cell_count,
            "audit_passed": adapted.target.audit.passed,
            "masked_versus_committed_max_difference": difference,
        },
        indent=2,
    )
)
