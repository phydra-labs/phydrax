#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adaptive tensor-hp Poisson solve on a refined quadrilateral epoch."""

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def run() -> dict[str, float | int | bool]:
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "quadrilateral",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )
    topology, geometry = phx.discretization.fem.initial_finite_element_hp_topology(
        mesh,
        2,
        16,
    )
    estimate = phx.discretization.fem.FiniteElementHPErrorEstimate(
        topology,
        jnp.asarray((1.0,) + (0.0,) * 15),
        smoothness=jnp.ones((16, 2)),
    )
    decision = phx.discretization.fem.finite_element_hp_decision(
        topology,
        estimate,
        maximum_active_cells=4,
    )
    marked_ids = np.asarray(topology.cell_global_ids)[np.asarray(decision.refine)]
    refined = phx.discretization.fem.refine_tensor_hp_cells(
        topology,
        geometry,
        marked_ids,
        target_degrees=jnp.asarray(((3, 3),), dtype=jnp.int32),
    )
    epoch = phx.discretization.fem.prepare_finite_element_hp_epoch(
        refined.topology,
        refined.geometry,
        "u",
    )

    def exact(points):
        return points[..., 0] ** 3 + points[..., 1] ** 3

    source = phx.equations.coefficient(
        lambda points, args: -6.0 * (points[..., 0] + points[..., 1]),
        coefficient_id="adaptive-hp-poisson-source",
    )
    form = phx.equations.FiniteElementForm(
        "adaptive-hp-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source),
        ),
    )
    constraint = phx.discretization.dirichlet_constraint(epoch.discretization, "u")
    compiled = phx.equations.compile_finite_element_problem(
        form,
        epoch.discretization,
        constraint=constraint,
        dirichlet_values=lambda points: exact(points),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )
    operator, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(operator, right_hand_side)
    solution = compiled.expand(result.value)
    expected = exact(epoch.discretization.dof_maps[0].dof_coordinates)
    error = float(jnp.max(jnp.abs(solution - expected)))
    successful = bool(jnp.all(result.successful))
    if not successful or error > 5.0e-10:
        raise RuntimeError("Adaptive tensor-hp Poisson qualification failed.")
    return {
        "initial_cells": 1,
        "active_cells": refined.topology.active_count,
        "degree": 3,
        "global_dofs": epoch.discretization.dof_maps[0].global_dof_count,
        "maximum_error": error,
        "successful": successful,
    }


if __name__ == "__main__":
    print(run())
