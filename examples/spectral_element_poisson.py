#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run() -> dict[str, float | int | bool]:
    coordinates = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    block = phx.discretization.CellBlock(
        "quadrilaterals",
        "quadrilateral",
        jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
    )
    mesh = phx.discretization.CellMesh(coordinates, (block,))
    element = phx.discretization.fem.ReferenceNodalFamily(
        "quadrilateral", 3
    ).finite_element()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("u", element),
    ).prepare()

    def exact_solution(points):
        return points[..., 0] ** 3 + points[..., 1] ** 3

    form = phx.equations.FiniteElementForm(
        "spectral-element-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    lambda points, args: -6.0 * (points[..., 0] + points[..., 1]),
                    coefficient_id="manufactured-poisson-source",
                ),
            ),
        ),
    )
    constraint = phx.discretization.dirichlet_constraint(discretization, "u")
    dense = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: exact_solution(points),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free", local_kernel="dense"
        ),
    )
    factorized = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: exact_solution(points),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free", local_kernel="sum_factorized"
        ),
    )
    state = discretization.project(
        "u",
        lambda points, args: exact_solution(points),
    )
    dense_residual = dense.full_residual(state)
    factorized_residual = factorized.full_residual(state)
    defect = jnp.max(jnp.abs(dense_residual - factorized_residual))
    system, right_hand_side = factorized.linear_system()
    result = phx.linalg.solve(system, right_hand_side)
    solution = factorized.expand(result.value)
    expected = exact_solution(discretization.dof_maps[0].dof_coordinates)
    error = jnp.max(jnp.abs(solution - expected))
    successful = bool(jnp.all(result.successful))
    if not successful or float(defect) > 2.0e-11 or float(error) > 2.0e-10:
        raise RuntimeError("Spectral-element Poisson qualification failed.")
    return {
        "degree": 3,
        "global_dofs": discretization.dof_maps[0].global_dof_count,
        "dense_factorized_defect": float(defect),
        "maximum_error": float(error),
        "successful": successful,
    }


if __name__ == "__main__":
    print(run())
