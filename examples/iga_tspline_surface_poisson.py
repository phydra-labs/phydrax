#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import iga
from phydrax.geometry.brep import TSplineSurfacePatch


def _exact_solution(uv: jax.Array) -> jax.Array:
    u, v = uv[..., 0], uv[..., 1]
    return u * (1.0 - u) * v * (1.0 - v)


def _manufactured_source(points: jax.Array, args: object) -> jax.Array:
    del args
    uv = points[..., :2].reshape((-1, 2))

    def source_at(parameter: jax.Array) -> jax.Array:
        u, v = parameter
        determinant = 1.0 + u * u + v * v

        def flux(coordinates: jax.Array) -> jax.Array:
            x, y = coordinates
            metric_determinant = 1.0 + x * x + y * y
            inverse_metric = (
                jnp.asarray(((1.0 + x * x, -x * y), (-x * y, 1.0 + y * y)))
                / metric_determinant
            )
            return jnp.sqrt(metric_determinant) * (
                inverse_metric @ jax.grad(_exact_solution)(coordinates)
            )

        return -jnp.trace(jax.jacfwd(flux)(parameter)) / jnp.sqrt(determinant)

    return jax.vmap(source_at)(uv).reshape(points.shape[:-1])


def main() -> None:
    breaks = tuple(Fraction(index, 4) for index in range(5))
    source_mesh = iga.TMesh2D.tensor_product(breaks, breaks)
    greville_points = tuple(
        anchor.local_knots.greville_parameter for anchor in source_mesh.anchors
    )
    exact_control_points = tuple((u, v, u * v) for u, v in greville_points)
    control_points = np.asarray(
        [[float(value) for value in row] for row in exact_control_points],
        dtype=np.float64,
    )
    exact_weights = tuple(Fraction(1) for _ in source_mesh.anchors)
    surface = TSplineSurfacePatch(
        source_mesh,
        control_points,
        np.ones((source_mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_control_points,
        exact_weights=exact_weights,
    )
    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    plan = iga.TSplineIsogeometricPlan(
        iga.TSplineGeometryState(surface),
        iga.TSplineH1FieldSpec(
            "u", weights_from_geometry=True, anchor_count=source_mesh.coefficient_count
        ),
        trim=trim,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(6),
    )
    refinement = plan.refine((source_mesh.cells[5].cell_id,))
    target_plan = refinement.target_plan
    target_mesh = target_plan.geometry.mesh
    full_tensor_count = (len(target_mesh.axis_breaks[0]) - 1) * (
        len(target_mesh.axis_breaks[1]) - 1
    )
    if target_mesh.t_junction_count == 0 or target_mesh.cell_count >= full_tensor_count:
        raise RuntimeError("ASTS refinement did not retain a local non-tensor T-mesh.")
    if target_plan.geometry.surface_patch.source_mesh.mesh_id != source_mesh.mesh_id:
        raise RuntimeError("Refined CAD geometry lost its source T-mesh correspondence.")

    prepared = target_plan.prepare(numeric_version="curved-tspline-poisson")
    source = phx.equations.coefficient(
        _manufactured_source,
        coefficient_id="iga-tspline-curved-surface-poisson-source",
    )
    form = phx.equations.FiniteElementForm(
        "iga-tspline-curved-surface-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=prepared.homogeneous_trace_constraint("u"),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="dense",
        ),
    )
    operator, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(operator, right_hand_side)
    solution = compiled.expand(result.value)
    residual = compiled.residual(result.value)
    residual_norm = jnp.linalg.norm(residual)
    rhs_norm = jnp.linalg.norm(right_hand_side)
    check_points = jnp.asarray(
        ((0.11, 0.19), (0.23, 0.77), (0.48, 0.36), (0.74, 0.58), (0.91, 0.82))
    )
    route = prepared.prepare_query_route(check_points)
    solved_values = prepared.evaluate_field_route("u", route, solution)
    solution_error = jnp.max(jnp.abs(solved_values - _exact_solution(check_points)))
    residual_bound = 2.0e-8 * jnp.maximum(rhs_norm, 1.0)
    if not bool(jnp.all(result.successful)) or float(residual_norm) > float(
        residual_bound
    ):
        raise RuntimeError(
            "The locally refined T-spline Poisson solve failed its residual gate."
        )
    print(
        json.dumps(
            {
                "basis_certificate_id": target_plan.geometry.surface_patch.basis_certificate.certificate_id,
                "cell_count": target_mesh.cell_count,
                "exact_geometry_certificate_id": target_plan.exact_geometry_certificate.certificate_id,
                "exact_map_passed": target_plan.exact_geometry_certificate.passed,
                "field_error_at_check_points": float(solution_error),
                "marked_cell_ids": list(refinement.basis_refinement.marked_cell_ids),
                "mesh_id": target_mesh.mesh_id,
                "numeric_geometry_certificate_id": target_plan.geometry_certificate.certificate_id,
                "numeric_map_passed": target_plan.geometry_certificate.passed,
                "normalized_free_residual": float(
                    residual_norm / jnp.maximum(rhs_norm, 1.0)
                ),
                "source_mesh_id": source_mesh.mesh_id,
                "source_definition_id": surface.source_definition_id,
                "source_to_target_transfer_id": refinement.basis_refinement.transfer_plan.plan_id,
                "t_junction_count": target_mesh.t_junction_count,
                "target_tensor_cell_count": full_tensor_count,
                "successful": bool(jnp.all(result.successful)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
