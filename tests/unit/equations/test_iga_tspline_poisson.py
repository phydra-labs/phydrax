#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import iga
from phydrax.geometry.brep import TSplineSurfacePatch


def _exact_solution(uv: jax.Array) -> jax.Array:
    u, v = uv[..., 0], uv[..., 1]
    return u * (1.0 - u) * v * (1.0 - v)


def _curved_trimmed_refinement() -> tuple[iga.TSplineIsogeometricPlan, iga.TMesh2D]:
    breaks = tuple(Fraction(index, 4) for index in range(5))
    source_mesh = iga.TMesh2D.tensor_product(breaks, breaks)
    greville_points = tuple(
        anchor.local_knots.greville_parameter for anchor in source_mesh.anchors
    )
    exact_points = tuple((u, v, u * v) for u, v in greville_points)
    numeric_points = np.asarray(
        [[float(value) for value in point] for point in exact_points], dtype=np.float64
    )
    exact_weights = tuple(Fraction(1) for _ in source_mesh.anchors)
    carrier = TSplineSurfacePatch(
        source_mesh,
        numeric_points,
        np.ones((source_mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    field = iga.TSplineH1FieldSpec(
        "u", weights_from_geometry=True, anchor_count=source_mesh.coefficient_count
    )
    plan = iga.TSplineIsogeometricPlan(
        carrier,
        field,
        trim=trim,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(12),
    )
    refinement = plan.refine((source_mesh.cells[5].cell_id,))
    return refinement.target_plan, source_mesh


def _manufactured_source(points: jax.Array, args: object) -> jax.Array:
    del args
    parameter_points = points[..., :2]
    original_shape = parameter_points.shape[:-1]
    flat = parameter_points.reshape((-1, 2))

    def source_at(uv: jax.Array) -> jax.Array:
        u, v = uv[0], uv[1]
        determinant = 1.0 + u * u + v * v

        def flux(coordinates: jax.Array) -> jax.Array:
            x, y = coordinates[0], coordinates[1]
            metric_determinant = 1.0 + x * x + y * y
            local_inverse = (
                jnp.asarray(((1.0 + x * x, -x * y), (-x * y, 1.0 + y * y)))
                / metric_determinant
            )
            local_gradient = jax.grad(_exact_solution)(coordinates)
            return jnp.sqrt(metric_determinant) * (local_inverse @ local_gradient)

        return -jnp.trace(jax.jacfwd(flux)(uv)) / jnp.sqrt(determinant)

    return jax.vmap(source_at)(flat).reshape(original_shape)


def test_locally_refined_curved_trimmed_tspline_solves_manufactured_surface_pde() -> None:
    plan, source_mesh = _curved_trimmed_refinement()
    prepared = plan.prepare(numeric_version="manufactured-tspline-surface")
    mesh = plan.geometry.mesh
    assert mesh.t_junction_count > 0
    full_tensor_count = prod(len(axis) - 1 for axis in mesh.axis_breaks)
    assert mesh.cell_count < full_tensor_count
    assert plan.geometry.surface_patch.source_mesh.mesh_id == source_mesh.mesh_id

    collocation_axis = np.linspace(0.037, 0.963, 17)
    collocation = np.asarray(
        [(u, v) for u in collocation_axis for v in collocation_axis], dtype=np.float64
    )
    collocation_route = prepared.prepare_query_route(collocation)
    polynomial_basis = prepared.basis_realization.evaluate(
        jnp.asarray(collocation),
        jnp.eye(mesh.coefficient_count, dtype=jnp.float64),
        cell_ids=collocation_route.cell_ids,
    )
    expected_coefficients, _, _, _ = np.linalg.lstsq(
        np.asarray(polynomial_basis),
        np.asarray(_exact_solution(jnp.asarray(collocation))),
        rcond=None,
    )
    check_points = np.asarray(
        ((0.11, 0.19), (0.23, 0.77), (0.48, 0.36), (0.74, 0.58), (0.91, 0.82)),
        dtype=np.float64,
    )
    check_route = prepared.prepare_query_route(check_points)
    represented = prepared.evaluate_field_route(
        "u", check_route, jnp.asarray(expected_coefficients)
    )
    np.testing.assert_allclose(
        represented,
        _exact_solution(jnp.asarray(check_points)),
        rtol=2.0e-10,
        atol=2.0e-11,
    )

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
    residual_bound = 2.0e-9 * jnp.maximum(rhs_norm, 1.0)
    solution_route = prepared.prepare_query_route(check_points)
    solved_values = prepared.evaluate_field_route("u", solution_route, solution)

    assert bool(jnp.all(result.successful))
    assert float(residual_norm) <= float(residual_bound)
    np.testing.assert_allclose(
        solution,
        expected_coefficients,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        solved_values,
        _exact_solution(jnp.asarray(check_points)),
        rtol=2.0e-7,
        atol=2.0e-8,
    )
