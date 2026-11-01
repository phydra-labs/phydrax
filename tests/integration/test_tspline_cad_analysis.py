#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._interpolation._tspline import TMesh2D
from phydrax.discretization import iga
from phydrax.geometry.brep._tspline import TSplineSurfacePatch


# The assertions below compare coefficients to 2e-10; the solve must certify a
# residual whose error image (condition number ~40 here) stays below that.
_SOLVE_POLICY = phx.linalg.LinearSolvePolicy(
    tolerance=phx.linalg.TolerancePolicy(relative=1.0e-13, absolute=0.0)
)


def _plan(divisions: int = 4) -> iga.TSplineIsogeometricPlan:
    breaks = tuple(Fraction(index, divisions) for index in range(divisions + 1))
    mesh = TMesh2D.tensor_product(breaks, breaks)
    control_points = []
    for anchor in mesh.anchors:
        u, v = anchor.local_knots.greville_parameter
        control_points.append((u, v, Fraction(0)))
    exact_points = tuple(control_points)
    exact_weights = tuple(Fraction(1) for _ in mesh.anchors)
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in row] for row in exact_points]),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    plan = iga.TSplineIsogeometricPlan(
        patch,
        iga.TSplineH1FieldSpec("u"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(4),
    )
    marked = next(
        cell.cell_id
        for cell in mesh.cells
        if cell.parameter_bounds
        == (Fraction(0), Fraction(1, divisions), Fraction(0), Fraction(1, divisions))
    )
    return plan.refine((marked,)).target_plan


def _solve_poisson(
    prepared: Any, source: Callable[[Any, Any], Any], identifier: str
) -> tuple[Any, Any]:
    form = phx.equations.FiniteElementForm(
        identifier,
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(source, coefficient_id=f"{identifier}-source"),
            ),
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
    result = phx.linalg.solve(operator, right_hand_side, policy=_SOLVE_POLICY)
    return result, compiled.expand(result.value)


def test_curved_trimmed_ast_spline_source_solves_manufactured_h1_problem() -> None:
    plan = _plan()
    prepared = plan.prepare(numeric_version="asts-h1-manufactured")
    realization = prepared.basis_realization
    coefficient_points = jnp.asarray(
        [
            [
                float(anchor.local_knots.greville_parameter[0]),
                float(anchor.local_knots.greville_parameter[1]),
            ]
            for anchor in plan.geometry.mesh.anchors
        ],
        dtype=jnp.float64,
    )
    exact_coefficients = (
        coefficient_points[:, 0]
        * (1.0 - coefficient_points[:, 0])
        * coefficient_points[:, 1]
        * (1.0 - coefficient_points[:, 1])
    )

    def source(points: Any, args: Any) -> Any:
        del args
        shape = points.shape[:-1]
        uv = points[..., :2].reshape((-1, 2))
        second_u = realization.evaluate(uv, exact_coefficients, derivative=(2, 0))
        second_v = realization.evaluate(uv, exact_coefficients, derivative=(0, 2))
        return -(second_u + second_v).reshape(shape)

    result, recovered = _solve_poisson(prepared, source, "asts2d-manufactured-h1")

    assert bool(jnp.all(result.successful))
    np.testing.assert_allclose(
        np.asarray(recovered), np.asarray(exact_coefficients), rtol=2.0e-8, atol=2.0e-10
    )


def _sine_l2_error(divisions: int) -> float:
    plan = _plan(divisions)
    prepared = plan.prepare(numeric_version=f"asts-h1-sine-{divisions}")

    def source(points: Any, args: Any) -> Any:
        del args
        x, y = points[..., 0], points[..., 1]
        return 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    result, solution = _solve_poisson(prepared, source, f"asts2d-sine-h1-{divisions}")
    assert bool(jnp.all(result.successful))
    # Midpoint samples of a 32 x 32 grid: an L2 estimate resolving each cell.
    axis = (np.arange(32, dtype=np.float64) + 0.5) / 32.0
    samples = np.asarray([(u, v) for u in axis for v in axis], dtype=np.float64)
    route = prepared.prepare_query_route(samples)
    values = np.asarray(prepared.evaluate_field_route("u", route, solution))
    exact = np.sin(np.pi * samples[:, 0]) * np.sin(np.pi * samples[:, 1])
    return float(np.sqrt(np.mean((values - exact) ** 2)))


def test_locally_refined_ast_spline_h1_solution_converges_at_bicubic_rate() -> None:
    coarse = _sine_l2_error(4)
    fine = _sine_l2_error(8)
    rate = math.log2(coarse / fine)

    # Bicubic H1-conforming Galerkin: L2 error O(h^4).
    assert fine < coarse
    assert rate >= 3.5, (coarse, fine, rate)
