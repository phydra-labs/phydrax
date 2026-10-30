#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._interpolation._tspline import TMesh2D
from phydrax.discretization import iga
from phydrax.geometry.brep._tspline import TSplineSurfacePatch


def _plan() -> iga.TSplineIsogeometricPlan:
    mesh = TMesh2D.tensor_product(
        tuple(Fraction(index, 4) for index in range(5)),
        tuple(Fraction(index, 4) for index in range(5)),
    )
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
        == (Fraction(0), Fraction(1, 4), Fraction(0), Fraction(1, 4))
    )
    return plan.refine((marked,)).target_plan


def test_curved_trimmed_ast_spline_source_solves_manufactured_h1_problem() -> None:
    plan = _plan()
    prepared = plan.prepare(numeric_version="asts-h1-manufactured")
    realization = prepared.basis_realization
    coefficient_points = jnp.asarray(
        [
            [float(anchor.local_knots.greville_parameter[0]),
             float(anchor.local_knots.greville_parameter[1])]
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
        second_u = realization.evaluate(
            uv, exact_coefficients, derivative=(2, 0)
        )
        second_v = realization.evaluate(
            uv, exact_coefficients, derivative=(0, 2)
        )
        return -(second_u + second_v).reshape(shape)

    forcing = phx.equations.coefficient(
        source,
        coefficient_id="asts2d-manufactured-source",
    )
    form = phx.equations.FiniteElementForm(
        "asts2d-manufactured-h1",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", forcing),
        ),
    )
    constraint = prepared.homogeneous_trace_constraint("u")
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=constraint,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="dense",
        ),
    )
    operator, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(operator, right_hand_side)
    recovered = compiled.expand(result.value)

    assert bool(jnp.all(result.successful))
    np.testing.assert_allclose(
        np.asarray(recovered), np.asarray(exact_coefficients), rtol=2.0e-8, atol=2.0e-10
    )
