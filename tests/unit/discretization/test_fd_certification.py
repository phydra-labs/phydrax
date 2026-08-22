#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _bounded_operator(resolution):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(resolution),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    request = phx.discretization.DerivativeRequest(
        "dxx",
        grid,
        "x",
        derivative_order=2,
        accuracy_order=2,
    )
    discretization = phx.discretization.FiniteDifferencePlan(
        grid,
        (request,),
        field_name="u",
    ).prepare()
    return grid, discretization.operator("dxx")


def _bounded_second_derivative(resolution):
    grid, operator = _bounded_operator(resolution)
    return phx.equations.ManufacturedSpatialOperator(
        grid,
        operator,
        operator_id=f"bounded-dxx-{resolution}",
    )


def test_prepared_stencil_operator_carries_matrix_free_evidence_reports():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                257,
                endpoint=False,
                periodic=True,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.periodic_finite_difference(
        grid,
        accuracy_order=4,
    )
    operator = discretization.operator("d_x_1")

    assert operator.consistency_report.passed
    assert operator.consistency_report.minimum_accuracy_order == 4
    assert operator.consistency_report.failed_rows == ()
    assert operator.adjoint_report.passed
    assert operator.conservation_report.conservative
    assert operator.conservation_report.constant_state_residual < 1e-12
    assert operator.conservation_report.global_balance_residual < 1e-12


def test_bounded_derivative_does_not_claim_unverified_global_conservation():
    _, operator = _bounded_operator(17)
    report = operator.conservation_report

    assert report.global_balance_residual is None
    assert report.conservative is None
    assert report.constant_state_residual < 1e-12


def test_unknown_stability_evidence_cannot_carry_a_residual():
    with pytest.raises(ValueError, match="Unknown evidence"):
        phx.discretization.FDStabilityReport(
            "semibounded",
            residual=0.0,
            tolerance=1e-12,
            assumptions=(),
            evidence="unknown",
            subject_id="operator",
        )


def test_manufactured_case_derives_forcing_by_time_differentiation():
    prepared = _bounded_second_derivative(17)
    case = phx.equations.ManufacturedPDECase(
        lambda time, points, args: (1.0 + time) * jnp.exp(points[:, 0]),
        lambda time, points, args: (1.0 + time) * jnp.exp(points[:, 0]),
        case_id="transient-exponential",
    )

    forcing = case.forcing(prepared.grid, 0.5)

    np.testing.assert_allclose(
        forcing,
        -0.5 * jnp.exp(prepared.grid.axes[0].nodes),
        rtol=2e-12,
        atol=2e-12,
    )


def test_manufactured_convergence_separates_interior_and_boundary_rates():
    case = phx.equations.ManufacturedPDECase(
        lambda time, points, args: jnp.exp(points[:, 0]),
        lambda time, points, args: jnp.exp(points[:, 0]),
        case_id="spatial-exponential",
    )
    plan = phx.equations.ManufacturedConvergencePlan(
        (17, 33, 65),
        _bounded_second_derivative,
        expected_total_order=2.0,
        expected_interior_order=2.0,
        expected_boundary_order=2.0,
        rate_tolerance=0.35,
        plan_id="bounded-dxx-convergence",
    )

    result = plan.run(case)

    assert result.total_passed
    assert result.interior_passed
    assert result.boundary_passed
    assert result.total_errors[-1] < result.total_errors[0]
    assert result.interior_errors[-1] < result.interior_errors[0]
    assert result.boundary_errors[-1] < result.boundary_errors[0]
