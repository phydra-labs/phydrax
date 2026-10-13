#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(points: Any = 33, *, lower: Any = 0.0, upper: Any = 1.0) -> Any:
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(points),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[lower], [upper]]))


def test_sbp_sat_scenario_1() -> None:
    for order in [2, 4, 6, 8]:
        grid = _grid(41)
        sbp = phx.discretization.SBPDerivativePlan(
            grid,
            "x",
            interior_order=order,
        ).prepare()
        x = grid.axes[0].nodes
        degree = order // 2
        values = x**degree
        exact = degree * x ** (degree - 1) if degree > 0 else jnp.zeros_like(x)

        derivative = sbp.operator.mv(values)

        assert sbp.stability_report.passed
        assert sbp.operator.consistency_report.passed
        assert sbp.operator.stencil_set.interior_accuracy_order == order
        assert sbp.operator.stencil_set.closure_accuracy_order == order // 2
        assert jnp.all(sbp.norm_weights > 0.0)
        assert jnp.max(jnp.abs(sbp.identity_residual())) < 5e-11
        np.testing.assert_allclose(derivative, exact, rtol=2e-8, atol=2e-8)
    grid = _grid(65)
    sbp = phx.discretization.SBPDerivativePlan(
        grid,
        "x",
        interior_order=6,
    ).prepare()
    x = grid.axes[0].nodes
    state = jnp.sin(jnp.pi * x)
    second = sbp.second_derivative

    constant = second.mv(jnp.ones_like(state))
    action = second.mv(state)
    energy_rate = jnp.sum(sbp.norm_weights * state * action)
    left = jnp.vdot(state, second.mv(state))
    right = jnp.vdot(second.transpose_mv(state), state)

    np.testing.assert_allclose(constant, 0.0, rtol=0.0, atol=2e-10)
    assert energy_rate < 0.0
    np.testing.assert_allclose(left, right, rtol=2e-11, atol=2e-11)
    grid = _grid(49)
    sbp = phx.discretization.SBPDerivativePlan(
        grid,
        "x",
        interior_order=4,
    ).prepare()
    sat = phx.discretization.SATBoundaryPlan.advection_inflow(sbp, 2.0)
    x = grid.axes[0].nodes
    state = 0.7 + jnp.sin(3.0 * jnp.pi * x)

    rhs = -2.0 * sbp.operator.mv(state) + sat.correction(state, 0.0, 0.0)
    energy_rate = 2.0 * jnp.sum(sbp.norm_weights * state * rhs)

    assert sat.stability_report.passed
    assert energy_rate <= 2e-10


def test_sbp_sat_scenario_2() -> None:
    left = phx.discretization.SBPDerivativePlan(
        _grid(33, lower=0.0, upper=0.5),
        "x",
        interior_order=4,
    ).prepare()
    right = phx.discretization.SBPDerivativePlan(
        _grid(33, lower=0.5, upper=1.0),
        "x",
        interior_order=4,
    ).prepare()
    left_state = jnp.sin(jnp.pi * left.grid.axes[0].nodes)
    right_state = 0.3 * jnp.sin(jnp.pi * (1.0 - right.grid.axes[0].nodes))

    rates = []
    for flux in ("central", "upwind"):
        interface = phx.discretization.SATInterfacePlan(
            left,
            right,
            1.0,
            flux=flux,
        )
        left_sat, right_sat = interface.corrections(left_state, right_state)
        left_rhs = -left.operator.mv(left_state) + left_sat
        right_rhs = -right.operator.mv(right_state) + right_sat
        rates.append(
            2.0
            * (
                jnp.sum(left.norm_weights * left_state * left_rhs)
                + jnp.sum(right.norm_weights * right_state * right_rhs)
            )
        )
        assert interface.stability_report.passed

    np.testing.assert_allclose(rates[0], 0.0, rtol=0.0, atol=2e-10)
    assert rates[1] <= rates[0] + 2e-10
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(33),
            phx.discretization.UniformCellAxisSpec(5),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, -1.0], [1.0, 2.0]]))
    sbp = phx.discretization.SBPDerivativePlan(
        grid,
        "x",
        interior_order=4,
    ).prepare()
    x = grid.axes[0].nodes[:, None]
    values = jnp.broadcast_to(x**2, grid.shape)

    derivative = sbp.operator.mv(values)

    assert derivative.shape == grid.shape
    assert sbp.norm_weights.shape == grid.shape
    np.testing.assert_allclose(
        derivative,
        jnp.broadcast_to(2.0 * x, grid.shape),
        rtol=2e-10,
        atol=2e-10,
    )
    with pytest.raises(ValueError, match="requires at least"):
        phx.discretization.SBPDerivativePlan(
            _grid(15),
            "x",
            interior_order=8,
        ).prepare()


def _dense_sbp_oracle(count: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent dense polynomial-moment construction of the diagonal-norm family."""
    norms = {
        2: (1 / 2,),
        4: (17 / 48, 59 / 48, 43 / 48, 49 / 48),
        6: (
            13649 / 43200,
            12013 / 8640,
            2711 / 4320,
            5359 / 4320,
            7877 / 8640,
            43801 / 43200,
        ),
        8: (
            1498139 / 5080320,
            1107307 / 725760,
            20761 / 80640,
            1304999 / 725760,
            299527 / 725760,
            103097 / 80640,
            670091 / 725760,
            5127739 / 5080320,
        ),
    }
    boundary = len(norms[order])
    norm = np.ones(count, dtype=np.float64)
    norm[:boundary] = norms[order]
    norm[-boundary:] = norms[order][::-1]
    offsets = np.arange(-order // 2, order // 2 + 1, dtype=np.float64)
    target = np.zeros(order + 1, dtype=np.float64)
    target[1] = 1.0
    central = np.linalg.solve(offsets[None, :] ** np.arange(order + 1)[:, None], target)
    q = np.zeros((count, count), dtype=np.float64)
    for row in range(boundary, count - boundary):
        q[row, row - order // 2 : row + order // 2 + 1] = central
    q[:boundary, boundary:] = -q[boundary:, :boundary].T
    q[-boundary:, :-boundary] = -q[:-boundary, -boundary:].T
    q[0, 0], q[-1, -1] = -0.5, 0.5
    pairs = [(i, j) for i in range(boundary) for j in range(i + 1, boundary)]
    if pairs:
        basis = np.zeros((len(pairs), count, count), dtype=np.float64)
        for lane, (i, j) in enumerate(pairs):
            basis[lane, i, j], basis[lane, j, i] = 1.0, -1.0
        monomials = (
            np.arange(count, dtype=np.float64)[None, :]
            ** np.arange(order // 2 + 1)[:, None]
        )
        equations = np.stack(
            [(item @ monomials.T)[:boundary].ravel() for item in basis], axis=1
        )
        derivatives = np.zeros_like(monomials)
        for degree in range(1, order // 2 + 1):
            derivatives[degree] = degree * monomials[degree - 1]
        rhs = (norm[:, None] * derivatives.T - q @ monomials.T)[:boundary].ravel()
        solution = np.linalg.lstsq(equations, rhs, rcond=None)[0]
        for coefficient, item in zip(solution, basis, strict=True):
            q += coefficient * item
    q[-boundary:, -boundary:] = -q[:boundary, :boundary][::-1, ::-1]
    return q / norm[:, None], norm


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_bounded_sbp_dense_oracle(order: Any) -> None:
    count = 41
    sbp = phx.discretization.SBPDerivativePlan(
        _grid(count), "x", interior_order=order
    ).prepare()
    expected, norm = _dense_sbp_oracle(count, order)
    delta = 1.0 / (count - 1)
    actual = np.asarray(sbp.operator._materialize())
    np.testing.assert_allclose(actual, expected / delta, rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(sbp.axis_norm_weights, delta * norm, rtol=1e-14)
    boundary = np.zeros_like(actual)
    boundary[0, 0], boundary[-1, -1] = -1.0, 1.0
    residual = (
        delta * norm[:, None] * actual + actual.T * (delta * norm)[None, :] - boundary
    )
    np.testing.assert_allclose(residual, 0.0, atol=5e-12)
    assert sbp.stability_report.passed
    assert sbp.operator.stencil_set.stencil.indices.shape[1] <= order + len(norm[:order])
