#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

import phydrax as phx
from phydrax.operators.differential import laplacian, partial_n
from phydrax.solver._kfac_derivative_requests import trace_derivative_requests


def test_trace_derivative_requests_keeps_laplacian_contracted():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @domain.Function("x")
    def u(x):
        return x[0] ** 2 + x[0] * x[1]

    condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda field: (
            laplacian(field, var="x")
            + partial_n(field, var="x", axis=1, order=1)
        ),
    )
    requests = trace_derivative_requests(condition.residual, {"u": u})

    assert len(requests) == 2
    assert any(request.contracted_laplacian for request in requests)
    assert any(request.axes == (1,) for request in requests)
    assert all(request.order <= 2 for request in requests)


def test_trace_derivative_requests_rejects_derivatives_above_order_two():
    domain = phx.domain.Interval1d(0.0, 1.0)

    @domain.Function("x")
    def u(x):
        return x[0] ** 3

    condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda field: partial_n(
            field,
            var="x",
            axis=0,
            order=3,
        ),
    )
    with pytest.raises(ValueError, match="through order two"):
        trace_derivative_requests(condition.residual, {"u": u})


def test_trace_derivative_requests_rejects_nested_laplacians():
    domain = phx.domain.Interval1d(0.0, 1.0)

    @domain.Function("x")
    def u(x):
        return x[0] ** 4

    condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda field: laplacian(
            laplacian(field, var="x"),
            var="x",
        ),
    )
    with pytest.raises(ValueError, match="order 4"):
        trace_derivative_requests(condition.residual, {"u": u})
