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

    requests = trace_derivative_requests(
        lambda functions: (
            laplacian(functions["u"], var="x")
            + partial_n(functions["u"], var="x", axis=1, order=1)
        ),
        {"u": u},
    )

    assert len(requests) == 2
    assert any(request.contracted_laplacian for request in requests)
    assert any(request.axes == (1,) for request in requests)
    assert all(request.order <= 2 for request in requests)


def test_trace_derivative_requests_rejects_derivatives_above_order_two():
    domain = phx.domain.Interval1d(0.0, 1.0)

    @domain.Function("x")
    def u(x):
        return x[0] ** 3

    with pytest.raises(ValueError, match="through order two"):
        trace_derivative_requests(
            lambda functions: partial_n(
                functions["u"],
                var="x",
                axis=0,
                order=3,
            ),
            {"u": u},
        )


def test_trace_derivative_requests_rejects_nested_laplacians():
    domain = phx.domain.Interval1d(0.0, 1.0)

    @domain.Function("x")
    def u(x):
        return x[0] ** 4

    with pytest.raises(ValueError, match="order 4"):
        trace_derivative_requests(
            lambda functions: laplacian(
                laplacian(functions["u"], var="x"),
                var="x",
            ),
            {"u": u},
        )
