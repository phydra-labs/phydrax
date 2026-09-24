#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.activations import squared_relu
from phydrax.operators.differential import (
    laplacian,
    partial_n,
    plan_derivative_execution,
    trace_derivative_requests,
)


def _mlp_field(domain, activation, *, depth=2, final_activation=None):
    network = phx.nn.models.MLP(
        in_size="scalar",
        out_size="scalar",
        width_size=8,
        depth=depth,
        activation=activation,
        final_activation=final_activation,
        key=jr.key(0),
    )
    return domain.Model("x")(network)


def _laplacian_residual(fields):
    return laplacian(fields["u"], var="x")


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
            laplacian(field, var="x") + partial_n(field, var="x", axis=1, order=1)
        ),
    )
    requests = trace_derivative_requests(condition.residual, {"u": u})

    assert len(requests) == 2
    assert any(request.contracted_laplacian for request in requests)
    assert any(request.axes == (1,) for request in requests)
    assert all(request.order <= 2 for request in requests)


def test_trace_derivative_requests_retains_high_order_for_generic_planning():
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
    requests = trace_derivative_requests(condition.residual, {"u": u})

    assert tuple(request.order for request in requests) == (1, 2, 3)
    assert plan_derivative_execution(requests).strategy == "jvp"


def test_trace_derivative_requests_retains_nested_laplacians():
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
    requests = trace_derivative_requests(condition.residual, {"u": u})

    assert tuple(request.order for request in requests) == (2, 4)
    assert plan_derivative_execution(requests).strategy == "jvp"


def test_degree_bound_algebra_decides_degeneracy_through_planning():
    domain = phx.domain.Interval1d(0.0, 1.0)
    u = _mlp_field(domain, jax.nn.relu)

    # Affine images of a piecewise-linear field keep a vanishing Laplacian.
    for field in (u, 2.0 * u + 1.0, u - u):
        with pytest.raises(ValueError, match="regularity-degenerate"):
            trace_derivative_requests(_laplacian_residual, {"u": field})
    # A product of piecewise-linear fields has quadratic pieces.
    (request,) = trace_derivative_requests(_laplacian_residual, {"u": u * u})
    assert request.admission.level(phx.DerivativeSurface.INPUT) is (
        phx.GradientLevel.ALMOST_EVERYWHERE
    )
    assert request.admission.conditions == ("singular-part-ignored",)


def test_accumulated_order_bounds_nested_derivatives():
    domain = phx.domain.Interval1d(0.0, 1.0)
    u = _mlp_field(domain, squared_relu, depth=1)

    (request,) = trace_derivative_requests(_laplacian_residual, {"u": u})
    assert request.order == 2
    assert request.admission.conditions == ()
    with pytest.raises(ValueError, match="Order-4 derivative.*regularity-degenerate"):
        trace_derivative_requests(
            lambda fields: laplacian(_laplacian_residual(fields), var="x"),
            {"u": u},
        )


def test_owner_authority_governs_undeclared_and_almost_everywhere_regularity():
    domain = phx.domain.Interval1d(0.0, 1.0)
    weight = domain.Function("x")(lambda x: x * (1.0 - x))
    trial = weight * _mlp_field(domain, jax.numpy.tanh)
    surrogate = phx.ComponentAuthority.SURROGATE
    exploratory = phx.RegularityPolicy(allow_undeclared=True)

    (direct,) = trace_derivative_requests(_laplacian_residual, {"u": trial})
    assert direct.admission.conditions == ("regularity-undeclared",)
    (admitted,) = trace_derivative_requests(
        _laplacian_residual, {"u": trial}, authority=surrogate, policy=exploratory
    )
    assert admitted.admission.conditions == ("regularity-undeclared",)
    for authority, policy in (
        (surrogate, None),
        (phx.ComponentAuthority.MODEL, exploratory),
    ):
        with pytest.raises(ValueError, match="regularity-undeclared"):
            trace_derivative_requests(
                _laplacian_residual, {"u": trial}, authority=authority, policy=policy
            )

    piecewise = _mlp_field(domain, jax.nn.relu, final_activation=jax.numpy.tanh)
    (direct,) = trace_derivative_requests(_laplacian_residual, {"u": piecewise})
    assert direct.admission.conditions == ("singular-part-ignored",)
    with pytest.raises(ValueError, match="almost-everywhere-not-allowed"):
        trace_derivative_requests(
            _laplacian_residual, {"u": piecewise}, authority=surrogate
        )


def test_derivatives_along_independent_variables_need_no_regularity():
    domain = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    u = _mlp_field(domain, jax.nn.relu)

    requests = trace_derivative_requests(
        lambda fields: partial_n(fields["u"], var="t", order=2),
        {"u": u},
        authority=phx.ComponentAuthority.MODEL,
    )
    assert requests
    assert all(request.admission is None for request in requests)


def test_direct_partial_n_rejects_only_proven_degeneracy():
    domain = phx.domain.Interval1d(0.0, 1.0)
    u = _mlp_field(domain, jax.nn.relu)

    assert partial_n(u, var="x", order=1).deps == ("x",)
    with pytest.raises(ValueError, match="regularity-degenerate"):
        partial_n(u, var="x", order=2)
