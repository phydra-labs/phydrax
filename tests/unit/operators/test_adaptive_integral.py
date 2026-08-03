#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


@pytest.mark.parametrize(
    "method",
    ("gauss_kronrod", "clenshaw_curtis", "tanh_sinh"),
)
def test_adaptive_methods_integrate_scalar_interval_polynomial(method):
    time = phx.domain.ScalarInterval(-1.0, 2.0, label="t")

    @time.Function("t")
    def polynomial(t):
        return t**3 - 2.0 * t + 1.0

    result = phx.operators.adaptive_integral(
        polynomial,
        component=time.component(),
        quadrature=phx.operators.AdaptiveQuadratureConfig(method=method),
    )

    assert bool(result.successful)
    assert result.value.dims == ()
    assert jnp.allclose(result.value.data, 3.75, rtol=1e-9, atol=1e-11)
    assert result.num_evaluations > 0
    assert result.estimated_error >= 0.0


def test_adaptive_integral_preserves_fixed_slice_masks_weights_and_breakpoints():
    space = phx.domain.Interval1d(0.0, 1.0)
    time = phx.domain.ScalarInterval(0.0, 2.0, label="t")
    domain = space @ time
    component = domain.component(
        {"t": phx.domain.Fixed(0.25)},
        where={"x": lambda x: x[0] <= 0.5},
        weight_all=lambda x, t: 2.0,
    )

    @domain.Function("x", "t")
    def field(x, t):
        return x[0] + t

    config = phx.operators.AdaptiveQuadratureConfig(breakpoints=(0.5,))
    run = jax.jit(
        lambda scale: phx.operators.adaptive_integral(
            scale * field,
            component=component,
            variable="x",
            quadrature=config,
        )
    )
    result = run(1.0)

    assert bool(result.successful)
    assert jnp.allclose(result.value.data, 0.5, rtol=1e-9, atol=1e-11)


def test_adaptive_integral_supports_vector_outputs_and_subinterval_diagnostics():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")

    @time.Function("t")
    def moments(t):
        return jnp.asarray((t, t**2))

    config = phx.operators.AdaptiveQuadratureConfig(collect_subintervals=True)
    result = phx.operators.adaptive_integral(
        moments,
        component=time.component(),
        quadrature=config,
    )

    assert result.value.dims == (None,)
    assert jnp.allclose(result.value.data, jnp.asarray((0.5, 1.0 / 3.0)), atol=1e-11)
    assert result.subintervals is not None
    assert int(result.subintervals.count) >= 1
    assert result.subintervals.lower_bounds.shape == (config.max_intervals,)
    assert result.subintervals.upper_bounds.shape == (config.max_intervals,)
    assert result.subintervals.integral_estimates.shape == (
        config.max_intervals,
        2,
    )


def test_adaptive_integral_is_jittable_and_differentiable():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")

    def integral_value(scale):
        @time.Function("t")
        def density(t):
            return scale * t**2

        return phx.operators.adaptive_integral(
            density,
            component=time.component(),
        ).value.data

    compiled_value = jax.jit(integral_value)(jnp.asarray(2.0))
    derivative = jax.grad(integral_value)(jnp.asarray(2.0))

    assert jnp.allclose(compiled_value, 2.0 / 3.0, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(derivative, 1.0 / 3.0, rtol=1e-9, atol=1e-11)


def test_adaptive_integral_reports_numerical_failure_without_throwing():
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")

    @time.Function("t")
    def discontinuity(t):
        return jnp.where(t < 0.123, 1.0, 0.0)

    result = phx.operators.adaptive_integral(
        discontinuity,
        component=time.component(),
        quadrature=phx.operators.AdaptiveQuadratureConfig(
            absolute_tolerance=1e-14,
            relative_tolerance=1e-14,
            max_intervals=1,
            throw=False,
        ),
    )

    assert not bool(result.successful)
    assert int(result.status) != 0
    assert result.estimated_error > 1e-14


def test_adaptive_integral_rejects_unsupported_domain_structure():
    space = phx.domain.Interval1d(0.0, 1.0)
    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    product = space @ time

    @product.Function("x", "t")
    def field(x, t):
        return x[0] + t

    with pytest.raises(ValueError, match="exactly one interior label"):
        phx.operators.adaptive_integral(
            field,
            component=product.component(),
        )

    square = phx.domain.Square(center=(0.0, 0.0), side=1.0)
    with pytest.raises(TypeError, match="ScalarInterval and Interval1d"):
        phx.operators.adaptive_integral(
            1.0,
            component=square.component(),
        )


def test_adaptive_quadrature_config_validates_backend_contracts():
    with pytest.raises(ValueError, match="order"):
        phx.operators.AdaptiveQuadratureConfig(method="gauss_kronrod", order=32)
    with pytest.raises(ValueError, match="strictly increasing"):
        phx.operators.AdaptiveQuadratureConfig(breakpoints=(0.5, 0.5))
    with pytest.raises(ValueError, match="initial intervals"):
        phx.operators.AdaptiveQuadratureConfig(
            breakpoints=(0.25, 0.5),
            max_intervals=2,
        )
