import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._numerics import fejer_first_data


def test_axis_domains_encode_finite_and_unbounded_support():
    interval = phx.discretization.AxisDomain.interval(-2.0, 3.0)
    periodic = phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi)
    positive = phx.discretization.AxisDomain.half_line(1.5)
    negative = phx.discretization.AxisDomain.half_line(
        -0.5,
        direction="negative",
    )
    line = phx.discretization.AxisDomain.real_line()

    np.testing.assert_allclose(interval.finite_bounds, [-2.0, 3.0])
    assert interval.length == 5.0
    assert periodic.periodic_axis
    assert positive.lower == 1.5 and positive.upper is None
    assert negative.lower is None and negative.upper == -0.5
    assert line.finite_bounds is None
    with pytest.raises(ValueError, match="finite length"):
        _ = line.length


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (lambda: phx.discretization.AxisDomain.interval(1.0, 0.0), "increasing"),
        (
            lambda: phx.discretization.AxisDomain.half_line(0.0, direction="bad"),
            "direction",
        ),
    ),
)
def test_axis_domains_reject_invalid_support(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_point_primary_tensor_measure_uses_declared_quadrature_and_endpoint_flags():
    domain = phx.discretization.AxisDomain.interval(-1.0, 1.0)
    gauss = phx.discretization.TensorSpectralPlan(
        (phx.discretization.LegendreBasisPlan(8, node_rule="gauss"),)
    ).prepare((domain,))
    lobatto = phx.discretization.TensorSpectralPlan(
        (phx.discretization.LegendreBasisPlan(8, node_rule="lobatto"),)
    ).prepare((domain,))

    np.testing.assert_allclose(gauss.quadrature_weights, gauss.axes[0].quadrature_weights)
    np.testing.assert_allclose(jnp.sum(gauss.quadrature_weights), 2.0, atol=1e-14)
    assert not bool(gauss.grid.primary_entity_layout.lower_boundary_masks[0].any())
    assert not bool(gauss.grid.primary_entity_layout.upper_boundary_masks[0].any())
    assert bool(lobatto.grid.primary_entity_layout.lower_boundary_masks[0][0])
    assert bool(lobatto.grid.primary_entity_layout.upper_boundary_masks[0][-1])


def test_first_fejer_rule_is_endpoint_free_and_polynomial_exact():
    rule = fejer_first_data(8)

    assert jnp.all(jnp.abs(rule.nodes) < 1.0)
    assert jnp.all(rule.weights > 0.0)
    np.testing.assert_allclose(jnp.sum(rule.weights), 2.0, atol=2e-14)
    np.testing.assert_allclose(
        jnp.sum(rule.weights * rule.nodes**6),
        2.0 / 7.0,
        atol=2e-14,
    )
