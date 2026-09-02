import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.domain._normalized_density import (
    density_normalization_evidence,
    normalize_density_field,
)


def _realization():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(16))
    return domain, phx.integration.materialize(target, plan)


def test_normalized_density_is_positive_shift_invariant_and_represented_unit_mass():
    domain, realization = _realization()
    base = domain.Function("x")(lambda x: 2.0 * x)
    shifted = domain.Function("x")(lambda x: 2.0 * x + 37.0)
    first = normalize_density_field(base, realization)
    second = normalize_density_field(shifted, realization)
    first_mass = phx.integration.reduce(first.field, realization)
    second_mass = phx.integration.reduce(second.field, realization)
    points = jnp.asarray([0.1, 0.4, 0.9])

    assert jnp.all(first.field.func(points) > 0.0)
    assert jnp.allclose(first.field.func(points), second.field.func(points))
    assert jnp.allclose(first_mass.value.data, 1.0)
    assert jnp.allclose(second_mass.value.data, 1.0)
    assert density_normalization_evidence(first).valid


def test_normalized_density_has_finite_fixed_realization_gradient():
    domain, realization = _realization()

    def objective(scale):
        log_field = domain.Function("x")(lambda x: scale * x)
        normalized = normalize_density_field(log_field, realization)
        moment = phx.integration.reduce(normalized.field * normalized.field, realization)
        return moment.value.data

    derivative = jax.grad(objective)(jnp.asarray(0.7))
    assert jnp.isfinite(derivative)
