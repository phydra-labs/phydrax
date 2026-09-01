import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from examples.differentiable_transit_photometry import build_workflow


def test_transit_photometry_workflow_has_finite_likelihood_and_gradient():
    cadence, occultation, plan, spectrum = build_workflow()
    baseline_relative = occultation.evaluate(jnp.abs(cadence), 0.1).relative_flux
    observed = jnp.floor(plan.evaluate(baseline_relative, spectrum).expected_counts)

    def objective(radius_ratio):
        relative = occultation.evaluate(jnp.abs(cadence), radius_ratio).relative_flux
        result = plan.evaluate(relative, spectrum)
        return phx.applications.astrophysics.transit_poisson_log_prob(result, observed)

    value, tangent = jax.jvp(objective, (jnp.asarray(0.1),), (jnp.asarray(1.0),))
    epsilon = 1.0e-4
    finite_difference = (objective(0.1 + epsilon) - objective(0.1 - epsilon)) / (
        2.0 * epsilon
    )
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(tangent))
    np.testing.assert_allclose(tangent, finite_difference, rtol=3.0e-3, atol=3.0e-2)
