import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def test_periodic_image_force_oracle_is_symmetric_and_gates_candidates():
    plan = cosmology.PeriodicImageForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.02,
        image_shells=1,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
    )
    positions = jnp.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    masses = jnp.asarray([1.0, 1.0])
    reference = plan.acceleration(positions, masses)
    np.testing.assert_allclose(reference[0], -reference[1], atol=1e-12)
    qualification = plan.qualify(positions, masses, reference)
    assert bool(qualification.successful)
    np.testing.assert_allclose(qualification.maximum_absolute_error, 0.0)
    rejected = plan.qualify(positions, masses, jnp.zeros_like(reference))
    assert not bool(rejected.successful)
    assert rejected.maximum_absolute_error > 0.0
