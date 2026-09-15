#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.stochastic.path_sampling import (
    DiscretePathThermodynamicsPlan,
    normalized_discrete_path_thermodynamics,
    PathBuffer,
)


def _paths():
    return tuple(
        PathBuffer.from_trajectory(
            jnp.asarray([[0.0], [offset], [1.0 + offset]]),
            jnp.asarray([0.0, 0.5, 1.0]),
            capacity=4,
        )
        for offset in (0.1, 0.3, 0.6)
    )


def test_normalized_path_thermodynamics_obeys_reversal_and_fluctuation_identity():
    paths = _paths()
    forward = jnp.asarray([0.6, 0.3, 0.1])
    reverse = jnp.asarray([0.3, 0.3, 0.4])
    entropy = jnp.log(forward) - jnp.log(reverse)
    heat = -entropy
    plan = DiscretePathThermodynamicsPlan(
        1.0,
        maximum_paths=4,
        maximum_steps=3,
        autocorrelation_lag=1,
    )

    result = normalized_discrete_path_thermodynamics(
        plan,
        paths,
        jnp.log(forward),
        jnp.log(reverse),
        jnp.zeros(3),
        -heat,
        heat,
        jnp.zeros(3),
    )

    assert result.successful
    np.testing.assert_allclose(jnp.sum(result.normalized_forward_probability), 1.0)
    np.testing.assert_allclose(jnp.sum(result.normalized_reverse_probability), 1.0)
    np.testing.assert_allclose(result.integral_fluctuation_average, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(result.first_law_residual, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.detailed_balance_residual, 0.0, atol=1.0e-14)

    reversed_result = normalized_discrete_path_thermodynamics(
        plan,
        tuple(path.time_reversed() for path in paths),
        jnp.log(reverse),
        jnp.log(forward),
        jnp.zeros(3),
        heat,
        -heat,
        jnp.zeros(3),
    )
    assert reversed_result.successful
    np.testing.assert_allclose(
        reversed_result.path_entropy_production,
        -result.path_entropy_production,
        atol=1.0e-14,
    )


def test_path_thermodynamics_retains_failed_detailed_balance_evidence():
    paths = _paths()
    plan = DiscretePathThermodynamicsPlan(
        1.0,
        maximum_paths=3,
        maximum_steps=3,
        autocorrelation_lag=1,
    )
    result = normalized_discrete_path_thermodynamics(
        plan,
        paths,
        jnp.log(jnp.asarray([0.6, 0.3, 0.1])),
        jnp.log(jnp.asarray([0.3, 0.3, 0.4])),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )

    assert not bool(result.successful)
    assert jnp.max(jnp.abs(result.detailed_balance_residual)) > 0.0
