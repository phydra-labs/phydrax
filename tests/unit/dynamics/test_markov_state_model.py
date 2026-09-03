#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _state_data(assignments, *, step=1.0):
    values = jnp.asarray(assignments, dtype=jnp.int32)
    return phx.dynamics.TrajectoryData(
        jnp.arange(values.size, dtype=jnp.float64) * step,
        values[:, None].astype(jnp.float64),
        state_layout=phx.dynamics.StateLayout((1,), component_names=("state",)),
        source_id="state-sequence",
    )


def test_hard_and_soft_reversible_models_agree_and_obey_detailed_balance():
    assignments = jnp.asarray([0, 0, 1, 1, 0, 1, 1, 0] * 20, dtype=jnp.int32)
    data = _state_data(assignments)
    hard = phx.dynamics.identification.fit_markov_state_model(
        data, assignments, state_count=2, reversible=True
    )
    soft = phx.dynamics.identification.fit_markov_state_model(
        data, jnp.eye(2)[assignments], reversible=True
    )

    assert bool(hard.valid & soft.valid)
    np.testing.assert_allclose(
        hard.transition_matrix, soft.transition_matrix, atol=1.0e-12
    )
    np.testing.assert_allclose(
        hard.stationary_probabilities @ hard.transition_matrix,
        hard.stationary_probabilities,
        atol=1.0e-12,
    )
    assert hard.diagnostics.detailed_balance_residual < 1.0e-12


def test_disconnected_markov_model_reports_nonunique_support():
    assignments = jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    reset = jnp.zeros((assignments.size - 1,), dtype=bool).at[2].set(True)
    data = phx.dynamics.TrajectoryData(
        jnp.arange(assignments.size, dtype=jnp.float64),
        assignments[:, None].astype(jnp.float64),
        state_layout=phx.dynamics.StateLayout((1,)),
        reset_mask=reset,
        source_id="disconnected-states",
    )
    model = phx.dynamics.identification.fit_markov_state_model(
        data, assignments, state_count=2
    )

    assert bool(model.valid)
    assert not bool(model.diagnostics.irreducible)
    assert int(model.diagnostics.communicating_class_count) == 2


def test_chapman_kolmogorov_validation_uses_independent_long_lag():
    assignments = jnp.asarray([0, 0, 1, 1, 0, 0, 1, 1] * 30, dtype=jnp.int32)
    data = _state_data(assignments, step=0.5)
    short = phx.dynamics.identification.fit_markov_state_model(
        data, assignments, state_count=2, lag=1, pseudocount=1.0e-6
    )
    long = phx.dynamics.identification.fit_markov_state_model(
        data, assignments, state_count=2, lag=2, pseudocount=1.0e-6
    )

    validation = phx.dynamics.analysis.validate_markov_models(short, long, 2)

    assert bool(validation.valid)
    assert jnp.isfinite(validation.chapman_kolmogorov_residual)
    assert validation.chapman_kolmogorov_residual >= 0.0


def test_invalid_soft_assignments_do_not_contribute_counts():
    hard = jnp.asarray([0, 1, 0, 1, 0], dtype=jnp.int32)
    data = _state_data(hard)
    soft = jnp.eye(2)[hard].at[2].set(jnp.asarray([0.8, 0.8]))

    model = phx.dynamics.identification.fit_markov_state_model(data, soft, state_count=2)

    assert bool(model.valid)
    assert model.diagnostics.lag.valid_pair_count == 4
    assert jnp.all(jnp.isfinite(model.transition_matrix))
