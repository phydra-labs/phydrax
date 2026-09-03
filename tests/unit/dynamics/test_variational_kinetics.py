#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _linear_data(matrix, *, steps=400, reset=None):
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    state = jnp.asarray([1.0, -0.4], dtype=jnp.float64)
    values = []
    for index in range(steps):
        values.append(state)
        forcing = jnp.asarray(
            [jnp.sin(0.37 * index), jnp.cos(0.23 * index)], dtype=state.dtype
        )
        state = matrix @ state + 0.03 * forcing
    values = jnp.stack(values)
    return phx.dynamics.TrajectoryData(
        jnp.arange(steps, dtype=jnp.float64) * 0.2,
        values,
        state_layout=phx.dynamics.StateLayout((2,), component_names=("x", "y")),
        reset_mask=(
            jnp.zeros((steps - 1,), dtype=bool)
            if reset is None
            else jnp.asarray(reset, dtype=bool)
        ),
        source_id="linear-process",
    )


def _identity_library(layout):
    return phx.dynamics.identification.CustomFeatureLibrary(
        lambda states, inputs: states,
        state_layout=layout,
        feature_names=("x", "y"),
        library_id="identity-two-state",
    )


def test_vamp_and_tica_return_diagnosed_slow_coordinates():
    data = _linear_data([[0.97, 0.0], [0.0, 0.55]])
    library = _identity_library(data.state_layout)

    vamp = phx.dynamics.identification.fit_vamp(
        data, library, lag=2, n_modes=2, regularization=1.0e-6
    )
    tica = phx.dynamics.identification.fit_tica(
        data, lag=2, n_modes=2, regularization=1.0e-6
    )

    assert bool(vamp.valid)
    assert bool(tica.valid)
    assert vamp.transform_source(data.states[:7]).shape == (7, 2)
    assert tica.transform(data.states[:7]).shape == (7, 2)
    assert vamp.diagnostics.singular_values[0] > vamp.diagnostics.singular_values[1]
    times, valid = tica.implied_timescales()
    assert bool(valid[0])
    assert jnp.isfinite(times[0])
    assert tica.diagnostics.lag.physical_lag_mean == 0.4


def test_lagged_fit_excludes_resets_and_rejects_irregular_physical_lag():
    reset = jnp.zeros((29,), dtype=bool).at[12].set(True)
    data = _linear_data([[0.9, 0.0], [0.0, 0.7]], steps=30, reset=reset)
    library = _identity_library(data.state_layout)
    fitted = phx.dynamics.identification.fit_vamp(data, library, lag=2, n_modes=1)

    assert int(fitted.diagnostics.lag.excluded_pair_count) >= 2

    irregular = phx.dynamics.TrajectoryData(
        data.coordinates.at[20:].add(0.05),
        data.states,
        state_layout=data.state_layout,
        reset_mask=data.reset_mask,
        source_id="irregular-process",
    )
    rejected = phx.dynamics.identification.fit_vamp(irregular, library, lag=1, n_modes=1)
    assert not bool(rejected.valid)
    assert not bool(rejected.diagnostics.lag.uniform_physical_lag)


def test_vamp_heldout_score_is_finite():
    training = _linear_data([[0.95, -0.08], [0.08, 0.78]], steps=250)
    validation = _linear_data([[0.95, -0.08], [0.08, 0.78]], steps=120)
    fitted = phx.dynamics.identification.fit_vamp(
        training, _identity_library(training.state_layout), n_modes=2
    )

    score = phx.dynamics.analysis.score_vamp(fitted, validation)

    assert bool(score.valid)
    assert jnp.isfinite(score.vamp_e_score)
    assert score.effective_samples > 0.0


def test_tica_repeated_modes_report_basis_ambiguity():
    angles = jnp.linspace(0.0, 12.0 * jnp.pi, 600)
    states = jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1)
    data = phx.dynamics.TrajectoryData(
        jnp.arange(states.shape[0], dtype=jnp.float64),
        states,
        state_layout=phx.dynamics.StateLayout((2,)),
        source_id="degenerate-rotation",
    )

    result = phx.dynamics.identification.fit_tica(data, n_modes=2, regularization=1.0e-6)

    assert bool(result.valid)
    np.testing.assert_allclose(
        np.abs(np.asarray(result.eigenvalues[0])),
        np.abs(np.asarray(result.eigenvalues[1])),
        rtol=3.0e-2,
    )
