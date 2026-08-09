#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _lorenz_problem():
    generator = np.random.default_rng(91)
    states = generator.normal(size=(256, 3))
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    derivatives = np.stack(
        (
            sigma * (states[:, 1] - states[:, 0]),
            states[:, 0] * (rho - states[:, 2]) - states[:, 1],
            states[:, 0] * states[:, 1] - beta * states[:, 2],
        ),
        axis=-1,
    )
    layout = phx.dynamics.StateLayout((3,), component_names=("x", "y", "z"))
    data = phx.dynamics.TrajectoryData(
        jnp.arange(states.shape[0], dtype=float),
        states,
        state_layout=layout,
        derivatives=derivatives,
        coordinate_id="time",
        source_id="lorenz-equations",
    )
    library = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2)
    return phx.dynamics.identification.SINDyProblem(
        data=data,
        library=library,
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )


def test_stlsq_recovers_lorenz_and_executes_through_continuous_system():
    result = phx.dynamics.identification.fit_sindy(
        _lorenz_problem(),
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            0.05,
            ridge=1e-10,
            threshold_space="physical",
            unbiased_refit=True,
        ),
    )

    assert bool(result.valid)
    names = result.design.feature_names
    coefficients = np.asarray(result.coefficients)
    expected = np.zeros_like(coefficients)
    expected[0, names.index("state:x")] = -10.0
    expected[0, names.index("state:y")] = 10.0
    expected[1, names.index("state:x")] = 28.0
    expected[1, names.index("state:y")] = -1.0
    expected[1, names.index("state:x * state:z")] = -1.0
    expected[2, names.index("state:z")] = -8.0 / 3.0
    expected[2, names.index("state:x * state:y")] = 1.0
    np.testing.assert_allclose(coefficients, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_array_equal(np.asarray(result.support), np.asarray(expected != 0.0))

    state = jnp.asarray([1.2, -0.7, 2.5])
    system = result.to_system(system_id="identified-lorenz")
    predicted = eqx.filter_jit(system.evaluate)(jnp.asarray(0.0), state, None)
    direct = result.evaluate(state)
    np.testing.assert_allclose(np.asarray(predicted), np.asarray(direct), atol=1e-12)
    assert result.render_equations()[0].startswith("dx/dtime =")


def test_physical_coefficients_are_unscaled_after_feature_normalization():
    state = jnp.linspace(-0.002, 0.002, 101)[:, None]
    derivative = 3.5 + 1200.0 * state
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    data = phx.dynamics.TrajectoryData(
        jnp.linspace(0.0, 1.0, state.shape[0]),
        state,
        state_layout=layout,
        derivatives=derivative,
        source_id="scaled-linear",
    )
    problem = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=1),
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )

    result = phx.dynamics.identification.fit_sindy(
        problem,
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            1e-8, scale_features=True, threshold_space="physical"
        ),
    )

    np.testing.assert_allclose(
        np.asarray(result.coefficients[0]), np.asarray([3.5, 1200.0]), rtol=1e-10
    )
    assert not np.allclose(
        np.asarray(result.regression.normalized_coefficients[0]),
        np.asarray(result.coefficients[0]),
    )


def test_empty_sparse_model_requires_explicit_zero_tolerance():
    time = jnp.linspace(0.0, 1.0, 20)
    layout = phx.dynamics.StateLayout((1,))
    data = phx.dynamics.TrajectoryData(
        time,
        time[:, None],
        state_layout=layout,
        derivatives=jnp.zeros((time.size, 1)),
        source_id="zero-law",
    )
    problem = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2),
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )

    rejected = phx.dynamics.identification.fit_sindy(
        problem,
        phx.dynamics.identification.SequentialThresholdedLeastSquares(0.1),
    )
    accepted = phx.dynamics.identification.fit_sindy(
        problem,
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            0.1, zero_tolerance=0.0
        ),
    )

    assert not bool(rejected.valid)
    assert bool(accepted.valid)
    assert not bool(jnp.any(accepted.support))
    np.testing.assert_allclose(np.asarray(accepted.coefficients), 0.0)
