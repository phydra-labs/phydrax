import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _interval_function(function):
    domain = phx.domain.Interval1d(-2.0, 2.0)
    return domain.Function("x")(function)


def _square_function(function):
    domain = phx.domain.Square(center=(0.0, 0.0), side=4.0)
    return domain.Function("x")(function)


def test_diffusion_covariance_accepts_rectangular_diffusion_and_covariance():
    diffusion = _square_function(
        lambda x: jnp.asarray([[1.0, x[0], -0.5], [2.0, 0.0, x[1]]])
    )
    point = jnp.asarray([0.4, -0.3])
    expected = diffusion.func(point) @ diffusion.func(point).T

    from_diffusion = phx.operators.diffusion_covariance(diffusion)
    observable = _square_function(lambda x: jnp.dot(x, x))
    zero_drift = _square_function(lambda x: jnp.zeros((2,)))
    from_covariance = phx.operators.kolmogorov_generator(
        observable,
        zero_drift,
        covariance=_square_function(lambda x: expected),
    )

    assert from_diffusion.func(point).shape == (2, 2)
    assert jnp.allclose(from_diffusion.func(point), expected)
    assert jnp.allclose(from_covariance.func(point), jnp.trace(expected))


def test_kolmogorov_generator_matches_quadratic_formula_and_vector_components():
    observable = _square_function(lambda x: x[0] ** 2 + 3.0 * x[1] ** 2)
    drift = _square_function(lambda x: jnp.asarray([0.4, -0.2]))
    diffusion = _square_function(lambda x: jnp.asarray([[1.0, 0.5], [0.0, 2.0]]))
    point = jnp.asarray([0.3, -0.4])
    covariance = diffusion.func(point) @ diffusion.func(point).T
    expected = jnp.dot(jnp.asarray([0.6, -2.4]), drift.func(point)) + 0.5 * (
        2.0 * covariance[0, 0] + 6.0 * covariance[1, 1]
    )

    generator = phx.operators.kolmogorov_generator(
        observable,
        drift,
        diffusion=diffusion,
    )

    vector_observable = _interval_function(
        lambda x: jnp.asarray([x[0] ** 2, jnp.sin(x[0])])
    )
    vector_generator = phx.operators.kolmogorov_generator(
        vector_observable,
        _interval_function(lambda x: jnp.asarray([0.4])),
        diffusion=_interval_function(lambda x: jnp.asarray([[0.3]])),
    )
    vector_point = jnp.asarray([0.3])
    vector_expected = jnp.asarray(
        [
            2.0 * vector_point[0] * 0.4 + 0.3**2,
            0.4 * jnp.cos(vector_point[0]) - 0.5 * 0.3**2 * jnp.sin(vector_point[0]),
        ]
    )

    assert jnp.allclose(generator.func(point), expected)
    assert vector_generator.func(vector_point).shape == (2,)
    assert jnp.allclose(vector_generator.func(vector_point), vector_expected)


def test_fokker_planck_uses_full_state_dependent_adjoint_and_time_dependence():
    density = _interval_function(lambda x: x[0] ** 4)
    state_dependent = phx.operators.fokker_planck_operator(
        density,
        _interval_function(lambda x: jnp.asarray([0.0])),
        covariance=_interval_function(lambda x: jnp.asarray([[x[0] ** 2]])),
    )
    point = jnp.asarray([0.4])

    spatial = phx.domain.Interval1d(-2.0, 2.0)
    time = phx.domain.TimeInterval(0.0, 1.0)
    spacetime = spatial @ time
    timed_density = spacetime.Function("x", "t")(lambda x, t: x[0] ** 2)
    timed_drift = spacetime.Function("x", "t")(lambda x, t: jnp.asarray([t * x[0]]))
    time_dependent = phx.operators.fokker_planck_operator(
        timed_density,
        timed_drift,
        var="x",
    )

    assert jnp.allclose(state_dependent.func(point), 15.0 * point[0] ** 4)
    assert jnp.allclose(
        time_dependent.func(point, jnp.asarray(0.7)),
        -3.0 * 0.7 * point[0] ** 2,
    )


def test_stratonovich_correction_handles_multiplicative_and_rectangular_noise():
    zero_drift = _interval_function(lambda x: jnp.asarray([0.0]))
    scalar_diffusion = _interval_function(lambda x: jnp.asarray([[1.5 * x[0]]]))
    point = jnp.asarray([0.4])
    scalar = phx.operators.stratonovich_to_ito_drift(
        zero_drift,
        scalar_diffusion,
    )

    vector_drift = _square_function(lambda x: jnp.asarray([0.1, -0.2]))
    rectangular = _square_function(lambda x: jnp.asarray([[x[0]], [2.0 * x[1]]]))
    vector_point = jnp.asarray([0.3, -0.4])
    vector = phx.operators.stratonovich_to_ito_drift(
        vector_drift,
        rectangular,
    )
    additive = phx.operators.stratonovich_to_ito_drift(
        vector_drift,
        _square_function(lambda x: jnp.asarray([[1.0], [2.0]])),
    )

    assert jnp.allclose(scalar.func(point), jnp.asarray([0.5 * 1.5**2 * point[0]]))
    assert jnp.allclose(
        vector.func(vector_point),
        vector_drift.func(vector_point)
        + jnp.asarray([0.5 * vector_point[0], 2.0 * vector_point[1]]),
    )
    assert jnp.allclose(additive.func(vector_point), vector_drift.func(vector_point))


def test_stratonovich_generators_equal_explicit_corrected_ito_forms():
    observable = _interval_function(lambda x: x[0] ** 3)
    density = _interval_function(lambda x: jnp.exp(-(x[0] ** 2)))
    drift = _interval_function(lambda x: jnp.asarray([0.2 * x[0]]))
    diffusion = _interval_function(lambda x: jnp.asarray([[0.7 * x[0]]]))
    corrected = phx.operators.stratonovich_to_ito_drift(drift, diffusion)
    point = jnp.asarray([0.6])

    stratonovich_generator = phx.operators.kolmogorov_generator(
        observable,
        drift,
        diffusion=diffusion,
        interpretation="stratonovich",
    )
    ito_generator = phx.operators.kolmogorov_generator(
        observable,
        corrected,
        diffusion=diffusion,
        interpretation="ito",
    )
    stratonovich_adjoint = phx.operators.fokker_planck_operator(
        density,
        drift,
        diffusion=diffusion,
        interpretation="stratonovich",
    )
    ito_adjoint = phx.operators.fokker_planck_operator(
        density,
        corrected,
        diffusion=diffusion,
        interpretation="ito",
    )

    assert jnp.allclose(stratonovich_generator.func(point), ito_generator.func(point))
    assert jnp.allclose(stratonovich_adjoint.func(point), ito_adjoint.func(point))


def test_stochastic_operators_jit_and_differentiate_through_diffusion():
    domain = phx.domain.Interval1d(-2.0, 2.0)
    observable = domain.Function("x")(lambda x: x[0] ** 2)
    drift = domain.Function("x")(lambda x: jnp.asarray([0.1 * x[0]]))
    point = jnp.asarray([0.4])

    def evaluate(scale):
        diffusion = domain.Function("x")(lambda x: jnp.asarray([[scale * x[0]]]))
        generated = phx.operators.kolmogorov_generator(
            observable,
            drift,
            diffusion=diffusion,
            interpretation="stratonovich",
        )
        return generated.func(point)

    compiled = eqx.filter_jit(lambda scale: evaluate(scale))(jnp.asarray(1.5))
    gradient = jax.grad(evaluate)(jnp.asarray(1.5))

    assert jnp.isfinite(compiled)
    assert jnp.allclose(gradient, 4.0 * 1.5 * point[0] ** 2)


def test_stochastic_operator_contracts_reject_ambiguous_or_malformed_fields():
    observable = _interval_function(lambda x: x[0] ** 2)
    density = _interval_function(lambda x: jnp.asarray([x[0], x[0] ** 2]))
    drift = _interval_function(lambda x: jnp.asarray([0.0]))
    diffusion = _interval_function(lambda x: jnp.asarray([[1.0]]))
    bad_diffusion = _interval_function(lambda x: jnp.asarray([1.0]))
    point = jnp.asarray([0.4])

    with pytest.raises(ValueError, match="require diffusion"):
        phx.operators.kolmogorov_generator(
            observable,
            drift,
            covariance=_interval_function(lambda x: jnp.asarray([[1.0]])),
            interpretation="stratonovich",
        )
    with pytest.raises(ValueError, match="trailing shape"):
        phx.operators.diffusion_covariance(bad_diffusion).func(point)
    with pytest.raises(ValueError, match="scalar-valued"):
        phx.operators.fokker_planck_operator(
            density,
            drift,
            diffusion=diffusion,
        ).func(point)


def test_factor_hvp_generator_matches_dense_contraction_componentwise():
    observable = _square_function(
        lambda x: jnp.asarray(
            [
                x[0] ** 4 + x[0] * x[1],
                jnp.sin(x[0]) + 2.0 * x[1] ** 2,
            ]
        )
    )
    drift = _square_function(lambda x: jnp.asarray([0.2 * x[0], -0.3 * x[1]]))
    diffusion = _square_function(
        lambda x: jnp.asarray([[1.0, x[0], 0.2], [0.4, -0.5, x[1]]])
    )
    point = jnp.asarray([0.3, -0.4])

    factor_hvp = phx.operators.kolmogorov_generator(
        observable,
        drift,
        diffusion=diffusion,
    )
    dense = phx.operators.kolmogorov_generator(
        observable,
        drift,
        diffusion=diffusion,
        contraction="dense",
    )

    assert jnp.allclose(factor_hvp.func(point), dense.func(point))
    assert jnp.allclose(
        eqx.filter_jit(lambda x: factor_hvp.func(x))(point),
        dense.func(point),
    )


def test_stochastic_trace_estimate_reports_replayable_probe_uncertainty():
    matrix = jnp.asarray([[1.4, 0.3, -0.2], [0.3, 0.8, 0.1], [-0.2, 0.1, 1.1]])
    state = jnp.asarray([0.2, -0.4, 0.7])
    observable = lambda x: jnp.asarray(
        [jnp.dot(x, x), x[0] ** 4 + 2.0 * x[1] ** 2 + x[2] ** 2]
    )
    exact = jnp.einsum(
        "ij,oij->o",
        matrix,
        jax.jacrev(jax.jacrev(observable))(state),
    )
    policy = phx.operators.StochasticTracePolicy(2048)

    first = phx.operators.estimate_stochastic_trace(
        observable,
        state,
        lambda x, vector: matrix @ vector,
        jax.random.key(17),
        policy=policy,
    )
    replay = phx.operators.estimate_stochastic_trace(
        observable,
        state,
        lambda x, vector: matrix @ vector,
        jax.random.key(17),
        policy=policy,
    )

    assert jnp.array_equal(first.value, replay.value)
    assert jnp.array_equal(first.standard_error, replay.standard_error)
    assert jnp.all(jnp.abs(first.value - exact) <= 5.0 * first.standard_error + 1e-12)
    assert first.num_probes == 2048


def test_probability_current_divergence_is_fokker_planck_operator():
    density = _square_function(lambda x: jnp.exp(-jnp.dot(x, x)))
    drift = _square_function(lambda x: jnp.asarray([0.3 * x[0], -0.4 * x[1]]))
    diffusion = _square_function(lambda x: jnp.asarray([[1.0, 0.2], [x[0], 0.7]]))
    current = phx.operators.probability_current(
        density,
        drift,
        diffusion=diffusion,
    )
    forward = phx.operators.fokker_planck_operator(
        density,
        drift,
        diffusion=diffusion,
    )
    point = jnp.asarray([0.3, -0.4])

    assert current.func(point).shape == (2,)
    assert jnp.allclose(
        forward.func(point),
        -phx.operators.div(current, var="x").func(point),
    )
