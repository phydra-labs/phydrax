import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import phydrax as phx


def test_array_complex_and_pytree_event_layouts_round_trip():
    array_layout = phx.stochastic.ArrayEventLayout((2, 2))
    array = jnp.arange(12.0).reshape((3, 2, 2))
    assert jnp.array_equal(
        array_layout.from_real_coordinates(array_layout.to_real_coordinates(array)), array
    )

    complex_layout = phx.stochastic.ComplexEventLayout((2,))
    complex_value = jnp.asarray([[1.0 + 2.0j, -3.0 + 0.5j]])
    assert jnp.array_equal(
        complex_layout.from_real_coordinates(
            complex_layout.to_real_coordinates(complex_value)
        ),
        complex_value,
    )

    tree_layout = phx.stochastic.PyTreeEventLayout(
        {
            "left": jnp.zeros((2,), dtype=jnp.float32),
            "right": jnp.zeros((1,), dtype=jnp.float64),
        }
    )
    tree = {
        "left": jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        "right": jnp.asarray([[5.0], [6.0]], dtype=jnp.float64),
    }
    restored = tree_layout.from_real_coordinates(tree_layout.to_real_coordinates(tree))
    assert jnp.array_equal(restored["left"], tree["left"])
    assert jnp.array_equal(restored["right"], tree["right"])
    assert restored["left"].dtype == jnp.dtype(jnp.float32)
    assert restored["right"].dtype == jnp.dtype(jnp.float64)


def test_gaussian_factor_law_distinguishes_lebesgue_and_hausdorff_scores():
    full = phx.uq.GaussianFactorLaw(
        jnp.zeros((2,)),
        phx.uq.GaussianFactor(jnp.asarray([[1.0, 0.0], [0.2, 0.8]])),
        event_shape=(2,),
    )
    value = jnp.asarray([0.2, -0.3])
    assert full.density_measure_kind == "lebesgue"
    assert jnp.all(jnp.isfinite(full.score(value)))

    singular = phx.uq.GaussianFactorLaw(
        jnp.zeros((2,)),
        phx.uq.GaussianFactor(jnp.asarray([[1.0], [0.0]])),
        event_shape=(2,),
    )
    assert singular.density_measure_kind == "hausdorff"
    assert singular.contains(jnp.asarray([0.3, 0.0]))
    assert not singular.contains(jnp.asarray([0.3, 0.1]))
    with pytest.raises(ValueError, match="no ambient Lebesgue score"):
        singular.score(jnp.asarray([0.3, 0.0]))


def test_matrix_and_state_dependent_reverse_drift_include_correct_covariance_terms():
    matrix = phx.stochastic.MatrixGaussianDiffusion(
        -0.2 * jnp.eye(2), jnp.asarray([[0.5, 0.0], [0.1, 0.4]])
    )
    marginal = matrix.marginal_transition(jnp.asarray([0.5, -0.2]), t0=0.0, t1=0.3)
    assert jnp.allclose(
        matrix.conditional_score(
            marginal.mean, jnp.asarray([0.5, -0.2]), t0=0.0, t1=0.3
        ),
        0.0,
    )

    process = phx.stochastic.StateDependentItoDiffusion(
        lambda t, x: -0.1 * x,
        lambda t, x: jnp.diag(0.5 + 0.1 * x**2),
        dimension=2,
        noise_dimension=2,
        process_id="multiplicative-diagonal",
    )
    state = jnp.asarray([0.3, -0.4])
    expected = 0.4 * state * (0.5 + 0.1 * state**2)
    assert jnp.allclose(process.covariance_divergence(0.2, state), expected)


def test_multiple_structured_wiener_blocks_solve_without_dense_concatenation():
    scalar_layout = phx.solver.WienerNoiseLayout((("scalar", (), None),))
    assert scalar_layout.blocks[0].shape == ()
    assert scalar_layout.total_size == 1
    problem = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.zeros((2,)),
        t0=0.0,
        t1=0.05,
        wiener_terms=(
            phx.solver.WienerTerm(
                "operator",
                lambda t, state, args: lx.MatrixLinearOperator(0.2 * jnp.eye(2)),
                (2,),
                structure="additive",
                representation="operator",
            ),
            phx.solver.WienerTerm(
                "diagonal",
                lambda t, state, args: jnp.full((2,), 0.1),
                (2,),
                structure="additive",
                representation="diagonal",
            ),
        ),
    )
    assert problem.noise_layout.total_size == 4
    realization = phx.stochastic.WienerRealization.independent(
        jr.key(0),
        problem.noise_shape,
        support=(0.0, 0.05),
        sample_shape=(3,),
        tolerance=1e-4,
    )
    result = phx.solver.solve_diffrax_ensemble(
        problem,
        save_times=jnp.asarray([0.05]),
        realization=realization,
        dt0=0.01,
    )
    assert jnp.all(result.successful)


def test_discrete_gaussian_prediction_conversions_and_samplers_are_consistent():
    schedule = phx.stochastic.DiscreteGaussianDiffusionSchedule.linear(8)
    clean = jnp.asarray([[0.2, -0.1], [0.4, 0.7]])
    noise = jr.normal(jr.key(1), clean.shape)
    timestep = jnp.asarray([2, 5])
    noisy = schedule.corrupt(clean, noise, timestep)
    assert jnp.allclose(schedule.clean_from_epsilon(noisy, noise, timestep), clean)
    assert jnp.allclose(schedule.epsilon_from_clean(noisy, clean, timestep), noise)

    def predictor(state, time, *, key=None):
        del time, key
        return jnp.zeros_like(state)

    ancestral_sampler = phx.stochastic.AncestralGaussianDiffusion(
        schedule, predictor, (2,)
    )
    ancestral = ancestral_sampler.sample(jr.key(2), (4,))
    multi_axis = ancestral_sampler.sample(jr.key(20), (2, 3))
    ddim = phx.stochastic.DDIMTransport(
        schedule, predictor, (2,), num_inference_steps=4
    ).sample(jr.key(3), (4,))
    assert jnp.all(ancestral.valid)
    assert jnp.all(ddim.valid)
    assert multi_axis.final_state.shape == (2, 3, 2)
    assert ancestral.terminal_relationship == "approximate"
    assert ddim.terminal_reference_id == "standard-normal"


def test_categorical_diffusion_exact_posterior_normalizes_and_respects_absorbing_state():
    schedule = phx.stochastic.CategoricalDiffusionSchedule.absorbing(5, 3, 2)
    clean = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32)
    timestep = jnp.asarray([1, 3])
    noisy = schedule.corrupt(clean, timestep, jr.key(4))
    posterior = schedule.posterior_probabilities(clean, noisy, timestep)
    clean_logits = jnp.where(
        jax.nn.one_hot(clean, schedule.num_classes).astype(bool),
        0.0,
        -jnp.inf,
    )
    model_posterior = schedule.reverse_probabilities_from_clean_logits(
        noisy, clean_logits, timestep
    )
    assert jnp.allclose(model_posterior, posterior)
    assert jnp.allclose(jnp.sum(posterior, axis=-1), 1.0)
    multi_clean = jnp.zeros((2, 3, 2), dtype=jnp.int32)
    multi_time = jnp.asarray([[0, 1, 2], [2, 3, 4]], dtype=jnp.int32)
    probabilities = schedule.marginal_probabilities(multi_clean, multi_time)
    assert probabilities.shape == (2, 3, 2, 3)
    assert jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0)
    assert jnp.array_equal(schedule.transition[:, 2, 2], jnp.ones((5,)))

    def predictor(state, time, *, key=None):
        del time, key
        return jnp.zeros(state.shape + (schedule.num_classes,))

    reverse = phx.stochastic.CategoricalReverseDiffusion(
        schedule,
        predictor,
        (2,),
        terminal_probabilities=jnp.asarray([0.0, 0.0, 1.0]),
        terminal_relationship="exact",
        terminal_reference_id="absorbing-class",
    ).sample(jr.key(5), (2, 3))
    assert reverse.final_state.shape == (2, 3, 2)
    assert reverse.terminal_relationship == "exact"
