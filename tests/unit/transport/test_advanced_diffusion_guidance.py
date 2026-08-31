import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _domains():
    state = phx.domain.HyperRectangle(
        jnp.full((2,), -10.0), jnp.full((2,), 10.0), label="x"
    )
    observation = phx.domain.HyperRectangle(
        jnp.full((2,), -10.0), jnp.full((2,), 10.0), label="observation"
    )
    return state @ phx.domain.TimeInterval(0.0, 1.0) @ observation


def test_time_conditioned_likelihood_guidance_adds_exact_score_gradient():
    domain = _domains()
    base_function = domain.Function("x", "t")(lambda state, time: -state)
    likelihood = domain.Function("x", "t", "observation")(
        lambda state, time, observation: -0.5 * jnp.sum((state - observation) ** 2)
    )
    base = phx._score_field.StateTimeScoreField(
        base_function,
        state_label="x",
        time_label="t",
    )
    guidance = phx.transport.TimeConditionedLikelihoodGuidance(
        likelihood,
        context_labels=("observation",),
    )
    guided = phx.transport.GuidedScoreField(base, (guidance,))
    context = phx.transport.ScoreContext({"observation": jnp.asarray([1.0, -1.0])})
    state = jnp.asarray([0.2, 0.1])
    score, evaluations, valid = guided.evaluate(state, 0.4, context, key=jr.key(0))

    expected = -state + (context.values["observation"] - state)
    assert valid
    assert evaluations[0].exactness == "exact"
    assert jnp.allclose(score, expected)


def test_classifier_free_guidance_marks_tempered_weights_heuristic():
    domain = _domains()
    unconditional = phx._score_field.StateTimeScoreField(
        domain.Function("x", "t")(lambda state, time: -state),
        state_label="x",
        time_label="t",
    )
    conditional = phx._score_field.StateTimeScoreField(
        domain.Function("x", "t", "observation")(
            lambda state, time, observation: -state + observation
        ),
        state_label="x",
        time_label="t",
        context_labels=("observation",),
    )
    context = phx.transport.ScoreContext({"observation": jnp.asarray([0.5, -0.5])})
    guidance = phx.transport.ClassifierFreeGuidance(
        unconditional, conditional, weight=2.0
    )
    evaluation = guidance.evaluate(jnp.zeros((2,)), 0.3, context, key=jr.key(1))

    assert evaluation.exactness == "heuristic"
    assert jnp.allclose(evaluation.correction, jnp.asarray([1.0, -1.0]))
    exact_guidance = phx.transport.ClassifierFreeGuidance(
        unconditional, conditional, weight=1.0
    )
    guided = phx.transport.GuidedScoreField(unconditional, (exact_guidance,))
    score, evaluations, valid = guided.evaluate(
        jnp.zeros((2,)), 0.3, context, key=jr.key(2)
    )
    assert valid
    assert evaluations[0].exactness == "exact"
    assert jnp.allclose(score, jnp.asarray([0.5, -0.5]))


def test_general_reverse_problem_uses_operator_noise_and_covariance_divergence():
    process = phx.stochastic.StateDependentItoDiffusion(
        lambda time, state: -0.1 * state,
        lambda time, state: jnp.diag(0.5 + 0.1 * state**2),
        dimension=2,
        noise_dimension=2,
        process_id="state-dependent-guided-test",
    )
    state_domain = phx.domain.HyperRectangle(
        jnp.full((2,), -10.0), jnp.full((2,), 10.0), label="x"
    )
    domain = state_domain @ phx.domain.TimeInterval(0.0, 1.0)
    score = phx._score_field.StateTimeScoreField(
        domain.Function("x", "t")(lambda state, time: -state),
        state_label="x",
        time_label="t",
    )
    result = phx.transport.general_reverse_diffusion_problem(
        process,
        score,
        jnp.asarray([0.2, -0.3]),
        score_id="analytic-score",
    )

    assert result.problem.stochastic
    assert result.problem.wiener_terms[0].representation == "operator"
    assert result.problem.noise_layout.total_size == 2
    system = phx.transport.general_probability_flow_system(
        process,
        score,
        context=phx.transport.ScoreContext({"unused": jnp.asarray(1.0)}),
        score_id="analytic-score",
        state_layout=phx.dynamics.StateLayout((2,)),
    )
    assert system(0.0, jnp.asarray([0.2, -0.3])).shape == (2,)
