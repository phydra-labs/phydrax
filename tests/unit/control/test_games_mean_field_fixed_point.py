import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.control.games._mean_field import (
    FrozenLawBestResponseProblem,
    solve_frozen_law_best_response,
)
from phydrax.control.games._mean_field_fixed_point import (
    MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE,
    MeanFieldGameFixedPointPlan,
    MeanFieldGameFixedPointProblem,
    MeanFieldGameFixedPointStatus,
    solve_mean_field_game_fixed_point,
)
from phydrax.stochastic import (
    adapt_mean_field_control_bsde,
    BSDEPathBatch,
    EmpiricalMeanField,
    MeanFieldBSDEControlAdapter,
)


def _law(
    value: float,
    flow_id: str,
    source_path_id: str | None,
    *,
    weights=None,
    particles: int = 2,
) -> EmpiricalMeanField:
    return EmpiricalMeanField(
        jnp.asarray([0.0, 1.0]),
        jnp.full((particles, 2, 1), value),
        sample_shape=(particles,),
        state_shape=(1,),
        mean_field_id=flow_id,
        weights=weights,
        source_path_id=source_path_id,
    )


def _mean(flow: EmpiricalMeanField):
    return jnp.mean(jax_means(flow))


def jax_means(flow: EmpiricalMeanField):
    return jnp.stack([flow.snapshot(time).mean[0] for time in flow.times])


def _response(flow: EmpiricalMeanField, *, minimum_ess: float = 2.0):
    path_id = f"best-response-evaluation:{flow.mean_field_id}"
    particles = flow.num_particles
    paths = BSDEPathBatch(
        flow.times,
        jnp.zeros((particles, 2, 1)),
        jnp.zeros((particles, 1, 1)),
        sample_shape=(particles,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id=path_id,
        process_id="one-period-process",
    )
    adapter = MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, args: -z.reshape((1,)),
        lambda time, state, law, action, args: 0.5 * action**2,
        lambda time, state, law, action, args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="analytic-best-response",
    )
    base = adapt_mean_field_control_bsde(
        lambda key: paths,
        flow,
        lambda time, state, law, args: jnp.zeros((1,)),
        lambda time, state, law, args: jnp.ones((1, 1)),
        lambda state, law, args: jnp.zeros((1,)),
        adapter,
        state_shape=(1,),
        problem_id=f"base:{flow.mean_field_id}",
        process_id=paths.process_id,
    )
    frozen = FrozenLawBestResponseProblem(
        base,
        adapter,
        supplied_law_id=f"supplied:{flow.mean_field_id}",
        problem_id=f"frozen:{flow.mean_field_id}",
    )
    return solve_frozen_law_best_response(
        frozen,
        paths,
        lambda time, state: jnp.zeros((1,)),
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
        key=jr.key(11),
        minimum_effective_sample_size=minimum_ess,
    )


def _induced(response, value, *, weights=None, source=None):
    flow_id = f"induced:{response.flow_id}"
    return _law(
        value,
        flow_id,
        source or f"new-forward-paths:{response.flow_id}",
        weights=weights,
        particles=response.mean_field.num_particles,
    )


def _normalised_weights(flow: EmpiricalMeanField):
    weights = flow.weights.reshape((flow.num_particles, flow.times.size))
    return weights / jnp.sum(weights, axis=0, keepdims=True)


def _law_mixture(current, induced, damping, iteration, args):
    del args
    time_count = current.times.size
    particles = jnp.concatenate(
        (
            current.particles.reshape(
                (current.num_particles, time_count) + current.state_shape
            ),
            induced.particles.reshape(
                (induced.num_particles, time_count) + induced.state_shape
            ),
        )
    )
    weights = jnp.concatenate(
        (
            (1.0 - damping) * _normalised_weights(current),
            damping * _normalised_weights(induced),
        )
    )
    valid = jnp.concatenate(
        (
            current.valid.reshape((current.num_particles, time_count)),
            induced.valid.reshape((induced.num_particles, time_count)),
        )
    )
    return EmpiricalMeanField(
        current.times,
        particles,
        sample_shape=(current.num_particles + induced.num_particles,),
        state_shape=current.state_shape,
        mean_field_id=(
            f"union-mixture:{iteration}:{current.mean_field_id}+{induced.mean_field_id}"
        ),
        weights=weights,
        valid=valid,
        source_path_id=None,
    )


def _problem(
    initial,
    induced,
    *,
    distance=None,
    best_response=_response,
    law_mixture=_law_mixture,
    law_mixture_id="exact-union-support-mixture",
):
    if distance is None:
        distance = lambda current, candidate, args: jnp.max(
            jnp.abs(jax_means(current) - jax_means(candidate))
        )
    mixture_keywords = (
        {}
        if law_mixture is None and law_mixture_id is None
        else {
            "law_mixture": law_mixture,
            "law_mixture_id": law_mixture_id,
        }
    )
    return MeanFieldGameFixedPointProblem(
        initial,
        lambda current, args: best_response(current),
        induced,
        distance,
        **mixture_keywords,
        best_response_id="analytic-frozen-best-response",
        induced_flow_id="independent-forward-law",
        law_distance_id="maximum-node-mean-distance",
        problem_id="one-period-mean-field-game",
    )


def _plan(
    maximum_iterations=4,
    *,
    tolerance=1.0e-8,
    damping=1.0,
    minimum_ess=2.0,
):
    return MeanFieldGameFixedPointPlan(
        maximum_iterations=maximum_iterations,
        consistency_tolerance=tolerance,
        damping=damping,
        minimum_effective_sample_size=minimum_ess,
        problem_id="one-period-mean-field-game",
    )


def test_analytic_one_period_fixed_point_retains_both_evidence_layers():
    initial = _law(1.0, "initial-fixed-point", "initial-paths")
    problem = _problem(initial, lambda response, args: _induced(response, 1.0))

    result = solve_mean_field_game_fixed_point(problem, _plan())

    assert result.status == MeanFieldGameFixedPointStatus.SUCCESS
    assert result.successful
    assert result.converged
    assert result.iterations == 1
    assert result.accepted_iteration == 0
    assert result.best_response_result is not None
    assert result.best_response_result.valid
    assert result.best_response_result.mean_field is result.flow
    assert result.induced_flow is not None
    assert result.induced_flow.mean_field_id == "induced:initial-fixed-point"
    assert result.current_flow_ids == ("initial-fixed-point", None, None, None)
    assert result.induced_flow_ids == (
        "induced:initial-fixed-point",
        None,
        None,
        None,
    )
    np.testing.assert_allclose(result.distance_history[0], 0.0)
    assert result.best_response_validity_history[0]
    assert result.consistency_validity_history[0]


def test_successful_frozen_response_does_not_hide_deliberately_wrong_induced_law():
    initial = _law(0.0, "wrong-law-initial", "wrong-law-input-paths")
    problem = _problem(initial, lambda response, args: _induced(response, 3.0))

    result = solve_mean_field_game_fixed_point(problem, _plan(1, tolerance=1.0e-10))

    assert result.best_response_result is not None
    assert result.best_response_result.valid
    assert result.best_response_validity_history[0]
    assert result.consistency_validity_history[0]
    np.testing.assert_allclose(result.distance_history[0], 3.0)
    assert result.status == MeanFieldGameFixedPointStatus.MAX_ITERATIONS
    assert not result.valid


def test_half_damping_forms_a_union_support_mixture_instead_of_a_midpoint():
    initial = _law(-1.0, "damping-initial", "damping-input-paths")
    problem = _problem(initial, lambda response, args: _induced(response, 1.0))

    result = solve_mean_field_game_fixed_point(
        problem,
        _plan(1, tolerance=0.0, damping=0.5),
    )

    assert result.status == MeanFieldGameFixedPointStatus.MAX_ITERATIONS
    np.testing.assert_array_equal(
        jnp.unique(result.flow.particles), jnp.asarray([-1.0, 1.0])
    )
    assert not bool(jnp.any(result.flow.particles == 0.0))
    np.testing.assert_allclose(result.flow.snapshot(0.0).mean, jnp.asarray([0.0]))
    np.testing.assert_allclose(
        result.flow.snapshot(0.0).expectation(lambda value: value**2),
        jnp.asarray([1.0]),
    )
    assert result.flow.source_path_id is None
    assert result.law_mixture_id == "exact-union-support-mixture"


def test_subunit_damping_requires_an_identified_law_mixture():
    initial = _law(-1.0, "missing-mixture-initial", "missing-mixture-paths")
    with pytest.raises(ValueError, match="must be supplied together"):
        _problem(
            initial,
            lambda response, args: _induced(response, 1.0),
            law_mixture=_law_mixture,
            law_mixture_id=None,
        )

    problem = _problem(
        initial,
        lambda response, args: _induced(response, 1.0),
        law_mixture=None,
        law_mixture_id=None,
    )

    with pytest.raises(ValueError, match="law_mixture and law_mixture_id"):
        solve_mean_field_game_fixed_point(problem, _plan(1, damping=0.5))


@pytest.mark.parametrize(
    "invalid_kind", ["non-law", "reused-law", "midpoint", "claimed-source"]
)
def test_invalid_law_mixture_callback_fails_closed(invalid_kind):
    initial = _law(-1.0, "invalid-mixture-initial", "invalid-mixture-paths")

    def invalid_mixture(current, induced, damping, iteration, args):
        if invalid_kind == "non-law":
            return object()
        if invalid_kind == "reused-law":
            return current
        if invalid_kind == "midpoint":
            return _law(0.0, "synthesised-midpoint", None)
        candidate = _law_mixture(current, induced, damping, iteration, args)
        return EmpiricalMeanField(
            candidate.times,
            candidate.particles,
            sample_shape=candidate.sample_shape,
            state_shape=candidate.state_shape,
            mean_field_id=candidate.mean_field_id,
            weights=candidate.weights,
            valid=candidate.valid,
            source_path_id="falsely-claimed-source-paths",
        )

    problem = _problem(
        initial,
        lambda response, args: _induced(response, 1.0),
        law_mixture=invalid_mixture,
    )

    result = solve_mean_field_game_fixed_point(
        problem, _plan(1, tolerance=0.0, damping=0.5)
    )

    assert result.status == MeanFieldGameFixedPointStatus.INVALID_LAW_MIXTURE
    assert not result.valid


def test_unit_damping_uses_the_induced_law_without_calling_the_mixture():
    initial = _law(0.0, "unit-damping-initial", "unit-damping-paths")

    def forbidden_mixture(current, induced, damping, iteration, args):
        raise AssertionError("law_mixture must not be called when damping is one")

    problem = _problem(
        initial,
        lambda response, args: _induced(response, 1.0),
        law_mixture=forbidden_mixture,
    )

    result = solve_mean_field_game_fixed_point(
        problem, _plan(1, tolerance=0.0, damping=1.0)
    )

    assert result.status == MeanFieldGameFixedPointStatus.MAX_ITERATIONS
    assert result.flow is result.induced_flow
    assert result.flow.source_path_id is not None


def test_singular_nonconvergent_induced_map_exhausts_fixed_capacity():
    initial = _law(1.0, "oscillation-initial", "oscillation-input-paths")
    problem = _problem(
        initial,
        lambda response, args: _induced(response, -float(_mean(response.mean_field))),
    )

    result = solve_mean_field_game_fixed_point(
        problem, _plan(4, tolerance=1.0e-12, damping=1.0)
    )

    assert result.status == MeanFieldGameFixedPointStatus.MAX_ITERATIONS
    assert result.iterations == 4
    assert result.accepted_iterations == 4
    np.testing.assert_allclose(result.distance_history, 2.0)
    assert jnp.all(result.iteration_validity_history)
    assert not result.converged


def test_low_effective_sample_size_is_rejected_before_distance_acceptance():
    initial = _law(0.0, "ess-initial", "ess-input-paths")
    problem = _problem(
        initial,
        lambda response, args: _induced(
            response,
            0.0,
            weights=jnp.asarray([[1.0, 1.0], [0.0, 0.0]]),
        ),
    )

    result = solve_mean_field_game_fixed_point(problem, _plan(minimum_ess=2.0))

    assert result.status == MeanFieldGameFixedPointStatus.LOW_EFFECTIVE_SAMPLE_SIZE
    np.testing.assert_allclose(result.induced_effective_sample_size_history[0], 1.0)
    assert result.induced_flow_validity_history[0]
    assert not result.consistency_validity_history[0]
    assert jnp.isnan(result.distance_history[0])


@pytest.mark.parametrize("reuse", ["flow", "source", "best-response-paths"])
def test_induced_law_requires_new_flow_and_forward_path_identities(reuse):
    initial = _law(0.0, f"identity-initial:{reuse}", f"identity-input-paths:{reuse}")

    def induced(response, args):
        if reuse == "flow":
            return response.mean_field
        if reuse == "source":
            source = response.mean_field.source_path_id
        else:
            source = response.paths.path_id
        return _induced(response, 0.0, source=source)

    result = solve_mean_field_game_fixed_point(
        _problem(initial, induced), _plan(maximum_iterations=1)
    )

    assert result.status == MeanFieldGameFixedPointStatus.INVALID_INDUCED_LAW
    assert result.best_response_validity_history[0]
    assert not result.induced_flow_validity_history[0]
    assert jnp.isnan(result.distance_history[0])


def test_invalid_response_invalid_law_and_nonfinite_metric_fail_closed():
    initial = _law(0.0, "failure-initial", "failure-input-paths")
    invalid_response = _problem(
        initial,
        lambda response, args: _induced(response, 0.0),
        best_response=lambda current: object(),
    )
    invalid_law = _problem(initial, lambda response, args: object())
    nonfinite = _problem(
        initial,
        lambda response, args: _induced(response, 0.0),
        distance=lambda current, candidate, args: jnp.asarray(jnp.nan),
    )

    response_result = solve_mean_field_game_fixed_point(invalid_response, _plan())
    law_result = solve_mean_field_game_fixed_point(invalid_law, _plan())
    distance_result = solve_mean_field_game_fixed_point(nonfinite, _plan())

    assert response_result.status == MeanFieldGameFixedPointStatus.INVALID_BEST_RESPONSE
    assert law_result.status == MeanFieldGameFixedPointStatus.INVALID_INDUCED_LAW
    assert distance_result.status == MeanFieldGameFixedPointStatus.NONFINITE_LAW_DISTANCE
    assert not response_result.valid
    assert not law_result.valid
    assert not distance_result.valid


def test_fixed_capacity_histories_and_ids_are_deterministic():
    initial = _law(0.0, "deterministic-initial", "deterministic-input-paths")
    problem = _problem(initial, lambda response, args: _induced(response, 1.0))
    plan = _plan(3, tolerance=0.0, damping=0.5)

    first = solve_mean_field_game_fixed_point(problem, plan)
    second = solve_mean_field_game_fixed_point(problem, plan)

    assert first.distance_history.shape == (3,)
    assert first.current_effective_sample_size_history.shape == (3,)
    assert first.best_response_validity_history.shape == (3,)
    assert plan.plan_id == _plan(3, tolerance=0.0, damping=0.5).plan_id
    assert plan.plan_id != _plan(3, tolerance=0.0, damping=0.25).plan_id
    np.testing.assert_array_equal(first.distance_history, second.distance_history)
    np.testing.assert_array_equal(
        first.iteration_validity_history, second.iteration_validity_history
    )
    assert first.current_flow_ids == second.current_flow_ids
    assert first.induced_flow_ids == second.induced_flow_ids
    assert first.current_flow_id == second.current_flow_id


def test_result_label_and_claim_boundaries_are_explicit():
    initial = _law(2.0, "label-initial", "label-input-paths")
    result = solve_mean_field_game_fixed_point(
        _problem(initial, lambda response, args: _induced(response, 2.0)), _plan()
    )

    assert result.status == MeanFieldGameFixedPointStatus.SUCCESS
    assert result.certificate_label == MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE
    assert result.candidate_evaluation_only
    assert result.law_consistency_evaluated
    assert not result.best_response_optimality_evaluated
    assert not result.mean_field_control_optimum_claimed
    assert not result.finite_population_game_claimed
    assert not result.common_noise_equilibrium_claimed
    assert not result.mean_field_game_equilibrium_claimed
