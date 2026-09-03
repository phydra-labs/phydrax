import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.control.games._mean_field import (
    FrozenLawBestResponseProblem,
    solve_frozen_law_best_response,
)
from phydrax.control.games._mean_field_common_noise import (
    COMMON_NOISE_MFG_FIXED_POINT_CANDIDATE,
    CommonNoiseMeanFieldPlan,
    CommonNoiseMeanFieldProblem,
    CommonNoiseMeanFieldStatus,
    solve_common_noise_mean_field_fixed_point,
)
from phydrax.stochastic import (
    adapt_mean_field_control_bsde,
    BSDEPathBatch,
    EmpiricalMeanField,
    MeanFieldBSDEControlAdapter,
)


def _law(value, flow_id, source_path_id, *, particles=4, weights=None):
    return EmpiricalMeanField(
        jnp.asarray([0.0, 1.0]),
        jnp.full((particles, 2, 1), value),
        sample_shape=(particles,),
        state_shape=(1,),
        mean_field_id=flow_id,
        weights=weights,
        source_path_id=source_path_id,
    )


def _flow_means(flow):
    return jnp.stack([flow.snapshot(time).mean[0] for time in flow.times])


def _flow_mean(flow):
    return float(jnp.mean(_flow_means(flow)))


def _response(flow):
    particles = flow.sample_shape[0]
    paths = BSDEPathBatch(
        flow.times,
        jnp.zeros((particles, 2, 1)),
        jnp.zeros((particles, 1, 1)),
        sample_shape=flow.sample_shape,
        state_shape=(1,),
        noise_shape=(1,),
        path_id=f"best-response-paths:{flow.mean_field_id}",
        process_id="common-noise-one-period-process",
    )
    adapter = MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, args: -z.reshape((1,)),
        lambda time, state, law, action, args: 0.5 * action**2,
        lambda time, state, law, action, args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="common-noise-analytic-best-response",
    )
    base = adapt_mean_field_control_bsde(
        lambda key: paths,
        flow,
        lambda time, state, law, args: jnp.zeros((1,)),
        lambda time, state, law, args: jnp.ones((1, 1)),
        lambda state, law, args: jnp.zeros((1,)),
        adapter,
        state_shape=(1,),
        problem_id=f"common-noise-base:{flow.mean_field_id}",
        process_id=paths.process_id,
    )
    frozen = FrozenLawBestResponseProblem(
        base,
        adapter,
        supplied_law_id=f"conditional-law:{flow.mean_field_id}",
        problem_id=f"common-noise-frozen:{flow.mean_field_id}",
    )
    return solve_frozen_law_best_response(
        frozen,
        paths,
        lambda time, state: jnp.zeros((1,)),
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
        key=jr.key(91),
    )


def _induced(response, value, *, source_path_id=None):
    return _law(
        value,
        f"induced:{response.flow_id}",
        (
            f"independent-forward-paths:{response.flow_id}"
            if source_path_id is None
            else source_path_id
        ),
        particles=response.mean_field.sample_shape[0],
    )


def _normalised_weights(flow):
    weights = flow.weights.reshape((flow.num_particles, flow.times.size))
    return weights / jnp.sum(weights, axis=0, keepdims=True)


def _law_mixture(current, induced, damping, iteration, history, args):
    del history, args
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
            f"conditional-union-mixture:{iteration}:"
            f"{current.mean_field_id}+{induced.mean_field_id}"
        ),
        weights=weights,
        valid=valid,
        source_path_id=None,
    )


def _problem(
    initial_flows,
    histories,
    probabilities,
    labels,
    induced_value,
    *,
    scenario_ids=("down", "up"),
    observed=None,
    induced_source=None,
    law_mixture=_law_mixture,
    law_mixture_id="conditional-exact-union-support-mixture",
):
    def best_response(flow, history, args):
        if observed is not None:
            observed.append(("response", flow.mean_field_id, history))
        return _response(flow)

    def induced(response, history, args):
        if observed is not None:
            observed.append(("induced", response.flow_id, history))
        return _induced(
            response,
            induced_value(response.mean_field, history),
            source_path_id=(
                None if induced_source is None else induced_source(response, history)
            ),
        )

    def distance(current, candidate, history, args):
        if observed is not None:
            observed.append(("distance", current.mean_field_id, history))
        return jnp.max(jnp.abs(_flow_means(current) - _flow_means(candidate)))

    mixture_keywords = (
        {}
        if law_mixture is None and law_mixture_id is None
        else {
            "law_mixture": law_mixture,
            "law_mixture_id": law_mixture_id,
        }
    )

    return CommonNoiseMeanFieldProblem(
        initial_flows,
        histories,
        probabilities,
        labels,
        best_response,
        induced,
        distance,
        **mixture_keywords,
        scenario_ids=scenario_ids,
        common_history_ids=tuple(f"public-history:{value}" for value in scenario_ids),
        best_response_id="conditional-frozen-best-response",
        induced_flow_id="conditional-independent-forward-law",
        law_distance_id="conditional-maximum-mean-distance",
        problem_id="two-atom-common-noise-mfg",
    )


def _plan(
    maximum_iterations=3,
    *,
    tolerance=1.0e-9,
    damping=1.0,
    minimum_ess=2.0,
    minimum_clusters=2,
):
    return CommonNoiseMeanFieldPlan(
        maximum_iterations=maximum_iterations,
        consistency_tolerance=tolerance,
        damping=damping,
        minimum_effective_sample_size=minimum_ess,
        minimum_independent_clusters=minimum_clusters,
        problem_id="two-atom-common-noise-mfg",
    )


def _balanced_initial():
    return (
        _law(-1.0, "conditional-down", "idiosyncratic-down"),
        _law(1.0, "conditional-up", "idiosyncratic-up"),
    )


def test_two_public_scenarios_keep_distinct_conditional_laws_and_histories():
    histories = (jnp.asarray([-0.5, -1.0]), jnp.asarray([0.5, 1.0]))
    observed = []
    problem = _problem(
        _balanced_initial(),
        histories,
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: _flow_mean(flow),
        observed=observed,
    )

    result = solve_common_noise_mean_field_fixed_point(problem, _plan())

    assert result.status == CommonNoiseMeanFieldStatus.SUCCESS
    assert result.valid
    assert result.scenario_ids == ("down", "up")
    np.testing.assert_allclose(
        [_flow_mean(flow) for flow in result.conditional_flows], [-1.0, 1.0]
    )
    assert (
        sum(
            probability * _flow_mean(flow)
            for probability, flow in zip(
                result.scenario_probabilities, result.conditional_flows, strict=True
            )
        )
        == 0.0
    )
    np.testing.assert_allclose(result.distance_history[0], [0.0, 0.0])
    assert result.common_histories[0] is histories[0]
    assert result.common_histories[1] is histories[1]
    assert [entry[0] for entry in observed] == [
        "response",
        "induced",
        "distance",
        "response",
        "induced",
        "distance",
    ]
    assert observed[0][2] is histories[0]
    assert observed[3][2] is histories[1]
    assert result.best_response_flow_ids[0] == (
        "conditional-down",
        "conditional-up",
    )
    assert result.best_response_common_history_ids[0] == (
        "public-history:down",
        "public-history:up",
    )
    assert result.best_response_path_ids[0][0].startswith("best-response-paths:")
    assert result.induced_source_path_ids[0][0].startswith("independent-forward-paths:")


def test_scenario_permutation_preserves_id_keyed_conditional_evidence():
    initial = _balanced_initial()
    histories = (jnp.asarray([-1.0]), jnp.asarray([1.0]))
    labels = (("a", "b", "c", "d"), ("e", "f", "g", "h"))
    first = solve_common_noise_mean_field_fixed_point(
        _problem(
            initial,
            histories,
            jnp.asarray([0.3, 0.7]),
            labels,
            lambda flow, history: _flow_mean(flow),
        ),
        _plan(),
    )
    second = solve_common_noise_mean_field_fixed_point(
        _problem(
            initial[::-1],
            histories[::-1],
            jnp.asarray([0.7, 0.3]),
            labels[::-1],
            lambda flow, history: _flow_mean(flow),
            scenario_ids=("up", "down"),
        ),
        _plan(),
    )

    first_by_id = dict(
        zip(first.scenario_ids, np.asarray(first.final_distances), strict=True)
    )
    second_by_id = dict(
        zip(second.scenario_ids, np.asarray(second.final_distances), strict=True)
    )
    assert first_by_id == second_by_id == {"down": 0.0, "up": 0.0}
    np.testing.assert_allclose(
        first.aggregate_distance_history[0], second.aggregate_distance_history[0]
    )


def test_zero_probability_atom_is_retained_but_not_required_or_evaluated():
    histories = (jnp.asarray([-1.0]), jnp.asarray([9.0]))
    observed = []
    problem = _problem(
        (
            _law(-1.0, "supported", "supported-paths"),
            _law(99.0, "null-atom", "null-path", particles=1),
        ),
        histories,
        jnp.asarray([1.0, 0.0]),
        (("a", "b", "c", "d"), ("only-cluster",)),
        lambda flow, history: _flow_mean(flow),
        observed=observed,
    )

    result = solve_common_noise_mean_field_fixed_point(problem, _plan())

    assert result.valid
    assert result.supported_scenarios.tolist() == [True, False]
    assert jnp.isnan(result.final_distances[1])
    assert result.scenario_status_history[0, 1] == int(
        CommonNoiseMeanFieldStatus.ZERO_PROBABILITY_SCENARIO
    )
    assert all(entry[2] is histories[0] for entry in observed)


def test_positive_atom_with_one_independent_cluster_is_rejected():
    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("same", "same", "same", "same"), ("e", "f", "g", "h")),
        lambda flow, history: _flow_mean(flow),
    )

    result = solve_common_noise_mean_field_fixed_point(problem, _plan())

    assert result.status == CommonNoiseMeanFieldStatus.INSUFFICIENT_INDEPENDENT_CLUSTERS
    assert not result.valid
    assert result.independent_cluster_count_history[0, 0] == 1
    assert not result.best_response_validity_history[0, 0]


def test_conditional_forward_laws_must_use_distinct_idiosyncratic_paths():
    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: _flow_mean(flow),
        induced_source=lambda response, history: "shared-forward-paths",
    )

    result = solve_common_noise_mean_field_fixed_point(problem, _plan())

    assert result.status == CommonNoiseMeanFieldStatus.INVALID_INDUCED_LAW
    assert not result.valid
    assert result.induced_flow_validity_history[0].tolist() == [True, False]


def test_matching_unconditional_mean_cannot_replace_conditional_consistency():
    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: -_flow_mean(flow),
    )

    result = solve_common_noise_mean_field_fixed_point(
        problem, _plan(maximum_iterations=1, tolerance=1.0e-12)
    )

    current_unconditional_mean = sum(
        probability * _flow_mean(flow)
        for probability, flow in zip(
            result.scenario_probabilities,
            problem.initial_conditional_flows,
            strict=True,
        )
    )
    induced_unconditional_mean = sum(
        probability * _flow_mean(flow)
        for probability, flow in zip(
            result.scenario_probabilities,
            result.induced_conditional_flows,
            strict=True,
        )
    )
    np.testing.assert_allclose(current_unconditional_mean, induced_unconditional_mean)
    np.testing.assert_allclose(result.distance_history[0], [2.0, 2.0])
    assert result.status == CommonNoiseMeanFieldStatus.MAX_ITERATIONS
    assert not result.valid
    assert not result.unconditional_law_consistency_evaluated


def test_damping_is_applied_separately_inside_each_conditional_law():
    observed_means = []
    mixture_histories = []

    def induced_target(flow, history):
        observed_means.append((_flow_mean(flow), float(history[0])))
        return 1.0 if float(history[0]) < 0.0 else 3.0

    def conditional_mixture(current, induced, damping, iteration, history, args):
        mixture_histories.append(float(history[0]))
        return _law_mixture(current, induced, damping, iteration, history, args)

    problem = _problem(
        (
            _law(0.0, "damping-down", "damping-down-paths"),
            _law(2.0, "damping-up", "damping-up-paths"),
        ),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        induced_target,
        law_mixture=conditional_mixture,
    )

    result = solve_common_noise_mean_field_fixed_point(
        problem, _plan(maximum_iterations=2, tolerance=0.0, damping=0.25)
    )

    assert observed_means == [(0.0, -1.0), (2.0, 1.0), (0.25, -1.0), (2.25, 1.0)]
    assert mixture_histories == [-1.0, 1.0, -1.0, 1.0]
    np.testing.assert_allclose(
        [_flow_mean(flow) for flow in result.conditional_flows], [0.4375, 2.4375]
    )
    np.testing.assert_array_equal(
        jnp.unique(result.conditional_flows[0].particles), jnp.asarray([0.0, 1.0])
    )
    np.testing.assert_array_equal(
        jnp.unique(result.conditional_flows[1].particles), jnp.asarray([2.0, 3.0])
    )
    assert result.current_flow_ids[1][0].startswith("conditional-union-mixture:0:")
    assert result.current_flow_ids[1][1].startswith("conditional-union-mixture:0:")
    assert result.law_mixture_id == "conditional-exact-union-support-mixture"


def test_subunit_conditional_damping_requires_an_identified_law_mixture():
    with pytest.raises(ValueError, match="must be supplied together"):
        _problem(
            _balanced_initial(),
            (jnp.asarray([-1.0]), jnp.asarray([1.0])),
            jnp.asarray([0.5, 0.5]),
            (("a", "b", "c", "d"), ("e", "f", "g", "h")),
            lambda flow, history: -_flow_mean(flow),
            law_mixture=_law_mixture,
            law_mixture_id=None,
        )

    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: -_flow_mean(flow),
        law_mixture=None,
        law_mixture_id=None,
    )

    with pytest.raises(ValueError, match="law_mixture and law_mixture_id"):
        solve_common_noise_mean_field_fixed_point(
            problem, _plan(maximum_iterations=1, damping=0.5)
        )


def test_invalid_conditional_law_mixture_callback_fails_closed():
    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: -_flow_mean(flow),
        law_mixture=lambda current, induced, damping, iteration, history, args: current,
    )

    result = solve_common_noise_mean_field_fixed_point(
        problem, _plan(maximum_iterations=1, tolerance=0.0, damping=0.5)
    )

    assert result.status == CommonNoiseMeanFieldStatus.INVALID_LAW_MIXTURE
    assert result.scenario_statuses.tolist() == [
        int(CommonNoiseMeanFieldStatus.INVALID_LAW_MIXTURE),
        int(CommonNoiseMeanFieldStatus.INVALID_LAW_MIXTURE),
    ]
    assert not result.valid


def test_unit_conditional_damping_uses_each_induced_law_directly():
    def forbidden_mixture(current, induced, damping, iteration, history, args):
        raise AssertionError("law_mixture must not be called when damping is one")

    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: -_flow_mean(flow),
        law_mixture=forbidden_mixture,
    )

    result = solve_common_noise_mean_field_fixed_point(
        problem, _plan(maximum_iterations=1, tolerance=0.0, damping=1.0)
    )

    assert result.status == CommonNoiseMeanFieldStatus.MAX_ITERATIONS
    assert all(
        flow is induced
        for flow, induced in zip(
            result.conditional_flows,
            result.induced_conditional_flows,
            strict=True,
        )
    )
    assert all(flow.source_path_id is not None for flow in result.conditional_flows)


def test_nonconvergent_conditional_map_exhausts_outer_capacity():
    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: -_flow_mean(flow),
    )

    result = solve_common_noise_mean_field_fixed_point(
        problem, _plan(maximum_iterations=4, tolerance=1.0e-12, damping=1.0)
    )

    assert result.status == CommonNoiseMeanFieldStatus.MAX_ITERATIONS
    assert result.iterations == 4
    assert result.accepted_iterations == 4
    np.testing.assert_allclose(result.distance_history, 2.0)
    assert jnp.all(result.scenario_iteration_validity_history)
    assert not result.converged


def test_success_label_is_candidate_evidence_without_unconditional_or_mfc_claim():
    problem = _problem(
        _balanced_initial(),
        (jnp.asarray([-1.0]), jnp.asarray([1.0])),
        jnp.asarray([0.5, 0.5]),
        (("a", "b", "c", "d"), ("e", "f", "g", "h")),
        lambda flow, history: _flow_mean(flow),
    )

    result = solve_common_noise_mean_field_fixed_point(problem, _plan())

    assert result.certificate_label == COMMON_NOISE_MFG_FIXED_POINT_CANDIDATE
    assert result.candidate_evaluation_only
    assert result.conditional_law_consistency_evaluated
    assert not result.unconditional_law_consistency_evaluated
    assert not result.best_response_optimality_evaluated
    assert not result.mean_field_game_equilibrium_claimed
    assert not result.common_noise_equilibrium_claimed
    assert not result.unconditional_mean_field_equilibrium_claimed
    assert not result.mean_field_control_optimum_claimed
    assert not result.finite_population_game_claimed
