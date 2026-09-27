from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx


def _regression_design(time: Any, context: Any) -> Any:
    return jnp.asarray((1.0, time + context.args["design_shift"]))


def _deterministic_transition(t0: Any, t1: Any, context: Any) -> Any:
    interval = t1 - t0
    return jnp.asarray(((1.0, interval), (0.0, context.args["deterministic_retention"])))


def _deterministic_observation(time: Any, context: Any) -> Any:
    del time
    return jnp.asarray((context.args["deterministic_loading"], 0.0))


def test_structural_state_space_scenario_1() -> None:
    components = (
        phx.stochastic.DampedTrendComponent(
            "trend",
            damping=0.8,
            level_variance=0.1,
            slope_variance=0.2,
            initial_mean=(2.0, -0.5),
            initial_covariance=jnp.diag(jnp.asarray((3.0, 4.0))),
        ),
        phx.stochastic.SeasonalComponent(
            "quarterly",
            4.0,
            harmonics=1,
            process_variance=0.3,
            initial_covariance=2.0,
        ),
        phx.stochastic.RegressionComponent(
            "regression",
            _regression_design,
            initial_coefficients=jnp.asarray((0.25, -0.75)),
            initial_covariance=jnp.diag(jnp.asarray((5.0, 6.0))),
            process_covariance=0.0,
        ),
        phx.stochastic.AutoregressiveComponent(
            "ar2",
            jnp.asarray((0.6, -0.2)),
            process_variance=0.4,
            initial_covariance=7.0,
        ),
        phx.stochastic.DeterministicTransitionComponent(
            "deterministic",
            _deterministic_transition,
            _deterministic_observation,
            initial_mean=jnp.asarray((1.0, 0.0)),
            initial_covariance=8.0,
            transition_id="physical-deterministic-transition",
        ),
    )
    model = phx.stochastic.compile_structural_state_space(
        components,
        0.25,
        model_id="known-structural",
        parameter_id="theta",
        discretization_id="physical-time",
    )
    context = phx.stochastic.StateSpaceStepContext.empty(
        args={
            "design_shift": jnp.asarray(0.5),
            "deterministic_retention": jnp.asarray(0.9),
            "deterministic_loading": jnp.asarray(2.0),
        }
    )
    assert isinstance(model.transition, phx.stochastic.LinearGaussianTransitionKernel)
    assert isinstance(model.observation, phx.stochastic.LinearGaussianObservationModel)
    parameters = model.transition.parameters(0.0, 1.0, context)
    matrix, offset, observation_covariance = model.observation.parameters(2.0, context)
    slices = model.metadata["structural_component_slices"]

    assert model.state_shape == (10,)
    assert slices["trend"] == slice(0, 2)
    assert slices["quarterly"] == slice(2, 4)
    assert slices["regression"] == slice(4, 6)
    assert slices["ar2"] == slice(6, 8)
    assert slices["deterministic"] == slice(8, 10)
    assert jnp.allclose(
        parameters.transition[slices["trend"], slices["trend"]],
        jnp.asarray(((1.0, 0.8), (0.0, 0.8))),
    )
    assert jnp.allclose(
        parameters.transition[slices["quarterly"], slices["quarterly"]],
        jnp.asarray(((0.0, 1.0), (-1.0, 0.0))),
        atol=1e-6,
    )
    assert jnp.allclose(
        parameters.transition[slices["ar2"], slices["ar2"]],
        jnp.asarray(((0.6, -0.2), (1.0, 0.0))),
    )
    assert jnp.allclose(
        parameters.transition[slices["deterministic"], slices["deterministic"]],
        jnp.asarray(((1.0, 1.0), (0.0, 0.9))),
    )
    assert jnp.allclose(
        parameters.covariance[0:2, 0:2], jnp.diag(jnp.asarray((0.1, 0.2)))
    )
    assert jnp.allclose(parameters.covariance[2:4, 2:4], 0.3 * jnp.eye(2))
    assert parameters.covariance[6, 6] == pytest.approx(0.4)
    assert jnp.allclose(
        matrix,
        jnp.asarray([[1.0, 0.0, 1.0, 0.0, 1.0, 2.5, 1.0, 0.0, 2.0, 0.0]]),
    )
    assert jnp.allclose(offset, jnp.zeros((1,)))
    assert jnp.allclose(observation_covariance, jnp.asarray(((0.25,),)))
    assert model.approximation_id == "exact-structural-linear-gaussian"
    provenance = model.metadata["structural_component_provenance"]
    assert (
        tuple(item.name for item in provenance)
        == model.metadata["structural_component_order"]
    )
    assert provenance[4].transition_id == "physical-deterministic-transition"
    context = phx.stochastic.StateSpaceStepContext.empty()
    level = phx.stochastic.LocalLevelComponent(
        "level", process_variance=0.25, initial_variance=2.0
    )
    trend = phx.stochastic.TrendComponent("trend", level_variance=0.1, slope_variance=0.3)
    damped = phx.stochastic.DampedTrendComponent(
        "damped",
        damping=0.5,
        level_variance=0.1,
        slope_variance=0.3,
    )

    assert jnp.allclose(
        # ty: ignore[invalid-argument-type]
        level.process_covariance(1.0, 3.5, context),
        jnp.asarray(((0.625,),)),
    )
    assert jnp.allclose(
        # ty: ignore[invalid-argument-type]
        trend.transition_matrix(1.0, 3.5, context),
        jnp.asarray(((1.0, 2.5), (0.0, 1.0))),
    )
    assert jnp.allclose(
        # ty: ignore[invalid-argument-type]
        trend.process_covariance(1.0, 3.5, context),
        jnp.diag(jnp.asarray((0.25, 0.75))),
    )
    assert jnp.allclose(
        # ty: ignore[invalid-argument-type]
        damped.transition_matrix(0.0, 2.0, context),
        jnp.asarray(((1.0, 0.75), (0.0, 0.25))),
    )
    model = phx.stochastic.compile_structural_state_space(
        (
            phx.stochastic.LocalLevelComponent(
                "level", process_variance=0.1, initial_variance=1.0
            ),
            phx.stochastic.SeasonalComponent(
                "seasonal", 6.0, harmonics=1, initial_covariance=0.5
            ),
        ),
        0.2,
        case_shape=(2,),
        model_id="case-structural",
    )
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray((1.0, 2.0, 3.0)),
        jnp.asarray(
            (
                ((0.0,), (1.0,), (2.0,)),
                ((3.0,), (4.0,), (0.0,)),
            )
        ),
        case_axes=("experiment_case",),
        case_shape=(2,),
        step_valid=jnp.asarray(((True, True, True), (True, True, False))),
        observation_mask=jnp.asarray(
            (
                ((True,), (False,), (True,)),
                ((True,), (True,), (False,)),
            )
        ),
        case_ids=("case-a", "case-b"),
        sequence_id="masked-structural",
    )
    problem = phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=jnp.asarray((0.0, 0.25)),
        problem_id="masked-structural-problem",
    )
    result = phx.uq.exact_state_space_log_likelihood(problem)

    assert isinstance(model.prior, phx.stochastic.GaussianStatePrior)
    assert model.prior.batch_shape == (2,)
    assert model.prior.mean.shape == (2, 3)
    assert problem.observations.case_axes == ("experiment_case",)
    assert problem.observations.case_ids == ("case-a", "case-b")
    assert jnp.array_equal(result.step_valid, sequence.step_valid)
    assert result.incremental_log_likelihood[0, 1] == 0.0
    assert result.incremental_log_likelihood[1, 2] == 0.0
    assert jnp.all(result.successful)


def test_structural_state_space_scenario_2() -> None:
    for components, message in [
        (
            (
                phx.stochastic.LocalLevelComponent("level", process_variance=0.1),
                phx.stochastic.TrendComponent(
                    "trend", level_variance=0.1, slope_variance=0.1
                ),
            ),
            "redundant",
        ),
        (
            (
                phx.stochastic.TrendComponent(
                    "signal", level_variance=0.1, slope_variance=0.1
                ),
                phx.stochastic.ProcessNoiseComponent("signal", variance=0.2),
            ),
            "unique",
        ),
        (
            (
                phx.stochastic.LocalLevelComponent("level-a", process_variance=0.1),
                phx.stochastic.LocalLevelComponent("level-b", process_variance=0.2),
            ),
            "label-unidentifiable",
        ),
        (
            (
                phx.stochastic.SeasonalComponent("annual", 12.0, harmonics=2),
                phx.stochastic.SeasonalComponent("semiannual", 6.0, harmonics=1),
            ),
            "duplicated harmonic",
        ),
        (
            (
                phx.stochastic.AutoregressiveComponent(
                    "ar-a",
                    # ty: ignore[invalid-argument-type]
                    (0.5,),
                    process_variance=0.1,
                ),
                phx.stochastic.AutoregressiveComponent(
                    "ar-b",
                    # ty: ignore[invalid-argument-type]
                    (-0.5,),
                    process_variance=0.1,
                ),
            ),
            "label-unidentifiable",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            phx.stochastic.compile_structural_state_space(components, 0.2)
    with pytest.raises(ValueError, match="unidentifiable"):
        phx.stochastic.RegressionComponent(
            "fixed",
            jnp.asarray((1.0, 2.0)),
            initial_coefficients=jnp.zeros((2,)),
        )
    with pytest.raises(ValueError, match="identically zero.*unidentifiable"):
        phx.stochastic.RegressionComponent(
            "zero-design",
            jnp.asarray((0.0,)),
            initial_coefficients=jnp.asarray((1.0,)),
        )
    model = phx.stochastic.compile_structural_state_space(
        (
            phx.stochastic.LocalLevelComponent(
                "level",
                process_variance=0.0,
                initial_variance=0.0,
            ),
        ),
        0.0,
    )
    context = phx.stochastic.StateSpaceStepContext.empty()
    assert isinstance(model.observation, phx.stochastic.LinearGaussianObservationModel)
    _, _, covariance = model.observation.parameters(jnp.asarray(1.0), context)
    matched = model.observation.log_prob(
        jnp.asarray([1.5]),
        jnp.asarray([1.5]),
        jnp.asarray(1.0),
        jnp.asarray([True]),
        context,
    )
    mismatched = model.observation.log_prob(
        jnp.asarray([2.0]),
        jnp.asarray([1.5]),
        jnp.asarray(1.0),
        jnp.asarray([True]),
        context,
    )
    masked = model.observation.log_prob(
        jnp.asarray([2.0]),
        jnp.asarray([1.5]),
        jnp.asarray(1.0),
        jnp.asarray([False]),
        context,
    )

    assert jnp.array_equal(covariance, jnp.zeros((1, 1)))
    assert jnp.isfinite(matched)
    assert matched == 0.0
    assert mismatched == -jnp.inf
    assert masked == 0.0


def test_structural_state_space_scenario_3() -> None:
    component = phx.stochastic.ProcessNoiseComponent("white", variance=0.5)

    model = phx.stochastic.compile_structural_state_space((component,), 0.25)
    context = phx.stochastic.StateSpaceStepContext.empty()
    assert isinstance(model.transition, phx.stochastic.LinearGaussianTransitionKernel)
    assert isinstance(model.observation, phx.stochastic.LinearGaussianObservationModel)
    transition_matrix, _, process_covariance = model.transition.parameters(
        jnp.asarray(0.0), jnp.asarray(1.0), context
    )
    observation_matrix, _, observation_covariance = model.observation.parameters(
        jnp.asarray(1.0), context
    )

    assert jnp.array_equal(transition_matrix, jnp.zeros((1, 1)))
    assert jnp.array_equal(process_covariance, jnp.asarray([[0.5]]))
    assert jnp.array_equal(observation_matrix, jnp.ones((1, 1)))
    assert jnp.array_equal(observation_covariance, jnp.asarray([[0.25]]))
    trend = phx.stochastic.TrendComponent("trend", level_variance=0.1, slope_variance=0.1)

    with pytest.raises(ValueError, match="at most 1 states"):
        phx.stochastic.compile_structural_state_space((trend,), 0.2, max_state_size=1)
