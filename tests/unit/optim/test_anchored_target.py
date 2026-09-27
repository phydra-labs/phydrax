#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._anchored_target import (
    AnchoredResponseModel,
    AnchoredTargetMethod,
    AnchoredTargetProblem,
    solve_anchored_target,
)


def _response(design: Any, args: Any) -> Any:
    del args
    return design, jnp.asarray(True), {"evaluated_design": design}


def _problem(
    fine: Any = _response,
    predictor: Any = _response,
    *,
    target: Any = 1.0,
    bounds: Any = None,
    **kwargs: Any,
) -> Any:
    return AnchoredTargetProblem(
        fine,
        AnchoredResponseModel(predictor),
        targets=jnp.asarray([target]),
        response_names=("physical-response",),
        scales=jnp.asarray([1.0]),
        bounds=phx.optim.Bounds(-4.0, 4.0) if bounds is None else bounds,
        **kwargs,
    )


def test_multiplicative_exact_scale_target_uses_identical_merit_and_one_trial() -> None:
    physical_scale = jnp.asarray([1.0e3, 1.0e-3])

    def fine(design: Any, args: Any) -> Any:
        del args
        return physical_scale * design, jnp.asarray(True), {"state": design}

    problem = AnchoredTargetProblem(
        fine,
        AnchoredResponseModel(
            _response, correction="multiplicative", minimum_denominator=1e-8
        ),
        targets=physical_scale * jnp.asarray([1.5, 2.0]),
        scales=physical_scale,
        response_names=("force", "displacement"),
        bounds=phx.optim.Bounds(0.1, 3.0),
    )
    result = eqx.filter_jit(solve_anchored_target)(
        problem,
        jnp.asarray([1.0, 1.0]),
        method=AnchoredTargetMethod(maximum_evaluations=2, target_tolerance=1e-7),
    )
    assert result.successful
    assert int(result.evaluations) == 2
    np.testing.assert_allclose(result.design, [1.5, 2.0], atol=1e-7)
    np.testing.assert_allclose(result.history["ratio"][0], 1.0, atol=1e-10)
    np.testing.assert_allclose(
        result.history["actual_reduction"][0],
        result.history["predicted_reduction"][0],
        atol=1e-12,
    )


def test_anchored_target_scenario_1() -> None:
    for correction in ["additive", "multiplicative"]:
        model = AnchoredResponseModel(
            _response,
            correction=correction,
            **({"minimum_denominator": 1e-15} if correction == "multiplicative" else {}),
        )
        model_anchor = jnp.asarray([1.0e12, 1.0e-8])
        fine_anchor = jnp.asarray([1.0e-8, 1.0e12])
        np.testing.assert_array_equal(
            model.correct(model_anchor, model_anchor, fine_anchor), fine_anchor
        )
    result = solve_anchored_target(
        _problem(target=2.0, bounds=phx.optim.Bounds(0.0, 1.0)),
        jnp.asarray([0.0]),
    )
    assert not result.successful
    assert int(result.status) == int(phx.optim.OptimizationStatus.STAGNATION)
    np.testing.assert_allclose(result.design, [1.0], atol=1e-8)
    np.testing.assert_allclose(result.target_error, 1.0, atol=1e-8)
    assert int(result.evaluations) == 2
    result = solve_anchored_target(
        _problem(),
        jnp.asarray([0.0]),
        method=AnchoredTargetMethod(initial_radius=0.1, maximum_evaluations=2),
    )
    assert not result.successful
    assert result.accepted
    assert int(result.status) == int(
        phx.optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    np.testing.assert_allclose(result.design, [0.1], atol=1e-8)
    assert result.target_error > 0.8


def test_unsafe_multiplicative_anchor_is_rejected_without_a_physical_trial() -> None:
    for denominator in [0.0, 1e-13, -1e-13]:

        def model(design: Any, args: Any) -> Any:
            del args
            return design + denominator, jnp.asarray(True), design

        problem = AnchoredTargetProblem(
            _response,
            AnchoredResponseModel(
                model, correction="multiplicative", minimum_denominator=1e-10
            ),
            targets=jnp.asarray([1.0]),
            scales=1.0,
            response_names=("load",),
            bounds=phx.optim.Bounds(-2.0, 2.0),
        )
        result = solve_anchored_target(problem, jnp.asarray([0.0]))
        assert not result.successful
        assert int(result.status) == int(
            phx.optim.OptimizationStatus.CERTIFICATION_FAILED
        )
        assert int(result.evaluations) == 1
        np.testing.assert_array_equal(result.design, [0.0])
        np.testing.assert_array_equal(result.values, [0.0])


def test_wrong_model_rejections_keep_fixed_anchor_and_consume_budget() -> None:
    def reversed_model(design: Any, args: Any) -> Any:
        del args
        return -design, jnp.asarray(True), {"evaluated_design": design}

    initial = jnp.asarray([0.0])
    result = solve_anchored_target(
        _problem(predictor=reversed_model),
        initial,
        method=AnchoredTargetMethod(maximum_evaluations=4),
    )
    assert not result.successful
    assert int(result.status) == int(
        phx.optim.OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert int(result.evaluations) == 4
    assert int(result.rejected_steps) == 3
    np.testing.assert_array_equal(result.design, initial)
    np.testing.assert_array_equal(result.fine_evidence["evaluated_design"], initial)
    np.testing.assert_array_equal(result.model_evidence["evaluated_design"], initial)
    np.testing.assert_array_equal(result.history["anchor_design"][:3], np.zeros((3, 1)))
    np.testing.assert_allclose(result.history["radius"][:3], [1.0, 0.5, 0.25])
    assert np.all(np.asarray(result.history["actual_reduction"][:3]) < 0)
    assert np.all(np.asarray(result.history["predicted_reduction"][:3]) > 0)


def test_invalid_fine_trials_roll_back_state_values_and_evidence() -> None:
    def fine(design: Any, args: Any) -> Any:
        del args
        valid = design[0] <= 0.25
        # A converged-looking number does not override explicit physical failure.
        return design, valid, {"accepted_state": valid, "evaluated_design": design}

    result = solve_anchored_target(
        _problem(fine=fine),
        jnp.asarray([0.0]),
        method=AnchoredTargetMethod(maximum_evaluations=3),
    )
    assert not result.successful
    assert int(result.evaluations) == 3
    np.testing.assert_array_equal(result.design, [0.0])
    np.testing.assert_array_equal(result.values, [0.0])
    np.testing.assert_array_equal(result.fine_evidence["evaluated_design"], [0.0])
    assert result.fine_evidence["accepted_state"]
    assert not result.last_trial_evidence["accepted_state"]
    assert np.all(np.asarray(result.history["fine_evaluated"][:2]))
    assert not np.any(np.asarray(result.history["accepted"][:2]))


def test_invalid_initial_physics_never_claims_target_success() -> None:
    def invalid(design: Any, args: Any) -> Any:
        del args
        return jnp.ones_like(design), jnp.asarray(False), {"residual": jnp.asarray(3.0)}

    result = solve_anchored_target(_problem(fine=invalid), jnp.asarray([0.0]))
    assert not result.successful
    assert not result.accepted
    assert int(result.evaluations) == 1
    assert int(result.iterations) == 0


def test_anchored_target_scenario_2() -> None:
    constraints = (
        phx.optim.NonlinearConstraint(
            lambda point, args: point[1],
            lower=0.0,
            upper=0.0,
            constraint_id="fixed-response",
        ),
        phx.optim.NonlinearConstraint(
            lambda point, args: point[1],
            lower=-2.0,
            upper=0.5,
            constraint_id="response-envelope",
        ),
    )
    result = solve_anchored_target(
        _problem(constraints=constraints),
        jnp.asarray([0.0]),
        method=AnchoredTargetMethod(maximum_evaluations=3, constraint_weight=10.0),
    )
    assert not result.successful
    # Equality first; all lower rows (response, design), then upper rows.
    x = float(result.design[0])
    np.testing.assert_allclose(result.equality, [x], atol=1e-9)
    np.testing.assert_allclose(result.inequality, [-2.0 - x, -4.0 - x, x - 0.5, x - 4.0])
    assert result.target_error > 0.8
    assert result.constraint_violation > 1e-3
    expected = 0.5 * ((x - 1.0) ** 2 + 10.0 * x**2)
    np.testing.assert_allclose(result.merit, expected, atol=1e-10)
    np.testing.assert_allclose(result.history["ratio"][0], 1.0, atol=1e-10)
    with pytest.raises(ValueError):
        AnchoredResponseModel(_response, correction="multiplicative")
    with pytest.raises(ValueError):
        _problem(realization="resampled")
    with pytest.raises(ValueError):
        AnchoredTargetProblem(
            _response,
            AnchoredResponseModel(_response),
            targets=[1.0, 2.0],
            response_names=("force", "force"),
            scales=1.0,
            bounds=phx.optim.Bounds(),
        )


def test_coupled_native_implicit_predictor_matches_physical_target() -> None:
    def coupled_residual(state: Any, design: Any) -> Any:
        return jnp.asarray(
            [
                2.0 * state[0] + state[1] - design[0],
                state[0] + 3.0 * state[1] - design[1],
            ]
        )

    def predictor(design: Any, args: Any) -> Any:
        del args
        state = phx.optim.implicit_least_squares(
            coupled_residual,
            jnp.zeros((2,)),
            args=design,
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1e-10,
                relative_optimality=0.0,
                maximum_steps=16,
            ),
        )
        defect = jnp.max(jnp.abs(coupled_residual(state, design)))
        return state, defect < 1e-8, {"state": state, "defect": defect}

    def fine(design: Any, args: Any) -> Any:
        state, accepted, evidence = predictor(design, args)
        return 2.0 * state, accepted, evidence

    problem = AnchoredTargetProblem(
        fine,
        AnchoredResponseModel(
            predictor,
            correction="multiplicative",
            minimum_denominator=1e-8,
        ),
        targets=jnp.asarray([1.0, 1.0]),
        scales=jnp.asarray([1.0, 1.0]),
        response_names=("coupled-displacement-a", "coupled-displacement-b"),
        bounds=phx.optim.Bounds(0.1, 4.0),
    )
    result = solve_anchored_target(
        problem,
        jnp.asarray([1.0, 1.0]),
        method=AnchoredTargetMethod(target_tolerance=1e-7, maximum_evaluations=3),
    )
    assert result.successful
    np.testing.assert_allclose(result.design, [1.5, 2.0], atol=1e-7)
    np.testing.assert_allclose(result.values, [1.0, 1.0], atol=1e-7)
    assert result.model_evidence["defect"] < 1e-8


def test_invalid_predictor_does_not_spend_physical_trial_budget() -> None:
    def predictor(design: Any, args: Any) -> Any:
        del args
        return design, jnp.asarray(False), {"defect": jnp.asarray(1.0)}

    result = solve_anchored_target(_problem(predictor=predictor), jnp.asarray([0.0]))
    assert not result.successful
    assert int(result.status) == int(phx.optim.OptimizationStatus.CERTIFICATION_FAILED)
    assert int(result.evaluations) == 1
    np.testing.assert_array_equal(result.design, [0.0])


def test_host_blackbox_is_never_traced_and_matches_target() -> None:
    evaluations = []

    def blackbox(design: Any, args: Any) -> Any:
        del args
        coordinate = float(design[0])
        evaluations.append(coordinate)
        return (
            jnp.asarray([coordinate**2]),
            jnp.asarray(True),
            {"coordinate": jnp.asarray(coordinate)},
        )

    def approximate(design: Any, args: Any) -> Any:
        del args
        return design**2, jnp.asarray(True), {"coordinate": design[0]}

    result = solve_anchored_target(
        _problem(fine=blackbox, predictor=approximate),
        jnp.asarray([0.5]),
        execution="host",
    )
    assert result.successful
    np.testing.assert_allclose(result.design, [1.0], atol=1e-7)
    assert len(evaluations) == int(result.evaluations)
    np.testing.assert_allclose(result.fine_evidence["coordinate"], evaluations[-1])


def test_host_rejected_blackbox_calls_are_budgeted_and_not_promoted() -> None:
    evaluations = []

    def blackbox(design: Any, args: Any) -> Any:
        del args
        coordinate = float(design[0])
        evaluations.append(coordinate)
        return (
            jnp.asarray([coordinate]),
            jnp.asarray(True),
            {"coordinate": jnp.asarray(coordinate)},
        )

    def wrong_predictor(design: Any, args: Any) -> Any:
        del args
        return -design, jnp.asarray(True), {"coordinate": design[0]}

    result = solve_anchored_target(
        _problem(fine=blackbox, predictor=wrong_predictor),
        jnp.asarray([0.0]),
        method=AnchoredTargetMethod(maximum_evaluations=4),
        execution="host",
    )
    assert not result.successful
    np.testing.assert_allclose(evaluations, [0.0, -1.0, -0.5, -0.25], atol=1e-8)
    assert len(evaluations) == int(result.evaluations) == 4
    np.testing.assert_array_equal(result.design, [0.0])
    np.testing.assert_array_equal(result.fine_evidence["coordinate"], 0.0)
    np.testing.assert_allclose(result.last_trial_evidence["coordinate"], -0.25, atol=1e-8)


def test_failed_inner_prediction_never_promotes_an_unaccepted_trial() -> None:
    def invalid_away_from_anchor(design: Any, args: Any) -> Any:
        del args
        return design, design[0] == 0.0, {"coordinate": design[0]}

    result = solve_anchored_target(
        _problem(predictor=invalid_away_from_anchor),
        jnp.asarray([0.0]),
        method=AnchoredTargetMethod(
            maximum_steps=2,
            inner_termination=phx.optim.OptimizationTermination(maximum_steps=4),
        ),
    )
    assert not result.successful
    assert int(result.evaluations) == 1
    np.testing.assert_array_equal(result.design, [0.0])
    np.testing.assert_array_equal(result.values, [0.0])
    assert not np.any(np.asarray(result.history["fine_evaluated"]))
