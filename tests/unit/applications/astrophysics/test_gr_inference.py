#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from phydrax.applications.astrophysics._gr_inference import (
    fixed_branch_ray_inverse_adapter,
    FixedBranchRayInferencePlan,
    gr_ray_model_evaluation,
    GRPosteriorRealizationBinding,
)
from phydrax.applications.compact_objects._inverse import FixedBranchModelEvaluation
from phydrax.observation import CholeskyCovarianceAction, CoordinateLayout


def _ray_evaluator(parameters):
    prediction = jnp.asarray(
        [parameters[0] ** 2 + 0.5 * parameters[1], jnp.sin(parameters[1])]
    )
    return FixedBranchModelEvaluation(
        prediction,
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        finite=True,
        converged=True,
        physically_valid=True,
        qualified=True,
        derivative_valid=True,
        shock_free=True,
        event_free=True,
        topology_fixed=True,
        realization_id="ray:fixed-camera",
        branch_id="event-free-rays",
    )


def test_fixed_branch_ray_likelihood_and_sensitivity_are_evidenced():
    adapter = fixed_branch_ray_inverse_adapter(
        _ray_evaluator,
        jnp.asarray([0, 0, 1, 1]),
        parameter_count=2,
        output_count=2,
        realization_id="ray:fixed-camera",
        branch_id="event-free-rays",
        adapter_id="ray-forward",
        evaluator_semantic_id="gr-ray-theory",
        evaluator_numeric_id="gr-ray-forward:r1",
    )
    layout = CoordinateLayout(("pixel-0", "pixel-1"))
    covariance = CholeskyCovarianceAction(jnp.diag(jnp.asarray([0.2, 0.3])), layout)
    plan = FixedBranchRayInferencePlan(
        adapter,
        jnp.asarray([1.2, 0.25]),
        covariance,
        observation_id="synthetic-image",
        prior_log_density=lambda value: -0.5 * jnp.sum(value**2),
        plan_id="ray-inference",
    )
    parameters = jnp.asarray([1.1, 0.3])
    result = plan.evaluate(parameters)
    sensitivity = plan.sensitivity(parameters, jnp.asarray([0.2, -0.15]), epsilon=2.0e-4)

    assert bool(result.finite)
    assert bool(result.qualified)
    assert bool(result.derivative_valid)
    assert jnp.isfinite(result.log_density)
    assert bool(sensitivity.derivative_valid)
    assert float(sensitivity.jvp_finite_difference_residual) < 2.0e-3
    assert float(sensitivity.vjp_pairing_residual) < 1.0e-6


def test_ray_events_are_forward_evidence_but_not_gradient_evidence():
    status = SimpleNamespace(
        finite=jnp.asarray(True),
        converged=jnp.asarray(True),
        physically_valid=jnp.asarray(True),
        qualified=jnp.asarray(True),
        derivative_valid=jnp.asarray(True),
    )
    result = SimpleNamespace(
        status=jnp.asarray([0], dtype=jnp.int32),
        event_ledger=SimpleNamespace(
            event_code=jnp.asarray([1], dtype=jnp.int32),
            recorded=jnp.asarray([True]),
        ),
        active=jnp.asarray([[True, False]]),
        status_evidence=status,
    )
    evaluation = gr_ray_model_evaluation(
        result,
        jnp.asarray([3.0]),
        realization_id="ray:captured",
        branch_id="capture",
    )

    assert bool(evaluation.qualified)
    assert not bool(evaluation.event_free)
    assert not bool(evaluation.sensitivity_eligible)


def test_posterior_binding_preserves_chain_draw_axes_and_realization_identity():
    binding = GRPosteriorRealizationBinding(
        "posterior:nuts-42",
        "inference:ray-1",
        "ray:fixed-camera",
        chain_count=2,
        draw_count=3,
    )
    prediction = {
        "intensity": jnp.arange(24.0).reshape(2, 3, 4),
        "valid": jnp.ones((2, 3), dtype="bool"),
    }
    bound = binding.bind_prediction(prediction)

    assert bound.posterior_id == "posterior:nuts-42"
    assert bound.forward_realization_id == "ray:fixed-camera"
    assert bound.values["intensity"].shape == (2, 3, 4)
    changed_realization = GRPosteriorRealizationBinding(
        "posterior:nuts-42",
        "inference:ray-1",
        "ray:other-camera",
        chain_count=2,
        draw_count=3,
    ).bind_prediction(prediction)
    assert changed_realization.binding_id != bound.binding_id
    assert changed_realization.prediction_id != bound.prediction_id

    with pytest.raises(ValueError, match="collapsed"):
        binding.bind_prediction({"intensity": prediction["intensity"].reshape(6, 4)})


def test_ray_inference_rejects_a_switched_branch_in_normal_evaluation():
    def switched(parameters):
        result = _ray_evaluator(parameters)
        return FixedBranchModelEvaluation(
            result.values,
            jnp.asarray([0, 1, 1, 1]),
            finite=result.finite,
            converged=result.converged,
            physically_valid=result.physically_valid,
            qualified=result.qualified,
            derivative_valid=result.derivative_valid,
            realization_id=result.realization_id,
            branch_id=result.branch_id,
        )

    adapter = fixed_branch_ray_inverse_adapter(
        switched,
        jnp.asarray([0, 0, 1, 1]),
        parameter_count=2,
        output_count=2,
        realization_id="ray:fixed-camera",
        branch_id="event-free-rays",
        adapter_id="ray-switched-branch",
        evaluator_semantic_id="gr-ray-theory",
        evaluator_numeric_id="gr-ray-forward:switched",
    )
    layout = CoordinateLayout(("pixel-0", "pixel-1"))
    covariance = CholeskyCovarianceAction(jnp.eye(2), layout)
    result = FixedBranchRayInferencePlan(
        adapter,
        jnp.zeros(2),
        covariance,
        observation_id="synthetic-image",
        plan_id="ray-switched-inference",
    ).evaluate(jnp.asarray([1.0, 0.2]))

    assert bool(result.finite)
    assert bool(result.physically_valid)
    assert not bool(result.qualified)
    assert not bool(result.derivative_valid)
    assert jnp.isneginf(result.log_density)
