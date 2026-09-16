#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _solve_training_candidate():
    training = phx.solver.prepare_elliptic_curve(jax.random.key(101), line_count=2)
    result = phx.solver.solve_calabi_yau_metric(
        training.problem,
        policy=phx.solver.CalabiYauSolvePolicy(
            iterations=1,
            learning_rate=1e-4,
            maximum_backtracks=1,
        ),
    )
    heldout = phx.solver.prepare_elliptic_curve(jax.random.key(202), line_count=2)
    assert heldout.hypersurface.hypersurface_id == training.hypersurface.hypersurface_id
    plan = phx.solver.CalabiYauMetricEvidencePlan(
        training.problem.samples,
        heldout.problem.samples,
        batch_count=2,
        residual_rms_tolerance=1e6,
        positivity_floor=0.0,
        minimum_valid_fraction=0.5,
    )
    return training, heldout, result, plan


def test_heldout_metric_evidence_retains_weights_ess_and_ancestry():
    training, _, result, plan = _solve_training_candidate()
    evidence = phx.solver.evaluate_calabi_yau_metric_evidence(
        result,
        training.hypersurface,
        plan,
    )
    assert evidence.training_sample_id != evidence.heldout_sample_id
    np.testing.assert_allclose(
        jnp.sum(evidence.normalized_weights), 1.0, rtol=1e-12, atol=1e-12
    )
    assert float(evidence.effective_sample_size) > 0.0
    assert evidence.absolute_residual_quantiles.shape == (3,)
    assert evidence.batch_rms_residuals.shape == (2,)
    assert bool(evidence.kahler_by_construction)
    assert not bool(evidence.ricci_available)
    assert "no-exact-ricci-flat" in evidence.claim


def test_frozen_metric_artifact_includes_heldout_evidence_identity():
    training, _, result, plan = _solve_training_candidate()
    evidence = phx.solver.evaluate_calabi_yau_metric_evidence(
        result,
        training.hypersurface,
        plan,
    )
    artifact = phx.solver.freeze_calabi_yau_result(
        result,
        training.hypersurface,
        evidence=evidence,
    )
    metadata = artifact.metadata()
    assert metadata["metric_evidence_id"] == evidence.evidence_id
    assert metadata["heldout_evidence_accepted"] == bool(evidence.accepted)
    point = plan.heldout_samples.homogeneous_points[0]
    evaluation = artifact.evaluate(training.hypersurface, point)
    assert jnp.all(jnp.isfinite(evaluation.metric))


def test_metric_evidence_refuses_shared_train_holdout_points():
    campaign = phx.solver.prepare_elliptic_curve(jax.random.key(303), line_count=1)
    with pytest.raises(ValueError, match="share an exact point"):
        phx.solver.CalabiYauMetricEvidencePlan(
            campaign.problem.samples,
            campaign.problem.samples,
            batch_count=1,
        )


def test_required_ricci_evaluator_is_explicit():
    training, _, result, plan = _solve_training_candidate()
    required = phx.solver.CalabiYauMetricEvidencePlan(
        plan.training_samples,
        plan.heldout_samples,
        batch_count=2,
        residual_rms_tolerance=1e6,
        positivity_floor=0.0,
        minimum_valid_fraction=0.5,
        require_ricci=True,
        ricci_tolerance=1e-12,
    )
    with pytest.raises(ValueError, match="requires"):
        phx.solver.evaluate_calabi_yau_metric_evidence(
            result, training.hypersurface, required
        )
    evidence = phx.solver.evaluate_calabi_yau_metric_evidence(
        result,
        training.hypersurface,
        required,
        ricci_evaluator=lambda point, chart, pivot: jnp.asarray(0.0),
    )
    assert bool(evidence.ricci_available)
    np.testing.assert_allclose(evidence.maximum_ricci_residual, 0.0)
