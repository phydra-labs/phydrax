#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax.applications.cosmology._full_dark_sector_inference import (
    FixedProfileEvaluation,
    FixedProfileSmoothSensitivityPlan,
    FullDarkSectorDifferentiationPolicy,
    FullPathProbabilityLaw,
    FullPathSampleBatch,
    FullPathScoreCRNPlan,
)


def _policy() -> FullDarkSectorDifferentiationPolicy:
    return FullDarkSectorDifferentiationPolicy(
        (("runtime", "profile-runtime"), ("radiation", "profile-radiation")),
        (("matrix-element", "revision-matrix"),),
        topology_id="topology-fixed",
        provider_ids=("provider-pinned",),
        external_artifact_ids=("artifact-pinned",),
        differentiable_parameters=("coupling",),
    )


def _evaluation(
    policy: FullDarkSectorDifferentiationPolicy, values: jnp.ndarray, /
) -> FixedProfileEvaluation:
    return FixedProfileEvaluation(
        values,
        jnp.asarray((3, 5), dtype=jnp.int32),
        finite=True,
        successful=True,
        smooth=True,
        topology_fixed=True,
        provider_fixed=True,
        external_artifacts_constant=True,
        fixed_profile_id=policy.fixed_profile_id,
        product_id="observable",
    )


def test_fixed_profile_sensitivity_is_audited_against_one_branch() -> None:
    policy = _policy()
    plan = FixedProfileSmoothSensitivityPlan(
        policy,
        jnp.asarray((3, 5), dtype=jnp.int32),
        output_count=2,
        product_id="observable",
    )
    epsilon = 1.0e-3
    center = _evaluation(policy, jnp.asarray((2.0, -1.0)))
    derivative = jnp.asarray((0.5, 4.0))
    lower = _evaluation(policy, center.values - epsilon * derivative)
    upper = _evaluation(policy, center.values + epsilon * derivative)

    result = plan.audit(center, lower, upper, derivative, epsilon=epsilon)

    assert bool(result.successful)
    assert jnp.allclose(result.jvp, derivative)
    assert jnp.allclose(result.finite_difference, derivative, rtol=2.0e-4)


def test_discrete_derivative_targets_are_refused() -> None:
    policy = _policy()

    for target in ("topology", "provider", "external-artifact"):
        with pytest.raises(ValueError, match="refuses"):
            policy.require_target(target)


def _law(
    policy: FullDarkSectorDifferentiationPolicy,
    complete: bool = True,
) -> FullPathProbabilityLaw:
    directional_score = jnp.asarray((1.0, -1.0, 1.0, -1.0))
    scores = jnp.stack(
        (
            0.25 * directional_score,
            0.75 * directional_score,
        ),
        axis=1,
    )[..., None]
    completeness = jnp.ones((4, 2), dtype=bool)
    if not complete:
        completeness = completeness.at[0, 1].set(False)
    return FullPathProbabilityLaw(
        jnp.zeros((4, 2)),
        scores,
        completeness,
        component_names=("event-law", "radiation-law"),
        fixed_profile_id=policy.fixed_profile_id,
    )


def _batch(
    law: FullPathProbabilityLaw,
    values: jnp.ndarray,
    /,
) -> FullPathSampleBatch:
    draw_ids = jnp.arange(32, dtype=jnp.uint32).reshape((4, 8))
    return FullPathSampleBatch(
        values[:, None],
        draw_ids,
        law,
        successful=True,
        product_id="observable",
    )


def test_full_path_score_and_common_random_difference_share_complete_law() -> None:
    policy = _policy()
    plan = FullPathScoreCRNPlan(
        policy,
        jnp.ones((4,)),
        parameter_count=1,
        output_count=1,
        required_law_components=("event-law", "radiation-law"),
        minimum_effective_sample_size=4.0,
        product_id="observable",
        bias_standard_error_multiplier=0.0,
        bias_absolute_tolerance=1.0e-3,
    )
    law = _law(policy)
    center_values = jnp.asarray((1.0, -1.0, 1.0, -1.0))
    epsilon = 1.0e-3
    center = _batch(law, center_values)
    lower = _batch(law, center_values - epsilon)
    upper = _batch(law, center_values + epsilon)

    result = plan.sensitivity(
        center,
        lower,
        upper,
        jnp.ones((1,)),
        epsilon=epsilon,
    )

    assert bool(result.complete_probability_law)
    assert bool(result.common_random_numbers)
    assert bool(result.successful)
    assert jnp.allclose(result.score_estimate, 1.0)
    assert jnp.allclose(result.common_random_finite_difference, 1.0, rtol=2.0e-4)


def test_full_path_product_refuses_incomplete_probability_law() -> None:
    policy = _policy()
    plan = FullPathScoreCRNPlan(
        policy,
        jnp.ones((4,)),
        parameter_count=1,
        output_count=1,
        required_law_components=("event-law", "radiation-law"),
        minimum_effective_sample_size=2.0,
        product_id="observable",
    )
    incomplete = _law(policy, complete=False)
    values = jnp.asarray((1.0, -1.0, 1.0, -1.0))
    batch = _batch(incomplete, values)

    with pytest.raises(Exception, match="complete probability law"):
        plan.sensitivity(batch, batch, batch, jnp.ones((1,)), epsilon=1.0e-3)
