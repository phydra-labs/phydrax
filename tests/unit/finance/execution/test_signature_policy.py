#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.finance.execution._signature_policy import (
    CausalSignaturePolicy,
    CausalSignaturePolicySpec,
    evaluate_causal_signature_policy,
    evaluate_signature_causality,
    prepare_causal_signature_policy,
    SignaturePolicySampleSet,
)


def _training_sample():
    return SignaturePolicySampleSet(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([[[0.0], [1.0], [99.0]]]),
        jnp.asarray([2]),
        realization_ids=("train-path",),
        independence_labels=jnp.asarray([0]),
        sample_role="training",
        dataset_id="train",
    )


def _policy():
    spec = CausalSignaturePolicySpec(
        history_dimension=1,
        action_size=1,
        depth=1,
        feature_kind="signature",
        include_scalar=True,
        spec_id="causal-linear",
    )
    prepared = prepare_causal_signature_policy(spec)
    return CausalSignaturePolicy(
        prepared,
        jnp.asarray([[0.0, 0.0, 1.0]]),
        jnp.asarray([0.0]),
        jnp.asarray([-10.0]),
        jnp.asarray([10.0]),
        training_sample=_training_sample(),
        policy_id="signature-policy",
    )


def test_signature_action_is_invariant_to_future_padding_but_uses_past_marks():
    policy = _policy()
    times = jnp.asarray([0.0, 1.0, 2.0])
    reference = jnp.asarray([[0.0], [1.0], [100.0]])
    perturbed_future = jnp.asarray([[0.0], [1.0], [-500.0]])
    evidence = evaluate_signature_causality(policy, times, reference, perturbed_future, 2)

    assert bool(evidence.passed)
    assert evidence.maximum_future_sensitivity == 0.0
    changed_past = jnp.asarray([[0.0], [2.0], [-500.0]])
    assert policy.action(times, changed_past, jnp.asarray(2))[0] == 2.0
    assert policy.action(times, reference, jnp.asarray(2))[0] == 1.0


def test_holdout_paths_and_independence_labels_must_be_disjoint_from_training():
    policy = _policy()
    holdout = SignaturePolicySampleSet(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([[[0.0], [0.5], [7.0]]]),
        jnp.asarray([2]),
        realization_ids=("holdout-path",),
        independence_labels=jnp.asarray([1]),
        sample_role="holdout",
        dataset_id="holdout",
    )
    evaluation = evaluate_causal_signature_policy(policy, holdout)
    assert evaluation.independent_of_training
    assert bool(evaluation.valid)

    overlapping = SignaturePolicySampleSet(
        holdout.times,
        holdout.histories,
        holdout.lengths,
        realization_ids=("holdout-path-2",),
        independence_labels=jnp.asarray([0]),
        sample_role="holdout",
        dataset_id="overlap",
    )
    with pytest.raises(ValueError, match="independence labels"):
        evaluate_causal_signature_policy(policy, overlapping)
