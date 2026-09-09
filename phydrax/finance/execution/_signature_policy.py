#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit linear causal policies over canonical path-signature features."""

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...stochastic._signature_features import (
    LogSignatureFeatures,
    SignatureFeatures,
    time_augment_path,
)


SignatureFeatureKind: TypeAlias = Literal["signature", "logsignature"]
PolicySampleRole: TypeAlias = Literal["training", "holdout"]


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _identifiers(values: tuple[str, ...], owner: str, /) -> tuple[str, ...]:
    resolved = tuple(values)
    if any(not isinstance(value, str) or not value for value in resolved):
        raise ValueError(f"{owner} must contain non-empty strings.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{owner} must contain unique identities.")
    return resolved


class CausalSignaturePolicySpec(StrictModule):
    """Static history/action dimensions and truncated signature convention."""

    history_dimension: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    feature_kind: SignatureFeatureKind = eqx.field(static=True)
    include_scalar: bool = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        history_dimension: int,
        action_size: int,
        depth: int,
        feature_kind: SignatureFeatureKind = "signature",
        include_scalar: bool = True,
        spec_id: str,
    ):
        if history_dimension <= 0 or action_size <= 0 or depth <= 0:
            raise ValueError(
                "history_dimension, action_size, and depth must be positive."
            )
        if feature_kind not in ("signature", "logsignature"):
            raise ValueError("feature_kind must be 'signature' or 'logsignature'.")
        if feature_kind == "logsignature" and include_scalar:
            raise ValueError("logsignature features do not include a scalar level.")
        self.history_dimension = int(history_dimension)
        self.action_size = int(action_size)
        self.depth = int(depth)
        self.feature_kind = feature_kind
        self.include_scalar = bool(include_scalar)
        self.spec_id = _identifier(spec_id, "spec_id")


class PreparedCausalSignaturePolicy(StrictModule):
    """Prepared time-augmented signature feature map with fixed output size."""

    spec: CausalSignaturePolicySpec
    signature_features: SignatureFeatures | None
    logsignature_features: LogSignatureFeatures | None
    augmented_dimension: int = eqx.field(static=True)
    feature_size: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def prepare_causal_signature_policy(
    spec: CausalSignaturePolicySpec, /
) -> PreparedCausalSignaturePolicy:
    """Prepare exactly one declared signature feature convention."""

    if not isinstance(spec, CausalSignaturePolicySpec):
        raise TypeError("spec must be a CausalSignaturePolicySpec.")
    dimension = spec.history_dimension + 1
    if spec.feature_kind == "signature":
        signature = SignatureFeatures(
            dimension,
            spec.depth,
            include_scalar=spec.include_scalar,
            stream=False,
        )
        logsignature = None
        feature_size = signature.output_size
    else:
        signature = None
        logsignature = LogSignatureFeatures(dimension, spec.depth, stream=False)
        feature_size = logsignature.output_size
    return PreparedCausalSignaturePolicy(
        spec=spec,
        signature_features=signature,
        logsignature_features=logsignature,
        augmented_dimension=dimension,
        feature_size=feature_size,
        prepared_id=(
            f"prepared-causal-{spec.feature_kind}:{spec.spec_id}:"
            f"dimension={dimension}:depth={spec.depth}"
        ),
    )


class SignaturePolicySampleSet(StrictModule):
    """Fixed-shape path prefixes with explicit training or holdout provenance."""

    times: Array
    histories: Array
    lengths: Array
    independence_labels: Array
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    sample_role: PolicySampleRole = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    num_paths: int = eqx.field(static=True)
    max_knots: int = eqx.field(static=True)
    history_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        histories: ArrayLike,
        lengths: ArrayLike,
        /,
        *,
        realization_ids: tuple[str, ...],
        independence_labels: ArrayLike,
        sample_role: PolicySampleRole,
        dataset_id: str,
    ):
        history = jnp.asarray(histories)
        if history.ndim != 3 or 0 in history.shape:
            raise ValueError(
                "histories must have shape (num_paths, max_knots, dimension)."
            )
        if jnp.issubdtype(history.dtype, jnp.complexfloating):
            raise TypeError("histories must be real-valued.")
        history = history.astype(jnp.result_type(history, float))
        num_paths, max_knots, dimension = map(int, history.shape)
        time_values = jnp.asarray(times)
        if time_values.shape == (max_knots,):
            time_values = jnp.broadcast_to(time_values, (num_paths, max_knots))
        if time_values.shape != (num_paths, max_knots):
            raise ValueError(
                f"times must have shape ({max_knots},) or ({num_paths}, {max_knots})."
            )
        if jnp.issubdtype(time_values.dtype, jnp.complexfloating):
            raise TypeError("times must be real-valued.")
        time_values = time_values.astype(jnp.result_type(time_values, float))
        length_values = jnp.asarray(lengths)
        if length_values.shape != (num_paths,) or not jnp.issubdtype(
            length_values.dtype, jnp.integer
        ):
            raise TypeError("lengths must be an integer vector with one entry per path.")
        if bool(jnp.any((length_values < 2) | (length_values > max_knots))):
            raise ValueError("Each path length must lie between two and max_knots.")
        for path in range(num_paths):
            length = int(length_values[path])
            if not bool(jnp.all(jnp.isfinite(history[path, :length]))):
                raise ValueError("Every valid history prefix must be finite.")
            if not bool(jnp.all(jnp.isfinite(time_values[path, :length]))):
                raise ValueError("Every valid time prefix must be finite.")
            if bool(jnp.any(jnp.diff(time_values[path, :length]) <= 0.0)):
                raise ValueError("Every valid time prefix must be strictly increasing.")
        labels = jnp.asarray(independence_labels)
        if labels.shape != (num_paths,) or not jnp.issubdtype(labels.dtype, jnp.integer):
            raise TypeError(
                "independence_labels must be an integer vector with one entry per path."
            )
        if bool(jnp.any(labels < 0)):
            raise ValueError("independence_labels must be nonnegative.")
        if sample_role not in ("training", "holdout"):
            raise ValueError("sample_role must be 'training' or 'holdout'.")
        identities = _identifiers(tuple(realization_ids), "realization_ids")
        if len(identities) != num_paths:
            raise ValueError("realization_ids must contain one identity per path.")
        self.times = time_values
        self.histories = history
        self.lengths = length_values.astype(jnp.int32)
        self.independence_labels = labels.astype(jnp.int32)
        self.realization_ids = identities
        self.sample_role = sample_role
        self.dataset_id = _identifier(dataset_id, "dataset_id")
        self.num_paths = num_paths
        self.max_knots = max_knots
        self.history_dimension = dimension


class CausalSignaturePolicy(StrictModule):
    """An explicit bounded-domain linear map of canonical causal features."""

    prepared: PreparedCausalSignaturePolicy
    weights: Array
    bias: Array
    action_lower_bounds: Array
    action_upper_bounds: Array
    training_independence_labels: tuple[int, ...] = eqx.field(static=True)
    training_realization_ids: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedCausalSignaturePolicy,
        weights: ArrayLike,
        bias: ArrayLike,
        action_lower_bounds: ArrayLike,
        action_upper_bounds: ArrayLike,
        /,
        *,
        training_sample: SignaturePolicySampleSet,
        policy_id: str,
    ):
        if not isinstance(prepared, PreparedCausalSignaturePolicy):
            raise TypeError("prepared must be a PreparedCausalSignaturePolicy.")
        if not isinstance(training_sample, SignaturePolicySampleSet):
            raise TypeError("training_sample must be a SignaturePolicySampleSet.")
        if training_sample.sample_role != "training":
            raise ValueError("training_sample must declare sample_role='training'.")
        if training_sample.history_dimension != prepared.spec.history_dimension:
            raise ValueError("training history dimension does not match the policy spec.")
        expected_weights = (prepared.spec.action_size, prepared.feature_size)
        matrix = jnp.asarray(weights)
        offset = jnp.asarray(bias)
        lower = jnp.asarray(action_lower_bounds)
        upper = jnp.asarray(action_upper_bounds)
        if matrix.shape != expected_weights:
            raise ValueError(f"weights must have shape {expected_weights}.")
        expected_action = (prepared.spec.action_size,)
        if (
            offset.shape != expected_action
            or lower.shape != expected_action
            or upper.shape != expected_action
        ):
            raise ValueError(f"bias and action bounds must have shape {expected_action}.")
        for owner, value in (
            ("weights", matrix),
            ("bias", offset),
            ("action_lower_bounds", lower),
            ("action_upper_bounds", upper),
        ):
            if jnp.issubdtype(value.dtype, jnp.complexfloating):
                raise TypeError(f"{owner} must be real-valued.")
            if not bool(jnp.all(jnp.isfinite(value))):
                raise ValueError(f"{owner} must be finite.")
        if bool(jnp.any(lower > upper)):
            raise ValueError("action lower bounds cannot exceed upper bounds.")
        self.prepared = prepared
        self.weights = matrix.astype(jnp.result_type(matrix, float))
        self.bias = offset.astype(jnp.result_type(offset, float))
        self.action_lower_bounds = lower.astype(jnp.result_type(lower, float))
        self.action_upper_bounds = upper.astype(jnp.result_type(upper, float))
        self.training_independence_labels = tuple(
            int(value) for value in np.asarray(training_sample.independence_labels)
        )
        self.training_realization_ids = training_sample.realization_ids
        self.policy_id = _identifier(policy_id, "policy_id")

    def features(
        self,
        times: ArrayLike,
        history: ArrayLike,
        length: ArrayLike,
        /,
    ) -> Array:
        """Canonicalize the padded suffix before computing time-augmented features."""

        values = jnp.asarray(history)
        if values.ndim != 2 or values.shape[1] != self.prepared.spec.history_dimension:
            raise ValueError("history must have shape (max_knots, history_dimension).")
        length_value = jnp.asarray(length)
        if length_value.shape != () or not jnp.issubdtype(
            length_value.dtype, jnp.integer
        ):
            raise TypeError("length must be an integer scalar.")
        augmented = time_augment_path(times, values, lengths=length_value)
        if self.prepared.signature_features is not None:
            return self.prepared.signature_features(augmented)
        logsignature = self.prepared.logsignature_features
        if logsignature is None:
            raise ValueError("Prepared policy has no configured feature map.")
        return logsignature(augmented)

    def action(
        self,
        times: ArrayLike,
        history: ArrayLike,
        length: ArrayLike,
        /,
    ) -> Array:
        """Evaluate the unprojected action from the declared valid prefix."""

        return self.weights @ self.features(times, history, length) + self.bias

    def action_is_admissible(self, action: ArrayLike, /) -> Array:
        value = jnp.asarray(action)
        if value.shape != (self.prepared.spec.action_size,):
            raise ValueError(
                f"action must have shape ({self.prepared.spec.action_size},)."
            )
        return (
            jnp.all(jnp.isfinite(value))
            & jnp.all(value >= self.action_lower_bounds)
            & jnp.all(value <= self.action_upper_bounds)
        )


class SignaturePolicyEvaluation(StrictModule):
    """Actions and validity on a provenance-preserving training or holdout sample."""

    actions: Array
    valid: Array
    sample: SignaturePolicySampleSet
    policy_id: str = eqx.field(static=True)
    sample_role: PolicySampleRole = eqx.field(static=True)
    independent_of_training: bool = eqx.field(static=True)


class SignatureCausalityEvidence(StrictModule):
    """Observed future-suffix invariance for paired equal-prefix histories."""

    reference_action: Array
    perturbed_future_action: Array
    maximum_future_sensitivity: Array
    tolerance: Array
    passed: Array
    policy_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def evaluate_causal_signature_policy(
    policy: CausalSignaturePolicy,
    sample: SignaturePolicySampleSet,
    /,
) -> SignaturePolicyEvaluation:
    """Evaluate a policy, enforcing disjoint provenance for holdout samples."""

    if not isinstance(policy, CausalSignaturePolicy):
        raise TypeError("policy must be a CausalSignaturePolicy.")
    if not isinstance(sample, SignaturePolicySampleSet):
        raise TypeError("sample must be a SignaturePolicySampleSet.")
    if sample.history_dimension != policy.prepared.spec.history_dimension:
        raise ValueError("sample history dimension does not match the policy spec.")
    independent = True
    if sample.sample_role == "holdout":
        if set(sample.realization_ids) & set(policy.training_realization_ids):
            raise ValueError(
                "Holdout realization IDs must be disjoint from training IDs."
            )
        holdout_labels = set(
            int(value) for value in np.asarray(sample.independence_labels)
        )
        if holdout_labels & set(policy.training_independence_labels):
            raise ValueError(
                "Holdout independence labels must be disjoint from training labels."
            )
    else:
        independent = False
        if sample.realization_ids != policy.training_realization_ids:
            raise ValueError(
                "A training evaluation must use the policy's declared training paths."
            )
        training_labels = tuple(
            int(value) for value in np.asarray(sample.independence_labels)
        )
        if training_labels != policy.training_independence_labels:
            raise ValueError(
                "A training evaluation must preserve the policy's training "
                "independence labels."
            )
    actions = jnp.stack(
        tuple(
            policy.action(
                sample.times[index], sample.histories[index], sample.lengths[index]
            )
            for index in range(sample.num_paths)
        )
    )
    valid = jnp.stack(tuple(policy.action_is_admissible(action) for action in actions))
    return SignaturePolicyEvaluation(
        actions=actions,
        valid=valid,
        sample=sample,
        policy_id=policy.policy_id,
        sample_role=sample.sample_role,
        independent_of_training=independent,
    )


def evaluate_signature_causality(
    policy: CausalSignaturePolicy,
    times: ArrayLike,
    reference_history: ArrayLike,
    perturbed_future_history: ArrayLike,
    length: int,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> SignatureCausalityEvidence:
    """Check that changing only a padded future suffix cannot change the action."""

    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    reference = jnp.asarray(reference_history)
    perturbed = jnp.asarray(perturbed_future_history)
    if reference.shape != perturbed.shape:
        raise ValueError("Paired histories must have equal shape.")
    if length < 2 or length > reference.shape[0]:
        raise ValueError("length must lie between two and the number of knots.")
    if not bool(jnp.array_equal(reference[:length], perturbed[:length])):
        raise ValueError("Paired histories must have identical declared prefixes.")
    reference_action = policy.action(times, reference, jnp.asarray(length))
    perturbed_action = policy.action(times, perturbed, jnp.asarray(length))
    sensitivity = jnp.max(jnp.abs(reference_action - perturbed_action))
    return SignatureCausalityEvidence(
        reference_action=reference_action,
        perturbed_future_action=perturbed_action,
        maximum_future_sensitivity=sensitivity,
        tolerance=jnp.asarray(threshold),
        passed=sensitivity <= threshold,
        policy_id=policy.policy_id,
        scope="one-canonical-equal-prefix-paired-suffix-perturbation",
    )


__all__ = [
    "CausalSignaturePolicy",
    "CausalSignaturePolicySpec",
    "PolicySampleRole",
    "PreparedCausalSignaturePolicy",
    "SignatureCausalityEvidence",
    "SignatureFeatureKind",
    "SignaturePolicyEvaluation",
    "SignaturePolicySampleSet",
    "evaluate_causal_signature_policy",
    "evaluate_signature_causality",
    "prepare_causal_signature_policy",
]
