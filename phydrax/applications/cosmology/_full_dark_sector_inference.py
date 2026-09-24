#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed differentiation products for the full dark-sector workflow.

Smooth derivatives are admitted only on one fixed profile, revision set, topology,
provider set, external-artifact set, and branch. Stochastic derivatives require an
explicit factorization of the complete path probability and an immutable
common-random-number draw identity. Discrete topology/provider/artifact choices are
never silently differentiated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._differentiation import DerivativeContract, DerivativeRoute
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import (
    DerivativeEstimatorKind,
    DerivativeEvidence,
    ScientificArtifactEnvelope,
)


FullDarkSectorDerivativeTarget: TypeAlias = Literal[
    "fixed-profile-parameters",
    "full-path-probability",
    "topology",
    "provider",
    "external-artifact",
]


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return normalized


def _named_identities(
    values: Sequence[tuple[str, str]], name: str, /
) -> tuple[tuple[str, str], ...]:
    records = tuple(
        (_identifier(key, f"{name} name"), _identifier(value, f"{name} ID"))
        for key, value in values
    )
    names = tuple(key for key, _ in records)
    if not records or len(set(names)) != len(names):
        raise ValueError(f"{name} must contain distinct named identities.")
    return records


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    records = tuple(_identifier(value, name) for value in values)
    if len(set(records)) != len(records):
        raise ValueError(f"{name} values must be distinct.")
    return records


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _scalar_flag(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value, dtype=jnp.bool_)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return jax.lax.stop_gradient(result)


def _real_vector(value: ArrayLike, size: int, name: str, /, *, dtype=None) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    if not eqx.is_inexact_array(result):
        result = result.astype("float64")
    return result


class FullDarkSectorDifferentiationPolicy(StrictModule, NonTrainableState):
    """Static derivative boundary for one fully resolved production profile."""

    profile_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    revision_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    provider_ids: tuple[str, ...] = eqx.field(static=True)
    external_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    differentiable_parameters: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile_ids: Sequence[tuple[str, str]],
        revision_ids: Sequence[tuple[str, str]],
        /,
        *,
        topology_id: str,
        provider_ids: Sequence[str],
        external_artifact_ids: Sequence[str],
        differentiable_parameters: Sequence[str],
    ):
        profiles = _named_identities(profile_ids, "profile_ids")
        revisions = _named_identities(revision_ids, "revision_ids")
        providers = _identifiers(provider_ids, "provider_id")
        artifacts = _identifiers(external_artifact_ids, "external_artifact_id")
        parameters = _identifiers(differentiable_parameters, "differentiable_parameter")
        if not parameters:
            raise ValueError("At least one smooth physical parameter is required.")
        self.profile_ids = profiles
        self.revision_ids = revisions
        self.topology_id = _identifier(topology_id, "topology_id")
        self.provider_ids = providers
        self.external_artifact_ids = artifacts
        self.differentiable_parameters = parameters
        self.policy_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-differentiation-policy",
                "profiles": [list(value) for value in profiles],
                "revisions": [list(value) for value in revisions],
                "topology": self.topology_id,
                "providers": list(providers),
                "external_artifacts": list(artifacts),
                "differentiable_parameters": list(parameters),
                "constants": ["topology", "provider", "external-artifact"],
            }
        )

    @property
    def fixed_profile_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "full-dark-sector-fixed-profile",
                "policy": self.policy_id,
            }
        )

    def require_target(self, target: FullDarkSectorDerivativeTarget, /) -> None:
        if target in ("topology", "provider", "external-artifact"):
            raise ValueError(
                f"Full dark-sector differentiation refuses the discrete {target} target."
            )
        if target not in ("fixed-profile-parameters", "full-path-probability"):
            raise ValueError("Unknown full dark-sector derivative target.")

    def constant_artifact_evidence(
        self, artifact: ScientificArtifactEnvelope, /
    ) -> DerivativeEvidence:
        if not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be ScientificArtifactEnvelope.")
        if artifact.artifact_id not in self.external_artifact_ids:
            raise ValueError("External artifact is not bound by this fixed profile.")
        return DerivativeEvidence(
            DerivativeContract(route=DerivativeRoute.DIRECT),
            estimator=DerivativeEstimatorKind.UNSUPPORTED,
            discrete_parameters=("external-artifact",),
            stopped_events=("external-artifact-selection",),
            support_id=self.fixed_profile_id,
            evidence_ids=(artifact.artifact_id,),
        )


class FixedProfileEvaluation(StrictModule):
    """Values and admission evidence at one fixed-profile parameter point."""

    values: Array
    branch_signature: Array
    finite: Array
    successful: Array
    smooth: Array
    topology_fixed: Array
    provider_fixed: Array
    external_artifacts_constant: Array
    fixed_profile_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        branch_signature: ArrayLike,
        /,
        *,
        finite: ArrayLike,
        successful: ArrayLike,
        smooth: ArrayLike,
        topology_fixed: ArrayLike,
        provider_fixed: ArrayLike,
        external_artifacts_constant: ArrayLike,
        fixed_profile_id: str,
        product_id: str,
    ):
        value = jnp.asarray(values).reshape((-1,))
        if value.size == 0 or not eqx.is_inexact_array(value):
            raise TypeError("Fixed-profile values must be a nonempty inexact vector.")
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Fixed-profile values must be real.")
        signature = jax.lax.stop_gradient(
            jnp.asarray(branch_signature, dtype=jnp.int32).reshape((-1,))
        )
        if signature.size == 0:
            raise ValueError("branch_signature must be nonempty.")
        self.values = value
        self.branch_signature = signature
        self.finite = _scalar_flag(finite, "finite")
        self.successful = _scalar_flag(successful, "successful")
        self.smooth = _scalar_flag(smooth, "smooth")
        self.topology_fixed = _scalar_flag(topology_fixed, "topology_fixed")
        self.provider_fixed = _scalar_flag(provider_fixed, "provider_fixed")
        self.external_artifacts_constant = _scalar_flag(
            external_artifacts_constant, "external_artifacts_constant"
        )
        self.fixed_profile_id = _identifier(fixed_profile_id, "fixed_profile_id")
        self.product_id = _identifier(product_id, "product_id")
        self.evaluation_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-fixed-profile-evaluation",
                "profile": self.fixed_profile_id,
                "product": self.product_id,
                "branch": array_tree_fingerprint(signature),
            }
        )

    @property
    def sensitivity_eligible(self) -> Array:
        return (
            self.finite
            & self.successful
            & self.smooth
            & self.topology_fixed
            & self.provider_fixed
            & self.external_artifacts_constant
            & jnp.all(jnp.isfinite(self.values))
        )


class FixedProfileSensitivityProduct(StrictModule):
    value: Array
    jvp: Array
    finite_difference: Array
    absolute_residual: Array
    relative_residual: Array
    center_eligible: Array
    stencil_eligible: Array
    branch_stable: Array
    successful: Array
    epsilon: Array
    policy_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class FixedProfileSmoothSensitivityPlan(StrictModule, NonTrainableState):
    """Audit an owner-computed JVP on one immutable full-closure profile."""

    policy: FullDarkSectorDifferentiationPolicy
    expected_branch_signature: Array
    output_count: int = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: FullDarkSectorDifferentiationPolicy,
        expected_branch_signature: ArrayLike,
        /,
        *,
        output_count: int,
        product_id: str,
        absolute_tolerance: float = 1.0e-7,
        relative_tolerance: float = 1.0e-4,
    ):
        if not isinstance(policy, FullDarkSectorDifferentiationPolicy):
            raise TypeError("policy must be FullDarkSectorDifferentiationPolicy.")
        policy.require_target("fixed-profile-parameters")
        signature = jax.lax.stop_gradient(
            jnp.asarray(expected_branch_signature, dtype=jnp.int32).reshape((-1,))
        )
        if signature.size == 0:
            raise ValueError("expected_branch_signature must be nonempty.")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not np.isfinite(absolute)
            or absolute < 0.0
            or not np.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("Sensitivity tolerances must be finite and nonnegative.")
        self.policy = policy
        self.expected_branch_signature = signature
        self.output_count = _positive_integer(output_count, "output_count")
        self.product_id = _identifier(product_id, "product_id")
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-fixed-profile-sensitivity-plan",
                "policy": policy.policy_id,
                "branch": array_tree_fingerprint(signature),
                "outputs": self.output_count,
                "product": self.product_id,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def _validate(self, evaluation: FixedProfileEvaluation, name: str, /) -> None:
        if not isinstance(evaluation, FixedProfileEvaluation):
            raise TypeError(f"{name} must be FixedProfileEvaluation.")
        if (
            evaluation.values.shape != (self.output_count,)
            or evaluation.branch_signature.shape != self.expected_branch_signature.shape
            or evaluation.fixed_profile_id != self.policy.fixed_profile_id
            or evaluation.product_id != self.product_id
        ):
            raise ValueError(f"{name} changed the fixed-profile product contract.")

    def audit(
        self,
        center: FixedProfileEvaluation,
        lower: FixedProfileEvaluation,
        upper: FixedProfileEvaluation,
        jvp: ArrayLike,
        /,
        *,
        epsilon: float,
    ) -> FixedProfileSensitivityProduct:
        self._validate(center, "center")
        self._validate(lower, "lower")
        self._validate(upper, "upper")
        raw_jvp = _real_vector(jvp, self.output_count, "jvp", dtype=center.values.dtype)
        step = float(epsilon)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        finite_difference = (upper.values - lower.values) / (2.0 * step)
        branch_stable = (
            jnp.all(center.branch_signature == self.expected_branch_signature)
            & jnp.all(lower.branch_signature == center.branch_signature)
            & jnp.all(upper.branch_signature == center.branch_signature)
        )
        stencil_eligible = lower.sensitivity_eligible & upper.sensitivity_eligible
        finite = jnp.all(jnp.isfinite(raw_jvp)) & jnp.all(jnp.isfinite(finite_difference))
        eligible = center.sensitivity_eligible & stencil_eligible & branch_stable & finite
        admitted_jvp = eqx.error_if(
            raw_jvp,
            ~eligible,
            "Full dark-sector smooth sensitivity crossed a profile, revision, "
            "topology, provider, artifact, or branch boundary.",
        )
        residual = jnp.abs(admitted_jvp - finite_difference)
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(admitted_jvp), jnp.abs(finite_difference)), 1.0
        )
        relative = residual / scale
        successful = eligible & jnp.all(
            residual <= self.absolute_tolerance + self.relative_tolerance * scale
        )
        return FixedProfileSensitivityProduct(
            center.values,
            admitted_jvp,
            finite_difference,
            residual,
            relative,
            center.sensitivity_eligible,
            stencil_eligible,
            branch_stable,
            successful,
            jnp.asarray(step, dtype=center.values.dtype),
            self.policy.policy_id,
            self.product_id,
            self.plan_id,
        )


class FullPathProbabilityLaw(StrictModule):
    """Complete factorized log law and parameter score for every sampled path."""

    component_log_probabilities: Array
    component_scores: Array
    component_complete: Array
    finite: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    fixed_profile_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_log_probabilities: ArrayLike,
        component_scores: ArrayLike,
        component_complete: ArrayLike,
        /,
        *,
        component_names: Sequence[str],
        fixed_profile_id: str,
    ):
        log_probability = jnp.asarray(component_log_probabilities)
        if (
            log_probability.ndim != 2
            or log_probability.shape[0] < 2
            or log_probability.shape[1] == 0
            or jnp.issubdtype(log_probability.dtype, jnp.complexfloating)
        ):
            raise ValueError(
                "component_log_probabilities must have shape (draw, component)."
            )
        if not eqx.is_inexact_array(log_probability):
            log_probability = log_probability.astype("float64")
        scores = jnp.asarray(component_scores, dtype=log_probability.dtype)
        complete = jax.lax.stop_gradient(jnp.asarray(component_complete, dtype=jnp.bool_))
        if scores.ndim != 3 or scores.shape[:2] != log_probability.shape:
            raise ValueError(
                "component_scores must have shape (draw, component, parameter)."
            )
        if scores.shape[-1] == 0 or complete.shape != log_probability.shape:
            raise ValueError("Full-path score or completeness shape is invalid.")
        names = _identifiers(component_names, "probability component")
        if len(names) != log_probability.shape[1]:
            raise ValueError("Probability component names do not match the law.")
        self.component_log_probabilities = log_probability
        self.component_scores = scores
        self.component_complete = complete
        self.finite = jnp.all(jnp.isfinite(log_probability)) & jnp.all(
            jnp.isfinite(scores)
        )
        self.component_names = names
        self.fixed_profile_id = _identifier(fixed_profile_id, "fixed_profile_id")
        self.law_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-full-path-probability-law",
                "components": list(names),
                "profile": self.fixed_profile_id,
                "shape": list(log_probability.shape),
                "parameters": scores.shape[-1],
            }
        )

    @property
    def draw_count(self) -> int:
        return self.component_log_probabilities.shape[0]

    @property
    def parameter_count(self) -> int:
        return self.component_scores.shape[-1]

    @property
    def log_probability(self) -> Array:
        return jnp.sum(self.component_log_probabilities, axis=1)

    @property
    def score(self) -> Array:
        return jnp.sum(self.component_scores, axis=1)

    @property
    def complete(self) -> Array:
        return self.finite & jnp.all(self.component_complete)


class FullPathSampleBatch(StrictModule):
    values: Array
    random_draw_ids: Array
    law: FullPathProbabilityLaw
    finite: Array
    successful: Array
    product_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        random_draw_ids: ArrayLike,
        law: FullPathProbabilityLaw,
        /,
        *,
        successful: ArrayLike,
        product_id: str,
    ):
        if not isinstance(law, FullPathProbabilityLaw):
            raise TypeError("law must be FullPathProbabilityLaw.")
        value = jnp.asarray(values)
        if (
            value.ndim != 2
            or value.shape[0] != law.draw_count
            or value.shape[1] == 0
            or jnp.issubdtype(value.dtype, jnp.complexfloating)
        ):
            raise ValueError("values must have shape (draw, output) and be real.")
        if not eqx.is_inexact_array(value):
            value = value.astype("float64")
        draw_ids = jax.lax.stop_gradient(jnp.asarray(random_draw_ids, dtype=jnp.uint32))
        if draw_ids.shape != (law.draw_count, 8):
            raise ValueError("random_draw_ids must have shape (draw, 8).")
        self.values = value
        self.random_draw_ids = draw_ids
        self.law = law
        self.finite = law.finite & jnp.all(jnp.isfinite(value))
        self.successful = _scalar_flag(successful, "successful")
        self.product_id = _identifier(product_id, "product_id")
        self.batch_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-full-path-sample-batch",
                "law": law.law_id,
                "profile": law.fixed_profile_id,
                "product": self.product_id,
                "draw_ids": array_tree_fingerprint(draw_ids),
                "shape": list(value.shape),
            }
        )


class FullPathScoreCRNProduct(StrictModule):
    value: Array
    score_estimate: Array
    score_standard_error: Array
    common_random_finite_difference: Array
    finite_difference_standard_error: Array
    paired_difference: Array
    paired_standard_error: Array
    bias_z_score: Array
    normalized_weights: Array
    effective_sample_size: Array
    complete_probability_law: Array
    common_random_numbers: Array
    fixed_profile: Array
    finite: Array
    successful: Array
    epsilon: Array
    policy_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _weighted_mean_standard_error(
    samples: Array, normalized_weights: Array, /
) -> tuple[Array, Array]:
    mean = ein.contract("n,no->o", normalized_weights, samples)
    centered = samples - mean
    squared_weight = jnp.sum(normalized_weights * normalized_weights)
    correction = jnp.maximum(
        1.0 - squared_weight, jnp.finfo(normalized_weights.dtype).eps
    )
    variance = (
        ein.contract("n,no,no->o", normalized_weights, centered, centered) / correction
    )
    return mean, jnp.sqrt(jnp.maximum(variance * squared_weight, 0.0))


class FullPathScoreCRNPlan(StrictModule, NonTrainableState):
    """Score/CRN audit admitted only for a complete, fixed-profile path law."""

    policy: FullDarkSectorDifferentiationPolicy
    draw_weights: Array
    parameter_count: int = eqx.field(static=True)
    output_count: int = eqx.field(static=True)
    draw_count: int = eqx.field(static=True)
    required_law_components: tuple[str, ...] = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    bias_absolute_tolerance: float = eqx.field(static=True)
    bias_standard_error_multiplier: float = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: FullDarkSectorDifferentiationPolicy,
        draw_weights: ArrayLike,
        /,
        *,
        parameter_count: int,
        output_count: int,
        required_law_components: Sequence[str],
        minimum_effective_sample_size: float,
        product_id: str,
        bias_absolute_tolerance: float = 0.0,
        bias_standard_error_multiplier: float = 2.0,
    ):
        if not isinstance(policy, FullDarkSectorDifferentiationPolicy):
            raise TypeError("policy must be FullDarkSectorDifferentiationPolicy.")
        policy.require_target("full-path-probability")
        parameters = _positive_integer(parameter_count, "parameter_count")
        if parameters != len(policy.differentiable_parameters):
            raise ValueError("parameter_count must cover every differentiable parameter.")
        outputs = _positive_integer(output_count, "output_count")
        weights_host = np.asarray(draw_weights, dtype=np.float64).reshape((-1,))
        if (
            weights_host.size < 2
            or np.any(~np.isfinite(weights_host))
            or np.any(weights_host < 0.0)
            or not np.any(weights_host > 0.0)
        ):
            raise ValueError("draw_weights must contain finite nonnegative mass.")
        components = _identifiers(required_law_components, "required law component")
        if not components:
            raise ValueError("A complete path law requires named probability components.")
        minimum = float(minimum_effective_sample_size)
        absolute = float(bias_absolute_tolerance)
        multiplier = float(bias_standard_error_multiplier)
        if (
            not np.isfinite(minimum)
            or not 1.0 <= minimum <= weights_host.size
            or not np.isfinite(absolute)
            or absolute < 0.0
            or not np.isfinite(multiplier)
            or multiplier < 0.0
        ):
            raise ValueError("Score/CRN evidence thresholds are invalid.")
        self.policy = policy
        self.draw_weights = jax.lax.stop_gradient(jnp.asarray(weights_host))
        self.parameter_count = parameters
        self.output_count = outputs
        self.draw_count = weights_host.size
        self.required_law_components = components
        self.minimum_effective_sample_size = minimum
        self.bias_absolute_tolerance = absolute
        self.bias_standard_error_multiplier = multiplier
        self.product_id = _identifier(product_id, "product_id")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-score-crn-plan",
                "policy": policy.policy_id,
                "draw_weights": weights_host.tolist(),
                "parameters": parameters,
                "outputs": outputs,
                "required_law_components": list(components),
                "minimum_effective_sample_size": minimum,
                "bias_absolute_tolerance": absolute,
                "bias_standard_error_multiplier": multiplier,
                "product": self.product_id,
            }
        )

    def _validate(self, batch: FullPathSampleBatch, name: str, /) -> None:
        if not isinstance(batch, FullPathSampleBatch):
            raise TypeError(f"{name} must be FullPathSampleBatch.")
        if (
            batch.values.shape != (self.draw_count, self.output_count)
            or batch.law.parameter_count != self.parameter_count
            or batch.law.component_names != self.required_law_components
            or batch.law.fixed_profile_id != self.policy.fixed_profile_id
            or batch.product_id != self.product_id
        ):
            raise ValueError(f"{name} changed the complete path-law contract.")

    def sensitivity(
        self,
        center: FullPathSampleBatch,
        lower: FullPathSampleBatch,
        upper: FullPathSampleBatch,
        direction: ArrayLike,
        /,
        *,
        epsilon: float,
    ) -> FullPathScoreCRNProduct:
        self._validate(center, "center")
        self._validate(lower, "lower")
        self._validate(upper, "upper")
        tangent = _real_vector(direction, self.parameter_count, "direction")
        step = float(epsilon)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        common_random = jnp.all(
            center.random_draw_ids == lower.random_draw_ids
        ) & jnp.all(center.random_draw_ids == upper.random_draw_ids)
        fixed_profile = jnp.asarray(
            center.law.fixed_profile_id
            == lower.law.fixed_profile_id
            == upper.law.fixed_profile_id
            == self.policy.fixed_profile_id
        )
        complete_law = center.law.complete & lower.law.complete & upper.law.complete
        center_values = eqx.error_if(
            center.values,
            ~complete_law | ~common_random | ~fixed_profile,
            "Full-path score/CRN sensitivity requires a complete probability law, "
            "one immutable random tape, and one fixed profile.",
        )
        weights = self.draw_weights / jnp.sum(self.draw_weights)
        effective = 1.0 / jnp.sum(weights * weights)
        value, _ = _weighted_mean_standard_error(center_values, weights)
        directional_score = ein.contract("np,p->n", center.law.score, tangent)
        score_samples = center_values * directional_score[:, None]
        score, score_error = _weighted_mean_standard_error(score_samples, weights)
        finite_difference_samples = (upper.values - lower.values) / (2.0 * step)
        finite_difference, finite_difference_error = _weighted_mean_standard_error(
            finite_difference_samples, weights
        )
        paired_samples = score_samples - finite_difference_samples
        paired, paired_error = _weighted_mean_standard_error(paired_samples, weights)
        safe_error = jnp.maximum(paired_error, jnp.finfo(paired_error.dtype).tiny)
        bias_z_score = jnp.abs(paired) / safe_error
        finite = (
            center.finite
            & lower.finite
            & upper.finite
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(score))
            & jnp.all(jnp.isfinite(score_error))
            & jnp.all(jnp.isfinite(finite_difference))
            & jnp.all(jnp.isfinite(finite_difference_error))
            & jnp.all(jnp.isfinite(paired))
            & jnp.all(jnp.isfinite(paired_error))
            & jnp.all(jnp.isfinite(bias_z_score))
            & jnp.isfinite(effective)
        )
        threshold = (
            self.bias_absolute_tolerance
            + self.bias_standard_error_multiplier * paired_error
        )
        successful = (
            finite
            & center.successful
            & lower.successful
            & upper.successful
            & complete_law
            & common_random
            & fixed_profile
            & (effective >= self.minimum_effective_sample_size)
            & jnp.all(jnp.abs(paired) <= threshold)
        )
        return FullPathScoreCRNProduct(
            value,
            score,
            score_error,
            finite_difference,
            finite_difference_error,
            paired,
            paired_error,
            bias_z_score,
            weights,
            effective,
            complete_law,
            common_random,
            fixed_profile,
            finite,
            successful,
            jnp.asarray(step, dtype=center.values.dtype),
            self.policy.policy_id,
            self.product_id,
            self.plan_id,
        )


__all__ = [
    "FixedProfileEvaluation",
    "FixedProfileSensitivityProduct",
    "FixedProfileSmoothSensitivityPlan",
    "FullDarkSectorDerivativeTarget",
    "FullDarkSectorDifferentiationPolicy",
    "FullPathProbabilityLaw",
    "FullPathSampleBatch",
    "FullPathScoreCRNPlan",
    "FullPathScoreCRNProduct",
]
