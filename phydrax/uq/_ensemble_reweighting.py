#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Finite-support Bayesian/maximum-entropy ensemble refinement.

The reference weights in this module retain their declared provenance. Only a
``physical-equilibrium`` support can contribute physical-equilibrium evidence;
weights on empirical or proposal-only collections remain conditional finite-
support summaries. In particular, proposal-only weights are not interpreted as
proposal densities and cannot manufacture an equilibrium measure.

For a normalized reference distribution :math:`p`, per-sample observables
:math:`a_i`, observations :math:`y`, covariance :math:`\Sigma`, and entropy
strength :math:`\theta > 0`, refinement minimizes

.. math::

    \tfrac12 (\sum_i q_i a_i-y)^T\Sigma^{-1}
    (\sum_i q_i a_i-y) + \theta D_{KL}(q\|p)

over the finite probability simplex. The implementation solves the smooth,
convex dual in covariance-whitened observable coordinates. Thus ``theta`` is a
dimensionless strength multiplying relative entropy, rather than an inferred
measurement uncertainty.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..observation import CholeskyCovarianceAction, PrecisionCovarianceAction
from ..units import (
    conversion_factor,
    convert_value,
    derived_unit,
    SECOND,
    UnitDefinition,
)


EnsembleSourceKind: TypeAlias = Literal[
    "physical-equilibrium", "empirical", "proposal-only"
]
EnsembleObservationUsage: TypeAlias = Literal[
    "calibration", "model-selection", "held-out"
]
ThermodynamicClosureOutcome: TypeAlias = Literal["passed", "failed", "inconclusive"]

_SOURCE_KINDS = ("physical-equilibrium", "empirical", "proposal-only")
_OBSERVATION_USAGES = ("calibration", "model-selection", "held-out")
_BME_CONVENTION = "gaussian-observation-plus-theta-kl-q-reference"
_RATE_PER_SECOND = derived_unit("s^-1", ((SECOND, -1),))


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifiers(
    values: Sequence[str], name: str, /, *, minimum_size: int = 1
) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name) for value in values)
    if len(normalized) < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique.")
    return normalized


def _lineage_identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    return tuple(sorted(_identifiers(values, name)))


def _lineages_overlap(
    first: EnsembleObservablePlan, second: EnsembleObservablePlan, /
) -> bool:
    return bool(set(first.lineage_ids) & set(second.lineage_ids))


def _positive_float(value: float, name: str, /) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return normalized


def _positive_integer(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _as_host_float_array(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _scalar_bool(value: ArrayLike, /) -> bool:
    return bool(np.asarray(jax.device_get(jnp.asarray(value))).reshape(()))


class PhysicalEquilibriumSupportProvenance(StrictModule, NonTrainableState):
    """Typed evidence that a support represents an authorized equilibrium measure."""

    source_id: str = eqx.field(static=True)
    reference_manifest_id: str = eqx.field(static=True)
    equilibrium_sampling_method_id: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    state_ids: tuple[str, ...] = eqx.field(static=True)
    convergence_evidence_id: str = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    rights_manifest_id: str = eqx.field(static=True)
    authorized_use_ids: tuple[str, ...] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        reference_manifest_id: str,
        equilibrium_sampling_method_id: str,
        condition_id: str,
        state_ids: Sequence[str],
        convergence_evidence_id: str,
        /,
        *,
        converged: bool,
        rights_manifest_id: str,
        authorized_use_ids: Sequence[str],
    ):
        if not isinstance(converged, bool):
            raise TypeError("converged must be boolean.")
        source = _identifier(source_id, "Source ID")
        reference = _identifier(reference_manifest_id, "Reference manifest ID")
        method = _identifier(
            equilibrium_sampling_method_id, "Equilibrium sampling method ID"
        )
        condition = _identifier(condition_id, "Condition ID")
        states = _lineage_identifiers(state_ids, "State IDs")
        convergence = _identifier(convergence_evidence_id, "Convergence evidence ID")
        rights = _identifier(rights_manifest_id, "Rights manifest ID")
        uses = _lineage_identifiers(authorized_use_ids, "Authorized use IDs")
        self.source_id = source
        self.reference_manifest_id = reference
        self.equilibrium_sampling_method_id = method
        self.condition_id = condition
        self.state_ids = states
        self.convergence_evidence_id = convergence
        self.converged = converged
        self.rights_manifest_id = rights
        self.authorized_use_ids = uses
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "physical-equilibrium-support-provenance-v1",
                "source_id": source,
                "reference_manifest_id": reference,
                "equilibrium_sampling_method_id": method,
                "condition_id": condition,
                "state_ids": list(states),
                "convergence_evidence_id": convergence,
                "converged": converged,
                "rights_manifest_id": rights,
                "authorized_use_ids": list(uses),
            }
        )

    @property
    def eligible_for_reweighting(self) -> bool:
        return self.converged and "ensemble-reweighting" in self.authorized_use_ids


class EnsembleSupport(StrictModule, NonTrainableState):
    """A normalized finite support with explicit scientific provenance.

    ``statistical_inefficiency`` is the declared factor used to convert the
    ordinary importance-weight ESS into a correlation-adjusted ESS. It must be
    at least one. For proposal-only collections, ``log_reference_weights`` are
    merely baseline analysis weights; they are never treated as density ratios.
    """

    log_reference_weights: Array
    statistical_inefficiency: Array
    source_kind: EnsembleSourceKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    physical_provenance: PhysicalEquilibriumSupportProvenance | None
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        log_reference_weights: ArrayLike,
        source_kind: EnsembleSourceKind,
        source_id: str | PhysicalEquilibriumSupportProvenance,
        /,
        *,
        statistical_inefficiency: float = 1.0,
    ):
        values = _as_host_float_array(log_reference_weights, "Reference log weights")
        if values.ndim != 1 or values.size < 2:
            raise ValueError(
                "Reference log weights must be a vector with at least two samples."
            )
        kind = str(source_kind).strip()
        if kind not in _SOURCE_KINDS:
            raise ValueError(
                "source_kind must be physical-equilibrium, empirical, or proposal-only."
            )
        if kind == "physical-equilibrium":
            if not isinstance(source_id, PhysicalEquilibriumSupportProvenance):
                raise TypeError(
                    "physical-equilibrium support requires typed physical provenance."
                )
            identifier = source_id.source_id
            physical_provenance = source_id
        else:
            if isinstance(source_id, PhysicalEquilibriumSupportProvenance):
                raise TypeError(
                    "Typed physical provenance cannot label empirical or proposal support."
                )
            identifier = _identifier(source_id, "Source ID")
            physical_provenance = None
        inefficiency = _positive_float(
            statistical_inefficiency, "Statistical inefficiency"
        )
        if inefficiency < 1.0:
            raise ValueError("Statistical inefficiency must be at least one.")
        maximum = float(np.max(values))
        log_normalizer = maximum + float(np.log(np.sum(np.exp(values - maximum))))
        normalized = jax.lax.stop_gradient(jnp.asarray(values - log_normalizer))
        inefficiency_array = jax.lax.stop_gradient(
            jnp.asarray(inefficiency, dtype=normalized.dtype)
        )
        self.log_reference_weights = normalized
        self.statistical_inefficiency = inefficiency_array
        self.source_kind = kind  # type: ignore[assignment]
        self.source_id = identifier
        self.physical_provenance = physical_provenance
        self.support_id = canonical_fingerprint(
            {
                "kind": "ensemble-support-v2",
                "source_kind": kind,
                "source_id": identifier,
                "physical_provenance_id": (
                    None
                    if physical_provenance is None
                    else physical_provenance.provenance_id
                ),
                "log_reference_weights": array_tree_fingerprint(normalized),
                "statistical_inefficiency": inefficiency,
                "proposal_density_available": False if kind == "proposal-only" else None,
            }
        )

    @property
    def sample_count(self) -> int:
        return int(self.log_reference_weights.shape[0])

    @property
    def physical_equilibrium(self) -> bool:
        return (
            self.source_kind == "physical-equilibrium"
            and self.physical_provenance is not None
            and self.physical_provenance.eligible_for_reweighting
        )


class EnsembleSupportPolicy(StrictModule, NonTrainableState):
    """Predeclared absolute and fractional effective-support requirements."""

    minimum_effective_sample_size: float = eqx.field(static=True)
    minimum_effective_sample_fraction: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        minimum_effective_sample_size: float,
        minimum_effective_sample_fraction: float,
        /,
    ):
        minimum_size = _positive_float(
            minimum_effective_sample_size, "Minimum effective sample size"
        )
        if minimum_size <= 1.0:
            raise ValueError("Minimum effective sample size must exceed one.")
        minimum_fraction = _positive_float(
            minimum_effective_sample_fraction,
            "Minimum effective sample fraction",
        )
        if minimum_fraction >= 1.0:
            raise ValueError("Minimum effective sample fraction must be below one.")
        self.minimum_effective_sample_size = minimum_size
        self.minimum_effective_sample_fraction = minimum_fraction
        self.policy_id = canonical_fingerprint(
            {
                "kind": "ensemble-support-policy-v1",
                "minimum_effective_sample_size": minimum_size.hex(),
                "minimum_effective_sample_fraction": minimum_fraction.hex(),
                "boundary": "less-than-or-equal-is-invalid",
            }
        )


CovarianceAction: TypeAlias = CholeskyCovarianceAction | PrecisionCovarianceAction


class EnsembleObservablePlan(StrictModule, NonTrainableState):
    """Fixed observables with immutable measurement and derivation lineage.

    Molecular forward models must be evaluated before constructing this plan:
    its leading axis enumerates conformations and reweighting averages those
    already-evaluated observations. Source, case, and parent IDs identify the
    underlying datum independently of its role-specific ``observation_id``.
    """

    per_sample_observables: Array
    observed: Array
    covariance: CovarianceAction
    observation_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    parent_ids: tuple[str, ...] = eqx.field(static=True)
    usage: EnsembleObservationUsage = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        per_sample_observables: ArrayLike,
        observed: ArrayLike,
        covariance: CovarianceAction,
        observation_id: str,
        /,
        *,
        source_ids: Sequence[str],
        case_ids: Sequence[str],
        parent_ids: Sequence[str],
        usage: EnsembleObservationUsage = "calibration",
    ):
        if not isinstance(
            covariance, (CholeskyCovarianceAction, PrecisionCovarianceAction)
        ):
            raise TypeError(
                "covariance must be a CholeskyCovarianceAction or "
                "PrecisionCovarianceAction."
            )
        per_sample = _as_host_float_array(
            per_sample_observables, "Per-sample observables"
        )
        aggregate = _as_host_float_array(observed, "Observed aggregate")
        if per_sample.ndim != 2 or per_sample.shape[0] < 2:
            raise ValueError(
                "Per-sample observables must have shape (sample, observable) "
                "with at least two samples."
            )
        if aggregate.shape != (per_sample.shape[1],):
            raise ValueError("Observed aggregate must match the observable axis exactly.")
        if covariance.layout.size != per_sample.shape[1]:
            raise ValueError("Covariance layout must match the observable axis exactly.")
        identifier = _identifier(observation_id, "Observation ID")
        sources = _lineage_identifiers(source_ids, "Source IDs")
        cases = _lineage_identifiers(case_ids, "Case IDs")
        parents = _lineage_identifiers(parent_ids, "Parent IDs")
        usage_ = str(usage).strip()
        if usage_ not in _OBSERVATION_USAGES:
            raise ValueError("usage must be calibration, model-selection, or held-out.")
        dtype = (
            covariance.lower_cholesky.dtype
            if isinstance(covariance, CholeskyCovarianceAction)
            else covariance.precision.dtype
        )
        per_sample_array = jax.lax.stop_gradient(jnp.asarray(per_sample, dtype=dtype))
        aggregate_array = jax.lax.stop_gradient(jnp.asarray(aggregate, dtype=dtype))
        self.per_sample_observables = per_sample_array
        self.observed = aggregate_array
        self.covariance = covariance
        self.observation_id = identifier
        self.source_ids = sources
        self.case_ids = cases
        self.parent_ids = parents
        self.usage = usage_  # type: ignore[assignment]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ensemble-observable-plan-v2",
                "observation_id": identifier,
                "source_ids": list(sources),
                "case_ids": list(cases),
                "parent_ids": list(parents),
                "usage": usage_,
                "per_sample_observables": array_tree_fingerprint(per_sample_array),
                "observed": array_tree_fingerprint(aggregate_array),
                "covariance": covariance.action_id,
                "averaging_order": "per-sample-forward-model-then-ensemble-average",
            }
        )

    @property
    def sample_count(self) -> int:
        return int(self.per_sample_observables.shape[0])

    @property
    def observable_count(self) -> int:
        return int(self.per_sample_observables.shape[1])

    @property
    def lineage_ids(self) -> tuple[str, ...]:
        return self.source_ids + self.case_ids + self.parent_ids

    def whiten(self, residual: ArrayLike, /) -> Array:
        """Apply the covariance-native whitening action to one or more rows."""
        values = jnp.asarray(residual, dtype=self.observed.dtype)
        if values.shape[-1:] != (self.observable_count,):
            raise ValueError("Residual trailing axis must match the observable axis.")
        if isinstance(self.covariance, CholeskyCovarianceAction):
            flattened = values.reshape((-1, self.observable_count))
            whitened = jsp.linalg.solve_triangular(
                self.covariance.lower_cholesky,
                flattened.T,
                lower=True,
            ).T
            return whitened.reshape(values.shape)
        precision_cholesky = jnp.linalg.cholesky(self.covariance.precision)
        return values @ precision_cholesky


class EnsembleOptimizationEvidence(StrictModule):
    """Numerical evidence for one convex dual solve."""

    regularization: Array
    primal_objective: Array
    dual_objective: Array
    dual_gradient_norm: Array
    iterations: Array
    converged: Array
    finite: Array
    successful: Array
    convention: str = eqx.field(static=True)


class ConvexSupportDiagnostics(StrictModule):
    """Distance from the observation to the finite observable convex hull."""

    witness_weights: Array
    whitened_distance: Array
    duality_gap: Array
    iterations: Array
    converged: Array
    within_support: Array
    valid: Array


class EnsembleReweightingResult(StrictModule):
    """Refined weights and explicit finite-support diagnostics."""

    log_weights: Array
    predicted_observables: Array
    relative_entropy: Array
    importance_effective_sample_size: Array
    effective_sample_size: Array
    effective_sample_fraction: Array
    optimization_evidence: EnsembleOptimizationEvidence
    convex_support: ConvexSupportDiagnostics
    support_collapsed: Array
    support_valid: Array
    physical_equilibrium_valid: Array
    support: EnsembleSupport
    fit_observation_id: str = eqx.field(static=True)
    fit_plan_id: str = eqx.field(static=True)
    fit_source_ids: tuple[str, ...] = eqx.field(static=True)
    fit_case_ids: tuple[str, ...] = eqx.field(static=True)
    fit_parent_ids: tuple[str, ...] = eqx.field(static=True)
    model_selection_plan_id: str = eqx.field(static=True)
    model_selection_source_ids: tuple[str, ...] = eqx.field(static=True)
    model_selection_case_ids: tuple[str, ...] = eqx.field(static=True)
    model_selection_parent_ids: tuple[str, ...] = eqx.field(static=True)
    support_policy_valid: Array
    support_policy_id: str = eqx.field(static=True)


def _dual_objective(
    log_reference: Array,
    whitened_residuals: Array,
    dual: Array,
    regularization: Array,
    /,
) -> Array:
    return (
        jax.scipy.special.logsumexp(log_reference - whitened_residuals @ dual)
        + 0.5 * regularization * jnp.vdot(dual, dual).real
    )


def _dual_solve(
    log_reference: Array,
    whitened_residuals: Array,
    regularization: Array,
    /,
    *,
    maximum_iterations: int,
    gradient_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dimension = whitened_residuals.shape[1]
    dual = jnp.zeros((dimension,), dtype=whitened_residuals.dtype)
    finite = jnp.all(jnp.isfinite(whitened_residuals))
    converged = jnp.asarray(False)
    gradient_norm = jnp.asarray(jnp.inf, dtype=dual.dtype)
    completed = 0
    if not _scalar_bool(finite):
        return dual, gradient_norm, jnp.asarray(completed), converged, finite, jnp.inf

    objective = _dual_objective(log_reference, whitened_residuals, dual, regularization)
    for step_index in range(maximum_iterations + 1):
        log_weights = log_reference - whitened_residuals @ dual
        log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
        weights = jnp.exp(log_weights)
        mean = weights @ whitened_residuals
        gradient = -mean + regularization * dual
        gradient_norm = jnp.linalg.norm(gradient)
        finite = (
            jnp.isfinite(objective)
            & jnp.isfinite(gradient_norm)
            & jnp.all(jnp.isfinite(log_weights))
        )
        completed = step_index
        if not _scalar_bool(finite):
            break
        if _scalar_bool(gradient_norm <= gradient_tolerance):
            converged = jnp.asarray(True)
            break
        if step_index == maximum_iterations:
            break

        centered = whitened_residuals - mean[None, :]
        hessian = centered.T @ (weights[:, None] * centered) + regularization * jnp.eye(
            dimension, dtype=dual.dtype
        )
        newton_direction = jnp.linalg.solve(hessian, gradient)
        direction_finite = jnp.all(jnp.isfinite(newton_direction))
        if not _scalar_bool(direction_finite):
            finite = jnp.asarray(False)
            break

        directional_derivative = jnp.vdot(gradient, newton_direction).real
        step_size = 1.0
        accepted = False
        candidate = dual
        candidate_objective = objective
        for _ in range(32):
            proposed = dual - step_size * newton_direction
            proposed_objective = _dual_objective(
                log_reference, whitened_residuals, proposed, regularization
            )
            sufficient_decrease = proposed_objective <= (
                objective - 1.0e-4 * step_size * directional_derivative
            )
            if _scalar_bool(jnp.isfinite(proposed_objective) & sufficient_decrease):
                candidate = proposed
                candidate_objective = proposed_objective
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break
        dual = candidate
        objective = candidate_objective

    return (
        dual,
        gradient_norm,
        jnp.asarray(completed, dtype=jnp.int32),
        converged,
        finite,
        objective,
    )


def _convex_support_diagnostics(
    whitened_residuals: Array,
    initial_weights: Array,
    /,
    *,
    tolerance: float,
    maximum_iterations: int,
) -> ConvexSupportDiagnostics:
    weights = initial_weights
    mean = weights @ whitened_residuals
    distance = jnp.linalg.norm(mean)
    gap = jnp.asarray(jnp.inf, dtype=mean.dtype)
    finite = jnp.all(jnp.isfinite(whitened_residuals))
    converged = jnp.asarray(False)
    completed = 0
    if _scalar_bool(finite):
        for step_index in range(maximum_iterations + 1):
            distance = jnp.linalg.norm(mean)
            gradient = whitened_residuals @ mean
            vertex_index = jnp.argmin(gradient)
            gap = jnp.vdot(weights, gradient).real - gradient[vertex_index]
            completed = step_index
            if _scalar_bool(
                (distance <= tolerance) | (gap <= tolerance * jnp.maximum(1.0, distance))
            ):
                converged = jnp.asarray(True)
                break
            if step_index == maximum_iterations:
                break
            displacement = whitened_residuals[vertex_index] - mean
            denominator = jnp.vdot(displacement, displacement).real
            step_size = jnp.where(
                denominator > 0.0,
                jnp.clip(-jnp.vdot(mean, displacement).real / denominator, 0.0, 1.0),
                0.0,
            )
            weights = (1.0 - step_size) * weights
            weights = weights.at[vertex_index].add(step_size)
            mean = mean + step_size * displacement
        finite = (
            finite
            & jnp.isfinite(distance)
            & jnp.isfinite(gap)
            & jnp.all(jnp.isfinite(weights))
        )
    else:
        distance = jnp.asarray(jnp.inf, dtype=mean.dtype)
    within = finite & converged & (distance <= tolerance)
    return ConvexSupportDiagnostics(
        witness_weights=weights,
        whitened_distance=distance,
        duality_gap=gap,
        iterations=jnp.asarray(completed, dtype=jnp.int32),
        converged=converged,
        within_support=within,
        valid=finite & converged,
    )


def reweight_ensemble(
    support: EnsembleSupport,
    plan: EnsembleObservablePlan,
    /,
    *,
    regularization: float | EnsembleRegularizationSelectionResult,
    support_policy: EnsembleSupportPolicy | None = None,
    maximum_iterations: int = 100,
    gradient_tolerance: float = 1.0e-10,
    convex_support_tolerance: float = 1.0e-8,
    convex_support_maximum_iterations: int = 4096,
) -> EnsembleReweightingResult:
    """Fit calibration observables under the documented BME convention.

    Geometrically incompatible observations and effective-support collapse are
    returned as failed support evidence. They do not raise runtime exceptions or
    acquire mass outside the declared finite support.
    """
    if not isinstance(support, EnsembleSupport):
        raise TypeError("support must be EnsembleSupport.")
    if not isinstance(plan, EnsembleObservablePlan):
        raise TypeError("plan must be EnsembleObservablePlan.")
    if plan.usage != "calibration":
        raise ValueError("Only calibration observations may fit ensemble weights.")
    if support.sample_count != plan.sample_count:
        raise ValueError("Support and observable plan sample axes must match.")
    if isinstance(regularization, EnsembleRegularizationSelectionResult):
        if not _scalar_bool(regularization.valid):
            raise ValueError("Regularization selection must be valid before fitting.")
        if regularization.calibration_plan_id != plan.plan_id:
            raise ValueError(
                "Regularization selection was not computed from this calibration plan."
            )
        theta = _positive_float(
            regularization.selected_regularization, "Selected regularization"
        )
        model_selection_plan_id = regularization.validation_plan_id
        model_selection_source_ids = regularization.validation_source_ids
        model_selection_case_ids = regularization.validation_case_ids
        model_selection_parent_ids = regularization.validation_parent_ids
    else:
        theta = _positive_float(regularization, "Regularization")
        model_selection_plan_id = ""
        model_selection_source_ids = ()
        model_selection_case_ids = ()
        model_selection_parent_ids = ()
    if support_policy is not None and not isinstance(
        support_policy, EnsembleSupportPolicy
    ):
        raise TypeError("support_policy must be EnsembleSupportPolicy or None.")
    policy_available = support_policy is not None
    steps = _positive_integer(maximum_iterations, "Maximum iterations")
    tolerance = _positive_float(gradient_tolerance, "Gradient tolerance")
    support_tolerance = _positive_float(
        convex_support_tolerance, "Convex-support tolerance"
    )
    support_steps = _positive_integer(
        convex_support_maximum_iterations, "Convex-support maximum iterations"
    )

    centered = plan.per_sample_observables - plan.observed[None, :]
    whitened = plan.whiten(centered)
    theta_array = jnp.asarray(theta, dtype=whitened.dtype)
    (
        dual,
        gradient_norm,
        completed,
        converged,
        finite,
        dual_objective,
    ) = _dual_solve(
        support.log_reference_weights,
        whitened,
        theta_array,
        maximum_iterations=steps,
        gradient_tolerance=tolerance,
    )

    usable_whitening = jnp.all(jnp.isfinite(whitened))
    candidate_log_weights = support.log_reference_weights - whitened @ dual
    candidate_log_weights = candidate_log_weights - jax.scipy.special.logsumexp(
        candidate_log_weights
    )
    log_weights = jnp.where(
        usable_whitening, candidate_log_weights, support.log_reference_weights
    )
    weights = jnp.exp(log_weights)
    predicted = weights @ plan.per_sample_observables
    residual = predicted - plan.observed
    relative_entropy = jnp.sum(weights * (log_weights - support.log_reference_weights))
    importance_ess = 1.0 / jnp.sum(weights * weights)
    effective_ess = importance_ess / support.statistical_inefficiency
    effective_fraction = effective_ess / support.sample_count
    if support_policy is None:
        collapsed = jnp.asarray(True)
        support_policy_id = ""
    else:
        collapsed = (effective_ess <= support_policy.minimum_effective_sample_size) | (
            effective_fraction <= support_policy.minimum_effective_sample_fraction
        )
        support_policy_id = support_policy.policy_id
    normalized_finite = (
        jnp.all(jnp.isfinite(log_weights))
        & jnp.isfinite(relative_entropy)
        & jnp.isfinite(effective_ess)
        & jnp.isclose(jnp.sum(weights), 1.0, rtol=1.0e-6, atol=1.0e-8)
    )
    primal_objective = (
        0.5 * plan.covariance.quadratic(residual) + theta_array * relative_entropy
    )
    optimization = EnsembleOptimizationEvidence(
        regularization=theta_array,
        primal_objective=primal_objective,
        dual_objective=dual_objective,
        dual_gradient_norm=gradient_norm,
        iterations=completed,
        converged=converged,
        finite=finite & normalized_finite & jnp.isfinite(primal_objective),
        successful=converged
        & finite
        & normalized_finite
        & jnp.isfinite(primal_objective),
        convention=_BME_CONVENTION,
    )
    convex_support = _convex_support_diagnostics(
        whitened,
        jnp.exp(support.log_reference_weights),
        tolerance=support_tolerance,
        maximum_iterations=support_steps,
    )
    support_valid = (
        optimization.successful
        & convex_support.within_support
        & ~collapsed
        & policy_available
    )
    physical_valid = support_valid & support.physical_equilibrium
    return EnsembleReweightingResult(
        log_weights=log_weights,
        predicted_observables=predicted,
        relative_entropy=relative_entropy,
        importance_effective_sample_size=importance_ess,
        effective_sample_size=effective_ess,
        effective_sample_fraction=effective_fraction,
        optimization_evidence=optimization,
        convex_support=convex_support,
        support_collapsed=collapsed,
        support_policy_valid=jnp.asarray(policy_available) & ~collapsed,
        support_policy_id=support_policy_id,
        support_valid=support_valid,
        physical_equilibrium_valid=physical_valid,
        support=support,
        fit_observation_id=plan.observation_id,
        fit_plan_id=plan.plan_id,
        fit_source_ids=plan.source_ids,
        fit_case_ids=plan.case_ids,
        fit_parent_ids=plan.parent_ids,
        model_selection_plan_id=model_selection_plan_id,
        model_selection_source_ids=model_selection_source_ids,
        model_selection_case_ids=model_selection_case_ids,
        model_selection_parent_ids=model_selection_parent_ids,
    )


class EnsembleObservablePrediction(StrictModule):
    """A no-refit prediction of one explicitly held-out observable plan."""

    predicted_observables: Array
    residual: Array
    whitened_residual: Array
    quadratic: Array
    log_probability: Array
    finite: Array
    valid: Array
    support_id: str = eqx.field(static=True)
    fit_plan_id: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    observation_plan_id: str = eqx.field(static=True)
    observation_source_ids: tuple[str, ...] = eqx.field(static=True)
    observation_case_ids: tuple[str, ...] = eqx.field(static=True)
    observation_parent_ids: tuple[str, ...] = eqx.field(static=True)


def predict_held_out_observables(
    result: EnsembleReweightingResult,
    plan: EnsembleObservablePlan,
    /,
) -> EnsembleObservablePrediction:
    """Predict a held-out per-conformation observable without changing weights."""
    if not isinstance(result, EnsembleReweightingResult):
        raise TypeError("result must be EnsembleReweightingResult.")
    if not isinstance(plan, EnsembleObservablePlan):
        raise TypeError("plan must be EnsembleObservablePlan.")
    if plan.usage != "held-out":
        raise ValueError("Held-out prediction requires usage='held-out'.")
    fit_lineage = (
        result.fit_source_ids
        + result.fit_case_ids
        + result.fit_parent_ids
        + result.model_selection_source_ids
        + result.model_selection_case_ids
        + result.model_selection_parent_ids
    )
    if set(plan.lineage_ids) & set(fit_lineage):
        raise ValueError(
            "Held-out observation lineage must be disjoint from fitted and "
            "model-selection lineages."
        )
    if plan.sample_count != result.support.sample_count:
        raise ValueError("Held-out plan and fitted support sample axes must match.")
    weights = jnp.exp(result.log_weights)
    predicted = weights @ plan.per_sample_observables
    residual = predicted - plan.observed
    whitened = plan.whiten(residual)
    quadratic = plan.covariance.quadratic(residual)
    dimension = plan.observable_count
    log_probability = -0.5 * (
        quadratic
        + plan.covariance.logdet_covariance
        + dimension * jnp.log(jnp.asarray(2.0 * np.pi, dtype=predicted.dtype))
    )
    finite = (
        jnp.all(jnp.isfinite(predicted))
        & jnp.all(jnp.isfinite(whitened))
        & jnp.isfinite(quadratic)
        & jnp.isfinite(log_probability)
    )
    return EnsembleObservablePrediction(
        predicted_observables=predicted,
        residual=residual,
        whitened_residual=whitened,
        quadratic=quadratic,
        log_probability=log_probability,
        finite=finite,
        valid=finite & result.support_valid,
        support_id=result.support.support_id,
        fit_plan_id=result.fit_plan_id,
        observation_id=plan.observation_id,
        observation_plan_id=plan.plan_id,
        observation_source_ids=plan.source_ids,
        observation_case_ids=plan.case_ids,
        observation_parent_ids=plan.parent_ids,
    )


class EnsembleRegularizationSelectionResult(StrictModule):
    """Calibration-only regularization candidates scored on model-selection data."""

    candidate_regularizations: Array
    validation_negative_log_likelihood: Array
    candidate_valid: Array
    selected_index: Array
    selected_regularization: Array
    valid: Array
    calibration_plan_id: str = eqx.field(static=True)
    validation_plan_id: str = eqx.field(static=True)
    validation_source_ids: tuple[str, ...] = eqx.field(static=True)
    validation_case_ids: tuple[str, ...] = eqx.field(static=True)
    validation_parent_ids: tuple[str, ...] = eqx.field(static=True)


def select_ensemble_regularization(
    support: EnsembleSupport,
    calibration: EnsembleObservablePlan,
    validation: EnsembleObservablePlan,
    candidate_regularizations: ArrayLike,
    /,
    *,
    support_policy: EnsembleSupportPolicy | None = None,
    maximum_iterations: int = 100,
    gradient_tolerance: float = 1.0e-10,
    convex_support_tolerance: float = 1.0e-8,
) -> EnsembleRegularizationSelectionResult:
    """Select entropy strength without consulting held-out/locked outcomes."""
    if not isinstance(calibration, EnsembleObservablePlan) or not isinstance(
        validation, EnsembleObservablePlan
    ):
        raise TypeError("calibration and validation must be observable plans.")
    if calibration.usage != "calibration":
        raise ValueError("Calibration plan must have usage='calibration'.")
    if validation.usage != "model-selection":
        raise ValueError("Validation plan must have usage='model-selection'.")
    if _lineages_overlap(calibration, validation):
        raise ValueError(
            "Calibration and model-selection observation lineages must be disjoint."
        )
    if validation.sample_count != support.sample_count:
        raise ValueError("Validation and support sample axes must match.")
    candidates_host = _as_host_float_array(
        candidate_regularizations, "Candidate regularizations"
    )
    if candidates_host.ndim != 1 or candidates_host.size == 0:
        raise ValueError("Candidate regularizations must be a non-empty vector.")
    if np.any(candidates_host <= 0.0) or len(set(candidates_host.tolist())) != len(
        candidates_host
    ):
        raise ValueError("Candidate regularizations must be positive and unique.")

    scores: list[Array] = []
    valid: list[Array] = []
    for candidate in candidates_host:
        result = reweight_ensemble(
            support,
            calibration,
            regularization=float(candidate),
            support_policy=support_policy,
            maximum_iterations=maximum_iterations,
            gradient_tolerance=gradient_tolerance,
            convex_support_tolerance=convex_support_tolerance,
        )
        weights = jnp.exp(result.log_weights)
        prediction = weights @ validation.per_sample_observables
        residual = prediction - validation.observed
        quadratic = validation.covariance.quadratic(residual)
        score = 0.5 * (
            quadratic
            + validation.covariance.logdet_covariance
            + validation.observable_count
            * jnp.log(jnp.asarray(2.0 * np.pi, dtype=prediction.dtype))
        )
        candidate_valid = result.support_valid & jnp.isfinite(score)
        scores.append(score)
        valid.append(candidate_valid)
    scores_array = jnp.stack(scores)
    valid_array = jnp.stack(valid)
    comparable = jnp.where(valid_array, scores_array, jnp.inf)
    best_valid_index = jnp.argmin(comparable).astype(jnp.int32)
    candidates = jnp.asarray(candidates_host, dtype=scores_array.dtype)
    selection_valid = jnp.any(valid_array)
    selected_index = jnp.where(
        selection_valid, best_valid_index, jnp.asarray(-1, dtype=jnp.int32)
    )
    selected_regularization = jnp.where(
        selection_valid, candidates[best_valid_index], jnp.nan
    )
    return EnsembleRegularizationSelectionResult(
        candidate_regularizations=candidates,
        validation_negative_log_likelihood=scores_array,
        candidate_valid=valid_array,
        selected_index=selected_index,
        selected_regularization=selected_regularization,
        valid=selection_valid,
        calibration_plan_id=calibration.plan_id,
        validation_plan_id=validation.plan_id,
        validation_source_ids=validation.source_ids,
        validation_case_ids=validation.case_ids,
        validation_parent_ids=validation.parent_ids,
    )


class TwoStateEquilibriumStateAssignment(StrictModule, NonTrainableState):
    """Support-bound state and independent-replica assignments."""

    support_id: str = eqx.field(static=True)
    support_provenance_id: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    state_ids: tuple[str, str] = eqx.field(static=True)
    replica_ids: tuple[str, ...] = eqx.field(static=True)
    sample_state_ids: tuple[str, ...] = eqx.field(static=True)
    sample_replica_ids: tuple[str, ...] = eqx.field(static=True)
    assignment_evidence_id: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: EnsembleSupport,
        condition_id: str,
        state_ids: tuple[str, str],
        replica_ids: Sequence[str],
        sample_state_ids: Sequence[str],
        sample_replica_ids: Sequence[str],
        assignment_evidence_id: str,
        /,
    ):
        if not isinstance(support, EnsembleSupport):
            raise TypeError("support must be EnsembleSupport.")
        condition = _identifier(condition_id, "Condition ID")
        states = _identifiers(state_ids, "State IDs", minimum_size=2)
        if len(states) != 2:
            raise ValueError("Two-state assignments require exactly two state IDs.")
        replicas = _identifiers(replica_ids, "Replica IDs", minimum_size=2)
        sample_states = tuple(
            _identifier(value, "Sample state IDs") for value in sample_state_ids
        )
        sample_replicas = tuple(
            _identifier(value, "Sample replica IDs") for value in sample_replica_ids
        )
        if (
            len(sample_states) != support.sample_count
            or len(sample_replicas) != support.sample_count
        ):
            raise ValueError("Every support sample requires one state and replica ID.")
        if not set(sample_states).issubset(states):
            raise ValueError("Sample state IDs must belong to the declared states.")
        if set(sample_replicas) != set(replicas):
            raise ValueError("Every declared replica must own support samples.")
        provenance = support.physical_provenance
        if provenance is not None and (
            provenance.condition_id != condition
            or set(provenance.state_ids) != set(states)
        ):
            raise ValueError(
                "State assignments must match physical support condition and states."
            )
        provenance_id = "" if provenance is None else provenance.provenance_id
        evidence = _identifier(assignment_evidence_id, "Assignment evidence ID")
        self.support_id = support.support_id
        self.support_provenance_id = provenance_id
        self.condition_id = condition
        self.state_ids = (states[0], states[1])
        self.replica_ids = replicas
        self.sample_state_ids = sample_states
        self.sample_replica_ids = sample_replicas
        self.assignment_evidence_id = evidence
        self.record_id = canonical_fingerprint(
            {
                "kind": "two-state-equilibrium-state-assignment-v1",
                "support_id": support.support_id,
                "support_provenance_id": provenance_id,
                "condition_id": condition,
                "state_ids": list(states),
                "replica_ids": list(replicas),
                "sample_state_ids": list(sample_states),
                "sample_replica_ids": list(sample_replicas),
                "assignment_evidence_id": evidence,
            }
        )


class TwoStateEquilibriumRecord(StrictModule, NonTrainableState):
    """Replica populations derived only from one bound reweighting result."""

    state_probabilities: Array
    log_ratio_standard_error: Array
    replica_state_probabilities: Array
    replica_effective_sample_sizes: Array
    minimum_state_overlap: Array
    condition_id: str = eqx.field(static=True)
    state_ids: tuple[str, str] = eqx.field(static=True)
    replica_ids: tuple[str, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    support_provenance_id: str = eqx.field(static=True)
    support_policy_id: str = eqx.field(static=True)
    reweighting_id: str = eqx.field(static=True)
    state_assignment_record_id: str = eqx.field(static=True)
    population_aggregation: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: EnsembleReweightingResult,
        assignment: TwoStateEquilibriumStateAssignment,
        log_ratio_standard_error: float | None,
        /,
        *,
        condition_id: str,
        state_ids: tuple[str, str],
        replica_ids: Sequence[str],
        observation_id: str,
        source_ids: Sequence[str],
    ):
        if not isinstance(result, EnsembleReweightingResult):
            raise TypeError("result must be EnsembleReweightingResult.")
        if not isinstance(assignment, TwoStateEquilibriumStateAssignment):
            raise TypeError("assignment must be TwoStateEquilibriumStateAssignment.")
        condition = _identifier(condition_id, "Condition ID")
        states = _identifiers(state_ids, "State IDs", minimum_size=2)
        replicas = _identifiers(replica_ids, "Replica IDs", minimum_size=2)
        observation = _identifier(observation_id, "Equilibrium observation ID")
        sources = _lineage_identifiers(source_ids, "Equilibrium source IDs")
        support = result.support
        provenance = support.physical_provenance
        provenance_id = "" if provenance is None else provenance.provenance_id
        expected_sources = (
            (support.source_id,)
            if provenance is None
            else tuple(sorted((provenance.source_id, provenance.reference_manifest_id)))
        )
        if sources != expected_sources:
            raise ValueError(
                "Equilibrium sources must exactly match support source/reference IDs."
            )
        if (
            assignment.support_id != support.support_id
            or assignment.support_provenance_id != provenance_id
            or assignment.condition_id != condition
            or assignment.state_ids != tuple(states)
            or assignment.replica_ids != tuple(replicas)
        ):
            raise ValueError(
                "Equilibrium assignment must exactly match result support, provenance, "
                "condition, state order, and replicas."
            )
        weights = np.exp(np.asarray(result.log_weights, dtype=float))
        inefficiency = float(np.asarray(support.statistical_inefficiency))
        replica_probabilities: list[np.ndarray] = []
        replica_ess: list[float] = []
        for replica in replicas:
            mask = np.asarray(
                [value == replica for value in assignment.sample_replica_ids],
                dtype=bool,
            )
            local_weights = weights[mask]
            local_weights = local_weights / np.sum(local_weights)
            local_states = np.asarray(assignment.sample_state_ids)[mask]
            replica_probabilities.append(
                np.asarray(
                    [np.sum(local_weights[local_states == state]) for state in states]
                )
            )
            replica_ess.append(
                float(1.0 / np.sum(local_weights * local_weights) / inefficiency)
            )
        replica_probability_array = np.stack(replica_probabilities)
        replica_ess_array = np.asarray(replica_ess)
        probabilities = (replica_ess_array @ replica_probability_array) / np.sum(
            replica_ess_array
        )
        overlap = float(np.min(replica_probability_array))
        standard_error = (
            np.nan
            if log_ratio_standard_error is None
            else float(log_ratio_standard_error)
        )
        reweighting_id = canonical_fingerprint(
            {
                "kind": "bound-ensemble-reweighting-v1",
                "support_id": support.support_id,
                "fit_plan_id": result.fit_plan_id,
                "support_policy_id": result.support_policy_id,
                "log_weights": array_tree_fingerprint(result.log_weights),
            }
        )
        self.state_probabilities = jax.lax.stop_gradient(jnp.asarray(probabilities))
        self.log_ratio_standard_error = jax.lax.stop_gradient(jnp.asarray(standard_error))
        self.replica_state_probabilities = jax.lax.stop_gradient(
            jnp.asarray(replica_probability_array)
        )
        self.replica_effective_sample_sizes = jax.lax.stop_gradient(
            jnp.asarray(replica_ess_array)
        )
        self.minimum_state_overlap = jax.lax.stop_gradient(jnp.asarray(overlap))
        self.condition_id = condition
        self.state_ids = (states[0], states[1])
        self.replica_ids = replicas
        self.observation_id = observation
        self.source_ids = sources
        self.support_id = support.support_id
        self.support_provenance_id = provenance_id
        self.support_policy_id = result.support_policy_id
        self.reweighting_id = reweighting_id
        self.state_assignment_record_id = assignment.record_id
        self.population_aggregation = "replica-effective-sample-size-weighted"
        self.record_id = canonical_fingerprint(
            {
                "kind": "two-state-equilibrium-record-v3",
                "condition_id": condition,
                "state_ids": list(states),
                "replica_ids": list(replicas),
                "observation_id": observation,
                "source_ids": list(sources),
                "support_id": support.support_id,
                "support_provenance_id": provenance_id,
                "support_policy_id": result.support_policy_id,
                "reweighting_id": reweighting_id,
                "state_assignment_record_id": assignment.record_id,
                "population_aggregation": self.population_aggregation,
                "values": array_tree_fingerprint(
                    (
                        self.state_probabilities,
                        self.log_ratio_standard_error,
                        self.replica_state_probabilities,
                        self.replica_effective_sample_sizes,
                        self.minimum_state_overlap,
                    )
                ),
            }
        )


class TwoStateKineticRecord(StrictModule, NonTrainableState):
    """Directional two-state rates with explicit, canonicalized time units."""

    forward_rate: Array
    reverse_rate: Array
    log_ratio_standard_error: Array
    condition_id: str = eqx.field(static=True)
    state_ids: tuple[str, str] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    forward_rate_unit: UnitDefinition = eqx.field(static=True)
    reverse_rate_unit: UnitDefinition = eqx.field(static=True)
    canonical_rate_unit: UnitDefinition = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward_rate: ArrayLike,
        reverse_rate: ArrayLike,
        log_ratio_standard_error: float | None,
        /,
        *,
        condition_id: str,
        state_ids: tuple[str, str],
        observation_id: str,
        source_ids: Sequence[str],
        forward_rate_unit: UnitDefinition,
        reverse_rate_unit: UnitDefinition,
    ):
        condition = _identifier(condition_id, "Condition ID")
        states = _identifiers(state_ids, "State IDs", minimum_size=2)
        if len(states) != 2:
            raise ValueError("Two-state kinetic records require exactly two states.")
        observation = _identifier(observation_id, "Kinetic observation ID")
        sources = _lineage_identifiers(source_ids, "Kinetic source IDs")
        if not isinstance(forward_rate_unit, UnitDefinition) or not isinstance(
            reverse_rate_unit, UnitDefinition
        ):
            raise TypeError("Rate units must be UnitDefinition values.")
        conversion_factor(forward_rate_unit, _RATE_PER_SECOND)
        conversion_factor(reverse_rate_unit, _RATE_PER_SECOND)
        forward = np.asarray(
            convert_value(
                forward_rate,
                source=forward_rate_unit,
                target=_RATE_PER_SECOND,
            ),
            dtype=float,
        )
        reverse = np.asarray(
            convert_value(
                reverse_rate,
                source=reverse_rate_unit,
                target=_RATE_PER_SECOND,
            ),
            dtype=float,
        )
        if forward.shape != () or reverse.shape != ():
            raise ValueError("Two-state rates must be scalar.")
        standard_error = (
            np.nan
            if log_ratio_standard_error is None
            else float(log_ratio_standard_error)
        )
        self.forward_rate = jax.lax.stop_gradient(jnp.asarray(forward))
        self.reverse_rate = jax.lax.stop_gradient(jnp.asarray(reverse))
        self.log_ratio_standard_error = jax.lax.stop_gradient(jnp.asarray(standard_error))
        self.condition_id = condition
        self.state_ids = (states[0], states[1])
        self.observation_id = observation
        self.source_ids = sources
        self.forward_rate_unit = forward_rate_unit
        self.reverse_rate_unit = reverse_rate_unit
        self.canonical_rate_unit = _RATE_PER_SECOND
        self.record_id = canonical_fingerprint(
            {
                "kind": "two-state-kinetic-record-v2",
                "condition_id": condition,
                "state_ids": list(states),
                "observation_id": observation,
                "source_ids": list(sources),
                "forward_rate_unit": forward_rate_unit.unit_id,
                "reverse_rate_unit": reverse_rate_unit.unit_id,
                "canonical_rate_unit": _RATE_PER_SECOND.unit_id,
                "values": array_tree_fingerprint(
                    (
                        self.forward_rate,
                        self.reverse_rate,
                        self.log_ratio_standard_error,
                    )
                ),
                "rate_order": "forward-state-0-to-1;reverse-state-1-to-0",
            }
        )


class TwoStateThermodynamicClosurePlan(StrictModule, NonTrainableState):
    """Provenance binding for independent equilibrium/kinetic closure evidence."""

    condition_id: str = eqx.field(static=True)
    state_ids: tuple[str, str] = eqx.field(static=True)
    replica_ids: tuple[str, ...] = eqx.field(static=True)
    equilibrium_observation_id: str = eqx.field(static=True)
    kinetic_observation_id: str = eqx.field(static=True)
    equilibrium_source_ids: tuple[str, ...] = eqx.field(static=True)
    kinetic_source_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_policy_id: str = eqx.field(static=True)
    equivalence_margin: float = eqx.field(static=True)
    confidence_multiplier: float = eqx.field(static=True)
    maximum_combined_standard_error: float = eqx.field(static=True)

    def __init__(
        self,
        condition_id: str,
        state_ids: tuple[str, str],
        replica_ids: Sequence[str],
        equilibrium_observation_id: str,
        kinetic_observation_id: str,
        /,
        *,
        support_policy_id: str,
        equilibrium_source_ids: Sequence[str],
        kinetic_source_ids: Sequence[str],
        equivalence_margin: float,
        confidence_multiplier: float,
        maximum_combined_standard_error: float,
    ):
        condition = _identifier(condition_id, "Condition ID")
        states_ = _identifiers(state_ids, "State IDs", minimum_size=2)
        if len(states_) != 2:
            raise ValueError("Two-state closure requires exactly two state IDs.")
        replicas = _identifiers(replica_ids, "Replica IDs", minimum_size=2)
        equilibrium_observation = _identifier(
            equilibrium_observation_id, "Equilibrium observation ID"
        )
        kinetic_observation = _identifier(
            kinetic_observation_id, "Kinetic observation ID"
        )
        if equilibrium_observation == kinetic_observation:
            raise ValueError("Equilibrium and kinetic observations must be distinct.")
        equilibrium_sources = _lineage_identifiers(
            equilibrium_source_ids, "Equilibrium source IDs"
        )
        policy = _identifier(support_policy_id, "Support policy ID")
        kinetic_sources = _lineage_identifiers(kinetic_source_ids, "Kinetic source IDs")
        if set(equilibrium_sources) & set(kinetic_sources):
            raise ValueError(
                "Equilibrium and kinetic closure evidence must use disjoint sources."
            )
        margin = _positive_float(equivalence_margin, "Equivalence margin")
        self.support_policy_id = policy
        confidence = _positive_float(confidence_multiplier, "Confidence multiplier")
        maximum_uncertainty = _positive_float(
            maximum_combined_standard_error,
            "Maximum combined standard error",
        )
        self.condition_id = condition
        self.state_ids = (states_[0], states_[1])
        self.replica_ids = replicas
        self.equilibrium_observation_id = equilibrium_observation
        self.kinetic_observation_id = kinetic_observation
        self.equilibrium_source_ids = equilibrium_sources
        self.kinetic_source_ids = kinetic_sources
        self.equivalence_margin = margin
        self.confidence_multiplier = confidence
        self.maximum_combined_standard_error = maximum_uncertainty
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-state-thermodynamic-closure-plan-v2",
                "condition_id": condition,
                "state_ids": list(states_),
                "replica_ids": list(replicas),
                "equilibrium_observation_id": equilibrium_observation,
                "kinetic_observation_id": kinetic_observation,
                "equilibrium_source_ids": list(equilibrium_sources),
                "kinetic_source_ids": list(kinetic_sources),
                "support_policy_id": policy,
                "equivalence_margin": margin.hex(),
                "confidence_multiplier": confidence.hex(),
                "maximum_combined_standard_error": maximum_uncertainty.hex(),
                "evidence_relation": "independent-p1-p0-equals-k01-k10",
            }
        )

    def state_assignment_record(
        self,
        support: EnsembleSupport,
        sample_state_ids: Sequence[str],
        sample_replica_ids: Sequence[str],
        assignment_evidence_id: str,
        /,
    ) -> TwoStateEquilibriumStateAssignment:
        """Bind state and replica assignments to one exact support."""
        return TwoStateEquilibriumStateAssignment(
            support,
            self.condition_id,
            self.state_ids,
            self.replica_ids,
            sample_state_ids,
            sample_replica_ids,
            assignment_evidence_id,
        )

    def equilibrium_record(
        self,
        result: EnsembleReweightingResult,
        assignment: TwoStateEquilibriumStateAssignment,
        log_ratio_standard_error: float | None,
        /,
    ) -> TwoStateEquilibriumRecord:
        """Derive replica evidence from one bound reweighting and assignment."""
        return TwoStateEquilibriumRecord(
            result,
            assignment,
            log_ratio_standard_error,
            condition_id=self.condition_id,
            state_ids=self.state_ids,
            replica_ids=self.replica_ids,
            observation_id=self.equilibrium_observation_id,
            source_ids=self.equilibrium_source_ids,
        )

    def kinetic_record(
        self,
        forward_rate: ArrayLike,
        reverse_rate: ArrayLike,
        log_ratio_standard_error: float | None,
        /,
        *,
        forward_rate_unit: UnitDefinition,
        reverse_rate_unit: UnitDefinition,
    ) -> TwoStateKineticRecord:
        """Bind directional kinetic values to this closure plan's identity."""
        return TwoStateKineticRecord(
            forward_rate,
            reverse_rate,
            log_ratio_standard_error,
            condition_id=self.condition_id,
            state_ids=self.state_ids,
            forward_rate_unit=forward_rate_unit,
            reverse_rate_unit=reverse_rate_unit,
            observation_id=self.kinetic_observation_id,
            source_ids=self.kinetic_source_ids,
        )


class ThermodynamicClosureEvidence(StrictModule):
    """Equilibrium/kinetic agreement without transferring claims between them."""

    equilibrium_log_population_ratio: Array
    kinetic_log_rate_ratio: Array
    log_ratio_residual: Array
    combined_standard_error: Array
    absolute_z_score: Array
    equivalence_interval_upper: Array
    uncertainty_useful: Array
    minimum_replica_effective_sample_size: Array
    minimum_state_overlap: Array
    maximum_replica_population_difference: Array
    held_out_observation_valid: Array
    sampling_valid: Array
    equilibrium_evidence_valid: Array
    kinetic_evidence_valid: Array
    closure_consistent: Array
    valid: Array
    outcome: ThermodynamicClosureOutcome = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    support_source_kind: EnsembleSourceKind = eqx.field(static=True)
    support_policy_id: str = eqx.field(static=True)
    equilibrium_record_id: str = eqx.field(static=True)
    kinetic_record_id: str = eqx.field(static=True)


def evaluate_two_state_thermodynamic_closure(
    result: EnsembleReweightingResult,
    held_out_prediction: EnsembleObservablePrediction,
    plan: TwoStateThermodynamicClosurePlan,
    equilibrium_record: TwoStateEquilibriumRecord,
    kinetic_record: TwoStateKineticRecord,
    /,
    *,
    required_replica_effective_sample_size: float,
    required_state_overlap: float,
    maximum_replica_population_difference: float,
    maximum_held_out_quadratic: float,
) -> ThermodynamicClosureEvidence:
    """Evaluate held-out two-state thermodynamic closure.

    Population and rate ratios are compared only after independent-replica,
    correlation-adjusted ESS, state-overlap, held-out-observation, provenance,
    and physical-reference gates. A missing gate is inconclusive; a finite ratio
    disagreement after all gates is a failed closure result. No kinetic validity
    is inferred from equilibrium sampling.
    """
    if not isinstance(result, EnsembleReweightingResult):
        raise TypeError("result must be EnsembleReweightingResult.")
    if not isinstance(held_out_prediction, EnsembleObservablePrediction):
        raise TypeError("held_out_prediction must be EnsembleObservablePrediction.")
    if not isinstance(plan, TwoStateThermodynamicClosurePlan):
        raise TypeError("plan must be TwoStateThermodynamicClosurePlan.")
    if not isinstance(equilibrium_record, TwoStateEquilibriumRecord):
        raise TypeError("equilibrium_record must be TwoStateEquilibriumRecord.")
    if not isinstance(kinetic_record, TwoStateKineticRecord):
        raise TypeError("kinetic_record must be TwoStateKineticRecord.")
    if held_out_prediction.support_id != result.support.support_id:
        raise ValueError("Held-out prediction and reweighting support disagree.")
    physical_provenance = result.support.physical_provenance
    if physical_provenance is not None and (
        physical_provenance.condition_id != plan.condition_id
        or set(physical_provenance.state_ids) != set(plan.state_ids)
    ):
        raise ValueError(
            "Physical support condition and state support must match the closure plan."
        )
    if plan.support_policy_id != result.support_policy_id:
        raise ValueError("Closure plan support policy must match the reweighting result.")
    if (
        held_out_prediction.observation_id != plan.equilibrium_observation_id
        or held_out_prediction.observation_source_ids != plan.equilibrium_source_ids
    ):
        raise ValueError(
            "Held-out prediction does not match the closure equilibrium observation "
            "and sources."
        )
    equilibrium_identity = (
        equilibrium_record.condition_id,
        equilibrium_record.state_ids,
        equilibrium_record.replica_ids,
        equilibrium_record.observation_id,
        equilibrium_record.source_ids,
    )
    planned_equilibrium_identity = (
        plan.condition_id,
        plan.state_ids,
        plan.replica_ids,
        plan.equilibrium_observation_id,
        plan.equilibrium_source_ids,
    )
    if equilibrium_identity != planned_equilibrium_identity:
        raise ValueError(
            "Equilibrium record condition, state order, replicas, observation, "
            "and sources must exactly match the closure plan."
        )
    expected_reweighting_id = canonical_fingerprint(
        {
            "kind": "bound-ensemble-reweighting-v1",
            "support_id": result.support.support_id,
            "fit_plan_id": result.fit_plan_id,
            "support_policy_id": result.support_policy_id,
            "log_weights": array_tree_fingerprint(result.log_weights),
        }
    )
    expected_provenance_id = (
        "" if physical_provenance is None else physical_provenance.provenance_id
    )
    if (
        equilibrium_record.support_id != result.support.support_id
        or equilibrium_record.support_provenance_id != expected_provenance_id
        or equilibrium_record.support_policy_id != result.support_policy_id
        or equilibrium_record.reweighting_id != expected_reweighting_id
    ):
        raise ValueError(
            "Equilibrium record must be bound to this exact support, provenance, "
            "support policy, and reweighting result."
        )
    kinetic_identity = (
        kinetic_record.condition_id,
        kinetic_record.state_ids,
        kinetic_record.observation_id,
        kinetic_record.source_ids,
    )
    planned_kinetic_identity = (
        plan.condition_id,
        plan.state_ids,
        plan.kinetic_observation_id,
        plan.kinetic_source_ids,
    )
    if kinetic_identity != planned_kinetic_identity:
        raise ValueError(
            "Kinetic record condition, state order, observation, and sources "
            "must exactly match the closure plan."
        )

    equilibrium = equilibrium_record.state_probabilities
    replica_probabilities = equilibrium_record.replica_state_probabilities
    replica_ess = equilibrium_record.replica_effective_sample_sizes
    equilibrium_se = equilibrium_record.log_ratio_standard_error
    kinetic_se = kinetic_record.log_ratio_standard_error
    required_ess = _positive_float(
        required_replica_effective_sample_size,
        "Required replica effective sample size",
    )
    required_overlap = _positive_float(required_state_overlap, "Required state overlap")
    maximum_disagreement = _positive_float(
        maximum_replica_population_difference,
        "Maximum replica population difference",
    )
    maximum_held_out = _positive_float(
        maximum_held_out_quadratic, "Maximum held-out quadratic"
    )

    forward = kinetic_record.forward_rate
    reverse = kinetic_record.reverse_rate
    overlap = equilibrium_record.minimum_state_overlap
    equilibrium_numeric_valid = (
        jnp.all(jnp.isfinite(equilibrium))
        & jnp.all(equilibrium > 0.0)
        & jnp.isclose(jnp.sum(equilibrium), 1.0, rtol=1.0e-6, atol=1.0e-8)
    )
    replica_numeric_valid = (
        jnp.all(jnp.isfinite(replica_probabilities))
        & jnp.all(replica_probabilities > 0.0)
        & jnp.all(
            jnp.isclose(
                jnp.sum(replica_probabilities, axis=1),
                1.0,
                rtol=1.0e-6,
                atol=1.0e-8,
            )
        )
    )
    kinetic_numeric_valid = (
        jnp.isfinite(forward) & jnp.isfinite(reverse) & (forward > 0.0) & (reverse > 0.0)
    )
    equilibrium_uncertainty_valid = jnp.isfinite(equilibrium_se) & (equilibrium_se > 0.0)
    kinetic_uncertainty_valid = jnp.isfinite(kinetic_se) & (kinetic_se > 0.0)
    ess_valid = jnp.all(jnp.isfinite(replica_ess)) & jnp.all(replica_ess >= required_ess)
    overlap_valid = jnp.isfinite(overlap) & (overlap >= required_overlap)
    replica_difference = jnp.max(replica_probabilities[:, 1]) - jnp.min(
        replica_probabilities[:, 1]
    )
    replica_agreement = replica_numeric_valid & (
        replica_difference <= maximum_disagreement
    )
    sampling_valid = ess_valid & overlap_valid & replica_agreement

    safe_equilibrium = jnp.where(equilibrium_numeric_valid, equilibrium, 0.5)
    safe_forward = jnp.where(kinetic_numeric_valid, forward, 1.0)
    safe_reverse = jnp.where(kinetic_numeric_valid, reverse, 1.0)
    population_log_ratio = jnp.log(safe_equilibrium[1] / safe_equilibrium[0])
    kinetic_log_ratio = jnp.log(safe_forward / safe_reverse)
    residual = population_log_ratio - kinetic_log_ratio
    combined_se = jnp.sqrt(equilibrium_se**2 + kinetic_se**2)
    absolute_z = jnp.abs(residual) / combined_se
    equivalence_upper = jnp.abs(residual) + plan.confidence_multiplier * combined_se
    uncertainty_useful = jnp.isfinite(combined_se) & (
        combined_se <= plan.maximum_combined_standard_error
    )
    held_out_valid = (
        held_out_prediction.valid
        & jnp.isfinite(held_out_prediction.quadratic)
        & (held_out_prediction.quadratic <= maximum_held_out)
    )

    equilibrium_valid = (
        result.physical_equilibrium_valid
        & held_out_valid
        & equilibrium_numeric_valid
        & equilibrium_uncertainty_valid
        & sampling_valid
    )
    kinetic_valid = kinetic_numeric_valid & kinetic_uncertainty_valid
    prerequisites = equilibrium_valid & kinetic_valid & uncertainty_useful
    closure_consistent = (
        prerequisites
        & jnp.isfinite(equivalence_upper)
        & (equivalence_upper <= plan.equivalence_margin)
    )
    valid = prerequisites & closure_consistent
    if not _scalar_bool(prerequisites):
        outcome: ThermodynamicClosureOutcome = "inconclusive"
    elif _scalar_bool(closure_consistent):
        outcome = "passed"
    else:
        outcome = "failed"
    return ThermodynamicClosureEvidence(
        equilibrium_log_population_ratio=population_log_ratio,
        kinetic_log_rate_ratio=kinetic_log_ratio,
        log_ratio_residual=residual,
        combined_standard_error=jnp.asarray(combined_se),
        absolute_z_score=absolute_z,
        equivalence_interval_upper=equivalence_upper,
        uncertainty_useful=uncertainty_useful,
        minimum_replica_effective_sample_size=jnp.min(replica_ess),
        minimum_state_overlap=overlap,
        maximum_replica_population_difference=replica_difference,
        held_out_observation_valid=held_out_valid,
        sampling_valid=sampling_valid,
        equilibrium_evidence_valid=equilibrium_valid,
        kinetic_evidence_valid=kinetic_valid,
        closure_consistent=closure_consistent,
        valid=valid,
        outcome=outcome,
        plan_id=plan.plan_id,
        support_id=result.support.support_id,
        support_source_kind=result.support.source_kind,
        support_policy_id=result.support_policy_id,
        equilibrium_record_id=equilibrium_record.record_id,
        kinetic_record_id=kinetic_record.record_id,
    )


__all__ = [
    "ConvexSupportDiagnostics",
    "EnsembleObservablePlan",
    "EnsembleObservablePrediction",
    "EnsembleObservationUsage",
    "EnsembleOptimizationEvidence",
    "EnsembleRegularizationSelectionResult",
    "EnsembleReweightingResult",
    "EnsembleSourceKind",
    "EnsembleSupport",
    "EnsembleSupportPolicy",
    "PhysicalEquilibriumSupportProvenance",
    "ThermodynamicClosureEvidence",
    "ThermodynamicClosureOutcome",
    "TwoStateEquilibriumRecord",
    "TwoStateKineticRecord",
    "TwoStateEquilibriumStateAssignment",
    "TwoStateThermodynamicClosurePlan",
    "evaluate_two_state_thermodynamic_closure",
    "predict_held_out_observables",
    "reweight_ensemble",
    "select_ensemble_regularization",
]
