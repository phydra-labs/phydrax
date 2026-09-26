#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Riemannian mesh metrics: validation, normalization, gradation, and combination.

Eigenvalues of a metric are inverse squared target edge lengths along the
corresponding eigenvectors. Every symmetric eigendecomposition and SPD decision
routes through :mod:`phydrax.linalg` (``verify_dense_properties`` and
``HermitianSpectrum``). Fields are immutable host-validated artifacts; batched
spectral kernels execute as stable module-level compiled JAX functions.
"""

from __future__ import annotations

import heapq
import math
from enum import StrEnum
from functools import partial
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..linalg import (
    DensePropertyVerificationPolicy,
    hermitian_exp,
    hermitian_log,
    HermitianSpectrum,
    verify_dense_properties,
)
from ..nonlinear import Brent, NonlinearTermination, scalar_root, ScalarRootProblem
from ._scope import MeshingScope


_SYMMETRY_POLICY = DensePropertyVerificationPolicy(require_hermitian=True)
_SPD_POLICY = DensePropertyVerificationPolicy(require_positive_definite=True)
# Residual of log(complexity): a relative complexity error of 1e-11. Bracket
# width never terminates the solve early; only the residual certifies it.
_COMPLEXITY_TERMINATION = NonlinearTermination(
    absolute_residual=1.0e-11,
    relative_residual=0.0,
    absolute_step=0.0,
    relative_step=0.0,
    maximum_steps=200,
)


def _positive(value: object, name: str, /) -> float:
    result = finite_real_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _at_least_one(value: object, name: str, /) -> float:
    result = finite_real_scalar(value, name)
    if result < 1.0:
        raise ValueError(f"{name} must be finite and at least one.")
    return result


def _tensor_array(values: ArrayLike, rows: int, /) -> np.ndarray:
    tensors = np.asarray(values, dtype=np.float64)
    if (
        tensors.ndim != 3
        or tensors.shape[0] != rows
        or tensors.shape[1] != tensors.shape[2]
        or tensors.shape[1] not in (1, 2, 3)
        or not np.all(np.isfinite(tensors))
    ):
        raise ValueError(
            "Metric tensors must be finite aligned square matrices of dimension 1-3."
        )
    return tensors


@jax.jit
def _symmetric_spectrum(values: Array, /) -> tuple[Array, Array, Array]:
    spectrum = HermitianSpectrum(values)
    return spectrum.eigenvalues, spectrum.eigenvectors, spectrum.valid


def _host_spectrum(values: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Host eigenpairs of exactly symmetric tensors, ascending eigenvalues."""
    eigenvalues, eigenvectors, valid = _symmetric_spectrum(
        jnp.asarray(values, dtype=jnp.float64)
    )
    if not np.all(np.asarray(valid)):
        raise ValueError("Metric tensors must have a finite symmetric spectrum.")
    return np.asarray(eigenvalues), np.asarray(eigenvectors)


def _compose(eigenvalues: np.ndarray, eigenvectors: np.ndarray, /) -> np.ndarray:
    values = (eigenvectors * eigenvalues[:, None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    # Reconstruction rounding only; the spectral factors are exactly symmetric.
    return 0.5 * (values + np.swapaxes(values, -1, -2))


class _TensorProperties(NamedTuple):
    symmetric: np.ndarray
    hermitian: np.ndarray
    positive_definite: np.ndarray
    defect: np.ndarray
    tolerance: np.ndarray


def _tensor_properties(values: np.ndarray, /) -> _TensorProperties:
    evidence = verify_dense_properties(jnp.asarray(values), policy=_SYMMETRY_POLICY)
    return _TensorProperties(
        np.asarray(evidence.matrix),
        np.asarray(evidence.hermitian),
        np.asarray(evidence.positive_definite),
        np.asarray(evidence.hermitian_defect),
        np.asarray(evidence.tolerance),
    )


class _BoundViolations(NamedTuple):
    """Row masks of metric tensors outside their declared hard bounds."""

    minimum_size: np.ndarray
    maximum_size: np.ndarray
    maximum_anisotropy: np.ndarray


def _bound_violations(
    metric: np.ndarray,
    minimum_size: float,
    maximum_size: float,
    maximum_anisotropy: float,
    /,
) -> _BoundViolations:
    """Verify SPD tensors and mask rows outside their declared hard bounds.

    The bounds are ``1/maximum_size**2 <= lambda <= 1/minimum_size**2`` and
    ``lambda_max <= maximum_anisotropy**2 lambda_min`` on the
    `verify_dense_properties` spectrum. Its per-row tolerance, a dtype-epsilon
    multiple of ``lambda_max``, bounds the backward error of every computed
    eigenvalue; relative to ``lambda_min`` it grows with the row's condition
    estimate, so the admitted anisotropy roundoff is condition-scaled. Only
    violations beyond that roundoff are reported.
    """
    evidence = verify_dense_properties(jnp.asarray(metric), policy=_SPD_POLICY)
    if not np.all(np.asarray(evidence.hermitian)):
        raise ValueError("Mesh metrics must be symmetric.")
    if not np.all(np.asarray(evidence.positive_definite)):
        raise ValueError("Mesh metrics must be positive definite.")
    eigenvalues = np.asarray(evidence.eigenvalues)
    tolerance = np.asarray(evidence.tolerance)
    smallest, largest = eigenvalues[:, 0], eigenvalues[:, -1]
    return _BoundViolations(
        largest - tolerance > minimum_size**-2,
        smallest + tolerance < maximum_size**-2,
        largest - tolerance > maximum_anisotropy**2 * (smallest + tolerance),
    )


class MeshMetricField(StrictModule, NonTrainableState):
    """Vertex-associated SPD Riemannian metric within its declared hard bounds.

    Rows follow the scope's entity order. A metric eigenvalue ``lambda`` requests
    Euclidean edge length ``1 / sqrt(lambda)`` along its eigenvector. Construction
    certifies every row from the :func:`phydrax.linalg.verify_dense_properties`
    spectrum: ``1 / maximum_size**2 <= lambda_i <= 1 / minimum_size**2`` and
    ``sqrt(lambda_max / lambda_min) <= maximum_anisotropy``, admitting only
    eigenvalue roundoff (dtype epsilon scaled by the row's spectrum and
    condition). Values are stored unchanged; out-of-bound or untrusted tensors
    enter as :class:`MeshMetricSamples` and are bounded explicitly by
    :func:`normalize_mesh_metric`. Gradation is not a field property: it is
    requested by a `MetricGradationPolicy` (or provider options) and certified
    by `MetricGradationEvidence`.
    """

    scope: MeshingScope
    values: Array
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_anisotropy: float = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: MeshingScope,
        values: ArrayLike,
        /,
        *,
        minimum_size: float,
        maximum_size: float,
        maximum_anisotropy: float = 100.0,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        minimum = _positive(minimum_size, "minimum_size")
        maximum = _positive(maximum_size, "maximum_size")
        anisotropy = _at_least_one(maximum_anisotropy, "maximum_anisotropy")
        if minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        metric = _tensor_array(values, scope.entity_ids.shape[0])
        violations = _bound_violations(metric, minimum, maximum, anisotropy)
        for name, rows in zip(violations._fields, violations, strict=True):
            if np.any(rows):
                raise ValueError(
                    f"Mesh metric rows {np.flatnonzero(rows)[:8].tolist()} violate "
                    f"the declared {name}; bound untrusted tensors explicitly with "
                    "normalize_mesh_metric."
                )
        self.scope = scope
        self.values = jnp.asarray(metric)
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.maximum_anisotropy = anisotropy
        self.metric_id = canonical_fingerprint(
            {
                "kind": "mesh-metric-field",
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(metric),
                "minimum_size": minimum,
                "maximum_size": maximum,
                "maximum_anisotropy": anisotropy,
            }
        )


class MeshMetricSamples(StrictModule, NonTrainableState):
    """Untrusted square tensors bound to one scope.

    Samples may be asymmetric or indefinite; only an explicit
    :class:`MetricNormalizationPolicy` may repair them, and the repair is reported.
    """

    scope: MeshingScope
    values: Array
    samples_id: str = eqx.field(static=True)

    def __init__(self, scope: MeshingScope, values: ArrayLike, /):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        tensors = _tensor_array(values, scope.entity_ids.shape[0])
        self.scope = scope
        self.values = jnp.asarray(tensors)
        self.samples_id = canonical_fingerprint(
            {
                "kind": "mesh-metric-samples",
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(tensors),
            }
        )


class MetricGradationKind(StrEnum):
    """Edge-length-aware size growth law.

    ``PHYSICAL``: ``h_q <= h_p + (beta - 1) |pq|`` (arithmetic growth in physical
    length). ``METRIC``: ``h_q <= h_p beta**l_p(pq)`` with ``l_p`` the length of
    ``pq`` in the metric at ``p`` (Alauzet 2010 metric-space growth).
    """

    PHYSICAL = "physical"
    METRIC = "metric"


class MetricGradationStatus(StrEnum):
    """Termination of one metric-gradation request."""

    CONVERGED = "converged"
    BOUNDS_CONFLICT = "bounds_conflict"
    SWEEP_LIMIT = "sweep_limit"


class MetricGradationPolicy(StrictModule, NonTrainableState):
    """Scalar or anisotropic gradation with an explicit convergence budget.

    Scalar gradation bounds the determinant size ``det(M)**(-1/(2d))`` by an
    exact minimum-first relaxation and realizes it by uniform log-eigenvalue
    shifts saturated at the field's upper eigenvalue bound, which never
    increases anisotropy. Anisotropic gradation (Alauzet 2010) grows every metric
    along each incident edge and intersects the grown metric at the neighbor; an
    intersection outside the field's hard size or anisotropy bounds is withheld
    (the hard bounds win, every row keeps dominating its input, and the
    a-posteriori ``maximum_violation`` reports the growth the bounds prevent).
    Sweeps repeat until no admissible metric grows by more than
    ``relative_tolerance`` or ``maximum_sweeps`` is exhausted; exhaustion is
    reported as non-convergence, never hidden. The policy owns the requested
    gradation: metric fields carry no gradation bound.
    """

    maximum_gradation: float = eqx.field(static=True)
    kind: MetricGradationKind = eqx.field(static=True)
    anisotropic: bool = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_sweeps: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_gradation: float,
        /,
        *,
        kind: MetricGradationKind = MetricGradationKind.PHYSICAL,
        anisotropic: bool = False,
        relative_tolerance: float = 1.0e-10,
        maximum_sweeps: int = 1024,
    ):
        gradation = _at_least_one(maximum_gradation, "maximum_gradation")
        if not isinstance(kind, MetricGradationKind):
            raise TypeError("kind must be MetricGradationKind.")
        if not isinstance(anisotropic, (bool, np.bool_)):
            raise TypeError("anisotropic must be a bool.")
        tolerance = finite_real_scalar(relative_tolerance, "relative_tolerance")
        if tolerance < 0.0:
            raise ValueError("relative_tolerance must be non-negative.")
        if not isinstance(maximum_sweeps, (int, np.integer)) or maximum_sweeps <= 0:
            raise ValueError("maximum_sweeps must be a positive integer.")
        self.maximum_gradation = gradation
        self.kind = kind
        self.anisotropic = bool(anisotropic)
        self.relative_tolerance = tolerance
        self.maximum_sweeps = int(maximum_sweeps)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "metric-gradation-policy",
                "maximum_gradation": gradation,
                "law": kind.value,
                "anisotropic": bool(anisotropic),
                "relative_tolerance": tolerance,
                "maximum_sweeps": int(maximum_sweeps),
            }
        )


class MetricGradationEvidence(StrictModule, NonTrainableState):
    """Termination and a-posteriori edge certification of one gradation.

    ``CONVERGED`` requires ``maximum_violation <= relative_tolerance``.
    ``BOUNDS_CONFLICT`` means the hard size or anisotropy bounds withheld every
    remaining update; ``SWEEP_LIMIT`` means admissible updates remained at the
    execution bound. ``maximum_violation`` is measured independently after
    termination over every directed edge: the relative excess of a size over its
    growth bound (scalar) or the largest eigenvalue of
    ``M_q**(-1/2) G_(p->q) M_q**(-1/2)`` minus one (anisotropic), clipped at zero.
    """

    policy_id: str = eqx.field(static=True)
    status: MetricGradationStatus = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    updates: int = eqx.field(static=True)
    modified_count: int = eqx.field(static=True)
    maximum_violation: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: MetricGradationPolicy,
        /,
        *,
        status: MetricGradationStatus,
        iterations: int,
        updates: int,
        modified_count: int,
        maximum_violation: float,
    ):
        if not isinstance(policy, MetricGradationPolicy):
            raise TypeError("policy must be MetricGradationPolicy.")
        if not isinstance(status, MetricGradationStatus):
            raise TypeError("status must be MetricGradationStatus.")
        self.policy_id = policy.policy_id
        self.status = status
        self.iterations = int(iterations)
        self.updates = int(updates)
        self.modified_count = int(modified_count)
        self.maximum_violation = float(maximum_violation)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "metric-gradation-evidence",
                "policy": policy.policy_id,
                "status": self.status.value,
                "iterations": self.iterations,
                "updates": self.updates,
                "modified_count": self.modified_count,
                "maximum_violation": self.maximum_violation,
            }
        )

    @property
    def converged(self) -> bool:
        return self.status is MetricGradationStatus.CONVERGED


class MetricGradationError(ValueError):
    """A metric-gradation request could not satisfy its scientific contract."""

    def __init__(self, evidence: MetricGradationEvidence, /):
        if not isinstance(evidence, MetricGradationEvidence):
            raise TypeError("evidence must be MetricGradationEvidence.")
        self.evidence = evidence
        super().__init__(
            "Metric gradation failed "
            f"({evidence.status.value}, maximum violation "
            f"{evidence.maximum_violation:.17g})."
        )


class MetricComplexityStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    MET = "met"
    TARGET_BELOW_MINIMUM = "target_below_minimum"
    TARGET_ABOVE_MAXIMUM = "target_above_maximum"
    ROOT_FAILED = "root_failed"


class MetricNormalizationPolicy(StrictModule, NonTrainableState):
    """Explicit size, anisotropy, complexity, repair, and gradation controls.

    Eigenvalues are clipped to ``[1/maximum_size**2, 1/minimum_size**2]``; the
    anisotropy bound then raises small eigenvalues to
    ``lambda_max / maximum_anisotropy**2`` (refinement only). A target complexity
    ``N = sum_i V_i sqrt(det M_i)`` over vertex volumes ``V_i`` selects one
    global scale ``s`` of the raw eigenvalues so the bounded metric meets ``N``,
    solved by the native bracketed scalar root. Gradation follows and can only
    refine, so the final complexity is reported separately. Asymmetric samples
    are replaced by their symmetric part only when ``symmetrize`` is set, and
    non-positive eigenvalues are lifted to the maximum-size bound only when
    ``project_indefinite`` is set.
    """

    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_anisotropy: float = eqx.field(static=True)
    target_complexity: float | None = eqx.field(static=True)
    gradation: MetricGradationPolicy | None
    symmetrize: bool = eqx.field(static=True)
    project_indefinite: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_size: float,
        maximum_size: float,
        maximum_anisotropy: float = 100.0,
        target_complexity: float | None = None,
        gradation: MetricGradationPolicy | None = None,
        symmetrize: bool = False,
        project_indefinite: bool = False,
    ):
        minimum = _positive(minimum_size, "minimum_size")
        maximum = _positive(maximum_size, "maximum_size")
        if minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        anisotropy = _at_least_one(maximum_anisotropy, "maximum_anisotropy")
        complexity = (
            None
            if target_complexity is None
            else _positive(target_complexity, "target_complexity")
        )
        if gradation is not None and not isinstance(gradation, MetricGradationPolicy):
            raise TypeError("gradation must be MetricGradationPolicy or None.")
        if not isinstance(symmetrize, (bool, np.bool_)) or not isinstance(
            project_indefinite, (bool, np.bool_)
        ):
            raise TypeError("symmetrize and project_indefinite must be bools.")
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.maximum_anisotropy = anisotropy
        self.target_complexity = complexity
        self.gradation = gradation
        self.symmetrize = bool(symmetrize)
        self.project_indefinite = bool(project_indefinite)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "metric-normalization-policy",
                "minimum_size": minimum,
                "maximum_size": maximum,
                "maximum_anisotropy": anisotropy,
                "target_complexity": complexity,
                "gradation": None if gradation is None else gradation.policy_id,
                "symmetrize": bool(symmetrize),
                "project_indefinite": bool(project_indefinite),
            }
        )


class MetricNormalizationEvidence(StrictModule, NonTrainableState):
    """Every repair, clamp, complexity, and gradation decision of one normalization."""

    policy_id: str = eqx.field(static=True)
    input_id: str = eqx.field(static=True)
    symmetrized_count: int = eqx.field(static=True)
    maximum_asymmetry: float = eqx.field(static=True)
    projected_eigenvalue_count: int = eqx.field(static=True)
    projected_tensor_count: int = eqx.field(static=True)
    minimum_size_clamped_count: int = eqx.field(static=True)
    maximum_size_clamped_count: int = eqx.field(static=True)
    anisotropy_clamped_count: int = eqx.field(static=True)
    complexity_status: MetricComplexityStatus = eqx.field(static=True)
    complexity_scale: float = eqx.field(static=True)
    target_complexity: float | None = eqx.field(static=True)
    scaled_complexity: float | None = eqx.field(static=True)
    final_complexity: float | None = eqx.field(static=True)
    gradation: MetricGradationEvidence | None
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: MetricNormalizationPolicy,
        input_id: str,
        /,
        *,
        symmetrized_count: int,
        maximum_asymmetry: float,
        projected_eigenvalue_count: int,
        projected_tensor_count: int,
        minimum_size_clamped_count: int,
        maximum_size_clamped_count: int,
        anisotropy_clamped_count: int,
        complexity_status: MetricComplexityStatus,
        complexity_scale: float,
        scaled_complexity: float | None,
        final_complexity: float | None,
        gradation: MetricGradationEvidence | None,
    ):
        if not isinstance(policy, MetricNormalizationPolicy):
            raise TypeError("policy must be MetricNormalizationPolicy.")
        if not isinstance(complexity_status, MetricComplexityStatus):
            raise TypeError("complexity_status must be MetricComplexityStatus.")
        if gradation is not None and not isinstance(gradation, MetricGradationEvidence):
            raise TypeError("gradation must be MetricGradationEvidence or None.")
        self.policy_id = policy.policy_id
        self.input_id = str(input_id)
        self.symmetrized_count = int(symmetrized_count)
        self.maximum_asymmetry = float(maximum_asymmetry)
        self.projected_eigenvalue_count = int(projected_eigenvalue_count)
        self.projected_tensor_count = int(projected_tensor_count)
        self.minimum_size_clamped_count = int(minimum_size_clamped_count)
        self.maximum_size_clamped_count = int(maximum_size_clamped_count)
        self.anisotropy_clamped_count = int(anisotropy_clamped_count)
        self.complexity_status = complexity_status
        self.complexity_scale = float(complexity_scale)
        self.target_complexity = policy.target_complexity
        self.scaled_complexity = (
            None if scaled_complexity is None else float(scaled_complexity)
        )
        self.final_complexity = (
            None if final_complexity is None else float(final_complexity)
        )
        self.gradation = gradation
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "metric-normalization-evidence",
                "policy": policy.policy_id,
                "input": self.input_id,
                "symmetrized": self.symmetrized_count,
                "asymmetry": self.maximum_asymmetry,
                "projected_eigenvalues": self.projected_eigenvalue_count,
                "projected_tensors": self.projected_tensor_count,
                "minimum_size_clamped": self.minimum_size_clamped_count,
                "maximum_size_clamped": self.maximum_size_clamped_count,
                "anisotropy_clamped": self.anisotropy_clamped_count,
                "complexity_status": complexity_status.value,
                "complexity_scale": self.complexity_scale,
                "scaled_complexity": self.scaled_complexity,
                "final_complexity": self.final_complexity,
                "gradation": None if gradation is None else gradation.evidence_id,
            }
        )

    @property
    def passed(self) -> bool:
        complexity = self.complexity_status in (
            MetricComplexityStatus.NOT_REQUESTED,
            MetricComplexityStatus.MET,
        )
        return complexity and (self.gradation is None or self.gradation.converged)


def _bounded_eigenvalues(
    values: Array, lower: float, upper: float, anisotropy: float, /
) -> Array:
    clipped = jnp.clip(values, lower, upper)
    floor = jnp.max(clipped, axis=-1, keepdims=True) / anisotropy**2
    return jnp.maximum(clipped, floor)


def _log_complexity(
    scale_log: Array,
    eigenvalues: Array,
    log_volumes: Array,
    lower: float,
    upper: float,
    anisotropy: float,
    /,
) -> Array:
    bounded = _bounded_eigenvalues(
        jnp.exp(scale_log) * eigenvalues, lower, upper, anisotropy
    )
    return logsumexp(0.5 * jnp.sum(jnp.log(bounded), axis=-1) + log_volumes)


class _ComplexityArguments(NamedTuple):
    eigenvalues: Array
    log_volumes: Array
    log_target: Array


class _ComplexityResidual(StrictModule):
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    anisotropy: float = eqx.field(static=True)

    def __call__(self, scale_log: Array, arguments: _ComplexityArguments, /) -> Array:
        return (
            _log_complexity(
                scale_log,
                arguments.eigenvalues,
                arguments.log_volumes,
                self.lower,
                self.upper,
                self.anisotropy,
            )
            - arguments.log_target
        )


def _vertex_volumes(volumes: ArrayLike | None, rows: int, /) -> np.ndarray:
    if volumes is None:
        raise ValueError("Metric complexity requires vertex_volumes.")
    result = np.asarray(volumes, dtype=np.float64)
    if (
        result.shape != (rows,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError("vertex_volumes must be positive, finite, and row aligned.")
    return result


def _complexity(eigenvalues: np.ndarray, volumes: np.ndarray, /) -> float:
    return float(np.sum(volumes * np.exp(0.5 * np.sum(np.log(eigenvalues), axis=-1))))


def _complexity_scale(
    eigenvalues: np.ndarray,
    volumes: np.ndarray,
    target: float,
    lower: float,
    upper: float,
    anisotropy: float,
    /,
) -> tuple[float, MetricComplexityStatus]:
    """Solve ``C(bounded(s * lambda)) = target`` for the global log-scale."""
    positive = eigenvalues[eigenvalues > 0.0]
    residual = _ComplexityResidual(lower, upper, anisotropy)
    arguments = _ComplexityArguments(
        jnp.asarray(eigenvalues),
        jnp.asarray(np.log(volumes)),
        jnp.asarray(math.log(target)),
    )
    if positive.size == 0:
        # Zero raw tensors saturate at the maximum size for every scale.
        value = float(residual(jnp.asarray(0.0), arguments))
        if abs(value) <= _COMPLEXITY_TERMINATION.absolute_residual:
            return 0.0, MetricComplexityStatus.MET
        return 0.0, (
            MetricComplexityStatus.TARGET_BELOW_MINIMUM
            if value > 0.0
            else MetricComplexityStatus.TARGET_ABOVE_MAXIMUM
        )
    # Beyond these endpoints every eigenvalue saturates at one size bound.
    low = math.log(lower / float(np.max(positive))) - 1.0
    high = math.log(upper / float(np.min(positive))) + 1.0
    if float(residual(jnp.asarray(low), arguments)) >= 0.0:
        return low, MetricComplexityStatus.TARGET_BELOW_MINIMUM
    if float(residual(jnp.asarray(high), arguments)) <= 0.0:
        return high, MetricComplexityStatus.TARGET_ABOVE_MAXIMUM
    result = scalar_root(
        ScalarRootProblem(
            residual,
            bracket=(low, high),
            problem_id="mesh-metric-complexity",
        ),
        method=Brent(),
        termination=_COMPLEXITY_TERMINATION,
        args=arguments,
    )
    status = (
        MetricComplexityStatus.MET
        if bool(np.asarray(result.nonlinear_result.successful))
        else MetricComplexityStatus.ROOT_FAILED
    )
    return float(np.asarray(result.root)), status


class _BoundedSpectrum(NamedTuple):
    eigenvalues: np.ndarray
    scale_log: float
    status: MetricComplexityStatus
    minimum_size_clamped: int
    maximum_size_clamped: int
    anisotropy_clamped: int
    scaled_complexity: float | None


def _bound_spectrum(
    raw: np.ndarray,
    /,
    *,
    minimum_size: float,
    maximum_size: float,
    anisotropy: float,
    target_complexity: float | None,
    volumes: np.ndarray | None,
) -> _BoundedSpectrum:
    lower = 1.0 / maximum_size**2
    upper = 1.0 / minimum_size**2
    if target_complexity is None:
        scale_log, status = 0.0, MetricComplexityStatus.NOT_REQUESTED
    else:
        assert volumes is not None
        scale_log, status = _complexity_scale(
            raw, volumes, target_complexity, lower, upper, anisotropy
        )
    scaled = math.exp(scale_log) * raw
    clipped = np.clip(scaled, lower, upper)
    bounded = np.asarray(
        _bounded_eigenvalues(jnp.asarray(scaled), lower, upper, anisotropy)
    )
    return _BoundedSpectrum(
        bounded,
        scale_log,
        status,
        int(np.count_nonzero(scaled > upper)),
        int(np.count_nonzero(scaled < lower)),
        int(np.count_nonzero(bounded > clipped)),
        None if volumes is None else _complexity(bounded, volumes),
    )


def _edges(adjacency: ArrayLike, rows: int, /) -> np.ndarray:
    edges = np.asarray(adjacency)
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("Metric adjacency must contain integer vertex rows.")
    edges = edges.astype(np.int64, copy=False)
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or np.any(edges < 0)
        or np.any(edges >= rows)
        or np.any(edges[:, 0] == edges[:, 1])
    ):
        raise ValueError("Metric adjacency must be (edges, 2) rows without self loops.")
    return edges


def _points(coordinates: ArrayLike, rows: int, dimension: int, /) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.shape != (rows, dimension) or not np.all(np.isfinite(points)):
        raise ValueError("Metric coordinates must be finite and row aligned.")
    return points


def _grade_scalar_sizes(
    sizes: np.ndarray,
    edges: np.ndarray,
    lengths: np.ndarray,
    growth: np.ndarray,
    kind: MetricGradationKind,
    /,
) -> tuple[np.ndarray, int, int]:
    """Exact minimum-first label-setting relaxation, ``O(E log V)`` on the host.

    Each candidate ``c(h_p, l) >= h_p``, so a finalized vertex is never lowered by
    a later (larger) one, and every edge is certified when its smaller endpoint is
    finalized. Processing order depends only on values (ties by row, and tied
    values never relax each other), so the result is independent of numbering.
    Per-edge growth rates are supported for size-control resolution.
    """
    count = sizes.shape[0]
    values = sizes.astype(np.float64).tolist()
    if edges.shape[0] == 0:
        return np.asarray(values, dtype=np.float64), 0, 0
    first = np.concatenate((edges[:, 0], edges[:, 1]))
    second = np.concatenate((edges[:, 1], edges[:, 0]))
    order = np.argsort(first, kind="stable")
    offsets = np.searchsorted(first[order], np.arange(count + 1)).tolist()
    targets = second[order].tolist()
    lengths_ = np.concatenate((lengths, lengths))[order]
    rates = np.concatenate((growth, growth))[order]
    physical = kind is MetricGradationKind.PHYSICAL
    increments = ((rates - 1.0) * lengths_).tolist()
    log_rates = (np.log(rates) * lengths_).tolist()
    heap = [(value, row) for row, value in enumerate(values)]
    heapq.heapify(heap)
    pops = 0
    updates = 0
    while heap:
        value, row = heapq.heappop(heap)
        if value != values[row]:
            continue
        pops += 1
        for slot in range(offsets[row], offsets[row + 1]):
            neighbor = targets[slot]
            current = values[neighbor]
            if physical:
                candidate = value + increments[slot]
                if candidate >= current:
                    continue
            else:
                # Compare in log space: beta**(l/h) overflows for tiny sizes.
                exponent = log_rates[slot] / value
                if math.log(value) + exponent >= math.log(current):
                    continue
                candidate = value * math.exp(exponent)
            values[neighbor] = candidate
            updates += 1
            heapq.heappush(heap, (candidate, neighbor))
    return np.asarray(values, dtype=np.float64), pops, updates


def _scalar_violation(
    sizes: np.ndarray,
    edges: np.ndarray,
    lengths: np.ndarray,
    growth: np.ndarray,
    kind: MetricGradationKind,
    /,
) -> float:
    if edges.shape[0] == 0:
        return 0.0
    first = np.concatenate((edges[:, 0], edges[:, 1]))
    second = np.concatenate((edges[:, 1], edges[:, 0]))
    lengths_ = np.concatenate((lengths, lengths))
    rates = np.concatenate((growth, growth))
    source = sizes[first]
    if kind is MetricGradationKind.PHYSICAL:
        log_bound = np.log(source + (rates - 1.0) * lengths_)
    else:
        log_bound = np.log(source) + np.log(rates) * lengths_ / source
    excess = np.expm1(np.log(sizes[second]) - log_bound)
    return float(max(0.0, np.max(excess)))


def _realize_mean_sizes(
    eigenvalues: np.ndarray, sizes: np.ndarray, upper: float, /
) -> np.ndarray:
    """Uniform log-eigenvalue shifts reaching ``det(M)**(-1/(2d)) = size``.

    Shifts saturate at ``max(lambda, upper)``: no eigenvalue decreases, none is
    raised above the upper bound, and anisotropy never increases. The piecewise
    linear mean is solved exactly for each saturation count.
    """
    dimension = eigenvalues.shape[1]
    logarithms = np.log(eigenvalues)
    caps = np.maximum(logarithms, math.log(upper))
    target = -2.0 * np.log(sizes)
    thresholds = caps - logarithms
    order = np.argsort(thresholds, axis=1, kind="stable")
    sorted_thresholds = np.take_along_axis(thresholds, order, axis=1)
    sorted_logarithms = np.take_along_axis(logarithms, order, axis=1)
    sorted_caps = np.take_along_axis(caps, order, axis=1)
    shift = np.full(sizes.shape, sorted_thresholds[:, -1])
    solved = np.zeros(sizes.shape, dtype=np.bool_)
    for saturated in range(dimension):
        candidate = (
            dimension * target
            - np.sum(sorted_caps[:, :saturated], axis=1)
            - np.sum(sorted_logarithms[:, saturated:], axis=1)
        ) / (dimension - saturated)
        lower_threshold = 0.0 if saturated == 0 else sorted_thresholds[:, saturated - 1]
        accept = (
            ~solved
            & (candidate >= lower_threshold)
            & (candidate <= sorted_thresholds[:, saturated])
        )
        shift = np.where(accept, candidate, shift)
        solved |= accept
    shift = np.where(target <= np.mean(logarithms, axis=1), 0.0, shift)
    return np.exp(np.minimum(logarithms + shift[:, None], caps))


def _metric_sqrt_factors(eigenvalues: Array, eigenvectors: Array, /):
    transpose = jnp.swapaxes(eigenvectors, -1, -2)
    root = (eigenvectors * jnp.sqrt(eigenvalues)[..., None, :]) @ transpose
    inverse_root = (eigenvectors / jnp.sqrt(eigenvalues)[..., None, :]) @ transpose
    return root, inverse_root


def _intersect(first: Array, second: Array, /) -> Array:
    """Simultaneous-reduction intersection ``M1**(1/2) max(I, S) M1**(1/2)``.

    ``S = M1**(-1/2) M2 M1**(-1/2)``; the result equals ``P**-T max(mu, gamma) P**-1``
    in the common eigenbasis ``P`` and dominates both inputs in Loewner order.
    """
    spectrum = HermitianSpectrum(first)
    root, inverse_root = _metric_sqrt_factors(spectrum.eigenvalues, spectrum.eigenvectors)
    reduced = HermitianSpectrum(inverse_root @ second @ inverse_root)
    lifted = (
        reduced.eigenvectors * jnp.maximum(reduced.eigenvalues, 1.0)[..., None, :]
    ) @ jnp.swapaxes(reduced.eigenvectors, -1, -2)
    combined = root @ lifted @ root
    return 0.5 * (combined + jnp.swapaxes(combined, -1, -2))


def _canonical_intersection(candidates: Array, valid: Array, /) -> Array:
    """Order-independent fold of ``(n, k, d, d)`` metrics under row validity.

    The binary intersection is not associative, so candidates are folded in the
    canonical lexicographic order of their upper-triangular entries (invalid
    slots last). Permuting the inputs therefore cannot change the result.
    """
    dimension = candidates.shape[-1]
    rows, columns = np.triu_indices(dimension)
    entries = candidates[..., rows, columns]
    keys = tuple(
        jnp.where(valid, entries[..., index], jnp.inf)
        for index in reversed(range(rows.size))
    ) + (~valid,)
    order = jnp.lexsort(keys, axis=-1)
    ordered = jnp.take_along_axis(candidates, order[..., None, None], axis=1)
    ordered_valid = jnp.take_along_axis(valid, order, axis=1)

    def fold(accumulated, slot):
        tensor, active = slot
        combined = _intersect(accumulated, tensor)
        return jnp.where(active[:, None, None], combined, accumulated), None

    result, _ = jax.lax.scan(
        fold,
        ordered[:, 0],
        (jnp.swapaxes(ordered[:, 1:], 0, 1), jnp.swapaxes(ordered_valid[:, 1:], 0, 1)),
    )
    return result


_canonical_intersection_kernel = jax.jit(_canonical_intersection)


def _grow_metrics(
    eigenvalues: Array,
    eigenvectors: Array,
    metrics: Array,
    sources: Array,
    deltas: Array,
    log_growth: Array,
    kind: str,
    /,
) -> Array:
    """Grow each source metric along its directed edge (Alauzet 2010)."""
    values = eigenvalues[sources]
    vectors = eigenvectors[sources]
    if kind == MetricGradationKind.PHYSICAL.value:
        lengths = jnp.linalg.norm(deltas, axis=-1)
        factor = 1.0 + jnp.sqrt(values) * jnp.expm1(log_growth) * lengths[:, None]
        grown = values / factor**2
        return (vectors * grown[:, None, :]) @ jnp.swapaxes(vectors, -1, -2)
    projected = contract("eji,ej->ei", vectors, deltas)
    metric_length = jnp.sqrt(jnp.sum(values * projected**2, axis=-1))
    return jnp.exp(-2.0 * log_growth * metric_length)[:, None, None] * metrics[sources]


def _maximum_growth(metrics: Array, candidates: Array, /) -> Array:
    """Largest eigenvalue of ``M**(-1/2) C M**(-1/2)`` for aligned rows."""
    spectrum = HermitianSpectrum(metrics)
    _, inverse_root = _metric_sqrt_factors(spectrum.eigenvalues, spectrum.eigenvectors)
    reduced = HermitianSpectrum(inverse_root @ candidates @ inverse_root)
    return reduced.eigenvalues[..., -1]


def _within_bounds(metrics: Array, bounds: Array, /) -> Array:
    """Rows whose spectrum respects the hard bounds ``(lower, upper, anisotropy)``.

    The roundoff allowance is half the `MeshMetricField` certification tolerance
    (``64 eps lambda_max``), so every admitted row also passes that certification.
    """
    values = HermitianSpectrum(metrics).eigenvalues
    slack = 32.0 * jnp.finfo(values.dtype).eps * values[..., -1]
    smallest, largest = values[..., 0], values[..., -1]
    return (
        (smallest + slack >= bounds[0])
        & (largest - slack <= bounds[1])
        & (largest - slack <= bounds[2] ** 2 * (smallest + slack))
    )


@partial(jax.jit, static_argnames=("kind", "maximum_sweeps"))
def _anisotropic_gradation_kernel(
    metrics: Array,
    sources: Array,
    targets: Array,
    deltas: Array,
    slots: Array,
    slot_valid: Array,
    log_growth: Array,
    tolerance: Array,
    bounds: Array,
    *,
    kind: str,
    maximum_sweeps: int,
) -> tuple[Array, Array, Array, Array, Array]:
    rows = metrics.shape[0]
    always = jnp.ones((rows, 1), dtype=jnp.bool_)
    candidate_valid = jnp.concatenate((always, slot_valid), axis=1)

    def body(state):
        current, sweeps, updates, _ = state
        spectrum = HermitianSpectrum(current)
        grown = _grow_metrics(
            spectrum.eigenvalues,
            spectrum.eigenvectors,
            current,
            sources,
            deltas,
            log_growth,
            kind,
        )
        candidates = jnp.concatenate((current[:, None], grown[slots]), axis=1)
        combined = _canonical_intersection(candidates, candidate_valid)
        # Only admissible growth is accepted, so every row keeps dominating its
        # input; an intersection outside the hard bounds is withheld and the
        # a-posteriori violation reports the gradation the bounds prevent.
        accept = (_maximum_growth(current, combined) > 1.0 + tolerance) & (
            _within_bounds(combined, bounds)
        )
        updated = jnp.where(accept[:, None, None], combined, current)
        return (
            updated,
            sweeps + 1,
            updates + jnp.sum(accept, dtype=jnp.int32),
            jnp.any(accept),
        )

    def condition(state):
        return state[3] & (state[1] < maximum_sweeps)

    final, sweeps, updates, changed = jax.lax.while_loop(
        condition,
        body,
        (
            metrics,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True),
        ),
    )
    spectrum = HermitianSpectrum(final)
    grown = _grow_metrics(
        spectrum.eigenvalues,
        spectrum.eigenvectors,
        final,
        sources,
        deltas,
        log_growth,
        kind,
    )
    violation = jnp.max(
        jnp.maximum(_maximum_growth(final[targets], grown) - 1.0, 0.0),
        initial=0.0,
    )
    return final, sweeps, updates, ~changed, violation


def _incoming_slots(targets: np.ndarray, rows: int, /) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(targets, kind="stable")
    counts = np.bincount(targets, minlength=rows)
    capacity = max(1, int(np.max(counts, initial=0)))
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    positions = np.arange(order.size) - np.repeat(offsets, counts)
    slots = np.zeros((rows, capacity), dtype=np.int32)
    valid = np.zeros((rows, capacity), dtype=np.bool_)
    slots[targets[order], positions] = order
    valid[targets[order], positions] = True
    return slots, valid


def _grade_eigenpairs(
    bounds: tuple[float, float, float],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    edges: np.ndarray,
    points: np.ndarray,
    policy: MetricGradationPolicy,
    /,
) -> tuple[np.ndarray, MetricGradationEvidence]:
    rows = eigenvalues.shape[0]
    if not policy.anisotropic:
        sizes = np.exp(-0.5 * np.mean(np.log(eigenvalues), axis=1))
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        growth = np.full((edges.shape[0],), policy.maximum_gradation)
        graded, pops, updates = _grade_scalar_sizes(
            sizes, edges, lengths, growth, policy.kind
        )
        realized = _realize_mean_sizes(eigenvalues, graded, bounds[0] ** -2)
        achieved = np.exp(-0.5 * np.mean(np.log(realized), axis=1))
        violation = _scalar_violation(achieved, edges, lengths, growth, policy.kind)
        evidence = MetricGradationEvidence(
            policy,
            status=MetricGradationStatus.CONVERGED
            if violation <= policy.relative_tolerance
            else MetricGradationStatus.BOUNDS_CONFLICT,
            iterations=pops,
            updates=updates,
            modified_count=int(np.count_nonzero(graded < sizes)),
            maximum_violation=violation,
        )
        return _compose(realized, eigenvectors), evidence
    sources = np.concatenate((edges[:, 0], edges[:, 1]))
    targets = np.concatenate((edges[:, 1], edges[:, 0]))
    slots, slot_valid = _incoming_slots(targets, rows)
    initial = _compose(eigenvalues, eigenvectors)
    final, sweeps, updates, fixed_point, violation = _anisotropic_gradation_kernel(
        jnp.asarray(initial),
        jnp.asarray(sources, dtype=jnp.int32),
        jnp.asarray(targets, dtype=jnp.int32),
        jnp.asarray(points[targets] - points[sources]),
        jnp.asarray(slots),
        jnp.asarray(slot_valid),
        jnp.asarray(math.log(policy.maximum_gradation)),
        jnp.asarray(policy.relative_tolerance),
        jnp.asarray((bounds[1] ** -2, bounds[0] ** -2, bounds[2]), dtype=jnp.float64),
        kind=policy.kind.value,
        maximum_sweeps=policy.maximum_sweeps,
    )
    graded = np.asarray(final)
    modified = np.any(graded != initial, axis=(1, 2))
    maximum_violation = float(np.asarray(violation))
    reached_fixed_point = bool(np.asarray(fixed_point))
    status = (
        MetricGradationStatus.CONVERGED
        if maximum_violation <= policy.relative_tolerance
        else MetricGradationStatus.BOUNDS_CONFLICT
        if reached_fixed_point
        else MetricGradationStatus.SWEEP_LIMIT
    )
    evidence = MetricGradationEvidence(
        policy,
        status=status,
        iterations=int(np.asarray(sweeps)),
        updates=int(np.asarray(updates)),
        modified_count=int(np.count_nonzero(modified)),
        maximum_violation=maximum_violation,
    )
    return graded, evidence


def grade_mesh_metric(
    metric: MeshMetricField,
    /,
    *,
    policy: MetricGradationPolicy,
    adjacency: ArrayLike,
    coordinates: ArrayLike,
) -> tuple[MeshMetricField, MetricGradationEvidence]:
    """Bound metric growth along every edge; refine only, never coarsen.

    ``adjacency`` holds undirected edges between metric rows and ``coordinates``
    the row-aligned vertex positions. The returned evidence certifies every edge
    a posteriori and reports convergence.
    """
    if not isinstance(metric, MeshMetricField):
        raise TypeError("metric must be MeshMetricField.")
    if not isinstance(policy, MetricGradationPolicy):
        raise TypeError("policy must be MetricGradationPolicy.")
    values = np.asarray(metric.values, dtype=np.float64)
    rows, dimension = values.shape[0], values.shape[-1]
    edges = _edges(adjacency, rows)
    points = _points(coordinates, rows, dimension)
    eigenvalues, eigenvectors = _host_spectrum(_tensor_properties(values).symmetric)
    graded, evidence = _grade_eigenpairs(
        (metric.minimum_size, metric.maximum_size, metric.maximum_anisotropy),
        eigenvalues,
        eigenvectors,
        edges,
        points,
        policy,
    )
    if not evidence.converged:
        raise MetricGradationError(evidence)
    field = MeshMetricField(
        metric.scope,
        graded,
        minimum_size=metric.minimum_size,
        maximum_size=metric.maximum_size,
        maximum_anisotropy=metric.maximum_anisotropy,
    )
    return field, evidence


def normalize_mesh_metric(
    metric: MeshMetricField | MeshMetricSamples,
    /,
    *,
    policy: MetricNormalizationPolicy,
    adjacency: ArrayLike | None = None,
    coordinates: ArrayLike | None = None,
    vertex_volumes: ArrayLike | None = None,
) -> tuple[MeshMetricField, MetricNormalizationEvidence]:
    """Repair (only as requested), scale, bound, and grade one metric.

    Order: symmetric-part repair, indefinite projection, complexity scaling with
    size and anisotropy bounds, then gradation. ``vertex_volumes`` (dual volumes
    aligned with metric rows) are required for a target complexity and enable
    complexity reporting; ``adjacency`` and ``coordinates`` are required for
    gradation.
    """
    if not isinstance(metric, (MeshMetricField, MeshMetricSamples)):
        raise TypeError("metric must be MeshMetricField or MeshMetricSamples.")
    if not isinstance(policy, MetricNormalizationPolicy):
        raise TypeError("policy must be MetricNormalizationPolicy.")
    raw = np.asarray(metric.values, dtype=np.float64)
    rows = raw.shape[0]
    properties = _tensor_properties(raw)
    asymmetric = ~properties.hermitian
    if np.any(asymmetric) and not policy.symmetrize:
        raise ValueError(
            "Metric samples are not symmetric; the policy does not request symmetrization."
        )
    eigenvalues, eigenvectors = _host_spectrum(properties.symmetric)
    nonpositive = eigenvalues <= properties.tolerance[:, None]
    if np.any(nonpositive) and not policy.project_indefinite:
        raise ValueError(
            "Metric samples are not positive definite; the policy does not request "
            "indefinite projection."
        )
    volumes = (
        None
        if vertex_volumes is None and policy.target_complexity is None
        else _vertex_volumes(vertex_volumes, rows)
    )
    bounded = _bound_spectrum(
        np.where(nonpositive, 0.0, eigenvalues),
        minimum_size=policy.minimum_size,
        maximum_size=policy.maximum_size,
        anisotropy=policy.maximum_anisotropy,
        target_complexity=policy.target_complexity,
        volumes=volumes,
    )
    values = _compose(bounded.eigenvalues, eigenvectors)
    gradation = None
    final_complexity = bounded.scaled_complexity
    if policy.gradation is not None:
        if adjacency is None or coordinates is None:
            raise ValueError("Metric gradation requires adjacency and coordinates.")
        values, gradation = _grade_eigenpairs(
            (policy.minimum_size, policy.maximum_size, policy.maximum_anisotropy),
            bounded.eigenvalues,
            eigenvectors,
            _edges(adjacency, rows),
            _points(coordinates, rows, raw.shape[-1]),
            policy.gradation,
        )
        if volumes is not None:
            final_complexity = _complexity(_host_spectrum(values)[0], volumes)
    evidence = MetricNormalizationEvidence(
        policy,
        metric.metric_id if isinstance(metric, MeshMetricField) else metric.samples_id,
        symmetrized_count=int(np.count_nonzero(asymmetric)),
        maximum_asymmetry=float(np.max(properties.defect, initial=0.0)),
        projected_eigenvalue_count=int(np.count_nonzero(nonpositive)),
        projected_tensor_count=int(np.count_nonzero(np.any(nonpositive, axis=1))),
        minimum_size_clamped_count=bounded.minimum_size_clamped,
        maximum_size_clamped_count=bounded.maximum_size_clamped,
        anisotropy_clamped_count=bounded.anisotropy_clamped,
        complexity_status=bounded.status,
        complexity_scale=math.exp(bounded.scale_log),
        scaled_complexity=bounded.scaled_complexity,
        final_complexity=final_complexity,
        gradation=gradation,
    )
    if gradation is not None and not gradation.converged:
        raise MetricGradationError(gradation)
    field = MeshMetricField(
        metric.scope,
        values,
        minimum_size=policy.minimum_size,
        maximum_size=policy.maximum_size,
        maximum_anisotropy=policy.maximum_anisotropy,
    )
    return field, evidence


class MetricCombinationEvidence(StrictModule, NonTrainableState):
    """Conflicts between one metric intersection and the combined hard bounds.

    The combined bounds are the most restrictive declared ones: the largest input
    ``minimum_size``, the smallest input ``maximum_size``, and the smallest input
    ``maximum_anisotropy``. ``size_interval_conflict`` reports mutually exclusive
    declared size intervals before a tensor intersection exists. Otherwise,
    ``minimum_size_conflict`` marks rows whose intersection demands sizes below
    ``minimum_size``, ``maximum_size_conflict`` rows allowing sizes above
    ``maximum_size`` (excluded in exact arithmetic because the intersection
    dominates every input), and ``anisotropy_conflict`` rows exceeding
    ``maximum_anisotropy``. All row decisions admit only the eigenvalue roundoff
    certified by the `MeshMetricField` bound check.
    """

    input_ids: tuple[str, ...] = eqx.field(static=True)
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_anisotropy: float = eqx.field(static=True)
    size_interval_conflict: bool = eqx.field(static=True)
    minimum_size_conflict: Array
    maximum_size_conflict: Array
    anisotropy_conflict: Array
    conflict_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_ids: tuple[str, ...],
        minimum_size_conflict: ArrayLike,
        maximum_size_conflict: ArrayLike,
        anisotropy_conflict: ArrayLike,
        /,
        *,
        minimum_size: float,
        maximum_size: float,
        maximum_anisotropy: float,
        size_interval_conflict: bool = False,
    ):
        masks = tuple(
            np.asarray(value, dtype=np.bool_)
            for value in (
                minimum_size_conflict,
                maximum_size_conflict,
                anisotropy_conflict,
            )
        )
        if masks[0].ndim != 1 or any(mask.shape != masks[0].shape for mask in masks):
            raise ValueError("Combination conflicts must be aligned row masks.")
        if not isinstance(size_interval_conflict, (bool, np.bool_)):
            raise TypeError("size_interval_conflict must be bool.")
        self.input_ids = tuple(str(value) for value in input_ids)
        self.minimum_size = _positive(minimum_size, "minimum_size")
        self.maximum_size = _positive(maximum_size, "maximum_size")
        self.maximum_anisotropy = _at_least_one(maximum_anisotropy, "maximum_anisotropy")
        self.size_interval_conflict = bool(size_interval_conflict)
        self.minimum_size_conflict = jnp.asarray(masks[0])
        self.maximum_size_conflict = jnp.asarray(masks[1])
        self.anisotropy_conflict = jnp.asarray(masks[2])
        self.conflict_count = int(self.size_interval_conflict) + int(
            np.count_nonzero(masks[0] | masks[1] | masks[2])
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "metric-combination-evidence",
                "inputs": self.input_ids,
                "minimum_size": self.minimum_size,
                "maximum_size": self.maximum_size,
                "maximum_anisotropy": self.maximum_anisotropy,
                "size_interval_conflict": self.size_interval_conflict,
                "minimum_size_conflict": array_tree_fingerprint(masks[0]),
                "maximum_size_conflict": array_tree_fingerprint(masks[1]),
                "anisotropy_conflict": array_tree_fingerprint(masks[2]),
            }
        )

    @property
    def passed(self) -> bool:
        return self.conflict_count == 0


@final
class MetricCombinationResult(StrictModule, NonTrainableState):
    """Outcome of :func:`combine_mesh_metrics`.

    ``field`` is the combined metric exactly when ``successful``: any hard-bound
    conflict in ``evidence`` withholds it, so a contradictory intersection can
    never be executed by an adaptation route.
    """

    field: MeshMetricField | None
    evidence: MetricCombinationEvidence
    successful: bool = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: MeshMetricField | None,
        evidence: MetricCombinationEvidence,
        /,
    ):
        if field is not None and not isinstance(field, MeshMetricField):
            raise TypeError("field must be MeshMetricField or None.")
        if not isinstance(evidence, MetricCombinationEvidence):
            raise TypeError("evidence must be MetricCombinationEvidence.")
        if (field is not None) != evidence.passed:
            raise ValueError(
                "A combined field exists exactly when the combination has no conflict."
            )
        self.field = field
        self.evidence = evidence
        self.successful = evidence.passed
        self.result_id = canonical_fingerprint(
            {
                "kind": "metric-combination-result",
                "field": None if field is None else field.metric_id,
                "evidence": evidence.evidence_id,
            }
        )


def combine_mesh_metrics(
    metrics: tuple[MeshMetricField, ...], /
) -> MetricCombinationResult:
    """Intersect metrics on one scope by canonical simultaneous reduction.

    The intersection dominates every input (it requests the smallest size in
    every direction), is idempotent, and is independent of input order. Declared
    bounds combine to the most restrictive interval. An empty hard size interval,
    sizes below the combined minimum, or anisotropy beyond the combined bound are
    explicit conflicts: the result retains evidence but withholds its executable
    field.
    """
    fields = tuple(metrics)
    if not fields or not all(isinstance(field, MeshMetricField) for field in fields):
        raise TypeError("metrics must be a non-empty tuple of MeshMetricField.")
    scope = fields[0].scope
    if any(field.scope.scope_id != scope.scope_id for field in fields):
        raise ValueError("Combined metrics must share one exact scope.")
    if len({field.values.shape for field in fields}) != 1:
        raise ValueError("Combined metrics must share one tensor dimension.")
    minimum = max(field.minimum_size for field in fields)
    maximum = min(field.maximum_size for field in fields)
    anisotropy = min(field.maximum_anisotropy for field in fields)
    input_ids = tuple(sorted(field.metric_id for field in fields))
    if minimum > maximum:
        empty = np.zeros((scope.entity_ids.shape[0],), dtype=np.bool_)
        evidence = MetricCombinationEvidence(
            input_ids,
            empty,
            empty,
            empty,
            minimum_size=minimum,
            maximum_size=maximum,
            maximum_anisotropy=anisotropy,
            size_interval_conflict=True,
        )
        return MetricCombinationResult(None, evidence)
    candidates = jnp.stack(tuple(field.values for field in fields), axis=1)
    combined = np.asarray(
        _canonical_intersection_kernel(
            candidates, jnp.ones(candidates.shape[:2], dtype=jnp.bool_)
        )
    )
    evidence = MetricCombinationEvidence(
        input_ids,
        *_bound_violations(combined, minimum, maximum, anisotropy),
        minimum_size=minimum,
        maximum_size=maximum,
        maximum_anisotropy=anisotropy,
    )
    field = (
        MeshMetricField(
            scope,
            combined,
            minimum_size=minimum,
            maximum_size=maximum,
            maximum_anisotropy=anisotropy,
        )
        if evidence.passed
        else None
    )
    return MetricCombinationResult(field, evidence)


@jax.jit
def _log_euclidean_mean(tensors: Array, weights: Array, /) -> Array:
    logarithm = hermitian_log(tensors)
    mean = contract("...k,...kij->...ij", weights, logarithm.value)
    result = hermitian_exp(mean)
    total = jnp.sum(weights, axis=-1)
    tolerance = 64.0 * jnp.finfo(weights.dtype).eps * weights.shape[-1]
    valid = (
        jnp.all(logarithm.valid)
        & jnp.all(result.valid)
        & jnp.all(weights >= 0.0)
        & jnp.all(jnp.abs(total - 1.0) <= tolerance)
    )
    return eqx.error_if(
        result.value,
        ~valid,
        "Log-Euclidean interpolation requires SPD metrics and convex weights.",
    )


def interpolate_mesh_metric(values: ArrayLike, weights: ArrayLike, /) -> Array:
    """Log-Euclidean interpolation ``exp(sum_k w_k log M_k)`` (Arsigny et al. 2006).

    ``values`` has shape ``(..., k, d, d)`` and ``weights`` shape ``(..., k)``
    with non-negative weights summing to one. The result is SPD, preserves
    determinant interpolation ``log det = sum_k w_k log det M_k``, and is
    invariant under reordering of the ``k`` samples.
    """
    tensors = jnp.asarray(values)
    if not jnp.issubdtype(tensors.dtype, jnp.floating):
        tensors = tensors.astype(jnp.float64)
    coefficients = jnp.asarray(weights, dtype=tensors.dtype)
    if (
        tensors.ndim < 3
        or tensors.shape[-1] != tensors.shape[-2]
        or coefficients.shape != tensors.shape[:-2]
    ):
        raise ValueError("values must be (..., k, d, d) and weights (..., k).")
    return _log_euclidean_mean(tensors, coefficients)


@jax.jit
def _edge_lengths(tensors: Array, points: Array, edges: Array, /) -> Array:
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    first = jnp.sqrt(contract("ei,eij,ej->e", delta, tensors[edges[:, 0]], delta))
    second = jnp.sqrt(contract("ei,eij,ej->e", delta, tensors[edges[:, 1]], delta))
    ratio = second / jnp.where(first > 0.0, first, 1.0)
    nearly_equal = jnp.abs(ratio - 1.0) <= 1.0e-6
    safe_ratio = jnp.where(nearly_equal, 2.0, ratio)
    # Log-mean of endpoint lengths: exact for geometric size variation along pq.
    logarithmic = (first - second) / jnp.log(1.0 / safe_ratio)
    return jnp.where(nearly_equal, 0.5 * (first + second), logarithmic)


def metric_edge_lengths(
    values: ArrayLike, coordinates: ArrayLike, edges: ArrayLike, /
) -> Array:
    """Riemannian edge lengths ``(l_p - l_q) / log(l_p / l_q)`` (Alauzet 2010).

    ``l_p`` is the length of ``pq`` in the endpoint metric ``M_p``; the log-mean is
    the exact length when the size varies geometrically along the edge.
    """
    tensors = jnp.asarray(values)
    points = jnp.asarray(coordinates, dtype=tensors.dtype)
    pairs = jnp.asarray(edges)
    if (
        tensors.ndim != 3
        or tensors.shape[1:] != (points.shape[-1], points.shape[-1])
        or points.shape != (tensors.shape[0], tensors.shape[-1])
        or pairs.ndim != 2
        or pairs.shape[1] != 2
        or not jnp.issubdtype(pairs.dtype, jnp.integer)
    ):
        raise ValueError(
            "values (n, d, d), coordinates (n, d), and integer edges (e, 2) are required."
        )
    return _edge_lengths(tensors, points, pairs)


class HessianMetricEvidence(StrictModule, NonTrainableState):
    """Spectral handling of one Hessian field before Lp normalization.

    ``|H|`` is formed by eigenvalue absolute values. Tensors whose largest
    absolute eigenvalue is below ``relative_zero_tolerance`` times the global
    largest one carry no interpolation-error information and receive the maximum
    size; other eigenvalues are floored at ``lambda_max / maximum_anisotropy**2``
    per tensor so the determinant is positive.
    """

    p: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    negative_eigenvalue_count: int = eqx.field(static=True)
    indefinite_tensor_count: int = eqx.field(static=True)
    zero_tensor_count: int = eqx.field(static=True)
    floored_eigenvalue_count: int = eqx.field(static=True)
    normalization: MetricNormalizationEvidence
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        normalization: MetricNormalizationEvidence,
        /,
        *,
        p: float,
        exponent: float,
        negative_eigenvalue_count: int,
        indefinite_tensor_count: int,
        zero_tensor_count: int,
        floored_eigenvalue_count: int,
    ):
        if not isinstance(normalization, MetricNormalizationEvidence):
            raise TypeError("normalization must be MetricNormalizationEvidence.")
        self.p = float(p)
        self.exponent = float(exponent)
        self.negative_eigenvalue_count = int(negative_eigenvalue_count)
        self.indefinite_tensor_count = int(indefinite_tensor_count)
        self.zero_tensor_count = int(zero_tensor_count)
        self.floored_eigenvalue_count = int(floored_eigenvalue_count)
        self.normalization = normalization
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "hessian-metric-evidence",
                "p": "inf" if math.isinf(self.p) else self.p,
                "exponent": self.exponent,
                "negative": self.negative_eigenvalue_count,
                "indefinite": self.indefinite_tensor_count,
                "zero": self.zero_tensor_count,
                "floored": self.floored_eigenvalue_count,
                "normalization": normalization.evidence_id,
            }
        )

    @property
    def passed(self) -> bool:
        return self.normalization.passed


def lp_metric_from_hessian(
    hessian: ArrayLike,
    /,
    *,
    p: float,
    target_complexity: float,
    minimum_size: float,
    maximum_size: float,
    maximum_anisotropy: float,
    vertex_volumes: ArrayLike,
    relative_zero_tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, HessianMetricEvidence]:
    """Lp-optimal continuous metric of Loseille and Alauzet (2011).

    ``M = D det(|H|)**(-1/(2p+d)) |H|`` with ``D`` the global constant meeting
    ``target_complexity = sum_i V_i sqrt(det M_i)``. The constant is solved after
    size and anisotropy bounds (native bracketed root), so the returned metric
    meets the complexity whenever the bounds admit it; ``p=inf`` gives the
    ``L_inf`` metric ``D |H|``. Returns row-aligned metric tensors and evidence.
    """
    tensors = np.asarray(hessian, dtype=np.float64)
    if (
        tensors.ndim != 3
        or tensors.shape[1] != tensors.shape[2]
        or tensors.shape[1] not in (1, 2, 3)
        or tensors.shape[0] == 0
        or not np.all(np.isfinite(tensors))
    ):
        raise ValueError("hessian must be finite (n, d, d) with d in 1-3.")
    order = finite_real_scalar(p, "p") if not math.isinf(float(p)) else math.inf
    if order < 1.0:
        raise ValueError("p must be at least one.")
    zero_tolerance = finite_real_scalar(
        relative_zero_tolerance, "relative_zero_tolerance"
    )
    if zero_tolerance < 0.0:
        raise ValueError("relative_zero_tolerance must be non-negative.")
    policy = MetricNormalizationPolicy(
        minimum_size=minimum_size,
        maximum_size=maximum_size,
        maximum_anisotropy=maximum_anisotropy,
        target_complexity=target_complexity,
    )
    rows, dimension = tensors.shape[0], tensors.shape[-1]
    volumes = _vertex_volumes(vertex_volumes, rows)
    properties = _tensor_properties(tensors)
    if not np.all(properties.hermitian):
        raise ValueError(
            "Hessians must be symmetric; symmetrize recovered Hessians explicitly."
        )
    eigenvalues, eigenvectors = _host_spectrum(properties.symmetric)
    tolerance = properties.tolerance[:, None]
    negative = eigenvalues < -tolerance
    indefinite = np.any(negative, axis=1) & np.any(eigenvalues > tolerance, axis=1)
    absolute = np.abs(eigenvalues)
    local_maximum = np.max(absolute, axis=1)
    zero = local_maximum <= zero_tolerance * float(np.max(local_maximum))
    floor = local_maximum[:, None] / policy.maximum_anisotropy**2
    floored = ~zero[:, None] & (absolute < floor)
    absolute = np.where(zero[:, None], 1.0, np.maximum(absolute, floor))
    exponent = 0.0 if math.isinf(order) else -1.0 / (2.0 * order + dimension)
    scale = np.exp(exponent * np.sum(np.log(absolute), axis=1))
    raw = np.where(zero[:, None], 0.0, scale[:, None] * absolute)
    eigenvectors = np.where(zero[:, None, None], np.eye(dimension), eigenvectors)
    bounded = _bound_spectrum(
        raw,
        minimum_size=policy.minimum_size,
        maximum_size=policy.maximum_size,
        anisotropy=policy.maximum_anisotropy,
        target_complexity=policy.target_complexity,
        volumes=volumes,
    )
    normalization = MetricNormalizationEvidence(
        policy,
        canonical_fingerprint(
            {"kind": "hessian-field", "values": array_tree_fingerprint(tensors)}
        ),
        symmetrized_count=0,
        maximum_asymmetry=float(np.max(properties.defect, initial=0.0)),
        projected_eigenvalue_count=0,
        projected_tensor_count=0,
        minimum_size_clamped_count=bounded.minimum_size_clamped,
        maximum_size_clamped_count=bounded.maximum_size_clamped,
        anisotropy_clamped_count=bounded.anisotropy_clamped,
        complexity_status=bounded.status,
        complexity_scale=math.exp(bounded.scale_log),
        scaled_complexity=bounded.scaled_complexity,
        final_complexity=bounded.scaled_complexity,
        gradation=None,
    )
    evidence = HessianMetricEvidence(
        normalization,
        p=order,
        exponent=exponent,
        negative_eigenvalue_count=int(np.count_nonzero(negative)),
        indefinite_tensor_count=int(np.count_nonzero(indefinite)),
        zero_tensor_count=int(np.count_nonzero(zero)),
        floored_eigenvalue_count=int(np.count_nonzero(floored)),
    )
    return _compose(bounded.eigenvalues, eigenvectors), evidence


__all__ = [
    "HessianMetricEvidence",
    "MeshMetricField",
    "MeshMetricSamples",
    "MetricCombinationEvidence",
    "MetricCombinationResult",
    "MetricComplexityStatus",
    "MetricGradationError",
    "MetricGradationEvidence",
    "MetricGradationKind",
    "MetricGradationPolicy",
    "MetricGradationStatus",
    "MetricNormalizationEvidence",
    "MetricNormalizationPolicy",
    "combine_mesh_metrics",
    "grade_mesh_metric",
    "interpolate_mesh_metric",
    "lp_metric_from_hessian",
    "metric_edge_lengths",
    "normalize_mesh_metric",
]
