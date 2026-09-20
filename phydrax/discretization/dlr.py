#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generated discrete-Lehmann representations for finite-temperature kernels.

Basis generation is a preparation operation.  It samples the requested kernel on
native Gauss--Legendre designs, performs deterministic rank-revealing skeleton
selection, and prepares fixed-capacity interpolation matrices.  Runtime kernel
evaluation is array-only and JAX compatible.  Bosonic bases use the regularized
kernel whose coefficients represent ``rho(omega) / omega``; this removes the
removable zero-frequency singularity without changing physical Green functions.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..integration import GaussLegendreRule
from ..linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    RankPolicy,
)


ThermalStatistics: TypeAlias = Literal["fermionic", "bosonic"]


class DLRBasisStatus(IntEnum):
    """Terminal evidence for one generated basis."""

    SUCCESS = 0
    RANK_LIMIT_REACHED = 1
    NONFINITE_KERNEL = 2
    ILL_CONDITIONED = 3


class DLRBasisPolicy(StrictModule):
    """Accuracy and fixed-shape resource contract for DLR generation."""

    tolerance: float = eqx.field(static=True)
    maximum_rank: int = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        tolerance: float = 1e-10,
        maximum_rank: int = 64,
        candidate_count: int = 192,
        maximum_bytes: int = 256 * 1024**2,
        condition_limit: float = 1e14,
    ):
        tolerance_ = float(tolerance)
        condition_ = float(condition_limit)
        if not isfinite(tolerance_) or tolerance_ <= 0.0 or tolerance_ >= 1.0:
            raise ValueError(
                "tolerance must be finite and lie strictly between zero and one."
            )
        if not isfinite(condition_) or condition_ <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        for name, value, lower in (
            ("maximum_rank", maximum_rank, 1),
            ("candidate_count", candidate_count, 4),
            ("maximum_bytes", maximum_bytes, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < lower:
                raise ValueError(f"{name} must be at least {lower}.")
        if int(maximum_rank) > int(candidate_count):
            raise ValueError("maximum_rank must not exceed candidate_count.")
        self.tolerance = tolerance_
        self.maximum_rank = int(maximum_rank)
        self.candidate_count = int(candidate_count)
        self.maximum_bytes = int(maximum_bytes)
        self.condition_limit = condition_


class DLRBasisCostEstimate(StrictModule):
    """Dense preparation and persistent storage estimate."""

    candidate_kernel_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    persistent_bytes: int = eqx.field(static=True)


class DLRBasisPlan(StrictModule):
    """Immutable structure-only DLR generation plan."""

    beta: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    statistics: ThermalStatistics = eqx.field(static=True)
    policy: DLRBasisPolicy
    cost: DLRBasisCostEstimate
    plan_id: str = eqx.field(static=True)


class DLRBasisEvidence(StrictModule):
    """Rank, approximation, conditioning, and resource evidence."""

    singular_values: Array
    active: Array
    numerical_rank: Array
    requested_tolerance: Array
    achieved_tolerance: Array
    tau_interpolation_residual: Array
    matsubara_interpolation_residual: Array
    condition_estimate: Array
    finite: Array
    saturated: Array
    valid: Array
    status: Array
    spectral_cutoff: float = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)
    maximum_rank: int = eqx.field(static=True)


class PreparedDLRBasis(StrictModule):
    """Fixed-capacity generated basis and its prepared interpolation designs."""

    plan: DLRBasisPlan
    frequencies: Array
    tau_nodes: Array
    matsubara_indices: Array
    matsubara_frequencies: Array
    tau_matrix: Array
    matsubara_matrix: Array
    evidence: DLRBasisEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def beta(self) -> float:
        return self.plan.beta

    @property
    def cutoff(self) -> float:
        return self.plan.cutoff

    @property
    def statistics(self) -> ThermalStatistics:
        return self.plan.statistics

    @property
    def rank(self) -> Array:
        return self.evidence.numerical_rank

    @property
    def active(self) -> Array:
        return self.evidence.active

    @property
    def valid(self) -> Array:
        return self.evidence.valid

    def tau_kernel(self, tau: ArrayLike, /) -> Array:
        return (
            thermal_tau_kernel(
                tau,
                self.frequencies,
                beta=self.beta,
                statistics=self.statistics,
            )
            * self.active
        )

    def matsubara_kernel(self, indices: ArrayLike, /) -> Array:
        return (
            thermal_matsubara_kernel(
                indices,
                self.frequencies,
                beta=self.beta,
                statistics=self.statistics,
            )
            * self.active
        )


class DLRTransformEvidence(StrictModule):
    """Evidence returned by a sample-to-DLR coefficient solve."""

    residual_norm: Array
    relative_residual: Array
    rank: Array
    condition_estimate: Array
    finite: Array
    valid: Array
    status: Array


class DLRTransformResult(StrictModule):
    """Fixed-capacity coefficients and their solve evidence."""

    coefficients: Array
    evidence: DLRTransformEvidence


def _statistics(value: str, /) -> ThermalStatistics:
    if value not in ("fermionic", "bosonic"):
        raise ValueError("statistics must be 'fermionic' or 'bosonic'.")
    return value


def _positive_finite(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def thermal_tau_kernel(
    tau: ArrayLike,
    frequencies: ArrayLike,
    /,
    *,
    beta: float,
    statistics: ThermalStatistics = "fermionic",
) -> Array:
    """Evaluate stable fermionic or regularized bosonic Lehmann kernels.

    The returned shape is ``tau.shape + frequencies.shape``.  Fermionic columns
    are ``-exp(-tau*w)/(1 + exp(-beta*w))``.  Bosonic columns are the regularized
    ``-w*exp(-tau*w)/(1 - exp(-beta*w))``, with the continuous value ``-1/beta``
    at zero frequency.
    """

    beta_ = _positive_finite(beta, "beta")
    statistics_ = _statistics(statistics)
    tau_ = jnp.asarray(tau)
    omega = jnp.asarray(frequencies)
    if not jnp.issubdtype(tau_.dtype, jnp.inexact):
        tau_ = tau_.astype("float64")
    if not jnp.issubdtype(omega.dtype, jnp.inexact):
        omega = omega.astype("float64")
    dtype = jnp.result_type(tau_, omega)
    tau_ndim = tau_.ndim
    omega_ndim = omega.ndim
    tau_ = tau_.astype(dtype).reshape(tau_.shape + (1,) * omega_ndim)
    omega = omega.astype(dtype).reshape((1,) * tau_ndim + omega.shape)
    if statistics_ == "fermionic":
        positive = -jnp.exp(-tau_ * omega) / (1.0 + jnp.exp(-beta_ * omega))
        negative = -jnp.exp((beta_ - tau_) * omega) / (1.0 + jnp.exp(beta_ * omega))
        return jnp.where(omega >= 0.0, positive, negative)
    absolute = jnp.abs(omega)
    denominator = -jnp.expm1(-beta_ * absolute)
    safe_denominator = jnp.where(absolute > 0.0, denominator, 1.0)
    positive = -omega * jnp.exp(-tau_ * omega) / safe_denominator
    negative = -omega * jnp.exp((beta_ - tau_) * omega) / safe_denominator
    regular = jnp.where(omega >= 0.0, positive, negative)
    return jnp.where(absolute > 0.0, regular, -jnp.ones_like(regular) / beta_)


def matsubara_frequencies(
    indices: ArrayLike,
    /,
    *,
    beta: float,
    statistics: ThermalStatistics = "fermionic",
) -> Array:
    """Map integer labels to physical Matsubara angular frequencies."""

    beta_ = _positive_finite(beta, "beta")
    statistics_ = _statistics(statistics)
    labels = jnp.asarray(indices)
    shift = 1 if statistics_ == "fermionic" else 0
    return (2 * labels + shift) * jnp.pi / beta_


def thermal_matsubara_kernel(
    indices: ArrayLike,
    frequencies: ArrayLike,
    /,
    *,
    beta: float,
    statistics: ThermalStatistics = "fermionic",
) -> Array:
    """Evaluate the Matsubara transform of :func:`thermal_tau_kernel`."""

    statistics_ = _statistics(statistics)
    nu = matsubara_frequencies(indices, beta=beta, statistics=statistics_)
    omega = jnp.asarray(frequencies)
    dtype = jnp.result_type(nu, omega, 1j)
    nu_ndim = nu.ndim
    omega_ndim = omega.ndim
    nu = nu.astype(dtype).reshape(nu.shape + (1,) * omega_ndim)
    omega = omega.astype(dtype).reshape((1,) * nu_ndim + omega.shape)
    denominator = 1j * nu - omega
    if statistics_ == "fermionic":
        return jnp.reciprocal(denominator)
    nonzero = omega != 0.0
    safe = jnp.where(nonzero, denominator, 1.0 + 0.0j)
    regular = omega / safe
    zero_mode_limit = jnp.where(nu == 0.0, -jnp.ones_like(regular), 0.0)
    return jnp.where(nonzero, regular, zero_mode_limit)


def plan_dlr_basis(
    beta: float,
    cutoff: float,
    /,
    *,
    statistics: ThermalStatistics = "fermionic",
    policy: DLRBasisPolicy | None = None,
    tolerance: float | None = None,
    maximum_rank: int | None = None,
    candidate_count: int | None = None,
    maximum_bytes: int | None = None,
) -> DLRBasisPlan:
    """Plan bounded generation without allocating a candidate kernel."""

    beta_ = _positive_finite(beta, "beta")
    cutoff_ = _positive_finite(cutoff, "cutoff")
    statistics_ = _statistics(statistics)
    if policy is not None and any(
        value is not None
        for value in (tolerance, maximum_rank, candidate_count, maximum_bytes)
    ):
        raise ValueError("An explicit policy cannot be combined with policy overrides.")
    if policy is None:
        policy_ = DLRBasisPolicy(
            tolerance=1e-10 if tolerance is None else tolerance,
            maximum_rank=64 if maximum_rank is None else maximum_rank,
            candidate_count=192 if candidate_count is None else candidate_count,
            maximum_bytes=256 * 1024**2 if maximum_bytes is None else maximum_bytes,
        )
    else:
        if not isinstance(policy, DLRBasisPolicy):
            raise TypeError("policy must be a DLRBasisPolicy or None.")
        policy_ = policy
    count = policy_.candidate_count
    rank = policy_.maximum_rank
    itemsize = np.dtype(np.float64).itemsize
    candidate_bytes = itemsize * count * count
    workspace_bytes = itemsize * (4 * count * count + 4 * count * rank)
    persistent_bytes = itemsize * (2 * count + 4 * rank * rank + 8 * rank)
    if candidate_bytes + workspace_bytes + persistent_bytes > policy_.maximum_bytes:
        raise ValueError("DLR generation exceeds maximum_bytes before allocation.")
    cost = DLRBasisCostEstimate(
        candidate_bytes,
        workspace_bytes,
        persistent_bytes,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "generated-dlr-basis-plan",
            "beta": beta_,
            "cutoff": cutoff_,
            "statistics": statistics_,
            "tolerance": policy_.tolerance,
            "maximum_rank": rank,
            "candidate_count": count,
            "maximum_bytes": policy_.maximum_bytes,
            "condition_limit": policy_.condition_limit,
            "cost": {
                "candidate_kernel_bytes": candidate_bytes,
                "preparation_workspace_bytes": workspace_bytes,
                "persistent_bytes": persistent_bytes,
            },
        }
    )
    return DLRBasisPlan(beta_, cutoff_, statistics_, policy_, cost, plan_id)


def _greedy_columns(matrix: np.ndarray, capacity: int, tolerance: float) -> list[int]:
    residual = np.array(matrix, copy=True)
    selected: list[int] = []
    initial = 0.0
    for _ in range(capacity):
        norms = np.sqrt(np.sum(np.abs(residual) ** 2, axis=0))
        index = int(np.argmax(norms))
        magnitude = float(norms[index])
        if not selected:
            initial = magnitude
        if selected and magnitude <= tolerance * max(initial, np.finfo(np.float64).tiny):
            break
        if not np.isfinite(magnitude) or magnitude == 0.0:
            break
        selected.append(index)
        vector = residual[:, index] / magnitude
        projection = np.conj(vector) @ residual
        residual = residual - vector[:, None] * projection[None, :]
        correction = np.conj(vector) @ residual
        residual = residual - vector[:, None] * correction[None, :]
    return selected


def _native_factor(matrix: Array, tolerance: float, /):
    cutoff = max(64.0 * np.finfo(np.dtype(matrix.real.dtype)).eps, tolerance * 1e-4)
    return factorize(
        DenseLinearOperator(matrix),
        FactorizationPolicy("svd", rank=RankPolicy(relative_cutoff=cutoff)),
    )


def _fixed_square(matrix: Array, capacity: int, rank: int, /) -> Array:
    dtype = matrix.dtype
    result = jnp.eye(capacity, dtype=dtype)
    if rank == 0:
        return result
    return result.at[:rank, :rank].set(matrix)


def _candidate_design(plan: DLRBasisPlan, /) -> tuple[Array, Array, Array]:
    data = GaussLegendreRule(plan.policy.candidate_count).data()
    reference = jnp.asarray(data.nodes)
    tau = 0.5 * plan.beta * (reference + 1.0)
    omega = plan.cutoff * reference
    matsubara_count = plan.policy.candidate_count
    lower = -(matsubara_count // 2)
    labels = jnp.arange(lower, lower + matsubara_count, dtype=jnp.int32)
    return tau, omega, labels


def prepare_dlr_basis(plan: DLRBasisPlan, /) -> PreparedDLRBasis:
    """Generate and prepare one content-addressed fixed-capacity DLR basis."""

    if not isinstance(plan, DLRBasisPlan):
        raise TypeError("plan must be a DLRBasisPlan.")
    tau_candidates, omega_candidates, matsubara_candidates = _candidate_design(plan)
    kernel = thermal_tau_kernel(
        tau_candidates,
        omega_candidates,
        beta=plan.beta,
        statistics=plan.statistics,
    )
    finite_kernel = bool(np.all(np.isfinite(np.asarray(kernel))))
    selected_columns = (
        _greedy_columns(
            np.asarray(kernel),
            plan.policy.maximum_rank,
            plan.policy.tolerance * 0.25,
        )
        if finite_kernel
        else []
    )
    if not selected_columns:
        selected_columns = [0]
    selected_kernel = np.asarray(kernel)[:, selected_columns]
    selected_tau = _greedy_columns(
        selected_kernel.T,
        len(selected_columns),
        max(plan.policy.tolerance * 1e-2, np.finfo(np.float64).eps),
    )
    rank = min(len(selected_columns), len(selected_tau))
    selected_columns = selected_columns[:rank]
    selected_tau = selected_tau[:rank]
    frequencies_active = omega_candidates[jnp.asarray(selected_columns, dtype=jnp.int32)]
    tau_active = tau_candidates[jnp.asarray(selected_tau, dtype=jnp.int32)]

    matsubara_full = thermal_matsubara_kernel(
        matsubara_candidates,
        frequencies_active,
        beta=plan.beta,
        statistics=plan.statistics,
    )
    selected_matsubara = _greedy_columns(
        np.asarray(matsubara_full).T,
        rank,
        max(plan.policy.tolerance * 1e-2, np.finfo(np.float64).eps),
    )
    if len(selected_matsubara) < rank:
        unused = [
            index
            for index in range(plan.policy.candidate_count)
            if index not in selected_matsubara
        ]
        selected_matsubara.extend(unused[: rank - len(selected_matsubara)])
    matsubara_active = matsubara_candidates[
        jnp.asarray(selected_matsubara[:rank], dtype=jnp.int32)
    ]

    tau_block = thermal_tau_kernel(
        tau_active,
        frequencies_active,
        beta=plan.beta,
        statistics=plan.statistics,
    )
    matsubara_block = thermal_matsubara_kernel(
        matsubara_active,
        frequencies_active,
        beta=plan.beta,
        statistics=plan.statistics,
    )
    tau_factor = _native_factor(tau_block, plan.policy.tolerance)
    matsubara_factor = _native_factor(matsubara_block, plan.policy.tolerance)
    interpolation = tau_factor.solve(kernel[jnp.asarray(selected_tau), :]).value
    reconstruction = kernel[:, jnp.asarray(selected_columns)] @ interpolation
    kernel_scale = jnp.maximum(jnp.max(jnp.abs(kernel)), jnp.finfo(kernel.dtype).tiny)
    achieved = jnp.max(jnp.abs(kernel - reconstruction)) / kernel_scale
    tau_identity = jnp.eye(rank, dtype=tau_block.dtype)
    matsubara_identity = jnp.eye(rank, dtype=matsubara_block.dtype)
    tau_inverse_residual = jnp.max(
        jnp.abs(tau_block @ tau_factor.solve(tau_identity).value - tau_identity)
    )
    matsubara_inverse_residual = jnp.max(
        jnp.abs(
            matsubara_block @ matsubara_factor.solve(matsubara_identity).value
            - matsubara_identity
        )
    )
    candidate_factor = _native_factor(kernel, plan.policy.tolerance)
    singular_values = candidate_factor.singular_values()
    leading = jnp.maximum(singular_values[0], jnp.finfo(singular_values.dtype).tiny)
    requested_rank = jnp.sum(singular_values > plan.policy.tolerance * leading)
    selected_singular_values = tau_factor.singular_values()
    matsubara_singular_values = matsubara_factor.singular_values()
    tau_condition = selected_singular_values[0] / jnp.maximum(
        selected_singular_values[-1],
        jnp.finfo(selected_singular_values.dtype).tiny,
    )
    matsubara_condition = matsubara_singular_values[0] / jnp.maximum(
        matsubara_singular_values[-1],
        jnp.finfo(matsubara_singular_values.dtype).tiny,
    )
    condition = jnp.maximum(tau_condition, matsubara_condition)

    capacity = plan.policy.maximum_rank
    active = jnp.arange(capacity) < rank
    frequencies = jnp.zeros((capacity,), dtype=frequencies_active.dtype)
    frequencies = frequencies.at[:rank].set(frequencies_active)
    tau_nodes = jnp.zeros((capacity,), dtype=tau_active.dtype)
    tau_nodes = tau_nodes.at[:rank].set(tau_active)
    matsubara_indices_ = jnp.zeros((capacity,), dtype=jnp.int32)
    matsubara_indices_ = matsubara_indices_.at[:rank].set(matsubara_active)
    matsubara_values = matsubara_frequencies(
        matsubara_indices_, beta=plan.beta, statistics=plan.statistics
    )
    tau_matrix = _fixed_square(tau_block, capacity, rank)
    matsubara_matrix = _fixed_square(matsubara_block, capacity, rank)
    finite = (
        jnp.asarray(finite_kernel)
        & jnp.all(jnp.isfinite(tau_matrix))
        & jnp.all(jnp.isfinite(matsubara_matrix))
        & jnp.isfinite(achieved)
        & jnp.isfinite(condition)
    )
    saturated = (requested_rank > capacity) | (achieved > plan.policy.tolerance)
    conditioned = condition <= plan.policy.condition_limit
    valid = finite & ~saturated & conditioned & (achieved <= plan.policy.tolerance)
    status = jnp.where(
        ~finite,
        int(DLRBasisStatus.NONFINITE_KERNEL),
        jnp.where(
            saturated,
            int(DLRBasisStatus.RANK_LIMIT_REACHED),
            jnp.where(
                ~conditioned,
                int(DLRBasisStatus.ILL_CONDITIONED),
                int(DLRBasisStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = DLRBasisEvidence(
        singular_values=singular_values,
        active=active,
        numerical_rank=jnp.asarray(rank, dtype=jnp.int32),
        requested_tolerance=jnp.asarray(plan.policy.tolerance, dtype=achieved.dtype),
        achieved_tolerance=achieved,
        tau_interpolation_residual=tau_inverse_residual,
        matsubara_interpolation_residual=matsubara_inverse_residual,
        condition_estimate=condition,
        finite=finite,
        saturated=saturated,
        valid=valid,
        status=status,
        spectral_cutoff=plan.cutoff,
        candidate_count=plan.policy.candidate_count,
        maximum_rank=capacity,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-generated-dlr-basis",
            "plan": plan.plan_id,
            "nodes": array_tree_fingerprint(
                {
                    "frequencies": frequencies,
                    "tau_nodes": tau_nodes,
                    "matsubara_indices": matsubara_indices_,
                    "active": active,
                }
            ),
        }
    )
    return PreparedDLRBasis(
        plan=plan,
        frequencies=frequencies,
        tau_nodes=tau_nodes,
        matsubara_indices=matsubara_indices_,
        matsubara_frequencies=matsubara_values,
        tau_matrix=tau_matrix,
        matsubara_matrix=matsubara_matrix,
        evidence=evidence,
        prepared_id=prepared_id,
    )


def generate_dlr_basis(
    beta: float,
    cutoff: float,
    /,
    *,
    statistics: ThermalStatistics = "fermionic",
    policy: DLRBasisPolicy | None = None,
    tolerance: float | None = None,
    maximum_rank: int | None = None,
    candidate_count: int | None = None,
    maximum_bytes: int | None = None,
) -> PreparedDLRBasis:
    """Plan and prepare a generated DLR basis."""

    plan = plan_dlr_basis(
        beta,
        cutoff,
        statistics=statistics,
        policy=policy,
        tolerance=tolerance,
        maximum_rank=maximum_rank,
        candidate_count=candidate_count,
        maximum_bytes=maximum_bytes,
    )
    return prepare_dlr_basis(plan)


def _fit(
    design: Array,
    values: ArrayLike,
    basis: PreparedDLRBasis,
    /,
    *,
    tolerance: float | None,
    sample_active: ArrayLike | None,
) -> DLRTransformResult:
    data = jnp.asarray(values)
    if data.ndim < 1 or data.shape[0] != design.shape[0]:
        raise ValueError("values must have one leading entry per sample.")
    if sample_active is None:
        active = jnp.ones((design.shape[0],), dtype=jnp.bool_)
    else:
        active = jnp.asarray(sample_active, dtype=jnp.bool_)
        if active.shape != (design.shape[0],):
            raise ValueError("sample_active must have one entry per sample.")
    dtype = jnp.result_type(design, data)
    row_mask = active.astype(dtype)
    design_ = design.astype(dtype) * row_mask[:, None]
    payload_mask = row_mask.reshape(row_mask.shape + (1,) * (data.ndim - 1))
    data = data.astype(dtype) * payload_mask
    threshold = basis.plan.policy.tolerance if tolerance is None else float(tolerance)
    if not isfinite(threshold) or threshold <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    factor = _native_factor(design_, threshold)
    coefficients = factor.solve(data).value
    predicted = design_ @ coefficients.reshape((coefficients.shape[0], -1))
    residual = predicted - data.reshape((data.shape[0], -1))
    residual_norm = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2))
    data_norm = jnp.sqrt(jnp.sum(jnp.abs(data) ** 2))
    relative = residual_norm / jnp.maximum(data_norm, jnp.finfo(residual_norm.dtype).tiny)
    singular_values = factor.singular_values()
    rank = factor.rank()
    nonzero = singular_values > threshold * singular_values[0]
    last = jnp.maximum(jnp.sum(nonzero) - 1, 0)
    condition = singular_values[0] / jnp.maximum(
        singular_values[last], jnp.finfo(singular_values.dtype).tiny
    )
    finite = jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(relative)
    valid = finite & basis.valid & (relative <= threshold)
    status = jnp.where(
        ~finite,
        int(DLRBasisStatus.NONFINITE_KERNEL),
        jnp.where(
            valid,
            int(DLRBasisStatus.SUCCESS),
            int(DLRBasisStatus.ILL_CONDITIONED),
        ),
    ).astype(jnp.int32)
    evidence = DLRTransformEvidence(
        residual_norm,
        relative,
        rank,
        condition,
        finite,
        valid,
        status,
    )
    return DLRTransformResult(coefficients, evidence)


def fit_dlr_from_tau(
    basis: PreparedDLRBasis,
    tau: ArrayLike,
    values: ArrayLike,
    /,
    *,
    tolerance: float | None = None,
    sample_active: ArrayLike | None = None,
) -> DLRTransformResult:
    """Fit fixed-capacity DLR coefficients to imaginary-time samples."""

    if not isinstance(basis, PreparedDLRBasis):
        raise TypeError("basis must be a PreparedDLRBasis.")
    design = basis.tau_kernel(tau)
    return _fit(
        design,
        values,
        basis,
        tolerance=tolerance,
        sample_active=sample_active,
    )


def fit_dlr_from_matsubara(
    basis: PreparedDLRBasis,
    indices: ArrayLike,
    values: ArrayLike,
    /,
    *,
    tolerance: float | None = None,
    sample_active: ArrayLike | None = None,
) -> DLRTransformResult:
    """Fit fixed-capacity DLR coefficients to Matsubara samples."""

    if not isinstance(basis, PreparedDLRBasis):
        raise TypeError("basis must be a PreparedDLRBasis.")
    design = basis.matsubara_kernel(indices)
    return _fit(
        design,
        values,
        basis,
        tolerance=tolerance,
        sample_active=sample_active,
    )


__all__ = [
    "DLRBasisCostEstimate",
    "DLRBasisEvidence",
    "DLRBasisPlan",
    "DLRBasisPolicy",
    "DLRBasisStatus",
    "DLRTransformEvidence",
    "DLRTransformResult",
    "PreparedDLRBasis",
    "ThermalStatistics",
    "fit_dlr_from_matsubara",
    "fit_dlr_from_tau",
    "generate_dlr_basis",
    "matsubara_frequencies",
    "plan_dlr_basis",
    "prepare_dlr_basis",
    "thermal_matsubara_kernel",
    "thermal_tau_kernel",
]
