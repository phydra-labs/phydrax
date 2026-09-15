#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-temperature Green functions, DLR transforms, and exact Lehmann sums."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from numbers import Integral
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization.dlr import (
    DLRTransformEvidence,
    fit_dlr_from_matsubara,
    fit_dlr_from_tau,
    matsubara_frequencies,
    PreparedDLRBasis,
    thermal_tau_kernel,
    ThermalStatistics,
)
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    HermitianSpectrum,
    RankPolicy,
)


class GreenFunctionStatus(IntEnum):
    """Representation or algebra status for thermal Green functions."""

    SUCCESS = 0
    NONFINITE = 1
    INCOMPATIBLE_REPRESENTATION = 2
    RESIDUAL_TOO_LARGE = 3
    RANK_DEFICIENT = 4
    RESOURCE_REJECTED = 5


class GreenRepresentationEvidence(StrictModule):
    """Consumer-visible validity and residual evidence for one representation."""

    residual_norm: Array
    relative_residual: Array
    sum_rule_residual: Array
    finite: Array
    valid: Array
    status: Array
    sample_count: int = eqx.field(static=True)
    source: str = eqx.field(static=True)
    representation: str = eqx.field(static=True)


class GreenFunctionMoments(StrictModule):
    """High-frequency coefficients ``G(z)=sum_k M[k]/z**(k+1)``."""

    values: Array
    active: Array
    statistics: ThermalStatistics = eqx.field(static=True)
    convention: str = eqx.field(static=True)


class ImaginaryTimeGreenFunction(StrictModule):
    """Green-function samples on ``0 <= tau <= beta`` with explicit support."""

    tau: Array
    values: Array
    sample_active: Array
    moments: GreenFunctionMoments | None
    evidence: GreenRepresentationEvidence
    beta: float = eqx.field(static=True)
    statistics: ThermalStatistics = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        beta: float,
        tau: ArrayLike,
        values: ArrayLike,
        /,
        *,
        statistics: ThermalStatistics = "fermionic",
        sample_active: ArrayLike | None = None,
        moments: GreenFunctionMoments | None = None,
        evidence: GreenRepresentationEvidence | None = None,
        representation_id: str | None = None,
    ):
        beta_ = _positive(beta, "beta")
        statistics_ = _statistics(statistics)
        tau_ = jnp.asarray(tau)
        values_ = jnp.asarray(values)
        if tau_.ndim != 1:
            raise ValueError("tau must be rank one.")
        if values_.ndim < 1 or values_.shape[0] != tau_.shape[0]:
            raise ValueError("values must have one leading entry per tau sample.")
        active = _active_mask(sample_active, int(tau_.shape[0]))
        domain_valid = jnp.all((tau_ >= 0.0) & (tau_ <= beta_) | ~active)
        finite = jnp.all(jnp.isfinite(values_) | _inactive_broadcast(~active, values_))
        evidence_ = (
            _representation_evidence(
                int(tau_.shape[0]),
                finite & domain_valid,
                source="samples",
                representation="imaginary-time",
            )
            if evidence is None
            else evidence
        )
        if moments is not None and not isinstance(moments, GreenFunctionMoments):
            raise TypeError("moments must be GreenFunctionMoments or None.")
        if not isinstance(evidence_, GreenRepresentationEvidence):
            raise TypeError("evidence must be GreenRepresentationEvidence or None.")
        identifier = _representation_id(
            representation_id,
            "imaginary-time-green",
            beta_,
            statistics_,
            tau_,
            active,
        )
        self.beta = beta_
        self.statistics = statistics_
        self.tau = tau_
        self.values = values_
        self.sample_active = active
        self.moments = moments
        self.evidence = evidence_
        self.representation_id = identifier

    @property
    def valid(self) -> Array:
        return self.evidence.valid


class MatsubaraGreenFunction(StrictModule):
    """Integer-labelled Matsubara samples with optional high-frequency moments."""

    indices: Array
    values: Array
    sample_active: Array
    moments: GreenFunctionMoments | None
    evidence: GreenRepresentationEvidence
    beta: float = eqx.field(static=True)
    statistics: ThermalStatistics = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        beta: float,
        indices: ArrayLike,
        values: ArrayLike,
        /,
        *,
        statistics: ThermalStatistics = "fermionic",
        sample_active: ArrayLike | None = None,
        moments: GreenFunctionMoments | None = None,
        evidence: GreenRepresentationEvidence | None = None,
        representation_id: str | None = None,
    ):
        beta_ = _positive(beta, "beta")
        statistics_ = _statistics(statistics)
        indices_ = jnp.asarray(indices)
        values_ = jnp.asarray(values)
        if indices_.ndim != 1:
            raise ValueError("indices must be rank one.")
        if not jnp.issubdtype(indices_.dtype, jnp.integer):
            raise TypeError("Matsubara indices must have integer dtype.")
        if values_.ndim < 1 or values_.shape[0] != indices_.shape[0]:
            raise ValueError("values must have one leading entry per Matsubara index.")
        active = _active_mask(sample_active, int(indices_.shape[0]))
        finite = jnp.all(jnp.isfinite(values_) | _inactive_broadcast(~active, values_))
        evidence_ = (
            _representation_evidence(
                int(indices_.shape[0]),
                finite,
                source="samples",
                representation="matsubara",
            )
            if evidence is None
            else evidence
        )
        if moments is not None and not isinstance(moments, GreenFunctionMoments):
            raise TypeError("moments must be GreenFunctionMoments or None.")
        if not isinstance(evidence_, GreenRepresentationEvidence):
            raise TypeError("evidence must be GreenRepresentationEvidence or None.")
        identifier = _representation_id(
            representation_id,
            "matsubara-green",
            beta_,
            statistics_,
            indices_,
            active,
        )
        self.beta = beta_
        self.statistics = statistics_
        self.indices = indices_.astype(jnp.int32)
        self.values = values_
        self.sample_active = active
        self.moments = moments
        self.evidence = evidence_
        self.representation_id = identifier

    @property
    def frequencies(self) -> Array:
        return matsubara_frequencies(
            self.indices,
            beta=self.beta,
            statistics=self.statistics,
        )

    @property
    def valid(self) -> Array:
        return self.evidence.valid


class DLRGreenFunction(StrictModule):
    """Fixed-capacity DLR pole coefficients with explicit transform evidence."""

    basis: PreparedDLRBasis
    coefficients: Array
    moments: GreenFunctionMoments
    evidence: GreenRepresentationEvidence
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PreparedDLRBasis,
        coefficients: ArrayLike,
        /,
        *,
        moment_count: int = 4,
        evidence: GreenRepresentationEvidence | DLRTransformEvidence | None = None,
        representation_id: str | None = None,
    ):
        if not isinstance(basis, PreparedDLRBasis):
            raise TypeError("basis must be a PreparedDLRBasis.")
        values = jnp.asarray(coefficients)
        if values.ndim < 1 or values.shape[0] != basis.frequencies.shape[0]:
            raise ValueError(
                "coefficients must have one leading entry per DLR basis slot."
            )
        count = _positive_int(moment_count, "moment_count")
        masked = values * _payload_mask(basis.active, values)
        moments = _moments_from_coefficients(basis, masked, count)
        finite = jnp.all(jnp.isfinite(masked))
        if evidence is None:
            evidence_ = _representation_evidence(
                int(values.shape[0]),
                finite & basis.valid,
                source="coefficients",
                representation="dlr",
            )
        elif isinstance(evidence, DLRTransformEvidence):
            evidence_ = GreenRepresentationEvidence(
                evidence.residual_norm,
                evidence.relative_residual,
                jnp.asarray(0.0, dtype=evidence.relative_residual.dtype),
                evidence.finite,
                evidence.valid & basis.valid,
                evidence.status,
                int(values.shape[0]),
                "sample-fit",
                "dlr",
            )
        elif isinstance(evidence, GreenRepresentationEvidence):
            evidence_ = evidence
        else:
            raise TypeError(
                "evidence must be GreenRepresentationEvidence, DLRTransformEvidence, or None."
            )
        identifier = _representation_id(
            representation_id,
            "dlr-green",
            basis.beta,
            basis.statistics,
            basis.frequencies,
            basis.active,
            parent=basis.prepared_id,
        )
        self.basis = basis
        self.coefficients = masked
        self.moments = moments
        self.evidence = evidence_
        self.representation_id = identifier

    @property
    def beta(self) -> float:
        return self.basis.beta

    @property
    def statistics(self) -> ThermalStatistics:
        return self.basis.statistics

    @property
    def valid(self) -> Array:
        return self.evidence.valid


class ThermalLehmannPolicy(StrictModule):
    """Fixed-capacity and numerical contract for finite Lehmann preparation."""

    maximum_states: int = eqx.field(static=True)
    maximum_channels: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    weight_tolerance: float = eqx.field(static=True)
    hamiltonian_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_states: int = 4096,
        maximum_channels: int = 64,
        maximum_bytes: int = 512 * 1024**2,
        weight_tolerance: float = 0.0,
        hamiltonian_tolerance: float = 1e-10,
    ):
        for name, value in (
            ("maximum_states", maximum_states),
            ("maximum_channels", maximum_channels),
            ("maximum_bytes", maximum_bytes),
        ):
            _positive_int(value, name)
        weight_ = float(weight_tolerance)
        hamiltonian_ = float(hamiltonian_tolerance)
        if (
            not isfinite(weight_)
            or weight_ < 0.0
            or not isfinite(hamiltonian_)
            or hamiltonian_ < 0.0
        ):
            raise ValueError("Lehmann tolerances must be finite and non-negative.")
        self.maximum_states = int(maximum_states)
        self.maximum_channels = int(maximum_channels)
        self.maximum_bytes = int(maximum_bytes)
        self.weight_tolerance = weight_
        self.hamiltonian_tolerance = hamiltonian_


class ThermalLehmannPlan(StrictModule):
    """Immutable finite-eigensystem shape and resource plan."""

    policy: ThermalLehmannPolicy
    beta: float = eqx.field(static=True)
    statistics: ThermalStatistics = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)
    scalar_operator: bool = eqx.field(static=True)
    persistent_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ThermalLehmannEvidence(StrictModule):
    """Partition, spectral-weight, and finite-eigensystem evidence."""

    partition_function: Array
    log_partition_function: Array
    ground_shift: Array
    spectral_sum: Array
    minimum_probability: Array
    discarded_weight: Array
    hamiltonian_residual: Array
    finite: Array
    valid: Array
    status: Array
    state_count: int = eqx.field(static=True)
    transition_capacity: int = eqx.field(static=True)


class ThermalLehmannRepresentation(StrictModule):
    """Exact fixed-capacity finite-eigensystem Lehmann representation."""

    energies: Array
    probabilities: Array
    poles: Array
    residues: Array
    active: Array
    evidence: ThermalLehmannEvidence
    beta: float = eqx.field(static=True)
    statistics: ThermalStatistics = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.evidence.valid


class DysonPolicy(StrictModule):
    """Bounded dense Dyson preparation and residual contract."""

    maximum_samples: int = eqx.field(static=True)
    maximum_matrix_dimension: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_samples: int = 4096,
        maximum_matrix_dimension: int = 256,
        maximum_bytes: int = 512 * 1024**2,
        rank_tolerance: float = 1e-12,
        residual_tolerance: float = 1e-9,
    ):
        for name, value in (
            ("maximum_samples", maximum_samples),
            ("maximum_matrix_dimension", maximum_matrix_dimension),
            ("maximum_bytes", maximum_bytes),
        ):
            _positive_int(value, name)
        rank_ = _positive(rank_tolerance, "rank_tolerance")
        residual_ = _positive(residual_tolerance, "residual_tolerance")
        self.maximum_samples = int(maximum_samples)
        self.maximum_matrix_dimension = int(maximum_matrix_dimension)
        self.maximum_bytes = int(maximum_bytes)
        self.rank_tolerance = rank_
        self.residual_tolerance = residual_


class DysonPlan(StrictModule):
    """Immutable structure and resource plan for one Dyson identity."""

    policy: DysonPolicy
    sample_count: int = eqx.field(static=True)
    matrix_dimension: int = eqx.field(static=True)
    scalar: bool = eqx.field(static=True)
    persistent_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedDysonSolve(StrictModule):
    """Numerical Dyson denominators and reusable native factorizations."""

    plan: DysonPlan
    noninteracting: MatsubaraGreenFunction
    self_energy: MatsubaraGreenFunction
    denominator: Array
    factors: tuple[Any, ...]
    prepared_id: str = eqx.field(static=True)


class DysonEvidence(StrictModule):
    """Per-frequency identity residual and rank evidence."""

    residual_norm: Array
    relative_residual: Array
    numerical_rank: Array
    condition_estimate: Array
    finite: Array
    valid: Array
    status: Array


class DysonResult(StrictModule):
    green: MatsubaraGreenFunction
    evidence: DysonEvidence


def _statistics(value: str, /) -> ThermalStatistics:
    if value not in ("fermionic", "bosonic"):
        raise ValueError("statistics must be 'fermionic' or 'bosonic'.")
    return value


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _active_mask(value: ArrayLike | None, count: int, /) -> Array:
    if value is None:
        return jnp.ones((count,), dtype=bool)
    active = jnp.asarray(value, dtype=bool)
    if active.shape != (count,):
        raise ValueError(f"sample_active must have shape {(count,)}.")
    return active


def _inactive_broadcast(inactive: Array, values: Array, /) -> Array:
    return inactive.reshape(inactive.shape + (1,) * (values.ndim - 1))


def _payload_mask(active: Array, values: Array, /) -> Array:
    return active.reshape(active.shape + (1,) * (values.ndim - 1)).astype(values.dtype)


def _representation_evidence(
    count: int,
    valid: ArrayLike,
    /,
    *,
    source: str,
    representation: str,
) -> GreenRepresentationEvidence:
    valid_ = jnp.asarray(valid, dtype=bool)
    dtype = jnp.asarray(0.0).dtype
    status = jnp.where(
        valid_, int(GreenFunctionStatus.SUCCESS), int(GreenFunctionStatus.NONFINITE)
    ).astype(jnp.int32)
    return GreenRepresentationEvidence(
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        valid_,
        valid_,
        status,
        count,
        source,
        representation,
    )


def _representation_id(
    value: str | None,
    kind: str,
    beta: float,
    statistics: ThermalStatistics,
    coordinates: Array,
    active: Array,
    /,
    *,
    parent: str | None = None,
) -> str:
    if value is not None:
        identifier = str(value)
        if not identifier:
            raise ValueError("representation_id must be nonempty.")
        return identifier
    return canonical_fingerprint(
        {
            "kind": kind,
            "beta": beta,
            "statistics": statistics,
            "parent": parent,
            "design": array_tree_fingerprint(
                {"coordinates": coordinates, "active": active}
            ),
        }
    )


def _contract_basis(
    kernel: Array, coefficients: Array, query_shape: tuple[int, ...]
) -> Array:
    flattened = kernel.reshape((-1, kernel.shape[-1]))
    result = contract("qr,r...->q...", flattened, coefficients)
    return result.reshape(query_shape + coefficients.shape[1:])


def _moments_from_coefficients(
    basis: PreparedDLRBasis,
    coefficients: Array,
    count: int,
    /,
) -> GreenFunctionMoments:
    orders = jnp.arange(count)
    exponent = orders + (1 if basis.statistics == "bosonic" else 0)
    powers = basis.frequencies[None, :] ** exponent[:, None]
    powers = powers * basis.active[None, :]
    values = contract("kr,r...->k...", powers, coefficients)
    return GreenFunctionMoments(
        values,
        jnp.ones((count,), dtype=bool),
        basis.statistics,
        "inverse-frequency",
    )


def evaluate_dlr_tau(green: DLRGreenFunction, tau: ArrayLike, /) -> Array:
    """Evaluate a DLR Green function at arbitrary imaginary times."""

    if not isinstance(green, DLRGreenFunction):
        raise TypeError("green must be a DLRGreenFunction.")
    tau_ = jnp.asarray(tau)
    kernel = green.basis.tau_kernel(tau_)
    return _contract_basis(kernel, green.coefficients, tau_.shape)


def evaluate_dlr_matsubara(green: DLRGreenFunction, indices: ArrayLike, /) -> Array:
    """Evaluate a DLR Green function at integer Matsubara labels."""

    if not isinstance(green, DLRGreenFunction):
        raise TypeError("green must be a DLRGreenFunction.")
    indices_ = jnp.asarray(indices)
    kernel = green.basis.matsubara_kernel(indices_)
    return _contract_basis(kernel, green.coefficients, indices_.shape)


def evaluate_dlr(green: DLRGreenFunction, frequency: ArrayLike, /) -> Array:
    """Evaluate the meromorphic DLR representation at complex frequencies."""

    if not isinstance(green, DLRGreenFunction):
        raise TypeError("green must be a DLRGreenFunction.")
    z = jnp.asarray(frequency)
    omega = green.basis.frequencies
    active = green.basis.active
    denominator = jnp.where(active, z[..., None] - omega, 1.0 + 0.0j)
    kernel = jnp.where(active, jnp.reciprocal(denominator), 0.0)
    if green.statistics == "bosonic":
        kernel = kernel * omega
    return _contract_basis(kernel, green.coefficients, z.shape)


def dlr_from_poles(
    basis: PreparedDLRBasis,
    poles: ArrayLike,
    residues: ArrayLike,
    /,
    *,
    moment_count: int = 4,
    tolerance: float | None = None,
) -> DLRGreenFunction:
    """Project one or many physical poles into a prepared DLR basis."""

    if not isinstance(basis, PreparedDLRBasis):
        raise TypeError("basis must be a PreparedDLRBasis.")
    poles_ = jnp.asarray(poles)
    residues_ = jnp.asarray(residues)
    if poles_.ndim != 1 or residues_.ndim < 1 or residues_.shape[0] != poles_.shape[0]:
        raise ValueError("residues must have one leading entry per rank-one pole array.")
    labels = basis.matsubara_indices
    nu = matsubara_frequencies(labels, beta=basis.beta, statistics=basis.statistics)
    physical_kernel = jnp.reciprocal(1j * nu[:, None] - poles_[None, :])
    values = _contract_basis(physical_kernel, residues_, labels.shape)
    values = values * _payload_mask(basis.active, values)
    fit = fit_dlr_from_matsubara(
        basis,
        labels,
        values,
        tolerance=basis.plan.policy.tolerance if tolerance is None else tolerance,
        sample_active=basis.active,
    )
    within_cutoff = jnp.all(jnp.abs(poles_) <= basis.cutoff)
    evidence = GreenRepresentationEvidence(
        fit.evidence.residual_norm,
        fit.evidence.relative_residual,
        jnp.asarray(0.0, dtype=fit.evidence.relative_residual.dtype),
        fit.evidence.finite,
        fit.evidence.valid & within_cutoff,
        jnp.where(
            within_cutoff,
            fit.evidence.status,
            int(GreenFunctionStatus.INCOMPATIBLE_REPRESENTATION),
        ).astype(jnp.int32),
        int(labels.shape[0]),
        "physical-poles",
        "dlr",
    )
    return DLRGreenFunction(
        basis,
        fit.coefficients,
        moment_count=moment_count,
        evidence=evidence,
    )


def imaginary_time_to_matsubara(
    samples: ImaginaryTimeGreenFunction,
    basis: PreparedDLRBasis,
    indices: ArrayLike | None = None,
    /,
    *,
    tolerance: float | None = None,
) -> MatsubaraGreenFunction:
    """Transform imaginary-time samples to Matsubara values through DLR."""

    return dlr_to_matsubara(
        imaginary_time_to_dlr(samples, basis, tolerance=tolerance),
        indices,
    )


def matsubara_to_imaginary_time(
    samples: MatsubaraGreenFunction,
    basis: PreparedDLRBasis,
    tau: ArrayLike | None = None,
    /,
    *,
    tolerance: float | None = None,
) -> ImaginaryTimeGreenFunction:
    """Transform Matsubara samples to imaginary time through DLR."""

    return dlr_to_imaginary_time(
        matsubara_to_dlr(samples, basis, tolerance=tolerance),
        tau,
    )


def imaginary_time_to_dlr(
    samples: ImaginaryTimeGreenFunction,
    basis: PreparedDLRBasis,
    /,
    *,
    moment_count: int = 4,
    tolerance: float | None = None,
) -> DLRGreenFunction:
    """Transform imaginary-time samples to fixed-capacity DLR coefficients."""

    _compatible_basis(samples.beta, samples.statistics, basis)
    fit = fit_dlr_from_tau(
        basis,
        samples.tau,
        samples.values,
        tolerance=basis.plan.policy.tolerance if tolerance is None else tolerance,
        sample_active=samples.sample_active,
    )
    return DLRGreenFunction(
        basis,
        fit.coefficients,
        moment_count=moment_count,
        evidence=fit.evidence,
    )


def matsubara_to_dlr(
    samples: MatsubaraGreenFunction,
    basis: PreparedDLRBasis,
    /,
    *,
    moment_count: int = 4,
    tolerance: float | None = None,
) -> DLRGreenFunction:
    """Transform Matsubara samples to fixed-capacity DLR coefficients."""

    _compatible_basis(samples.beta, samples.statistics, basis)
    fit = fit_dlr_from_matsubara(
        basis,
        samples.indices,
        samples.values,
        tolerance=basis.plan.policy.tolerance if tolerance is None else tolerance,
        sample_active=samples.sample_active,
    )
    return DLRGreenFunction(
        basis,
        fit.coefficients,
        moment_count=moment_count,
        evidence=fit.evidence,
    )


def _compatible_basis(
    beta: float, statistics: ThermalStatistics, basis: PreparedDLRBasis, /
) -> None:
    if not isinstance(basis, PreparedDLRBasis):
        raise TypeError("basis must be a PreparedDLRBasis.")
    if beta != basis.beta or statistics != basis.statistics:
        raise ValueError("Green-function beta/statistics do not match the DLR basis.")


def dlr_to_imaginary_time(
    green: DLRGreenFunction,
    tau: ArrayLike | None = None,
    /,
) -> ImaginaryTimeGreenFunction:
    """Materialize DLR values on caller points or the prepared fixed design."""

    points = green.basis.tau_nodes if tau is None else jnp.asarray(tau)
    active = green.basis.active if tau is None else jnp.ones(points.shape, dtype=bool)
    values = evaluate_dlr_tau(green, points)
    evidence = GreenRepresentationEvidence(
        green.evidence.residual_norm,
        green.evidence.relative_residual,
        green.evidence.sum_rule_residual,
        green.evidence.finite,
        green.evidence.valid,
        green.evidence.status,
        int(points.size),
        "dlr",
        "imaginary-time",
    )
    return ImaginaryTimeGreenFunction(
        green.beta,
        points.reshape((-1,)),
        values.reshape((points.size,) + green.coefficients.shape[1:]),
        statistics=green.statistics,
        sample_active=active.reshape((-1,)),
        moments=green.moments,
        evidence=evidence,
    )


def dlr_to_matsubara(
    green: DLRGreenFunction,
    indices: ArrayLike | None = None,
    /,
) -> MatsubaraGreenFunction:
    """Materialize DLR values on caller labels or the prepared fixed design."""

    labels = green.basis.matsubara_indices if indices is None else jnp.asarray(indices)
    active = green.basis.active if indices is None else jnp.ones(labels.shape, dtype=bool)
    values = evaluate_dlr_matsubara(green, labels)
    evidence = GreenRepresentationEvidence(
        green.evidence.residual_norm,
        green.evidence.relative_residual,
        green.evidence.sum_rule_residual,
        green.evidence.finite,
        green.evidence.valid,
        green.evidence.status,
        int(labels.size),
        "dlr",
        "matsubara",
    )
    return MatsubaraGreenFunction(
        green.beta,
        labels.reshape((-1,)).astype(jnp.int32),
        values.reshape((labels.size,) + green.coefficients.shape[1:]),
        statistics=green.statistics,
        sample_active=active.reshape((-1,)),
        moments=green.moments,
        evidence=evidence,
    )


def differentiate_dlr(
    green: DLRGreenFunction,
    order: int = 1,
    /,
) -> DLRGreenFunction:
    """Differentiate in imaginary time exactly within the DLR representation."""

    order_ = _positive_int(order, "order")
    scale = (-green.basis.frequencies) ** order_
    coefficients = green.coefficients * scale.reshape(
        scale.shape + (1,) * (green.coefficients.ndim - 1)
    )
    return DLRGreenFunction(
        green.basis,
        coefficients,
        moment_count=int(green.moments.values.shape[0]),
        evidence=green.evidence,
    )


def dlr_moments(
    green: DLRGreenFunction,
    count: int | None = None,
    /,
) -> GreenFunctionMoments:
    """Return existing moments or compute a requested fixed number."""

    if count is None:
        return green.moments
    return _moments_from_coefficients(
        green.basis,
        green.coefficients,
        _positive_int(count, "count"),
    )


def evaluate_matsubara_tail(
    moments: GreenFunctionMoments,
    indices: ArrayLike,
    /,
    *,
    beta: float,
) -> Array:
    """Evaluate a finite high-frequency moment tail."""

    if not isinstance(moments, GreenFunctionMoments):
        raise TypeError("moments must be GreenFunctionMoments.")
    labels = jnp.asarray(indices)
    nu = matsubara_frequencies(labels, beta=beta, statistics=moments.statistics).astype(
        jnp.result_type(moments.values, 1j)
    )
    orders = jnp.arange(moments.values.shape[0]) + 1
    powers = (1j * nu[..., None]) ** (-orders)
    powers = powers * moments.active
    return _contract_basis(powers, moments.values, labels.shape)


def convolve_dlr(
    left: DLRGreenFunction,
    right: DLRGreenFunction,
    /,
    *,
    output_basis: PreparedDLRBasis | None = None,
    tolerance: float | None = None,
) -> DLRGreenFunction:
    """Project the thermal convolution, a pointwise Matsubara product, to DLR."""

    if left.beta != right.beta:
        raise ValueError("Convolved Green functions must have the same beta.")
    if left.statistics != right.statistics:
        raise ValueError("A Matsubara convolution requires a common frequency lattice.")
    basis = left.basis if output_basis is None else output_basis
    _compatible_basis(left.beta, left.statistics, basis)
    labels = basis.matsubara_indices
    left_values = evaluate_dlr_matsubara(left, labels)
    right_values = evaluate_dlr_matsubara(right, labels)
    if left_values.ndim == 3 and right_values.ndim == 3:
        values = contract("nab,nbc->nac", left_values, right_values)
    else:
        values = left_values * right_values
    fit = fit_dlr_from_matsubara(
        basis,
        labels,
        values,
        tolerance=basis.plan.policy.tolerance if tolerance is None else tolerance,
        sample_active=basis.active,
    )
    return DLRGreenFunction(basis, fit.coefficients, evidence=fit.evidence)


def plan_thermal_lehmann(
    energies: ArrayLike,
    operators: ArrayLike,
    beta: float,
    /,
    *,
    statistics: ThermalStatistics = "fermionic",
    policy: ThermalLehmannPolicy | None = None,
) -> ThermalLehmannPlan:
    """Validate eigensystem shapes and reject oversized transition banks."""

    beta_ = _positive(beta, "beta")
    statistics_ = _statistics(statistics)
    policy_ = ThermalLehmannPolicy() if policy is None else policy
    if not isinstance(policy_, ThermalLehmannPolicy):
        raise TypeError("policy must be a ThermalLehmannPolicy or None.")
    energy = jnp.asarray(energies)
    operator = jnp.asarray(operators)
    if energy.ndim != 1 or energy.shape[0] == 0:
        raise ValueError("energies must be one nonempty rank-one array.")
    count = int(energy.shape[0])
    scalar = operator.ndim == 2
    channel_count = 1 if scalar else int(operator.shape[0]) if operator.ndim == 3 else 0
    if operator.ndim not in (2, 3) or operator.shape[-2:] != (count, count):
        raise ValueError(
            "operators must have shape (state,state) or (channel,state,state)."
        )
    if count > policy_.maximum_states:
        raise ValueError("Lehmann state count exceeds maximum_states.")
    if channel_count > policy_.maximum_channels:
        raise ValueError("Lehmann channel count exceeds maximum_channels.")
    itemsize = np.dtype(jnp.result_type(energy, operator, 1j)).itemsize
    transitions = count * count
    persistent = (
        itemsize * (2 * count + transitions + transitions * channel_count * channel_count)
        + transitions
    )
    workspace = itemsize * (
        2 * count * count + transitions * channel_count * channel_count
    )
    if persistent + workspace > policy_.maximum_bytes:
        raise ValueError(
            "Lehmann representation exceeds maximum_bytes before allocation."
        )
    plan_id = canonical_fingerprint(
        {
            "kind": "finite-thermal-lehmann-plan",
            "beta": beta_,
            "statistics": statistics_,
            "state_count": count,
            "channel_count": channel_count,
            "scalar_operator": scalar,
            "energy_dtype": str(energy.dtype),
            "operator_dtype": str(operator.dtype),
            "maximum_states": policy_.maximum_states,
            "maximum_channels": policy_.maximum_channels,
            "maximum_bytes": policy_.maximum_bytes,
            "weight_tolerance": policy_.weight_tolerance,
            "hamiltonian_tolerance": policy_.hamiltonian_tolerance,
            "persistent_bytes": persistent,
            "workspace_bytes": workspace,
        }
    )
    return ThermalLehmannPlan(
        policy_,
        beta_,
        statistics_,
        count,
        channel_count,
        scalar,
        persistent,
        workspace,
        plan_id,
    )


def prepare_thermal_lehmann(
    plan: ThermalLehmannPlan,
    energies: ArrayLike,
    operators: ArrayLike,
    /,
    *,
    eigenvectors: ArrayLike | None = None,
    hamiltonian_residual: ArrayLike = 0.0,
) -> ThermalLehmannRepresentation:
    """Prepare the exact fixed-capacity Lehmann sum for a validated plan."""

    if not isinstance(plan, ThermalLehmannPlan):
        raise TypeError("plan must be a ThermalLehmannPlan.")
    beta_ = plan.beta
    statistics_ = plan.statistics
    tolerance = plan.policy.weight_tolerance
    energy = jnp.asarray(energies)
    operator = jnp.asarray(operators)
    if energy.shape != (plan.state_count,):
        raise ValueError("energies do not match the Lehmann plan.")
    expected = (
        (plan.state_count, plan.state_count)
        if plan.scalar_operator
        else (plan.channel_count, plan.state_count, plan.state_count)
    )
    if operator.shape != expected:
        raise ValueError("operators do not match the Lehmann plan.")
    count = plan.state_count
    scalar_operator = plan.scalar_operator
    if scalar_operator:
        operator = operator[None, ...]
    if eigenvectors is not None:
        vectors = jnp.asarray(eigenvectors)
        if vectors.shape != (count, count):
            raise ValueError("eigenvectors must have shape (state,state).")
        operator = contract("im,cij,jn->cmn", jnp.conj(vectors), operator, vectors)
    ground = jnp.min(energy)
    boltzmann = jnp.exp(-beta_ * (energy - ground))
    partition_scaled = jnp.sum(boltzmann)
    probabilities = boltzmann / partition_scaled
    log_partition = jnp.log(partition_scaled) - beta_ * ground
    partition = jnp.exp(log_partition)
    poles_matrix = energy[None, :] - energy[:, None]
    if statistics_ == "fermionic":
        factors = probabilities[:, None] + probabilities[None, :]
    else:
        factors = probabilities[:, None] - probabilities[None, :]
    amplitudes = operator[:, :, :]
    residue_bank = contract("mn,amn,bmn->mnab", factors, amplitudes, jnp.conj(amplitudes))
    poles = poles_matrix.reshape((-1,))
    residues = residue_bank.reshape((count * count,) + residue_bank.shape[-2:])
    if scalar_operator:
        residues = residues[:, 0, 0]
    residue_magnitude = jnp.max(jnp.abs(residues).reshape((count * count, -1)), axis=1)
    active = residue_magnitude > tolerance
    discarded = jnp.sum(jnp.where(active, 0.0, residue_magnitude))
    residues = residues * _payload_mask(active, residues)
    spectral_sum = jnp.sum(residues, axis=0)
    hamiltonian_residual_ = jnp.asarray(hamiltonian_residual)
    finite = (
        jnp.all(jnp.isfinite(energy))
        & jnp.all(jnp.isfinite(probabilities))
        & jnp.all(jnp.isfinite(residues))
        & jnp.isfinite(hamiltonian_residual_)
        & jnp.isfinite(log_partition)
        & jnp.isfinite(partition)
    )
    hamiltonian_valid = hamiltonian_residual_ <= plan.policy.hamiltonian_tolerance
    valid = finite & (partition_scaled > 0.0) & hamiltonian_valid
    status = jnp.where(
        ~finite,
        int(GreenFunctionStatus.NONFINITE),
        jnp.where(
            hamiltonian_valid,
            int(GreenFunctionStatus.SUCCESS),
            int(GreenFunctionStatus.RESIDUAL_TOO_LARGE),
        ),
    ).astype(jnp.int32)
    evidence = ThermalLehmannEvidence(
        partition,
        log_partition,
        ground,
        spectral_sum,
        jnp.min(probabilities),
        discarded,
        hamiltonian_residual_,
        finite,
        valid,
        status,
        count,
        count * count,
    )
    representation_id = canonical_fingerprint(
        {
            "kind": "finite-thermal-lehmann",
            "plan": plan.plan_id,
            "beta": beta_,
            "statistics": statistics_,
            "eigensystem": array_tree_fingerprint(
                {"energies": energy, "operators": operator}
            ),
        }
    )
    return ThermalLehmannRepresentation(
        energy,
        probabilities,
        poles,
        residues,
        active,
        evidence,
        beta_,
        statistics_,
        representation_id,
    )


def thermal_lehmann_sum(
    energies: ArrayLike,
    operators: ArrayLike,
    beta: float,
    /,
    *,
    statistics: ThermalStatistics = "fermionic",
    eigenvectors: ArrayLike | None = None,
    weight_tolerance: float = 0.0,
    hamiltonian_residual: ArrayLike = 0.0,
    maximum_states: int = 4096,
    maximum_channels: int = 64,
    maximum_bytes: int = 512 * 1024**2,
    hamiltonian_tolerance: float = 1e-10,
) -> ThermalLehmannRepresentation:
    """Plan and prepare an exact finite-eigensystem Lehmann representation."""

    policy = ThermalLehmannPolicy(
        maximum_states=maximum_states,
        maximum_channels=maximum_channels,
        maximum_bytes=maximum_bytes,
        weight_tolerance=weight_tolerance,
        hamiltonian_tolerance=hamiltonian_tolerance,
    )
    plan = plan_thermal_lehmann(
        energies,
        operators,
        beta,
        statistics=statistics,
        policy=policy,
    )
    return prepare_thermal_lehmann(
        plan,
        energies,
        operators,
        eigenvectors=eigenvectors,
        hamiltonian_residual=hamiltonian_residual,
    )


def thermal_lehmann_from_hamiltonian(
    hamiltonian: ArrayLike,
    operators: ArrayLike,
    beta: float,
    /,
    *,
    statistics: ThermalStatistics = "fermionic",
    hermiticity_tolerance: float = 1e-10,
    weight_tolerance: float = 0.0,
    maximum_states: int = 4096,
    maximum_channels: int = 64,
    maximum_bytes: int = 512 * 1024**2,
) -> ThermalLehmannRepresentation:
    """Diagonalize a finite Hamiltonian through Phydrax linalg and sum exactly."""

    spectrum = HermitianSpectrum(hamiltonian, tolerance=hermiticity_tolerance)
    return thermal_lehmann_sum(
        spectrum.eigenvalues,
        operators,
        beta,
        statistics=statistics,
        eigenvectors=spectrum.eigenvectors,
        weight_tolerance=weight_tolerance,
        hamiltonian_residual=spectrum.hermiticity_residual,
        maximum_states=maximum_states,
        maximum_channels=maximum_channels,
        maximum_bytes=maximum_bytes,
        hamiltonian_tolerance=hermiticity_tolerance,
    )


def evaluate_lehmann(
    lehmann: ThermalLehmannRepresentation,
    frequency: ArrayLike,
    /,
) -> Array:
    """Evaluate an exact finite Lehmann sum at complex frequencies."""

    if not isinstance(lehmann, ThermalLehmannRepresentation):
        raise TypeError("lehmann must be a ThermalLehmannRepresentation.")
    z = jnp.asarray(frequency)
    denominator = z[..., None] - lehmann.poles
    safe = jnp.where(lehmann.active, denominator, 1.0 + 0.0j)
    kernel = jnp.where(lehmann.active, jnp.reciprocal(safe), 0.0)
    return _contract_basis(kernel, lehmann.residues, z.shape)


def evaluate_lehmann_matsubara(
    lehmann: ThermalLehmannRepresentation,
    indices: ArrayLike,
    /,
) -> Array:
    labels = jnp.asarray(indices)
    frequency = matsubara_frequencies(
        labels, beta=lehmann.beta, statistics=lehmann.statistics
    )
    return evaluate_lehmann(lehmann, 1j * frequency)


def evaluate_lehmann_tau(
    lehmann: ThermalLehmannRepresentation,
    tau: ArrayLike,
    /,
) -> Array:
    """Evaluate the exact imaginary-time Lehmann sum with stable kernels."""

    tau_ = jnp.asarray(tau)
    if lehmann.statistics == "fermionic":
        kernel = thermal_tau_kernel(
            tau_,
            lehmann.poles,
            beta=lehmann.beta,
            statistics="fermionic",
        )
    else:
        omega = lehmann.poles
        regularized = thermal_tau_kernel(
            tau_, omega, beta=lehmann.beta, statistics="bosonic"
        )
        safe_omega = jnp.where(omega != 0.0, omega, 1.0)
        kernel = jnp.where(omega != 0.0, regularized / safe_omega, 0.0)
    kernel = kernel * lehmann.active
    return _contract_basis(kernel, lehmann.residues, tau_.shape)


def lehmann_moments(
    lehmann: ThermalLehmannRepresentation,
    count: int = 4,
    /,
) -> GreenFunctionMoments:
    """Compute exact high-frequency moments of a finite Lehmann sum."""

    if not isinstance(lehmann, ThermalLehmannRepresentation):
        raise TypeError("lehmann must be a ThermalLehmannRepresentation.")
    count_ = _positive_int(count, "count")
    powers = lehmann.poles[None, :] ** jnp.arange(count_)[:, None]
    powers = powers * lehmann.active[None, :]
    values = contract("kr,r...->k...", powers, lehmann.residues)
    return GreenFunctionMoments(
        values,
        jnp.ones((count_,), dtype=bool),
        lehmann.statistics,
        "inverse-frequency",
    )


def lehmann_to_dlr(
    lehmann: ThermalLehmannRepresentation,
    basis: PreparedDLRBasis,
    /,
    *,
    tolerance: float | None = None,
) -> DLRGreenFunction:
    """Project an exact finite Lehmann sum to a compatible DLR basis."""

    _compatible_basis(lehmann.beta, lehmann.statistics, basis)
    values = evaluate_lehmann_matsubara(lehmann, basis.matsubara_indices)
    values = values * _payload_mask(basis.active, values)
    fit = fit_dlr_from_matsubara(
        basis,
        basis.matsubara_indices,
        values,
        tolerance=basis.plan.policy.tolerance if tolerance is None else tolerance,
        sample_active=basis.active,
    )
    return DLRGreenFunction(basis, fit.coefficients, evidence=fit.evidence)


def _aligned_matsubara(
    left: MatsubaraGreenFunction, right: MatsubaraGreenFunction, /
) -> None:
    if left.beta != right.beta or left.statistics != right.statistics:
        raise ValueError("Matsubara Green functions must share beta and statistics.")
    if left.indices.shape != right.indices.shape or not np.array_equal(
        np.asarray(left.indices), np.asarray(right.indices)
    ):
        raise ValueError("Matsubara Green functions must share exactly the same indices.")
    if left.values.shape != right.values.shape:
        raise ValueError("Matsubara Green-function values must have matching shapes.")


def _native_factor(matrix: Array, tolerance: float, /):
    return factorize(
        DenseLinearOperator(matrix),
        FactorizationPolicy("svd", rank=RankPolicy(relative_cutoff=float(tolerance))),
    )


def plan_dyson_solve(
    noninteracting: MatsubaraGreenFunction,
    self_energy: MatsubaraGreenFunction,
    /,
    *,
    policy: DysonPolicy | None = None,
) -> DysonPlan:
    """Validate one Dyson structure and reject oversized dense work up front."""

    if not isinstance(noninteracting, MatsubaraGreenFunction) or not isinstance(
        self_energy, MatsubaraGreenFunction
    ):
        raise TypeError("Dyson inputs must be MatsubaraGreenFunction values.")
    _aligned_matsubara(noninteracting, self_energy)
    policy_ = DysonPolicy() if policy is None else policy
    if not isinstance(policy_, DysonPolicy):
        raise TypeError("policy must be a DysonPolicy or None.")
    sample_count = int(noninteracting.indices.shape[0])
    scalar = noninteracting.values.ndim == 1
    if not scalar and (
        noninteracting.values.ndim != 3
        or noninteracting.values.shape[-1] != noninteracting.values.shape[-2]
    ):
        raise ValueError("Dyson values must be scalar samples or square matrix samples.")
    dimension = 1 if scalar else int(noninteracting.values.shape[-1])
    if sample_count > policy_.maximum_samples:
        raise ValueError("Dyson sample count exceeds maximum_samples.")
    if dimension > policy_.maximum_matrix_dimension:
        raise ValueError("Dyson matrix dimension exceeds maximum_matrix_dimension.")
    itemsize = np.dtype(noninteracting.values.dtype).itemsize
    persistent = itemsize * sample_count * dimension * dimension * 3
    workspace = itemsize * sample_count * dimension * dimension * 5
    if persistent + workspace > policy_.maximum_bytes:
        raise ValueError("Dyson solve exceeds maximum_bytes before factorization.")
    plan_id = canonical_fingerprint(
        {
            "kind": "matsubara-dyson-plan",
            "samples": sample_count,
            "matrix_dimension": dimension,
            "scalar": scalar,
            "dtype": str(noninteracting.values.dtype),
            "rank_tolerance": policy_.rank_tolerance,
            "residual_tolerance": policy_.residual_tolerance,
            "persistent_bytes": persistent,
            "workspace_bytes": workspace,
        }
    )
    return DysonPlan(
        policy_, sample_count, dimension, scalar, persistent, workspace, plan_id
    )


def prepare_dyson_solve(
    plan: DysonPlan,
    noninteracting: MatsubaraGreenFunction,
    self_energy: MatsubaraGreenFunction,
    /,
) -> PreparedDysonSolve:
    """Prepare numerical inverse Green-function denominators."""

    if not isinstance(plan, DysonPlan):
        raise TypeError("plan must be a DysonPlan.")
    _aligned_matsubara(noninteracting, self_energy)
    if int(noninteracting.indices.shape[0]) != plan.sample_count:
        raise ValueError("Dyson inputs do not match the planned sample count.")
    if plan.scalar:
        denominator = jnp.reciprocal(noninteracting.values) - self_energy.values
        factors: tuple[Any, ...] = ()
    else:
        inverses = []
        denominator_rows = []
        identity = jnp.eye(plan.matrix_dimension, dtype=noninteracting.values.dtype)
        for sample in range(plan.sample_count):
            g0_factor = _native_factor(
                noninteracting.values[sample], plan.policy.rank_tolerance
            )
            inverse = g0_factor.solve(identity).value
            inverses.append(g0_factor)
            denominator_rows.append(inverse - self_energy.values[sample])
        denominator = jnp.stack(denominator_rows)
        factors = tuple(
            _native_factor(row, plan.policy.rank_tolerance) for row in denominator
        )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-matsubara-dyson",
            "plan": plan.plan_id,
            "inputs": [noninteracting.representation_id, self_energy.representation_id],
            "denominator": array_tree_fingerprint(denominator),
        }
    )
    return PreparedDysonSolve(
        plan,
        noninteracting,
        self_energy,
        denominator,
        factors,
        prepared_id,
    )


def solve_dyson(prepared: PreparedDysonSolve, /) -> DysonResult:
    """Execute a prepared Dyson solve and retain the identity residual."""

    if not isinstance(prepared, PreparedDysonSolve):
        raise TypeError("prepared must be a PreparedDysonSolve.")
    plan = prepared.plan
    if plan.scalar:
        values = jnp.reciprocal(prepared.denominator)
        residual = jnp.abs(prepared.denominator * values - 1.0)
        ranks = jnp.where(jnp.abs(prepared.denominator) > 0.0, 1, 0).astype(jnp.int32)
        conditions = jnp.ones_like(residual.real)
    else:
        identity = jnp.eye(plan.matrix_dimension, dtype=prepared.denominator.dtype)
        solved = tuple(factor.solve(identity) for factor in prepared.factors)
        values = jnp.stack(tuple(result.value for result in solved))
        residual = jnp.max(
            jnp.abs(prepared.denominator @ values - identity), axis=(-2, -1)
        )
        ranks = jnp.stack(tuple(factor.rank() for factor in prepared.factors))
        conditions = jnp.stack(
            tuple(
                factor.singular_values()[0]
                / jnp.maximum(
                    factor.singular_values()[-1],
                    jnp.finfo(factor.singular_values().dtype).tiny,
                )
                for factor in prepared.factors
            )
        )
    scale = jnp.maximum(
        1.0,
        jnp.abs(prepared.denominator)
        if plan.scalar
        else jnp.max(jnp.abs(prepared.denominator), axis=(-2, -1)),
    )
    relative = residual / scale
    finite = jnp.isfinite(relative) & (
        jnp.isfinite(values)
        if plan.scalar
        else jnp.all(jnp.isfinite(values), axis=(-2, -1))
    )
    valid = (
        finite
        & (ranks == plan.matrix_dimension)
        & (relative <= plan.policy.residual_tolerance)
    )
    status = jnp.where(
        ~finite,
        int(GreenFunctionStatus.NONFINITE),
        jnp.where(
            ranks < plan.matrix_dimension,
            int(GreenFunctionStatus.RANK_DEFICIENT),
            jnp.where(
                valid,
                int(GreenFunctionStatus.SUCCESS),
                int(GreenFunctionStatus.RESIDUAL_TOO_LARGE),
            ),
        ),
    ).astype(jnp.int32)
    evidence = DysonEvidence(
        residual,
        relative,
        ranks,
        conditions,
        finite,
        valid,
        status,
    )
    aggregate_valid = jnp.all(valid)
    green_evidence = GreenRepresentationEvidence(
        jnp.max(residual),
        jnp.max(relative),
        jnp.asarray(0.0, dtype=relative.dtype),
        jnp.all(finite),
        aggregate_valid,
        jnp.where(
            aggregate_valid,
            int(GreenFunctionStatus.SUCCESS),
            jnp.max(status),
        ).astype(jnp.int32),
        plan.sample_count,
        "dyson",
        "matsubara",
    )
    green = MatsubaraGreenFunction(
        prepared.noninteracting.beta,
        prepared.noninteracting.indices,
        values,
        statistics=prepared.noninteracting.statistics,
        sample_active=(
            prepared.noninteracting.sample_active & prepared.self_energy.sample_active
        ),
        evidence=green_evidence,
    )
    return DysonResult(green, evidence)


def dyson_solve(
    noninteracting: MatsubaraGreenFunction,
    self_energy: MatsubaraGreenFunction,
    /,
    *,
    policy: DysonPolicy | None = None,
) -> DysonResult:
    """Plan, prepare, and execute the Matsubara Dyson identity."""

    plan = plan_dyson_solve(noninteracting, self_energy, policy=policy)
    prepared = prepare_dyson_solve(plan, noninteracting, self_energy)
    return solve_dyson(prepared)


def extract_self_energy(
    noninteracting: MatsubaraGreenFunction,
    interacting: MatsubaraGreenFunction,
    /,
    *,
    policy: DysonPolicy | None = None,
) -> DysonResult:
    """Extract ``Sigma = G0**-1 - G**-1`` with inverse residual evidence."""
    if not isinstance(noninteracting, MatsubaraGreenFunction) or not isinstance(
        interacting, MatsubaraGreenFunction
    ):
        raise TypeError("Self-energy inputs must be MatsubaraGreenFunction values.")
    _aligned_matsubara(noninteracting, interacting)

    zero = MatsubaraGreenFunction(
        noninteracting.beta,
        noninteracting.indices,
        jnp.zeros_like(noninteracting.values),
        statistics=noninteracting.statistics,
        sample_active=noninteracting.sample_active,
    )
    plan = plan_dyson_solve(noninteracting, zero, policy=policy)
    if plan.scalar:
        values = jnp.reciprocal(noninteracting.values) - jnp.reciprocal(
            interacting.values
        )
    else:
        identity = jnp.eye(plan.matrix_dimension, dtype=noninteracting.values.dtype)
        values = jnp.stack(
            tuple(
                _native_factor(noninteracting.values[index], plan.policy.rank_tolerance)
                .solve(identity)
                .value
                - _native_factor(interacting.values[index], plan.policy.rank_tolerance)
                .solve(identity)
                .value
                for index in range(plan.sample_count)
            )
        )
    sigma = MatsubaraGreenFunction(
        noninteracting.beta,
        noninteracting.indices,
        values,
        statistics=noninteracting.statistics,
        sample_active=noninteracting.sample_active & interacting.sample_active,
    )
    round_trip = dyson_solve(noninteracting, sigma, policy=plan.policy)
    difference = round_trip.green.values - interacting.values
    residual = (
        jnp.abs(difference)
        if plan.scalar
        else jnp.max(jnp.abs(difference), axis=(-2, -1))
    )
    scale = (
        jnp.maximum(jnp.abs(interacting.values), 1.0)
        if plan.scalar
        else jnp.maximum(jnp.max(jnp.abs(interacting.values), axis=(-2, -1)), 1.0)
    )
    relative = residual / scale
    valid = round_trip.evidence.valid & (relative <= plan.policy.residual_tolerance)
    evidence = DysonEvidence(
        residual,
        relative,
        round_trip.evidence.numerical_rank,
        round_trip.evidence.condition_estimate,
        jnp.isfinite(relative),
        valid,
        jnp.where(
            valid,
            int(GreenFunctionStatus.SUCCESS),
            int(GreenFunctionStatus.RESIDUAL_TOO_LARGE),
        ).astype(jnp.int32),
    )
    return DysonResult(sigma, evidence)


__all__ = [
    "DLRGreenFunction",
    "DysonEvidence",
    "DysonPlan",
    "DysonPolicy",
    "DysonResult",
    "GreenFunctionMoments",
    "GreenFunctionStatus",
    "GreenRepresentationEvidence",
    "ImaginaryTimeGreenFunction",
    "MatsubaraGreenFunction",
    "PreparedDysonSolve",
    "ThermalLehmannPlan",
    "ThermalLehmannPolicy",
    "ThermalLehmannEvidence",
    "ThermalLehmannRepresentation",
    "convolve_dlr",
    "differentiate_dlr",
    "dlr_from_poles",
    "dlr_moments",
    "dlr_to_imaginary_time",
    "dlr_to_matsubara",
    "dyson_solve",
    "evaluate_dlr",
    "evaluate_dlr_matsubara",
    "evaluate_dlr_tau",
    "evaluate_lehmann",
    "evaluate_lehmann_matsubara",
    "evaluate_lehmann_tau",
    "evaluate_matsubara_tail",
    "extract_self_energy",
    "imaginary_time_to_dlr",
    "imaginary_time_to_matsubara",
    "lehmann_moments",
    "lehmann_to_dlr",
    "matsubara_to_dlr",
    "matsubara_to_imaginary_time",
    "plan_dyson_solve",
    "plan_thermal_lehmann",
    "prepare_dyson_solve",
    "prepare_thermal_lehmann",
    "solve_dyson",
    "thermal_lehmann_from_hamiltonian",
    "thermal_lehmann_sum",
]
