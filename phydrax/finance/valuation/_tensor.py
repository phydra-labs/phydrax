#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...tensor_train import (
    qtt_evaluate,
    QuanticsLayout,
    TensorizedGrid,
    TensorTrain,
    TensorTrainCompressionResult,
    TTCrossResult,
)
from ..core import FinanceEvidenceBinding, PricingLaw


TensorApproximation: TypeAlias = TensorTrainCompressionResult | TTCrossResult
TensorRoute = Literal["tt", "qtt"]


class TensorValuationApplicability(StrictModule):
    """Immutable grid, rank, law, support and resource envelope."""

    pricing_law: PricingLaw
    grid: TensorizedGrid
    quantics_layout: QuanticsLayout | None
    contract_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    training_independence_id: str = eqx.field(static=True)
    route: TensorRoute = eqx.field(static=True)
    max_ranks: tuple[int, ...] = eqx.field(static=True)
    validation_tolerance: float = eqx.field(static=True)
    reconstruction_tolerance: float = eqx.field(static=True)
    maximum_core_bytes: int = eqx.field(static=True)
    maximum_validation_points: int = eqx.field(static=True)

    def __init__(
        self,
        pricing_law: PricingLaw,
        grid: TensorizedGrid,
        /,
        *,
        contract_id: str,
        domain_id: str,
        support_id: str,
        training_independence_id: str,
        route: TensorRoute,
        max_ranks: tuple[int, ...],
        validation_tolerance: float,
        reconstruction_tolerance: float,
        maximum_core_bytes: int,
        maximum_validation_points: int,
        quantics_layout: QuanticsLayout | None = None,
    ):
        if not isinstance(pricing_law, PricingLaw):
            raise TypeError("pricing_law must be a PricingLaw.")
        if not isinstance(grid, TensorizedGrid):
            raise TypeError("grid must be a TensorizedGrid.")
        if route not in ("tt", "qtt"):
            raise ValueError("route must be 'tt' or 'qtt'.")
        if route == "qtt":
            if not isinstance(quantics_layout, QuanticsLayout):
                raise TypeError("QTT applicability requires a QuanticsLayout.")
            if quantics_layout.axis_sizes != grid.mode_sizes:
                raise ValueError("QTT layout axes must match the physical grid modes.")
            order = quantics_layout.digit_count
        else:
            if quantics_layout is not None:
                raise ValueError("A TT route must not carry a quantics layout.")
            order = len(grid.mode_sizes)
        identifiers = tuple(
            str(value)
            for value in (contract_id, domain_id, support_id, training_independence_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Tensor applicability identities must be nonempty.")
        ranks = tuple(int(rank) for rank in max_ranks)
        if len(ranks) != max(order - 1, 0) or any(rank < 1 for rank in ranks):
            raise ValueError("max_ranks must provide one positive cap per tensor cut.")
        validation = float(validation_tolerance)
        reconstruction = float(reconstruction_tolerance)
        byte_budget = int(maximum_core_bytes)
        point_budget = int(maximum_validation_points)
        if (
            not isfinite(validation)
            or validation < 0.0
            or not isfinite(reconstruction)
            or reconstruction < 0.0
            or byte_budget < 1
            or point_budget < 1
        ):
            raise ValueError("Tensor tolerances/resource budgets are invalid.")
        self.pricing_law = pricing_law
        self.grid = grid
        self.quantics_layout = quantics_layout
        (
            self.contract_id,
            self.domain_id,
            self.support_id,
            self.training_independence_id,
        ) = identifiers
        self.route = route
        self.max_ranks = ranks
        self.validation_tolerance = validation
        self.reconstruction_tolerance = reconstruction
        self.maximum_core_bytes = byte_budget
        self.maximum_validation_points = point_budget


class TensorSupportEvidence(StrictModule):
    """Typed grid-domain support audit."""

    maximum_support_violation: Array
    domain_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        maximum_support_violation: ArrayLike,
        /,
        *,
        domain_id: str,
        support_id: str,
        factor_layout_id: str,
        evidence_id: str,
        tolerance: float,
    ):
        violation = jnp.asarray(maximum_support_violation, dtype=float)
        identifiers = tuple(
            str(value) for value in (domain_id, support_id, factor_layout_id, evidence_id)
        )
        tolerance_ = float(tolerance)
        if violation.shape != () or any(not value for value in identifiers):
            raise ValueError("Tensor support evidence is malformed.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Support tolerance must be finite and nonnegative.")
        self.maximum_support_violation = violation
        self.domain_id, self.support_id, self.factor_layout_id, self.evidence_id = (
            identifiers
        )
        self.tolerance = tolerance_
        self.valid = jnp.isfinite(violation) & (violation <= tolerance_)


class TensorCausalityEvidence(StrictModule):
    """Predictability audit for tensor coordinates carrying path/history state."""

    maximum_future_dependency: Array
    filtration_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        maximum_future_dependency: ArrayLike,
        /,
        *,
        filtration_id: str,
        evidence_id: str,
        tolerance: float,
    ):
        dependency = jnp.asarray(maximum_future_dependency, dtype=float)
        filtration = str(filtration_id)
        identifier = str(evidence_id)
        tolerance_ = float(tolerance)
        if dependency.shape != () or not filtration or not identifier:
            raise ValueError("Tensor causality evidence is malformed.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Causality tolerance must be finite and nonnegative.")
        self.maximum_future_dependency = dependency
        self.filtration_id = filtration
        self.evidence_id = identifier
        self.tolerance = tolerance_
        self.valid = jnp.isfinite(dependency) & (dependency <= tolerance_)


class TensorRankEvidence(StrictModule):
    """Observed ranks, declared caps and fail-closed saturation evidence."""

    observed_ranks: tuple[int, ...] = eqx.field(static=True)
    max_ranks: tuple[int, ...] = eqx.field(static=True)
    rank_saturated: bool = eqx.field(static=True)
    ranks_within_caps: bool = eqx.field(static=True)
    base_approximation_converged: bool = eqx.field(static=True)
    reported_relative_error: Array
    reported_error_is_bound: bool = eqx.field(static=True)


class TensorResourceEvidence(StrictModule):
    core_entries: int = eqx.field(static=True)
    core_bytes: int = eqx.field(static=True)
    dense_entries: int = eqx.field(static=True)
    maximum_core_bytes: int = eqx.field(static=True)
    within_budget: bool = eqx.field(static=True)


class TensorIndependentValidation(StrictModule):
    """Explicit independent point reconstruction evidence."""

    indices: Array
    predictions: Array
    reference_values: Array
    root_mean_square_error: Array
    relative_error: Array
    maximum_absolute_error: Array
    validation_point_count: int = eqx.field(static=True)
    validation_id: str = eqx.field(static=True)
    validation_independence_id: str = eqx.field(static=True)
    training_independence_id: str = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    finite: Array
    valid: Array


class TensorValuationEvidence(StrictModule):
    rank: TensorRankEvidence
    resources: TensorResourceEvidence
    validation: TensorIndependentValidation | None
    support: TensorSupportEvidence
    causality: TensorCausalityEvidence
    evidence_binding: FinanceEvidenceBinding
    law_compatible: bool = eqx.field(static=True)
    domain_compatible: bool = eqx.field(static=True)
    complete_evidence_binding: bool = eqx.field(static=True)


class TensorValuationCandidate(StrictModule):
    """Candidate-only TT/QTT approximation; dense reconstruction stays bounded."""

    applicability: TensorValuationApplicability
    tensor: TensorTrain
    evidence: TensorValuationEvidence
    accepted: Array
    candidate_only: bool = eqx.field(static=True, default=True)

    def reconstruct(self, /, *, max_entries: int) -> Array:
        """Materialize in physical grid ordering under an explicit entry budget."""
        if self.applicability.route == "tt":
            return self.tensor.to_dense(max_entries=max_entries)
        layout = self.applicability.quantics_layout
        if layout is None:
            raise ValueError("QTT reconstruction requires its declared QuanticsLayout.")
        digitized = self.tensor.to_dense(max_entries=max_entries)
        return layout.untensorize(digitized)


def _tensor_and_base_evidence(
    approximation: TensorApproximation, /
) -> tuple[TensorTrain, bool, Array, bool]:
    if isinstance(approximation, TensorTrainCompressionResult):
        evidence = approximation.evidence
        return (
            approximation.tensor,
            evidence.tolerance_met,
            evidence.relative_error_bound,
            True,
        )
    if isinstance(approximation, TTCrossResult):
        return (
            approximation.tensor,
            approximation.converged,
            approximation.evidence.holdout_relative_error_estimator,
            False,
        )
    raise TypeError(
        "approximation must be TensorTrainCompressionResult or TTCrossResult; "
        "a bare tensor has no compression/convergence evidence."
    )


def tensor_rank_evidence(
    applicability: TensorValuationApplicability,
    approximation: TensorApproximation,
    /,
) -> TensorRankEvidence:
    if not isinstance(applicability, TensorValuationApplicability):
        raise TypeError("applicability must be TensorValuationApplicability.")
    tensor, converged, reported, is_bound = _tensor_and_base_evidence(approximation)
    if len(tensor.ranks) != len(applicability.max_ranks):
        raise ValueError("Tensor order does not match the declared rank-cap layout.")
    within = all(
        rank <= cap
        for rank, cap in zip(tensor.ranks, applicability.max_ranks, strict=True)
    )
    saturated = any(
        rank >= cap
        for rank, cap in zip(tensor.ranks, applicability.max_ranks, strict=True)
    )
    return TensorRankEvidence(
        tensor.ranks,
        applicability.max_ranks,
        saturated,
        within,
        converged,
        jnp.asarray(reported),
        is_bound,
    )


def tensor_independent_validation(
    applicability: TensorValuationApplicability,
    approximation: TensorApproximation,
    indices: ArrayLike,
    reference_values: ArrayLike,
    /,
    *,
    validation_id: str,
    validation_independence_id: str,
) -> TensorIndependentValidation:
    """Evaluate explicit held-out physical-grid indices and retain reconstruction."""
    if not isinstance(applicability, TensorValuationApplicability):
        raise TypeError("applicability must be TensorValuationApplicability.")
    tensor, _, _, _ = _tensor_and_base_evidence(approximation)
    points = jnp.asarray(indices, dtype=jnp.int32)
    reference = jnp.asarray(reference_values)
    if (
        points.ndim != 2
        or points.shape[0] < 1
        or points.shape[1] != applicability.grid.dimension
    ):
        raise ValueError("Validation indices need nonempty (point, physical_axis) shape.")
    if points.shape[0] > applicability.maximum_validation_points:
        raise ValueError("Validation point count exceeds the declared resource budget.")
    limits = jnp.asarray(applicability.grid.mode_sizes, dtype=jnp.int32)
    if bool(jnp.any((points < 0) | (points >= limits))):
        raise ValueError("Validation index lies outside the tensorized grid.")
    if reference.shape != (points.shape[0],):
        raise ValueError("reference_values must contain one scalar per validation point.")
    if applicability.route == "tt":
        if tensor.mode_sizes != applicability.grid.mode_sizes:
            raise ValueError("TT modes do not match the declared physical grid.")
        predictions = tensor.evaluate(points)
    else:
        layout = applicability.quantics_layout
        if layout is None or tensor.mode_sizes != layout.digit_mode_sizes:
            raise ValueError("QTT modes do not match the declared digit layout.")
        predictions = qtt_evaluate(tensor, layout, points)
    residual = predictions - reference
    rmse = jnp.sqrt(jnp.mean(jnp.abs(residual) ** 2))
    scale = jnp.sqrt(jnp.mean(jnp.abs(reference) ** 2))
    relative = rmse / jnp.where(scale > 0.0, scale, 1.0)
    maximum = jnp.max(jnp.abs(residual))
    identifier = str(validation_id)
    independence = str(validation_independence_id)
    if not identifier or not independence:
        raise ValueError("Validation IDs must be nonempty.")
    independent = independence != applicability.training_independence_id
    finite = (
        jnp.all(jnp.isfinite(predictions))
        & jnp.all(jnp.isfinite(reference))
        & jnp.isfinite(relative)
    )
    valid = (
        independent
        & finite
        & (relative <= applicability.validation_tolerance)
        & (maximum <= applicability.reconstruction_tolerance)
    )
    return TensorIndependentValidation(
        points,
        predictions,
        reference,
        rmse,
        relative,
        maximum,
        int(points.shape[0]),
        identifier,
        independence,
        applicability.training_independence_id,
        independent,
        finite,
        valid,
    )


def _resource_evidence(
    applicability: TensorValuationApplicability, tensor: TensorTrain, /
) -> TensorResourceEvidence:
    entries = sum(int(core.size) for core in tensor.cores)
    bytes_ = sum(int(core.size * core.dtype.itemsize) for core in tensor.cores)
    dense_entries = applicability.grid.point_count
    return TensorResourceEvidence(
        entries,
        bytes_,
        dense_entries,
        applicability.maximum_core_bytes,
        bytes_ <= applicability.maximum_core_bytes,
    )


def _complete_binding(binding: FinanceEvidenceBinding, /) -> bool:
    return bool(
        binding.data_evidence_ids
        and binding.model_evidence_ids
        and binding.numerical_evidence_ids
        and binding.use_evidence_ids
    )


def assess_tensor_valuation_candidate(
    applicability: TensorValuationApplicability,
    approximation: TensorApproximation,
    pricing_law: PricingLaw,
    validation: TensorIndependentValidation | None,
    support: TensorSupportEvidence,
    causality: TensorCausalityEvidence,
    evidence_binding: FinanceEvidenceBinding,
    /,
) -> TensorValuationCandidate:
    """Fail closed on absent holdout, rank saturation, resources, law or domain."""
    if not isinstance(applicability, TensorValuationApplicability):
        raise TypeError("applicability must be TensorValuationApplicability.")
    tensor, _, _, _ = _tensor_and_base_evidence(approximation)
    if not isinstance(pricing_law, PricingLaw):
        raise TypeError("pricing_law must be a PricingLaw.")
    if validation is not None and not isinstance(validation, TensorIndependentValidation):
        raise TypeError("validation has the wrong evidence type.")
    if not isinstance(support, TensorSupportEvidence):
        raise TypeError("support must be TensorSupportEvidence.")
    if not isinstance(causality, TensorCausalityEvidence):
        raise TypeError("causality must be TensorCausalityEvidence.")
    if not isinstance(evidence_binding, FinanceEvidenceBinding):
        raise TypeError("evidence_binding must be a FinanceEvidenceBinding.")
    rank = tensor_rank_evidence(applicability, approximation)
    resources = _resource_evidence(applicability, tensor)
    expected = applicability.pricing_law
    law_compatible = (
        pricing_law.law_id == expected.law_id
        and pricing_law.measure_id == expected.measure_id
        and pricing_law.numeraire_id == expected.numeraire_id
        and pricing_law.collateral_convention_id == expected.collateral_convention_id
        and pricing_law.factor_layout_id == expected.factor_layout_id
        and pricing_law.filtration_id == expected.filtration_id
    )
    expected_modes = (
        applicability.grid.mode_sizes
        if applicability.route == "tt"
        else applicability.quantics_layout.digit_mode_sizes
    )
    domain_compatible = (
        tensor.mode_sizes == expected_modes
        and support.domain_id == applicability.domain_id
        and support.support_id == applicability.support_id
        and support.factor_layout_id == pricing_law.factor_layout_id
        and causality.filtration_id == pricing_law.filtration_id
    )
    complete = _complete_binding(evidence_binding)
    accepted = (
        law_compatible
        & domain_compatible
        & support.valid
        & causality.valid
        & rank.ranks_within_caps
        & (not rank.rank_saturated)
        & rank.base_approximation_converged
        & (rank.reported_relative_error <= applicability.validation_tolerance)
        & resources.within_budget
        & (False if validation is None else validation.valid)
        & complete
    )
    evidence = TensorValuationEvidence(
        rank,
        resources,
        validation,
        support,
        causality,
        evidence_binding,
        law_compatible,
        domain_compatible,
        complete,
    )
    return TensorValuationCandidate(
        applicability,
        tensor,
        evidence,
        accepted,
        True,
    )


__all__ = [
    "TensorApproximation",
    "TensorCausalityEvidence",
    "TensorIndependentValidation",
    "TensorRankEvidence",
    "TensorResourceEvidence",
    "TensorRoute",
    "TensorSupportEvidence",
    "TensorValuationApplicability",
    "TensorValuationCandidate",
    "TensorValuationEvidence",
    "assess_tensor_valuation_candidate",
    "tensor_independent_validation",
    "tensor_rank_evidence",
]
