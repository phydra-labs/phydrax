#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...ml._batch import MLBatch
from ...ml._contracts import FitResult
from ...ml.covariance._estimators import (
    EmpiricalCovariance,
    FactorCovariance as MLFactorCovariance,
    LedoitWolfCovariance,
    OASCovariance,
)
from ...ml.covariance._random_matrix import (
    clean_covariance_spectrum,
    MarchenkoPasturDiagnostics,
    RandomMatrixCleaningResult,
)
from ...uq._covariance import DenseCovariance
from ..core import FinanceEvidenceBinding, PhysicalLaw
from ._returns import ReturnResult


CovarianceMethod: TypeAlias = Literal["sample", "ledoit-wolf", "oas"]


class CovarianceDefinition(StrictModule):
    method: CovarianceMethod = eqx.field(static=True)
    correction: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: CovarianceMethod = "ledoit-wolf",
        correction: float = 1.0,
        regularization: float = 1e-8,
    ):
        if method not in ("sample", "ledoit-wolf", "oas"):
            raise ValueError("method must be sample, ledoit-wolf, or oas.")
        correction_ = float(correction)
        regularization_ = float(regularization)
        if not jnp.isfinite(correction_) or correction_ < 0.0:
            raise ValueError("correction must be finite and nonnegative.")
        if not jnp.isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("regularization must be finite and nonnegative.")
        self.method = method
        self.correction = correction_
        self.regularization = regularization_
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-covariance-definition",
                "method": method,
                "correction": correction_,
                "regularization": regularization_,
            }
        )


class FactorModelDefinition(StrictModule):
    rank: int = eqx.field(static=True)
    correction: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        rank: int,
        /,
        *,
        correction: float = 1.0,
        regularization: float = 1e-8,
    ):
        rank_ = int(rank)
        correction_ = float(correction)
        regularization_ = float(regularization)
        if rank_ < 1:
            raise ValueError("rank must be positive.")
        if correction_ < 0.0 or regularization_ < 0.0:
            raise ValueError("correction and regularization must be nonnegative.")
        self.rank = rank_
        self.correction = correction_
        self.regularization = regularization_
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-factor-model-definition",
                "rank": rank_,
                "correction": correction_,
                "regularization": regularization_,
            }
        )


class RMTCleaningDefinition(StrictModule):
    replacement: Literal["bulk-mean", "upper-edge", "hard-floor"] = eqx.field(static=True)
    preserve_trace: bool = eqx.field(static=True)
    eigenvalue_floor: float = eqx.field(static=True)
    edge_tolerance: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        replacement: Literal["bulk-mean", "upper-edge", "hard-floor"] = "bulk-mean",
        preserve_trace: bool = True,
        eigenvalue_floor: float = 0.0,
        edge_tolerance: float = 0.0,
    ):
        if replacement not in ("bulk-mean", "upper-edge", "hard-floor"):
            raise ValueError("unsupported RMT replacement rule.")
        floor = float(eigenvalue_floor)
        tolerance = float(edge_tolerance)
        if floor < 0.0 or tolerance < 0.0:
            raise ValueError("RMT floor and edge tolerance must be nonnegative.")
        self.replacement = replacement
        self.preserve_trace = bool(preserve_trace)
        self.eigenvalue_floor = floor
        self.edge_tolerance = tolerance
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-rmt-cleaning-definition",
                "replacement": replacement,
                "preserve_trace": bool(preserve_trace),
                "eigenvalue_floor": floor,
                "edge_tolerance": tolerance,
            }
        )


class CovariancePlan(StrictModule):
    values: Array
    feature_mask: Array
    sample_mask: Array
    effective_sample_count: Array
    data_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    asset_count: int = eqx.field(static=True)
    observation_capacity: int = eqx.field(static=True)


class CovarianceFit(StrictModule):
    covariance: DenseCovariance
    mean: Array
    ml_fit: FitResult
    sample_covariance: Array
    shrinkage_target: Array
    shrinkage_intensity: Array
    eigenvalues: Array
    effective_rank: Array
    condition_number: Array
    effective_sample_count: Array
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class CovarianceResult(StrictModule):
    covariance: DenseCovariance
    mean: Array
    eigenvalues: Array
    rank: Array
    condition_number: Array
    fit: CovarianceFit


class FactorModelFit(StrictModule):
    covariance: DenseCovariance
    mean: Array
    loadings: Array
    idiosyncratic_variance: Array
    explained_variance: Array
    ml_fit: FitResult
    effective_sample_count: Array
    condition_number: Array
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class RMTCleaningResult(StrictModule):
    covariance: DenseCovariance
    cleaning: RandomMatrixCleaningResult
    diagnostics: MarchenkoPasturDiagnostics
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    source_covariance_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


def _binding(data: str, model: str, numerical: str) -> FinanceEvidenceBinding:
    return FinanceEvidenceBinding(
        (canonical_fingerprint({"kind": "covariance-data-evidence", "data": data}),),
        (canonical_fingerprint({"kind": "covariance-model-evidence", "model": model}),),
        (
            canonical_fingerprint(
                {"kind": "covariance-numerical-evidence", "route": numerical}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "covariance-use-evidence", "trade_emission": False}
            ),
        ),
    )


def prepare_covariance(returns: ReturnResult, /) -> CovariancePlan:
    if not isinstance(returns, ReturnResult):
        raise TypeError("returns must be a ReturnResult.")
    values = returns.values.T
    feature_mask = returns.valid_mask.T
    sample_mask = jnp.all(feature_mask, axis=-1)
    effective = jnp.sum(sample_mask).astype(jnp.int32)
    plan_id = canonical_fingerprint(
        {
            "kind": "finance-covariance-plan",
            "returns": returns.result_id,
            "shape": tuple(int(size) for size in values.shape),
        }
    )
    return CovariancePlan(
        values=values,
        feature_mask=feature_mask,
        sample_mask=sample_mask,
        effective_sample_count=effective,
        data_id=returns.result_id,
        plan_id=plan_id,
        asset_count=int(values.shape[-1]),
        observation_capacity=int(values.shape[0]),
    )


def fit_covariance(
    returns: ReturnResult,
    definition: CovarianceDefinition,
    law: PhysicalLaw,
    /,
) -> CovarianceFit:
    """Fit native covariance estimators on complete masked return rows."""

    if not isinstance(definition, CovarianceDefinition):
        raise TypeError("definition must be a CovarianceDefinition.")
    if not isinstance(law, PhysicalLaw):
        raise TypeError("covariance estimation requires a PhysicalLaw.")
    plan = prepare_covariance(returns)
    batch = MLBatch(
        plan.values,
        feature_mask=plan.feature_mask,
        sample_mask=plan.sample_mask,
    )
    if definition.method == "sample":
        recipe = EmpiricalCovariance(
            correction=definition.correction,
            regularization=definition.regularization,
        )
    elif definition.method == "ledoit-wolf":
        recipe = LedoitWolfCovariance(
            correction=definition.correction,
            regularization=definition.regularization,
        )
    else:
        recipe = OASCovariance(
            correction=definition.correction,
            regularization=definition.regularization,
        )
    fit = recipe.fit_batch(batch)
    covariance_model = fit.model.model
    matrix = covariance_model.covariance
    values = jnp.linalg.eigvalsh(matrix)
    safe_values = jnp.where(plan.sample_mask[:, None], plan.values, 0.0)
    sample_count = jnp.maximum(plan.effective_sample_count, 1)
    sample_mean = jnp.sum(safe_values, axis=0) / sample_count
    centered = jnp.where(
        plan.sample_mask[:, None], plan.values - sample_mean[None, :], 0.0
    )
    denominator = jnp.maximum(
        plan.effective_sample_count.astype(plan.values.dtype) - definition.correction,
        1.0,
    )
    sample_covariance = centered.T @ centered / denominator
    target_scale = jnp.trace(sample_covariance) / plan.asset_count
    target = target_scale * jnp.eye(plan.asset_count, dtype=plan.values.dtype)
    direction = target - sample_covariance
    direction_norm = jnp.sum(jnp.square(jnp.abs(direction)))
    intensity = jnp.where(
        definition.method == "sample",
        0.0,
        jnp.clip(
            jnp.sum(jnp.real((matrix - sample_covariance) * direction))
            / jnp.maximum(direction_norm, jnp.finfo(plan.values.dtype).tiny),
            0.0,
            1.0,
        ),
    )
    return CovarianceFit(
        covariance=DenseCovariance(matrix),
        mean=covariance_model.mean,
        ml_fit=fit,
        sample_covariance=sample_covariance,
        shrinkage_target=target,
        shrinkage_intensity=intensity,
        eigenvalues=values,
        effective_rank=fit.diagnostics.rank,
        condition_number=fit.diagnostics.condition,
        effective_sample_count=fit.diagnostics.effective_samples,
        evidence=_binding(returns.result_id, definition.definition_id, definition.method),
        data_id=returns.result_id,
        definition_id=definition.definition_id,
        law_id=law.law_id,
    )


def summarize_covariance(fit: CovarianceFit, /) -> CovarianceResult:
    if not isinstance(fit, CovarianceFit):
        raise TypeError("fit must be a CovarianceFit.")
    return CovarianceResult(
        covariance=fit.covariance,
        mean=fit.mean,
        eigenvalues=fit.eigenvalues,
        rank=fit.effective_rank,
        condition_number=fit.condition_number,
        fit=fit,
    )


def fit_factor_model(
    returns: ReturnResult,
    definition: FactorModelDefinition,
    law: PhysicalLaw,
    /,
) -> FactorModelFit:
    """Fit the native factor-plus-diagonal covariance representation."""

    if not isinstance(definition, FactorModelDefinition):
        raise TypeError("definition must be a FactorModelDefinition.")
    if not isinstance(law, PhysicalLaw):
        raise TypeError("factor estimation requires a PhysicalLaw.")
    plan = prepare_covariance(returns)
    if definition.rank > plan.asset_count:
        raise ValueError("factor rank cannot exceed the asset count.")
    batch = MLBatch(
        plan.values,
        feature_mask=plan.feature_mask,
        sample_mask=plan.sample_mask,
    )
    fit = MLFactorCovariance(
        definition.rank,
        correction=definition.correction,
        regularization=definition.regularization,
    ).fit_batch(batch)
    covariance_model = fit.model.model
    loadings = covariance_model.factor_loadings[:, ::-1]
    pivot = jnp.argmax(jnp.abs(loadings), axis=0)
    signs = jnp.sign(loadings[pivot, jnp.arange(definition.rank)])
    loadings = loadings * jnp.where(signs == 0.0, 1.0, signs)[None, :]
    factor_variance = jnp.sum(jnp.square(jnp.abs(loadings)), axis=0)
    return FactorModelFit(
        covariance=DenseCovariance(covariance_model.covariance),
        mean=covariance_model.mean,
        loadings=loadings,
        idiosyncratic_variance=covariance_model.diagonal,
        explained_variance=factor_variance,
        ml_fit=fit,
        effective_sample_count=fit.diagnostics.effective_samples,
        condition_number=fit.diagnostics.condition,
        evidence=_binding(
            returns.result_id, definition.definition_id, "native-factor-covariance"
        ),
        data_id=returns.result_id,
        definition_id=definition.definition_id,
        law_id=law.law_id,
    )


def clean_covariance_rmt(
    fit: CovarianceFit,
    definition: RMTCleaningDefinition,
    law: PhysicalLaw,
    /,
) -> RMTCleaningResult:
    """Clean a fitted covariance with the generic random-matrix substrate."""

    if not isinstance(fit, CovarianceFit):
        raise TypeError("fit must be a CovarianceFit.")
    if not isinstance(definition, RMTCleaningDefinition):
        raise TypeError("definition must be an RMTCleaningDefinition.")
    if not isinstance(law, PhysicalLaw) or law.law_id != fit.law_id:
        raise ValueError("RMT cleaning must preserve the fitted PhysicalLaw binding.")
    effective_samples = float(fit.effective_sample_count)
    if not math.isfinite(effective_samples) or effective_samples < 2.0:
        raise ValueError("RMT cleaning requires at least two effective observations.")
    observations = int(effective_samples)
    cleaning = clean_covariance_spectrum(
        fit.covariance.matrix,
        observations,
        replacement=definition.replacement,
        preserve_trace=definition.preserve_trace,
        eigenvalue_floor=definition.eigenvalue_floor,
        edge_tolerance=definition.edge_tolerance,
    )
    source_id = canonical_fingerprint(
        {
            "kind": "fitted-covariance",
            "data": fit.data_id,
            "definition": fit.definition_id,
            "law": fit.law_id,
        }
    )
    return RMTCleaningResult(
        covariance=DenseCovariance(cleaning.covariance),
        cleaning=cleaning,
        diagnostics=cleaning.diagnostics,
        evidence=_binding(
            source_id, definition.definition_id, "marchenko-pastur-cleaning"
        ),
        source_covariance_id=source_id,
        definition_id=definition.definition_id,
        law_id=law.law_id,
    )


__all__ = [
    "CovarianceDefinition",
    "CovarianceFit",
    "CovarianceMethod",
    "CovariancePlan",
    "CovarianceResult",
    "FactorModelDefinition",
    "FactorModelFit",
    "RMTCleaningDefinition",
    "RMTCleaningResult",
    "clean_covariance_rmt",
    "fit_covariance",
    "fit_factor_model",
    "prepare_covariance",
    "summarize_covariance",
]
