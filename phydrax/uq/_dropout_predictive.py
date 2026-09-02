#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..domain import DomainFunction
from ..nn.layers._dropout import Dropout
from ._predictive import PredictionInterval, PredictiveField, SampleAxis


MCDropoutCalibrationMethod = Literal[
    "gaussian_scale", "normalized_conformal", "functional_conformal"
]


class MCDropoutCalibrationEvidence(StrictModule):
    """Held-out evidence for one calibrated MC-dropout interval."""

    nominal_coverage: Array
    empirical_heldout_coverage: Array
    mean_width: Array
    calibration_count: Array
    valid: Array
    approximation: str = eqx.field(static=True)
    method: MCDropoutCalibrationMethod = eqx.field(static=True)
    split_identity: str = eqx.field(static=True)
    draw_count: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        nominal_coverage: ArrayLike,
        empirical_heldout_coverage: ArrayLike,
        mean_width: ArrayLike,
        calibration_count: int,
        valid: ArrayLike,
        method: MCDropoutCalibrationMethod,
        split_identity: str,
        draw_count: int,
    ):
        self.nominal_coverage = jnp.asarray(nominal_coverage, dtype=float).reshape(())
        self.empirical_heldout_coverage = jnp.asarray(
            empirical_heldout_coverage, dtype=float
        ).reshape(())
        self.mean_width = jnp.asarray(mean_width, dtype=float).reshape(())
        self.calibration_count = jnp.asarray(calibration_count, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool).reshape(())
        self.approximation = "mc_dropout_heldout_calibrated"
        self.method = method
        self.split_identity = split_identity
        self.draw_count = int(draw_count)


class MCDropoutCalibration(StrictModule):
    """Immutable held-out scale or split-conformal MC-dropout calibration."""

    coefficient: Array
    nominal_coverage: float = eqx.field(static=True)
    method: MCDropoutCalibrationMethod = eqx.field(static=True)
    split_identity: str = eqx.field(static=True)
    case_dim: str | None = eqx.field(static=True)
    evidence: MCDropoutCalibrationEvidence

    def __init__(
        self,
        coefficient: ArrayLike,
        /,
        *,
        nominal_coverage: float,
        method: MCDropoutCalibrationMethod,
        split_identity: str,
        case_dim: str | None,
        evidence: MCDropoutCalibrationEvidence,
    ):
        coefficient_ = jnp.asarray(coefficient, dtype=float).reshape(())
        if not bool(jnp.isfinite(coefficient_)) or not bool(coefficient_ > 0.0):
            raise ValueError("Calibration coefficient must be finite and positive.")
        coverage = float(nominal_coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("nominal_coverage must lie strictly between zero and one.")
        if method not in (
            "gaussian_scale",
            "normalized_conformal",
            "functional_conformal",
        ):
            raise ValueError("Unknown MC-dropout calibration method.")
        identity = str(split_identity)
        if not identity:
            raise ValueError("split_identity must be non-empty.")
        if method == "functional_conformal" and not case_dim:
            raise ValueError("functional_conformal requires case_dim.")
        if not isinstance(evidence, MCDropoutCalibrationEvidence):
            raise TypeError("evidence must be MCDropoutCalibrationEvidence.")
        self.coefficient = coefficient_
        self.nominal_coverage = coverage
        self.method = method
        self.split_identity = identity
        self.case_dim = case_dim
        self.evidence = evidence

    @classmethod
    def fit(
        cls,
        predictive: Any,
        target: cx.Field | ArrayLike,
        /,
        *,
        nominal_coverage: float,
        method: MCDropoutCalibrationMethod,
        split_identity: str,
        mask: cx.Field | ArrayLike | None = None,
        weights: cx.Field | ArrayLike | None = None,
        case_dim: str | None = None,
    ) -> MCDropoutCalibration:
        """Fit outside transformed execution on a caller-declared held-out split."""
        from ._operator import _output_mask, OperatorPredictiveField

        if isinstance(predictive, OperatorPredictiveField):
            operator_mask = _output_mask(
                predictive.query, predictive.output_spec, predictive.case_shape
            )
            mask = (
                operator_mask
                if mask is None
                else operator_mask
                & jnp.asarray(
                    mask.data if isinstance(mask, cx.Field) else mask,
                    dtype=bool,
                )
            )
            predictive = predictive.predictive
        if not isinstance(predictive, PredictiveField):
            raise TypeError(
                "predictive must be PredictiveField or OperatorPredictiveField."
            )
        coverage = float(nominal_coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("nominal_coverage must lie strictly between zero and one.")
        identity = str(split_identity)
        if not identity:
            raise ValueError("split_identity must be non-empty.")
        center_field = predictive.mean(sources="epistemic")
        scale_field = predictive.std(sources="epistemic")
        center = jnp.asarray(center_field.data)
        scale = jnp.asarray(scale_field.data)
        target_array = _aligned_array(target, center_field, "target")
        active = (
            jnp.ones(center.shape, dtype=bool)
            if mask is None
            else _aligned_array(mask, center_field, "mask").astype(bool)
        )
        active = active & jnp.isfinite(center) & jnp.isfinite(target_array)
        count = int(jnp.sum(active))
        if count == 0:
            raise ValueError("MC-dropout calibration split has no active finite values.")
        if bool(jnp.any((~jnp.isfinite(scale) | (scale <= 0.0)) & active)):
            raise ValueError(
                "Active MC-dropout calibration scales must be finite and positive."
            )
        normalized = jnp.where(active, jnp.abs(target_array - center) / scale, 0.0)
        weight_array = (
            jnp.ones(center.shape, dtype=float)
            if weights is None
            else _aligned_array(weights, center_field, "weights").astype(float)
        )
        if bool(jnp.any((~jnp.isfinite(weight_array) | (weight_array <= 0.0)) & active)):
            raise ValueError("Active calibration weights must be finite and positive.")
        if method == "gaussian_scale":
            effective = jnp.where(active, weight_array, 0.0)
            coefficient = jnp.sqrt(
                jnp.sum(effective * normalized**2) / jnp.sum(effective)
            )
            width_coefficient = coefficient * jsp.special.ndtri(
                jnp.asarray(0.5 * (1.0 + coverage))
            )
        elif method == "normalized_conformal":
            if weights is not None:
                raise ValueError(
                    "Normalized split conformal currently requires unweighted cases."
                )
            coefficient = _finite_sample_quantile(normalized[active], coverage)
            width_coefficient = coefficient
        elif method == "functional_conformal":
            if weights is not None:
                raise ValueError(
                    "Functional max-score conformal does not accept element weights."
                )
            if case_dim is None or case_dim not in center_field.dims:
                raise ValueError("case_dim must name one calibration field dimension.")
            axis = center_field.dims.index(case_dim)
            residual = jnp.moveaxis(normalized, axis, 0)
            active_by_case = jnp.moveaxis(active, axis, 0)
            flat = residual.reshape((residual.shape[0], -1))
            flat_active = active_by_case.reshape((active_by_case.shape[0], -1))
            valid_case = jnp.any(flat_active, axis=1)
            if not bool(jnp.all(valid_case)):
                raise ValueError(
                    "Every functional calibration case needs active support."
                )
            case_scores = jnp.max(jnp.where(flat_active, flat, -jnp.inf), axis=1)
            coefficient = _finite_sample_quantile(case_scores, coverage)
            width_coefficient = coefficient
            count = int(case_scores.shape[0])
        else:
            raise ValueError("Unknown MC-dropout calibration method.")
        lower = center - width_coefficient * scale
        upper = center + width_coefficient * scale
        covered = (target_array >= lower) & (target_array <= upper)
        if method == "functional_conformal":
            covered_by_case = jnp.moveaxis(covered, axis, 0).reshape(flat_active.shape)
            empirical = jnp.mean(jnp.all((~flat_active) | covered_by_case, axis=1))
        else:
            empirical = jnp.sum(jnp.where(active, covered, False)) / jnp.sum(active)
        mean_width = jnp.sum(jnp.where(active, upper - lower, 0.0)) / jnp.sum(active)
        draw_count = _epistemic_draw_count(predictive)
        evidence = MCDropoutCalibrationEvidence(
            nominal_coverage=coverage,
            empirical_heldout_coverage=empirical,
            mean_width=mean_width,
            calibration_count=count,
            valid=jnp.isfinite(coefficient) & (coefficient > 0.0),
            method=method,
            split_identity=identity,
            draw_count=draw_count,
        )
        return cls(
            coefficient,
            nominal_coverage=coverage,
            method=method,
            split_identity=identity,
            case_dim=case_dim,
            evidence=evidence,
        )

    def interval(self, predictive: Any, /) -> Any:
        """Apply the frozen held-out calibration to MC-dropout summaries."""
        from ..nn.operator.data import OperatorPrediction
        from ._operator import OperatorPredictionInterval, OperatorPredictiveField

        if isinstance(predictive, OperatorPredictiveField):
            generic = self.interval(predictive.predictive)
            lower = OperatorPrediction.from_field(
                predictive.field_name,
                generic.lower.data,
                predictive.query_name,
                predictive.query,
                spec=predictive.output_spec,
                case_axes=predictive.case_axes,
                case_shape=predictive.case_shape,
            )
            upper = OperatorPrediction.from_field(
                predictive.field_name,
                generic.upper.data,
                predictive.query_name,
                predictive.query,
                spec=predictive.output_spec,
                case_axes=predictive.case_axes,
                case_shape=predictive.case_shape,
            )
            return OperatorPredictionInterval(
                lower,
                upper,
                nominal_coverage=generic.nominal_coverage,
                simultaneous=generic.simultaneous,
                calibrated=True,
            )
        if not isinstance(predictive, PredictiveField):
            raise TypeError("predictive must be a PredictiveField.")
        center = predictive.mean(sources="epistemic")
        scale = predictive.std(sources="epistemic")
        scale_array = jnp.asarray(scale.data)
        if bool(jnp.any(~jnp.isfinite(scale_array))) or bool(jnp.any(scale_array <= 0.0)):
            raise ValueError("Applied MC-dropout scales must be finite and positive.")
        if self.method == "gaussian_scale":
            quantile = jsp.special.ndtri(jnp.asarray(0.5 * (1.0 + self.nominal_coverage)))
            width = quantile * self.coefficient * scale_array
        else:
            width = self.coefficient * scale_array
        center_array = jnp.asarray(center.data)
        return PredictionInterval(
            cx.Field(center_array - width, dims=center.dims),
            cx.Field(center_array + width, dims=center.dims),
            nominal_coverage=self.nominal_coverage,
            simultaneous=self.method == "functional_conformal",
            calibrated=True,
        )


def sample_mc_dropout_predictive(
    function: DomainFunction,
    points: Any,
    /,
    *,
    key: Array,
    num_draws: int,
    draw_batch_size: int | None = None,
    draw_dim: str = "__phydra_uq_draw",
    valid_policy: Literal["record", "raise"] = "record",
    **kwargs: Any,
) -> PredictiveField:
    """Draw coherent whole-function MC-dropout evaluations.

    Draws are uncalibrated diagnostics until consumed by
    :class:`MCDropoutCalibration`; they are not posterior samples.
    """
    if not isinstance(function, DomainFunction):
        raise TypeError("function must be a DomainFunction.")
    dropout_leaves = tuple(
        leaf
        for leaf in jax.tree_util.tree_leaves(
            function.func, is_leaf=lambda value: isinstance(value, Dropout)
        )
        if isinstance(leaf, Dropout)
    )
    if not dropout_leaves:
        raise ValueError("MC-dropout prediction requires at least one Dropout leaf.")
    if any(leaf.inference for leaf in dropout_leaves):
        raise ValueError("MC-dropout prediction rejects inference-mode Dropout leaves.")
    if not any(leaf.p > 0.0 for leaf in dropout_leaves):
        raise ValueError("MC-dropout prediction requires an active nonzero dropout rate.")
    draws = int(num_draws)
    if draws < 2:
        raise ValueError("num_draws must be at least two.")
    chunk = draws if draw_batch_size is None else int(draw_batch_size)
    if chunk <= 0:
        raise ValueError("draw_batch_size must be positive.")
    if not isinstance(draw_dim, str) or not draw_dim:
        raise ValueError("draw_dim must be non-empty.")
    if valid_policy not in ("record", "raise"):
        raise ValueError("valid_policy must be 'record' or 'raise'.")
    address = SampleAddress(
        "uq.mc-dropout", "function-draw", target=draw_dim, role="dropout"
    )
    values: list[Array] = []
    template: cx.Field | None = None
    for start in range(0, draws, chunk):
        for draw in range(start, min(start + chunk, draws)):
            field = function(points, key=derive_key(key, address, draw), **kwargs)
            if template is None:
                template = field
            elif field.dims != template.dims or field.data.shape != template.data.shape:
                raise ValueError("MC-dropout draws changed field geometry.")
            values.append(jnp.asarray(field.data))
    assert template is not None
    stacked = jnp.stack(values, axis=0)
    finite = jnp.all(jnp.isfinite(stacked).reshape((draws, -1)), axis=1)
    if valid_policy == "raise" and not bool(jnp.all(finite)):
        raise FloatingPointError("MC-dropout produced a nonfinite whole-function draw.")
    return PredictiveField(
        cx.Field(stacked, dims=(draw_dim, *template.dims)),
        (SampleAxis(draw_dim, "epistemic"),),
        valid=cx.Field(finite, dims=(draw_dim,)),
    )


def _aligned_array(value: cx.Field | ArrayLike, template: cx.Field, name: str) -> Array:
    if isinstance(value, cx.Field):
        if value.dims != template.dims or value.data.shape != template.data.shape:
            raise ValueError(f"{name} field must match predictive summary geometry.")
        return jnp.asarray(value.data)
    array = jnp.asarray(value)
    if array.shape != template.data.shape:
        raise ValueError(f"{name} must match predictive summary shape.")
    return array


def _finite_sample_quantile(scores: Array, coverage: float, /) -> Array:
    count = int(scores.shape[0])
    if count <= 0:
        raise ValueError("Conformal calibration requires at least one score.")
    rank = min(count, int(math.ceil((count + 1) * coverage)))
    return jnp.sort(scores)[rank - 1]


def _epistemic_draw_count(predictive: PredictiveField, /) -> int:
    axes = tuple(axis for axis in predictive.sample_axes if axis.source == "epistemic")
    if len(axes) != 1:
        raise ValueError(
            "MC-dropout calibration requires exactly one epistemic draw axis."
        )
    return int(predictive.samples.named_shape[axes[0].dim])


__all__ = [
    "MCDropoutCalibration",
    "MCDropoutCalibrationEvidence",
    "MCDropoutCalibrationMethod",
    "sample_mc_dropout_predictive",
]
