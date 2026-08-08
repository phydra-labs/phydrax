#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._strict import StrictModule
from ..nn.operator.data import OperatorPrediction
from ._conformal import FunctionalConformal
from ._operator import (
    _output_mask,
    _output_weights,
    _physical_dims,
    _select_prediction_field,
    OperatorPredictionInterval,
)
from ._operator_metrics import _operator_target_values


class OperatorFunctionalConformal(StrictModule):
    """Whole-physical-case conformal calibration for operator predictions."""

    calibrator: FunctionalConformal
    case_axis: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)

    def __init__(
        self,
        calibrator: FunctionalConformal,
        /,
        *,
        case_axis: str,
        field_name: str,
    ):
        if not isinstance(calibrator, FunctionalConformal):
            raise TypeError("calibrator must be a FunctionalConformal.")
        axis = str(case_axis)
        name = str(field_name)
        if not axis or not name:
            raise ValueError("case_axis and field_name must be non-empty.")
        self.calibrator = calibrator
        self.case_axis = axis
        self.field_name = name

    @classmethod
    def calibrate(
        cls,
        center: OperatorPrediction,
        target: ArrayLike | OperatorPrediction,
        /,
        *,
        alpha: float,
        field_name: str,
        case_axis: str | None = None,
        scale: OperatorPrediction | None = None,
        score: Literal["max", "l2"] = "max",
        min_scale: float = 1e-8,
    ) -> OperatorFunctionalConformal:
        if not isinstance(center, OperatorPrediction):
            raise TypeError("center must be an OperatorPrediction.")
        selected_name, center_field, query = _select_prediction_field(
            center,
            field_name,
        )
        selected_axis = _calibration_case_axis(center, case_axis)
        physical_dims = _physical_dims(query, center_field.spec, center.case_axes)
        center_values = _operator_target_values(
            center,
            query=query,
            output_spec=center_field.spec,
            case_axes=center.case_axes,
            case_shape=center.case_shape,
            field_name=selected_name,
        )
        target_values = _operator_target_values(
            target,
            query=query,
            output_spec=center_field.spec,
            case_axes=center.case_axes,
            case_shape=center.case_shape,
            field_name=selected_name,
        )
        mask = _output_mask(query, center_field.spec, center.case_shape)
        _require_nonempty_calibration_cases(
            mask,
            case_axes=center.case_axes,
            case_axis=selected_axis,
        )
        scale_field = None
        if scale is not None:
            scale_values = _operator_target_values(
                scale,
                query=query,
                output_spec=center_field.spec,
                case_axes=center.case_axes,
                case_shape=center.case_shape,
                field_name=selected_name,
            )
            scale_field = cx.Field(scale_values, dims=physical_dims)
        weights = None
        if score == "l2":
            weights = _output_weights(
                query,
                center_field.spec,
                center.case_shape,
                normalized=False,
            )
        calibrator = FunctionalConformal.calibrate(
            cx.Field(center_values, dims=physical_dims),
            cx.Field(target_values, dims=physical_dims),
            alpha=alpha,
            case_dim=selected_axis,
            scale=scale_field,
            min_scale=min_scale,
            mask=mask,
            weights=weights,
            score=score,
        )
        return cls(
            calibrator,
            case_axis=selected_axis,
            field_name=selected_name,
        )

    def interval(
        self,
        center: OperatorPrediction,
        scale: OperatorPrediction | None = None,
        /,
    ) -> OperatorPredictionInterval:
        if not isinstance(center, OperatorPrediction):
            raise TypeError("center must be an OperatorPrediction.")
        _, center_field, query = _select_prediction_field(
            center,
            self.field_name,
        )
        physical_dims = _physical_dims(query, center_field.spec, center.case_axes)
        center_values = _operator_target_values(
            center,
            query=query,
            output_spec=center_field.spec,
            case_axes=center.case_axes,
            case_shape=center.case_shape,
            field_name=self.field_name,
        )
        scale_field = None
        if scale is not None:
            scale_values = _operator_target_values(
                scale,
                query=query,
                output_spec=center_field.spec,
                case_axes=center.case_axes,
                case_shape=center.case_shape,
                field_name=self.field_name,
            )
            scale_field = cx.Field(scale_values, dims=physical_dims)
        generic = self.calibrator.interval(
            cx.Field(center_values, dims=physical_dims),
            scale_field,
        )
        mask = _output_mask(query, center_field.spec, center.case_shape)
        lower = OperatorPrediction.from_field(
            self.field_name,
            jnp.where(mask, jnp.asarray(generic.lower.data), 0.0),
            center_field.query_name,
            query,
            spec=center_field.spec,
            case_axes=center.case_axes,
            case_shape=center.case_shape,
        )
        upper = OperatorPrediction.from_field(
            self.field_name,
            jnp.where(mask, jnp.asarray(generic.upper.data), 0.0),
            center_field.query_name,
            query,
            spec=center_field.spec,
            case_axes=center.case_axes,
            case_shape=center.case_shape,
        )
        return OperatorPredictionInterval(
            lower,
            upper,
            nominal_coverage=generic.nominal_coverage,
            simultaneous=generic.simultaneous,
            calibrated=generic.calibrated,
        )


def _calibration_case_axis(
    prediction: OperatorPrediction,
    requested: str | None,
    /,
) -> str:
    if requested is None:
        if len(prediction.case_axes) != 1:
            raise ValueError(
                "case_axis is required unless exactly one operator case axis exists."
            )
        return prediction.case_axes[0]
    axis = str(requested)
    if axis not in prediction.case_axes:
        raise ValueError(f"Unknown operator calibration case axis {axis!r}.")
    return axis


def _require_nonempty_calibration_cases(
    mask,
    /,
    *,
    case_axes: tuple[str, ...],
    case_axis: str,
) -> None:
    position = case_axes.index(case_axis)
    selected = jnp.moveaxis(jnp.asarray(mask, dtype=bool), position, 0)
    if bool(jnp.any(~jnp.any(selected.reshape((selected.shape[0], -1)), axis=-1))):
        raise ValueError("Every calibration case must contain positive physical support.")


__all__ = ["OperatorFunctionalConformal"]
