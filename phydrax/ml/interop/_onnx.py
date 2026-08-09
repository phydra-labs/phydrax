#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..._model import AbstractArrayModel
from ...export import OnnxExportResult, save_onnx


def save_ml_onnx(
    model: AbstractArrayModel,
    path: str | Path,
    /,
    *,
    inputs: Sequence[Any],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    model_name: str | None = None,
    vectorize: bool | None = None,
    validate: bool = True,
    validation_inputs: Sequence[Any] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> OnnxExportResult:
    """Export and numerically validate one supported native fitted ML predictor."""
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("save_ml_onnx requires an AbstractArrayModel.")
    binding = model.input_binding()
    if binding.batch_mode == "axis":
        raise ValueError("Axis-batched models do not define one portable ONNX boundary.")
    vectorize_ = (
        binding.batch_mode == "pointwise" if vectorize is None else bool(vectorize)
    )
    validation_inputs_ = inputs if validation_inputs is None else validation_inputs
    return save_onnx(
        model,
        path,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        model_name=model_name,
        vectorize=vectorize_,
        validate=bool(validate),
        validation_inputs=validation_inputs_ if validate else None,
        rtol=rtol,
        atol=atol,
    )


__all__ = ["save_ml_onnx"]
