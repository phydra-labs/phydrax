#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.ml.interop._onnx as onnx_module
from phydrax._model import ModelBinding


def _fitted_ridge():
    features = jnp.asarray(
        [
            [-1.0, 0.5],
            [0.0, -0.5],
            [1.0, 0.25],
            [2.0, 1.0],
        ]
    )
    targets = 1.5 * features[:, 0] - 0.25 * features[:, 1]
    return phx.ml.fit(phx.ml.linear.RidgeRecipe(alpha=1e-3), features, targets).model


def test_save_ml_onnx_infers_pointwise_vectorization_and_validation_inputs(
    monkeypatch, tmp_path
):
    model = _fitted_ridge()
    calls = []
    sentinel = object()

    def fake_save_onnx(exported_model, path, **kwargs):
        calls.append((exported_model, path, kwargs))
        return sentinel

    monkeypatch.setattr(onnx_module, "save_onnx", fake_save_onnx)
    inputs = [("B", 2)]
    path = tmp_path / "ridge.onnx"

    result = phx.ml.interop.save_ml_onnx(model, path, inputs=inputs)

    assert result is sentinel
    assert calls == [
        (
            model,
            path,
            {
                "inputs": inputs,
                "input_names": None,
                "output_names": None,
                "model_name": None,
                "vectorize": True,
                "validate": True,
                "validation_inputs": inputs,
                "rtol": 1e-3,
                "atol": 1e-5,
            },
        )
    ]


def test_save_ml_onnx_rejects_axis_batched_model(monkeypatch, tmp_path):
    model = _fitted_ridge()
    monkeypatch.setattr(type(model), "_input_binding", ModelBinding.axis())

    with pytest.raises(ValueError, match="Axis-batched"):
        phx.ml.interop.save_ml_onnx(
            model,
            tmp_path / "ridge.onnx",
            inputs=[("B", 2)],
        )
