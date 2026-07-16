#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import sys
import types
from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx


class _FakeJax2Onnx:
    to_onnx_calls: list[dict[str, Any]]
    allclose_calls: list[dict[str, Any]]

    def __init__(self):
        self.to_onnx_calls = []
        self.allclose_calls = []

    def to_onnx(self, fn, **kwargs):
        width = int(kwargs["inputs"][0][-1])
        sample = jnp.arange(1, 1 + 2 * width, dtype=float).reshape((2, width))
        self.to_onnx_calls.append(
            {
                "kwargs": kwargs,
                "value": fn(sample),
            }
        )
        return kwargs["output_path"]

    def allclose(self, fn, path, *, inputs, rtol, atol, enable_double_precision):
        self.allclose_calls.append(
            {
                "path": path,
                "value": fn(*inputs),
                "rtol": rtol,
                "atol": atol,
                "enable_double_precision": enable_double_precision,
            }
        )
        return True, "ok"


def _install_fake_jax2onnx(monkeypatch):
    fake = _FakeJax2Onnx()
    module = types.ModuleType("jax2onnx")
    module.__dict__["to_onnx"] = fake.to_onnx
    module.__dict__["allclose"] = fake.allclose
    monkeypatch.setitem(sys.modules, "jax2onnx", module)
    return fake


def test_save_onnx_vectorizes_preprocesses_and_postprocesses(monkeypatch, tmp_path):
    fake = _install_fake_jax2onnx(monkeypatch)

    def model(row, *, key=None):
        assert key is None
        return jnp.sum(row)

    result = phx.export.save_onnx(
        model,
        tmp_path / "u.onnx",
        inputs=[("B", 2)],
        input_names=["x"],
        output_names=["y"],
        model_name="u",
        vectorize=True,
        preprocess=lambda x: x + 1.0,
        postprocess=lambda y: 2.0 * y,
    )

    assert result.path == tmp_path / "u.onnx"
    call = fake.to_onnx_calls[0]
    assert call["kwargs"]["inputs"] == [("B", 2)]
    assert call["kwargs"]["input_names"] == ["x"]
    assert call["kwargs"]["output_names"] == ["y"]
    assert call["kwargs"]["model_name"] == "u"
    assert jnp.allclose(call["value"], jnp.asarray([10.0, 18.0]))


def test_save_onnx_validates_with_supplied_inputs(monkeypatch, tmp_path):
    fake = _install_fake_jax2onnx(monkeypatch)

    def model(x, *, key=None):
        del key
        return x + 1.0

    result = phx.export.save_onnx(
        model,
        tmp_path / "u.onnx",
        inputs=[("B", 2)],
        validate=True,
        validation_inputs=[jnp.asarray([[0.0, 1.0]])],
        rtol=1e-4,
        atol=1e-6,
    )

    assert result.validation_ok is True
    assert result.validation_message == "ok"
    assert fake.allclose_calls[0]["rtol"] == 1e-4
    assert fake.allclose_calls[0]["atol"] == 1e-6
    assert jnp.allclose(fake.allclose_calls[0]["value"], jnp.asarray([[1.0, 2.0]]))


def test_save_onnx_validate_requires_validation_inputs(tmp_path):
    def model(x, *, key=None):
        del key
        return x

    with pytest.raises(ValueError, match="validation_inputs"):
        phx.export.save_onnx(
            model,
            tmp_path / "u.onnx",
            inputs=[("B", 2)],
            validate=True,
        )


def test_solver_save_onnx_exports_named_ansatz_function(monkeypatch, tmp_path):
    fake = _install_fake_jax2onnx(monkeypatch)
    geom = phx.domain.Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x + 1.0

    solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=())
    result = solver.save_onnx(
        "u",
        tmp_path / "solver_u.onnx",
        inputs=[("B", 1)],
        vectorize=True,
    )

    assert result.path == tmp_path / "solver_u.onnx"
    assert jnp.allclose(fake.to_onnx_calls[0]["value"], jnp.asarray([[2.0], [3.0]]))
