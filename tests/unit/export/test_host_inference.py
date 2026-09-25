#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.interchange import ExternalTensorSpec


def _binding():
    semantic = phx.SemanticProvenance({"kind": "affine-host-runtime"})
    return phx.ArtifactBindingIdentity(
        semantic,
        phx.NumericRevision(semantic, {"scale": 2.0}),
        phx.ExecutableSignature(shapes={"x": (3,)}, dtypes={"x": np.float64}),
    )


def _adapter(runner, **options):
    return phx.export.HostInferenceAdapter(
        runner,
        (ExternalTensorSpec("x", (3,), np.float64),),
        (ExternalTensorSpec("y", (3,), np.float64),),
        _binding(),
        **options,
    )


def test_host_inference_refuses_schema_mismatches_before_the_runtime():
    calls = []

    def runner(values):
        calls.append(values)
        return [2.0 * values[0]]

    adapter = _adapter(runner)
    with pytest.raises(ValueError, match="shape"):
        adapter(jnp.ones(4))
    with pytest.raises(TypeError, match="dtype"):
        adapter(jnp.ones(3, dtype=jnp.float32))
    with pytest.raises(ValueError, match="expects 1 inputs"):
        adapter(jnp.ones(3), jnp.ones(3))
    assert calls == []

    with pytest.raises(RuntimeError, match="non-finite"):
        _adapter(lambda values: [np.full(3, np.inf)])(jnp.ones(3))
    with pytest.raises(RuntimeError, match="returned 2 outputs"):
        _adapter(lambda values: [values[0], values[0]])(jnp.ones(3))


def test_host_inference_transports_return_equal_detached_values():
    x = jnp.asarray([1.0, -2.0, 3.0])

    def runner(values):
        return [2.0 * values[0]]

    copied = _adapter(runner)(x)
    shared = _adapter(runner, transport="dlpack")(x)

    assert isinstance(copied, jax.Array) and isinstance(shared, jax.Array)
    np.testing.assert_array_equal(copied, 2.0 * x)
    np.testing.assert_array_equal(shared, copied)
    with pytest.raises(TypeError, match="__dlpack__"):
        _adapter(runner, transport="dlpack")([1.0, 2.0, 3.0])


def _affine_onnx_model(weight, bias, *, batch):
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "weight"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["y"]),
        ],
        "affine",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [batch, 3])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [batch, 2])],
        [
            onnx.numpy_helper.from_array(weight, "weight"),
            onnx.numpy_helper.from_array(bias, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()


def _write_model(path, model_bytes):
    path.write_bytes(model_bytes)
    return hashlib.sha256(model_bytes).hexdigest()


def test_onnx_runtime_host_inference_is_pinned_bound_and_host_only(tmp_path):
    # Skips (with the reason in the report) when the optional
    # `phydrax[onnx-inference]` runtime or the `onnx` builder is not installed.
    pytest.importorskip("onnxruntime")
    weight = np.asarray([[1.0, 0.5], [-1.0, 2.0], [0.25, 0.0]], dtype=np.float32)
    bias = np.asarray([0.5, -1.0], dtype=np.float32)
    path = tmp_path / "affine.onnx"
    digest = _write_model(path, _affine_onnx_model(weight, bias, batch=2))

    with pytest.raises(PermissionError, match="trusted pin"):
        phx.export.load_onnx(path, trusted_model_sha256="0" * 64)
    adapter = phx.export.load_onnx(path, trusted_model_sha256=digest)

    x = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(adapter(jnp.asarray(x)), x @ weight + bias, rtol=1e-6)
    assert [spec.name for spec in adapter.input_schema] == ["x"]
    assert adapter.output_schema[0].shape == (2, 2)
    assert adapter.capabilities.tier == "host-inference"
    assert adapter.capabilities.host_only
    assert adapter.derivative_support.route == "none"
    with pytest.raises(TypeError, match="JAX transformations"):
        jax.jit(adapter)(jnp.asarray(x))

    changed = tmp_path / "changed.onnx"
    other = phx.export.load_onnx(
        changed,
        trusted_model_sha256=_write_model(
            changed, _affine_onnx_model(2.0 * weight, bias, batch=2)
        ),
    )
    assert other.binding.numeric_revision_id != adapter.binding.numeric_revision_id
    assert (
        other.binding.executable_signature_id == adapter.binding.executable_signature_id
    )

    symbolic = tmp_path / "symbolic.onnx"
    symbolic_digest = _write_model(
        symbolic, _affine_onnx_model(weight, bias, batch="batch")
    )
    with pytest.raises(ValueError, match="symbolic dimensions"):
        phx.export.load_onnx(symbolic, trusted_model_sha256=symbolic_digest)
