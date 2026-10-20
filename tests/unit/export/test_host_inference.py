#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import gc
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig
from phydrax.interchange import ExternalTensorSpec


def _binding() -> Any:
    semantic = phx.SemanticProvenance({"kind": "affine-host-runtime"})
    return phx.ArtifactBindingIdentity(
        semantic,
        phx.NumericRevision(semantic, {"scale": 2.0}),
        phx.ExecutableSignature(shapes={"x": (3,)}, dtypes={"x": np.float64}),
    )


def _adapter(runner: Any, **options: Any) -> Any:
    return phx.export.HostInferenceAdapter(
        runner,
        (ExternalTensorSpec("x", (3,), np.float64),),
        (ExternalTensorSpec("y", (3,), np.float64),),
        _binding(),
        **options,
    )


def test_host_inference_refuses_schema_mismatches_before_the_runtime() -> None:
    calls = []

    def runner(values: Any) -> Any:
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


def _host_buffer(values: Any, misalignment: int) -> Any:
    # A read-only producer buffer `misalignment` bytes past a 64-byte boundary.
    raw = np.empty(values.nbytes + 128, dtype=np.uint8)
    start = -raw.ctypes.data % 64 + misalignment
    array = raw[start : start + values.nbytes].view(values.dtype)
    array[...] = values
    array.setflags(write=False)
    return array


def test_host_inference_dlpack_aliases_read_only_buffers_and_copy_detaches() -> None:
    x = jnp.asarray([1.0, -2.0, 3.0])
    borrowed = []
    produced = []

    def runner(values: Any) -> Any:
        (host,) = values
        borrowed.append((host.ctypes.data, host.flags.writeable))
        output = _host_buffer(2.0 * host, 0)
        produced.append(output.ctypes.data)
        return [output]

    shared = _adapter(runner, transport="dlpack")(x)
    assert borrowed == [(x.unsafe_buffer_pointer(), False)]
    assert shared.unsafe_buffer_pointer() == produced[0]
    # The runner dropped its producer reference; the result keeps the buffer alive.
    gc.collect()
    np.testing.assert_array_equal(shared, [2.0, -4.0, 6.0])

    copied = _adapter(runner)(x)
    assert isinstance(copied, jax.Array) and isinstance(shared, jax.Array)
    assert copied.unsafe_buffer_pointer() != produced[1]
    np.testing.assert_array_equal(copied, shared)

    with pytest.raises(TypeError, match="__dlpack__"):
        _adapter(runner, transport="dlpack")([1.0, 2.0, 3.0])
    misaligned = _adapter(lambda values: [_host_buffer(values[0], 8)], transport="dlpack")
    with pytest.raises(ValueError, match="copy"):
        misaligned(x)


def test_host_inference_refuses_x64_outputs_with_x64_disabled() -> None:
    calls = []
    x = _host_buffer(np.asarray([1.0, -2.0, 3.0]), 0)
    for transport in ("copy", "dlpack"):
        with (
            pxconfig.context(pxconfig.NumericsConfig(enable_x64=False)),
            pytest.raises(TypeError, match="x64"),
        ):
            _adapter(calls.append, transport=transport)(x)
    assert calls == []


def _affine_onnx_model(weight: Any, bias: Any, *, batch: Any) -> Any:
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


def _write_model(path: Any, model_bytes: Any) -> Any:
    path.write_bytes(model_bytes)
    return hashlib.sha256(model_bytes).hexdigest()


def test_onnx_runtime_host_inference_is_pinned_bound_and_host_only(tmp_path: Any) -> None:
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
