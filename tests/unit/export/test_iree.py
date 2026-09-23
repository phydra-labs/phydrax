from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _manifest():
    return phx.export.IREEArtifactManifest(
        format="phydrax-iree-inference",
        artifact_id="artifact",
        module_file="module.vmfb",
        module_sha256="0" * 64,
        compiler_version="3.11.0",
        runtime_version="3.11.0",
        target_backend="llvm-cpu",
        runtime_driver="local-task",
        function_name="model",
        entry_point="main",
        calling_convention_version=10,
        input_names=("x",),
        input_shapes=((2,),),
        input_dtypes=(np.dtype(np.float32).str,),
        output_names=("prediction",),
        output_shapes=((2,),),
        output_dtypes=(np.dtype(np.float32).str,),
        vectorized=False,
        has_preprocess=False,
        has_postprocess=False,
        validation_ok=True,
        maximum_absolute_errors=(0.0,),
        maximum_relative_errors=(0.0,),
    )


def test_iree_manifest_round_trip_is_strict_and_json_safe():
    manifest = _manifest()
    restored = phx.export.IREEArtifactManifest.from_dict(
        json.loads(json.dumps(manifest.to_dict()))
    )
    assert restored == manifest
    invalid = manifest.to_dict()
    invalid["unknown"] = 1
    with pytest.raises(ValueError, match="not canonical"):
        phx.export.IREEArtifactManifest.from_dict(invalid)


class _HostArray:
    def __init__(self, value):
        self._value = np.asarray(value)

    def to_host(self):
        return self._value


def _fake_executable(manifest, result):
    executable = object.__new__(phx.export.IREEExecutable)
    executable.manifest = manifest
    executable._function = lambda *_: result
    return executable


def test_iree_executable_validates_each_ordered_heterogeneous_output():
    manifest = replace(
        _manifest(),
        output_names=("prediction", "accepted", "iteration"),
        output_shapes=((2,), (), (1,)),
        output_dtypes=(
            np.dtype(np.float32).str,
            np.dtype(np.bool_).str,
            np.dtype(np.int32).str,
        ),
        maximum_absolute_errors=(0.0, 0.0, 0.0),
        maximum_relative_errors=(0.0, 0.0, 0.0),
    )
    argument = np.ones((2,), dtype=np.float32)
    executable = _fake_executable(
        manifest,
        (
            _HostArray(np.asarray((1.0, 2.0), dtype=np.float32)),
            _HostArray(np.asarray(True, dtype=np.bool_)),
            _HostArray(np.asarray((3,), dtype=np.int32)),
        ),
    )

    result = executable(argument)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0].dtype == np.dtype(np.float32)
    assert result[1].dtype == np.dtype(np.bool_)
    assert result[2].dtype == np.dtype(np.int32)
    with pytest.raises(RuntimeError, match="arity"):
        _fake_executable(manifest, _HostArray(argument))(argument)
    with pytest.raises(RuntimeError, match=r"output 1 .*shape"):
        _fake_executable(
            manifest,
            (
                _HostArray(argument),
                _HostArray(np.asarray((True,), dtype=np.bool_)),
                _HostArray(np.asarray((3,), dtype=np.int32)),
            ),
        )(argument)
    with pytest.raises(RuntimeError, match=r"output 1 .*dtype"):
        _fake_executable(
            manifest,
            (
                _HostArray(argument),
                _HostArray(np.asarray(1, dtype=np.int32)),
                _HostArray(np.asarray((3,), dtype=np.int32)),
            ),
        )(argument)
    with pytest.raises(RuntimeError, match=r"output 0 .*non-finite"):
        _fake_executable(
            manifest,
            (
                _HostArray(np.asarray((np.nan, 2.0), dtype=np.float32)),
                _HostArray(np.asarray(True, dtype=np.bool_)),
                _HostArray(np.asarray((3,), dtype=np.int32)),
            ),
        )(argument)


def test_iree_export_rejects_dynamic_key_empty_inputs_and_invalid_policy(tmp_path):
    def model(x, *, key=None):
        del key
        return x

    with pytest.raises(ValueError, match="key=None"):
        phx.export.save_iree(
            model,
            tmp_path / "keyed",
            inputs=(jnp.ones((2,)),),
            key=jnp.asarray((0, 1), dtype=jnp.uint32),
        )
    with pytest.raises(ValueError, match="at least one"):
        phx.export.save_iree(model, tmp_path / "empty", inputs=())
    with pytest.raises(ValueError, match="non-empty"):
        phx.export.IREEExportPolicy(target_backend="")

    def multi_output_model(x, *, key=None):
        del key
        return x, jnp.sum(x)

    with pytest.raises(ValueError, match="name every output"):
        phx.export.save_iree(
            multi_output_model,
            tmp_path / "unnamed-output",
            inputs=(jnp.ones((2,)),),
            output_names=("values",),
        )
    with pytest.raises(ValueError, match="unique"):
        phx.export.save_iree(
            multi_output_model,
            tmp_path / "duplicate-output",
            inputs=(jnp.ones((2,)),),
            output_names=("value", "value"),
        )

    def list_output_model(x, *, key=None):
        del key
        return [x, jnp.sum(x)]

    with pytest.raises(TypeError, match="tuple of arrays"):
        phx.export.save_iree(
            list_output_model,
            tmp_path / "list-output",
            inputs=(jnp.ones((2,)),),
        )


def test_iree_load_requires_an_out_of_band_module_pin(tmp_path):
    module_bytes = b"self-checksummed but not independently trusted"
    manifest = replace(
        _manifest(),
        module_sha256=hashlib.sha256(module_bytes).hexdigest(),
    )
    bundle = tmp_path / "untrusted.phxiree"
    bundle.mkdir()
    (bundle / manifest.module_file).write_bytes(module_bytes)
    (bundle / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )

    with pytest.raises(PermissionError, match="caller-supplied trusted pin"):
        phx.export.load_iree(
            bundle,
            trusted_module_sha256=hashlib.sha256(b"different authority").hexdigest(),
        )


_HAS_IREE = importlib.util.find_spec("iree") is not None
if _HAS_IREE:
    _HAS_IREE = (
        importlib.util.find_spec("iree.compiler") is not None
        and importlib.util.find_spec("iree.runtime") is not None
    )


@pytest.mark.skipif(not _HAS_IREE, reason="IREE optional packages are not installed")
def test_iree_compiles_validates_loads_and_rejects_wrong_inputs(tmp_path):
    def model(x, *, key=None):
        del key
        coefficients = jnp.asarray(((1.0, -0.5), (0.25, 2.0)), dtype=x.dtype)
        return jnp.tanh(x @ coefficients)

    inputs = (jnp.asarray(((0.2, -0.3), (1.0, 0.5)), dtype=jnp.float32),)
    destination = tmp_path / "model.phxiree"
    result = phx.export.save_iree(
        model,
        destination,
        inputs=inputs,
        input_names=("x",),
        validate=True,
    )
    executable = phx.export.load_iree(
        destination,
        trusted_module_sha256=result.manifest.module_sha256,
    )
    expected = np.asarray(model(inputs[0]))
    actual = executable(np.asarray(inputs[0]))

    assert isinstance(actual, np.ndarray)
    assert result.manifest.output_names == ("output_0",)
    assert result.manifest.validation_ok
    np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-6)
    with pytest.raises(ValueError, match="shape"):
        executable(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="dtype"):
        executable(np.asarray(inputs[0], dtype=np.float64))
    module = destination / result.manifest.module_file
    payload = bytearray(module.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    module.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        phx.export.load_iree(
            destination,
            trusted_module_sha256=result.manifest.module_sha256,
        )


@pytest.mark.skipif(not _HAS_IREE, reason="IREE optional packages are not installed")
def test_iree_compiles_ordered_heterogeneous_outputs_without_packing(tmp_path):
    def model(x, *, key=None):
        del key
        return (
            jnp.sin(x),
            jnp.all(jnp.isfinite(x)),
            jnp.asarray(x.size, dtype=jnp.int32),
        )

    sample = jnp.asarray((0.25, -0.5, 1.0), dtype=jnp.float32)
    destination = tmp_path / "multi-output.phxiree"
    result = phx.export.save_iree(
        model,
        destination,
        inputs=(sample,),
        input_names=("x",),
        output_names=("values", "finite", "count"),
        validate=True,
    )
    deployed = phx.export.load_iree(
        destination,
        trusted_module_sha256=result.manifest.module_sha256,
    )
    actual = deployed(np.asarray(sample))
    expected = model(sample)

    assert isinstance(actual, tuple)
    assert result.manifest.output_names == ("values", "finite", "count")
    assert result.manifest.output_shapes == ((3,), (), ())
    assert result.manifest.output_dtypes == tuple(
        np.dtype(value.dtype).str for value in expected
    )
    assert len(result.manifest.maximum_absolute_errors) == 3
    assert len(result.manifest.maximum_relative_errors) == 3
    for deployed_value, native_value in zip(actual, expected, strict=True):
        native_array = np.asarray(native_value)
        if np.issubdtype(native_array.dtype, np.inexact):
            np.testing.assert_allclose(
                deployed_value, native_array, rtol=1.0e-4, atol=1.0e-6
            )
        else:
            np.testing.assert_array_equal(deployed_value, native_array)
    assert actual[1].dtype == np.dtype(np.bool_)
    assert actual[2].dtype == np.dtype(np.int32)
