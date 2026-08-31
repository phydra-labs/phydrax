from __future__ import annotations

import importlib.util
import json

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
        output_shape=(2,),
        output_dtype=np.dtype(np.float32).str,
        vectorized=False,
        has_preprocess=False,
        has_postprocess=False,
        validation_ok=True,
        maximum_absolute_error=0.0,
        maximum_relative_error=0.0,
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
        return jnp.tanh(x @ jnp.asarray(((1.0, -0.5), (0.25, 2.0))))

    inputs = (jnp.asarray(((0.2, -0.3), (1.0, 0.5)), dtype=jnp.float32),)
    destination = tmp_path / "model.phxiree"
    result = phx.export.save_iree(
        model,
        destination,
        inputs=inputs,
        input_names=("x",),
        validate=True,
    )
    executable = phx.export.load_iree(destination)
    expected = np.asarray(model(inputs[0]))
    actual = executable(np.asarray(inputs[0]))

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
        phx.export.load_iree(destination)
