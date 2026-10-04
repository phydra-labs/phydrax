from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _manifest() -> Any:
    return phx.export.IREEArtifactManifest(
        format="phydrax-iree-inference",
        artifact_id="artifact",
        module_file="module.vmfb",
        module_sha256="0" * 64,
        compiler_version="3.11.0",
        runtime_version="3.11.0",
        target_backend="llvm-cpu",
        runtime_driver="local-task",
        executable_format="embedded-elf",
        executable_platform="embedded-elf-arm64",
        system_linker=None,
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


def test_iree_contracts() -> None:
    manifest = _manifest()
    restored = phx.export.IREEArtifactManifest.from_dict(
        json.loads(json.dumps(manifest.to_dict()))
    )
    assert restored == manifest
    invalid = manifest.to_dict()
    invalid["unknown"] = 1
    with pytest.raises(ValueError, match="not canonical"):
        phx.export.IREEArtifactManifest.from_dict(invalid)


def test_iree_export_rejects_dynamic_key_empty_inputs_and_invalid_policy(
    tmp_path: Any,
) -> None:
    def model(x: Any, *, key: Any = None) -> Any:
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

    def multi_output_model(x: Any, *, key: Any = None) -> Any:
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

    def list_output_model(x: Any, *, key: Any = None) -> Any:
        del key
        return [x, jnp.sum(x)]

    with pytest.raises(TypeError, match="tuple of arrays"):
        phx.export.save_iree(
            list_output_model,
            tmp_path / "list-output",
            inputs=(jnp.ones((2,)),),
        )


def test_iree_load_requires_an_out_of_band_module_pin(tmp_path: Any) -> None:
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
def test_iree_compiles_validates_loads_and_rejects_wrong_inputs(tmp_path: Any) -> None:
    def model(x: Any, *, key: Any = None) -> Any:
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
def test_iree_compiles_ordered_heterogeneous_outputs_without_packing(
    tmp_path: Any,
) -> None:
    def model(x: Any, scale: Any, *, key: Any = None) -> Any:
        del key
        return (
            scale * jnp.sin(x),
            jnp.all(jnp.isfinite(x)),
            x > 0.0,
            jnp.asarray(x.size, dtype=jnp.int32),
            (x + 1j * x).astype(jnp.complex64),
            (4.0 * jnp.abs(x)).astype(jnp.uint8),
        )

    sample = jnp.asarray((0.25, -0.5, 1.0), dtype=jnp.float32)
    # A scalar input keeps its rank-0 HAL shape.
    scale = jnp.asarray(2.0, dtype=jnp.float32)
    destination = tmp_path / "multi-output.phxiree"
    result = phx.export.save_iree(
        model,
        destination,
        inputs=(sample, scale),
        input_names=("x", "scale"),
        output_names=("values", "finite", "positive", "count", "complex", "bins"),
        validate=True,
    )
    deployed = phx.export.load_iree(
        destination,
        trusted_module_sha256=result.manifest.module_sha256,
    )
    actual = deployed(np.asarray(sample), np.asarray(scale))
    expected = model(sample, scale)

    assert isinstance(actual, tuple)
    assert result.manifest.output_shapes == ((3,), (), (3,), (), (3,), (3,))
    assert result.manifest.output_dtypes == tuple(
        np.dtype(value.dtype).str for value in expected
    )
    for deployed_value, native_value in zip(actual, expected, strict=True):
        native_array = np.asarray(native_value)
        assert deployed_value.dtype == native_array.dtype
        if np.issubdtype(native_array.dtype, np.inexact):
            np.testing.assert_allclose(
                deployed_value, native_array, rtol=1.0e-4, atol=1.0e-6
            )
        else:
            np.testing.assert_array_equal(deployed_value, native_array)


@pytest.mark.skipif(not _HAS_IREE, reason="IREE optional packages are not installed")
def test_iree_load_refuses_a_manifest_abi_the_pinned_module_does_not_declare(
    tmp_path: Any,
) -> None:
    def model(x: Any, *, key: Any = None) -> Any:
        del key
        return jnp.sqrt(x)

    destination = tmp_path / "root.phxiree"
    result = phx.export.save_iree(
        model, destination, inputs=(jnp.ones((2,), dtype=jnp.float32),)
    )
    pin = result.manifest.module_sha256
    with pytest.raises(RuntimeError, match=r"output 0 .*non-finite"):
        phx.export.load_iree(destination, trusted_module_sha256=pin)(
            np.asarray((-1.0, 4.0), dtype=np.float32)
        )
    original = result.manifest.to_dict()
    # Same-width forgeries would otherwise reinterpret float32 bit patterns.
    for forged, refusal in (
        ({"output_dtypes": ["<i4"]}, r"output 0 .*dtype <f4.* dtype <i4"),
        ({"output_shapes": [[1, 2]]}, r"output 0 as shape \(2,\)"),
        ({"input_dtypes": ["<i4"]}, r"input 0 .*dtype <f4.* dtype <i4"),
        (
            {"input_shapes": [[1]], "output_shapes": [[1]]},
            r"input 0 as shape \(2,\)",
        ),
    ):
        (destination / "manifest.json").write_text(
            json.dumps({**original, **forged}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match=refusal):
            phx.export.load_iree(destination, trusted_module_sha256=pin)


@pytest.mark.skipif(not _HAS_IREE, reason="IREE optional packages are not installed")
def test_iree_scatter_and_gather_keep_stablehlo_out_of_range_semantics(
    tmp_path: Any,
) -> None:
    # Segment ids 4, 5 and -1 lie outside the four segments.
    ids = jnp.asarray((0, 1, 1, 4, 5, 5, -1, 3), dtype=jnp.int32)
    # Three windows leave the (6, 5) operand on a scattered axis.
    starts = jnp.asarray(
        ((0, 1), (2, 4), (3, -1), (5, 2), (6, 0), (1, 2)), dtype=jnp.int32
    )
    window = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0, 1),
    )

    def model(x: Any, segment: Any, start: Any, *, key: Any = None) -> Any:
        del key
        # Segment axis is not leading; both sums start from the same zeros.
        summed = jax.vmap(
            lambda column: jax.ops.segment_sum(column, segment, 4),
            in_axes=1,
            out_axes=1,
        )(x)
        doubled = jax.vmap(
            lambda column: jax.ops.segment_sum(2.0 * column, segment, 4),
            in_axes=1,
            out_axes=1,
        )(x)
        batched = jax.vmap(lambda row, index: jnp.zeros(4, x.dtype).at[index].add(row))(
            x.T, jnp.stack((segment % 4, segment[::-1] % 4, (3 * segment) % 4))
        )
        windows = jax.lax.scatter_add(
            jnp.zeros((6, 5), x.dtype), start, x[:6], window, mode="drop"
        )
        clamped = x[:, 0].at[segment].get(mode="promise_in_bounds")
        return summed, doubled, batched, windows, clamped

    x = jnp.asarray(np.random.default_rng(0).normal(size=(8, 3)), dtype=jnp.float32)
    result = phx.export.save_iree(
        model,
        tmp_path / "scatter.phxiree",
        inputs=(x, ids, starts),
        input_names=("x", "segment", "start"),
        validate=True,
    )
    deployed = phx.export.load_iree(
        result.path, trusted_module_sha256=result.manifest.module_sha256
    )
    actual = deployed(np.asarray(x), np.asarray(ids), np.asarray(starts))

    for deployed_value, native_value in zip(actual, model(x, ids, starts), strict=True):
        np.testing.assert_allclose(
            deployed_value, np.asarray(native_value), rtol=1.0e-6, atol=1.0e-6
        )


@pytest.mark.skipif(not _HAS_IREE, reason="IREE optional packages are not installed")
def test_iree_export_refuses_in_place_updates_iree_would_alias(tmp_path: Any) -> None:
    def shared_constant(x: Any, *, key: Any = None) -> Any:
        del key
        base = jnp.arange(x.size, dtype=x.dtype).reshape(x.shape)
        return base.at[0].add(x[0]), base.at[1].add(x[1])

    def returned_argument(x: Any, *, key: Any = None) -> Any:
        del key
        # A legalized scatter writes a padded copy; dynamic_update_slice stays in place.
        return jax.lax.dynamic_update_slice(x, x[1:2] + 1.0, (0, 0)), x

    def fenced_destination(x: Any, *, key: Any = None) -> Any:
        del key
        values = jnp.sin(x)
        summed = jax.ops.segment_sum(values, jnp.asarray((0, 1, 1)), 2)
        return summed, jax.lax.dynamic_update_slice(values, x[:1], (1, 0))

    x = jnp.ones((3, 2), dtype=jnp.float32)
    for model, reason in (
        (shared_constant, "shared input-independent destination"),
        (returned_argument, "returned while updated in place"),
        (fenced_destination, "operand reaches an in-place destination"),
    ):
        with pytest.raises(ValueError, match=reason):
            phx.export.save_iree(model, tmp_path / model.__name__, inputs=(x,))
        assert not (tmp_path / model.__name__).exists()
