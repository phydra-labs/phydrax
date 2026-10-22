from __future__ import annotations

import hashlib
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks._memory import host_allocation_peak
from phydrax._array_archive import array_collection_digest
from phydrax._fingerprint import (
    array_bytes_digest,
    array_tree_fingerprint,
    canonical_array_hex_fingerprint,
    iter_array_bytes,
    logical_array_payloads,
    logical_array_value_collection_digest,
    tree_content_equal,
)
from phydrax._strict import StrictModule
from tests._support.fingerprint import legacy_collection_digest
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


def _legacy(tree: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    signature = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        value = np.asarray(leaf)
        if value.dtype.hasobject:
            continue
        name = jax.tree_util.keystr(path) or "<root>"
        contiguous = np.ascontiguousarray(value)
        metadata = json.dumps(
            {
                "path": name,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        payload = contiguous.tobytes(order="C")
        for chunk in (metadata, payload):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)
        signature.append(
            {"path": name, "shape": list(value.shape), "dtype": value.dtype.str}
        )
    return {"signature": signature, "sha256": digest.hexdigest()}


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int64,
        np.complex64,
        np.complex128,
        np.bool_,
        np.dtype(">f8"),
        np.dtype([("x", "<i4"), ("y", ">f8")]),
    ],
)
@pytest.mark.parametrize(
    "layout", ["scalar", "contiguous", "strided", "reversed", "empty"]
)
def test_array_fingerprint_matches_legacy(dtype: np.dtype, layout: str) -> None:
    value = np.zeros((12, 10), dtype=dtype)
    if value.dtype.fields is None:
        value[...] = np.arange(120).reshape(12, 10) % 7
    else:
        value["x"] = np.arange(120).reshape(12, 10)
        value["y"] = 0.25
    match layout:
        case "scalar":
            value = np.asarray(value[0, 0])
        case "strided":
            value = value[::2, ::3].T
        case "reversed":
            value = value[::-1, ::-1]
        case "empty":
            value = value[:0]
    assert array_tree_fingerprint({"array": value}) == _legacy({"array": value})


def test_array_fingerprint_preserves_nan_bits() -> None:
    values = np.asarray(
        [0x7FF8000000000001, 0x7FF8000000000002, 0x8000000000000000], dtype=np.uint64
    ).view(np.float64)
    assert array_tree_fingerprint(values) == _legacy(values)
    assert array_tree_fingerprint(values[:1]) != array_tree_fingerprint(values[1:2])


@pytest.mark.parametrize("strided", [False, True])
def test_array_fingerprint_has_bounded_scratch(strided: bool) -> None:
    small = np.arange(131072, dtype=np.float64)
    large = np.arange(524288, dtype=np.float64)
    if strided:
        small = small.reshape(-1, 2).T
        large = large.reshape(-1, 2).T
    array_tree_fingerprint(small)
    _, small_peak = host_allocation_peak(lambda: array_tree_fingerprint(small))
    _, large_peak = host_allocation_peak(lambda: array_tree_fingerprint(large))
    assert large_peak <= max(16384, small_peak * 1.5)


def test_array_fingerprint_rejects_object_arrays() -> None:
    with pytest.raises(TypeError, match=r"array_tree_fingerprint\.leaf\.dtype"):
        array_tree_fingerprint(np.asarray(["value"], dtype=np.object_))


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]", "U3", "S3", "V0"])
def test_non_numeric_array_fingerprint_matches_legacy(dtype: str) -> None:
    value = np.zeros((3, 4), dtype=np.dtype(dtype))
    assert array_tree_fingerprint(value.T) == _legacy(value.T)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.bool_, ">f8"])
@pytest.mark.parametrize("empty", [False, True])
def test_bounded_array_byte_and_hex_hashes(dtype: np.dtype, empty: bool) -> None:
    value = np.arange(24).reshape(4, 6).astype(dtype)[::-1, ::2]
    if empty:
        value = value[:0]
    payload = value.tobytes(order="C")
    assert array_bytes_digest(value) == hashlib.sha256(payload).hexdigest()
    canonical_hex = json.dumps(payload.hex(), separators=(",", ":")).encode("ascii")
    assert (
        canonical_array_hex_fingerprint(value)
        == hashlib.sha256(canonical_hex).hexdigest()
    )
    assert b"".join(bytes(chunk) for chunk in iter_array_bytes(value)) == payload
    assert all(len(chunk) <= 65536 for chunk in iter_array_bytes(value))


class _Record(StrictModule):
    values: np.ndarray
    label: str = eqx.field(static=True)
    scale: float = eqx.field(static=True, default=float("nan"))


def _record(
    values: np.ndarray, label: str = "source", scale: float = float("nan")
) -> _Record:
    return _Record(values, label, scale)


def test_identical_nan_sentinels_compare_equal() -> None:
    values = np.asarray([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float64)
    assert tree_content_equal(_record(values), _record(values.copy()))
    assert tree_content_equal(jnp.asarray(values), jnp.asarray(values.copy()))
    assert tree_content_equal({"x": float("nan")}, {"x": float("nan")})
    # Independently constructed NaN static metadata (distinct float objects).
    assert tree_content_equal(
        _record(values, scale=float("nan")), _record(values, scale=float("nan"))
    )


@pytest.mark.parametrize(
    "change",
    [
        "one-ulp",
        "sentinel-to-zero",
        "signed-zero",
        "shape",
        "dtype",
        "leaf-type",
        "static",
        "static-signed-zero",
        "static-nan-payload",
        "structure",
        "scalar-rank",
        "static-mapping-key-type",
    ],
)
def test_any_content_change_is_unequal(change: str) -> None:
    values = np.asarray([[np.nan, 1.0], [0.0, 2.0]], dtype=np.float64)
    first = _record(values)
    changed = values.copy()
    match change:
        case "one-ulp":
            changed[0, 1] = np.nextafter(1.0, 2.0)
            second = _record(changed)
        case "sentinel-to-zero":
            changed[0, 0] = 0.0
            second = _record(changed)
        case "signed-zero":
            changed[1, 0] = -0.0
            second = _record(changed)
        case "shape":
            second = _record(changed.reshape(4, 1))
        case "dtype":
            second = _record(changed.astype(np.float32))
        case "leaf-type":
            assert not tree_content_equal(values, jnp.asarray(values))
            return
        case "static":
            second = _record(changed, "rebuilt")
        case "static-signed-zero":
            first = _record(values, scale=0.0)
            second = _record(values, scale=-0.0)
        case "static-nan-payload":
            payload = np.frombuffer(
                np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64
            )[0]
            second = _record(values, scale=float(payload))
        case "structure":
            assert not tree_content_equal((values,), (values, values))
            return
        case "scalar-rank":
            # A 0-d leaf differs from a length-one axis with the same bytes.
            assert not tree_content_equal(np.asarray(1.0), np.asarray([1.0]))
            return
        case "static-mapping-key-type":
            # Integer, float and boolean keys are distinct metadata.
            assert not tree_content_equal(
                MappingProxyType({1: "a"}), MappingProxyType({1.0: "a"})
            )
            assert not tree_content_equal(
                MappingProxyType({1: "a"}), MappingProxyType({True: "a"})
            )
            assert tree_content_equal(
                MappingProxyType({1: "a"}), MappingProxyType({1: "a"})
            )
            return
        case _:
            raise AssertionError(change)
    assert not tree_content_equal(first, second)


def _aliased_pairs(leaf: np.ndarray, depth: int) -> tuple[object, object]:
    node: object = leaf
    for _ in range(depth):
        node = (node, node)
    return node, node


def test_aliased_subtrees_compare_by_distinct_content() -> None:
    # Sixty doubling levels expand to 2**61 leaves; the comparison must visit
    # each distinct aliased node pair once to finish at all.
    values = np.asarray([np.nan, 1.0], dtype=np.float64)
    first = _aliased_pairs(values, 60)
    assert tree_content_equal(first, _aliased_pairs(values.copy(), 60))
    changed = values.copy()
    changed[1] = -1.0
    assert not tree_content_equal(first, _aliased_pairs(changed, 60))


@dataclass(frozen=True)
class _ArrayMetadata:
    values: np.ndarray
    label: str
    retained: float = field(compare=False, default=0.0)


def test_dataclass_array_metadata_compares_every_field_bitwise() -> None:
    values = np.asarray([np.nan, 0.0], dtype=np.float64)
    first = _ArrayMetadata(values, "source")
    assert tree_content_equal(first, _ArrayMetadata(values.copy(), "source"))
    changed = values.copy()
    changed[1] = -0.0
    assert not tree_content_equal(first, _ArrayMetadata(changed, "source"))
    assert not tree_content_equal(first, _ArrayMetadata(values, "other"))
    assert not tree_content_equal(first, _ArrayMetadata(values, "source", -0.0))


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint16,
        np.int32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.dtype(">f8"),
        np.dtype(">i4"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    ["scalar", "contiguous", "strided", "reversed", "fortran", "empty", "zero-trailing"],
)
def test_logical_array_digest_retains_canonical_archive_bytes(
    dtype: np.dtype, layout: str
) -> None:
    value = np.arange(48, dtype=np.int32).astype(dtype).reshape(8, 6)
    if value.dtype.kind in "fc":
        value[0, :4] = np.asarray([np.nan, -0.0, np.inf, -np.inf], dtype=dtype)
    match layout:
        case "scalar":
            value = np.asarray(value[0, 0])
        case "strided":
            value = value[::2, ::2].T
        case "reversed":
            value = value[::-1, ::-1]
        case "fortran":
            value = np.asfortranarray(value)
        case "empty":
            value = value[:0]
        case "zero-trailing":
            value = value[:, :0]
    logical_shape = (value.shape[0] // 2, *value.shape[1:]) if value.ndim else ()
    logical = value[: logical_shape[0]] if logical_shape else value
    # Name lengths count UTF-8 bytes, names sort canonically, and scalars keep
    # rank zero in NPY metadata (unlike the array-tree fingerprint recipe).
    expected_arrays = {"zeta": logical, "α/source": np.asarray(7, dtype=np.int32)}
    expected = legacy_collection_digest(expected_arrays)
    assert array_collection_digest(expected_arrays) == expected
    for limit in (max(4, value.dtype.itemsize), 33, 257):
        assert logical_array_value_collection_digest(
            {"α/source": expected_arrays["α/source"], "zeta": value},
            logical_shapes={"zeta": logical_shape},
            maximum_chunk_bytes=limit,
        ) == expected


@pytest.mark.parametrize(
    "logical_shape",
    [(-1, 3), (5, 3), (2, 2), (2,), (), (True, 3), (np.bool_(True), 3), (1.5, 3)],
)
def test_logical_stream_apis_reject_non_prefix_shapes_consistently(
    logical_shape: tuple[int, ...],
) -> None:
    host = np.arange(12, dtype=np.int32).reshape(4, 3)
    device = jnp.asarray(host)
    for value in (host, device):
        with pytest.raises(ValueError, match="trim only leading-axis capacity padding"):
            logical_array_value_collection_digest(
                {"value": value}, logical_shapes={"value": logical_shape}
            )
    with pytest.raises(ValueError, match="trim only leading-axis capacity padding"):
        list(logical_array_payloads(device, logical_shape))


@pytest.mark.parametrize("logical_shape", [(0, 3), (2, 3), (np.int64(4), np.int64(3))])
def test_logical_payload_prefix_matches_original_bytes(
    logical_shape: tuple[int, ...],
) -> None:
    host = np.arange(12, dtype=np.int32).reshape(4, 3)
    device = jnp.asarray(host)
    for limit in (4, 7, 513):
        payloads = list(
            logical_array_payloads(device, logical_shape, maximum_chunk_bytes=limit)
        )
        assert all(0 < len(payload) <= limit for payload in payloads)
        assert b"".join(payloads) == host[: logical_shape[0]].tobytes(order="C")


def test_logical_payload_scalar_and_empty_collection_recipes() -> None:
    scalar = np.asarray(-0.0, dtype=np.float32)
    payloads = logical_array_payloads(jnp.asarray(scalar), (), maximum_chunk_bytes=4)
    assert b"".join(payloads) == scalar.tobytes()
    assert logical_array_value_collection_digest({}) == hashlib.sha256().hexdigest()
    with pytest.raises(ValueError, match="trim only leading-axis capacity padding"):
        list(logical_array_payloads(jnp.asarray(scalar), (1,)))
    with pytest.raises(ValueError, match="refer to declared arrays"):
        logical_array_value_collection_digest({}, logical_shapes={"absent": (0,)})


@pytest.mark.parametrize("limit", [True, 0, -1, 3])
def test_logical_stream_apis_preserve_one_element_admission(limit: int) -> None:
    host = np.asarray([1], dtype=np.int32)
    device = jnp.asarray(host)
    for value in (host, device):
        with pytest.raises(ValueError, match="maximum_chunk_bytes"):
            logical_array_value_collection_digest(
                {"value": value}, maximum_chunk_bytes=limit
            )
    with pytest.raises(ValueError, match="maximum_chunk_bytes"):
        list(logical_array_payloads(device, device.shape, maximum_chunk_bytes=limit))


@pytest.mark.parametrize("dtype", ["O", "V0", "U3", "M8[ns]"])
def test_logical_array_digest_rejects_nonprimitive_dtypes_before_streaming(
    dtype: str,
) -> None:
    value = np.zeros((2,), dtype=np.dtype(dtype))
    with pytest.raises(TypeError, match="primitive numeric"):
        logical_array_value_collection_digest({"value": value})


def test_logical_fingerprint_fully_addressable_layouts() -> None:
    record = run_on_forced_devices(
        "tests._support.fingerprint", "logical_layouts", (), devices=4, env={}
    )
    assert record["layouts"] == 2
    assert record["devices"] == 4
    assert record["no_collectives"]


def test_logical_fingerprint_real_distributed_bytes_and_strict_buffers() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_on_forced_devices,
                "tests._support.fingerprint",
                "logical_distributed",
                (),
                devices=1,
                env=gloo_rank_environment(coordinator, rank),
            )
            for rank in range(2)
        ]
        records = [future.result() for future in futures]
    assert [record["rank"] for record in records] == [0, 1]
    assert records[0]["digest"] == records[1]["digest"]
    assert records[0]["rounds"] == records[1]["rounds"]
    for round_record in records[0]["rounds"]:
        assert 0 < round_record["largest_buffer"] <= round_record["limit"]
