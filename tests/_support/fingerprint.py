# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Independent byte recipes and real CPU-placement fingerprint regressions."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax._fingerprint import (
    array_tree_fingerprint,
    logical_array_payloads,
    logical_array_value_collection_digest,
)


def legacy_collection_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Original sorted-name, length-prefixed canonical NPY byte recipe."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        output = io.BytesIO()
        np.save(output, np.array(arrays[name], order="C", copy=True), allow_pickle=False)
        encoded_name = name.encode("utf-8")
        payload = output.getvalue()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_arrays() -> dict[str, np.ndarray]:
    real = np.arange(120, dtype=np.float64).reshape(40, 3)
    real.reshape(-1)[:6] = np.asarray(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000000041,
            0x7FF8000000000042,
        ],
        dtype=np.uint64,
    ).view(np.float64)
    return {
        "a_real": real,
        "c_complex": real.astype(np.complex128),
        "d_integer": np.arange(120, dtype=np.int16).reshape(40, 3),
        "e_boolean": np.arange(120, dtype=np.int32).reshape(40, 3) % 2 == 0,
        "f_empty": np.empty((0, 3), dtype=np.float32),
        "g_zero_trailing": np.empty((40, 0), dtype=np.float64),
    }


def _placed(value: np.ndarray, sharding: NamedSharding) -> jax.Array:
    def local_payload(index: tuple[slice, ...] | None) -> np.ndarray:
        return value if index is None else value[index]

    return jax.make_array_from_callback(value.shape, sharding, local_payload)


def logical_layouts() -> dict[str, Any]:
    """Fully addressable partitioning and replication never route host bytes."""
    from phydrax.meshing._device_adaptation import _logical_array_tree_fingerprint

    assert jax.process_count() == 1
    assert jax.local_device_count() == 4
    mesh = Mesh(np.asarray(jax.devices(), dtype=np.object_), ("owner",))
    sources = _source_arrays()
    shapes = {
        name: ((27, *value.shape[1:]) if value.shape[0] else value.shape)
        for name, value in sources.items()
    }
    expected = legacy_collection_digest(
        {name: value[:shapes[name][0]] for name, value in sources.items()}
    )

    def no_collective(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Fully addressable arrays must not route bytes")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(multihost_utils, "process_allgather", no_collective)
        patch.setattr(multihost_utils, "broadcast_one_to_all", no_collective)
        for spec in (PartitionSpec("owner"), PartitionSpec()):
            arrays = {
                name: jax.device_put(value, NamedSharding(mesh, spec))
                for name, value in sources.items()
            }
            for name, value in arrays.items():
                assert _logical_array_tree_fingerprint(value, shapes[name]) == (
                    array_tree_fingerprint(sources[name][:shapes[name][0]])
                )
            for limit in (16, 33, 257, 779, 1023):
                assert logical_array_value_collection_digest(
                    arrays, logical_shapes=shapes, maximum_chunk_bytes=limit
                ) == expected
                for name, value in arrays.items():
                    payloads = list(
                        logical_array_payloads(
                            value, shapes[name], maximum_chunk_bytes=limit
                        )
                    )
                    assert all(0 < len(payload) <= limit for payload in payloads)
                    expected_payload = sources[name][:shapes[name][0]].tobytes(order="C")
                    assert b"".join(payloads) == expected_payload
        tiled = jax.device_put(
            np.zeros((40, 4), dtype=np.int32),
            NamedSharding(mesh, PartitionSpec(None, "owner")),
        )
        with pytest.raises(ValueError, match="contiguous leading-axis shard"):
            list(logical_array_payloads(tiled, tiled.shape))
        with pytest.raises(ValueError, match="contiguous leading-axis shard"):
            logical_array_value_collection_digest({"tiled": tiled})
    return {"digest": expected, "layouts": 2, "devices": 4, "no_collectives": True}


def logical_distributed() -> dict[str, Any]:
    """Real Gloo all-gathers and small-limit broadcasts keep canonical bytes."""
    from phydrax.meshing._device_adaptation import _logical_array_tree_fingerprint

    try:
        assert jax.process_count() == 2
        assert jax.local_device_count() == 1
        mesh = Mesh(np.asarray(jax.devices(), dtype=np.object_), ("owner",))
        sharding = NamedSharding(mesh, PartitionSpec("owner"))
        replicated = NamedSharding(mesh, PartitionSpec())
        sources = _source_arrays()
        arrays: dict[str, jax.Array | np.ndarray] = {
            name: _placed(value, sharding)
            for name, value in sources.items()
        }
        shapes = {
            name: ((27, *value.shape[1:]) if value.shape[0] else value.shape)
            for name, value in sources.items()
        }
        # Interleave host and fully addressable leaves with routed streams.
        sources["b_host"] = np.arange(12, dtype=np.dtype(">f8")).reshape(4, 3)[::-1]
        arrays["b_host"] = sources["b_host"]
        sources["h_local"] = np.asarray([3, -1], dtype=np.int32)
        arrays["h_local"] = jax.device_put(sources["h_local"], jax.local_devices()[0])
        sources["i_scalar"] = np.asarray(-0.0, dtype=np.float64)
        arrays["i_scalar"] = _placed(sources["i_scalar"], replicated)
        sources["j_replicated"] = sources["a_real"]
        arrays["j_replicated"] = _placed(sources["j_replicated"], replicated)
        shapes["j_replicated"] = (27, 3)
        expected_sources = {
            name: value[:shapes[name][0]] if name in shapes else value
            for name, value in sources.items()
        }
        expected = legacy_collection_digest(expected_sources)
        for name in shapes:
            value = arrays[name]
            assert isinstance(value, jax.Array)
            assert _logical_array_tree_fingerprint(value, shapes[name]) == (
                array_tree_fingerprint(expected_sources[name])
            )
        gather = multihost_utils.process_allgather
        broadcast = multihost_utils.broadcast_one_to_all
        rounds: list[dict[str, int]] = []
        for limit in (16, 33, 257, 779, 1023):
            sizes = {"gathers": 0, "broadcasts": 0, "largest_buffer": 0}

            def checked_gather(value: np.ndarray, *, tiled: bool) -> np.ndarray:
                assert value.nbytes * jax.process_count() <= limit
                result = np.asarray(gather(value, tiled=tiled))
                assert result.nbytes <= limit
                sizes["gathers"] += 1
                sizes["largest_buffer"] = max(sizes["largest_buffer"], result.nbytes)
                return result

            def checked_broadcast(value: np.ndarray, *, is_source: bool) -> np.ndarray:
                assert value.nbytes <= limit
                result = np.asarray(broadcast(value, is_source=is_source))
                assert result.nbytes <= limit
                sizes["broadcasts"] += 1
                sizes["largest_buffer"] = max(sizes["largest_buffer"], result.nbytes)
                return result

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(multihost_utils, "process_allgather", checked_gather)
                patch.setattr(multihost_utils, "broadcast_one_to_all", checked_broadcast)
                assert logical_array_value_collection_digest(
                    arrays, logical_shapes=shapes, maximum_chunk_bytes=limit
                ) == expected
                real = arrays["a_real"]
                assert isinstance(real, jax.Array)
                payloads = list(
                    logical_array_payloads(
                        real, shapes["a_real"], maximum_chunk_bytes=limit
                    )
                )
                assert all(0 < len(payload) <= limit for payload in payloads)
                assert b"".join(payloads) == expected_sources["a_real"].tobytes(order="C")
            if limit == 16:
                assert sizes["broadcasts"] > 0
            else:
                assert sizes["gathers"] > 0 and sizes["broadcasts"] == 0
            rounds.append({"limit": limit, **sizes})
        # Even one byte remains admissible when an all-gather cannot fit.
        boolean = arrays["e_boolean"]
        assert isinstance(boolean, jax.Array)
        payloads = list(logical_array_payloads(boolean, (27, 3), maximum_chunk_bytes=1))
        assert all(len(payload) == 1 for payload in payloads)
        assert b"".join(payloads) == expected_sources["e_boolean"].tobytes(order="C")
        real = arrays["a_real"]
        assert isinstance(real, jax.Array)
        assert list(logical_array_payloads(real, (0, 3), maximum_chunk_bytes=8)) == []
        assert logical_array_value_collection_digest(
            {"empty_prefix": real},
            logical_shapes={"empty_prefix": (0, 3)},
            maximum_chunk_bytes=8,
        ) == legacy_collection_digest({"empty_prefix": sources["a_real"][:0]})
        return {"rank": jax.process_index(), "digest": expected, "rounds": rounds}
    finally:
        jax.distributed.shutdown()
