import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from phydrax._external_exchange import read_exchange, write_exchange
from phydrax._external_runtime import NativeWorkerPolicy
from phydrax.meshing import MeshingFailure, MeshingFailureCategory, MeshingLimits
from phydrax.meshing.providers._worker import ProviderWorker


# A protocol-faithful stand-in for a native worker: stdout is the private
# control channel, library noise goes to stderr, arrays travel as NPY files.
_FAKE_WORKER = """#!{python}
import hashlib, json, os, sys, time
import numpy as np

PREFIX = "@phydrax-worker "
protocol = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
mode = sys.argv[1]


def send(record):
    protocol.write(PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":")) + "\\n")
    protocol.flush()


if mode == "exit":
    sys.exit(7)
send({{"hello": {{"identity": {{"worker": "fake", "version": "r1"}},
                "memory_enforcement": "peak-rss-audit", "ranks": 1}}, "ok": True}})
for line in sys.stdin:
    request = json.loads(line)
    sequence = request["sequence"]
    if request["operation"] == "close":
        send({{"elapsed_seconds": 0.0, "ok": True, "peak_rss_bytes": 1, "result": {{}},
              "sequence": sequence}})
        break
    print("upstream library noise", flush=True)
    if mode == "sleep":
        time.sleep(60)
    if mode == "slow":
        time.sleep(0.5)
    if mode == "reject":
        send({{"error": "no such route", "kind": "unsupported", "ok": False,
              "peak_rss_bytes": 1, "sequence": sequence}})
        mode = "echo"
        continue
    values = 2.0 * np.load(os.path.join(request["input"], "x.npy"))
    with open(os.path.join(request["output"], "y.npy"), "wb") as stream:
        np.lib.format.write_array(stream, values, version=(1, 0))
    digest = "0" * 64 if mode == "corrupt" else hashlib.sha256(values.tobytes()).hexdigest()
    record = {{"dtype": values.dtype.str, "name": "y", "sha256": digest,
              "shape": list(values.shape)}}
    with open(os.path.join(request["output"], "manifest.json"), "w") as stream:
        stream.write(json.dumps({{"arrays": [record], "parts": []}}, sort_keys=True,
                                separators=(",", ":")))
    send({{"elapsed_seconds": 0.0, "ok": True,
          "peak_rss_bytes": 2**40 if mode == "memory" else 1,
          "result": {{"pid": os.getpid()}}, "sequence": sequence}})
"""


def _worker(tmp_path: Any, mode: Any, **options: Any) -> Any:
    path = tmp_path / "fake-worker"
    path.write_text(_FAKE_WORKER.format(python=sys.executable), encoding="utf-8")
    path.chmod(0o755)
    return ProviderWorker(
        "fake",
        executable=str(path),
        environment_variable="PHYDRAX_FAKE_WORKER",
        default_executable="phydrax-fake-worker",
        build_hint="Build the fake worker.",
        arguments=(mode,),
        **options,
    )


def test_worker_session_is_reused_and_identity_is_probed_once(tmp_path: Any) -> None:
    worker = _worker(tmp_path, "echo")
    values = np.arange(6, dtype=np.float64).reshape(3, 2)
    first = worker.call("scale", {"factor": 2}, {"x": values}, limits=MeshingLimits())
    second = worker.call("scale", {"factor": 2}, {"x": values}, limits=MeshingLimits())
    worker.close()

    assert worker.launches == 1
    assert first.result["pid"] == second.result["pid"]
    assert first.evidence["session_id"] == second.evidence["session_id"]
    assert (first.sequence, second.sequence) == (1, 2)
    np.testing.assert_array_equal(second.arrays["y"], 2.0 * values)
    assert not second.arrays["y"].flags.writeable


def test_worker_timeout_is_refused_and_the_next_call_relaunches(tmp_path: Any) -> None:
    worker = _worker(tmp_path, "sleep")
    values = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(MeshingFailure) as failure:
        worker.call(
            "scale", {}, {"x": values}, limits=MeshingLimits(maximum_wall_seconds=0.5)
        )

    assert failure.value.category is MeshingFailureCategory.TIMED_OUT
    # ty: ignore[unresolved-attribute]
    assert "upstream library noise" in failure.value.__cause__.evidence["log"]
    # ty: ignore[unresolved-attribute]
    first_session = failure.value.__cause__.evidence["session_id"]
    assert worker.session().identity.session_id != first_session
    assert worker.launches == 2
    worker.close()


def test_worker_rejection_maps_its_kind_and_keeps_the_session(tmp_path: Any) -> None:
    worker = _worker(tmp_path, "reject")
    values = np.ones((1, 1), dtype=np.float64)

    with pytest.raises(MeshingFailure) as failure:
        worker.call("scale", {}, {"x": values}, limits=MeshingLimits())
    result = worker.call("scale", {}, {"x": values}, limits=MeshingLimits())
    worker.close()

    assert failure.value.category is MeshingFailureCategory.UNSUPPORTED_CAPABILITY
    assert result.sequence == 2 and worker.launches == 1


@pytest.mark.parametrize(
    ("mode", "category"),
    (
        ("exit", MeshingFailureCategory.PROVIDER_EXECUTION_FAILED),
        ("corrupt", MeshingFailureCategory.PROVIDER_EXECUTION_FAILED),
        ("memory", MeshingFailureCategory.RESOURCE_EXHAUSTED),
    ),
)
def test_worker_failures_surface_as_meshing_failures_with_evidence(
    tmp_path: Any, mode: Any, category: Any
) -> None:
    worker = _worker(
        tmp_path, mode, policy=NativeWorkerPolicy(maximum_memory_bytes=2**30)
    )

    with pytest.raises(MeshingFailure) as failure:
        worker.call(
            "scale", {}, {"x": np.ones((2,), dtype=np.float64)}, limits=MeshingLimits()
        )

    assert failure.value.category is category
    # ty: ignore[unresolved-attribute]
    evidence = failure.value.__cause__.evidence
    if mode == "exit":
        assert evidence["returncode"] == 7
    else:
        assert evidence["kind"] in ("protocol", "resource")
    worker.close()


def test_concurrent_calls_share_one_session_in_sequence(tmp_path: Any) -> None:
    worker = _worker(tmp_path, "echo")
    barrier = threading.Barrier(8)

    def run(index: Any) -> Any:
        values = np.full((2, 2), index, dtype=np.float64)
        barrier.wait()
        return worker.call("scale", {}, {"x": values}, limits=MeshingLimits())

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(8)))
    worker.close()

    assert worker.launches == 1
    assert sorted(result.sequence for result in results) == list(range(1, 9))
    assert len({result.result["pid"] for result in results}) == 1
    for index, result in enumerate(results):
        np.testing.assert_array_equal(result.arrays["y"], np.full((2, 2), 2.0 * index))


def test_concurrent_calls_replace_exhausted_sessions_without_failure(
    tmp_path: Any,
) -> None:
    worker = _worker(tmp_path, "echo", policy=NativeWorkerPolicy(maximum_calls=1))
    barrier = threading.Barrier(4)
    values = np.ones((2,), dtype=np.float64)

    def run(_: Any) -> Any:
        barrier.wait()
        return worker.call("scale", {}, {"x": values}, limits=MeshingLimits())

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    worker.close()

    assert worker.launches == 4
    assert len({result.evidence["session_id"] for result in results}) == 4
    assert all(result.sequence == 1 for result in results)


def test_session_close_waits_for_the_call_in_flight(tmp_path: Any) -> None:
    worker = _worker(tmp_path, "slow")
    session = worker.session()
    values = np.arange(3, dtype=np.float64)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            session.call,
            "scale",
            {},
            {"x": values},
            timeout=30.0,
            maximum_input_bytes=4096,
            maximum_output_bytes=4096,
        )
        while session.call_count == 0:
            time.sleep(0.001)
        session.close()
        result = pending.result()

    assert session.closed
    np.testing.assert_array_equal(result.arrays["y"], 2.0 * values)
    replacement = worker.call("scale", {}, {"x": values}, limits=MeshingLimits())
    worker.close()
    assert replacement.evidence["session_id"] != result.evidence["session_id"]
    assert worker.launches == 2


def test_missing_worker_is_provider_unavailable(tmp_path: Any) -> None:
    worker = ProviderWorker(
        "fake",
        executable=str(tmp_path / "missing-worker"),
        environment_variable="PHYDRAX_FAKE_WORKER",
        default_executable="phydrax-fake-worker",
        build_hint="Build the fake worker.",
    )

    with pytest.raises(MeshingFailure) as failure:
        worker.call("scale", {}, {}, limits=MeshingLimits())

    assert failure.value.category is MeshingFailureCategory.PROVIDER_UNAVAILABLE


def test_exchange_round_trip_verifies_checksums_and_declared_entries(
    tmp_path: Any,
) -> None:
    arrays = {
        "coordinates": np.linspace(0.0, 1.0, 12).reshape(4, 3),
        "ids": np.array((9, 3, 7), dtype=np.int64),
        "flags": np.zeros((0, 2), dtype=np.uint8),
    }
    manifest = write_exchange(tmp_path, arrays, maximum_bytes=4096)
    contents = read_exchange(tmp_path, maximum_bytes=4096)

    assert contents.manifest.manifest_sha256 == manifest.manifest_sha256
    for name, values in arrays.items():
        np.testing.assert_array_equal(contents.arrays[name], values)
        assert contents.arrays[name].dtype == values.dtype
    with pytest.raises(ValueError, match="maximum_bytes"):
        read_exchange(tmp_path, maximum_bytes=64)

    payload = bytearray((tmp_path / "ids.npy").read_bytes())
    payload[-1] ^= 1
    (tmp_path / "ids.npy").write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="checksum"):
        read_exchange(tmp_path, maximum_bytes=4096)
    payload[-1] ^= 1
    (tmp_path / "ids.npy").write_bytes(bytes(payload))
    (tmp_path / "undeclared.npy").write_bytes(b"")
    with pytest.raises(ValueError, match="undeclared"):
        read_exchange(tmp_path, maximum_bytes=4096)
