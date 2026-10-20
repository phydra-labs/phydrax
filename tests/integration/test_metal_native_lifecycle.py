"""Buffer ownership, executable/callback lifetime, and teardown on the Metal runtime.

This is a device file: its process starts with ``METAL_STARTUP_ENVIRONMENT``.
Destructive cases (cache clearing, allocation failure, exit with work in
flight) run in a fresh child process started the same way; the child reports
the runtime identity of the device that produced its results.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config
from phydrax._jax_config import default_device
from tests._support.metal_process import json_result, run_python
from tests._support.metal_qualification import (
    metal_device,
    METAL_STARTUP_ENVIRONMENT,
    record_metal_evidence,
    record_metal_execution,
)


pytestmark = pytest.mark.metal_native

type RecordProperty = Callable[[str, object], None]

# Native teardown failures that a zero exit status must not hide.
_TEARDOWN_FAILURES = (
    "exception ignored",
    "fatal python error",
    "segmentation fault",
    "abort trap",
    "libc++abi",
    "terminating due to uncaught exception",
    "leaked semaphore",
)


@pytest.fixture
def device() -> Iterator[jax.Device]:
    selected = metal_device()
    assert config.snapshot().enable_x64
    with default_device(selected):
        yield selected


def _child(source: str) -> dict[str, object]:
    completed = run_python(source, environment=METAL_STARTUP_ENVIRONMENT, timeout=600.0)
    result = json_result(completed)
    diagnostics = completed.stderr.lower()
    for failure in _TEARDOWN_FAILURES:
        assert failure not in diagnostics, completed.stderr
    return result


def _evidence(result: dict[str, object]) -> dict[str, object]:
    evidence = result["evidence"]
    assert isinstance(evidence, dict)
    return evidence


def test_non_donated_input_survives_repeated_execution(
    device: jax.Device, record_property: RecordProperty
) -> None:
    # Dyadic values: products and the shift are exact, with or without fusion.
    original = np.arange(-2048, 2049, dtype=np.float32) / np.float32(8.0)
    source = jax.device_put(original, device)
    scale = jax.jit(lambda x, k: x * k + 1.0)
    results = [scale(source, np.float32(index)) for index in range(32)]
    for index, result in enumerate(results):
        np.testing.assert_array_equal(
            np.asarray(result), original * np.float32(index) + np.float32(1.0)
        )
    assert not source.is_deleted()
    np.testing.assert_array_equal(np.asarray(source), original)
    record_metal_execution(record_property, source, *results)


def test_donated_input_ownership_is_truthful(
    device: jax.Device, record_property: RecordProperty
) -> None:
    original = np.arange(1024, dtype=np.float32)
    donated = jax.device_put(original, device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = jax.block_until_ready(
            jax.jit(lambda x: x * 2.0 + 1.0, donate_argnums=0)(donated)
        )
    expected = original * np.float32(2.0) + np.float32(1.0)
    np.testing.assert_array_equal(np.asarray(result), expected)
    unusable = [
        str(item.message)
        for item in caught
        if "donated buffers were not usable" in str(item.message)
    ]
    if donated.is_deleted():
        # Honored donation: the caller's handle is invalidated, never silently shared.
        assert not unusable, unusable
        with pytest.raises(RuntimeError, match="deleted"):
            np.asarray(donated)
    else:
        # Declined donation must be reported, and the input keeps its value.
        assert unusable, "Donation was neither honored nor reported as unusable."
        np.testing.assert_array_equal(np.asarray(donated), original)
    # Work on the (possibly aliased) output stays correct after the donation.
    follow = jax.jit(lambda y: y - 1.0)(result)
    np.testing.assert_array_equal(np.asarray(follow), original * np.float32(2.0))
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result, follow)


def test_host_transfer_and_readback_do_not_alias_device_buffers(
    device: jax.Device, record_property: RecordProperty
) -> None:
    host = np.arange(256, dtype=np.float32)
    expected = host.copy()
    # An explicit copy request must hold even on unified memory; mutate before any wait.
    placed = jax.device_put(host, device, may_alias=False)
    host[:] = -1.0
    np.testing.assert_array_equal(np.asarray(placed), expected)

    readback = np.asarray(placed)
    if readback.flags.writeable:
        readback[:] = 99.0
    else:
        with pytest.raises(ValueError):
            readback[0] = 99.0
    copied = np.array(placed)
    copied[:] = 7.0
    recomputed = jax.jit(lambda x: x + 0.0)(placed)
    np.testing.assert_array_equal(np.asarray(recomputed), expected)

    # Strided and Fortran-ordered host layouts transfer by value, not by stride guess.
    strided = np.arange(64, dtype=np.int64).reshape(8, 8)[:, ::3]
    fortran = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(4, 6))
    strided_device = jax.device_put(strided, device)
    fortran_device = jax.device_put(fortran, device)
    np.testing.assert_array_equal(np.asarray(strided_device), strided)
    np.testing.assert_array_equal(np.asarray(fortran_device), fortran)
    transposed = jax.jit(lambda x: x.T)(fortran_device)
    np.testing.assert_array_equal(np.asarray(transposed), fortran.T)
    record_metal_execution(
        record_property, placed, recomputed, strided_device, fortran_device, transposed
    )


def test_device_copy_is_independent_of_deleted_source(
    device: jax.Device, record_property: RecordProperty
) -> None:
    original = np.arange(512, dtype=np.uint64) * np.uint64(2**40 + 3)
    source = jax.device_put(original, device)
    explicit = jnp.array(source, copy=True)
    transferred = jax.device_put(source, device, may_alias=False)
    source.delete()
    assert source.is_deleted()
    with pytest.raises(RuntimeError, match="deleted"):
        np.asarray(source)
    for survivor in (explicit, transferred):
        assert not survivor.is_deleted()
        np.testing.assert_array_equal(np.asarray(survivor), original)
    incremented = jax.jit(lambda x: x + np.uint64(1))(explicit)
    np.testing.assert_array_equal(np.asarray(incremented), original + np.uint64(1))
    record_metal_execution(record_property, explicit, transferred, incremented)


def test_zero_copy_export_is_explicit(
    device: jax.Device, record_property: RecordProperty
) -> None:
    """A no-copy request either truly shares memory or refuses; it never copies."""
    original = np.arange(64, dtype=np.float32)
    array = jax.device_put(original, device)
    refusals = (BufferError, RuntimeError, TypeError, ValueError, NotImplementedError)
    try:
        pointer: int | None = array.unsafe_buffer_pointer()
    except refusals as error:
        assert str(error)
        pointer = None
    try:
        exported = np.from_dlpack(array, copy=False)
    except refusals as error:
        assert str(error)
    else:
        assert pointer is not None, (
            "A no-copy export must expose a verifiable device buffer."
        )
        assert exported.ctypes.data == pointer
        np.testing.assert_array_equal(exported, original)

    host = np.arange(32, dtype=np.float32)
    try:
        imported = jax.dlpack.from_dlpack(host, device=device, copy=False)
    except refusals as error:
        assert str(error)
        imported = jax.device_put(host, device)
    else:
        assert imported.devices() == {device}
        assert imported.unsafe_buffer_pointer() == host.ctypes.data
    np.testing.assert_array_equal(np.asarray(imported), host)
    record_metal_execution(record_property, array, imported)


_RELEASE = """
import gc
import json
import weakref

import jax
import numpy as np

from tests._support.metal_qualification import metal_device, metal_runtime_evidence

device = metal_device()


class Owner:
    # Array-bearing host state reachable only through one compiled callback.
    def __init__(self, size):
        self.payload = np.full(size, size, np.float32)


baseline = list(jax.live_arrays())
owners, executables = [], []
for size in range(1, 25):
    owner = Owner(size)

    def host(value, owner=owner):
        return np.asarray(value + owner.payload[: value.shape[0]], np.float32)

    def program(x, host=host):
        return jax.pure_callback(host, jax.ShapeDtypeStruct(x.shape, x.dtype), x) * 2.0

    compiled_program = jax.jit(program)
    x = jax.device_put(np.ones(size, np.float32), device)
    traced = compiled_program(x)
    explicit = compiled_program.lower(x).compile()
    lowered = explicit(x)
    for result in (traced, lowered):
        np.testing.assert_array_equal(np.asarray(result), np.full(size, 2.0 * (1 + size), np.float32))
    owners.append(weakref.ref(owner))
    executables.append(weakref.ref(explicit))
    del owner, host, program, compiled_program, explicit, x, traced, lowered, result

jax.clear_caches()
for _ in range(3):
    gc.collect()
assert [reference for reference in owners if reference() is not None] == []
assert [reference for reference in executables if reference() is not None] == []
retained = [array for array in jax.live_arrays() if not any(array is kept for kept in baseline)]
assert retained == [], [array.shape for array in retained]
final = jax.jit(lambda value: value + 1.0)(jax.device_put(np.zeros(4, np.float32), device))
np.testing.assert_array_equal(np.asarray(final), np.ones(4, np.float32))
print(json.dumps({"evidence": metal_runtime_evidence(final)}))
"""


def test_repeated_compilation_releases_callbacks_and_executables(
    device: jax.Device, record_property: RecordProperty
) -> None:
    result = _child(_RELEASE)
    record_metal_evidence(record_property, _evidence(result))


_ALLOCATION = """
import json

import jax
import numpy as np

from tests._support.metal_qualification import metal_device, metal_runtime_evidence

device = metal_device()
# This shape requests a 1 TiB output. Admission must refuse it without ever
# dispatching an allocation or a kernel; this is not an OS out-of-memory test.
oversized = jax.jit(lambda fill: jax.lax.iota(np.float32, 2**38) * fill)
try:
    oversized.lower(jax.ShapeDtypeStruct((), np.float32)).compile()
except jax.errors.JaxRuntimeError as error:
    message = str(error)
else:
    raise AssertionError("A 1 TiB output was admitted for device execution.")
assert "RESOURCE_EXHAUSTED" in message and "phydrax-metal" in message and "not executed" in message, message
recovered = jax.jit(lambda value: value * 3.0)(jax.device_put(np.arange(8, dtype=np.float32), device))
np.testing.assert_array_equal(np.asarray(recovered), np.arange(8, dtype=np.float32) * 3.0)
print(json.dumps({"evidence": metal_runtime_evidence(recovered), "error": message[:2000]}))
"""


def test_allocation_failure_is_reported_and_runtime_recovers(
    device: jax.Device, record_property: RecordProperty
) -> None:
    result = _child(_ALLOCATION)
    error = result["error"]
    assert isinstance(error, str) and error
    record_property("metal_allocation_failure", error)
    record_metal_evidence(record_property, _evidence(result))


_UNSUPPORTED_ELEMENT = """
import json
import jax
import numpy as np

from tests._support.metal_qualification import metal_device, metal_runtime_evidence

device = metal_device()
module = 'module { func.func @main(%arg0: tensor<index>) -> tensor<index> { return %arg0 : tensor<index> } }'
try:
    device.client.compile_and_load(module, [device])
except jax.errors.JaxRuntimeError as error:
    message = str(error)
else:
    raise AssertionError("An unsupported index tensor was admitted.")
assert "phydrax-metal" in message and "index" in message and "not supported" in message, message
recovered = jax.jit(lambda value: value * 3)(jax.device_put(np.arange(4, dtype=np.float32), device))
np.testing.assert_array_equal(np.asarray(recovered), np.arange(4, dtype=np.float32) * 3)
print(json.dumps({"evidence": metal_runtime_evidence(recovered)}))
"""


def test_unsupported_element_type_refuses_without_aborting(
    device: jax.Device, record_property: RecordProperty
) -> None:
    result = _child(_UNSUPPORTED_ELEMENT)
    record_metal_evidence(record_property, _evidence(result))


_EXIT = """
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

from tests._support.metal_qualification import metal_device, metal_runtime_evidence

device = metal_device()
effects = Path(sys.argv[1])


def append(value):
    with effects.open("a", encoding="utf-8") as stream:
        stream.write(f"{float(value)!r}\\n")
    return np.asarray(value, np.float32)


@jax.jit
def chain(matrix, scale):
    for _ in range(50):
        matrix = jnp.tanh(matrix @ matrix * scale)
    total = jnp.sum(matrix)
    return io_callback(append, jax.ShapeDtypeStruct((), np.float32), total, ordered=True)

matrix = jax.device_put(np.eye(2048, dtype=np.float32), device)
settled = jax.block_until_ready(chain(matrix, np.float32(0.5)))
print(json.dumps({"evidence": metal_runtime_evidence(settled), "settled": float(settled)}), flush=True)
# Leave live arrays, a live executable, and dispatched but unawaited work with a
# pending ordered host effect; normal interpreter exit must drain or release them.
in_flight = [chain(matrix, np.float32(0.25)) for _ in range(4)]
"""


def test_process_exit_with_live_and_inflight_work(
    device: jax.Device, record_property: RecordProperty, tmp_path: Path
) -> None:
    effects = tmp_path / "effects.txt"
    completed = run_python(
        _EXIT, str(effects), environment=METAL_STARTUP_ENVIRONMENT, timeout=600.0
    )
    result = json_result(completed)
    diagnostics = completed.stderr.lower()
    for failure in _TEARDOWN_FAILURES:
        assert failure not in diagnostics, completed.stderr
    # Ordered host effects of every dispatched execution complete before exit.
    recorded = effects.read_text(encoding="utf-8").splitlines()
    assert len(recorded) == 5, recorded
    assert float(recorded[0]) == result["settled"]
    record_metal_evidence(record_property, _evidence(result))
