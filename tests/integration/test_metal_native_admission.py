"""Static custom-call admission on the real, explicitly selected MPS device.

Inherited CPU LAPACK custom-call targets are refused when the program is
compiled, wherever they appear (entry, unselected branch, branch nested in a
loop), so no executable can run them on the CPU under the GPU device label.
Admitted GPU targets keep executing on the device. Host NumPy calculations and
published known-answer vectors are independent references.
Run this device file with METAL_STARTUP_ENVIRONMENT (mps only, x64 enabled).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from phydrax._jax_config import default_device
from tests._support.metal_qualification import metal_device, record_metal_execution


pytestmark = pytest.mark.metal_native

type RecordProperty = Callable[[str, object], None]
type Factor = Callable[[jax.Array], jax.Array]


@pytest.fixture
def device() -> Iterator[jax.Device]:
    selected = metal_device()
    with default_device(selected):
        yield selected


def _put(value: np.ndarray, device: jax.Device) -> jax.Array:
    result = jax.device_put(value, device)
    assert result.dtype == value.dtype
    return result


# Finite, symmetric, well-conditioned input: refusal never depends on values.
_MATRIX = np.asarray([[4, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 5]], np.float32)
_SQUARE = jax.ShapeDtypeStruct((4, 4), np.float32)
_VECTOR = jax.ShapeDtypeStruct((4,), np.float32)
_STATUS = jax.ShapeDtypeStruct((), np.int32)

# Result signature of each CPU LAPACK target and the index of its 4x4 factor.
_CPU_LAPACK_TARGETS: dict[str, tuple[tuple[jax.ShapeDtypeStruct, ...], int]] = {
    "mps.eigh": ((_SQUARE, _VECTOR), 0),
    "mps.qr": ((_SQUARE, _SQUARE), 0),
    "mps.svd": ((_VECTOR, _SQUARE, _SQUARE), 1),
    "lapack_ssyevd_ffi": ((_SQUARE, _VECTOR, _STATUS), 0),
}

_PLACEMENTS = ["entry", "unselected-cond-branch", "while-nested-switch-branch"]


def _cpu_lapack_factor(target: str) -> Factor:
    results, square = _CPU_LAPACK_TARGETS[target]

    def factor(matrix: jax.Array) -> jax.Array:
        return jax.ffi.ffi_call(target, results)(matrix)[square]

    return factor


def _gpu_factor(matrix: jax.Array) -> jax.Array:
    # Float32 contraction of small integers: exact on the GPU.
    return matrix @ matrix.T + matrix


def _host_factor(matrix: np.ndarray) -> np.ndarray:
    return matrix @ matrix.T + matrix


def _program(
    placement: str, factor: Factor
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    if placement == "entry":
        return lambda matrix, selector: factor(matrix)
    if placement == "unselected-cond-branch":
        return lambda matrix, selector: lax.cond(
            selector > 0, factor, lambda value: value + 1.0, matrix
        )

    def nested(matrix: jax.Array, selector: jax.Array) -> jax.Array:
        def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            step, value = state
            branch = jnp.clip(selector, 0, 1)
            return step + 1, lax.switch(branch, [lambda v: v * 0.5, factor], value)

        return lax.while_loop(lambda state: state[0] < 2, body, (jnp.int32(0), matrix))[1]

    return nested


def _host_program(placement: str, matrix: np.ndarray) -> np.ndarray:
    # Reference for selector=1, which selects the factor branch.
    result = _host_factor(matrix)
    if placement == "while-nested-switch-branch":
        result = _host_factor(result)
    return result


@pytest.mark.parametrize("target", list(_CPU_LAPACK_TARGETS))
@pytest.mark.parametrize("placement", _PLACEMENTS)
def test_cpu_lapack_custom_call_refused_at_compile(
    device: jax.Device, record_property: RecordProperty, target: str, placement: str
) -> None:
    matrix = _put(_MATRIX, device)
    selector = _put(np.asarray(0, np.int32), device)
    refusal = (
        rf"(?s)(?=.*phydrax-metal)"
        rf"(?=.*custom_call target '{re.escape(target)}')(?=.*not supported)"
    )
    # Refused when compiled: the selector is a runtime value, so the target is
    # refused even where execution would never select its branch.
    with pytest.raises(jax.errors.JaxRuntimeError, match=refusal):
        jax.jit(_program(placement, _cpu_lapack_factor(target))).lower(
            matrix, selector
        ).compile()
    # The same control structure with a GPU branch executes on the device.
    result = jax.jit(_program(placement, _gpu_factor))(
        matrix, _put(np.asarray(1, np.int32), device)
    )
    expected = _host_program(placement, _MATRIX.astype(np.float64)).astype(np.float32)
    actual = np.asarray(result)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    record_metal_execution(record_property, result)


# Threefry-2x32 with R=20 rounds (the schedule jax.random uses and the plugin's
# mps.threefry2x32 kernel implements: five 4-round blocks, key injection after
# each). Rows copied verbatim from the Random123 known-answer file, whose row
# layout is ``name R CTR KEY EXPECTED`` with EXPECTED = (x0, x1):
# https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors#L48-L50
# JAX checks the same three answers (key, count argument order) in
# testThreefry2x32 of https://github.com/jax-ml/jax/blob/main/tests/random_test.py
# Entries: (counter0, counter1), (key0, key1), (out0, out1).
_THREEFRY2X32_20_KNOWN_ANSWERS = (
    ((0x00000000, 0x00000000), (0x00000000, 0x00000000), (0x6B200159, 0x99BA4EFE)),
    ((0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF), (0x1CB996FC, 0xBB002BE7)),
    ((0x243F6A88, 0x85A308D3), (0x13198A2E, 0x03707344), (0xC4923A9C, 0x483DF7A0)),
)


@pytest.mark.parametrize("placement", ["entry", "selected-cond-branch"])
def test_admitted_gpu_custom_call_executes_in_branches(
    device: jax.Device, record_property: RecordProperty, placement: str
) -> None:
    lanes = np.asarray(_THREEFRY2X32_20_KNOWN_ANSWERS, np.uint32)
    counts, keys, expected = lanes[:, 0], lanes[:, 1], lanes[:, 2]
    lane = jax.ShapeDtypeStruct((lanes.shape[0],), np.uint32)

    def threefry(operands: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array]:
        first, second = jax.ffi.ffi_call("mps.threefry2x32", (lane, lane))(*operands)
        return first, second

    def program(
        operands: tuple[jax.Array, ...], selector: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if placement == "entry":
            return threefry(operands)
        return lax.cond(
            selector > 0,
            threefry,
            lambda values: (values[0] ^ values[2], values[1] ^ values[3]),
            operands,
        )

    # mps.threefry2x32 operands are (key0, key1, count0, count1); results are
    # (out0, out1), i.e. Random123's (x0, x1).
    operands = tuple(
        _put(np.ascontiguousarray(column), device)
        for column in (keys[:, 0], keys[:, 1], counts[:, 0], counts[:, 1])
    )
    first, second = jax.jit(program)(operands, _put(np.asarray(1, np.int32), device))
    assert first.dtype == second.dtype == np.uint32
    np.testing.assert_array_equal(np.asarray(first), expected[:, 0])
    np.testing.assert_array_equal(np.asarray(second), expected[:, 1])
    record_metal_execution(record_property, first, second)
