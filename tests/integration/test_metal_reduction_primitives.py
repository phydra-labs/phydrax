"""Bounded primitive qualification on the real, explicitly selected MPS device.

Host NumPy calculations are independent references, never numerical fallbacks.
Run this device file with METAL_STARTUP_ENVIRONMENT (mps only, x64 enabled).
"""

from __future__ import annotations

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


@pytest.fixture
def device() -> Iterator[jax.Device]:
    selected = metal_device()
    with default_device(selected):
        yield selected


def _put(value: np.ndarray, device: jax.Device) -> jax.Array:
    result = jax.device_put(value, device)
    assert result.dtype == value.dtype
    return result


def _assert_float_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual.view(np.uint32)[valid], expected.view(np.uint32)[valid]
    )


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize(
    "case", ["ordinary", "empty-reduced", "no-axes", "scalar", "empty-output"]
)
def test_reduce_extrema_initial_values_and_empty_axes(
    device: jax.Device, record_property: RecordProperty, kind: str, case: str
) -> None:
    shape, axes = {
        "ordinary": ((2, 3), (1,)),
        "empty-reduced": ((2, 0), (1,)),
        "no-axes": ((2, 3), ()),
        "scalar": ((), ()),
        "empty-output": ((0, 3), (1,)),
    }[case]
    host = (np.arange(np.prod(shape, dtype=np.int64), dtype=np.float32) - 3).reshape(
        shape
    )
    # lax.reduce requires an identity init; StableHLO's reduce schedule may
    # insert any number of init leaves (spec.md, reduce / Semantics).
    # Unlike NumPy's seed-once initial, a nonidentity init is not portable:
    # https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce.html
    initial = np.float32(-np.inf if kind == "max" else np.inf)
    reducer = lax.max if kind == "max" else lax.min
    reference = np.maximum if kind == "max" else np.minimum
    result = jax.jit(lambda x, init: lax.reduce(x, init, reducer, axes))(
        _put(host, device), _put(np.asarray(initial), device)
    )
    expected = reference.reduce(host, axis=axes, initial=initial)
    _assert_float_bits(np.asarray(result), np.asarray(expected, dtype=np.float32))
    record_metal_execution(record_property, result)


def test_neutral_sum_and_mean_scientific_axes(
    device: jax.Device, record_property: RecordProperty
) -> None:
    # Every partial sum is exactly representable, irrespective of reassociation.
    # Ordinary add reductions must not acquire a compensation policy.
    host = np.asarray([[[1, 2, 3], [4, 5, 6]], [[-4, -3, -2], [7, 8, 9]]], np.float32)
    cancellation = np.asarray([2**12, 1, -(2**12)], np.float32)
    empty = np.empty((2, 0), np.float32)

    @jax.jit
    def program(x: jax.Array, y: jax.Array, z: jax.Array) -> tuple[jax.Array, ...]:
        return (
            jnp.sum(x, axis=(0, 2)),
            jnp.mean(x, axis=(0, 2)),
            jnp.sum(y),
            jnp.sum(z, axis=1),
        )

    results = program(_put(host, device), _put(cancellation, device), _put(empty, device))
    references = (
        host.astype(np.float64).sum(axis=(0, 2)),
        host.astype(np.float64).mean(axis=(0, 2)),
        np.asarray(1, np.float32),
        np.zeros(2, np.float32),
    )
    for result, expected in zip(results, references, strict=True):
        assert result.dtype == np.float32
        np.testing.assert_allclose(np.asarray(result), expected, rtol=2**-22, atol=0)
    record_metal_execution(record_property, *results)


@pytest.mark.parametrize("large_index", [0, 500_000, 1_000_000])
def test_long_mixed_magnitude_sum_host64_accuracy(
    device: jax.Device, record_property: RecordProperty, large_index: int
) -> None:
    host = np.full(1_000_001, np.float32(1e-8), np.float32)
    host[large_index] = np.float32(1)
    expected = host.astype(np.float64).sum()
    result = jax.jit(jnp.sum)(_put(host, device))
    assert result.dtype == np.float32
    # Predeclared absolute error budget: two parts per million near unity,
    # retaining at least 99.98% of the small terms' 0.01 contribution.
    # This is an accuracy contract for this well-conditioned positive fixture,
    # not a pinned FP32 tree/order or a compensation requirement. A sequential
    # FP32 fold starting with 1 loses the entire small-term contribution.
    np.testing.assert_allclose(np.asarray(result), expected, rtol=0, atol=2e-6)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("axes", [(0,), (1,), (0, 2), (0, 1, 2)])
def test_parallel_sum_arbitrary_axes_declared_fp32(
    device: jax.Device, record_property: RecordProperty, axes: tuple[int, ...]
) -> None:
    host = ((np.arange(257 * 3 * 129) % 31) - 15).astype(np.float32).reshape(257, 3, 129)
    result = jax.jit(lambda x: jnp.sum(x, axis=axes))(_put(host, device))
    assert result.dtype == np.float32
    # All partial sums fit exactly in FP32, independently of the ordinary tree.
    expected = host.astype(np.float64).sum(axis=axes)
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize(
    "dtype_name",
    ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"],
)
@pytest.mark.parametrize("kind", ["sum", "prod"])
@pytest.mark.parametrize("axes", [(0,), (0, 2), (0, 1, 2)])
def test_associative_integer_reduction_exact_declared_width(
    device: jax.Device,
    record_property: RecordProperty,
    dtype_name: str,
    kind: str,
    axes: tuple[int, ...],
) -> None:
    dtype = np.dtype(dtype_name)
    width = np.iinfo(dtype).bits
    # Odd high-bit operands exercise multiplication and modular overflow,
    # including values far outside floating-point exact-integer ranges.
    raw = np.arange(33 * 3 * 65, dtype=np.uint64) * np.uint64(2)
    raw += np.uint64((1 << (width - 2)) + 1)
    host = raw.astype(dtype).reshape(33, 3, 65)
    reducer = lax.add if kind == "sum" else lax.mul
    initial = np.asarray(0 if kind == "sum" else 1, dtype=dtype)
    result = jax.jit(lambda x, init: lax.reduce(x, init, reducer, axes))(
        _put(host, device), _put(initial, device)
    )
    reference = np.add if kind == "sum" else np.multiply
    expected = reference.reduce(host, axis=axes, dtype=dtype)
    assert result.dtype == dtype
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("kind", ["all", "any"])
@pytest.mark.parametrize("axes", [(0,), (0, 2), (0, 1, 2)])
def test_associative_boolean_reduction_exact(
    device: jax.Device,
    record_property: RecordProperty,
    kind: str,
    axes: tuple[int, ...],
) -> None:
    host = (np.arange(257 * 3 * 129).reshape(257, 3, 129) % 7) == 0
    host[:, 1, :] = True
    host[:, 2, :] = False
    reducer = lax.bitwise_and if kind == "all" else lax.bitwise_or
    initial = np.asarray(kind == "all", dtype=np.bool_)
    result = jax.jit(lambda x, init: lax.reduce(x, init, reducer, axes))(
        _put(host, device), _put(initial, device)
    )
    reference = np.all if kind == "all" else np.any
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(np.asarray(result), reference(host, axis=axes))
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("kind", ["sum", "prod", "all", "any"])
@pytest.mark.parametrize("case", ["empty-reduced", "no-axes", "scalar", "empty-output"])
def test_associative_reduction_empty_and_no_axes(
    device: jax.Device, record_property: RecordProperty, kind: str, case: str
) -> None:
    shape, axes = {
        "empty-reduced": ((2, 0), (1,)),
        "no-axes": ((2, 3), ()),
        "scalar": ((), ()),
        "empty-output": ((0, 3), (1,)),
    }[case]
    dtype = np.dtype(np.bool_ if kind in ("all", "any") else np.int64)
    host = (np.arange(np.prod(shape, dtype=np.int64)) % 3).astype(dtype).reshape(shape)
    reducer, reference, identity = {
        "sum": (lax.add, np.add, 0),
        "prod": (lax.mul, np.multiply, 1),
        "all": (lax.bitwise_and, np.bitwise_and, True),
        "any": (lax.bitwise_or, np.bitwise_or, False),
    }[kind]
    initial = np.asarray(identity, dtype=dtype)
    result = jax.jit(lambda x, init: lax.reduce(x, init, reducer, axes))(
        _put(host, device), _put(initial, device)
    )
    expected = reference.reduce(host, axis=axes, initial=initial, dtype=dtype)
    assert result.dtype == dtype
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


_WINDOW_CASES = {
    "padded": ((4,), (3,), (2,), ((1, 2),), (1,), (1,)),
    "cumulative": ((4,), (4,), (1,), ((3, 0),), (1,), (1,)),
    "dilated": ((4,), (3,), (1,), ((2, 1),), (2,), (2,)),
    "cropped-two-axis": ((2, 3), (2, 2), (1, 2), ((1, 0), (-1, 2)), (2, 1), (1, 2)),
    "empty-input": ((0,), (2,), (1,), ((1, 2),), (1,), (1,)),
    "scalar": ((), (), (), (), (), ()),
}


def _window_reference(
    host: np.ndarray,
    initial: np.float32,
    kind: str,
    window: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    base_dilation: tuple[int, ...],
    window_dilation: tuple[int, ...],
) -> np.ndarray:
    shape = tuple(
        max(
            0,
            (
                (0 if size == 0 else (size - 1) * base + 1)
                + low
                + high
                - ((width - 1) * dilation + 1)
            )
            // stride
            + 1,
        )
        for size, base, (low, high), width, dilation, stride in zip(
            host.shape,
            base_dilation,
            padding,
            window,
            window_dilation,
            strides,
            strict=True,
        )
    )
    result = np.empty(shape, np.float32)
    combine = {"max": np.maximum, "min": np.minimum, "sum": np.add, "prod": np.multiply}[
        kind
    ]
    # This small host-only oracle enumerates specified padding/dilation slots;
    # it does not call JAX, MLX, or the plugin under qualification.
    # All callers use neutral init values: extra init leaves cannot change the
    # result, while padding/dilation slots still contribute the specified init.
    for output in np.ndindex(shape):
        accumulator = initial
        for coordinate in np.ndindex(window):
            padded = tuple(
                o * s - pad[0] + w * dilation
                for o, s, pad, w, dilation in zip(
                    output, strides, padding, coordinate, window_dilation, strict=True
                )
            )
            valid = all(
                p >= 0 and p % base == 0 and p // base < size
                for p, base, size in zip(padded, base_dilation, host.shape, strict=True)
            )
            index = tuple(
                p // base for p, base in zip(padded, base_dilation, strict=True)
            )
            value = host[index] if valid else initial
            accumulator = np.float32(combine(accumulator, value))
        result[output] = accumulator
    return result


@pytest.mark.parametrize("kind", ["max", "min", "sum", "prod"])
@pytest.mark.parametrize("case", list(_WINDOW_CASES))
def test_reduce_window_padding_dilation_and_initial_values(
    device: jax.Device, record_property: RecordProperty, kind: str, case: str
) -> None:
    shape, window, strides, padding, base, dilation = _WINDOW_CASES[case]
    size = int(np.prod(shape, dtype=np.int64))
    host = ((np.arange(size, dtype=np.float32) % 5) - 2).reshape(shape)
    initial = np.float32({"max": -np.inf, "min": np.inf, "sum": 0, "prod": 1}[kind])
    reducer = {"max": lax.max, "min": lax.min, "sum": lax.add, "prod": lax.mul}[kind]
    result = jax.jit(
        lambda x, init: lax.reduce_window(
            x,
            init,
            reducer,
            window,
            strides,
            padding,
            base_dilation=base,
            window_dilation=dilation,
        )
    )(_put(host, device), _put(np.asarray(initial), device))
    expected = _window_reference(
        host, initial, kind, window, strides, padding, base, dilation
    )
    _assert_float_bits(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_reduce_extrema_nan_and_signed_zero(
    device: jax.Device, record_property: RecordProperty, kind: str
) -> None:
    host = np.asarray(
        [[-0.0, 0.0], [-0.0, -0.0], [0.0, 0.0], [np.nan, 1.0], [1.0, np.nan]], np.float32
    )
    reducer = lax.max if kind == "max" else lax.min
    initial = np.asarray(-np.inf if kind == "max" else np.inf, np.float32)
    result = jax.jit(lambda x, init: lax.reduce(x, init, reducer, (1,)))(
        _put(host, device), _put(initial, device)
    )
    expected = np.asarray(
        [0.0 if kind == "max" else -0.0, -0.0, 0.0, np.nan, np.nan], np.float32
    )
    _assert_float_bits(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_paired_arg_reduction_full_width_ties_and_nan(
    device: jax.Device, record_property: RecordProperty, kind: str, index_kind: str
) -> None:
    dtype = np.dtype(index_kind)
    offset = 2**63 if index_kind == "uint64" else 2**60
    indices = np.asarray([[9, 3, 5], [8, 6, 2], [7, 1, 4]], dtype=dtype) + np.asarray(
        offset, dtype=dtype
    )
    host = np.asarray([[4, 4, 4], [np.nan, 5, np.nan], [-0.0, 0.0, -0.0]], np.float32)
    initial = np.asarray(-np.inf if kind == "max" else np.inf, np.float32)
    initial_index = np.asarray(np.iinfo(dtype).max, dtype=dtype)

    def reducer(
        left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        a, ai = left
        b, bi = right
        nan_a, nan_b = a != a, b != b
        better = (a > b) if kind == "max" else (a < b)
        better = better | (nan_a & ~nan_b)
        equal = (a == b) | (nan_a & nan_b)
        choose_left = better | (equal & (ai < bi))
        return lax.select(choose_left, a, b), lax.select(choose_left, ai, bi)

    @jax.jit
    def program(
        x: jax.Array, ids: jax.Array, init: jax.Array, init_id: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return lax.reduce((x, ids), (init, init_id), reducer, (1,))

    result, index = program(
        _put(host, device),
        _put(indices, device),
        _put(initial, device),
        _put(initial_index, device),
    )
    assert index.dtype == dtype
    np.testing.assert_array_equal(np.asarray(index), indices[np.arange(3), [1, 2, 1]])
    _assert_float_bits(np.asarray(result), np.asarray([4, np.nan, 0.0], np.float32))
    record_metal_execution(record_property, result, index)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_jax_arg_reduction_declares_int64_and_uses_first_tie(
    device: jax.Device, record_property: RecordProperty, kind: str
) -> None:
    host = np.asarray([[2, 2, 2], [np.nan, 3, np.nan], [-0.0, 0.0, -0.0]], np.float32)
    operation = jnp.argmax if kind == "max" else jnp.argmin
    result = jax.jit(lambda x: operation(x, axis=1))(_put(host, device))
    assert result.dtype == np.int64
    np.testing.assert_array_equal(np.asarray(result), np.zeros(3, np.int64))
    record_metal_execution(record_property, result)


def test_sign_preserves_nan_signed_zero_and_subnormal_sign(
    device: jax.Device, record_property: RecordProperty
) -> None:
    bits = np.asarray(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x3F800000,
            0xBF800000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0xFFC05678,
        ],
        np.uint32,
    )
    host = bits.view(np.float32)
    result = jax.jit(lax.sign)(_put(host, device))
    # NumPy sign need not preserve -0, whereas StableHLO sign must.
    expected = np.where(host == 0, host, np.sign(host)).astype(np.float32)
    _assert_float_bits(np.asarray(result), expected)
    # Also require the unchanged payload for the quiet NaNs in this fixture.
    np.testing.assert_array_equal(np.asarray(result).view(np.uint32)[-2:], bits[-2:])
    record_metal_execution(record_property, result)


def test_floating_remainder_extreme_exponents_subnormals_and_special_values(
    device: jax.Device, record_property: RecordProperty
) -> None:
    positive_bits = np.asarray(
        [
            0,
            1,
            2,
            0x007FFFFF,
            0x00800000,
            0x00800001,
            0x3DCCCCCD,
            0x3F800000,
            0x40400000,
            0x4B800001,
            0x7F000001,
            0x7F7FFFFF,
            0x7F800000,
            0x7FC01234,
        ],
        np.uint32,
    )
    signed = np.concatenate((positive_bits, positive_bits | np.uint32(0x80000000))).view(
        np.float32
    )
    lhs, rhs = np.broadcast_arrays(signed[:, None], signed[None, :])
    lhs, rhs = lhs.copy(), rhs.copy()
    with np.errstate(invalid="ignore", divide="ignore", under="ignore"):
        expected = np.fmod(lhs, rhs)
    result = jax.jit(lax.rem)(_put(lhs, device), _put(rhs, device))
    _assert_float_bits(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_integer_remainder_never_passes_through_float(
    device: jax.Device, record_property: RecordProperty, index_kind: str
) -> None:
    dtype = np.dtype(index_kind)
    if index_kind == "int64":
        left = [-(2**63), -(2**61) - 17, -17, 0, 17, 2**61 + 17, 2**63 - 1]
        right = [7, -(2**53) - 1, 5, -7, -5, 2**53 + 3, -11]
    else:
        left = [0, 17, 2**53 + 1, 2**63, 2**64 - 1, 2**64 - 5, 2**61 + 17]
        right = [7, 5, 11, 2**53 + 1, 2**63 + 3, 17, 2**53 + 3]
    expected = np.asarray(
        [
            (abs(a) % abs(b)) * (-1 if a < 0 else 1)
            for a, b in zip(left, right, strict=True)
        ],
        dtype=dtype,
    )
    result = jax.jit(lax.rem)(
        _put(np.asarray(left, dtype=dtype), device),
        _put(np.asarray(right, dtype=dtype), device),
    )
    assert result.dtype == dtype
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


def test_cbrt_fp32_ulp_bound_and_special_values(
    device: jax.Device, record_property: RecordProperty
) -> None:
    bits = [0, 1, 2, 0x007FFFFF, 0x7F800000, 0x7FC01234]
    bits += [
        (exponent << 23) | fraction
        for exponent in (1, 2, 5, 32, 64, 126, 127, 128, 191, 252, 254)
        for fraction in (0, 1, 0x400000, 0x7FFFFF)
    ]
    positive = np.asarray(bits, np.uint32)
    host = np.concatenate((positive, positive | np.uint32(0x80000000))).view(np.float32)
    host = np.concatenate(
        (host, np.asarray([-27, -8, -1, 0.125, 1, 8, 27, 64], np.float32))
    )
    expected = np.cbrt(host.astype(np.float64)).astype(np.float32)
    result = jax.jit(lax.cbrt)(_put(host, device))
    actual = np.asarray(result)
    assert actual.dtype == np.float32
    special = ~np.isfinite(expected) | (expected == 0)
    _assert_float_bits(actual[special], expected[special])
    finite = ~special
    np.testing.assert_array_equal(
        np.signbit(actual[finite]), np.signbit(expected[finite])
    )
    actual_magnitude = (actual[finite].view(np.uint32) & np.uint32(0x7FFFFFFF)).astype(
        np.int64
    )
    expected_magnitude = (
        expected[finite].view(np.uint32) & np.uint32(0x7FFFFFFF)
    ).astype(np.int64)
    ulp_error = np.abs(actual_magnitude - expected_magnitude)
    assert np.all(ulp_error <= 2), f"FP32 cbrt exceeds two ULP: {ulp_error.tolist()}"
    record_metal_execution(record_property, result)
