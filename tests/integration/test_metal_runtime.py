"""Real-device runtime contracts of the selected Metal (``mps``) PJRT runtime.

This is a device file: its process starts with ``METAL_STARTUP_ENVIRONMENT``
(only ``mps`` selected, x64 on). Every case records the identity of the device
that produced its actual results. Independent references are host float64
arithmetic or a separate CPU process; this process has no CPU backend.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.nn import functions as _nn_functions
from jax.experimental import io_callback

from phydrax import config
from phydrax._execution_device import device_vendor
from phydrax._jax_config import default_device
from phydrax.backends import metal_environment
from tests._support.metal_process import json_result, run_python
from tests._support.metal_qualification import metal_device, record_metal_execution


pytestmark = pytest.mark.metal_native

type RecordProperty = Callable[[str, object], None]

_CPU_REFERENCE_ENVIRONMENT = {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}
_F32_UNIT_ROUNDOFF = 2.0**-24


@pytest.fixture
def device() -> Iterator[jax.Device]:
    selected = metal_device()
    assert config.snapshot().enable_x64
    with default_device(selected):
        yield selected


def _put(value: np.ndarray, device: jax.Device) -> jax.Array:
    array = jax.device_put(value, device)
    assert array.dtype == value.dtype
    return array


def _cpu_reference(source: str, payload: dict[str, object]) -> dict[str, object]:
    """Evaluate ``source`` in a separate float64-capable CPU-only process."""
    program = (
        "import json, sys\n"
        "import jax\n"
        "assert {device.platform for device in jax.devices()} == {'cpu'}\n"
        + source
        + "\nprint(json.dumps(reference(json.loads(sys.stdin.read()))))\n"
    )
    return json_result(
        run_python(
            program, environment=_CPU_REFERENCE_ENVIRONMENT, stdin=json.dumps(payload)
        )
    )


def _wrap_signed(value: int) -> int:
    return (value + 2**63) % 2**64 - 2**63


def test_runtime_identity_matches_installed_companion(
    device: jax.Device, record_property: RecordProperty
) -> None:
    report = metal_environment()
    assert report.successful, report.issues
    facts = report.facts
    assert facts.device_probed and facts.runtime_platform == "mps"
    assert facts.build_id is not None and re.fullmatch(r"[0-9a-f]{64}", facts.build_id)
    assert device.platform == "mps"
    assert device.client.platform_version == (
        f"phydrax-metal {metadata.version('phydrax-metal')} {facts.source_id}"
    )
    assert device_vendor(device) == "apple"
    assert jax.devices() == [device]
    value = jax.jit(lambda x: x + 1)(_put(np.asarray(1.5, np.float32), device))
    assert value.dtype == np.float32 and float(value) == 2.5
    record_metal_execution(record_property, value)


def test_float32_complex64_and_exact_64_bit_integers_share_one_x64_program(
    device: jax.Device, record_property: RecordProperty
) -> None:
    rng = np.random.default_rng(20261019)
    real = rng.standard_normal(16).astype(np.float32)
    complex_ = (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(
        np.complex64
    )
    signed_values = [-(2**63), -(2**53) - 1, -1, 0, 1, 2**32 + 3, 2**53 + 1, 2**63 - 1]
    unsigned_values = [
        0,
        1,
        2**32 + 5,
        2**53 + 1,
        2**63,
        2**64 - 1,
        0xDEADBEEFCAFEBABE,
        2**63 + 2**53 + 1,
    ]
    signed = np.asarray(signed_values, np.int64)
    unsigned = np.asarray(unsigned_values, np.uint64)
    multiplier = np.uint64(0x9E3779B97F4A7C15)

    @jax.jit
    def program(
        a: jax.Array, z: jax.Array, s: jax.Array, u: jax.Array, m: jax.Array
    ) -> tuple[jax.Array, ...]:
        return (
            a * a + 1.0,
            z * jnp.conj(z) + z,
            s ^ (s >> np.int64(7)),
            s + np.int64(1),
            s * np.int64(3),
            u * m + np.uint64(1),
            u >> np.uint64(33),
        )

    results = program(
        _put(real, device),
        _put(complex_, device),
        _put(signed, device),
        _put(unsigned, device),
        _put(np.asarray(multiplier), device),
    )
    dtypes = [
        np.float32,
        np.complex64,
        np.int64,
        np.int64,
        np.int64,
        np.uint64,
        np.uint64,
    ]
    assert [result.dtype for result in results] == dtypes

    real64 = real.astype(np.float64)
    complex128 = complex_.astype(np.complex128)
    # Two float32 roundings of an O(1) value: well inside four unit roundoffs.
    np.testing.assert_allclose(
        results[0], real64 * real64 + 1.0, rtol=4 * _F32_UNIT_ROUNDOFF, atol=0
    )
    expected_complex = complex128 * np.conj(complex128) + complex128
    np.testing.assert_allclose(
        results[1],
        expected_complex,
        rtol=0,
        atol=8
        * _F32_UNIT_ROUNDOFF
        * float(np.max(np.abs(complex128) ** 2 + np.abs(complex128))),
    )
    assert [int(v) for v in np.asarray(results[2])] == [
        v ^ (v >> 7) for v in signed_values
    ]
    assert [int(v) for v in np.asarray(results[3])] == [
        _wrap_signed(v + 1) for v in signed_values
    ]
    assert [int(v) for v in np.asarray(results[4])] == [
        _wrap_signed(3 * v) for v in signed_values
    ]
    assert [int(v) for v in np.asarray(results[5])] == [
        (v * int(multiplier) + 1) % 2**64 for v in unsigned_values
    ]
    assert [int(v) for v in np.asarray(results[6])] == [v >> 33 for v in unsigned_values]
    record_metal_execution(record_property, *results)


_WIDE_FLOATS = {
    # Values a float32 narrowing cannot represent, so truncation cannot pass.
    "float64": (np.float64, np.float32, 1.0 + 2.0**-40),
    "complex128": (np.complex128, np.complex64, complex(1.0 + 2.0**-40, -1.0 + 2.0**-41)),
}


def _wide_boundary(
    boundary: str,
    wide: type[np.generic],
    narrow: type[np.generic],
    value: complex,
    device: jax.Device,
) -> jax.Array:
    narrow_input = _put(np.asarray([1.0, 2.0], narrow), device)
    constant = np.asarray([value, value], wide)
    match boundary:
        case "transfer":
            return jax.device_put(constant, device)
        case "constant":
            return jax.jit(lambda x: (x + constant).astype(narrow))(narrow_input)
        case "conversion":
            return jax.jit(lambda x: x.astype(wide))(narrow_input)
        case "intermediate":
            return jax.jit(
                lambda x: (x.astype(wide) * value - x.astype(wide)).astype(narrow)
            )(narrow_input)
        case "result":
            return jax.jit(lambda x: jnp.zeros(x.shape, wide))(narrow_input)
        case _:
            raise AssertionError(boundary)


@pytest.mark.parametrize(
    "boundary", ["transfer", "constant", "conversion", "intermediate", "result"]
)
@pytest.mark.parametrize("dtype_name", ["float64", "complex128"])
def test_wide_float_is_refused_without_narrowing(
    device: jax.Device, record_property: RecordProperty, dtype_name: str, boundary: str
) -> None:
    wide, narrow, value = _WIDE_FLOATS[dtype_name]
    refusal = rf"(?s)(?=.*phydrax-metal)(?=.*{dtype_name})(?=.*not supported)"
    with pytest.raises(jax.errors.JaxRuntimeError, match=refusal):
        jax.block_until_ready(_wide_boundary(boundary, wide, narrow, value, device))
    # The refusal leaves the runtime usable for the admitted precision.
    admitted = jax.jit(lambda x: x * 2)(_put(np.asarray([0.75, -1.5], narrow), device))
    assert admitted.dtype == narrow
    np.testing.assert_array_equal(np.asarray(admitted), np.asarray([1.5, -3.0], narrow))
    record_metal_execution(record_property, admitted)


def test_float32_contraction_is_exact_where_representable_and_within_fp32_bound(
    device: jax.Device, record_property: RecordProperty
) -> None:
    # 16 * (1 + 2**-20) and every partial sum are exact in float32 under any
    # summation order, but vanish below FP16/BF16/TF32 input mantissas.
    near_one = _put(np.full((1, 16), 1.0 + 2.0**-20, np.float32), device)
    ones = _put(np.ones((16, 1), np.float32), device)
    representable = jax.jit(jnp.matmul)(near_one, ones)
    assert representable.dtype == np.float32
    assert np.asarray(representable).item() == 16.0 + 2.0**-16

    rng = np.random.default_rng(7)
    left = rng.standard_normal((48, 128)).astype(np.float32)
    right = rng.standard_normal((128, 40)).astype(np.float32)
    exact = left.astype(np.float64) @ right.astype(np.float64)
    depth = left.shape[1]
    gamma = depth * _F32_UNIT_ROUNDOFF / (1.0 - depth * _F32_UNIT_ROUNDOFF)
    bound = gamma * (np.abs(left).astype(np.float64) @ np.abs(right).astype(np.float64))
    implicit = jax.jit(jnp.matmul)(_put(left, device), _put(right, device))
    explicit = jax.jit(lambda a, b: jnp.matmul(a, b, precision=lax.Precision.HIGHEST))(
        _put(left, device), _put(right, device)
    )
    for product in (implicit, explicit):
        assert product.dtype == np.float32
        error = np.abs(np.asarray(product, np.float64) - exact)
        assert np.all(error <= bound), float(np.max(error / bound))
    record_metal_execution(record_property, representable, implicit, explicit)


def test_integer_contraction_is_exact_or_explicitly_refused(
    device: jax.Device, record_property: RecordProperty
) -> None:
    # Products near 2**60 cannot pass through any floating-point contraction.
    left_values = [[2**30 + 1, -(2**29) - 3], [2**28 + 7, 2**30 - 5]]
    right_values = [[2**30 - 1, 2**29 + 11], [-(2**30) + 9, 2**27 + 1]]
    left = _put(np.asarray(left_values, np.int64), device)
    right = _put(np.asarray(right_values, np.int64), device)
    expected = [
        [sum(left_values[i][k] * right_values[k][j] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]
    try:
        product = jax.block_until_ready(jax.jit(jnp.matmul)(left, right))
    except jax.errors.JaxRuntimeError as error:
        assert re.search(r"(?s)(?=.*phydrax-metal)(?=.*not supported)", str(error)), str(
            error
        )
        evidence = jax.jit(lambda a: a * np.int64(3))(left)
        assert np.asarray(evidence).tolist() == [
            [3 * v for v in row] for row in left_values
        ]
    else:
        assert product.dtype == np.int64
        assert np.asarray(product).tolist() == expected
        evidence = product
    record_metal_execution(record_property, evidence)


def test_full_width_integer_keys_sort_stably_and_compute_exactly(
    device: jax.Device, record_property: RecordProperty
) -> None:
    unsigned_values = [
        2**64 - 1,
        2**60 + 1,
        2**60,
        2**53 + 1,
        2**53,
        2**60,
        3,
        2**64 - 1,
        2**32,
        2**60 + 1,
        2**63,
        2**63 - 1,
    ]
    signed_values = [
        2**63 - 1,
        -(2**63),
        2**53 + 1,
        -(2**53) - 1,
        2**53 + 1,
        0,
        -1,
        -(2**63),
        2**53,
        2**62,
        2**53 + 1,
        -1,
    ]
    count = len(unsigned_values)
    unsigned = _put(np.asarray(unsigned_values, np.uint64), device)
    signed = _put(np.asarray(signed_values, np.int64), device)
    identifiers = _put(
        np.asarray([5, 3, 9, 1, 2, 9, 0, 4, 8, 7, 6, 2], np.uint64), device
    )
    identifier_values = [5, 3, 9, 1, 2, 9, 0, 4, 8, 7, 6, 2]
    low_mask = np.uint64(0x5555_5555_5555_5555)

    @jax.jit
    def program(u: jax.Array, s: jax.Array, ids: jax.Array) -> tuple[jax.Array, ...]:
        order = lax.broadcasted_iota(np.int64, u.shape, 0)
        _, _, lexicographic = lax.sort((s, ids, order), num_keys=2, is_stable=True)
        return (
            jnp.argsort(u, stable=True),
            jnp.argsort(u, stable=True, descending=True),
            jnp.sort(u),
            jnp.argsort(s, stable=True),
            jnp.argsort(s, stable=True, descending=True),
            lexicographic,
            (u & low_mask) | ((u >> np.uint64(1)) & ~low_mask),
            u << np.uint64(7),
            s >> np.int64(62),
        )

    results = program(unsigned, signed, identifiers)
    indices = range(count)
    expected = (
        sorted(indices, key=lambda i: unsigned_values[i]),
        sorted(indices, key=lambda i: -unsigned_values[i]),
        sorted(unsigned_values),
        sorted(indices, key=lambda i: signed_values[i]),
        sorted(indices, key=lambda i: -signed_values[i]),
        sorted(indices, key=lambda i: (signed_values[i], identifier_values[i])),
        [
            (v & int(low_mask)) | ((v >> 1) & (2**64 - 1 - int(low_mask)))
            for v in unsigned_values
        ],
        [(v << 7) % 2**64 for v in unsigned_values],
        [v >> 62 for v in signed_values],
    )
    for result, reference in zip(results, expected, strict=True):
        assert [int(v) for v in np.asarray(result)] == list(reference)
    assert results[2].dtype == np.uint64 and results[6].dtype == np.uint64
    assert results[8].dtype == np.int64
    record_metal_execution(record_property, *results)


_THREEFRY = """
import jax
import jax.numpy as jnp

def threefry_program(seed):
    key = jax.random.key(seed)
    first, second = jax.random.split(key)
    folded = jax.random.fold_in(second, 7)
    return (
        jax.random.key_data(first),
        jax.random.key_data(folded),
        jax.random.bits(folded, (16,), jnp.uint32),
        jax.random.bits(first, (4,), jnp.uint64),
    )

def reference(payload):
    seed = jnp.asarray(payload["seed"], jnp.int64)
    impl = str(jax.random.key_impl(jax.random.key(seed)))
    return {
        "impl": impl,
        "values": [[int(v) for v in jax.numpy.ravel(part)] for part in jax.jit(threefry_program)(seed)],
    }
"""


def test_threefry_typed_keys_match_separate_cpu_reference(
    device: jax.Device, record_property: RecordProperty
) -> None:
    seed = 2**40 + 12345
    namespace: dict[str, object] = {}
    exec(_THREEFRY, namespace)
    program = namespace["threefry_program"]
    assert callable(program)
    assert config.snapshot().default_prng_implementation == "threefry2x32"
    results = jax.jit(program)(_put(np.asarray(seed, np.int64), device))
    reference = _cpu_reference(_THREEFRY, {"seed": seed})
    assert str(jax.random.key_impl(jax.random.key(seed))) == reference["impl"]
    assert [result.dtype for result in results] == [
        np.uint32,
        np.uint32,
        np.uint32,
        np.uint64,
    ]
    assert [[int(v) for v in np.ravel(result)] for result in results] == reference[
        "values"
    ]
    record_metal_execution(record_property, *results)


def _accumulate_guarded(values: jax.Array) -> jax.Array:
    """A guard inside a conditional branch inside a compiled loop."""

    def step(index: jax.Array, total: jax.Array) -> jax.Array:
        value = values[index]

        def checked(running: jax.Array) -> jax.Array:
            return eqx.error_if(
                running + value, value > 10.0, "metal guarded value exceeds bound"
            )

        return lax.cond(value >= 0, checked, lambda running: running - value, total)

    return lax.fori_loop(0, values.shape[0], step, jnp.zeros((), values.dtype))


@pytest.mark.parametrize("outcome", ["accepted", "refused"])
@pytest.mark.parametrize("transform", ["jit", "filter_jit"])
def test_compiled_guard_in_nested_control_flow(
    device: jax.Device, record_property: RecordProperty, transform: str, outcome: str
) -> None:
    compiled = (
        jax.jit(_accumulate_guarded)
        if transform == "jit"
        else eqx.filter_jit(_accumulate_guarded)
    )
    accepted = _put(np.asarray([1.0, -2.0, 3.0, 4.0], np.float32), device)
    if outcome == "refused":
        # filter_jit synchronizes and rethrows; jit surfaces the runtime error at sync.
        expected = (
            eqx.EquinoxRuntimeError
            if transform == "filter_jit"
            else jax.errors.JaxRuntimeError
        )
        with pytest.raises(expected, match="metal guarded value exceeds bound"):
            jax.block_until_ready(
                compiled(_put(np.asarray([1.0, 20.0, -2.0], np.float32), device))
            )
    total = jax.block_until_ready(compiled(accepted))
    assert total.dtype == np.float32
    assert np.asarray(total).item() == 10.0
    record_metal_execution(record_property, total)


def _gelu_reference(
    x: np.ndarray, *, approximate: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not approximate:
        cdf = np.asarray([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in x])
        pdf = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return x * cdf, cdf + x * pdf, pdf * (2.0 - x * x)
    c = math.sqrt(2.0 / math.pi)
    a = 0.044715
    t = np.tanh(c * (x + a * x**3))
    s = 1.0 - t * t
    du = c * (1.0 + 3.0 * a * x * x)
    ddu = 6.0 * c * a * x
    value = 0.5 * x * (1.0 + t)
    first = 0.5 * (1.0 + t) + 0.5 * x * s * du
    second = s * du + 0.5 * x * (s * ddu - 2.0 * t * s * du * du)
    return value, first, second


@pytest.mark.parametrize("approximate", [True, False], ids=["tanh", "erf"])
def test_ordinary_gelu_derivatives_match_analytic_reference(
    device: jax.Device, record_property: RecordProperty, approximate: bool
) -> None:
    gelu = jax.nn.gelu
    # The ordinary source-addressed callable, not a runtime replacement.
    assert gelu is _nn_functions.gelu
    assert (
        Path(gelu.__code__.co_filename).resolve()
        == Path(_nn_functions.__file__).resolve()
    )

    def function(y: jax.Array) -> jax.Array:
        return gelu(y, approximate=approximate)

    host = np.linspace(-4.0, 4.0, 33).astype(np.float32)
    x = _put(host, device)
    tangent = jnp.ones_like(x)

    @jax.jit
    def derivatives(y: jax.Array) -> tuple[jax.Array, ...]:
        value, forward = jax.jvp(function, (y,), (tangent,))
        forward_second = jax.jvp(
            lambda z: jax.jvp(function, (z,), (tangent,))[1], (y,), (tangent,)
        )[1]
        reverse = jax.vmap(jax.grad(function))(y)
        reverse_second = jax.vmap(jax.grad(jax.grad(function)))(y)
        return value, forward, forward_second, reverse, reverse_second

    value, forward, forward_second, reverse, reverse_second = derivatives(x)
    expected_value, expected_first, expected_second = _gelu_reference(
        host.astype(np.float64), approximate=approximate
    )
    # Elementary float32 functions are accurate to a few ulps; derivatives are O(1).
    tolerance = {"rtol": 1e-5, "atol": 1e-6}
    np.testing.assert_allclose(value, expected_value, **tolerance)
    for first in (forward, reverse):
        np.testing.assert_allclose(first, expected_first, **tolerance)
    for second in (forward_second, reverse_second):
        np.testing.assert_allclose(second, expected_second, **tolerance)
    results = (value, forward, forward_second, reverse, reverse_second)
    assert all(result.dtype == np.float32 for result in results)
    record_metal_execution(record_property, *results)


_ATTENTION = """
import jax
import jax.numpy as jnp

def attention_loss(query, key, value, weight):
    return jnp.sum(jax.nn.dot_product_attention(query, key, value) * weight)

def attention_derivatives(primals, tangents):
    loss_value, directional = jax.jvp(attention_loss, primals, tangents)
    gradient = jax.grad(attention_loss, argnums=(0, 1, 2))
    _, hessian_action = jax.jvp(
        lambda q, k, v: gradient(q, k, v, primals[3]), primals[:3], tangents[:3]
    )
    return loss_value, directional, hessian_action

def reference(payload):
    to = lambda name: jnp.asarray(payload[name], jnp.float64)
    primals = tuple(to(name) for name in ("query", "key", "value", "weight"))
    tangents = (*(to(name) for name in ("d_query", "d_key", "d_value")), jnp.zeros_like(primals[3]))
    loss_value, directional, hessian_action = jax.jit(attention_derivatives)(primals, tangents)
    return {
        "loss": float(loss_value),
        "directional": float(directional),
        "hessian_action": [jnp.ravel(part).tolist() for part in hessian_action],
    }
"""


def test_ordinary_attention_second_order_matches_separate_cpu_reference(
    device: jax.Device, record_property: RecordProperty
) -> None:
    attention = jax.nn.dot_product_attention
    assert attention is _nn_functions.dot_product_attention
    assert (
        Path(attention.__code__.co_filename).resolve()
        == Path(_nn_functions.__file__).resolve()
    )
    namespace: dict[str, object] = {}
    exec(_ATTENTION, namespace)
    derivatives = namespace["attention_derivatives"]
    assert callable(derivatives)
    rng = np.random.default_rng(11)
    shape = (1, 4, 2, 8)
    names = ("query", "key", "value", "weight", "d_query", "d_key", "d_value")
    host = {name: (0.5 * rng.standard_normal(shape)).astype(np.float32) for name in names}
    # Positive values keep the loss away from cancellation, so its relative error is meaningful.
    host["value"] = (0.5 + 0.25 * np.abs(rng.standard_normal(shape))).astype(np.float32)
    primals = tuple(_put(host[name], device) for name in names[:4])
    tangents = (
        *(_put(host[name], device) for name in names[4:]),
        jnp.zeros_like(primals[3]),
    )
    loss_value, directional, hessian_action = jax.jit(derivatives)(primals, tangents)
    reference = _cpu_reference(_ATTENTION, {name: host[name].tolist() for name in names})
    reference_action = reference["hessian_action"]
    assert isinstance(reference_action, list)

    # Same float32-rounded inputs; the float64 CPU result is the reference.
    # A 1e-4 normwise bound (floored at unit scale) leaves ~1000 float32 ulps for
    # this well-scaled softmax composition while rejecting reduced-mantissa
    # (TF32/BF16) execution.
    def close(actual: np.ndarray, expected: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(actual - expected)
            <= 1e-4 * max(float(np.linalg.norm(expected)), 1.0)
        )

    assert close(
        np.asarray(loss_value, np.float64), np.asarray(reference["loss"], np.float64)
    )
    assert close(
        np.asarray(directional, np.float64),
        np.asarray(reference["directional"], np.float64),
    )
    expected_action = np.concatenate(
        [np.asarray(part, np.float64) for part in reference_action]
    )
    actual_action = np.concatenate(
        [np.ravel(np.asarray(part, np.float64)) for part in hessian_action]
    )
    assert close(actual_action, expected_action)
    results = (loss_value, directional, *hessian_action)
    assert all(result.dtype == np.float32 for result in results)
    record_metal_execution(record_property, *results)


def test_dependent_dispatch_completes_with_exact_result(
    device: jax.Device, record_property: RecordProperty
) -> None:
    modulus = np.int64(2**61 - 1)
    step = jax.jit(lambda x: (x * np.int64(3) + np.int64(1)) % modulus)
    start = np.arange(1 << 20, dtype=np.int64) * np.int64(2**40 + 1) % modulus
    expected = start.copy()
    for _ in range(40):
        expected = (expected * 3 + 1) % modulus

    # Readback without an explicit wait must observe the completed chain.
    implicit = _put(start, device)
    for _ in range(40):
        implicit = step(implicit)
    np.testing.assert_array_equal(np.asarray(implicit), expected)

    explicit = _put(start, device)
    for _ in range(40):
        explicit = step(explicit)
    explicit.block_until_ready()
    assert explicit.is_ready()
    np.testing.assert_array_equal(np.asarray(explicit), expected)
    record_metal_execution(record_property, implicit, explicit)


def test_execution_error_propagates_to_dependent_results(
    device: jax.Device, record_property: RecordProperty
) -> None:
    guarded = jax.jit(
        lambda x: eqx.error_if(x, x[0] > 0, "metal dependent failure sentinel")
    )
    follow = jax.jit(lambda y: y * 2 + 1)
    # Wherever the failure surfaces, no value of the dependent chain is observable.
    with pytest.raises(
        jax.errors.JaxRuntimeError, match="metal dependent failure sentinel"
    ):
        failed = guarded(_put(np.asarray([1.0, 2.0], np.float32), device))
        dependent = follow(failed)
        np.asarray(dependent)
    recovered = follow(guarded(_put(np.asarray([-1.0, 2.0], np.float32), device)))
    np.testing.assert_array_equal(
        np.asarray(recovered), np.asarray([-1.0, 5.0], np.float32)
    )
    record_metal_execution(record_property, recovered)


@pytest.mark.parametrize("trip_count", [0, 6], ids=["zero-trip", "six-steps"])
def test_ordered_host_callbacks_preserve_effects_and_errors(
    device: jax.Device, record_property: RecordProperty, trip_count: int
) -> None:
    observed: list[tuple[int, float]] = []

    def record(index: np.ndarray, value: np.ndarray) -> np.ndarray:
        observed.append((int(index), float(value)))
        return np.asarray(value, np.float32)

    def shift(value: np.ndarray) -> np.ndarray:
        return np.asarray(value + np.float32(1.0), np.float32)

    scalar = jax.ShapeDtypeStruct((), np.float32)

    @jax.jit
    def program(start: jax.Array) -> jax.Array:
        def step(index: jax.Array, carry: jax.Array) -> jax.Array:
            value = carry * 0.5 + index.astype(np.float32)
            echoed = io_callback(record, scalar, index, value, ordered=True)
            return jax.pure_callback(shift, scalar, echoed)

        return lax.fori_loop(0, trip_count, step, start)

    first = jax.block_until_ready(program(_put(np.asarray(1.0, np.float32), device)))
    second = jax.block_until_ready(program(_put(np.asarray(-3.0, np.float32), device)))

    # Dyadic values: every float32 step below is exact.
    def expected(start: float) -> tuple[list[tuple[int, float]], float]:
        effects, carry = [], start
        for index in range(trip_count):
            value = carry * 0.5 + index
            effects.append((index, value))
            carry = value + 1.0
        return effects, carry

    first_effects, first_value = expected(1.0)
    second_effects, second_value = expected(-3.0)
    assert observed == first_effects + second_effects
    assert (
        np.asarray(first).item() == first_value
        and np.asarray(second).item() == second_value
    )

    def fail(value: np.ndarray) -> np.ndarray:
        raise ValueError("metal callback failure sentinel")

    failing = jax.jit(lambda x: jax.pure_callback(fail, scalar, x) * 2)
    with pytest.raises(
        jax.errors.JaxRuntimeError, match="metal callback failure sentinel"
    ):
        jax.block_until_ready(failing(_put(np.asarray(1.0, np.float32), device)))
    record_metal_execution(record_property, first, second)


def test_bfloat16_host_callback_preserves_dtype_and_strided_result(
    device: jax.Device, record_property: RecordProperty
) -> None:
    dtype = np.dtype(jnp.bfloat16)
    host = np.asarray([1.5, -2.0, 0.0, 4.0], dtype=dtype)

    def reverse(value: np.ndarray) -> np.ndarray:
        return value[::-1]

    signature = jax.ShapeDtypeStruct(host.shape, dtype)
    program = jax.jit(lambda value: jax.pure_callback(reverse, signature, value))
    result = program(_put(host, device))
    assert result.dtype == dtype
    np.testing.assert_array_equal(np.asarray(result), host[::-1])
    record_metal_execution(record_property, result)


def test_explicit_float32_literal_cast_preserves_rounding_and_ieee_classes(
    device: jax.Device, record_property: RecordProperty
) -> None:
    host = np.asarray(
        [
            0.0,
            -0.0,
            1.0 + 2.0**-40,
            1.0 + 2.0**-24,
            np.nextafter(1.0 + 2.0**-24, np.inf),
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float64,
    )

    @jax.jit
    def program(flag: jax.Array) -> tuple[jax.Array, jax.Array]:
        literal = jnp.asarray(host).astype(jnp.float32)
        selected = jnp.where(flag, -0.0, jnp.asarray(2.0, dtype=jnp.float32))
        return literal, selected

    result, selected = program(_put(np.asarray(True), device))
    expected = host.astype(np.float32)
    actual = np.asarray(result)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual.view(np.uint32)[valid], expected.view(np.uint32)[valid]
    )
    assert np.asarray(selected).view(np.uint32).item() == 0x80000000
    record_metal_execution(record_property, result, selected)
