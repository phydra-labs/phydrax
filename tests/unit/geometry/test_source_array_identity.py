import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._source_array_identity import source_arrays_equal


@pytest.mark.parametrize(
    ("integer", "floating", "sign", "nan"),
    (
        (np.uint32, np.float32, 0x80000000, 0x7FC00000),
        (np.uint64, np.float64, 0x8000000000000000, 0x7FF8000000000000),
    ),
)
def test_float_identity_distinguishes_zero_sign_and_nan_payload(
    integer: type[np.uint32] | type[np.uint64],
    floating: type[np.float32] | type[np.float64],
    sign: int,
    nan: int,
) -> None:
    original = np.asarray((0, nan), dtype=integer).view(floating)
    changed_zero = np.asarray((sign, nan), dtype=integer).view(floating)
    changed_nan = np.asarray((0, nan + 1), dtype=integer).view(floating)
    equal = jax.jit(source_arrays_equal)
    assert bool(equal(original, original))
    assert not bool(equal(original, changed_zero))
    assert not bool(equal(original, changed_nan))


@pytest.mark.parametrize(
    ("integer", "complex_type", "sign", "nan"),
    (
        (np.uint32, np.complex64, 0x80000000, 0x7FC00000),
        (np.uint64, np.complex128, 0x8000000000000000, 0x7FF8000000000000),
    ),
)
def test_complex_identity_checks_each_component_bits(
    integer: type[np.uint32] | type[np.uint64],
    complex_type: type[np.complex64] | type[np.complex128],
    sign: int,
    nan: int,
) -> None:
    original = np.asarray((0, 0, 0, nan), dtype=integer).view(complex_type)
    changed_zero = np.asarray((0, sign, 0, nan), dtype=integer).view(complex_type)
    changed_nan = np.asarray((0, 0, 0, nan + 1), dtype=integer).view(complex_type)
    equal = jax.jit(source_arrays_equal)
    assert bool(equal(original, original))
    assert not bool(equal(original, changed_zero))
    assert not bool(equal(original, changed_nan))


def test_integer_boolean_and_array_layout_identity() -> None:
    assert bool(source_arrays_equal(jnp.asarray((0, 1)), jnp.asarray((0, 1))))
    assert not bool(
        source_arrays_equal(jnp.asarray((False, True)), jnp.asarray((False, False)))
    )
    with pytest.raises(ValueError, match="same shape and dtype"):
        source_arrays_equal(jnp.asarray((1,)), jnp.asarray((1, 2)))


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("tangent_value", (0, 1))
def test_source_identity_rejects_all_inexact_source_jvps(
    dtype: type[np.float32] | type[np.float64] | type[np.complex64] | type[np.complex128],
    tangent_value: int,
) -> None:
    source = jnp.asarray((1, 2), dtype=dtype)
    tangent = jnp.full_like(source, tangent_value)
    for comparison in (
        lambda value: source_arrays_equal(value, source),
        lambda value: source_arrays_equal(source, value),
        jax.jit(lambda value: source_arrays_equal(value, source)),
    ):
        with pytest.raises(ValueError, match="Fixed source geometry differentiation"):
            jax.jvp(comparison, (source,), (tangent,))


def test_state_jvp_with_fixed_source_remains_supported() -> None:
    source = jnp.asarray((1.0, 2.0))

    def response(state: jax.Array) -> jax.Array:
        return jnp.where(source_arrays_equal(source, source), state * state, jnp.nan)

    primal, tangent = jax.jvp(jax.jit(response), (jnp.asarray(2.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(primal, 4.0)
    np.testing.assert_allclose(tangent, 4.0)
