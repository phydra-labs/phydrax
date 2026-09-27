import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from phydrax._dtype_names import (
    category_dtype_rule,
    dtype_matches,
    DTypeCategory,
    DTypeRule,
    exact_dtype_rule,
)
from phydrax.precision import (
    complex_precision_dtype,
    precision_dtype_name,
    real_precision_dtype_name,
)


def test_dtype_names_scenario_1() -> None:
    assert precision_dtype_name(np.float32) == "float32"
    assert precision_dtype_name("bfloat16") == "bfloat16"
    assert real_precision_dtype_name(jnp.float64) == "float64"
    assert complex_precision_dtype("float64") == "complex128"
    assert complex_precision_dtype("float16") == "complex64"
    with pytest.raises(ValueError):
        precision_dtype_name(np.int32)
    with pytest.raises(ValueError):
        real_precision_dtype_name(np.complex64)
    rule = exact_dtype_rule(np.float64)

    assert dtype_matches(rule, np.dtype(np.float64))
    assert not dtype_matches(rule, np.dtype(np.float32))
    cases: tuple[tuple[DTypeCategory, npt.DTypeLike, bool], ...] = (
        ("boolean", np.bool_, True),
        ("boolean", np.int8, False),
        ("integer", np.uint16, True),
        ("integer", np.bool_, False),
        ("floating", jnp.bfloat16, True),
        ("floating", np.complex64, False),
        ("complex", np.complex128, True),
        ("inexact", np.float16, True),
        ("inexact", np.int64, False),
        ("numeric", np.int32, True),
        ("numeric", np.bool_, False),
    )
    for category, dtype, matches in cases:
        assert dtype_matches(category_dtype_rule(category), np.dtype(dtype)) is matches


def test_dtype_names_scenario_2() -> None:
    rule = category_dtype_rule("prng_key")

    assert dtype_matches(rule, jax.random.key(0).dtype)
    assert not dtype_matches(rule, jax.random.PRNGKey(0).dtype)
    with pytest.raises(ValueError):
        DTypeRule(None, None)
    with pytest.raises(ValueError):
        DTypeRule(np.dtype(np.float64), "floating")
