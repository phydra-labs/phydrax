import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._dtype_names import (
    category_dtype_rule,
    dtype_matches,
    DTypeRule,
    exact_dtype_rule,
)
from phydrax.precision import (
    complex_precision_dtype,
    precision_dtype_name,
    real_precision_dtype_name,
)


def test_precision_names_are_canonical_supported_dtypes():
    assert precision_dtype_name(np.float32) == "float32"
    assert precision_dtype_name("bfloat16") == "bfloat16"
    assert real_precision_dtype_name(jnp.float64) == "float64"
    assert complex_precision_dtype("float64") == "complex128"
    assert complex_precision_dtype("float16") == "complex64"
    with pytest.raises(ValueError):
        precision_dtype_name(np.int32)
    with pytest.raises(ValueError):
        real_precision_dtype_name(np.complex64)


def test_exact_rules_match_only_their_canonical_dtype():
    rule = exact_dtype_rule(np.float64)

    assert dtype_matches(rule, np.dtype(np.float64))
    assert not dtype_matches(rule, np.dtype(np.float32))


@pytest.mark.parametrize(
    ("category", "dtype", "matches"),
    [
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
    ],
)
def test_category_rules_follow_the_jax_dtype_hierarchy(category, dtype, matches):
    assert dtype_matches(category_dtype_rule(category), np.dtype(dtype)) is matches


def test_prng_key_category_accepts_only_typed_keys():
    rule = category_dtype_rule("prng_key")

    assert dtype_matches(rule, jax.random.key(0).dtype)
    assert not dtype_matches(rule, jax.random.PRNGKey(0).dtype)


def test_rules_name_exactly_one_dtype_or_category():
    with pytest.raises(ValueError):
        DTypeRule(None, None)
    with pytest.raises(ValueError):
        DTypeRule(np.dtype(np.float64), "floating")
