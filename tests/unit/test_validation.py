import numpy as np
import pytest

from phydrax._validation import (
    canonical_identifier,
    nonnegative_integer,
    optional_identifier,
    positive_finite_float,
    positive_integer,
    unique_identifiers,
)


def test_canonical_identifier_accepts_a_canonical_string_unchanged():
    assert canonical_identifier("reactor-7", "reactor_id") == "reactor-7"


@pytest.mark.parametrize("value", [None, 7, b"reactor", ("reactor",)])
def test_canonical_identifier_rejects_non_strings_as_wrong_kind(value):
    with pytest.raises(TypeError):
        canonical_identifier(value, "reactor_id")


@pytest.mark.parametrize("value", ["", " reactor", "reactor ", "\treactor\n"])
def test_canonical_identifier_rejects_empty_or_padded_strings_as_invalid_values(value):
    with pytest.raises(ValueError):
        canonical_identifier(value, "reactor_id")


def test_optional_identifier_keeps_none_and_classifies_wrong_kinds():
    assert optional_identifier(None, "parent_id") is None
    with pytest.raises(TypeError):
        optional_identifier(3, "parent_id")


def test_unique_identifiers_classify_element_kind_and_value_failures():
    assert unique_identifiers(("b", "a"), "labels", sort=True) == ("a", "b")
    with pytest.raises(TypeError):
        unique_identifiers(("a", 1), "labels")
    with pytest.raises(ValueError):
        unique_identifiers(("a", "a"), "labels")
    with pytest.raises(ValueError):
        unique_identifiers(("a", " b"), "labels")


def test_positive_finite_float_converts_and_requires_a_finite_positive_value():
    assert positive_finite_float(2, "scale") == 2.0
    assert positive_finite_float(np.float32(0.5), "scale") == 0.5
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            positive_finite_float(invalid, "scale")


@pytest.mark.parametrize(
    ("validate", "smallest"), [(positive_integer, 1), (nonnegative_integer, 0)]
)
def test_integer_validators_refuse_booleans_and_non_integral_values(validate, smallest):
    assert validate(np.int64(smallest), "count") == smallest
    assert type(validate(np.int64(smallest), "count")) is int
    for wrong_kind in (True, 1.0, "1"):
        with pytest.raises(TypeError):
            validate(wrong_kind, "count")
    with pytest.raises(ValueError):
        validate(smallest - 1, "count")
