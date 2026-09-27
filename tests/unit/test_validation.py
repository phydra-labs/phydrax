from __future__ import annotations

from collections.abc import Callable

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


def test_validation_scenario_1() -> None:
    assert canonical_identifier("reactor-7", "reactor_id") == "reactor-7"
    wrong_kinds: tuple[object, ...] = (None, 7, b"reactor", ("reactor",))
    for value in wrong_kinds:
        with pytest.raises(TypeError):
            canonical_identifier(value, "reactor_id")
    for value in ("", " reactor", "reactor ", "\treactor\n"):
        with pytest.raises(ValueError):
            canonical_identifier(value, "reactor_id")
    assert optional_identifier(None, "parent_id") is None
    with pytest.raises(TypeError):
        optional_identifier(3, "parent_id")

    assert unique_identifiers(("b", "a"), "labels", sort=True) == ("a", "b")
    with pytest.raises(TypeError):
        unique_identifiers(("a", 1), "labels")
    for values in (("a", "a"), ("a", " b")):
        with pytest.raises(ValueError):
            unique_identifiers(values, "labels")
    assert positive_finite_float(2, "scale") == 2.0
    assert positive_finite_float(np.float32(0.5), "scale") == 0.5
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            positive_finite_float(invalid, "scale")


def test_integer_validators_refuse_booleans_and_nonintegral_values() -> None:
    cases: tuple[tuple[Callable[[object, str], int], int], ...] = (
        (positive_integer, 1),
        (nonnegative_integer, 0),
    )
    for validate, smallest in cases:
        assert validate(np.int64(smallest), "count") == smallest
        assert type(validate(np.int64(smallest), "count")) is int
        for wrong_kind in (True, 1.0, "1"):
            with pytest.raises(TypeError):
                validate(wrong_kind, "count")
        with pytest.raises(ValueError):
            validate(smallest - 1, "count")
