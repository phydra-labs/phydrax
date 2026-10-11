from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.materials import PropertyValidityRange


def test_correlation_validity_requires_all_physical_coordinates() -> None:
    validity = PropertyValidityRange(
        {"temperature": (300.0, 1000.0, "K"), "pressure": (1.0e5, 2.0e7, "Pa")},
        source_id="synthetic-reference:two-dimensional-support",
    )
    with pytest.raises(ValueError, match="exactly the declared variables"):
        validity.contains(temperature=500.0)
    with pytest.raises(ValueError, match="exactly the declared variables"):
        validity.contains(temperature=500.0, pressure=2e6, burnup=0.0)


def test_closed_ranges_include_endpoints_and_refuse_nonfinite_values() -> None:
    validity = PropertyValidityRange(
        {"temperature": (300.0, 1000.0, "K")},
        source_id="synthetic-reference:closed-temperature-support",
    )
    contains = jax.jit(lambda value: validity.contains(temperature=value))
    values = jnp.asarray([300.0, 1000.0, 299.0, 1001.0, np.nan, np.inf])
    np.testing.assert_array_equal(
        contains(values), [True, True, False, False, False, False]
    )


def test_unbounded_support_still_requires_finite_physical_values() -> None:
    validity = PropertyValidityRange(
        {"burnup": (0.0, math.inf, "MWd/kgHM")},
        source_id="synthetic-reference:unbounded-mathematical-support",
    )
    np.testing.assert_array_equal(
        validity.contains(burnup=jnp.asarray([0.0, 1e6, -1.0, np.inf])),
        [True, True, False, False],
    )
    assert validity.bound("burnup") == (0.0, math.inf, "MWd/kgHM")


@pytest.mark.parametrize(
    "bounds",
    [(400.0, 300.0), (math.inf, math.inf), (-math.inf, -math.inf), (math.nan, 300.0)],
    ids=["reversed", "positive-infinity-only", "negative-infinity-only", "nan-limit"],
)
def test_invalid_correlation_support_is_refused(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="ordered limits"):
        PropertyValidityRange(
            {"temperature": (bounds[0], bounds[1], "K")}, source_id="synthetic-reference"
        )
