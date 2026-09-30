"""Proxy distinctions and dimension-independent FE exterior differentiation."""

from itertools import combinations

import numpy as np
import pytest

from phydrax.equations.fem._operators import (
    curl,
    divergence,
    normal_trace,
    tangential_trace,
)
from phydrax.exterior._form_type import FormType, FormValueSpec


def test_planar_flux_and_circulation_have_distinct_differentials() -> None:
    gradient = np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float64)
    circulation = FormValueSpec(FormType(2, 1, twist="untwisted"), proxy="circulation")
    flux = FormValueSpec(FormType(2, 1, twist="twisted"), proxy="flux")
    np.testing.assert_allclose(curl(gradient, value_spec=circulation), 1.0)
    np.testing.assert_allclose(divergence(gradient, value_spec=flux), 5.0)
    with pytest.raises(ValueError):
        divergence(gradient, value_spec=circulation)
    with pytest.raises(ValueError):
        curl(gradient, value_spec=flux)


@pytest.mark.parametrize("dimension", (2, 3, 4, 5))
def test_circulation_derivative_matches_antisymmetric_gradient(dimension: int) -> None:
    gradient = np.arange(dimension**2, dtype=np.float64).reshape((dimension, dimension))
    components = np.asarray(
        [
            gradient[second, first] - gradient[first, second]
            for first, second in combinations(range(dimension), 2)
        ],
        dtype=np.float64,
    )
    specification = FormValueSpec(
        FormType(dimension, 1, twist="untwisted"), proxy="circulation"
    )
    result = curl(gradient, value_spec=specification)
    if dimension == 2:
        expected = components[0]
    elif dimension == 3:
        expected = np.asarray(
            (components[2], -components[1], components[0]), dtype=np.float64
        )
    else:
        expected = components
    np.testing.assert_allclose(result, expected, atol=1e-14)


@pytest.mark.parametrize("dimension", (1, 2, 3, 4, 5))
def test_flux_divergence_and_normal_trace_use_physical_vector(dimension: int) -> None:
    gradient = np.arange(dimension**2, dtype=np.float64).reshape((dimension, dimension))
    specification = FormValueSpec(
        FormType(dimension, dimension - 1, twist="twisted"), proxy="flux"
    )
    np.testing.assert_allclose(
        divergence(gradient, value_spec=specification), np.trace(gradient), atol=1e-14
    )
    field = np.linspace(-0.7, 1.2, dimension, dtype=np.float64)
    normal = np.arange(1, dimension + 1, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(
        normal_trace(field, normal, value_spec=specification), field @ normal, atol=1e-14
    )


@pytest.mark.parametrize("dimension", (3, 4, 5))
def test_circulation_tangential_trace_is_orthogonal_projection(dimension: int) -> None:
    specification = FormValueSpec(
        FormType(dimension, 1, twist="untwisted"), proxy="circulation"
    )
    field = np.linspace(-0.7, 1.2, dimension, dtype=np.float64)
    normal = np.arange(1, dimension + 1, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    expected = field - (field @ normal) * normal
    actual = tangential_trace(field, normal, value_spec=specification)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(np.asarray(actual) @ normal, 0.0, atol=1e-14)
