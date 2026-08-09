#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_polynomial_library_has_stable_order_and_jit_evaluation():
    layout = phx.dynamics.StateLayout((2,), component_names=("x", "y"))
    library = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2)

    assert library.feature_names == (
        "1",
        "state:x",
        "state:y",
        "state:x^2",
        "state:x * state:y",
        "state:y^2",
    )
    values = jax.jit(library)(jnp.asarray([[2.0, 3.0]]))
    np.testing.assert_allclose(
        np.asarray(values), np.asarray([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])
    )


def test_polynomial_guard_runs_before_large_feature_allocation():
    with pytest.raises(ValueError, match="max_features"):
        phx.dynamics.identification.PolynomialFeatureLibrary(
            phx.dynamics.StateLayout((12,)),
            degree=8,
            max_features=100,
        )


def test_composed_fourier_libraries_preserve_names_and_values():
    layout = phx.dynamics.StateLayout((1,), component_names=("angle",))
    sine = phx.dynamics.identification.FourierFeatureLibrary(
        layout,
        jnp.asarray([[1.0], [2.0]]),
        include_bias=False,
        include_cosine=False,
    )
    cosine = phx.dynamics.identification.FourierFeatureLibrary(
        layout,
        jnp.asarray([[1.0], [2.0]]),
        include_bias=False,
        include_sine=False,
    )
    library = phx.dynamics.identification.ConcatenatedFeatureLibrary((sine, cosine))

    value = library(jnp.asarray([jnp.pi / 2.0]))

    assert library.num_features == 4
    np.testing.assert_allclose(
        np.asarray(value), np.asarray([1.0, 0.0, 0.0, -1.0]), atol=1e-12
    )
