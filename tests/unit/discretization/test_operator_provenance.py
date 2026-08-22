#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.nn.operator import FunctionSamples


def test_operator_samples_separate_support_and_measure_identities():
    coordinates = jnp.asarray([[0.0], [0.5], [1.0]])
    weights = jnp.asarray([0.25, 0.5, 0.25])

    first = FunctionSamples(
        values=jnp.asarray([[1.0], [2.0], [3.0]]),
        coordinates=coordinates,
        quadrature_weights=weights,
    )
    same = FunctionSamples(
        values=jnp.asarray([[4.0], [5.0], [6.0]]),
        coordinates=coordinates,
        quadrature_weights=weights,
    )
    changed_measure = FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=jnp.asarray([0.2, 0.6, 0.2]),
    )
    changed_support = FunctionSamples(
        values=None,
        coordinates=jnp.asarray([[0.0], [0.4], [1.0]]),
        quadrature_weights=weights,
    )

    assert first.support_id == same.support_id == changed_measure.support_id
    assert first.measure_id == same.measure_id
    assert changed_measure.measure_id != first.measure_id
    assert changed_support.support_id != first.support_id


def test_operator_measure_identity_requires_weights():
    with pytest.raises(ValueError, match="requires quadrature_weights"):
        FunctionSamples(
            values=None,
            coordinates=jnp.asarray([[0.0], [1.0]]),
            measure_id="invalid-measure",
        )
