#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax._training_objective import (
    _combine_objective_contributions,
    _ObjectiveAccumulator,
    _ObjectiveContribution,
)


def test_scaled_objective_contributions_merge_without_absolute_exponentiation():
    left = _ObjectiveContribution(2.0, 1.0, 1000.0)
    right = _ObjectiveContribution(12.0, 3.0, 999.0)

    combined = _combine_objective_contributions((left, right))
    relative = jnp.exp(jnp.asarray(-1.0))
    expected = (2.0 + relative * 12.0) / (1.0 + relative * 3.0)

    assert float(combined.log_scale) == pytest.approx(1000.0)
    assert float(combined.value) == pytest.approx(float(expected))


def test_objective_accumulator_ignores_zero_support_when_selecting_scale():
    accumulator = _ObjectiveAccumulator()
    accumulator = accumulator.add(_ObjectiveContribution(jnp.nan, 0.0, 1.0e4))
    accumulator = accumulator.add(_ObjectiveContribution(6.0, 2.0, -1.0e4))

    assert float(accumulator.value) == pytest.approx(3.0)
    assert float(accumulator.contribution.log_scale) == pytest.approx(-1.0e4)
