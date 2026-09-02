#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax._training_objective import (
    _combine_objective_contributions,
    _GradientAccumulationState,
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


def test_gradient_accumulator_merges_scales_and_preserves_none_leaves():
    template = {
        "weight": jnp.zeros((2,), dtype=jnp.float32),
        "static": None,
    }
    accumulator = _GradientAccumulationState.empty(
        template,
        accumulation_dtype=jnp.float64,
    )
    accumulator = accumulator.add(
        {"weight": jnp.asarray((2.0, 4.0), dtype=jnp.float32), "static": None},
        _ObjectiveContribution(0.0, 2.0, 10.0),
    )
    accumulator = accumulator.add(
        {"weight": jnp.asarray((3.0, 6.0), dtype=jnp.float32), "static": None},
        _ObjectiveContribution(0.0, 1.0, 9.0),
    )

    gradient = accumulator.normalized_gradient(template)
    relative = jnp.exp(jnp.asarray(-1.0, dtype=jnp.float64))
    expected = (
        jnp.asarray((2.0, 4.0), dtype=jnp.float64)
        + relative * jnp.asarray((3.0, 6.0), dtype=jnp.float64)
    ) / (2.0 + relative)

    assert accumulator.gradient_numerator["weight"].dtype == jnp.float64
    assert gradient["weight"].dtype == jnp.float32
    assert gradient["static"] is None
    assert jnp.allclose(gradient["weight"], expected.astype(jnp.float32))


def test_gradient_accumulator_zero_support_cannot_leak_nonfinite_gradient():
    template = (jnp.zeros((), dtype=jnp.float32),)
    accumulator = _GradientAccumulationState.empty(
        template,
        accumulation_dtype=jnp.float32,
    )
    accumulator = accumulator.add(
        (jnp.asarray(jnp.nan, dtype=jnp.float32),),
        _ObjectiveContribution(0.0, 0.0, 100.0),
    )

    assert accumulator.microsteps == 1
    assert not bool(accumulator.has_positive_support)
    assert bool(jnp.isfinite(accumulator.gradient_numerator[0]))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="zero-support"):
        accumulator.normalized_gradient(template)
