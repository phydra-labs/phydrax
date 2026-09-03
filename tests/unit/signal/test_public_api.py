#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax._trainable import partition_trainable


def test_signal_namespace_is_public_and_old_resampling_export_is_removed():
    assert "signal" in phx.__all__
    assert "fourier_resample" in phx.signal.__all__
    assert "DiscreteWaveletTransform" in phx.signal.__all__
    assert "spectral_resample" not in phx.nn.operator.architectures.__all__


def test_signal_plans_are_fixed_but_carried_history_remains_differentiable():
    fir_plan = phx.signal.FIRFilterPlan(3)
    fir_parameters, _ = partition_trainable(fir_plan)
    fir_state = fir_plan.initial_state((8,), dtype=jnp.float64)

    resampling_plan = phx.signal.RationalResamplingPlan(3, 2, 7, 4)
    resampling_parameters, _ = partition_trainable(resampling_plan)
    resampling_state = resampling_plan.initial_state((4,), dtype=jnp.float64)

    assert not jax.tree.leaves(fir_parameters)
    assert not jax.tree.leaves(resampling_parameters)
    assert fir_state.history.dtype == jnp.float64
    assert resampling_state.history.dtype == jnp.float64
