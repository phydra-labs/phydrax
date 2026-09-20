#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp

from ._frequency import FrequencyAxis


def evaluate_frequency_sweep(axis: FrequencyAxis, evaluator, /):
    return jnp.stack(tuple(evaluator(value) for value in axis.frequency_hz))


__all__ = ["evaluate_frequency_sweep"]
