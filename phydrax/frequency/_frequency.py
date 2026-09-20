#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class FrequencyAxis(StrictModule, NonTrainableState):
    frequency_hz: Array
    axis_id: str = eqx.field(static=True)

    def __init__(self, frequency_hz: ArrayLike, /):
        f = np.asarray(frequency_hz, float)
        if (
            f.ndim != 1
            or f.size == 0
            or np.any(f < 0)
            or np.any(np.diff(f) <= 0)
            or not np.all(np.isfinite(f))
        ):
            raise ValueError("Frequencies must be finite, non-negative, and increasing.")
        self.frequency_hz = jnp.asarray(f)
        self.axis_id = canonical_fingerprint(
            {"kind": "frequency-axis", "frequency_hz": f.tolist()}
        )

    @property
    def angular_frequency_rad_s(self):
        return 2 * jnp.pi * self.frequency_hz


__all__ = ["FrequencyAxis"]
