#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class MaterialHistory(StrictModule, NonTrainableState):
    times_s: Array
    temperatures_k: Array
    phase_fractions: Array
    history_id: str = eqx.field(static=True)

    def __init__(self, times_s, temperatures_k, phase_fractions, /):
        t = np.asarray(times_s, float)
        T = np.asarray(temperatures_k, float)
        p = np.asarray(phase_fractions, float)
        if (
            t.ndim != 1
            or t.size < 1
            or np.any(np.diff(t) <= 0)
            or T.shape[0] != t.size
            or p.shape[0] != t.size
            or np.any(T <= 0)
            or np.any(p < 0)
            or not np.allclose(p.sum(axis=-1), 1)
        ):
            raise ValueError("Material history is inadmissible.")
        self.times_s = jnp.asarray(t)
        self.temperatures_k = jnp.asarray(T)
        self.phase_fractions = jnp.asarray(p)
        self.history_id = canonical_fingerprint(
            {
                "kind": "material-history",
                "times": t.tolist(),
                "temperatures": T.tolist(),
                "phases": p.tolist(),
            }
        )


__all__ = ["MaterialHistory"]
