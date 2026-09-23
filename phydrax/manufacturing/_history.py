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


class ProcessHistory(StrictModule, NonTrainableState):
    times_s: Array
    deposited_mass_kg: Array
    supplied_energy_j: Array
    active_measure: Array
    history_id: str = eqx.field(static=True)

    def __init__(self, times_s, deposited_mass_kg, supplied_energy_j, active_measure, /):
        t = np.asarray(times_s, float)
        m = np.asarray(deposited_mass_kg, float)
        e = np.asarray(supplied_energy_j, float)
        a = np.asarray(active_measure, float)
        if (
            t.ndim != 1
            or any(x.shape != t.shape for x in (m, e, a))
            or t.size == 0
            or not all(np.all(np.isfinite(x)) for x in (t, m, e, a))
            or np.any(np.diff(t) <= 0)
            or np.any(np.diff(m) < 0)
            or np.any(np.diff(e) < 0)
            or m[0] < 0
            or e[0] < 0
            or np.any(a < 0)
        ):
            raise ValueError("Process history is inadmissible.")
        self.times_s = jnp.asarray(t)
        self.deposited_mass_kg = jnp.asarray(m)
        self.supplied_energy_j = jnp.asarray(e)
        self.active_measure = jnp.asarray(a)
        self.history_id = canonical_fingerprint(
            {
                "kind": "process-history",
                "times": t.tolist(),
                "mass": m.tolist(),
                "energy": e.tolist(),
                "active": a.tolist(),
            }
        )


__all__ = ["ProcessHistory"]
