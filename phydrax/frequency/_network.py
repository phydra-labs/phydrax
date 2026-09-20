#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


@dataclass(frozen=True, slots=True)
class Port:
    port_id: str
    reference_impedance_ohm: complex
    orientation: tuple[float, ...]
    frame_id: str

    def __post_init__(self):
        if (
            not self.port_id
            or not self.frame_id
            or not self.orientation
            or self.reference_impedance_ohm.real <= 0
        ):
            raise ValueError("Port is invalid.")


class ScatteringMatrix(StrictModule, NonTrainableState):
    values: Array
    reference_impedances_ohm: Array
    network_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, reference_impedances_ohm: ArrayLike, /):
        s = jnp.asarray(values)
        z = jnp.asarray(reference_impedances_ohm)
        if s.ndim != 3 or s.shape[-1] != s.shape[-2] or z.shape != (s.shape[-1],):
            raise ValueError("Scattering data require (frequency,port,port).")
        self.values = s
        self.reference_impedances_ohm = z
        self.network_id = canonical_fingerprint(
            {
                "kind": "scattering-matrix",
                "shape": s.shape,
                "impedances": list(map(complex, z.tolist())),
            }
        )

    @property
    def reciprocal_error(self):
        return jnp.max(jnp.abs(self.values - jnp.swapaxes(self.values, -1, -2)))

    @property
    def maximum_power_gain(self):
        return jnp.max(jnp.linalg.svd(self.values, compute_uv=False) ** 2)

    @property
    def passive(self):
        return self.maximum_power_gain <= 1 + 1e-10


__all__ = ["Port", "ScatteringMatrix"]
