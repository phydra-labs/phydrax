#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class PassivityEvidence:
    maximum_gain: Array
    passive: Array


def check_scattering_passivity(scattering: ArrayLike, /) -> PassivityEvidence:
    gain = jnp.max(jnp.linalg.svd(jnp.asarray(scattering), compute_uv=False) ** 2)
    return PassivityEvidence(gain, gain <= 1 + 1e-10)


__all__ = ["PassivityEvidence", "check_scattering_passivity"]
