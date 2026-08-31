#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ..discretization.pic._current import PICMaxwellCurrentArguments
from ._maxwell_sources import (
    AbstractMaxwellSourcePlan,
    MaxwellSourceForcing,
)


class PreparedPICMaxwellCurrentSource(StrictModule, NonTrainableState):
    electric_count: int = eqx.field(static=True)
    magnetic_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    magnetic_closedness_preserving: bool = eqx.field(static=True)

    def sample(self, time, args=None, /) -> MaxwellSourceForcing:
        del time
        if not isinstance(args, PICMaxwellCurrentArguments):
            raise TypeError("PIC Maxwell source requires PICMaxwellCurrentArguments.")
        current = jnp.asarray(args.particle_current)
        if current.shape != (self.electric_count,):
            raise ValueError("PIC current must match retained electric cochains.")
        return MaxwellSourceForcing(
            current,
            jnp.zeros((self.magnetic_count,), dtype=current.dtype),
        )


class PICMaxwellCurrentSourcePlan(AbstractMaxwellSourcePlan, NonTrainableState):
    """Dynamic full-cochain current supplied by a PIC step argument."""

    source_id: str = eqx.field(static=True)

    def __init__(self, source_id: str = "pic-midpoint-current", /):
        identifier = str(source_id)
        if not identifier:
            raise ValueError("source_id must be nonempty.")
        self.source_id = identifier

    def prepare(self, bridge: StructuredCochainBridge, layout, /):
        return PreparedPICMaxwellCurrentSource(
            layout.electric_count,
            layout.magnetic_count,
            canonical_fingerprint(
                {
                    "kind": "prepared-pic-maxwell-current-source",
                    "source": self.source_id,
                    "bridge": bridge.bridge_id,
                    "layout": layout.layout_id,
                }
            ),
            True,
        )


__all__ = ["PICMaxwellCurrentSourcePlan", "PreparedPICMaxwellCurrentSource"]
