# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..parameters import OrderedOrdinalCutpoints


class OrdinalCumulativeLinkHead(AbstractArrayModel):
    """Wrap a scalar location model with globally ordered learned cutpoints."""

    location_model: AbstractArrayModel
    cutpoints: OrderedOrdinalCutpoints
    _in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)

    def __init__(
        self,
        location_model: AbstractArrayModel,
        cutpoints: OrderedOrdinalCutpoints,
        /,
    ):
        if not isinstance(location_model, AbstractArrayModel):
            raise TypeError("location_model must be an AbstractArrayModel.")
        if not isinstance(cutpoints, OrderedOrdinalCutpoints):
            raise TypeError("cutpoints must be OrderedOrdinalCutpoints.")
        output_size = location_model.out_size
        if output_size not in (1, "scalar", ()):
            raise ValueError(
                "OrdinalCumulativeLinkHead requires a scalar location model."
            )
        self.location_model = location_model
        self.cutpoints = cutpoints
        self._in_size = location_model.in_size
        self._out_size = cutpoints.class_count - 1

    @property
    def in_size(self) -> int | tuple[int, ...] | Literal["scalar"]:
        return self._in_size

    @property
    def out_size(self) -> int:
        return self._out_size

    @property
    def class_count(self) -> int:
        return self.cutpoints.class_count

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        location = jnp.asarray(self.location_model(x, key=key))
        if location.ndim >= 1 and int(location.shape[-1]) == 1:
            location = location[..., 0]
        return self.cutpoints() - location[..., None]


__all__ = ["OrdinalCumulativeLinkHead"]
