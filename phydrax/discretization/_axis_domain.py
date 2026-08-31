#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


AxisDomainKind = Literal["bounded", "periodic", "half_line", "real_line"]
HalfLineDirection = Literal["positive", "negative"]


class AxisDomain(StrictModule, NonTrainableState):
    """Physical support of one numerical coordinate axis."""

    lower: Array | None
    upper: Array | None
    kind: AxisDomainKind = eqx.field(static=True)
    direction: HalfLineDirection | None = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: AxisDomainKind,
        /,
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        direction: HalfLineDirection | None = None,
    ):
        if kind not in ("bounded", "periodic", "half_line", "real_line"):
            raise ValueError("Unknown axis domain kind.")
        lower_ = None if lower is None else jnp.asarray(lower, dtype=float).reshape(())
        upper_ = None if upper is None else jnp.asarray(upper, dtype=float).reshape(())
        direction_: HalfLineDirection | None = direction
        if kind in ("bounded", "periodic"):
            if lower_ is None or upper_ is None or direction_ is not None:
                raise ValueError("Bounded and periodic domains require two endpoints.")
            endpoints = jnp.stack((lower_, upper_))
            endpoints = eqx.error_if(
                endpoints,
                ~(jnp.all(jnp.isfinite(endpoints)) & (endpoints[1] > endpoints[0])),
                "Axis domain endpoints must be finite and increasing.",
            )
            lower_, upper_ = endpoints[0], endpoints[1]
        elif kind == "half_line":
            if direction_ not in ("positive", "negative"):
                raise ValueError("Half-line domains require a direction.")
            endpoint = lower_ if direction_ == "positive" else upper_
            absent = upper_ if direction_ == "positive" else lower_
            if endpoint is None or absent is not None:
                raise ValueError("Half-line domains require exactly one finite endpoint.")
            endpoint = eqx.error_if(
                endpoint,
                ~jnp.isfinite(endpoint),
                "Half-line endpoint must be finite.",
            )
            if direction_ == "positive":
                lower_ = endpoint
            else:
                upper_ = endpoint
        elif lower_ is not None or upper_ is not None or direction_ is not None:
            raise ValueError("Real-line domains do not accept endpoints or direction.")

        payload = {
            "kind": "axis-domain",
            "domain_kind": kind,
            "direction": direction_,
            "lower": None if lower_ is None else array_tree_fingerprint(lower_),
            "upper": None if upper_ is None else array_tree_fingerprint(upper_),
        }
        self.lower = lower_
        self.upper = upper_
        self.kind = kind
        self.direction = direction_
        self.domain_id = canonical_fingerprint(payload)

    @classmethod
    def interval(cls, lower: ArrayLike, upper: ArrayLike, /) -> "AxisDomain":
        return cls("bounded", lower=lower, upper=upper)

    @classmethod
    def periodic(cls, lower: ArrayLike, upper: ArrayLike, /) -> "AxisDomain":
        return cls("periodic", lower=lower, upper=upper)

    @classmethod
    def half_line(
        cls,
        endpoint: ArrayLike,
        /,
        *,
        direction: HalfLineDirection = "positive",
    ) -> "AxisDomain":
        if direction == "positive":
            return cls("half_line", lower=endpoint, direction=direction)
        return cls("half_line", upper=endpoint, direction=direction)

    @classmethod
    def real_line(cls) -> "AxisDomain":
        return cls("real_line")

    @property
    def periodic_axis(self) -> bool:
        return self.kind == "periodic"

    @property
    def finite_bounds(self) -> Array | None:
        if self.kind not in ("bounded", "periodic"):
            return None
        return jnp.stack((self.lower, self.upper))

    @property
    def length(self) -> Array:
        if self.kind not in ("bounded", "periodic"):
            raise ValueError("Unbounded axis domains do not have a finite length.")
        return self.upper - self.lower


__all__ = ["AxisDomain", "AxisDomainKind", "HalfLineDirection"]
