#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._conservation_boundary import AbstractConservationBoundary


class FiniteVolumeBoundaryPair(StrictModule, NonTrainableState):
    """Lower and upper physical boundaries for one bounded axis."""

    lower: AbstractConservationBoundary
    upper: AbstractConservationBoundary
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: AbstractConservationBoundary,
        upper: AbstractConservationBoundary,
        /,
    ):
        if not isinstance(lower, AbstractConservationBoundary) or not isinstance(
            upper, AbstractConservationBoundary
        ):
            raise TypeError("Boundary pairs require finite-volume boundary policies.")
        self.lower = lower
        self.upper = upper
        self.pair_id = canonical_fingerprint(
            {
                "kind": "fv-boundary-pair",
                "lower": lower.boundary_id,
                "upper": upper.boundary_id,
            }
        )


class FiniteVolumeBoundarySet(StrictModule, NonTrainableState):
    """Axis-ordered bounded policies; periodic axes use ``None``."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    pairs: tuple[FiniteVolumeBoundaryPair | None, ...]
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        pairs: Sequence[FiniteVolumeBoundaryPair | None],
        /,
    ):
        names = tuple(str(name) for name in axis_names)
        pairs_ = tuple(pairs)
        if (
            not names
            or len(names) != len(pairs_)
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Boundary axes and pairs must align with unique names.")
        if any(
            pair is not None and not isinstance(pair, FiniteVolumeBoundaryPair)
            for pair in pairs_
        ):
            raise TypeError("Boundary entries must be FiniteVolumeBoundaryPair or None.")
        self.axis_names = names
        self.pairs = pairs_
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "fv-boundary-set",
                "axes": list(names),
                "pairs": [None if pair is None else pair.pair_id for pair in pairs_],
            }
        )

    @classmethod
    def periodic(cls, axis_names: Sequence[str], /) -> "FiniteVolumeBoundarySet":
        names = tuple(axis_names)
        return cls(names, (None,) * len(names))


__all__ = [
    "FiniteVolumeBoundaryPair",
    "FiniteVolumeBoundarySet",
]
