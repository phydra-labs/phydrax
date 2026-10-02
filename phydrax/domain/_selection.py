#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..typing import parse


CoordinateFaceSide: TypeAlias = Literal["lower", "upper"]


class Selection(StrictModule):
    """Semantic stratum or slice selection for a factor coordinate."""

    __strict_abstract__ = True


class Interior(Selection):
    """Select the full-dimensional interior support."""

    def __init__(self) -> None:
        pass


class Boundary(Selection):
    """Select a full boundary or certified source-entity subset."""

    tags: tuple[str, ...] | None = eqx.field(static=True)
    entity_ids: tuple[int, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        tags: Sequence[str] | None = None,
        entity_ids: Sequence[int] | None = None,
    ) -> None:
        tags_ = None if tags is None else tuple(str(tag) for tag in tags)
        entity_ids_ = None if entity_ids is None else tuple(map(int, entity_ids))
        if tags_ is not None and (not tags_ or any(not tag for tag in tags_)):
            raise ValueError("Boundary.tags must contain non-empty names.")
        if entity_ids_ is not None and not entity_ids_:
            raise ValueError("Boundary.entity_ids must be non-empty.")
        self.tags = tags_
        self.entity_ids = entity_ids_


class CoordinateFace(Selection):
    """Select one Cartesian coordinate face ``x[axis] = lower`` or ``x[axis] = upper``.

    The face is identified by the selected label's coordinate component ``axis``
    (the component of that label's `ValuePort`) and the endpoint ``side``. It is an
    exact stratum of an axis-aligned interval or box: one-dimensional faces carry a
    unit counting measure and higher-dimensional faces their exact transverse
    Hausdorff measure. Scalar intervals select their endpoints with
    `FixedStart()`/`FixedEnd()` instead.
    """

    axis: int = eqx.field(static=True)
    side: CoordinateFaceSide = eqx.field(static=True)

    def __init__(self, axis: int, side: CoordinateFaceSide, /) -> None:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(
                "CoordinateFace.axis must be an integer coordinate component."
            )
        if axis < 0:
            raise ValueError("CoordinateFace.axis must be nonnegative.")
        side_ = parse(side, CoordinateFaceSide, "side")
        self.axis = axis
        self.side = side_


class Fixed(Selection):
    """Select a unit-mass Dirac slice at an explicit coordinate value."""

    value: Array

    def __init__(self, value: ArrayLike, /) -> None:
        self.value = jnp.asarray(value, dtype=jnp.float64)


class FixedStart(Selection):
    """Select a factor-defined start endpoint or row-specific initial state."""

    def __init__(self) -> None:
        pass


class FixedEnd(Selection):
    """Select a factor-defined end endpoint or row-specific terminal state."""

    def __init__(self) -> None:
        pass


class SelectionSpec(StrictModule):
    """Immutable public-label-to-selection mapping with interior defaults."""

    by_label: frozendict[str, Selection]

    def __init__(self, by_label: Mapping[str, Selection] | None = None, /) -> None:
        resolved = {} if by_label is None else dict(by_label)
        for label, selection in resolved.items():
            if not isinstance(label, str) or not label:
                raise ValueError("Selection labels must be non-empty strings.")
            if not isinstance(selection, Selection):
                raise TypeError(
                    f"Selection for {label!r} must be a Selection, got {type(selection).__name__}."
                )
        self.by_label = frozendict(resolved)

    def selection_for(self, label: str, /) -> Selection:
        return self.by_label.get(label, Interior())


__all__ = [
    "Boundary",
    "CoordinateFace",
    "CoordinateFaceSide",
    "Fixed",
    "FixedEnd",
    "FixedStart",
    "Interior",
    "Selection",
    "SelectionSpec",
]
