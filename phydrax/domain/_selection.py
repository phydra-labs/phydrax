#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule


class Selection(StrictModule):
    """Semantic stratum or slice selection for a factor coordinate."""

    __strict_abstract__ = True


class Interior(Selection):
    """Select the full-dimensional interior support."""

    def __init__(self):
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
    ):
        tags_ = None if tags is None else tuple(str(tag) for tag in tags)
        entity_ids_ = None if entity_ids is None else tuple(map(int, entity_ids))
        if tags_ is not None and (not tags_ or any(not tag for tag in tags_)):
            raise ValueError("Boundary.tags must contain non-empty names.")
        if entity_ids_ is not None and not entity_ids_:
            raise ValueError("Boundary.entity_ids must be non-empty.")
        self.tags = tags_
        self.entity_ids = entity_ids_


class Fixed(Selection):
    """Select a unit-mass Dirac slice at an explicit coordinate value."""

    value: Array

    def __init__(self, value: ArrayLike, /):
        self.value = jnp.asarray(value, dtype=float)


class FixedStart(Selection):
    """Select a factor-defined start endpoint or row-specific initial state."""

    def __init__(self):
        pass


class FixedEnd(Selection):
    """Select a factor-defined end endpoint or row-specific terminal state."""

    def __init__(self):
        pass


class SelectionSpec(StrictModule):
    """Immutable public-label-to-selection mapping with interior defaults."""

    by_label: frozendict[str, Selection]

    def __init__(self, by_label: Mapping[str, Selection] | None = None, /):
        resolved = {} if by_label is None else dict(by_label)
        for label, selection in resolved.items():
            if not isinstance(label, str) or not label:
                raise ValueError("Selection labels must be non-empty strings.")
            if not isinstance(selection, Selection):
                raise TypeError(
                    f"Selection for {label!r} must be a Selection, got "
                    f"{type(selection).__name__}."
                )
        self.by_label = frozendict(resolved)

    def selection_for(self, label: str, /) -> Selection:
        return self.by_label.get(label, Interior())


__all__ = [
    "Boundary",
    "Fixed",
    "FixedEnd",
    "FixedStart",
    "Interior",
    "Selection",
    "SelectionSpec",
]
