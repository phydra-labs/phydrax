#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared result and protocol for complete physical cell location."""

from __future__ import annotations

import abc
from enum import IntEnum
from typing import Protocol

import equinox as eqx
from jax import Array

from .._strict import StrictModule
from ..typing import ConvertibleToArray, Identifier


class CellLocationStatus(IntEnum):
    LOCATED = 0
    OUTSIDE = 1
    DEGENERATE_CELL = 2
    NONFINITE = 3
    RESOURCE_EXCEEDED = 4
    INVERSE_MAP_EXHAUSTED = 5
    CANDIDATE_SEARCH_INCOMPLETE = 6
    AMBIGUOUS = 7


class CellLocationResult(StrictModule):
    """Location plus every containing cell and explicit completeness evidence."""

    cell_ids: Array
    reference_coordinates: Array
    barycentric: Array
    geometry_residual: Array
    iterations: Array
    jacobian_condition: Array
    inside: Array
    used_fallback: Array
    candidate_count: Array
    status: Array
    successful: Array
    candidate_cells: Array
    candidate_reference: Array
    candidates_complete: Array
    locator_id: str = eqx.field(static=True)


class LocatedCellMap(Protocol):
    """Metadata and paired physical/reference evaluation for one cell block."""

    cell_count: int
    coordinate_count: int
    ambient_dimension: int
    reference_dimension: int
    cell_map_id: str
    topology_id: Identifier


class AbstractCellLocator(StrictModule):
    """Inverse map of a prepared block with complete candidate reporting."""

    cell_map: eqx.AbstractVar[LocatedCellMap]
    coordinates: eqx.AbstractVar[Array]
    locator_id: eqx.AbstractVar[str]

    @property
    def canonical_locator(self) -> AbstractCellLocator:
        """Return the source-complete locator, independent of query-side filters."""
        return self

    @abc.abstractmethod
    def locate(
        self,
        points: ConvertibleToArray,
        /,
        *,
        cell_mask: ConvertibleToArray | None = None,
    ) -> CellLocationResult:
        raise NotImplementedError

    def support_cells(self, points: ConvertibleToArray, /) -> Array:
        """Return every located candidate; negative slots are padding only."""
        return self.locate(points).candidate_cells


__all__ = [
    "AbstractCellLocator",
    "CellLocationResult",
    "CellLocationStatus",
    "LocatedCellMap",
]
