#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared result and protocol for complete physical cell location."""

from __future__ import annotations

import abc
from enum import IntEnum
from fractions import Fraction
from typing import final, Literal, Protocol, TypeAlias

import equinox as eqx
from jax import Array

from .._strict import StrictModule
from ..typing import ConvertibleToArray, Identifier, parse


class CellLocationStatus(IntEnum):
    LOCATED = 0
    OUTSIDE = 1
    DEGENERATE_CELL = 2
    NONFINITE = 3
    RESOURCE_EXCEEDED = 4
    INVERSE_MAP_EXHAUSTED = 5
    CANDIDATE_SEARCH_INCOMPLETE = 6
    AMBIGUOUS = 7


InverseRefusalReason: TypeAlias = Literal[
    "singular-tolerance-root",
    "rootless-tolerance-root",
    "parameter-resolution",
    "unbounded-enclosure",
    "root-refinement-stalled",
    "duplicate-root-unmerged",
]


@final
class CellInverseRefusal(StrictModule):
    """Exact reference-box evidence for one undecided inverse map of one query in one cell.

    ``witness`` is empty, one exact tolerance root (a reference point inside the
    cell margin whose physical residual is within tolerance), or the two ends of a
    segment of tolerance roots across which the Jacobian determinant changes sign.
    All fields are host metadata; the evidence carries no device data.
    """

    point: int = eqx.field(static=True)
    cell: int = eqx.field(static=True)
    reason: InverseRefusalReason = eqx.field(static=True)
    lower: tuple[Fraction, ...] = eqx.field(static=True)
    upper: tuple[Fraction, ...] = eqx.field(static=True)
    witness: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        point: int,
        cell: int,
        reason: InverseRefusalReason,
        lower: tuple[Fraction, ...],
        upper: tuple[Fraction, ...],
        witness: tuple[tuple[Fraction, ...], ...],
        /,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (point, cell)
        ):
            raise ValueError(
                "Inverse refusal point and cell must be non-negative integers."
            )
        reason_ = parse(reason, InverseRefusalReason, "reason")
        lower_, upper_ = tuple(lower), tuple(upper)
        witness_ = tuple(tuple(end) for end in witness)
        dimension = len(lower_)
        if (
            dimension < 1
            or len(upper_) != dimension
            or len(witness_) > 2
            or any(len(end) != dimension for end in witness_)
        ):
            raise ValueError(
                "Inverse refusal boxes and witnesses must share one reference dimension."
            )
        if not all(
            isinstance(value, Fraction)
            for values in (lower_, upper_, *witness_)
            for value in values
        ):
            raise TypeError("Inverse refusal coordinates must be exact Fractions.")
        if any(first > last for first, last in zip(lower_, upper_, strict=True)) or any(
            not first <= value <= last
            for end in witness_
            for value, first, last in zip(end, lower_, upper_, strict=True)
        ):
            raise ValueError("Inverse refusal witnesses must lie in an ordered box.")
        self.point, self.cell, self.reason = point, cell, reason_
        self.lower, self.upper, self.witness = lower_, upper_, witness_


class CellLocationRefusalError(ValueError):
    """A location consumer refused queries whose location stayed undecided.

    ``inverse_refusals`` carries the locator's exact box evidence for those queries,
    with ``point`` re-indexed to the consumer's own query rows as its message states.
    It is empty when the refusal has no undecided inverse map behind it (for
    example an ambiguous or incomplete candidate search).
    """

    def __init__(
        self, message: str, inverse_refusals: tuple[CellInverseRefusal, ...], /
    ) -> None:
        super().__init__(message)
        self.inverse_refusals = inverse_refusals


class CellLocationResult(StrictModule):
    """Location plus every containing cell and explicit completeness evidence.

    ``inverse_refusals`` holds the certified box evidence of every undecided exact
    inverse map, ordered by query point and cell; it is empty for locators whose
    inverse is a traced iteration, which report its evidence through
    ``iterations``, ``geometry_residual`` and ``status``.
    """

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
    inverse_refusals: tuple[CellInverseRefusal, ...]
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
    "CellInverseRefusal",
    "CellLocationRefusalError",
    "CellLocationResult",
    "CellLocationStatus",
    "InverseRefusalReason",
    "LocatedCellMap",
]
