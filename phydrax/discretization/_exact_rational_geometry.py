from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
)
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import CoordinateSourceBank


type RationalAction = tuple[tuple[int, int, int], ...]
type RationalActions = tuple[RationalAction, ...]


@final
class ExactRationalCellGeometrySource(StrictModule, NonTrainableState):
    """Parent-bound exact rational CSR actions for a lifted coordinate bank."""

    parent: CellGeometrySpec
    actions: RationalActions = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        parent: CellGeometrySpec,
        actions: Sequence[Sequence[tuple[int, int, int]]],
        source_id: str,
        source_revision: str,
        /,
    ) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be a non-empty string.")
        if not isinstance(source_revision, str) or not source_revision.strip():
            raise ValueError("source_revision must be a non-empty string.")
        rows = tuple(tuple(tuple(term) for term in row) for row in actions)
        parent_coordinates = parent.rational_source_coordinates()
        if not rows or not parent_coordinates:
            raise ValueError(
                "Rational coordinate lifting requires parent and target rows."
            )
        for row in rows:
            if not 1 <= len(row) <= 8:
                raise ValueError(
                    "Rational coordinate actions require one to eight CSR entries per row."
                )
            previous = -1
            total = Fraction(0)
            for term in row:
                if len(term) != 3 or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    for value in term
                ):
                    raise TypeError(
                        "Rational coordinate actions require integer triples."
                    )
                index, numerator, denominator = (int(value) for value in term)
                if (
                    index <= previous
                    or index >= len(parent_coordinates)
                    or numerator == 0
                    or denominator <= 0
                ):
                    raise ValueError(
                        "Rational coordinate actions require sorted unique sources and nonzero ratios."
                    )
                coefficient = Fraction(numerator, denominator)
                if (coefficient.numerator, coefficient.denominator) != (
                    numerator,
                    denominator,
                ):
                    raise ValueError("Rational coordinate action ratios must be reduced.")
                previous = index
                total += coefficient
            if total != 1:
                raise ValueError(
                    "Rational coordinate action rows must sum exactly to one."
                )
        canonical_rows = tuple(
            tuple(
                (int(index), int(numerator), int(denominator))
                for index, numerator, denominator in row
            )
            for row in rows
        )
        self.parent = parent
        self.actions = canonical_rows
        self.source_id = source_id.strip()
        self.source_revision = source_revision.strip()
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-rational-cell-geometry-source",
                "parent_layout": parent.geometry_layout_id,
                "parent_source_map": parent.source_map_id,
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "actions": canonical_rows,
            }
        )

    def source_coordinates(self, /) -> CoordinateSourceBank:
        """Apply each rational CSR row to the authenticated exact parent bank."""
        parent_coordinates = self.parent.rational_source_coordinates()
        dimension = len(parent_coordinates[0])
        return tuple(
            tuple(
                sum(
                    (
                        Fraction(numerator, denominator) * parent_coordinates[index][axis]
                        for index, numerator, denominator in row
                    ),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            for row in self.actions
        )


@checked
def rational_geometry_lift(
    parent: CellGeometrySpec,
    mesh: CellMesh,
    elements: Mapping[str, CellGeometryElement],
    geometry_dofs: Mapping[str, ArrayLike],
    actions: Sequence[Sequence[tuple[int, int, int]]],
    /,
    *,
    source_id: str,
    source_revision: str,
) -> CellGeometrySpec:
    """Lift coordinates from a parent through exact rational interpolation CSR."""
    if not isinstance(parent, CellGeometrySpec) or not isinstance(mesh, CellMesh):
        raise TypeError(
            "Rational geometry lifting requires parent geometry and target mesh owners."
        )
    source = ExactRationalCellGeometrySource(parent, actions, source_id, source_revision)
    exact = source.source_coordinates()
    coordinates = np.asarray(
        [[float(value) for value in row] for row in exact], dtype=np.float64
    )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "Rational geometry coordinates must have finite binary64 representatives."
        )
    return CellGeometrySpec(
        elements,
        geometry_dofs,
        coordinates,
        source_map=source,
        source_mesh=mesh,
    )


__all__ = [
    "ExactRationalCellGeometrySource",
    "RationalAction",
    "RationalActions",
    "rational_geometry_lift",
]
