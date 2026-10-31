from __future__ import annotations

from fractions import Fraction
from typing import Sequence, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._exact_algebraic_geometry import ExactScalar, RadicalElement


if TYPE_CHECKING:
    from ._periodic_exact import PeriodicMeshIdentification


def _token(value: ExactScalar, /) -> tuple[object, ...]:
    if isinstance(value, Fraction):
        return ("rational", value.numerator, value.denominator)
    if isinstance(value, RadicalElement):
        return (
            "radical",
            tuple(
                (radicand, coefficient.numerator, coefficient.denominator)
                for radicand, coefficient in value.terms
            ),
        )
    raise TypeError("PLC exact coordinates require canonical rational/radical scalars.")


class ExactPlcCellGeometrySource(StrictModule, NonTrainableState):
    """Exact output-coordinate bank with source-owned PLC vertex ancestry.

    The execution mesh sees correctly rounded binary64 coordinates. Transfer and
    reconstruction retain the exact source-coordinate leaves and stable authored
    entity/occurrence identity; rounded coordinates are never promoted back to
    source truth.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    vertex_global_ids: Array
    rounded_vertices: Array
    exact_coordinates: tuple[tuple[ExactScalar, ...], ...] = eqx.field(static=True)
    vertex_source_entity_ids: tuple[str | None, ...] = eqx.field(static=True)
    vertex_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_id: str,
        source_revision: str,
        vertex_global_ids: ArrayLike,
        exact_coordinates: Sequence[Sequence[ExactScalar]],
        /,
        *,
        vertex_source_entity_ids: Sequence[str | None],
        vertex_occurrence_paths: Sequence[tuple[str, ...]],
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        identifiers = np.asarray(vertex_global_ids)
        exact = tuple(tuple(row) for row in exact_coordinates)
        entities = tuple(vertex_source_entity_ids)
        occurrences = tuple(tuple(path) for path in vertex_occurrence_paths)
        if (
            not source
            or not revision
            or identifiers.ndim != 1
            or identifiers.dtype.kind not in "iu"
            or identifiers.size == 0
            or np.any(identifiers < 0)
            or np.unique(identifiers).size != identifiers.size
            or len(exact) != identifiers.size
            or not exact[0]
            or len(entities) != identifiers.size
            or len(occurrences) != identifiers.size
            or any(len(row) != len(exact[0]) for row in exact)
            or any(
                entity is not None and (not isinstance(entity, str) or not entity)
                for entity in entities
            )
            or any(
                any(
                    not isinstance(part, str) or not part or part != part.strip()
                    for part in path
                )
                for path in occurrences
            )
        ):
            raise ValueError(
                "Exact PLC coordinate rows and source identities must align."
            )
        rounded = np.asarray(
            [[float(value) for value in row] for row in exact], dtype=np.float64
        )
        if not np.all(np.isfinite(rounded)):
            raise ValueError(
                "Exact PLC coordinates must have finite binary64 execution views."
            )
        for row in exact:
            for value in row:
                _token(value)
        ids = np.asarray(identifiers, dtype=np.int64).copy()
        ids.setflags(write=False)
        rounded = np.array(rounded, copy=True)
        rounded.setflags(write=False)
        self.source_id = source
        self.source_revision = revision
        self.vertex_global_ids = jnp.asarray(ids)
        self.rounded_vertices = jnp.asarray(rounded)
        self.exact_coordinates = exact
        self.vertex_source_entity_ids = entities
        self.vertex_occurrence_paths = occurrences
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-plc-cell-geometry-source",
                "source_id": source,
                "source_revision": revision,
                "vertex_global_ids": tuple(int(value) for value in ids),
                "exact_coordinates": tuple(
                    tuple(_token(value) for value in row) for row in exact
                ),
                "vertex_source_entity_ids": entities,
                "vertex_occurrence_paths": occurrences,
            }
        )

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> tuple[tuple[ExactScalar, ...], ...]:
        if indices is None:
            rows = np.arange(len(self.exact_coordinates), dtype=np.int64)
        else:
            rows = np.asarray(indices, dtype=np.int64)
            if (
                rows.ndim != 1
                or np.any(rows < 0)
                or np.any(rows >= len(self.exact_coordinates))
            ):
                raise IndexError(
                    "PLC coordinate source index lies outside the exact vertex bank."
                )
        return tuple(self.exact_coordinates[int(row)] for row in rows)

    def require_periodic_lift(
        self,
        coordinates: ArrayLike,
        cell: PeriodicMeshIdentification,
        representatives: ArrayLike,
        shifts: ArrayLike,
        /,
    ) -> None:
        """Prove exact deck action on the bank and its canonical binary64 shadow.

        Every lifted exact row is its root's exact row translated exactly by the
        row's winding; the execution coordinates are the canonical roundings.
        """
        from ._periodic_exact import exact_periodic_element, identification_orders

        points = np.asarray(coordinates, dtype=np.float64)
        exact = self.exact_coordinates
        if points.shape != (len(exact), len(exact[0])) or not np.array_equal(
            points, np.asarray(self.rounded_vertices)
        ):
            raise ValueError(
                "Exact periodic PLC coordinates do not match their canonical binary64 shadows."
            )
        roots = np.asarray(representatives)
        exponents = np.asarray(shifts)
        rank = len(identification_orders(cell))
        if (
            roots.shape != (points.shape[0],)
            or roots.dtype.kind not in "iu"
            or exponents.shape != (points.shape[0], rank)
            or exponents.dtype.kind not in "iu"
        ):
            raise ValueError("Exact periodic PLC orbit rows are malformed.")
        dimension = points.shape[1]
        transforms: dict[tuple[int, ...], tuple[tuple[Fraction, ...], ...]] = {}
        for row, root in enumerate(roots.astype(np.int64).tolist()):
            key = tuple(int(value) for value in exponents[row])
            matrix = transforms.get(key)
            if matrix is None:
                matrix = exact_periodic_element(cell, exponents[row])
                transforms[key] = matrix
            base = tuple(exact[int(root)]) + (Fraction(1),)
            expected = tuple(
                sum(
                    (
                        matrix[axis][column] * base[column]
                        for column in range(dimension + 1)
                    ),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            if expected != tuple(exact[row]):
                raise ValueError(
                    "Exact PLC coordinates violate their declared periodic deck action."
                )

    @classmethod
    def from_stored_fields(
        cls, value: ExactPlcCellGeometrySource, /
    ) -> ExactPlcCellGeometrySource:
        if not isinstance(value, ExactPlcCellGeometrySource):
            raise TypeError("value must be ExactPlcCellGeometrySource.")
        rebuilt = cls(
            value.source_id,
            value.source_revision,
            value.vertex_global_ids,
            value.exact_coordinates,
            vertex_source_entity_ids=value.vertex_source_entity_ids,
            vertex_occurrence_paths=value.vertex_occurrence_paths,
        )
        if rebuilt.source_map_id != value.source_map_id or not np.array_equal(
            np.asarray(rebuilt.rounded_vertices), np.asarray(value.rounded_vertices)
        ):
            raise ValueError(
                "Stored exact PLC coordinate source does not reproduce its source leaves."
            )
        return rebuilt


__all__ = ["ExactPlcCellGeometrySource"]
