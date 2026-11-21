#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact rational vertices of conforming Voronoi cells.

The native kernel defines every vertex exactly from its seeds and the source
surface (a circumcentre, a surface edge on a bisector, a triangle on two
bisectors, or a surface vertex) and exports it as ``x_k = N_k / D`` with each
``N_k`` and ``D`` a floating-point expansion. Bisector faces are then exactly
planar on their bisector planes and surface faces on their source triangles,
and the binary64 mesh coordinates are the nearest values of these rationals.
"""

from __future__ import annotations

from fractions import Fraction
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._exact_power_geometry import ExactPoint3


def _expansion_values(
    offsets: np.ndarray, values: np.ndarray, /
) -> list[tuple[int, int]]:
    """Each expansion as an integer and a binary exponent: ``n * 2**e``."""
    mantissas, exponents = np.frexp(values)
    integers = (mantissas * 2.0**53).astype(np.int64).tolist()
    powers = (exponents.astype(np.int64) - 53).tolist()
    result: list[tuple[int, int]] = []
    for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        if start == stop:
            result.append((0, 0))
            continue
        lowest = min(powers[start:stop])
        total = 0
        for index in range(start, stop):
            total += integers[index] << (powers[index] - lowest)
        result.append((total, lowest))
    return result


def _quotient(numerator: tuple[int, int], denominator: tuple[int, int], /) -> Fraction:
    top, top_power = numerator
    bottom, bottom_power = denominator
    shift = top_power - bottom_power
    if shift >= 0:
        return Fraction(top << shift, bottom)
    return Fraction(top, bottom << -shift)


@final
class ExactVoronoiCellGeometrySource(StrictModule, NonTrainableState):
    """Exact rational conforming Voronoi vertices and their binary64 shadows."""

    runtime_vertices: jnp.ndarray
    exact_vertices: tuple[ExactPoint3, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        exact_offsets: ArrayLike,
        exact_values: ArrayLike,
        /,
        *,
        source_id: str,
        source_revision: str,
    ) -> None:
        rounded = np.asarray(vertices, dtype=np.float64)
        offsets = np.asarray(exact_offsets)
        components = np.asarray(exact_values, dtype=np.float64)
        if (
            rounded.ndim != 2
            or rounded.shape[1] != 3
            or offsets.dtype.kind not in "iu"
            or offsets.shape != (4 * rounded.shape[0] + 1,)
            or offsets[0] != 0
            or offsets[-1] != components.size
            or np.any(np.diff(offsets) < 0)
            or not np.all(np.isfinite(components))
        ):
            raise ValueError(
                "Exact Voronoi vertex expansions must align with the vertices."
            )
        if not source_id.strip() or not source_revision.strip():
            raise ValueError("Voronoi source identity and revision must be nonempty.")
        expansions = _expansion_values(offsets.astype(np.int64), components)
        exact: list[ExactPoint3] = []
        for vertex in range(rounded.shape[0]):
            denominator = expansions[4 * vertex + 3]
            if denominator[0] == 0:
                raise ValueError("An exact Voronoi vertex has a zero denominator.")
            exact.append(
                (
                    _quotient(expansions[4 * vertex], denominator),
                    _quotient(expansions[4 * vertex + 1], denominator),
                    _quotient(expansions[4 * vertex + 2], denominator),
                )
            )
        shadows = np.asarray(
            [[float(value) for value in point] for point in exact], dtype=np.float64
        ).reshape((-1, 3))
        if not np.array_equal(shadows, rounded):
            raise ValueError(
                "Voronoi vertices must be the nearest binary64 values of their exact points."
            )
        self.runtime_vertices = jnp.asarray(rounded)
        self.exact_vertices = tuple(exact)
        self.source_id = source_id.strip()
        self.source_revision = source_revision.strip()
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-voronoi-cell-geometry-source",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "exact": array_tree_fingerprint((offsets, components)),
                "vertices": array_tree_fingerprint(rounded),
            }
        )

    @property
    def rounded_vertices(self) -> np.ndarray:
        return np.asarray(self.runtime_vertices)

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> tuple[ExactPoint3, ...]:
        """Exact rational coordinates of the selected (default: all) vertices."""
        if indices is None:
            return self.exact_vertices
        rows = np.asarray(indices)
        if rows.dtype.kind not in "iu" or rows.ndim != 1:
            raise TypeError("Voronoi coordinate selection must be an integer vector.")
        if np.any(rows < 0) or np.any(rows >= len(self.exact_vertices)):
            raise IndexError("Voronoi coordinate selection is outside the vertex bank.")
        return tuple(self.exact_vertices[index] for index in rows.tolist())


__all__ = ["ExactVoronoiCellGeometrySource"]
