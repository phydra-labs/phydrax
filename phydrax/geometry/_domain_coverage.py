#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared source-domain boundaries and exact oriented region measures."""

from __future__ import annotations

from fractions import Fraction
from typing import final, Literal, TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

from .._deferred_exports import deferred_exports
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..geometry._rational import rational_vector
from ..linalg._exact import determinant_exact
from ..typing import Dim, HostFloat64, HostInt64, parse, Scope


if TYPE_CHECKING:
    from ._domain_certification import (
        certify_domain_coverage as certify_domain_coverage,
        DomainCoverageCertificate as DomainCoverageCertificate,
    )


class _DomainVertexDim(Dim, minimum=1):
    """Vertices of one source-domain boundary."""


class _DomainFacetDim(Dim, minimum=1):
    """Oriented source facets."""


class _DomainAmbientDim(Dim, minimum=2):
    """Ambient dimension of the source-domain boundary."""


@final
class PiecewiseLinearDomain(StrictModule, NonTrainableState):
    """An independently declared, oriented affine boundary and region partition.

    Facets are directed edges in two dimensions and oriented triangles in three.
    For each facet, its order points from ``facet_regions[:, 0]`` into
    ``facet_regions[:, 1]``; ``-1`` denotes the exterior. Every region chain is
    required to be closed and positively oriented before it can serve as source
    authority.
    """

    __strict_contract__ = True

    vertices: HostFloat64[_DomainVertexDim, _DomainAmbientDim]
    facets: HostInt64[_DomainFacetDim, _DomainAmbientDim]
    facet_regions: HostInt64[_DomainFacetDim, Literal[2]]
    region_ids: tuple[str, ...]
    region_occurrence_paths: tuple[tuple[str, ...], ...]
    source_id: str
    source_revision: str
    domain_id: str

    def __init__(
        self,
        vertices: ArrayLike,
        facets: ArrayLike,
        facet_regions: ArrayLike,
        region_ids: tuple[str, ...],
        /,
        *,
        source_id: str,
        region_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> None:
        scope = Scope()
        points = parse(
            np.asarray(vertices, dtype=np.float64),
            HostFloat64[_DomainVertexDim, _DomainAmbientDim],
            "vertices",
            scope=scope,
        )
        raw_facets = np.asarray(facets)
        raw_pairs = np.asarray(facet_regions)
        if raw_facets.dtype.kind not in "iu" or raw_pairs.dtype.kind not in "iu":
            raise TypeError("facets and facet_regions must contain integer indices.")
        if raw_facets.dtype.kind == "u" and np.any(raw_facets > np.iinfo(np.int64).max):
            raise ValueError("facet vertex indices exceed the int64 range.")
        if raw_pairs.dtype.kind == "u" and np.any(raw_pairs > np.iinfo(np.int64).max):
            raise ValueError("facet region indices exceed the int64 range.")
        rows = parse(
            raw_facets.astype(np.int64, copy=False),
            HostInt64[_DomainFacetDim, _DomainAmbientDim],
            "facets",
            scope=scope,
        )
        pairs = parse(
            raw_pairs.astype(np.int64, copy=False),
            HostInt64[_DomainFacetDim, Literal[2]],
            "facet_regions",
            scope=scope,
        )
        points = np.array(points, copy=True)
        rows = np.array(rows, copy=True)
        pairs = np.array(pairs, copy=True)
        identifiers = tuple(canonical_identifier(value, "region_ids") for value in region_ids)
        paths = (
            tuple(() for _ in identifiers)
            if region_occurrence_paths is None
            else tuple(tuple(path) for path in region_occurrence_paths)
        )
        if len(paths) != len(identifiers):
            raise ValueError("region_occurrence_paths must align with region_ids.")
        if any(
            not isinstance(raw_path, tuple)
            or any(
                not isinstance(component, str)
                or not component
                or component != component.strip()
                for component in path
            )
            for raw_path, path in zip(
                region_occurrence_paths
                if region_occurrence_paths is not None
                else tuple(() for _ in identifiers),
                paths,
                strict=True,
            )
        ):
            raise ValueError(
                "Region occurrence paths must contain canonical names or be empty."
            )
        if len(set(zip(identifiers, paths, strict=True))) != len(identifiers):
            raise ValueError(
                "region_ids and occurrence paths must uniquely identify each source region."
            )
        source = canonical_identifier(source_id, "source_id")
        dimension = points.shape[1]
        if dimension not in (2, 3) or rows.shape[1] != dimension:
            raise ValueError("facets must be edges in 2-D or triangles in 3-D.")
        if not np.all(np.isfinite(points)):
            raise ValueError("source-domain vertices must be finite.")
        if np.any(rows < 0) or np.any(rows >= points.shape[0]):
            raise ValueError("source facets must index declared source vertices.")
        if (
            np.any(pairs < -1)
            or np.any(pairs >= len(identifiers))
            or np.any(pairs[:, 0] == pairs[:, 1])
            or np.all(pairs == -1, axis=1).any()
        ):
            raise ValueError(
                "facet regions must be distinct declared regions or exterior."
            )

        exact_points = tuple(rational_vector(point) for point in points)
        facet_keys: set[tuple[int, ...]] = set()
        for facet in rows:
            indices = tuple(int(value) for value in facet)
            key = tuple(sorted(indices))
            if key in facet_keys:
                raise ValueError("source boundary contains a duplicate facet identity.")
            facet_keys.add(key)
            coordinates = tuple(exact_points[index] for index in indices)
            if dimension == 2:
                if coordinates[0] == coordinates[1]:
                    raise ValueError("source boundary edges must be nondegenerate.")
            elif not any(
                (
                    (coordinates[1][1] - coordinates[0][1])
                    * (coordinates[2][2] - coordinates[0][2])
                    - (coordinates[1][2] - coordinates[0][2])
                    * (coordinates[2][1] - coordinates[0][1]),
                    (coordinates[1][2] - coordinates[0][2])
                    * (coordinates[2][0] - coordinates[0][0])
                    - (coordinates[1][0] - coordinates[0][0])
                    * (coordinates[2][2] - coordinates[0][2]),
                    (coordinates[1][0] - coordinates[0][0])
                    * (coordinates[2][1] - coordinates[0][1])
                    - (coordinates[1][1] - coordinates[0][1])
                    * (coordinates[2][0] - coordinates[0][0]),
                )
            ):
                raise ValueError("source boundary triangles must be nondegenerate.")

        measures = _oriented_region_measures(exact_points, rows, pairs, len(identifiers))
        if any(value <= 0 for value in measures):
            raise ValueError(
                "each source region must be a closed, positively oriented chain."
            )
        for array in (points, rows, pairs):
            array.setflags(write=False)
        source_revision = canonical_fingerprint(
            {
                "vertices": array_tree_fingerprint(points),
                "facets": array_tree_fingerprint(rows),
                "facet_regions": array_tree_fingerprint(pairs),
                "region_ids": identifiers,
                "region_occurrence_paths": paths,
            }
        )
        self.vertices = points
        self.facets = rows
        self.facet_regions = pairs
        self.region_ids = identifiers
        self.region_occurrence_paths = paths
        self.source_id = source
        self.source_revision = source_revision
        self.domain_id = canonical_fingerprint(
            {
                "kind": "piecewise-linear-domain",
                "source_id": source,
                "source_revision": source_revision,
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return self.vertices.shape[1]

    def exact_region_measures(self, /) -> tuple[Fraction, ...]:
        exact_points = tuple(rational_vector(point) for point in self.vertices)
        return _oriented_region_measures(
            exact_points, self.facets, self.facet_regions, len(self.region_ids)
        )


def _oriented_region_measures(
    points: tuple[tuple[Fraction, ...], ...],
    facets: np.ndarray,
    facet_regions: np.ndarray,
    region_count: int,
    /,
) -> tuple[Fraction, ...]:
    dimension = len(points[0])
    measures = [Fraction(0) for _ in range(region_count)]
    incidence: list[dict[tuple[int, int], tuple[int, int]]] = [
        {} for _ in range(region_count)
    ]
    for row, (first_region, second_region) in zip(facets, facet_regions, strict=True):
        vertices = tuple(int(value) for value in row)
        for region, orientation in ((int(first_region), 1), (int(second_region), -1)):
            if region < 0:
                continue
            if dimension == 2:
                start, end = vertices if orientation > 0 else vertices[::-1]
                key = (start, end)
                previous = incidence[region].get(key, (0, 0))
                incidence[region][key] = (previous[0] + 1, previous[1] + 1)
                measures[region] += (
                    points[start][0] * points[end][1] - points[start][1] * points[end][0]
                ) / 2
            else:
                signed = determinant_exact(tuple(points[index] for index in vertices)) / 6
                measures[region] += signed if orientation > 0 else -signed
                ordered = vertices if orientation > 0 else vertices[::-1]
                for first, second in zip(ordered, ordered[1:] + ordered[:1], strict=True):
                    key = (min(first, second), max(first, second))
                    sign = 1 if first < second else -1
                    count, balance = incidence[region].get(key, (0, 0))
                    incidence[region][key] = (count + 1, balance + sign)
    for region, edges in enumerate(incidence):
        if not edges:
            raise ValueError(f"source region {region} has no oriented boundary facets.")
        if dimension == 2:
            outgoing: dict[int, int] = {}
            incoming: dict[int, int] = {}
            for (start, end), (count, _) in edges.items():
                if count != 1:
                    raise ValueError("source region boundary repeats an oriented edge.")
                outgoing[start] = outgoing.get(start, 0) + 1
                incoming[end] = incoming.get(end, 0) + 1
            if outgoing != incoming:
                raise ValueError(
                    "source region boundary edges do not form a closed chain."
                )
        elif any(count != 2 or balance != 0 for count, balance in edges.values()):
            raise ValueError(
                "source region boundary triangles do not form a closed oriented chain."
            )
    return tuple(measures)


__getattr__, __dir__ = deferred_exports(
    __name__,
    globals(),
    {
        "DomainCoverageCertificate": ".._domain_certification",
        "certify_domain_coverage": ".._domain_certification",
    },
)


__all__ = [
    "DomainCoverageCertificate",
    "PiecewiseLinearDomain",
    "certify_domain_coverage",
]
