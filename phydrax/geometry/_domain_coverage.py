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
        identifiers = tuple(
            canonical_identifier(value, "region_ids") for value in region_ids
        )
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

        integers, exponent = _dyadic_integer_coordinates(points)
        _, first_rows, key_rows = np.unique(
            np.sort(rows, axis=1), axis=0, return_index=True, return_inverse=True
        )
        duplicates = np.flatnonzero(
            first_rows[key_rows.reshape(-1)] != np.arange(rows.shape[0])
        )
        corners = integers[rows]
        if dimension == 2:
            degenerate = np.flatnonzero(np.all(corners[:, 0] == corners[:, 1], axis=1))
        else:
            first_edge = corners[:, 1] - corners[:, 0]
            second_edge = corners[:, 2] - corners[:, 0]
            degenerate = np.flatnonzero(
                (
                    first_edge[:, 1] * second_edge[:, 2]
                    - first_edge[:, 2] * second_edge[:, 1]
                    == 0
                )
                & (
                    first_edge[:, 2] * second_edge[:, 0]
                    - first_edge[:, 0] * second_edge[:, 2]
                    == 0
                )
                & (
                    first_edge[:, 0] * second_edge[:, 1]
                    - first_edge[:, 1] * second_edge[:, 0]
                    == 0
                )
            )
        # Report the first offending facet in row order; a duplicate precedes its degeneracy.
        if duplicates.size and (not degenerate.size or duplicates[0] <= degenerate[0]):
            raise ValueError("source boundary contains a duplicate facet identity.")
        if degenerate.size:
            raise ValueError(
                "source boundary edges must be nondegenerate."
                if dimension == 2
                else "source boundary triangles must be nondegenerate."
            )

        measures = _oriented_region_measures(
            integers, exponent, rows, pairs, len(identifiers)
        )
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
        integers, exponent = _dyadic_integer_coordinates(self.vertices)
        return _oriented_region_measures(
            integers, exponent, self.facets, self.facet_regions, len(self.region_ids)
        )


def _dyadic_integer_coordinates(points: np.ndarray, /) -> tuple[np.ndarray, int]:
    """Python-integer coordinates ``m`` and one exponent ``e`` with ``points == m * 2**e``."""
    mantissas, exponents = np.frexp(points)
    # |mantissa| < 1, so the 53-bit significand is an exact int64.
    significands = np.ldexp(mantissas, 53).astype(np.int64)
    shifts = exponents.astype(np.int64) - 53
    nonzero = significands != 0
    exponent = int(shifts[nonzero].min()) if np.any(nonzero) else 0
    shifts[~nonzero] = exponent
    return significands.astype(object) << (shifts - exponent).astype(object), exponent


def _oriented_region_measures(
    integers: np.ndarray,
    exponent: int,
    facets: np.ndarray,
    facet_regions: np.ndarray,
    region_count: int,
    /,
) -> tuple[Fraction, ...]:
    """Exact oriented measures of closed region chains over dyadic integer coordinates."""
    dimension = integers.shape[1]
    corners = integers[facets]
    if dimension == 2:
        # Twice the signed area swept from the origin by each directed edge.
        swept = corners[:, 0, 0] * corners[:, 1, 1] - corners[:, 0, 1] * corners[:, 1, 0]
        scale = Fraction(2) ** (2 * exponent) / 2
    else:
        # Six times the signed volume of the tetrahedron spanned with the origin.
        swept = (
            corners[:, 0, 0]
            * (corners[:, 1, 1] * corners[:, 2, 2] - corners[:, 1, 2] * corners[:, 2, 1])
            - corners[:, 0, 1]
            * (corners[:, 1, 0] * corners[:, 2, 2] - corners[:, 1, 2] * corners[:, 2, 0])
            + corners[:, 0, 2]
            * (corners[:, 1, 0] * corners[:, 2, 1] - corners[:, 1, 1] * corners[:, 2, 0])
        )
        scale = Fraction(2) ** (3 * exponent) / 6
    first_regions = facet_regions[:, 0]
    second_regions = facet_regions[:, 1]
    # Each facet bounds facet_regions[:, 0] positively and facet_regions[:, 1] negatively.
    sides = np.concatenate(
        (np.flatnonzero(first_regions >= 0), np.flatnonzero(second_regions >= 0))
    )
    side_regions = np.concatenate(
        (first_regions[first_regions >= 0], second_regions[second_regions >= 0])
    )
    side_orientations = np.concatenate(
        (
            np.ones((np.count_nonzero(first_regions >= 0),), dtype=np.int64),
            -np.ones((np.count_nonzero(second_regions >= 0),), dtype=np.int64),
        )
    )
    region_order = np.argsort(side_regions, kind="stable")
    ordered_regions = side_regions[region_order]
    region_starts = np.flatnonzero(
        np.concatenate(
            (np.ones((1,), dtype=np.bool_), ordered_regions[1:] != ordered_regions[:-1])
        )
    )
    signed = swept[sides] * side_orientations.astype(object)
    totals = dict(
        zip(
            ordered_regions[region_starts].tolist(),
            np.add.reduceat(signed[region_order], region_starts).tolist(),
            strict=True,
        )
    )
    measures = tuple(totals.get(region, 0) * scale for region in range(region_count))
    side_facets = facets[sides]
    if dimension == 2:
        forward = side_orientations > 0
        starts = np.where(forward, side_facets[:, 0], side_facets[:, 1])
        ends = np.where(forward, side_facets[:, 1], side_facets[:, 0])
        edge_regions = side_regions
        edge_low, edge_high = starts, ends
        edge_signs = np.ones_like(starts)
    else:
        following = side_facets[:, (1, 2, 0)]
        edge_low = np.minimum(side_facets, following).reshape(-1)
        edge_high = np.maximum(side_facets, following).reshape(-1)
        edge_signs = (
            np.where(side_facets < following, 1, -1) * side_orientations[:, None]
        ).reshape(-1)
        edge_regions = np.repeat(side_regions, 3)
    order = np.lexsort((edge_high, edge_low, edge_regions))
    edge_regions, edge_low, edge_high, edge_signs = (
        edge_regions[order],
        edge_low[order],
        edge_high[order],
        edge_signs[order],
    )
    starts_group = np.ones((edge_regions.size,), dtype=np.bool_)
    starts_group[1:] = (
        (edge_regions[1:] != edge_regions[:-1])
        | (edge_low[1:] != edge_low[:-1])
        | (edge_high[1:] != edge_high[:-1])
    )
    group_starts = np.flatnonzero(starts_group)
    counts = np.diff(np.append(group_starts, edge_regions.size))
    balances = np.add.reduceat(edge_signs, group_starts)
    group_regions = edge_regions[group_starts]
    present = np.zeros((region_count,), dtype=np.bool_)
    present[group_regions] = True
    if dimension == 2:
        repeated = np.zeros((region_count,), dtype=np.bool_)
        repeated[group_regions[counts != 1]] = True
        # Every vertex of a closed chain has equal in- and out-degree per region.
        endpoint_regions = np.concatenate((edge_regions, edge_regions))
        endpoints = np.concatenate((edge_low, edge_high))
        endpoint_signs = np.concatenate(
            (np.ones_like(edge_low), -np.ones_like(edge_high))
        )
        endpoint_order = np.lexsort((endpoints, endpoint_regions))
        endpoint_regions = endpoint_regions[endpoint_order]
        endpoints = endpoints[endpoint_order]
        endpoint_starts = np.flatnonzero(
            np.concatenate(
                (
                    np.ones((1,), dtype=np.bool_),
                    (endpoint_regions[1:] != endpoint_regions[:-1])
                    | (endpoints[1:] != endpoints[:-1]),
                )
            )
        )
        imbalance = np.add.reduceat(endpoint_signs[endpoint_order], endpoint_starts)
        open_chain = np.zeros((region_count,), dtype=np.bool_)
        open_chain[endpoint_regions[endpoint_starts][imbalance != 0]] = True
    else:
        repeated = np.zeros((region_count,), dtype=np.bool_)
        open_chain = np.zeros((region_count,), dtype=np.bool_)
        open_chain[group_regions[(counts != 2) | (balances != 0)]] = True
    for region in range(region_count):
        if not present[region]:
            raise ValueError(f"source region {region} has no oriented boundary facets.")
        if repeated[region]:
            raise ValueError("source region boundary repeats an oriented edge.")
        if open_chain[region]:
            raise ValueError(
                "source region boundary edges do not form a closed chain."
                if dimension == 2
                else "source region boundary triangles do not form a closed oriented chain."
            )
    return measures


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
