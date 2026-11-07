#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One explicit invertible affine image of an entire reference PLC."""

from __future__ import annotations

from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..linalg._exact import determinant_exact
from ._compartments import CompartmentMeshingSource
from ._domain_coverage import PiecewiseLinearDomain
from ._rational import rational_vector


@final
class AffinePlcDomain(StrictModule, NonTrainableState):
    """Reference PLC with one binary64-authored, exactly interpreted affine map.

    Coordinates are evaluated over rationals, not by rounding each operation.
    Reflections reverse the facet order while retaining its material pair.
    The original PLC and all of its region identities remain authoritative.
    """

    reference_domain: PiecewiseLinearDomain
    authority: CompartmentMeshingSource | None
    matrix: np.ndarray
    offset: np.ndarray
    vertices: np.ndarray
    facets: np.ndarray
    facet_regions: np.ndarray
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    determinant: Fraction = eqx.field(static=True)
    exact_vertices: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    exact_matrix: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    exact_offset: tuple[Fraction, ...] = eqx.field(static=True)

    def __init__(
        self,
        reference_domain: PiecewiseLinearDomain,
        matrix: ArrayLike,
        offset: ArrayLike,
        /,
        *,
        source_revision: str,
        authority: CompartmentMeshingSource | None = None,
    ) -> None:
        if type(reference_domain) is not PiecewiseLinearDomain:
            raise TypeError("reference_domain must be an authored PiecewiseLinearDomain.")
        dimension = reference_domain.ambient_dimension
        linear = np.array(matrix, dtype=np.float64, copy=True)
        shift = np.array(offset, dtype=np.float64, copy=True)
        if linear.shape != (dimension, dimension) or shift.shape != (dimension,):
            raise ValueError(
                "Affine matrix and offset must match the reference dimension."
            )
        if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(shift)):
            raise ValueError("Affine matrix and offset must be finite.")
        exact_matrix = tuple(rational_vector(row) for row in linear)
        exact_offset = rational_vector(shift)
        determinant = determinant_exact(exact_matrix)
        if not determinant:
            raise ValueError("Affine matrix must be exactly invertible.")
        if authority is not None:
            _require_compartment_authority(
                reference_domain, linear, shift, source_revision, authority
            )
        exact_vertices = tuple(
            tuple(
                sum((a * x for a, x in zip(row, point, strict=True)), b)
                for row, b in zip(exact_matrix, exact_offset, strict=True)
            )
            for point in (rational_vector(value) for value in reference_domain.vertices)
        )
        try:
            vertices = np.asarray(exact_vertices, dtype=np.float64)
        except OverflowError as error:
            raise ValueError(
                "Affine image vertices must have finite binary64 carriers."
            ) from error
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Affine image vertices must have finite binary64 carriers.")
        facets = np.array(reference_domain.facets, copy=True)
        if determinant < 0:
            facets[:, [0, 1]] = facets[:, [1, 0]]
        for value in (linear, shift, vertices, facets):
            value.setflags(write=False)
        revision = canonical_identifier(source_revision, "source_revision")
        domain_id = canonical_fingerprint(
            {
                "kind": "global-affine-plc-source",
                "source_revision": revision,
                "authority": None if authority is None else authority.source_id,
                "reference_domain": reference_domain.domain_id,
                "matrix": array_tree_fingerprint(linear),
                "offset": array_tree_fingerprint(shift),
            }
        )
        self.reference_domain = reference_domain
        self.authority = authority
        self.matrix = linear
        self.offset = shift
        self.vertices = vertices
        self.facets = facets
        self.facet_regions = reference_domain.facet_regions
        self.region_ids = reference_domain.region_ids
        self.region_occurrence_paths = reference_domain.region_occurrence_paths
        self.source_id = reference_domain.source_id
        self.source_revision = revision
        self.domain_id = domain_id
        self.determinant = determinant
        self.exact_vertices = exact_vertices
        self.exact_matrix = exact_matrix
        self.exact_offset = exact_offset

    @property
    def ambient_dimension(self) -> int:
        return self.reference_domain.ambient_dimension

    def require_current(self, /) -> None:
        reference = self.reference_domain
        renewed = PiecewiseLinearDomain(
            reference.vertices,
            reference.facets,
            reference.facet_regions,
            reference.region_ids,
            source_id=reference.source_id,
            region_occurrence_paths=reference.region_occurrence_paths,
        )
        if renewed.domain_id != reference.domain_id:
            raise ValueError("Reference PLC differs from its retained identity.")
        rebuilt = type(self)(
            renewed,
            self.matrix,
            self.offset,
            source_revision=self.source_revision,
            authority=self.authority,
        )
        if (
            rebuilt.domain_id != self.domain_id
            or rebuilt.exact_vertices != self.exact_vertices
            or rebuilt.exact_matrix != self.exact_matrix
            or rebuilt.exact_offset != self.exact_offset
            or rebuilt.determinant != self.determinant
            or not np.array_equal(rebuilt.vertices, self.vertices)
            or not np.array_equal(rebuilt.facets, self.facets)
            or not np.array_equal(rebuilt.facet_regions, self.facet_regions)
            or rebuilt.region_ids != self.region_ids
            or rebuilt.region_occurrence_paths != self.region_occurrence_paths
            or rebuilt.source_id != self.source_id
        ):
            raise ValueError("Affine PLC source differs from its retained authority.")

    def transform_point(self, point: tuple[Fraction, ...], /) -> tuple[Fraction, ...]:
        if len(point) != self.ambient_dimension or any(
            not isinstance(x, Fraction) for x in point
        ):
            raise ValueError(
                "Affine source points must be exact reference-coordinate rationals."
            )
        return tuple(
            sum((a * x for a, x in zip(row, point, strict=True)), b)
            for row, b in zip(self.exact_matrix, self.exact_offset, strict=True)
        )

    def exact_region_measures(self, /) -> tuple[Fraction, ...]:
        return tuple(
            abs(self.determinant) * value
            for value in self.reference_domain.exact_region_measures()
        )


def _require_compartment_authority(
    reference: PiecewiseLinearDomain,
    matrix: np.ndarray,
    offset: np.ndarray,
    revision: str,
    authority: CompartmentMeshingSource,
    /,
) -> None:
    """Reconstruct categorical face incidence and require the exact PLC partition."""
    if not isinstance(authority, CompartmentMeshingSource):
        raise TypeError("authority must be a CompartmentMeshingSource.")
    from ..meshing._compartments import prepare_compartment_plc

    if (reference.source_id, revision) != (
        authority.source_id,
        authority.source_revision,
    ):
        raise ValueError(
            "Affine PLC source identity differs from the categorical authority."
        )
    affine = np.asarray(authority.labels.asset.spatial_affine.matrix, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.array_equal(
            matrix.view(np.uint64), affine[:3, :3].copy().view(np.uint64)
        )
        or not np.array_equal(
            offset.view(np.uint64), affine[:3, 3].copy().view(np.uint64)
        )
    ):
        raise ValueError("Affine PLC mapping differs from the categorical image affine.")
    complex_ = prepare_compartment_plc(authority).complex
    if (
        not np.array_equal(reference.vertices, complex_.vertices)
        or reference.region_ids != complex_.region_ids
        or reference.region_occurrence_paths != complex_.region_occurrence_paths
    ):
        raise ValueError(
            "Reference PLC vertices or regions differ from the categorical source."
        )
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    pairs = np.asarray(complex_.facet_regions, dtype=np.int64)
    lookup: dict[tuple[int, ...], tuple[int, ...]] = {}
    loops: list[tuple[int, ...]] = []
    for row, facet in enumerate(facets):
        loop = tuple(int(v) for v in vertices[offsets[row] : offsets[row + 1]])
        if len(loop) != 4:
            raise ValueError("Categorical source faces must be quadrilaterals.")
        loops.append(loop)
        for missing in range(4):
            key = tuple(sorted(loop[:missing] + loop[missing + 1 :]))
            lookup[key] = lookup.get(key, ()) + (row,)
    assigned: list[list[tuple[int, ...]]] = [[] for _ in loops]
    for triangle, pair in zip(
        np.asarray(reference.facets).tolist(), reference.facet_regions, strict=True
    ):
        values = tuple(int(v) for v in triangle)
        candidates = tuple(
            row
            for row in lookup.get(tuple(sorted(values)), ())
            if np.array_equal(pair, pairs[int(facets[row]), ::-1])
        )
        if len(candidates) != 1:
            raise ValueError(
                "Reference triangle has no unique categorical face and material ancestry."
            )
        row = candidates[0]
        loop = loops[row]
        order = [loop.index(v) for v in values]
        if (order[1] - order[0]) % 4 + (order[2] - order[1]) % 4 + (
            order[0] - order[2]
        ) % 4 != 4:
            raise ValueError(
                "Reference triangle reverses its categorical face orientation."
            )
        assigned[row].append(values)
    for loop, triangles in zip(loops, assigned, strict=True):
        if len(triangles) != 2 or set(triangles[0]) | set(triangles[1]) != set(loop):
            raise ValueError(
                "Reference triangles do not partition each categorical face."
            )
        shared = set(triangles[0]) & set(triangles[1])
        if len(shared) != 2 or any(
            {loop[i], loop[(i + 1) % 4]} == shared for i in range(4)
        ):
            raise ValueError(
                "Reference triangles do not share a categorical face diagonal."
            )


__all__ = ["AffinePlcDomain"]
