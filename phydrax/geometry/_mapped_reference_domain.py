#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Independent mapped-volume declarations with declared reference charts."""

from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_geometry_validity as cell_geometry_validity_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._mapped_coverage as mapped_coverage_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, integer
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked, Dim, HostInt64, parse
from ._mapped_corners import mapped_source_corner_coordinates


class _MappedSourceCellDim(Dim, minimum=1):
    """Cells of an independently declared mapped source."""


@final
class MappedReferenceDomain(StrictModule, NonTrainableState):
    """A declared reference domain and its independently owned physical maps.

    Source identity is explicit and distinct from any target approximation.
    Coverage requires rational root restrictions, independent reference coverage,
    source physical embedding and target physical embedding. It does not imply
    equivalence of these physical maps to a CAD or implicit source.
    """

    __strict_contract__ = True

    reference_domain: domain_coverage_owner.PiecewiseLinearDomain
    reference_mesh: CellMesh
    source_geometry: CellGeometrySpec
    cell_regions: HostInt64[_MappedSourceCellDim]
    certificate_limits: embedding_state_owner.MeshCertificateLimits
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reference_domain: domain_coverage_owner.PiecewiseLinearDomain,
        reference_mesh: cell_mesh_owner.CellMesh,
        source_geometry: cell_geometry_owner.CellGeometrySpec,
        cell_regions: ArrayLike,
        /,
        *,
        source_id: str,
        source_revision: str,
        certificate_limits: embedding_state_owner.MeshCertificateLimits | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        if reference_mesh.storage is not None or source_geometry.storage is not None:
            raise ValueError(
                "MappedReferenceDomain.reference_mesh must be serial with serial source geometry; got owner-local storage"
            )
        dimension = reference_mesh.topological_dimension
        if (
            dimension not in (2, 3)
            or dimension != reference_mesh.ambient_dimension
            or dimension != reference_domain.ambient_dimension
        ):
            raise ValueError(
                "MappedReferenceDomain.reference_mesh must have matching volume and domain dimensions of two or three; got incompatible dimensions"
            )
        if source_geometry.coordinates.shape[1] != dimension:
            raise ValueError(
                "MappedReferenceDomain.source_geometry.coordinates must have the declared volume dimension; got incompatible ambient dimension"
            )
        if source_geometry.restriction_source is not None:
            raise ValueError(
                "MappedReferenceDomain.source_geometry must declare unrestricted roots; got target restriction ancestry"
            )
        regions = parse(
            np.asarray(cell_regions, dtype=np.int64),
            HostInt64[_MappedSourceCellDim],
            "cell_regions",
        )
        count = sum(block.cell_count for block in reference_mesh.blocks)
        if (
            regions.shape != (count,)
            or np.any(regions < 0)
            or np.any(regions >= len(reference_domain.region_ids))
        ):
            raise ValueError(
                "MappedReferenceDomain.cell_regions must assign a declared region to every root; got invalid shape or region index"
            )
        limits = (
            embedding_state_owner.MeshCertificateLimits()
            if certificate_limits is None
            else certificate_limits
        )
        with embedding_state_owner.coordinate_certificate_request(
            limits, budget=budget
        ) as budget:
            elements, routes, _ = source_geometry.resolve(reference_mesh, budget=budget)
            values = source_geometry.source_coordinates(budget=budget)
            for block, element, route in zip(
                reference_mesh.blocks, elements, routes, strict=True
            ):
                if (
                    block.cell_kind
                    not in (
                        "triangle",
                        "quadrilateral",
                        "tetrahedron",
                        "hexahedron",
                        "prism",
                        "pyramid",
                    )
                    or element.cell_kind != block.cell_kind
                ):
                    raise ValueError(
                        "MappedReferenceDomain.roots must match supported reference cell kinds; got an unsupported or mismatched cell map"
                    )
                for local in (
                    tuple(values[index] for index in row)
                    for row in np.asarray(route, dtype=np.int64)
                ):
                    if (
                        coordinate_enclosure_owner.coordinate_polynomials(
                            element, local, budget=budget
                        )
                        is None
                    ):
                        raise ValueError(
                            "MappedReferenceDomain.source_geometry must resolve coordinate polynomials for every root; got unresolved source expressions"
                        )
            mapped_source_corner_coordinates(
                reference_mesh, source_geometry, budget=budget
            )
            source = canonical_identifier(source_id, "source_id")
            revision = canonical_identifier(source_revision, "source_revision")
            self.reference_domain = reference_domain
            self.reference_mesh = reference_mesh
            self.source_geometry = source_geometry
            self.cell_regions = regions
            self.certificate_limits = limits
            self.source_id = source
            self.source_revision = revision
            self.domain_id = canonical_fingerprint(
                {
                    "kind": "mapped-reference-domain",
                    "source": source,
                    "revision": revision,
                    "reference_domain": reference_domain.domain_id,
                    "reference_topology": reference_mesh.topology_id,
                    "reference_coordinates": array_tree_fingerprint(
                        np.asarray(reference_mesh.coordinates)
                    ),
                    "source_geometry": cell_geometry_validity_owner.cell_geometry_id(
                        source_geometry
                    ),
                    "cell_regions": array_tree_fingerprint(regions),
                    "certificate_limits": {
                        field.name: getattr(limits, field.name)
                        for field in fields(limits)
                    },
                }
            )

    def entity_set_id(self, dimension: int) -> str:
        """Scientific image identity of one measured reference entity set."""
        dimension = integer(dimension, "MappedReferenceDomain.dimension")
        if dimension < 0 or dimension > self.reference_mesh.topological_dimension:
            raise ValueError(
                "MappedReferenceDomain.dimension must name a reference entity set; got an out-of-range dimension"
            )
        entities = self.reference_mesh.entity_set(int(dimension))
        return canonical_fingerprint(
            {
                "kind": "mapped-reference-image-entity-set",
                "domain": self.domain_id,
                "reference_entity_set": entities.entity_set_id,
                "dimension": int(dimension),
            }
        )

    def image_entity_id(self, dimension: int, root_global_id: int) -> str:
        """Bind a physical image to a validated, explicitly declared source entity."""
        image_set = self.entity_set_id(dimension)
        root_global_id = integer(root_global_id, "MappedReferenceDomain.root_global_id")
        entities = self.reference_mesh.entity_set(int(dimension))
        if not np.any(np.asarray(entities.entity_ids, dtype=np.int64) == root_global_id):
            raise ValueError(
                "MappedReferenceDomain.root_global_id must identify a reference entity; got an absent scientific ID"
            )
        return canonical_fingerprint(
            {
                "kind": "mapped-reference-image-entity",
                "domain": self.domain_id,
                "reference_entity_set": entities.entity_set_id,
                "image_entity_set": image_set,
                "dimension": int(dimension),
                "root_global_id": int(root_global_id),
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return self.reference_domain.ambient_dimension

    @property
    def region_ids(self) -> tuple[str, ...]:
        return self.reference_domain.region_ids

    @property
    def facets(self) -> np.ndarray:
        """Declared reference-fragment rows; they are not planar physical facets."""
        return self.reference_domain.facets

    def exact_region_measures(
        self, *, budget: CoordinateEnclosureBudget | None = None
    ) -> tuple[Fraction, ...]:
        """Integrate the physical root Jacobians, including collapsed pyramids."""

        with embedding_state_owner.coordinate_certificate_request(
            self.certificate_limits, budget=budget
        ) as budget:
            elements, routes, _ = self.source_geometry.resolve(self.reference_mesh, budget=budget)
            values = self.source_geometry.source_coordinates(budget=budget)
            totals = [Fraction(0) for _ in self.region_ids]
            cursor = 0
            for block, element, route in zip(
                self.reference_mesh.blocks, elements, routes, strict=True
            ):
                for local in (
                    tuple(values[index] for index in row)
                    for row in np.asarray(route, dtype=np.int64)
                ):
                    polynomial = coordinate_enclosure_owner.coordinate_polynomials(
                        element, local, budget=budget
                    )
                    if polynomial is None:
                        raise ValueError(
                            "MappedReferenceDomain.source_geometry must resolve coordinate polynomials; got absent source expressions"
                        )
                    jacobian = tuple(
                        tuple(
                            coordinate_enclosure_owner.derivative(
                                value, axis, budget=budget
                            )
                            for axis in range(self.ambient_dimension)
                        )
                        for value in polynomial
                    )
                    totals[int(self.cell_regions[cursor])] += (
                        mapped_coverage_owner.integrate(
                            coordinate_enclosure_owner.determinant(
                                jacobian, budget=budget
                            ),
                            mapped_coverage_owner.map_domain(block.cell_kind),
                            self.ambient_dimension,
                            budget=budget,
                        )
                    )
                    cursor += 1
            return tuple(totals)
