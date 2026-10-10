#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Independent mapped-volume declarations with authoritative reference charts."""

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
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..typing import Dim, HostInt64, parse
from ._mesh_certificates import PiecewiseLinearDomain


def mapped_source_corner_coordinates(
    reference_mesh: CellMesh, source_geometry: CellGeometrySpec
) -> np.ndarray:
    """Validate the exact physical corner view shared by independent root maps."""
    from ..discretization import _coordinate_enclosure as algebra

    elements, routes, _ = source_geometry.resolve(reference_mesh)
    values = source_geometry.source_coordinates()
    ids = np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64)
    positions: dict[int, tuple[Fraction, ...]] = {}
    for block, element, route in zip(
        reference_mesh.blocks, elements, routes, strict=True
    ):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for local, row in zip(local_values, np.asarray(block.vertices), strict=True):
            images = algebra.coordinate_corner_images(element, local)
            if images is None:
                raise ValueError(
                    "Mapped source roots require authoritative exact coordinate expressions."
                )
            for vertex, image in zip(ids[row], images, strict=True):
                if int(vertex) in positions and positions[int(vertex)] != image:
                    raise ValueError(
                        "Mapped source roots disagree at a declared shared corner."
                    )
                positions[int(vertex)] = image
    if any(int(vertex) not in positions for vertex in ids):
        raise ValueError(
            "Mapped reference mesh contains vertices without an authoritative source chart."
        )
    return np.asarray(
        tuple(tuple(float(value) for value in positions[int(vertex)]) for vertex in ids),
        dtype=np.float64,
    )


class _MappedSourceCellDim(Dim, minimum=1):
    """Cells of an independently declared mapped source."""


@final
class MappedReferenceDomain(StrictModule, NonTrainableState):
    """A declared reference domain and its independently owned physical maps.

    Source identity is explicit and distinct from any target approximation.
    Coverage requires exact root restrictions, independent reference coverage,
    source physical embedding and target physical embedding. It does not imply
    equivalence of these physical maps to a CAD or implicit source.
    """

    __strict_contract__ = True

    reference_domain: PiecewiseLinearDomain
    reference_mesh: CellMesh
    source_geometry: CellGeometrySpec
    cell_regions: HostInt64[_MappedSourceCellDim]
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_domain: PiecewiseLinearDomain,
        reference_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        cell_regions: ArrayLike,
        /,
        *,
        source_id: str,
        source_revision: str,
    ) -> None:
        from ..discretization import _coordinate_enclosure as algebra
        from ..discretization._cell_geometry import CellGeometrySpec
        from ..discretization._cell_geometry_validity import cell_geometry_id
        from ..discretization._cell_mesh import CellMesh
        from ._mesh_certificates import PiecewiseLinearDomain

        if not isinstance(reference_domain, PiecewiseLinearDomain):
            raise TypeError("reference_domain must be PiecewiseLinearDomain.")
        if not isinstance(reference_mesh, CellMesh):
            raise TypeError("reference_mesh must be CellMesh.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec.")
        if reference_mesh.storage is not None or source_geometry.storage is not None:
            raise ValueError(
                "Mapped source declarations require independent global reference charts."
            )
        dimension = reference_mesh.topological_dimension
        if (
            dimension not in (2, 3)
            or dimension != reference_mesh.ambient_dimension
            or dimension != reference_domain.ambient_dimension
        ):
            raise ValueError(
                "Mapped source reference charts must be two- or three-dimensional volumes."
            )
        if source_geometry.coordinates.shape[1] != dimension:
            raise ValueError(
                "Mapped physical coordinates must have the declared volume dimension."
            )
        if source_geometry.restriction_source is not None:
            raise ValueError(
                "Mapped source declarations must own their roots independently of target restrictions."
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
                "Mapped source cell regions must assign every declared root."
            )
        elements, routes, _ = source_geometry.resolve(reference_mesh)
        values = source_geometry.source_coordinates()
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
                    "Mapped source roots require corresponding standard reference cells."
                )
            for local in (
                tuple(values[index] for index in row)
                for row in np.asarray(route, dtype=np.int64)
            ):
                if algebra.coordinate_polynomials(element, local) is None:
                    raise ValueError(
                        "Mapped source roots require authoritative exact coordinate expressions."
                    )
        mapped_source_corner_coordinates(reference_mesh, source_geometry)
        source = canonical_identifier(source_id, "source_id")
        revision = canonical_identifier(source_revision, "source_revision")
        self.reference_domain = reference_domain
        self.reference_mesh = reference_mesh
        self.source_geometry = source_geometry
        self.cell_regions = regions
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
                "source_geometry": cell_geometry_id(source_geometry),
                "cell_regions": array_tree_fingerprint(regions),
            }
        )

    def entity_set_id(self, dimension: int) -> str:
        """Scientific image identity of one actual reference entity set."""
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise TypeError("dimension must be an integer.")
        if dimension < 0 or dimension > self.reference_mesh.topological_dimension:
            raise ValueError("dimension must name an actual reference entity set.")
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
        if isinstance(root_global_id, bool) or not isinstance(
            root_global_id, (int, np.integer)
        ):
            raise TypeError("root_global_id must be an integer.")
        entities = self.reference_mesh.entity_set(int(dimension))
        if not np.any(np.asarray(entities.entity_ids, dtype=np.int64) == root_global_id):
            raise ValueError("root_global_id must identify an actual reference entity.")
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
        """Authoritative reference-fragment rows; they are not planar physical facets."""
        return self.reference_domain.facets

    def exact_region_measures(self) -> tuple[Fraction, ...]:
        """Integrate the physical root Jacobians, including collapsed pyramids."""
        from ..discretization import _coordinate_enclosure as algebra
        from ._mapped_coverage import integrate, map_domain

        elements, routes, _ = self.source_geometry.resolve(self.reference_mesh)
        values = self.source_geometry.source_coordinates()
        totals = [Fraction(0) for _ in self.region_ids]
        cursor = 0
        for block, element, route in zip(
            self.reference_mesh.blocks, elements, routes, strict=True
        ):
            for local in (
                tuple(values[index] for index in row)
                for row in np.asarray(route, dtype=np.int64)
            ):
                polynomial = algebra.coordinate_polynomials(element, local)
                if polynomial is None:
                    raise ValueError("Mapped source coordinate expression is absent.")
                jacobian = tuple(
                    tuple(
                        algebra.derivative(value, axis)
                        for axis in range(self.ambient_dimension)
                    )
                    for value in polynomial
                )
                totals[int(self.cell_regions[cursor])] += integrate(
                    algebra.determinant(jacobian),
                    map_domain(block.cell_kind),
                    self.ambient_dimension,
                )
                cursor += 1
        return tuple(totals)
