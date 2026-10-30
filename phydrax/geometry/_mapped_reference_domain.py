#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound physical maps over an independently declared reference domain."""

from __future__ import annotations

from fractions import Fraction
from typing import final

import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, integer
from ..discretization._cell_geometry import CellGeometrySpec, ScalarCoordinateElement
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import coordinate_map_expressions, ExactScalar
from ..typing import Dim, HostInt64, parse
from ._domain_coverage import PiecewiseLinearDomain
from ._mapped_coverage import integrate_jacobian
from ._source_identity import source_entity_id


def _coordinate_route_matrix(value: ArrayLike, /) -> np.ndarray:
    rows = np.asarray(value, dtype=np.int64)
    if rows.ndim != 2:
        raise ValueError("Coordinate DOF routes must be a rank-two matrix.")
    return rows


class _MappedSourceCellDim(Dim, minimum=1):
    """Cells of one independently declared mapped source."""


@final
class MappedReferenceDomain(StrictModule, NonTrainableState):
    """An immutable view of root cells and their exact represented maps.

    Vertex and cell identities are the source mesh's explicit global IDs, not
    coordinate-derived or row-inferred names. Occurrence paths disambiguate
    repeated authored identities in placed source instances.
    The coordinate contract describes the physical output frame of source_geometry;
    reference_domain coordinates remain the declared mathematical chart.
    """

    __strict_contract__ = True

    reference_domain: PiecewiseLinearDomain
    reference_mesh: CellMesh
    source_geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    cell_regions: HostInt64[_MappedSourceCellDim]
    source_id: str
    source_revision: str
    domain_id: str
    root_vertex_ids: tuple[int, ...]
    root_vertex_occurrence_paths: tuple[tuple[str, ...], ...]
    root_cell_ids: tuple[int, ...]
    root_cell_occurrence_paths: tuple[tuple[str, ...], ...]
    reference_mesh_id: str
    reference_domain_vertex_root_ids: tuple[int, ...] | None
    reference_geometry_id: str

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
        coordinate_contract: SpatialCoordinateContract,
        root_vertex_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        root_cell_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        reference_domain_vertex_root_ids: ArrayLike | None = None,
    ) -> None:
        if not isinstance(reference_domain, PiecewiseLinearDomain):
            raise TypeError("reference_domain must be PiecewiseLinearDomain.")
        if not isinstance(reference_mesh, CellMesh):
            raise TypeError("reference_mesh must be CellMesh.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if coordinate_contract.coordinate_system != "cartesian":
            raise ValueError(
                "MappedReferenceDomain source_geometry must use Cartesian physical coordinates."
            )
        if (
            getattr(reference_mesh, "storage", None) is not None
            or getattr(source_geometry, "storage", None) is not None
        ):
            raise ValueError(
                "MappedReferenceDomain requires serial root mesh and geometry."
            )
        dimension = reference_mesh.topological_dimension
        if (
            dimension not in (2, 3)
            or dimension != reference_mesh.ambient_dimension
            or dimension != reference_domain.ambient_dimension
        ):
            raise ValueError(
                "MappedReferenceDomain requires matching two- or three-dimensional reference volume maps."
            )
        if source_geometry.coordinates.shape[1] != dimension:
            raise ValueError(
                "MappedReferenceDomain source geometry must have the reference volume dimension."
            )
        if source_geometry.restriction_source is not None:
            raise ValueError(
                "MappedReferenceDomain source geometry must retain unrestricted root maps."
            )
        if (
            not isinstance(reference_domain.domain_id, str)
            or not reference_domain.domain_id
        ):
            raise ValueError(
                "reference_domain must retain a canonical source-domain identity."
            )
        if (
            not isinstance(reference_domain.region_ids, tuple)
            or not reference_domain.region_ids
        ):
            raise ValueError(
                "reference_domain must declare at least one region identity."
            )
        region_values = np.asarray(cell_regions)
        if region_values.dtype.kind not in "iu":
            raise TypeError("cell_regions must contain integer region indices.")
        if region_values.dtype.kind == "u" and np.any(
            region_values > np.iinfo(np.int64).max
        ):
            raise ValueError("cell_regions contains values outside the int64 range.")
        regions = parse(
            region_values.astype(np.int64, copy=False),
            HostInt64[_MappedSourceCellDim],
            "cell_regions",
        )
        cell_count = sum(block.cell_count for block in reference_mesh.blocks)
        if (
            regions.shape != (cell_count,)
            or np.any(regions < 0)
            or np.any(regions >= len(reference_domain.region_ids))
        ):
            raise ValueError(
                "cell_regions must assign each root cell to one declared region."
            )
        region_copy = np.array(regions, dtype=np.int64, copy=True)
        region_copy.setflags(write=False)

        vertex_ids = tuple(
            int(value) for value in np.asarray(reference_mesh.vertex_global_ids)
        )
        if len(set(vertex_ids)) != len(vertex_ids) or any(
            value < 0 for value in vertex_ids
        ):
            raise ValueError(
                "reference mesh vertex global IDs must be unique nonnegative IDs."
            )
        cell_ids = tuple(
            int(value)
            for block in reference_mesh.blocks
            for value in np.asarray(block.global_ids, dtype=np.int64)
        )
        if len(set(cell_ids)) != len(cell_ids) or any(value < 0 for value in cell_ids):
            raise ValueError(
                "reference mesh cell global IDs must be unique nonnegative IDs."
            )

        vertex_paths = _occurrence_paths(
            root_vertex_occurrence_paths, len(vertex_ids), "root_vertex_occurrence_paths"
        )
        cell_paths = _occurrence_paths(
            root_cell_occurrence_paths, len(cell_ids), "root_cell_occurrence_paths"
        )
        reference_vertex_root_ids: tuple[int, ...] | None = None
        if reference_domain_vertex_root_ids is not None:
            raw_reference_root_ids = np.asarray(reference_domain_vertex_root_ids)
            if (
                raw_reference_root_ids.dtype.kind not in "iu"
                or raw_reference_root_ids.shape != (len(reference_domain.vertices),)
            ):
                raise TypeError(
                    "reference_domain_vertex_root_ids must contain one integer root global ID per reference-domain vertex."
                )
            if raw_reference_root_ids.dtype.kind == "u" and np.any(
                raw_reference_root_ids > np.iinfo(np.int64).max
            ):
                raise ValueError("Reference-domain root vertex IDs must fit int64.")
            reference_vertex_root_ids = tuple(
                int(value) for value in raw_reference_root_ids.tolist()
            )
            if len(set(reference_vertex_root_ids)) != len(reference_vertex_root_ids):
                raise ValueError(
                    "Reference-domain vertices must map bijectively to root vertex IDs."
                )
            root_row_by_id = {
                identifier: row for row, identifier in enumerate(vertex_ids)
            }
            try:
                root_rows = np.asarray(
                    [
                        root_row_by_id[identifier]
                        for identifier in reference_vertex_root_ids
                    ],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    "Reference-domain vertex relation names an absent root vertex ID."
                ) from error
            if not np.array_equal(
                np.asarray(reference_domain.vertices, dtype=np.float64),
                np.asarray(reference_mesh.coordinates, dtype=np.float64)[root_rows],
            ):
                raise ValueError(
                    "Declared reference-domain/root vertex correspondence does not match exact source coordinates."
                )
        source = canonical_identifier(source_id, "source_id")
        revision = canonical_identifier(source_revision, "source_revision")
        source_map = source_geometry.source_map
        if source_map is not None and (
            source_map.source_id != source or source_map.source_revision != revision
        ):
            raise ValueError(
                "source_geometry coordinate map must match the declared source identity."
            )
        supported = {
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "hexahedron",
            "prism",
            "pyramid",
        }
        elements, routes, _ = source_geometry.resolve(reference_mesh)
        coefficients = source_geometry.source_coordinates()
        for block, element, route in zip(
            reference_mesh.blocks, elements, routes, strict=True
        ):
            if block.cell_kind not in supported or element.cell_kind != block.cell_kind:
                raise ValueError(
                    "MappedReferenceDomain root cells must have supported matching reference maps."
                )
            if not isinstance(element, ScalarCoordinateElement):
                raise TypeError(
                    "MappedReferenceDomain requires scalar coordinate elements."
                )
            route_rows = _coordinate_route_matrix(route)
            if route_rows.shape[0] != block.cell_count:
                raise ValueError(
                    "MappedReferenceDomain root routes must align with every cell."
                )
            for row_index in range(route_rows.shape[0]):
                row = route_rows[row_index]
                local = tuple(coefficients[int(index)] for index in row)
                coordinate_map_expressions(element, local, cell_row=row_index)

        reference_mesh_id = canonical_fingerprint(
            {
                "kind": "mapped-reference-mesh",
                "topology": reference_mesh.topology_id,
                "coordinates": array_tree_fingerprint(
                    np.asarray(reference_mesh.coordinates)
                ),
                "vertex_ids": array_tree_fingerprint(
                    np.asarray(reference_mesh.vertex_global_ids)
                ),
                "cell_ids": cell_ids,
            }
        )
        reference_geometry_id = cell_geometry_id(source_geometry)
        identity_payload = {
            "kind": "mapped-reference-domain",
            "source": source,
            "revision": revision,
            "reference_domain": reference_domain.domain_id,
            "reference_mesh": reference_mesh_id,
            "reference_geometry": reference_geometry_id,
            "coordinate_contract": coordinate_contract.spatial_id,
            "regions": array_tree_fingerprint(region_copy),
            "vertex_occurrences": vertex_paths,
            "cell_occurrences": cell_paths,
        }
        if reference_vertex_root_ids is not None:
            identity_payload["reference_domain_vertex_root_ids"] = (
                reference_vertex_root_ids
            )
        identity = canonical_fingerprint(identity_payload)
        self.reference_domain = reference_domain
        self.reference_mesh = reference_mesh
        self.source_geometry = source_geometry
        self.coordinate_contract = coordinate_contract
        self.cell_regions = region_copy
        self.source_id = source
        self.source_revision = revision
        self.domain_id = identity
        self.root_vertex_ids = vertex_ids
        self.root_vertex_occurrence_paths = vertex_paths
        self.root_cell_ids = cell_ids
        self.root_cell_occurrence_paths = cell_paths
        self.reference_mesh_id = reference_mesh_id
        self.reference_domain_vertex_root_ids = reference_vertex_root_ids
        self.reference_geometry_id = reference_geometry_id

    @property
    def ambient_dimension(self) -> int:
        return self.reference_domain.ambient_dimension

    @property
    def region_ids(self) -> tuple[str, ...]:
        return self.reference_domain.region_ids

    @property
    def facets(self) -> np.ndarray:
        return self.reference_domain.facets

    def exact_region_measures(self, /) -> tuple[ExactScalar, ...]:
        """Integrate every declared root Jacobian in exact reference coordinates."""
        elements, routes, _ = self.source_geometry.resolve(self.reference_mesh)
        coefficients = self.source_geometry.source_coordinates()
        totals: list[ExactScalar] = [Fraction(0) for _ in self.region_ids]
        cell_index = 0
        for block, element, route in zip(
            self.reference_mesh.blocks, elements, routes, strict=True
        ):
            if not isinstance(element, ScalarCoordinateElement):
                raise TypeError(
                    "MappedReferenceDomain requires scalar coordinate elements."
                )
            route_rows = _coordinate_route_matrix(route)
            if route_rows.shape[0] != block.cell_count:
                raise ValueError(
                    "MappedReferenceDomain root routes must align with every cell."
                )
            for row_index in range(route_rows.shape[0]):
                row = route_rows[row_index]
                local = tuple(coefficients[int(index)] for index in row)
                coordinates = coordinate_map_expressions(
                    element, local, cell_row=row_index
                )
                measure = integrate_jacobian(coordinates, block.cell_kind)
                if measure <= 0:
                    raise ValueError(
                        "MappedReferenceDomain root maps must have positive oriented exact measure."
                    )
                region = int(self.cell_regions[cell_index])
                totals[region] += measure
                cell_index += 1
        return tuple(totals)

    def root_vertex_id(self, row: int, /) -> int:
        index = integer(row, "MappedReferenceDomain.vertex_row")
        if index < 0 or index >= len(self.root_vertex_ids):
            raise ValueError("vertex_row does not identify a reference root vertex.")
        return self.root_vertex_ids[index]

    def resolve_root_vertex(
        self,
        source_id: str,
        source_revision: str,
        entity_id: str | int,
        /,
        *,
        occurrence_path: tuple[str, ...] = (),
    ) -> int:
        """Resolve one authored source vertex to its unique reference mesh row."""
        source = canonical_identifier(source_id, "source_id")
        revision = canonical_identifier(source_revision, "source_revision")
        if (source, revision) != (self.source_id, self.source_revision):
            raise ValueError(
                "Source vertex identity belongs to a different source revision."
            )
        wanted = source_entity_id(
            source, revision, "vertex", entity_id, occurrence_path=occurrence_path
        )
        matches = tuple(
            row
            for row, (identifier, path) in enumerate(
                zip(self.root_vertex_ids, self.root_vertex_occurrence_paths, strict=True)
            )
            if source_entity_id(
                source, revision, "vertex", identifier, occurrence_path=path
            )
            == wanted
        )
        if not matches:
            raise KeyError(
                "Source vertex identity is absent from the reference root mesh."
            )
        if len(matches) != 1:
            raise ValueError(
                "Source vertex identity is ambiguous across reference occurrences."
            )
        return matches[0]

    def resolve_root_cell(
        self,
        source_id: str,
        source_revision: str,
        entity_id: str | int,
        /,
        *,
        occurrence_path: tuple[str, ...] = (),
    ) -> int:
        """Resolve one authored source cell to its unique reference mesh row."""
        source = canonical_identifier(source_id, "source_id")
        revision = canonical_identifier(source_revision, "source_revision")
        if (source, revision) != (self.source_id, self.source_revision):
            raise ValueError(
                "Source cell identity belongs to a different source revision."
            )
        wanted = source_entity_id(
            source, revision, "cell", entity_id, occurrence_path=occurrence_path
        )
        matches = tuple(
            row
            for row, (identifier, path) in enumerate(
                zip(self.root_cell_ids, self.root_cell_occurrence_paths, strict=True)
            )
            if source_entity_id(
                source, revision, "cell", identifier, occurrence_path=path
            )
            == wanted
        )
        if not matches:
            raise KeyError("Source cell identity is absent from the reference root mesh.")
        if len(matches) != 1:
            raise ValueError(
                "Source cell identity is ambiguous across reference occurrences."
            )
        return matches[0]

    def entity_set_id(self, dimension: int, /) -> str:
        selected = integer(dimension, "MappedReferenceDomain.dimension")
        if selected < 0 or selected > self.reference_mesh.topological_dimension:
            raise ValueError("dimension is outside the reference mesh topology.")
        entities = self.reference_mesh.entity_set(selected)
        return canonical_fingerprint(
            {
                "kind": "mapped-reference-entity-set",
                "domain": self.domain_id,
                "dimension": selected,
                "entity_set": entities.entity_set_id,
            }
        )

    def image_entity_id(self, dimension: int, root_global_id: int, /) -> str:
        selected = integer(dimension, "MappedReferenceDomain.dimension")
        identifier = integer(root_global_id, "MappedReferenceDomain.root_global_id")
        if selected < 0 or selected > self.reference_mesh.topological_dimension:
            raise ValueError("dimension is outside the reference mesh topology.")
        entities = self.reference_mesh.entity_set(selected)
        if not np.any(np.asarray(entities.entity_ids, dtype=np.int64) == identifier):
            raise ValueError(
                "root_global_id is absent from the declared reference entity set."
            )
        return canonical_fingerprint(
            {
                "kind": "mapped-reference-image-entity",
                "domain": self.domain_id,
                "entity_set": entities.entity_set_id,
                "dimension": selected,
                "root_global_id": identifier,
            }
        )


def _occurrence_paths(
    paths: tuple[tuple[str, ...], ...] | None, count: int, name: str, /
) -> tuple[tuple[str, ...], ...]:
    if paths is None:
        return tuple(() for _ in range(count))
    if len(paths) != count:
        raise ValueError(f"{name} must align with the reference entity inventory.")
    result: list[tuple[str, ...]] = []
    for path in paths:
        if isinstance(path, (str, bytes)):
            raise TypeError(f"{name} entries must be occurrence paths.")
        result.append(
            tuple(canonical_identifier(value, "occurrence_id") for value in path)
        )
    return tuple(result)


__all__ = ["MappedReferenceDomain"]
