#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Request-owned bindings for identical cells in successive certified carriers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import assert_never, Literal, TYPE_CHECKING

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._exact_algebraic_geometry import RadicalElement


if TYPE_CHECKING:
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ._cell_geometry import CellGeometrySpec
    from ._cell_geometry_validity import CellValidityCertificate
    from ._cell_mesh import CellMesh
    from ._coordinate_budget import CoordinateEnclosureBudget
    from ._coordinate_enclosure import ExactScalar


def cell_certificate_source_content_id(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    embedding: GlobalEmbeddingCertificate | None,
    /,
) -> str:
    """Bind certificate payloads and their numerical source, including bound arrays."""
    from ..lifecycle._meshing_source_records import stored_record_content_id
    from ._cell_geometry_validity import cell_geometry_id

    return canonical_fingerprint(
        {
            "kind": "cell-certificate-source-content",
            "mesh": (mesh.mesh_id, array_tree_fingerprint(mesh)),
            "geometry": (
                cell_geometry_id(geometry),
                stored_record_content_id(geometry, excluded=""),
            ),
            "validity": stored_record_content_id(validity, excluded=""),
            "embedding": None
            if embedding is None
            else stored_record_content_id(embedding, excluded=""),
        }
    )


@dataclass(frozen=True, slots=True)
class CellGeometryPredecessor:
    """Bind unchanged target rows to authenticated source-cell certificates."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate | None
    source_rows: tuple[int, ...]
    target_mesh: CellMesh
    target_geometry: CellGeometrySpec
    content_id: str

    def require_bound(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        """Refuse altered source certificates, target content, or row correspondence."""
        if self.content_id != _predecessor_content_id(
            self.source_mesh,
            self.source_geometry,
            self.source_validity,
            self.source_embedding,
            mesh,
            geometry,
            self.source_rows,
        ):
            raise ValueError(
                "CellGeometryPredecessor.content_id must bind its source certificates and target rows; got changed content"
            )
        self.source_validity.require_bound(self.source_geometry, mesh=self.source_mesh)
        if not self.source_validity.all_certified:
            raise ValueError(
                "CellGeometryPredecessor.source_validity must certify every imported source cell; got undecided source cells"
            )
        if self.source_embedding is not None:
            self.source_embedding.binding.require(self.source_mesh, self.source_geometry)
            if (
                self.source_embedding.validity_certificate_id
                != self.source_validity.certificate_id
            ):
                raise ValueError(
                    "CellGeometryPredecessor.source_embedding must bind its source validity certificate; got another validity ID"
                )


def _predecessor_content_id(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    source_embedding: GlobalEmbeddingCertificate | None,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_rows: tuple[int, ...],
    /,
) -> str:
    from ..lifecycle._meshing_source_records import stored_record_content_id
    from ._cell_geometry_validity import cell_geometry_id

    return canonical_fingerprint(
        {
            "kind": "cell-geometry-predecessor",
            "source": cell_certificate_source_content_id(
                source_mesh, source_geometry, source_validity, source_embedding
            ),
            "target": (target_mesh.mesh_id, array_tree_fingerprint(target_mesh)),
            "target_geometry": (
                cell_geometry_id(target_geometry),
                stored_record_content_id(target_geometry, excluded=""),
            ),
            "source_rows": source_rows,
        }
    )


def _coefficient_content_key(
    value: ExactScalar, /
) -> tuple[int, int] | tuple[Literal["radical"], tuple[tuple[int, int, int], ...]]:
    """Preserve rational identity tokens and encode reduced radical terms."""
    match value:
        case Fraction():
            return value.numerator, value.denominator
        case RadicalElement():
            return (
                "radical",
                tuple(
                    (radicand, coefficient.numerator, coefficient.denominator)
                    for radicand, coefficient in value.terms
                ),
            )
        case _:
            assert_never(value)


def _cell_content_keys(
    mesh: CellMesh, geometry: CellGeometrySpec, budget: CoordinateEnclosureBudget, /
) -> tuple[tuple[int, str], ...]:
    """Bind each cell's source map, ordered incidence, and carrier bytes."""
    from ._coordinate_enclosure import (
        coordinate_source_signature,
        prepared_coordinate_source_bank,
    )

    bank = prepared_coordinate_source_bank(geometry, budget=budget)
    elements, routes, _ = geometry.resolve(
        mesh, exact_source_prepared=True, budget=budget
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    binding = (
        None if geometry.exact_source is None else geometry.exact_source.source_id,
        None if geometry.periodic_source is None else geometry.periodic_source.source_id,
    )
    keys: list[tuple[int, str]] = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        definition = canonical_fingerprint(
            (coordinate_source_signature(element), array_tree_fingerprint(element))
        )
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        vertex_rows = np.asarray(block.vertices, dtype=np.int64)
        dof_rows = np.asarray(route, dtype=np.int64)
        for row in range(block.cell_count):
            vertices = vertex_rows[row : row + 1].reshape(-1)
            dofs = dof_rows[row : row + 1].reshape(-1)
            local = tuple(bank[int(dofs[index])] for index in range(dofs.size))
            budget.reserve(vertices.size * (mesh.ambient_dimension + 1) + len(local))
            keys.append(
                (
                    int(identifiers[row]),
                    canonical_fingerprint(
                        {
                            "kind": block.cell_kind,
                            "vertices": vertex_ids[vertices],
                            "carrier": coordinates[vertices],
                            "element": definition,
                            "source_binding": binding,
                            "coefficients": tuple(
                                tuple(_coefficient_content_key(value) for value in row)
                                for row in local
                            ),
                        }
                    ),
                )
            )
    return tuple(keys)


def prepare_cell_geometry_predecessor(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    source_embedding: GlobalEmbeddingCertificate | None,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CellGeometryPredecessor:
    """Authenticate content correspondence without rebuilding cell polynomials."""
    source_validity.require_bound(source_geometry, mesh=source_mesh)
    if not source_validity.all_certified:
        raise ValueError(
            "CellGeometryPredecessor.source_validity must certify every imported source cell; got undecided source cells"
        )
    source = {
        key: row
        for row, key in enumerate(
            _cell_content_keys(source_mesh, source_geometry, budget)
        )
    }
    rows = tuple(
        source.get(key, -1)
        for key in _cell_content_keys(target_mesh, target_geometry, budget)
    )
    predecessor = CellGeometryPredecessor(
        source_mesh,
        source_geometry,
        source_validity,
        source_embedding,
        rows,
        target_mesh,
        target_geometry,
        _predecessor_content_id(
            source_mesh,
            source_geometry,
            source_validity,
            source_embedding,
            target_mesh,
            target_geometry,
            rows,
        ),
    )
    predecessor.require_bound(target_mesh, target_geometry)
    budget.retain_basis((tuple(source), rows))
    budget.retain_owner(predecessor)
    return predecessor
