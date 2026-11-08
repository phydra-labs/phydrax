#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact affine cell geometry for an index-space PLC mesh."""

from __future__ import annotations

from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization.fem._reference import lagrange_element
from ..linalg._exact import determinant_exact
from ._affine_plc_domain import AffinePlcDomain
from ._planar_coverage import _source_point, PlcSourcePointWitness
from ._rational import rational_vector


@final
class AffinePlcGeometrySource(StrictModule, NonTrainableState):
    """Index mesh and original affine PLC binding an exact coordinate bank."""

    domain: AffinePlcDomain
    reference_mesh: CellMesh
    witness: PlcSourcePointWitness | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: AffinePlcDomain,
        reference_mesh: CellMesh,
        /,
        *,
        witness: PlcSourcePointWitness | None = None,
    ) -> None:
        if type(domain) is not AffinePlcDomain or not isinstance(
            reference_mesh, CellMesh
        ):
            raise TypeError("An affine PLC domain and index-space mesh are required.")
        if reference_mesh.ambient_dimension != domain.ambient_dimension:
            raise ValueError("Index mesh and affine PLC dimensions must agree.")
        if witness is not None:
            if not isinstance(witness, PlcSourcePointWitness):
                raise TypeError("witness must be PLC source-point evidence.")
            if (
                witness.domain_id != domain.domain_id
                or not np.array_equal(
                    witness.vertex_global_ids, reference_mesh.vertex_global_ids
                )
                or domain.ambient_dimension != 3
                or not np.array_equal(
                    witness.faces[witness.domain_face_rows],
                    domain.reference_domain.facets,
                )
                or np.any(witness.faces >= domain.vertices.shape[0])
                or np.any(witness.segments >= domain.vertices.shape[0])
                or np.any(witness.vertex_kinds == 3)
                or np.any(np.isfinite(witness.face_planes))
            ):
                raise ValueError(
                    "PLC source-point evidence does not bind this reference mesh and domain."
                )
        self.domain = domain
        self.reference_mesh = reference_mesh
        self.witness = witness
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "affine-plc-cell-geometry",
                "domain": domain.domain_id,
                "reference_mesh": reference_mesh.mesh_id,
                "witness": None if witness is None else witness.witness_id,
            }
        )

    def source_coordinates(self, /) -> tuple[tuple[Fraction, ...], ...]:
        points = tuple(
            rational_vector(row)
            for row in np.asarray(self.reference_mesh.coordinates, dtype=np.float64)
        )
        if self.witness is not None:
            vertices = tuple(
                rational_vector(row) for row in self.domain.reference_domain.vertices
            )
            selected = list(points)
            for row, kind in enumerate(self.witness.vertex_kinds.tolist()):
                if kind < 0:
                    continue
                exact = _source_point(self.witness, row, vertices)
                if tuple(float(value) for value in exact) != tuple(
                    float(value) for value in points[row]
                ):
                    raise ValueError(
                        "Reference mesh carrier does not round its exact PLC source point."
                    )
                selected[row] = exact
            points = tuple(selected)
        return tuple(self.domain.transform_point(point) for point in points)


def map_affine_plc_mesh(
    domain: AffinePlcDomain,
    reference_mesh: CellMesh,
    /,
    *,
    witness: PlcSourcePointWitness | None = None,
) -> tuple[CellMesh, CellGeometrySpec]:
    """Map positive index simplices once, retaining exact rational coordinate values.

    Negative-determinant maps reverse each simplex's connectivity. Global vertex
    and cell IDs are preserved; no source roots or material labels are inferred.
    """
    source = AffinePlcGeometrySource(domain, reference_mesh, witness=witness)
    dimension = domain.ambient_dimension
    kind = "triangle" if dimension == 2 else "tetrahedron"
    if reference_mesh.periodic_topology is not None or any(
        block.cell_kind not in (kind, f"simplex:{dimension}")
        for block in reference_mesh.blocks
    ):
        raise ValueError("Affine PLC geometry requires nonperiodic simplex blocks.")
    exact = source.source_coordinates()
    blocks = []
    for block in reference_mesh.blocks:
        cells = np.array(block.vertices, dtype=np.int64, copy=True)
        for cell in cells:
            points = tuple(exact[int(index)] for index in cell)
            jacobian = tuple(
                tuple(
                    points[column + 1][axis] - points[0][axis]
                    for column in range(dimension)
                )
                for axis in range(dimension)
            )
            if determinant_exact(jacobian) * domain.determinant <= 0:
                raise ValueError("Index simplices must have positive exact orientation.")
        if domain.determinant < 0:
            cells[:, [0, 1]] = cells[:, [1, 0]]
        blocks.append(
            CellBlock(block.name, block.cell_kind, cells, global_ids=block.global_ids)
        )
    try:
        coordinates = np.asarray(exact, dtype=np.float64)
    except OverflowError as error:
        raise ValueError(
            "Affine mesh coordinates must have finite binary64 carriers."
        ) from error
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Affine mesh coordinates must have finite binary64 carriers.")
    mesh = CellMesh(
        coordinates,
        tuple(blocks),
        vertex_global_ids=reference_mesh.vertex_global_ids,
        numeric_version=reference_mesh.numeric_version,
    )
    geometry = CellGeometrySpec(
        {block.name: lagrange_element(block.cell_kind, 1) for block in mesh.blocks},
        {block.name: np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks},
        coordinates,
        source_map=source,
        source_mesh=mesh,
    )
    return mesh, geometry


__all__ = ["AffinePlcGeometrySource", "map_affine_plc_mesh"]
