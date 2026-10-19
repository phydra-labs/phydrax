# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import final

import numpy as np

from ..._strict import StrictModule
from ...discretization import PolyhedralConnectivity
from ...geometry._embedding_state import coordinate_certificate_request
from ...geometry._exact_polyhedral_geometry import determinant3, star_tetrahedra
from ...typing import checked
from .._result import CellMeshingResult
from ._native_lifecycle import QualificationFailureFactory


@final
class PeriodicSourceQuadrature(StrictModule):
    points: np.ndarray
    cell_vertices: tuple[np.ndarray, ...]
    cell_points: tuple[np.ndarray, ...]
    cell_weights: tuple[np.ndarray, ...]


@checked
def prepare_periodic_source_quadrature(
    result: CellMeshingResult,
    /,
    *,
    failure_factory: QualificationFailureFactory,
) -> PeriodicSourceQuadrature:
    """Integrate degree-four source polynomials with oriented star tetrahedra."""
    mesh, geometry = (result.mesh, result.geometry)
    if geometry is None or mesh.periodic_topology is None:
        raise ValueError(
            "qualification.polyhedral.result must provide geometry and periodic topology; "
            f"got geometry={type(geometry).__name__}, "
            f"periodic_topology={type(mesh.periodic_topology).__name__}"
        )
    certification = result.certification
    if certification is None:
        raise ValueError(
            "qualification.polyhedral_certificate must retain request limits; got None"
        )
    with coordinate_certificate_request(certification.request.limits) as budget:
        exact = geometry.source_coordinates(budget=budget)
        points = np.asarray(exact, dtype=np.float64)
        connectivity = mesh.connectivity
        match connectivity:
            case PolyhedralConnectivity():
                pass
            case _:
                raise TypeError(
                    "qualification.polyhedral.connectivity must retain polyhedral cells; "
                    f"got {type(connectivity).__name__}"
                )
        exact_points = np.asarray(exact, dtype=object)
        tetrahedra = star_tetrahedra(
            mesh, exact_points, require_positive=False, budget=budget
        )
        vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
        vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
        nodes, weights = np.polynomial.legendre.leggauss(4)
        nodes, weights = ((nodes + 1.0) / 2.0, weights / 2.0)
        r, s, t = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        wr, ws, wt = np.meshgrid(weights, weights, weights, indexing="ij")
        r, s, t = (r.ravel(), s.ravel(), t.ravel())
        reference_weights = (wr * ws * wt).ravel() * (1 - r) ** 2 * (1 - s)
        barycentric = np.stack(
            (
                1 - r - (1 - r) * s - (1 - r) * (1 - s) * t,
                r,
                (1 - r) * s,
                (1 - r) * (1 - s) * t,
            ),
            axis=1,
        )
        cell_points, cell_weights, cell_vertices = ([], [], [])
        for cell in range(connectivity.cell_count):
            ids = vertices[vertex_offsets[cell] : vertex_offsets[cell + 1]]
            local_points, local_weights = ([], [])
            for tetrahedron in tetrahedra[cell]:
                center, *corners_exact = tetrahedron
                columns = tuple(
                    (
                        tuple(
                            (
                                value - base
                                for value, base in zip(corner, center, strict=True)
                            )
                        )
                        for corner in corners_exact
                    )
                )
                determinant = float(determinant3(*columns, budget=budget))
                if not np.isfinite(determinant) or determinant <= 0:
                    raise failure_factory(
                        "_polyhedral_qualification_source_quadrature",
                        "Source cell is not positively star-visible under source-face orientation.",
                        {"determinant": determinant},
                    )
                corners = np.asarray(tetrahedron, dtype=np.float64)
                local_points.append(barycentric @ corners)
                local_weights.append(reference_weights * determinant)
            cell_points.append(np.concatenate(local_points))
            cell_weights.append(np.concatenate(local_weights))
            cell_vertices.append(ids)
        return PeriodicSourceQuadrature(
            points=points,
            cell_vertices=tuple(cell_vertices),
            cell_points=tuple(cell_points),
            cell_weights=tuple(cell_weights),
        )
