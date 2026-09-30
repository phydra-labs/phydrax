"""Canonical triangle realizations used by graph and operator scenarios."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from phydrax.discretization import (
    CochainDiscretization,
    polygonal_cell_complex,
    reorient_cell_complex,
    simplicial_dual_hodges,
)
from phydrax.discretization._cell_complex import simplicial_cell_geometry
from phydrax.graph import CochainComplexIR


def triangle_cochain_lowering(
    vertices: ArrayLike, faces: ArrayLike, /
) -> CochainComplexIR:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int32)
    topology = polygonal_cell_complex(triangles, None, points.shape[0])
    cells, _ = simplicial_cell_geometry(topology)
    coordinates = tuple(jnp.asarray(np.mean(points[cell], axis=1)) for cell in cells)
    edge_vectors = points[cells[1][:, 1]] - points[cells[1][:, 0]]
    face_vectors = points[cells[2][:, 1:]] - points[cells[2][:, :1]]
    gram = face_vectors @ np.swapaxes(face_vectors, -1, -2)
    primal = (
        jnp.ones((cells[0].shape[0],), dtype=jnp.float64),
        jnp.asarray(np.linalg.norm(edge_vectors, axis=-1)),
        jnp.asarray(np.sqrt(np.linalg.det(gram)) / 2.0),
    )
    hodges = simplicial_dual_hodges(topology, points, dual="barycentric")
    boundary_edges = (
        np.asarray(abs(topology.incidences[1].scipy_boundary()).sum(axis=1)).reshape(-1)
        == 1
    )
    boundary_vertices = np.zeros((points.shape[0],), dtype=np.bool_)
    boundary_vertices[cells[1][boundary_edges].reshape(-1)] = True
    realization = CochainDiscretization(
        topology,
        hodges,
        coordinates=coordinates,
        boundary_masks=(
            boundary_vertices,
            boundary_edges,
            np.zeros((triangles.shape[0],), dtype=np.bool_),
        ),
        primal_measures=primal,
        dual_measures=tuple(
            hodge.weights * measure for hodge, measure in zip(hodges, primal, strict=True)
        ),
    )
    return CochainComplexIR(realization)


def reoriented_lowering(
    lowering: CochainComplexIR, signs: Sequence[ArrayLike], /
) -> CochainComplexIR:
    realization = lowering.discretization
    topology = reorient_cell_complex(realization.topology, signs)
    return CochainComplexIR(
        CochainDiscretization(
            topology,
            realization.hodges,
            coordinates=realization.coordinates,
            boundary_masks=realization.boundary_masks,
            primal_measures=realization.primal_measures,
            dual_measures=realization.dual_measures,
        ),
        boundary=lowering.boundary,
    )
