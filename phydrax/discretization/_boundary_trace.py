#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._cell_complex import PolyhedralConnectivity, TetrahedralConnectivity
from .finite_volume._unstructured import UnstructuredFiniteVolumeDiscretization


class BoundarySurfaceTrace(StrictModule):
    """Fixed boundary-face surface cells and exact conservative parent routing.

    Values on a surface cell are values on its parent volume boundary face.
    Integrated rates need no area rescaling. Positive exchange is outward from
    the volume and inward to the surface. Geometry is prepared on the host.
    """

    parent_faces: Array
    parent_cells: Array
    areas: Array
    centers: Array
    normals: Array
    edge_cells: Array
    edge_vertices: Array
    vertices: Array
    volume_cell_count: int = eqx.field(static=True)
    volume_face_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)

    def __init__(
        self, discretization: UnstructuredFiniteVolumeDiscretization, faces: ArrayLike
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Boundary trace requires prepared unstructured FV geometry.")
        if discretization.cell_dimension != 3:
            raise ValueError(
                "Boundary surface trace requires a three-dimensional volume."
            )
        selected = np.asarray(faces)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("Parent faces must be a one-dimensional integer array.")
        count = discretization.face_measures.size
        if (
            selected.size == 0
            or np.any(selected < 0)
            or np.any(selected >= count)
            or np.unique(selected).size != selected.size
        ):
            raise ValueError(
                "Boundary face selection must be nonempty, unique and in range."
            )
        neighbours = np.asarray(discretization.neighbour_cells)
        if np.any(neighbours[selected] >= 0):
            raise ValueError("A surface trace cannot include interior volume faces.")
        connectivity = discretization.connectivity
        if isinstance(connectivity, PolyhedralConnectivity):
            offsets = np.asarray(connectivity.face_vertex_offsets)
            values = np.asarray(connectivity.face_vertex_values)
            polygons = [values[offsets[f] : offsets[f + 1]] for f in selected]
        elif isinstance(connectivity, TetrahedralConnectivity):
            polygons = np.asarray(connectivity.faces)[selected]
        else:
            raise TypeError("Unsupported volume face connectivity.")
        incidence: dict[tuple[int, int], list[int]] = {}
        for cell, polygon in enumerate(polygons):
            for a, b in zip(polygon, np.roll(polygon, -1), strict=True):
                key: tuple[int, int] = (min(int(a), int(b)), max(int(a), int(b)))
                if key not in incidence:
                    incidence[key] = []
                incidence[key].append(cell)
        if any(len(cells) > 2 for cells in incidence.values()):
            raise ValueError("Selected surface is nonmanifold along an edge.")
        edges = sorted(incidence)
        self.parent_faces = jnp.asarray(selected, dtype=jnp.int32)
        self.parent_cells = discretization.owner_cells[self.parent_faces]
        self.areas = discretization.face_measures[self.parent_faces]
        self.centers = discretization.face_centers[self.parent_faces]
        self.normals = (
            discretization.area_vectors[self.parent_faces] / self.areas[:, None]
        )
        self.edge_cells = jnp.asarray(
            [
                (incidence[e][0], incidence[e][1] if len(incidence[e]) == 2 else -1)
                for e in edges
            ],
            dtype=jnp.int32,
        )
        self.edge_vertices = jnp.asarray(edges, dtype=jnp.int32)
        self.vertices = discretization.vertices
        self.volume_cell_count = discretization.cell_volumes.size
        self.volume_face_count = count
        self.geometry_id = discretization.geometry_id
        self.trace_id = canonical_fingerprint(
            {
                "kind": "boundary-surface-trace",
                "geometry": self.geometry_id,
                "faces": selected.tolist(),
            }
        )

    def require_geometry(self, discretization: UnstructuredFiniteVolumeDiscretization):
        if discretization.geometry_id != self.geometry_id:
            raise ValueError("Boundary trace is bound to different volume geometry.")

    def gather(self, volume_face_values: ArrayLike) -> Array:
        values = jnp.asarray(volume_face_values)
        if values.ndim == 0 or values.shape[0] != self.volume_face_count:
            raise ValueError("Face values do not match the bound volume geometry.")
        return values[self.parent_faces]

    def scatter(self, surface_rates: ArrayLike) -> Array:
        rates = jnp.asarray(surface_rates)
        if rates.ndim == 0 or rates.shape[0] != self.parent_faces.size:
            raise ValueError("Surface rates do not match the trace cells.")
        return (
            jnp.zeros((self.volume_face_count,) + rates.shape[1:], rates.dtype)
            .at[self.parent_faces]
            .set(rates)
        )

    def volume_content_rate(self, surface_rates: ArrayLike) -> Array:
        """Negative volume inventory rate for positive outward exchange."""
        rates = jnp.asarray(surface_rates)
        if rates.ndim == 0 or rates.shape[0] != self.parent_faces.size:
            raise ValueError("Surface rates do not match the trace cells.")
        return (
            jnp.zeros((self.volume_cell_count,) + rates.shape[1:], rates.dtype)
            .at[self.parent_cells]
            .add(-rates)
        )


__all__ = ["BoundarySurfaceTrace"]
