#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compatible cochain realization on cut complexes."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import PolyhedralConnectivity
from .._cochain import CochainDiscretization
from .._cochain_hodge import DiagonalHodge
from .._core import DiscretizationKey, DiscretizationRole
from .._topology import CellComplexTopology
from ._cut_complex import MultivaluedCutCellComplex


@final
class CutCellCochainPlan(StrictModule, NonTrainableState):
    """Prepare node/edge/face/cell metrics from a multivalued polyhedral shell."""

    complex: MultivaluedCutCellComplex
    topology: CellComplexTopology
    hodges: tuple[DiagonalHodge, ...]
    coordinates: tuple[Array, ...]
    primal_measures: tuple[Array, ...]
    dual_measures: tuple[Array, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, complex_: MultivaluedCutCellComplex, /) -> None:
        if not isinstance(complex_, MultivaluedCutCellComplex):
            raise TypeError("Cut-cell cochains require MultivaluedCutCellComplex.")
        topology = complex_.mesh.topology
        coordinates, primal, dual = _cut_metric_data(complex_)
        hodges = tuple(
            DiagonalHodge(dual_value / primal_value).admit()
            for dual_value, primal_value in zip(dual, primal, strict=True)
        )
        self.complex = complex_
        self.topology = topology
        self.hodges = hodges
        self.coordinates = coordinates
        self.primal_measures = primal
        self.dual_measures = dual
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cut-cell-cochain-plan",
                "complex": complex_.topology_id,
                "geometry": complex_.geometry_id,
            }
        )

    def prepare(
        self,
        /,
        *,
        time: ArrayLike = 0.0,
        numeric_revision: str | None = None,
    ) -> CochainDiscretization:
        return CochainDiscretization(
            self.topology,
            self.hodges,
            boundary_masks=self.complex.mesh.boundary_masks,
            coordinates=self.coordinates,
            primal_measures=self.primal_measures,
            dual_measures=self.dual_measures,
            key=DiscretizationKey(
                "cochain", DiscretizationRole.PHYSICAL, domain_labels=(self.plan_id,)
            ),
            numeric_revision=self.plan_id
            if numeric_revision is None
            else numeric_revision,
            time=time,
        )


def _cut_metric_data(
    complex_: MultivaluedCutCellComplex, /
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    geometry = complex_.finite_volume_plan().prepare(numeric_version="cut-cell-cochain")
    connectivity = complex_.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise RuntimeError("Cut-cell complexes must have polyhedral connectivity.")
    coordinates = np.asarray(complex_.mesh.coordinates, dtype=np.float64)
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    edge_centers = np.mean(coordinates[edges], axis=1)
    edge_measures = np.linalg.norm(
        coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=-1
    )
    face_centers = np.asarray(geometry.face_centers, dtype=np.float64)
    face_measures = np.asarray(geometry.face_measures, dtype=np.float64)
    cell_centers = np.asarray(geometry.cell_centers, dtype=np.float64)
    cell_volumes = np.asarray(geometry.cell_volumes, dtype=np.float64)
    primal = (
        np.ones((coordinates.shape[0],), dtype=np.float64),
        edge_measures,
        face_measures,
        cell_volumes,
    )
    dual = [np.zeros_like(value) for value in primal]
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int32)
    cell_face_values = np.asarray(connectivity.cell_face_values, dtype=np.int32)
    cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
    cell_vertex_values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
    face_edge_offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int32)
    face_edge_values = np.asarray(connectivity.face_edge_values, dtype=np.int32)
    for cell in range(connectivity.cell_count):
        faces = cell_face_values[cell_face_offsets[cell] : cell_face_offsets[cell + 1]]
        vertices = cell_vertex_values[
            cell_vertex_offsets[cell] : cell_vertex_offsets[cell + 1]
        ]
        edge_set = {
            int(edge)
            for face in faces
            for edge in face_edge_values[
                face_edge_offsets[face] : face_edge_offsets[face + 1]
            ]
        }
        volume = cell_volumes[cell]
        dual[0][vertices] += volume / len(vertices)
        dual[1][np.asarray(sorted(edge_set), dtype=np.int32)] += volume / len(edge_set)
        dual[2][faces] += volume / len(faces)
        dual[3][cell] = 1.0
    if any(np.any(~np.isfinite(value) | (value <= 0.0)) for value in (*primal, *dual)):
        raise ValueError("Cut-cell cochain primal/dual metrics must be positive.")
    return (
        tuple(
            jnp.asarray(value)
            for value in (coordinates, edge_centers, face_centers, cell_centers)
        ),
        tuple(jnp.asarray(value) for value in primal),
        tuple(jnp.asarray(value) for value in dual),
    )


__all__ = ["CutCellCochainPlan"]
