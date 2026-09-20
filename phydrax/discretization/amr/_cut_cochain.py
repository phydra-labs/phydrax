#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compatible cochain topology and positive metric state on cut complexes."""

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cochain_metrics import (
    CochainMetricPlan,
    CochainMetricState,
    PreparedCochainTopology,
)
from ._cut_complex import MultivaluedCutCellComplex


class CutCellCochainState(StrictModule):
    """Cut-complex incidence and one accepted mass-lumped metric state."""

    topology: PreparedCochainTopology
    metrics: CochainMetricState
    complex_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


class CutCellCochainPlan(StrictModule, NonTrainableState):
    """Prepare node/edge/face/cell metrics from a multivalued polyhedral shell."""

    complex: MultivaluedCutCellComplex
    topology: PreparedCochainTopology
    metric_plan: CochainMetricPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, complex_: MultivaluedCutCellComplex, /):
        if not isinstance(complex_, MultivaluedCutCellComplex):
            raise TypeError("Cut-cell cochains require MultivaluedCutCellComplex.")
        topology = PreparedCochainTopology(complex_.mesh.topology)
        counts = tuple(entity.count for entity in complex_.mesh.topology.entity_sets)
        coordinate_shapes = tuple(
            (count, complex_.hierarchy.dimension) for count in counts
        )
        metric_plan = CochainMetricPlan(
            topology,
            geometry_family_id=complex_.body_set_id,
            geometry_layout_id=complex_.topology_id,
            coordinate_shapes=coordinate_shapes,
        )
        self.complex = complex_
        self.topology = topology
        self.metric_plan = metric_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cut-cell-cochain-plan",
                "complex": complex_.topology_id,
                "geometry": complex_.geometry_id,
                "metric_layout": metric_plan.metric_layout_id,
            }
        )

    def prepare(
        self,
        /,
        *,
        time=0.0,
        revision=0,
    ) -> CutCellCochainState:
        geometry = self.complex.finite_volume_plan().prepare(
            numeric_version="cut-cell-cochain"
        )
        connectivity = self.complex.mesh.connectivity
        coordinates = np.asarray(self.complex.mesh.coordinates, dtype=np.float64)
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
            faces = cell_face_values[
                int(cell_face_offsets[cell]) : int(cell_face_offsets[cell + 1])
            ]
            vertices = cell_vertex_values[
                int(cell_vertex_offsets[cell]) : int(cell_vertex_offsets[cell + 1])
            ]
            edge_set = {
                int(edge)
                for face in faces
                for edge in face_edge_values[
                    int(face_edge_offsets[face]) : int(face_edge_offsets[face + 1])
                ]
            }
            volume = cell_volumes[cell]
            dual[0][vertices] += volume / len(vertices)
            dual[1][np.asarray(sorted(edge_set), dtype=np.int32)] += volume / len(
                edge_set
            )
            dual[2][faces] += volume / len(faces)
            dual[3][cell] = 1.0
        if any(
            np.any(~np.isfinite(value) | (value <= 0.0)) for value in (*primal, *dual)
        ):
            raise ValueError("Cut-cell cochain primal/dual metrics must be positive.")
        hodge = tuple(
            dual_value / primal_value
            for dual_value, primal_value in zip(dual, primal, strict=True)
        )
        metric_state = self.metric_plan.prepare(
            hodge,
            primal_measures=primal,
            dual_measures=dual,
            coordinates=(coordinates, edge_centers, face_centers, cell_centers),
            time=time,
            revision=revision,
        )
        return CutCellCochainState(
            topology=self.topology,
            metrics=metric_state,
            complex_id=self.complex.topology_id,
            state_id=canonical_fingerprint(
                {
                    "kind": "cut-cell-cochain-state",
                    "plan": self.plan_id,
                    "revision": int(revision),
                }
            ),
        )


__all__ = ["CutCellCochainPlan", "CutCellCochainState"]
