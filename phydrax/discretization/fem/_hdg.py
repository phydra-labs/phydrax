#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import LocalEliminationPlan, LocalEliminationResult
from .._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from .._cell_mesh import CellMesh


class HDGTraceSpace(StrictModule, NonTrainableState):
    """One scalar trace coordinate per mesh facet with cell-local gathers."""

    cell_trace_dofs: Array
    trace_valid: Array
    trace_dof_count: int = eqx.field(static=True)
    trace_space_id: str = eqx.field(static=True)

    def __init__(self, mesh: CellMesh, /):
        connectivity = mesh.connectivity
        if isinstance(connectivity, PolygonalConnectivity):
            width = int(np.max(np.asarray(connectivity.cell_kinds)))
            routes = np.asarray(connectivity.cell_edges, dtype=np.int32)[:, :width]
            valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)[:, :width]
            if not np.all(
                valid
                == (
                    np.arange(routes.shape[1])[None, :]
                    < np.asarray(connectivity.cell_kinds)[:, None]
                )
            ):
                raise ValueError("Polygonal trace validity is inconsistent.")
            count = int(connectivity.edges.shape[0])
        elif isinstance(connectivity, TetrahedralConnectivity):
            routes = np.asarray(connectivity.cell_faces, dtype=np.int32)
            count = int(connectivity.faces.shape[0])
        else:
            raise TypeError("Unsupported HDG mesh connectivity.")
        self.cell_trace_dofs = jnp.asarray(routes)
        self.trace_valid = jnp.asarray(
            np.ones_like(routes, dtype=bool)
            if not isinstance(connectivity, PolygonalConnectivity)
            else valid
        )
        self.trace_dof_count = count
        self.trace_space_id = canonical_fingerprint(
            {
                "kind": "hdg-trace-space",
                "mesh": mesh.topology_id,
                "routes": routes.tolist(),
                "valid": np.asarray(self.trace_valid).tolist(),
                "trace_dof_count": count,
            }
        )


class HDGCondensationPlan(StrictModule, NonTrainableState):
    """Local interior elimination onto one mesh-facet trace skeleton."""

    trace_space: HDGTraceSpace
    elimination: LocalEliminationPlan
    interior_dof_count: int = eqx.field(static=True)
    local_trace_dof_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        trace_space: HDGTraceSpace,
        interior_dof_count: int,
        /,
    ):
        if not isinstance(trace_space, HDGTraceSpace):
            raise TypeError("trace_space must be HDGTraceSpace.")
        interior = int(interior_dof_count)
        if interior <= 0:
            raise ValueError("interior_dof_count must be positive.")
        local_trace = int(trace_space.cell_trace_dofs.shape[1])
        retained = np.arange(interior, interior + local_trace, dtype=np.int32)
        elimination = LocalEliminationPlan(
            interior + local_trace,
            retained,
        )
        self.trace_space = trace_space
        self.elimination = elimination
        self.interior_dof_count = interior
        self.local_trace_dof_count = local_trace
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hdg-condensation-plan",
                "trace_space": trace_space.trace_space_id,
                "interior_dof_count": interior,
                "elimination": elimination.plan_id,
            }
        )

    def condense(
        self,
        local_matrix: ArrayLike,
        local_rhs: ArrayLike,
        /,
    ) -> LocalEliminationResult:
        return self.elimination.condense(local_matrix, local_rhs)

    def reconstruct(
        self,
        trace_solution: ArrayLike,
        result: LocalEliminationResult,
        /,
    ) -> Array:
        return self.elimination.reconstruct(trace_solution, result)


__all__ = ["HDGCondensationPlan", "HDGTraceSpace"]
