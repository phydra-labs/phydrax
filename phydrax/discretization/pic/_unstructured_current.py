#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import SparseLinearMap
from ...typing import Bool, Dim, Float, Int32, Scalar
from .._cell_complex import simplicial_cell_complex
from .._simplicial_locator import PreparedSimplicialCellLocator
from ..fem._simplicial_whitney_chains import SimplicialWhitneyKernel


class CurrentVertexDim(Dim):
    pass


class CurrentEdgeDim(Dim):
    pass


class CurrentCellDim(Dim):
    pass


class CurrentLocalEdgeDim(Dim):
    pass


@final
class UnstructuredWhitneyCurrentResult(StrictModule):
    __strict_contract__ = True

    start_charge: Float[CurrentVertexDim]
    end_charge: Float[CurrentVertexDim]
    end_charge_magnitude: Float[CurrentVertexDim]
    edge_current: Float[CurrentEdgeDim]
    continuity_residual: Float[CurrentVertexDim]
    maximum_continuity_defect: Float[Scalar]
    route_overflow: Bool[Scalar]
    finite: Bool[Scalar]
    successful: Bool[Scalar]
    plan_id: str = eqx.field(static=True)


@final
class UnstructuredWhitneyCurrentPlan(StrictModule, NonTrainableState):
    """Conservative positive Whitney line current with exact fixed-capacity facets."""

    __strict_contract__ = True

    locator: PreparedSimplicialCellLocator
    kernel: SimplicialWhitneyKernel
    edges: Int32[CurrentEdgeDim, Literal[2]]
    cell_edges: Int32[CurrentCellDim, CurrentLocalEdgeDim]
    cell_edge_signs: Int32[CurrentCellDim, CurrentLocalEdgeDim]
    incidence: SparseLinearMap
    maximum_segments: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        locator: PreparedSimplicialCellLocator,
        /,
        *,
        maximum_segments: int = 8,
        tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(locator, PreparedSimplicialCellLocator):
            raise TypeError("locator must be a prepared simplicial locator.")
        if locator.cell_map.coordinate_element.degree != 1:
            raise ValueError("Whitney current requires an order-one cell map.")
        if maximum_segments <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Whitney current capacity and tolerance must be positive.")
        cells = np.asarray(locator.cells, dtype=np.int32)
        simplices = [np.arange(locator.coordinate_count, dtype=np.int32)[:, None]]
        for degree in range(1, locator.dimension + 1):
            supports = sorted(
                {
                    tuple(sorted(int(cell[i]) for i in subset))
                    for cell in cells
                    for subset in itertools.combinations(
                        range(cells.shape[1]), degree + 1
                    )
                }
            )
            simplices.append(np.asarray(supports, dtype=np.int32))
        topology = simplicial_cell_complex(tuple(simplices))
        kernel = SimplicialWhitneyKernel(topology, locator)
        self.locator = locator
        self.kernel = kernel
        self.edges = jnp.asarray(simplices[1])
        self.cell_edges = kernel.cell_routes[1].indices
        self.cell_edge_signs = kernel.cell_routes[1].signs
        self.incidence = topology.incidences[0].exterior_derivative()
        self.maximum_segments = maximum_segments
        self.tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-whitney-current",
                "kernel": kernel.kernel_id,
                "maximum_segments": maximum_segments,
                "tolerance": tolerance,
            }
        )

    def deposit(
        self,
        start_position: ArrayLike,
        end_position: ArrayLike,
        macrocharge: ArrayLike,
        active_mask: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> UnstructuredWhitneyCurrentResult:
        start = jnp.asarray(start_position)
        end = jnp.asarray(end_position, dtype=start.dtype)
        charge = jnp.asarray(macrocharge, dtype=start.dtype)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        dt = jnp.asarray(step_size, dtype=start.dtype).reshape(())
        if (
            start.shape != end.shape
            or charge.shape != active.shape
            or charge.shape != (start.shape[0],)
        ):
            raise ValueError("Unstructured current particle arrays disagree.")
        first, last, route = self.kernel.prepare_trajectory(
            start, end, maximum_segments=self.maximum_segments
        )
        weights = jnp.where(active, charge, 0)
        start_charge = first.deposit(weights)
        end_charge = last.deposit(weights)
        magnitude = last.deposit(jnp.abs(weights))
        edge_current = route.deposit(weights / dt)
        continuity = (end_charge - start_charge) / dt - self.incidence.transpose_mv(
            edge_current
        )
        maximum = jnp.max(jnp.abs(continuity), initial=0.0)
        scale = jnp.maximum(
            1.0, jnp.max(jnp.abs((end_charge - start_charge) / dt), initial=0.0)
        )
        overflow = jnp.any(active & route.overflow)
        finite = jnp.all(jnp.isfinite(edge_current)) & jnp.all(jnp.isfinite(continuity))
        successful = (
            jnp.all(~active | (first.successful & last.successful & route.successful))
            & ~overflow
            & finite
            & (dt > 0)
            & (maximum <= self.tolerance * scale)
        )
        return UnstructuredWhitneyCurrentResult(
            start_charge,
            end_charge,
            magnitude,
            edge_current,
            continuity,
            maximum,
            overflow,
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["UnstructuredWhitneyCurrentPlan", "UnstructuredWhitneyCurrentResult"]
