#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._simplicial_locator import PreparedSimplicialCellLocator


class UnstructuredWhitneyCurrentResult(StrictModule):
    start_charge: Array
    end_charge: Array
    edge_current: Array
    continuity_residual: Array
    maximum_continuity_defect: Array
    route_overflow: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class UnstructuredWhitneyCurrentPlan(StrictModule, NonTrainableState):
    """Affine-simplex Whitney-0/1 trajectory current with bounded subdivision."""

    locator: PreparedSimplicialCellLocator
    edges: Array
    cell_edges: Array
    cell_edge_signs: Array
    incidence: Array
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
    ):
        if locator.cell_map.coordinate_element.degree != 1:
            raise ValueError("Whitney current requires an order-one cell map.")
        cells = np.asarray(locator.cells, dtype=np.int32)
        local_pairs = tuple(itertools.combinations(range(cells.shape[1]), 2))
        edge_map: dict[tuple[int, int], int] = {}
        edges = []
        cell_edges = np.empty((cells.shape[0], len(local_pairs)), dtype=np.int32)
        signs = np.empty_like(cell_edges)
        for cell_index, cell in enumerate(cells):
            for local_index, (left, right) in enumerate(local_pairs):
                a, b = int(cell[left]), int(cell[right])
                canonical = (min(a, b), max(a, b))
                if canonical not in edge_map:
                    edge_map[canonical] = len(edges)
                    edges.append(canonical)
                cell_edges[cell_index, local_index] = edge_map[canonical]
                signs[cell_index, local_index] = 1 if (a, b) == canonical else -1
        edge_array = np.asarray(edges, dtype=np.int32)
        incidence = np.zeros((edge_array.shape[0], locator.coordinate_count))
        incidence[np.arange(edge_array.shape[0]), edge_array[:, 0]] = -1.0
        incidence[np.arange(edge_array.shape[0]), edge_array[:, 1]] = 1.0
        segments = int(maximum_segments)
        if segments <= 0:
            raise ValueError("maximum_segments must be positive.")
        self.locator = locator
        self.edges = jnp.asarray(edge_array)
        self.cell_edges = jnp.asarray(cell_edges)
        self.cell_edge_signs = jnp.asarray(signs)
        self.incidence = jnp.asarray(incidence)
        self.maximum_segments = segments
        self.tolerance = float(tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-whitney-current",
                "locator": locator.locator_id,
                "maximum_segments": segments,
                "tolerance": float(tolerance),
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
        active = jnp.asarray(active_mask, dtype=bool)
        dt = jnp.asarray(step_size, dtype=start.dtype).reshape(())
        if (
            start.shape != end.shape
            or charge.shape != active.shape
            or charge.shape != (start.shape[0],)
        ):
            raise ValueError("Unstructured current particle arrays disagree.")
        start_location = self.locator.locate(start)
        end_location = self.locator.locate(end)
        start_charge = jnp.zeros((self.locator.coordinate_count,), dtype=start.dtype)
        end_charge = jnp.zeros_like(start_charge)
        safe_start = jnp.maximum(start_location.cell_ids, 0)
        safe_end = jnp.maximum(end_location.cell_ids, 0)
        for local in range(self.locator.cells.shape[1]):
            start_charge = start_charge.at[self.locator.cells[safe_start, local]].add(
                jnp.where(
                    active & start_location.inside,
                    charge * start_location.barycentric[:, local],
                    0.0,
                )
            )
            end_charge = end_charge.at[self.locator.cells[safe_end, local]].add(
                jnp.where(
                    active & end_location.inside,
                    charge * end_location.barycentric[:, local],
                    0.0,
                )
            )
        times = jnp.linspace(0.0, 1.0, self.maximum_segments + 1, dtype=start.dtype)
        edge_flow = jnp.zeros((self.edges.shape[0],), dtype=start.dtype)
        route_overflow = jnp.asarray(False)
        local_pairs = tuple(itertools.combinations(range(self.locator.cells.shape[1]), 2))
        for segment in range(self.maximum_segments):
            left_point = start + times[segment] * (end - start)
            right_point = start + times[segment + 1] * (end - start)
            left_location = self.locator.locate(left_point)
            right_location = self.locator.locate(right_point)
            same = (
                active
                & left_location.inside
                & right_location.inside
                & (left_location.cell_ids == right_location.cell_ids)
            )
            route_overflow = route_overflow | jnp.any(
                active & left_location.inside & right_location.inside & ~same
            )
            cell = jnp.maximum(left_location.cell_ids, 0)
            for local_index, (a, b) in enumerate(local_pairs):
                integral = (
                    left_location.barycentric[:, a] * right_location.barycentric[:, b]
                    - left_location.barycentric[:, b] * right_location.barycentric[:, a]
                )
                edge = self.cell_edges[cell, local_index]
                sign = self.cell_edge_signs[cell, local_index]
                edge_flow = edge_flow.at[edge].add(
                    jnp.where(same, -charge * sign * integral / dt, 0.0)
                )
        continuity = (end_charge - start_charge) / dt + self.incidence.T @ edge_flow
        maximum = jnp.max(jnp.abs(continuity), initial=0.0)
        scale = jnp.maximum(
            1.0, jnp.max(jnp.abs((end_charge - start_charge) / dt), initial=0.0)
        )
        finite = jnp.all(jnp.isfinite(edge_flow)) & jnp.all(jnp.isfinite(continuity))
        successful = (
            start_location.successful.all()
            & end_location.successful.all()
            & ~route_overflow
            & finite
            & (maximum <= self.tolerance * scale)
        )
        return UnstructuredWhitneyCurrentResult(
            start_charge,
            end_charge,
            edge_flow,
            continuity,
            maximum,
            route_overflow,
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["UnstructuredWhitneyCurrentPlan", "UnstructuredWhitneyCurrentResult"]
