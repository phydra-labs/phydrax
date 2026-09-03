#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class PMJAttachmentEpoch(StrictModule, NonTrainableState):
    """Geometry epochs for a Purkinje graph and myocardial support."""

    graph_geometry: Array
    myocardial_geometry: Array

    def __init__(
        self,
        graph_geometry: int | ArrayLike,
        myocardial_geometry: int | ArrayLike,
        /,
    ):
        graph_host = np.asarray(graph_geometry)
        myocardial_host = np.asarray(myocardial_geometry)
        if graph_host.shape != () or myocardial_host.shape != ():
            raise ValueError("PMJ attachment epochs must be scalar values.")
        if not np.issubdtype(graph_host.dtype, np.integer) or not np.issubdtype(
            myocardial_host.dtype, np.integer
        ):
            raise TypeError("PMJ attachment epochs must be integers.")
        if int(graph_host) < 0 or int(myocardial_host) < 0:
            raise ValueError("PMJ attachment epochs must be non-negative.")
        self.graph_geometry = jnp.asarray(graph_host, dtype=jnp.int32)
        self.myocardial_geometry = jnp.asarray(myocardial_host, dtype=jnp.int32)

    def matches(self, other: PMJAttachmentEpoch, /) -> Array:
        if not isinstance(other, PMJAttachmentEpoch):
            raise TypeError("other must be a PMJAttachmentEpoch.")
        return (self.graph_geometry == other.graph_geometry) & (
            self.myocardial_geometry == other.myocardial_geometry
        )


class PMJAttachmentEvidence(StrictModule, NonTrainableState):
    """Distance, coverage, capacity, and epoch evidence for fixed PMJ routes."""

    distances_mm: Array
    within_distance: Array
    route_active: Array
    attached_count: Array
    uncovered_count: Array
    coverage_fraction: Array
    capacity_remaining: Array
    capacity_ok: Array
    epoch_matches: Array
    finite: Array
    accepted: Array
    fixed_routes: bool = eqx.field(static=True)
    attachment_id: str = eqx.field(static=True)


class PMJAttachmentCandidate(StrictModule, NonTrainableState):
    """Fixed-shape graph/support pairs and evidence for one geometry realization."""

    graph_points_mm: Array
    myocardial_points_mm: Array
    evidence: PMJAttachmentEvidence


class PurkinjeAttachmentPlan(StrictModule, NonTrainableState):
    """Host preparation policy for capacity-bounded nearest-support PMJ routes."""

    pmj_capacity: int = eqx.field(static=True)
    maximum_distance_mm: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pmj_capacity: int,
        maximum_distance_mm: float,
        /,
        *,
        plan_id: str | None = None,
    ):
        capacity = int(pmj_capacity)
        maximum_distance = float(maximum_distance_mm)
        if capacity <= 0:
            raise ValueError("pmj_capacity must be positive.")
        if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
            raise ValueError("maximum_distance_mm must be finite and positive.")
        payload = {
            "kind": "purkinje-attachment-plan",
            "pmj_capacity": capacity,
            "maximum_distance_mm": maximum_distance,
            "tie_break": "lowest-myocardial-support-index",
            "length_unit": "mm",
        }
        self.pmj_capacity = capacity
        self.maximum_distance_mm = maximum_distance
        self.plan_id = _resolved_id("plan_id", plan_id, payload)

    def prepare(
        self,
        graph_points_mm: ArrayLike,
        myocardial_support_points_mm: ArrayLike,
        /,
        *,
        pmj_candidate_mask: ArrayLike | None = None,
        graph_active_mask: ArrayLike | None = None,
        myocardial_active_mask: ArrayLike | None = None,
        graph_geometry_id: str,
        myocardial_geometry_id: str,
        epoch: PMJAttachmentEpoch,
        attachment_id: str | None = None,
    ) -> PreparedPurkinjeAttachment:
        if not isinstance(epoch, PMJAttachmentEpoch):
            raise TypeError("epoch must be a PMJAttachmentEpoch.")
        graph = _host_points(graph_points_mm, "graph_points_mm")
        myocardial = _host_points(
            myocardial_support_points_mm, "myocardial_support_points_mm"
        )
        graph_active = _host_mask(
            graph_active_mask,
            graph.shape[0],
            "graph_active_mask",
        )
        candidates = _host_mask(
            pmj_candidate_mask,
            graph.shape[0],
            "pmj_candidate_mask",
        )
        candidates &= graph_active
        myocardial_active = _host_mask(
            myocardial_active_mask,
            myocardial.shape[0],
            "myocardial_active_mask",
        )
        candidate_indices = np.flatnonzero(candidates)
        support_indices = np.flatnonzero(myocardial_active)
        if candidate_indices.size == 0:
            raise ValueError("At least one active PMJ candidate is required.")
        if support_indices.size == 0:
            raise ValueError("At least one active myocardial support is required.")
        if candidate_indices.size > self.pmj_capacity:
            raise ValueError(
                f"PMJ candidate count {candidate_indices.size} exceeds configured "
                f"capacity {self.pmj_capacity}."
            )

        graph_routes = np.zeros((self.pmj_capacity,), dtype=np.int32)
        myocardial_routes = np.zeros((self.pmj_capacity,), dtype=np.int32)
        route_active = np.zeros((self.pmj_capacity,), dtype=bool)
        for route_slot, graph_index in enumerate(candidate_indices):
            delta = myocardial[support_indices] - graph[graph_index]
            squared_distance = np.sum(delta * delta, axis=1)
            nearest_active_slot = int(np.argmin(squared_distance))
            graph_routes[route_slot] = graph_index
            myocardial_routes[route_slot] = support_indices[nearest_active_slot]
            route_active[route_slot] = True

        graph_id = str(graph_geometry_id)
        myocardial_id = str(myocardial_geometry_id)
        if not graph_id or not myocardial_id:
            raise ValueError("PMJ attachment geometry IDs must be non-empty.")
        graph_routes_array = jnp.asarray(graph_routes, dtype=jnp.int32)
        myocardial_routes_array = jnp.asarray(myocardial_routes, dtype=jnp.int32)
        route_active_array = jnp.asarray(route_active, dtype=bool)
        payload = {
            "kind": "prepared-purkinje-attachment",
            "plan": self.plan_id,
            "graph_geometry": graph_id,
            "myocardial_geometry": myocardial_id,
            "epochs": [
                int(np.asarray(epoch.graph_geometry)),
                int(np.asarray(epoch.myocardial_geometry)),
            ],
            "graph_point_capacity": graph.shape[0],
            "myocardial_support_capacity": myocardial.shape[0],
            "graph_points": array_tree_fingerprint(jnp.asarray(graph)),
            "myocardial_support_points": array_tree_fingerprint(jnp.asarray(myocardial)),
            "graph_routes": array_tree_fingerprint(graph_routes_array),
            "myocardial_routes": array_tree_fingerprint(myocardial_routes_array),
            "route_active": array_tree_fingerprint(route_active_array),
        }
        identifier = _resolved_id("attachment_id", attachment_id, payload)
        return PreparedPurkinjeAttachment(
            plan=self,
            graph_indices=graph_routes_array,
            myocardial_support_indices=myocardial_routes_array,
            route_active=route_active_array,
            graph_point_capacity=graph.shape[0],
            myocardial_support_capacity=myocardial.shape[0],
            graph_geometry_id=graph_id,
            myocardial_geometry_id=myocardial_id,
            prepared_epoch=epoch,
            attachment_id=identifier,
        )


class PreparedPurkinjeAttachment(StrictModule, NonTrainableState):
    """Immutable fixed-route PMJ attachment executable for one geometry epoch."""

    plan: PurkinjeAttachmentPlan
    graph_indices: Array
    myocardial_support_indices: Array
    route_active: Array
    prepared_epoch: PMJAttachmentEpoch
    graph_point_capacity: int = eqx.field(static=True)
    myocardial_support_capacity: int = eqx.field(static=True)
    graph_geometry_id: str = eqx.field(static=True)
    myocardial_geometry_id: str = eqx.field(static=True)
    attachment_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: PurkinjeAttachmentPlan,
        graph_indices: Array,
        myocardial_support_indices: Array,
        route_active: Array,
        graph_point_capacity: int,
        myocardial_support_capacity: int,
        graph_geometry_id: str,
        myocardial_geometry_id: str,
        prepared_epoch: PMJAttachmentEpoch,
        attachment_id: str,
    ):
        if not isinstance(plan, PurkinjeAttachmentPlan):
            raise TypeError("plan must be a PurkinjeAttachmentPlan.")
        graph_capacity = int(graph_point_capacity)
        myocardial_capacity = int(myocardial_support_capacity)
        if graph_capacity <= 0 or myocardial_capacity <= 0:
            raise ValueError("Prepared PMJ geometry capacities must be positive.")
        graph_indices_host = np.asarray(graph_indices)
        myocardial_indices_host = np.asarray(myocardial_support_indices)
        active_host = np.asarray(route_active)
        route_shape = (plan.pmj_capacity,)
        if (
            graph_indices_host.shape != route_shape
            or myocardial_indices_host.shape != route_shape
            or active_host.shape != route_shape
        ):
            raise ValueError("Prepared PMJ route arrays must match pmj_capacity.")
        if not np.issubdtype(graph_indices_host.dtype, np.integer) or not np.issubdtype(
            myocardial_indices_host.dtype, np.integer
        ):
            raise TypeError("Prepared PMJ route indices must be integers.")
        if not np.issubdtype(active_host.dtype, np.bool_):
            raise TypeError("Prepared PMJ route activity must be boolean.")
        if np.any(graph_indices_host < 0) or np.any(graph_indices_host >= graph_capacity):
            raise ValueError("Prepared PMJ graph indices are out of bounds.")
        if np.any(myocardial_indices_host < 0) or np.any(
            myocardial_indices_host >= myocardial_capacity
        ):
            raise ValueError("Prepared myocardial support indices are out of bounds.")
        active_count = int(np.sum(active_host))
        if active_count <= 0:
            raise ValueError("Prepared PMJ attachments require an active route.")
        expected_active = np.arange(plan.pmj_capacity) < active_count
        if not np.array_equal(active_host, expected_active):
            raise ValueError("Prepared PMJ active routes must occupy a packed prefix.")
        if np.unique(graph_indices_host[active_host]).size != active_count:
            raise ValueError("Each active graph point may have only one PMJ route.")
        if np.any(graph_indices_host[~active_host] != 0) or np.any(
            myocardial_indices_host[~active_host] != 0
        ):
            raise ValueError(
                "Inactive PMJ route indices must use canonical zero padding."
            )
        if not isinstance(prepared_epoch, PMJAttachmentEpoch):
            raise TypeError("prepared_epoch must be a PMJAttachmentEpoch.")
        graph_id = str(graph_geometry_id)
        myocardial_id = str(myocardial_geometry_id)
        identifier = str(attachment_id)
        if not graph_id or not myocardial_id or not identifier:
            raise ValueError("Prepared PMJ attachment identities must be non-empty.")
        self.plan = plan
        self.graph_indices = jnp.asarray(graph_indices_host, dtype=jnp.int32)
        self.myocardial_support_indices = jnp.asarray(
            myocardial_indices_host, dtype=jnp.int32
        )
        self.route_active = jnp.asarray(active_host, dtype=bool)
        self.graph_point_capacity = graph_capacity
        self.myocardial_support_capacity = myocardial_capacity
        self.graph_geometry_id = graph_id
        self.myocardial_geometry_id = myocardial_id
        self.prepared_epoch = prepared_epoch
        self.attachment_id = identifier

    def evaluate(
        self,
        graph_points_mm: ArrayLike,
        myocardial_support_points_mm: ArrayLike,
        current_epoch: PMJAttachmentEpoch,
        /,
    ) -> PMJAttachmentCandidate:
        if not isinstance(current_epoch, PMJAttachmentEpoch):
            raise TypeError("current_epoch must be a PMJAttachmentEpoch.")
        graph = jnp.asarray(graph_points_mm)
        myocardial = jnp.asarray(myocardial_support_points_mm)
        if graph.shape != (self.graph_point_capacity, 3):
            raise ValueError(
                "graph_points_mm shape changed from the prepared fixed capacity."
            )
        if myocardial.shape != (self.myocardial_support_capacity, 3):
            raise ValueError(
                "myocardial_support_points_mm shape changed from the prepared fixed capacity."
            )
        if not jnp.issubdtype(graph.dtype, jnp.inexact) or not jnp.issubdtype(
            myocardial.dtype, jnp.inexact
        ):
            raise TypeError("PMJ attachment coordinates must be floating-point arrays.")
        graph_routes = graph[self.graph_indices]
        myocardial_routes = myocardial[self.myocardial_support_indices]
        delta = graph_routes - myocardial_routes
        squared_distance = oe.contract("ri,ri->r", delta, delta)
        safe_squared_distance = jnp.where(
            self.route_active,
            jnp.maximum(squared_distance, 0.0),
            jnp.ones_like(squared_distance),
        )
        distances = jnp.sqrt(safe_squared_distance)
        within = self.route_active & (distances <= self.plan.maximum_distance_mm)
        attached_count = jnp.sum(self.route_active, dtype=jnp.int32)
        covered_count = jnp.sum(within, dtype=jnp.int32)
        uncovered_count = attached_count - covered_count
        coverage_fraction = covered_count.astype(distances.dtype) / jnp.maximum(
            attached_count, 1
        ).astype(distances.dtype)
        capacity_remaining = (
            jnp.asarray(self.plan.pmj_capacity, dtype=jnp.int32) - attached_count
        )
        capacity_ok = (attached_count <= self.plan.pmj_capacity) & (
            capacity_remaining >= 0
        )
        epoch_matches = self.prepared_epoch.matches(current_epoch)
        finite = (
            jnp.all(jnp.isfinite(graph))
            & jnp.all(jnp.isfinite(myocardial))
            & jnp.all((~self.route_active) | jnp.isfinite(distances))
        )
        complete = jnp.all((~self.route_active) | within)
        accepted = finite & complete & capacity_ok & epoch_matches
        graph_output = jnp.where(
            self.route_active[:, None], graph_routes, jnp.zeros_like(graph_routes)
        )
        myocardial_output = jnp.where(
            self.route_active[:, None],
            myocardial_routes,
            jnp.zeros_like(myocardial_routes),
        )
        evidence = PMJAttachmentEvidence(
            distances_mm=jnp.where(
                self.route_active, distances, jnp.zeros_like(distances)
            ),
            within_distance=within,
            route_active=self.route_active,
            attached_count=attached_count,
            uncovered_count=uncovered_count,
            coverage_fraction=coverage_fraction,
            capacity_remaining=capacity_remaining,
            capacity_ok=capacity_ok,
            epoch_matches=epoch_matches,
            finite=finite,
            accepted=accepted,
            fixed_routes=True,
            attachment_id=self.attachment_id,
        )
        return PMJAttachmentCandidate(
            graph_points_mm=graph_output,
            myocardial_points_mm=myocardial_output,
            evidence=evidence,
        )

    def gather_graph(self, graph_values: ArrayLike, /) -> Array:
        values = jnp.asarray(graph_values)
        if values.ndim == 0 or values.shape[0] != self.graph_point_capacity:
            raise ValueError("graph_values must begin with graph_point_capacity.")
        gathered = values[self.graph_indices]
        mask = self.route_active.reshape(
            (self.plan.pmj_capacity,) + (1,) * (values.ndim - 1)
        )
        return jnp.where(mask, gathered, jnp.zeros_like(gathered))

    def gather_myocardium(self, myocardial_values: ArrayLike, /) -> Array:
        values = jnp.asarray(myocardial_values)
        if values.ndim == 0 or values.shape[0] != self.myocardial_support_capacity:
            raise ValueError(
                "myocardial_values must begin with myocardial_support_capacity."
            )
        gathered = values[self.myocardial_support_indices]
        mask = self.route_active.reshape(
            (self.plan.pmj_capacity,) + (1,) * (values.ndim - 1)
        )
        return jnp.where(mask, gathered, jnp.zeros_like(gathered))

    def scatter_to_myocardium(self, pmj_values: ArrayLike, /) -> Array:
        values = jnp.asarray(pmj_values)
        if values.ndim == 0 or values.shape[0] != self.plan.pmj_capacity:
            raise ValueError("pmj_values must begin with pmj_capacity.")
        mask = self.route_active.reshape(
            (self.plan.pmj_capacity,) + (1,) * (values.ndim - 1)
        )
        active_values = jnp.where(mask, values, jnp.zeros_like(values))
        output = jnp.zeros(
            (self.myocardial_support_capacity,) + values.shape[1:], dtype=values.dtype
        )
        return output.at[self.myocardial_support_indices].add(active_values)


def _host_points(values: ArrayLike, name: str, /) -> np.ndarray:
    points = np.asarray(values)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{name} must have shape (positive_count, 3).")
    if not np.issubdtype(points.dtype, np.inexact):
        points = points.astype(float)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite.")
    return points


def _host_mask(values: ArrayLike | None, size: int, name: str, /) -> np.ndarray:
    if values is None:
        return np.ones((size,), dtype=bool)
    mask = np.asarray(values)
    if mask.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}.")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"{name} must be boolean.")
    return mask.copy()


def _resolved_id(name: str, value: str | None, payload: dict[str, object], /) -> str:
    if value is None:
        return canonical_fingerprint(payload)
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


__all__ = [
    "PMJAttachmentCandidate",
    "PMJAttachmentEpoch",
    "PMJAttachmentEvidence",
    "PreparedPurkinjeAttachment",
    "PurkinjeAttachmentPlan",
]
