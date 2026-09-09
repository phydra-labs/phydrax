#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class TravelTimeResult(StrictModule):
    travel_times_s: Array
    predecessor_edges: Array
    path_margin_s: Array
    converged: Array
    derivative_available: Array


class TravelTimeGraphPlan(StrictModule, NonTrainableState):
    """Exact finite graph eikonal relaxation with fixed-path derivative evidence."""

    node_positions_m: Array
    edge_nodes: Array
    edge_lengths_m: Array
    iteration_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, node_positions_m: ArrayLike, edge_nodes: ArrayLike, /):
        positions = np.asarray(node_positions_m, dtype=float)
        edges = np.asarray(edge_nodes)
        if (
            positions.ndim != 2
            or positions.shape[0] < 2
            or positions.shape[1] not in (2, 3)
            or np.any(~np.isfinite(positions))
            or edges.ndim != 2
            or edges.shape[1] != 2
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0)
            or np.any(edges >= positions.shape[0])
            or np.any(edges[:, 0] == edges[:, 1])
        ):
            raise ValueError("Travel-time graph positions or edges are invalid.")
        directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
        lengths = np.sqrt(
            np.sum((positions[directed[:, 1]] - positions[directed[:, 0]]) ** 2, axis=1)
        )
        if np.any(lengths <= 0):
            raise ValueError("Travel-time graph edges must have positive length.")
        self.node_positions_m = jnp.asarray(positions)
        self.edge_nodes = jnp.asarray(directed, dtype=jnp.int32)
        self.edge_lengths_m = jnp.asarray(lengths)
        self.iteration_count = positions.shape[0] - 1
        self.plan_id = canonical_fingerprint(
            {"kind": "travel-time-graph", "positions_m": positions, "edges": directed}
        )

    def solve(
        self,
        wavespeed_m_s: ArrayLike,
        source_node: int,
        /,
        *,
        path_margin_tolerance_s: float = 1e-8,
    ) -> TravelTimeResult:
        speed = jnp.broadcast_to(jnp.asarray(wavespeed_m_s), self.edge_lengths_m.shape)
        speed = eqx.error_if(
            speed,
            jnp.any(~jnp.isfinite(speed)) | jnp.any(speed <= 0),
            "Travel-time edge wavespeeds must be finite and positive.",
        )
        source = int(source_node)
        if not 0 <= source < self.node_positions_m.shape[0]:
            raise ValueError("Travel-time source node is out of range.")
        edge_time = self.edge_lengths_m / speed
        infinity = jnp.asarray(jnp.inf, dtype=edge_time.dtype)
        times = jnp.full((self.node_positions_m.shape[0],), infinity).at[source].set(0.0)
        predecessor = jnp.full(times.shape, -1, dtype=jnp.int32)
        second = jnp.full_like(times, infinity)
        for _ in range(self.iteration_count):
            candidates = times[self.edge_nodes[:, 0]] + edge_time
            destination = self.edge_nodes[:, 1]
            best = jnp.full_like(times, infinity).at[destination].min(candidates)
            sentinel = jnp.asarray(self.edge_nodes.shape[0], dtype=jnp.int32)
            best_edge = jnp.full(times.shape, sentinel, dtype=jnp.int32)
            is_best = candidates == best[destination]
            edge_ids = jnp.arange(candidates.size, dtype=jnp.int32)
            best_edge = best_edge.at[destination].min(
                jnp.where(is_best, edge_ids, sentinel)
            )
            candidate_second = jnp.where(is_best, infinity, candidates)
            second_best = (
                jnp.full_like(times, infinity).at[destination].min(candidate_second)
            )
            improved = best < times
            second = jnp.where(
                improved,
                jnp.minimum(times, second_best),
                jnp.minimum(second, second_best),
            )
            predecessor = jnp.where(improved, best_edge, predecessor)
            times = jnp.minimum(times, best)
            times = times.at[source].set(0.0)
        residual_candidates = times[self.edge_nodes[:, 0]] + edge_time
        bellman = (
            jnp.full_like(times, infinity)
            .at[self.edge_nodes[:, 1]]
            .min(residual_candidates)
        )
        converged = jnp.all(
            times <= bellman + 64 * jnp.finfo(times.dtype).eps * jnp.maximum(times, 1.0)
        )
        margin = second - times
        margin = margin.at[source].set(infinity)
        derivative_available = converged & jnp.all(
            jnp.isinf(margin) | (margin > path_margin_tolerance_s)
        )
        return TravelTimeResult(
            times, predecessor, margin, converged, derivative_available
        )


class EventLocationPlan(StrictModule, NonTrainableState):
    station_positions_m: Array
    observed_arrival_s: Array
    standard_deviation_s: Array
    phase_velocity_m_s: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        station_positions_m: ArrayLike,
        observed_arrival_s: ArrayLike,
        standard_deviation_s: ArrayLike,
        phase_velocity_m_s: ArrayLike,
        /,
    ):
        stations = jnp.asarray(station_positions_m)
        arrivals, deviation, velocity = jnp.broadcast_arrays(
            jnp.asarray(observed_arrival_s),
            jnp.asarray(standard_deviation_s),
            jnp.asarray(phase_velocity_m_s),
        )
        if (
            stations.ndim != 2
            or stations.shape[1] != 3
            or arrivals.shape != (stations.shape[0],)
        ):
            raise ValueError("Event-location stations and pick vectors have wrong shape.")
        stations = eqx.error_if(
            stations,
            jnp.any(~jnp.isfinite(stations))
            | jnp.any(~jnp.isfinite(arrivals))
            | jnp.any(~jnp.isfinite(deviation))
            | jnp.any(deviation <= 0)
            | jnp.any(~jnp.isfinite(velocity))
            | jnp.any(velocity <= 0),
            "Event-location stations, picks, uncertainties, and velocities must be physical.",
        )
        self.station_positions_m = stations
        self.observed_arrival_s, self.standard_deviation_s = arrivals, deviation
        self.phase_velocity_m_s = velocity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "homogeneous-event-location",
                "stations_m": np.asarray(stations),
                "pick_count": stations.shape[0],
            }
        )

    def predict(self, hypocenter_m: ArrayLike, origin_time_s: ArrayLike, /) -> Array:
        hypocenter, origin = jnp.asarray(hypocenter_m), jnp.asarray(origin_time_s)
        if hypocenter.shape != (3,) or origin.shape != ():
            raise ValueError("Hypocenter and origin-time shapes are invalid.")
        distance = jnp.sqrt(
            jnp.sum((self.station_positions_m - hypocenter[None, :]) ** 2, axis=1)
        )
        return origin + distance / self.phase_velocity_m_s

    def log_likelihood(
        self, hypocenter_m: ArrayLike, origin_time_s: ArrayLike, /
    ) -> Array:
        standardized = (
            self.observed_arrival_s - self.predict(hypocenter_m, origin_time_s)
        ) / self.standard_deviation_s
        return jnp.sum(
            -0.5 * standardized**2
            - jnp.log(self.standard_deviation_s)
            - 0.5 * jnp.log(2 * jnp.pi)
        )

    def associate(self, predicted_arrival_s: ArrayLike, maximum_sigma: float, /) -> Array:
        predicted = jnp.asarray(predicted_arrival_s)
        threshold = float(maximum_sigma)
        if (
            predicted.shape != self.observed_arrival_s.shape
            or not np.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("Association prediction/threshold are invalid.")
        # This Boolean host decision is intentionally outside the derivative contract.
        return (
            jnp.abs(predicted - self.observed_arrival_s)
            <= threshold * self.standard_deviation_s
        )


class MomentTensorRadiationPlan(StrictModule, NonTrainableState):
    directions: Array
    p_polarizations: Array
    s_polarizations: Array

    def __init__(self, source_to_receiver_directions: ArrayLike, /):
        directions = jnp.asarray(source_to_receiver_directions)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("Moment-tensor directions must have shape (receivers,3).")
        norm = jnp.sqrt(jnp.sum(directions**2, axis=1))
        directions = eqx.error_if(
            directions,
            jnp.any(~jnp.isfinite(directions)) | jnp.any(jnp.abs(norm - 1.0) > 1e-10),
            "Moment-tensor directions must be finite unit vectors.",
        )
        self.directions = directions
        self.p_polarizations = directions
        # Two deterministic transverse basis vectors per receiver.
        seed = jnp.where(
            (jnp.abs(directions[:, 2]) < 0.9)[:, None],
            jnp.asarray((0.0, 0.0, 1.0)),
            jnp.asarray((0.0, 1.0, 0.0)),
        )
        first = jnp.cross(directions, seed)
        first /= jnp.sqrt(jnp.sum(first**2, axis=1))[:, None]
        second = jnp.cross(directions, first)
        self.s_polarizations = jnp.stack((first, second), axis=1)

    def amplitudes(self, moment_tensor_N_m: ArrayLike, /) -> tuple[Array, Array]:
        tensor = jnp.asarray(moment_tensor_N_m)
        if tensor.shape != (3, 3):
            raise ValueError("Moment tensor must be 3x3.")
        tensor = eqx.error_if(
            tensor,
            jnp.any(~jnp.isfinite(tensor)) | jnp.any(jnp.abs(tensor - tensor.T) > 1e-10),
            "Moment tensor must be finite and symmetric.",
        )
        traction = self.directions @ tensor.T
        p = jnp.sum(self.directions * traction, axis=1)
        shear = traction - p[:, None] * self.directions
        s = ein.contract("rki,ri->rk", self.s_polarizations, shear)
        return p, s


__all__ = [
    "EventLocationPlan",
    "MomentTensorRadiationPlan",
    "TravelTimeGraphPlan",
    "TravelTimeResult",
]
