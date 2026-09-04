#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity muscle routes through points attached to articulated bodies."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization.particle import PreparedReducedArticulation
from ...linalg import ArraySpace, FunctionLinearOperator


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class FixedBodyRoutePlan(StrictModule):
    """Static CSR topology for body-attached, piecewise-linear tensile routes.

    ``point_offsets`` has one extra entry and partitions ``body_ids`` into routes.
    Every route has at least two points. ``route_mask`` preserves a fixed route
    capacity while disabling selected rows without changing any runtime shape.
    """

    route_names: tuple[str, ...] = eqx.field(static=True)
    point_offsets: tuple[int, ...] = eqx.field(static=True)
    body_ids: tuple[int, ...] = eqx.field(static=True)
    route_mask: tuple[bool, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        route_names: Sequence[str],
        point_offsets: Sequence[int],
        body_ids: Sequence[int],
        /,
        *,
        route_mask: Sequence[bool] | None = None,
        plan_id: str | None = None,
    ):
        names = tuple(_identifier(name, "route name") for name in route_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("route_names must be non-empty and unique.")
        offsets = tuple(int(value) for value in point_offsets)
        if len(offsets) != len(names) + 1 or offsets[0] != 0:
            raise ValueError("point_offsets must be CSR offsets beginning at zero.")
        if any(stop - start < 2 for start, stop in zip(offsets[:-1], offsets[1:])):
            raise ValueError("Every fixed body route must contain at least two points.")
        if any(right <= left for left, right in zip(offsets[:-1], offsets[1:])):
            raise ValueError("point_offsets must be strictly increasing.")
        bodies: list[int] = []
        for value in body_ids:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("body_ids must contain integer body IDs.")
            bodies.append(int(value))
        body_tuple = tuple(bodies)
        if offsets[-1] != len(body_tuple):
            raise ValueError("The final CSR offset must equal the body_ids size.")
        mask = (
            (True,) * len(names)
            if route_mask is None
            else tuple(bool(value) for value in route_mask)
        )
        if len(mask) != len(names):
            raise ValueError("route_mask must have one entry per route.")
        generated = canonical_fingerprint(
            {
                "kind": "fixed-body-attached-route-plan-v1",
                "route_names": list(names),
                "point_offsets": list(offsets),
                "body_ids": list(body_tuple),
                "route_mask": list(mask),
                "length_unit": "m",
            }
        )
        self.route_names = names
        self.point_offsets = offsets
        self.body_ids = body_tuple
        self.route_mask = mask
        self.plan_id = generated if plan_id is None else _identifier(plan_id, "plan_id")

    @property
    def route_capacity(self) -> int:
        return len(self.route_names)

    @property
    def point_capacity(self) -> int:
        return len(self.body_ids)

    @property
    def segment_capacity(self) -> int:
        return self.point_capacity - self.route_capacity

    def prepare(
        self,
        articulation: PreparedReducedArticulation,
        local_positions_m: ArrayLike,
        /,
        *,
        minimum_segment_length_m: float = 1.0e-10,
    ) -> PreparedFixedBodyRoute:
        """Bind dynamic body-local coordinates to one prepared articulation."""

        return PreparedFixedBodyRoute(
            self,
            articulation,
            local_positions_m,
            minimum_segment_length_m=minimum_segment_length_m,
        )


class FixedBodyRouteEvidence(StrictModule):
    """Case-local geometric admissibility for one fixed-shape evaluation."""

    route_finite: Array
    route_nondegenerate: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class FixedBodyRouteEvaluation(StrictModule):
    """Route points, segments, lengths, and JVP length rates in SI units."""

    world_points_m: Array
    segment_vectors_m: Array
    segment_lengths_m: Array
    route_lengths_m: Array
    route_length_rates_m_per_s: Array
    evidence: FixedBodyRouteEvidence
    route_names: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class FixedBodyRoutePullbackEvidence(StrictModule):
    """Virtual-power identity for positive-tensile route-force pullback."""

    tensile_force_N: Array
    route_length_rates_m_per_s: Array
    route_power_W: Array
    generalized_power_W: Array
    power_residual_W: Array
    power_scale_W: Array
    tensile: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedFixedBodyRoute(StrictModule):
    """Prepared smooth route geometry with JVP and exact transpose actions.

    Local point coordinates are dynamic JAX leaves and therefore trainable. The
    CSR topology, route mask, body ownership, capacities, and IDs are static.
    There is no wrap/contact branch: all points are fixed in body-local frames.
    """

    local_positions_m: Array
    articulation: PreparedReducedArticulation
    plan: FixedBodyRoutePlan
    minimum_segment_length_m: float = eqx.field(static=True)
    point_body_indices: tuple[int, ...] = eqx.field(static=True)
    segment_start_indices: tuple[int, ...] = eqx.field(static=True)
    segment_stop_indices: tuple[int, ...] = eqx.field(static=True)
    segment_route_indices: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FixedBodyRoutePlan,
        articulation: PreparedReducedArticulation,
        local_positions_m: ArrayLike,
        /,
        *,
        minimum_segment_length_m: float,
    ):
        if not isinstance(plan, FixedBodyRoutePlan):
            raise TypeError("plan must be FixedBodyRoutePlan.")
        if not isinstance(articulation, PreparedReducedArticulation):
            raise TypeError("articulation must be PreparedReducedArticulation.")
        local = jnp.asarray(local_positions_m, dtype=articulation.reference_position.dtype)
        if local.shape != (plan.point_capacity, 3):
            raise ValueError(
                "local_positions_m must have fixed shape "
                f"{(plan.point_capacity, 3)}."
            )
        if not bool(np.all(np.isfinite(np.asarray(local)))):
            raise ValueError("local_positions_m must be finite at preparation.")
        minimum = float(minimum_segment_length_m)
        if not isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_segment_length_m must be positive and finite.")
        point_body_indices = tuple(
            articulation._body_index(body_id) for body_id in plan.body_ids
        )

        starts: list[int] = []
        stops: list[int] = []
        owners: list[int] = []
        for route, (begin, end) in enumerate(
            zip(plan.point_offsets[:-1], plan.point_offsets[1:])
        ):
            for point in range(begin, end - 1):
                starts.append(point)
                stops.append(point + 1)
                owners.append(route)
        self.local_positions_m = local
        self.articulation = articulation
        self.plan = plan
        self.point_body_indices = point_body_indices
        self.minimum_segment_length_m = minimum
        self.segment_start_indices = tuple(starts)
        self.segment_stop_indices = tuple(stops)
        self.segment_route_indices = tuple(owners)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-body-attached-route-v1",
                "plan": plan.plan_id,
                "articulation": articulation.prepared_id,
                "point_capacity": plan.point_capacity,
                "segment_capacity": plan.segment_capacity,
                "route_capacity": plan.route_capacity,
            }
        )

    @property
    def route_capacity(self) -> int:
        return self.plan.route_capacity

    @property
    def point_capacity(self) -> int:
        return self.plan.point_capacity

    @property
    def segment_capacity(self) -> int:
        return self.plan.segment_capacity

    def _configuration(self, value: ArrayLike, /) -> Array:
        configuration = jnp.asarray(value, dtype=self.local_positions_m.dtype)
        if configuration.shape != (self.articulation.nq,):
            raise ValueError(
                f"configuration must have shape {(self.articulation.nq,)}."
            )
        return configuration

    def _velocity(self, value: ArrayLike, /) -> Array:
        velocity = jnp.asarray(value, dtype=self.local_positions_m.dtype)
        if velocity.shape != (self.articulation.nv,):
            raise ValueError(
                f"generalized_velocity must have shape {(self.articulation.nv,)}."
            )
        return velocity

    def world_points(self, configuration: ArrayLike, /) -> Array:
        """Return body-local attachments transformed into the world frame, in m."""

        point = self._configuration(configuration)
        body_transforms = self.articulation.body_transforms(point)
        point_transforms = body_transforms[
            jnp.asarray(self.point_body_indices, dtype=jnp.int32)
        ]
        return contract(
            "pij,pj->pi",
            point_transforms[:, :3, :3],
            self.local_positions_m,
        ) + point_transforms[:, :3, 3]

    def _length_geometry(
        self, configuration: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        points = self.world_points(configuration)
        start = jnp.asarray(self.segment_start_indices, dtype=jnp.int32)
        stop = jnp.asarray(self.segment_stop_indices, dtype=jnp.int32)
        owner = jnp.asarray(self.segment_route_indices, dtype=jnp.int32)
        route_mask = jnp.asarray(self.plan.route_mask, dtype=bool)
        segment = points[stop] - points[start]
        squared_length = jnp.sum(segment * segment, axis=-1)
        nondegenerate = (
            squared_length > self.minimum_segment_length_m**2
        )
        safe_squared_length = jnp.where(nondegenerate, squared_length, 1.0)
        segment_length = jnp.where(
            nondegenerate, jnp.sqrt(safe_squared_length), 0.0
        )
        active_segment = route_mask[owner]
        lengths = jnp.zeros((self.route_capacity,), dtype=points.dtype).at[owner].add(
            jnp.where(active_segment, segment_length, 0.0)
        )
        return points, segment, segment_length, lengths

    def lengths(self, configuration: ArrayLike, /) -> Array:
        """Return fixed-capacity route lengths in m; disabled routes are zero."""

        point = self._configuration(configuration)
        return self._length_geometry(point)[3]

    def length_jacobian_operator(
        self, configuration: ArrayLike, /
    ) -> FunctionLinearOperator:
        """Return the matrix-free map from generalized velocity to route rate."""

        point = self._configuration(configuration)
        source = ArraySpace((self.articulation.nv,), dtype=point.dtype)
        target = ArraySpace((self.route_capacity,), dtype=point.dtype)

        def action(generalized_velocity):
            return jax.jvp(self.lengths, (point,), (generalized_velocity,))[1]

        def transpose_action(route_covector):
            return jax.linear_transpose(
                action, jnp.zeros((self.articulation.nv,), dtype=point.dtype)
            )(route_covector)[0]

        return FunctionLinearOperator(
            action,
            source=source,
            target=target,
            transpose_action=transpose_action,
            operator_id=f"{self.prepared_id}:length-jacobian",
        )

    def evaluate(
        self,
        configuration: ArrayLike,
        generalized_velocity: ArrayLike,
        /,
    ) -> FixedBodyRouteEvaluation:
        """Evaluate length and its exact JVP under the articulation velocity."""

        point = self._configuration(configuration)
        velocity = self._velocity(generalized_velocity)
        world, segment, segment_length, lengths = self._length_geometry(point)
        rates = self.length_jacobian_operator(point).mv(velocity)
        owner = jnp.asarray(self.segment_route_indices, dtype=jnp.int32)
        route_mask = jnp.asarray(self.plan.route_mask, dtype=bool)
        segment_finite = (
            jnp.all(jnp.isfinite(segment), axis=-1) & jnp.isfinite(segment_length)
        )
        segment_nondegenerate = segment_length > self.minimum_segment_length_m
        finite_count = jnp.zeros((self.route_capacity,), dtype=jnp.int32).at[owner].add(
            segment_finite.astype(jnp.int32)
        )
        valid_count = jnp.zeros((self.route_capacity,), dtype=jnp.int32).at[owner].add(
            segment_nondegenerate.astype(jnp.int32)
        )
        expected_count = jnp.asarray(
            tuple(
                self.plan.point_offsets[index + 1]
                - self.plan.point_offsets[index]
                - 1
                for index in range(self.route_capacity)
            ),
            dtype=jnp.int32,
        )
        route_finite = (~route_mask) | (
            (finite_count == expected_count)
            & jnp.isfinite(lengths)
            & jnp.isfinite(rates)
        )
        route_nondegenerate = (~route_mask) | (valid_count == expected_count)
        successful = route_finite & route_nondegenerate
        evidence = FixedBodyRouteEvidence(
            route_finite,
            route_nondegenerate,
            successful,
            self.plan.plan_id,
            self.prepared_id,
        )
        return FixedBodyRouteEvaluation(
            world,
            segment,
            segment_length,
            lengths,
            rates,
            evidence,
            self.plan.route_names,
        )

    def tensile_force_pullback(
        self,
        configuration: ArrayLike,
        generalized_velocity: ArrayLike,
        tensile_force_N: ArrayLike,
        /,
    ) -> tuple[Array, FixedBodyRoutePullbackEvidence]:
        r"""Pull positive tension back as $Q=-J_L^T T$ and audit power.

        Positive route force is tensile. Consequently the route does work on the
        articulation at ``-T * length_rate``; shortening produces positive power.
        Disabled or inadmissible route rows contribute exactly zero load.
        """

        point = self._configuration(configuration)
        velocity = self._velocity(generalized_velocity)
        tension = jnp.asarray(tensile_force_N, dtype=point.dtype)
        if tension.shape != (self.route_capacity,):
            raise ValueError(
                f"tensile_force_N must have shape {(self.route_capacity,)}."
            )
        evaluation = self.evaluate(point, velocity)
        route_mask = jnp.asarray(self.plan.route_mask, dtype=bool)
        tensile = (~route_mask) | (tension >= 0.0)
        finite_routes = evaluation.evidence.successful & jnp.isfinite(tension)
        active_tension = jnp.where(route_mask & tensile & finite_routes, tension, 0.0)
        operator = self.length_jacobian_operator(point)
        generalized_load = operator.transpose_mv(-active_tension)
        route_power = contract(
            "i,i->", -active_tension, evaluation.route_length_rates_m_per_s
        )
        generalized_power = contract("i,i->", generalized_load, velocity)
        residual = route_power - generalized_power
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(route_power), jnp.abs(generalized_power))
        )
        finite = (
            jnp.all(finite_routes | ~route_mask)
            & jnp.all(jnp.isfinite(generalized_load))
            & jnp.all(
                jnp.isfinite(
                    jnp.stack((route_power, generalized_power, residual, scale))
                )
            )
        )
        tolerance = jnp.finfo(point.dtype).eps * max(
            64, 8 * max(self.articulation.nv, 1) * self.route_capacity
        )
        successful = (
            finite
            & jnp.all(tensile)
            & (jnp.abs(residual) <= tolerance * scale)
        )
        evidence = FixedBodyRoutePullbackEvidence(
            tension,
            evaluation.route_length_rates_m_per_s,
            route_power,
            generalized_power,
            residual,
            scale,
            tensile,
            finite,
            successful,
            self.prepared_id,
        )
        return generalized_load, evidence


__all__ = [
    "FixedBodyRouteEvaluation",
    "FixedBodyRouteEvidence",
    "FixedBodyRoutePlan",
    "FixedBodyRoutePullbackEvidence",
    "PreparedFixedBodyRoute",
]
