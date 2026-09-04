#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DualSpace,
    FunctionLinearOperator,
)
from ._rod_dynamics import PreparedRod, RodState
from ._rod_reduction import PreparedReducedRod, ReducedRodState


RodPreparation: TypeAlias = PreparedRod | PreparedReducedRod
RodActuationState: TypeAlias = RodState | ReducedRodState


def _finite_pair(name: str, value: tuple[float, float], /) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a pair.")
    lower, upper = float(value[0]), float(value[1])
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError(f"{name} must be finite and ordered.")
    return lower, upper


def _positive_finite(name: str, value: float, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _rotation_matrices(orientations: Array, dimension: int, /) -> Array:
    if dimension == 2:
        cosine = jnp.cos(orientations)
        sine = jnp.sin(orientations)
        return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(
            orientations.shape + (2, 2)
        )
    norm = jnp.sqrt(jnp.sum(orientations * orientations, axis=-1, keepdims=True))
    quaternion = orientations / norm
    scalar = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]
    return jnp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - scalar * z),
            2.0 * (x * z + scalar * y),
            2.0 * (x * y + scalar * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - scalar * x),
            2.0 * (x * z - scalar * y),
            2.0 * (y * z + scalar * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _all_finite(*values: Array) -> Array:
    finite = jnp.asarray(True)
    for value in values:
        finite = finite & jnp.all(jnp.isfinite(value))
    return finite


class RodMaterialStation(StrictModule, NonTrainableState):
    """One fixed tendon eyelet in a discrete rod segment's material frame."""

    offset: Array
    segment_id: int = eqx.field(static=True)
    xi: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    station_id: str = eqx.field(static=True)

    def __init__(self, segment_id: int, xi: float, offset: ArrayLike, /):
        if isinstance(segment_id, bool) or int(segment_id) != segment_id:
            raise TypeError("segment_id must be an integer.")
        segment = int(segment_id)
        if segment < 0:
            raise ValueError("segment_id must be nonnegative.")
        coordinate = float(xi)
        if not isfinite(coordinate) or coordinate < 0.0 or coordinate > 1.0:
            raise ValueError("xi must be finite and lie in [0, 1].")
        offset_ = np.asarray(offset)
        if (
            offset_.ndim != 1
            or offset_.shape[0] not in (2, 3)
            or not np.issubdtype(offset_.dtype, np.floating)
            or np.iscomplexobj(offset_)
            or not np.all(np.isfinite(offset_))
        ):
            raise ValueError("offset must be a finite real vector of length 2 or 3.")
        self.offset = jnp.asarray(offset_)
        self.segment_id = segment
        self.xi = coordinate
        self.dimension = int(offset_.shape[0])
        self.station_id = canonical_fingerprint(
            {
                "kind": "rod-material-tendon-station",
                "segment": segment,
                "xi": coordinate,
                "offset": array_tree_fingerprint(offset_),
            }
        )


class TendonRoutePlan(StrictModule, NonTrainableState):
    """Ordered fixed material eyelets defining one frictionless tendon route."""

    stations: tuple[RodMaterialStation, ...]
    station_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    span_count: int = eqx.field(static=True)
    minimum_span_length: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stations: tuple[RodMaterialStation, ...],
        /,
        *,
        minimum_span_length: float = 1.0e-9,
        label: str | None = None,
    ):
        if not isinstance(stations, tuple) or len(stations) < 2:
            raise ValueError("stations must be a tuple containing at least two eyelets.")
        if not all(isinstance(station, RodMaterialStation) for station in stations):
            raise TypeError("Every route station must be a RodMaterialStation.")
        dimension = stations[0].dimension
        if any(station.dimension != dimension for station in stations):
            raise ValueError("Every route station offset must have the same dimension.")
        minimum = _positive_finite("minimum_span_length", minimum_span_length)
        self.stations = stations
        self.station_count = len(stations)
        self.dimension = dimension
        self.span_count = len(stations) - 1
        self.minimum_span_length = minimum
        self.label = None if label is None else str(label)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-material-eyelet-tendon-route-plan",
                "stations": tuple(station.station_id for station in stations),
                "minimum_span_length": minimum,
            }
        )

    def prepare(self, rod: RodPreparation, /) -> "PreparedTendonRoute":
        return PreparedTendonRoute(self, rod)


class PreparedTendonRoute(StrictModule, NonTrainableState):
    """Fixed-work tendon geometry and exact velocity/effort dual actions."""

    plan: TendonRoutePlan
    rod: PreparedRod
    reduction: PreparedReducedRod | None
    segment_ids: Array
    start_node_ids: Array
    end_node_ids: Array
    xis: Array
    offsets: Array
    span_length_rate_space: ArraySpace
    span_tension_space: DualSpace
    length_rate_space: ArraySpace
    tension_space: DualSpace
    span_count: int = eqx.field(static=True)
    workset_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: TendonRoutePlan, rod: RodPreparation, /):
        if not isinstance(plan, TendonRoutePlan):
            raise TypeError("plan must be a TendonRoutePlan.")
        if isinstance(rod, PreparedReducedRod):
            reduction: PreparedReducedRod | None = rod
            native = rod.rod
        elif isinstance(rod, PreparedRod):
            reduction = None
            native = rod
        else:
            raise TypeError("rod must be a PreparedRod or PreparedReducedRod.")
        if plan.dimension != native.plan.dimension:
            raise ValueError("Route station offsets do not match the rod dimension.")
        segment_ids = np.asarray(
            tuple(station.segment_id for station in plan.stations), dtype=np.int32
        )
        if np.any(segment_ids >= native.plan.segment_count):
            raise ValueError("A route station references a segment outside the rod.")
        node_ids = np.asarray(native.plan.segment_node_ids)[segment_ids]
        dtype = np.dtype(native.plan.rest_positions.dtype)
        xis = np.asarray(tuple(station.xi for station in plan.stations), dtype=dtype)
        offsets = np.stack(
            tuple(np.asarray(station.offset, dtype=dtype) for station in plan.stations)
        )
        workset_id = canonical_fingerprint(
            {
                "kind": "prepared-native-rod-tendon-route-workset",
                "rod_material_worksets": native.material_workset_id,
                "route": plan.plan_id,
                "stations": array_tree_fingerprint(
                    {
                        "segment_ids": segment_ids,
                        "node_ids": node_ids,
                        "xis": xis,
                        "offsets": offsets,
                    }
                ),
            }
        )
        representation_id = "native" if reduction is None else reduction.prepared_id
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-rod-tendon-route",
                "rod": native.prepared_id,
                "reduction": representation_id,
                "route": plan.plan_id,
                "material_worksets": native.material_workset_id,
                "route_workset": workset_id,
            }
        )
        span_length_rate_space = ArraySpace(
            (plan.span_count,),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {"kind": "tendon-span-length-rate-space", "route": prepared_id}
            ),
        )
        length_rate_space = ArraySpace(
            (),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {"kind": "tendon-length-rate-space", "route": prepared_id}
            ),
        )
        self.plan = plan
        self.rod = native
        self.reduction = reduction
        self.segment_ids = jnp.asarray(segment_ids)
        self.start_node_ids = jnp.asarray(node_ids[:, 0])
        self.end_node_ids = jnp.asarray(node_ids[:, 1])
        self.xis = jnp.asarray(xis)
        self.span_count = plan.span_count
        self.offsets = jnp.asarray(offsets)
        self.span_length_rate_space = span_length_rate_space
        self.span_tension_space = DualSpace(
            span_length_rate_space,
            space_id=canonical_fingerprint(
                {"kind": "tendon-span-tension-dual-space", "route": prepared_id}
            ),
        )
        self.length_rate_space = length_rate_space
        self.tension_space = DualSpace(
            length_rate_space,
            space_id=canonical_fingerprint(
                {"kind": "tendon-tension-dual-space", "route": prepared_id}
            ),
        )
        self.workset_id = workset_id
        self.prepared_id = prepared_id

    def _native_state(self, state: RodActuationState, /) -> RodState:
        if self.reduction is None:
            if not isinstance(state, RodState):
                raise TypeError("A native tendon route requires a RodState.")
            return state
        if not isinstance(state, ReducedRodState):
            raise TypeError("A reduced tendon route requires a ReducedRodState.")
        return self.reduction.lift(state)

    def _world_points_from_configuration(
        self, positions: Array, orientations: Array, /
    ) -> tuple[Array, Array]:
        frames = _rotation_matrices(orientations[self.segment_ids], self.plan.dimension)
        centers = (1.0 - self.xis)[:, None] * positions[self.start_node_ids] + self.xis[
            :, None
        ] * positions[self.end_node_ids]
        rotated_offsets = ein.contract("sij,sj->si", frames, self.offsets)
        return centers + rotated_offsets, frames

    def _world_points_from_native(self, state: RodState, /) -> tuple[Array, Array]:
        positions, orientations = self.rod.configuration_from_state(state)
        return self._world_points_from_configuration(positions, orientations)

    def _world_velocities_from_native(
        self, state: RodState, frames: Array | None = None, /
    ) -> Array:
        if frames is None:
            _, frames = self._world_points_from_native(state)
        linear, angular = self.rod.velocity_from_state(state)
        center_velocity = (1.0 - self.xis)[:, None] * linear[
            self.start_node_ids
        ] + self.xis[:, None] * linear[self.end_node_ids]
        rotated_offsets = ein.contract("sij,sj->si", frames, self.offsets)
        if self.plan.dimension == 2:
            rotational = angular[self.segment_ids, None] * jnp.stack(
                (-rotated_offsets[:, 1], rotated_offsets[:, 0]), axis=-1
            )
        else:
            world_angular = ein.contract("sij,sj->si", frames, angular[self.segment_ids])
            rotational = jnp.cross(world_angular, rotated_offsets)
        return center_velocity + rotational

    def world_points(self, state: RodActuationState, /) -> Array:
        """Return ordered world eyelet points with shape ``(stations, d)``."""
        if self.reduction is None:
            if not isinstance(state, RodState):
                raise TypeError("A native tendon route requires a RodState.")
            points, _ = self._world_points_from_native(state)
            return points
        if not isinstance(state, ReducedRodState):
            raise TypeError("A reduced tendon route requires a ReducedRodState.")
        self.reduction.validate_state(state)
        configuration = self.reduction.lift_configuration(state.coefficients)
        points, _ = self._world_points_from_configuration(*configuration)
        return points

    def world_velocities(self, state: RodActuationState, /) -> Array:
        """Return eyelet velocities including exact material-offset transport."""
        native = self._native_state(state)
        _, frames = self._world_points_from_native(native)
        return self._world_velocities_from_native(native, frames)

    def span_geometry(self, state: RodActuationState, /) -> tuple[Array, Array, Array]:
        """Return span vectors, lengths, and unit directions in route order."""
        points = self.world_points(state)
        spans = points[1:] - points[:-1]
        lengths = jnp.sqrt(jnp.sum(spans * spans, axis=-1))
        directions = spans / lengths[:, None]
        return spans, lengths, directions

    def length(self, state: RodActuationState, /) -> Array:
        """Return the deployed geometric length of the piecewise-straight tendon."""
        points = self.world_points(state)
        spans = points[1:] - points[:-1]
        return jnp.sum(jnp.sqrt(jnp.sum(spans * spans, axis=-1)))

    def _native_span_length_pullback(
        self,
        frames: Array,
        directions: Array,
        span_covector: Array,
        /,
    ) -> tuple[Array, Array]:
        dimension = self.plan.dimension
        station_effort = jnp.zeros(
            (self.plan.station_count, dimension), dtype=frames.dtype
        )
        station_effort = station_effort.at[:-1].add(-span_covector[:, None] * directions)
        station_effort = station_effort.at[1:].add(span_covector[:, None] * directions)
        forces = jnp.zeros((self.rod.plan.node_count, dimension), dtype=frames.dtype)
        forces = forces.at[self.start_node_ids].add(
            (1.0 - self.xis)[:, None] * station_effort
        )
        forces = forces.at[self.end_node_ids].add(self.xis[:, None] * station_effort)
        material_effort = ein.contract("sji,sj->si", frames, station_effort)
        if dimension == 2:
            station_moment = (
                self.offsets[:, 0] * material_effort[:, 1]
                - self.offsets[:, 1] * material_effort[:, 0]
            )
            moments = jnp.zeros((self.rod.plan.segment_count,), dtype=frames.dtype)
        else:
            station_moment = jnp.cross(self.offsets, material_effort)
            moments = jnp.zeros((self.rod.plan.segment_count, 3), dtype=frames.dtype)
        moments = moments.at[self.segment_ids].add(station_moment)
        return forces, moments

    def native_span_length_rate_operator(
        self, state: RodState, /
    ) -> AbstractLinearOperator:
        """Linearize every routed span length against native rod velocity."""
        points, frames = self._world_points_from_native(state)
        spans = points[1:] - points[:-1]
        span_lengths = jnp.sqrt(jnp.sum(spans * spans, axis=-1))
        directions = spans / span_lengths[:, None]
        dimension = self.plan.dimension
        xis = self.xis
        start_ids = self.start_node_ids
        end_ids = self.end_node_ids
        segment_ids = self.segment_ids
        offsets = self.offsets

        def action(velocity):
            linear, angular = velocity
            center_velocity = (1.0 - xis)[:, None] * linear[start_ids] + xis[
                :, None
            ] * linear[end_ids]
            rotated_offsets = ein.contract("sij,sj->si", frames, offsets)
            if dimension == 2:
                rotational = angular[segment_ids, None] * jnp.stack(
                    (-rotated_offsets[:, 1], rotated_offsets[:, 0]), axis=-1
                )
            else:
                world_angular = ein.contract("sij,sj->si", frames, angular[segment_ids])
                rotational = jnp.cross(world_angular, rotated_offsets)
            station_velocity = center_velocity + rotational
            return jnp.sum(
                directions * (station_velocity[1:] - station_velocity[:-1]),
                axis=-1,
            )

        def transpose_action(covector):
            return self._native_span_length_pullback(
                frames, directions, jnp.asarray(covector)
            )

        return FunctionLinearOperator(
            action,
            source=self.rod.velocity_space,
            target=self.span_length_rate_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "native-rod-tendon-span-length-rate-action",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def native_length_rate_operator(self, state: RodState, /) -> AbstractLinearOperator:
        """Linearize total route length against native physical rod velocity."""
        span_operator = self.native_span_length_rate_operator(state)

        def action(velocity):
            return jnp.sum(span_operator.mv(velocity))

        def transpose_action(covector):
            repeated = jnp.broadcast_to(jnp.asarray(covector), (self.span_count,))
            return span_operator.transpose_mv(repeated)

        return FunctionLinearOperator(
            action,
            source=self.rod.velocity_space,
            target=self.length_rate_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "native-rod-tendon-length-rate-action",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def reduced_span_length_rate_operator(
        self, state: ReducedRodState, /
    ) -> AbstractLinearOperator:
        """Linearize every routed span directly against the reduced tangent."""
        if self.reduction is None:
            raise TypeError("This tendon route was not prepared against a reduction.")
        self.reduction.validate_state(state)
        configuration = self.reduction.lift_configuration(state.coefficients)
        native = self.rod.state_from_configuration(configuration)
        native_operator = self.native_span_length_rate_operator(native)
        lift = self.reduction.lift_velocity_operator(state.coefficients)

        def action(rates):
            return native_operator.mv(lift.mv(rates))

        def transpose_action(covector):
            return lift.transpose_mv(native_operator.transpose_mv(covector))

        return FunctionLinearOperator(
            action,
            source=self.reduction.coefficient_space,
            target=self.span_length_rate_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-tendon-span-length-rate-action",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "reduction": self.reduction.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def reduced_length_rate_operator(
        self, state: ReducedRodState, /
    ) -> AbstractLinearOperator:
        """Linearize total route length against the reduced rod tangent."""
        if self.reduction is None:
            raise TypeError("This tendon route was not prepared against a reduction.")
        span_operator = self.reduced_span_length_rate_operator(state)

        def action(rates):
            return jnp.sum(span_operator.mv(rates))

        def transpose_action(covector):
            repeated = jnp.broadcast_to(jnp.asarray(covector), (self.span_count,))
            return span_operator.transpose_mv(repeated)

        return FunctionLinearOperator(
            action,
            source=self.reduction.coefficient_space,
            target=self.length_rate_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-tendon-length-rate-action",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "reduction": self.reduction.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def span_length_rate_operator(
        self, state: RodActuationState, /
    ) -> AbstractLinearOperator:
        if self.reduction is None:
            if not isinstance(state, RodState):
                raise TypeError("A native tendon route requires a RodState.")
            return self.native_span_length_rate_operator(state)
        if not isinstance(state, ReducedRodState):
            raise TypeError("A reduced tendon route requires a ReducedRodState.")
        return self.reduced_span_length_rate_operator(state)

    def length_rate_operator(self, state: RodActuationState, /) -> AbstractLinearOperator:
        if self.reduction is None:
            if not isinstance(state, RodState):
                raise TypeError("A native tendon route requires a RodState.")
            return self.native_length_rate_operator(state)
        if not isinstance(state, ReducedRodState):
            raise TypeError("A reduced tendon route requires a ReducedRodState.")
        return self.reduced_length_rate_operator(state)

    def span_length_rates(self, state: RodActuationState, /) -> Array:
        """Return one exact geometric length rate per routed span."""
        operator = self.span_length_rate_operator(state)
        if isinstance(state, RodState):
            tangent = self.rod.velocity_from_state(state)
        else:
            tangent = state.coefficient_velocities
        return operator.mv(tangent)

    def length_rate(self, state: RodActuationState, /) -> Array:
        operator = self.length_rate_operator(state)
        if isinstance(state, RodState):
            tangent = self.rod.velocity_from_state(state)
        else:
            tangent = state.coefficient_velocities
        return operator.mv(tangent)

    def native_span_effort_pullback_operator(
        self, state: RodState, /
    ) -> AbstractLinearOperator:
        """Map independently varying span tensions to native rod effort."""
        span_rate = self.native_span_length_rate_operator(state)

        def action(tensions):
            return span_rate.transpose_mv(-tensions)

        def transpose_action(effort_coordinates):
            return -span_rate.mv(effort_coordinates)

        return FunctionLinearOperator(
            action,
            source=self.span_tension_space,
            target=self.rod.effort_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "native-rod-tendon-span-effort-pullback",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def reduced_span_effort_pullback_operator(
        self, state: ReducedRodState, /
    ) -> AbstractLinearOperator:
        """Map independently varying span tensions to reduced rod effort."""
        if self.reduction is None:
            raise TypeError("This tendon route was not prepared against a reduction.")
        span_rate = self.reduced_span_length_rate_operator(state)

        def action(tensions):
            return span_rate.transpose_mv(-tensions)

        def transpose_action(effort_coordinates):
            return -span_rate.mv(effort_coordinates)

        return FunctionLinearOperator(
            action,
            source=self.span_tension_space,
            target=self.reduction.reduced_effort_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-tendon-span-effort-pullback",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "reduction": self.reduction.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def native_effort_pullback_operator(
        self, state: RodState, /
    ) -> AbstractLinearOperator:
        """Map positive tendon tension to the declared native rod effort dual."""
        length_rate = self.native_length_rate_operator(state)

        def action(tension):
            return length_rate.transpose_mv(-tension)

        def transpose_action(effort_coordinates):
            return -length_rate.mv(effort_coordinates)

        return FunctionLinearOperator(
            action,
            source=self.tension_space,
            target=self.rod.effort_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "native-rod-tendon-effort-pullback",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def reduced_effort_pullback_operator(
        self, state: ReducedRodState, /
    ) -> AbstractLinearOperator:
        """Map positive tendon tension to the declared reduced rod effort dual."""
        if self.reduction is None:
            raise TypeError("This tendon route was not prepared against a reduction.")
        length_rate = self.reduced_length_rate_operator(state)

        def action(tension):
            return length_rate.transpose_mv(-tension)

        def transpose_action(effort_coordinates):
            return -length_rate.mv(effort_coordinates)

        return FunctionLinearOperator(
            action,
            source=self.tension_space,
            target=self.reduction.reduced_effort_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-tendon-effort-pullback",
                    "route": self.prepared_id,
                    "rod": self.rod.prepared_id,
                    "reduction": self.reduction.prepared_id,
                    "workset": self.workset_id,
                }
            ),
        )

    def native_span_effort(
        self, state: RodState, tensions: ArrayLike, /
    ) -> tuple[Array, Array]:
        effort = self.native_span_effort_pullback_operator(state).mv(
            jnp.asarray(tensions)
        )
        return self.rod.effort_space.validate(effort)

    def reduced_span_effort(
        self, state: ReducedRodState, tensions: ArrayLike, /
    ) -> Array:
        if self.reduction is None:
            raise TypeError("This tendon route was not prepared against a reduction.")
        effort = self.reduced_span_effort_pullback_operator(state).mv(
            jnp.asarray(tensions)
        )
        return self.reduction.reduced_effort_space.validate(effort)

    def native_effort(
        self, state: RodState, tension: ArrayLike, /
    ) -> tuple[Array, Array]:
        effort = self.native_effort_pullback_operator(state).mv(jnp.asarray(tension))
        return self.rod.effort_space.validate(effort)

    def reduced_effort(self, state: ReducedRodState, tension: ArrayLike, /) -> Array:
        if self.reduction is None:
            raise TypeError("This tendon route was not prepared against a reduction.")
        effort = self.reduced_effort_pullback_operator(state).mv(jnp.asarray(tension))
        return self.reduction.reduced_effort_space.validate(effort)


class FrictionlessElasticTendonPlan(StrictModule, NonTrainableState):
    """Unilateral linear-elastic tendon calibration and operating ratings."""

    route: TendonRoutePlan
    stiffness: float = eqx.field(static=True)
    minimum_free_length: float = eqx.field(static=True)
    maximum_free_length: float = eqx.field(static=True)
    minimum_payout_rate: float = eqx.field(static=True)
    maximum_payout_rate: float = eqx.field(static=True)
    minimum_tendon_length: float = eqx.field(static=True)
    maximum_tendon_length: float = eqx.field(static=True)
    maximum_tension: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: TendonRoutePlan,
        stiffness: float,
        /,
        *,
        free_length_bounds: tuple[float, float],
        payout_rate_bounds: tuple[float, float],
        tendon_length_bounds: tuple[float, float],
        maximum_tension: float,
        power_tolerance: float = 1.0e-8,
        label: str | None = None,
    ):
        if not isinstance(route, TendonRoutePlan):
            raise TypeError("route must be a TendonRoutePlan.")
        stiffness_ = _positive_finite("stiffness", stiffness)
        free_lower, free_upper = _finite_pair("free_length_bounds", free_length_bounds)
        payout_lower, payout_upper = _finite_pair(
            "payout_rate_bounds", payout_rate_bounds
        )
        length_lower, length_upper = _finite_pair(
            "tendon_length_bounds", tendon_length_bounds
        )
        if free_lower <= 0.0 or length_lower <= 0.0:
            raise ValueError(
                "Free-length and tendon-length lower bounds must be positive."
            )
        maximum_tension_ = _positive_finite("maximum_tension", maximum_tension)
        tolerance = _positive_finite("power_tolerance", power_tolerance)
        calibration_id = canonical_fingerprint(
            {
                "kind": "frictionless-linear-elastic-tendon-calibration",
                "stiffness": stiffness_,
                "free_length_bounds": (free_lower, free_upper),
                "payout_rate_bounds": (payout_lower, payout_upper),
                "tendon_length_bounds": (length_lower, length_upper),
                "maximum_tension": maximum_tension_,
                "power_tolerance": tolerance,
            }
        )
        self.route = route
        self.stiffness = stiffness_
        self.minimum_free_length = free_lower
        self.maximum_free_length = free_upper
        self.minimum_payout_rate = payout_lower
        self.maximum_payout_rate = payout_upper
        self.minimum_tendon_length = length_lower
        self.maximum_tendon_length = length_upper
        self.maximum_tension = maximum_tension_
        self.power_tolerance = tolerance
        self.label = None if label is None else str(label)
        self.calibration_id = calibration_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "frictionless-elastic-routed-tendon-plan",
                "route": route.plan_id,
                "calibration": calibration_id,
            }
        )

    def prepare(self, rod: RodPreparation, /) -> "PreparedFrictionlessElasticTendon":
        return PreparedFrictionlessElasticTendon(self, self.route.prepare(rod))


class TendonActuatorState(StrictModule):
    """Dynamic deployed stress-free tendon length."""

    free_length: Array

    def __init__(self, free_length: ArrayLike, /):
        value = jnp.asarray(free_length)
        if value.shape != ():
            raise ValueError("free_length must be scalar.")
        if not jnp.issubdtype(value.dtype, jnp.inexact) or jnp.iscomplexobj(value):
            raise TypeError("free_length must be a real inexact scalar.")
        self.free_length = value


class TendonPayoutCommand(StrictModule):
    """Signed stress-free-length rate; positive values pay tendon out."""

    payout_rate: Array

    def __init__(self, payout_rate: ArrayLike, /):
        value = jnp.asarray(payout_rate)
        if value.shape != ():
            raise ValueError("payout_rate must be scalar.")
        if not jnp.issubdtype(value.dtype, jnp.inexact) or jnp.iscomplexobj(value):
            raise TypeError("payout_rate must be a real inexact scalar.")
        self.payout_rate = value


class TendonActuationEvaluation(StrictModule):
    """Tendon mechanics, ratings, and instantaneous/discrete power ledger."""

    candidate_state: TendonActuatorState
    station_points: Array
    station_velocities: Array
    span_lengths: Array
    length: Array
    length_rate: Array
    extension: Array
    extension_rate: Array
    tension: Array
    native_forces: Array
    native_moments: Array
    reduced_effort: Array | None
    stored_energy: Array
    stored_energy_rate: Array
    rod_power: Array
    native_rod_power: Array
    reduced_rod_power: Array | None
    spool_power: Array
    virtual_work_residual: Array
    instantaneous_power_residual: Array
    payout_increment: Array
    candidate_extension: Array
    candidate_tension: Array
    candidate_stored_energy: Array
    stored_energy_change: Array
    spool_work: Array
    discrete_energy_residual: Array
    free_length_margin: Array
    candidate_free_length_margin: Array
    payout_rate_margin: Array
    tendon_length_margin: Array
    tension_margin: Array
    candidate_tension_margin: Array
    finite: Array
    nondegenerate: Array
    slack: Array
    taut: Array
    state_within_bounds: Array
    candidate_state_within_bounds: Array
    payout_rate_within_bounds: Array
    tendon_length_within_bounds: Array
    tension_within_bounds: Array
    candidate_tension_within_bounds: Array
    within_rating: Array
    time_step_valid: Array
    power_balanced: Array
    valid: Array
    tendon_id: str = eqx.field(static=True)


class PreparedFrictionlessElasticTendon(StrictModule, NonTrainableState):
    """One calibrated frictionless elastic tendon bound to a fixed rod route."""

    plan: FrictionlessElasticTendonPlan
    route: PreparedTendonRoute
    tendon_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FrictionlessElasticTendonPlan,
        route: PreparedTendonRoute,
        /,
    ):
        if not isinstance(plan, FrictionlessElasticTendonPlan):
            raise TypeError("plan must be a FrictionlessElasticTendonPlan.")
        if not isinstance(route, PreparedTendonRoute):
            raise TypeError("route must be a PreparedTendonRoute.")
        if route.plan.plan_id != plan.route.plan_id:
            raise ValueError("Prepared tendon route does not belong to this tendon plan.")
        self.plan = plan
        self.route = route
        self.tendon_id = canonical_fingerprint(
            {
                "kind": "prepared-frictionless-elastic-rod-tendon",
                "rod": route.rod.prepared_id,
                "reduction": (
                    "native" if route.reduction is None else route.reduction.prepared_id
                ),
                "route": route.plan.plan_id,
                "calibration": plan.calibration_id,
                "material_worksets": route.rod.material_workset_id,
                "route_workset": route.workset_id,
            }
        )

    def initialize_state(self, free_length: ArrayLike, /) -> TendonActuatorState:
        state = TendonActuatorState(free_length)
        if np.dtype(state.free_length.dtype) != self.route.length_rate_space.dtype:
            raise TypeError("Tendon state dtype must match the prepared rod dtype.")
        return state

    def integrate_payout(
        self,
        state: TendonActuatorState,
        command: TendonPayoutCommand,
        time_step: ArrayLike,
        /,
    ) -> TendonActuatorState:
        """Advance deployed free length without clipping or hidden saturation."""
        if not isinstance(state, TendonActuatorState):
            raise TypeError("state must be a TendonActuatorState.")
        if not isinstance(command, TendonPayoutCommand):
            raise TypeError("command must be a TendonPayoutCommand.")
        dtype = self.route.length_rate_space.dtype
        if (
            np.dtype(state.free_length.dtype) != dtype
            or np.dtype(command.payout_rate.dtype) != dtype
        ):
            raise TypeError(
                "Tendon state and payout command must match the prepared rod dtype."
            )
        time_step_ = jnp.asarray(time_step, dtype=dtype)
        if time_step_.shape != ():
            raise ValueError("time_step must be scalar.")
        return TendonActuatorState(state.free_length + time_step_ * command.payout_rate)

    def evaluate(
        self,
        rod_state: RodActuationState,
        state: TendonActuatorState,
        command: TendonPayoutCommand,
        /,
        *,
        time_step: ArrayLike = 0.0,
    ) -> TendonActuationEvaluation:
        """Evaluate unilateral tension, pullbacks, ratings, and the power ledger."""
        if not isinstance(state, TendonActuatorState):
            raise TypeError("state must be a TendonActuatorState.")
        if not isinstance(command, TendonPayoutCommand):
            raise TypeError("command must be a TendonPayoutCommand.")
        dtype = self.route.length_rate_space.dtype
        if (
            np.dtype(state.free_length.dtype) != dtype
            or np.dtype(command.payout_rate.dtype) != dtype
        ):
            raise TypeError(
                "Tendon state and payout command must match the prepared rod dtype."
            )
        native_state = self.route._native_state(rod_state)
        station_points, frames = self.route._world_points_from_native(native_state)
        station_velocities = self.route._world_velocities_from_native(
            native_state, frames
        )
        spans = station_points[1:] - station_points[:-1]
        span_lengths = jnp.sqrt(jnp.sum(spans * spans, axis=-1))
        directions = spans / span_lengths[:, None]
        length = jnp.sum(span_lengths)
        length_rate = jnp.sum(
            directions * (station_velocities[1:] - station_velocities[:-1])
        )
        raw_extension = length - state.free_length
        extension = jnp.maximum(raw_extension, 0.0)
        stiffness = jnp.asarray(self.plan.stiffness, dtype=length.dtype)
        tension = stiffness * extension
        relative_length_rate = length_rate - command.payout_rate
        extension_rate = jnp.where(
            (raw_extension > 0.0)
            | ((raw_extension == 0.0) & (relative_length_rate > 0.0)),
            relative_length_rate,
            0.0,
        )
        stored_energy = 0.5 * stiffness * extension * extension
        stored_energy_rate = tension * extension_rate
        native_forces, native_moments = self.route._native_span_length_pullback(
            frames,
            directions,
            jnp.broadcast_to(-tension, (self.route.span_count,)),
        )
        native_velocity = self.route.rod.velocity_from_state(native_state)
        native_rod_power = self.route.rod.effort_space.pair(
            (native_forces, native_moments), native_velocity
        ).real
        reduced_effort: Array | None
        reduced_rod_power: Array | None
        if self.route.reduction is None:
            reduced_effort = None
            reduced_rod_power = None
            rod_power = native_rod_power
        else:
            if not isinstance(rod_state, ReducedRodState):
                raise TypeError("A reduced tendon route requires a ReducedRodState.")
            reduced_effort = self.route.reduction.lift_effort_pullback_operator(
                rod_state.coefficients
            ).mv((native_forces, native_moments))
            reduced_rod_power = self.route.reduction.reduced_effort_space.pair(
                reduced_effort, rod_state.coefficient_velocities
            ).real
            rod_power = reduced_rod_power
        spool_power = tension * command.payout_rate
        virtual_work_residual = rod_power + tension * length_rate
        instantaneous_power_residual = stored_energy_rate + rod_power + spool_power

        time_step_ = jnp.asarray(time_step, dtype=length.dtype)
        if time_step_.shape != ():
            raise ValueError("time_step must be scalar.")
        payout_increment = time_step_ * command.payout_rate
        candidate_state = TendonActuatorState(state.free_length + payout_increment)
        candidate_extension = jnp.maximum(length - candidate_state.free_length, 0.0)
        candidate_tension = stiffness * candidate_extension
        candidate_stored_energy = (
            0.5 * stiffness * candidate_extension * candidate_extension
        )
        stored_energy_change = candidate_stored_energy - stored_energy
        # Integrate the piecewise-linear tension exactly over the fixed-geometry
        # payout substep, including either direction of a taut/slack crossing.
        spool_work = (
            0.5 * (tension + candidate_tension) * (extension - candidate_extension)
        )
        discrete_energy_residual = stored_energy_change + spool_work

        free_length_margin = jnp.minimum(
            state.free_length - self.plan.minimum_free_length,
            self.plan.maximum_free_length - state.free_length,
        )
        candidate_free_length_margin = jnp.minimum(
            candidate_state.free_length - self.plan.minimum_free_length,
            self.plan.maximum_free_length - candidate_state.free_length,
        )
        payout_rate_margin = jnp.minimum(
            command.payout_rate - self.plan.minimum_payout_rate,
            self.plan.maximum_payout_rate - command.payout_rate,
        )
        tendon_length_margin = jnp.minimum(
            length - self.plan.minimum_tendon_length,
            self.plan.maximum_tendon_length - length,
        )
        tension_margin = self.plan.maximum_tension - tension
        candidate_tension_margin = self.plan.maximum_tension - candidate_tension
        state_within_bounds = free_length_margin >= 0.0
        candidate_state_within_bounds = candidate_free_length_margin >= 0.0
        payout_rate_within_bounds = payout_rate_margin >= 0.0
        tendon_length_within_bounds = tendon_length_margin >= 0.0
        tension_within_bounds = tension_margin >= 0.0
        candidate_tension_within_bounds = candidate_tension_margin >= 0.0
        nondegenerate = jnp.all(span_lengths >= self.route.plan.minimum_span_length)
        slack = raw_extension <= 0.0
        taut = raw_extension > 0.0
        within_rating = (
            state_within_bounds
            & candidate_state_within_bounds
            & payout_rate_within_bounds
            & tendon_length_within_bounds
            & tension_within_bounds
            & candidate_tension_within_bounds
        )
        time_step_valid = jnp.isfinite(time_step_) & (time_step_ >= 0.0)
        finite_values = [
            station_points,
            station_velocities,
            span_lengths,
            length,
            length_rate,
            state.free_length,
            command.payout_rate,
            extension,
            extension_rate,
            tension,
            native_forces,
            native_moments,
            stored_energy,
            stored_energy_rate,
            rod_power,
            native_rod_power,
            spool_power,
            virtual_work_residual,
            instantaneous_power_residual,
            payout_increment,
            candidate_state.free_length,
            candidate_extension,
            candidate_tension,
            candidate_stored_energy,
            stored_energy_change,
            spool_work,
            discrete_energy_residual,
        ]
        if reduced_effort is not None and reduced_rod_power is not None:
            finite_values.extend((reduced_effort, reduced_rod_power))
        finite = _all_finite(*finite_values)
        tolerance = jnp.asarray(self.plan.power_tolerance, dtype=length.dtype)
        power_scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(stored_energy_rate),
                jnp.maximum(jnp.abs(rod_power), jnp.abs(spool_power)),
            ),
        )
        discrete_scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(stored_energy_change), jnp.abs(spool_work)),
        )
        power_balanced = (
            (jnp.abs(virtual_work_residual) <= tolerance * power_scale)
            & (jnp.abs(instantaneous_power_residual) <= tolerance * power_scale)
            & (jnp.abs(discrete_energy_residual) <= tolerance * discrete_scale)
        )
        valid = finite & nondegenerate & within_rating & time_step_valid & power_balanced
        return TendonActuationEvaluation(
            candidate_state,
            station_points,
            station_velocities,
            span_lengths,
            length,
            length_rate,
            extension,
            extension_rate,
            tension,
            native_forces,
            native_moments,
            reduced_effort,
            stored_energy,
            stored_energy_rate,
            rod_power,
            native_rod_power,
            reduced_rod_power,
            spool_power,
            virtual_work_residual,
            instantaneous_power_residual,
            payout_increment,
            candidate_extension,
            candidate_tension,
            candidate_stored_energy,
            stored_energy_change,
            spool_work,
            discrete_energy_residual,
            free_length_margin,
            candidate_free_length_margin,
            payout_rate_margin,
            tendon_length_margin,
            tension_margin,
            candidate_tension_margin,
            finite,
            nondegenerate,
            slack,
            taut,
            state_within_bounds,
            candidate_state_within_bounds,
            payout_rate_within_bounds,
            tendon_length_within_bounds,
            tension_within_bounds,
            candidate_tension_within_bounds,
            within_rating,
            time_step_valid,
            power_balanced,
            valid,
            self.tendon_id,
        )


def prepare_tendon_route(
    plan: TendonRoutePlan, rod: RodPreparation, /
) -> PreparedTendonRoute:
    """Bind a fixed material-eyelet route to native or fixed-base reduced rod mechanics."""
    return PreparedTendonRoute(plan, rod)


def prepare_frictionless_elastic_tendon(
    plan: FrictionlessElasticTendonPlan, rod: RodPreparation, /
) -> PreparedFrictionlessElasticTendon:
    """Bind one calibrated tendon, its route, rod, reduction, and worksets."""
    return plan.prepare(rod)


def integrate_tendon_payout(
    prepared: PreparedFrictionlessElasticTendon,
    state: TendonActuatorState,
    command: TendonPayoutCommand,
    time_step: ArrayLike,
    /,
) -> TendonActuatorState:
    """Advance only the deployed stress-free length without saturation."""
    if not isinstance(prepared, PreparedFrictionlessElasticTendon):
        raise TypeError("prepared must be a PreparedFrictionlessElasticTendon.")
    return prepared.integrate_payout(state, command, time_step)


def evaluate_tendon_actuation(
    prepared: PreparedFrictionlessElasticTendon,
    rod_state: RodActuationState,
    state: TendonActuatorState,
    command: TendonPayoutCommand,
    /,
    *,
    time_step: ArrayLike = 0.0,
) -> TendonActuationEvaluation:
    """Evaluate routed unilateral tendon mechanics and conservation evidence."""
    if not isinstance(prepared, PreparedFrictionlessElasticTendon):
        raise TypeError("prepared must be a PreparedFrictionlessElasticTendon.")
    return prepared.evaluate(rod_state, state, command, time_step=time_step)


__all__ = [
    "FrictionlessElasticTendonPlan",
    "PreparedFrictionlessElasticTendon",
    "PreparedTendonRoute",
    "RodMaterialStation",
    "TendonActuationEvaluation",
    "TendonActuatorState",
    "TendonPayoutCommand",
    "TendonRoutePlan",
    "evaluate_tendon_actuation",
    "integrate_tendon_payout",
    "prepare_frictionless_elastic_tendon",
    "prepare_tendon_route",
]
