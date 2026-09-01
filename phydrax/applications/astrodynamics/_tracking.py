#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._status import AstrodynamicsStatus


TrackingObservable: TypeAlias = Literal[
    "range", "range_rate", "azimuth_elevation", "right_ascension_declination"
]


class TrackingStationCatalog(StrictModule, NonTrainableState):
    position: Array
    velocity: Array
    horizon_elevation: Array
    context: AstrodynamicsContext
    provenance: AstrodynamicsDataProvenance
    station_ids: tuple[str, ...] = eqx.field(static=True)
    catalog_id: str = eqx.field(static=True)

    def __init__(
        self, station_ids, position, velocity, horizon_elevation, context, provenance, /
    ):
        ids = tuple(str(value) for value in station_ids)
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity)
        horizon = jnp.asarray(horizon_elevation)
        if (
            position_.shape != (len(ids), 3)
            or velocity_.shape != position_.shape
            or horizon.shape != (len(ids),)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Tracking station arrays are inconsistent.")
        self.position = position_
        self.velocity = velocity_
        self.horizon_elevation = horizon
        self.context = context
        self.provenance = provenance
        self.station_ids = ids
        self.catalog_id = canonical_fingerprint(
            {
                "kind": "tracking-station-catalog",
                "stations": list(ids),
                "context": context.context_id,
                "provenance": provenance.provenance_id,
            }
        )


class ObservationSchedule(StrictModule, NonTrainableState):
    times: Array
    station_index: Array
    observable_index: Array
    observed: Array
    covariance_root: Array
    mask: Array
    observable_kinds: tuple[TrackingObservable, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        times,
        station_index,
        observable_index,
        observed,
        covariance_root,
        mask,
        observable_kinds,
        /,
    ):
        times_ = jnp.asarray(times)
        stations = jnp.asarray(station_index, dtype=jnp.int32)
        kinds = jnp.asarray(observable_index, dtype=jnp.int32)
        observed_ = jnp.asarray(observed)
        root = jnp.asarray(covariance_root)
        mask_ = jnp.asarray(mask, dtype=bool)
        count = int(times_.size)
        if (
            times_.shape != (count,)
            or stations.shape != (count,)
            or kinds.shape != (count,)
            or observed_.shape[0] != count
            or root.shape[0] != count
            or mask_.shape != (count,)
        ):
            raise ValueError("Observation schedule arrays are inconsistent.")
        self.times = times_
        self.station_index = stations
        self.observable_index = kinds
        self.observed = observed_
        self.covariance_root = root
        self.mask = mask_
        self.observable_kinds = tuple(observable_kinds)
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "observation-schedule",
                "count": count,
                "kinds": list(observable_kinds),
            }
        )


class TrackingObservationResult(StrictModule):
    predicted: Array
    jacobian: Array
    available: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class TrackingObservationPlan(StrictModule, NonTrainableState):
    stations: TrackingStationCatalog
    schedule: ObservationSchedule
    plan_id: str = eqx.field(static=True)

    def __init__(self, stations, schedule, /):
        self.stations = stations
        self.schedule = schedule
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tracking-observation-plan",
                "stations": stations.catalog_id,
                "schedule": schedule.schedule_id,
            }
        )

    def _predict_one(
        self,
        state: Array,
        station: Array,
        station_velocity: Array,
        kind: Array,
        horizon: Array,
        /,
    ):
        relative = state[:3] - station
        relative_velocity = state[3:] - station_velocity
        distance = jnp.sqrt(jnp.sum(relative * relative))
        unit = relative / jnp.where(distance > 0.0, distance, 1.0)
        range_rate = jnp.sum(unit * relative_velocity)
        right_ascension = jnp.mod(jnp.arctan2(relative[1], relative[0]), 2.0 * jnp.pi)
        declination = jnp.arcsin(jnp.clip(unit[2], -1.0, 1.0))
        azimuth = right_ascension
        elevation = declination
        predictions = jnp.asarray(
            (
                (distance, 0.0),
                (range_rate, 0.0),
                (azimuth, elevation),
                (right_ascension, declination),
            )
        )
        available = elevation >= horizon
        return predictions[kind], available & (distance > 0.0)

    def evaluate(self, spacecraft_states: ArrayLike, /) -> TrackingObservationResult:
        states = jnp.asarray(spacecraft_states)
        count = int(self.schedule.times.size)
        if states.shape != (count, 6):
            raise ValueError("Spacecraft states must have shape (num_observations,6).")
        stations = self.stations.position[self.schedule.station_index]
        velocities = self.stations.velocity[self.schedule.station_index]
        horizons = self.stations.horizon_elevation[self.schedule.station_index]
        predicted, available = jax.vmap(self._predict_one)(
            states, stations, velocities, self.schedule.observable_index, horizons
        )
        predict_only = lambda state, station, velocity, kind, horizon: self._predict_one(
            state, station, velocity, kind, horizon
        )[0]
        jacobian = jax.vmap(jax.jacfwd(predict_only, argnums=0))(
            states, stations, velocities, self.schedule.observable_index, horizons
        )
        finite = jnp.all(jnp.isfinite(predicted), axis=-1)
        valid = finite & (~self.schedule.mask | available)
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            jnp.where(
                available,
                int(AstrodynamicsStatus.NONFINITE_INPUT),
                int(AstrodynamicsStatus.NO_SOLUTION),
            ),
        ).astype(jnp.int32)
        return TrackingObservationResult(
            predicted, jacobian, available, valid, status, self.plan_id
        )


__all__ = [
    "ObservationSchedule",
    "TrackingObservable",
    "TrackingObservationPlan",
    "TrackingObservationResult",
    "TrackingStationCatalog",
]
