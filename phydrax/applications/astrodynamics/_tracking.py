#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
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
        ids = tuple(str(value).strip() for value in station_ids)
        position_ = np.asarray(position, dtype=np.float64)
        velocity_ = np.asarray(velocity, dtype=np.float64)
        horizon = np.asarray(horizon_elevation, dtype=np.float64)
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        if not isinstance(provenance, AstrodynamicsDataProvenance):
            raise TypeError("provenance must be AstrodynamicsDataProvenance.")
        if (
            not ids
            or any(not value for value in ids)
            or position_.shape != (len(ids), 3)
            or velocity_.shape != position_.shape
            or horizon.shape != (len(ids),)
            or len(set(ids)) != len(ids)
            or np.any(~np.isfinite(position_))
            or np.any(~np.isfinite(velocity_))
            or np.any(~np.isfinite(horizon))
        ):
            raise ValueError("Tracking station arrays are inconsistent or nonfinite.")
        self.position = jnp.asarray(position_)
        self.velocity = jnp.asarray(velocity_)
        self.horizon_elevation = jnp.asarray(horizon)
        self.context = context
        self.provenance = provenance
        self.station_ids = ids
        self.catalog_id = canonical_fingerprint(
            {
                "kind": "tracking-station-catalog",
                "stations": list(ids),
                "arrays": array_tree_fingerprint((position_, velocity_, horizon)),
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
        times_ = np.asarray(times, dtype=np.float64)
        stations = np.asarray(station_index)
        kinds = np.asarray(observable_index)
        observed_ = np.asarray(observed, dtype=np.float64)
        root = np.asarray(covariance_root, dtype=np.float64)
        mask_ = np.asarray(mask)
        labels = tuple(observable_kinds)
        catalog = (
            "range",
            "range_rate",
            "azimuth_elevation",
            "right_ascension_declination",
        )
        count = times_.size
        if (
            times_.ndim != 1
            or count < 1
            or stations.shape != (count,)
            or kinds.shape != (count,)
            or observed_.shape != (count, 2)
            or root.shape != (count, 2, 2)
            or mask_.shape != (count,)
            or not np.issubdtype(stations.dtype, np.integer)
            or not np.issubdtype(kinds.dtype, np.integer)
            or not np.issubdtype(mask_.dtype, np.bool_)
            or not labels
            or len(set(labels)) != len(labels)
            or any(label not in catalog for label in labels)
            or np.any(kinds < 0)
            or np.any(kinds >= len(labels))
            or np.any(~np.isfinite(times_))
            or np.any(np.diff(times_) < 0.0)
            or np.any(~np.isfinite(observed_))
            or np.any(~np.isfinite(root))
            or not np.allclose(root, np.tril(root))
            or np.any(np.diagonal(root, axis1=-2, axis2=-1) <= 0.0)
        ):
            raise ValueError(
                "Observation schedule arrays or observable catalog are invalid."
            )
        kind_codes = np.asarray(
            [catalog.index(labels[int(index)]) for index in kinds],
            dtype=np.int32,
        )
        self.times = jnp.asarray(times_)
        self.station_index = jnp.asarray(stations, dtype=jnp.int32)
        self.observable_index = jnp.asarray(kind_codes)
        self.observed = jnp.asarray(observed_)
        self.covariance_root = jnp.asarray(root)
        self.mask = jnp.asarray(mask_, dtype=jnp.bool_)
        self.observable_kinds = catalog
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "observation-schedule",
                "arrays": array_tree_fingerprint(
                    (times_, stations, kind_codes, observed_, root, mask_)
                ),
                "kinds": list(labels),
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
        if not isinstance(stations, TrackingStationCatalog):
            raise TypeError("stations must be a TrackingStationCatalog.")
        if not isinstance(schedule, ObservationSchedule):
            raise TypeError("schedule must be an ObservationSchedule.")
        station_indices = np.asarray(schedule.station_index)
        if np.any(station_indices < 0) or np.any(
            station_indices >= len(stations.station_ids)
        ):
            raise ValueError(
                "Observation schedule references an absent tracking station."
            )
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
        count = self.schedule.times.size
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
        finite = (
            jnp.all(jnp.isfinite(states), axis=-1)
            & jnp.all(jnp.isfinite(predicted), axis=-1)
            & jnp.all(jnp.isfinite(jacobian), axis=(-2, -1))
        )
        valid = finite & (~self.schedule.mask | available)
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                valid,
                int(AstrodynamicsStatus.SUCCESS),
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
