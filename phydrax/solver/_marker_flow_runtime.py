#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_signature, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


MarkerFlowArtifactKind = Literal["checkpoint", "trajectory", "output", "benchmark"]
MarkerFlowStepLimiter = Literal[
    "advection",
    "diffusion",
    "marker",
    "contact",
    "lubrication",
    "geometry",
    "stochastic",
    "maximum",
]


class HydrodynamicLoadRecord(StrictModule):
    """Interval load with explicit channel availability and immutable provenance."""

    body_ids: Array
    force: Array
    torque: Array
    pressure_force: Array
    pressure_torque: Array
    viscous_force: Array
    viscous_torque: Array
    marker_force: Array
    marker_torque: Array
    lubrication_force: Array
    lubrication_torque: Array
    contact_impulse: Array
    contact_angular_impulse: Array
    pressure_available: Array
    viscous_available: Array
    marker_available: Array
    lubrication_available: Array
    contact_available: Array
    start_time: Array
    end_time: Array
    power: Array
    work: Array
    force_balance_residual: Array
    torque_balance_residual: Array
    finite: Array
    successful: Array
    marker_set_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    reference_point_id: str = eqx.field(static=True)
    interval_id: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)


class HydrodynamicLoadPlan(StrictModule, NonTrainableState):
    body_ids: Array
    ambient_dimension: int = eqx.field(static=True)
    marker_set_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    reference_point_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        body_ids: ArrayLike,
        ambient_dimension: int,
        /,
        *,
        marker_set_id: str,
        geometry_id: str,
        route_id: str,
        topology_epoch_id: str,
        reference_point_id: str,
        tolerance: float = 1.0e-9,
    ):
        ids = np.asarray(body_ids)
        dimension = int(ambient_dimension)
        tolerance_ = float(tolerance)
        provenance_values = (
            marker_set_id,
            geometry_id,
            route_id,
            topology_epoch_id,
            reference_point_id,
        )
        if any(not isinstance(value, str) for value in provenance_values):
            raise TypeError("Hydrodynamic load provenance IDs must be strings.")
        provenance = tuple(provenance_values)
        if (
            ids.ndim != 1
            or ids.size == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or np.unique(ids).size != ids.size
            or dimension not in (2, 3)
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or any(not value or value != value.strip() for value in provenance)
        ):
            raise ValueError("Hydrodynamic load plan is invalid.")
        self.body_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.ambient_dimension = dimension
        (
            self.marker_set_id,
            self.geometry_id,
            self.route_id,
            self.topology_epoch_id,
            self.reference_point_id,
        ) = provenance
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydrodynamic-load-plan",
                "body_ids": ids.tolist(),
                "ambient_dimension": dimension,
                "marker_set_id": self.marker_set_id,
                "geometry_id": self.geometry_id,
                "route_id": self.route_id,
                "topology_epoch_id": self.topology_epoch_id,
                "reference_point_id": self.reference_point_id,
                "tolerance": tolerance_,
            }
        )

    def record(
        self,
        start_time: ArrayLike,
        end_time: ArrayLike,
        velocity: ArrayLike,
        angular_velocity: ArrayLike,
        /,
        *,
        interval_id: str,
        pressure_force: ArrayLike | None = None,
        pressure_torque: ArrayLike | None = None,
        viscous_force: ArrayLike | None = None,
        viscous_torque: ArrayLike | None = None,
        marker_force: ArrayLike | None = None,
        marker_torque: ArrayLike | None = None,
        lubrication_force: ArrayLike | None = None,
        lubrication_torque: ArrayLike | None = None,
        contact_impulse: ArrayLike | None = None,
        contact_angular_impulse: ArrayLike | None = None,
        pressure_available: ArrayLike | None = None,
        viscous_available: ArrayLike | None = None,
        marker_available: ArrayLike | None = None,
        lubrication_available: ArrayLike | None = None,
        contact_available: ArrayLike | None = None,
    ) -> HydrodynamicLoadRecord:
        if not isinstance(interval_id, str):
            raise TypeError("interval_id must be a string.")
        interval = interval_id
        if not interval or interval != interval.strip():
            raise ValueError("interval_id must be a non-empty canonical identifier.")
        count = int(self.body_ids.size)
        vector_shape = (count, self.ambient_dimension)
        angular_shape = (count, 1 if self.ambient_dimension == 2 else 3)
        velocity_ = jnp.asarray(velocity)
        angular = jnp.asarray(angular_velocity, dtype=velocity_.dtype)
        if velocity_.shape != vector_shape or angular.shape != angular_shape:
            raise ValueError("Hydrodynamic load velocities have incompatible shapes.")

        def channel(
            vector_value: ArrayLike | None,
            moment_value: ArrayLike | None,
            availability_value: ArrayLike | None,
            name: str,
        ) -> tuple[Array, Array, Array]:
            if (vector_value is None) != (moment_value is None):
                raise ValueError(f"{name} force and torque must be supplied together.")
            supplied = vector_value is not None
            vector = (
                jnp.zeros(vector_shape, dtype=velocity_.dtype)
                if vector_value is None
                else jnp.asarray(vector_value, dtype=velocity_.dtype)
            )
            moment = (
                jnp.zeros(angular_shape, dtype=velocity_.dtype)
                if moment_value is None
                else jnp.asarray(moment_value, dtype=velocity_.dtype)
            )
            if vector.shape != vector_shape or moment.shape != angular_shape:
                raise ValueError(
                    f"Hydrodynamic {name} force/torque have incompatible shapes."
                )
            if availability_value is None:
                available = jnp.full((count,), supplied, dtype=bool)
            else:
                raw_available = jnp.asarray(availability_value, dtype=bool)
                if raw_available.shape == ():
                    available = jnp.broadcast_to(raw_available, (count,))
                elif raw_available.shape == (count,):
                    available = raw_available
                else:
                    raise ValueError(
                        f"Hydrodynamic {name} availability must be scalar or per body."
                    )
            if not supplied:
                available = eqx.error_if(
                    available,
                    jnp.any(available),
                    f"Available hydrodynamic {name} data must be supplied explicitly.",
                )
            vector = eqx.error_if(
                vector,
                jnp.any(~available[:, None] & (vector != 0.0)),
                f"Unavailable hydrodynamic {name} force must be exactly zero.",
            )
            moment = eqx.error_if(
                moment,
                jnp.any(~available[:, None] & (moment != 0.0)),
                f"Unavailable hydrodynamic {name} torque must be exactly zero.",
            )
            return vector, moment, available

        pressure, pressure_moment, pressure_known = channel(
            pressure_force,
            pressure_torque,
            pressure_available,
            "pressure",
        )
        viscous, viscous_moment, viscous_known = channel(
            viscous_force,
            viscous_torque,
            viscous_available,
            "viscous",
        )
        marker, marker_moment, marker_known = channel(
            marker_force,
            marker_torque,
            marker_available,
            "marker",
        )
        lubrication, lubrication_moment, lubrication_known = channel(
            lubrication_force,
            lubrication_torque,
            lubrication_available,
            "lubrication",
        )
        impulse, angular_impulse, contact_known = channel(
            contact_impulse,
            contact_angular_impulse,
            contact_available,
            "contact",
        )
        vectors = (pressure, viscous, marker, lubrication, impulse)
        moments = (
            pressure_moment,
            viscous_moment,
            marker_moment,
            lubrication_moment,
            angular_impulse,
        )
        start = jnp.asarray(start_time, dtype=velocity_.dtype)
        end = jnp.asarray(end_time, dtype=velocity_.dtype)
        if start.shape != () or end.shape != ():
            raise ValueError("Hydrodynamic load interval endpoints must be scalar.")
        step = end - start
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Hydrodynamic load interval must be positive and finite.",
        )
        resolved_force = pressure + viscous + marker + lubrication
        resolved_torque = (
            pressure_moment + viscous_moment + marker_moment + lubrication_moment
        )
        force = resolved_force + impulse / step
        torque = resolved_torque + angular_impulse / step
        power = jnp.sum(force * velocity_, axis=-1) + jnp.sum(torque * angular, axis=-1)
        work = step * power
        force_residual = force - (resolved_force + impulse / step)
        torque_residual = torque - (resolved_torque + angular_impulse / step)
        all_values = (
            velocity_,
            angular,
            *vectors,
            *moments,
            force,
            torque,
            power,
            work,
        )
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in all_values))
        )
        scale = jnp.maximum(
            1.0,
            jnp.max(jnp.abs(force)) + jnp.max(jnp.abs(torque)),
        )
        successful = (
            finite
            & (jnp.max(jnp.abs(force_residual)) <= self.tolerance * scale)
            & (jnp.max(jnp.abs(torque_residual)) <= self.tolerance * scale)
        )
        record_id = canonical_fingerprint(
            {
                "kind": "hydrodynamic-load-record",
                "plan": self.plan_id,
                "interval_id": interval,
            }
        )
        return HydrodynamicLoadRecord(
            self.body_ids,
            force,
            torque,
            pressure,
            pressure_moment,
            viscous,
            viscous_moment,
            marker,
            marker_moment,
            lubrication,
            lubrication_moment,
            impulse,
            angular_impulse,
            pressure_known,
            viscous_known,
            marker_known,
            lubrication_known,
            contact_known,
            start,
            end,
            power,
            work,
            force_residual,
            torque_residual,
            finite,
            successful,
            self.marker_set_id,
            self.geometry_id,
            self.route_id,
            self.topology_epoch_id,
            self.reference_point_id,
            interval,
            record_id,
        )


class MarkerFlowStepRestriction(StrictModule):
    step_size: Array
    limiter_index: Array
    candidates: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)

    @property
    def limiter(self) -> MarkerFlowStepLimiter:
        names: tuple[MarkerFlowStepLimiter, ...] = (
            "advection",
            "diffusion",
            "marker",
            "contact",
            "lubrication",
            "geometry",
            "stochastic",
            "maximum",
        )
        return names[int(self.limiter_index)]


class MarkerFlowAdaptiveStepPlan(StrictModule, NonTrainableState):
    safety: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        safety: float = 0.8,
        minimum_step: float = 1.0e-12,
        maximum_step: float = 1.0,
    ):
        safety_ = float(safety)
        minimum = float(minimum_step)
        maximum = float(maximum_step)
        if not 0.0 < safety_ <= 1.0 or not 0.0 < minimum <= maximum:
            raise ValueError("Marker-flow adaptive-step policy is invalid.")
        self.safety = safety_
        self.minimum_step = minimum
        self.maximum_step = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "marker-flow-adaptive-step",
                "safety": safety_,
                "minimum": minimum,
                "maximum": maximum,
            }
        )

    def restrict(
        self,
        /,
        *,
        advection: ArrayLike,
        diffusion: ArrayLike,
        marker: ArrayLike,
        contact: ArrayLike,
        lubrication: ArrayLike,
        geometry: ArrayLike,
        stochastic: ArrayLike,
    ) -> MarkerFlowStepRestriction:
        candidates = jnp.asarray(
            (
                advection,
                diffusion,
                marker,
                contact,
                lubrication,
                geometry,
                stochastic,
                self.maximum_step,
            )
        )
        finite = jnp.all(jnp.isfinite(candidates)) & jnp.all(candidates > 0.0)
        limiter = jnp.argmin(candidates)
        selected = self.safety * candidates[limiter]
        valid_minimum = selected >= self.minimum_step
        successful = finite & valid_minimum
        step = jnp.where(successful, selected, self.minimum_step)
        return MarkerFlowStepRestriction(
            step,
            limiter.astype(jnp.int32),
            candidates,
            finite,
            successful,
            self.plan_id,
        )


class MarkerFlowTrajectoryResult(StrictModule):
    time: Array
    observations: object
    accepted: Array
    status: Array
    final_state: object
    finite: Array
    successful: Array
    adapter_id: str = eqx.field(static=True)


class MarkerFlowTrajectoryAdapter(StrictModule, NonTrainableState):
    """Adapter from accepted marker-flow steps to replayable trajectory arrays."""

    step: Callable = eqx.field(static=True)
    observe: Callable = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        step: Callable,
        observe: Callable,
        /,
        *,
        adapter_id: str,
    ):
        if not callable(step) or not callable(observe):
            raise TypeError("Trajectory step and observation must be callable.")
        identifier = str(adapter_id)
        if not identifier:
            raise ValueError("adapter_id must be nonempty.")
        self.step = step
        self.observe = observe
        self.adapter_id = identifier

    def rollout(
        self,
        initial_state,
        initial_time: ArrayLike,
        step_size: ArrayLike,
        event_parameter: ArrayLike,
        stochastic_counter: ArrayLike,
        route_epoch: ArrayLike,
        /,
    ) -> MarkerFlowTrajectoryResult:
        steps = jnp.asarray(step_size)
        events = jnp.asarray(event_parameter)
        counters = jnp.asarray(stochastic_counter)
        routes = jnp.asarray(route_epoch)
        count = int(steps.size)
        if (
            count <= 0
            or counters.shape != (count,)
            or routes.shape != (count,)
            or events.shape[0] != count
        ):
            raise ValueError("Trajectory schedule arrays are incompatible.")
        state = initial_state
        time = jnp.asarray(initial_time)
        observations = []
        accepted_values = []
        status_values = []
        finite = jnp.asarray(True)
        times = []
        for index in range(count):
            candidate, accepted, status = self.step(
                state,
                steps[index],
                events[index],
                counters[index],
                routes[index],
            )
            state = jax.tree.map(
                lambda proposed, previous, accepted_=accepted: jnp.where(
                    accepted_, proposed, previous
                ),
                candidate,
                state,
            )
            time = jnp.where(accepted, time + steps[index], time)
            observation = self.observe(time, state)
            finite = finite & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value))
                        for value in jax.tree.leaves(observation)
                    )
                )
            )
            observations.append(observation)
            accepted_values.append(accepted)
            status_values.append(status)
            times.append(time)
        stacked_observations = jax.tree.map(
            lambda *values: jnp.stack(values), *observations
        )
        accepted_array = jnp.stack(accepted_values)
        status_array = jnp.stack(status_values)
        return MarkerFlowTrajectoryResult(
            jnp.stack(times),
            stacked_observations,
            accepted_array,
            status_array,
            state,
            finite,
            finite & jnp.all(accepted_array),
            self.adapter_id,
        )


class MarkerFlowArtifactReference(StrictModule, NonTrainableState):
    path: str = eqx.field(static=True)
    kind: MarkerFlowArtifactKind = eqx.field(static=True)
    identity: str = eqx.field(static=True)
    sha256: str = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)


def marker_flow_artifact_reference(
    path: str | Path,
    kind: MarkerFlowArtifactKind,
    identity: str,
    /,
) -> MarkerFlowArtifactReference:
    target = Path(path)
    if kind not in ("checkpoint", "trajectory", "output", "benchmark"):
        raise ValueError("Unknown marker-flow artifact kind.")
    identifier = str(identity)
    if not identifier or not target.is_file():
        raise ValueError("Marker-flow artifact identity/path is invalid.")
    payload = target.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return MarkerFlowArtifactReference(
        str(target),
        kind,
        identifier,
        digest,
        len(payload),
        canonical_fingerprint(
            {
                "kind": "marker-flow-artifact-reference",
                "artifact_kind": kind,
                "identity": identifier,
                "sha256": digest,
                "byte_count": len(payload),
            }
        ),
    )


class MarkerFlowCompiledExportReport(StrictModule, NonTrainableState):
    state_signature_matches: bool = eqx.field(static=True)
    fixed_routes: bool = eqx.field(static=True)
    fixed_topology: bool = eqx.field(static=True)
    fixed_random_schedule: bool = eqx.field(static=True)
    exportable: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class MarkerFlowCompiledExportPlan(StrictModule, NonTrainableState):
    state_signature: object = eqx.field(static=True)
    fixed_routes: bool = eqx.field(static=True)
    fixed_topology: bool = eqx.field(static=True)
    fixed_random_schedule: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_template,
        /,
        *,
        fixed_routes: bool,
        fixed_topology: bool,
        fixed_random_schedule: bool,
    ):
        signature = array_tree_signature(state_template)
        self.state_signature = signature
        self.fixed_routes = bool(fixed_routes)
        self.fixed_topology = bool(fixed_topology)
        self.fixed_random_schedule = bool(fixed_random_schedule)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "marker-flow-compiled-export",
                "state_signature": signature,
                "fixed_routes": fixed_routes,
                "fixed_topology": fixed_topology,
                "fixed_random_schedule": fixed_random_schedule,
            }
        )

    def validate(self, state, /) -> MarkerFlowCompiledExportReport:
        matches = array_tree_signature(state) == self.state_signature
        exportable = (
            matches
            and self.fixed_routes
            and self.fixed_topology
            and self.fixed_random_schedule
        )
        return MarkerFlowCompiledExportReport(
            matches,
            self.fixed_routes,
            self.fixed_topology,
            self.fixed_random_schedule,
            exportable,
            self.plan_id,
        )


__all__ = [
    "HydrodynamicLoadPlan",
    "HydrodynamicLoadRecord",
    "MarkerFlowAdaptiveStepPlan",
    "MarkerFlowArtifactKind",
    "MarkerFlowArtifactReference",
    "MarkerFlowCompiledExportPlan",
    "MarkerFlowCompiledExportReport",
    "MarkerFlowStepLimiter",
    "MarkerFlowStepRestriction",
    "MarkerFlowTrajectoryAdapter",
    "MarkerFlowTrajectoryResult",
    "marker_flow_artifact_reference",
]
