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
    start_time: Array
    end_time: Array
    power: Array
    work: Array
    force_balance_residual: Array
    torque_balance_residual: Array
    finite: Array
    successful: Array
    record_id: str = eqx.field(static=True)


class HydrodynamicLoadPlan(StrictModule, NonTrainableState):
    body_ids: Array
    ambient_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        body_ids: ArrayLike,
        ambient_dimension: int,
        /,
        *,
        tolerance: float = 1.0e-9,
    ):
        ids = np.asarray(body_ids)
        dimension = int(ambient_dimension)
        tolerance_ = float(tolerance)
        if (
            ids.ndim != 1
            or ids.size == 0
            or np.unique(ids).size != ids.size
            or dimension not in (2, 3)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Hydrodynamic load plan is invalid.")
        self.body_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.ambient_dimension = dimension
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydrodynamic-load-plan",
                "body_ids": ids.tolist(),
                "ambient_dimension": dimension,
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
        pressure_force: ArrayLike,
        pressure_torque: ArrayLike,
        viscous_force: ArrayLike,
        viscous_torque: ArrayLike,
        marker_force: ArrayLike,
        marker_torque: ArrayLike,
        lubrication_force: ArrayLike,
        lubrication_torque: ArrayLike,
        contact_impulse: ArrayLike,
        contact_angular_impulse: ArrayLike,
    ) -> HydrodynamicLoadRecord:
        count = int(self.body_ids.size)
        vector_shape = (count, self.ambient_dimension)
        angular_shape = (count, 1 if self.ambient_dimension == 2 else 3)
        velocity_ = jnp.asarray(velocity)
        angular = jnp.asarray(angular_velocity, dtype=velocity_.dtype)
        pressure = jnp.asarray(pressure_force, dtype=velocity_.dtype)
        pressure_moment = jnp.asarray(pressure_torque, dtype=velocity_.dtype)
        viscous = jnp.asarray(viscous_force, dtype=velocity_.dtype)
        viscous_moment = jnp.asarray(viscous_torque, dtype=velocity_.dtype)
        marker = jnp.asarray(marker_force, dtype=velocity_.dtype)
        marker_moment = jnp.asarray(marker_torque, dtype=velocity_.dtype)
        lubrication = jnp.asarray(lubrication_force, dtype=velocity_.dtype)
        lubrication_moment = jnp.asarray(lubrication_torque, dtype=velocity_.dtype)
        impulse = jnp.asarray(contact_impulse, dtype=velocity_.dtype)
        angular_impulse = jnp.asarray(contact_angular_impulse, dtype=velocity_.dtype)
        vectors = (pressure, viscous, marker, lubrication, impulse)
        moments = (
            pressure_moment,
            viscous_moment,
            marker_moment,
            lubrication_moment,
            angular_impulse,
        )
        if velocity_.shape != vector_shape or angular.shape != angular_shape:
            raise ValueError("Hydrodynamic load velocities have incompatible shapes.")
        if any(value.shape != vector_shape for value in vectors) or any(
            value.shape != angular_shape for value in moments
        ):
            raise ValueError(
                "Hydrodynamic force/torque components have incompatible shapes."
            )
        start = jnp.asarray(start_time)
        end = jnp.asarray(end_time)
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
            start,
            end,
            power,
            work,
            force_residual,
            torque_residual,
            finite,
            successful,
            canonical_fingerprint(
                {"kind": "hydrodynamic-load-record", "plan": self.plan_id}
            ),
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
