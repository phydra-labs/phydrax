#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""CSF Taylor--Hood preparation, PVS network flow, and transport schedules."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PreparedMetricNetwork,
    PreparedMixedFiniteElementConstraint,
    PressureGaugePolicy,
)
from ...equations.fem import stokes_form
from ...linalg import ArraySpace, FunctionLinearOperator, LinearSystem, solve


class PreparedTaylorHoodCSFFlow(StrictModule):
    mixed: PreparedMixedFiniteElementConstraint
    plan_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class TaylorHoodCSFFlowPlan:
    mesh: CellMesh
    viscosity: float
    gauge: PressureGaugePolicy
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        viscosity = float(self.viscosity)
        if not np.isfinite(viscosity) or viscosity <= 0.0:
            raise ValueError("viscosity must be finite and positive.")
        if not isinstance(self.gauge, PressureGaugePolicy) or self.gauge.mode == "none":
            raise ValueError("Taylor--Hood CSF flow requires an explicit pressure gauge.")
        object.__setattr__(self, "viscosity", viscosity)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "taylor-hood-csf-flow-plan",
                    "mesh": self.mesh.mesh_id,
                    "viscosity": viscosity.hex(),
                    "gauge": self.gauge.gauge_id,
                }
            ),
        )

    def prepare(self) -> PreparedTaylorHoodCSFFlow:
        mixed = MixedFiniteElementConstraintPlan(
            self.mesh,
            self.gauge,
            displacement_field="velocity",
            pressure_field="pressure",
        ).prepare(
            stokes_form(
                "velocity",
                "pressure",
                self.viscosity,
                form_id=f"{self.plan_id}:stokes",
            )
        )
        return PreparedTaylorHoodCSFFlow(mixed, self.plan_id)


class PVSFlowEvidence(StrictModule):
    branch_balance: Array
    boundary_pressure_residual: Array
    finite: Array
    successful: Array


class PVSFlowResult(StrictModule):
    pressure: Array
    volume_flow: Array
    evidence: PVSFlowEvidence
    plan_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class PVSNetworkFlowPlan:
    network: PreparedMetricNetwork
    hydraulic_conductance: np.ndarray
    boundary_node_indices: np.ndarray
    boundary_pressures: np.ndarray
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.network, PreparedMetricNetwork):
            raise TypeError("network must be PreparedMetricNetwork.")
        edge_count = len(self.network.lengths)
        conductance = np.asarray(self.hydraulic_conductance, dtype=float)
        indices = np.asarray(self.boundary_node_indices)
        pressure = np.asarray(self.boundary_pressures, dtype=float)
        if (
            conductance.shape != (edge_count,)
            or np.any(~np.isfinite(conductance))
            or np.any(conductance <= 0.0)
        ):
            raise ValueError(
                "hydraulic_conductance must be positive and finite per edge."
            )
        if not np.issubdtype(indices.dtype, np.integer) or indices.ndim != 1:
            raise TypeError("boundary_node_indices must be one integer vector.")
        if pressure.shape != indices.shape or np.any(~np.isfinite(pressure)):
            raise ValueError("boundary_pressures must be finite per boundary node.")
        if (
            np.unique(indices).size != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(self.network.node_measures))
        ):
            raise ValueError("Boundary node indices must be unique and in bounds.")
        if len(indices) < 2:
            raise ValueError("Network flow requires at least two pressure boundaries.")
        for name, value in (
            ("hydraulic_conductance", conductance),
            ("boundary_node_indices", indices.astype(np.int32)),
            ("boundary_pressures", pressure),
        ):
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "pvs-network-flow-plan",
                    "network": self.network.network_id,
                    "conductance": array_tree_fingerprint(conductance),
                    "boundary_indices": array_tree_fingerprint(indices),
                    "boundary_pressures": array_tree_fingerprint(pressure),
                }
            ),
        )

    def solve(self) -> PVSFlowResult:
        node_count = len(self.network.node_measures)
        senders = jnp.asarray(self.network.senders, dtype=jnp.int32)
        receivers = jnp.asarray(self.network.receivers, dtype=jnp.int32)
        conductance = jnp.asarray(self.hydraulic_conductance)
        boundary = np.asarray(self.boundary_node_indices)
        free = np.setdiff1d(np.arange(node_count), boundary)
        boundary_pressure = (
            jnp.zeros((node_count,), dtype=conductance.dtype)
            .at[jnp.asarray(boundary)]
            .set(jnp.asarray(self.boundary_pressures))
        )

        def balance_action(pressure):
            volume_flow = conductance * (pressure[senders] - pressure[receivers])
            balance = jnp.zeros((node_count,), dtype=pressure.dtype)
            balance = balance.at[senders].add(volume_flow)
            return balance.at[receivers].add(-volume_flow)

        if free.size:
            free_indices = jnp.asarray(free, dtype=jnp.int32)

            def free_action(free_pressure):
                pressure = (
                    jnp.zeros_like(boundary_pressure).at[free_indices].set(free_pressure)
                )
                return balance_action(pressure)[free_indices]

            space = ArraySpace((len(free),), dtype=conductance.dtype)
            operator = FunctionLinearOperator(
                free_action,
                source=space,
                target=space,
                operator_id=f"{self.plan_id}:interior-pressure",
            )
            right = -balance_action(boundary_pressure)[free_indices]
            solved = solve(LinearSystem(operator), right)
            pressure = boundary_pressure.at[free_indices].set(solved.value)
            linear_success = jnp.all(solved.successful)
        else:
            pressure = boundary_pressure
            linear_success = jnp.asarray(True)
        volume_flow = conductance * (pressure[senders] - pressure[receivers])
        balance = balance_action(pressure)
        interior_balance = balance[jnp.asarray(free, dtype=jnp.int32)]
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(volume_flow), initial=0.0))
        balance_residual = jnp.max(jnp.abs(interior_balance), initial=0.0) / scale
        boundary_residual = jnp.max(
            jnp.abs(
                pressure[jnp.asarray(boundary)] - jnp.asarray(self.boundary_pressures)
            ),
            initial=0.0,
        )
        finite = jnp.all(jnp.isfinite(pressure)) & jnp.all(jnp.isfinite(volume_flow))
        evidence = PVSFlowEvidence(
            balance_residual,
            boundary_residual,
            finite,
            finite
            & linear_success
            & (balance_residual <= 1.0e-10)
            & (boundary_residual <= 1.0e-12),
        )
        return PVSFlowResult(pressure, volume_flow, evidence, self.plan_id)


class FlowScheduleEvidence(StrictModule):
    period_coverage: Array
    endpoint_periodicity_residual: Array
    finite: Array
    successful: Array


class FlowTransportSchedule(StrictModule):
    sample_times: Array
    volume_flow: Array
    period: float = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, sample_times: ArrayLike, volume_flow: ArrayLike, period: float, /):
        times = np.asarray(sample_times, dtype=float)
        flow = np.asarray(volume_flow, dtype=float)
        width = float(period)
        if (
            times.ndim != 1
            or len(times) < 2
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("sample_times must be finite and strictly increasing.")
        if flow.ndim != 2 or flow.shape[0] != len(times) or np.any(~np.isfinite(flow)):
            raise ValueError("volume_flow must have shape (sample_count, edge_count).")
        if not np.isfinite(width) or width <= 0.0 or times[-1] - times[0] > width:
            raise ValueError("period must be positive and cover the sampled horizon.")
        self.sample_times = jnp.asarray(times)
        self.volume_flow = jnp.asarray(flow)
        self.period = width
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "flow-transport-schedule",
                "times": array_tree_fingerprint(times),
                "flow": array_tree_fingerprint(flow),
                "period": width.hex(),
            }
        )

    def mean(self) -> tuple[Array, FlowScheduleEvidence]:
        times = self.sample_times
        values = self.volume_flow
        interval = jnp.diff(times)
        integral = jnp.sum(0.5 * interval[:, None] * (values[:-1] + values[1:]), axis=0)
        coverage = (times[-1] - times[0]) / self.period
        mean = integral / (times[-1] - times[0])
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.max(jnp.abs(values[0])), jnp.max(jnp.abs(values[-1])))
        )
        residual = jnp.max(jnp.abs(values[-1] - values[0])) / scale
        finite = jnp.all(jnp.isfinite(mean))
        coverage_complete = jnp.abs(coverage - 1.0) <= 1.0e-12
        evidence = FlowScheduleEvidence(
            coverage,
            residual,
            finite,
            finite & coverage_complete & (residual <= 1.0e-12),
        )
        return mean, evidence


__all__ = [
    "FlowScheduleEvidence",
    "FlowTransportSchedule",
    "PreparedTaylorHoodCSFFlow",
    "PVSFlowEvidence",
    "PVSFlowResult",
    "PVSNetworkFlowPlan",
    "TaylorHoodCSFFlowPlan",
]
