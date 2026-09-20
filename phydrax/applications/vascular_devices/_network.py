#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compliant vascular network with inertial and device pressure losses."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class VascularNetworkState:
    node_pressure_pa: Array
    edge_flow_m3_s: Array
    time_s: Array


@dataclass(frozen=True, slots=True)
class VascularNetworkStep:
    state: VascularNetworkState
    node_mass_balance_residual_m3_s: Array
    edge_momentum_residual_pa: Array
    dissipated_power_w: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class VascularDeviceNetwork:
    incidence: Array
    node_compliance_m3_pa: Array
    edge_inertance_pa_s2_m3: Array
    edge_resistance_pa_s_m3: Array
    device_quadratic_loss_pa_s2_m6: Array

    @classmethod
    def create(
        cls,
        incidence: ArrayLike,
        node_compliance_m3_pa: ArrayLike,
        edge_inertance_pa_s2_m3: ArrayLike,
        edge_resistance_pa_s_m3: ArrayLike,
        device_quadratic_loss_pa_s2_m6: ArrayLike,
        /,
    ) -> VascularDeviceNetwork:
        graph = np.asarray(incidence, dtype=float)
        compliance = np.asarray(node_compliance_m3_pa, dtype=float)
        inertance = np.asarray(edge_inertance_pa_s2_m3, dtype=float)
        resistance = np.asarray(edge_resistance_pa_s_m3, dtype=float)
        quadratic = np.asarray(device_quadratic_loss_pa_s2_m6, dtype=float)
        if graph.ndim != 2 or graph.shape[0] == 0 or graph.shape[1] == 0:
            raise ValueError("Vascular incidence must have node-by-edge shape.")
        if not np.all(np.isin(graph, (-1.0, 0.0, 1.0))) or np.any(
            np.count_nonzero(graph, axis=0) != 2
        ):
            raise ValueError(
                "Every vascular edge must connect exactly two oriented nodes."
            )
        if compliance.shape != (graph.shape[0],) or np.any(compliance <= 0):
            raise ValueError("Vascular node compliance must be positive and aligned.")
        if any(
            value.shape != (graph.shape[1],)
            for value in (inertance, resistance, quadratic)
        ):
            raise ValueError(
                "Vascular edge properties must align with incidence columns."
            )
        if np.any(inertance <= 0) or np.any(resistance < 0) or np.any(quadratic < 0):
            raise ValueError(
                "Vascular inertance must be positive and losses non-negative."
            )
        return cls(
            jnp.asarray(graph),
            jnp.asarray(compliance),
            jnp.asarray(inertance),
            jnp.asarray(resistance),
            jnp.asarray(quadratic),
        )

    def advance(
        self,
        state: VascularNetworkState,
        node_source_m3_s: ArrayLike,
        step_size_s: float,
        /,
    ) -> VascularNetworkStep:
        pressure = jnp.asarray(state.node_pressure_pa)
        flow = jnp.asarray(state.edge_flow_m3_s)
        source = jnp.asarray(node_source_m3_s)
        if (
            pressure.shape != self.node_compliance_m3_pa.shape
            or source.shape != pressure.shape
        ):
            raise ValueError("Vascular node state and sources have incompatible shapes.")
        if flow.shape != self.edge_inertance_pa_s2_m3.shape or step_size_s <= 0:
            raise ValueError("Vascular edge flow or step size is invalid.")
        dt = float(step_size_s)
        nonlinear_loss = self.device_quadratic_loss_pa_s2_m6 * flow * jnp.abs(flow)
        pressure_block = jnp.diag(self.node_compliance_m3_pa / dt)
        flow_block = jnp.diag(
            self.edge_inertance_pa_s2_m3 / dt + self.edge_resistance_pa_s_m3
        )
        matrix = jnp.block(
            [
                [pressure_block, self.incidence],
                [-self.incidence.T, flow_block],
            ]
        )
        right = jnp.concatenate(
            (
                self.node_compliance_m3_pa / dt * pressure + source,
                self.edge_inertance_pa_s2_m3 / dt * flow - nonlinear_loss,
            )
        )
        solved = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        node_count = pressure.size
        next_pressure = solved.value[:node_count]
        next_flow = solved.value[node_count:]
        node_residual = (
            self.node_compliance_m3_pa * (next_pressure - pressure) / dt
            + self.incidence @ next_flow
            - source
        )
        edge_residual = (
            self.edge_inertance_pa_s2_m3 * (next_flow - flow) / dt
            - self.incidence.T @ next_pressure
            + self.edge_resistance_pa_s_m3 * next_flow
            + nonlinear_loss
        )
        dissipated = jnp.sum(
            self.edge_resistance_pa_s_m3 * next_flow**2
            + self.device_quadratic_loss_pa_s2_m6 * jnp.abs(next_flow) ** 3
        )
        next_state = VascularNetworkState(
            next_pressure,
            next_flow,
            jnp.asarray(state.time_s) + dt,
        )
        successful = (
            solved.successful & jnp.all(jnp.isfinite(solved.value)) & (dissipated >= 0)
        )
        return VascularNetworkStep(
            next_state,
            node_residual,
            edge_residual,
            dissipated,
            successful,
        )


__all__ = ["VascularDeviceNetwork", "VascularNetworkState", "VascularNetworkStep"]
