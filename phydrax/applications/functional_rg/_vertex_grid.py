#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._regulators import FunctionalRGStatus, Regulator, ThresholdQuadraturePlan
from ._wetterich import _volume_factor


class MomentumVertexState(StrictModule):
    inverse_propagator: Array
    four_point_vertex: Array
    wavefunction_renormalization: Array

    def __init__(
        self,
        inverse_propagator: ArrayLike,
        four_point_vertex: ArrayLike,
        wavefunction_renormalization: ArrayLike = 1.0,
        /,
    ):
        self.inverse_propagator = jnp.asarray(inverse_propagator)
        self.four_point_vertex = jnp.asarray(four_point_vertex)
        self.wavefunction_renormalization = jnp.asarray(
            wavefunction_renormalization
        ).reshape(())


class MomentumVertexFlowEvaluation(StrictModule):
    beta_inverse_propagator: Array
    beta_four_point_vertex: Array
    loop_integral: Array
    interpolation_support_fraction: Array
    quadrature_indicator: Array
    finite: Array
    admissible: Array
    status: Array
    prepared_id: str = eqx.field(static=True)


class MomentumVertexGridFlowPlan(StrictModule, NonTrainableState):
    """Finite crossing-symmetric one-channel vertex-grid Wetterich reference."""

    __hash__ = object.__hash__

    regulator: Regulator
    radial: ThresholdQuadraturePlan
    momentum_nodes: Array
    angular_nodes: Array
    angular_weights: Array
    dimension: float = eqx.field(static=True)
    angular_order: int = eqx.field(static=True)
    maximum_grid_entries: int = eqx.field(static=True)
    volume_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        momentum_nodes: ArrayLike,
        regulator: Regulator,
        radial: ThresholdQuadraturePlan,
        /,
        *,
        angular_order: int = 16,
        maximum_grid_entries: int = 1_000_000,
    ):
        momentum = np.asarray(momentum_nodes, dtype=np.float64)
        order = int(angular_order)
        capacity = int(maximum_grid_entries)
        if not isinstance(regulator, Regulator) or not isinstance(
            radial, ThresholdQuadraturePlan
        ):
            raise TypeError("Vertex-grid flow requires regulator and radial plans.")
        required = momentum.size * radial.quadrature_order * order
        if (
            momentum.ndim != 1
            or momentum.size < 3
            or np.any(~np.isfinite(momentum))
            or momentum[0] != 0.0
            or np.any(np.diff(momentum) <= 0.0)
            or order < 4
            or capacity <= 0
            or required > capacity
        ):
            raise ValueError("Momentum/angle grid exceeds the fixed vertex-flow budget.")
        angular_rule = gauss_legendre_data(order)
        angular_nodes = jnp.asarray(angular_rule.nodes)
        raw_weights = jnp.asarray(angular_rule.weights) * (1.0 - angular_nodes**2) ** (
            0.5 * (radial.dimension - 3.0)
        )
        angular_weights = raw_weights / jnp.sum(raw_weights)
        self.regulator = regulator
        self.radial = radial
        self.momentum_nodes = jnp.asarray(momentum)
        self.angular_nodes = angular_nodes
        self.angular_weights = angular_weights
        self.dimension = radial.dimension
        self.angular_order = order
        self.maximum_grid_entries = capacity
        self.volume_factor = _volume_factor(radial.dimension)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "momentum-dependent-four-vertex-grid-flow",
                "momentum_nodes": array_tree_fingerprint(momentum),
                "regulator": regulator.regulator_id,
                "radial": radial.plan_id,
                "angular_order": order,
                "maximum_grid_entries": capacity,
                "channel": "crossing-symmetrized-s-t-u-reference",
            }
        )

    def prepare(self, /) -> "PreparedMomentumVertexGridFlow":
        return PreparedMomentumVertexGridFlow(self)


class PreparedMomentumVertexGridFlow(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: MomentumVertexGridFlowPlan
    shifted_momenta: Array
    support_mask: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MomentumVertexGridFlowPlan, /):
        if not isinstance(plan, MomentumVertexGridFlowPlan):
            raise TypeError("plan must be MomentumVertexGridFlowPlan.")
        p = plan.momentum_nodes[:, None, None]
        q = jnp.sqrt(plan.radial.nodes)[None, :, None]
        cosine = plan.angular_nodes[None, None, :]
        shifted = jnp.sqrt(jnp.maximum(p**2 + q**2 + 2.0 * p * q * cosine, 0.0))
        support = shifted <= plan.momentum_nodes[-1]
        self.plan = plan
        self.shifted_momenta = shifted
        self.support_mask = support
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-momentum-vertex-grid-flow",
                "plan": plan.plan_id,
                "shifted_shape": list(shifted.shape),
                "support": array_tree_fingerprint(np.asarray(support)),
            }
        )

    def evaluate(
        self,
        state: MomentumVertexState,
        anomalous_dimension: ArrayLike = 0.0,
        /,
    ) -> MomentumVertexFlowEvaluation:
        if not isinstance(state, MomentumVertexState):
            raise TypeError("state must be MomentumVertexState.")
        shape = self.plan.momentum_nodes.shape
        if (
            state.inverse_propagator.shape != shape
            or state.four_point_vertex.shape != shape
        ):
            raise ValueError("Propagator and vertex data must match the momentum grid.")
        eta = jnp.asarray(anomalous_dimension).reshape(())
        q = jnp.sqrt(self.plan.radial.nodes)
        propagator_q = linear_interpolate(
            self.plan.momentum_nodes,
            state.inverse_propagator,
            q,
            bounds="clip",
        ).values
        vertex_q = linear_interpolate(
            self.plan.momentum_nodes,
            state.four_point_vertex,
            q,
            bounds="clip",
        ).values
        flat_shifted = self.shifted_momenta.reshape((-1,))
        shifted_propagator = linear_interpolate(
            self.plan.momentum_nodes,
            state.inverse_propagator,
            flat_shifted,
            bounds="clip",
        ).values.reshape(self.shifted_momenta.shape)
        shifted_vertex = linear_interpolate(
            self.plan.momentum_nodes,
            state.four_point_vertex,
            flat_shifted,
            bounds="clip",
        ).values.reshape(self.shifted_momenta.shape)
        y = self.plan.radial.nodes
        cutoff_shape = self.plan.regulator.shape(y)
        cutoff_derivative = self.plan.regulator.derivative(y)
        cutoff_rate = (2.0 - eta) * cutoff_shape - 2.0 * y * cutoff_derivative
        radial_weight = (
            0.5
            * self.plan.radial.weights
            * y ** (0.5 * self.plan.dimension - 1.0)
            * cutoff_rate
        )
        safe_q = jnp.where(jnp.abs(propagator_q) > 1.0e-15, propagator_q, 1.0)
        safe_shifted = jnp.where(
            jnp.abs(shifted_propagator) > 1.0e-15,
            shifted_propagator,
            1.0,
        )
        integrand = jnp.where(
            self.support_mask,
            vertex_q[None, :, None]
            * shifted_vertex
            / (safe_q[None, :, None] ** 2 * safe_shifted),
            0.0,
        )
        angular_average = contract("pqa,a->pq", integrand, self.plan.angular_weights)
        loops = self.plan.volume_factor * contract(
            "pq,q->p", angular_average, radial_weight
        )
        tadpole = self.plan.volume_factor * contract(
            "q,q,q->", radial_weight, vertex_q, 1.0 / safe_q**2
        )
        momentum_squared = self.plan.momentum_nodes**2
        beta_inverse = (
            (-2.0 + eta) * state.inverse_propagator
            + (2.0 - eta) * momentum_squared
            + 0.5 * tadpole
        )
        beta_vertex = (
            self.plan.dimension - 4.0 + 2.0 * eta
        ) * state.four_point_vertex - 3.0 * loops
        support_fraction = jnp.mean(self.support_mask)
        support_complete = jnp.all(self.support_mask)
        finite = (
            jnp.all(jnp.isfinite(state.inverse_propagator))
            & jnp.all(jnp.isfinite(state.four_point_vertex))
            & jnp.isfinite(state.wavefunction_renormalization)
            & (state.wavefunction_renormalization > 0.0)
            & jnp.isfinite(eta)
            & jnp.all(jnp.isfinite(beta_inverse))
            & jnp.all(jnp.isfinite(beta_vertex))
        )
        admissible = finite & support_complete & jnp.all(state.inverse_propagator > 0.0)
        status = jnp.where(
            ~finite,
            int(FunctionalRGStatus.NONFINITE),
            jnp.where(
                ~support_complete,
                int(FunctionalRGStatus.CAPACITY_EXCEEDED),
                jnp.where(
                    admissible,
                    int(FunctionalRGStatus.SUCCESS),
                    int(FunctionalRGStatus.POLE_ENCOUNTERED),
                ),
            ),
        ).astype(jnp.int32)
        return MomentumVertexFlowEvaluation(
            beta_inverse,
            beta_vertex,
            loops,
            support_fraction,
            jnp.max(jnp.abs(radial_weight[-1] * angular_average[:, -1])),
            finite,
            admissible,
            status,
            self.prepared_id,
        )


__all__ = [
    "MomentumVertexFlowEvaluation",
    "MomentumVertexGridFlowPlan",
    "MomentumVertexState",
    "PreparedMomentumVertexGridFlow",
]
