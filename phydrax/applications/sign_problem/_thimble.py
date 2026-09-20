#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite generalized-thimble quadrature with exact discrete-flow Jacobians."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ... import ein
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics import log_normalize
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    PreparedFactorization,
)


class HolomorphicFlowStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_FLOW = 1
    SINGULAR_JACOBIAN = 2
    IMAGINARY_ACTION_DRIFT = 3
    INSUFFICIENT_RESIDUAL_PHASE = 4
    INVALID_NORMALIZATION = 5
    NONFINITE_OBSERVABLE = 6


class HolomorphicFlowQuadraturePlan(StrictModule, NonTrainableState):
    """Static RK flow and dense Jacobian resource contract."""

    flow_time: float = eqx.field(static=True)
    flow_steps: int = eqx.field(static=True)
    maximum_nodes: int = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    maximum_jacobian_entries: int = eqx.field(static=True)
    maximum_imaginary_action_drift: float = eqx.field(static=True)
    minimum_average_residual_phase: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        flow_time: float,
        flow_steps: int,
        maximum_nodes: int = 4096,
        maximum_dimension: int = 64,
        maximum_jacobian_entries: int = 4_194_304,
        maximum_imaginary_action_drift: float = 1e-5,
        minimum_average_residual_phase: float = 1e-3,
    ):
        time = float(flow_time)
        steps = int(flow_steps)
        nodes = int(maximum_nodes)
        dimension = int(maximum_dimension)
        entries = int(maximum_jacobian_entries)
        action_drift = float(maximum_imaginary_action_drift)
        residual_phase = float(minimum_average_residual_phase)
        if not np.isfinite(time) or time < 0.0 or steps <= 0:
            raise ValueError("Holomorphic flow time/step count is invalid.")
        if nodes <= 0 or dimension <= 0 or entries <= 0:
            raise ValueError("Holomorphic flow resource limits must be positive.")
        if not np.isfinite(action_drift) or action_drift < 0.0:
            raise ValueError(
                "maximum_imaginary_action_drift must be finite and non-negative."
            )
        if not np.isfinite(residual_phase) or not 0.0 < residual_phase <= 1.0:
            raise ValueError("minimum_average_residual_phase must lie in (0, 1].")
        self.flow_time = time
        self.flow_steps = steps
        self.maximum_nodes = nodes
        self.maximum_dimension = dimension
        self.maximum_jacobian_entries = entries
        self.maximum_imaginary_action_drift = action_drift
        self.minimum_average_residual_phase = residual_phase
        self.plan_id = canonical_fingerprint(
            {
                "kind": "holomorphic-flow-quadrature-plan",
                "flow_time": time,
                "flow_steps": steps,
                "maximum_nodes": nodes,
                "maximum_dimension": dimension,
                "maximum_jacobian_entries": entries,
                "maximum_imaginary_action_drift": action_drift,
                "minimum_average_residual_phase": residual_phase,
            }
        )


class PreparedHolomorphicFlowQuadrature(StrictModule, NonTrainableState):
    """Frozen real quadrature and holomorphic action awaiting deformation."""

    action: Callable[[Array], Array] = eqx.field(static=True)
    nodes: Array
    weights: Array
    plan: HolomorphicFlowQuadraturePlan
    action_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def node_count(self) -> int:
        return self.nodes.shape[0]

    @property
    def dimension(self) -> int:
        return self.nodes.shape[1]


class HolomorphicFlowGeometry(StrictModule):
    """Deformed nodes and exact Jacobians of the finite RK map."""

    deformed_nodes: Array
    jacobians: Array
    determinant_phase: Array
    log_abs_determinant: Array
    original_action: Array
    deformed_action: Array
    imaginary_action_drift: Array
    maximum_imaginary_action_drift: Array
    factorization: PreparedFactorization
    finite: Array
    nonsingular: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class HolomorphicFlowQuadratureDiagnostics(StrictModule):
    status: Array
    average_residual_phase: Array
    average_residual_phase_magnitude: Array
    phase_quenched_log_mass: Array
    complex_mass_phase: Array
    maximum_imaginary_action_drift: Array
    minimum_jacobian_log_magnitude: Array
    maximum_jacobian_log_magnitude: Array
    finite_nodes: Array
    abstained: Array


class HolomorphicFlowQuadratureResult(StrictModule):
    value: Array
    numerator: Array
    denominator: Array
    observable_values: Array
    geometry: HolomorphicFlowGeometry
    diagnostics: HolomorphicFlowQuadratureDiagnostics
    successful: Array
    prepared_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_holomorphic_flow_quadrature(
    plan: HolomorphicFlowQuadraturePlan,
    action: Callable[[Array], Array],
    nodes: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    action_id: str,
) -> PreparedHolomorphicFlowQuadrature:
    """Validate a positive finite real reference quadrature before allocating Jacobians."""
    if not isinstance(plan, HolomorphicFlowQuadraturePlan):
        raise TypeError("plan must be HolomorphicFlowQuadraturePlan.")
    if not callable(action):
        raise TypeError("action must be callable.")
    nodes_host = np.asarray(nodes)
    weights_host = np.asarray(weights)
    if nodes_host.ndim != 2 or nodes_host.shape[0] < 1 or nodes_host.shape[1] < 1:
        raise ValueError("nodes must have shape (node_count, dimension).")
    count, dimension = (size for size in nodes_host.shape)
    if weights_host.shape != (count,):
        raise ValueError("weights must match the quadrature node axis.")
    if nodes_host.dtype.kind != "f" or weights_host.dtype.kind != "f":
        raise TypeError("Reference nodes and weights must use real floating dtypes.")
    if np.any(~np.isfinite(nodes_host)) or np.any(~np.isfinite(weights_host)):
        raise ValueError("Reference quadrature data must be finite.")
    if np.any(weights_host <= 0.0):
        raise ValueError("Generalized-thimble reference weights must be positive.")
    if count > plan.maximum_nodes or dimension > plan.maximum_dimension:
        raise ValueError("Reference quadrature exceeds node or dimension capacity.")
    if count * dimension * dimension > plan.maximum_jacobian_entries:
        raise ValueError("Flow Jacobians exceed maximum_jacobian_entries.")
    identifier = str(action_id)
    if not identifier:
        raise ValueError("action_id must be non-empty.")
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-holomorphic-flow-quadrature",
            "plan": plan.plan_id,
            "action": identifier,
            "nodes": array_tree_fingerprint(nodes_host),
            "weights": array_tree_fingerprint(weights_host),
        }
    )
    return PreparedHolomorphicFlowQuadrature(
        action=action,
        nodes=jnp.asarray(nodes_host),
        weights=jnp.asarray(weights_host),
        plan=plan,
        action_id=identifier,
        prepared_id=prepared_id,
    )


def _action_value(prepared: PreparedHolomorphicFlowQuadrature, point: Array, /) -> Array:
    value = jnp.asarray(prepared.action(point))
    if value.shape != () or not jnp.iscomplexobj(value):
        raise TypeError("Holomorphic-flow action must return one complex scalar.")
    return value


def _upward_flow_vector(
    prepared: PreparedHolomorphicFlowQuadrature, point: Array, /
) -> Array:
    gradient = jax.grad(lambda value: _action_value(prepared, value), holomorphic=True)(
        point
    )
    return jnp.conj(gradient)


def _flow_map(prepared: PreparedHolomorphicFlowQuadrature, real_point: Array, /) -> Array:
    step_size = prepared.plan.flow_time / prepared.plan.flow_steps
    initial = real_point.astype(jnp.result_type(real_point.dtype, 1j))

    def step(point, _):
        k1 = _upward_flow_vector(prepared, point)
        k2 = _upward_flow_vector(prepared, point + 0.5 * step_size * k1)
        k3 = _upward_flow_vector(prepared, point + 0.5 * step_size * k2)
        k4 = _upward_flow_vector(prepared, point + step_size * k3)
        return point + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    deformed, _ = jax.lax.scan(
        step,
        initial,
        None,
        length=prepared.plan.flow_steps,
    )
    return deformed


def deform_holomorphic_quadrature(
    prepared: PreparedHolomorphicFlowQuadrature, /
) -> HolomorphicFlowGeometry:
    """Execute the finite flow and differentiate that exact discrete map."""
    if not isinstance(prepared, PreparedHolomorphicFlowQuadrature):
        raise TypeError("prepared must be PreparedHolomorphicFlowQuadrature.")
    deformed = jax.vmap(lambda point: _flow_map(prepared, point))(prepared.nodes)
    jacobians = jax.vmap(jax.jacfwd(lambda point: _flow_map(prepared, point)))(
        prepared.nodes
    )
    factorization = factorize(
        DenseLinearOperator(jacobians),
        FactorizationPolicy("lu"),
    )
    determinant_phase = factorization.determinant_sign()
    log_abs_determinant = factorization.log_abs_determinant()
    original_action = jax.vmap(
        lambda point: _action_value(
            prepared, point.astype(jnp.result_type(point.dtype, 1j))
        )
    )(prepared.nodes)
    deformed_action = jax.vmap(lambda point: _action_value(prepared, point))(deformed)
    imaginary_drift = jnp.abs(jnp.imag(deformed_action - original_action))
    maximum_drift = jnp.max(imaginary_drift)
    finite = (
        jnp.all(jnp.isfinite(jnp.real(deformed)))
        & jnp.all(jnp.isfinite(jnp.imag(deformed)))
        & jnp.all(jnp.isfinite(jnp.real(jacobians)))
        & jnp.all(jnp.isfinite(jnp.imag(jacobians)))
        & jnp.all(jnp.isfinite(jnp.real(deformed_action)))
        & jnp.all(jnp.isfinite(jnp.imag(deformed_action)))
        & jnp.all(jnp.isfinite(log_abs_determinant))
        & jnp.all(jnp.isfinite(jnp.real(determinant_phase)))
        & jnp.all(jnp.isfinite(jnp.imag(determinant_phase)))
    )
    nonsingular = jnp.all(determinant_phase != 0)
    return HolomorphicFlowGeometry(
        deformed_nodes=deformed,
        jacobians=jacobians,
        determinant_phase=determinant_phase,
        log_abs_determinant=log_abs_determinant,
        original_action=original_action,
        deformed_action=deformed_action,
        imaginary_action_drift=imaginary_drift,
        maximum_imaginary_action_drift=maximum_drift,
        factorization=factorization,
        finite=finite,
        nonsingular=nonsingular,
        prepared_id=prepared.prepared_id,
        claim="exact-jacobian-of-finite-runge-kutta-holomorphic-flow-map",
    )


def integrate_holomorphic_flow_quadrature(
    prepared: PreparedHolomorphicFlowQuadrature,
    geometry: HolomorphicFlowGeometry,
    observable: Callable[[Array], Array],
    /,
) -> HolomorphicFlowQuadratureResult:
    """Integrate on the deformed manifold or explicitly abstain on failed evidence."""
    if not isinstance(prepared, PreparedHolomorphicFlowQuadrature):
        raise TypeError("prepared must be PreparedHolomorphicFlowQuadrature.")
    if not isinstance(geometry, HolomorphicFlowGeometry):
        raise TypeError("geometry must be HolomorphicFlowGeometry.")
    if geometry.prepared_id != prepared.prepared_id:
        raise ValueError("Flow geometry belongs to another preparation.")
    if not callable(observable):
        raise TypeError("observable must be callable.")
    observable_values = jax.vmap(lambda point: jnp.asarray(observable(point)))(
        geometry.deformed_nodes
    )
    if observable_values.ndim < 1 or observable_values.shape[0] != prepared.node_count:
        raise ValueError(
            "observable must return one fixed-shape value per deformed node."
        )
    finite_observable = jnp.all(jnp.isfinite(jnp.real(observable_values))) & jnp.all(
        jnp.isfinite(jnp.imag(observable_values))
    )
    log_magnitude = (
        jnp.log(prepared.weights)
        + geometry.log_abs_determinant
        - jnp.real(geometry.deformed_action)
    )
    normalized_magnitude, log_mass, weight_valid = log_normalize(log_magnitude, axes=0)
    residual_phases = geometry.determinant_phase * jnp.exp(
        -1j * jnp.imag(geometry.deformed_action)
    )
    average_phase = ein.contract("n,n->", normalized_magnitude, residual_phases)
    phase_magnitude = jnp.abs(average_phase)
    phase_shape = (prepared.node_count,) + (1,) * (observable_values.ndim - 1)
    numerator = ein.contract(
        "n,n...->...",
        normalized_magnitude,
        residual_phases.reshape(phase_shape) * observable_values,
    )
    denominator = average_phase
    denominator_valid = jnp.isfinite(denominator) & (jnp.abs(denominator) > 0.0)
    safe_denominator = jnp.where(denominator_valid, denominator, 1.0 + 0.0j)
    value = numerator / safe_denominator
    status = jnp.where(
        ~geometry.finite,
        int(HolomorphicFlowStatus.NONFINITE_FLOW),
        jnp.where(
            ~geometry.nonsingular,
            int(HolomorphicFlowStatus.SINGULAR_JACOBIAN),
            jnp.where(
                geometry.maximum_imaginary_action_drift
                > prepared.plan.maximum_imaginary_action_drift,
                int(HolomorphicFlowStatus.IMAGINARY_ACTION_DRIFT),
                jnp.where(
                    phase_magnitude < prepared.plan.minimum_average_residual_phase,
                    int(HolomorphicFlowStatus.INSUFFICIENT_RESIDUAL_PHASE),
                    jnp.where(
                        ~weight_valid | ~denominator_valid,
                        int(HolomorphicFlowStatus.INVALID_NORMALIZATION),
                        jnp.where(
                            finite_observable
                            & jnp.all(jnp.isfinite(jnp.real(value)))
                            & jnp.all(jnp.isfinite(jnp.imag(value))),
                            int(HolomorphicFlowStatus.SUCCESS),
                            int(HolomorphicFlowStatus.NONFINITE_OBSERVABLE),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    successful = status == int(HolomorphicFlowStatus.SUCCESS)
    nan = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=jnp.result_type(value, 1j))
    diagnostics = HolomorphicFlowQuadratureDiagnostics(
        status=status,
        average_residual_phase=average_phase,
        average_residual_phase_magnitude=phase_magnitude,
        phase_quenched_log_mass=log_mass,
        complex_mass_phase=jnp.where(
            phase_magnitude > 0.0, average_phase / phase_magnitude, 0.0j
        ),
        maximum_imaginary_action_drift=geometry.maximum_imaginary_action_drift,
        minimum_jacobian_log_magnitude=jnp.min(geometry.log_abs_determinant),
        maximum_jacobian_log_magnitude=jnp.max(geometry.log_abs_determinant),
        finite_nodes=jnp.sum(
            jnp.all(jnp.isfinite(jnp.real(geometry.deformed_nodes)), axis=1)
            & jnp.all(jnp.isfinite(jnp.imag(geometry.deformed_nodes)), axis=1),
            dtype=jnp.int32,
        ),
        abstained=~successful,
    )
    return HolomorphicFlowQuadratureResult(
        value=jnp.where(successful, value, nan),
        numerator=numerator,
        denominator=denominator,
        observable_values=observable_values,
        geometry=geometry,
        diagnostics=diagnostics,
        successful=successful,
        prepared_id=prepared.prepared_id,
        action_id=prepared.action_id,
        claim="finite-generalized-thimble-quadrature-with-residual-phase-abstention",
    )


__all__ = [
    "HolomorphicFlowGeometry",
    "HolomorphicFlowQuadratureDiagnostics",
    "HolomorphicFlowQuadraturePlan",
    "HolomorphicFlowQuadratureResult",
    "HolomorphicFlowStatus",
    "PreparedHolomorphicFlowQuadrature",
    "deform_holomorphic_quadrature",
    "integrate_holomorphic_flow_quadrature",
    "prepare_holomorphic_flow_quadrature",
]
