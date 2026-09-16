#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Timelike conformal-boundary geometry, radiation, and corner evidence."""

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein
from phydrax.linalg import inverse

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._conformal_einstein import ConformalEinsteinState


AdSBoundarySide: TypeAlias = Literal["lower", "upper"]
AdSRadiationPolicy: TypeAlias = Literal["reflecting", "driven", "dissipative"]


class AdSConformalBoundaryPlan(StrictModule):
    """One Cartesian timelike conformal boundary with explicit free radiation data."""

    target_induced_metric: Array
    spatial_axis: int = eqx.field(static=True)
    side: AdSBoundarySide = eqx.field(static=True)
    radiation_policy: AdSRadiationPolicy = eqx.field(static=True)
    target_incoming_radiation: float = eqx.field(static=True)
    conformal_factor_tolerance: float = eqx.field(static=True)
    minimum_normal_gradient: float = eqx.field(static=True)
    minimum_normal_norm: float = eqx.field(static=True)
    metric_tolerance: float = eqx.field(static=True)
    radiation_tolerance: float = eqx.field(static=True)
    corner_tolerance: float = eqx.field(static=True)
    require_corner_compatibility: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_axis: int,
        side: AdSBoundarySide,
        target_induced_metric: ArrayLike,
        /,
        *,
        radiation_policy: AdSRadiationPolicy,
        target_incoming_radiation: float = 0.0,
        conformal_factor_tolerance: float = 1e-10,
        minimum_normal_gradient: float = 1e-8,
        minimum_normal_norm: float = 1e-8,
        metric_tolerance: float = 1e-8,
        radiation_tolerance: float = 1e-8,
        corner_tolerance: float = 1e-8,
        require_corner_compatibility: bool = True,
    ):
        axis = int(spatial_axis)
        side_value = str(side)
        policy = str(radiation_policy)
        metric = np.asarray(target_induced_metric, dtype=float)
        incoming = float(target_incoming_radiation)
        tolerances = tuple(
            float(value)
            for value in (
                conformal_factor_tolerance,
                minimum_normal_gradient,
                minimum_normal_norm,
                metric_tolerance,
                radiation_tolerance,
                corner_tolerance,
            )
        )
        if axis not in (0, 1, 2) or side_value not in {"lower", "upper"}:
            raise ValueError("AdS boundary axis or side is invalid.")
        if policy not in {"reflecting", "driven", "dissipative"}:
            raise ValueError("Unknown AdS radiation policy.")
        if metric.shape != (3, 3) or not np.all(np.isfinite(metric)):
            raise ValueError("target_induced_metric must be one finite 3x3 tensor.")
        if not np.allclose(metric, metric.T):
            raise ValueError("target_induced_metric must be symmetric.")
        if not np.isfinite(incoming) or any(
            not np.isfinite(value) or value < 0.0 for value in tolerances
        ):
            raise ValueError("AdS boundary data and tolerances are invalid.")
        self.target_induced_metric = jnp.asarray(metric)
        self.spatial_axis = axis
        self.side = side_value
        self.radiation_policy = policy
        self.target_incoming_radiation = incoming
        self.conformal_factor_tolerance = tolerances[0]
        self.minimum_normal_gradient = tolerances[1]
        self.minimum_normal_norm = tolerances[2]
        self.metric_tolerance = tolerances[3]
        self.radiation_tolerance = tolerances[4]
        self.corner_tolerance = tolerances[5]
        self.require_corner_compatibility = bool(require_corner_compatibility)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "timelike-ads-conformal-boundary-plan",
                "spatial_axis": axis,
                "side": side_value,
                "target_induced_metric": array_tree_fingerprint(metric),
                "radiation_policy": policy,
                "target_incoming_radiation": incoming,
                "tolerances": tolerances,
                "require_corner_compatibility": bool(require_corner_compatibility),
            }
        )


class AdSConformalBoundaryEvidence(StrictModule):
    conformal_factor_residual: Array
    minimum_normal_gradient: Array
    minimum_normal_norm: Array
    induced_metric_residual: Array
    radiation_residual: Array
    corner_residual: Array
    corner_available: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _boundary_slice(
    value: Array, spatial_axis: int, side: str, spatial_rank: int, /
) -> Array:
    array_axis = value.ndim - spatial_rank + spatial_axis
    return jnp.take(value, 0 if side == "lower" else -1, axis=array_axis)


def evaluate_ads_conformal_boundary(
    plan: AdSConformalBoundaryPlan,
    state: ConformalEinsteinState,
    conformal_gradient: ArrayLike,
    incoming_radiation: ArrayLike,
    /,
    *,
    conformal_rate: ArrayLike | None = None,
) -> AdSConformalBoundaryEvidence:
    if not isinstance(plan, AdSConformalBoundaryPlan):
        raise TypeError("plan must be AdSConformalBoundaryPlan.")
    if not isinstance(state, ConformalEinsteinState):
        raise TypeError("state must be ConformalEinsteinState.")
    spatial_rank = len(state.spatial_shape)
    if spatial_rank != 3:
        raise ValueError(
            "AdS Cartesian boundary evidence requires a three-dimensional grid."
        )
    gradient = jnp.asarray(conformal_gradient, dtype=state.metric.dtype)
    if gradient.shape != (4,) + state.spatial_shape:
        raise ValueError("conformal_gradient must have one spacetime covector axis.")
    omega_boundary = _boundary_slice(
        state.conformal_factor, plan.spatial_axis, plan.side, spatial_rank
    )
    gradient_boundary = _boundary_slice(
        gradient, plan.spatial_axis, plan.side, spatial_rank
    )
    metric_boundary = _boundary_slice(
        state.metric, plan.spatial_axis, plan.side, spatial_rank
    )
    trailing = jnp.moveaxis(metric_boundary, (0, 1), (-2, -1))
    inverse_metric = jnp.moveaxis(inverse(trailing).value, (-2, -1), (0, 1))
    raised = ein.contract("ab...,b...->a...", inverse_metric, gradient_boundary)
    normal_norm = ein.contract("a...,a...->...", gradient_boundary, raised)
    normal_component = plan.spatial_axis + 1
    normal_gradient = jnp.abs(gradient_boundary[normal_component])
    tangential = tuple(index for index in range(4) if index != normal_component)
    induced = metric_boundary[
        jnp.asarray(tangential)[:, None], jnp.asarray(tangential)[None, :]
    ]
    target_shape = (3, 3) + (1,) * (induced.ndim - 2)
    target = plan.target_induced_metric.reshape(target_shape)
    metric_scale = jnp.maximum(1.0, jnp.max(jnp.abs(target)))
    induced_residual = jnp.max(jnp.abs(induced - target)) / metric_scale
    incoming = jnp.asarray(incoming_radiation, dtype=state.metric.dtype)
    if incoming.shape != omega_boundary.shape:
        raise ValueError("incoming_radiation must match the selected boundary grid.")
    radiation_residual = jnp.max(jnp.abs(incoming - plan.target_incoming_radiation))
    if conformal_rate is None:
        corner_residual = jnp.asarray(jnp.nan, dtype=state.metric.dtype)
        corner_available = jnp.asarray(False)
    else:
        rate = jnp.asarray(conformal_rate, dtype=state.metric.dtype)
        if rate.shape != state.spatial_shape:
            raise ValueError("conformal_rate must match the state grid.")
        corner_residual = jnp.max(
            jnp.abs(_boundary_slice(rate, plan.spatial_axis, plan.side, spatial_rank))
        )
        corner_available = jnp.asarray(True)
    omega_residual = jnp.max(jnp.abs(omega_boundary))
    minimum_gradient = jnp.min(normal_gradient)
    minimum_norm = jnp.min(normal_norm)
    finite = (
        jnp.isfinite(omega_residual)
        & jnp.isfinite(minimum_gradient)
        & jnp.isfinite(minimum_norm)
        & jnp.isfinite(induced_residual)
        & jnp.isfinite(radiation_residual)
        & jnp.where(corner_available, jnp.isfinite(corner_residual), True)
    )
    corner_ok = jnp.asarray(not plan.require_corner_compatibility) | (
        corner_available & (corner_residual <= plan.corner_tolerance)
    )
    accepted = (
        finite
        & (omega_residual <= plan.conformal_factor_tolerance)
        & (minimum_gradient >= plan.minimum_normal_gradient)
        & (minimum_norm >= plan.minimum_normal_norm)
        & (induced_residual <= plan.metric_tolerance)
        & (radiation_residual <= plan.radiation_tolerance)
        & corner_ok
    )
    return AdSConformalBoundaryEvidence(
        conformal_factor_residual=omega_residual,
        minimum_normal_gradient=minimum_gradient,
        minimum_normal_norm=minimum_norm,
        induced_metric_residual=induced_residual,
        radiation_residual=radiation_residual,
        corner_residual=corner_residual,
        corner_available=corner_available,
        finite=finite,
        accepted=accepted,
        plan_id=plan.plan_id,
        state_id=state.state_id,
        claim="finite-timelike-conformal-boundary-and-corner-evidence",
    )


__all__ = [
    "AdSBoundarySide",
    "AdSConformalBoundaryEvidence",
    "AdSConformalBoundaryPlan",
    "AdSRadiationPolicy",
    "evaluate_ads_conformal_boundary",
]
