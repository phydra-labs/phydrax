#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._shallow_water import ShallowWaterWetDryPolicy


ShallowWaterBoundaryStatus: TypeAlias = Literal[
    "ready", "dry", "critical-ambiguous", "invalid"
]
ShorelineDerivativeStatus: TypeAlias = Literal[
    "fixed-mask", "isolated-event", "grazing", "simultaneous", "overflow", "unsupported"
]


def _weno_z5(values: Array, epsilon: float, power: int, /) -> tuple[Array, Array]:
    um2, um1 = jnp.roll(values, 2, 0), jnp.roll(values, 1, 0)
    up1, up2, up3 = (
        jnp.roll(values, -1, 0),
        jnp.roll(values, -2, 0),
        jnp.roll(values, -3, 0),
    )
    left = jnp.stack(
        (
            (2 * um2 - 7 * um1 + 11 * values) / 6,
            (-um1 + 5 * values + 2 * up1) / 6,
            (2 * values + 5 * up1 - up2) / 6,
        )
    )
    beta_l = jnp.stack(
        (
            13 / 12 * (um2 - 2 * um1 + values) ** 2
            + 0.25 * (um2 - 4 * um1 + 3 * values) ** 2,
            13 / 12 * (um1 - 2 * values + up1) ** 2 + 0.25 * (um1 - up1) ** 2,
            13 / 12 * (values - 2 * up1 + up2) ** 2
            + 0.25 * (3 * values - 4 * up1 + up2) ** 2,
        )
    )
    right = jnp.stack(
        (
            (-up3 + 5 * up2 + 2 * up1) / 6,
            (2 * up2 + 5 * up1 - values) / 6,
            (11 * up1 - 7 * values + 2 * um1) / 6,
        )
    )
    beta_r = jnp.stack(
        (
            13 / 12 * (up3 - 2 * up2 + up1) ** 2 + 0.25 * (up3 - 4 * up2 + 3 * up1) ** 2,
            13 / 12 * (up2 - 2 * up1 + values) ** 2 + 0.25 * (up2 - values) ** 2,
            13 / 12 * (up1 - 2 * values + um1) ** 2
            + 0.25 * (3 * up1 - 4 * values + um1) ** 2,
        )
    )
    optimal = jnp.asarray((0.1, 0.6, 0.3), dtype=values.dtype).reshape(
        (3,) + (1,) * values.ndim
    )

    def combine(candidates: Array, beta: Array, /) -> Array:
        tau = jnp.abs(beta[0] - beta[2])
        alpha = optimal * (1 + (tau[None] / (beta + epsilon)) ** power)
        return jnp.sum(alpha * candidates, axis=0) / jnp.sum(alpha, axis=0)

    return combine(left, beta_l), combine(right, beta_r)


class ShallowWaterReconstructionEvidence(StrictModule):
    characteristic_used: Array
    dry_stencil_fallback: Array
    eigenbasis_condition: Array


class ShallowWaterEquilibriumWENOZPlan(StrictModule, NonTrainableState):
    """Fifth-order WENO-Z on equilibrium surface/discharge variables."""

    characteristic: bool = eqx.field(static=True)
    characteristic_depth: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    power: int = eqx.field(static=True)
    wet_dry: ShallowWaterWetDryPolicy
    dry_stencil_policy: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        order: int = 5,
        /,
        *,
        characteristic: bool = False,
        wet_dry: ShallowWaterWetDryPolicy | None = None,
        characteristic_depth: float = 1e-8,
        condition_limit: float = 1e8,
        epsilon: float = 1e-12,
        power: int = 2,
    ):
        if int(order) != 5:
            raise ValueError("Equilibrium WENO-Z currently requires order=5.")
        values = (float(characteristic_depth), float(condition_limit), float(epsilon))
        if any(not np.isfinite(x) or x <= 0 for x in values) or int(power) <= 0:
            raise ValueError("WENO-Z thresholds must be positive and finite.")
        policy = ShallowWaterWetDryPolicy() if wet_dry is None else wet_dry
        if not isinstance(policy, ShallowWaterWetDryPolicy):
            raise TypeError("wet_dry must be ShallowWaterWetDryPolicy or None.")
        self.characteristic = bool(characteristic)
        self.characteristic_depth, self.condition_limit, self.epsilon = values
        self.power, self.wet_dry = int(power), policy
        self.dry_stencil_policy = "equilibrium-componentwise"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shallow-water-equilibrium-weno-z",
                "characteristic": self.characteristic,
                "depth": values[0],
                "condition": values[1],
                "epsilon": values[2],
                "power": self.power,
                "wet_dry": policy.policy_id,
            }
        )

    @property
    def radius(self) -> int:
        return 3

    def reconstruct(
        self,
        state: ArrayLike,
        bathymetry: ArrayLike,
        axis: int = 0,
        /,
        *,
        normal: ArrayLike | None = None,
        gravity: float = 9.81,
    ):
        value, bed, axis_ = jnp.asarray(state), jnp.asarray(bathymetry), int(axis)
        if value.shape[:-1] != bed.shape or value.shape[-1] < 2:
            raise ValueError("State and bathymetry shapes do not agree.")
        if bed.shape[axis_] < 6:
            raise ValueError("WENO-Z requires at least six cells on its axis.")
        state_m, bed_m = jnp.moveaxis(value, axis_, 0), jnp.moveaxis(bed, axis_, 0)
        depth = state_m[..., 0]
        variables = jnp.concatenate(((depth + bed_m)[..., None], state_m[..., 1:]), -1)
        left, right = _weno_z5(variables, self.epsilon, self.power)
        bed_l, bed_r = _weno_z5(bed_m, self.epsilon, self.power)
        bed_l = jnp.minimum(bed_l, left[..., 0])
        bed_r = jnp.minimum(bed_r, right[..., 0])
        state_l = jnp.concatenate(
            ((left[..., :1] - bed_l[..., None]), left[..., 1:]),
            -1,
        )
        state_r = jnp.concatenate(
            ((right[..., :1] - bed_r[..., None]), right[..., 1:]),
            -1,
        )
        wet = depth > self.characteristic_depth
        stencil_wet = (
            wet
            & jnp.roll(wet, 1, 0)
            & jnp.roll(wet, 2, 0)
            & jnp.roll(wet, -1, 0)
            & jnp.roll(wet, -2, 0)
            & jnp.roll(wet, -3, 0)
        )
        c = jnp.sqrt(jnp.maximum(float(gravity) * depth, 0))
        condition = (c + self.characteristic_depth) / jnp.maximum(
            c, self.characteristic_depth
        )
        characteristic = (
            jnp.asarray(self.characteristic)
            & stencil_wet
            & (condition <= self.condition_limit)
        )
        evidence = ShallowWaterReconstructionEvidence(
            jnp.moveaxis(characteristic, 0, axis_),
            jnp.moveaxis(jnp.asarray(self.characteristic) & ~stencil_wet, 0, axis_),
            jnp.moveaxis(condition, 0, axis_),
        )
        return (
            jnp.moveaxis(state_l, 0, axis_),
            jnp.moveaxis(state_r, 0, axis_),
            jnp.moveaxis(bed_l, 0, axis_),
            jnp.moveaxis(bed_r, 0, axis_),
            evidence,
        )


class ShallowWaterBoundaryTrace(StrictModule):
    exterior_state: Array
    boundary_mass_flux: Array
    status_code: Array
    ready: Array
    status: ShallowWaterBoundaryStatus = eqx.field(static=True)
    regime: str = eqx.field(static=True)


class ShallowWaterNormalDischargeBoundary(StrictModule, NonTrainableState):
    normal_discharge: Callable[[Array, Array, Any], ArrayLike]
    exterior_surface: Callable[[Array, Array, Any], ArrayLike]
    tangential_velocity: Callable[[Array, Array, Any], ArrayLike] | None
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        normal_discharge,
        exterior_surface,
        /,
        *,
        tangential_velocity=None,
        boundary_id: str,
    ):
        if not callable(normal_discharge) or not callable(exterior_surface):
            raise TypeError("Discharge and surface data must be callable.")
        if tangential_velocity is not None and not callable(tangential_velocity):
            raise TypeError("tangential_velocity must be callable or None.")
        if not str(boundary_id):
            raise ValueError("boundary_id must be non-empty.")
        self.normal_discharge, self.exterior_surface = normal_discharge, exterior_surface
        self.tangential_velocity, self.boundary_id = tangential_velocity, str(boundary_id)

    def trace(self, time, interior, coordinates, normal, bed, wet_dry, args=None, /):
        state, points, normal_ = (
            jnp.asarray(interior),
            jnp.asarray(coordinates),
            jnp.asarray(normal),
        )
        surface = jnp.asarray(
            self.exterior_surface(time, points, args), dtype=state.dtype
        )
        discharge = jnp.asarray(
            self.normal_discharge(time, points, args), dtype=state.dtype
        )
        depth = jnp.maximum(surface - jnp.asarray(bed), 0)
        tangent = (
            jnp.zeros_like(normal_)
            if self.tangential_velocity is None
            else jnp.asarray(self.tangential_velocity(time, points, args))
        )
        tangent = (
            tangent
            - ein.contract("...d,...d->...", tangent, normal_, backend="jax")[..., None]
            * normal_
        )
        exterior = wet_dry.enforce_dry_momentum(
            jnp.concatenate(
                (
                    depth[..., None],
                    discharge[..., None] * normal_ + depth[..., None] * tangent,
                ),
                -1,
            )
        )
        return ShallowWaterBoundaryTrace(
            exterior,
            discharge,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.all(jnp.isfinite(exterior)),
            "ready",
            "prescribed-normal-discharge",
        )


class ShallowWaterCharacteristicOpenBoundary(StrictModule, NonTrainableState):
    exterior_surface: Callable[[Array, Array, Any], ArrayLike]
    exterior_velocity: Callable[[Array, Array, Any], ArrayLike]
    critical_tolerance: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        exterior_surface,
        exterior_velocity,
        /,
        *,
        critical_tolerance=1e-6,
        boundary_id: str,
    ):
        if not callable(exterior_surface) or not callable(exterior_velocity):
            raise TypeError("Open-boundary data must be callable.")
        tolerance = float(critical_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0 or not str(boundary_id):
            raise ValueError("Open-boundary tolerance/id is invalid.")
        self.exterior_surface, self.exterior_velocity = (
            exterior_surface,
            exterior_velocity,
        )
        self.critical_tolerance, self.boundary_id = tolerance, str(boundary_id)

    def trace(
        self, time, interior, coordinates, normal, bed, wet_dry, gravity, args=None, /
    ):
        state, normal_, points = (
            jnp.asarray(interior),
            jnp.asarray(normal),
            jnp.asarray(coordinates),
        )
        hi, ui = state[..., 0], wet_dry.velocity(state)
        he = jnp.maximum(
            jnp.asarray(self.exterior_surface(time, points, args)) - jnp.asarray(bed), 0
        )
        ue = jnp.asarray(self.exterior_velocity(time, points, args))
        uni = ein.contract("...d,...d->...", ui, normal_, backend="jax")
        une = ein.contract("...d,...d->...", ue, normal_, backend="jax")
        ci, ce = (
            jnp.sqrt(jnp.maximum(gravity * hi, 0)),
            jnp.sqrt(jnp.maximum(gravity * he, 0)),
        )
        critical = jnp.abs(jnp.abs(uni) - ci) <= self.critical_tolerance * jnp.maximum(
            ci, 1
        )
        outflow, inflow = uni > ci, uni < -ci
        un = 0.5 * ((uni + 2 * ci) + (une - 2 * ce))
        c = jnp.maximum(0.25 * ((uni + 2 * ci) - (une - 2 * ce)), 0)
        h = jnp.where(outflow, hi, jnp.where(inflow, he, c * c / gravity))
        un = jnp.where(outflow, uni, jnp.where(inflow, une, un))
        tangent = jnp.where(
            inflow[..., None],
            ue - une[..., None] * normal_,
            ui - uni[..., None] * normal_,
        )
        exterior = wet_dry.enforce_dry_momentum(
            jnp.concatenate(
                (h[..., None], h[..., None] * (un[..., None] * normal_ + tangent)),
                -1,
            )
        )
        status_code = jnp.max(
            jnp.where(critical, 2, jnp.where(h <= wet_dry.wet_depth, 1, 0))
        )
        return ShallowWaterBoundaryTrace(
            exterior,
            h * un,
            status_code.astype(jnp.int32),
            jnp.all(~critical) & jnp.all(jnp.isfinite(exterior)),
            "critical-ambiguous",
            "riemann-invariant",
        )


class PreparedGeostrophicBalance(StrictModule, NonTrainableState):
    reference_state: Array
    reference_residual: Array
    geometry_id: str = eqx.field(static=True)
    balance_id: str = eqx.field(static=True)

    def deviation_residual(self, residual, state, args=None, /):
        return jnp.asarray(residual(jnp.asarray(state), args)) - self.reference_residual


class GeostrophicBalancePlan(StrictModule, NonTrainableState):
    reference_surface: Array
    reference_discharge: Array
    coriolis_source: Any
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_surface,
        reference_discharge,
        coriolis_source,
        /,
        *,
        tolerance=1e-10,
    ):
        surface = np.asarray(reference_surface)
        discharge = np.asarray(reference_discharge)
        tol = float(tolerance)
        if (
            discharge.shape[:-1] != surface.shape
            or np.any(~np.isfinite(surface))
            or np.any(~np.isfinite(discharge))
            or not np.isfinite(tol)
            or tol <= 0
        ):
            raise ValueError("Geostrophic reference/tolerance is invalid.")
        self.reference_surface = jnp.asarray(surface)
        self.reference_discharge = jnp.asarray(discharge)
        self.coriolis_source, self.tolerance = coriolis_source, tol
        self.plan_id = canonical_fingerprint(
            {
                "kind": "geostrophic-balance",
                "surface": array_tree_fingerprint(surface),
                "discharge": array_tree_fingerprint(discharge),
                "coriolis": str(coriolis_source),
                "tolerance": tol,
            }
        )

    def prepare(self, bathymetry, geometry_id, residual, args=None, /):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        bed = jnp.asarray(bathymetry)
        depth = self.reference_surface - bed
        if bed.shape != self.reference_surface.shape or np.any(np.asarray(depth) < 0):
            raise ValueError("Geostrophic bed/reference is invalid.")
        state = jnp.concatenate((depth[..., None], self.reference_discharge), -1)
        reference = jnp.asarray(residual(state, args))
        if (
            reference.shape != state.shape
            or float(np.asarray(jnp.max(jnp.abs(reference)))) > self.tolerance
        ):
            raise ValueError(
                "Geostrophic reference fails the discrete balance tolerance."
            )
        identity = str(geometry_id)
        if not identity:
            raise ValueError("geometry_id must be non-empty.")
        return PreparedGeostrophicBalance(
            state,
            reference,
            identity,
            canonical_fingerprint({"plan": self.plan_id, "geometry": identity}),
        )


class ShallowWaterShorelineEvent(StrictModule):
    cell_index: Array
    face_index: Array
    root_time: Array
    pre_state: Array
    post_state: Array
    event_normal: Array
    event_normal_derivative: Array
    wet_mask_change: Array
    reset_jacobian: Array
    status: ShorelineDerivativeStatus = eqx.field(static=True)

    def saltation_action(self, tangent, pre_rate, post_rate, /):
        if self.status != "isolated-event":
            raise ValueError("Saltation requires an isolated transverse shoreline event.")
        tangent_ = jnp.asarray(tangent)
        before = jnp.asarray(pre_rate)
        after = jnp.asarray(post_rate)
        denominator = eqx.error_if(
            self.event_normal_derivative,
            jnp.abs(self.event_normal_derivative) <= jnp.finfo(tangent_.dtype).eps,
            "Grazing shoreline event has no derivative.",
        )
        root_shift = (
            ein.contract("i,i->", self.event_normal, tangent_, backend="jax")
            / denominator
        )
        return (
            self.reset_jacobian @ tangent_
            + (after - self.reset_jacobian @ before) * root_shift
        )


__all__ = [
    "GeostrophicBalancePlan",
    "PreparedGeostrophicBalance",
    "ShallowWaterBoundaryStatus",
    "ShallowWaterBoundaryTrace",
    "ShallowWaterCharacteristicOpenBoundary",
    "ShallowWaterEquilibriumWENOZPlan",
    "ShallowWaterNormalDischargeBoundary",
    "ShallowWaterReconstructionEvidence",
    "ShallowWaterShorelineEvent",
    "ShorelineDerivativeStatus",
]
