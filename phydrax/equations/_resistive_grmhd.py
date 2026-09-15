#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry
from ..metrix._spacetime_conventions import RelativityConvention


def _metric_inner(left: Array, metric: Array, right: Array, /) -> Array:
    return contract("...i,...ij,...j->...", left, metric, right, backend="jax")


def _raise(covector: Array, inverse_metric: Array, /) -> Array:
    return contract("...ij,...j->...i", inverse_metric, covector, backend="jax")


class RelativisticOhmEvaluation(StrictModule):
    spatial_current: Array
    advective_current: Array
    conduction_current: Array
    comoving_electric_covector: Array
    ideal_electric_covector: Array
    ideal_residual_norm: Array
    ideal_limit: Array
    lorentz_factor: Array
    entropy_production: Array
    metric_residual: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    closure_id: str = eqx.field(static=True)


class ResistiveGRMHDOhmicClosure(StrictModule, NonTrainableState):
    """Scalar-conductivity relativistic Ohm closure in a 3+1 Eulerian frame.

    Electric fields are spatial covectors, while magnetic field, velocity and
    current are spatial vectors.  The supplied velocity has ordinary code-speed
    units and must be subluminal.  Conductivity zero gives pure charge advection;
    the ideal field ``E_i = -(v x B)_i / c`` makes the conductive current vanish
    for every finite conductivity, exposing both resistive and ideal limits.
    Hall, pressure-anisotropy and kinetic closures are not represented here.
    """

    scale: RelativityScaleContract
    convention: RelativityConvention
    conductivity: float = eqx.field(static=True)
    metric_tolerance: float = eqx.field(static=True)
    ideal_tolerance: float = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        conductivity: float,
        metric_tolerance: float = 1.0e-9,
        ideal_tolerance: float = 1.0e-10,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if convention.metric_signature != "mostly_plus":
            raise ValueError("Ohmic closure requires the mostly-plus convention.")
        conductivity_ = float(conductivity)
        light_speed = float(scale.speed_of_light)
        metric_tolerance_ = float(metric_tolerance)
        ideal_tolerance_ = float(ideal_tolerance)
        if (
            not np.isfinite(conductivity_)
            or conductivity_ < 0.0
            or not np.isfinite(light_speed)
            or light_speed <= 0.0
            or not np.isfinite(metric_tolerance_)
            or metric_tolerance_ <= 0.0
            or not np.isfinite(ideal_tolerance_)
            or ideal_tolerance_ <= 0.0
        ):
            raise ValueError("Resistive GRMHD Ohmic closure parameters are invalid.")
        self.scale = scale
        self.convention = convention
        self.conductivity = conductivity_
        self.metric_tolerance = metric_tolerance_
        self.ideal_tolerance = ideal_tolerance_
        self.closure_id = canonical_fingerprint(
            {
                "kind": "resistive-grmhd-scalar-ohm-closure",
                "conductivity": conductivity_,
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "metric_tolerance": metric_tolerance_,
                "ideal_tolerance": ideal_tolerance_,
            }
        )

    @property
    def speed_of_light(self) -> float:
        return float(self.scale.speed_of_light)

    def ideal_electric_field(
        self,
        magnetic_field: ArrayLike,
        velocity: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> Array:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and Ohmic closure contracts differ.")
        magnetic = jnp.asarray(magnetic_field)
        velocity_ = jnp.asarray(velocity, dtype=magnetic.dtype)
        if (
            magnetic.shape != geometry.leading_shape + (3,)
            or velocity_.shape != magnetic.shape
        ):
            raise ValueError("Magnetic field and velocity must match ADM geometry.")
        cross_covector = (
            self.convention.spacetime_orientation
            * self.convention.future_time_orientation
            * geometry.sqrt_det_spatial_metric[..., None]
            * jnp.cross(velocity_, magnetic)
        )
        return -cross_covector / self.speed_of_light

    def evaluate(
        self,
        electric_covector: ArrayLike,
        magnetic_field: ArrayLike,
        velocity: ArrayLike,
        charge_density: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> RelativisticOhmEvaluation:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and Ohmic closure contracts differ.")
        electric = jnp.asarray(electric_covector)
        magnetic = jnp.asarray(magnetic_field, dtype=electric.dtype)
        velocity_ = jnp.asarray(velocity, dtype=electric.dtype)
        charge = jnp.asarray(charge_density, dtype=electric.dtype)
        metric = geometry.spatial_metric.astype(electric.dtype)
        inverse = geometry.inverse_spatial_metric.astype(electric.dtype)
        volume = geometry.sqrt_det_spatial_metric.astype(electric.dtype)
        if electric.shape[-1:] != (3,):
            raise ValueError("Electric covector must have trailing size three.")
        cell_shape = geometry.leading_shape
        if (
            electric.shape != cell_shape + (3,)
            or magnetic.shape != cell_shape + (3,)
            or velocity_.shape != cell_shape + (3,)
            or charge.shape != cell_shape
        ):
            raise ValueError(
                "Resistive GRMHD fields and metric have incompatible shapes."
            )
        light_speed = jnp.asarray(self.speed_of_light, dtype=electric.dtype)
        normalized_velocity = velocity_ / light_speed
        velocity_covector = contract(
            "...ij,...j->...i", metric, normalized_velocity, backend="jax"
        )
        speed_squared = _metric_inner(normalized_velocity, metric, normalized_velocity)
        safe_lorentz_denominator = jnp.maximum(
            1.0 - speed_squared, jnp.finfo(electric.dtype).tiny
        )
        lorentz = 1.0 / jnp.sqrt(safe_lorentz_denominator)
        cross_covector = (
            self.convention.spacetime_orientation
            * self.convention.future_time_orientation
            * volume[..., None]
            * jnp.cross(velocity_, magnetic)
            / light_speed
        )
        ideal_electric = -cross_covector
        velocity_electric = contract(
            "...i,...i->...", normalized_velocity, electric, backend="jax"
        )
        comoving_electric = (
            electric + cross_covector - velocity_electric[..., None] * velocity_covector
        )
        conduction = (
            jnp.asarray(self.conductivity, dtype=electric.dtype)
            * lorentz[..., None]
            * _raise(comoving_electric, inverse)
        )
        advective = charge[..., None] * velocity_
        current = advective + conduction
        ideal_residual = electric - ideal_electric
        ideal_residual_squared = _metric_inner(
            _raise(ideal_residual, inverse), metric, _raise(ideal_residual, inverse)
        )
        ideal_residual_norm = jnp.sqrt(jnp.maximum(ideal_residual_squared, 0.0))
        ideal_limit = ideal_residual_norm <= self.ideal_tolerance
        comoving_squared = _metric_inner(
            _raise(comoving_electric, inverse),
            metric,
            _raise(comoving_electric, inverse),
        )
        entropy = self.conductivity * lorentz * jnp.maximum(comoving_squared, 0.0)
        identity = contract("...ik,...kj->...ij", metric, inverse, backend="jax")
        metric_residual = jnp.max(
            jnp.abs(identity - jnp.eye(3, dtype=electric.dtype)), axis=(-2, -1)
        )
        finite = (
            geometry.finite
            & jnp.all(jnp.isfinite(electric), axis=-1)
            & jnp.all(jnp.isfinite(magnetic), axis=-1)
            & jnp.all(jnp.isfinite(velocity_), axis=-1)
            & jnp.isfinite(charge)
            & jnp.all(jnp.isfinite(current), axis=-1)
            & jnp.isfinite(entropy)
        )
        physically_valid = (
            finite
            & geometry.physically_valid
            & (speed_squared >= 0.0)
            & (speed_squared < 1.0)
            & (entropy >= 0.0)
        )
        qualified = physically_valid & (metric_residual <= self.metric_tolerance)
        derivative_valid = qualified & (
            speed_squared < 1.0 - 32.0 * jnp.finfo(electric.dtype).eps
        )
        return RelativisticOhmEvaluation(
            current,
            advective,
            conduction,
            comoving_electric,
            ideal_electric,
            ideal_residual_norm,
            ideal_limit,
            lorentz,
            entropy,
            metric_residual,
            finite,
            physically_valid,
            qualified,
            derivative_valid,
            self.closure_id,
        )
