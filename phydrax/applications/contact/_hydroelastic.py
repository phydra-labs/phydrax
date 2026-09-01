#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._interface import (
    assemble_contact_interface_traction,
    ContactInterfaceKinematics,
    ContactInterfacePlan,
    ContactInterfaceResidual,
)


class HydroelasticMaterialPlan(StrictModule, NonTrainableState):
    modulus: float = eqx.field(static=True)
    slab_thickness: float = eqx.field(static=True)
    dissipation: float = eqx.field(static=True)
    friction: float = eqx.field(static=True)
    velocity_regularization: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        modulus: float,
        slab_thickness: float,
        dissipation: float = 0.0,
        friction: float = 0.0,
        velocity_regularization: float = 1.0e-6,
    ):
        values = tuple(
            float(value)
            for value in (
                modulus,
                slab_thickness,
                dissipation,
                friction,
                velocity_regularization,
            )
        )
        if (
            values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] <= 0.0
            or any(not np.isfinite(value) for value in values)
        ):
            raise ValueError("Hydroelastic material parameters are invalid.")
        (
            self.modulus,
            self.slab_thickness,
            self.dissipation,
            self.friction,
            self.velocity_regularization,
        ) = values
        self.material_id = canonical_fingerprint(
            {
                "kind": "hydroelastic-material-plan",
                "parameters": tuple(value.hex() for value in values),
            }
        )

    def pressure(self, compression: ArrayLike, /) -> Array:
        compression_ = jnp.asarray(compression)
        return self.modulus / self.slab_thickness * jnp.maximum(compression_, 0.0)


class HydroelasticContactEvidence(StrictModule):
    patch_area: Array
    resultant_force: Array
    resultant_moment: Array
    minimum_pressure: Array
    dissipated_power: Array
    action_reaction_residual: Array
    finite: Array
    pressure_nonnegative: Array
    dissipative: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class HydroelasticContactResult(StrictModule):
    pressure: Array
    traction: Array
    residual: ContactInterfaceResidual
    evidence: HydroelasticContactEvidence


def evaluate_hydroelastic_contact(
    interface: ContactInterfacePlan,
    kinematics: ContactInterfaceKinematics,
    plus_material: HydroelasticMaterialPlan | None,
    minus_material: HydroelasticMaterialPlan | None,
    relative_velocity: ArrayLike,
    /,
    *,
    plus_pressure: ArrayLike | None = None,
    minus_pressure: ArrayLike | None = None,
) -> HydroelasticContactResult:
    if plus_material is None and minus_material is None:
        raise ValueError("Hydroelastic contact requires at least one compliant side.")
    if kinematics.interface_id != interface.interface_id:
        raise ValueError("Hydroelastic kinematics belongs to another interface.")
    velocity = jnp.asarray(relative_velocity, dtype=kinematics.gap.dtype)
    if velocity.shape != (
        interface.capacity,
        interface.ambient_dimension,
    ):
        raise ValueError("Hydroelastic relative velocity has invalid shape.")
    compression = jnp.maximum(-kinematics.gap, 0.0)

    def side_pressure(material, supplied):
        if supplied is not None:
            value = jnp.asarray(supplied, dtype=kinematics.gap.dtype)
            if value.shape != (interface.capacity,):
                raise ValueError("Hydroelastic pressure trace has invalid shape.")
            return jnp.maximum(value, 0.0)
        if material is None:
            return jnp.full((interface.capacity,), jnp.inf, dtype=kinematics.gap.dtype)
        return material.pressure(compression)

    plus_value = side_pressure(plus_material, plus_pressure)
    minus_value = side_pressure(minus_material, minus_pressure)
    if plus_material is None:
        pressure = minus_value
    elif minus_material is None:
        pressure = plus_value
    else:
        pressure = 0.5 * (plus_value + minus_value)
    normal_velocity = jnp.sum(velocity * kinematics.normal, axis=-1)
    dissipation = (
        (0.0 if plus_material is None else plus_material.dissipation)
        + (0.0 if minus_material is None else minus_material.dissipation)
    ) * 0.5
    pressure = pressure * jnp.maximum(0.0, 1.0 - dissipation * normal_velocity)
    friction = 0.5 * (
        (0.0 if plus_material is None else plus_material.friction)
        + (0.0 if minus_material is None else minus_material.friction)
    )
    regularization = 0.5 * (
        (0.0 if plus_material is None else plus_material.velocity_regularization)
        + (0.0 if minus_material is None else minus_material.velocity_regularization)
    )
    regularization = max(regularization, 1.0e-12)
    tangential_velocity = velocity - normal_velocity[:, None] * kinematics.normal
    tangential_speed = jnp.sqrt(
        jnp.sum(tangential_velocity * tangential_velocity, axis=-1)
        + regularization * regularization
    )
    friction_traction = (
        -(friction * pressure / tangential_speed)[:, None] * tangential_velocity
    )
    active = interface.valid & (pressure > 0.0)
    traction = pressure[:, None] * kinematics.normal + friction_traction
    traction = jnp.where(active[:, None], traction, 0.0)
    pressure = jnp.where(active, pressure, 0.0)
    residual = assemble_contact_interface_traction(interface, traction)
    weights = interface.quadrature_weight.astype(traction.dtype)
    resultant_force = jnp.sum(weights[:, None] * traction, axis=0)
    resultant_moment = (
        jnp.sum(
            weights[:, None] * jnp.cross(kinematics.plus_point, traction),
            axis=0,
        )
        if interface.ambient_dimension == 3
        else jnp.sum(
            weights
            * (
                kinematics.plus_point[:, 0] * traction[:, 1]
                - kinematics.plus_point[:, 1] * traction[:, 0]
            )
        )[None]
    )
    dissipated_power = -jnp.sum(
        weights[:, None] * friction_traction * tangential_velocity
    )
    finite = (
        jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(traction))
        & jnp.all(jnp.isfinite(resultant_force))
        & jnp.all(jnp.isfinite(resultant_moment))
        & jnp.isfinite(dissipated_power)
        & residual.finite
    )
    pressure_nonnegative = jnp.all(pressure >= 0.0)
    dissipative = dissipated_power >= -64.0 * jnp.finfo(traction.dtype).eps
    plan_id = canonical_fingerprint(
        {
            "kind": "hydroelastic-contact-evaluation",
            "interface": interface.interface_id,
            "plus": None if plus_material is None else plus_material.material_id,
            "minus": None if minus_material is None else minus_material.material_id,
        }
    )
    evidence = HydroelasticContactEvidence(
        jnp.sum(jnp.where(active, weights, 0.0)),
        resultant_force,
        resultant_moment,
        jnp.min(jnp.where(active, pressure, jnp.inf), initial=0.0),
        dissipated_power,
        residual.action_reaction_residual,
        finite,
        pressure_nonnegative,
        dissipative,
        finite & pressure_nonnegative & dissipative & residual.successful,
        plan_id,
    )
    return HydroelasticContactResult(pressure, traction, residual, evidence)


__all__ = [
    "HydroelasticContactEvidence",
    "HydroelasticContactResult",
    "HydroelasticMaterialPlan",
    "evaluate_hydroelastic_contact",
]
