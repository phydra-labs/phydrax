#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import (
    DeformableContactEvaluation,
    DeformableContactTransposeResult,
    PreparedDeformableContact,
)


DeformableContactKinematics = Callable[[Array, Array, Any], tuple[Array, Array]]
DeformableContactAssembly = Callable[[Array, Array, Array, Any], Array]


class DeformableContactResidualEvaluation(StrictModule):
    residual: Array
    contact: DeformableContactEvaluation
    transpose: DeformableContactTransposeResult
    route_force: Array
    elastic_energy: Array
    dissipation_rate: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DeformableContactResidualPlan(StrictModule, NonTrainableState):
    """Native deformable-contact geometry and transpose as a structural residual."""

    contact: PreparedDeformableContact
    query_kinematics: DeformableContactKinematics = eqx.field(static=True)
    surface_kinematics: DeformableContactKinematics = eqx.field(static=True)
    assemble_residual: DeformableContactAssembly = eqx.field(static=True)
    stiffness: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        contact: PreparedDeformableContact,
        query_kinematics: DeformableContactKinematics,
        surface_kinematics: DeformableContactKinematics,
        assemble_residual: DeformableContactAssembly,
        /,
        *,
        stiffness: float,
        damping: float = 0.0,
        kinematics_id: str,
        assembly_id: str,
    ):
        if not isinstance(contact, PreparedDeformableContact):
            raise TypeError("contact must be PreparedDeformableContact.")
        if not all(
            callable(value)
            for value in (
                query_kinematics,
                surface_kinematics,
                assemble_residual,
            )
        ):
            raise TypeError("Deformable contact adapters must be callable.")
        stiffness_ = float(stiffness)
        damping_ = float(damping)
        if (
            not np.isfinite(stiffness_)
            or stiffness_ <= 0.0
            or not np.isfinite(damping_)
            or damping_ < 0.0
            or not str(kinematics_id)
            or not str(assembly_id)
        ):
            raise ValueError("Deformable contact residual policy is invalid.")
        self.contact = contact
        self.query_kinematics = query_kinematics
        self.surface_kinematics = surface_kinematics
        self.assemble_residual = assemble_residual
        self.stiffness = stiffness_
        self.damping = damping_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "deformable-contact-structural-residual",
                "contact": contact.prepared_id,
                "kinematics": str(kinematics_id),
                "assembly": str(assembly_id),
                "stiffness": stiffness_,
                "damping": damping_,
            }
        )

    def evaluate(
        self,
        configuration: Array,
        velocity: Array,
        args: Any = None,
        /,
    ) -> DeformableContactResidualEvaluation:
        query_position, query_velocity = self.query_kinematics(
            configuration, velocity, args
        )
        surface_position, surface_velocity = self.surface_kinematics(
            configuration, velocity, args
        )
        contact = self.contact.evaluate(
            query_position,
            query_velocity,
            surface_position,
            surface_velocity,
        )
        normal_velocity = jnp.sum(contact.relative_velocity * contact.normal, axis=-1)
        penetration = jnp.maximum(-contact.gap, 0.0)
        closing = jnp.maximum(-normal_velocity, 0.0)
        magnitude = self.stiffness * penetration + self.damping * closing
        route_force = jnp.where(
            contact.valid[:, None], magnitude[:, None] * contact.normal, 0.0
        )
        transpose = self.contact.transpose(contact, route_force)
        residual = self.assemble_residual(
            -transpose.query_action,
            -transpose.surface_action,
            -transpose.plane_action,
            args,
        )
        elastic_energy = 0.5 * self.stiffness * jnp.sum(penetration**2)
        dissipation_rate = self.damping * jnp.sum(closing**2)
        finite = (
            contact.finite
            & transpose.finite
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(elastic_energy)
            & jnp.isfinite(dissipation_rate)
        )
        successful = contact.successful & transpose.successful & finite
        return DeformableContactResidualEvaluation(
            residual,
            contact,
            transpose,
            route_force,
            elastic_energy,
            dissipation_rate,
            finite,
            successful,
            self.plan_id,
        )

    def __call__(
        self,
        configuration: Array,
        velocity: Array,
        args: Any = None,
        /,
    ) -> Array:
        return self.evaluate(configuration, velocity, args).residual


__all__ = [
    "DeformableContactAssembly",
    "DeformableContactKinematics",
    "DeformableContactResidualEvaluation",
    "DeformableContactResidualPlan",
]
