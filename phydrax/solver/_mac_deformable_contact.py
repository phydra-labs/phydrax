#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


DeformableContactKinematics = Callable[[Array, Array, Any], tuple[Array, Array]]
DeformableContactAssembly = Callable[[Array, Array, Any], Array]


class DeformableContactResidualEvaluation(StrictModule):
    residual: Array
    contact: object
    normal_power: Array
    dissipation_rate: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DeformableContactResidualPlan(StrictModule, NonTrainableState):
    """Native deformable-contact adapter exposed as a structural residual."""

    contact: object
    query_kinematics: DeformableContactKinematics = eqx.field(static=True)
    surface_kinematics: DeformableContactKinematics = eqx.field(static=True)
    assemble_residual: DeformableContactAssembly = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        contact: object,
        query_kinematics: DeformableContactKinematics,
        surface_kinematics: DeformableContactKinematics,
        assemble_residual: DeformableContactAssembly,
        /,
        *,
        kinematics_id: str,
        assembly_id: str,
    ):
        if not all(
            callable(value)
            for value in (
                query_kinematics,
                surface_kinematics,
                assemble_residual,
            )
        ):
            raise TypeError("Deformable contact adapters must be callable.")
        kinematics_identifier = str(kinematics_id)
        assembly_identifier = str(assembly_id)
        if not kinematics_identifier or not assembly_identifier:
            raise ValueError("Deformable contact residual identities must be nonempty.")
        self.contact = contact
        self.query_kinematics = query_kinematics
        self.surface_kinematics = surface_kinematics
        self.assemble_residual = assemble_residual
        self.plan_id = canonical_fingerprint(
            {
                "kind": "deformable-contact-structural-residual",
                "contact": contact.adapter_id,
                "kinematics": kinematics_identifier,
                "assembly": assembly_identifier,
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
        residual = self.assemble_residual(
            -contact.transpose.query_force,
            -contact.transpose.surface_force,
            args,
        )
        dissipation_rate = jnp.maximum(-contact.normal_power, 0.0)
        finite = (
            contact.finite
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(contact.normal_power)
            & jnp.isfinite(dissipation_rate)
        )
        successful = contact.successful & finite
        return DeformableContactResidualEvaluation(
            residual,
            contact,
            contact.normal_power,
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
