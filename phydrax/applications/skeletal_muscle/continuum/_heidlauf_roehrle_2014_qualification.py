#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Numerical consistency evidence, not held-out muscle validation, for 2014 Eq. 1."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._identity import SemanticProvenance
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._heidlauf_roehrle_2014 import PreparedHeidlaufRoehrle2014Material


class HeidlaufRoehrle2014QualificationEvidence(StrictModule, NonTrainableState):
    passive_gradient_relative_error: Array
    objectivity_relative_error: Array
    tangent_difference_relative_error: Array
    active_power_relative_error: Array
    pressure_adjoint_relative_error: Array
    valid: Array
    qualification_id: str = eqx.field(static=True)


class HeidlaufRoehrle2014QualificationPlan(StrictModule, NonTrainableState):
    """One fixed smooth positive-J point, including compression below lambda=1."""

    relative_tolerance: float = eqx.field(static=True)
    difference_step: float = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)

    def __init__(self, *, relative_tolerance=5.0e-3, difference_step=2.0e-3):
        tolerance = float(relative_tolerance)
        step = float(difference_step)
        if (
            not isfinite(tolerance)
            or tolerance <= 0.0
            or not isfinite(step)
            or step <= 0.0
        ):
            raise ValueError(
                "Qualification tolerance and difference step must be finite and positive."
            )
        self.relative_tolerance = tolerance
        self.difference_step = step
        self.qualification_id = SemanticProvenance(
            {
                "kind": "heidlauf-roehrle-2014-point-consistency",
                "relative_tolerance": tolerance,
                "difference_step": step,
                "rotation_angle": 0.37,
                "scope": "material-implementation-only;no-physiological-validation",
            }
        ).semantic_id

    def evaluate(
        self,
        material: PreparedHeidlaufRoehrle2014Material,
        deformation,
        pressure,
        deformation_rate,
        /,
    ) -> HeidlaufRoehrle2014QualificationEvidence:
        if not isinstance(material, PreparedHeidlaufRoehrle2014Material):
            raise TypeError("material must be PreparedHeidlaufRoehrle2014Material.")
        deformation = jnp.asarray(deformation)
        rate = jnp.asarray(deformation_rate)
        if deformation.shape != (3, 3) or rate.shape != (3, 3):
            raise ValueError("Qualification deformation and rate must be (3, 3).")
        angle = jnp.asarray(0.37, dtype=deformation.dtype)
        rotation = jnp.asarray(
            (
                (jnp.cos(angle), -jnp.sin(angle), 0.0),
                (jnp.sin(angle), jnp.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        response = material.evaluate(deformation, pressure)
        rotated = material.evaluate(rotation @ deformation, pressure)
        gradient = jax.grad(material.passive_energy_density)(deformation)
        tangent = material.block_tangent(deformation, pressure)
        directional = jnp.tensordot(
            tangent.deformation_deformation, rate, axes=((2, 3), (0, 1))
        )
        step = self.difference_step
        positive = material.evaluate(deformation + step * rate, pressure)
        negative = material.evaluate(deformation - step * rate, pressure)
        difference = (positive.first_piola - negative.first_piola) / (2.0 * step)
        current_direction = deformation @ material.architecture.reference_direction
        current_direction = current_direction / jnp.linalg.norm(current_direction)
        stretch_rate = jnp.dot(
            current_direction, rate @ material.architecture.reference_direction
        )
        active_power = jnp.sum(response.active_first_piola * rate)
        expected_power = response.active_nominal_stress_pa * stretch_rate

        def relative(left, right):
            return jnp.linalg.norm(left - right) / jnp.maximum(
                1.0, jnp.linalg.norm(right)
            )

        errors = jnp.stack(
            (
                relative(gradient, response.passive_first_piola),
                relative(rotated.first_piola, rotation @ response.first_piola),
                relative(difference, directional),
                relative(active_power, expected_power),
                relative(tangent.deformation_pressure, tangent.constraint_deformation),
            )
        )
        valid = (
            jnp.all(jnp.isfinite(errors))
            & jnp.all(errors <= self.relative_tolerance)
            & response.evidence.valid
            & rotated.evidence.valid
            & positive.evidence.valid
            & negative.evidence.valid
        )
        return HeidlaufRoehrle2014QualificationEvidence(
            *errors, valid, self.qualification_id
        )


__all__ = [
    "HeidlaufRoehrle2014QualificationEvidence",
    "HeidlaufRoehrle2014QualificationPlan",
]
