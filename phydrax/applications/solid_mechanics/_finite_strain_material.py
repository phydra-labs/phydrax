#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._trainable import fixed_field
from ...equations import (
    AbstractConstitutiveModel,
    AbstractImplicitMPMConstitutivePlan,
    ConstitutiveResponse,
)
from ...typing import checked


@final
class FiniteStrainConstitutiveModel(AbstractConstitutiveModel):
    """Adapt a three-dimensional MPM material to the FEM constitutive slot.

    Kinematics are the deformation gradient, the response is first Piola stress,
    and history is the material's unchanged MPM history. Reference density is a
    dynamic scalar or site array. Material coefficients remain explicit in
    ``parameters``; the adapter does not change either slot's semantic identity.
    Tangent failure invalidates the FEM response and is also exposed separately.
    Initialize integration-site history with ``material.initialize_state`` before
    constructing a ``MaterialTransaction``; a zero plastic metric is not valid.
    """

    material: AbstractImplicitMPMConstitutivePlan
    reference_density: Array = fixed_field()
    state_shape: tuple[int, ...] = eqx.field(static=True)
    response_shape: tuple[int, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        material: AbstractImplicitMPMConstitutivePlan,
        reference_density: ArrayLike,
        /,
        *,
        model_id: str | None = None,
    ) -> None:
        if material.dimension != 3 or material.kinematics != "three_dimensional":
            raise ValueError("The finite-strain FEM adapter requires a 3-D material.")
        if not material.capabilities.has_algorithmic_tangent:
            raise ValueError("The finite-strain FEM adapter requires a material tangent.")
        density = jnp.asarray(reference_density)
        if not jnp.issubdtype(density.dtype, jnp.inexact):
            density = density.astype(jnp.float64)
        if not bool(jnp.all(jnp.isfinite(density) & (density > 0.0))):
            raise ValueError("Reference density must be finite and positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-strain-constitutive-model",
                    "material": material.plan_id,
                    "reference_density": array_tree_fingerprint(density),
                }
            )
            if model_id is None
            else str(model_id).strip()
        )
        if not identifier:
            raise ValueError("model_id must be non-empty or None.")
        self.material = material
        self.reference_density = density
        self.state_shape = material.state_shape
        self.response_shape = (3, 3)
        self.model_id = identifier

    def local_response(
        self,
        kinematics: Array,
        committed_state: Array,
        parameters: object,
        time: Array,
        dt: Array,
        /,
    ) -> ConstitutiveResponse:
        if kinematics.shape[-2:] != (3, 3):
            raise ValueError("Finite-strain kinematics must end in a 3-by-3 gradient.")
        density = jnp.broadcast_to(
            self.reference_density.astype(kinematics.dtype), kinematics.shape[:-2]
        )
        linearized = self.material.evaluate_linearized(
            kinematics, committed_state, density, parameters, time, dt
        )
        response = linearized.response
        return ConstitutiveResponse(
            response.first_piola,
            response.trial_state,
            consistent_tangent=linearized.algorithmic_tangent,
            energy=response.reference_energy_density,
            dissipation=response.dissipation_increment,
            valid=(
                response.successful & response.admissible & linearized.tangent_successful
            ),
            diagnostics={
                **response.diagnostics,
                "branch_code": response.branch_code,
                "material_successful": response.successful,
                "material_admissible": response.admissible,
                "tangent_successful": linearized.tangent_successful,
                "maximum_wave_speed": response.maximum_wave_speed,
                "suggested_step": response.suggested_step,
            },
        )


__all__ = ["FiniteStrainConstitutiveModel"]
