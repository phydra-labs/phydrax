#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)
from ...operators.mechanics import (
    finite_strain_kinematics,
    neo_hookean_first_piola,
    neo_hookean_reference_energy,
    neo_hookean_tangent,
    NeoHookeanParameters,
)


class NeoHookeanMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan):
    """Stateless logarithmic Neo-Hookean material for 1-D, plane strain, or 3-D MPM."""

    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Neo-Hookean MPM supports dimensions one, two, and three.")
        self.dimension = dimension_
        self.kinematics = (
            "one_dimensional"
            if dimension_ == 1
            else ("plane_strain" if dimension_ == 2 else "three_dimensional")
        )
        self.state_shape = (0,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=False,
            has_free_energy=True,
            has_algorithmic_tangent=True,
            has_dissipation=False,
            supports_implicit=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "neo-hookean-mpm-constitutive",
                "dimension": dimension_,
                "kinematics": self.kinematics,
                "stress_measure": "first-piola",
                "energy_measure": "reference-volume",
            }
        )

    def _embed(self, deformation):
        if self.dimension == 3:
            return deformation
        shape = deformation.shape[:-2] + (3, 3)
        embedded = jnp.zeros(shape, dtype=deformation.dtype)
        embedded = embedded.at[..., : self.dimension, : self.dimension].set(deformation)
        for axis in range(self.dimension, 3):
            embedded = embedded.at[..., axis, axis].set(1.0)
        return embedded

    def initialize_state(self, batch_shape, dtype, /):
        return jnp.empty(tuple(batch_shape) + (0,), dtype=dtype)

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: NeoHookeanParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time, step_size
        if not isinstance(parameters, NeoHookeanParameters):
            raise TypeError("parameters must be NeoHookeanParameters.")
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError(
                "Deformation gradients must end in the constitutive dimension."
            )
        batch_shape = deformation.shape[:-2]
        history = jnp.asarray(committed_state, dtype=deformation.dtype)
        if history.shape != batch_shape + self.state_shape:
            raise ValueError(
                f"Material history must have shape {batch_shape + self.state_shape}."
            )
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        if density.shape != batch_shape:
            raise ValueError(f"Reference density must have shape {batch_shape}.")

        kinematic_response = finite_strain_kinematics(self._embed(deformation))
        determinant = kinematic_response.jacobian
        finite_input = jnp.all(
            jnp.isfinite(kinematic_response.deformation_gradient), axis=(-2, -1)
        )
        valid = (
            finite_input
            & kinematic_response.admissible
            & jnp.isfinite(density)
            & (density > 0.0)
        )

        first_piola_3d = neo_hookean_first_piola(kinematic_response, parameters)
        energy = neo_hookean_reference_energy(kinematic_response, parameters)
        first_piola = first_piola_3d[..., : self.dimension, : self.dimension]

        inverse_in_plane = kinematic_response.inverse_deformation_gradient[
            ..., : self.dimension, : self.dimension
        ]
        absolute_inverse = jnp.abs(inverse_in_plane)
        norm_one = jnp.max(jnp.sum(absolute_inverse, axis=-2), axis=-1)
        norm_infinity = jnp.max(jnp.sum(absolute_inverse, axis=-1), axis=-1)
        inverse_norm_squared_bound = norm_one * norm_infinity
        safe_determinant = jnp.where(valid, determinant, 1.0)
        acoustic_coefficient = (
            parameters.lame_lambda * (1.0 - jnp.log(safe_determinant))
            + parameters.shear_modulus
        )
        longitudinal_modulus_bound = (
            parameters.shear_modulus
            + jnp.maximum(acoustic_coefficient, 0.0) * inverse_norm_squared_bound
        )
        speed = jnp.sqrt(
            jnp.maximum(longitudinal_modulus_bound, 0.0) / jnp.where(valid, density, 1.0)
        )

        first_piola = jnp.where(valid[..., None, None], first_piola, 0.0)
        energy = jnp.where(valid, energy, 0.0)
        speed = jnp.where(valid, speed, 0.0)
        return MPMConstitutiveResponse(
            first_piola,
            history,
            energy,
            speed,
            successful=valid,
            admissible=valid,
            diagnostics={
                "determinant": determinant,
                "inverse_condition_estimate": kinematic_response.inverse_condition_estimate,
                "inverse_norm_squared_bound": inverse_norm_squared_bound,
                "longitudinal_modulus_bound": longitudinal_modulus_bound,
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: NeoHookeanParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMLinearizedConstitutiveResponse:
        response = self.evaluate(
            deformation_gradient,
            committed_state,
            reference_density,
            parameters,
            time,
            step_size,
        )
        tangent_3d = neo_hookean_tangent(
            self._embed(jnp.asarray(deformation_gradient)), parameters
        )
        tangent = tangent_3d[
            ...,
            : self.dimension,
            : self.dimension,
            : self.dimension,
            : self.dimension,
        ]
        tangent_successful = response.successful & jnp.all(
            jnp.isfinite(tangent), axis=(-4, -3, -2, -1)
        )
        tangent = jnp.where(
            tangent_successful[..., None, None, None, None],
            tangent,
            jnp.zeros_like(tangent),
        )
        return MPMLinearizedConstitutiveResponse(response, tangent, tangent_successful)


__all__ = ["NeoHookeanMPMConstitutivePlan"]
