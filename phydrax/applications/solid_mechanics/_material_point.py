#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...equations import AbstractMPMConstitutivePlan, MPMConstitutiveResponse
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._models import (
    _embed_neo_hookean_deformation,
    neo_hookean_first_piola,
    neo_hookean_reference_energy,
    NeoHookeanParameters,
)


class NeoHookeanMPMConstitutivePlan(AbstractMPMConstitutivePlan):
    """Stateless logarithmic Neo-Hookean material for plane strain or 3-D MPM."""

    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Neo-Hookean MPM supports plane strain and 3-D only.")
        self.dimension = dimension_
        self.kinematics = "plane_strain" if dimension_ == 2 else "three_dimensional"
        self.state_shape = (0,)
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
        return _embed_neo_hookean_deformation(deformation)

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

        embedded = self._embed(deformation)
        inverse = solve_small_linear(
            SmallLinearSolvePlan(3),
            embedded,
            jnp.broadcast_to(jnp.eye(3, dtype=embedded.dtype), embedded.shape),
        )
        determinant = inverse.determinant
        finite_input = jnp.all(jnp.isfinite(embedded), axis=(-2, -1))
        valid = (
            finite_input
            & inverse.successful
            & jnp.isfinite(determinant)
            & (determinant > 0.0)
            & jnp.isfinite(density)
            & (density > 0.0)
        )

        first_piola_3d = neo_hookean_first_piola(embedded, parameters)
        energy = neo_hookean_reference_energy(embedded, parameters)
        first_piola = first_piola_3d[..., : self.dimension, : self.dimension]

        inverse_in_plane = inverse.value[..., : self.dimension, : self.dimension]
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
                "inverse_condition_estimate": inverse.condition_estimate,
                "inverse_norm_squared_bound": inverse_norm_squared_bound,
                "longitudinal_modulus_bound": longitudinal_modulus_bound,
            },
        )


__all__ = ["NeoHookeanMPMConstitutivePlan"]
