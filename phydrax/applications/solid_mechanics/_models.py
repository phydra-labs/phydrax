#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import CellEnergyAction, FiniteElementForm
from ...linalg import SmallLinearSolvePlan, solve_small_linear


class NeoHookeanParameters(StrictModule, NonTrainableState):
    """Logarithmic compressible Neo-Hookean Lamé parameters."""

    shear_modulus: Array
    lame_lambda: Array

    def __init__(self, shear_modulus: ArrayLike, lame_lambda: ArrayLike, /):
        shear = jnp.asarray(shear_modulus)
        lambda_ = jnp.asarray(lame_lambda)
        bulk = lambda_ + (2.0 / 3.0) * shear
        if (
            shear.shape != ()
            or lambda_.shape != ()
            or not bool(jnp.isfinite(shear))
            or not bool(jnp.isfinite(lambda_))
            or shear <= 0.0
            or bulk <= 0.0
        ):
            raise ValueError(
                "Neo-Hookean shear modulus and implied bulk modulus must be "
                "positive finite scalars."
            )
        self.shear_modulus = shear
        self.lame_lambda = lambda_

    @classmethod
    def from_shear_bulk(
        cls,
        shear_modulus: ArrayLike,
        bulk_modulus: ArrayLike,
        /,
    ) -> "NeoHookeanParameters":
        shear = jnp.asarray(shear_modulus)
        bulk = jnp.asarray(bulk_modulus)
        if (
            shear.shape != ()
            or bulk.shape != ()
            or not bool(jnp.isfinite(shear))
            or not bool(jnp.isfinite(bulk))
            or shear <= 0.0
            or bulk <= 0.0
        ):
            raise ValueError(
                "Neo-Hookean shear and bulk moduli must be positive finite scalars."
            )
        return cls(shear, bulk - (2.0 / 3.0) * shear)

    @property
    def bulk_modulus(self) -> Array:
        return self.lame_lambda + (2.0 / 3.0) * self.shear_modulus


def _embed_neo_hookean_deformation(deformation_gradient: ArrayLike, /) -> Array:
    deformation = jnp.asarray(deformation_gradient)
    if deformation.shape[-2:] == (3, 3):
        return deformation
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("Neo-Hookean deformation gradients must end in 2x2 or 3x3.")
    embedded = jnp.zeros(deformation.shape[:-2] + (3, 3), dtype=deformation.dtype)
    embedded = embedded.at[..., :2, :2].set(deformation)
    return embedded.at[..., 2, 2].set(1.0)


def neo_hookean_first_piola(
    deformation_gradient: ArrayLike,
    parameters: NeoHookeanParameters,
    /,
) -> Array:
    deformation = jnp.asarray(deformation_gradient)
    if deformation.shape[-2:] != (3, 3):
        raise ValueError("Neo-Hookean deformation gradients must end in 3x3.")
    plan = SmallLinearSolvePlan(3)
    inverse = solve_small_linear(
        plan,
        deformation,
        jnp.broadcast_to(jnp.eye(3, dtype=deformation.dtype), deformation.shape),
    )
    determinant = inverse.determinant
    valid = inverse.successful & jnp.isfinite(determinant) & (determinant > 0.0)
    determinant = jnp.where(valid, determinant, jnp.nan)
    inverse_transpose = inverse.value.swapaxes(-1, -2)
    return parameters.shear_modulus * (deformation - inverse_transpose) + (
        parameters.lame_lambda * jnp.log(determinant)[..., None, None] * inverse_transpose
    )


def neo_hookean_reference_energy(
    deformation_gradient: ArrayLike,
    parameters: NeoHookeanParameters,
    /,
) -> Array:
    """Reference-volume logarithmic compressible Neo-Hookean energy."""
    deformation = jnp.asarray(deformation_gradient)
    if deformation.shape[-2:] != (3, 3):
        raise ValueError("Neo-Hookean deformation gradients must end in 3x3.")
    plan = SmallLinearSolvePlan(3)
    inverse = solve_small_linear(
        plan,
        deformation,
        jnp.broadcast_to(jnp.eye(3, dtype=deformation.dtype), deformation.shape),
    )
    determinant = inverse.determinant
    valid = inverse.successful & jnp.isfinite(determinant) & (determinant > 0.0)
    logarithm = jnp.log(jnp.where(valid, determinant, jnp.nan))
    invariant = jnp.sum(deformation * deformation, axis=(-2, -1))
    return (
        0.5 * parameters.shear_modulus * (invariant - 3.0)
        - parameters.shear_modulus * logarithm
        + 0.5 * parameters.lame_lambda * logarithm**2
    )


def neo_hookean_form(
    field_name: str,
    parameters: NeoHookeanParameters,
    /,
    *,
    form_id: str = "neo-hookean-equilibrium",
) -> FiniteElementForm:
    if not isinstance(parameters, NeoHookeanParameters):
        raise TypeError("parameters must be NeoHookeanParameters.")

    def density(values, gradients, points, context):
        del values, points, context
        displacement_gradient = jnp.swapaxes(jnp.asarray(gradients), -1, -2)
        if displacement_gradient.shape[-2:] not in ((2, 2), (3, 3)):
            raise ValueError("Neo-Hookean displacement gradients must end in 2x2 or 3x3.")
        deformation = _embed_neo_hookean_deformation(
            jnp.eye(displacement_gradient.shape[-1]) + displacement_gradient
        )
        return neo_hookean_reference_energy(deformation, parameters)

    return FiniteElementForm(
        form_id,
        field_name,
        (
            CellEnergyAction(
                field_name,
                density,
                action_id="neo-hookean-reference-energy",
            ),
        ),
    )


class J2PlasticityParameters(StrictModule, NonTrainableState):
    shear_modulus: Array
    bulk_modulus: Array
    yield_stress: Array
    hardening_modulus: Array

    def __init__(
        self,
        shear_modulus: ArrayLike,
        bulk_modulus: ArrayLike,
        yield_stress: ArrayLike,
        hardening_modulus: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                shear_modulus,
                bulk_modulus,
                yield_stress,
                hardening_modulus,
            )
        )
        if any(value.shape != () or not bool(jnp.isfinite(value)) for value in values):
            raise ValueError("J2 parameters must be finite scalars.")
        if any(value <= 0.0 for value in values[:3]) or values[3] < 0.0:
            raise ValueError("J2 moduli/yield data are inadmissible.")
        (
            self.shear_modulus,
            self.bulk_modulus,
            self.yield_stress,
            self.hardening_modulus,
        ) = values


class J2PlasticityState(StrictModule):
    plastic_strain: Array
    equivalent_plastic_strain: Array

    def __init__(
        self,
        plastic_strain: ArrayLike,
        equivalent_plastic_strain: ArrayLike,
        /,
    ):
        plastic = jnp.asarray(plastic_strain)
        equivalent = jnp.asarray(equivalent_plastic_strain)
        if plastic.shape[-2:] != (3, 3) or equivalent.shape != plastic.shape[:-2]:
            raise ValueError("J2 state shapes are incompatible.")
        self.plastic_strain = plastic
        self.equivalent_plastic_strain = equivalent


class J2PlasticityUpdate(StrictModule):
    stress: Array
    state: J2PlasticityState
    yielded: Array
    plastic_increment: Array


def j2_radial_return(
    strain: ArrayLike,
    state: J2PlasticityState,
    parameters: J2PlasticityParameters,
    /,
) -> J2PlasticityUpdate:
    strain_ = jnp.asarray(strain)
    if strain_.shape[-2:] != (3, 3):
        raise ValueError("J2 strain must end in 3x3.")
    identity = jnp.eye(3, dtype=strain_.dtype)
    elastic = strain_ - state.plastic_strain
    trace = jnp.trace(elastic, axis1=-2, axis2=-1)
    deviatoric_strain = elastic - trace[..., None, None] * identity / 3.0
    trial_deviator = 2.0 * parameters.shear_modulus * deviatoric_strain
    trial_norm = jnp.sqrt(jnp.sum(trial_deviator**2, axis=(-2, -1)))
    equivalent_stress = jnp.sqrt(1.5) * trial_norm
    current_yield = parameters.yield_stress + parameters.hardening_modulus * (
        state.equivalent_plastic_strain
    )
    yielded = equivalent_stress > current_yield
    denominator = 3.0 * parameters.shear_modulus + parameters.hardening_modulus
    increment = jnp.where(yielded, (equivalent_stress - current_yield) / denominator, 0.0)
    direction = trial_deviator / jnp.maximum(
        trial_norm[..., None, None], jnp.finfo(strain_.dtype).tiny
    )
    plastic_increment = 1.5 * increment[..., None, None] * direction
    plastic_strain = state.plastic_strain + plastic_increment
    equivalent_plastic = state.equivalent_plastic_strain + increment
    corrected_deviator = (
        trial_deviator - 2.0 * parameters.shear_modulus * plastic_increment
    )
    pressure = parameters.bulk_modulus * trace
    stress = corrected_deviator + pressure[..., None, None] * identity
    return J2PlasticityUpdate(
        stress=stress,
        state=J2PlasticityState(plastic_strain, equivalent_plastic),
        yielded=yielded,
        plastic_increment=increment,
    )


__all__ = [
    "J2PlasticityParameters",
    "J2PlasticityState",
    "J2PlasticityUpdate",
    "NeoHookeanParameters",
    "j2_radial_return",
    "neo_hookean_first_piola",
    "neo_hookean_reference_energy",
    "neo_hookean_form",
]
