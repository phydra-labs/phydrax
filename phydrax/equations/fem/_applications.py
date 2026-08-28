#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._finite_element_material import ConstitutiveResponse


class J2PlasticityParameters(StrictModule):
    shear_modulus: Array
    bulk_modulus: Array
    yield_stress: Array
    hardening_modulus: Array

    def __init__(
        self,
        shear_modulus: ArrayLike,
        bulk_modulus: ArrayLike,
        yield_stress: ArrayLike,
        hardening_modulus: ArrayLike = 0.0,
        /,
    ):
        self.shear_modulus = jnp.asarray(shear_modulus)
        self.bulk_modulus = jnp.asarray(bulk_modulus)
        self.yield_stress = jnp.asarray(yield_stress)
        self.hardening_modulus = jnp.asarray(hardening_modulus)
        if any(
            value.shape != ()
            for value in (
                self.shear_modulus,
                self.bulk_modulus,
                self.yield_stress,
                self.hardening_modulus,
            )
        ):
            raise ValueError("J2 material parameters must be scalar.")


def j2_radial_return(
    strain: ArrayLike,
    committed_state: ArrayLike,
    parameters: J2PlasticityParameters,
    /,
) -> ConstitutiveResponse:
    """Small-strain 3-D Voigt radial return; state is plastic strain plus alpha."""

    strain_ = jnp.asarray(strain)
    state = jnp.asarray(committed_state)
    if strain_.shape[-1] != 6 or state.shape[-1] != 7:
        raise ValueError("J2 strain/state trailing shapes must be 6 and 7.")
    plastic = state[..., :6]
    alpha = state[..., 6]
    elastic = strain_ - plastic
    trace = elastic[..., 0] + elastic[..., 1] + elastic[..., 2]
    deviatoric = elastic.at[..., :3].add(-trace[..., None] / 3.0)
    trial = 2.0 * parameters.shear_modulus * deviatoric
    trial = trial.at[..., :3].add(parameters.bulk_modulus * trace[..., None])
    trial_trace = trial[..., 0] + trial[..., 1] + trial[..., 2]
    stress_dev = trial.at[..., :3].add(-trial_trace[..., None] / 3.0)
    metric_norm = jnp.sqrt(
        jnp.sum(stress_dev[..., :3] ** 2, axis=-1)
        + 2.0 * jnp.sum(stress_dev[..., 3:] ** 2, axis=-1)
    )
    equivalent = jnp.sqrt(1.5) * metric_norm
    yield_value = (
        equivalent - parameters.yield_stress - parameters.hardening_modulus * alpha
    )
    denominator = 3.0 * parameters.shear_modulus + parameters.hardening_modulus
    increment = jnp.maximum(yield_value, 0.0) / denominator
    safe_norm = jnp.where(metric_norm > 0.0, metric_norm, 1.0)
    flow = 1.5 * stress_dev / safe_norm[..., None]
    updated_plastic = plastic + increment[..., None] * flow
    updated_alpha = alpha + increment
    corrected = trial - 2.0 * parameters.shear_modulus * increment[..., None] * flow
    trial_state = jnp.concatenate((updated_plastic, updated_alpha[..., None]), axis=-1)
    return ConstitutiveResponse(
        corrected,
        trial_state,
        diagnostics={
            "plastic_increment": increment,
            "yield_value": yield_value,
            "elastic": yield_value <= 0.0,
        },
    )


class AllenCahnModel(StrictModule):
    mobility: Array
    gradient_coefficient: Array
    local_derivative: object

    def __init__(
        self,
        mobility: ArrayLike,
        gradient_coefficient: ArrayLike,
        local_derivative,
        /,
    ):
        self.mobility = jnp.asarray(mobility)
        self.gradient_coefficient = jnp.asarray(gradient_coefficient)
        self.local_derivative = local_derivative
        if not callable(local_derivative):
            raise TypeError("Allen-Cahn local derivative must be callable.")

    def rate(
        self,
        order_parameter: ArrayLike,
        laplacian: ArrayLike,
        args: object = None,
        /,
    ) -> Array:
        value = jnp.asarray(order_parameter)
        return -self.mobility * (
            self.local_derivative(value, args)
            - self.gradient_coefficient * jnp.asarray(laplacian)
        )


class CahnHilliardModel(StrictModule):
    mobility: Array
    gradient_coefficient: Array
    chemical_derivative: object

    def __init__(
        self,
        mobility: ArrayLike,
        gradient_coefficient: ArrayLike,
        chemical_derivative,
        /,
    ):
        self.mobility = jnp.asarray(mobility)
        self.gradient_coefficient = jnp.asarray(gradient_coefficient)
        self.chemical_derivative = chemical_derivative
        if not callable(chemical_derivative):
            raise TypeError("Cahn-Hilliard chemical derivative must be callable.")

    def chemical_potential(
        self,
        concentration: ArrayLike,
        laplacian_concentration: ArrayLike,
        args: object = None,
        /,
    ) -> Array:
        concentration_ = jnp.asarray(concentration)
        return self.chemical_derivative(
            concentration_, args
        ) - self.gradient_coefficient * jnp.asarray(laplacian_concentration)

    def rate(self, laplacian_chemical_potential: ArrayLike, /) -> Array:
        return self.mobility * jnp.asarray(laplacian_chemical_potential)


class CrystalSlipSystem(StrictModule):
    direction: Array
    normal: Array
    schmid: Array

    def __init__(self, direction: ArrayLike, normal: ArrayLike, /):
        direction_ = jnp.asarray(direction)
        normal_ = jnp.asarray(normal)
        if direction_.shape != (3,) or normal_.shape != (3,):
            raise ValueError("Crystal slip direction and normal must be 3-vectors.")
        direction_ = direction_ / jnp.linalg.norm(direction_)
        normal_ = normal_ / jnp.linalg.norm(normal_)
        if jnp.abs(jnp.dot(direction_, normal_)) > 1.0e-10:
            raise ValueError("Slip direction and normal must be orthogonal.")
        self.direction = direction_
        self.normal = normal_
        self.schmid = jnp.outer(direction_, normal_)


class CrystalPlasticityModel(StrictModule):
    slip_systems: tuple[CrystalSlipSystem, ...]
    reference_rate: Array
    rate_exponent: Array

    def __init__(
        self,
        slip_systems: tuple[CrystalSlipSystem, ...],
        reference_rate: ArrayLike,
        rate_exponent: ArrayLike,
        /,
    ):
        systems = tuple(slip_systems)
        if not systems or not all(
            isinstance(system, CrystalSlipSystem) for system in systems
        ):
            raise ValueError("Crystal plasticity needs one or more slip systems.")
        self.slip_systems = systems
        self.reference_rate = jnp.asarray(reference_rate)
        self.rate_exponent = jnp.asarray(rate_exponent)

    def slip_rates(
        self,
        stress: ArrayLike,
        strengths: ArrayLike,
        /,
    ) -> Array:
        stress_ = jnp.asarray(stress)
        strengths_ = jnp.asarray(strengths)
        schmid = jnp.stack(tuple(system.schmid for system in self.slip_systems))
        resolved = oe.contract("aij,...ij->...a", schmid, stress_)
        ratio = jnp.abs(resolved) / strengths_
        return self.reference_rate * jnp.sign(resolved) * ratio**self.rate_exponent


class PhaseFieldFractureModel(StrictModule):
    critical_energy_release_rate: Array
    length_scale: Array
    residual_stiffness: Array

    def __init__(
        self,
        critical_energy_release_rate: ArrayLike,
        length_scale: ArrayLike,
        residual_stiffness: ArrayLike = 1.0e-8,
        /,
    ):
        self.critical_energy_release_rate = jnp.asarray(critical_energy_release_rate)
        self.length_scale = jnp.asarray(length_scale)
        self.residual_stiffness = jnp.asarray(residual_stiffness)

    def degradation(self, damage: ArrayLike, /) -> Array:
        damage_ = jnp.asarray(damage)
        return (1.0 - damage_) ** 2 + self.residual_stiffness

    def updated_history(
        self,
        tensile_energy: ArrayLike,
        committed_history: ArrayLike,
        /,
    ) -> Array:
        return jnp.maximum(jnp.asarray(tensile_energy), jnp.asarray(committed_history))

    def crack_density(
        self,
        damage: ArrayLike,
        damage_gradient: ArrayLike,
        /,
    ) -> Array:
        damage_ = jnp.asarray(damage)
        gradient = jnp.asarray(damage_gradient)
        return self.critical_energy_release_rate * (
            0.5 * damage_**2 / self.length_scale
            + 0.5 * self.length_scale * jnp.sum(gradient**2, axis=-1)
        )


class FrictionlessContactLaw(StrictModule):
    penalty: Array

    def __init__(self, penalty: ArrayLike, /):
        penalty_ = jnp.asarray(penalty)
        if penalty_.shape != () or penalty_ <= 0.0:
            raise ValueError("Contact penalty must be a positive scalar.")
        self.penalty = penalty_

    def response(
        self,
        gap: ArrayLike,
        normal: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        gap_ = jnp.asarray(gap)
        normal_ = jnp.asarray(normal)
        pressure = self.penalty * jnp.maximum(-gap_, 0.0)
        traction = pressure[..., None] * normal_
        tangent = self.penalty * (gap_ < 0.0)
        return traction, tangent


__all__ = [
    "AllenCahnModel",
    "CahnHilliardModel",
    "CrystalPlasticityModel",
    "CrystalSlipSystem",
    "FrictionlessContactLaw",
    "J2PlasticityParameters",
    "PhaseFieldFractureModel",
    "j2_radial_return",
]
