#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)
from ...linalg import SmallLinearSolvePlan, solve_small_linear


class FiniteStrainJ2Parameters(StrictModule, NonTrainableState):
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
        values = tuple(
            jnp.asarray(value)
            for value in (
                shear_modulus,
                bulk_modulus,
                yield_stress,
                hardening_modulus,
            )
        )
        if any(value.shape != () for value in values) or any(
            not bool(jnp.isfinite(value)) for value in values
        ):
            raise ValueError("Finite-strain J2 parameters must be finite scalars.")
        if values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or values[3] < 0.0:
            raise ValueError("Finite-strain J2 moduli/yield stress are inadmissible.")
        (
            self.shear_modulus,
            self.bulk_modulus,
            self.yield_stress,
            self.hardening_modulus,
        ) = values


class FiniteStrainJ2MPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan):
    """Multiplicative finite-strain J2 update with Hencky elasticity."""

    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    yield_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, yield_tolerance: float = 1.0e-10):
        tolerance = float(yield_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("yield_tolerance must be finite and positive.")
        self.dimension = 3
        self.kinematics = "three_dimensional"
        self.state_shape = (10,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=True,
            has_algorithmic_tangent=True,
            has_dissipation=True,
            supports_implicit=True,
        )
        self.yield_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-strain-j2-mpm",
                "elasticity": "hencky",
                "flow": "multiplicative-associated-j2",
                "hardening": "linear-isotropic",
                "yield_tolerance": tolerance,
            }
        )

    def initialize_state(self, batch_shape, dtype, /):
        shape = tuple(batch_shape)
        plastic = jnp.broadcast_to(jnp.eye(3, dtype=dtype), shape + (3, 3))
        alpha = jnp.zeros(shape + (1,), dtype=dtype)
        return jnp.concatenate((plastic.reshape(shape + (9,)), alpha), axis=-1)

    @staticmethod
    def _inverse(value):
        identity = jnp.broadcast_to(jnp.eye(3, dtype=value.dtype), value.shape)
        return solve_small_linear(SmallLinearSolvePlan(3), value, identity)

    def _point(self, deformation, history, density, parameters):
        plastic = history[:9].reshape((3, 3))
        alpha = history[9]
        inverse_plastic = self._inverse(plastic)
        elastic_trial = deformation @ inverse_plastic.value
        left, stretches, right_transpose = jnp.linalg.svd(
            elastic_trial, full_matrices=False
        )
        valid_stretch = jnp.all(jnp.isfinite(stretches)) & jnp.all(stretches > 0.0)
        log_stretch = jnp.log(jnp.where(stretches > 0.0, stretches, 1.0))
        mean_log = jnp.mean(log_stretch)
        deviatoric_log = log_stretch - mean_log
        trial_deviatoric = 2.0 * parameters.shear_modulus * deviatoric_log
        trial_q = jnp.sqrt(1.5 * jnp.sum(trial_deviatoric**2))
        yield_strength = parameters.yield_stress + parameters.hardening_modulus * alpha
        yield_function = trial_q - yield_strength
        plastic_branch = yield_function > self.yield_tolerance
        denominator = 3.0 * parameters.shear_modulus + parameters.hardening_modulus
        increment = jnp.where(
            plastic_branch,
            jnp.maximum(yield_function, 0.0) / denominator,
            0.0,
        )
        scale = jnp.where(
            trial_q > self.yield_tolerance,
            jnp.maximum(
                0.0,
                1.0 - 3.0 * parameters.shear_modulus * increment / trial_q,
            ),
            1.0,
        )
        corrected_deviatoric = trial_deviatoric * scale
        corrected_log = corrected_deviatoric / (2.0 * parameters.shear_modulus) + mean_log
        corrected_elastic = (left * jnp.exp(corrected_log)[None, :]) @ right_transpose
        inverse_elastic = self._inverse(corrected_elastic)
        next_plastic = inverse_elastic.value @ deformation
        inverse_next_plastic = self._inverse(next_plastic)
        plastic_determinant = inverse_next_plastic.determinant
        normalization = jnp.cbrt(
            jnp.where(plastic_determinant > 0.0, plastic_determinant, 1.0)
        )
        next_plastic = next_plastic / normalization
        next_alpha = alpha + increment
        principal_tau = corrected_deviatoric + parameters.bulk_modulus * jnp.sum(
            corrected_log
        )
        kirchhoff = (left * principal_tau[None, :]) @ left.T
        inverse_deformation = self._inverse(deformation)
        first_piola = kirchhoff @ inverse_deformation.value.T
        energy = (
            parameters.shear_modulus
            * jnp.sum((corrected_log - jnp.mean(corrected_log)) ** 2)
            + 0.5 * parameters.bulk_modulus * jnp.sum(corrected_log) ** 2
            + 0.5 * parameters.hardening_modulus * next_alpha**2
        )
        dissipation = parameters.yield_stress * increment
        inverse_transpose = inverse_deformation.value.T
        absolute_inverse = jnp.abs(inverse_transpose)
        norm_bound = jnp.sqrt(
            jnp.max(jnp.sum(absolute_inverse, axis=0))
            * jnp.max(jnp.sum(absolute_inverse, axis=1))
        )
        speed = jnp.sqrt(
            (parameters.bulk_modulus + 4.0 * parameters.shear_modulus / 3.0)
            / jnp.where(density > 0.0, density, 1.0)
        ) * jnp.maximum(norm_bound, 1.0)
        finite = (
            inverse_plastic.successful
            & inverse_elastic.successful
            & inverse_next_plastic.successful
            & inverse_deformation.successful
            & valid_stretch
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.isfinite(energy)
            & jnp.isfinite(speed)
            & (plastic_determinant > 0.0)
            & (density > 0.0)
        )
        state = jnp.concatenate((next_plastic.reshape((9,)), next_alpha[None]))
        return (
            jnp.where(finite, first_piola, 0.0),
            jnp.where(finite, state, history),
            jnp.where(finite, energy, 0.0),
            jnp.where(finite, speed, 0.0),
            jnp.where(finite, dissipation, 0.0),
            jnp.where(plastic_branch, 1, 0).astype(jnp.int32),
            finite,
            yield_function,
            increment,
            plastic_determinant,
        )

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: FiniteStrainJ2Parameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time
        if not isinstance(parameters, FiniteStrainJ2Parameters):
            raise TypeError("parameters must be FiniteStrainJ2Parameters.")
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (3, 3):
            raise ValueError("Finite-strain J2 deformation must end in 3x3.")
        batch_shape = deformation.shape[:-2]
        history = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        if history.shape != batch_shape + (10,) or density.shape != batch_shape:
            raise ValueError("Finite-strain J2 state/density shape changed.")
        flat_f = deformation.reshape((-1, 3, 3))
        flat_history = history.reshape((-1, 10))
        flat_density = density.reshape((-1,))
        outputs = jax.vmap(
            lambda value, state, rho: self._point(value, state, rho, parameters)
        )(flat_f, flat_history, flat_density)
        (
            stress,
            trial,
            energy,
            speed,
            dissipation,
            branch,
            finite,
            yield_value,
            increment,
            detp,
        ) = outputs
        suggested = jnp.where(
            finite,
            jnp.asarray(jnp.inf, dtype=deformation.dtype),
            0.5 * jnp.asarray(step_size, dtype=deformation.dtype),
        )
        return MPMConstitutiveResponse(
            stress.reshape(batch_shape + (3, 3)),
            trial.reshape(batch_shape + (10,)),
            energy.reshape(batch_shape),
            speed.reshape(batch_shape),
            dissipation_increment=dissipation.reshape(batch_shape),
            branch_code=branch.reshape(batch_shape),
            suggested_step=suggested.reshape(batch_shape),
            successful=finite.reshape(batch_shape),
            admissible=finite.reshape(batch_shape),
            diagnostics={
                "yield_function": yield_value.reshape(batch_shape),
                "plastic_multiplier": increment.reshape(batch_shape),
                "plastic_determinant": detp.reshape(batch_shape),
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: FiniteStrainJ2Parameters,
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
        deformation = jnp.asarray(deformation_gradient)
        history = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        batch_shape = deformation.shape[:-2]
        flat_f = deformation.reshape((-1, 3, 3))
        flat_history = history.reshape((-1, 10))
        flat_density = density.reshape((-1,))

        def stress(value, state, rho):
            return self._point(value, state, rho, parameters)[0]

        tangent = jax.vmap(jax.jacfwd(stress, argnums=0))(
            flat_f, flat_history, flat_density
        )
        tangent = tangent.reshape(batch_shape + (3, 3, 3, 3))
        finite_tangent = jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1))
        identity = jnp.eye(3, dtype=deformation.dtype)
        lame = parameters.bulk_modulus - 2.0 * parameters.shear_modulus / 3.0
        elastic_identity_tangent = lame * oe.contract(
            "ij,kl->ijkl", identity, identity
        ) + parameters.shear_modulus * (
            oe.contract("ik,jl->ijkl", identity, identity)
            + oe.contract("il,jk->ijkl", identity, identity)
        )
        near_identity = jnp.linalg.norm(deformation - identity, axis=(-2, -1)) <= 1.0e-10
        replace = (~finite_tangent) & near_identity & (response.branch_code == 0)
        tangent = jnp.where(
            replace[..., None, None, None, None],
            elastic_identity_tangent,
            tangent,
        )
        successful = jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1))
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)


def finite_strain_j2_plane_stress_plan(
    *,
    yield_tolerance: float = 1.0e-10,
):
    from ._mpm_plane_stress import IsotropicPlaneStressMPMConstitutivePlan

    return IsotropicPlaneStressMPMConstitutivePlan(
        FiniteStrainJ2MPMConstitutivePlan(yield_tolerance=yield_tolerance)
    )


__all__ = [
    "FiniteStrainJ2MPMConstitutivePlan",
    "FiniteStrainJ2Parameters",
    "finite_strain_j2_plane_stress_plan",
]
