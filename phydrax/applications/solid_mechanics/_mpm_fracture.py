#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
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
from ._models import NeoHookeanParameters


class MPMPhaseFieldParameters(StrictModule, NonTrainableState):
    material: NeoHookeanParameters
    critical_energy_release_rate: Array
    length_scale: Array
    residual_stiffness: Array

    def __init__(
        self,
        material: NeoHookeanParameters,
        critical_energy_release_rate: ArrayLike,
        length_scale: ArrayLike,
        residual_stiffness: ArrayLike = 1.0e-6,
        /,
    ):
        if not isinstance(material, NeoHookeanParameters):
            raise TypeError("material must be NeoHookeanParameters.")
        values = tuple(
            jnp.asarray(value)
            for value in (
                critical_energy_release_rate,
                length_scale,
                residual_stiffness,
            )
        )
        if any(value.shape != () for value in values) or any(
            not bool(jnp.isfinite(value)) for value in values
        ):
            raise ValueError("Phase-field parameters must be finite scalars.")
        if values[0] <= 0.0 or values[1] <= 0.0 or not 0.0 < values[2] < 1.0:
            raise ValueError("Phase-field fracture parameters are inadmissible.")
        self.material = material
        self.critical_energy_release_rate = values[0]
        self.length_scale = values[1]
        self.residual_stiffness = values[2]


class PhaseFieldNeoHookeanMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan):
    """Spectral tension/compression split with particle damage/history state."""

    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Phase-field MPM supports plane strain and 3-D.")
        self.dimension = dimension_
        self.kinematics = "plane_strain" if dimension_ == 2 else "three_dimensional"
        self.state_shape = (2,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=True,
            has_algorithmic_tangent=True,
            has_dissipation=True,
            has_tension_compression_split=True,
            supports_implicit=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "phase-field-neo-hookean-mpm",
                "dimension": dimension_,
                "split": "spectral-hencky",
                "damage": "AT2",
            }
        )

    def initialize_state(self, batch_shape, dtype, /):
        return jnp.zeros(tuple(batch_shape) + (2,), dtype=dtype)

    def _embed(self, deformation):
        if self.dimension == 3:
            return deformation
        embedded = jnp.eye(3, dtype=deformation.dtype)
        return embedded.at[:2, :2].set(deformation)

    @staticmethod
    def _split_energy(embedded, parameters):
        right_cauchy = embedded.T @ embedded
        eigenvalues = jnp.linalg.eigvalsh(right_cauchy)
        valid = jnp.all(eigenvalues > 0.0) & jnp.all(jnp.isfinite(eigenvalues))
        logarithmic = 0.5 * jnp.log(jnp.where(eigenvalues > 0.0, eigenvalues, 1.0))
        positive = jnp.maximum(logarithmic, 0.0)
        negative = jnp.minimum(logarithmic, 0.0)
        positive_trace = jnp.maximum(jnp.sum(logarithmic), 0.0)
        negative_trace = jnp.minimum(jnp.sum(logarithmic), 0.0)
        positive_energy = (
            parameters.material.shear_modulus * jnp.sum(positive**2)
            + 0.5 * parameters.material.bulk_modulus * positive_trace**2
        )
        negative_energy = (
            parameters.material.shear_modulus * jnp.sum(negative**2)
            + 0.5 * parameters.material.bulk_modulus * negative_trace**2
        )
        return positive_energy, negative_energy, valid

    def _point(self, deformation, state, density, parameters):
        damage = jnp.clip(state[0], 0.0, 1.0)
        history = jnp.maximum(state[1], 0.0)

        def degraded_energy(value):
            embedded = self._embed(value)
            positive, negative, _ = self._split_energy(embedded, parameters)
            degradation = (1.0 - damage) ** 2 + parameters.residual_stiffness
            return degradation * positive + negative

        energy = degraded_energy(deformation)
        stress = jax.grad(degraded_energy)(deformation)
        embedded = self._embed(deformation)
        positive, _, split_valid = self._split_energy(embedded, parameters)
        next_history = jnp.maximum(history, positive)
        speed = jnp.sqrt(
            (
                parameters.material.bulk_modulus
                + 4.0 * parameters.material.shear_modulus / 3.0
            )
            / jnp.where(density > 0.0, density, 1.0)
        )
        valid = (
            split_valid
            & jnp.all(jnp.isfinite(stress))
            & jnp.isfinite(energy)
            & jnp.isfinite(speed)
            & (density > 0.0)
        )
        trial = jnp.asarray((damage, next_history), dtype=deformation.dtype)
        return stress, trial, energy, speed, valid, positive

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: MPMPhaseFieldParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time, step_size
        if not isinstance(parameters, MPMPhaseFieldParameters):
            raise TypeError("parameters must be MPMPhaseFieldParameters.")
        deformation = jnp.asarray(deformation_gradient)
        batch_shape = deformation.shape[:-2]
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        if deformation.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError("Phase-field deformation dimension changed.")
        if state.shape != batch_shape + (2,) or density.shape != batch_shape:
            raise ValueError("Phase-field state/density shape changed.")
        outputs = jax.vmap(
            lambda value, history, rho: self._point(value, history, rho, parameters)
        )(
            deformation.reshape((-1, self.dimension, self.dimension)),
            state.reshape((-1, 2)),
            density.reshape((-1,)),
        )
        stress, trial, energy, speed, valid, positive = outputs
        dissipation = parameters.critical_energy_release_rate * jnp.maximum(
            trial[:, 0] - state.reshape((-1, 2))[:, 0], 0.0
        )
        return MPMConstitutiveResponse(
            stress.reshape(batch_shape + (self.dimension, self.dimension)),
            trial.reshape(batch_shape + (2,)),
            energy.reshape(batch_shape),
            speed.reshape(batch_shape),
            dissipation_increment=dissipation.reshape(batch_shape),
            branch_code=(trial[:, 0] > 0.0).astype(jnp.int32).reshape(batch_shape),
            successful=valid.reshape(batch_shape),
            admissible=valid.reshape(batch_shape),
            diagnostics={"tensile_energy_density": positive.reshape(batch_shape)},
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: MPMPhaseFieldParameters,
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
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        batch_shape = deformation.shape[:-2]
        flat_f = deformation.reshape((-1, self.dimension, self.dimension))
        flat_state = state.reshape((-1, 2))
        flat_density = density.reshape((-1,))

        def stress(value, history, rho):
            return self._point(value, history, rho, parameters)[0]

        tangent = jax.vmap(jax.jacfwd(stress, argnums=0))(
            flat_f, flat_state, flat_density
        ).reshape(
            batch_shape
            + (
                self.dimension,
                self.dimension,
                self.dimension,
                self.dimension,
            )
        )
        successful = jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1))
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)


__all__ = [
    "MPMPhaseFieldParameters",
    "PhaseFieldNeoHookeanMPMConstitutivePlan",
]
