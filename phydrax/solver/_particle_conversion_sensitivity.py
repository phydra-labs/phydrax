#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._hybrid_sensitivity import HybridSensitivityMode
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import (
    particle_conversion_diagnostics,
    ParticleConversionState,
)
from ..equations import ParticleConversionEvaluation


class ParticleConversionSensitivityPolicy(StrictModule, NonTrainableState):
    mode: HybridSensitivityMode = eqx.field(static=True)
    species_margin: float = eqx.field(static=True)
    porosity_margin: float = eqx.field(static=True)
    scale_margin: float = eqx.field(static=True)
    temperature_margin: float = eqx.field(static=True)
    phase_margin: float = eqx.field(static=True)
    reaction_margin: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: HybridSensitivityMode = HybridSensitivityMode.SHARP_BRANCHWISE,
        species_margin: float = 1.0e-10,
        porosity_margin: float = 1.0e-8,
        scale_margin: float = 1.0e-10,
        temperature_margin: float = 1.0e-6,
        phase_margin: float = 1.0e-10,
        reaction_margin: float = 1.0e-10,
    ):
        if not isinstance(mode, HybridSensitivityMode):
            raise TypeError("mode must be a HybridSensitivityMode.")
        values = tuple(
            float(value)
            for value in (
                species_margin,
                porosity_margin,
                scale_margin,
                temperature_margin,
                phase_margin,
                reaction_margin,
            )
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Conversion sensitivity margins must be finite and nonnegative."
            )
        self.mode = mode
        (
            self.species_margin,
            self.porosity_margin,
            self.scale_margin,
            self.temperature_margin,
            self.phase_margin,
            self.reaction_margin,
        ) = values


class ParticleConversionValidityCertificate(StrictModule, NonTrainableState):
    species_margin: Array
    porosity_margin: Array
    scale_margin: Array
    temperature_margin: Array
    phase_margin: Array
    reaction_margin: Array
    locally_valid: Array
    successful: Array


class ParticleConversionSensitivityResult(StrictModule):
    primal: Any
    sensitivity: Any
    certificate: ParticleConversionValidityCertificate
    usable: Array
    mode: HybridSensitivityMode = eqx.field(static=True)


class ParticleConversionSurrogateBiasCertificate(StrictModule):
    state_relative_error: Array
    observable_relative_error: Array
    acceptable: Array


def particle_conversion_validity_certificate(
    state: ParticleConversionState,
    evaluation: ParticleConversionEvaluation,
    policy: ParticleConversionSensitivityPolicy,
    /,
) -> ParticleConversionValidityCertificate:
    if not isinstance(policy, ParticleConversionSensitivityPolicy):
        raise TypeError("policy must be ParticleConversionSensitivityPolicy.")
    diagnostics = particle_conversion_diagnostics(state)
    temperature_margin = jnp.min(
        jnp.stack(
            tuple(
                jnp.min(value.transport.thermodynamic_state.temperature_margin)
                for value in evaluation.batches
            )
        )
    )
    phase_margin = jnp.min(
        jnp.stack(
            tuple(
                jnp.asarray(jnp.inf, dtype=temperature_margin.dtype)
                if value.phase_change is None
                else value.phase_change.phase_margin
                for value in evaluation.batches
            )
        )
    )
    reaction_margin = jnp.min(
        jnp.stack(
            tuple(
                jnp.asarray(jnp.inf, dtype=temperature_margin.dtype)
                if value.reaction is None
                else value.reaction.reactant_margin
                for value in evaluation.batches
            )
        )
    )
    locally_valid = (
        (diagnostics.minimum_species_margin > policy.species_margin)
        & (diagnostics.minimum_porosity_margin > policy.porosity_margin)
        & (diagnostics.minimum_scale_margin > policy.scale_margin)
        & (temperature_margin > policy.temperature_margin)
        & (phase_margin > policy.phase_margin)
        & (reaction_margin > policy.reaction_margin)
    )
    successful = diagnostics.successful & evaluation.successful
    return ParticleConversionValidityCertificate(
        diagnostics.minimum_species_margin,
        diagnostics.minimum_porosity_margin,
        diagnostics.minimum_scale_margin,
        temperature_margin,
        phase_margin,
        reaction_margin,
        locally_valid,
        successful,
    )


def sharp_particle_conversion_jvp(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    direction: PyTree[Any],
    state: ParticleConversionState,
    evaluation: ParticleConversionEvaluation,
    policy: ParticleConversionSensitivityPolicy,
    /,
) -> ParticleConversionSensitivityResult:
    if policy.mode is not HybridSensitivityMode.SHARP_BRANCHWISE:
        raise ValueError("sharp_particle_conversion_jvp requires branchwise mode.")
    primal, sensitivity = jax.jvp(function, (parameters,), (direction,))
    certificate = particle_conversion_validity_certificate(state, evaluation, policy)
    usable = certificate.locally_valid & certificate.successful
    sensitivity = _mask_sensitivity(sensitivity, usable)
    return ParticleConversionSensitivityResult(
        primal, sensitivity, certificate, usable, policy.mode
    )


def sharp_particle_conversion_vjp(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    cotangent: PyTree[Any],
    state: ParticleConversionState,
    evaluation: ParticleConversionEvaluation,
    policy: ParticleConversionSensitivityPolicy,
    /,
) -> ParticleConversionSensitivityResult:
    if policy.mode is not HybridSensitivityMode.SHARP_BRANCHWISE:
        raise ValueError("sharp_particle_conversion_vjp requires branchwise mode.")
    primal, pullback = jax.vjp(function, parameters)
    sensitivity = pullback(cotangent)[0]
    certificate = particle_conversion_validity_certificate(state, evaluation, policy)
    usable = certificate.locally_valid & certificate.successful
    sensitivity = _mask_sensitivity(sensitivity, usable)
    return ParticleConversionSensitivityResult(
        primal, sensitivity, certificate, usable, policy.mode
    )


def particle_conversion_surrogate_bias(
    sharp_state: Array,
    smooth_state: Array,
    sharp_observable: Array,
    smooth_observable: Array,
    /,
    *,
    tolerance: float,
) -> ParticleConversionSurrogateBiasCertificate:
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    state_error = jnp.linalg.norm(smooth_state - sharp_state) / jnp.maximum(
        jnp.linalg.norm(sharp_state), 1.0e-30
    )
    observable_error = jnp.linalg.norm(
        smooth_observable - sharp_observable
    ) / jnp.maximum(jnp.linalg.norm(sharp_observable), 1.0e-30)
    return ParticleConversionSurrogateBiasCertificate(
        state_error,
        observable_error,
        (state_error <= tolerance_) & (observable_error <= tolerance_),
    )


def _mask_sensitivity(value, usable):
    return jax.tree.map(
        lambda leaf: (
            jnp.where(usable, leaf, jnp.nan) if eqx.is_inexact_array(leaf) else leaf
        ),
        value,
    )


__all__ = [
    "ParticleConversionSensitivityPolicy",
    "ParticleConversionSensitivityResult",
    "ParticleConversionSurrogateBiasCertificate",
    "ParticleConversionValidityCertificate",
    "particle_conversion_surrogate_bias",
    "particle_conversion_validity_certificate",
    "sharp_particle_conversion_jvp",
    "sharp_particle_conversion_vjp",
]
