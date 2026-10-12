#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMKinematics,
    MPMLinearizedConstitutiveResponse,
)
from ...linalg import (
    hermitian_log,
    hermitian_spectral_map,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...typing import checked
from ._models import NeoHookeanParameters


# Stress, history, energy, wave speed, validity, tensile energy, split regularity.
type _PhaseFieldPointOutputs = tuple[Array, Array, Array, Array, Array, Array, Array]


def _split_principal(
    logarithmic: Array, parameters: tuple[Array, Array, Array]
) -> tuple[Array, dict[str, Array]]:
    """Permutation-equivariant AT2 split with invariant energy and kink evidence."""
    shear, bulk, dimension = parameters
    positive = jnp.maximum(logarithmic, 0.0)
    negative = jnp.minimum(logarithmic, 0.0)
    trace = jnp.sum(logarithmic)
    positive_trace = jnp.maximum(trace, 0.0)
    negative_trace = jnp.minimum(trace, 0.0)
    # Plane strain has one identically zero, inactive out-of-plane strain.
    active_zero = jnp.sum(logarithmic == 0.0, dtype=jnp.int32) > (3 - dimension)
    confluent_tolerance = jnp.sqrt(jnp.finfo(logarithmic.dtype).eps) * jnp.maximum(
        jnp.max(jnp.abs(logarithmic)), 1.0
    )
    mixed_sign = (logarithmic[:, None] > 0.0) & (logarithmic[None, :] < 0.0)
    mixed_confluent = jnp.any(
        mixed_sign
        & (jnp.abs(logarithmic[:, None] - logarithmic[None, :]) <= confluent_tolerance)
    )
    return jnp.stack((positive, negative)), {
        "positive_energy": shear * jnp.sum(positive**2) + 0.5 * bulk * positive_trace**2,
        "negative_energy": shear * jnp.sum(negative**2) + 0.5 * bulk * negative_trace**2,
        "positive_trace": positive_trace,
        "negative_trace": negative_trace,
        "split_derivative_valid": ~active_zero & ~mixed_confluent & (trace != 0.0),
    }


@final
class MPMPhaseFieldParameters(StrictModule, NonTrainableState):
    material: NeoHookeanParameters
    critical_energy_release_rate: Array
    length_scale: Array
    residual_stiffness: Array

    @checked
    def __init__(
        self,
        material: NeoHookeanParameters,
        critical_energy_release_rate: ArrayLike,
        length_scale: ArrayLike,
        residual_stiffness: ArrayLike = 1.0e-6,
        /,
    ) -> None:
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


@final
class PhaseFieldNeoHookeanMPMConstitutivePlan(
    AbstractImplicitMPMConstitutivePlan,
    NonTrainableState,
):
    """AT2 spectral Hencky split with particle damage/history state.

    H = log(FᵀF)/2 is evaluated through the Hermitian spectral substrate.
    ψ± = G H±:H± + K <tr H>±²/2 retains this material's elastic energy convention.
    With fixed damage d, T = ∂ψ/∂H and P = F⁻ᵀ T because T commutes with H.
    This explicit first-Piola update needs only first-order spectral derivatives;
    differentiating a spectral energy gradient would require unsupported second
    derivatives at repeated spectra. Primal stress is finite at split kinks, but
    the algorithmic tangent is certified only away from active principal-strain
    zeros, mixed-sign confluent pairs, and tr H = 0. The plane-strain zero is inactive.
    Invalid kinematics or history retain committed history, without clipping.
    """

    dimension: int = eqx.field(static=True)
    kinematics: MPMKinematics = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)
    solve_plan: SmallLinearSolvePlan

    def __init__(self, dimension: int, /) -> None:
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
        self.solve_plan = SmallLinearSolvePlan(3)

    def initialize_state(
        self, batch_shape: tuple[int, ...], dtype: DTypeLike, /
    ) -> Array:
        return jnp.zeros(tuple(batch_shape) + (2,), dtype=dtype)

    def _embed(self, deformation: Array) -> Array:
        if self.dimension == 3:
            return deformation
        embedded = jnp.eye(3, dtype=deformation.dtype)
        return embedded.at[:2, :2].set(deformation)

    def _point(
        self,
        deformation: Array,
        state: Array,
        density: Array,
        parameters: MPMPhaseFieldParameters,
    ) -> _PhaseFieldPointOutputs:
        embedded = self._embed(deformation)
        probe = solve_small_linear(self.solve_plan, embedded.T, jnp.zeros_like(embedded))
        domain = probe.successful & (probe.determinant > 0.0)
        # Domain rejection is explicit: the identity only keeps the rejected
        # numerical branch finite, and never becomes an accepted deformation.
        safe_deformation = jnp.where(domain, embedded, jnp.eye(3, dtype=embedded.dtype))
        logarithm = hermitian_log(safe_deformation.T @ safe_deformation, tolerance=0.0)
        split = hermitian_spectral_map(
            0.5 * logarithm.value,
            _split_principal,
            (
                parameters.material.shear_modulus,
                parameters.material.bulk_modulus,
                jnp.asarray(self.dimension, dtype=jnp.int32),
            ),
            function_id="mpm-at2-hencky-split",
        )
        invariants = split.invariants
        positive = invariants["positive_energy"]
        degradation = (1.0 - state[0]) ** 2 + parameters.residual_stiffness
        energy = degradation * positive + invariants["negative_energy"]
        hencky_stress = 2.0 * parameters.material.shear_modulus * (
            degradation * split.values[0] + split.values[1]
        ) + parameters.material.bulk_modulus * (
            degradation * invariants["positive_trace"] + invariants["negative_trace"]
        ) * jnp.eye(3, dtype=embedded.dtype)
        piola = solve_small_linear(self.solve_plan, safe_deformation.T, hencky_stress)
        stress = piola.value[: self.dimension, : self.dimension]
        speed = jnp.sqrt(
            (
                parameters.material.bulk_modulus
                + 4.0 * parameters.material.shear_modulus / 3.0
            )
            / jnp.where(density > 0.0, density, 1.0)
        )
        valid = (
            domain
            & logarithm.valid
            & split.valid
            & piola.successful
            & jnp.all(jnp.isfinite(state))
            & (state[0] >= 0.0)
            & (state[0] <= 1.0)
            & (state[1] >= 0.0)
            & jnp.all(jnp.isfinite(stress))
            & jnp.isfinite(energy)
            & jnp.isfinite(speed)
            & (density > 0.0)
        )
        trial = jnp.stack((state[0], jnp.maximum(state[1], positive)))
        split_regular = (
            valid
            & logarithm.derivative_valid
            & split.derivative_valid
            & invariants["split_derivative_valid"]
        )
        return (
            jnp.where(valid, stress, jnp.zeros_like(stress)),
            jnp.where(valid, trial, state),
            jnp.where(valid, energy, 0.0),
            jnp.where(valid, speed, 0.0),
            valid,
            jnp.where(valid, positive, 0.0),
            split_regular,
        )

    @checked
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
        stress, trial, energy, speed, valid, positive, split_regular = outputs
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
            diagnostics={
                "tensile_energy_density": positive.reshape(batch_shape),
                "split_derivative_valid": split_regular.reshape(batch_shape),
            },
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

        def stress(value: Array, history: Array, rho: Array) -> Array:
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
        successful = (
            response.successful
            & response.diagnostics["split_derivative_valid"]
            & jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1))
        )
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)


__all__ = [
    "MPMPhaseFieldParameters",
    "PhaseFieldNeoHookeanMPMConstitutivePlan",
]
