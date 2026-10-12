#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-strain J2 radial return in principal elastic Hencky strains.

With tension-positive tau, p=-tr(tau)/3 and q=sqrt(3 s:s/2), associated
flow reduces the trial q by 3G delta_gamma. Linear isotropic hardening
increases the strength by H delta_gamma, hence consistency gives
delta_gamma=(q_trial-sigma_y-H alpha)/(3G+H). The returned elastic
metric exp(2 epsilon_e) is pulled back by the shared multiplicative kernel.
Stored hardening energy is H alpha²/2 and dissipation is sigma_y delta_gamma.
"""

from __future__ import annotations

from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)
from ...typing import checked
from ._principal_hencky import (
    elastic_principal_metric,
    evaluate_principal_point,
    initialize_metric_state,
    initialize_state_from_stress,
    PrestressedMaterialState,
    principal_trial,
)


if TYPE_CHECKING:
    from ._mpm_plane_stress import PlaneStressMPMConstitutivePlan


def _j2_principal_map(
    eigenvalues: Array,
    context: tuple[FiniteStrainJ2Parameters, Array],
) -> tuple[Array, dict[str, Array]]:
    parameters, scalars = context
    alpha = scalars[0]
    _, _, pressure, q, deviatoric = principal_trial(
        eigenvalues, parameters.shear_modulus, parameters.bulk_modulus
    )
    strength = parameters.yield_stress + parameters.hardening_modulus * alpha
    trial_yield = q - strength
    stress_scale = jnp.maximum(
        parameters.shear_modulus, jnp.maximum(q, jnp.abs(strength))
    )
    branch_tolerance = (
        jnp.maximum(1e-10, 64 * jnp.finfo(eigenvalues.dtype).eps) * stress_scale
    )
    increment = jax.lax.cond(
        trial_yield > 0.0,
        lambda f: f / (3.0 * parameters.shear_modulus + parameters.hardening_modulus),
        lambda f: jnp.zeros_like(f),
        trial_yield,
    )
    scale = jax.lax.cond(
        q > 0.0,
        lambda value: 1.0 - 3.0 * parameters.shear_modulus * increment / value,
        lambda value: jnp.ones_like(value),
        q,
    )
    tau = scale * deviatoric - pressure
    corrected = elastic_principal_metric(
        tau, parameters.shear_modulus, parameters.bulk_modulus
    )
    next_alpha = alpha + increment
    mean = jnp.mean(tau)
    elastic_energy = jnp.sum((tau - mean) ** 2) / (
        4.0 * parameters.shear_modulus
    ) + mean**2 / (2.0 * parameters.bulk_modulus)
    residual = (
        q
        - 3.0 * parameters.shear_modulus * increment
        - parameters.yield_stress
        - parameters.hardening_modulus * next_alpha
    )
    return jnp.stack((tau, corrected)), {
        "scalars": next_alpha[None],
        "branch_code": jnp.where(increment > 0.0, 1, 0).astype(jnp.int32),
        "successful": (alpha >= 0.0) & jnp.isfinite(residual),
        "strict_branch": jnp.abs(trial_yield) > branch_tolerance,
        "energy": elastic_energy + 0.5 * parameters.hardening_modulus * next_alpha**2,
        "dissipation": parameters.yield_stress * increment,
        "yield_residual": residual,
        "plastic_multiplier": increment,
    }


def _j2_yield(
    stress: Array, parameters: FiniteStrainJ2Parameters, scalars: Array
) -> Array:
    mean = jnp.trace(stress, axis1=-2, axis2=-1) / 3.0
    deviatoric = stress - mean[..., None, None] * jnp.eye(3, dtype=stress.dtype)
    squared = 1.5 * jnp.sum(deviatoric**2, axis=(-2, -1))
    q = jnp.sqrt(squared)
    return q - parameters.yield_stress - parameters.hardening_modulus * scalars[..., 0]


@final
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
    ) -> None:
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


@final
class FiniteStrainJ2MPMConstitutivePlan(
    AbstractImplicitMPMConstitutivePlan,
    NonTrainableState,
):
    """Multiplicative finite-strain J2 update with Hencky elasticity."""

    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    yield_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, yield_tolerance: float = 1.0e-10) -> None:
        tolerance = float(yield_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("yield_tolerance must be finite and positive.")
        self.dimension = 3
        self.kinematics = "three_dimensional"
        self.state_shape = (7,)
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

    def initialize_state(
        self, batch_shape: tuple[int, ...], dtype: DTypeLike, /
    ) -> Array:
        return initialize_metric_state(batch_shape, dtype)

    def initialize_state_from_stress(
        self,
        kirchhoff_stress: ArrayLike,
        parameters: FiniteStrainJ2Parameters,
        /,
        *,
        equivalent_plastic_strain: ArrayLike = 0.0,
    ) -> PrestressedMaterialState:
        stress = jnp.asarray(kirchhoff_stress)
        if stress.shape[-2:] != (3, 3):
            raise ValueError("Kirchhoff stress must end in 3x3.")
        alpha = jnp.asarray(equivalent_plastic_strain, dtype=stress.dtype)
        alpha = jnp.broadcast_to(alpha, stress.shape[:-2])
        result = initialize_state_from_stress(
            stress,
            parameters,
            alpha[..., None],
            _j2_yield,
            tolerance=self.yield_tolerance,
        )
        return PrestressedMaterialState(
            result.state, result.admissible & (alpha >= 0.0), result.yield_value
        )

    def _point(
        self,
        deformation: Array,
        history: Array,
        density: Array,
        parameters: FiniteStrainJ2Parameters,
        step_size: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        return evaluate_principal_point(
            deformation,
            history,
            density,
            parameters,
            _j2_principal_map,
            function_id="finite-strain-j2-hencky",
            step_size=step_size,
            wave_modulus=parameters.bulk_modulus + 4.0 * parameters.shear_modulus / 3.0,
        )

    @checked
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
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (3, 3):
            raise ValueError("Finite-strain J2 deformation must end in 3x3.")
        batch_shape = deformation.shape[:-2]
        history = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        if history.shape != batch_shape + (7,) or density.shape != batch_shape:
            raise ValueError("Finite-strain J2 state/density shape changed.")
        flat_f = deformation.reshape((-1, 3, 3))
        flat_history = history.reshape((-1, 7))
        flat_density = density.reshape((-1,))
        outputs = jax.vmap(
            lambda value, state, rho: self._point(
                value,
                state,
                rho,
                parameters,
                jnp.asarray(step_size, dtype=deformation.dtype),
            )
        )(flat_f, flat_history, flat_density)
        (
            stress,
            trial,
            energy,
            speed,
            dissipation,
            branch,
            finite,
            diagnostics,
        ) = outputs
        diagnostics["tangent_successful"] &= diagnostics["strict_branch"]
        suggested = jnp.where(
            finite,
            jnp.asarray(jnp.inf, dtype=deformation.dtype),
            0.5 * jnp.asarray(step_size, dtype=deformation.dtype),
        )
        return MPMConstitutiveResponse(
            stress.reshape(batch_shape + (3, 3)),
            trial.reshape(batch_shape + (7,)),
            energy.reshape(batch_shape),
            speed.reshape(batch_shape),
            dissipation_increment=dissipation.reshape(batch_shape),
            branch_code=branch.reshape(batch_shape),
            suggested_step=suggested.reshape(batch_shape),
            successful=finite.reshape(batch_shape),
            admissible=finite.reshape(batch_shape),
            diagnostics={
                key: value.reshape(batch_shape + value.shape[1:])
                for key, value in diagnostics.items()
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
        flat_history = history.reshape((-1, 7))
        flat_density = density.reshape((-1,))

        def stress(value: Array, state: Array, rho: Array) -> Array:
            return self._point(
                value, state, rho, parameters, jnp.asarray(step_size, dtype=value.dtype)
            )[0]

        tangent = jax.vmap(jax.jacfwd(stress, argnums=0))(
            flat_f, flat_history, flat_density
        )
        tangent = tangent.reshape(batch_shape + (3, 3, 3, 3))
        successful = (
            response.successful
            & response.diagnostics["tangent_successful"]
            & jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1))
        )
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)


def finite_strain_j2_plane_stress_plan(
    *,
    yield_tolerance: float = 1.0e-10,
) -> PlaneStressMPMConstitutivePlan:
    from ._mpm_plane_stress import PlaneStressMPMConstitutivePlan

    return PlaneStressMPMConstitutivePlan(
        FiniteStrainJ2MPMConstitutivePlan(yield_tolerance=yield_tolerance)
    )


__all__ = [
    "FiniteStrainJ2MPMConstitutivePlan",
    "FiniteStrainJ2Parameters",
    "finite_strain_j2_plane_stress_plan",
]
