#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)
from ...operators.mechanics import (
    finite_strain_kinematics,
    first_piola_to_cauchy,
    HyperelasticLaw,
    HyperelasticResponse,
)
from ._plane_stress import (
    BlockDiagonalPlaneStressReductionPlan,
    PlaneStressFailure,
)


class _MPMPointHyperelasticLaw(HyperelasticLaw):
    base: AbstractImplicitMPMConstitutivePlan
    history: Array
    reference_density: Array
    parameters: Any
    time: Array
    step_size: Array

    def __init__(
        self,
        base: AbstractImplicitMPMConstitutivePlan,
        history: Array,
        reference_density: Array,
        parameters: Any,
        time: Array,
        step_size: Array,
        /,
    ):
        self.base = base
        self.history = history
        self.reference_density = reference_density
        self.parameters = parameters
        self.time = time
        self.step_size = step_size

    def evaluate(self, deformation_gradient: ArrayLike, /) -> HyperelasticResponse:
        deformation = jnp.asarray(deformation_gradient)
        linearized = self.base.evaluate_linearized(
            deformation[None, ...],
            self.history[None, ...],
            self.reference_density[None],
            self.parameters,
            self.time,
            self.step_size,
        )
        response = linearized.response
        kinematics = finite_strain_kinematics(deformation)
        first_piola = response.first_piola[0]
        material_admissible = (
            response.successful[0]
            & response.admissible[0]
            & linearized.tangent_successful[0]
        )
        admissible = kinematics.admissible & material_admissible
        return HyperelasticResponse(
            kinematics,
            response.reference_energy_density[0],
            first_piola,
            first_piola_to_cauchy(kinematics, first_piola),
            linearized.algorithmic_tangent[0],
            kinematics.admissible,
            material_admissible,
            admissible,
        )


class PlaneStressMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan):
    """Block-diagonal plane-stress closure for an implicit-capable 3-D MPM law."""

    base: AbstractImplicitMPMConstitutivePlan
    reduction: BlockDiagonalPlaneStressReductionPlan
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    base_state_width: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractImplicitMPMConstitutivePlan,
        /,
        *,
        reduction: BlockDiagonalPlaneStressReductionPlan | None = None,
    ):
        if not isinstance(base, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("base must be AbstractImplicitMPMConstitutivePlan.")
        if base.dimension != 3 or base.kinematics != "three_dimensional":
            raise ValueError("Plane-stress closure requires one 3-D base material.")
        reduction_ = (
            BlockDiagonalPlaneStressReductionPlan() if reduction is None else reduction
        )
        if not isinstance(reduction_, BlockDiagonalPlaneStressReductionPlan):
            raise TypeError(
                "reduction must be BlockDiagonalPlaneStressReductionPlan or None."
            )
        width = int(np.prod(base.state_shape))
        self.base = base
        self.reduction = reduction_
        self.dimension = 2
        self.kinematics = "plane_stress"
        self.base_state_width = width
        self.state_shape = (width + 1,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=base.capabilities.has_free_energy,
            has_algorithmic_tangent=True,
            has_dissipation=base.capabilities.has_dissipation,
            has_tension_compression_split=base.capabilities.has_tension_compression_split,
            supports_implicit=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-diagonal-plane-stress-mpm",
                "base": base.plan_id,
                "reduction": reduction_.plan_id,
                "unknown": "log-out-of-plane-stretch",
                "residual": "first-piola-33",
            }
        )

    def initialize_state(self, batch_shape, dtype, /):
        base = self.base.initialize_state(tuple(batch_shape), dtype).reshape(
            tuple(batch_shape) + (self.base_state_width,)
        )
        eta = jnp.zeros(tuple(batch_shape) + (1,), dtype=dtype)
        return jnp.concatenate((base, eta), axis=-1)

    @staticmethod
    def _embed(deformation, eta):
        embedded = jnp.zeros((3, 3), dtype=deformation.dtype)
        embedded = embedded.at[:2, :2].set(deformation)
        return embedded.at[2, 2].set(jnp.exp(eta))

    def _point_law(
        self,
        history,
        density,
        parameters,
        time,
        step_size,
    ) -> _MPMPointHyperelasticLaw:
        return _MPMPointHyperelasticLaw(
            self.base,
            history,
            density,
            parameters,
            jnp.asarray(time),
            jnp.asarray(step_size),
        )

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (2, 2):
            raise ValueError("Plane-stress deformation gradients must end in 2x2.")
        batch_shape = deformation.shape[:-2]
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        if state.shape != batch_shape + self.state_shape:
            raise ValueError("Plane-stress history has the wrong shape.")
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        if density.shape != batch_shape:
            raise ValueError("Plane-stress reference density shape changed.")
        flat_f = deformation.reshape((-1, 2, 2))
        flat_state = state.reshape((-1, self.state_shape[0]))
        flat_density = density.reshape((-1,))

        def point(value, history, density_value):
            base_history = history[: self.base_state_width].reshape(self.base.state_shape)
            law = self._point_law(
                base_history,
                density_value,
                parameters,
                time,
                step_size,
            )
            reduced = self.reduction.evaluate(value, law)
            eta = reduced.kinematics.log_thickness_stretch
            embedded = self._embed(value, eta)
            response = self.base.evaluate(
                embedded[None, ...],
                base_history[None, ...],
                density_value[None],
                parameters,
                time,
                step_size,
            )
            trial = jnp.concatenate(
                (response.trial_state.reshape((-1,)), eta[None]), axis=0
            )
            valid = reduced.successful & response.successful[0] & response.admissible[0]
            failure = jnp.where(
                valid,
                int(PlaneStressFailure.OK),
                jnp.where(
                    reduced.successful,
                    int(PlaneStressFailure.BASE_LAW_REJECTED),
                    reduced.failure,
                ),
            ).astype(jnp.int32)
            return (
                reduced.first_piola,
                trial,
                reduced.reference_energy_density,
                response.maximum_wave_speed[0],
                response.dissipation_increment[0],
                response.branch_code[0],
                response.suggested_step[0],
                valid,
                eta,
                reduced.residual,
                reduced.log_stretch_sensitivity,
                reduced.condensed_tangent,
                failure,
            )

        outputs = jax.vmap(point)(flat_f, flat_state, flat_density)
        (
            stress,
            trial,
            energy,
            speed,
            dissipation,
            branch,
            suggested,
            valid,
            eta,
            residual,
            sensitivity,
            tangent,
            failure,
        ) = outputs
        return MPMConstitutiveResponse(
            stress.reshape(batch_shape + (2, 2)),
            trial.reshape(batch_shape + self.state_shape),
            energy.reshape(batch_shape),
            speed.reshape(batch_shape),
            dissipation_increment=dissipation.reshape(batch_shape),
            branch_code=branch.reshape(batch_shape),
            suggested_step=suggested.reshape(batch_shape),
            successful=valid.reshape(batch_shape),
            admissible=valid.reshape(batch_shape),
            diagnostics={
                "out_of_plane_log_stretch": eta.reshape(batch_shape),
                "out_of_plane_stretch": jnp.exp(eta).reshape(batch_shape),
                "plane_stress_residual": residual.reshape(batch_shape),
                "plane_stress_log_stretch_sensitivity": sensitivity.reshape(
                    batch_shape + (2, 2)
                ),
                "plane_stress_condensed_tangent": tangent.reshape(
                    batch_shape + (2, 2, 2, 2)
                ),
                "plane_stress_failure": failure.reshape(batch_shape),
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters,
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
        tangent = response.diagnostics["plane_stress_condensed_tangent"]
        successful = response.successful & jnp.all(
            jnp.isfinite(tangent), axis=(-4, -3, -2, -1)
        )
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)


__all__ = ["PlaneStressMPMConstitutivePlan"]
