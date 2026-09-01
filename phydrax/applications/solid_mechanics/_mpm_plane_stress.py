#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    LocalConstitutiveRootPlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)


class IsotropicPlaneStressMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan):
    """Scalar out-of-plane closure for block-diagonal isotropic 3-D materials."""

    base: AbstractImplicitMPMConstitutivePlan
    root: LocalConstitutiveRootPlan
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
        root: LocalConstitutiveRootPlan | None = None,
    ):
        if not isinstance(base, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("base must be AbstractImplicitMPMConstitutivePlan.")
        if base.dimension != 3 or base.kinematics != "three_dimensional":
            raise ValueError("Plane-stress closure requires one 3-D base material.")
        root_ = (
            LocalConstitutiveRootPlan(plan_id="plane-stress-thickness")
            if root is None
            else root
        )
        if not isinstance(root_, LocalConstitutiveRootPlan):
            raise TypeError("root must be LocalConstitutiveRootPlan or None.")
        width = int(np.prod(base.state_shape))
        self.base = base
        self.root = root_
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
                "kind": "isotropic-plane-stress-mpm",
                "base": base.plan_id,
                "root": root_.plan_id,
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
            eta0 = history[-1]

            def residual(eta):
                embedded = self._embed(value, eta)
                response = self.base.evaluate(
                    embedded[None, ...],
                    base_history[None, ...],
                    density_value[None],
                    parameters,
                    time,
                    step_size,
                )
                return response.first_piola[0, 2, 2]

            eta, root_diagnostics = self.root.solve_with_diagnostics(residual, eta0)
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
            valid = (
                root_diagnostics.converged
                & response.successful[0]
                & response.admissible[0]
                & jnp.isfinite(eta)
            )
            return (
                response.first_piola[0, :2, :2],
                trial,
                response.reference_energy_density[0],
                response.maximum_wave_speed[0],
                response.dissipation_increment[0],
                response.branch_code[0],
                response.suggested_step[0],
                valid,
                eta,
                root_diagnostics.residual,
                root_diagnostics.derivative,
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
            derivative,
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
                "plane_stress_derivative": derivative.reshape(batch_shape),
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
        deformation = jnp.asarray(deformation_gradient)
        batch_shape = deformation.shape[:-2]
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        eta = response.trial_state[..., -1]
        flat_f = deformation.reshape((-1, 2, 2))
        flat_state = state.reshape((-1, self.state_shape[0]))
        flat_density = density.reshape((-1,))
        flat_eta = eta.reshape((-1,))

        def point(value, history, density_value, eta_value):
            embedded = self._embed(value, eta_value)
            base_history = history[: self.base_state_width].reshape(self.base.state_shape)
            linearized = self.base.evaluate_linearized(
                embedded[None, ...],
                base_history[None, ...],
                density_value[None],
                parameters,
                time,
                step_size,
            )
            tangent = linearized.algorithmic_tangent[0]
            denominator = tangent[2, 2, 2, 2]
            safe = jnp.where(jnp.abs(denominator) > 1.0e-12, denominator, 1.0)
            in_plane = tangent[:2, :2, :2, :2] - (
                tangent[:2, :2, 2, 2][..., None, None]
                * tangent[2, 2, :2, :2][None, None, ...]
                / safe
            )
            successful = (
                linearized.tangent_successful[0]
                & jnp.isfinite(denominator)
                & (jnp.abs(denominator) > 1.0e-12)
                & jnp.all(jnp.isfinite(in_plane))
            )
            return in_plane, successful

        tangent, successful = jax.vmap(point)(flat_f, flat_state, flat_density, flat_eta)
        tangent = tangent.reshape(batch_shape + (2, 2, 2, 2))
        return MPMLinearizedConstitutiveResponse(
            response,
            tangent,
            successful.reshape(batch_shape),
        )


__all__ = ["IsotropicPlaneStressMPMConstitutivePlan"]
