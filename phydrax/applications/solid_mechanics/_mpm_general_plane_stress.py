#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
    VectorLocalConstitutiveRootPlan,
)
from ...linalg import SmallLinearSolvePlan, solve_small_linear


class MPMMaterialOrientation(StrictModule, NonTrainableState):
    rotation: Array
    orientation_id: str = eqx.field(static=True)

    def __init__(self, rotation: ArrayLike, /, *, tolerance: float = 1.0e-10):
        value = np.asarray(rotation, dtype=float)
        if value.shape != (3, 3) or np.any(~np.isfinite(value)):
            raise ValueError("Material orientation must be one finite 3x3 rotation.")
        defect = np.linalg.norm(value.T @ value - np.eye(3))
        determinant = np.linalg.det(value)
        if defect > float(tolerance) or abs(determinant - 1.0) > float(tolerance):
            raise ValueError("Material orientation must be proper orthonormal.")
        self.rotation = jnp.asarray(value)
        self.orientation_id = canonical_fingerprint(
            {
                "kind": "mpm-material-orientation",
                "rotation": array_tree_fingerprint(value),
            }
        )


class OrientedMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan):
    base: AbstractImplicitMPMConstitutivePlan
    orientation: MPMMaterialOrientation
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractImplicitMPMConstitutivePlan,
        orientation: MPMMaterialOrientation,
        /,
    ):
        if not isinstance(base, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("base must be AbstractImplicitMPMConstitutivePlan.")
        if base.dimension != 3:
            raise ValueError("Oriented material wrapper currently requires 3-D base.")
        if not isinstance(orientation, MPMMaterialOrientation):
            raise TypeError("orientation must be MPMMaterialOrientation.")
        self.base = base
        self.orientation = orientation
        self.dimension = 3
        self.kinematics = "three_dimensional"
        self.state_shape = base.state_shape
        self.capabilities = base.capabilities
        self.plan_id = canonical_fingerprint(
            {
                "kind": "oriented-mpm-material",
                "base": base.plan_id,
                "orientation": orientation.orientation_id,
            }
        )

    def initialize_state(self, batch_shape, dtype, /):
        return self.base.initialize_state(batch_shape, dtype)

    def _to_material(self, deformation):
        rotation = self.orientation.rotation.astype(deformation.dtype)
        return ein.contract("ij,...jk,lk->...il", rotation, deformation, rotation)

    def _to_global_stress(self, stress):
        rotation = self.orientation.rotation.astype(stress.dtype)
        return ein.contract("ij,...jk,lk->...il", rotation.T, stress, rotation.T)

    def evaluate(
        self,
        deformation_gradient,
        committed_state,
        reference_density,
        parameters,
        time,
        step_size,
        /,
    ):
        material_deformation = self._to_material(jnp.asarray(deformation_gradient))
        response = self.base.evaluate(
            material_deformation,
            committed_state,
            reference_density,
            parameters,
            time,
            step_size,
        )
        return MPMConstitutiveResponse(
            self._to_global_stress(response.first_piola),
            response.trial_state,
            response.reference_energy_density,
            response.maximum_wave_speed,
            dissipation_increment=response.dissipation_increment,
            branch_code=response.branch_code,
            suggested_step=response.suggested_step,
            successful=response.successful,
            admissible=response.admissible,
            diagnostics=response.diagnostics,
        )

    def evaluate_linearized(
        self,
        deformation_gradient,
        committed_state,
        reference_density,
        parameters,
        time,
        step_size,
        /,
    ):
        response = self.evaluate(
            deformation_gradient,
            committed_state,
            reference_density,
            parameters,
            time,
            step_size,
        )
        deformation = jnp.asarray(deformation_gradient)
        base = self.base.evaluate_linearized(
            self._to_material(deformation),
            committed_state,
            reference_density,
            parameters,
            time,
            step_size,
        )
        rotation = self.orientation.rotation.astype(deformation.dtype)
        tangent = ein.contract(
            "ia,Jb,...abCD,kC,LD->...iJkL",
            rotation,
            rotation,
            base.algorithmic_tangent,
            rotation,
            rotation,
        )
        return MPMLinearizedConstitutiveResponse(
            response, tangent, base.tangent_successful
        )


class GeneralPlaneStressMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan):
    """Three-component traction-free director closure P[:,3] = 0."""

    base: AbstractImplicitMPMConstitutivePlan
    root: VectorLocalConstitutiveRootPlan
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
        root: VectorLocalConstitutiveRootPlan | None = None,
    ):
        if not isinstance(base, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("base must be AbstractImplicitMPMConstitutivePlan.")
        if base.dimension != 3 or base.kinematics != "three_dimensional":
            raise ValueError("General plane stress requires one 3-D base material.")
        root_ = (
            VectorLocalConstitutiveRootPlan(
                3,
                plan_id="general-plane-stress-director",
            )
            if root is None
            else root
        )
        if not isinstance(root_, VectorLocalConstitutiveRootPlan) or root_.dimension != 3:
            raise TypeError("General plane stress needs a three-component local root.")
        width = int(np.prod(base.state_shape))
        self.base = base
        self.root = root_
        self.dimension = 2
        self.kinematics = "plane_stress"
        self.base_state_width = width
        self.state_shape = (width + 3,)
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
                "kind": "general-plane-stress-mpm",
                "base": base.plan_id,
                "root": root_.plan_id,
                "unknown": "transverse-director-column",
                "residual": "first-piola-column-3",
            }
        )

    def initialize_state(self, batch_shape, dtype, /):
        base = self.base.initialize_state(tuple(batch_shape), dtype).reshape(
            tuple(batch_shape) + (self.base_state_width,)
        )
        director = jnp.broadcast_to(
            jnp.asarray((0.0, 0.0, 1.0), dtype=dtype), tuple(batch_shape) + (3,)
        )
        return jnp.concatenate((base, director), axis=-1)

    @staticmethod
    def _embed(deformation, director):
        embedded = jnp.zeros((3, 3), dtype=deformation.dtype)
        embedded = embedded.at[:2, :2].set(deformation)
        return embedded.at[:, 2].set(director)

    def _point(self, deformation, state, density, parameters, time, step_size):
        base_state = state[: self.base_state_width].reshape(self.base.state_shape)
        director0 = state[-3:]

        def residual(director):
            response = self.base.evaluate(
                self._embed(deformation, director)[None],
                base_state[None],
                density[None],
                parameters,
                time,
                step_size,
            )
            return response.first_piola[0, :, 2]

        director, root_evidence = self.root.solve_with_diagnostics(residual, director0)
        embedded = self._embed(deformation, director)
        response = self.base.evaluate(
            embedded[None],
            base_state[None],
            density[None],
            parameters,
            time,
            step_size,
        )
        determinant = jnp.linalg.det(embedded)
        valid = (
            root_evidence.converged
            & response.successful[0]
            & response.admissible[0]
            & jnp.isfinite(determinant)
            & (determinant > 0.0)
        )
        trial = jnp.concatenate((response.trial_state.reshape((-1,)), director), axis=0)
        return (
            response.first_piola[0, :2, :2],
            trial,
            response.reference_energy_density[0],
            response.maximum_wave_speed[0],
            response.dissipation_increment[0],
            response.branch_code[0],
            response.suggested_step[0],
            valid,
            director,
            root_evidence.residual,
            root_evidence.condition_estimate,
            embedded,
        )

    def evaluate(
        self,
        deformation_gradient,
        committed_state,
        reference_density,
        parameters,
        time,
        step_size,
        /,
    ):
        deformation = jnp.asarray(deformation_gradient)
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.asarray(reference_density, dtype=deformation.dtype)
        batch_shape = deformation.shape[:-2]
        if deformation.shape[-2:] != (2, 2):
            raise ValueError("General plane-stress deformation must end in 2x2.")
        if state.shape != batch_shape + self.state_shape or density.shape != batch_shape:
            raise ValueError("General plane-stress state/density shape changed.")
        outputs = jax.vmap(
            lambda value, history, rho: self._point(
                value, history, rho, parameters, time, step_size
            )
        )(
            deformation.reshape((-1, 2, 2)),
            state.reshape((-1, self.state_shape[0])),
            density.reshape((-1,)),
        )
        (
            stress,
            trial,
            energy,
            speed,
            dissipation,
            branch,
            suggested,
            valid,
            director,
            residual,
            condition,
            _,
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
                "transverse_director": director.reshape(batch_shape + (3,)),
                "plane_stress_residual": residual.reshape(batch_shape + (3,)),
                "plane_stress_condition": condition.reshape(batch_shape),
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient,
        committed_state,
        reference_density,
        parameters,
        time,
        step_size,
        /,
    ):
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
        director = response.trial_state[..., -3:]

        def point(value, history, rho, current_director):
            embedded = self._embed(value, current_director)
            base_state = history[: self.base_state_width].reshape(self.base.state_shape)
            linearized = self.base.evaluate_linearized(
                embedded[None],
                base_state[None],
                rho[None],
                parameters,
                time,
                step_size,
            )
            tangent = linearized.algorithmic_tangent[0]
            prescribed = tangent[:2, :2, :2, :2].reshape((4, 4))
            output_director = tangent[:2, :2, :, 2].reshape((4, 3))
            residual_prescribed = tangent[:, 2, :2, :2].reshape((3, 4))
            residual_director = tangent[:, 2, :, 2]
            inverse = solve_small_linear(
                SmallLinearSolvePlan(3), residual_director, jnp.eye(3, dtype=value.dtype)
            )
            condensed = prescribed - output_director @ inverse.value @ residual_prescribed
            successful = (
                linearized.tangent_successful[0]
                & inverse.successful
                & jnp.all(jnp.isfinite(condensed))
            )
            return condensed.reshape((2, 2, 2, 2)), successful

        tangent, successful = jax.vmap(point)(
            deformation.reshape((-1, 2, 2)),
            state.reshape((-1, self.state_shape[0])),
            density.reshape((-1,)),
            director.reshape((-1, 3)),
        )
        return MPMLinearizedConstitutiveResponse(
            response,
            tangent.reshape(batch_shape + (2, 2, 2, 2)),
            successful.reshape(batch_shape),
        )


__all__ = [
    "GeneralPlaneStressMPMConstitutivePlan",
    "MPMMaterialOrientation",
    "OrientedMPMConstitutivePlan",
]
