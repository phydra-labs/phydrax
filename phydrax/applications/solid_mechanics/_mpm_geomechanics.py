#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    LocalConstitutiveRootPlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)
from ...linalg import SmallLinearSolvePlan, solve_small_linear


def _validated_frictional_values(name, values):
    arrays = tuple(jnp.asarray(value) for value in values)
    if any(value.shape != () for value in arrays) or any(
        not bool(jnp.isfinite(value)) for value in arrays
    ):
        raise ValueError(f"{name} parameters must be finite scalars.")
    if (
        arrays[0] <= 0.0
        or arrays[1] <= 0.0
        or arrays[2] < 0.0
        or not 0.0 <= arrays[3] < 0.5 * pi
        or not -0.5 * pi < arrays[4] < 0.5 * pi
        or arrays[5] < 0.0
    ):
        raise ValueError(f"{name} parameters are inadmissible.")
    return arrays


class DruckerPragerParameters(StrictModule, NonTrainableState):
    shear_modulus: Array
    bulk_modulus: Array
    cohesion: Array
    friction_angle: Array
    dilation_angle: Array
    hardening_modulus: Array

    def __init__(
        self,
        shear_modulus,
        bulk_modulus,
        cohesion,
        friction_angle,
        dilation_angle,
        hardening_modulus=0.0,
        /,
    ):
        values = _validated_frictional_values(
            "Drucker-Prager",
            (
                shear_modulus,
                bulk_modulus,
                cohesion,
                friction_angle,
                dilation_angle,
                hardening_modulus,
            ),
        )
        (
            self.shear_modulus,
            self.bulk_modulus,
            self.cohesion,
            self.friction_angle,
            self.dilation_angle,
            self.hardening_modulus,
        ) = values


class MohrCoulombParameters(StrictModule, NonTrainableState):
    shear_modulus: Array
    bulk_modulus: Array
    cohesion: Array
    friction_angle: Array
    dilation_angle: Array
    hardening_modulus: Array

    def __init__(
        self,
        shear_modulus,
        bulk_modulus,
        cohesion,
        friction_angle,
        dilation_angle,
        hardening_modulus=0.0,
        /,
    ):
        values = _validated_frictional_values(
            "Mohr-Coulomb",
            (
                shear_modulus,
                bulk_modulus,
                cohesion,
                friction_angle,
                dilation_angle,
                hardening_modulus,
            ),
        )
        (
            self.shear_modulus,
            self.bulk_modulus,
            self.cohesion,
            self.friction_angle,
            self.dilation_angle,
            self.hardening_modulus,
        ) = values


class ModifiedCamClayParameters(StrictModule, NonTrainableState):
    shear_modulus: Array
    bulk_modulus: Array
    critical_state_slope: Array
    hardening_modulus: Array

    def __init__(
        self, shear_modulus, bulk_modulus, critical_state_slope, hardening_modulus, /
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                shear_modulus,
                bulk_modulus,
                critical_state_slope,
                hardening_modulus,
            )
        )
        if any(value.shape != () for value in values) or any(
            not bool(jnp.isfinite(value)) or value <= 0.0 for value in values
        ):
            raise ValueError(
                "Modified Cam-Clay parameters must be positive finite scalars."
            )
        (
            self.shear_modulus,
            self.bulk_modulus,
            self.critical_state_slope,
            self.hardening_modulus,
        ) = values


class NonlocalSofteningPlan(StrictModule, NonTrainableState):
    characteristic_length: float = eqx.field(static=True)
    viscosity: float = eqx.field(static=True)
    minimum_modulus: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        characteristic_length: float,
        /,
        *,
        viscosity: float = 0.0,
        minimum_modulus: float = 0.0,
    ):
        length = float(characteristic_length)
        viscosity_ = float(viscosity)
        minimum = float(minimum_modulus)
        if (
            not np.isfinite(length)
            or length <= 0.0
            or not np.isfinite(viscosity_)
            or viscosity_ < 0.0
            or not np.isfinite(minimum)
            or minimum < 0.0
        ):
            raise ValueError("Nonlocal softening plan is invalid.")
        self.characteristic_length = length
        self.viscosity = viscosity_
        self.minimum_modulus = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonlocal-softening-plan",
                "characteristic_length": length,
                "viscosity": viscosity_,
                "minimum_modulus": minimum,
            }
        )

    def regularize(self, local_history, nonlocal_history, step_size, /):
        local = jnp.asarray(local_history)
        nonlocal_ = jnp.asarray(nonlocal_history, dtype=local.dtype)
        dt = jnp.asarray(step_size, dtype=local.dtype)
        if self.viscosity == 0.0:
            return nonlocal_
        weight = dt / (self.viscosity + dt)
        return local + weight * (nonlocal_ - local)


class GeomechanicalFailureEvidence(StrictModule):
    branch_code: Array
    yield_residual: Array
    plastic_multiplier: Array
    apex_or_corner: Array
    dissipation: Array
    suggested_step: Array
    successful: Array


def _inverse(value):
    identity = jnp.broadcast_to(jnp.eye(3, dtype=value.dtype), value.shape)
    return solve_small_linear(SmallLinearSolvePlan(3), value, identity)


def _log_strain(deformation):
    right = deformation.T @ deformation
    eigenvalues, eigenvectors = jnp.linalg.eigh(right)
    valid = jnp.all(eigenvalues > 0.0) & jnp.all(jnp.isfinite(eigenvalues))
    logarithmic = 0.5 * jnp.log(jnp.where(eigenvalues > 0.0, eigenvalues, 1.0))
    return (eigenvectors * logarithmic[None, :]) @ eigenvectors.T, valid


def _elastic_stress(elastic_strain, shear, bulk):
    trace = jnp.trace(elastic_strain)
    deviatoric = elastic_strain - trace / 3.0 * jnp.eye(3, dtype=elastic_strain.dtype)
    return 2.0 * shear * deviatoric + bulk * trace * jnp.eye(
        3, dtype=elastic_strain.dtype
    )


def _invariants(stress):
    mean = jnp.trace(stress) / 3.0
    deviatoric = stress - mean * jnp.eye(3, dtype=stress.dtype)
    return -mean, jnp.sqrt(1.5 * jnp.sum(deviatoric * deviatoric)), deviatoric


def _first_piola(kirchhoff, deformation):
    inverse = _inverse(deformation)
    return kirchhoff @ inverse.value.T, inverse


class _AbstractPressureDependentPlan(AbstractImplicitMPMConstitutivePlan):
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def initialize_state(self, batch_shape, dtype, /):
        raise NotImplementedError

    @abc.abstractmethod
    def _point(self, deformation, history, density, parameters, step_size):
        raise NotImplementedError

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

        def stress(value, history, rho):
            return self._point(value, history, rho, parameters, step_size)[0]

        tangent = jax.vmap(jax.jacfwd(stress, argnums=0))(
            deformation.reshape((-1, 3, 3)),
            state.reshape((-1, self.state_shape[0])),
            density.reshape((-1,)),
        ).reshape(batch_shape + (3, 3, 3, 3))
        successful = jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1))
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)

    def _response(self, deformation, state, density, parameters, step_size):
        batch_shape = deformation.shape[:-2]
        outputs = jax.vmap(
            lambda value, history, rho: self._point(
                value, history, rho, parameters, step_size
            )
        )(
            deformation.reshape((-1, 3, 3)),
            state.reshape((-1, self.state_shape[0])),
            density.reshape((-1,)),
        )
        (
            stress,
            next_state,
            energy,
            speed,
            dissipation,
            branch,
            valid,
            diagnostics,
        ) = outputs
        suggested = jnp.where(valid, jnp.inf, 0.5 * jnp.asarray(step_size))
        return MPMConstitutiveResponse(
            stress.reshape(batch_shape + (3, 3)),
            next_state.reshape(batch_shape + self.state_shape),
            energy.reshape(batch_shape),
            speed.reshape(batch_shape),
            dissipation_increment=dissipation.reshape(batch_shape),
            branch_code=branch.reshape(batch_shape),
            suggested_step=suggested.reshape(batch_shape),
            successful=valid.reshape(batch_shape),
            admissible=valid.reshape(batch_shape),
            diagnostics={
                name: value.reshape(batch_shape) for name, value in diagnostics.items()
            },
        )


class DruckerPragerMPMConstitutivePlan(_AbstractPressureDependentPlan):
    def __init__(self):
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
        self.plan_id = canonical_fingerprint(
            {"kind": "drucker-prager-mpm", "flow": "non-associated"}
        )

    def initialize_state(self, batch_shape, dtype, /):
        return jnp.zeros(tuple(batch_shape) + self.state_shape, dtype=dtype)

    def _point(self, deformation, history, density, parameters, step_size):
        del step_size
        plastic = history[:9].reshape((3, 3))
        hardening = history[9]
        strain, strain_valid = _log_strain(deformation)
        trial = _elastic_stress(
            strain - plastic, parameters.shear_modulus, parameters.bulk_modulus
        )
        pressure, q, deviatoric = _invariants(trial)
        phi = parameters.friction_angle
        psi = parameters.dilation_angle
        alpha = 2.0 * jnp.sin(phi) / (jnp.sqrt(3.0) * (3.0 - jnp.sin(phi)))
        beta = 2.0 * jnp.sin(psi) / (jnp.sqrt(3.0) * (3.0 - jnp.sin(psi)))
        strength = (
            6.0
            * parameters.cohesion
            * jnp.cos(phi)
            / (jnp.sqrt(3.0) * (3.0 - jnp.sin(phi)))
        )
        yield_value = (
            q + alpha * pressure - (strength + parameters.hardening_modulus * hardening)
        )
        denominator = (
            3.0 * parameters.shear_modulus
            + parameters.bulk_modulus * alpha * beta
            + parameters.hardening_modulus
        )
        multiplier = jnp.maximum(yield_value, 0.0) / denominator
        apex = q <= 1.0e-12
        direction = 1.5 * deviatoric / jnp.where(q > 1.0e-12, q, 1.0)
        flow = direction - beta / 3.0 * jnp.eye(3, dtype=strain.dtype)
        next_plastic = plastic + multiplier * flow
        next_hardening = hardening + multiplier
        stress = _elastic_stress(
            strain - next_plastic, parameters.shear_modulus, parameters.bulk_modulus
        )
        first_piola, inverse = _first_piola(stress, deformation)
        p_new, q_new, _ = _invariants(stress)
        residual = (
            q_new
            + alpha * p_new
            - (strength + parameters.hardening_modulus * next_hardening)
        )
        energy = 0.5 * jnp.sum(stress * (strain - next_plastic))
        dissipation = jnp.maximum(strength * multiplier, 0.0)
        speed = jnp.sqrt(
            (parameters.bulk_modulus + 4.0 * parameters.shear_modulus / 3.0)
            / jnp.maximum(density, 1.0e-30)
        )
        valid = (
            strain_valid
            & inverse.successful
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.isfinite(residual)
            & (density > 0.0)
        )
        state = jnp.concatenate((next_plastic.reshape((9,)), next_hardening[None]))
        diagnostics = {
            "yield_residual": residual,
            "plastic_multiplier": multiplier,
            "apex_or_corner": apex,
        }
        return (
            jnp.where(valid, first_piola, 0.0),
            jnp.where(valid, state, history),
            jnp.where(valid, energy, 0.0),
            jnp.where(valid, speed, 0.0),
            jnp.where(valid, dissipation, 0.0),
            jnp.where(yield_value > 1.0e-10, jnp.where(apex, 2, 1), 0).astype(jnp.int32),
            valid,
            diagnostics,
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
        del time
        if not isinstance(parameters, DruckerPragerParameters):
            raise TypeError("parameters must be DruckerPragerParameters.")
        return self._response(
            jnp.asarray(deformation_gradient),
            jnp.asarray(committed_state),
            jnp.asarray(reference_density),
            parameters,
            step_size,
        )


class MohrCoulombMPMConstitutivePlan(_AbstractPressureDependentPlan):
    def __init__(self):
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
        self.plan_id = canonical_fingerprint(
            {"kind": "mohr-coulomb-mpm", "return": "principal-semismooth"}
        )

    def initialize_state(self, batch_shape, dtype, /):
        return jnp.zeros(tuple(batch_shape) + self.state_shape, dtype=dtype)

    def _point(self, deformation, history, density, parameters, step_size):
        del step_size
        plastic = history[:9].reshape((3, 3))
        hardening = history[9]
        strain, strain_valid = _log_strain(deformation)
        trial = _elastic_stress(
            strain - plastic, parameters.shear_modulus, parameters.bulk_modulus
        )
        principal, vectors = jnp.linalg.eigh(trial)
        phi, psi = parameters.friction_angle, parameters.dilation_angle
        cohesion = parameters.cohesion + parameters.hardening_modulus * hardening
        yield_value = (
            principal[2]
            - principal[0]
            + (principal[2] + principal[0]) * jnp.sin(phi)
            - 2.0 * cohesion * jnp.cos(phi)
        )
        yield_gradient = jnp.asarray((-1.0 + jnp.sin(phi), 0.0, 1.0 + jnp.sin(phi)))
        flow_gradient = jnp.asarray((-1.0 + jnp.sin(psi), 0.0, 1.0 + jnp.sin(psi)))
        elasticity = 2.0 * parameters.shear_modulus * (
            jnp.eye(3) - jnp.ones((3, 3)) / 3.0
        ) + parameters.bulk_modulus * jnp.ones((3, 3))
        denominator = (
            yield_gradient @ elasticity @ flow_gradient + parameters.hardening_modulus
        )
        multiplier = jnp.maximum(yield_value, 0.0) / jnp.maximum(denominator, 1.0e-12)
        flow = (vectors * flow_gradient[None, :]) @ vectors.T
        next_plastic = plastic + multiplier * flow
        next_hardening = hardening + multiplier
        stress = _elastic_stress(
            strain - next_plastic, parameters.shear_modulus, parameters.bulk_modulus
        )
        corrected, _ = jnp.linalg.eigh(stress)
        edge = (corrected[2] - corrected[1] <= 1.0e-10) | (
            corrected[1] - corrected[0] <= 1.0e-10
        )
        apex = corrected[2] - corrected[0] <= 1.0e-10
        first_piola, inverse = _first_piola(stress, deformation)
        residual = jnp.maximum(yield_value - multiplier * denominator, 0.0)
        energy = 0.5 * jnp.sum(stress * (strain - next_plastic))
        dissipation = jnp.maximum(cohesion * multiplier, 0.0)
        speed = jnp.sqrt(
            (parameters.bulk_modulus + 4.0 * parameters.shear_modulus / 3.0)
            / jnp.maximum(density, 1.0e-30)
        )
        valid = (
            strain_valid
            & inverse.successful
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.isfinite(residual)
            & (density > 0.0)
        )
        state = jnp.concatenate((next_plastic.reshape((9,)), next_hardening[None]))
        diagnostics = {
            "yield_residual": residual,
            "plastic_multiplier": multiplier,
            "apex_or_corner": apex | edge,
        }
        branch = jnp.where(
            yield_value > 1.0e-10,
            jnp.where(apex, 3, jnp.where(edge, 2, 1)),
            0,
        ).astype(jnp.int32)
        return (
            jnp.where(valid, first_piola, 0.0),
            jnp.where(valid, state, history),
            jnp.where(valid, energy, 0.0),
            jnp.where(valid, speed, 0.0),
            jnp.where(valid, dissipation, 0.0),
            branch,
            valid,
            diagnostics,
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
        del time
        if not isinstance(parameters, MohrCoulombParameters):
            raise TypeError("parameters must be MohrCoulombParameters.")
        return self._response(
            jnp.asarray(deformation_gradient),
            jnp.asarray(committed_state),
            jnp.asarray(reference_density),
            parameters,
            step_size,
        )


class ModifiedCamClayMPMConstitutivePlan(_AbstractPressureDependentPlan):
    initial_preconsolidation_pressure: float = eqx.field(static=True)
    initial_void_ratio: float = eqx.field(static=True)
    root: LocalConstitutiveRootPlan

    def __init__(
        self, *, initial_preconsolidation_pressure: float, initial_void_ratio: float
    ):
        pressure = float(initial_preconsolidation_pressure)
        void = float(initial_void_ratio)
        if pressure <= 0.0 or void <= 0.0:
            raise ValueError("Modified Cam-Clay initial state is inadmissible.")
        self.dimension = 3
        self.kinematics = "three_dimensional"
        self.state_shape = (11,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=True,
            has_algorithmic_tangent=True,
            has_dissipation=True,
            supports_implicit=True,
        )
        self.initial_preconsolidation_pressure = pressure
        self.initial_void_ratio = void
        self.root = LocalConstitutiveRootPlan(plan_id="modified-cam-clay-consistency")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "modified-cam-clay-mpm",
                "initial_preconsolidation_pressure": pressure,
                "initial_void_ratio": void,
                "root": self.root.plan_id,
            }
        )

    def initialize_state(self, batch_shape, dtype, /):
        shape = tuple(batch_shape)
        return jnp.concatenate(
            (
                jnp.zeros(shape + (9,), dtype=dtype),
                jnp.full(
                    shape + (1,), self.initial_preconsolidation_pressure, dtype=dtype
                ),
                jnp.full(shape + (1,), self.initial_void_ratio, dtype=dtype),
            ),
            axis=-1,
        )

    def _point(self, deformation, history, density, parameters, step_size):
        del step_size
        plastic = history[:9].reshape((3, 3))
        pc_old, void_old = history[9], history[10]
        strain, strain_valid = _log_strain(deformation)
        trial = _elastic_stress(
            strain - plastic, parameters.shear_modulus, parameters.bulk_modulus
        )
        p_trial, q_trial, deviatoric = _invariants(trial)
        slope = parameters.critical_state_slope
        yield_trial = q_trial**2 + slope**2 * p_trial * (p_trial - pc_old)
        flow = 3.0 * deviatoric - slope**2 * (2.0 * p_trial - pc_old) / 3.0 * jnp.eye(3)

        def consistency(multiplier):
            next_plastic = plastic + multiplier * flow
            stress = _elastic_stress(
                strain - next_plastic, parameters.shear_modulus, parameters.bulk_modulus
            )
            pressure, q, _ = _invariants(stress)
            pc = pc_old * jnp.exp(parameters.hardening_modulus * multiplier)
            return q**2 + slope**2 * pressure * (pressure - pc)

        multiplier = jax.lax.cond(
            yield_trial > 1.0e-10,
            lambda _: self.root.solve(
                consistency, jnp.asarray(0.0, dtype=deformation.dtype)
            ),
            lambda _: jnp.asarray(0.0, dtype=deformation.dtype),
            operand=None,
        )
        next_plastic = plastic + multiplier * flow
        stress = _elastic_stress(
            strain - next_plastic, parameters.shear_modulus, parameters.bulk_modulus
        )
        pc = pc_old * jnp.exp(parameters.hardening_modulus * multiplier)
        volumetric_plastic = jnp.trace(next_plastic - plastic)
        void = jnp.maximum(void_old - (1.0 + void_old) * volumetric_plastic, 1.0e-8)
        pressure, q, _ = _invariants(stress)
        residual = q**2 + slope**2 * pressure * (pressure - pc)
        first_piola, inverse = _first_piola(stress, deformation)
        energy = 0.5 * jnp.sum(stress * (strain - next_plastic))
        dissipation = jnp.maximum(pc_old * multiplier, 0.0)
        speed = jnp.sqrt(
            (parameters.bulk_modulus + 4.0 * parameters.shear_modulus / 3.0)
            / jnp.maximum(density, 1.0e-30)
        )
        valid = (
            strain_valid
            & inverse.successful
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.isfinite(residual)
            & (pc > 0.0)
            & (void > 0.0)
            & (density > 0.0)
        )
        state = jnp.concatenate((next_plastic.reshape((9,)), pc[None], void[None]))
        diagnostics = {
            "yield_residual": residual,
            "plastic_multiplier": multiplier,
            "preconsolidation_pressure": pc,
            "void_ratio": void,
        }
        return (
            jnp.where(valid, first_piola, 0.0),
            jnp.where(valid, state, history),
            jnp.where(valid, energy, 0.0),
            jnp.where(valid, speed, 0.0),
            jnp.where(valid, dissipation, 0.0),
            jnp.where(yield_trial > 1.0e-10, 1, 0).astype(jnp.int32),
            valid,
            diagnostics,
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
        del time
        if not isinstance(parameters, ModifiedCamClayParameters):
            raise TypeError("parameters must be ModifiedCamClayParameters.")
        return self._response(
            jnp.asarray(deformation_gradient),
            jnp.asarray(committed_state),
            jnp.asarray(reference_density),
            parameters,
            step_size,
        )


__all__ = [
    "DruckerPragerMPMConstitutivePlan",
    "DruckerPragerParameters",
    "GeomechanicalFailureEvidence",
    "ModifiedCamClayMPMConstitutivePlan",
    "ModifiedCamClayParameters",
    "MohrCoulombMPMConstitutivePlan",
    "MohrCoulombParameters",
    "NonlocalSofteningPlan",
]
