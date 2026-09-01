#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    PreparedRigidBodySet,
    rigid_body_kick_drift_kick,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._sites import AtomisticInteractionSiteState


class SplittingOperatorKind(StrEnum):
    DRIFT = "drift"
    FORCE_KICK = "force-kick"
    THERMOSTAT = "thermostat"
    POSITION_CONSTRAINT = "position-constraint"
    MOMENTUM_CONSTRAINT = "momentum-constraint"
    BAROSTAT = "barostat"
    CELL_UPDATE = "cell-update"


class AtomisticSplittingPlan(StrictModule, NonTrainableState):
    operators: tuple[SplittingOperatorKind, ...] = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, operators, coefficients, /):
        ops = tuple(operators)
        coeff = tuple(float(value) for value in coefficients)
        if (
            not ops
            or len(ops) != len(coeff)
            or any(not isinstance(value, SplittingOperatorKind) for value in ops)
            or any(not np.isfinite(value) for value in coeff)
        ):
            raise ValueError("Splitting operators and coefficients are invalid.")
        self.operators = ops
        self.coefficients = coeff
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-splitting",
                "operators": [value.value for value in ops],
                "coefficients": list(coeff),
            }
        )

    def apply(self, state, step_size, actions, /):
        missing = tuple(kind for kind in set(self.operators) if kind not in actions)
        if missing:
            raise ValueError(
                "Splitting actions are missing: "
                + ", ".join(kind.value for kind in missing)
            )
        value = state
        for kind, coefficient in zip(self.operators, self.coefficients, strict=True):
            value = actions[kind](value, coefficient * step_size)
        return value


class ThermostatResult(StrictModule):
    momenta: Array
    auxiliary: Array
    heat: Array
    successful: Array


class AbstractThermostatPlan(StrictModule, NonTrainableState):
    plan_id: AbstractAttribute[str]

    @abc.abstractmethod
    def apply(
        self, momenta, masses, mobile_mask, key, step, dt, units, /, *, auxiliary=None
    ) -> ThermostatResult:
        raise NotImplementedError


class BussiThermostatPlan(AbstractThermostatPlan):
    temperature: float = eqx.field(static=True)
    time_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, temperature: float, time_constant: float, /):
        if min(float(temperature), float(time_constant)) <= 0.0:
            raise ValueError("Bussi temperature and time constant must be positive.")
        self.temperature, self.time_constant = float(temperature), float(time_constant)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bussi-thermostat",
                "temperature": self.temperature,
                "time_constant": self.time_constant,
            }
        )

    def apply(
        self, momenta, masses, mobile_mask, key, step, dt, units, /, *, auxiliary=None
    ):
        del auxiliary
        p = jnp.asarray(momenta)
        mobile = jnp.asarray(mobile_mask)
        dof = jnp.sum(mobile) * p.shape[-1]
        kinetic = (
            0.5
            * units.kinetic_to_energy
            * jnp.sum(jnp.where(mobile[:, None], p * p / masses[:, None], 0.0))
        )
        target = 0.5 * dof * units.boltzmann_constant * self.temperature
        decay = jnp.exp(-dt / self.time_constant)
        subkey = jr.fold_in(key, jnp.asarray(step, dtype=jnp.uint32))
        normal = jr.normal(subkey)
        chi = jr.chisquare(jr.fold_in(subkey, 1), jnp.maximum(dof - 1, 1))
        ratio = target / jnp.maximum(kinetic, 1e-30)
        scale_squared = (
            decay
            + (1.0 - decay) * ratio * (chi + normal**2) / dof
            + 2.0 * normal * jnp.sqrt(decay * (1.0 - decay) * ratio / dof)
        )
        scale = jnp.sqrt(jnp.maximum(scale_squared, 0.0))
        result = jnp.where(mobile[:, None], scale * p, 0.0)
        after = 0.5 * units.kinetic_to_energy * jnp.sum(result * result / masses[:, None])
        return ThermostatResult(
            result,
            jnp.asarray([scale]),
            after - kinetic,
            (kinetic > 0.0)
            & (dof > 0)
            & jnp.isfinite(scale)
            & jnp.all(jnp.isfinite(result)),
        )


class NoseHooverChainPlan(AbstractThermostatPlan):
    temperature: float = eqx.field(static=True)
    chain_length: int = eqx.field(static=True)
    time_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, temperature: float, /, *, chain_length: int = 3, time_constant: float = 0.1
    ):
        if float(temperature) <= 0 or int(chain_length) <= 0 or float(time_constant) <= 0:
            raise ValueError("Nose-Hoover chain parameters are invalid.")
        self.temperature, self.chain_length, self.time_constant = (
            float(temperature),
            int(chain_length),
            float(time_constant),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nose-hoover-chain",
                "temperature": self.temperature,
                "chain_length": self.chain_length,
                "time_constant": self.time_constant,
            }
        )

    def apply(
        self, momenta, masses, mobile_mask, key, step, dt, units, /, *, auxiliary=None
    ):
        del key, step
        p = jnp.asarray(momenta)
        mobile = jnp.asarray(mobile_mask)
        dof = jnp.sum(mobile) * p.shape[-1]
        thermal = units.boltzmann_constant * self.temperature
        kinetic = (
            0.5
            * units.kinetic_to_energy
            * jnp.sum(jnp.where(mobile[:, None], p * p / masses[:, None], 0.0))
        )
        velocity = (
            jnp.zeros((self.chain_length,), dtype=p.dtype)
            if auxiliary is None
            else jnp.asarray(auxiliary, dtype=p.dtype)
        )
        if velocity.shape != (self.chain_length,):
            raise ValueError("Nose-Hoover auxiliary state has the wrong chain length.")
        masses_chain = jnp.full(
            (self.chain_length,), thermal * self.time_constant**2, dtype=p.dtype
        )
        masses_chain = masses_chain.at[0].set(dof * thermal * self.time_constant**2)
        forces = jnp.zeros_like(velocity)
        forces = forces.at[0].set((2.0 * kinetic - dof * thermal) / masses_chain[0])
        if self.chain_length > 1:
            forces = forces.at[1:].set(
                (masses_chain[:-1] * velocity[:-1] ** 2 - thermal) / masses_chain[1:]
            )
        velocity = velocity + 0.25 * dt * forces
        scale = jnp.exp(-0.5 * dt * velocity[0])
        result = jnp.where(mobile[:, None], scale * p, 0.0)
        after = (
            0.5
            * units.kinetic_to_energy
            * jnp.sum(jnp.where(mobile[:, None], result * result / masses[:, None], 0.0))
        )
        updated_forces = forces.at[0].set((2.0 * after - dof * thermal) / masses_chain[0])
        if self.chain_length > 1:
            updated_forces = updated_forces.at[1:].set(
                (masses_chain[:-1] * velocity[:-1] ** 2 - thermal) / masses_chain[1:]
            )
        velocity = velocity + 0.25 * dt * updated_forces
        successful = (
            (dof > 0) & jnp.all(jnp.isfinite(result)) & jnp.all(jnp.isfinite(velocity))
        )
        return ThermostatResult(result, velocity, after - kinetic, successful)


class GeneralizedLangevinPlan(AbstractThermostatPlan):
    drift_matrix: Array
    diffusion_factor: Array
    temperature: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, drift_matrix: ArrayLike, diffusion_factor: ArrayLike, temperature: float, /
    ):
        drift_host = np.asarray(drift_matrix, dtype=float)
        diffusion_host = np.asarray(diffusion_factor, dtype=float)
        if (
            drift_host.ndim != 2
            or drift_host.shape[0] != drift_host.shape[1]
            or diffusion_host.shape != drift_host.shape
            or np.any(~np.isfinite(drift_host))
            or np.any(~np.isfinite(diffusion_host))
            or float(temperature) <= 0.0
        ):
            raise ValueError("GLE matrices or temperature are invalid.")
        drift, diffusion = jnp.asarray(drift_host), jnp.asarray(diffusion_host)
        self.drift_matrix, self.diffusion_factor, self.temperature = (
            drift,
            diffusion,
            float(temperature),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "generalized-langevin",
                "drift": drift_host.tolist(),
                "diffusion": diffusion_host.tolist(),
                "temperature": self.temperature,
            }
        )

    def apply(
        self, momenta, masses, mobile_mask, key, step, dt, units, /, *, auxiliary=None
    ):
        p = jnp.asarray(momenta)
        auxiliary_count = self.drift_matrix.shape[0] - 1
        expected = p.shape + (auxiliary_count,)
        memory = (
            jnp.zeros(expected, dtype=p.dtype)
            if auxiliary is None
            else jnp.asarray(auxiliary, dtype=p.dtype)
        )
        if memory.shape != expected:
            raise ValueError("GLE auxiliary state has the wrong shape.")
        mass = jnp.asarray(masses, dtype=p.dtype)
        if mass.shape != p.shape[:1]:
            raise ValueError("GLE masses must align with particles.")
        momentum_scale = jnp.sqrt(
            mass[:, None]
            * units.boltzmann_constant
            * self.temperature
            / units.kinetic_to_energy
        )
        normalized_momenta = p / momentum_scale
        state = jnp.concatenate((normalized_momenta[..., None], memory), axis=-1)
        noise = jr.normal(
            jr.fold_in(key, jnp.asarray(step, dtype=jnp.uint32)),
            state.shape,
        )
        update = (
            state
            - dt * contract("...i,ji->...j", state, self.drift_matrix)
            + jnp.sqrt(dt) * contract("...i,ji->...j", noise, self.diffusion_factor)
        )
        mobile = jnp.asarray(mobile_mask)
        result = jnp.where(mobile[:, None], update[..., 0] * momentum_scale, 0.0)
        memory = jnp.where(mobile[:, None, None], update[..., 1:], 0.0)
        heat = 0.5 * units.kinetic_to_energy * jnp.sum((result**2 - p**2) / mass[:, None])
        successful = (
            jnp.all(jnp.isfinite(update))
            & jnp.all(jnp.isfinite(momentum_scale))
            & jnp.all(momentum_scale > 0.0)
        )
        return ThermostatResult(result, memory, heat, successful)


class NoisyForceLangevinPlan(AbstractThermostatPlan):
    base: GeneralizedLangevinPlan
    force_noise_variance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, friction: float, force_noise_variance: float, temperature: float, /
    ):
        if min(friction, force_noise_variance, temperature) <= 0:
            raise ValueError("Noisy-force Langevin parameters must be positive.")
        base = GeneralizedLangevinPlan(
            jnp.asarray([[friction]]),
            jnp.asarray([[jnp.sqrt(2.0 * friction + force_noise_variance)]]),
            temperature,
        )
        self.base = base
        self.force_noise_variance = float(force_noise_variance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "noisy-force-langevin",
                "base": base.plan_id,
                "force_noise_variance": self.force_noise_variance,
            }
        )

    def apply(
        self, momenta, masses, mobile_mask, key, step, dt, units, /, *, auxiliary=None
    ):
        return self.base.apply(
            momenta,
            masses,
            mobile_mask,
            key,
            step,
            dt,
            units,
            auxiliary=auxiliary,
        )


class AnisotropicPressurePlan(StrictModule, NonTrainableState):
    target_pressure: Array
    compressibility: Array
    semi_isotropic: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_pressure: ArrayLike,
        compressibility: ArrayLike,
        /,
        *,
        semi_isotropic: bool = False,
    ):
        pressure_host = np.asarray(target_pressure, dtype=float)
        compress_host = np.asarray(compressibility, dtype=float)
        if (
            pressure_host.shape not in ((), (3,), (3, 3))
            or compress_host.shape not in ((), (3,), (3, 3))
            or np.any(~np.isfinite(pressure_host))
            or np.any(~np.isfinite(compress_host) | (compress_host <= 0.0))
            or (
                pressure_host.ndim == 2
                and not np.allclose(pressure_host, pressure_host.T)
            )
            or (
                compress_host.ndim == 2
                and not np.allclose(compress_host, compress_host.T)
            )
        ):
            raise ValueError("Pressure target or compressibility is invalid.")
        pressure = jnp.asarray(pressure_host)
        compress = jnp.asarray(compress_host)
        self.target_pressure, self.compressibility, self.semi_isotropic = (
            pressure,
            compress,
            bool(semi_isotropic),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "anisotropic-pressure",
                "pressure_shape": list(pressure.shape),
                "compressibility_shape": list(compress.shape),
                "semi_isotropic": self.semi_isotropic,
            }
        )

    def update_cell(self, cell_vectors, observed_pressure, dt, /):
        cell = jnp.asarray(cell_vectors)
        observed_value = jnp.asarray(observed_pressure, dtype=cell.dtype)
        if cell.shape != (3, 3) or observed_value.shape not in ((), (3,), (3, 3)):
            raise ValueError(
                "Pressure update requires cell and scalar/vector/tensor pressure."
            )
        target = (
            jnp.eye(3, dtype=cell.dtype) * self.target_pressure
            if self.target_pressure.ndim == 0
            else jnp.diag(self.target_pressure)
            if self.target_pressure.ndim == 1
            else self.target_pressure
        )
        observed = (
            jnp.eye(3, dtype=cell.dtype) * observed_value
            if observed_value.ndim == 0
            else jnp.diag(observed_value)
            if observed_value.ndim == 1
            else observed_value
        )
        compress = (
            jnp.eye(3, dtype=cell.dtype) * self.compressibility
            if self.compressibility.ndim == 0
            else jnp.diag(self.compressibility)
            if self.compressibility.ndim == 1
            else self.compressibility
        )
        strain = jnp.asarray(dt) * compress * (observed - target)
        strain = 0.5 * (strain + strain.T)
        if self.semi_isotropic:
            lateral = 0.5 * (strain[0, 0] + strain[1, 1])
            strain = jnp.diag(jnp.asarray([lateral, lateral, strain[2, 2]]))
        candidate = cell @ (jnp.eye(3, dtype=cell.dtype) + strain).T
        determinant = jnp.sum(candidate[0] * jnp.cross(candidate[1], candidate[2]))
        successful = (
            (jnp.asarray(dt) > 0.0) & jnp.isfinite(determinant) & (determinant > 0.0)
        )
        return jnp.where(successful, candidate, jnp.nan)


class RigidAtomisticCoordinateMap(StrictModule, NonTrainableState):
    bodies: PreparedRigidBodySet
    body_indices: Array
    local_positions: Array
    site_ids: Array
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        bodies: PreparedRigidBodySet,
        body_indices: ArrayLike,
        local_positions: ArrayLike,
        site_ids: ArrayLike,
        /,
    ):
        index, local, ids = (
            jnp.asarray(body_indices, dtype=jnp.int32),
            jnp.asarray(local_positions),
            jnp.asarray(site_ids, dtype=jnp.int64),
        )
        if index.ndim != 1 or local.shape != (index.size, 3) or ids.shape != index.shape:
            raise ValueError("Rigid atomistic site routes are invalid.")
        self.bodies, self.body_indices, self.local_positions, self.site_ids = (
            bodies,
            index,
            local,
            ids,
        )
        self.map_id = canonical_fingerprint(
            {
                "kind": "rigid-atomistic-map",
                "bodies": bodies.prepared_id,
                "count": int(index.size),
            }
        )

    def realize(
        self, kinematics: RigidBodyKinematics, /
    ) -> AtomisticInteractionSiteState:
        quaternion = kinematics.orientation[self.body_indices]
        w, vector = quaternion[:, :1], quaternion[:, 1:]
        local = self.local_positions
        rotated = local + 2.0 * jnp.cross(vector, jnp.cross(vector, local) + w * local)
        positions = kinematics.position[self.body_indices] + rotated
        valid = jnp.all(jnp.isfinite(positions))
        mask = jnp.ones((positions.shape[0],), dtype=bool)
        return AtomisticInteractionSiteState(
            positions, mask, mask, mask, jnp.asarray(jnp.inf), valid, self.map_id
        )


class RotationalAtomisticState(StrictModule):
    kinematics: RigidBodyKinematics
    force: Array
    torque: Array
    successful: Array


def rotational_velocity_verlet(
    bodies: PreparedRigidBodySet,
    state: RotationalAtomisticState,
    step_size: ArrayLike,
    load_function,
    /,
):
    result = rigid_body_kick_drift_kick(
        bodies,
        state.kinematics,
        RigidBodyLoad(state.force, state.torque),
        jnp.zeros((), dtype=jnp.asarray(step_size).dtype),
        jnp.asarray(step_size),
        load_function,
    )
    return RotationalAtomisticState(
        result.kinematics, result.load.force, result.load.torque, result.successful
    )


class BrownianDynamicsPlan(StrictModule, NonTrainableState):
    mobility: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, mobility: float, temperature: float, /):
        if min(float(mobility), float(temperature)) <= 0:
            raise ValueError("Brownian mobility and temperature must be positive.")
        self.mobility, self.temperature = float(mobility), float(temperature)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "brownian-dynamics",
                "mobility": self.mobility,
                "temperature": self.temperature,
            }
        )

    def step(self, positions, forces, dt, key, units, /):
        noise = jr.normal(key, jnp.asarray(positions).shape)
        diffusion = self.mobility * units.boltzmann_constant * self.temperature
        return (
            positions
            + dt * self.mobility * forces
            + jnp.sqrt(2.0 * diffusion * dt) * noise
        )


__all__ = [
    "AbstractThermostatPlan",
    "AnisotropicPressurePlan",
    "AtomisticSplittingPlan",
    "BrownianDynamicsPlan",
    "BussiThermostatPlan",
    "GeneralizedLangevinPlan",
    "NoisyForceLangevinPlan",
    "NoseHooverChainPlan",
    "RigidAtomisticCoordinateMap",
    "RotationalAtomisticState",
    "SplittingOperatorKind",
    "ThermostatResult",
    "rotational_velocity_verlet",
]
