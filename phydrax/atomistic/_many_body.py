#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticTermEvaluation,
)
from ._system import PreparedAtomisticSystem


class WallKind(StrEnum):
    PLANE = "plane"
    SPHERE = "sphere"
    CYLINDER = "cylinder"


class ScalarWallPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    kind: WallKind = eqx.field(static=True)
    parameters: Array
    stiffness: float = eqx.field(static=True)
    inside: bool = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        kind: WallKind,
        parameters: ArrayLike,
        stiffness: float,
        /,
        *,
        inside: bool = True,
        name: str = "wall",
        force_group: int = 0,
    ):
        if not isinstance(kind, WallKind):
            raise TypeError("kind must be WallKind.")
        parameter = np.asarray(parameters, dtype=float)
        expected = {
            WallKind.PLANE: 4,
            WallKind.SPHERE: 4,
            WallKind.CYLINDER: 7,
        }
        if (
            parameter.shape != (expected[kind],)
            or np.any(~np.isfinite(parameter))
            or float(stiffness) <= 0.0
            or (kind is not WallKind.PLANE and parameter[-1] <= 0.0)
            or (
                kind is WallKind.CYLINDER
                and np.sum(parameter[3:6] * parameter[3:6]) <= 0.0
            )
            or (kind is WallKind.PLANE and np.sum(parameter[:3] * parameter[:3]) <= 0.0)
        ):
            raise ValueError("Wall geometry and stiffness are invalid.")
        if kind is WallKind.PLANE:
            normal_norm = np.sqrt(np.sum(parameter[:3] * parameter[:3]))
            parameter = parameter.copy()
            parameter[:3] /= normal_norm
            parameter[3] /= normal_norm
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError("Wall name and force group are invalid.")
        (
            self.kind,
            self.parameters,
            self.stiffness,
            self.inside,
            self.name,
            self.force_group,
        ) = (
            kind,
            jnp.asarray(parameter),
            float(stiffness),
            bool(inside),
            identifier,
            group,
        )
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            orthorhombic_periodic=False,
            triclinic_periodic=False,
            local_energy=True,
        )
        self.requirements = AtomisticPotentialRequirements()
        self.term_id = canonical_fingerprint(
            {
                "kind": "scalar-wall",
                "wall_kind": kind.value,
                "parameters": list(map(float, self.parameters)),
                "stiffness": self.stiffness,
                "inside": self.inside,
            }
        )

    def prepare(self, system: PreparedAtomisticSystem, /):
        return PreparedScalarWallPotential(self, system)


class PreparedScalarWallPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: ScalarWallPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan, system, /):
        (
            self.plan,
            self.system,
            self.name,
            self.force_group,
            self.term_id,
            self.capabilities,
            self.requirements,
        ) = (
            plan,
            system,
            plan.name,
            plan.force_group,
            plan.term_id,
            plan.capabilities,
            plan.requirements,
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-wall", "plan": plan.term_id, "system": system.prepared_id}
        )

    def energy(self, context, /):
        q = context.positions
        p = self.plan.parameters
        if self.plan.kind is WallKind.PLANE:
            normal, offset = p[:3], p[3]
            signed = q @ normal - offset
        elif self.plan.kind is WallKind.SPHERE:
            signed = jnp.sqrt(jnp.sum((q - p[:3]) ** 2, axis=-1)) - p[3]
        else:
            axis = p[3:6] / jnp.sqrt(jnp.sum(p[3:6] ** 2))
            relative = q - p[:3]
            radial = relative - jnp.sum(relative * axis, axis=-1)[:, None] * axis
            signed = jnp.sqrt(jnp.sum(radial**2, axis=-1)) - p[6]
        violation = jnp.maximum(signed if self.plan.inside else -signed, 0.0)
        atom = 0.5 * self.plan.stiffness * violation**2 * self.system.active_mask
        successful = jnp.all(jnp.isfinite(atom))
        return AtomisticTermEvaluation(jnp.sum(atom), atom, successful)


class ManifoldProjection(StrictModule):
    positions: Array
    velocities: Array
    position_residual: Array
    velocity_residual: Array
    successful: Array


class ManifoldConstraintPlan(StrictModule, NonTrainableState):
    kind: WallKind = eqx.field(static=True)
    parameters: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, kind: WallKind, parameters: ArrayLike, /):
        if not isinstance(kind, WallKind):
            raise TypeError("kind must be WallKind.")
        parameter = np.asarray(parameters, dtype=float)
        expected = {
            WallKind.PLANE: 4,
            WallKind.SPHERE: 4,
            WallKind.CYLINDER: 7,
        }
        if (
            parameter.shape != (expected[kind],)
            or np.any(~np.isfinite(parameter))
            or (kind is not WallKind.PLANE and parameter[-1] <= 0.0)
        ):
            raise ValueError("Manifold geometry is invalid.")
        axis = parameter[:3] if kind is WallKind.PLANE else parameter[3:6]
        if np.sum(axis * axis) <= 0.0:
            raise ValueError("Manifold normal or axis must be nonzero.")
        if kind is WallKind.PLANE:
            norm = np.sqrt(np.sum(parameter[:3] * parameter[:3]))
            parameter = parameter.copy()
            parameter[:3] /= norm
            parameter[3] /= norm
        self.kind, self.parameters = kind, jnp.asarray(parameter)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "manifold-constraint",
                "manifold": kind.value,
                "parameters": list(map(float, parameter)),
            }
        )

    def project(self, positions: ArrayLike, velocities: ArrayLike, /):
        q, v, p = jnp.asarray(positions), jnp.asarray(velocities), self.parameters
        if q.ndim != 2 or q.shape[1] != 3 or v.shape != q.shape:
            raise ValueError("Manifold positions and velocities must have shape (N,3).")
        nonsingular = jnp.asarray(True)
        if self.kind is WallKind.SPHERE:
            relative = q - p[:3]
            norm = jnp.sqrt(jnp.sum(relative**2, axis=-1, keepdims=True))
            nonsingular = jnp.all(norm > 0.0)
            normal = relative / jnp.where(norm > 0.0, norm, 1.0)
            q = p[:3] + p[3] * normal
            position_residual = jnp.max(
                jnp.abs(jnp.sqrt(jnp.sum((q - p[:3]) ** 2, axis=-1)) - p[3])
            )
        elif self.kind is WallKind.CYLINDER:
            axis = p[3:6] / jnp.sqrt(jnp.sum(p[3:6] ** 2))
            relative = q - p[:3]
            axial = jnp.sum(relative * axis, axis=-1, keepdims=True) * axis
            radial = relative - axial
            radial_norm = jnp.sqrt(jnp.sum(radial**2, axis=-1, keepdims=True))
            nonsingular = jnp.all(radial_norm > 0.0)
            normal = radial / jnp.where(radial_norm > 0.0, radial_norm, 1.0)
            q = p[:3] + axial + p[6] * normal
            projected_radial = q - p[:3] - axial
            position_residual = jnp.max(
                jnp.abs(jnp.sqrt(jnp.sum(projected_radial**2, axis=-1)) - p[6])
            )
        else:
            normal = p[:3] / jnp.sqrt(jnp.sum(p[:3] ** 2))
            distance = q @ normal - p[3]
            q = q - distance[:, None] * normal
            normal = jnp.broadcast_to(normal, q.shape)
            position_residual = jnp.max(jnp.abs(q @ normal[0] - p[3]))
        v = v - jnp.sum(v * normal, axis=-1)[:, None] * normal
        velocity_residual = jnp.max(jnp.abs(jnp.sum(v * normal, axis=-1)))
        successful = (
            nonsingular
            & jnp.all(jnp.isfinite(q))
            & jnp.all(jnp.isfinite(v))
            & (position_residual <= 1.0e-6)
            & (velocity_residual <= 1.0e-6)
        )
        return ManifoldProjection(q, v, position_residual, velocity_residual, successful)


class ActiveForceEvaluation(StrictModule):
    forces: Array
    orientations: Array
    successful: Array


class ActiveForcePlan(StrictModule, NonTrainableState):
    magnitude: float = eqx.field(static=True)
    rotational_diffusion: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, magnitude: float, rotational_diffusion: float, /):
        if (
            not np.isfinite(magnitude)
            or not np.isfinite(rotational_diffusion)
            or magnitude < 0.0
            or rotational_diffusion < 0.0
        ):
            raise ValueError("Active force parameters must be finite and non-negative.")
        self.magnitude, self.rotational_diffusion = (
            float(magnitude),
            float(rotational_diffusion),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "active-force",
                "magnitude": self.magnitude,
                "rotational_diffusion": self.rotational_diffusion,
            }
        )

    def evaluate(self, orientations: ArrayLike, key, dt, /):
        direction = jnp.asarray(orientations)
        step = jnp.asarray(dt, dtype=direction.dtype)
        if direction.ndim != 2 or direction.shape[-1] != 3:
            raise ValueError("Active orientations must have shape (particles, 3).")
        noise = jr.normal(key, direction.shape)
        proposed = (
            direction
            + jnp.sqrt(jnp.maximum(2.0 * self.rotational_diffusion * step, 0.0)) * noise
        )
        norm = jnp.sqrt(jnp.sum(proposed**2, axis=-1, keepdims=True))
        successful = (step >= 0.0) & jnp.all(norm > 0.0) & jnp.all(jnp.isfinite(proposed))
        direction = proposed / jnp.where(norm > 0.0, norm, 1.0)
        forces = self.magnitude * direction
        return ActiveForceEvaluation(
            jnp.where(successful, forces, jnp.nan),
            jnp.where(successful, direction, jnp.nan),
            successful,
        )


class DissipativeParticleDynamicsPlan(StrictModule, NonTrainableState):
    conservative: float = eqx.field(static=True)
    friction: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, conservative: float, friction: float, temperature: float, cutoff: float, /
    ):
        parameters = tuple(
            float(value) for value in (conservative, friction, temperature, cutoff)
        )
        if (
            any(not np.isfinite(value) for value in parameters)
            or min(parameters[:3]) < 0.0
            or parameters[3] <= 0.0
        ):
            raise ValueError("DPD parameters are invalid.")
        self.conservative, self.friction, self.temperature, self.cutoff = parameters
        self.plan_id = canonical_fingerprint(
            {"kind": "dpd", "parameters": list(parameters)}
        )

    def pair_force(
        self, displacement, relative_velocity, normal_noise, boltzmann_constant, dt, /
    ):
        displacement = jnp.asarray(displacement)
        relative_velocity = jnp.asarray(relative_velocity, dtype=displacement.dtype)
        normal_noise = jnp.asarray(normal_noise, dtype=displacement.dtype)
        if (
            displacement.ndim != 2
            or displacement.shape[1] != 3
            or relative_velocity.shape != displacement.shape
            or normal_noise.shape != displacement.shape[:1]
        ):
            raise ValueError("DPD pair arrays have incompatible shapes.")
        step = jnp.asarray(dt, dtype=displacement.dtype)
        thermal = jnp.asarray(boltzmann_constant, dtype=displacement.dtype)
        r = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
        direction = displacement / jnp.where(r[:, None] > 0.0, r[:, None], 1.0)
        weight = jnp.maximum(1.0 - r / self.cutoff, 0.0)
        conservative = self.conservative * weight
        dissipative = (
            -self.friction * weight**2 * jnp.sum(relative_velocity * direction, axis=-1)
        )
        random = (
            jnp.sqrt(
                2.0
                * self.friction
                * thermal
                * self.temperature
                / jnp.where(step > 0.0, step, 1.0)
            )
            * weight
            * normal_noise
        )
        successful = (
            (step > 0.0)
            & (thermal > 0.0)
            & jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(relative_velocity))
            & jnp.all(jnp.isfinite(normal_noise))
            & jnp.all((r > 0.0) | (weight == 0.0))
        )
        force = (conservative + dissipative + random)[:, None] * direction
        return jnp.where(successful, force, jnp.nan)


class ManyBodyKind(StrEnum):
    EAM = "eam"
    STILLINGER_WEBER = "stillinger-weber"
    TERSOFF = "tersoff"


class ManyBodyPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    kind: ManyBodyKind = eqx.field(static=True)
    parameters: Array
    cutoff: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        kind: ManyBodyKind,
        parameters: ArrayLike,
        cutoff: float,
        /,
        *,
        name: str | None = None,
        force_group: int = 0,
    ):
        if not isinstance(kind, ManyBodyKind):
            raise TypeError("kind must be ManyBodyKind.")
        parameter = np.asarray(parameters, dtype=float).reshape((-1,))
        expected_size = {
            ManyBodyKind.EAM: 5,
            ManyBodyKind.STILLINGER_WEBER: 10,
            ManyBodyKind.TERSOFF: 13,
        }[kind]
        valid_parameters = (
            np.all(np.isfinite(parameter)) and parameter.size == expected_size
        )
        if kind is ManyBodyKind.EAM:
            valid_parameters = valid_parameters and np.all(parameter > 0.0)
        elif kind is ManyBodyKind.STILLINGER_WEBER and valid_parameters:
            valid_parameters = (
                np.all(parameter[:5] > 0.0)
                and parameter[5] >= 0.0
                and parameter[6] > 1.0
                and np.all(parameter[7:9] > 0.0)
                and -1.0 <= parameter[9] <= 1.0
                and np.isclose(float(cutoff), parameter[6] * parameter[1])
            )
        elif kind is ManyBodyKind.TERSOFF and valid_parameters:
            valid_parameters = (
                np.all(parameter[:4] > 0.0)
                and parameter[4] >= 0.0
                and np.all(parameter[5:9] > 0.0)
                and -1.0 <= parameter[9] <= 1.0
                and parameter[10] > parameter[11] > 0.0
                and parameter[12] > 0.0
                and np.isclose(float(cutoff), parameter[10] + parameter[11])
                and parameter[12] == int(parameter[12])
                and int(parameter[12]) in (1, 3)
            )
        if (
            not valid_parameters
            or not np.isfinite(cutoff)
            or float(cutoff) <= 0.0
            or int(force_group) < 0
        ):
            raise ValueError("Many-body potential parameters are invalid.")
        self.kind, self.parameters, self.cutoff, self.name, self.force_group = (
            kind,
            jnp.asarray(parameter),
            float(cutoff),
            kind.value if name is None else str(name),
            int(force_group),
        )
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=True,
        )
        self.requirements = AtomisticPotentialRequirements(
            cutoff=self.cutoff, pair_geometry=True
        )
        self.term_id = canonical_fingerprint(
            {
                "kind": "many-body-potential",
                "many_body_kind": kind.value,
                "parameters": list(map(float, self.parameters.reshape((-1,)))),
                "cutoff": self.cutoff,
            }
        )

    def prepare(self, system, /):
        return PreparedManyBodyPotential(self, system)


class PreparedManyBodyPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: ManyBodyPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan, system, /):
        (
            self.plan,
            self.system,
            self.name,
            self.force_group,
            self.term_id,
            self.capabilities,
            self.requirements,
        ) = (
            plan,
            system,
            plan.name,
            plan.force_group,
            plan.term_id,
            plan.capabilities,
            plan.requirements,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-many-body",
                "plan": plan.term_id,
                "system": system.prepared_id,
            }
        )

    def energy(self, context, /):
        q = context.positions
        n = q.shape[0]
        displacement = q[:, None, :] - q[None, :, :]
        if context.cell is not None:
            displacement = context.cell.minimum_image(displacement)
        identity = jnp.eye(n, dtype=bool)
        squared_distance = jnp.sum(displacement**2, axis=-1)
        distance = jnp.sqrt(jnp.where(identity, 1.0, squared_distance))
        pair_mask = (
            (~identity)
            & self.system.active_mask[:, None]
            & self.system.active_mask[None, :]
            & (distance < self.plan.cutoff)
        )
        safe = jnp.where(pair_mask, distance, 1.0)
        p = self.plan.parameters
        if self.plan.kind is ManyBodyKind.EAM:
            cutoff_weight = jnp.where(
                pair_mask,
                0.5 * (1.0 + jnp.cos(jnp.pi * safe / self.plan.cutoff)),
                0.0,
            )
            density = jnp.sum(cutoff_weight * jnp.exp(-p[0] * (safe - p[1])), axis=1)
            embedding = jnp.where(
                density > 0.0,
                -p[2] * jnp.sqrt(jnp.where(density > 0.0, density, 1.0)),
                0.0,
            )
            pair = 0.5 * jnp.sum(
                cutoff_weight * p[3] * jnp.exp(-p[4] * (safe - p[1])), axis=1
            )
            atom = embedding + pair
        elif self.plan.kind is ManyBodyKind.STILLINGER_WEBER:
            (
                epsilon,
                sigma,
                amplitude,
                repulsion,
                repulsive_power,
                attractive_power,
                _,
                angular_strength,
                angular_range,
                target_cosine,
            ) = p
            cutoff_gap = jnp.where(pair_mask, safe - self.plan.cutoff, -1.0)
            radial_window = jnp.exp(sigma / cutoff_gap)
            ratio = sigma / safe
            directed_pair = (
                amplitude
                * epsilon
                * (repulsion * ratio**repulsive_power - ratio**attractive_power)
                * radial_window
            )
            atom = 0.5 * jnp.sum(jnp.where(pair_mask, directed_pair, 0.0), axis=1)
            for center in range(n):
                vectors = -displacement[center]
                norms = safe[center]
                cosine = contract("id,jd->ij", vectors, vectors) / (
                    norms[:, None] * norms[None, :]
                )
                triplet_mask = (
                    pair_mask[center, :, None] & pair_mask[center, None, :] & ~identity
                )
                radial = jnp.exp(
                    angular_range
                    * sigma
                    / jnp.where(pair_mask[center], norms - self.plan.cutoff, -1.0)
                )
                angular = (
                    angular_strength
                    * epsilon
                    * (cosine - target_cosine) ** 2
                    * radial[:, None]
                    * radial[None, :]
                )
                atom = atom.at[center].add(
                    0.5 * jnp.sum(jnp.where(triplet_mask, angular, 0.0))
                )
        else:
            (
                repulsive_amplitude,
                attractive_amplitude,
                repulsive_decay,
                attractive_decay,
                coordination_decay,
                beta,
                bond_order_power,
                angular_c,
                angular_d,
                angular_h,
                cutoff_center,
                cutoff_half_width,
                coordination_power,
            ) = p
            inner = cutoff_center - cutoff_half_width
            outer = cutoff_center + cutoff_half_width
            transition = 0.5 - 0.5 * jnp.sin(
                jnp.pi * (distance - cutoff_center) / (2.0 * cutoff_half_width)
            )
            cutoff_function = jnp.where(
                distance < inner,
                1.0,
                jnp.where(distance < outer, transition, 0.0),
            )
            cutoff_function = jnp.where(
                (~identity)
                & self.system.active_mask[:, None]
                & self.system.active_mask[None, :],
                cutoff_function,
                0.0,
            )
            repulsive = repulsive_amplitude * jnp.exp(-repulsive_decay * safe)
            attractive = -attractive_amplitude * jnp.exp(-attractive_decay * safe)
            atom = jnp.zeros((n,), dtype=q.dtype)
            for center in range(n):
                vectors = -displacement[center]
                norms = safe[center]
                cosine = contract("id,jd->ij", vectors, vectors) / (
                    norms[:, None] * norms[None, :]
                )
                angular = (
                    1.0
                    + angular_c**2 / angular_d**2
                    - angular_c**2 / (angular_d**2 + (angular_h - cosine) ** 2)
                )
                for neighbor in range(n):
                    third_mask = (
                        (~identity[neighbor])
                        & (~identity[center])
                        & (jnp.arange(n) != neighbor)
                    )
                    separation = norms[neighbor] - norms
                    coordination = jnp.sum(
                        jnp.where(
                            third_mask,
                            cutoff_function[center]
                            * angular[neighbor]
                            * jnp.exp(
                                (coordination_decay * separation) ** coordination_power
                            ),
                            0.0,
                        )
                    )
                    bond_order = (1.0 + (beta * coordination) ** bond_order_power) ** (
                        -1.0 / (2.0 * bond_order_power)
                    )
                    directed = cutoff_function[center, neighbor] * (
                        repulsive[center, neighbor]
                        + bond_order * attractive[center, neighbor]
                    )
                    atom = atom.at[center].add(0.5 * directed)
        success = jnp.all(jnp.isfinite(atom)) & jnp.all(~pair_mask | (distance > 0.0))
        return AtomisticTermEvaluation(jnp.sum(atom), atom, success)


def EAMPotential(parameters, cutoff, **kwargs):
    """Analytic Finnis-Sinclair EAM with [density_decay, r0, embed, pair, pair_decay]."""
    return ManyBodyPotential(ManyBodyKind.EAM, parameters, cutoff, **kwargs)


def StillingerWeberPotential(parameters, cutoff, **kwargs):
    """Stillinger-Weber with [epsilon, sigma, A, B, p, q, a, lambda, gamma, cos0]."""
    return ManyBodyPotential(ManyBodyKind.STILLINGER_WEBER, parameters, cutoff, **kwargs)


def TersoffPotential(parameters, cutoff, **kwargs):
    """Tersoff with [A, B, lambda1, lambda2, lambda3, beta, n, c, d, h, R, D, m]."""
    return ManyBodyPotential(ManyBodyKind.TERSOFF, parameters, cutoff, **kwargs)


__all__ = [
    "ActiveForceEvaluation",
    "ManifoldProjection",
    "ActiveForcePlan",
    "DissipativeParticleDynamicsPlan",
    "EAMPotential",
    "ManifoldConstraintPlan",
    "ManyBodyKind",
    "ManyBodyPotential",
    "ScalarWallPotential",
    "StillingerWeberPotential",
    "TersoffPotential",
    "WallKind",
]
