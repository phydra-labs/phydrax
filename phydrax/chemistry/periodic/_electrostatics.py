#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable 3D Ewald electrostatics and separable GTH pseudopotentials."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract


class PeriodicEwaldResult(StrictModule, NonTrainableState):
    energy: Array
    forces: Array
    stress: Array
    real_energy: Array
    reciprocal_energy: Array
    self_energy: Array
    background_energy: Array
    net_charge: Array
    force_balance_residual: Array
    stress_symmetry_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy,
        forces,
        stress,
        components,
        net_charge,
        successful,
        plan_id,
        /,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        force = jnp.asarray(forces, dtype=energy_.dtype)
        stress_ = jnp.asarray(stress, dtype=energy_.dtype)
        real, reciprocal, self_energy, background = (
            jnp.asarray(value, dtype=energy_.dtype).reshape(()) for value in components
        )
        if force.ndim != 2 or force.shape[1] != 3 or stress_.shape != (3, 3):
            raise ValueError("Ewald forces and stress have invalid shapes.")
        balance = jnp.max(jnp.abs(jnp.sum(force, axis=0)), initial=0.0)
        stress_residual = jnp.max(jnp.abs(stress_ - stress_.T), initial=0.0)
        valid = (
            jnp.asarray(successful, dtype=bool)
            & jnp.isfinite(energy_)
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(stress_))
        )
        self.energy = energy_
        self.forces = force
        self.stress = stress_
        self.real_energy = real
        self.reciprocal_energy = reciprocal
        self.self_energy = self_energy
        self.background_energy = background
        self.net_charge = jnp.asarray(net_charge, dtype=energy_.dtype).reshape(())
        self.force_balance_residual = balance
        self.stress_symmetry_residual = stress_residual
        self.successful = valid
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-result",
                "plan": self.plan_id,
                "successful": bool(valid),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy_),
                        "forces": np.asarray(force),
                        "stress": np.asarray(stress_),
                        "components": np.asarray(
                            (real, reciprocal, self_energy, background)
                        ),
                    }
                ),
            }
        )


class PeriodicEwaldPlan(StrictModule, NonTrainableState):
    alpha: float = eqx.field(static=True)
    real_shell: int = eqx.field(static=True)
    reciprocal_shell: int = eqx.field(static=True)
    neutralizing_background: bool = eqx.field(static=True)
    minimum_distance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        alpha: float,
        /,
        *,
        real_shell: int = 4,
        reciprocal_shell: int = 4,
        neutralizing_background: bool = False,
        minimum_distance: float = 1.0e-10,
    ):
        alpha_ = float(alpha)
        real = int(real_shell)
        reciprocal = int(reciprocal_shell)
        minimum = float(minimum_distance)
        if (
            not isfinite(alpha_)
            or alpha_ <= 0.0
            or real < 0
            or reciprocal < 1
            or not isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("Ewald splitting, shells, or minimum distance is invalid.")
        self.alpha = alpha_
        self.real_shell = real
        self.reciprocal_shell = reciprocal
        self.neutralizing_background = bool(neutralizing_background)
        self.minimum_distance = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-plan",
                "alpha": alpha_,
                "real_shell": real,
                "reciprocal_shell": reciprocal,
                "neutralizing_background": self.neutralizing_background,
                "minimum_distance": minimum,
            }
        )

    def _energy(self, positions, charges, cell):
        determinant = jnp.dot(cell[0], jnp.cross(cell[1], cell[2]))
        volume = jnp.abs(determinant)
        inverse = (
            jnp.stack(
                (
                    jnp.cross(cell[1], cell[2]),
                    jnp.cross(cell[2], cell[0]),
                    jnp.cross(cell[0], cell[1]),
                ),
                axis=1,
            )
            / determinant
        )
        integers_real = jnp.stack(
            jnp.meshgrid(
                *(jnp.arange(-self.real_shell, self.real_shell + 1) for _ in range(3)),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))
        translations = integers_real @ cell
        displacement = (
            positions[:, None, None, :]
            - positions[None, :, None, :]
            + translations[None, None, :, :]
        )
        distance_squared = jnp.sum(displacement**2, axis=-1)
        zero_translation = jnp.all(integers_real == 0, axis=1)
        diagonal = jnp.eye(positions.shape[0], dtype=bool)[:, :, None]
        include = ~(diagonal & zero_translation[None, None, :])
        safe_distance = jnp.sqrt(
            jnp.where(
                include,
                jnp.maximum(distance_squared, self.minimum_distance**2),
                1.0,
            )
        )
        pair_charge = charges[:, None, None] * charges[None, :, None]
        real_energy = 0.5 * jnp.sum(
            jnp.where(
                include,
                pair_charge
                * jsp.special.erfc(self.alpha * safe_distance)
                / safe_distance,
                0.0,
            )
        )
        integers_reciprocal = jnp.stack(
            jnp.meshgrid(
                *(
                    jnp.arange(
                        -self.reciprocal_shell,
                        self.reciprocal_shell + 1,
                    )
                    for _ in range(3)
                ),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))
        nonzero = jnp.any(integers_reciprocal != 0, axis=1)
        wavevectors = 2.0 * jnp.pi * integers_reciprocal @ inverse.T
        squared = jnp.sum(wavevectors**2, axis=1)
        phase = positions @ wavevectors.T
        structure = jnp.sum(charges[:, None] * jnp.exp(1.0j * phase), axis=0)
        reciprocal_kernel = jnp.where(
            nonzero,
            jnp.exp(-squared / (4.0 * self.alpha**2)) / jnp.where(nonzero, squared, 1.0),
            0.0,
        )
        reciprocal_energy = (2.0 * jnp.pi / volume) * jnp.sum(
            reciprocal_kernel * jnp.abs(structure) ** 2
        )
        self_energy = -self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2)
        net_charge = jnp.sum(charges)
        background = (
            -jnp.pi * net_charge**2 / (2.0 * self.alpha**2 * volume)
            if self.neutralizing_background
            else jnp.asarray(0.0, dtype=positions.dtype)
        )
        return (
            real_energy + reciprocal_energy + self_energy + background,
            (real_energy, reciprocal_energy, self_energy, background),
        )

    def evaluate(
        self,
        positions: ArrayLike,
        charges: ArrayLike,
        cell_vectors: ArrayLike,
        /,
    ) -> PeriodicEwaldResult:
        coordinate = jnp.asarray(positions)
        charge = jnp.asarray(charges, dtype=coordinate.dtype)
        cell = jnp.asarray(cell_vectors, dtype=coordinate.dtype)
        if (
            coordinate.ndim != 2
            or coordinate.shape[1] != 3
            or charge.shape != (coordinate.shape[0],)
            or cell.shape != (3, 3)
        ):
            raise ValueError("Ewald positions, charges, and cell have invalid shapes.")
        coordinate_host = np.asarray(coordinate)
        cell_host = np.asarray(cell)
        determinant_host = float(np.linalg.det(cell_host))
        if (
            np.any(~np.isfinite(coordinate_host))
            or np.any(~np.isfinite(np.asarray(charge)))
            or np.any(~np.isfinite(cell_host))
            or abs(determinant_host) <= np.finfo(cell_host.dtype).eps
        ):
            raise ValueError(
                "Ewald geometry, charges, and cell must be finite and nonsingular."
            )
        fractional = coordinate_host @ np.linalg.inv(cell_host)
        fractional_displacement = fractional[:, None, :] - fractional[None, :, :]
        fractional_displacement -= np.rint(fractional_displacement)
        pair_distance = np.linalg.norm(fractional_displacement @ cell_host, axis=-1)
        np.fill_diagonal(pair_distance, np.inf)
        if np.min(pair_distance, initial=np.inf) <= self.minimum_distance:
            raise ValueError("Distinct Ewald sites violate minimum_distance.")
        net_charge = jnp.sum(charge)
        if not self.neutralizing_background and not bool(
            jnp.isclose(net_charge, 0.0, atol=1.0e-12)
        ):
            raise ValueError("Charged Ewald cells require neutralizing_background=True.")
        (energy, components), position_gradient = jax.value_and_grad(
            self._energy, argnums=0, has_aux=True
        )(coordinate, charge, cell)
        determinant = jnp.dot(cell[0], jnp.cross(cell[1], cell[2]))
        volume = jnp.abs(determinant)

        def strained_energy(strain):
            deformation = jnp.eye(3, dtype=cell.dtype) + strain
            deformed_positions = coordinate @ deformation.T
            deformed_cell = cell @ deformation.T
            return self._energy(deformed_positions, charge, deformed_cell)[0]

        strain_gradient = jax.grad(strained_energy)(jnp.zeros_like(cell))
        stress = 0.5 * (strain_gradient + jnp.conj(strain_gradient.T)) / volume
        successful = (
            (volume > 0.0)
            & jnp.all(jnp.isfinite(coordinate))
            & jnp.all(jnp.isfinite(charge))
        )
        return PeriodicEwaldResult(
            energy,
            -position_gradient,
            stress,
            components,
            net_charge,
            successful,
            self.plan_id,
        )


class GTHProjectorChannel(StrictModule, NonTrainableState):
    angular_momentum: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    coupling: Array
    channel_id: str = eqx.field(static=True)

    def __init__(self, angular_momentum: int, radius: float, coupling: ArrayLike, /):
        angular = int(angular_momentum)
        radius_ = float(radius)
        coupling_ = jnp.asarray(coupling)
        if (
            angular < 0
            or angular > 6
            or not isfinite(radius_)
            or radius_ <= 0.0
            or coupling_.ndim != 2
            or coupling_.shape[0] != coupling_.shape[1]
        ):
            raise ValueError(
                "GTH projector angular momentum, radius, or coupling is invalid."
            )
        if not np.allclose(np.asarray(coupling_), np.asarray(jnp.conj(coupling_.T))):
            raise ValueError("GTH projector coupling must be Hermitian.")
        self.angular_momentum = angular
        self.radius = radius_
        self.coupling = coupling_
        self.channel_id = canonical_fingerprint(
            {
                "kind": "gth-projector-channel",
                "angular_momentum": angular,
                "radius": radius_,
                "coupling": array_tree_fingerprint(np.asarray(coupling_)),
            }
        )


class GTHPseudopotentialPlan(StrictModule, NonTrainableState):
    ionic_charge: float = eqx.field(static=True)
    local_radius: float = eqx.field(static=True)
    local_coefficients: Array
    channels: tuple[GTHProjectorChannel, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ionic_charge: float,
        local_radius: float,
        local_coefficients: ArrayLike,
        channels: tuple[GTHProjectorChannel, ...] = (),
        /,
    ):
        charge = float(ionic_charge)
        radius = float(local_radius)
        coefficients = jnp.asarray(local_coefficients)
        channels_ = tuple(channels)
        if (
            not isfinite(charge)
            or charge <= 0.0
            or not isfinite(radius)
            or radius <= 0.0
            or coefficients.shape != (4,)
            or any(not isinstance(value, GTHProjectorChannel) for value in channels_)
        ):
            raise ValueError("GTH local parameters or projector channels are invalid.")
        angular = tuple(value.angular_momentum for value in channels_)
        if len(set(angular)) != len(angular):
            raise ValueError("GTH projector angular momenta must be unique.")
        self.ionic_charge = charge
        self.local_radius = radius
        self.local_coefficients = coefficients
        self.channels = channels_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gth-pseudopotential-plan",
                "ionic_charge": charge,
                "local_radius": radius,
                "local_coefficients": array_tree_fingerprint(np.asarray(coefficients)),
                "channels": [value.channel_id for value in channels_],
            }
        )

    def local_reciprocal(self, squared_wavevectors: ArrayLike, /) -> Array:
        squared = jnp.asarray(squared_wavevectors)
        scaled = self.local_radius**2 * squared
        gaussian = jnp.exp(-0.5 * scaled)
        polynomial = (
            self.local_coefficients[0]
            + self.local_coefficients[1] * (3.0 - scaled)
            + self.local_coefficients[2] * (15.0 - 10.0 * scaled + scaled**2)
            + self.local_coefficients[3]
            * (105.0 - 105.0 * scaled + 21.0 * scaled**2 - scaled**3)
        )
        coulomb = (
            -4.0
            * jnp.pi
            * self.ionic_charge
            * gaussian
            / jnp.where(squared > 0.0, squared, 1.0)
        )
        local = (2.0 * jnp.pi) ** 1.5 * self.local_radius**3 * gaussian * polynomial
        return jnp.where(squared > 0.0, coulomb + local, local)

    def nonlocal_energy(self, projector_overlaps: tuple[ArrayLike, ...], /) -> Array:
        if len(projector_overlaps) != len(self.channels):
            raise ValueError("GTH projector overlaps must align with channels.")
        energy = jnp.asarray(0.0, dtype=self.local_coefficients.dtype)
        for channel, overlaps in zip(self.channels, projector_overlaps, strict=True):
            values = jnp.asarray(overlaps)
            if values.shape[-1] != channel.coupling.shape[0]:
                raise ValueError(
                    "GTH projector overlap rank differs from channel coupling."
                )
            energy = energy + jnp.real(
                contract("...i,ij,...j->", jnp.conj(values), channel.coupling, values)
            )
        return energy


__all__ = [
    "GTHProjectorChannel",
    "GTHPseudopotentialPlan",
    "PeriodicEwaldPlan",
    "PeriodicEwaldResult",
]
