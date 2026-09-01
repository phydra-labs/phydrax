#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._sites import AtomisticInteractionSiteState


class PermanentMultipoleSiteData(StrictModule, NonTrainableState):
    charges: Array
    dipoles: Array
    quadrupoles: Array
    polarizabilities: Array
    damping: Array
    multipole_id: str = eqx.field(static=True)

    def __init__(
        self,
        charges: ArrayLike,
        dipoles: ArrayLike,
        quadrupoles: ArrayLike,
        polarizabilities: ArrayLike,
        damping: ArrayLike,
        /,
    ):
        charge = np.asarray(charges, dtype=float)
        dipole = np.asarray(dipoles, dtype=float)
        quadrupole = np.asarray(quadrupoles, dtype=float)
        polar = np.asarray(polarizabilities, dtype=float)
        damp = np.asarray(damping, dtype=float)
        count = charge.size
        if (
            charge.shape != (count,)
            or dipole.shape != (count, 3)
            or quadrupole.shape != (count, 3, 3)
            or polar.shape != (count,)
            or damp.shape != (count,)
        ):
            raise ValueError("Multipole arrays have incompatible shapes.")
        if (
            np.any(~np.isfinite(charge))
            or np.any(~np.isfinite(dipole))
            or np.any(~np.isfinite(quadrupole))
            or np.any(polar < 0.0)
            or np.any(damp <= 0.0)
        ):
            raise ValueError("Multipole arrays are invalid.")
        (
            self.charges,
            self.dipoles,
            self.quadrupoles,
            self.polarizabilities,
            self.damping,
        ) = (jnp.asarray(value) for value in (charge, dipole, quadrupole, polar, damp))
        self.multipole_id = canonical_fingerprint(
            {
                "kind": "permanent-multipoles",
                "arrays": array_tree_fingerprint(
                    {
                        "q": charge,
                        "mu": dipole,
                        "Q": quadrupole,
                        "alpha": polar,
                        "damping": damp,
                    }
                ),
            }
        )


class PolarizationPlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_iterations: int = 100,
        tolerance: float = 1e-8,
        relaxation: float = 0.7,
    ):
        if (
            int(maximum_iterations) <= 0
            or float(tolerance) <= 0
            or not 0 < float(relaxation) <= 1
        ):
            raise ValueError("Polarization solver parameters are invalid.")
        self.maximum_iterations, self.tolerance, self.relaxation = (
            int(maximum_iterations),
            float(tolerance),
            float(relaxation),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-plan",
                "iterations": self.maximum_iterations,
                "tolerance": self.tolerance,
                "relaxation": self.relaxation,
            }
        )


class PolarizationState(StrictModule):
    induced_dipoles: Array
    residual: Array
    iterations: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PolarizationEvaluation(StrictModule):
    energy: Array
    forces: Array
    state: PolarizationState
    successful: Array


def _electric_field(
    positions: Array, multipoles: PermanentMultipoleSiteData, induced: Array, /
):
    displacement = positions[:, None, :] - positions[None, :, :]
    distance2 = jnp.sum(displacement**2, axis=-1)
    identity_mask = jnp.eye(positions.shape[0], dtype=bool)
    safe2 = jnp.where(identity_mask, 1.0, distance2)
    distance = jnp.sqrt(safe2)
    direction = displacement / distance[..., None]
    damping = 1.0 - jnp.exp(
        -jnp.sqrt(multipoles.damping[:, None] * multipoles.damping[None, :]) * distance**3
    )
    charge_field = multipoles.charges[None, :, None] * direction / safe2[..., None]
    total_dipole = multipoles.dipoles + induced
    dipole_dot = contract("ijd,jd->ij", direction, total_dipole)
    dipole_field = (
        3.0 * direction * dipole_dot[..., None] - total_dipole[None, :, :]
    ) / (distance**3)[..., None]
    field = jnp.sum(
        jnp.where(
            identity_mask[..., None],
            0.0,
            damping[..., None] * (charge_field + dipole_field),
        ),
        axis=1,
    )
    return field, jnp.min(jnp.where(identity_mask, jnp.inf, distance))


def solve_induced_dipoles(
    plan: PolarizationPlan,
    positions: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
) -> PolarizationState:
    coordinate = jnp.asarray(positions)
    induced = jnp.zeros_like(multipoles.dipoles)

    def iteration(_, carry):
        value, residual, iterations = carry
        field, _ = _electric_field(coordinate, multipoles, value)
        proposed = multipoles.polarizabilities[:, None] * field
        relaxed = (1.0 - plan.relaxation) * value + plan.relaxation * proposed
        next_residual = jnp.max(jnp.sqrt(jnp.sum((relaxed - value) ** 2, axis=-1)))
        active = residual > plan.tolerance
        return (
            jnp.where(active, relaxed, value),
            jnp.where(active, next_residual, residual),
            iterations + active.astype(jnp.int32),
        )

    induced, residual, iterations = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        iteration,
        (
            induced,
            jnp.asarray(jnp.inf),
            jnp.zeros((), dtype=jnp.int32),
        ),
    )
    converged = residual <= plan.tolerance
    successful = converged & jnp.all(jnp.isfinite(induced))
    return PolarizationState(
        induced,
        residual,
        iterations,
        converged,
        successful,
        plan.plan_id,
    )


def polarization_energy(
    plan: PolarizationPlan,
    positions: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
):
    coordinate = jnp.asarray(positions)
    state = solve_induced_dipoles(plan, coordinate, multipoles)
    induced = jax.lax.stop_gradient(state.induced_dipoles)
    permanent_field, margin = _electric_field(
        coordinate, multipoles, jnp.zeros_like(induced)
    )
    total_field, _ = _electric_field(coordinate, multipoles, induced)
    induced_field = total_field - permanent_field
    self_energy = 0.5 * jnp.sum(
        induced**2
        / jnp.where(
            multipoles.polarizabilities[:, None] > 0.0,
            multipoles.polarizabilities[:, None],
            1.0,
        )
    )
    interaction = -jnp.sum(induced * permanent_field) - 0.5 * jnp.sum(
        induced * induced_field
    )
    energy = self_energy + interaction
    return energy, (state, margin)


def evaluate_polarization(
    plan: PolarizationPlan,
    positions: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
) -> PolarizationEvaluation:
    (energy, auxiliary), gradient = jax.value_and_grad(
        lambda value: polarization_energy(plan, value, multipoles), has_aux=True
    )(jnp.asarray(positions))
    state, margin = auxiliary
    successful = (
        state.successful
        & (margin > 0.0)
        & jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(gradient))
    )
    return PolarizationEvaluation(
        jnp.where(successful, energy, jnp.nan),
        jnp.where(successful, -gradient, jnp.nan),
        state,
        successful,
    )


def implicit_polarization_jvp(
    plan: PolarizationPlan,
    positions: ArrayLike,
    tangent: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
):
    coordinate = jnp.asarray(positions)
    direction = jnp.asarray(tangent)
    state = solve_induced_dipoles(plan, coordinate, multipoles)

    def fixed_map(position, induced):
        field, _ = _electric_field(position, multipoles, induced)
        return multipoles.polarizabilities[:, None] * field

    _, forcing = jax.jvp(
        lambda value: fixed_map(value, state.induced_dipoles),
        (coordinate,),
        (direction,),
    )

    def iteration(_, carry):
        value, residual = carry
        _, action = jax.jvp(
            lambda induced: fixed_map(coordinate, induced),
            (state.induced_dipoles,),
            (value,),
        )
        proposed = forcing + action
        next_value = (1.0 - plan.relaxation) * value + plan.relaxation * proposed
        residual = jnp.max(jnp.sqrt(jnp.sum((next_value - value) ** 2, axis=-1)))
        return next_value, residual

    derivative, residual = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        iteration,
        (jnp.zeros_like(state.induced_dipoles), jnp.asarray(jnp.inf)),
    )
    successful = (
        state.successful
        & (residual <= plan.tolerance)
        & jnp.all(jnp.isfinite(derivative))
    )
    return state.induced_dipoles, jnp.where(successful, derivative, jnp.nan)


class MultipolePMEPlan(StrictModule, NonTrainableState):
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, grid_shape: tuple[int, int, int], alpha: float, /):
        shape = tuple(int(value) for value in grid_shape)
        if len(shape) != 3 or any(value < 4 for value in shape) or float(alpha) <= 0.0:
            raise ValueError("Multipole PME grid and splitting parameter are invalid.")
        self.grid_shape, self.alpha = shape, float(alpha)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multipole-pme",
                "grid_shape": list(shape),
                "alpha": self.alpha,
            }
        )

    def energy(
        self,
        site_state: AtomisticInteractionSiteState,
        multipoles: PermanentMultipoleSiteData,
        cell_vectors: ArrayLike,
        coulomb_constant: float,
        /,
    ):
        if multipoles.charges.shape != site_state.active_mask.shape:
            raise ValueError("Multipoles and interaction sites must have equal capacity.")
        vectors = jnp.asarray(cell_vectors)
        determinant = jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2]))
        inverse = (
            jnp.stack(
                (
                    jnp.cross(vectors[1], vectors[2]),
                    jnp.cross(vectors[2], vectors[0]),
                    jnp.cross(vectors[0], vectors[1]),
                ),
                axis=1,
            )
            / determinant
        )
        shape = jnp.asarray(self.grid_shape, dtype=site_state.positions.dtype)
        fractional = contract("nd,di->ni", site_state.positions, inverse)
        anchor_index = jnp.argmax(site_state.active_mask.astype(jnp.int32))
        relative_fractional = fractional - fractional[anchor_index]
        scaled = jnp.mod(relative_fractional + 0.5, 1.0) * shape
        base = jax.lax.stop_gradient(jnp.floor(scaled).astype(jnp.int32))
        remainder = scaled - base
        charge_grid = jnp.zeros(self.grid_shape, dtype=site_state.positions.dtype)
        dipole_grid = jnp.zeros(self.grid_shape + (3,), dtype=site_state.positions.dtype)
        quadrupole_grid = jnp.zeros(
            self.grid_shape + (3, 3), dtype=site_state.positions.dtype
        )
        active = site_state.active_mask.astype(site_state.positions.dtype)
        for x_offset in (0, 1):
            for y_offset in (0, 1):
                for z_offset in (0, 1):
                    corner = jnp.asarray((x_offset, y_offset, z_offset))
                    weight_axis = jnp.where(
                        corner[None, :] == 1, remainder, 1.0 - remainder
                    )
                    weight = jnp.prod(weight_axis, axis=-1) * active
                    index = (base + corner[None, :]) % jnp.asarray(
                        self.grid_shape, dtype=jnp.int32
                    )
                    route = (index[:, 0], index[:, 1], index[:, 2])
                    charge_grid = charge_grid.at[route].add(weight * multipoles.charges)
                    dipole_grid = dipole_grid.at[route].add(
                        weight[:, None] * multipoles.dipoles
                    )
                    quadrupole_grid = quadrupole_grid.at[route].add(
                        weight[:, None, None] * multipoles.quadrupoles
                    )
        charge_modes = jnp.fft.fftn(charge_grid)
        dipole_modes = jnp.fft.fftn(dipole_grid, axes=(0, 1, 2))
        quadrupole_modes = jnp.fft.fftn(quadrupole_grid, axes=(0, 1, 2))
        integer_axes = tuple(jnp.fft.fftfreq(size) * size for size in self.grid_shape)
        mode_components = jnp.meshgrid(*integer_axes, indexing="ij")
        modes = jnp.stack(mode_components, axis=-1)
        wave = 2.0 * jnp.pi * contract("...i,ji->...j", modes, inverse)
        squared = jnp.sum(wave * wave, axis=-1)
        dipole_structure = 1.0j * jnp.sum(wave * dipole_modes, axis=-1)
        quadrupole_structure = -0.5 * contract(
            "...i,...ij,...j->...", wave, quadrupole_modes, wave
        )
        window = jnp.prod(
            jnp.stack(
                tuple(
                    jnp.sinc(mode_components[axis] / self.grid_shape[axis]) ** 2
                    for axis in range(3)
                ),
                axis=-1,
            ),
            axis=-1,
        )
        structure = (charge_modes + dipole_structure + quadrupole_structure) / jnp.where(
            jnp.abs(window) > 0.0, window, 1.0
        )
        safe_squared = jnp.where(squared > 0.0, squared, 1.0)
        kernel = jnp.where(
            squared > 0.0,
            jnp.exp(-safe_squared / (4.0 * self.alpha**2)) / safe_squared,
            0.0,
        )
        volume = jnp.abs(determinant)
        energy = (
            2.0
            * jnp.pi
            * coulomb_constant
            / volume
            * jnp.sum(kernel * jnp.real(structure * jnp.conj(structure)))
        )
        successful = (
            site_state.successful
            & jnp.isfinite(volume)
            & (volume > 0.0)
            & jnp.isfinite(energy)
        )
        return jnp.where(successful, energy, jnp.nan)


class ImplicitSolventPlan(StrictModule, NonTrainableState):
    model: str = eqx.field(static=True)
    solvent_dielectric: float = eqx.field(static=True)
    solute_dielectric: float = eqx.field(static=True)
    surface_tension: float = eqx.field(static=True)
    kirkwood_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: str = "gb",
        /,
        *,
        solvent_dielectric: float = 78.5,
        solute_dielectric: float = 1.0,
        surface_tension: float = 0.005,
        kirkwood_factor: float = 2.455,
    ):
        if (
            model not in ("gb", "gk")
            or min(solvent_dielectric, solute_dielectric, kirkwood_factor) <= 0
            or not np.isfinite(
                [solvent_dielectric, solute_dielectric, surface_tension, kirkwood_factor]
            ).all()
            or surface_tension < 0
        ):
            raise ValueError("Implicit-solvent parameters are invalid.")
        (
            self.model,
            self.solvent_dielectric,
            self.solute_dielectric,
            self.surface_tension,
            self.kirkwood_factor,
        ) = (
            model,
            float(solvent_dielectric),
            float(solute_dielectric),
            float(surface_tension),
            float(kirkwood_factor),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "implicit-solvent",
                "model": model,
                "solvent_dielectric": self.solvent_dielectric,
                "solute_dielectric": self.solute_dielectric,
                "surface_tension": self.surface_tension,
                "kirkwood_factor": self.kirkwood_factor,
            }
        )

    def energy(
        self,
        positions: ArrayLike,
        charges: ArrayLike,
        radii: ArrayLike,
        coulomb_constant: float,
        /,
    ):
        coordinate, charge, radius = (
            jnp.asarray(positions),
            jnp.asarray(charges),
            jnp.asarray(radii),
        )
        count = coordinate.shape[0]
        if (
            coordinate.shape != (count, 3)
            or charge.shape != (count,)
            or radius.shape != (count,)
            or not np.isfinite(coulomb_constant)
            or float(coulomb_constant) <= 0.0
        ):
            raise ValueError("Implicit-solvent arrays or Coulomb constant are invalid.")
        valid = (
            jnp.all(jnp.isfinite(coordinate))
            & jnp.all(jnp.isfinite(charge))
            & jnp.all(jnp.isfinite(radius) & (radius > 0.0))
        )
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        displacement = coordinate[:, None, :] - coordinate[None, :, :]
        distance2 = jnp.sum(displacement**2, axis=-1)
        radius_product = safe_radius[:, None] * safe_radius[None, :]
        denominator = 4.0 if self.model == "gb" else self.kirkwood_factor
        effective_distance = jnp.sqrt(
            distance2
            + radius_product * jnp.exp(-distance2 / (denominator * radius_product))
        )
        dielectric_factor = 1.0 / self.solvent_dielectric - 1.0 / self.solute_dielectric
        polar = (
            0.5
            * coulomb_constant
            * dielectric_factor
            * jnp.sum(charge[:, None] * charge[None, :] / effective_distance)
        )
        area = 4.0 * jnp.pi * jnp.sum(safe_radius**2)
        energy = polar + self.surface_tension * area
        return jnp.where(valid & jnp.isfinite(energy), energy, jnp.nan)


__all__ = [
    "ImplicitSolventPlan",
    "MultipolePMEPlan",
    "PermanentMultipoleSiteData",
    "PolarizationEvaluation",
    "PolarizationPlan",
    "PolarizationState",
    "evaluate_polarization",
    "implicit_polarization_jvp",
    "polarization_energy",
    "solve_induced_dipoles",
]
