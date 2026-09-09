#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Equilibrium-paired Cartesian fluxes with conservative gravitational work."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._gas_dynamics import HomogeneousMixtureEulerSystem
from ._riemann import RusanovFluxPlan
from ._structured import FiniteVolumeDiscretization


def _minmod(left: Array, right: Array) -> Array:
    return jnp.where(
        left * right > 0.0,
        jnp.sign(left) * jnp.minimum(jnp.abs(left), jnp.abs(right)),
        0.0,
    )


class AtmosphericBalanceEvaluation(StrictModule):
    residual: Array
    normal_fluxes: tuple[Array, ...]
    mass_fluxes: tuple[Array, ...]
    boundary_outward_flux: Array
    source_integral: Array
    conservation_defect: Array
    maximum_rate: Array
    successful: Array


class PreparedAtmosphericBalance(StrictModule, NonTrainableState):
    """MUSCL on conserved departures, not on the stratified total state.

    The reference flux is evaluated by the *same* numerical flux as the actual
    faces. Subtraction occurs before divergence; a reference state therefore
    has identically zero residual, independent of EOS inversion tolerance.
    The remaining vertical force is -g (rho-rho_reference). Gas energy uses
    -div(F_E + Phi_face F_mass) + Phi_cell div(F_mass), with exactly the
    species-summed continuity flux. This conserves gas plus discrete potential
    energy, including open-boundary transport, without a separate rho*u*g rule.
    """

    system: HomogeneousMixtureEulerSystem
    discretization: FiniteVolumeDiscretization
    reference: Array
    face_reference: tuple[Array, ...]
    reference_fluxes: tuple[Array, ...]
    potential: Array
    face_potential: tuple[Array, ...]
    flux: RusanovFluxPlan
    boundaries: tuple[tuple[str, str], ...] = eqx.field(static=True)
    prescribed: tuple[tuple[Array | None, Array | None], ...]
    widths: tuple[float, ...] = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    order: int = eqx.field(static=True)
    balance_id: str = eqx.field(static=True)

    def __init__(
        self,
        system,
        discretization,
        reference,
        face_reference,
        *,
        gravity,
        boundaries,
        prescribed,
        order=2,
    ):
        if not isinstance(system, HomogeneousMixtureEulerSystem):
            raise TypeError("Atmospheric balance requires HomogeneousMixtureEulerSystem.")
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("Atmospheric balance requires fixed Cartesian FV geometry.")
        if order not in (1, 2):
            raise ValueError("Atmospheric reconstruction order must be one or two.")
        widths = []
        for axis in discretization.grid.structured_axes:
            values = np.asarray(axis.interval_widths)
            if values.size < 2 or not np.allclose(
                values, values[0], rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    "Atmospheric balance requires at least two uniform cells per axis."
                )
            widths.append(float(values[0]))
        self.system = system
        self.discretization = discretization
        self.reference = jnp.asarray(reference)
        self.face_reference = tuple(jnp.asarray(face) for face in face_reference)
        self.flux = RusanovFluxPlan()
        self.reference_fluxes = tuple(
            self.flux.face_flux(system, face, face, axis).normal_flux
            for axis, face in enumerate(self.face_reference)
        )
        self.gravity = float(gravity)
        self.boundaries = boundaries
        self.prescribed = prescribed
        self.widths = tuple(widths)
        self.order = int(order)
        self.potential = self.gravity * discretization.cell_centers[..., -1]
        potentials = []
        for axis, points in enumerate(discretization.face_centers):
            value = self.gravity * points[..., -1]
            if boundaries[axis][0] == "periodic":
                value = jnp.concatenate(
                    (value, jnp.take(value, jnp.asarray([0]), axis=axis)), axis=axis
                )
            potentials.append(value)
        self.face_potential = tuple(potentials)
        self.balance_id = canonical_fingerprint(
            {
                "kind": "cartesian-atmospheric-equilibrium-pair",
                "system": system.system_id,
                "geometry": discretization.prepared_id,
                "reference": array_tree_fingerprint(self.reference),
                "faces": array_tree_fingerprint(self.face_reference),
                "gravity": self.gravity,
                "boundaries": boundaries,
                "prescribed": array_tree_fingerprint(prescribed),
                "order": self.order,
                "flux": self.flux.flux_id,
            }
        )

    def _faces(self, state: Array, axis: int) -> tuple[Array, Array]:
        delta = jnp.moveaxis(state - self.reference, axis, 0)
        reference = jnp.moveaxis(self.face_reference[axis], axis, 0)
        lower, upper = self.boundaries[axis]
        if lower == "periodic":
            ghost_lower, ghost_upper = delta[-1], delta[0]
        else:
            ghost_lower, ghost_upper = delta[0], delta[-1]
            normal = self.system.species_count + axis
            if lower == "closed":
                ghost_lower = ghost_lower.at[..., normal].multiply(-1.0)
            else:
                ghost_lower = self.prescribed[axis][0] - reference[0]
            if upper == "closed":
                ghost_upper = ghost_upper.at[..., normal].multiply(-1.0)
            else:
                ghost_upper = self.prescribed[axis][1] - reference[-1]
        padded = jnp.concatenate((ghost_lower[None], delta, ghost_upper[None]), axis=0)
        slope = jnp.zeros_like(padded)
        if self.order == 2:
            # Monotonized-central limiter; slopes are cell increments on this
            # uniform Cartesian route. Boundary ghosts deliberately have zero slope.
            backward, forward = padded[1:-1] - padded[:-2], padded[2:] - padded[1:-1]
            limited = _minmod(
                0.5 * (backward + forward), _minmod(2.0 * backward, 2.0 * forward)
            )
            slope = slope.at[1:-1].set(limited)
            if lower == "periodic":
                slope = slope.at[0].set(limited[-1]).at[-1].set(limited[0])
        left = reference + padded[:-1] + 0.5 * slope[:-1]
        right = reference + padded[1:] - 0.5 * slope[1:]
        # A closed wall is an exact reflected *face* pair, not reflected cell
        # averages; otherwise MUSCL can leak mass and gravitational energy.
        normal = self.system.species_count + axis
        if lower == "closed":
            left = left.at[0].set(right[0].at[..., normal].multiply(-1.0))
        if upper == "closed":
            right = right.at[-1].set(left[-1].at[..., normal].multiply(-1.0))
        return jnp.moveaxis(left, 0, axis), jnp.moveaxis(right, 0, axis)

    def evaluate(self, state: Array) -> AtmosphericBalanceEvaluation:
        value = jnp.asarray(state)
        if value.shape != self.reference.shape:
            raise ValueError("Atmospheric state shape must match prepared geometry.")
        system, geometry = self.system, self.discretization
        residual = jnp.zeros_like(value)
        boundary = jnp.zeros((system.component_count,), dtype=value.dtype)
        source = jnp.zeros_like(boundary)
        rate = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        valid = jnp.all(system.admissible(value))
        fluxes, mass_fluxes = [], []
        for axis, width in enumerate(self.widths):
            left, right = self._faces(value, axis)
            valid = (
                valid
                & jnp.all(system.admissible(left))
                & jnp.all(system.admissible(right))
            )
            numerical = self.flux.face_flux(system, left, right, axis)
            flux = numerical.normal_flux
            paired = flux - self.reference_fluxes[axis]
            mass_flux = jnp.sum(flux[..., : system.species_count], axis=-1)
            # One shared flux is used for both continuity and potential transport.
            total_energy_flux = paired[..., -1] + self.face_potential[axis] * mass_flux
            paired = paired.at[..., -1].set(total_energy_flux)
            residual = residual - jnp.diff(paired, axis=axis) / width
            residual = residual.at[..., -1].add(
                self.potential * jnp.diff(mass_flux, axis=axis) / width
            )
            speed = jnp.moveaxis(numerical.max_speed, axis, 0)
            rate = (
                rate + jnp.moveaxis(jnp.maximum(speed[:-1], speed[1:]), 0, axis) / width
            )
            fluxes.append(flux)
            mass_fluxes.append(mass_flux)
            if self.boundaries[axis][0] != "periodic":
                budget_flux = flux.at[..., -1].set(
                    flux[..., -1] + self.face_potential[axis] * mass_flux
                )
                face = jnp.moveaxis(budget_flux, axis, 0)
                measure = jnp.moveaxis(geometry.face_measures[axis], axis, 0)
                boundary = boundary + jnp.sum(
                    (
                        face[-1] * measure[-1][..., None]
                        - face[0] * measure[0][..., None]
                    ).reshape((-1, system.component_count)),
                    axis=0,
                )
        density_departure = system.density(value) - system.density(self.reference)
        vertical = system.species_count + system.dimension - 1
        residual = residual.at[..., vertical].add(-self.gravity * density_departure)
        reference_force = (
            jnp.diff(self.reference_fluxes[-1][..., vertical], axis=system.dimension - 1)
            / self.widths[-1]
        )
        source = source.at[vertical].set(
            jnp.sum(
                (reference_force - self.gravity * density_departure)
                * geometry.cell_volumes
            )
        )
        budget_rate = residual.at[..., -1].add(
            self.potential * jnp.sum(residual[..., : system.species_count], axis=-1)
        )
        integral = jnp.sum(
            (budget_rate * geometry.cell_volumes[..., None]).reshape(
                (-1, system.component_count)
            ),
            axis=0,
        )
        defect = integral + boundary - source
        return AtmosphericBalanceEvaluation(
            residual,
            tuple(fluxes),
            tuple(mass_fluxes),
            boundary,
            source,
            defect,
            jnp.max(rate),
            valid & jnp.all(jnp.isfinite(residual)),
        )


__all__ = ["AtmosphericBalanceEvaluation", "PreparedAtmosphericBalance"]
