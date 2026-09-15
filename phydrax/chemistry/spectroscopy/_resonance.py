#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frequency-dependent Kramers--Heisenberg--Dirac Raman response."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract


class ResonanceRamanResult(StrictModule, NonTrainableState):
    dynamic_polarizability: Array
    polarizability_derivatives: Array
    isotropic_invariants: Array
    anisotropy_invariants: Array
    activities: Array
    relative_intensities: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamic_polarizability: ArrayLike,
        polarizability_derivatives: ArrayLike,
        isotropic_invariants: ArrayLike,
        anisotropy_invariants: ArrayLike,
        activities: ArrayLike,
        relative_intensities: ArrayLike,
        successful: ArrayLike,
        plan_id: str,
        /,
    ):
        polarizability = jnp.asarray(dynamic_polarizability)
        derivative = jnp.asarray(polarizability_derivatives, dtype=polarizability.dtype)
        isotropic = jnp.asarray(isotropic_invariants, dtype=polarizability.real.dtype)
        anisotropy = jnp.asarray(anisotropy_invariants, dtype=polarizability.real.dtype)
        activities = jnp.asarray(activities, dtype=polarizability.real.dtype)
        intensities = jnp.asarray(relative_intensities, dtype=polarizability.real.dtype)
        modes = derivative.shape[0] if derivative.ndim == 3 else -1
        if polarizability.shape != (3, 3) or derivative.shape != (modes, 3, 3):
            raise ValueError("Dynamic Raman tensors must have Cartesian rank two.")
        if any(
            value.shape != (modes,)
            for value in (isotropic, anisotropy, activities, intensities)
        ):
            raise ValueError("Dynamic Raman invariants must align by mode.")
        self.dynamic_polarizability = polarizability
        self.polarizability_derivatives = derivative
        self.isotropic_invariants = isotropic
        self.anisotropy_invariants = anisotropy
        self.activities = activities
        self.relative_intensities = intensities
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "resonance-raman-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "dynamic_polarizability": np.asarray(polarizability),
                        "polarizability_derivatives": np.asarray(derivative),
                        "isotropic": np.asarray(isotropic),
                        "anisotropy": np.asarray(anisotropy),
                        "activities": np.asarray(activities),
                        "intensities": np.asarray(intensities),
                    }
                ),
            }
        )


class ResonanceRamanPlan(StrictModule, NonTrainableState):
    excitation_energies: Array
    transition_dipoles: Array
    transition_dipole_derivatives: Array
    excitation_energy_derivatives: Array
    mode_energies: Array
    laser_energy: float = eqx.field(static=True)
    damping: Array
    include_antiresonant: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        excitation_energies: ArrayLike,
        transition_dipoles: ArrayLike,
        transition_dipole_derivatives: ArrayLike,
        excitation_energy_derivatives: ArrayLike,
        mode_energies: ArrayLike,
        laser_energy: float,
        damping: ArrayLike,
        /,
        *,
        include_antiresonant: bool = True,
    ):
        excitation = jnp.asarray(excitation_energies)
        dipoles = jnp.asarray(transition_dipoles, dtype=excitation.dtype)
        dipole_derivatives = jnp.asarray(
            transition_dipole_derivatives, dtype=excitation.dtype
        )
        energy_derivatives = jnp.asarray(
            excitation_energy_derivatives, dtype=excitation.real.dtype
        )
        modes = jnp.asarray(mode_energies, dtype=excitation.real.dtype)
        damping_ = jnp.asarray(damping, dtype=excitation.real.dtype)
        laser = float(laser_energy)
        roots = int(excitation.size)
        mode_count = int(modes.size)
        if (
            dipoles.shape != (roots, 3)
            or dipole_derivatives.shape != (roots, mode_count, 3)
            or energy_derivatives.shape != (roots, mode_count)
            or damping_.shape not in ((), (roots,))
            or bool(jnp.any(excitation <= 0.0))
            or bool(jnp.any(modes <= 0.0))
            or bool(jnp.any(damping_ <= 0.0))
            or not isfinite(laser)
            or laser <= 0.0
        ):
            raise ValueError(
                "Resonance Raman energies, dipoles, modes, or damping are invalid."
            )
        if damping_.shape == ():
            damping_ = jnp.full((roots,), damping_)
        self.excitation_energies = excitation
        self.transition_dipoles = dipoles
        self.transition_dipole_derivatives = dipole_derivatives
        self.excitation_energy_derivatives = energy_derivatives
        self.mode_energies = modes
        self.laser_energy = laser
        self.damping = damping_
        self.include_antiresonant = bool(include_antiresonant)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "resonance-raman-plan",
                "laser_energy": laser,
                "include_antiresonant": self.include_antiresonant,
                "arrays": array_tree_fingerprint(
                    {
                        "excitation_energies": np.asarray(excitation),
                        "transition_dipoles": np.asarray(dipoles),
                        "transition_dipole_derivatives": np.asarray(dipole_derivatives),
                        "excitation_energy_derivatives": np.asarray(energy_derivatives),
                        "mode_energies": np.asarray(modes),
                        "damping": np.asarray(damping_),
                    }
                ),
            }
        )

    def evaluate(self, /) -> ResonanceRamanResult:
        excitation = self.excitation_energies
        dipoles = self.transition_dipoles
        damping = self.damping
        resonant = excitation - self.laser_energy - 1.0j * damping
        outer = contract("ea,eb->eab", dipoles, jnp.conj(dipoles))
        polarizability = jnp.sum(outer / resonant[:, None, None], axis=0)
        numerator_derivative = contract(
            "ema,eb->emab",
            self.transition_dipole_derivatives,
            jnp.conj(dipoles),
        ) + contract(
            "ea,emb->emab",
            dipoles,
            jnp.conj(self.transition_dipole_derivatives),
        )
        derivatives = jnp.sum(
            numerator_derivative / resonant[:, None, None, None]
            - outer[:, None, :, :]
            * self.excitation_energy_derivatives[:, :, None, None]
            / resonant[:, None, None, None] ** 2,
            axis=0,
        )
        if self.include_antiresonant:
            antiresonant = excitation + self.laser_energy + 1.0j * damping
            anti_outer = jnp.swapaxes(outer, 1, 2)
            anti_numerator_derivative = jnp.swapaxes(numerator_derivative, 2, 3)
            polarizability = polarizability + jnp.sum(
                anti_outer / antiresonant[:, None, None], axis=0
            )
            derivatives = derivatives + jnp.sum(
                anti_numerator_derivative / antiresonant[:, None, None, None]
                - anti_outer[:, None, :, :]
                * self.excitation_energy_derivatives[:, :, None, None]
                / antiresonant[:, None, None, None] ** 2,
                axis=0,
            )
        diagonal = jnp.diagonal(derivatives, axis1=1, axis2=2)
        isotropic_amplitude = jnp.trace(derivatives, axis1=1, axis2=2) / 3.0
        isotropic = jnp.abs(isotropic_amplitude) ** 2
        symmetric = 0.5 * (derivatives + jnp.swapaxes(derivatives, 1, 2))
        anisotropy = 0.5 * (
            jnp.abs(diagonal[:, 0] - diagonal[:, 1]) ** 2
            + jnp.abs(diagonal[:, 1] - diagonal[:, 2]) ** 2
            + jnp.abs(diagonal[:, 2] - diagonal[:, 0]) ** 2
            + 6.0
            * (
                jnp.abs(symmetric[:, 0, 1]) ** 2
                + jnp.abs(symmetric[:, 1, 2]) ** 2
                + jnp.abs(symmetric[:, 2, 0]) ** 2
            )
        )
        activities = 45.0 * isotropic + 7.0 * anisotropy
        stokes_energy = self.laser_energy - self.mode_energies
        relative = jnp.where(
            stokes_energy > 0.0,
            activities * stokes_energy**4 / self.mode_energies,
            0.0,
        )
        relative = relative / jnp.maximum(
            jnp.max(relative), jnp.finfo(relative.dtype).tiny
        )
        successful = (
            jnp.all(stokes_energy > 0.0)
            & jnp.all(jnp.isfinite(polarizability))
            & jnp.all(jnp.isfinite(derivatives))
            & jnp.all(jnp.isfinite(relative))
        )
        return ResonanceRamanResult(
            polarizability,
            derivatives,
            isotropic,
            anisotropy,
            activities,
            relative,
            successful,
            self.plan_id,
        )


__all__ = ["ResonanceRamanPlan", "ResonanceRamanResult"]
