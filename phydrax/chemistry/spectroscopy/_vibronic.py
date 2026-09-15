#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Duschinsky rotation, multidimensional Franck--Condon factors, and HT emission."""

from __future__ import annotations

from math import factorial, isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from numpy.polynomial.hermite import hermgauss
from scipy.special import eval_hermite

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure
from ..vibration import VibrationalAnalysisResult


_FINE_STRUCTURE = 7.2973525643e-3
_ATOMIC_TIME_SECONDS = 2.4188843265864e-17


class DuschinskyResult(StrictModule, NonTrainableState):
    rotation: Array
    displacement: Array
    orthogonality_residual: Array
    subspace_residual: Array
    successful: Array
    initial_vibration_id: str = eqx.field(static=True)
    final_vibration_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        rotation: ArrayLike,
        displacement: ArrayLike,
        orthogonality_residual: ArrayLike,
        subspace_residual: ArrayLike,
        successful: ArrayLike,
        initial_vibration_id: str,
        final_vibration_id: str,
        /,
    ):
        matrix = jnp.asarray(rotation)
        shift = jnp.asarray(displacement, dtype=matrix.dtype)
        count = matrix.shape[0] if matrix.ndim == 2 else -1
        if matrix.shape != (count, count) or shift.shape != (count,):
            raise ValueError(
                "Duschinsky rotation and displacement must share mode space."
            )
        self.rotation = matrix
        self.displacement = shift
        self.orthogonality_residual = jnp.asarray(
            orthogonality_residual, dtype=matrix.dtype
        ).reshape(())
        self.subspace_residual = jnp.asarray(
            subspace_residual, dtype=matrix.dtype
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.initial_vibration_id = str(initial_vibration_id)
        self.final_vibration_id = str(final_vibration_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "duschinsky-result",
                "initial_vibration": self.initial_vibration_id,
                "final_vibration": self.final_vibration_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "rotation": np.asarray(matrix),
                        "displacement": np.asarray(shift),
                        "orthogonality_residual": np.asarray(self.orthogonality_residual),
                        "subspace_residual": np.asarray(self.subspace_residual),
                    }
                ),
            }
        )


def duschinsky_analysis(
    initial_structure: AtomicStructure,
    initial_vibration: VibrationalAnalysisResult,
    final_structure: AtomicStructure,
    final_vibration: VibrationalAnalysisResult,
    /,
    *,
    tolerance: float = 1.0e-7,
) -> DuschinskyResult:
    if not isinstance(initial_structure, AtomicStructure) or not isinstance(
        final_structure, AtomicStructure
    ):
        raise TypeError("Duschinsky analysis requires two atomic structures.")
    if not isinstance(initial_vibration, VibrationalAnalysisResult) or not isinstance(
        final_vibration, VibrationalAnalysisResult
    ):
        raise TypeError("Duschinsky analysis requires two vibrational results.")
    initial_modes_full = np.asarray(initial_vibration.normal_modes)
    final_modes_full = np.asarray(final_vibration.normal_modes)
    initial_masses_full = np.asarray(initial_structure.masses)
    final_masses_full = np.asarray(final_structure.masses)
    initial_active = np.asarray(initial_structure.active_mask, dtype=bool)
    final_active = np.asarray(final_structure.active_mask, dtype=bool)
    if (
        initial_modes_full.shape != final_modes_full.shape
        or initial_masses_full.shape != final_masses_full.shape
        or not np.array_equal(initial_active, final_active)
        or not np.array_equal(
            np.asarray(initial_structure.particle_ids),
            np.asarray(final_structure.particle_ids),
        )
        or not np.allclose(
            initial_masses_full[initial_active],
            final_masses_full[initial_active],
            rtol=0.0,
            atol=0.0,
        )
        or initial_structure.positions.shape != final_structure.positions.shape
        or initial_structure.scale.scale_id != final_structure.scale.scale_id
    ):
        raise ValueError(
            "Duschinsky states must share IDs, active atoms, masses, units, and modes."
        )
    initial_modes = initial_modes_full[initial_active]
    final_modes = final_modes_full[initial_active]
    initial_masses = initial_masses_full[initial_active]
    final_masses = final_masses_full[initial_active]
    mode_count = initial_modes.shape[-1]
    initial_mass_weighted = (
        np.sqrt(initial_masses)[:, None, None]
        * initial_modes
        / np.sqrt(np.asarray(initial_vibration.reduced_masses))[None, None, :]
    ).reshape((-1, mode_count))
    final_mass_weighted = (
        np.sqrt(final_masses)[:, None, None]
        * final_modes
        / np.sqrt(np.asarray(final_vibration.reduced_masses))[None, None, :]
    ).reshape((-1, mode_count))
    rotation = final_mass_weighted.T @ initial_mass_weighted
    cartesian_shift = (
        np.sqrt(initial_masses)[:, None]
        * (
            np.asarray(initial_structure.positions)[initial_active]
            - np.asarray(final_structure.positions)[initial_active]
        )
    ).reshape((-1,))
    displacement = final_mass_weighted.T @ cartesian_shift
    identity = np.eye(mode_count, dtype=rotation.dtype)
    orthogonality = np.max(np.abs(rotation.T @ rotation - identity), initial=0.0)
    initial_projector = initial_mass_weighted @ initial_mass_weighted.T
    final_projector = final_mass_weighted @ final_mass_weighted.T
    subspace = np.max(np.abs(initial_projector - final_projector), initial=0.0)
    successful = (
        bool(initial_vibration.successful)
        and bool(final_vibration.successful)
        and np.all(np.isfinite(rotation))
        and np.all(np.isfinite(displacement))
        and orthogonality <= float(tolerance)
        and subspace <= float(tolerance)
    )
    return DuschinskyResult(
        rotation,
        displacement,
        orthogonality,
        subspace,
        successful,
        initial_vibration.result_id,
        final_vibration.result_id,
    )


class FranckCondonResult(StrictModule, NonTrainableState):
    final_quanta: Array
    overlap_amplitudes: Array
    factors: Array
    transition_dipoles: Array | None
    photon_energies: Array | None
    emission_rates: Array | None
    quadrature_points: int = eqx.field(static=True)
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        final_quanta: ArrayLike,
        overlap_amplitudes: ArrayLike,
        factors: ArrayLike,
        quadrature_points: int,
        successful: ArrayLike,
        plan_id: str,
        /,
        *,
        transition_dipoles: ArrayLike | None = None,
        photon_energies: ArrayLike | None = None,
        emission_rates: ArrayLike | None = None,
    ):
        quanta = jnp.asarray(final_quanta, dtype=jnp.int32)
        overlaps = jnp.asarray(overlap_amplitudes)
        factors_ = jnp.asarray(factors, dtype=overlaps.real.dtype)
        states = quanta.shape[0] if quanta.ndim == 2 else -1
        dipoles = (
            None
            if transition_dipoles is None
            else jnp.asarray(transition_dipoles, dtype=overlaps.dtype)
        )
        photons = (
            None
            if photon_energies is None
            else jnp.asarray(photon_energies, dtype=factors_.dtype)
        )
        rates = (
            None
            if emission_rates is None
            else jnp.asarray(emission_rates, dtype=factors_.dtype)
        )
        if overlaps.shape != (states,) or factors_.shape != (states,):
            raise ValueError("Franck--Condon states, overlaps, and factors must align.")
        if dipoles is not None and dipoles.shape != (states, 3):
            raise ValueError("Herzberg--Teller dipoles must have shape (state, 3).")
        if (photons is None) != (rates is None):
            raise ValueError("Photon energies and emission rates must align by state.")
        if photons is not None:
            if rates is None:
                raise RuntimeError("Emission-rate optional presence invariant failed.")
            if photons.shape != (states,) or rates.shape != (states,):
                raise ValueError(
                    "Photon energies and emission rates must align by state."
                )
        self.final_quanta = quanta
        self.overlap_amplitudes = overlaps
        self.factors = factors_
        self.transition_dipoles = dipoles
        self.photon_energies = photons
        self.emission_rates = rates
        self.quadrature_points = int(quadrature_points)
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "franck-condon-result",
                "plan": self.plan_id,
                "quadrature_points": self.quadrature_points,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "quanta": np.asarray(quanta),
                        "overlaps": np.asarray(overlaps),
                        "factors": np.asarray(factors_),
                        "transition_dipoles": None
                        if dipoles is None
                        else np.asarray(dipoles),
                        "photon_energies": None
                        if photons is None
                        else np.asarray(photons),
                        "emission_rates": None if rates is None else np.asarray(rates),
                    }
                ),
            }
        )


class DuschinskyFranckCondonPlan(StrictModule, NonTrainableState):
    initial_angular_frequencies: Array
    final_angular_frequencies: Array
    duschinsky: DuschinskyResult
    hbar: float = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    maximum_quadrature_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_angular_frequencies: ArrayLike,
        final_angular_frequencies: ArrayLike,
        duschinsky: DuschinskyResult,
        hbar: float,
        /,
        *,
        quadrature_order: int = 12,
        maximum_quadrature_points: int = 2_000_000,
    ):
        if not isinstance(duschinsky, DuschinskyResult):
            raise TypeError("duschinsky must be DuschinskyResult.")
        initial = jnp.asarray(initial_angular_frequencies)
        final = jnp.asarray(final_angular_frequencies, dtype=initial.dtype)
        hbar_ = float(hbar)
        order = int(quadrature_order)
        maximum = int(maximum_quadrature_points)
        if (
            initial.ndim != 1
            or final.shape != initial.shape
            or duschinsky.rotation.shape != (initial.size, initial.size)
            or bool(jnp.any(initial <= 0.0))
            or bool(jnp.any(final <= 0.0))
            or not isfinite(hbar_)
            or hbar_ <= 0.0
            or order < 2
            or maximum < 1
        ):
            raise ValueError(
                "Franck--Condon frequencies, units, or quadrature are invalid."
            )
        point_count = order ** int(initial.size)
        if point_count > maximum:
            raise ValueError(
                "Franck--Condon tensor quadrature exceeds maximum_quadrature_points."
            )
        self.initial_angular_frequencies = initial
        self.final_angular_frequencies = final
        self.duschinsky = duschinsky
        self.hbar = hbar_
        self.quadrature_order = order
        self.maximum_quadrature_points = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "duschinsky-franck-condon-plan",
                "duschinsky": duschinsky.result_id,
                "hbar": hbar_,
                "quadrature_order": order,
                "maximum_quadrature_points": maximum,
                "arrays": array_tree_fingerprint(
                    {
                        "initial_frequencies": np.asarray(initial),
                        "final_frequencies": np.asarray(final),
                    }
                ),
            }
        )

    def evaluate(
        self,
        final_quanta: ArrayLike,
        /,
        *,
        condon_dipole: ArrayLike | None = None,
        herzberg_teller_derivatives: ArrayLike | None = None,
        zero_zero_energy_hartree: float | None = None,
    ) -> FranckCondonResult:
        quanta = np.asarray(final_quanta, dtype=int)
        mode_count = int(self.initial_angular_frequencies.size)
        if quanta.ndim != 2 or quanta.shape[1] != mode_count or np.any(quanta < 0):
            raise ValueError("Final vibrational quanta must have shape (state, mode).")
        condon = None if condon_dipole is None else np.asarray(condon_dipole)
        derivatives = (
            None
            if herzberg_teller_derivatives is None
            else np.asarray(herzberg_teller_derivatives)
        )
        if (condon is None) != (derivatives is None):
            raise ValueError(
                "Condon dipole and Herzberg--Teller derivatives must be supplied together."
            )
        if condon is not None:
            if derivatives is None:
                raise RuntimeError("Herzberg--Teller optional presence invariant failed.")
            if condon.shape != (3,) or derivatives.shape != (mode_count, 3):
                raise ValueError(
                    "Condon/Herzberg--Teller arrays must have shapes (3,) and (mode,3)."
                )
        initial_frequency = np.asarray(self.initial_angular_frequencies) / self.hbar
        final_frequency = np.asarray(self.final_angular_frequencies) / self.hbar
        rotation = np.asarray(self.duschinsky.rotation)
        shift = np.asarray(self.duschinsky.displacement)
        initial_quadratic = rotation @ np.diag(initial_frequency) @ rotation.T
        quadratic = np.diag(final_frequency) + initial_quadratic
        eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("Franck--Condon combined Gaussian is not positive definite.")
        mean = np.linalg.solve(quadratic, initial_quadratic @ shift)
        constant = shift @ initial_quadratic @ shift - mean @ quadratic @ mean
        nodes, weights = hermgauss(self.quadrature_order)
        meshes = np.meshgrid(*([nodes] * mode_count), indexing="ij")
        points = np.stack(meshes, axis=-1).reshape((-1, mode_count))
        weight_meshes = np.meshgrid(*([weights] * mode_count), indexing="ij")
        point_weights = np.prod(np.stack(weight_meshes, axis=-1), axis=-1).reshape((-1,))
        inverse_root = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
        final_coordinates = mean[None, :] + np.sqrt(2.0) * points @ inverse_root.T
        determinant_factor = (
            np.exp(-0.5 * constant)
            * 2.0 ** (0.5 * mode_count)
            / np.sqrt(np.prod(eigenvalues))
        )
        ground_normalization = np.prod(
            (initial_frequency * final_frequency) ** 0.25 / np.sqrt(np.pi)
        )
        overlaps = []
        dipole_amplitudes = []
        for occupation in quanta:
            polynomial = np.ones((points.shape[0],), dtype=final_coordinates.dtype)
            normalization = ground_normalization
            for mode, quantum in enumerate(occupation):
                polynomial *= eval_hermite(
                    int(quantum),
                    np.sqrt(final_frequency[mode]) * final_coordinates[:, mode],
                )
                normalization /= np.sqrt(2.0 ** int(quantum) * factorial(int(quantum)))
            weighted_polynomial = point_weights * polynomial
            overlap = determinant_factor * normalization * np.sum(weighted_polynomial)
            overlaps.append(overlap)
            if condon is not None:
                dipole_function = condon[None, :] + final_coordinates @ derivatives
                dipole_amplitudes.append(
                    determinant_factor
                    * normalization
                    * np.sum(weighted_polynomial[:, None] * dipole_function, axis=0)
                )
        overlap_values = np.asarray(overlaps)
        factors = np.abs(overlap_values) ** 2
        dipoles = None if condon is None else np.asarray(dipole_amplitudes)
        photons = rates = None
        if zero_zero_energy_hartree is not None:
            if dipoles is None:
                raise ValueError(
                    "Emission rates require Condon/Herzberg--Teller dipoles."
                )
            zero_zero = float(zero_zero_energy_hartree)
            if not isfinite(zero_zero) or zero_zero <= 0.0:
                raise ValueError("zero_zero_energy_hartree must be positive finite.")
            final_vibrational_energy = quanta @ (
                self.hbar * np.asarray(self.final_angular_frequencies)
            )
            photons = zero_zero - final_vibrational_energy
            rates = (
                (4.0 / 3.0)
                * _FINE_STRUCTURE**3
                * np.maximum(photons, 0.0) ** 3
                * np.sum(np.abs(dipoles) ** 2, axis=1)
                / _ATOMIC_TIME_SECONDS
            )
        successful = (
            bool(self.duschinsky.successful)
            and np.all(np.isfinite(overlap_values))
            and np.all(np.isfinite(factors))
            and (rates is None or np.all(np.isfinite(rates)))
        )
        return FranckCondonResult(
            quanta,
            overlap_values,
            factors,
            points.shape[0],
            successful,
            self.plan_id,
            transition_dipoles=dipoles,
            photon_energies=photons,
            emission_rates=rates,
        )


__all__ = [
    "DuschinskyFranckCondonPlan",
    "DuschinskyResult",
    "FranckCondonResult",
    "duschinsky_analysis",
]
