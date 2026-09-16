#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""DOS, metric PDOS, physical velocities, and raw Fermi-surface evidence."""

from __future__ import annotations

from itertools import product
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._reciprocal import ReciprocalMeshPlan
from ...ein import contract
from ...operators.periodic._family import PeriodicResourceError
from ...units import conversion_factor, derived_unit, JOULE, METER, SECOND, UnitDefinition
from ._orbital_model import PreparedPeriodicOrbitalPencil
from ._spectrum import PeriodicSpectrumResult


_HBAR_JOULE_SECOND = 1.054_571_817e-34
_VELOCITY_UNIT = derived_unit("m/s", ((METER, 1), (SECOND, -1)))


class PeriodicDensityOfStatesResult(StrictModule, NonTrainableState):
    energy_grid: Array
    density: Array
    state_count: Array
    normalization_residual: Array
    successful: Array
    energy_unit: UnitDefinition
    result_id: str = eqx.field(static=True)


class PeriodicDensityOfStatesPlan(StrictModule, NonTrainableState):
    spectrum: PeriodicSpectrumResult
    energy_grid: Array
    broadening: float = eqx.field(static=True)
    states_per_band: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spectrum: PeriodicSpectrumResult,
        energy_grid: ArrayLike,
        broadening: float,
        /,
        *,
        states_per_band: float = 1.0,
        maximum_kernel_entries: int = 8_000_000,
    ):
        if not isinstance(spectrum, PeriodicSpectrumResult):
            raise TypeError("spectrum must be PeriodicSpectrumResult.")
        grid = np.asarray(energy_grid)
        width = float(broadening)
        multiplicity = float(states_per_band)
        if (
            grid.ndim != 1
            or grid.size < 2
            or np.any(~np.isfinite(grid))
            or np.any(np.diff(grid) <= 0.0)
            or not isfinite(width)
            or width <= 0.0
            or not isfinite(multiplicity)
            or multiplicity <= 0.0
        ):
            raise ValueError("DOS grid, broadening, and band multiplicity are invalid.")
        entries = grid.size * spectrum.energies.size
        if entries > int(maximum_kernel_entries):
            raise PeriodicResourceError("DOS kernel exceeds maximum_kernel_entries.")
        self.spectrum = spectrum
        self.energy_grid = jnp.asarray(grid)
        self.broadening = width
        self.states_per_band = multiplicity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-density-of-states-plan",
                "spectrum": spectrum.result_id,
                "broadening": width,
                "states_per_band": multiplicity,
                "grid": array_tree_fingerprint(grid),
            }
        )

    def evaluate(self, /) -> PeriodicDensityOfStatesResult:
        scaled = (
            self.energy_grid[:, None, None] - self.spectrum.energies[None, :, :]
        ) / self.broadening
        kernel = jnp.exp(-0.5 * scaled * scaled) / (
            self.broadening * jnp.sqrt(2.0 * jnp.pi)
        )
        density = self.states_per_band * contract(
            "k,gkn->g", self.spectrum.weights, kernel, backend="jax"
        )
        count = jnp.trapezoid(density, self.energy_grid)
        expected = self.states_per_band * self.spectrum.energies.shape[1]
        residual = jnp.abs(count - expected)
        successful = jnp.all(jnp.isfinite(density)) & jnp.all(density >= 0.0)
        return PeriodicDensityOfStatesResult(
            self.energy_grid,
            density,
            count,
            residual,
            successful,
            self.spectrum.energy_unit,
            canonical_fingerprint(
                {
                    "kind": "periodic-density-of-states-result",
                    "plan": self.plan_id,
                    "arrays": array_tree_fingerprint(
                        {
                            "grid": np.asarray(self.energy_grid),
                            "density": np.asarray(density),
                            "state_count": np.asarray(count),
                        }
                    ),
                }
            ),
        )


class PeriodicProjectorGroups(StrictModule, NonTrainableState):
    """Named disjoint orbital groups for the generalized Mulliken metric route."""

    membership: Array
    names: tuple[str, ...] = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    metric_route: str = eqx.field(static=True)
    groups_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: tuple[str, ...],
        orbital_group_indices: ArrayLike,
        basis_id: str,
        /,
    ):
        names_ = tuple(str(value).strip() for value in names)
        indices = np.asarray(orbital_group_indices)
        basis = str(basis_id).strip()
        if (
            not names_
            or any(not value for value in names_)
            or len(set(names_)) != len(names_)
            or indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(names_))
            or not basis
        ):
            raise ValueError(
                "Projector group names, membership, or basis identity is invalid."
            )
        membership = np.equal(
            np.arange(len(names_))[:, None], indices.astype(np.int64)[None, :]
        ).astype(float)
        self.membership = jnp.asarray(membership)
        self.names = names_
        self.basis_id = basis
        self.metric_route = "generalized-mulliken-partition"
        self.groups_id = canonical_fingerprint(
            {
                "kind": "periodic-projector-groups",
                "names": list(names_),
                "basis": basis,
                "membership": array_tree_fingerprint(membership),
            }
        )


class PeriodicProjectedDOSResult(StrictModule, NonTrainableState):
    energy_grid: Array
    total_density: Array
    grouped_density: Array
    band_group_weights: Array
    partition_residual: Array
    successful: Array
    group_names: tuple[str, ...] = eqx.field(static=True)
    metric_route: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class PeriodicProjectedDOSPlan(StrictModule, NonTrainableState):
    pencil: PreparedPeriodicOrbitalPencil
    spectrum: PeriodicSpectrumResult
    groups: PeriodicProjectorGroups
    energy_grid: Array
    broadening: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        spectrum: PeriodicSpectrumResult,
        groups: PeriodicProjectorGroups,
        energy_grid: ArrayLike,
        broadening: float,
        /,
        *,
        maximum_kernel_entries: int = 8_000_000,
    ):
        if (
            not isinstance(pencil, PreparedPeriodicOrbitalPencil)
            or not isinstance(spectrum, PeriodicSpectrumResult)
            or not isinstance(groups, PeriodicProjectorGroups)
        ):
            raise TypeError(
                "PDOS requires a prepared pencil, spectrum, and projector groups."
            )
        if (
            spectrum.pencil_id != pencil.plan.pencil_id
            or groups.basis_id != pencil.plan.basis.basis_id
            or groups.membership.shape[1] != pencil.plan.basis.orbital_count
        ):
            raise ValueError("PDOS pencil, spectrum, and projector basis do not match.")
        grid = np.asarray(energy_grid)
        width = float(broadening)
        entries = grid.size * spectrum.energies.size * len(groups.names)
        if (
            grid.ndim != 1
            or grid.size < 2
            or np.any(np.diff(grid) <= 0.0)
            or np.any(~np.isfinite(grid))
            or not isfinite(width)
            or width <= 0.0
        ):
            raise ValueError("PDOS grid and broadening are invalid.")
        if entries > int(maximum_kernel_entries):
            raise PeriodicResourceError("PDOS kernel exceeds maximum_kernel_entries.")
        self.pencil = pencil
        self.spectrum = spectrum
        self.groups = groups
        self.energy_grid = jnp.asarray(grid)
        self.broadening = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-projected-dos-plan",
                "pencil": pencil.prepared_id,
                "spectrum": spectrum.result_id,
                "groups": groups.groups_id,
                "grid": array_tree_fingerprint(grid),
                "broadening": width,
            }
        )

    def evaluate(self, /) -> PeriodicProjectedDOSResult:
        pencil_evaluation = self.pencil.evaluate(self.spectrum.fractional_points)
        coefficients = self.spectrum.coefficients
        metric_vectors = pencil_evaluation.overlaps @ coefficients
        orbital_weights = jnp.real(jnp.conj(coefficients) * metric_vectors)
        group_weights = contract(
            "ga,kan->kng", self.groups.membership, orbital_weights, backend="jax"
        )
        scaled = (
            self.energy_grid[:, None, None] - self.spectrum.energies[None, :, :]
        ) / self.broadening
        kernel = jnp.exp(-0.5 * scaled * scaled) / (
            self.broadening * jnp.sqrt(2.0 * jnp.pi)
        )
        grouped = contract(
            "k,gkn,kna->ag", self.spectrum.weights, kernel, group_weights, backend="jax"
        )
        total = jnp.sum(grouped, axis=0)
        partition = jnp.max(jnp.abs(jnp.sum(group_weights, axis=-1) - 1.0), initial=0.0)
        successful = (
            pencil_evaluation.successful
            & jnp.all(jnp.isfinite(grouped))
            & (partition <= 1.0e-8)
        )
        return PeriodicProjectedDOSResult(
            self.energy_grid,
            total,
            grouped,
            group_weights,
            partition,
            successful,
            self.groups.names,
            self.groups.metric_route,
            canonical_fingerprint(
                {
                    "kind": "periodic-projected-dos-result",
                    "plan": self.plan_id,
                    "arrays": array_tree_fingerprint(
                        {
                            "total": np.asarray(total),
                            "grouped": np.asarray(grouped),
                            "weights": np.asarray(group_weights),
                        }
                    ),
                }
            ),
        )


class PeriodicVelocityResult(StrictModule, NonTrainableState):
    velocity_matrices: Array
    band_velocities: Array
    degeneracy_mask: Array
    hermiticity_residual: Array
    successful: Array
    velocity_unit: UnitDefinition
    derivative_basis: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class PeriodicVelocityPlan(StrictModule, NonTrainableState):
    """Physical Cartesian generalized velocities including the metric derivative."""

    pencil: PreparedPeriodicOrbitalPencil
    spectrum: PeriodicSpectrumResult
    degeneracy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        spectrum: PeriodicSpectrumResult,
        /,
        *,
        degeneracy_tolerance: float = 1.0e-8,
        maximum_matrix_entries: int = 8_000_000,
    ):
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil) or not isinstance(
            spectrum, PeriodicSpectrumResult
        ):
            raise TypeError("Velocity plan requires a prepared pencil and spectrum.")
        if spectrum.pencil_id != pencil.plan.pencil_id:
            raise ValueError("Velocity spectrum belongs to a different pencil.")
        tolerance = float(degeneracy_tolerance)
        entries = (
            spectrum.energies.shape[0]
            * spectrum.energies.shape[1] ** 2
            * pencil.plan.basis.cell.ambient_dimension
        )
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("degeneracy_tolerance must be finite and non-negative.")
        if entries > int(maximum_matrix_entries):
            raise PeriodicResourceError(
                "Velocity matrices exceed maximum_matrix_entries."
            )
        self.pencil = pencil
        self.spectrum = spectrum
        self.degeneracy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-velocity-plan",
                "pencil": pencil.prepared_id,
                "spectrum": spectrum.result_id,
                "degeneracy_tolerance": tolerance,
            }
        )

    def evaluate(self, /) -> PeriodicVelocityResult:
        evaluation = self.pencil.evaluate(self.spectrum.fractional_points)
        reciprocal_right_inverse = self.pencil.plan.basis.cell.vectors.T / (2.0 * jnp.pi)
        d_h = contract(
            "krij,dr->kdij",
            evaluation.d_hamiltonians,
            reciprocal_right_inverse,
            backend="jax",
        )
        d_s = contract(
            "krij,dr->kdij",
            evaluation.d_overlaps,
            reciprocal_right_inverse,
            backend="jax",
        )
        energies = self.spectrum.energies
        average_energy = 0.5 * (energies[:, :, None] + energies[:, None, :])
        coefficients = self.spectrum.coefficients
        h_matrix = contract(
            "kin,kdij,kjm->knmd",
            jnp.conj(coefficients),
            d_h,
            coefficients,
            backend="jax",
        )
        s_matrix = contract(
            "kin,kdij,kjm->knmd",
            jnp.conj(coefficients),
            d_s,
            coefficients,
            backend="jax",
        )
        unit_scale = (
            float(conversion_factor(self.spectrum.energy_unit, JOULE))
            * float(conversion_factor(self.pencil.plan.basis.length_unit, METER))
            / _HBAR_JOULE_SECOND
        )
        velocity = (h_matrix - average_energy[..., None] * s_matrix) * unit_scale
        diagonal = jnp.real(jnp.diagonal(velocity, axis1=1, axis2=2)).swapaxes(1, 2)
        degeneracy = (
            jnp.abs(energies[:, :, None] - energies[:, None, :])
            <= self.degeneracy_tolerance
        )
        hermiticity = jnp.max(
            jnp.abs(velocity - jnp.conj(jnp.swapaxes(velocity, 1, 2))), initial=0.0
        )
        successful = (
            evaluation.successful
            & jnp.all(jnp.isfinite(velocity))
            & (hermiticity <= 1.0e-7 * jnp.maximum(jnp.max(jnp.abs(velocity)), 1.0))
        )
        return PeriodicVelocityResult(
            velocity,
            diagonal,
            degeneracy,
            hermiticity,
            successful,
            _VELOCITY_UNIT,
            "cartesian-wavevector-m-per-s",
            self.spectrum.cell_id,
            self.spectrum.result_id,
            canonical_fingerprint(
                {
                    "kind": "periodic-velocity-result",
                    "plan": self.plan_id,
                    "arrays": array_tree_fingerprint(
                        {
                            "velocity": np.asarray(velocity),
                            "degeneracy": np.asarray(degeneracy),
                        }
                    ),
                }
            ),
        )


class FermiSurfaceEvidence(StrictModule, NonTrainableState):
    cell_indices: Array
    band_indices: Array
    corner_indices: Array
    corner_energies: Array
    unresolved_mask: Array
    lifshitz_mask: Array
    fermi_energy: Array
    complete: Array
    mesh_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def fermi_surface_evidence(
    mesh: ReciprocalMeshPlan,
    spectrum: PeriodicSpectrumResult,
    fermi_energy: float,
    /,
    *,
    unresolved_tolerance: float = 1.0e-8,
    maximum_crossing_cells: int = 1_000_000,
) -> FermiSurfaceEvidence:
    """Return every regular-mesh cell that brackets the Fermi level; no interpolation claim."""

    if not isinstance(mesh, ReciprocalMeshPlan) or not isinstance(
        spectrum, PeriodicSpectrumResult
    ):
        raise TypeError("Fermi-surface evidence requires a mesh and periodic spectrum.")
    if spectrum.support_id != mesh.mesh_id:
        raise ValueError("Fermi-surface spectrum belongs to a different reciprocal mesh.")
    fermi = float(fermi_energy)
    tolerance = float(unresolved_tolerance)
    if not isfinite(fermi) or not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Fermi energy and unresolved tolerance are invalid.")
    index_rows = np.asarray(mesh.mesh_indices)
    lookup = {tuple(int(v) for v in row): i for i, row in enumerate(index_rows)}
    energies = np.asarray(spectrum.energies)
    cell_rows: list[np.ndarray] = []
    bands: list[int] = []
    corners_out: list[list[int]] = []
    values_out: list[np.ndarray] = []
    unresolved: list[bool] = []
    lifshitz: list[bool] = []
    offsets = tuple(product((0, 1), repeat=mesh.rank))
    for coordinate in index_rows:
        corners = [
            lookup[
                tuple(
                    int((coordinate[axis] + offset[axis]) % mesh.mesh_shape[axis])
                    for axis in range(mesh.rank)
                )
            ]
            for offset in offsets
        ]
        corner_values = energies[corners]
        for band in range(energies.shape[1]):
            values = corner_values[:, band]
            if np.min(values) <= fermi <= np.max(values):
                cell_rows.append(coordinate.copy())
                bands.append(band)
                corners_out.append(corners)
                values_out.append(values)
                near = np.any(np.abs(values - fermi) <= tolerance)
                unresolved.append(bool(near or np.ptp(values) <= tolerance))
                other_crossing = np.any(
                    (np.min(corner_values, axis=0) <= fermi)
                    & (np.max(corner_values, axis=0) >= fermi)
                    & (np.arange(energies.shape[1]) != band)
                )
                lifshitz.append(bool(near or other_crossing))
    if len(cell_rows) > int(maximum_crossing_cells):
        raise PeriodicResourceError(
            "Fermi-surface extraction exceeds maximum_crossing_cells."
        )
    cell_array = np.asarray(cell_rows, dtype=np.int32).reshape((-1, mesh.rank))
    corner_array = np.asarray(corners_out, dtype=np.int32).reshape((-1, 2**mesh.rank))
    value_array = np.asarray(values_out, dtype=energies.dtype).reshape((-1, 2**mesh.rank))
    unresolved_array = np.asarray(unresolved, dtype=bool)
    lifshitz_array = np.asarray(lifshitz, dtype=bool)
    complete = not np.any(unresolved_array)
    return FermiSurfaceEvidence(
        jnp.asarray(cell_array),
        jnp.asarray(bands, dtype=jnp.int32),
        jnp.asarray(corner_array),
        jnp.asarray(value_array),
        jnp.asarray(unresolved_array),
        jnp.asarray(lifshitz_array),
        jnp.asarray(fermi),
        jnp.asarray(complete),
        mesh.mesh_id,
        canonical_fingerprint(
            {
                "kind": "fermi-surface-evidence",
                "mesh": mesh.mesh_id,
                "spectrum": spectrum.result_id,
                "fermi_energy": fermi,
                "arrays": array_tree_fingerprint(
                    {
                        "cells": cell_array,
                        "bands": np.asarray(bands, dtype=np.int32),
                        "corners": corner_array,
                        "energies": value_array,
                        "unresolved": unresolved_array,
                        "lifshitz": lifshitz_array,
                    }
                ),
            }
        ),
    )


__all__ = [
    "FermiSurfaceEvidence",
    "PeriodicDensityOfStatesPlan",
    "PeriodicDensityOfStatesResult",
    "PeriodicProjectedDOSPlan",
    "PeriodicProjectedDOSResult",
    "PeriodicProjectorGroups",
    "PeriodicVelocityPlan",
    "PeriodicVelocityResult",
    "fermi_surface_evidence",
]
