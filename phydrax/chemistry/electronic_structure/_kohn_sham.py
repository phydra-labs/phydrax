#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native restricted Slater-LDA SCF and finite-field polarizability."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan
from ...ein import contract
from ...operators.quantum.gaussian import (
    ao_values,
    molecular_integrals,
    PreparedGaussianBasis,
)
from ...units import (
    BOHR,
    conversion_factor,
    derived_unit,
    ELEMENTARY_CHARGE,
    HARTREE,
    UnitDefinition,
)
from .._kernel import ElectronicKernelEvaluation
from ._hartree_fock import _symmetric_eigh, SCFState


class MolecularIntegrationGridPlan(StrictModule, NonTrainableState):
    """Fixed molecular quadrature points and volume weights in native length units."""

    points: Array
    weights: Array
    system_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        points: ArrayLike,
        weights: ArrayLike,
        /,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        points_ = jnp.asarray(points, dtype=np.dtype(system.coordinate_dtype))
        weights_ = jnp.asarray(weights, dtype=points_.dtype)
        if (
            points_.ndim != 2
            or points_.shape[1] != 3
            or weights_.shape != (points_.shape[0],)
        ):
            raise ValueError("Grid points and weights must have shapes (P,3) and (P,).")
        if (
            np.any(~np.isfinite(np.asarray(points_)))
            or np.any(~np.isfinite(np.asarray(weights_)))
            or np.any(np.asarray(weights_) <= 0.0)
        ):
            raise ValueError("Grid points must be finite and weights positive finite.")
        self.points = points_
        self.weights = weights_
        self.system_id = system.system_id
        self.grid_id = canonical_fingerprint(
            {
                "kind": "molecular-integration-grid",
                "system": system.system_id,
                "length_unit": system.units.scale.length_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {"points": np.asarray(points_), "weights": np.asarray(weights_)}
                ),
            }
        )

    @classmethod
    def cartesian_box(
        cls,
        system: AtomisticSystemPlan,
        center: ArrayLike,
        /,
        *,
        half_width: float,
        points_per_axis: int,
    ) -> MolecularIntegrationGridPlan:
        center_ = np.asarray(center, dtype=np.dtype(system.coordinate_dtype))
        width = float(half_width)
        count = int(points_per_axis)
        if center_.shape != (3,) or not np.all(np.isfinite(center_)):
            raise ValueError("Grid center must be a finite three-vector.")
        if not isfinite(width) or width <= 0.0 or count < 3:
            raise ValueError("Grid half_width/count are invalid.")
        axis = np.linspace(-width, width, count)
        spacing = float(axis[1] - axis[0])
        mesh = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        points = center_[None, :] + mesh.reshape((-1, 3))
        weights = np.full((points.shape[0],), spacing**3)
        return cls(system, points, weights)


class StaticPolarizabilityResult(StrictModule, NonTrainableState):
    tensor: Array
    symmetry_residual: Array
    successful: Array
    unit: UnitDefinition
    source_state_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensor: ArrayLike,
        symmetry_residual: ArrayLike,
        successful: ArrayLike,
        unit: UnitDefinition,
        source_state_ids: tuple[str, ...],
        /,
    ):
        tensor_ = jnp.asarray(tensor)
        if tensor_.shape != (3, 3):
            raise ValueError("Polarizability tensor must have shape (3, 3).")
        self.tensor = tensor_
        self.symmetry_residual = jnp.asarray(
            symmetry_residual, dtype=tensor_.dtype
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        if not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition.")
        self.unit = unit
        self.source_state_ids = source_state_ids
        self.result_id = canonical_fingerprint(
            {
                "kind": "static-polarizability-result",
                "unit": self.unit.unit_id,
                "states": list(source_state_ids),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "tensor": np.asarray(tensor_),
                        "symmetry_residual": np.asarray(self.symmetry_residual),
                    }
                ),
            }
        )


class NativeLDAPlan(StrictModule, NonTrainableState):
    """Restricted Slater exchange-only LDA over a fixed real-space quadrature."""

    system: AtomisticSystemPlan
    basis: PreparedGaussianBasis
    grid: MolecularIntegrationGridPlan
    electron_count: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    force_displacement: float = eqx.field(static=True)
    field_displacement: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        basis: PreparedGaussianBasis,
        grid: MolecularIntegrationGridPlan,
        electron_count: int,
        /,
        *,
        convergence_tolerance: float = 1.0e-9,
        maximum_iterations: int = 128,
        damping: float = 0.3,
        force_displacement: float = 1.0e-4,
        field_displacement: float = 1.0e-3,
    ):
        if basis.system_id != system.system_id or grid.system_id != system.system_id:
            raise ValueError("LDA basis/grid must belong to the supplied system.")
        if not bool(np.all(np.asarray(system.active_mask))) or not bool(
            np.all(np.asarray(system.element_mask))
        ):
            raise ValueError(
                "Native restricted LDA requires an unpadded all-element system."
            )
        electrons = int(electron_count)
        if electrons <= 0 or electrons % 2 or electrons > 2 * basis.basis_function_count:
            raise ValueError(
                "Native restricted LDA requires a positive even electron count."
            )
        values = tuple(
            float(value)
            for value in (
                convergence_tolerance,
                damping,
                force_displacement,
                field_displacement,
            )
        )
        if (
            not isfinite(values[0])
            or values[0] <= 0.0
            or not isfinite(values[1])
            or values[1] < 0.0
            or values[1] >= 1.0
            or any(not isfinite(value) or value <= 0.0 for value in values[2:])
        ):
            raise ValueError("LDA convergence/damping/displacement values are invalid.")
        iterations = int(maximum_iterations)
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        self.system = system
        self.basis = basis
        self.grid = grid
        self.electron_count = electrons
        self.convergence_tolerance = values[0]
        self.maximum_iterations = iterations
        self.damping = values[1]
        self.force_displacement = values[2]
        self.field_displacement = values[3]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-slater-lda-plan",
                "system": system.system_id,
                "basis": basis.prepared_id,
                "grid": grid.grid_id,
                "electron_count": electrons,
                "convergence_tolerance": values[0],
                "maximum_iterations": iterations,
                "damping": values[1],
                "force_displacement": values[2],
                "field_displacement": values[3],
            }
        )

    def _ao_values(self, positions_bohr: Array, grid_bohr: Array) -> Array:
        return ao_values(self.basis, positions_bohr, grid_bohr)

    def solve_atomic_units(
        self, positions_bohr: ArrayLike, /, *, field_atomic: ArrayLike | None = None
    ) -> SCFState:
        positions = jnp.asarray(positions_bohr)
        charges = jnp.asarray(self.system.atomic_numbers, dtype=positions.dtype)
        integrals = molecular_integrals(self.basis, positions, charges)
        overlap_values, overlap_vectors = _symmetric_eigh(integrals.overlap)
        if bool(jnp.min(overlap_values) <= 1.0e-9):
            raise ValueError("LDA overlap matrix is numerically rank deficient.")
        orthogonalizer = (
            overlap_vectors @ jnp.diag(overlap_values**-0.5) @ overlap_vectors.T
        )
        length_to_bohr = float(
            conversion_factor(self.system.units.scale.length_unit, BOHR)
        )
        grid_points = self.grid.points * length_to_bohr
        grid_weights = self.grid.weights * length_to_bohr**3
        ao = self._ao_values(positions, grid_points)
        field = (
            jnp.zeros((3,), dtype=positions.dtype)
            if field_atomic is None
            else jnp.asarray(field_atomic, dtype=positions.dtype)
        )
        core = integrals.core_hamiltonian + contract("x,xab->ab", field, integrals.dipole)
        nuclear_field = -contract("a,ax,x->", charges, positions, field)
        occupied = self.electron_count // 2
        density = jnp.zeros_like(core)
        previous_energy = jnp.asarray(jnp.inf, dtype=positions.dtype)
        residual = jnp.asarray(jnp.inf, dtype=positions.dtype)
        coefficients = jnp.eye(core.shape[0], dtype=positions.dtype)
        orbital_energies = jnp.zeros((core.shape[0],), dtype=positions.dtype)
        electronic_energy = jnp.asarray(jnp.nan, dtype=positions.dtype)
        converged = False
        completed = 0
        exchange_factor = -0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)
        for iteration in range(self.maximum_iterations):
            coulomb = contract("cd,abcd->ab", density, integrals.electron_repulsion)
            rho = jnp.maximum(contract("ab,pa,pb->p", density, ao, ao), 1.0e-18)
            exchange_energy = exchange_factor * jnp.sum(grid_weights * rho ** (4.0 / 3.0))
            exchange_potential = -((3.0 / jnp.pi) ** (1.0 / 3.0)) * rho ** (1.0 / 3.0)
            exchange_matrix = contract(
                "p,p,pa,pb->ab", grid_weights, exchange_potential, ao, ao
            )
            fock = core + coulomb + exchange_matrix
            orbital_energies, transformed = _symmetric_eigh(
                orthogonalizer.T @ fock @ orthogonalizer
            )
            coefficients = orthogonalizer @ transformed
            occupied_coefficients = coefficients[:, :occupied]
            proposed = 2.0 * occupied_coefficients @ occupied_coefficients.T
            mixed = (1.0 - self.damping) * proposed + self.damping * density
            one_electron = contract("ab,ab->", mixed, core)
            coulomb_energy = 0.5 * contract("ab,ab->", mixed, coulomb)
            electronic_energy = one_electron + coulomb_energy + exchange_energy
            residual = jnp.maximum(
                jnp.max(jnp.abs(mixed - density)),
                jnp.abs(electronic_energy - previous_energy),
            )
            density = mixed
            previous_energy = electronic_energy
            completed = iteration + 1
            if bool(residual <= self.convergence_tolerance):
                converged = True
                break
        coulomb = contract("cd,abcd->ab", density, integrals.electron_repulsion)
        rho = jnp.maximum(contract("ab,pa,pb->p", density, ao, ao), 1.0e-18)
        exchange_energy = exchange_factor * jnp.sum(grid_weights * rho ** (4.0 / 3.0))
        exchange_potential = -((3.0 / jnp.pi) ** (1.0 / 3.0)) * rho ** (1.0 / 3.0)
        exchange_matrix = contract(
            "p,p,pa,pb->ab", grid_weights, exchange_potential, ao, ao
        )
        fock = core + coulomb + exchange_matrix
        orbital_energies, transformed = _symmetric_eigh(
            orthogonalizer.T @ fock @ orthogonalizer
        )
        coefficients = orthogonalizer @ transformed
        electronic_energy = (
            contract("ab,ab->", density, core)
            + 0.5 * contract("ab,ab->", density, coulomb)
            + exchange_energy
        )
        total = electronic_energy + integrals.nuclear_repulsion + nuclear_field
        return SCFState(
            density,
            coefficients,
            orbital_energies,
            electronic_energy,
            total,
            residual,
            completed,
            converged,
        )

    def dipole_atomic_units(self, positions_bohr: ArrayLike, state: SCFState, /) -> Array:
        positions = jnp.asarray(positions_bohr)
        charges = jnp.asarray(self.system.atomic_numbers, dtype=positions.dtype)
        integrals = molecular_integrals(self.basis, positions, charges)
        return -contract("ab,xab->x", state.density, integrals.dipole) + contract(
            "a,ax->x", charges, positions
        )

    def static_polarizability(
        self, positions: ArrayLike, /
    ) -> StaticPolarizabilityResult:
        coordinate = jnp.asarray(positions)
        length_to_bohr = float(
            conversion_factor(self.system.units.scale.length_unit, BOHR)
        )
        coordinate_bohr = coordinate * length_to_bohr
        columns = []
        state_ids: list[str] = []
        successful = True
        for axis in range(3):
            field = (
                jnp.zeros((3,), dtype=coordinate.dtype)
                .at[axis]
                .set(self.field_displacement)
            )
            plus = self.solve_atomic_units(coordinate_bohr, field_atomic=field)
            minus = self.solve_atomic_units(coordinate_bohr, field_atomic=-field)
            plus_dipole = self.dipole_atomic_units(coordinate_bohr, plus)
            minus_dipole = self.dipole_atomic_units(coordinate_bohr, minus)
            columns.append((plus_dipole - minus_dipole) / (2.0 * self.field_displacement))
            state_ids.extend((plus.state_id, minus.state_id))
            successful = successful and bool(plus.converged) and bool(minus.converged)
        raw_tensor_atomic = jnp.stack(tuple(columns), axis=1)
        raw_residual = jnp.max(jnp.abs(raw_tensor_atomic - raw_tensor_atomic.T))
        tensor_atomic = 0.5 * (raw_tensor_atomic + raw_tensor_atomic.T)
        source_unit = derived_unit(
            "e^2*bohr^2/hartree",
            ((ELEMENTARY_CHARGE, 2), (BOHR, 2), (HARTREE, -1)),
        )
        target_unit = derived_unit(
            "native-polarizability",
            (
                (self.system.units.charge_unit, 2),
                (self.system.units.scale.length_unit, 2),
                (self.system.units.scale.energy_unit, -1),
            ),
        )
        factor = float(conversion_factor(source_unit, target_unit))
        tensor = tensor_atomic * factor
        residual = raw_residual * factor
        return StaticPolarizabilityResult(
            tensor,
            residual,
            successful,
            target_unit,
            tuple(state_ids),
        )

    def evaluate(
        self, positions: ArrayLike, /
    ) -> tuple[ElectronicKernelEvaluation, SCFState]:
        coordinate = jnp.asarray(positions)
        length_to_bohr = float(
            conversion_factor(self.system.units.scale.length_unit, BOHR)
        )
        energy_factor = float(
            conversion_factor(HARTREE, self.system.units.scale.energy_unit)
        )
        state = self.solve_atomic_units(coordinate * length_to_bohr)
        all_converged = bool(state.converged)
        forces = jnp.zeros_like(coordinate)
        for atom in range(coordinate.shape[0]):
            for component in range(3):
                displacement = (
                    jnp.zeros_like(coordinate)
                    .at[atom, component]
                    .set(self.force_displacement)
                )
                plus = self.solve_atomic_units(
                    (coordinate + displacement) * length_to_bohr
                )
                minus = self.solve_atomic_units(
                    (coordinate - displacement) * length_to_bohr
                )
                all_converged = (
                    all_converged and bool(plus.converged) and bool(minus.converged)
                )
                forces = forces.at[atom, component].set(
                    -(plus.total_energy - minus.total_energy)
                    * energy_factor
                    / (2.0 * self.force_displacement)
                )
        dipole_atomic = self.dipole_atomic_units(coordinate * length_to_bohr, state)
        charge_factor = float(
            conversion_factor(ELEMENTARY_CHARGE, self.system.units.charge_unit)
        )
        length_from_bohr = float(
            conversion_factor(BOHR, self.system.units.scale.length_unit)
        )
        return (
            ElectronicKernelEvaluation(
                state.total_energy * energy_factor,
                forces,
                all_converged,
                dipole=dipole_atomic * charge_factor * length_from_bohr,
                iterations=state.iterations,
                residual=state.residual,
            ),
            state,
        )


__all__ = [
    "MolecularIntegrationGridPlan",
    "NativeLDAPlan",
    "StaticPolarizabilityResult",
]
