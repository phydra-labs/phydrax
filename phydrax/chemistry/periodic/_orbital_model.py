#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic orbital bases, Bloch H/S pencils, and separate Hubbard fields."""

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._periodic_cell import PeriodicCell
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ...operators.periodic._family import (
    differentiate_periodic_translation_family,
    evaluate_periodic_translation_family,
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
    PreparedPeriodicTranslationFamily,
)
from ...sparse import EdgeRelation
from ...units import ENERGY, LENGTH, UnitDefinition


BlochGaugeKind = Literal["lattice", "atomic"]
SpinOrderKind = Literal["spinless", "blocked-alpha-beta", "interleaved-alpha-beta"]


class PeriodicBlochGauge(StrictModule, NonTrainableState):
    """Explicit phase gauge for localized periodic orbitals."""

    kind: BlochGaugeKind = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(self, kind: BlochGaugeKind, /):
        if kind not in ("lattice", "atomic"):
            raise ValueError("Periodic Bloch gauge must be 'lattice' or 'atomic'.")
        self.kind = kind
        self.gauge_id = canonical_fingerprint(
            {"kind": "periodic-bloch-gauge", "gauge": kind}
        )


class PeriodicOrbitalBasisPlan(StrictModule, NonTrainableState):
    """Ordered localized basis with centers and physical cell-length units."""

    cell: PeriodicCell
    centers_fractional: Array
    labels: tuple[str, ...] = eqx.field(static=True)
    spin_order: SpinOrderKind = eqx.field(static=True)
    statistics: str = eqx.field(static=True)
    gauge: PeriodicBlochGauge
    length_unit: UnitDefinition
    cell_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        labels: tuple[str, ...],
        centers_fractional: ArrayLike,
        length_unit: UnitDefinition,
        gauge: PeriodicBlochGauge,
        /,
        *,
        spin_order: SpinOrderKind = "spinless",
    ):
        if not isinstance(cell, PeriodicCell):
            raise TypeError("Periodic orbital bases require PeriodicCell.")
        labels_ = tuple(str(value).strip() for value in labels)
        centers = np.asarray(centers_fractional)
        if (
            not labels_
            or any(not value for value in labels_)
            or len(set(labels_)) != len(labels_)
            or centers.shape != (len(labels_), cell.rank)
            or np.any(~np.isfinite(centers))
        ):
            raise ValueError(
                "Orbital labels must be unique and centers finite with shape (A, rank)."
            )
        if not isinstance(length_unit, UnitDefinition) or not isinstance(
            gauge, PeriodicBlochGauge
        ):
            raise TypeError("Orbital basis requires typed length unit and Bloch gauge.")
        if length_unit.dimension != LENGTH:
            raise ValueError(
                "Periodic orbital basis length_unit must have length dimension."
            )
        if spin_order not in (
            "spinless",
            "blocked-alpha-beta",
            "interleaved-alpha-beta",
        ):
            raise ValueError("Periodic spin order is invalid.")
        self.cell = cell
        self.centers_fractional = jnp.asarray(centers)
        self.labels = labels_
        self.spin_order = spin_order
        self.statistics = "fermion"
        self.gauge = gauge
        self.length_unit = length_unit
        self.cell_id = cell.cell_id
        self.basis_id = canonical_fingerprint(
            {
                "kind": "periodic-orbital-basis-plan",
                "cell": cell.cell_id,
                "labels": list(labels_),
                "spin_order": spin_order,
                "statistics": "fermion",
                "gauge": gauge.gauge_id,
                "length_unit": length_unit.unit_id,
                "centers": array_tree_fingerprint(centers),
            }
        )

    @property
    def orbital_count(self) -> int:
        return len(self.labels)


class PeriodicPencilEvaluation(StrictModule, NonTrainableState):
    fractional_points: Array
    hamiltonians: Array
    overlaps: Array
    d_hamiltonians: Array
    d_overlaps: Array
    overlap_minimum_eigenvalues: Array
    overlap_condition_numbers: Array
    successful: Array
    pencil_id: str = eqx.field(static=True)


class PeriodicOrbitalPencilPlan(StrictModule, NonTrainableState):
    """One periodic Hamiltonian/overlap pencil in an explicit orbital gauge."""

    basis: PeriodicOrbitalBasisPlan
    hamiltonian_plan: PeriodicTranslationFamilyPlan
    hamiltonian_state: PeriodicTranslationFamilyState
    overlap_plan: PeriodicTranslationFamilyPlan
    overlap_state: PeriodicTranslationFamilyState
    energy_unit: UnitDefinition
    overlap_eigenvalue_floor: float = eqx.field(static=True)
    maximum_overlap_condition: float = eqx.field(static=True)
    pencil_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PeriodicOrbitalBasisPlan,
        hamiltonian_plan: PeriodicTranslationFamilyPlan,
        hamiltonian_state: PeriodicTranslationFamilyState,
        overlap_plan: PeriodicTranslationFamilyPlan,
        overlap_state: PeriodicTranslationFamilyState,
        energy_unit: UnitDefinition,
        /,
        *,
        overlap_eigenvalue_floor: float = 1.0e-10,
        maximum_overlap_condition: float = 1.0e10,
    ):
        if not isinstance(basis, PeriodicOrbitalBasisPlan):
            raise TypeError("basis must be PeriodicOrbitalBasisPlan.")
        prepared_h = prepare_periodic_translation_family(
            hamiltonian_plan, hamiltonian_state
        )
        prepared_s = prepare_periodic_translation_family(overlap_plan, overlap_state)
        if (
            hamiltonian_plan.rank != basis.cell.rank
            or overlap_plan.rank != basis.cell.rank
            or prepared_h.output_size != basis.orbital_count
            or prepared_h.input_size != basis.orbital_count
            or prepared_s.output_size != basis.orbital_count
            or prepared_s.input_size != basis.orbital_count
            or hamiltonian_state.output_block_size != 1
            or hamiltonian_state.input_block_size != 1
            or overlap_state.output_block_size != 1
            or overlap_state.input_block_size != 1
        ):
            raise ValueError(
                "Orbital H/S families must be scalar-block square operators on the basis."
            )
        if not hamiltonian_plan.hermitian or not overlap_plan.hermitian:
            raise ValueError("Periodic orbital H and S families must be Hermitian.")
        if (
            hamiltonian_plan.convention.convention_id
            != overlap_plan.convention.convention_id
        ):
            raise ValueError(
                "Periodic Hamiltonian and overlap families must share one Fourier convention."
            )
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        if energy_unit.dimension != ENERGY:
            raise ValueError(
                "Periodic orbital pencil energy_unit must have energy dimension."
            )
        floor = float(overlap_eigenvalue_floor)
        condition = float(maximum_overlap_condition)
        if (
            not isfinite(floor)
            or floor <= 0.0
            or not isfinite(condition)
            or condition < 1.0
        ):
            raise ValueError("Overlap positivity and condition limits are invalid.")
        self.basis = basis
        self.hamiltonian_plan = hamiltonian_plan
        self.hamiltonian_state = hamiltonian_state
        self.overlap_plan = overlap_plan
        self.overlap_state = overlap_state
        self.energy_unit = energy_unit
        self.overlap_eigenvalue_floor = floor
        self.maximum_overlap_condition = condition
        self.pencil_id = canonical_fingerprint(
            {
                "kind": "periodic-orbital-pencil-plan",
                "basis": basis.basis_id,
                "hamiltonian": hamiltonian_state.numeric_id,
                "overlap": overlap_state.numeric_id,
                "energy_unit": energy_unit.unit_id,
                "overlap_eigenvalue_floor": floor,
                "maximum_overlap_condition": condition,
            }
        )

    @classmethod
    def orthonormal(
        cls,
        basis: PeriodicOrbitalBasisPlan,
        hamiltonian_plan: PeriodicTranslationFamilyPlan,
        hamiltonian_state: PeriodicTranslationFamilyState,
        energy_unit: UnitDefinition,
        /,
        **kwargs,
    ) -> "PeriodicOrbitalPencilPlan":
        count = basis.orbital_count
        relation = EdgeRelation(
            np.arange(count),
            np.arange(count),
            source_size=count,
            target_size=count,
        )
        overlap_plan = PeriodicTranslationFamilyPlan(
            relation,
            np.zeros((count, basis.cell.rank), dtype=np.int32),
            np.arange(count),
            maximum_dense_entries=hamiltonian_plan.maximum_dense_entries,
            maximum_finite_entries=hamiltonian_plan.maximum_finite_entries,
        )
        overlap_state = PeriodicTranslationFamilyState(
            overlap_plan, np.ones((count, 1, 1))
        )
        return cls(
            basis,
            hamiltonian_plan,
            hamiltonian_state,
            overlap_plan,
            overlap_state,
            energy_unit,
            **kwargs,
        )

    def prepare(self, /) -> "PreparedPeriodicOrbitalPencil":
        return prepare_periodic_orbital_pencil(self)


class PreparedPeriodicOrbitalPencil(StrictModule, NonTrainableState):
    plan: PeriodicOrbitalPencilPlan
    hamiltonian: PreparedPeriodicTranslationFamily
    overlap: PreparedPeriodicTranslationFamily
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: PeriodicOrbitalPencilPlan, /):
        if not isinstance(plan, PeriodicOrbitalPencilPlan):
            raise TypeError("plan must be PeriodicOrbitalPencilPlan.")
        hamiltonian = prepare_periodic_translation_family(
            plan.hamiltonian_plan, plan.hamiltonian_state
        )
        overlap = prepare_periodic_translation_family(
            plan.overlap_plan, plan.overlap_state
        )
        self.plan = plan
        self.hamiltonian = hamiltonian
        self.overlap = overlap
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-orbital-pencil",
                "plan": plan.pencil_id,
                "hamiltonian": hamiltonian.prepared_id,
                "overlap": overlap.prepared_id,
            }
        )

    def evaluate(self, fractional_points: ArrayLike, /) -> PeriodicPencilEvaluation:
        points = jnp.asarray(fractional_points)
        if points.ndim == 1:
            points = points[None, :]
        hamiltonian = evaluate_periodic_translation_family(self.hamiltonian, points)
        overlap = evaluate_periodic_translation_family(self.overlap, points)
        d_hamiltonian = differentiate_periodic_translation_family(
            self.hamiltonian, points
        )
        d_overlap = differentiate_periodic_translation_family(self.overlap, points)
        if self.plan.basis.gauge.kind == "atomic":
            centers = self.plan.basis.centers_fractional.astype(points.dtype)
            scale = (
                self.hamiltonian.plan.convention.sign
                * self.hamiltonian.plan.convention.phase_scale
            )
            phases = jnp.exp(1.0j * scale * (points @ centers.T))
            gauge = jnp.conj(phases[:, :, None]) * phases[:, None, :]
            center_difference = centers[None, :, :] - centers[:, None, :]
            h_lattice = hamiltonian
            s_lattice = overlap
            hamiltonian = gauge * h_lattice
            overlap = gauge * s_lattice
            gauge_derivative = jnp.transpose(1.0j * scale * center_difference, (2, 0, 1))
            d_hamiltonian = gauge[:, None, :, :] * (
                d_hamiltonian + gauge_derivative[None, :, :, :] * h_lattice[:, None, :, :]
            )
            d_overlap = gauge[:, None, :, :] * (
                d_overlap + gauge_derivative[None, :, :, :] * s_lattice[:, None, :, :]
            )
        overlap_spectrum = eigensolve(
            Eigenproblem(
                DenseLinearOperator(
                    0.5 * (overlap + jnp.conj(jnp.swapaxes(overlap, -1, -2))),
                    properties=OperatorProperties(
                        self_adjoint=True,
                        evidence={"self_adjoint": "construction"},
                    ),
                ),
                problem_id=f"{self.prepared_id}:overlap-positivity",
            ),
            policy=EigenSolvePolicy(
                DenseEigh(),
                count=self.plan.basis.orbital_count,
                which="smallest-algebraic",
            ),
        )
        overlap_values = overlap_spectrum.eigenvalues.real
        minimum = overlap_values[:, 0]
        condition = overlap_values[:, -1] / jnp.maximum(
            minimum,
            jnp.asarray(self.plan.overlap_eigenvalue_floor, dtype=minimum.dtype),
        )
        successful = (
            jnp.all(overlap_spectrum.successful)
            & jnp.all(jnp.isfinite(hamiltonian))
            & jnp.all(jnp.isfinite(overlap))
            & jnp.all(minimum > self.plan.overlap_eigenvalue_floor)
            & jnp.all(condition <= self.plan.maximum_overlap_condition)
        )
        return PeriodicPencilEvaluation(
            points,
            hamiltonian,
            overlap,
            d_hamiltonian,
            d_overlap,
            minimum,
            condition,
            successful,
            self.plan.pencil_id,
        )


class PeriodicHubbardMeanFieldPlan(StrictModule, NonTrainableState):
    """Diagonal Hubbard/reference/ionic fields, separate from the H/S pencil."""

    basis_id: str = eqx.field(static=True)
    onsite_hubbard: Array
    reference_populations: Array
    ionic_energy: Array
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PeriodicOrbitalBasisPlan,
        onsite_hubbard: ArrayLike,
        reference_populations: ArrayLike,
        ionic_energy: ArrayLike,
        energy_unit: UnitDefinition,
        /,
    ):
        if not isinstance(basis, PeriodicOrbitalBasisPlan):
            raise TypeError("basis must be PeriodicOrbitalBasisPlan.")
        hubbard = np.asarray(onsite_hubbard)
        reference = np.asarray(reference_populations)
        ionic = np.asarray(ionic_energy)
        if (
            hubbard.shape != (basis.orbital_count,)
            or reference.shape != hubbard.shape
            or ionic.shape != ()
            or np.any(~np.isfinite(hubbard))
            or np.any(hubbard < 0.0)
            or np.any(~np.isfinite(reference))
            or np.any(reference < 0.0)
            or not np.isfinite(ionic)
        ):
            raise ValueError("Hubbard, reference, and ionic fields are invalid.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        if energy_unit.dimension != ENERGY:
            raise ValueError(
                "Periodic Hubbard mean-field energy_unit must have energy dimension."
            )
        self.basis_id = basis.basis_id
        self.onsite_hubbard = jnp.asarray(hubbard)
        self.reference_populations = jnp.asarray(reference)
        self.ionic_energy = jnp.asarray(ionic)
        self.energy_unit = energy_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-hubbard-mean-field-plan",
                "basis": basis.basis_id,
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "onsite_hubbard": hubbard,
                        "reference_populations": reference,
                        "ionic_energy": ionic,
                    }
                ),
            }
        )


def prepare_periodic_orbital_pencil(
    plan: PeriodicOrbitalPencilPlan, /
) -> PreparedPeriodicOrbitalPencil:
    return PreparedPeriodicOrbitalPencil(plan)


__all__ = [
    "BlochGaugeKind",
    "PeriodicBlochGauge",
    "PeriodicHubbardMeanFieldPlan",
    "PeriodicOrbitalBasisPlan",
    "PeriodicOrbitalPencilPlan",
    "PeriodicPencilEvaluation",
    "PreparedPeriodicOrbitalPencil",
    "SpinOrderKind",
    "prepare_periodic_orbital_pencil",
]
