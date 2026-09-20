#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Harmonic phonons through the canonical periodic translation-family engine."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticUnitSystem
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ...operators.periodic import (
    differentiate_periodic_translation_family,
    evaluate_periodic_translation_family,
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
    PreparedPeriodicTranslationFamily,
)
from ._lattice_force_constants import (
    second_order_force_constant_unit,
    SecondOrderForceConstants,
)


class NonanalyticPhononCorrection(StrictModule, NonTrainableState):
    """Directional three-dimensional q→0 polar correction in Cartesian convention."""

    born_effective_charges: Array
    dielectric_tensor: Array
    charge_neutrality_residual: Array
    dielectric_symmetry_residual: Array
    dielectric_eigenvalues: Array
    units: AtomisticUnitSystem
    cell_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    correction_id: str = eqx.field(static=True)

    def __init__(
        self,
        born_effective_charges: ArrayLike,
        dielectric_tensor: ArrayLike,
        units: AtomisticUnitSystem,
        /,
        *,
        cell_id: str,
        charge_neutrality_tolerance: float = 1.0e-8,
        dielectric_symmetry_tolerance: float = 1.0e-10,
    ):
        born = np.asarray(born_effective_charges, dtype=np.float64)
        dielectric = np.asarray(dielectric_tensor, dtype=np.float64)
        if born.ndim != 3 or born.shape[1:] != (3, 3) or dielectric.shape != (3, 3):
            raise ValueError(
                "Born charges and dielectric must have shapes (N,3,3) and (3,3)."
            )
        if np.any(~np.isfinite(born)) or np.any(~np.isfinite(dielectric)):
            raise ValueError("Polar tensors must be finite.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        neutrality = float(np.max(np.abs(np.sum(born, axis=0)), initial=0.0))
        symmetry = float(np.max(np.abs(dielectric - dielectric.T), initial=0.0))
        dielectric_symmetric = 0.5 * (dielectric + dielectric.T)
        eigenvalues = np.linalg.eigvalsh(dielectric_symmetric)
        if neutrality > float(charge_neutrality_tolerance):
            raise ValueError(
                "Born effective charges violate charge neutrality; implicit projection is refused."
            )
        if symmetry > float(dielectric_symmetry_tolerance) or np.any(eigenvalues <= 0.0):
            raise ValueError(
                "The relative dielectric tensor must be symmetric positive definite."
            )
        self.born_effective_charges = jnp.asarray(born)
        self.dielectric_tensor = jnp.asarray(dielectric_symmetric)
        self.charge_neutrality_residual = jnp.asarray(neutrality)
        self.dielectric_symmetry_residual = jnp.asarray(symmetry)
        self.dielectric_eigenvalues = jnp.asarray(eigenvalues)
        self.units = units
        self.cell_id = str(cell_id)
        self.convention_id = (
            "Zstar[a,b]=cell-volume/charge*dP[a]/du[b];relative-epsilon-cartesian"
        )
        self.correction_id = canonical_fingerprint(
            {
                "kind": "nonanalytic-phonon-correction-3d",
                "cell": self.cell_id,
                "units": units.unit_system_id,
                "convention": self.convention_id,
                "arrays": array_tree_fingerprint(
                    {"born": born, "dielectric": dielectric_symmetric}
                ),
            }
        )


class PhononDispersionResult(StrictModule, NonTrainableState):
    fractional_qpoints: Array
    cartesian_qpoints: Array
    angular_frequencies: Array
    eigenvectors: Array
    dynamical_matrices: Array
    imaginary_mask: Array
    acoustic_mask: Array
    eigen_residuals: Array
    orthogonality_residuals: Array
    hermiticity_residual: Array
    acoustic_residual: Array
    successful: Array
    ifc_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        qpoints,
        cartesian,
        frequencies,
        eigenvectors,
        matrices,
        imaginary,
        acoustic,
        eigen_residuals,
        orthogonality,
        hermiticity,
        acoustic_residual,
        successful,
        ifc_id,
        unit_system_id,
        /,
    ):
        self.fractional_qpoints = jnp.asarray(qpoints)
        self.cartesian_qpoints = jnp.asarray(
            cartesian, dtype=self.fractional_qpoints.dtype
        )
        self.angular_frequencies = jnp.asarray(
            frequencies, dtype=self.fractional_qpoints.dtype
        )
        self.eigenvectors = jnp.asarray(eigenvectors)
        self.dynamical_matrices = jnp.asarray(matrices)
        self.imaginary_mask = jnp.asarray(imaginary, dtype=jnp.bool_)
        self.acoustic_mask = jnp.asarray(acoustic, dtype=jnp.bool_)
        self.eigen_residuals = jnp.asarray(
            eigen_residuals, dtype=self.fractional_qpoints.dtype
        )
        self.orthogonality_residuals = jnp.asarray(
            orthogonality, dtype=self.fractional_qpoints.dtype
        )
        self.hermiticity_residual = jnp.asarray(
            hermiticity, dtype=self.fractional_qpoints.dtype
        ).reshape(())
        self.acoustic_residual = jnp.asarray(
            acoustic_residual, dtype=self.fractional_qpoints.dtype
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.ifc_id = str(ifc_id)
        self.unit_system_id = str(unit_system_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "phonon-dispersion-result",
                "ifc": self.ifc_id,
                "units": self.unit_system_id,
                "arrays": array_tree_fingerprint(
                    {
                        "q": np.asarray(self.fractional_qpoints),
                        "frequency": np.asarray(self.angular_frequencies),
                        "eigenvectors": np.asarray(self.eigenvectors),
                        "dynamical": np.asarray(self.dynamical_matrices),
                    }
                ),
            }
        )


class PhononGroupVelocityResult(StrictModule, NonTrainableState):
    velocities: Array
    projected_velocity_matrices: Array
    cluster_ids: Array
    successful: Array
    result_id: str = eqx.field(static=True)

    def __init__(self, velocities, projected, cluster_ids, successful, dispersion_id, /):
        self.velocities = jnp.asarray(velocities)
        self.projected_velocity_matrices = jnp.asarray(projected)
        self.cluster_ids = jnp.asarray(cluster_ids, dtype=jnp.int32)
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "phonon-group-velocity-result",
                "dispersion": str(dispersion_id),
                "arrays": array_tree_fingerprint(
                    {
                        "velocities": np.asarray(self.velocities),
                        "projected": np.asarray(self.projected_velocity_matrices),
                        "clusters": np.asarray(self.cluster_ids),
                    }
                ),
            }
        )


class HarmonicPhononPlan(StrictModule, NonTrainableState):
    ifc2: SecondOrderForceConstants
    masses: Array
    units: AtomisticUnitSystem
    maximum_qpoints: int = eqx.field(static=True)
    maximum_dense_eigen_work: int = eqx.field(static=True)
    acoustic_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ifc2: SecondOrderForceConstants,
        masses: ArrayLike,
        units: AtomisticUnitSystem,
        /,
        *,
        maximum_qpoints: int = 32768,
        maximum_dense_eigen_work: int = 10_000_000_000,
        acoustic_tolerance: float = 1.0e-7,
    ):
        if not isinstance(ifc2, SecondOrderForceConstants):
            raise TypeError("ifc2 must be SecondOrderForceConstants.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        expected_unit = second_order_force_constant_unit(
            units.scale.energy_unit, units.scale.length_unit
        )
        if ifc2.unit.unit_id != expected_unit.unit_id:
            raise ValueError("IFC2 unit differs from the atomistic unit system.")
        mass = np.asarray(masses, dtype=np.float64)
        atoms = ifc2.relation.source_size
        if mass.shape != (atoms,) or np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("Masses must be finite positive values in IFC atom order.")
        if (
            int(maximum_qpoints) <= 0
            or int(maximum_dense_eigen_work) <= 0
            or not isfinite(float(acoustic_tolerance))
            or float(acoustic_tolerance) < 0.0
        ):
            raise ValueError(
                "Harmonic resource limits and acoustic tolerance are invalid."
            )
        self.ifc2 = ifc2
        self.masses = jnp.asarray(mass)
        self.units = units
        self.maximum_qpoints = int(maximum_qpoints)
        self.maximum_dense_eigen_work = int(maximum_dense_eigen_work)
        self.acoustic_tolerance = float(acoustic_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "harmonic-phonon-plan",
                "ifc": ifc2.ifc_id,
                "units": units.unit_system_id,
                "maximum_qpoints": self.maximum_qpoints,
                "maximum_dense_eigen_work": self.maximum_dense_eigen_work,
                "acoustic_tolerance": self.acoustic_tolerance,
                "masses": array_tree_fingerprint(mass),
            }
        )

    def prepare(self, /) -> "PreparedHarmonicPhonons":
        relation = self.ifc2.relation
        family_plan = PeriodicTranslationFamilyPlan(
            relation,
            self.ifc2.translations,
            self.ifc2.reverse_indices,
            maximum_dense_entries=(
                self.maximum_qpoints * (3 * relation.source_size) ** 2
            ),
        )
        family_state = PeriodicTranslationFamilyState(family_plan, self.ifc2.values)
        return PreparedHarmonicPhonons(
            self, prepare_periodic_translation_family(family_plan, family_state)
        )


class PreparedHarmonicPhonons(StrictModule, NonTrainableState):
    plan: HarmonicPhononPlan
    family: PreparedPeriodicTranslationFamily
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, family, /):
        self.plan = plan
        self.family = family
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-harmonic-phonons",
                "plan": plan.plan_id,
                "family": family.prepared_id,
            }
        )

    def evaluate(
        self,
        qpoints: ArrayLike,
        /,
        *,
        nonanalytic: NonanalyticPhononCorrection | None = None,
        gamma_directions: ArrayLike | None = None,
    ) -> PhononDispersionResult:
        q = jnp.asarray(qpoints, dtype=self.plan.masses.dtype)
        rank = self.plan.ifc2.cell.rank
        if q.ndim != 2 or q.shape[1] != rank or q.shape[0] == 0:
            raise ValueError("Phonon qpoints must have shape (Q, cell.rank).")
        branches = 3 * self.plan.masses.size
        if (
            q.shape[0] > self.plan.maximum_qpoints
            or q.shape[0] * int(branches) ** 3 > self.plan.maximum_dense_eigen_work
        ):
            raise ValueError(
                "Harmonic q/eigensolve resource capacity exceeded before allocation."
            )
        directions = (
            None
            if gamma_directions is None
            else jnp.asarray(gamma_directions, dtype=q.dtype)
        )
        if nonanalytic is not None:
            if (
                not isinstance(nonanalytic, NonanalyticPhononCorrection)
                or rank != 3
                or self.plan.ifc2.cell.ambient_dimension != 3
            ):
                raise ValueError(
                    "The 3D polar correction cannot be used for a low-dimensional periodic cell."
                )
            if (
                nonanalytic.cell_id != self.plan.ifc2.cell.cell_id
                or nonanalytic.units.unit_system_id != self.plan.units.unit_system_id
            ):
                raise ValueError(
                    "Polar correction cell or unit identity differs from harmonic phonons."
                )
            if (
                nonanalytic.born_effective_charges.shape[0] != self.plan.masses.size
                or directions is None
                or directions.shape != (q.shape[0], 3)
            ):
                raise ValueError(
                    "Polar correction requires one explicit Cartesian direction per q row."
                )
        elif directions is not None:
            raise ValueError(
                "gamma_directions are only valid with a nonanalytic correction."
            )
        matrices = evaluate_periodic_translation_family(self.family, q)
        root_mass = jnp.repeat(jnp.sqrt(self.plan.masses), 3)
        matrices = (
            matrices
            / root_mass[None, :, None]
            / root_mass[None, None, :]
            / self.plan.units.kinetic_to_energy
        )
        gamma = jnp.linalg.norm(q, axis=1) <= 1.0e-12
        if nonanalytic is not None:
            direction_norm = jnp.linalg.norm(directions, axis=1)
            if bool(jnp.any(gamma & (direction_norm <= 0.0))) or bool(
                jnp.any(~gamma & (direction_norm > 0.0))
            ):
                raise ValueError(
                    "Only Γ rows require a nonzero LO-TO direction; finite-q correction is refused."
                )
            safe_direction = jnp.where(
                gamma[:, None], directions / direction_norm[:, None], 0.0
            )
            projected_charge = contract(
                "iab,qb->qia", nonanalytic.born_effective_charges, safe_direction
            ).reshape((q.shape[0], -1))
            dielectric_denominator = contract(
                "qa,ab,qb->q",
                safe_direction,
                nonanalytic.dielectric_tensor,
                safe_direction,
            )
            coefficient = (
                4.0
                * np.pi
                * self.plan.units.coulomb_constant
                / self.plan.ifc2.cell.volume
            )
            correction = (
                coefficient
                * projected_charge[:, :, None]
                * projected_charge[:, None, :]
                / dielectric_denominator[:, None, None]
            )
            correction = (
                correction
                / root_mass[None, :, None]
                / root_mass[None, None, :]
                / self.plan.units.kinetic_to_energy
            )
            matrices = matrices + jnp.where(gamma[:, None, None], correction, 0.0)
        hermiticity = jnp.max(
            jnp.abs(matrices - jnp.conj(jnp.swapaxes(matrices, -1, -2))), initial=0.0
        )
        values = []
        vectors = []
        eigen_residuals = []
        orthogonality = []
        for matrix in matrices:
            matrix = 0.5 * (matrix + jnp.conj(matrix.T))
            solved = eigensolve(
                Eigenproblem(
                    DenseLinearOperator(
                        matrix,
                        properties=OperatorProperties(
                            self_adjoint=True, evidence={"self_adjoint": "construction"}
                        ),
                    )
                ),
                policy=EigenSolvePolicy(
                    DenseEigh(), count=branches, which="smallest-algebraic"
                ),
            )
            values.append(solved.eigenvalues.real)
            vectors.append(solved.eigenvectors)
            eigen_residuals.append(jnp.max(solved.diagnostics.residual_norms))
            orthogonality.append(solved.diagnostics.orthogonality_error)
        eigenvalues = jnp.stack(values)
        frequencies = jnp.sign(eigenvalues) * jnp.sqrt(jnp.abs(eigenvalues))
        eigenvectors = jnp.stack(vectors)
        acoustic_mask = gamma[:, None] & (
            jnp.abs(frequencies) <= self.plan.acoustic_tolerance
        )
        nonpolar_gamma = gamma if nonanalytic is None else jnp.zeros_like(gamma)
        gamma_acoustic = jnp.where(
            nonpolar_gamma[:, None], jnp.sort(jnp.abs(frequencies), axis=1)[:, :3], 0.0
        )
        acoustic_residual = jnp.max(gamma_acoustic, initial=0.0)
        successful = (
            jnp.all(jnp.isfinite(frequencies))
            & (hermiticity <= 1.0e-8)
            & (acoustic_residual <= self.plan.acoustic_tolerance)
        )
        cartesian = contract("qr,rd->qd", q, self.plan.ifc2.cell.reciprocal_vectors)
        return PhononDispersionResult(
            q,
            cartesian,
            frequencies,
            eigenvectors,
            matrices,
            frequencies < 0.0,
            acoustic_mask,
            jnp.stack(eigen_residuals),
            jnp.stack(orthogonality),
            hermiticity,
            acoustic_residual,
            successful,
            self.plan.ifc2.ifc_id,
            self.plan.units.unit_system_id,
        )

    def group_velocity(
        self,
        dispersion: PhononDispersionResult,
        /,
        *,
        degeneracy_tolerance: float = 1.0e-7,
    ) -> PhononGroupVelocityResult:
        if dispersion.ifc_id != self.plan.ifc2.ifc_id:
            raise ValueError("Dispersion belongs to a different IFC artifact.")
        derivatives = differentiate_periodic_translation_family(
            self.family, dispersion.fractional_qpoints, order=1
        )
        root_mass = jnp.repeat(jnp.sqrt(self.plan.masses), 3)
        derivatives = (
            derivatives
            / root_mass[None, None, :, None]
            / root_mass[None, None, None, :]
            / self.plan.units.kinetic_to_energy
        )
        cartesian_derivatives = contract(
            "qrab,rd->qdab", derivatives, self.plan.ifc2.cell.vectors / (2.0 * np.pi)
        )
        projected = contract(
            "qai,qdab,qbj->qijd",
            jnp.conj(dispersion.eigenvectors),
            cartesian_derivatives,
            dispersion.eigenvectors,
        )
        frequencies = dispersion.angular_frequencies
        denominator = frequencies[:, :, None, None] + frequencies[:, None, :, None]
        velocity_matrices = jnp.where(
            jnp.abs(denominator) > float(degeneracy_tolerance),
            projected / denominator,
            0.0,
        ).real
        velocities = jnp.diagonal(velocity_matrices, axis1=1, axis2=2).transpose(
            (0, 2, 1)
        )
        host_frequency = np.asarray(frequencies)
        clusters = np.zeros(host_frequency.shape, dtype=np.int32)
        for q_index, row in enumerate(host_frequency):
            cluster = 0
            for branch in range(1, row.size):
                if abs(row[branch] - row[branch - 1]) > float(degeneracy_tolerance):
                    cluster += 1
                clusters[q_index, branch] = cluster
        successful = jnp.all(jnp.isfinite(velocities)) & jnp.all(
            jnp.isfinite(velocity_matrices)
        )
        return PhononGroupVelocityResult(
            velocities, velocity_matrices, clusters, successful, dispersion.result_id
        )


def prepare_harmonic_phonons(plan: HarmonicPhononPlan, /) -> PreparedHarmonicPhonons:
    return plan.prepare()


def evaluate_phonon_dispersion(
    prepared: PreparedHarmonicPhonons, qpoints: ArrayLike, /, **kwargs
) -> PhononDispersionResult:
    if not isinstance(prepared, PreparedHarmonicPhonons):
        raise TypeError("prepared must be PreparedHarmonicPhonons.")
    return prepared.evaluate(qpoints, **kwargs)


def evaluate_phonon_group_velocity(
    prepared: PreparedHarmonicPhonons, dispersion: PhononDispersionResult, /, **kwargs
) -> PhononGroupVelocityResult:
    if not isinstance(prepared, PreparedHarmonicPhonons):
        raise TypeError("prepared must be PreparedHarmonicPhonons.")
    return prepared.group_velocity(dispersion, **kwargs)


__all__ = [
    "HarmonicPhononPlan",
    "NonanalyticPhononCorrection",
    "PhononDispersionResult",
    "PhononGroupVelocityResult",
    "PreparedHarmonicPhonons",
    "evaluate_phonon_dispersion",
    "evaluate_phonon_group_velocity",
    "prepare_harmonic_phonons",
]
