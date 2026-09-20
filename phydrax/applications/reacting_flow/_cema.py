#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_mechanism import PreparedChemicalMechanism


class ChemicalModeTrackingState(StrictModule):
    eigenvalue: Array
    right_eigenvector: Array
    initialized: Array
    plan_id: str = eqx.field(static=True)


class ChemicalExplosiveModeEvidence(StrictModule):
    conservation_residual: Array
    biorthogonality_residual: Array
    eigen_residual: Array
    eigenvalue_separation: Array
    condition_estimate: Array
    tracking_overlap: Array
    tracking_ambiguous: Array
    derivative_valid: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ChemicalExplosiveModeEvaluation(StrictModule):
    chemical_jacobian: Array
    projected_jacobian: Array
    eigenvalues: Array
    leading_eigenvalue: Array
    left_eigenvector: Array
    right_eigenvector: Array
    reaction_participation: Array
    species_explosion_index: Array
    source_contributions: Array
    proposed_tracking: ChemicalModeTrackingState
    evidence: ChemicalExplosiveModeEvidence
    plan_id: str = eqx.field(static=True)


class ChemicalExplosiveModePlan(StrictModule, NonTrainableState):
    mechanism: PreparedChemicalMechanism
    reactive_basis: Array
    conservation_matrix: Array
    contribution_labels: tuple[str, ...] = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        /,
        *,
        contribution_labels: tuple[str, ...] = (),
        minimum_separation: float = 1.0e-8,
        maximum_condition: float = 1.0e10,
        tolerance: float = 1.0e-9,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        labels = tuple(str(value).strip() for value in contribution_labels)
        if len(set(labels)) != len(labels) or any(not value for value in labels):
            raise ValueError("CEMA contribution labels must be unique and nonempty.")
        separation = float(minimum_separation)
        condition = float(maximum_condition)
        tolerance_ = float(tolerance)
        if (
            any(not isfinite(value) for value in (separation, condition, tolerance_))
            or separation <= 0.0
            or condition <= 1.0
            or tolerance_ <= 0.0
        ):
            raise ValueError("CEMA separation, conditioning, or tolerance is invalid.")
        conservation = np.concatenate(
            (
                np.asarray(mechanism.schema.element_composition, dtype=np.float64),
                np.asarray(mechanism.schema.charges, dtype=np.float64)[None, :],
            ),
            axis=0,
        )
        _, singular, right = np.linalg.svd(conservation, full_matrices=True)
        cutoff = tolerance_ * max(float(singular[0]) if singular.size else 1.0, 1.0)
        rank = int(np.sum(singular > cutoff))
        basis = right[rank:].T
        if basis.shape[1] == 0:
            raise ValueError("CEMA requires at least one reactive composition direction.")
        self.mechanism = mechanism
        self.reactive_basis = jnp.asarray(basis)
        self.conservation_matrix = jnp.asarray(conservation)
        self.contribution_labels = labels
        self.minimum_separation = separation
        self.maximum_condition = condition
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "chemical-explosive-mode",
                "mechanism": mechanism.mechanism_id,
                "reactive_basis": array_tree_fingerprint(basis),
                "contribution_labels": labels,
                "minimum_separation": separation,
                "maximum_condition": condition,
                "tolerance": tolerance_,
            }
        )

    @property
    def reactive_dimension(self) -> int:
        return self.reactive_basis.shape[1]

    def initial_tracking_state(self, dtype=jnp.complex128) -> ChemicalModeTrackingState:
        return ChemicalModeTrackingState(
            jnp.asarray(0.0 + 0.0j, dtype=dtype),
            jnp.zeros((self.mechanism.schema.species_count,), dtype=dtype),
            jnp.asarray(False),
            self.plan_id,
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        tracking: ChemicalModeTrackingState | None = None,
        source_contributions: ArrayLike | None = None,
    ) -> ChemicalExplosiveModeEvaluation:
        concentration = jnp.asarray(concentrations)
        temperature_ = jnp.asarray(temperature, dtype=concentration.dtype)
        pressure_ = jnp.asarray(pressure, dtype=concentration.dtype)
        count = self.mechanism.schema.species_count
        if concentration.shape != (count,):
            raise ValueError("CEMA concentrations must contain one value per species.")
        if temperature_.shape != () or pressure_.shape != ():
            raise ValueError("CEMA temperature and pressure must be scalar.")
        tracking_ = self.initial_tracking_state() if tracking is None else tracking
        if not isinstance(tracking_, ChemicalModeTrackingState):
            raise TypeError("tracking must be ChemicalModeTrackingState or None.")
        if tracking_.plan_id != self.plan_id:
            raise ValueError("CEMA tracking state belongs to another plan.")
        contribution_count = len(self.contribution_labels)
        if source_contributions is None:
            contributions = jnp.zeros(
                (contribution_count, count), dtype=concentration.dtype
            )
        else:
            contributions = jnp.asarray(source_contributions, dtype=concentration.dtype)
            if contributions.shape != (contribution_count, count):
                raise ValueError(
                    "source_contributions must align with labels and species."
                )

        def rates(value):
            return self.mechanism.evaluate(
                value, temperature_, pressure_
            ).species_amount_rate

        chemical = self.mechanism.evaluate(concentration, temperature_, pressure_)
        jacobian = jax.jacfwd(rates)(concentration)
        basis = self.reactive_basis.astype(jacobian.dtype)
        projected = contract("si,st,tj->ij", basis, jacobian, basis, backend="jax")
        eigenvalues, right = jnp.linalg.eig(projected)
        inverse_right = jnp.linalg.solve(
            right, jnp.eye(self.reactive_dimension, dtype=right.dtype)
        )
        left = jnp.conj(jnp.swapaxes(inverse_right, -1, -2))
        full_right = basis.astype(right.dtype) @ right
        full_left = basis.astype(left.dtype) @ left
        overlap = jnp.abs(jnp.conj(tracking_.right_eigenvector) @ full_right)
        leading = jnp.argmax(jnp.real(eigenvalues))
        tracked = jnp.argmax(overlap)
        selected = jnp.where(tracking_.initialized, tracked, leading)
        selected_eigenvalue = eigenvalues[selected]
        selected_right = full_right[:, selected]
        selected_left = full_left[:, selected]
        selected_overlap = jnp.where(
            tracking_.initialized, overlap[selected], jnp.asarray(1.0, overlap.dtype)
        )
        if self.reactive_dimension > 1:
            sorted_overlap = jnp.sort(overlap)
            overlap_margin = sorted_overlap[-1] - sorted_overlap[-2]
        else:
            overlap_margin = jnp.asarray(1.0, overlap.dtype)
        tracking_ambiguous = tracking_.initialized & (overlap_margin <= self.tolerance)
        differences = jnp.abs(eigenvalues - selected_eigenvalue)
        separation = jnp.min(
            jnp.where(
                jnp.arange(self.reactive_dimension) == selected,
                jnp.inf,
                differences,
            )
        )
        separation = jnp.where(self.reactive_dimension == 1, jnp.inf, separation)
        right_norm = jnp.linalg.norm(right)
        inverse_norm = jnp.linalg.norm(inverse_right)
        condition = right_norm * inverse_norm
        eigen_residual = jnp.linalg.norm(
            jacobian @ selected_right - selected_eigenvalue * selected_right
        )
        conservation_residual = jnp.linalg.norm(
            self.conservation_matrix.astype(selected_right.dtype) @ selected_right
        )
        biorthogonality = jnp.linalg.norm(jnp.conj(selected_left) @ selected_right - 1.0)
        reaction_jacobian = jax.jacfwd(
            lambda value: (
                self.mechanism.evaluate(value, temperature_, pressure_).net_progress_rates
            )
        )(concentration)
        left_stoichiometry = contract(
            "s,rs->r",
            jnp.conj(selected_left),
            self.mechanism.net_stoichiometry.astype(selected_left.dtype),
            backend="jax",
        )
        directional_rate = contract(
            "rs,s->r", reaction_jacobian, selected_right, backend="jax"
        )
        raw_participation = jnp.abs(left_stoichiometry * directional_rate)
        participation = raw_participation / jnp.maximum(
            jnp.sum(raw_participation), jnp.finfo(raw_participation.dtype).tiny
        )
        raw_explosion = jnp.abs(jnp.conj(selected_left) * selected_right)
        explosion = raw_explosion / jnp.maximum(
            jnp.sum(raw_explosion), jnp.finfo(raw_explosion.dtype).tiny
        )
        projected_contributions = jnp.real(
            contract(
                "s,cs->c",
                jnp.conj(selected_left),
                contributions,
                backend="jax",
            )
        )
        scale = jnp.maximum(jnp.linalg.norm(jacobian), 1.0)
        finite = (
            jnp.all(jnp.isfinite(jacobian))
            & jnp.all(jnp.isfinite(eigenvalues))
            & jnp.isfinite(condition)
            & jnp.isfinite(eigen_residual)
        )
        derivative_valid = (
            finite
            & (separation > self.minimum_separation * scale)
            & (condition <= self.maximum_condition)
            & ~tracking_ambiguous
        )
        successful = (
            chemical.successful
            & finite
            & (conservation_residual <= self.tolerance * scale)
            & (biorthogonality <= self.tolerance * jnp.maximum(condition, 1.0))
            & (eigen_residual <= self.tolerance * scale * jnp.maximum(condition, 1.0))
            & ~tracking_ambiguous
        )
        proposed = ChemicalModeTrackingState(
            selected_eigenvalue,
            selected_right,
            successful,
            self.plan_id,
        )
        evidence = ChemicalExplosiveModeEvidence(
            conservation_residual,
            biorthogonality,
            eigen_residual,
            separation,
            condition,
            selected_overlap,
            tracking_ambiguous,
            derivative_valid,
            finite,
            successful,
            self.plan_id,
        )
        return ChemicalExplosiveModeEvaluation(
            jacobian,
            projected,
            eigenvalues,
            selected_eigenvalue,
            selected_left,
            selected_right,
            participation,
            explosion,
            projected_contributions,
            proposed,
            evidence,
            self.plan_id,
        )

    def commit_tracking(
        self,
        accepted: ChemicalModeTrackingState,
        evaluation: ChemicalExplosiveModeEvaluation,
        commit: ArrayLike,
        /,
    ) -> ChemicalModeTrackingState:
        if accepted.plan_id != self.plan_id or evaluation.plan_id != self.plan_id:
            raise ValueError("CEMA tracking values belong to another plan.")
        decision = jnp.asarray(commit, dtype=jnp.bool_)
        if decision.shape != ():
            raise ValueError("CEMA tracking commit decision must be scalar.")
        return jax.tree.map(
            lambda proposed, prior: jnp.where(decision, proposed, prior),
            evaluation.proposed_tracking,
            accepted,
        )


__all__ = [
    "ChemicalExplosiveModeEvaluation",
    "ChemicalExplosiveModeEvidence",
    "ChemicalExplosiveModePlan",
    "ChemicalModeTrackingState",
]
