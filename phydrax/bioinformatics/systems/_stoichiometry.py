#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    PreparedFactorization,
    RankPolicy,
)
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._network import StoichiometricNetwork, TIME


class StoichiometryStatus(IntEnum):
    """Terminal status for stoichiometric semantic audits."""

    SUCCESS = 0
    MASS_IMBALANCE = 1
    CHARGE_IMBALANCE = 2
    UNIT_MISMATCH = 3
    INCOMPLETE_COMPOSITION = 4
    MULTIPLE_FAILURES = 5
    NONFINITE_STOICHIOMETRY = 6


class ConservationStatus(IntEnum):
    """Terminal status for numerical conservation-space analysis."""

    SUCCESS = 0
    NONFINITE_STOICHIOMETRY = 1
    NUMERICAL_FAILURE = 2


class StoichiometryCapacityError(ValueError):
    """Raised before materialization when a declared audit capacity is exceeded."""


class StoichiometryEvidence(StrictModule):
    """Per-reaction elemental, charge, and dimensional residual evidence."""

    element_residuals: Array
    charge_residuals: Array
    mass_balanced: Array
    charge_balanced: Array
    unit_consistent: Array
    balance_applicable: Array
    composition_complete: Array
    element_names: tuple[str, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    balance_claim: str = eqx.field(static=True)


class StoichiometryAudit(StrictModule):
    """Audited chemical consistency of a compiled stoichiometric network."""

    valid: Array
    status: Array
    evidence: StoichiometryEvidence
    method_contract: BioinformaticsMethodContract
    network_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(StoichiometryStatus.SUCCESS))


class ConservationEvidence(StrictModule):
    """Singular spectrum, rank cutoff, and nullspace residual evidence."""

    singular_values: Array
    factorization: PreparedFactorization | None
    numerical_rank: Array
    left_nullspace_residual: Array
    maximum_residual: Array
    cutoff: Array
    complete_basis: bool = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    approximation: str = eqx.field(static=True)


class ConservationAnalysis(StrictModule):
    """Numerical left-nullspace basis for conserved internal-species pools."""

    valid: Array
    status: Array
    conservation_laws: Array
    internal_species_indices: Array
    evidence: ConservationEvidence
    method_contract: BioinformaticsMethodContract
    network_id: str = eqx.field(static=True)

    @property
    def num_conservation_laws(self) -> int:
        return int(self.conservation_laws.shape[0])


def _audit_contract(tolerance: float, /) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "stoichiometric-semantic-audit",
        MethodKind.EXACT_MODEL if tolerance == 0.0 else MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Elemental and charge residuals inherit the scaling of supplied "
            "stoichiometric coefficients."
        ),
        truncation_statement="Every reaction and every declared element is audited.",
        capacity_semantics=(
            "The element-by-reaction audit is fully materialized after an explicit "
            "capacity preflight; excess capacity is rejected."
        ),
        assumptions=(
            "Stoichiometric coefficients are dimensionless.",
            "Exchange reactions are open-system boundaries and are not mass-balanced internally.",
        ),
        nondifferentiable_outputs=("status", "valid", "balance flags"),
        absolute_tolerance=tolerance,
    )


def _conservation_contract(relative_tolerance: float, /) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "stoichiometric-conservation-analysis",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Numerical rank is determined relative to the largest singular value; "
            "near-dependent reactions may change rank under perturbation."
        ),
        truncation_statement="The complete numerical left-nullspace basis is returned.",
        capacity_semantics=(
            "The full dense stoichiometric matrix is preflighted against max_matrix_entries."
        ),
        assumptions=("Conservation is assessed only over internal species rows.",),
        nondifferentiable_outputs=("numerical_rank", "status", "valid"),
        relative_tolerance=relative_tolerance,
    )


def audit_stoichiometry(
    network: StoichiometricNetwork,
    /,
    *,
    tolerance: float = 1.0e-10,
    max_audit_entries: int = 1_000_000,
) -> StoichiometryAudit:
    """Audit elemental mass, formal charge, and reaction-rate dimensions."""

    if not isinstance(network, StoichiometricNetwork):
        raise TypeError("network must be a StoichiometricNetwork.")
    tolerance_ = float(tolerance)
    capacity = int(max_audit_entries)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    if capacity < 1:
        raise ValueError("max_audit_entries must be positive.")
    element_names = tuple(
        sorted(
            {
                symbol
                for species in network.species
                if species.composition is not None
                for symbol, _ in species.composition.elements
            }
        )
    )
    required_entries = max(1, len(element_names)) * network.num_reactions
    if required_entries > capacity:
        raise StoichiometryCapacityError(
            f"Stoichiometric audit requires {required_entries} entries; capacity is {capacity}."
        )
    element_index = {name: index for index, name in enumerate(element_names)}
    residuals = np.zeros((len(element_names), network.num_reactions), dtype=np.float64)
    charge_residuals = np.zeros((network.num_reactions,), dtype=np.float64)
    complete = np.ones((network.num_reactions,), dtype=bool)
    applicable = np.asarray(
        [not reaction.exchange for reaction in network.reactions], dtype=bool
    )
    unit_consistent = np.ones((network.num_reactions,), dtype=bool)
    for reaction_index, reaction in enumerate(network.reactions):
        expected_dimensions = []
        for local_index, species_id in enumerate(reaction.species_ids):
            species = network.species[network.species_index(species_id)]
            expected_dimensions.append(
                species.substance_unit.multiply(TIME.power(-1)).exponents
            )
            composition = species.composition
            if composition is None:
                complete[reaction_index] = False
                continue
            coefficient = float(
                np.asarray(reaction.stoichiometric_coefficients[local_index])
            )
            for symbol, count in composition.elements:
                residuals[element_index[symbol], reaction_index] += coefficient * count
            charge_residuals[reaction_index] += coefficient * composition.charge
        unit_consistent[reaction_index] = all(
            dimensions == reaction.flux_unit.exponents
            for dimensions in expected_dimensions
        )
    mass_balanced = (
        np.all(np.abs(residuals) <= tolerance_, axis=0)
        if element_names
        else np.ones((network.num_reactions,), dtype=bool)
    )
    mass_balanced = mass_balanced | ~applicable
    charge_balanced = (np.abs(charge_residuals) <= tolerance_) | ~applicable
    complete_for_claim = complete | ~applicable
    finite = bool(np.isfinite(np.asarray(network.stoichiometric_matrix)).all())
    failures = np.asarray(
        [
            not bool(np.all(mass_balanced)),
            not bool(np.all(charge_balanced)),
            not bool(np.all(unit_consistent)),
            not bool(np.all(complete_for_claim)),
        ],
        dtype=bool,
    )
    failure_count = int(np.count_nonzero(failures))
    if not finite:
        status = StoichiometryStatus.NONFINITE_STOICHIOMETRY
    elif failure_count > 1:
        status = StoichiometryStatus.MULTIPLE_FAILURES
    elif failures[0]:
        status = StoichiometryStatus.MASS_IMBALANCE
    elif failures[1]:
        status = StoichiometryStatus.CHARGE_IMBALANCE
    elif failures[2]:
        status = StoichiometryStatus.UNIT_MISMATCH
    elif failures[3]:
        status = StoichiometryStatus.INCOMPLETE_COMPOSITION
    else:
        status = StoichiometryStatus.SUCCESS
    evidence = StoichiometryEvidence(
        element_residuals=jnp.asarray(residuals),
        charge_residuals=jnp.asarray(charge_residuals),
        mass_balanced=jnp.asarray(mass_balanced),
        charge_balanced=jnp.asarray(charge_balanced),
        unit_consistent=jnp.asarray(unit_consistent),
        balance_applicable=jnp.asarray(applicable),
        composition_complete=jnp.asarray(complete),
        element_names=element_names,
        tolerance=tolerance_,
        balance_claim=(
            "exact-zero floating-point comparison"
            if tolerance_ == 0.0
            else "absolute-tolerance floating-point comparison"
        ),
    )
    status_array = jnp.asarray(int(status), dtype=jnp.int32)
    return StoichiometryAudit(
        valid=status_array == int(StoichiometryStatus.SUCCESS),
        status=status_array,
        evidence=evidence,
        method_contract=_audit_contract(tolerance_),
        network_id=network.network_id,
    )


def conservation_analysis(
    network: StoichiometricNetwork,
    /,
    *,
    relative_tolerance: float = 1.0e-10,
    max_matrix_entries: int = 1_000_000,
) -> ConservationAnalysis:
    """Return a complete numerical basis of internal-species conservation laws."""

    if not isinstance(network, StoichiometricNetwork):
        raise TypeError("network must be a StoichiometricNetwork.")
    tolerance = float(relative_tolerance)
    capacity = int(max_matrix_entries)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    if capacity < 1:
        raise ValueError("max_matrix_entries must be positive.")
    internal_indices = np.flatnonzero(np.asarray(network.internal_species_mask))
    matrix = network.stoichiometric_matrix[jnp.asarray(internal_indices, dtype=jnp.int32)]
    if matrix.size > capacity:
        raise StoichiometryCapacityError(
            f"Conservation analysis requires {matrix.size} matrix entries; capacity is {capacity}."
        )
    finite = bool(np.asarray(jnp.all(jnp.isfinite(matrix))))
    if finite and matrix.shape[0] > 0:
        factorization = factorize(
            DenseLinearOperator(matrix),
            FactorizationPolicy(
                "svd",
                rank=RankPolicy(relative_cutoff=tolerance),
            ),
        )
        singular_values = factorization.singular_values()
        largest = jnp.max(singular_values, initial=0.0)
        cutoff = tolerance * largest
        rank = factorization.rank()
        left_nullspace = factorization.left_nullspace()
        dimension = int(np.asarray(left_nullspace.dimension))
        laws = left_nullspace.basis[:, :dimension].T
        residual = laws @ matrix
        maximum = jnp.max(jnp.abs(residual), initial=0.0)
        valid = jnp.isfinite(maximum) & (maximum <= cutoff)
        status = (
            ConservationStatus.SUCCESS
            if bool(np.asarray(valid))
            else ConservationStatus.NUMERICAL_FAILURE
        )
    elif finite:
        factorization = None
        singular_values = jnp.zeros((0,), dtype=matrix.dtype)
        cutoff = jnp.asarray(0.0, dtype=matrix.dtype)
        rank = jnp.asarray(0, dtype=jnp.int32)
        laws = jnp.zeros((0, 0), dtype=matrix.dtype)
        residual = jnp.zeros((0, matrix.shape[1]), dtype=matrix.dtype)
        maximum = jnp.asarray(0.0, dtype=matrix.dtype)
        valid = jnp.asarray(True)
        status = ConservationStatus.SUCCESS
    else:
        factorization = None
        singular_values = jnp.full((min(matrix.shape),), jnp.nan)
        cutoff = jnp.asarray(jnp.nan)
        rank = jnp.asarray(0, dtype=jnp.int32)
        laws = jnp.zeros((0, matrix.shape[0]), dtype=jnp.float64)
        residual = jnp.full((0, matrix.shape[1]), jnp.nan)
        maximum = jnp.asarray(jnp.nan)
        valid = jnp.asarray(False)
        status = ConservationStatus.NONFINITE_STOICHIOMETRY
    evidence = ConservationEvidence(
        singular_values=jnp.asarray(singular_values),
        factorization=factorization,
        numerical_rank=jnp.asarray(rank, dtype=jnp.int32),
        left_nullspace_residual=jnp.asarray(residual),
        maximum_residual=jnp.asarray(maximum),
        cutoff=jnp.asarray(cutoff),
        complete_basis=True,
        exact=False,
        approximation="phydrax native dense SVD factorization",
    )
    return ConservationAnalysis(
        valid=jnp.asarray(valid),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        conservation_laws=jnp.asarray(laws),
        internal_species_indices=jnp.asarray(internal_indices, dtype=jnp.int32),
        evidence=evidence,
        method_contract=_conservation_contract(tolerance),
        network_id=network.network_id,
    )


__all__ = [
    "audit_stoichiometry",
    "conservation_analysis",
    "ConservationAnalysis",
    "ConservationEvidence",
    "ConservationStatus",
    "StoichiometryAudit",
    "StoichiometryCapacityError",
    "StoichiometryEvidence",
    "StoichiometryStatus",
]
