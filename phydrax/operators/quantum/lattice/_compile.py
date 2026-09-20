#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host planning and fixed-shape compilation for canonical quantum lattices."""

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ._model import LocalOperatorPlan, QuantumLatticeSpecification, QuantumLatticeTerm
from ._sector import SectorChargeMap


class QuantumLatticeResourcePolicy(StrictModule):
    """Hard compile and matrix-free action bounds, not a release claim."""

    maximum_terms: int = eqx.field(static=True)
    maximum_factors_per_term: int = eqx.field(static=True)
    maximum_branches_per_input: int = eqx.field(static=True)
    maximum_sector_dimension: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_terms: int,
        maximum_factors_per_term: int,
        maximum_branches_per_input: int,
        maximum_sector_dimension: int,
        maximum_workspace_bytes: int,
    ):
        values = tuple(
            (
                maximum_terms,
                maximum_factors_per_term,
                maximum_branches_per_input,
                maximum_sector_dimension,
                maximum_workspace_bytes,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("Quantum-lattice resource limits must be positive.")
        (
            self.maximum_terms,
            self.maximum_factors_per_term,
            self.maximum_branches_per_input,
            self.maximum_sector_dimension,
            self.maximum_workspace_bytes,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quantum-lattice-resource-policy",
                "maximum_terms": values[0],
                "maximum_factors_per_term": values[1],
                "maximum_branches_per_input": values[2],
                "maximum_sector_dimension": values[3],
                "maximum_workspace_bytes": values[4],
            }
        )


class QuantumLatticeCompilerPlan(StrictModule):
    """Admitted symbolic compile plan with conservative action estimates."""

    resources: QuantumLatticeResourcePolicy = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    expanded_term_count: int = eqx.field(static=True)
    maximum_factor_count: int = eqx.field(static=True)
    total_branches_per_input: int = eqx.field(static=True)
    maximum_term_branches: int = eqx.field(static=True)
    action_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class CompiledMonomial(StrictModule):
    """One fixed ordered product in the compiled sum."""

    factors: tuple[LocalOperatorPlan, ...]
    coefficient: Array
    charge_delta: tuple[tuple[str, int], ...] = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    source_term_id: str = eqx.field(static=True)
    adjoint_component: bool = eqx.field(static=True)
    monomial_id: str = eqx.field(static=True)


class PreparedQuantumLattice(StrictModule):
    """Numerically bound fixed-shape lattice execution object."""

    specification: QuantumLatticeSpecification
    monomials: tuple[CompiledMonomial, ...]
    plan: QuantumLatticeCompilerPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array


class ChargeMapCertification(StrictModule):
    """Per-monomial proof that one compiled operator has a unique sector map."""

    observed_deltas: Array
    term_valid: Array
    accepted: Array
    charge_labels: tuple[str, ...] = eqx.field(static=True)
    map_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    certification_id: str = eqx.field(static=True)


def plan_quantum_lattice(
    specification: QuantumLatticeSpecification,
    resources: QuantumLatticeResourcePolicy,
    /,
) -> QuantumLatticeCompilerPlan:
    """Forecast and reject before compiling any branch state."""
    if not isinstance(specification, QuantumLatticeSpecification):
        raise TypeError("specification must be QuantumLatticeSpecification.")
    if not isinstance(resources, QuantumLatticeResourcePolicy):
        raise TypeError("resources must be QuantumLatticeResourcePolicy.")
    expanded = sum(2 if term.add_adjoint else 1 for term in specification.terms)
    maximum_factors = max(len(term.factors) for term in specification.terms)
    branches = tuple(
        prod(factor.space.dimension for factor in term.factors)
        for term in specification.terms
    )
    total_branches = sum(
        branch * (2 if term.add_adjoint else 1)
        for branch, term in zip(branches, specification.terms, strict=True)
    )
    maximum_branches = max(branches)
    coordinate_bytes = len(specification.spaces) * np.dtype(np.int32).itemsize
    action_workspace = total_branches * (
        coordinate_bytes + np.dtype(np.complex128).itemsize + np.dtype(np.bool_).itemsize
    )
    violations = []
    if expanded > resources.maximum_terms:
        violations.append("expanded term count")
    if maximum_factors > resources.maximum_factors_per_term:
        violations.append("factor count")
    if total_branches > resources.maximum_branches_per_input:
        violations.append("branch count")
    if action_workspace > resources.maximum_workspace_bytes:
        violations.append("action workspace")
    if violations:
        raise ValueError(
            "Quantum-lattice resource admission rejected: " + ", ".join(violations) + "."
        )
    return QuantumLatticeCompilerPlan(
        resources=resources,
        specification_id=specification.specification_id,
        expanded_term_count=expanded,
        maximum_factor_count=maximum_factors,
        total_branches_per_input=total_branches,
        maximum_term_branches=maximum_branches,
        action_workspace_bytes=action_workspace,
        plan_id=canonical_fingerprint(
            {
                "kind": "quantum-lattice-compiler-plan",
                "specification": specification.specification_id,
                "resources": resources.policy_id,
                "expanded_term_count": expanded,
                "maximum_factor_count": maximum_factors,
                "total_branches_per_input": total_branches,
                "action_workspace_bytes": action_workspace,
            }
        ),
    )


def prepare_quantum_lattice(
    specification: QuantumLatticeSpecification,
    plan_or_resources: QuantumLatticeCompilerPlan | QuantumLatticeResourcePolicy,
    /,
) -> PreparedQuantumLattice:
    plan = (
        plan_or_resources
        if isinstance(plan_or_resources, QuantumLatticeCompilerPlan)
        else plan_quantum_lattice(specification, plan_or_resources)
    )
    _validate_plan(specification, plan)
    return _prepared(specification, plan, numeric_version=0, refresh_count=0)


def refresh_quantum_lattice(
    prepared: PreparedQuantumLattice,
    specification: QuantumLatticeSpecification,
    /,
) -> PreparedQuantumLattice:
    if not isinstance(prepared, PreparedQuantumLattice):
        raise TypeError("prepared must be PreparedQuantumLattice.")
    _validate_plan(specification, prepared.plan)
    return _prepared(
        specification,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
        prepared_id=prepared.prepared_id,
    )


def _validate_plan(
    specification: QuantumLatticeSpecification,
    plan: QuantumLatticeCompilerPlan,
    /,
) -> None:
    if not isinstance(plan, QuantumLatticeCompilerPlan):
        raise TypeError("Expected QuantumLatticeCompilerPlan or resource policy.")
    if plan.specification_id != specification.specification_id:
        raise ValueError("Compiler plan does not match the lattice symbolic structure.")
    replanned = plan_quantum_lattice(specification, plan.resources)
    if replanned.plan_id != plan.plan_id:
        raise ValueError("Compiler plan resource estimates no longer match.")


def _expanded_monomials(term: QuantumLatticeTerm, /) -> tuple[CompiledMonomial, ...]:
    def build(
        factors: tuple[LocalOperatorPlan, ...], coefficient: Array, adjoint: bool
    ) -> CompiledMonomial:
        changes: dict[str, int] = {}
        for factor in factors:
            for label, delta in zip(
                factor.space.charge_labels, factor.charge_delta, strict=True
            ):
                changes[label] = changes.get(label, 0) + delta
        capacity = prod(factor.space.dimension for factor in factors)
        return CompiledMonomial(
            factors=factors,
            coefficient=coefficient,
            charge_delta=tuple(sorted(changes.items())),
            branch_capacity=capacity,
            source_term_id=term.term_id,
            adjoint_component=adjoint,
            monomial_id=canonical_fingerprint(
                {
                    "kind": "compiled-quantum-lattice-monomial",
                    "term": term.term_id,
                    "adjoint": adjoint,
                }
            ),
        )

    direct = build(term.factors, term.coefficient, False)
    if not term.add_adjoint:
        return (direct,)
    adjoint = build(
        tuple(factor.adjoint() for factor in reversed(term.factors)),
        jnp.conj(term.coefficient),
        True,
    )
    return direct, adjoint


def _prepared(
    specification: QuantumLatticeSpecification,
    plan: QuantumLatticeCompilerPlan,
    *,
    numeric_version: Array | int,
    refresh_count: Array | int,
    prepared_id: str | None = None,
) -> PreparedQuantumLattice:
    monomials = tuple(
        monomial for term in specification.terms for monomial in _expanded_monomials(term)
    )
    arrays = tuple(
        (monomial.coefficient,) + tuple(factor.matrix for factor in monomial.factors)
        for monomial in monomials
    )
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-quantum-lattice",
                "plan": plan.plan_id,
                "numeric": array_tree_fingerprint(arrays),
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedQuantumLattice(
        specification=specification,
        monomials=monomials,
        plan=plan,
        prepared_id=identifier,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
    )


def certify_charge_map(
    prepared: PreparedQuantumLattice, charge_map: SectorChargeMap, /
) -> ChargeMapCertification:
    if not isinstance(prepared, PreparedQuantumLattice):
        raise TypeError("prepared must be PreparedQuantumLattice.")
    if not isinstance(charge_map, SectorChargeMap):
        raise TypeError("charge_map must be SectorChargeMap.")
    if (
        charge_map.source.site_ids != prepared.specification.site_ids
        or charge_map.source.site_dimensions != prepared.specification.local_dimensions
    ):
        raise ValueError(
            "Sector coordinate layout does not match the lattice specification."
        )
    labels = prepared.specification.charge_labels
    if charge_map.charge_label not in labels:
        raise ValueError("Sector charge label is absent from the lattice specification.")
    expected = np.asarray(
        [
            charge_map.charge_delta if label == charge_map.charge_label else 0
            for label in labels
        ],
        dtype=np.int32,
    )
    observed = np.asarray(
        [
            [dict(monomial.charge_delta).get(label, 0) for label in labels]
            for monomial in prepared.monomials
        ],
        dtype=np.int32,
    )
    valid = np.all(observed == expected[None, :], axis=1)
    accepted = bool(np.all(valid))
    return ChargeMapCertification(
        observed_deltas=jnp.asarray(observed),
        term_valid=jnp.asarray(valid),
        accepted=jnp.asarray(accepted),
        charge_labels=labels,
        map_id=charge_map.map_id,
        prepared_id=prepared.prepared_id,
        certification_id=canonical_fingerprint(
            {
                "kind": "quantum-lattice-charge-map-certification",
                "prepared": prepared.prepared_id,
                "map": charge_map.map_id,
                "observed": array_tree_fingerprint(observed),
            }
        ),
    )


__all__ = [
    "ChargeMapCertification",
    "CompiledMonomial",
    "PreparedQuantumLattice",
    "QuantumLatticeCompilerPlan",
    "QuantumLatticeResourcePolicy",
    "certify_charge_map",
    "plan_quantum_lattice",
    "prepare_quantum_lattice",
    "refresh_quantum_lattice",
]
