#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit periodic one-particle to finite fermion-lattice bridge."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...periodic._family import PeriodicFiniteRealization
from .._fermionic_fock import FermionModeOrder
from ._model import (
    LocalOperatorPlan,
    LocalSpacePlan,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)


class FermionInteractionTerm(StrictModule):
    """Provider-bound ordered CAR monomial added to a periodic one-body model."""

    coefficient: Array
    operations: tuple[tuple[str, str], ...] = eqx.field(static=True)
    add_adjoint: bool = eqx.field(static=True)
    label: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        operations: Sequence[tuple[str, str]],
        coefficient: ArrayLike,
        /,
        *,
        add_adjoint: bool = False,
        label: str,
    ):
        values = tuple((str(mode), str(action)) for mode, action in operations)
        scalar = jnp.asarray(coefficient)
        name = str(label)
        if not values or any(
            not mode or action not in ("create", "annihilate") for mode, action in values
        ):
            raise ValueError(
                "Interaction operations must be create/annihilate CAR symbols."
            )
        if scalar.shape != () or not bool(jnp.isfinite(scalar)) or not name:
            raise ValueError(
                "Interaction coefficient and label must be finite/non-empty."
            )
        self.coefficient = scalar.astype(jnp.result_type(scalar.dtype, 1j))
        self.operations = values
        self.add_adjoint = bool(add_adjoint)
        self.label = name
        self.term_id = canonical_fingerprint(
            {
                "kind": "periodic-fermion-interaction-term",
                "operations": values,
                "add_adjoint": self.add_adjoint,
                "label": name,
            }
        )


class FermionInteractionPlan(StrictModule):
    """Explicit interaction content and provenance; an empty tuple means none."""

    terms: tuple[FermionInteractionTerm, ...]
    provenance_id: str = eqx.field(static=True)
    units_id: str = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    maximum_operations_per_term: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[FermionInteractionTerm],
        /,
        *,
        provenance_id: str,
        units_id: str,
        maximum_terms: int,
        maximum_operations_per_term: int,
    ):
        values = tuple(terms)
        provenance = str(provenance_id)
        units = str(units_id)
        term_limit = int(maximum_terms)
        operation_limit = int(maximum_operations_per_term)
        if any(not isinstance(value, FermionInteractionTerm) for value in values):
            raise TypeError("terms must contain FermionInteractionTerm values.")
        if not provenance or not units or term_limit < 0 or operation_limit < 1:
            raise ValueError("Interaction provenance, units, and limits are invalid.")
        if len(values) > term_limit or any(
            len(value.operations) > operation_limit for value in values
        ):
            raise ValueError(
                "Interactions exceed their explicit term/operation admission."
            )
        self.terms = values
        self.provenance_id = provenance
        self.units_id = units
        self.maximum_terms = term_limit
        self.maximum_operations_per_term = operation_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-fermion-interaction-plan",
                "terms": tuple(value.term_id for value in values),
                "provenance": provenance,
                "units": units,
                "maximum_terms": term_limit,
                "maximum_operations_per_term": operation_limit,
            }
        )


def _fermion_operators(space: LocalSpacePlan, /) -> dict[str, LocalOperatorPlan]:
    create = np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.complex128)
    annihilate = np.conj(create.T)
    return {
        "create": LocalOperatorPlan(space, "create", create, (1,)),
        "annihilate": LocalOperatorPlan(space, "annihilate", annihilate, (-1,)),
    }


def periodic_finite_to_fermion_lattice(
    realization: PeriodicFiniteRealization,
    mode_order: FermionModeOrder,
    realization_mode_labels: Sequence[str],
    interactions: FermionInteractionPlan,
    /,
    *,
    hermiticity_tolerance: float = 1e-10,
) -> QuantumLatticeSpecification:
    """Bridge only an explicitly ordered finite realization plus interactions.

    ``realization_mode_labels`` binds the realization's cell-major coordinates to
    the supplied CAR order. Interactions are mandatory even when explicitly empty.
    """
    if not isinstance(realization, PeriodicFiniteRealization):
        raise TypeError("realization must be PeriodicFiniteRealization.")
    if not isinstance(mode_order, FermionModeOrder):
        raise TypeError("mode_order must be FermionModeOrder.")
    if not isinstance(interactions, FermionInteractionPlan):
        raise TypeError("interactions must be an explicit FermionInteractionPlan.")
    labels = tuple(str(value) for value in realization_mode_labels)
    tolerance = float(hermiticity_tolerance)
    if labels != mode_order.labels:
        raise ValueError(
            "Realization mode labels must exactly equal FermionModeOrder labels."
        )
    if realization.output_size != realization.input_size or realization.input_size != len(
        labels
    ):
        raise ValueError(
            "Periodic finite realization dimensions do not match mode order."
        )
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("hermiticity_tolerance must be finite and non-negative.")
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in mode_order.labels)
    operators = {space.fermion_mode_label: _fermion_operators(space) for space in spaces}
    entries: dict[tuple[int, int], complex] = {}
    for row, column, value in zip(
        np.asarray(realization.row_indices),
        np.asarray(realization.column_indices),
        np.asarray(realization.values),
        strict=True,
    ):
        key = (int(row), int(column))
        entries[key] = entries.get(key, 0.0j) + complex(value)
    scale = max((abs(value) for value in entries.values()), default=1.0)
    for (row, column), value in entries.items():
        reverse = entries.get((column, row), 0.0j)
        if abs(value - np.conj(reverse)) > tolerance * max(scale, 1.0):
            raise ValueError("Periodic one-particle realization is not Hermitian.")
    terms = []
    for row in range(len(labels)):
        diagonal = entries.get((row, row), 0.0j)
        if abs(diagonal.imag) > tolerance * max(scale, 1.0):
            raise ValueError("A Hermitian one-particle diagonal must be real.")
        if abs(diagonal) > 0.0:
            mode = labels[row]
            terms.append(
                QuantumLatticeTerm(
                    (operators[mode]["create"], operators[mode]["annihilate"]),
                    coefficient=diagonal.real,
                    label=f"one-body-diagonal:{mode}",
                )
            )
        for column in range(row + 1, len(labels)):
            value = entries.get((row, column), 0.0j)
            if abs(value) > 0.0:
                terms.append(
                    QuantumLatticeTerm(
                        (
                            operators[labels[row]]["create"],
                            operators[labels[column]]["annihilate"],
                        ),
                        coefficient=value,
                        add_adjoint=True,
                        label=f"one-body-hopping:{labels[row]}:{labels[column]}",
                    )
                )
    for interaction in interactions.terms:
        if any(mode not in operators for mode, _ in interaction.operations):
            raise ValueError(
                "An interaction references a mode absent from FermionModeOrder."
            )
        factors = tuple(
            operators[mode][action] for mode, action in interaction.operations
        )
        terms.append(
            QuantumLatticeTerm(
                factors,
                coefficient=interaction.coefficient,
                add_adjoint=interaction.add_adjoint,
                label=interaction.label,
            )
        )
    if not terms:
        raise ValueError("The periodic bridge produced an empty many-body operator.")
    return QuantumLatticeSpecification(spaces, terms, fermion_mode_order=mode_order)


__all__ = [
    "FermionInteractionPlan",
    "FermionInteractionTerm",
    "periodic_finite_to_fermion_lattice",
]
