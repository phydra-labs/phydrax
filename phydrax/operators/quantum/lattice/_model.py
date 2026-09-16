#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical finite quantum-lattice spaces, local operators, and terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from .._fermionic_fock import FermionModeOrder


LocalStatistics: TypeAlias = Literal["finite", "fermion", "spin", "boson"]


class LocalSpacePlan(StrictModule):
    """One ordered local basis with exact integral conserved charges."""

    charges: Array
    site_id: str = eqx.field(static=True)
    basis_labels: tuple[str, ...] = eqx.field(static=True)
    charge_labels: tuple[str, ...] = eqx.field(static=True)
    statistics: LocalStatistics = eqx.field(static=True)
    fermion_mode_label: str | None = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_id: str,
        basis_labels: Sequence[str],
        charge_labels: Sequence[str],
        charges: ArrayLike,
        /,
        *,
        statistics: LocalStatistics = "finite",
        fermion_mode_label: str | None = None,
    ):
        site = str(site_id)
        labels = tuple(str(value) for value in basis_labels)
        charge_names = tuple(str(value) for value in charge_labels)
        values = np.asarray(charges)
        if not site or not labels or any(not value for value in labels):
            raise ValueError("Local site and basis labels must be non-empty.")
        if len(set(labels)) != len(labels):
            raise ValueError("Local basis labels must be unique.")
        if any(not value for value in charge_names) or len(set(charge_names)) != len(
            charge_names
        ):
            raise ValueError("Local charge labels must be unique and non-empty.")
        if values.shape != (len(labels), len(charge_names)) or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise ValueError(
                "charges must be integral with shape (dimension, charge_count)."
            )
        if statistics not in ("finite", "fermion", "spin", "boson"):
            raise ValueError("Unknown local statistics.")
        mode = None if fermion_mode_label is None else str(fermion_mode_label)
        if statistics == "fermion":
            if len(labels) != 2 or tuple(values[:, 0]) != (0, 1):
                raise ValueError(
                    "Fermion local spaces require empty/occupied charge 0/1."
                )
            if not mode:
                raise ValueError("Fermion local spaces require a fermion_mode_label.")
        elif mode is not None:
            raise ValueError("Only fermion local spaces may carry a fermion mode label.")
        self.charges = jnp.asarray(values, dtype=jnp.int32)
        self.site_id = site
        self.basis_labels = labels
        self.charge_labels = charge_names
        self.statistics = statistics
        self.fermion_mode_label = mode
        self.dimension = len(labels)
        self.space_id = canonical_fingerprint(
            {
                "kind": "quantum-lattice-local-space",
                "site": site,
                "basis": labels,
                "charge_labels": charge_names,
                "charges": array_tree_fingerprint(values),
                "statistics": statistics,
                "fermion_mode": mode,
            }
        )

    @classmethod
    def fermion(
        cls,
        site_id: str,
        mode_label: str,
        /,
        *,
        charge_label: str = "particle-number",
    ) -> LocalSpacePlan:
        return cls(
            site_id,
            ("empty", "occupied"),
            (charge_label,),
            ((0,), (1,)),
            statistics="fermion",
            fermion_mode_label=mode_label,
        )

    @classmethod
    def spin(
        cls,
        site_id: str,
        twice_spin: int,
        /,
        *,
        charge_label: str = "twice-spin-projection",
    ) -> LocalSpacePlan:
        spin = int(twice_spin)
        if spin < 1:
            raise ValueError("twice_spin must be positive.")
        projections = tuple(range(-spin, spin + 1, 2))
        return cls(
            site_id,
            tuple(str(value) for value in projections),
            (charge_label,),
            tuple((value,) for value in projections),
            statistics="spin",
        )

    @classmethod
    def boson(
        cls,
        site_id: str,
        cutoff: int,
        /,
        *,
        charge_label: str = "boson-number",
    ) -> LocalSpacePlan:
        dimension = int(cutoff)
        if dimension < 2:
            raise ValueError(
                "A boson cutoff must retain at least occupations zero and one."
            )
        return cls(
            site_id,
            tuple(str(value) for value in range(dimension)),
            (charge_label,),
            tuple((value,) for value in range(dimension)),
            statistics="boson",
        )


class LocalOperatorPlan(StrictModule):
    """One local matrix with certified charge change and fermion parity."""

    space: LocalSpacePlan
    matrix: Array
    charge_delta: tuple[int, ...] = eqx.field(static=True)
    fermion_parity: int = eqx.field(static=True)
    label: str = eqx.field(static=True)
    support_mask: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: LocalSpacePlan,
        label: str,
        matrix: ArrayLike,
        charge_delta: Sequence[int],
        /,
        *,
        fermion_parity: int | None = None,
    ):
        if not isinstance(space, LocalSpacePlan):
            raise TypeError("space must be LocalSpacePlan.")
        name = str(label)
        value = np.asarray(matrix)
        delta = tuple(int(item) for item in charge_delta)
        if not name:
            raise ValueError("A local operator label must be non-empty.")
        if value.shape != (space.dimension, space.dimension) or np.any(
            ~np.isfinite(value)
        ):
            raise ValueError(
                "Local operator matrices must be finite and square on their space."
            )
        if len(delta) != len(space.charge_labels):
            raise ValueError("charge_delta must align with the local charge labels.")
        support = value != 0
        rows, columns = np.nonzero(support)
        if rows.size == 0:
            raise ValueError("A local operator matrix must have nonzero support.")
        if np.any(
            np.asarray(space.charges)[rows] - np.asarray(space.charges)[columns]
            != np.asarray(delta)[None, :]
        ):
            raise ValueError(
                "Local operator support contradicts its declared charge_delta."
            )
        inferred = abs(delta[0]) % 2 if space.statistics == "fermion" else 0
        parity = inferred if fermion_parity is None else int(fermion_parity)
        if parity not in (0, 1) or parity != inferred:
            raise ValueError(
                "fermion_parity contradicts the local statistics and charge change."
            )
        dtype = np.result_type(value.dtype, np.complex128)
        self.space = space
        self.matrix = jnp.asarray(value, dtype=dtype)
        self.charge_delta = delta
        self.fermion_parity = parity
        self.label = name
        self.support_mask = jnp.asarray(support)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "quantum-lattice-local-operator",
                "space": space.space_id,
                "label": name,
                "charge_delta": delta,
                "fermion_parity": parity,
                "support": array_tree_fingerprint(support),
                "shape": value.shape,
            }
        )

    def adjoint(self, /) -> LocalOperatorPlan:
        return LocalOperatorPlan(
            self.space,
            f"{self.label}-adjoint",
            np.conj(np.asarray(self.matrix).T),
            tuple(-value for value in self.charge_delta),
            fermion_parity=self.fermion_parity,
        )


class QuantumLatticeTerm(StrictModule):
    """Ordered local-operator monomial; the rightmost factor acts first.

    With ``add_adjoint=True`` the represented term is ``c M + conj(c) M†``.
    """

    factors: tuple[LocalOperatorPlan, ...]
    coefficient: Array
    add_adjoint: bool = eqx.field(static=True)
    label: str = eqx.field(static=True)
    charge_delta: tuple[tuple[str, int], ...] = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        factors: Sequence[LocalOperatorPlan],
        /,
        *,
        coefficient: ArrayLike = 1.0,
        add_adjoint: bool = False,
        label: str,
    ):
        values = tuple(factors)
        scalar = jnp.asarray(coefficient)
        name = str(label)
        if not values or any(
            not isinstance(value, LocalOperatorPlan) for value in values
        ):
            raise ValueError("A lattice term requires local operator factors.")
        if scalar.shape != () or not bool(jnp.isfinite(scalar)):
            raise ValueError("A lattice-term coefficient must be one finite scalar.")
        if not name:
            raise ValueError("A lattice-term label must be non-empty.")
        changes: dict[str, int] = {}
        capacity = 1
        for factor in values:
            capacity *= factor.space.dimension
            for charge_label, delta in zip(
                factor.space.charge_labels, factor.charge_delta, strict=True
            ):
                changes[charge_label] = changes.get(charge_label, 0) + delta
        self.factors = values
        self.coefficient = scalar.astype(jnp.result_type(scalar.dtype, 1j))
        self.add_adjoint = bool(add_adjoint)
        self.label = name
        self.charge_delta = tuple(sorted(changes.items()))
        self.branch_capacity = capacity * (2 if add_adjoint else 1)
        self.term_id = canonical_fingerprint(
            {
                "kind": "quantum-lattice-term",
                "factors": tuple(value.operator_id for value in values),
                "add_adjoint": self.add_adjoint,
                "label": name,
            }
        )


class QuantumLatticeSpecification(StrictModule):
    """Canonical ordered local spaces and terms for one finite lattice."""

    spaces: tuple[LocalSpacePlan, ...]
    terms: tuple[QuantumLatticeTerm, ...]
    fermion_mode_order: FermionModeOrder | None = eqx.field(static=True)
    site_ids: tuple[str, ...] = eqx.field(static=True)
    local_dimensions: tuple[int, ...] = eqx.field(static=True)
    charge_labels: tuple[str, ...] = eqx.field(static=True)
    self_adjoint: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    def __init__(
        self,
        spaces: Sequence[LocalSpacePlan],
        terms: Sequence[QuantumLatticeTerm],
        /,
        *,
        fermion_mode_order: FermionModeOrder | None = None,
    ):
        local_spaces = tuple(spaces)
        local_terms = tuple(terms)
        if not local_spaces or any(
            not isinstance(space, LocalSpacePlan) for space in local_spaces
        ):
            raise ValueError("A lattice specification requires local spaces.")
        if not local_terms or any(
            not isinstance(term, QuantumLatticeTerm) for term in local_terms
        ):
            raise ValueError("A lattice specification requires local terms.")
        if len({term.term_id for term in local_terms}) != len(local_terms):
            raise ValueError("Duplicate quantum-lattice terms must be coalesced.")
        site_ids = tuple(space.site_id for space in local_spaces)
        if len(set(site_ids)) != len(site_ids):
            raise ValueError("Lattice site IDs must be unique.")
        by_site = {space.site_id: space for space in local_spaces}
        for term in local_terms:
            for factor in term.factors:
                if (
                    factor.space.site_id not in by_site
                    or by_site[factor.space.site_id].space_id != factor.space.space_id
                ):
                    raise ValueError(
                        "Every term factor must use one declared local space."
                    )
        fermion_labels = tuple(
            space.fermion_mode_label
            for space in local_spaces
            if space.statistics == "fermion"
        )
        if len(set(fermion_labels)) != len(fermion_labels):
            raise ValueError("Fermion mode labels must be unique across local spaces.")
        if fermion_labels:
            if not isinstance(fermion_mode_order, FermionModeOrder):
                raise TypeError("Fermionic specifications require FermionModeOrder.")
            if set(fermion_labels) != set(fermion_mode_order.labels):
                raise ValueError(
                    "FermionModeOrder must contain exactly the lattice modes."
                )
        elif fermion_mode_order is not None:
            raise ValueError("A nonfermionic lattice must not carry FermionModeOrder.")
        self_adjoint = all(
            term.add_adjoint
            or _manifestly_self_adjoint(term, local_spaces, fermion_mode_order)
            for term in local_terms
        )
        self.spaces = local_spaces
        self.terms = local_terms
        self.fermion_mode_order = fermion_mode_order
        self.site_ids = site_ids
        self.local_dimensions = tuple(space.dimension for space in local_spaces)
        self.charge_labels = tuple(
            sorted({label for space in local_spaces for label in space.charge_labels})
        )
        self.self_adjoint = self_adjoint
        self.specification_id = canonical_fingerprint(
            {
                "kind": "quantum-lattice-specification",
                "spaces": tuple(space.space_id for space in local_spaces),
                "terms": tuple(term.term_id for term in local_terms),
                "fermion_mode_order": (
                    None if fermion_mode_order is None else fermion_mode_order.order_id
                ),
            }
        )

    def space(self, site_id: str, /) -> LocalSpacePlan:
        site = str(site_id)
        if site not in self.site_ids:
            raise KeyError(f"Unknown quantum-lattice site {site!r}.")
        return self.spaces[self.site_ids.index(site)]


def _manifestly_self_adjoint(
    term: QuantumLatticeTerm,
    spaces: tuple[LocalSpacePlan, ...],
    mode_order: FermionModeOrder | None,
    /,
) -> bool:
    coefficient = complex(np.asarray(term.coefficient))
    if abs(coefficient.imag) > 1e-12 * max(abs(coefficient), 1.0):
        return False
    local = [np.eye(space.dimension, dtype=np.complex128) for space in spaces]
    site_ids = tuple(space.site_id for space in spaces)
    for factor in term.factors:
        if factor.fermion_parity:
            if mode_order is None or factor.space.fermion_mode_label is None:
                return False
            ordinal = mode_order.ordinal(factor.space.fermion_mode_label)
            predecessors = set(mode_order.labels[:ordinal])
            for index, space in enumerate(spaces):
                if space.fermion_mode_label in predecessors:
                    local[index] = local[index] @ np.diag((1.0, -1.0))
        site = site_ids.index(factor.space.site_id)
        local[site] = local[site] @ np.asarray(factor.matrix)
    return all(
        np.allclose(value, np.conj(value.T), rtol=1e-12, atol=1e-12) for value in local
    )


__all__ = [
    "LocalOperatorPlan",
    "LocalSpacePlan",
    "LocalStatistics",
    "QuantumLatticeSpecification",
    "QuantumLatticeTerm",
]
