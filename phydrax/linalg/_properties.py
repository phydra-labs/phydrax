#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule
from ..typing import parse


PropertyEvidence: TypeAlias = Literal[
    "unknown",
    "construction",
    "transformed",
    "verified",
    "asserted",
]


class LinearCapabilityError(ValueError):
    """Raised when a requested operation lacks a declared operator capability."""


class LinearResourceLimitError(LinearCapabilityError):
    """A resource owner refuses a measured or bounded request before growth."""

    resource: str
    limit: int
    requested: int
    completed: int | None
    symbolic_work: int | None
    storage_bytes_upper: int | None

    def __init__(
        self,
        message: str,
        /,
        *,
        resource: str,
        limit: int,
        requested: int,
        completed: int | None = None,
        symbolic_work: int | None = None,
        storage_bytes_upper: int | None = None,
    ) -> None:
        if not resource or any(
            value is not None and value < 0
            for value in (limit, requested, completed, symbolic_work, storage_bytes_upper)
        ):
            raise ValueError(
                "Linear resource refusal requires a named nonnegative allowance and evidence."
            )
        super().__init__(message)
        self.resource = resource
        self.limit = limit
        self.requested = requested
        self.completed = completed
        self.symbolic_work = symbolic_work
        self.storage_bytes_upper = storage_bytes_upper


class OperatorCapabilities(StrictModule):
    """Immutable executable capabilities, separate from mathematical properties."""

    transpose: bool = eqx.field(static=True)
    adjoint: bool = eqx.field(static=True)
    materialize: bool = eqx.field(static=True)
    diagonal_assembly: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        transpose: bool,
        adjoint: bool,
        materialize: bool,
        diagonal_assembly: bool = False,
    ) -> None:
        if adjoint and not transpose:
            raise ValueError("An adjoint capability requires a transpose capability.")
        self.transpose = bool(transpose)
        self.adjoint = bool(adjoint)
        self.materialize = bool(materialize)
        self.diagonal_assembly = bool(diagonal_assembly)


class OperatorProperties(StrictModule):
    """Immutable structural claims with per-claim trust evidence."""

    diagonal: bool = eqx.field(static=True)
    triangular: bool = eqx.field(static=True)
    self_adjoint: bool = eqx.field(static=True)
    positive_definite: bool = eqx.field(static=True)
    positive_semidefinite: bool = eqx.field(static=True)
    block_diagonal: bool = eqx.field(static=True)
    rank: int | None = eqx.field(static=True)
    evidence: tuple[tuple[str, PropertyEvidence], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        diagonal: bool = False,
        triangular: bool = False,
        self_adjoint: bool = False,
        positive_definite: bool = False,
        positive_semidefinite: bool = False,
        block_diagonal: bool = False,
        rank: int | None = None,
        evidence: Mapping[str, PropertyEvidence] | None = None,
    ) -> None:
        rank_ = None if rank is None else int(rank)
        if rank_ is not None and rank_ < 0:
            raise ValueError("rank must be non-negative or None.")
        if (positive_definite or positive_semidefinite) and not self_adjoint:
            raise ValueError(
                "positive_definite and positive_semidefinite require self_adjoint=True."
            )
        if positive_definite:
            positive_semidefinite = True
        claims = {
            "diagonal": bool(diagonal),
            "triangular": bool(triangular),
            "self_adjoint": bool(self_adjoint),
            "positive_definite": bool(positive_definite),
            "positive_semidefinite": bool(positive_semidefinite),
            "block_diagonal": bool(block_diagonal),
            "rank": rank_ is not None,
        }
        supplied = {} if evidence is None else dict(evidence)
        unknown_keys = supplied.keys() - claims.keys()
        if unknown_keys:
            names = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown operator-property evidence keys: {names}.")
        positive_evidence = supplied.get("positive_definite", "unknown")
        semidefinite_evidence = supplied.get("positive_semidefinite", "unknown")
        if positive_definite and positive_evidence != "unknown":
            supplied.setdefault("positive_semidefinite", "transformed")
            semidefinite_evidence = supplied["positive_semidefinite"]
        if positive_evidence != "unknown" or semidefinite_evidence != "unknown":
            supplied.setdefault("self_adjoint", "transformed")
        supplied = {
            name: parse(value, PropertyEvidence, f"evidence[{name!r}]")
            for name, value in supplied.items()
        }
        if any(not claims[name] for name in supplied):
            raise ValueError("Evidence may only be attached to claimed properties.")
        self.diagonal = claims["diagonal"]
        self.triangular = claims["triangular"]
        self.self_adjoint = claims["self_adjoint"]
        self.positive_definite = claims["positive_definite"]
        self.positive_semidefinite = claims["positive_semidefinite"]
        self.block_diagonal = claims["block_diagonal"]
        self.rank = rank_
        self.evidence = tuple(
            (name, supplied.get(name, "unknown"))
            for name, claimed in claims.items()
            if claimed
        )

    def evidence_for(self, property_name: str, /) -> PropertyEvidence:
        """Return evidence for one claimed property, or ``"unknown"``."""
        return dict(self.evidence).get(property_name, "unknown")

    def certifies(self, property_name: str, /) -> bool:
        """Return whether one claim exists and carries non-unknown evidence."""
        claims = {
            "diagonal": self.diagonal,
            "triangular": self.triangular,
            "self_adjoint": self.self_adjoint,
            "positive_definite": self.positive_definite,
            "positive_semidefinite": self.positive_semidefinite,
            "block_diagonal": self.block_diagonal,
            "rank": self.rank is not None,
        }
        if property_name not in claims:
            raise ValueError(f"Unknown operator property {property_name!r}.")
        return claims[property_name] and self.evidence_for(property_name) != "unknown"


__all__ = [
    "LinearCapabilityError",
    "OperatorCapabilities",
    "OperatorProperties",
    "PropertyEvidence",
]
