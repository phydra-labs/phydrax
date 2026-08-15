#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx

from .._strict import StrictModule
from ._properties import PropertyEvidence


class PreconditionerProperties(StrictModule):
    """Immutable approximate-inverse claims with explicit trust evidence."""

    linear: bool = eqx.field(static=True)
    stationary: bool = eqx.field(static=True)
    self_adjoint: bool = eqx.field(static=True)
    positive_definite: bool = eqx.field(static=True)
    evidence: tuple[tuple[str, PropertyEvidence], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear: bool = False,
        stationary: bool = False,
        self_adjoint: bool = False,
        positive_definite: bool = False,
        evidence: Mapping[str, PropertyEvidence] | None = None,
    ):
        claims = {
            "linear": bool(linear),
            "stationary": bool(stationary),
            "self_adjoint": bool(self_adjoint),
            "positive_definite": bool(positive_definite),
        }
        if claims["self_adjoint"] and not claims["linear"]:
            raise ValueError("A self-adjoint preconditioner must be linear.")
        if claims["positive_definite"] and not all(
            claims[name] for name in ("linear", "stationary", "self_adjoint")
        ):
            raise ValueError(
                "A positive-definite preconditioner must be linear, stationary, "
                "and self-adjoint."
            )
        supplied = {} if evidence is None else dict(evidence)
        unknown = supplied.keys() - claims.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown preconditioner-property evidence keys: {names}.")
        valid = {"unknown", "construction", "transformed", "verified", "asserted"}
        if any(value not in valid for value in supplied.values()):
            raise ValueError("Unknown preconditioner-property evidence.")
        if any(not claims[name] for name in supplied):
            raise ValueError("Evidence may only be attached to claimed properties.")
        positive_evidence = supplied.get("positive_definite", "unknown")
        if positive_evidence != "unknown":
            supplied.setdefault("linear", "transformed")
            supplied.setdefault("stationary", "transformed")
            supplied.setdefault("self_adjoint", "transformed")
        self.linear = claims["linear"]
        self.stationary = claims["stationary"]
        self.self_adjoint = claims["self_adjoint"]
        self.positive_definite = claims["positive_definite"]
        self.evidence = tuple(
            (name, supplied.get(name, "unknown"))
            for name, claimed in claims.items()
            if claimed
        )

    def evidence_for(self, property_name: str, /) -> PropertyEvidence:
        """Return evidence for one claimed property, or ``"unknown"``."""
        if property_name not in (
            "linear",
            "stationary",
            "self_adjoint",
            "positive_definite",
        ):
            raise ValueError(f"Unknown preconditioner property {property_name!r}.")
        return dict(self.evidence).get(property_name, "unknown")

    def certifies(self, property_name: str, /) -> bool:
        """Return whether one claim exists and carries non-unknown evidence."""
        claims = {
            "linear": self.linear,
            "stationary": self.stationary,
            "self_adjoint": self.self_adjoint,
            "positive_definite": self.positive_definite,
        }
        if property_name not in claims:
            raise ValueError(f"Unknown preconditioner property {property_name!r}.")
        return claims[property_name] and self.evidence_for(property_name) != "unknown"


def _preconditioner_properties_payload(
    properties: PreconditionerProperties,
    /,
) -> dict[str, object]:
    return {
        "linear": properties.linear,
        "stationary": properties.stationary,
        "self_adjoint": properties.self_adjoint,
        "positive_definite": properties.positive_definite,
        "evidence": [list(item) for item in properties.evidence],
    }


__all__ = ["PreconditionerProperties"]
