#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._resources import AlgebraResourceBudget
from ._structure import AlgebraRationalMap, AlgebraRationalVector, AlgebraStructureTable


AlgebraClaimStatus: TypeAlias = Literal["proven", "disproven", "unknown"]
AlgebraClaimSource: TypeAlias = Literal[
    "exact_basis_audit",
    "family_construction",
    "explicit_witness",
    "unavailable",
]


class AlgebraClaimEvidence(StrictModule, NonTrainableState):
    property_name: str = eqx.field(static=True)
    status: AlgebraClaimStatus = eqx.field(static=True)
    source: AlgebraClaimSource = eqx.field(static=True)
    witness: tuple[str, ...] = eqx.field(static=True)
    work: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        property_name: str,
        status: AlgebraClaimStatus,
        source: AlgebraClaimSource,
        /,
        *,
        witness: Sequence[str] = (),
        work: int = 0,
    ):
        name = str(property_name)
        if not name:
            raise ValueError("Algebra property name must be non-empty.")
        if status not in ("proven", "disproven", "unknown"):
            raise ValueError("Unknown algebra claim status.")
        if source not in (
            "exact_basis_audit",
            "family_construction",
            "explicit_witness",
            "unavailable",
        ):
            raise ValueError("Unknown algebra claim source.")
        work_ = int(work)
        if work_ < 0:
            raise ValueError("Algebra claim work must be nonnegative.")
        witness_ = tuple(str(value) for value in witness)
        if any(not value for value in witness_):
            raise ValueError("Algebra witnesses must contain non-empty labels.")
        self.property_name = name
        self.status = status
        self.source = source
        self.witness = witness_
        self.work = work_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "algebra-claim-evidence-v1",
                "property": name,
                "status": status,
                "source": source,
                "witness": list(witness_),
                "work": work_,
            }
        )


class AlgebraPropertyEvidence(StrictModule, NonTrainableState):
    claims: tuple[AlgebraClaimEvidence, ...]
    evidence_id: str = eqx.field(static=True)

    def __init__(self, claims: Sequence[AlgebraClaimEvidence], /):
        values = tuple(claims)
        if not values or any(
            not isinstance(value, AlgebraClaimEvidence) for value in values
        ):
            raise TypeError("Algebra property evidence requires claim records.")
        names = tuple(value.property_name for value in values)
        if len(set(names)) != len(names):
            raise ValueError("Algebra property claims must be unique.")
        self.claims = values
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "algebra-property-evidence-v1",
                "claims": [value.evidence_id for value in values],
            }
        )

    def claim(self, property_name: str, /) -> AlgebraClaimEvidence:
        for value in self.claims:
            if value.property_name == property_name:
                return value
        raise KeyError(f"Unknown algebra property {property_name!r}.")

    def proven(self, property_name: str, /) -> bool:
        return self.claim(property_name).status == "proven"


_ZERO = Fraction(0)


def _basis(dimension: int, position: int, /) -> tuple[Fraction, ...]:
    return tuple(Fraction(int(index == position)) for index in range(dimension))


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _zero(value) -> bool:
    return all(entry == _ZERO for entry in value)


def _claim_from_witness(name, witness, work):
    return AlgebraClaimEvidence(
        name,
        "proven" if witness is None else "disproven",
        "exact_basis_audit" if witness is None else "explicit_witness",
        witness=() if witness is None else witness,
        work=work,
    )


def audit_algebra_properties(
    table: AlgebraStructureTable,
    basis_ids: Sequence[str],
    unit: AlgebraRationalVector,
    conjugation: AlgebraRationalMap,
    budget: AlgebraResourceBudget,
    /,
    *,
    family_claims: Mapping[
        str,
        tuple[AlgebraClaimStatus, AlgebraClaimSource, Sequence[str]]
        | tuple[AlgebraClaimStatus, AlgebraClaimSource, Sequence[str], int],
    ]
    | None = None,
) -> AlgebraPropertyEvidence:
    dimension = table.coordinate_dimension
    labels = tuple(str(value) for value in basis_ids)
    if (
        len(labels) != dimension
        or len(unit.entries) != dimension
        or conjugation.dimension != dimension
    ):
        raise ValueError("Algebra audit metadata dimensions do not match.")
    audit_work = dimension**3 * 8 + dimension**2 * 8 + dimension * 4
    budget.admit_audit(audit_work)
    vectors = tuple(_basis(dimension, position) for position in range(dimension))
    unit_vector = unit.fractions

    unital_witness = None
    for index_, value in enumerate(vectors):
        if (
            table.multiply(unit_vector, value) != value
            or table.multiply(value, unit_vector) != value
        ):
            unital_witness = (labels[index_],)
            break

    commutative_witness = None
    for left in range(dimension):
        for right in range(dimension):
            if table.multiply(vectors[left], vectors[right]) != table.multiply(
                vectors[right], vectors[left]
            ):
                commutative_witness = (labels[left], labels[right])
                break
        if commutative_witness is not None:
            break

    associative_witness = None
    left_alternative_witness = None
    right_alternative_witness = None
    flexible_witness = None
    for left in range(dimension):
        for middle in range(dimension):
            for right in range(dimension):
                associator = table.associator(
                    vectors[left], vectors[middle], vectors[right]
                )
                if associative_witness is None and not _zero(associator):
                    associative_witness = (labels[left], labels[middle], labels[right])
                if left_alternative_witness is None:
                    swapped = table.associator(
                        vectors[middle], vectors[left], vectors[right]
                    )
                    if not _zero(_add(associator, swapped)):
                        left_alternative_witness = (
                            labels[left],
                            labels[middle],
                            labels[right],
                        )
                if right_alternative_witness is None:
                    swapped = table.associator(
                        vectors[left], vectors[right], vectors[middle]
                    )
                    if not _zero(_add(associator, swapped)):
                        right_alternative_witness = (
                            labels[left],
                            labels[middle],
                            labels[right],
                        )
                if flexible_witness is None:
                    reversed_outer = table.associator(
                        vectors[right], vectors[middle], vectors[left]
                    )
                    if not _zero(_add(associator, reversed_outer)):
                        flexible_witness = (labels[left], labels[middle], labels[right])

    involutive_witness = None
    anti_witness = None
    for left in range(dimension):
        value = conjugation.apply(conjugation.apply(vectors[left]))
        if involutive_witness is None and value != vectors[left]:
            involutive_witness = (labels[left],)
        for right in range(dimension):
            product = table.multiply(vectors[left], vectors[right])
            expected = table.multiply(
                conjugation.apply(vectors[right]),
                conjugation.apply(vectors[left]),
            )
            if anti_witness is None and conjugation.apply(product) != expected:
                anti_witness = (labels[left], labels[right])

    claims = {
        "unital": _claim_from_witness("unital", unital_witness, audit_work),
        "commutative": _claim_from_witness(
            "commutative", commutative_witness, audit_work
        ),
        "associative": _claim_from_witness(
            "associative", associative_witness, audit_work
        ),
        "left_alternative": _claim_from_witness(
            "left_alternative", left_alternative_witness, audit_work
        ),
        "right_alternative": _claim_from_witness(
            "right_alternative", right_alternative_witness, audit_work
        ),
        "flexible": _claim_from_witness("flexible", flexible_witness, audit_work),
        "conjugation_involutive": _claim_from_witness(
            "conjugation_involutive", involutive_witness, audit_work
        ),
        "conjugation_anti_automorphism": _claim_from_witness(
            "conjugation_anti_automorphism", anti_witness, audit_work
        ),
    }
    alternative = (
        claims["left_alternative"].status == "proven"
        and claims["right_alternative"].status == "proven"
    )
    claims["alternative"] = AlgebraClaimEvidence(
        "alternative",
        "proven" if alternative else "disproven",
        "exact_basis_audit" if alternative else "explicit_witness",
        witness=()
        if alternative
        else (left_alternative_witness or right_alternative_witness or ("unknown",)),
        work=audit_work,
    )
    claims["power_associative"] = AlgebraClaimEvidence(
        "power_associative",
        "proven"
        if alternative or claims["associative"].status == "proven"
        else "unknown",
        "exact_basis_audit"
        if alternative or claims["associative"].status == "proven"
        else "unavailable",
        work=audit_work,
    )
    defaults = (
        "division_algebra",
        "has_zero_divisors",
        "positive_norm",
        "norm_multiplicative",
    )
    for name in defaults:
        claims[name] = AlgebraClaimEvidence(name, "unknown", "unavailable")
    for name, claim in ({} if family_claims is None else family_claims).items():
        if len(claim) == 3:
            status, source, witness = claim
            work = 0
        else:
            status, source, witness, work = claim
        claims[name] = AlgebraClaimEvidence(
            name,
            status,
            source,
            witness=witness,
            work=work,
        )
    return AlgebraPropertyEvidence(tuple(claims[name] for name in sorted(claims)))


__all__ = [
    "AlgebraClaimEvidence",
    "AlgebraClaimSource",
    "AlgebraClaimStatus",
    "AlgebraPropertyEvidence",
    "audit_algebra_properties",
]
