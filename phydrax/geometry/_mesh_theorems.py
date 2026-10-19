#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Describe the documented implications of mesh certificate checks.

The generated register preserves each documented contract separately when a
check has several consumers. N marks certified numerical bounds, X marks
combinatorial or identity predicates, and T marks an external topology result.
An empty failure-status tuple denotes an evaluated success obligation, rather
than a finding code. Administrative predicates cite their defining contract.
Coverage applies to the closed finding/obligation selector, not descriptive
strings stored in a certificate's ``evaluated_checks`` field.
``premise_unproven`` preserves a boundary-degree route refusal; its caller can
still certify the mesh using a different proof. It is not an unresolved
mesh-certificate finding.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import final, get_args, Literal

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..typing import checked, parse
from ._embedding_state import MeshFindingStatus
from ._global_embedding import MappedBoundaryDegreeStatus
from ._mesh_check_names import MeshCertificateCheck


type MeshCertificatePremiseTag = Literal["N", "X", "T"]
type MeshCertificateCitationKind = Literal["definition", "mathematical"]
type MeshCertificateCheckFailureStatus = MeshFindingStatus | MappedBoundaryDegreeStatus

_METADATA_START = "Mesh certificate checks\n-----------------------\n"
_METADATA_END = "End mesh certificate checks"


def _text(value: object, field: str, /, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string; got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be nonempty; got an empty string")
    return value


def _mapping(value: object, field: str, /) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(
            f"{field} must be a JSON object with string keys; got {type(value).__name__}"
        )
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, field: str, /) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array; got {type(value).__name__}")
    return list(value)


def _fields(row: dict[str, object], expected: set[str], field: str, /) -> None:
    if set(row) != expected:
        raise ValueError(
            f"{field} must contain fields {sorted(expected)}; got {sorted(row)}"
        )


@final
class MeshCertificatePremise(StrictModule, NonTrainableState):
    """State one documented hypothesis and its evidence category."""

    tag: MeshCertificatePremiseTag = eqx.field(static=True)
    statement: str = eqx.field(static=True)

    def __init__(self, tag: MeshCertificatePremiseTag, statement: str, /) -> None:
        tag_ = parse(tag, MeshCertificatePremiseTag, "MeshCertificatePremise.tag")
        statement_ = _text(statement, "MeshCertificatePremise.statement")
        self.tag = tag_
        self.statement = statement_

    def to_record(self) -> dict[str, str]:
        """Return the JSON representation of this hypothesis."""
        return {"tag": self.tag, "statement": self.statement}


@final
class MeshCertificateCitation(StrictModule, NonTrainableState):
    """Identify a mathematical result or an administrative definition."""

    kind: MeshCertificateCitationKind = eqx.field(static=True)
    reference: str = eqx.field(static=True)
    locator: str = eqx.field(static=True)
    url: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MeshCertificateCitationKind,
        reference: str,
        locator: str,
        url: str,
        /,
    ) -> None:
        kind_ = parse(kind, MeshCertificateCitationKind, "MeshCertificateCitation.kind")
        reference_ = _text(reference, "MeshCertificateCitation.reference")
        locator_ = _text(locator, "MeshCertificateCitation.locator")
        url_ = _text(
            url, "MeshCertificateCitation.url", allow_empty=kind_ == "definition"
        )
        if url_ and not url_.startswith(("https://", "http://")):
            raise ValueError(
                f"MeshCertificateCitation.url must use HTTP or HTTPS; got {url_!r}"
            )
        self.kind = kind_
        self.reference = reference_
        self.locator = locator_
        self.url = url_

    def to_record(self) -> dict[str, str]:
        """Return the JSON representation of this citation."""
        return {
            "kind": self.kind,
            "reference": self.reference,
            "locator": self.locator,
            "url": self.url,
        }


@final
class MeshCertificateCheckContract(StrictModule, NonTrainableState):
    """Describe one consumer's documented implication and failure statuses."""

    owner: str = eqx.field(static=True)
    documentation_owner: str = eqx.field(static=True)
    premises: tuple[MeshCertificatePremise, ...] = eqx.field(static=True)
    conclusion: str = eqx.field(static=True)
    citation: MeshCertificateCitation = eqx.field(static=True)
    failure_statuses: tuple[MeshCertificateCheckFailureStatus, ...] = eqx.field(
        static=True
    )

    @checked
    def __init__(
        self,
        owner: str,
        documentation_owner: str,
        premises: tuple[MeshCertificatePremise, ...],
        conclusion: str,
        citation: MeshCertificateCitation,
        failure_statuses: tuple[MeshCertificateCheckFailureStatus, ...],
        /,
    ) -> None:
        owner_ = canonical_identifier(owner, "MeshCertificateCheckContract.owner")
        documentation_ = canonical_identifier(
            documentation_owner, "MeshCertificateCheckContract.documentation_owner"
        )
        conclusion_ = _text(conclusion, "MeshCertificateCheckContract.conclusion")
        statuses = tuple(
            sorted(
                {
                    parse(
                        value,
                        MeshCertificateCheckFailureStatus,
                        "MeshCertificateCheckContract.failure_statuses",
                    )
                    for value in failure_statuses
                }
            )
        )
        if "embedded" in statuses:
            raise ValueError(
                "MeshCertificateCheckContract.failure_statuses must contain failure or "
                "route-refusal outcomes; got successful embedded status"
            )
        if not premises:
            raise ValueError(
                "MeshCertificateCheckContract.premises must contain at least one hypothesis; got an empty tuple"
            )
        if (
            any(value.tag == "T" for value in premises)
            and citation.kind != "mathematical"
        ):
            raise ValueError(
                "MeshCertificateCheckContract.citation must cite a mathematical result for T hypotheses; got a definition"
            )
        self.owner = owner_
        self.documentation_owner = documentation_
        self.premises = premises
        self.conclusion = conclusion_
        self.citation = citation
        self.failure_statuses = statuses

    def to_record(self) -> dict[str, object]:
        """Return the JSON representation of this documented implication."""
        return {
            "owner": self.owner,
            "documentation_owner": self.documentation_owner,
            "premises": [value.to_record() for value in self.premises],
            "conclusion": self.conclusion,
            "citation": self.citation.to_record(),
            "failure_statuses": list(self.failure_statuses),
        }


@final
class MeshCertificateTheorem(StrictModule, NonTrainableState):
    """Collect every documented contract for one closed certificate check."""

    check: MeshCertificateCheck = eqx.field(static=True)
    contracts: tuple[MeshCertificateCheckContract, ...] = eqx.field(static=True)
    failure_statuses: tuple[MeshCertificateCheckFailureStatus, ...] = eqx.field(
        static=True
    )

    def __init__(
        self,
        check: MeshCertificateCheck,
        contracts: tuple[MeshCertificateCheckContract, ...],
        /,
    ) -> None:
        check_ = parse(check, MeshCertificateCheck, "MeshCertificateTheorem.check")
        if not contracts:
            raise ValueError(
                "MeshCertificateTheorem.contracts must contain a documented implication; got an empty tuple"
            )
        contracts_ = tuple(
            sorted(contracts, key=lambda value: (value.owner, value.documentation_owner))
        )
        owners = tuple((value.owner, value.documentation_owner) for value in contracts_)
        if len(set(owners)) != len(owners):
            raise ValueError(
                f"MeshCertificateTheorem.contracts must have unique owner/documentation pairs; got {owners}"
            )
        self.check = check_
        self.contracts = contracts_
        self.failure_statuses = tuple(
            sorted({status for value in contracts_ for status in value.failure_statuses})
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON representation of this check's contracts."""
        return {
            "check": self.check,
            "contracts": [value.to_record() for value in self.contracts],
            "failure_statuses": list(self.failure_statuses),
        }


def _contract(
    row: dict[str, object], documentation_owner: str, /
) -> MeshCertificateCheckContract:
    premises = []
    for value in _sequence(row["premises"], "MeshCertificateCheckContract.premises"):
        premise = _mapping(value, "MeshCertificatePremise")
        _fields(premise, {"tag", "statement"}, "MeshCertificatePremise")
        premises.append(
            MeshCertificatePremise(
                parse(
                    premise["tag"],
                    MeshCertificatePremiseTag,
                    "MeshCertificatePremise.tag",
                ),
                _text(premise["statement"], "MeshCertificatePremise.statement"),
            )
        )
    citation = _mapping(row["citation"], "MeshCertificateCitation")
    _fields(citation, {"kind", "reference", "locator", "url"}, "MeshCertificateCitation")
    statuses = tuple(
        parse(
            value,
            MeshCertificateCheckFailureStatus,
            "MeshCertificateCheckContract.failure_statuses",
        )
        for value in _sequence(
            row["failure_statuses"], "MeshCertificateCheckContract.failure_statuses"
        )
    )
    return MeshCertificateCheckContract(
        _text(
            row.get("owner", documentation_owner), "MeshCertificateCheckContract.owner"
        ),
        documentation_owner,
        tuple(premises),
        _text(row["conclusion"], "MeshCertificateCheckContract.conclusion"),
        MeshCertificateCitation(
            parse(
                citation["kind"],
                MeshCertificateCitationKind,
                "MeshCertificateCitation.kind",
            ),
            _text(citation["reference"], "MeshCertificateCitation.reference"),
            _text(citation["locator"], "MeshCertificateCitation.locator"),
            _text(citation["url"], "MeshCertificateCitation.url", allow_empty=True),
        ),
        statuses,
    )


def generate_mesh_certificate_theorem_records(
    checks: tuple[str, ...],
    owner_docstrings: tuple[tuple[str, str], ...],
    /,
) -> tuple[MeshCertificateTheorem, ...]:
    """Generate sorted check records from explicit structured owner docstrings.

    Missing checks, unknown checks, repeated owner/check declarations,
    malformed metadata, and unsupported failure or route statuses refuse generation.
    Several consumers of one check retain separate implications.

    The delimited ``Mesh certificate checks`` section holds a JSON array.
    Each row declares ``checks``, ``premises`` (tag/statement objects),
    ``conclusion``, ``citation`` (kind/reference/locator/url), and
    ``failure_statuses``. An optional ``owner`` identifies a more specific
    source symbol than the docstring containing the row. Only checks with the
    same implication and failure or route outcomes may share one row.
    """
    closed = {parse(value, MeshCertificateCheck, "checks") for value in checks}
    if len(closed) != len(checks):
        raise ValueError(
            f"MeshCertificateTheorem.checks must contain unique selector members; got {checks}"
        )
    contracts: dict[MeshCertificateCheck, list[MeshCertificateCheckContract]] = {
        value: [] for value in closed
    }
    for owner, docstring in sorted(owner_docstrings):
        if _METADATA_START not in docstring:
            continue
        metadata_field = f"{owner}.metadata"
        checks_field = f"{owner}.checks"
        _, _, block = docstring.partition(_METADATA_START)
        payload, ending, _ = block.partition(_METADATA_END)
        if not ending or _METADATA_START in block:
            raise ValueError(
                f"{metadata_field} must contain one terminated check block; got malformed delimiters"
            )
        rows: object = json.loads(payload)
        for value in _sequence(rows, metadata_field):
            row = _mapping(value, metadata_field)
            required = {
                "checks",
                "premises",
                "conclusion",
                "citation",
                "failure_statuses",
            }
            _fields(
                row,
                required | ({"owner"} if "owner" in row else set()),
                metadata_field,
            )
            contract = _contract(row, owner)
            declared = tuple(
                parse(name, MeshCertificateCheck, checks_field)
                for name in _sequence(row["checks"], checks_field)
            )
            if not declared or len(set(declared)) != len(declared):
                raise ValueError(
                    f"{checks_field} must contain unique nonempty check names; got {declared}"
                )
            for check in declared:
                if check not in contracts:
                    raise ValueError(
                        f"{checks_field} must belong to the selected closed set; got {check!r}"
                    )
                contracts[check].append(contract)
    missing = sorted(check for check, values in contracts.items() if not values)
    if missing:
        raise ValueError(
            f"MeshCertificateCheck.metadata must cover every selector member; got missing {missing}"
        )
    return tuple(
        MeshCertificateTheorem(check, tuple(values))
        for check, values in sorted(contracts.items())
    )


@cache
def mesh_certificate_theorems() -> tuple[MeshCertificateTheorem, ...]:
    """Return the installed generated register, sorted by closed check name."""
    payload: object = json.loads(
        files(__package__)
        .joinpath("_mesh_theorem_register.json")
        .read_text(encoding="utf-8")
    )
    root = _mapping(payload, "MeshCertificateTheorem.register")
    _fields(root, {"selector_source", "records"}, "MeshCertificateTheorem.register")
    _text(root["selector_source"], "MeshCertificateTheorem.selector_source")
    records = []
    for value in _sequence(root["records"], "MeshCertificateTheorem.records"):
        row = _mapping(value, "MeshCertificateTheorem")
        _fields(row, {"check", "contracts", "failure_statuses"}, "MeshCertificateTheorem")
        contracts = []
        for item in _sequence(row["contracts"], "MeshCertificateTheorem.contracts"):
            contract = _mapping(item, "MeshCertificateCheckContract")
            _fields(
                contract,
                {
                    "owner",
                    "documentation_owner",
                    "premises",
                    "conclusion",
                    "citation",
                    "failure_statuses",
                },
                "MeshCertificateCheckContract",
            )
            contracts.append(
                _contract(
                    contract,
                    _text(
                        contract["documentation_owner"],
                        "MeshCertificateCheckContract.documentation_owner",
                    ),
                )
            )
        record = MeshCertificateTheorem(
            parse(row["check"], MeshCertificateCheck, "MeshCertificateTheorem.check"),
            tuple(contracts),
        )
        if list(record.failure_statuses) != row["failure_statuses"]:
            raise ValueError(
                f"MeshCertificateTheorem.failure_statuses must equal its documented union; got {row['failure_statuses']}"
            )
        records.append(record)
    names = tuple(value.check for value in records)
    expected = tuple(sorted(get_args(MeshCertificateCheck.__value__)))
    if names != expected:
        raise ValueError(
            f"MeshCertificateTheorem.register must cover the sorted closed selector; got {names}"
        )
    return tuple(records)
