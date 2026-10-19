#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer contracts for the generated mesh certificate register."""

from __future__ import annotations

import json
from typing import get_args

import pytest

from phydrax.geometry import (
    mesh_certificate_theorems,
    MeshCertificateCheck,
    MeshCertificateCheckContract,
    MeshCertificateCitation,
    MeshCertificateFinding,
    MeshCertificatePremise,
    MeshCertificatePremiseTag,
    MeshCertificateTheorem,
    MeshFindingStatus,
)
from phydrax.geometry._mesh_theorems import generate_mesh_certificate_theorem_records


def _row(
    check: str, /, *, statuses: tuple[str, ...] = ("unresolved",)
) -> dict[str, object]:
    return {
        "checks": [check],
        "premises": [
            {
                "tag": "X",
                "statement": "The requested bounded check has not been completed.",
            }
        ],
        "conclusion": "The consumer retains an unresolved finding instead of a certified result.",
        "citation": {
            "kind": "definition",
            "reference": "MeshCertificateFinding status contract",
            "locator": "phydrax.geometry.MeshCertificateFinding",
            "url": "",
        },
        "failure_statuses": list(statuses),
    }


def _documentation(*rows: dict[str, object]) -> str:
    return (
        "Document the check implications.\n\nMesh certificate checks\n-----------------------\n"
        + json.dumps(rows)
        + "\nEnd mesh certificate checks"
    )


def test_installed_register_covers_each_closed_check_once() -> None:
    records = mesh_certificate_theorems()
    assert tuple(value.check for value in records) == tuple(
        sorted(get_args(MeshCertificateCheck.__value__))
    )
    assert all(value.contracts for value in records)


def test_rectangular_p_matrix_obligation_cites_its_univalence_result() -> None:
    record = next(
        value
        for value in mesh_certificate_theorems()
        if value.check == "mapped_box_p_matrix_univalence"
    )
    assert record.contracts
    assert record.failure_statuses == ()
    for contract in record.contracts:
        assert contract.citation.kind == "mathematical"
        assert contract.citation.url.endswith("10.1007/BF01360282")
        assert "theorem4" in contract.citation.locator.replace(" ", "").lower()
        assert {value.tag for value in contract.premises} == set(
            get_args(MeshCertificatePremiseTag.__value__)
        )


@pytest.mark.parametrize("status", get_args(MeshFindingStatus), ids=str)
def test_declaration_only_check_retains_caller_declared_status(
    status: MeshFindingStatus,
) -> None:
    finding = MeshCertificateFinding("maximum_samples", status, "mesh")
    record = next(
        value for value in mesh_certificate_theorems() if value.check == finding.check
    )
    assert finding.status in record.failure_statuses
    assert all(value.citation.kind == "definition" for value in record.contracts)
    assert all(
        premise.tag == "X" for value in record.contracts for premise in value.premises
    )


@pytest.mark.parametrize(
    ("check", "status"),
    (
        ("exact_source_resource_budget", "unresolved"),
        ("mapped_domain_source_identity", "violated"),
    ),
    ids=("resource-refusal", "source-identity"),
)
def test_administrative_check_cites_its_definition(
    check: MeshCertificateCheck, status: str
) -> None:
    record = next(value for value in mesh_certificate_theorems() if value.check == check)
    assert record.failure_statuses == (status,)
    assert all(value.citation.kind == "definition" for value in record.contracts)
    assert all(
        premise.tag != "T" for value in record.contracts for premise in value.premises
    )


@pytest.mark.parametrize("check", get_args(MeshCertificateCheck.__value__), ids=str)
def test_registered_check_retains_traceable_contracts(
    check: MeshCertificateCheck,
) -> None:
    record = next(value for value in mesh_certificate_theorems() if value.check == check)
    statuses = set()
    for contract in record.contracts:
        assert contract.owner.startswith("phydrax.")
        assert contract.documentation_owner.startswith("phydrax.")
        assert contract.premises
        assert all(
            value.tag in get_args(MeshCertificatePremiseTag.__value__) and value.statement
            for value in contract.premises
        )
        assert contract.conclusion
        assert contract.citation.reference
        assert contract.citation.locator
        if contract.citation.kind == "mathematical":
            assert contract.citation.url.startswith(("https://", "http://"))
        if any(value.tag == "T" for value in contract.premises):
            assert contract.citation.kind == "mathematical"
        statuses.update(contract.failure_statuses)
    assert record.failure_statuses == tuple(sorted(statuses))


def test_register_generation_is_independent_of_document_order() -> None:
    documents = (
        ("phydrax.geometry.first", _documentation(_row("boundary_contact"))),
        ("phydrax.geometry.second", _documentation(_row("cell_contact"))),
    )
    first = generate_mesh_certificate_theorem_records(
        ("cell_contact", "boundary_contact"), documents
    )
    second = generate_mesh_certificate_theorem_records(
        ("boundary_contact", "cell_contact"), tuple(reversed(documents))
    )
    assert tuple(value.to_record() for value in first) == tuple(
        value.to_record() for value in second
    )


def test_register_keeps_distinct_consumer_implications() -> None:
    records = generate_mesh_certificate_theorem_records(
        ("boundary_contact",),
        (
            (
                "phydrax.geometry.first",
                _documentation(_row("boundary_contact", statuses=("violated",))),
            ),
            ("phydrax.geometry.second", _documentation(_row("boundary_contact"))),
        ),
    )
    assert len(records) == 1
    assert tuple(value.owner for value in records[0].contracts) == (
        "phydrax.geometry.first",
        "phydrax.geometry.second",
    )
    assert tuple(value.failure_statuses for value in records[0].contracts) == (
        ("violated",),
        ("unresolved",),
    )
    assert records[0].failure_statuses == ("unresolved", "violated")


def test_success_obligation_does_not_invent_a_failure_status() -> None:
    records = generate_mesh_certificate_theorem_records(
        ("facet_pairing",),
        (("phydrax.geometry.first", _documentation(_row("facet_pairing", statuses=()))),),
    )
    assert records[0].failure_statuses == ()
    assert records[0].contracts[0].failure_statuses == ()


def test_route_refusal_is_not_translated_to_a_finding_status() -> None:
    row = _row("mapped_boundary_injectivity", statuses=("premise_unproven",))
    row["conclusion"] = (
        "The boundary-degree route declines this hypothesis; pairwise cell contacts "
        "may still certify the mesh."
    )
    records = generate_mesh_certificate_theorem_records(
        ("mapped_boundary_injectivity",),
        (("phydrax.geometry.first", _documentation(row)),),
    )
    assert records[0].failure_statuses == ("premise_unproven",)
    assert records[0].contracts[0].conclusion == row["conclusion"]


def test_successful_route_status_is_not_a_failure_outcome() -> None:
    with pytest.raises(ValueError, match="MeshCertificateCheckContract.failure_statuses"):
        generate_mesh_certificate_theorem_records(
            ("mapped_boundary_injectivity",),
            (
                (
                    "phydrax.geometry.first",
                    _documentation(
                        _row("mapped_boundary_injectivity", statuses=("embedded",))
                    ),
                ),
            ),
        )


def test_missing_check_refuses_generation() -> None:
    with pytest.raises(ValueError, match="MeshCertificateCheck.metadata"):
        generate_mesh_certificate_theorem_records(
            ("boundary_contact", "cell_contact"),
            (("phydrax.geometry.first", _documentation(_row("boundary_contact"))),),
        )


def test_metadata_outside_selected_checks_refuses_generation() -> None:
    with pytest.raises(ValueError, match="phydrax.geometry.first.checks"):
        generate_mesh_certificate_theorem_records(
            ("boundary_contact",),
            (("phydrax.geometry.first", _documentation(_row("cell_contact"))),),
        )


def test_repeated_owner_check_refuses_generation() -> None:
    with pytest.raises(ValueError, match="MeshCertificateTheorem.contracts"):
        generate_mesh_certificate_theorem_records(
            ("boundary_contact",),
            (
                (
                    "phydrax.geometry.first",
                    _documentation(_row("boundary_contact"), _row("boundary_contact")),
                ),
            ),
        )


def test_unsupported_finding_status_refuses_generation() -> None:
    with pytest.raises(ValueError, match="MeshCertificateCheckContract.failure_statuses"):
        generate_mesh_certificate_theorem_records(
            ("boundary_contact",),
            (
                (
                    "phydrax.geometry.first",
                    _documentation(_row("boundary_contact", statuses=("certified",))),
                ),
            ),
        )


def test_topological_hypothesis_requires_a_mathematical_citation() -> None:
    row = _row("boundary_contact")
    row["premises"] = [
        {
            "tag": "T",
            "statement": "A planar crosscut separates the disk into two components.",
        }
    ]
    with pytest.raises(ValueError, match="MeshCertificateCheckContract.citation"):
        generate_mesh_certificate_theorem_records(
            ("boundary_contact",),
            (("phydrax.geometry.first", _documentation(row)),),
        )


def test_unterminated_owner_metadata_refuses_generation() -> None:
    with pytest.raises(ValueError, match="phydrax.geometry.first.metadata"):
        generate_mesh_certificate_theorem_records(
            ("boundary_contact",),
            (
                (
                    "phydrax.geometry.first",
                    "Mesh certificate checks\n-----------------------\n[]",
                ),
            ),
        )


def test_record_json_preserves_declared_hypotheses_and_citation() -> None:
    premise = MeshCertificatePremise(
        "T", "Each positively oriented regular preimage contributes local degree +1."
    )
    citation = MeshCertificateCitation(
        "mathematical",
        "Allen Hatcher, Algebraic Topology (2002)",
        "Section 2.2, local degrees and Proposition 2.30",
        "https://pi.math.cornell.edu/~hatcher/AT/AT.pdf",
    )
    contract = MeshCertificateCheckContract(
        "phydrax.geometry.first",
        "phydrax.geometry.first",
        (premise,),
        "The degree equals the number of preimages.",
        citation,
        (),
    )
    record = MeshCertificateTheorem(
        "mapped_planar_jordan_injectivity", (contract,)
    ).to_record()
    assert record == {
        "check": "mapped_planar_jordan_injectivity",
        "failure_statuses": [],
        "contracts": [
            {
                "owner": "phydrax.geometry.first",
                "documentation_owner": "phydrax.geometry.first",
                "premises": [{"tag": "T", "statement": premise.statement}],
                "conclusion": contract.conclusion,
                "citation": {
                    "kind": "mathematical",
                    "reference": citation.reference,
                    "locator": citation.locator,
                    "url": citation.url,
                },
                "failure_statuses": [],
            }
        ],
    }
