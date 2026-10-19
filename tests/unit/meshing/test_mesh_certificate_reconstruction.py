#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stored certificate reconstruction preserves proof identity and refuses corruption."""

from dataclasses import fields
from typing import Any

import pytest

from phydrax._model import model_from_structure_recipe, model_structure_recipe
from phydrax.geometry import GlobalEmbeddingCertificate
from phydrax.lifecycle import register_meshing_source_artifacts
from tests._support.mesh_certificate_fixtures import (
    embedding,
    TETRAHEDRON,
    two_tetrahedra,
)


def _stored_embedding(
    certificate: GlobalEmbeddingCertificate,
    field_name: str | None = None,
    value: Any = None,
) -> GlobalEmbeddingCertificate:
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(certificate)
    if field_name is not None:
        field_names = tuple(field.name for field in fields(GlobalEmbeddingCertificate))
        recipe["items"][field_names.index(field_name)] = {
            "kind": "literal",
            "value": value,
        }
    restored = model_from_structure_recipe(recipe)
    assert isinstance(restored, GlobalEmbeddingCertificate)
    return restored


def test_global_embedding_reconstructs_stored_proof_and_expression_work() -> None:
    _, _, certificate = embedding(two_tetrahedra(TETRAHEDRON + 5.0))
    stored = _stored_embedding(certificate)
    rebuilt = GlobalEmbeddingCertificate.from_stored_fields(stored)

    assert rebuilt.certificate_id == certificate.certificate_id
    assert rebuilt.binding.binding_id == certificate.binding.binding_id
    assert rebuilt.status == certificate.status == "certified"
    assert rebuilt.evaluated_checks == certificate.evaluated_checks
    assert rebuilt.cell_count == certificate.cell_count == 2
    assert rebuilt.candidate_pair_count == certificate.candidate_pair_count
    assert rebuilt.ray_test_count == certificate.ray_test_count
    assert (
        rebuilt.source_expression_work_units == certificate.source_expression_work_units
    )
    assert (
        rebuilt.source_expression_peak_bytes == certificate.source_expression_peak_bytes
    )


@pytest.mark.parametrize(
    ("field_name", "value", "diagnostic"),
    (
        pytest.param(
            "cell_count", -1, "EmbeddingCounts.cell_count", id="negative-entity-count"
        ),
        pytest.param(
            "cell_count",
            3,
            "GlobalEmbeddingCertificate.stored_fields",
            id="changed-entity-count",
        ),
        pytest.param(
            "status",
            "unresolved",
            "GlobalEmbeddingCertificate.stored_fields",
            id="changed-verdict",
        ),
        pytest.param(
            "certificate_id",
            "foreign-proof",
            "GlobalEmbeddingCertificate.stored_fields",
            id="changed-proof-id",
        ),
        pytest.param(
            "source_expression_work_units",
            -1,
            "EmbeddingExpressionWork.work_units",
            id="negative-phase-work",
        ),
    ),
)
def test_global_embedding_refuses_corrupted_stored_fields(
    field_name: str,
    value: int | str,
    diagnostic: str,
) -> None:
    _, _, certificate = embedding(two_tetrahedra(TETRAHEDRON + 5.0))
    with pytest.raises(ValueError, match=diagnostic):
        stored = _stored_embedding(certificate, field_name, value)
        GlobalEmbeddingCertificate.from_stored_fields(stored)


def test_expression_work_is_reconstructed_without_changing_scientific_proof_identity() -> (
    None
):
    _, _, certificate = embedding(two_tetrahedra(TETRAHEDRON + 5.0))
    work = certificate.source_expression_work_units + 17
    stored = _stored_embedding(certificate, "source_expression_work_units", work)
    rebuilt = GlobalEmbeddingCertificate.from_stored_fields(stored)

    assert rebuilt.certificate_id == certificate.certificate_id
    assert rebuilt.status == certificate.status
    assert rebuilt.source_expression_work_units == work
