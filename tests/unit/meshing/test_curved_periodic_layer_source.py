# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact source identities for the corrected and immutable original W09 cases."""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from tools._hybrid_layer_source import (
    AuthoredPeriodicLayerSource,
    corrected_periodic_layer_source,
    ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST,
    ORIGINAL_CURVED_PERIODIC_SOURCE_ID,
    ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION,
    ORIGINAL_PERIODIC_TRACE_RESIDUALS,
    PERIOD,
)
from tools._meshing_cases import load_curved_periodic_source_registration
from tools.meshing_qualification import (
    _corpus_profiles,
    _hybrid_qualification_original_curved_refusal_profile,
    _hybrid_qualification_profile_contracts,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="corrected periodic layer source requires meshcore"
)


@pytest.fixture(scope="module")
def corrected_source() -> tuple[
    AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult
]:
    limits = phx.meshing.MeshingLimits(maximum_wall_seconds=120.0)
    authored = corrected_periodic_layer_source(limits=limits)
    result = (
        phx.meshing.NativeMeshingProvider(authored.options)
        .plan(
            authored.source,
            authored.specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    return authored, result


def _exact_difference(first: np.ndarray, second: np.ndarray) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(float(right)) - Fraction(float(left))
        for left, right in zip(first, second, strict=True)
    )


def test_corrected_source_authors_wall_and_coefficient_mates_by_unit_translation(
    corrected_source: tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult],
) -> None:
    authored, _ = corrected_source
    topology = authored.wall.periodic_topology
    if topology is None or authored.source.mapped_domain is None:
        raise AssertionError(
            "The corrected source requires explicit wall and mapped owners."
        )
    points = np.asarray(authored.wall.coordinates, dtype=np.float64)
    roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    for row in range(points.shape[0]):
        expected = tuple(Fraction(float(value)) for value in shifts[row, 0] * PERIOD)
        assert _exact_difference(points[roots[row]], points[row]) == expected

    counts = Counter(authored.coefficient_orbit_ids)
    assert sorted(counts.values()).count(2) >= 3
    groups: dict[str, list[int]] = defaultdict(list)
    for row, orbit in enumerate(authored.coefficient_orbit_ids):
        groups[orbit].append(row)
    geometry = authored.source.mapped_domain.source_geometry
    bank = np.asarray(geometry.coordinates, dtype=np.float64)
    start = authored.source.layers.mesh.coordinates.shape[0]
    count = len(authored.coefficient_orbit_ids)
    for rows in groups.values():
        if len(rows) != 2:
            continue
        first, second = sorted(rows, key=lambda row: bank[start + row, 0])
        assert _exact_difference(bank[start + first], bank[start + second]) == (
            Fraction(1),
            Fraction(0),
            Fraction(0),
        )
        assert _exact_difference(
            bank[start + count + first], bank[start + count + second]
        ) == (Fraction(1), Fraction(0), Fraction(0))


def test_corrected_source_passes_exact_periodic_hybrid_certification(
    corrected_source: tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult],
) -> None:
    authored, result = corrected_source
    certificate = result.certification
    assert certificate is not None and certificate.passed
    assert certificate.embedding is not None
    assert not any(
        finding.check == "periodic_mapped_trace_mismatch"
        for finding in certificate.embedding.findings
    )
    assert "periodic_layer_column_graph_source" in certificate.embedding.evaluated_checks
    assert (
        "periodic_layer_column_affine_reference_contact"
        in certificate.embedding.evaluated_checks
    )
    assert result.mesh.periodic_topology is not None
    assert {block.cell_kind for block in result.mesh.blocks} == {"prism", "tetrahedron"}
    assert authored.source_id != ORIGINAL_CURVED_PERIODIC_SOURCE_ID
    assert authored.source_revision != ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION
    assert authored.source_digest != ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST


def test_corrected_source_archive_restores_distinct_identity(
    tmp_path: Path,
    corrected_source: tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult],
) -> None:
    authored, result = corrected_source
    certificate = result.certification
    if certificate is None:
        raise AssertionError("The corrected source must be certified before persistence.")
    records = {
        "certification_inputs": certificate.request,
        "report": certificate,
        "associations": result.associations,
        "generation_source": authored.source,
        "generation_specification": authored.specification,
        "generation_options": authored.options,
        "generation_part": phx.meshing.MeshPart("corrected-periodic-layer-core", result),
    }
    corrected = write_meshing_source_closure(tmp_path / "corrected", records)
    assert corrected.content_id != ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST
    restored = read_meshing_source_closure(
        corrected.path, expected_content_id=corrected.content_id
    )
    reopened = restored["generation_source"]
    assert reopened.source_id == authored.source_id
    assert reopened.source_revision == authored.source_revision
    assert reopened.binding_id == authored.source.binding_id
    mapped = reopened.mapped_domain
    if mapped is None:
        raise AssertionError("The restored corrected source lost its mapped owner.")
    assert all(
        getattr(element, "fiber_graph", False)
        for element in mapped.source_geometry.elements
    )


def test_original_source_remains_an_exact_periodic_refusal() -> None:
    owners = _hybrid_qualification_original_curved_refusal_profile()
    evidence = owners["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["source_id"] == ORIGINAL_CURVED_PERIODIC_SOURCE_ID
    assert evidence["original_top_trace_x_residuals"] == list(
        ORIGINAL_PERIODIC_TRACE_RESIDUALS
    )
    assert all(Fraction(value) != 0 for value in ORIGINAL_PERIODIC_TRACE_RESIDUALS)
    assert evidence["source_bits_preserved"] is True
    assert "Exact periodic certification refused" in str(evidence["rejection"])


def test_reviewed_source_registration_binds_distinct_revisions(
    corrected_source: tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult],
) -> None:
    authored, _ = corrected_source
    registration = load_curved_periodic_source_registration()
    sources = registration["sources"]
    assert isinstance(sources, list)
    records: list[dict[str, object]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise AssertionError("Source registrations must be mappings.")
        records.append(source)
    by_role = {str(source["role"]): source for source in records}
    assert set(by_role) == {"positive", "negative-refusal"}
    positive = by_role["positive"]
    negative = by_role["negative-refusal"]
    assert (
        positive["source_id"],
        positive["source_revision"],
        positive["source_digest"],
    ) == (authored.source_id, authored.source_revision, authored.source_digest)
    assert positive["orbit_ids"] == sorted(
        set((*authored.wall_orbit_ids, *authored.coefficient_orbit_ids))
    )
    artifact = positive["source_artifact"]
    assert isinstance(artifact, dict)
    assert artifact["fiber_graph"] is True
    mapped = authored.source.mapped_domain
    if mapped is None:
        raise AssertionError("The registered corrected source lost its mapped domain.")
    assert artifact["mapped_domain_id"] == mapped.domain_id
    assert artifact["source_binding_id"] == authored.source.binding_id
    assert negative["source_id"] == ORIGINAL_CURVED_PERIODIC_SOURCE_ID
    assert negative["source_revision"] == ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION
    assert negative["source_digest"] == ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST


def test_corrected_and_original_profiles_have_distinct_corpus_admission() -> None:
    contracts = _hybrid_qualification_profile_contracts()
    corrected = contracts["curved-periodic-layer-core"]
    original = contracts["original-curved-periodic-refusal"]
    assert corrected["expected_admission"] == "positive"
    assert corrected["full_workflow"] is True
    assert original["expected_admission"] == "refusal"
    assert original["expected_outcome"] == "expected-refusal"
    profiles = _corpus_profiles()
    positive = profiles["native-hybrid-lifecycle:curved-periodic-layer-core"]
    negative = profiles["native-hybrid-lifecycle:original-curved-periodic-refusal"]
    assert (
        positive["source_reference"]["source_id"]
        != negative["source_reference"]["source_id"]
    )
    assert (
        positive["source_reference"]["source_revision"]
        != negative["source_reference"]["source_revision"]
    )
