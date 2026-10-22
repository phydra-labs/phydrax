# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Check source identities for the corrected and immutable W09 cases."""

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
from phydrax.meshing.qualification import (
    AuthoredPeriodicLayerSource,
    CURVED_PERIODIC_LAYER_PERIOD,
    ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST,
    ORIGINAL_CURVED_PERIODIC_SOURCE_ID,
    ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION,
    ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS,
    prepare_corrected_periodic_layer_source,
)
from tools._meshing_cases import load_curved_periodic_source_registration
from tools.meshing_qualification._campaign import corpus_phase_evidence
from tools.meshing_qualification._controls import _hybrid_qualification_profile_contracts
from tools.meshing_qualification._records import _corpus_profiles
from tools.meshing_qualification.scenarios.hybrid import (
    _hybrid_qualification_original_curved_refusal_profile,
    _hybrid_record_cold_epochs,
    _HybridContinuationEvidence,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="corrected periodic layer source requires meshcore"
)


@pytest.fixture(scope="module")
def corrected_source() -> tuple[
    AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult
]:
    limits = phx.meshing.MeshingLimits(maximum_wall_seconds=120.0)
    authored = prepare_corrected_periodic_layer_source(limits=limits)
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
            "fixture.source must retain wall topology and mapped domain; got missing owner"
        )
    points = np.asarray(authored.wall.coordinates, dtype=np.float64)
    roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    for row in range(points.shape[0]):
        expected = tuple(
            Fraction(float(value))
            for value in shifts[row, 0] * CURVED_PERIODIC_LAYER_PERIOD
        )
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
        raise AssertionError(
            "fixture.certification must exist before persistence; got None"
        )
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
        raise AssertionError("fixture.mapped_domain must survive restoration; got None")
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
        ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS
    )
    assert all(Fraction(value) != 0 for value in ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS)
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
            raise AssertionError(
                "fixture.source_registration must be a mapping; got incompatible registration"
            )
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
        raise AssertionError(
            "fixture.mapped_domain must exist for registration; got None"
        )
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


def test_cold_continuation_epochs_publish_campaign_phase_evidence() -> None:
    def epoch(index: int, name: str, compiled: bool) -> dict[str, object]:
        stages = {"solver_warm_execution": 0.25}
        if compiled:
            stages.update(
                solver_lowering=0.5, solver_compile=1.5, solver_first_execution=0.75
            )
        return {
            "epoch": index,
            "name": name,
            "result": {"result_id": name},
            "dofs": 9,
            "compiler": {
                "temporary_bytes": 64,
                "output_bytes": 8,
                "generated_code_bytes": 0,
                "warm_samples_seconds": [0.25],
                "compilation_count": int(compiled),
            },
            "stages_seconds": stages,
            "retained_bytes": 128,
        }

    continued = _HybridContinuationEvidence(
        1.0e-12,
        (
            epoch(0, "initial", True),
            epoch(1, "refine", True),
            epoch(2, "coarsen", False),
        ),
        {"result_id": "source"},
    )
    record: dict[str, object] = {"epochs": [], "stages_seconds": {}}
    for cycle in range(2):
        _hybrid_record_cold_epochs(record, continued, cycle)
    epochs = record["epochs"]
    assert isinstance(epochs, list)
    assert [value["epoch"] for value in epochs] == list(range(6))
    assert [value["name"] for value in epochs] == [
        "initial_0",
        "refine_0",
        "coarsen_0",
        "initial_1",
        "refine_1",
        "coarsen_1",
    ]
    assert [value["continuation_epoch"] for value in epochs] == [0, 1, 2] * 2
    compilers = corpus_phase_evidence(record, 20_000)["compiler_epochs"]
    assert [value["compile_seconds"] for value in compilers] == [1.5, 1.5, None] * 2
    assert [value["first_execution_seconds"] for value in compilers] == [
        0.75,
        0.75,
        None,
    ] * 2
    assert all(value["logical_retained_bytes"] == 128 for value in compilers)
    assert all(value["warm_samples_seconds"] == [0.25] for value in compilers)


def test_mapped_source_evidence_reuses_only_current_live_source(
    corrected_source: tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref
    from typing import Any

    import phydrax.geometry._global_embedding as embedding_owner
    import phydrax.geometry._mapped_reference_coverage as source_owner
    from phydrax.geometry._embedding_state import coordinate_certificate_request

    authored, result = corrected_source
    assert result.certification is not None
    evidence = result.certification.request.mapped_source_evidence
    assert evidence is not None and authored.source.mapped_domain is not None
    domain, limits = authored.source.mapped_domain, evidence.limits
    calls = []
    original = embedding_owner.certify_global_embedding

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0].mesh_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(embedding_owner, "certify_global_embedding", counted)
    with coordinate_certificate_request(limits) as budget:
        source_owner.retain_mapped_reference_source_evidence(
            evidence, domain, limits, budget=budget
        )
        physical, embedding, coverage = source_owner.certify_mapped_reference_source(
            domain, limits, budget=budget
        )
        assert physical is evidence.physical_mesh
        assert embedding is evidence.physical_embedding
        assert coverage is evidence.reference_coverage
        assert calls == []

    with coordinate_certificate_request(limits) as cold_budget:
        fresh = source_owner.reprove_mapped_reference_source_evidence(
            evidence, budget=cold_budget
        )
        assert fresh.evidence_id == evidence.evidence_id
        assert fresh is not evidence
        first_count = len(calls)
        assert first_count >= 2
        assert source_owner.reprove_mapped_reference_source_evidence(
            evidence, budget=cold_budget
        ) is fresh
        assert len(calls) == first_count
    reference = weakref.ref(fresh)
    del fresh, cold_budget
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("changed", ("coordinates", "materials", "periodic", "limits"))
def test_mapped_source_evidence_refuses_changed_banks_and_policy(
    corrected_source: tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult],
    changed: str,
) -> None:
    from dataclasses import replace

    import equinox as eqx
    import jax.numpy as jnp

    _, result = corrected_source
    assert result.certification is not None
    evidence = result.certification.request.mapped_source_evidence
    assert evidence is not None
    domain, limits = evidence.domain, evidence.limits
    if changed == "coordinates":
        coordinates = np.asarray(domain.source_geometry.coordinates).copy()
        coordinates.flat[0] = np.nextafter(coordinates.flat[0], np.inf)
        domain = eqx.tree_at(
            lambda value: value.source_geometry.coordinates,
            domain,
            jnp.asarray(coordinates),
        )
    elif changed == "materials":
        regions = np.asarray(domain.cell_regions).copy()
        regions[0] = 1 - regions[0]
        domain = eqx.tree_at(lambda value: value.cell_regions, domain, regions)
    elif changed == "periodic":
        topology = domain.reference_mesh.periodic_topology
        assert topology is not None
        shifts = np.asarray(topology.vertex_shifts).copy()
        shifts[0, 0] += 1
        changed_topology = eqx.tree_at(
            lambda value: value.vertex_shifts, topology, jnp.asarray(shifts)
        )
        domain = eqx.tree_at(
            lambda value: value.reference_mesh.periodic_topology,
            domain,
            changed_topology,
        )
    else:
        limits = replace(limits, maximum_work_units=limits.maximum_work_units + 1)
    with pytest.raises(ValueError, match="exact source"):
        evidence.require_live(domain, limits)


def test_mapped_source_consumer_refuses_caller_minted_proof(
    corrected_source: tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult],
) -> None:
    from phydrax.geometry._embedding_state import coordinate_certificate_request
    from phydrax.geometry._global_embedding import (
        EmbeddingCounts,
        GlobalEmbeddingCertificate,
    )
    from phydrax.geometry._mapped_reference_coverage import (
        MappedReferenceSourceEvidence,
        restore_mapped_reference_source_evidence,
        retain_mapped_reference_source_evidence,
    )

    _, result = corrected_source
    assert result.certification is not None
    request = result.certification.request
    issued = request.mapped_source_evidence
    assert issued is not None
    forged = MappedReferenceSourceEvidence(
        issued.domain,
        issued.limits,
        issued.physical_mesh,
        issued.reference_validity,
        issued.reference_embedding,
        issued.reference_coverage,
        issued.physical_validity,
        issued.physical_embedding,
    )
    # Matching certified records and IDs do not mint issuing-process authority.
    assert forged.evidence_id == issued.evidence_id
    with coordinate_certificate_request(issued.limits) as budget:
        with pytest.raises(ValueError, match="issued"):
            retain_mapped_reference_source_evidence(
                forged, issued.domain, issued.limits, budget=budget
            )
        with pytest.raises(ValueError, match="issued"):
            phx.meshing.MeshCertificationInputs(
                result.mesh,
                result.geometry,
                request.schedule,
                domain=request.domain,
                cell_regions=request.cell_regions,
                source=request.source,
                fidelity_tolerance=request.fidelity_tolerance,
                fidelity_sample_order=request.fidelity_sample_order,
                limits=request.limits,
                junction_vertices=request.junction_vertices,
                scoped_fidelity=request.scoped_fidelity,
                mapped_source_evidence=forged,
                budget=budget,
            )
        fresh = restore_mapped_reference_source_evidence(forged, budget=budget)
        assert fresh is not forged
        assert fresh.evidence_id == forged.evidence_id
        forged.require_live(issued.domain, issued.limits)
        assert restore_mapped_reference_source_evidence(forged, budget=budget) is forged

        fabricated_embedding = GlobalEmbeddingCertificate(
            issued.physical_embedding.binding,
            issued.physical_validity.certificate_id,
            (),
            (),
            counts=EmbeddingCounts(
                issued.physical_embedding.cell_count, 0, 0, 0, 0, 0
            ),
        )
        assert fabricated_embedding.status == "certified"
        fabricated = MappedReferenceSourceEvidence(
            issued.domain,
            issued.limits,
            issued.physical_mesh,
            issued.reference_validity,
            issued.reference_embedding,
            issued.reference_coverage,
            issued.physical_validity,
            fabricated_embedding,
        )
        with pytest.raises(ValueError, match="cold_proof"):
            restore_mapped_reference_source_evidence(fabricated, budget=budget)
        with pytest.raises(ValueError, match="issued"):
            fabricated.require_live(issued.domain, issued.limits)
