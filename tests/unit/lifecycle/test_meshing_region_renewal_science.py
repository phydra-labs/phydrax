"""Renewal compares full science while retaining both real accounting histories."""

from copy import copy
from dataclasses import fields

import equinox as eqx
import numpy as np
import pytest

from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._fingerprint import tree_content_equal
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._compartments import CompartmentMeshingSource
from phydrax.geometry._domain_coverage import DomainCoverageCertificate
from phydrax.lifecycle._meshing_sources import _same_renewed_region_science
from phydrax.meshing import (
    CellMeshingResult,
    NativeMeshingOptions,
    RegionMeshingEvidence,
    VolumeMeshingSpec,
)
from phydrax.meshing._compartments import revalidate_region_evidence
from phydrax.meshing.qualification._native_lifecycle import execute_native_consumer
from tests._support.recipes import restore_with_identity
from tests.unit.meshing.test_live_compartment_material_registration_archiveclosure import (
    cold_material_publication as cold_material_publication,
)


type Accounting = tuple[int | None, int | None, int | None]
type ActualRenewal = tuple[RegionMeshingEvidence, RegionMeshingEvidence, Accounting]


def _costs(coverage: DomainCoverageCertificate) -> Accounting:
    return (coverage.source_expression_work_units,
            coverage.source_expression_required_work_units,
            coverage.source_expression_peak_bytes)


def _fail(stage: str, message: str, data: object) -> ValueError:
    return ValueError(f"{stage}: {message}; {data}")


@pytest.fixture(scope="module")
def actual_region_renewal(
    cold_material_publication: tuple[
        CompartmentMeshingSource, VolumeMeshingSpec, NativeMeshingOptions, CellMeshingResult
    ],
) -> ActualRenewal:
    source, specification, _, carrier = cold_material_publication
    archived = carrier.region_evidence
    assert archived is not None
    before = _costs(archived.coverage)

    def renew(budget: CoordinateEnclosureBudget) -> RegionMeshingEvidence:
        actual = revalidate_region_evidence(
            carrier, carrier.mesh, carrier.geometry, carrier.zones, tuple(carrier.patches),
            compartment_source=source, certificate_limits=archived.certificate_limits,
            limits=specification.limits, budget=budget,
        )
        assert budget.retained_owner_objects[id(actual)] is actual
        fresh = actual.region_evidence
        assert _same_renewed_region_science(archived, fresh, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
        assert _costs(archived.coverage) == before
        return fresh

    result = execute_native_consumer(
        carrier, specification.limits, renew, failure_factory=_fail,
    )
    assert result.result.execution_evidence is not None
    assert int(np.asarray(result.result.execution_evidence.total_work_units)) > 0
    return archived, result.value, before


def test_real_renewal_preserves_full_science_and_both_accounting_records(
    actual_region_renewal: ActualRenewal,
) -> None:
    archived, renewed, before = actual_region_renewal
    accounting = {
        "source_expression_work_units", "source_expression_required_work_units",
        "source_expression_peak_bytes",
    }
    coverage = archived.coverage
    assert type(coverage) is DomainCoverageCertificate
    assert {name for name, _ in coverage.scientific_payload()} == {
        member.name for member in fields(coverage)
    } - accounting
    assert tree_content_equal(coverage.scientific_payload(), renewed.coverage.scientific_payload())
    assert _costs(archived.coverage) == before
    assert _costs(renewed.coverage) != (None, None, None)
    assert archived is not renewed


def test_renewal_projection_does_not_ignore_proof_identity(
    actual_region_renewal: ActualRenewal,
) -> None:
    archived, renewed, _ = actual_region_renewal
    changed = restore_with_identity(renewed.coverage, "certificate_id", "foreign-certificate")
    forged = eqx.tree_at(lambda value: value.coverage, renewed, changed)
    assert not _same_renewed_region_science(archived, forged, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)


def test_renewal_projection_does_not_ignore_numerical_coverage_banks(
    actual_region_renewal: ActualRenewal,
) -> None:
    archived, renewed, _ = actual_region_renewal
    lower = renewed.coverage.facet_source_overlaps.lower
    assert lower.size
    forged = eqx.tree_at(
        lambda value: value.coverage.facet_source_overlaps.lower,
        renewed, lower + 0.125,
    )
    assert not _same_renewed_region_science(archived, forged, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)


@pytest.mark.parametrize(
    "field",
    (
        "source_revision",
        "source_complex_id",
        "cell_region_ids",
        "interface_definitions",
        "interface_facets",
    ),
)
def test_renewal_projection_rejects_material_and_interface_changes(
    actual_region_renewal: ActualRenewal, field: str,
) -> None:
    archived, renewed, _ = actual_region_renewal
    changed = copy(renewed)
    value = getattr(renewed, field)
    replacement = f"{value}:changed" if isinstance(value, str) else (*value, ("changed",))
    object.__setattr__(changed, field, replacement)
    assert not _same_renewed_region_science(
        archived, changed, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )


@pytest.mark.parametrize(
    "field",
    (
        "source_expression_work_units",
        "source_expression_required_work_units",
        "source_expression_peak_bytes",
    ),
)
def test_renewal_projection_keeps_phase_observations_separate(
    actual_region_renewal: ActualRenewal, field: str,
) -> None:
    archived, renewed, before = actual_region_renewal
    coverage = copy(renewed.coverage)
    original = getattr(coverage, field)
    assert original is not None
    object.__setattr__(coverage, field, original + 1)
    observed = eqx.tree_at(lambda value: value.coverage, renewed, coverage)
    assert observed.coverage.certificate_id == renewed.coverage.certificate_id
    assert _same_renewed_region_science(
        archived, observed, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    assert _costs(archived.coverage) == before
    assert getattr(renewed.coverage, field) == original
