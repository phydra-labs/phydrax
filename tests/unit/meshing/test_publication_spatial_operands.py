"""Fresh source/coverage decisions over request-local exact clipping operands."""

from dataclasses import fields
from math import inf
from time import sleep
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from benchmarks.publication_spatial_operands import fixture
from phydrax._issued_proofs import IssuedProofs
from phydrax._meshcore import MeshcoreError, NativeExecutionLimits, NativeExecutionScope
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_operands import (
    CoordinateOperandCollector,
    live_affine_area_operands,
)
from phydrax.geometry import _domain_coverage as coverage_owner, _planar_coverage as owner
from tests._support.surface_metric import surface_resources


def _same_groups(first: Any, second: Any) -> None:
    assert first[1:] == second[1:]
    assert len(first[0]) == len(second[0])
    for left, right in zip(first[0], second[0], strict=True):
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
            else:
                assert a == b


@pytest.mark.parametrize("maximum_pairs", [0, 1, 100_000])
@pytest.mark.parametrize("overlap", [False, True])
def test_source_decisions_remain_fresh(maximum_pairs: int, overlap: bool) -> None:
    _, _, _, domain, _ = fixture(2)
    vertices, facets, regions = domain.vertices, domain.facets, domain.facet_regions
    if overlap:
        facets = np.concatenate((facets, facets[:1]))
        regions = np.concatenate((regions, regions[:1]))
    with surface_resources() as (scope, budget):
        expected = owner.source_groups(vertices, facets, regions, maximum_pairs, budget=budget)
        with CoordinateOperandCollector(capture_affine_areas=True).live_affine_areas(scope):
            earned = owner.source_groups(vertices, facets, regions, maximum_pairs, budget=budget)
            prepared = owner.source_groups(vertices, facets, regions, maximum_pairs, budget=budget)
            _same_groups(expected, earned)
            _same_groups(expected, prepared)


def test_source_mutation_cold_and_expiry() -> None:
    _, _, _, domain, _ = fixture(2)
    vertices = np.asarray(domain.vertices).copy()
    with surface_resources() as (scope, budget):
        assert live_affine_area_operands(vertices, domain.facets, domain.facet_regions, budget=budget) is None
        with CoordinateOperandCollector(capture_affine_areas=True).live_affine_areas(scope):
            operands = live_affine_area_operands(vertices, domain.facets, domain.facet_regions, budget=budget)
            assert operands is not None
            vertices[0, 0] += 0.125
            assert live_affine_area_operands(vertices, domain.facets, domain.facet_regions, budget=budget) is None
            vertices[0, 0] -= 0.125
            assert live_affine_area_operands(vertices, domain.facets, domain.facet_regions, budget=budget) is operands
        assert live_affine_area_operands(vertices, domain.facets, domain.facet_regions, budget=budget) is None
        assert not operands._current(budget)
    with surface_resources() as (_, foreign):
        assert not operands._current(foreign)


@pytest.mark.parametrize("unassigned", [False, True])
def test_acceptance_certificate_science_and_ids_are_fresh(
    monkeypatch: pytest.MonkeyPatch, unassigned: bool,
) -> None:
    mesh, geometry, audit, domain, regions = fixture(2)
    if unassigned:
        regions[0] = -1

    def certify(budget: Any) -> Any:
        # Force independent certificates: the test does not exercise the
        # pre-existing process issued-proof shortcut.
        monkeypatch.setattr(coverage_owner, "_ISSUED_COVERAGE", IssuedProofs())
        budget.global_embedding_certificate_cache.clear()
        return phx.meshing.certify_meshing_acceptance(
            mesh, geometry, audit,
            schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
            domain=domain, cell_regions=regions, budget=budget,
        )

    with surface_resources() as (scope, budget):
        expected = certify(budget)
        with CoordinateOperandCollector(capture_affine_areas=True).live_affine_areas(scope):
            owner.source_groups(domain.vertices, domain.facets, domain.facet_regions, 100_000, budget=budget)
            actual = certify(budget)
        assert actual is not expected
        assert actual.report_id == expected.report_id
        assert actual.passed == expected.passed
        assert actual.coverage is not expected.coverage
        assert actual.coverage.certificate_id == expected.coverage.certificate_id
        assert eqx.tree_equal(actual.coverage.scientific_payload(), expected.coverage.scientific_payload())
        assert actual.embedding.certificate_id == expected.embedding.certificate_id


def _resource_trial(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    work: int = 10_000_000,
    storage: int = 100_000_000,
    expired: bool = False,
) -> Any:
    with NativeExecutionScope(
        NativeExecutionLimits(work, 10_000_000, 10_000_000, storage, 0.01 if expired else inf)
    ) as scope:
        scope.observe_managed_admission()
        ticket = scope.batch(work, queries_upper=10_000_000, bytes_upper=storage)
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=storage
        )
        with ticket, budget.activate(), ticket.native_target():
            with CoordinateOperandCollector(capture_affine_areas=True).live_affine_areas(scope):
                owner.source_groups(*arrays, 100_000, budget=budget)
                if expired:
                    sleep(0.02)
                owner.source_groups(*arrays, 100_000, budget=budget)
                scope.charge(work=0)
    assert scope.evidence is not None
    return scope.evidence


def test_live_operand_work_storage_exact_frontiers_and_deadline() -> None:
    vertices = np.asarray(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
    facets = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    regions = np.tile(np.asarray(((0, -1),), dtype=np.int64), (2, 1))
    arrays = (vertices, facets, regions)
    measured = _resource_trial(arrays)
    required_work = measured.work_counters.construction_work
    required_storage = measured.managed_admission_peak_bytes_upper
    assert required_work > 0 and required_storage is not None
    _resource_trial(arrays, work=required_work)
    _resource_trial(arrays, storage=required_storage)
    with pytest.raises((MeshcoreError, CoordinateEnclosureResourceError)):
        _resource_trial(arrays, work=required_work - 1)
    with pytest.raises((MeshcoreError, CoordinateEnclosureResourceError)):
        _resource_trial(arrays, storage=required_storage - 1)
    with pytest.raises(MeshcoreError, match="deadline|TIMEOUT"):
        _resource_trial(arrays, expired=True)
