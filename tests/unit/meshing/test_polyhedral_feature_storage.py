"""Allocation-bound regressions for the real polyhedral feature helpers."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionLimits,
    NativeExecutionScope,
    recover_plc_3d,
)
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.linalg._hermitian_spectral import fraction_storage_upper
from phydrax.meshing import MeshingLimits
from phydrax.meshing import _polyhedral_generation as owner
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.polyhedral_meshes import POLYHEDRAL_CUBE_LOOPS, voxel_plc


def _scope(cap: int) -> NativeExecutionScope:
    return NativeExecutionScope(NativeExecutionLimits(1 << 30, 1 << 30, 1 << 20, cap, 120.0))


def _cube_recovery():
    source = voxel_plc(((0, 0, 0),))
    recovery = recover_plc_3d(
        source.vertices, source.polygon_offsets, source.polygon_vertices,
        source.polygon_facets, source.facet_regions,
        boundary_policy="fixed", max_vertices=256, max_tetrahedra=1024,
        work_limit=1 << 24, max_scratch_bytes=1 << 26,
    )
    return source, recovery


@pytest.mark.parametrize("denominator_bits", (0, 4096))
def test_segment_storage_exact_boundary_precedes_rational_lifts(
    monkeypatch: pytest.MonkeyPatch, denominator_bits: int,
) -> None:
    shift = Fraction(1, 1 << denominator_bits) if denominator_bits else Fraction(0)
    a = np.asarray((shift, Fraction(0), Fraction(0)), dtype=object)
    b = np.asarray((shift + 1, Fraction(0), Fraction(0)), dtype=object)
    ends = np.asarray(((shift + Fraction(1, 4), 0, 0), (shift + Fraction(3, 4), 0, 0)), dtype=object)
    bits = max(owner._feature_fraction_bits(ends), owner._feature_fraction_bits((a, b)))
    required = fraction_storage_upper(32 + 2 * len(ends), 384 * bits + 512)
    limits = MeshCertificateLimits(maximum_scratch_bytes=required)
    with coordinate_reference_budget(limits) as budget:
        result = owner._segment_evidence(ends, a, b, budget=budget)
        assert result == (Fraction(1, 4), Fraction(3, 4), 0.0, float(np.nextafter(1.0, np.inf)))
        assert budget.peak_bytes_upper == required
        assert budget.work_units == 0  # storage admission must not reprice the science
    lifts = 0
    original = owner._feature_coordinate_fraction

    def lift(value):
        nonlocal lifts
        lifts += 1
        return original(value)

    monkeypatch.setattr(owner, "_feature_coordinate_fraction", lift)
    with coordinate_reference_budget(
        MeshCertificateLimits(maximum_scratch_bytes=required - 1)
    ) as budget:
        with pytest.raises(CoordinateEnclosureResourceError):
            owner._segment_evidence(ends, a, b, budget=budget)
        assert lifts == 0
        assert budget.work_units == 0


def test_source_incidence_admission_exact_boundary_precedes_array_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, recovery = _cube_recovery()
    occurrences, triangles, semantic = (
        source.polygon_vertices.size, len(recovery.input_triangles), len(recovery.plc_edges)
    )
    bound = (
        owner._feature_entry_storage(4 * occurrences + 2 * semantic + triangles)
        + 24 * len(source.vertices) + 72 * triangles
        + 8 * occurrences + 256 * (triangles + 16)
    )
    # Observe the real pool token overhead; R is not an invented RSS estimate.
    cap = 1 << 26
    with _scope(cap) as scope, scope.host_workspace() as workspace:
        workspace.set_bound(bound)
        required = cap - scope.remaining().remaining_managed_bytes
    original = np.asarray
    reached = 0

    class ReachedAllocation(Exception):
        pass

    def allocation(value, *args, **kwargs):
        nonlocal reached
        if value is source.vertices:
            reached += 1
            raise ReachedAllocation
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", allocation)
    with _scope(required) as scope:
        with pytest.raises(ReachedAllocation):
            owner._source_feature_edges(source, recovery, scope=scope)
    assert reached == 1
    with pytest.raises(MeshcoreError) as error, _scope(required - 1) as scope:
        owner._source_feature_edges(source, recovery, scope=scope)
    assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert reached == 1


def test_real_cube_features_and_diameters_remain_deterministic() -> None:
    source, recovery = _cube_recovery()
    mesh = CellMesh.from_polyhedra(source.vertices, (POLYHEDRAL_CUBE_LOOPS,))
    geometry = CellGeometrySpec.affine(mesh)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        edges = owner._source_feature_edges(source, recovery, scope=scope)
        assert sorted(tuple(sorted(map(int, edge))) for edge in edges) == [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        ]
        with owner._polyhedral_coordinate_phase(
            1 << 24, 1 << 27, scope=scope, budget=budget
        ):
            diameters = owner._diameters(mesh, geometry, scope=scope, budget=budget)
        np.testing.assert_array_equal(diameters, (owner.sqrt_upper(Fraction(3)),))
        # Outputs remain charged after their temporary helper workspace closes.
        assert scope.remaining().remaining_managed_bytes < MeshingLimits().maximum_scratch_bytes
        np.testing.assert_array_equal(edges, owner._source_feature_edges(source, recovery, scope=scope))


def test_feature_preparation_retains_real_pool_admission_until_consumed() -> None:
    source, recovery = _cube_recovery()
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        edges = owner._source_feature_edges(source, recovery, scope=scope)
        with owner._polyhedral_coordinate_phase(
            1 << 24, 1 << 27, scope=scope, budget=budget
        ):
            ticket = budget.ticket
            assert isinstance(ticket, NativeBatchTicket)
            preparation = owner._prepare_feature_ancestry(
                recovery, edges, scope=scope, budget=budget
            )
            retained = ticket.remaining_managed_bytes
            key = tuple(map(int, edges[0]))
            matched = owner._match_feature_carrier(
                preparation, recovery, key, 0, 0, 1 << 24,
                scope=scope, budget=budget,
            )
            assert matched.source == 0
            assert matched.work_units > 0
            assert preparation.lengths == (
                float(np.nextafter(1.0, np.inf)),
            ) * len(edges)
            gaps = scope.allocate_host_array((len(edges),), np.float64)
            coverage = owner._feature_coverage(
                [[(Fraction(0), Fraction(1))] for _ in edges],
                preparation.lengths, gaps, budget=budget,
            )
            np.testing.assert_array_equal(coverage.gaps, np.zeros(len(edges)))
            before_release = ticket.remaining_managed_bytes
            count = len(edges)
            preparation_bound = (
                owner._feature_entry_storage(count + len(recovery.segments))
                + owner._feature_bvh_storage(count) + 48 * count + 32 * count + 256
            )
            del preparation
            assert ticket.remaining_managed_bytes - before_release >= preparation_bound
            assert retained < ticket.bytes_upper


def test_many_feature_candidates_reuse_one_rational_temporary_admission() -> None:
    source, recovery = _cube_recovery()
    single_segment = fraction_storage_upper(36, 384 * 1075 + 512)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        edges = owner._source_feature_edges(source, recovery, scope=scope)
        # Four one-segment banks leave room for concrete BVH/cache/native
        # storage, but cannot admit twelve accumulated source-segment banks.
        with owner._polyhedral_coordinate_phase(
            1 << 24, 4 * single_segment, scope=scope, budget=budget
        ):
            baseline_storage = budget.temporary_bytes_upper
            baseline_work = budget.work_units
            preparation = owner._prepare_feature_ancestry(
                recovery, edges, scope=scope, budget=budget
            )
            assert budget.temporary_bytes_upper == baseline_storage
            peak = budget.peak_bytes_upper
            assert peak <= baseline_storage + single_segment
            work_units = 0
            for candidate in range(64):
                source_index = candidate % len(edges)
                matched = owner._match_feature_carrier(
                    preparation, recovery, tuple(map(int, edges[source_index])),
                    candidate, work_units, 1 << 24, scope=scope, budget=budget,
                )
                assert matched.source == source_index
                assert matched.work_units > work_units
                work_units = matched.work_units
                assert budget.temporary_bytes_upper == baseline_storage
                assert budget.peak_bytes_upper == peak
                assert budget.work_units - baseline_work == work_units
            del preparation
