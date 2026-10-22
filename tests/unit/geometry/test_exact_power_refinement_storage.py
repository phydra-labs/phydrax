# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exercise real exact-power outputs and pre-operation request admission."""

from fractions import Fraction

import numpy as np
import pytest

import phydrax.geometry._exact_power_refinement as owner
import phydrax.geometry._exact_polyhedral_geometry as overlap_owner
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._exact_polyhedral_geometry import (
    ExactConvexCell, TetrahedronPlanePreparation,
)
from phydrax.geometry._supermesh import CommonRefinementPolicy, CommonRefinementStatus
from phydrax.meshing import MeshingLimits
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume, NativePolyhedralSchedule,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.polyhedral_meshes import periodic_adaptation_source


_MEMORY = 16 * 1024**3
_LIMITS = MeshingLimits(maximum_work_units=1 << 30, maximum_scratch_bytes=_MEMORY)


def _box(low: Fraction = Fraction(0)) -> ExactConvexCell:
    vertices = tuple(
        tuple(low if not bits & (1 << axis) else low + 1 for axis in range(3))
        for bits in range(8)
    )
    edges = tuple(
        (bits, bits | axis) for bits in range(8) for axis in (1, 2, 4)
        if not bits & axis
    )
    planes = tuple(
        tuple(Fraction(sign if coordinate == axis else 0) for coordinate in range(3))
        + ((low if sign < 0 else -low - 1),)
        for axis in range(3) for sign in (-1, 1)
    )
    return ExactConvexCell(vertices, edges, planes, (low,) * 3, (low + 1,) * 3)


def _pair(first, second, policy, budget, storage, workspace):
    return owner._power_pair(
        (), (), first, second, policy, 0, 0,
        budget=budget, preparation=TetrahedronPlanePreparation(budget),
        storage=storage, workspace=workspace,
    )


@pytest.mark.parametrize("denominator", (3, (1 << 521) - 1))
@pytest.mark.parametrize("shortfall", (0, 1))
def test_pair_r_and_r_minus_one_refuse_before_side_allocation(
    monkeypatch, denominator, shortfall,
):
    first, second = _box(), _box(Fraction(1, denominator))
    attempted = []

    class ReachedSides(Exception):
        pass

    def stop_before_sides(*args, **kwargs):
        attempted.append(True)
        raise ReachedSides

    monkeypatch.setattr(overlap_owner, "_convex_side_rows", stop_before_sides)
    # Profile the actual entry banks, not a synthetic whole-pair estimate.
    with coordinate_reference_budget(_LIMITS) as profiling:
        with profiling.live_storage() as storage, profiling.live_storage() as workspace:
            with pytest.raises(ReachedSides):
                _pair(
                    first, second, CommonRefinementPolicy(second_moments=True),
                    profiling, storage, workspace,
                )
            required = profiling.peak_bytes_upper
    attempted.clear()
    limits = MeshingLimits(
        maximum_work_units=1 << 30, maximum_scratch_bytes=required - shortfall,
    )
    with coordinate_reference_budget(limits) as budget:
        with budget.live_storage() as storage, budget.live_storage() as workspace:
            if shortfall:
                pair = _pair(
                    first, second, CommonRefinementPolicy(second_moments=True),
                    budget, storage, workspace,
                )
                assert pair.refusal is owner._PowerRefusal.RATIONAL_STORAGE
            else:
                with pytest.raises(ReachedSides):
                    _pair(
                        first, second, CommonRefinementPolicy(second_moments=True),
                        budget, storage, workspace,
                    )
            assert workspace.bound == 0
    assert bool(attempted) is (shortfall == 0)


def test_large_fraction_denominators_grow_admission_and_preserve_integrals():
    first, second = _box(), _box(Fraction(1, (1 << 521) - 1))
    with coordinate_reference_budget(_LIMITS) as budget:
        small = owner._overlap_bound(first, _box(Fraction(1, 3)), budget=budget)
        large = owner._overlap_bound(first, second, budget=budget)
        assert large.coordinate_bits > small.coordinate_bits
        assert large.plane_bits > small.plane_bits
        assert large.crossing_bits > small.crossing_bits
        with budget.live_storage() as storage, budget.live_storage() as workspace:
            pair = _pair(
                first, second, CommonRefinementPolicy(second_moments=True, overlap_simplices=True),
                budget, storage, workspace,
            )
            length = 1 - second.lower[0]
            assert pair.refusal is owner._PowerRefusal.NONE
            assert pair.volume == length**3
            assert pair.moment == (length**3 * (1 + second.lower[0]) / 2,) * 3
            assert pair.pieces
            assert workspace.bound == 0


def test_repeated_pair_workspaces_release_without_retaining_intersections():
    first, second = _box(), _box(Fraction(1, 3))
    with coordinate_reference_budget(_LIMITS) as budget:
        retained = budget.retained_basis_bytes
        for _ in range(4):
            with budget.live_storage() as storage, budget.live_storage() as workspace:
                pair = _pair(
                    first, second, CommonRefinementPolicy(overlap_simplices=True),
                    budget, storage, workspace,
                )
                assert pair.volume == Fraction(8, 27)
                assert workspace.bound == 0
                assert storage.bound > 0
                del pair
            assert budget.temporary_bytes_upper == 0
            assert budget.retained_basis_bytes == retained


def test_pair_local_work_refusal_reports_only_executed_prefix():
    with coordinate_reference_budget(_LIMITS) as budget:
        with budget.live_storage() as storage, budget.live_storage() as workspace:
            pair = _pair(
                _box(), _box(), CommonRefinementPolicy(maximum_exact_work=1),
                budget, storage, workspace,
            )
            assert pair.refusal is owner._PowerRefusal.WORK
            assert pair.work == 0
            # Input-bit profiling is separately priced metadata, not an
            # executed overlap primitive or an uncharged direct-ticket path.
            assert budget.work_units > 0
            assert budget.ticket.work_units == budget.work_units


@pytest.mark.parametrize("shortfall", (0, 1))
def test_pair_actual_root_work_boundary_does_not_reserve_unvisited_branches(shortfall):
    first, second = _box(), _box(Fraction(1, 3))
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    with coordinate_reference_budget(_LIMITS) as profiling:
        with profiling.live_storage() as storage, profiling.live_storage() as workspace:
            pair = _pair(first, second, policy, profiling, storage, workspace)
            assert pair.refusal is owner._PowerRefusal.NONE
            required = profiling.ticket.work_units
            assert required == profiling.work_units
    limits = MeshingLimits(
        maximum_work_units=required - shortfall, maximum_scratch_bytes=_MEMORY,
    )
    with coordinate_reference_budget(limits) as budget:
        with budget.live_storage() as storage, budget.live_storage() as workspace:
            pair = _pair(first, second, policy, budget, storage, workspace)
            assert budget.work_units == budget.ticket.work_units
            if shortfall:
                assert pair.refusal is owner._PowerRefusal.WORK
                assert budget.work_units <= required - 1
            else:
                assert pair.refusal is owner._PowerRefusal.NONE
                assert pair.volume == Fraction(8, 27)
                assert budget.work_units == required


@pytest.mark.parametrize("shortfall", (0, 1))
def test_pair_actual_root_memory_boundary_preserves_real_integrals(shortfall):
    first, second = _box(), _box(Fraction(1, 3))
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    with coordinate_reference_budget(_LIMITS) as profiling:
        with profiling.live_storage() as storage, profiling.live_storage() as workspace:
            pair = _pair(first, second, policy, profiling, storage, workspace)
            assert pair.refusal is owner._PowerRefusal.NONE
            required = profiling.peak_bytes_upper
    limits = MeshingLimits(
        maximum_work_units=1 << 30, maximum_scratch_bytes=required - shortfall,
    )
    with coordinate_reference_budget(limits) as budget:
        with budget.live_storage() as storage, budget.live_storage() as workspace:
            pair = _pair(first, second, policy, budget, storage, workspace)
            assert workspace.bound == 0
            if shortfall:
                assert pair.refusal is owner._PowerRefusal.RATIONAL_STORAGE
                assert budget.peak_bytes_upper <= required - 1
            else:
                assert pair.refusal is owner._PowerRefusal.NONE
                assert pair.volume == Fraction(8, 27)
                assert pair.pieces
                assert budget.peak_bytes_upper == required


def test_face_primitive_scratch_ends_before_next_face_and_pair_outputs_remain(monkeypatch):
    original = overlap_owner._integrate_overlap_face
    entries = []

    def observe_face(*args):
        work = args[-1]
        entries.append(work.budget.temporary_bytes_upper)
        return original(*args)

    monkeypatch.setattr(overlap_owner, "_integrate_overlap_face", observe_face)
    with coordinate_reference_budget(_LIMITS) as budget:
        with budget.live_storage() as storage, budget.live_storage() as workspace:
            pair = _pair(
                _box(), _box(Fraction(1, 3)),
                CommonRefinementPolicy(second_moments=True, overlap_simplices=True),
                budget, storage, workspace,
            )
            assert pair.refusal is owner._PowerRefusal.NONE
            assert len(entries) > 1
            assert len(set(entries)) == 1
            assert budget.temporary_bytes_upper == storage.bound
            assert storage.bound > 0
            assert pair.volume == Fraction(8, 27)
            assert pair.pieces
            del pair
        assert budget.temporary_bytes_upper == 0


def test_repeated_real_edges_do_not_inflate_retained_intersection_bank():
    first, second = _box(), _box(Fraction(1, 3))
    repeated = ExactConvexCell(
        second.vertices, second.edges * 8, second.planes, second.lower, second.upper,
    )
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    with coordinate_reference_budget(_LIMITS) as budget:
        with (
            budget.live_storage() as original_storage,
            budget.live_storage() as repeated_storage,
            budget.live_storage() as workspace,
        ):
            original = _pair(first, second, policy, budget, original_storage, workspace)
            duplicate = _pair(first, repeated, policy, budget, repeated_storage, workspace)
            assert original.refusal is duplicate.refusal is owner._PowerRefusal.NONE
            assert original.volume == duplicate.volume
            assert original.moment == duplicate.moment
            assert original.pieces == duplicate.pieces
            assert original_storage.bound == repeated_storage.bound
            # Every repeated crossing is still executed and charged; only its
            # already-retained vertex avoids a duplicate allocation.
            assert duplicate.work > original.work
            del original, duplicate


def _construction(sites):
    topology = reference_cell_topology("hexahedron")
    domain = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64), topology.entities[2],
        np.zeros(6, dtype=np.int64), np.asarray([[-1, 0]], dtype=np.int64), ("material",),
    )
    with NativeExecutionScope(NativePolyhedralSchedule().native_limits()) as scope:
        return generate_polyhedral_volume(domain, sites=np.asarray(sites), scope=scope)


@pytest.mark.parametrize("shortfall", (0, 1))
def test_decomposition_r_and_r_minus_one_precede_star_allocation(monkeypatch, shortfall):
    construction = _construction(((0.5, 0.5, 0.5),))
    mesh, geometry = construction.mesh, construction.geometry
    attempted = []

    class ReachedStars(Exception):
        pass

    def stop_before_stars(*args, **kwargs):
        attempted.append(True)
        raise ReachedStars

    with coordinate_reference_budget(_LIMITS, queries_upper=1 << 20) as budget:
        # Warm the real canonical source bank; it is not part of the star
        # allocation whose precise additional admission this test isolates.
        points = owner.exact_vertices(mesh, geometry, budget=budget)
        bound = owner._decomposition_bound(mesh, points, budget=budget)
        connectivity = mesh.connectivity
        topology_items = sum(
            value.size for value in (
                connectivity.cell_vertex_offsets, connectivity.cell_vertex_values,
                connectivity.cell_face_offsets, connectivity.cell_face_values,
                connectivity.cell_face_sign_values, connectivity.face_vertex_offsets,
                connectivity.face_vertex_values,
            )
        ) + sum(
            block.vertices.size + block.vertex_valid.size + block.global_ids.size
            for block in mesh.blocks
        )
        required = (
            budget.retained_basis_bytes + 4096 + 32 * mesh.coordinates.size
            + 32 * topology_items + bound
        )
        monkeypatch.setattr(owner, "star_tetrahedra", stop_before_stars)
        expected = CoordinateEnclosureResourceError if shortfall else ReachedStars
        with pytest.raises(expected):
            with (
                budget.bound_stage(1 << 30, required - shortfall),
                budget.live_storage() as storage,
            ):
                owner._decomposition(mesh, geometry, budget=budget, storage=storage)
    assert bool(attempted) is (shortfall == 0)


@pytest.mark.parametrize("case", ("nested", "nonnested", "periodic"))
def test_real_power_refinement_retains_positive_outputs_and_original_identities(case):
    if case == "periodic":
        _, _, source = periodic_adaptation_source()
        target = source
    else:
        source = _construction(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)))
        target = _construction(
            ((0.5, 0.5, 0.5),) if case == "nested"
            else ((0.5, 0.25, 0.5), (0.5, 0.75, 0.5))
        )
    identities = (
        source.mesh.mesh_id, target.mesh.mesh_id,
        source.mesh.topology_id, target.mesh.topology_id,
    )
    with coordinate_reference_budget(_LIMITS, queries_upper=1 << 20) as budget:
        prepared = owner.prepare_exact_power_refinement(
            source.mesh, target.mesh, source.geometry, target.geometry,
            CommonRefinementPolicy(second_moments=True, overlap_simplices=True),
            budget=budget,
        )
        assert prepared.evidence.status is CommonRefinementStatus.SUCCESS
        assert (
            prepared.source_mesh_id, prepared.target_mesh_id,
            prepared.source_topology_id, prepared.target_topology_id,
        ) == identities
        assert np.all(np.asarray(prepared.volumes) > 0)
        assert np.sum(np.asarray(prepared.volumes)) == pytest.approx(1.0, abs=1e-13)
        assert prepared.simplices is not None
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_owner_objects[id(prepared)] is prepared
        retained = budget.retained_basis_bytes
        budget.retain_owner(prepared)
        assert budget.retained_basis_bytes == retained


def test_actual_periodic_regenerate_uses_original_request_caps():
    import phydrax as phx
    from phydrax._meshcore import NativeParallelExecution
    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
    from tools._meshing_cases import prepare_polyhedral_case

    authority, specification, options = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 4, 20000, 120, source=True,
    )
    limits = specification.limits
    assert (
        limits.maximum_work_units, limits.maximum_geometry_queries,
        limits.maximum_scratch_bytes, limits.maximum_data_bytes,
        limits.maximum_wall_seconds,
    ) == (640000, 1280000, 81920000, 20480000, 120)
    with NativeParallelExecution(None, 120):
        accepted = phx.meshing.NativeMeshingProvider(options).plan(
            authority, specification, coordinate_contract=phx.SpatialCoordinateContract.si(),
        ).execute()
        request = phx.meshing.PolyhedralMeshAdaptation(
            phx.meshing.PolyhedralAdaptationOperation.REGENERATE,
            complex_=authority.complex,
            sites=((0.5, 0.25, 0.5), (0.5, 0.75, 0.5)),
            periodic_constraints=specification.periodic_constraints,
        )
        policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
            limits=limits,
            audit_policy=phx.meshing.CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
            ),
        )
        prepared = phx.meshing.prepare_mesh_adaptation(accepted, request, policy=policy)
        collector = CoordinateOperandCollector()
        result = phx.meshing.execute_mesh_adaptation(prepared, coordinate_operands=collector)
        assert result.transition is not None
        assert result.transition.target is result.target
        assert result.common_refinement is not None
        assert result.common_refinement.status is CommonRefinementStatus.SUCCESS
        assert collector.prepared is not None
        record = result.target.execution_evidence
        assert int(np.asarray(record.total_work_units)) <= limits.maximum_work_units
        assert int(np.asarray(record.total_geometry_queries)) <= limits.maximum_geometry_queries


def test_tiny_policy_refuses_before_source_export_without_fabricating_artifact(monkeypatch):
    source = _construction(((0.5, 0.5, 0.5),))
    attempted = []

    def forbidden(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("source export allocated before admission")

    monkeypatch.setattr(owner, "exact_vertices", forbidden)
    with coordinate_reference_budget(_LIMITS, queries_upper=1 << 20) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            owner.prepare_exact_power_refinement(
                source.mesh, source.mesh, source.geometry, source.geometry,
                CommonRefinementPolicy(maximum_memory_bytes=1), budget=budget,
            )
        assert caught.value.resource == "polynomial_storage"
        assert caught.value.limit == 1
        assert caught.value.requested > caught.value.limit
    assert not attempted


def test_real_work_refusal_retains_truthful_negative_coverage_artifact():
    source = _construction(((0.5, 0.5, 0.5),))
    with coordinate_reference_budget(_LIMITS, queries_upper=1 << 20) as budget:
        prepared = owner.prepare_exact_power_refinement(
            source.mesh, source.mesh, source.geometry, source.geometry,
            CommonRefinementPolicy(maximum_exact_work=1), budget=budget,
        )
        assert prepared.status is CommonRefinementStatus.RESOURCE_LIMIT
        assert prepared.volumes.size == 0
        np.testing.assert_array_equal(
            prepared.evidence.source_coverage_defects, -np.asarray(prepared.source_measures),
        )
        assert np.all(np.asarray(prepared.source_measures) > 0)
        assert budget.retained_owner_objects[id(prepared)] is prepared
