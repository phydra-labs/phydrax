# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Actual ended-source receipts remain distinct from parent/child measurements."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionBudget,
)
from phydrax.meshing import (
    CurveMeshingSpec,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    NativeCurveSource,
    NativeMeshingProvider,
)
from phydrax.meshing._association import PlcAssociationTransfer
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing._volume_generation import (
    _import_native_preparation,
    native_volume_execution_budget,
)


_NATIVE = pytest.mark.skipif(not meshcore_available(), reason="native meshcore required")


@_NATIVE
@pytest.mark.parametrize("remaining", [0, 1])
def test_plc_support_batch_uses_authentic_parent_query_allowance(
    monkeypatch: pytest.MonkeyPatch, remaining: int
) -> None:
    import phydrax.meshing._association as association

    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    owner = SimpleNamespace(
        domain=SimpleNamespace(ambient_dimension=3, vertices=np.eye(3)),
        triangle_vertices=triangles,
        triangle_facets=np.asarray((7,)),
        _spend=lambda work, count: PlcAssociationTransfer._spend(None, work, count),  # ty: ignore[invalid-argument-type] -- borrowed method does not inspect self
    )
    dispatched: list[int] = []

    def support(
        points: np.ndarray, source: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dispatched.append(points.shape[0])
        return np.zeros(1), np.asarray((6,)), np.asarray((MeshcoreStatus.OK,))

    monkeypatch.setattr(association, "point_triangle_locations", support)
    parent = NativeExecutionBudget(
        max_work=100_000,
        max_geometry_queries=1,
        max_cavity_cells=10,
        max_scratch_bytes=4 * 1024**2,
        max_wall_seconds=120.0,
    )
    child = NativeExecutionBudget(
        max_work=100_000,
        max_geometry_queries=100,
        max_cavity_cells=10,
        max_scratch_bytes=4 * 1024**2,
        max_wall_seconds=120.0,
    )
    work = [100]
    if remaining:
        with parent, child:
            assert PlcAssociationTransfer._point_support(owner, 2, 7, np.zeros(3), work)  # ty: ignore[invalid-argument-type] -- minimal provider boundary owner
    else:
        with pytest.raises(MeshcoreError), parent:
            parent.charge(geometry_queries=1)
            with child:
                PlcAssociationTransfer._point_support(owner, 2, 7, np.zeros(3), work)  # ty: ignore[invalid-argument-type] -- minimal provider boundary owner
    assert dispatched == ([1] if remaining else [])
    assert work == [99]
    assert parent.evidence is not None and child.evidence is not None
    assert int(parent.evidence.work_evidence[1]) == 1
    assert int(child.evidence.work_evidence[1]) == remaining
    assert int(parent.evidence.work_evidence[4]) == 1 - remaining


def _actual_source_receipt() -> NativeExecutionRecord:
    with NativeExecutionBudget(
        max_work=3,
        max_geometry_queries=2,
        max_cavity_cells=10,
        max_scratch_bytes=4 * 1024**2,
        max_wall_seconds=120.0,
    ) as source:
        source.charge(work=3, geometry_queries=2)
    assert source.evidence is not None
    return NativeExecutionRecord(source.evidence, owner_id="receipt-source")


@_NATIVE
def test_actual_source_receipt_imports_once_and_retains_child_evidence() -> None:
    receipt = _actual_source_receipt()
    limits = MeshingLimits(
        maximum_work_units=100_000,
        maximum_geometry_queries=3,
        maximum_scratch_bytes=4 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    with native_volume_execution_budget(limits) as parent:
        from phydrax._meshcore import current_native_host_workspace

        storage = current_native_host_workspace()
        assert storage is not None
        _import_native_preparation(parent, storage, receipt)
        once = parent.remaining()
        _import_native_preparation(parent, storage, receipt)
        twice = parent.remaining()
        assert twice.remaining_work_units == once.remaining_work_units
        assert twice.remaining_geometry_queries == once.remaining_geometry_queries == 1
        with native_volume_execution_budget(
            limits,
            source_work_units=3,
            source_geometry_queries=2,
            borrow_active=False,
        ) as child:
            child.charge(work=2, geometry_queries=1)
        assert child.evidence is not None
        consumer = NativeExecutionRecord(
            child.evidence,
            preparation_evidence=receipt,
            owner_id=receipt.owner_id,
        )
        assert int(np.asarray(consumer.total_work_units)) == 5
        assert int(np.asarray(consumer.total_geometry_queries)) == 3
        assert consumer.preparation_evidence is receipt
        assert float(np.asarray(consumer.total_elapsed_seconds)) == (
            child.evidence.elapsed_seconds
            + float(np.asarray(receipt.total_elapsed_seconds))
        )
    assert parent.evidence is not None
    assert parent.evidence.status is MeshcoreStatus.OK
    assert parent.evidence.externally_charged_geometry_queries == 3
    assert parent.evidence.prior_elapsed_seconds == float(
        np.asarray(receipt.total_elapsed_seconds)
    )
    assert child.evidence.prior_elapsed_seconds == 0.0
    original = NativeExecutionRecord(
        parent.evidence,
        consumer_evidence=consumer,
        owner_id=receipt.owner_id,
    )
    assert original.consumer_evidence is consumer
    assert original.consumer_evidence.preparation_evidence is receipt
    assert int(np.asarray(original.total_work_units)) == int(
        parent.evidence.work_evidence[0]
    )
    assert int(np.asarray(original.total_geometry_queries)) == 3
    assert float(np.asarray(original.total_elapsed_seconds)) == (
        parent.evidence.elapsed_seconds + parent.evidence.prior_elapsed_seconds
    )


@_NATIVE
def test_source_receipt_query_refusal_never_dispatches_or_imports_duration() -> None:
    receipt = _actual_source_receipt()
    dispatched: list[str] = []
    limits = MeshingLimits(
        maximum_work_units=100_000,
        maximum_geometry_queries=1,
        maximum_scratch_bytes=4 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    with pytest.raises(MeshingFailure) as caught:
        with native_volume_execution_budget(limits) as parent:
            from phydrax._meshcore import current_native_host_workspace

            storage = current_native_host_workspace()
            assert storage is not None
            _import_native_preparation(parent, storage, receipt)
            dispatched.append("inadmissible")
    assert not dispatched
    assert caught.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert parent.evidence is not None
    assert parent.evidence.externally_charged_geometry_queries == 0
    assert parent.evidence.prior_elapsed_seconds == 0.0
    assert int(parent.evidence.work_evidence[4]) == 1


@_NATIVE
def test_negative_prior_duration_cannot_refund_original_allowance() -> None:
    with NativeExecutionBudget(
        max_work=3,
        max_geometry_queries=2,
        max_cavity_cells=10,
        max_scratch_bytes=4 * 1024**2,
        max_wall_seconds=120.0,
    ) as execution:
        with pytest.raises(ValueError, match="duration"):
            execution.import_preparation(work=3, geometry_queries=2, elapsed_seconds=-1.0)
        assert execution.remaining().remaining_work_units == 3
        assert execution.remaining().remaining_geometry_queries == 2
        execution.charge(work=3, geometry_queries=2)
    assert execution.evidence is not None
    assert execution.evidence.prior_elapsed_seconds == 0.0
    assert execution.evidence.externally_charged_work == 3
    assert execution.evidence.externally_charged_geometry_queries == 2


@_NATIVE
def test_consumer_constructor_preserves_original_operation_ownership() -> None:
    receipt = _actual_source_receipt()
    with NativeExecutionBudget(
        max_work=0,
        max_geometry_queries=0,
        max_cavity_cells=10,
        max_scratch_bytes=4 * 1024**2,
        max_wall_seconds=120.0,
    ) as consumer:
        consumer.charge()
    assert consumer.evidence is not None
    with pytest.raises(ValueError, match="different declared operations"):
        NativeExecutionRecord(
            consumer.evidence,
            consumer_evidence=receipt,
            owner_id="another-operation",
        )


@_NATIVE
@pytest.mark.parametrize("native_refusal", [False, True])
def test_volume_wrapper_retains_actual_cut_failure_prefix(native_refusal: bool) -> None:
    from phydrax._meshcore import current_native_execution_budget
    from phydrax.meshing._hex_cut_generation import _cut_stage

    limits = MeshingLimits(
        maximum_work_units=10,
        maximum_geometry_queries=10,
        maximum_scratch_bytes=4 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    receipts = []
    original = None
    prefix = None
    with pytest.raises(MeshingFailure) as caught:
        with native_volume_execution_budget(limits) as root:
            try:
                with _cut_stage("original_source_link", None, limits, receipts):
                    if native_refusal:
                        active = current_native_execution_budget()
                        assert active is not None
                        active.charge(work=11)
                    else:
                        raise MeshingFailure(
                            MeshingFailureCategory.RESOURCE_EXHAUSTED,
                            "Original source link refused.",
                        )
            except (MeshcoreError, MeshingFailure) as error:
                original = error
                prefix = error.cut_failure_prefix
                raise
    assert original is not None and prefix is not None
    assert caught.value.__cause__ is original
    assert caught.value.cut_failure_prefix is prefix
    assert prefix[0] is receipts[0]
    assert prefix[0].stage == "original_source_link"
    assert prefix[0].execution is not None
    assert root.evidence is not None
    assert root.evidence.work_evidence[0] == prefix[0].execution.work_evidence[0]


def _circle_plan_inputs(
    limits: MeshingLimits,
) -> tuple[
    NativeMeshingProvider,
    NativeCurveSource,
    CurveMeshingSpec,
    SpatialCoordinateContract,
]:
    import phydrax as phx

    meshing = phx.meshing
    atlas = (
        phx.geometry.Circle(
            (0.0, 0.0),
            1.0,
            feature_id="receipt-circle",
        )
        .compile()
        .boundary_atlas
    )
    source = meshing.NativeCurveSource(atlas, "receipt-revision")
    scope = meshing.MeshingScope(
        atlas.source_id,
        source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        1,
        "arcs",
        np.arange(4),
    )
    specification = meshing.CurveMeshingSpec(
        meshing.CellMeshingTarget(
            1,
            2,
            meshing.CellFamilyPolicy(required=("interval",)),
        ),
        scope,
        junctions=tuple(
            meshing.CurveJunction(f"j{i}", ((i, "end"), ((i + 1) % 4, "start")))
            for i in range(4)
        ),
        size_controls=(
            meshing.UniformSizeControl(
                scope,
                1.0,
                strength=meshing.SizeControlStrength.SOFT,
            ),
        ),
        limits=limits,
    )
    provider = meshing.NativeMeshingProvider(
        meshing.NativeMeshingOptions("curve_arc_length")
    )
    return provider, source, specification, phx.SpatialCoordinateContract.si()


def _plan_limits(work: int = 1_000_000) -> MeshingLimits:
    return MeshingLimits(
        maximum_work_units=work,
        maximum_geometry_queries=100_000,
        maximum_scratch_bytes=32 * 1024**2,
        maximum_wall_seconds=120.0,
    )


@_NATIVE
def test_active_plan_preparation_and_execution_retain_once_only_receipt() -> None:
    from phydrax._meshcore import current_native_host_workspace
    from phydrax.meshing.providers._native_curve import PreparedCurveNetwork

    provider, source, specification, contract = _circle_plan_inputs(_plan_limits())
    with native_volume_execution_budget(_plan_limits(2_000_000)) as root:
        plan = provider.plan(source, specification, coordinate_contract=contract)
        assert isinstance(plan.prepared, PreparedCurveNetwork)
        preparation = plan.preparation_evidence
        assert preparation is not None and preparation.owner_id == plan.plan_id
        assert int(np.asarray(preparation.work[0])) > 0
        storage = current_native_host_workspace()
        assert storage is not None
        before = root.remaining()
        _import_native_preparation(root, storage, preparation)
        after = root.remaining()
        assert after.remaining_work_units == before.remaining_work_units
        assert after.remaining_geometry_queries == before.remaining_geometry_queries
        result = plan.execute()
    assert result.execution_evidence is not None and root.evidence is not None
    assert result.execution_evidence.preparation_evidence is preparation
    assert int(root.evidence.work_evidence[0]) >= int(
        np.asarray(preparation.work[0] + result.execution_evidence.work[0]),
    )


@_NATIVE
def test_active_plan_stricter_preparation_cap_refuses_before_parent_renewal() -> None:
    provider, source, specification, contract = _circle_plan_inputs(_plan_limits(1))
    with pytest.raises(MeshingFailure) as caught:
        with native_volume_execution_budget(_plan_limits()) as root:
            provider.plan(source, specification, coordinate_contract=contract)
    assert caught.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert root.evidence is not None
    assert int(root.evidence.work_evidence[0]) <= 1
    assert int(root.evidence.work_evidence[3]) == 1


@_NATIVE
def test_retained_active_plan_carries_authentic_preparation_across_roots() -> None:
    provider, source, specification, contract = _circle_plan_inputs(_plan_limits())
    with native_volume_execution_budget(_plan_limits()) as original:
        plan = provider.plan(source, specification, coordinate_contract=contract)
    preparation = plan.preparation_evidence
    assert original.evidence is not None and preparation is not None
    original_work = np.asarray(preparation.work).copy()
    with native_volume_execution_budget(_plan_limits(2_000_000)) as consumer:
        result = plan.execute()
    assert result.execution_evidence is not None and consumer.evidence is not None
    assert result.execution_evidence.preparation_evidence is preparation
    np.testing.assert_array_equal(preparation.work, original_work)
    assert int(consumer.evidence.work_evidence[0]) >= int(
        np.asarray(result.execution_evidence.total_work_units),
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_affine_simplex_bernstein_preserves_exact_vertex_order(dimension: int) -> None:
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import bernstein_coefficients

    origin = Fraction(2, 7)
    polynomial = {(0,) * dimension: origin}
    increments = tuple(Fraction(axis + 1, 11) for axis in range(dimension))
    for axis, increment in enumerate(increments):
        polynomial[tuple(int(index == axis) for index in range(dimension))] = increment
    assert bernstein_coefficients(polynomial, "simplex", dimension) == (
        origin,
        *(origin + increment for increment in reversed(increments)),
    )


@_NATIVE
def test_retained_host_owner_reuses_complete_admitted_traversal() -> None:
    limits = _plan_limits()
    owner = (np.arange(16, dtype=np.int64), ("source", 3))
    with native_volume_execution_budget(limits) as root:
        from phydrax._meshcore import current_native_host_workspace

        storage = current_native_host_workspace()
        assert storage is not None
        storage.retain_owner(owner)
        before = root.remaining()
        bound = storage.bound
        storage.retain_owner(owner)
        after = root.remaining()
        assert after.remaining_work_units == before.remaining_work_units
        assert storage.bound == bound


@_NATIVE
def test_exact_triangle_preparation_preserves_orientation_rank_and_capacity() -> None:
    from fractions import Fraction

    from phydrax.geometry._mesh_certificates import _loop_triangles

    points = np.asarray(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1, 3), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1, 7), Fraction(0)),
            (Fraction(2, 3), Fraction(0), Fraction(0)),
        ),
        dtype=object,
    )
    loops = np.asarray(((0, 2, 1), (0, 1, 3)), dtype=np.int64)
    triangles, owners, failed, exhausted = _loop_triangles(points, loops, 54)
    np.testing.assert_array_equal(triangles, loops[:1])
    np.testing.assert_array_equal(owners, (0,))
    np.testing.assert_array_equal(failed, (False, True))
    assert not np.any(exhausted)
    triangles, owners, failed, exhausted = _loop_triangles(points, loops, 26)
    assert not triangles.size and not owners.size and not np.any(failed)
    assert np.all(exhausted)
