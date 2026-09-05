import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.export._array_archive import (
    fused_bem_result_archive_record,
    laplace_dp0_plan_archive_record,
    read_bem_array_archive,
    write_bem_array_archive,
)
from phydrax.geometry import MeshRegion
from phydrax.geometry.surface._contracts import SurfaceMetadata
from phydrax.geometry.surface._model import SurfaceModel
from phydrax.linalg import MaterializationPolicy
from phydrax.operators.integral.layer_potential._adaptive_boundary import (
    BoundaryEpochError,
    BoundaryMeshEpoch,
    BoundaryRefinementPolicy,
    mark_boundary_faces,
    refine_boundary_h,
)
from phydrax.operators.integral.layer_potential._fast_provider import (
    BEMBlockActionStatus,
    BEMFastCapabilityError,
    boundary_fast_provider_capabilities,
    FusedBlockedBEMAction3D,
    LaplaceDP0ExactNearProvider3D,
)
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
    prepare_laplace_single_layer_dp0_3d,
)


_VERTICES = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _prepared():
    policy = LaplaceSingleLayerDP0GalerkinPolicy3D(
        singular_order=3,
        near_ratio=1.0,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        target_block_size=3,
        source_block_size=2,
        dense_oracle=MaterializationPolicy(max_entries=100, max_bytes=4096),
    )
    return prepare_laplace_single_layer_dp0_3d(
        MeshRegion(_VERTICES, _FACES), policy=policy
    )


def _epoch():
    metadata = SurfaceMetadata(
        source_id="unit-tetrahedron",
        source_revision="1",
        coordinate_contract=SpatialCoordinateContract.si(),
        provenance=("analytic-test-fixture",),
    )
    surface = SurfaceModel.from_triangles(
        _VERTICES,
        _FACES,
        metadata,
        cell_global_ids=np.asarray([40, 10, 30, 20], dtype=np.int64),
        numeric_version="boundary-initial",
    )
    return BoundaryMeshEpoch(surface)


def test_fused_block_forward_transpose_and_per_column_status_match_serial(tmp_path):
    prepared = _prepared()
    action = FusedBlockedBEMAction3D(prepared, 3)
    right_hand_sides = jnp.asarray(
        [[0.3, -0.4, 0.2], [-0.2, 0.1, 0.7], [0.7, 0.2, -0.5], [0.5, 0.8, 0.1]]
    )

    forward = action.apply(right_hand_sides)
    transpose = action.transpose_apply(right_hand_sides)
    serial_forward = jnp.stack(
        tuple(prepared.strong_operator.mv(right_hand_sides[:, i]) for i in range(3)),
        axis=1,
    )
    serial_transpose = jnp.stack(
        tuple(
            prepared.strong_operator.transpose_mv(right_hand_sides[:, i])
            for i in range(3)
        ),
        axis=1,
    )

    assert jnp.allclose(forward.values, serial_forward, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(transpose.values, serial_transpose, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.array_equal(
        forward.column_status,
        jnp.full((3,), int(BEMBlockActionStatus.SUCCESS), dtype=jnp.int32),
    )
    assert not action.envelope.accelerated
    assert not action.envelope.continuum_certified

    invalid = right_hand_sides.at[2, 1].set(jnp.nan)
    failed = action.apply(invalid)
    assert int(failed.column_status[0]) == int(BEMBlockActionStatus.SUCCESS)
    assert int(failed.column_status[1]) == int(BEMBlockActionStatus.NONFINITE_INPUT)
    assert int(failed.column_status[2]) == int(BEMBlockActionStatus.SUCCESS)
    assert jnp.all(jnp.isnan(failed.values[:, 1]))
    assert jnp.all(jnp.isfinite(failed.values[:, (0, 2)]))
    plan_record = laplace_dp0_plan_archive_record(prepared)
    result_record = fused_bem_result_archive_record(failed, plan_record.plan_id)
    assert plan_record.metadata["report_id"] == prepared.assembly_report.report_id
    assert np.array_equal(
        result_record.arrays["column_status"], np.asarray([0, 1, 0], dtype=np.int32)
    )
    assert np.all(np.isnan(result_record.arrays["values"][:, 1]))
    plan_path = tmp_path / "prepared-plan.pba"
    result_path = tmp_path / "fused-result.pba"
    write_bem_array_archive(plan_path, plan_record)
    write_bem_array_archive(result_path, result_record)
    restored_plan = read_bem_array_archive(plan_path)
    restored_result = read_bem_array_archive(result_path)
    assert np.array_equal(
        restored_plan.arrays["pairs/exception_keys"],
        plan_record.arrays["pairs/exception_keys"],
    )
    assert np.array_equal(
        restored_result.arrays["column_status"],
        result_record.arrays["column_status"],
    )


def test_exact_near_local_blocks_and_diagonal_match_prepared_operator():
    prepared = _prepared()
    provider = LaplaceDP0ExactNearProvider3D(prepared, max_block_entries=32)
    dense = prepared.dense_oracle.matrix
    targets = jnp.asarray([0, 2], dtype=jnp.int32)
    sources = jnp.asarray([1, 3], dtype=jnp.int32)

    block = provider.local_block(targets, sources)
    diagonal = provider.diagonal()

    assert jnp.allclose(block.values, dense[np.ix_([0, 2], [1, 3])])
    assert jnp.all(block.exact_near_mask)
    assert jnp.allclose(diagonal.values, jnp.diag(dense))
    assert jnp.all(diagonal.exact_near_mask)
    assert jnp.all(diagonal.pair_classes == 0)
    assert jnp.all(block.accuracy_supported)
    assert jnp.all(diagonal.accuracy_supported)
    assert not diagonal.envelope.continuum_certified


def test_unsupported_3d_fast_capabilities_fail_closed():
    with pytest.raises(BEMFastCapabilityError, match="No 3D FMM"):
        boundary_fast_provider_capabilities("fmm-3d", ambient_dimension=3)
    with pytest.raises(BEMFastCapabilityError, match="not an accelerator"):
        boundary_fast_provider_capabilities(
            "blocked-direct-dp0-galerkin-3d",
            ambient_dimension=3,
            require_acceleration=True,
        )
    capabilities = boundary_fast_provider_capabilities(
        "laplace-fmm-2d", ambient_dimension=2, require_acceleration=True
    )
    assert capabilities.accelerated
    assert capabilities.exact_prepared_near
    assert not capabilities.exact_transpose


def test_deterministic_refinement_transfer_conserves_charge_and_invalidates_epoch():
    epoch = _epoch()
    policy = BoundaryRefinementPolicy(
        strategy="dorfler",
        fraction=0.25,
        max_marked_faces=2,
        max_target_faces=32,
    )
    indicators = jnp.ones((4,))
    first = mark_boundary_faces(epoch, indicators, policy)
    second = mark_boundary_faces(epoch, indicators, policy)

    assert jnp.array_equal(first.marked_face_global_ids, second.marked_face_global_ids)
    assert jnp.array_equal(first.marked_face_global_ids, jnp.asarray([10]))
    zero_marking = mark_boundary_faces(epoch, jnp.zeros((4,)), policy)
    assert zero_marking.marked_face_global_ids.size == 0
    with pytest.raises(ValueError, match="positive indicator"):
        refine_boundary_h(epoch, jnp.zeros((4,)), policy)

    refined = refine_boundary_h(epoch, indicators, policy)
    transfer = refined.transfer
    density = jnp.asarray([0.5, 1.25, -0.75, 2.0])
    prolonged = transfer.apply(density, epoch)
    source_charge = jnp.sum(transfer.source_face_areas * density)
    target_charge = jnp.sum(transfer.target_face_areas * prolonged)

    assert jnp.allclose(target_charge, source_charge, rtol=1.0e-13, atol=1.0e-13)
    assert float(transfer.maximum_area_defect) < 1.0e-14
    assert refined.target_epoch.parent_epoch_id == epoch.epoch_id
    assert refined.target_epoch.epoch_id != epoch.epoch_id
    assert epoch.surface_model is not None
    assert refined.target_epoch.surface_model is not None
    source_metadata = epoch.surface_model.metadata
    target_metadata = refined.target_epoch.surface_model.metadata
    assert target_metadata.source_id == source_metadata.source_id
    assert (
        target_metadata.coordinate_contract.spatial_id
        == source_metadata.coordinate_contract.spatial_id
    )
    assert target_metadata.source_revision != source_metadata.source_revision
    assert target_metadata.provenance[:-1] == source_metadata.provenance
    with pytest.raises(BoundaryEpochError, match="stale source epoch"):
        transfer.apply(density, refined.target_epoch)
    with pytest.raises(BoundaryEpochError, match="stale or foreign epoch"):
        epoch.validate_mesh(refined.target_epoch.mesh)

    target_dual = jnp.arange(transfer.target_face_count, dtype=density.dtype) + 0.25
    pullback = transfer.transpose_apply(target_dual, refined.target_epoch)
    assert jnp.allclose(jnp.vdot(target_dual, prolonged), jnp.vdot(pullback, density))
