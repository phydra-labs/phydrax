from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._assembly import MeshAssembly, MeshPart


def test_assembly_contracts() -> None:
    contract = phx.SpatialCoordinateContract.si()
    tensor = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2),
            phx.discretization.UniformCellAxisSpec(2),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    points = np.asarray([(x, y) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
    cloud = phx.discretization.PointCloudPlan(points, np.ones(9), neighbors=9)
    iga = phx.discretization.iga
    axis = iga.BSplineGrid.open_uniform(2, 1)
    xx, yy = jnp.meshgrid(axis.greville_abscissae, axis.greville_abscissae, indexing="ij")
    spline = iga.IsogeometricPlan.isoparametric(
        (axis, axis),
        iga.NURBSGeometryState(jnp.stack((xx, yy), axis=-1), jnp.ones(xx.shape)),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    parts = tuple(
        MeshPart(name, carrier, coordinate_contract=contract)
        for name, carrier in (("grid", tensor), ("cloud", cloud), ("spline", spline))
    )
    assembly = MeshAssembly(parts)
    assert assembly.assembly_id == MeshAssembly(tuple(reversed(parts))).assembly_id
    assert isinstance(
        assembly.part("grid").carrier, phx.discretization.PreparedTensorGrid
    )
    assert isinstance(assembly.part("cloud").carrier, phx.discretization.PointCloudPlan)
    assert isinstance(assembly.part("spline").carrier, iga.IsogeometricPlan)
    np.testing.assert_allclose(
        # ty: ignore[invalid-argument-type]
        assembly.part("cloud").point_coordinates(parts[1].scope(0, [2, 4])),
        points[[2, 4]],
    )
    # ty: ignore[invalid-argument-type]
    spline_scope = parts[2].scope(2, [0])
    changed_geometry = iga.NURBSGeometryState(
        spline.geometry.control_points * 2, spline.geometry.weights
    )
    changed_spline = eqx.tree_at(lambda value: value.geometry, spline, changed_geometry)
    changed_part = MeshPart("spline", changed_spline, coordinate_contract=contract)
    assert changed_spline.plan_id == spline.plan_id
    with pytest.raises(ValueError, match="stale"):
        changed_part.require_scope(spline_scope)
    with pytest.raises(ValueError, match="coefficient vertices"):
        # ty: ignore[invalid-argument-type]
        parts[2].scope(0, [0])
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    part = MeshPart("fluid", grid, coordinate_contract=phx.SpatialCoordinateContract.si())
    with pytest.raises(ValueError, match="unique name"):
        MeshAssembly((part, part))
    other = MeshPart(
        "solid",
        grid,
        coordinate_contract=phx.SpatialCoordinateContract(
            phx.SpatialCoordinateContract.si().length_unit, reference_frame="body"
        ),
    )
    with pytest.raises(ValueError, match="coordinate contract"):
        MeshAssembly((part, other))
    with pytest.raises(ValueError, match="unknown or inactive"):
        # ty: ignore[invalid-argument-type]
        part.scope(1, [9])


def test_point_assembly_archive_preserves_normal_bits_and_rejects_stale_science(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    points = np.asarray(
        [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 0.5, 1.0)],
        dtype=np.float64,
    )
    boundary = np.zeros((12,), dtype=np.bool_)
    boundary[0] = True
    normals = np.zeros((12, 3), dtype=np.float64)
    normals[0] = (0.6028348487212272, 1.990084546552389, 1.47836943401503)
    plan = phx.discretization.PointCloudPlan(
        points,
        np.ones((12,), dtype=np.float64),
        boundary_mask=boundary,
        boundary_normals=normals,
        neighbors=12,
    )
    assembly = MeshAssembly(
        (
            MeshPart(
                "cloud",
                plan,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
            ),
        )
    )
    receipt = write_meshing_source_closure(tmp_path / "points.zip", assembly)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, MeshAssembly)
    carrier = restored.part("cloud").carrier
    assert isinstance(carrier, phx.discretization.PointCloudPlan)
    np.testing.assert_array_equal(
        np.asarray(carrier.boundary_normals).view(np.uint64),
        np.asarray(plan.boundary_normals).view(np.uint64),
    )
    assert carrier.plan_id == plan.plan_id

    changed_normals = plan.boundary_normals.at[0, 0].set(
        np.nextafter(np.asarray(plan.boundary_normals)[0, 0], np.float64(np.inf)),
    )
    corrupted_plan = eqx.tree_at(
        lambda value: value.boundary_normals, plan, changed_normals
    )
    corrupted = MeshAssembly(
        (
            MeshPart(
                "cloud",
                corrupted_plan,
                coordinate_contract=assembly.coordinate_contract,
            ),
        )
    )
    with pytest.raises(ValueError):
        write_meshing_source_closure(tmp_path / "corrupted-points.zip", corrupted)


def test_contact_assembly_archive_preserves_custom_tolerance_and_physical_force(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._coupling import ContactCoupling

    plan = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(2),),
        axis_names=("x",),
    )
    contract = phx.SpatialCoordinateContract.si()
    source = MeshPart(
        "source",
        plan.prepare(np.asarray(((0.0,), (1.0,)), dtype=np.float64)),
        coordinate_contract=contract,
    )
    target = MeshPart(
        "target",
        plan.prepare(np.asarray(((1.0,), (2.0,)), dtype=np.float64)),
        coordinate_contract=contract,
    )
    ids = np.asarray((0,), dtype=np.int64)
    coupling = ContactCoupling(
        source,
        target,
        source.scope(0, ids),
        target.scope(0, ids),
        np.asarray(((1.000001,),), dtype=np.float64),
        tolerance=1e-4,
    )
    assembly = MeshAssembly((source, target), couplings=(coupling,))
    receipt = write_meshing_source_closure(tmp_path / "contact.zip", assembly)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, MeshAssembly)
    actual = restored.couplings[0]
    assert isinstance(actual, ContactCoupling)
    assert actual.tolerance == 1e-4
    np.testing.assert_array_equal(actual.reference_gap, coupling.reference_gap)
    left, right = actual.penalty_forces(
        np.asarray(((0.0,),), dtype=np.float64),
        np.asarray(((-1.5,),), dtype=np.float64),
        10.0,
    )
    expected = np.asarray(((10.0 * 0.5000005 * 1.000001,),), dtype=np.float64)
    np.testing.assert_allclose(right, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(left, -right)


def test_spline_assembly_archive_preserves_nonuniform_gauge_and_refuses_stale_geometry(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    iga = phx.discretization.iga
    axis = iga.BSplineGrid.open_uniform(2, 1)
    x, y = jnp.meshgrid(axis.greville_abscissae, axis.greville_abscissae, indexing="ij")
    weights = jnp.asarray(
        (
            (2.1240769803833386, 1.6247919790586969, 2.822069264029495),
            (1.7916438687301306, 0.28393723586614106, 2.5695945618510003),
            (1.964376843680476, 1.6506833062545976, 0.38139636203310107),
        ),
        dtype=jnp.float64,
    )
    spline = iga.IsogeometricPlan.isoparametric(
        (axis, axis),
        iga.NURBSGeometryState(jnp.stack((x, y), axis=-1), weights),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    part = MeshPart(
        "spline", spline, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    assembly = MeshAssembly((part,))
    np.testing.assert_array_equal(
        np.asarray(spline.geometry.weights).view(np.uint64),
        np.asarray(weights).view(np.uint64),
    )
    receipt = write_meshing_source_closure(tmp_path / "spline.zip", assembly)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, MeshAssembly)
    actual = restored.part("spline").carrier
    assert isinstance(actual, iga.IsogeometricPlan)
    np.testing.assert_array_equal(
        np.asarray(actual.geometry.weights).view(np.uint64),
        np.asarray(spline.geometry.weights).view(np.uint64),
    )
    assert restored.part("spline").part_id == part.part_id

    changed_weights = spline.geometry.weights.at[0, 0].set(
        np.nextafter(np.asarray(spline.geometry.weights)[0, 0], np.float64(np.inf)),
    )
    changed_spline = eqx.tree_at(
        lambda value: value.geometry.weights, spline, changed_weights
    )
    stale_part = eqx.tree_at(lambda value: value.carrier, part, changed_spline)
    with pytest.raises(ValueError):
        write_meshing_source_closure(
            tmp_path / "stale-spline.zip", MeshAssembly((stale_part,))
        )


def test_mesh_backed_scope_checkpoint_preserves_inventory_and_refuses_missing_revision(
    tmp_path: Path,
) -> None:
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope

    mesh = phx.discretization.CellMesh(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        (
            phx.discretization.CellBlock(
                "cell", "triangle", np.asarray([[0, 1, 2]], dtype=np.int32)
            ),
        ),
    )
    edges = mesh.entity_set(1)
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        edges.entity_set_id,
        np.asarray(edges.entity_ids)[:2],
        local_mesh=mesh,
    )
    limits = ArrayArchiveLimits(max_members=4096, max_manifest_nesting=128)
    receipt = write_meshing_source_closure(
        tmp_path / "mesh-scope.zip", (mesh, scope), limits=limits
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    restored_mesh, restored_scope = restored
    assert restored_scope.scope_id == scope.scope_id
    assert restored_scope.logical_entity_set_id == scope.logical_entity_set_id
    np.testing.assert_array_equal(
        restored_scope.global_entity_ids, scope.global_entity_ids
    )
    np.testing.assert_array_equal(
        restored_scope.lower(restored_mesh).entity_ids, scope.entity_ids
    )
    changed = mesh.with_coordinates(
        mesh.coordinates + jnp.asarray([0.125, 0.0]),
        numeric_version="changed-scope-owner",
    )
    with pytest.raises(ValueError, match="authenticated local constructor context"):
        write_meshing_source_closure(
            tmp_path / "stale-mesh-scope.zip", (changed, scope), limits=limits
        )
