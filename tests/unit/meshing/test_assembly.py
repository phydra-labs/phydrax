from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax.lifecycle import read_meshing_source_closure, write_meshing_source_closure
from phydrax.meshing import (
    ContactCoupling,
    MeshAssembly,
    MeshingEntityKind,
    MeshingScope,
    MeshPart,
)


# Empirical `test_contact_assembly_archive_preserves_custom_tolerance_and_physical_force`: matched max |delta|=1.77636e-15; joint allowance fraction=0.0296059, margin=97.0394%.
CONTACT_ARCHIVE_FORCE_RTOL = 1e-14
# Empirical `test_contact_assembly_archive_preserves_custom_tolerance_and_physical_force`: matched max |delta|=1.77636e-15; joint allowance fraction=0.0296059, margin=97.0394%.
CONTACT_ARCHIVE_FORCE_ATOL = 1e-14


@pytest.fixture(scope="module")
def assembly_parts() -> tuple[MeshPart, ...]:
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
    return parts


def test_assembly_identity_is_independent_of_part_order(
    assembly_parts: tuple[MeshPart, ...],
) -> None:
    assert (
        MeshAssembly(assembly_parts).assembly_id
        == MeshAssembly(tuple(reversed(assembly_parts))).assembly_id
    )


def test_assembly_retains_declared_carrier_families_and_point_selection(
    assembly_parts: tuple[MeshPart, ...],
) -> None:
    assembly = MeshAssembly(assembly_parts)
    iga = phx.discretization.iga
    assert isinstance(
        assembly.part("grid").carrier, phx.discretization.PreparedTensorGrid
    )
    assert isinstance(assembly.part("cloud").carrier, phx.discretization.PointCloudPlan)
    assert isinstance(assembly.part("spline").carrier, iga.IsogeometricPlan)
    points = np.asarray(
        [(x, y) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        assembly.part("cloud").point_coordinates(
            assembly_parts[1].scope(0, np.asarray([2, 4], dtype=np.int64))
        ),
        points[[2, 4]],
    )


def test_spline_part_scope_refuses_changed_geometry(
    assembly_parts: tuple[MeshPart, ...],
) -> None:
    iga = phx.discretization.iga
    spline = assembly_parts[2].carrier
    assert isinstance(spline, iga.IsogeometricPlan)
    spline_scope = assembly_parts[2].scope(2, np.asarray([0], dtype=np.int64))
    changed_geometry = iga.NURBSGeometryState(
        spline.geometry.control_points * 2, spline.geometry.weights
    )
    changed_spline = eqx.tree_at(lambda value: value.geometry, spline, changed_geometry)
    changed_part = MeshPart(
        "spline",
        changed_spline,
        coordinate_contract=assembly_parts[2].coordinate_contract,
    )
    assert changed_spline.plan_id == spline.plan_id
    with pytest.raises(ValueError, match=r"MeshPart\.scope"):
        changed_part.require_scope(spline_scope)


def test_spline_part_refuses_unsupported_entity_dimension(
    assembly_parts: tuple[MeshPart, ...],
) -> None:
    with pytest.raises(ValueError, match=r"MeshPart\.entity_dimension"):
        assembly_parts[2].scope(0, np.asarray([0], dtype=np.int64))


@pytest.fixture(scope="module")
def fluid_part() -> MeshPart:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    return MeshPart("fluid", grid, coordinate_contract=phx.SpatialCoordinateContract.si())


def test_assembly_refuses_duplicate_part_names(fluid_part: MeshPart) -> None:
    with pytest.raises(ValueError, match=r"MeshAssembly\.parts"):
        MeshAssembly((fluid_part, fluid_part))


def test_assembly_refuses_incompatible_reference_frames(fluid_part: MeshPart) -> None:
    other = MeshPart(
        "solid",
        fluid_part.carrier,
        coordinate_contract=phx.SpatialCoordinateContract(
            phx.SpatialCoordinateContract.si().length_unit, reference_frame="body"
        ),
    )
    with pytest.raises(ValueError, match=r"MeshAssembly\.parts"):
        MeshAssembly((fluid_part, other))


def test_part_scope_refuses_out_of_range_entity(fluid_part: MeshPart) -> None:
    with pytest.raises(ValueError, match=r"MeshPart\.scope"):
        fluid_part.scope(1, np.asarray([9], dtype=np.int64))


@pytest.fixture(scope="module")
def point_assembly() -> MeshAssembly:
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
    return MeshAssembly(
        (
            MeshPart(
                "cloud",
                plan,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
            ),
        )
    )


def test_point_assembly_archive_preserves_normal_bits(
    tmp_path: Path,
    point_assembly: MeshAssembly,
) -> None:
    assembly = point_assembly
    plan = assembly.part("cloud").carrier
    assert isinstance(plan, phx.discretization.PointCloudPlan)
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


def test_point_assembly_archive_refuses_stale_normals(
    tmp_path: Path,
    point_assembly: MeshAssembly,
) -> None:
    assembly = point_assembly
    plan = assembly.part("cloud").carrier
    assert isinstance(plan, phx.discretization.PointCloudPlan)
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
    with pytest.raises(ValueError, match=r"PointCloudPlan\.plan_id"):
        write_meshing_source_closure(tmp_path / "corrupted-points.zip", corrupted)


@pytest.fixture(scope="module")
def contact_assembly() -> MeshAssembly:
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
    return MeshAssembly((source, target), couplings=(coupling,))


def test_contact_assembly_archive_preserves_custom_tolerance_and_physical_force(
    tmp_path: Path,
    contact_assembly: MeshAssembly,
) -> None:
    assembly = contact_assembly
    coupling = assembly.couplings[0]
    assert isinstance(coupling, ContactCoupling)
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
    np.testing.assert_allclose(
        right, expected, rtol=CONTACT_ARCHIVE_FORCE_RTOL, atol=CONTACT_ARCHIVE_FORCE_ATOL
    )
    np.testing.assert_array_equal(left, -right)


def test_contact_stored_fields_preserve_retained_endpoint_revisions(
    contact_assembly: MeshAssembly,
) -> None:
    coupling = contact_assembly.couplings[0]
    assert isinstance(coupling, ContactCoupling)
    rebuilt = ContactCoupling.from_stored_fields(
        coupling,
        source=contact_assembly.part("source"),
        target=contact_assembly.part("target"),
    )
    assert rebuilt.coupling_id == coupling.coupling_id
    assert rebuilt.tolerance == 1e-4
    for name in ("source_rows", "normals", "reference_gap"):
        expected = np.asarray(getattr(coupling, name))
        actual = np.asarray(getattr(rebuilt, name))
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def test_contact_stored_fields_refuse_changed_reference_gap_bit(
    contact_assembly: MeshAssembly,
) -> None:
    coupling = contact_assembly.couplings[0]
    assert isinstance(coupling, ContactCoupling)
    changed = np.nextafter(np.asarray(coupling.reference_gap)[0], np.float64(np.inf))
    faulty = eqx.tree_at(
        lambda value: value.reference_gap,
        coupling,
        coupling.reference_gap.at[0].set(changed),
    )
    with pytest.raises(ValueError, match=r"ContactCoupling\.stored_fields"):
        ContactCoupling.from_stored_fields(
            faulty,
            source=contact_assembly.part("source"),
            target=contact_assembly.part("target"),
        )


def test_contact_stored_fields_refuse_changed_normal_bit(
    contact_assembly: MeshAssembly,
) -> None:
    coupling = contact_assembly.couplings[0]
    assert isinstance(coupling, ContactCoupling)
    changed = np.nextafter(np.asarray(coupling.normals)[0, 0], np.float64(np.inf))
    faulty = eqx.tree_at(
        lambda value: value.normals, coupling, coupling.normals.at[0, 0].set(changed)
    )
    with pytest.raises(ValueError, match=r"ContactCoupling\.stored_fields"):
        ContactCoupling.from_stored_fields(
            faulty,
            source=contact_assembly.part("source"),
            target=contact_assembly.part("target"),
        )


def test_contact_stored_fields_refuse_foreign_endpoint_revision(
    contact_assembly: MeshAssembly,
) -> None:
    coupling = contact_assembly.couplings[0]
    assert isinstance(coupling, ContactCoupling)
    with pytest.raises(ValueError, match=r"MeshPart\.scope"):
        ContactCoupling.from_stored_fields(
            coupling,
            source=contact_assembly.part("target"),
            target=contact_assembly.part("source"),
        )


@pytest.fixture(scope="module")
def spline_assembly() -> MeshAssembly:
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
    return assembly


def test_spline_assembly_archive_preserves_nonuniform_gauge(
    tmp_path: Path,
    spline_assembly: MeshAssembly,
) -> None:
    assembly = spline_assembly
    part = assembly.part("spline")
    spline = part.carrier
    assert isinstance(spline, phx.discretization.iga.IsogeometricPlan)
    receipt = write_meshing_source_closure(tmp_path / "spline.zip", assembly)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, MeshAssembly)
    actual = restored.part("spline").carrier
    assert isinstance(actual, phx.discretization.iga.IsogeometricPlan)
    np.testing.assert_array_equal(
        np.asarray(actual.geometry.weights).view(np.uint64),
        np.asarray(spline.geometry.weights).view(np.uint64),
    )
    assert restored.part("spline").part_id == part.part_id


def test_spline_assembly_archive_refuses_stale_geometry(
    tmp_path: Path,
    spline_assembly: MeshAssembly,
) -> None:
    part = spline_assembly.part("spline")
    spline = part.carrier
    assert isinstance(spline, phx.discretization.iga.IsogeometricPlan)
    changed_weights = spline.geometry.weights.at[0, 0].set(
        np.nextafter(np.asarray(spline.geometry.weights)[0, 0], np.float64(np.inf)),
    )
    changed_spline = eqx.tree_at(
        lambda value: value.geometry.weights, spline, changed_weights
    )
    stale_part = eqx.tree_at(lambda value: value.carrier, part, changed_spline)
    with pytest.raises(ValueError, match=r"stored_fields\.MeshPart"):
        write_meshing_source_closure(
            tmp_path / "stale-spline.zip", MeshAssembly((stale_part,))
        )


@pytest.fixture(scope="module")
def mesh_scope_state() -> tuple[phx.discretization.CellMesh, MeshingScope]:
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
    return mesh, scope


def test_mesh_backed_scope_checkpoint_preserves_inventory(
    tmp_path: Path,
    mesh_scope_state: tuple[phx.discretization.CellMesh, MeshingScope],
) -> None:
    mesh, scope = mesh_scope_state
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


def test_mesh_backed_scope_checkpoint_refuses_missing_revision(
    tmp_path: Path,
    mesh_scope_state: tuple[phx.discretization.CellMesh, MeshingScope],
) -> None:
    mesh, scope = mesh_scope_state
    limits = ArrayArchiveLimits(max_members=4096, max_manifest_nesting=128)
    changed = mesh.with_coordinates(
        mesh.coordinates + jnp.asarray([0.125, 0.0]),
        numeric_version="changed-scope-owner",
    )
    with pytest.raises(ValueError, match=r"MeshingScope\.local_mesh"):
        write_meshing_source_closure(
            tmp_path / "stale-mesh-scope.zip", (changed, scope), limits=limits
        )
