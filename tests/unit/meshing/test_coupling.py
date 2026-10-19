from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._assembly import MeshAssembly
from phydrax.meshing._coupling import (
    ConformalCoupling,
    ContactCoupling,
    OversetCoupling,
    PeriodicCoupling,
)
from phydrax.meshing._mesh_part import MeshPart


# Empirical `test_periodic_coupling_transforms_rank_two_tensors`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
PERIODIC_TENSOR_COMPONENT_ATOL = 1e-14


def _part(name: Any, coordinates: Any = None, cells: Any = None) -> Any:
    points = (
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        if coordinates is None
        else np.asarray(coordinates)
    )
    triangles = (
        np.asarray(((0, 1, 2),), dtype=np.int32)
        if cells is None
        else np.asarray(cells, dtype=np.int32)
    )
    mesh = phx.discretization.CellMesh.from_triangles(points, triangles)
    return MeshPart(
        name, phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    )


def test_coupling_scenario_1() -> None:
    source = _part("left")
    target = _part("right", ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)), ((1, 0, 2),))
    source_scope, target_scope = source.scope(0, [0, 1]), target.scope(0, [0, 1])
    coupling = ConformalCoupling(
        source, target, source_scope, target_scope, source_ids=np.asarray([1, 0])
    )
    np.testing.assert_array_equal(coupling.transfer(jnp.asarray([3.0, 8.0])), [8.0, 3.0])
    np.testing.assert_array_equal(coupling.transpose(jnp.asarray([2.0, 7.0])), [7.0, 2.0])
    moved = _part("right", ((2.0, 0.0), (1.0, 0.0), (1.0, 1.0)), ((1, 0, 2),))
    with pytest.raises(ValueError, match=r"MeshPart\.scope"):
        MeshAssembly((source, moved), couplings=(coupling,))
    with pytest.raises(ValueError, match="coincide"):
        ConformalCoupling(source, target, source_scope, target_scope)
    with pytest.raises(ValueError, match="bijection"):
        ConformalCoupling(
            source, target, source_scope, target_scope, source_ids=np.asarray([0, 0])
        )
    source = _part("source")
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)))
    translation = np.asarray((2.0, 1.0))
    target = _part(
        "target", np.asarray(source.carrier.mesh.coordinates) @ rotation.T + translation
    )
    coupling = PeriodicCoupling(
        source,
        target,
        source.scope(0, [0, 1, 2]),
        target.scope(0, [0, 1, 2]),
        rotation,
        translation,
    )
    values = jnp.asarray(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))
    np.testing.assert_allclose(
        coupling.transfer_vectors(values), np.asarray(values) @ rotation.T
    )
    with pytest.raises(ValueError, match="isometry"):
        PeriodicCoupling(
            source,
            target,
            source.scope(0, [0]),
            target.scope(0, [0]),
            2 * rotation,
            translation,
        )
    source = _part("source")
    target = _part("target", np.asarray(source.carrier.mesh.coordinates) - [0.0, 0.1])
    coupling = ContactCoupling(
        source,
        target,
        source.scope(0, [0, 1, 2]),
        target.scope(0, [0, 1, 2]),
        np.tile([0.0, 1.0], (3, 1)),
    )
    zero = jnp.zeros((3, 2))
    left, right = coupling.penalty_forces(zero, zero, 10.0)
    np.testing.assert_allclose(right, np.tile([0.0, 1.0], (3, 1)))
    np.testing.assert_allclose(jnp.sum(left + right, axis=0), 0.0)
    separated = zero.at[:, 1].set(0.2)
    np.testing.assert_allclose(coupling.penalty_forces(zero, separated, 10.0)[1], 0.0)
    derivative = jax.grad(
        lambda displacement: jnp.sum(coupling.penalty_forces(zero, displacement, 10.0)[1])
    )(zero)
    np.testing.assert_allclose(derivative, np.tile([0.0, -10.0], (3, 1)))


def test_coupling_scenario_2() -> None:
    source = _part("donor")
    target = _part("receptor", ((0.25, 0.25), (0.5, 0.25), (0.25, 0.5)))
    overlay = OversetCoupling(
        source,
        target,
        source.scope(0, [0, 1, 2]),
        target.scope(0, [0]),
        np.asarray([[0, 1, 2, -1]]),
        np.asarray([[0.5, 0.25, 0.25, 0.0]]),
        hole_scope=target.scope(0, [2]),
    )
    values = jnp.asarray([2.0, 5.0, 7.0])
    np.testing.assert_allclose(overlay.transfer(jnp.ones(3)), 1.0)
    np.testing.assert_allclose(overlay.transfer(values), [4.0])
    cotangent = jnp.asarray([3.0])
    np.testing.assert_allclose(
        jnp.vdot(overlay.transfer(values), cotangent),
        jnp.vdot(values, overlay.transpose(cotangent)),
    )
    np.testing.assert_allclose(
        jax.grad(lambda field: jnp.vdot(overlay.transfer(field), cotangent))(values),
        overlay.transpose(cotangent),
    )
    MeshAssembly((source, target), couplings=(overlay,))
    duplicate_donor = _part("other-donor")
    duplicate = OversetCoupling(
        duplicate_donor,
        target,
        duplicate_donor.scope(0, [0]),
        target.scope(0, [0]),
        np.asarray([[0]]),
        np.asarray([[1.0]]),
    )
    with pytest.raises(ValueError, match=r"MeshAssembly\.couplings"):
        MeshAssembly((source, target, duplicate_donor), couplings=(overlay, duplicate))
    source, target = _part("source"), _part("target")
    args = (source, target, source.scope(0, [0, 1]), target.scope(0, [0]))
    with pytest.raises(ValueError, match="summing to one"):
        OversetCoupling(*args, np.asarray([[0, 1]]), np.asarray([[0.2, 0.3]]))
    with pytest.raises(ValueError, match="non-negative weights"):
        OversetCoupling(*args, np.asarray([[0, 1]]), np.asarray([[1.1, -0.1]]))
    with pytest.raises(ValueError, match="donor IDs"):
        OversetCoupling(*args, np.asarray([[2]]), np.asarray([[1.0]]))
    with pytest.raises(ValueError, match="disjoint"):
        OversetCoupling(
            *args, np.asarray([[0]]), np.asarray([[1.0]]), hole_scope=target.scope(0, [0])
        )
    overlay = OversetCoupling(*args, np.asarray([[0, 1]]), np.asarray([[1.0, 0.0]]))
    np.testing.assert_allclose(overlay.transfer(jnp.asarray([2.0, jnp.nan])), [2.0])
    donor = _grid_part("donor", 4)
    receptor = _grid_part("receptor", 3, offset=(0.13, 0.21), scale=0.6)
    source_scope, target_scope = _vertex_scope(donor), _vertex_scope(receptor)
    overlay = OversetCoupling.search(donor, receptor, source_scope, target_scope)
    assert not overlay.conservative
    evidence = overlay.search_evidence
    # ty: ignore[unresolved-attribute]
    assert np.all(np.asarray(evidence.found))
    # ty: ignore[unresolved-attribute]
    assert evidence.method.endswith("interpolation")
    donors = np.asarray(donor.point_coordinates(source_scope))
    receptors = np.asarray(receptor.point_coordinates(target_scope))
    affine = jnp.asarray(donors @ np.asarray([2.0, -3.0]) + 0.5)
    np.testing.assert_allclose(
        overlay.transfer(affine), receptors @ np.asarray([2.0, -3.0]) + 0.5, atol=1e-12
    )
    cotangent = jnp.linspace(1.0, 2.0, receptors.shape[0])
    np.testing.assert_allclose(
        jnp.vdot(overlay.transfer(affine), cotangent),
        jnp.vdot(affine, overlay.transpose(cotangent)),
    )
    MeshAssembly((donor, receptor), couplings=(overlay,))
    donor = _grid_part("donor", 2)
    receptor = _part("receptor", ((0.2, 0.2), (0.7, 0.3), (1.3, 1.2)))
    target_scope = _vertex_scope(receptor)
    with pytest.raises(phx.meshing.CouplingSearchError, match="outside=1") as outside:
        OversetCoupling.search(donor, receptor, _vertex_scope(donor), target_scope)
    evidence = outside.value.evidence
    status = np.asarray(evidence.status)
    assert np.count_nonzero(status == phx.meshing.CouplingSearchStatus.OUTSIDE) == 1
    assert np.all(np.asarray(evidence.donor_ids)[status != 0] == -1)
    coordinates = np.asarray(donor.carrier.mesh.coordinates)
    identifiers = np.asarray(donor.carrier.mesh.vertex_global_ids)
    first = receptor.scope(0, np.asarray(target_scope.entity_ids)[:1])
    far_corner = np.all(np.isclose(coordinates, 1.0), axis=1)
    corner = OversetCoupling.search(
        donor, receptor, _vertex_scope(donor, identifiers[~far_corner]), first
    )
    # ty: ignore[unresolved-attribute]
    assert np.all(np.asarray(corner.search_evidence.found))
    center = np.all(np.isclose(coordinates, 0.5), axis=1)
    with pytest.raises(phx.meshing.CouplingSearchError, match="excluded_donor=1"):
        OversetCoupling.search(
            donor, receptor, _vertex_scope(donor, identifiers[~center]), first
        )
    source = _grid_part("source", 2)
    coordinates = np.asarray(source.carrier.mesh.coordinates)
    permutation = np.random.default_rng(5).permutation(coordinates.shape[0])
    target = _part(
        "target",
        coordinates[permutation] + [0.0, 0.01],
        np.argsort(permutation)[np.asarray(source.carrier.mesh.blocks[0].vertices)],
    )
    source_scope, target_scope = _vertex_scope(source), _vertex_scope(target)
    normals = np.tile([0.0, 1.0], (coordinates.shape[0], 1))
    coupling = ContactCoupling.search(
        source, target, source_scope, target_scope, normals, capture_radius=0.05
    )
    np.testing.assert_allclose(
        coupling.transfer(source.point_coordinates(source_scope)),
        np.asarray(target.point_coordinates(target_scope)) - [0.0, 0.01],
        atol=1e-12,
    )
    np.testing.assert_allclose(coupling.reference_gap, 0.01, atol=1e-12)
    # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(coupling.search_evidence.distances, 0.01, atol=1e-12)
    with pytest.raises(phx.meshing.CouplingSearchError, match="outside=9"):
        ContactCoupling.search(
            source, target, source_scope, target_scope, normals, capture_radius=0.001
        )
    crowded = coordinates.copy()
    crowded[np.all(np.isclose(coordinates, 1.0), axis=1)] = (0.55, 0.52)
    crowded_part = _part(
        "crowded", crowded, np.asarray(source.carrier.mesh.blocks[0].vertices)
    )
    with pytest.raises(phx.meshing.CouplingSearchError, match="ambiguous=2"):
        ContactCoupling.search(
            source,
            crowded_part,
            source_scope,
            _vertex_scope(crowded_part),
            normals,
            capture_radius=1.0,
        )


def _grid_part(
    name: Any, count: Any, *, offset: Any = (0.0, 0.0), scale: Any = 1.0
) -> Any:
    xs, ys = np.meshgrid(
        np.linspace(0.0, scale, count + 1),
        np.linspace(0.0, scale, count + 1),
        indexing="ij",
    )
    points = np.stack((xs.ravel(), ys.ravel()), axis=1) + np.asarray(offset)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    lower, right = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    upper, left = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate(
        (np.stack((lower, right, upper), 1), np.stack((lower, upper, left), 1))
    )
    return _part(name, points, triangles)


def _vertex_scope(part: Any, ids: Any = None) -> Any:
    identifiers = (
        np.asarray(part.carrier.mesh.vertex_global_ids)
        if ids is None
        else np.asarray(ids)
    )
    return part.scope(0, np.sort(identifiers))


_SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_SQUARE_TRIANGLES = ((0, 1, 2), (0, 2, 3))


def _edge_id(part: Any, first: int, second: int) -> int:
    mesh = part.carrier.mesh
    edges = np.sort(np.asarray(mesh.connectivity.edges), axis=1)
    row = np.flatnonzero(np.all(edges == sorted((first, second)), axis=1))[0]
    return int(np.asarray(mesh.entity_set(1).entity_ids)[row])


def test_periodic_coupling_binds_oriented_edge_orbits() -> None:
    part = _part("square", _SQUARE, _SQUARE_TRIANGLES)
    # Translating the bottom edge 0->1 upward reverses the stored top edge 2->3
    # orientation, while the left edge 0->3 maps onto the right edge 1->2.
    vertical = PeriodicCoupling(
        part,
        part,
        part.scope(1, [_edge_id(part, 0, 1)]),
        part.scope(1, [_edge_id(part, 2, 3)]),
        np.eye(2),
        np.asarray((0.0, 1.0)),
    )
    horizontal = PeriodicCoupling(
        part,
        part,
        part.scope(1, [_edge_id(part, 0, 3)]),
        part.scope(1, [_edge_id(part, 1, 2)]),
        np.eye(2),
        np.asarray((1.0, 0.0)),
    )

    np.testing.assert_array_equal(vertical.orientations, (-1,))
    np.testing.assert_array_equal(horizontal.orientations, (1,))
    np.testing.assert_array_equal(
        vertical.transfer_oriented(jnp.asarray((5.0,))), (-5.0,)
    )
    np.testing.assert_array_equal(
        horizontal.transfer_oriented(jnp.asarray((5.0,))), (5.0,)
    )
    with pytest.raises(ValueError, match=r"PeriodicCoupling\.entity_pair_corners"):
        PeriodicCoupling(
            part,
            part,
            part.scope(1, [_edge_id(part, 0, 1)]),
            part.scope(1, [_edge_id(part, 2, 3)]),
            np.eye(2),
            np.asarray((1.0, 0.0)),
        )


def test_periodic_coupling_transforms_rank_two_tensors() -> None:
    source = _part("source")
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)))
    translation = np.asarray((2.0, 1.0))
    target = _part(
        "target", np.asarray(source.carrier.mesh.coordinates) @ rotation.T + translation
    )
    coupling = PeriodicCoupling(
        source,
        target,
        source.scope(0, [0, 1, 2]),
        target.scope(0, [0, 1, 2]),
        rotation,
        translation,
    )
    tensors = np.random.default_rng(3).normal(size=(3, 2, 2))

    expected = np.stack([rotation @ tensor @ rotation.T for tensor in tensors])
    np.testing.assert_allclose(
        coupling.transfer_tensors(jnp.asarray(tensors)),
        expected,
        atol=PERIODIC_TENSOR_COMPONENT_ATOL,
    )


def test_periodic_coupling_refuses_quotient_orbit_reidentification() -> None:
    points = np.asarray(_SQUARE)
    block = phx.discretization.CellBlock(
        "torus", "triangle", np.asarray(_SQUARE_TRIANGLES, dtype=np.int32)
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    periodic = phx.discretization.PeriodicMeshTopology(
        lifted,
        phx.discretization.PeriodicCell(np.eye(2)),
        np.zeros(4, dtype=np.int64),
        points.astype(np.int64),
    )
    torus = MeshPart(
        "torus",
        phx.meshing.certify_cell_mesh(
            phx.discretization.CellMesh(points, (block,), periodic_topology=periodic),
            phx.SpatialCoordinateContract.si(),
        ),
    )

    with pytest.raises(ValueError, match=r"PeriodicCoupling\.entity_pairs"):
        PeriodicCoupling(
            torus,
            torus,
            torus.scope(0, np.asarray((0,))),
            torus.scope(0, np.asarray((1,))),
            np.eye(2),
            np.asarray((1.0, 0.0)),
        )


def test_periodic_constraint_binds_explicit_entity_orbits() -> None:
    part = _part("square", _SQUARE, _SQUARE_TRIANGLES)
    left, right = (
        part.scope(1, [_edge_id(part, 0, 3)]),
        part.scope(1, [_edge_id(part, 1, 2)]),
    )
    transform = np.eye(3)
    transform[0, 2] = 1.0

    bound = phx.meshing.PeriodicConstraint(
        left,
        right,
        transform,
        source_entity_ids=left.entity_ids,
        orientations=np.asarray((1,)),
    )
    unbound = phx.meshing.PeriodicConstraint(left, right, transform)

    np.testing.assert_array_equal(bound.source_entity_ids, left.entity_ids)
    assert bound.orientation_preserving
    assert bound.constraint_id != unbound.constraint_id
    with pytest.raises(ValueError, match="orientation witnesses"):
        phx.meshing.PeriodicConstraint(
            left, right, transform, source_entity_ids=left.entity_ids
        )
    with pytest.raises(ValueError, match="bijectively"):
        phx.meshing.PeriodicConstraint(
            left,
            right,
            transform,
            source_entity_ids=right.entity_ids,
            orientations=np.asarray((1,)),
        )
    scaled = transform.copy()
    scaled[0, 0] = 2.0
    with pytest.raises(ValueError, match="isometry"):
        phx.meshing.PeriodicConstraint(left, right, scaled)


def _stored_contact_fixture() -> tuple[ContactCoupling, MeshPart, MeshPart]:
    source = _part("contact-source")
    target = _part(
        "contact-target",
        np.asarray(((0.0, -0.1), (1.0, -0.1), (0.0, 0.9)), dtype=np.float64),
    )
    value = ContactCoupling(
        source,
        target,
        source.scope(0, [0]),
        target.scope(0, [0]),
        np.asarray(((0.0, 1.000001),), dtype=np.float64),
        tolerance=1.0e-4,
    )
    return value, source, target


def test_contact_stored_fields_preserve_tolerance_scope_and_physical_force() -> None:
    value, source, target = _stored_contact_fixture()
    restored = ContactCoupling.from_stored_fields(value, source=source, target=target)
    assert restored is not value
    assert restored.coupling_id == value.coupling_id
    assert restored.tolerance == value.tolerance == 1.0e-4
    assert restored.source_scope.scope_id == value.source_scope.scope_id
    assert restored.target_scope.scope_id == value.target_scope.scope_id
    np.testing.assert_array_equal(restored.source_rows, value.source_rows)
    np.testing.assert_array_equal(restored.normals, value.normals)
    np.testing.assert_array_equal(restored.reference_gap, value.reference_gap)
    left, right = restored.penalty_forces(
        np.zeros((1, 2), dtype=np.float64),
        np.asarray(((0.0, -0.4),), dtype=np.float64),
        10.0,
    )
    expected = np.asarray(((0.0, 10.0 * 0.5000005 * 1.000001),), dtype=np.float64)
    # The gap and penalty chain has fewer than eight scalar operations; 64 eps
    # bounds accumulated relative rounding without cancellation in the force.
    np.testing.assert_allclose(
        right, expected, rtol=64 * np.finfo(np.float64).eps, atol=0.0
    )
    np.testing.assert_array_equal(left, -right)


def _corrupt_stored_contact(value: ContactCoupling, fault: str) -> ContactCoupling:
    match fault:
        case "row-range":
            return eqx.tree_at(
                lambda item: item.source_rows,
                value,
                jnp.full_like(value.source_rows, value.source_scope.entity_ids.size),
            )
        case "row-dtype":
            return eqx.tree_at(
                lambda item: item.source_rows,
                value,
                value.source_rows.astype(jnp.float64),
            )
        case "normal":
            return eqx.tree_at(lambda item: item.normals, value, 2 * value.normals)
        case "gap":
            return eqx.tree_at(
                lambda item: item.reference_gap, value, value.reference_gap + 0.01
            )
        case _:
            raise ValueError(
                f"fixture.fault must name a contact corruption; got {fault!r}"
            )


@pytest.mark.parametrize(
    ("fault", "field"),
    (
        ("row-range", "source_rows"),
        ("row-dtype", "source_rows"),
        ("normal", "normals"),
        ("gap", "stored_fields"),
    ),
    ids=("source-row-range", "source-row-dtype", "unit-normal", "physical-gap"),
)
def test_contact_stored_fields_refuse_corrupt_state(fault: str, field: str) -> None:
    value, source, target = _stored_contact_fixture()
    corrupted = _corrupt_stored_contact(value, fault)
    with pytest.raises(ValueError, match=rf"ContactCoupling\.{field}"):
        ContactCoupling.from_stored_fields(corrupted, source=source, target=target)
