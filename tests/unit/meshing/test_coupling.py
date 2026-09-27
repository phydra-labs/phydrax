from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._coupling import (
    ConformalCoupling,
    ContactCoupling,
    OversetCoupling,
    PeriodicCoupling,
)


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


def test_conformal_bijection_maps_global_ids_and_rejects_stale_endpoint() -> None:
    source = _part("left")
    target = _part("right", ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)), ((1, 0, 2),))
    source_scope, target_scope = source.scope(0, [0, 1]), target.scope(0, [0, 1])
    coupling = ConformalCoupling(
        source, target, source_scope, target_scope, source_ids=np.asarray([1, 0])
    )
    np.testing.assert_array_equal(coupling.transfer(jnp.asarray([3.0, 8.0])), [8.0, 3.0])
    np.testing.assert_array_equal(coupling.transpose(jnp.asarray([2.0, 7.0])), [7.0, 2.0])
    moved = _part("right", ((2.0, 0.0), (1.0, 0.0), (1.0, 1.0)), ((1, 0, 2),))
    with pytest.raises(ValueError, match="stale"):
        MeshAssembly((source, moved), couplings=(coupling,))
    with pytest.raises(ValueError, match="coincide"):
        ConformalCoupling(source, target, source_scope, target_scope)
    with pytest.raises(ValueError, match="bijection"):
        ConformalCoupling(
            source, target, source_scope, target_scope, source_ids=np.asarray([0, 0])
        )


def test_periodic_isometry_transports_vectors_not_just_scalar_values() -> None:
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


def test_node_contact_activation_and_equal_opposite_differentiable_forces() -> None:
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


def test_overset_interpolates_constants_and_linear_values_with_exact_transpose() -> None:
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
    with pytest.raises(ValueError, match="exactly one"):
        MeshAssembly((source, target, duplicate_donor), couplings=(overlay, duplicate))


def test_overset_rejects_nonpartition_weights_unknown_donors_and_conflicting_holes() -> (
    None
):
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


def test_overset_search_interpolates_linear_fields_through_located_donor_cells() -> None:
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


def test_overset_search_reports_outside_and_excluded_donors() -> None:
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


def test_contact_search_recovers_the_nearest_node_bijection() -> None:
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
