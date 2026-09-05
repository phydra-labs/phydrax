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


def _part(name, coordinates=None, cells=None):
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


def test_conformal_bijection_maps_global_ids_and_rejects_stale_endpoint():
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


def test_periodic_isometry_transports_vectors_not_just_scalar_values():
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


def test_node_contact_activation_and_equal_opposite_differentiable_forces():
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


def test_overset_interpolates_constants_and_linear_values_with_exact_transpose():
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


def test_overset_rejects_nonpartition_weights_unknown_donors_and_conflicting_holes():
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
