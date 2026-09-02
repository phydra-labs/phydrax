import jax.numpy as jnp
import numpy as np

from phydrax.discretization import PeriodicCell
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization.vem import prepare_polyhedral_h1_virtual_element_3d


def test_rank_two_periodic_cell_preserves_orthogonal_component():
    cell = PeriodicCell([[2.0, 0.0, 0.0], [0.5, 1.5, 0.0]])
    point = jnp.asarray([2.25, -0.1, 3.0])
    wrapped, images = cell.wrap(point)
    assert cell.rank == 2
    assert cell.ambient_dimension == 3
    assert wrapped[2] == point[2]
    assert images.shape == (2,)
    assert jnp.allclose(
        cell.vectors @ cell.reciprocal_vectors.T, 2.0 * jnp.pi * jnp.eye(2)
    )


def test_root_polyhedral_connectivity_drives_degree_one_vem():
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = (
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    )
    mesh = CellMesh.from_polyhedra(points, (faces,))
    assert isinstance(mesh.connectivity, PolyhedralConnectivity)
    assert bool(jnp.all(mesh.connectivity.boundary_faces))
    prepared = prepare_polyhedral_h1_virtual_element_3d(mesh)
    constant = jnp.ones((4,))
    assert jnp.linalg.norm(prepared.mv(constant)) < 1e-10
    assert prepared.evidence.minimum_volume > 0.0
