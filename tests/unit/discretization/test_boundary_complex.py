from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._boundary_complex import boundary_subcomplex
from phydrax.discretization._cell_complex import (
    tetrahedral_cell_complex,
    tetrahedral_connectivity,
)
from phydrax.discretization._cochain import CochainDiscretization
from phydrax.discretization._cochain_hodge import DiagonalHodge, SparseHodge
from phydrax.exterior._traces import trace_evidence, trace_map
from phydrax.linalg import ArraySpace


_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TETRAHEDRA = np.array([[0, 1, 2, 3]], dtype=np.int32)


def _complex() -> CochainDiscretization:
    topology = tetrahedral_cell_complex(_TETRAHEDRA, 4)
    counts = tuple(entities.count for entities in topology.entity_sets)
    return CochainDiscretization(
        topology,
        tuple(DiagonalHodge(jnp.ones((count,), dtype=jnp.float64)) for count in counts),
        boundary_masks=tuple(
            np.ones((count,), dtype=np.bool_)
            if degree < 3
            else np.zeros((count,), dtype=np.bool_)
            for degree, count in enumerate(counts)
        ),
    )


def test_induced_tetrahedron_boundary_normals_point_outward() -> None:
    complex = _complex()
    boundary = boundary_subcomplex(
        complex.topology, boundary_mask=complex.boundary_masks[2]
    )
    connectivity = tetrahedral_connectivity(_TETRAHEDRA, 4)
    triangles = np.asarray(connectivity.faces)[np.asarray(boundary.parent_indices[2])]
    corners = _VERTICES[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals *= np.asarray(boundary.orientation_signs[2])[:, None]
    assert np.all(
        np.sum(normals * (np.mean(corners, axis=1) - np.mean(_VERTICES, axis=0)), axis=1)
        > 0.0
    )
    boundary_of_boundary = (
        boundary.topology.incidences[0].scipy_boundary()
        @ boundary.topology.incidences[1].scipy_boundary()
    )
    np.testing.assert_array_equal(boundary_of_boundary.toarray(), np.zeros((4, 4)))


def test_trace_commutes_and_obeys_integrated_stokes() -> None:
    complex = _complex()
    mapping = trace_map(complex)
    values = (jnp.array([0.2, -0.4, 0.7, 1.1]), jnp.linspace(-1.0, 2.0, 6))
    evidence = trace_evidence(mapping, values)
    assert bool(evidence.successful)
    np.testing.assert_allclose(evidence.commuting_residuals, 0.0, atol=1.0e-13)
    face_content = jnp.array([0.4, -0.8, 1.2, 2.0])
    oriented_boundary_content = mapping.maps[2].mv(face_content)
    volume_derivative = complex.exterior_derivative(2, face_content)
    np.testing.assert_allclose(
        jnp.sum(oriented_boundary_content), jnp.sum(volume_derivative), atol=1.0e-13
    )


def test_boundary_pairing_inverse_is_principal_restriction() -> None:
    complex = _complex()
    gram = np.array(
        [
            [4.0, 1.0, 0.5, 0.2],
            [1.0, 3.0, 0.7, 0.4],
            [0.5, 0.7, 2.0, 0.3],
            [0.2, 0.4, 0.3, 2.0],
        ]
    )
    rows, columns = np.triu_indices(4)
    complex = CochainDiscretization(
        complex.topology,
        (
            SparseHodge(rows, columns, jnp.asarray(gram[rows, columns]), 4),
            *complex.hodges[1:],
        ),
        boundary_masks=complex.boundary_masks,
    )
    facet_mask = np.array([True, False, False, False])
    boundary = boundary_subcomplex(complex.topology, boundary_mask=facet_mask)
    mapping = trace_map(complex, boundary_mask=facet_mask)
    selected = np.asarray(boundary.parent_indices[0])
    space = mapping.target.space(0)
    if not isinstance(space, ArraySpace):
        raise TypeError("Cell trace must return an array space.")
    rhs = jnp.arange(1.0, selected.size + 1.0)
    actual = space.pairing.inverse_riesz(rhs)
    expected = np.linalg.solve(gram[np.ix_(selected, selected)], np.asarray(rhs))
    np.testing.assert_allclose(actual, expected, atol=1.0e-11, rtol=1.0e-11)
    wrong = np.linalg.solve(gram, np.eye(4)[:, selected] @ np.asarray(rhs))[selected]
    assert np.linalg.norm(expected - wrong) > 1.0e-3


def test_boundary_selection_refuses_shared_interior_facet() -> None:
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int32)
    topology = tetrahedral_cell_complex(cells, 5)
    connectivity = tetrahedral_connectivity(cells, 5)
    mask = np.asarray(connectivity.face_cell_counts) == 2
    with pytest.raises(ValueError, match="exactly one"):
        boundary_subcomplex(topology, boundary_mask=mask)
