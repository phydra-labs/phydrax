from itertools import permutations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization.fem import FormElementFamily
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
from phydrax.meshing._periodic import periodic_entity_orbit_map


def _periodic_tetrahedral_cover(
    periodic_axes: tuple[bool, bool, bool] = (True, True, True),
) -> CellMesh:
    shape = 3
    vertex_rows = np.arange(shape**3, dtype=np.int64).reshape((shape,) * 3)
    lattice_points = np.stack(
        np.meshgrid(*(np.arange(shape) for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    coordinates = lattice_points.astype(np.float64) / 2.0
    indices = np.arange(shape, dtype=np.int64)
    representative_axes = tuple(
        indices % 2 if periodic else indices for periodic in periodic_axes
    )
    representatives = vertex_rows[
        representative_axes[0][:, None, None],
        representative_axes[1][None, :, None],
        representative_axes[2][None, None, :],
    ].reshape(-1)
    shift_axes = tuple(
        indices // 2 if periodic else np.zeros_like(indices) for periodic in periodic_axes
    )
    shifts = np.stack(np.meshgrid(*shift_axes, indexing="ij"), axis=-1).reshape((-1, 3))
    tetrahedra = []
    for cube in np.ndindex((shape - 1,) * 3):
        for order in permutations(range(3)):
            point = np.asarray(cube, dtype=np.int64)
            path = [point.copy()]
            for axis in order:
                point = point.copy()
                point[axis] += 1
                path.append(point)
            tetrahedron = np.asarray(
                [vertex_rows[tuple(row)] for row in path], dtype=np.int32
            )
            if (
                np.linalg.det(coordinates[tetrahedron[1:]] - coordinates[tetrahedron[0]])
                < 0.0
            ):
                tetrahedron[[1, 2]] = tetrahedron[[2, 1]]
            tetrahedra.append(tetrahedron)
    block = CellBlock(
        "periodic-tetrahedra",
        "tetrahedron",
        np.asarray(tetrahedra, dtype=np.int32),
        global_ids=np.arange(3000, 3000 + len(tetrahedra), dtype=np.int64),
    )
    vertex_ids = np.arange(shape**3, dtype=np.int64)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(3, dtype=np.float64), periodic_axes=periodic_axes),
        representatives,
        shifts,
    )
    return CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )


@pytest.mark.parametrize(
    ("family", "source_order", "target_order"),
    (("trimmed", 1, 2), ("full", 0, 1)),
)
def test_periodic_high_order_de_rham_transfer_descends_and_commutes(
    family: FormElementFamily, source_order: int, target_order: int
) -> None:
    mesh = _periodic_tetrahedral_cover()
    source = FiniteElementDeRhamComplex(mesh, family=family, order=source_order)
    target = FiniteElementDeRhamComplex(mesh, family=family, order=target_order)
    transfer = source.transfer_periodic_quotient(target)
    quotient_map = transfer.complex_map

    assert transfer.lifted_transfer_result.entry_evidence
    assert transfer.quotient_transfer_result.complex_map.map_id == quotient_map.map_id
    np.testing.assert_allclose(
        transfer.commuting_error_bounds,
        transfer.quotient_transfer_result.commuting_error_bounds,
    )
    assert quotient_map.source.top_degree == 3
    assert quotient_map.target.top_degree == 3
    assert np.max(np.asarray(transfer.deck_descent_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.commuting_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.source_closure_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.target_closure_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.source_nilpotency_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.target_nilpotency_defects), initial=0.0) <= 1.0e-10

    edge_orbits = periodic_entity_orbit_map(mesh, 1)
    assert np.unique(np.asarray(edge_orbits.lifted_to_quotient)).size < len(
        edge_orbits.lifted_entity_ids
    )
    assert np.any(np.asarray(edge_orbits.winding_shifts) != 0)

    generator = np.random.default_rng(19)
    for degree in range(4):
        source_space = quotient_map.source.space(degree)
        target_space = quotient_map.target.space(degree)
        source_values = jnp.asarray(
            generator.normal(size=source_space.size), dtype=jnp.float64
        )
        target_values = jnp.asarray(
            generator.normal(size=target_space.size), dtype=jnp.float64
        )
        action = quotient_map.map(degree)
        np.testing.assert_allclose(
            target_space.inner(action.mv(source_values), target_values),
            source_space.inner(
                source_values, quotient_map.adjoint().map(degree).mv(target_values)
            ),
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        if degree < 3:
            left = quotient_map.target.differential(degree).mv(action.mv(source_values))
            right = quotient_map.map(degree + 1).mv(
                quotient_map.source.differential(degree).mv(source_values)
            )
            np.testing.assert_allclose(left, right, rtol=2.0e-10, atol=2.0e-10)


def test_periodic_relative_transfer_uses_quotient_physical_boundary() -> None:
    mesh = _periodic_tetrahedral_cover((True, False, False))
    source = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    target = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    transfer = source.transfer_periodic_quotient(target, boundary="relative")
    quotient_map = transfer.complex_map
    assert transfer.lifted_transfer_result.evidence_id
    assert transfer.quotient_transfer_result.entry_evidence

    assert transfer.boundary == "relative"
    assert len(transfer.target_active_indices[0]) < transfer.target_lifts[0].source.size
    generator = np.random.default_rng(23)
    for degree in range(4):
        source_space = quotient_map.source.space(degree)
        values = jnp.asarray(generator.normal(size=source_space.size), dtype=jnp.float64)
        mapped = quotient_map.map(degree).mv(values)
        lifted = transfer.lift_target(degree, mapped)
        np.testing.assert_allclose(
            transfer.collapse_target(degree, lifted), mapped, atol=2.0e-12
        )
        if degree < 3:
            np.testing.assert_allclose(
                quotient_map.target.differential(degree).mv(mapped),
                quotient_map.map(degree + 1).mv(
                    quotient_map.source.differential(degree).mv(values)
                ),
                atol=2.0e-10,
                rtol=2.0e-10,
            )
