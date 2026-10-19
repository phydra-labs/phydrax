"""Exercise the quadratic overset fixture's source and coefficient-bank invariants."""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing import CellMeshingResult
from tests._support.meshing_fixtures import certified_carrier
from tests._support.overset_mapped_sources import mapped_source


def _assert_shear_source(carrier: CellMeshingResult, kinds: Sequence[str]) -> None:
    elements, routes, _ = carrier.geometry.resolve(carrier.mesh)
    corner_banks = []
    coefficient_banks = []
    corner_offset = coefficient_offset = 0
    assert tuple(block.cell_kind for block in carrier.mesh.blocks) == tuple(kinds)
    for index, (kind, block, element, route) in enumerate(
        zip(kinds, carrier.mesh.blocks, elements, routes, strict=True)
    ):
        assert block.name == f"block{index}"
        np.testing.assert_array_equal(block.global_ids, (90 - 17 * index,))
        corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
        corners[:, 0] += 0.1 * corners[:, 1] * corners[:, 2] + 2 * index
        corner_banks.append(corners)
        np.testing.assert_array_equal(
            block.vertices, np.arange(len(corners))[None] + corner_offset
        )
        reference = coordinate_lagrange_element(kind, 2)
        assert element.element_id == reference.element_id
        coefficients = np.asarray(reference.reference_nodes).copy()
        coefficients[:, 0] += 0.1 * coefficients[:, 1] * coefficients[:, 2] + 2 * index
        coefficient_banks.append(coefficients)
        np.testing.assert_array_equal(
            route, np.arange(len(coefficients))[None] + coefficient_offset
        )
        corner_offset += len(corners)
        coefficient_offset += len(coefficients)
    np.testing.assert_array_equal(carrier.mesh.coordinates, np.concatenate(corner_banks))
    np.testing.assert_array_equal(
        carrier.geometry.coordinates, np.concatenate(coefficient_banks)
    )


@pytest.mark.parametrize(
    "kinds",
    (("tetrahedron",), ("hexahedron",), ("prism",), ("hexahedron", "prism")),
)
def test_mapped_source_preserves_shear_banks_and_block_offsets(
    kinds: tuple[str, ...],
) -> None:
    _assert_shear_source(certified_carrier(mapped_source(kinds)), kinds)


@pytest.mark.parametrize("fault", ("source-corner", "coefficient-bank", "block-offset"))
def test_mapped_source_oracle_rejects_corrupted_source_or_coefficients(
    fault: str,
) -> None:
    kinds = ("hexahedron", "prism")
    carrier = certified_carrier(mapped_source(kinds))
    match fault:
        case "source-corner":
            altered = eqx.tree_at(
                lambda value: value.mesh.coordinates,
                carrier,
                jnp.asarray(carrier.mesh.coordinates).at[0, 0].add(0.125),
            )
        case "coefficient-bank":
            altered = eqx.tree_at(
                lambda value: value.geometry.coordinates,
                carrier,
                jnp.asarray(carrier.geometry.coordinates).at[0, 0].add(0.125),
            )
        case "block-offset":
            route = carrier.geometry.geometry_dofs[1]
            altered = eqx.tree_at(
                lambda value: value.geometry.geometry_dofs[1],
                carrier,
                jnp.asarray(route) - 1,
            )
        case _:
            raise AssertionError(
                f"fixture.fault must name a mapped-source fault; got {fault!r}"
            )
    with pytest.raises(AssertionError):
        _assert_shear_source(altered, kinds)
