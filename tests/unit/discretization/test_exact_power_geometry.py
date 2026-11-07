from fractions import Fraction
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.discretization._exact_power_geometry import (
    _prepare_power_site_bank,
    _vertex_power_point,
    ExactPowerCellGeometrySource,
)


def test_nested_chart_structural_equality_requires_complete_exact_premises() -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        _structurally_equal_maps,
    )

    zero, one = Fraction(0), Fraction(1)
    basis: tuple[Polynomial, ...] = ({(0,): one, (1,): -one}, {(1,): one})
    coefficients = ((zero,), (one,))
    mapped_chart: tuple[Polynomial, ...] = ({(0,): Fraction(1, 3), (1,): Fraction(2)},)
    arguments: tuple[Polynomial, ...] = ({(1,): Fraction(1, 4)},)
    composed: tuple[Polynomial, ...] = ({(0,): Fraction(1, 3), (1,): Fraction(1, 2)},)
    source = (mapped_chart, basis, coefficients)
    assert _structurally_equal_maps(
        (composed, basis, coefficients), source, arguments, {}
    )
    assert not _structurally_equal_maps(
        (({(1,): Fraction(1, 2)},), basis, coefficients), source, arguments, {}
    )
    assert not _structurally_equal_maps(
        (composed, basis, ((zero,), (Fraction(2),))), source, arguments, {}
    )
    changed_basis: tuple[Polynomial, ...] = ({(0,): one}, {(1,): one})
    assert not _structurally_equal_maps(
        (composed, changed_basis, coefficients), source, arguments, {}
    )
    assert not _structurally_equal_maps(
        (composed, basis, cast(Any, ((0,), (one,)))), source, arguments, {}
    )
    assert not _structurally_equal_maps(
        (composed, cast(Any, ({(0,): 1, (1,): -one}, basis[1])), coefficients),
        source,
        arguments,
        {},
    )
    changed_arguments: tuple[Polynomial, ...] = ({(1,): Fraction(1, 2)},)
    assert not _structurally_equal_maps(
        (cast(Any, ({(0,): Fraction(1, 3), (1,): 1},)), basis, coefficients),
        source,
        changed_arguments,
        {},
    )


def test_exact_power_geometry_cold_reconstruction_retains_source_mapping() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [9.0 / 16.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    carrier = CellMesh.from_simplices(
        vertices,
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        dimension=3,
        block_name="carrier-tets",
        vertex_global_ids=np.asarray([31, 11, 41, 21], dtype=np.int64),
        cell_global_ids=np.asarray([105], dtype=np.int64),
        numeric_version="carrier-v7",
    )
    restricted_cells = SimpleNamespace(
        vertices=vertices,
        vertex_site_offsets=np.asarray([0, 1, 3, 4, 5], dtype=np.int64),
        vertex_sites=np.asarray([0, 0, 1, 0, 0], dtype=np.int32),
        vertex_carriers=np.asarray(
            [[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1], [3, -1, -1, -1]],
            dtype=np.int32,
        ),
        cell_sites=np.asarray([0], dtype=np.int32),
        cell_regions=np.asarray([7], dtype=np.int32),
    )
    source = ExactPowerCellGeometrySource(
        np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        np.asarray([1.0 / 16.0, 0.0], dtype=np.float64),
        carrier,
        restricted_cells,
        source_id="power-source",
        source_revision="r7",
        domain_source={"identity": "domain-owner"},
    )
    output_mesh = CellMesh.from_simplices(
        vertices,
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        dimension=3,
        block_name="power-cells",
    )
    original_geometry = CellGeometrySpec.power(output_mesh, source)
    changed_carrier = CellMesh(
        vertices,
        (
            CellBlock(
                "renamed-carrier",
                "tetrahedron",
                np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                global_ids=np.asarray([106], dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray([32, 12, 42, 22], dtype=np.int64),
        numeric_version="carrier-v8",
    )
    changed_source = ExactPowerCellGeometrySource(
        source.site_points,
        source.site_weights,
        changed_carrier,
        restricted_cells,
        source_id=source.source_id,
        source_revision=source.source_revision,
        domain_source=source.domain_source,
    )
    assert changed_source.source_map_id != source.source_map_id

    renewed_domain_source = {"identity": "renewed-domain-owner"}
    restored = ExactPowerCellGeometrySource.from_stored_fields(
        source, domain_source=renewed_domain_source
    )
    restored_geometry = CellGeometrySpec.power(output_mesh, restored)

    assert restored.carrier_mesh is not carrier
    assert restored.carrier_mesh.mesh_id == carrier.mesh_id
    assert restored.carrier_mesh.topology_id == carrier.topology_id
    assert tuple(block.block_id for block in restored.carrier_mesh.blocks) == tuple(
        block.block_id for block in carrier.blocks
    )
    restored_block = restored.carrier_mesh.blocks[0]
    source_block = carrier.blocks[0]
    assert restored_block.name == source_block.name
    assert restored_block.cell_kind == source_block.cell_kind
    assert restored.carrier_mesh.numeric_version == carrier.numeric_version
    np.testing.assert_array_equal(restored_block.global_ids, source_block.global_ids)
    np.testing.assert_array_equal(
        restored.carrier_mesh.vertex_global_ids, carrier.vertex_global_ids
    )
    assert restored.domain_source is renewed_domain_source
    assert restored.source_id == source.source_id
    assert restored.source_revision == source.source_revision
    assert restored.source_map_id == source.source_map_id
    assert restored_geometry.source_map_id == original_geometry.source_map_id
    expected_coordinates = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(9, 16), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    assert restored_geometry.source_coordinates() == expected_coordinates
    assert (
        restored_geometry.source_coordinates() == original_geometry.source_coordinates()
    )

    selected_rows = np.asarray((1, 2, 3), dtype=np.int64)
    selected_source = ExactPowerCellGeometrySource(
        source.site_points,
        source.site_weights,
        carrier,
        restricted_cells,
        source_id=source.source_id,
        source_revision=source.source_revision,
        domain_source=source.domain_source,
        output_vertex_indices=selected_rows,
    )
    selected_mesh = CellMesh.from_simplices(
        vertices[selected_rows],
        np.asarray(((0, 1, 2),), dtype=np.int64),
        dimension=2,
        block_name="power-surface",
    )
    selected_geometry = CellGeometrySpec.power(selected_mesh, selected_source)
    restored_selected = ExactPowerCellGeometrySource.from_stored_fields(selected_source)
    restored_selected_geometry = CellGeometrySpec.power(selected_mesh, restored_selected)
    expected_selected = tuple(expected_coordinates[index] for index in selected_rows)
    assert restored_selected.output_vertex_indices.tolist() == selected_rows.tolist()
    assert restored_selected_geometry.source_coordinates() == expected_selected
    assert restored_selected_geometry.source_coordinates() == (
        selected_geometry.source_coordinates()
    )


def test_exact_power_vertex_redundant_equal_sites_and_dominance() -> None:
    zero, one = Fraction(0), Fraction(1)
    sites = tuple(
        (Fraction(x), Fraction(y), Fraction(z))
        for x in (0, 1)
        for y in (0, 1)
        for z in (0, 1)
    )
    weights = (zero,) * len(sites)
    carrier = (
        (Fraction(-1), Fraction(-1), Fraction(-1)),
        (Fraction(7), Fraction(-1), Fraction(-1)),
        (Fraction(-1), Fraction(7), Fraction(-1)),
        (Fraction(-1), Fraction(-1), Fraction(7)),
    )
    center = (one / 2, one / 2, one / 2)
    rounded = np.asarray(center, dtype=np.float64)
    exact, barycentric = _vertex_power_point(
        rounded,
        sites,
        weights,
        tuple(range(len(sites))),
        carrier,
        prepared_site_bank=_prepare_power_site_bank(sites, weights),
    )
    assert exact == center
    assert barycentric == (
        Fraction(7, 16),
        Fraction(3, 16),
        Fraction(3, 16),
        Fraction(3, 16),
    )

    dominating_sites = (*sites, center)
    dominating_weights = (*weights, zero)
    with pytest.raises(ValueError):
        _vertex_power_point(
            rounded,
            dominating_sites,
            dominating_weights,
            tuple(range(len(sites))),
            carrier,
            prepared_site_bank=_prepare_power_site_bank(
                dominating_sites, dominating_weights
            ),
        )


def test_exact_power_vertex_rejects_nonzero_ulp_rounding_mismatch() -> None:
    sites = (
        (Fraction(1, 4), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(3, 4), Fraction(1, 2), Fraction(1, 2)),
    )
    weights = (Fraction(1, 16), Fraction(0))
    carrier_point = (Fraction(9, 16), Fraction(0), Fraction(0))
    prepared = _prepare_power_site_bank(sites, weights)
    rounded = np.asarray((np.nextafter(9.0 / 16.0, np.inf), 0.0, 0.0), dtype=np.float64)

    with pytest.raises(ValueError, match="does not match the rounded binary64 vertex"):
        _vertex_power_point(
            rounded,
            sites,
            weights,
            (0, 1),
            (carrier_point,),
            prepared_site_bank=prepared,
        )


def test_exact_power_vertex_preserves_signed_zero_native_bits() -> None:
    sites = ((Fraction(0), Fraction(0), Fraction(0)),)
    weights = (Fraction(0),)
    rounded = np.asarray((-0.0, 0.0, 0.0), dtype=np.float64)
    original_bits = rounded.view(np.uint64).copy()
    prepared = _prepare_power_site_bank(sites, weights)

    exact, _ = _vertex_power_point(
        rounded,
        sites,
        weights,
        (0,),
        ((Fraction(0), Fraction(0), Fraction(0)),),
        prepared_site_bank=prepared,
    )

    assert exact == (Fraction(0), Fraction(0), Fraction(0))
    np.testing.assert_array_equal(rounded.view(np.uint64), original_bits)
    assert array_tree_fingerprint(rounded) != array_tree_fingerprint(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    )
