"""Check integrable varying-frame extraction against an independent source oracle."""

from collections.abc import Iterator
from itertools import product

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.geometry._domain_coverage import certify_domain_coverage
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._source_fidelity import (
    certify_source_fidelity,
    MappedDomainBoundarySource,
)
from phydrax.meshing import CellMeshingResult, NativeMappedHexSource
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._hex_frame_map import (
    chart_face_permutation,
    extract_source_block_integer_grid,
    extract_source_frame_grid,
    prepare_hex_frame_holonomy,
    prepare_source_block_connection,
    prepare_source_hex_frame,
)
from phydrax.meshing._hex_frame_records import SourceBlockIntegerGrid
from phydrax.meshing._hex_generation import MappedGridHexConstruction
from phydrax.meshing.qualification import (
    check_stationary_euler,
    curved_occupied_source,
    solve_linear_diffusion,
)
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.hex_fields import allfield_inventory, allfield_state
from tests._support.hex_sources import (
    sheared_hex_specification,
    sheared_mapped_hex_source,
)


# Preserve the mapped Q2 example's fixed empirical solver acceptance limits.
_DIFFUSION_ERROR_LIMIT = 1.0e-8
_EULER_RESIDUAL_LIMIT = 1.0e-10


type ShearedHexPublication = tuple[
    NativeMappedHexSource,
    SourceBlockIntegerGrid,
    MappedGridHexConstruction,
    CellMeshingResult,
]


@pytest.mark.parametrize(
    "occupied,materials",
    (
        (
            {(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 2, 0), (1, 1, 1), (1, 1, 2)},
            False,
        ),
        (set(product(range(3), repeat=3)) - {(1, 1, 1)}, True),
    ),
)
def test_original_nonconstant_frame_has_exact_embedded_hex_coverage(
    occupied: set[tuple[int, int, int]], materials: bool
) -> None:
    source = curved_occupied_source(occupied, materials=materials)
    domain = source.domain
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            domain.reference_mesh,
            domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert np.all(prepared.varying_roots)
        assert np.any(prepared.corner_frames[:, 1:] != prepared.corner_frames[:, :1])
        assert prepared.face_connections.shape[0] > 0
        before = prepared.frame_id
        limits = MeshingLimits()
        counts = np.full((len(prepared.root_cells), 3), 2, dtype=np.int64)
        grid = extract_source_block_integer_grid(
            prepared,
            counts,
            domain.cell_regions,
            limits,
            10,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = extract_source_frame_grid(
            prepared,
            grid,
            domain.cell_regions,
            limits,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert prepared.frame_id == before
        assert {block.cell_kind for block in mapped.mesh.blocks} == {"hexahedron"}
        validity = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh)
        embedding = certify_global_embedding(mapped.mesh, mapped.geometry, validity)
        coverage = certify_domain_coverage(
            mapped.mesh, mapped.geometry, domain, mapped.cell_regions, embedding=embedding
        )
        fidelity = certify_source_fidelity(
            mapped.mesh,
            mapped.geometry,
            MappedDomainBoundarySource(domain, mapped.cell_regions),
            tolerance=0.0,
        )
        assert embedding.status == coverage.status == fidelity.status == "certified"
        assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0


def test_incompatible_chart_face_parity_refuses_without_changing_identity() -> None:

    authored = (37, 11, 29, 43)
    assert chart_face_permutation(authored, (29, 11, 37, 43)) == (2, 1, 0, 3)
    with pytest.raises(MeshingFailure, match="source_frame.boundary_parity"):
        chart_face_permutation(authored, (11, 29, 43, 37))
    assert authored == (37, 11, 29, 43)


def test_original_source_cycle_has_exact_singularity_witness_and_tree_cut() -> None:

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    mesh = source.domain.reference_mesh
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    edges = np.asarray(
        [
            (left, right)
            for left in range(len(cells))
            for right in range(left + 1, len(cells))
            if len(set(cells[left]) & set(cells[right])) == 4
        ],
        dtype=np.int64,
    )
    actions = np.broadcast_to(np.eye(3, dtype=np.int64), (len(edges), 3, 3)).copy()
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        regular = prepare_hex_frame_holonomy(mesh, edges, actions, scope=execution_scope)
        assert regular.cycle_edges.size == 1
        assert regular.singular_cycles.size == 0
        actions[regular.cycle_edges[0]] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        singular = prepare_hex_frame_holonomy(mesh, edges, actions, scope=execution_scope)
        assert singular.singular_cycles.tolist() == [0]
        assert not np.array_equal(singular.cycle_holonomy[0], np.eye(3, dtype=np.int64))
        assert singular.spanning_edges.size == len(cells) - 1
        assert regular.singular_cycles.size == 0


def test_block_extraction_capacity_refusal_preserves_prepared_and_requested_state() -> (
    None
):

    source = curved_occupied_source({(0, 0, 0)})
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        counts = np.full((1, 3), 2, dtype=np.int64)
        frames = prepared.corner_frames.copy()
        with pytest.raises(MeshingFailure, match="cells"):
            extract_source_block_integer_grid(
                prepared,
                counts,
                source.domain.cell_regions,
                MeshingLimits(maximum_cells=1),
                10,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert np.array_equal(counts, np.full((1, 3), 2))
        assert np.array_equal(prepared.corner_frames, frames)


@pytest.fixture(scope="module")
def sheared_hex_publication() -> Iterator[ShearedHexPublication]:
    source = sheared_mapped_hex_source()
    domain = source.domain
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        _,
    ):
        prepared = prepare_source_hex_frame(
            domain.reference_mesh,
            domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        requested = np.full((len(prepared.root_cells), 3), 2, dtype=np.int64)
        requested[0, 1] = 4
        grid = extract_source_block_integer_grid(
            prepared,
            requested,
            domain.cell_regions,
            MeshingLimits(),
            10,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = extract_source_frame_grid(
            prepared,
            grid,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        request = sheared_hex_specification(source, prepared)
        plan = phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("mapped_frame_grid_hex")
        ).plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        yield source, grid, mapped, plan.execute()


def test_sheared_authored_reference_charts_do_not_need_one_orthogonal_grid_frame(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    source, grid, mapped, _ = sheared_hex_publication
    assert np.all(grid.root_intervals[:, 1] == 4)
    validity = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh)
    embedding = certify_global_embedding(mapped.mesh, mapped.geometry, validity)
    coverage = certify_domain_coverage(
        mapped.mesh,
        mapped.geometry,
        source.domain,
        mapped.cell_regions,
        embedding=embedding,
    )
    assert embedding.status == coverage.status == "certified"


def test_sheared_hex_publication_retains_materials_and_interface(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    _, _, _, result = sheared_hex_publication
    assert result.audit.passed and result.compliance.passed
    assert result.certification is not None and result.certification.passed
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert {zone.material_id for zone in result.zones} == {
        "material-left",
        "material-right",
    }
    assert "original-material-interface" in {patch.name for patch in result.patches}


def test_sheared_hex_publication_has_zero_two_sided_fidelity(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    source, _, _, result = sheared_hex_publication
    domain = source.domain
    root_ids = np.asarray(domain.reference_mesh.topology.entities(3).entity_ids)
    ancestry = result.geometry.restriction_source
    assert ancestry is not None
    root_regions = dict(zip(root_ids.tolist(), domain.cell_regions.tolist(), strict=True))
    public_regions = np.asarray(
        [
            root_regions[int(parent)]
            for block in result.mesh.blocks
            for parent in np.asarray(ancestry.block_parent_cell_ids[block.name])
        ],
        dtype=np.int64,
    )
    fidelity = certify_source_fidelity(
        result.mesh,
        result.geometry,
        MappedDomainBoundarySource(domain, public_regions),
        tolerance=0.0,
    )
    assert fidelity.status == "certified"
    assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0


def test_sheared_hex_publication_supports_q2_diffusion(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    assert solve_linear_diffusion(sheared_hex_publication[3], 2) < _DIFFUSION_ERROR_LIMIT


def test_sheared_hex_publication_supports_stationary_euler(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    assert check_stationary_euler(sheared_hex_publication[3]) < _EULER_RESIDUAL_LIMIT


def test_sheared_hex_publication_retains_scalar_form_and_material_inventory(
    sheared_hex_publication: ShearedHexPublication,
) -> None:
    result = sheared_hex_publication[3]
    fields = allfield_state(result)
    inventory = allfield_inventory(result, fields)
    assert set(fields) == {"H1", "DG", "Hcurl", "Hdiv", "material_content"}
    for kind in ("H1", "DG", "Hcurl", "Hdiv"):
        form = inventory[kind]
        assert "components" in form, kind
        assert form["components"] == 2, kind
    assert fields["material_content"].shape[1] == 5
    material = inventory["material_content"]
    assert "region_inventories" in material
    regions = material["region_inventories"]
    assert isinstance(regions, list)
    assert len(regions) == 2


def test_source_authored_tensor_connection_retains_integer_offsets_and_regular_cycles() -> (
    None
):

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        connection = prepare_source_block_connection(prepared, scope=execution_scope)
        assert connection.frame_id == prepared.frame_id
        assert np.all(connection.transports == np.eye(3, dtype=np.int64))
        assert np.all(np.count_nonzero(connection.offsets, axis=1) == 1)
        assert connection.holonomy.singular_cycles.size == 0


def test_incompatible_singular_connection_cannot_replace_original_chart_transaction() -> (
    None
):

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        accepted = prepare_source_block_connection(prepared, scope=execution_scope)
        actions = accepted.transports.copy()
        actions[accepted.holonomy.cycle_edges[0]] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        holonomy = prepare_hex_frame_holonomy(
            prepared.reference_mesh,
            prepared.face_connections[:, (0, 2)],
            actions,
            scope=execution_scope,
        )
        proposed = eqx.tree_at(
            lambda value: (value.transports, value.holonomy),
            accepted,
            (actions, holonomy),
        )
        assert proposed.holonomy.singular_cycles.size == 1
        with pytest.raises(MeshingFailure, match="source_connection.offsets"):
            extract_source_block_integer_grid(
                prepared,
                np.full((len(prepared.root_cells), 3), 2),
                source.domain.cell_regions,
                MeshingLimits(),
                10,
                source_connection=proposed,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert accepted.holonomy.singular_cycles.size == 0
        assert np.all(accepted.transports == np.eye(3, dtype=np.int64))


def test_nonbinary_integer_charts_retain_exact_thirds_not_rounded_reference_fits() -> (
    None
):
    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    domain = source.domain
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget() as reference_budget,
    ):
        prepared = prepare_source_hex_frame(
            domain.reference_mesh,
            domain.source_geometry,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        grid = extract_source_block_integer_grid(
            prepared,
            np.full((1, 3), 3, dtype=np.int64),
            domain.cell_regions,
            MeshingLimits(),
            10,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert np.all(grid.root_intervals == 3)
        assert np.all(grid.chart_denominators == 3)
        assert sum(block.cell_count for block in grid.mesh.blocks) == 27
        mapped = extract_source_frame_grid(
            prepared,
            grid,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        original = np.asarray(
            domain.source_geometry.source_coordinates(budget=reference_budget),
            dtype=np.float64,
        )
        retained = np.asarray(
            mapped.geometry.source_coordinates(budget=reference_budget), dtype=np.float64
        )
        assert np.array_equal(original.view(np.uint64), retained.view(np.uint64))
        validity = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh)
        embedding = certify_global_embedding(mapped.mesh, mapped.geometry, validity)
        coverage = certify_domain_coverage(
            mapped.mesh, mapped.geometry, domain, mapped.cell_regions, embedding=embedding
        )
        fidelity = certify_source_fidelity(
            mapped.mesh,
            mapped.geometry,
            MappedDomainBoundarySource(domain, mapped.cell_regions),
            tolerance=0.0,
        )
        assert embedding.status == coverage.status == fidelity.status == "certified"
        assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0
