"""Independent original-source oracle for integrable varying-frame extraction."""

from itertools import product

import numpy as np
import pytest

from examples.native_quad_hex_meshing import curved_occupied_source
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.geometry._mesh_certificates import (
    certify_domain_coverage,
    certify_global_embedding,
    certify_source_fidelity,
    MappedDomainBoundarySource,
)
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._hex_frame_map import (
    extract_source_block_integer_grid,
    extract_source_frame_grid,
    prepare_source_hex_frame,
)


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
    prepared = prepare_source_hex_frame(domain.reference_mesh, domain.source_geometry)
    assert np.all(prepared.varying_roots)
    assert np.any(prepared.corner_frames[:, 1:] != prepared.corner_frames[:, :1])
    assert prepared.face_connections.shape[0] > 0
    before = prepared.frame_id
    limits = MeshingLimits()
    counts = np.full((len(prepared.root_cells), 3), 2, dtype=np.int64)
    grid = extract_source_block_integer_grid(
        prepared, counts, domain.cell_regions, limits, 10
    )
    mapped = extract_source_frame_grid(prepared, grid, domain.cell_regions, limits)
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
    from phydrax.meshing._hex_frame_map import _chart_face_permutation
    from phydrax.meshing._quad_generation import MeshingFailure

    authored = (37, 11, 29, 43)
    assert _chart_face_permutation(authored, (29, 11, 37, 43)) == (2, 1, 0, 3)
    with pytest.raises(MeshingFailure, match="boundary parity is incompatible"):
        _chart_face_permutation(authored, (11, 29, 43, 37))
    assert authored == (37, 11, 29, 43)


def test_original_source_cycle_has_exact_singularity_witness_and_tree_cut() -> None:
    from phydrax.meshing._hex_frame_map import prepare_hex_frame_holonomy

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
    regular = prepare_hex_frame_holonomy(mesh, edges, actions)
    assert regular.cycle_edges.size == 1
    assert regular.singular_cycles.size == 0
    actions[regular.cycle_edges[0]] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    singular = prepare_hex_frame_holonomy(mesh, edges, actions)
    assert singular.singular_cycles.tolist() == [0]
    assert not np.array_equal(singular.cycle_holonomy[0], np.eye(3, dtype=np.int64))
    assert singular.spanning_edges.size == len(cells) - 1
    assert regular.singular_cycles.size == 0


def test_block_extraction_capacity_refusal_preserves_prepared_and_requested_state() -> (
    None
):
    from phydrax.meshing._quad_generation import MeshingFailure

    source = curved_occupied_source({(0, 0, 0)})
    prepared = prepare_source_hex_frame(
        source.domain.reference_mesh, source.domain.source_geometry
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
        )
    assert np.array_equal(counts, np.full((1, 3), 2))
    assert np.array_equal(prepared.corner_frames, frames)


def test_sheared_authored_reference_charts_do_not_need_one_orthogonal_grid_frame() -> (
    None
):
    import phydrax as phx
    from examples.native_quad_hex_meshing import (
        check_stationary_euler,
        solve_linear_diffusion,
    )
    from phydrax.discretization import CellMesh
    from phydrax.discretization._hexahedral import HexahedralConnectivity
    from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
    from phydrax.meshing._volume_generation import (
        declared_plc_domain,
        PiecewiseLinearComplex,
    )

    # A separate authored reference atlas, not modification of qualification
    # inputs: its oblique faces cannot be one orthogonal global frame grid.
    original_source = curved_occupied_source({(1, 0, 0), (2, 0, 0)}, materials=True)
    original = original_source.domain
    shear = np.asarray(((1.0, 0.5, 0.0), (0.0, 1.0, 0.25), (0.0, 0.0, 1.0)))
    reference = CellMesh(
        np.asarray(original.reference_mesh.coordinates) @ shear.T,
        original.reference_mesh.blocks,
        vertex_global_ids=original.reference_mesh.vertex_global_ids,
    )
    original_plc = original_source.reference.complex
    loops = [
        original_plc.polygon_vertices[
            original_plc.polygon_offsets[row] : original_plc.polygon_offsets[row + 1]
        ]
        for row in range(len(original_plc.polygon_facets))
    ]
    plc = PiecewiseLinearComplex(
        np.asarray(original_plc.vertices) @ shear.T,
        loops,
        original_plc.polygon_facets,
        original_plc.facet_regions,
        original_plc.region_ids,
        segments=original_plc.segments,
    )
    reference_source = phx.meshing.NativePlcSource(
        plc, "authored-oblique-reference", "original-r1"
    )
    boundary = declared_plc_domain(plc, reference_source.source_id)
    domain = MappedReferenceDomain(
        boundary,
        reference,
        original.source_geometry,
        original.cell_regions,
        source_id="independently-authored-oblique-mapped-source",
        source_revision="original-r1",
    )
    prepared = prepare_source_hex_frame(reference, domain.source_geometry)
    requested = np.full((len(prepared.root_cells), 3), 2, dtype=np.int64)
    requested[0, 1] = 4
    grid = extract_source_block_integer_grid(
        prepared, requested, domain.cell_regions, MeshingLimits(), 10
    )
    assert np.all(grid.root_intervals[:, 1] == 4)
    mapped = extract_source_frame_grid(
        prepared, grid, domain.cell_regions, MeshingLimits()
    )
    validity = certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh)
    embedding = certify_global_embedding(mapped.mesh, mapped.geometry, validity)
    coverage = certify_domain_coverage(
        mapped.mesh, mapped.geometry, domain, mapped.cell_regions, embedding=embedding
    )
    assert embedding.status == coverage.status == "certified"
    source = phx.meshing.NativeMappedHexSource(reference_source, domain)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        reference.topology.entities(2).entity_ids,
    )
    root_ids = np.asarray(reference.topology.entities(3).entity_ids)
    regions = tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope(
                source.source_id,
                source.source_revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                3,
                domain.entity_set_id(3),
                root_ids[domain.cell_regions == row],
            ),
            name,
            f"material-{name}",
            phx.meshing.RegionRole.SOLID,
        )
        for row, name in enumerate(domain.region_ids)
    )
    connectivity = reference.connectivity
    if not isinstance(connectivity, HexahedralConnectivity):
        raise TypeError("Sheared source fixture requires hexahedral connectivity.")
    common = set(prepared.root_cells[0]) & set(prepared.root_cells[1])
    source_faces = np.asarray(connectivity.faces)
    interface_ids = np.asarray(reference.topology.entities(2).entity_ids)[
        np.asarray(
            [set(face) == common for face in source_faces],
            dtype=np.bool_,
        )
    ]
    interface = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        interface_ids,
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        scope,
        phx.meshing.VolumeFillStrategy.MULTIZONE,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5, maximum_size=1.0),),
        size_compliance=phx.meshing.SizeCompliancePolicy(relative_tolerance=0.95),
        region_controls=regions,
        patch_controls=(
            phx.meshing.PatchControl(
                "original-material-interface", interface, ("left", "right")
            ),
        ),
    )
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("mapped_frame_grid_hex")
    ).plan(
        source,
        request,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    result = plan.execute()
    assert result.audit.passed and result.compliance.passed
    assert result.certification is not None and result.certification.passed
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert {zone.material_id for zone in result.zones} == {
        "material-left",
        "material-right",
    }
    assert "original-material-interface" in {patch.name for patch in result.patches}
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
    assert solve_linear_diffusion(result, 2) < 1.0e-8
    assert check_stationary_euler(result) < 1.0e-10
    from tools.meshing_qualification import (
        _family_allfield_inventory,
        _family_allfield_state,
    )

    fields = _family_allfield_state(result)
    inventory = _family_allfield_inventory(result, fields)
    assert set(fields) == {"H1", "DG", "Hcurl", "Hdiv", "material_content"}
    assert all(
        inventory[kind]["components"] == 2 for kind in ("H1", "DG", "Hcurl", "Hdiv")
    )
    assert fields["material_content"].shape[1] == 5
    assert len(inventory["material_content"]["region_inventories"]) == 2


def test_source_authored_tensor_connection_retains_integer_offsets_and_regular_cycles() -> (
    None
):
    from phydrax.meshing._hex_frame_map import prepare_source_block_connection

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    prepared = prepare_source_hex_frame(
        source.domain.reference_mesh, source.domain.source_geometry
    )
    connection = prepare_source_block_connection(prepared)
    assert connection.frame_id == prepared.frame_id
    assert np.all(connection.transports == np.eye(3, dtype=np.int64))
    assert np.all(np.count_nonzero(connection.offsets, axis=1) == 1)
    assert connection.holonomy.singular_cycles.size == 0


def test_incompatible_singular_connection_cannot_replace_original_chart_transaction() -> (
    None
):
    import equinox as eqx

    from phydrax.meshing._hex_frame_map import (
        prepare_hex_frame_holonomy,
        prepare_source_block_connection,
    )
    from phydrax.meshing._quad_generation import MeshingFailure

    source = curved_occupied_source(set(product(range(2), range(2), (0,))))
    prepared = prepare_source_hex_frame(
        source.domain.reference_mesh, source.domain.source_geometry
    )
    accepted = prepare_source_block_connection(prepared)
    actions = accepted.transports.copy()
    actions[accepted.holonomy.cycle_edges[0]] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    holonomy = prepare_hex_frame_holonomy(
        prepared.reference_mesh, prepared.face_connections[:, (0, 2)], actions
    )
    proposed = eqx.tree_at(
        lambda value: (value.transports, value.holonomy), accepted, (actions, holonomy)
    )
    assert proposed.holonomy.singular_cycles.size == 1
    with pytest.raises(MeshingFailure, match="original complete face trace"):
        extract_source_block_integer_grid(
            prepared,
            np.full((len(prepared.root_cells), 3), 2),
            source.domain.cell_regions,
            MeshingLimits(),
            10,
            source_connection=proposed,
        )
    assert accepted.holonomy.singular_cycles.size == 0
    assert np.all(accepted.transports == np.eye(3, dtype=np.int64))


def test_nonbinary_integer_charts_retain_exact_thirds_not_rounded_reference_fits() -> (
    None
):
    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    domain = source.domain
    prepared = prepare_source_hex_frame(domain.reference_mesh, domain.source_geometry)
    grid = extract_source_block_integer_grid(
        prepared,
        np.full((1, 3), 3, dtype=np.int64),
        domain.cell_regions,
        MeshingLimits(),
        10,
    )
    assert np.all(grid.root_intervals == 3)
    assert np.all(grid.chart_denominators == 3)
    assert sum(block.cell_count for block in grid.mesh.blocks) == 27
    mapped = extract_source_frame_grid(
        prepared, grid, domain.cell_regions, MeshingLimits()
    )
    original = np.asarray(domain.source_geometry.source_coordinates(), dtype=np.float64)
    retained = np.asarray(mapped.geometry.source_coordinates(), dtype=np.float64)
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
