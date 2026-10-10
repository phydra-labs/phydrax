#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import phydrax as phx
from examples.native_multiblock_meshing import (
    construct_multiblock,
    poisson_error,
    publish_multiblock,
    rectangle,
)
from phydrax.discretization import CellGeometrySpec
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.geometry.brep._patches import CircleCurve, LineCurve, PlanePatch
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._contracts import MeshingFailure
from phydrax.meshing._controls import (
    BlockInterfaceControl,
    LayerSchedule,
    TransfiniteCurveControl,
    TransfiniteSurfaceControl,
)
from phydrax.meshing._coupling import OversetCoupling
from phydrax.meshing._multiblock import (
    assemble_independent_blocks,
    glue_structured_blocks,
)
from phydrax.meshing._structured import generate_structured_block, TransfiniteBlock
from phydrax.meshing._sweep import generate_sweep, SweepControl, SweepMapKind


def test_native_provider_publishes_required_pure_quads_with_continuous_fidelity() -> None:
    result = publish_multiblock(4)
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    assert result.compliance.passed
    assert result.audit.passed
    assert result.certification is not None
    assert result.certification.passed
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert result.certification.fidelity.mesh_to_source_upper <= 0.08
    assert result.certification.fidelity.source_to_mesh_upper <= 0.08


def test_conforming_interface_has_exact_shared_global_ids() -> None:
    construction = construct_multiblock(3)
    routes = dict(construction.block_vertex_ids)
    np.testing.assert_array_equal(
        routes["left"].reshape(4, 4)[-1], routes["right"].reshape(4, 4)[0]
    )
    assert construction.mesh.coordinates.shape[0] == 28
    assert sum(block.cell_count for block in construction.mesh.blocks) == 18
    assert construction.maximum_gluing_residual == 0.0


def test_coordinate_coincidence_never_substitutes_for_declared_gluing() -> None:
    left = generate_structured_block(rectangle("left", 0.0, 0.5, 2))
    right = generate_structured_block(rectangle("right", 0.5, 1.0, 2))
    with pytest.raises(MeshingFailure, match="embedding"):
        glue_structured_blocks((left, right), ())


def test_wrong_face_permutation_rejected_even_with_matching_counts() -> None:
    left = generate_structured_block(rectangle("left", 0.0, 0.5, 2))
    right = generate_structured_block(rectangle("right", 0.5, 1.0, 2))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (True,))
    with pytest.raises(ValueError, match="incompatible coordinates"):
        glue_structured_blocks((left, right), (interface,))


def test_nonconforming_face_counts_require_explicit_coupling() -> None:
    left = generate_structured_block(rectangle("left", 0.0, 0.5, 2))
    right = generate_structured_block(rectangle("right", 0.5, 1.0, 3))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    with pytest.raises(ValueError, match="exact common"):
        glue_structured_blocks((left, right), (interface,))


def test_declared_nonconforming_parts_transfer_linear_interface_field() -> None:
    left_mesh = generate_structured_block(rectangle("left", 0.0, 0.5, 2)).mesh
    right_mesh = generate_structured_block(rectangle("right", 0.5, 1.0, 3)).mesh
    contract = phx.SpatialCoordinateContract.si()
    left = MeshPart("left", phx.meshing.certify_cell_mesh(left_mesh, contract))
    right = MeshPart("right", phx.meshing.certify_cell_mesh(right_mesh, contract))
    source = left.scope(0, np.asarray((6, 7, 8), dtype=np.int64))
    target = right.scope(0, np.arange(4, dtype=np.int64))
    transfer = OversetCoupling(
        left,
        right,
        source,
        target,
        np.asarray(((6, 7), (6, 7), (7, 8), (7, 8)), dtype=np.int64),
        np.asarray(
            ((1.0, 0.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0), (0.0, 1.0)),
            dtype=np.float64,
        ),
    )
    interface = BlockInterfaceControl(
        "left", 1, "right", 0, (0,), (False,), conforming=False
    )
    assembly = assemble_independent_blocks(
        (left, right),
        (interface,),
        {interface.control_id: transfer},
        logical_shapes={"left": (3, 3), "right": (4, 4)},
    )
    values = 1.0 + 2.0 * left.point_coordinates(source)[:, 1]
    expected = 1.0 + 2.0 * right.point_coordinates(target)[:, 1]
    np.testing.assert_allclose(
        assembly.couplings[0].transfer(values), expected, atol=1e-14
    )
    assert left_mesh.coordinates.shape[0] == 9
    assert right_mesh.coordinates.shape[0] == 16
    assert not transfer.conservative
    with pytest.raises(ValueError):
        assemble_independent_blocks(
            (left, right),
            (interface, interface),
            {interface.control_id: transfer},
            logical_shapes={"left": (3, 3), "right": (4, 4)},
        )


def test_multiblock_fem_manufactured_poisson_converges() -> None:
    coarse = poisson_error(4)
    fine = poisson_error(8)
    assert 0.0 < fine < coarse / 3.0


def _annulus_sector(index: int, /) -> TransfiniteBlock:
    first = index * np.pi / 2.0
    last = (index + 1) * np.pi / 2.0
    name = f"sector:{index}"
    first_ray = np.asarray((np.cos(first), np.sin(first)), dtype=np.float64)
    last_ray = np.asarray((np.cos(last), np.sin(last)), dtype=np.float64)
    return TransfiniteBlock(
        name,
        (
            TransfiniteCurveControl(
                CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0),
                "inner",
                (first, last),
                6,
            ),
            TransfiniteCurveControl(
                CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0),
                "outer",
                (first, last),
                6,
            ),
            TransfiniteCurveControl(
                LineCurve(first_ray, first_ray), f"ray:{index}", (0.0, 1.0), 2
            ),
            TransfiniteCurveControl(
                LineCurve(last_ray, last_ray), f"ray:{(index + 1) % 4}", (0.0, 1.0), 2
            ),
        ),
    )


def test_cylinder_ogrid_periodic_block_cycle_closes_without_axis_collapse() -> None:
    sectors = tuple(
        generate_structured_block(_annulus_sector(index)) for index in range(4)
    )
    interfaces = tuple(
        BlockInterfaceControl(
            f"sector:{index}", 3, f"sector:{(index + 1) % 4}", 2, (0,), (False,)
        )
        for index in range(4)
    )
    annulus = glue_structured_blocks(sectors, interfaces)
    routes = dict(annulus.block_vertex_ids)
    np.testing.assert_array_equal(
        routes["sector:3"].reshape(3, 7)[:, -1], routes["sector:0"].reshape(3, 7)[:, 0]
    )
    cylinder = generate_sweep(
        annulus.mesh,
        SweepControl(
            SweepMapKind.EXTRUSION,
            LayerSchedule(np.asarray((0.25, 0.25, 0.5), dtype=np.float64)),
            "annular-cylinder",
        ),
    )
    assert {block.cell_kind for block in cylinder.mesh.blocks} == {"hexahedron"}
    assert sum(block.cell_count for block in cylinder.mesh.blocks) == 144
    assert (
        np.min(np.linalg.norm(np.asarray(cylinder.mesh.coordinates)[:, :2], axis=-1))
        > 0.99
    )
    assert cylinder.embedding.status == "certified"


def test_positive_reversed_hex_face_map_glues_exact_topology() -> None:
    constructions = []
    for name, origin, diagonal in (
        ("left", (0.0, 0.0, 0.0), (0.5, 1.0, 1.0)),
        ("right", (0.5, 1.0, 1.0), (0.5, -1.0, -1.0)),
    ):
        matrix = np.diag(np.asarray(diagonal, dtype=np.float64))
        controls = []
        for axis in range(3):
            other = tuple(index for index in range(3) if index != axis)
            for side in range(2):
                patch = PlanePatch(
                    np.asarray(origin) + side * matrix[:, axis],
                    matrix[:, other[0]],
                    matrix[:, other[1]],
                )
                controls.append(
                    TransfiniteSurfaceControl(
                        patch,
                        f"{name}:{2 * axis + side}",
                        np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
                        (1, 1),
                    )
                )
        constructions.append(
            generate_structured_block(TransfiniteBlock(name, tuple(controls)))
        )
    interface = BlockInterfaceControl("left", 1, "right", 0, (0, 1), (True, True))
    glued = glue_structured_blocks(tuple(constructions), (interface,))
    routes = dict(glued.block_vertex_ids)
    np.testing.assert_array_equal(
        routes["left"].reshape(2, 2, 2)[-1],
        routes["right"].reshape(2, 2, 2)[0, ::-1, ::-1],
    )
    assert glued.mesh.coordinates.shape[0] == 12
    assert {block.cell_kind for block in glued.mesh.blocks} == {"hexahedron"}


def test_gluing_retains_curved_cell_interiors_and_rejects_a_cracked_trace() -> None:
    element = coordinate_lagrange_element("quadrilateral", 2)
    nodes = np.asarray(element.reference_nodes)
    u, v = nodes[:, 0], nodes[:, 1]
    constructions = []
    for name, left in (("left", 0.0), ("right", 0.5)):
        construction = generate_structured_block(rectangle(name, left, left + 0.5, 1))
        controls = np.column_stack(
            (left + 0.5 * u, v + 0.125 * u * (1.0 - u) * v * (1.0 - v))
        )
        geometry = CellGeometrySpec(
            {name: element}, {name: np.arange(9, dtype=np.int32)[None, :]}, controls
        )
        constructions.append(replace(construction, geometry=geometry))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    glued = glue_structured_blocks(tuple(constructions), (interface,))
    for coordinate_element, route in zip(
        glued.geometry.elements, glued.geometry.geometry_dofs, strict=True
    ):
        if not isinstance(coordinate_element, phx.discretization.FiniteElementSpec):
            raise TypeError("Curved gluing must retain the actual tabulated source maps.")
        value = (
            np.asarray(
                coordinate_element.tabulate(np.asarray(((0.5, 0.5),), dtype=np.float64))[
                    0
                ]
            )
            @ np.asarray(glued.geometry.coordinates)[np.asarray(route)[0]]
        )
        np.testing.assert_allclose(value[:, 1], 0.5 + 0.125 / 16.0, atol=1e-14, rtol=0.0)
    changed = np.asarray(constructions[1].geometry.coordinates).copy()
    changed[1, 0] += 0.01  # Same corners, different interior point of the shared trace.
    cracked_geometry = CellGeometrySpec(
        {"right": element}, {"right": np.arange(9, dtype=np.int32)[None, :]}, changed
    )
    cracked = replace(constructions[1], geometry=cracked_geometry)
    with pytest.raises(MeshingFailure, match="mapped_trace_mismatch"):
        glue_structured_blocks((constructions[0], cracked), (interface,))


def test_interface_tolerances_do_not_accumulate_at_a_shared_corner() -> None:
    delta = 0.00075
    constructions = []
    for name, x, y in (
        ("a", 0.0, 0.0),
        ("b", 1.0 + delta, 0.0),
        ("c", 1.0 + 2.0 * delta, 1.0),
    ):
        boundaries = tuple(
            TransfiniteCurveControl(
                LineCurve(origin, direction), f"{name}:{edge}", (0.0, 1.0), 1
            )
            for edge, origin, direction in (
                ("left", (x, y), (0.0, 1.0)),
                ("right", (x + 1.0, y), (0.0, 1.0)),
                ("bottom", (x, y), (1.0, 0.0)),
                ("top", (x, y + 1.0), (1.0, 0.0)),
            )
        )
        constructions.append(
            generate_structured_block(TransfiniteBlock(name, boundaries))
        )
    interfaces = (
        BlockInterfaceControl("a", 1, "b", 0, (0,), (False,), tolerance=0.001),
        BlockInterfaceControl("b", 3, "c", 2, (0,), (False,), tolerance=0.001),
    )
    with pytest.raises(ValueError, match="final node displacement tolerance"):
        glue_structured_blocks(tuple(constructions), interfaces)


def test_gluing_coordinate_bank_borrows_original_live_storage() -> None:
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    blocks = (
        generate_structured_block(rectangle("left", 0.0, 0.5, 16)),
        generate_structured_block(rectangle("right", 0.5, 1.0, 16)),
    )
    original = tuple(np.asarray(block.mesh.coordinates).copy() for block in blocks)
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    limits = phx.meshing.MeshingLimits(maximum_scratch_bytes=1024**2)
    with pytest.raises(MeshingFailure) as refused:
        with native_volume_execution_budget(limits) as execution:
            retained = execution.allocate_host_array((1024**2 - 4096,), np.uint8)
            retained.fill(59)
            glue_structured_blocks(blocks, (interface,))
    assert refused.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert retained[0] == retained[-1] == 59
    for block, coordinates in zip(blocks, original, strict=True):
        np.testing.assert_array_equal(block.mesh.coordinates, coordinates)


def test_public_multiblock_refuses_aggregate_cell_capacity_without_changing_sources() -> (
    None
):
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.geometry._mesh_certificates import ParametricCurveBoundarySource

    blocks = (rectangle("left", 0.0, 0.5, 2), rectangle("right", 0.5, 1.0, 2))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    curves = (
        LineCurve((0.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((0.0, 1.0), (1.0, 0.0)),
    )
    revision = canonical_fingerprint(
        {"kind": "unit-square-boundary", "curves": array_tree_fingerprint(curves)}
    )
    boundary = ParametricCurveBoundarySource(
        curves,
        ((0.0, 1.0),) * 4,
        source_id="unit-square",
        source_revision=revision,
        covering_radius=0.01,
    )
    source = phx.meshing.NativeStructuredSource(
        blocks,
        (interface,),
        "unit-square",
        revision,
        fidelity_source=boundary,
        maximum_deviation=0.08,
    )
    original = source.binding_id
    scope = phx.meshing.MeshingScope(
        "unit-square",
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "unit-square-domain",
        np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("quadrilateral",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.5, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        limits=phx.meshing.MeshingLimits(maximum_cells=7),
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("structured_transfinite")
    )
    with pytest.raises(MeshingFailure) as refused:
        provider.plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        ).execute()
    assert refused.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert source.binding_id == original


def test_original_material_multiblock_publishes_exclusive_regions_and_exact_interface() -> (
    None
):
    from argparse import Namespace

    from phydrax._fingerprint import array_tree_fingerprint
    from phydrax.meshing._quad_generation import _entities
    from tools.meshing_qualification import _family_qualification_fixture

    source, specification, options, _ = _family_qualification_fixture(
        Namespace(
            family_case="multiblock-material-hex",
            resolution=4,
            capacity=20000,
            timeout=120.0,
        )
    )
    original = source.binding_id, array_tree_fingerprint(source)
    result = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    assert result.certification is not None and result.certification.passed
    zones = {zone.name: zone for zone in result.zones}
    assert set(zones) == {"left", "right"}
    assert zones["left"].material_id == "material-left"
    assert zones["right"].material_id == "material-right"
    left = set(map(int, np.asarray(zones["left"].scope.entity_ids)))
    right = set(map(int, np.asarray(zones["right"].scope.entity_ids)))
    assert not left & right
    assert left | right == set(map(int, np.asarray(result.mesh.entity_set(3).entity_ids)))
    interface = next(
        patch for patch in result.patches if patch.name == "material-interface"
    )
    assert set(interface.adjacent_zone_ids) == {zone.zone_id for zone in zones.values()}
    assert interface.source_adjacent_region_ids == ("left", "right")
    face_ids = np.asarray(result.mesh.entity_set(2).entity_ids)
    rows = np.flatnonzero(np.isin(face_ids, np.asarray(interface.scope.entity_ids)))
    assert rows.size
    assert np.all(
        np.asarray(result.mesh.coordinates)[_entities(result.mesh, 2)[rows], 0] == 1.0
    )
    assert (source.binding_id, array_tree_fingerprint(source)) == original
