# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from fractions import Fraction
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from tools.meshing_qualification._controls import (
    _family_qualification_requirements,
    capacity_meshing_limits,
    FAMILY_CASES,
    QualificationControls,
)
from tools.meshing_qualification._runtime import QualificationFailure
from tools.meshing_qualification.scenarios.family import (
    _family_material_acceptance,
    _family_nonsweepable_plc_source,
    _family_nonsweepable_source_proof,
    _family_qualification_fixture,
)


def _executed_fixture(args: QualificationControls) -> tuple[Any, Any, Any, dict[str, Any]]:
    """The qualification fixture, executed at the registered lifecycle work cap.

    Work follows the registered native-family-lifecycle envelope: the priced
    CellMesh/CellBlock constructions of the dual-hex mesh alone need about
    6.3M work units at resolution two, beyond 100 per capacity unit.
    """
    source, request, options, case = _family_qualification_fixture(args)
    old = request.limits
    limits = phx.meshing.MeshingLimits(
        maximum_vertices=old.maximum_vertices,
        maximum_edges=old.maximum_edges,
        maximum_faces=old.maximum_faces,
        maximum_cells=old.maximum_cells,
        maximum_connectivity_entries=old.maximum_connectivity_entries,
        maximum_data_bytes=old.maximum_data_bytes,
        maximum_work_units=128_000_000,
        maximum_cavity_cells=old.maximum_cavity_cells,
        maximum_geometry_queries=old.maximum_geometry_queries,
        maximum_scratch_bytes=old.maximum_scratch_bytes,
        maximum_wall_seconds=old.maximum_wall_seconds,
    )
    return source, eqx.tree_at(lambda value: value.limits, request, limits), options, case


def test_nonsweepable_profile_is_additive_and_uses_generic_native_plc() -> None:
    original = (
        "dual-quad",
        "curved-quarter-extrusion-quad",
        "dual-hex",
        "hex-dominant",
        "transfinite-quad",
        "multiblock-quad",
        "sweep-hex",
        "sweep-prism",
        "curved-frame-sweep-hex",
        "multiblock-reversed-hex",
        "multiblock-material-hex",
        "multiblock-cycle-quad",
        "curved-nonsweepable-hex",
        "curved-cavity-hex",
        "curved-material-hex",
    )
    requirements = _family_qualification_requirements()
    assert FAMILY_CASES == (*original, "nonsweepable-dual-hex")
    assert requirements["parser_choices"] == FAMILY_CASES
    profiles = requirements["profiles"]
    assert all(profiles[name]["nonsweepability_required"] is False for name in original)
    profile = profiles["nonsweepable-dual-hex"]
    assert profile["nonsweepability_required"] is True
    assert profile["route"] == "plc_dual_hex"
    assert profile["family"] == "hexahedron"
    assert profile["dimension"] == 3
    assert (
        profile["source"]
        == "NativePlcSource:unit-tetrahedron-two-enclosed-tetrahedral-cavities"
    )


def test_nonsweepable_source_has_exact_independently_oriented_three_sphere_boundary() -> (
    None
):
    source = _family_nonsweepable_plc_source()
    assert type(source) is phx.meshing.NativePlcSource
    assert source.source_id == "qualification-nonsweepable-dual-hex"
    assert source.source_revision == "dyadic-two-cavities-r1"
    complex_ = source.complex
    lattice = np.asarray(complex_.vertices) * 16
    np.testing.assert_array_equal(lattice, lattice.astype(np.int64))
    points = [[Fraction(int(value), 16) for value in row] for row in lattice]
    np.testing.assert_array_equal(
        lattice[:4], ((0, 0, 0), (16, 0, 0), (0, 16, 0), (0, 0, 16))
    )
    # For the independently checked unit tetrahedron, barycentric coordinates
    # are x, y, z, 1-x-y-z; strict positivity extends to each convex cavity.
    for point in points[4:]:
        assert min(point) > 0
        assert sum(point) < 1
    assert max(point[0] for point in points[4:8]) < min(point[0] for point in points[8:])
    faces = np.asarray(complex_.polygon_vertices).reshape(12, 3)
    np.testing.assert_array_equal(complex_.polygon_facets, np.arange(12))
    np.testing.assert_array_equal(complex_.facet_regions, np.tile((-1, 0), (12, 1)))
    assert complex_.region_ids == ("solid",)
    volumes = []
    component_vertices = []
    for shell in range(3):
        local = faces[4 * shell : 4 * shell + 4]
        vertices = set(map(int, local.flat))
        component_vertices.append(vertices)
        assert len(vertices) == 4
        edges = {}
        volume = Fraction(0)
        for face in local:
            a, b, c = (points[int(index)] for index in face)
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            normal = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
            opposite = points[next(iter(vertices - set(map(int, face))))]
            side = sum(normal[i] * (opposite[i] - a[i]) for i in range(3))
            assert side < 0 if shell == 0 else side > 0
            volume += sum(a[i] * normal[i] for i in range(3)) / 6
            for i in range(3):
                u_id, v_id = int(face[i]), int(face[(i + 1) % 3])
                edges.setdefault(tuple(sorted((u_id, v_id))), []).append((u_id, v_id))
        assert len(edges) == 6
        assert len(vertices) - len(edges) + len(local) == 2
        assert all(
            len(sides) == 2 and sides[0] == sides[1][::-1] for sides in edges.values()
        )
        # Every pair of tetrahedron vertices shares an edge: each shell is connected.
        assert all(
            tuple(sorted((u, v))) in edges for u in vertices for v in vertices if u != v
        )
        volumes.append(volume)
    assert all(
        component_vertices[i].isdisjoint(component_vertices[j])
        for i in range(3)
        for j in range(i)
    )
    assert volumes == [Fraction(1, 6), -Fraction(1, 3072), -Fraction(1, 3072)]
    assert sum(volumes) == Fraction(85, 512) > 0
    # Disjoint strictly interior convex balls removed from a ball leave one
    # connected solid, rather than three disconnected material components.
    proof = _family_nonsweepable_source_proof(complex_)
    assert proof["solid_connected"] is True
    assert proof["strict_cavity_containment"] is True
    assert proof["strict_cavity_disjointness"] is True
    assert proof["minimum_outer_barycentric_coordinates"] == ("1/8", "1/8")
    assert proof["separation"] == "3/16"
    assert proof["solid_volume"] == "85/512"
    assert [item["euler_characteristic"] for item in proof["boundary_components"]] == [
        2,
        2,
        2,
    ]
    assert [
        Fraction(item["signed_volume"]) for item in proof["boundary_components"]
    ] == volumes


@pytest.mark.parametrize(
    "damage",
    (
        "touching-outer",
        "touching-cavity",
        "reversed-cavity",
        "missing-face",
        "wrong-material",
    ),
)
def test_nonsweepable_exact_source_proof_rejects_broken_premises(damage: str) -> None:
    complex_ = _family_nonsweepable_plc_source().complex
    points = np.array(complex_.vertices, copy=True)
    faces = np.array(complex_.polygon_vertices, copy=True).reshape(12, 3)
    regions = np.array(complex_.facet_regions, copy=True)
    if damage == "touching-outer":
        points[4:8, 2] -= 1 / 8
    elif damage == "touching-cavity":
        points[8:12, 0] -= 3 / 16
    elif damage == "reversed-cavity":
        faces[4:8] = faces[4:8, ::-1]
    elif damage == "missing-face":
        faces[4] = faces[5]
    else:
        regions[4:8] = (0, -1)
    changed = phx.meshing.PiecewiseLinearComplex(
        points, tuple(faces), complex_.polygon_facets, regions, complex_.region_ids
    )
    with pytest.raises(ValueError, match="Nonsweepable"):
        _family_nonsweepable_source_proof(changed)


def test_nonsweepable_exact_chart_midpoints_release_only_proved_slanted_edges() -> None:
    from phydrax.meshing.qualification._family import (
        prepare_family_association_transfer,
    )

    args = QualificationControls(family_case="nonsweepable-dual-hex", resolution=2)
    source, request, options, case = _executed_fixture(args)
    result = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    transfer = prepare_family_association_transfer(
        source,
        result,
        request,
        domain=case["domain"],
        limits=request.limits,
        record_phase=lambda _: None,
    )
    assert isinstance(transfer, phx.meshing.PlcAssociationTransfer)
    classes = transfer.classes(result)
    rounded = transfer.protected_edges(result, _classes=classes)
    exact = transfer.protected_edges(result, exact_chart_midpoints=True, _classes=classes)
    # Rounded midpoints of exact dual vertices leave the slanted source facets;
    # the mapped proof covers each whole exact edge chart instead.
    assert np.any(rounded & classes[1].resolved)
    assert not np.any(exact & classes[1].resolved)
    assert np.all(exact <= rounded)


def test_nonsweepable_fixture_preserves_hard_controls_and_void_material_provenance() -> (
    None
):
    args = QualificationControls(family_case="nonsweepable-dual-hex", resolution=2)
    source, request, options, case = _family_qualification_fixture(args)
    assert type(source) is phx.meshing.NativePlcSource
    assert options.route == "plc_dual_hex"
    assert request.target.cell_families.required == ("hexahedron",)
    assert not request.target.cell_families.allow_mixed
    assert request.target.geometry_order == 1
    assert case["expected_regions"] == {"solid": 85 / 512}
    assert case["source_construction_proof"]["solid_connected"] is True
    assert case["nonsweepability_required"] is True
    (size,) = request.size_controls
    assert size.target_size == 1 / args.resolution
    assert size.maximum_size == 2 / args.resolution
    assert size.strength is phx.meshing.SizeControlStrength.HARD
    assert request.size_compliance.relative_tolerance == 0.95
    assert request.size_compliance.target_statistics == ("p50",)
    expected_limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        edges=12,
        faces=12,
        cells=1,
        connectivity_entries=64,
        data_bytes=1024,
        work_units=100,
        geometry_queries=100,
        scratch_bytes=4096,
    )
    assert request.limits == expected_limits
    (region,) = request.region_controls
    assert region.region_name == "solid"
    assert region.material_id == "material-solid"
    assert region.role is phx.meshing.RegionRole.SOLID
    assert [patch.name for patch in request.patch_controls] == [
        "outer-wall",
        "cavity-wall-0",
        "cavity-wall-1",
    ]
    for index, patch in enumerate(request.patch_controls):
        np.testing.assert_array_equal(
            patch.scope.entity_ids, np.arange(4 * index, 4 * index + 4)
        )
        assert patch.scope.source_id == source.source_id
        assert patch.scope.source_revision == source.source_revision
    assert len(request.hole_seeds) == 2
    for index, hole in enumerate(request.hole_seeds):
        offset = 4 * (index + 1)
        np.testing.assert_array_equal(
            hole.point,
            np.asarray(source.complex.vertices)[offset : offset + 4].mean(axis=0),
        )
        np.testing.assert_array_equal(
            hole.scope.entity_ids, np.arange(offset, offset + 4)
        )
        assert hole.scope.source_id == source.source_id
        assert hole.scope.source_revision == source.source_revision


@pytest.fixture(scope="module")
def published_cavity_material() -> tuple[
    phx.meshing.CellMeshingResult, dict[str, Any], tuple[str, ...]
]:
    args = QualificationControls(family_case="nonsweepable-dual-hex", resolution=2)
    source, request, options, case = _executed_fixture(args)
    result = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    return result, case, tuple(control.name for control in request.patch_controls[1:])


def test_published_cavities_satisfy_the_authored_material_controls(
    published_cavity_material: tuple[
        phx.meshing.CellMeshingResult, dict[str, Any], tuple[str, ...]
    ],
) -> None:
    result, case, _ = published_cavity_material
    regions = np.zeros(
        sum(block.cell_count for block in result.mesh.blocks), dtype=np.int64
    )
    accepted = _family_material_acceptance(result, case, result.mesh, regions, {})
    assert accepted["material_ids"] == ["material-solid"]


@pytest.mark.parametrize("cavity_index", (0, 1), ids=("first-cavity", "second-cavity"))
def test_material_acceptance_refuses_a_missing_authored_cavity(
    published_cavity_material: tuple[
        phx.meshing.CellMeshingResult, dict[str, Any], tuple[str, ...]
    ],
    cavity_index: int,
) -> None:
    result, case, names = published_cavity_material
    damaged = eqx.tree_at(
        lambda value: value.patches,
        result,
        tuple(patch for patch in result.patches if patch.name != names[cavity_index]),
    )
    regions = np.zeros(
        sum(block.cell_count for block in result.mesh.blocks), dtype=np.int64
    )
    with pytest.raises(QualificationFailure):
        _family_material_acceptance(damaged, case, result.mesh, regions, {})


def test_material_acceptance_refuses_overlapping_cavity_memberships(
    published_cavity_material: tuple[
        phx.meshing.CellMeshingResult, dict[str, Any], tuple[str, ...]
    ],
) -> None:
    result, case, names = published_cavity_material
    first = next(patch for patch in result.patches if patch.name == names[0])
    patches = tuple(
        eqx.tree_at(lambda value: value.scope, patch, first.scope)
        if patch.name == names[1]
        else patch
        for patch in result.patches
    )
    damaged = eqx.tree_at(lambda value: value.patches, result, patches)
    regions = np.zeros(
        sum(block.cell_count for block in result.mesh.blocks), dtype=np.int64
    )
    with pytest.raises(QualificationFailure):
        _family_material_acceptance(damaged, case, result.mesh, regions, {})
