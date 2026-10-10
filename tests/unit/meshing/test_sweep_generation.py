#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import numpy as np
import pytest

import phydrax as phx
from examples.native_multiblock_meshing import rectangle
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    swept_coordinate_element,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._mesh_certificates import (
    ImplicitBoundarySource,
    MappedDomainBoundarySource,
    PiecewiseLinearDomain,
)
from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._controls import LayerSchedule
from phydrax.meshing._structured import generate_structured_block
from phydrax.meshing._sweep import generate_sweep, SweepControl, SweepMapKind


@pytest.mark.parametrize(
    "family,vertices,cells,expected",
    (
        ("triangle", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),), "prism"),
        (
            "quadrilateral",
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ((0, 1, 2, 3),),
            "hexahedron",
        ),
    ),
    ids=("triangle-prisms", "quad-hexes"),
)
def test_extrusion_preserves_physical_schedule_and_cap_correspondence(
    family: str,
    vertices: tuple[tuple[float, float], ...],
    cells: tuple[tuple[int, ...], ...],
    expected: str,
) -> None:
    profile = CellMesh(
        np.asarray(vertices, dtype=np.float64),
        (CellBlock("profile", family, np.asarray(cells, dtype=np.int32)),),
        vertex_global_ids=np.arange(len(vertices), dtype=np.int64) + 40,
    )
    schedule = LayerSchedule(np.asarray((0.1, 0.2, 0.4), dtype=np.float64))
    result = generate_sweep(
        profile, SweepControl(SweepMapKind.EXTRUSION, schedule, "straight-extrusion")
    )
    points = np.asarray(result.mesh.coordinates).reshape(4, len(vertices), 3)
    np.testing.assert_allclose(
        np.diff(points[..., 2], axis=0),
        np.broadcast_to(np.asarray(schedule.thicknesses)[:, None], (3, len(vertices))),
        atol=1e-14,
    )
    np.testing.assert_array_equal(
        result.source_vertex_ids, np.asarray(profile.vertex_global_ids)
    )
    np.testing.assert_array_equal(
        result.layer_vertex_ids[-1],
        np.asarray(result.mesh.vertex_global_ids)[-len(vertices) :],
    )
    assert result.mesh.blocks[0].cell_kind == expected
    assert result.embedding.status == "certified"


def test_revolution_preserves_profile_radius_and_declared_angle() -> None:
    # Axial-then-radial source orientation has normal in the sweep direction.
    points = ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (2.0, 0.0, 0.0))
    profile = CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            CellBlock(
                "profile", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    result = generate_sweep(
        profile,
        SweepControl(
            SweepMapKind.REVOLUTION,
            LayerSchedule(np.full(4, np.pi / 8.0, dtype=np.float64)),
            "quarter-turn",
        ),
    )
    coordinates = np.asarray(result.mesh.coordinates).reshape(5, 4, 3)
    np.testing.assert_allclose(
        np.linalg.norm(coordinates[..., :2], axis=-1),
        np.broadcast_to((1.0, 1.0, 2.0, 2.0), (5, 4)),
        atol=1e-13,
    )
    np.testing.assert_allclose(coordinates[-1, :, 0], 0.0, atol=1e-13)
    np.testing.assert_allclose(coordinates[-1, :, 1], (1.0, 1.0, 2.0, 2.0), atol=1e-13)
    assert result.mesh.blocks[0].cell_kind == "hexahedron"
    assert result.embedding.status == "certified"


def test_axis_collapse_is_not_silently_replaced_with_tetrahedra() -> None:
    profile = CellMesh(
        np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "profile", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    with pytest.raises(ValueError, match="core correspondence collapses"):
        generate_sweep(
            profile,
            SweepControl(
                SweepMapKind.REVOLUTION,
                LayerSchedule(np.asarray((0.2, 0.2), dtype=np.float64)),
                "axis-collapse",
            ),
        )


def test_twist_with_interior_column_collapse_rejected() -> None:
    profile = CellMesh(
        np.asarray(
            ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)), dtype=np.float64
        ),
        (
            CellBlock(
                "profile", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    frames = np.stack((np.eye(3, dtype=np.float64), np.diag((-1.0, -1.0, 1.0))))
    control = SweepControl(
        SweepMapKind.FRAMES,
        LayerSchedule(np.asarray((1.0,), dtype=np.float64)),
        "half-turn-collapse",
        frames=frames,
        translations=np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64),
    )
    with pytest.raises(MeshingFailure, match="Jacobian"):
        generate_sweep(profile, control)


def test_closed_turn_requires_explicit_seam_topology() -> None:
    with pytest.raises(ValueError, match="explicitly glued seam"):
        SweepControl(
            SweepMapKind.REVOLUTION,
            LayerSchedule(np.asarray((np.pi, np.pi), dtype=np.float64)),
            "undeclared-seam",
        )


def test_declared_full_revolution_glues_exact_source_vertex_seam() -> None:
    profile = CellMesh(
        np.asarray(
            ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "profile", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    control = SweepControl(
        SweepMapKind.REVOLUTION,
        LayerSchedule(np.full(12, np.pi / 6.0, dtype=np.float64)),
        "closed-cylinder",
        closed=True,
    )
    result = generate_sweep(profile, control)
    np.testing.assert_array_equal(result.layer_vertex_ids[0], result.layer_vertex_ids[-1])
    np.testing.assert_array_equal(
        np.asarray(result.mesh.blocks[0].vertices)[-1, 4:], (0, 1, 2, 3)
    )
    assert result.mesh.coordinates.shape[0] == 48
    assert result.mesh.blocks[0].cell_count == 12
    assert result.embedding.status == "certified"


@pytest.mark.parametrize("perturbation", ("translation", "rotation"))
def test_closed_frame_sweep_refuses_nearby_but_unequal_source_transforms(
    perturbation: str,
) -> None:
    frames = np.tile(np.eye(3, dtype=np.float64), (4, 1, 1))
    translations = np.zeros((4, 3), dtype=np.float64)
    if perturbation == "translation":
        translations[-1, 0] = 5e-13
    else:
        angle = 5e-13
        frames[-1] = (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    with pytest.raises(ValueError, match="exact first/last source transforms"):
        SweepControl(
            SweepMapKind.FRAMES,
            LayerSchedule(np.ones(3)),
            "cracked-frame-seam",
            frames=frames,
            translations=translations,
            closed=True,
        )


def test_revolution_identity_station_retains_original_source_bits() -> None:
    from phydrax.meshing._sweep import _mapped_stations

    points = np.asarray(
        ((1.2345678901234567, -0.0, 0.3456789012345678),), dtype=np.float64
    )
    control = SweepControl(
        SweepMapKind.REVOLUTION,
        LayerSchedule(np.asarray((0.2,))),
        "offset-axis",
        origin=np.asarray((0.1234567890123456, 0.2, 0.3), dtype=np.float64),
    )
    mapped, _ = _mapped_stations(points, control)
    np.testing.assert_array_equal(mapped[0].view(np.uint64), points.view(np.uint64))


def test_native_sweep_publishes_coverage_and_exact_cell_corner_correspondence() -> None:
    geometry = phx.geometry.Box(
        (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), feature_id="box"
    ).compile()
    query = ImplicitBoundarySource(geometry, source_id="box", spacing=0.025)
    # This source boundary is independent of generated lattice vertices.
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    facets = np.asarray(
        tuple(triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))),
        dtype=np.int64,
    )
    domain = PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (12, 1)),
        ("solid",),
        source_id="box",
    )
    raw = generate_structured_block(rectangle("profile", 0.0, 1.0, 2)).mesh
    profile = CellMesh(
        raw.coordinates, raw.blocks, vertex_global_ids=np.arange(9, dtype=np.int64) + 50
    )
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        LayerSchedule(np.asarray((0.25, 0.75), dtype=np.float64)),
        "box-extrusion",
    )
    source = phx.meshing.NativeSweepSource(
        profile,
        control,
        "box",
        query.source_revision,
        fidelity_source=query,
        maximum_deviation=0.2,
        domain=domain,
    )
    scope = phx.meshing.MeshingScope(
        "box",
        query.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "box-boundary",
        np.arange(12, dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("hexahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.5, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("sweep")
    )
    result = provider.plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()
    attributes = {
        attribute.name: np.asarray(attribute.values) for attribute in result.attributes
    }
    identifiers = attributes["profile:source_profile_vertex_id"]
    stations = attributes["profile:sweep_stations"]
    np.testing.assert_array_equal(
        attributes["profile:source_profile_cell_id"],
        np.tile(np.asarray(profile.blocks[0].global_ids), control.schedule.layer_count),
    )
    reconstructed = np.pad(
        np.asarray(profile.coordinates)[identifiers - 50], ((0, 0), (0, 0), (0, 1))
    )
    reconstructed[..., 2] = np.repeat(stations, 4, axis=1)
    actual = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    np.testing.assert_allclose(reconstructed, actual, atol=1e-14)
    assert result.certification is not None
    assert result.certification.passed
    assert result.certification.coverage is not None
    assert result.certification.coverage.status == "certified"
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"


def _twisted_curved_source(
    degree: int = 2,
    /,
    *,
    angle: float = 0.15,
    cap_offset: tuple[float, float, float] = (0.1, -0.05, 1.0),
) -> tuple[phx.meshing.NativeSweepSource, phx.meshing.VolumeMeshingSpec]:
    """Author a non-box source independently of target construction."""
    profile_element = coordinate_lagrange_element("quadrilateral", degree)
    nodes = np.asarray(profile_element.reference_nodes)
    u, v = nodes[:, 0], nodes[:, 1]
    controls = np.column_stack((u + 0.25 * v, v + 0.125 * u * (1.0 - u)))
    corners = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.25, 1.0), (0.25, 1.0)), dtype=np.float64
    )
    profile = CellMesh(
        corners,
        (
            CellBlock(
                "profile", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    profile_geometry = CellGeometrySpec(
        {"profile": profile_element},
        {"profile": np.arange(profile_element.local_dof_count, dtype=np.int32)[None, :]},
        controls,
    )
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    frames = np.stack((np.eye(3, dtype=np.float64), rotation))
    offsets = np.asarray(((0.0, 0.0, 0.0), cap_offset), dtype=np.float64)
    control = SweepControl(
        SweepMapKind.FRAMES,
        LayerSchedule(np.asarray((1.0,), dtype=np.float64)),
        "authored-piecewise-affine-twist",
        frames=frames,
        translations=offsets,
    )
    # Root reference topology and all physical controls preexist generation.
    cube = reference_cell_topology("hexahedron")
    reference_points = np.asarray(cube.vertices, dtype=np.float64)
    reference_mesh = CellMesh(
        reference_points,
        (CellBlock("root", "hexahedron", np.arange(8, dtype=np.int32)[None, :]),),
    )
    facets = np.asarray(
        tuple(
            triangle
            for a, b, c, d in cube.entities[2]
            for triangle in ((a, b, c), (a, c, d))
        ),
        dtype=np.int64,
    )
    reference_domain = PiecewiseLinearDomain(
        reference_points,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1)),
        ("solid",),
        source_id="twisted-source-reference",
    )
    source_points = np.pad(controls, ((0, 0), (0, 1)))
    root_controls = np.concatenate(
        (source_points, source_points @ rotation.T + offsets[1])
    )
    root_element = (
        coordinate_lagrange_element("hexahedron", 1)
        if degree == 1
        else swept_coordinate_element(profile_element)
    )
    root_geometry = CellGeometrySpec(
        {"root": root_element},
        {"root": np.arange(root_element.local_dof_count, dtype=np.int32)[None, :]},
        root_controls,
    )
    revision = canonical_fingerprint(
        {
            "kind": "independently-authored-twisted-source",
            "profile_basis": profile_element.element_id,
            "profile_controls": controls,
            "frames": frames,
            "offsets": offsets,
        }
    )
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        root_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="twisted-source",
        source_revision=revision,
    )
    query = MappedDomainBoundarySource(domain, np.asarray((0,), dtype=np.int64))
    correspondence = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        reference_mesh.topology_id,
        {"sweep:profile": np.asarray((0,), dtype=np.int64)},
        {"sweep:profile": np.arange(8, dtype=np.int64)[None, :]},
    )
    source = phx.meshing.NativeSweepSource(
        profile,
        control,
        domain.source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=1e-10,
        domain=domain,
        profile_geometry=profile_geometry,
        root_correspondence=correspondence,
    )
    scope = phx.meshing.MeshingScope(
        domain.source_id,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "twisted-source-boundary",
        np.arange(facets.shape[0], dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("hexahedron",)),
            geometry_order=degree,
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 1.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    return source, specification


@pytest.mark.parametrize(
    "degree", (1, 2), ids=("standard-bilinear-root", "curved-profile-root")
)
def test_twisted_curved_profile_publication_retains_the_full_source_map(
    degree: int,
) -> None:
    source, specification = _twisted_curved_source(degree)
    result = (
        phx.meshing.NativeMeshingProvider(phx.meshing.NativeMeshingOptions("sweep"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    reference = np.asarray(
        ((0.25, 0.25, 0.25), (0.5, 0.5, 0.5), (0.75, 0.4, 0.8)), dtype=np.float64
    )
    u, v, w = reference.T
    bow = 0.125 * u * (1.0 - u) if degree == 2 else np.zeros_like(u)
    profile_points = np.column_stack((u + 0.25 * v, v + bow, np.zeros_like(u)))
    transformed = profile_points @ np.asarray(source.control.frames[1]).T + np.asarray(
        source.control.translations[1]
    )
    expected = profile_points * (1.0 - w[:, None]) + transformed * w[:, None]
    element = result.geometry.elements[0]
    if not isinstance(element, phx.discretization.FiniteElementSpec):
        raise TypeError(
            "The published sweep must retain its actual tabulated coordinate element."
        )
    route = np.asarray(result.geometry.geometry_dofs[0])[0]
    actual = (
        np.asarray(element.tabulate(reference)[0])
        @ np.asarray(result.geometry.coordinates)[route]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)
    assert result.certification is not None
    assert result.certification.passed
    assert result.certification.coverage is not None
    assert result.certification.coverage.status == "certified"
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.mesh_to_source_upper <= 1e-10
    assert result.certification.fidelity.source_to_mesh_upper <= 1e-10


def test_sweep_geometry_identity_changes_when_only_profile_curvature_changes() -> None:
    source, _ = _twisted_curved_source()
    controls = np.asarray(source.profile_geometry.coordinates).copy()
    controls[4, 1] += 0.01
    changed = CellGeometrySpec(
        dict(
            zip(
                source.profile_geometry.block_names,
                source.profile_geometry.elements,
                strict=True,
            )
        ),
        dict(
            zip(
                source.profile_geometry.block_names,
                source.profile_geometry.geometry_dofs,
                strict=True,
            )
        ),
        controls,
    )
    another = phx.meshing.NativeSweepSource(
        source.profile,
        source.control,
        source.source_id,
        source.source_revision,
        fidelity_source=source.fidelity_source,
        maximum_deviation=source.maximum_deviation,
        domain=source.domain,
        profile_geometry=changed,
        root_correspondence=source.root_correspondence,
    )
    assert another.binding_id != source.binding_id


def test_sweep_does_not_move_a_fixed_profile_corner_to_match_a_coordinate_map() -> None:
    source, _ = _twisted_curved_source()
    controls = np.asarray(source.profile_geometry.coordinates).copy()
    controls[0, 0] = 1e-13
    changed = CellGeometrySpec(
        dict(
            zip(
                source.profile_geometry.block_names,
                source.profile_geometry.elements,
                strict=True,
            )
        ),
        dict(
            zip(
                source.profile_geometry.block_names,
                source.profile_geometry.geometry_dofs,
                strict=True,
            )
        ),
        controls,
    )
    with pytest.raises(ValueError):
        generate_sweep(source.profile, source.control, profile_geometry=changed)


def test_authoritative_root_binding_cannot_downgrade_requested_geometry_order() -> None:
    source, specification = _twisted_curved_source(
        1, angle=0.0, cap_offset=(0.0, 0.0, 1.0)
    )
    element = coordinate_lagrange_element("quadrilateral", 2)
    nodes = np.asarray(element.reference_nodes)
    controls = np.column_stack((nodes[:, 0] + 0.25 * nodes[:, 1], nodes[:, 1]))
    geometry = CellGeometrySpec(
        {"profile": element}, {"profile": np.arange(9, dtype=np.int32)[None, :]}, controls
    )
    quadratic_source = phx.meshing.NativeSweepSource(
        source.profile,
        source.control,
        source.source_id,
        source.source_revision,
        fidelity_source=source.fidelity_source,
        maximum_deviation=source.maximum_deviation,
        domain=source.domain,
        profile_geometry=geometry,
        root_correspondence=source.root_correspondence,
    )
    quadratic_specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, specification.target.cell_families, geometry_order=2
        ),
        specification.boundary_scope,
        specification.fill_strategy,
        size_controls=specification.size_controls,
    )
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("sweep")
    ).plan(
        quadratic_source,
        quadratic_specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    with pytest.raises(MeshingFailure) as failure:
        plan.execute()
    assert failure.value.evidence.category is MeshingFailureCategory.COMPLIANCE_FAILED
    assert dict(failure.value.evidence.requested)["geometry_order"] == 2.0
    assert dict(failure.value.evidence.achieved)["geometry_order"] == 1.0


def test_sweep_station_bank_refuses_against_original_live_storage_without_changing_profile() -> (
    None
):
    from phydrax.meshing._sweep import _mapped_stations
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    points = np.tile(np.asarray(((1.0, 2.0, 3.0),)), (1000, 1))
    original = points.copy()
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        LayerSchedule(np.asarray((0.2, 0.3))),
        "storage-bound-extrusion",
    )
    limits = phx.meshing.MeshingLimits(maximum_scratch_bytes=1024**2)
    with pytest.raises(MeshingFailure) as refused:
        with native_volume_execution_budget(limits) as execution:
            retained = execution.allocate_host_array((1024**2 - 4096,), np.uint8)
            retained.fill(41)
            _mapped_stations(points, control)
    assert refused.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert retained[0] == retained[-1] == 41
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize(
    "closed,expected",
    (
        (False, ((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8), (6, 7, 8, 9, 10, 11))),
        (True, ((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8), (6, 7, 8, 0, 1, 2))),
    ),
)
def test_sweep_columns_retain_layer_major_order_and_declared_cap_identity(
    closed: bool, expected: tuple[tuple[int, ...], ...]
) -> None:
    from phydrax.meshing._sweep import _column_connectivity

    result = _column_connectivity(np.asarray(((0, 1, 2),), dtype=np.int32), 3, 3, closed)
    np.testing.assert_array_equal(result, expected)


def test_sweep_columns_refuse_scientific_node_identity_wraparound() -> None:
    from phydrax.meshing._sweep import _column_connectivity

    with pytest.raises(MeshingFailure) as refused:
        _column_connectivity(np.asarray(((0, 1, 2),), dtype=np.int32), 2, 2**30, False)
    assert refused.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_curved_sweep_physical_edges_use_original_whole_map_arc_enclosures() -> None:
    from phydrax.meshing.providers._native_structured import _mapped_edge_evidence

    source, specification = _twisted_curved_source()
    identity = source.binding_id
    original = np.asarray(source.profile_geometry.resolve(source.profile)[2]).copy()
    construction = generate_sweep(
        source.profile, source.control, profile_geometry=source.profile_geometry
    )
    edges, lengths, lower, upper = _mapped_edge_evidence(
        construction.mesh, construction.geometry, specification
    )
    row = next(index for index, pair in enumerate(edges.tolist()) if set(pair) == {0, 1})
    points = np.asarray(construction.mesh.coordinates)
    chord = np.linalg.norm(points[1] - points[0])
    assert lower[row] > chord
    assert lower[row] <= lengths[row] <= upper[row]
    assert source.binding_id == identity
    np.testing.assert_array_equal(
        np.asarray(source.profile_geometry.resolve(source.profile)[2]), original
    )


def test_closed_revolution_retains_physical_cap_compatibility_refusal() -> None:
    profile = CellMesh(
        np.asarray(
            (
                (1000.0, 0.0, 0.0),
                (1000.0, 0.0, 1.0),
                (1001.0, 0.0, 1.0),
                (1001.0, 0.0, 0.0),
            )
        ),
        (
            CellBlock(
                "profile", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    control = SweepControl(
        SweepMapKind.REVOLUTION,
        LayerSchedule(np.full(4, (2.0 * np.pi + 5e-13) / 4.0)),
        "incompatible-large-radius-cap",
        closed=True,
    )
    with pytest.raises(ValueError, match="source/cap correspondence"):
        generate_sweep(profile, control)
