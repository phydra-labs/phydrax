from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellLocationStatus,
    CellMesh,
)
from phydrax.discretization._cell_geometry import (
    PerCellRestrictedCellGeometryElement,
)
from phydrax.discretization._coordinate_enclosure import (
    axes,
    constant,
    multiply,
    scale,
    sum_polynomials,
)
from phydrax.discretization._polyhedral_locator import PreparedPolyhedralCellLocator
from phydrax.discretization.fem import lagrange_element
from phydrax.discretization.finite_volume import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.linalg._exact import solve_exact
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    NativeMeshingOptions,
    NativeMeshingProvider,
    RegionRole,
    RegionSeed,
    SizeControlStrength,
    SurfaceMeshingSpec,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._hex_schedule import NativeHexGridSchedule
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._polyhedral_generation import NativePolyhedralSchedule
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native_sources import (
    NativePlanarSource,
    NativePlcSource,
    NativePolyhedralSource,
)


pytestmark = pytest.mark.meshcore

_COORDINATES = SpatialCoordinateContract.si()


def _bipyramid_source() -> NativePlcSource:
    vertices = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    polygons = np.asarray(
        (
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (2, 1, 4),
            (0, 4, 1),
            (0, 2, 4),
            (0, 1, 2),
        ),
        dtype=np.int64,
    )
    facet_regions = np.asarray(
        ((-1, 0), (-1, 0), (-1, 0), (-1, 1), (-1, 1), (-1, 1), (0, 1)),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        polygons,
        np.arange(polygons.shape[0], dtype=np.int64),
        facet_regions,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"bipyramid-vertex-{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"bipyramid-facet-{row}" for row in range(polygons.shape[0])),
        facet_occurrence_paths=((),) * polygons.shape[0],
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
    )
    return NativePlcSource(complex_, "native-general-bipyramid", "r1")


def _cube_source() -> NativePlcSource:
    vertices = np.asarray(
        tuple(
            (float(x), float(y), float(z))
            for z in range(2)
            for y in range(2)
            for x in range(2)
        ),
        dtype=np.float64,
    )
    row = {
        tuple(int(value) for value in point): index
        for index, point in enumerate(vertices)
    }
    polygons = (
        (row[(0, 0, 0)], row[(0, 1, 0)], row[(1, 1, 0)], row[(1, 0, 0)]),
        (row[(0, 0, 1)], row[(1, 0, 1)], row[(1, 1, 1)], row[(0, 1, 1)]),
        (row[(0, 0, 0)], row[(0, 0, 1)], row[(0, 1, 1)], row[(0, 1, 0)]),
        (row[(1, 0, 0)], row[(1, 1, 0)], row[(1, 1, 1)], row[(1, 0, 1)]),
        (row[(0, 0, 0)], row[(1, 0, 0)], row[(1, 0, 1)], row[(0, 0, 1)]),
        (row[(0, 1, 0)], row[(0, 1, 1)], row[(1, 1, 1)], row[(1, 1, 0)]),
    )
    segments = tuple(
        tuple(sorted((row[first], row[second])))
        for first, second in (
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 0, 1)),
            ((1, 1, 1), (0, 1, 1)),
            ((1, 1, 1), (1, 0, 1)),
            ((1, 1, 1), (1, 1, 0)),
            ((1, 0, 0), (1, 1, 0)),
            ((1, 0, 0), (1, 0, 1)),
            ((0, 1, 0), (1, 1, 0)),
            ((0, 1, 0), (0, 1, 1)),
            ((0, 0, 1), (1, 0, 1)),
            ((0, 0, 1), (0, 1, 1)),
        )
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(np.asarray(face, dtype=np.int64) for face in polygons),
        np.arange(len(polygons), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(polygons), 1)),
        ("solid",),
        segments=np.asarray(segments, dtype=np.int64),
        vertex_ids=tuple(f"cube-vertex-{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"cube-facet-{row}" for row in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segment_ids=tuple(f"cube-edge-{row}" for row in range(len(segments))),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
    )
    return NativePlcSource(complex_, "native-general-cube", "r1")


def _volume_request(
    source: NativePlcSource,
    /,
    *,
    family_policy: CellFamilyPolicy,
    fill_strategy: VolumeFillStrategy,
    size: float | None = None,
    single_region_seed: tuple[float, float, float] = (0.5, 0.5, 0.5),
    single_region_material: str | None = None,
    two_region_seed_points: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] = ((0.0, 0.2, 0.25), (0.0, 0.2, -0.25)),
) -> VolumeMeshingSpec:
    boundary = MeshingScope.whole(source, 2)
    controls = (
        ()
        if size is None
        else (
            UniformSizeControl(
                boundary,
                size,
                strength=SizeControlStrength.SOFT,
            ),
        )
    )
    seeds = (
        (
            RegionSeed(
                np.asarray(two_region_seed_points[0], dtype=np.float64),
                "upper",
                "upper-material",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray(two_region_seed_points[1], dtype=np.float64),
                "lower",
                "lower-material",
                RegionRole.SOLID,
            ),
        )
        if source.complex.region_ids == ("upper", "lower")
        else (
            RegionSeed(
                np.asarray(single_region_seed, dtype=np.float64),
                source.complex.region_ids[0],
                (
                    f"{source.complex.region_ids[0]}-material"
                    if single_region_material is None
                    else single_region_material
                ),
                RegionRole.SOLID,
            ),
        )
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, family_policy),
        boundary,
        fill_strategy,
        size_controls=controls,
        region_seeds=seeds,
    )


def _accepted_public_result(result: Any, source: Any, /) -> None:
    assert result.certification is not None and result.certification.passed
    inputs = result.certification.inputs
    assert inputs.source_id == source.source_id
    assert inputs.source_revision == source.source_revision
    assert result.audit.passed
    assert result.compliance.passed
    assert result.provider.name == "phydrax-native"
    assert result.associations
    assert any(association.complete for association in result.associations)


def _assert_plc_facet_associations(result: Any, /) -> None:
    facet_associations = tuple(
        association
        for association in result.associations
        if GeometrySourceEntityRole.FACET in association.source_roles
    )
    assert facet_associations
    assert any(association.complete for association in facet_associations)


def _region_materials(result: Any, /) -> set[str]:
    return {zone.material_id for zone in result.zones if zone.material_id is not None}


def test_polyhedral_power_keeps_region_sites_and_fv_field_measure() -> None:
    plc = _bipyramid_source()
    source = NativePolyhedralSource(
        plc.complex,
        plc.source_id,
        plc.source_revision,
        sites=np.asarray(
            ((0.0, 0.25, 0.5), (0.0, 0.25, -0.5)),
            dtype=np.float64,
        ),
        weights=np.asarray((0.0, 0.2), dtype=np.float64),
        site_ids=("upper-site", "lower-site"),
    )
    scope = MeshingScope.whole(source, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("polyhedron",)),
        ),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
        region_seeds=(
            RegionSeed(
                np.asarray((0.0, 0.25, 0.25), dtype=np.float64),
                "upper",
                "upper-material",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray((0.0, 0.25, -0.25), dtype=np.float64),
                "lower",
                "lower-material",
                RegionRole.SOLID,
            ),
        ),
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_restricted_power",
            polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
        )
    ).plan(source, specification, coordinate_contract=_COORDINATES)

    result = plan.execute()

    _accepted_public_result(result, source)
    _assert_plc_facet_associations(result)
    assert {block.cell_kind for block in result.mesh.blocks} == {"polyhedron"}
    assert _region_materials(result) == {"upper-material", "lower-material"}
    geometry = result.geometry
    assert geometry is not None and geometry.source_map is not None
    assert geometry.source_map.source_id == source.source_id
    assert geometry.source_map.source_revision == source.source_revision
    exact_vertices = geometry.source_coordinates()
    assert any(
        Fraction.from_float(float(value)) != value
        for point in exact_vertices
        for value in point
    )

    locator = PreparedPolyhedralCellLocator(result.mesh, geometry=geometry)
    located = locator.locate(
        np.asarray(
            ((0.0, 0.25, 0.5), (0.0, 0.25, -0.5)),
            dtype=np.float64,
        )
    )
    assert located.status.tolist() == [
        int(CellLocationStatus.LOCATED),
        int(CellLocationStatus.LOCATED),
    ]
    interface_sides = {
        frozenset(pair)
        for association in result.associations
        for pair in association.source_region_sides
        if pair[0] is not None and pair[1] is not None and pair[0] != pair[1]
    }
    assert frozenset(("upper", "lower")) in interface_sides

    cell_ids = np.asarray(result.mesh.entity_set(3).entity_ids, dtype=np.int64)
    attributes = {attribute.name: attribute for attribute in result.attributes}
    site_attribute = attributes["power_site_index"]
    scope_ids = np.asarray(site_attribute.scope.entity_ids, dtype=np.int64)
    np.testing.assert_array_equal(scope_ids, np.sort(cell_ids))
    site_indices = np.asarray(site_attribute.values, dtype=np.int64)[
        np.searchsorted(scope_ids, cell_ids)
    ]
    labels = {label.name: label for label in result.labels}
    for site_index, site_id in enumerate(("upper-site", "lower-site")):
        site_cells = cell_ids[site_indices == site_index]
        assert site_cells.size
        np.testing.assert_array_equal(
            labels[f"power-site:{site_id}"].scope.entity_ids, np.sort(site_cells)
        )
    for point_row, expected_site in enumerate((0, 1)):
        located_cells = np.asarray(located.candidate_cells[point_row], dtype=np.int64)
        located_cells = located_cells[located_cells >= 0]
        assert located_cells.size
        assert np.any(site_indices[located_cells] == expected_site)

    finite_volume = prepare_polyhedral_finite_volume_geometry(result.mesh)
    cell_volumes = np.asarray(finite_volume.cell_volumes, dtype=np.float64)
    cut_height = Fraction.from_float(0.2) / 2
    remaining_scale = Fraction(1) - cut_height
    np.testing.assert_allclose(
        np.bincount(site_indices, weights=cell_volumes, minlength=2),
        (
            float(remaining_scale**3 / 3),
            float((Fraction(2) - remaining_scale**3) / 3),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    field_by_site = np.asarray((2.0, -1.0), dtype=np.float64)
    cell_field = field_by_site[site_indices]
    np.testing.assert_allclose(
        np.dot(cell_field, cell_volumes),
        float((3 * remaining_scale**3 - 2) / 3),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_public_planar_dual_quad_route_executes_feature_guided_acceptance() -> None:
    region = PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id="native-public-square",
    )
    source = NativePlanarSource(
        region,
        "r1",
        region_scope_entity_ids=np.asarray((501,), dtype=np.int64),
        vertex_entity_ids=np.asarray((101, 102, 103, 104), dtype=np.int64),
        edge_entity_ids=np.asarray((201, 202, 203, 204), dtype=np.int64),
    )
    scope = MeshingScope.whole(source, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("quadrilateral",)),
        ),
        scope,
        # The unit-square diagonal (sqrt 2) must not force a central Steiner
        # point: a four-triangle star has no positive-quality quad pairing.
        size_controls=(
            UniformSizeControl(scope, 1.5, strength=SizeControlStrength.SOFT),
        ),
    )
    plan = NativeMeshingProvider(NativeMeshingOptions("planar_dual_quad")).plan(
        source, specification, coordinate_contract=_COORDINATES
    )

    assert plan.route == "planar_dual_quad"
    result = plan.execute()

    _accepted_public_result(result, source)
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}


def test_public_plc_dual_hex_route_keeps_material_and_facet_ancestry() -> None:
    source = _bipyramid_source()
    specification = _volume_request(
        source,
        family_policy=CellFamilyPolicy(required=("hexahedron",)),
        fill_strategy=VolumeFillStrategy.SIMPLEX,
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_dual_hex",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
        )
    ).plan(source, specification, coordinate_contract=_COORDINATES)

    assert plan.route == "plc_dual_hex"
    result = plan.execute()

    _accepted_public_result(result, source)
    _assert_plc_facet_associations(result)
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert _region_materials(result) == {"upper-material", "lower-material"}
    assert any(
        positive is not None and negative is not None and positive != negative
        for association in result.associations
        for positive, negative in association.source_region_sides
    )


def test_public_plc_hex_dominant_route_reports_real_mixed_transitions() -> None:
    source = _bipyramid_source()
    specification = _volume_request(
        source,
        family_policy=CellFamilyPolicy(
            required=("hexahedron",),
            allowed_transitions=("pyramid", "tetrahedron"),
            allow_mixed=True,
        ),
        fill_strategy=VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION,
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_hex_dominant",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
            hex_core_fraction=0.5,
        )
    ).plan(source, specification, coordinate_contract=_COORDINATES)

    assert plan.route == "plc_hex_dominant"
    result = plan.execute()

    _accepted_public_result(result, source)
    _assert_plc_facet_associations(result)
    kinds = {block.cell_kind for block in result.mesh.blocks}
    assert "hexahedron" in kinds
    assert kinds <= {"hexahedron", "pyramid", "tetrahedron"}
    assert _region_materials(result) == {"upper-material", "lower-material"}


def test_public_plc_balanced_grid_route_accepts_cut_cell_hexes() -> None:
    source = _bipyramid_source()
    specification = _volume_request(
        source,
        family_policy=CellFamilyPolicy(required=("hexahedron",)),
        fill_strategy=VolumeFillStrategy.SIMPLEX,
        size=0.8,
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_balanced_grid_hex",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
            grid_schedule=NativeHexGridSchedule(
                "balanced_grid",
                minimum_scaled_jacobian=1.0e-10,
                minimum_mean_ratio=1.0e-10,
                maximum_aspect_ratio=1.0e10,
            ),
        )
    ).plan(source, specification, coordinate_contract=_COORDINATES)

    assert plan.route == "plc_balanced_grid_hex"
    result = plan.execute()

    _accepted_public_result(result, source)
    _assert_plc_facet_associations(result)
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert _region_materials(result) == {"upper-material", "lower-material"}


def test_public_plc_frame_grid_route_uses_varying_integer_layout() -> None:
    source = _cube_source()
    specification = _volume_request(
        source,
        family_policy=CellFamilyPolicy(required=("hexahedron",)),
        fill_strategy=VolumeFillStrategy.SIMPLEX,
        size=2.0,
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_frame_grid_hex",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
            grid_schedule=NativeHexGridSchedule(
                "frame_grid",
                minimum_scaled_jacobian=1.0e-10,
                minimum_mean_ratio=1.0e-10,
                maximum_aspect_ratio=1.0e10,
            ),
        )
    ).plan(source, specification, coordinate_contract=_COORDINATES)

    assert plan.route == "plc_frame_grid_hex"
    result = plan.execute()

    _accepted_public_result(result, source)
    _assert_plc_facet_associations(result)
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert _region_materials(result) == {"solid-material"}


def _curved_public_plc_source() -> NativePlcSource:
    source_id, source_revision = "native-public-curved-bipyramid", "r1"
    vertices = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.2),
            (0.0, 0.0, -0.2),
        ),
        dtype=np.float64,
    )
    root_vertex_ids = np.asarray((101, 103, 107, 109, 113), dtype=np.int64)
    tetrahedra = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
    root_cell_ids = np.asarray((901, 1703), dtype=np.int64)
    u, v, w = axes(3)
    variables = (u, v, w)
    alpha = Fraction(1, 16)
    root_charts = []
    for tetrahedron in tetrahedra:
        corners = tuple(
            tuple(Fraction(float(value)) for value in vertices[int(row)])
            for row in tetrahedron
        )
        matrix = tuple(
            tuple(corners[column + 1][axis] - corners[0][axis] for column in range(3))
            for axis in range(3)
        )
        direction = tuple(
            row[0]
            for row in solve_exact(
                matrix,
                ((Fraction(0),), (Fraction(0),), (Fraction(1),)),
            )
        )
        carrier = tuple(
            sum_polynomials(
                (
                    constant(corners[0][axis], 3),
                    *(
                        scale(variables[column], matrix[axis][column])
                        for column in range(3)
                    ),
                ),
                3,
            )
            for axis in range(3)
        )
        shear = scale(multiply(multiply(carrier[0], carrier[1]), carrier[1]), alpha)
        root_charts.append(
            tuple(
                sum_polynomials(
                    (variables[axis], scale(shear, direction[axis])),
                    3,
                )
                for axis in range(3)
            )
        )
    block = CellBlock(
        "native-plc-tetrahedra",
        "simplex:3",
        tetrahedra,
        global_ids=root_cell_ids,
    )
    root_mesh = CellMesh(
        vertices,
        (block,),
        vertex_global_ids=root_vertex_ids,
    )
    root_cell_id_tuple = tuple(int(value) for value in root_cell_ids.tolist())
    root_element = PerCellRestrictedCellGeometryElement(
        lagrange_element("tetrahedron", 1),
        "simplex:3",
        root_cell_id_tuple,
        root_cell_id_tuple,
        tuple(root_charts),
    )
    geometry = CellGeometrySpec(
        {"native-plc-tetrahedra": root_element},
        {"native-plc-tetrahedra": tetrahedra},
        vertices,
    )
    facets = np.asarray(
        (
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (2, 1, 4),
            (0, 4, 1),
            (0, 2, 4),
            (0, 1, 2),
        ),
        dtype=np.int64,
    )
    domain_region_sides = np.asarray(
        ((0, -1),) * 3 + ((1, -1),) * 3 + ((1, 0),), dtype=np.int64
    )
    reference_domain = PiecewiseLinearDomain(
        vertices,
        facets,
        domain_region_sides,
        ("upper", "lower"),
        source_id=source_id,
    )
    mapped_domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        geometry,
        np.asarray((0, 1), dtype=np.int64),
        source_id=source_id,
        source_revision=source_revision,
        coordinate_contract=_COORDINATES,
        reference_domain_vertex_root_ids=root_vertex_ids,
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        facets,
        np.arange(facets.shape[0], dtype=np.int64),
        np.asarray(
            ((-1, 0),) * 3 + ((-1, 1),) * 3 + ((0, 1),),
            dtype=np.int64,
        ),
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(str(value) for value in root_vertex_ids.tolist()),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(
            f"curved-bipyramid-facet-{row}" for row in range(facets.shape[0])
        ),
        facet_occurrence_paths=((),) * facets.shape[0],
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        geometry=geometry,
    )
    return NativePlcSource(
        complex_,
        source_id,
        source_revision,
        mapped_domain=mapped_domain,
        maximum_deviation=0.1,
    )


def _assert_curved_root_feature(source: NativePlcSource, /) -> None:
    domain = source.mapped_domain
    assert isinstance(domain, MappedReferenceDomain)
    mapped = domain.source_geometry.evaluate_exact(
        domain.reference_mesh,
        np.asarray((901,), dtype=np.int64),
        (((Fraction(5, 8), Fraction(1, 4), Fraction(0)),),),
    )
    assert mapped[0][0] == (
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(1, 512),
    )


def test_public_plc_balanced_grid_accepts_source_authored_curved_chart() -> None:
    source = _curved_public_plc_source()
    _assert_curved_root_feature(source)
    specification = _volume_request(
        source,
        family_policy=CellFamilyPolicy(required=("hexahedron",)),
        fill_strategy=VolumeFillStrategy.SIMPLEX,
        size=0.8,
        two_region_seed_points=((0.0, 0.2, 0.05), (0.0, 0.2, -0.05)),
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_balanced_grid_hex",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
            grid_schedule=NativeHexGridSchedule(
                "balanced_grid",
                minimum_scaled_jacobian=1.0e-10,
                minimum_mean_ratio=1.0e-10,
                maximum_aspect_ratio=1.0e10,
            ),
        )
    ).plan(source, specification, coordinate_contract=_COORDINATES)

    assert plan.route == "plc_balanced_grid_hex"
    result = plan.execute()

    _accepted_public_result(result, source)
    _assert_plc_facet_associations(result)
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert _region_materials(result) == {"upper-material", "lower-material"}
    assert any(
        positive is not None and negative is not None and positive != negative
        for association in result.associations
        for positive, negative in association.source_region_sides
    )
    assert result.geometry.source_map is not None
    assert result.geometry.source_map.source_id == source.source_id
    assert result.geometry.source_map.source_revision == source.source_revision


def test_public_plc_hex_dominant_composes_the_mapped_curved_source() -> None:
    source = _curved_public_plc_source()
    _assert_curved_root_feature(source)
    specification = _volume_request(
        source,
        family_policy=CellFamilyPolicy(
            required=("hexahedron",),
            allowed_transitions=("pyramid", "tetrahedron"),
            allow_mixed=True,
        ),
        fill_strategy=VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION,
        two_region_seed_points=((0.0, 0.2, 0.05), (0.0, 0.2, -0.05)),
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "plc_hex_dominant",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
            hex_core_fraction=0.5,
        )
    ).plan(source, specification, coordinate_contract=_COORDINATES)

    assert plan.route == "plc_hex_dominant"
    result = plan.execute()

    _accepted_public_result(result, source)
    _assert_plc_facet_associations(result)
    kinds = {block.cell_kind for block in result.mesh.blocks}
    assert "hexahedron" in kinds
    assert kinds <= {"hexahedron", "pyramid", "tetrahedron"}
    assert result.geometry.restriction_source is not None
    assert _region_materials(result) == {"upper-material", "lower-material"}
