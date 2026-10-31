from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.geometry._compartments import CompartmentMeshingSource
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._compartments import prepare_compartment_plc
from phydrax.meshing._volume_generation import NativeVolumeSchedule


def _manifest() -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-labels",
        checksum_algorithm="sha256",
        checksum="2" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )


def _compartment_source(
    *,
    inner_index: tuple[int, int, int] = (2, 2, 2),
) -> tuple[CompartmentMeshingSource, phx.SpatialCoordinateContract]:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="patient",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    values = np.ones((5, 5, 5), dtype=np.int16)
    values[inner_index] = 2
    asset = phx.imaging.MedicalImageAsset(
        "labels",
        "segmentation",
        values,
        affine,
        phx.imaging.ImageFieldSpec.named(
            "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
        ),
        phx.imaging.DeidentificationEvidence(
            "deid", "subject", "protocol", True, True, True
        ),
        (_manifest(),),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="synthetic-compartment-generator",
        ),
    )
    labels = phx.imaging.LabelVolume(
        asset,
        phx.imaging.LabelOntology(
            "ontology",
            "synthetic",
            "1",
            (
                phx.imaging.LabelDefinition(1, "outer-label", "Outer"),
                phx.imaging.LabelDefinition(2, "inner-label", "Inner"),
            ),
        ),
    )
    definitions = (
        phx.geometry.CompartmentDefinition(
            "outer", ("outer-label",), "fluid", allowed_neighbor_ids=("inner",)
        ),
        phx.geometry.CompartmentDefinition(
            "inner",
            ("inner-label",),
            "solid",
            containment_parent_id="outer",
            allowed_neighbor_ids=("outer",),
        ),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "outer-inner", "outer", "inner", "exchange"
    )
    complex_ = phx.imaging.build_compartment_complex(labels, definitions, (interface,))
    interfaces = phx.imaging.extract_compartment_surfaces(labels, complex_)

    lower, upper = -0.5, 4.5
    vertices = np.asarray(
        (
            (lower, lower, lower),
            (upper, lower, lower),
            (upper, upper, lower),
            (lower, upper, lower),
            (lower, lower, upper),
            (upper, lower, upper),
            (upper, upper, upper),
            (lower, upper, upper),
        ),
        dtype=np.float64,
    )
    quads = (
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 3, 2, 1),
        (4, 5, 6, 7),
    )
    triangles = np.asarray(
        tuple(
            triangle
            for quad in quads
            for triangle in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3]))
        ),
        dtype=np.int64,
    )
    outer = phx.geometry.SurfaceModel.from_triangles(
        vertices,
        triangles,
        phx.geometry.SurfaceMetadata(
            source_id="outer-surface",
            source_revision=labels.label_volume_id,
            coordinate_contract=contract,
            provenance=("synthetic",),
        ),
    )
    return CompartmentMeshingSource(labels, complex_, outer, interfaces), contract


def _native_compartment_mesh(
    source: CompartmentMeshingSource,
    contract: phx.SpatialCoordinateContract,
    *,
    material_overrides: dict[str, str] | None = None,
) -> phx.meshing.CellMeshingResult:
    boundary_scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    compartments = {
        value.compartment_id: value for value in source.compartments.compartments
    }
    overrides = {} if material_overrides is None else material_overrides
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        boundary_scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(boundary_scope, 1.5),),
        region_seeds=tuple(
            phx.meshing.RegionSeed(
                np.asarray((0.0, 0.0, 0.0) if name == "outer" else (2.0, 2.0, 2.0)),
                name,
                overrides.get(name, name),
                phx.meshing.RegionRole(compartment.material_role),
            )
            for name, compartment in compartments.items()
        ),
    )
    options = phx.meshing.NativeMeshingOptions(
        "image_material_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=8.0,
            minimum_dihedral_degrees=0.0,
        ),
    )
    return (
        phx.meshing.NativeMeshingProvider(options)
        .plan(source, specification, coordinate_contract=contract)
        .execute()
    )


def _network(contract: phx.SpatialCoordinateContract) -> Any:
    return phx.discretization.MetricNetworkPlan.from_arrays(
        np.asarray(((1.0, 1.0, 1.0), (3.0, 3.0, 3.0))),
        np.asarray(((0, 1),)),
        contract,
        areas=np.asarray((0.1,)),
        perimeters=np.asarray((0.5,)),
        root_vertex_ids=np.asarray((0,)),
        tip_vertex_ids=np.asarray((1,)),
    ).prepare()


@pytest.mark.meshcore
def test_native_compartment_case_preserves_regions_and_coupled_transport() -> None:
    source, contract = _compartment_source()
    result = _native_compartment_mesh(source, contract)
    network = _network(contract)
    case = phx.applications.neurofluid.NeurofluidCase(
        "synthetic-case", source, result, network
    )
    assert case.case_revision
    certification = result.certification
    assert certification is not None
    coverage = certification.coverage
    assert coverage is not None and coverage.status == "certified"
    domain = certification.inputs.domain
    assert domain is not None
    assert set(domain.region_ids) == {"outer", "inner"}
    assert source.source_revision == source.labels.label_volume_id
    assert source.compartments.observed_adjacencies == (("inner", "outer"),)
    assert source.compartments.interfaces[0].interface_role == "exchange"
    assert result.boundary is not None
    outer = source.outer_surface
    assert result.boundary.metadata.source_id == outer.metadata.source_id
    assert result.boundary.metadata.source_revision == outer.metadata.source_revision
    assert outer.model_id in result.boundary.metadata.provenance
    boundary_points = np.asarray(result.boundary.mesh.coordinates)
    assert np.all(np.any(np.abs(np.abs(boundary_points - 2.0) - 2.5) == 0.0, axis=1))
    boundary_triangles = boundary_points[
        np.asarray(result.boundary.mesh.blocks[0].vertices)
    ]
    boundary_vectors = np.cross(
        boundary_triangles[:, 1] - boundary_triangles[:, 0],
        boundary_triangles[:, 2] - boundary_triangles[:, 0],
    )
    np.testing.assert_allclose(boundary_vectors.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        0.5 * np.linalg.norm(boundary_vectors, axis=1).sum(), 150.0, rtol=1e-14
    )
    assert (
        np.einsum("ij,ij->i", boundary_vectors, boundary_triangles.mean(axis=1) - 2.0)
        > 0.0
    ).all()
    zones_by_material = {zone.material_id: zone for zone in result.zones}
    assert set(zones_by_material) == {"outer", "inner"}
    assert zones_by_material["outer"].region_role is phx.meshing.RegionRole.FLUID
    assert zones_by_material["inner"].region_role is phx.meshing.RegionRole.SOLID
    interface = next(patch for patch in result.patches if patch.name == "outer-inner")
    preparation = prepare_compartment_plc(source)
    interface_row = preparation.facet_interface_ids.index("outer-inner")
    facet_id = preparation.complex.facet_ids[interface_row]
    source_sides = preparation.facet_region_ids[interface_row]
    assert set(source_sides) == {"outer", "inner"}
    assert interface.source_adjacent_region_ids == source_sides
    face_association = next(
        association
        for association in result.associations
        if association.target_entity_set_id == result.mesh.entity_set(2).entity_set_id
    )
    source_face_rows = tuple(
        row
        for row, (source_id, role) in enumerate(
            zip(
                face_association.source_entity_ids,
                face_association.source_roles,
                strict=True,
            )
        )
        if source_id == facet_id and role is GeometrySourceEntityRole.FACET
    )
    assert source_face_rows
    assert {face_association.source_region_sides[row] for row in source_face_rows} == {
        source_sides
    }
    assert set(interface.adjacent_zone_ids) == {
        zones_by_material["outer"].zone_id,
        zones_by_material["inner"].zone_id,
    }

    mesh = result.mesh
    cell_association = next(
        value
        for value in result.associations
        if value.target_entity_set_id == mesh.entity_set(3).entity_set_id
    )
    assert cell_association.source_id == source.source_id
    assert cell_association.source_revision == source.source_revision
    assert set(cell_association.source_entity_ids) == {"outer", "inner"}
    assert all(
        role is not None and role.value == "region"
        for role in cell_association.source_roles
    )
    cell_count = mesh.entity_set(3).count
    connectivity = mesh.connectivity
    assert isinstance(connectivity, (TetrahedralConnectivity, SimplicialConnectivity))
    boundary_count = np.count_nonzero(np.asarray(connectivity.boundary_faces))
    parameters = phx.applications.neurofluid.NeurofluidTransportParameters(
        porosity=np.ones(cell_count),
        bulk_diffusivity=np.zeros(cell_count),
        bulk_velocity=np.zeros((cell_count, 3)),
        bulk_boundary_volume_flux=np.zeros(boundary_count),
        bulk_boundary_inflow_concentration=np.zeros(boundary_count),
        bulk_removal_rate=np.zeros(cell_count),
        network_diffusivity=np.zeros(1),
        units=phx.applications.neurofluid.NeurofluidTransportUnits(
            phx.units.MILLIMETER, phx.units.SECOND, phx.units.ONE
        ),
        network_volume_flow=np.zeros(1),
        exchange_coefficients=np.zeros(2),
        averaging_radius=0.05,
        reservoir_volumes=np.ones(1),
        reservoir_coefficients=np.zeros(1),
    )
    runtime = phx.applications.neurofluid.NeurofluidTransportPlan(
        case, parameters
    ).prepare()
    state = phx.equations.MixedDimensionalTransportState(
        jnp.ones(cell_count, dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
        jnp.ones(1, dtype=jnp.float64),
    )
    assert bool(runtime.step_backward_euler(state, 0.1).accepted)

    wrong_source, _ = _compartment_source(inner_index=(1, 2, 2))
    with pytest.raises(ValueError, match="outer surface|source revision"):
        phx.applications.neurofluid.NeurofluidCase(
            "wrong-source", wrong_source, result, network
        )

    wrong_material_result = _native_compartment_mesh(
        source, contract, material_overrides={"inner": "outer"}
    )
    with pytest.raises(ValueError, match="source-region cell associations"):
        phx.applications.neurofluid.NeurofluidCase(
            "wrong-material", source, wrong_material_result, network
        )

    foreign_contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="other-patient",
    )
    with pytest.raises(ValueError, match="coordinate contracts"):
        phx.applications.neurofluid.NeurofluidCase(
            "wrong-frame", source, result, _network(foreign_contract)
        )


@pytest.mark.parametrize("inner_index", [(2, 2, 2), (0, 1, 4)])
def test_compartment_plc_facet_rows_name_the_right_hand_normal_side_first(
    inner_index: tuple[int, int, int],
) -> None:
    source, _ = _compartment_source(inner_index=inner_index)
    complex_ = prepare_compartment_plc(source).complex
    vertices = np.asarray(complex_.vertices)
    offsets = np.asarray(complex_.polygon_offsets)
    loops = np.asarray(complex_.polygon_vertices)

    def region_at(point: np.ndarray) -> str | None:
        voxel = tuple(int(value) for value in np.rint(point))
        if not all(0 <= value < 5 for value in voxel):
            return None
        return "inner" if voxel == inner_index else "outer"

    for row, facet in enumerate(np.asarray(complex_.polygon_facets).tolist()):
        loop = vertices[loops[offsets[row] : offsets[row + 1]]]
        normal = np.cross(loop[1] - loop[0], loop[2] - loop[0])
        normal /= np.linalg.norm(normal)
        center = loop.mean(axis=0)
        positive, negative = (
            None if region < 0 else complex_.region_ids[region]
            for region in np.asarray(complex_.facet_regions)[facet].tolist()
        )
        assert region_at(center + 0.25 * normal) == positive
        assert region_at(center - 0.25 * normal) == negative


def test_optional_compartment_interface_requires_declared_endpoints() -> None:
    compartments = (
        phx.geometry.CompartmentDefinition("inside", ("inside",), "material"),
        phx.geometry.CompartmentDefinition("outside", ("outside",), "material"),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "invalid-optional",
        "inside",
        "undeclared",
        "diagnostic",
        required=False,
    )

    with pytest.raises(
        ValueError, match="Interface endpoints must identify declared compartments"
    ):
        phx.geometry.CompartmentComplex(
            "revision",
            compartments,
            (interface,),
            (("inside", 1.0), ("outside", 1.0)),
            (),
        )
