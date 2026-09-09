import manifold3d
import numpy as np
import pytest

import phydrax as phx


def _manifest():
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


@pytest.mark.meshing_ftetwild
def test_real_ftetwild_compartment_mesh_preserves_nested_zone_interface():
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="patient",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    values = np.ones((5, 5, 5), dtype=np.int16)
    values[2, 2, 2] = 2
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
        _manifest(),
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
    compartments = (
        phx.geometry.CompartmentDefinition(
            "outer", ("outer-label",), "material", allowed_neighbor_ids=("inner",)
        ),
        phx.geometry.CompartmentDefinition(
            "inner",
            ("inner-label",),
            "material",
            containment_parent_id="outer",
            allowed_neighbor_ids=("outer",),
        ),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "outer-inner", "outer", "inner", "exchange"
    )
    complex_ = phx.imaging.build_compartment_complex(labels, compartments, (interface,))
    interfaces = phx.imaging.extract_compartment_surfaces(labels, complex_)
    arrays = (
        manifold3d.Manifold.cube((5.0, 5.0, 5.0))
        .translate((-0.5, -0.5, -0.5))
        .to_mesh64()
    )
    outer = phx.geometry.SurfaceModel.from_triangles(
        arrays.vert_properties[:, :3],
        arrays.tri_verts,
        phx.geometry.SurfaceMetadata(
            source_id="outer-surface",
            source_revision="0",
            coordinate_contract=contract,
            provenance=("synthetic",),
        ),
    )
    result = phx.meshing.FTetWildCompartmentProvider().execute(
        phx.meshing.CompartmentMeshingSpec(
            labels,
            complex_,
            outer,
            interfaces,
            1.5,
            options=phx.meshing.FTetWildOptions(
                envelope_distance=0.05, maximum_iterations=20, maximum_threads=1
            ),
        )
    )
    assert {zone.name for zone in result.zones} == {"outer", "inner"}
    assert {patch.name for patch in result.interfaces} == {"outer-inner"}
    assert result.adjacency_pairs == (("inner", "outer"),)
    assert result.result.audit.passed
    assert result.result.provider.name == "ftetwild"
    assert result.result.runtime.actual_version.startswith("wildmeshing ")
    network = phx.discretization.MetricNetworkPlan.from_arrays(
        np.asarray(((1.0, 1.0, 1.0), (3.0, 3.0, 3.0))),
        np.asarray(((0, 1),)),
        contract,
        areas=np.asarray((0.1,)),
        perimeters=np.asarray((0.5,)),
        root_vertex_ids=np.asarray((0,)),
        tip_vertex_ids=np.asarray((1,)),
    ).prepare()
    case = phx.applications.neurofluid.NeurofluidCase(
        "synthetic-case", labels, complex_, result, network
    )
    assert case.case_revision
    mesh = result.result.mesh
    cell_count = mesh.entity_set(3).count
    boundary_count = int(np.count_nonzero(np.asarray(mesh.connectivity.boundary_faces)))
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
        np.ones(cell_count), np.ones(2), np.ones(1)
    )
    step = runtime.step_backward_euler(state, 0.1)
    assert bool(step.accepted)
