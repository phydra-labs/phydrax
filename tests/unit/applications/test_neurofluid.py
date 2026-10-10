import sys
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _manifest() -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-reference",
        checksum_algorithm="sha256",
        checksum="1" * 64,
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


def _image(
    values: Any,
    unit: Any,
    asset_id: Any,
    *,
    value_kind: phx.measurement.ValueKind = phx.measurement.ValueKind.REAL_SCALAR,
) -> Any:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="patient",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxel", contract, phx.imaging.ImageAxisConvention.LPS
    )
    return phx.imaging.MedicalImageAsset(
        asset_id,
        "t1-map",
        np.asarray(values),
        affine,
        phx.imaging.ImageFieldSpec.named("t1", unit, value_kind),
        phx.imaging.DeidentificationEvidence(
            "deid", "subject", "protocol", True, True, True
        ),
        (_manifest(),),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="synthetic-neurofluid-generator",
        ),
    )


def test_neurofluid_scenario_1() -> None:
    baseline = _image(np.full((2, 2, 2), 2.0), phx.units.SECOND, "baseline")
    contrast_values = np.full((2, 2, 2), 1.0)
    contrast_values[0, 0, 0] = 4.0
    contrast = _image(contrast_values, phx.units.SECOND, "contrast")
    relaxivity_unit = phx.units.derived_unit("1/s", ((phx.units.SECOND, -1),))
    result = phx.applications.neurofluid.TracerRelaxivityCalibration(
        0.5, relaxivity_unit, phx.units.ONE, phx.units.SECOND
    ).evaluate(baseline, contrast)
    np.testing.assert_allclose(result.values[1, 1, 1], 1.0)
    assert result.values[0, 0, 0] < 0.0
    assert bool(result.evidence.successful)
    assert not bool(result.evidence.nonnegative_state_candidate)
    units = phx.applications.neurofluid.NeurofluidTransportUnits(
        phx.units.MILLIMETER, phx.units.SECOND, phx.units.MILLIMOLAR
    )
    diffusivity = phx.units.derived_unit(
        "mm2/s", ((phx.units.MILLIMETER, 2), (phx.units.SECOND, -1))
    )
    volume_flow = phx.units.derived_unit(
        "mm3/s", ((phx.units.MILLIMETER, 3), (phx.units.SECOND, -1))
    )
    assert units.diffusivity_unit.dimension == diffusivity.dimension
    assert units.volume_flow_unit.dimension == volume_flow.dimension
    schedule = phx.applications.neurofluid.FlowTransportSchedule(
        np.asarray((0.0, 0.5, 1.0)),
        np.asarray(((1.0,), (2.0,), (1.0,))),
        1.0,
    )
    mean, evidence = schedule.mean()
    np.testing.assert_allclose(mean, (1.5,))
    assert bool(evidence.successful)
    _, partial = phx.applications.neurofluid.FlowTransportSchedule(
        np.asarray((0.0, 0.5, 1.0)),
        np.asarray(((1.0,), (2.0,), (1.0,))),
        2.0,
    ).mean()
    assert not bool(partial.successful)


def test_neurofluid_scenario_2() -> None:
    operator = phx.spatial_sampling.ObservationSamplingPlan(
        np.asarray(((0,), (1,))),
        np.ones((2, 1)),
        (2,),
        operator_kind="identity",
        source_geometry_id="state",
        require_complete_coverage=True,
    ).prepare()
    observation = phx.applications.neurofluid.ImageSpaceObservation(
        operator,
        np.asarray((3.0, -1.0)),
        np.asarray((True, True)),
        np.asarray((0.1, 0.1)),
        observation_id="synthetic-image",
    )
    schema = phx.applications.neurofluid.NeurofluidParameterSchema(
        ("sum", "difference"),
        np.asarray((-10.0, -10.0)),
        np.asarray((10.0, 10.0)),
        np.asarray((0.0, 0.0)),
    )
    forward = lambda design: jnp.asarray((design[0] + design[1], design[0] - design[1]))
    inverse = phx.applications.neurofluid.NeurofluidInverseProblem(
        schema, observation, forward
    )
    report = inverse.identifiability(np.asarray((1.0, 2.0)))
    assert int(report.numerical_rank) == 2
    assert bool(report.full_rank)
    problem = inverse.state_design_problem()
    state = forward(jnp.asarray((1.0, 2.0)))
    np.testing.assert_allclose(problem.residual(state, jnp.asarray((1.0, 2.0))), 0.0)
    np.testing.assert_allclose(problem.objective(state, jnp.asarray((1.0, 2.0))), 0.0)
    operator = phx.spatial_sampling.ObservationSamplingPlan(
        np.asarray(((0,), (1,))),
        np.ones((2, 1)),
        (2,),
        operator_kind="identity",
        source_geometry_id="rank-deficient-state",
        require_complete_coverage=True,
    ).prepare()
    observation = phx.applications.neurofluid.ImageSpaceObservation(
        operator,
        np.asarray((1.0, 1.0)),
        np.asarray((True, True)),
        np.asarray((0.1, 0.1)),
        observation_id="rank-deficient-image",
    )
    schema = phx.applications.neurofluid.NeurofluidParameterSchema(
        ("first", "second"),
        np.asarray((-10.0, -10.0)),
        np.asarray((10.0, 10.0)),
        np.asarray((0.0, 0.0)),
    )
    inverse = phx.applications.neurofluid.NeurofluidInverseProblem(
        schema,
        observation,
        lambda design: jnp.asarray((design[0] + design[1],) * 2),
    )

    report = inverse.identifiability(np.asarray((1.0, 2.0)))
    assert not bool(report.full_rank)
    assert jnp.isinf(report.condition_number)
    first = phx.applications.neurofluid.NeurofluidPipelineStage(
        "ingest", "image-ingest", ("source",), ("image",)
    )
    second = phx.applications.neurofluid.NeurofluidPipelineStage(
        "mesh", "compartment-mesh", ("image",), ("mesh",), ("ingest",)
    )
    manifest = phx.applications.neurofluid.NeurofluidPipelineManifest(
        "case-revision", (first, second)
    )
    assert manifest.manifest_id


def test_external_tool_provider_records_real_output(tmp_path: Any) -> None:
    provider = phx.imaging.MedicalToolProvider(
        "python", sys.executable, ("--version",), "PSF-2.0", "test-runtime"
    )
    result = provider.execute(
        ("-c", "from pathlib import Path; Path('output.txt').write_text('complete')"),
        ("output.txt",),
        (_manifest(),),
        working_directory=tmp_path,
        output_license_id="synthetic",
        timeout_seconds=30.0,
    )
    assert result.artifacts[0].status == "complete"
    assert result.output_paths[0].read_text() == "complete"
    assert result.stdout == result.stderr == ""


@pytest.fixture(scope="module")
def native_case() -> phx.applications.neurofluid.NeurofluidCase:
    values = np.ones((2, 2, 2), dtype=np.int16)
    values[1] = 2
    labels = phx.imaging.LabelVolume(
        _image(
            values,
            phx.units.ONE,
            "native-labels",
            value_kind=phx.measurement.ValueKind.CATEGORICAL,
        ),
        phx.imaging.LabelOntology(
            "native-labels",
            "synthetic",
            "1",
            (
                phx.imaging.LabelDefinition(1, "first-label", "First"),
                phx.imaging.LabelDefinition(2, "second-label", "Second"),
            ),
        ),
    )
    compartments = phx.imaging.build_compartment_complex(
        labels,
        (
            phx.geometry.CompartmentDefinition(
                "first", ("first-label",), "material", allowed_neighbor_ids=("second",)
            ),
            phx.geometry.CompartmentDefinition(
                "second", ("second-label",), "material", allowed_neighbor_ids=("first",)
            ),
        ),
        (
            phx.geometry.CompartmentInterfaceDefinition(
                "first-second", "first", "second", "exchange"
            ),
        ),
    )
    contract = labels.asset.spatial_affine.coordinate_contract
    outer = phx.geometry.SurfaceModel.from_triangles(
        np.asarray(
            (
                (-0.5, -0.5, -0.5),
                (1.5, -0.5, -0.5),
                (1.5, 1.5, -0.5),
                (-0.5, 1.5, -0.5),
                (-0.5, -0.5, 1.5),
                (1.5, -0.5, 1.5),
                (1.5, 1.5, 1.5),
                (-0.5, 1.5, 1.5),
            ),
            dtype=np.float64,
        ),
        np.asarray(
            (
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (3, 7, 6),
                (3, 6, 2),
                (0, 4, 7),
                (0, 7, 3),
                (1, 2, 6),
                (1, 6, 5),
            ),
            dtype=np.int64,
        ),
        phx.geometry.SurfaceMetadata(
            source_id="native-outer",
            source_revision=labels.label_volume_id,
            coordinate_contract=contract,
            provenance=("synthetic",),
        ),
    )
    source = phx.geometry.CompartmentMeshingSource(
        labels,
        compartments,
        outer,
        phx.imaging.extract_compartment_surfaces(labels, compartments),
    )
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        f"{source.source_id}:boundary",
        np.asarray((0,), dtype=np.int64),
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 1.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
        )
        .plan(source, request, coordinate_contract=contract)
        .execute()
    )
    network = phx.discretization.MetricNetworkPlan.from_arrays(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1),), dtype=np.int64),
        contract,
        areas=np.asarray((0.01,), dtype=np.float64),
        perimeters=np.asarray((0.2,), dtype=np.float64),
        root_vertex_ids=np.asarray((0,), dtype=np.int64),
        tip_vertex_ids=np.asarray((1,), dtype=np.int64),
    ).prepare()
    return phx.applications.neurofluid.NeurofluidCase(
        "native-case", labels, compartments, result, network
    )


def test_native_compartment_transport_preserves_region_and_closed_inventory(
    native_case: phx.applications.neurofluid.NeurofluidCase,
) -> None:
    result = native_case.bulk_mesh
    evidence = result.region_evidence
    assert evidence is not None
    assert evidence.source_revision == native_case.segmentation.label_volume_id
    assert evidence.source_complex_id == native_case.compartments.complex_id
    assert set(evidence.cell_region_ids) == {"first", "second"}
    assert evidence.adjacency_pairs == (("first", "second"),)
    cell_count = result.mesh.entity_set(3).count
    boundary_count = np.count_nonzero(
        np.asarray(result.mesh.entity_set(2).subset("boundary").mask)
    )
    parameters = phx.applications.neurofluid.NeurofluidTransportParameters(
        porosity=np.full(cell_count, 0.8, dtype=np.float64),
        bulk_diffusivity=np.full(cell_count, 0.01, dtype=np.float64),
        bulk_velocity=np.zeros((cell_count, 3), dtype=np.float64),
        bulk_boundary_volume_flux=np.zeros(boundary_count, dtype=np.float64),
        bulk_boundary_inflow_concentration=np.zeros(boundary_count, dtype=np.float64),
        bulk_removal_rate=np.zeros(cell_count, dtype=np.float64),
        network_diffusivity=np.asarray((0.01,), dtype=np.float64),
        units=phx.applications.neurofluid.NeurofluidTransportUnits(
            phx.units.MILLIMETER, phx.units.SECOND, phx.units.MILLIMOLAR
        ),
        network_volume_flow=np.zeros(1, dtype=np.float64),
        exchange_coefficients=np.full(2, 0.5, dtype=np.float64),
        averaging_radius=0.01,
        reservoir_volumes=np.asarray((0.003,), dtype=np.float64),
        reservoir_coefficients=np.asarray((0.2,), dtype=np.float64),
    )
    runtime = phx.applications.neurofluid.NeurofluidTransportPlan(
        native_case, parameters
    ).prepare()
    state = phx.equations.MixedDimensionalTransportState(
        jnp.full(cell_count, 3.0, dtype=jnp.float64),
        jnp.asarray((1.0, 1.5), dtype=jnp.float64),
        jnp.asarray((0.5,), dtype=jnp.float64),
    )
    initial = runtime.ledger(state)
    step = runtime.step_backward_euler(
        state,
        0.05,
        policy=phx.linalg.LinearSolvePolicy(
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-12, absolute=1.0e-14, max_steps=2048
            )
        ),
    )
    assert bool(step.accepted)
    final = runtime.ledger(step.state)
    assert bool(final.successful)
    np.testing.assert_allclose(
        final.total_mass, initial.total_mass, rtol=0.0, atol=1.0e-9
    )
    np.testing.assert_allclose(final.exchange_defect, 0.0, rtol=0.0, atol=1.0e-12)
    assert np.sum(np.asarray(runtime.bulk.mass * step.state.bulk)) < np.sum(
        np.asarray(runtime.bulk.mass * state.bulk)
    )
    assert step.state.reservoirs[0] > state.reservoirs[0]
    region_indices = {"first": 0, "second": 1}
    diagnostics = phx.applications.neurofluid.neurofluid_diagnostics(
        runtime,
        step.state,
        jnp.asarray(
            tuple(region_indices[region] for region in evidence.cell_region_ids),
            dtype=jnp.int32,
        ),
        ("first", "second"),
    )
    assert bool(diagnostics.successful)
    np.testing.assert_allclose(
        np.sum(np.asarray(diagnostics.compartment_mass)),
        np.sum(np.asarray(runtime.bulk.mass * step.state.bulk)),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert parameters.units.length_unit == result.coordinate_contract.length_unit


def test_neurofluid_rejects_mesh_of_different_source_revision(
    native_case: phx.applications.neurofluid.NeurofluidCase,
) -> None:
    values = np.ones((2, 2, 2), dtype=np.int16)
    values[0] = 2
    labels = phx.imaging.LabelVolume(
        _image(
            values,
            phx.units.ONE,
            "different-label-revision",
            value_kind=phx.measurement.ValueKind.CATEGORICAL,
        ),
        native_case.segmentation.ontology,
    )
    compartments = phx.imaging.build_compartment_complex(
        labels, native_case.compartments.compartments, native_case.compartments.interfaces
    )
    with pytest.raises(ValueError):
        replace(native_case, segmentation=labels, compartments=compartments)


def test_neurofluid_rejects_different_complex_with_same_region_ids(
    native_case: phx.applications.neurofluid.NeurofluidCase,
) -> None:
    first, second = native_case.compartments.compartments
    compartments = replace(
        native_case.compartments,
        compartments=(replace(first, material_role="changed-material"), second),
    )
    assert compartments.source_revision == native_case.segmentation.label_volume_id
    assert compartments.complex_id != native_case.compartments.complex_id
    with pytest.raises(ValueError):
        replace(native_case, compartments=compartments)


def test_neurofluid_requires_region_evidence_even_with_matching_zone_names(
    native_case: phx.applications.neurofluid.NeurofluidCase,
) -> None:
    result = native_case.bulk_mesh
    unbound = phx.meshing.CellMeshingResult(
        result.mesh,
        result.geometry,
        result.coordinate_contract,
        result.audit,
        result.quality,
        result.compliance,
        result.trace,
        result.provider,
        result.runtime,
        result.derivative_mode,
        result.provenance,
        boundary=result.boundary,
        patches=result.patches,
        zones=result.zones,
        labels=result.labels,
        attributes=result.attributes,
        associations=result.associations,
        adapter_reports=result.adapter_reports,
        certification=result.certification,
    )
    with pytest.raises(ValueError, match="source-region evidence"):
        replace(native_case, bulk_mesh=unbound)
