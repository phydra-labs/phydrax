import sys
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


# Empirical `test_native_compartment_transport_preserves_region_and_closed_inventory`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CLOSED_TRANSPORT_TOTAL_MASS_ATOL = 1.0e-9
# Empirical `test_native_compartment_transport_preserves_region_and_closed_inventory`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_PAIRED_TRANSPORT_EXCHANGE_DEFECT_ATOL = 1.0e-12
# Empirical `test_native_compartment_transport_preserves_region_and_closed_inventory`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_COMPARTMENT_MASS_REDUCTION_ATOL = 1.0e-12


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
    # Native two-label/one-network-edge/one-reservoir acceptance fixture from
    # 03ef670bc, this node: use its 1e-12 relative / 1e-14 absolute Krylov
    # residual target for the 0.05-second closed backward-Euler solve.
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
    # Retain this fixture's 03ef670bc absolute mass ceiling: 1e-9 bounds the
    # difference of bulk, network and reservoir weighted inventories after that
    # solve (initial concentrations 3, 1/1.5 and 0.5); rtol=0 excludes scaling it.
    np.testing.assert_allclose(
        final.total_mass,
        initial.total_mass,
        rtol=0.0,
        atol=_CLOSED_TRANSPORT_TOTAL_MASS_ATOL,
    )
    # The 03ef670bc 1e-12 exchange ceiling bounds cancellation of the paired
    # bulk/network and network/reservoir flux reductions, not solver convergence.
    np.testing.assert_allclose(
        final.exchange_defect,
        0.0,
        rtol=0.0,
        atol=_PAIRED_TRANSPORT_EXCHANGE_DEFECT_ATOL,
    )
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
    # The 03ef670bc 1e-12 compartment ceiling compares two summation groupings
    # of the same mass*concentration terms, with no time-stepping error.
    np.testing.assert_allclose(
        np.sum(np.asarray(diagnostics.compartment_mass)),
        np.sum(np.asarray(runtime.bulk.mass * step.state.bulk)),
        rtol=0.0,
        atol=_COMPARTMENT_MASS_REDUCTION_ATOL,
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
    with pytest.raises(
        ValueError, match=r"RegionMeshingEvidence\.source_complex_id must bind"
    ):
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
    with pytest.raises(
        ValueError, match=r"RegionMeshingEvidence\.source_complex_id must bind"
    ):
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
    with pytest.raises(ValueError, match=r"NeurofluidCase\.bulk_mesh\.region_evidence"):
        replace(native_case, bulk_mesh=unbound)


def test_synthetic_neurofluid_qualification_solves_unit_normal_flux() -> None:
    from tools import neurofluid_qualification as qualification

    record: Any = qualification.qualify()

    assert record["scenario"] == "synthetic"
    assert record["successful"]
    assert record["adjacency_successful"]
    assert record["hdiv_preparation_successful"]
    assert record["compartments"] == 2
    assert record["interfaces"] == 1
    assert record["bdm2_velocity_dofs"] == 30
    assert record["dg1_pressure_dofs"] == 4
    for metric in (
        "rt0_flux_error",
        "normal_flow_error",
        "resistance_power_error",
        "normal_constraint_error",
    ):
        assert record[metric] <= 1.0e-12
    lift = record["normal_flux_lift"]
    assert lift["status"] == int(phx.linalg.LinearSolveStatus.SUCCESS)
    assert lift["successful"]
    assert lift["rank"] == 1
    assert lift["residual_norm"] <= 1.0e-12
    assert lift["normal_residual_norm"] <= 1.0e-12


@pytest.mark.parametrize(
    ("scale", "status", "failed_contracts"),
    (
        (
            2.0,
            phx.linalg.LinearSolveStatus.SUCCESS,
            ("normal_flow", "resistance_power", "normal_constraint"),
        ),
        (
            1.0,
            phx.linalg.LinearSolveStatus.RESIDUAL_TOO_LARGE,
            ("normal_flux_lift",),
        ),
        (
            np.nan,
            phx.linalg.LinearSolveStatus.NONFINITE_OUTPUT,
            (
                "normal_flux_lift",
                "normal_flow",
                "resistance_power",
                "normal_constraint",
            ),
        ),
    ),
)
def test_synthetic_neurofluid_qualification_preserves_failed_numerical_evidence(
    monkeypatch: pytest.MonkeyPatch,
    scale: float,
    status: phx.linalg.LinearSolveStatus,
    failed_contracts: tuple[str, ...],
) -> None:
    from tools import neurofluid_qualification as qualification

    native_factorize = phx.linalg.factorize

    def factorize(*args: Any, **kwargs: Any) -> Any:
        prepared = native_factorize(*args, **kwargs)

        def solve(target: Any) -> Any:
            result = prepared.solve(target)
            return eqx.tree_at(
                lambda value: (value.value, value.status),
                result,
                (scale * result.value, jnp.asarray(int(status), dtype=jnp.int32)),
            )

        return SimpleNamespace(solve=solve)

    monkeypatch.setattr(qualification.phx.linalg, "factorize", factorize)
    record: Any = qualification.qualify()

    assert not record["successful"]
    assert record["status"] == "failed"
    assert record["failure"]["phase"] == "synthetic-invariants"
    assert record["failure"]["failed_contracts"] == list(failed_contracts)
    assert record["normal_flux_lift"]["status"] == int(status)
    for metric in (
        "normal_flow_error",
        "resistance_power_error",
        "normal_constraint_error",
    ):
        if np.isnan(scale):
            assert np.isnan(record[metric])
        elif scale == 1.0:
            assert record[metric] <= 1.0e-12
        else:
            assert record[metric] > 1.0e-12


def test_junction_voxel_reference_measures_the_sheared_label_complex() -> None:
    from tools import neurofluid_qualification as qualification

    source, _ = qualification._junction_source(4)
    reference = qualification._voxel_reference(source)
    linear = qualification._JUNCTION_AFFINE[:3, :3]
    offset = qualification._JUNCTION_AFFINE[:3, 3]
    # Face areas normal to each index axis and edge lengths along it.
    face = [
        float(np.linalg.norm(np.cross(linear[:, first], linear[:, second])))
        for first, second in ((1, 2), (0, 2), (0, 1))
    ]
    edge = np.linalg.norm(linear, axis=0)
    names = reference.compartments
    assert names == ("alpha", "beta", "cap", "delta", "gamma", "tiny")
    counts = dict(zip(names, reference.counts.tolist(), strict=True))
    assert counts == {
        "alpha": 8,
        "beta": 8,
        "gamma": 8,
        "delta": 8,
        "cap": 4 * 4 * 3 - 1,
        "tiny": 1,
    }
    np.testing.assert_allclose(
        reference.volumes, reference.counts * abs(np.linalg.det(linear)), rtol=1e-14
    )
    areas = {
        (names[low], names[high]): value
        for (low, high), value in reference.interface_areas.items()
    }
    expected = {
        ("alpha", "beta"): 4 * face[1],
        ("alpha", "gamma"): 4 * face[0],
        ("beta", "delta"): 4 * face[0],
        ("delta", "gamma"): 4 * face[1],
        ("alpha", "cap"): 4 * face[2],
        ("beta", "cap"): 4 * face[2],
        ("cap", "delta"): 4 * face[2],
        ("cap", "gamma"): 4 * face[2],
        ("cap", "tiny"): 2 * sum(face),
    }
    # Diagonal quadrants share only the quadruple line, never a face.
    assert set(areas) == set(expected) == set(qualification._JUNCTION_INTERFACES)
    for pair, value in expected.items():
        np.testing.assert_allclose(areas[pair], value, rtol=1e-14)
    # Two quadruple-line edges along index axis 2, four triple-line edges
    # along each in-plane axis where two quadrants meet the cap.
    np.testing.assert_allclose(
        reference.junction_length, 2 * edge[2] + 4 * edge[0] + 4 * edge[1], rtol=1e-14
    )
    np.testing.assert_allclose(reference.quadruple_line_length, 2 * edge[2], rtol=1e-14)
    # The quadruple line is straight: only its boundary endpoint and the
    # quintuple point under the cap are zero-dimensional junction points.
    np.testing.assert_allclose(
        reference.quadruple_points,
        (np.asarray(((1.5, 1.5, -0.5), (1.5, 1.5, 1.5))) @ linear.T) + offset,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        reference.quintuple_points,
        (np.asarray(((1.5, 1.5, 1.5),)) @ linear.T) + offset,
        rtol=1e-14,
    )


def _kuhn_mesh(
    voxels: np.ndarray, linear: np.ndarray, offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conforming six-tetrahedron Kuhn split of every voxel under an affine map."""
    from itertools import permutations, product

    shape = voxels.shape

    def corner(index: list[int]) -> int:
        return (index[0] * (shape[1] + 1) + index[1]) * (shape[2] + 1) + index[2]

    lattice = np.asarray(
        list(product(*(range(extent + 1) for extent in shape))), dtype=np.float64
    )
    points = (lattice - 0.5) @ linear.T + offset
    cells, regions = [], []
    for voxel in product(*(range(extent) for extent in shape)):
        for order in permutations(range(3)):
            current = list(voxel)
            cell = [corner(current)]
            for axis in order:
                current[axis] += 1
                cell.append(corner(current))
            cells.append(cell)
            regions.append(int(voxels[voxel]))
    return points, np.asarray(cells, dtype=np.int64), np.asarray(regions, dtype=np.int64)


def test_independent_bulk_assembly_conserves_declared_transport_terms() -> None:
    from tools import neurofluid_qualification as qualification

    linear = qualification._JUNCTION_AFFINE[:3, :3]
    offset = qualification._JUNCTION_AFFINE[:3, 3]
    # Materials 0 and 2 are diagonal neighbours: they share an edge, no face.
    voxels = np.asarray((((0,), (1,)), ((1,), (2,))), dtype=np.int64)
    points, cells, regions = _kuhn_mesh(voxels, linear, offset)
    porosity = np.asarray((0.8, 0.6, 0.9))[regions]
    diffusivity = np.asarray((0.01, 0.03, 0.02))[regions]
    removal_rate = np.asarray((0.0, 0.1, 0.0))[regions]
    velocity = np.asarray(qualification._JUNCTION_VELOCITY)
    inflow = 0.8
    assembly = qualification._assemble_bulk(
        points, cells, regions, porosity, diffusivity, removal_rate, velocity, inflow
    )
    reference = qualification._voxel_complex(("a", "b", "c"), voxels, linear, offset)
    np.testing.assert_allclose(
        np.bincount(regions, weights=assembly.volumes), reference.volumes, rtol=1e-13
    )
    np.testing.assert_allclose(assembly.mass, porosity * assembly.volumes)
    assert set(assembly.pair_area) == set(reference.interface_areas) == {(0, 1), (1, 2)}
    for key, value in reference.interface_areas.items():
        np.testing.assert_allclose(assembly.pair_area[key], value, rtol=1e-13)
    # Uniform velocity is divergence free cell by cell and across the boundary.
    assert np.max(np.abs(assembly.divergence)) <= 1e-15
    assert abs(np.sum(assembly.boundary_flux)) <= 1e-15
    assert np.any(assembly.boundary_flux < 0.0) and np.any(assembly.boundary_flux > 0.0)
    np.testing.assert_allclose(
        np.sum(assembly.inflow), -inflow * np.sum(np.minimum(assembly.boundary_flux, 0.0))
    )
    # Interior diffusion and advection cancel in the inventory: only boundary
    # outflow and removal leave.
    np.testing.assert_allclose(
        np.sum(assembly.matrix, axis=0),
        assembly.outflow + assembly.removal,
        atol=1e-15,
    )
    np.testing.assert_allclose(assembly.removal, assembly.mass * removal_rate)
    # Without removal, a state equal to the inflow concentration is steady.
    steady = qualification._assemble_bulk(
        points, cells, regions, porosity, diffusivity, 0.0 * removal_rate, velocity, inflow
    )
    assert (
        np.max(np.abs(steady.matrix @ np.full(len(cells), inflow) + steady.forcing))
        <= 1e-15
    )
    # Each backward-Euler step closes the inventory with inflow, outflow and removal.
    system = qualification._TransportReference(
        assembly.mass, assembly.matrix, assembly.forcing
    )
    state = np.full(len(cells), 2.0)
    dt = 0.5
    following = qualification._backward_euler(
        system, state, dt=dt, steps=1, homogeneous=False
    )
    change = assembly.mass @ (following - state)
    external = (
        assembly.outflow @ following
        + assembly.removal @ following
        - np.sum(assembly.inflow)
    )
    assert abs(change + dt * external) <= 1e-14
    assert abs(dt * assembly.removal @ following) > 1e-6
    # Each material-pair flux functional is active on the solved state.
    for row in assembly.pair_flux.values():
        crossing = row @ following
        assert np.isfinite(crossing) and crossing != 0.0


def test_exact_affine_transport_matches_closed_form_and_backward_euler_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import neurofluid_qualification as qualification
    from tools.meshing_qualification.scenarios import image

    mass = np.asarray((2.0, 0.5))
    matrix = np.asarray(((0.6, -0.2), (-0.2, 0.3)))
    forcing = np.asarray((-0.4, 0.1))
    system = qualification._TransportReference(mass, matrix, forcing)
    state = np.asarray((1.0, 3.0))
    horizon = 1.5
    generator = -matrix / mass[:, None]
    steady = np.linalg.solve(matrix, -forcing)
    values, vectors = np.linalg.eig(generator)
    closed = steady + (
        vectors @ (np.exp(values * horizon) * np.linalg.solve(vectors, state - steady))
    )
    np.testing.assert_allclose(
        qualification._exact_transport(system, state, horizon), closed.real, rtol=1e-12
    )

    def refined(
        runtime: Any, initial: Any, *, steps: int, dt: float
    ) -> tuple[np.ndarray, bool]:
        del runtime
        return (
            qualification._backward_euler(
                system, initial, dt=dt, steps=steps, homogeneous=False
            ),
            True,
        )

    monkeypatch.setattr(qualification, "_refined_final_state", refined)
    runtime = SimpleNamespace(flatten=lambda value: np.asarray(value))
    steps, dt = 3, horizon / 3
    exact = qualification._exact_transport(system, state, horizon)
    coarse = qualification._relative_mass_error(
        system,
        qualification._backward_euler(system, state, dt=dt, steps=steps, homogeneous=False),
        exact,
    )
    study: Any = qualification._time_convergence(
        runtime, state, system, coarse, steps=steps, dt=dt, phase=lambda name: None
    )
    assert study["passed"] and image._time_step_convergence(study)
    assert all(0.75 <= order <= 1.25 for order in study["observed_orders"])

    # A refined runtime solving a different spatial system stalls at its
    # own steady state and must fail the order obligation.
    other = qualification._TransportReference(mass, 1.1 * matrix, forcing)

    def mismatched(
        runtime: Any, initial: Any, *, steps: int, dt: float
    ) -> tuple[np.ndarray, bool]:
        del runtime
        return (
            qualification._exact_transport(other, initial, steps * dt),
            True,
        )

    monkeypatch.setattr(qualification, "_refined_final_state", mismatched)
    stalled = qualification._time_convergence(
        runtime, state, system, coarse, steps=steps, dt=dt, phase=lambda name: None
    )
    assert not stalled["passed"] and not image._time_step_convergence(stalled)


@pytest.mark.parametrize(
    "corruption",
    ("missing_refinement", "changed_refinement", "lowered_order", "zero_width", "nan_width"),
)
def test_image_convergence_refuses_changed_scientific_controls(corruption: str) -> None:
    from tools.meshing_qualification.scenarios import image

    study = {
        "step_widths_seconds": [0.01, 0.005, 0.0025],
        "relative_errors": [0.004, 0.002, 0.001],
        "required_minimum_order": 0.75,
        "all_steps_accepted": True,
    }
    assert image._time_step_convergence(study)
    if corruption == "missing_refinement":
        study["step_widths_seconds"] = [0.01, 0.005]
        study["relative_errors"] = [0.004, 0.002]
    elif corruption == "changed_refinement":
        study["step_widths_seconds"] = [0.01, 0.005, 0.00125]
    elif corruption == "lowered_order":
        study["required_minimum_order"] = 0.0
    elif corruption == "zero_width":
        study["step_widths_seconds"] = [0.01, 0.0, 0.0025]
    else:
        study["step_widths_seconds"] = [0.01, np.nan, 0.0025]
    assert not image._time_step_convergence(study)


def _declared_parameters(qualification: Any, exchange: Any) -> Any:
    """Product parameters of one bulk cell plus the declared network controls."""
    return phx.applications.neurofluid.NeurofluidTransportParameters(
        porosity=np.asarray((0.8,)),
        bulk_diffusivity=np.asarray((0.01,)),
        bulk_velocity=np.zeros((1, 3)),
        bulk_boundary_volume_flux=np.zeros(4),
        bulk_boundary_inflow_concentration=np.zeros(4),
        bulk_removal_rate=np.zeros(1),
        units=phx.applications.neurofluid.NeurofluidTransportUnits(
            phx.units.MILLIMETER, phx.units.SECOND, phx.units.MILLIMOLAR
        ),
        **qualification._network_parameter_values(exchange),
    )


@pytest.mark.parametrize("case", ("two-slab", "oblique-junction-transport"))
def test_network_parameter_validation_refuses_one_perturbed_product_control(
    case: str,
) -> None:
    from tools import neurofluid_qualification as qualification

    declared = qualification.native_transport_case(case, 4)
    parameters = _declared_parameters(qualification, declared.exchange)
    assert qualification._network_parameter_error(declared.exchange, parameters) == 0.0
    tolerance = 1.0e-10
    perturbed = {
        "network_diffusivity": np.asarray((0.011,)),
        "network_volume_flow": np.asarray((1.0e-6,)),
        "exchange_coefficients": np.asarray((0.5, 0.55)),
        "averaging_radius": 0.0101,
        "reservoir_volumes": np.asarray((0.0031,)),
        "reservoir_coefficients": np.asarray((0.21,)),
    }
    for name, value in perturbed.items():
        faulty = replace(parameters, **{name: value})
        error = qualification._network_parameter_error(declared.exchange, faulty)
        assert error > tolerance, name
    # A product with an extra exchange site is refused, not silently truncated.
    extra = replace(parameters, exchange_coefficients=np.asarray((0.5, 0.5, 0.5)))
    assert qualification._network_parameter_error(declared.exchange, extra) == np.inf


def test_routine_network_controls_keep_their_product_identity() -> None:
    from tools import neurofluid_qualification as qualification

    declared = qualification.native_transport_case("two-slab", 2)
    historic = replace(
        _declared_parameters(qualification, declared.exchange),
        network_diffusivity=np.asarray((0.01,), dtype=np.float64),
        network_volume_flow=np.zeros(1, dtype=np.float64),
        exchange_coefficients=np.full(2, 0.5, dtype=np.float64),
        averaging_radius=0.01,
        reservoir_volumes=np.asarray((0.003,), dtype=np.float64),
        reservoir_coefficients=np.asarray((0.2,), dtype=np.float64),
    )
    assert (
        _declared_parameters(qualification, declared.exchange).parameter_id
        == historic.parameter_id
    )
    assert declared.exchange.network_initial == (1.0, 1.5)
    assert declared.exchange.reservoir_initial == (0.5,)


def test_image_size_request_admits_both_statistics_within_half_target() -> None:
    from phydrax.meshing.providers._native_route_checks import uniform_size_compliance
    from tools import neurofluid_qualification as qualification

    approved = qualification.NATIVE_IMAGE_SIZE_COMPLIANCE
    exact = qualification.FORMER_EXACT_IMAGE_SIZE_COMPLIANCE
    assert approved.target_statistics == exact.target_statistics == ("p50", "p95")
    assert (approved.relative_tolerance, approved.absolute_tolerance) == (0.5, 0.0)
    assert (exact.relative_tolerance, exact.absolute_tolerance) == (0.0, 0.0)
    record = qualification._size_request_record(approved)
    assert record["qualification"] == "approved-hard-request"
    assert record["strength"] == "HARD" and record["target_size"] == 1.0
    stress = qualification._size_request_record(exact)
    assert stress["qualification"] == "unqualified-stress-request"
    scope = phx.meshing.MeshingScope(
        "image-size", "r0", phx.meshing.MeshingEntityKind.GEOMETRY, 2, "boundary",
        np.asarray((0,), dtype=np.int64),
    )
    control = phx.meshing.UniformSizeControl(
        scope, 1.0, strength=phx.meshing.SizeControlStrength.HARD
    )
    # Compliance boundary: both statistics stay meaningful hard limits.
    for lengths, approved_issues in (
        (np.full(20, 1.0), ()),
        (np.linspace(0.6, 1.45, 20), ()),
        (np.linspace(0.4, 0.9, 20), ()),
        (np.linspace(0.3, 0.45, 20), ("p50", "p95")),
        (np.linspace(0.9, 1.8, 20), ("p95",)),
    ):
        _, _, issues = uniform_size_compliance(control, approved, lengths, 1.0)
        statistics = tuple(
            sorted(issue.split(":")[0].removeprefix("target_size_") for issue in issues)
        )
        assert statistics == approved_issues
        _, _, former = uniform_size_compliance(control, exact, lengths, 1.0)
        assert bool(former) == (not np.all(lengths == 1.0))


def test_declared_network_coupling_conserves_inventory_and_follows_the_case() -> None:
    from tools import neurofluid_qualification as qualification

    exchange = qualification._NETWORK_EXCHANGE_PHYSICS
    bulk_mass = np.asarray((0.4, 0.6, 0.5))
    # Normalized averaging rows: each node samples a convex bulk combination.
    average = np.asarray(((0.25, 0.75, 0.0), (0.0, 0.0, 1.0)))
    nodes = np.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 4.0)))
    area = 0.01
    system, weights = qualification._couple_network(
        bulk_mass, np.zeros((3, 3)), np.zeros(3), average, nodes, area, 1, exchange
    )
    length = 5.0
    np.testing.assert_allclose(system.mass[3:5], area * length / 2.0)
    np.testing.assert_allclose(system.mass[5], exchange.reservoir_volumes[0])
    np.testing.assert_allclose(
        weights, np.asarray(exchange.exchange_coefficients) * area * length / 2.0
    )
    np.testing.assert_allclose(system.matrix, system.matrix.T)
    # Exchange, edge diffusion and reservoir coupling move inventory without
    # creating it, and a uniform state is steady.
    np.testing.assert_allclose(np.sum(system.matrix, axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(system.matrix @ np.ones(6), 0.0, atol=1e-15)
    state = np.asarray((3.0, 2.0, 1.0, 1.0, 1.5, 0.5))
    final = qualification._exact_transport(system, state, 50.0)
    np.testing.assert_allclose(system.mass @ final, system.mass @ state, rtol=1e-12)
    # The network-to-reservoir exchange rate follows the declared coefficients.
    rate = -(system.matrix @ state)[5]
    np.testing.assert_allclose(
        rate,
        exchange.reservoir_volumes[0]
        * exchange.reservoir_coefficients[0]
        * (state[4] - state[5]),
    )
    stronger = exchange._replace(reservoir_coefficients=(0.4,))
    doubled, _ = qualification._couple_network(
        bulk_mass, np.zeros((3, 3)), np.zeros(3), average, nodes, area, 1, stronger
    )
    np.testing.assert_allclose(-(doubled.matrix @ state)[5], 2.0 * rate)
    with pytest.raises(RuntimeError, match="diffusive two-node"):
        qualification._couple_network(
            bulk_mass,
            np.zeros((3, 3)),
            np.zeros(3),
            average,
            nodes,
            area,
            1,
            exchange._replace(network_volume_flow=(1.0,)),
        )


def test_independent_circle_average_locates_points_in_kuhn_tetrahedra() -> None:
    from tools import neurofluid_qualification as qualification

    # One unit voxel: tetrahedron 0 is x >= y >= z, tetrahedron 2 is y >= x >= z.
    points, cells, _ = _kuhn_mesh(
        np.zeros((1, 1, 1), dtype=np.int64), np.eye(3), np.full(3, 0.5)
    )
    vertical = np.asarray(((0.0, 0.0, 1.0),))
    # A small circle well inside tetrahedron 0 puts its whole weight there.
    inside = qualification._exchange_average(
        points, cells, np.asarray(((0.7, 0.4, 0.15),)), vertical, 0.05, 12,
        np.zeros((1, 6)),
    )
    np.testing.assert_allclose(inside.rows, np.eye(6)[:1], atol=1e-15)
    assert inside.unlocated_points == 0 and inside.shared_face_points == 0
    # Crossing the face x = y: e1 = -x, e2 = -y, so the point at angle theta
    # has x - y = -0.05 + 0.1 (sin theta - cos theta), positive for the four
    # angles 90..180 degrees out of twelve.
    across = qualification._exchange_average(
        points, cells, np.asarray(((0.6, 0.65, 0.2),)), vertical, 0.1, 12,
        np.zeros((1, 6)),
    )
    np.testing.assert_allclose(
        across.rows[0], np.asarray((4.0, 0.0, 8.0, 0.0, 0.0, 0.0)) / 12.0, atol=1e-15
    )
    assert across.shared_face_points == 0
    # Reversing the tangent maps the circle onto itself.
    reversed_ = qualification._exchange_average(
        points, cells, np.asarray(((0.6, 0.65, 0.2),)), -vertical, 0.1, 12,
        across.rows,
    )
    assert reversed_.stencil_error <= qualification.EXCHANGE_STENCIL_TOLERANCE
    np.testing.assert_allclose(
        qualification._node_tangents(
            np.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 4.0))),
            np.asarray((0,)),
            np.asarray((1,)),
        ),
        np.asarray(((0.6, 0.0, 0.8), (0.6, 0.0, 0.8))),
    )


def test_independent_circle_average_refuses_a_perturbed_product_stencil() -> None:
    from tools import neurofluid_qualification as qualification

    points, cells, _ = _kuhn_mesh(
        np.zeros((1, 1, 1), dtype=np.int64), np.eye(3), np.full(3, 0.5)
    )
    vertical = np.asarray(((0.0, 0.0, 1.0),))
    center = np.asarray(((0.6, 0.65, 0.2),))
    # Radius 0.05 puts the points at 90 and 180 degrees on the face x = y:
    # two strictly in tetrahedron 0, two shared, eight in tetrahedron 2.
    product = np.asarray(((2.0, 0.0, 10.0, 0.0, 0.0, 0.0),)) / 12.0
    admitted = qualification._exchange_average(
        points, cells, center, vertical, 0.05, 12, product
    )
    assert admitted.shared_face_points == 2
    assert admitted.stencil_error <= qualification.EXCHANGE_STENCIL_TOLERANCE
    np.testing.assert_allclose(admitted.rows, product, atol=1e-15)
    # Moving one point's weight to a cell that does not contain it is refused.
    moved = product.copy()
    moved[0, 2] -= 1.0 / 12.0
    moved[0, 4] += 1.0 / 12.0
    refused = qualification._exchange_average(
        points, cells, center, vertical, 0.05, 12, moved
    )
    assert refused.stencil_error > qualification.EXCHANGE_STENCIL_TOLERANCE
    # So are a rescaled stencil, a different radius and a different point count.
    scaled = qualification._exchange_average(
        points, cells, center, vertical, 0.05, 12, 1.01 * product
    )
    assert scaled.stencil_error > qualification.EXCHANGE_STENCIL_TOLERANCE
    wider = qualification._exchange_average(
        points, cells, center, vertical, 0.1, 12, product
    )
    assert wider.stencil_error > qualification.EXCHANGE_STENCIL_TOLERANCE
    coarser = qualification._exchange_average(
        points, cells, center, vertical, 0.05, 8, product
    )
    assert coarser.stencil_error > qualification.EXCHANGE_STENCIL_TOLERANCE
    # A circle in the plane x = 0.95 stays inside the voxel; one reaching
    # z = 1.05 leaves the mesh and is reported as unlocated.
    contained = qualification._exchange_average(
        points, cells, np.asarray(((0.95, 0.5, 0.5),)), np.asarray(((1.0, 0.0, 0.0),)),
        0.1, 12, np.zeros((1, 6)),
    )
    assert contained.unlocated_points == 0
    escaping = qualification._exchange_average(
        points, cells, np.asarray(((0.5, 0.5, 0.95),)), np.asarray(((1.0, 0.0, 0.0),)),
        0.1, 12, np.zeros((1, 6)),
    )
    assert escaping.unlocated_points > 0


def test_native_transport_expired_whole_clock_prevents_source_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import neurofluid_qualification as qualification

    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("An expired whole clock must not begin another scientific stage")

    monkeypatch.setattr(qualification, "_execute_native_transport", forbidden)
    record = qualification.native_transport(timeout=0.0)
    assert record["successful"] is False
    assert record["failure"]["exception_type"] == "TimeoutError"
    assert record["failure"]["phase"] == "source-preparation"


@pytest.mark.parametrize("end_warm", (False, True), ids=("standalone-order", "warm-then-cold"))
def test_transport_cold_publication_keeps_the_original_deadline_and_process_binding(
    monkeypatch: pytest.MonkeyPatch, end_warm: bool,
) -> None:
    from tools import neurofluid_qualification as qualification

    assert callable(qualification.cold_worker)
    events: list[str] = []
    deadline = qualification.Deadline(qualification.perf_counter() + 120.0)
    fingerprint = {"accepted": "all-fields-and-history"}
    context: Any = {}
    ledger: Any = None

    class Worker:
        def __enter__(self) -> Any:
            events.append("prepare")
            return self

        def __exit__(self, *args: Any) -> None:
            events.append("close")

        def execute(self, path: Any, content_id: str, controls: Any) -> Any:
            assert events[-1] == ("end-warm" if end_warm else "fingerprint")
            events.append("execute")
            assert path.name == "state.zip"
            assert content_id == "actual-content"
            assert controls == {"expected_identity": fingerprint}
            ledger.record["cold_process"] = {"process_id": 123, "status": "passed"}
            return {"passed": True, "process_id": 123}

    def prepare(module: str, **kwargs: Any) -> Any:
        nonlocal ledger
        assert module == "tools.neurofluid_qualification"
        assert kwargs["deadline"] is deadline
        ledger = kwargs["ledger"]
        assert ledger.deadline is deadline
        return Worker()

    def publish(path: Any, records: Any) -> Any:
        events.append("publish")
        assert events == ["prepare", "publish"]
        return SimpleNamespace(path=path, content_id="actual-content")

    def fingerprint_records(records: Any) -> Any:
        events.append("fingerprint")
        return fingerprint

    def finish(content_id: str) -> None:
        assert content_id == "actual-content"
        assert events[-1] == "fingerprint"
        events.append("end-warm")

    monkeypatch.setattr(qualification, "prepare_cold_worker", prepare)
    monkeypatch.setattr(qualification, "write_meshing_source_closure", publish)
    monkeypatch.setattr(qualification, "array_tree_fingerprint", fingerprint_records)
    records: Any = SimpleNamespace()
    evidence = qualification._publish_transport_checkpoint(
        records, phase=lambda name: None, context=context, deadline=deadline,
        end_warm_request=finish if end_warm else None,
    )
    assert events == [
        "prepare", "publish", "fingerprint",
        *(["end-warm"] if end_warm else []), "execute", "close",
    ]
    assert evidence["content_id"] == "actual-content"
    assert evidence["process"] == context["cold_process"]
    assert evidence["process_id"] == evidence["process"]["process_id"]


@pytest.fixture(scope="module")
def accepted_transport_checkpoint() -> Any:
    """Retain actual solved fields, both trajectories, and the genuine AD action."""
    import jax

    from tools import neurofluid_qualification as qualification

    generated = qualification._prepare_transport_generation(
        case="two-slab",
        size=2,
        maximum_cells=4096,
        maximum_vertices=4096,
        size_compliance=qualification.NATIVE_IMAGE_SIZE_COMPLIANCE,
        phase=lambda name: None,
        context={},
        deadline=qualification.Deadline(qualification.perf_counter() + 120.0),
    )
    execution = generated["native_execution"]
    root = execution["phases"][execution["root"]]
    consumer_index = root["consumer_evidence"]
    assert consumer_index is not None
    consumer = execution["phases"][consumer_index]
    assert consumer_index != execution["root"]
    assert root["status_name"] == consumer["status_name"] == "OK"
    assert root["owner_id"] == consumer["owner_id"] == generated["plan"].plan_id
    assert root["total_work_units"] >= consumer["total_work_units"]
    assert root["total_geometry_queries"] >= consumer["total_geometry_queries"]
    assert root["total_work_units"] <= generated["request"].limits.maximum_work_units
    assert root["total_geometry_queries"] <= (
        generated["request"].limits.maximum_geometry_queries
    )
    source, result, network = (
        generated["source"],
        generated["result"],
        generated["network"],
    )
    case = phx.applications.neurofluid.NeurofluidCase(
        "checkpoint-transport", source.labels, source.compartments, result, network
    )
    parameters = qualification._transport_parameters(result, generated["transport_case"])
    plan = phx.applications.neurofluid.NeurofluidTransportPlan(case, parameters)
    runtime = plan.prepare()
    assert result.region_evidence is not None
    initial = phx.equations.MixedDimensionalTransportState(
        bulk=jnp.asarray(
            [
                generated["transport_case"].physics[region].initial_concentration
                for region in result.region_evidence.cell_region_ids
            ],
            dtype=jnp.float64,
        ),
        network=jnp.asarray((1.0, 1.5), dtype=jnp.float64),
        reservoirs=jnp.asarray((0.5,), dtype=jnp.float64),
    )
    steps, dt = 2, 0.01
    execute = jax.jit(
        lambda owner, state: qualification._transport_loop(
            owner, state, steps=steps, dt=dt
        )
    )
    final, (history, accepted) = jax.block_until_ready(execute(runtime, initial))
    continued = jax.block_until_ready(execute(runtime, final))
    tangent = runtime.unflatten(jnp.ones(sum(runtime.sizes), dtype=jnp.float64))
    _, ad_value = jax.block_until_ready(
        jax.jvp(lambda state: execute(runtime, state)[0], (initial,), (tangent,))
    )
    return phx.applications.neurofluid.NeurofluidTransportCheckpoint(
        source=source,
        result=result,
        network_plan=qualification._network_source_plan(network),
        parameters=parameters,
        initial=initial,
        history=history,
        accepted=accepted,
        final=final,
        continued=continued,
        ad_tangent=tangent,
        ad_value=ad_value,
        case_id=case.case_id,
        physics_ids=(case.case_revision, plan.plan_id, runtime.runtime_id),
        size=2,
        steps=steps,
        dt=dt,
        step_cursor=steps,
        time=steps * dt,
        target_error=0.05,
        balance_tolerance=1e-9,
    )


@pytest.mark.parametrize("field", ("bulk", "network", "reservoirs"))
@pytest.mark.parametrize("trajectory", ("accepted", "continued"))
def test_transport_checkpoint_refuses_corrupt_final_history(
    accepted_transport_checkpoint: Any, field: str, trajectory: str
) -> None:
    checkpoint = accepted_transport_checkpoint
    if trajectory == "accepted":
        changed = eqx.tree_at(
            lambda state: getattr(state, field),
            checkpoint.final,
            getattr(checkpoint.final, field) + 0.01,
        )
        changes = {"final": changed}
    else:
        changed = eqx.tree_at(
            lambda state: getattr(state, field),
            checkpoint.continued[0],
            getattr(checkpoint.continued[0], field) + 0.01,
        )
        changes = {"continued": (changed, checkpoint.continued[1])}
    with pytest.raises(ValueError, match=rf"final\.{field} must equal"):
        replace(checkpoint, **changes)


@pytest.mark.parametrize(
    "changes,match",
    (
        ({"step_cursor": 0}, "clock"),
        ({"time": 0.0}, "clock"),
        ({"physics_ids": ("changed", "plan", "runtime")}, "physics_ids"),
        ({"accepted": np.asarray((True, False), dtype=np.bool_)}, "accepted must be"),
    ),
)
def test_transport_checkpoint_refuses_corrupt_clocks_physics_and_acceptance(
    accepted_transport_checkpoint: Any, changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(accepted_transport_checkpoint, **changes)


def test_transport_checkpoint_roundtrip_preserves_every_field_and_continues(
    accepted_transport_checkpoint: Any, tmp_path: Any
) -> None:
    import jax

    from tools import neurofluid_qualification as qualification

    checkpoint = accepted_transport_checkpoint
    receipt = qualification.write_meshing_source_closure(
        tmp_path / "accepted-transport.zip", checkpoint
    )
    restored = qualification.read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert restored.checkpoint_id == checkpoint.checkpoint_id
    assert qualification.array_tree_fingerprint(restored) == (
        qualification.array_tree_fingerprint(checkpoint)
    )
    network = restored.network_plan.prepare()
    case = phx.applications.neurofluid.NeurofluidCase(
        restored.case_id,
        restored.source.labels,
        restored.source.compartments,
        restored.result,
        network,
    )
    runtime = phx.applications.neurofluid.NeurofluidTransportPlan(
        case, restored.parameters
    ).prepare()
    actual = jax.block_until_ready(
        jax.jit(
            lambda owner, state: qualification._transport_loop(
                owner, state, steps=restored.steps, dt=restored.dt
            )
        )(runtime, restored.final)
    )
    for first, second in zip(
        jax.tree.leaves(actual), jax.tree.leaves(checkpoint.continued), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    replay = qualification.write_meshing_source_closure(
        tmp_path / "replayed-transport.zip", restored
    )
    assert replay.content_id == receipt.content_id


@pytest.mark.parametrize("field", ("bulk", "network", "reservoirs"))
@pytest.mark.parametrize("history", (False, True))
@pytest.mark.parametrize("corruption", (0.01, np.nan, np.inf))
def test_cold_transport_comparison_detects_every_corrupt_field(
    accepted_transport_checkpoint: Any,
    field: str,
    history: bool,
    corruption: float,
) -> None:
    from tools import neurofluid_qualification as qualification

    expected = accepted_transport_checkpoint.continued

    def select(value: Any) -> Any:
        return getattr(value[1][0] if history else value[0], field)

    changed = eqx.tree_at(select, expected, select(expected) + corruption)
    assert qualification._transport_continuation_error(expected, expected) == 0.0
    assert qualification._transport_continuation_error(changed, expected) > (
        accepted_transport_checkpoint.balance_tolerance
    )


@pytest.mark.parametrize("field", ("bulk", "network", "reservoirs"))
def test_transport_scientific_history_gate_refuses_interior_step_corruption(
    accepted_transport_checkpoint: Any, field: str
) -> None:
    from tools import neurofluid_qualification as qualification

    checkpoint = accepted_transport_checkpoint
    changed_history = eqx.tree_at(
        lambda state: getattr(state, field),
        checkpoint.history,
        getattr(checkpoint.history, field).at[0, 0].add(0.01),
    )
    # Preserve the final accepted state and every clock/identity. Structural
    # history validation alone cannot detect an altered interior physical step.
    changed = replace(checkpoint, history=changed_history)
    network = changed.network_plan.prepare()
    case = phx.applications.neurofluid.NeurofluidCase(
        changed.case_id,
        changed.source.labels,
        changed.source.compartments,
        changed.result,
        network,
    )
    runtime = phx.applications.neurofluid.NeurofluidTransportPlan(
        case, changed.parameters
    ).prepare()
    invariants, _ = qualification._transport_invariants(
        changed.source,
        changed.result,
        qualification.native_transport_case("two-slab", changed.size),
        runtime,
        network,
        changed.parameters,
        changed.initial,
        changed.history,
        changed.accepted,
        steps=changed.steps,
        dt=changed.dt,
        target_error=changed.target_error,
        balance_tolerance=changed.balance_tolerance,
    )
    assert invariants["successful"] is False
    assert invariants["inventory_error"] > changed.balance_tolerance
    assert invariants["local_balance_error"] > changed.balance_tolerance


def test_oblique_source_uses_the_authored_affine_not_rounded_outer_planes() -> None:
    from itertools import product

    from tools import neurofluid_qualification as qualification

    source, _ = qualification._junction_source(6)
    geometry = source.outer_geometry
    assert geometry is not None and geometry.exact_source is not None
    indices = (
        np.asarray(tuple(product((0, 1), repeat=3)), dtype=np.float64) * (6, 6, 5) - 0.5
    )
    matrix = qualification._JUNCTION_AFFINE
    exact_images = np.asarray(
        [
            [
                float(
                    Fraction(float(matrix[axis, 3]))
                    + sum(
                        (
                            Fraction(float(matrix[axis, direction]))
                            * Fraction(float(index[direction]))
                            for direction in range(3)
                        ),
                        Fraction(0),
                    )
                )
                for axis in range(3)
            ]
            for index in indices
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(source.outer_surface.mesh.coordinates, exact_images)
    np.testing.assert_array_equal(geometry.coordinates, exact_images)
    rounded_outer = source.labels.asset.spatial_affine.index_to_world(indices)
    assert not np.array_equal(exact_images, rounded_outer)
    # The same displayed coordinates without their authored affine source
    # are a different scientific/container identity, never a compatibility alias.
    literal = phx.geometry.CompartmentMeshingSource(
        source.labels, source.compartments, source.outer_surface, source.interfaces
    )
    assert source.source_id != literal.source_id
    assert source.source_revision == literal.source_revision == source.labels.label_volume_id
    with pytest.raises(RuntimeError, match="transport.cold_source"):
        qualification._declared_transport_case(literal, 6)


@pytest.mark.parametrize("field", ("bulk", "network", "reservoirs"))
def test_continued_transport_balances_every_interior_physical_step(
    accepted_transport_checkpoint: Any, field: str
) -> None:
    from tools import neurofluid_qualification as qualification

    checkpoint = accepted_transport_checkpoint
    network = checkpoint.network_plan.prepare()
    case = phx.applications.neurofluid.NeurofluidCase(
        checkpoint.case_id,
        checkpoint.source.labels,
        checkpoint.source.compartments,
        checkpoint.result,
        network,
    )
    runtime = phx.applications.neurofluid.NeurofluidTransportPlan(
        case, checkpoint.parameters
    ).prepare()
    system = qualification._independent_system(
        checkpoint.source,
        checkpoint.result,
        qualification.native_transport_case("two-slab", checkpoint.size),
        runtime,
        network,
        checkpoint.parameters,
    )
    history = checkpoint.continued[1][0]
    changed = eqx.tree_at(
        lambda state: getattr(state, field),
        history,
        getattr(history, field).at[0, 0].add(0.01),
    )

    def defect(value: Any) -> float:
        return qualification._transport_history_balance_error(
            system.reference,
            np.asarray(runtime.flatten(checkpoint.final)),
            np.concatenate(
                [
                    np.asarray(getattr(value, name))
                    for name in ("bulk", "network", "reservoirs")
                ],
                axis=1,
            ),
            dt=checkpoint.dt,
        )

    assert defect(history) <= checkpoint.balance_tolerance
    assert defect(changed) > checkpoint.balance_tolerance


def test_exact_image_authoring_cannot_reset_a_supplied_exhausted_work_budget() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from tools import neurofluid_qualification as qualification

    limits = phx.meshing.MeshingLimits(maximum_work_units=1)
    with qualification.NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(
            limits.maximum_work_units,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
        ) as ticket:
            budget = qualification.CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket.native_target(), budget.activate():
                with pytest.raises(CoordinateEnclosureResourceError):
                    qualification._junction_source(6, coordinate_budget=budget)


def test_native_image_source_refuses_before_voxel_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import NoReturn

    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from tools import neurofluid_qualification as qualification

    allocated: list[tuple[int, ...]] = []

    def observe_allocation(shape: tuple[int, ...], /, **kwargs: Any) -> NoReturn:
        allocated.append(shape)
        raise AssertionError("Resource refusal must precede voxel allocation.")

    limits = phx.meshing.MeshingLimits(maximum_work_units=1)
    with qualification.NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(
            0,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
        ) as ticket:
            budget = qualification.CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket.native_target(), budget.activate():
                monkeypatch.setattr(
                    qualification.np, "ones", observe_allocation
                )
                with pytest.raises(CoordinateEnclosureResourceError):
                    qualification._segmentation(2, coordinate_budget=budget)
    assert allocated == []


@pytest.mark.parametrize("operation", ("complex", "surfaces", "source"))
@pytest.mark.parametrize("shortfall", ("work", "storage"))
def test_unicode_compartment_metadata_has_exact_source_admission(
    native_case: phx.applications.neurofluid.NeurofluidCase,
    operation: str,
    shortfall: str,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    repeated = "𐐷" * 4096

    def identifier(suffix: str) -> str:
        return f"{repeated}:{suffix}"

    label_ids = (identifier("label-a"), identifier("label-b"))
    compartment_ids = (identifier("region-a"), identifier("region-b"))
    values = np.asarray(native_case.segmentation.asset.values)
    labels = phx.imaging.LabelVolume(
        _image(
            values,
            phx.units.ONE,
            identifier("source-asset"),
            value_kind=phx.measurement.ValueKind.CATEGORICAL,
        ),
        phx.imaging.LabelOntology(
            "long-ontology",
            "synthetic",
            "1",
            (
                phx.imaging.LabelDefinition(1, label_ids[0], "label-a"),
                phx.imaging.LabelDefinition(2, label_ids[1], "label-b"),
            ),
        ),
    )
    compartments = (
        phx.geometry.CompartmentDefinition(
            compartment_ids[0],
            (label_ids[0],),
            identifier("material-a"),
            allowed_neighbor_ids=(compartment_ids[1],),
        ),
        phx.geometry.CompartmentDefinition(
            compartment_ids[1],
            (label_ids[1],),
            identifier("material-b"),
            containment_parent_id=compartment_ids[0],
            allowed_neighbor_ids=(compartment_ids[0],),
        ),
    )
    interfaces = (
        phx.geometry.CompartmentInterfaceDefinition(
            identifier("interface"),
            compartment_ids[0],
            compartment_ids[1],
            identifier("interface-role"),
        ),
    )
    expected_complex = phx.imaging.build_compartment_complex(
        labels, compartments, interfaces
    )
    expected_surfaces = phx.imaging.extract_compartment_surfaces(
        labels, expected_complex
    )
    source_complex = phx.imaging.build_compartment_complex(labels, compartments, ())
    source_surfaces = phx.imaging.extract_compartment_surfaces(labels, source_complex)
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
            source_id=identifier("outer"),
            source_revision=labels.label_volume_id,
            coordinate_contract=contract,
            provenance=(identifier("outer-provenance"),),
        ),
    )
    expected_source = phx.geometry.CompartmentMeshingSource(
        labels, source_complex, outer, source_surfaces
    )
    type SourceResult = (
        phx.geometry.CompartmentComplex
        | phx.imaging.CompartmentSurfaceResult
        | phx.geometry.CompartmentMeshingSource
    )

    if operation == "complex":
        expected = expected_complex

        def invoke(budget: CoordinateEnclosureBudget) -> SourceResult:
            return phx.imaging.build_compartment_complex(
                labels, compartments, interfaces, budget=budget
            )

    elif operation == "surfaces":
        expected = expected_surfaces

        def invoke(budget: CoordinateEnclosureBudget) -> SourceResult:
            return phx.imaging.extract_compartment_surfaces(
                labels, expected_complex, budget=budget
            )

    else:
        expected = expected_source

        def invoke(budget: CoordinateEnclosureBudget) -> SourceResult:
            return phx.geometry.CompartmentMeshingSource(
                labels, source_complex, outer, source_surfaces, budget=budget
            )

    def run(work: int, storage: int) -> tuple[SourceResult, int, int]:
        with HostBatchTicket(work, bytes_upper=storage) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=work,
                maximum_memory_bytes=storage,
            )
            with budget.activate():
                result = invoke(budget)
            return result, budget.required_work_units, budget.peak_bytes_upper

    _, required_work, required_storage = run(1 << 30, 1 << 30)
    assert required_work > 1 and required_storage > 1
    escaped_text_upper = 12 * len(repeated)
    # These floors exceed the prior item-only bounds for even one long field.
    assert required_work > escaped_text_upper
    assert required_storage > 2 * escaped_text_upper
    limits = {
        "work": (required_work - 1, required_storage, "coefficient_work"),
        "storage": (required_work, required_storage - 1, "polynomial_storage"),
    }
    short_work, short_storage, resource = limits[shortfall]
    with pytest.raises(CoordinateEnclosureResourceError) as refused:
        run(short_work, short_storage)
    assert refused.value.resource == resource

    actual, _, _ = run(required_work, required_storage)
    if operation == "complex":
        assert isinstance(actual, phx.geometry.CompartmentComplex)
        assert isinstance(expected, phx.geometry.CompartmentComplex)
        assert actual.complex_id == expected.complex_id
        assert actual.adjacency.report_id == expected.adjacency.report_id
        assert actual.observed_adjacencies == expected.observed_adjacencies
    elif operation == "surfaces":
        assert isinstance(actual, phx.imaging.CompartmentSurfaceResult)
        assert isinstance(expected, phx.imaging.CompartmentSurfaceResult)
        assert actual.extraction_id == expected.extraction_id
        assert tuple(value.surface_id for value in actual.surfaces) == tuple(
            value.surface_id for value in expected.surfaces
        )
        assert tuple(
            value.surface.metadata.metadata_id for value in actual.surfaces
        ) == tuple(value.surface.metadata.metadata_id for value in expected.surfaces)
    else:
        assert isinstance(actual, phx.geometry.CompartmentMeshingSource)
        assert isinstance(expected, phx.geometry.CompartmentMeshingSource)
        assert actual.source_id == expected.source_id
        assert actual.source_revision == expected.source_revision
        assert actual.interface_definitions == expected.interface_definitions


def test_compartment_source_revalidation_rejects_mutated_labels_and_releases_scratch() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from tools import neurofluid_qualification as qualification

    source, _ = qualification._native_source(4)
    values = source.labels.asset.values
    values.setflags(write=True)
    values[0, 0, 0] = 2
    values.setflags(write=False)
    limits = phx.meshing.MeshingLimits()
    with qualification.NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(
            limits.maximum_work_units,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
        ) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket.native_target(), budget.activate():
                with pytest.raises(
                    ValueError,
                    match="CompartmentMeshingSource.compartments must match image",
                ):
                    source.validate_source_integrity(budget=budget)
            assert budget.work_units > 0
            assert budget.temporary_bytes_upper == 0


@pytest.mark.parametrize("shortfall", (0, 1), ids=("R", "R-minus-one"))
def test_cold_declared_image_borrows_exact_remaining_work(
    monkeypatch: pytest.MonkeyPatch, shortfall: int
) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from tools import neurofluid_qualification as qualification

    source, _ = qualification._junction_source(6)
    limits = phx.meshing.MeshingLimits()
    scope_type = qualification.NativeExecutionScope
    with scope_type(limits.native_limits()) as scope:
        measured = qualification.CoordinateEnclosureBudget(
            scope.ticket_view(),
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        qualification._declared_transport_case(source, 6, coordinate_budget=measured)
        required = measured.work_units
    assert required > 1

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Borrowed cold authoring must not open another native scope")

    monkeypatch.setattr(qualification, "NativeExecutionScope", forbidden)
    with scope_type(limits.native_limits()) as scope:
        budget = qualification.CoordinateEnclosureBudget(
            scope.ticket_view(),
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        # Exhaust the original ledger down to R or R-1 without changing the
        # authored source's retained preparation controls.
        budget.reserve(limits.maximum_work_units - required + shortfall)
        if shortfall:
            with pytest.raises(CoordinateEnclosureResourceError):
                qualification._declared_transport_case(source, 6, coordinate_budget=budget)
        else:
            declared = qualification._declared_transport_case(
                source, 6, coordinate_budget=budget
            )
            assert declared.source.source_id == source.source_id
            assert declared.source.source_revision == source.source_revision
            assert budget.work_units == limits.maximum_work_units
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == budget.work_units


@pytest.mark.parametrize("case,size", (("two-slab", 2), ("oblique-junction-transport", 6)))
def test_declared_image_explicit_standalone_path(case: str, size: int) -> None:
    from tools import neurofluid_qualification as qualification

    declared = qualification.native_transport_case(case, size, coordinate_budget=None)
    restored = qualification._declared_transport_case(
        declared.source, size, coordinate_budget=None
    )
    assert restored.source.source_id == declared.source.source_id
    assert restored.source.source_revision == declared.source.source_revision


@pytest.mark.parametrize("ended", (False, True), ids=("foreign", "ended"))
def test_cold_declared_image_refuses_foreign_or_ended_budget(ended: bool) -> None:
    from contextvars import Context

    from tools import neurofluid_qualification as qualification

    source, _ = qualification._junction_source(6)
    limits = phx.meshing.MeshingLimits()
    with qualification.NativeExecutionScope(limits.native_limits()) as scope:
        budget = qualification.CoordinateEnclosureBudget(
            scope.ticket_view(),
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        if not ended:
            with pytest.raises(RuntimeError, match="foreign scope"):
                Context().run(
                    qualification._declared_transport_case,
                    source, 6, coordinate_budget=budget,
                )
    if ended:
        with pytest.raises(RuntimeError, match="new work or growth on a retired ticket"):
            qualification._declared_transport_case(source, 6, coordinate_budget=budget)


def test_exact_source_root_keeps_an_already_expired_whole_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import neurofluid_qualification as qualification

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Expired source authoring must not begin input construction")

    monkeypatch.setattr(qualification, "native_transport_case", forbidden)
    with pytest.raises(TimeoutError):
        qualification._prepare_transport_generation(
            case="oblique-junction-transport",
            size=6,
            maximum_cells=20000,
            maximum_vertices=20000,
            size_compliance=qualification.NATIVE_IMAGE_SIZE_COMPLIANCE,
            phase=lambda name: None,
            context={},
            deadline=qualification.Deadline(qualification.perf_counter() - 1.0),
        )


def test_transport_cold_bootstrap_precedes_decode_and_shares_one_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import neurofluid_qualification as qualification

    events: list[str] = []
    limits, historical_receipt, budget, carrier = (object() for _ in range(4))
    packet: Any = SimpleNamespace(
        archive_path="accepted.zip",
        content_id="actual-content",
        deadline=qualification.Deadline(qualification.perf_counter() + 120.0),
    )
    report = qualification.ColdWorkerReport({"passed": True}, read_seconds=0.25)
    ended = SimpleNamespace(to_record=lambda: {"ended": True})

    def controls(path: Any, **kwargs: Any) -> Any:
        events.append("raw-controls")
        assert path == packet.archive_path
        assert kwargs == {"expected_content_id": packet.content_id}
        return SimpleNamespace(execution_evidence=historical_receipt, limits=limits)

    def continue_transport(
        incoming: Any, shared_budget: Any, carriers: list[Any]
    ) -> Any:
        events.append("reader-reproof-preparation-continuation")
        assert incoming is packet
        assert shared_budget is budget
        carriers.append(carrier)
        return report

    def execute(source: Any, original: Any, callback: Any, **kwargs: Any) -> Any:
        events.append("request")
        assert source is historical_receipt
        assert original is limits
        assert 0.0 < kwargs["workflow_remaining_seconds"] <= 120.0
        value = callback(budget)
        assert kwargs["target"](value) is carrier
        return SimpleNamespace(
            value=value, result=SimpleNamespace(execution_evidence=ended)
        )

    monkeypatch.setattr(qualification, "read_meshing_source_execution_controls", controls)
    monkeypatch.setattr(qualification, "_continue_cold_transport", continue_transport)
    monkeypatch.setattr(qualification, "execute_native_request", execute)
    restored = qualification.cold_worker(packet)
    assert events == [
        "raw-controls", "request", "reader-reproof-preparation-continuation"
    ]
    assert restored is report
    assert restored.record["native_execution"] == {"ended": True}


def test_transport_warm_science_enters_one_original_root_before_authoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._meshcore import current_native_execution_scope
    from tools import neurofluid_qualification as qualification

    observed: list[Any] = []

    class StopAfterAdmission(Exception):
        pass

    def scientific_consumers(**kwargs: Any) -> Any:
        root = current_native_execution_scope()
        assert root is not None
        budget = kwargs["coordinate_budget"]
        budget.reserve(7)
        observed.append(root)
        raise StopAfterAdmission

    monkeypatch.setattr(qualification, "_run_warm_native_transport", scientific_consumers)
    with pytest.raises(StopAfterAdmission):
        qualification._execute_native_transport(
            case="two-slab", size=2, repeats=3, steps=5, dt=0.01,
            target_error=0.05, balance_tolerance=1e-9,
            maximum_cells=20000, maximum_vertices=20000, oracle_capacity=2048,
            phase=lambda name: None, context={},
            deadline=qualification.Deadline(qualification.perf_counter() + 120.0),
        )
    assert len(observed) == 1
    evidence = observed[0].completed_evidence()
    assert evidence is not None
    assert evidence.managed_admission_peak_bytes_upper is not None
    assert current_native_execution_scope() is None


@pytest.mark.parametrize("time", (0.0, 0.01, 0.5, 3.0))
def test_transport_exponential_action_matches_full_dense_affine_solution(time: float) -> None:
    from scipy.linalg import expm

    from tools import neurofluid_qualification as qualification

    reference = qualification._TransportReference(
        mass=np.asarray((2.0, 3.0, 5.0, 7.0)),
        matrix=np.asarray(
            (
                (3.0, -1.0, -2.0, 0.0),
                (-0.5, 2.0, -1.0, -0.5),
                (-1.0, 0.0, 2.0, -1.0),
                (0.0, -1.0, -1.0, 3.0),
            )
        ),
        forcing=np.asarray((-0.25, 0.5, -0.75, 0.125)),
    )
    initial = np.asarray((1.0, 2.0, 0.5, 0.25))
    generator = np.zeros((5, 5))
    generator[:4, :4] = -reference.matrix / reference.mass[:, None]
    generator[:4, 4] = -reference.forcing / reference.mass
    expected = (expm(time * generator) @ np.append(initial, 1.0))[:4]
    actual = qualification._exact_transport(reference, initial, time)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(initial, np.asarray((1.0, 2.0, 0.5, 0.25)))
