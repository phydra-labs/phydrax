import sys

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _manifest():
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


def _image(values, unit, asset_id):
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
        phx.imaging.ImageFieldSpec.named(
            "t1", unit, phx.measurement.ValueKind.REAL_SCALAR
        ),
        phx.imaging.DeidentificationEvidence(
            "deid", "subject", "protocol", True, True, True
        ),
        _manifest(),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="synthetic-neurofluid-generator",
        ),
    )


def test_relaxivity_calibration_retains_negative_noisy_observations():
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


def test_neurofluid_transport_units_derive_physical_scales():
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


def test_periodic_flow_schedule_requires_complete_matching_cycle():
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


def test_image_space_inverse_reports_rank_and_builds_state_design_problem():
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


def test_pipeline_manifest_requires_topological_order():
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


def test_external_tool_provider_records_real_output(tmp_path):
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
