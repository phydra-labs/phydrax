import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _geometry():
    theta = 2.0 * math.pi * np.arange(16) / 16
    contours = np.zeros((3, 16, 2))
    contours[0, :, 0] = 2.0
    for index, radius in enumerate((0.0, 0.3, 0.6)):
        contours[index, :, 0] = 2.0 + radius * np.cos(theta)
        contours[index, :, 1] = radius * np.sin(theta)
    return phx.applications.tokamak.FluxSurfaceGeometry(
        np.asarray([0.0, 0.5, 0.9]),
        contours,
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 2.0, 4.0]),
        np.asarray([2.0, 2.0, 2.0]),
        np.asarray([0.0, 0.3, 0.6]),
        np.asarray([1.0, 1.5, 2.0]),
        "synthetic-equilibrium",
    )


def _equilibrium():
    count = 5
    r = np.linspace(1.0, 3.0, count)
    z = np.linspace(-1.0, 1.0, count)
    rr, zz = np.meshgrid(r, z)
    profile = np.linspace(0.0, 1.0, count)
    boundary = np.asarray([[1.2, -0.8], [2.8, -0.8], [2.8, 0.8], [1.2, 0.8]])
    return phx.applications.tokamak.AxisymmetricEquilibrium(
        phx.applications.tokamak.AxisymmetricMachineFrame("fixture-machine"),
        phx.applications.tokamak.TokamakMagneticConvention.canonical(),
        r,
        z,
        (rr - 2.0) ** 2 + zz**2,
        np.ones(count),
        1.0 - profile,
        np.zeros(count),
        -np.ones(count),
        1.0 + profile,
        boundary,
        boundary,
        np.asarray([2.0, 0.0]),
        0.0,
        1.0,
        2.0,
        5.0,
        1.0e6,
        "fixture-equilibrium",
    )


def _reference():
    payload = b"imas-record"
    return phx.qualification.ReferenceArtifactManifest(
        "imas-record",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )


def test_tokamak_core_plant_uses_transactional_runtime():
    transport = phx.applications.tokamak.TokamakCoreTransportPlan(
        _geometry(), 1.0
    ).prepare()
    faces = np.zeros((transport.cell_count + 1,))
    coefficients = phx.applications.tokamak.TokamakTransportCoefficients(
        faces, faces, faces
    )
    reset = phx.applications.tokamak.TokamakCoreState(
        np.ones(transport.cell_count),
        np.ones(transport.cell_count),
        np.ones(transport.cell_count),
    )
    command_size = 3 * transport.cell_count + 3
    bundle = phx.applications.tokamak.TokamakCorePlantPlan(
        transport,
        coefficients,
        reset,
        np.zeros(command_size),
        np.ones(command_size),
    ).prepare()
    reset_result = bundle.plant.reset(jax.random.key(0), bundle.parameters)
    step = bundle.plant.step(
        phx.dynamics.PlantStepContext(0.0, 0.1, 0),
        reset_result.accepted_state,
        jnp.zeros((command_size,)),
        bundle.parameters,
    )

    assert bool(reset_result.successful)
    assert bool(step.successful)
    np.testing.assert_allclose(
        step.accepted_state.payload, reset_result.accepted_state.payload
    )


def test_tokamak_shot_record_reuses_governed_measurement_assets():
    time = phx.measurement.SampleTimeAxis.uniform("shot-time", 3, 0.1, phx.units.SECOND)
    support = phx.measurement.IndexSampleSupport(
        (3, 2),
        ("time", "rho"),
        time_axis=time,
        time_axis_dimension=0,
        frame_id="fixture-machine",
    )
    quantity = phx.applications.tokamak.resolve_tokamak_quantity(
        "electron-temperature",
        "temperature",
        phx.units.KELVIN,
        axes=("time", "rho"),
        support_association="flux-surface-cell",
    )
    field = phx.measurement.QuantityField(
        "electron-temperature-field",
        quantity,
        phx.measurement.ValueLayout.scalar(),
        support,
        phx.measurement.SamplingSemantics(
            phx.measurement.SpatialSamplingKind.CELL_AVERAGE
        ),
        np.ones((3, 2)),
    )
    asset = phx.measurement.MeasurementAsset.from_single_reference(
        "temperature-asset",
        field,
        _reference(),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.EXTERNAL,
            phx.measurement.DataStage.CALIBRATED,
        ),
        acquisition=phx.measurement.AcquisitionIdentity(
            "acquisition", "shot-1", "thomson", "fixture-protocol"
        ),
    )
    record = phx.applications.tokamak.TokamakShotRecord(
        "fixture-machine", "shot-1", time, (asset,), ("equilibrium-1",)
    )
    split = phx.applications.tokamak.TokamakShotSplit(
        (record.record_id,), (), ("locked-record",)
    )

    assert record.assets[0].content_id == asset.content_id
    assert record.record_id in split.training_record_ids


def test_imas_equilibrium_mapping_round_trips_exact_native_semantics():
    equilibrium = _equilibrium()
    exported = phx.applications.tokamak.interchange.export_imas_equilibrium_slice(
        equilibrium,
        phx.applications.tokamak.TokamakMagneticConvention.canonical(),
        ids_release="fixture-release",
        occurrence=0,
        time_s=0.5,
    )
    imported = phx.applications.tokamak.interchange.import_imas_equilibrium_slice(
        exported.record,
        _reference(),
        equilibrium.machine_frame,
        equilibrium.convention,
    )

    assert exported.report.valid
    assert imported.report.valid
    np.testing.assert_allclose(
        imported.equilibrium.poloidal_flux_wb_per_rad,
        equilibrium.poloidal_flux_wb_per_rad,
    )
    assert imported.equilibrium.plasma_current_a == equilibrium.plasma_current_a
