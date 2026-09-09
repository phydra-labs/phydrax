import numpy as np

import phydrax as phx


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "foundation-source",
        checksum_algorithm="sha256",
        checksum="6" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"signal": 1.0},
        uncertainty={"signal": 0.1},
        lineage_ids=("synthetic",),
    )


def _asset(name, values):
    support = phx.measurement.IndexSampleSupport((len(values),), ("sample",))
    field = phx.measurement.QuantityField(
        f"{name}.field",
        phx.measurement.QuantitySpec(
            "test", name, "signal", phx.units.ONE, "test.signal"
        ),
        phx.measurement.ValueLayout.scalar(),
        support,
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
        np.asarray(values, dtype=float),
    )
    return phx.measurement.MeasurementAsset.from_single_reference(
        name,
        field,
        _manifest(),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RAW,
            transformation_id="foundation-generator",
        ),
    )


def test_collection_relations_and_bounded_selection_remain_explicit():
    first, second = _asset("first", (1, 2, 3)), _asset("second", (4, 5, 6))
    relation = phx.measurement.MeasurementRelation(
        first.asset_id,
        second.asset_id,
        phx.measurement.MeasurementRelationKind.SYNCHRONIZED_WITH,
        "clock-calibration",
    )
    collection = phx.measurement.MeasurementCollection(
        "campaign",
        "campaign",
        (first, second),
        (
            phx.measurement.MeasurementRoleAssignment(
                first.asset_id, phx.measurement.MeasurementRole.REFERENCE
            ),
            phx.measurement.MeasurementRoleAssignment(
                second.asset_id, phx.measurement.MeasurementRole.OBSERVATION
            ),
        ),
        (relation,),
    )
    selected = phx.measurement.MeasurementSelectionPlan(
        (first.asset_id, second.asset_id), 1, 3, maximum_samples=4
    ).apply(collection)
    np.testing.assert_allclose(selected.assets[0].field.values, (2, 3))
    assert selected.parent_collection_ids == (collection.content_id,)
    assert selected.relations == ()


def test_affine_piecewise_clocks_and_frame_routes_are_bounded():
    source = phx.measurement.ClockIdentity("sensor", phx.units.SECOND, "relative")
    target = phx.measurement.ClockIdentity("experiment", phx.units.SECOND, "relative")
    affine = phx.measurement.AffineClockMap(
        source, target, 0.5, 1.001, "clock-fit", (0.0, 10.0)
    ).prepare()
    mapped, evidence = affine.map(np.asarray((0.0, 10.0)))
    np.testing.assert_allclose(mapped, (0.5, 10.51))
    assert bool(evidence.successful)
    piecewise = phx.measurement.PiecewiseClockMap(
        source,
        target,
        np.asarray((0.0, 5.0, 10.0)),
        np.asarray((1.0, 6.0, 12.0)),
        "drift-fit",
    ).prepare()
    values, evidence = piecewise.map(np.asarray((2.5, 7.5)))
    np.testing.assert_allclose(values, (3.5, 9.0))
    assert bool(evidence.successful)

    axis = phx.measurement.SampleTimeAxis(
        "frame-time", np.asarray((0.0, 1.0)), phx.units.SECOND
    )
    sensor_to_body = phx.geometry.FrameTransformTimeline(
        "sensor",
        "body",
        axis,
        (
            phx.geometry.RigidFrame.identity(3),
            phx.geometry.RigidFrame(np.eye(3), np.asarray((1.0, 0.0, 0.0))),
        ),
        "sensor-extrinsic",
    )
    body_to_world = phx.geometry.FrameTransformTimeline(
        "body",
        "world",
        axis,
        (
            phx.geometry.RigidFrame(np.eye(3), np.asarray((0.0, 2.0, 0.0))),
            phx.geometry.RigidFrame(np.eye(3), np.asarray((0.0, 2.0, 0.0))),
        ),
        "body-pose",
    )
    route = phx.geometry.FrameTransformGraph(
        (sensor_to_body, body_to_world)
    ).prepare_route("sensor", "world")
    rotation, translation, evidence = route.evaluate(np.asarray(0.5))
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(translation, (0.5, 2.0, 0.0), atol=1e-12)
    assert bool(evidence.successful)
