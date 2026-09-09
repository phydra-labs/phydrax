"""Compose selected observations with explicit clocks and frame routes."""

import numpy as np

import phydrax as phx


def manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "multimodal-example",
        checksum_algorithm="sha256",
        checksum="8" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"signal": 1.0},
        uncertainty={"signal": 0.0},
        lineage_ids=("synthetic",),
    )


def asset(name, values):
    support = phx.measurement.IndexSampleSupport((len(values),), ("sample",))
    field = phx.measurement.QuantityField(
        f"{name}.field",
        phx.measurement.QuantitySpec(
            "example", name, "signal", phx.units.ONE, "example.signal"
        ),
        phx.measurement.ValueLayout.scalar(),
        support,
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
        np.asarray(values, dtype=float),
    )
    return phx.measurement.MeasurementAsset.from_single_reference(
        name,
        field,
        manifest(),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RAW,
            transformation_id="example-generator",
        ),
    )


first, second = asset("reference", (1, 2, 3)), asset("observation", (4, 5, 6))
collection = phx.measurement.MeasurementCollection(
    "example-collection",
    "example-campaign",
    (first, second),
    (
        phx.measurement.MeasurementRoleAssignment(
            first.asset_id, phx.measurement.MeasurementRole.REFERENCE
        ),
        phx.measurement.MeasurementRoleAssignment(
            second.asset_id, phx.measurement.MeasurementRole.OBSERVATION
        ),
    ),
)
selected = phx.measurement.MeasurementSelectionPlan((second.asset_id,), 1, 3).apply(
    collection
)
print(
    {
        "collection": collection.content_id,
        "selected": selected.assets[0].field.values.tolist(),
    }
)
