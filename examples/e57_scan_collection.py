"""Construct the E57 multi-scan contract from an admitted point product."""

import numpy as np

import phydrax as phx


reference = phx.qualification.ReferenceArtifactManifest(
    "e57-example",
    checksum_algorithm="sha256",
    checksum="a" * 64,
    size_bytes=1,
    license_id="synthetic",
    commercial_use_permitted=True,
    redistribution_permitted=True,
    training_use_permitted=True,
    export_permitted=True,
    export_classification="public",
    nondimensionalization={"length": 1.0},
    uncertainty={"position": 0.0},
    lineage_ids=("synthetic",),
)
contract = phx.SpatialCoordinateContract(
    phx.units.METER, coordinate_system="cartesian", reference_frame="scanner"
)
support = phx.measurement.PointSampleSupport(
    np.asarray(((1.0, 2.0, 3.0),)), ("point",), contract
)
derivation = phx.measurement.DerivationRecord(
    phx.measurement.DataOrigin.EXTERNAL,
    phx.measurement.DataStage.DERIVED,
    transformation_id="e57-lowering",
)
product = phx.measurement.LidarPointProduct(support, (), derivation, "scan", (reference,))
field = phx.measurement.QuantityField(
    "anchor.field",
    phx.measurement.QuantitySpec("e57", "active", "active", phx.units.ONE, "e57.active"),
    phx.measurement.ValueLayout(phx.measurement.ValueKind.COUNT),
    support,
    phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
    np.asarray((1,)),
)
asset = phx.measurement.MeasurementAsset.from_single_reference(
    "scan-anchor", field, reference, derivation
)
collection = phx.measurement.MeasurementCollection(
    "e57",
    "e57",
    (asset,),
    (
        phx.measurement.MeasurementRoleAssignment(
            asset.asset_id, phx.measurement.MeasurementRole.DERIVED_PRODUCT
        ),
    ),
)
report = phx.interchange.AdapterReport(
    phx.interchange.AdapterStatus.LOSSLESS,
    "E57",
    "phydrax-e57",
    source_id=reference.manifest_id,
    target_id=collection.content_id,
)
scan = phx.sensing.E57ScanRecord("guid", product, phx.geometry.RigidFrame.identity(3))
result = phx.sensing.E57ScanCollection(collection, (scan,), report)
print({"scan_count": len(result.scans), "collection_id": result.collection_id})
