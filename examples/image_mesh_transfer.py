#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Project one physical scalar image onto a tetrahedral P1 field."""

import json
from hashlib import sha256

import numpy as np

import phydrax as phx


values = np.fromfunction(lambda i, j, k: i + 2.0 * j + 3.0 * k, (2, 2, 2))
contract = phx.SpatialCoordinateContract(
    phx.units.MILLIMETER,
    coordinate_system="cartesian-lps",
    reference_frame="synthetic-patient",
)
affine = phx.imaging.ImageIndexAffine(
    np.eye(4), "voxel-index", contract, phx.imaging.ImageAxisConvention.LPS
)
raw = values.tobytes()
reference = phx.qualification.ReferenceArtifactManifest(
    "synthetic-image",
    checksum_algorithm="sha256",
    checksum=sha256(raw).hexdigest(),
    size_bytes=len(raw),
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
asset = phx.imaging.MedicalImageAsset(
    "synthetic-image",
    "scalar-phantom",
    values,
    affine,
    phx.imaging.ImageValueLayout(
        "affine-scalar", phx.units.ONE, phx.imaging.ImageValueKind.SCALAR
    ),
    phx.imaging.DeidentificationEvidence(
        "synthetic-deid", "subject-0", "synthetic", True, True, True
    ),
    reference,
)
mesh = phx.discretization.CellMesh.from_tetrahedra(
    np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    np.asarray(((0, 1, 2, 3),)),
)
result = (
    phx.imaging.ImageToP1ProjectionPlan(asset, mesh, contract).prepare().apply(values)
)
expected = np.asarray((0.0, 1.0, 2.0, 3.0))
error = float(np.max(np.abs(np.asarray(result.values) - expected)))
if error > 1.0e-11 or not bool(result.evidence.successful):
    raise RuntimeError("Image-to-P1 projection failed affine-field exactness.")
print(
    json.dumps(
        {"maximum_error": error, "successful": bool(result.evidence.successful)}, indent=2
    )
)
