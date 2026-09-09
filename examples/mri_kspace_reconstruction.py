"""Encode and reconstruct a complex image with matched Cartesian MRI actions."""

import numpy as np

import phydrax as phx


coils = phx.imaging.mri.CoilSensitivityField(
    np.ones((1, 8, 8), dtype=np.complex64), ("coil-0",), field_id="uniform-coil"
)
encoding = phx.imaging.mri.CartesianMRIEncodingPlan(coils)
truth = np.zeros((8, 8), dtype=np.complex64)
truth[3:5, 3:5] = 1.0 + 0.25j
kspace, evidence = encoding.forward(truth)
reconstruction = phx.imaging.mri.CGSensePlan(encoding, 4).reconstruct(kspace)
if not bool(evidence.successful & reconstruction.successful):
    raise RuntimeError("MRI encoding or reconstruction failed.")
print({"maximum_error": float(np.max(np.abs(np.asarray(reconstruction.image) - truth)))})
