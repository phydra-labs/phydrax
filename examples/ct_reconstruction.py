"""Project and reconstruct a bounded voxel attenuation field."""

import numpy as np

import phydrax as phx


contract = phx.SpatialCoordinateContract(
    phx.units.METER, coordinate_system="cartesian", reference_frame="scanner"
)
rays = phx.measurement.RaySampleSupport(
    np.asarray(((-1, 0.5, 0.5), (-1, 1.5, 0.5))),
    np.asarray(((1, 0, 0), (1, 0, 0))),
    ("ray-0", "ray-1"),
    contract,
    far=np.asarray((5.0, 5.0)),
)
support = phx.imaging.tomography.ProjectionSupport(rays, (2,), ("view-0", "view-1"))
transform = phx.imaging.tomography.VoxelXRayTransformPlan(
    support, (2, 2, 1), (0, 0, 0), (1, 1, 1)
)
truth = np.asarray((1.0, 2.0, 3.0, 4.0)).reshape((2, 2, 1))
projections = transform.forward(truth)
reconstruction = phx.imaging.tomography.IterativeCTPlan(transform, 8).solve(
    projections.values
)
if not bool(reconstruction.successful):
    raise RuntimeError("CT reconstruction failed.")
print(
    {
        "projections": np.asarray(projections.values).tolist(),
        "residual": float(reconstruction.residual_norms[-1]),
    }
)
