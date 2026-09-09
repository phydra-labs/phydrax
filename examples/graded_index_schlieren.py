"""Trace a ray fan through a graded index and retain caustic evidence."""

import numpy as np

import phydrax as phx


contract = phx.SpatialCoordinateContract(
    phx.units.METER, coordinate_system="cartesian", reference_frame="world"
)
x = np.arange(8)[:, None, None]
index = np.broadcast_to(1.0 + 1.0e-3 * (x - 3.5) ** 2, (8, 8, 8)).copy()
field = phx.optics.geometric.StructuredRefractiveIndexField(
    index, (0, 0, 0), (1, 1, 1), contract, field_id="quadratic-index"
)
plan = phx.optics.geometric.GradedIndexRayPlan(field, 0.02, 100).prepare()
positions = np.asarray(((2.5, 3.0, 1.0), (3.5, 3.0, 1.0), (4.5, 3.0, 1.0)))
directions = np.tile((0.0, 0.0, 1.0), (3, 1))
result = plan.integrate(positions, directions)
fan = phx.optics.geometric.RayFanPlan(np.asarray(((1, 0, 0), (0, 1, 0)))).evaluate(result)
if not bool(result.evidence.successful):
    raise RuntimeError("Graded-index ray evidence rejected the result.")
print(
    {
        "final_positions": np.asarray(result.state.positions).tolist(),
        "caustic_detected": np.asarray(fan.caustic_detected).tolist(),
    }
)
