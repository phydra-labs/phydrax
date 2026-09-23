import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.units import ANGSTROM, ELECTRONVOLT


plan = qh.HaldaneModelPlan(
    1.0,
    0.0,
    0.2,
    np.pi / 2.0,
    1.0,
    ELECTRONVOLT,
    ANGSTROM,
)
result = qh.evaluate_haldane_topology(plan, mesh_shape=(21, 21))
print(
    {
        "chern": int(np.asarray(result.chern.nearest_integer)),
        "minimum_gap": float(np.asarray(result.chern.minimum_direct_gap)),
        "successful": bool(np.asarray(result.successful)),
    }
)
