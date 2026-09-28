"""Unequal soap-film double bubble at quasi-static equilibrium.

Seeds the closed-form equal-tension double bubble, solves the discrete
volume-constrained surface-energy minimum and compares pressures, energy and
the separating-film curvature with the closed-form standard double bubble.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from phydrax.applications.foams import (
    FoamEquilibriumPlan,
    FoamMaterialPlan,
    PreparedFoamEquilibrium,
    StandardDoubleBubble,
)
from phydrax.geometry.multiregion_surface import (
    PreparedMultiRegionSurface,
    seed_double_bubble,
)


SURFACE_TENSION = 0.025


def run() -> dict[str, Any]:
    reference = StandardDoubleBubble(1.0, 0.8, 2.0 * SURFACE_TENSION)
    seed = seed_double_bubble(1.0, 0.8, ring_points=24)
    topology = seed.topology(seed.capacity_plan(resource_id="double-bubble-example"))
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state)
    material = FoamMaterialPlan.soap_film(topology.region_ids, SURFACE_TENSION)
    equilibrium = PreparedFoamEquilibrium(FoamEquilibriumPlan(), surface, material, state)
    targets = jnp.asarray((reference.volume_first, reference.volume_second))
    result = equilibrium.solve(state, equilibrium.parameters(targets))
    evidence = result.evidence
    pressures = np.asarray(result.pressures[:2])
    wall_pressure = float(pressures[1] - pressures[0])
    return {
        "successful": result.successful,
        "kkt_status": int(evidence.kkt_status),
        "pressures": pressures.tolist(),
        "reference_pressures": list(reference.pressures),
        "energy": float(result.energy),
        "reference_energy": reference.energy,
        "separating_radius": 2.0 * reference.effective_tension / wall_pressure,
        "reference_separating_radius": reference.interface_radius,
        "virial_residual": float(evidence.virial_residual),
        "junction_angle_range_deg": [
            float(np.degrees(evidence.junction_minimum_angle)),
            float(np.degrees(evidence.junction_maximum_angle)),
        ],
    }


if __name__ == "__main__":
    print(run())
