from fractions import Fraction

import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.units import ELECTRONVOLT


sphere = qh.HaldaneSpherePlan(
    2,
    qh.MonopoleLandauLevel(3, 0, qh.SPIN_POLARIZED_ELECTRON),
    "fermion",
    qh.QuantumHallEnergyScale(
        ELECTRONVOLT,
        1.602_176_634e-19,
        "electronvolt",
    ),
    filling=Fraction(1, 3),
    shift=3,
)
interaction = qh.HaldanePseudopotentialPlan(
    sphere,
    {1: 1.0, 3: 0.0},
    "v1-parent",
)
prepared = qh.prepare_haldane_sphere_hamiltonian(
    interaction,
    twice_projection=0,
)
spectrum = qh.HaldaneSphereSpectrumPlan(prepared, 1).evaluate()
print(
    {
        "dimension": prepared.many_body.dimension,
        "ground_energy": float(np.asarray(spectrum.energies[0])),
        "residual": float(np.asarray(spectrum.residual_norms[0])),
        "successful": bool(np.asarray(spectrum.successful)),
    }
)
