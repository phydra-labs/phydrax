from fractions import Fraction

import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.units import ELECTRONVOLT


manifold = qh.MonopoleLandauLevel(5, 1, qh.SPIN_POLARIZED_ELECTRON)
sphere = qh.HaldaneSpherePlan(
    2,
    manifold,
    "fermion",
    qh.QuantumHallEnergyScale(ELECTRONVOLT, 1.602_176_634e-19, "electronvolt"),
    filling=Fraction(1, 3),
    shift=1,
)
interaction = qh.coulomb_haldane_pseudopotentials(sphere)
prepared = qh.prepare_haldane_sphere_hamiltonian(interaction, twice_projection=0)
print(
    {
        "landau_level": manifold.landau_level,
        "orbital_count": manifold.orbital_count,
        "sector_dimension": prepared.many_body.dimension,
        "channels": interaction.relative_channels,
        "accepted": bool(np.asarray(prepared.many_body.evidence.accepted)),
    }
)
