#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite lattice-field model compositions over canonical Phydrax substrates."""

from ._schwinger import (
    reconstruct_schwinger_flux,
    schwinger_background_schedule,
    schwinger_charge_values,
    schwinger_gauss_residual,
    schwinger_local_background_schedule,
    schwinger_local_hamiltonian,
    schwinger_mpo,
    schwinger_observables,
    SchwingerBackgroundSchedule,
    SchwingerChainModel,
    SchwingerMPOResult,
    SchwingerObservableSet,
)
from ._z2_gauge import (
    prepare_z2_gauss_sector,
    z2_gauge_hamiltonian,
    z2_gauss_eigenvalues,
    z2_gauss_terms,
    z2_homology,
    z2_loop_operator,
    Z2GaugeModel,
    Z2GaussSector,
)


__all__ = [
    "SchwingerBackgroundSchedule",
    "SchwingerChainModel",
    "SchwingerMPOResult",
    "SchwingerObservableSet",
    "reconstruct_schwinger_flux",
    "schwinger_background_schedule",
    "schwinger_charge_values",
    "schwinger_gauss_residual",
    "schwinger_local_hamiltonian",
    "schwinger_mpo",
    "schwinger_local_background_schedule",
    "schwinger_observables",
    "Z2GaussSector",
    "Z2GaugeModel",
    "prepare_z2_gauss_sector",
    "z2_gauss_eigenvalues",
    "z2_gauss_terms",
    "z2_gauge_hamiltonian",
    "z2_homology",
    "z2_loop_operator",
]
