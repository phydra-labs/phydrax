#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite lattice-field model compositions over canonical Phydrax substrates."""

from ._continuum_study import *  # noqa: F403
from ._continuum_study import __all__ as _continuum_study_all
from ._distributed_qcd import *  # noqa: F403
from ._distributed_qcd import __all__ as _distributed_qcd_all
from ._finite_density import *  # noqa: F403
from ._finite_density import __all__ as _finite_density_all
from ._finite_density_methods import *  # noqa: F403
from ._finite_density_methods import __all__ as _finite_density_methods_all
from ._finite_density_models import *  # noqa: F403
from ._finite_density_models import __all__ as _finite_density_models_all
from ._finite_density_table import *  # noqa: F403
from ._finite_density_table import __all__ as _finite_density_table_all
from ._finite_density_taylor import *  # noqa: F403
from ._finite_density_taylor import __all__ as _finite_density_taylor_all
from ._gauge_fixing import *  # noqa: F403
from ._gauge_fixing import __all__ as _gauge_fixing_all
from ._hamiltonian_gauge import *  # noqa: F403
from ._hamiltonian_gauge import __all__ as _hamiltonian_gauge_all
from ._production_contracts import *  # noqa: F403
from ._production_contracts import __all__ as _production_contracts_all
from ._qcd_ensembles import *  # noqa: F403
from ._qcd_ensembles import __all__ as _qcd_ensembles_all
from ._qcd_io import *  # noqa: F403
from ._qcd_io import __all__ as _qcd_io_all
from ._qcd_observables import *  # noqa: F403
from ._qcd_observables import __all__ as _qcd_observables_all
from ._qcd_recipes import *  # noqa: F403
from ._qcd_transport import *  # noqa: F403
from ._qcd_transport import __all__ as _qcd_transport_all
from ._qcd_recipes import __all__ as _qcd_recipes_all
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
    *_continuum_study_all,
    *_distributed_qcd_all,
    *_finite_density_all,
    *_finite_density_methods_all,
    *_finite_density_models_all,
    *_finite_density_table_all,
    *_finite_density_taylor_all,
    *_gauge_fixing_all,
    *_hamiltonian_gauge_all,
    *_production_contracts_all,
    *_qcd_ensembles_all,
    *_qcd_io_all,
    *_qcd_observables_all,
    *_qcd_recipes_all,
    *_qcd_transport_all,
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
