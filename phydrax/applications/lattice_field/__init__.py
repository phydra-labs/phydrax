#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite lattice-field model compositions over canonical Phydrax substrates."""

from importlib import import_module

from ._continuum_study import __all__ as _continuum_study_all
from ._distributed_qcd import __all__ as _distributed_qcd_all
from ._finite_density import __all__ as _finite_density_all
from ._finite_density_methods import __all__ as _finite_density_methods_all
from ._finite_density_models import __all__ as _finite_density_models_all
from ._finite_density_table import __all__ as _finite_density_table_all
from ._finite_density_taylor import __all__ as _finite_density_taylor_all
from ._gauge_fixing import __all__ as _gauge_fixing_all
from ._hamiltonian_gauge import __all__ as _hamiltonian_gauge_all
from ._production_contracts import __all__ as _production_contracts_all
from ._qcd_ensembles import __all__ as _qcd_ensembles_all
from ._qcd_io import __all__ as _qcd_io_all
from ._qcd_observables import __all__ as _qcd_observables_all
from ._qcd_recipes import __all__ as _qcd_recipes_all
from ._qcd_transport import __all__ as _qcd_transport_all
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


_FACADE_EXPORT_MODULES = (
    "._continuum_study",
    "._distributed_qcd",
    "._finite_density",
    "._finite_density_methods",
    "._finite_density_models",
    "._finite_density_table",
    "._finite_density_taylor",
    "._gauge_fixing",
    "._hamiltonian_gauge",
    "._production_contracts",
    "._qcd_ensembles",
    "._qcd_io",
    "._qcd_observables",
    "._qcd_recipes",
    "._qcd_transport",
)


def __getattr__(name: str):
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
