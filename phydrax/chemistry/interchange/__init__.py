#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optional molecular electronic-structure providers and interchange."""

from ._ase_calculator import (
    ASECalculatorProvider,
    ASEElectronicStateBinding,
    is_ase_calculator_available,
    PreparedASECalculator,
    require_ase_calculator,
)
from ._basis_set_exchange import (
    GaussianBasisImport,
    import_basis_set_exchange,
    is_basis_set_exchange_available,
)
from ._pyscf import (
    is_pyscf_available,
    PreparedPySCFCalculation,
    PySCFProvider,
    require_pyscf,
)
from ._pyscf_correlation import (
    is_pyscf_correlation_available,
    PySCFCoupledClusterProvider,
    require_pyscf_correlation,
)
from ._pyscf_molecular_correlation import (
    is_pyscf_molecular_correlation_available,
    PySCFMolecularCoupledClusterGradientProvider,
    require_pyscf_molecular_correlation,
)
from ._qcengine import (
    is_qcengine_available,
    PreparedQCEngineCalculation,
    QCEngineProvider,
    require_qcengine,
)
from ._qcschema import (
    electronic_calculation_to_qcelemental,
    electronic_calculation_to_qcschema,
    electronic_evaluation_from_qcschema,
    is_qcelemental_available,
    require_qcelemental,
)
from ._wannier90_hr import (
    read_wannier90_hr,
    Wannier90HRImport,
    write_wannier90_hr,
)
from ._wannier90_mmn import (
    lower_wannier90_mmn,
    read_wannier90_mmn,
    Wannier90MMNImport,
    write_wannier90_mmn,
)


__all__ = [
    "ASECalculatorProvider",
    "ASEElectronicStateBinding",
    "GaussianBasisImport",
    "PreparedASECalculator",
    "PreparedPySCFCalculation",
    "PreparedQCEngineCalculation",
    "PySCFProvider",
    "PySCFCoupledClusterProvider",
    "PySCFMolecularCoupledClusterGradientProvider",
    "QCEngineProvider",
    "Wannier90HRImport",
    "Wannier90MMNImport",
    "electronic_calculation_to_qcelemental",
    "electronic_calculation_to_qcschema",
    "electronic_evaluation_from_qcschema",
    "import_basis_set_exchange",
    "is_ase_calculator_available",
    "is_pyscf_available",
    "is_pyscf_correlation_available",
    "is_pyscf_molecular_correlation_available",
    "is_qcelemental_available",
    "is_basis_set_exchange_available",
    "lower_wannier90_mmn",
    "is_qcengine_available",
    "require_ase_calculator",
    "require_pyscf",
    "require_pyscf_correlation",
    "require_pyscf_molecular_correlation",
    "require_qcelemental",
    "read_wannier90_hr",
    "read_wannier90_mmn",
    "write_wannier90_hr",
    "write_wannier90_mmn",
    "require_qcengine",
]
