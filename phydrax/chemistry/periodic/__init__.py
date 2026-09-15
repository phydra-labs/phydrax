#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic electronic-structure plans and results."""

from ._electrostatics import (
    GTHProjectorChannel,
    GTHPseudopotentialPlan,
    PeriodicEwaldPlan,
    PeriodicEwaldResult,
)
from ._gamma import GammaFFTDFPlan, GammaGDFPlan, GammaSCFResult
from ._lattice import (
    anharmonic_rta_transport,
    AnharmonicTransportResult,
    lattice_thermodynamics,
    LatticeThermodynamicsResult,
    PeriodicPhononPlan,
    PhononDispersionResult,
    quasi_harmonic_thermodynamics,
    QuasiHarmonicResult,
    SupercellForceConstantPlan,
    SupercellForceConstantResult,
)
from ._many_body import BetheSalpeterPlan, DiagonalGWPlan, GWQuasiparticleResult
from ._model_scf import (
    AbstractPeriodicElectronicProvider,
    CallablePeriodicElectronicProvider,
    KPointMeshPlan,
    NativePeriodicSCFPlan,
    PeriodicAOModelPlan,
    PeriodicElectronicSectorPlan,
    PeriodicSCFResult,
)
from ._properties import (
    BandStructurePlan,
    BandStructureResult,
    berry_wannier_from_neighbor_overlaps,
    BerryWannierResult,
    defect_formation_energy,
    DefectFormationEnergyResult,
    PeriodicEnergyDerivativePlan,
    PeriodicEnergyDerivativeResult,
)
from ._reference import (
    AbstractPeriodicReferenceProvider,
    CallablePeriodicReferenceProvider,
    PeriodicElectronicReferenceResult,
    PeriodicElectronicTaskPlan,
)
from ._spin import SpinPeriodicSCFPlan, SpinPeriodicSCFResult


__all__ = [
    "AnharmonicTransportResult",
    "BetheSalpeterPlan",
    "DiagonalGWPlan",
    "GWQuasiparticleResult",
    "LatticeThermodynamicsResult",
    "PeriodicPhononPlan",
    "PhononDispersionResult",
    "QuasiHarmonicResult",
    "SupercellForceConstantPlan",
    "SupercellForceConstantResult",
    "anharmonic_rta_transport",
    "lattice_thermodynamics",
    "quasi_harmonic_thermodynamics",
    "AbstractPeriodicReferenceProvider",
    "BandStructurePlan",
    "BandStructureResult",
    "BerryWannierResult",
    "CallablePeriodicReferenceProvider",
    "DefectFormationEnergyResult",
    "GTHProjectorChannel",
    "GTHPseudopotentialPlan",
    "GammaFFTDFPlan",
    "GammaGDFPlan",
    "GammaSCFResult",
    "AbstractPeriodicElectronicProvider",
    "CallablePeriodicElectronicProvider",
    "KPointMeshPlan",
    "NativePeriodicSCFPlan",
    "PeriodicAOModelPlan",
    "PeriodicElectronicSectorPlan",
    "PeriodicElectronicReferenceResult",
    "PeriodicElectronicTaskPlan",
    "PeriodicEnergyDerivativePlan",
    "PeriodicEnergyDerivativeResult",
    "PeriodicEwaldPlan",
    "PeriodicEwaldResult",
    "PeriodicSCFResult",
    "SpinPeriodicSCFPlan",
    "SpinPeriodicSCFResult",
    "berry_wannier_from_neighbor_overlaps",
    "defect_formation_energy",
]
