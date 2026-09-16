#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Molecular vibrational, electronic, and Raman spectroscopy."""

from ._advanced import (
    AbstractPeriodicSpectroscopyProvider,
    AbstractRamanOpticalActivityProvider,
    CallablePeriodicSpectroscopyProvider,
    CallableRamanOpticalActivityProvider,
    PeriodicSpectroscopyResult,
    RamanOpticalActivityResult,
)
from ._electron import (
    ARPESEvidence,
    ARPESPlan,
    ARPESResult,
    PhotoemissionMatrixElementRequest,
    PhotoemissionMatrixElementResult,
    TersoffHamannEvidence,
    TersoffHamannPlan,
    TersoffHamannResult,
    VacuumLDOSRequest,
    VacuumLDOSResult,
)
from ._infrared import IRSpectrumPlan, IRSpectrumResult
from ._instrument import (
    apply_spectral_instrument,
    plan_spectral_instrument,
    prepare_spectral_instrument,
    PreparedSpectralInstrument,
    refresh_spectral_instrument,
    SpectralInstrumentEvidence,
    SpectralInstrumentKind,
    SpectralInstrumentPlan,
    SpectralInstrumentResult,
)
from ._loss import (
    ElectronEnergyLossEvidence,
    ElectronEnergyLossPlan,
    ElectronEnergyLossResult,
    MacroscopicDielectricResult,
)
from ._optical import (
    optical_dielectric_from_conductivity,
    OpticalDielectricEvidence,
    OpticalDielectricPlan,
    OpticalDielectricResult,
)
from ._periodic_vibrational import (
    periodic_ir_lines,
    periodic_raman_lines,
    periodic_vibrational_lines,
    PeriodicSpectroscopyEvidence,
    PeriodicSpectroscopyRequest,
    PeriodicSpectroscopyTensorResult,
    PeriodicVibrationalSpectroscopyPlan,
    PeriodicVibrationalSpectrumResult,
)
from ._profile import (
    SpectralAxis,
    SpectralLineShape,
    SpectralProfilePlan,
    SpectralProfileResult,
    transition_energy_axis,
)
from ._qualification import (
    material_spectroscopy_candidate_campaigns,
    material_spectroscopy_candidate_profiles,
    material_spectroscopy_support_tuples,
)
from ._raman import (
    AbstractPolarizabilityProvider,
    CallablePolarizabilityProvider,
    NativeLDAPolarizabilityProvider,
    RamanSpectrumPlan,
    RamanSpectrumResult,
)
from ._resonance import ResonanceRamanPlan, ResonanceRamanResult
from ._response import (
    SpectralResponseConvention,
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)
from ._scattering import (
    DynamicStructureFactorEvidence,
    DynamicStructureFactorPlan,
    DynamicStructureFactorResult,
    ElasticNeutronScatteringPlan,
    ElasticScatteringEvidence,
    ElasticScatteringResult,
    ElasticXRayScatteringPlan,
    XRayFormFactorRequest,
    XRayFormFactorResult,
)
from ._vertical import UVVisibleSpectrumPlan, UVVisibleSpectrumResult
from ._vibronic import (
    duschinsky_analysis,
    DuschinskyFranckCondonPlan,
    DuschinskyResult,
    FranckCondonResult,
)


__all__ = [
    "AbstractPeriodicSpectroscopyProvider",
    "AbstractPolarizabilityProvider",
    "AbstractRamanOpticalActivityProvider",
    "CallablePeriodicSpectroscopyProvider",
    "CallablePolarizabilityProvider",
    "CallableRamanOpticalActivityProvider",
    "ARPESEvidence",
    "ARPESPlan",
    "ARPESResult",
    "DynamicStructureFactorEvidence",
    "DynamicStructureFactorPlan",
    "DynamicStructureFactorResult",
    "ElasticNeutronScatteringPlan",
    "ElasticScatteringEvidence",
    "ElasticScatteringResult",
    "ElasticXRayScatteringPlan",
    "ElectronEnergyLossEvidence",
    "ElectronEnergyLossPlan",
    "ElectronEnergyLossResult",
    "DuschinskyFranckCondonPlan",
    "DuschinskyResult",
    "FranckCondonResult",
    "IRSpectrumPlan",
    "IRSpectrumResult",
    "NativeLDAPolarizabilityProvider",
    "MacroscopicDielectricResult",
    "material_spectroscopy_candidate_campaigns",
    "material_spectroscopy_candidate_profiles",
    "material_spectroscopy_support_tuples",
    "OpticalDielectricEvidence",
    "OpticalDielectricPlan",
    "OpticalDielectricResult",
    "PeriodicSpectroscopyEvidence",
    "PeriodicSpectroscopyRequest",
    "PeriodicSpectroscopyTensorResult",
    "PeriodicVibrationalSpectroscopyPlan",
    "PeriodicVibrationalSpectrumResult",
    "PhotoemissionMatrixElementRequest",
    "PhotoemissionMatrixElementResult",
    "PreparedSpectralInstrument",
    "PeriodicSpectroscopyResult",
    "RamanSpectrumPlan",
    "RamanSpectrumResult",
    "RamanOpticalActivityResult",
    "ResonanceRamanPlan",
    "ResonanceRamanResult",
    "SpectralAxis",
    "SpectralLineShape",
    "SpectralProfilePlan",
    "SpectralProfileResult",
    "SpectralInstrumentEvidence",
    "SpectralInstrumentKind",
    "SpectralInstrumentPlan",
    "SpectralInstrumentResult",
    "SpectralResponseConvention",
    "SpectralResponseEvidence",
    "SpectralResponseProduct",
    "SpectralResponseRepresentation",
    "UVVisibleSpectrumPlan",
    "UVVisibleSpectrumResult",
    "TersoffHamannEvidence",
    "TersoffHamannPlan",
    "TersoffHamannResult",
    "transition_energy_axis",
    "VacuumLDOSRequest",
    "VacuumLDOSResult",
    "XRayFormFactorRequest",
    "XRayFormFactorResult",
    "apply_spectral_instrument",
    "duschinsky_analysis",
    "optical_dielectric_from_conductivity",
    "periodic_ir_lines",
    "periodic_raman_lines",
    "periodic_vibrational_lines",
    "plan_spectral_instrument",
    "prepare_spectral_instrument",
    "refresh_spectral_instrument",
]
