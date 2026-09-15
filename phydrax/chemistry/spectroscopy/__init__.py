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
from ._infrared import IRSpectrumPlan, IRSpectrumResult
from ._profile import (
    SpectralAxis,
    SpectralLineShape,
    SpectralProfilePlan,
    SpectralProfileResult,
    transition_energy_axis,
)
from ._raman import (
    AbstractPolarizabilityProvider,
    CallablePolarizabilityProvider,
    NativeLDAPolarizabilityProvider,
    RamanSpectrumPlan,
    RamanSpectrumResult,
)
from ._resonance import ResonanceRamanPlan, ResonanceRamanResult
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
    "DuschinskyFranckCondonPlan",
    "DuschinskyResult",
    "FranckCondonResult",
    "IRSpectrumPlan",
    "IRSpectrumResult",
    "NativeLDAPolarizabilityProvider",
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
    "UVVisibleSpectrumPlan",
    "UVVisibleSpectrumResult",
    "transition_energy_axis",
    "duschinsky_analysis",
]
