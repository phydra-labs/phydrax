#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity spectral, polarized photon Monte Carlo through triangle media."""

from importlib import import_module
from typing import Any

from ._optical_media import (
    HenyeyGreensteinScattering,
    lorenz_mie,
    LorenzMieResult,
    MieParticles,
    MieTableEvidence,
    RayleighScattering,
    SpectralOpticalMedium,
    WavelengthShiftDelay,
    WavelengthShifter,
)
from ._optical_monte_carlo import (
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalDetectorArrivals,
    OpticalDetectorResponse,
    OpticalInterfaceBranching,
    OpticalMedium,
    OpticalMonteCarloPlan,
    OpticalPhotonSource,
    OpticalPhotonState,
    OpticalScatteringSample,
    OpticalSurfaceHit,
    OpticalSurfaceInteraction,
    OpticalSurfaceModel,
    OpticalTransportResult,
    OpticalTransportStatus,
    OpticalTransportTallies,
    OpticalVarianceReduction,
    prepare_optical_monte_carlo,
    PreparedOpticalMonteCarlo,
    rotate_jones_to_scattering_frame,
    ScalarFresnelSurfaceModel,
    simulate_optical_photons,
    TissueOpticalMedium,
    UnitDetectorResponse,
)
from ._optical_sources import (
    charged_steps_from_trajectory,
    charged_steps_from_transport,
    ChargedOpticalSteps,
    CherenkovEmission,
    emit_optical_photons,
    OpticalEmissionProcess,
    OpticalEmissionStatus,
    OpticalPhotonEmission,
    OpticalPhotonSourcePlan,
    ScintillationEmission,
)
from ._optical_surfaces import OpticalSurfaceFinish, UnifiedSurfaceModel


# Photodetection emits the detector application's hit banks; resolving it on
# first use keeps the optics import independent of the applications package,
# which itself imports optics consumers during package initialization.
_PHOTODETECTION_EXPORTS = frozenset(
    {
        "OpticalPhotodetector",
        "PhotodetectionPlan",
        "PhotodetectionResult",
        "PhotodetectionStatus",
        "detect_optical_arrivals",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _PHOTODETECTION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module("._photodetection", __package__).__dict__[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PHOTODETECTION_EXPORTS)


__all__ = [
    "ChargedOpticalSteps",
    "CherenkovEmission",
    "ExplicitPhotonSource",
    "HenyeyGreensteinScattering",
    "LorenzMieResult",
    "MieParticles",
    "MieTableEvidence",
    "OpticalDetectorArrivals",
    "OpticalDetectorResponse",
    "OpticalEmissionProcess",
    "OpticalEmissionStatus",
    "OpticalInterfaceBranching",
    "OpticalMedium",
    "OpticalMonteCarloPlan",
    "OpticalPhotodetector",
    "OpticalPhotonEmission",
    "OpticalPhotonSource",
    "OpticalPhotonSourcePlan",
    "OpticalPhotonState",
    "OpticalScatteringSample",
    "OpticalSurfaceFinish",
    "OpticalSurfaceHit",
    "OpticalSurfaceInteraction",
    "OpticalSurfaceModel",
    "OpticalTransportResult",
    "OpticalTransportStatus",
    "OpticalTransportTallies",
    "OpticalVarianceReduction",
    "PhotodetectionPlan",
    "PhotodetectionResult",
    "PhotodetectionStatus",
    "PreparedOpticalMonteCarlo",
    "RayleighScattering",
    "ScalarFresnelSurfaceModel",
    "ScintillationEmission",
    "SpectralOpticalMedium",
    "TissueOpticalMedium",
    "UnifiedSurfaceModel",
    "UnitDetectorResponse",
    "WavelengthShiftDelay",
    "WavelengthShifter",
    "charged_steps_from_trajectory",
    "charged_steps_from_transport",
    "detect_optical_arrivals",
    "emit_optical_photons",
    "launch_optical_photons",
    "lorenz_mie",
    "prepare_optical_monte_carlo",
    "rotate_jones_to_scattering_frame",
    "simulate_optical_photons",
]
