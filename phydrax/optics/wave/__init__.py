#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable physical plane fields and scalar wave-optics actions."""

from ._angular_spectrum import (
    AngularSpectrumEvidence,
    AngularSpectrumPlan,
    AngularSpectrumResult,
    AngularSpectrumStatus,
    PreparedAngularSpectrum,
    propagate_angular_spectrum,
)
from ._atmosphere import *  # noqa: F403
from ._atmosphere import __all__ as _atmosphere_all
from ._coherence import coherent_mode_intensity
from ._fields import (
    IntensityPlane,
    PlaneFieldSpace,
    ScalarPlaneField,
    TangentialPlaneField,
)
from ._imaging import *  # noqa: F403
from ._imaging import __all__ as _imaging_all
from ._maxwell_adapter import (
    fourier_modal_field_to_tangential_plane,
    FourierModalPlaneAdapterResult,
    FourierModalPlaneEvidence,
    PeriodicWindowAdapterResult,
    PeriodicWindowConversionEvidence,
    TangentialElectromagneticPlane,
    tile_periodic_plane_to_finite_window,
)
from ._measurement import ideal_square_law, integrate_intensity
from ._nonlinear_response import (
    instantaneous_nonlinear_polarization,
    InstantaneousScalarSusceptibility,
    OrientedTensorSusceptibility,
)
from ._pupil import *  # noqa: F403
from ._pupil import __all__ as _pupil_all
from ._pupil_adapter import (
    PupilFieldAdapterEvidence,
    PupilFieldAdapterStatus,
    PupilToScalarFieldResult,
    sequential_pupil_to_scalar_field,
)
from ._statistical_ao import *  # noqa: F403
from ._statistical_ao import __all__ as _statistical_ao_all
from ._thin import JonesThinTransmission, ScalarThinTransmission, thin_lens
from ._unidirectional import (
    AnalyticPulseField,
    prepare_unidirectional_propagation,
    PreparedUnidirectionalPropagation,
    propagate_unidirectional,
    UnidirectionalApproximationEvidence,
    UnidirectionalPropagationPlan,
    UnidirectionalPropagationResult,
    UnidirectionalPropagationStatus,
)


__all__ = [
    "AngularSpectrumEvidence",
    "AnalyticPulseField",
    "AngularSpectrumPlan",
    "AngularSpectrumResult",
    "AngularSpectrumStatus",
    "FourierModalPlaneAdapterResult",
    "FourierModalPlaneEvidence",
    "IntensityPlane",
    "JonesThinTransmission",
    "InstantaneousScalarSusceptibility",
    "PlaneFieldSpace",
    "PreparedAngularSpectrum",
    "OrientedTensorSusceptibility",
    "PeriodicWindowAdapterResult",
    "PeriodicWindowConversionEvidence",
    "ScalarPlaneField",
    "PreparedUnidirectionalPropagation",
    "PupilFieldAdapterEvidence",
    "PupilFieldAdapterStatus",
    "PupilToScalarFieldResult",
    "ScalarThinTransmission",
    "TangentialPlaneField",
    "TangentialElectromagneticPlane",
    "UnidirectionalApproximationEvidence",
    "UnidirectionalPropagationPlan",
    "UnidirectionalPropagationResult",
    "UnidirectionalPropagationStatus",
    "coherent_mode_intensity",
    "fourier_modal_field_to_tangential_plane",
    "ideal_square_law",
    "integrate_intensity",
    "instantaneous_nonlinear_polarization",
    "prepare_unidirectional_propagation",
    "propagate_unidirectional",
    "propagate_angular_spectrum",
    "sequential_pupil_to_scalar_field",
    "thin_lens",
    "tile_periodic_plane_to_finite_window",
]

__all__ += [
    name
    for name in (
        *_atmosphere_all,
        *_imaging_all,
        *_pupil_all,
        *_statistical_ao_all,
    )
    if name not in __all__
]
