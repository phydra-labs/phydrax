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
from ._coupled_mode import (
    BidirectionalCoupledModeEvidence,
    BidirectionalCoupledModePlan,
    BidirectionalCoupledModeResult,
    BidirectionalCoupledModeStatus,
    CoupledModeBoundary,
    prepare_bidirectional_coupled_mode,
    PreparedBidirectionalCoupledMode,
    solve_bidirectional_coupled_mode,
)
from ._cylindrical import (
    CylindricalAnalyticPulseField,
    CylindricalUnidirectionalPropagationEvidence,
    CylindricalUnidirectionalPropagationPlan,
    CylindricalUnidirectionalPropagationResult,
    CylindricalUnidirectionalPropagationStatus,
    prepare_cylindrical_unidirectional_propagation,
    PreparedCylindricalUnidirectionalPropagation,
    propagate_cylindrical_unidirectional,
)
from ._envelope import (
    analytic_field_to_envelope,
    envelope_to_analytic_field,
    GaussianPulseEnvelopeEvidence,
    GaussianPulseEnvelopePlan,
    GaussianPulseEnvelopeResult,
    GaussianPulseEnvelopeStatus,
    prepare_gaussian_pulse_envelope,
    prepare_pulse_envelope_bridge,
    PreparedGaussianPulseEnvelope,
    PreparedPulseEnvelopeBridge,
    PulseEnvelopeBridgeEvidence,
    PulseEnvelopeBridgePlan,
    PulseEnvelopeBridgeResult,
    PulseEnvelopeBridgeStatus,
    PulseEnvelopeField,
    PulseEnvelopePolarization,
    sample_gaussian_pulse_envelope,
)
from ._fields import (
    IntensityPlane,
    PlaneFieldSpace,
    ScalarPlaneField,
    TangentialPlaneField,
)
from ._fresnel import (
    DirectFresnelPlan,
    FresnelPropagationEvidence,
    FresnelPropagationResult,
    FresnelPropagationStatus,
    prepare_direct_fresnel,
    PreparedDirectFresnel,
    propagate_direct_fresnel,
)
from ._imaging import *  # noqa: F403
from ._imaging import __all__ as _imaging_all
from ._material_response import (
    DelayedRamanResponsePlan,
    DrudePlasmaEvaluation,
    DrudePlasmaResponsePlan,
    IonizingDrudeResponsePlan,
    MultiphotonIonizationRatePlan,
    PreparedDelayedRamanResponse,
    PreparedDrudePlasmaResponse,
    PreparedIonizingDrudeResponse,
)
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
    AbstractCarrierResolvedResponse,
    AnalyticPulseField,
    CarrierResolvedResponseEvaluation,
    CarrierResolvedResponseEvidence,
    CarrierResolvedResponseStatus,
    instantaneous_nonlinear_polarization,
    InstantaneousScalarSusceptibility,
    OrientedTensorSusceptibility,
    prepare_carrier_resolved_response,
    PreparedCarrierResolvedResponse,
    PulseResponseLedger,
)
from ._pulse_time import PulseTimeSpace, PulseTimeTopology
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
    prepare_unidirectional_propagation,
    PreparedUnidirectionalPropagation,
    propagate_unidirectional,
    UnidirectionalApproximationEvidence,
    UnidirectionalPropagationPlan,
    UnidirectionalPropagationResult,
    UnidirectionalPropagationStatus,
)


__all__ = [
    "AbstractCarrierResolvedResponse",
    "AnalyticPulseField",
    "AngularSpectrumEvidence",
    "AngularSpectrumPlan",
    "AngularSpectrumResult",
    "AngularSpectrumStatus",
    "BidirectionalCoupledModeEvidence",
    "BidirectionalCoupledModePlan",
    "BidirectionalCoupledModeResult",
    "BidirectionalCoupledModeStatus",
    "CarrierResolvedResponseEvaluation",
    "CarrierResolvedResponseEvidence",
    "CarrierResolvedResponseStatus",
    "CoupledModeBoundary",
    "CylindricalAnalyticPulseField",
    "CylindricalUnidirectionalPropagationEvidence",
    "CylindricalUnidirectionalPropagationPlan",
    "CylindricalUnidirectionalPropagationResult",
    "CylindricalUnidirectionalPropagationStatus",
    "DelayedRamanResponsePlan",
    "DirectFresnelPlan",
    "DrudePlasmaEvaluation",
    "DrudePlasmaResponsePlan",
    "FourierModalPlaneAdapterResult",
    "FourierModalPlaneEvidence",
    "FresnelPropagationEvidence",
    "FresnelPropagationResult",
    "FresnelPropagationStatus",
    "GaussianPulseEnvelopeEvidence",
    "GaussianPulseEnvelopePlan",
    "GaussianPulseEnvelopeResult",
    "GaussianPulseEnvelopeStatus",
    "IntensityPlane",
    "InstantaneousScalarSusceptibility",
    "IonizingDrudeResponsePlan",
    "JonesThinTransmission",
    "OrientedTensorSusceptibility",
    "MultiphotonIonizationRatePlan",
    "PeriodicWindowAdapterResult",
    "PeriodicWindowConversionEvidence",
    "PlaneFieldSpace",
    "PreparedAngularSpectrum",
    "PreparedBidirectionalCoupledMode",
    "PreparedCarrierResolvedResponse",
    "PreparedCylindricalUnidirectionalPropagation",
    "PreparedDelayedRamanResponse",
    "PreparedDirectFresnel",
    "PreparedDrudePlasmaResponse",
    "PreparedGaussianPulseEnvelope",
    "PreparedPulseEnvelopeBridge",
    "PreparedIonizingDrudeResponse",
    "PreparedUnidirectionalPropagation",
    "PulseEnvelopeBridgeEvidence",
    "PulseEnvelopeBridgePlan",
    "PulseEnvelopeBridgeResult",
    "PulseEnvelopeBridgeStatus",
    "PulseEnvelopeField",
    "PulseEnvelopePolarization",
    "PulseResponseLedger",
    "PulseTimeSpace",
    "PulseTimeTopology",
    "PupilFieldAdapterEvidence",
    "PupilFieldAdapterStatus",
    "PupilToScalarFieldResult",
    "ScalarPlaneField",
    "ScalarThinTransmission",
    "TangentialElectromagneticPlane",
    "TangentialPlaneField",
    "UnidirectionalApproximationEvidence",
    "UnidirectionalPropagationPlan",
    "UnidirectionalPropagationResult",
    "UnidirectionalPropagationStatus",
    "analytic_field_to_envelope",
    "coherent_mode_intensity",
    "envelope_to_analytic_field",
    "fourier_modal_field_to_tangential_plane",
    "ideal_square_law",
    "instantaneous_nonlinear_polarization",
    "integrate_intensity",
    "prepare_bidirectional_coupled_mode",
    "prepare_carrier_resolved_response",
    "prepare_cylindrical_unidirectional_propagation",
    "prepare_direct_fresnel",
    "prepare_gaussian_pulse_envelope",
    "prepare_pulse_envelope_bridge",
    "prepare_unidirectional_propagation",
    "propagate_angular_spectrum",
    "propagate_direct_fresnel",
    "propagate_cylindrical_unidirectional",
    "propagate_unidirectional",
    "sample_gaussian_pulse_envelope",
    "sequential_pupil_to_scalar_field",
    "solve_bidirectional_coupled_mode",
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
