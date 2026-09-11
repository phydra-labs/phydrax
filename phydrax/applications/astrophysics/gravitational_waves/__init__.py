#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native gravitational-wave data, response, likelihood, and inference contracts."""

from ._approximation import (
    LikelihoodApproximationPolicy,
    LikelihoodApproximationReport,
    LinearQuadraticCompressedLikelihood,
    QualifiedGravitationalWaveLikelihood,
    qualify_likelihood,
    qualify_likelihood_approximation,
)
from ._data import (
    DetectorNetworkData,
    DetectorStrainData,
    GravitationalWaveDataPlan,
    OneSidedPowerSpectralDensity,
    WindowKind,
)
from ._detector import (
    DetectorResponsePlan,
    DetectorResponseResult,
    InterferometerGeometry,
    SkyFrame,
)
from ._inference import (
    prepare_gravitational_wave_inference,
    PreparedGravitationalWaveInference,
)
from ._likelihood import (
    AbstractGravitationalWaveLikelihood,
    GravitationalWaveLikelihoodEvaluation,
    GravitationalWaveLikelihoodPlan,
    GravitationalWavePosteriorTerm,
)
from ._marginalization import (
    CalibrationCorrectionConvention,
    CalibrationMarginalizationPlan,
    CalibrationResponseEnsemble,
    DistanceMarginalizationPlan,
    GravitationalWaveMarginalizationEvaluation,
    GravitationalWaveMarginalizationPlan,
    GravitationalWaveMarginalizedPosteriorTerm,
    MarginalizedParameterDraws,
    PhaseMarginalizationPlan,
    reconstruct_marginalized_parameters,
    TimeMarginalizationPlan,
)
from ._multiband import prepare_multiband_likelihood
from ._parameters import (
    chirp_mass_from_component_masses,
    component_masses_from_chirp_mass_mass_ratio,
    default_extrinsic_parameters,
    detector_frame_mass,
    GravitationalWaveParameterPlan,
    mass_ratio_from_component_masses,
    source_frame_mass,
    symmetric_mass_ratio,
)
from ._relative_binning import prepare_relative_binning_likelihood
from ._roq import prepare_reduced_order_quadrature_likelihood
from ._status import gravitational_wave_status_message, GravitationalWaveStatus
from ._waveform import (
    AbstractFrequencyDomainWaveform,
    CallableFrequencyDomainWaveform,
    DerivativeLevel,
    FrequencyDomainPolarizations,
    SineGaussianWaveformPlan,
    WaveformCapabilities,
)


__all__ = [
    "AbstractGravitationalWaveLikelihood",
    "AbstractFrequencyDomainWaveform",
    "CallableFrequencyDomainWaveform",
    "CalibrationCorrectionConvention",
    "CalibrationMarginalizationPlan",
    "CalibrationResponseEnsemble",
    "DerivativeLevel",
    "DetectorNetworkData",
    "DetectorResponsePlan",
    "DetectorResponseResult",
    "DetectorStrainData",
    "DistanceMarginalizationPlan",
    "FrequencyDomainPolarizations",
    "GravitationalWaveDataPlan",
    "GravitationalWaveLikelihoodEvaluation",
    "GravitationalWaveLikelihoodPlan",
    "GravitationalWaveParameterPlan",
    "GravitationalWavePosteriorTerm",
    "GravitationalWaveMarginalizationEvaluation",
    "GravitationalWaveMarginalizationPlan",
    "GravitationalWaveMarginalizedPosteriorTerm",
    "GravitationalWaveStatus",
    "InterferometerGeometry",
    "LikelihoodApproximationPolicy",
    "LikelihoodApproximationReport",
    "LinearQuadraticCompressedLikelihood",
    "MarginalizedParameterDraws",
    "OneSidedPowerSpectralDensity",
    "PreparedGravitationalWaveInference",
    "PhaseMarginalizationPlan",
    "QualifiedGravitationalWaveLikelihood",
    "SineGaussianWaveformPlan",
    "SkyFrame",
    "WaveformCapabilities",
    "TimeMarginalizationPlan",
    "WindowKind",
    "chirp_mass_from_component_masses",
    "component_masses_from_chirp_mass_mass_ratio",
    "default_extrinsic_parameters",
    "detector_frame_mass",
    "gravitational_wave_status_message",
    "prepare_multiband_likelihood",
    "prepare_relative_binning_likelihood",
    "prepare_reduced_order_quadrature_likelihood",
    "qualify_likelihood",
    "qualify_likelihood_approximation",
    "reconstruct_marginalized_parameters",
    "mass_ratio_from_component_masses",
    "prepare_gravitational_wave_inference",
    "source_frame_mass",
    "symmetric_mass_ratio",
]
