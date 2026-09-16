#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable signal-processing primitives with explicit axes and state."""

from phydrax._spectral import (
    DiscreteWaveletTransform,
    MultiresolutionCoefficients,
    WaveletBoundary,
    WaveletFilterBank,
)

from ._advanced import (
    cross_spectrum_and_coherence,
    design_fir,
    design_iir_sos,
    FFTConvolutionState,
    IIRDesignKind,
    multitaper_spectrum,
    MultitaperSpectrumResult,
    resample_nonuniform,
    SOSFilterPlan,
    SOSFilterResult,
    SOSFilterState,
    STFTPlan,
    StreamingFFTConvolutionPlan,
)
from ._convolution import ConvolutionMethod, ConvolutionMode, convolve
from ._fir import fir_filter, FIRFilterPlan, FIRFilterResult, FIRFilterState
from ._fourier import (
    fourier_resample,
    FourierSpectrumPlan,
    FourierSpectrumResult,
    FourierWindow,
)
from ._framing import frame, overlap_add
from ._resampling import (
    kaiser_sinc_resampling_filter,
    RationalResamplingPlan,
    RationalResamplingResult,
    RationalResamplingState,
    resample_poly,
    upfirdn,
)
from ._spectral_estimation import WelchSpectrumPlan, WelchSpectrumResult
from ._windows import (
    blackman_window,
    hamming_window,
    hann_window,
    kaiser_window,
    tukey_window,
)


__all__ = [
    "cross_spectrum_and_coherence",
    "design_fir",
    "design_iir_sos",
    "FFTConvolutionState",
    "IIRDesignKind",
    "MultitaperSpectrumResult",
    "multitaper_spectrum",
    "resample_nonuniform",
    "SOSFilterPlan",
    "SOSFilterResult",
    "SOSFilterState",
    "STFTPlan",
    "StreamingFFTConvolutionPlan",
    "ConvolutionMethod",
    "ConvolutionMode",
    "DiscreteWaveletTransform",
    "FIRFilterPlan",
    "FIRFilterResult",
    "FIRFilterState",
    "FourierSpectrumPlan",
    "FourierSpectrumResult",
    "FourierWindow",
    "MultiresolutionCoefficients",
    "RationalResamplingPlan",
    "RationalResamplingResult",
    "RationalResamplingState",
    "WaveletBoundary",
    "WelchSpectrumPlan",
    "WelchSpectrumResult",
    "WaveletFilterBank",
    "blackman_window",
    "convolve",
    "fir_filter",
    "fourier_resample",
    "frame",
    "hamming_window",
    "hann_window",
    "kaiser_sinc_resampling_filter",
    "kaiser_window",
    "overlap_add",
    "resample_poly",
    "tukey_window",
    "upfirdn",
]
