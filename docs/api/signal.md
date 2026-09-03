# Ordered series and signal processing

`phydrax.series` is the neutral substrate for fixed-capacity numerical data on
one ordered scalar coordinate. A `SeriesSupport` owns shared or per-series
coordinates, node validity, and adjacent-edge connectivity. A `SampledSeries`
attaches a numerical PyTree explicitly to nodes or edges. Inactive edges are
hard component boundaries: pair views and reconstructions never cross them.

Reconstruction is always explicit. Nearest, previous-value, linear, local
cubic-Hermite, and interval-held policies declare their bounds, knot side,
derivative order, and causal capability. They do not infer Euclidean geometry
for a dynamics state or erase trajectory, stochastic, solver, or physical
provenance.

## Ordered scalar series

::: phydrax.series.SeriesSupport

::: phydrax.series.SampledSeries

::: phydrax.series.SeriesPairView

::: phydrax.series.SampledSeriesReconstruction

::: phydrax.series.SeriesEvaluation

## Windows

::: phydrax.signal.hann_window

::: phydrax.signal.hamming_window

::: phydrax.signal.blackman_window

::: phydrax.signal.kaiser_window

::: phydrax.signal.tukey_window

## Framing

::: phydrax.signal.frame

::: phydrax.signal.overlap_add

## Convolution and FIR filtering

::: phydrax.signal.convolve

::: phydrax.signal.fir_filter

::: phydrax.signal.FIRFilterPlan

::: phydrax.signal.FIRFilterState

::: phydrax.signal.FIRFilterResult

## Rate conversion

::: phydrax.signal.upfirdn

::: phydrax.signal.kaiser_sinc_resampling_filter

::: phydrax.signal.resample_poly

::: phydrax.signal.RationalResamplingPlan

::: phydrax.signal.RationalResamplingState

::: phydrax.signal.RationalResamplingResult

## Periodic Fourier resampling

::: phydrax.signal.fourier_resample

## Fixed wavelet transforms

::: phydrax.signal.WaveletFilterBank

::: phydrax.signal.DiscreteWaveletTransform

::: phydrax.signal.MultiresolutionCoefficients
