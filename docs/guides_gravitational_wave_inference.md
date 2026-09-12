# Gravitational-wave inference

Phydrax provides a native gravitational-wave inference path under
`phydrax.applications.astrophysics.gravitational_waves`. It composes immutable
frequency-domain data, detector geometry, waveform capabilities, normalized
likelihoods, existing `phydrax.uq` posterior problems and samplers, reduced-order
models, portable result archives, and explicit qualification evidence. It does not
introduce a second sampler runtime.

## Scope

The current surface covers:

- canonical one-sided real-FFT strain and PSD preparation;
- fixed interferometer geometry, antenna factors, and geocentric delays;
- native sine-Gaussian signals and declared callable frequency-domain waveforms;
- absolute and noise-relative network likelihood semantics;
- native nested-sampling preparation, posterior context, and result export;
- phase, distance, time, and discrete calibration marginalization with keyed
  joint reconstruction;
- held-out-qualified relative binning, empirical-interpolation reduced-order
  quadrature, and multiband compression;
- posterior reweighting, hierarchical event recycling, selection effects, and
  simulation-based calibration; and
- bounded, data-only import of current plain Bilby JSON results.

It does not bundle a waveform provider, detector calibration file, event data,
LALSuite, Bilby, or a detector-frame download client. Those remain external
assets or providers admitted through explicit provenance and callable boundaries.

## Data and normalization

`OneSidedPowerSpectralDensity` requires the exact `rfftfreq` grid implied by
`sample_count` and `sample_interval`. Active PSD values are finite and strictly
positive. DC and, for even sample counts, Nyquist are excluded from active support.
`DetectorStrainData` uses the same grid and stores inactive bins as zero. A
`DetectorNetworkData` requires every detector to share sample support and GPS start
time.

For duration `T`, one-sided PSD `S`, and retained window-power correction `w`, the
proper-complex bin variance is `T S w / 2`. Every active bin contributes
`-|d-h|² / variance - log(π variance)`. Therefore
`GravitationalWaveLikelihoodPlan.log_probability` is an absolute normalized log
likelihood. `log_likelihood_ratio` subtracts the normalized noise likelihood and is
not interchangeable with absolute evidence. Invalid waveform support produces
negative infinity with explicit evaluation status so posterior samplers can reject a
proposal without aborting a run.

Time-series preparation is explicit:

```python
import jax.numpy as jnp
import phydrax as phx

physics = phx.applications.astrophysics
gw = physics.gravitational_waves
provenance = physics.ObservationDataProvenance.native("local-segment")

plan = gw.GravitationalWaveDataPlan(
    sample_count=4096,
    sample_interval=1.0 / 1024.0,
    start_time_gps=0.0,
    minimum_frequency=20.0,
    maximum_frequency=480.0,
    window="tukey",
    tukey_alpha=0.2,
)
psd = gw.OneSidedPowerSpectralDensity(
    plan.frequencies,
    jnp.ones_like(plan.frequencies),
    provenance,
    sample_count=plan.sample_count,
    sample_interval=plan.sample_interval,
)
strain = plan.prepare("D1", jnp.zeros(4096), psd, provenance)
```

A PSD can be supplied directly or estimated through the shared
`phydrax.signal.WelchSpectrumPlan`. Welch exposes Hann or Tukey windows and mean or
bias-corrected median averaging. PSD estimation and event-strain preparation retain
separate provenance; no PSD is inferred from frequency-domain strain.

## Detector response and waveform boundaries

`InterferometerGeometry` owns one vertex and two orthonormal arm directions in an
Earth-fixed Cartesian frame. `DetectorResponsePlan` binds an ordered detector
network to matching geometries. Its ITRS route evaluates antenna factors and delays
without an external time service. The GCRS route requires explicit GPS-to-UTC and
Earth-orientation plans; missing orientation data is an unsupported contract, not an
identity fallback.

`SineGaussianWaveformPlan` is the native qualification waveform. External waveform
implementations enter through `CallableFrequencyDomainWaveform` together with a
`WaveformCapabilities` declaration. Capabilities identify polarization order,
parameterization, arbitrary-frequency support, transformation behavior, phase and
distance coordinates, and derivative level. The callable returns polarizations only;
detector projection, delay phase, calibration, likelihood normalization, and status
remain Phydrax-owned.

`GravitationalWaveParameterPlan` makes the physical parameter PyTree, priors,
continuous nested coordinates, finite supports, periodic coordinates, bijectors,
waveform mapping, extrinsic mapping, and derived-parameter mapping one coherent
contract. Mass conversion helpers distinguish component, chirp, symmetric-ratio,
source-frame, and detector-frame coordinates. They reject nonphysical inputs instead
of sorting or clipping masses.

## Native posterior execution

`prepare_gravitational_wave_inference` returns a
`PreparedGravitationalWaveInference` containing:

- a normalized `PosteriorProblem`;
- the `GravitationalWavePosteriorTerm` used by that problem;
- a topology-compatible `NestedSamplingPlan`; and
- a `UQResultContext` binding analysis, parameterization, likelihood, data,
  provider, approximation, and normalization identities.

The prepared posterior works with the ordinary Phydrax samplers when their support
and differentiability requirements are satisfied. The runnable one-dimensional
nested-sampling example performs an actual injection recovery and writes a portable
archive:

```console
python examples/gravitational_wave_inference.py
```

Pass the prepared context when exporting another supported result:

```python
path = phx.uq.export_result(
    result,
    "event.phxresult",
    context=prepared.result_context,
)
```

The archive stores the context separately from numerical result fields. It does not
upgrade a likelihood ratio to absolute evidence.

## Marginalization and reconstruction

`GravitationalWaveMarginalizationPlan` combines any supported subset of:

- `PhaseMarginalizationPlan`, using the harmonic phase structure and a normalized
  periodic measure;
- `DistanceMarginalizationPlan`, using fixed Gauss-Legendre nodes transformed by an
  explicit distance prior;
- `TimeMarginalizationPlan`, using a declared finite uniform time grid; and
- `CalibrationMarginalizationPlan`, using a finite weighted ensemble of complex
  detector corrections.

Distance and time routes are fixed quadrature, not adaptive integration. Calibration
corrections declare whether they multiply the response or data. Every marginalized
coordinate is omitted from the sampled parameter tree.
`reconstruct_marginalized_parameters` draws all omitted coordinates jointly from the
same conditional discrete measure. One root key and fixed fold-in scheme make replay
exact; the returned masks and status preserve failed conditional rows.

## Qualified acceleration

Approximate likelihoods are never selected by name alone. Every preparation function
takes held-out physical parameter values and stable validation IDs, evaluates the
exact and candidate log likelihoods, and applies a
`LikelihoodApproximationPolicy`. A failed maximum or RMS error gate raises before a
`QualifiedGravitationalWaveLikelihood` can be constructed.

- `prepare_relative_binning_likelihood` builds linear and quadratic sufficient
  statistics relative to an explicit fiducial waveform.
- `phydrax.rom.prepare_empirical_interpolation` converts a trained real POD artifact
  into selected nodes and a reconstruction operator.
  `prepare_reduced_order_quadrature_likelihood` consumes separate linear and
  quadratic prepared interpolations.
- `prepare_multiband_likelihood` uses declared band edges and integer strides, then
  qualifies the resulting interpolation against exact evaluations.

Qualification proves only the supplied validation set under the recorded data,
waveform, and policy identities. It is not a global error certificate. Calibration
callbacks are deliberately excluded from the reduced linear/quadratic routes because
they alter the sufficient statistics.

## Reweighting, populations, and calibration

The general UQ layer owns cross-event operations:

- `PosteriorReweightingPlan` retains old and new sample log densities, normalized
  target weights, normalizer ratio, support-loss count, effective sample size, and
  maximum normalized weight;
- `EventPosterior` binds a weighted posterior to its original sampling prior,
  evidence kind, provider, inference method, approximation, and source ESS;
- `prepare_population_sample_batch` pads heterogeneous event sample counts behind an
  explicit mask without treating recycled draws as independent;
- `PopulationPosteriorTerm` evaluates event importance integrals and an optional
  `SelectionInjectionSet`; `PoissonPopulationPosteriorTerm` additionally owns a
  population-rate term; and
- `SimulationCalibrationPlan` records component paths, tie policy, histogram bins,
  minimum valid cases, and multiple-testing correction. Failed analyses remain in the
  result by case ID instead of disappearing from rank statistics.

These operations require a shared physical population parameterization. Event
posterior samples are dependent weighted measures; sample count is not evidence of
independence or effective sample size.

## Bilby JSON import

`phydrax.interchange.read_bilby_result_json` is a host-only migration boundary for
current plain JSON results. The caller supplies a trusted root and `ResourceLimits`.
The importer accepts declared sampled columns plus optional weights, `log_prior`,
`log_likelihood`, and evidence metadata. It normalizes weights and returns an
`ImportedBilbyResult`, which can expose a `WeightedSampleTarget` or, when original
sample-prior values are available, an `EventPosterior`.

Pickle, HDF5, Python object tags, prior reconstruction, callable reconstruction,
legacy arbitrary object encodings, and sampler execution are unsupported. Imported
metadata remains external provenance; import does not certify the upstream run or
convert omitted constants into evidence.

## Qualification and benchmarks

Run the complete native injection recovery plus exact, marginalized, compressed,
population, and calibration checks with:

```console
python tools/gravitational_wave_qualification.py
```

The UQ benchmark matrix registers seven `gravitational_wave_*` scenarios. Run the
release profiles with repeated `--scenario` selections or use the qualification tool
for the focused smoke set. Reports retain frequency-bin, reduced-node, event, sample,
policy, and qualification identities next to accuracy and timing metrics.

See [Gravitational-wave API](api/applications/gravitational_waves.md) and
[gravitational-wave sources and rights](gravitational_wave_sources.md).
