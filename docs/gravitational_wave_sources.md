# Gravitational-wave sources, provenance, and rights

## Implementation boundary

The gravitational-wave runtime is an independent Phydrax implementation against its
native JAX, Equinox, signal, reduced-order-model, integration, UQ, lifecycle, and
result-archive contracts. The repository does not redistribute Bilby source, Bilby
fixtures, LALSuite, detector strain, PSDs, calibration envelopes, waveform tables,
posterior files, sampler state, or event metadata.

The API decomposition and supported workflow were informed by Bilby's public
software and documentation:

- [Bilby source repository](https://github.com/bilby-dev/bilby)
- [Bilby documentation](https://bilby-dev.github.io/bilby/)
- Ashton et al., “BILBY: A User-friendly Bayesian Inference Library for
  Gravitational-wave Astronomy” (2019),
  [doi:10.3847/1538-4365/ab06fc](https://doi.org/10.3847/1538-4365/ab06fc)
- Romero-Shaw et al., “Bayesian inference for compact binary coalescences with
  Bilby: validation and application to the first LIGO–Virgo gravitational-wave
  transient catalogue” (2020),
  [doi:10.1093/mnras/staa2850](https://doi.org/10.1093/mnras/staa2850)

Bilby is MIT-licensed, copyright 2018 Paul D. Lasky. Its license is reproduced at
`LICENSES/BILBY-MIT.txt`. Phydrax does not import or depend on Bilby at runtime. The
bounded JSON reader is format interoperability, not embedded upstream execution.

## Scientific references

The implemented numerical contracts also follow public methods described in:

- Veitch et al., “Parameter estimation for compact binaries with ground-based
  gravitational-wave observations using the LALInference software library” (2015),
  [doi:10.1103/PhysRevD.91.042003](https://doi.org/10.1103/PhysRevD.91.042003), for
  frequency-domain Gaussian inference conventions;
- Thrane and Talbot, “An introduction to Bayesian inference in gravitational-wave
  astronomy” (2019),
  [doi:10.1017/pasa.2019.2](https://doi.org/10.1017/pasa.2019.2), for likelihood,
  evidence, marginalization, and population-inference semantics;
- Zackay, Dai, and Venumadhav, “Relative Binning and Fast Likelihood Evaluation for
  Gravitational Wave Parameter Estimation” (2018),
  [arXiv:1806.08792](https://arxiv.org/abs/1806.08792);
- Cañizares et al., “Accelerating gravitational wave parameter estimation with a
  reduced order quadrature” (2015),
  [doi:10.1103/PhysRevLett.114.071104](https://doi.org/10.1103/PhysRevLett.114.071104);
- Morisaki, “Accelerating parameter estimation of gravitational waves from compact
  binary coalescence using adaptive frequency resolutions” (2021),
  [doi:10.1103/PhysRevD.104.044062](https://doi.org/10.1103/PhysRevD.104.044062);
- Mandel, Farr, and Gair, “Extracting distribution parameters from multiple uncertain
  observations with selection biases” (2019),
  [doi:10.1093/mnras/stz896](https://doi.org/10.1093/mnras/stz896); and
- Talts et al., “Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration” (2018), [arXiv:1804.06788](https://arxiv.org/abs/1804.06788).

These references describe methods and semantics. No numerical table, source file,
fixture, posterior, benchmark result, or generated artifact from them is bundled.

## External data and provider admission

Callers are responsible for retaining, at minimum:

- detector and channel identity, acquisition interval, sample rate, GPS/UTC route,
  and any clock corrections;
- raw and conditioned strain lineage, gating, resampling, window, frequency band,
  and notch policy;
- PSD estimator, segment policy, averaging policy, source interval, and uncertainty;
- detector geometry and Earth-orientation source when a celestial-to-terrestrial
  route is used;
- calibration response identity and uncertainty convention;
- waveform provider, release, approximant, reference frequency, mode set, units,
  and parameterization;
- every local resource checksum, byte count, license, and usage permission; and
- the exact likelihood normalization, approximation qualification report, sampler
  plan, root key, and result context.

Permission evidence is separate from scientific validity. Numerical agreement does
not establish redistribution, commercial-use, or training rights. Importing a Bilby
JSON result establishes only that bounded data matched the supported schema; it does
not certify the upstream assets, inference setup, convergence, or evidence meaning.
