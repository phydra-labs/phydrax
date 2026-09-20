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
  transient catalog” (2020),
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

## Native numerical-relativity surrogate boundary

`PolynomialEmpiricalField` evaluates fixed-capacity integer-monomial node fits and
applies an empirical-interpolation reconstruction matrix.
`NRSurrogateResourcePolicy` bounds normalized bytes, time/mode/node/term/order/
reconstruction capacities, angular and interpolated modal products, normalization
losses, and output samples using normalized target dtypes before allocation.
`AlignedNRSurrogateArtifact` then binds the fields to a strictly increasing
geometric time grid, nonnegative-m mode set, aligned-spin and transformed fit
support, source frame and time origin, spin weight, spherical normalization,
strain/amplitude conventions, differentiation contract, and content identity.

Construction accepts caller-supplied arrays and caller-asserted provenance, then
content-binds exactly what was supplied. It never treats public provenance strings
as Phydrax release authority: `source_authenticated` is false and every mode/
polarization result remains `qualified=false`. Externally admitted model bytes are
not silently upgraded into trusted executable coefficients; a qualified future route
must execute a trusted decoder or verify a separately authenticated normalization
receipt. The structural normalization report records the bounded interpolation
derivative downgrade, not source authenticity.

`AlignedNRSurrogatePlan` maps `(q, chi1z, chi2z)` to
`(log(q), chi_hat, chi_a)`, reconstructs negative-m modes through nonprecessing
equatorial symmetry, uses the canonical spin-minus-two spherical discretization,
and exposes geometric and SI observer-time evaluations. The physical route requires
the redshifted detector-frame total mass in kilograms and luminosity distance in
meters. Requested times outside the artifact interval are zero-filled with
per-sample support/status; parameter extrapolation is refused. Separate intrinsic,
extrinsic, detector-mass-scaling, and time derivative masks exclude physical/fit
boundaries, interpolation knots, support edges, and polar-coordinate singularities;
the artifact contract does not claim stored-value or global higher-order derivatives.

This normalized array contract was informed by the useful static-shape and
mode-reconstruction patterns in
[JaxNRSur](https://github.com/kazewong/JaxNRSur) at commit
`6839d7c491b3b81b12a8739ec05d05119677bd0e`. JaxNRSur source is MIT-licensed,
copyright Kaze Wong 2023; its license is reproduced at
`LICENSES/JAXNRSUR-MIT.txt`. Phydrax does not import that runtime or reproduce its
implicit downloader, mutable HDF5 dictionaries, spline, harmonic implementation,
or FFT wrapper.

The numerical model data are separate from the source license. Zenodo record
[3348115](https://doi.org/10.5281/zenodo.3348115) declares CC-BY-4.0:
`NRHybSur3dq8.h5` is 212,935,992 bytes with catalog MD5
`b42cd577f497b1db3da14f1e4ee0ccd1` and independently observed SHA-256
`511fef79c75c41dd3c90928cf4839af378324bb591d3a7d997548dc6dfbf4d3d`;
`NRSur7dq4.h5` is 17,479,528 bytes with catalog MD5
`8e033ba4e4da1534b3738ae51549fb98`. Neither artifact is bundled. The native
polynomial-EIM contract is not a claim that either complete named model has been
ported or qualified: NRHybSur3dq8 uses stored GPR nodes, while NRSur7dq4 additionally
requires learned precessing dynamics and Wigner mode mixing.

## Waveform comparison and aligned-remnant sources

`WaveformMatchPlan` consumes the canonical `OneSidedPowerSpectralDensity`. It
evaluates a fixed-time overlap with analytic phase maximization and a discrete
canonical-grid IFFT search over time and phase. It reports canonical
`4 * delta_frequency` one-sided norms, signed lag, match/mismatch, status, and
zero-norm or normalization failures. Fixed shifts are evaluated modulo the FFT
duration while the promoted caller shift is retained. This diagnostic is not a
likelihood or a differentiable contract for the maximizing `argmax`.

`AlignedBinaryRemnantPlan` implements the UIB2016v2 nonprecessing BBH final-mass
and final-spin fit of Jiménez-Forteza et al.,
[doi:10.1103/PhysRevD.95.064024](https://doi.org/10.1103/PhysRevD.95.064024).
The admitted envelope is deliberately bounded to `1 <= q <= 8` and
`abs(chi1z), abs(chi2z) <= 0.8`; out-of-domain inputs return an explicit status
rather than extrapolated remnant data. The result preserves mass units, reports the
radiated-energy and final-mass fractions, enforces the Kerr bound, and distinguishes
interior derivative validity.

These additions were informed by
[Ripple](https://github.com/tedwards2412/ripple) at commit
`58c53024718697e0430b666ef5dd40ff3c2e029d`, MIT-licensed, copyright Adam Coogan
and Thomas Edwards 2022; see `LICENSES/RIPPLE-MIT.txt`. The UIB2016v2 coefficient
algebra is adapted under that notice, while the typed domain/status contract and
canonical-grid comparison API are Phydrax-owned. No upstream source file, runtime,
PSD data, QNM table, fixture, or phenomenological waveform coefficient table is
bundled. Ripple's IMRPhenomD, IMRPhenomXAS, IMRPhenomPv2, TaylorF2, and
IMRPhenomD_NRTidalv2 implementations remain external references, not Phydrax
support claims.

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
