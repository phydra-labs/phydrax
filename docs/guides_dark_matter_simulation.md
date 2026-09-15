# Dark-matter simulation

Phydrax treats dark matter as several physically distinct application families rather
than one state or solver. Wavefunctions, cosmological macro-particles, stochastic
body-crossing trajectories, emitted spectra, deposition histories, and halo tracks
have different measures, coordinates, and derivative contracts.

```text
external or native cosmology products
  -> collisionless PM, wave DM, or rare-scattering SIDM
  -> snapshots and halo catalogues
  -> longitudinal halo lineage and observables

incoming phase-space measure
  -> terrestrial or stellar marked-jump transport
  -> weighted surface crossings
  -> detector-facing distributions

annihilation or decay model
  -> continuum plus exact lines
  -> indirect flux or early-universe injection
  -> channel-resolved deposition and thermodynamics
```

## Support matrix

| Capability | Admitted native profile | Explicit boundary |
| --- | --- | --- |
| Wave dark matter | Flat periodic tensor Fourier grid, wave-only self-gravity, fixed increasing scale schedule | No AMR, distributed FFT, particles, gas, contact self-interaction, or QCD strings/domain walls |
| SIDM | Equal active macro masses, constant isotropic elastic cross section per mass, periodic PM, rare/large-Knudsen regime | No unequal weights, anisotropic/inelastic interactions, frequent-collision fluid limit, AMR, or distributed neighbor exchange |
| Terrestrial transport | Concentric manifest-qualified layers, physical SI body frame, elastic marked jumps, weighted spherical crossings | No implicit Earth model, detector response, or numerical cutoff interpreted as capture |
| Solar transport | Smooth manifest-qualified radial stellar profile, analytic exterior Kepler motion, interior radial gravity, thermal target rates | No implicit stellar table, plasma model, or unqualified capture criterion |
| Particle yields | Per-annihilation or per-decay continuum plus exact lines, explicit uncertainty | No bundled particle-model catalogue or experiment limits |
| Energy deposition | Species-resolved injection, state-conditioned table action, complete channel/CMB ledger, H/He thermal history | No hidden table clamping or native precision claim outside admitted assets |
| Halo lineage | Stable tracks, separate sink/descendant edges, fixed-capacity events and tracer evidence | No claim that adjacent core overlap equals a production history-based subhalo finder |
| External matter power | Exact-grid subprocess protocol returning `MatterPowerTable` | No automatic extrapolation, training, or upstream gradients |

## Coordinate and identity contracts

Cosmological particles use comoving position `x` and canonical momentum
`p = m a² dx/dt = m a v_pec`. Wave states use a complex physical-grid
wavefunction and one increasing scale factor. Transport states use Cartesian SI
position and velocity in an explicit body frame. Adapters must convert these
representations rather than relabel arrays.

Every plan binds one support identity. Particle neighborhoods and PM gravity must share
the same prepared particles and periodic box. Spectral wave evolution binds one tensor
Fourier discretization, transform normalization, zero-mode policy, and cosmology scale.
External products carry content identity, producer/version, numerical and physics policy,
artifact lineage, and a constant differentiation contract.

## Periodic wave dark matter

`WaveDarkMatterPlan.prepare` binds a `TensorSpectralDiscretization` and flat
`FLRWBackground`, producing `PreparedPeriodicWaveDarkMatter`. The state is
`WaveDarkMatterState(psi, scale_factor)` with `rho = |psi|²` in the declared code
normalization.

The prepared route applies a symmetric split:

```text
dealiased density and mean-zero Poisson source
  -> half potential phase
  -> full modal kinetic phase
  -> recomputed density and potential
  -> half potential phase
  -> transactional acceptance
```

The periodic Poisson gauge removes only the zero mode. Diagnostics report wave norm,
represented mass, kinetic/potential/total energy, Poisson and zero-mode residuals,
maximum kinetic and potential phase, de Broglie resolution, dealiasing evidence, and
accepted-step status. A rejected step rolls back the complete state and masks later
steps.

`PreparedPeriodicWaveDarkMatter.jvp` is admitted only for an accepted interior
fixed-grid trajectory. It does not make topology changes, phase unwrapping, rejected
steps, or resolution boundaries differentiable.

## Rare-scattering SIDM

`CosmologicalSIDMPlan` composes the existing PM/KDK interval with two collision half
steps. At fixed scale factor it performs the physical conversions

```text
v_pec = p / (m a)
dt = integral da / (a H(a))
W_phys = W_com / a³
P_ij = m_macro W_phys,ij |v_i - v_j| (sigma/m) dt
```

Candidate pairs come from one prepared particle neighborhood. Random streams derive
from stable endpoint identities and collision epoch. Accepted candidates are reduced to
a deterministic endpoint-disjoint set before the shared unequal-mass elastic pair map is
applied. Momentum updates return through `p' = m a v_pec'`.

Admission requires finite state, successful neighborhood construction and adaptive
smoothing, equal active macro masses, per-pair and per-particle probability bounds,
event capacity, conservative exchange, and a declared minimum mean-free-path-to-support
ratio. A failure rolls back the complete collision stage or rollout. Discrete event
selection has no pathwise-gradient claim.

## Marked transport through spherical bodies

`ProfiledElasticJumpProcess` combines a body profile and `ElasticScatteringTable` with
state-dependent target hazards and thermal/isotropic marks. `solve_jump_differential`
accepts path-specific initial states and an existing `HybridSchedulePlan`, so stochastic
thresholds compete with deterministic detector, layer, surface, and terminal guards.
The stochastic and deterministic tapes remain separate.

`LayeredTerrestrialProfile` supplies concentric material regions. Generic
Gauss--Legendre hazard integration is the reference path; piecewise-constant layers may
use the analytic intersection sum. `TerrestrialTransportPlan` returns complete event and
collision-invariant evidence plus weighted detector crossings.

`SmoothStellarRadialProfile` supplies radius, density, target number densities,
temperature, enclosed mass, and radial gravity. `SolarTransportPlan` composes analytic
exterior Kepler propagation with interior guarded marked-jump evolution and classifies
reflected, captured, escaped, and unresolved paths without conflating numerical failure.

`SurfaceCrossingMeasure` derives both flux and local-density measures from one
`WeightedSampleBatch`. The density estimator includes the crossing factor
`(v_initial/v_crossing) / |n dot vhat_crossing|`; grazing crossings and invalid support
are reported explicitly. `ObservationFlux` applies declared sphere area and exposure.

## Particle yields and indirect detection

`ParticleYieldSpectrum` keeps a continuum `SpectralField` and `ExactLineTable`
separate. `mix_particle_yields` combines exclusive channels only when branching
fractions form a valid physical partition and propagates numerical uncertainty.
`AnnihilationProcessDescriptor` and `DecayProcessDescriptor` prevent density-squared and
density-linear normalizations from being interchanged.

`annihilation_flux` consumes a `JFactor`; `decay_flux` consumes a `DFactor`.
`BinnedIndirectDetectionPlan` integrates the piecewise-linear continuum, places exact
lines into source bins, and then uses the existing `BinnedResponsePlan`. Instrument and
target uncertainty remains observable in the result.

External yield providers return `ExternalYieldProviderResult`. Their arrays are constant
with respect to upstream physical parameters unless a narrower derivative contract is
implemented and qualified.

## Exotic energy deposition

`InjectionSpectrum` preserves the provider's original `1+z` axis while storing one
canonical increasing scale-factor representation. It distinguishes photon/electron
species and per-event differential-number normalization.

`CascadeKernelProduct` is a state-conditioned energy action. Its result is an
`EnergyDepositionLedger` with deposited channel energy, escaped/propagated energy,
underflow or closure residual, and borrowed CMB energy kept separate from exotic
injection. A zero-injection epoch is a valid physical baseline, not a numerical failure.

`SpeciesResolvedThermodynamicsHistory` stores H II, He II, He III, electron fraction,
and matter temperature. `project_to_thermodynamics_history` is the only bridge to the
coarser `ThermodynamicsHistory`; it returns an `AdapterReport` declaring the lost
species and deposition semantics.

## Halo lineage and interchange

`HaloCatalog` remains a single-snapshot product. `HaloLineageProduct` owns persistent
track identity, ordered `HaloTrackSnapshot` records, a fixed-capacity
`HaloLineageEventLedger`, separate descendant and sink edges, and ranked
`HaloTracerEvidence`. `ParticleCoreLineagePlan` is a bounded adjacent-snapshot core
matcher, not a production history-space finder.

`phydrax.interchange.cosmology` supplies explicit readers for CONCEPT particle
snapshots, HBT-HERONS catalogues, and PINOCCHIO catalogue/lineage products. Readers
verify rights and exact bytes before decoding, preserve producer-specific sidecars, and
return `AdapterReport` values for every omission or semantic conversion. PINOCCHIO
products remain permanently approximation-tagged. CONCEPT imports require the caller
to declare whether a source array is peculiar velocity or canonical momentum.

## External matter-power providers

`MatterPowerEvaluationRequest` binds a physical cosmology request, exact scale-factor
and wavenumber nodes, and a `MatterPowerDescriptor`. `SubprocessMatterPowerBackend`
uses a bounded JSON/NPZ process boundary and returns `ExternalMatterPowerResult`.

`EmulatorSupportEvidence` distinguishes rectangular coordinate coverage from a
provider's complete scientific support claim. Results are rejected for grid, units,
scale, field, stage, neutrino semantics, manifest, clamping, or extrapolation mismatch.
Accepted values are stop-gradient `MatterPowerTable` data.

## Artifact admission

Every external profile, response, table, weight file, or catalogue requires a
`ReferenceArtifactManifest` before use. The manifest fixes checksum, size, license,
commercial/redistribution/training/export rights, nondimensionalization, uncertainty,
and lineage. A code license does not confer rights to separately distributed data or
trained weights.

## Differentiation summary

| Operation | Contract |
| --- | --- |
| Fixed-grid accepted wave evolution | Native smooth JVP |
| External power/yield/deposition/catalogue data | Constant |
| Transport event times, target marks, terminal outcomes | Discrete stochastic; no pathwise claim |
| SIDM pair acceptance and conflict resolution | Discrete stochastic; no pathwise claim |
| Halo grouping and lineage | Discrete constant product |
| Spectral response after a fixed yield | Differentiable only through admitted stored values/coordinates |

No straight-through estimator or silent topology derivative is used.
