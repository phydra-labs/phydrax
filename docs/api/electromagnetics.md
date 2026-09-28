# Electromagnetics

`phydrax.electromagnetics` owns electromagnetic microphysics that does not
require a field solve.

## Trajectory radiation

`TrajectoryRadiationPlan` computes the vacuum far-field spectrum radiated by
sampled charged-particle trajectories. All radiation routes share one Fourier
convention: phasors `exp(−iωt)`, transient spectra
`F(ω) = ∫ f(t) exp(+iωt) dt` over observer time `τ = t − n·r/c`, and the
one-sided spectral energy `d²W/(dω dΩ) = ε₀ c |r Ẽ|² / π` for `ω > 0`. The
field is the acceleration form of the Liénard–Wiechert far field,
`r Ẽ(ω) = q/(4π ε₀ c) ∫ d/dt[n × (n × β)/κ] exp(iωτ) dt`, with prefactors from
the bound `ElectromagneticScaleContract` and the retardation factor evaluated
in the cancellation-free form `κ = (1/γ² + |n × β|²)/(1 + n·β)`.

`ChargedTrajectory` holds lanes of float64 samples: times (one shared vector
`[T]` or per-lane `[T, P]`), positions, proper velocities `u = γv`, optional
proper accelerations `du/dt`, per-lane charges and multiplicities, an
`active[T, P]` mask, and persistent `(hi, lo)` identities. Every non-float64
numerical input is refused with `TypeError`. `RadiationObserverPlan` holds unit
directions and the right-handed polarization basis `e1 = normalize(a − (a·n)n)`,
`e2 = n × e1` for a reference axis `a`.

Routes (`TrajectoryRadiationRoute`):

- `"segment-exact"`: piecewise-constant `β` per segment; each segment
  contributes `a_j Δτ_j exp(iωτ_mid) sinc(ωΔτ_j/2)` with
  `a_j = n × (n × β_j)/(1 − n·β_j)`, closed by activity-boundary terms so the
  result equals the node jump sum `Σ Δa_j exp(iωτ_j)` exactly.
- `"segment-hermite"`: requires proper accelerations; a quintic Hermite
  reconstruction of the position and a cubic one of `u` are integrated in the
  acceleration form with a `quadrature_order` Gauss–Legendre rule per segment
  (fourth-order convergent).
- `"node-gridded"`: the node jump sum evaluated with one batched 1-D Type-3
  nonuniform Fourier transform per direction over the declared
  `observer_time_window`; the absolute error floor `ε · |q m| Σ|Δa_j|` is
  reported per direction, and nodes outside the window raise
  `UNSUPPORTED_NODE`. When `ω_max · window ≤ 1` the sum is evaluated directly.

Jumps are attributed only to interior nodes between active segments: window
edges and activity changes contribute nothing, which is the spectrum of a
trajectory continued with uniform motion. `emission="complete"` declares that
the window holds the whole emission; acceleration at the window edge then
contradicts the declaration (`WINDOW_EDGE_ACCELERATION`, not `resolved`).

Coherence (`RadiationCoherence`): `"coherent"` sums multiplicity-weighted lane
fields; `"incoherent"` sums per-particle coherency; `"gaussian-form-factor"`
and `"tabulated-form-factor"` combine `N(1 − |F|²)|R|² + N²|F|²|R|²` with
`|F|² = exp(−ω² Σ nᵢ² σᵢ²/c²)` or a tabulated magnitude per frequency. The
result reports `field_spectrum[F, D, 2]` (coherent lane sum in `(e1, e2)`),
`coherency[F, D, 2, 2] = ⟨R_i R_j*⟩`, `spectral_energy[F, D]`, and Stokes
`(I, Q, U, V)` with `U = 2 Re(R1 R2*)`, `V = −2 Im(R1 R2*)`; `V > 0` is a field
rotating from `e1` toward `e2`. `waveform(result, times)` synthesizes the
observer-time field of a coherent result by trapezoid inversion of the
one-sided spectrum on the plan's frequency grid.

`TrajectoryRadiationEvidence` reports `status` bits
(`TrajectoryRadiationStatus`: `UNRESOLVED_PHASE` when `ω Δτ` per segment
exceeds one radian, or the Hermite quadrature order; `UNRESOLVED_AMPLITUDE`
when one jump exceeds half of the peak amplitude; `WINDOW_EDGE_ACCELERATION`;
`ACTIVITY_TRANSITION`; `UNSUPPORTED_NODE`; `NONMONOTONE_TIME`;
`LANE_MISMATCH`; `NONFINITE`), `finite`, `resolved`, `derivative_valid[F, D]`
(false for unresolved frequencies and everywhere after an activity
transition, a discrete event), the minimum retardation factor, maximum phase
and relative amplitude increments, the window-edge rate, segments used, the
gridded floor, and the static `resource_estimate`.

Execution is bounded: lanes run in `lax.scan` chunks of
`TrajectoryRadiationResources.particle_chunk` vectorized lanes, segments in
`lax.scan` blocks of `segment_block`, and work or state above the declared
byte limits raises `TrajectoryRadiationResourceError` before execution. The
prepared plan exposes streaming `initialize`/`accumulate`/`finalize` over
consecutive time chunks of the same lanes; the streamed result equals
`evaluate` of the whole trajectory. Every route is differentiable with respect
to trajectory samples.

See [Charged-particle radiation](../guides_charged_particle_radiation.md) for
worked examples and [Electromagnetic radiation sources](../electromagnetic_radiation_sources.md)
for references.

::: phydrax.electromagnetics.TrajectoryRadiationPlan

---

::: phydrax.electromagnetics.PreparedTrajectoryRadiation

---

::: phydrax.electromagnetics.ChargedTrajectory

---

::: phydrax.electromagnetics.RadiationObserverPlan

---

::: phydrax.electromagnetics.TrajectoryRadiationResult

---

::: phydrax.electromagnetics.TrajectoryRadiationEvidence

---

::: phydrax.electromagnetics.TrajectoryRadiationStatus

---

::: phydrax.electromagnetics.TrajectoryRadiationState

---

::: phydrax.electromagnetics.TrajectoryRadiationResources

---

::: phydrax.electromagnetics.TrajectoryRadiationResourceEstimate

---

::: phydrax.electromagnetics.TrajectoryRadiationResourceError

## SRW single-electron oracle

`run_srw` runs SRW (Chubar and Elleaume, EPAC 1998; EPICS Open License;
validated against srwpy 4.2.1) in a caller-pinned Python interpreter
(`SRWProvider`) as an external oracle for `TrajectoryRadiationPlan`.
`srw_input` translates the plan (uniform angular frequencies, forward
directions, one electron) and one source into SRW's driver and JSON deck:
either a `ChargedTrajectory` lane sampled at uniform lab times, whose
positions and `β` SRW integrates directly, or an `SRWFieldMapSource` (a
`FieldMapTrackingPlan` of static magnetic elements and a one-electron
`AcceleratorBunch` at `ζ = 0`) for which SRW integrates its own trajectory
through the beamline sampled onto a uniform table (`"tabulated"`) or through
its ideal undulator (`"ideal-undulator"`). Anything outside that subset is
refused with `ValueError` before SRW runs. Photon energies are requested with
SRW's own wavenumber constant so that SRW radiates at exactly the plan's `ω`.

`read_srw_output` converts SRW's field at the point `o + D n` into the far-field
convention above: `|r Ẽ| = D √(10⁹ π ħ e/(I ε₀ c)) |E|` from SRW's
`√(photons/s/0.1%bw/mm²)` normalization, `E_z` from `n · Ẽ = 0`, projection
onto `(e1, e2)`, and removal of SRW's paraxial clock and Fresnel phase so the
phase refers to the trajectory's retarded time. `SRWSpectrumResult` carries
the `SRWFieldSpectrum` in the plan's units, the pinned version, interpreter
digest, license, output digest, and an `AdapterReport` whose losses list
every approximation (finite distance, paraxial phase, reconstructed `E_z`,
single precision, trajectory or field representation, missing evidence).
Against an X1a-tracked ten-period undulator SRW agrees with
`TrajectoryRadiationPlan` to a few 10⁻³ in spectral energy and field.

::: phydrax.electromagnetics.run_srw

---

::: phydrax.electromagnetics.srw_input

---

::: phydrax.electromagnetics.read_srw_output

---

::: phydrax.electromagnetics.SRWProvider

---

::: phydrax.electromagnetics.SRWFieldMapSource

---

::: phydrax.electromagnetics.SRWFieldSpectrum

---

::: phydrax.electromagnetics.SRWSpectrumResult

## Near-zone Liénard–Wiechert fields

`LienardWiechertFieldPlan` evaluates the retarded fields of the point-charge
lanes of a `ChargedTrajectory` at observer events `[O, 4] = (t, x, y, z)`:

```text
E = q/(4π ε₀) [ (n − β)(1 − β²)/(κ³ R²) + n × ((n − β) × β̇)/(c κ³ R) ],   B = n × E / c
```

returning `electric_field`, `magnetic_field`, the `velocity_field` and
`acceleration_field` parts of `E`, and `retarded_times[O, P]`. The retarded
condition is monotone for subluminal lanes: a fixed `⌈log₂(T − 1)⌉`-step
index bisection selects the segment and the native bracketed `scalar_root`
(TOMS748) solves it on the Hermite reconstruction selected by
`LienardWiechertInterpolation` (`"hermite-cubic"`: positions and velocities;
`"hermite-quintic"`: also proper accelerations), with implicit derivatives.
`LienardWiechertHistory` decides retarded times before the first sample
(`"refuse"` or `"inertial-extrapolation"`); pairs within `exclusion_radius`
are excluded and their charge reported. Unsupported observers carry NaN fields.
`LienardWiechertEvidence` reports `LienardWiechertStatus` bits per observer
and per pair, support, resolution, derivative-valid masks, excluded and absent
charge, retardation factors, root residuals, Lorentz-consistency mismatch, and
the static resource estimate; `LienardWiechertResources` bounds observer and
particle chunks and refuses oversized work with `LienardWiechertResourceError`.

::: phydrax.electromagnetics.LienardWiechertFieldPlan

---

::: phydrax.electromagnetics.PreparedLienardWiechertField

---

::: phydrax.electromagnetics.LienardWiechertFieldResult

---

::: phydrax.electromagnetics.LienardWiechertEvidence

---

::: phydrax.electromagnetics.LienardWiechertStatus

---

::: phydrax.electromagnetics.LienardWiechertResources

---

::: phydrax.electromagnetics.LienardWiechertResourceEstimate

---

::: phydrax.electromagnetics.LienardWiechertResourceError

## Uniform-motion fields

`UniformMotionFieldPlan` evaluates the exact `exp(-iωt)` transform of the field
of a point (3-D) or line (2-D) charge in uniform motion through a homogeneous,
isotropic, passive, dispersive `UniformMotionMedium` (absolute `ε(ω)`, `μ(ω)`):
`K₀, K₁(sρ)` with `s² = ω²/v² − ω²εμ` for the point charge and `exp(−s|η|)` for
the line charge. The branch is `Im k_ρ ≥ 0` (`s = −i k_ρ`); above the Cherenkov
threshold this is the outgoing `H⁽¹⁾(k_ρ ρ)` wave, and lossless negative-index
media take the passive limit (reversed phase flow). `radial_energy_flux` gives
the one-sided `d²W/(dω dl)` through a cylinder (point) or slab (line), which is
the Frank–Tamm spectrum above threshold and zero below; `UniformMotionEvidence`
reports the branch, `γβλ` bound-field reach, and cone angle.

::: phydrax.electromagnetics.UniformMotionGeometry

---

::: phydrax.electromagnetics.UniformMotionMedium

---

::: phydrax.electromagnetics.UniformMotionFieldPlan

---

::: phydrax.electromagnetics.UniformMotionField

---

::: phydrax.electromagnetics.UniformMotionEvidence

## Frequency-domain Maxwell systems

::: phydrax.electromagnetics.MaxwellFrequencySystem

---

::: phydrax.electromagnetics.MaxwellFrequencyResult

## Cold magnetized plasma

`ColdPlasmaDielectric` binds an `ElectromagneticScaleContract` to a
multi-species cold plasma: number densities, signed charge numbers, masses in
electron-mass units, the static field `B₀` and per-species collision
frequencies. It returns the Stix parameters `S`, `D`, `P`, `R`, `L` (complex
when collisions are present, with `ω → ω + iν_s` in each species' momentum
equation so that `Im n² > 0` is absorption under the `exp(−iωt)` phasor
convention), the dielectric tensor in the Stix frame, and the two roots of the
Stix biquadratic `A n⁴ − B n² + C = 0` at real angular frequency `ω` and
wave-normal angle `θ` to `B₀`.

`refractive_indices` reports both roots with explicit branch identity.
`parallel_mode` labels each root by numerical continuation in `θ` from the
`θ = 0` closed forms `n² = R` and `n² = L`; `perpendicular_mode` labels it by
continuation from the `θ = π/2` closed forms `n² = P` (ordinary) and
`n² = RL/S` (extraordinary). The continuation follows the pole-free
discriminant root `F` of the biquadratic, so resonances do not obstruct it;
each path reports the minimum relative root separation and the largest
per-step turn `|Δ arg F²|/π`, and a label is flagged ambiguous in `status`
when that turn is not below `1/2`.
Pick a labeled root with `ColdPlasmaWaveResult.select(PlasmaWaveMode.RIGHT, …)`.
Evanescent roots (`Re n² < 0`), resonances (`n² = ∞`), coincident roots, and
undefined polarizations are reported as `ColdPlasmaWaveStatus` bits, never
hidden behind NaN. `quasi_longitudinal_term` and `quasi_transverse_term` are
the two parts of the discriminant that decide the QL/QT regime.

`characteristic_frequencies` returns every zero of `R`, `L`, `P` (cutoffs) and
`S` (hybrid resonances) of the rational Stix parameters, with the residual of
the parameter at each zero; `resonance_cone` returns `tan²θ_res = −P/S`.
`faraday_coefficients` returns the Faraday rotation `ρ_V = (ω/2c)(n_L − n_R)`
along `B₀` and the conversion `ρ_Q = (ω/2c)(n_X − n_O)` across it, computed
for any angle from the two mode indices and their transverse polarizations.

See [Plasma waves and emission](../guides_plasma_waves_and_emission.md) for
worked examples.

::: phydrax.electromagnetics.ColdPlasmaDielectric

---

::: phydrax.electromagnetics.StixParameters

---

::: phydrax.electromagnetics.ColdPlasmaWaveResult

---

::: phydrax.electromagnetics.PlasmaWaveMode

---

::: phydrax.electromagnetics.ColdPlasmaWaveStatus

---

::: phydrax.electromagnetics.ColdPlasmaResonanceCone

---

::: phydrax.electromagnetics.ColdPlasmaCharacteristicFrequencies

---

::: phydrax.electromagnetics.FaradayCoefficients

## Magnetobremsstrahlung

`MagnetobremsstrahlungPlan(plasma, distribution, emitter_density=N, ...)`
computes the spontaneous emission and absorption of a gyrotropic population
(charge `emitter_charge_number · e`, mass `emitter_mass_ratio · m_e`) into both
modes of a `ColdPlasmaDielectric`, using the mode index `n_σ(ω, θ)` and
polarization `e_σ` of that dielectric. `evaluate(ω, θ)` returns, per root,
`emission` `j_σ` (power per volume, unit angular frequency and wave-normal
steradian) and `absorption` `α_σ` (per length along the wave normal), with the
mode-energy normalization `n_σ/|e_T|²`, so a thermal population obeys
`j_σ = n_σ² (k T ω²/(8π³c²)) α_σ` exactly. The two roots combine into a Stokes
emission vector and a propagation matrix in the `FaradayCoefficients` basis
whose rotation part is the cold-plasma Faraday rotation and conversion.

Routes (`MagnetobremsstrahlungRoute`):

- `"harmonic-sum"`: the exact integer harmonic sum on each resonance curve
  `γ = sY + N∥u∥`. The harmonic set is every `s` whose resonance ellipse
  (`N∥² < 1`) or hyperbola (`N∥² ≥ 1`, including anomalous-Doppler `s ≤ 0`)
  meets the distribution's momentum support; sets larger than
  `maximum_harmonics` are refused per root.
- `"continuous-harmonic"`: the continuous-harmonic limit `Σ_s → ∫ ds` with
  exact real-order Bessel functions over the whole momentum plane, clustered on
  the Razin-widened emission cone. Superluminal roots (`|N∥| ≥ 1`) are refused
  and results whose emission-weighted harmonic number is below
  `minimum_continuous_harmonic` are flagged.

Distributions (`AbstractGyrotropicDistribution`, normalized per `d³u`,
`u = p/(m c)`): `ThermalJuttnerDistribution(θ)` with a rigorous tail bound,
`PowerLawDistribution(index, u_min, u_max)`, `KappaDistribution(θ, κ, u_max)`,
and `TabulatedGyrotropicDistribution(u, cos α, log f)`. Absorption uses exact
JVPs of `log f`.

`MagnetobremsstrahlungStatus` bits `EVANESCENT`, `RESONANCE_CONE`,
`POLARIZATION_UNDEFINED`, `HARMONIC_CAPACITY_EXCEEDED`, `ANOMALOUS_DOPPLER` and
`NONFINITE` make a root unsupported (NaN, never zero); `QUADRATURE_UNRESOLVED`,
`LOW_HARMONIC` and `COLLISIONAL_MEDIUM` qualify supported values. The result
also reports the resonant harmonic range and count, the emission-weighted
harmonic number, the Kronrod–Gauss quadrature error and the distribution's
omitted tail mass.

::: phydrax.electromagnetics.MagnetobremsstrahlungPlan

---

::: phydrax.electromagnetics.MagnetobremsstrahlungResult

---

::: phydrax.electromagnetics.MagnetobremsstrahlungStatus

---

::: phydrax.electromagnetics.AbstractGyrotropicDistribution

---

::: phydrax.electromagnetics.ThermalJuttnerDistribution

---

::: phydrax.electromagnetics.PowerLawDistribution

---

::: phydrax.electromagnetics.KappaDistribution

---

::: phydrax.electromagnetics.TabulatedGyrotropicDistribution

### Gyrosynchrotron oracles (UFGC, Symphony)

Pinned external oracles for `MagnetobremsstrahlungPlan` (GPL-3.0 providers,
run only as caller-pinned libraries through a pinned interpreter).
`run_ufgc` returns UFGC's ordinary/extraordinary `j_σ`, `κ_σ` from its exact
or continuous code; `run_symphony` returns Symphony's vacuum Stokes `(j, α)`.
Both results carry the provider version, library digest, license, output
digest and an `AdapterReport` of declared losses; unsupported plans are
refused before running.

::: phydrax.electromagnetics.UFGCProvider

---

::: phydrax.electromagnetics.run_ufgc

---

::: phydrax.electromagnetics.ufgc_input

---

::: phydrax.electromagnetics.read_ufgc_output

---

::: phydrax.electromagnetics.UFGCCoefficients

---

::: phydrax.electromagnetics.UFGCResult

---

::: phydrax.electromagnetics.SymphonyProvider

---

::: phydrax.electromagnetics.run_symphony

---

::: phydrax.electromagnetics.symphony_input

---

::: phydrax.electromagnetics.read_symphony_output

---

::: phydrax.electromagnetics.SymphonyCoefficients

---

::: phydrax.electromagnetics.SymphonyResult

## Thermal synchrotron (MNY96) and free–free

`ThermalSynchrotronModel` is the fast angle-averaged ultra-relativistic thermal
route of Mahadevan, Narayan & Yi (1996), eq. 31, bound to the CODATA 2022 SI
`ElectromagneticScaleContract`; its Stokes-I support, validated `K₂`
approximation and unqualified polarization/Faraday approximations are reported
as evidence. `ThermalFreeFreeModel` gives thermal electron–ion bremsstrahlung
`j_ω` and Kirchhoff `α` with the Born thermal Gaunt factor
`born_thermal_gaunt(u) = (√3/π) e^{u/2} K₀(u/2)`, flagging the Born
(`Z² Ry ≪ kT`), nonrelativistic and `ω > ω_p` supports. Both provide
`gray_means(...)`, the Planck (at the matter and at the radiation temperature)
and Rosseland means as `GrayMeanOpacities` with per-mean edge-decay and
quadrature evidence; compact-object gray closures are computed from them.
`phydrax.applications.astrophysics.invariant_emission_coefficients` converts
either coefficient set to the GR-invariant transfer convention.

::: phydrax.electromagnetics.ThermalSynchrotronModel

---

::: phydrax.electromagnetics.ThermalSynchrotronCoefficients

---

::: phydrax.electromagnetics.ThermalFreeFreeModel

---

::: phydrax.electromagnetics.ThermalFreeFreeCoefficients

---

::: phydrax.electromagnetics.GrayMeanOpacities

## Kinetic plasma dielectric

`KineticPlasmaDielectric` is the hot counterpart of `ColdPlasmaDielectric`:
drifting bi-Maxwellian species (densities, charge numbers, electron-mass
ratios, `k_B T∥`, `k_B T⊥` in the scale's energy unit, field-aligned drifts)
in the same Stix frame and `exp(−iωt)` convention. `susceptibility(ω, k∥, k⊥)`
accepts complex `ω` and signed `k∥`, and returns per-species `χ_s`, the
dielectric tensor and evidence (`KineticSusceptibilityResult`).

- `model="nonrelativistic"` sums cyclotron harmonics `n = −N … N` with
  `Z(ζ) = i√π w(ζ)`; `w` is entire, so every `Im ω` is on Landau's contour,
  and `k∥ < 0` uses the causal `sgn(k∥) Z(sgn(k∥) ζ)`. `k∥ = 0` (Bernstein
  waves) and `k⊥ = 0` are exact limits.
- `model="weakly-relativistic"` (isotropic, drift-free species) keeps
  `γ ≈ 1 + u²/2` in the resonance and the lowest Larmor-radius order of each
  harmonic, through Shkarofsky functions `F_q(z, a)` evaluated from closed forms
  in `Z` on the sheet continued from `Im ω > 0`.
- `truncation_ratio` evaluates the first omitted harmonic pair `±(N + 1)`
  relative to the retained susceptibility and sets `HARMONIC_TRUNCATION` above
  `truncation_tolerance`; `larmor_parameter` `λ = k⊥²T⊥/(mΩ²)` sets
  `LARMOR_RADIUS_LIMIT` for the weakly relativistic model above
  `larmor_tolerance`.

`KineticDispersionProblem` solves `det[n²(κ̂κ̂ − I) + ε]/(1 + n²)² = 0`
(`"electromagnetic"`) or `κ̂·ε·κ̂ = 0` (`"electrostatic"`) for complex `ω`
along a wavenumber path with `phydrax.nonlinear.VectorLocalRootPlan`, starting
each point from the secant prediction of the last two accepted roots.
`KineticDispersionResult` keeps every iterate and reports `converged`,
residual norm, Jacobian condition estimate, predicted frequencies and
`KineticDispersionStatus` bits (`NOT_CONVERGED`, `ILL_CONDITIONED`,
`BRANCH_JUMP`, truncation, Larmor, nonfinite); failed points are never
zero-filled and never used to continue the branch.

`RelativisticWeakGrowthPlan` gives the weak growth rate
`γ = −ω² e*·χᴬ·e / e*·∂_ω(ω²Kᴴ)·e` of a `ColdPlasmaDielectric` mode driven by a
dilute energetic species with any `AbstractGyrotropicDistribution`. The
anti-Hermitian relativistic susceptibility is integrated over the resonance
ellipse `γ − N∥u∥ − sΩ/ω = 0` (`N∥² < 1`) with a Gauss–Legendre rule in the
eccentric anomaly; the half-order rule gives `quadrature_error`.
`LossConeDistribution` (Dory–Guest–Harris), `RingDistribution` and
`HorseshoeDistribution` (shell with smooth loss cone) are maser drivers.
`WeakGrowthStatus` marks non-propagating modes, non-elliptic resonances
(NaN rate), harmonics without resonance, unresolved quadrature, violated weak
growth and ambiguous mode labels.

::: phydrax.electromagnetics.KineticPlasmaDielectric

---

::: phydrax.electromagnetics.KineticSusceptibilityResult

---

::: phydrax.electromagnetics.KineticSusceptibilityModel

---

::: phydrax.electromagnetics.KineticSusceptibilityStatus

---

::: phydrax.electromagnetics.KineticDispersionProblem

---

::: phydrax.electromagnetics.KineticDispersionResult

---

::: phydrax.electromagnetics.KineticDispersionModel

---

::: phydrax.electromagnetics.KineticDispersionStatus

---

::: phydrax.electromagnetics.RelativisticWeakGrowthPlan

---

::: phydrax.electromagnetics.RelativisticWeakGrowthResult

---

::: phydrax.electromagnetics.WeakGrowthStatus

---

::: phydrax.electromagnetics.LossConeDistribution

---

::: phydrax.electromagnetics.RingDistribution

---

::: phydrax.electromagnetics.HorseshoeDistribution
