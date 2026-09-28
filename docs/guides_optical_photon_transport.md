# Optical photon transport

`phydrax.optics.transport` owns incoherent photon-packet Monte Carlo through
piecewise-homogeneous media bounded by oriented triangles. Use it after
coherence and diffraction have intentionally been discarded; rigorous
electromagnetics stays under `phydrax.solver.maxwell`.

## Photon state

`OpticalPhotonState` carries, per lane, the position, unit direction, a
transverse axis `e1`, the complex Jones vector `J = J1 e1 + J2 e2` with
`e2 = direction × e1`, the vacuum wavelength, the time, the statistical weight,
and the persistent 64-bit identity `(id_hi, id_lo)`. The frame `(e1, e2,
direction)` is right-handed and the Jones vector is unit-norm: weight alone
carries power and every frame change is a real rotation.

`launch_optical_photons` prepares a launch batch on the host. Identities are
allocated consecutively from `first_identity` in launch order with explicit
carry between words, exactly like particle populations, and the reserved
all-ones identity is refused. Launch two disjoint batches with disjoint
identity ranges and they reproduce the corresponding photons of one larger
batch.

The tissue configuration below reflects a normally incident beam from a
semi-infinite, index-mismatched scattering half-space; it is
`examples/optical_tissue_reflectance.py`.

```python
import jax.random as jr
import numpy as np
import phydrax as phx
from phydrax.optics.transport import (
    ExplicitPhotonSource,
    OpticalMonteCarloPlan,
    OpticalVarianceReduction,
    TissueOpticalMedium,
    launch_optical_photons,
    prepare_optical_monte_carlo,
    simulate_optical_photons,
)

# Tissue (medium 0, n = 1.5) fills z > 0 under an ambient half-space (medium 1).
extent = 1.0e3
surfaces = phx.optics.geometric.NonSequentialSurfaceTable(
    np.asarray(
        (
            (-extent, -extent, 0.0),
            (extent, -extent, 0.0),
            (extent, extent, 0.0),
            (-extent, extent, 0.0),
        )
    ),
    np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
    np.asarray((1, 1)),  # medium on the negative side of each triangle
    np.asarray((0, 0)),  # medium on the positive side
    np.asarray((1.5, 1.0)),
    surface_ids=np.asarray((0, 0)),
)
# mu_a, mu_s [1/cm], Henyey-Greenstein g, refractive index per medium.
medium = TissueOpticalMedium(
    np.asarray((10.0, 0.0)),
    np.asarray((90.0, 0.0)),
    np.asarray((0.0, 0.0)),
    np.asarray((1.5, 1.0)),
)
relativity = phx.RelativityScaleContract.from_si(
    phx.DimensionalScaleContract(
        phx.units.CENTIMETER,
        phx.units.KILOGRAM,
        phx.units.SECOND,
        length_coordinate_kind="physical",
    )
)
prepared = prepare_optical_monte_carlo(
    OpticalMonteCarloPlan(
        surfaces,
        medium,
        relativity=relativity,
        maximum_interactions=400,
        photon_batch_size=2048,
        variance_reduction=OpticalVarianceReduction(
            roulette_threshold=1.0e-4, roulette_survival_probability=0.1
        ),
    )
)
photons = launch_optical_photons(
    np.tile((0.0, 0.0, -1.0e-3), (4096, 1)),
    np.tile((0.0, 0.0, 1.0), (4096, 1)),
    1,
    wavelengths=633.0e-9,
)
result = simulate_optical_photons(prepared, ExplicitPhotonSource(photons), jr.key(0))
total_reflectance = result.tallies.escape  # Giovanelli (1955): 0.2600
```

Lengths are in the unit of the medium's inverse coefficients; `relativity`
supplies the exact speed of light in that length unit per time unit, so lane
times advance by `n · distance / c` through each medium.

## Randomness and invariance

Every random draw is derived with `derive_key` from the root key, a static
`SampleAddress` naming the draw purpose (free path, scattering, interface,
roulette), the interaction index, the photon identity words, and the lane's
branch. Keys therefore follow the photon rather than its storage slot: each
photon replays the same history regardless of `photon_batch_size`, of the
order of photons in a batch, and of which other photons are launched
alongside. Integer outcomes (status, interaction counts, media) are identical;
floating-point fields agree up to the rounding of differently vectorized
arithmetic.

## Media, surfaces, sources, detectors

The transport kernel is closed; physics enters through four protocols.

- `OpticalMedium` supplies `refractive_index`, `extinction`, `albedo`,
  `spectral_support`, and `scatter` per lane as functions of medium index and
  wavelength. `scatter` receives the Jones vector in the carried frame and
  per-lane keys and returns an `OpticalScatteringSample` (cosine, azimuth, the
  Jones vector on the scattering-plane frame, the post-event vacuum
  wavelength, and a re-emission delay added to the lane time). Lanes whose
  wavelength lies outside `spectral_support` of their medium, or of the
  medium behind a dielectric they reach, are refused before any coefficient
  is used: their weight is reported as truncated with a
  `SPECTRAL_SUPPORT_EXCEEDED` status. `TissueOpticalMedium` is the
  wavelength-independent scalar Henyey-Greenstein medium: it rotates the Jones
  vector into the scattering frame with `rotate_jones_to_scattering_frame`
  and leaves its components unchanged. `SpectralOpticalMedium` is the
  tabulated spectral medium described below.
- `OpticalSurfaceModel` turns an `OpticalSurfaceHit` (direction, normal from
  incident to transmitted medium, indices, surface kind, wavelength, Jones
  vector on `(s, p)`, the unit `s` axis, and the surface identity) and
  per-lane identity-addressed keys into an `OpticalSurfaceInteraction` with
  reflected and transmitted directions, power fractions, Jones vectors, and
  the transverse axis each candidate carries. Power the surface neither
  reflects nor transmits (`reflectance + transmittance < 1`) is absorbed at
  the surface in the incident medium; stochastic branching absorbs the lane
  with that probability and expected-split tallies that fraction.
  `validate_surfaces` refuses surface tables the model cannot describe when
  the plan is built. `ScalarFresnelSurfaceModel` uses the unpolarized mean of
  the `s` and `p` Fresnel power coefficients (the MCML convention) and
  transports the Jones components unchanged; `UnifiedSurfaceModel` is the
  polarized model described below. Absorber and detector surface kinds are
  owner semantics: their weight is tallied directly.
- `OpticalDetectorResponse` returns the fraction of accepted weight a detector
  records; the remainder is absorbed at the detector surface so the ledger
  closes. `UnitDetectorResponse` records everything the acceptance cone passes.
  With `detector_arrival_capacity > 0` the plan also retains each photon's
  accepted detector arrivals (detector, position, time, wavelength, recorded
  weight, incidence cosine) in history order as `OpticalDetectorArrivals`;
  arrivals beyond the capacity are counted and reported with a
  `DETECTOR_ARRIVAL_CAPACITY_EXHAUSTED` status.
- `OpticalPhotonSource` launches an `OpticalPhotonState` from a key;
  `ExplicitPhotonSource` launches one prepared batch exactly.

## Spectral media

`SpectralOpticalMedium(wavelengths, refractive_indices, absorption_lengths, *,
rayleigh, henyey_greenstein, mie, wavelength_shifter)` tabulates every medium on
one strictly increasing vacuum-wavelength grid. Refractive indices are real;
lengths are positive, with `+inf` for an absent process. Inverse lengths and the
Henyey-Greenstein `g` are interpolated linearly in wavelength by the native
piecewise-linear substrate, and wavelengths outside the grid are unsupported,
never extrapolated. The extinction sums bulk absorption, Rayleigh,
Henyey-Greenstein, Mie extinction, and wavelength-shifter absorption; the
albedo counts elastic scattering plus quantum-yield-weighted re-emission, so
implicit capture tallies absorbed quanta per medium. Volume events pick one
process in proportion to its rate.

Polarized scattering uses the Bohren-Huffman amplitude functions `S1`
(perpendicular) and `S2` (parallel), with absorbing indices `m = n + i kappa`.
For a unit Jones vector with carried-frame Stokes `Q`, `U` the joint density of
the scattering angle and azimuth factors exactly into the unpolarized phase
function `(|S1|^2 + |S2|^2) / 2` and the azimuth law
`(1 - a cos 2(phi - psi)) / (2 pi)` with `a = L (|S1|^2 - |S2|^2) /
(|S1|^2 + |S2|^2)`, `L = sqrt(Q^2 + U^2)`, `psi = atan2(U, Q) / 2`. Both are
sampled exactly, and the scattered Jones vector is `(S1 J_s, S2 J_p)`
normalized: the weight carries power and the Jones vector the polarization.

- `RayleighScattering(scattering_lengths)` is the point dipole (`S1 = 1`,
  `S2 = cos theta`). The polar angle is the exact Cardano inverse of
  `1 + cos^2 theta`; a linearly polarized photon radiates as `sin^2` of the
  angle to its field and never along it.
- `HenyeyGreensteinScattering(scattering_lengths, anisotropy)` is the scalar
  specialization (`S1 = S2`) shared with `TissueOpticalMedium`.
- `MieParticles(radii, refractive_indices, number_densities, *,
  length_per_wavelength_unit, angle_count, table_tolerance)` suspends
  monodisperse spheres. `length_per_wavelength_unit` links the transport length
  unit to the wavelength unit (100 for centimeters and metres). At every grid
  node the medium prepares the Lorenz-Mie series with Wiscombe's truncation
  `N_stop`, `D_n(m x)` recurred downward from a Lentz continued-fraction start,
  and upward Riccati-Bessel recurrences; the coefficients give the scattering
  and absorption coefficients and, at run time, `S1`/`S2` at the sampled angle.
  The polar angle is drawn by the exact inverse CDF of a phase density
  piecewise linear in the cosine on `angle_count` angles uniform in `theta`;
  between nodes the angular law is the interpolation-weighted mixture of the
  two node laws. `mie_evidence` (`MieTableEvidence`) reports size parameters,
  relative indices, term counts, Lentz lengths, the scattering share of the
  last retained order, efficiencies, asymmetry, and the normalization and
  asymmetry residuals of the tables; tables missing the series by more than
  `table_tolerance` are refused.
- `WavelengthShifter(absorption_lengths, emission_wavelengths,
  emission_spectra, quantum_yields, delay_times, *, delay)` absorbs on the
  medium grid and re-emits isotropically and unpolarized (a Jones vector
  uniform on the Poincaré sphere) with probability `quantum_yields`, at a
  wavelength drawn exactly from the piecewise-linear emission density and
  after a delay that is exponential with mean `delay_times` or exactly
  `delay_times` (`delay="delta"`). The emission grid must lie inside the
  medium grid.

`lorenz_mie(size_parameter, relative_index, cosines)` exposes the series for
one sphere as a `LorenzMieResult`: extinction, scattering, absorption, and
backscattering efficiencies, asymmetry, `S1`/`S2` at the requested cosines,
the coefficients `a_n`, `b_n`, and the truncation evidence.
`examples/optical_spectral_media.py` shifts ultraviolet light in a
scattering, dye-doped water slab.

## Polarized surfaces and UNIFIED finishes

`UnifiedSurfaceModel(finishes, *, sigma_alpha, specular_spike, specular_lobe,
backscatter, reflectivity, facet_attempts)` declares one finish per surface id
of the table: `"polished"`, `"ground"`, `"polished-front-painted"`, or
`"ground-front-painted"` (the `OpticalSurfaceFinish` selector). Every
interaction happens on a facet with normal `m`: the Jones vector is
re-expressed on the facet frame `(s', d × s')`, `s' = d × m`, multiplied by
the complex `(r_s, r_p)` or `(t_s, t_p)` amplitudes of the refractive-interface
owner, and renormalized, so total internal reflection keeps its phase and the
reflected light becomes elliptical. The reflected power is
`|r_s J_s|^2 + |r_p J_p|^2` and the transmitted power carries the flux factor
`n_2 cos theta_t / (n_1 cos theta_i)`. Mirror surfaces and paints use the
perfect-conductor amplitudes `(-1, 1)`.

- `"polished"` is the exact polarized Fresnel boundary (or a specular mirror).
- `"ground"` follows the Geant4 UNIFIED model: a facet tilt with density
  proportional to `exp(-alpha^2 / 2 sigma_alpha^2) sin alpha` on `[0, pi/2)`
  (sampled exactly from a truncated-Rayleigh proposal) and a uniform azimuth;
  the facet powers decide reflection or refraction, and reflections are
  redirected as a specular spike about the mean normal, a specular lobe about
  the facet, a backscatter, or a Lambertian reflection with probabilities
  `specular_spike`, `specular_lobe`, `backscatter`, and the remainder. Spike,
  backscatter, and Lambertian redirections apply the mirror polarization
  transform about the normal that specularly produces the outgoing direction.
  Facets facing away from the photon and outcomes leaving through the wrong
  side of the mean surface are resampled within `facet_attempts`; a lane with
  no accepted attempt fails with `INTERFACE_FAILURE`.
- `"polished-front-painted"` and `"ground-front-painted"` are opaque specular
  and Lambertian paints.

`reflectivity` is the probability that the surface does not absorb the photon
before the interaction. `sigma_alpha` and the branch probabilities apply only
to `"ground"`; other finishes, negative or non-finite values, probabilities
outside `[0, 1]` or summing above one, and non-default declarations on
absorber or detector surfaces are refused.

## Photodetection

`OpticalPhotodetector(wavelength_nodes, quantum_efficiency,
reference_positions, *, collection_efficiency, transit_times,
transit_time_spreads, single_photoelectron_charges,
single_photoelectron_spreads, dark_count_rates)` describes each detector of the
table. `PhotodetectionPlan(photodetector, hit_plan, *, gate, hit_capacity,
dark_count_capacity)` binds it to a detector `SensitiveHitPlan` (element-to-
channel map and conditions) and a readout gate. `detect_optical_arrivals(plan,
arrivals, key, *, event_ids, photon_events)` converts the transport's
`OpticalDetectorArrivals` into the canonical `(event, hit)`
`SensitiveHitBank`, which `digitize_sensitive_hits` digitizes directly.

An arrival with recorded weight `w` releases a photoelectron with probability
`w QE(lambda) CE`, quantum efficiency interpolated linearly on the nodes
(wavelengths off the nodes are refused with `UNSUPPORTED_WAVELENGTH`); unit
weights give binomial counts and implicit-capture weights keep the expected
count exact, and probabilities above one are reported. Photoelectrons emitted
inside the gate reach the anode after the mean transit time plus a Gaussian
transit-time spread, with a gamma (Polya) single-photoelectron charge of the
declared mean and standard deviation, stored as the hit energy. Dark counts are
Poisson with mean `rate (end - start)` per event and detector, uniform in the
gate, at the detector reference position, and pass through the same transit
and gain. Every photon draw is keyed by photon identity and arrival slot and
every dark-count draw by event identity, detector, and ordinal; hits are
ordered per event by anode time, then photon identity, so the bank does not
depend on photon order or batching. `PhotodetectionResult` carries the hits,
dark-count and photon-identity provenance per hit, expected and sampled
photoelectron counts, expected and sampled dark counts, dropped hits, and a
per-event `PhotodetectionStatus`. Pair it with `UnitDetectorResponse`, or a
response that models only optics before the photocathode, so quantum
efficiency is not applied twice. `examples/optical_light_guide_photodetection.py`
reads a ground-surface light guide out through a photomultiplier and the
detector digitization.

## Cherenkov and scintillation sources

Charged particles become optical sources through `ChargedOpticalSteps`:
straight steps with endpoint speeds `start_beta`, `end_beta` (units of `c`),
speed linear in path length along each step, step start times, the optical
medium of each step (`-1` for a non-optical material), deposited energy, charge
numbers, multiplicities, persistent parent identities, and a completeness flag.
`charged_steps_from_transport(result, material_media, relativity)` reads the
step bank of a `ChargedParticleTransportPlan` result (electromagnetic showers
expose one per generation): `material_media[m]` maps charged material `m` to an
optical medium, and step times accumulate the exact transit time
`L ln(b1 / b0) / (c (b1 - b0))` of the preceding steps from each history's
start. `charged_steps_from_trajectory(trajectory, scale, medium_indices, *,
deposited_energy)` joins consecutive active samples of a `ChargedTrajectory`,
for instance detector tracks from `applications.detector.charged_trajectory`,
with `beta = |u| / sqrt(c^2 + |u|^2)` and charge numbers `q / e`.

`CherenkovEmission(medium, wavelengths, *, radiating_media)` reads each
medium's refractive index once on the declared wavelength nodes.
`phydrax.equations.cherenkov_step_spectral_yield` is the single Frank-Tamm
owner: `d2N / (dx dlambda) = 2 pi alpha z^2 / lambda^2 (1 - 1 / (beta^2
n(lambda)^2))` above threshold, integrated exactly along a step whose speed is
linear in path. The nodal yields are represented piecewise linearly, so the
band integral (trapezoidal) and the sampled spectrum (exact inverse CDF) are
the same distribution and converge at second order in the node spacing. Given
the wavelength, the emission speed is drawn by the exact inverse CDF of
`1 - 1 / (n^2 beta^2)` on the above-threshold part of the step (a quadratic in
closed form); the photon leaves on the cone `cos theta = 1 / (beta n(lambda))`
at that local speed with a uniform azimuth, polarized along `p x (p x v)`:
transverse and radial in the plane of the track and the photon.

`ScintillationEmission(photon_yields, birks_constants, component_fractions,
rise_times, decay_times, emission_wavelengths, emission_spectra)` converts a
step's deposit into visible energy `E / (1 + kB E / L)` (Birks) and a mean
photon count `Y E_vis`. Each photon picks a time component, is emitted
uniformly along the step, isotropically, with a random linear polarization,
after the sum of exponential rise and decay delays (the bi-exponential law
`(exp(-t / tau_d) - exp(-t / tau_r)) / (tau_d - tau_r)`), at a wavelength drawn
exactly from the component's piecewise-linear emission spectrum.

`OpticalPhotonSourcePlan(*, relativity, photon_capacity, cherenkov,
scintillation, length_per_wavelength_unit, batch_size)` combines the processes;
`emit_optical_photons(plan, steps, key, *, first_identity)` samples Poisson
counts per step and fills one fixed-capacity `OpticalPhotonState` in canonical
order (parent identity, step, process, ordinal). Every draw is keyed by the
parent identity, the step index, and the photon ordinal, so the bank does not
depend on parent order or batching, and splitting a step at its interpolated
speed leaves the expected yields unchanged. Accepted photons take consecutive
identities from `first_identity`; `(next_id_hi, next_id_lo)` chains the next
emission. `OpticalPhotonEmission` keeps the lineage (process, parent identity,
parent step), per-parent expected and sampled counts, deposited and visible
energy, emitted photon energy `2 pi hbar c / lambda` in the relativity scale's
energy unit, and non-optical step counts. Capacity overflow, incomplete step
banks, invalid steps, identity exhaustion, or non-finite photons set an
`OpticalEmissionStatus` flag and refuse the whole bank: no photon is allocated.
Launch the bank with `ExplicitPhotonSource(emission.state)`; empty slots carry
zero weight. `examples/cherenkov_water_detector.py` follows a 3 MeV electron
through water, transports its Cherenkov light to the walls, and reads it out
through photodetection.

## Variance reduction

`OpticalVarianceReduction` is explicit. `"stochastic"` interface branching keeps
one lane per interface event with the reflectance as the reflection
probability; `"expected-split"` launches both lanes weighted by reflectance and
transmittance within `branch_capacity`, and the weight of candidates beyond the
capacity is reported as truncated with a `BRANCH_CAPACITY_EXHAUSTED` status.
Roulette below `roulette_threshold` kills lanes with probability
`1 - roulette_survival_probability` and boosts survivors, keeping the expected
weight unchanged; the signed roulette tally has zero mean.

## Result and evidence

`OpticalTransportResult` returns per-photon tallies, their means, and standard
errors (absorption per medium, signed surface flux per surface, detector
weight, escape, roulette, live, truncated, launched, ledger residual), the
terminal `OpticalPhotonState` per photon and branch lane, remaining optical
depths, live masks, interaction counts, per-photon `OpticalTransportStatus`,
the maximum absolute ledger residual, and `maximum_polarization_defect`, the
largest deviation of a live lane from a unit-norm, unit-axis, transverse
polarization frame, and the `detector_arrivals`. The prepared plan reports
maximum triangle tests, random draws, bytes per photon, and the working-set
bytes of one photon batch, and states that pathwise differentiability is not
claimed.

## References

- Wang, Jacques, Zheng, "MCML — Monte Carlo modeling of light transport in
  multi-layered tissues", Comput. Methods Programs Biomed. 47 (1995) 131–146,
  Tables 1 and 2 (`mu_a = 10 /cm`, `mu_s = 90 /cm`), reproduced by the tissue
  tests within four standard errors: van de Hulst (1980) diffuse reflectance
  0.09739 and total transmittance 0.66096 for an index-matched `g = 0.75` slab
  of thickness 0.02 cm, and Giovanelli (1955) total reflectance 0.2600 of a
  normally incident beam (entrance Fresnel reflection included) on a
  semi-infinite isotropic medium with relative index 1.5.
- W. J. Wiscombe, "Improved Mie scattering algorithms", Appl. Opt. 19 (1980)
  1505–1509, and "Mie scattering calculations: advances in technique and fast,
  vector-speed computer codes", NCAR/TN-140+STR (1979, revised 1996): series
  truncation, downward `D_n` recurrence, and the MIEV0 test cases reproduced
  by the tests (non-absorbing, weakly and strongly absorbing spheres, and the
  angular `S1`/`S2` case; MIEV0 amplitudes are the complex conjugates of the
  Bohren-Huffman ones used here).
- W. J. Lentz, "Generating Bessel functions in Mie scattering calculations
  using continued fractions", Appl. Opt. 15 (1976) 668–671.
- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small
  Particles* (Wiley, 1983), Ch. 4–5: amplitude functions, efficiencies,
  optical theorem, and the small-sphere (Rayleigh) limit.
- L. G. Henyey and J. L. Greenstein, "Diffuse radiation in the Galaxy",
  Astrophys. J. 93 (1941) 70–83.
- Geant4 Collaboration, *Book For Application Developers*, "Optical photon
  processes" (bulk absorption, Rayleigh, Mie, and wavelength-shifting
  processes with tabulated material properties).
- M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge, 1999),
  §1.5: Fresnel amplitudes and the total-internal-reflection phase
  `tan(delta / 2) = cos theta sqrt(sin^2 theta - n^2) / sin^2 theta` (Fresnel
  rhomb), reproduced by the tests.
- A. Levin and C. Moisan, "A more physical approach to model the surface
  treatment of scintillation counters and its implementation into DETECT",
  IEEE Nuclear Science Symposium Conference Record (1996) 702–706; S. K.
  Nayar, K. Ikeuchi, and T. Kanade, "Surface reflection: physical and
  geometrical perspectives", IEEE Trans. Pattern Anal. Mach. Intell. 13 (1991)
  611–634; Geant4 *Book For Application Developers*, "Boundary process"
  (UNIFIED model, finishes, micro-facet sampling).
- Hamamatsu Photonics, *Photomultiplier Tubes: Basics and Applications*, 4th
  ed. (2017), ch. 4 (quantum and collection efficiency, transit-time spread,
  single-photoelectron response, dark counts); G. F. Knoll, *Radiation
  Detection and Measurement*, 4th ed. (Wiley, 2010), ch. 9; J. R. Prescott,
  "A statistical model for photomultiplier single-electron statistics", Nucl.
  Instrum. Methods 39 (1966) 173–179.
- I. M. Frank and I. E. Tamm, "Coherent visible radiation of fast electrons
  passing through matter", C. R. Acad. Sci. URSS 14 (1937) 109–114; J. D.
  Jackson, *Classical Electrodynamics*, 3rd ed. (Wiley, 1999), §13.5:
  photon yield per unit path and wavelength, reproduced for water by the
  tests.
- J. B. Birks, "Scintillations from organic crystals: specific fluorescence
  and relative response to different radiations", Proc. Phys. Soc. A 64
  (1951) 874–877: `dL/dx = S (dE/dx) / (1 + kB dE/dx)`.
- Geant4 Collaboration, *Physics Reference Manual*, "Cerenkov effect" and
  "Scintillation" (photon counts, cone kinematics, polarization, yield,
  Birks quenching, and rise/decay time components).
