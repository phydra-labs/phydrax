# Radiation transport and material coupling

Phydrax separates radiation transport, spectral material coefficients, closure assumptions, and matter exchange.

`RayTransferPlan` performs absorption-emission transfer along prescribed independent rays. It does not implement isotropic scattering; a physical scattering source requires angular coupling. `PolarizedRadiativeTransferPlan` uses an augmented matrix exponential and remains valid for singular or zero propagation matrices.

`MultigroupM1RadiationSystem` provides hyperbolic moment transport and checks realizability. Closure clipping is a numerical guard, not proof that an arbitrary discretization preserves the realizable cone.

`GRGrayM1RadiationSystem` is the separate 3+1 gray moment transport and closure
system. `GRGrayRadiationInteractionPlan` consumes a distinct
`AbstractGRGrayOpacityPlan` to construct the fluid-frame four-force. This separation
allows one transport discretization to use constant, composite, bremsstrahlung,
synchrotron, Klein--Nishina, or caller-defined coefficients without changing the
hyperbolic system.

`FixedGridGRM1SSPRK3Plan` evolves densitized moments with metric-aware geometric
sources, explicit periodic or physical boundaries, PLM realizability limiting, and
asymptotic-preserving thick-limit dissipation. `GRMultigroupM1RadiationSystem` composes
bounded frequency groups. `GRNeutrinoM1System` composes species and groups and adds
exact opposite material energy, momentum, and lepton-number exchange.

`GrayLinearRadiationDiffusionPlan` is constant-coefficient linear diffusion. It distinguishes transport extinction from absorption and treats its supplied equilibrium radiation energy as frozen during a step.

## Angular closure and polarization

`VariableEddingtonTensorClosurePlan`, `DiscreteOrdinatesRadiationPlan`, and
`MonteCarloRadiationClosurePlan` expose the pressure tensor provenance instead of
presenting it as M1. They respectively validate an external positive trace-one tensor,
retain positive directional intensities, or derive moments with effective-packet and
sampling-error evidence.

`GRPolarizedRadiationFeedbackPlan` is the dynamical counterpart to observer-ray
polarized transfer. It advances local weighted Stokes beams through a canonical
absorption/dichroism/Faraday matrix exponential and applies the exact opposite
Stokes-$I$ energy-momentum change to material. It validates the Stokes cone and
propagation-matrix structure before atomic commit.

## Spectral coefficients

`SpectralFrequencyGrid` records physical frequencies and quadrature weights. `RadiationCoefficientTable` assigns every table one role: absorption, scattering, or transport. Table interpolation uses the native rectilinear gather substrate and returns explicit support.

`radiation_means` computes a Planck absorption mean and Rosseland transport mean. Supplying one undifferentiated opacity for both roles is not supported.


### Diagnostic-photon coefficients

`PhotonEnergyGrid` and `DiagnosticPhotonCoefficientTable` are separate from the
thermal frequency/temperature opacity path. They retain ordered material IDs,
mass-coefficient role, explicit area-per-mass units, evaluated-data provenance,
bounded linear or log-log interpolation, and optional exact provenance pinning.
There is no extrapolation or material-axis reordering. A table alone is not a
transport solver; `RadiationCrossSectionLibrary` prepares three such source-pinned
photoelectric, Compton, and Rayleigh tables with material mass density for native
photon histories.

## Conservative matter exchange

`RadiationMatterExchangePlan` couples radiation energy to the full homogeneous material internal energy. It solves the local backward exchange equation while holding species mass densities fixed and enforces exact combined radiation-material energy conservation. Its light-speed contract distinguishes physical and reduced light speed explicitly.

## General-relativistic ray transfer

Black-hole imaging uses invariant transfer in
`phydrax.applications.astrophysics`, not `RayTransferPlan`.
`InvariantScalarTransferPlan` advances `I_nu / nu**3` through a fixed active prefix of
piecewise-constant incident-to-observer segments with explicit path and intensity
units. `PolarizedRayPath` first binds one exact null `GRRayResult` and metric and
recomputes its transported-basis evidence. `PolarizedInvariantTransferPlan` advances
`(I,Q,U,V)` by matrix exponentials on that typed path. Support, propagation convention,
Stokes cone, basis/ray residual, qualification, and derivative validity remain in the
result.

`phydrax.electromagnetics.ThermalSynchrotronModel` qualifies only the declared MNY96 Stokes-$I$ support and
validated $K_2$ approximation. Its active linear-polarization and Faraday forms are
independent reference-unqualified approximations, so a composite active polarized
prediction is not reference-qualified. `FastLightSnapshot.prepare_path_sampling`
requires exact chart identity with `PolarizedRayPath` and binds the sampling stencil to
active/valid segment midpoints from that path. `MonotoneSlowLightWorldtube` supplies
explicit fixed-capacity spacetime sampling. These are
not selected automatically by the transfer plan. See
[General-relativistic rays, transfer, and imaging](guides_black_hole_imaging.md) and
[Relativistic matter, plasma, and radiation](guides_relativistic_matter.md).

## Separation from scattering and Hawking radiation

Material scattering in transfer, real-frequency black-hole potential scattering, and
semiclassical Hawking occupation are different problems.
`BlackHoleScatteringPlan` closes an independently solved scalar Killing-energy flux
ledger and reports signed graybody factors. `HawkingSpectrumPlan` consumes independently
qualified graybody data, corotation slopes, and tail bounds; it does not solve
radiative transfer or radial scattering. See the
[perturbation and Hawking guide](guides_black_hole_perturbations.md).

## Governed ionizing interaction data

`RadiationCrossSectionLibrary` is the prepared macroscopic runtime view of three
required canonical `DiagnosticPhotonCoefficientTable` inputs and an optional,
inseparable nuclear/electron-field pair-production pair. It requires one exact
material basis and energy grid, checks each source manifest's requested rights,
preserves each table's interpolation policy, converts area-per-mass
coefficients and kg/m³ density to inverse-meter rates, retains every
table/provenance identity, and never extrapolates.

`ChargedRadiationMaterialLibrary` separately owns stopping power, scattering
power, and bremsstrahlung rates. `SeltzerBergerBremsstrahlungTable` owns a
rights-checked differential spectrum and normalized, monotone row CDFs.
`tools/import_nist_seltzer_berger.py` only verifies and canonicalizes a
caller-supplied, SHA-256-pinned NIST CSV export; it neither downloads nor
reconstructs values. When no authoritative table has been admitted,
`bremsstrahlung_spectrum=\"screened-bethe-heitler\"` is the independent
complete-screening Tsai analytic route. Repository licenses do not grant
rights to external atomic or material data.

## Diagnostic photon Monte Carlo

`VoxelRadiationGeometryPlan` supplies material lookup and exact global/voxel boundary distances. `PhotonTransportPlan` addresses every random draw by the history's persistent `(id_hi, id_lo)` identity words through `derive_key`, uses Woodcock delta tracking, bounded Klein–Nishina and Rayleigh rejection sampling, local photoelectron/recoil KERMA, and optional governed Bethe–Heitler nuclear/electron-field pair production. A pair event records both charged daughters and their electron/positron `particle_kind`; the photon rest-mass transfer remains in the local ledger. Without a secondary stack the plan transports photons only. It labels deposition KERMA and makes no absorbed-dose claim near interfaces or outside charged-particle equilibrium. `compton_kinematics` selects `"free-electron"` or `"impulse-approximation"` Doppler broadening, which requires the governed Compton profile.

`AliasSpectrumPlan`, `DiagnosticXRaySourcePlan`, `PlanarXRayDetectorPlan`, and `DiagnosticXRayExperimentPlan` compose spectrum, cone source, transport, and detector response. Native deposited events can be materialized at the explicit host boundary with `photon_result_to_interaction_ledger`; virtual collisions and aggregate-only values are not fabricated as biophysical records.

## Discrete ordinates

`CertifiedSlabAngularQuadrature` proves the zeroth, first, and second Gauss–Legendre moments. `MultigroupSlabTransportProblem` binds one-dimensional cells, total and group-transfer scattering cross sections, sources, group sets, and vacuum/incident/reflecting boundaries. `DiscreteOrdinatesTransportPlan` performs directional sweeps, source iteration, optional diffusion synthetic acceleration, current/leakage calculation, response integration, and global balance evidence. The landed support is slab geometry with isotropic group transfer; arbitrary-mesh OpenSn-style sweeps are not claimed.

## Charged-particle condensed history

`ChargedParticleTransportPlan` advances electron or positron kinetic energy with boundary-limited continuous stopping, multiple-scattering deflection, cutoff deposition, escape, and bremsstrahlung. `bremsstrahlung_spectrum` selects the unchanged M1a `"bounded"` diagnostic route, source-faithful `"screened-bethe-heitler"`, or governed `"seltzer-berger"` inverse-CDF sampling. Optional `lpm_energy_ev` and `plasma_energy_ev` multiply the sampled event by the Migdal LPM and Ter–Mikaelian dielectric suppression factors; disabling either is the exact identity. Every history retains its kinetic-energy ledger and identity-addressed draws. `step_bank_capacity` records the M1a pre/post position, speed, deposit, and material contract.

`delta_ray_kinematics` supplies exact two-body lab kinematics for Møller
(electron) and Bhabha (positron) delta rays, and
`positron_annihilation_in_flight` boosts an exactly back-to-back two-photon COM
state to the target-electron rest frame. `atomic_relaxation` is the optional
single-vacancy fluorescence/Auger energy partition. These reference kernels
report support and four-momentum or energy residuals and are available to
material-process policies without silently inventing rate tables.

## Charged-step X-ray and optical emission

`FoilStackTransitionRadiationPlan` evaluates Garibian/Cherry coherent interface
amplitudes for regular or irregular foil positions, including formation-zone
phase and Beer–Lambert amplitude absorption. `ChargedStepRadiationPlan`
consumes a complete M1a `ChargedStepBank`, integrates that spectrum on declared
energy/angle grids for a traversed stack, and evaluates analytic Frank–Tamm
Cherenkov photon count and energy over a declared wavelength band on every
step. It emits expected-energy photon packets into a
`SecondaryParticleStack`; packet multiplicity remains explicit. Per-history
capacity refusal is atomic, and emitted, sub-threshold local, and refused
energy close its ledger.

Individual optical photons (dispersive Frank–Tamm cone photons and
scintillation with Birks quenching) are generated from the same step banks by
`phydrax.optics.transport.emit_optical_photons`; see
[Optical photon transport](guides_optical_photon_transport.md).

`examples/matter_radiation_processes.py` smoke-runs the no-table pair,
formation-length, and Frank–Tamm references. It does not represent a
material-qualified prediction.

## Secondary stacks and electromagnetic showers

`SecondaryStackSpec(capacity, minimum_energy=...)` attaches a fixed-capacity per-history `SecondaryParticleStack` to a transport or charged-step process. Each slot retains `particle_kind`, allowing Bethe–Heitler electron/positron daughters to enter the same charged shower batch. With an `electron_stack`, `PhotonTransportPlan` records Sauter photoelectrons, Klein–Nishina recoil electrons, and optional pairs. With a `photon_stack`, `ChargedParticleTransportPlan` records bremsstrahlung photons. Sub-threshold transfers are deposited locally; overflow is truncated and reported rather than partially committed.

The Sauter acceptance step accepts roughly one proposal in three between 10 keV and 1 MeV, so the default 16 `angular_sampling_attempts` refuse about one photoelectron in 500 with `ANGULAR_SAMPLING_EXHAUSTED`; showers should raise it (64 attempts bring the refusal probability below 1e-10). A history keeps its first failure status; event or step capacity is reported only for histories that were otherwise healthy.

`EMShowerPlan` couples photon and charged transport on one geometry for bounded generation rounds and fixed-capacity `ShowerParticleBatch` launches. Secondaries are compacted in identity order; pair daughters use distinct creation indices, so the stride is `max(2 * photon events, charged steps) + 1`. Every draw, launch order, and tally is independent of primary slot order. `EMShowerResult` closes `primary_energy = deposited + escaped + truncated + stack_remainder` (positron rest-energy photons remain a separately reported source), reports both cross-type transfers, per-generation counts and energies, every transport result, and atomic refusal evidence.

`examples/em_shower_slab.py` runs a two-generation shower of 200 keV photons in a synthetic slab and prints the closed ledger and secondary counts.

`phydrax.applications.detector.run_geant4_shower` is the pinned Geant4 oracle for these showers. `Geant4Provider(executable, data_directory)` pins a Python interpreter with `geant4_pybind` (the executable version is the `geant4_pybind` release, checked against the running interpreter) and the Geant4 dataset directory passed as `GEANT4_DATA_DIR`; set `PHYDRAX_GEANT4_PYTHON`, `PHYDRAX_GEANT4_PYTHON_VERSION`, and `PHYDRAX_GEANT4_DATA` for the live tests. `geant4_shower_input` translates a homogeneous-slab `EMShowerPlan` and its primary batches into a deck: one Geant4 event per active primary in identity order, the composition of an explicitly named Geant4 NIST material at the Phydrax mass density, one electromagnetic constructor (`"standard"`, `"standard-option4"`, `"livermore"`, `"penelope"`; no hadronic physics), and the photon-stack and charged-cutoff thresholds realized exactly as Geant4 range cuts (refused below Geant4's 990 eV table edge). `Geant4ShowerResult` returns per-event depth-binned deposits in eV, Geant4's radiation length, the realized thresholds and range cuts, the Geant4 release and datasets, and an `AdapterReport` declaring every loss (hadronic channels, interaction models, synthesized composition, threshold and tracking-cutoff transformations, unbounded generations, midpoint binning). `longitudinal_profile()` gives `dE/dt` per radiation length for comparison with `longo_shower_profile`. Heterogeneous geometries are refused.

## Hybrid IMC/DDMC

`HybridIMCDDMCPlan` stores fixed-capacity packet census and material energy on one-dimensional multigroup cells. Fleck effective absorption couples packets to material; optically thick cells use DDMC leakage. Packet, material, and escaped energy close one transactional ledger. The current profile does not emit new thermal packets or feed radiation momentum into hydrodynamics.

## Spectral and polarized experiments

`CorrelatedKDistributionPlan` owns positive normalized k quadrature per band. `ScalarRadiativeExperimentPlan` evaluates prescribed rays for every band/ordinate and applies one `RadiativeSensorPlan`. `PolarizedRadiativeExperimentPlan` composes existing Stokes matrix-exponential transfer over a prescribed frequency set and checks the Stokes cone. Scene construction, scattering path tracing, canopy geometry, and atmospheric databases remain caller-owned.

## Qualification and nonclaims

`radiation_transport_candidate_profiles()` and `radiation_transport_candidate_campaigns()` keep photon, shower matter-process, charged-step photon-production, S_n, charged-history, IMC/DDMC, and spectral/polarized claims separate. `tools/radiation_transport_qualification.py` establishes native analytic/numerical validity. The Longo–Sestili gamma profile and radiation-length scaling are native qualification references. Geant4 remains provider-only: comparison is admissible only through a separately pinned executable/artifact at the provider boundary, and no Geant4 executable is vendored. Authoritative cross sections, benchmark decks, experimental measurements, and independent locked comparisons require governed manifests before release. See [Radiation transport sources](radiation_transport_sources.md).
