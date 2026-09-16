# Astronomy closure systems

The closure layer extends the native astrodynamics, astrophysics, cosmology, and solver
contracts without adding a provider registry or a second simulation runtime.

## Time, frames, and products

`JulianDate`, `TimeInstant`, and `ReferenceEpoch` separate a physical instant from the
continuous relative-seconds coordinate used by solvers. `LeapSecondTable`,
`PreparedTimeRoute`, `EarthOrientationRecordSet`, and `PreparedEarthOrientation` make
UTC, TAI, GPS, TT, TCG, TDB, TCB, UT1, IERS Earth rotation, and terrestrial-frame data
explicit. `FrameTransformGraph.compile` resolves an immutable qualified path on the
host; only the fixed path enters transformed execution.

`AstrodynamicsDataStore` resolves explicitly configured, checksummed artifacts. It does
not scan the environment or fetch data automatically. Chebyshev ephemerides provide
analytic velocity and acceleration. CCSDS KVN and TLE products retain source text,
metadata, checksums, frame/time conventions, and provider errors.

## High-fidelity dynamics

Spherical-harmonic gravity, coefficient corrections, atmosphere/drag, eclipse-aware
radiation pressure, first post-Newtonian gravity, light time, analytical J2, DSST,
adaptive Gauss--Radau IAS15, multi-event schedules, manoeuvres, encounter evidence,
and hierarchical gravity are fixed-capacity plans. Adaptive schedules, hierarchy
refresh, collision topology, event ordering, and provider selection remain explicit
piecewise-differentiable boundaries.

## Vehicles and mission analysis

`CoupledVehiclePlan` evolves translation, quaternion attitude, tank masses, wheel
momentum, and time-varying mass/inertia through a block physical state. Effectors,
sensors, FSW commands, stations, tracking schedules, variational propagation, orbit
determination, access, targeting, and conjunction products compose existing Phydrax
linear algebra, nonlinear, control, filtering, and UQ owners.

## Observations and cosmology

Astronomical TAN/SIP WCS, calibrated detector formation, surveys,
absorption-emission ray transfer, polarized transfer, QNM/ringdown products,
detector networks, oblate occultation, and finite-source microlensing extend the
concrete observation operators.

Native early-universe products add relic backgrounds, fixed reaction BBN, recombination,
halo/nonlinear products, CMB lensing, light cones, lensing planes, and baryonic
feedback. `ScalarEinsteinBoltzmannPlan` now generates its own flat-FLRW synchronous
scalar evolution for CDM, baryons, photon temperature/polarization, massless relics,
and metric variables. Its prepared fixed scans emit cold+baryon/total transfer tables
and unlensed scalar TT/TE/EE products with constraint, tight-coupling overlap,
hierarchy-tail, line-of-sight, and schedule evidence. The supplied
`ScalarEvolutionOperatorTable` route remains an explicitly frozen external operator
product, not a second native equation solver. Compact-object EOS/TOV models live in
`phydrax.applications.compact_objects`, not inside cosmology.

## Black-hole closure ownership

The black-hole surface is layered rather than gathered under one relativity facade:

- `phydrax.metrix` owns exact charted Schwarzschild/Kerr metrics, explicit chart-domain
  evidence, Killing/tetrad geometry, sign/orientation conventions, and ADM exchange
  with static geometry lineage and exact dynamic snapshot token.
- `phydrax.applications.compact_objects` owns stationary Killing-horizon
  thermodynamics, perturbations/QNMs/scattering/Hawking products, accretion initial
  data, ingoing-Kerr GRRMHD torus/fast-light products, plasma microphysics, and EOS/TOV.
- `phydrax.equations` owns relativistic EOS, SRHD/Valencia GRHD, ideal GRMHD,
  resistive Ohm, force-free, and grey/multigroup/neutrino radiation closures.
- `phydrax.solver` owns bounded primitive recovery, boundary-aware finite-volume
  GRHD/GRMHD/CT, conservative GRRMHD IMEX, resistive and force-free transitions, and
  polarized radiation feedback.
- `phydrax.applications.astrophysics` owns observer screens, GR rays/events,
  fast-/slow-light medium sampling with chart/path/snapshot-bound ray-segment routes,
  MNY96 Stokes-$I$/validated-$K_2$ synchrotron with reference-unqualified
  polarization/Faraday, typed ray transfer, Jy Stokes images, visibility closure
  products, neutral payloads, and fixed-branch inference.
- `phydrax.applications.numerical_relativity` owns Z4c evolution and coupling,
  marginal/apparent/quasilocal/Hamilton-evolved event-horizon products, corrected
  characteristic/BMS products, fixed-epoch block-AMR specialization, distribution,
  typed committed-shard restart, output/checkpoint receipts, and exact production
  bindings.
- `phydrax.interchange` owns inert, rights-aware host artifact admission.

The focused guides document [geometry](guides_black_hole_geometry.md),
[perturbations/QNMs/scattering/Hawking](guides_black_hole_perturbations.md),
[relativistic matter](guides_relativistic_matter.md),
[GR imaging](guides_black_hole_imaging.md),
[numerical relativity](guides_numerical_relativity.md),
[execution/qualification](guides_black_hole_execution.md), and the
[source/rights ledger](black_hole_sources.md).

Horizon terminology is never interchangeable. Stationary Kerr supplies a Killing
horizon. A converged `MOTSSolveResult` is only a marginal-surface candidate.
`ApparentHorizonResult` requires complete outermost-search certification.
`QuasilocalHorizonWorldtube` plus `DynamicalHorizonBalancePlan` supplies
isolated/dynamical balance evidence. `OfflineEventHorizonTrace` requires a completed
global spacetime history and backward Hamilton evolution of terminal null covectors and
positions.

Likewise, a complex QNM root with independent radial qualification, a real-frequency
scattering channel, and an exact-state-bound Hawking spectrum have distinct boundary
data, status and qualification.
GR ray capture is a trajectory event and does not assign any horizon class.

Every external product requires source artifact, producer/version, model/coverage,
checksum/size, license/source/attribution, explicit use rights, frame, epoch, scale and
differentiability provenance. No external data access occurs in JIT, pytest, or
executable documentation.
