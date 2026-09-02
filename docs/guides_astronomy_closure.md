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

Every external product requires source, version, checksum, license, frame, epoch, scale,
coverage, and differentiability provenance. No external data access occurs in JIT,
pytest, or executable documentation.
