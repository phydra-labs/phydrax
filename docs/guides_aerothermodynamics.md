# Production aerothermodynamics

Phydrax composes ionized multitemperature continuum flow, multigroup radiation,
finite-rate gas--surface chemistry, porous material response, DSMC, hybrid ownership,
turbulence, and topology through exact immutable profiles. The application facade does
not define a second flux, thermodynamics, radiation, or particle runtime.

## Support identity

`AerothermodynamicSupportTuple` binds the gas, transport, chemistry,
electromagnetic, radiation, surface, material, kinetic, hybrid, turbulence,
discretization, topology, backend, precision, species, modes, and group counts. A
`released=True` capability status requires independent scientific, performance,
operational, and security gates plus explicit evidence identities.

## Ionized gas

`IonizedMixtureThermodynamicsPlan` wraps the explicit thermal-mode state from the
high-speed-flow layer. One negative electron species and one pressure-bearing
ideal-degrees-of-freedom electron mode are mandatory. All species remain conserved;
quasi-neutrality is evidence rather than dependent-species reconstruction.

`IonizedMultitemperatureEulerSystem` adds electron pressure to the heavy-particle
pressure and acoustic bound. `IonizedMultitemperatureNavierStokesSystem` retains the
canonical molecular/modal diffusive tensor. `ReactionTemperatureSpec` assigns heavy,
mode, electron, or geometric-mean temperature to every plasma reaction.

`FixedWorkThermochemicalSourcePlan` uses fixed-work implicit source steps with a
physical positivity line fraction. Element, charge, and total-energy ledgers gate the
accepted state.

## Plasma transport and electrostatics

`AmbipolarPlasmaTransportPlan` projects provisional species diffusion onto simultaneous
zero-total-mass and zero-current constraints. It reports the actual electric current,
ambipolar field, entropy production, and maximum diffusivity.

`ElectrostaticPlasmaCouplingPlan` projects cell charge to the existing compatible
cochain Poisson solve and returns cell electric field, Lorentz force, Joule work, and
electron-energy exchange. Projection matrices are part of the plan identity.

## Radiation

`NonLTELevelPopulationPlan` evaluates explicit electronic level populations from
species amounts, electron temperature, degeneracies, energies, and optional departure
coefficients. `NonLTERadiationCoefficientPlan` accumulates bound--bound absorption and
spontaneous emission into exact spectral groups.

`MultigroupRadiationMatterProcessPlan` advances frozen-coefficient absorption/emission
analytically over fixed subcycles. Gas total energy, electron-mode energy, and radiation
energy commit atomically. Transport and matter light speeds remain distinct identities.

## Gas--surface chemistry and materials

`SurfaceSpeciesSchema` stores site occupancy, elements, and charge. A
`PreparedGasSurfaceMechanism` rejects reactions that violate sites, elements, or charge.
`ReactingPlasmaWallPlan` produces the exterior gas state, conservative species and
energy flux, catalytic heat, electric current, blowing velocity, and persistent surface
candidate.

`PorousAblatingMaterialPlan` advances solid constituents, pore gas, energy, and porosity
with finite-rate decomposition. Reaction yields must conserve closed-system mass.
`ConjugateAerothermalInterfacePlan` transfers extensive heat and pore-gas mass on a
common refinement.

`FixedConnectivityRecessionPlan` converts accepted surface mass flux to normal motion.
`ConservativeRecessionRemapPlan` remaps extensive gas and material states with overlap
matrices whose columns sum to one.

## DSMC and hybrid ownership

`phydrax.discretization.dsmc` provides fixed-capacity species, particles, structured
collision cells, streaming, VSS/VHS elastic collisions, Larsen--Borgnakke-style internal
redistribution, bounded two-to-two chemistry, and gas--surface exchange. Slot identity,
incarnation, random key, and capacity remain explicit runtime state.

`DSMCProductionPlan` performs one streaming/collision/internal epoch atomically and
returns cell moments. `MaxwellianReservoirPlan` corrects sampled velocities to exact mean
and translational energy.

`FixedContinuumDSMCInterfacePlan` computes one uncertainty-weighted common flux and
returns equal-opposite extensive exchanges. `DynamicHybridOwnershipPlan` applies
enter/leave hysteresis, dwell time, adjacency buffers, and particle-capacity refusal.
Ownership changes only between accepted physical steps.

## Turbulence and topology

`SSTTurbulencePlan` names the exact `sst-1994-m`, `sst-2003-m`, or `sst-2003-v`
variant. `DelayedDetachedEddyPlan` names SA-DDES, SA-IDDES, or SST-DDES and consumes a
geometry-owned `HybridRANSLESGridScalePlan`.

`MulticomponentAdmissibilityFilterPlan` performs a cell-mean-preserving convex filter
for any `AbstractAdmissibleSystem`. `HighEnthalpyAMRIndicatorPlan` retains every named
shock, chemistry, radiation, rarefaction, wall, or turbulence indicator rather than
collapsing them into an opaque scalar. `AerothermodynamicALEPlan` gates motion on swept
volume and geometric-conservation evidence.

## Production profiles

The facade exposes exact compositions:

- `IonizedContinuumProfile`
- `RadiatingContinuumProfile`
- `AblatingEntryProfile`
- `RarefiedDSMCProfile`
- `FixedContinuumDSMCProfile`
- `DynamicContinuumDSMCProfile`

`AerothermodynamicProductionPlan` advances the applicable source, radiation, wall,
material, DSMC, interface, and ownership stages, then commits or rolls back the complete
runtime state.

## Qualification

`AerothermodynamicValidationCase` requires a rights-bearing reference artifact, named
observables, reference values, and positive uncertainties. Campaign acceptance uses
normalized uncertainty, not arbitrary absolute tolerances. No external mechanism,
opacity, collision, surface, material, mesh, or experimental dataset is bundled by this
module.

The initial implementation is a production substrate, not a universal release claim.
Every concrete physical support tuple begins unreleased until its exact RAM-C, EAST,
FIRE, HyMETS, TACOT, turbulence, rarefied-flow, restart, distribution, and capacity
evidence is admitted.
