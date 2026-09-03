# Chemical kinetics

PHYDRAX chemical mechanisms use one immutable species/phase schema across continuum,
particle, lattice-Boltzmann, deterministic, and stochastic execution. Preparation
normalizes units, certifies elemental and charge balance, lowers reaction records to
fixed JAX structures, and assigns a structural mechanism identity.

## Species and phases

`ChemicalSpeciesSchema` declares names, phase membership, molar masses, elemental
composition, and integer charge. `ChemicalPhaseSpec` distinguishes volume and surface
measures and carries standard concentrations and surface site densities. Charge is
always explicit, including for neutral mechanisms.

## Thermodynamics

`PolynomialSpeciesThermodynamicsPlan` supplies a conservative internal-energy model
for particle materials. `NASASpeciesThermodynamicsPlan` evaluates piecewise NASA-7 or
NASA-9 heat capacity, enthalpy, internal energy, entropy, and Gibbs energy. Runtime
evidence reports interval selection and temperature margin; out-of-range states fail
rather than extrapolate silently.

## Mechanisms and rates

`ChemicalMechanismIR` contains `ChemicalReactionSpec` records and prepares a
`PreparedChemicalMechanism`. Species source is formed only by multiplying net reaction
progress by certified stoichiometry, so elemental and charge conservation remain
structural.

Supported rate plans include modified Arrhenius, third-body, Lindemann, Troe, PLOG,
Chebyshev, photolysis, and Butler-Volmer kinetics. Thermodynamic reversibility derives
the reverse rate from Gibbs energy and standard concentrations. Exact-zero reactants
produce exact-zero progress without logarithmic clipping.

## Reactors and transport

`ChemicalReactorPlan` supports isothermal or adiabatic constant-volume and
constant-pressure batch reactors. Adiabatic state stores total internal energy or total
enthalpy; temperature is recovered by bounded inversion. Stiff integration supports
the native adaptive Rosenbrock-W and variable-step BDF substrates.

`ThermochemistryProcessPlan` consumes the same prepared mechanism inside conservative
multispecies transport. Particle reaction processes add only location measures—bulk,
internal surface, or outer surface—without redefining kinetics.

## Mechanism interchange and inference

`load_chemical_mechanism_yaml` imports the canonical PHYDRAX YAML surface, converts
declared units to SI, and rejects unsupported rate or thermodynamic forms. A
`ChemicalCalibrationPlan` applies additive, multiplicative, log-multiplicative, or
bounded coordinates to Arrhenius parameters without changing mechanism structure.

## Failure semantics

No pathway clips species amount, concentration, temperature, or rate parameters. A
failed thermodynamic inversion, invalid rate, negative candidate, conservation defect,
or nonlinear solve rejects the complete attempted update.

## Reacting-flow application facade

`phydrax.applications.reacting_flow` composes the canonical chemical and homogeneous
gas owners. `ChemicalComponentCatalog` owns component identity, molar masses,
elements, charge, and provenance. `ChemicalSpeciesSchema` owns phase-specific species
occurrences and an explicit gas standard pressure. Species calorics feed
`IdealGasReferenceHelmholtzTerm`; ideal mixing and any residual term form one
`HomogeneousHelmholtzPlan`.

The canonical gas state is

```text
U = (rho_1, ..., rho_S, rho u_1, ..., rho u_d, rho E),
rho = sum_s rho_s .
```

There is no dependent-species reconstruction and no separate density component.
`HomogeneousMixtureEulerSystem` owns primitive/conserved conversion, density-energy
recovery, pressure, temperature, frozen-composition sound speed, fluxes, reflection,
and admissibility. `PreparedChemicalMechanism` owns reaction-rate and stoichiometric
identity. Reaction sources write `M_s omega_s` into every species slot and exactly zero
into momentum and total energy: reference and formation chemical energy already live
inside `rho E`. Heat release is diagnostic and is never added again as an energy source.

`MixtureAveragedTransportPlan` applies Wilke mixture properties and a conservative
correction velocity. `StefanMaxwellTransportPlan` is a bounded dense research route;
it is never silently selected. Both use the canonical homogeneous state, produce all
`S` species fluxes with zero total diffusive mass flux, and add full species-enthalpy
transport to the energy flux.

`ReactiveStrangPlan` and `ReactiveIMEXPlan` consume an existing
`PreparedFiniteVolumeRuntime`, the exact canonical gas system bound by that runtime,
and a matching `PreparedChemicalMechanism`. They own only schedule, complete
accepted-state rollback/restart, and evidence:

Preparation order is explicit: construct the canonical homogeneous system; compile it
with the selected finite-volume discretization and method; build the existing
`PreparedFiniteVolumeRuntime`; prepare the canonical mechanism; then pass those two
prepared owners to `ReactiveStrangPlan` or `ReactiveIMEXPlan`. No application-local
Euler, mechanism compiler, or finite-volume wrapper participates.

`ReactiveClosureTargetPlan.build()` accepts explicit full-species source and flux
arrays, diagnostic heat release, heat flux, and scalar dissipation. It reports
species-source and diffusive-mass closure and cannot inject chemical energy.
`ReactiveFlowStatisticsPlan.evaluate()` applies caller-supplied positive cell weights
and reports Reynolds/Favre velocity, temperature/species covariances, element amount,
canonical internal energy/enthalpy, and optional closure-target heat release.

`LowMachReactingFormulation` is a separate full-species divergence-constraint model at
uniform thermodynamic pressure. It uses canonical thermodynamic response derivatives
and deliberately does not inherit incompressible MAC projection semantics. It is
neither the compressible Euler/FV route nor an automatic all-speed switch.

`CanteraYAMLAdapter` is a host-only importer for an explicit ideal-gas, SI-mol,
NASA-7/NASA-9 single-gas-phase subset and selected elementary, three-body, falloff,
PLOG, and Chebyshev reactions. It builds the canonical component catalog, gas phase
with explicit standard pressure, schema, homogeneous Helmholtz model, and prepared
mechanism. Unsupported or ambiguous standard-state features fail before import.
`CanteraReferenceAdapter` accepts one host scalar state at a time and rejects JAX
arrays/tracers; it is a non-differentiable reference provider, not execution
thermodynamics. No Cantera package, file, mechanism, or reference result is supplied
implicitly.
