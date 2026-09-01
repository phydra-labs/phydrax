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
