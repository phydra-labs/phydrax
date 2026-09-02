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

`phydrax.applications.reacting_flow` owns the current gas-phase reacting-flow
candidate. `ReactiveConservedLayout` stores

```text
U = (rho, rho Y_1, ..., rho Y_(S-1), rho u_1, ..., rho u_d, rho E),
rho Y_S = rho - sum_(s=1)^(S-1) rho Y_s .
```

`ReactingGasModel` evaluates ideal-mixture thermodynamics with explicit formation
enthalpies and bounded temperature inversion. `ChemicalMechanismCompiler` accepts a
prepared gas-phase mechanism and returns immutable compiled arrays with element,
charge, and energy evidence. `MixtureAveragedTransportPlan` applies Wilke mixture
properties and a conservative correction velocity. `StefanMaxwellTransportPlan` is a
bounded dense research route; it is not silently selected from the mixture-averaged
plan.

`ReactiveStructuredFiniteVolumePlan` binds `ReactiveEulerSystem` to prepared structured
or mapped finite-volume geometry. `prepare_runtime()` uses the existing SSPRK runtime
with an explicit Rusanov positivity fallback. `ReactiveStrangPlan.advance()` executes
transport half-step, chemistry step, transport half-step on its fixed substep schedule
and commits the complete FV runtime tree only when every stage and final cell is
admissible. `ReactiveIMEXPlan` instead uses explicit FV transport and a fixed-count
implicit-trapezoidal chemistry iteration; a failed nonlinear residual rolls back the
macro step.

```python
from phydrax.applications import reacting_flow

layout = reacting_flow.ReactiveConservedLayout(gas_model, 3)
compiled_mechanism = reacting_flow.ChemicalMechanismCompiler().compile(
    mechanism,
    gas_model=gas_model,
)
transport = reacting_flow.ReactiveStructuredFiniteVolumePlan(
    layout,
    fv_method,
    boundaries,
).prepare_runtime(fv_discretization)
advance = reacting_flow.ReactiveStrangPlan(transport, compiled_mechanism)
state = advance.initial_state(initial_conserved)
result = advance.advance(state, step_size)
```

`ReactiveClosureTargetPlan.build()` accepts explicit species source, heat-release,
species/heat flux, and scalar-dissipation arrays and reports species-source and
diffusive-mass closure; it does not infer targets from a trajectory.
`ReactiveFlowStatisticsPlan.evaluate()` applies caller-supplied positive cell weights
and reports Reynolds/Favre velocity, temperature/species covariances, element amount,
internal energy, enthalpy, and optional closure-target heat release.

`LowMachReactingFormulation` is a separate divergence-constraint formulation with
thermodynamic pressure; it deliberately does not inherit incompressible MAC projection
semantics. It is neither the compressible Euler/FV route above nor an automatic
all-speed switch.

`CanteraYAMLAdapter` is a host-only importer for an explicit ideal-gas, SI-mol,
NASA-7/NASA-9 gas-phase subset and selected elementary, three-body, falloff, PLOG, and
Chebyshev reactions. Unsupported features are reported before import.
`CanteraReferenceAdapter` accepts one host scalar state at a time and rejects JAX
arrays/tracers; it is a non-differentiable reference provider, not the execution
thermodynamics. No Cantera package, file, mechanism, or reference result is supplied
implicitly.
