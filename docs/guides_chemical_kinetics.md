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

## Conditional-affine chemical transitions

`ChemicalConditionalAffinePlan` compiles a user-declared conditional-affine
factorization without sampling a Jacobian. The physical species vector always
remains authoritative. `affine_species` selects the coordinates advanced by a
frozen affine operator, while `driver_species` names auxiliary concentrations
that parameterize reaction coefficients. A species may appear in both roles,
but a predicted driver never replaces its physical endpoint.

For every forward and active reverse direction, preparation requires either:

- exactly one affine pivot with mass-action exponent one, with every remaining
  concentration factor declared as a driver; or
- no pivot, with every concentration factor declared as a driver, producing an
  affine forcing channel.

Concentration dependencies inside third-body, Lindemann, Troe, and surface
coverage rate plans must also be drivers. Temperature, pressure, photolysis,
and electrochemical runtime values remain explicit physical inputs. Unknown
rate-plan types fail certification.

`ChemicalConditionalAffineCertificate` records the directional channels,
selected pivots, hidden rate dependencies, and every rejection reason.
`prepare` succeeds only when all active directions certify. Runtime
`assemble` applies one positive multiplier per base reaction to both forward
and reverse channels before constructing the affine operator.

The frozen affine block is advanced without an inverse. If directional
progress is `r = K x + d`, then `A = N_l^T K` and `b = N_l^T d`. The integrated
reaction extent is evaluated with the existing phi-function actions,

```text
delta_xi = h K phi1(h A) x + h^2 K phi2(h A) b + h d
q_next = q + N^T delta_xi
```

so every invariant in the left nullspace of stoichiometry remains structural.
The implementation stores one pivot index per channel instead of allocating a
dense dynamic `K`. Negative or nonfinite reconstructed species, matrix-action
failure, and invariant defects are reported through
`ChemicalConditionalAffineResult`; no value is clipped or replaced.

`ChemicalConditionalAffineOperator` is the research-tier learned wrapper. It
predicts midpoint driver values, optionally applies
`StoichiometricRateCorrection`, and returns the reconstructed full state.
Operator-level normalization and output pipelines are intentionally rejected:
physical chemistry sees canonical units, while
`ChemicalConditionalAffineScaling` scales neural features internally.

This capability is a local transition, not a general stiff-ODE solver or an
exact time-ordered LTV exponential. Use BDF or Rosenbrock-W when no useful
partition certifies, when the deployment state lies outside trained support,
or when the returned status is unsuccessful. Fallback remains caller-owned
and explicit.

The architecture is an independent PHYDRAX implementation informed by the
[MENO paper](https://doi.org/10.1038/s44387-026-00150-x). The public
[reference implementation](https://github.com/ivanZanardi/pycomet) and
[CC0 zero-dimensional data](https://doi.org/10.5281/zenodo.18305933) are
qualification references, not runtime dependencies.

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

## Qualification

Run `tools/conditional_affine_chemistry_qualification.py` for structural,
reference-solver, invariant, refinement, and differentiation evidence. Run
`benchmarks/conditional_affine_chemistry.py` separately for synchronized
compile and steady-state throughput at declared state and batch sizes. The
benchmark compares no devices implicitly and establishes no speedup claim.

## Failure semantics

No pathway clips species amount, concentration, temperature, or rate parameters. A
failed thermodynamic inversion, invalid rate, negative candidate, conservation defect,
or nonlinear solve rejects the complete attempted update.
