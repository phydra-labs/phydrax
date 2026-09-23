# Reacting flow

Phydrax uses one all-species chemical schema, one homogeneous thermodynamic owner, and explicit transport, chemistry, flow, and qualification routes. No species is reconstructed as a dependent last species. Formation energy is already present in canonical thermodynamic enthalpy/energy; reaction heat is diagnostic and is never added a second time.

## Transport properties

`AbstractGasTransportPropertyPlan` separates property data from mixture flux assembly. The implemented providers are:

- `ReferencePowerLawGasTransportPlan` for the previous explicit reference scaling;
- `KineticTheoryGasTransportPlan` for bounded Lennard–Jones/Neufeld-style collision-integral transport;
- `LogPolynomialGasTransportPlan` for a governed fitted model with explicit temperature support and relative-error bounds.

`MixtureAveragedTransportPlan` and `StefanMaxwellTransportPlan` consume exactly one provider. `TransportPropertyReusePlan` admits accepted-state reuse only through caller-declared logarithmic sensitivity bounds, support intervals, relative-error limits, and a maximum reuse count. Rejected macro-steps never commit cache state.

## Constrained chemical equilibrium

`ChemicalEquilibriumPlan` supports ideal-phase TP, TV, HP, UV, SP, and SV ensembles. Gas phases use ideal partial-pressure activities; liquid and solid phases use ideal within-phase activities and negligible volume. Elements and charge are explicit constraints. Active phases and derivative validity remain in the evidence. Surface phases require their separate site-balanced chemistry and are refused.

`EquilibriumShockPlan` and `DetonationJumpPlan` compose TP equilibrium with Rankine–Hugoniot mass, momentum, energy, and driven sonic residuals. They are reference/initialization solvers, not transient flow replacements.

## Chemical explosive modes

`ChemicalExplosiveModePlan` differentiates the exact prepared mechanism, removes elemental and charge conservation directions, computes biorthogonal left/right modes, reports conditioning and separation, tracks a mode by overlap, and evaluates reaction participation, species explosion indices, and caller-supplied source projections. Degenerate or ambiguous modes fail closed. CEMA is diagnostic and is not a universal flame-front definition.

## Spatial low-Mach reacting flow

`LowMachReactingFlowPlan` is the conservative periodic structured profile. Its accepted state contains face velocity, every `rho Y_s`, `rho h`, scalar thermodynamic pressure `p0`, mechanical projection pressure, time, and optional accepted transport-reuse state. Temperature is recovered from enthalpy. A two-node iterative SDC/trapezoidal update composes conservative scalar advection, full mixture diffusion, exact chemistry, EOS response, and a variable-density target-divergence projection. Constant, prescribed, and closed-chamber pressure modes are distinct. Candidate state, conservation, EOS drift, projection defect, SDC residual, and atomic rollback are explicit.

The initial profile requires periodic axes and synchronized stepping. It is not an automatic incompressible route or all-speed switch.

## AMR, ALE, sources, and scheduling

`ReactingAMRSynchronizationPlan` is a post-reflux chemistry specialist for the
existing block-AMR runtime with level subcycling disabled. A configured
`specialist_synchronization` hook receives
`(level, hierarchy, end_time, interval_dt, args)` for the exact completed level
interval; it must not infer either time from hierarchy state. AMR payloads end in all
species mass densities and enthalpy density. `FixedConnectivityReactingALERemapPlan`
conserves species and enthalpy extensives under supplied old/new cell measures.
`EnergyDepositionSourcePlan` uses a smooth finite-duration pulse whose volume integral
and cumulative work equal the declared energy. `ChemistryWorkSchedulePlan` updates
deterministic worker assignments only from accepted measured RHS counts.

These narrow profiles do not claim a complete moving-engine DNS or arbitrary AMR low-Mach projection.

## Learned chemical transitions

`LearnedChemicalTransitionPlan` consumes a units-exact feature schema, model and training rights manifests, a reaction-extent model, and an uncertainty model. Extents are mapped through the exact stoichiometric matrix, so element and charge closure are structural. Unsupported, uncertain, nonfinite, negative, or invariant-failing lanes execute the exact prepared mechanism route and expose the fallback reason. Route transitions invalidate smooth derivatives.

## Qualification

`reacting_flow_candidate_profiles()` and `reacting_flow_candidate_campaigns()` create unreleased, leakage-controlled profiles. `tools/reacting_flow_closure_qualification.py` is synthetic numerical evidence only. Governed property, equilibrium, flame, extinction/reignition, wall, DNS, and experimental artifacts are required for scientific release. See [Reacting-flow sources](reacting_flow_sources.md).
