# Soft condensed matter

PhydraX treats soft condensed matter as a composition of existing finite-element, lattice-Boltzmann, MAC, atomistic, stochastic-path, and UQ owners. There is no separate soft-matter solver stack.

## Passive phase fields

`BinaryPhaseFieldModel` provides the same binary thermodynamic closure used by
the free-energy lattice-Boltzmann method. `AllenCahnFEMPlan` and
`CahnHilliardFEMPlan` prepare fixed-mesh convex-split finite-element methods,
including diffuse-interface resolution admission before compilation.
`PhaseFieldAcceptancePolicy` sets absolute and relative energy and mass
tolerances. `step_detailed` returns candidate and accepted states together with
physical evidence:

- Allen–Cahn accepts only a finite converged candidate satisfying its discrete
  energy-dissipation balance.
- Cahn–Hilliard additionally preserves the initial reference mass within the
  declared tolerance.
- Every rejected state leaf is restored exactly from the incoming accepted
  state; nonlinear, finiteness, energy, mass, and work evidence remains
  inspectable.

Prepared methods compose directly with fixed-step production and checkpoint
plans; retries never promote a failed candidate.

## Binary free-energy lattice Boltzmann

`FreeEnergyLBMMethod` is a bounded Model-H realization with separate hydrodynamic and conservative phase populations. Its diagnostics retain mixture mass, phase mass, bulk and gradient free energy, kinetic energy, equilibrium moment defects, Mach number, capillary number, and interface resolution. Collision, boundary routing, thermodynamic admissibility, conservation, energy, and population checks form one atomic acceptance predicate. The method’s `maximum_cells` guard rejects an oversized grid before state allocation.

Chemical-potential-gradient and stress-divergence forcing remain explicit alternative discrete representations. Wetting requires a wall normal and mask together; natural wetting and periodic no-flux identities are not inferred from missing boundary data.

## Passive nematic–MAC composition

`MACNematicCouplingPlan` binds a passive `PreparedNematicDynamics` to compatible periodic `PreparedMACOperators` on the same grid. It rejects activity, grid mismatch, nonperiodic boundaries, and cell counts beyond `maximum_cells`.

Cell velocity and velocity gradient are reconstructed from staggered face values. The passive Beris–Edwards stress is lowered to faces with the exact weighted transpose of that reconstruction. Consequently the reported fluid work and nematic stress work are equal and opposite up to `work_tolerance`. A coupled step proposes Q and all face-velocity components together, checks total nematic-plus-kinetic energy, and either commits every leaf or rolls every leaf back.

## Atomistic observables and protocols

`StaticStructureFactorPlan` evaluates the normalized coherent static structure factor for an explicit set of wave vectors and retains per-frame values. `LaggedCorrelationPlan` evaluates all-origin MSD and VACF at explicit integer lags. Positions passed to `lagged_msd_vacf` must already be unwrapped; applying a minimum-image displacement would corrupt long-time diffusion.

`DiffusionFitPlan` fits the MSD slope only over its declared lag window. `DiffusionEvidence` retains the slope, intercept, standard error, R², minimum origin count, Green–Kubo VACF integral, and Einstein/Green–Kubo discrepancy. A coefficient is successful only when every configured evidence predicate passes.

`SoftMatterAtomisticProtocol` validates four bounded compositions of the existing atomistic runtime:

- monodisperse unbonded Lennard-Jones liquid;
- fixed 80:20 explicit-parameter binary glass;
- non-element WCA Langevin colloid;
- one fixed linear harmonic-bond chain with bonded LJ exclusions.

The colloid and polymer protocols require the existing BAOAB integrator. `langevin_fdt_report` receives the matching prepared thermodynamic-state table and records the discrete Ornstein–Uhlenbeck fluctuation–dissipation identity plus stable particle-ID/step/operator/realization noise addressing. WCA additionally requires energy shifting at `2¹ᐟ⁶σ`; a merely truncated LJ energy is not admitted. Protocol validation does not create a new integrator or potential engine.

A `MolecularCoarseMapEvaluation` retains mass, charge, total-force, and total-momentum residuals. Coarse force matching is an equilibrium force projection. `kinetic_fidelity_claimed` is always false: this map does not claim preservation of fine dynamical kinetics or memory.

## Polymer physics

Polymer capabilities are split by semantic owner:

- [Polymer liquids](guides_polymer_liquids.md): FENE–WCA particles, polymer observables, PRISM, Brownian/GLE runtimes, scattering adapters, and equilibrium Green–Kubo rheology.
- [Polymer field theory](guides_polymer_field_theory.md): linear and acyclic-branched periodic SCFT, fixed-branch derivatives, continuation, isotropic cell optimization, finite-group symmetry evidence, real partial-saddle FTS, and bounded complex-Langevin FTS.
- [Polymer construction](guides_polymer_construction.md): explicit material recipes, stable-ID topology lowering, admitted adapters, transactional reaction epochs, periodic winding, and network observables.

Particle topology, construction recipes, PRISM site mixtures, reaction templates, and field-theory contour trees remain distinct representations. Cross-representation operations always produce an explicit lowering record, tabulated form factor, or observation product.

## Discrete stochastic thermodynamics

`normalized_discrete_path_thermodynamics` consumes canonical `PathBuffer` values and correlated uncertainty from the existing path-sampling owner. Heat is positive into the system, so the conventions are

- first law: energy change = work + heat;
- medium entropy change: minus inverse temperature times heat;
- total entropy production: system entropy change plus medium entropy change.

Forward and reverse path weights are normalized separately. The result retains every path, normalized probabilities, effective sample sizes, first-law residuals, local detailed-balance residuals, the integral fluctuation average, reversal residual, and autocorrelation-aware uncertainty. Missing, nonfinite, oversized, or inconsistent evidence cannot produce a successful result.

## Scope and nonclaims

These APIs do not claim polymer entanglement or reptation, hydrodynamic interactions, fluctuating hydrodynamics, multicomponent phase fields, resolved-hydrodynamic colloids, driven-flow rheology, general reactive MD kinetics, automatically faithful coarse kinetics, automatic phase discovery, or thermodynamic-limit inference. Current network reactions are construction epochs; current rheology is equilibrium finite-system Green–Kubo evidence; current GLE kernels require separately supplied calibration. Tests, smoke output, and benchmarks are engineering evidence, not release or material-validation evidence. Benchmark reports retain logical payload bytes separately and do not substitute unavailable allocator peaks with estimates.
