# Protein folding: identity, physical models, and qualified inference

`phydrax.applications.protein_folding` separates molecular identity, static
structure hypotheses, parameterized physical realizations, experimental
inference, and conditional coordinate proposals. It is not a pretrained folding
predictor. Imports do not download structures, run a provider, assign protonation,
construct force-field parameters, or establish biological folding accuracy.

See the [public API](api/applications/protein_folding.md), the
[runnable cookbook](cookbook/protein_folding.md), and the
[source and rights disposition](biophysical_sources.md).

## Delivered surfaces and their scientific meaning

```python
from phydrax.applications import protein_folding as protein

construct = protein.ProteinConstruct(("A",), ("NLYIQWLKDGGPSSGRPPPS",))
residue = protein.ResidueKey("A", 0)
atom = protein.ProteinAtomKey(residue, "CA")
```

| Surface | Delivered contract | Not established by numerical success |
|---|---|---|
| P1: root, `interchange`, `workflows` | Stable identity, explicit chemistry, conservative force-field handoff, short NVE/NVT, geometry evidence | Experimental folding, equilibrium sampling, or kinetic timescales |
| P2: `experiments` | Joint fluorescence/equilibrium/relaxation models, named-parameter fitting, likelihood rank, actual posterior sampling | Applicability to irreversible unfolding or global identifiability |
| P3: `thermodynamics` | Matched-composition paired enthalpy, finite-interval heat-capacity fit, experimental free-energy closure | Absolute folding free energy from enthalpy alone |
| P4: `potentials` | Conditional single-site rotamer free energy, exact enumeration or qualified implicit BP | A calibrated all-atom side-chain or protein folding model |
| G1: `generation` | Offline provider admission, fixed-chemistry conditional flow matching, bounded sampling and persistence | Pretrained prediction, Boltzmann weights, coordinate likelihood, calibrated confidence |
| X1: `hybrid` | Elastic protein network plus native rigid nucleotide mechanics and explicit cross interactions | Atomistic protein chemistry or calibrated protein–nucleotide binding |
| X2: `cotranslation` | One-bead-per-residue activation epochs, native MD dwells, insertion work, checkpoint replay | Biological translation timing or non-native folding without separate evidence |

The numerical owners remain `phydrax.atomistic`, `phydrax.pgm`, `phydrax.uq`,
`phydrax.dynamics`, and the native rigid-body and series APIs. The protein
application does not create a second molecular engine or a universal biomolecule
identity owner. Shared generation contracts are public through both biological
`generation` leaves; import them through the relevant application leaf.

## P1: retain identity before preparing physics

### Construct, source, selection, and realization

`ProteinConstruct` stores ordered canonical amino-acid chains. `ResidueKey`
identifies a zero-based position within a named construct chain, not a PDB author
residue number. `ProteinAtomKey` adds the canonical atom name. A construct can
exist without complete coordinate coverage; coordinate absence does not change
its sequence.

`ProteinSourceAtom` retains the source record, model, author chain/residue,
insertion code, alternate location, occupancy, element, and optional label
numbering beside the construct-local atom key. `ProteinStructureHypothesis`
requires finite coordinates in an exact length unit, a complete original
`ScientificArtifactEnvelope`, and inherited `ReferenceArtifactManifest` records.
Each hypothesis contains one explicitly selected model and one row per selected
chemical atom. Duplicate records/atoms and unresolved alternate conformers are
refused. Named provider confidence remains provider-specific; it is not an
energy or an equilibrium probability.

Use `interchange.protein_hypothesis_from_pdb_records` after the neutral
`phydrax.atomistic.interchange.read_pdb_atom_records` and `select_pdb_model`
operations. Supply the explicit map from `(author chain, author residue number,
insertion code)` to `ResidueKey`. An optional record-ID-to-canonical-name map
handles an externally resolved naming convention without replacing the original
source identity. Keep the original artifact with every model and alternate
conformer. `ProteinHypothesisView` retains the full hypothesis tuple and a
separate selection policy rather than deleting unselected candidates.

Binding creates `parameterized-protein-realization` lineage; dynamics creates a
separate trajectory artifact. Neither overwrites the raw coordinates. Keep the
raw artifact, hypothesis ID, chemistry source, parameter manifests, mapping,
binding ID, qualification ID, and trajectory artifact together.

### Exact admitted all-atom chemistry

`ResolvedProteinChemistry.profile` is
`"canonical-L-single-chain-explicit"`. The physical binding profile is an
**isolated, nonperiodic, uncapped canonical-L single chain** with every heavy
atom and hydrogen supplied. This is narrower than the construct/hypothesis
surface. Required inputs include:

- Unique atom keys and atomic numbers for the complete chemical inventory,
  including terminal `OXT`.
- One source-supported state label and positive expected hydrogen count per
  residue. Admitted labels are `standard`, `protonated`, `deprotonated`,
  `delta-tautomer`, `epsilon-tautomer`, and `thiol`. Cysteine requires `thiol`;
  histidine requires `protonated`, `delta-tautomer`, or `epsilon-tautomer`.
- Explicit N terminus `NH2` or `NH3+`, C terminus `COOH` or `COO-`, and a chemical
  realization source ID. These are caller-audited assignments, not predictions
  from pH or coordinates.
- A fully parameterized native `PreparedAtomisticForceField`, complete stable
  atom-ID map, and parameter rights. Coefficients must already use the exact
  declared system energy unit; mismatches are refused rather than relabelled.

Caps, disulfides, PTMs, covalent ligands, solvent, ions, virtual degrees of freedom,
periodicity, and multichain all-atom binding are outside this profile. The binder
requires elemental active material to match the chemical inventory exactly,
checks source/system element agreement, and requires connected topology with
backbone/peptide connectivity. It does not perform chemical completion or prove
that caller-supplied parameters describe the declared protonation state. Missing
atoms cannot be disguised as inactive padding. Use `protein_mapping_coverage`
to inspect missing/unexpected atoms before binding.

`interchange.bind_protein_openmm` accepts an **already parameterized** OpenMM
`System` and a bijection from its particle ordering to original source record
IDs. OpenMM is an optional, lazily used interchange dependency, not the native
execution engine. The complete-conversion report is required. This adapter's
bounded native treatment of OpenMM NoCutoff Lennard–Jones must be explicitly
accepted with `accept_bounded_no_cutoff=True`; retain the warning. A cutoff
larger than one fixture's extent proves a same-configuration comparison, not
unbounded equivalence along arbitrary later trajectories. Unsupported force
terms are not silently discarded.

### Conservative force path and short physical trajectories

`PreparedProteinBinding.evaluate(neighborhood, positions=None)` consumes a
realized native neighborhood state and evaluates the prepared scalar potential.
Forces come from that conservative energy; fixed active atoms retain reactions.
Host identity/rights preparation is outside JIT, while the fixed-shape numeric
potential path is JIT/differentiation compatible under its native validity gates.

`PreparedProteinQualification` requires per-native-bond lower/upper bounds and
reports covalent geometry, non-glycine backbone chirality, nonlocal clashes,
peptide cis/trans planarity, and finiteness. Bounds and clash distances use
`bounds_unit`, chiral volume uses its cube, and peptide tolerance is radians.
It is a declared geometry screen, not a folding basin assignment or an
experimental structure-quality score. Its finite clash-pair capacity is enforced.

`workflows.prepare_protein_dynamics` composes either `VelocityVerletPlan` (NVE)
or `BAOABLangevinPlan` (NVT) with the existing force field, neighborhood, and
constraints. `run_protein_dynamics` additionally requires a qualification bound
to the same binding, finite initial velocity with an explicit velocity unit, a
PRNG key, and trajectory sampling policy. Step size, friction, and temperature
are in the native unit system. Failed initial geometry raises; failed native
execution or final geometry gives a failed trajectory artifact. Inspect both
rollout and geometry evidence before consuming the result. There is no silent
relaxation, hydrogen addition, or atom completion.

Conservative biases must already be in the caller's force-field bundle and must
have a `bias_id`. A biased trajectory is not unbiased kinetic data. The result's
`trajectory_data()` preserves native time units, sample validity, and the
position/momentum feature-state layout. A short isolated NVE/NVT execution is
physical time for that model, not proof of equilibrium, solvent behavior, or
experimental folding time.

## P2: observations, named models, and inference

`ExperimentConditions` contains **paired condition rows**, not an implicit
Cartesian product and not a thermal-ramp trajectory. Prepare the desired
thermal × denaturant grid explicitly. Temperatures must be positive Kelvin,
denaturant and protein concentration nonnegative, and dimer fluorescence needs
positive total monomer-equivalent concentration. `celsius_to_kelvin` performs the
explicit offset conversion; Celsius is not a multiplicative unit conversion.

`ThermodynamicConvention` distinguishes `basis="molar"` from
`basis="single-system"`; use energy/amount units with R or ordinary energy units
with kB respectively. This distinction is not removed by a unit relabelling.
The default concentration unit is mol/m³, with standard concentration 1000
mol/m³, not a numeric convention of mol/L. Every model and its condition rows
must use the same convention.

For thermal models, ΔG means unfolded minus folded and the reference is at zero
denaturant. The constant-ΔCp/linear-denaturant law is:

> ΔG(T, d) = ΔHᵣ(1 − T/Tᵣ) + ΔGᵣ T/Tᵣ + ΔCp[T − Tᵣ − T ln(T/Tᵣ)] − [mᵣ + (dm/dT)(T − Tᵣ)]d.

| Public model | State/observation contract |
|---|---|
| `TwoStateUnfolding` | Reversible monomer F ⇌ U |
| `ThreeStateUnfolding` | Reversible F ⇌ I ⇌ U, with consecutive free-energy differences |
| `DimerTwoStateUnfolding` | N₂ ⇌ 2U; monomer-equivalent fractions and baselines, concentration-dependent mass balance |
| `DimerThreeStateUnfolding` | N₂ ⇌ 2I ⇌ 2U; I-to-U energy is per monomer |
| `RepeatTransferUnfolding` | Heterogeneous open binary-repeat chain; fluorescence observes mean folded fraction |
| `ChevronKinetics` | Isothermal two-state relaxation, observed rate kf + ku |
| `ParallelPathKinetics` | Two F ⇌ U barriers sharing equilibrium free energy; each obeys kf/ku = exp(ΔG/RT), while observation sees summed rates |

`repeat_transfer_statistics` computes the open-chain log partition and folded
marginals with linear-cost log-space forward/backward messages. Its direct
formation-energy inputs use negative values to favor folding/contact; do not
confuse that sign with the unfolding ΔG parameters of the observation model.
No periodic repeat bond is implied.

`FluorescenceExperiment` assigns every row a channel/replicate group with
state-specific nuisance baselines. Available baseline terms are `intercept`,
`temperature`, `denaturant`, and `temperature_denaturant`; all groups share the
same populations at a condition. It requires reversibility and explicit
equilibrium-applicability evidence. Irreversible, aggregation-dependent, or
ramp-dependent data are refused by this equilibrium contract.

`KineticRateExperiment` observes natural log numerical rates in the supplied
time unit; preparation converts to log numerical rates in s⁻¹. Its conditions
must equal the model's reference temperature: neither kinetic model supplies an
activation-enthalpy law. A relaxation trace does not resolve individual parallel
pathways or establish intermediate lifetimes.

Both observation types require finite active measurements and calibrated
positive standard errors, with optional masks. A supplied
`covariance_cholesky` is a lower-triangular **covariance** root over active rows
in mask order, with marginal variances matching the errors. Noise is not fitted
from residual scatter. `source_kind="experimental"` additionally requires a
rights-cleared reference with measured uncertainty. Synthetic declarations
exercise the model and solver; they do not become experimental evidence by
carrying the same array shape.

### Fit, identifiability, and posterior are different results

Use each observation's `parameter_slots()` to obtain required names/units.
`ExperimentParameter` and `NamedParameterMap` share parameters by explicit names
or observation `bindings`. Free coordinates decode as initial + scale × z;
fixed parameters are excluded from free-coordinate sensitivity and uncertainty.
Unknown bindings, missing parameters, and unused parameters are refused.
`prepare_protein_experiments` creates one joint residual/normalized likelihood
for every named model. `fit_protein_experiments` executes native nonlinear least
squares, not a specialized forward-only approximation.

- Check `fit.optimization.successful` independently of
  `fit.identifiability.locally_identifiable`. Likelihood rank comes from the
  noise-whitened sensitivity in standardized free coordinates. Retain singular
  values, threshold, condition number, and null vectors in `free_names` order.
  An isotherm generally cannot identify a full thermal law.
- `fit.covariance` is supplied only for a successful full-rank fit. It is a local
  inverse-Fisher approximation in free physical parameter units, not a posterior
  interval. Rank deficiency returns `None`, not zero uncertainty in null modes.
  Full local rank does not resolve global symmetries such as pathway exchange.
- `protein_experiment_posterior_problem` requires explicit Gaussian priors on z;
  they are not inferred from fit covariance. `sample_protein_experiments` runs
  native NUTS and retains chain/draw axes and diagnostics. Priors do not repair
  likelihood identifiability. Inspect chain diagnostics before interpreting
  samples. `predictive_samples()` returns conditional mean draws, not newly
  simulated measurement noise.
- `phi_posterior` uses aligned WT/mutant posterior draws at the same condition,
  energy convention, and log-rate unit. It propagates their pairing, does not
  clip Φ to [0, 1], and marks a near-zero stability-change denominator invalid.
  Credible intervals are NaN unless every draw passes that denominator gate;
  retain `valid` and `valid_fraction` rather than hiding invalid draws.

Numeric residuals/likelihoods are differentiable. Host rank diagnostics,
optimization decisions, and MCMC orchestration are not pathwise derivatives of
molecular kinetics.

## P3: paired enthalpy and explicitly experimental closure

`ProteinEnsembleComposition` identifies the **entire** box: construct, chemical
state, species counts (including solvent, ions, and cofactors), and parameters.
Externally solvated ensembles can feed this estimator even though P1's physical
binding profile is isolated. Folded and unfolded basin definitions are separate
inputs, not selected afterward by diameter filtering or an apparent stable trace.

`native_enthalpy_series` forms H = U + K + pV from a retained native trajectory
and explicitly supplied pressure and positive volumes in declared units. It
never estimates volume from a protein bounding box. An isolated calculation may
explicitly choose zero pressure. Output remains single-system energy; any
molar transformation is separate. Preserve masks and physical time identity.

Each `EnthalpyReplica` requires a complete, finite, uninterrupted scalar series
on increasing physical time; equilibration evidence; independent replica and
realization IDs; pressure condition; and a source-supported positive upper bound
on correlation time. At least two complete nonoverlapping blocks are required,
with no discarded tail. Every block must span at least five correlation-time
bounds. These declarations must be supported by the actual source; the type
cannot establish physical equilibration for the caller.

`paired_state_enthalpy` requires distinct independent replicas across both
basins and temperatures, at least two replicas per state/temperature, identical
temperature coverage, and matched whole-box composition and pressure. It reports
ΔH = mean(HU) − mean(HF), equally weighting independent replica means. Each
state's variance retains the larger of propagated block-mean and between-replica
mean variance. The result is conditional sampling uncertainty, not force-field
or basin-representativeness uncertainty.

`fit_heat_capacity_slope` needs at least three temperatures. It reports the
finite-interval constant-ΔCp fit, residuals, and propagated sampling covariance.
Residuals expose the inadequacy of constant ΔCp; the covariance is not a posterior
or an allowance for model-form error.

**Enthalpy does not fix the entropy integration constant.**
`close_free_energy_at_reference` requires a rights-cleared reference with
uncertainty, a measured `(T_ref, ΔG_ref)`, and their symmetric positive-semidefinite
2 × 2 covariance. Allowed `closure_kind` values are:

- `"measured-melting-temperature"`, requiring ΔG(Tm) = 0;
- `"experimental-thermodynamic-cycle"`, requiring the caller's closed-cycle ΔG
  and uncertainty. A ligand Kd alone is not a folding free energy.

The closure point must lie within the sampled enthalpy interval. Out-of-interval
evaluation is invalid and returns NaN, not an extrapolated claim. The result
retains experimental dependencies and assumes independent experimental/MD
sources. A synthetic melting control demonstrates uncertainty propagation, not
experimental closure for a real protein.

### Free energies and physical kinetics have separate admission

`workflows.ProteinFreeEnergyWorkflow` composes native FEP, TI, BAR, MBAR, and
targeted-map estimators. Supply independently identified thermodynamic states,
composition, temperature, source ensemble, decorrelation evidence, all sampling
bias IDs, and full physical energies/reduced potentials/work. MBAR origins must
reproduce state counts; TI identifies every quadrature state. Targeted mapping
requires round-trip/support qualification. Inspect native overlap, convergence,
and uncertainty evidence. Configuration weights are not path weights.

`workflows.ProteinKineticWorkflow` accepts reset-aware `TrajectoryData` with
`source_kind="physical-dynamics"`, an SI-convertible time unit matching the data
coordinate, condition identity, and an explicit time-calibration evidence ID.
Optimizer iterations, provider structures, Monte Carlo sweeps, and generative
pseudotime are refused. A non-None `configuration_bias_id` is refused: equilibrium
reweighting alone cannot recover unbiased kinetics. Coarse dynamics needs its
own calibration even if its time unit is expressed in seconds.

VAMP, VAC, TICA, Markov and Chapman–Kolmogorov consumers require positive uniform
physical lags after reset/missing-sample exclusion. `ProteinBasinDefinitions`
uses independently declared native region predicates; observed overlap is an
error and unassigned support is marked −1. First-passage/committor inputs require
physical shooting outcomes and their source, not structure scores. Retain
censoring evidence separately when omitting incomplete first-hit outcomes.

## P4: conditional rotamer free energy

`RotamerGeometryPlan` defines one right-handed backbone frame per construct
residue using three distinct active stable atom IDs and a heterogeneous table of
allowed local point sites. `RotamerParameterPlan` provides source-pinned unary
energies and Gaussian pair energies on unique fixed residue pairs:

> Epair(r) = A exp(−r²/(2σ²)); G(x) = −kBT ln Z(x).

These are caller-defined **single-site** rotamer states, not generated side-chain
atoms or another topology. Widths/sites use the declared length unit; energies
are single-system energies, not molar coefficients; effective temperature is
positive and must equal `sampling_temperature`. Parameter and geometry rights
are admitted separately. No Upside tables, calibrated mini-protein parameters,
or temperature-transferability claim is bundled.

`RotamerFreeEnergyTerm` plugs the scalar G into the native potential program, so
forces derive from the same energy. `inference_method="exact"` uses enumeration
bounded by `maximum_configurations`. `"bethe"` uses implicit sum-product BP with
zero-message initialization on a fixed graph, not history-dependent warm starts.
Trees have an exact normalizer when inference succeeds; loops use the Bethe
approximation. Small numerical residual is not an error bound against exact
loop inference.

On loops, a sufficient global contraction certificate must satisfy
`contraction_bound <= maximum_contraction < 1`. It certifies the admitted branch
and bounds inverse sensitivity, rather than accepting any numerically converged
root. The numeric result separately retains inference, `geometry_valid`,
`derivative_qualified`, contraction, status, and success. Degenerate frames,
inference failure, an unqualified branch, or nonfinite energy yield failed
status and NaN energy. The statuses are `SUCCESS`, `INVALID_GEOMETRY`,
`INFERENCE_FAILED`, `UNQUALIFIED_BRANCH`, and `NONFINITE_ENERGY`.

Differentiate only the accepted fixed-support branch and retain native solver
and derivative evidence. There are no geometry-dependent graph-switch cutoffs.
Atom-energy attribution is the declared G × weights partition, not a unique
physical atomic observable. Exact/BP comparisons establish numerical behavior
of supplied tables, not biological rotamer populations or folding accuracy.

## G1: offline providers and native conditional generation

`generation.import_protein_hypotheses` imports all explicitly mapped local
outputs of shape `(hypothesis, source_atom, 3)`, with one raw source envelope and
provider confidence record per hypothesis. `CoordinateProviderProvenance`
separates output rights, model-weight rights, prepared-input rights, and any
inherited code restrictions. Learned-provider admission requires separate
weight and input rights plus input artifact identities; retain MSA/template
artifacts there. Raw output digest/license must match an admitted output
manifest. Declared egress requires an explicitly authorized destination and
input export rights. Admission performs no network request or provider execution.
ColabFold, AlphaFold3, and SimpleFold are not installed predictors or bundled
checkpoints; code availability does not authorize their weights, outputs,
commercial reuse, or distillation.

`prepare_protein_coordinate_support` binds construct residue/atom tokens to a
single explicit native `AtomisticBatch` template, stable active atom IDs, gauge
anchors, sparse bond/chirality policy, and `CoordinateResourcePolicy`.
`map_protein_hypothesis` requires complete coverage of that declared model
support, preserves the raw hypothesis, checks token/element identity, and
converts length units. An explicitly heavy-atom model support is possible; it
is not the complete all-atom chemical realization required by P1.

`prepare_coordinate_training_data` admits mapped conformers with per-record
source manifests, rights permitting training, condition feature names, and
caller-defined split groups. Training and validation groups must be nonempty
and disjoint; identical canonical conformers across the split are refused.
The caller must establish that its group labels represent real acquisition,
trajectory, or construct independence. All training conformers must satisfy the
declared geometry/gauge policy.

`fit_coordinate_model` actually trains `ConditionalCoordinateVelocity` through
native flow matching and `FunctionalSolver`. It is a dense global model with a
fixed support/order ABI, not permutation equivariance or general sequence
prediction. Mass centering and a proper three-anchor frame handle rigid gauge
without reflections. Resource caps bound atoms, records, pairs, steps, network
width/depth, condition features, samples, and ODE steps; no automatic truncation
or hidden corpus fetching occurs.

`prepare_coordinate_sampler` prepares the numeric sampler;
`sample_coordinate_proposals` returns **every** raw ODE output and separate
canonical view, solver status/validity, conditions, sample IDs, inherited rights,
and sparse geometry qualification. It does not repair or resample rejected
proposals. The integration coordinate 0 to 1 is generative pseudotime, not MD
time. This singular gauge-fixed support exposes no coordinate likelihood,
Boltzmann weights, or calibrated confidence. A low training loss or accepted
bond/chirality screen is not physical qualification.

`save_coordinate_model` uses native pickle-free ML artifacts and retains inherited
restrictions. `load_coordinate_model` requires checksum/size-bound weight rights,
the same prepared support, and admission of all retained parent restrictions.
Training, redistribution, commercial use, and export must be requested and
permitted separately. Rights metadata records authorization; it does not grant it.

## X1: mixed-resolution conservative mechanics

`hybrid.PreparedHybridModel` couples a native `PreparedElasticNetwork` to a
`PreparedNucleotideModel` and `HybridCrossInteractionPlan`. Protein Cartesian
momenta and nucleotide rigid world twists remain distinct owners at one physical
time. `HybridSupportMap` creates reversible disjoint namespaces for protein
DOFs/sites and nucleotide bodies/sites without changing their original IDs.
Padding has identity but never becomes material.

Cross pairs refer to active physical interaction sites, not differential frame
markers. The caller supplies independent soft steric, harmonic linker, and
screened electrostatic coefficients:

> Ucross(r) = ε max(1 − r/a, 0)⁴ + k(r − r₀)²/2 + A exp(−κr)/r.

Coefficients use energy, length, energy/length², energy × length, and inverse
length as appropriate; the signed electrostatic prefactor already includes the
caller's charges/dielectric convention. Zero coefficients disable a term.
Coupled coincident sites fail rather than being regularized into a purported
physical model. Protein reference, nucleotide parameters, and cross parameters
must pass requested-use rights admission, and all three models must share the
exact unit system. Periodicity and Cartesian holonomic constraints are refused.

The scalar energy is the sum of protein network, nucleotide model, and the three
cross components. Site forces are equal and opposite, with native Cartesian
pullback and rigid force/torque transfer. `HybridForceEvaluation` retains full,
mobile, and fixed-support reaction loads; reactions are forces **on** fixed
material. `step` makes a joint kick/drift/kick candidate with no heat bath.
Inspect `HybridStepResult.successful` before accepting it. Finite-step anisotropic
rotation is not exact free-rotor flow or exact energy preservation. Reference
network stiffness and force/torque balance are numerical mechanics diagnostics,
not a calibrated molecular binding or folding result.

## X2: cotranslation and activation epochs

`CotranslationProtocol` admits one nonperiodic single chain with one distinct
stable coarse particle ID per residue and a caller-supplied complete stage
schedule. Fixed capacity ordering is preserved. At each stage, active material
is exactly the nascent sequence prefix plus the persistent environment; future
residues are dormant, not zero-mass physical particles.

Each `CotranslationStage` owns a prepared native dynamics runtime, positive dwell
steps, source ID, and nascent length. The initial nascent state is supplied once.
Later epochs activate exactly one residue or perform a same-length protocol
switch. Insertions require a matching standard-code RNA sense codon (no automatic
T-to-U conversion), explicit positions and momenta in inserted-ID capacity order,
and an optional work bound. Same-length switches have `codon=None`. The complete
protocol must reach the full construct.

`RibosomeBoundaryPotential` provides a harmonic nascent-end tether and
caller-declared soft excluded spheres in native units, not reconstructed
ribosome chemistry. These supports are fixed within a stage; release or changes
belong at a work-accounted epoch boundary. `transition` composes the native
`TopologyEpochTransition`; `run` executes native MD and atomic activation,
returning per-stage `SampledSeries`, insertion ledgers, a cursor, success and
refusal. A rejected activation retains the exact preactivation cursor for
rollback/replay, without changing event-addressed randomness. Never form lag
pairs or interpolate across activation boundaries. Checkpoint read/write uses
the native state owner and verifies protocol, epoch, schedule, and integrity.

Dwell time is `dwell_steps × step_size` in the runtime's unit. Biological timing
requires both a rights-cleared `timing_calibration` with uncertainty and a stated
`timing_calibration_scope`. MD dwells alone do not calibrate codon kinetics.
The default is `reference_conditioned=True`; setting it false requires independent
`non_native_qualification` with uncertainty. Neither declaration substitutes for
the underlying evidence.

`NascentChainObservations` reports reference-contact similarity/availability and
oriented polygonal Gauss entanglement. Future contacts and curves are unavailable
until their material exists. Open-curve Gauss integrals are geometric quantities,
not integer linking numbers or knot labels. Quadrature refinement difference is
convergence evidence, not a certified error bound; intersecting/degenerate curves
fail. Differentiability is limited to the fixed nonsingular geometry branch.

## Qualification and rights checklist

Before a scientific claim, retain the exact construct, conditions, chemistry,
parameter bytes, source mapping, raw-versus-realized lineage, units, masks,
algorithm/approximation policy, precision, solver evidence, and requested-use
rights. Unknown reference uncertainty remains `None`; it is not zero and cannot
support quantitative reference comparison.

The repository's admitted 1L2Y/Amber14 workflow exercises a real all-atom
force-field handoff and short NVE. The P2/P3 analytical controls, P4 rotamer
tables, four-atom generative fixture, and X1/X2 reference-conditioned mechanics
exercise their respective numerical contracts. Neither those controls nor the
38 correlated 1L2Y NMR models establish experimental folding, biological time,
independent predictor accuracy, or commercial authorization. Those claims need
rights-cleared experimental/parameter/training evidence with uncertainty and
qualification covering the actual model and conditions.
