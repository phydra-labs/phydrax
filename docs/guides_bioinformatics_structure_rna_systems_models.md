# Structure, RNA, systems biology, and biological models

This guide joins four related but scientifically separate layers: macromolecular
identity/topology, declared RNA folding grammars, biochemical network analysis, and
native learned model artifacts.

## Macromolecular records and numeric structure

`MacromolecularRecord` is a host-resident, identity-preserving record. Entities,
label/auth chains, residues, atom names, alternate locations, coordinate models,
occupancies, B-factors, chemical components, bonds, missing atoms/residues, and assembly
operations remain distinct. `parse_mmcif` and `load_mmcif` produce this record without
placing strings in a PyTree.

`StructureLoweringPlan` declares atom, residue, chain, model, bond, assembly, and
missingness capacities, chemistry policy, and coordinate dtype. `for_record` computes
capacities from one concrete host record. `lower_macromolecular_record` resolves all
identity and chemistry before producing:

- `MacromolecularStructure`, the fixed numeric macromolecular topology and model-indexed
  coordinates;
- an atomistic `AtomicStructure` in the declared scale contract; and
- a `MolecularTopologyPlan` for resolved bonds.

Lowering is all-or-nothing. Unresolved atomic number, component atom, bond reference, or
capacity invalidates the result; chemistry is never guessed. The built-in atomic
masses are conventional element masses, not a force field. Lowering does not assign
charges, protonation, missing coordinates, bond parameters, periodic boxes, or an
energy model.

Ensemble analysis is occupancy-aware and assumes fixed topology. Rigid alignment uses
weighted least squares. Residue contacts compute exact minimum resolved-atom distance
under a declared cutoff. `assign_geometric_secondary_structure` is a coarse geometric
assignment and explicitly does not claim DSSP hydrogen-bond thermodynamics. Chain
interfaces are finite contact/centroid geometry, not binding affinity.

## RNA folding contracts

`RNAEnergyModel` is an explicit additive grammar: pair energies, allowed pairs,
unpaired energies, temperature, gas constant, hairpin constraint, unit, and alphabet
identity. `nussinov_energy_model` supplies only a unit-declared A/C/G/U pairing score.
It is not a bundled empirical nearest-neighbor free-energy parameter set.

For that declared grammar:

- `minimum_free_energy` is exact dynamic programming over noncrossing partial matchings;
- `partition_function` is exact log-space inside/outside DP and reports pair marginals;
- `rna_log_partition` exposes the scalar whose energy gradients equal negative expected
  counts divided by thermal energy; and
- `RNAConstraints` applies hard pairing constraints and additive position energies.

“Exact” means exact for the finite pseudoknot-free additive grammar, not exact RNA
thermodynamics or tertiary structure. `restricted_pseudoknot_fold` is a deterministic
capacity-bounded greedy heuristic and makes no global optimum claim. `RNATertiaryRestraints`
and `lower_tertiary_restraints` map explicit residue anchors into atomistic distance
constraints; they do not predict a tertiary fold.

Sequence codes passed to RNA kernels must share the model's alphabet order/fingerprint.
Discrete pair choices and hard constraints are nondifferentiable even when MFE or log-Z
is differentiated with respect to energy parameters.

## Stoichiometric networks

A `StoichiometricNetwork` compiles typed `Compartment`, `Species`, and `Reaction`
objects. Units use `UnitDimension`; exact elemental composition and charge are optional
but, when present, are audited by `audit_stoichiometry`. Boundary species, reaction
bounds, objective sense, exchange status, gene-reaction rules, and objective
coefficients remain explicit.

`conservation_analysis` computes a numerical left-nullspace for internal-species pools
with a stated singular-value tolerance. This is a local numerical rank result, not a
symbolic proof for arbitrary parameters.

`flux_balance_analysis` solves the declared linear steady-state model through the
native convex lifecycle and can audit the complete optimal face for alternate optima.
`flux_variability_analysis` solves complete per-reaction extrema under a retained
objective fraction. Both are exact-model claims executed by numerical optimization:
KKT/solver evidence, feasibility, unboundedness, tolerances, active gene rules, and
auxiliary-solve capacity must be inspected. FBA predicts feasible flux under its
constraints; it does not establish kinetic realizability, regulation, growth, or
causality.

FVA and alternate-optimum auditing preflight the complete solve family. Exceeding
`max_auxiliary_solves` raises `FluxCapacityError`; there is no partial “first reactions”
result. Gene-reaction rules are finite DNF logic over the caller's active-gene set and
do not infer expression thresholds.

## Kinetics, regulation, and identifiability

`KineticReactionSystem` provides closed JAX-native rate-law kinds over a compiled
network. `simulate_kinetics` uses the native dynamics lifecycle and reports positivity,
conservation, and integration evidence. `KineticTrajectoryObjective` is a weighted
least-squares adapter over saved observations. Unsupported free-form expression
runtimes are not evaluated.

`DiscreteRegulatoryNetwork` is an exact synchronous finite-state truth-table system.
One-step transitions and two-slice factor-graph lowering retain capacity and cycle
semantics. They are not continuous-time regulatory kinetics.

`local_identifiability` differentiates an observation map and diagnoses local
sensitivity rank at the supplied parameters. It cannot be promoted to global
identifiability. `global_candidate_identifiability` is conclusive only for an exhaustive
finite candidate set; otherwise witnesses may disprove uniqueness, but no-witness is
inconclusive. Pair materialization is capacity guarded.

## Native biological models

`models` contains native, alphabet-bound sequence embeddings and recurrent/attention
encoders, token/label/pair/contact heads and objectives, an equivariant macromolecular
encoder, and finite-capacity sequence design.

Masks and identity are part of every model boundary. `TokenPrediction`,
`TokenLabelPrediction`, and `PairPrediction` retain alphabet/tokenizer/label/pair-space
identity. Objectives reduce only explicitly selected valid entries. A differentiable
loss does not make labels, token selection, pair masks, or hard sequence decisions
smooth.

`SequenceDesignProblem` separates a categorical relaxation, exact constraint repair,
full discrete scoring, candidate diversity, and final ranking. `solve_sequence_design`
therefore reports hard feasibility and relaxed evidence separately. It does not prove a
global combinatorial optimum or biological activity.

## Learned artifact and leakage policy

A native learned callable is usable as a foundation model only after exact binding:

- `TokenizerProvenance` binds tokenizer bytes, normalization, vocabulary, and alphabet;
- `LicenseProvenance` records the reviewed license identity and separate inference,
  adaptation, and redistribution permissions;
- `PretrainingOverlapProvenance` records `unknown`, `no-detected-overlap`, or
  `known-overlap` for one evaluation split and homology partition;
- `FoundationModelManifest` binds artifact bytes, numeric parameter content, model
  structure, tokenizer, license, overlap, and optional base model; and
- `bind_native_foundation_model` rechecks every identity before returning a
  `BoundNativeFoundationModel`.

`ExternalFoundationRuntime` is host-only and is never presented as a JAX/Equinox
PyTree. Low-rank adapters require `LowRankAdapterProvenance` and exact binding to the
base artifact and parameter hash. The package does not download weights, execute remote
code, convert arbitrary external frameworks, or infer license permissions.

Split biological families, subjects, specimens, and homologous sequences before fitting
or choosing a model. Record homology-aware partitions, not only random row indices.
Unknown pretraining overlap remains an explicit warning and cannot be rewritten as “no
overlap.” Embeddings and predictions are learned outputs, not mechanistic explanations
or validated clinical/functional claims.

## Unsupported boundaries

The native structure layer is not a PDB repair, protonation, force-field assignment,
docking, or structure-prediction suite. The RNA layer does not ship empirical
thermodynamic parameter tables or an exact unrestricted pseudoknot solver. The systems
layer does not execute arbitrary SBML MathML, events, rules, algebraic constraints, or
external optimizers as silent fallbacks. The model layer does not provide pretrained
weights or certify a model's intended use. Use qualification artifacts and the exact
method/artifact contracts to state the narrower claim that was actually evaluated.
