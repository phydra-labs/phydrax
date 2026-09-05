# Nucleic-acid biophysics

`phydrax.applications.nucleic_acid_biophysics` composes explicit DNA/RNA identity,
structural observations, chemical-mapping inference, rigid-nucleotide mechanics,
secondary-state kinetics, electronic dynamics, and fixed-chemistry coordinate proposals.
These are different scientific models, not interchangeable routes to a folding prediction.

See the [public API](api/applications/nucleic_acid_biophysics.md), the
[runnable cookbook](cookbook/nucleic_acid_biophysics.md), and
[source and rights dispositions](biophysical_sources.md).

## Choose the output before the model

| Surface | Input and output meaning | Time and qualification |
|---|---|---|
| `structure` (N1) | Atom coordinates → frames, eRMSD, contacts, torsions, sugar pucker | Structural observations; a sampled coordinate axis is not automatically time |
| `observations` (N2) | Processed reactivities and hypothesis features → fitted observation law and held-out scores; distance intervals → local coordinate reconstruction | Neither chemical reaction kinetics nor an all-atom folding engine |
| `coarse` (N3) | Caller-parameterized rigid nucleotides → energy, force/torque, conservative or thermostatted trajectories | Declared mechanical units; no bundled experimental coarse-grained clock |
| `secondary_kinetics` (N4) | Exhaustive restricted secondary states and admitted energies/rates → CTMC paths, populations, first passage | Rate-law time unit; experimental timing requires independent kinetic calibration |
| `electronics` (N5) | Explicit orbital graph, energies and environmental channels → density matrices and quantum trajectories | Electronic time, not mechanical motion, atom charges, or radiation lesions |
| `generation` | Rights-cleared fixed-support conformers → trained conditional coordinate proposals | Generative pseudotime 0–1, not molecular time or Boltzmann sampling |

No calibrated nucleotide coefficient tables, experimental CG clock, or independently
pretrained coordinate model are bundled. Implemented equations and synthetic numerical
checks do not satisfy those gates. A reference's unknown uncertainty is `None`, not zero;
it cannot justify a quantitative calibration comparison. Preserve source rights separately
from the numerical success of a computation.

## N1: identity, source maps and structural observations

### Directed chemistry is the authority

`NucleicAcidConstruct(strand_ids, sequences, polymer_types, circular)` takes immutable,
aligned tuples. Strands have declared 5′→3′ order. `NucleotideKey(strand_id, position)` uses
zero-based positions within a strand; it is neither an atom ID nor a source PDB residue
number. Only unmodified uppercase DNA `ACGT` and RNA `ACGU` are admitted. DNA/RNA choice
is explicit: no U/T substitution or inferred modified chemistry. Linear ends are declared
uncapped termini, not a protonation or charge assignment. Circular strands explicitly
close the last-to-first directed edge.

`NucleotideAtomMapping` binds stable nonnegative int64 atom IDs to nucleotide keys and
canonical atom names. `prepare_nucleotide_binding` resolves an existing atom support;
it does not relabel it. Coordinate availability is distinct from inactive capacity:
a chemically present mapped atom cannot disappear as padding. Missing coordinates remain
masked, not completed. Duplicate alternate atom locations require explicit selection.

`nucleic_hypothesis_from_pdb_records` retains both raw and explicitly selected PDB rows.
The caller supplies a total assignment from selected record IDs to atom/nucleotide
identities, selects one model, and resolves alternate locations. Canonical residue
chemistry must agree with the construct. `NucleicStructureHypothesis` retains source
artifacts, length units, coverage, image policy, and inherited rights.
`normalize_nucleic_hypothesis` performs only requested unit conversion and stable-ID
reordering, with separate raw/normalized hypotheses and an adapter report. It neither
imputes atoms nor discards unresolved source information.

### Full interaction graphs are not dot-bracket strings

`BaseInteractionGraph` retains annotated `BaseInteraction` records, including noncanonical,
stacking, crossing, and multiple-interaction records with their sources. The graph does
not establish that an annotation is experimentally correct.

`to_dot_bracket()` is deliberately narrower: only linear connectivity, canonical
`kind="pair"` / `annotation="canonical"` records, one partner per nucleotide, and
noncrossing pairs are exportable. Circularity, pseudoknots, geometric contacts, and richer
annotations raise instead of being flattened. The secondary CTMC's one-partner planar
state space does not replace this full structural graph.

### Frames, exact eRMSD and a distinct smooth approximation

`base_frames` uses the centroid of C2/C6/C4 for purines and C2/C4/C6 for pyrimidines;
the x axis points toward C2 and the right-handed axes retain base-dependent orientation.
Missing or degenerate frames keep their construct position with `valid=False`.
Coordinates must be explicitly `"nonperiodic"` or already `"unwrapped"`. Independently
minimum-imaging atoms cannot define a base frame.

`NucleotideGDescriptor(binding, length_unit=..., image_policy=..., pairs=None,
cutoff=2.4, smooth_width=0.0)` implements the published directed-pair G descriptor.
It converts the declared coordinate length to Å and scales local displacements by
(5 Å, 5 Å, 3 Å). The cutoff is dimensionless. `compare(positions, reference)` divides
the sum of squared G differences by the **nucleotide count**, not the pair count;
`value` is the square root and `squared_distance` is the squared result.

- `pairs=None` selects every off-diagonal directed nucleotide pair. Explicit pairs use
  `NucleotideKey` identities, not atom IDs.
- Sparse support changes descriptor identity. It equals full eRMSD only if every omitted
  contribution vanishes in **both** conformations, which the caller must establish.
- `smooth_width=0` is the published continuous but not globally differentiable cutoff.
  Positive width selects a separately identified C2-tapered G, not exact published eRMSD.
- Inspect `pair_valid`, `coverage_fraction`, `successful`, and `cutoff_margin`. A masked
  zero contribution is not evidence of structural similarity. Differentiation is local
  to valid frame/cutoff branches; the distance norm itself is not smooth at coincidence.

`ERMSDCollectiveVariableProgram` exposes this comparison through the native atomistic
CV ABI and requires complete selected-pair coverage in its reference. It refuses an
implicit cell policy at evaluation. `NucleotideGDescriptor.observe_series` preserves
`SampledSeries` support/reset semantics and per-coordinate availability; it does not
turn trajectory indices or generative pseudotime into physical time.

`geometric_contacts` consumes explicit `GeometricContactCriteria` in Å and returns
coplanar/stacked masks and geometry. `contact_interaction_graph` labels these as
`geometric-contact`, not as inferred hydrogen bonds or canonical secondary pairs.
`NucleotideStructureQualifier` checks declared ring planarity and directed O3′→P
continuity bounds in native coordinate length units. Missing rings/backbone atoms fail
those checks; passing them is not complete physical qualification.

### Torsions and sugar pucker

`NucleotideTorsionProgram(mapping, system, coordinate_mask=..., image_policy=...)`
uses the native atomistic torsion CVs. Its fixed `(nucleotide, 12)` output is
α, β, γ, δ, ε, ζ, χ, ν₀, …, ν₄ in radians. Directed construct edges determine neighbor
atoms; missing sites and undefined terminal torsions stay masked. Values and
`branch_margin` must be interpreted together, particularly near angular branch cuts.
Periodic inputs must already be unwrapped.

`sugar_pseudorotation` fits a five-ring Fourier mode and reports phase, amplitude,
harmonic residual, and validity. This is a least-squares harmonic descriptor, not an
exact noisy-ring Altona–Sundaralingam fit or an empirical conformer assignment. A planar
sugar has undefined phase and is masked; nonideal-ring residuals are retained.
`observe_series` and `observe_pseudorotation_series` preserve sampled support and masks.

## N2: processed chemical mapping and reconstruction

### Retain the assay, not just a pairing label

`ChemicalMappingObservation` represents one construct/condition/replicate, including
stable nucleotide mapping, reagent, preprocessing, values, measured standard deviations,
source rights, and `observed` mask. Negative corrected reactivities are legitimate.
Missing observations supply no likelihood equation, zero accessibility, or pairing label.
`ChemicalMappingCondition.exposure` is positive in its declared time unit or `None` when
unreported; exposure does not convert a fitted reactivity law into kinetics.

A supplied `covariance_lower` is the covariance Cholesky factor of **observed rows in
their original relative order**, not a precision factor or a sliced Cholesky factor of
a larger system. Its diagonal covariance must agree with the measured SDs. Without it,
the law is explicitly a diagonal-SD approximation. Separate observation records retain
replicate identity; this model does not infer cross-replicate covariance.

`import_processed_rdat` admits checksum-matched raw bytes of its explicit RDAT 0.34
profile. It retains raw bytes/records, source position labels, per-row mutant sequences,
annotations and declared structures. It requires explicit
`error_semantics="standard-deviation"`; covariance and replicate grouping cannot be
inferred from absent metadata. Rows are unpooled unless replicate IDs are supplied.
Unsupported fields and ambiguous mappings refuse instead of being ignored.

The repository retains an actual RMDB observation artifact at
`tests/fixtures/nucleic_acid_biophysics/TODEX_DMS_0000.rdat` with its sibling
`.source.json` provenance record. Its CC0 declaration and exact checksum are recorded;
its `REACTIVITY_ERROR` values are marginal supplied errors, not a measured covariance.
Depositor-designed structures are hypotheses, not independently solved structures.

`AccessibilityReactivityModel` fits an affine processed-reactivity law: explicit baseline
groups, one shared accessibility slope, and named numerical condition effects. The
caller supplies accessibility features in [0, 1]; the model never derives pairing from
reactivity. One fit admits one reagent/preprocessing profile. `fit()` enforces training
rights and returns optimizer status, whitened-design rank, singular values and
`identifiable`, alongside predictions and scores. Optimization success is not
identifiability. Hold out whole independent constructs/acquisitions, keep the same
parameter-name/order contract, and evaluate their `score`/`residual` without refitting.
The retained-RMDB workflow converged with rank two, but its whole-construct held-out
chi-square per observation was about 230.30 under the supplied SDs. This simple model
is **not experimentally adequate under that noise assumption**. The cookbook retains
the poor predictive result; numerical completion is not scientific qualification.

### Distance reconstruction remains a separate inference

`IntervalDistanceReconstruction` consumes an existing nonperiodic atomistic support,
stable atom-ID pairs, distance intervals, measured SDs, weights, length units, and source
rights. It optimizes native distance CV residuals; it does not create atoms, chemistry,
force fields, or restraints from chemical reactivity. Such a mapping would need its own
admitted observation model and uncertainty.

Distance intervals cannot distinguish reflections. Explicit four-atom chirality restraints
supply signs, minimum signed tetrahedral volumes and uncertainties in the corresponding
length-cubed scale. `reconstruct(initial_positions, fixed_mask=...)` preserves fixed and
nonmobile coordinates. Check optimizer status, `interval_satisfied`,
`restraints_satisfied`, and `chirality_qualified` separately. Empty chirality support is
allowed only as an explicitly unqualified distance reconstruction. Local restraint
satisfaction is not all-atom physical validity or a unique structure.

## N3: caller-parameterized rigid nucleotides

`NucleotideParameterArtifact` checks exact immutable JSON bytes against a
`ReferenceArtifactManifest` and admits the requested commercial, redistribution,
training, and export uses. It does not retrieve a parameter table. Supported families are:

| Family | Equation/geometry profile |
|---|---|
| `average-dna` | DNA1, collinear average-DNA sites |
| `groove-salt-dna` | DNA2 unequal-groove geometry and screening, average strengths |
| `sequence-dna` | DNA2 with explicitly supplied sequence-dependent strengths |
| `rna` | RNA-specific angular terms and distinct directed 3′/5′ stacking sites |
| `dna-rna-hybrid` | Separate DNA2, RNA and independently supplied cross-acid profiles |

The payload explicitly supplies `family`, `source_model`, `temperature`,
`salt_concentration`, `salt_unit`, `geometry`, `profiles`, and `sequence_strengths`.
Salt uses `"mole/litre"`; screened profiles require positive salt. Temperature and all
mechanical coefficients use the supplied `AtomisticUnitSystem`. Geometry, the stacking
temperature coefficient, screening scale and terminal charge factors are data, not
family-name defaults. Cross-acid coefficients are mandatory and never averaged. Sequence
matrices use A/G/C/T order (U occupies the fourth RNA slot without changing chemistry).
Average-family admission refuses hidden sequence-dependent well strengths.

The model includes the selected profile's backbone FENE, excluded-volume, stacking,
hydrogen-bond, cross-stacking, coaxial and screening terms. RNA can include GU wobble;
DNA and cross-acid hydrogen-bond support follows the implemented canonical-pair rules.
`nucleotide_reference_sites` creates five physical sites plus three zero-physical-mass/
charge differential frame markers. The latter carry orientation derivatives, not extra
physical interaction sites.

`NucleotideModelPlan` takes stable body IDs in construct order, eight stable site IDs
per nucleotide, exact reference geometry, caller masses and COM inertia, optional fixed
bodies and a cell. `prepare()` binds the generic rigid-body and marker machinery.
`evaluate()` returns energy, site forces, and full body force/torque loads. The load
result preserves `load`, `mobile_load`, and `reaction_load`: reactions are loads **on**
fixed bodies, so fixture forces have the opposite sign. Fixed material bodies are not
inactive padding.

`step()` performs conservative kick–drift–kick; `heat_bath(translation_friction,
rotation_friction)` provides native finite-time rigid OU thermalization and split
thermostatted stepping. Mechanical load and kinetic-energy conversion use the declared
unit system; do not insert an additional conversion. Frictions are inverse native time.
A short heat-bath run does not establish configurational equilibrium or an experimental
clock.

Periodic qualification is restricted to fully periodic orthorhombic cells. COM states
remain unwrapped; bonded interactions stay unwrapped, while each nonbonded pair uses
one COM image for all its sites. Interaction reach plus site offsets must fit inside
the unique-image radius or preparation refuses. Forces/torques are differentiable on
fixed topology/image branches; no global smoothness or collinear-frame Hessian claim
is made. Numerical Hamiltonian checks still need independently cleared coefficients,
reference-engine comparisons, duplex/mechanical data, and clock calibration before
experimental claims.

## N4: bounded secondary-state kinetics

`SecondaryStructureState` is a one-partner, pseudoknot-free, ordered-planar pairing
state. `StrandComplexPartition` follows connected components of intermolecular pairs.
Identical sequences with different strand IDs remain distinct physical copies; no
unpaired encounter complexes or indistinguishability correction is invented.
The compiler supports only linear DNA, RNA or explicitly parameterized DNA–RNA mixtures.

`SecondaryEnergyModel.from_bytes` admits exact JSON and all four requested-use flags.
`pair_loop` includes pair, hairpin, bulge, internal/multibranch and association-initiation
terms; `nearest_neighbor_loop` additionally includes antiparallel stacks. Exterior and
nick-containing loops have zero penalty by this profile, with no hidden dangling-end,
coaxial-stack or mismatch corrections. Required missing entries refuse; no loop-table
extrapolation occurs. Energies are explicitly dimensionless molar G/(RT) at one source
temperature, and execution temperature in K must match it. Watson–Crick pairing or the
named RNA GU-wobble extension is explicit.

`AssociationConvention` distinguishes `standard_state` from `fixed_volume`. Each
independent association in fixed volume adds log(c° Nₐ V) to G/(RT), using explicit
molar concentration and volume conversion. Standard-state mode sets this factor to one
and makes no finite-concentration timing claim. `SecondaryRateLaw` names `metropolis`,
`symmetric_barrier`, or `association_metropolis` and supplies independent kinetic
prefactors in inverse `time_unit`.

The first two laws use symmetric attempt frequencies. `association_metropolis` applies
the explicit inverse standard-volume factor to joins and keeps dissociation independent
of volume. Only its fixed-volume elementary joins support
`elementary_association_rate_constant`, in m³/(mol·time_unit). A multistep first-passage
macrostate does not automatically define an elementary second-order rate constant.

`prepare_secondary_kinetics` exhaustively enumerates legal states and reversible pair
toggles. State/channel capacity excess is a preparation error, not a reflecting
truncation. The runtime state is a one-element **state-index** array; use `encode` and
`decode`, not nucleotide IDs. Numerical lookup is JIT-compatible, but enumeration,
chemical decisions and discrete event choices are not pathwise differentiable.

Use `prepared.process` with native `phydrax.solver.solve_direct_ssa`, and compiled targets
from `target`, `pair_count_target`, `exact_state_target`, or `joined_target` with
`phydrax.solver.event_first_hit`. First hits scan the initial and event post-states, not
a save grid. An initial hit occurs at t₀; an unobserved hit has infinite `time`.
Successful non-hits are right-censored at the requested horizon. Failed non-hits are
`incomplete`, with `observation_end` at the last verified event; capacity exhaustion
is separately recorded. A hit before later exhaustion remains exact. Never relabel
incomplete paths as censored or drop them to claim an unbiased population estimate.

`prepared.generator()` exposes a finite generator and escaped-rate evidence.
`phydrax.solver.finite_generator_hitting` requires closed support: it reports hitting
probabilities, unconditional MFPTs, reachability, almost-sure hitting and closed avoiding
classes. Unreachable targets have probability zero; any positive probability of avoiding
the target forever gives infinite unconditional MFPT. Numerical failure gives NaN,
not a regularized finite answer. Inspect residuals and `successful`.

## N5: electronic states, not nucleotide mechanics

`ElectronicSiteGraph` binds stable **orbital/site IDs** to nucleotide keys and orbital
labels. These are not atom IDs. Edges specify allowed coupling support; the compiler
does not infer energies from a sequence or coordinates. `ElectronicParameterArtifact`
retains source bytes, checksum, single-system energy units, model scope, orbital gauge,
site energies, Hermitian couplings and explicit environmental channels. Each Hermitian
edge is supplied once as H[row, column]; the conjugate is assembled automatically.
Structure-derived parameters require corresponding structure artifacts and rights.

`prepare_electronics` converts energy to the selected native energy unit and divides
by ℏ **once**. The prepared `hamiltonian` is already H/ℏ in inverse native time, with
phase convention exp(−iHt/ℏ), not cyclic frequency. Channels use dimensionless `jumps`
and separate inverse-time `rates`. `collapse_operators` multiplies by √rate once for
dense/jump solvers; do not multiply by the rate again or insert a 2π factor.

Channels are explicit `dephasing`, population-transfer `bath`, or `recombination`.
For projector dephasing, coherence ij decays at (rateᵢ + rateⱼ)/2. A declared bath is
not an inferred thermal-equilibrium law. Recombination requires explicit vacuum support
and denotes loss from this electronic sector, not a lesion probability.
`prepare_electron_hole` builds electron-major/hole-minor tensor support with an explicit
interaction artifact, even for a zero interaction. It does not infer partial-loss
sectors by lifting single-carrier vacuum sinks.

`evolve_electronics` uses `method="lindblad"` or `"cptp"`; `evolve_electronic_jumps`
uses native **fixed-step** quantum trajectories, not event-exact SSA. The latter checks
an all-state step/rate bound and finite history capacity, but still requires time-step
and ensemble convergence evidence. Neither sampled jump choices nor host preparation
are pathwise differentiable; prepared operators and density observables retain their
numeric JAX paths.

Inspect solver validity and the retained density history. `electronic_populations`,
`electronic_coherences`, and `nucleotide_electronic_populations` preserve declared
site/nucleotide order. `electronic_reduced_density` excludes vacuum without
renormalizing away loss: its trace is surviving carrier weight. Coherences depend on
the declared orbital gauge. These outputs do not assign atom charges, modify forces,
predict radiation damage, or demonstrate quantum-computing advantage. Resource limits
bound dense dimension, Liouville elements, channels and trajectory history; excess
refuses rather than dropping material basis states.

## Generation: admitted coordinate hypotheses on fixed chemical support

Import shared coordinate types through
`phydrax.applications.nucleic_acid_biophysics.generation`; the protein generation leaf
exposes the same shared types. Biological wrappers, not a universal biomolecule owner,
bind each construct's identity.

`CoordinateProviderProvenance` admits output/data rights separately from learned weight
and prepared-input rights. Learned providers require weight, input and output lineage;
MSA/template inputs retain their own identities and rights. Any egress destination must
be explicitly authorized. `import_nucleic_hypotheses` only admits offline-supplied
outputs; it neither calls a provider nor downloads checkpoints. It retains every
conformer, its matching source artifact, and named provider-specific confidence.
Those confidence values are not standardized probabilities.

`prepare_nucleic_coordinate_support` requires a single nonperiodic `AtomisticBatch`
template, exact material atom coverage, DNA/RNA-specific tokens, three gauge atom IDs,
and a `CoordinateGeometryPolicy`. Stable atom order, chemistry, units and condition
names are model ABI. `map_nucleic_hypothesis` performs stable-ID reordering and exact
unit conversion only; incomplete coverage, a different construct or token mapping,
and periodic input refuse. It is not missing-atom completion or permutation equivariance.

`prepare_coordinate_training_data` requires actual conformers, rights, record/source
IDs, conditions and disjoint caller-defined training/validation groups. It checks group
separation and identical canonical-conformer leakage, not biological independence.
`fit_coordinate_model` actually optimizes conditional flow matching. Its validation
loss is not experimental folding accuracy; no pretrained capability is implied.
The gauge uses mass centering and a proper anchor frame, never reflection.

`prepare_coordinate_sampler` provides the numeric JIT/gradient boundary;
`sample_coordinate_proposals` retains raw ODE states, canonical views, solver status,
and one geometry qualification per sample. It performs no hidden rejection sampling
or repairs. Geometry screening checks declared bond bounds and signed chiral volumes
(or an explicitly achiral profile), not full force-field validity. Confidence remains
uncalibrated, and the gauge-fixed singular support exposes no coordinate likelihood,
Boltzmann weights, or equilibrium free-energy claim. Sampling integration from 0 to 1
is generative pseudotime.

`save_coordinate_model` / `load_coordinate_model` use native pickle-free ML artifacts,
match the support/feature ABI, and re-admit inherited and checkpoint rights. Exporting
or training on a proposal does not erase provider restrictions. A physically meaningful
handoff still requires independently parameterized mechanical support and its own
qualification; a generated coordinate batch is not automatically a simulation state.
