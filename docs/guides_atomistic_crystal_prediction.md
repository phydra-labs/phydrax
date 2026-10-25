# Native molecular crystal prediction

`phydrax.atomistic.crystal` composes exact space-group data, molecular input,
asymmetric-unit packing, native relaxation, radial screening, certified
deduplication, and same-definition ranking. Each stage owns one claim, and none
of them is a substitute for another:

- packing is a geometric search, not physical ranking;
- a symmetry-constrained (projected) minimum is not a full minimum;
- radial similarity is a screen, not certified equivalence;
- static-lattice energy ordering is not finite-temperature polymorph stability
  and not a likelihood (`RANKING_NONCLAIM`).

Generated proposals flow through the ordinal-addressed
[generation stream](guides_atomistic_generation.md); this guide covers the
crystal-specific owners. The [crystal API page](api/atomistic_crystal.md) lists
every public name.

## Space groups, settings, and sites

The packaged exact operation tables cover all 530 Hall settings with standard
choices for the 230 International numbers. Each row stores the integer rotation
and a translation in twelfths. Their provenance (spglib v2.7.0 as the
operation/Hall oracle and the pinned NVIDIA `nvalchemi-toolkit` standard-setting
map, with licenses and file digests) is recorded in `SPACE_GROUP_SOURCES` and
bound by `SPACE_GROUP_DATA_ID`. `tools/generate_space_group_tables.py`
regenerates or `--check`s the table from those admitted sources. These tables
are operation and setting data only: no Wyckoff-letter table, radius table,
atomic-volume table, or crystal-structure frequency prior is packaged.

- `SpaceGroupSetting(hall_number, transformation=None)` binds exact operations;
  `setting_id` covers the International number and the canonical operations, not
  their labels. `standard_space_group_setting(number)` uses origin choice 2 where
  applicable and hexagonal axes for rhombohedral groups.
  `SettingTransformation(matrix, origin_shift)` is exact rational `x = P x' + p`.
- `HallSymbol` parses Hall notation independently of the generated table.
- `SettingMetricChart(setting)` parameterizes the invariant metric of a setting
  (six coefficients for triclinic down to one for cubic) and realizes cells
  with evidence; `periodic_cell` refuses a failed realization.
  `niggli_reduce` and the bounded sublattice enumeration are exact host owners.
- `SitePosition`, `site_of_point`, and `enumerate_site_orbits(setting, *,
  maximum_sites, maximum_systems)` derive site orbits from exact operations. A
  table with status `"site-limit"` or `"system-limit"` is incomplete: its found
  sites are correct, but it cannot refute membership.

## Molecular input

`MolecularComponent(atomic_numbers, bonds, *, bond_orders, masses,
spin_multiplicity, ...)` is one connected, explicitly indexed molecular graph
with explicit hydrogens, formal charges, and declared tetrahedral and double-bond
stereo. Undeclared stereocenters are refused by default; `valence="unchecked"`
records the absence of valence qualification. `MolecularConformer` and
`ConformerPool` hold mass-centered principal-axis conformers that realize the
declared stereo; `native_conformer_pools(plans, key, count=...)` generates pools
with native distance geometry (`ConformerGenerationPlan`) and refuses a
component without accepted conformers rather than dropping it.

Element data are admitted inputs with provenance. `ElementTable(quantity,
values, provenance=...)` carries masses (dalton), radii (angstrom), or atomic
volumes (cubic angstrom); a missing element is refused and there is no fallback
table. `ContactPolicy(radii, rule="radius-sum" | "donor-acceptor-reduced")` and
`VolumePolicy(formula_unit_volume=... | atomic_volumes=...)` (exactly one) form
a `MolecularCrystalInput(pools, contact=..., volume=..., formula_counts=None)`;
`contact` may instead be a caller-declared contact-distance matrix.
Atom types for symmetry-equivalent atoms are assigned by exact whole-component
correspondences (`atom_type_assignment`); an undetermined correspondence raises
instead of treating a partial search as proof.

Admitted inputs are kept as host restoration witnesses, separate from the
scientific identities that bind the laws they define:

- `MolecularComponent.mass_table` retains the full admitted mass `ElementTable`
  (`None` for per-atom masses), and `mass_provenance` is that table's identity;
  `enantiomer()` keeps the same mass source;
- `MolecularConformer.admitted_positions` is a read-only host float64 copy of the
  constructor coordinates; `conformer_id` binds the local geometry, source, and
  provenance;
- `ConformerPool.declared_weights` are the admitted weights; `pool_id` binds their
  normalization;
- `MolecularCrystalInput.contact_policy` (`None` for a declared matrix) and
  `volume_policy` retain the admitted policies whose identities
  `contact_provenance` and `volume_provenance` bind;
- `CrystalSpaceGroupPolicy.declared_weights` retains the declared prior weights.

Rescaling declared weights changes no scientific identity; only a persisted
recipe's content address changes (see [Recipe persistence](#recipe-persistence)).

## Packing

```text
groups = phx.atomistic.crystal.CrystalSpaceGroupPolicy(
    [phx.atomistic.crystal.standard_space_group_setting(number) for number in (1, 4)],
    weights=(1.0, 1.0),
    provenance="uniform demonstration prior (not a CSD frequency table)",
).filter(sohncke_only=True)
plan = phx.atomistic.crystal.CrystalPackingPlan(
    molecular,
    groups,
    z_prime=1,
    chirality="enantiopure",
    resources=phx.atomistic.crystal.CrystalContactResources(
        phx.atomistic.AtomisticDenseImageSearch(maximum_dense_routes=lanes * 1_000_000),
        search_capacities=(capacity,),
        role_capacities=(phx.atomistic.AtomisticRoleCapacity(maximum_routes=routes, maximum_degree=degree),),
        skin=0.5,
    ),
    cell=phx.atomistic.crystal.CrystalCellPolicy(volume_range=(1.0, 1.3), minimum_height=3.2),
    relief=phx.atomistic.crystal.CrystalReliefPolicy(maximum_steps=400, overlap_tolerance=0.1),
)
packer = plan.prepare(lanes=lanes)
```

- `CrystalSpaceGroupPolicy(settings, weights=None, provenance="fixed")` routes
  attempt ordinals by declared weights. Several settings need explicit positive
  weights and a provenance other than `"fixed"`; the weights are caller data,
  not a packaged prior. `filter(z=, z_prime=, crystal_systems=, sohncke_only=)`
  derives compatibility from the operations (`Z = |G| Z'`, handedness from
  operation determinants).
- `CrystalPackingPlan` places a positive integer `z_prime` of molecules at
  general positions in each admitted setting and expands the asymmetric unit
  with explicit chirality. Special-site and fractional component `Z'` exist in
  the lower-level `AsymmetricUnitPlan`, not as packing-plan options. Every
  setting's expanded atom count must fit `maximum_atoms`.
- Each setting gets one stateful `PreparedCrystalGroupPacker` lane sampler. Its
  overlap penalty sums `1/2 max(0, r_ij - |d_ij|)^2` for contact distance `r_ij`
  and pair-image separation `d_ij` over every unordered pair-image once through the native
  neighborhood bundle (`IMAGE_PAIR_SUM`); only same-copy zero-shift pairs are
  excluded, so periodic molecular self-images count. Rigid steps are
  mass/inertia preconditioned with a displacement cap and a native quaternion
  retraction. The cell step is the tangent (deviatoric) projection of the chart
  gradient under a metric-strain cap; because that projection preserves volume
  only to first order, the finite step is retracted exactly onto its pre-step
  volume before a declared fraction of the excess above the reference volume is
  compressed, so the volume never grows and never drifts below the reference.
- The compiled sampler holds the setting's `AsymmetricUnitLayout`, built once
  from its `AsymmetricUnitPlan`: the metric chart, operation arrays, conformer
  table, copy/atom/site-action indices, and the element of every expanded atom,
  whose numerical leaves are int32/float64/bool JAX arrays, plus the plan's
  static `plan_id` string. It
  carries no sites, pools, components, setting, or restoration witness. The host
  `PreparedCrystalPacker` keeps the plans as `units` and uses them in `packed`
  and `record_builder`.
  `expand_asymmetric_unit` accepts the layout (the compiled form) or a plan (the
  cold host form, which builds a layout on every call).
- Acceptance requires the overlap tolerance and the two-sided volume tolerance
  `|V - V_ref| <= volume_tolerance * V_ref`. A cell outside the declared bounds,
  a refused cell update (height floor, axis ratio, or a metric that fails the
  positive-definite Cholesky certificate, counted in `cell_refusals` and never
  repaired), exhausted contact capacity, an exhausted step budget, and sampler or
  nonfinite failures are explicit statuses; capacity overflow is refused, never
  truncated. The penalty and its cell tensor are not a physical energy or
  stress.
- `CrystalReliefPolicy.policy_id`, and through it the packing plan, sampler,
  recipe, checkpoint, and store-lineage identities, binds this cell-step and
  acceptance definition. Continuations, recipes, and records produced by a relief
  whose identity lacks it (a first-order-only cell step with an upper-only volume
  check, which could accept under-volume cells) fail identity admission instead of
  remaining accepted; re-prepare the plan from the same constructor arguments
  (or write a new recipe from it) and pack again.
- `CrystalPackingPlan.law_id` is the scientific packing law: the molecular input
  (components, conformer measure, contact distances, volume), the space-group
  policy, `Z'`, chirality, the cell prior, and the relief numerics with the
  attempt definition. `plan_id` is the complete plan: that law plus the contact
  resources and `maximum_atoms`. Each member's `sampler_id` binds only its own
  law (molecular input, its `AsymmetricUnitPlan` with setting, `Z'`, and
  chirality, the cell prior, relief numerics, and the attempt definition), so
  proposal identities and ordinal RNG addresses do not depend on the lane count,
  the contact search, or its capacities; for the same root key and campaign,
  attempts at the same ordinal have the same proposal identity and status, and
  their coordinates agree to within `1e-9`. A capacity overflow is still
  reported as an explicit `CAPACITY_REFUSED` status for that attempt.
  `group_index` is the member's address in the prepared family (its route index
  and payload field), not part of its law. The lane count, contact-bundle plan,
  and contact resources are executable `signature_facts`; stage signatures and
  generation checkpoints bind them, so another layout of the same law is
  refused on restore rather than resumed. Packed records use `law_id` as their
  lineage. Checkpoints and recipes whose packer identities still include lanes or
  resources are refused, and records written under such identities do not match
  current proposal or lineage identities; re-prepare the plan and pack again.
- Vacant and terminal lanes are padding, not cells. A vacant lane carries the
  reference-cell metric coefficients (not a singular zero metric); only running
  lanes enter the differentiated contact energy and may rebuild contact
  resources, so inactive rows of a partially occupied block do not perturb them.
  A member with no running lane is not advanced at all. A native contact,
  certification, or finiteness failure sets the lane's failure status and
  prevents its update and retraction.

`packer.pack(key, *, campaign_id, target, attempt_budget, sink=None,
maximum_transitions=None, state=None)` runs the attempts through a local
`AtomisticGenerationStreamPlan` (`packer.stream(...)`). A zero target or budget
draws nothing, a finished budget reports its shortfall, and an exhausted
transition limit returns a running continuation rather than a truncated success.
With an `AtomisticStructureStoreGenerationSink`, every attempt, accepted or
rejected, is published idempotently to a structure store. The plan returned by
`packer.stream(campaign_id, ...)` is an ordinary generation stream, so it can also
be prepared with an `AtomisticGenerationPlacement`
([distributed quotas](guides_atomistic_generation.md#distributed-quotas)).

## Recipe persistence {#recipe-persistence}

`CrystalPackingRecipe(packer, *, campaign_id, pending_capacity=None,
steps_per_transition=None)` binds one `PreparedCrystalPacker` (its plan and lane
count) to the stream configuration it runs under. `recipe_id` binds the complete
plan identity `plan_id` (the scientific `law_id` with contact resources and atom
bound), the lane count, and the identities of the unfiltered prepared stream
(stream, samplers, numeric revisions, executables); `recipe.stream(filters=...)`
rebuilds that stream with explicit proposal filters, and generation checkpoints
authenticate against it.

`write_crystal_packing_recipe(path, recipe)` atomically writes one bounded,
pickle-free array archive holding every reconstructible immutable input: molecular
components (atoms, charges, bonds and orders, stereo, spin, valence, automorphism
bounds, and the admitted masses with their provenance), conformer pools (admitted
constructor coordinates as float64 arrays, thresholds, source, provenance, and
declared weights), formula counts, the contact policy with its admitted radii or
the declared contact matrix, the volume policy and its data, each setting's
reference Hall entry and exact transformation, declared group weights and prior
provenance, `Z'`, chirality, the atom bound, cell, relief, and contact-resource
policies, the lane count, and the stream configuration. The returned
`CrystalPackingRecipeReceipt` records `recipe_id`, `law_id`, `plan_id`,
`content_id` (the address of the complete stored recipe, including display names
and provenance text), and `array_digest`. The stored plan records both `law_id`
and `plan_id`, and the reader verifies them; recipes without them, or whose
stream identity lacks `executables`, are refused and must be written again from
a re-prepared plan.

`read_crystal_packing_recipe(path)` admits the archive under explicit limits,
verifies the content address, rebuilds every value through its owning
constructor, checks every recorded identity, and re-encodes the rebuilt recipe.
It refuses unless the re-encoding reproduces the stored recipe and arrays byte for
byte. Unknown, missing, or mistyped fields, unreferenced or altered arrays, and
altered inputs are refused; nothing is imported dynamically or unpickled.
Scientific identities bind the laws the witnesses define; exact constructor
replay, byte-exact re-encoding, and `content_id` bind the witnesses themselves.

Replay is exact only in the numerical profile that wrote the recipe (backend,
precision, and JAX configuration). Conformer frames are recomputed by the
conformer constructor, so a recipe written under another profile can be refused on
its conformer identities rather than restore different coordinates; no
GPU-to-CPU bitwise replay is claimed. The recipe holds no generation
continuation, which stays in the stream's own checkpoints. The rights and
provenance of caller-supplied tables, priors, and conformers travel inside the
recipe; the recipe does not admit them.

## Storage

`crystal_structure_schema()` records the cell, stage and status, International
and Hall numbers, `Z` and `Z'`, the padded metric coefficients, the attempt
ordinal, relief evidence, optional energy in eV, asymmetric-unit centers,
orientations, and conformer indices, and per-atom expansion provenance (source
site, local atom, component, copy, operation, whether the copy is the mirror
stereoisomer of its component (`source_enantiomer`), and lattice shift).
`crystal_structure_record(...)` builds packed or relaxed records; a packed record
names the packing law (`law_id`) as lineage, a relaxed record names its packed
record, and the source setting of an unconstrained relaxed record is provenance,
not retained symmetry.

`crystal_structure_from_expansion` and the relaxation handoff declare each
mirror copy of a stereo-chiral component (`AsymmetricUnitPlan.copy_enantiomer`)
as `component.enantiomer()`, so a racemate's opposite-handed molecules carry
opposite stereo declarations and stereo-aware equivalence compares them as
such. A `DedupBank` stores per-molecule component identities; reopening a bank
that holds racemates requires resupplying both the component and its
`enantiomer()`.

## Relaxation

`relax_crystal_candidates(plan, inputs, hamiltonian, chart, *, system_atoms,
method, convergence, maximum_rounds, target_pressure=0.0, maximum_growths=8)`
runs one batched native relaxation of `CrystalRelaxationInput`s under a prepared
Hamiltonian with an explicit chart object:

- `AtomisticDeformationChart` relaxes atoms and cell without symmetry
  (`"variable-cell"`);
- `AtomisticSymmetryChart` with `AtomicSymmetryPlan` or `RigidSymmetryLayout`
  relaxes inside the setting's atomic or rigid-molecule symmetry chart
  (`"symmetry-atomic"`, `"symmetry-rigid"`).

Crystal geometry is angstrom-valued. Before any numerical work the handoff
derives the exact factor from angstrom to the Hamiltonian's length unit. Under
`AtomisticDeformationChart` the start positions and cells are converted to that
unit and the relaxed geometry back to angstrom; a symmetry chart realizes the
plan's angstrom geometry itself and therefore requires an angstrom Hamiltonian.
A length unit with no calibration to the angstrom (such as the uncalibrated
reduced unit system) is refused rather than reinterpreted. Residuals,
`target_pressure`, and the convergence policy are in the Hamiltonian's units,
and the relaxation definition identity binds its length unit.

Both symmetry layouts are built from a host `AsymmetricUnitPlan` but keep only its
prepared `AsymmetricUnitLayout`: `AtomicSymmetryPlan.unit` is the layout, and
`RigidSymmetryLayout.atomic` is its atom-level plan (`RigidSymmetryLayout.unit` is
the same layout object). No molecular witness reaches the compiled chart.

`CrystalRelaxationEvidence` keeps the projected and full force (and stress)
residuals, convergence of both, the evaluation count, and the post-relaxation
symmetry deviation. A constrained minimum with a small projected residual and a
large full residual is reported as projected-only. `system_atoms` maps every
active system atom exactly once with its element identity. Run one relaxation
batch per compatible atom layout.

## Screening, deduplication, and ranking

`screen_dedup_rank(candidates, bank, *, screen, ranking, fingerprint_chunk_size,
pair_chunk_size)`:

1. admits every candidate against one ranking definition: semantic Hamiltonian
   ID, numeric revision, and energy unit; mixed definitions are refused;
2. orders admitted candidates by ranking value, ordinal, and identity — energy
   per formula unit, or `(E - Σ N_Z μ_Z) / N_atoms` with an explicit
   `ChemicalPotentialReference` when compositions differ;
3. applies the greedy, order-dependent radial screen (`RadialScreenPolicy`,
   capacity-checked fingerprints, inclusive threshold). Its decisions are
   evidence only;
4. submits every qualified candidate, including radial duplicates, to the
   persistent `DedupBank`, which merges only on an `EQUIVALENT` certificate from
   `certify_equivalence`. `UNDETERMINED` and search exhaustion never merge;
5. ranks the bank representatives of this submission
   (`rank_crystal_candidates`), excluding the others as duplicates with their
   representative's identity.

`CrystalScreeningOutcome` reports radial screen misses and unconfirmed radial
proposals next to the certified comparisons. `CrystalRankingPolicy` requires
converged relaxations by default; `require_full_minimum=True` also excludes
projected-only minima, and symmetry-deviation and uncertainty bounds are
explicit. A dedup bank holds one ranking definition by caller contract; it is
persisted as canonical commits and must be reopened with the same molecular
components. In `"components"` mode the transitive closure of certified edges
does not certify every member pair.

## Example and gates

`examples/atomistic_crystal_prediction.py` runs a declared CHFClBr graph with
explicit stereo through native conformers, Sohncke-only P1/P2₁ packing under a
uniform demonstration prior, written as a packing recipe and re-admitted from it
(the packer and molecule it uses come from the admitted recipe), with every
attempt published to a structure store, rigid-molecule symmetry-constrained
relaxation with a Lennard-Jones test Hamiltonian, and screening, certified
deduplication, and ranking. The physical cutoff and admitted cell heights are
compatible with minimum-image evaluation. Ranking requires converged projected
minima and successful full evaluations at their exact accepted geometry;
unconverged or failed candidates are excluded rather than assigned an energy.
The bounded packing budget can report a shortfall. Neither the test Hamiltonian
nor the uniform prior is a chemically qualified force field or a
crystal-structure frequency prior.

The closure obligation `atomistic-c20-reconstructible-packing-recipe` asks for a
recipe reconstructed in a fresh process. The fresh-process worker
(`tests/_support/crystal_recipe_restart.py`, driven by
`tests/unit/atomistic/crystal/test_recipe.py`) produces that evidence; until a run
is observed and retained, the obligation is an implemented, unqualified
candidate. Weighted space-group priors with packaged radius and volume data, and
polymorph recovery against independent reference structures, remain external
prerequisites (`atomistic-c20-space-group-prior-and-atomic-data`,
`atomistic-c20-crystal-prediction-scientific-validation`).

Actual crystal-structure prediction additionally needs caller-admitted element
tables, setting priors, and conformer sets with their rights and provenance; a
native physical model whose parameters and weights are admitted for the
chemistry; every declared setting, `Z'`, and molecule envelope; and observed
evidence for each distributed or accelerator configuration used. Packaged
reference priors or prior campaigns are not provided by this package.
