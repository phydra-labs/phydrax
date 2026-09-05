# Radiation biophysics

`phydrax.applications.radiation_biophysics` converts retained external radiation
records into source-linked **initial** lesions, contour clusters, dose-normalized
yields, and qualified two-cause probability fits. R1 is the import → target mapping
→ lesion → cluster → yield workflow. R2 fits probabilities on fixed candidate
support and evaluates independent held-out conditions and upstream stage evidence.

This is not a native ionizing-particle transport, spatial-radiolysis, DNA repair,
cell-survival, or clinical dose-response engine. An adapter success, a converged
fit, or a synthetic benchmark does not establish any of those capabilities.

## The exact external profile

The implemented adapter is `radiation_biophysics.interchange`'s
`Geant4-dnadamage1-ROOT-11.3.0` profile, pinned to Geant4 `v11.3.0`. It is not a
generic Geant4, TOPAS, gMicroMC, or arbitrary ROOT importer. The format authority is
the pinned [RunAction.cc](https://github.com/Geant4/geant4/blob/v11.3.0/examples/extended/medical/dna/dnadamage1/src/RunAction.cc),
[SteppingAction.cc](https://github.com/Geant4/geant4/blob/v11.3.0/examples/extended/medical/dna/dnadamage1/src/SteppingAction.cc),
and [TimeStepAction.cc](https://github.com/Geant4/geant4/blob/v11.3.0/examples/extended/medical/dna/dnadamage1/src/TimeStepAction.cc)
writers. See also [biophysical source dispositions](biophysical_sources.md).

| ROOT tree | Required columns | Meaning |
|---|---|---|
| `ntuple/ntuple_1` | `x`, `y`, `z`, `edep`, `diffKin`, `volumeName`, `CopyNumber`, `EventID` | Positions in nm; deposited energy and kinetic-energy loss in eV; source volume/copy and primary-history identifiers. |
| `ntuple/ntuple_2` | `x`, `y`, `z`, `RadName`, `EventID` | Positions in nm and selected radical/deoxyribose damage-reaction records. `RadName` must be decoded text. |

The physical column view must contain exactly that physical column set, and the
chemical view exactly that chemical set. IDs must be nonnegative integers, not
truncated floats; columns and original-entry-ID arrays must align. Original entry
IDs are unique within each ntuple.

`edep` becomes `PhysicalInteraction.deposited_energy`. `diffKin` becomes
`kinetic_energy_loss`: it is **not** transported/carried energy and must not replace
energy deposition. `carried_energy` remains `None`. Track, parent-track, process,
physical particle species, and event time are also unreported. The chemistry writer
records selected OH reactions producing damaged deoxyribose, not every radiolysis
reaction. The adapter assigns the channel `OH-deoxyribose-damage`, with recorded
radical and `Deoxyribose` reactants and `DamagedDeoxyribose` product. This selected,
untimed ledger cannot establish time-dependent chemical G-values.

### Real ROOT and retained-column entrypoints

`import_dnadamage1_root(path, ...)` opens a real binary ROOT file with `uproot`, reads
the two trees, and preserves original tree-entry numbers. Install the optional
extra when this route is needed:

```bash
python -m pip install -e '.[radiation-interop]'
```

The extra is `radiation-interop`, with `uproot>=5.6,<6`. `uproot` is imported only
inside the ROOT reader, after source-rights and file-byte admission. Importing the
application or its interchange facade does not import an optional ROOT provider.
No Geant4 execution or PyROOT installation is supplied by this extra.

`import_dnadamage1_columns(physical_columns, chemical_columns, ...)` consumes the
same pinned profile without loading a ROOT library. It requires explicit
`physical_entry_ids` and `chemical_entry_ids`; do not replace them with new row
numbers after filtering or reordering. `dnadamage1_column_payload(...)` supplies the
canonical derivative bytes for its checksum. Those bytes are not the original ROOT
bytes: retain the governed ROOT artifact as a parent when admitting a derivative.

Both entrypoints require a `RadiationSource`, `run_id`, `fraction_id`, and an explicit
`volume_materials` mapping. The source must declare engine `Geant4-dnadamage1`,
revision `v11.3.0`, nm/eV writer units, coordinate frame, retained configuration,
random lineage, source-table artifacts, cutoffs, and governing rights manifests.
Nonempty chemistry additionally requires its physical endpoint, chemistry model,
and scavenging model from retained run artifacts. These are not inferred from
coordinates or a citation.

The first governing manifest must match the admitted bytes' checksum and size;
the complete `ScientificArtifactEnvelope` binds its digest. Every retained parent
is re-admitted by `RadiationSource.require_rights` for the requested downstream use.
Permission to read is not permission for commercial use, redistribution, training,
or export. The importer grants no rights and generates no external parameter table.

Inspect `ImportedRadiationLedgers.report`: its status is `DECLARED_LOSS`, documenting
unreported fields and incomplete chemical coverage, not falsely claiming a
lossless transport record. Set `required_semantics` to capabilities your workflow
actually requires. Asking for `event_time`, `track_id`, `parent_track_id`, `process`,
`physical_species`, or `carried_energy` is refused; unknown semantic names are also
refused. The supported recorded semantics are `event_identity`, `primary_history`,
`deposited_energy`, `reaction_channel`, `coordinates`, and `material` (the latter
uses the supplied physical volume map; it does not fill absent chemical material).

## Preserve raw history and derived target identities

`PrimaryHistoryKey` binds source artifact, run, primary ID, and dose fraction.
`RadiationEventKey` adds physical/chemical stage and original record ID
(`ntuple_1:<entry>` or `ntuple_2:<entry>`). A repeated `EventID` in another run,
artifact, or fraction is not the same primary. Ledgers sort by these identities,
deduplicate identical records, and refuse conflicting duplicates.

Targets have separate identities. A source physical site is `volumeName:CopyNumber`;
a `TargetSite.target_id` is a derived scoring identity, not that source ID or an
atom ID. `TargetMolecule` declares one or two strand IDs, a positive contour length,
and explicit linear/circular topology. The caller must align contour coordinates
across strands. Sites declare molecule, strand, integer contour position,
`backbone` or `base` component, center, positive radius, and material.

`RadiationTargetGeometry` retains source-geometry identity, units, source/target
frames, a proper rotation and translation, approximation, and material policy.
`prepare_radiation_targets` compiles native scoring spheres. This geometry is not
an atomistic material or a transport mesh.

`map_radiation_targets` uses explicit `SourceTargetRoute` entries for records with
reported source sites. Their many-to-many deposition fractions must sum to one
per source site. It does not silently fall back to coordinate matching when a
reported site lacks a route. Records without source sites use transformed sphere
membership. Overlap and unmapped records fail by default:

- `overlap_policy="equal-share"` explicitly divides an overlapping hit among
  targets instead of copying its energy into each;
- `unmapped_policy="outside"` retains unmapped event keys and a report declaring
  that they lie outside scored support;
- `material_policy="require-match"` refuses missing or mismatched mapped material;
  `"scoring-only"` explicitly does not assert material equivalence.

Mapping is host-orchestrated in bounded coordinate batches. Changed ledger,
geometry, or configuration identities invalidate the mapping used for lesions.

## From deposition and reactions to initial lesions

A `LesionPolicy` makes direct threshold, probability, component selection, indirect
channel/species rules, and chemistry endpoint/model/scavenging assumptions explicit.
The default component support is backbone; add base only when the intended policy
supports it.

`candidate_radiation_lesions` sums allocated **deposited** energy separately for each
primary and target. The direct threshold is inclusive. It retains every contributing
event key and never combines independent histories or fractions to cross a threshold.
Carried energy and kinetic-energy loss do not enter this decision.

An indirect candidate is one matched chemical reaction, not a dose proxy. Its
probability includes the declared reaction rule and target allocation. Ambiguous
multiple rule matches are refused. The policy must match the source chemistry and
scavenging models and stay within the source endpoint. Untimed reactions cannot be
re-filtered to a different endpoint; the pinned profile therefore requires the
retained endpoint itself. Chemistry time is not a repair time.

`realize_radiation_lesions(..., random_lineage=...)` uses stable candidate-addressed
Bernoulli draws, or explicitly supplied uniforms covering every candidate exactly
once. Acceptance is `u < p`, so zero and one probabilities have exact endpoint
behavior. Accepted candidates at one target/history form one initial lesion with
all accepted causes and parent event keys. A lesion can be both direct and indirect:
those cause counts are not disjoint and must not be added as if they were.

Thresholding, target membership, Bernoulli decisions, and clustering are discrete
host operations, not pathwise-differentiable radiation dynamics.

## Contour clusters, not repair states

`cluster_radiation_lesions(..., maximum_contour_gap=...)` constructs transitive
connected components under an inclusive contour-distance rule, within one molecule
and primary history only. Circular distance includes closure around the molecule;
Euclidean proximity alone does not define a strand-break cluster.

A `DSB` cluster requires an actual opposite-strand backbone pair within the declared
gap. A transitive cluster containing both strands, perhaps bridged by base lesions,
does not by itself establish that pair. Other classifications are `SSB`,
`SSB-cluster`, and `base-damage`. Classification, lesion IDs, cause provenance,
backbone/base counts, geometry identity, and realization identity remain available.
There is no repair-state evolution or inferred lethality attached to these labels.

## Dose, scored mass, yield, and unknown uncertainty

Supply one `HistoryExposure` for **every** independent primary, including zero-dose
and zero-lesion histories. It stores total deposition into the whole scored mass,
not just deposition inside DNA target spheres. Dose is deposited energy in joules
divided by scored mass in kilograms, in Gy. Base pairs count duplex pairs once for
the scored ensemble, not individual nucleotides.

`radiation_yield` reports a ratio of summed counts to summed exposure denominators:

| Convention | Per-history denominator | Returned unit |
|---|---|---|
| `per-primary` | 1 | Dimensionless count per primary |
| `per-Gy` | Dose in Gy | Gy⁻¹ |
| `per-Gy-per-Mbp` | Dose × base pairs / 1,000,000 | Gy⁻¹, with explicit Mbp convention |
| `per-Gy-per-molecule` | Dose × molecule count | Gy⁻¹, with explicit molecule convention |
| `per-Gy-per-kg` | Dose × scored mass in kg | J⁻¹ |

`per-Gy` refers to the declared scored ensemble, not an implicit single plasmid.
Duplicate/missing exposures, foreign sources, mixed dose fractions, and nonpositive
total denominators are refused. Available observables are `lesions`, `direct`,
`indirect`, `SSB`, `SSB-cluster`, `DSB`, and `base-damage`; cluster observables count
clusters of that classification.

`history_sampling_standard_error` is an independent-primary delta-method sampling
error and is `None` for a single history. `normalization_standard_error` separately
propagates supplied independent dose errors, conditional on exact declared mass,
molecule count, and base-pair count. If any needed `dose_standard_error_gy` is
unknown, the normalization error remains `None`, not zero. Per-primary normalization
has exact declared denominator and zero normalization error. Neither field silently
includes external transport, chemistry, geometry, policy, or model discrepancy.

## R2: probability fitting and staged qualification

`prepare_lesion_expectation(candidates, denominator=...)` groups direct/indirect
multiplicities on fixed history/site support. It requires **unthinned,
unit-probability candidates**: fractional geometry routes or previously selected
lesions cannot masquerade as probability-calibration trials. Use the canonical
physical denominator for the declared yield convention.

`expected_initial_lesion_yield` differentiates the independent-candidate union law,
not a sum that double-counts repeated damage:

> Site lesion probability = 1 − (1 − p_direct)ⁿ_direct × (1 − p_indirect)ⁿ_indirect.

Its array support is logits `(2,)`, multiplicities and active mask
`(condition, site)`, and denominators `(condition,)`. Inactive padding is not an
extra physical undamaged site. This is an expected initial-site-lesion yield, not
an expected DSB-cluster model or differentiable transport/chemistry model.

`RadiationCalibrationData` binds observation IDs, physical conditions (dose,
oxygen, scavenger, chemistry endpoint), candidate supports, measured yields,
positive standard errors, exact yield unit, normalization convention, source kind,
and reference manifest. Unknown reference uncertainty is `None` and cannot be
admitted for quantitative comparison. Declaring uncertainty does not make it true:
retain its experimental or reference justification.

`calibrate_radiation_lesions` runs native least squares and UQ for two logit
probabilities, with an explicitly supplied proper Gaussian logit prior. The Gaussian
likelihood assumes independently calibrated observation errors; correlated errors
or transport uncertainty need a different explicit likelihood. Training requires
training rights; held-out and stage references retain their own use-rights checks.
Training and held-out splits must have disjoint observation IDs, condition IDs,
**and physical condition tuples**, with matching normalization conventions.

The result separates likelihood rank/singular values from its Laplace posterior
approximation and linearized held-out parameter variance. A prior can give finite
posterior intervals without making the likelihood identify both causes. Rank below
two remains an explicit scientific gate. Held-out residuals combine declared
measurement errors with propagated parameter variance; they do not acquire
unmodeled transport or biological uncertainty.

`RadiationStageEvidence` independently evaluates `transport`, `chemical-G`,
`target-reactions`, and `lesion-yields`, each with retained upstream artifacts,
reference rights/uncertainty, units, and a declared standardized-RMS threshold.
Time-dependent chemical-G comparisons require external complete radiolysis evidence;
the selected dnadamage1 damage ntuple cannot supply it. The calibration result's
`scientifically_qualified` is true only when its `gates` are empty: both cause
probabilities are identified, training and held-out lesion observations are
experimental, held-out error meets the criterion, and all four stages have accepted
non-synthetic evidence. Fitting a downstream yield never validates an upstream engine.

## Runnable workflow and qualification boundary

The [radiation cookbook](cookbook/radiation_biophysics.md) runs the repository's
actual pinned-column mechanics and calibration benchmark and explains the ROOT
admission route. The recorded smoke used 128 histories, 384 physical records, and
128 chemical records, produced 128 DSB clusters invariant to record reordering,
and had held-out synthetic maximum absolute error `9.983941036906252e-6` with
likelihood rank two. Real binary ROOT reading is also covered by synthetic
pinned-format evidence; this does not turn the benchmark's hand-authored columns
into a provider-generated campaign.

A rights-cleared real provider event corpus and experimental campaign with
uncertainty are still missing. Those are completion gates for external transport,
radiolysis, target-reaction, and biological initial-yield qualification, not grounds
to substitute zero uncertainty or claim clinical/repair accuracy. See the
[public API](api/applications/radiation_biophysics.md) for the implemented contracts.
