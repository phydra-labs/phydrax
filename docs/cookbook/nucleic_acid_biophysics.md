# Nucleic-acid biophysics workflows

This cookbook separates an actual processed-assay fit from structural, mechanical,
secondary-CTMC and electronic numerical checks. They do not share one inferred
physical clock or one accuracy claim. See the [scientific guide](../guides_nucleic_acid_biophysics.md),
[API reference](../api/applications/nucleic_acid_biophysics.md), and
[source dispositions](../biophysical_sources.md).

## Prerequisites

Run from a repository checkout containing `benchmarks/` and `tests/fixtures/`, with
Python 3.11–3.13 and the checkout's Phydrax package/dependencies installed in `.venv`.
For a new environment, use the repository installation instructions; the commands below
assume `.venv/bin/python` imports that checkout. No external prediction provider,
commercial parameter table, network download, or pretrained checkpoint is required for
these workflows. The retained RMDB fixture and its `.source.json` record must remain
together. Enable JAX 64-bit arithmetic before launching each process.

Entry points intentionally differ: `nucleic_structure.py` and
`nucleic_chemical_mapping.py` import the sibling `_runtime` and run as **direct scripts**;
rigid, secondary and electronic benchmarks import `benchmarks._runtime` and run with
**`-m`**. Do not mechanically interchange those forms.

## Complete scientific workflow: fit retained RMDB observations and test held-out constructs

The retained artifact is
`tests/fixtures/nucleic_acid_biophysics/TODEX_DMS_0000.rdat`, alongside
`TODEX_DMS_0000.source.json`. The source record identifies
[RMDB TODEX_DMS_0000](https://rmdb.stanford.edu/detail/TODEX_DMS_0000/), its
[raw download](https://github.com/DasLab/rmdb.github.io/releases/download/data-general/TODEX_DMS_0000.rdat),
CC0-1.0 rights, and SHA256
`2a597de8277f0965543340210381b0ff6debe4f406c67a445571460c355cc2b5`.
This is real retained processed DMS data, not synthetic reactivity.

```bash
JAX_ENABLE_X64=true .venv/bin/python benchmarks/nucleic_chemical_mapping.py \
  --fixture tests/fixtures/nucleic_acid_biophysics/TODEX_DMS_0000.rdat \
  --training-constructs 12 --repeats 3 --output nucleic-mapping.json
```

The actual entry point performs the full workflow:

1. Read exact RDAT bytes and sibling provenance; admit the source checksum and requested
   training use through `ReferenceArtifactManifest` and `import_processed_rdat`.
2. Preserve row-specific mutant sequences, source position mapping, preprocessing,
   annotations, negative corrected values and unpooled observation identities.
3. Explicitly treat aligned `REACTIVITY_ERROR` as standard deviations. The likelihood
   is a diagonal-noise approximation because the file supplies no measured
   cross-position covariance. Unreported source-level uncertainty remains `None`.
4. Use depositor-designed dot-bracket structures only as **hypothesis features**:
   unpaired positions receive accessibility one and other positions zero. These are
   not independently solved experimental pairing labels.
5. Fit an `AccessibilityReactivityModel` on the first 12 complete constructs with a
   shared baseline and accessibility slope, and no condition covariates. The last
   four complete constructs remain withheld; their observations are never refitted.
6. Report optimizer success, whitened-design rank/identifiability, held-out RMSE,
   standardized residual chi-square, individual held-out log likelihoods, retained
   negative-value count, environment and separated compilation/execution timing.

### Interpret the observed result, including model failure

The completed standalone run with this 12/4 split observed:

| Quantity | Observed result |
|---|---:|
| Training / withheld constructs | 12 / 4 |
| Withheld measurements | 540 |
| Negative reactivities retained across the import | 702 |
| Optimizer | Converged |
| Design rank / identifiable | 2 / true |
| Withheld RMSE | 0.7700647162757772 |
| Withheld chi-square per observation | 230.2991608301379 |

**The simple affine accessibility model is not experimentally adequate under the
supplied-error assumption.** Convergence and rank two prove neither predictive adequacy
nor correct pairing. The large held-out standardized residuals remain part of the result;
do not clip negative reactivities, relabel designed structures as solved references,
exclude difficult constructs, inflate uncertainties, or retune the split to manufacture
an accuracy claim. A richer observation/noise model would require independent scientific
justification and a new held-out evaluation.

`withheld_rmse` is in the processed normalized-reactivity scale. The standardized
chi-square is dimensionless under the declared SD law. `withheld_log_likelihood` is an
observation likelihood, not the probability that an RNA structure is correct.
`score_compile` and `score_execution_seconds` measure the held-out score kernel;
`fit_total_seconds_including_native_preparation_and_compilation` has a broader boundary.
Numerical completion and **negative scientific qualification** are both useful outcomes.

This workflow ends at an assay-model evaluation. It does not manufacture distance
restraints, reconstruct an atomistic fold, calibrate a rigid Hamiltonian, or infer rates.
`IntervalDistanceReconstruction` is available only when independently sourced intervals,
uncertainty, existing atom support, gauge constraints and explicit chirality evidence
are supplied. A local interval fit cannot remove reflection ambiguity without chirality
or certify all-atom physical validity.

## N1: structural descriptor and torsion qualification

```bash
JAX_ENABLE_X64=true .venv/bin/python benchmarks/nucleic_structure.py \
  --lengths 8 32 --repeats 3 --output nucleic-structure.json
```

This independently constructed geometry workflow binds nucleotide/atom identities,
compares dense and sparse directed-pair eRMSD, applies a proper rigid transformation,
differentiates the squared descriptor, and observes torsion/pucker geometry. Inspect
`frame_valid`, `frame_orthogonality_error`, `rigid_error`,
`force_translation_residual`, `sparse_dense_absolute_error`, and the torsion probe.
Pair capacity, active pairs, compilation evidence, logical bytes and synchronized
execution time distinguish support size from run cost.

The observed length-8/32 standalone run had maximum dense/sparse absolute error
`1.3552527156068805e-20`, maximum rigid error `1.2166854568952983e-14`, and proper-rigid
torsion error `8.326672684688674e-16`. These are synthetic descriptor checks, not
experimental structure validation. The nonideal sugar's harmonic residual
`0.9086033956406383` was retained, not hidden.

The command uses published G (`--smooth-width 0`, the default). A positive
`--smooth-width` exercises a distinctly identified C2 taper and must not be reported as
exact published eRMSD. Sparse/full equivalence is fixture-specific: general sparse
support needs proof that omitted contributions vanish in both conformations.

## N3: rigid mechanics, not calibrated duplex kinetics

```bash
JAX_ENABLE_X64=true .venv/bin/python -m benchmarks.nucleic_rigid \
  --bodies 16 --steps 100 --repeats 3
```

The benchmark uses independently authored **noncalibrated** coefficients and paired
rigid bodies. It computes energy/force/torque, finite-difference wrench agreement,
conservative trajectories at two step sizes, thermostatted evolution with explicit
random keys, and finite-time OU velocity covariance checks. Read `conservative_successful`,
`thermal_steps_successful`, force/torque balance, wrench error, both energy drifts and
OU variance errors together; an energy number alone is not qualification.

Body capacity, five physical sites per body, and eight total physical/frame markers per
body have different meanings. Model units here are reduced. The short OU comparison is
not evidence of configurational equilibrium. This run supplies no actual published
coefficient calibration, melting/duplex observable, physical clock, or experimental
DNA/RNA accuracy. Using real coefficients requires exact rights-cleared parameter bytes,
matching geometry/chemistry/conditions, and independent reference uncertainty and data.

## N4: exact CTMC first hits and capacity-aware reference comparison

```bash
JAX_ENABLE_X64=true .venv/bin/python -m benchmarks.nucleic_secondary_kinetics \
  --paths 2048 --copies 1 --capacity 64 --repeats 3
```

This is an independently specified labelled A/T binding CTMC, not experimental DNA
kinetics. It admits its mathematical parameter artifact, compiles the complete legal
support, runs native direct SSA with a fixed random key, and compares with finite-generator
transition probabilities and analytical first-hit behavior. `--copies` increases
competing physically labelled T partners, not sequence length.

The analysis uses the actual event ledger even though only endpoint save times are
requested. Inspect `successful_paths`, `capacity_failures`, `censored_paths`,
`incomplete_first_hits`, `maximum_observed_event_count`, escaped rates and generator/
hitting/MFPT residuals. Exact hits before later exhaustion survive; unsuccessful
non-hits are incomplete, not right-censored. CDF lower/upper bounds retain unknown
portions of incomplete paths. `final_probability_comparison_qualified` is false and
`final_probability_max_error` is `null` unless all paths complete; the successful-only
empirical distribution is not silently promoted to an unconditional result.

Compare first-hit CDF discrepancies against the reported Monte Carlo standard error,
not a deterministic exact-equality criterion. An intentionally reduced generator is
also reported with escaped-rate evidence, not treated as closed. For other closed models,
`finite_generator_hitting` explicitly handles unreachable targets and non-hitting closed
classes: unconditional MFPT is infinite whenever hitting is not almost sure.
Model rate units do not establish an experimental clock or a macroscopic second-order
association constant.

## N5: electronic compiler and open-quantum numerical workflow

```bash
JAX_ENABLE_X64=true .venv/bin/python -m benchmarks.nucleic_electronics \
  --sizes 2 4 8 --steps 4 --trajectories 1024 --repeats 3
```

This physical-unit benchmark refuses disabled x64. It uses explicit analytic electronic
parameters, not calibrated DNA/RNA tables. `analytic` and `scaling` results separate
reference checks from preparation, compilation and execution measurements. Electronic
basis size is not the number of atoms, and dense Liouville cost is not rigid-body cost.

Interpret electronic populations/coherences and density validity separately from any
mechanical or secondary-state result. Energies are converted once to H/ℏ; dimensionless
jump operators and inverse-time rates remain separate until √rate forms collapse
operators. Fixed-step quantum-trajectory estimates require ensemble and step-size
convergence; they are not the event-exact SSA used above. Recombination removes carrier
weight into an explicit vacuum, without renormalizing it away or turning it into a
radiation lesion. Numerical agreement on these analytic cases does not provide an
experimental electronic parameterization.

## Coordinate proposals: keep the training-data gate

The public `generation` leaf supports actual rights-admitted training and sampling,
not an included pretrained nucleotide predictor. A nucleotide workflow must supply its
own complete, nonperiodic fixed-chemistry `AtomisticBatch`, exact `NucleotideAtomMapping`,
gauge atom IDs, geometry/chirality policy, conformers, conditions and disjoint
training/validation groups. The source data, provider outputs, prepared inputs and
learned weights keep their independent use restrictions.

Use `prepare_nucleic_coordinate_support` and `map_nucleic_hypothesis` before
`prepare_coordinate_training_data`, `fit_coordinate_model`, and
`sample_coordinate_proposals`; persistence uses `save_coordinate_model` and
`load_coordinate_model`. These are documented in the [API](../api/applications/nucleic_acid_biophysics.md#coordinate-proposals-and-provider-admission).
There is no automatic completion of missing material atoms or transfer to another
sequence/support. Retain raw and canonical proposals, solver validity and every sample's
geometry status. A generated sample's geometry acceptance is not confidence, likelihood,
experimental folding accuracy or an equilibrium weight; integration time 0–1 is
pseudotime. Neither synthetic descriptor geometry nor the inadequate RMDB accessibility
fit above supplies the missing nucleotide training corpus or pretrained model.
