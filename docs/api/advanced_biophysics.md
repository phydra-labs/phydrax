# Advanced biophysics

## Path-space sampling

::: phydrax.stochastic.path_sampling
    options:
      show_root_heading: true
      members_order: source

## Electrophysiology

::: phydrax.applications.electrophysiology
    options:
      show_root_heading: true
      members_order: source

## Cardiovascular platform

The [cardiovascular application API](applications/cardiovascular.md) composes
generic equation, discretization, solver, lifecycle, and qualification owners
into bounded research workflows. Numerical success does not establish a
clinical, diagnostic, treatment, regulatory, or commercial claim.

::: phydrax.applications.cardiovascular
    options:
      show_root_heading: true
      members_order: source

## Cellular mechanics

::: phydrax.applications.cellular_mechanics
    options:
      show_root_heading: true
      members_order: source

## Systems biology

::: phydrax.applications.systems_biology
    options:
      show_root_heading: true
      members_order: source

### Single-cell transcripts

The public leaf `phydrax.applications.systems_biology.single_cell` composes the
existing telegraph, count-measurement, exact SSA, series, optimization, and UQ
contracts. The [systems-biology guide](../guides_systems_biology.md#single-cell-transcript-scenarios-s1)
and [runnable cookbook](../cookbook/single_cell_transcripts.md) explain the exact
profile and scientific limits without replacing the systems-biology material above.

- `PiecewiseConstantRates` and `TranscriptScenario` admit finite positive
  `(interval, gene, 5)` telegraph rates, declared physical units, bounded
  cells/genes/segments, and bounded interval events. Scenario forks and repeated
  protocols are not cell division; joined series disconnect every segment reset.
- `generate_transcripts` preserves stable cell/gene/segment/interval/event random
  addresses. `TranscriptPath` keeps latent `(promoter_on, U, S)` counts separate
  from conditional mature-count drift. Exact interval means do not make sampled
  SSA paths differentiable, and smooth callable hazards are not admitted.
- `TranscriptCountAssay` and `observe_transcripts` use independently calibrated
  capture/background and a separate random namespace. `TranscriptCounts` contains
  measured U/S snapshots, masks, source/assay/preprocessing identities, and explicit
  physical-time/pseudotime/absent-coordinate semantics, not an inferred lineage.
- `fit_stationary_counts` fits observable moments with declared error scales, not
  an exact count likelihood. Stationary data identify rate combinations rather
  than an absolute clock; independent fixed-rate calibration and local rank are
  separate evidence. Returned first-order conditional covariance is **not a
  posterior** and does not include unpropagated assay/rate/model discrepancy.
- `predict_transcript_velocity` uses calibrated measured counts, never stored
  latent truth. It requires a successful identifiable fit and independent rate
  clock. Imported count and velocity arrays retain explicit mapping, masks, rights,
  preprocessing, estimator/representation identity, and unknown uncertainty;
  neither embedded arrows nor pseudotime acquire physical-time or energy meaning.

Unknown calibration/reference uncertainty is `None`, not zero. Rights-cleared
experimental assay/rate calibration and held-out biological evidence remain
scientific gates; successful synthetic execution does not establish biological
timing, experimental velocity accuracy, or commercial readiness.

::: phydrax.applications.systems_biology.single_cell
    options:
      show_root_heading: true
      members_order: source
      members:
        - CellIdentity
        - GeneIdentity
        - PiecewiseConstantRates
        - ScenarioSegment
        - TranscriptScenario
        - ScenarioExecutionError
        - TranscriptPath
        - TranscriptExperiment
        - generate_transcripts
        - transient_transcript_mean
        - scheduled_transcript_mean
        - TranscriptCountAssay
        - TranscriptCounts
        - observe_transcripts
        - StationaryCountTarget
        - TranscriptIdentifiability
        - TranscriptFit
        - TranscriptVelocityEvidence
        - predicted_count_moments
        - fit_stationary_counts
        - predict_transcript_velocity
        - ImportedTranscriptCounts
        - ImportedVelocityField
        - import_transcript_arrays
        - import_velocity_field

## Biophysical observations

::: phydrax.observation
    options:
      show_root_heading: true
      members_order: source
