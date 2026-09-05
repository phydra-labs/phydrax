# Generate, observe, and fit single-cell transcripts

This recipe runs the implemented S1 application in
`phydrax.applications.systems_biology.single_cell`: native exact piecewise-constant
telegraph simulation, independent count capture, stationary count-moment fitting,
and disjoint-cell held-out prediction. Stored latent truth is available to audit
the synthetic generator; it is not fed to the fit or count-derived velocity estimator.

## Prerequisites and runnable command

Use Python 3.11–3.13 with this checkout installed and a working supported JAX backend.
Run from the repository root:

```bash
python -m pip install -e .
python benchmarks/single_cell_transcripts.py --cells 64 --genes 2
```

The explicitly configured smoke command is:

```bash
python benchmarks/single_cell_transcripts.py --cells 64 --genes 2 --repeats 2 --event-capacity 1024
```

No AnnData, scVelo, VeloSim, or VeloDyn installation or dataset download is needed.
The CLI requires at least 16 cells and one gene; useful empirical moment errors can
still require more informative replicate cells. `--event-capacity` bounds native
SSA events per interval. Capacity failure raises `ScenarioExecutionError` rather
than returning a truncated path as an experiment; inspect its retained solution
status and increase capacity when exhaustion is the cause.

The command emits an `environment` object and a `benchmark` object as JSON to
standard output. The repeated SSA timing includes host scenario orchestration;
count-moment lowering/compilation is reported separately, not estimated by
subtracting a warm run from a cold run.

## What is generated and what is measured

The benchmark declares 64 stable cell IDs, two stable gene IDs, and one scenario
segment with two exact rate intervals: boundaries at 0, 1, and 12 seconds and saved
nodes at 0, 1, 4, and 12 seconds. Initial promoter, unspliced (U), and spliced (S)
counts are zero. Rates are ordered activation, deactivation, transcription,
splicing β, and mature degradation γ.

The terminal first-gene rates are `(2, 3, 12, 4, 1.5)` per second; transcription is
4 per second in the first interval, and gene-specific transcription is scaled by
gene index. These are deliberately synthetic coefficients, not an experimental
biological clock. Exact native SSA executes each constant-rate interval. Restarting
exponential clocks at the exogenous boundary is exact by memorylessness, but
inserting new boundaries does not promise the same realized path.

Only the first gene is observed and fitted. The independent assay uses binomial
capture probabilities 0.7 for U and 0.6 for S, plus independent Poisson backgrounds
0.15 and 0.2. Each channel's observation capacity is 1024. Assay randomness is in a
separate address namespace from latent SSA, including when callers reuse a root key.
The manifest explicitly identifies synthetic independent coefficient calibration;
it does not pretend to be experimental calibration.

Measured cells are split into disjoint training and held-out halves. The fitting
target contains mean(U), mean(S), var(U), var(S), and cov(U,S), with empirical
influence-based standard errors from measured cells only. The benchmark fixes
activation, deactivation, splicing, and degradation using the independent synthetic
rate calibration and fits **one** free transcription rate. It does not infer five
absolute rates from a stationary snapshot distribution.

Finally, `predict_transcript_velocity` estimates βU − γS after subtracting assay
background and dividing by capture. It uses measured counts and fitted rates, not
stored latent states. The latent comparison in the JSON is an audit available only
because this is a synthetic experiment.

## Interpret the JSON

| Field under `benchmark` | Meaning |
|---|---|
| `profile`, `cells`, `genes`, `paths`, `intervals_per_path`, `event_capacity_per_interval`, `active_events` | Declared synthetic support and realized event workload; no biological population expansion. |
| `ssa_cold_orchestration_including_compilation_seconds`, `ssa_repeated_orchestration` | Cold and repeated host-orchestrated native SSA costs. |
| `observe_seconds`, `fit_seconds`, `count_prediction_compilation`, `count_prediction_execution`, `count_prediction_compiler` | Separate assay, fit, and compiled stationary-observable costs/evidence. |
| `logical_result_bytes` | Unique logical array payload, not total resident or accelerator memory. |
| `boundary_mean_error`, `terminal_mean_error`, `terminal_mean_monte_carlo_standard_error` | Monte Carlo means compared with the exact scheduled first-moment law, with sampling uncertainty. |
| `terminal_nonstationary_mean_bias` | Exact terminal mean minus the stationary mean. The terminal window is not silently declared equilibrated. |
| `stationary_moments`, `expected_observed_stationary_moments` | Distinct latent and capture-transformed stationary observables. |
| `held_out_standardized_moment_residuals` | Predicted moments minus disjoint-cell observed moments, divided by held-out moment errors. This is not an exact count likelihood or posterior predictive test. |
| `capture_noise_mean`, `capture_cross_channel_noise_covariance` | Finite-sample assay-noise diagnostics, not imposed zeros. |
| `held_out_drift_rmse_against_stored_truth_for_benchmark_only` | Count-derived drift against known synthetic latent drift; not an experimental velocity measurement. |
| `fit_rank`, `fit_free_parameters` | Local sensitivity rank for the free rates; the calibrated fixed rates remain fixed. |
| `scientific_gates` | Explicit experimental/time/velocity/UQ limitations. |

The observed 64-cell × 2-gene smoke executed **128 paths and 33,613 events**.
Maximum absolute held-out standardized moment residual was
`2.4761358858990175`; fit rank was one for one free parameter. That residual is a
recorded synthetic outcome, not an assertion that every held-out moment lies
within two standard errors, or that the model has experimental accuracy. No
cross-hardware timing or scaling claim is implied.

## Change the protocol without changing its meaning

Use `PiecewiseConstantRates` for finite positive `(interval, gene, 5)` arrays with
strictly increasing boundaries and explicit inverse-time units. Schedule values
at an interior boundary use the interval to its right. The exact profile admits
no smooth callable hazards. `repeat(cycles)` unrolls a finite external protocol;
it does not synthesize cell-cycle biology.

`TranscriptScenario` bounds cells × genes × segments through `max_paths` and
interval events through `max_events_per_interval`. A root segment starts from the
supplied initial states. A child segment starts at its parent's terminal physical
time and inherits its state with new addressed randomness. Multiple children are
counterfactual continuations, **not cell division or molecule partitioning**.
`joined_series` disconnects every segment reset so branches cannot create false
lag pairs. Cell/gene worksets preserve stable random addresses; schedule/segment
changes are not a promise of pathwise coupling.

## Use real counts or an external velocity field

For real measurements, extract raw U/S columns and stable source-row-to-cell IDs
explicitly, then call `import_transcript_arrays`. It is an array adapter, not an
AnnData or scVelo provider loader. Preserve the validity mask, gene identity,
assay calibration, preprocessing identity, source manifest, and coordinate
semantics (`physical_time`, `pseudotime`, or `none`). Active values must be finite
nonnegative integer counts. Do not pass normalized/log expression even if some
values happen to be integral. The adapter cannot infer biological provenance from
an integer array.

Pseudotime cannot carry a physical time unit and does not become seconds on import.
Snapshot `TranscriptCounts.to_series()` has no connecting trajectory edges. Use
`import_velocity_field` to retain an externally inferred array with estimator,
preprocessing, representation, and uncertainty identities, aligned masks, and
optional standard errors. `standard_errors=None` retains unknown UQ. An embedded
arrow is not a physical CTMC path, biological lineage, or energy landscape.

For native count-based inference, independently calibrate capture/background and
any fixed physical rates with rights-cleared reference artifacts and known
uncertainty. `StationaryCountTarget.from_counts` additionally needs justified
equilibrium evidence and positive moment-error scales. Stationary snapshots identify
rate combinations, not an absolute clock; rank-deficient fits do not acquire
covariance or physical velocity from an arbitrary gauge. Check `result.successful`
and `identifiability` before interpretation. Covariance is first-order conditional
fit covariance, **not a posterior**, and excludes unpropagated calibration/model
discrepancy. Negative background-corrected drift estimates are retained, not clipped.

The application does not supply rights-cleared experimental assay/rate calibration,
held-out biological validation, smooth-hazard simulation, division, or physical
pseudotime. Those scientific prerequisites cannot be replaced by the successful
synthetic workflow. Read the [focused single-cell guide](../guides_systems_biology.md#single-cell-transcript-scenarios-s1),
[public API](../api/advanced_biophysics.md#single-cell-transcripts), and
[source dispositions](../biophysical_sources.md) for admission and refusal details.
