# Systems biology

`phydrax.applications.systems_biology` provides a fixed-shape intermediate representation for compartmental biological process networks, exact stochastic simulation through the PhydraX jump-process solvers, declared deterministic and chemical-Langevin approximations, a prepared telegraph gene-expression model, host evidence bindings, and atomic multiprocess whole-cell assembly.

## Compartmental process plans

A plan separates host-side biological meaning from the arrays used by compiled execution:

```python
import jax.numpy as jnp
from phydrax.applications import systems_biology as sb

cell = sb.CompartmentSpec("cell", 1.0, unit="cell")
plan = sb.StoichiometricNetworkPlan(
    "conversion",
    (cell,),
    (sb.SpeciesSpec("a", "cell"), sb.SpeciesSpec("b", "cell")),
    (
        sb.StoichiometricProcessSpec(
            "forward",
            {"a": -1, "b": 1},
            sb.MassActionPropensity(0.1, {"a": 1}),
        ),
    ),
    stoichiometry_capacity=2,
)
network = plan.prepare()
state = network.initial_state(jnp.asarray([100.0, 0.0]))
evaluation = network.evaluate(state)
```

Stoichiometry is supplied sparsely with nonzero integer coefficients. Preparation resolves names once and produces dense stoichiometry, reservoir-masked dynamic stoichiometry, and fixed-capacity sparse index/value/mask arrays. Unknown species, duplicate identities, nonintegral coefficients, cross-compartment higher-order mass action, and insufficient sparse capacity are rejected on the host.

Four propensity laws are available:

- `MassActionPropensity` uses falling-factorial combinatorics for exact counts and the corresponding power law for deterministic/CLE execution. Compartment measure scaling and factorial normalization are identical in both paths.
- `HillPropensity` supports activating and repressing responses, a positive Hill coefficient and half-saturation concentration, and a nonnegative basal rate.
- `MichaelisMentenPropensity` is an explicit saturating substrate law.
- `PromoterTransitionPropensity` is a first-order discrete transition gated by the source promoter count.

Every evaluation reports state validity, parameter validity, finite/nonnegative rates, a status code, source and sink rates caused by fixed reservoirs, and residuals in the prepared conservation basis. Invalid rates are returned as `NaN` rather than silently clipped. A reaction that lacks the molecules it consumes has zero exact propensity, so an SSA jump cannot make a valid count state negative.

## Exact SSA and approximations

`network.exact_jump_process()` implements the native `AbstractJumpProcess` contract. Compose it with `PoissonClockRealization` and `solve_direct_ssa` or `solve_next_reaction`; the realization's `process_id` must equal the adapter's identity. Reusing a realization therefore reproduces the same path, and capacity extension preserves existing random prefixes.

```python
import jax
from phydrax.solver import solve_direct_ssa
from phydrax.stochastic import PoissonClockRealization

process = network.exact_jump_process()
noise = PoissonClockRealization(
    jax.random.key(0),
    process.num_channels,
    support=(0.0, 10.0),
    max_events_per_channel=128,
    process_id=process.process_id,
)
solution = solve_direct_ssa(
    process,
    noise,
    state,
    t0=0.0,
    t1=10.0,
    save_times=jnp.linspace(0.0, 10.0, 101),
    args=network.default_runtime(),
)
```

`deterministic_step` and `cle_step` return candidates; call `result.commit(state)` to retain the input on numerical failure. Their `ApproximationEvidence` deliberately distinguishes numerical validity from scientific regime validity:

- deterministic validity declares a configurable minimum copy number across input and candidate endpoints for nonreservoir species whose quantity is `count`; non-count deterministic fields are not compared to copy thresholds;
- CLE is restricted to count-valued networks and additionally declares a configurable minimum expected event count over the step;
- neither approximation hides a negative or nonfinite candidate;
- deterministic differentiation is declared only away from Hill/Michaelis--Menten driver boundaries and reservoir-ledger kinks;
- CLE differentiation additionally requires positive duration and channel intensities for its fixed-noise reparameterized candidate;
- exact SSA sample paths are not declared differentiable because discrete event choices change with parameters.

A false `regime_valid` value does not rewrite the mathematical candidate. It tells the caller that exact SSA, a smaller step, or a better approximation is required.

## Thermochemical specialization

This package does not redefine chemical species thermodynamics, Arrhenius rates, reversibility, charge balance, or element balance. Those semantics remain in `phydrax.equations.ChemicalMechanismIR` and `PreparedChemicalMechanism`, with exact-count chemical jumps in `phydrax.stochastic.ChemicalJumpProcess`.

For a biological process that is also a thermochemical reaction:

1. give each `SpeciesSpec` a `thermochemical_name`;
2. give each mass-action `StoichiometricProcessSpec` its `thermochemical_reaction` name;
3. call `network.bind_thermochemical(prepared_mechanism)`.

The binding fails unless every species resolves and every declared reaction has exactly matching reactant coefficients, product coefficients, and forward mass-action orders. The biological reactant side is reconstructed from its explicit kinetic orders, including catalysts restored on the product side. `ThermochemicalInteropEvidence` retains the exact resolved process, reaction, and species index arrays plus a `mechanism_content_id` that covers every static and dynamic field of the prepared mechanism and rate plans. Thermochemical kinetics must still be evaluated by the chemical mechanism; the systems-biology propensity is not a substitute for those semantics.

## Telegraph transcription and count measurement

`TelegraphGeneExpressionPlan` prepares four species (`promoter_off`, `promoter_on`, `nascent`, and `mature`) and five reactions: activation, deactivation, transcription, splicing, and mature-RNA degradation.

```python
model = sb.TelegraphGeneExpressionPlan(
    1.5,  # promoter activation
    2.5,  # promoter deactivation
    8.0,  # transcription
    3.0,  # splicing
    1.0,  # mature degradation
).prepare()
initial = model.initial_state(promoter_on=False, nascent=0, mature=0)
moments = model.stationary_moments()
```

`stationary_moments` gives exact means, variances, and promoter/nascent/mature covariances for this linear Markov network. `TelegraphFitTarget` stores the five fitting observables and their uncertainty scales. `model.fitting_objective(log_rates, target)` is differentiable and works directly with `jax.grad`. Stationary moments determine rate ratios but not an overall time scale, so `identifiability_evidence` normally reports rank four for the five rate parameters. Time-resolved evidence is needed to identify the missing scale.

`CountMeasurementPlan` describes binomial molecular capture followed by independent Poisson false-positive counts. Preparation fixes the maximum observable count, making the exact convolution JIT-compatible. `log_likelihood(observed, latent_count)` reports invalid integer counts and capacity overflow explicitly. `observed_moments` propagates latent mean and variance through the same measurement model.

## Biological evidence bindings

Biological values are not silently copied into plans. A `BiologicalFact` has a namespace, unit, and exact `BiologicalReference`; a `BiologicalCondition` names the context; and `PlanFieldAssertion` names one field from a target's closed evidence-field set. `bind_biological_evidence` requires both value and declared target unit to match and ties the result to the exact prepared identity. Network time units are explicit; general propensity units are derived from changed-species, driver-species, compartment-measure, order, and time units, while telegraph rates declare `s^-1`.

Conflicting facts with the same namespaced key, conflicting conditions, two different facts targeting one field under one condition, unknown fields, and conflicting references are errors. Reserved path/key delimiters are rejected at construction so field paths and namespaced keys are injective. A value, boolean type, or unit mismatch produces `valid=False`, preserving auditable negative evidence. Repeating an identical binding produces the same `binding_id`.

## Atomic multiprocess whole-cell assembly

Whole-cell assembly shares typed scalar fields without merging the internal identities of its component networks:

1. declare `ExchangeFieldSpec` values with quantity, unit, and global reservoir semantics;
2. map every species of each prepared network with `WholeCellProcessBinding`;
3. assign each process a static substep count, minimum copy number, and regime-enforcement policy using `MultirateScheduleEntry`;
4. reserve explicit `field_capacity` and `process_capacity` in `WholeCellAssemblyPlan`;
5. prepare once and use `step` followed by `commit`.

Every process advances from the same macro-step snapshot. Its internal substeps may differ, but all process deltas are combined into one candidate. Regime evidence checks both each isolated substep path and the final coupled shared-field candidate. `require_regime_valid=True` rejects a low-copy candidate; setting it to `False` is an explicit policy waiver that retains `regime_valid=False` in the result. A commit succeeds only if the base lineage, epoch, amounts, and both ledgers still match and every required process, finite-value check, nonnegativity check, and conservation check succeeds. Otherwise amounts, source ledger, sink ledger, and epoch all remain unchanged.

Reservoir values stay fixed. Every process mapping a field must agree with its global reservoir declaration, preventing one process from treating a shared dynamic field as a chemostat. Consumption from a reservoir increments the source ledger; production into a reservoir increments the sink ledger. Boundary contributions are split per reaction before summation, so simultaneous opposing flows remain visible rather than cancelling. Conservation is checked on `dynamic_delta + sink_delta - source_delta`, which reconstructs the full closed-system change. `checkpoint(state)` computes a host identity over the assembly, lineage, epoch, values, and both cumulative ledgers, so even ledger-only differences cannot collide.

Run `python benchmarks/systems_biology.py` to measure lowering, compilation, steady execution, logical memory, compiler cost/memory evidence, and conservation for a compiled gene-expression/translation whole-cell step.

## Single-cell transcript scenarios (S1)

`phydrax.applications.systems_biology.single_cell` composes the existing telegraph,
exact jump-process, sampled-series, measurement, optimization, and UQ owners into
bounded generate → observe → fit → held-out prediction workflows. It does not
replace the compartmental process or whole-cell contracts above.

The admitted native model is a telegraph promoter with activation, deactivation,
transcription, splicing, and mature-RNA degradation. Each cell/gene path has latent
state `(promoter_on, U, S)`, where the promoter is binary, U is nascent/unspliced
RNA count, and S is mature/spliced RNA count. The mature-count conditional drift
is βU − γS. It is a generator expectation conditional on state, **not the derivative
of a sampled jump trajectory**.

### Exact schedules, finite supports, and resets

`PiecewiseConstantRates(boundaries, rates, rate_unit=..., time_unit=...)` accepts
only finite positive rates of shape `(interval, gene, 5)`, ordered activation,
deactivation, transcription, splicing β, and degradation γ. Boundaries are finite
and strictly increasing; there is one more boundary than interval. Rate units are
converted to the inverse of the exact declared runtime time unit. Values at an
interior boundary use the interval on its right.

```python
import numpy as np
from phydrax.applications.systems_biology import single_cell as sc
from phydrax.units import derived_unit, SECOND

rates = np.broadcast_to(
    np.asarray([2.0, 3.0, 12.0, 4.0, 1.5]), (2, 2, 5)
).copy()
rates[0, :, 2] = 4.0
schedule = sc.PiecewiseConstantRates(
    (0.0, 1.0, 12.0),
    rates,
    rate_unit=derived_unit("per-second", ((SECOND, -1),)),
)
segment = sc.ScenarioSegment(11, schedule, (0.0, 1.0, 4.0, 12.0))
scenario = sc.TranscriptScenario(
    tuple(sc.CellIdentity(10_000 + i, f"cell-{i}") for i in range(4)),
    (sc.GeneIdentity(1_000, "gene-0"), sc.GeneIdentity(1_001, "gene-1")),
    (segment,),
    np.zeros((4, 2, 3)),
    max_paths=8,
    max_events_per_interval=1024,
)
```

These numbers are illustrative synthetic coefficients, not measured biological
rates. `generate_transcripts(scenario, key)` runs native direct SSA on each
constant-rate interval. Exponential clocks are restarted at exogenous rate
boundaries; memorylessness makes that execution exact for the specified
piecewise-constant law. This does **not** admit smooth callable hazards or promise
the same path after schedule refinement. `transient_transcript_mean` gives the exact
affine first-moment law for one constant interval; `scheduled_transcript_mean`
composes it at every declared boundary. These moment maps are distinct from a
pathwise derivative of SSA events.

Cell, gene, and segment IDs are explicit stable nonnegative signed-int64 identities,
not array positions. `TranscriptScenario` requires unique cell/gene/segment support,
binary initial promoter and nonnegative integer U/S counts, and identical exact
runtime time units across segments. The number of cells × genes × segments cannot
exceed `max_paths`. `max_events_per_interval` bounds native event storage/execution;
an incomplete interval raises `ScenarioExecutionError` with its `solution`, and no
descendant is executed from that failed state. Exhaustion is not a valid truncated
experiment.

A root segment starts from the declared initial states. A child names an already
declared parent, begins exactly at that parent's terminal physical time, and
inherits its terminal state with newly addressed randomness. Multiple children
are counterfactual continuations: molecules are **not partitioned**, and this is
not biological cell division. `PiecewiseConstantRates.repeat(cycles)` finitely
unrolls an external protocol; it is not a cell-cycle or lineage model. Cycles in
the parent graph are not admitted.

Every interval boundary is retained in `TranscriptPath.latent`, even when omitted
from the requested saved nodes. `conditional_drift` is a separate series in
transcript counts per runtime time unit. `TranscriptExperiment.joined_series`
disconnects the edge at **every segment reset**, including parent/child joins, so
branch/reset concatenation cannot create spurious lag pairs. Selecting a physical
continuation remains a caller decision.

### Stable stochastic identity and separate measurement

Latent randomness is addressed by stable cell, gene, segment, interval, and native
event identity. `generate_transcripts(..., cell_ids=..., gene_ids=...)` selects
declared worksets without renumbering random paths. Replaying the same explicit
schedule, identities, and root key preserves stochastic addressing; changing
segment identities or interval partitioning changes that experiment.

`TranscriptCountAssay` holds two prepared `CountMeasurementPlan` channels and an
independent calibration `ReferenceArtifactManifest`. Each channel performs binomial
molecular capture followed by independent Poisson background. It does not estimate
capture from the same expression counts it will fit. The reference must admit the
intended use and declare actual uncertainty; unknown uncertainty is `None` and
cannot be silently replaced with zero to pass calibration admission.

`observe_transcripts(experiment, assay, key, gene_id=..., segment_id=...,
sample_time=...)` measures one actually saved physical snapshot per cell. The
default is the segment endpoint. An unsaved time is refused, not interpolated into
a molecular count. Assay draws are addressed by cell/gene/segment, exact physical
sample time, channel, and capture/background. They use a namespace disjoint from
SSA even if the caller supplies the same root key. Observation-capacity overflow
is refused without clipping.

`TranscriptCounts` stores one gene's measured U/S snapshots of shape `(cell, 2)`,
separate boolean validity masks, stable cell IDs, assay/source/preprocessing
identities, and explicit coordinate semantics. Active counts must be finite
nonnegative integers. Masked entries are represented separately from active
measured zeros. Snapshots are not inferred descendants or sampled latent paths:
`to_series()` has no connecting trajectory edges.

### Identifiable combinations and calibrated inference

`StationaryCountTarget.from_counts(observations, standard_errors,
equilibrium_evidence_id=...)` uses at least three complete independent U/S snapshot
pairs. Its observable order is mean(U), mean(S), var(U), var(S), cov(U,S), with
unbiased sample variances/covariance. Equilibrium must have its own justification;
neither a saved endpoint nor a pseudotime ordering proves it.

`predicted_count_moments` passes the exact latent stationary moments through the
same calibrated capture/background law, including cross-channel covariance
scaling by the product of capture probabilities. `fit_stationary_counts` fits
positive log rates with native least squares. Its objective assumes the caller's
declared independent moment-error scales. It is **not an exact count likelihood**,
and the five empirical moments can be statistically correlated; choose an
appropriate explicitly correlated formulation when that assumption is unsuitable.

Stationary observations are invariant to multiplying all five rates by the same
positive factor. They constrain combinations such as activation/γ,
deactivation/γ, transcription/γ, and β/γ, not an absolute clock. Local sensitivity
rank can be smaller still for uninformative data or assay support. The returned
`TranscriptIdentifiability` reports sensitivity, singular values, rank, free
parameter indices, and whether an independent rate clock was supplied.

`fixed_rates` maps indices 0–4 to independently calibrated positive physical rates
in inverse `rate_time_unit`; fixing any rates requires `rate_calibration` with
admitted rights and known uncertainty. A calibration manifest with no fixed rates
does not identify a clock and is refused. At least one rate must remain free.
Fixing an arbitrary gauge is not biological timing calibration.

Inspect `TranscriptFit.result.successful` as well as identifiability. Only a
successful locally identifiable fit receives `free_log_rate_covariance` and
linearized `count_prediction_uq`; otherwise they remain `None`. The covariance is
first-order **conditional fit covariance**, not a posterior. Fixed-rate and assay
coefficients are conditioned upon; their uncertainty and model discrepancy are
not automatically propagated merely because manifests retain them.

`fit.held_out_residuals(target)` requires disjoint cell identities with the same
gene and assay, and divides moment prediction errors by held-out moment standard
errors. These residuals are not posterior predictive probabilities or automatic
experimental acceptance. Retain experimental split design, calibration uncertainty,
equilibrium evidence, and an independently declared scientific criterion.

### Count-derived velocity and explicit external arrays

`predict_transcript_velocity(fit, observations)` requires a successful locally
identifiable fit, an independent physical rate clock, matching gene/assay, and
positive capture in both channels. It subtracts background and divides by capture,
then estimates βU − γS solely from those corrected measured counts. Invalid pairs
remain masked/NaN. Negative corrected estimates are retained because clipping would
introduce bias.

`TranscriptVelocityEvidence` records observation/fit identities, inverse-time
unit, estimator and preprocessing, and the conditional/non-posterior uncertainty
boundary. It is neither a latent-velocity posterior nor a sampled-path derivative,
lineage, energy landscape, or proof of a biological clock. Importantly, stored
`TranscriptPath.conditional_drift` is generator truth, whereas this estimator uses
independent measured counts.

`import_transcript_arrays` accepts caller-extracted raw U/S columns, explicit
gene/cell mapping, validity masks, source manifest, assay/preprocessing identities,
and coordinate meaning. It preserves the raw arrays alongside native counts and
an `AdapterReport`; it does not import an AnnData/scVelo provider or infer which
layers were raw. Active noninteger/log-normalized expression is refused by count
admission; callers must not relabel transformed data that happen to be integral.
Requested commercial, training, redistribution, and export rights are checked on
the source artifact.

Coordinates must be `physical_time`, `pseudotime`, or `none`. Only physical time may
carry an exact time unit. Pseudotime is not promoted into physical seconds, and
snapshot order never supplies a lineage. `import_velocity_field` separately admits
a `(cell, representation_dimension)` external field with estimator, preprocessing,
representation, and uncertainty identities, aligned validity, source rights, and
optional standard errors. `standard_errors=None` means unreported uncertainty.
Retaining an embedded field losslessly does not establish physical-time, energy,
or velocity accuracy.

### Evidence and scientific completion gates

Run the [single-cell cookbook](cookbook/single_cell_transcripts.md) from the
repository root:

```bash
python benchmarks/single_cell_transcripts.py --cells 64 --genes 2
```

The recorded synthetic smoke executed 64 cells × 2 genes, 128 paths, and 33,613
events. Maximum absolute held-out standardized moment residual was
`2.4761358858990175`; sensitivity rank was one for the one fitted free rate, with
four independently specified synthetic rates fixed. The workflow also distinguishes
finite-time moment bias, latent sampling error, assay noise, and count-derived
drift error. These are synthetic mechanics observations, not experimental accuracy.

Rights-cleared experimental counts, independent assay and physical-rate calibration
with uncertainty, justified stationary or time-resolved evidence, and held-out
biological qualification remain scientific prerequisites. This implementation
does not claim smooth-hazard SSA, biological division, physical pseudotime, or
experimentally validated timescales. See the
[single-cell API](api/advanced_biophysics.md#single-cell-transcripts) and
[biophysical source dispositions](biophysical_sources.md).
