# Dark showers, hadronization, bound states and cascades

Full Standard Model shower/hadronization remains a pinned external-provider profile.
Native execution is restricted to explicitly declared dark-sector models.

## Dark shower epochs

`DarkShowerEpochPlan` binds ordering variable, running coupling, splitting channels,
certified veto envelopes, recoil, dark-color rules, cutoff and fixed epoch capacity.
Every shower/continuation execution supplies a caller-owned `draw_id` and the exact
species revision; random addressing is never inferred from call order or a display
name. Sudakov/no-emission probabilities, rejected trials, branch ordering,
four-momentum and charge/color evidence are explicit. Deferred branch work enters the
durable dark-sector frontier.

## Matching and provider execution

`ProviderExecutionChain` and `record_provider_execution` bind hard-process, matching,
shower and hadronization provider revisions, exact unit/frame realization, capability
manifests and named signed weights. `assign_exclusive_matching_bins` produces one
exclusive assignment. Metadata strings alone are never treated as matching execution.

## Native dark hadronization

`DarkStringFragmentationPlan` and `DarkClusterHadronizationPlan` require a named dark
model, exact species revision, spectrum, tune, charges and
fragmentation/fission/decay policy. Fragmentation, fission, hadronization, and decay
runtime calls require an explicit `draw_id`, so retries and continuation retain
semantic RNG identity. Every chain or cluster result reports normalized
probabilities, lineage, exact charge and four-momentum evidence. These profiles
explicitly refuse generic-QCD claims.

## Bound states

`DarkBoundStateSpectrum` and `RadiativeCapturePlan` model named bound levels,
radiative capture and photo-dissociation with pointwise and thermal detailed balance.
The capture plan requires a typed cross-section area unit and an independent
`photo_dissociation_coefficient`; reverse kinetics is not inferred by reusing the
capture coefficient. Bound-state formation is a typed reaction profile, not an
implicit hadronization option.

## Unbounded cascades

`DarkDecayCascadePlan` executes a finite resident DAG epoch with one decay owner per
unstable species, exact species revision, explicit `draw_id`, proper-time
prompt/delayed paths, fixed product/work capacities and atomic rollback.
`DarkDecayDurableFrontier` materializes remaining work into the event graph, allowing
arbitrarily deep cascades over committed epochs without unbounded device memory.
