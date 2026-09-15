# Detector and calorimeter production

`phydrax.applications.detector` owns typed detector conditions, bounded transport records, sensitive hits, digits, fixed-association tracking, calorimeter response, and reconstructed-object boundaries. Detailed shower transport and full combinatorial reconstruction remain pinned external capabilities.

## Conditions and records

`DetectorConditions` binds geometry, material, field, alignment, calibration, units, and a half-open validity interval. Transport tracks, truth steps, sensitive hits, digits, track measurements, fitted tracks, clusters, and reconstructed particles use distinct types and stable IDs.

`ChargedPropagationPlan` is a bounded constant-field reference. It reuses the relativistic Boris substrate and may apply a declared deterministic mean energy loss. It makes no shower, secondary-production, fluctuating-loss, or general navigation claim.

`SensitiveHitPlan` maps detector elements to channels. `DigitizationPlan` composes calibration, noise, crosstalk, ADC quantization, saturation, thresholding, and zero suppression. Event-ID-folded noise is batching independent. Derivatives are invalid through random draws, quantization, saturation, thresholds, and changing channel support.

`TrackFitPlan` fits only caller-fixed measurement associations. It is not seeding, combinatorial track finding, ambiguity resolution, or vertexing. Those remain provider capabilities such as ACTS.

## Calorimeter geometry and truth

`CalorimeterGeometry` is a heterogeneous cell vector with stable cell, channel, layer, subdetector, material, and readout identities; cell centroids and volumes; active/dead masks; and sparse adjacency. An image is an adapter-specific view, not the canonical representation.

`route_calorimeter_hits` retains the full known deposit ledger:

```text
source deposit = routed cells + outside + unmapped + dead + rejected
```

Incident-energy closure is checked separately when leakage is known. Unreported leakage remains unknown rather than becoming zero.

`CalorimeterResponsePlan` applies cell gain, noise, sparse-compatible crosstalk, ADC conversion, saturation, threshold, and dead-cell suppression. `CalorimeterClusteringPlan` is a fixed reference partition; dynamic production clustering and particle flow remain provider-owned.

## Governed fast simulation

`prepare_calorimeter_corpus` requires immutable rights manifests, exact source lineage, fixed geometry, visible/leakage/dead/outside/unmapped ledgers, and group-disjoint train/validation/test partitions. Identical showers cannot cross partitions.

`ConditionalCalorimeterVelocity` uses one shared cell network plus sparse neighbor aggregation over the declared geometry. `fit_calorimeter_flow` reuses Phydrax `FlowMatchingTerm` and `FunctionalSolver`; `prepare_calorimeter_sampler` reuses the continuous-system and Diffrax evolution substrate. Sampling refuses conditions outside the training envelope and nonfinite or negative decoded cell energies.

CaloChallenge profiles are the primary broad benchmark; CaloGAN is a historical profile. Neither alone qualifies a named detector deployment.

## Qualification

Required evidence includes energy response/resolution, layer fractions, occupancy, zero structure, depth, transverse width, hottest-cell fractions, correlations, tails, energy-ledger closure, reconstruction-level response, OOD behavior, throughput, and memory. A classifier score is diagnostic only. Qualification is specific to geometry, incident species and energy range, conditions, observables, dtype, device, and fallback policy.
