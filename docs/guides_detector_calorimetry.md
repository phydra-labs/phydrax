# Detector and calorimeter production

`phydrax.applications.detector` owns typed detector conditions, bounded transport records, sensitive hits, digits, fixed-association tracking, calorimeter response, and reconstructed-object boundaries. Detailed shower transport and full combinatorial reconstruction remain pinned external capabilities.

## Conditions and records

`DetectorConditions` binds geometry, material, field, alignment, calibration, units, and a half-open validity interval. Transport tracks, truth steps, sensitive hits, digits, track measurements, fitted tracks, clusters, and reconstructed particles use distinct types and stable IDs.

`ChargedPropagationPlan` is a bounded constant-field reference. It reuses the `RelativisticPushPlan` substrate (Boris in `PIC_CODE_RELATIVITY` unless a pusher is supplied) and may apply a declared deterministic mean energy loss. It makes no shower, secondary-production, fluctuating-loss, or general navigation claim. `ChargedPropagationResult.active_history` marks, per step, whether that step committed a finite state of an existing track; a track stopped by the mean loss is active at the stopping step and inactive afterwards.

`ChargedPropagationPlan(..., radiation_reaction=RadiationReactionPlan(...), radiation_key=None)` applies classical or quantum-corrected radiation reaction (see [Particle-in-cell methods](guides_particle_in_cell.md#radiation-reaction)) after every push, before the drift and the mean loss. The constant field has exactly zero gradients and time derivatives, so the full and reduced Landau–Lifshitz models coincide. The reaction's scale must share the pusher's units and speed of light, and every active track must carry the reaction species' charge and mass (checked on the host). The stochastic Fokker–Planck model requires `radiation_key`; its Wiener increments are addressed by step and track identity `(event_id, track_id)`. `ChargedPropagationResult.radiated_energy_history[step, event, track]` is the energy radiated in each committed step (equal to the kinetic energy the reaction removed; zero without radiation reaction) and `radiation_flags_history` holds `RadiationReactionFlag` bits; a step whose reaction is unsupported (for example `χ` above the declared bound) is not committed and the track is not accepted.

`SensitiveHitPlan` maps detector elements to channels. `DigitizationPlan` composes calibration, noise, crosstalk, ADC quantization, saturation, thresholding, and zero suppression. Event-ID-folded noise is batching independent. Derivatives are invalid through random draws, quantization, saturation, thresholds, and changing channel support.

`TrackFitPlan` fits only caller-fixed measurement associations. It is not seeding, combinatorial track finding, ambiguity resolution, or vertexing. Those remain provider capabilities such as ACTS.

## Radiation from propagated tracks

`charged_trajectory(plan, tracks, result, scale)` turns `result = propagate_charged_tracks(plan, tracks)` into a `phydrax.electromagnetics.ChargedTrajectory` for far-field spectra (see [Charged-particle radiation](guides_charged_particle_radiation.md)):

- Lane `e * track_capacity + k` is track `k` of event `e`, with identity words `(event_id, track_id)`. Event IDs are checked on the host and must lie in `[0, 2³²)`; int32 track IDs are reinterpreted bit-exactly as uint32.
- Sample 0 is the initial state in `tracks` at time 0, and sample `s` is the state after `s` steps at `s * step_size`, on every lane.
- A sample is active when it is a committed state of an existing, valid track. Inactive samples hold the last committed position and have zero proper velocity, so every lane stays causal.
- Charges are the per-particle charges the pusher uses, read in `scale.charge_unit`. Each lane has multiplicity one. Proper velocities are `u = p c² / E₀`.
- The pusher's relativity scale must share the dimensional scale and the exact speed of light of `scale`. Otherwise the call is refused, as are detector kinematics in any dtype other than float64.

For a uniform axial field `B`, the helix of a track with Lorentz factor `γ` radiates its line at `ω = |q| B / (γ m)` to a transverse observer, and at `ω / (1 − β∥)` along the field.

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
