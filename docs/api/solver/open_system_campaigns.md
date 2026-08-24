# Precision-aware open-system campaigns

Open-system production status is derived from verified campaign
artifacts. Graduation accepts `VerifiedOpenSystemCampaign` values, never caller
Booleans. A campaign requires execution, approximation, physicality,
work/capacity, native precision, archive integrity, and independent
reproduction evidence.

## Artifact verification

Artifacts serialize the complete campaign record: approximation axes and
thresholds, physicality properties and tolerances, replay, capacity, work,
unsupported claims, precision contracts, and solver/reference arrays. Request,
resolution, and evidence identities are linked during deserialization.
Independent reproduction compares the complete record and exact arrays.

## Precision and replay

Every artifact stores linked `PrecisionRequest`, `PrecisionResolution`, and
nested `PrecisionEvidenceEnvelope` records. Stochastic campaigns preserve
semantic addresses and variates; derived event and observable differences use
campaign-declared tolerances.

## Campaign matrix

The current matrix covers Gaussian, explicit-vector trajectory, MPS trajectory,
LPDO, HEOM, causal memory refit, sequential-process tomography, and connected
VMC neural trajectories.

Process campaigns generate a seeded informationally complete intervention/effect
design for training and a different seeded design for held-out evaluation.
Held-out errors compare fitted probabilities with observed count/trial
frequencies; duplicated settings are rejected. Recovery must show nonzero
pre-fit error and a post-fit error ratio below one before promotion.
Each experiment caches a canonical selected-outcome Choi/effect fingerprint;
design-disjointness checks are set intersections rather than pairwise scans.

Missing map physicality, representation closure, refinement, held-out evidence,
or archive qualification remains `not-promoted`.

Campaign definitions, artifact verification, and graduation are developer
qualification tooling under `tools.open_system_campaigns`; they are
intentionally absent from `phydrax.solver`. Run the full matrix with:

```text
python -m tools.open_system_campaign_matrix --output-directory <directory>
```

## Permanent stop claims

Graduation never claims exact general interacting bosonic dynamics, universal
memory-kernel complete positivity, exact MPS/neural unraveling without closure,
infinite-depth HEOM convergence, unique process recovery outside memory gauge,
physical arbitrary-MPO compression, or global steady-state uniqueness from a
finite stationarity window.
