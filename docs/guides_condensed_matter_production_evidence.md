# Condensed-matter production evidence

Phydrax does not define an umbrella condensed-matter physics object or support tuple. The production-evidence surface composes the existing periodic, lattice/phonon, Green/embedding, quantum-lattice, spectroscopy, magnetism/superconductivity, magnetic-resonance, semiconductor, soft-matter, and bounded frontier owners without moving their scientific kernels. Closure is the conjunction of exact owner profile/support dependencies in a global `ReleaseIndex`.

## Maturity and authority

`condensed_matter_candidate_profiles()` returns exact implemented production-baseline candidates in stable content-address order: periodic, lattice/phonon/QHA/RTA, impurity/DMFT/continuation, quantum lattice, spectroscopy, magnetism/superconductivity, magnetic resonance, semiconductor, and soft matter. `condensed_matter_candidate_closure()` derives one `SupportDependency` for every exact baseline profile/support pair. Its `CondensedMatterClosureLedger` has no `SupportTuple`, release flag, gates, or signature and its record states `release_claim: false`.

`require_condensed_matter_closure(index, ledger, trust_policy, at_time=...)` is the only closure admission helper. It requires the supplied `ReleaseIndex` to be trusted and then calls the normal exact-profile admission path for every ledger dependency. A missing profile, changed support tuple, unreleased profile, missing or stale gate, failed transitive dependency, or untrusted index refuses the whole conjunction. A capacity in a plan or benchmark is only a measurement/input policy; it is not a release claim.

Because gate evidence and the release flag are part of `CapabilityProfile`
identity, the candidate ledger is intentionally not reused as a released-profile
ledger. A release decision constructs `CondensedMatterClosureLedger.from_profiles`
from the exact profiles being assessed in the index; the unchanged support-tuple
IDs preserve scientific scope while the profile IDs bind the actual evidence.

Candidate and released support content is identical. Promotion changes profile evidence and maturity, not the `SupportTuple` capability or attributes. The library ships no authority key, fabricated signature, release index, or signed condensed-matter evidence.

The inventory composes these factories and existing constants:

- `periodic_candidate_profiles`, `lattice_material_candidate_profiles`, and
  `green_embedding_candidate_profiles`;
- `quantum_lattice_candidate_profiles`,
  `material_spectroscopy_candidate_profiles`,
  `magnetism_candidate_profiles`, `superconductivity_candidate_profiles`, and
  `magnetic_resonance_candidate_profiles`;
- `semiconductor_candidate_profiles` and `soft_matter_candidate_profiles`.

`condensed_matter_frontier_candidate_profiles()` separately inventories
`candidate_lattice_profiles` plus existing fRG, lattice-parquet, sign-free
CT-INT, low-order diagram-MC, fermionic second-Born, and controlled-sign
candidates. Frontier entries never enter `condensed_matter_candidate_closure()`.

Every added support tuple describes executable code already present. Frontier
capacity IDs remain code-admission inputs and do not assert qualified scale.

## Leakage-controlled campaigns

`condensed_matter_candidate_campaigns()` returns the production-baseline `CondensedMatterCampaignAggregation`. `condensed_matter_frontier_candidate_campaigns()` returns a separate frontier inventory. Both reference owner campaigns or application-leaf declarations rather than flattening them into one broad scientific campaign, and every referenced `ScientificCampaign` keeps its own cases, criteria, calibration membership, and locked-evaluation membership.

The aggregation additionally refuses:

- duplicate campaign or case identities;
- reuse of an independent unit, preparation, or batch across calibration and locked roles, including across owners;
- a value that is not an owner `ScientificCampaign`.

Campaign declarations and successful campaign runs remain prospective candidate evidence. They do not release a profile and cannot cross-qualify another approximation.

## Immutable array artifacts

`ArrayArtifactProvenance` binds a producer, source IDs, profile IDs, and unit IDs. The generic lifecycle codec stores only pickle-free numeric PyTree leaves in `ArrayArchive`; static structure and executable behavior are never serialized. Every archive binds:

- one concrete result type and artifact kind;
- exact caller-supplied source, profile, producer, and unit provenance;
- owner-selected structure IDs such as cell, basis, plan, route, convention, or result IDs;
- the PyTree paths, array shapes and dtypes, complete array digest, and artifact content address.

Restore requires a caller-prepared template of the same concrete type and exact structure/provenance. Static plans, providers, callbacks, solvers, and callables come from that caller-owned template, never from archive bytes. Changed structure, shape, dtype, provenance, inventory, checksum, or content identity refuses restoration.

### Periodic chemistry

`write_periodic_artifact_archive` and `read_periodic_artifact_archive` cover the canonical periodic translation-family state, pencil evaluation, spectrum result, second- and third-order force constants, finite-displacement IFC2 result, and single-site DMFT result. These codecs retain raw family/IFC values, masks, corrected values, residuals, eigenspaces, Green/self-energy/bath arrays, success evidence, units, source IDs, and plan/result identities. They do not archive a force evaluator or impurity provider.

### Quantum lattices and response

`write_quantum_lattice_artifact_archive` and `read_quantum_lattice_artifact_archive` cover direct fixed-cardinality fermion, fixed-projection spin, and fixed-number boson bases plus a prepared matrix-free sector operator. Direct rank/unrank tables, charge-map certification, coefficients, and basis/operator IDs are retained without an ambient dense basis.

`write_quantum_result_archive` and `read_quantum_result_archive` cover TPQ and zero-/finite-temperature response results. Raw probes, thermal vectors, norm weights, correlations, estimator errors, shifted-solve residuals, moments, positivity, KMS evidence, validity masks, source/target sector IDs, and probe IDs remain distinct.

### Semiconductor detector results

`write_detector_artifact_archive` and `read_detector_artifact_archive` cover bias electrostatics, zero-charge weighting fields, and prescribed Shockley–Ramo response. Potentials, fields, capacitance, electrode/route/sample masks, induced charge, interval current, endpoint closure residuals, resource observations, plan/trajectory/route IDs, and the sign convention are retained. The archive never advances a carrier or upgrades a prescribed trajectory to detector dynamics.

## Orchestration and example

The public smoke example executes an actual periodic translation family, archives and restores its state, re-evaluates it, and inventories the candidate closure and owner campaigns:

```text
python examples/condensed_matter_production_evidence.py
```

The orchestrator invokes existing owner tools directly:

```text
python tools/cm_production_closure.py --group qualification
python tools/cm_production_closure.py --group benchmark
python tools/cm_production_closure.py --group all
```

Each component writes its normal output to the terminal. The final JSON lists every component and its exact return code. The orchestrator runs all selected components and exits nonzero if any component fails; it neither rewrites a failure as success nor turns benchmark output into release evidence.

## Explicit nonclaims

This evidence layer does not claim a production release, transferable material accuracy, a first-principles provider, arbitrary system size, GPU/distributed parity, float32 support, automatic backend selection, native Wannierization, an interacting approximation outside its exact profile, a sign-problem solution, carrier collection/trapping/avalanche dynamics, or coverage of functionality absent from an implemented owner. Baseline soft-matter and magnetic-resonance entries remain exact unreleased profiles with their own nonclaims. Frontier candidates and campaigns remain in their separate inventory and do not become closure dependencies or cross-qualify the baseline. Periodic electronic VMC and any other unlisted vertical remain separate until they expose an exact owner profile.
