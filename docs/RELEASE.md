# Release process

This document defines the technical release process for PhydraX, including the cardiovascular G0–G7 overlay. It does not grant commercial rights or authorize medical, clinical, or regulated-device use. The release operator must comply with `LICENSE`, notices, third-party terms, data rights, and any separate written agreement.

## Release invariants

A release is fail-closed and content-addressed. The process must preserve:

- plan → prepare → fixed-shape state → candidate/evidence → commit/checkpoint;
- explicit stable IDs for source, dependencies, build, plans, support tuple, artifacts, lifecycle records, evidence, candidate, decision, and final profile;
- exact tuple matching with no wildcard, nearest-profile, or fallback behavior;
- fixed-topology differentiation boundaries and explicit invalid-derivative outcomes;
- units, axes, sign conventions, support associations, and reference configurations;
- immutable run, result, checkpoint, evidence, and decision records;
- independent review and a release decision that cannot override evidence.

Do not release from an uncommitted build, an unpinned dependency set, an incomplete run, an expired record, or an altered dossier.

## Roles and separation of duties

Record five non-empty and mutually distinct identities:

1. **author** — assembles the implementation and evidence;
2. **technical reviewer** — reviews intended use, code verification, and derivative validity;
3. **validation reviewer** — reviews solution verification, validation/UQ, and dossier completeness;
4. **security reviewer** — reviews provenance, supply chain, privacy, security, and operations;
5. **release approver** — makes the final decision only after candidate evaluation.

The release approver does not author or review G0–G7 evidence. Each gate reviewer signs the canonical gate record, common dossier ID, reviewer ID, and complete evidence-ID tuple. G7 is the validation review of dossier completeness; it is not the final release decision.

## Cardiovascular support profile

Create exactly one `CardiovascularCommercialSupportProfile` for each exact `SupportTuple`. The only allowed boundary is local, isolated, non-PHI, non-networked, non-regulated engineering use. The matching claims-matrix decision must remain a technical candidate and must exclude diagnosis, treatment, clinical decision support, and regulated medical-device use.

A profile cannot weaken the required privacy, security, resource, artifact, or non-claim prerequisites. A profile and its final generic `CapabilityProfile` are distinct records: the former binds cardiovascular policy; the latter is the provider-neutral release registry entry.

## G0–G7 checklist

### G0 — intended use

- exact support tuple and technical scope approved;
- claims-matrix decision present;
- prohibited uses explicit in both use policy and claim decision;
- four current signed non-claims present and signature-verified;
- no commercial grant, clinical claim, or regulated-device claim.

### G1 — code verification

- equation-to-code traceability reviewed;
- deterministic verification cases pass under the exact build and tuple;
- failure paths and status propagation exercised;
- gate cites at least one completed lifecycle run and its case-bound `CardiovascularExecutionManifest`.

### G2 — solution verification

- spatial, temporal, coupling, and solver convergence assessed as applicable;
- conservation, residual, and tolerance evidence retained;
- reference solutions are versioned and rights-cleared;
- gate cites at least one completed lifecycle run and its case-bound `CardiovascularExecutionManifest`.

### G3 — validation and uncertainty quantification

- validation domain and limits declared;
- observation operators, likelihoods, noise, uncertainty, and calibration leakage reviewed;
- validation data rights and permitted use recorded;
- no patient or PHI data enters the supplied support profile;
- gate cites at least one completed lifecycle run and its case-bound `CardiovascularExecutionManifest`.

### G4 — derivative validity

- fixed topology, event, contact, remeshing, and branch boundaries identified;
- primal and transpose residual criteria reviewed where applicable;
- invalid or unsupported sensitivities fail closed rather than returning an implied valid gradient;
- gate cites at least one completed lifecycle run and its case-bound `CardiovascularExecutionManifest`.

### G5 — provenance and supply chain

G5 must cite one current `CardiovascularArtifactReference` for every required kind:

- SBOM;
- build provenance;
- separate commercial-licence authorization;
- notice audit;
- data-rights determination;
- supply-chain attestation.

Each reference wraps a real `ArtifactManifest`, has an issued/expiry interval, and names its dependency reference IDs. The dependency graph must be complete and current. The supply-chain attestation should depend on the exact SBOM and build-provenance references. Review optional backends actually included by the tuple, not merely base dependencies.

### G6 — quality and operations

- resource ceilings reviewed and exercised;
- privacy policy remains non-PHI, local, non-telemetry, and non-transfer;
- security policy requires isolation, dependency locking, signed evidence, and reviewer allow-lists;
- recovery, checkpoint, archive integrity, diagnostics, support, and incident paths reviewed;
- gate cites all resource, privacy, and security policy IDs.

### G7 — independent release review

- all G0–G6 records and dependencies are present;
- reviewers are the role owners specified by policy;
- deviations are closed; an open deviation is a failed gate;
- evidence is current under the release trust policy;
- gate cites the immutable independent-role record.

## Candidate evaluation

Call `evaluate_cardiovascular_release_candidate` with:

- the exact support profile;
- a `CardiovascularQualificationBundle` containing the non-identifying `CardiovascularCaseManifest`, case-bound execution manifests, unique reviewer-signed G0–G7 records sharing one dossier ID, artifact references, completed lifecycle records, signed non-claims, and roles;
- the approved `ReleaseTrustPolicy`;
- `CardiovascularSignatureVerifier` instances keyed by every trusted gate, non-claim, and release-decision signer ID;
- the evaluation timestamp;
- a case manifest whose support-profile, build, SBOM, commercial-licence, data-rights, and explicit non-PHI bindings match the dossier;
- every referenced dependency `CapabilityProfile`.

The returned candidate contains deterministic blockers in evaluation order. Missing prerequisite records are reported rather than inferred. Duplicate gates, artifact kinds, tuple decisions, lifecycle IDs, exclusions, or independent role identities are rejected at construction.

A candidate is not a release. Its `commercial_ready` property always returns false, even when `qualified` is true. Preserve the candidate ID and earliest evidence expiry.

## Independent release decision

After a qualified candidate exists, the named release approver calls `make_cardiovascular_release_decision` with the approved `ReleaseSigner`. The decision payload is independently authenticated and later checked through the approver's `CardiovascularSignatureVerifier`. Approval is rejected when:

- the candidate has any blocker;
- the approver identity differs from the role record;
- the decision predates evaluation;
- any evidence has expired.

A refusal is a valid immutable decision. Do not alter a refusal; create a new evidence bundle, candidate, and decision after remediation.

Call `assess_cardiovascular_release` with the candidate, signed decision, approved trust policy, and verifier mapping. It verifies the approver signature without re-signing, reapplies trust at decision time, and rejects rejected or stale evidence, mismatched candidates, mismatched approvers, stale decisions, and non-approval. Only a blocker-free assessment marks its generic capability profile released. The assessment still states that it grants no commercial licence and makes no regulated-device claim.

## Repository qualification tool

Run:

```console
python tools/cardiovascular_release_qualification.py
```

The tool audits regular, non-empty on-disk records for the repository licence and notice; six artifact kinds; eight reviewer-signed gate evidence files; four signed non-claims; and the separately signed decision. For signed JSON it also requires signer and algorithm fields, hexadecimal signature encoding, gate dossier/reviewer/evidence bindings, and reviewer/signer equality. It records SHA-256 for every present file, prints deterministic JSON, and exits 2 while preflight blockers exist.

Expected default release records are below `release/cardiovascular/`. Sealed records may be supplied with `--artifact KIND=PATH`, `--gate-evidence-directory`, `--non-claim-directory`, and `--release-decision`. The tool does not mutate the dossier.

Preflight checks file identity and presence only. It deliberately never sets `commercial_ready` true. After preflight, the typed evaluation must reconstruct and content-verify records, apply trust and freshness policy, evaluate the candidate, and bind the independent decision.

### Derived artifact build

`--build-artifacts DIRECTORY` resolves every package and dependency edge in
`uv.lock`. The emitted SPDX 2.3 document carries package source/download
locations, all locked SHA-256 checksums, licence conclusions, a document
`DESCRIBES` relationship, and package `DEPENDS_ON` relationships. The
CycloneDX 1.6 document carries corresponding hashes, licences, and a complete
dependency graph. Build provenance hashes each exact wheel, sdist, or
container supplied through repeatable `--distribution-artifact KIND=PATH`
arguments. The build also emits a dependency-complete supply-chain evidence
manifest, notice audit, SHA-256 artifact manifest, and explicitly unsigned
G0–G7 dossier.

The builder accepts only externally produced release authority and evidence:
`--commercial-license-record`, `--data-rights-record`, `--signer-record`,
`--verifier-record`, `--license-report`, `--vulnerability-report`, and
`--supply-chain-attestation`. Each JSON envelope has `schema_version: 1` and
binds its expected kind, exact source commit and lock SHA-256, signer identity,
signature algorithm, and hexadecimal signature. Scanner records must identify
their tool/version and cover every locked package and hash. Licence results
must conclude and declare a non-`NOASSERTION` licence and copyright text.
Vulnerability results must explicitly pass with an empty vulnerability list.
The verifier record must bind every signed external record by SHA-256, and the
attestation must bind the lock, authority records, scan reports, signer
record, and supplied distributions.

Any absent package, unresolved graph edge, missing source or hash,
`NOASSERTION`, incomplete or non-passing scan, stale source/lock binding,
missing signature metadata, or mismatched verified/attested subject is a
blocker and leaves `g5_evidence_ready` false. The builder hashes and references
external records but never creates commercial licence authority, data rights,
scanner results, signatures, verification, attestations, notices, or release
approval. PNPL and the absent `LICENSES/SING-MIT.txt` and
`LICENSES/ASDEX-MIT.txt` files remain explicit blockers in the current tree.
Even a complete build only supplies evidence for G5: `commercial_ready`
remains false until the typed evaluation and independent release decision.

## Current source-distribution status

The current repository `LICENSE` is the Phydra Non-Production License and explicitly requires a separate licence for commercial or production use. The source distribution does not contain the sealed cardiovascular commercial-licence authorization, signed G0–G7 dossier, signed non-claims, or supply-chain release attestations expected by the preflight tool. Therefore the current source distribution is **not commercial-ready**. This is an intentional, deterministic refusal, not an incomplete implied approval.

Do not add synthetic attestations, placeholder signatures, empty records, unchecked fallback evidence, or a hard-coded pass to change that result.

## Publication and post-release

Before publishing an authorized release:

1. verify the final source commit, clean build environment, dependency lock, SBOM, and build provenance;
2. verify licence authority, root and third-party notices, and data/model rights with the responsible owners;
3. verify every signed record and trust-policy identity from sealed storage;
4. archive the exact candidate, decision, capability profile, and all transitive artifacts;
5. publish checksums and only the artifacts authorized for distribution;
6. retain revocation and incident-response contacts;
7. invalidate the profile if a signer, artifact, dependency, licence, data right, security finding, validation limit, or material release assumption changes.

Any post-release change to the exact tuple or evidence creates a new candidate. Never edit a released record in place.
