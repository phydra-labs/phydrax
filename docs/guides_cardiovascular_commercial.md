# Cardiovascular commercial qualification

The cardiovascular commercial layer is a fail-closed **technical release-readiness** system. It does not grant a commercial licence, certify a medical device, or authorize diagnosis, treatment, or clinical decision support. The repository is distributed under the licence in `LICENSE`; where that licence requires a separate commercial agreement, a passing technical assessment does not replace it.

## Supported boundary

A `CardiovascularCommercialSupportProfile` describes exactly one `SupportTuple`. There is no partial matching, wildcard route, or fallback to a nearby profile. The only accepted commercial-support boundary is:

| Coordinate | Required value |
| --- | --- |
| `capability` | `cardiovascular.workflow` |
| `deployment` | `local` |
| `data_classification` | `non-phi` |
| `regulated_device` | `false` |

The tuple may contain additional exact coordinates such as fidelity route, precision, anatomy representation, solver policy, and observation configuration. Changing any coordinate creates a different tuple and requires a separate profile and evidence bundle.

The supplied profile also requires isolated local execution, a pinned dependency lock, no network access, no telemetry, no external data transfer, positive hard resource ceilings, and the categorical exclusion of:

- clinical decision support;
- diagnosis;
- regulated medical-device use;
- treatment.

These exclusions are authenticated as signed non-claim records. They are not disclaimers that turn an unsupported use into a supported use.

## Claims matrix

`CardiovascularClaimsMatrix` permits one and only one `CardiovascularClaimDecision` per exact tuple. `TECHNICAL_SUPPORT_CANDIDATE` means that the tuple may enter qualification. It is not a commercial, performance, clinical, safety, or regulatory claim. `NOT_SUPPORTED` and `PROHIBITED` remain fail-closed dispositions.

Each candidate decision names its technical scope, evidence IDs, rationale, and excluded uses. A commercial support profile is invalid if its exact decision omits a required medical-use exclusion.

## G0–G7 gates

Every candidate must contain every gate exactly once. Missing, failed, deviated, stale, untrusted, incorrectly reviewed, or unresolved evidence is a blocker.

| Gate | Required conclusion | Principal records |
| --- | --- | --- |
| G0 intended use | Exact tuple, technical scope, and prohibited uses are explicit | claims matrix, use policy, every signed non-claim |
| G1 code verification | Implemented equations and code paths satisfy their verification plan | completed lifecycle run and verification artifacts |
| G2 solution verification | Discretization and solver errors satisfy declared tolerances | completed lifecycle run and convergence evidence |
| G3 validation and UQ | Validation domain, uncertainty, and data rights are acceptable | completed lifecycle run and validation evidence |
| G4 derivative validity | Fixed-topology derivative boundaries and failure semantics are verified | completed lifecycle run and derivative evidence |
| G5 provenance and supply chain | Build, dependencies, licences, notices, and data rights are traceable | all required artifact references |
| G6 quality and operations | Resource, privacy, security, recovery, and support policies are reviewed | immutable policy records |
| G7 independent release review | An independent reviewer confirms dossier completeness | independent role record |

G0, G1, and G4 are reviewed and signed by the technical reviewer; G2, G3, and G7 by the validation reviewer; G5 and G6 by the security reviewer. Every signature binds the common dossier ID, reviewer, complete evidence-ID tuple, and canonical gate record. The evidence author and final release approver are different people. All five role identities are mutually distinct.

## Evidence and artifact references

The commercial layer references existing owners instead of storing payloads or introducing another archive:

- `phydrax.artifacts.ArtifactManifest` owns artifact identity and SHA-256 metadata;
- `CardiovascularCaseManifest` owns the non-identifying case, support-profile, release, build, SBOM, licence, and data-rights binding;
- `phydrax.lifecycle.RunRecord` owns immutable execution lifecycle state;
- `CardiovascularExecutionManifest` owns the exact case, topology, numeric revision, solver, precision, backend, route, and fixed-capacity execution identity;
- `phydrax.qualification.ReleaseGateEvidence` owns gate outcome, citations, reviewer, deviations, and validity interval;
- `phydrax.qualification.CapabilityProfile` owns the final exact-tuple capability record;
- `phydrax.qualification.ReleaseSigner` and `ReleaseTrustPolicy` own signing and trust primitives.

`CardiovascularArtifactReference` adds only the cardiovascular artifact kind, validity interval, and reference dependencies. `CardiovascularArtifactSet` rejects duplicate kinds and reports absent or stale dependencies. G5 must cite the reference ID for every required artifact kind:

1. SBOM;
2. build provenance;
3. separate commercial-licence authorization;
4. notice audit;
5. data-rights determination;
6. supply-chain attestation.

A filename, package metadata entry, or unchecked digest is not evidence. Each reference points to a real `ArtifactManifest`. The supply-chain attestation should depend explicitly on the SBOM and build-provenance reference IDs. The case manifest must cite the exact build, SBOM, commercial-licence, and data-rights artifact IDs and declare `data_classification=non-phi`. Every lifecycle run used by G1–G4 must bind a cardiovascular execution manifest for that case, and each of those gates cites both records. Gate evidence is accepted only through the caller's `ReleaseTrustPolicy` at evaluation and again at decision time. The final decision must occur before the earliest gate, artifact, dependency, or non-claim expiry.

## Signed non-claims

Issue each required exclusion with `CardiovascularSignedNonClaim.issue`. The payload binds the exact support-tuple ID, exclusion, statement, author, validity interval, signer identity, signature algorithm, non-claim-only effect, and the fact that it grants no commercial rights. Qualification requires a trusted signer, a current record, and a matching `CardiovascularSignatureVerifier`. Verification calls `verifier.verify(payload, signature)`; it never re-signs with a private or shared signing key.

Signing remains the responsibility of the existing `ReleaseSigner` owner. Never commit signing keys or pass them in a dossier. Production release infrastructure supplies approved signers, public-key or trust-service verifiers, and key-custody controls.

## Qualification and decision separation

The workflow has three explicit records:

1. `evaluate_cardiovascular_release_candidate(...)` evaluates the exact tuple, case and execution manifests, artifacts, completed lifecycle records, signed non-claims, role assignments, dependency profiles, and fresh G0–G7 evidence. It returns a candidate. A candidate's `commercial_ready` property is always `False`.
2. `make_cardiovascular_release_decision(...)` has the separately named release approver sign the decision payload. It refuses approval of a blocked or expired candidate.
3. `assess_cardiovascular_release(...)` combines the candidate and decision, verifies the approver signature through the supplied verifier, and reapplies the release trust policy at decision time. Approval cannot erase a blocker, apply to another candidate, predate evaluation, or outlive evidence.

Only the final assessment can contain a released generic `CapabilityProfile`, and only when there are no blockers. Even then, `commercial_ready` means the technical release gate passed; the record explicitly grants no licence and makes no regulated-device claim.

## Repository preflight

Run the non-mutating preflight auditor from the repository root:

```console
python tools/cardiovascular_release_qualification.py
```

It inspects the repository licence and notice, six external artifact paths, eight signed gate records, four signed non-claims, and the separately stored decision record. It prints canonical JSON and exits with status 2 while prerequisites are missing. Override an artifact location with a repeatable option such as:

```console
python tools/cardiovascular_release_qualification.py \
  --artifact sbom=/sealed/release/sbom.spdx.json \
  --artifact build-provenance=/sealed/release/build-provenance.json
```

To derive reproducible evidence from the current tree and `uv.lock`, supply the
exact externally produced authority, scanner, signature-verification, and
attestation records plus every distribution being qualified:

```console
python tools/cardiovascular_release_qualification.py \
  --build-artifacts /sealed/release/derived \
  --commercial-license-record /sealed/authority/commercial-license.json \
  --data-rights-record /sealed/authority/data-rights.json \
  --signer-record /sealed/trust/signer.json \
  --verifier-record /sealed/trust/signature-verification.json \
  --license-report /sealed/scans/licenses.json \
  --vulnerability-report /sealed/scans/vulnerabilities.json \
  --supply-chain-attestation /sealed/attestations/supply-chain.json \
  --distribution-artifact wheel=/sealed/dist/phydrax.whl \
  --distribution-artifact sdist=/sealed/dist/phydrax.tar.gz \
  --distribution-artifact container=/sealed/dist/phydrax.oci.tar
```

Every external JSON record uses `schema_version: 1` and binds its exact
`kind`, `source_commit`, `lock_sha256`, non-empty `signer_id` and
`signature_algorithm`, and hexadecimal `signature`. The licence and
vulnerability reports additionally identify the scanner and report
`scan_status: passed`. They contain one entry for every locked name/version;
each entry binds all locked SHA-256 values. Licence entries provide
`license_concluded`, `license_declared`, and `copyright_text`. Vulnerability
entries provide `status: passed` and an explicitly empty `vulnerabilities`
list. The signature-verification record uses
`verification_status: verified` and binds the SHA-256 of every signed external
record in `subjects`. The supply-chain attestation uses
`attestation_status: verified` and binds the lock, licence and vulnerability
reports, authority records, signer record, and every supplied distribution in
the same way. These are evidence fields supplied by the responsible external
systems; the builder does not infer or create them.

The builder follows every runtime, optional, and development edge present in
the lock. SPDX 2.3 contains package SHA-256 checksums, source/download
locations, licence conclusions, `DESCRIBES`, and `DEPENDS_ON` relationships.
CycloneDX 1.6 contains the same package hashes/licences and its complete
`dependencies` graph. Build provenance hashes the exact wheel, sdist, and
container inputs. A dependency-complete supply-chain evidence manifest binds
the generated files to all external records. Missing packages, unresolved
edges, absent or malformed hashes, `NOASSERTION` licence/source/copyright
values, non-passing scans, stale commit or lock bindings, missing signature
metadata, or mismatched attestation subjects keep `g5_evidence_ready` false.

The builder never manufactures licence authority, data rights, scanner
results, signatures, verification, attestations, notices, or approval.
`commercial_ready` remains false even when the generated evidence can support
G5; the typed G0–G7 evaluation and independent decision must still run. A
dirty or non-Git tree, missing lock, PNPL repository licence, absent external
records, and missing notice licence texts are blockers. In particular, the
current NOTICE references `SING-MIT.txt` and `ASDEX-MIT.txt`; both absent files
are reported explicitly. The current source distribution therefore remains a
deterministic refusal and must not be described as commercial-ready.
