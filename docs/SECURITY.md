# Security policy

## Scope

This policy covers security defects in the PhydraX source distribution and its cardiovascular application code. It does not certify the software for clinical, safety-critical, regulated-device, diagnostic, treatment, or patient-care use.

The cardiovascular commercial-support boundary is isolated local execution with non-PHI inputs. Network access, external transfer, telemetry, PHI processing, clinical decision support, and regulated medical-device use are outside that boundary and are not enabled by a release assessment.

## Reporting a vulnerability

Report a suspected vulnerability privately through the project's published private contact channel. Do not open a public issue containing exploit details, credentials, signing material, personal data, or embargoed dependency information. Include, where available:

- affected version and source commit;
- platform, Python version, backend, and dependency-lock digest;
- exact support-tuple coordinates and execution-plan ID;
- minimal reproduction using synthetic non-PHI data;
- expected and observed security boundary;
- impact and whether credentials, artifacts, archives, or signatures are involved.

Never send protected health information, patient data, production credentials, private keys, HMAC secrets, proprietary anatomy, or restricted validation data. Replace sensitive inputs with a synthetic reproducer. If synthetic reproduction is impossible, first agree on an authorized secure transfer mechanism with the project maintainer.

Receipt is not an admission of impact. Maintainers will acknowledge the report through the private channel, reproduce it in an isolated environment, determine affected versions, coordinate remediation and disclosure, and publish an advisory when disclosure is appropriate. Release timing depends on verification of the fix and downstream coordination rather than a promised fixed deadline.

## Cardiovascular threat boundary

Qualification treats the following as explicit threats rather than assumptions:

- an approximate or wildcard support-tuple match;
- unpinned code, solver, backend, precision, or dependency state;
- missing, substituted, expired, or cyclic evidence dependencies;
- incomplete lifecycle runs presented as verification evidence;
- unsigned, incorrectly signed, differently bound, or altered G0–G7 and non-claim records;
- a reviewer acting as author or release approver;
- an approval applied to another or expired candidate;
- archive traversal, overwrite, decompression, or resource-exhaustion attacks;
- unexpected network egress, telemetry, or external data transfer;
- PHI or other disallowed data entering a non-PHI workflow;
- an engineering result being represented as a diagnosis, treatment recommendation, clinical decision, safety claim, or regulated-device output.

The commercial layer fails closed. Missing evidence is a blocker; it is never replaced by a default pass, inferred licence, nearby capability profile, or best-effort continuation.

## Required controls

A cardiovascular release candidate must bind:

1. one exact `SupportTuple` and one technical claims-matrix decision;
2. independent author, technical reviewer, validation reviewer, security reviewer, and release approver identities;
3. current, reviewer-signed G0–G7 records whose common dossier ID, reviewer, and evidence-ID tuples are authenticated and whose underlying `ReleaseGateEvidence` is accepted by the configured trust policy;
4. completed immutable lifecycle records whose analysis plan and numeric revision match a case-bound execution manifest, with each computational gate citing the exact run/manifest pair;
5. content-addressed SBOM, build-provenance, commercial-licence authorization, notice-audit, data-rights, and supply-chain references;
6. explicit artifact dependencies and SHA-256 manifests;
7. current signed non-claims for diagnosis, treatment, clinical decision support, and regulated medical-device use;
8. hard wall-time, memory, artifact-size, and concurrency limits;
9. isolated local execution with a dependency lock and without network access, telemetry, external transfer, or PHI;
10. a separately signed release decision verified against the independent approver identity while all evidence remains current.

The release decision cannot override failed qualification. G7 is an independent dossier review, not the release decision itself.

## Signing and secrets

Signing secrets and private keys must remain outside the repository and release dossier. Inject them from the approved secret store only for the signing operation. Restrict signer identities through the security policy, rotate compromised material, and reissue every record signed by a revoked key. A content digest proves identity, not authorship; trust requires both a valid signature and an allow-listed signer.

Verification must use a `CardiovascularSignatureVerifier` and its `verify(payload, signature)` operation. It must never reproduce a signature with a signing/private key. Release environments should provide an approved public-key or trust-service verifier corresponding to their existing `ReleaseSigner`; symmetric local schemes require controlled verifier custody and are not public provenance.

## Dependency and artifact security

Generate the SBOM and build provenance from the same pinned source and dependency lock used by qualification. The supply-chain attestation must reference those exact artifact IDs. Review all bundled licence and notice obligations, including optional backends and data/model artefacts actually included in the tuple. A clean vulnerability scan does not establish licence or data rights, and licence compatibility does not establish security.

Release artifacts must be immutable, regular files in controlled storage. Verify SHA-256 before use. Reject symlinks where a release process expects a sealed regular file, unexpected parents, duplicate artifact kinds, missing dependencies, expired references, and records outside configured size limits.

## Privacy and incident handling

Cardiovascular qualification accepts only data explicitly classified as non-PHI. Operators are responsible for classifying inputs before preparation. On suspected PHI ingestion, credential exposure, artifact substitution, or signer compromise:

1. stop the run and prevent publication;
2. revoke or quarantine affected credentials and artifacts;
3. preserve minimal non-sensitive audit evidence;
4. invalidate dependent gate records and release candidates;
5. follow applicable organizational incident-response and notification obligations;
6. requalify from a clean build after remediation.

Do not attach raw patient or proprietary data to an incident ticket. PhydraX provides technical controls, not legal, privacy, medical, or regulatory advice.
