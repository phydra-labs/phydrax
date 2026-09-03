# Support policy

## What support means

Support for the PhydraX cardiovascular application means reproducible engineering assistance within a declared, exact capability profile. It is not a warranty, service-level agreement, commercial licence, medical opinion, clinical service, safety assurance, regulatory submission, or certification.

The governing licence remains the `LICENSE` file in the distribution. A technical release record does not expand its grant. Contact the project's published commercial contact for a separate agreement where required.

## Cardiovascular support boundary

The only commercial-support profile defined by the source is local, isolated, non-PHI, non-networked engineering execution. It requires an exact `cardiovascular.workflow` `SupportTuple` with:

- `deployment=local`;
- `data_classification=non-phi`;
- `regulated_device=false`.

All additional tuple coordinates are conjunctive. Fidelity route, precision, anatomy, equations, constitutive choices, solver policies, observation operators, data sources, derivative policy, and capacity envelope must match the released profile exactly. Similarity to a released tuple is not support. There is one capability profile per exact tuple.

The following uses are categorically outside support:

- processing PHI or patient-identifiable data;
- diagnosis or screening;
- treatment selection, planning, or recommendation;
- clinical decision support;
- regulated medical-device use or a device safety/effectiveness claim;
- autonomous patient-care decisions;
- hosted, managed-service, telemetry-enabled, networked, or externally transferring operation;
- use beyond the limits in the applicable licence or agreement;
- a tuple, dependency profile, artifact, backend, or optional component not named by the released profile.

A signed non-claim records each medical-use exclusion. These records document the boundary; they do not make an excluded workflow acceptable.

## Qualification states

| State | Meaning |
| --- | --- |
| Undeclared | No exact claims-matrix decision exists; unsupported |
| Technical candidate | Exact tuple may collect G0–G7 evidence; not released |
| Qualified candidate | Current evidence passes; still lacks independent release authorization |
| Released profile | Qualification and the separate release decision pass while evidence is current |
| Blocked or expired | At least one prerequisite fails; unsupported until requalified |

Only the final typed assessment can mark its generic capability profile released. The candidate object intentionally always reports `commercial_ready=False`. An approval cannot erase a qualification blocker or be reused for another candidate.

## Requesting engineering support

Use the project's published support channel. A useful request includes:

- PhydraX version and source commit;
- operating system, Python version, backend, device, and precision policy;
- dependency-lock and build IDs;
- exact support-tuple record and profile ID;
- analysis/execution plan IDs and immutable lifecycle run ID;
- deterministic steps using the smallest synthetic non-PHI input;
- observed status, diagnostic IDs, and complete error text;
- expected behavior and whether the issue reproduces without optional backends.

Do not send PHI, patient data, proprietary clinical data, production credentials, signing keys, commercial agreements, or confidential data-rights documents through a public issue. Redact paths and identifiers that reveal sensitive infrastructure. Security defects should follow `docs/SECURITY.md` rather than a public support request.

## Resource and operational limits

The released tuple's `CardiovascularResourcePolicy` is a hard admission boundary, not a planning estimate. Runs beyond wall-time, resident-memory, artifact-size, or concurrency ceilings are unsupported and should fail before or at the controlled boundary. Operators remain responsible for host capacity, backups, access control, key custody, and retention.

A completed run is required where a gate cites lifecycle evidence. Planned, queued, running, failed, or cancelled records do not become verification evidence. Checkpoints and results must retain their plan, numeric revision, execution, unit, axis, sign, reference-configuration, and content identities.

## Versions and evidence lifetime

Support follows an explicitly released profile version, not the latest branch. Any change to a tuple coordinate, dependency, build, SBOM, licence determination, notice audit, data rights, validation domain, security posture, or evidence validity may require a new profile and release decision.

Evidence is accepted only within its issued and expiry times and any stricter trust-policy age. When the earliest gate, artifact reference, signed non-claim, or dependency profile expires, create a new candidate and repeat independent review. Do not extend dates in place or reinterpret an expired record.

## No implied claims

Examples, documentation, passing tests, benchmark output, numerical convergence, or a `commercial_ready` technical assessment do not by themselves establish fitness for a purpose, patient benefit, clinical validity, device compliance, data rights, or permission for commercial use. Those determinations require the appropriate independent legal, security, quality, domain, and regulatory processes outside this source distribution.
