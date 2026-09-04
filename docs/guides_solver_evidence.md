# Solver evidence gates

Phydrax separates implementation status from scientific, differentiation, backend,
performance, and scaling claims. A finite result or successful solver status is not
physical validation.

## Evidence classes

Every published claim identifies the applicable evidence classes and the checked
artifact that carries the observations, thresholds, and verdict.

- **Contract evidence** records public shapes, units, ordering, wave/sign conventions,
  static versus dynamic inputs, lifecycle identity, failure behavior, and base-import
  behavior.
- **Numerical-reference evidence** compares deterministic float64 or complex128 results
  against an analytical or independently assembled oracle and reports original-equation
  absolute and relative residuals.
- **Invariant evidence** covers the conservation, compatibility, gauge, symmetry,
  trace, divergence, energy/work/dissipation-sign, flux-sharing, and restart
  identities required by the exact route, including circuit, Maxwell, and LES
  specialization-specific ledgers.
- **Derivative evidence** compares directional derivatives of named real controls with
  centered finite differences. Topology, port ordering, mode selection, rank changes,
  singular solves, parsed files, padding layouts, and shard layouts are ineligible
  unless qualified separately.
- **Scientific qualification evidence** covers convergence and parameter sweeps such as
  mesh, time step, harmonic count, CPML width, frequency, angle, polarization, and
  corners. Reference data retain checksums and provenance.
- **Backend evidence** covers native-provider parity, status and diagnostics, transfer
  counts, supported dtype and feature subsets, and missing package, transitive import,
  linker, device, and ABI failures. Explicit provider selection never falls back.
- **Performance evidence** reports raw samples, environment identity, cold planning and
  compilation, warm execution, end-to-end placement and output, and peak host/device
  memory against the same-device, same-precision native baseline.
- **Scaling evidence** adds unsharded parity, communicated bytes, per-device memory, and
  strong or weak scaling only on hardware actually exercised.
- **Release evidence** covers clean base and extra installations, minimal imports,
  licenses and notices, exports, documentation, artifact checksums, and supported
  versions.

Scientific correctness requires contract, numerical-reference, and invariant evidence.
A differentiability claim additionally requires derivative evidence. A speed claim
requires performance evidence without weakening numerical or invariant gates. A
sharding claim requires backend and scaling evidence.

## Checked artifacts

Checked JSON artifacts contain the artifact kind, git commit, environment fingerprint,
problem, plan, prepared, and policy identities, raw observations, thresholds, and
verdict. The current artifact shape is canonical: readers validate the exact expected
fields, reject non-finite values, and recompute content identities rather than guessing
at compatibility.

Resource reports count retained and temporary state, factors, projection workspace,
CPML and material auxiliary state, observers, checkpoints, acquisition/output,
padding, halos, compiler temporaries when available, and per-device storage. Core-field
bytes alone are not a memory claim.

## Fail-closed publication

README, changelog, API, and guide claims must link to passing checked evidence. Missing
hardware or compiler memory data are recorded as unavailable rather than inferred.
Claims are removed or narrowed when the relevant gate no longer passes.

## Candidate, evidence, and release ownership

`phydrax.qualification` keeps three questions separate. `SupportTuple` names one exact
provider-neutral conjunction of capability coordinates. `QualificationMatrix.evaluate`
matches current `QualificationEvidence` to every named predicate and distinguishes
failed proof from missing or inconclusive proof. `CapabilityProfile` then declares the
support tuples, exact dependencies, release gates, and evidence for one capability
family. A profile with `released=False` is a candidate even when its numerical matrix
passes.

`ObservedResourceRecord` is measured use for one subject/build/environment/backend/
topology and names its raw artifacts. `ForecastResourceRecord` is separately labeled,
time-bounded model output with uncertainty bounds and source record IDs; a forecast is
never treated as an observation.


`ReleaseIndex` is the signed publication boundary and `require_profile` is the
fail-closed consumer boundary. A released profile must carry accepted, current evidence
for every required gate and satisfy its dependency graph. HMAC support is suitable for
local registries and sealed CI secrets; it is not a claim that any candidate shipped
with signed evidence. `ReferenceArtifactManifest` separately binds offline reference
files, checksums, provenance, units, and nondimensionalization.

### LES evidence

`tools/large_eddy_simulation_qualification.py` must publish through these same generic
contracts. A LES result binds the exact resolved/test filters, parameter provenance,
prepared closure/action, base compilation, discretization, temporal method,
boundary/geometry route, reference manifest, resolved run, environment, and raw
diagnostics. Applicable predicates include stress symmetry/trace, divergence and
reality, conservation, modal or variational work, SGS transfer sign/policy, timestep
restriction, scalar/KSGS/Favre closure, restart, and backend/resource evidence.

The producer does not define another evidence or gate schema. Generated LES profiles
remain candidate/unreleased even after a matrix passes. The corresponding base
incompressible profile is an external release dependency and cannot be inferred,
signed, or waived by the LES campaign. One-device distributed parity never supplies
multi-device scaling or release evidence. See the
[LES guide](guides_large_eddy_simulation.md#qualification-and-release-boundary).

## Configuration and lifecycle

`resolve_run_spec` binds scientific and deployment `SupportDependency` records to exact
configuration, build, backend, precision, topology, scheduler, repository, and
authentication-policy identities before execution. `CompatibilityRegistry` accepts
only explicit acyclic, forward transformations to its current writer format; ambiguous
or unsupported paths fail, loss requires opt-in, and each `MigrationReport` retains the
complete content lineage. `migrate_configuration` commits that report and payload
through an `ArtifactRepository`; `rollback_configuration` is available only when every
edge is reversible.

`POSIXArtifactRepository` is crash-consistent only for an
`HPCFilesystemProfile` that declares same-filesystem atomic rename, file and directory
fsync, advisory locking, and attempt-private staging. `S3ArtifactRepository` instead
requires a caller-supplied `ConditionalObjectClient` whose declared
`ObjectStoreProfile` provides conditional create/replace, strongly consistent reads and
listing, bounded whole objects, and no multipart object assumption. Neither repository
contains a vendor SDK fallback. S3 pointer commits, leases, legal holds, retention,
tombstones, and garbage collection share one non-expiring per-artifact conditional
guard. A crashed guard fails closed; after the old worker is externally fenced,
`ArtifactGuardRecoveryAuthorization` binds the exact provider, artifact, guard ETag,
authority, and fencing evidence consumed by `recover_artifact_guard`. Time alone never
permits another writer to overtake destructive collection. Immutable chunk checksums
and garbage-collection reports remain explicit.

`TopologyRestartRelation` and `TopologyRestartPolicy` decide exact versus
topology-changing restart before I/O. `prepare_direct_restore` validates complete
canonical byte coverage and destination ownership; `execute_direct_restore` calls the
injected `ChunkRangeReader` and `DestinationShardWriter` for direct range transfer
without a global payload gather. A semantic change is not topology migration.

## Service, identity, and security boundary

`InProcessReferenceService` is a synchronous, thread-safe reference service, not a
network server. It authenticates before resource lookup, authorizes tenant-scoped
operations, admits exact support dependencies, enforces quotas and job transitions,
keeps provider selection server-side, and commits complete checkpoint/artifact/audit
state or fails closed. Each provider execution captures an immutable attempt/version
fence; `ExecutionContext.heartbeat()` renews its durable lease through compare-and-swap,
and checkpoint/terminal commits reject superseded attempts. `SQLiteServiceStore`
supplies the local durable job, quota, hash-chained audit, and replay-safe outbox
implementation. Slurm and Kubernetes schedulers are explicit adapters over
`CommandExecutor` and `HTTPTransport`; infrastructure credentials, retry cadence, and
deployment remain provider responsibilities.

`OIDCJWKSTokenValidator` accepts only configured algorithms and uses an injected
`JWKSProvider`; `HTTPSJWKSProvider` owns bounded HTTPS retrieval and caching.
Asymmetric JWT and X.509 verification, plus `Ed25519Signer` and
`Ed25519Verifier`, require the optional `commercial` dependency extra, which installs
`cryptography`; construction fails explicitly when it is absent. `KMSSigner` and
`KMSVerifier` delegate bytes to injected KMS providers and do
not claim a particular cloud. `LocalSecretHandleBroker` is process-local reference
storage, while `SigningTrustStore` makes activation, rotation, and revocation explicit.
No API claim implies globally certified infrastructure.
