# Black-hole execution, qualification, and production boundaries

Black-hole calculations use the shared `phydrax.execution`, lifecycle, qualification,
artifact, solver-runtime, and interchange owners. Numerical capability, scientific
qualification, production engineering, and permission to deploy are separate claims.
No `Simulation` or campaign facade weakens those boundaries.

## Resolve before execution

`ExecutionRequirements` declares scientific axes, operations, dtypes, transformations,
process-local input, and output constraints. `ResourceRequest` declares total CPUs,
memory, accelerators, hosts/processes, and optional per-device, host, compile-cache,
halo-collective, checkpoint-staging, and output-backlog byte ceilings plus required
dtypes/backends/collectives. `ExecutionResourceEvidence` contains static estimates and
explicit capability attestations; an estimate is not proof that the operation ran.

An `ExecutionPlan` binds decomposition, value placements, provider roles, solver and
precision policies, determinism scope, recovery policy, and allocation. Distribution is
explicitly `single`, `auto`, or `distributed`; determinism is `logical`, `topology`, or
`bitwise`; recovery is `fail_fast` or `checkpoint_restart`. Resolution fails if
resources, placement, transformations, or providers do not satisfy the request. It does
not silently choose a smaller physical problem or another provider.

For numerical relativity, `NumericalRelativityDistributedPlan` and
`NumericalRelativityAMRDistributionPlan` lower the already resolved spatial topology to
named shardings or canonical Morton-contiguous block ownership. A one-device execution
is valid local evidence only. Multi-device, multi-process, and multi-host claims require
those exact environments and their own scaling/restart evidence.

## Checkpoint, output, failure, and cancellation

A `NumericalRelativityCheckpointPlan` binds the exact formulation, runtime, geometry,
topology and epoch, analysis plan, numeric revision, execution plan, field signature,
constrained-transport plan, and typed restart-state template. It writes pickle-free
local archives or publishes canonical addressable shards through the existing
lifecycle repositories. A distributed publication must be repository-committed before
assembly; restore validates bounded reconstruction metadata and committed rank,
artifact, transaction, path, shape and dtype identities before returning the complete
typed `DistributedNumericalRelativityRestart`.

Exact restart requires exact array equality. Tolerance restart reports maximum absolute
and relative errors under an explicit nonzero policy. Both require the same formulation,
runtime, geometry, topology/epoch, field structure, and lifecycle admission; neither
permits topology change and both return `derivative_valid=False`.

Production output, failure, and cancellation are committed records, not log messages:

- `NumericalRelativityOutputCommitter` accepts only runtime-generated publish event
  IDs, calls a bounded writer, and retains `NumericalRelativityCommittedOutputReceipt`
  only after successful return. Receipts bind writer/event/cursor, exact production
  state, artifacts, bytes, numerical status and all scientific dispositions; they are
  consumable only after the shared publisher acknowledges the event.
- `NumericalRelativityOutputManifest` requires that acknowledged receipt plus the
  authoritative `ProductionRunResult`, verifies run/state/cursor lineage, and reapplies
  output count/byte limits.
- `NumericalRelativityFailureManifest` retains the admitted category/error code, last
  accepted step/time, optional terminal checkpoint identity, and the
  `Z4cProductionState` dispositions.
- `NumericalRelativityCancellationManifest` requires a verified
  `CheckpointCommitReceipt` for the cancelled authoritative state and retains its
  content digest, commit ID/locator, durable size and SHA-256.

## Neutral external artifacts

`phydrax.interchange` maps caller-supplied local `field`, `image`, `visibility`,
`waveform`, and inert `numeric-model` bytes through the same security boundary.
The shared artifact-admission policy confines access to a trusted existing root, rejects symbolic
links and path escapes, enforces size and suffix allow-lists, and forbids pickle-family
formats. SHA-256, exact byte size, license ID, and manifest are verified before content
is retained.

`BlackHoleArtifactRights` binds artifact/source identity, producer/version, model,
coverage, exact digest/size, license/source/attribution and separate permission bits
for commercial use, training, redistribution, derivative use, model execution, and
export. `BlackHoleArtifactUsePolicy` declares the requested use; absent permission fails
closed. Production artifact bindings retain both complete records, not only a license
string. Rights are caller-supplied evidence, not inferred from a citation or filename.
`phydrax.service.ArtifactRights` is a separate service-delivery record binding a
scientific artifact to rights/use-policy IDs and allowed principals/actions. It cannot
replace `BlackHoleArtifactRights` or add permission to the underlying content.
`BlackHoleArtifactSchema` requires kind-specific semantic identities:

| Kind | Required semantics |
| --- | --- |
| field | chart, coordinate frame, quantity, topology, unit |
| image | observable, screen frame, unit |
| visibility | baseline frame, frequency axis, polarization basis, unit |
| waveform | mode basis, quantity, time reference, unit |
| numeric model | architecture, input schema, output schema, precision |

A numeric model may use ONNX, SafeTensors, NPZ, or the native pickle-free artifact
format, but `map_numeric_model_artifact` never deserializes or executes it. All mapping
functions return `NeutralBlackHoleArtifact` with bytes, admission, resource manifest,
rights, policy, schema, and explicit `AdapterReport` losses/waivers. They do not fetch
external content, call a provider, or imply scientific validity.

## Exact production binding

`NumericalRelativityDomainBinding` fixes formulation, chart, gauge, EOS, topology,
precision, and all admitted input artifacts. `NumericalRelativitySupportBinding` keeps
scientific and deployment support tuples separate. `NumericalRelativityProductionPlan`
requires those tuples to agree exactly with the domain, `ResolvedRunSpec`,
`ProductionCaseManifest`, resolver-produced `ExecutionPlan`, `ProductionRunPlan`,
checkpoint policy, and `NumericalRelativityProductionLimits`.

The limits own the exact `ExecutionPolicy`/`ResourceRequest`; checkpoint-staging and
output-backlog maxima are mandatory. Device/host peak plus reserve, compilation cache,
halo collective, checkpoint staging, output backlog, required dtype/backend/collective,
input artifact count/bytes, output manifest/artifact count/bytes, and cancellation
detail are admitted against resolver evidence. Missing observed evidence fails rather
than passing an optional ceiling.

The only production method admitted here is `FixedGridZ4cProductionMethod`.
`Z4cProductionState` keeps numerical status and finite/converged/physical/qualified/
derivative dispositions with the committed runtime state. Initial scientific
dispositions are unavailable until an evaluated step supplies evidence. Matter input,
when used, is a snapshot-aware provider called against each exact ADM
`geometry_lineage_id`/`snapshot_token`; a detached stress-energy snapshot is not reused.
The method does not reduce its required step or repair rejection.
`compile_numerical_relativity_production` compiles bindings only; it creates no
scientific evidence, deployment support, rights, or release approval.

The current architecture can therefore make a precise technical statement: the
production runtime has bounded resources, checkpoint/restart, output, failure,
cancellation, and exact-domain admission mechanisms. It does **not** mean that every
black-hole kernel, AMR route, coupled-matter runtime, hardware topology, or scientific
case is production-qualified.

## Validation authorities

Each authority answers a different question:

| Authority | Evidence it owns | It does not establish |
| --- | --- | --- |
| kernel/result | finite, residual, conservation, physical, derivative status for one evaluation | case convergence, external validity, or release |
| benchmark | bounded setup/lowering/compile/synchronized execution, measured work/resources for the exercised profile | a new domain, scaling not exercised, or deployment |
| scientific qualification | comparison to an independent reference or invariant for one exact support tuple | hardware operations, legal rights, or broader astrophysical fidelity |
| execution qualification | exact backend/topology/precision/resource/restart behavior actually exercised | scientific accuracy |
| production binding | identity, resource, artifact, checkpoint, output, failure, and recovery controls | release decision or permission to operate |
| independent deployment authority | environment-specific legal, security, quality, operational, and intended-use approval | any broader tuple than the approved one |

A `qualified=True` field is local to its product's declared evidence. It must not be
promoted to “production qualified” without an accepted support profile and the
necessary execution and production evidence.

## Benchmark and qualification lanes

The seven benchmark entry points exercise these exact bounded synthetic profiles:

| Entry point | Exercised path |
| --- | --- |
| `benchmarks/black_hole_geometry.py` | exterior Kerr Boyer--Lindquist metric/domain, curvature invariants, coordinate JVP |
| `benchmarks/gr_imaging.py` | public Kerr exterior null-ray history, invariant scalar transfer, emission-scale derivative |
| `benchmarks/black_hole_perturbations.py` | scalar $s=\ell=m=0$ Schwarzschild $M=1$ Riccati/log-amplitude solve at $\omega=0.4-0.05i$, default 65 nodes to $40M$, fixed 32 substeps and order-12 infinity series; complex match, Chebyshev ODE gate and residual JVP; no solved QNM |
| `benchmarks/relativistic_matter.py` | SRHD recovery, flux, causal bounds, and JVP |
| `benchmarks/numerical_relativity.py` | periodic linear-wave Z4c RHS, constraints, and JVP |
| `benchmarks/black_hole_runtime.py` | atomic two-dimensional periodic all-active Valencia GRMHD SSPRK3/CT rollout and step-size JVP |
| `benchmarks/grrmhd_radiation.py` | batched implicit GRRMHD four-force source, nonlinear residual, realizability, and exact energy-momentum defect |

They record bounded local JAX setup/lowering/compile/synchronized execution,
landed-kernel physics and statuses, logical bytes, compiler evidence, and derivative
evidence where meaningful. They do not establish convergence beyond configured
capacities, observational fidelity outside their stated synthetic profiles, distributed
scaling, deployment authorization, or production qualification. The exact current
qualification commands and profile dispositions are listed in
[Black-hole sources, provenance, and qualification](black_hole_sources.md).

The qualification runner requires `--valid-at`. Every selected profile carries one
runtime manifest binding the runner and installed source-tree SHA-256 values, dependency
versions, Python/OS/machine/byte-order, JAX backend/process/device inventory, and
precision disposition. Its evaluated and expiry timestamps both equal `--valid-at`;
the evidence is observation-time-only and asserts no future validity. A profile result
without that exact `manifest_id` is not reusable evidence.

## PNPL and deployment

The repository is distributed under PNPL. Technical qualification records describe
software behavior; they are not a license grant and do not authorize deployment.
Commercial, hosted-service, redistribution, training, model-execution, output-egress,
and other uses remain subject to PNPL, third-party terms, artifact/data rights, and any
required written authorization. Even a fully passing numerical, scientific, execution,
and production dossier cannot manufacture those rights or an independent deployment
decision.

## Derivative boundary

Resource discovery, provider resolution, allocation, process topology, sharding,
repartitioning, AMR compilation, checkpoint I/O, restart admission, artifact reading,
rights decisions, output publication, failure/cancellation, and release/deployment
review are host boundaries with no scientific derivative. Array kernels retain only the
derivatives admitted by their product inside a frozen execution plan, topology, branch,
active mask, and source identity.