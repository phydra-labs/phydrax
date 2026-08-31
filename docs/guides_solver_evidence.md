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
- **Invariant evidence** covers circuit KCL/KVL and gauge compatibility, and Maxwell
  chain identity, Gauss continuity, magnetic closedness, harmonic periods, energy,
  power, and dissipation signs.
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
verdict. The current artifact shape is canonical; Phydrax does not add a parallel schema
version or evidence-registry abstraction.

Resource reports count retained and temporary state, factors, projection workspace,
CPML and material auxiliary state, observers, checkpoints, acquisition/output,
padding, halos, compiler temporaries when available, and per-device storage. Core-field
bytes alone are not a memory claim.

## Fail-closed publication

README, changelog, API, and guide claims must link to passing checked evidence. Missing
hardware or compiler memory data are recorded as unavailable rather than inferred.
Claims are removed or narrowed when the relevant gate no longer passes.
