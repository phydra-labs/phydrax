# Testing Phydrax

Phydrax tests are organized around consumer-visible contracts and independent
scientific oracles. Test modules do not pin import spelling, private wiring,
forwarding calls, source text, or incidental defaults.

## Running affected tests

Select the smallest conservative set affected since the latest merge into `dev`, then
run that set in parallel:

```bash
uv run --extra tests pytest -n auto tests/unit/<area>
```

A change to shared support additionally runs every family that imports the changed
support module. A repository-wide test refactor or cross-cutting configuration change
runs the full default suite. Real-provider and multi-device sessions run separately
when their requirements are available.

Use stable parametrization IDs so pytest's cache can address failures precisely:

```bash
uv run --extra tests pytest -n auto --lf
uv run --extra tests pytest -n auto --ff
```

Rerunning a failed statistical test is not an acceptance policy. Statistical tests use
deterministic key addressing and a tolerance justified by their standard error.

## Memory benchmark smoke runs

Run memory drivers serially with `python -m benchmarks.memory_<area> --smoke`.
Use `--output /tmp/memory_<area>.json` to avoid replacing qualification records.
The paired `memory_cut_cochain_transfer` and `memory_integration_lowering` drivers
instead accept `--output-dir /tmp/memory-smoke`; they measure independent reference
routes in the same run and write both records there. `memory_linalg_operators`
also requires `--phase candidate`, and `memory_linalg_spectral` requires an
explicit `--output` path.

`memory_operator_workspace` defaults to the candidate campaign at the committed
baseline capacities. Its `--smoke` campaign uses the first two matching capacities
of each route; `--full` explicitly selects the default full campaign. Gate
comparisons match both target capacity and source width, not record position.
Missing or duplicate historical problem sizes are errors; baseline measurements
are neither synthesized nor extrapolated. Capture a new baseline only deliberately
with `--phase baseline`, before changing the implementation being qualified.

## Test structure

Shared test infrastructure lives under `tests/_support`. Area-specific construction
stays beside its tests:

```text
tests/unit/<area>/
  _cases.py
  _fixtures.py
  test_contracts.py
  test_<unique_science>.py
```

Case rows are frozen, precisely typed records. They state expected capabilities
independently of the subject under test. A conformance test first compares the product
declaration with that expectation, then exercises every advertised route and refuses
relevant unadvertised routes.

Use one collected test for one coherent contract family. Do not hide independently
selectable cases in runtime subtests or a loop merely to reduce pytest's item count.
Small data matrices may be asserted as one contract when the failure reports the
failing row.

A helper belongs in shared support only when it owns a real invariant. Domain-specific
physics, tolerances, evidence, and provenance stay with their owner.

## Typing

Every function, method, fixture, nested helper, and factory is fully annotated. Prefer:

- Protocols for subjects, prepared artifacts, results, and adapters;
- typed factory callables;
- package-owned selector aliases and Literals;
- native `phydrax.typing` forms for arrays;
- precise PyTree types.

`Any` is limited to external or runtime-dynamic boundaries that static types cannot
express. A deliberate static misuse may pair a line-local ty suppression with the
runtime refusal:

```python
with pytest.raises(TypeError):
    function(invalid)  # ty: ignore[invalid-argument-type]
```

The repository-wide typing gate rejects unused suppressions:

```bash
uv run --extra qa python tools/check_typing.py check
```

Phydrax does not install a test-only annotation hook. Opted-in StrictModules validate
construction in production, and `phydrax.typing.checked` boundaries check their
arguments in production. Tests call `phydrax.typing.validate` explicitly after a
tree transformation or reconstruction.

`tests/unit/typing/test_signature.py` owns the semantics of checked signatures. The
repository declaration test compiles every checked boundary, so a consumer test of a
migrated constructor or method asserts its scientific behavior and the refusal
category (`TypeError` naming the argument), not the wording of a removed hand-written
guard.

## Hypothesis and JAX

Hypothesis generates host-side semantic descriptors and NumPy arrays. Tests convert
through the public Phydrax boundary once. Shapes come from a small declared set so one
compiled callable can be reused across examples.

Use properties for:

- scope binding and rollback;
- sparse permutation and duplicate reduction;
- linearity, transpose, and adjoint identities;
- deterministic key replay;
- canonical ordering and tamper refusal;
- prepared, refresh, checkpoint, and rollback sequences;
- capacity boundaries and resource refusal.

Do not use property generation to replace independent analytic references, create
unbounded shape sets, or compile a distinct JAX program for every example.

## Numerical and failure assertions

Tree comparisons require matching structure, leaf types, shapes, and dtypes in
addition to values. Tolerances are explicit and justified by dtype, conditioning,
problem scale, discretization order, or statistical error.

Failure tests assert the public error or result contract:

- `TypeError` for wrong object kinds;
- `ValueError` for invalid values or incompatible semantics;
- domain errors for failed scientific contracts;
- failed status, validity masks, and evidence for data-dependent traced failures.

Constructor, static-planning, and resource-admission errors fail before assignment or
allocation. Dynamic numerical invalidity remains in compiled execution and is reported
through status and evidence; it is not forced into host validation.

## Providers and topology

Provider tests consume each owner's availability evidence and retain real-provider
coverage. A fake may test process containment, timeout, or failure parsing, but never
stand in for a claimed scientific provider result.

Ordinary tests use the natural device topology. Multi-device tests run in a fresh
subprocess whose startup environment configures the requested devices. The root
conftest never changes the global device count.

## Reference fixtures

JSON fixtures use the owner's canonical JSON writer and may carry a separately
computed canonical fingerprint. Independent numerical references use a neutral host
format such as NPZ plus a canonical JSON provenance manifest. Product archives are
used only when the archive format itself is under test.

Reference generators live in `tools/`, provide `generate` and `check` modes, and never
update an oracle from Phydrax output. Manifests record source, method, generator,
provider release, precision, input regime, tolerance rationale, digest, and license.
They do not carry internal schema-version fields.

### Bounded identity hashing

Array fingerprints preserve the historical metadata framing and C-order byte stream,
including scalar promotion, byte order, and NaN payloads. Contiguous host arrays are
hashed through a byte view; strided arrays use bounded 64-KiB staging chunks.
Atomistic surface evaluation transfers identity leaves together and reuses the host
snapshot for its source and evaluation IDs. Runtime distribution integrity hashes
files incrementally and retains before/after mutation checks.

Prepared worksets count active lanes while forming buckets and stream the historical
sorted canonical JSON into their identity digest. Bucket order, padding, semantic RNG
addresses, and `prepared_id` do not change. Tests compare array hashing to an
independent legacy byte-stream oracle and workset identity to ordinary Python rows.
Run `python -m benchmarks.memory_fingerprint_workset --smoke` for phase-separated
host-allocation evidence, including legacy-reference versus streamed identity scratch.

### Host preparation retention

Dynamic host-table caches use deterministic LRU eviction under both entry and
logical-payload byte ceilings. Site budgets are the larger of one MiB and the
next power of two covering eight times the measured representative hot set;
oversized results are returned without retention. The quota excludes in-flight
calculations, allocator overhead, and caller-held references. Weak class keys
keep trainable-role metadata from retaining transient classes. Virasoro plans
admit descendant counts before enumerating partitions.

Run `python -m benchmarks.memory_host_caches --smoke` to exercise all migrated
sites, oversized-result behavior, and the retained-byte plateau.

## Performance evidence

`--durations`, collection timing, import timing, process RSS, and JAX compile logs only
locate candidates. An accepted performance change uses `benchmarks/_runtime.py` to
measure separately:

- lowering;
- compilation;
- first synchronized execution;
- warmed execution;
- compiler argument, output, temporary, and generated-code bytes;
- logical retained bytes;
- the controlling capacity.

Correctness is checked first: RNG addressing, reduction order, numerical stability,
masks, overflow evidence, rollback, failure status, caller-state reuse, and donation
ownership must be unchanged.

Persistent compilation-cache campaigns keep cold and warm evidence separate. Mutation
campaigns always disable the cache, and writable caches are trusted per-run resources.
