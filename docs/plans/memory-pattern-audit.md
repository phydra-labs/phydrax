# Memory-pattern audit and scaling ratchet

The memory audit is a deterministic, read-only AST review of `phydrax/`. It never
imports package modules and does not claim that a logical JAX intermediate survives
compiler fusion. Its integration test collects the repository scan. The benchmark
ratchet separately checks committed, observed memory evidence.

## Commands

```sh
python tools/audit_memory_patterns.py
python tools/audit_memory_patterns.py --json /tmp/memory-patterns.json
python -m benchmarks.memory_scaling
python -m benchmarks.memory_scaling --check
python -m pytest -q -n 3 tests/integration/test_audit_memory_patterns.py tests/unit/test_benchmark_memory.py
```

The JSON audit report is a sorted array whose records contain exactly `file`,
`line`, `symbol`, `rule`, and `detail`. Symbols include enclosing classes and nested
functions. The CLI also prints findings and sorted per-rule counts. It exits
nonzero for an undisposed finding or an invalid, duplicate, unsorted, or stale
allowlist entry. `--root` and `--allowlist` support isolated fixture scans.

## High-signal syntactic rules

| Rule | Candidate |
| --- | --- |
| `inverse-by-identity` | `solve(A, eye(n))` (including locally assigned identity RHS or named `b`/`rhs`) or NumPy/JAX `linalg.inv`, consumed only by local matvec, diagonal/trace, or selected-entry extraction. |
| `vmap-identity` | `vmap(action)(eye(n))`, including local aliases, with the same action-only consumers. |
| `dense-derivative` | Materialized `jax.jacfwd`, `jacrev`, `hessian`, or `jacobian` result used only by local matvec, diagonal/trace, or integer-selected entries. Imported JAX aliases are resolved. |
| `pairwise-broadcast` | Binary operations/comparisons between differently expanded views of the same array and identical remaining slices, with only direct reduction consumers in the same function. |
| `array-cache` | Module-level `cache`/`lru_cache` memoization (unbounded or count-only), cache-named dictionary initialization, or empty dictionaries explicitly annotated with array values, requiring numerical-payload/ownership review. The weighted host-table cache is not a candidate. |
| `payload-tobytes` | Whole-array `.tobytes()` in payload-path modules. |
| `payload-read-bytes` | Whole-file `.read_bytes()` in payload-path modules. |
| `whole-buffer` | `.getvalue()` in payload-path modules. |
| `identity-tolist` | `.tolist()` in identity, fingerprint, digest, payload, or canonical functions in payload-path modules. |
| `scan-reduce` | Locally unpacked `lax.scan` output or directly selected scan output immediately consumed by a reduction in the same function. |
| `dense-zero-sparse` | Locally allocated two-axis NumPy zero array subsequently passed to nonzero/argwhere or CSR/COO/CSC extraction. |
| `quadratic-prefix` | A dynamic-prefix sum recomputed inside a runtime/payload loop. |

Payload-path modules are those whose repository paths contain `archive`,
`checkpoint`, `resource`, `runtime`, `interchange`, `_fingerprint`, or
`_execution_workset`. This deliberate restriction targets serialization, input
staging, and identity boundaries rather than flagging every public byte-output or
arbitrary scientific identity constructor throughout the package.

Derivative/inverse consumers recognize `@`, `dot`/`vdot`, `diag`/`diagonal`,
`trace`, integer-entry indexing, and the explicit matvec contraction strings
`ij,j->i`, `ij,i->j`, `ij,j`, and `ij,i`. General tensor contractions, arbitrary
method names `inverse` (including inverse Fourier transforms, bijectors, groups,
and finite fields), returned dense derivative/identity materializations, and
returned pair tables are not automatically classified as misuse. Dense outputs
that remain part of a consumer contract need not be replaced by an action.

These restrictions keep findings individually reviewable, not exhaustive. The
analysis is lexical and local: it does not infer runtime dimensions, solve array
aliasing, propagate across calls, prove fusion, or perform control-flow analysis.
Named reassignments and indirect consumers can require manual review. Host
preparation versus runtime, physical tensor ownership, and public output size are
verified at each exact finding rather than exempting complete files.

## Exact allowlist policy

`memory-pattern-allowlist.tsv` has exactly four tab-separated columns:

```text
path\tsymbol\trule\trationale
```

Keys are exact `(path, qualified symbol, rule)` tuples in lexicographic order;
there are no line numbers, glob patterns, whole-file exceptions, or version fields.
One symbol may contain multiple instances of a rule, so review all its instances.
A source move or removal must remove its old disposition; stale entries fail even
when the rest of the scan is clean. Rationales are nonempty single lines and must
state a verified category (`bounded/static`, `heterogeneous`, `host-topology`,
`external-provider`, or `output-size`) and the specific bound/ownership/output
contract. Closed finite caches retain their exact fixed-domain rationale.

An avoidable dense temporary, unbounded numerical cache, or whole-payload hash
copy cannot receive a soft exception merely because its values are scientifically
necessary. Such findings must be remediated or explicitly resolved by the owner
before the repository gate is clean. Compiler-fused candidates may be retained
only with recorded baseline/candidate evidence and the shared gate decision.

## Aggregate benchmark ratchet

`benchmarks/memory_scaling.py` reads committed candidate `memory_*.json` records
and their baselines, excluding the aggregate itself and baseline-only records.
It recomputes declared adoption decisions through
`benchmarks._memory.evaluate_memory_gate` and fitted exponents through the shared
`fitted_scaling_exponent`; it does not create a second compiler gate. Exact record
family adapters own heterogeneous case/evidence layouts and fail closed on
unsupported or malformed evidence.

Comparisons require matching environment fingerprints and matching driver family,
cases, and evidence fields. Source fingerprints remain provenance and may differ
between baseline and candidate. Environment mismatches are explicit
not-compared dispositions, never portable passes. Ratios and declared scaling
bounds own acceptance; exact compiler byte values, sampled RSS, and elapsed time
are not portable thresholds. The aggregate summary is deterministically ordered
and has no timestamps or schema/version fields. `--check` recomputes and compares
that summary without rewriting it. Update the summary after any observed candidate
or baseline record changes; never seed measurements with guessed bytes.

Baseline artifacts lacking recorded driver/evidence identities receive
`baseline-identity-missing`; they are not cross-record comparisons even when the
environment matches. This currently applies to the historical linalg-spectral,
ocean-shell, and structured-transfer baselines. Their candidate scaling and any
same-record paired evidence remain available without inventing historical provenance.

Exact additive evidence fields are recognized only for the reviewed FEM
(`compiler_route`, `requested_tensor_families`), polynomial
(`compiler_scaling_exponent`, `declared_scaling_exponent`, `gate`), and linalg-solve
(`auxiliary`) records; common case controls and comparison metrics must still
match. Other evidence-identity changes fail closed when a historical comparison
is requested. Native MPC preparation uses
host retained bytes rather than nonexistent compiler memory, and interior-point
action products remain fixed while scaling problem size. Reactor preparation
checks its exact sparse-stencil capacity formula; its reported host and retained
fits remain observations, not an invented power-law bound on affine storage.

### Budget-derived default routes

Automatic defaults now have two distinct evidence cohorts. Adoption gates use
the current-run default versus its explicit dense/full-tile reference only at
above-budget capacities; the shared gate still decides at the largest capacity,
not at every row. Historical baseline identities and configurations remain
informational for these paired gates rather than being repurposed as their
reference. Below-budget rows have separate `fitting/` dispositions and never
count as evidence of a memory reduction.

The adapters verify driver-owned tile formulas and scientific controls, preserve
all explicit/restricted routes, and recompute declared fitting compiler equality
from the actual paired bytes. GP Lanczos instead separates single-tile
row-batch/output-storage semantics from compiler-byte observations.
`fitting/main` reports both actual byte series, whether equal or unequal; it does
not assume a historical relationship or claim direct array equality. GP
variational and the other drivers' declared fitting comparisons enforce their
recorded compiler-byte equality.

Astrodynamics, interface, graph, GP, MAC, Kubo, ringdown, and transport/attention
gates use their exact recorded cohorts and same-run references. Kubo fits its
declared compiler/logical envelopes only above budget; ringdown retains the
driver's adjacent-ratio checks across both fitting and tiled rows. Automatic
transport block sizes and MAC resolved routes may change within one controlling
capacity series and are not used as accidental grouping keys.
