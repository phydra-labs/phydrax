# Bioinformatics qualification and benchmarks

Qualification answers a narrow question about a pinned method, input, environment, and
tolerance. It does not turn a numerical primitive into a clinical, regulatory,
mechanistic, or generalization claim.

## Evidence levels

Keep these artifacts separate:

1. **Contract tests** defend observable semantics such as masks, coordinate conversion,
   no-call behavior, exact/conditional classification, experimental units, and
   overflow failure.
2. **Reference qualification** compares a native result with an analytic result,
   independently implemented finite example, or explicitly pinned external dataset.
3. **Performance benchmarks** measure execution, compilation, and memory behavior for
   declared shapes and hardware.
4. **Learned-model evaluation** measures an immutable artifact on an immutable,
   leakage-audited split and homology partition.

Passing one level does not imply another. Runtime parity is not scientific parity;
scientific parity on a toy case is not capacity or scale evidence; and a held-out metric
is not evidence of no pretraining overlap.

## Claim matrix

Every qualification record should state:

| Field | Required question |
| --- | --- |
| Method contract | exact model, approximate model, relaxed objective, heuristic, or learned? What execution and differentiation semantics? |
| Conditioning | full state/path/tree domain, declared band, supplied candidates, supplied graph, or bounded search? |
| Identity | input digest, reference/feature/alphabet/tree/model/tokenizer/artifact fingerprints, implementation revision, and environment? |
| Units and precision | biological units, coordinate frame, dtype, tolerances, accelerator precision policy? |
| Capacity | required/configured records, positions, routes, states, edges, atoms, spectra/peaks, reactions/solves, and memory? |
| Experimental design | subject/specimen/donor/unit, technical replication, exchangeability, split, and leakage audit? |
| Outcome | validity/status and named evidence, not only a scalar error or elapsed time? |

Do not rename a `HEURISTIC` result “exact” because it matches one reference case. For an
`EXACT_MODEL` result, state the finite model and all conditioning. Approximate p-values,
relaxed scores, and stochastic estimates need their approximation/sample policy.
Learned results need artifact and training provenance.

## Opt-in external datasets

Qualification tooling must never download data. External inputs are admitted only when
the caller explicitly supplies a local path and expected digest. Verify the digest
before parsing or allocating domain state. A path, URL, release name, or filename is not
an identity.

A reproducible record includes the expected and observed digest, parser/adapter,
reference/annotation release and digest, selected records, exclusion policy, units,
capacity buckets, and any expected result/tolerance source. On mismatch, fail before
producing scientific output. Do not “refresh” a digest automatically.

Licenses and redistribution remain caller responsibilities. A qualification artifact
may record a local dataset identity without copying source data into the repository or
JSON output.

## Running the producers

Each producer runs deterministic built-in cases and writes JSON only when `--output` is
supplied:

```console
python -m tools.bioinformatics_sequence_dp_qualification --output artifacts/sequence.json
python -m tools.bioinformatics_phylogenetics_qualification --output artifacts/phylogenetics.json
python -m tools.bioinformatics_genomics_variant_qualification --output artifacts/genomics.json
python -m tools.bioinformatics_omics_statistics_qualification --output artifacts/omics.json
python -m tools.bioinformatics_structure_rna_qualification --output artifacts/structure-rna.json
python -m tools.bioinformatics_spatial_spectrometry_qualification --output artifacts/spatial-spectrometry.json
python -m tools.bioinformatics_systems_qualification --output artifacts/systems.json
python -m tools.bioinformatics_quick_benchmarks --warmup 1 --repeats 5 --output artifacts/benchmarks.json
```

External campaigns are opt-in and require a matching path/digest pair:

```console
python -m tools.bioinformatics_genomics_variant_qualification \
  --giab-root /local/giab --giab-sha256 EXPECTED_SHA256 \
  --cami-root /local/cami --cami-sha256 EXPECTED_SHA256 \
  --output artifacts/genomics-external.json

python -m tools.bioinformatics_omics_statistics_qualification \
  --omics-standard-root /local/omics-standard \
  --omics-standard-sha256 EXPECTED_SHA256 \
  --output artifacts/omics-external.json

python -m tools.bioinformatics_structure_rna_qualification \
  --mmcif-root /local/mmcif \
  --mmcif-sha256 EXPECTED_SHA256 \
  --output artifacts/structure-rna-external.json
```

Replace every digest with the digest of the intended local dataset root. The tools do
not fetch a missing input or infer a digest. `external_campaigns_requested` records the
request independently from completed `external_campaigns`.

## Biological splits and leakage

Create `BiologicalSplit` from a grouping appropriate to the claim: donor, subject,
biospecimen, clonal lineage, family, species, homologous sequence family, acquisition
batch, or a stricter hierarchy. Run `LeakageAudit` before fitting normalization,
dispersion trends, feature selection, alignment/integration models, calibration,
thresholds, or learned models.

For pretrained artifacts, retain `PretrainingOverlapProvenance` for the exact evaluation
split and homology partition. `unknown` is a legitimate warning, not equivalent to
`no-detected-overlap`. Technical cells, spots, reads, spectra, crops, or augmentations
from one experimental unit must not cross partitions or inflate replicate count.

## Benchmark protocol

Bioinformatics benchmarks follow the shared `benchmarks._runtime` lifecycle. A record
must distinguish:

- first call/compilation from synchronized steady-state execution;
- host parsing/lowering from JAX kernel execution;
- static plan preparation from data-dependent calls;
- shape/capacity and valid occupancy;
- backend/device, precision, and warmup/repeat policy; and
- memory/resource evidence when the backend exposes it.

JAX is asynchronous. Synchronize the result tree before stopping each timer; otherwise a
benchmark measures dispatch. Compile evidence should identify the callable and static
shape/plan. Never silently reuse a compiled executable for a different claimed capacity.
Memory values are optional only when the runtime cannot provide them, and absence must
remain explicit rather than recorded as zero.

Use small deterministic generated cases for repository benchmarks. Do not check in
measured “golden” timings, fabricated accelerator results, or machine-specific pass/fail
thresholds. The producer writes deterministic JSON only when invoked; importing it must
not create an artifact. Repeated runs with the same semantic inputs should retain stable
identity fields even though timing values differ.

## JSON and provenance

Use the repository's canonical benchmark/qualification artifact shape; do not add a
bioinformatics schema-version field. Qualification output records `domain`,
`environment`, `input_fingerprint`, `method_fingerprint`,
`method_claim_taxonomy`, `unit_qualification` (`scope`, `cases`, and `passed`),
`external_campaigns`, `external_campaigns_requested`, and overall `passed`.
Benchmark output records `benchmark`, `environment`, `environment_fingerprint`,
`configuration`, input/method fingerprints and claim taxonomy, synchronized kernels
for lowering, compilation, first/steady execution, compiler memory/cost analysis and
logical bytes, and `passed`. Sort semantic collections deterministically and store
measured values only from the actual run; never embed source datasets or arbitrary
external objects.

## Domain-specific minimum evidence

- **Sequence/alignment:** alphabet, lowering loss, mode, full/band domain, gap semantics,
  traceback capacity, score/path consistency, and ambiguity handling.
- **Genomics:** exact reference digest, coordinate convention, complete genotype/state
  and candidate evidence, depth, prior, posterior/no-call thresholds, and normalization.
- **Phylogenetics/population:** topology and tip order, substitution/rate/ascertainment
  model, branches, relatedness/group split, finite search bounds, and SFS/coalescent
  assumptions.
- **Omics/spatial:** observed/missing/structural-zero semantics, biological unit, design
  rank/estimability, dispersion and multiple testing, donor/section/exchangeability,
  frame/unit, graph capacity, and permutation count.
- **Spectrometry:** acquisition and unit identity, peak/spectrum capacities, calibrants,
  tolerance, target/decoy competition, FDR level, and library/standard digest.
- **Structure/RNA/systems:** mmCIF/chemistry identity and lowering evidence; RNA grammar,
  energy unit, constraints, and pseudoknot status; network units, mass/charge balance,
  objective/bounds, solver/KKT evidence, and auxiliary-solve completeness.
- **Learned models:** exact bytes/parameters/structure/tokenizer/base/adapter hashes,
  reviewed license permissions, split and homology identity, overlap assessment,
  selection policy, seeds, and capacity.

## Interpreting a pass

A pass establishes only that the observed evidence meets the declared tolerance and
policy for that pinned artifact. It does not imply exhaustive discovery, model
adequacy, causal interpretation, clinical validity, dataset representativeness,
production scale, or hardware portability. Publish failures and unsupported evidence as
such; never omit invalid cases from the denominator after inspecting their outcome.
